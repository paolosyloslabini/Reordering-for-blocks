"""
Roofline-lite: does a blocked SpMM kernel pay per *tile* or per *nonzero*?

One question, one figure (``tile_roof_nc{N}.png``) and one table
(``tile_roof_nc{N}.csv``), computed only from kernel timings, nnz and block
densities.  No byte traffic is measured, so this is NOT a full roofline.

The question
------------
A blocked / tensor-core kernel does a fixed amount of work per tile it
touches, whether the tile holds one nonzero or 256.  Reordering changes only
how many tiles the same nonzeros occupy.  So:

    When a reordering packs the same nonzeros into fewer tiles, does the
    kernel's time drop in proportion (it is *tile-bound*: reordering buys
    speed), or does the time stay tied to the number of nonzeros (it is
    *nnz-bound*: reordering cannot help)?  And how close to the tensor-core
    peak is the per-tile work being executed?

The figure
----------
Top row (level):  executed FLOP/s / peak  vs  tile density, one panel per
    blocked kernel, all (matrix, reordering) pairs.  Executed FLOP/s =
    useful FLOP/s / tile density, i.e. counting the padded zeros the MMA
    units multiply.  A point at 1.0 is on the tensor-core roof.

Bottom row (slope, matrix-controlled):  time ratio vs tile-count ratio,
    each reordering relative to the *same matrix's* Original.  The log-log
    slope beta is fitted through the origin separately for reorderings that
    reduce the tile count (density improves) and ones that increase it.
    beta = 1: time ~ tiles (tile-bound).  beta = 0: time ~ nnz (nnz-bound).
    beta is the elasticity alpha of the speedup-vs-density scatter, but
    anchored per matrix (no size confound) and split by direction (no
    mixing of losses and gains).

The table also reports beta fitted per matrix (median) and per nnz tercile
as robustness checks.

Caveats
-------
* Executed FLOPs assume the kernel multiplies every padded zero of a
  bs x bs tile, with bs from KERNEL_TILE_DENSITY_BS.
* SMaT is modelled with 16x16 tiles (its MMA tile), not the 32x32 of its id.
* DTC-SpMM and FlashSparse use 16x8 tiles, approximated by 16x16 density,
  which OVER-estimates padding (16x8 density >= 16x16 density), so their
  executed/peak is an upper bound (up to 2x too high at very low density).
* DTC-SpMM is assumed to issue TF32 MMAs (peak 156 TFLOP/s).
* Peaks are datasheet dense tensor-core peaks of the A100-SXM4-80GB.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from settings import (KERNEL_NAMES, A100_PEAK_GFLOPS, KERNEL_PRECISION,
                      KERNEL_TILE_DENSITY_BS)
from correlation_table import _ordered_kernels

C_FEWER = '#4477AA'   # reordering reduced the tile count (density up)
C_MORE = '#CC6677'    # reordering increased the tile count (density down)
C_ALL = '#999999'


def add_executed_gflops(df):
    """Add ``tile_density``, ``exec_gflops`` and ``peak_gflops`` columns."""
    df = df.copy()
    df['tile_density'] = 1.0
    for kernel, bs in KERNEL_TILE_DENSITY_BS.items():
        col = f'block_density_{bs}'
        if col in df.columns:
            mask = df['kernel_id'] == kernel
            df.loc[mask, 'tile_density'] = df.loc[mask, col]
    df['exec_gflops'] = df['gflops'] / df['tile_density']
    df['peak_gflops'] = df['kernel_id'].map(
        lambda k: A100_PEAK_GFLOPS.get(KERNEL_PRECISION.get(k, 'FP32')))
    return df


def _relative_to_original(sub):
    """One row per (matrix, strategy != Original): tile ratio, time ratio."""
    keys = ['matrix', 'strategy']
    g = (sub.groupby(keys)
            .agg(dens=('tile_density', 'mean'), t=('time_operation_ms', 'mean'),
                 nnz=('nnz', 'first'))
            .reset_index())
    orig = g[g['strategy'] == 'Original'][['matrix', 'dens', 't']]
    orig = orig.rename(columns={'dens': 'dens0', 't': 't0'})
    r = g[g['strategy'] != 'Original'].merge(orig, on='matrix', how='inner')
    r['tile_ratio'] = r['dens0'] / r['dens']          # nnz fixed => tiles ~ 1/density
    r['time_ratio'] = r['t'] / r['t0']
    r = r.replace([np.inf, -np.inf], np.nan).dropna(subset=['tile_ratio', 'time_ratio'])
    r = r[(r['tile_ratio'] > 0) & (r['time_ratio'] > 0)]
    r['lx'] = np.log(r['tile_ratio'])
    r['ly'] = np.log(r['time_ratio'])
    return r


def _slope0(lx, ly):
    """Log-log slope through the origin (Original = (1, 1))."""
    return float((lx * ly).sum() / (lx ** 2).sum()) if len(lx) else np.nan


def _side_fits(r, thresh=0.1):
    """Pooled and per-matrix-median beta for each direction."""
    out = {}
    for side, mask in (('fewer', r['lx'] < -thresh), ('more', r['lx'] > thresh)):
        s = r[mask]
        out[f'beta_{side}'] = _slope0(s['lx'], s['ly'])
        out[f'n_{side}'] = int(len(s))
        per_m = [_slope0(g['lx'], g['ly']) for _, g in s.groupby('matrix') if len(g) >= 2]
        out[f'beta_{side}_matrix_median'] = float(np.median(per_m)) if per_m else np.nan
        # size-confound check: beta per nnz tercile
        if len(s) >= 30:
            terc = pd.qcut(s['nnz'], 3, labels=['small', 'mid', 'large'], duplicates='drop')
            for lab, g in s.groupby(terc, observed=True):
                out[f'beta_{side}_{lab}'] = _slope0(g['lx'], g['ly'])
    return out


def tile_roof(df, kernels, out_dir, n_cols, thresh=0.1):
    blocked = [k for k in kernels if k in KERNEL_TILE_DENSITY_BS]
    dfn = df[df['n_cols'] == n_cols].replace([np.inf, -np.inf], np.nan)
    dfn = dfn.dropna(subset=['gflops', 'tile_density', 'time_operation_ms'])
    dfn = dfn[dfn['tile_density'] > 0]

    n = len(blocked)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 5.6))
    axes = np.atleast_2d(axes)
    rows = []
    for j, k in enumerate(blocked):
        s = dfn[dfn['kernel_id'] == k]
        s = s.assign(exec_frac=s['exec_gflops'] / s['peak_gflops'])
        name = KERNEL_NAMES.get(k, k)
        bs = KERNEL_TILE_DENSITY_BS[k]
        prec = KERNEL_PRECISION.get(k, 'FP32')

        # ── top: level vs the tensor-core roof ─────────────────────────────
        ax = axes[0, j]
        ax.scatter(s['tile_density'], s['exec_frac'], s=5, alpha=0.35,
                   color=C_ALL, edgecolors='none', rasterized=True)
        bins = np.logspace(np.log10(s['tile_density'].min()),
                           np.log10(s['tile_density'].max()), 12)
        b = s.assign(_b=pd.cut(s['tile_density'], bins))
        med = b.groupby('_b', observed=True).agg(
            x=('tile_density', 'median'), y=('exec_frac', 'median'), n=('exec_frac', 'size'))
        med = med[med['n'] >= 10]
        ax.plot(med['x'], med['y'], color='black', linewidth=1.8, zorder=5,
                label='binned median')
        lvl = {}
        for strat, mk, col in (('Original', 'o', 'black'), ('Random', 'D', C_MORE)):
            ss = s[s['strategy'] == strat]
            if ss.empty:
                continue
            lvl[strat] = (ss['tile_density'].median(), ss['exec_frac'].median())
            ax.plot(*lvl[strat], marker=mk, color=col, markersize=7,
                    markeredgecolor='white', linestyle='none', zorder=6,
                    label=f'{strat} (median)')
        ax.axhline(1.0, color='black', linestyle=':', linewidth=1)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(3e-3, 2.5)
        ax.set_title(f'{name}\n{bs}$\\times${bs} tiles, {prec} peak', fontsize=9)
        ax.set_xlabel('tile density', fontsize=8)
        if j == 0:
            ax.set_ylabel('executed FLOP/s / peak', fontsize=8)
            ax.text(s['tile_density'].min(), 1.05, ' tensor-core roof',
                    fontsize=7, va='bottom')
        ax.tick_params(labelsize=7)
        ax.grid(True, which='major', alpha=0.2)

        # ── bottom: per-matrix time vs tiles ───────────────────────────────
        ax = axes[1, j]
        r = _relative_to_original(s)
        fits = _side_fits(r, thresh)
        fewer, more = r[r['lx'] < -thresh], r[r['lx'] > thresh]
        for pts, col, lab in ((fewer, C_FEWER, 'fewer tiles'), (more, C_MORE, 'more tiles')):
            if not pts.empty:
                ax.scatter(pts['tile_ratio'], pts['time_ratio'], s=5, alpha=0.4,
                           color=col, edgecolors='none', rasterized=True, label=lab)
        xx = np.array([0.02, 50.0])
        ax.plot(xx, xx, color='black', linewidth=1, linestyle='--',
                label=r'time $\propto$ tiles ($\beta$=1)')
        ax.plot(xx, [1, 1], color='black', linewidth=1, linestyle=':',
                label=r'time $\propto$ nnz ($\beta$=0)')
        for side, col, xr in (('fewer', C_FEWER, np.array([0.02, 1.0])),
                              ('more', C_MORE, np.array([1.0, 50.0]))):
            bta = fits[f'beta_{side}']
            if np.isfinite(bta):
                ax.plot(xr, xr ** bta, color=col, linewidth=2.2, zorder=5)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(0.03, 30); ax.set_ylim(0.1, 10)
        ax.set_xlabel('tiles / tiles(Original)', fontsize=8)
        if j == 0:
            ax.set_ylabel('time / time(Original)', fontsize=8)
        txt = [f"$\\beta_{{{side}}}$ = {fits[f'beta_{side}']:.2f}"
               for side in ('fewer', 'more') if np.isfinite(fits[f'beta_{side}'])]
        ax.text(0.03, 0.97, '\n'.join(txt),
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))
        ax.tick_params(labelsize=7)
        ax.grid(True, which='major', alpha=0.2)

        row = {'kernel': name, 'kernel_id': k, 'tile_bs': bs, 'precision': prec,
               'n_cols': int(n_cols), 'n_matrices': s['matrix'].nunique(),
               'exec_frac_original': lvl.get('Original', (np.nan, np.nan))[1],
               'dens_original': lvl.get('Original', (np.nan, np.nan))[0],
               'exec_frac_random': lvl.get('Random', (np.nan, np.nan))[1],
               'dens_random': lvl.get('Random', (np.nan, np.nan))[0]}
        row.update(fits)
        rows.append(row)

    # unblocked kernels: useful/peak as context only (no panel)
    for k in kernels:
        if k in KERNEL_TILE_DENSITY_BS:
            continue
        s = dfn[(dfn['kernel_id'] == k) & (dfn['strategy'] == 'Original')]
        if s.empty:
            continue
        rows.append({'kernel': KERNEL_NAMES.get(k, k), 'kernel_id': k,
                     'tile_bs': np.nan, 'precision': KERNEL_PRECISION.get(k, 'FP32'),
                     'n_cols': int(n_cols), 'n_matrices': s['matrix'].nunique(),
                     'exec_frac_original': (s['gflops'] / s['peak_gflops']).median()})

    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, -1].legend(h, l, fontsize=6.5, loc='lower left', framealpha=0.9)
    h, l = axes[1, 0].get_legend_handles_labels()
    axes[1, -1].legend(h, l, fontsize=6.5, loc='lower right', framealpha=0.9,
                       markerscale=2.5)
    fig.suptitle(f'$n_{{cols}}$ = {int(n_cols)}.  Top: where each kernel runs relative '
                 'to the tensor-core roof (executed = useful / tile density).  '
                 'Bottom: per matrix, does time follow the tile count?',
                 fontsize=8.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    png = out_dir / f'tile_roof_nc{int(n_cols)}.png'
    plt.savefig(png, dpi=300)
    plt.close()
    tab = pd.DataFrame(rows)
    csv = out_dir / f'tile_roof_nc{int(n_cols)}.csv'
    tab.to_csv(csv, index=False, float_format='%.3g')
    print(f"  Saved: {png}\n  Saved: {csv}")
    cols = ['kernel', 'exec_frac_original', 'exec_frac_random', 'beta_fewer',
            'beta_fewer_matrix_median', 'beta_more', 'beta_more_matrix_median']
    print(tab[[c for c in cols if c in tab.columns]].round(3).to_string(index=False))
    return tab


def generate_roofline_lite_plots(df, out_dir):
    out_dir = Path(out_dir) / 'roofline_lite'
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_executed_gflops(df)
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    for n in sorted(df['n_cols'].unique()):
        tile_roof(df, kernels, out_dir, n)
