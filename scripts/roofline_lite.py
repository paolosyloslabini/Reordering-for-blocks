"""
Roofline-lite analysis: FLOP-side roof placement from existing timing data.

No byte traffic is measured, so this is *not* a full roofline.  It answers
three questions using only kernel timings, nnz, and block densities:

1. How does useful throughput scale with n_cols?  Linear growth means the
   kernel is bound by streaming the sparse operand; a plateau means it has
   hit a roof (``gflops_vs_ncols*.png``).
2. For blocked kernels, how close is the *executed* throughput (useful
   FLOPs / block density at the kernel's tile size, i.e. including the
   padded zeros the MMA units actually multiply) to the tensor-core peak?
   Near peak => compute-saturated, reordering can only remove padding
   (``executed_vs_useful_nc*.png``).
3. Does reordering raise executed throughput (memory behaviour improved,
   point moves *toward* the roof) or leave it flat while useful throughput
   rises (kernel already at a roof, reordering only converts waste into
   work)?  (``executed_orig_vs_reordered_nc*.png``).

A summary table (CSV + LaTeX) with per-kernel medians is written alongside.

All constants (peaks, per-kernel precision, tile block size, reference
reordering) live in ``settings.py``.

Caveats
-------
* Executed FLOPs assume the kernel multiplies every padded zero of its tile
  at the block density of the square block size in KERNEL_TILE_DENSITY_BS.
* SMaT is modelled with 16x16 tiles (its actual MMA tile), not the 32x32
  suggested by its kernel id.
* DTC-SpMM and FlashSparse use 16x8 tiles, approximated by 16x16 density.
* Only FLOP-side roofs are drawn; no byte traffic is measured, so this is
  not a full roofline.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_utils as pu
from settings import (KERNEL_NAMES, PALETTE, A100_PEAK_GFLOPS,
                      KERNEL_PRECISION, KERNEL_TILE_DENSITY_BS,
                      ROOFLINE_REF_PERM, get_perm_display)
from correlation_table import _ordered_kernels


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def add_executed_gflops(df):
    """Add ``exec_gflops``, ``tile_density`` and ``peak_gflops`` columns.

    exec_gflops = gflops / block_density_{bs} for blocked kernels (bs from
    KERNEL_TILE_DENSITY_BS); equal to gflops for unblocked kernels.
    """
    df = df.copy()
    df['tile_density'] = 1.0
    for kernel, bs in KERNEL_TILE_DENSITY_BS.items():
        col = f'block_density_{bs}'
        if col not in df.columns:
            continue
        mask = df['kernel_id'] == kernel
        df.loc[mask, 'tile_density'] = df.loc[mask, col]
    df['exec_gflops'] = df['gflops'] / df['tile_density']
    df['peak_gflops'] = df['kernel_id'].map(
        lambda k: A100_PEAK_GFLOPS.get(KERNEL_PRECISION.get(k, 'FP32')))
    return df


def build_paired_frame(df, ref_perm=ROOFLINE_REF_PERM):
    """One row per (matrix, kernel_id, n_cols) with Original and ref columns.

    Columns: gflops_orig, gflops_ref, exec_orig, exec_ref, dens_orig,
    dens_ref, peak_gflops, speedup (= gflops_ref / gflops_orig).
    """
    keys = ['matrix', 'kernel_id', 'n_cols']
    agg = {'gflops': 'mean', 'exec_gflops': 'mean', 'tile_density': 'mean',
           'peak_gflops': 'first'}
    orig = (df[df['strategy'] == 'Original']
            .groupby(keys).agg(agg).reset_index()
            .rename(columns={'gflops': 'gflops_orig', 'exec_gflops': 'exec_orig',
                             'tile_density': 'dens_orig'}))
    ref = (df[df['perm'] == ref_perm]
           .groupby(keys).agg({'gflops': 'mean', 'exec_gflops': 'mean',
                               'tile_density': 'mean'}).reset_index()
           .rename(columns={'gflops': 'gflops_ref', 'exec_gflops': 'exec_ref',
                            'tile_density': 'dens_ref'}))
    paired = orig.merge(ref, on=keys, how='inner')
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['gflops_orig', 'gflops_ref', 'exec_orig', 'exec_ref'])
    paired['speedup'] = paired['gflops_ref'] / paired['gflops_orig']
    paired['exec_ratio'] = paired['exec_ref'] / paired['exec_orig']
    paired['dens_imp'] = paired['dens_ref'] / paired['dens_orig']
    return paired


def _kernel_palette(kernels):
    return {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(kernels)}


def _peak_lines(ax, precisions, xmin, xmax, fontsize=8):
    """Draw dotted horizontal peak lines with right-aligned labels."""
    styles = {'FP32': ':', 'TF32': '--', 'FP16': '-.'}
    for prec in sorted(set(precisions), key=lambda p: A100_PEAK_GFLOPS[p]):
        y = A100_PEAK_GFLOPS[prec]
        ax.axhline(y, color='grey', linestyle=styles.get(prec, ':'),
                   linewidth=0.9, alpha=0.8, zorder=0)
        ax.text(xmax, y, f' {prec} peak', va='center', ha='left',
                fontsize=fontsize, color='dimgrey')


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: throughput vs n_cols
# ─────────────────────────────────────────────────────────────────────────────

def plot_gflops_vs_ncols(paired, kernels, out_dir, ref_label):
    """Median useful GFLOPS vs n_cols, Original (solid) vs ref (dashed)."""
    pal = _kernel_palette(kernels)
    n_cols_values = sorted(paired['n_cols'].unique())

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k in kernels:
        sub = paired[paired['kernel_id'] == k]
        if sub.empty:
            continue
        med = sub.groupby('n_cols')[['gflops_orig', 'gflops_ref']].median()
        ax.plot(med.index, med['gflops_orig'], '-o', color=pal[k],
                label=KERNEL_NAMES.get(k, k), markersize=5, linewidth=1.6)
        ax.plot(med.index, med['gflops_ref'], '--s', color=pal[k],
                markersize=4, linewidth=1.2, alpha=0.9)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(n_cols_values)
    ax.set_xticklabels([str(int(v)) for v in n_cols_values])
    ax.set_xlabel('$n_{cols}$')
    ax.set_ylabel('Median useful GFLOP/s')
    _peak_lines(ax, [KERNEL_PRECISION.get(k, 'FP32') for k in kernels],
                n_cols_values[0], n_cols_values[-1])
    # Style legend for orig/ref
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color='k', linestyle='-', marker='o', markersize=4),
                Line2D([0], [0], color='k', linestyle='--', marker='s', markersize=4)]
    labels += ['Original', ref_label]
    ax.legend(handles, labels, fontsize=8, ncol=2, loc='upper left', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.25)
    plt.tight_layout()
    path = out_dir / 'gflops_vs_ncols.png'
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_gflops_vs_ncols_panels(paired, kernels, out_dir, ref_label):
    """Per-kernel boxplots (5/95 whiskers) of useful GFLOPS at each n_cols."""
    n_cols_values = sorted(paired['n_cols'].unique())
    n_k = len(kernels)
    ncols_fig = 4
    nrows_fig = int(np.ceil(n_k / ncols_fig))
    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(3.2 * ncols_fig, 2.8 * nrows_fig),
                             sharey=False)
    axes = np.atleast_1d(axes).ravel()
    c_orig, c_ref = '#88CCEE', '#CC6677'
    for ax, k in zip(axes, kernels):
        sub = paired[paired['kernel_id'] == k]
        pos = np.arange(len(n_cols_values))
        data_o = [sub.loc[sub['n_cols'] == n, 'gflops_orig'].values for n in n_cols_values]
        data_r = [sub.loc[sub['n_cols'] == n, 'gflops_ref'].values for n in n_cols_values]
        bp_o = ax.boxplot(data_o, positions=pos - 0.18, widths=0.3, whis=(5, 95),
                          showfliers=False, patch_artist=True)
        bp_r = ax.boxplot(data_r, positions=pos + 0.18, widths=0.3, whis=(5, 95),
                          showfliers=False, patch_artist=True)
        for bp, c in ((bp_o, c_orig), (bp_r, c_ref)):
            for box in bp['boxes']:
                box.set(facecolor=c, alpha=0.8, linewidth=0.7)
            for med in bp['medians']:
                med.set(color='black', linewidth=1.0)
        ax.set_yscale('log')
        ax.set_xticks(pos)
        ax.set_xticklabels([str(int(n)) for n in n_cols_values])
        ax.set_title(KERNEL_NAMES.get(k, k), fontsize=10)
        ax.set_xlabel('$n_{cols}$', fontsize=9)
        peak = A100_PEAK_GFLOPS[KERNEL_PRECISION.get(k, 'FP32')]
        ax.axhline(peak, color='grey', linestyle=':', linewidth=0.9)
        ax.text(pos[-1] + 0.45, peak, f'{KERNEL_PRECISION.get(k, "FP32")} peak',
                fontsize=7, color='dimgrey', va='bottom', ha='right')
        ax.grid(True, which='major', axis='y', alpha=0.25)
    for ax in axes[len(kernels):]:
        ax.axis('off')
    axes[0].set_ylabel('Useful GFLOP/s')
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c_orig), Patch(facecolor=c_ref)],
               ['Original', ref_label], loc='lower right', fontsize=9,
               bbox_to_anchor=(0.98, 0.04))
    plt.tight_layout()
    path = out_dir / 'gflops_vs_ncols_panels.png'
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: executed vs useful throughput against the peak
# ─────────────────────────────────────────────────────────────────────────────

def plot_executed_vs_useful(paired, kernels, out_dir, n_cols, ref_label,
                            normalise=False):
    """Grouped bars per kernel: useful/executed GFLOPS, Original vs ref.

    With ``normalise=True`` values are divided by the kernel's peak, so the
    y-axis reads as "fraction of peak".
    """
    sub = paired[paired['n_cols'] == n_cols]
    series = [('gflops_orig', 'Useful, Original', '#b2dfdb', ''),
              ('gflops_ref', f'Useful, {ref_label}', '#00897b', ''),
              ('exec_orig', 'Executed, Original', '#f4c7c3', '///'),
              ('exec_ref', f'Executed, {ref_label}', '#c0392b', '///')]
    n_s = len(series)
    x = np.arange(len(kernels))
    bw = 0.8 / n_s

    fig, ax = plt.subplots(figsize=(8, 4))
    for si, (col, label, color, hatch) in enumerate(series):
        vals = []
        for k in kernels:
            s = sub.loc[sub['kernel_id'] == k]
            if s.empty:
                vals.append(np.nan)
                continue
            v = s[col].median()
            if normalise:
                v = v / s['peak_gflops'].iloc[0]
            # Unblocked kernels: executed == useful; grey them out
            vals.append(v)
        bars = ax.bar(x + (si - (n_s - 1) / 2) * bw, vals, bw * 0.92,
                      label=label, color=color, hatch=hatch,
                      edgecolor='black', linewidth=0.4)
        if col.startswith('exec'):
            for k, b in zip(kernels, bars):
                if k not in KERNEL_TILE_DENSITY_BS:
                    b.set_alpha(0.25)
    if not normalise:
        for i, k in enumerate(kernels):
            peak = A100_PEAK_GFLOPS[KERNEL_PRECISION.get(k, 'FP32')]
            ax.hlines(peak, i - 0.42, i + 0.42, color='black', linewidth=1.2)
        ax.plot([], [], color='black', linewidth=1.2, label='Peak (kernel precision)')
        ax.set_ylabel('Median GFLOP/s')
    else:
        ax.axhline(1.0, color='black', linewidth=1.0)
        ax.set_ylabel('Median fraction of peak')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_NAMES.get(k, k) for k in kernels],
                       rotation=25, ha='right')
    ax.legend(fontsize=8, ncol=2, loc='upper left', framealpha=0.9)
    ax.grid(True, axis='y', which='both', alpha=0.25)
    ax.set_title(f'$n_{{cols}} = {int(n_cols)}$  '
                 '(faded executed bars: unblocked kernel, executed = useful)',
                 fontsize=9)
    plt.tight_layout()
    suffix = '_frac' if normalise else ''
    path = out_dir / f'executed_vs_useful{suffix}_nc{int(n_cols)}.png'
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: executed throughput, Original vs reordered (per matrix)
# ─────────────────────────────────────────────────────────────────────────────

def plot_executed_orig_vs_ref(paired, kernels, out_dir, n_cols, ref_label):
    """Scatter of executed GFLOPS after vs before reordering, one panel per
    blocked kernel.  Points on the diagonal: reordering only removed padding
    (kernel at a roof).  Above: memory behaviour improved too."""
    blocked = [k for k in kernels if k in KERNEL_TILE_DENSITY_BS]
    sub = paired[(paired['n_cols'] == n_cols) & paired['kernel_id'].isin(blocked)]
    n = len(blocked)
    ncols_fig = min(3, n)
    nrows_fig = int(np.ceil(n / ncols_fig))
    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(3.4 * ncols_fig, 3.2 * nrows_fig))
    axes = np.atleast_1d(axes).ravel()
    pal = _kernel_palette(kernels)
    for ax, k in zip(axes, blocked):
        s = sub[sub['kernel_id'] == k]
        peak = A100_PEAK_GFLOPS[KERNEL_PRECISION.get(k, 'FP32')]
        sc = ax.scatter(s['exec_orig'], s['exec_ref'], s=10, alpha=0.55,
                        c=np.log10(s['dens_imp']), cmap='viridis',
                        vmin=-0.5, vmax=1.0, edgecolors='none')
        lo = min(s['exec_orig'].min(), s['exec_ref'].min()) * 0.7
        hi = max(s['exec_orig'].max(), s['exec_ref'].max(), peak) * 1.4
        ax.plot([lo, hi], [lo, hi], color='grey', linewidth=0.8)
        ax.axhline(peak, color='grey', linestyle=':', linewidth=0.9)
        ax.axvline(peak, color='grey', linestyle=':', linewidth=0.9)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        med_ratio = s['exec_ratio'].median()
        frac_up = (s['exec_ratio'] > 1.1).mean()
        ax.set_title(KERNEL_NAMES.get(k, k), fontsize=10)
        ax.text(0.04, 0.96,
                f'median exec ratio {med_ratio:.2f}\n'
                f'{100*frac_up:.0f}% of matrices >1.1x\n'
                f'n={len(s)}',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_xlabel('Executed GFLOP/s, Original', fontsize=8)
        ax.set_ylabel(f'Executed GFLOP/s, {ref_label}', fontsize=8)
        ax.grid(True, which='major', alpha=0.25)
    for ax in axes[n:]:
        ax.axis('off')
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.7, pad=0.02)
    cbar.set_label('log$_{10}$ tile-density improvement', fontsize=8)
    path = out_dir / f'executed_orig_vs_reordered_nc{int(n_cols)}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_summary_table(paired, kernels, out_dir, ref_label):
    rows = []
    for k in kernels:
        for n in sorted(paired['n_cols'].unique()):
            s = paired[(paired['kernel_id'] == k) & (paired['n_cols'] == n)]
            if s.empty:
                continue
            peak = s['peak_gflops'].iloc[0]
            rows.append({
                'kernel': KERNEL_NAMES.get(k, k),
                'kernel_id': k,
                'precision': KERNEL_PRECISION.get(k, 'FP32'),
                'n_cols': int(n),
                'n_matrices': len(s),
                'peak_gflops': peak,
                'useful_orig': s['gflops_orig'].median(),
                'useful_ref': s['gflops_ref'].median(),
                'exec_orig': s['exec_orig'].median() if k in KERNEL_TILE_DENSITY_BS else np.nan,
                'exec_ref': s['exec_ref'].median() if k in KERNEL_TILE_DENSITY_BS else np.nan,
                'useful_orig_frac_peak': s['gflops_orig'].median() / peak,
                'exec_orig_frac_peak': (s['exec_orig'].median() / peak
                                        if k in KERNEL_TILE_DENSITY_BS else np.nan),
                'exec_ref_frac_peak': (s['exec_ref'].median() / peak
                                       if k in KERNEL_TILE_DENSITY_BS else np.nan),
                'median_speedup': s['speedup'].median(),
                'median_exec_ratio': (s['exec_ratio'].median()
                                      if k in KERNEL_TILE_DENSITY_BS else np.nan),
                'median_density_imp': (s['dens_imp'].median()
                                       if k in KERNEL_TILE_DENSITY_BS else np.nan),
            })
    tab = pd.DataFrame(rows)
    csv_path = out_dir / 'roofline_lite_summary.csv'
    tab.to_csv(csv_path, index=False, float_format='%.4g')
    print(f"  Saved: {csv_path}")

    # Compact LaTeX (n_cols=256 rows only, one line per kernel)
    def fmt(v, p='{:.0f}'):
        return '--' if pd.isna(v) else p.format(v)
    lines = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
             r'\textbf{Kernel} & \textbf{Peak} & '
             r'\multicolumn{2}{c}{\textbf{Useful GFLOP/s}} & '
             r'\multicolumn{2}{c}{\textbf{Executed GFLOP/s}} & '
             r'\textbf{Exec./peak} & \textbf{Speedup} \\',
             r' & & Orig. & ' + ref_label + r' & Orig. & ' + ref_label + r' & Orig. / ' + ref_label + r' & \\',
             r'\midrule']
    for _, r in tab[tab['n_cols'] == 256].iterrows():
        lines.append(
            f"{r['kernel']} & {fmt(r['peak_gflops']/1000, '{:.1f}T')} & "
            f"{fmt(r['useful_orig'])} & {fmt(r['useful_ref'])} & "
            f"{fmt(r['exec_orig'])} & {fmt(r['exec_ref'])} & "
            f"{fmt(r['exec_orig_frac_peak'], '{:.2f}')} / {fmt(r['exec_ref_frac_peak'], '{:.2f}')} & "
            f"{fmt(r['median_speedup'], '{:.2f}')} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    tex_path = out_dir / 'roofline_lite_summary_nc256.tex'
    tex_path.write_text('\n'.join(lines) + '\n')
    print(f"  Saved: {tex_path}")
    return tab


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_roofline_lite_plots(df, out_dir, ref_perm=ROOFLINE_REF_PERM):
    out_dir = Path(out_dir) / 'roofline_lite'
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_label = get_perm_display(ref_perm)

    df = add_executed_gflops(df)
    paired = build_paired_frame(df, ref_perm)
    if paired.empty:
        print(f"  No (Original, {ref_perm}) pairs found — skipping roofline-lite.")
        return
    kernels = [k for k in _ordered_kernels(df, KERNEL_NAMES)
               if k in paired['kernel_id'].unique()]
    print(f"  Reference reordering: {ref_label}; kernels: "
          f"{[KERNEL_NAMES.get(k, k) for k in kernels]}; "
          f"{paired['matrix'].nunique()} matrices")

    plot_gflops_vs_ncols(paired, kernels, out_dir, ref_label)
    plot_gflops_vs_ncols_panels(paired, kernels, out_dir, ref_label)
    for n in sorted(paired['n_cols'].unique()):
        plot_executed_vs_useful(paired, kernels, out_dir, n, ref_label)
        plot_executed_vs_useful(paired, kernels, out_dir, n, ref_label, normalise=True)
        plot_executed_orig_vs_ref(paired, kernels, out_dir, n, ref_label)
    write_summary_table(paired, kernels, out_dir, ref_label)
    # Reference-free views over all reorderings
    for n in sorted(df['n_cols'].unique()):
        plot_exec_frac_vs_density(df, kernels, out_dir, n)
        write_per_perm_table(df, kernels, out_dir, n)


# ─────────────────────────────────────────────────────────────────────────────
# Reference-free view: executed fraction of peak vs tile density, all
# (matrix, reordering) pairs.  A kernel-intrinsic floor appears as a
# horizontal asymptote that every reordering converges to.
# ─────────────────────────────────────────────────────────────────────────────

def plot_exec_frac_vs_density(df, kernels, out_dir, n_cols):
    blocked = [k for k in kernels if k in KERNEL_TILE_DENSITY_BS]
    sub = df[(df['n_cols'] == n_cols) & df['kernel_id'].isin(blocked)].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['exec_gflops', 'tile_density'])
    sub['exec_frac'] = sub['exec_gflops'] / sub['peak_gflops']
    n = len(blocked)
    ncols_fig = min(3, n)
    nrows_fig = int(np.ceil(n / ncols_fig))
    fig, axes = plt.subplots(nrows_fig, ncols_fig,
                             figsize=(3.6 * ncols_fig, 3.1 * nrows_fig))
    axes = np.atleast_1d(axes).ravel()
    strategies = pu.get_strategy_order(sub)
    pal = pu.get_strategy_palette(strategies)
    for ax, k in zip(axes, blocked):
        s = sub[sub['kernel_id'] == k]
        for strat in strategies:
            ss = s[s['strategy'] == strat]
            if ss.empty:
                continue
            ax.scatter(ss['tile_density'], ss['exec_frac'], s=7, alpha=0.5,
                       color=pal.get(strat, 'grey'), edgecolors='none',
                       label=strat)
        # Binned median trend across all reorderings
        bins = np.logspace(np.log10(s['tile_density'].min()),
                           np.log10(s['tile_density'].max()), 12)
        s = s.assign(_bin=pd.cut(s['tile_density'], bins))
        med = s.groupby('_bin', observed=True).agg(
            x=('tile_density', 'median'), y=('exec_frac', 'median'), n=('exec_frac', 'size'))
        med = med[med['n'] >= 10]
        ax.plot(med['x'], med['y'], color='black', linewidth=1.6, zorder=5)
        ax.axhline(1.0, color='grey', linestyle=':', linewidth=0.9)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_ylim(top=2.0)
        bs = KERNEL_TILE_DENSITY_BS[k]
        ax.set_title(f"{KERNEL_NAMES.get(k, k)}  ({bs}$\\times${bs} tiles)", fontsize=10)
        ax.set_xlabel('Tile density', fontsize=8)
        ax.set_ylabel('Executed / peak', fontsize=8)
        ax.grid(True, which='major', alpha=0.25)
    for ax in axes[n:]:
        ax.axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=7, ncol=2,
               markerscale=2.5, bbox_to_anchor=(0.98, 0.04))
    fig.suptitle(f'$n_{{cols}} = {int(n_cols)}$; all (matrix, reordering) pairs; '
                 'black: binned median', fontsize=9)
    plt.tight_layout()
    path = out_dir / f'exec_frac_vs_density_nc{int(n_cols)}.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def write_per_perm_table(df, kernels, out_dir, n_cols):
    """Median executed/peak and useful/peak per (kernel, reordering)."""
    sub = df[df['n_cols'] == n_cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=['exec_gflops'])
    sub['exec_frac'] = sub['exec_gflops'] / sub['peak_gflops']
    sub['useful_frac'] = sub['gflops'] / sub['peak_gflops']
    sub['kernel'] = sub['kernel_id'].map(lambda k: KERNEL_NAMES.get(k, k))
    strategies = list(dict.fromkeys(pu.get_strategy_order(sub)))
    tab = (sub.groupby(['kernel', 'strategy'])
              .agg(n=('exec_frac', 'size'),
                   exec_frac=('exec_frac', 'median'),
                   useful_frac=('useful_frac', 'median'),
                   tile_density=('tile_density', 'median'))
              .reset_index())
    rank = {s_: i for i, s_ in enumerate(strategies)}
    tab['_r'] = tab['strategy'].map(lambda x: rank.get(x, len(rank)))
    tab = tab.sort_values(['kernel', '_r']).drop(columns='_r')
    csv_path = out_dir / f'exec_frac_by_perm_nc{int(n_cols)}.csv'
    tab.to_csv(csv_path, index=False, float_format='%.3g')
    wide = tab.pivot(index='strategy', columns='kernel', values='exec_frac')
    wide = wide.reindex([s_ for s_ in strategies if s_ in wide.index])
    wide.to_csv(out_dir / f'exec_frac_by_perm_nc{int(n_cols)}_wide.csv', float_format='%.3g')
    print(f"  Saved: {csv_path}")
    return wide
