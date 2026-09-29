#!/usr/bin/env python3
"""
Black-box view of reordering vs. dense width N and starting block density.

Only inputs we control (N, the matrix) and measured outputs (GFLOPS) are used;
no kernel model. Every ratio is taken *within one matrix*, one kernel, one N:
speedup = GFLOPS(reordered) / GFLOPS(original).

Reordering per matrix: the one with the highest 16x16 block density ("picked
by block density"). The pick uses no timing, and is the same for every kernel
and N, so each cell/curve compares the same pair of matrices (original vs. one
reordered version) and is not biased by selecting on the measured outcome.

Figures (original matrices, symmetric reordering):
  n_density_heatmap.pdf  x = N, y = block density of the *original* matrix
                         (quintile bins over matrices), colour = median over
                         the matrices of the bin of the per-matrix speedup;
                         one panel per kernel.
  n_saturation.pdf       GFLOPS vs N for three representative matrices (low /
                         mid / high starting block density), original vs.
                         reordered; one panel per (matrix, kernel). Absolute
                         GFLOPS are only compared within a panel.
  n_speedup_scatter.pdf  one point per matrix: speedup at N=32 (x) vs.
                         at N=1024 (y), colour = original block density;
                         one panel per kernel. Above the diagonal, the
                         reordering pays more with the wider dense operand
                         (above / below counted with a 5 % tolerance).
  corr_blocksize_ncols_beta.pdf
                         the paper's corr_blocksize_ncols plus a third panel:
                         median within-matrix elasticity beta by N (95 % CI)

Run from the repo root:
    .venv/Scripts/python.exe scripts/n_density_figures.py [--out plots/n_density]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

import plot_utils as pu
from paper_figures import (COL_W, PAGE_W, PAPER_RC, PAPER_KERNEL_NAMES,
                           display_name, load_pipeline)
from correlation_table import _ordered_kernels
from settings import KERNEL_NAMES

BD = 'block_density_16'
N_BINS = 5
N_VALUES = (32, 256, 1024)
SPEEDUP_LIM = (1 / 2, 2)          # colour range, log-symmetric around 1
# Diverging: orange (slower) - neutral grey - purple (faster).
CMAP = LinearSegmentedColormap.from_list(
    'loss_gain', ['#b35806', '#f1a340', '#ebebeb', '#998ec3', '#542788'])
C_ORIG, C_REORD = '#333333', '#6a2c91'


def picked(dataset='original', perm_type='SYMMETRIC'):
    """Per (matrix, kernel, N): original and block-density-picked reordering.

    Returns rows of the picked reordering with gflops, gflops_original,
    speedup, the original block density and the name of the pick."""
    df, da = load_pipeline(dataset, perm_type)
    da = da.copy()
    da['strategy'] = da['perm'].fillna('None').astype(str).map(display_name)
    cand = da[da['strategy'] != 'Original'].dropna(subset=[BD])
    pick = (cand.sort_values(BD, ascending=False)
                .drop_duplicates('matrix')[['matrix', 'strategy']]
                .rename(columns={'strategy': 'pick'}))
    d = df[df['strategy'].replace({'Metis': 'METIS'}) != 'Original'].copy()
    d['strategy'] = d['strategy'].replace({'Metis': 'METIS'})
    d = d.merge(pick, on='matrix')
    d = d[(d['strategy'] == d['pick']) & (d['speedup'] > 0)]
    d = d.dropna(subset=['speedup', f'{BD}_original'])
    # Several runs of one configuration: average the times, i.e. the GFLOPS.
    return (d.groupby(['matrix', 'kernel_id', 'n_cols', 'pick'], as_index=False)
             .agg(gflops=('gflops', 'mean'),
                  gflops_original=('gflops_original', 'mean'),
                  bd_original=(f'{BD}_original', 'first'),
                  bd_picked=(BD, 'first'))
             .assign(speedup=lambda x: x['gflops'] / x['gflops_original']))


def kernels_of(d):
    return _ordered_kernels(d, KERNEL_NAMES)


def density_bins(d):
    """Quintile edges of the original block density, over matrices (not rows)."""
    per_matrix = d.drop_duplicates('matrix')['bd_original']
    edges = np.quantile(per_matrix, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = edges[0] * 0.999, edges[-1] * 1.001
    return edges


def _fmt(x):
    return f'{x:.2f}' if x >= 0.1 else f'{x:.3f}'


def fig_heatmap(d, out):
    kernels = kernels_of(d)
    edges = density_bins(d)
    d = d.assign(bin=pd.cut(d['bd_original'], edges, labels=False))
    labels = [f'{_fmt(a)}–{_fmt(b)}' for a, b in zip(edges[:-1], edges[1:])]
    norm = LogNorm(*SPEEDUP_LIM)

    fig, axes = plt.subplots(1, len(kernels), figsize=(PAGE_W, 2.1),
                             sharey=True)
    rows = []
    for ax, k in zip(axes, kernels):
        s = d[d['kernel_id'] == k]
        med = s.groupby(['bin', 'n_cols'])['speedup'].median().unstack()
        cnt = s.groupby(['bin', 'n_cols'])['matrix'].nunique().unstack()
        med = med.reindex(index=range(N_BINS), columns=N_VALUES)
        cnt = cnt.reindex(index=range(N_BINS), columns=N_VALUES).fillna(0)
        ax.imshow(med.values, origin='lower', cmap=CMAP, norm=norm,
                  aspect='auto')
        for i in range(N_BINS):
            for j, n in enumerate(N_VALUES):
                v = med.values[i, j]
                if np.isnan(v):
                    continue
                dark = abs(np.log(v)) > 0.45 * np.log(SPEEDUP_LIM[1])
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6.5, color='white' if dark else '#222222')
                rows.append((PAPER_KERNEL_NAMES.get(k, k), labels[i], n, v,
                             int(cnt.values[i, j])))
        ax.set_xticks(range(len(N_VALUES)))
        ax.set_xticklabels(['32', '256', '1k'])
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels(labels)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        title = PAPER_KERNEL_NAMES.get(k, k)
        if len(title) > 11:           # cuSPARSE-BSR / -CSR: too wide for a panel
            title = title.replace('-', '-\n', 1)
        ax.set_title(title, fontsize=7.5, pad=3, linespacing=0.95)
        ax.grid(False)
    axes[0].set_ylabel('Original block density\n'
                       r'($16\times16$, quintiles)')
    fig.supxlabel(r'Dense operand width $N$ (columns of $B$)', fontsize=9,
                  y=0.0)
    fig.subplots_adjust(left=0.115, right=0.9, top=0.8, bottom=0.2,
                        wspace=0.1)
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.6])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
    ticks = [0.5, 0.71, 1, 1.41, 2]
    cb.set_ticks(ticks)
    cb.set_ticklabels(['≤0.5×', '0.71×', '1×', '1.41×', '≥2×'])
    cb.minorticks_off()
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label('Median speedup', fontsize=8)
    fig.savefig(out / 'n_density_heatmap.pdf')
    fig.savefig(out / 'n_density_heatmap.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(rows, columns=['kernel', 'bd_original_bin', 'n_cols',
                                'median_speedup', 'n_matrices']).to_csv(
        out / 'n_density_heatmap.csv', index=False)


def _gflops(y, _=None):
    return f'{y / 1000:g}k' if y >= 1000 else f'{y:g}'


def representatives(d, kernels):
    """One matrix at the median original block density of the low / mid / high
    tercile, among matrices measured for every (kernel, N) shown."""
    full = (d[d['kernel_id'].isin(kernels)].groupby('matrix')
             .apply(lambda s: len(s[['kernel_id', 'n_cols']].drop_duplicates()),
                    include_groups=False))
    ok = full[full == len(kernels) * len(N_VALUES)].index
    m = d[d['matrix'].isin(ok)].drop_duplicates('matrix').set_index('matrix')
    q = np.quantile(m['bd_original'], [1 / 6, 1 / 2, 5 / 6])
    return [(m['bd_original'] - t).abs().idxmin() for t in q]


def fig_saturation(d, out):
    kernels = kernels_of(d)
    mats = representatives(d, kernels)
    fig, axes = plt.subplots(len(mats), len(kernels),
                             figsize=(PAGE_W, 1.05 * len(mats) + 0.55),
                             sharex=True)
    for r, m in enumerate(mats):
        dm = d[d['matrix'] == m]
        info = dm.iloc[0]
        for c, k in enumerate(kernels):
            ax = axes[r][c]
            s = dm[dm['kernel_id'] == k].sort_values('n_cols')
            ax.plot(s['n_cols'], s['gflops_original'], '-o', color=C_ORIG,
                    lw=1.1, ms=3)
            ax.plot(s['n_cols'], s['gflops'], '-o', color=C_REORD, lw=1.1,
                    ms=3)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels(['32', '256', '1k'])
            ax.set_xlim(N_VALUES[0] / 1.6, N_VALUES[-1] * 1.6)
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_gflops))
            ax.tick_params(axis='y', labelsize=6.5)
            ax.grid(True, which='major', color='#b0b0b0', lw=0.6)
            lo = min(s['gflops_original'].min(), s['gflops'].min())
            hi = max(s['gflops_original'].max(), s['gflops'].max())
            ax.set_ylim(lo / 1.5, hi * 1.5)
            subs = (1, 2, 5) if hi / lo > 4 else (1, 1.5, 2, 3, 5, 7)
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=subs))
            if r == 0:
                ax.set_title(PAPER_KERNEL_NAMES.get(k, k), fontsize=7.5, pad=3)
        name = m.removesuffix('.mtx')
        axes[r][0].set_ylabel(
            f'{name}\n$\\rho_{{16}}$ {info.bd_original:.2f}→{info.bd_picked:.2f}'
            f'\nGFLOPS', fontsize=7.5)
    fig.supxlabel(r'Dense operand width $N$ (columns of $B$)', fontsize=9,
                  y=0.0)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.88, bottom=0.1,
                        wspace=0.34, hspace=0.18)
    handles = [Line2D([], [], color=C_ORIG, marker='o', ms=3, lw=1.1,
                      label='Original'),
               Line2D([], [], color=C_REORD, marker='o', ms=3, lw=1.1,
                      label='Reordered (highest block density)')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.0))
    fig.savefig(out / 'n_saturation.pdf')
    fig.savefig(out / 'n_saturation.png', dpi=300)
    plt.close(fig)
    d[d['matrix'].isin(mats)].to_csv(out / 'n_saturation.csv', index=False)


SCATTER_LIM = (1 / 4, 16)
DIAG_TOL = 1.05   # above / below the diagonal only if > 5 % apart
BD_CMAP = LinearSegmentedColormap.from_list('bd', ['#9ecae1', '#6baed6',
                                                   '#2171b5', '#08306b'])


def fig_speedup_n_scatter(d, out, n_lo=N_VALUES[0], n_hi=N_VALUES[-1]):
    """One point per matrix: speedup at n_lo (x) vs at n_hi (y), colour =
    original block density. Above the diagonal: reordering pays more at the
    wider dense operand."""
    kernels = kernels_of(d)
    w = (d[d['n_cols'].isin([n_lo, n_hi])]
         .pivot_table(index=['kernel_id', 'matrix'], columns='n_cols',
                      values='speedup')
         .dropna().reset_index())
    bd = d.drop_duplicates('matrix').set_index('matrix')['bd_original']
    w['bd'] = w['matrix'].map(bd)
    norm = LogNorm(bd.min(), bd.max())
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(PAGE_W, 3.75), sharex=True,
                             sharey=True)
    axes = axes.ravel()
    lo, hi = SCATTER_LIM
    rows = []
    for ax, k in zip(axes, kernels):
        s = w[w['kernel_id'] == k].sort_values('bd', ascending=False)
        x = s[n_lo].clip(lo * 1.05, hi / 1.05)
        y = s[n_hi].clip(lo * 1.05, hi / 1.05)
        ax.plot([lo, hi], [lo, hi], color='#777777', lw=0.7, zorder=1)
        ax.axhline(1, color='#bbbbbb', lw=0.5, zorder=1)
        ax.axvline(1, color='#bbbbbb', lw=0.5, zorder=1)
        ax.scatter(x, y, c=s['bd'], cmap=BD_CMAP, norm=norm, s=7, lw=0.25,
                   edgecolors='white', zorder=2, rasterized=True)
        ratio = s[n_hi] / s[n_lo]
        above = (ratio > DIAG_TOL).mean()
        below = (ratio < 1 / DIAG_TOL).mean()
        ax.text(0.04, 0.96, f'above: {above:.0%}, below: {below:.0%}\n'
                f'$n$={len(s)}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6.5,
                color='#333333', linespacing=1.1)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ticks = [0.5, 1, 2, 4, 8, 16]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        fmt = mpl.ticker.FuncFormatter(lambda v, _: f'{v:g}×')
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.tick_params(labelsize=6.5)
        ax.grid(True, color='#e4e4e4', lw=0.4)
        ax.set_title(PAPER_KERNEL_NAMES.get(k, k), fontsize=8, pad=2)
        rows.append((PAPER_KERNEL_NAMES.get(k, k), len(s), above, below,
                     float(np.exp(np.log(s[n_lo]).mean())),
                     float(np.exp(np.log(s[n_hi]).mean()))))
    # Last slot: colour bar for the original block density.
    cax = axes[-1]
    cax.set_visible(False)
    pos = cax.get_position()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.11,
                        wspace=0.12, hspace=0.25)
    pos = axes[-1].get_position()
    bar = fig.add_axes([pos.x0 + 0.02, pos.y0 + 0.02, 0.018, pos.height - 0.04])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=BD_CMAP), cax=bar)
    cb.set_label('Original block density\n' r'($16\times16$)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.4)
    fig.supxlabel(f'Speedup at $N={n_lo}$', fontsize=9, y=0.0)
    fig.supylabel(f'Speedup at $N={n_hi}$', fontsize=9, x=0.0)
    fig.savefig(out / 'n_speedup_scatter.pdf')
    fig.savefig(out / 'n_speedup_scatter.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(rows, columns=['kernel', 'n_matrices', 'share_above', 'share_below',
                                f'geomean_speedup_N{n_lo}',
                                f'geomean_speedup_N{n_hi}']).to_csv(
        out / 'n_speedup_scatter.csv', index=False)


# ---------------------------------------------------------------------------
# All reorderings of a matrix: within-matrix elasticity of speedup to block
# density, and whether the fastest reordering changes with N.
# ---------------------------------------------------------------------------

ELAST_MIN_REORD = 4           # reorderings needed to fit a matrix's slope
ELAST_MIN_RANGE = np.log(1.5)  # ... spanning >= 1.5x in block density
N_BOOT = 2000
# Okabe-Ito (colour-blind safe); lines are also labelled directly.
KERNEL_COLORS = {
    'ACCSPMM_SPMM': '#E69F00', 'ASPT_SPMM': '#999999',
    'CUSPARSE_SPMM_BSR_bs32': '#0072B2', 'CUSPARSE_SPMM_CSR': '#56B4E9',
    'DTC_SPMM': '#D55E00', 'FLASHSPARSE_SPMM': '#009E73',
    'SMAT_SPMM_bs32': '#CC79A7',
}
CENTRED_KERNELS = ('DTC_SPMM', 'CUSPARSE_SPMM_BSR_bs32', 'CUSPARSE_SPMM_CSR')


def all_reorderings(dataset='original', perm_type='SYMMETRIC'):
    """(matrix, kernel, N, reordering): log speedup and log block-density
    improvement, both relative to the same matrix unreordered. The original
    ordering is included as the point (0, 0)."""
    df, _ = load_pipeline(dataset, perm_type)
    d = df[df['strategy'] != 'Original'].dropna(
        subset=['speedup', f'density_improvement_16'])
    d = d[(d['speedup'] > 0) & (d['density_improvement_16'] > 0)]
    d = (d.groupby(['matrix', 'kernel_id', 'n_cols', 'strategy'], as_index=False)
          .agg(speedup=('speedup', 'mean'),
               dens=('density_improvement_16', 'first')))
    d['ly'], d['lx'] = np.log(d['speedup']), np.log(d['dens'])
    orig = d[['matrix', 'kernel_id', 'n_cols']].drop_duplicates().assign(
        strategy='Original', speedup=1.0, dens=1.0, ly=0.0, lx=0.0)
    return pd.concat([d, orig], ignore_index=True)


def matrix_slopes(r):
    """OLS slope of log speedup on log block-density improvement, per
    (matrix, kernel, N). Only matrices whose reorderings vary enough in block
    density, and that qualify at every N for that kernel."""
    rows = []
    for (m, k, n), g in r.groupby(['matrix', 'kernel_id', 'n_cols']):
        if len(g) - 1 < ELAST_MIN_REORD or np.ptp(g['lx']) < ELAST_MIN_RANGE:
            continue
        x = g['lx'] - g['lx'].mean()
        rows.append((m, k, n, float((x * g['ly']).sum() / (x * x).sum())))
    b = pd.DataFrame(rows, columns=['matrix', 'kernel_id', 'n_cols', 'beta'])
    full = b.groupby(['kernel_id', 'matrix'])['n_cols'].transform('nunique')
    return b[full == len(N_VALUES)]


def _boot_median(v, rng):
    idx = rng.integers(0, len(v), (N_BOOT, len(v)))
    return np.quantile(np.median(v[idx], axis=1), [0.025, 0.975])


def fig_elasticity(r, out):
    b = matrix_slopes(r)
    kernels = kernels_of(b)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    rows, ends = [], []
    for k in kernels:
        s = b[b['kernel_id'] == k]
        med, lo, hi = [], [], []
        for n in N_VALUES:
            v = s.loc[s['n_cols'] == n, 'beta'].to_numpy()
            med.append(np.median(v))
            ci = _boot_median(v, rng)
            lo.append(ci[0]); hi.append(ci[1])
            rows.append((PAPER_KERNEL_NAMES.get(k, k), n, med[-1], ci[0],
                         ci[1], s['matrix'].nunique()))
        c = KERNEL_COLORS.get(k, '#333333')
        ax.fill_between(N_VALUES, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(N_VALUES, med, '-o', color=c, lw=1.3, ms=3)
        ends.append([med[-1], PAPER_KERNEL_NAMES.get(k, k), c,
                     s['matrix'].nunique()])
    # Direct labels at the right end, nudged apart so they do not overlap.
    ends.sort()
    ymin, ymax = ax.get_ylim()
    gap = 0.065 * (ymax - ymin)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, name, c, nm in ends:
        ax.text(N_VALUES[-1] * 1.25, y, f'{name} ({nm})', color=c,
                fontsize=7, va='center', ha='left', fontweight='bold')
    ax.axhline(0, color='#777777', lw=0.6)
    ax.set_xscale('log', base=2)
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels(N_VALUES)
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(N_VALUES[0] / 1.4, N_VALUES[-1] * 8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True, axis='y', color='#e4e4e4', lw=0.4)
    ax.grid(False, axis='x')
    for n in N_VALUES:
        ax.axvline(n, color='#e4e4e4', lw=0.4, zorder=0)
    ax.set_xlabel(r'Dense operand width $N$')
    ax.set_ylabel('Within-matrix elasticity $\\beta$\n'
                  r'(speedup vs. block density, median)')
    fig.savefig(out / 'n_elasticity.pdf')
    fig.savefig(out / 'n_elasticity.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(rows, columns=['kernel', 'n_cols', 'median_beta', 'ci_lo',
                                'ci_hi', 'n_matrices']).to_csv(
        out / 'n_elasticity.csv', index=False)
    b.to_csv(out / 'n_elasticity_per_matrix.csv', index=False)
    return b


def fig_elasticity_centred(r, b, out, n_show=(N_VALUES[0], N_VALUES[-1])):
    """Every (matrix, reordering) point, both axes centred on the matrix's own
    mean: only within-matrix variation is left. Same matrices as the slopes."""
    kernels = [k for k in CENTRED_KERNELS if k in set(b['kernel_id'])]
    fig, axes = plt.subplots(len(n_show), len(kernels),
                             figsize=(PAGE_W, 1.95 * len(n_show) + 0.35),
                             sharex=True, sharey=True)
    lim = 1.6
    for c, k in enumerate(kernels):
        keep = set(b.loc[b['kernel_id'] == k, 'matrix'])
        for rr, n in enumerate(n_show):
            ax = axes[rr][c]
            g = r[(r['kernel_id'] == k) & (r['n_cols'] == n)
                  & r['matrix'].isin(keep)].copy()
            g['cx'] = g['lx'] - g.groupby('matrix')['lx'].transform('mean')
            g['cy'] = g['ly'] - g.groupby('matrix')['ly'].transform('mean')
            ax.hexbin(g['cx'], g['cy'], gridsize=34, bins='log', mincnt=1,
                      cmap='Greys', extent=(-lim, lim, -lim, lim),
                      linewidths=0.1, rasterized=True)
            beta = float((g['cx'] * g['cy']).sum() / (g['cx'] ** 2).sum())
            xs = np.array([-lim, lim])
            ax.plot(xs, beta * xs, color=KERNEL_COLORS[k], lw=1.4)
            ax.axhline(0, color='#bbbbbb', lw=0.5)
            ax.axvline(0, color='#bbbbbb', lw=0.5)
            ax.text(0.04, 0.96, f'pooled $\\beta$ = {beta:.2f}\n'
                    f'{g["matrix"].nunique()} matrices',
                    transform=ax.transAxes, ha='left', va='top', fontsize=7,
                    color='#222222')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ticks = np.log([1 / 4, 1 / 2, 1, 2, 4])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            f = mpl.ticker.FuncFormatter(lambda v, _: f'{np.exp(v):g}×')
            ax.xaxis.set_major_formatter(f)
            ax.yaxis.set_major_formatter(f)
            ax.tick_params(labelsize=7)
            ax.grid(False)
            if rr == 0:
                ax.set_title(PAPER_KERNEL_NAMES.get(k, k), fontsize=8.5, pad=3)
        for rr, n in enumerate(n_show):
            axes[rr][-1].yaxis.set_label_position('right')
            axes[rr][-1].set_ylabel(f'$N={n}$', rotation=270, labelpad=10,
                                    fontweight='bold')
    fig.supxlabel('Block-density improvement, relative to the matrix mean',
                  fontsize=9, y=0.0)
    fig.supylabel('Speedup, relative to the matrix mean', fontsize=9, x=0.0)
    fig.subplots_adjust(left=0.1, right=0.93, top=0.93, bottom=0.1,
                        wspace=0.08, hspace=0.1)
    fig.savefig(out / 'n_elasticity_centred.pdf')
    fig.savefig(out / 'n_elasticity_centred.png', dpi=300)
    plt.close(fig)


def best_reordering_changes(r, out, n_lo=N_VALUES[0], n_hi=N_VALUES[-1]):
    """Share of matrices whose fastest ordering (original included) at n_lo
    is not the fastest at n_hi; and the speedup lost at n_hi by keeping the
    n_lo choice. Only orderings measured at both N compete."""
    rows = []
    for k, s in r.groupby('kernel_id'):
        w = s[s['n_cols'].isin([n_lo, n_hi])].pivot_table(
            index=['matrix', 'strategy'], columns='n_cols', values='speedup'
        ).dropna().reset_index()
        changed, loss = [], []
        for m, g in w.groupby('matrix'):
            if len(g) < 3:
                continue
            a = g.loc[g[n_lo].idxmax()]
            best_hi = g[n_hi].max()
            changed.append(a['strategy'] != g.loc[g[n_hi].idxmax(), 'strategy'])
            loss.append(best_hi / a[n_hi])
        rows.append((PAPER_KERNEL_NAMES.get(k, k), len(changed),
                     float(np.mean(changed)), float(np.median(loss)),
                     float(np.mean(np.array(loss) > 1.05))))
    t = pd.DataFrame(rows, columns=['kernel', 'n_matrices',
                                    f'share_best_changes_{n_lo}_to_{n_hi}',
                                    f'median_loss_keeping_N{n_lo}_pick',
                                    'share_loss_over_5pct'])
    t.to_csv(out / 'n_best_reordering_changes.csv', index=False)
    print(t.to_string(index=False))


def fig_corr_with_beta(out):
    """The paper's corr_blocksize_ncols (r by block size at N=256; r by N at
    16x16) with a third panel: median within-matrix elasticity beta by N,
    bootstrap 95 % CI. Same kernels, same N colours."""
    from paper_figures import (_bars, _corr_values, _legend_top, BS_COLORS,
                               NCOLS_COLORS, NCOLS_HATCHES)
    from settings import BLOCK_SIZES
    from correlation_table import compute_imp_correlations
    df, _ = load_pipeline('original', 'SYMMETRIC')
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    bs_metrics = [f'density_improvement_{bs}' for bs in BLOCK_SIZES]
    corr = compute_imp_correlations(df, 256, bs_metrics, kernels,
                                    method='pearson', log_transform=True)
    top = [(_corr_values(corr, kernels, m), f'${bs}{{\\times}}{bs}$', c, '')
           for m, bs, c in zip(bs_metrics, BLOCK_SIZES, BS_COLORS)]
    ncols = sorted(df['n_cols'].unique())
    mid = []
    for nc, c, h in zip(ncols, NCOLS_COLORS, NCOLS_HATCHES):
        cd = compute_imp_correlations(df, nc, ['density_improvement_16'],
                                      kernels, method='pearson',
                                      log_transform=True)
        mid.append((_corr_values(cd, kernels, 'density_improvement_16'),
                    f'{int(nc)}', c, h))

    b = matrix_slopes(all_reorderings())
    rng = np.random.default_rng(0)
    bottom, errs, rows = [], [], []
    for nc, c, h in zip(ncols, NCOLS_COLORS, NCOLS_HATCHES):
        med, lo, hi = [], [], []
        for k in kernels:
            v = b.loc[(b['kernel_id'] == k) & (b['n_cols'] == nc), 'beta'].to_numpy()
            m = np.median(v)
            ci = _boot_median(v, rng)
            med.append(m); lo.append(m - ci[0]); hi.append(ci[1] - m)
            rows.append((PAPER_KERNEL_NAMES.get(k, k), int(nc), m, ci[0], ci[1],
                         len(v)))
        bottom.append((np.array(med), f'{int(nc)}', c, h))
        errs.append((np.array(lo), np.array(hi)))

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(COL_W, 4.65), sharex=True)
    _bars(a1, top, kernels)
    _bars(a2, mid, kernels)
    _bars(a3, bottom, kernels)
    n = len(bottom)
    width = 0.84 / n
    x = np.arange(len(kernels))
    for i, (lo, hi) in enumerate(errs):
        a3.errorbar(x + (i - (n - 1) / 2) * width, bottom[i][0],
                    yerr=[lo, hi], fmt='none', ecolor='#222222',
                    elinewidth=0.6, capsize=1.2, capthick=0.6, zorder=4)
    top_b = max(v.max() + e[1].max() for (v, *_), e in zip(bottom, errs))
    a3.set_ylim(0, np.ceil(top_b / 0.2) * 0.2 + 1e-9)
    a1.tick_params(axis='x', labelbottom=False)
    a2.tick_params(axis='x', labelbottom=False)
    _legend_top(a1, 'Block size:', ncol=7)
    _legend_top(a2, r'$n_{\mathrm{cols}}$:', ncol=4)
    _legend_top(a3, r'$n_{\mathrm{cols}}$:', ncol=4)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.95, bottom=0.07,
                        hspace=0.24)
    ymid = (a1.get_position().y1 + a2.get_position().y0) / 2
    fig.text(0.0, ymid, 'Pearson correlation of block density and speedup',
             rotation=90, ha='left', va='center', fontsize=9)
    p3 = a3.get_position()
    fig.text(0.0, (p3.y0 + p3.y1) / 2, 'Elasticity $\\beta$',
             rotation=90, ha='left', va='center', fontsize=9)
    fig.savefig(out / 'corr_blocksize_ncols_beta.pdf')
    fig.savefig(out / 'corr_blocksize_ncols_beta.png', dpi=300)
    plt.close(fig)
    pd.DataFrame(rows, columns=['kernel', 'n_cols', 'median_beta', 'ci_lo',
                                'ci_hi', 'n_matrices']).to_csv(
        out / 'corr_blocksize_ncols_beta.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--out', default='plots/n_density')
    ap.add_argument('--only-corr-beta', action='store_true',
                    help='only the three-panel r / r / beta figure')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pu.set_professional_style()
    mpl.rcParams.update(PAPER_RC)
    if args.only_corr_beta:
        fig_corr_with_beta(out)
        return
    d = picked()
    print(d.groupby('pick')['matrix'].nunique().sort_values(ascending=False))
    fig_heatmap(d, out)
    fig_saturation(d, out)
    fig_speedup_n_scatter(d, out)
    r = all_reorderings()
    b = fig_elasticity(r, out)
    fig_elasticity_centred(r, b, out)
    best_reordering_changes(r, out)
    fig_corr_with_beta(out)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
