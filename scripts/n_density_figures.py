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
    ks = _ordered_kernels(d, KERNEL_NAMES)
    return [k for k in ks if k != 'CUSPARSE_SPMM_BSR_bs32']


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

    fig, axes = plt.subplots(1, len(kernels), figsize=(PAGE_W, 1.95),
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
        ax.set_xticklabels(N_VALUES)
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels(labels)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(PAPER_KERNEL_NAMES.get(k, k), fontsize=8.5, pad=3)
        ax.grid(False)
    axes[0].set_ylabel('Original block density\n'
                       r'($16\times16$, quintiles)')
    fig.supxlabel(r'Dense operand width $N$ (columns of $B$)', fontsize=9,
                  y=0.0)
    fig.subplots_adjust(left=0.115, right=0.9, top=0.86, bottom=0.2,
                        wspace=0.08)
    cax = fig.add_axes([0.915, 0.2, 0.012, 0.66])
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
            ax.set_xticklabels(N_VALUES)
            ax.yaxis.set_major_locator(mpl.ticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_gflops))
            ax.tick_params(axis='y', labelsize=6.5)
            ax.grid(True, which='major', color='#b0b0b0', lw=0.6)
            lo = min(s['gflops_original'].min(), s['gflops'].min())
            hi = max(s['gflops_original'].max(), s['gflops'].max())
            ax.set_ylim(lo / 1.5, hi * 1.5)
            if r == 0:
                ax.set_title(PAPER_KERNEL_NAMES.get(k, k), fontsize=8.5, pad=3)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--out', default='plots/n_density')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pu.set_professional_style()
    mpl.rcParams.update(PAPER_RC)
    d = picked()
    print(d.groupby('pick')['matrix'].nunique().sort_values(ascending=False))
    fig_heatmap(d, out)
    fig_saturation(d, out)
    print(f'Saved to {out}')


if __name__ == '__main__':
    main()
