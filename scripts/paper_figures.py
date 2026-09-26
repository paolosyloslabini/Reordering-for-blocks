#!/usr/bin/env python3
"""
Figures for the IPDPS paper, drawn at their printed size.

Every figure is sized in inches to its final width in the IEEE two-column
layout (COL_W for one column, PAGE_W for figure*), so LaTeX includes it at
scale 1 and all text prints at the font sizes set in PAPER_RC (8-9 pt).

Figures (original matrices, symmetric reordering, n_cols = 256 unless noted):
  bsr_sanity.pdf            density improvement 32x32 vs speedup, cuSPARSE-BSR
  improvement_vs_speedup.pdf  density improvement 16x16 vs speedup, 2x3 kernels
  improvement_vs_speedup_6x1.pdf  the same, six panels in one row
  corr_blocksize_ncols.pdf  correlation by block size (top) and by n_cols (bottom)
  corr_by_metric.pdf        correlation of each structural metric
  sym_density.pdf, row_density.pdf
                            block density improvement 16x16 per reordering,
                            original (top) and scrambled (bottom) matrices
  sym_density_profile.pdf, row_density_profile.pdf
                            performance profiles of 16x16 block density
  sym_density_profile_pooled.pdf, row_density_profile_pooled.pdf
                            one-panel alternative: original + scrambled pooled,
                            reorderings only (no Original / Random)
  density_boxes_2x2.pdf, density_profiles_2x2.pdf
                            block density improvement boxes and profiles,
                            original / scrambled x symmetric / row, one column
  sym_density_combo.pdf, row_density_combo.pdf
                            alternative: density boxes (left) and profile
                            (right) side by side, original over scrambled
  sym_all_metrics.pdf       improvement of every structural metric (symmetric)
  speedup_grid_nc256.pdf    per-kernel speedup, original/scrambled x symmetric/row
  speedup_column_nc256.pdf  the same, four panels stacked in one column
  pick_by_block_density.pdf speedup of picking, within a matrix, the ordering
                            block density prefers over the one another metric
                            prefers (symmetric / row, original matrices)

Run from the repo root:
    .venv/Scripts/python.exe scripts/paper_figures.py [--out plots/paper] [--only NAME ...]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import (FuncFormatter, LogLocator, MultipleLocator,
                               NullFormatter)

import plot_utils as pu
from settings import (KERNEL_NAMES, GROUPED_SCATTER_EXCLUDE, BLOCK_SIZES,
                      PERMS, get_perm_display)
from correlation_table import compute_imp_correlations, _ordered_kernels

# IEEEtran widths (\columnwidth = 252pt, \textwidth = 516pt).
COL_W = 3.49
PAGE_W = 7.16

PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.title_fontsize': 8.5,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.45,
    'ytick.minor.width': 0.45,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.labelpad': 2,
    'hatch.linewidth': 0.5,
    'savefig.dpi': 600,          # for rasterised scatter points in the PDFs
    'pdf.fonttype': 42,          # TrueType, not Type 3 (IEEE PDF eXpress)
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
}

# Kernel names as the paper spells them (macros.tex).
PAPER_KERNEL_NAMES = {
    'ACCSPMM_SPMM': 'Acc-SpMM',
    'ASPT_SPMM': 'ASpT',
    'CUSPARSE_SPMM_BSR_bs32': 'cuSPARSE-BSR',
    'CUSPARSE_SPMM_CSR': 'cuSPARSE-CSR',
    'DTC_SPMM': 'DTC-SpMM',
    'FLASHSPARSE_SPMM': 'FlashSparse',
    'SMAT_SPMM_bs32': 'SMaT',
}

SPEEDUP_LABEL = r'Speedup ($t_{\mathrm{original}}\,/\,t_{\mathrm{reordered}}$)'
RATIO_YLIM = (1 / 7, 7)
RATIO_XLIM = (0.05, 10)

# Scatter colours: both ratios above 1, both below 1, disagreement.
C_UP, C_DOWN, C_MIXED = '#006400', '#8B0000', '#A0A0A0'

# Block-size shades (purple) and n_cols shades (teal), as in the thesis.
BS_COLORS = ['#f0d9f0', '#d9a6d9', '#b06cb0', '#8c3f8c', '#6a1b6a', '#3d003d']
NCOLS_COLORS = ['#b2dfdb', '#4db6ac', '#00695c']
NCOLS_HATCHES = ['', '//////', '']

# Structural metrics (Fig. corr_by_metric). Neutral greys + hatches, so they
# cannot be mistaken for the reordering palette; block density keeps the
# purple of the block-size panel.
METRICS = [
    ('bandwidth_improvement', 'Bandwidth', '#f7f7f7', ''),
    ('col_spread_improvement', 'Column spread', '#dedede', '//////'),
    ('vertical_adjacency_improvement', r'VAR', '#c4c4c4', ''),
    ('profile_improvement', 'Profile', '#a8a8a8', 'xxxx'),
    ('reuse_distance_improvement', 'Reuse distance', '#8c8c8c', ''),
    ('index_distance_improvement', 'Index distance', '#6e6e6e', '\\\\\\\\\\\\'),
    ('density_improvement_16', r'Block density', BS_COLORS[2], ''),
]


def ratio_label(x, pos=None):
    """Tick label for a ratio axis: 0.5x, 1x, 2x."""
    return f'{x:g}×'


def format_ratio_axis(axis, majors):
    axis.set_major_locator(mpl.ticker.FixedLocator(majors))
    axis.set_major_formatter(FuncFormatter(ratio_label))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
    axis.set_minor_formatter(NullFormatter())


def style_ratio_scatter(ax, x_majors=(0.1, 0.3, 1, 3, 10)):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(RATIO_XLIM)
    ax.set_ylim(RATIO_YLIM)
    format_ratio_axis(ax.xaxis, list(x_majors))
    format_ratio_axis(ax.yaxis, [0.2, 0.5, 1, 2, 5])
    ax.grid(True, which='major', color='#b0b0b0', linewidth=0.6, alpha=0.8)
    ax.grid(True, which='minor', color='#d0d0d0', linewidth=0.4, alpha=0.6)
    ax.set_axisbelow(True)
    ax.axhline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
    ax.axvline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)


def quadrant_scatter(ax, x, y, size):
    x, y = np.asarray(x, float), np.asarray(y, float)
    colors = np.where((x >= 1) & (y >= 1), C_UP,
                      np.where((x < 1) & (y < 1), C_DOWN, C_MIXED))
    ax.scatter(x, y, s=size, c=colors, alpha=0.6, edgecolor='none',
               zorder=3, rasterized=True)


def annotate_fit(ax, x, y, label, fs=8.5):
    r = pu._correlation_for_scatter(x, y, 'pearson', log_x=True, log_y=True)
    alpha = pu._loglog_slope(x, y, log_x=True, log_y=True)
    ax.text(0.03, 0.97, f'$r_{{\\log}}={r:.2f}$\n$\\alpha={alpha:.2f}$',
            transform=ax.transAxes, va='top', ha='left', fontsize=fs,
            linespacing=1.1,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#999999',
                      lw=0.4, alpha=0.9), zorder=5)
    ax.text(0.97, 0.03, label, transform=ax.transAxes, ha='right',
            va='bottom', fontsize=fs + 0.5, fontweight='bold',
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none',
                      alpha=0.8), zorder=5)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_bsr_sanity(df, out):
    d = df[df['kernel_id'] == 'CUSPARSE_SPMM_BSR_bs32']
    d = d.dropna(subset=['density_improvement_32', 'speedup'])
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    quadrant_scatter(ax, d['density_improvement_32'], d['speedup'], 2)
    style_ratio_scatter(ax)
    annotate_fit(ax, d['density_improvement_32'], d['speedup'],
                 PAPER_KERNEL_NAMES['CUSPARSE_SPMM_BSR_bs32'])
    ax.set_xlabel(r'Block density improvement ($32{\times}32$)')
    ax.set_ylabel(SPEEDUP_LABEL)
    fig.savefig(out / 'bsr_sanity.pdf')
    plt.close(fig)


def fig_improvement_vs_speedup(df, out):
    kernels = [k for k in _ordered_kernels(df, KERNEL_NAMES)
               if k not in GROUPED_SCATTER_EXCLUDE]
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, 3.3), sharex=True,
                             sharey=True)
    for ax, k in zip(axes.flat, kernels):
        d = df[df['kernel_id'] == k].dropna(subset=['density_improvement_16', 'speedup'])
        quadrant_scatter(ax, d['density_improvement_16'], d['speedup'], 2)
        # No 10x label: it would collide with the next panel's first label.
        style_ratio_scatter(ax, x_majors=(0.1, 0.3, 1, 3))
        annotate_fit(ax, d['density_improvement_16'], d['speedup'],
                     PAPER_KERNEL_NAMES.get(k, k))
    fig.supxlabel(r'Block density improvement ($16{\times}16$)', fontsize=9,
                  y=0.005)
    fig.supylabel(SPEEDUP_LABEL, fontsize=9, x=0.0)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.995, bottom=0.1,
                        wspace=0.06, hspace=0.08)
    fig.savefig(out / 'improvement_vs_speedup.pdf')
    plt.close(fig)


def fig_improvement_vs_speedup_row(df, out):
    """Same as fig_improvement_vs_speedup, six panels in one row (6x1)."""
    kernels = [k for k in _ordered_kernels(df, KERNEL_NAMES)
               if k not in GROUPED_SCATTER_EXCLUDE]
    fig, axes = plt.subplots(1, 6, figsize=(PAGE_W, 1.5), sharex=True, sharey=True)
    for ax, k in zip(axes, kernels):
        d = df[df['kernel_id'] == k].dropna(subset=['density_improvement_16', 'speedup'])
        quadrant_scatter(ax, d['density_improvement_16'], d['speedup'], 1.2)
        style_ratio_scatter(ax, x_majors=(0.1, 1, 5))
        annotate_fit(ax, d['density_improvement_16'], d['speedup'],
                     PAPER_KERNEL_NAMES.get(k, k), fs=7.5)
    fig.supxlabel(r'Block density improvement ($16{\times}16$)', fontsize=9, y=0.0)
    axes[0].set_ylabel('Speedup')
    fig.subplots_adjust(left=0.06, right=0.998, top=0.995, bottom=0.2, wspace=0.05)
    fig.savefig(out / 'improvement_vs_speedup_6x1.pdf')
    plt.close(fig)


def _corr_values(corr_df, kernels, metric):
    col = f'{metric}_corr'
    vals = []
    for k in kernels:
        row = corr_df[corr_df['kernel'] == k]
        v = row.iloc[0].get(col, np.nan) if not row.empty else np.nan
        vals.append(0.0 if np.isnan(v) else v)
    return np.array(vals)


def _bars(ax, series, kernels):
    """series: list of (values, label, color, hatch)."""
    n = len(series)
    width = 0.84 / n
    x = np.arange(len(kernels))
    for i, (vals, label, color, hatch) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label,
               color=color, hatch=hatch, edgecolor='#222222', linewidth=0.5,
               zorder=3)
    ax.set_xlim(-0.5, len(kernels) - 0.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#d8d8d8', linewidth=0.4)
    ax.grid(False, axis='x', which='both')
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    names = [PAPER_KERNEL_NAMES.get(k, k) for k in kernels]
    labels = ax.set_xticklabels([two_line(n) for n in names], linespacing=0.9,
                                fontsize=7.5)
    # In one column the two adjacent "cuSPARSE" labels are wider than a group:
    # push them apart, into the room left by their short neighbours.
    from matplotlib.transforms import ScaledTranslation
    for lab, n in zip(labels, names):
        dx = {'cuSPARSE-BSR': -2, 'cuSPARSE-CSR': 2}.get(n, 0)
        if dx:
            lab.set_transform(lab.get_transform()
                              + ScaledTranslation(dx / 72, 0, ax.figure.dpi_scale_trans))


def two_line(name):
    """Horizontal kernel tick label: cuSPARSE-BSR -> cuSPARSE / BSR, etc."""
    return (name.replace('cuSPARSE-', 'cuSPARSE\n').replace('-SpMM', '-\nSpMM')
            .replace('FlashSparse', 'Flash-\nSparse'))


def _legend_right(ax, title):
    ax.legend(title=title, loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, handlelength=1.3, handleheight=0.9,
              labelspacing=0.3, borderaxespad=0.0, alignment='left')


def _legend_top(ax, title, ncol, y=1.01, inline=True):
    """Legend above the axes; the title inline on the left (one row) or on top."""
    handles, labels = ax.get_legend_handles_labels()
    kw = dict(loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
              handlelength=1.1, handleheight=0.9, columnspacing=0.8,
              handletextpad=0.3, labelspacing=0.25, borderaxespad=0.0)
    if inline:
        ax.legend([Patch(visible=False)] + handles, [title] + labels, **kw)
    else:
        ax.legend(handles, labels, title=title, alignment='center', **kw)


def fig_corr_blocksize_ncols(df, out):
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    bs_metrics = [f'density_improvement_{bs}' for bs in BLOCK_SIZES]
    corr = compute_imp_correlations(df, 256, bs_metrics, kernels,
                                    method='pearson', log_transform=True)
    top = [(_corr_values(corr, kernels, m), f'${bs}{{\\times}}{bs}$', c, '')
           for m, bs, c in zip(bs_metrics, BLOCK_SIZES, BS_COLORS)]

    ncols = sorted(df['n_cols'].unique())
    bottom = []
    for nc, c, h in zip(ncols, NCOLS_COLORS, NCOLS_HATCHES):
        cd = compute_imp_correlations(df, nc, ['density_improvement_16'],
                                      kernels, method='pearson',
                                      log_transform=True)
        bottom.append((_corr_values(cd, kernels, 'density_improvement_16'),
                       f'{int(nc)}', c, h))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 3.2), sharex=True)
    _bars(a1, top, kernels)
    _bars(a2, bottom, kernels)
    a1.tick_params(axis='x', labelbottom=False)
    _legend_top(a1, 'Block size:', ncol=7)
    _legend_top(a2, r'$n_{\mathrm{cols}}$:', ncol=4)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.93, bottom=0.1, hspace=0.2)
    # One label for both panels, centred on the axes (not the tick labels).
    mid = (a1.get_position().y1 + a2.get_position().y0) / 2
    fig.text(0.0, mid, 'Pearson correlation of block density and speedup',
             rotation=90, ha='left', va='center', fontsize=9)
    fig.savefig(out / 'corr_blocksize_ncols.pdf')
    plt.close(fig)


def fig_corr_by_metric(df, out):
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    corr = compute_imp_correlations(df, 256, [m for m, *_ in METRICS],
                                    kernels, method='pearson',
                                    log_transform=True)
    series = [(_corr_values(corr, kernels, m), lab, c, h)
              for m, lab, c, h in METRICS]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    _bars(ax, series, kernels)
    ax.set_ylabel('Pearson corr. with speedup')
    _legend_top(ax, 'Improvement of', ncol=4, inline=False)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.74, bottom=0.16)
    fig.savefig(out / 'corr_by_metric.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reordering-quality and speedup figures (four pipelines)
# ---------------------------------------------------------------------------

PIPELINES = {  # (random, perm_type)
    ('original', 'SYMMETRIC'): (False, 'SYMMETRIC'),
    ('scrambled', 'SYMMETRIC'): (True, 'SYMMETRIC'),
    ('original', 'ROW'): (False, 'ROW'),
    ('scrambled', 'ROW'): (True, 'ROW'),
}
DATASET_TITLE = {'original': 'Original matrices', 'scrambled': 'Scrambled matrices'}
REORDER_TITLE = {'SYMMETRIC': 'Symmetric reordering', 'ROW': 'Row reordering'}

# Structural metrics in the all-metrics box plot: raw column, improvement
# column, higher_is_better, display name (paper spelling).
QUALITY_METRICS = [
    ('bandwidth_max', 'bandwidth_improvement', False, 'Bandwidth'),
    ('locality_avg_col_spread', 'col_spread_improvement', False, 'Column spread'),
    ('locality_vertical_adjacency_ratio', 'vertical_adjacency_improvement', True, 'VAR'),
    ('locality_profile', 'profile_improvement', False, 'Profile'),
    ('access_dist_reuse_distance_mean', 'reuse_distance_improvement', False, 'Reuse distance'),
    ('access_dist_index_distance_mean', 'index_distance_improvement', False, 'Index distance'),
    ('block_density_16', 'density_improvement_16', True, r'Block density $16{\times}16$'),
]

_CACHE = {}


def load_pipeline(dataset, perm_type):
    """(ops with speedup, analysis) for one pipeline, filtered as in plot.py."""
    key = (dataset, perm_type)
    if key not in _CACHE:
        random = dataset == 'scrambled'
        df, da, cfg = pu.load_and_filter_data(cli_overrides={'random': random})
        pkey = ('random_' if random else '') + perm_type.lower()
        df, da = pu.split_by_perm_type(df, da, perm_type,
                                       cfg.get('pipelines', {}).get(pkey, {}))
        df = pu.prepare_full_dataframe(df) if not df.empty else df
        _CACHE[key] = (df, da)
    return _CACHE[key]


def display_name(perm):
    d = 'Original' if perm == 'None' else get_perm_display(perm)
    return 'METIS' if d == 'Metis' else d


def strategy_colors():
    return {display_name(p): v['color'] for p, v in PERMS.items()}


def strategy_order(present):
    return [s for s in strategy_colors() if s in present and s != 'Original']


def analysis_improvements(dataset, perm_type):
    """Analysis rows of the reordered matrices with every improvement ratio."""
    _, da = load_pipeline(dataset, perm_type)
    da = da.copy()
    da['strategy'] = da['perm'].fillna('None').astype(str).map(display_name)
    cfg = {raw: {'improvement_name': imp, 'higher_is_better': hib}
           for raw, imp, hib, _ in QUALITY_METRICS if raw in da.columns}
    da = pu.compute_improvements(da, cfg)
    return da[da['strategy'] != 'Original']


def square_legend(fig, labels, colors, y, ncol=6, x=0.5, alpha=0.85):
    """Legend of small coloured squares above the axes, read row by row."""
    rows = -(-len(labels) // ncol)
    ordered = [labels[r * ncol + c] for c in range(ncol) for r in range(rows)
               if r * ncol + c < len(labels)]
    handles = [Patch(facecolor=colors[l], edgecolor='#222222', linewidth=0.5,
                     alpha=alpha, label=l) for l in ordered]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(x, y),
               ncol=ncol, frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=0.9, handletextpad=0.35, labelspacing=0.25)


def panel_title(ax, text, loc='upper right'):
    xy = {'upper right': (0.985, 0.95, 'right', 'top'),
          'upper left': (0.015, 0.95, 'left', 'top'),
          'lower right': (0.985, 0.05, 'right', 'bottom')}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=9, fontweight='bold', zorder=6,
            bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none',
                      alpha=0.85))


def shared_ylabel(fig, axes, text, x=0.0):
    mid = (axes[0].get_position().y1 + axes[-1].get_position().y0) / 2
    fig.text(x, mid, text, rotation=90, ha='left', va='center', fontsize=9)


def style_ratio_boxaxis(ax, lo, hi):
    """Log y axis for ratios (2x, 0.5x) with a dashed line at 1x."""
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    if hi / lo > 2e3:          # many decades: one tick per decade, 10^k x
        majors = [10.0 ** k for k in range(int(np.floor(np.log10(lo))),
                                           int(np.ceil(np.log10(hi))) + 1)
                  if lo <= 10.0 ** k <= hi]
        format_ratio_axis(ax.yaxis, majors)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: '1\u00d7' if abs(v - 1) < 1e-9
            else f'$10^{{{int(round(np.log10(v)))}}}\\times$'))
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    else:
        cands = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
        majors = [m for m in cands if lo <= m <= hi]
        if len(majors) > 6:
            majors = [m for m in majors if f'{m:g}'.lstrip('0.')[0] in '15']
        format_ratio_axis(ax.yaxis, majors)
    ax.axhline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8,
               zorder=2)
    ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
    ax.grid(False, axis='x', which='both')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', which='both', length=0)


def boxes(ax, data, positions, colors, width, lw=0.6):
    bp = ax.boxplot(data, positions=positions, widths=width, whis=(5, 95),
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='#CC0000', linewidth=lw + 0.4),
                    whiskerprops=dict(linewidth=lw), capprops=dict(linewidth=lw),
                    boxprops=dict(linewidth=lw), zorder=3)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor('#222222')


def grouped_boxes(ax, df, group_col, groups, hues, value_col, colors,
                  width=0.86, lw=0.45):
    """One cluster of boxes per group, one box per hue (strategy)."""
    n = len(hues)
    w = width / n
    data, pos, cols = [], [], []
    for gi, g in enumerate(groups):
        dg = df[df[group_col] == g]
        for hi, h in enumerate(hues):
            v = dg.loc[dg['strategy'] == h, value_col].values
            v = v[np.isfinite(v) & (v > 0)]
            if len(v):
                data.append(v)
                pos.append(gi + (hi - (n - 1) / 2) * w)
                cols.append(colors[h])
    boxes(ax, data, pos, cols, w * 0.8, lw)
    for gi in range(len(groups) - 1):
        ax.axvline(gi + 0.5, color='#c8c8c8', linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    return data


def _whisker_range(data, pad=1.35):
    lo = min(np.percentile(v, 5) for v in data)
    hi = max(np.percentile(v, 95) for v in data)
    return lo / pad, hi * pad


def fig_density_boxes(out, perm_type, fname):
    """Block density improvement per reordering, original over scrambled."""
    panels = [(DATASET_TITLE[d], analysis_improvements(d, perm_type))
              for d in ('original', 'scrambled')]
    present = set().union(*(set(d['strategy']) for _, d in panels))
    order = strategy_order(present)
    colors = strategy_colors()

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.9), sharex=True)
    for ax, (title, d) in zip(axes, panels):
        d = d[d['density_improvement_16'] > 0]
        pos = [i for i, s in enumerate(order) if s in set(d['strategy'])]
        data = [d.loc[d['strategy'] == order[i], 'density_improvement_16'].values
                for i in pos]
        boxes(ax, data, pos, [colors[order[i]] for i in pos], 0.62, lw=0.7)
        style_ratio_boxaxis(ax, *_whisker_range(data, 1.4))
        panel_title(ax, title)
    axes[1].set_xlim(-0.6, len(order) - 0.4)
    axes[1].set_xticks(np.arange(len(order)))
    axes[1].set_xticklabels(order, rotation=35, ha='right',
                            rotation_mode='anchor')
    fig.subplots_adjust(left=0.135, right=0.995, top=0.99, hspace=0.07)
    shared_ylabel(fig, axes, r'Block density improvement ($16{\times}16$)')
    fig.savefig(out / fname)
    plt.close(fig)


def profile_curves(perm_type, dataset):
    """Dolan-More survival profile of 16x16 block density (Original included)."""
    _, da = load_pipeline(dataset, perm_type)
    sub = da[['matrix', 'perm', 'block_density_16']].copy()
    sub['perm'] = sub['perm'].fillna('None').astype(str)
    n_perms = sub['perm'].nunique()
    complete = sub.groupby('matrix')['perm'].nunique()
    sub = sub[sub['matrix'].isin(complete[complete == n_perms].index)]
    sub = sub.dropna(subset=['block_density_16'])
    sub = sub[sub['block_density_16'] > 0]
    best = sub.groupby('matrix')['block_density_16'].transform('max')
    sub['tau'] = sub['block_density_16'] / best
    n = sub['matrix'].nunique()
    return n, {p: np.sort(g['tau'].values) for p, g in sub.groupby('perm')}


def fig_profiles(out, perm_type, fname):
    colors = strategy_colors()
    colors['Original'] = '#CC0000'
    colors['Random'] = '#000000'
    panels = [(DATASET_TITLE[d], *profile_curves(perm_type, d))
              for d in ('original', 'scrambled')]
    present = set()
    for _, _, curves in panels:
        present |= {display_name(p) for p in curves}
    labels = [s for s in colors if s in present and s not in ('Original', 'Unscramble')]
    labels += ['Original'] if 'Original' in present else []

    xlo, xhi = 2 ** -4, 1.04
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.3), sharex=True)
    for ax, (title, n, curves) in zip(axes, panels):
        for lab in labels:
            perm = next((p for p in curves if display_name(p) == lab), None)
            if perm is None:
                continue
            taus = curves[perm][::-1]                    # descending
            fracs = np.arange(1, len(taus) + 1) / n
            xs = np.concatenate([[xhi], taus, [xlo]])
            ys = np.concatenate([[(taus >= 1 - 1e-9).sum() / n], fracs, [fracs[-1]]])
            orig = lab == 'Original'
            ax.step(xs, ys, where='post', color=colors[lab],
                    linewidth=1.1 if orig else 0.85,
                    linestyle=(0, (1.2, 1.2)) if orig else '-',
                    zorder=4 if orig else 3)
        ax.set_xscale('log', base=2)
        ax.set_xlim(xhi, xlo)                            # 100% on the left
        ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 0.5, 0.25, 0.125, 0.0625]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:.0f}%'))
        ax.grid(True, which='major', color='#b8b8b8', linewidth=0.6)
        ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
        ax.set_axisbelow(True)
        panel_title(ax, title, loc='lower right')
    axes[1].set_xlabel('Block density threshold (relative to best)')
    fig.subplots_adjust(left=0.12, right=0.99, top=0.87, bottom=0.1, hspace=0.07)
    shared_ylabel(fig, axes, 'Fraction of reorderings above threshold')
    square_legend(fig, labels, colors, y=0.875, ncol=6, x=0.555, alpha=1.0)
    fig.savefig(out / fname)
    plt.close(fig)


def fig_all_metrics(out):
    """One column; Random's VAR on the original matrices (down to ~1e-4) is
    left off-scale so it does not squash every other box."""
    metrics = [imp for _, imp, _, _ in QUALITY_METRICS]
    names = {imp: name.replace(r' $16{\times}16$', '')
             for _, imp, _, name in QUALITY_METRICS}
    panels = []
    for d in ('original', 'scrambled'):
        da = analysis_improvements(d, 'SYMMETRIC')
        long = da.melt(id_vars=['matrix', 'strategy'], value_vars=metrics,
                       var_name='metric', value_name='improvement')
        panels.append((DATASET_TITLE[d], long))
    present = set().union(*(set(l['strategy']) for _, l in panels))
    order = strategy_order(present)
    colors = strategy_colors()

    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.6), sharex=True)
    for ax, (title, long) in zip(axes, panels):
        data = grouped_boxes(ax, long, 'metric', metrics, order,
                             'improvement', colors, width=0.9, lw=0.35)
        clipped = long[~((long['strategy'] == 'Random')
                         & (long['metric'] == 'vertical_adjacency_improvement'))]
        rng = [g['improvement'].values for _, g in clipped.groupby(['metric', 'strategy'])]
        rng = [v[np.isfinite(v) & (v > 0)] for v in rng]
        style_ratio_boxaxis(ax, *_whisker_range([v for v in rng if len(v)], 1.25))
        ax.set_title(title, fontsize=9, fontweight='bold', loc='left', pad=2)
    axes[1].set_xticks(np.arange(len(metrics)))
    axes[1].set_xticklabels([names[m] for m in metrics], rotation=30,
                            ha='right', rotation_mode='anchor')
    fig.subplots_adjust(left=0.14, right=0.995, top=0.84, bottom=0.1, hspace=0.18)
    shared_ylabel(fig, axes, 'Improvement ratio')
    square_legend(fig, order, colors, y=0.87, ncol=6, x=0.56)
    fig.savefig(out / 'sym_all_metrics.pdf')
    plt.close(fig)


PROFILE_EXCLUDE = {'None', 'random1D', 'Unscramble'}


def pooled_profile_curves(perm_type, datasets=('original', 'scrambled')):
    """Profile over the given datasets pooled, reorderings only.

    Original, Random and Unscramble are left out (each exists in only one of
    the two datasets), so tau is relative to the best reordering algorithm.
    """
    taus, n = {}, 0
    for dataset in datasets:
        _, da = load_pipeline(dataset, perm_type)
        sub = da[['matrix', 'perm', 'block_density_16']].copy()
        sub['perm'] = sub['perm'].fillna('None').astype(str)
        sub = sub[~sub['perm'].isin(PROFILE_EXCLUDE)]
        sub = sub.dropna(subset=['block_density_16'])
        sub = sub[sub['block_density_16'] > 0]
        n_perms = sub['perm'].nunique()
        complete = sub.groupby('matrix')['perm'].nunique()
        sub = sub[sub['matrix'].isin(complete[complete == n_perms].index)]
        sub['tau'] = sub['block_density_16'] / sub.groupby('matrix')['block_density_16'].transform('max')
        n += sub['matrix'].nunique()
        for p, g in sub.groupby('perm'):
            taus.setdefault(p, []).append(g['tau'].values)
    return n, {p: np.sort(np.concatenate(v)) for p, v in taus.items()}


def fig_profiles_pooled(out, perm_type, fname):
    """One-panel alternative to fig_profiles: original + scrambled pooled."""
    colors = strategy_colors()
    n, curves = pooled_profile_curves(perm_type)
    present = {display_name(p) for p in curves}
    labels = [s for s in colors if s in present]
    xlo, xhi = (2 ** -3 if perm_type == 'ROW' else 2 ** -4), 1.04   # row: all done by 12.5%
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    for lab in labels:
        perm = next(p for p in curves if display_name(p) == lab)
        taus = curves[perm][::-1]
        fracs = np.arange(1, len(taus) + 1) / n
        xs = np.concatenate([[xhi], taus, [xlo]])
        ys = np.concatenate([[(taus >= 1 - 1e-9).sum() / n], fracs, [fracs[-1]]])
        ax.step(xs, ys, where='post', color=colors[lab], linewidth=0.85, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xlim(xhi, xlo)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 0.5, 0.25, 0.125, 0.0625]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:.0f}%'))
    ax.grid(True, which='major', color='#b8b8b8', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel('Block density threshold (relative to best)')
    ax.set_ylabel('Fraction of reorderings above threshold')
    fig.subplots_adjust(left=0.12, right=0.99, top=0.8, bottom=0.17)
    square_legend(fig, labels, colors, y=0.81, ncol=5, x=0.555, alpha=1.0)
    fig.savefig(out / fname)
    plt.close(fig)
    print(f'{fname}: {n} matrices (original + scrambled)')


def _profile_axis(ax, n, curves, labels, colors, xlo=2 ** -4, xhi=1.04):
    for lab in labels:
        perm = next((p for p in curves if display_name(p) == lab), None)
        if perm is None:
            continue
        taus = curves[perm][::-1]
        fracs = np.arange(1, len(taus) + 1) / n
        xs = np.concatenate([[xhi], taus, [xlo]])
        ys = np.concatenate([[(taus >= 1 - 1e-9).sum() / n], fracs, [fracs[-1]]])
        ax.step(xs, ys, where='post', color=colors[lab], linewidth=0.8, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xlim(xhi, xlo)
    ax.set_ylim(0, 1.03)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 0.5, 0.25, 0.125]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:.0f}%'))
    ax.grid(True, which='major', color='#b8b8b8', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
    ax.set_axisbelow(True)


def fig_density_combo(out, perm_type, fname):
    """Alternative merging the density boxes and the profiles: per row
    (original / scrambled), improvement boxes left and profile right.
    Reorderings only (no Original / Random); one legend, no x labels."""
    colors = strategy_colors()
    exclude = {display_name(p) for p in PROFILE_EXCLUDE}
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.1),
                             gridspec_kw=dict(width_ratios=[1, 1]))
    labels = None
    for r, dataset in enumerate(('original', 'scrambled')):
        d = analysis_improvements(dataset, perm_type)
        d = d[(d['density_improvement_16'] > 0) & ~d['strategy'].isin(exclude)]
        order = strategy_order(set(d['strategy']))
        labels = labels or order
        box_ax, prof_ax = axes[r]
        pos = [i for i, s in enumerate(labels) if s in set(d['strategy'])]
        data = [d.loc[d['strategy'] == labels[i], 'density_improvement_16'].values
                for i in pos]
        boxes(box_ax, data, pos, [colors[labels[i]] for i in pos], 0.66, lw=0.55)
        lo, hi = _whisker_range(data, 1.4)
        style_ratio_boxaxis(box_ax, lo, hi)
        if hi / lo > 100:      # short panel, many decades: 1-3-10 ticks only
            format_ratio_axis(box_ax.yaxis, [m for m in (0.03, 0.1, 0.3, 1, 3, 10, 30)
                                             if lo <= m <= hi])
        box_ax.set_xlim(-0.6, len(labels) - 0.4)
        box_ax.set_xticks([])
        n, curves = pooled_profile_curves(perm_type, (dataset,))
        _profile_axis(prof_ax, n, curves, labels, colors,
                      xlo=2 ** -3 if perm_type == 'ROW' else 2 ** -4)
        prof_ax.yaxis.tick_right()
        if r == 0:
            prof_ax.tick_params(axis='x', labelbottom=False)
        panel_title(box_ax, DATASET_TITLE[dataset].split()[0], loc='upper right')
    axes[0][0].set_title('Improvement', fontsize=9, pad=2)
    axes[0][1].set_title('Profile (vs. best)', fontsize=9, pad=2)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.8, bottom=0.07,
                        hspace=0.08, wspace=0.06)
    shared_ylabel(fig, axes[:, 0], r'Block density improvement ($16{\times}16$)')
    mid = (axes[0][1].get_position().y1 + axes[1][1].get_position().y0) / 2
    fig.text(axes[0][1].get_position().x1 + 0.095, mid, 'Matrices above threshold',
             rotation=270, ha='left', va='center', fontsize=9)
    square_legend(fig, labels, colors, y=0.845, ncol=5, x=0.5, alpha=0.85)
    fig.savefig(out / fname)
    plt.close(fig)


CELLS_2X2 = [('original', 'SYMMETRIC'), ('original', 'ROW'),
             ('scrambled', 'SYMMETRIC'), ('scrambled', 'ROW')]


def _grid_titles(fig, axes):
    """Column titles (reordering) on top, row titles (dataset) on the right."""
    for c, pt in enumerate(('SYMMETRIC', 'ROW')):
        axes[0][c].set_title(REORDER_TITLE[pt], fontsize=9, fontweight='bold', pad=3)
    for r, ds in enumerate(('original', 'scrambled')):
        pos = axes[r][1].get_position()
        fig.text(pos.x1 + 0.012, (pos.y0 + pos.y1) / 2, DATASET_TITLE[ds].split()[0],
                 rotation=270, ha='left', va='center', fontsize=9, fontweight='bold')


def fig_density_boxes_2x2(out, fname='density_boxes_2x2.pdf'):
    """Block density improvement per reordering: original / scrambled (rows)
    x symmetric / row reordering (columns). Random included; no x labels."""
    colors = strategy_colors()
    data = {cell: analysis_improvements(*cell) for cell in CELLS_2X2}
    for cell, d in data.items():
        data[cell] = d[d['density_improvement_16'] > 0]
    present = set().union(*(set(d['strategy']) for d in data.values()))
    labels = strategy_order(present)
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.3), sharey='row')
    for (ds, pt), ax in zip(CELLS_2X2, axes.flat):
        d = data[(ds, pt)]
        pos = [i for i, s in enumerate(labels) if s in set(d['strategy'])]
        vals = [d.loc[d['strategy'] == labels[i], 'density_improvement_16'].values
                for i in pos]
        boxes(ax, vals, pos, [colors[labels[i]] for i in pos], 0.66, lw=0.55)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_xticks([])
    for r, ds in enumerate(('original', 'scrambled')):
        vals = [d.loc[d['strategy'] == s, 'density_improvement_16'].values
                for (dd, _), d in data.items() if dd == ds
                for s in set(d['strategy'])]
        lo, hi = _whisker_range(vals, 1.4)
        for ax in axes[r]:
            style_ratio_boxaxis(ax, lo, hi)
            if hi / lo > 100:
                format_ratio_axis(ax.yaxis, [m for m in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)
                                             if lo <= m <= hi])
        axes[r][1].tick_params(axis='y', which='both', length=0, labelleft=False)
    fig.subplots_adjust(left=0.105, right=0.95, top=0.82, bottom=0.01,
                        wspace=0.04, hspace=0.06)
    shared_ylabel(fig, axes[:, 0], r'Block density improvement ($16{\times}16$)')
    _grid_titles(fig, axes)
    square_legend(fig, labels, colors, y=0.875, ncol=6, x=0.53)
    fig.savefig(out / fname)
    plt.close(fig)


def fig_profiles_2x2(out, fname='density_profiles_2x2.pdf'):
    """Block density profiles: original / scrambled (rows) x symmetric / row
    (columns), with Original (red dotted) and Random (black) curves."""
    colors = strategy_colors()
    colors['Original'] = '#CC0000'
    colors['Random'] = '#000000'
    panels = {cell: profile_curves(cell[1], cell[0]) for cell in CELLS_2X2}
    present = set()
    for _, curves in panels.values():
        present |= {display_name(p) for p in curves}
    labels = [s for s in colors if s in present and s not in ('Original', 'Unscramble')]
    labels += ['Original'] if 'Original' in present else []
    fig, axes = plt.subplots(2, 2, figsize=(COL_W, 3.3), sharex='col', sharey=True)
    for (ds, pt), ax in zip(CELLS_2X2, axes.flat):
        n, curves = panels[(ds, pt)]
        xlo, xhi = (2 ** -3 if pt == 'ROW' else 2 ** -4), 1.04
        for lab in labels:
            perm = next((p for p in curves if display_name(p) == lab), None)
            if perm is None:
                continue
            taus = curves[perm][::-1]
            fracs = np.arange(1, len(taus) + 1) / n
            xs = np.concatenate([[xhi], taus, [xlo]])
            ys = np.concatenate([[(taus >= 1 - 1e-9).sum() / n], fracs, [fracs[-1]]])
            orig = lab == 'Original'
            ax.step(xs, ys, where='post', color=colors[lab],
                    linewidth=1.0 if orig else 0.75,
                    linestyle=(0, (1.2, 1.2)) if orig else '-',
                    zorder=4 if orig else 3)
        ax.set_xscale('log', base=2)
        ax.set_xlim(xhi, xlo)
        ax.set_ylim(0, 1.03)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 0.5, 0.25, 0.125]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(MultipleLocator(0.125))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 100:.0f}%'))
        ax.grid(True, which='major', color='#b8b8b8', linewidth=0.6)
        ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
        ax.set_axisbelow(True)
    for ax in axes[:, 1]:
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)
    for ax in axes[1]:
        for lab in ax.get_xticklabels():
            lab.set_fontsize(7)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.82, bottom=0.1,
                        wspace=0.05, hspace=0.06)
    shared_ylabel(fig, axes[:, 0], 'Matrices above threshold')
    fig.text((axes[1][0].get_position().x0 + axes[1][1].get_position().x1) / 2, 0.015,
             'Block density threshold (relative to best)', ha='center', va='bottom',
             fontsize=9)
    _grid_titles(fig, axes)
    square_legend(fig, labels, colors, y=0.875, ncol=6, x=0.53, alpha=1.0)
    fig.savefig(out / fname)
    plt.close(fig)


def _speedup_cells(n_cols):
    cells, present = {}, set()
    for key in PIPELINES:
        df, _ = load_pipeline(*key)
        d = df[(df['n_cols'] == n_cols) & (df['strategy'] != 'Original')
               & (df['kernel_id'] != 'CUSPARSE_SPMM_BSR_bs32')].copy()
        d['strategy'] = d['strategy'].replace({'Metis': 'METIS'})
        cells[key] = d
        present |= set(d['strategy'])
    kernels = [k for k in _ordered_kernels(cells[('original', 'SYMMETRIC')], KERNEL_NAMES)
               if k != 'CUSPARSE_SPMM_BSR_bs32']
    return cells, strategy_order(present), kernels


# Fixed speedup ranges per dataset (shared by both reorderings).
SPEEDUP_YLIM = {'original': (0.1, 5), 'scrambled': (0.3, 10)}


def _set_kernel_ticks(ax, kernels):
    ax.set_xticks(np.arange(len(kernels)))
    ax.set_xticklabels([PAPER_KERNEL_NAMES.get(k, k) for k in kernels],
                       rotation=25, ha='right', rotation_mode='anchor')


def fig_speedup_grid(out, n_cols=256):
    """Per-kernel speedup: rows = original / scrambled matrices, columns =
    symmetric / row reordering. Each row shares its y range; one legend."""
    colors = strategy_colors()
    cells, order, kernels = _speedup_cells(n_cols)
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, 4.0), sharex=True,
                             sharey='row')
    for r, dataset in enumerate(('original', 'scrambled')):
        for c, perm_type in enumerate(('SYMMETRIC', 'ROW')):
            ax = axes[r][c]
            grouped_boxes(ax, cells[(dataset, perm_type)], 'kernel_id',
                          kernels, order, 'speedup', colors, lw=0.4)
            style_ratio_boxaxis(ax, *SPEEDUP_YLIM[dataset])
    for c, perm_type in enumerate(('SYMMETRIC', 'ROW')):
        axes[0][c].set_title(REORDER_TITLE[perm_type], fontsize=9,
                             fontweight='bold', pad=3)
        _set_kernel_ticks(axes[1][c], kernels)
    fig.subplots_adjust(left=0.052, right=0.983, top=0.9, bottom=0.1,
                        wspace=0.012, hspace=0.1)
    for r, dataset in enumerate(('original', 'scrambled')):
        pos = axes[r][1].get_position()
        fig.text(pos.x1 + 0.003, (pos.y0 + pos.y1) / 2, DATASET_TITLE[dataset],
                 rotation=270, ha='left', va='center', fontsize=9,
                 fontweight='bold')
    # Put the speedup label right against the tick labels (no dead margin).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_x0 = min(t.get_window_extent(renderer).x0
                  for ax in axes[:, 0] for t in ax.get_yticklabels() if t.get_text())
    lab = fig.text(0, 0.5, SPEEDUP_LABEL, rotation=90, ha='right', va='center',
                   fontsize=9)
    mid = (axes[0][0].get_position().y1 + axes[1][0].get_position().y0) / 2
    lab.set_position(((tick_x0 - 2) / fig.bbox.width, mid))
    square_legend(fig, order, colors, y=0.935, ncol=len(order), x=0.52)
    fig.savefig(out / f'speedup_grid_nc{n_cols}.pdf')
    plt.close(fig)


def fig_speedup_column(out, n_cols=256):
    """Same data as fig_speedup_grid, four panels stacked in one column."""
    colors = strategy_colors()
    cells, order, kernels = _speedup_cells(n_cols)
    panels = [(d, p) for d in ('original', 'scrambled') for p in ('SYMMETRIC', 'ROW')]
    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 5.6), sharex=True)
    for ax, (dataset, perm_type) in zip(axes, panels):
        grouped_boxes(ax, cells[(dataset, perm_type)], 'kernel_id', kernels,
                      order, 'speedup', colors, lw=0.35)
        style_ratio_boxaxis(ax, *SPEEDUP_YLIM[dataset])
        label = (f'{REORDER_TITLE[perm_type]}, '
                 f'{DATASET_TITLE[dataset][0].lower()}{DATASET_TITLE[dataset][1:]}')
        ax.set_title(label, fontsize=9, fontweight='bold', loc='left', pad=2)
    _set_kernel_ticks(axes[-1], kernels)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.9, bottom=0.07, hspace=0.2)
    shared_ylabel(fig, axes, SPEEDUP_LABEL)
    square_legend(fig, order, colors, y=0.917, ncol=6, x=0.56)
    fig.savefig(out / f'speedup_column_nc{n_cols}.pdf')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Picking an ordering by block density vs. by another metric
# ---------------------------------------------------------------------------

PICK_T = 1.1           # a metric prefers an ordering if it is >= 10 % better
PICK_MIN_MATRICES = 10  # leave a bar out if fewer matrices have a conflict
PICK_PANELS = [('SYMMETRIC', 'original'), ('ROW', 'original')]
PICK_BD = 'density_improvement_16'


def _pick_candidates(dataset, perm_type, n_cols=256):
    """All candidate orderings per matrix, 'Original' (no reordering) included
    with every ratio 1. Ratios are relative to the unreordered matrix."""
    df, _ = load_pipeline(dataset, perm_type)
    d = df[df['n_cols'] == n_cols].copy()
    orig = d['strategy'] == 'Original'
    for m, *_ in METRICS:
        d.loc[orig, m] = 1.0
    d.loc[orig, 'speedup'] = 1.0
    d = d.dropna(subset=['speedup', PICK_BD])
    return d[d['speedup'] > 0]


def pick_by_block_density(d, kernels):
    """Within each matrix, every pair of orderings (R1, R2) where block density
    prefers R1 and metric X prefers R2, each by at least PICK_T. Value: speedup
    of R1 over R2, the gain of picking by block density instead of by X;
    geomean per matrix, then over matrices. Also returns the share of ordering
    pairs on which block density and at least one other metric conflict."""
    cols = ['matrix', 'strategy', 'speedup'] + [m for m, *_ in METRICS]
    rows, share = [], None
    for k in kernels:
        s = d.loc[d['kernel_id'] == k, cols]
        p = s.merge(s, on='matrix', suffixes=('_1', '_2'))
        p = p[p['strategy_1'] != p['strategy_2']]
        bd_pref = p[PICK_BD + '_1'] / p[PICK_BD + '_2'] >= PICK_T
        conflict = np.zeros(len(p), bool)
        for m, label, *_ in METRICS[:-1]:
            sel = bd_pref & (p[m + '_2'] / p[m + '_1'] >= PICK_T)
            conflict |= sel.values
            g = p[sel]
            gain = np.log(g['speedup_1'] / g['speedup_2']).groupby(g['matrix']).mean()
            adv = (np.exp(gain.mean()) if len(gain) >= PICK_MIN_MATRICES
                   else np.nan)
            rows.append((m, label, PAPER_KERNEL_NAMES.get(k, k),
                         int(sel.sum()), len(gain), adv))
        if share is None:   # ordered pairs: each conflict is counted once
            share = 2 * conflict.sum() / len(p)
    t = pd.DataFrame(rows, columns=['metric_id', 'vs_metric', 'kernel',
                                    'n_pairs', 'n_matrices', 'advantage'])
    return t, share


def fig_pick_by_block_density(out, panels=PICK_PANELS,
                              fname='pick_by_block_density.pdf', csv=None,
                              n_cols=256):
    """Bars start at 1x: up = picking the ordering block density prefers beat
    picking the one the other metric prefers. Same fills as corr_by_metric."""
    df, _ = load_pipeline('original', 'SYMMETRIC')
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(COL_W, 0.3 + 1.5 * len(panels)),
                             sharex=True)
    others = METRICS[:-1]
    n, width = len(others), 0.84 / len(others)
    x = np.arange(len(kernels))
    tabs = []
    for ax, (perm_type, dataset) in zip(axes, panels):
        t, share = pick_by_block_density(
            _pick_candidates(dataset, perm_type, n_cols), kernels)
        tabs.append(t.assign(reordering=perm_type.lower(), matrices=dataset,
                             conflict_share=round(share, 4)))
        for i, (m, label, color, hatch) in enumerate(others):
            v = t[t['metric_id'] == m].set_index('kernel').loc[
                [PAPER_KERNEL_NAMES.get(k, k) for k in kernels], 'advantage'].values
            ax.bar(x + (i - (n - 1) / 2) * width, v - 1, width, bottom=1,
                   label=label, color=color, hatch=hatch,
                   edgecolor='#222222', linewidth=0.5, zorder=3)
        ax.set_yscale('log')
        ax.set_ylim(0.65, 2.3)
        format_ratio_axis(ax.yaxis, (0.75, 1, 1.5, 2))
        ax.axhline(1, color='#222222', lw=0.6, zorder=4)
        ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
        ax.grid(False, axis='x', which='both')
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, len(kernels) - 0.5)
        pct = f'{100 * share:.0f}%' if share >= 0.01 else '<1%'
        ax.set_title(f'{REORDER_TITLE[perm_type]}, {dataset} matrices '
                     f'(conflicting verdicts: {pct})', fontsize=7.5, pad=2)
    axes[-1].set_xticks(x)
    labels = axes[-1].set_xticklabels(
        [two_line(PAPER_KERNEL_NAMES.get(k, k)) for k in kernels],
        linespacing=0.9, fontsize=7)
    from matplotlib.transforms import ScaledTranslation
    for lab, k in zip(labels, kernels):
        dx = {'cuSPARSE-BSR': -1.5, 'cuSPARSE-CSR': 1.5}.get(
            PAPER_KERNEL_NAMES.get(k, k), 0)
        if dx:
            lab.set_transform(lab.get_transform() + ScaledTranslation(
                dx / 72, 0, fig.dpi_scale_trans))
    shared_ylabel(fig, axes, 'Speedup by picking\naccording to block density',
                  x=0.0)
    h = fig.get_figheight()
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, title='Block density vs.', loc='upper center',
               bbox_to_anchor=(0.58, 1.0), ncol=3, frameon=False,
               handlelength=1.1, handleheight=0.9, columnspacing=0.8,
               handletextpad=0.3, labelspacing=0.25, borderaxespad=0.0)
    fig.subplots_adjust(left=0.17, right=0.995, top=1 - 0.66 / h,
                        bottom=0.33 / h, hspace=0.3)
    fig.savefig(out / fname)
    plt.close(fig)
    if csv is not None:
        pd.concat(tabs).drop(columns='metric_id').round(3).to_csv(csv, index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--out', default='plots/paper')
    ap.add_argument('--only', nargs='+', metavar='NAME',
                    help='only these figures (output names without .pdf)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pu.set_professional_style()
    mpl.rcParams.update(PAPER_RC)

    def ops():
        df, _ = load_pipeline("original", "SYMMETRIC")
        df256 = df[(df['strategy'] != 'Original') & (df['n_cols'] == 256)]
        return df, df256

    figures = {
        'bsr_sanity': lambda: fig_bsr_sanity(ops()[1], out),
        'improvement_vs_speedup': lambda: fig_improvement_vs_speedup(ops()[1], out),
        'improvement_vs_speedup_6x1': lambda: fig_improvement_vs_speedup_row(ops()[1], out),
        'corr_blocksize_ncols': lambda: fig_corr_blocksize_ncols(ops()[0], out),
        'corr_by_metric': lambda: fig_corr_by_metric(ops()[0], out),
        'sym_density': lambda: fig_density_boxes(out, 'SYMMETRIC', 'sym_density.pdf'),
        'row_density': lambda: fig_density_boxes(out, 'ROW', 'row_density.pdf'),
        'sym_density_profile': lambda: fig_profiles(out, 'SYMMETRIC', 'sym_density_profile.pdf'),
        'row_density_profile': lambda: fig_profiles(out, 'ROW', 'row_density_profile.pdf'),
        'sym_density_profile_pooled': lambda: fig_profiles_pooled(
            out, 'SYMMETRIC', 'sym_density_profile_pooled.pdf'),
        'row_density_profile_pooled': lambda: fig_profiles_pooled(
            out, 'ROW', 'row_density_profile_pooled.pdf'),
        'sym_density_combo': lambda: fig_density_combo(out, 'SYMMETRIC', 'sym_density_combo.pdf'),
        'row_density_combo': lambda: fig_density_combo(out, 'ROW', 'row_density_combo.pdf'),
        'density_boxes_2x2': lambda: fig_density_boxes_2x2(out),
        'density_profiles_2x2': lambda: fig_profiles_2x2(out),
        'sym_all_metrics': lambda: fig_all_metrics(out),
        'speedup_grid_nc256': lambda: fig_speedup_grid(out),
        'speedup_column_nc256': lambda: fig_speedup_column(out),
        'pick_by_block_density': lambda: fig_pick_by_block_density(out),
    }
    for name in args.only or figures:
        figures[name]()
    print(f'Saved paper figures to {out}')


if __name__ == '__main__':
    main()
