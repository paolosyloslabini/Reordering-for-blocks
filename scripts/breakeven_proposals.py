#!/usr/bin/env python3
"""
Break-even proposals: is reordering worth its cost?

Policy modelled: reorder the matrix, run one trial SpMM on it, and keep the
reordering only if that trial is faster than the original; otherwise go back
to the original ordering.  Per (matrix, kernel):

    trial cost   C  = t_reorder + t_reordered          (paid once, always)
    gain / op    g  = t_original - t_reordered         (only if g > 0)
    break-even   N* = C / g  SpMM operations           (never, if g <= 0)

A matrix has "paid off" after N operations if N* <= N.  The fraction of
matrices that have paid off, as a function of N, rises to a plateau: the
plateau height is the chance that the reordering helps at all, and where the
curve rises is how many operations it takes.

Format-conversion (preprocessing) time is left out: it is recorded only for
DTC-SpMM and FlashSparse.  Reordering time is the same for row and symmetric
application (results_reordering*.csv has no perm_type).

Run from the repo root (needs results/ and datasets/):
    .venv/Scripts/python.exe scripts/breakeven_proposals.py [--out DIR] [--n-cols 256]
"""

import argparse
import contextlib
import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import paper_figures as pf
import plot_utils as pu

# Kernels in the paper's order, validated categorical slots 1-7 (no red: red
# means "slower" elsewhere in the paper).
KERNELS = ['ACCSPMM_SPMM', 'ASPT_SPMM', 'CUSPARSE_SPMM_BSR_bs32',
           'CUSPARSE_SPMM_CSR', 'DTC_SPMM', 'FLASHSPARSE_SPMM', 'SMAT_SPMM_bs32']
KERNEL_COLORS = dict(zip(KERNELS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                   '#e87ba4', '#008300', '#4a3aa7']))
KNAME = pf.PAPER_KERNEL_NAMES

REORDER_CSV = {'original': 'results/results_reordering.csv',
               'scrambled': 'results/results_reordering_random.csv'}
DATASET_COLORS = {'original': '#2a78d6', 'scrambled': '#eb6834'}
N_GRID = np.logspace(0, 7, 400)
MIN_GAIN = 0.05        # "clear" win: at least 5% faster


def ops_label(x, pos=None):
    if x < 1:
        return ''
    e = int(round(np.log10(x)))
    return {0: '1', 1: '10', 2: '100', 3: '1k', 4: '10k'}.get(e, f'$10^{{{e}}}$')


def pct(x, pos=None):
    return f'{x:.0%}'


def breakeven_table(n_cols):
    """One row per (pipeline, kernel, strategy, matrix) with N* and speedup."""
    rows = []
    for dataset, perm_type in pf.PIPELINES:
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = pf.load_pipeline(dataset, perm_type)
        df = df[df['n_cols'] == n_cols]
        base = (df[df['strategy'] == 'Original']
                .groupby(['matrix', 'kernel_id'])['time_operation_ms'].mean()
                .rename('t_orig'))
        reo = df[~df['strategy'].isin(['Original', 'Random'])]
        reo = reo.groupby(['matrix', 'kernel_id', 'strategy', 'perm'],
                          as_index=False)['time_operation_ms'].mean()
        reo = reo.join(base, on=['matrix', 'kernel_id'], how='inner')
        t = pd.read_csv(REORDER_CSV[dataset])
        t = t.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
        reo = reo.merge(t, on=['matrix', 'perm'], how='inner')
        reo['strategy'] = reo['strategy'].replace({'Metis': 'METIS'})
        reo['dataset'], reo['perm_type'] = dataset, perm_type
        rows.append(reo)
    d = pd.concat(rows, ignore_index=True)
    d['speedup'] = d['t_orig'] / d['time_operation_ms']
    d['cost'] = d['time_reordering_ms'] + d['time_operation_ms']
    gain = d['t_orig'] - d['time_operation_ms']
    d['n_star'] = np.where(gain > 0, d['cost'] / gain.where(gain > 0), np.inf)
    return d


def paid_off(n_star, grid=N_GRID):
    """Fraction of matrices with N* <= N, for each N in grid."""
    s = np.sort(np.asarray(n_star, float))
    return np.searchsorted(s, grid, side='right') / len(s)


def style_ops_axis(ax, lo=1, hi=1e7):
    ax.set_xscale('log')
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(ops_label))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which='major', color='#d0d0d0', lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# A: RCM, one curve per kernel, 2x2 pipelines (both effects in one line)
# ---------------------------------------------------------------------------

def fig_a(d, out, strategy='RCM', n_max=1e4):
    """Paid-off curves up to n_max operations (every curve has levelled off
    by 1e4 except ASpT, which gains 2-5 points more later). Laid out like the
    paper's 2x2 profiles: reordering in columns, dataset in rows."""
    grid = np.logspace(0, np.log10(n_max), 300)
    fig, axes = plt.subplots(2, 2, figsize=(pf.COL_W, 2.3), sharex=True,
                             sharey=True)
    for (dataset, perm_type), ax in zip(pf.CELLS_2X2, axes.flat):
        sub = d[(d.strategy == strategy) & (d.dataset == dataset) & (d.perm_type == perm_type)]
        for k in KERNELS:
            ns = sub.loc[sub.kernel_id == k, 'n_star']
            if ns.empty:
                continue
            y = paid_off(ns, grid)
            ax.plot(grid, y, color=KERNEL_COLORS[k], lw=1.0, zorder=3)
        style_ops_axis(ax, 1, n_max)
        ax.grid(True, which='major', color='#b0b0b0', lw=0.6)
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([1, 10, 100, 1e3, 1e4]))
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    for ax in axes[:, 1]:
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)
    handles = [Line2D([], [], color=KERNEL_COLORS[k], lw=1.4, label=KNAME[k])
               for k in KERNELS]
    fig.canvas.draw()                  # end tick labels point inwards, so
    for ax in axes[1]:                 # "10k" and "1" of adjacent panels
        ticks = ax.get_xticklabels()   # don't touch
        ticks[0].set_ha('left')
        ticks[-1].set_ha('right')
        for lab in ticks:
            lab.set_fontsize(7)
    for lab in axes[0][0].get_yticklabels():   # 0% would touch the 100% below
        if lab.get_text() == '0%':
            lab.set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.78, bottom=0.12,
                        wspace=0.05, hspace=0.06)
    pf.shared_ylabel(fig, axes[:, 0], f'Matrices where {strategy} paid off')
    fig.text((axes[1][0].get_position().x0 + axes[1][1].get_position().x1) / 2, 0.005,
             'SpMM operations after reordering', ha='center', va='bottom', fontsize=9)
    pf._grid_titles(fig, axes)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.54, 0.855),
               ncol=4, frameon=False, fontsize=7, handlelength=1.2,
               columnspacing=0.8, handletextpad=0.4, labelspacing=0.2,
               borderaxespad=0)
    save(fig, out, f'A_paid_off_curves_{strategy.lower()}')


# ---------------------------------------------------------------------------
# B: RCM, the two effects separated (symmetric, original vs scrambled)
# ---------------------------------------------------------------------------

def fig_b(d, out, strategy='RCM', perm_type='SYMMETRIC'):
    sub = d[(d.strategy == strategy) & (d.perm_type == perm_type)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(pf.COL_W, 2.9), sharey=True,
                                   gridspec_kw={'width_ratios': [1, 1.5],
                                                'wspace': 0.08})
    ys = np.arange(len(KERNELS))[::-1]
    h = 0.36
    for j, dataset in enumerate(('original', 'scrambled')):
        c = DATASET_COLORS[dataset]
        off = (0.5 - j) * h * 1.1
        for y, k in zip(ys, KERNELS):
            s = sub[(sub.dataset == dataset) & (sub.kernel_id == k)]
            if s.empty:
                continue
            clear = (s.speedup >= 1 + MIN_GAIN).mean()
            any_ = (s.speedup > 1).mean()
            ax1.barh(y + off, clear, h, color=c, edgecolor='none')
            ax1.barh(y + off, any_ - clear, h, left=clear, color=c, alpha=0.35,
                     edgecolor='none')
            ax1.text(any_ + 0.02, y + off, f'{any_:.0%}', va='center', fontsize=6)
            ns = s.n_star[np.isfinite(s.n_star)]
            if len(ns) >= 5:
                q = np.percentile(ns, [5, 25, 50, 75, 95])
                ax2.plot([q[0], q[4]], [y + off] * 2, color=c, lw=0.7)
                ax2.add_patch(mpl.patches.Rectangle((q[1], y + off - h / 2),
                                                    q[3] - q[1], h, facecolor=c,
                                                    alpha=0.55, edgecolor=c, lw=0.5))
                ax2.plot([q[2]] * 2, [y + off - h / 2, y + off + h / 2],
                         color='#111111', lw=0.9)
    ax1.set_xlim(0, 1.18)
    ax1.set_xticks([0, 0.5, 1])
    ax1.xaxis.set_major_formatter(FuncFormatter(pct))
    ax1.set_xlabel('Reordering kept\n(faster in the trial)')
    ax1.set_yticks(ys)
    ax1.set_yticklabels([KNAME[k] for k in KERNELS])
    ax1.grid(True, axis='x', color='#d0d0d0', lw=0.5)
    style_ops_axis(ax2, 0.3, 1e5)
    ax2.set_xlabel('SpMM operations to break even\n(kept reorderings only)')
    handles = [Patch(color=DATASET_COLORS['original'], label='Original'),
               Patch(color=DATASET_COLORS['scrambled'], label='Scrambled'),
               Patch(facecolor='#888888', label=f'$\\geq${MIN_GAIN:.0%} faster'),
               Patch(facecolor='#888888', alpha=0.35, label=f'<{MIN_GAIN:.0%} faster')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.55, 0.88),
               ncol=2, frameon=False, fontsize=7, handlelength=1, columnspacing=1)
    save(fig, out, f'B_split_{strategy.lower()}')


# ---------------------------------------------------------------------------
# C: RCM, effective speedup after N operations, reorder cost included
# ---------------------------------------------------------------------------

def fig_c(d, out, strategy='RCM', perm_type='SYMMETRIC'):
    sub = d[(d.strategy == strategy) & (d.perm_type == perm_type)]
    fig, axes = plt.subplots(2, len(KERNELS), figsize=(pf.PAGE_W, 2.6),
                             sharex=True, sharey='row')
    for i, dataset in enumerate(('original', 'scrambled')):
        for j, k in enumerate(KERNELS):
            ax = axes[i, j]
            s = sub[(sub.dataset == dataset) & (sub.kernel_id == k)]
            t_o = s.t_orig.values[:, None]
            t_best = np.minimum(s.t_orig, s.time_operation_ms).values[:, None]
            n = N_GRID[None, :]
            eff = n * t_o / (s.cost.values[:, None] + n * t_best)
            q25, q50, q75 = np.percentile(eff, [25, 50, 75], axis=0)
            c = KERNEL_COLORS[k]
            ax.fill_between(N_GRID, q25, q75, color=c, alpha=0.25, lw=0)
            ax.plot(N_GRID, q50, color=c, lw=1.2)
            ax.axhline(1, color='#CC0000', ls='--', lw=0.7)
            style_ops_axis(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(KNAME[k], fontsize=8, pad=2)
        axes[i, 0].set_ylabel(f'{dataset.capitalize()}\nnet speedup', fontsize=8)
    axes[0, 0].set_ylim(0.1, 4)
    axes[1, 0].set_ylim(0.1, 30)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}×'))
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.supxlabel('SpMM operations N (reordering + one trial run included)',
                  fontsize=8.5, y=-0.02)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    save(fig, out, f'C_net_speedup_{strategy.lower()}')


# ---------------------------------------------------------------------------
# D: all reorderings x kernels, % paid off within N operations
# ---------------------------------------------------------------------------

def fig_d(d, out, n_budget=1000, perm_type='SYMMETRIC'):
    order = [s for s in pf.strategy_order(set(d.strategy)) if s in set(d.strategy)]
    fig, axes = plt.subplots(1, 2, figsize=(pf.PAGE_W * 0.8, 2.7), sharey=True)
    for ax, dataset in zip(axes, ('original', 'scrambled')):
        sub = d[(d.dataset == dataset) & (d.perm_type == perm_type)]
        m = np.full((len(order), len(KERNELS)), np.nan)
        cov = []
        for i, s in enumerate(order):
            ss = sub[sub.strategy == s]
            cov.append(ss.matrix.nunique())
            for j, k in enumerate(KERNELS):
                ns = ss.loc[ss.kernel_id == k, 'n_star']
                if len(ns):
                    m[i, j] = (ns <= n_budget).mean()
        ax.imshow(m, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f'{m[i, j]*100:.0f}', ha='center', va='center',
                            fontsize=6.5, color='white' if m[i, j] > 0.55 else '#222222')
        ax.set_xticks(range(len(KERNELS)))
        ax.set_xticklabels([KNAME[k] for k in KERNELS], rotation=40, ha='right',
                           fontsize=7)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=7)
        for i, c in enumerate(cov):                 # matrices with timings
            ax.text(len(KERNELS) - 0.4, i, f'{c}', va='center', ha='left',
                    fontsize=6.5, color='#555555')
        ax.text(len(KERNELS) - 0.4, -0.75, 'n', va='center', ha='left',
                fontsize=6.5, color='#555555', style='italic')
        ax.set_title(f'{dataset.capitalize()} matrices', fontsize=8.5, pad=3)
        ax.tick_params(length=0)
    fig.suptitle(f'% of matrices where reordering has paid off within '
                 f'{n_budget:,} SpMM operations ({pf.REORDER_TITLE[perm_type].lower()}; '
                 f'n = matrices with timings)', fontsize=8, y=1.02)
    fig.subplots_adjust(wspace=0.12)
    save(fig, out, f'D_heatmap_all_N{n_budget}')


# ---------------------------------------------------------------------------
# E: all reorderings, paid-off curves, one panel per kernel
# ---------------------------------------------------------------------------

def fig_e(d, out, dataset='original', perm_type='SYMMETRIC'):
    sub = d[(d.dataset == dataset) & (d.perm_type == perm_type)]
    order = [s for s in pf.strategy_order(set(sub.strategy))]
    colors = pf.strategy_colors()
    colors['METIS'] = colors.get('METIS', colors.get('Metis'))
    fig, axes = plt.subplots(1, len(KERNELS), figsize=(pf.PAGE_W, 1.9),
                             sharey=True)
    for ax, k in zip(axes, KERNELS):
        for s in order:
            ns = sub.loc[(sub.kernel_id == k) & (sub.strategy == s), 'n_star']
            if len(ns):
                lw = 1.5 if s == 'RCM' else 0.9
                ax.plot(N_GRID, paid_off(ns), color=colors[s], lw=lw)
        style_ops_axis(ax)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.set_ylim(0, 1)
        ax.set_title(KNAME[k], fontsize=8, pad=2)
    axes[0].yaxis.set_major_formatter(FuncFormatter(pct))
    axes[0].set_ylabel('Matrices paid off')
    fig.supxlabel('SpMM operations after reordering', fontsize=8.5, y=-0.06)
    handles = [Line2D([], [], color=colors[s], lw=1.4, label=s) for s in order]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.98),
               ncol=len(order), frameon=False, fontsize=7, handlelength=1.3,
               columnspacing=0.8)
    fig.subplots_adjust(wspace=0.08)
    save(fig, out, f'E_paid_off_all_{dataset}_{perm_type.lower()}')


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png', dpi=200)
    plt.close(fig)
    print('  wrote', name)


def summary(d, out):
    """Numbers quoted in the notes: plateau, clear wins, median N*."""
    g = d[d.strategy == 'RCM'].groupby(['dataset', 'perm_type', 'kernel_id'])
    s = pd.DataFrame({
        'matrices': g.matrix.nunique(),
        'kept_%': g.speedup.apply(lambda x: 100 * (x > 1).mean()).round(0),
        'kept_5pct_%': g.speedup.apply(lambda x: 100 * (x >= 1 + MIN_GAIN).mean()).round(0),
        'median_N*_kept': g.n_star.apply(lambda x: np.median(x[np.isfinite(x)])).round(0),
        'paid_off_1k_%': g.n_star.apply(lambda x: 100 * (x <= 1000).mean()).round(0),
    })
    s.to_csv(out / 'rcm_summary.csv')
    print(s.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--out', default='plots/breakeven_proposals')
    ap.add_argument('--n-cols', type=int, default=256)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pu.set_professional_style()
    mpl.rcParams.update(pf.PAPER_RC)

    d = breakeven_table(args.n_cols)
    d.to_csv(out / 'breakeven_table.csv', index=False)
    summary(d, out)
    fig_a(d, out)
    fig_b(d, out)
    fig_c(d, out)
    fig_d(d, out)
    fig_e(d, out)


if __name__ == '__main__':
    main()
