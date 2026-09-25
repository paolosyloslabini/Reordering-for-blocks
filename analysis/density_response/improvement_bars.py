"""How often the densest reordering helps, and by how much when it does (N = 32).

Four cells: {original, scrambled} matrices x {symmetric, row} reordering.
For every (matrix, kernel) the candidates are that cell's reorderings; the
strategy keeps the candidate with the highest 16x16 block density, or the
original ordering if none is denser. Speedups are relative to the cell's own
starting matrix (for scrambled matrices: the scrambled matrix).

Bottom of each cell: share of matrices where the chosen reordering was faster
(improved) or slower; the rest kept the original ordering (no denser candidate).
Top of each cell: geometric-mean speedup among improved matrices, with a 95%
bootstrap interval over matrices.

All kernels use 32-column runs. ASpT, which barely reacts to reordering, shows
what timing noise alone produces (about half "improved", by about 10%).
Impossible cuSPARSE-BSR / SMaT runs are dropped, as elsewhere.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from common import KERNEL_NAMES, OUT, PAGE_W, load_pipeline, save, style
from decision_test import PEAK_TFLOPS

N_COLS = 32
# Paper palette (scripts/paper_figures.py): improved, degraded, neutral.
GOOD, BAD, NEUTRAL = '#006400', '#8B0000', '#A0A0A0'
KERNEL_ORDER = sorted(KERNEL_NAMES.values(), key=str.lower)   # as in the paper figures
CELLS = [('original', 'SYMMETRIC'), ('original', 'ROW'),
         ('scrambled', 'SYMMETRIC'), ('scrambled', 'ROW')]
TITLES = {'original': 'Original matrices', 'scrambled': 'Scrambled matrices',
          'SYMMETRIC': 'symmetric reordering', 'ROW': 'row reordering'}


def densest_speedups(dataset, perm_type):
    df = load_pipeline(dataset, perm_type)
    df = df[(df['n_cols'] == N_COLS) & (df['strategy'] != 'Original')
            & (df['speedup'] > 0) & np.isfinite(df['speedup'])
            & (df['block_density_16'] > 0) & (df['block_density_16_original'] > 0)]
    for k, peak in PEAK_TFLOPS.items():
        executed = (2 * df['nonzero_blocks_32'] * 1024 * df['n_cols']
                    / (df['time_operation_ms'] / 1e3) / 1e12)
        df = df[~((df['kernel_id'] == k) & (executed > peak))]
    rows = []
    for (k, m), g in df.groupby(['kernel_id', 'matrix']):
        best = g.loc[g['block_density_16'].idxmax()]
        denser = best['block_density_16'] > g['block_density_16_original'].iloc[0]
        rows.append({'kernel_id': k, 'matrix': m,
                     'speedup': best['speedup'] if denser else 1.0})
    return pd.DataFrame(rows)


def summarise(d, rng, n_boot=5000):
    out = []
    for k in KERNEL_NAMES:
        s = d.loc[d['kernel_id'] == k, 'speedup'].values
        imp = s > 1
        gain = np.exp(np.log(s[imp]).mean())
        boot = []
        for _ in range(n_boot):
            b = s[rng.integers(0, len(s), len(s))]
            boot.append(np.exp(np.log(b[b > 1]).mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({'kernel': KERNEL_NAMES[k], 'matrices': len(s),
                    'improved': imp.mean(), 'slower': (s < 1).mean(),
                    'unchanged': (s == 1).mean(),
                    'gain_improved': gain, 'gain_lo': lo, 'gain_hi': hi})
    return pd.DataFrame(out)


def ratio_axis(ax, lo, hi):
    """Paper-style ratio axis: log scale, 1x/2x/5x ticks, red dashed 1x line."""
    import matplotlib as mpl
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    majors = [m for m in (1, 1.5, 2, 3, 5, 8) if lo <= m <= hi]
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(majors))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
    ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)


def kernel_separators(ax, n):
    for i in range(n - 1):
        ax.axvline(i + 0.5, color='#c8c8c8', linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(False, axis='x', which='both')
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_axisbelow(True)


def figure(results):
    names = KERNEL_ORDER
    x = np.arange(len(names))
    ylims = {'original': (0.88, 3.2), 'scrambled': (0.88, 9.0)}
    fig = plt.figure(figsize=(PAGE_W, 4.6))
    outer = GridSpec(2, 2, figure=fig, left=0.075, right=0.965, top=0.9,
                     bottom=0.11, wspace=0.012, hspace=0.12)
    axes = {}
    for r, ds in enumerate(('original', 'scrambled')):
        for c, pt in enumerate(('SYMMETRIC', 'ROW')):
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=0.07)
            top = fig.add_subplot(inner[0])
            bot = fig.add_subplot(inner[1], sharex=top)
            axes[(r, c)] = (top, bot)
            res = results[(ds, pt)].set_index('kernel').reindex(names)

            g = res['gain_improved'].values
            top.errorbar(x, g, yerr=[g - res['gain_lo'].values, res['gain_hi'].values - g],
                         fmt='_', ms=11, mew=1.6, color='#222222', ecolor='#222222',
                         elinewidth=0.8, capsize=2.2, zorder=4)
            ratio_axis(top, *ylims[ds])
            kernel_separators(top, len(names))
            top.tick_params(axis='x', labelbottom=False)

            imp, slow = 100 * res['improved'].values, 100 * res['slower'].values
            kw = dict(width=0.6, edgecolor='#222222', linewidth=0.5, alpha=0.85, zorder=3)
            bot.bar(x, imp, color=GOOD, **kw)
            bot.bar(x, slow, bottom=imp, color=BAD, **kw)
            bot.bar(x, 100 - imp - slow, bottom=imp + slow, color=NEUTRAL, **kw)
            bot.set_ylim(0, 100)
            bot.set_yticks([0, 50, 100])
            bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
            bot.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
            kernel_separators(bot, len(names))
            bot.set_xticks(x)
            if r == 1:
                bot.set_xticklabels(names, rotation=25, ha='right', rotation_mode='anchor')
            else:
                bot.tick_params(axis='x', labelbottom=False)
            if c == 1:
                top.tick_params(axis='y', which='both', labelleft=False)
                bot.tick_params(axis='y', labelleft=False)
            else:
                top.set_ylabel('Speedup\nwhen improved', fontsize=7.5)
                bot.set_ylabel('Matrices', fontsize=7.5)
    for c, pt in enumerate(('SYMMETRIC', 'ROW')):
        axes[(0, c)][0].set_title(f"{TITLES[pt].capitalize()}", fontsize=8,
                                  fontweight='bold', pad=3)
    for r, ds in enumerate(('original', 'scrambled')):
        y0 = axes[(r, 1)][1].get_position().y0
        y1 = axes[(r, 1)][0].get_position().y1
        x1 = axes[(r, 1)][0].get_position().x1
        fig.text(x1 + 0.004, (y0 + y1) / 2, TITLES[ds], rotation=270, ha='left',
                 va='center', fontsize=8, fontweight='bold')
    handles = [Patch(facecolor=col, edgecolor='#222222', linewidth=0.5, alpha=0.85, label=lab)
               for col, lab in ((GOOD, 'Faster'), (BAD, 'Slower'),
                                (NEUTRAL, 'Kept original (no denser candidate)'))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.52, 0.935),
               ncol=3, frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.2, handletextpad=0.35)
    save(fig, 'improvement_bars_nc32')
    plt.close(fig)


def main():
    style()
    rng = np.random.default_rng(0)
    results, tables = {}, []
    for ds, pt in CELLS:
        r = summarise(densest_speedups(ds, pt), rng)
        results[(ds, pt)] = r
        tables.append(r.assign(dataset=ds, perm_type=pt))
        print(f'== {ds} / {pt}')
        print(r.round(3).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    pd.concat(tables).round(4).to_csv(OUT / 'improvement_bars_nc32.csv', index=False)
    figure(results)


if __name__ == '__main__':
    main()
