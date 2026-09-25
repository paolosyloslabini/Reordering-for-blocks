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

from common import (INK, INK2, KERNEL_NAMES, OUT, PAGE_W, clean_axes,
                    load_pipeline, save, style)
from decision_test import PEAK_TFLOPS

N_COLS = 32
GOOD, BAD, NEUTRAL = '#0ca30c', '#d03b3b', '#dcdbd6'
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


def figure(results):
    fig = plt.figure(figsize=(PAGE_W, 5.3))
    outer = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.10,
                     top=0.9, bottom=0.12, left=0.09, right=0.99)
    ymax = max(r['gain_hi'].max() for r in results.values()) * 1.15
    names = list(KERNEL_NAMES.values())
    x = np.arange(len(names))
    for i, (ds, pt) in enumerate(CELLS):
        row, col = divmod(i, 2)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1.25, 1], hspace=0.08)
        top = fig.add_subplot(inner[0])
        bot = fig.add_subplot(inner[1], sharex=top)
        r = results[(ds, pt)].set_index('kernel').reindex(names)

        # top: gain among improved matrices, horizontal tick + 95% interval
        top.errorbar(x, r['gain_improved'],
                     yerr=[r['gain_improved'] - r['gain_lo'], r['gain_hi'] - r['gain_improved']],
                     fmt='_', ms=14, mew=2.0, color=INK, ecolor=INK2, elinewidth=1.0, capsize=2.5)
        top.axhline(1, color=INK2, lw=0.8)
        top.set_yscale('log', base=2)
        top.set_ylim(0.95, ymax)
        top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
        top.set_title(f"{TITLES[ds]}, {TITLES[pt]}", color=INK, fontsize=8)
        top.tick_params(axis='x', labelbottom=False, length=0)
        clean_axes(top)
        top.grid(False, axis='x')

        # bottom: share improved (green, from 0 up) and slower (red, above it)
        imp, slow = 100 * r['improved'].values, 100 * r['slower'].values
        bot.bar(x, imp, 0.62, color=GOOD, edgecolor='white', lw=0.8)
        bot.bar(x, slow, 0.62, bottom=imp, color=BAD, edgecolor='white', lw=0.8)
        bot.bar(x, 100 - imp - slow, 0.62, bottom=imp + slow, color=NEUTRAL,
                edgecolor='white', lw=0.8)
        bot.set_ylim(0, 100)
        bot.set_yticks([0, 50, 100])
        bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        bot.set_xticks(x)
        bot.set_xticklabels(names, rotation=30, ha='right', rotation_mode='anchor',
                            fontsize=6.5)
        bot.tick_params(axis='x', length=0, labelbottom=(row == 1))
        clean_axes(bot)
        bot.grid(False, axis='x')
        if col == 0:
            top.set_ylabel('Speedup when\nimproved', fontsize=7)
            bot.set_ylabel('Matrices', fontsize=7)
        else:
            top.tick_params(labelleft=False)
            bot.tick_params(labelleft=False)
    handles = [Patch(color=GOOD, label='Faster (improved)'),
               Patch(color=BAD, label='Slower'),
               Patch(color=NEUTRAL, label='Kept original (no denser candidate)')]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=7)
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
