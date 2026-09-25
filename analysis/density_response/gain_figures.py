"""Block-density gain of the four strongest reorderings, conditioned on the
starting state of the matrix (original SuiteSparse matrices only).

  gain_vs_start_block_density  x = 16x16 block density before reordering
  gain_vs_start_density        x = nnz / (rows * cols), which no permutation changes

Lines are kernel-smoothed medians, bands the interquartile range.
Run from anywhere:  python analysis/density_response/gain_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (INK, INK2, PAGE_W, PALETTE, clean_axes, kernel_quantiles,
                    load_pipeline, save, style)

TOP = ['RCM', 'AMD', 'Rabbit', 'DTC-LSH']
PANELS = [('SYMMETRIC', 'Symmetric reordering'), ('ROW', 'Row reordering')]


def gains():
    rows = []
    for pt, _ in PANELS:
        df = load_pipeline('original', pt)
        d = df[df['strategy'].isin(TOP)].drop_duplicates(['matrix', 'strategy'])
        d = d[(d['block_density_16'] > 0) & (d['block_density_16_original'] > 0)]
        rows.append(pd.DataFrame({
            'perm_type': pt,
            'matrix': d['matrix'].values,
            'strategy': d['strategy'].values,
            'start_block_density': d['block_density_16_original'].values,
            'start_density': d['density'].values,
            'gain': (d['block_density_16'] / d['block_density_16_original']).values,
        }))
    return pd.concat(rows, ignore_index=True)


def figure(g, xcol, xlabel, name):
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, 2.5), sharey=True)
    for ax, (pt, title) in zip(axes, PANELS):
        sub = g[g['perm_type'] == pt]
        lo, hi = np.quantile(sub[xcol], [0.03, 0.97])
        grid = np.geomspace(lo, hi, 80)
        for s, col in zip(TOP, PALETTE):
            t = sub[sub['strategy'] == s]
            q = kernel_quantiles(t[xcol].values, np.log2(t['gain'].values), grid,
                                 bandwidth=0.15)
            ok = ~np.isnan(q[:, 1])
            ax.fill_between(grid[ok], 2 ** q[ok, 0], 2 ** q[ok, 2], color=col,
                            alpha=0.14, lw=0)
            ax.plot(grid[ok], 2 ** q[ok, 1], color=col, lw=1.8, label=s)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log', base=2)
        ax.set_title(title, color=INK)
        ax.set_xlabel(xlabel)
        clean_axes(ax)
    axes[0].set_ylabel('Block density gain (after / before)')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.07), handlelength=1.6)
    fig.tight_layout()
    save(fig, name)
    plt.close(fig)


def main():
    style()
    g = gains()
    print(g.groupby(['perm_type', 'strategy']).size().unstack())
    figure(g, 'start_block_density', r'Block density before reordering ($16{\times}16$)',
           'gain_vs_start_block_density')
    figure(g, 'start_density', r'Matrix density $\mathrm{nnz}/(m \cdot n)$',
           'gain_vs_start_density')
    # Held-out check: how well does each x predict log gain? (5-fold by matrix)
    rng = np.random.default_rng(0)
    for xcol in ('start_block_density', 'start_density'):
        res = []
        for (pt, s), t in g.groupby(['perm_type', 'strategy']):
            mats = t['matrix'].unique()
            fold = dict(zip(rng.permutation(mats), np.arange(len(mats)) % 5))
            f = t['matrix'].map(fold).values
            y = np.log2(t['gain'].values)
            pred = np.zeros(len(t))
            for k in range(5):
                tr = f != k
                q = kernel_quantiles(t[xcol].values[tr], y[tr], t[xcol].values[~tr],
                                     qs=(0.5,), min_eff_n=1)
                pred[~tr] = q[:, 0]
            ok = ~np.isnan(pred)
            r2 = 1 - ((y[ok] - pred[ok]) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
            res.append((pt, s, round(r2, 2), round(np.mean(np.sign(pred[ok]) == np.sign(y[ok])), 2)))
        print(xcol, 'held-out (perm_type, reordering, R2, direction right):', res)


if __name__ == '__main__':
    main()
