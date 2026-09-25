"""Block-density gain of the four strongest reorderings, conditioned on the
starting state of the matrix (original SuiteSparse matrices only).

  gain_vs_start_block_density  x = 16x16 block density before reordering
  gain_vs_start_density        x = nnz / (rows * cols), which no permutation changes

Lines are kernel-smoothed medians, bands the interquartile range.
Run from anywhere:  python analysis/density_response/gain_figures.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from common import COL_W, kernel_quantiles, load_pipeline, save, style
from settings import PERMS

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


def figure(g, xcol, xlabel, name, pct=False):
    """Paper style: one column, symmetric over row, square-patch legend."""
    colors = {v['display']: v['color'] for v in PERMS.values()}
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.0), sharex=True, sharey=True)
    for ax, (pt, title) in zip(axes, PANELS):
        sub = g[g['perm_type'] == pt]
        lo, hi = np.quantile(sub[xcol], [0.03, 0.97])
        grid = np.geomspace(lo, hi, 80)
        for s in TOP:
            t = sub[sub['strategy'] == s]
            q = kernel_quantiles(t[xcol].values, np.log2(t['gain'].values), grid,
                                 bandwidth=0.15)
            ok = ~np.isnan(q[:, 1])
            ax.fill_between(grid[ok], 2 ** q[ok, 0], 2 ** q[ok, 2], color=colors[s],
                            alpha=0.15, lw=0)
            ax.plot(grid[ok], 2 ** q[ok, 1], color=colors[s], lw=1.1, zorder=3)
        ax.axhline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log', base=2)
        ax.grid(True, which='major', color='#b0b0b0', linewidth=0.6)
        ax.grid(True, axis='x', which='minor', color='#e4e4e4', linewidth=0.4)
        ax.set_axisbelow(True)
        ax.text(0.985, 0.95, title, transform=ax.transAxes, ha='right', va='top',
                fontsize=8, fontweight='bold', zorder=6,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.85))
    axes[0].yaxis.set_major_locator(mpl.ticker.FixedLocator([0.25, 0.5, 1, 2, 4]))
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    axes[0].yaxis.set_minor_locator(mpl.ticker.NullLocator())
    if pct:
        axes[1].xaxis.set_major_locator(mpl.ticker.FixedLocator([0.005, 0.01, 0.02, 0.05,
                                                                 0.1, 0.2]))
        axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v * 100:g}%'))
        axes[1].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes[1].set_xlabel(xlabel)
    fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.12, hspace=0.07)
    mid = (axes[0].get_position().y1 + axes[1].get_position().y0) / 2
    fig.text(0.0, mid, 'Block density gain (after / before)', rotation=90,
             ha='left', va='center', fontsize=8)
    handles = [Patch(facecolor=colors[s], edgecolor='#222222', linewidth=0.5, label=s)
               for s in TOP]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.555, 0.905),
               ncol=4, frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=0.9, handletextpad=0.35)
    save(fig, name)
    plt.close(fig)


def main():
    style()
    g = gains()
    print(g.groupby(['perm_type', 'strategy']).size().unstack())
    figure(g, 'start_block_density', r'Block density before reordering ($16{\times}16$)',
           'gain_vs_start_block_density', pct=True)
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
