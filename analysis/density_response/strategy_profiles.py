"""Performance profiles of simple reordering strategies, per kernel.

For every (matrix, kernel) on the original SuiteSparse matrices, the best
available speedup is the maximum over the original ordering (1x) and every
reordering that ran (symmetric and row). A strategy's ratio is
best / achieved >= 1; its profile at x is the share of matrices with ratio <= x.

Strategies
  Highest block density   the candidate with the highest 16x16 block density
                          (original ordering if no reordering raises it)
  RCM / AMD / Rabbit      always that reordering, symmetric
  ... (d < 10%)           that reordering only if the starting block density
                          is below 10%, otherwise the original ordering
If a strategy's run is missing (reordering or kernel failed), it keeps the
original ordering (speedup 1).

Dense operand: 256 columns, except SMaT and ASpT (their runs always used 32).
Impossible cuSPARSE-BSR / SMaT runs are dropped, as in decision_test.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIXED_32_COLS, INK, INK2, KERNEL_NAMES, OUT, PAGE_W, PALETTE,
                    clean_axes, load_pipeline, save, style)
from decision_test import PEAK_TFLOPS

THRESHOLD = 0.10
STRATEGIES = [  # (label, colour, linestyle)
    ('Highest block density', PALETTE[0], '-'),
    ('RCM', PALETTE[1], '-'), (f'RCM (d < {THRESHOLD:.0%})', PALETTE[1], '--'),
    ('AMD', PALETTE[2], '-'), (f'AMD (d < {THRESHOLD:.0%})', PALETTE[2], '--'),
    ('Rabbit', PALETTE[3], '-'), (f'Rabbit (d < {THRESHOLD:.0%})', PALETTE[3], '--'),
]


def data():
    df = pd.concat([load_pipeline('original', pt) for pt in ('SYMMETRIC', 'ROW')],
                   ignore_index=True)
    width = np.where(df['kernel_id'].isin(FIXED_32_COLS), 32, 256)
    df = df[(df['n_cols'] == width) & (df['strategy'] != 'Original')
            & (df['speedup'] > 0) & np.isfinite(df['speedup'])
            & (df['block_density_16'] > 0) & (df['block_density_16_original'] > 0)]
    for k, peak in PEAK_TFLOPS.items():
        executed = (2 * df['nonzero_blocks_32'] * 1024 * df['n_cols']
                    / (df['time_operation_ms'] / 1e3) / 1e12)
        df = df[~((df['kernel_id'] == k) & (executed > peak))]
    return df


def outcomes(df):
    rows = []
    for (k, m), g in df.groupby(['kernel_id', 'matrix']):
        d0 = g['block_density_16_original'].iloc[0]
        best = max(g['speedup'].max(), 1.0)
        densest = g.loc[g['block_density_16'].idxmax()]
        sym = g[g['perm_type'] == 'SYMMETRIC'].set_index('strategy')['speedup']
        out = {'Highest block density':
               densest['speedup'] if densest['block_density_16'] > d0 else 1.0}
        for s in ('RCM', 'AMD', 'Rabbit'):
            out[s] = sym.get(s, 1.0)
            out[f'{s} (d < {THRESHOLD:.0%})'] = sym.get(s, 1.0) if d0 < THRESHOLD else 1.0
        rows.append({'kernel': KERNEL_NAMES[k], 'matrix': m, 'best': best,
                     **{s: best / v for s, v in out.items()}})
    return pd.DataFrame(rows)


def figure(R):
    kernels = list(KERNEL_NAMES)
    fig, axes = plt.subplots(2, 4, figsize=(PAGE_W, 3.8), sharey=True)
    for ax, k in zip(axes.flat, kernels):
        name = KERNEL_NAMES[k]
        t = R[R['kernel'] == name]
        xmax = 4 if k in ('CUSPARSE_SPMM_BSR_bs32', 'SMAT_SPMM_bs32') else 2
        xs = np.geomspace(1, xmax, 200)
        for label, col, ls in STRATEGIES:
            ratio = t[label].values
            ax.plot(xs, [(ratio <= x * (1 + 1e-9)).mean() for x in xs],
                    color=col, ls=ls, lw=1.4, label=label)
        title = name + (' (32 cols)' if k in FIXED_32_COLS else '')
        ax.set_title(f'{title}  (n={len(t)})', color=INK, fontsize=7.5)
        ax.set_xscale('log', base=2)
        ax.set_xlim(1, xmax)
        ax.set_xlabel('Within a factor of the best')
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
        clean_axes(ax)
    legend_ax = axes.flat[-1]
    legend_ax.axis('off')
    h, l = axes.flat[0].get_legend_handles_labels()
    legend_ax.legend(h, l, loc='center', frameon=False, handlelength=2.2, fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel('Share of matrices')
    fig.tight_layout()
    save(fig, 'strategy_profiles')
    plt.close(fig)


def main():
    style()
    R = outcomes(data())
    labels = [s for s, _, _ in STRATEGIES]
    summary = []
    for name, t in R.groupby('kernel'):
        for s in labels:
            summary.append({'kernel': name, 'strategy': s,
                            'is_best': np.mean(t[s] <= 1 + 1e-9),
                            'within_10pct': np.mean(t[s] <= 1.1),
                            'within_25pct': np.mean(t[s] <= 1.25),
                            'geo_speedup': np.exp(np.mean(np.log(t['best'] / t[s])))})
    S = pd.DataFrame(summary)
    print(S.pivot(index='kernel', columns='strategy', values='within_10pct')
          .reindex(list(KERNEL_NAMES.values()))[labels].round(2).to_string())
    print(S.pivot(index='kernel', columns='strategy', values='geo_speedup')
          .reindex(list(KERNEL_NAMES.values()))[labels].round(3).to_string())
    OUT.mkdir(exist_ok=True)
    S.round(3).to_csv(OUT / 'strategy_profiles_summary.csv', index=False)
    figure(R)


if __name__ == '__main__':
    main()
