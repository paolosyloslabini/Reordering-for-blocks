"""Performance profiles of simple reordering strategies.

Figure: one plot, three strategies per kernel (colour = kernel, line style =
strategy): highest block density; highest block density only if the starting
block density is below 10%; never reorder. The summary CSV covers every strategy.

For every (matrix, kernel) on the original SuiteSparse matrices, the best
available speedup is the maximum over the original ordering (1x) and every
reordering that ran (symmetric and row). A strategy's ratio is
best / achieved >= 1; its profile at x is the share of matrices with ratio <= x.

Strategies
  Highest block density   the candidate with the highest 16x16 block density
                          (original ordering if no reordering raises it)
  ... (d < 10%)           the same, only if the starting block density is
                          below 10%, otherwise the original ordering
  RCM / AMD / Rabbit      always that reordering, symmetric
  ... (d < 10%)           that reordering only if the starting block density
                          is below 10%, otherwise the original ordering
  Never reorder           the original ordering (speedup 1), as a reference
If a strategy's run is missing (reordering or kernel failed), it keeps the
original ordering (speedup 1).

Dense operand: --n-cols=32 or --n-cols=256 (default); at 256, SMaT and ASpT use
their real 32-column runs. Outputs carry an _nc<N> suffix.
Impossible cuSPARSE-BSR / SMaT runs are dropped, as in decision_test.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (kernel_label, kernel_width, n_cols_from_argv, FIXED_32_COLS, INK, INK2, KERNEL_NAMES, OUT, PAGE_W, PALETTE,
                    clean_axes, load_pipeline, save, style)
from decision_test import PEAK_TFLOPS

THRESHOLD = 0.10
N_COLS = n_cols_from_argv()
STRATEGIES = [  # (label, colour, linestyle)
    ('Highest block density', PALETTE[0], '-'),
    (f'Highest block density (d < {THRESHOLD:.0%})', PALETTE[0], '--'),
    ('RCM', PALETTE[1], '-'), (f'RCM (d < {THRESHOLD:.0%})', PALETTE[1], '--'),
    ('AMD', PALETTE[2], '-'), (f'AMD (d < {THRESHOLD:.0%})', PALETTE[2], '--'),
    ('Rabbit', PALETTE[3], '-'), (f'Rabbit (d < {THRESHOLD:.0%})', PALETTE[3], '--'),
    ('Never reorder', INK2, ':'),
]


def data():
    df = pd.concat([load_pipeline('original', pt) for pt in ('SYMMETRIC', 'ROW')],
                   ignore_index=True)
    width = kernel_width(df['kernel_id'], N_COLS)
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
        dense = densest['speedup'] if densest['block_density_16'] > d0 else 1.0
        out = {'Highest block density': dense,
               f'Highest block density (d < {THRESHOLD:.0%})': dense if d0 < THRESHOLD else 1.0}
        for s in ('RCM', 'AMD', 'Rabbit'):
            out[s] = sym.get(s, 1.0)
            out[f'{s} (d < {THRESHOLD:.0%})'] = sym.get(s, 1.0) if d0 < THRESHOLD else 1.0
        out['Never reorder'] = 1.0
        rows.append({'kernel': KERNEL_NAMES[k], 'matrix': m, 'best': best, 'd0': d0,
                     'densest_speedup': dense,
                     **{s: best / v for s, v in out.items()}})
    return pd.DataFrame(rows)


FIGURE_STRATEGIES = [('Highest block density', '-'),
                     (f'Highest block density (d < {THRESHOLD:.0%})', '--'),
                     ('Never reorder', ':')]


def figure(R):
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.72, 3.0))
    xs = np.geomspace(1, 4, 300)
    for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE):
        t = R[R['kernel'] == name]
        for label, ls in FIGURE_STRATEGIES:
            ratio = t[label].values
            ax.plot(xs, [(ratio <= x * (1 + 1e-9)).mean() for x in xs],
                    color=col, ls=ls, lw=1.4)
    ax.set_xscale('log', base=2)
    ax.set_xlim(1, 4)
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.set_xlabel('Within a factor of the best available speedup')
    ax.set_ylabel('Share of matrices')
    clean_axes(ax)
    kernel_handles = [Line2D([], [], color=col, lw=1.8,
                             label=kernel_label(k, N_COLS))
                      for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE)]
    style_handles = [Line2D([], [], color=INK2, lw=1.4, ls=ls,
                            label={'Never reorder': 'Never reorder'}.get(lab, lab.replace('Highest block density', 'Densest')))
                     for lab, ls in FIGURE_STRATEGIES]
    leg1 = ax.legend(handles=kernel_handles, title='Kernel', loc='upper left',
                     bbox_to_anchor=(1.02, 1.0), frameon=False, handlelength=1.8,
                     alignment='left')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, title='Strategy', loc='lower left',
              bbox_to_anchor=(1.02, 0.0), frameon=False, handlelength=2.4,
              alignment='left')
    fig.tight_layout()
    save(fig, f'strategy_profiles_nc{N_COLS}')
    plt.close(fig)


def figure_vs_start(R, bandwidth=0.15, min_eff_n=20):
    """Speedup achieved by the densest candidate vs starting block density."""
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.72, 2.9))
    lo, hi = np.quantile(R['d0'], [0.03, 0.97])
    grid = np.geomspace(lo, hi, 120)
    lg = np.log10(grid)
    table = {}
    for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE):
        t = R[R['kernel'] == name]
        x = np.log10(t['d0'].values)
        y = np.log2(t['densest_speedup'].values)
        geo = np.full(len(grid), np.nan)
        for i, c in enumerate(lg):
            w = np.exp(-0.5 * ((x - c) / bandwidth) ** 2)
            if w.sum() ** 2 / (w ** 2).sum() >= min_eff_n:
                geo[i] = 2 ** ((w * y).sum() / w.sum())
        table[name] = {f'{d:g}': np.interp(np.log10(d), lg, geo) for d in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)}
        label = kernel_label(k, N_COLS)
        ax.plot(grid, geo, color=col, lw=1.6, label=label)
    T = pd.DataFrame(table).T.round(3)
    print('densest speedup by starting block density'); print(T.to_string())
    T.to_csv(OUT / f'densest_speedup_vs_start_density_nc{N_COLS}.csv')
    ax.axhline(1, color=INK2, lw=0.8)
    ax.axvline(THRESHOLD, color=INK2, lw=0.8, ls='--')
    ax.text(THRESHOLD * 1.06, 0.97, 'cut-off: above it,\nkeep the original (1×)',
            transform=ax.get_xaxis_transform(), va='top', fontsize=6.5, color=INK2)
    ax.set_xscale('log')
    ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.set_xlabel(r'Block density before reordering ($16{\times}16$)')
    ax.set_ylabel('Speedup of the densest candidate\n(geometric mean)')
    clean_axes(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              handlelength=1.6)
    fig.tight_layout()
    save(fig, f'strategy_speedup_vs_start_density_nc{N_COLS}')
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
    S.round(3).to_csv(OUT / f'strategy_profiles_summary_nc{N_COLS}.csv', index=False)
    figure(R)
    figure_vs_start(R)


if __name__ == '__main__':
    main()
