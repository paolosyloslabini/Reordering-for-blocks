"""Decision test: how much speedup do simple rules for choosing a reordering achieve?

For every (matrix, dataset, permutation type, kernel), the candidates are the
original ordering (speedup 1) and every reordering that ran. Each policy picks
one candidate; we report the geometric mean speedup it achieves and how often
it captures at least 90% of the best available speedup.

Policies
  best            the fastest candidate (upper bound; see the caveat in README)
  max_density     the candidate with the highest 16x16 block density
                  (keeps the original ordering if no reordering raises it)
  density_reuse   the highest predicted speedup from a held-out model:
                  log2 speedup = f(log2 d_after) - f(log2 d_before) + g * log2(reuse ratio),
                  f a natural spline, fitted per kernel with 5-fold CV grouped by matrix
  always_RCM, always_AMD
  never           keep the original ordering

A "highest predicted speedup from the density curve" policy is not listed: the
fitted curves increase monotonically, so it always picks the same candidate as
max_density.

Dense operand: 256 columns, except SMaT and ASpT (their runs always used 32).
Runs whose padded 32x32 work would exceed the hardware peak are dropped for
cuSPARSE-BSR and SMaT.
"""
import numpy as np
import pandas as pd

from common import FIXED_32_COLS, KERNEL_NAMES, OUT, load_pipeline

N_COLS = 256
PEAK_TFLOPS = {'CUSPARSE_SPMM_BSR_bs32': 19.5, 'SMAT_SPMM_bs32': 312.0}
POLICIES = ['best', 'max_density', 'density_reuse', 'always_RCM', 'always_AMD', 'never']


def data():
    parts = [load_pipeline(ds, pt) for ds in ('original', 'scrambled')
             for pt in ('SYMMETRIC', 'ROW')]
    df = pd.concat(parts, ignore_index=True)
    width = np.where(df['kernel_id'].isin(FIXED_32_COLS), 32, N_COLS)
    df = df[(df['n_cols'] == width) & (df['strategy'] != 'Original')]
    for k, peak in PEAK_TFLOPS.items():
        executed = (2 * df['nonzero_blocks_32'] * 1024 * df['n_cols']
                    / (df['time_operation_ms'] / 1e3) / 1e12)
        df = df[~((df['kernel_id'] == k) & (executed > peak))]
    cols = ['block_density_16', 'block_density_16_original', 'speedup',
            'access_dist_reuse_distance_mean', 'access_dist_reuse_distance_mean_original']
    df = df[(df[cols] > 0).all(axis=1) & np.isfinite(df['speedup'])].copy()
    df['x1'] = np.log2(df['block_density_16'])
    df['x0'] = np.log2(df['block_density_16_original'])
    df['reuse'] = np.log2(df['access_dist_reuse_distance_mean_original']
                          / df['access_dist_reuse_distance_mean'])
    df['y'] = np.log2(df['speedup'])
    return df.reset_index(drop=True)


def ns_basis(x, knots):
    k_last = knots[-1]

    def d(k):
        return ((np.clip(x - knots[k], 0, None) ** 3
                 - np.clip(x - k_last, 0, None) ** 3) / (k_last - knots[k]))
    return np.column_stack([x] + [d(k) - d(len(knots) - 2)
                                  for k in range(len(knots) - 2)])


def heldout_predictions(t, rng):
    """5-fold CV (grouped by matrix) predictions of log2 speedup."""
    knots = np.quantile(np.r_[t['x0'], t['x1']], [.05, .275, .5, .725, .95])
    X = np.column_stack([ns_basis(t['x1'].values, knots) - ns_basis(t['x0'].values, knots),
                         t['reuse'].values])
    mats = t['matrix'].unique()
    fold = dict(zip(rng.permutation(mats), np.arange(len(mats)) % 5))
    f = t['matrix'].map(fold).values
    pred = np.zeros(len(t))
    for i in range(5):
        tr = f != i
        beta = np.linalg.lstsq(X[tr], t['y'].values[tr], rcond=None)[0]
        pred[~tr] = X[~tr] @ beta
    return pred


def decisions(df, rng):
    rows = []
    for k, name in KERNEL_NAMES.items():
        t = df[df['kernel_id'] == k].reset_index(drop=True)
        t['pred'] = heldout_predictions(t, rng)
        for (m, ds, pt), g in t.groupby(['matrix', 'dataset', 'perm_type']):
            if len(g) < 5:
                continue
            y = g['y']

            def pick(score, threshold):
                return 0.0 if score.max() <= threshold else y.loc[score.idxmax()]

            def fixed(s):
                hit = g.index[g['strategy'] == s]
                return y.loc[hit[0]] if len(hit) else np.nan

            rows.append(dict(
                kernel=name, dataset=ds, perm_type=pt,
                best=max(y.max(), 0.0),
                max_density=pick(g['x1'], g['x0'].iloc[0]),
                density_reuse=pick(g['pred'], 0.0),
                always_RCM=fixed('RCM'), always_AMD=fixed('AMD'), never=0.0))
    return pd.DataFrame(rows)


def main():
    df = data()
    D = decisions(df, np.random.default_rng(0))
    kernels = list(KERNEL_NAMES.values())
    geo = (2 ** D.groupby(['dataset', 'kernel'])[POLICIES].mean())
    near = D.groupby(['dataset', 'kernel']).apply(
        lambda g: pd.Series({p: np.mean(g[p] >= g['best'] - np.log2(1 / 0.9))
                             for p in POLICIES[1:]}))
    n = D.groupby(['dataset', 'kernel']).size().rename('cases')
    for ds in ('original', 'scrambled'):
        print(f'== {ds}: geometric-mean speedup per policy')
        print(geo.loc[ds].reindex(kernels).round(3).to_string())
        print(f'== {ds}: share of cases within 90% of the best')
        print(near.loc[ds].reindex(kernels).round(2).join(n.loc[ds]).to_string())
    OUT.mkdir(exist_ok=True)
    table = geo.round(3).add_prefix('speedup_').join(near.round(3).add_prefix('within90_')).join(n)
    table.to_csv(OUT / 'decision_table.csv')
    print('saved', OUT / 'decision_table.csv')


if __name__ == '__main__':
    main()
