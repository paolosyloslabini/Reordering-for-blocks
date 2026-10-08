"""
Elasticity alpha of block density (16x16) improvement vs speedup, n_cols=256,
original matrices, symmetric + row reorderings of each matrix together.

Two estimators over the same per-matrix groups (all orderings of a matrix):
  pooled_fit  -- the paper's within_fit: centre logs per matrix, one
                 least-squares slope over all centred points (equivalent to a
                 sum(x^2)-weighted mean of per-matrix slopes)
  per_matrix  -- one least-squares slope per matrix, then median / mean of
                 those slopes across matrices (each matrix weighs the same)

Usage:  python scripts/per_matrix_alpha.py
"""
import numpy as np
import pandas as pd

import paper_figures as pf

XCOL = 'density_improvement_16'


def per_matrix_slopes(dk):
    m, cx, cy = pf._centred(dk, XCOL)
    t = pd.DataFrame({'m': m, 'xy': cx * cy, 'xx': cx * cx, 'n': 1})
    g = t.groupby('m').sum()
    g = g[g['xx'] > 1e-12]
    return g['xy'] / g['xx'], g['n']


def main():
    rows = []
    for pt in ['SYMMETRIC', 'ROW', 'BOTH']:
        df = pf.correlation_ops(pt)
        d = df[(df['n_cols'] == 256) & (df['strategy'] != 'Original')]
        for k in pf._ordered_kernels(d, pf.KERNEL_NAMES):
            dk = d[d['kernel_id'] == k]
            r, a, nm = pf.within_fit(dk, XCOL)
            s, n = per_matrix_slopes(dk)
            rows.append(dict(kernel=pf.PAPER_KERNEL_NAMES.get(k, k),
                             perm_type=pt, pooled_fit=a,
                             per_matrix_median=s.median(),
                             per_matrix_mean=s.mean(),
                             orderings_per_matrix=n.median(),
                             n_matrices=len(s)))
    t = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    for c in ['pooled_fit', 'per_matrix_median', 'per_matrix_mean',
              'orderings_per_matrix']:
        print(f'\n== {c}')
        print(t.pivot(index='kernel', columns='perm_type', values=c)
              [['SYMMETRIC', 'ROW', 'BOTH']].round(2).to_string())
    t.to_csv('plots/per_matrix_alpha.csv', index=False)


if __name__ == '__main__':
    main()
