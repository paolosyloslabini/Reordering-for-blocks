"""
Within-matrix r and elasticity alpha of block density improvement vs speedup,
for symmetric, row and pooled (both) reorderings of the original matrices.

Same fits as the paper's correlation figures (paper_figures.within_fit):
bsr_sanity (cuSPARSE-BSR, 32x32), improvement_vs_speedup (16x16, n_cols=256)
and corr_blocksize_ncols (block sizes, n_cols). Writes one long CSV.

Usage:  python scripts/correlation_fits_table.py [--out plots/paper_fits.csv]
"""
import argparse
from pathlib import Path

import pandas as pd

import paper_figures as pf
from settings import BLOCK_SIZES

PERM_TYPES = ['SYMMETRIC', 'ROW', 'BOTH']


def fits(df, perm_type):
    rows = []
    kernels = pf._ordered_kernels(df, pf.KERNEL_NAMES)
    for nc in sorted(df['n_cols'].unique()):
        d = df[(df['n_cols'] == nc) & (df['strategy'] != 'Original')]
        for k in kernels:
            dk = d[d['kernel_id'] == k]
            for bs in BLOCK_SIZES:
                col = f'density_improvement_{bs}'
                if col not in dk.columns:
                    continue
                r, alpha, n = pf.within_fit(dk, col)
                rows.append(dict(perm_type=perm_type, n_cols=int(nc),
                                 kernel=pf.PAPER_KERNEL_NAMES.get(k, k),
                                 block=bs, r=r, alpha=alpha, n_matrices=n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='plots/paper_fits.csv')
    args = ap.parse_args()
    rows = []
    for pt in PERM_TYPES:
        rows += fits(pf.correlation_ops(pt), pt)
    t = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)

    pd.set_option('display.width', 200)
    for title, sel in [('alpha, 16x16, n_cols=256', (t.block == 16) & (t.n_cols == 256)),
                       ('alpha, cuSPARSE-BSR 32x32, n_cols=256',
                        (t.block == 32) & (t.n_cols == 256)
                        & t.kernel.str.contains('BSR'))]:
        print(f'\n== {title}')
        print(t[sel].pivot(index='kernel', columns='perm_type',
                           values=['alpha', 'r', 'n_matrices'])
              .reindex(columns=PERM_TYPES, level=1).round(3))
    print('\n== alpha by n_cols, 16x16')
    s = t[t.block == 16]
    print(s.pivot_table(index='kernel', columns=['perm_type', 'n_cols'],
                        values='alpha').reindex(columns=PERM_TYPES, level=0)
          .round(2))
    print(f'\nSaved {args.out}')


if __name__ == '__main__':
    main()
