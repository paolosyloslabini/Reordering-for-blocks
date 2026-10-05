#!/usr/bin/env python3
"""
Sanity check: SpMM time must grow with the number of dense columns.

For every kernel, pairs runs of the same (matrix, perm, perm_type) at different
n_cols and reports the median ratio t(n_cols) / t(smallest n_cols). A kernel
whose time stays flat while the work grows k-fold did not compute all columns
(this is how the ASpT sspmm_32 and SMaT n_mult=1 bugs showed up: ratio ~1.0).

Usage:
    python scripts/check_ncols_scaling.py [results/results_operations.csv ...] [--min-ratio 1.5]

Exits with status 1 if any kernel's median ratio at the widest n_cols is below --min-ratio.
"""
import argparse
import re
import sys

import pandas as pd

KEYS = ['matrix', 'perm', 'perm_type']


def kernel_of(tag):
    """ASPT_SPMM_SYMMETRIC_RANDOM -> ASPT_SPMM (drop the pipeline suffix)."""
    return re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)(_RANDOM)?$', '', tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', nargs='*', default=['results/results_operations.csv'])
    parser.add_argument('--min-ratio', type=float, default=1.5,
                        help='Minimum median t(widest)/t(narrowest) to pass (default: 1.5)')
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    df = df.dropna(subset=['time_operation_ms'])
    df = df[df['time_operation_ms'] > 0]
    df['kernel'] = df['tag'].map(kernel_of)
    df['pipeline'] = df['tag']  # keep random and original runs apart
    # Average repeated measurements of the same configuration
    df = df.groupby(['kernel', 'pipeline'] + KEYS + ['n_cols'], as_index=False)['time_operation_ms'].median()

    failed = []
    print(f"{'kernel':<24} {'n_cols':>14} {'pairs':>7} {'median ratio':>13}")
    for kernel, g in df.groupby('kernel'):
        widths = sorted(g['n_cols'].unique())
        if len(widths) < 2:
            print(f"{kernel:<24} {'only ' + str(widths[0]):>14}")
            continue
        wide = g.pivot_table(index=['pipeline'] + KEYS, columns='n_cols', values='time_operation_ms')
        base = widths[0]
        for w in widths[1:]:
            ratio = (wide[w] / wide[base]).dropna()
            if ratio.empty:
                continue
            med = ratio.median()
            flag = ''
            if w == widths[-1] and med < args.min_ratio:
                flag = '  <-- FLAT'
                failed.append(kernel)
            print(f"{kernel:<24} {f'{w}/{base}':>14} {len(ratio):>7} {med:>13.2f}{flag}")

    if failed:
        print(f"\nTime does not grow with n_cols for: {', '.join(failed)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
