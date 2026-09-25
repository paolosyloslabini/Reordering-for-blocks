"""Shared loading and styling for the density-response figures.

Data come from the committed results CSVs through ``plot_utils``, with the same
matrix filters as the paper (``scripts/filter_config.yaml``).
"""
import contextlib
import io
import os
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures'
sys.path.insert(0, str(ROOT / 'scripts'))
os.chdir(ROOT)
import plot_utils as pu  # noqa: E402

# IEEEtran widths (inches)
COL_W, PAGE_W = 3.49, 7.16

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']

RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.linewidth': 0.6, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
}

KERNEL_NAMES = {
    'CUSPARSE_SPMM_BSR_bs32': 'cuSPARSE-BSR',
    'SMAT_SPMM_bs32': 'SMaT',
    'DTC_SPMM': 'DTC-SpMM',
    'FLASHSPARSE_SPMM': 'FlashSparse',
    'ACCSPMM_SPMM': 'Acc-SpMM',
    'CUSPARSE_SPMM_CSR': 'cuSPARSE-CSR',
    'ASPT_SPMM': 'ASpT',
}
# SMaT and ASpT ignore the n_cols argument: every run used 32 columns.
FIXED_32_COLS = {'SMAT_SPMM_bs32', 'ASPT_SPMM'}


def n_cols_from_argv(default=256):
    """Dense-operand width from ``--n-cols=N`` on the command line (32 or 256)."""
    for arg in sys.argv[1:]:
        if arg.startswith('--n-cols='):
            return int(arg.split('=', 1)[1])
    return default


def kernel_width(kernel_ids, n_cols):
    """Width to select per row: SMaT/ASpT always use their real 32-column runs."""
    return np.where(pd.Series(kernel_ids).isin(FIXED_32_COLS).values, 32, n_cols)


def kernel_label(kernel_id, n_cols):
    name = KERNEL_NAMES[kernel_id]
    return name + (' (32 cols)' if kernel_id in FIXED_32_COLS and n_cols != 32 else '')


def style():
    mpl.rcParams.update(RC)


def clean_axes(ax):
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def load_pipeline(dataset, perm_type):
    """Operations joined with structure metrics for one pipeline, as in the paper."""
    random = dataset == 'scrambled'
    with contextlib.redirect_stdout(io.StringIO()):
        df, da, cfg = pu.load_and_filter_data(cli_overrides={'random': random})
        key = ('random_' if random else '') + perm_type.lower()
        df, da = pu.split_by_perm_type(df, da, perm_type,
                                       cfg.get('pipelines', {}).get(key, {}))
        df = pu.prepare_full_dataframe(df)
    df = df.copy()
    df['strategy'] = df['strategy'].replace({'Metis': 'METIS'})
    df['dataset'] = dataset
    df['perm_type'] = perm_type
    return df


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300)
    print('saved', OUT / name)


def kernel_quantiles(x, y, grid, qs=(0.25, 0.5, 0.75), bandwidth=0.12,
                     min_eff_n=25):
    """Gaussian-kernel-weighted quantiles of y along log10(x).

    Returns an array (len(grid), len(qs)); NaN where the effective sample size
    (Kish) of the window is below ``min_eff_n``.
    """
    lx = np.log10(x)
    order = np.argsort(y)
    ys, lxs = y[order], lx[order]
    out = np.full((len(grid), len(qs)), np.nan)
    for i, g in enumerate(np.log10(grid)):
        w = np.exp(-0.5 * ((lxs - g) / bandwidth) ** 2)
        if w.sum() ** 2 / (w ** 2).sum() < min_eff_n:
            continue
        cw = np.cumsum(w) / w.sum()
        out[i] = np.interp(qs, cw, ys)
    return out
