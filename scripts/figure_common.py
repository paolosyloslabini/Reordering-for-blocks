"""Shared loading and styling for gain_figures.py and improvement_bars.py.

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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'plots' / 'paper'
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
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
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


def n_cols_from_argv(default=256):
    """Dense-operand width from ``--n-cols=N`` on the command line."""
    for arg in sys.argv[1:]:
        if arg.startswith('--n-cols='):
            return int(arg.split('=', 1)[1])
    return default


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


def save(fig, name, tight=False):
    """tight: crop the page to the drawn content (no blank band above a
    legend), as the paper places the PDF at its natural size."""
    OUT.mkdir(parents=True, exist_ok=True)
    kw = dict(bbox_inches='tight', pad_inches=0.01) if tight else {}
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, **kw)
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
