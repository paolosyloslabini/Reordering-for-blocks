"""
Plot Utilities - Clean Architecture

This module provides:
1. Data loading and processing functions that prepare DataFrames with all derived columns
2. Generic plot functions parameterized by column names
3. Metric configuration for display names and properties
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator
from pathlib import Path
from scipy import stats
import re
import sys
import yaml
from settings import (
    PALETTE, PERMS, ALL_METRICS,
    get_metric_display, get_perm_display, get_perm_color,
    use_log_scale,
)


# =============================================================================
# Professional Plot Style
# =============================================================================

def set_professional_style():
    """Configure matplotlib/seaborn for publication-quality plots.

    Call once at the start of a script. Produces clean, serif-font figures
    suitable for LaTeX papers.
    """
    sns.set_theme(style='whitegrid', font='serif', palette=PALETTE)
    mpl.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 17,
        'axes.labelsize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        # Lines & markers
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        # Axes
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })



def format_log_axes(ax, which='both'):
    """Format log-scale axes with proper ticks and log-paper grid background.

    Adds major ticks at powers of 10, minor ticks at intermediate values
    (2, 3, …, 9 within each decade), and subtle minor gridlines to give the
    characteristic 'log-paper' look.

    Args:
        ax: Matplotlib Axes object.
        which: ``'x'``, ``'y'``, or ``'both'`` — which axes to format.
    """
    targets = []
    if which in ('x', 'both') and ax.get_xscale() == 'log':
        targets.append(ax.xaxis)
    if which in ('y', 'both') and ax.get_yscale() == 'log':
        targets.append(ax.yaxis)

    for axis in targets:
        # Major ticks: powers of 10 AND intermediate values (1, 2, 3, 5)
        axis.set_major_locator(LogLocator(base=10, subs=[1.0, 2.0, 3.0, 5.0], numticks=30))
        # Minor ticks: all other values within each decade
        axis.set_minor_locator(
            LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
        axis.set_minor_formatter(NullFormatter())

    # Grid: prominent major lines, subtle minor lines ("log paper")
    ax.grid(True, which='major', alpha=0.5, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)


def get_display_name(col):
    """Get display name for a column, or format the column name if not configured."""
    return get_metric_display(col)


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_data(ops_path, analysis_path):
    """Load operations and analysis CSVs and merge them.

    Returns:
        Tuple of (merged_df, analysis_df).  merged_df may be empty when the
        operations CSV is missing (analysis-only sections can still run).
    """
    # Analysis CSV is always required
    try:
        df_analysis = pd.read_csv(analysis_path)
    except Exception as e:
        print(f"Error reading analysis CSV: {e}")
        sys.exit(1)

    if df_analysis.empty:
        print("Analysis CSV is empty.")
        sys.exit(1)

    # Operations CSV is optional (e.g. random pipeline may not have one yet)
    try:
        df_op = pd.read_csv(ops_path)
    except FileNotFoundError:
        print(f"Operations CSV not found at {ops_path} — "
              "kernel/breakeven plots will be skipped.")
        df_op = pd.DataFrame()
    except Exception as e:
        print(f"Error reading operations CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df_op)} operation rows and {len(df_analysis)} analysis rows.")

    # Normalize keys
    df_analysis['perm'] = df_analysis['perm'].fillna('None').astype(str)
    df_analysis['perm_type'] = df_analysis['perm_type'].fillna('UNKNOWN').astype(str)
    df_analysis['matrix'] = df_analysis['matrix'].astype(str)

    if df_op.empty:
        return pd.DataFrame(), df_analysis

    df_op['perm'] = df_op['perm'].fillna('None').astype(str)
    df_op['perm_type'] = df_op['perm_type'].fillna('UNKNOWN').astype(str)
    df_op['matrix'] = df_op['matrix'].astype(str)

    # Merge
    df = pd.merge(df_op, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')
    print(f"Merged DataFrame has {len(df)} rows.")

    if df.empty:
        print("Merged DataFrame is empty! Check if keys match in both CSVs.")

    return df, df_analysis


def add_base_metrics(df):
    """Add base derived metrics: gflops, kernel_id, strategy.
    
    Args:
        df: DataFrame with operations data (must have time_operation_ms, nnz, n_cols, algo, perm)
        
    Returns:
        DataFrame with added columns: gflops, kernel_id, strategy
    """
    df = df.copy()
    
    # Ensure n_cols exists
    if 'n_cols' in df.columns:
        df['n_cols'] = pd.to_numeric(df['n_cols'], errors='coerce').fillna(32)
    else:
        df['n_cols'] = 32
    
    # Calculate GFLOPS
    df['gflops'] = (2 * df['nnz'] * df['n_cols']) / (df['time_operation_ms'] * 1e-3) / 1e9
    
    # Create kernel_id by stripping reordering suffixes from algo
    def get_kernel_id(row):
        algo = row['algo']
        algo = re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', algo)
        if pd.notnull(row.get('block_size')) and row['block_size'] > 0:
            return f"{algo}_bs{int(row['block_size'])}"
        return algo

    df['kernel_id'] = df.apply(get_kernel_id, axis=1)
    
    # Strategy label (use display names from settings)
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )
    
    return df


def add_relative_metrics(df):
    """Add relative/normalized metrics.
    
    Adds:
        - rel_bandwidth: bandwidth_max / rows
        - rel_row_spread: locality_avg_row_spread / cols
    """
    df = df.copy()
    
    if 'bandwidth_max' in df.columns and 'rows' in df.columns:
        df['rel_bandwidth'] = df['bandwidth_max'] / df['rows']
    
    if 'locality_avg_row_spread' in df.columns and 'cols' in df.columns:
        df['rel_row_spread'] = df['locality_avg_row_spread'] / df['cols']
    
    return df


def add_speedup(df):
    """Add speedup column relative to Original (perm='None') baseline.
    
    Speedup is calculated per (matrix, kernel_id, n_cols) tuple.
    """
    df = df.copy()
    
    # Get baseline GFLOPS for Original - must include n_cols to avoid mixing different workloads
    original = df[df['strategy'] == 'Original'].groupby(['matrix', 'kernel_id', 'n_cols'])['gflops'].mean()
    original = original.reset_index().rename(columns={'gflops': 'gflops_original'})
    
    df = pd.merge(df, original, on=['matrix', 'kernel_id', 'n_cols'], how='left')
    df['speedup'] = df['gflops'] / df['gflops_original']
    
    return df


def build_metrics_config(df, include_extended=False):
    """Build a metrics config dict for improvement calculation.

    Args:
        df: DataFrame whose columns determine which metrics are available.
        include_extended: If True, include additional metrics (bandwidth_avg,
            max spreads, blocks-per-row) used by reorder analysis plots.

    Returns:
        Dict mapping metric column names to dicts with keys
        'improvement_name' and 'higher_is_better'.
    """
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'bandwidth_avg': {'improvement_name': 'bandwidth_avg_improvement', 'higher_is_better': False},
        'locality_avg_row_spread': {'improvement_name': 'row_spread_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
        'locality_profile': {'improvement_name': 'profile_improvement', 'higher_is_better': False},
    }

    if include_extended:
        metrics_config.update({
            'locality_max_row_spread': {'improvement_name': 'max_row_spread_improvement', 'higher_is_better': False},
            'locality_max_col_spread': {'improvement_name': 'max_col_spread_improvement', 'higher_is_better': False},
        })

    # Add block density improvements
    for col in df.columns:
        if col.startswith('block_density_'):
            bs = col.split('_')[-1]
            metrics_config[col] = {'improvement_name': f'density_improvement_{bs}', 'higher_is_better': True}

    # Add avg blocks-per-row improvements (always available for correlation tables)
    for bs in [4, 8, 16, 32, 64, 128]:
        col = f'avg_blocks_per_row_{bs}'
        if col in df.columns:
            metrics_config[col] = {
                'improvement_name': f'avg_blocks_per_row_improvement_{bs}',
                'higher_is_better': False,
            }

    # Add max blocks-per-row improvements (extended mode only)
    if include_extended:
        for bs in [4, 8, 16, 32, 64, 128]:
            col = f'max_blocks_per_row_{bs}'
            if col in df.columns:
                metrics_config[col] = {
                    'improvement_name': f'max_blocks_per_row_improvement_{bs}',
                    'higher_is_better': False,
                }

    return metrics_config


def compute_improvements(df, metrics_config):
    """Compute improvement columns from a metrics config.

    For each metric present in both the DataFrame and the config, merges in
    the per-matrix Original baseline and computes the improvement ratio.

    Args:
        df: DataFrame that must contain a 'strategy' column with 'Original'
            rows and a 'matrix' column.
        metrics_config: Dict produced by build_metrics_config.

    Returns:
        DataFrame with improvement columns added.
    """
    df = df.copy()

    available_metrics = [m for m in metrics_config if m in df.columns]
    if not available_metrics:
        return df

    original = df[df['strategy'] == 'Original'][['matrix'] + available_metrics].drop_duplicates()
    original = original.groupby('matrix')[available_metrics].mean().reset_index()
    original = original.rename(columns={m: f'{m}_original' for m in available_metrics})

    df = pd.merge(df, original, on='matrix', how='left')

    for metric, config in metrics_config.items():
        if metric not in available_metrics:
            continue
        orig_col = f'{metric}_original'
        imp_col = config['improvement_name']
        if config['higher_is_better']:
            df[imp_col] = df[metric] / df[orig_col]
        else:
            df[imp_col] = df[orig_col] / df[metric]

    return df


def add_improvement_columns(df):
    """Add improvement columns comparing reordered to original.

    Adds for each metric:
        - {metric}_improvement: ratio showing improvement factor

    For metrics where higher is better: improvement = reordered / original
    For metrics where lower is better: improvement = original / reordered
    """
    metrics_config = build_metrics_config(df)
    return compute_improvements(df, metrics_config)


def add_size_class(df):
    """Add size_class column based on nnz."""
    df = df.copy()
    
    if 'nnz' not in df.columns:
        return df
    
    bins = [0, 5e4, 2e5, 1e6, np.inf]
    labels = ['<50K', '50K-200K', '200K-1M', '>1M']
    df['size_class'] = pd.cut(df['nnz'], bins=bins, labels=labels)
    
    return df


def prepare_full_dataframe(df):
    """Apply all transformations to prepare a fully processed DataFrame.
    
    This is a convenience function that applies all processing steps.
    """
    df = add_base_metrics(df)
    df = add_relative_metrics(df)
    df = add_speedup(df)
    df = add_improvement_columns(df)
    df = add_size_class(df)
    return df


# =============================================================================
# Filtering Functions
# =============================================================================




def load_matrix_family_map(matrices_list_path):
    """Load mapping from matrix name to family/group."""
    matrix_to_family = {}
    
    if not Path(matrices_list_path).exists():
        print(f"Warning: Matrices list file {matrices_list_path} not found.")
        return matrix_to_family
    
    with open(matrices_list_path, 'r') as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            path = path.replace('\\', '/')
            parts = path.split('/')
            if len(parts) >= 3:
                matrix_name = parts[-1]  # e.g., bcspwr10.mtx
                family = parts[-3]       # e.g., HB (the SuiteSparse group)
                matrix_to_family[matrix_name] = family
    
    return matrix_to_family


def filter_one_per_family(df, matrices_list_path, keep_full_families=None):
    """Filter DataFrame to keep only one matrix per family."""
    if keep_full_families is None:
        keep_full_families = load_filter_config().get('filters', {}).get('keep_full_families', [])
    
    matrix_to_family = load_matrix_family_map(matrices_list_path)
    
    if not matrix_to_family:
        print("Warning: No matrix mapping found. Skipping family filtering.")
        return df
    
    df = df.copy()
    df['family'] = df['matrix'].map(matrix_to_family).fillna(df['matrix'])
    
    # Separate families to keep full vs filter
    keep_full_mask = df['family'].isin(keep_full_families)
    df_keep_full = df[keep_full_mask]
    df_to_filter = df[~keep_full_mask]
    
    # Select one matrix per family
    unique_matrices = df_to_filter[['matrix', 'family']].drop_duplicates().sort_values('matrix')
    selected = unique_matrices.groupby('family').first()['matrix'].tolist()
    
    # Combine with kept-full families
    all_selected = list(set(selected + df_keep_full['matrix'].unique().tolist()))
    
    result = df[df['matrix'].isin(all_selected)]
    print(f"Family filter: {len(df)} -> {len(result)} rows ({len(all_selected)} matrices)")
    
    return result


def filter_trivial_matrices(df, df_analysis, bandwidth_threshold=5):
    """Filter out matrices with original bandwidth below threshold."""
    if 'bandwidth_max' not in df_analysis.columns:
        return df
    
    trivial = df_analysis[
        (df_analysis['perm'] == 'None') & 
        (df_analysis['bandwidth_max'] < bandwidth_threshold)
    ]['matrix'].unique()
    
    if len(trivial) > 0:
        print(f"Filtering {len(trivial)} trivial matrices (bandwidth < {bandwidth_threshold})")
        df = df[~df['matrix'].isin(trivial)]
    
    return df


def filter_sparse_matrices(df, df_analysis, nnz_factor=2):
    """Filter out very sparse matrices (nnz < factor * rows)."""
    if 'nnz' not in df_analysis.columns or 'rows' not in df_analysis.columns:
        return df
    
    sparse = df_analysis[
        (df_analysis['perm'] == 'None') & 
        (df_analysis['nnz'] < nnz_factor * df_analysis['rows'])
    ]['matrix'].unique()
    
    if len(sparse) > 0:
        print(f"Filtering {len(sparse)} very sparse matrices (nnz < {nnz_factor}*N)")
        df = df[~df['matrix'].isin(sparse)]
    
    return df


def filter_diagonal_matrices(df, df_analysis):
    """Filter out purely diagonal matrices (nnz == rows for square matrices)."""
    if 'nnz' not in df_analysis.columns or 'rows' not in df_analysis.columns:
        return df
    
    diagonal = df_analysis[
        (df_analysis['perm'] == 'None') & 
        (df_analysis['rows'] == df_analysis['cols']) &
        (df_analysis['nnz'] == df_analysis['rows'])
    ]['matrix'].unique()
    
    if len(diagonal) > 0:
        print(f"Filtering {len(diagonal)} purely diagonal matrices")
        df = df[~df['matrix'].isin(diagonal)]
    
    return df


def filter_square_only(df):
    """Keep only square matrices."""
    if 'rows' in df.columns and 'cols' in df.columns:
        result = df[df['rows'] == df['cols']]
        print(f"Square filter: {len(df)} -> {len(result)} rows")
        return result
    return df


def filter_min_size(df, min_rows):
    """Keep only matrices with at least min_rows rows."""
    if 'rows' in df.columns:
        result = df[df['rows'] >= min_rows]
        print(f"Min size filter (>={min_rows}): {len(df)} -> {len(result)} rows")
        return result
    return df


def filter_max_size(df, max_dim):
    """Remove matrices where rows or cols exceed *max_dim*."""
    if 'rows' in df.columns and 'cols' in df.columns:
        result = df[(df['rows'] <= max_dim) & (df['cols'] <= max_dim)]
        print(f"Max size filter (<={max_dim}): {len(df)} -> {len(result)} rows")
        return result
    return df


def _filter_matrices_from_both(df, df_analysis, matrices_to_remove, reason):
    """Remove matrices from both DataFrames and print a message."""
    if len(matrices_to_remove) > 0:
        print(f"Filtering {len(matrices_to_remove)} {reason}")
        if not df.empty:
            df = df[~df['matrix'].isin(matrices_to_remove)]
        df_analysis = df_analysis[~df_analysis['matrix'].isin(matrices_to_remove)]
    return df, df_analysis


# =============================================================================
# Filter Config Loading
# =============================================================================

# Default path to the centralized filter config (next to this file)
_DEFAULT_FILTER_CONFIG = Path(__file__).resolve().parent / 'filter_config.yaml'


def load_filter_config(config_path=None):
    """Load the centralized filter configuration from a YAML file.

    Args:
        config_path: Path to YAML config. Defaults to
                     ``scripts/filter_config.yaml`` next to this module.

    Returns:
        dict with keys ``data`` and ``filters``.
    """
    if config_path is None:
        config_path = _DEFAULT_FILTER_CONFIG
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: filter config {config_path} not found. Using built-in defaults.",
              file=sys.stderr)
        return {
            'data': {
                'operations_csv': 'results/results_operations.csv',
                'analysis_csv': 'results/results_analysis.csv',
                'matrices_list': 'datasets/matrices_list_mtx.txt',
            },
            'filters': {
                'one_per_family': True,
                'square_only': True,
                'min_size': None,
                'max_size': None,
                'min_bandwidth': None,
                'max_sparsity_factor': 2,
                'filter_diagonal': True,
                'keep_full_families': [],
            }
        }

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_filters(df, df_analysis, matrices_list_path=None,
                  one_per_family=True, square_only=True,
                  min_size=None, max_size=None, min_bandwidth=None,
                  max_sparsity_factor=None, filter_diagonal=True,
                  keep_full_families=None):
    """Apply all configured filters to both DataFrames.

    Args:
        min_bandwidth: Remove matrices whose original bandwidth is below
            this value.  ``None`` disables the filter.
        max_sparsity_factor: Remove matrices where
            ``nnz < factor * rows``.  ``None`` disables the filter.

    Returns:
        Tuple of (filtered_df, filtered_df_analysis)
    """
    ops_empty = df.empty

    if one_per_family and matrices_list_path:
        if not ops_empty:
            df = filter_one_per_family(df, matrices_list_path, keep_full_families)
        df_analysis = filter_one_per_family(df_analysis, matrices_list_path, keep_full_families)

    if min_bandwidth is not None:
        if 'bandwidth_max' in df_analysis.columns:
            trivial_matrices = df_analysis[
                (df_analysis['perm'] == 'None') &
                (df_analysis['bandwidth_max'] < min_bandwidth)
            ]['matrix'].unique()
            df, df_analysis = _filter_matrices_from_both(
                df, df_analysis, trivial_matrices,
                f"trivial matrices (bandwidth < {min_bandwidth})")

    if max_sparsity_factor is not None:
        if 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
            sparse_matrices = df_analysis[
                (df_analysis['perm'] == 'None') &
                (df_analysis['nnz'] < max_sparsity_factor * df_analysis['rows'])
            ]['matrix'].unique()
            df, df_analysis = _filter_matrices_from_both(
                df, df_analysis, sparse_matrices,
                f"very sparse matrices (nnz < {max_sparsity_factor}*rows)")

    # Filter purely diagonal matrices
    if filter_diagonal:
        if 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
            diagonal_matrices = df_analysis[
                (df_analysis['perm'] == 'None') &
                (df_analysis['rows'] == df_analysis['cols']) &
                (df_analysis['nnz'] == df_analysis['rows'])
            ]['matrix'].unique()
            df, df_analysis = _filter_matrices_from_both(
                df, df_analysis, diagonal_matrices, "purely diagonal matrices")

    if square_only:
        if not ops_empty:
            df = filter_square_only(df)
        df_analysis = filter_square_only(df_analysis)

    if min_size is not None:
        if not ops_empty:
            df = filter_min_size(df, min_size)
        df_analysis = filter_min_size(df_analysis, min_size)

    if max_size is not None:
        if not ops_empty:
            df = filter_max_size(df, max_size)
        df_analysis = filter_max_size(df_analysis, max_size)

    return df, df_analysis


def load_and_filter_data(config_path=None, cli_overrides=None):
    """Single entry-point: load CSVs, apply every filter from config.

    This is the **only** function that plotting / table scripts should call
    to obtain their DataFrames.  It guarantees that all consumers work on
    identically filtered data.

    Args:
        config_path: Path to ``filter_config.yaml``. ``None`` -> default.
        cli_overrides: Optional dict of overrides (e.g. from argparse).
            Only keys whose values are not ``None`` override the config.
            Recognised keys mirror the YAML ``filters`` section plus
            ``operations_csv``, ``analysis_csv``, ``matrices_list``.

    Returns:
        Tuple of (filtered_operations_df, filtered_analysis_df, cfg)
    """
    cfg = load_filter_config(config_path)
    data_cfg = dict(cfg.get('data', {}))
    filt_cfg = dict(cfg.get('filters', {}))

    # Apply CLI overrides --------------------------------------------------
    if cli_overrides:
        for key in ('operations_csv', 'analysis_csv', 'matrices_list'):
            if cli_overrides.get(key) is not None:
                data_cfg[key] = cli_overrides[key]
        for key in filt_cfg:
            if key in cli_overrides and cli_overrides[key] is not None:
                filt_cfg[key] = cli_overrides[key]

    # Load -----------------------------------------------------------------
    print("Loading data...")
    df, df_analysis = load_data(
        data_cfg.get('operations_csv', 'results/results_operations.csv'),
        data_cfg.get('analysis_csv', 'results/results_analysis.csv'),
    )

    # Filter ---------------------------------------------------------------
    print("\nApplying filters (from filter_config.yaml)...")
    df, df_analysis = apply_filters(
        df, df_analysis,
        matrices_list_path=data_cfg.get('matrices_list',
                                        'datasets/matrices_list_mtx.txt'),
        one_per_family=filt_cfg.get('one_per_family', True),
        square_only=filt_cfg.get('square_only', True),
        min_size=filt_cfg.get('min_size'),
        max_size=filt_cfg.get('max_size'),
        min_bandwidth=filt_cfg.get('min_bandwidth'),
        max_sparsity_factor=filt_cfg.get('max_sparsity_factor'),
        filter_diagonal=filt_cfg.get('filter_diagonal', True),
        keep_full_families=filt_cfg.get('keep_full_families', []),
    )

    print(f"After filtering: {len(df)} operation rows, "
          f"{len(df_analysis)} analysis rows")

    return df, df_analysis, cfg


# =============================================================================
# Correlation Method Configuration
# =============================================================================

_VALID_CORR_METHODS = {'pearson', 'spearman', 'kendall'}

# Module-level cache so the config is read once.
_correlation_method_cache = None


def get_correlation_method(cfg=None):
    """Return the configured correlation method (pearson/spearman/kendall).

    Reads from ``display.correlation_method`` in filter_config.yaml.
    Falls back to ``'pearson'`` when the key is absent.
    """
    global _correlation_method_cache
    if _correlation_method_cache is not None:
        return _correlation_method_cache

    if cfg is None:
        cfg = load_filter_config()
    method = cfg.get('display', {}).get('correlation_method', 'pearson')
    method = method.lower()
    if method not in _VALID_CORR_METHODS:
        print(f"Warning: unknown correlation_method '{method}', falling back to 'pearson'",
              file=sys.stderr)
        method = 'pearson'
    _correlation_method_cache = method
    return method


def compute_correlation(x, y, method=None):
    """Compute correlation between *x* and *y* using the configured method.

    Args:
        x, y: array-like values (NaNs should be dropped beforehand).
        method: 'pearson', 'spearman', or 'kendall'.
                ``None`` reads from config.

    Returns:
        (correlation_coefficient, p_value)
    """
    if method is None:
        method = get_correlation_method()
    if method == 'pearson':
        return stats.pearsonr(x, y)
    elif method == 'spearman':
        return stats.spearmanr(x, y)
    elif method == 'kendall':
        return stats.kendalltau(x, y)
    raise ValueError(f"Unknown correlation method: {method}")


def correlation_display_symbol(method=None):
    """Return a short display symbol for the configured correlation method."""
    if method is None:
        method = get_correlation_method()
    return {'pearson': 'r', 'spearman': r'\rho', 'kendall': r'\tau'}[method]


def correlation_display_name(method=None):
    """Return a human-readable name for the configured correlation method."""
    if method is None:
        method = get_correlation_method()
    return {
        'pearson': "Pearson's $r$",
        'spearman': "Spearman's $\\rho$",
        'kendall': "Kendall's $\\tau$",
    }[method]


# =============================================================================
# Generic Plot Functions
# =============================================================================

def _setup_figure(figsize=(10, 8)):
    """Create figure with standard settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_figure(path, dpi=300):
    """Save figure and close."""
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    print(f"Saved: {Path(path).name}")


def scatter_with_correlation(df, x_col, y_col, output_path,
                              title=None, hue_col=None,
                              log_x=None, log_y=None,
                              show_correlation=True,
                              figsize=(12, 9)):
    """Create scatter plot with correlation annotation.

    The correlation method is determined by ``display.correlation_method``
    in ``filter_config.yaml`` (default: pearson).

    Args:
        df: DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis
        output_path: Path to save figure
        title: Plot title (auto-generated if None)
        hue_col: Column for color grouping
        log_x: Use log scale for x (auto-detect if None, use False for linear)
        log_y: Use log scale for y (auto-detect if None, use False for linear)
        show_correlation: Whether to show correlations in title
        figsize: Figure size
    """
    # Clean data
    cols = [x_col, y_col] + ([hue_col] if hue_col else [])
    plot_df = df.dropna(subset=[x_col, y_col])

    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    # Auto-detect scales (only if None)
    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)

    # Calculate correlation using configured method
    method = get_correlation_method()
    sym = correlation_display_symbol(method)
    corr_val = _correlation_for_scatter(plot_df[x_col], plot_df[y_col], method)

    # Create plot
    fig, ax = _setup_figure(figsize)

    if hue_col and hue_col in plot_df.columns:
        pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col,
                        alpha=0.7, ax=ax, palette=pal)
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.7, ax=ax)

    # Set scales
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Format log axes with proper ticks and grid
    if log_x or log_y:
        format_log_axes(ax)
    else:
        ax.grid(True, alpha=0.3)

    # Labels
    ax.set_xlabel(get_display_name(x_col))
    ax.set_ylabel(get_display_name(y_col))

    # Title
    if title is None:
        title = f"{get_display_name(y_col)} vs {get_display_name(x_col)}"
    if show_correlation:
        title += f"\n${sym} = {corr_val:.3f}$"
    ax.set_title(title)

    _save_figure(output_path)


def boxplot_by_category(df, x_col, y_col, output_path,
                         title=None, order=None,
                         baseline=None, show_points=True,
                         clip_percentile=None,
                         log_y=False,
                         ylim=None,
                         palette=None,
                         figsize=(12, 8)):
    """Create boxplot with optional stripplot overlay.
    
    Args:
        df: DataFrame
        x_col: Column for categories (x-axis)
        y_col: Column for values (y-axis)
        output_path: Path to save figure
        title: Plot title
        order: Order of categories
        baseline: Value for horizontal reference line
        show_points: Whether to overlay individual points
        clip_percentile: Tuple of (lower, upper) percentiles for clipping (default: None, no clipping)
        log_y: Whether to use log scale on y-axis
        ylim: Tuple of (ymin, ymax) for y-axis limits (None for auto)
        palette: Color palette dict {category: color}. When x_col is
                 'strategy', defaults to ``get_strategy_palette()``.
        figsize: Figure size
    """
    plot_df = df.dropna(subset=[x_col, y_col]).copy()
    
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return
    
    # For log scale, filter out non-positive values
    if log_y:
        plot_df = plot_df[plot_df[y_col] > 0]
        if plot_df.empty:
            print(f"Skipping {output_path}: no positive data for log scale")
            return
    
    # Clip extreme values
    if clip_percentile:
        lower = plot_df[y_col].quantile(clip_percentile[0] / 100)
        upper = plot_df[y_col].quantile(clip_percentile[1] / 100)
        plot_df = plot_df[(plot_df[y_col] >= lower) & (plot_df[y_col] <= upper)]
    
    fig, ax = _setup_figure(figsize)
    
    # Resolve palette: use strategy colors when plotting by strategy
    if palette is None and x_col == 'strategy':
        palette = get_strategy_palette(order)
    
    # Draw stripplot first if requested
    if show_points:
        sns.stripplot(data=plot_df, x=x_col, y=y_col, order=order,
                      color='black', alpha=0.4, jitter=0.25, size=3, ax=ax)
    
    # Draw boxplot with whiskers at 5th–95th percentiles
    sns.boxplot(data=plot_df, x=x_col, y=y_col, order=order,
                showfliers=False, whis=(5, 95), palette=palette, width=0.6,
                boxprops={'alpha': 0.6},
                medianprops={'color': 'red', 'linewidth': 2},
                ax=ax)
    
    if baseline is not None:
        ax.axhline(baseline, color='red', linestyle='--', alpha=0.7, label=f'Baseline = {baseline}')
    
    # Set log scale if requested
    if log_y:
        ax.set_yscale('log')
    
    # Set y-axis limits if specified
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Format axes (must be after scale and limits are set)
    if log_y:
        format_log_axes(ax, which='y')
    else:
        ax.grid(True, axis='y', alpha=0.3)
    
    ax.set_xlabel(get_display_name(x_col))
    ax.set_ylabel(get_display_name(y_col))
    
    if title is None:
        title = f"{get_display_name(y_col)} by {get_display_name(x_col)}"
    if clip_percentile:
        title += f"\n({clip_percentile[0]}-{clip_percentile[1]} percentile)"
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    
    _save_figure(output_path)


def breakeven_boxplot(df_valid, df_harmful, x_col, y_col, output_path,
                       title=None, order=None,
                       cap_value=None,
                       clip_percentile=None,
                       palette=None,
                       figsize=(12, 9)):
    """Create breakeven plot with two aligned subplots.

    **Top subplot**: boxplot of break-even iteration counts (valid data only,
    log-scale y-axis).

    **Bottom subplot**: stacked percentage bar for each strategy — green
    (beneficial, breaks even) on top, red (harmful, never breaks even) on
    the bottom — giving an immediate read on how often a reordering helps.

    Args:
        df_valid: DataFrame with valid (positive) break-even counts.
        df_harmful: DataFrame with harmful reorderings (never break even).
        x_col: Category column (e.g. 'strategy').
        y_col: Value column (break-even count).
        output_path: Path to save figure.
        title: Plot title.
        order: Category order for x-axis.
        cap_value: Unused (kept for API compatibility).
        clip_percentile: Percentile range to clip valid data (lower, upper). Default: None (no clipping).
        palette: Color palette dict {category: color}.
        figsize: Figure size.
    """
    plot_df = df_valid.dropna(subset=[x_col, y_col]).copy()
    plot_df = plot_df[plot_df[y_col] > 0]

    has_valid = not plot_df.empty
    has_harmful = df_harmful is not None and not df_harmful.empty

    if not has_valid and not has_harmful:
        print(f"Skipping {output_path}: no data")
        return

    # Clip extreme valid values
    if has_valid and clip_percentile:
        lower = plot_df[y_col].quantile(clip_percentile[0] / 100)
        upper = plot_df[y_col].quantile(clip_percentile[1] / 100)
        plot_df = plot_df[(plot_df[y_col] >= lower) & (plot_df[y_col] <= upper)]

    # Resolve palette
    if palette is None and x_col == 'strategy':
        palette = get_strategy_palette(order)

    # Determine category list and mapping
    if order is not None:
        cats = list(order)
    elif has_valid:
        cats = sorted(plot_df[x_col].unique())
    elif has_harmful:
        cats = sorted(df_harmful[x_col].unique())
    else:
        cats = []
    positions = list(range(len(cats)))

    # --- Compute per-strategy valid / harmful counts ---
    valid_pcts = []
    harm_pcts = []
    for cat in cats:
        n_v = int((plot_df[x_col] == cat).sum()) if has_valid else 0
        n_h = int((df_harmful[x_col] == cat).sum()) if has_harmful else 0
        tot = n_v + n_h
        valid_pcts.append(100.0 * n_v / tot if tot > 0 else 100.0)
        harm_pcts.append(100.0 * n_h / tot if tot > 0 else 0.0)

    # --- Create two-row figure: boxplot on top (tall), bar chart on bottom ---
    fig, (ax_box, ax_bar) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
    )

    BOX_WIDTH = 0.6

    # ===== Top subplot: breakeven boxplot =====
    if has_valid:
        rng = np.random.default_rng(0)
        for i, cat in enumerate(cats):
            cat_data = plot_df[plot_df[x_col] == cat]
            if cat_data.empty:
                continue
            half_w = BOX_WIDTH / 2 * 0.8
            x_jitter = rng.uniform(-half_w, half_w, size=len(cat_data))
            ax_box.scatter(i + x_jitter, cat_data[y_col].values,
                           color='black', alpha=0.4, s=9, zorder=2,
                           edgecolors='none', rasterized=True)

        box_data = [plot_df[plot_df[x_col] == cat][y_col].dropna().values
                    for cat in cats]
        bp = ax_box.boxplot(box_data, positions=positions,
                            widths=BOX_WIDTH, showfliers=False,
                            whis=(5, 95),
                            patch_artist=True,
                            medianprops={'color': 'red', 'linewidth': 2},
                            zorder=3)

        for patch, cat in zip(bp['boxes'], cats):
            color = palette.get(cat, '#4c72b0') if palette else '#4c72b0'
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax_box.set_yscale('log')
    ax_box.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0], numticks=15))
    ax_box.yaxis.set_minor_locator(NullLocator())
    ax_box.yaxis.set_minor_formatter(NullFormatter())
    ax_box.grid(True, which='major', alpha=0.5, linewidth=0.8)
    ax_box.grid(False, which='minor')
    ax_box.set_ylabel('# of Operations to Break Even')

    if title is None:
        title = f"Break-even Operations by {get_display_name(x_col)}"
    ax_box.set_title(title)

    # ===== Bottom subplot: success / failure percentage bars =====
    bar_w = 0.6
    # Red (harmful) on bottom, green (beneficial) on top
    ax_bar.bar(positions, harm_pcts, width=bar_w,
               color='#d9534f', edgecolor='white', linewidth=0.5,
               label='Harmful (never breaks even)')
    ax_bar.bar(positions, valid_pcts, width=bar_w, bottom=harm_pcts,
               color='#5cb85c', edgecolor='white', linewidth=0.5,
               label='Beneficial (finite break-even)')

    # Annotate percentages inside the bars
    for i, (vp, hp) in enumerate(zip(valid_pcts, harm_pcts)):
        if hp >= 8:
            ax_bar.text(i, hp / 2, f"{hp:.0f}%", ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')
        if vp >= 8:
            ax_bar.text(i, hp + vp / 2, f"{vp:.0f}%", ha='center', va='center',
                        fontsize=9, fontweight='bold', color='white')

    ax_bar.set_ylim(0, 100)
    ax_bar.set_yticks([0, 25, 50, 75, 100])
    ax_bar.set_ylabel('Success (%)')
    ax_bar.set_xlabel(get_display_name(x_col))
    ax_bar.legend(loc='lower right', fontsize=8, ncol=2)

    # Shared x-axis labels
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels(cats, rotation=45, ha='right')

    _save_figure(output_path)


def binned_bar_chart(df, bin_col, value_col, output_path,
                      bins=None, labels=None,
                      title=None, baseline=None,
                      agg_func='median',
                      min_count=5,
                      figsize=(10, 6)):
    """Create bar chart showing aggregated values per bin.
    
    Args:
        df: DataFrame
        bin_col: Column to bin
        value_col: Column to aggregate
        output_path: Path to save figure
        bins: Bin edges (default: [0, 0.5, 1.0, 2.0, 5.0, 10.0, 1000.0])
        labels: Bin labels (default: ['<0.5x', '0.5-1x', '1-2x', '2-5x', '5-10x', '>10x'])
        title: Plot title
        baseline: Value for horizontal reference line
        agg_func: Aggregation function ('median' or 'mean')
        min_count: Minimum samples per bin to include
        figsize: Figure size
    """
    if bins is None:
        bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 1000.0]
    if labels is None:
        labels = ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-5x', '5-10x', '>10x']
    
    plot_df = df.dropna(subset=[bin_col, value_col]).copy()
    
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return
    
    # Create bins
    plot_df['bin'] = pd.cut(plot_df[bin_col], bins=bins, labels=labels)
    
    # Aggregate
    stats_df = plot_df.groupby('bin', observed=True)[value_col].agg([agg_func, 'count']).reset_index()
    stats_df = stats_df[stats_df['count'] >= min_count]
    
    if stats_df.empty:
        print(f"Skipping {output_path}: insufficient data per bin")
        return
    
    fig, ax = _setup_figure(figsize)
    
    bars = sns.barplot(data=stats_df, x='bin', y=agg_func, palette="viridis", ax=ax)
    
    # Add count labels
    for i, p in enumerate(ax.patches):
        if i < len(stats_df):
            ax.annotate(f"n={int(stats_df.iloc[i]['count'])}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, 5), textcoords='offset points')
    
    if baseline is not None:
        ax.axhline(baseline, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel(f"{get_display_name(bin_col)} Bin")
    ax.set_ylabel(f"{agg_func.title()} {get_display_name(value_col)}")
    
    if title is None:
        title = f"{agg_func.title()} {get_display_name(value_col)} by {get_display_name(bin_col)}"
    ax.set_title(title)
    
    ax.grid(True, axis='y', alpha=0.3)
    _save_figure(output_path)


def binned_boxplot(df, bin_col, value_col, output_path,
                    bins=None, labels=None,
                    title=None, baseline=None,
                    min_count=5,
                    show_points=False,
                    figsize=(10, 6)):
    """Create boxplot showing distribution of values per bin.
    
    Args:
        df: DataFrame
        bin_col: Column to bin
        value_col: Column to plot distribution for
        output_path: Path to save figure
        bins: Bin edges (default: [0, 0.5, 1.0, 2.0, 5.0, 10.0, 1000.0])
        labels: Bin labels (default: ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-5x', '5-10x', '>10x'])
        title: Plot title
        baseline: Value for horizontal reference line
        min_count: Minimum samples per bin to include
        show_points: Whether to overlay individual points with stripplot
        figsize: Figure size
    """
    if bins is None:
        bins = [0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 1000.0]
    if labels is None:
        labels = ['<0.5x', '0.5-1x', '1-1.5x', '1.5-2x', '2-5x', '5-10x', '>10x']
    
    plot_df = df.dropna(subset=[bin_col, value_col]).copy()
    
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return
    
    # Create bins
    plot_df['bin'] = pd.cut(plot_df[bin_col], bins=bins, labels=labels)
    
    # Filter bins with insufficient counts
    bin_counts = plot_df['bin'].value_counts()
    valid_bins = bin_counts[bin_counts >= min_count].index
    plot_df = plot_df[plot_df['bin'].isin(valid_bins)]
    
    if plot_df.empty:
        print(f"Skipping {output_path}: insufficient data per bin")
        return
    
    # Get bin order and counts for annotation
    bin_order = [label for label in labels if label in valid_bins]
    
    fig, ax = _setup_figure(figsize)
    
    # Draw stripplot first if requested
    if show_points:
        sns.stripplot(data=plot_df, x='bin', y=value_col, order=bin_order,
                      color='black', alpha=0.4, jitter=0.25, size=3, ax=ax)
    
    # Draw boxplot with whiskers at 5th–95th percentiles
    sns.boxplot(data=plot_df, x='bin', y=value_col, order=bin_order,
                showfliers=False, whis=(5, 95), palette="viridis", width=0.6,
                boxprops={'alpha': 0.6},
                medianprops={'color': 'red', 'linewidth': 2},
                ax=ax)

    # Add count labels positioned just above the top whisker (95th percentile)
    for i, label in enumerate(bin_order):
        count = bin_counts[label]
        bin_data = plot_df[plot_df['bin'] == label][value_col]
        y_pos = bin_data.quantile(0.95)
        ax.text(i, y_pos, f"n={int(count)}",
                ha='center', va='bottom', fontsize=9)
    
    if baseline is not None:
        ax.axhline(baseline, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel(f"{get_display_name(bin_col)} Bin")
    ax.set_ylabel(f"{get_display_name(value_col)}")
    
    if title is None:
        title = f"{get_display_name(value_col)} Distribution by {get_display_name(bin_col)}"
    ax.set_title(title)
    
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    _save_figure(output_path)


def cdf_plot(df, value_col, output_path,
              hue_col=None, hue_order=None,
              title=None, baseline=None,
              log_x=False,
              figsize=(10, 6)):
    """Create CDF plot.
    
    Args:
        df: DataFrame
        value_col: Column to plot CDF for
        output_path: Path to save figure
        hue_col: Column for separate CDFs
        hue_order: Order for hue categories
        title: Plot title
        baseline: Value for vertical reference line
        log_x: Use log scale for x-axis
        figsize: Figure size
    """
    plot_df = df.dropna(subset=[value_col])
    
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return
    
    fig, ax = _setup_figure(figsize)
    
    if hue_col and hue_col in plot_df.columns:
        categories = hue_order if hue_order else sorted(plot_df[hue_col].unique())
        strategy_pal = get_strategy_palette(categories) if hue_col == 'strategy' else {}
        for cat in categories:
            subset = plot_df[plot_df[hue_col] == cat]
            values = subset[value_col].sort_values()
            cdf_y = np.arange(1, len(values) + 1) / len(values)
            color = strategy_pal.get(cat)
            ax.step(values, cdf_y, label=cat, where='post', linewidth=2,
                    **({"color": color} if color else {}))
        ax.legend(title=get_display_name(hue_col))
    else:
        values = plot_df[value_col].sort_values()
        cdf_y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, cdf_y, where='post', linewidth=2)
    
    if baseline is not None:
        ax.axvline(baseline, color='gray', linestyle='--', alpha=0.7)
    
    if log_x:
        ax.set_xscale('log')
        format_log_axes(ax, which='x')
    else:
        ax.grid(True, alpha=0.3)
    
    ax.set_xlabel(get_display_name(value_col))
    ax.set_ylabel('CDF')
    
    if title is None:
        title = f"CDF of {get_display_name(value_col)}"
    ax.set_title(title)
    
    _save_figure(output_path)


def violin_plot(df, x_col, y_col, output_path,
                 title=None, order=None,
                 show_points=True,
                 figsize=(12, 8)):
    """Create violin plot with optional point overlay.
    
    Args:
        df: DataFrame
        x_col: Column for categories
        y_col: Column for values
        output_path: Path to save figure
        title: Plot title
        order: Order of categories
        show_points: Whether to overlay individual points
        figsize: Figure size
    """
    plot_df = df.dropna(subset=[x_col, y_col])
    
    if plot_df.empty:
        print(f"Skipping {output_path}: no data")
        return
    
    fig, ax = _setup_figure(figsize)
    
    sns.violinplot(data=plot_df, x=x_col, y=y_col, order=order,
                   palette="Set2", inner="quartile", cut=0, ax=ax)
    
    if show_points:
        sns.stripplot(data=plot_df, x=x_col, y=y_col, order=order,
                      color='black', alpha=0.3, jitter=True, size=3, ax=ax)
    
    ax.set_xlabel(get_display_name(x_col))
    ax.set_ylabel(get_display_name(y_col))
    
    if title is None:
        title = f"{get_display_name(y_col)} by {get_display_name(x_col)}"
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    _save_figure(output_path)


def correlation_heatmap(df, cols, output_path,
                         title=None, method=None,
                         figsize=(10, 8)):
    """Create correlation heatmap for selected columns.

    Args:
        df: DataFrame
        cols: List of columns to correlate
        output_path: Path to save figure
        title: Plot title
        method: Correlation method. ``None`` reads from config.
        figsize: Figure size
    """
    if method is None:
        method = get_correlation_method()
    available_cols = [c for c in cols if c in df.columns]
    
    if len(available_cols) < 2:
        print(f"Skipping {output_path}: need at least 2 columns")
        return
    
    corr = df[available_cols].corr(method=method)
    
    fig, ax = _setup_figure(figsize)
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, vmin=-1, vmax=1, ax=ax)
    
    if title is None:
        title = f"Correlation Matrix ({method.title()})"
    ax.set_title(title)
    
    _save_figure(output_path)


# =============================================================================
# Utility Functions
# =============================================================================

def get_strategy_order(df):
    """Get standard order for strategies (Original first, then sorted).

    The order follows ``PERMS`` definition order for known algorithms,
    with any unknown strategies appended alphabetically.
    """
    strategies = set(df['strategy'].unique())
    # Canonical order from settings
    canonical = [p['display'] for p in PERMS.values() if p['display'] in strategies]
    # Unknown strategies not in PERMS
    extra = sorted(strategies - set(canonical) - {'Original'})
    result = canonical + extra
    if 'Original' in strategies:
        result = ['Original'] + result
    return result


def get_strategy_palette(strategies=None):
    """Return a {strategy_display_name: color} dict for seaborn palette kwarg.

    If *strategies* is given (list of display names), only those entries are
    returned.  'Original' gets a neutral grey.
    """
    base = {p['display']: p['color'] for p in PERMS.values()}
    base['Original'] = '#888888'
    if strategies is not None:
        return {s: base.get(s, '#333333') for s in strategies}
    return base


def get_density_columns(df):
    """Get list of block density columns (only block_density_N format)."""
    return [c for c in df.columns if c.startswith('block_density_') and c.split('_')[-1].isdigit()]


def get_improvement_columns(df):
    """Get list of improvement columns."""
    return [c for c in df.columns if c.endswith('_improvement')]


def safe_filename(name):
    """Convert string to safe filename."""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')


# =============================================================================
# Publication-Ready Scatter Plots
# =============================================================================

def _pearson_for_scatter(x_vals, y_vals):
    """Compute Pearson r on raw (linear) values."""
    valid = x_vals.notna() & y_vals.notna()
    xp, yp = x_vals[valid], y_vals[valid]
    if len(xp) >= 2:
        r, _ = stats.pearsonr(xp, yp)
        return r
    return np.nan


def _correlation_for_scatter(x_vals, y_vals, method=None):
    """Compute correlation on raw values using the configured method."""
    valid = x_vals.notna() & y_vals.notna()
    xp, yp = x_vals[valid], y_vals[valid]
    if len(xp) >= 2:
        r, _ = compute_correlation(xp, yp, method=method)
        return r
    return np.nan


def scatter_publication(df, x_col, y_col, output_path,
                        hue_col=None, log_x=None, log_y=None,
                        show_correlation=True, label=None,
                        figsize=(3.5, 3.0)):
    """Publication-ready scatter plot sized for 6-per-half-page layout.

    No title.  Large ticks and axis labels for readability at small print
    size.  Correlation shown as in-plot annotation.
    """
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)

    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in plot_df.columns:
        pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col,
                        alpha=0.7, ax=ax, palette=pal, s=15)
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.7, ax=ax, s=15)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if log_x or log_y:
        format_log_axes(ax)
    else:
        ax.grid(True, alpha=0.3)

    ax.set_xlabel(get_display_name(x_col), fontsize=13)
    ax.set_ylabel(get_display_name(y_col), fontsize=13)
    ax.tick_params(axis='both', labelsize=11, which='major')

    if show_correlation:
        method = get_correlation_method()
        sym = correlation_display_symbol(method)
        corr_val = _correlation_for_scatter(plot_df[x_col], plot_df[y_col], method)
        ax.text(0.03, 0.97, f"${sym}={corr_val:.2f}$",
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    if label:
        ax.text(0.97, 0.03, label, transform=ax.transAxes, fontsize=10,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.8))

    _save_figure(output_path)


def grouped_scatter_publication(df, x_col, y_col, group_col, group_order,
                                output_path, group_labels=None,
                                hue_col=None, log_x=None, log_y=None,
                                show_correlation=True,
                                figsize=(7.0, 4.5)):
    """2x3 grouped scatter with shared axes.  One subplot per group.

    Designed for 6 kernels on half a page: y-labels only on the left
    column, x-labels only on the bottom row.
    """
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) < 2:
        print(f"Skipping {output_path}: insufficient data")
        return

    if log_x is None:
        log_x = use_log_scale(x_col)
    if log_y is None:
        log_y = use_log_scale(y_col)
    if group_labels is None:
        group_labels = {}

    nrows, ncols = 2, 3
    n_groups = min(len(group_order), nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    for idx in range(n_groups):
        group = group_order[idx]
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df_g = plot_df[plot_df[group_col] == group]

        if len(df_g) < 2:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', fontsize=8)
        else:
            if hue_col and hue_col in df_g.columns:
                pal = get_strategy_palette() if hue_col == 'strategy' else "Set2"
                sns.scatterplot(data=df_g, x=x_col, y=y_col, hue=hue_col,
                                alpha=0.7, ax=ax, palette=pal, s=10,
                                legend=(idx == 0))
            else:
                sns.scatterplot(data=df_g, x=x_col, y=y_col,
                                alpha=0.7, ax=ax, s=10)

            if show_correlation:
                method = get_correlation_method()
                sym = correlation_display_symbol(method)
                cr = _correlation_for_scatter(df_g[x_col], df_g[y_col], method)
                ax.text(0.03, 0.97, f"${sym}={cr:.2f}$",
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                  alpha=0.8))

        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        # Group label (kernel name)
        ax.text(0.97, 0.03, group_labels.get(group, group),
                transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
                fontweight='bold')

        # Edge labels only
        if c == 0:
            ax.set_ylabel(get_display_name(y_col), fontsize=10)
        else:
            ax.set_ylabel('')
        if r == nrows - 1:
            ax.set_xlabel(get_display_name(x_col), fontsize=10)
        else:
            ax.set_xlabel('')

        ax.tick_params(axis='both', labelsize=8)

        if log_x or log_y:
            format_log_axes(ax)
        else:
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_groups, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {Path(output_path).name}")
