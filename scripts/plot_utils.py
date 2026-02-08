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
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import re
import sys


# =============================================================================
# Metric Configuration
# =============================================================================

METRIC_CONFIG = {
    # Performance metrics
    'gflops': {'display': 'GFLOPS', 'log_scale': True},
    'speedup': {'display': 'Speedup', 'log_scale': False},
    
    # Bandwidth metrics
    'bandwidth_max': {'display': 'Bandwidth', 'log_scale': True},
    'rel_bandwidth': {'display': 'Relative Bandwidth', 'log_scale': True},
    'bandwidth_improvement': {'display': 'Bandwidth Reduction', 'log_scale': False},
    
    # Density metrics (block sizes added dynamically)
    'density': {'display': 'Density', 'log_scale': True},
    
    # Locality metrics  
    'rel_row_spread': {'display': 'Relative Row Spread', 'log_scale': True},
    'locality_vertical_adjacency_ratio': {'display': 'Vertical Adjacency Ratio', 'log_scale': False},
    'row_spread_improvement': {'display': 'Row Spread Reduction', 'log_scale': False},
    'col_spread_improvement': {'display': 'Col Spread Reduction', 'log_scale': False},
    'vertical_adjacency_improvement': {'display': 'Vertical Adjacency Improvement', 'log_scale': False},
}

for bs in [4, 8, 16, 32, 64, 128]:
    METRIC_CONFIG[f'block_density_{bs}'] = {'display': f'Block Density (BS {bs})', 'log_scale': True}
    METRIC_CONFIG[f'density_improvement_{bs}'] = {'display': f'Density Improvement (BS {bs})', 'log_scale': False}


def get_display_name(col):
    """Get display name for a column, or format the column name if not configured."""
    if col in METRIC_CONFIG:
        return METRIC_CONFIG[col]['display']
    # Format unknown columns: replace underscores, title case
    return col.replace('_', ' ').title()


def use_log_scale(col):
    """Check if a column should use log scale."""
    return METRIC_CONFIG.get(col, {}).get('log_scale', False)


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_data(ops_path, analysis_path):
    """Load operations and analysis CSVs and merge them.
    
    Returns:
        Tuple of (merged_df, analysis_df)
    """
    try:
        df_op = pd.read_csv(ops_path)
        df_analysis = pd.read_csv(analysis_path)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)

    if df_op.empty or df_analysis.empty:
        print("One of the CSVs is empty.")
        sys.exit(1)

    print(f"Loaded {len(df_op)} operation rows and {len(df_analysis)} analysis rows.")

    # Normalize keys
    for df in [df_op, df_analysis]:
        df['perm'] = df['perm'].fillna('None').astype(str)
        df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)
        df['matrix'] = df['matrix'].astype(str)

    # Merge
    df = pd.merge(df_op, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')
    print(f"Merged DataFrame has {len(df)} rows.")
    
    if df.empty:
        print("Merged DataFrame is empty! Check if keys match in both CSVs.")
        sys.exit(1)
        
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
    
    # Strategy label
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    
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


def add_improvement_columns(df):
    """Add improvement columns comparing reordered to original.
    
    Adds for each metric:
        - {metric}_improvement: ratio showing improvement factor
        
    For metrics where higher is better: improvement = reordered / original
    For metrics where lower is better: improvement = original / reordered
    """
    df = df.copy()
    
    # Define which metrics exist and their improvement direction
    # higher_is_better: True means improvement = reordered/original
    # higher_is_better: False means improvement = original/reordered
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'locality_avg_row_spread': {'improvement_name': 'row_spread_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
    }
    
    # Add block density improvements
    density_cols = [c for c in df.columns if c.startswith('block_density_')]
    for col in density_cols:
        bs = col.split('_')[-1]
        metrics_config[col] = {'improvement_name': f'density_improvement_{bs}', 'higher_is_better': True}
    
    # Get available metrics
    available_metrics = [m for m in metrics_config.keys() if m in df.columns]
    
    if not available_metrics:
        return df
    
    # Get original values
    original = df[df['strategy'] == 'Original'][['matrix'] + available_metrics].drop_duplicates()
    original = original.groupby('matrix')[available_metrics].mean().reset_index()
    original = original.rename(columns={m: f'{m}_original' for m in available_metrics})
    
    df = pd.merge(df, original, on='matrix', how='left')
    
    # Calculate improvements
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

# Families to keep fully (not reduce to one representative)
# These contain diverse, non-duplicate matrices commonly used in sparse matrix research
KEEP_FULL_FAMILIES = [
    'DIMACS10',    # Graph challenge benchmarks
    'SNAP',        # Stanford social/web networks
    'LAW',         # Web graphs (Laboratory for Web Algorithmics)
    'Newman',      # Network science graphs
    'Gleich',      # Web and social network graphs
    'Janna',       # Large-scale FEM problems
    'Norris',      # Structural engineering benchmarks
    'vanHeukelum', # Cage graphs, unique structure
]


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
        keep_full_families = KEEP_FULL_FAMILIES
    
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


def apply_filters(df, df_analysis, matrices_list_path=None, 
                  one_per_family=False, square_only=True, 
                  min_size=None, filter_trivial=True, filter_sparse=True):
    """Apply all configured filters to both DataFrames.
    
    Returns:
        Tuple of (filtered_df, filtered_df_analysis)
    """
    if one_per_family and matrices_list_path:
        df = filter_one_per_family(df, matrices_list_path)
        df_analysis = filter_one_per_family(df_analysis, matrices_list_path)
    
    if filter_trivial:
        trivial_matrices = []
        if 'bandwidth_max' in df_analysis.columns:
            trivial_matrices = df_analysis[
                (df_analysis['perm'] == 'None') & 
                (df_analysis['bandwidth_max'] < 5)
            ]['matrix'].unique()
        if len(trivial_matrices) > 0:
            print(f"Filtering {len(trivial_matrices)} trivial matrices")
            df = df[~df['matrix'].isin(trivial_matrices)]
            df_analysis = df_analysis[~df_analysis['matrix'].isin(trivial_matrices)]
    
    if filter_sparse:
        sparse_matrices = []
        if 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
            sparse_matrices = df_analysis[
                (df_analysis['perm'] == 'None') & 
                (df_analysis['nnz'] < 2 * df_analysis['rows'])
            ]['matrix'].unique()
        if len(sparse_matrices) > 0:
            print(f"Filtering {len(sparse_matrices)} very sparse matrices")
            df = df[~df['matrix'].isin(sparse_matrices)]
            df_analysis = df_analysis[~df_analysis['matrix'].isin(sparse_matrices)]
    
    # Filter purely diagonal matrices
    if 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
        diagonal_matrices = df_analysis[
            (df_analysis['perm'] == 'None') & 
            (df_analysis['rows'] == df_analysis['cols']) &
            (df_analysis['nnz'] == df_analysis['rows'])
        ]['matrix'].unique()
        if len(diagonal_matrices) > 0:
            print(f"Filtering {len(diagonal_matrices)} purely diagonal matrices")
            df = df[~df['matrix'].isin(diagonal_matrices)]
            df_analysis = df_analysis[~df_analysis['matrix'].isin(diagonal_matrices)]
    
    if square_only:
        df = filter_square_only(df)
        df_analysis = filter_square_only(df_analysis)
    
    if min_size is not None:
        df = filter_min_size(df, min_size)
        df_analysis = filter_min_size(df_analysis, min_size)
    
    return df, df_analysis


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
                              figsize=(10, 8)):
    """Create scatter plot with Kendall's Tau and Pearson correlations.
    
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
    
    # Calculate Kendall's tau correlation
    tau, tau_p = stats.kendalltau(plot_df[x_col], plot_df[y_col])
    
    # Calculate Pearson correlation (on log values if using log scale)
    x_vals = plot_df[x_col]
    y_vals = plot_df[y_col]
    
    # For Pearson on log scale, use log-transformed values (filter out non-positive)
    if log_x or log_y:
        valid_mask = (x_vals > 0) & (y_vals > 0)
        x_for_pearson = np.log10(x_vals[valid_mask]) if log_x else x_vals[valid_mask]
        y_for_pearson = np.log10(y_vals[valid_mask]) if log_y else y_vals[valid_mask]
    else:
        x_for_pearson = x_vals
        y_for_pearson = y_vals
    
    if len(x_for_pearson) >= 2:
        pearson_r, pearson_p = stats.pearsonr(x_for_pearson, y_for_pearson)
    else:
        pearson_r, pearson_p = np.nan, np.nan
    
    # Create plot
    fig, ax = _setup_figure(figsize)
    
    if hue_col and hue_col in plot_df.columns:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=hue_col, 
                        alpha=0.7, ax=ax, palette="Set2")
    else:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, alpha=0.7, ax=ax)
    
    # Set scales
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    # Labels
    ax.set_xlabel(get_display_name(x_col))
    ax.set_ylabel(get_display_name(y_col))
    
    # Title
    if title is None:
        title = f"{get_display_name(y_col)} vs {get_display_name(x_col)}"
    if show_correlation:
        title += f"\nτ = {tau:.3f}, r = {pearson_r:.3f}"
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    _save_figure(output_path)


def boxplot_by_category(df, x_col, y_col, output_path,
                         title=None, order=None,
                         baseline=None, show_points=True,
                         clip_percentile=(1, 99),
                         log_y=False,
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
        clip_percentile: Tuple of (lower, upper) percentiles for clipping
        log_y: Whether to use log scale on y-axis
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
    
    # Draw stripplot first if requested
    if show_points:
        sns.stripplot(data=plot_df, x=x_col, y=y_col, order=order,
                      color='black', alpha=0.4, jitter=0.25, size=3, ax=ax)
    
    # Draw boxplot
    sns.boxplot(data=plot_df, x=x_col, y=y_col, order=order,
                showfliers=False, palette="Set2", width=0.6,
                boxprops={'alpha': 0.6},
                medianprops={'color': 'red', 'linewidth': 2},
                ax=ax)
    
    if baseline is not None:
        ax.axhline(baseline, color='red', linestyle='--', alpha=0.7, label=f'Baseline = {baseline}')
    
    # Set log scale if requested
    if log_y:
        ax.set_yscale('log')
    
    ax.set_xlabel(get_display_name(x_col))
    ax.set_ylabel(get_display_name(y_col))
    
    if title is None:
        title = f"{get_display_name(y_col)} by {get_display_name(x_col)}"
    if clip_percentile:
        title += f"\n({clip_percentile[0]}-{clip_percentile[1]} percentile)"
    ax.set_title(title)
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
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
        for cat in categories:
            subset = plot_df[plot_df[hue_col] == cat]
            values = subset[value_col].sort_values()
            cdf_y = np.arange(1, len(values) + 1) / len(values)
            ax.step(values, cdf_y, label=cat, where='post', linewidth=2)
        ax.legend(title=get_display_name(hue_col))
    else:
        values = plot_df[value_col].sort_values()
        cdf_y = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, cdf_y, where='post', linewidth=2)
    
    if baseline is not None:
        ax.axvline(baseline, color='gray', linestyle='--', alpha=0.7)
    
    if log_x:
        ax.set_xscale('log')
    
    ax.set_xlabel(get_display_name(value_col))
    ax.set_ylabel('CDF')
    
    if title is None:
        title = f"CDF of {get_display_name(value_col)}"
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
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
                         title=None, method='kendall',
                         figsize=(10, 8)):
    """Create correlation heatmap for selected columns.
    
    Args:
        df: DataFrame
        cols: List of columns to correlate
        output_path: Path to save figure
        title: Plot title
        method: Correlation method ('kendall', 'pearson', 'spearman')
        figsize: Figure size
    """
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
    """Get standard order for strategies (Original first, then sorted)."""
    strategies = df['strategy'].unique().tolist()
    if 'Original' in strategies:
        strategies.remove('Original')
        return ['Original'] + sorted(strategies)
    return sorted(strategies)


def get_density_columns(df):
    """Get list of block density columns (only block_density_N format)."""
    return [c for c in df.columns if c.startswith('block_density_') and c.split('_')[-1].isdigit()]


def get_improvement_columns(df):
    """Get list of improvement columns."""
    return [c for c in df.columns if c.endswith('_improvement')]


def safe_filename(name):
    """Convert string to safe filename."""
    return name.replace('/', '_').replace(' ', '_').replace('\\', '_')
