"""
Generate correlation tables between metrics and kernel performance (GFLOPS).
Outputs LaTeX tables for each n_cols value.
"""

import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')


# =============================================================================
# Display Name Dictionaries - Edit these to change table labels
# =============================================================================

# Kernel display names (row labels)
KERNEL_NAMES = {
    'ASPT_SPMM': 'ASPT',
    'CUSPARSE_SPMM_BSR_bs32': 'cuSPARSE BSR',
    'CUSPARSE_SPMM_CSR': 'cuSPARSE CSR',
    'DTC_SPMM': 'DTC',
    'FLASHSPARSE_SPMM': 'FlashSparse',
    'SMAT_SPMM_bs32': 'SMAT',
}

# Metric display names (column labels)
METRIC_NAMES = {
    'rel_bandwidth': 'Rel. Bandwidth',
    'rel_row_spread': 'Rel. Row Spread',
    'locality_vertical_adjacency_ratio': 'Vert. Adj. Ratio',
    'block_density_32': 'Block Density',
}

# Metrics to include in the table (in order)
METRICS = [
    'rel_bandwidth',
    'rel_row_spread',
    'locality_vertical_adjacency_ratio',
    'block_density_32',
]

# Block sizes available for block density metrics
BLOCK_SIZES = [4, 8, 16, 32, 64, 128]

# Block density metric display names
BLOCK_DENSITY_METRIC_NAMES = {f'block_density_{bs}': f'${bs}\\times{bs}$' for bs in BLOCK_SIZES}

# Block density metrics in order
BLOCK_DENSITY_METRICS = [f'block_density_{bs}' for bs in BLOCK_SIZES]


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_and_process_data(ops_path, analysis_path):
    """Load and merge operations and analysis data."""
    df_op = pd.read_csv(ops_path)
    df_analysis = pd.read_csv(analysis_path)

    # Normalize keys
    for df in [df_op, df_analysis]:
        df['perm'] = df['perm'].fillna('None').astype(str)
        df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)
        df['matrix'] = df['matrix'].astype(str)

    # Merge
    df = pd.merge(df_op, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')

    # Ensure n_cols exists
    if 'n_cols' in df.columns:
        df['n_cols'] = pd.to_numeric(df['n_cols'], errors='coerce').fillna(32)
    else:
        df['n_cols'] = 32

    # Calculate GFLOPS
    df['gflops'] = (2 * df['nnz'] * df['n_cols']) / (df['time_operation_ms'] * 1e-3) / 1e9

    # Create kernel_id
    def get_kernel_id(row):
        algo = row['algo']
        algo = re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', algo)
        if pd.notnull(row.get('block_size')) and row['block_size'] > 0:
            return f"{algo}_bs{int(row['block_size'])}"
        return algo

    df['kernel_id'] = df.apply(get_kernel_id, axis=1)

    # Add relative metrics
    if 'bandwidth_max' in df.columns and 'rows' in df.columns:
        df['rel_bandwidth'] = df['bandwidth_max'] / df['rows']
    if 'locality_avg_row_spread' in df.columns and 'cols' in df.columns:
        df['rel_row_spread'] = df['locality_avg_row_spread'] / df['cols']

    return df


def compute_correlations(df, n_cols, metrics, kernels=None):
    """Compute Kendall's tau correlations between metrics and GFLOPS.
    
    Args:
        df: DataFrame with merged data
        n_cols: Filter to this n_cols value
        metrics: List of metric columns to compute correlations for
        kernels: List of kernels to include (None = all)
    
    Returns:
        DataFrame with correlations (rows=kernels, cols=metrics)
    """
    df_nc = df[df['n_cols'] == n_cols]
    
    if kernels is None:
        kernels = sorted(df_nc['kernel_id'].unique())
    
    results = []
    for kernel in kernels:
        df_k = df_nc[df_nc['kernel_id'] == kernel]
        row_data = {'kernel': kernel}
        
        for metric in metrics:
            if metric not in df_k.columns:
                row_data[metric] = np.nan
                continue
                
            valid = df_k[[metric, 'gflops']].dropna()
            if len(valid) >= 10:
                tau, p = stats.kendalltau(valid[metric], valid['gflops'])
                row_data[metric] = tau
            else:
                row_data[metric] = np.nan
        
        results.append(row_data)
    
    return pd.DataFrame(results)


# =============================================================================
# LaTeX Generation
# =============================================================================

def correlation_to_latex(corr_df, metrics, kernel_names, metric_names, 
                         caption=None, label=None):
    """Convert correlation DataFrame to LaTeX table string.
    
    Args:
        corr_df: DataFrame with correlations (must have 'kernel' column)
        metrics: List of metric columns in desired order
        kernel_names: Dict mapping kernel IDs to display names
        metric_names: Dict mapping metric columns to display names
        caption: Table caption (optional)
        label: Table label for referencing (optional)
    
    Returns:
        LaTeX table string
    """
    lines = []
    
    # Table header
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    
    # Column specification: first column left-aligned, rest centered
    col_spec = 'l' + 'c' * len(metrics)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    
    # Header row
    header_cols = ['Kernel'] + [metric_names.get(m, m) for m in metrics]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')
    
    # Data rows
    for _, row in corr_df.iterrows():
        kernel = row['kernel']
        kernel_display = kernel_names.get(kernel, kernel)
        
        # Find the maximum value in this row (by absolute value for correlations)
        metric_values = {m: row.get(m, np.nan) for m in metrics}
        valid_values = {m: v for m, v in metric_values.items() if not pd.isna(v)}
        max_metric = max(valid_values, key=lambda m: abs(valid_values[m])) if valid_values else None
        
        values = [kernel_display]
        for metric in metrics:
            val = row.get(metric, np.nan)
            if pd.isna(val):
                values.append('--')
            elif metric == max_metric:
                values.append(r'\textbf{' + f'{val:.3f}' + '}')
            else:
                values.append(f'{val:.3f}')
        
        lines.append(' & '.join(values) + r' \\')
    
    # Table footer
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    
    if caption:
        lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')
    
    lines.append(r'\end{table}')
    
    return '\n'.join(lines)


def generate_all_tables(df, output_dir, metrics=None, kernel_names=None, 
                        metric_names=None):
    """Generate LaTeX tables for all n_cols values.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save .tex files
        metrics: List of metrics to include (default: METRICS)
        kernel_names: Dict for kernel display names (default: KERNEL_NAMES)
        metric_names: Dict for metric display names (default: METRIC_NAMES)
    """
    if metrics is None:
        metrics = METRICS
    if kernel_names is None:
        kernel_names = KERNEL_NAMES
    if metric_names is None:
        metric_names = METRIC_NAMES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get kernels in the order defined by KERNEL_NAMES, or all if not defined
    all_kernels = sorted(df['kernel_id'].unique())
    kernels = [k for k in kernel_names.keys() if k in all_kernels]
    if not kernels:
        kernels = all_kernels
    
    n_cols_values = sorted(df['n_cols'].unique())
    
    for n_cols in n_cols_values:
        n_cols_int = int(n_cols)
        
        # Count unique (matrix, perm, perm_type) combinations for this n_cols
        df_nc = df[df['n_cols'] == n_cols]
        n_matrices = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
        
        corr_df = compute_correlations(df, n_cols, metrics, kernels)
        
        caption = (f"Kendall's $\\tau$ correlation between metrics and SpMM GFLOPS ($n_{{cols}} = {n_cols_int}$). "
                   f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                   f"for a total of {n_matrices:,} configurations.")
        label = f"tab:correlation_ncols_{n_cols_int}"
        
        latex = correlation_to_latex(
            corr_df, metrics, kernel_names, metric_names,
            caption=caption, label=label
        )
        
        output_path = output_dir / f"correlation_ncols_{n_cols_int}.tex"
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"Saved: {output_path}")


def generate_blocksize_tables(df, output_dir, kernel_names=None):
    """Generate LaTeX tables for block density correlations across block sizes.
    
    Creates tables showing how block density at different block sizes 
    correlates with GFLOPS for each kernel.
    
    Args:
        df: Processed DataFrame
        output_dir: Directory to save .tex files
        kernel_names: Dict for kernel display names (default: KERNEL_NAMES)
    """
    if kernel_names is None:
        kernel_names = KERNEL_NAMES
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get kernels in the order defined by KERNEL_NAMES, or all if not defined
    all_kernels = sorted(df['kernel_id'].unique())
    kernels = [k for k in kernel_names.keys() if k in all_kernels]
    if not kernels:
        kernels = all_kernels
    
    n_cols_values = sorted(df['n_cols'].unique())
    
    for n_cols in n_cols_values:
        n_cols_int = int(n_cols)
        
        # Count unique (matrix, perm, perm_type) combinations for this n_cols
        df_nc = df[df['n_cols'] == n_cols]
        n_matrices = df_nc[['matrix', 'perm', 'perm_type']].drop_duplicates().shape[0]
        
        corr_df = compute_correlations(df, n_cols, BLOCK_DENSITY_METRICS, kernels)
        
        caption = (f"Kendall's $\\tau$ correlation between block density and SpMM GFLOPS across block sizes ($n_{{cols}} = {n_cols_int}$). "
                   f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                   f"for a total of {n_matrices:,} configurations.")
        label = f"tab:blocksize_correlation_ncols_{n_cols_int}"
        
        latex = correlation_to_latex(
            corr_df, BLOCK_DENSITY_METRICS, kernel_names, BLOCK_DENSITY_METRIC_NAMES,
            caption=caption, label=label
        )
        
        output_path = output_dir / f"blocksize_correlation_ncols_{n_cols_int}.tex"
        with open(output_path, 'w') as f:
            f.write(latex)
        
        print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate correlation tables as LaTeX files"
    )
    parser.add_argument(
        "--operations", 
        default="results/results_operations.csv",
        help="Path to operations CSV"
    )
    parser.add_argument(
        "--analysis", 
        default="results/results_analysis.csv",
        help="Path to analysis CSV"
    )
    parser.add_argument(
        "--output", 
        default="plots/correlation_tables",
        help="Output directory for .tex files"
    )
    parser.add_argument(
        "--blocksize-only",
        action="store_true",
        help="Generate only block size correlation tables"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("Loading data...")
    df = load_and_process_data(args.operations, args.analysis)
    print(f"Loaded {len(df)} rows")
    
    print("\nGenerating LaTeX tables...")
    if args.blocksize_only:
        generate_blocksize_tables(df, args.output)
    else:
        generate_all_tables(df, args.output)
        generate_blocksize_tables(df, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
