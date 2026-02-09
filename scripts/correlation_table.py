"""
Generate correlation tables between metrics and kernel performance (GFLOPS),
and median structural improvement ratio tables for reordering algorithms.
Outputs LaTeX tables.
"""

import pandas as pd
import numpy as np
from scipy import stats
import re
import sys
import warnings
from pathlib import Path
import argparse

# Allow importing plot_utils from the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_utils as pu

warnings.filterwarnings('ignore')


# =============================================================================
# Display Name Dictionaries - Edit these to change table labels
# =============================================================================

# Reordering algorithm display names (row labels for improvement tables)
PERM_NAMES = {
    'SB_rcm':          'RCM',
    'SB_amd':          'AMD',
    'SB_metis':        'Metis',
    'SB_patoh':        'PaToH',
    'SB_rabbit':       'Rabbit',
    'SB_gray':         'Gray',
    'SB_degree':       'Degree',
    'GROOT_reorder':   'GROOT',
    'SPARTA_reorder':  'SPARTA',
    'random1D':        'Random',
}

# Kernel display names (row labels)
KERNEL_NAMES = {
    'ASPT_SPMM': 'ASPT',
    'CUSPARSE_SPMM_BSR_bs32': 'cuSP BSR',
    'CUSPARSE_SPMM_CSR': 'cuSP CSR',
    'DTC_SPMM': 'DTC',
    'FLASHSPARSE_SPMM': 'FlashSP',
    'SMAT_SPMM_bs32': 'SMAT',
}

# ---------------------------------------------------------------------------
# All structural metrics: set 'enabled' to True/False to include/exclude.
# 'name' is the short display name for column headers.
# Edit this dictionary to choose which metrics appear in the correlation table.
# ---------------------------------------------------------------------------
ALL_METRICS = {
    # --- Bandwidth ---
    'bandwidth_max':                       {'name': 'BW',      'full_name': 'Bandwidth',                          'enabled': False, 'higher_is_better': False},
    'bandwidth_avg':                       {'name': 'ABW',     'full_name': 'Average Bandwidth',                  'enabled': False, 'higher_is_better': False},
    'rel_bandwidth':                       {'name': 'RBW',     'full_name': 'Relative Bandwidth',                 'enabled': True,  'higher_is_better': False},
    # --- Row spread / locality ---
    'locality_avg_row_spread':             {'name': 'ARS',     'full_name': 'Average Row Spread',                 'enabled': False, 'higher_is_better': False},
    'locality_max_row_spread':             {'name': 'MRS',     'full_name': 'Maximum Row Spread',                 'enabled': False, 'higher_is_better': False},
    'rel_row_spread':                      {'name': 'RRS',     'full_name': 'Relative Row Spread',                'enabled': True,  'higher_is_better': False},
    # --- Column spread ---
    'locality_avg_col_spread':             {'name': 'ACS',     'full_name': 'Average Column Spread',              'enabled': False, 'higher_is_better': False},
    'locality_max_col_spread':             {'name': 'MCS',     'full_name': 'Maximum Column Spread',              'enabled': False, 'higher_is_better': False},
    # --- Vertical adjacency ---
    'locality_consecutive_vertical_pairs': {'name': 'CVP',     'full_name': 'Consecutive Vertical Pairs',         'enabled': False, 'higher_is_better': True},
    'locality_vertical_adjacency_ratio':   {'name': 'VAR',     'full_name': 'Vertical Adjacency Ratio',           'enabled': True,  'higher_is_better': True},
    # --- NNZ distribution ---
    'locality_avg_nnz_per_row':            {'name': 'ANR',     'full_name': 'Average NNZ per Row',                'enabled': False, 'higher_is_better': None},
    'locality_max_nnz_per_row':            {'name': 'MNR',     'full_name': 'Maximum NNZ per Row',                'enabled': False, 'higher_is_better': None},
    'locality_num_empty_rows':             {'name': 'NER',     'full_name': 'Number of Empty Rows',               'enabled': False, 'higher_is_better': None},
    'locality_num_empty_cols':             {'name': 'NEC',     'full_name': 'Number of Empty Columns',            'enabled': False, 'higher_is_better': None},
    # --- Profile ---
    'locality_profile':                    {'name': 'Prof',    'full_name': 'Profile',                            'enabled': False, 'higher_is_better': False},
    # --- Overall density ---
    'density':                             {'name': 'Dens',    'full_name': 'Density',                            'enabled': False, 'higher_is_better': None},
    # --- Block density (per block size) ---
    'block_density_4':                     {'name': 'BD4',     'full_name': 'Block Density $4{\\times}4$',         'enabled': False, 'higher_is_better': True},
    'block_density_8':                     {'name': 'BD8',     'full_name': 'Block Density $8{\\times}8$',         'enabled': True,  'higher_is_better': True},
    'block_density_16':                    {'name': 'BD16',    'full_name': 'Block Density $16{\\times}16$',       'enabled': False, 'higher_is_better': True},
    'block_density_32':                    {'name': 'BD32',    'full_name': 'Block Density $32{\\times}32$',       'enabled': True,  'higher_is_better': True},
    'block_density_64':                    {'name': 'BD64',    'full_name': 'Block Density $64{\\times}64$',       'enabled': False, 'higher_is_better': True},
    'block_density_128':                   {'name': 'BD128',   'full_name': 'Block Density $128{\\times}128$',     'enabled': True,  'higher_is_better': True},
    # --- Avg blocks per row (per block size) ---
    'avg_blocks_per_row_4':                {'name': 'ABR4',    'full_name': 'Avg Blocks/Row $4{\\times}4$',        'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_8':                {'name': 'ABR8',    'full_name': 'Avg Blocks/Row $8{\\times}8$',        'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_16':               {'name': 'ABR16',   'full_name': 'Avg Blocks/Row $16{\\times}16$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_32':               {'name': 'ABR32',   'full_name': 'Avg Blocks/Row $32{\\times}32$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_64':               {'name': 'ABR64',   'full_name': 'Avg Blocks/Row $64{\\times}64$',      'enabled': False, 'higher_is_better': False},
    'avg_blocks_per_row_128':              {'name': 'ABR128',  'full_name': 'Avg Blocks/Row $128{\\times}128$',    'enabled': False, 'higher_is_better': False},
    # --- Max blocks per row (per block size) ---
    'max_blocks_per_row_4':                {'name': 'MBR4',    'full_name': 'Max Blocks/Row $4{\\times}4$',        'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_8':                {'name': 'MBR8',    'full_name': 'Max Blocks/Row $8{\\times}8$',        'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_16':               {'name': 'MBR16',   'full_name': 'Max Blocks/Row $16{\\times}16$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_32':               {'name': 'MBR32',   'full_name': 'Max Blocks/Row $32{\\times}32$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_64':               {'name': 'MBR64',   'full_name': 'Max Blocks/Row $64{\\times}64$',      'enabled': False, 'higher_is_better': False},
    'max_blocks_per_row_128':              {'name': 'MBR128',  'full_name': 'Max Blocks/Row $128{\\times}128$',    'enabled': False, 'higher_is_better': False},
}

# Derived lists from the dictionary (do not edit manually)
METRICS = [k for k, v in ALL_METRICS.items() if v['enabled']]
METRIC_NAMES = {k: v['name'] for k, v in ALL_METRICS.items()}
METRIC_FULL_NAMES = {k: v['full_name'] for k, v in ALL_METRICS.items()}

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
    """Compute Kendall's tau and Pearson correlations between metrics and GFLOPS.
    
    Args:
        df: DataFrame with merged data
        n_cols: Filter to this n_cols value
        metrics: List of metric columns to compute correlations for
        kernels: List of kernels to include (None = all)
    
    Returns:
        DataFrame with correlations (rows=kernels, cols=metrics with _tau and _pearson suffixes)
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
                row_data[f'{metric}_tau'] = np.nan
                row_data[f'{metric}_pearson'] = np.nan
                continue
                
            valid = df_k[[metric, 'gflops']].dropna()
            # Filter positive values for log-based Pearson
            valid_pos = valid[(valid[metric] > 0) & (valid['gflops'] > 0)]
            
            if len(valid) >= 10:
                tau, _ = stats.kendalltau(valid[metric], valid['gflops'])
                row_data[f'{metric}_tau'] = tau
            else:
                row_data[f'{metric}_tau'] = np.nan
            
            if len(valid_pos) >= 10:
                # Pearson on log values (both metric and gflops typically benefit from log scale)
                pearson_r, _ = stats.pearsonr(np.log10(valid_pos[metric]), np.log10(valid_pos['gflops']))
                row_data[f'{metric}_pearson'] = pearson_r
            else:
                row_data[f'{metric}_pearson'] = np.nan
        
        results.append(row_data)
    
    return pd.DataFrame(results)


# =============================================================================
# LaTeX Generation
# =============================================================================

def _multiline_header(name):
    """Wrap a metric name into a \\shortstack with up to 3 lines for compact headers."""
    import re as _re
    # Special case: names ending with a $...$ block (e.g. 'Block Density $32{\times}32$')
    m = _re.match(r'^(.+?)\s+(\$.*\$)$', name)
    if m:
        prefix = m.group(1)
        suffix = m.group(2)
        prefix_words = prefix.split()
        if len(prefix_words) == 1:
            return r'\shortstack{' + prefix_words[0] + r'\\' + suffix + '}'
        else:
            return r'\shortstack{' + r'\\'.join(prefix_words) + r'\\' + suffix + '}'
    words = name.split()
    if len(words) <= 1:
        return name
    if len(words) == 2:
        return r'\shortstack{' + words[0] + r'\\' + words[1] + '}'
    # 3+ words: one word per line (up to ~3-4 lines)
    return r'\shortstack{' + r'\\'.join(words) + '}'


def correlation_to_latex(corr_df, metrics, kernel_names, metric_names, 
                         corr_type='tau', caption=None, label=None,
                         metric_full_names=None):
    """Convert correlation DataFrame to LaTeX table string.
    
    Args:
        corr_df: DataFrame with correlations (must have 'kernel' column and 
                 columns with _tau and _pearson suffixes)
        metrics: List of metric columns in desired order (base names without suffix)
        kernel_names: Dict mapping kernel IDs to display names
        metric_names: Dict mapping metric columns to display names
        corr_type: 'tau' for Kendall's tau or 'pearson' for Pearson's r
        caption: Table caption (optional)
        label: Table label for referencing (optional)
        metric_full_names: Dict mapping metric columns to full display names
                           (used for multiline column headers). Falls back to
                           metric_names if not provided.
    
    Returns:
        LaTeX table string
    """
    header_names = metric_full_names if metric_full_names else metric_names
    
    lines = []
    
    # Table header
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')
    
    # Column specification: first column left-aligned, rest centered
    col_spec = 'l' + 'c' * len(metrics)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')
    
    # Header row (multiline full names)
    header_cols = ['Kernel'] + [_multiline_header(header_names.get(m, m)) for m in metrics]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')
    
    # Data rows
    suffix = f'_{corr_type}'
    for _, row in corr_df.iterrows():
        kernel = row['kernel']
        kernel_display = kernel_names.get(kernel, kernel)
        
        # Find the maximum value in this row (for bolding)
        metric_values = {m: row.get(f'{m}{suffix}', np.nan) for m in metrics}
        valid_values = {m: v for m, v in metric_values.items() if not pd.isna(v)}
        max_metric = max(valid_values, key=lambda m: abs(valid_values[m])) if valid_values else None
        
        values = [kernel_display]
        for metric in metrics:
            val = row.get(f'{metric}{suffix}', np.nan)
            
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


def _build_metric_legend(metrics, metric_names, metric_full_names):
    """Build a LaTeX legend string expanding metric acronyms.
    
    Example output: 'RBW: Relative Bandwidth, RRS: Relative Row Spread, ...'
    """
    parts = []
    for m in metrics:
        abbr = metric_names.get(m, m)
        full = metric_full_names.get(m, m)
        parts.append(f"{abbr}: {full}")
    return ', '.join(parts) + '.'


def generate_all_tables(df, output_dir, metrics=None, kernel_names=None, 
                        metric_names=None):
    """Generate LaTeX tables for all n_cols values.
    
    Creates separate tables for Kendall's tau and Pearson's r correlations.
    
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
        
        # Generate Kendall's tau table
        caption_tau = (f"Kendall's $\\tau$ correlation between metrics and SpMM GFLOPS ($n_{{cols}} = {n_cols_int}$). "
                       f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                       f"for a total of {n_matrices:,} configurations.")
        label_tau = f"tab:correlation_tau_ncols_{n_cols_int}"
        
        latex_tau = correlation_to_latex(
            corr_df, metrics, kernel_names, metric_names,
            corr_type='tau', caption=caption_tau, label=label_tau,
            metric_full_names=METRIC_FULL_NAMES
        )
        
        output_path_tau = output_dir / f"correlation_tau_ncols_{n_cols_int}.tex"
        with open(output_path_tau, 'w') as f:
            f.write(latex_tau)
        print(f"Saved: {output_path_tau}")
        
        # Generate Pearson's r table
        caption_pearson = (f"Pearson's $r$ correlation (on log values) between metrics and SpMM GFLOPS ($n_{{cols}} = {n_cols_int}$). "
                           f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                           f"for a total of {n_matrices:,} configurations.")
        label_pearson = f"tab:correlation_pearson_ncols_{n_cols_int}"
        
        latex_pearson = correlation_to_latex(
            corr_df, metrics, kernel_names, metric_names,
            corr_type='pearson', caption=caption_pearson, label=label_pearson,
            metric_full_names=METRIC_FULL_NAMES
        )
        
        output_path_pearson = output_dir / f"correlation_pearson_ncols_{n_cols_int}.tex"
        with open(output_path_pearson, 'w') as f:
            f.write(latex_pearson)
        print(f"Saved: {output_path_pearson}")


def generate_blocksize_tables(df, output_dir, kernel_names=None):
    """Generate LaTeX tables for block density correlations across block sizes.
    
    Creates separate tables for Kendall's tau and Pearson's r showing how 
    block density at different block sizes correlates with GFLOPS for each kernel.
    
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
        
        # Generate Kendall's tau table
        caption_tau = (f"Kendall's $\\tau$ correlation between block density and SpMM GFLOPS across block sizes ($n_{{cols}} = {n_cols_int}$). "
                       f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                       f"for a total of {n_matrices:,} configurations.")
        label_tau = f"tab:blocksize_tau_ncols_{n_cols_int}"
        
        latex_tau = correlation_to_latex(
            corr_df, BLOCK_DENSITY_METRICS, kernel_names, BLOCK_DENSITY_METRIC_NAMES,
            corr_type='tau', caption=caption_tau, label=label_tau,
            metric_full_names=METRIC_FULL_NAMES
        )
        
        output_path_tau = output_dir / f"blocksize_tau_ncols_{n_cols_int}.tex"
        with open(output_path_tau, 'w') as f:
            f.write(latex_tau)
        print(f"Saved: {output_path_tau}")
        
        # Generate Pearson's r table
        caption_pearson = (f"Pearson's $r$ correlation (on log values) between block density and SpMM GFLOPS across block sizes ($n_{{cols}} = {n_cols_int}$). "
                           f"Correlation is calculated across matrices in SuiteSparse and all their reorderings, "
                           f"for a total of {n_matrices:,} configurations.")
        label_pearson = f"tab:blocksize_pearson_ncols_{n_cols_int}"
        
        latex_pearson = correlation_to_latex(
            corr_df, BLOCK_DENSITY_METRICS, kernel_names, BLOCK_DENSITY_METRIC_NAMES,
            corr_type='pearson', caption=caption_pearson, label=label_pearson,
            metric_full_names=METRIC_FULL_NAMES
        )
        
        output_path_pearson = output_dir / f"blocksize_pearson_ncols_{n_cols_int}.tex"
        with open(output_path_pearson, 'w') as f:
            f.write(latex_pearson)
        print(f"Saved: {output_path_pearson}")


# =============================================================================
# Median Structural Improvement Ratio Tables
# =============================================================================

def load_analysis_data(analysis_path):
    """Load analysis data and add derived metrics (rel_bandwidth, rel_row_spread)."""
    df = pd.read_csv(analysis_path)
    df['perm'] = df['perm'].fillna('None').astype(str)
    df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)
    df['matrix'] = df['matrix'].astype(str)

    # Add relative metrics
    if 'bandwidth_max' in df.columns and 'rows' in df.columns:
        df['rel_bandwidth'] = df['bandwidth_max'] / df['rows']
    if 'locality_avg_row_spread' in df.columns and 'cols' in df.columns:
        df['rel_row_spread'] = df['locality_avg_row_spread'] / df['cols']

    return df


def compute_improvement_ratios(df_analysis, metrics=None):
    """Compute per-(matrix, perm, perm_type) improvement ratios vs the original.

    For each matrix the original is the row with perm='None'.
    Improvement ratio:
        higher_is_better=True  -> reordered / original  (>1 means improved)
        higher_is_better=False -> original / reordered   (>1 means improved)

    Returns:
        DataFrame with columns: matrix, perm, perm_type, and one
        improvement ratio column per metric (named '<metric>_imp').
    """
    if metrics is None:
        metrics = METRICS

    # Only keep metrics that have a direction defined
    metrics = [m for m in metrics
               if m in df_analysis.columns and ALL_METRICS[m].get('higher_is_better') is not None]

    originals = df_analysis[df_analysis['perm'] == 'None'][['matrix'] + metrics].copy()
    originals = originals.groupby('matrix')[metrics].mean().reset_index()
    originals = originals.rename(columns={m: f'{m}_orig' for m in metrics})

    reordered = df_analysis[df_analysis['perm'] != 'None'].copy()
    reordered = reordered.merge(originals, on='matrix', how='left')

    for m in metrics:
        orig_col = f'{m}_orig'
        imp_col = f'{m}_imp'
        if ALL_METRICS[m]['higher_is_better']:
            reordered[imp_col] = reordered[m] / reordered[orig_col]
        else:
            reordered[imp_col] = reordered[orig_col] / reordered[m]

    keep_cols = ['matrix', 'perm', 'perm_type'] + [f'{m}_imp' for m in metrics]
    return reordered[keep_cols], metrics


def improvement_to_latex(median_df, metrics, perm_names, caption=None, label=None):
    """Convert a median-improvement DataFrame to a LaTeX table.

    Args:
        median_df: DataFrame with 'perm' column and '<metric>_imp' value columns.
        metrics: ordered list of base metric names.
        perm_names: dict mapping perm id -> display name.
        caption, label: LaTeX caption / label.

    Returns:
        LaTeX table string.
    """
    header_names = METRIC_FULL_NAMES

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\footnotesize')

    col_spec = 'l' + 'c' * len(metrics)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    header_cols = ['Algorithm'] + [_multiline_header(header_names.get(m, m)) for m in metrics]
    lines.append(' & '.join(header_cols) + r' \\')
    lines.append(r'\midrule')

    for _, row in median_df.iterrows():
        perm = row['perm']
        perm_display = perm_names.get(perm, perm)

        values_dict = {}
        for m in metrics:
            val = row.get(f'{m}_imp', np.nan)
            if not pd.isna(val):
                values_dict[m] = val

        # Bold the best (highest) median improvement in each row
        best_metric = max(values_dict, key=lambda m: values_dict[m]) if values_dict else None

        cells = [perm_display]
        for m in metrics:
            val = row.get(f'{m}_imp', np.nan)
            if pd.isna(val):
                cells.append('--')
            elif m == best_metric:
                cells.append(r'\textbf{' + f'{val:.2f}' + '}')
            else:
                cells.append(f'{val:.2f}')

        lines.append(' & '.join(cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    if caption:
        lines.append(r'\caption{' + caption + '}')
    if label:
        lines.append(r'\label{' + label + '}')

    lines.append(r'\end{table}')
    return '\n'.join(lines)


def generate_improvement_tables(df_analysis, output_dir, metrics=None,
                                perm_names=None):
    """Generate LaTeX tables of median structural improvement ratios.

    Creates one table per perm_type (SYMMETRIC, ROW).

    Args:
        df_analysis: raw analysis DataFrame (from load_analysis_data).
        output_dir: directory for .tex output.
        metrics: list of metric keys (default: enabled METRICS).
        perm_names: display-name dict (default: PERM_NAMES).
    """
    if metrics is None:
        metrics = METRICS
    if perm_names is None:
        perm_names = PERM_NAMES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    imp_df, used_metrics = compute_improvement_ratios(df_analysis, metrics)

    if imp_df.empty:
        print("No improvement data to generate tables.")
        return

    # Determine perm order from PERM_NAMES (or alphabetical fallback)
    all_perms = sorted(imp_df['perm'].unique())
    ordered_perms = [p for p in perm_names.keys() if p in all_perms]
    remaining = [p for p in all_perms if p not in ordered_perms]
    ordered_perms += remaining

    n_matrices = df_analysis[df_analysis['perm'] == 'None']['matrix'].nunique()

    for perm_type in sorted(imp_df['perm_type'].unique()):
        df_pt = imp_df[imp_df['perm_type'] == perm_type]

        imp_cols = [f'{m}_imp' for m in used_metrics]
        median_df = df_pt.groupby('perm')[imp_cols].median().reset_index()

        # Sort by PERM_NAMES order
        perm_order = {p: i for i, p in enumerate(ordered_perms)}
        median_df['_order'] = median_df['perm'].map(perm_order).fillna(999)
        median_df = median_df.sort_values('_order').drop(columns='_order')

        caption = (f"Median structural improvement ratio per reordering algorithm "
                   f"({perm_type} permutation, {n_matrices} matrices). "
                   f"Values $>1$ indicate improvement over the original ordering.")
        label = f"tab:improvement_{perm_type.lower()}"

        latex = improvement_to_latex(
            median_df, used_metrics, perm_names,
            caption=caption, label=label
        )

        out_path = output_dir / f"improvement_{perm_type.lower()}.tex"
        with open(out_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {out_path}")


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
        "--matrices-list",
        default="datasets/matrices_list_mtx.txt",
        help="Path to matrices list (for --one-per-family filter)"
    )
    parser.add_argument(
        "--one-per-family", action="store_true", default=True,
        help="Keep only one matrix per SuiteSparse family (default: True)"
    )
    parser.add_argument(
        "--no-one-per-family", action="store_false", dest="one_per_family",
        help="Disable one-per-family filtering"
    )
    parser.add_argument(
        "--blocksize-only",
        action="store_true",
        help="Generate only block size correlation tables"
    )
    parser.add_argument(
        "--improvement-only",
        action="store_true",
        help="Generate only median improvement ratio tables"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load & filter data using the same pipeline as plot.py
    # ------------------------------------------------------------------
    print("Loading data...")
    df_merged, df_analysis = pu.load_data(args.operations, args.analysis)

    print("\nApplying filters...")
    df_merged, df_analysis = pu.apply_filters(
        df_merged, df_analysis,
        matrices_list_path=args.matrices_list,
        one_per_family=args.one_per_family,
        square_only=True,
    )
    print(f"After filtering: {len(df_merged)} operation rows, "
          f"{len(df_analysis)} analysis rows")

    # Add derived columns needed by correlation tables
    df = load_and_process_data(args.operations, args.analysis)
    # Keep only filtered matrices
    keep_matrices = df_merged['matrix'].unique()
    df = df[df['matrix'].isin(keep_matrices)]
    print(f"Merged+filtered: {len(df)} rows")

    # Add derived metrics to analysis df for improvement tables
    if 'bandwidth_max' in df_analysis.columns and 'rows' in df_analysis.columns:
        df_analysis['rel_bandwidth'] = df_analysis['bandwidth_max'] / df_analysis['rows']
    if 'locality_avg_row_spread' in df_analysis.columns and 'cols' in df_analysis.columns:
        df_analysis['rel_row_spread'] = df_analysis['locality_avg_row_spread'] / df_analysis['cols']

    # ------------------------------------------------------------------
    # Generate tables
    # ------------------------------------------------------------------
    print("\nGenerating LaTeX tables...")
    if args.improvement_only:
        generate_improvement_tables(df_analysis, args.output)
    elif args.blocksize_only:
        generate_blocksize_tables(df, args.output)
    else:
        generate_all_tables(df, args.output)
        generate_blocksize_tables(df, args.output)
        generate_improvement_tables(df_analysis, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
