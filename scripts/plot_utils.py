import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re

def load_and_merge_data(ops_path, analysis_path):
    """Loads operations and analysis CSVs and merges them."""
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
    
    # Check for empty merge
    if df.empty:
        print("Merged DataFrame is empty! Check if keys (matrix, perm, perm_type) match in both CSVs.")
        sys.exit(1)
        
    return df, df_analysis

def filter_one_per_family(df, matrices_list_path):
    """Filters the DataFrame to keep only one matrix per family."""
    print("Filtering: One matrix per family...")
    try:
        matrix_to_family = {}
        if Path(matrices_list_path).exists():
            with open(matrices_list_path, 'r') as f:
                for line in f:
                    path = line.strip()
                    if not path: continue
                    path = path.replace('\\', '/')
                    parts = path.split('/')
                    if len(parts) >= 3:
                        matrix_name = parts[-1]
                        family = parts[-3]
                        matrix_to_family[matrix_name] = family
        else:
            print(f"Warning: Matrices list file {matrices_list_path} not found.")

        if not matrix_to_family:
            print("Warning: No matrix mapping found or file empty. Skipping family filtering.")
            return df

        df['family'] = df['matrix'].map(matrix_to_family)
        
        unique_matrices = df[['matrix', 'family']].drop_duplicates()
        unique_matrices = unique_matrices.sort_values('matrix')
        unique_matrices['family'] = unique_matrices['family'].fillna(unique_matrices['matrix'])
        
        selected_matrices = unique_matrices.groupby('family').first()['matrix'].tolist()
        
        print(f"Selected {len(selected_matrices)} matrices representing {len(selected_matrices)} families.")
        df = df[df['matrix'].isin(selected_matrices)]
        print(f"DataFrame has {len(df)} rows after family filtering.")
        return df
            
    except Exception as e:
        print(f"Error during family filtering: {e}")
        return df

def filter_trivial_matrices(df, df_analysis):
    """Filters out matrices with original bandwidth < 5."""
    if 'bandwidth_max' in df_analysis.columns:
        trivial_matrices = df_analysis[
            (df_analysis['perm'] == 'None') & 
            (df_analysis['bandwidth_max'] < 5)
        ]['matrix'].unique()
        
        if len(trivial_matrices) > 0:
            print(f"Filtering out {len(trivial_matrices)} trivial matrices (original bandwidth < 5): {trivial_matrices}")
            df = df[~df['matrix'].isin(trivial_matrices)]
            print(f"DataFrame has {len(df)} rows after filtering.")
    return df

def calculate_metrics(df):
    """Calculates GFLOPS and creates kernel_id."""
    if 'time_operation_ms' not in df.columns or 'nnz' not in df.columns:
        print("Missing required columns (time_operation_ms, nnz).")
        sys.exit(1)

    if 'n_cols' in df.columns:
        df['n_cols'] = pd.to_numeric(df['n_cols'], errors='coerce').fillna(32)
    else:
        df['n_cols'] = 32
        
    df['gflops'] = (2 * df['nnz'] * df['n_cols']) / (df['time_operation_ms'] * 1e-3) / 1e9
    
    # Create kernel_id by stripping reordering suffixes from algo
    # Suffixes: _NO_REORDER, _ROW, _SYMMETRIC, _ASYMMETRIC
    def get_kernel_id(row):
        algo = row['algo']
        # Remove reordering suffixes
        algo = re.sub(r'_(NO_REORDER|ROW|SYMMETRIC|ASYMMETRIC)', '', algo)
        
        if pd.notnull(row.get('block_size')) and row['block_size'] > 0:
            return f"{algo}_bs{int(row['block_size'])}"
        else:
            return algo

    df['kernel_id'] = df.apply(get_kernel_id, axis=1)
    print("Unique kernels found:", df['kernel_id'].unique())
        
    return df

def get_density_columns(df):
    """Identifies density columns."""
    density_cols = [c for c in df.columns if c.startswith('block_density_')]
    if 'density' in df.columns:
        density_cols.append('density')
    return density_cols

def add_ref_line(ax, x_data, y_data):
    """Adds a proportional reference line (y ~ x) to the plot."""
    if len(x_data) > 0:
        x_mid = np.median(x_data)
        y_mid = np.median(y_data)
        k = y_mid / x_mid
        
        x_range = np.logspace(np.log10(x_data.min()), np.log10(x_data.max()), 100)
        y_range = k * x_range
        
        ax.plot(x_range, y_range, 'r--', alpha=0.5, label='Proportional (y ~ x)')
        ax.legend()

def plot_gflops_vs_density(df, out_dir):
    """Generates GFLOPS vs Density scatter plots."""
    density_cols = get_density_columns(df)
    if not density_cols:
        print("No density columns found.")
        return

    unique_kernels = df['kernel_id'].unique()
    
    for kernel in unique_kernels:
        kernel_data = df[df['kernel_id'] == kernel]
        
        # Filter density columns based on operation block size
        current_density_cols = []
        bs_match = re.search(r"_bs(\d+)", kernel)
        if bs_match:
            bs_val = bs_match.group(1)
            target_col = f"block_density_{bs_val}"
            if target_col in density_cols:
                current_density_cols = [target_col]
        
        if not current_density_cols:
            current_density_cols = density_cols

        for dens_col in current_density_cols:
            if dens_col == 'density':
                bs = "1"
            else:
                try:
                    bs = dens_col.split('_')[-1]
                except:
                    bs = "unknown"
                
            safe_kernel_name = kernel.replace('/', '_').replace(' ', '_')

            # 1. Original Matrices Only
            # Filter for perm='None' (Original)
            original_data = kernel_data[kernel_data['perm'] == 'None']
            if not original_data.empty:
                plt.figure(figsize=(8, 8)) 
                ax = sns.scatterplot(
                    data=original_data,
                    x=dens_col,
                    y='gflops',
                    alpha=0.7
                )
                add_ref_line(ax, original_data[dens_col], original_data['gflops'])
                
                plt.title(f"GFLOPS vs Density {bs} ({kernel}) - Original Matrices")
                plt.xlabel(f"Density (Block Size {bs})")
                plt.ylabel("GFLOPS")
                plt.xscale('log')
                plt.yscale('log')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(out_dir / f"gflops_vs_density{bs}_{safe_kernel_name}_original.png")
                plt.close()
                print(f"Generated gflops_vs_density{bs}_{safe_kernel_name}_original.png")

            # 2. All Matrices
            if not kernel_data.empty:
                plt.figure(figsize=(8, 8))
                ax = sns.scatterplot(
                    data=kernel_data,
                    x=dens_col,
                    y='gflops',
                    hue='perm_type',
                    style='perm_type',
                    alpha=0.7
                )
                add_ref_line(ax, kernel_data[dens_col], kernel_data['gflops'])
                
                plt.title(f"GFLOPS vs Density {bs} ({kernel}) - All Matrices")
                plt.xlabel(f"Density (Block Size {bs})")
                plt.ylabel("GFLOPS")
                plt.xscale('log')
                plt.yscale('log')
                plt.grid(True, which="both", ls="-", alpha=0.2)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(out_dir / f"gflops_vs_density{bs}_{safe_kernel_name}_all.png")
                plt.close()
                print(f"Generated gflops_vs_density{bs}_{safe_kernel_name}_all.png")

def plot_gflops_distribution(df, out_dir):
    """Generates Violin plots for GFLOPS distribution."""
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    strategies = sorted([s for s in df['strategy'].unique() if s != 'Original'])
    if 'Original' in df['strategy'].unique():
        order = ['Original'] + strategies
    else:
        order = strategies
    
    df['strategy'] = pd.Categorical(df['strategy'], categories=order, ordered=True)
    unique_kernels = df['kernel_id'].unique()

    for kernel in unique_kernels:
        kernel_data = df[df['kernel_id'] == kernel]
        if kernel_data.empty: continue
        
        safe_kernel_name = kernel.replace('/', '_').replace(' ', '_')

        plt.figure(figsize=(12, 8))
        sns.violinplot(
            data=kernel_data, 
            x='strategy', 
            y='gflops', 
            order=order,
            palette="Set2",
            inner="quartile",
            cut=0
        )
        sns.stripplot(
            data=kernel_data, 
            x='strategy', 
            y='gflops', 
            order=order,
            color='black', 
            alpha=0.3, 
            jitter=True,
            size=3
        )
        
        plt.title(f"GFLOPS Distribution by Strategy ({kernel})", fontsize=14)
        plt.ylabel("GFLOPS", fontsize=12)
        plt.xlabel("Reordering Strategy", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / f"gflops_violin_{safe_kernel_name}.png", dpi=300)
        plt.close()
        print(f"Generated gflops_violin_{safe_kernel_name}.png")

def plot_speedup_distribution(df, out_dir):
    """Generates Speedup boxplots."""
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    strategies = sorted([s for s in df['strategy'].unique() if s != 'Original'])
    if 'Original' in df['strategy'].unique():
        order = ['Original'] + strategies
    else:
        order = strategies

    # Calculate baseline GFLOPS per matrix and kernel
    # We need to be careful here: 'Original' strategy might appear multiple times if we have multiple runs
    # But typically for a given kernel and matrix, there is one original run.
    # However, since we merged kernels (e.g. NO_REORDER and ROW), we might have duplicates if both experiments ran the original matrix?
    # Usually NO_REORDER runs original, ROW runs reordered.
    # But sometimes ROW experiment might include original matrix as baseline?
    # Let's take the mean if there are duplicates for 'Original'
    
    original_gflops = df[df['strategy'] == 'Original'].groupby(['matrix', 'kernel_id'])['gflops'].mean().reset_index()
    original_gflops = original_gflops.rename(columns={'gflops': 'gflops_original'})
    
    df_speedup = pd.merge(df, original_gflops, on=['matrix', 'kernel_id'], how='left')
    df_speedup['speedup'] = df_speedup['gflops'] / df_speedup['gflops_original']
    
    reordered_data = df_speedup[df_speedup['strategy'] != 'Original']
    unique_kernels = df['kernel_id'].unique()
    
    if not reordered_data.empty:
        for kernel in unique_kernels:
            kernel_data = reordered_data[reordered_data['kernel_id'] == kernel]
            if kernel_data.empty: continue
            if kernel_data['speedup'].notna().sum() == 0: continue

            plt.figure(figsize=(12, 8))
            op_strategies = [s for s in order if s in kernel_data['strategy'].unique()]
            
            sns.boxplot(
                data=kernel_data, 
                x='strategy', 
                y='speedup', 
                order=op_strategies,
                showfliers=False,
                palette="Set2"
            )
            sns.stripplot(
                data=kernel_data, 
                x='strategy', 
                y='speedup', 
                order=op_strategies,
                color='black', 
                alpha=0.3, 
                jitter=True,
                size=4
            )
            
            plt.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Baseline (Original)')
            
            plt.title(f"Speedup Distribution by Strategy ({kernel})", fontsize=14)
            plt.ylabel("Speedup (vs Original)", fontsize=12)
            plt.xlabel("Reordering Strategy", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            safe_kernel_name = kernel.replace('/', '_').replace(' ', '_')
            plt.savefig(out_dir / f"speedup_boxplot_{safe_kernel_name}.png", dpi=300)
            plt.close()
            print(f"Generated speedup_boxplot_{safe_kernel_name}.png")
