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
        
        # Use all available density columns to provide comprehensive views
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
    """Generates Violin plots for GFLOPS distribution, separated by perm_type."""
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    strategies = sorted([s for s in df['strategy'].unique() if s != 'Original'])
    if 'Original' in df['strategy'].unique():
        order = ['Original'] + strategies
    else:
        order = strategies
    
    df['strategy'] = pd.Categorical(df['strategy'], categories=order, ordered=True)
    unique_kernels = df['kernel_id'].unique()

    for kernel in unique_kernels:
        kernel_data_full = df[df['kernel_id'] == kernel]
        if kernel_data_full.empty: continue
        
        safe_kernel_name = kernel.replace('/', '_').replace(' ', '_')

        # Identify reordering types present (excluding Original/None)
        reordered_slice = kernel_data_full[kernel_data_full['perm'] != 'None']
        
        # If no reordered data, maybe just plot Original? 
        # Or if we want to be strict about separating types, we can treat 'Original' as a type if it's the only one.
        # But usually we want to see ROW, SYMMETRIC, etc.
        if reordered_slice.empty:
            perm_types = [] # Skip if no reordering to show distribution for? Or just show Original?
            # Let's show Original if it exists
            if not kernel_data_full.empty:
                perm_types = ['Original']
        else:
            perm_types = reordered_slice['perm_type'].unique()

        for p_type in perm_types:
            if p_type == 'Original':
                plot_data = kernel_data_full[kernel_data_full['perm'] == 'None']
                suffix = "_original"
                current_order = ['Original']
            else:
                # Data for this perm_type + Original data
                # We filter by (perm_type == p_type) OR (perm == 'None')
                plot_data = kernel_data_full[
                    (kernel_data_full['perm_type'] == p_type) | 
                    (kernel_data_full['perm'] == 'None')
                ]
                suffix = f"_{p_type}"
                # Filter order to only include relevant strategies
                current_strategies = plot_data['strategy'].unique()
                current_order = [s for s in order if s in current_strategies]

            if plot_data.empty: continue

            plt.figure(figsize=(12, 8))
            sns.violinplot(
                data=plot_data, 
                x='strategy', 
                y='gflops', 
                order=current_order,
                palette="Set2",
                inner="quartile",
                cut=0
            )
            sns.stripplot(
                data=plot_data, 
                x='strategy', 
                y='gflops', 
                order=current_order,
                color='black', 
                alpha=0.3, 
                jitter=True,
                size=3
            )
            
            plt.title(f"GFLOPS Distribution by Strategy ({kernel}) - {p_type}", fontsize=14)
            plt.ylabel("GFLOPS", fontsize=12)
            plt.xlabel("Reordering Strategy", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(out_dir / f"gflops_violin_{safe_kernel_name}{suffix}.png", dpi=300)
            plt.close()
            print(f"Generated gflops_violin_{safe_kernel_name}{suffix}.png")

def plot_speedup_distribution(df, out_dir):
    """Generates Speedup plots (Boxplot, CDF, KDE), separated by perm_type."""
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    strategies = sorted([s for s in df['strategy'].unique() if s != 'Original'])
    if 'Original' in df['strategy'].unique():
        order = ['Original'] + strategies
    else:
        order = strategies

    # Calculate baseline GFLOPS per matrix and kernel
    original_gflops = df[df['strategy'] == 'Original'].groupby(['matrix', 'kernel_id'])['gflops'].mean().reset_index()
    original_gflops = original_gflops.rename(columns={'gflops': 'gflops_original'})
    
    df_speedup = pd.merge(df, original_gflops, on=['matrix', 'kernel_id'], how='left')
    df_speedup['speedup'] = df_speedup['gflops'] / df_speedup['gflops_original']
    
    reordered_data = df_speedup[df_speedup['strategy'] != 'Original']
    unique_kernels = df['kernel_id'].unique()
    
    if not reordered_data.empty:
        for kernel in unique_kernels:
            kernel_reordered = reordered_data[reordered_data['kernel_id'] == kernel]
            if kernel_reordered.empty: 
                print(f"Skipping {kernel}: No reordered data found.")
                continue
            
            # Iterate over perm_types (ROW, SYMMETRIC, ASYMMETRIC)
            unique_perm_types = kernel_reordered['perm_type'].unique()
            
            for p_type in unique_perm_types:
                kernel_data = kernel_reordered[kernel_reordered['perm_type'] == p_type]
                
                valid_speedups = kernel_data['speedup'].notna().sum()
                if valid_speedups == 0: 
                    print(f"Skipping {kernel} ({p_type}): No valid speedup data.")
                    continue
                
                print(f"Plotting speedup for {kernel} - {p_type} ({valid_speedups} data points)...")

                op_strategies = [s for s in order if s in kernel_data['strategy'].unique()]
                safe_kernel_name = kernel.replace('/', '_').replace(' ', '_')
                suffix = f"_{p_type}"

                # 1. Boxplot
                plt.figure(figsize=(12, 8))
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
                plt.title(f"Speedup Distribution by Strategy ({kernel}) - {p_type}", fontsize=14)
                plt.ylabel("Speedup (vs Original)", fontsize=12)
                plt.xlabel("Reordering Strategy", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / f"speedup_boxplot_{safe_kernel_name}{suffix}.png", dpi=300)
                plt.close()
                print(f"Generated speedup_boxplot_{safe_kernel_name}{suffix}.png")

                # 2. CDF Plot
                plt.figure(figsize=(10, 6))
                for strategy in op_strategies:
                    subset = kernel_data[kernel_data['strategy'] == strategy]
                    speedups = subset['speedup'].dropna().sort_values()
                    if len(speedups) == 0: continue
                    y = np.arange(1, len(speedups) + 1) / len(speedups)
                    plt.step(speedups, y, label=strategy, where='post', linewidth=2)
                
                plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
                plt.xlabel('Speedup (vs Original)', fontsize=12)
                plt.ylabel('CDF (Fraction of Matrices)', fontsize=12)
                plt.title(f'Speedup CDF ({kernel}) - {p_type}', fontsize=14)
                plt.legend(title='Strategy')
                plt.grid(True, alpha=0.3)
                plt.xscale('log')
                plt.tight_layout()
                plt.savefig(out_dir / f"speedup_cdf_{safe_kernel_name}{suffix}.png", dpi=300)
                plt.close()
                print(f"Generated speedup_cdf_{safe_kernel_name}{suffix}.png")

                # 3. Histogram Plot
                plt.figure(figsize=(10, 6))
                try:
                    sns.histplot(
                        data=kernel_data, 
                        x='speedup', 
                        hue='strategy', 
                        hue_order=op_strategies,
                        common_norm=False, 
                        stat="percent",
                        element="step",
                        fill=True, 
                        alpha=0.2,
                        palette="Set2"
                    )
                    plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
                    plt.xlabel('Speedup (vs Original)', fontsize=12)
                    plt.ylabel('Percentage of Matrices (%)', fontsize=12)
                    plt.title(f'Speedup Distribution ({kernel}) - {p_type}', fontsize=14)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"speedup_hist_{safe_kernel_name}{suffix}.png", dpi=300)
                    plt.close()
                    print(f"Generated speedup_hist_{safe_kernel_name}{suffix}.png")
                except Exception as e:
                    print(f"Could not generate Histogram plot for {kernel} ({p_type}): {e}")

def plot_reordering_efficiency(df, out_dir):
    """Generates plots for Bandwidth Reduction and Block Density Improvement."""
    print("Generating Reordering Efficiency plots...")
    
    # 1. Prepare Data
    # We only need analysis columns. Drop duplicates to get unique matrix+perm combinations.
    analysis_cols = [c for c in df.columns if c.startswith('bandwidth') or c.startswith('block_density')]
    if not analysis_cols:
        print("No analysis columns found.")
        return

    # Unique matrix+perm configurations
    df_unique = df[['matrix', 'perm', 'perm_type'] + analysis_cols].drop_duplicates()
    
    # Define Strategy
    df_unique['strategy'] = df_unique['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    
    # 2. Calculate Baselines (Original)
    # Handle duplicates in Original if any (take mean)
    original_df = df_unique[df_unique['strategy'] == 'Original'].groupby('matrix')[analysis_cols].mean()
    
    if original_df.empty:
        print("No Original matrix data found for baseline.")
        return

    # 3. Calculate Improvements
    results = []
    
    for idx, row in df_unique.iterrows():
        if row['strategy'] == 'Original': continue
        
        matrix = row['matrix']
        if matrix not in original_df.index: continue
        
        orig = original_df.loc[matrix]
        
        res = {
            'matrix': matrix,
            'strategy': row['strategy'],
            'perm_type': row['perm_type']
        }
        
        # Bandwidth Improvement (Reduction) = Orig / Reordered
        if 'bandwidth_max' in row and 'bandwidth_max' in orig:
            if row['bandwidth_max'] > 0:
                res['bandwidth_improvement'] = orig['bandwidth_max'] / row['bandwidth_max']
            else:
                res['bandwidth_improvement'] = np.nan

        # Block Density Improvement = Reordered / Orig
        for col in analysis_cols:
            if col.startswith('block_density_'):
                bs = col.split('_')[-1]
                if orig[col] > 0:
                    res[f'density_improvement_{bs}'] = row[col] / orig[col]
                else:
                    res[f'density_improvement_{bs}'] = np.nan
        
        results.append(res)
        
    if not results:
        print("No reordered data to analyze.")
        return
        
    df_res = pd.DataFrame(results)
    
    # 4. Plotting
    strategies = sorted(df_res['strategy'].unique())
    
    def plot_dist(data, x, y, title, filename_suffix):
        # 1. Boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=data, x=x, y=y, order=strategies, showfliers=False, palette="Set2")
        sns.stripplot(data=data, x=x, y=y, order=strategies, color='black', alpha=0.3, jitter=True, size=3)
        plt.axhline(1.0, color='r', linestyle='--', label='Baseline')
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_suffix}_boxplot.png", dpi=300)
        plt.close()
        print(f"Generated {filename_suffix}_boxplot.png")

        # 2. CDF Plot
        plt.figure(figsize=(10, 6))
        current_strategies = [s for s in strategies if s in data[x].unique()]
        for strategy in current_strategies:
            subset = data[data[x] == strategy]
            values = subset[y].dropna().sort_values()
            if len(values) == 0: continue
            cdf_y = np.arange(1, len(values) + 1) / len(values)
            plt.step(values, cdf_y, label=strategy, where='post', linewidth=2)
        
        plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
        plt.xlabel(f"{title.split(' - ')[0]}", fontsize=12)
        plt.ylabel('CDF (Fraction of Matrices)', fontsize=12)
        plt.title(f"CDF: {title}", fontsize=14)
        plt.legend(title='Strategy')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(out_dir / f"{filename_suffix}_cdf.png", dpi=300)
        plt.close()
        print(f"Generated {filename_suffix}_cdf.png")

        # 3. Histogram Plot
        plt.figure(figsize=(10, 6))
        try:
            sns.histplot(
                data=data, 
                x=y, 
                hue=x, 
                hue_order=current_strategies,
                common_norm=False, 
                stat="percent",
                element="step",
                fill=True, 
                alpha=0.2,
                palette="Set2"
            )
            plt.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='Baseline')
            plt.xlabel(f"{title.split(' - ')[0]}", fontsize=12)
            plt.ylabel('Percentage of Matrices (%)', fontsize=12)
            plt.title(f"Distribution: {title}", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"{filename_suffix}_hist.png", dpi=300)
            plt.close()
            print(f"Generated {filename_suffix}_hist.png")
        except Exception as e:
            print(f"Could not generate Histogram plot for {filename_suffix}: {e}")

    # Plot Bandwidth Improvement
    if 'bandwidth_improvement' in df_res.columns:
        for p_type in df_res['perm_type'].unique():
            subset = df_res[df_res['perm_type'] == p_type]
            if subset.empty: continue
            # Filter NaNs
            subset = subset.dropna(subset=['bandwidth_improvement'])
            if subset.empty: continue
            
            plot_dist(subset, 'strategy', 'bandwidth_improvement', 
                      f"Bandwidth Reduction (Orig/Reordered) - {p_type}", 
                      f"bandwidth_reduction_{p_type}")

    # Plot Density Improvement for each block size
    density_cols = [c for c in df_res.columns if c.startswith('density_improvement_')]
    for col in density_cols:
        bs = col.split('_')[-1]
        for p_type in df_res['perm_type'].unique():
            subset = df_res[df_res['perm_type'] == p_type]
            if subset.empty: continue
            subset = subset.dropna(subset=[col])
            if subset.empty: continue
            
            plot_dist(subset, 'strategy', col, 
                      f"Block Density Improvement (BS {bs}) - {p_type}", 
                      f"density_improvement_bs{bs}_{p_type}")
