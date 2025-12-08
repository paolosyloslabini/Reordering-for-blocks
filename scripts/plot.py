import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")
    parser.add_argument("--operations", default="results/results_operations.csv", help="Path to results_operations.csv")
    parser.add_argument("--analysis", default="results/results_analysis.csv", help="Path to results_analysis.csv")
    parser.add_argument("--out", default="plots", help="Directory for output plot files")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df_op = pd.read_csv(args.operations)
        df_analysis = pd.read_csv(args.analysis)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)

    if df_op.empty or df_analysis.empty:
        print("One of the CSVs is empty.")
        sys.exit(1)

    print(f"Loaded {len(df_op)} operation rows and {len(df_analysis)} analysis rows.")

    # Merge dataframes
    # Keys: matrix, perm, perm_type
    # Note: perm can be 'None' (string) or NaN. Let's normalize.
    
    for df in [df_op, df_analysis]:
        df['perm'] = df['perm'].fillna('None').astype(str)
        df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)
        df['matrix'] = df['matrix'].astype(str)

    # Merge
    df = pd.merge(df_op, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')
    print(f"Merged DataFrame has {len(df)} rows.")
    
    # Debug: Check for merge issues
    if df.empty:
        print("Merged DataFrame is empty! Check if keys (matrix, perm, perm_type) match in both CSVs.")
        print("Operation keys sample:", df_op[['matrix', 'perm', 'perm_type']].head().to_dict('records'))
        print("Analysis keys sample:", df_analysis[['matrix', 'perm', 'perm_type']].head().to_dict('records'))
        return

    # Basic cleanup
    if 'time_operation_ms' not in df.columns or 'nnz' not in df.columns:
        print("Missing required columns (time_operation_ms, nnz).")
        print("Columns found:", df.columns.tolist())
        return

    # Calculate GFLOPS
    # Use n_cols from CSV if available, else default to 32
    if 'n_cols' in df.columns:
        df['n_cols'] = pd.to_numeric(df['n_cols'], errors='coerce').fillna(32)
    else:
        df['n_cols'] = 32
        
    df['gflops'] = (2 * df['nnz'] * df['n_cols']) / (df['time_operation_ms'] * 1e-3) / 1e9
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Plot: GFLOPS vs Block Density ---
    
    # Create a unique identifier for each operation configuration
    # Combine algo and block_size (if present)
    if 'block_size' in df.columns:
        df['op_id'] = df.apply(
            lambda row: f"{row['algo']}_bs{int(row['block_size'])}" if pd.notnull(row['block_size']) and row['block_size'] > 0 else row['algo'], 
            axis=1
        )
    else:
        df['op_id'] = df['algo']

    if 'op_id' in df.columns:
        print("Unique operations found:", df['op_id'].unique())
    
    # Let's find all block_density_* columns
    density_cols = [c for c in df.columns if c.startswith('block_density_')]
    print(f"Found density columns: {density_cols}")
    
    if not density_cols:
        print("No block_density columns found in analysis data.")
        return

    unique_ops = df['op_id'].unique()
    
    for op in unique_ops:
        op_data = df[df['op_id'] == op]
        
        # For each available block density metric, generate a plot
        for dens_col in density_cols:
            # Extract block size from column name (block_density_32 -> 32)
            try:
                bs = dens_col.split('_')[-1]
            except:
                bs = "unknown"
                
            # 1. Original Matrices Only
            original_data = op_data[op_data['perm'] == 'None']
            if not original_data.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    data=original_data,
                    x=dens_col,
                    y='gflops',
                    alpha=0.7
                )
                plt.title(f"GFLOPS vs Block Density {bs} ({op}) - Original Matrices")
                plt.xlabel(f"Block Density (Block Size {bs})")
                plt.ylabel("GFLOPS")
                plt.xscale('log')
                plt.tight_layout()
                safe_op_name = op.replace('/', '_').replace(' ', '_')
                plt.savefig(out_dir / f"gflops_vs_density{bs}_{safe_op_name}_original.png")
                plt.close()
                print(f"Generated gflops_vs_density{bs}_{safe_op_name}_original.png")

            # 2. All Matrices
            if not op_data.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    data=op_data,
                    x=dens_col,
                    y='gflops',
                    alpha=0.5
                )
                plt.title(f"GFLOPS vs Block Density {bs} ({op}) - All Matrices")
                plt.xlabel(f"Block Density (Block Size {bs})")
                plt.ylabel("GFLOPS")
                plt.xscale('log')
                plt.tight_layout()
                safe_op_name = op.replace('/', '_').replace(' ', '_')
                plt.savefig(out_dir / f"gflops_vs_density{bs}_{safe_op_name}_all.png")
                plt.close()
                print(f"Generated gflops_vs_density{bs}_{safe_op_name}_all.png")

    # --- Plot: GFLOPS Distribution (Violin & Paired Plots) ---
    # Compare Original vs Different Permutation Strategies
    
    # Define strategy column
    # If perm is 'None', it's 'Original'. Otherwise, use the perm name (e.g. 'SB_amd', 'random1D')
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    
    # Sort order: Original first, then others alphabetically
    strategies = sorted([s for s in df['strategy'].unique() if s != 'Original'])
    if 'Original' in df['strategy'].unique():
        order = ['Original'] + strategies
    else:
        order = strategies
        
    # Convert to categorical to ensure correct plotting order
    df['strategy'] = pd.Categorical(df['strategy'], categories=order, ordered=True)

    for op in unique_ops:
        op_data = df[df['op_id'] == op]
        if op_data.empty: continue
        
        safe_op_name = op.replace('/', '_').replace(' ', '_')

        # 1. Violin Plot
        plt.figure(figsize=(12, 8))
        sns.violinplot(
            data=op_data, 
            x='strategy', 
            y='gflops', 
            order=order,
            palette="Set2",
            inner="quartile",
            cut=0
        )
        # Add strip plot for individual points
        sns.stripplot(
            data=op_data, 
            x='strategy', 
            y='gflops', 
            order=order,
            color='black', 
            alpha=0.3, 
            jitter=True,
            size=3
        )
        
        plt.title(f"GFLOPS Distribution by Strategy ({op})", fontsize=14)
        plt.ylabel("GFLOPS", fontsize=12)
        plt.xlabel("Reordering Strategy", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_dir / f"gflops_violin_{safe_op_name}.png", dpi=300)
        plt.close()
        print(f"Generated gflops_violin_{safe_op_name}.png")

        # 2. Paired Comparison Plot (Slope Chart)
        # This visualizes how EACH matrix changes across strategies
        plt.figure(figsize=(12, 8))
        
        # Draw lines connecting the same matrix
        # We use a simple grey line for each matrix
        sns.lineplot(
            data=op_data,
            x='strategy',
            y='gflops',
            hue='matrix',
            units='matrix',
            estimator=None,
            legend=False,
            alpha=0.2,
            linewidth=1,
            palette=['gray'] * len(op_data['matrix'].unique()) # Force all lines to be gray
        )
        
        # Overlay points colored by strategy
        sns.scatterplot(
            data=op_data,
            x='strategy',
            y='gflops',
            hue='strategy',
            legend=False,
            s=40,
            zorder=10,
            palette="Set2"
        )
        
        plt.title(f"GFLOPS Comparison per Matrix ({op})", fontsize=14)
        plt.ylabel("GFLOPS (Log Scale)", fontsize=12)
        plt.xlabel("Reordering Strategy", fontsize=12)
        plt.yscale('log') # Log scale is better for comparing matrices of vastly different sizes
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / f"gflops_paired_{safe_op_name}.png", dpi=300)
        plt.close()
        print(f"Generated gflops_paired_{safe_op_name}.png")

    # --- Plot: Speedup Distribution ---
    # Calculate speedup relative to Original for each matrix
    
    # 1. Get Original GFLOPS per matrix and operation
    # We drop duplicates just in case, though there should be only one original per matrix/op
    original_gflops = df[df['strategy'] == 'Original'][['matrix', 'op_id', 'gflops']].drop_duplicates().rename(columns={'gflops': 'gflops_original'})
    
    # 2. Merge back
    df_speedup = pd.merge(df, original_gflops, on=['matrix', 'op_id'], how='left')
    
    # 3. Calculate Speedup
    df_speedup['speedup'] = df_speedup['gflops'] / df_speedup['gflops_original']
    
    # 4. Filter for Reordered rows only (exclude Original)
    reordered_data = df_speedup[df_speedup['strategy'] != 'Original']
    
    if not reordered_data.empty:
        for op in unique_ops:
            op_data = reordered_data[reordered_data['op_id'] == op]
            if op_data.empty: continue
            
            # Check if we have valid speedup data
            if op_data['speedup'].notna().sum() == 0:
                continue

            plt.figure(figsize=(12, 8))
            
            # Use strategy for x-axis
            # Filter order to only include present strategies
            op_strategies = [s for s in order if s in op_data['strategy'].unique()]
            
            sns.boxplot(
                data=op_data, 
                x='strategy', 
                y='speedup', 
                order=op_strategies,
                showfliers=False,
                palette="Set2"
            )
            sns.stripplot(
                data=op_data, 
                x='strategy', 
                y='speedup', 
                order=op_strategies,
                color='black', 
                alpha=0.3, 
                jitter=True,
                size=4
            )
            
            plt.axhline(1.0, color='r', linestyle='--', linewidth=2, label='Baseline (Original)')
            
            plt.title(f"Speedup Distribution by Strategy ({op})", fontsize=14)
            plt.ylabel("Speedup (vs Original)", fontsize=12)
            plt.xlabel("Reordering Strategy", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            safe_op_name = op.replace('/', '_').replace(' ', '_')
            plt.savefig(out_dir / f"speedup_boxplot_{safe_op_name}.png", dpi=300)
            plt.close()
            print(f"Generated speedup_boxplot_{safe_op_name}.png")

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()