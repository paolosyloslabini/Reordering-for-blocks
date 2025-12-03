import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")
    parser.add_argument("csv_file", help="Path to results_combined.csv")
    parser.add_argument("--out", default="plots", help="Directory for output plot files")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(args.csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    if df.empty:
        print("CSV is empty.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {args.csv_file}")

    # Basic cleanup
    # Ensure numeric columns are numeric
    numeric_cols = ['time_operation_ms', 'block_density', 'bandwidth_avg', 'locality_avg_row_spread', 'nnz', 'rows', 'cols']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Set style
    sns.set_theme(style="whitegrid")
    
    # --- Plot: GFLOPS vs Block Density (Single Operator) ---
    # Calculate GFLOPS: 2 * NNZ * N_COLS / (Time_ms * 1e-3) / 1e9
    # Assuming N_COLS is 32 (default) unless specified otherwise.
    
    if 'time_operation_ms' in df.columns and 'nnz' in df.columns and 'block_density' in df.columns:
        # Add GFLOPS column
        # Use n_cols from CSV if available, else default to 32
        if 'n_cols' in df.columns:
            n_cols = df['n_cols']
        else:
            n_cols = 32
            
        df['gflops'] = (2 * df['nnz'] * n_cols) / (df['time_operation_ms'] * 1e-3) / 1e9
        
        # Filter for BSR/SMAT algorithms
        block_algos = df[df['algo'].str.contains('BSR|SMAT', case=False, regex=True, na=False)]
        
        if not block_algos.empty:
            # Get unique algorithms to plot separately
            unique_algos = block_algos['algo'].unique()
            
            for algo in unique_algos:
                algo_data = block_algos[block_algos['algo'] == algo]
                
                # 1. Original Matrices Only (perm == 'None')
                original_data = algo_data[algo_data['perm'].astype(str) == 'None']
                
                if not original_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        data=original_data,
                        x='block_density',
                        y='gflops',
                        alpha=0.7
                    )
                    plt.title(f"GFLOPS vs Block Density ({algo}) - Original Matrices")
                    plt.xlabel("Block Density")
                    plt.ylabel("GFLOPS")
                    plt.xscale('log')
                    plt.tight_layout()
                    safe_algo_name = algo.replace('/', '_').replace(' ', '_')
                    plt.savefig(out_dir / f"gflops_vs_density_{safe_algo_name}_original.png")
                    plt.close()
                    print(f"Generated gflops_vs_density_{safe_algo_name}_original.png")
                
                # 2. All Matrices (Original + Reordered)
                if not algo_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(
                        data=algo_data,
                        x='block_density',
                        y='gflops',
                        alpha=0.5 # Lower alpha since there are more points
                    )
                    plt.title(f"GFLOPS vs Block Density ({algo}) - All Matrices")
                    plt.xlabel("Block Density")
                    plt.ylabel("GFLOPS")
                    plt.xscale('log')
                    plt.tight_layout()
                    safe_algo_name = algo.replace('/', '_').replace(' ', '_')
                    plt.savefig(out_dir / f"gflops_vs_density_{safe_algo_name}_all.png")
                    plt.close()
                    print(f"Generated gflops_vs_density_{safe_algo_name}_all.png")

    print(f"All plots saved to {out_dir}")

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()