import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")
    parser.add_argument("--spmm", default="results_spmm.csv", help="Path to results_spmm.csv")
    parser.add_argument("--analysis", default="results_analysis.csv", help="Path to results_analysis.csv")
    parser.add_argument("--out", default="plots", help="Directory for output plot files")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        df_spmm = pd.read_csv(args.spmm)
        df_analysis = pd.read_csv(args.analysis)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        sys.exit(1)

    if df_spmm.empty or df_analysis.empty:
        print("One of the CSVs is empty.")
        sys.exit(1)

    print(f"Loaded {len(df_spmm)} SpMM rows and {len(df_analysis)} analysis rows.")

    # Merge dataframes
    # Keys: matrix, perm, perm_type
    # Note: perm can be 'None' (string) or NaN. Let's normalize.
    
    for df in [df_spmm, df_analysis]:
        df['perm'] = df['perm'].fillna('None').astype(str)
        df['perm_type'] = df['perm_type'].fillna('UNKNOWN').astype(str)
        df['matrix'] = df['matrix'].astype(str)

    # Merge
    df = pd.merge(df_spmm, df_analysis, on=['matrix', 'perm', 'perm_type'], how='left')
    print(f"Merged DataFrame has {len(df)} rows.")

    # Basic cleanup
    if 'time_operation_ms' not in df.columns or 'nnz' not in df.columns:
        print("Missing required columns (time_operation_ms, nnz).")
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
    
    # --- Plot: GFLOPS vs Block Density (BSR/SMAT) ---
    
    # Filter for BSR/SMAT algorithms
    block_algos = df[df['algo'].str.contains('BSR|SMAT', case=False, regex=True, na=False)].copy()
    
    if not block_algos.empty:
        # We need to extract the correct block_density for each row based on block_size
        # block_size should be in the dataframe
        if 'block_size' in block_algos.columns:
            # Create a new column 'active_block_density'
            block_algos['active_block_density'] = np.nan
            
            # Iterate over unique block sizes present in the data
            unique_block_sizes = block_algos['block_size'].dropna().unique()
            
            for bs in unique_block_sizes:
                try:
                    bs_int = int(bs)
                    col_name = f'block_density_{bs_int}'
                    if col_name in block_algos.columns:
                        mask = (block_algos['block_size'] == bs)
                        block_algos.loc[mask, 'active_block_density'] = block_algos.loc[mask, col_name]
                except ValueError:
                    continue
            
            # Drop rows where we couldn't find density info
            plot_data = block_algos.dropna(subset=['active_block_density', 'gflops'])
            
            if not plot_data.empty:
                unique_algos = plot_data['algo'].unique()
                
                for algo in unique_algos:
                    algo_data = plot_data[plot_data['algo'] == algo]
                    
                    # 1. Original Matrices Only (perm == 'None')
                    original_data = algo_data[algo_data['perm'] == 'None']
                    
                    if not original_data.empty:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(
                            data=original_data,
                            x='active_block_density',
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
                    
                    # 2. All Matrices
                    if not algo_data.empty:
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(
                            data=algo_data,
                            x='active_block_density',
                            y='gflops',
                            alpha=0.5
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
        else:
            print("block_size column missing in SpMM results.")

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()