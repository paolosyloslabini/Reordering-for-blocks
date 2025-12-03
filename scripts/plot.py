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
    
    # Debug: Check for merge issues
    if df.empty:
        print("Merged DataFrame is empty! Check if keys (matrix, perm, perm_type) match in both CSVs.")
        print("SpMM keys sample:", df_spmm[['matrix', 'perm', 'perm_type']].head().to_dict('records'))
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

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()