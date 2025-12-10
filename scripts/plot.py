import argparse
from pathlib import Path
import seaborn as sns
import plot_utils

def main():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")
    parser.add_argument("--operations", default="results/results_operations.csv", help="Path to results_operations.csv")
    parser.add_argument("--analysis", default="results/results_analysis.csv", help="Path to results_analysis.csv")
    parser.add_argument("--out", default="plots", help="Directory for output plot files")
    parser.add_argument("--one-per-family", action="store_true", help="Only include one matrix per family/group")
    parser.add_argument("--matrices-list", default="datasets/matrices_list_mtx.txt", help="Path to matrices list file for family mapping")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and Merge Data
    df, df_analysis = plot_utils.load_and_merge_data(args.operations, args.analysis)

    # 2. Filter: One per family
    if args.one_per_family:
        print("Filtering Operations DataFrame:")
        df = plot_utils.filter_one_per_family(df, args.matrices_list)
        print("Filtering Analysis DataFrame:")
        df_analysis = plot_utils.filter_one_per_family(df_analysis, args.matrices_list)

    # 3. Filter: Trivial matrices
    # We need to filter both df and df_analysis
    # First, identify trivial matrices using df_analysis
    trivial_matrices = []
    if 'bandwidth_max' in df_analysis.columns:
        trivial_matrices = df_analysis[
            (df_analysis['perm'] == 'None') & 
            (df_analysis['bandwidth_max'] < 5)
        ]['matrix'].unique()
    
    if len(trivial_matrices) > 0:
        print(f"Filtering out {len(trivial_matrices)} trivial matrices (bandwidth < 5): {list(trivial_matrices)}")
        df = df[~df['matrix'].isin(trivial_matrices)]
        df_analysis = df_analysis[~df_analysis['matrix'].isin(trivial_matrices)]

    # 4. Filter: Sparse matrices (nnz < 3*N)
    # Identify matrices where nnz < 3 * rows (too sparse to benefit from reordering)
    sparse_matrices = []
    if 'nnz' in df_analysis.columns and 'rows' in df_analysis.columns:
        sparse_matrices = df_analysis[
            (df_analysis['perm'] == 'None') & 
            (df_analysis['nnz'] < 3 * df_analysis['rows'])
        ]['matrix'].unique()
    
    if len(sparse_matrices) > 0:
        print(f"Filtering out {len(sparse_matrices)} very sparse matrices (nnz < 3N): {list(sparse_matrices)}")
        df = df[~df['matrix'].isin(sparse_matrices)]
        df_analysis = df_analysis[~df_analysis['matrix'].isin(sparse_matrices)]

    # 5. Calculate Metrics (GFLOPS, op_id)
    df = plot_utils.calculate_metrics(df)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 5. Generate Plots per n_cols and routine
    unique_n_cols = sorted(df['n_cols'].unique())
    print(f"Found n_cols values: {unique_n_cols}")

    for n_cols in unique_n_cols:
        print(f"\n--- Processing n_cols = {n_cols} ---")
        
        # Filter data for this n_cols
        df_ncols = df[df['n_cols'] == n_cols].copy()
        
        if df_ncols.empty:
            print(f"No data for n_cols={n_cols}, skipping.")
            continue

        # Get unique routines (kernel_ids)
        unique_routines = sorted(df_ncols['kernel_id'].unique())
        print(f"Found routines: {unique_routines}")

        for routine in unique_routines:
            print(f"\nGenerating plots for routine: {routine} (n_cols={n_cols})")
            
            # Create subdirectory: plots/n_cols_X/routine_name
            # Sanitize routine name for folder
            safe_routine = routine.replace('/', '_').replace(' ', '_')
            sub_dir = out_dir / f"n_cols_{int(n_cols)}" / safe_routine
            sub_dir.mkdir(parents=True, exist_ok=True)
            
            # Filter data for this routine
            df_routine = df_ncols[df_ncols['kernel_id'] == routine].copy()
            
            if df_routine.empty:
                continue

            print("Generating GFLOPS vs Density plots...")
            plot_utils.plot_gflops_vs_density(df_routine, sub_dir)
            
            print("Generating GFLOPS Distribution plots...")
            plot_utils.plot_gflops_distribution(df_routine, sub_dir)
            
            print("Generating Speedup Distribution plots...")
            plot_utils.plot_speedup_distribution(df_routine, sub_dir)

    # 6. Generate Reordering Analysis Plots
    # These plots analyze the reordering quality itself (bandwidth, density) independent of SpMM performance.
    # We use df_analysis directly to include results (like SYMMETRIC) that might be missing from operations.
    print("\n--- Generating Reordering Analysis Plots ---")
    reorder_dir = out_dir / "reorder_analysis"
    reorder_dir.mkdir(parents=True, exist_ok=True)
    plot_utils.plot_reordering_efficiency(df_analysis, reorder_dir)

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()
