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
    
    # Filtering options
    parser.add_argument("--only-reorder-analysis", action="store_true", help="Only generate reorder analysis plots (skip kernel plots)")
    parser.add_argument("--only-kernels", action="store_true", help="Only generate kernel plots (skip reorder analysis)")
    parser.add_argument("--n-cols", type=int, default=None, help="Only process specific n_cols value (e.g., 32, 256, 1024)")
    parser.add_argument("--kernel", type=str, default=None, help="Only process specific kernel (e.g., CUSPARSE_SPMM_CSR)")
    
    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.only_reorder_analysis and args.only_kernels:
        print("Error: --only-reorder-analysis and --only-kernels are mutually exclusive.")
        return

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
    
    # 6. Generate Kernel Plots (unless --only-reorder-analysis)
    if not args.only_reorder_analysis:
        unique_n_cols = sorted(df['n_cols'].unique())
        
        # Filter n_cols if specified
        if args.n_cols is not None:
            if args.n_cols in unique_n_cols:
                unique_n_cols = [args.n_cols]
            else:
                print(f"Warning: n_cols={args.n_cols} not found in data. Available: {unique_n_cols}")
                unique_n_cols = []
        
        print(f"Processing n_cols values: {unique_n_cols}")

        for n_cols in unique_n_cols:
            print(f"\n--- Processing n_cols = {n_cols} ---")
            
            # Filter data for this n_cols
            df_ncols = df[df['n_cols'] == n_cols].copy()
            
            if df_ncols.empty:
                print(f"No data for n_cols={n_cols}, skipping.")
                continue

            # Get unique routines (kernel_ids)
            unique_routines = sorted(df_ncols['kernel_id'].unique())
            
            # Filter kernel if specified
            if args.kernel is not None:
                matching_kernels = [k for k in unique_routines if args.kernel.lower() in k.lower()]
                if matching_kernels:
                    unique_routines = matching_kernels
                else:
                    print(f"Warning: kernel '{args.kernel}' not found. Available: {unique_routines}")
                    continue
            
            print(f"Processing routines: {unique_routines}")

            for routine in unique_routines:
                print(f"\nGenerating plots for routine: {routine} (n_cols={n_cols})")
                
                # Create subdirectory structure: plots/n_cols_X/routine_name/plot_type
                safe_routine = routine.replace('/', '_').replace(' ', '_')
                base_dir = out_dir / f"n_cols_{int(n_cols)}" / safe_routine
                
                # Create plot-type subdirectories
                gflops_density_dir = base_dir / "gflops_vs_density"
                gflops_dist_dir = base_dir / "gflops_distribution"
                speedup_dir = base_dir / "speedup"
                speedup_vs_density_dir = base_dir / "speedup_vs_density"
                
                for d in [gflops_density_dir, gflops_dist_dir, speedup_dir, speedup_vs_density_dir]:
                    d.mkdir(parents=True, exist_ok=True)
                
                # Filter data for this routine
                df_routine = df_ncols[df_ncols['kernel_id'] == routine].copy()
                
                if df_routine.empty:
                    continue

                print("Generating GFLOPS vs Density plots...")
                plot_utils.plot_gflops_vs_density(df_routine, gflops_density_dir)
                
                print("Generating GFLOPS Distribution plots...")
                plot_utils.plot_gflops_distribution(df_routine, gflops_dist_dir)
                
                print("Generating Speedup Distribution plots...")
                plot_utils.plot_speedup_distribution(df_routine, speedup_dir)
                
                print("Generating Speedup vs Density Improvement plots...")
                plot_utils.plot_speedup_vs_density(df_routine, speedup_vs_density_dir)
                
                print("Generating Speedup vs Density Improvement plots (density > 1 only)...")
                plot_utils.plot_speedup_vs_density_improved_only(df_routine, speedup_vs_density_dir)

    # 7. Generate Reordering Analysis Plots (unless --only-kernels)
    if not args.only_kernels:
        # These plots analyze the reordering quality itself (bandwidth, density) independent of SpMM performance.
        # We use df_analysis directly to include results (like SYMMETRIC) that might be missing from operations.
        print("\n--- Generating Reordering Analysis Plots ---")
        reorder_dir = out_dir / "reorder_analysis"
        
        # Create subdirectories for bandwidth and density analysis
        bandwidth_dir = reorder_dir / "bandwidth"
        density_dir = reorder_dir / "density"
        bandwidth_dir.mkdir(parents=True, exist_ok=True)
        density_dir.mkdir(parents=True, exist_ok=True)
        
        plot_utils.plot_reordering_efficiency(df_analysis, reorder_dir)

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()
