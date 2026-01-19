"""
Plot Script - Clean Architecture

Simple iteration-based plotting script that:
1. Loads and processes data once
2. Applies filters
3. Iterates over dimensions (n_cols, kernel, perm_type) to generate plots
"""

import argparse
from pathlib import Path
import seaborn as sns
import plot_utils as pu


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")
    
    # Input/Output
    parser.add_argument("--operations", default="results/results_operations.csv")
    parser.add_argument("--analysis", default="results/results_analysis.csv")
    parser.add_argument("--out", default="plots", help="Output directory")
    
    # Filtering options
    parser.add_argument("--one-per-family", action="store_true")
    parser.add_argument("--matrices-list", default="datasets/matrices_list_mtx.txt")
    parser.add_argument("--include-rectangular", action="store_true")
    parser.add_argument("--min-size", type=int, default=None)
    
    # Plot selection
    parser.add_argument("--only-reorder-analysis", action="store_true")
    parser.add_argument("--only-kernels", action="store_true")
    parser.add_argument("--n-cols", type=int, default=None, help="Filter to specific n_cols")
    parser.add_argument("--kernel", type=str, default=None, help="Filter to specific kernel")
    
    return parser.parse_args()


def generate_kernel_plots(df, out_dir, args):
    """Generate performance plots for kernel operations."""
    
    # Get unique n_cols values
    n_cols_values = sorted(df['n_cols'].unique())
    if args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"Warning: n_cols={args.n_cols} not found. Available: {n_cols_values}")
            return
    
    # Get density and improvement columns
    density_cols = pu.get_density_columns(df)
    
    for n_cols in n_cols_values:
        print(f"\n=== n_cols = {n_cols} ===")
        df_nc = df[df['n_cols'] == n_cols]
        
        # Get unique kernels
        kernels = sorted(df_nc['kernel_id'].unique())
        if args.kernel:
            kernels = [k for k in kernels if args.kernel.lower() in k.lower()]
            if not kernels:
                print(f"No kernels matching '{args.kernel}'")
                continue
        
        for kernel in kernels:
            print(f"\n--- Kernel: {kernel} ---")
            df_k = df_nc[df_nc['kernel_id'] == kernel]
            
            kernel_safe = pu.safe_filename(kernel)
            base_dir = out_dir / f"n_cols_{int(n_cols)}" / kernel_safe
            
            # -----------------------------------------------------------------
            # 1. GFLOPS vs Metrics (scatter plots)
            # -----------------------------------------------------------------
            gflops_dir = base_dir / "gflops_correlations"
            gflops_dir.mkdir(parents=True, exist_ok=True)
            
            # GFLOPS vs each density column
            for dens_col in density_cols:
                bs = dens_col.split('_')[-1]
                pu.scatter_with_correlation(
                    df_k, dens_col, 'gflops',
                    gflops_dir / f"gflops_vs_density_bs{bs}.png",
                    title=f"GFLOPS vs Block Density (BS {bs})\n{kernel}"
                )
            
            # GFLOPS vs bandwidth
            if 'rel_bandwidth' in df_k.columns:
                pu.scatter_with_correlation(
                    df_k, 'rel_bandwidth', 'gflops',
                    gflops_dir / f"gflops_vs_rel_bandwidth.png",
                    title=f"GFLOPS vs Relative Bandwidth\n{kernel}"
                )
            
            # GFLOPS vs locality
            for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio']:
                if loc_col in df_k.columns:
                    pu.scatter_with_correlation(
                        df_k, loc_col, 'gflops',
                        gflops_dir / f"gflops_vs_{loc_col}.png",
                        title=f"GFLOPS vs {pu.get_display_name(loc_col)}\n{kernel}"
                    )
            
            # -----------------------------------------------------------------
            # 2. Speedup Distribution (by perm_type)
            # -----------------------------------------------------------------
            speedup_dir = base_dir / "speedup"
            speedup_dir.mkdir(parents=True, exist_ok=True)
            
            df_reordered = df_k[df_k['strategy'] != 'Original']
            
            for perm_type in df_reordered['perm_type'].unique():
                df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
                strategies = sorted(df_pt['strategy'].unique())
                
                # Boxplot of speedup by strategy
                pu.boxplot_by_category(
                    df_pt, 'strategy', 'speedup',
                    speedup_dir / f"speedup_boxplot_{perm_type}.png",
                    title=f"Speedup Distribution - {perm_type}\n{kernel}",
                    order=strategies,
                    baseline=1.0
                )
            
            # -----------------------------------------------------------------
            # 3. Binned Speedup Charts
            # -----------------------------------------------------------------
            binned_dir = base_dir / "binned_speedup"
            binned_dir.mkdir(parents=True, exist_ok=True)
            
            for bs in [4, 8, 16, 32, 64, 128]:
                imp_col = f'density_improvement_{bs}'
                if imp_col in df_reordered.columns:
                    pu.binned_bar_chart(
                        df_reordered, imp_col, 'speedup',
                        binned_dir / f"speedup_by_density_imp_bs{bs}.png",
                        title=f"Median Speedup by Density Improvement (BS {bs})\n{kernel}",
                        baseline=1.0
                    )
            
            for imp_col in ['row_spread_improvement', 'vertical_adjacency_improvement','bandwidth_improvement',]:
                if imp_col in df_reordered.columns:
                    pu.binned_bar_chart(
                        df_reordered, imp_col, 'speedup',
                        binned_dir / f"speedup_by_{imp_col}.png",
                        title=f"Median Speedup by {pu.get_display_name(imp_col)}\n{kernel}",
                        baseline=1.0
                    )


def generate_reorder_analysis_plots(df_analysis, out_dir):
    """Generate reordering analysis plots (independent of kernel performance)."""
    
    print("\n=== Reordering Analysis ===")
    
    reorder_dir = out_dir / "reorder_analysis"
    
    # Process analysis data
    df = df_analysis.copy()
    df['strategy'] = df['perm'].apply(lambda x: 'Original' if x == 'None' else str(x))
    
    # Calculate improvements (same logic as add_improvement_columns but for analysis-only data)
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'locality_avg_row_spread': {'improvement_name': 'row_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
    }
    
    # Add density improvements
    density_cols = [c for c in df.columns if c.startswith('block_density_')]
    for col in density_cols:
        bs = col.split('_')[-1]
        metrics_config[col] = {'improvement_name': f'density_improvement_{bs}', 'higher_is_better': True}
    
    available_metrics = [m for m in metrics_config.keys() if m in df.columns]
    
    # Get original values
    original = df[df['strategy'] == 'Original'][['matrix'] + available_metrics].drop_duplicates()
    original = original.groupby('matrix')[available_metrics].mean().reset_index()
    original = original.rename(columns={m: f'{m}_original' for m in available_metrics})
    
    df = df.merge(original, on='matrix', how='left')
    
    # Calculate improvements
    for metric, config in metrics_config.items():
        if metric not in available_metrics:
            continue
        orig_col = f'{metric}_original'
        imp_col = config['improvement_name']
        if config['higher_is_better']:
            df[imp_col] = df[metric] / df[orig_col]
        else:
            df[imp_col] = df[orig_col] / df[metric]
    
    # Filter to reordered only
    df_reordered = df[df['strategy'] != 'Original']
    
    if df_reordered.empty:
        print("No reordered data for analysis plots")
        return
    
    strategies = sorted(df_reordered['strategy'].unique())
    
    # -----------------------------------------------------------------
    # Bandwidth Reduction
    # -----------------------------------------------------------------
    if 'bandwidth_improvement' in df_reordered.columns:
        bw_dir = reorder_dir / "bandwidth"
        bw_dir.mkdir(parents=True, exist_ok=True)
        
        for perm_type in df_reordered['perm_type'].unique():
            df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
            pu.boxplot_by_category(
                df_pt, 'strategy', 'bandwidth_improvement',
                bw_dir / f"bandwidth_reduction_{perm_type}.png",
                title=f"Bandwidth Reduction (Original / Reordered) - {perm_type}",
                order=[s for s in strategies if s in df_pt['strategy'].unique()],
                baseline=1.0
            )
    
    # -----------------------------------------------------------------
    # Density Improvement
    # -----------------------------------------------------------------
    dens_dir = reorder_dir / "density"
    dens_dir.mkdir(parents=True, exist_ok=True)
    
    for bs in [4, 8, 16, 32, 64, 128]:
        imp_col = f'density_improvement_{bs}'
        if imp_col not in df_reordered.columns:
            continue
        
        for perm_type in df_reordered['perm_type'].unique():
            df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
            pu.boxplot_by_category(
                df_pt, 'strategy', imp_col,
                dens_dir / f"density_improvement_bs{bs}_{perm_type}.png",
                title=f"Density Improvement (BS {bs}) - {perm_type}",
                order=[s for s in strategies if s in df_pt['strategy'].unique()],
                baseline=1.0
            )
    
    # -----------------------------------------------------------------
    # Locality Improvement
    # -----------------------------------------------------------------
    loc_dir = reorder_dir / "locality"
    loc_dir.mkdir(parents=True, exist_ok=True)
    
    for imp_col, name in [
        ('row_spread_improvement', 'Row Spread Reduction'),
        ('vertical_adjacency_improvement', 'Vertical Adjacency Improvement')
    ]:
        if imp_col not in df_reordered.columns:
            continue
        
        for perm_type in df_reordered['perm_type'].unique():
            df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
            pu.boxplot_by_category(
                df_pt, 'strategy', imp_col,
                loc_dir / f"{imp_col}_{perm_type}.png",
                title=f"{name} - {perm_type}",
                order=[s for s in strategies if s in df_pt['strategy'].unique()],
                baseline=1.0
            )


def main():
    args = parse_args()
    
    # Validate mutually exclusive options
    if args.only_reorder_analysis and args.only_kernels:
        print("Error: --only-reorder-analysis and --only-kernels are mutually exclusive.")
        return
    
    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # -----------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------
    print("Loading data...")
    df, df_analysis = pu.load_data(args.operations, args.analysis)
    
    # -----------------------------------------------------------------
    # 2. Apply Filters
    # -----------------------------------------------------------------
    print("\nApplying filters...")
    df, df_analysis = pu.apply_filters(
        df, df_analysis,
        matrices_list_path=args.matrices_list,
        one_per_family=args.one_per_family,
        square_only=not args.include_rectangular,
        min_size=args.min_size
    )
    
    print(f"After filtering: {len(df)} operation rows, {len(df_analysis)} analysis rows")
    
    # -----------------------------------------------------------------
    # 3. Process Data (add all derived columns)
    # -----------------------------------------------------------------
    print("\nProcessing data...")
    df = pu.prepare_full_dataframe(df)
    
    print(f"Unique kernels: {df['kernel_id'].unique().tolist()}")
    print(f"Unique n_cols: {sorted(df['n_cols'].unique())}")
    
    # -----------------------------------------------------------------
    # 4. Generate Plots
    # -----------------------------------------------------------------
    
    # Kernel performance plots
    if not args.only_reorder_analysis:
        print("\n" + "="*60)
        print("Generating kernel performance plots...")
        print("="*60)
        generate_kernel_plots(df, out_dir, args)
    
    # Reordering analysis plots
    if not args.only_kernels:
        print("\n" + "="*60)
        print("Generating reordering analysis plots...")
        print("="*60)
        generate_reorder_analysis_plots(df_analysis, out_dir)
    
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
