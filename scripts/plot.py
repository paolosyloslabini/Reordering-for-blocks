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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
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
            # 1. GFLOPS vs Metrics (scatter plots) - Log-Log Scale
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
            # 1b. GFLOPS vs Density (Linear-Linear Scale)
            # -----------------------------------------------------------------
            linear_dir = base_dir / "gflops_vs_density_linear"
            linear_dir.mkdir(parents=True, exist_ok=True)
            
            # GFLOPS vs each density column in linear-linear scale
            for dens_col in density_cols:
                bs = dens_col.split('_')[-1]
                pu.scatter_with_correlation(
                    df_k, dens_col, 'gflops',
                    linear_dir / f"gflops_vs_density_bs{bs}_linear.png",
                    title=f"GFLOPS vs Block Density (BS {bs}) [Linear Scale]\n{kernel}",
                    log_x=False,
                    log_y=False
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
                    baseline=1.0,
                    log_y=True
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


def generate_reorderability_plots(df_analysis, out_dir):
    """Generate plots analyzing which matrices are easy/hard to reorder.
    
    Compares baseline (original) metrics against maximum improvement ratio achieved
    by any reordering algorithm. This helps identify structural predictors of reorderability.
    """
    print("\n=== Reorderability Analysis ===")
    
    reorder_dir = out_dir / "reorder_analysis" / "reorderability"
    reorder_dir.mkdir(parents=True, exist_ok=True)
    
    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)
    
    # Metrics to analyze: (metric_name, higher_is_better)
    # higher_is_better means improvement = best_reordered / baseline
    # higher_is_better=False means improvement = baseline / best_reordered
    metrics_to_analyze = [
        ('block_density_32', True, 'Block Density (32×32)'),
        ('block_density_64', True, 'Block Density (64×64)'),
        ('locality_vertical_adjacency_ratio', True, 'Vertical Adjacency Ratio'),
        ('bandwidth_max', False, 'Bandwidth'),
        ('locality_avg_row_spread', False, 'Avg Row Spread'),
    ]
    
    # Baseline metrics to correlate against
    baseline_features = [
        ('density', 'Matrix Density'),
        ('rows', 'Matrix Size (rows)'),
        ('nnz', 'Nonzeros'),
        ('locality_avg_nnz_per_row', 'Avg NNZ per Row'),
    ]
    
    results_list = []
    
    for metric, higher_is_better, metric_display in metrics_to_analyze:
        if metric not in df.columns:
            print(f"  Skipping {metric}: not in data")
            continue
        
        print(f"\n  Processing: {metric}")
        
        # For each matrix, get baseline and best reordering
        for matrix in df['matrix'].unique():
            df_m = df[df['matrix'] == matrix]
            
            baseline = df_m[df_m['perm'] == 'None']
            reordered = df_m[df_m['perm'] != 'None']
            
            if baseline.empty or reordered.empty:
                continue
            
            baseline_val = baseline[metric].iloc[0]
            if pd.isna(baseline_val) or baseline_val == 0:
                continue
            
            if higher_is_better:
                best_val = reordered[metric].max()
                best_perm = reordered.loc[reordered[metric].idxmax(), 'perm']
                improvement = best_val / baseline_val
            else:
                best_val = reordered[metric].min()
                best_perm = reordered.loc[reordered[metric].idxmin(), 'perm']
                improvement = baseline_val / best_val if best_val > 0 else np.nan
            
            row_data = {
                'matrix': matrix,
                'metric': metric,
                'metric_display': metric_display,
                'baseline_value': baseline_val,
                'best_value': best_val,
                'best_perm': best_perm,
                'improvement_ratio': improvement,
            }
            
            # Add baseline features
            for feat, _ in baseline_features:
                if feat in baseline.columns:
                    row_data[f'baseline_{feat}'] = baseline[feat].iloc[0]
            
            results_list.append(row_data)
    
    if not results_list:
        print("  No reorderability data to plot")
        return
    
    results_df = pd.DataFrame(results_list)
    
    # Save results to CSV
    results_df.to_csv(reorder_dir / "reorderability_summary.csv", index=False)
    print(f"  Saved summary to reorderability_summary.csv")
    
    # Generate plots for each metric
    for metric, _, metric_display in metrics_to_analyze:
        if metric not in df.columns:
            continue
        
        df_metric = results_df[results_df['metric'] == metric].copy()
        if df_metric.empty:
            continue
        
        metric_safe = pu.safe_filename(metric)
        y_label = f'Max {metric_display} Improvement Ratio'
        
        # Apply 99th percentile clipping to remove outliers
        upper_bound = df_metric['improvement_ratio'].quantile(0.99)
        df_metric_clipped = df_metric[df_metric['improvement_ratio'] <= upper_bound].copy()
        n_outliers = len(df_metric) - len(df_metric_clipped)
        clip_note = f'\n(99th percentile, {n_outliers} outliers removed)' if n_outliers > 0 else ''
        
        # 1. Histogram of improvement ratios
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df_metric_clipped['improvement_ratio'].dropna(), bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='No improvement')
        ax.set_xlabel(y_label)
        ax.set_ylabel('Number of Matrices')
        ax.set_title(f'Distribution of Reorderability\n{metric_display}{clip_note}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reorder_dir / f"improvement_hist_{metric_safe}.png", dpi=150)
        plt.close()
        print(f"  Saved: improvement_hist_{metric_safe}.png")
        
        # 2. Baseline value vs Improvement (can we predict from original structure?)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_metric_clipped['baseline_value'], df_metric_clipped['improvement_ratio'], alpha=0.5)
        
        # Add correlation (use full data for correlation)
        valid = df_metric[['baseline_value', 'improvement_ratio']].dropna()
        valid = valid[(valid['baseline_value'] > 0) & (valid['improvement_ratio'] > 0)]
        if len(valid) > 10:
            tau, p = stats.kendalltau(valid['baseline_value'], valid['improvement_ratio'])
            # Pearson on log values since we use log scale
            pearson_r, _ = stats.pearsonr(np.log10(valid['baseline_value']), np.log10(valid['improvement_ratio']))
            ax.set_title(f'Improvement vs {metric_display}\nτ = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
        else:
            ax.set_title(f'Improvement vs {metric_display}{clip_note}')
        
        ax.set_xlabel(f'{metric_display}')
        ax.set_ylabel(y_label)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reorder_dir / f"baseline_vs_improvement_{metric_safe}.png", dpi=150)
        plt.close()
        print(f"  Saved: baseline_vs_improvement_{metric_safe}.png")
        
        # 3. Matrix density vs Improvement
        if 'baseline_density' in df_metric_clipped.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_metric_clipped['baseline_density'], df_metric_clipped['improvement_ratio'], alpha=0.5)
            
            valid = df_metric[['baseline_density', 'improvement_ratio']].dropna()
            valid = valid[(valid['baseline_density'] > 0) & (valid['improvement_ratio'] > 0)]
            if len(valid) > 10:
                tau, p = stats.kendalltau(valid['baseline_density'], valid['improvement_ratio'])
                # Pearson on log values since we use log scale
                pearson_r, _ = stats.pearsonr(np.log10(valid['baseline_density']), np.log10(valid['improvement_ratio']))
                ax.set_title(f'{metric_display} Improvement vs Matrix Density\nτ = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
            else:
                ax.set_title(f'{metric_display} Improvement vs Matrix Density{clip_note}')
            
            ax.set_xlabel('Matrix Density')
            ax.set_ylabel(y_label)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reorder_dir / f"density_vs_improvement_{metric_safe}.png", dpi=150)
            plt.close()
            print(f"  Saved: density_vs_improvement_{metric_safe}.png")
        
        # 4. Matrix size vs Improvement
        if 'baseline_rows' in df_metric_clipped.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_metric_clipped['baseline_rows'], df_metric_clipped['improvement_ratio'], alpha=0.5)
            
            valid = df_metric[['baseline_rows', 'improvement_ratio']].dropna()
            valid = valid[(valid['baseline_rows'] > 0) & (valid['improvement_ratio'] > 0)]
            if len(valid) > 10:
                tau, p = stats.kendalltau(valid['baseline_rows'], valid['improvement_ratio'])
                # Pearson on log values since we use log scale
                pearson_r, _ = stats.pearsonr(np.log10(valid['baseline_rows']), np.log10(valid['improvement_ratio']))
                ax.set_title(f'{metric_display} Improvement vs Matrix Size\nτ = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
            else:
                ax.set_title(f'{metric_display} Improvement vs Matrix Size{clip_note}')
            
            ax.set_xlabel('Matrix Size (rows)')
            ax.set_ylabel(y_label)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reorder_dir / f"size_vs_improvement_{metric_safe}.png", dpi=150)
            plt.close()
            print(f"  Saved: size_vs_improvement_{metric_safe}.png")
        
        # 5. Avg NNZ per row vs Improvement
        if 'baseline_locality_avg_nnz_per_row' in df_metric_clipped.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df_metric_clipped['baseline_locality_avg_nnz_per_row'], df_metric_clipped['improvement_ratio'], alpha=0.5)
            
            valid = df_metric[['baseline_locality_avg_nnz_per_row', 'improvement_ratio']].dropna()
            valid = valid[(valid['baseline_locality_avg_nnz_per_row'] > 0) & (valid['improvement_ratio'] > 0)]
            if len(valid) > 10:
                tau, p = stats.kendalltau(valid['baseline_locality_avg_nnz_per_row'], valid['improvement_ratio'])
                # Pearson on log values since we use log scale
                pearson_r, _ = stats.pearsonr(np.log10(valid['baseline_locality_avg_nnz_per_row']), np.log10(valid['improvement_ratio']))
                ax.set_title(f'{metric_display} Improvement vs Avg NNZ/Row\nτ = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
            else:
                ax.set_title(f'{metric_display} Improvement vs Avg NNZ/Row{clip_note}')
            
            ax.set_xlabel('Avg Nonzeros per Row')
            ax.set_ylabel(y_label)
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reorder_dir / f"nnz_per_row_vs_improvement_{metric_safe}.png", dpi=150)
            plt.close()
            print(f"  Saved: nnz_per_row_vs_improvement_{metric_safe}.png")
    
    print(f"\n  All reorderability plots saved to {reorder_dir}")


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
        'bandwidth_avg': {'improvement_name': 'bandwidth_avg_improvement', 'higher_is_better': False},
        'locality_avg_row_spread': {'improvement_name': 'row_spread_improvement', 'higher_is_better': False},
        'locality_max_row_spread': {'improvement_name': 'max_row_spread_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_max_col_spread': {'improvement_name': 'max_col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
    }
    
    # Add density improvements
    density_cols = [c for c in df.columns if c.startswith('block_density_')]
    for col in density_cols:
        bs = col.split('_')[-1]
        metrics_config[col] = {'improvement_name': f'density_improvement_{bs}', 'higher_is_better': True}
    
    # Add blocks per row improvements (fewer blocks per row = better locality)
    for bs in [4, 8, 16, 32, 64, 128]:
        avg_col = f'avg_blocks_per_row_{bs}'
        max_col = f'max_blocks_per_row_{bs}'
        if avg_col in df.columns:
            metrics_config[avg_col] = {'improvement_name': f'avg_blocks_per_row_improvement_{bs}', 'higher_is_better': False}
        if max_col in df.columns:
            metrics_config[max_col] = {'improvement_name': f'max_blocks_per_row_improvement_{bs}', 'higher_is_better': False}
    
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
                baseline=1.0,
                log_y=True
            )
        
        # Avg bandwidth improvement
        if 'bandwidth_avg_improvement' in df_reordered.columns:
            for perm_type in df_reordered['perm_type'].unique():
                df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
                pu.boxplot_by_category(
                    df_pt, 'strategy', 'bandwidth_avg_improvement',
                    bw_dir / f"bandwidth_avg_reduction_{perm_type}.png",
                    title=f"Avg Bandwidth Reduction (Original / Reordered) - {perm_type}",
                    order=[s for s in strategies if s in df_pt['strategy'].unique()],
                    baseline=1.0,
                    log_y=True
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
                baseline=1.0,
                log_y=True
            )
    
    # -----------------------------------------------------------------
    # Blocks Per Row Improvement
    # -----------------------------------------------------------------
    blocks_dir = reorder_dir / "blocks_per_row"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    
    for bs in [4, 8, 16, 32, 64, 128]:
        for prefix, name in [('avg', 'Avg'), ('max', 'Max')]:
            imp_col = f'{prefix}_blocks_per_row_improvement_{bs}'
            if imp_col not in df_reordered.columns:
                continue
            
            for perm_type in df_reordered['perm_type'].unique():
                df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
                pu.boxplot_by_category(
                    df_pt, 'strategy', imp_col,
                    blocks_dir / f"{prefix}_blocks_per_row_improvement_bs{bs}_{perm_type}.png",
                    title=f"{name} Blocks/Row Reduction (BS {bs}) - {perm_type}",
                    order=[s for s in strategies if s in df_pt['strategy'].unique()],
                    baseline=1.0,
                    log_y=True
                )
    
    # -----------------------------------------------------------------
    # Locality Improvement
    # -----------------------------------------------------------------
    loc_dir = reorder_dir / "locality"
    loc_dir.mkdir(parents=True, exist_ok=True)
    
    for imp_col, name in [
        ('row_spread_improvement', 'Row Spread Reduction'),
        ('max_row_spread_improvement', 'Max Row Spread Reduction'),
        ('col_spread_improvement', 'Col Spread Reduction'),
        ('max_col_spread_improvement', 'Max Col Spread Reduction'),
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
                baseline=1.0,
                log_y=True
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
        
        print("\n" + "="*60)
        print("Generating reorderability analysis plots...")
        print("="*60)
        generate_reorderability_plots(df_analysis, out_dir)
    
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
