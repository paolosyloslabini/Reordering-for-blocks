"""
Plot Script - Clean Architecture

Simple iteration-based plotting script that:
1. Loads and processes data once
2. Applies filters
3. Iterates over dimensions (n_cols, kernel, perm_type) to generate plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import plot_utils as pu
from settings import get_perm_display, KERNEL_NAMES, GROUPED_SCATTER_EXCLUDE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")

    # Random pipeline shortcut
    parser.add_argument("--random", action="store_true",
                        help="Use random-pipeline data (filter_config_random.yaml, output to plots_random)")

    # Output
    parser.add_argument("--out", default=None, help="Output directory (default: plots or plots_random with --random)")

    # Filter config (all filtering is driven by filter_config.yaml;
    # these flags *override* settings in the config file)
    parser.add_argument("--filter-config", default=None,
                        help="Path to filter_config.yaml (default: scripts/filter_config.yaml)")
    parser.add_argument("--operations", default=None,
                        help="Override operations CSV path from config")
    parser.add_argument("--analysis", default=None,
                        help="Override analysis CSV path from config")
    parser.add_argument("--matrices-list", default=None,
                        help="Override matrices list path from config")
    parser.add_argument("--one-per-family", action="store_true", default=None,
                        help="Override: enable one-per-family filter")
    parser.add_argument("--no-one-per-family", action="store_false", dest="one_per_family",
                        help="Override: disable one-per-family filter")
    parser.add_argument("--include-rectangular", action="store_true", default=None,
                        help="Override: include rectangular matrices")
    parser.add_argument("--min-size", type=int, default=None,
                        help="Override: minimum matrix size (rows)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="Override: maximum matrix size (rows or cols)")
    
    # Plot selection
    parser.add_argument("--only-reorder-analysis", action="store_true")
    parser.add_argument("--only-kernels", action="store_true")
    parser.add_argument("--n-cols", type=int, default=None, help="Filter to specific n_cols")
    parser.add_argument("--kernel", type=str, default=None, help="Filter to specific kernel")
    
    # Fine-grained section selection (used by run_plots.py wrapper)
    SECTION_CHOICES = [
        'kernels', 'breakeven', 'original-scatter',
        'reorder-analysis', 'reorderability', 'per-matrix', 'timing',
    ]
    parser.add_argument(
        "--sections", nargs="+", choices=SECTION_CHOICES, default=None,
        help=(
            "Run only the listed plot sections. "
            "Overrides --only-kernels / --only-reorder-analysis. "
            f"Choices: {', '.join(SECTION_CHOICES)}"
        ),
    )

    # Parallelism
    parser.add_argument(
        "-j", "--jobs", type=int, default=None,
        help="Number of parallel workers for plot generation (default: min(cpu_count, 8), 1=sequential)",
    )

    return parser.parse_args()


def generate_kernel_plots(df, out_dir, args):
    """Generate performance plots for kernel operations."""
    n_jobs = getattr(args, 'jobs', None)

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
    tasks = []

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
            print(f"  Collecting tasks for kernel: {kernel}")
            df_k = df_nc[df_nc['kernel_id'] == kernel]

            kernel_safe = pu.safe_filename(kernel)
            base_dir = out_dir / f"n_cols_{int(n_cols)}" / kernel_safe
            kernel_label = KERNEL_NAMES.get(kernel, kernel)

            # Create directories upfront
            gflops_dir = base_dir / "gflops_correlations"
            gflops_dir.mkdir(parents=True, exist_ok=True)
            linear_dir = base_dir / "gflops_vs_density_linear"
            linear_dir.mkdir(parents=True, exist_ok=True)
            speedup_dir = base_dir / "speedup"
            speedup_dir.mkdir(parents=True, exist_ok=True)
            binned_dir = base_dir / "binned_speedup"
            binned_dir.mkdir(parents=True, exist_ok=True)

            # 1. GFLOPS vs Metrics (log-log)
            for dens_col in density_cols:
                bs = dens_col.split('_')[-1]
                tasks.append((pu.scatter_publication, dict(
                    df=df_k, x_col=dens_col, y_col='gflops',
                    output_path=gflops_dir / f"gflops_vs_density_bs{bs}.png",
                    label=kernel_label)))

            if 'rel_bandwidth' in df_k.columns:
                tasks.append((pu.scatter_publication, dict(
                    df=df_k, x_col='rel_bandwidth', y_col='gflops',
                    output_path=gflops_dir / f"gflops_vs_rel_bandwidth.png",
                    label=kernel_label)))

            for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio']:
                if loc_col in df_k.columns:
                    tasks.append((pu.scatter_publication, dict(
                        df=df_k, x_col=loc_col, y_col='gflops',
                        output_path=gflops_dir / f"gflops_vs_{loc_col}.png",
                        label=kernel_label)))

            # 1b. GFLOPS vs Density (linear-linear)
            for dens_col in density_cols:
                bs = dens_col.split('_')[-1]
                tasks.append((pu.scatter_publication, dict(
                    df=df_k, x_col=dens_col, y_col='gflops',
                    output_path=linear_dir / f"gflops_vs_density_bs{bs}_linear.png",
                    log_x=False, log_y=False, label=kernel_label)))

            if 'rel_bandwidth' in df_k.columns:
                tasks.append((pu.scatter_publication, dict(
                    df=df_k, x_col='rel_bandwidth', y_col='gflops',
                    output_path=linear_dir / f"gflops_vs_rel_bandwidth_linear.png",
                    log_x=False, log_y=False, label=kernel_label)))

            for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio']:
                if loc_col in df_k.columns:
                    tasks.append((pu.scatter_publication, dict(
                        df=df_k, x_col=loc_col, y_col='gflops',
                        output_path=linear_dir / f"gflops_vs_{loc_col}_linear.png",
                        log_x=False, log_y=False, label=kernel_label)))

            # 2. Speedup Distribution (by perm_type)
            df_reordered = df_k[df_k['strategy'] != 'Original']
            for perm_type in df_reordered['perm_type'].unique():
                df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
                strategies = sorted(df_pt['strategy'].unique())
                tasks.append((pu.boxplot_by_category, dict(
                    df=df_pt, x_col='strategy', y_col='speedup',
                    output_path=speedup_dir / f"speedup_boxplot_{perm_type}.png",
                    title=f"Speedup Distribution - {perm_type}\n{kernel}",
                    order=strategies, baseline=1.0, log_y=True, ylim=(0.5, 5))))

            # 3. Binned Speedup Charts
            density_bins = [0, 1.5, 2.0, 3.0, 1000.0]
            density_labels = ['<1.5x', '1.5-2x', '2-3x', '>3x']
            for bs in [4, 8, 16, 32, 64, 128]:
                imp_col = f'density_improvement_{bs}'
                if imp_col in df_reordered.columns:
                    tasks.append((pu.binned_boxplot, dict(
                        df=df_reordered, bin_col=imp_col, value_col='speedup',
                        output_path=binned_dir / f"speedup_by_density_imp_bs{bs}.png",
                        title=f"Speedup Distribution by Density Improvement (BS {bs})\n{kernel}",
                        baseline=1.0, bins=density_bins, labels=density_labels)))

            for imp_col in ['row_spread_improvement', 'vertical_adjacency_improvement', 'bandwidth_improvement']:
                if imp_col in df_reordered.columns:
                    tasks.append((pu.binned_boxplot, dict(
                        df=df_reordered, bin_col=imp_col, value_col='speedup',
                        output_path=binned_dir / f"speedup_by_{imp_col}.png",
                        title=f"Speedup Distribution by {pu.get_display_name(imp_col)}\n{kernel}",
                        baseline=1.0)))

        # -----------------------------------------------------------------
        # 4. Grouped Scatter Plots (all kernels in 2x3 grid)
        # -----------------------------------------------------------------
        grouped_kernels = [k for k in kernels if k not in GROUPED_SCATTER_EXCLUDE]
        kernel_labels = {k: KERNEL_NAMES.get(k, k) for k in grouped_kernels}

        grouped_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_scatter"
        grouped_dir.mkdir(parents=True, exist_ok=True)
        grouped_linear_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_scatter_linear"
        grouped_linear_dir.mkdir(parents=True, exist_ok=True)

        for dens_col in density_cols:
            bs = dens_col.split('_')[-1]
            tasks.append((pu.grouped_scatter_publication, dict(
                df=df_nc, x_col=dens_col, y_col='gflops', group_col='kernel_id',
                group_order=grouped_kernels,
                output_path=grouped_dir / f"gflops_vs_density_bs{bs}.png",
                group_labels=kernel_labels)))
            tasks.append((pu.grouped_scatter_publication, dict(
                df=df_nc, x_col=dens_col, y_col='gflops', group_col='kernel_id',
                group_order=grouped_kernels,
                output_path=grouped_linear_dir / f"gflops_vs_density_bs{bs}_linear.png",
                group_labels=kernel_labels, log_x=False, log_y=False)))

        if 'rel_bandwidth' in df_nc.columns:
            tasks.append((pu.grouped_scatter_publication, dict(
                df=df_nc, x_col='rel_bandwidth', y_col='gflops', group_col='kernel_id',
                group_order=grouped_kernels,
                output_path=grouped_dir / f"gflops_vs_rel_bandwidth.png",
                group_labels=kernel_labels)))
            tasks.append((pu.grouped_scatter_publication, dict(
                df=df_nc, x_col='rel_bandwidth', y_col='gflops', group_col='kernel_id',
                group_order=grouped_kernels,
                output_path=grouped_linear_dir / f"gflops_vs_rel_bandwidth_linear.png",
                group_labels=kernel_labels, log_x=False, log_y=False)))

        for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio']:
            if loc_col in df_nc.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc, x_col=loc_col, y_col='gflops', group_col='kernel_id',
                    group_order=grouped_kernels,
                    output_path=grouped_dir / f"gflops_vs_{loc_col}.png",
                    group_labels=kernel_labels)))
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc, x_col=loc_col, y_col='gflops', group_col='kernel_id',
                    group_order=grouped_kernels,
                    output_path=grouped_linear_dir / f"gflops_vs_{loc_col}_linear.png",
                    group_labels=kernel_labels, log_x=False, log_y=False)))

        # 5. Grouped Scatter: Improvement vs Speedup
        df_nc_reordered = df_nc[df_nc['strategy'] != 'Original']

        grouped_imp_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_improvement_vs_speedup"
        grouped_imp_dir.mkdir(parents=True, exist_ok=True)
        grouped_imp_log_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_improvement_vs_speedup_loglog"
        grouped_imp_log_dir.mkdir(parents=True, exist_ok=True)

        for bs in [4, 8, 16, 32, 64, 128]:
            imp_col = f'density_improvement_{bs}'
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_dir / f"speedup_vs_density_imp_bs{bs}.png",
                    group_labels=kernel_labels, log_x=False, log_y=False)))
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_log_dir / f"speedup_vs_density_imp_bs{bs}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True)))

        for imp_col in ['bandwidth_improvement', 'row_spread_improvement',
                        'vertical_adjacency_improvement']:
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_dir / f"speedup_vs_{imp_col}.png",
                    group_labels=kernel_labels, log_x=False, log_y=False)))
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_log_dir / f"speedup_vs_{imp_col}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True)))

    print(f"\n  Collected {len(tasks)} kernel plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def generate_original_scatter_plots(df, out_dir, args):
    """Generate grouped scatter plots using only original (unreordered) matrices.

    For each n_cols value, produces 2x3 grids (one subplot per kernel) showing
    how the original matrix structure relates to kernel performance.
    """
    print("\n=== Original-Only Grouped Scatter Plots ===")

    df_orig = df[df['strategy'] == 'Original'].copy()
    if df_orig.empty:
        print("No original (perm=None) data found — skipping.")
        return

    n_cols_values = sorted(df_orig['n_cols'].unique())
    if args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"Warning: n_cols={args.n_cols} not found. Available: {n_cols_values}")
            return

    density_cols = pu.get_density_columns(df_orig)

    # Structural x-metrics to plot against GFLOPS
    structural_metrics = []
    for dc in density_cols:
        structural_metrics.append(dc)
    for col in ['rel_bandwidth', 'rel_row_spread',
                'locality_vertical_adjacency_ratio', 'density']:
        if col in df_orig.columns:
            structural_metrics.append(col)

    for n_cols in n_cols_values:
        print(f"\n--- n_cols = {n_cols} ---")
        df_nc = df_orig[df_orig['n_cols'] == n_cols]

        kernels = sorted(df_nc['kernel_id'].unique())
        if args.kernel:
            kernels = [k for k in kernels if args.kernel.lower() in k.lower()]
            if not kernels:
                print(f"No kernels matching '{args.kernel}'")
                continue

        grouped_kernels_orig = [k for k in kernels if k not in GROUPED_SCATTER_EXCLUDE]
        kernel_labels = {k: KERNEL_NAMES.get(k, k) for k in grouped_kernels_orig}

        # Log-log grouped scatter (original only)
        scatter_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_scatter_original"
        scatter_dir.mkdir(parents=True, exist_ok=True)

        for x_col in structural_metrics:
            safe = pu.safe_filename(x_col)
            pu.grouped_scatter_publication(
                df_nc, x_col, 'gflops', 'kernel_id', grouped_kernels_orig,
                scatter_dir / f"gflops_vs_{safe}.png",
                group_labels=kernel_labels,
            )

        # Linear-linear grouped scatter (original only)
        scatter_linear_dir = out_dir / f"n_cols_{int(n_cols)}" / "grouped_scatter_original_linear"
        scatter_linear_dir.mkdir(parents=True, exist_ok=True)

        for x_col in structural_metrics:
            safe = pu.safe_filename(x_col)
            pu.grouped_scatter_publication(
                df_nc, x_col, 'gflops', 'kernel_id', grouped_kernels_orig,
                scatter_linear_dir / f"gflops_vs_{safe}_linear.png",
                group_labels=kernel_labels,
                log_x=False, log_y=False,
            )

    print(f"  Original-only scatter plots saved under {out_dir}/n_cols_*/grouped_scatter_original*/")


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
        
        # No clipping by default (can be enabled by setting apply_clipping=True)
        apply_clipping = False
        if apply_clipping:
            upper_bound = df_metric['improvement_ratio'].quantile(0.99)
            df_metric_clipped = df_metric[df_metric['improvement_ratio'] <= upper_bound].copy()
            n_outliers = len(df_metric) - len(df_metric_clipped)
            clip_note = f'\n(99th percentile, {n_outliers} outliers removed)' if n_outliers > 0 else ''
        else:
            df_metric_clipped = df_metric.copy()
            clip_note = ''
        
        # 1. Histogram of improvement ratios
        fig, ax = plt.subplots(figsize=(12, 7))
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
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.scatter(df_metric_clipped['baseline_value'], df_metric_clipped['improvement_ratio'], alpha=0.5)
        
        # Add correlation (use full data for correlation)
        valid = df_metric[['baseline_value', 'improvement_ratio']].dropna()
        if len(valid) > 10:
            tau, p = stats.kendalltau(valid['baseline_value'], valid['improvement_ratio'])
            pearson_r, _ = stats.pearsonr(valid['baseline_value'], valid['improvement_ratio'])
            ax.set_title(f'Improvement vs {metric_display}\nτ = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
        else:
            ax.set_title(f'Improvement vs {metric_display}{clip_note}')
        
        ax.set_xlabel(f'{metric_display}')
        ax.set_ylabel(y_label)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No improvement')
        ax.set_xscale('log')
        ax.set_yscale('log')
        pu.format_log_axes(ax)
        ax.legend()
        plt.tight_layout()
        plt.savefig(reorder_dir / f"baseline_vs_improvement_{metric_safe}.png", dpi=150)
        plt.close()
        print(f"  Saved: baseline_vs_improvement_{metric_safe}.png")
        
        # 3-6. Baseline features vs Improvement
        feature_scatter_specs = [
            ('baseline_density', 'Matrix Density', 'density_vs_improvement'),
            ('baseline_rows', 'Matrix Size (rows)', 'size_vs_improvement'),
            ('baseline_nnz', 'Nonzeros', 'nnz_vs_improvement'),
            ('baseline_locality_avg_nnz_per_row', 'Avg Nonzeros per Row', 'nnz_per_row_vs_improvement'),
        ]
        for x_col, x_label, filename_prefix in feature_scatter_specs:
            _reorderability_scatter(
                df_metric, df_metric_clipped, x_col, x_label, y_label,
                metric_display, clip_note,
                reorder_dir / f"{filename_prefix}_{metric_safe}.png")
    
    print(f"\n  All reorderability plots saved to {reorder_dir}")


def _reorderability_scatter(df_full, df_clipped, x_col, x_label, y_label,
                            metric_display, clip_note, output_path):
    """Scatter plot: x_col vs improvement_ratio with Kendall/Pearson annotation."""
    if x_col not in df_clipped.columns:
        return
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(df_clipped[x_col], df_clipped['improvement_ratio'], alpha=0.5)

    valid = df_full[[x_col, 'improvement_ratio']].dropna()
    if len(valid) > 10:
        tau, _ = stats.kendalltau(valid[x_col], valid['improvement_ratio'])
        pearson_r, _ = stats.pearsonr(valid[x_col], valid['improvement_ratio'])
        ax.set_title(f'{metric_display} Improvement vs {x_label}\n'
                     f'\u03c4 = {tau:.3f}, r = {pearson_r:.3f}{clip_note}')
    else:
        ax.set_title(f'{metric_display} Improvement vs {x_label}{clip_note}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    pu.format_log_axes(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_per_matrix_study(df_analysis, out_dir):
    """Per-matrix difficulty study.

    For each structural metric, compute the improvement ratio of every
    reordering relative to the original.  Then, per matrix, summarise
    across all reorderings:
        * geometric mean of improvement ratios  (= exp(mean(log(ratio))))
        * max improvement ratio                 (best single reordering)
        * variance of log-improvement           (spread across reorderings)

    Produces scatter plots in (geomean, variance) and (max, variance)
    space so that easy/hard matrices can be identified visually.
    """
    print("\n=== Per-Matrix Difficulty Study ===")

    study_dir = out_dir / "reorder_analysis" / "per_matrix_study"
    study_dir.mkdir(parents=True, exist_ok=True)

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)

    # Metrics to study: (column, higher_is_better, display_name)
    metrics_to_study = [
        ('block_density_32',  True,  'Block Density (32×32)'),
        ('block_density_64',  True,  'Block Density (64×64)'),
        ('locality_vertical_adjacency_ratio', True,  'Vertical Adjacency Ratio'),
        ('bandwidth_max',     False, 'Bandwidth'),
        ('locality_avg_row_spread', False, 'Avg Row Spread'),
    ]

    for metric, higher_is_better, metric_display in metrics_to_study:
        if metric not in df.columns:
            print(f"  Skipping {metric}: not in data")
            continue

        print(f"  Processing: {metric}")

        rows = []
        for matrix in df['matrix'].unique():
            df_m = df[df['matrix'] == matrix]
            baseline = df_m[df_m['perm'] == 'None']
            reordered = df_m[df_m['perm'] != 'None']

            if baseline.empty or reordered.empty:
                continue

            baseline_val = baseline[metric].iloc[0]
            if pd.isna(baseline_val) or baseline_val == 0:
                continue

            # Compute improvement ratio for every reordering
            if higher_is_better:
                ratios = reordered[metric] / baseline_val
            else:
                ratios = baseline_val / reordered[metric]

            ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
            ratios = ratios[ratios > 0]

            if len(ratios) < 2:
                continue

            log_ratios = np.log(ratios)
            geomean = np.exp(log_ratios.mean())
            max_improvement = ratios.max()
            variance = log_ratios.var()

            rows.append({
                'matrix': matrix,
                'geomean_improvement': geomean,
                'max_improvement': max_improvement,
                'log_variance': variance,
                'n_reorderings': len(ratios),
            })

        if not rows:
            print(f"    No data for {metric}")
            continue

        res = pd.DataFrame(rows)

        # Save CSV
        metric_safe = pu.safe_filename(metric)
        res.to_csv(study_dir / f"per_matrix_{metric_safe}.csv", index=False)

        # ── Scatter plot ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))

        ax.scatter(res['geomean_improvement'], res['log_variance'],
                   alpha=0.55, edgecolors='k', linewidths=0.3, s=36)

        # Reference line at geomean = 1 (no average improvement)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.2,
                   label='No avg. improvement')

        ax.set_xlabel('Geometric Mean of Improvement Ratio')
        ax.set_ylabel('Variance of log(Improvement Ratio)')
        ax.set_title(f'Per-Matrix Difficulty — {metric_display}\n'
                     f'({len(res)} matrices, improvement > 1 = beneficial)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Log scale on x if range is large
        if res['geomean_improvement'].max() / max(res['geomean_improvement'].min(), 1e-9) > 20:
            ax.set_xscale('log')
            pu.format_log_axes(ax)

        plt.tight_layout()
        plt.savefig(study_dir / f"geomean_vs_variance_{metric_safe}.png", dpi=150)
        plt.close()
        print(f"    Saved: geomean_vs_variance_{metric_safe}.png")

        # ── Max improvement vs variance scatter ──────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))

        ax.scatter(res['max_improvement'], res['log_variance'],
                   alpha=0.55, edgecolors='k', linewidths=0.3, s=36)

        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1.2,
                   label='No improvement')

        ax.set_xlabel('Max Improvement Ratio (Best Reordering)')
        ax.set_ylabel('Variance of log(Improvement Ratio)')
        ax.set_title(f'Per-Matrix Difficulty (Max) — {metric_display}\n'
                     f'({len(res)} matrices, improvement > 1 = beneficial)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if res['max_improvement'].max() / max(res['max_improvement'].min(), 1e-9) > 20:
            ax.set_xscale('log')
            pu.format_log_axes(ax)

        plt.tight_layout()
        plt.savefig(study_dir / f"max_vs_variance_{metric_safe}.png", dpi=150)
        plt.close()
        print(f"    Saved: max_vs_variance_{metric_safe}.png")

        # ── Annotate extreme matrices (optional info) ────────────────
        top_easy = res.nlargest(5, 'geomean_improvement')[['matrix', 'geomean_improvement', 'log_variance']]
        top_hard = res.nsmallest(5, 'geomean_improvement')[['matrix', 'geomean_improvement', 'log_variance']]
        top_max  = res.nlargest(5, 'max_improvement')[['matrix', 'max_improvement', 'log_variance']]
        top_variable = res.nlargest(5, 'log_variance')[['matrix', 'geomean_improvement', 'log_variance']]
        print(f"    Top-5 easiest (highest geomean): {top_easy['matrix'].tolist()}")
        print(f"    Top-5 hardest (lowest geomean):  {top_hard['matrix'].tolist()}")
        print(f"    Top-5 best max improvement:      {top_max['matrix'].tolist()}")
        print(f"    Top-5 most variable:             {top_variable['matrix'].tolist()}")

    print(f"\n  Per-matrix study saved to {study_dir}")


def _boxplot_for_perm_types(df_reordered, strategies, imp_col, title_template,
                            output_dir, filename_template):
    """Generate one boxplot per perm_type for a given improvement column."""
    if imp_col not in df_reordered.columns:
        return
    for perm_type in df_reordered['perm_type'].unique():
        df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
        pu.boxplot_by_category(
            df_pt, 'strategy', imp_col,
            output_dir / filename_template.format(perm_type=perm_type),
            title=title_template.format(perm_type=perm_type),
            order=[s for s in strategies if s in df_pt['strategy'].unique()],
            baseline=1.0,
            log_y=True,
        )


def generate_reorder_timing_plots(df_analysis, out_dir, reordering_csv='results/results_reordering.csv'):
    """Generate reordering timing analysis plots (nnz vs reordering time).

    Loads ``results_reordering.csv`` (produced by ``parse_results.py``) and
    joins it with the analysis DataFrame to get nnz per matrix. Then creates
    scatter plots of nnz vs reordering time, coloured by algorithm.
    """
    print("\n=== Reordering Timing Analysis ===")

    reorder_csv_path = Path(reordering_csv)
    if not reorder_csv_path.exists():
        print(f"  Reordering CSV not found at {reorder_csv_path} — skipping timing plots. "
              "Run  python scripts/parse_results.py  first to generate it.")
        return

    df_time = pd.read_csv(reorder_csv_path)
    print(f"  Loaded {len(df_time)} reordering timing rows")

    if df_time.empty or 'time_reordering_ms' not in df_time.columns:
        print("  No reordering timing data available — skipping.")
        return

    # Get nnz from analysis data (use the Original / baseline row for each matrix)
    df_base = df_analysis[df_analysis['perm'] == 'None'][['matrix', 'nnz', 'rows']].drop_duplicates('matrix')
    df_time = df_time.merge(df_base, on='matrix', how='inner')

    if df_time.empty:
        print("  No matching matrices between timing and analysis data — skipping.")
        return

    # Map perm tag to display name
    df_time['strategy'] = df_time['perm'].apply(
        lambda x: get_perm_display(x) if x != 'None' else 'Original'
    )
    # Drop any Original rows (shouldn't be any, but just in case)
    df_time = df_time[df_time['strategy'] != 'Original']

    timing_dir = out_dir / "reorder_analysis" / "timing"
    timing_dir.mkdir(parents=True, exist_ok=True)

    strategies = sorted(df_time['strategy'].unique())
    palette = pu.get_strategy_palette(strategies)

    # --- Scatter: nnz vs reordering time (all algorithms, coloured) ---
    pu.scatter_with_correlation(
        df_time, 'nnz', 'time_reordering_ms',
        timing_dir / "nnz_vs_reordering_time.png",
        title="Reordering Time vs Matrix NNZ",
        hue_col='strategy',
        log_x=True, log_y=True,
    )

    # --- Per-algorithm scatter: nnz vs reordering time ---
    for strat in strategies:
        df_s = df_time[df_time['strategy'] == strat]
        if len(df_s) < 2:
            continue
        pu.scatter_with_correlation(
            df_s, 'nnz', 'time_reordering_ms',
            timing_dir / f"nnz_vs_time_{pu.safe_filename(strat)}.png",
            title=f"Reordering Time vs NNZ — {strat}",
            log_x=True, log_y=True,
        )

    # --- Boxplot: reordering time by algorithm ---
    pu.boxplot_by_category(
        df_time, 'strategy', 'time_reordering_ms',
        timing_dir / "reordering_time_boxplot.png",
        title="Reordering Time by Algorithm",
        order=strategies,
        log_y=True,
        palette=palette,
    )

    # --- Scatter: rows vs reordering time ---
    pu.scatter_with_correlation(
        df_time, 'rows', 'time_reordering_ms',
        timing_dir / "rows_vs_reordering_time.png",
        title="Reordering Time vs Matrix Rows",
        hue_col='strategy',
        log_x=True, log_y=True,
    )

    print(f"  Timing plots saved to {timing_dir}")


def generate_breakeven_plots(df, out_dir, reordering_csv='results/results_reordering.csv', args=None):
    """Compute and plot break-even number of operations for each reordering.

    For every (matrix, perm, kernel_id, n_cols) tuple the break-even count is:

        breakeven_n = time_reordering_ms / (time_baseline_ms - time_reordered_ms)

    Cases where reordering is harmful (denominator ≤ 0) are shown as '×'
    markers at a cap line and excluded from boxplot statistics.

    Outputs one boxplot per (kernel, n_cols, perm_type) to
    ``plots/n_cols_{N}/{kernel}/breakeven/``.
    """
    print("\n=== Break-even Analysis ===")

    # --- Load reordering times ---
    reorder_csv_path = Path(reordering_csv)
    if not reorder_csv_path.exists():
        print(f"  Reordering CSV not found at {reorder_csv_path} — skipping break-even plots. "
              "Run  python scripts/parse_results.py  first to generate it.")
        return

    df_time = pd.read_csv(reorder_csv_path)
    if df_time.empty or 'time_reordering_ms' not in df_time.columns:
        print("  No reordering timing data available — skipping.")
        return

    # Keep only timing columns we need
    df_time = df_time[['matrix', 'perm', 'time_reordering_ms']].copy()
    # Average over duplicate runs for the same (matrix, perm)
    df_time = df_time.groupby(['matrix', 'perm'], as_index=False)['time_reordering_ms'].mean()
    print(f"  Loaded {len(df_time)} reordering timing entries")

    # --- Baseline operation times ---
    df_baseline = (
        df[df['strategy'] == 'Original']
        .groupby(['matrix', 'kernel_id', 'n_cols'])['time_operation_ms']
        .mean()
        .reset_index()
        .rename(columns={'time_operation_ms': 'time_baseline_ms'})
    )

    # --- Reordered rows only ---
    df_reord = df[df['strategy'] != 'Original'].copy()
    if df_reord.empty:
        print("  No reordered operation data — skipping.")
        return

    # Merge baseline time
    df_reord = df_reord.merge(df_baseline, on=['matrix', 'kernel_id', 'n_cols'], how='inner')

    # Merge reordering time
    df_reord = df_reord.merge(df_time, on=['matrix', 'perm'], how='inner')

    if df_reord.empty:
        print("  No overlapping data between operations and reordering timing — skipping.")
        return

    # --- Compute break-even ---
    df_reord['time_saved_ms'] = df_reord['time_baseline_ms'] - df_reord['time_operation_ms']
    df_reord['breakeven_n'] = np.where(
        df_reord['time_saved_ms'] > 0,
        df_reord['time_reordering_ms'] / df_reord['time_saved_ms'],
        np.nan,
    )
    df_reord['harmful'] = df_reord['time_saved_ms'] <= 0

    n_valid = int((~df_reord['harmful']).sum())
    n_harmful = int(df_reord['harmful'].sum())
    print(f"  Break-even computed: {n_valid} valid, {n_harmful} harmful (never breaks even)")

    # --- Apply CLI filters ---
    n_cols_values = sorted(df_reord['n_cols'].unique())
    if args is not None and args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"  Warning: n_cols={args.n_cols} not in data; available: {n_cols_values}")
            return

    kernel_filter = args.kernel if args is not None else None

    # --- Collect and generate plots ---
    n_jobs = getattr(args, 'jobs', None) if args is not None else None
    tasks = []
    for n_cols in n_cols_values:
        df_nc = df_reord[df_reord['n_cols'] == n_cols]

        kernels = sorted(df_nc['kernel_id'].unique())
        if kernel_filter:
            kernels = [k for k in kernels if kernel_filter.lower() in k.lower()]
            if not kernels:
                continue

        for kernel in kernels:
            df_k = df_nc[df_nc['kernel_id'] == kernel]
            kernel_safe = pu.safe_filename(kernel)
            breakeven_dir = out_dir / f"n_cols_{int(n_cols)}" / kernel_safe / "breakeven"
            breakeven_dir.mkdir(parents=True, exist_ok=True)

            for perm_type in sorted(df_k['perm_type'].unique()):
                df_pt = df_k[df_k['perm_type'] == perm_type]
                strategies = sorted(df_pt['strategy'].unique())
                palette = pu.get_strategy_palette(strategies)

                df_valid = df_pt[~df_pt['harmful']]
                df_harm = df_pt[df_pt['harmful']]

                tasks.append((pu.breakeven_boxplot, dict(
                    df_valid=df_valid, df_harmful=df_harm,
                    x_col='strategy', y_col='breakeven_n',
                    output_path=breakeven_dir / f"breakeven_{perm_type}.png",
                    title=f"Break-even Operations — {perm_type}\n{kernel}  (n_cols={int(n_cols)})",
                    order=[s for s in strategies if s in df_valid['strategy'].unique()
                           or s in df_harm['strategy'].unique()],
                    palette=palette)))

    print(f"  Collected {len(tasks)} break-even plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)
    print(f"  Break-even plots saved under {out_dir}/n_cols_*/*/breakeven/")


def generate_reorder_analysis_plots(df_analysis, out_dir, n_jobs=None):
    """Generate reordering analysis plots (independent of kernel performance)."""

    print("\n=== Reordering Analysis ===")

    reorder_dir = out_dir / "reorder_analysis"

    # Process analysis data
    df = df_analysis.copy()
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    # Calculate improvements using the shared utility (extended mode includes
    # bandwidth_avg, max spreads, and blocks-per-row metrics)
    metrics_config = pu.build_metrics_config(df, include_extended=True)
    df = pu.compute_improvements(df, metrics_config)

    # Filter to reordered only
    df_reordered = df[df['strategy'] != 'Original']

    if df_reordered.empty:
        print("No reordered data for analysis plots")
        return

    strategies = sorted(df_reordered['strategy'].unique())

    # -----------------------------------------------------------------
    # All boxplot sections: (imp_col, title_template, subdir, filename_template)
    # -----------------------------------------------------------------
    boxplot_specs = [
        ('bandwidth_improvement',
         "Bandwidth Reduction (Original / Reordered) - {perm_type}",
         "bandwidth", "bandwidth_reduction_{perm_type}.png"),
        ('bandwidth_avg_improvement',
         "Avg Bandwidth Reduction (Original / Reordered) - {perm_type}",
         "bandwidth", "bandwidth_avg_reduction_{perm_type}.png"),
    ]

    for bs in [4, 8, 16, 32, 64, 128]:
        boxplot_specs.append((
            f'density_improvement_{bs}',
            f"Density Improvement (BS {bs})" + " - {perm_type}",
            "density", f"density_improvement_bs{bs}_{{perm_type}}.png"))

    for bs in [4, 8, 16, 32, 64, 128]:
        for prefix, name in [('avg', 'Avg'), ('max', 'Max')]:
            boxplot_specs.append((
                f'{prefix}_blocks_per_row_improvement_{bs}',
                f"{name} Blocks/Row Reduction (BS {bs})" + " - {perm_type}",
                "blocks_per_row",
                f"{prefix}_blocks_per_row_improvement_bs{bs}_{{perm_type}}.png"))

    for imp_col, name in [
        ('row_spread_improvement', 'Row Spread Reduction'),
        ('max_row_spread_improvement', 'Max Row Spread Reduction'),
        ('col_spread_improvement', 'Col Spread Reduction'),
        ('max_col_spread_improvement', 'Max Col Spread Reduction'),
        ('vertical_adjacency_improvement', 'Vertical Adjacency Improvement'),
    ]:
        boxplot_specs.append((
            imp_col, name + " - {perm_type}",
            "locality", f"{imp_col}_{{perm_type}}.png"))

    # Collect all boxplot tasks
    tasks = []
    perm_types = df_reordered['perm_type'].unique()
    for imp_col, title_template, subdir, filename_template in boxplot_specs:
        if imp_col not in df_reordered.columns:
            continue
        output_dir = reorder_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        for perm_type in perm_types:
            df_pt = df_reordered[df_reordered['perm_type'] == perm_type]
            tasks.append((pu.boxplot_by_category, dict(
                df=df_pt, x_col='strategy', y_col=imp_col,
                output_path=output_dir / filename_template.format(perm_type=perm_type),
                title=title_template.format(perm_type=perm_type),
                order=[s for s in strategies if s in df_pt['strategy'].unique()],
                baseline=1.0, log_y=True)))

    print(f"  Collected {len(tasks)} reorder analysis plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def _should_run(section: str, args) -> bool:
    """Decide whether *section* should run given CLI flags.

    Priority: --sections (fine-grained) > --only-* (coarse) > default (all).
    """
    if args.sections is not None:
        return section in args.sections
    # Legacy coarse flags
    kernel_sections = {'kernels', 'breakeven', 'original-scatter'}
    reorder_sections = {'reorder-analysis', 'reorderability', 'per-matrix', 'timing'}
    if args.only_kernels:
        return section in kernel_sections
    if args.only_reorder_analysis:
        return section in reorder_sections
    return True  # no filter → run everything


def main():
    args = parse_args()

    # Apply --random defaults (before any other processing)
    if args.random:
        if args.filter_config is None:
            args.filter_config = str(Path(__file__).resolve().parent / 'filter_config_random.yaml')
        if args.out is None:
            args.out = 'plots_random'
    if args.out is None:
        args.out = 'plots'

    # Validate mutually exclusive options (only matters for legacy flags)
    if args.sections is None and args.only_reorder_analysis and args.only_kernels:
        print("Error: --only-reorder-analysis and --only-kernels are mutually exclusive.")
        return

    # Create output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set professional publication style
    pu.set_professional_style()
    
    # -----------------------------------------------------------------
    # 1. Load & Filter Data (single pipeline from filter_config.yaml)
    # -----------------------------------------------------------------
    cli_overrides = {
        'operations_csv': args.operations,
        'analysis_csv': args.analysis,
        'matrices_list': args.matrices_list,
        'one_per_family': args.one_per_family,
        'min_size': args.min_size,
        'max_size': args.max_size,
    }
    # --include-rectangular means square_only=False
    if args.include_rectangular is not None:
        cli_overrides['square_only'] = not args.include_rectangular

    df, df_analysis, _cfg = pu.load_and_filter_data(
        config_path=args.filter_config,
        cli_overrides=cli_overrides,
    )
    
    # -----------------------------------------------------------------
    # 2. Process Data (add all derived columns)
    # -----------------------------------------------------------------
    has_ops = not df.empty
    if has_ops:
        print("\nProcessing data...")
        df = pu.prepare_full_dataframe(df)
        print(f"Unique kernels: {df['kernel_id'].unique().tolist()}")
        print(f"Unique n_cols: {sorted(df['n_cols'].unique())}")
    else:
        print("\nNo operations data — kernel/breakeven/original-scatter plots will be skipped.")

    # Helper for reordering CSV path (shared by breakeven & timing)
    def _reordering_csv():
        return _cfg.get('data', {}).get('reordering_csv',
                                        'results/results_reordering.csv')

    # -----------------------------------------------------------------
    # 3. Generate Plots  (each guarded by _should_run)
    # -----------------------------------------------------------------

    if has_ops and _should_run('kernels', args):
        print("\n" + "="*60)
        print("Generating kernel performance plots...")
        print("="*60)
        generate_kernel_plots(df, out_dir, args)

    if has_ops and _should_run('original-scatter', args):
        print("\n" + "="*60)
        print("Generating original-only grouped scatter plots...")
        print("="*60)
        generate_original_scatter_plots(df, out_dir, args)

    if has_ops and _should_run('breakeven', args):
        print("\n" + "="*60)
        print("Generating break-even analysis plots...")
        print("="*60)
        generate_breakeven_plots(df, out_dir, reordering_csv=_reordering_csv(),
                                 args=args)

    if _should_run('reorder-analysis', args):
        print("\n" + "="*60)
        print("Generating reordering analysis plots...")
        print("="*60)
        generate_reorder_analysis_plots(df_analysis, out_dir, n_jobs=args.jobs)
        
    if _should_run('reorderability', args):
        print("\n" + "="*60)
        print("Generating reorderability analysis plots...")
        print("="*60)
        generate_reorderability_plots(df_analysis, out_dir)

    if _should_run('per-matrix', args):
        print("\n" + "="*60)
        print("Generating per-matrix difficulty study...")
        print("="*60)
        generate_per_matrix_study(df_analysis, out_dir)

    if _should_run('timing', args):
        print("\n" + "="*60)
        print("Generating reordering timing analysis...")
        print("="*60)
        generate_reorder_timing_plots(df_analysis, out_dir,
                                      reordering_csv=_reordering_csv())
    
    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
