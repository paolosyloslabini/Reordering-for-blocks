"""
Plot Script - Clean Architecture

Simple iteration-based plotting script that:
1. Loads and processes data once
2. Applies filters
3. Iterates over dimensions (n_cols, kernel) to generate plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import plot_utils as pu
from settings import get_perm_display, get_perm_color, KERNEL_NAMES, GROUPED_SCATTER_EXCLUDE, PERMS, ALL_METRICS, BLOCK_SIZES, PALETTE, enabled_metrics, get_metric_display
from correlation_table import (compute_imp_correlations,
                               _enabled_improvement_metrics,
                               _density_improvement_metrics, _corr_method_tag,
                               _ordered_kernels)

# Axis caps for ratio scatter plots (speedup, improvement)
RATIO_YLIM = (1 / 7, 7)
RATIO_XLIM = (0.05, 10)  # x-axis bounds for improvement ratios


def parse_args():
    parser = argparse.ArgumentParser(description="Generate plots from analysis results.")

    # Random pipeline shortcut
    parser.add_argument("--random", action="store_true",
                        help="Use random-pipeline data (data_random paths from filter_config.yaml)")

    # Perm-type pipeline selection
    parser.add_argument("--row", action="store_true",
                        help="Use ROW perm_type pipeline (output to plots_row/ or plots_random_row/)")
    parser.add_argument("--symmetric", action="store_true",
                        help="Use SYMMETRIC perm_type pipeline (default)")

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
        'reorder-analysis', 'reorderability', 'improvability', 'per-matrix', 'timing',
        'singular-improvement',
        'profiles',
        'speedup-profiles',
        'aggregate-improvement',
        'pairwise',
        'imp-correlation',
        'partial-correlation',
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

            for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio',
                            'access_dist_reuse_distance_mean',
                            'access_dist_index_distance_mean']:
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

            for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio',
                            'access_dist_reuse_distance_mean',
                            'access_dist_index_distance_mean']:
                if loc_col in df_k.columns:
                    tasks.append((pu.scatter_publication, dict(
                        df=df_k, x_col=loc_col, y_col='gflops',
                        output_path=linear_dir / f"gflops_vs_{loc_col}_linear.png",
                        log_x=False, log_y=False, label=kernel_label)))

            # 2. Speedup Distribution
            df_reordered = df_k[df_k['strategy'] != 'Original']
            if not df_reordered.empty:
                tasks.append((pu.boxplot_by_category, dict(
                    df=df_reordered, x_col='strategy', y_col='speedup',
                    output_path=speedup_dir / f"speedup_boxplot.png",
                    title=f"Speedup — {kernel}",
                    order=pu.get_strategy_order(df_reordered),
                    baseline=1.0, log_y=True, ylim=(0.5, 5))))

            # 3. Binned Speedup Charts (best reorder per matrix, by the binned metric)
            df_valid = df_reordered.dropna(subset=['speedup'])
            if not df_valid.empty:
                for bs in [4, 8, 16, 32, 64, 128]:
                    imp_col = f'density_improvement_{bs}'
                    if imp_col in df_valid.columns:
                        df_imp = df_valid.dropna(subset=[imp_col])
                        if not df_imp.empty:
                            df_best = df_imp.loc[df_imp.groupby('matrix')[imp_col].idxmax()]
                            tasks.append((pu.binned_boxplot, dict(
                                df=df_best, bin_col=imp_col, value_col='speedup',
                                output_path=binned_dir / f"speedup_by_density_imp_bs{bs}.png",
                                title=f"Speedup by Density Improvement (BS {bs})\n{kernel}",
                                baseline=1.0)))

                for imp_col in ['row_spread_improvement', 'vertical_adjacency_improvement', 'bandwidth_improvement']:
                    if imp_col in df_valid.columns:
                        df_imp = df_valid.dropna(subset=[imp_col])
                        if not df_imp.empty:
                            df_best = df_imp.loc[df_imp.groupby('matrix')[imp_col].idxmax()]
                            tasks.append((pu.binned_boxplot, dict(
                                df=df_best, bin_col=imp_col, value_col='speedup',
                                output_path=binned_dir / f"speedup_by_{imp_col}.png",
                                title=f"Speedup by {pu.get_display_name(imp_col)}\n{kernel}",
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

        for loc_col in ['rel_row_spread', 'locality_vertical_adjacency_ratio',
                        'access_dist_reuse_distance_mean',
                        'access_dist_index_distance_mean']:
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

        _ratio_scatter_kw = dict(ylim=RATIO_YLIM, xlim=RATIO_XLIM,
                                  baseline_x=1.0, baseline_y=1.0,
                                  quadrant_colors=True)

        for bs in [4, 8, 16, 32, 64, 128]:
            imp_col = f'density_improvement_{bs}'
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_dir / f"speedup_vs_density_imp_bs{bs}.png",
                    group_labels=kernel_labels, log_x=False, log_y=False,
                    **_ratio_scatter_kw)))
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_log_dir / f"speedup_vs_density_imp_bs{bs}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True,
                    **_ratio_scatter_kw)))

        for imp_col in ['bandwidth_improvement', 'row_spread_improvement',
                        'vertical_adjacency_improvement',
                        'reuse_distance_improvement',
                        'index_distance_improvement']:
            if imp_col in df_nc_reordered.columns:
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_dir / f"speedup_vs_{imp_col}.png",
                    group_labels=kernel_labels, log_x=False, log_y=False,
                    **_ratio_scatter_kw)))
                tasks.append((pu.grouped_scatter_publication, dict(
                    df=df_nc_reordered, x_col=imp_col, y_col='speedup',
                    group_col='kernel_id', group_order=grouped_kernels,
                    output_path=grouped_imp_log_dir / f"speedup_vs_{imp_col}_loglog.png",
                    group_labels=kernel_labels, log_x=True, log_y=True,
                    **_ratio_scatter_kw)))

    print(f"\n  Collected {len(tasks)} kernel plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def generate_singular_improvement_plots(df, out_dir, args):
    """Improvement-vs-speedup scatter for each kernel individually.

    Produces one ``scatter_publication`` per (kernel, improvement metric)
    pair, in both linear and log-log variants.  Disabled by default;
    invoke with ``--sections singular-improvement``.
    """
    print("\n=== Singular Kernel Improvement vs Speedup ===")
    n_jobs = args.jobs

    n_cols_values = sorted(df['n_cols'].unique())
    if args.n_cols is not None:
        if args.n_cols in n_cols_values:
            n_cols_values = [args.n_cols]
        else:
            print(f"Warning: n_cols={args.n_cols} not found. Available: {n_cols_values}")
            return

    tasks = []

    for n_cols in n_cols_values:
        print(f"\n--- n_cols = {n_cols} ---")
        df_nc = df[df['n_cols'] == n_cols]

        kernels = sorted(df_nc['kernel_id'].unique())
        if args.kernel:
            kernels = [k for k in kernels if args.kernel.lower() in k.lower()]
            if not kernels:
                print(f"No kernels matching '{args.kernel}'")
                continue

        for kernel in kernels:
            df_k = df_nc[df_nc['kernel_id'] == kernel]
            df_reordered = df_k[df_k['strategy'] != 'Original']
            if len(df_reordered) < 2:
                continue

            kernel_safe = pu.safe_filename(kernel)
            kernel_label = KERNEL_NAMES.get(kernel, kernel)
            base = out_dir / f"n_cols_{int(n_cols)}" / kernel_safe

            imp_dir = base / "improvement_vs_speedup"
            imp_dir.mkdir(parents=True, exist_ok=True)
            imp_log_dir = base / "improvement_vs_speedup_loglog"
            imp_log_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Collecting tasks for kernel: {kernel}")

            _ratio_pres_kw = dict(ylim=RATIO_YLIM, xlim=RATIO_XLIM,
                                   baseline_x=1.0, baseline_y=1.0,
                                   quadrant_colors=True)

            for bs in [4, 8, 16, 32, 64, 128]:
                imp_col = f'density_improvement_{bs}'
                if imp_col in df_reordered.columns:
                    tasks.append((pu.scatter_presentation, dict(
                        df=df_reordered, x_col=imp_col, y_col='speedup',
                        output_path=imp_dir / f"speedup_vs_density_imp_bs{bs}.png",
                        log_x=False, log_y=False, label=kernel_label,
                        **_ratio_pres_kw)))
                    tasks.append((pu.scatter_presentation, dict(
                        df=df_reordered, x_col=imp_col, y_col='speedup',
                        output_path=imp_log_dir / f"speedup_vs_density_imp_bs{bs}_loglog.png",
                        log_x=True, log_y=True, label=kernel_label,
                        **_ratio_pres_kw)))

            for imp_col in ['bandwidth_improvement', 'row_spread_improvement',
                            'vertical_adjacency_improvement',
                            'reuse_distance_improvement',
                            'index_distance_improvement']:
                if imp_col in df_reordered.columns:
                    tasks.append((pu.scatter_presentation, dict(
                        df=df_reordered, x_col=imp_col, y_col='speedup',
                        output_path=imp_dir / f"speedup_vs_{imp_col}.png",
                        log_x=False, log_y=False, label=kernel_label,
                        **_ratio_pres_kw)))
                    tasks.append((pu.scatter_presentation, dict(
                        df=df_reordered, x_col=imp_col, y_col='speedup',
                        output_path=imp_log_dir / f"speedup_vs_{imp_col}_loglog.png",
                        log_x=True, log_y=True, label=kernel_label,
                        **_ratio_pres_kw)))

    print(f"\n  Collected {len(tasks)} singular improvement plot tasks")
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
        ('access_dist_reuse_distance_mean', False, 'Mean Reuse Distance'),
        ('access_dist_index_distance_mean', False, 'Mean Index Distance'),
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
        ax.axhline(y=1.0, color='#CC0000', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(RATIO_YLIM)
        pu.format_log_axes(ax, which='x', dense=False)
        pu.format_log_axes(ax, which='y', dense=True)
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
    ax.set_ylim(RATIO_YLIM)
    pu.format_log_axes(ax, which='x', dense=False)
    pu.format_log_axes(ax, which='y', dense=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path.name}")


def generate_improvability_summary_plots(df_analysis, out_dir):
    """CDF of best improvement per metric.

    For each structural metric, compute the best improvement ratio any
    reordering algorithm achieves for each matrix.  Plot a single CDF
    with one line per metric showing the fraction of matrices that achieve
    at least a given best-improvement ratio.
    """
    imp_dir = Path(out_dir) / "reorder_analysis" / "improvability"
    imp_dir.mkdir(parents=True, exist_ok=True)

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)

    metrics_to_analyze = [
        ('block_density_16', True, 'Block Density (16×16)'),
        ('locality_vertical_adjacency_ratio', True, 'Vertical Adjacency Ratio'),
        ('bandwidth_max', False, 'Bandwidth'),
        ('locality_avg_row_spread', False, 'Avg Row Spread'),
        ('access_dist_reuse_distance_mean', False, 'Mean Reuse Distance'),
        ('access_dist_index_distance_mean', False, 'Mean Index Distance'),
    ]

    # Compute best improvement per matrix per metric
    records = []
    for metric, higher_is_better, metric_display in metrics_to_analyze:
        if metric not in df.columns:
            print(f"  Skipping {metric}: not in data")
            continue

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
                improvement = best_val / baseline_val
            else:
                best_val = reordered[metric].min()
                improvement = baseline_val / best_val if best_val > 0 else np.nan

            if pd.isna(improvement):
                continue

            records.append({
                'matrix': matrix,
                'metric_display': metric_display,
                'best_improvement': improvement,
            })

    if not records:
        print("  No improvability data to plot")
        return

    results_df = pd.DataFrame(records)

    # Plot CDF — one line per metric
    fig, ax = plt.subplots(figsize=(10, 6))

    # Consistent metric colors (matching imp-correlation plots)
    _metric_color_map = {
        'bandwidth_max':                       '#6B9AC4',  # steel blue
        'locality_avg_row_spread':             '#D4A574',  # warm sand
        'locality_vertical_adjacency_ratio':   '#7DB58A',  # sage green
        'access_dist_reuse_distance_mean':     '#E8A838',  # amber
        'access_dist_index_distance_mean':     '#E05858',  # coral red
        'block_density_16':                    '#9B82B0',  # muted purple
    }

    for metric, _, metric_display in metrics_to_analyze:
        vals = np.sort(results_df.loc[
            results_df['metric_display'] == metric_display, 'best_improvement'
        ].dropna().values)
        if len(vals) == 0:
            continue
        # CCDF: fraction of matrices with improvement >= x
        fractions = np.arange(len(vals), 0, -1) / len(vals)
        color = _metric_color_map.get(metric, '#333333')
        ax.step(vals, fractions, where='post', label=metric_display, color=color,
                linewidth=1.8)

    ax.axvline(x=1.0, color='grey', linestyle='--', linewidth=1, alpha=0.7,
               label='No improvement')
    ax.set_xscale('log')
    pu.format_log_axes(ax, which='x', dense=False)
    ax.set_xlabel('Best Improvement Ratio (any algorithm)')
    ax.set_ylabel('Fraction of Matrices ≥ x')
    ax.set_title('Improvability CDF — Best Improvement per Metric')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = imp_dir / "cdf_best_improvement.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


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


def generate_reorder_timing_plots(df_analysis, out_dir, reordering_csv='results/results_reordering.csv', cfg=None):
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
    # Apply the same exclude_perms filter used for ops/analysis data
    exclude_perms = (cfg or {}).get('filters', {}).get('exclude_perms', [])
    if exclude_perms:
        df_time = df_time[~df_time['perm'].isin(exclude_perms)]
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

    Outputs one boxplot per (kernel, n_cols) to
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

            strategies = sorted(df_k['strategy'].unique())
            palette = pu.get_strategy_palette(strategies)

            df_valid = df_k[~df_k['harmful']]
            df_harm = df_k[df_k['harmful']]

            tasks.append((pu.breakeven_boxplot, dict(
                df_valid=df_valid, df_harmful=df_harm,
                x_col='strategy', y_col='breakeven_n',
                output_path=breakeven_dir / f"breakeven.png",
                title=f"Break-even Operations\n{kernel}  (n_cols={int(n_cols)})",
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
         "Bandwidth Reduction (Original / Reordered)",
         "bandwidth", "bandwidth_reduction.png"),
        ('bandwidth_avg_improvement',
         "Avg Bandwidth Reduction (Original / Reordered)",
         "bandwidth", "bandwidth_avg_reduction.png"),
    ]

    for bs in [4, 8, 16, 32, 64, 128]:
        boxplot_specs.append((
            f'density_improvement_{bs}',
            f"Density Improvement (BS {bs})",
            "density", f"density_improvement_bs{bs}.png"))

    for bs in [4, 8, 16, 32, 64, 128]:
        for prefix, name in [('avg', 'Avg'), ('max', 'Max')]:
            boxplot_specs.append((
                f'{prefix}_blocks_per_row_improvement_{bs}',
                f"{name} Blocks/Row Reduction (BS {bs})",
                "blocks_per_row",
                f"{prefix}_blocks_per_row_improvement_bs{bs}.png"))

    for imp_col, name in [
        ('row_spread_improvement', 'Row Spread Reduction'),
        ('max_row_spread_improvement', 'Max Row Spread Reduction'),
        ('col_spread_improvement', 'Col Spread Reduction'),
        ('max_col_spread_improvement', 'Max Col Spread Reduction'),
        ('vertical_adjacency_improvement', 'Vertical Adjacency Improvement'),
        ('reuse_distance_improvement', 'Reuse Distance Reduction'),
        ('reuse_distance_median_improvement', 'Median Reuse Distance Reduction'),
        ('index_distance_improvement', 'Index Distance Reduction'),
        ('index_distance_median_improvement', 'Median Index Distance Reduction'),
    ]:
        boxplot_specs.append((
            imp_col, name,
            "locality", f"{imp_col}.png"))

    # Collect all boxplot tasks
    tasks = []
    for imp_col, title_text, subdir, filename in boxplot_specs:
        if imp_col not in df_reordered.columns:
            continue
        output_dir = reorder_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((pu.boxplot_by_category, dict(
            df=df_reordered, x_col='strategy', y_col=imp_col,
            output_path=output_dir / filename,
            title=title_text,
            order=[s for s in strategies if s in df_reordered['strategy'].unique()],
            baseline=1.0, log_y=True)))

    print(f"  Collected {len(tasks)} reorder analysis plot tasks")
    pu.parallel_execute(tasks, n_jobs=n_jobs)


def _profile_perm_order(perm_values):
    """Canonical perm order for profile plots — Original drawn last (on top)."""
    seen = set()
    order = [p for p in list(PERMS)
             if p != 'None' and p in perm_values
             and not (p in seen or seen.add(p))]
    if 'None' in perm_values:
        order.append('None')
    return order


def _draw_profile_curve(ax, taus_asc, n_matrices, perm, xlim_lo, xlim_hi,
                         higher_is_better):
    """Draw one perm's Dolan-Moré profile curve on *ax*.

    *taus_asc* must be sorted ascending.  For higher_is_better metrics the
    survival function is plotted (x from 1→0); for lower_is_better the CDF
    is plotted (x from 1→∞).
    """
    n = len(taus_asc)
    color = get_perm_color(perm)
    label = get_perm_display(perm)
    ls = ':' if perm == 'None' else '-'
    lw = 2.5 if perm == 'None' else 1.5
    if perm == 'None':
        color = 'red'

    if higher_is_better:
        taus = taus_asc[::-1]                       # descending
        fracs = np.arange(1, n + 1) / n_matrices
        n_best = (taus >= 1.0 - 1e-9).sum()
        y_at_1 = n_best / n_matrices
        taus_rest = taus[n_best:]
        fracs_rest = fracs[n_best:]
        if len(taus_rest) > 0:
            ax.step(taus_rest, fracs_rest, where='pre',
                    label=label, color=color, linewidth=lw, linestyle=ls)
            ax.plot([xlim_hi, taus_rest[0]], [y_at_1, y_at_1],
                    color=color, linewidth=lw, linestyle=ls)
            ax.plot([taus_rest[-1], xlim_lo], [fracs_rest[-1], fracs_rest[-1]],
                    color=color, linewidth=lw, linestyle=ls)
        else:
            ax.plot([xlim_hi, xlim_lo], [y_at_1, y_at_1],
                    label=label, color=color, linewidth=lw, linestyle=ls)
    else:
        fracs = np.arange(1, n + 1) / n_matrices
        n_best = (taus_asc <= 1.0 + 1e-9).sum()
        y_at_1 = n_best / n_matrices
        taus_rest = taus_asc[n_best:]
        fracs_rest = fracs[n_best:]
        if len(taus_rest) > 0:
            ax.step(taus_rest, fracs_rest, where='pre',
                    label=label, color=color, linewidth=lw, linestyle=ls)
            ax.plot([xlim_lo, taus_rest[0]], [y_at_1, y_at_1],
                    color=color, linewidth=lw, linestyle=ls)
            ax.plot([taus_rest[-1], xlim_hi], [fracs_rest[-1], fracs_rest[-1]],
                    color=color, linewidth=lw, linestyle=ls)
        else:
            ax.plot([xlim_lo, xlim_hi], [y_at_1, y_at_1],
                    label=label, color=color, linewidth=lw, linestyle=ls)


def generate_profile_plots(df_analysis, out_dir):
    """Generate Dolan-Moré performance-profile plots.

    For each metric and solver (perm) s, compute the ratio relative to the
    best solver on each matrix m:
        higher_is_better:  r_{m,s} = val_{m,s} / max_{s'} val_{m,s'}  ∈ (0, 1]
        lower_is_better:   r_{m,s} = val_{m,s} / min_{s'} val_{m,s'}  ∈ [1, ∞)

    In both cases r = 1 means s is the best solver on that matrix.

    The profile plots:
      - higher_is_better: survival function P(ratio >= τ), x from 1 toward 0
      - lower_is_better:  CDF P(ratio <= τ), x from 1 toward ∞

    Higher curve = better solver in both cases.

    All permutations *and* the original ordering (None) are included as
    competing solvers.
    """
    # Same raw metrics as the correlation heatmaps (no bandwidth).
    profile_metrics = [
        'bandwidth_max',
        'locality_avg_col_spread',
        'locality_vertical_adjacency_ratio',
        'locality_profile',
        'access_dist_reuse_distance_mean',
        'access_dist_index_distance_mean',
    ]
    profile_metrics = [m for m in profile_metrics
                       if m in df_analysis.columns
                       and ALL_METRICS.get(m, {}).get('higher_is_better') is not None]
    if not profile_metrics:
        print("  No usable metrics — skipping profile plots.")
        return

    # Work directly with raw metric values (including perm='None').
    # The DataFrame is already filtered to a single perm_type by
    # split_by_perm_type(), so no per-perm_type grouping is needed.
    base_cols = ['matrix', 'perm', 'perm_type']
    group_df = df_analysis[base_cols + profile_metrics].copy()

    if group_df.empty:
        print("  No data — skipping profile plots.")
        return

    # Keep only matrices that have all perms present
    all_perms = group_df['perm'].unique()
    n_perms = len(all_perms)
    perm_counts = group_df.groupby('matrix')['perm'].nunique()
    complete_matrices = perm_counts[perm_counts == n_perms].index
    group_df = group_df[group_df['matrix'].isin(complete_matrices)]
    print(f"  Restricted to {len(complete_matrices)} matrices with all {n_perms} perms")

    if group_df.empty:
        print("  No complete matrices — skipping profile plots.")
        return

    # For each metric, compute Dolan-Moré performance ratios
    profile_data = {}  # metric -> (n_matrices, {perm -> sorted_ratios})
    for m in profile_metrics:
        sub = group_df[['matrix', 'perm', m]].dropna(subset=[m])
        # Filter out zero / negative values
        sub = sub[sub[m] > 0]
        if sub.empty:
            profile_data[m] = (0, {})
            continue

        # ratio = value / best:
        #   higher_is_better → ∈ (0, 1], lower_is_better → ∈ [1, ∞)
        hib = ALL_METRICS[m]['higher_is_better']
        if hib:
            best = sub.groupby('matrix')[m].max().rename('best')
        else:
            best = sub.groupby('matrix')[m].min().rename('best')
        sub = sub.merge(best, on='matrix')
        # relative value (value / best)
        sub['tau'] = sub[m] / sub['best']

        n_matrices = sub['matrix'].nunique()
        perm_ratios = {}
        for perm, perm_sub in sub.groupby('perm'):
            perm_ratios[perm] = np.sort(perm_sub['tau'].values)
        profile_data[m] = (n_matrices, perm_ratios)

    # Layout: 3 columns
    n_metrics = len(profile_metrics)
    ncols = min(3, n_metrics)
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    perm_order = _profile_perm_order(set(group_df['perm'].unique()))

    for idx, m in enumerate(profile_metrics):
        ax = axes[idx // ncols][idx % ncols]
        n_matrices, data = profile_data[m]
        if n_matrices == 0:
            continue
        hib = ALL_METRICS[m]['higher_is_better']

        # Compute xlim early so extensions can reach plot borders
        if hib:
            xlim_lo, xlim_hi = 2**-4, 1.05
        else:
            all_taus = np.concatenate([v for v in data.values()])
            max_tau = min(np.percentile(all_taus, 99) * 1.2, 2**6)
            xlim_lo, xlim_hi = 0.95, max(max_tau, 2)

        for perm in perm_order:
            if perm not in data or len(data[perm]) == 0:
                continue
            _draw_profile_curve(ax, data[perm], n_matrices, perm,
                                xlim_lo, xlim_hi, hib)

        ax.set_title(get_metric_display(m), fontsize=10)
        ax.set_xlabel(f"{get_metric_display(m)} relative to the best")
        ax.set_ylabel('Fraction of matrices')
        ax.set_xscale('log', base=2)
        ax.set_ylim(0, 1.05)
        if hib:
            ax.set_xlim(xlim_lo, xlim_hi)
            ax.invert_xaxis()
        else:
            ax.set_xlim(xlim_lo, xlim_hi)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_metrics, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend above the plots
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(6, len(handles)), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout()

    prof_dir = out_dir / 'reorder_analysis' / 'profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(prof_dir / f'profiles.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved profiles.pdf/png")

    # --- Standalone block_density_16 profile plot ---
    # This uses its own data (not from profile_metrics above).
    bd_metric = 'block_density_16'
    if bd_metric in df_analysis.columns and ALL_METRICS.get(bd_metric, {}).get('higher_is_better') is not None:
        sub_bd = group_df[['matrix', 'perm']].copy()
        sub_bd[bd_metric] = df_analysis.set_index(['matrix', 'perm'])[bd_metric].reindex(
            pd.MultiIndex.from_frame(sub_bd[['matrix', 'perm']])).values
        sub_bd = sub_bd.dropna(subset=[bd_metric])
        sub_bd = sub_bd[sub_bd[bd_metric] > 0]
        if not sub_bd.empty:
            best_bd = sub_bd.groupby('matrix')[bd_metric].max().rename('best')
            sub_bd = sub_bd.merge(best_bd, on='matrix')
            sub_bd['tau'] = sub_bd[bd_metric] / sub_bd['best']
            n_matrices_bd = sub_bd['matrix'].nunique()
            data_bd = {}
            for perm, perm_sub in sub_bd.groupby('perm'):
                data_bd[perm] = np.sort(perm_sub['tau'].values)

            fig_bd, ax_bd = plt.subplots(figsize=(6, 4.5))
            xlim_lo_bd, xlim_hi_bd = 2**-4, 1.05
            for perm in perm_order:
                if perm not in data_bd or len(data_bd[perm]) == 0:
                    continue
                _draw_profile_curve(ax_bd, data_bd[perm], n_matrices_bd, perm,
                                    xlim_lo_bd, xlim_hi_bd, higher_is_better=True)
            ax_bd.set_xlabel(f'{get_metric_display(bd_metric)} relative to the best')
            ax_bd.set_ylabel('Fraction of matrices')
            ax_bd.set_xscale('log', base=2)
            ax_bd.set_ylim(0, 1.05)
            ax_bd.set_xlim(xlim_lo_bd, xlim_hi_bd)
            ax_bd.invert_xaxis()
            ax_bd.grid(True, alpha=0.3)
            handles_bd, labels_bd = ax_bd.get_legend_handles_labels()

            fig_bd.tight_layout()
            for ext in ('pdf', 'png'):
                fig_bd.savefig(prof_dir / f'profile_block_density_16.{ext}',
                               bbox_inches='tight', dpi=150)
            plt.close(fig_bd)
            print(f"  Saved profile_block_density_16.pdf/png")

            # --- Standalone legend ---
            ncol_leg = (len(handles_bd) + 1) // 2
            fig_leg = plt.figure(figsize=(6, 1.2))
            fig_leg.legend(handles_bd, labels_bd, loc='center',
                           ncol=ncol_leg, fontsize=9, frameon=False)
            for ext in ('pdf', 'png'):
                fig_leg.savefig(prof_dir / f'profile_legend.{ext}',
                                bbox_inches='tight', dpi=150)
            plt.close(fig_leg)
            print(f"  Saved profile_legend.pdf/png")


def generate_profile_aggregate_boxplot(df_analysis, out_dir):
    """Generate aggregated boxplot of improvement ratios across profile metrics.

    For each reordering algorithm and each of the 6 profile metrics, shows
    a boxplot of the improvement ratio (reordered / original, direction-aware)
    across all matrices. Algorithms on x-axis, 6 colored boxes per algorithm.
    """
    print("\n=== Aggregated Profile Improvement Boxplot ===")

    profile_metrics = [
        'bandwidth_max',
        'locality_avg_col_spread',
        'locality_vertical_adjacency_ratio',
        'locality_profile',
        'access_dist_reuse_distance_mean',
        'access_dist_index_distance_mean',
        'block_density_16',
    ]
    profile_metrics = [m for m in profile_metrics
                       if m in df_analysis.columns
                       and ALL_METRICS.get(m, {}).get('higher_is_better') is not None]
    if not profile_metrics:
        print("  No usable metrics — skipping aggregate boxplot.")
        return

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)
    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    # Compute improvement ratios for each metric
    metrics_config = {
        'bandwidth_max': {'improvement_name': 'bandwidth_improvement', 'higher_is_better': False},
        'locality_avg_col_spread': {'improvement_name': 'col_spread_improvement', 'higher_is_better': False},
        'locality_vertical_adjacency_ratio': {'improvement_name': 'vertical_adjacency_improvement', 'higher_is_better': True},
        'locality_profile': {'improvement_name': 'profile_improvement', 'higher_is_better': False},
        'access_dist_reuse_distance_mean': {'improvement_name': 'reuse_distance_improvement', 'higher_is_better': False},
        'access_dist_index_distance_mean': {'improvement_name': 'index_distance_improvement', 'higher_is_better': False},
        'block_density_16': {'improvement_name': 'density_improvement_16', 'higher_is_better': True},
    }
    metrics_config = {k: v for k, v in metrics_config.items() if k in profile_metrics}
    df = pu.compute_improvements(df, metrics_config)

    # Exclude Original (improvement = 1 by definition)
    df = df[df['strategy'] != 'Original']

    # Melt into long form: one row per (matrix, strategy, metric)
    imp_cols = [v['improvement_name'] for v in metrics_config.values()]
    imp_display = {}
    for m, cfg in metrics_config.items():
        display = get_metric_display(m)
        # Shorten for readability
        for suffix in [' Improvement', ' Reduction']:
            display = display.replace(suffix, '')
        imp_display[cfg['improvement_name']] = display

    id_cols = ['matrix', 'strategy']
    df_long = df[id_cols + imp_cols].melt(
        id_vars=id_cols, value_vars=imp_cols,
        var_name='metric', value_name='improvement'
    )
    df_long['metric_display'] = df_long['metric'].map(imp_display)
    df_long = df_long.dropna(subset=['improvement'])
    # Remove inf / extreme outliers
    df_long = df_long[np.isfinite(df_long['improvement'])]

    if df_long.empty:
        print("  No improvement data — skipping aggregate boxplot.")
        return

    # Metric color map (consistent with imp-correlation plots)
    _metric_color_map = {
        'bandwidth_improvement': '#6B9AC4',
        'col_spread_improvement': '#D4A574',
        'vertical_adjacency_improvement': '#7DB58A',
        'profile_improvement': '#C47A7A',
        'reuse_distance_improvement': '#E8A838',
        'index_distance_improvement': '#E05858',
        'density_improvement_16': '#9B82B0',
    }
    metric_palette = {imp_display[k]: _metric_color_map[k]
                      for k in imp_cols if k in _metric_color_map}

    # Strategy order (exclude Original)
    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_long['strategy'].unique()]

    metric_order = [imp_display[c] for c in imp_cols if c in imp_display]

    n_strategies = len(strat_order)
    fig, ax = plt.subplots(figsize=(max(14, n_strategies * 1.5), 6))

    import seaborn as sns
    sns.boxplot(
        data=df_long, x='strategy', y='improvement', hue='metric_display',
        order=strat_order, hue_order=metric_order, palette=metric_palette,
        ax=ax, fliersize=1.5, linewidth=0.8,
        whis=(5, 95), showfliers=False,
        medianprops=dict(color='red', linewidth=0.8),
    )

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('')
    ax.set_ylabel('Improvement ratio')
    ax.set_title('Structural Metric Improvement by Reordering Algorithm')

    ax.set_yscale('log')
    p05 = df_long['improvement'].quantile(0.02)
    p95 = df_long['improvement'].quantile(0.98)
    ax.set_ylim(p05 / 1.3, p95 * 1.3)
    pu.format_log_axes(ax, which='y')

    ax.legend(title=None, bbox_to_anchor=(0.5, -0.18), loc='upper center',
              ncol=len(metric_order), fontsize=8, frameon=False)
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()

    agg_dir = out_dir / 'reorder_analysis' / 'aggregate_improvement'
    agg_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(agg_dir / f'aggregate_improvement_boxplot.{ext}',
                    bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved aggregate_improvement_boxplot.pdf/png")


def generate_speedup_aggregate_boxplot(df, out_dir, args=None):
    """Generate aggregated boxplot of kernel speedups by reordering algorithm.

    For each algorithm and each kernel, shows a boxplot of the speedup
    across all matrices. Algorithms on x-axis, one colored box per kernel.
    One figure per n_cols value.
    """
    print("\n=== Aggregated Speedup Boxplot ===")

    if 'speedup' not in df.columns:
        print("  No speedup column — skipping aggregate speedup boxplot.")
        return

    import seaborn as sns

    # Exclude Original and cuSparse BSR (block-format outlier)
    df_reord = df[(df['strategy'] != 'Original')
                  & (df['kernel_id'] != 'CUSPARSE_SPMM_BSR_bs32')].copy()

    # Strategy order
    strat_order = [get_perm_display(p) for p in PERMS if p != 'None']
    strat_order = [s for s in strat_order if s in df_reord['strategy'].unique()]

    # Kernel order and display names
    kernels = sorted(df_reord['kernel_id'].unique())
    kernel_display = {k: KERNEL_NAMES.get(k, k) for k in kernels}

    # Distinct kernel palette (deliberately different from the reorder-analysis
    # aggregate improvement palette so the two chart types are easy to tell apart)
    _kernel_colors = [
        '#1B9E77', '#D95F02', '#7570B3', '#E7298A',
        '#66A61E', '#E6AB02', '#A6761D', '#666666',
    ]
    kernel_names_ordered = [kernel_display[k] for k in kernels]
    kernel_palette = {name: _kernel_colors[i % len(_kernel_colors)]
                      for i, name in enumerate(kernel_names_ordered)}

    agg_dir = out_dir / 'aggregate_speedup'
    agg_dir.mkdir(parents=True, exist_ok=True)

    for nc in sorted(df_reord['n_cols'].unique()):
        df_nc = df_reord[df_reord['n_cols'] == nc].copy()
        df_nc['kernel_display'] = df_nc['kernel_id'].map(kernel_display)

        df_nc = df_nc.dropna(subset=['speedup'])
        df_nc = df_nc[np.isfinite(df_nc['speedup'])]
        if df_nc.empty:
            continue

        n_strategies = len(strat_order)
        fig, ax = plt.subplots(figsize=(max(14, n_strategies * 1.5), 6))

        sns.boxplot(
            data=df_nc, x='strategy', y='speedup', hue='kernel_display',
            order=strat_order, hue_order=kernel_names_ordered,
            palette=kernel_palette,
            ax=ax, fliersize=1.5, linewidth=0.8,
            whis=(5, 95), showfliers=False,
            medianprops=dict(color='red', linewidth=0.8),
        )

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Kernel Speedup by Reordering Algorithm (n_cols={int(nc)})')

        ax.set_yscale('log')
        p05 = df_nc['speedup'].quantile(0.02)
        p95 = df_nc['speedup'].quantile(0.98)
        ax.set_ylim(p05 / 1.3, p95 * 1.3)
        pu.format_log_axes(ax, which='y')

        ax.legend(title=None, bbox_to_anchor=(0.5, -0.18), loc='upper center',
                  ncol=len(kernel_names_ordered), fontsize=8, frameon=False)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            fig.savefig(agg_dir / f'aggregate_speedup_boxplot_nc{int(nc)}.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved aggregate_speedup_boxplot_nc{int(nc)}.pdf/png")

    # --- Alternative: kernels on x-axis, reordering algorithms as hue ---
    print("\n=== Aggregated Speedup Boxplot (by kernel) ===")
    strat_palette = pu.get_strategy_palette()
    # Remove Original from palette
    strat_palette = {k: v for k, v in strat_palette.items() if k != 'Original'}

    for nc in sorted(df_reord['n_cols'].unique()):
        df_nc = df_reord[df_reord['n_cols'] == nc].copy()
        df_nc['kernel_display'] = df_nc['kernel_id'].map(kernel_display)

        df_nc = df_nc.dropna(subset=['speedup'])
        df_nc = df_nc[np.isfinite(df_nc['speedup'])]
        if df_nc.empty:
            continue

        n_kernels = len(kernel_names_ordered)
        fig, ax = plt.subplots(figsize=(max(14, n_kernels * 2.0), 6))

        sns.boxplot(
            data=df_nc, x='kernel_display', y='speedup', hue='strategy',
            order=kernel_names_ordered, hue_order=strat_order,
            palette=strat_palette,
            ax=ax, fliersize=1.5, linewidth=0.8,
            whis=(5, 95), showfliers=False,
            medianprops=dict(color='red', linewidth=0.8),
        )

        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_xlabel('')
        ax.set_ylabel('Speedup')
        ax.set_title(f'Reordering Speedup by Kernel (n_cols={int(nc)})')

        ax.set_yscale('log')
        p05 = df_nc['speedup'].quantile(0.02)
        p95 = df_nc['speedup'].quantile(0.98)
        ax.set_ylim(p05 / 1.3, p95 * 1.3)
        pu.format_log_axes(ax, which='y')

        ax.legend(title=None, bbox_to_anchor=(0.5, -0.18), loc='upper center',
                  ncol=min(len(strat_order), 6), fontsize=8, frameon=False)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            fig.savefig(agg_dir / f'aggregate_speedup_by_kernel_nc{int(nc)}.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved aggregate_speedup_by_kernel_nc{int(nc)}.pdf/png")


def generate_speedup_profile_plots(df, out_dir, args=None):
    """Generate Dolan-Moré performance-profile plots of kernel speedups.

    For each kernel and n_cols, compute the performance ratio of each
    strategy's speedup relative to the best strategy on each matrix:
        r_{m,s} = speedup_{m,s} / max_{s'} speedup_{m,s'}   ∈ (0, 1]

    Then plot the survival function P(ratio >= τ) vs τ.
    Higher curve = better strategy.  Original (speedup=1) is included.

    One grid figure per n_cols, with one subplot per kernel.
    """
    if 'speedup' not in df.columns:
        print("  No speedup column — skipping speedup profile plots.")
        return

    prof_dir = out_dir / 'speedup_profiles'
    prof_dir.mkdir(parents=True, exist_ok=True)

    perm_order = _profile_perm_order(set(df['perm'].unique()))
    strategy_order = [get_perm_display(p) for p in perm_order]

    for nc in sorted(df['n_cols'].unique()):
        df_nc = df[df['n_cols'] == nc]
        kernel_ids = sorted(df_nc['kernel_id'].unique())
        if not kernel_ids:
            continue

        ncols_grid = min(3, len(kernel_ids))
        nrows_grid = (len(kernel_ids) + ncols_grid - 1) // ncols_grid
        fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                 figsize=(5 * ncols_grid, 4 * nrows_grid),
                                 squeeze=False)

        for idx, kid in enumerate(kernel_ids):
            ax = axes[idx // ncols_grid][idx % ncols_grid]
            df_k = df_nc[df_nc['kernel_id'] == kid].copy()

            # Drop NaN / non-positive speedups
            df_k = df_k.dropna(subset=['speedup'])
            df_k = df_k[df_k['speedup'] > 0]

            if df_k.empty:
                ax.set_visible(False)
                continue

            # Keep only matrices with all strategies present
            all_strats = df_k['perm'].unique()
            n_strats = len(all_strats)
            strat_counts = df_k.groupby('matrix')['perm'].nunique()
            complete = strat_counts[strat_counts == n_strats].index
            df_k = df_k[df_k['matrix'].isin(complete)]

            if df_k.empty:
                ax.set_visible(False)
                continue

            # Compute ratio to best
            best = df_k.groupby('matrix')['speedup'].max().rename('best')
            df_k = df_k.merge(best, on='matrix')
            df_k['tau'] = df_k['speedup'] / df_k['best']

            n_matrices = df_k['matrix'].nunique()

            for perm in perm_order:
                sub = df_k[df_k['perm'] == perm]
                if sub.empty:
                    continue
                taus_asc = np.sort(sub['tau'].values)
                _draw_profile_curve(ax, taus_asc, n_matrices, perm,
                                    xlim_lo=-0.02, xlim_hi=1.05,
                                    higher_is_better=True)

            display_name = KERNEL_NAMES.get(kid, kid)
            ax.set_title(display_name, fontsize=10)
            ax.set_xlabel('Speedup relative to the best')
            ax.set_ylabel('Fraction of matrices')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(-0.02, 1.05)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(len(kernel_ids), nrows_grid * ncols_grid):
            axes[idx // ncols_grid][idx % ncols_grid].set_visible(False)

        # Shared legend — collect from all subplots to catch all strategies
        all_handles, all_labels = [], []
        seen = set()
        for ax_row in axes:
            for ax in ax_row:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l not in seen:
                        seen.add(l)
                        all_handles.append(h)
                        all_labels.append(l)
        # Sort by canonical strategy order
        label_order = {s: i for i, s in enumerate(strategy_order)}
        pairs = sorted(zip(all_handles, all_labels),
                       key=lambda p: label_order.get(p[1], 999))
        if pairs:
            all_handles, all_labels = zip(*pairs)
        fig.legend(all_handles, all_labels, loc='upper center',
                   ncol=min(6, len(all_handles)), fontsize=9,
                   bbox_to_anchor=(0.5, 1.02))

        fig.tight_layout()

        for ext in ('pdf', 'png'):
            fig.savefig(prof_dir / f'speedup_profiles_nc{nc}.{ext}',
                        bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved speedup_profiles_nc{nc}.pdf/png")


def _pairwise_for_group(df, order, metrics, group_label, output_dir):
    """Build per-metric and aggregate pairwise heatmaps for one group of rows.

    Uses pairwise-complete logic: for each cell (i, j), only matrices where
    both algorithms i and j have data are considered.
    """
    for metric in metrics:
        hib = ALL_METRICS[metric]['higher_is_better']
        metric_display = get_metric_display(metric)

        # Pivot without dropping NaN — each column may have different coverage
        pivot = df.pivot_table(index='matrix', columns='strategy', values=metric)
        if pivot.empty:
            print(f"    Skipping {metric}: no data")
            continue

        win_frac = pd.DataFrame(np.nan, index=order, columns=order)
        n_pairs = pd.DataFrame(0, index=order, columns=order)

        for i, si in enumerate(order):
            if si not in pivot.columns:
                continue
            for j, sj in enumerate(order):
                if i == j or sj not in pivot.columns:
                    continue
                # Only matrices where both si and sj have values
                mask = pivot[si].notna() & pivot[sj].notna()
                n = mask.sum()
                if n < 2:
                    continue
                if hib:
                    wins = (pivot.loc[mask, si] > pivot.loc[mask, sj]).sum()
                else:
                    wins = (pivot.loc[mask, si] < pivot.loc[mask, sj]).sum()
                win_frac.loc[si, sj] = wins / n
                n_pairs.loc[si, sj] = n

        n_min = int(n_pairs.values[n_pairs.values > 0].min()) if (n_pairs.values > 0).any() else 0
        n_max = int(n_pairs.values.max())

        pu.pairwise_heatmap(
            win_frac,
            output_dir / f"pairwise_{pu.safe_filename(metric)}_{group_label}.png",
            title=f"Pairwise Win Rate — {metric_display} ({group_label})\n({n_min}–{n_max} matrices per pair)",
        )


def generate_pairwise_heatmap(df_analysis, out_dir):
    """Generate pairwise win/loss heatmaps comparing reordering algorithms.

    For each structural metric, builds a matrix where cell (i, j) is the
    fraction of test matrices on which algorithm i strictly beats algorithm j.
    Uses pairwise-complete logic. Random reordering is excluded.
    """
    print("\n=== Pairwise Win/Loss Heatmaps ===")

    pairwise_dir = out_dir / "reorder_analysis" / "pairwise"
    pairwise_dir.mkdir(parents=True, exist_ok=True)

    df = df_analysis.copy()
    df['perm'] = df['perm'].fillna('None').astype(str)

    df['strategy'] = df['perm'].apply(
        lambda x: 'Original' if x == 'None' else get_perm_display(x)
    )

    # Exclude Random
    df = df[df['strategy'] != 'Random']

    # Metrics to compare
    profile_metrics = enabled_metrics()
    if 'bandwidth_max' not in profile_metrics:
        profile_metrics = ['bandwidth_max'] + profile_metrics
    profile_metrics = [m for m in profile_metrics
                       if m in df.columns
                       and ALL_METRICS.get(m, {}).get('higher_is_better') is not None]

    if not profile_metrics:
        print("  No usable metrics — skipping.")
        return

    # Data is already homogeneous (single perm_type) with Original rows
    # duplicated by split_by_perm_type — no per-perm_type loop needed.
    n_matrices = df['matrix'].nunique()
    n_strategies = df['strategy'].nunique()
    print(f"  {n_matrices} matrices, {n_strategies} strategies (pairwise-complete)")

    if df.empty:
        print("  No data — skipping.")
        return

    order = list(dict.fromkeys(s for s in pu.get_strategy_order(df) if s != 'Random'))
    _pairwise_for_group(df, order, profile_metrics, 'all', pairwise_dir)

    print(f"  Pairwise heatmaps saved to {pairwise_dir}")


def generate_imp_correlation_plots(df, out_dir):
    """Generate grouped bar charts of improvement-metric correlations.

    Uses the correlation method from config (pearson/spearman/kendall).
    Generates both linear and log-log variants.
    One image per (scale, n_cols) combination.
    x-axis: kernels, y-axis: correlation value, one bar per metric.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    metric_order = [get_metric_display(m) for m in imp_metrics]
    n_metrics = len(metric_order)
    n_kernels = len(kernel_order)

    # Muted palette: each metric gets a distinct hue, density = purple
    _metric_color_map = {
        'bandwidth_improvement': '#6B9AC4',          # steel blue
        'col_spread_improvement': '#D4A574',          # warm sand
        'vertical_adjacency_improvement': '#7DB58A',  # sage green
        'profile_improvement': '#C47A7A',             # dusty rose
        'reuse_distance_improvement': '#E8A838',      # amber
        'index_distance_improvement': '#E05858',      # coral red
    }
    # Any density_improvement_* metric gets muted purple
    metric_colors = {}
    for m, display in zip(imp_metrics, metric_order):
        if m in _metric_color_map:
            metric_colors[display] = _metric_color_map[m]
        else:
            metric_colors[display] = '#9B82B0'  # muted purple for density

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_metrics
            x = np.arange(n_kernels)

            for mi, mc in enumerate(imp_metrics):
                col = f'{mc}_corr'
                display = get_metric_display(mc)
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (mi - (n_metrics - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=display, color=metric_colors[display],
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel(f'{corr_display} Correlation with Speedup')
            ax.set_title(f'Improvement–Speedup Correlation{scale_label}  '
                         f'(n_cols={int(n_cols)})')
            ax.legend(title='Metric', fontsize=9, title_fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_correlation_bars_{tag}{scale_suffix}'
                     f'_ncols_{int(n_cols)}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def generate_imp_ncols_correlation_plots(df, out_dir):
    """Generate per-metric bar charts showing how improvement–speedup correlation varies with n_cols.

    One image per (metric, scale) combination.
    x-axis: kernels, grouped bars: one per n_cols value, y-axis: correlation.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_kernels = len(kernel_order)
    n_ncols = len(n_cols_values)

    # Teal/green shades with distinct hatches
    _teal_shades = ['#b2dfdb', '#4db6ac', '#00897b', '#004d40']
    _hatches = ['', '///', '...', 'xxx', '\\\\\\', '+++']
    ncols_colors = dict(zip(n_cols_values, _teal_shades[:n_ncols]))
    ncols_hatches = dict(zip(n_cols_values, _hatches[:n_ncols]))

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        # Pre-compute correlations for all n_cols values
        corr_by_ncols = {}
        for n_cols in n_cols_values:
            corr_by_ncols[n_cols] = compute_imp_correlations(
                df, n_cols, imp_metrics, kernels,
                method=method, log_transform=log_transform)

        for mc in imp_metrics:
            col = f'{mc}_corr'
            metric_display = get_metric_display(mc)
            metric_safe = pu.safe_filename(mc)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_ncols
            x = np.arange(n_kernels)

            for ni, n_cols in enumerate(n_cols_values):
                corr_df = corr_by_ncols[n_cols]
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (ni - (n_ncols - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=f'n_cols={int(n_cols)}',
                       color=ncols_colors[n_cols],
                       hatch=ncols_hatches[n_cols],
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel(f'{corr_display} Correlation with Speedup')
            ax.set_title(f'{metric_display} — Improvement–Speedup Correlation'
                         f' across n_cols{scale_label}')
            ax.legend(title='$n_{cols}$', fontsize=9, title_fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_ncols_bars_{tag}{scale_suffix}_{metric_safe}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def generate_imp_blocksize_correlation_plots(df, out_dir):
    """Generate per-kernel bar charts showing density-improvement–speedup correlation across block sizes.

    One image per (kernel, scale) combination.
    x-axis: block sizes, y-axis: correlation value.
    Mirrors the LaTeX tables from generate_imp_blocksize_tables in correlation_table.py.
    """
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    bd_imp_metrics = _density_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    kernel_order = [KERNEL_NAMES[k] for k in kernels if k in KERNEL_NAMES]
    n_kernels = len(kernel_order)
    n_block_sizes = len(bd_imp_metrics)
    block_labels = [f'{bs}' for bs in BLOCK_SIZES[:n_block_sizes]]

    # Purple/pink shades from light to dark, with distinct hatches
    _purple_shades = ['#e8c8e8', '#d4a0d4', '#b06cb0', '#8c3f8c', '#6a1b6a', '#3d003d']
    _hatches = ['', '///', '...', 'xxx', '\\\\\\', '+++']
    bs_colors = dict(zip(block_labels, _purple_shades[:n_block_sizes]))
    bs_hatches = dict(zip(block_labels, _hatches[:n_block_sizes]))

    plot_dir = Path(out_dir) / 'imp_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for log_transform in [False, True]:
        scale_suffix = '_loglog' if log_transform else ''
        scale_label = ' (log-log)' if log_transform else ''

        for n_cols in n_cols_values:
            corr_df = compute_imp_correlations(
                df, n_cols, bd_imp_metrics, kernels,
                method=method, log_transform=log_transform)

            fig, ax = plt.subplots(figsize=(12, 6))

            bar_width = 0.7 / n_block_sizes
            x = np.arange(n_kernels)

            for bi, (metric, label) in enumerate(zip(bd_imp_metrics, block_labels)):
                col = f'{metric}_corr'
                vals = []
                for kernel in kernels:
                    row = corr_df[corr_df['kernel'] == kernel]
                    if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
                        vals.append(0)
                    else:
                        vals.append(row.iloc[0][col])
                offset = (bi - (n_block_sizes - 1) / 2) * bar_width
                ax.bar(x + offset, vals, bar_width * 0.9,
                       label=f'{label}×{label}',
                       color=bs_colors[label],
                       hatch=bs_hatches[label],
                       edgecolor='black', linewidth=0.4)

            ax.axhline(0, color='grey', linestyle='--', alpha=0.5,
                       linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(kernel_order, rotation=30, ha='right')
            ax.set_xlabel('')
            ax.set_ylabel(f'{corr_display} Correlation with Speedup')
            ax.set_title(f'Block Density Improvement–Speedup Correlation '
                         f'across Block Sizes{scale_label}  '
                         f'(n_cols={int(n_cols)})')
            ax.legend(title='Block Size', fontsize=9, title_fontsize=10)
            ax.grid(True, axis='y', alpha=0.3)

            fname = (f'imp_blocksize_bars_{tag}{scale_suffix}'
                     f'_ncols_{int(n_cols)}.png')
            out_path = plot_dir / fname
            plt.tight_layout()
            plt.savefig(out_path, dpi=300)
            plt.close()
            print(f"  Saved: {out_path}")


def _compute_partial_imp_matrix(df, n_cols, imp_metrics, kernel, method=None):
    """Compute partial correlation matrix for one kernel.

    Returns a DataFrame of shape (n_metrics, n_metrics).
    Cell [i, j] = r(speedup, metric_i | metric_j).
    Diagonal = marginal r(speedup, metric_i).
    """
    if method is None:
        method = pu.get_correlation_method()

    df_k = df[(df['n_cols'] == n_cols)
              & (df['strategy'] != 'Original')
              & (df['kernel_id'] == kernel)]

    n = len(imp_metrics)
    mat = pd.DataFrame(np.full((n, n), np.nan),
                       index=imp_metrics, columns=imp_metrics)

    for i, m_target in enumerate(imp_metrics):
        if m_target not in df_k.columns:
            continue

        for j, m_control in enumerate(imp_metrics):
            if i == j or m_control not in df_k.columns:
                continue

            cols = [m_target, m_control, 'speedup']
            sub = df_k[cols].dropna()
            sub = sub[np.isfinite(sub).all(axis=1)]
            if len(sub) < 10:
                continue

            # Partial correlation via OLS residuals
            y = sub['speedup'].values
            x = sub[m_target].values
            z = sub[m_control].values
            z_aug = np.column_stack([z, np.ones(len(z))])
            try:
                coef_y = np.linalg.lstsq(z_aug, y, rcond=None)[0]
                coef_x = np.linalg.lstsq(z_aug, x, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            ry = y - z_aug @ coef_y
            rx = x - z_aug @ coef_x

            if np.std(ry) < 1e-12 or np.std(rx) < 1e-12:
                continue

            r, _ = pu.compute_correlation(pd.Series(rx), pd.Series(ry), method)
            mat.iloc[i, j] = r

    return mat


def generate_partial_correlation_heatmaps(df, out_dir):
    """Generate partial-correlation heatmaps: one per (kernel, n_cols).

    Each heatmap is a metric×metric matrix where cell [i,j] shows
    r(speedup, metric_i | metric_j).  Diagonal = marginal correlation.
    Reveals which metrics are redundant vs. independently predictive.
    """
    import seaborn as sns
    method = pu.get_correlation_method()
    corr_display = pu.correlation_display_name()
    tag = _corr_method_tag()

    imp_metrics = _enabled_improvement_metrics()
    kernels = _ordered_kernels(df, KERNEL_NAMES)
    n_cols_values = sorted(df['n_cols'].unique())

    def _clean_label(m):
        s = get_metric_display(m)
        for suffix in [' Improvement', ' Reduction']:
            s = s.replace(suffix, '')
        return s
    display_labels = [_clean_label(m) for m in imp_metrics]

    plot_dir = Path(out_dir) / 'partial_correlation'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for n_cols in n_cols_values:
        n_cols_int = int(n_cols)

        for kernel in kernels:
            mat = _compute_partial_imp_matrix(
                df, n_cols, imp_metrics, kernel, method=method)

            if mat.isna().all().all():
                continue

            # Relabel for display
            mat.index = display_labels
            mat.columns = display_labels

            kernel_display = KERNEL_NAMES.get(kernel, kernel)
            n = len(imp_metrics)
            cell_size = max(0.9, min(1.2, 8 / n))
            figsize = (n * cell_size + 2.5, n * cell_size + 1.5)
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                mat.abs().astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=0.7,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'|Partial {corr_display}|',
                          'shrink': 0.8},
                annot_kws={'size': 9},
            )

            ax.set_title(
                f'{kernel_display} — Partial Correlation with Speedup\n'
                f'|r(speedup, row | col)|   n_cols={n_cols_int}',
                fontsize=12, pad=12)
            ax.set_xlabel('Controlled metric')
            ax.set_ylabel('Target metric')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)

            kernel_safe = pu.safe_filename(kernel)
            fname = f'partial_corr_{tag}_{kernel_safe}_ncols_{n_cols_int}.png'
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(plot_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_dir / fname}")

        # Also generate a summary heatmap: mean |partial r| across kernels
        # to show which metrics are globally redundant
        all_mats = []
        for kernel in kernels:
            mat = _compute_partial_imp_matrix(
                df, n_cols, imp_metrics, kernel, method=method)
            if not mat.isna().all().all():
                all_mats.append(mat)

        if len(all_mats) >= 2:
            mean_mat = pd.concat(all_mats).groupby(level=0).mean()
            mean_mat = mean_mat.reindex(index=imp_metrics, columns=imp_metrics)
            mean_mat.index = display_labels
            mean_mat.columns = display_labels

            n = len(imp_metrics)
            cell_size = max(0.9, min(1.2, 8 / n))
            figsize = (n * cell_size + 2.5, n * cell_size + 1.5)
            fig, ax = plt.subplots(figsize=figsize)

            sns.heatmap(
                mean_mat.abs().astype(float), annot=True, fmt='.2f',
                cmap='YlOrRd', vmin=0, vmax=0.7,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': f'Mean |Partial {corr_display}|',
                          'shrink': 0.8},
                annot_kws={'size': 9},
            )

            ax.set_title(
                f'Mean Partial Correlation with Speedup (all kernels)\n'
                f'|r(speedup, row | col)|   n_cols={n_cols_int}',
                fontsize=12, pad=12)
            ax.set_xlabel('Controlled metric')
            ax.set_ylabel('Target metric')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(axis='y', rotation=0)

            fname = f'partial_corr_{tag}_mean_ncols_{n_cols_int}.png'
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig(plot_dir / fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {plot_dir / fname}")


def _should_run(section: str, args) -> bool:
    """Decide whether *section* should run given CLI flags.

    Priority: --sections (fine-grained) > --only-* (coarse) > default (all).
    """
    if args.sections is not None:
        return section in args.sections
    # Legacy coarse flags
    kernel_sections = {'kernels', 'breakeven', 'original-scatter', 'speedup-profiles'}
    reorder_sections = {'reorder-analysis', 'reorderability', 'improvability', 'per-matrix', 'timing', 'pairwise'}
    if args.only_kernels:
        return section in kernel_sections
    if args.only_reorder_analysis:
        return section in reorder_sections
    return True  # no filter → run everything


def main():
    args = parse_args()

    # Resolve perm_type pipeline
    perm_type_filter = 'ROW' if args.row else 'SYMMETRIC'
    pipeline_key = ('random_' if args.random else '') + perm_type_filter.lower()

    # Apply --random defaults (before any other processing)
    if args.out is None:
        data_label = 'random' if args.random else 'original'
        perm_label = 'row' if args.row else 'symmetric'
        args.out = f"plots/{data_label}_{perm_label}"

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
        'random': args.random,
    }
    # --include-rectangular means square_only=False
    if args.include_rectangular is not None:
        cli_overrides['square_only'] = not args.include_rectangular

    df, df_analysis, _cfg = pu.load_and_filter_data(
        config_path=args.filter_config,
        cli_overrides=cli_overrides,
    )

    # Split by perm_type — downstream code works on a single homogeneous DF
    pipeline_cfg = _cfg.get('pipelines', {}).get(pipeline_key, {})
    df, df_analysis = pu.split_by_perm_type(
        df, df_analysis, perm_type_filter, pipeline_cfg)

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

    if has_ops and _should_run('singular-improvement', args):
        print("\n" + "="*60)
        print("Generating singular kernel improvement plots...")
        print("="*60)
        generate_singular_improvement_plots(df, out_dir, args)

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

    if _should_run('improvability', args):
        print("\n" + "="*60)
        print("Generating improvability summary plots...")
        print("="*60)
        generate_improvability_summary_plots(df_analysis, out_dir)

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
                                      reordering_csv=_reordering_csv(), cfg=_cfg)

    if _should_run('profiles', args):
        print("\n" + "="*60)
        print("Generating performance profile plots...")
        print("="*60)
        generate_profile_plots(df_analysis, out_dir)

    if _should_run('aggregate-improvement', args):
        print("\n" + "="*60)
        print("Generating aggregate improvement boxplot...")
        print("="*60)
        generate_profile_aggregate_boxplot(df_analysis, out_dir)
        if has_ops:
            generate_speedup_aggregate_boxplot(df, out_dir, args)

    if has_ops and _should_run('speedup-profiles', args):
        print("\n" + "="*60)
        print("Generating speedup performance profile plots...")
        print("="*60)
        generate_speedup_profile_plots(df, out_dir, args)

    if _should_run('pairwise', args):
        print("\n" + "="*60)
        print("Generating pairwise win/loss heatmaps...")
        print("="*60)
        generate_pairwise_heatmap(df_analysis, out_dir)

    if has_ops and _should_run('imp-correlation', args):
        print("\n" + "="*60)
        print("Generating improvement-correlation bar charts...")
        print("="*60)
        generate_imp_correlation_plots(df, out_dir)
        generate_imp_ncols_correlation_plots(df, out_dir)
        generate_imp_blocksize_correlation_plots(df, out_dir)

    if has_ops and _should_run('partial-correlation', args):
        print("\n" + "="*60)
        print("Generating partial-correlation heatmaps...")
        print("="*60)
        generate_partial_correlation_heatmaps(df, out_dir)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
