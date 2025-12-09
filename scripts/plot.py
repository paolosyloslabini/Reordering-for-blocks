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
        df = plot_utils.filter_one_per_family(df, args.matrices_list)

    # 3. Filter: Trivial matrices
    df = plot_utils.filter_trivial_matrices(df, df_analysis)

    # 4. Calculate Metrics (GFLOPS, op_id)
    df = plot_utils.calculate_metrics(df)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 5. Generate Plots
    print("Generating GFLOPS vs Density plots...")
    plot_utils.plot_gflops_vs_density(df, out_dir)
    
    print("Generating GFLOPS Distribution plots...")
    plot_utils.plot_gflops_distribution(df, out_dir)
    
    print("Generating Speedup Distribution plots...")
    plot_utils.plot_speedup_distribution(df, out_dir)

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()
