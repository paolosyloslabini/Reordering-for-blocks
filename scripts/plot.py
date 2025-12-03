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
    
    # --- Plot 1: Algorithm Performance Overview (Boxplot) ---
    if 'algo' in df.columns and 'time_operation_ms' in df.columns:
        plt.figure(figsize=(12, 6))
        # Filter out extreme outliers for better visualization if needed, or use log scale
        sns.boxplot(data=df, x='algo', y='time_operation_ms', hue='perm_type')
        plt.title("Execution Time Distribution by Algorithm and Permutation Type")
        plt.yscale('log')
        plt.ylabel("Time (ms) - Log Scale")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_dir / "algo_performance_boxplot.png")
        plt.close()
        print("Generated algo_performance_boxplot.png")

    # --- Plot 2: Block Density vs Performance (Scatter) ---
    # Filter for BSR/SMAT algorithms where block density matters
    if 'block_density' in df.columns and 'time_operation_ms' in df.columns:
        block_algos = df[df['algo'].str.contains('BSR|SMAT', case=False, regex=True, na=False)]
        if not block_algos.empty:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=block_algos, 
                x='block_density', 
                y='time_operation_ms', 
                hue='algo', 
                style='perm_type',
                alpha=0.7
            )
            plt.title("Block Density vs Execution Time (BSR & SMAT)")
            plt.xlabel("Block Density (NNZ / Block Area)")
            plt.ylabel("Time (ms) - Log Scale")
            plt.yscale('log')
            plt.xscale('log') # Block density often spans orders of magnitude
            plt.tight_layout()
            plt.savefig(out_dir / "block_density_vs_time.png")
            plt.close()
            print("Generated block_density_vs_time.png")

    # --- Plot 3: Bandwidth vs Performance (Scatter) ---
    if 'bandwidth_avg' in df.columns and 'time_operation_ms' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='bandwidth_avg', 
            y='time_operation_ms', 
            hue='algo',
            alpha=0.6
        )
        plt.title("Average Bandwidth vs Execution Time")
        plt.xlabel("Average Bandwidth")
        plt.ylabel("Time (ms) - Log Scale")
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(out_dir / "bandwidth_vs_time.png")
        plt.close()
        print("Generated bandwidth_vs_time.png")

    # --- Plot 4: Locality vs Performance (Scatter) ---
    if 'locality_avg_row_spread' in df.columns and 'time_operation_ms' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='locality_avg_row_spread', 
            y='time_operation_ms', 
            hue='algo',
            alpha=0.6
        )
        plt.title("Average Row Spread (Locality) vs Execution Time")
        plt.xlabel("Average Row Spread")
        plt.ylabel("Time (ms) - Log Scale")
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(out_dir / "locality_vs_time.png")
        plt.close()
        print("Generated locality_vs_time.png")

    # --- Plot 5: Permutation Impact per Matrix (Bar plot for subset) ---
    if 'matrix' in df.columns and 'perm' in df.columns:
        # Pick top 3 matrices by NNZ to show detailed breakdown
        if 'nnz' in df.columns:
            top_matrices = df.sort_values('nnz', ascending=False)['matrix'].unique()[:3]
        else:
            top_matrices = df['matrix'].unique()[:3]
            
        subset = df[df['matrix'].isin(top_matrices)]
        
        if not subset.empty:
            g = sns.catplot(
                data=subset, 
                kind="bar",
                x="perm", 
                y="time_operation_ms", 
                hue="algo",
                col="matrix",
                col_wrap=1,
                height=4, 
                aspect=2,
                sharey=False # Allow different scales for different matrices
            )
            g.set_axis_labels("Permutation", "Time (ms)")
            g.set_titles("{col_name}")
            g.fig.suptitle("Impact of Permutation on Time (Top 3 Matrices)", y=1.02)
            plt.savefig(out_dir / "permutation_impact_subset.png", bbox_inches='tight')
            plt.close()
            print("Generated permutation_impact_subset.png")

    print(f"All plots saved to {out_dir}")

if __name__ == "__main__":
    main()