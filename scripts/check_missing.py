import pandas as pd
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Check for missing experiments in results CSVs.")
    parser.add_argument("--operations", default="results/results_operations.csv", help="Path to results_operations.csv")
    parser.add_argument("--analysis", default="results/results_analysis.csv", help="Path to results_analysis.csv")
    args = parser.parse_args()

    # 1. Check Operations
    if not Path(args.operations).exists():
        print(f"Operations CSV not found: {args.operations}")
        return

    print(f"--- Checking Operations ({args.operations}) ---")
    try:
        df_op = pd.read_csv(args.operations)
    except Exception as e:
        print(f"Error reading operations CSV: {e}")
        return
    
    if df_op.empty:
        print("Operations CSV is empty.")
        return

    # Normalize columns
    df_op['matrix'] = df_op['matrix'].astype(str)
    df_op['perm'] = df_op['perm'].fillna('None').astype(str)
    df_op['perm_type'] = df_op['perm_type'].fillna('UNKNOWN').astype(str)
    
    # Identify all matrices
    all_matrices = set(df_op['matrix'].unique())
    print(f"Found {len(all_matrices)} unique matrices.")

    # Identify all unique configurations (algo, perm_type, block_size, n_cols)
    # We assume that if a config exists for ONE matrix, it should exist for ALL.
    # We group by these columns to find unique configs.
    config_cols = ['algo', 'perm_type', 'block_size', 'n_cols']
    # Handle missing columns if old csv
    existing_config_cols = [c for c in config_cols if c in df_op.columns]
    
    unique_configs = df_op[existing_config_cols].drop_duplicates()
    
    print(f"Found {len(unique_configs)} unique configurations (algo/perm_type/block_size/n_cols).")
    
    missing_count = 0
    
    for _, config in unique_configs.iterrows():
        # Filter data for this config
        mask = pd.Series(True, index=df_op.index)
        for col in existing_config_cols:
            # Handle NaN for block_size if it exists
            if pd.isna(config[col]):
                mask &= df_op[col].isna()
            else:
                mask &= (df_op[col] == config[col])
        
        present_matrices = set(df_op[mask]['matrix'].unique())
        missing_matrices = all_matrices - present_matrices
        
        if missing_matrices:
            config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
            print(f"\n[MISSING] Config: {config_str}")
            print(f"  Missing for {len(missing_matrices)} matrices:")
            # Print first few
            sorted_missing = sorted(list(missing_matrices))
            print(f"  {', '.join(sorted_missing[:5])}" + (f" ... and {len(missing_matrices)-5} more" if len(missing_matrices) > 5 else ""))
            missing_count += len(missing_matrices)

    if missing_count == 0:
        print("\n[OK] No missing operation experiments detected (based on cross-product of matrices and configs).")
    else:
        print(f"\n[FAIL] Total missing operation experiments: {missing_count}")

    # 2. Check Analysis
    if not Path(args.analysis).exists():
        print(f"\nAnalysis CSV not found: {args.analysis}")
        return

    print(f"\n--- Checking Analysis ({args.analysis}) ---")
    try:
        df_analysis = pd.read_csv(args.analysis)
    except Exception as e:
        print(f"Error reading analysis CSV: {e}")
        return
    
    if df_analysis.empty:
        print("Analysis CSV is empty.")
        return

    # Normalize
    df_analysis['matrix'] = df_analysis['matrix'].astype(str)
    df_analysis['perm'] = df_analysis['perm'].fillna('None').astype(str)
    df_analysis['perm_type'] = df_analysis['perm_type'].fillna('UNKNOWN').astype(str)

    # Check if all (matrix, perm, perm_type) used in operations exist in analysis
    # We only care about the combinations used in operations
    op_keys = df_op[['matrix', 'perm', 'perm_type']].drop_duplicates()
    
    # Create a set of keys for fast lookup
    # Use tuples
    analysis_keys = set(zip(df_analysis['matrix'], df_analysis['perm'], df_analysis['perm_type']))
    
    missing_analysis_count = 0
    print("Checking if all operation inputs have corresponding analysis data...")
    
    for _, row in op_keys.iterrows():
        key = (row['matrix'], row['perm'], row['perm_type'])
        if key not in analysis_keys:
            print(f"[MISSING ANALYSIS] Matrix: {key[0]}, Perm: {key[1]}, Type: {key[2]}")
            missing_analysis_count += 1
            
    if missing_analysis_count == 0:
        print("[OK] All operation inputs have analysis data.")
    else:
        print(f"[FAIL] Total missing analysis entries: {missing_analysis_count}")

if __name__ == "__main__":
    main()
