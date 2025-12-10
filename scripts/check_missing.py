import pandas as pd
import argparse
import sys
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Check for missing experiments in results CSVs.")
    parser.add_argument("--operations", default="results/results_operations.csv", help="Path to results_operations.csv")
    parser.add_argument("--analysis", default="results/results_analysis.csv", help="Path to results_analysis.csv")
    parser.add_argument("--matrices-list", default="datasets/matrices_list_mtx.txt", help="Path to expected matrices list")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all missing items (not just summary)")
    parser.add_argument("--summary", "-s", action="store_true", help="Show only summary (no details)")
    args = parser.parse_args()

    # Load expected matrices if available
    expected_matrices = None
    if Path(args.matrices_list).exists():
        with open(args.matrices_list, 'r') as f:
            expected_matrices = set()
            for line in f:
                path = line.strip()
                if path:
                    # Extract matrix name from path
                    matrix_name = Path(path).name
                    expected_matrices.add(matrix_name)
        print(f"Loaded {len(expected_matrices)} expected matrices from {args.matrices_list}")

    # ========== 1. Check Operations ==========
    if not Path(args.operations).exists():
        print(f"Operations CSV not found: {args.operations}")
        return

    print(f"\n{'='*60}")
    print(f"OPERATIONS REPORT ({args.operations})")
    print(f"{'='*60}")
    
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
    if 'algo' in df_op.columns:
        df_op['algo'] = df_op['algo'].astype(str)
    
    # Basic stats
    all_matrices = set(df_op['matrix'].unique())
    print(f"\nMatrices in results: {len(all_matrices)}")
    
    # Check against expected matrices
    if expected_matrices:
        missing_from_results = expected_matrices - all_matrices
        extra_in_results = all_matrices - expected_matrices
        
        if missing_from_results:
            print(f"\n[WARNING] {len(missing_from_results)} matrices from list not in results:")
            if not args.summary:
                for m in sorted(missing_from_results)[:20]:
                    print(f"  - {m}")
                if len(missing_from_results) > 20:
                    print(f"  ... and {len(missing_from_results) - 20} more")
        
        if extra_in_results:
            print(f"\n[INFO] {len(extra_in_results)} matrices in results not in list (may be OK)")

    # Identify all unique algorithms/kernels
    if 'algo' in df_op.columns:
        unique_algos = df_op['algo'].unique()
        print(f"\nAlgorithms found: {', '.join(sorted(unique_algos))}")
    
    # Identify all unique permutations
    unique_perms = df_op['perm'].unique()
    print(f"Permutations found: {', '.join(sorted(unique_perms))}")
    
    # Identify all unique perm_types
    unique_perm_types = df_op['perm_type'].unique()
    print(f"Perm types found: {', '.join(sorted(unique_perm_types))}")
    
    # Identify n_cols if present
    if 'n_cols' in df_op.columns:
        unique_ncols = sorted(df_op['n_cols'].dropna().unique())
        print(f"n_cols values: {unique_ncols}")
    
    # ---- Detailed Missing Analysis ----
    print(f"\n{'-'*40}")
    print("MISSING EXPERIMENTS ANALYSIS")
    print(f"{'-'*40}")
    
    # Group by algo (or kernel) and check coverage
    config_cols = ['algo', 'perm_type']
    if 'n_cols' in df_op.columns:
        config_cols.append('n_cols')
    existing_config_cols = [c for c in config_cols if c in df_op.columns]
    
    # For each configuration, what matrices are missing?
    missing_by_config = defaultdict(list)
    
    unique_configs = df_op[existing_config_cols].drop_duplicates()
    
    for _, config in unique_configs.iterrows():
        # Filter data for this config
        mask = pd.Series(True, index=df_op.index)
        for col in existing_config_cols:
            if pd.isna(config[col]):
                mask &= df_op[col].isna()
            else:
                mask &= (df_op[col] == config[col])
        
        present_matrices = set(df_op[mask]['matrix'].unique())
        missing_matrices = all_matrices - present_matrices
        
        if missing_matrices:
            config_key = tuple(config[c] for c in existing_config_cols)
            missing_by_config[config_key] = sorted(missing_matrices)
    
    if not missing_by_config:
        print("\n[OK] All configurations have data for all matrices.")
    else:
        print(f"\n[MISSING] Found gaps in {len(missing_by_config)} configurations:")
        total_missing = 0
        for config_key, matrices in sorted(missing_by_config.items()):
            config_str = ", ".join([f"{c}={v}" for c, v in zip(existing_config_cols, config_key)])
            total_missing += len(matrices)
            print(f"\n  Config: {config_str}")
            print(f"  Missing {len(matrices)} matrices")
            if args.verbose:
                for m in matrices:
                    print(f"    - {m}")
            elif not args.summary:
                for m in matrices[:5]:
                    print(f"    - {m}")
                if len(matrices) > 5:
                    print(f"    ... and {len(matrices) - 5} more")
        
        print(f"\n  Total missing operation experiments: {total_missing}")

    # ---- Check for missing reorderings per matrix ----
    print(f"\n{'-'*40}")
    print("MISSING REORDERINGS PER MATRIX")
    print(f"{'-'*40}")
    
    # Get expected (perm, perm_type) combinations from data
    expected_reorderings = set(zip(df_op['perm'], df_op['perm_type']))
    
    missing_reorder_by_matrix = defaultdict(list)
    
    for matrix in all_matrices:
        matrix_data = df_op[df_op['matrix'] == matrix]
        present_reorderings = set(zip(matrix_data['perm'], matrix_data['perm_type']))
        missing = expected_reorderings - present_reorderings
        if missing:
            missing_reorder_by_matrix[matrix] = sorted(missing)
    
    if not missing_reorder_by_matrix:
        print("\n[OK] All matrices have all reorderings.")
    else:
        print(f"\n[MISSING] {len(missing_reorder_by_matrix)} matrices have missing reorderings:")
        
        # Group by what's missing for a cleaner report
        missing_pattern = defaultdict(list)
        for matrix, missing in missing_reorder_by_matrix.items():
            missing_pattern[tuple(missing)].append(matrix)
        
        for pattern, matrices in sorted(missing_pattern.items(), key=lambda x: -len(x[1])):
            print(f"\n  Missing reorderings: {list(pattern)}")
            print(f"  Affects {len(matrices)} matrices:")
            if args.verbose:
                for m in sorted(matrices):
                    print(f"    - {m}")
            elif not args.summary:
                for m in sorted(matrices)[:5]:
                    print(f"    - {m}")
                if len(matrices) > 5:
                    print(f"    ... and {len(matrices) - 5} more")

    # ========== 2. Check Analysis ==========
    if not Path(args.analysis).exists():
        print(f"\nAnalysis CSV not found: {args.analysis}")
        return

    print(f"\n{'='*60}")
    print(f"ANALYSIS REPORT ({args.analysis})")
    print(f"{'='*60}")
    
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

    analysis_matrices = set(df_analysis['matrix'].unique())
    print(f"\nMatrices in analysis: {len(analysis_matrices)}")
    
    # Unique perms in analysis
    analysis_perms = df_analysis['perm'].unique()
    print(f"Permutations in analysis: {', '.join(sorted(analysis_perms))}")
    
    analysis_perm_types = df_analysis['perm_type'].unique()
    print(f"Perm types in analysis: {', '.join(sorted(analysis_perm_types))}")

    # Check if all operation inputs have analysis
    print(f"\n{'-'*40}")
    print("MISSING ANALYSIS FOR OPERATIONS")
    print(f"{'-'*40}")
    
    op_keys = set(zip(df_op['matrix'], df_op['perm'], df_op['perm_type']))
    analysis_keys = set(zip(df_analysis['matrix'], df_analysis['perm'], df_analysis['perm_type']))
    
    missing_analysis = op_keys - analysis_keys
    
    if not missing_analysis:
        print("\n[OK] All operation inputs have analysis data.")
    else:
        print(f"\n[MISSING] {len(missing_analysis)} operation inputs lack analysis data:")
        
        # Group by (perm, perm_type) for cleaner report
        missing_by_reorder = defaultdict(list)
        for matrix, perm, perm_type in missing_analysis:
            missing_by_reorder[(perm, perm_type)].append(matrix)
        
        for (perm, perm_type), matrices in sorted(missing_by_reorder.items()):
            print(f"\n  Reordering: perm={perm}, type={perm_type}")
            print(f"  Missing analysis for {len(matrices)} matrices:")
            if args.verbose:
                for m in sorted(matrices):
                    print(f"    - {m}")
            elif not args.summary:
                for m in sorted(matrices)[:5]:
                    print(f"    - {m}")
                if len(matrices) > 5:
                    print(f"    ... and {len(matrices) - 5} more")

    # Check for expected reorderings in analysis
    print(f"\n{'-'*40}")
    print("MISSING REORDERINGS IN ANALYSIS")
    print(f"{'-'*40}")
    
    expected_analysis_reorderings = set(zip(df_analysis['perm'], df_analysis['perm_type']))
    
    missing_analysis_reorder_by_matrix = defaultdict(list)
    
    for matrix in analysis_matrices:
        matrix_data = df_analysis[df_analysis['matrix'] == matrix]
        present_reorderings = set(zip(matrix_data['perm'], matrix_data['perm_type']))
        missing = expected_analysis_reorderings - present_reorderings
        if missing:
            missing_analysis_reorder_by_matrix[matrix] = sorted(missing)
    
    if not missing_analysis_reorder_by_matrix:
        print("\n[OK] All matrices in analysis have all reorderings.")
    else:
        print(f"\n[MISSING] {len(missing_analysis_reorder_by_matrix)} matrices have missing analysis reorderings:")
        
        # Group by pattern
        missing_pattern = defaultdict(list)
        for matrix, missing in missing_analysis_reorder_by_matrix.items():
            missing_pattern[tuple(missing)].append(matrix)
        
        for pattern, matrices in sorted(missing_pattern.items(), key=lambda x: -len(x[1])):
            print(f"\n  Missing: {list(pattern)}")
            print(f"  Affects {len(matrices)} matrices:")
            if args.verbose:
                for m in sorted(matrices):
                    print(f"    - {m}")
            elif not args.summary:
                for m in sorted(matrices)[:5]:
                    print(f"    - {m}")
                if len(matrices) > 5:
                    print(f"    ... and {len(matrices) - 5} more")

    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Operations: {len(df_op)} rows, {len(all_matrices)} matrices")
    print(f"Analysis: {len(df_analysis)} rows, {len(analysis_matrices)} matrices")
    
    issues = []
    if missing_by_config:
        issues.append(f"{len(missing_by_config)} configs with missing matrices")
    if missing_reorder_by_matrix:
        issues.append(f"{len(missing_reorder_by_matrix)} matrices with missing reorderings")
    if missing_analysis:
        issues.append(f"{len(missing_analysis)} operations missing analysis")
    if missing_analysis_reorder_by_matrix:
        issues.append(f"{len(missing_analysis_reorder_by_matrix)} analysis matrices with missing reorderings")
    
    if issues:
        print(f"\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] No issues found!")

if __name__ == "__main__":
    main()
