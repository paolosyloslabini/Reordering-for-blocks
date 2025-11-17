#!/usr/bin/env python3
"""Test the complete matrix reordering pipeline.

This script tests:
1. Random permutation generation
2. Matrix reordering (1D and 2D)
3. cuSPARSE SpMM on original and reordered matrices
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, skip_on_error=False):
    """Run a command and print its output."""
    print(f"{description}...", flush=True)
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        print(f"✓ {description} completed\n", flush=True)
        return True
    except subprocess.CalledProcessError as e:
        if skip_on_error:
            print(f"⚠ {description} failed (skipping)\n", flush=True)
            if e.stderr:
                print(f"Error: {e.stderr}\n", flush=True)
            return False
        else:
            print(f"✗ {description} failed\n", flush=True)
            if e.stderr:
                print(f"Error: {e.stderr}\n", flush=True)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test the complete matrix reordering pipeline"
    )
    parser.add_argument("matrix", type=Path, help="Path to input .mtx matrix file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation")
    parser.add_argument("--iterations", type=int, default=10, help="Number of SpMM iterations")
    
    args = parser.parse_args()
    
    if not args.matrix.exists():
        print(f"Error: Matrix file '{args.matrix}' not found")
        sys.exit(1)
    
    print("=" * 50)
    print("Matrix Reordering Pipeline Test")
    print("=" * 50)
    print(f"Matrix: {args.matrix}")
    print(f"Random seed: {args.seed}")
    print(f"SpMM iterations: {args.iterations}")
    print()
    
    # Step 1: Generate random permutation
    run_command(
        f"python3 random_permutation_graphblas.py {args.matrix} --seed {args.seed} --output test/permutation.txt",
        "[1/5] Generating random permutation"
    )
    
    # Step 2: Reorder matrix (1D)
    run_command(
        f"python3 reorder_matrix_graphblas.py {args.matrix} permutation.txt 1D test/reordered_1d.mtx",
        "[2/5] Reordering matrix (1D - rows only)"
    )
    
    # Step 3: Reorder matrix (2D)
    run_command(
        f"python3 reorder_matrix_graphblas.py {args.matrix} permutation.txt 2D test/reordered_2d.mtx",
        "[3/5] Reordering matrix (2D - rows and columns)"
    )
    
    # Step 4: Run SpMM on original matrix
    gpu_available = run_command(
        f"python3 simple_cusparse_spmm.py {args.matrix} --n-iterations {args.iterations}",
        "[4/5] Running cuSPARSE SpMM on original matrix",
        skip_on_error=True
    )
    
    # Step 5: Run SpMM on reordered matrices
    if gpu_available:
        print("[5/5] Running cuSPARSE SpMM on reordered matrices...")
        
        print("  Testing 1D reordered matrix:")
        run_command(
            f"python3 simple_cusparse_spmm.py reordered_1d.mtx --n-iterations {args.iterations}",
            "  1D reordered SpMM",
            skip_on_error=True
        )
        
        print("  Testing 2D reordered matrix:")
        run_command(
            f"python3 simple_cusparse_spmm.py reordered_2d.mtx --n-iterations {args.iterations}",
            "  2D reordered SpMM",
            skip_on_error=True
        )
    else:
        print("[5/5] Skipping cuSPARSE SpMM tests (GPU not available or CUDA issue)\n")
    
    print("=" * 50)
    print("Pipeline Test Complete!")
    print("=" * 50)
    print("Generated files:")
    print("  - test/permutation.txt (1-based permutation vector)")
    print("  - test/reordered_1d.mtx (row-reordered matrix)")
    print("  - test/reordered_2d.mtx (row+column-reordered matrix)")
    print()


if __name__ == "__main__":
    main()
