#!/usr/bin/env python3
"""Generate a random permutation vector for a Matrix Market sparse matrix.

Loads a Matrix Market file using GraphBLAS and returns a random permutation
vector (1-based indices).

Requirements
------------
    pip install graphblas numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from graphblas import Matrix, io


def random_permutation(matrix_path: Path) -> np.ndarray:
    """Generate a random permutation vector for the given Matrix Market matrix."""
    
    # Load matrix from Matrix Market file
    A = io.mmread(str(matrix_path))
    
    # Get the number of rows in the matrix
    n_rows = A.nrows
    
    # Generate a random permutation of indices [0, 1, 2, ..., n_rows-1]
    permutation = np.random.permutation(n_rows)
    
    return permutation + 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a random permutation for a Matrix Market matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("matrix", type=Path, help="Path to input .mtx matrix")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file for permutation (1-based)")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Generate random permutation
    perm = random_permutation(args.matrix)
        
    print("PERMUTATION: ", perm)
    if args.output:
        np.savetxt(args.output, perm, fmt='%d')
        print(f"\nPermutation saved to {args.output} (1-based indices)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
