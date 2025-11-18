#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE SpMM using CuPy with configurable sparse format.
Supports CSR, BSR, and COO formats.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
from cusparse_utils import load_and_permute_matrix, convert_to_gpu, time_operation


def main():
    parser = argparse.ArgumentParser(description='Minimal cuSPARSE SpMM using CuPy with configurable format')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--format', type=str, default='csr', choices=['csr', 'bsr', 'coo'],
                        help='Sparse matrix format: csr, bsr, or coo (default: csr)')
    parser.add_argument('--perm_rows', type=str, default=None, help='Path to row permutation file)')
    parser.add_argument('--perm_cols', type=str, default=None, help='Path to cols permutation file)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha scalar (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta scalar (default: 0.0)')
    parser.add_argument('--n-cols', type=int, default=32, help='Number of columns in dense matrix B (default: 32)')
    parser.add_argument('--n-iterations', type=int, default=5, help='Number of timing iterations (default: 5)')
    parser.add_argument('--blocksize', type=int, default=8, help='Block size for BSR format (default: 8)')
    
    args = parser.parse_args()
    
    # Load and permute matrix
    A_cpu = load_and_permute_matrix(args.matrix_path, args.perm_rows, args.perm_cols)
    m, n = A_cpu.shape

    # Convert to requested format and transfer to GPU
    A_gpu = convert_to_gpu(A_cpu, args.format, args.blocksize)
    
    # Create dense matrix B (n x n_cols) and C (m x n_cols) on GPU
    B_gpu = cp.random.randn(n, args.n_cols, dtype=cp.float32)
    C_gpu = cp.random.randn(m, args.n_cols, dtype=cp.float32)
    
    # Time the operation
    def spmm_op():
        return args.alpha * A_gpu.dot(B_gpu) + args.beta * C_gpu
    
    avg_time_ms = time_operation(spmm_op, args.n_iterations)
    print(f"FORMAT:{args.format.upper()}")
    print(f"TIMING_MS:{avg_time_ms:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
