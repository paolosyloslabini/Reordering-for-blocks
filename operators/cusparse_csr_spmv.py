#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE CSR SpMV using CuPy.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
from cusparse_utils import load_and_permute_matrix, convert_to_gpu, time_operation


def main():
    parser = argparse.ArgumentParser(description='Minimal cuSPARSE CSR SpMV using CuPy')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm_rows', type=str, default=None, help='Path to row permutation file)')
    parser.add_argument('--perm_cols', type=str, default=None, help='Path to cols permutation file)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha scalar (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta scalar (default: 0.0)')
    parser.add_argument('--n-iterations', type=int, default=5, help='Number of timing iterations (default: 5)')
    
    args = parser.parse_args()
    
    # Load and permute matrix
    A_cpu = load_and_permute_matrix(args.matrix_path, args.perm_rows, args.perm_cols)
    m, n = A_cpu.shape

    # Convert to CSR and transfer to GPU
    A_gpu = convert_to_gpu(A_cpu, 'csr')
    
    # Create dense vector x (n) and y (m) on GPU
    x_gpu = cp.random.randn(n, dtype=cp.float32)
    y_gpu = cp.random.randn(m, dtype=cp.float32)
    
    # Time the operation
    def spmv_op():
        return args.alpha * A_gpu.dot(x_gpu) + args.beta * y_gpu
    
    avg_time_ms = time_operation(spmv_op, args.n_iterations)
    print(f"TIMING_MS:{avg_time_ms:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
