#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE BSR SpMV using CuPy.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
import time
from cusparse_utils import load_and_permute_matrix, convert_to_gpu, time_operation, print_timer


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE BSR SpMV using CuPy')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default='ROW', help='Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: ROW)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha scalar (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta scalar (default: 0.0)')
    parser.add_argument('--n-iterations', type=int, default=5, help='Number of timing iterations (default: 5)')
    parser.add_argument('--blocksize', type=int, default=8, help='Block size for BSR format (default: 8)')
    
    args = parser.parse_args()
    
    # Load and permute matrix
    t0 = time.perf_counter()
    A_cpu = load_and_permute_matrix(args.matrix_path, args.perm, args.perm_type)
    loading_ms = (time.perf_counter() - t0) * 1000
    m, n = A_cpu.shape

    # Convert to BSR and transfer to GPU
    t0 = time.perf_counter()
    A_gpu = convert_to_gpu(A_cpu, 'bsr', args.blocksize)
    transfer_ms = (time.perf_counter() - t0) * 1000
    
    # Create dense vector x (n) and y (m) on GPU
    x_gpu = cp.random.randn(n, dtype=cp.float32)
    y_gpu = cp.random.randn(m, dtype=cp.float32)
    
    # Time the operation
    def spmv_op():
        return args.alpha * A_gpu.dot(x_gpu) + args.beta * y_gpu
    
    avg_time_ms = time_operation(spmv_op, args.n_iterations)
    
    print_timer("loading", loading_ms)
    print_timer("transfer", transfer_ms)
    print_timer("operation", avg_time_ms)
    return 0


if __name__ == '__main__':
    sys.exit(main())
