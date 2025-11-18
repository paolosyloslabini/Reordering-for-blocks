#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE BSR SpMM using CuPy.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
import cupyx.scipy.sparse as cupyx_sp
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser(description='cuSPARSE BSR SpMM using CuPy')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm_rows', type=str, default=None, help='Path to row permutation file)')
    parser.add_argument('--perm_cols', type=str, default=None, help='Path to cols permutation file)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha scalar (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta scalar (default: 0.0)')
    parser.add_argument('--n-cols', type=int, default=32, help='Number of columns in dense matrix B (default: 32)')
    parser.add_argument('--n-iterations', type=int, default=5, help='Number of timing iterations (default: 5)')
    parser.add_argument('--blocksize', type=int, default=8, help='Block size for BSR format (default: 8)')
    
    args = parser.parse_args()
    
    matrix_path = args.matrix_path
    alpha = args.alpha
    beta = args.beta
    n_cols = args.n_cols
    n_iterations = args.n_iterations
    blocksize = args.blocksize
    perm_rows_path = args.perm_rows
    perm_cols_path = args.perm_cols

    # Load sparse matrix from MatrixMarket
    A_cpu = mmread(matrix_path).tocsr().astype(cp.float32)
    m, n = A_cpu.shape

    # Apply permutation if provided
    if perm_rows_path:
        perm_rows = cp.loadtxt(perm_rows_path, dtype=cp.int64) - 1
        A_cpu = A_cpu[perm_rows.get(), :]
    if perm_cols_path:
        perm_cols = cp.loadtxt(perm_cols_path, dtype=cp.int64) - 1
        A_cpu = A_cpu[:, perm_cols.get()]

    # Convert to BSR format
    A_bsr_cpu = A_cpu.tobsr(blocksize=(blocksize, blocksize))
    
    # Transfer to GPU
    A_gpu = cupyx_sp.bsr_matrix(A_bsr_cpu)
    
    # Create dense matrix B (n x n_cols) and C (m x n_cols) on GPU
    B_gpu = cp.random.randn(n, n_cols, dtype=cp.float32)
    C_gpu = cp.random.randn(m, n_cols, dtype=cp.float32)
    
    # Warmup
    _ = alpha * A_gpu.dot(B_gpu) + beta * C_gpu
    cp.cuda.Stream.null.synchronize()
    
    # Timed runs using CUDA events
    timings = []
    for _ in range(n_iterations):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        result = alpha * A_gpu.dot(B_gpu) + beta * C_gpu
        end_event.record()
        end_event.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        timings.append(elapsed_ms)
    
    avg_time_ms = sum(timings) / len(timings)
    print(f"TIMING_MS:{avg_time_ms:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
