#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE SpMM using CuPy with configurable sparse format.
Supports CSR, BSR, and COO formats.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
import cupyx.scipy.sparse as cupyx_sp
from scipy.io import mmread


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
    
    matrix_path = args.matrix_path
    sparse_format = args.format
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

    # Convert to requested format
    if sparse_format == 'csr':
        A_gpu = cupyx_sp.csr_matrix(A_cpu)
    elif sparse_format == 'bsr':
        A_gpu = cupyx_sp.bsr_matrix(A_cpu.tobsr(blocksize=(blocksize, blocksize)))
    elif sparse_format == 'coo':
        A_gpu = cupyx_sp.coo_matrix(A_cpu.tocoo())
    else:
        raise ValueError(f"Unsupported format: {sparse_format}")
    
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
    print(f"FORMAT:{sparse_format.upper()}")
    print(f"TIMING_MS:{avg_time_ms:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
