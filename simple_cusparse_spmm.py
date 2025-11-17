#!/usr/bin/env python3
"""
Minimal standalone cuSPARSE CSR SpMM using CuPy.
No fallbacks - raises error if GPU/cuSPARSE unavailable.
"""

import sys
import argparse
import cupy as cp
import cupyx.scipy.sparse as cupyx_sp
from scipy.io import mmread


def main():
    parser = argparse.ArgumentParser(description='Minimal cuSPARSE CSR SpMM using CuPy')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha scalar (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.0, help='Beta scalar (default: 0.0)')
    parser.add_argument('--n-cols', type=int, default=32, help='Number of columns in dense matrix B (default: 32)')
    parser.add_argument('--n-iterations', type=int, default=10, help='Number of timing iterations (default: 10)')
    
    args = parser.parse_args()
    
    matrix_path = args.matrix_path
    alpha = args.alpha
    beta = args.beta
    n_cols = args.n_cols
    n_iterations = args.n_iterations
    
    # Load sparse matrix from MatrixMarket
    A_cpu = mmread(matrix_path).tocsr().astype(cp.float32)
    m, n = A_cpu.shape
    
    # Transfer to GPU
    A_gpu = cupyx_sp.csr_matrix(A_cpu)
    
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
