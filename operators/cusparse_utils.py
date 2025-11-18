#!/usr/bin/env python3
"""
Common utilities for cuSPARSE operators.
"""

import cupy as cp
import cupyx.scipy.sparse as cupyx_sp
from scipy.io import mmread


def load_and_permute_matrix(matrix_path, perm_rows_path=None, perm_cols_path=None, dtype=cp.float32):
    """
    Load a matrix from MatrixMarket format and apply optional permutations.
    
    Args:
        matrix_path: Path to Matrix Market file
        perm_rows_path: Optional path to row permutation file (1-indexed)
        perm_cols_path: Optional path to column permutation file (1-indexed)
        dtype: Data type for the matrix (default: cp.float32)
    
    Returns:
        Sparse matrix in CSR format with permutations applied
    """
    # Load sparse matrix from MatrixMarket
    A_cpu = mmread(matrix_path).tocsr().astype(dtype)
    
    # Apply permutation if provided
    if perm_rows_path:
        perm_rows = cp.loadtxt(perm_rows_path, dtype=cp.int64) - 1
        A_cpu = A_cpu[perm_rows.get(), :]
    if perm_cols_path:
        perm_cols = cp.loadtxt(perm_cols_path, dtype=cp.int64) - 1
        A_cpu = A_cpu[:, perm_cols.get()]
    
    return A_cpu


def convert_to_gpu(A_cpu, sparse_format='csr', blocksize=8):
    """
    Convert CPU sparse matrix to GPU in the specified format.
    
    Args:
        A_cpu: CPU sparse matrix (CSR format)
        sparse_format: Target format ('csr', 'bsr', 'coo')
        blocksize: Block size for BSR format
    
    Returns:
        GPU sparse matrix in requested format
    """
    if sparse_format == 'csr':
        return cupyx_sp.csr_matrix(A_cpu)
    elif sparse_format == 'bsr':
        A_bsr_cpu = A_cpu.tobsr(blocksize=(blocksize, blocksize))
        return cupyx_sp.bsr_matrix(A_bsr_cpu)
    elif sparse_format == 'coo':
        return cupyx_sp.coo_matrix(A_cpu.tocoo())
    else:
        raise ValueError(f"Unsupported format: {sparse_format}")


def time_operation(operation, n_iterations=5):
    """
    Time a GPU operation using CUDA events.
    
    Args:
        operation: Callable that performs the GPU operation
        n_iterations: Number of timing iterations
    
    Returns:
        Average time in milliseconds
    """
    # Warmup
    _ = operation()
    cp.cuda.Stream.null.synchronize()
    
    # Timed runs using CUDA events
    timings = []
    for _ in range(n_iterations):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()
        
        start_event.record()
        _ = operation()
        end_event.record()
        end_event.synchronize()
        
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        timings.append(elapsed_ms)
    
    return sum(timings) / len(timings)
