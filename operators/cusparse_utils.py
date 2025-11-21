#!/usr/bin/env python3
"""
Common utilities for cuSPARSE operators.
"""

import cupy as cp
import cupyx.scipy.sparse as cupyx_sp
from scipy.io import mmread
import numpy as np


def load_permutation_file(perm_path, perm_type='ROW'):
    """
    Load permutation file and return row and/or column permutations.
    
    Args:
        perm_path: Path to permutation file
        perm_type: Type of permutation:
            'ROW' - single line with row permutation (only permute rows)
            'SYMMETRIC' - single line, apply to both rows and cols (requires square matrix)
            'ASYMMETRIC' - two lines: first=rows, second=cols
    
    Returns:
        Tuple of (row_perm, col_perm) as numpy arrays (0-indexed), or (perm, None) for ROW
    """
    perm_type = perm_type.upper()
    
    with open(perm_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if perm_type == 'ROW':
        # Single permutation - only apply to rows
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1  # Convert to 0-based
        return perm, None
    elif perm_type == 'SYMMETRIC':
        # Single permutation - apply to both rows and columns (requires square matrix)
        perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        return perm, perm.copy()  # Return copy to avoid aliasing
    elif perm_type == 'ASYMMETRIC':
        # Two permutations - read both lines
        if len(lines) < 2:
            raise ValueError(f"ASYMMETRIC permutation requires 2 lines, found {len(lines)}")
        row_perm = np.fromstring(lines[0], sep=' ', dtype=np.int64) - 1
        col_perm = np.fromstring(lines[1], sep=' ', dtype=np.int64) - 1
        return row_perm, col_perm
    else:
        raise ValueError(f"Unknown permutation type: {perm_type}. Use ROW, SYMMETRIC, or ASYMMETRIC")


def load_and_permute_matrix(matrix_path, perm_path=None, perm_type='ROW', dtype=np.float32):
    """
    Load a matrix from MatrixMarket format and apply optional permutations.
    
    Args:
        matrix_path: Path to Matrix Market file
        perm_path: Optional path to permutation file (1-indexed)
        perm_type: Type of permutation ('ROW', 'SYMMETRIC', or 'ASYMMETRIC')
        dtype: Data type for the matrix (default: np.float32)
    
    Returns:
        Sparse matrix in CSR format with permutations applied
    """
    # Load sparse matrix from MatrixMarket
    A_cpu = mmread(matrix_path).tocsr().astype(dtype)
    m, n = A_cpu.shape
    
    # Apply permutation if provided
    if perm_path:
        row_perm, col_perm = load_permutation_file(perm_path, perm_type)
        
        # Validate permutation sizes
        if row_perm is not None:
            if len(row_perm) != m:
                raise ValueError(f"Row permutation size ({len(row_perm)}) doesn't match matrix rows ({m})")
            A_cpu = A_cpu[row_perm, :]
        
        if col_perm is not None:
            if len(col_perm) != n:
                raise ValueError(f"Column permutation size ({len(col_perm)}) doesn't match matrix columns ({n})")
            A_cpu = A_cpu[:, col_perm]
        
        # Additional check for SYMMETRIC
        if perm_type.upper() == 'SYMMETRIC' and m != n:
            raise ValueError(f"SYMMETRIC permutation requires square matrix, got {m}x{n}")
    
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
        if blocksize is None:
            raise ValueError("blocksize must be specified for BSR format")
        blocksize = int(blocksize)  # Ensure it's an int
        
        # Pad matrix if dimensions are not divisible by blocksize
        m, n = A_cpu.shape
        m_pad = -(-m // blocksize) * blocksize  # Ceiling division
        n_pad = -(-n // blocksize) * blocksize
        
        if m_pad != m or n_pad != n:
            # Resize matrix to padded dimensions (fills with zeros)
            A_cpu = A_cpu.resize((m_pad, n_pad))
        
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


def print_timer(label, time_ms):
    """
    Print timing information in C++-compatible format.
    
    Args:
        label: Timer label (e.g., 'loading', 'execution')
        time_ms: Time in milliseconds
    """
    print(f"<Timer>[{label}] {time_ms:.6f} ms", flush=True)
