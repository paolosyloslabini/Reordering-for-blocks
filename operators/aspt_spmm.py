#!/usr/bin/env python3
import sys
import os
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
import scipy.io
import scipy.sparse
import time

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cusparse_utils import load_and_permute_matrix, print_timer, check_below_peak
from config import SPMM_N_COLS_DEFAULT, PERM_TYPE_DEFAULT, PEAK_GFLOPS_FP32

def main():
    parser = argparse.ArgumentParser(description='ASpT SpMM Benchmark')
    parser.add_argument('matrix_path', help='Path to Matrix Market file')
    parser.add_argument('--perm', type=str, default=None, help='Path to permutation file')
    parser.add_argument('--perm-type', type=str, default=PERM_TYPE_DEFAULT, help=f'Type of permutation: ROW, SYMMETRIC, or ASYMMETRIC (default: {PERM_TYPE_DEFAULT})')
    parser.add_argument('--base-perm', type=str, default=None, help='Path to base permutation file (applied first)')
    parser.add_argument('--base-perm-type', type=str, default='SYMMETRIC', help='Type of base permutation (default: SYMMETRIC)')
    parser.add_argument('--n-cols', type=int, default=SPMM_N_COLS_DEFAULT, help=f'Number of columns in dense matrix B (default: {SPMM_N_COLS_DEFAULT})')
    parser.add_argument('--n-iterations', type=int, default=100, help='Number of iterations (ignored: ASpT repeats ITER times, fixed at build time by install_aspt.sh)')
    parser.add_argument('--max-error-pct', type=float, default=0.1, help='Fail if more than this %% of output entries are wrong (default: 0.1)')
    
    args = parser.parse_args()

    # 1. Load and permute matrix
    t0 = time.perf_counter()
    try:
        A = load_and_permute_matrix(args.matrix_path, args.perm, args.perm_type,
                                    base_perm_path=args.base_perm, base_perm_type=args.base_perm_type)
    except Exception as e:
        print(f"Error loading matrix: {e}", file=sys.stderr)
        sys.exit(1)
    loading_ms = (time.perf_counter() - t0) * 1000

    # 2. Prepare environment
    script_dir = Path(__file__).parent.absolute()
    
    # 3. Write matrix to temp file
    # ASpT expects a Matrix Market file
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp_mtx:
        scipy.io.mmwrite(tmp_mtx, A)
        tmp_mtx_path = tmp_mtx.name
    transfer_ms = (time.perf_counter() - t0) * 1000

    # 4. Run ASpT
    # sspmm_32 computes 32 output columns; sspmm_128 tiles B in 64-column
    # slices (grid z = n_cols/64), so it handles any multiple of 64.
    if args.n_cols == 32:
        binary_name = "sspmm_32"
    elif args.n_cols % 64 == 0:
        binary_name = "sspmm_128"
    else:
        print(f"Error: ASpT supports n_cols = 32 or a multiple of 64, got {args.n_cols}", file=sys.stderr)
        os.remove(tmp_mtx_path)
        sys.exit(1)

    binary = script_dir / "ASpT" / "ASpT_SpMM_GPU" / binary_name
    if not binary.exists():
        print(f"Error: ASpT binary not found at {binary}. Please run install_aspt.sh", file=sys.stderr)
        os.remove(tmp_mtx_path)
        sys.exit(1)

    # Load all CUDA modules up front so lazy loading is not timed in the first launch.
    env = os.environ.copy()
    env["CUDA_MODULE_LOADING"] = "EAGER"

    # ASpT arguments: <matrix_file> <n_cols>; it appends "GFLOPS,error%," to
    # SpMM_GPU_SP.out in its CWD, so run it in a fresh temp dir.
    cmd = [str(binary), tmp_mtx_path, str(args.n_cols)]
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                subprocess.run(cmd, cwd=tmp_dir, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                print(f"Error running ASpT: {e}", file=sys.stderr)
                print(f"Stderr: {e.stderr.decode()}", file=sys.stderr)
                sys.exit(1)

            out_file = Path(tmp_dir) / "SpMM_GPU_SP.out"
            if not out_file.exists():
                print("Error: Output file SpMM_GPU_SP.out not generated", file=sys.stderr)
                sys.exit(1)
            content = out_file.read_text().strip()
    finally:
        if os.path.exists(tmp_mtx_path):
            os.remove(tmp_mtx_path)

    # Format: GFLOPS,ErrorRate,  (error rate = % of output entries off by > 1 %
    # from the CPU reference; GFLOPS is already averaged over the ITER timed launches)
    parts = [p for p in content.split(',') if p.strip()]
    try:
        gflops = float(parts[0])
        error_pct = float(parts[1])
    except (IndexError, ValueError):
        print(f"Error: Could not parse GFLOPS and error rate from output: {content}", file=sys.stderr)
        sys.exit(1)

    print(f"ASpT binary: {binary_name}, GFLOPS: {gflops:.3f}, error rate: {error_pct:.4f} %")
    if error_pct > args.max_error_pct:
        print(f"Error: ASpT result is wrong for {error_pct:.4f} % of the output "
              f"(limit {args.max_error_pct} %)", file=sys.stderr)
        sys.exit(1)
    if gflops <= 0:
        print(f"Error: non-positive GFLOPS from ASpT: {content}", file=sys.stderr)
        sys.exit(1)

    # GFLOPS = (2 * nnz * n_cols) / (time_ms * 1e6)  =>  time_ms = (2 * nnz * n_cols) / (GFLOPS * 1e6)
    time_ms = (2 * A.nnz * args.n_cols) / (gflops * 1e6)
    try:
        check_below_peak("ASpT", A.nnz, args.n_cols, time_ms, PEAK_GFLOPS_FP32)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_timer("loading", loading_ms)
    print_timer("transfer", transfer_ms)
    print_timer("operation", time_ms)

if __name__ == "__main__":
    main()
