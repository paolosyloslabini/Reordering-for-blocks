import numpy as np
import scipy.io
import scipy.sparse
import subprocess
import tempfile
import os
import sys
from cusparse_utils import load_and_permute_matrix


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wrapper for C++ cuSPARSE BSR SpMM operator with permutation support.")
    parser.add_argument("mtx", help="Input matrix Market file")
    parser.add_argument("--perm", help="Permutation file (optional)")
    parser.add_argument("--blocksize", type=int, default=8, help="Block size for BSR conversion")
    parser.add_argument("--perm-type", choices=["ROW", "SYMMETRIC", "ASYMMETRIC"], default="ROW", help="Permutation type")
    parser.add_argument("--n-cols", type=int, default=32, help="Number of columns in dense B")
    parser.add_argument("--n-iterations", type=int, default=5, help="Number of SpMM iterations")
    args = parser.parse_args()

    # Load and permute matrix (uses shared utility function)
    mat = load_and_permute_matrix(args.mtx, args.perm, args.perm_type)
    
    # Save permuted matrix to temp file
    with tempfile.NamedTemporaryFile(suffix=".mtx", delete=False) as tmp:
        scipy.io.mmwrite(tmp.name, mat)
        tmp_mtx = tmp.name
    # Build command for C++ binary
    cmd = [os.path.join(os.path.dirname(__file__), "cusparse_bsr_spmm"), tmp_mtx, "--blocksize", str(args.blocksize), "--n-cols", str(args.n_cols), "--n-iterations", str(args.n_iterations)]
    # Run C++ binary
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    # Clean up temp file
    os.remove(tmp_mtx)

if __name__ == "__main__":
    main()
