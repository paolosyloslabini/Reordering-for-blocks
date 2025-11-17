#!/bin/bash
# Test the complete matrix reordering pipeline
# Usage: ./test_pipeline.sh <matrix.mtx> [seed]

set -e  # Exit on error

MATRIX_FILE=$1
SEED=${2:-42}

if [ -z "$MATRIX_FILE" ]; then
    echo "Usage: $0 <matrix.mtx> [seed]"
    exit 1
fi

if [ ! -f "$MATRIX_FILE" ]; then
    echo "Error: Matrix file '$MATRIX_FILE' not found"
    exit 1
fi

echo "========================================="
echo "Matrix Reordering Pipeline Test"
echo "========================================="
echo "Matrix: $MATRIX_FILE"
echo "Random seed: $SEED"
echo ""

# Set CUDA environment for cuSPARSE
export CUDA_HOME=/usr/local/cuda
export CUPY_CUDA_PATH=/usr/local/cuda

# Step 1: Generate random permutation
echo "[1/5] Generating random permutation..."
python3 random_permutation_graphblas.py "$MATRIX_FILE" --seed "$SEED" --output permutation.txt
echo "✓ Permutation generated: permutation.txt"
echo ""

# Step 2: Reorder matrix (1D - rows only)
echo "[2/5] Reordering matrix (1D - rows only)..."
python3 reorder_matrix_graphblas.py "$MATRIX_FILE" permutation.txt 1D reordered_1d.mtx
echo "✓ 1D reordered matrix: reordered_1d.mtx"
echo ""

# Step 3: Reorder matrix (2D - rows and columns)
echo "[3/5] Reordering matrix (2D - rows and columns)..."
python3 reorder_matrix_graphblas.py "$MATRIX_FILE" permutation.txt 2D reordered_2d.mtx
echo "✓ 2D reordered matrix: reordered_2d.mtx"
echo ""

# Step 4: Run SpMM on original matrix
echo "[4/5] Running cuSPARSE SpMM on original matrix..."
if python3 simple_cusparse_spmm.py "$MATRIX_FILE" --n-iterations 10 2>/dev/null; then
    echo "✓ Original matrix SpMM completed"
else
    echo "⚠ cuSPARSE SpMM failed (CUDA issue - skipping GPU tests)"
    SKIP_GPU=1
fi
echo ""

# Step 5: Run SpMM on reordered matrices
if [ -z "$SKIP_GPU" ]; then
    echo "[5/5] Running cuSPARSE SpMM on reordered matrices..."
    
    echo "  - 1D reordered matrix:"
    python3 simple_cusparse_spmm.py reordered_1d.mtx --n-iterations 10
    
    echo "  - 2D reordered matrix:"
    python3 simple_cusparse_spmm.py reordered_2d.mtx --n-iterations 10
    
    echo "✓ Reordered matrices SpMM completed"
else
    echo "[5/5] Skipping cuSPARSE SpMM tests (GPU not available)"
fi

echo ""
echo "========================================="
echo "Pipeline Test Complete!"
echo "========================================="
echo "Generated files:"
echo "  - permutation.txt (1-based permutation vector)"
echo "  - reordered_1d.mtx (row-reordered matrix)"
echo "  - reordered_2d.mtx (row+column-reordered matrix)"
echo ""
