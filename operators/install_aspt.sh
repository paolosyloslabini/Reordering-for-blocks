#!/bin/bash
#
# Install script for ASpT (Adaptive Sparse Tiling)
# Clones the repository and builds the CUDA SpMM kernels
#
# Requirements:
#   - CUDA Toolkit
#   - nvcc compiler in PATH
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASPT_DIR="${SCRIPT_DIR}/ASpT"
ASPT_REPO="https://github.com/LucasWilkinson/ASpT-mirror"

echo "=========================================="
echo "ASpT Installation Script"
echo "=========================================="
echo "Target directory: ${ASPT_DIR}"
echo "Repository: ${ASPT_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/4] Checking prerequisites..."

# Try to load modules if command not found (common on clusters)
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found, attempting to load cuda module..."
    module load cuda || echo "Warning: Could not load cuda module"
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit and ensure nvcc is in PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $NF}')
echo "✓ CUDA found: $CUDA_VERSION"

# Step 2: Clone repository
echo ""
echo "[2/4] Cloning repository..."
if [ -d "${ASPT_DIR}" ]; then
    echo "Directory ${ASPT_DIR} already exists. Pulling latest changes..."
    cd "${ASPT_DIR}"
    git pull
else
    git clone "${ASPT_REPO}" "${ASPT_DIR}"
    cd "${ASPT_DIR}"
fi

# Step 3: Build
echo ""
echo "[3/4] Building ASpT kernels..."

# Navigate to the GPU SpMM directory
cd "${ASPT_DIR}/ASpT_SpMM_GPU"

# Define compilation flags
# Targeting A100 (sm_80) as per environment context
ARCH_FLAGS="-gencode arch=compute_80,code=sm_80"
NVCC_FLAGS="-std=c++11 -O3 ${ARCH_FLAGS} --use_fast_math -Xptxas -v,-dlcm=ca"

echo "Compiling sspmm_32 (Single Precision)..."
nvcc ${NVCC_FLAGS} sspmm_32.cu -o sspmm_32

echo "Compiling dspmm_32 (Double Precision)..."
nvcc ${NVCC_FLAGS} dspmm_32.cu -o dspmm_32

# Step 4: Verify
echo ""
echo "[4/4] Verifying build..."
if [ -f "sspmm_32" ] && [ -f "dspmm_32" ]; then
    echo "✓ Build successful!"
    echo "Binaries created:"
    echo "  - ${ASPT_DIR}/ASpT_SpMM_GPU/sspmm_32"
    echo "  - ${ASPT_DIR}/ASpT_SpMM_GPU/dspmm_32"
else
    echo "ERROR: Build failed - binaries not found"
    exit 1
fi

echo ""
echo "Installation complete."
