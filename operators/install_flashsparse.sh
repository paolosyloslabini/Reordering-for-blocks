#!/bin/bash
#
# Install script for FlashSparse
# Clones the repository and builds the CUDA kernels
#
# Requirements:
#   - CUDA Toolkit
#   - nvcc compiler in PATH
#   - PyTorch
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLASHSPARSE_DIR="${SCRIPT_DIR}/FlashSparse"
FLASHSPARSE_REPO="https://github.com/ParCIS/FlashSparse.git"

echo "=========================================="
echo "FlashSparse Installation Script"
echo "=========================================="
echo "Target directory: ${FLASHSPARSE_DIR}"
echo "Repository: ${FLASHSPARSE_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/4] Checking prerequisites..."

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

# Check for PyTorch
if python -c "import torch" &> /dev/null; then
    echo "✓ PyTorch found"
else
    echo "ERROR: PyTorch not found"
    echo "Please install PyTorch before continuing"
    exit 1
fi

# Step 2: Clone repository
echo ""
echo "[2/4] Cloning repository..."
if [ -d "${FLASHSPARSE_DIR}" ]; then
    echo "Directory ${FLASHSPARSE_DIR} already exists. Pulling latest changes..."
    cd "${FLASHSPARSE_DIR}"
    git pull
else
    git clone --recursive "${FLASHSPARSE_REPO}" "${FLASHSPARSE_DIR}"
    cd "${FLASHSPARSE_DIR}"
fi

# Step 3: Build
echo ""
echo "[3/4] Building FlashSparse..."

# Navigate to the FlashSparse source directory
cd "${FLASHSPARSE_DIR}/FlashSparse"

# Clean previous build
rm -rf build dist *.egg-info

# Install
python setup.py install

# Step 4: Verify
echo ""
echo "[4/4] Verifying build..."
if python -c "import FS_SpMM; print('FlashSparse SpMM module loaded successfully')" &> /dev/null; then
    echo "✓ Build successful!"
else
    echo "ERROR: Build failed - could not import FS_SpMM"
    exit 1
fi

echo ""
echo "Installation complete."
