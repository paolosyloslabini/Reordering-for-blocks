#!/bin/bash
#
# Install script for FlashSparse
# Clones the repository and builds the CUDA kernels
#
# Requirements:
#   - CUDA Toolkit >= 11.8
#   - nvcc compiler in PATH
#   - PyTorch with CUDA support
#   - cmake >= 3.29 (recommended)
#
# Usage:
#   ./install_flashsparse.sh                  # Standard install
#   ./install_flashsparse.sh --clean          # Clean and reinstall
#   TORCH_CUDA_ARCH_LIST="8.6 8.9" ./install_flashsparse.sh  # Specific arch
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
FLASHSPARSE_DIR="${SCRIPT_DIR}/FlashSparse"
FLASHSPARSE_REPO="https://github.com/ParCIS/FlashSparse.git"

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
    esac
done

echo "=========================================="
echo "FlashSparse Installation Script"
echo "=========================================="
echo "Target directory: ${FLASHSPARSE_DIR}"
echo "Repository: ${FLASHSPARSE_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/5] Checking prerequisites..."

# Try to load CUDA module on clusters
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found, attempting to load cuda module..."
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || echo "Warning: Could not load cuda module"
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit >= 11.8 and ensure nvcc is in PATH"
    echo "On clusters, try: module load cuda"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✓ CUDA found: $CUDA_VERSION"

# Check for PyTorch with CUDA
if python -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
    echo "✓ PyTorch found: $TORCH_VERSION (CUDA $TORCH_CUDA)"
else
    echo "ERROR: PyTorch with CUDA support not found"
    echo ""
    echo "Please install PyTorch with CUDA. Example:"
    echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "  # or"
    echo "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Detect GPU architecture if not specified
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    echo ""
    echo "Detecting GPU architecture..."
    GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_ARCH" ]; then
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
        echo "✓ Detected GPU compute capability: $GPU_ARCH"
    else
        echo "Warning: Could not detect GPU. Will use PyTorch defaults."
    fi
else
    echo "✓ Using specified TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
fi

# Step 2: Clone repository
echo ""
echo "[2/5] Cloning repository..."
if [ "$CLEAN_BUILD" = true ] && [ -d "${FLASHSPARSE_DIR}" ]; then
    echo "Removing existing directory for clean build..."
    rm -rf "${FLASHSPARSE_DIR}"
fi

if [ -d "${FLASHSPARSE_DIR}" ]; then
    echo "Directory ${FLASHSPARSE_DIR} already exists. Pulling latest changes..."
    cd "${FLASHSPARSE_DIR}"
    git pull || echo "Warning: git pull failed, continuing with existing code"
else
    git clone --recursive "${FLASHSPARSE_REPO}" "${FLASHSPARSE_DIR}"
    cd "${FLASHSPARSE_DIR}"
fi

# Step 3: Build FlashSparse
echo ""
echo "[3/5] Building FlashSparse..."

cd "${FLASHSPARSE_DIR}/FlashSparse"

# Clean previous build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning previous build artifacts..."
    rm -rf build dist *.egg-info
fi

# Use compile.sh if available (recommended by FlashSparse repo), otherwise setup.py
if [ -f "compile.sh" ]; then
    echo "Using compile.sh for build..."
    bash compile.sh
else
    echo "Using setup.py for build..."
    python setup.py install
fi

# Step 4: Verify SpMM module
echo ""
echo "[4/5] Verifying FS_SpMM module..."
cd "${SCRIPT_DIR}"  # Go back to avoid import issues with local files

if python -c "import FS_SpMM; print('✓ FS_SpMM module loaded successfully')" 2>/dev/null; then
    :
else
    echo "ERROR: Could not import FS_SpMM module"
    echo "Build may have failed. Check the output above for errors."
    exit 1
fi

# Step 5: Verify Block module
echo "[5/5] Verifying FS_Block_gpu module..."
if python -c "import FS_Block_gpu; print('✓ FS_Block_gpu module loaded successfully')" 2>/dev/null; then
    :
else
    echo "ERROR: Could not import FS_Block_gpu module"
    echo "Build may have failed. Check the output above for errors."
    exit 1
fi

echo ""
echo "=========================================="
echo "FlashSparse installation complete!"
echo "=========================================="
echo ""
echo "You can now use FlashSparse in your Python scripts:"
echo "  import FS_SpMM"
echo "  import FS_Block_gpu"
echo ""
echo "Test with:"
echo "  python operators/flashsparse_spmm.py <matrix.mtx> --n-cols 32"
echo ""
