#!/bin/bash
#
# Install script for ASpT (Adaptive Sparse Tiling)
# Clones the repository and builds the CUDA SpMM kernels
#
# Requirements:
#   - CUDA Toolkit
#   - nvcc compiler in PATH
#
# Usage:
#   ./install_aspt.sh                # Standard install
#   ./install_aspt.sh --clean        # Clean and reinstall
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASPT_DIR="${SCRIPT_DIR}/ASpT"
ASPT_REPO="https://github.com/LucasWilkinson/ASpT-mirror"

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
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || echo "Warning: Could not load cuda module"
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit and ensure nvcc is in PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✓ CUDA found: $CUDA_VERSION"

# Detect GPU architecture
GPU_ARCH=""
if python -c "import torch" 2>/dev/null; then
    GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}{cc[1]}')" 2>/dev/null || echo "")
fi
if [ -z "$GPU_ARCH" ]; then
    GPU_ARCH="80"  # Default to A100
    echo "Warning: Could not detect GPU, defaulting to sm_${GPU_ARCH}"
else
    echo "✓ Detected GPU: sm_${GPU_ARCH}"
fi

# Step 2: Clone repository
echo ""
echo "[2/4] Cloning repository..."
if [ "$CLEAN_BUILD" = true ] && [ -d "${ASPT_DIR}" ]; then
    echo "Removing existing directory for clean build..."
    rm -rf "${ASPT_DIR}"
fi

if [ -d "${ASPT_DIR}" ]; then
    echo "Directory ${ASPT_DIR} already exists. Pulling latest changes..."
    cd "${ASPT_DIR}"
    git pull || echo "Warning: git pull failed, continuing with existing code"
else
    git clone "${ASPT_REPO}" "${ASPT_DIR}"
    cd "${ASPT_DIR}"
fi

# Step 3: Build
echo ""
echo "[3/4] Building ASpT kernels..."

# Navigate to the GPU SpMM directory
cd "${ASPT_DIR}/ASpT_SpMM_GPU"

# Define compilation flags using detected architecture.
# -include shfl_fix.h maps the pre-Volta __shfl* intrinsics to __shfl*_sync;
# sspmm_32.cu includes it itself, sspmm_128.cu does not and fails without it.
ARCH_FLAGS="-gencode arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}"
NVCC_FLAGS="-std=c++11 -O3 ${ARCH_FLAGS} --use_fast_math -include shfl_fix.h -Xptxas -v,-dlcm=ca"

# Timed launches per run. Upstream hardcodes ITER=1 (a single cold launch);
# the binaries already average GFLOPS over ITER and scale the validation
# reference accordingly. Override with ASPT_ITER=<n>.
ASPT_ITER="${ASPT_ITER:-10}"
echo "Timed iterations per run (ITER): ${ASPT_ITER}"

# Build from a copy with ITER overridable, leaving the cloned sources untouched.
build() {
    local name=$1
    sed 's|^#define ITER (128/128)|#ifndef ITER\n#define ITER (128/128)\n#endif|' "${name}.cu" > "build_${name}.cu"
    grep -q '^#ifndef ITER' "build_${name}.cu" || { echo "ERROR: could not patch ITER in ${name}.cu"; exit 1; }
    echo "Compiling ${name} for sm_${GPU_ARCH}..."
    nvcc ${NVCC_FLAGS} -DITER=${ASPT_ITER} "build_${name}.cu" -o "${name}"
    rm -f "build_${name}.cu"
}

# sspmm_32 for n_cols = 32, sspmm_128 for n_cols a multiple of 64 (see aspt_spmm.py)
build sspmm_32
build sspmm_128
build dspmm_32

# Step 4: Verify
echo ""
echo "[4/4] Verifying build..."
if [ -f "sspmm_32" ] && [ -f "sspmm_128" ] && [ -f "dspmm_32" ]; then
    echo "✓ Build successful!"
    echo "Binaries created:"
    echo "  - ${ASPT_DIR}/ASpT_SpMM_GPU/sspmm_32"
    echo "  - ${ASPT_DIR}/ASpT_SpMM_GPU/sspmm_128"
    echo "  - ${ASPT_DIR}/ASpT_SpMM_GPU/dspmm_32"
else
    echo "ERROR: Build failed - binaries not found"
    exit 1
fi

echo ""
echo "Installation complete."
