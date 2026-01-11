#!/bin/bash
#
# Install script for DTC-SpMM (Dynamic Tensor Core SpMM)
# Clones the repository and builds the CUDA kernels
#
# Requirements:
#   - CUDA Toolkit >= 11.0
#   - nvcc compiler in PATH
#   - PyTorch with CUDA support
#   - cmake, make
#
# Usage:
#   ./install_dtc.sh                          # Standard install
#   ./install_dtc.sh --clean                  # Clean and reinstall
#   TORCH_CUDA_ARCH_LIST="8.6 8.9" ./install_dtc.sh  # Specific arch
#

set -e  # Exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DTC_DIR="${SCRIPT_DIR}/DTC-SpMM"
DTC_REPO="https://github.com/Shigangli/DTC-SpMM.git"

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
echo "DTC-SpMM Installation Script"
echo "=========================================="
echo "Target directory: ${DTC_DIR}"
echo "Repository: ${DTC_REPO}"
echo ""

# Step 1: Check prerequisites
echo "[1/6] Checking prerequisites..."

# Try to load modules on clusters
if ! command -v nvcc &> /dev/null; then
    echo "nvcc not found, attempting to load cuda module..."
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || echo "Warning: Could not load cuda module"
fi

if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found in PATH"
    echo "Please install CUDA Toolkit and ensure nvcc is in PATH"
    echo "On clusters, try: module load cuda"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✓ CUDA found: $CUDA_VERSION"

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found in PATH"
    exit 1
fi
echo "✓ cmake found: $(cmake --version | head -1)"

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
    exit 1
fi

# Detect GPU architecture if not specified
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
    echo ""
    echo "Detecting GPU architecture..."
    GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'{cc[0]}.{cc[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_ARCH" ]; then
        # DTC-SpMM cmake uses different format (e.g., "89" instead of "8.9")
        GPU_ARCH_CMAKE=$(echo "$GPU_ARCH" | tr -d '.')
        export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
        echo "✓ Detected GPU compute capability: $GPU_ARCH (cmake: $GPU_ARCH_CMAKE)"
    else
        # Default to common architectures
        GPU_ARCH_CMAKE="86;89"
        export TORCH_CUDA_ARCH_LIST="8.6 8.9"
        echo "Warning: Could not detect GPU. Using defaults: $TORCH_CUDA_ARCH_LIST"
    fi
else
    # Convert TORCH_CUDA_ARCH_LIST format to cmake format
    GPU_ARCH_CMAKE=$(echo "$TORCH_CUDA_ARCH_LIST" | tr ' ' ';' | tr -d '.')
    echo "✓ Using specified TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
fi

# Step 2: Clone repository
echo ""
echo "[2/6] Cloning DTC-SpMM repository..."
if [ "$CLEAN_BUILD" = true ] && [ -d "${DTC_DIR}" ]; then
    echo "Removing existing directory for clean build..."
    rm -rf "${DTC_DIR}"
fi

if [ -d "${DTC_DIR}" ]; then
    echo "Directory ${DTC_DIR} already exists. Pulling latest changes..."
    cd "${DTC_DIR}"
    git pull || echo "Warning: git pull failed, continuing with existing code"
    git submodule update --init --recursive || echo "Warning: submodule update failed"
else
    git clone --recursive "${DTC_REPO}" "${DTC_DIR}"
    cd "${DTC_DIR}"
fi

export DTC_HOME="${DTC_DIR}"

# Step 3: Build glog (dependency)
echo ""
echo "[3/6] Building glog..."
cd "${DTC_HOME}/third_party/glog"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf build
fi

mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX="${DTC_HOME}/third_party/glog/build" ..
make -j$(nproc)
make install

export GLOG_PATH="${DTC_HOME}/third_party/glog"
export LD_LIBRARY_PATH="${GLOG_PATH}/build/lib:${GLOG_PATH}/build/lib64:${LD_LIBRARY_PATH}"
export CPLUS_INCLUDE_PATH="${GLOG_PATH}/build/include:${CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="${GLOG_PATH}/build/lib:${GLOG_PATH}/build/lib64:${LIBRARY_PATH}"
echo "✓ glog built successfully"

# Step 4: Build Sputnik (dependency)
echo ""
echo "[4/6] Building Sputnik..."
cd "${DTC_HOME}/third_party/sputnik"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf build
fi

# Find the glog library (might be .so or .a, in lib or lib64)
GLOG_LIB=""
for lib in "${GLOG_PATH}/build/lib/libglog.so" "${GLOG_PATH}/build/lib64/libglog.so" \
           "${GLOG_PATH}/build/lib/libglog.a" "${GLOG_PATH}/build/lib64/libglog.a"; do
    if [ -f "$lib" ]; then
        GLOG_LIB="$lib"
        break
    fi
done
if [ -z "$GLOG_LIB" ]; then
    echo "Warning: Could not find glog library, using default path"
    GLOG_LIB="${GLOG_PATH}/build/lib/libglog.so"
fi

mkdir -p build && cd build
cmake .. \
    -DGLOG_INCLUDE_DIR="${GLOG_PATH}/build/include" \
    -DGLOG_LIBRARY="${GLOG_LIB}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TEST=OFF \
    -DBUILD_BENCHMARK=OFF \
    -DCUDA_ARCHS="${GPU_ARCH_CMAKE}"
make -j$(nproc)

export SPUTNIK_PATH="${DTC_HOME}/third_party/sputnik"
export LD_LIBRARY_PATH="${SPUTNIK_PATH}/build/sputnik:${LD_LIBRARY_PATH}"
echo "✓ Sputnik built successfully"

# Step 5: Build DTC-SpMM
echo ""
echo "[5/6] Building DTC-SpMM..."
cd "${DTC_HOME}/DTC-SpMM"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf build dist *.egg-info
fi

python setup.py install

# Step 6: Verify
echo ""
echo "[6/6] Verifying installation..."
cd "${SCRIPT_DIR}"  # Go back to avoid import issues

if python -c "import DTCSpMM; print('✓ DTCSpMM module loaded successfully')" 2>/dev/null; then
    :
else
    echo "ERROR: Could not import DTCSpMM module"
    echo "Build may have failed. Check the output above for errors."
    exit 1
fi

echo ""
echo "=========================================="
echo "DTC-SpMM installation complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Before using DTC-SpMM, set these environment variables:"
echo "  export LD_LIBRARY_PATH=\"${GLOG_PATH}/build/lib:${SPUTNIK_PATH}/build/sputnik:\$LD_LIBRARY_PATH\""
echo ""
echo "You can now use DTC-SpMM in your Python scripts:"
echo "  import DTCSpMM"
echo ""
echo "Test with:"
echo "  python operators/dtc_spmm.py <matrix.mtx> --n-cols 32"
echo ""
