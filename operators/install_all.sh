#!/bin/bash
#
# Master installation script for SpMM operator baselines
#
# This script helps set up the environment and install SpMM kernels
# for benchmarking sparse matrix operations.
#
# Usage:
#   ./install_all.sh                  # Interactive mode - choose what to install
#   ./install_all.sh --all            # Install everything
#   ./install_all.sh --flashsparse    # Install only FlashSparse
#   ./install_all.sh --dtc            # Install only DTC-SpMM
#   ./install_all.sh --aspt           # Install only ASpT
#   ./install_all.sh --smat           # Install only SMaT
#   ./install_all.sh --check          # Only check prerequisites
#   ./install_all.sh --clean          # Clean install (remove existing builds)
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
INSTALL_FLASHSPARSE=false
INSTALL_DTC=false
INSTALL_ASPT=false
INSTALL_SMAT=false
INSTALL_ALL=false
CHECK_ONLY=false
CLEAN=""

for arg in "$@"; do
    case $arg in
        --flashsparse) INSTALL_FLASHSPARSE=true ;;
        --dtc) INSTALL_DTC=true ;;
        --aspt) INSTALL_ASPT=true ;;
        --smat) INSTALL_SMAT=true ;;
        --all) INSTALL_ALL=true ;;
        --check) CHECK_ONLY=true ;;
        --clean) CLEAN="--clean" ;;
        --help|-h)
            head -20 "$0" | tail -17
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "SpMM Operators Installation"
echo "=========================================="
echo ""

# ========== PREREQUISITE CHECKS ==========
echo "Checking prerequisites..."
echo ""

PREREQ_OK=true

# Check CUDA
echo -n "CUDA (nvcc): "
if ! command -v nvcc &> /dev/null; then
    # Try loading module
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || true
fi
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}✓ Found ($CUDA_VER)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Install CUDA Toolkit or run: module load cuda"
    PREREQ_OK=false
fi

# Check Python
echo -n "Python: "
if command -v python &> /dev/null; then
    PYTHON_VER=$(python --version 2>&1)
    echo -e "${GREEN}✓ Found ($PYTHON_VER)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    PREREQ_OK=false
fi

# Check PyTorch with CUDA
echo -n "PyTorch (CUDA): "
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(f'{torch.__version__} CUDA {torch.version.cuda}')")
    echo -e "${GREEN}✓ Found ($TORCH_VER)${NC}"
else
    echo -e "${RED}✗ Not found or no CUDA support${NC}"
    echo "  Install with: pip install torch --index-url https://download.pytorch.org/whl/cu118"
    PREREQ_OK=false
fi

# Check cmake
echo -n "cmake: "
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version | head -1)
    echo -e "${GREEN}✓ Found ($CMAKE_VER)${NC}"
else
    echo -e "${YELLOW}⚠ Not found (needed for DTC-SpMM)${NC}"
fi

# Check git
echo -n "git: "
if command -v git &> /dev/null; then
    echo -e "${GREEN}✓ Found${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    PREREQ_OK=false
fi

# GPU info
echo -n "GPU: "
if python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_ARCH=$(python -c "import torch; cc = torch.cuda.get_device_capability(); print(f'sm_{cc[0]}{cc[1]}')")
    echo -e "${GREEN}✓ $GPU_NAME ($GPU_ARCH)${NC}"
else
    echo -e "${YELLOW}⚠ Could not detect (will use defaults)${NC}"
fi

echo ""

if [ "$PREREQ_OK" = false ]; then
    echo -e "${RED}Some prerequisites are missing. Please install them first.${NC}"
    exit 1
fi

if [ "$CHECK_ONLY" = true ]; then
    echo "All prerequisites OK!"
    exit 0
fi

# ========== INTERACTIVE SELECTION ==========
if [ "$INSTALL_ALL" = false ] && [ "$INSTALL_FLASHSPARSE" = false ] && \
   [ "$INSTALL_DTC" = false ] && [ "$INSTALL_ASPT" = false ] && [ "$INSTALL_SMAT" = false ]; then
    echo "Select components to install (or use --all for everything):"
    echo ""
    
    read -p "Install FlashSparse? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        INSTALL_FLASHSPARSE=true
    fi
    
    read -p "Install DTC-SpMM? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        INSTALL_DTC=true
    fi
    
    read -p "Install ASpT? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        INSTALL_ASPT=true
    fi
    
    read -p "Install SMaT? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        INSTALL_SMAT=true
    fi
    
    echo ""
fi

if [ "$INSTALL_ALL" = true ]; then
    INSTALL_FLASHSPARSE=true
    INSTALL_DTC=true
    INSTALL_ASPT=true
    INSTALL_SMAT=true
fi

# ========== INSTALLATIONS ==========

if [ "$INSTALL_FLASHSPARSE" = true ]; then
    echo ""
    echo "=========================================="
    echo "Installing FlashSparse..."
    echo "=========================================="
    bash "${SCRIPT_DIR}/install_flashsparse.sh" $CLEAN
fi

if [ "$INSTALL_DTC" = true ]; then
    echo ""
    echo "=========================================="
    echo "Installing DTC-SpMM..."
    echo "=========================================="
    bash "${SCRIPT_DIR}/install_dtc.sh" $CLEAN
fi

if [ "$INSTALL_ASPT" = true ]; then
    echo ""
    echo "=========================================="
    echo "Installing ASpT..."
    echo "=========================================="
    bash "${SCRIPT_DIR}/install_aspt.sh" $CLEAN
fi

if [ "$INSTALL_SMAT" = true ]; then
    echo ""
    echo "=========================================="
    echo "Installing SMaT..."
    echo "=========================================="
    bash "${SCRIPT_DIR}/install_smat.sh" $CLEAN
fi

# ========== SUMMARY ==========
echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="

check_module() {
    if python -c "import $1" 2>/dev/null; then
        echo -e "  $2: ${GREEN}✓ Installed${NC}"
        return 0
    else
        echo -e "  $2: ${RED}✗ Not installed${NC}"
        return 1
    fi
}

echo ""
echo "Module status:"
check_module "FS_SpMM" "FlashSparse" || true
check_module "DTCSpMM" "DTC-SpMM" || true
check_module "aspt" "ASpT" || true
check_module "smat" "SMaT" || true

echo ""
echo "cuSPARSE (via PyTorch) is always available with PyTorch CUDA."
echo ""
echo "Done!"
