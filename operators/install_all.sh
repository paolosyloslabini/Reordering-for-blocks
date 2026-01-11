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

# Initialize modules if available (needed for many clusters)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine the best python executable
if command -v python3 &> /dev/null; then
    PYTHON_EXE=$(command -v python3)
elif command -v python &> /dev/null; then
    PYTHON_EXE=$(command -v python)
else
    PYTHON_EXE="python"
fi

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
            head -25 "$0" | tail -22
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "SpMM Operators Installation (Login Node Friendly)"
echo "=========================================="
echo ""

# ========== PREREQUISITE CHECKS ==========
echo "Checking prerequisites..."
echo ""

PREREQ_OK=true

# Check CUDA
echo -n "CUDA (nvcc): "
if ! command -v nvcc &> /dev/null; then
    # Try loading common module names
    module load cuda 2>/dev/null || module load CUDA 2>/dev/null || module load nvidia/cuda 2>/dev/null || true
fi
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "${GREEN}✓ Found ($CUDA_VER)${NC}"
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Compilation requires nvcc. Please: module load cuda"
    PREREQ_OK=false
fi

# Check Python & PyTorch
echo -n "PyTorch: "
if $PYTHON_EXE -c "import torch" 2>/dev/null; then
    TORCH_VER=$($PYTHON_EXE -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}✓ Found ($TORCH_VER)${NC}"
    
    echo -n "GPU Support: "
    if $PYTHON_EXE -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo -e "${GREEN}✓ Available${NC}"
    else
        echo -e "${YELLOW}⚠ Not detected (Normal on login nodes, installation can proceed)${NC}"
    fi
else
    echo -e "${RED}✗ Not found${NC}"
    echo "  Please install PyTorch in your active environment first."
    PREREQ_OK=false
fi

# Check cmake
echo -n "cmake: "
if command -v cmake &> /dev/null; then
    CMAKE_VER=$(cmake --version | head -1)
    echo -e "${GREEN}✓ Found ($CMAKE_VER)${NC}"
else
    echo -e "${YELLOW}⚠ Not found (Required for DTC-SpMM)${NC}"
fi

echo ""

if [ "$PREREQ_OK" = false ] && [ "$CHECK_ONLY" = false ]; then
    echo -e "${RED}Essential prerequisites are missing. Cannot proceed with installation.${NC}"
    exit 1
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
