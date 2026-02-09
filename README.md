# ReorderingSurvey-2026

A comprehensive research framework for evaluating **sparse matrix reordering algorithms** and their impact on **GPU SpMM (Sparse Matrix-Matrix Multiplication) performance**.

## Purpose

This repository systematically:
1. Downloads and manages sparse matrices from the SuiteSparse Matrix Collection
2. Generates permutations using various reordering algorithms
3. Analyzes matrix structural properties (bandwidth, block density, locality)
4. Benchmarks SpMM performance across multiple GPU kernels
5. Correlates reordering-induced structural changes with performance gains

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| [`scripts/`](scripts/README.md) | Result parsing, plotting, and job monitoring utilities |
| [`yamls/`](yamls/README.md) | Experiment configuration files for SbatchMan |
| [`operators/`](operators/) | SpMM kernel implementations and Python wrappers |
| [`MtxPerm/`](MtxPerm/) | Permutation generation tools (SparseBase, GraphBLAS, SPARTA) |
| [`datasets/`](datasets/README.md) | Sparse matrices from SuiteSparse collection |
| [`results/`](results/README.md) | Aggregated CSV results from experiments |
| [`perms/`](perms/README.md) | Generated permutation files organized by algorithm |
| [`plots/`](plots/) | Generated visualization plots |
| [`ccutils/`](ccutils/) | Header-only C++/CUDA utility library |
| [`distributed_mmio/`](distributed_mmio/) | Matrix Market I/O library |
| [`SbatchMan/`](SbatchMan/) | SLURM batch job management tool |

## Quick Reference for Key Files

### Configuration
- **[yamls/perms.yaml](yamls/perms.yaml)** - All reordering algorithms and generation commands
- **[yamls/configs.yaml](yamls/configs.yaml)** - SLURM job configurations (GPU, CPU, memory)
- **[yamls/matrices.yaml](yamls/matrices.yaml)** - Matrix download criteria
- **[yamls/operations_*.yaml](yamls/)** - SpMM benchmark configurations
- **[yamls/analysis_*.yaml](yamls/)** - Matrix analysis configurations

### Core Scripts
- **[scripts/parse_results.py](scripts/parse_results.py)** - Aggregates job outputs into CSVs
- **[scripts/plot.py](scripts/plot.py)** - Generates correlation and performance plots
- **[scripts/job_check.py](scripts/job_check.py)** - Monitors SLURM job status

### Operator Wrappers
- **[operators/cusparse_csr_spmm.py](operators/cusparse_csr_spmm.py)** - cuSPARSE CSR SpMM
- **[operators/cusparse_bsr_spmm.py](operators/cusparse_bsr_spmm.py)** - cuSPARSE BSR SpMM
- **[operators/flashsparse_spmm.py](operators/flashsparse_spmm.py)** - FlashSparse (Tensor Core)
- **[operators/dtc_spmm.py](operators/dtc_spmm.py)** - DTC-SpMM
- **[operators/config.py](operators/config.py)** - Shared configuration and utilities

### Permutation Tools
- **[MtxPerm/SPARSEBASE/](MtxPerm/SPARSEBASE/)** - C++ reordering via SparseBase library
- **[MtxPerm/RANDOM/](MtxPerm/RANDOM/)** - GraphBLAS random permutations
- **[MtxPerm/SPARTA/](MtxPerm/SPARTA/)** - SPARTA blocking algorithm
- **[MtxPerm/ANALYSIS/](MtxPerm/ANALYSIS/)** - Matrix structure analysis

## Reordering Algorithms

Defined in [yamls/perms.yaml](yamls/perms.yaml):

| Algorithm | Source | Description |
|-----------|--------|-------------|
| `identity` | - | No reordering (baseline) |
| `random1D` | GraphBLAS | Random row permutation |
| `random2D` | GraphBLAS | Random row + column permutation |
| `SB_rcm` | SparseBase | Reverse Cuthill-McKee (bandwidth reduction) |
| `SB_degree` | SparseBase | Degree-based ordering |
| `SB_gray` | SparseBase | Gray code reordering |
| `SB_amd` | SparseBase | Approximate Minimum Degree |
| `SB_metis` | SparseBase | METIS graph partitioning |
| `SB_rabbit` | SparseBase | Rabbit reordering |
| `SB_slashburn` | SparseBase | SlashBurn algorithm |
| `SPARTA_reorder` | SPARTA | Sparse blocking for tensor cores |

## Permutation Types

Three modes for applying permutations:

- **ROW**: `P * A` - Permute rows only
- **SYMMETRIC**: `P * A * P^T` - Same permutation for rows and columns (square matrices)
- **ASYMMETRIC**: `P_row * A * P_col^T` - Different permutations for rows and columns

## SpMM Kernels Benchmarked

| Kernel | Description |
|--------|-------------|
| `CUSPARSE_SPMM_CSR` | cuSPARSE CSR format |
| `CUSPARSE_SPMM_BSR` | cuSPARSE Block Sparse Row |
| `FLASHSPARSE_SPMM` | FlashSparse Tensor Core (PPoPP'25) |
| `DTC_SPMM` | Dynamic Tensor Core SpMM |
| `ASPT_SPMM` | Adaptive Sparse Tiling |
| `SMAT_SPMM` | Sparse Matrix Tensor Core |

## Data Formats

| Format | Description |
|--------|-------------|
| `.mtx` | Matrix Market format (standard sparse matrix) |
| `.perm` | Permutation file: 1-based indices, space-separated, 1-2 lines |
| `.csv` | Result files (Pandas-compatible) |
| `.yaml` | SbatchMan experiment configurations |

## Permutation File Format

```
# Single line = symmetric or row-only permutation
3 1 4 2 5

# Two lines = asymmetric permutation (row perm, then col perm)
3 1 4 2 5
2 4 1 5 3
```

Indices are **1-based**.

## Typical Workflow

1. **Download matrices**: Configure [yamls/matrices.yaml](yamls/matrices.yaml), run SbatchMan
2. **Generate permutations**: Configure [yamls/perms.yaml](yamls/perms.yaml), run SbatchMan
3. **Run analysis**: Use `yamls/analysis_*.yaml` to compute matrix metrics
4. **Run benchmarks**: Use `yamls/operations_*.yaml` to benchmark SpMM kernels
5. **Parse results**: `python scripts/parse_results.py`
6. **Generate plots**: `python scripts/plot.py --one-per-family`

## Key Metrics

### Analysis Metrics (from MtxPerm/ANALYSIS)
- `bandwidth_max`: Maximum matrix bandwidth
- `block_density_{4,8,16,32,64,128}`: Nonzero fraction in occupied blocks
- `locality_vertical_adjacency_ratio`: Consecutive column pairs ratio
- `locality_avg_row_spread`: Average spread of nonzeros per row

### Performance Metrics
- `time_operation_ms`: SpMM execution time
- `gflops`: Computed from nnz and time
- `speedup`: Relative to identity (no reordering)

## Dependencies

### Python
- PyTorch (with CUDA), CuPy
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- python-graphblas
- ssgetpy (SuiteSparse download)

### C++/CUDA
- CUDA Toolkit (≥11.8)
- SparseBase library
- CMake (≥3.12)
- OpenMP
- Optional: METIS, PaToH

## Environment

Designed for SLURM cluster execution. Job configurations in [yamls/configs.yaml](yamls/configs.yaml).

**Note**: DTC-SpMM uses a separate conda env (`DTCSpMM`) and requires `operators/dtc_preprocess.sh` for module/library setup.
