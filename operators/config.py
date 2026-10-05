"""
Global configuration parameters for cuSPARSE operators.
"""

# SpMM/SpMV Operation Parameters
ALPHA_DEFAULT = 1.0
BETA_DEFAULT = 0.0

# SpMM Dense Matrix Parameters
SPMM_N_COLS_DEFAULT = 32

# Timing Parameters
N_ITERATIONS_DEFAULT = 5

# BSR Format Parameters
BSR_BLOCKSIZE_DEFAULT = 8

# Permutation Parameters
PERM_TYPE_DEFAULT = 'ROW'

# Hardware peak throughput (GFLOPS), used to reject physically impossible
# measurements (e.g. a kernel that silently computed fewer columns than asked).
# NVIDIA A100: 19.5 TFLOPS FP32 on CUDA cores, 312 TFLOPS FP16 on Tensor Cores.
PEAK_GFLOPS_FP32 = 19500.0
PEAK_GFLOPS_FP16_TC = 312000.0
