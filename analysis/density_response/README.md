# Density response: how reordering changes block density, and how kernels respond

Two questions, answered from the existing results CSVs (no new GPU runs):

1. **Given a matrix, what block-density gain will a reordering give?**
   See `gain_vs_start_block_density` and `gain_vs_start_density`.
2. **Given a block-density change, what speedup does each kernel give?**
   See `elasticity_block_density_original`.

All figures use **original SuiteSparse matrices only**. Scrambled matrices are left out
because they hide a good ordering, so reordering recovers far more from them than from real
inputs. The matrix set and filters are the paper's (`scripts/filter_config.yaml`, 389 matrices).

```
python analysis/density_response/gain_figures.py        # figures 1-2 (~1 min)
python analysis/density_response/elasticity_figure.py   # figure 3 (~10 min, bootstrap)
```

Use the repo's Windows venv (`.venv/Scripts/python.exe`). Outputs go to `figures/` (PDF sized for
IEEE page width, plus PNG).

## Figures 1 and 2: block-density gain vs starting state

Figure 1 is `gain_vs_start_block_density`; figure 2 is `gain_vs_start_density`.

- **y-axis:** gain = 16×16 block density after reordering divided by before, for RCM, AMD, Rabbit
  and DTC-LSH.
- **Lines:** Gaussian-kernel-weighted medians along log x (bandwidth 0.15 decades).
- **Bands:** the interquartile range.
- **Where curves are drawn:** only where the window's effective sample size is at least 25.

**x = starting block density** (figure 1). This is the useful predictor.
- **Below about 1% starting density:** symmetric RCM, AMD and Rabbit typically raise block density
  about 3.5–4×, and DTC-LSH about 2×.
- **Above about 3%:** gains are around or below 1×. RCM stays closest to 1×, and Rabbit loses the
  most (down to about 0.45× on well-blocked matrices).
- **Row reordering:** gains are 1.2–1.6× at low density and turn into losses above about 2%.
- **Held-out accuracy** (5-fold, grouped by matrix, predicting log gain from the smoothed median):
  R² 0.34–0.53, with the direction (gain or loss) right 75–88% of the time.

**x = matrix density nnz/(m·n)** (figure 2). This is invariant to permutation, so it is known before
any reordering, but it does **not** predict the gain: held-out R² is between −0.03 and 0.12, and the
direction is right only 58–67% of the time. The bump around 10⁻⁵ is a group of large, very sparse
graph matrices (the fully kept graph collections), not a density effect.

Caveat: the drop around 1% starting block density in figure 1 partly reflects the population
changing, from graph matrices below it to mesh and PDE matrices above it.

## Figure 3: response of speed to block density (`elasticity_block_density_original`)

A per-kernel fixed-effects spline model:

`log2 GFLOPS_ij = mu_i + f(log2 d_ij) + e_ij`

- **i:** the matrix. mu_i is a per-matrix fixed effect, so only changes *within* a matrix are used.
- **j:** the ordering, including the original one. Symmetric and row reorderings are pooled.
- **d:** 16×16 block density, the same axis for every kernel.
- **f:** a natural cubic spline, with knots at the 5, 27.5, 50, 72.5 and 95% quantiles.

The two panels:
- **Left:** speed relative to the median-density ordering, `2^(f(d) - f(d_med))`. The speedup
  between two densities is the ratio of the curve at those points.
- **Right:** local elasticity `alpha(d) = df/dlog2 d`, i.e. the % speed change per 1% change in block
  density at that density.
- **Bands:** 95% bootstrap over matrices (200 resamples).
- **Range:** each curve spans the 2nd to 98th percentile of d.

Setup details:
- **Dense operand:** 256 columns, except **SMaT and ASpT, which always ran with 32 columns** (their
  `n_cols` argument is ignored). Their 32-column runs are used.
- **Excluded runs:** for cuSPARSE-BSR and SMaT, runs whose padded 32×32 work would exceed the
  hardware peak are dropped (4 and 25 runs respectively). These are likely silent skips, concentrated
  on a few large graph matrices.

Elasticity by density (`figures/elasticity_table.csv`):

| kernel | 0.5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| cuSPARSE-BSR | 1.38 | 1.09 | 1.05 | 1.00 | 0.94 |
| SMaT (32 cols) | 0.95 | 0.68 | 0.76 | 0.74 | 0.70 |
| FlashSparse | 0.39 | 0.20 | 0.20 | 0.48 | 0.63 |
| DTC-SpMM | 0.31 | 0.15 | 0.18 | 0.41 | 0.53 |
| Acc-SpMM | 0.48 | 0.13 | 0.19 | 0.38 | 0.47 |
| cuSPARSE-CSR | 0.67 | 0.27 | 0.20 | 0.14 | 0.10 |
| ASpT (32 cols) | 0.18 | −0.02 | 0.05 | 0.04 | 0.02 |

How to read it:
- **BSR kernels** respond at a nearly constant rate.
- **Tensor-core kernels** have a U-shape. Near the lowest possible density there is a locality effect,
  which cuSPARSE-CSR shares. Around 1–2% there is a flat trough. Above about 3% there is a
  tile-filling effect that only the tensor-core kernels show.
- **α above 1 for cuSPARSE-BSR at low density** is a real statement about 16×16 density. The kernel
  uses 32×32 blocks, whose count changes faster than the 16×16 count when nonzeros start clustering.
  Measured against 32×32 density, its α is about 0.9 throughout.

The bands are confidence bands for the *average* curve. Individual matrices scatter much more around
it: for the tensor-core kernels, predicting a held-out matrix's speedup from its density change is off
by about ±15% typically.
