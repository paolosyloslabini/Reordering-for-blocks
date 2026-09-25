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
python analysis/density_response/elasticity_figure.py   # figures 3-5 (~10 min, bootstrap)
python analysis/density_response/decision_test.py       # decision table (~5 min)
python analysis/density_response/strategy_profiles.py   # strategy profiles (~2 min)
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

## Figure 3: elasticity of speed to block density (`elasticity_block_density_original`)

A per-kernel fixed-effects spline model:

`log2 GFLOPS_ij = mu_i + f(log2 d_ij) + e_ij`

- **i:** the matrix. mu_i is a per-matrix fixed effect, so only changes *within* a matrix are used.
- **j:** the ordering, including the original one. Symmetric and row reorderings are pooled.
- **d:** 16×16 block density, the same axis for every kernel.
- **f:** a natural cubic spline, with knots at the 5, 27.5, 50, 72.5 and 95% quantiles.

The figure plots the local elasticity `alpha(d) = df/dlog2 d`: the % change in speed per 1% change in
block density, at that density.
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

## Figure 4: elasticity at varying matrix density (`elasticity_vs_matrix_density`)

This asks whether a kernel's sensitivity to block density depends on how sparse the matrix is overall.
Matrix density nnz/(m·n) is fixed for a given matrix, so there is one within-matrix slope per window
of matrix density, not a spline.

- **Estimate:** `alpha = Σ_i w_i Σ_j xw·yw / Σ_i w_i Σ_j xw²`, where xw and yw are log2 block density
  and log2 GFLOPS, demeaned per matrix.
- **Weights:** Gaussian w_i on log10 matrix density, bandwidth 0.25 decades.
- **Bands:** 95% bootstrap over matrices.
- **Data:** the same as figure 3.

What it shows:
- **BSR kernels** are most sensitive on very sparse matrices. cuSPARSE-BSR drops from about 1.15 to
  about 0.8, and SMaT from about 0.85 to about 0.6, as matrix density rises from 10⁻⁵ to 10⁻².
- **The tensor-core kernels** sit at about 0.2–0.35 throughout, rising slightly (to about 0.35–0.45)
  for the densest matrices.
- **cuSPARSE-CSR** is sensitive only for the sparsest matrices (about 0.6 at 2·10⁻⁶), falling to about
  0.1–0.2. This fits a locality effect that matters most when rows are very short.
- **ASpT** stays near zero everywhere.

Compared with figure 3, matrix density separates the kernels less than block density does. What
matters is the ordering-dependent block density, not overall sparsity.

## Figure 5: local correlation with block density (`correlation_block_density_original`)

This is the correlation counterpart of figure 3, on the same data. At each block density d it gives the
**within-matrix Pearson r** between log2 GFLOPS and log2 16×16 block density.

- **Weights:** configurations are weighted with a Gaussian on log2 d (bandwidth 0.75, about a quarter
  of a decade).
- **Within-matrix:** x and y are demeaned with per-matrix weighted means, so only changes between
  orderings of the same matrix count.
- **Correlation:** r = Sxy / √(Sxx·Syy).
- **Bands:** 95% bootstrap over matrices.
- **Where curves are drawn:** only where the window's effective sample size is at least 50.
- **Bandwidth:** the same for every kernel. A narrower window leaves less spread in d and lowers r, so
  compare kernels with each other, not with the global r values in the paper.

α (figure 3) says **how much** speed changes per 1% of block density at that density. r says
**how reliably** block density predicts speed there. Values from `figures/correlation_table.csv`:

| kernel | 0.5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|
| cuSPARSE-BSR | 0.94 | 0.94 | 0.94 | 0.90 | 0.88 |
| SMaT (32 cols) | 0.58 | 0.67 | 0.72 | 0.66 | 0.66 |
| FlashSparse | 0.38 | 0.41 | 0.43 | 0.53 | 0.62 |
| DTC-SpMM | 0.26 | 0.31 | 0.34 | 0.47 | 0.57 |
| Acc-SpMM | 0.36 | 0.35 | 0.34 | 0.43 | 0.44 |
| cuSPARSE-CSR | 0.73 | 0.74 | 0.62 | 0.36 | 0.21 |
| ASpT (32 cols) | 0.20 | 0.14 | 0.11 | 0.10 | 0.07 |

How to read it:
- **cuSPARSE-BSR:** block density is a near-perfect predictor everywhere.
- **Tensor-core kernels:** the correlation rises with density, from about 0.3–0.4 to about 0.45–0.6.
  Unlike α, r has no trough at 1–2%. In that range, density changes still predict the *direction* of
  speed changes about as well, but the changes themselves are small.
- **cuSPARSE-CSR:** block density is a good predictor only while the matrix is scattered (r about 0.73
  below 1%) and a poor one once it is well blocked (about 0.2 at 10%). This fits block density acting as
  a locality proxy for CSR.
- **ASpT:** block density predicts nothing useful at any density.

## Decision test: which reordering to pick (`decision_test.py`, `figures/decision_table.csv`)

This tests how well simple rules choose a reordering. The unit is each (matrix, permutation type,
kernel); the candidates are the original ordering and every reordering that ran.

The rules:
- **best:** the fastest candidate (an upper bound).
- **max_density:** the candidate with the highest 16×16 block density, keeping the original ordering if
  nothing raises it.
- **density_reuse:** the highest predicted speedup from a model of the change in block density (a
  spline) plus the change in mean reuse distance. It is fitted per kernel with 5-fold CV grouped by
  matrix, so every prediction is for held-out matrices.
- **always_RCM**, **always_AMD**, and **never** (keep the original ordering).

A rule based only on the density curve is not listed: the curves rise monotonically, so it always makes
the same choice as max_density.

Same widths and exclusions as figure 3: 256 columns, SMaT and ASpT at their actual 32 columns, and the
impossible BSR/SMaT runs dropped.

**Original matrices, geometric-mean speedup (share of cases within 90% of the best):**

| kernel | best | max_density | density_reuse | always RCM | always AMD |
|---|---|---|---|---|---|
| cuSPARSE-BSR | 1.61 | 1.55 (90%) | 1.56 (90%) | 1.15 (31%) | 1.18 (36%) |
| SMaT (32 cols) | 1.52 | 1.25 (61%) | 1.27 (61%) | 1.13 (40%) | 0.91 (9%) |
| FlashSparse | 1.30 | 1.12 (59%) | 1.16 (67%) | 1.07 (32%) | 1.05 (38%) |
| DTC-SpMM | 1.28 | 1.09 (60%) | 1.14 (65%) | 1.08 (43%) | 1.04 (44%) |
| Acc-SpMM | 1.25 | 1.07 (62%) | 1.09 (63%) | 1.04 (50%) | 1.03 (42%) |
| cuSPARSE-CSR | 1.17 | 1.11 (85%) | 1.10 (85%) | 1.07 (68%) | 1.08 (75%) |
| ASpT (32 cols) | 1.12 | 1.01 (62%) | 1.01 (61%) | 0.99 (56%) | 0.99 (51%) |

Scrambled matrices show the same ordering of rules, at larger scale (e.g. FlashSparse: best 1.85,
max_density 1.65, density_reuse 1.70, always RCM 1.60). See the CSV.

What it shows:
- **"Reorder for block density" works as a rule.** Picking the densest candidate beats always using
  RCM or AMD for every kernel except ASpT.
- **For the BSR kernels and cuSPARSE-CSR,** it gets close to the best: 85–90% of cases are within 90%.
- **For the tensor-core kernels,** adding reuse distance improves the choice (DTC-SpMM 1.09 → 1.14,
  FlashSparse 1.12 → 1.16), consistent with locality carrying information that block density does not.
- **For ASpT,** no rule helps.

Caveat: **best** takes the maximum over about 11 noisy timings, each a mean of 5 runs (ASpT's are single
cold runs). So it partly selects lucky measurements, especially on original matrices, where true
differences are a few percent. The real gap between the rules and the best choice is smaller than the
table shows.

## Strategy profiles (`strategy_profiles`, `figures/strategy_profiles_summary.csv`)

These are per-kernel performance profiles of simple strategies on the original matrices.
- **Best available speedup:** for each (matrix, kernel), the maximum over the original ordering and
  every reordering that ran, symmetric or row.
- **Curve:** the share of matrices on which a strategy comes within a factor x of that best. Higher and
  further left is better.

The strategies:
- **Highest block density:** the densest candidate after running them all, or the original ordering if
  nothing is denser.
- **Always RCM, AMD or Rabbit:** symmetric reordering.
- **Never reorder:** keep the original ordering. This is the grey dotted reference line.
- **The same three, dashed, "d < 10%":** reorder only if the starting 16×16 block density is below 10%,
  otherwise keep the original ordering.

If a strategy's run is missing (the reordering or the kernel failed), it keeps the original ordering.
Widths and exclusions are as in the decision test. The x-axis runs to 4× for the BSR kernels and to 2×
for the others.

Geometric-mean speedup achieved, with the share of matrices within 10% of the best in brackets:

| kernel | Highest density | RCM | RCM d<10% | AMD | AMD d<10% | Rabbit | Rabbit d<10% | Never reorder |
|---|---|---|---|---|---|---|---|---|
| cuSPARSE-BSR | **1.93 (84%)** | 1.64 (45%) | 1.66 (47%) | 1.47 (33%) | 1.54 (40%) | 1.14 (16%) | 1.32 (24%) | 1.00 (23%) |
| SMaT (32 cols) | 1.43 (49%) | 1.51 (56%) | **1.53 (57%)** | 1.03 (7%) | 1.09 (14%) | 1.08 (21%) | 1.22 (30%) | 1.00 (24%) |
| DTC-SpMM | 1.08 (55%) | 1.09 (37%) | 1.09 (36%) | 1.05 (42%) | 1.06 (44%) | 1.11 (56%) | **1.16 (63%)** | 1.00 (27%) |
| FlashSparse | 1.13 (52%) | 1.05 (22%) | 1.06 (23%) | 1.04 (30%) | 1.05 (33%) | 1.08 (41%) | **1.14 (48%)** | 1.00 (23%) |
| Acc-SpMM | 1.06 (50%) | 1.04 (40%) | 1.04 (42%) | 1.03 (35%) | 1.04 (40%) | 1.06 (44%) | **1.10 (53%)** | 1.00 (32%) |
| cuSPARSE-CSR | **1.14 (74%)** | 1.12 (68%) | 1.12 (69%) | 1.12 (68%) | 1.12 (70%) | 1.09 (52%) | 1.10 (57%) | 1.00 (41%) |
| ASpT (32 cols) | 1.01 (54%) | 0.98 (49%) | 0.99 (50%) | 1.00 (47%) | 1.00 (48%) | 1.00 (49%) | 1.00 (50%) | 1.00 (41%) |

What it shows:
- **The 10% cut-off never hurts.** It helps most where a reordering damages well-blocked matrices:
  Rabbit on every kernel (cuSPARSE-BSR 1.14 → 1.32, SMaT 1.08 → 1.22), and AMD on the BSR kernels.
- **The best strategy depends on the kernel family:**
  - *cuSPARSE-BSR and cuSPARSE-CSR:* highest block density.
  - *SMaT:* RCM. Highest density is close behind, but for SMaT the densest candidate is not always
    the fastest.
  - *Tensor-core kernels (DTC-SpMM, FlashSparse, Acc-SpMM):* Rabbit with the cut-off. It edges out
    highest density, consistent with locality mattering beyond block density for these kernels.
- **ASpT:** no strategy matters.
- **Doing nothing** is within 10% of the best for only 23–41% of matrices. Always-Rabbit on the BSR
  kernels is *worse* than doing nothing near the best, which is why the 10% cut-off matters.

The "best" reference carries the same caveat as the decision test: it is the maximum of noisy timings,
so every strategy looks somewhat further from it than it really is.
