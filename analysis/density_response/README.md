# Density response: how reordering changes block density, and how kernels respond

Two questions, answered from the existing results CSVs (no new GPU runs):

1. **Given a matrix, what block-density gain will a reordering give?**
   See `gain_vs_start_block_density` and `gain_vs_start_density`.
2. **Given a block-density change, what speedup does each kernel give?**
   See `elasticity_block_density_original_nc*`.

All figures use **original SuiteSparse matrices only**. Scrambled matrices are left out
because they hide a good ordering, so reordering recovers far more from them than from real
inputs. The matrix set and filters are the paper's (`scripts/filter_config.yaml`, 389 matrices).

```
python analysis/density_response/gain_figures.py                      # figures 1-2 (~1 min)
python analysis/density_response/elasticity_figure.py --n-cols=256    # figures 3-5 (~10 min, bootstrap)
python analysis/density_response/decision_test.py --n-cols=256        # decision table (~5 min)
python analysis/density_response/strategy_profiles.py --n-cols=256    # strategy profiles (~2 min)
# same three with --n-cols=32
```

## Dense-operand width: `_nc32` and `_nc256`

Every output that depends on the kernel runs comes in two versions:
- **`_nc32`:** all kernels at 32 columns.
- **`_nc256`:** five kernels at 256 columns, with **SMaT and ASpT at 32**, labelled "(32 cols)" in the
  figures. Those two kernels ignore the width argument, so their "256" runs are really 32-column runs.
  (The ASpT investigation on `analysis/aspt-invariance` suggests SMaT may have used 8 columns; this
  still has to be checked in the cluster logs.)

The gain figures (1-2) depend only on structure, not on kernel runs, so they have no width suffix.
**All tables below are for `_nc256`.** The 32-column values are in the matching `_nc32` CSVs, and the
main differences are summarised at the end.

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

## Figure 3: elasticity of speed to block density (`elasticity_block_density_original_nc*`)

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
- **Dense operand:** see the width section above.
- **Excluded runs:** for cuSPARSE-BSR and SMaT, runs whose padded 32×32 work would exceed the
  hardware peak are dropped (4 and 25 runs respectively). These are likely silent skips, concentrated
  on a few large graph matrices.

Elasticity by density (`figures/elasticity_table_nc*.csv`):

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

## Figure 4: elasticity at varying matrix density (`elasticity_vs_matrix_density_nc*`)

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

## Figure 5: local correlation with block density (`correlation_block_density_original_nc*`)

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
**how reliably** block density predicts speed there. Values from `figures/correlation_table_nc*.csv`:

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

## Decision test: which reordering to pick (`decision_test.py`, `figures/decision_table_nc*.csv`)

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

Same widths and exclusions as figure 3: see the width section, with the impossible BSR/SMaT runs
dropped.

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

## Strategy profiles (`strategy_profiles_nc*`, `figures/strategy_profiles_summary_nc*.csv`)

These are performance profiles on the original matrices.
- **Best available speedup:** for each (matrix, kernel), the maximum over the original ordering and
  every reordering that ran, symmetric or row.
- **Curve:** the share of matrices on which a strategy comes within a factor x of that best. Higher and
  further left is better.

The figure has one plot, with **colour = kernel** and **line style = strategy**:
- **Densest (solid):** the candidate with the highest 16×16 block density, or the original ordering if
  nothing is denser.
- **Densest, d < 10% (dashed):** the same, but only if the starting block density is below 10%;
  otherwise keep the original ordering.
- **Never reorder (dotted):** the original ordering.

If a strategy's run is missing, it keeps the original ordering. Widths and exclusions are as in the
decision test. The CSV also covers always RCM, AMD and Rabbit, with and without the cut-off.

Geometric-mean speedup, with the share of matrices within 10% of the best in brackets:

| kernel | Densest | Densest, d < 10% | Never reorder | best fixed choice |
|---|---|---|---|---|
| cuSPARSE-BSR | **1.93 (84%)** | 1.90 (79%) | 1.00 (23%) | RCM d<10%: 1.66 (47%) |
| SMaT (32 cols) | 1.43 (49%) | 1.42 (47%) | 1.00 (24%) | **RCM d<10%: 1.53 (57%)** |
| DTC-SpMM | 1.08 (55%) | 1.08 (53%) | 1.00 (27%) | **Rabbit d<10%: 1.16 (63%)** |
| FlashSparse | 1.13 (52%) | 1.12 (48%) | 1.00 (23%) | **Rabbit d<10%: 1.14 (48%)** |
| Acc-SpMM | 1.06 (50%) | 1.06 (50%) | 1.00 (32%) | **Rabbit d<10%: 1.10 (53%)** |
| cuSPARSE-CSR | **1.14 (74%)** | 1.14 (75%) | 1.00 (41%) | RCM d<10%: 1.12 (69%) |
| ASpT (32 cols) | 1.01 (54%) | 1.01 (54%) | 1.00 (41%) | ≈1.00 |

What it shows:
- **Picking the densest candidate is far better than not reordering** for every kernel except ASpT.
  Doing nothing is within 10% of the best for only 23–41% of matrices.
- **The 10% cut-off barely matters for the densest rule** (the solid and dashed lines nearly coincide).
  The rule already keeps the original ordering when nothing is denser. Skipping reordering above 10%
  only gives up a few wins (cuSPARSE-BSR 84% → 79% within 10% of the best).
- **Densest is the best strategy for the BSR and CSR cuSPARSE kernels.** For SMaT and the tensor-core
  kernels, a fixed choice does slightly better: RCM for SMaT, Rabbit for the tensor-core kernels, both
  with the cut-off. The densest candidate is not always their fastest.

The "best" reference is the maximum of noisy timings, so every strategy looks somewhat further from it
than it really is.

### Expected speedup by starting block density (`strategy_speedup_vs_start_density_nc*`)

This shows the speedup actually achieved by the **densest** strategy, against the matrix's starting
16×16 block density. Each line is a Gaussian-kernel-weighted geometric mean over matrices (bandwidth
0.15 decades on log10 d), one line per kernel. Bands are 95% bootstrap over matrices: uncertainty of
the average, not the spread of single matrices.

The "densest, only below 10%" strategy is identical to this line left of the dashed marker and exactly
1× right of it, so it is not drawn separately. Values are in `figures/densest_speedup_vs_start_density_nc*.csv`:

| kernel | 0.5% | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| cuSPARSE-BSR | 3.92 | 3.03 | 1.65 | 1.25 | 1.18 | 1.09 |
| SMaT (32 cols) | 1.88 | 1.80 | 1.46 | 1.11 | 1.11 | 1.02 |
| FlashSparse | 1.19 | 1.22 | 1.12 | 1.07 | 1.07 | 1.04 |
| DTC-SpMM | 1.06 | 1.15 | 1.11 | 1.08 | 1.08 | 1.01 |
| Acc-SpMM | 1.09 | 1.11 | 1.06 | 1.03 | 1.05 | 1.01 |
| cuSPARSE-CSR | 1.36 | 1.26 | 1.04 | 1.00 | 1.02 | 1.02 |
| ASpT (32 cols) | 1.02 | 1.01 | 1.01 | 0.99 | 1.00 | 1.00 |

What it shows:
- **The expected gain falls steeply with starting block density** for the BSR kernels (cuSPARSE-BSR
  about 4× at 0.5% but about 1.2× at 10%) and for cuSPARSE-CSR (1.36× at 0.5%, none above 2%).
- **The tensor-core kernels gain about 1.1–1.2× below 2% and about 1.05× above.**
- **The densest rule does not lose on average above 10%.** Because it keeps the original ordering when
  nothing is denser, no curve drops below 1×. So the 10% cut-off gives up a little (cuSPARSE-BSR
  1.18× → 1× at 10%) rather than protecting against losses.
- These are averages. Individual matrices scatter widely around them (see the profiles above).

## What changes at 32 columns (`_nc32`)

With a narrow B, a larger part of each kernel's time is overhead that does not depend on B's width,
and reordering does not reduce it. So block density matters less, above all for the tensor-core kernels.
cuSPARSE-BSR, SMaT and ASpT barely change (SMaT and ASpT are at 32 columns in both versions).

- **Elasticity:** the tensor-core kernels lose most of their low-density sensitivity (DTC-SpMM 0.10
  vs 0.31 at 0.5%; FlashSparse 0.20 vs 0.39). Their rise at high density is a little lower (DTC-SpMM 0.44
  vs 0.53 at 10%). cuSPARSE-BSR is at about 0.85 at 10%, against 0.94.
- **Local correlation:** the tensor-core kernels start near zero (0.06–0.14 at 0.5%) and reach
  0.30–0.54 at 10%.
- **Decision test (original matrices):**
  - *Densest:* gains shrink for the tensor-core kernels (DTC-SpMM 1.035×, FlashSparse 1.059×,
    Acc-SpMM 1.024×) and for cuSPARSE-BSR (1.46×).
  - *Density + reuse:* this model now clearly beats densest for DTC-SpMM (1.147×) and FlashSparse
    (1.163×). Locality matters more when B is narrow.
- **Strategy profiles:** Rabbit with the 10% cut-off is the best simple strategy for the tensor-core
  kernels (1.07–1.14×). Densest is only 1.02–1.05× for them.
- **Densest speedup against starting block density:** at 32 columns, the densest reordering *slows
  down* the tensor-core kernels on very scattered matrices (about 0.90–0.98× at 0.5%). Gains appear only
  between about 1% and 10% (1.04–1.12×).

## How often reordering helps, and by how much (`improvement_bars_nc32`, `figures/improvement_bars_nc32.csv`)

Four cells: {original, scrambled} × {symmetric, row}, with **all kernels at 32 columns**. In each cell
the strategy takes the densest candidate among that cell's reorderings (or keeps the original ordering
if none is denser). Speedups are relative to the cell's own starting matrix, which for scrambled matrices
is the scrambled matrix.

- **Bottom bars:** share of matrices where the chosen reordering made the kernel faster (green) or
  slower (red). Grey means no candidate was denser, so the original ordering was kept.
- **Top:** geometric-mean speedup among the improved matrices, with a 95% bootstrap interval over
  matrices.

**Original matrices, symmetric reordering:**

| kernel | faster | slower | gain when faster |
|---|---|---|---|
| cuSPARSE-BSR | 67% | 12% | 2.43× |
| SMaT | 60% | 18% | 2.05× |
| cuSPARSE-CSR | 70% | 10% | 1.27× |
| DTC-SpMM, FlashSparse | 48% | 31–35% | 1.32–1.33× |
| Acc-SpMM | 48% | 31% | 1.23× |
| ASpT | 49% | 32% | 1.10× |

- **ASpT is the noise reference.** It barely reacts to reordering, yet about half of the matrices come
  out "faster", by about 1.1×.
- **cuSPARSE-BSR, SMaT and cuSPARSE-CSR** are faster clearly more often than that, and the BSR
  kernels by a lot.
- **The tensor-core kernels** are faster no more often than ASpT, but when they are faster, it is by
  1.2–1.3×, well above noise.
- **Row reordering on original matrices** keeps the original ordering more often (about a third of
  matrices), with smaller gains (1.2–1.5×).
- **Scrambled matrices** are improved in 62–97% of cases. Gains reach 6.5× (cuSPARSE-BSR, symmetric)
  and 1.6–1.9× for the tensor-core kernels.

### Same data as distributions (`improvement_strips_nc32`)

This is a sina plot of the per-matrix speedups behind the bar figure: one dot per matrix, with each
strip as wide as the local density of dots. Paper style, N = 32.

- **Dots:** green = faster, red = slower, grey on the 1× line = kept original.
- **Black tick:** geometric-mean speedup among faster matrices, with its 95% bootstrap interval.
- **Numbers on top:** share of matrices faster (↑) and slower (↓).
- **Clipping:** dots beyond the axis range (1/6× to 12× for original, 1/6× to 48× for scrambled) are
  drawn at the limit.

It carries everything the bars do, plus the spread. For example, the tensor-core kernels on original
matrices show a roughly symmetric cloud around 1× with a long tail of large wins, while cuSPARSE-BSR is
mostly green and far above 1×.
