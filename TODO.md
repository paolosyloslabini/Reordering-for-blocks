# ReorderingSurvey-2026


Think again on binned speedups plots.

For each reorder, how many matrices get improved, and how many damaged? in terms of speedup, block density, etc.

Random scrambling experiments (IN PROGRESS)

Reordering-time analysis and break-even analysis. 

Re-run GROOT

Re-run SPARTA

run geometric mean.

per-matrix study: are the same matrices easy/hard? calculate avg improvement over all reordering (geomean) + variance of the logs. plot avg-variance scatterplot.


Note on the breakeven analysis:
    - ignoring changes in SpMM preprocessing due to reordering. This may affect breakeven calculations slightly. FlashSparse, DTC: Preprocessing timers are genuine and correctly reported. The breakeven formula should account for the preprocessing delta as discussed earlier.
            - cuSPARSE BSR: The CSR→BSR conversion is completely invisible — it's not in any timer, yet it's a one-time, reorder-sensitive cost. This is the most critical gap.
            - SMaT, ASpT: Their external binaries hide internal preprocessing. You can't currently correct the breakeven for these without modifying the binaries.

    - Ignoring cost of applying permutation. 

# STYLE

per-column bold (instead of per-row) in table: Median structural improvement ratio per reordering algorithm

improve titles to look good in paper (too small now)

review tables captions.

~~better looking plots (especially nicer axes in log plots)~~

Axes span equal for similar plots