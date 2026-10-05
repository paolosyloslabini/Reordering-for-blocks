#!/usr/bin/env bash
# Regenerate every figure used by the IPDPS paper from the committed results CSVs.
#
# Outputs land under plots/ (git-ignored); the paper repo's plots/MANIFEST lists
# which of them the paper uses, and its sync_plots.sh copies them over.
#
# Usage (from anywhere; Windows venv by default, see scripts/README.md):
#   bash scripts/make_paper_figures.sh
#   PY=python3 bash scripts/make_paper_figures.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"

# plots/paper/: print-size paper figures
"$PY" scripts/paper_figures.py
"$PY" scripts/gain_figures.py
"$PY" scripts/improvement_bars.py --n-cols=256
"$PY" scripts/breakeven_proposals.py          # also writes plots/paper/rcm_breakeven.pdf
"$PY" scripts/block_spy.py

# Spy grid of wing_nodal under every reordering (from the committed plots/spy_plots/wing_nodal_spy.png)
"$PY" scripts/spy_relayout.py plots/spy_plots/wing_nodal_spy.png plots/paper/wing_nodal_spy_4x3.png 3 4 3.49
"$PY" scripts/spy_relayout.py plots/spy_plots/wing_nodal_spy.png plots/paper/wing_nodal_spy_6x2.png 2 6 7.16
