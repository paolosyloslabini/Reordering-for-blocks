#!/usr/bin/env python3
"""What each SpMM kernel stores and computes on for the same sparse matrix.

One 64x64 example matrix, one panel per kernel. Cell colours:
  black        true nonzero
  orange       zero materialised in memory (dense block formats)
  light orange zero materialised only in the MMA fragment (compacted tile formats)
Grey lines delimit storage units. Kernels that compact columns inside a row window are drawn
in their compacted layout (nonzero columns packed to the left of each window).

Facts checked against the released sources of each kernel (Sept 2026):
  cuSPARSE CSR   nonzeros only; work decomposition undocumented
  cuSPARSE BSR   dense 32x32 blocks, zeros stored in memory
  SMaT           BSR with 16x16 blocks (mma.m16n8k16 A operand), one warp per block row
  ASpT           CSR with intra-row reordering; heavy columns (>=16 nnz per 128-row panel)
                 grouped into dense tiles for reuse; no zero is ever stored
  DTC-SpMM       ME-TCF: 16-row windows, columns compacted, 16x8 tiles, one thread block
                 per window; zeros exist only in the fragment
  FlashSparse    ME-BCRS: 8-row windows, 8x1 nonzero vectors, 8 vectors per 8x8 block;
                 zeros inside a kept vector are stored, trailing padding is fragment-only
  Acc-SpMM       BitTCF: 8-row windows, columns compacted, 8x8 tiles + 64-bit bitmask;
                 zeros exist only in the fragment

Usage: python scripts/kernel_views.py [out.png]   (default: plots/spy_plots/kernel_views.png)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

N = 64
rng = np.random.default_rng(7)
A = np.zeros((N, N), dtype=bool)
for i in range(N):                                   # loose band
    for j in range(max(0, i - 2), min(N, i + 3)):
        if rng.random() < 0.3:
            A[i, j] = True
for (r, c, h, w, p) in [(4, 40, 10, 10, 0.6), (36, 8, 12, 9, 0.5), (48, 44, 8, 14, 0.55)]:
    A[r:r + h, c:c + w] |= rng.random((h, w)) < p    # three clusters
A[rng.random(N) < 0.3, 20] = True                    # one hub column
A |= rng.random((N, N)) < 0.004                      # scattered noise
np.fill_diagonal(A, True)
nnz = int(A.sum())

NZ, MEM, FRAG, GRID, WORK = "#111111", "#f4a261", "#f9d3ac", "#9a9a9a", "#1d4e89"
FS_TITLE, FS_LABEL, FS_SMALL = 26, 21, 18

EMPTY, NONZERO, ZMEM, ZFRAG = 0, 1, 2, 3


def blocked(A, h, w, zero_kind=ZMEM):
    """Fixed grid of h x w tiles; every tile with a nonzero is fully materialised."""
    S = np.zeros((N, N), dtype=int)
    for r in range(0, N, h):
        for c in range(0, N, w):
            if A[r:r + h, c:c + w].any():
                S[r:r + h, c:c + w] = zero_kind
    S[A] = NONZERO
    return S, None


def compacted(A, win, tile_w, kept_zero=ZFRAG, pad_zero=ZFRAG):
    """Row windows of height `win`; nonzero columns packed left and cut into tiles
    of width `tile_w`. Zeros inside a kept column get `kept_zero`, the columns that
    pad the last partial tile get `pad_zero`."""
    S = np.zeros((N, N), dtype=int)
    widths = []
    for r in range(0, N, win):
        W = A[r:r + win]
        cols = np.flatnonzero(W.any(0))
        k = len(cols)
        P = W[:, cols]
        S[r:r + win, :k] = np.where(P, NONZERO, kept_zero)
        k_pad = -(-k // tile_w) * tile_w
        S[r:r + win, k:k_pad] = pad_zero
        widths.append(k_pad)
    return S, (tile_w, widths)


def aspt(A, panel, thr):
    """CSR with intra-row reordering: heavy columns (>= thr nnz in the panel) first.
    Nothing is padded. Returns the layout plus the width of each panel's dense tile."""
    S = np.zeros((N, N), dtype=int)
    widths = []
    for r in range(0, N, panel):
        P = A[r:r + panel]
        cnt = P.sum(0)
        heavy = np.flatnonzero(cnt >= thr)
        light = np.flatnonzero(cnt < thr)
        order = np.concatenate([heavy, light])
        S[r:r + panel] = np.where(P[:, order], NONZERO, EMPTY)
        widths.append(len(heavy))
    return S, widths


def draw(ax, S, h=None, w=None, title="", unit="", work=None, work_label="", packed=None):
    img = np.ones((N, N, 3))
    img[S == ZMEM] = matplotlib.colors.to_rgb(MEM)
    img[S == ZFRAG] = matplotlib.colors.to_rgb(FRAG)
    img[S == NONZERO] = matplotlib.colors.to_rgb(NZ)
    ax.imshow(img, interpolation="nearest")
    if h:
        for r in range(0, N + 1, h):
            ax.axhline(r - 0.5, color=GRID, lw=0.6)
    if w and packed is None:
        for c in range(0, N + 1, w):
            ax.axvline(c - 0.5, color=GRID, lw=0.6)
    if packed is not None:  # vertical tile lines only inside each window's packed region
        for i, r in enumerate(range(0, N, h)):
            for c in range(0, packed[i] + 1, w):
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + h - 0.5], color=GRID, lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#444444")
    processed = int((S != EMPTY).sum())
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", pad=6)
    ax.set_xlabel(f"{unit}\nnnz / processed = {nnz / processed:.2f}",
                  fontsize=FS_LABEL, labelpad=6, linespacing=1.15)
    if work:
        x = -2.5
        ax.plot([x, x], [-0.5, work - 0.5], color=WORK, lw=3, clip_on=False)
        ax.plot([x, x + 1.5], [-0.5, -0.5], color=WORK, lw=3, clip_on=False)
        ax.plot([x, x + 1.5], [work - 0.5, work - 0.5], color=WORK, lw=3, clip_on=False)
        ax.text(x - 1.2, (work - 1) / 2, work_label, color=WORK, fontsize=FS_SMALL,
                ha="right", va="center", clip_on=False, linespacing=0.95)


fig, axes = plt.subplots(3, 3, figsize=(17, 13.8), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.3, h_pad=0.2, wspace=0.04, hspace=0.02)
ax = axes.ravel()

S, _ = blocked(A, 1, N, ZMEM); S[S == ZMEM] = EMPTY
draw(ax[0], S, h=1, title="cuSPARSE CSR", unit="CSR, nonzeros only")

S, _ = blocked(A, 32, 32)
draw(ax[1], S, 32, 32, "cuSPARSE BSR", unit=r"dense $32\times32$ blocks")

S, _ = blocked(A, 16, 16)
draw(ax[2], S, 16, 16, "SMaT", unit=r"dense $16\times16$ blocks")

PANEL, THR = 32, 4   # real kernel: 128-row panels, >= 16 nnz; same ratio here
S, widths = aspt(A, PANEL, THR)
draw(ax[3], S, h=PANEL, title="ASpT", unit="CSR, heavy cols first")
for i, r in enumerate(range(0, N, PANEL)):
    ax[3].add_patch(Rectangle((-0.5, r - 0.5), widths[i], PANEL, fill=False,
                              ec=WORK, lw=1.8, ls="--"))

S, tw = compacted(A, 16, 8)
draw(ax[4], S, 16, tw[0], "DTC-SpMM", unit=r"$16\times8$ tiles",
     packed=tw[1])

S, tw = compacted(A, 8, 8, kept_zero=ZMEM, pad_zero=ZFRAG)
draw(ax[5], S, 8, tw[0], "FlashSparse", unit=r"$8\times1$ vectors",
     packed=tw[1])

S, tw = compacted(A, 8, 8)
draw(ax[6], S, 8, tw[0], "Acc-SpMM", unit=r"$8\times8$ tiles + bitmask",
     packed=tw[1])

# legend in the two empty cells of the last row
ax[7].axis("off"); ax[8].axis("off")
handles = [
    Patch(color=NZ, label=f"nonzero ({nnz} in total)"),
    Patch(color=MEM, label="zero stored in memory"),
    Patch(color=FRAG, label="zero in MMA fragment only"),
    Patch(facecolor="white", edgecolor=GRID, label="storage unit"),
    Patch(facecolor="white", edgecolor=WORK, lw=1.8, ls="--", label="ASpT dense tile"),
]
leg = ax[7].legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.55), fontsize=FS_LABEL,
                   frameon=False, handlelength=1.6, labelspacing=1.0, borderaxespad=0)
leg.set_in_layout(False)   # keep the legend from widening the middle column
ax[8].text(0.5, 0.55, "Windowed formats\n(DTC-SpMM, FlashSparse, Acc-SpMM)\nare drawn compacted: nonzero\ncolumns packed left in each window.",
           transform=ax[8].transAxes, ha="center", va="center", fontsize=FS_SMALL, color="#444444", linespacing=1.4)

out = sys.argv[1] if len(sys.argv) > 1 else "plots/spy_plots/kernel_views.png"
fig.savefig(out, dpi=200)
print("wrote", out, "nnz", nnz)
