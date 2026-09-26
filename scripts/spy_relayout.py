#!/usr/bin/env python3
"""Re-lay out a 4x3 spy grid (the output of spy_retitle.py) at print size.

Cuts the 12 framed panels out of SRC and redraws them in a ROWS x COLS grid
WIDTH inches wide, with titles at print size (the IEEE paper uses 3.49 in
for one column and 7.16 in for figure*). Title colours are taken from the
source titles; very light ones are darkened so they stay readable.

    python scripts/spy_relayout.py SRC DST ROWS COLS WIDTH
    python scripts/spy_relayout.py plots/spy_plots/wing_nodal_spy.png \
        plots/paper/wing_nodal_spy_4x3.png 3 4 3.49
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Reading order of the panels in the source grid; names as in the paper.
LABELS = ['Original', 'RCM', 'AMD', 'Rabbit',
          'GROOT', 'Gray', 'METIS', 'PaToH',
          'SlashBurn', 'DTC-LSH', 'Degree', 'Random']

src, dst = sys.argv[1], sys.argv[2]
rows, cols, width = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])

Image.MAX_IMAGE_PIXELS = None
im = np.array(Image.open(src).convert('RGB')).astype(int)
dark = im.sum(axis=2) < 150


def runs(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


xr = runs(dark.mean(axis=0) > 0.6)
yr = runs(dark.mean(axis=1) > 0.6)
assert len(xr) == 8 and len(yr) == 6, f'frame detection failed: {xr} {yr}'
xs = [(xr[2 * i][1] + 1, xr[2 * i + 1][0] - 1) for i in range(4)]
ys = [(yr[2 * i][1] + 1, yr[2 * i + 1][0] - 1) for i in range(3)]

panels, colors = [], []
for (y0, y1) in ys:
    for (x0, x1) in xs:
        panels.append(im[y0:y1 + 1, x0:x1 + 1].astype(np.uint8))
        strip = im[max(0, y0 - 150):y0 - 20, x0:x1]
        ink = strip[strip.sum(axis=2) < 690]
        c = np.median(ink, axis=0) / 255 if len(ink) else np.zeros(3)
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        if lum > 0.7:          # e.g. DTC-LSH's light grey
            c = c * 0.62 / lum
        colors.append(tuple(c))

gap_w, title_h = 0.04, 0.155    # inches
cell = (width - gap_w * (cols - 1)) / cols
height = rows * (cell + title_h)
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Times New Roman', 'DejaVu Serif']})
fig = plt.figure(figsize=(width, height))
for k, (panel, label, color) in enumerate(zip(panels, LABELS, colors)):
    r, c = divmod(k, cols)
    left = c * (cell + gap_w) / width
    bottom = 1 - ((r + 1) * (cell + title_h)) / height
    ax = fig.add_axes([left, bottom, cell / width, cell / height])
    ax.imshow(panel, interpolation='lanczos')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.set_title(label, fontsize=9, fontweight='bold', color=color, pad=1.5)
fig.savefig(dst, dpi=400, bbox_inches='tight', pad_inches=0.01)
print('saved', dst)
