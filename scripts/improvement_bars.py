"""How often the densest reordering helps, and by how much when it does.

Two figures from the same data: improvement_bars_nc<N> (shares + gain among
improved) and improvement_strips_nc<N> (the full per-matrix distribution as a
sina plot, with the same shares and gain marked). improvement_strips_nc<N>_symmetric
(original above scrambled matrices, one column wide) is the version used in the paper;
improvement_strips_nc<N>_original_symmetric is its earlier one-panel form.

Four cells: {original, scrambled} matrices x {symmetric, row} reordering.
For every (matrix, kernel) the candidates are that cell's reorderings; the
strategy keeps the candidate with the highest 16x16 block density, or the
original ordering if none is denser. Speedups are relative to the cell's own
starting matrix (for scrambled matrices: the scrambled matrix).

Bottom of each cell: share of matrices where the chosen reordering was faster
(improved) or slower; the rest kept the original ordering (no denser candidate).
Top of each cell: geometric-mean speedup among improved matrices, with a 95%
bootstrap interval over matrices.

For cuSPARSE-BSR and SMaT, runs whose padded 32x32 block work would exceed the
hardware peak are dropped.

Dense operand: --n-cols=N (default 256, as in the paper).
Output: plots/paper/improvement_{bars,strips}_nc<N>*.{pdf,png} and improvement_bars_nc<N>.csv
Run from anywhere:  python scripts/improvement_bars.py --n-cols=256
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

from figure_common import (COL_W, KERNEL_NAMES, OUT, PAGE_W, load_pipeline,
                           n_cols_from_argv, save, style)

N_COLS = n_cols_from_argv(default=256)
# Hardware peak (TFLOPS) of the padded block work, used to drop impossible runs
PEAK_TFLOPS = {'CUSPARSE_SPMM_BSR_bs32': 19.5, 'SMAT_SPMM_bs32': 312.0}
# Paper palette (scripts/paper_figures.py): improved, degraded, neutral.
GOOD, BAD, NEUTRAL = '#006400', '#8B0000', '#A0A0A0'
KEPT = '#000000'    # strips figure: kept-original dots sit on 1x and must stay visible
GAIN = '#6A3D9A'    # strips figure: crossbar for the gain among faster matrices
# ASpT is left out of the paper (it barely reacts to reordering)
KERNEL_NAMES = {k: v for k, v in KERNEL_NAMES.items() if k != 'ASPT_SPMM'}
KERNEL_ORDER = sorted(KERNEL_NAMES.values(), key=str.lower)   # as in the paper figures
CELLS = [('original', 'SYMMETRIC'), ('original', 'ROW'),
         ('scrambled', 'SYMMETRIC'), ('scrambled', 'ROW')]
TITLES = {'original': 'Original matrices', 'scrambled': 'Scrambled matrices',
          'SYMMETRIC': 'symmetric reordering', 'ROW': 'row reordering'}


def densest_speedups(dataset, perm_type):
    df = load_pipeline(dataset, perm_type)
    df = df[(df['n_cols'] == N_COLS) & (df['strategy'] != 'Original')
            & (df['speedup'] > 0) & np.isfinite(df['speedup'])
            & (df['block_density_16'] > 0) & (df['block_density_16_original'] > 0)]
    for k, peak in PEAK_TFLOPS.items():
        executed = (2 * df['nonzero_blocks_32'] * 1024 * df['n_cols']
                    / (df['time_operation_ms'] / 1e3) / 1e12)
        df = df[~((df['kernel_id'] == k) & (executed > peak))]
    rows = []
    for (k, m), g in df.groupby(['kernel_id', 'matrix']):
        best = g.loc[g['block_density_16'].idxmax()]
        denser = best['block_density_16'] > g['block_density_16_original'].iloc[0]
        rows.append({'kernel_id': k, 'matrix': m,
                     'speedup': best['speedup'] if denser else 1.0})
    return pd.DataFrame(rows)


def summarise(d, rng, n_boot=5000):
    out = []
    for k in KERNEL_NAMES:
        s = d.loc[d['kernel_id'] == k, 'speedup'].values
        imp = s > 1
        gain = np.exp(np.log(s[imp]).mean())
        boot = []
        for _ in range(n_boot):
            b = s[rng.integers(0, len(s), len(s))]
            boot.append(np.exp(np.log(b[b > 1]).mean()))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append({'kernel': KERNEL_NAMES[k], 'matrices': len(s),
                    'improved': imp.mean(), 'slower': (s < 1).mean(),
                    'unchanged': (s == 1).mean(),
                    'gain_improved': gain, 'gain_lo': lo, 'gain_hi': hi})
    return pd.DataFrame(out)


def ratio_axis(ax, lo, hi):
    """Paper-style ratio axis: log scale, 1x/2x/5x ticks, red dashed 1x line."""
    import matplotlib as mpl
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    majors = [m for m in (1, 1.5, 2, 3, 5, 8) if lo <= m <= hi]
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(majors))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhline(1, color='#CC0000', linestyle='--', linewidth=0.8, alpha=0.8, zorder=2)
    ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)


def kernel_separators(ax, n):
    for i in range(n - 1):
        ax.axvline(i + 0.5, color='#c8c8c8', linewidth=0.5, zorder=1)
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(False, axis='x', which='both')
    ax.tick_params(axis='x', which='both', length=0)
    ax.set_axisbelow(True)


def figure(results):
    names = KERNEL_ORDER
    x = np.arange(len(names))
    ylims = {'original': (0.88, 3.2), 'scrambled': (0.88, 9.0)}
    fig = plt.figure(figsize=(PAGE_W, 4.6))
    outer = GridSpec(2, 2, figure=fig, left=0.075, right=0.965, top=0.9,
                     bottom=0.11, wspace=0.012, hspace=0.12)
    axes = {}
    for r, ds in enumerate(('original', 'scrambled')):
        for c, pt in enumerate(('SYMMETRIC', 'ROW')):
            inner = outer[r, c].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=0.07)
            top = fig.add_subplot(inner[0])
            bot = fig.add_subplot(inner[1], sharex=top)
            axes[(r, c)] = (top, bot)
            res = results[(ds, pt)].set_index('kernel').reindex(names)

            g = res['gain_improved'].values
            top.errorbar(x, g, yerr=[g - res['gain_lo'].values, res['gain_hi'].values - g],
                         fmt='_', ms=11, mew=1.6, color='#222222', ecolor='#222222',
                         elinewidth=0.8, capsize=2.2, zorder=4)
            ratio_axis(top, *ylims[ds])
            kernel_separators(top, len(names))
            top.tick_params(axis='x', labelbottom=False)

            imp, slow = 100 * res['improved'].values, 100 * res['slower'].values
            kw = dict(width=0.6, edgecolor='#222222', linewidth=0.5, alpha=0.85, zorder=3)
            bot.bar(x, imp, color=GOOD, **kw)
            bot.bar(x, slow, bottom=imp, color=BAD, **kw)
            bot.bar(x, 100 - imp - slow, bottom=imp + slow, color=NEUTRAL, **kw)
            bot.set_ylim(0, 100)
            bot.set_yticks([0, 50, 100])
            bot.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
            bot.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
            kernel_separators(bot, len(names))
            bot.set_xticks(x)
            if r == 1:
                bot.set_xticklabels(names, rotation=25, ha='right', rotation_mode='anchor')
            else:
                bot.tick_params(axis='x', labelbottom=False)
            if c == 1:
                top.tick_params(axis='y', which='both', labelleft=False)
                bot.tick_params(axis='y', labelleft=False)
            else:
                top.set_ylabel('Speedup\nwhen improved', fontsize=8.5)
                bot.set_ylabel('Matrices', fontsize=8.5)
    for c, pt in enumerate(('SYMMETRIC', 'ROW')):
        axes[(0, c)][0].set_title(f"{TITLES[pt].capitalize()}", fontsize=9,
                                  fontweight='bold', pad=3)
    for r, ds in enumerate(('original', 'scrambled')):
        y0 = axes[(r, 1)][1].get_position().y0
        y1 = axes[(r, 1)][0].get_position().y1
        x1 = axes[(r, 1)][0].get_position().x1
        fig.text(x1 + 0.004, (y0 + y1) / 2, TITLES[ds], rotation=270, ha='left',
                 va='center', fontsize=9, fontweight='bold')
    handles = [Patch(facecolor=col, edgecolor='#222222', linewidth=0.5, alpha=0.85, label=lab)
               for col, lab in ((GOOD, 'Faster'), (BAD, 'Slower'),
                                (NEUTRAL, 'Kept original (no denser candidate)'))]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.52, 0.935),
               ncol=3, frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.2, handletextpad=0.35)
    save(fig, f'improvement_bars_nc{N_COLS}')
    plt.close(fig)


class CrossbarHandle:
    """Legend entry for the strips crossbar: a box with a line through it."""
    def get_label(self):
        return 'Gain among faster'


class CrossbarHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        w, h = handlebox.width, handlebox.height
        box = Rectangle((x0, y0), w, h, facecolor='white', edgecolor=GAIN, linewidth=0.6,
                        transform=handlebox.get_transform())
        line = Line2D([x0, x0 + w], [y0 + h / 2] * 2, color=GAIN, lw=1.4,
                      solid_capstyle='butt', transform=handlebox.get_transform())
        handlebox.add_artist(box)
        handlebox.add_artist(line)
        return box


def draw_strips(ax, d, res, lo, hi, rng, pct_size=7):
    """One sina panel: per-matrix speedups per kernel with the gain crossbar and
    the faster/slower shares, on a log speedup axis clipped to [lo, hi]."""
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    from matplotlib.ticker import NullFormatter
    names = KERNEL_ORDER
    by_name = {v: k for k, v in KERNEL_NAMES.items()}
    for i, name in enumerate(names):
        s = d.loc[d['kernel_id'] == by_name[name], 'speedup'].values
        ls = np.log(np.clip(s, lo, hi))
        # sina jitter: horizontal spread proportional to local density
        grid = np.linspace(np.log(lo), np.log(hi), 300)
        bw = 0.12
        dens = np.exp(-0.5 * ((grid[:, None] - ls[None, :]) / bw) ** 2).sum(1)
        width = 0.42 * np.interp(ls, grid, dens) / dens.max()
        xj = i + rng.uniform(-1, 1, len(s)) * width
        col = np.where(s > 1, GOOD, np.where(s < 1, BAD, KEPT))
        ax.scatter(xj, np.exp(ls), s=1.6, c=col, alpha=0.55, lw=0,
                   rasterized=True, zorder=3)
        # gain among improved: crossbar = box over the 95% interval,
        # thicker line at the geometric mean
        g, glo, ghi = res.loc[name, ['gain_improved', 'gain_lo', 'gain_hi']]
        half = 0.24
        ax.add_patch(Rectangle((i - half, glo), 2 * half, ghi - glo,
                               facecolor='white', alpha=0.75, edgecolor=GAIN,
                               linewidth=0.6, zorder=5))
        ax.plot([i - half, i + half], [g, g], color=GAIN, lw=1.4, zorder=6,
                solid_capstyle='butt')
        # shares: faster along the top edge, slower along the bottom edge
        up, down = res.loc[name, 'improved'], res.loc[name, 'slower']
        halo = [pe.withStroke(linewidth=1.8, foreground='white')]
        ax.text(i, 0.985, f'{up:.0%}↑', color=GOOD, ha='center', va='top',
                transform=ax.get_xaxis_transform(), fontsize=pct_size,
                fontweight='bold', zorder=7, path_effects=halo)
        ax.text(i, 0.015, f'{down:.0%}↓', color=BAD, ha='center', va='bottom',
                transform=ax.get_xaxis_transform(), fontsize=pct_size,
                fontweight='bold', zorder=7, path_effects=halo)
    ax.set_yscale('log')
    ax.set_ylim(lo, hi)
    majors = [m for m in (0.2, 0.5, 1, 2, 5, 10, 20) if lo <= m <= hi]
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(majors))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}×'))
    ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(
        [m * k for m in (0.1, 1, 10) for k in range(2, 10) if lo <= m * k <= hi]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis='y', which='major', color='#b0b0b0', linewidth=0.6)
    ax.grid(True, axis='y', which='minor', color='#e4e4e4', linewidth=0.4)
    kernel_separators(ax, len(names))


def kernel_xticklabels(fig, ax, fontsize):
    from matplotlib.transforms import ScaledTranslation
    names = KERNEL_ORDER
    ax.set_xticks(np.arange(len(names)))
    labs = ax.set_xticklabels(
        [n.replace('cuSPARSE-', 'cuSPARSE\n').replace('-SpMM', '-\nSpMM')
          .replace('FlashSparse', 'Flash-\nSparse')
         for n in names],
        linespacing=0.9, fontsize=fontsize)
    # the two adjacent "cuSPARSE" labels are wider than a strip: push apart
    for lab, n in zip(labs, names):
        dx = {'cuSPARSE-BSR': -2.5, 'cuSPARSE-CSR': 2.5}.get(n, 0)
        if dx:
            lab.set_transform(lab.get_transform()
                              + ScaledTranslation(dx / 72, 0, fig.dpi_scale_trans))


def strips_legend_handles():
    handles = [Patch(facecolor=GOOD, edgecolor='#222222', linewidth=0.5, label='Faster'),
               Patch(facecolor=BAD, edgecolor='#222222', linewidth=0.5, label='Slower'),
               Patch(facecolor=KEPT, edgecolor='#222222', linewidth=0.5,
                     label='Kept original'),
               CrossbarHandle()]
    return handles


def strips_figure(raw, results, rng, cells=CELLS, width=PAGE_W, suffix=''):
    """Sina plot of per-matrix speedups per kernel, paper style.

    ``cells`` picks a subset of the four panels (e.g. only original/symmetric,
    drawn one column wide with ``width=COL_W``)."""
    ylims = {'original': (1 / 6, 12), 'scrambled': (1 / 6, 48)}
    dss = [d for d in ('original', 'scrambled') if any(c[0] == d for c in cells)]
    pts = [p for p in ('SYMMETRIC', 'ROW') if any(c[1] == p for c in cells)]
    single = len(dss) * len(pts) == 1
    fig, axes = plt.subplots(len(dss), len(pts), sharex=True, sharey='row', squeeze=False,
                             figsize=(width, 1.85 if single else 1.75 * len(dss)))
    for r, ds in enumerate(dss):
        lo, hi = ylims[ds]
        for c, pt in enumerate(pts):
            draw_strips(axes[r][c], raw[(ds, pt)], results[(ds, pt)].set_index('kernel'),
                        lo, hi, rng)
            if c == 1:   # shared y: no tick marks poking into the gap between columns
                axes[r][c].tick_params(axis='y', which='both', length=0, labelleft=False)
    for c, pt in enumerate(pts):
        if not single:
            axes[0][c].set_title(TITLES[pt].capitalize(), fontsize=9, fontweight='bold', pad=3)
        kernel_xticklabels(fig, axes[-1][c], 6.5 if single else 7.5)
    if single:
        fig.subplots_adjust(left=0.15, right=0.99, top=0.88, bottom=0.2)
    else:
        fig.subplots_adjust(left=0.085, right=0.965, top=0.89, bottom=0.085, wspace=0.012,
                            hspace=0.05)
        for r, ds in enumerate(dss):
            pos = axes[r][-1].get_position()
            fig.text(pos.x1 + 0.004, (pos.y0 + pos.y1) / 2, TITLES[ds], rotation=270,
                     ha='left', va='center', fontsize=9, fontweight='bold')
    if single:
        axes[0][0].set_ylabel('Speedup of the matrix with\nhighest block density', fontsize=7.5,
                              labelpad=1, linespacing=0.95)
    else:
        mid = (axes[0][0].get_position().y1 + axes[-1][0].get_position().y0) / 2
        fig.text(0.028, mid, 'Speedup of the densest candidate', rotation=90,
                 ha='left', va='center', fontsize=9)
    handles = strips_legend_handles()
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.56, 0.875) if single else (0.52, 0.925),
               ncol=4, frameon=False, handlelength=0.9, handleheight=0.9,
               columnspacing=1.2, handletextpad=0.35,
               handler_map={CrossbarHandle: CrossbarHandler()})
    save(fig, f'improvement_strips_nc{N_COLS}{suffix}')
    plt.close(fig)


def strips_stacked_figure(raw, results, rng, pt='SYMMETRIC', height=2.2):
    """Original (top) and scrambled (bottom) matrices for one reordering type,
    stacked one column wide with a shared legend, y label and kernel labels."""
    ylims = {'original': (1 / 6, 12), 'scrambled': (1 / 6, 48)}
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(COL_W, height))
    for ax, ds in zip(axes, ('original', 'scrambled')):
        lo, hi = ylims[ds]
        draw_strips(ax, raw[(ds, pt)], results[(ds, pt)].set_index('kernel'), lo, hi, rng,
                    pct_size=6.5)
        ax.tick_params(axis='y', labelsize=6.5)
    kernel_xticklabels(fig, axes[-1], 6.5)
    axes[-1].tick_params(axis='x', pad=1.5)
    fig.subplots_adjust(left=0.11, right=0.955, top=0.925, bottom=0.135, hspace=0.06)
    for ax, ds in zip(axes, ('original', 'scrambled')):
        pos = ax.get_position()
        fig.text(pos.x1 + 0.006, (pos.y0 + pos.y1) / 2, TITLES[ds].split()[0], rotation=270,
                 ha='left', va='center', fontsize=7, fontweight='bold')
    fig.text(0.005, (axes[0].get_position().y1 + axes[1].get_position().y0) / 2,
             'Speedup of the matrix with highest block density', rotation=90,
             ha='left', va='center', fontsize=7)
    fig.legend(handles=strips_legend_handles(), loc='lower center',
               bbox_to_anchor=(0.55, 0.925), ncol=4, frameon=False, handlelength=0.9,
               handleheight=0.9, columnspacing=1.0, handletextpad=0.35, borderaxespad=0,
               borderpad=0.1, fontsize=7, handler_map={CrossbarHandle: CrossbarHandler()})
    save(fig, f'improvement_strips_nc{N_COLS}_{pt.lower()}', tight=True)
    plt.close(fig)


def main():
    style()
    rng = np.random.default_rng(0)
    results, tables, raw = {}, [], {}
    for ds, pt in CELLS:
        raw[(ds, pt)] = densest_speedups(ds, pt)
        r = summarise(raw[(ds, pt)], rng)
        results[(ds, pt)] = r
        tables.append(r.assign(dataset=ds, perm_type=pt))
        print(f'== {ds} / {pt}')
        print(r.round(3).to_string(index=False))
    OUT.mkdir(exist_ok=True)
    pd.concat(tables).round(4).to_csv(OUT / f'improvement_bars_nc{N_COLS}.csv', index=False)
    figure(results)
    strips_figure(raw, results, np.random.default_rng(1))
    strips_figure(raw, results, np.random.default_rng(1), cells=[('original', 'SYMMETRIC')],
                  width=COL_W, suffix='_original_symmetric')
    strips_stacked_figure(raw, results, np.random.default_rng(1))


if __name__ == '__main__':
    main()
