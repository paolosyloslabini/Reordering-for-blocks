"""Within-matrix response of SpMM speed to 16x16 block density (original
SuiteSparse matrices, symmetric and row reorderings pooled).

Model, per kernel:  log2 GFLOPS_ij = mu_i + f(log2 d_ij) + e_ij
  i = matrix (fixed effect), j = ordering (incl. the original one),
  f = natural cubic spline (knots at the 5/27.5/50/72.5/95% quantiles of log2 d).
Plotted: local elasticity alpha(d) = df/dlog2 d.
Bands: 95% bootstrap over matrices. Curves span the 2nd-98th percentile of d.

Second figure (elasticity_vs_matrix_density): elasticity with respect to 16x16
block density for groups of matrices with similar plain density nnz/(m*n).
Matrix density is fixed per matrix, so for each point on the x-axis we fit one
within-matrix slope  alpha = sum w_i sum_j xw*yw / sum w_i sum_j xw^2  (xw, yw =
log2 block density and log2 GFLOPS, demeaned per matrix), with Gaussian weights
w_i on log10 matrix density (bandwidth 0.25 decades).

Third figure (correlation_block_density_original): local within-matrix Pearson
correlation r(d) between log2 GFLOPS and log2 block density. At each grid point,
configurations get Gaussian weights on log2 d (bandwidth 0.75); x and y are
demeaned with per-matrix weighted means; r = Sxy / sqrt(Sxx Syy). The bandwidth
is fixed across kernels, since a narrower window leaves less spread in d and
lowers r.

Dense operand: 256 columns, except SMaT and ASpT, which always ran with 32.
Runs whose padded 32x32 work would exceed the hardware peak (silent skips)
are dropped for cuSPARSE-BSR and SMaT.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIXED_32_COLS, INK, INK2, KERNEL_NAMES, PAGE_W, PALETTE,
                    clean_axes, load_pipeline, save, style)

N_COLS = 256
PEAK_TFLOPS = {'CUSPARSE_SPMM_BSR_bs32': 19.5, 'SMAT_SPMM_bs32': 312.0}
N_BOOT = 200


def data():
    parts = [load_pipeline('original', pt) for pt in ('SYMMETRIC', 'ROW')]
    df = pd.concat(parts, ignore_index=True)
    df.loc[df['strategy'] == 'Original', 'perm_type'] = '-'   # shared baseline
    df = df.drop_duplicates(['kernel_id', 'matrix', 'perm_type', 'strategy', 'n_cols'])
    width = np.where(df['kernel_id'].isin(FIXED_32_COLS), 32, N_COLS)
    df = df[(df['n_cols'] == width) & (df['gflops'] > 0) & np.isfinite(df['gflops'])
            & (df['block_density_16'] > 0)]
    for k, peak in PEAK_TFLOPS.items():
        m = df['kernel_id'] == k
        executed = (2 * df['nonzero_blocks_32'] * 1024 * df['n_cols']
                    / (df['time_operation_ms'] / 1e3) / 1e12)
        bad = m & (executed > peak)
        print(f'{KERNEL_NAMES[k]}: dropped {bad.sum()} impossible runs')
        df = df[~bad]
    return df


def ns_basis(x, knots):
    """Natural cubic spline basis (ESL 5.2.1): linear term + K-2 cubic terms."""
    k_last = knots[-1]

    def d(k):
        return ((np.clip(x - knots[k], 0, None) ** 3
                 - np.clip(x - k_last, 0, None) ** 3) / (k_last - knots[k]))
    return np.column_stack([x] + [d(k) - d(len(knots) - 2)
                                  for k in range(len(knots) - 2)])


def fit(x, y, groups, knots):
    B = ns_basis(x, knots)
    Bw = B - pd.DataFrame(B).groupby(groups).transform('mean').values
    yw = y - pd.Series(y).groupby(groups).transform('mean').values
    return np.linalg.lstsq(Bw, yw, rcond=None)[0]


def curves(df, rng):
    res = {}
    for k, name in KERNEL_NAMES.items():
        t = df[df['kernel_id'] == k]
        x = np.log2(t['block_density_16'].values)
        y = np.log2(t['gflops'].values)
        g = t['matrix'].values
        knots = np.quantile(x, [.05, .275, .5, .725, .95])
        grid = np.linspace(*np.quantile(x, [.02, .98]), 150)
        ref = np.median(x)
        Bg, Br = ns_basis(grid, knots), ns_basis(np.array([ref]), knots)

        def shape(beta):
            f = Bg @ beta - (Br @ beta)[0]
            return f, np.gradient(f, grid)

        f, e = shape(fit(x, y, g, knots))
        mats = np.unique(g)
        idx = {m: np.where(g == m)[0] for m in mats}
        F, E = [], []
        for _ in range(N_BOOT):
            draw = rng.choice(mats, len(mats))
            sel = np.concatenate([idx[m] for m in draw])
            lab = np.concatenate([[i] * len(idx[m]) for i, m in enumerate(draw)])
            fb, eb = shape(fit(x[sel], y[sel], lab, knots))
            F.append(fb)
            E.append(eb)
        res[k] = dict(grid=grid, f=f, e=e,
                      f_lo=np.percentile(F, 2.5, 0), f_hi=np.percentile(F, 97.5, 0),
                      e_lo=np.percentile(E, 2.5, 0), e_hi=np.percentile(E, 97.5, 0),
                      n=len(t), mats=len(mats))
        print(f'{name:13s} configs={len(t):6d} matrices={len(mats)}')
    return res


def figure(res):
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.62, 2.6))
    for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE):
        c = res[k]
        X = 2 ** c['grid']
        label = name + (' (32 cols)' if k in FIXED_32_COLS else '')
        ax.fill_between(X, c['e_lo'], c['e_hi'], color=col, alpha=0.13, lw=0)
        ax.plot(X, c['e'], color=col, lw=1.6, label=label)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axhline(1, color=INK2, lw=0.7, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r'Block density $d$ ($16{\times}16$)')
    ax.set_ylabel(r'Elasticity $\alpha(d)$')
    clean_axes(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              handlelength=1.6)
    fig.tight_layout()
    save(fig, 'elasticity_block_density_original')
    plt.close(fig)


def local_correlation(df, rng, bandwidth=0.75, min_eff_n=50):
    """Local within-matrix correlation of log speed with log block density.

    Per-matrix weighted sums are computed once per grid point; the matrix
    bootstrap then only reweights matrices by their draw counts.
    """
    res = {}
    for k in KERNEL_NAMES:
        t = df[df['kernel_id'] == k]
        x = np.log2(t['block_density_16'].values)
        y = np.log2(t['gflops'].values)
        codes, mats = pd.factorize(t['matrix'].values)
        M = len(mats)
        grid = np.linspace(*np.quantile(x, [.02, .98]), 150)
        sxx = np.zeros((M, len(grid)))
        syy = np.zeros_like(sxx)
        sxy = np.zeros_like(sxx)
        ok = np.zeros(len(grid), bool)
        for gi, c in enumerate(grid):
            w = np.exp(-0.5 * ((x - c) / bandwidth) ** 2)
            ok[gi] = w.sum() ** 2 / (w ** 2).sum() >= min_eff_n
            sw = np.bincount(codes, w, M)
            sx = np.bincount(codes, w * x, M)
            sy = np.bincount(codes, w * y, M)
            safe = np.where(sw > 0, sw, 1)
            sxx[:, gi] = np.bincount(codes, w * x * x, M) - sx ** 2 / safe
            syy[:, gi] = np.bincount(codes, w * y * y, M) - sy ** 2 / safe
            sxy[:, gi] = np.bincount(codes, w * x * y, M) - sx * sy / safe

        def corr(cnt):
            a, b, c2 = cnt @ sxy, cnt @ sxx, cnt @ syy
            return np.where(ok, a / np.sqrt(b * c2), np.nan)

        r = corr(np.ones(M))
        B = np.array([corr(np.bincount(rng.integers(0, M, M), minlength=M).astype(float))
                      for _ in range(N_BOOT)])
        res[k] = dict(grid=grid, r=r, lo=np.nanpercentile(B, 2.5, 0),
                      hi=np.nanpercentile(B, 97.5, 0))
    return res


def figure_correlation(res):
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.62, 2.6))
    for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE):
        c = res[k]
        X = 2 ** c['grid']
        ok = ~np.isnan(c['r'])
        label = name + (' (32 cols)' if k in FIXED_32_COLS else '')
        ax.fill_between(X[ok], c['lo'][ok], c['hi'][ok], color=col, alpha=0.13, lw=0)
        ax.plot(X[ok], c['r'][ok], color=col, lw=1.6, label=label)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xscale('log')
    ax.set_xlabel(r'Block density $d$ ($16{\times}16$)')
    ax.set_ylabel(r'Local correlation $r(d)$')
    clean_axes(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              handlelength=1.6)
    fig.tight_layout()
    save(fig, 'correlation_block_density_original')
    plt.close(fig)


def alpha_vs_matrix_density(df, rng, bandwidth=0.25, min_eff_n=25):
    res = {}
    for k in KERNEL_NAMES:
        t = df[df['kernel_id'] == k]
        x = np.log2(t['block_density_16'].values)
        y = np.log2(t['gflops'].values)
        g = t['matrix'].values
        xw = x - pd.Series(x).groupby(g).transform('mean').values
        yw = y - pd.Series(y).groupby(g).transform('mean').values
        per = pd.DataFrame({'m': g, 'sxy': xw * yw, 'sxx': xw * xw}).groupby('m').sum()
        rho = np.log10(t.groupby('matrix')['density'].first().reindex(per.index).values)
        sxy, sxx = per['sxy'].values, per['sxx'].values
        grid = np.linspace(*np.quantile(rho, [.03, .97]), 80)

        def curve(idx):
            out = np.full(len(grid), np.nan)
            for i, c in enumerate(grid):
                w = np.exp(-0.5 * ((rho[idx] - c) / bandwidth) ** 2)
                if w.sum() ** 2 / (w ** 2).sum() >= min_eff_n:
                    out[i] = (w * sxy[idx]).sum() / (w * sxx[idx]).sum()
            return out

        a = curve(np.arange(len(rho)))
        B = np.array([curve(rng.integers(0, len(rho), len(rho))) for _ in range(N_BOOT)])
        res[k] = dict(grid=grid, a=a, lo=np.nanpercentile(B, 2.5, 0),
                      hi=np.nanpercentile(B, 97.5, 0))
    return res


def figure_matrix_density(res):
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.62, 2.6))
    for (k, name), col in zip(KERNEL_NAMES.items(), PALETTE):
        c = res[k]
        X = 10 ** c['grid']
        ok = ~np.isnan(c['a'])
        label = name + (' (32 cols)' if k in FIXED_32_COLS else '')
        ax.fill_between(X[ok], c['lo'][ok], c['hi'][ok], color=col, alpha=0.13, lw=0)
        ax.plot(X[ok], c['a'][ok], color=col, lw=1.6, label=label)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.axhline(1, color=INK2, lw=0.7, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel(r'Matrix density $\mathrm{nnz}/(m \cdot n)$')
    ax.set_ylabel(r'Elasticity to block density ($16{\times}16$)')
    clean_axes(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False,
              handlelength=1.6)
    fig.tight_layout()
    save(fig, 'elasticity_vs_matrix_density')
    plt.close(fig)


def main():
    style()
    df = data()
    rc = local_correlation(df, np.random.default_rng(13))
    figure_correlation(rc)
    rows = []
    for k, c in rc.items():
        for d in (0.005, 0.01, 0.02, 0.05, 0.1):
            if c['grid'][0] <= np.log2(d) <= c['grid'][-1]:
                rows.append(dict(kernel=KERNEL_NAMES[k], density=d,
                                 r=np.interp(np.log2(d), c['grid'], c['r'])))
    rt = pd.DataFrame(rows).pivot(index='kernel', columns='density', values='r')
    print('local correlation r(d)'); print(rt.round(2).to_string())
    rt.round(3).to_csv('analysis/density_response/figures/correlation_table.csv')
    rm = alpha_vs_matrix_density(df, np.random.default_rng(11))
    figure_matrix_density(rm)
    for k, c in rm.items():
        pts = [(10 ** c['grid'][i], c['a'][i]) for i in np.linspace(0, len(c['grid']) - 1, 5).astype(int)]
        print(f"{KERNEL_NAMES[k]:13s}", ' '.join(f'{r:.0e}:{a:5.2f}' for r, a in pts))
    res = curves(df, np.random.default_rng(7))
    rows = []
    for k, c in res.items():
        for d in (0.005, 0.01, 0.02, 0.05, 0.1):
            if np.log2(d) < c['grid'][0] or np.log2(d) > c['grid'][-1]:
                continue
            rows.append(dict(kernel=KERNEL_NAMES[k], density=d,
                             elasticity=np.interp(np.log2(d), c['grid'], c['e'])))
    table = pd.DataFrame(rows).pivot(index='kernel', columns='density', values='elasticity')
    print(table.round(2).to_string())
    table.round(3).to_csv('analysis/density_response/figures/elasticity_table.csv')
    figure(res)


if __name__ == '__main__':
    main()
