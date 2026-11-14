"""Figures for the exact small-width densities: figs/smalln_density_n2.png, figs/smalln_summary.png."""
import sys, json, glob, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
from plotstyle import plt, setup, seq, CAT, INK, INK2

D = '/home/agent/workspaces/mats/work/stationary-numerics'
setup()
seeds = [1000, 1001, 1002]
fig, ax = plt.subplots(2, 3, figsize=(13, 7.6))
for c, s in enumerate(seeds):
    z = np.load(f'{D}/runs/S_n2_s{s}.npz')
    lo, dx, shape = z['lo'], float(z['dx']), tuple(z['shape'])
    m = z['m'].reshape(shape); qg = z['qg']; qm = z['qm']; W1 = z['W1']
    x = lo[0] + dx * np.arange(shape[0]); y = lo[1] + dx * np.arange(shape[1])
    dens = m / dx ** 2
    a = ax[0, c]
    a.grid(False)
    im = a.pcolormesh(x, y, np.log10(np.maximum(dens.T, 1e-12)), cmap='Blues', vmin=-6, vmax=float(np.log10(dens.max())), shading='auto')
    # Gaussian contours (moment-matched)
    a.contour(x, y, np.log10(np.maximum(qg.T / dx ** 2, 1e-12)), levels=[-4, -3, -2, -1], colors=[CAT[1]], linewidths=0.8)
    # next-layer gate boundaries w_i . h = 0
    xx = np.linspace(x[0], x[-1], 10)
    for i in range(2):
        w = W1[i]
        if abs(w[1]) > 1e-12:
            a.plot(xx, -w[0] * xx / w[1], color=INK, lw=0.9, ls='--')
    a.set_xlim(x[0], x[-1]); a.set_ylim(y[0], y[-1])
    a.set_xlabel('h_1'); a.set_ylabel('h_2')
    r = json.load(open(f'{D}/runs/S_n2_s{s}.json'))
    a.set_title(f'n = 2, net {s}: log10 density (TV to Gaussian {r["vs_gaussian"]["TV"]:.2f})', fontsize=9)
    fig.colorbar(im, ax=a, fraction=0.046)
    # boundary marginal of z_1 = w_1 . h
    b = ax[1, c]
    X = np.stack(np.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
    zz = X @ W1[0]
    bins = np.linspace(-6, 6, 241)
    for arr, lab, col, ls in ((m.ravel(), 'exact', CAT[0], '-'), (qg.ravel(), 'moment-matched Gaussian', CAT[1], '--'),
                              (qm.ravel(), 'gate-pattern mixture', CAT[2], ':')):
        hst, _ = np.histogram(zz, bins=bins, weights=arr)
        b.plot(0.5 * (bins[1:] + bins[:-1]), hst / (bins[1] - bins[0]), color=col, ls=ls, label=lab)
    sd = np.linalg.norm(W1[0])
    b.axvspan(-sd, sd, color='#e4e3df', alpha=0.6, lw=0, label='within one noise SD of the boundary')
    b.set_xlabel('next-layer preactivation z_1 = w_1 . h'); b.set_ylabel('density')
    b.set_title('Mass near the next gate boundary', fontsize=9); b.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{D}/figs/smalln_density_n2.png'); plt.close(fig)

# summary: TV to Gaussian and mixture for every net
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for i, n in enumerate((2, 3, 4)):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f'{D}/runs/S_n{n}_s*.json'))]
    if not rs:
        continue
    xg = [r['vs_gaussian']['TV'] for r in rs]; xm = [r['vs_mixture']['TV'] for r in rs]
    ax[0].scatter([n - 0.08] * len(xg), xg, color=CAT[1], s=20, label='moment-matched Gaussian' if i == 0 else None)
    ax[0].scatter([n + 0.08] * len(xm), xm, color=CAT[2], s=20, label='gate-pattern mixture' if i == 0 else None)
    qs = [r['Q_mean'] for r in rs]
    ax[1].scatter(qs, xg, color=CAT[1], marker='os^'[i], s=24, label=f'Gaussian, n = {n}')
    ax[1].scatter(qs, xm, color=CAT[2], marker='os^'[i], s=24, label=f'mixture, n = {n}')
ax[0].set_xticks([2, 3, 4]); ax[0].set_xlabel('width n'); ax[0].set_ylabel('total variation to the exact density')
ax[0].set_yscale('log'); ax[0].legend(fontsize=8)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('mean Q = |h_0|^2 / n of the net at layer 0')
ax[1].set_ylabel('total variation to the exact density'); ax[1].legend(fontsize=7, ncol=2)
ax[0].set_title('How far from Gaussian the exact quenched law is'); ax[1].set_title('Non-Gaussianity tracks the layer radius')
fig.tight_layout(); fig.savefig(f'{D}/figs/smalln_summary.png'); plt.close(fig)
print('ok')
