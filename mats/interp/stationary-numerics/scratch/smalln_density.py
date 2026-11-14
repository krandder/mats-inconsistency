"""Exact quenched stationary densities at very small width (grid transfer operator, no sampling).
For net `seed` (stationary lane's st.W_layer) at width n: pull back T layers from N(0, I) to layer 0 at two grid
steps; report discretisation TV, pullback TV (two starts), masses by the gate pattern of layer 0, moments,
TV and KL of the exact density against (i) the moment-matched Gaussian, (ii) the gate-pattern Gaussian mixture
sum_s pi_s N(mu_s, Sigma_s + I) (s = gate pattern of the last layer; mu_s, Sigma_s exact moments of relu(W_0 h_-1) on s),
and the next layer's gate statistics: per unit, P(z_1i > 0), the boundary density p_{z_1i}(0) and the mass within one
noise SD |w_i| of the boundary; entropy of the layer-1 gate pattern.
usage: smalln_density.py n seed T dx_coarse dx_fine --out file [--save-grid file.npz]"""
import sys, os, math, json, time, argparse
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
import grid_transfer as gt
import st

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('T', type=int)
ap.add_argument('dxc', type=float); ap.add_argument('dxf', type=float)
ap.add_argument('--out', required=True); ap.add_argument('--save-grid')
ap.add_argument('--marg', type=float, default=8.0); ap.add_argument('--skip-two-starts', action='store_true')
a = ap.parse_args()
n, seed, T = a.n, a.seed, a.T
t0 = time.time()
Gc, mc, *_ = gt.pullback(seed, n, T, a.dxc, marg=a.marg)
G, m, A, mA, Z = gt.pullback(seed, n, T, a.dxf, marg=a.marg)
res = dict(n=n, seed=seed, T=T, dx=[a.dxc, a.dxf], grid_shape=list(G.shape))
res['TV_refine'] = float(0.5 * np.abs(gt.resample_to(Gc, mc, G) - m).sum())
if a.skip_two_starts:
    res['TV_pullback_two_starts'] = None
else:
    Gb, mb, *_ = gt.pullback(seed, n, T, a.dxc, init=gt.gaussian_init(n, a.dxc, center=[5.0] + [0.0] * (n - 1), scale=2.0), marg=a.marg)
    res['TV_pullback_two_starts'] = float(0.5 * np.abs(gt.resample_to(Gc, mc, Gb) - mb).sum())
m = m / m.sum()
X = G.points()
p = m.ravel()
mu = p @ X
C = (X - mu).T @ ((X - mu) * p[:, None])
res['mean'] = mu.tolist(); res['cov'] = C.tolist()
res['Q_mean'] = float(np.sum(p * np.sum(X * X, 1)) / n)
# gate patterns of layer 0 (source cells)
mA = mA / mA.sum()
cls = ((Z > 0).astype(int) * (2 ** np.arange(n))[None]).sum(1)
pat = {}
mix = []
for s in range(2 ** n):
    sel = cls == s
    w = mA[sel].sum()
    if w <= 0:
        continue
    As = A[sel]
    mus = (mA[sel] @ As) / w
    Cs = (As - mus).T @ ((As - mus) * mA[sel][:, None]) / w
    pat[format(s, f'0{n}b')[::-1]] = float(w)
    mix.append((w, mus, Cs + np.eye(n)))
res['mass_by_gate_pattern_layer0'] = pat  # key: gate of unit 1..n (1 = on)
dv = G.dx ** n


def gauss_cells(mu_, C_):
    Ci = np.linalg.inv(C_)
    d = X - mu_
    q = np.exp(-0.5 * np.einsum('ij,jk,ik->i', d, Ci, d)) / math.sqrt((2 * math.pi) ** n * np.linalg.det(C_))
    return q * dv


qg = gauss_cells(mu, C)
qm = sum(w * gauss_cells(mus, Cs) for w, mus, Cs in mix)


def dist(q):
    tv = 0.5 * np.abs(p - q).sum()
    ok = p > 1e-300
    kl = float(np.sum(p[ok] * (np.log(p[ok]) - np.log(np.maximum(q[ok], 1e-300)))))
    return dict(TV=float(tv), KL=kl)


res['vs_gaussian'] = dist(qg)
res['vs_mixture'] = dist(qm)
# skewness / excess kurtosis of projections on the covariance eigenvectors
ev, V = np.linalg.eigh(C)
proj = []
for k in range(n):
    y = (X - mu) @ V[:, k]
    s2 = p @ y ** 2
    proj.append(dict(eig=float(ev[k]), skew=float((p @ y ** 3) / s2 ** 1.5), exkurt=float((p @ y ** 4) / s2 ** 2 - 3)))
res['projections'] = proj
# next layer gates
W1 = st.W_layer(seed, n, 1)
Z1 = X @ W1.T
units = []
for i in range(n):
    wn = float(np.linalg.norm(W1[i]))
    zi = Z1[:, i]
    # density at 0 by a histogram of width h0 around 0 (mass within |z| < h0 / (2 h0))
    h0 = 0.05 * wn
    dens0 = float(p[np.abs(zi) < h0].sum() / (2 * h0))
    band = float(p[np.abs(zi) < wn].sum())
    mg, sg = float(W1[i] @ mu), float(math.sqrt(W1[i] @ C @ W1[i]))
    from scipy.stats import norm
    band_g = float(norm.cdf((wn - mg) / sg) - norm.cdf((-wn - mg) / sg))
    dens0_g = float(norm.pdf(-mg / sg) / sg)
    # mixture: z ~ sum_s w N(w_i.mu_s, w_i^T C_s w_i)
    band_m = float(sum(w * (norm.cdf((wn - W1[i] @ ms) / math.sqrt(W1[i] @ Cs @ W1[i])) -
                            norm.cdf((-wn - W1[i] @ ms) / math.sqrt(W1[i] @ Cs @ W1[i]))) for w, ms, Cs in mix))
    dens0_m = float(sum(w * norm.pdf(-(W1[i] @ ms) / math.sqrt(W1[i] @ Cs @ W1[i])) / math.sqrt(W1[i] @ Cs @ W1[i])
                        for w, ms, Cs in mix))
    units.append(dict(P_on=float(p[zi > 0].sum()), P_on_gauss=float(norm.sf(-mg / sg)), noise_sd=wn,
                      boundary_density=dens0, boundary_density_gauss=dens0_g, boundary_density_mix=dens0_m,
                      mass_within_1sd=band, mass_within_1sd_gauss=band_g, mass_within_1sd_mix=band_m))
res['next_layer_units'] = units
g1 = ((Z1 > 0).astype(int) * (2 ** np.arange(n))[None]).sum(1)
pg1 = np.bincount(g1, weights=p, minlength=2 ** n)
res['gate_entropy_layer1_bits'] = float(-np.sum(pg1[pg1 > 0] * np.log2(pg1[pg1 > 0])))
res['gate_pattern_probs_layer1'] = pg1.tolist()
res['sec'] = time.time() - t0
if a.save_grid:
    np.savez_compressed(a.save_grid, lo=G.lo, dx=G.dx, shape=np.array(G.shape), m=m, W1=W1,
                        qg=qg.reshape(G.shape), qm=qm.reshape(G.shape))
json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps({k: res[k] for k in ('n', 'seed', 'TV_refine', 'TV_pullback_two_starts', 'vs_gaussian', 'vs_mixture',
                                      'gate_entropy_layer1_bits', 'Q_mean', 'sec')}), flush=True)
