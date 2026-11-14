"""Part 1a (PREREG Part 1): variance split and input gate-bit budget along depth, one net.
usage: part1a.py n seed [--K 16384] [--out file]
Depths D = {n/8, n/4, n/2, n, 2n, 32} (+ 4n if n <= 64); one pass to max(D) gives every depth (sigma = 0: the depth-L
output is h_L; sigma > 0: the depth-L output is relu(z_L), the noise after layer L does not touch it).
Per layer l <= max(D): input gate statistics over K unit directions (Hamming, marginal entropy sum, live units).
Per depth L in D: three-term split (var_split, unit directions, sigma = 0), Var_u log A; at sigma = 1e-3 (coupled,
Gaussian inputs x = r u with r ~ chi_n): noise-induced variance tr Var(a_s - a_0)/tr Var(a_0) and the change of the split
shares computed with the same as-if-independent estimator on (|x|, |a|/|x|, a/|a|).
Seeds: inputs [7001, seed, n], radii [7004, seed, n], noise [7005, seed, n, l]."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--K', type=int, default=16384); ap.add_argument('--sigma', type=float, default=1e-3)
ap.add_argument('--out')
a = ap.parse_args()
n, seed, K = a.n, a.seed, a.K
t0 = time.time()
D = sorted({max(1, n // 8), n // 4, n // 2, n, 2 * n, 32} | ({4 * n} if n <= 64 else set()))
Lmax = max(D)
net = ib.Net(seed, n, Lmax)
U = ib.sphere(np.random.default_rng([7001, seed, n]).standard_normal((K, n)))
r = np.sqrt(np.random.default_rng([7004, seed, n]).chisquare(n, K))
X = U * r[:, None]
res = dict(n=n, seed=seed, K=K, sigma=a.sigma, depths=D, layers=[], split={}, noise={})


def split_xa(Xin, Aout):
    """As-if-independent split estimator on Gaussian-input samples (for the coupled noise comparison)."""
    nx = np.linalg.norm(Xin, axis=1)
    Amp = np.linalg.norm(Aout, axis=1) / nx
    G = Aout / nx[:, None]          # = g(u) when sigma = 0
    vs = ib.var_split(G, n)
    trv = float(np.sum(Aout.var(0, ddof=1)))
    return dict(s_rad=vs['s_rad'], s_amp=vs['s_amp'], s_dir=vs['s_dir'], V_split=vs['V'], trvar=trv)


h = U.copy()
hx0 = X.copy(); hxs = X.copy()
for l in range(1, Lmax + 1):
    W = net.W[l - 1]
    z = h @ W.T
    st_ = ib.gate_layer_stats(z)
    dead = float(np.mean(np.all(z <= 0, 1)))
    res['layers'].append(dict(l=l, **st_, dead_frac=dead))
    h = np.maximum(z, 0.0)
    # sigma comparison on Gaussian inputs, coupled
    z0 = hx0 @ W.T; zs = hxs @ W.T
    a0 = np.maximum(z0, 0.0); as_ = np.maximum(zs, 0.0)
    if l in D:
        vs = ib.var_split(h, n)
        A = np.linalg.norm(h, axis=1)
        vs['logA_var'] = float(np.var(np.log(A[A > 0]), ddof=1)) if np.sum(A > 0) > 2 else None
        vs['dead_frac'] = float(np.mean(A == 0))
        res['split'][str(l)] = vs
        s0 = split_xa(X, a0); s1 = split_xa(X, as_)
        dvar = float(np.sum((as_ - a0).var(0, ddof=1))) / max(s0['trvar'], 1e-300)
        res['noise'][str(l)] = dict(zero=s0, sig=s1, coupled_diff_var=dvar)
        print(f'L={l}: s_rad {vs["s_rad"]:.3f} s_amp {vs["s_amp"]:.3f} s_dir {vs["s_dir"]:.3f} V {vs["V"]:.3f} '
              f'logA_var {vs["logA_var"]} | noise dvar {dvar:.2e} shares {s1["s_amp"]:.3f}/{s1["s_dir"]:.3f} vs '
              f'{s0["s_amp"]:.3f}/{s0["s_dir"]:.3f} | {time.time() - t0:.0f}s', flush=True)
    hx0 = a0
    if l < Lmax:
        xi = np.random.default_rng([7005, seed, n, l]).standard_normal(as_.shape)
        hxs = as_ + a.sigma * xi
    else:
        hxs = as_
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
