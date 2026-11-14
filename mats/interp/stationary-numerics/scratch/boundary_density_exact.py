"""Exact next-layer boundary densities for the small-width stationary laws (post hoc, round-2 review finding 7).
Because h_0 = relu(W_0 h_-1) + xi_0 with Gaussian xi_0, the density of z_i = w_i . h_0 (w_i a row of W_1) at 0 is
  p_i(0) = sum_j m_j phi(w_i . A_j / |w_i|) / |w_i|,   A_j = relu(W_0 x_j), m_j = grid masses of h_-1,
exact given the layer -1 measure (no histogram). The Gaussian approximation uses the exact mean and covariance of h_0;
the gate-pattern mixture uses each pattern's exact mean and covariance plus the noise.
usage: boundary_density_exact.py n seed T dx [--marg 8] --out file"""
import sys, math, json, argparse
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')
import numpy as np
import grid_transfer as gt
import st

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('T', type=int); ap.add_argument('dx', type=float)
ap.add_argument('--marg', type=float, default=8.0); ap.add_argument('--out', required=True)
a = ap.parse_args()
n, seed = a.n, a.seed
G, mass = gt.gaussian_init(n, a.dx)
for l in range(-a.T + 1, 0):
    G, mass = gt.layer(G, mass, st.W_layer(seed, n, l), a.dx, a.marg)
X = G.points(); m = mass.ravel(); keep = m > 1e-18 * m.max()
X, m = X[keep], m[keep] / m[keep].sum()
W0, W1 = st.W_layer(seed, n, 0), st.W_layer(seed, n, 1)
Z = X @ W0.T
A = np.maximum(Z, 0.0)
mu = m @ A
C = (A - mu).T @ ((A - mu) * m[:, None]) + np.eye(n)
cls = ((Z > 0).astype(int) * (2 ** np.arange(n))[None]).sum(1)
comps = []
for s in np.unique(cls):
    sel = cls == s
    w = m[sel].sum()
    ms = (m[sel] @ A[sel]) / w
    Cs = (A[sel] - ms).T @ ((A[sel] - ms) * m[sel][:, None]) / w + np.eye(n)
    comps.append((w, ms, Cs))
phi = lambda x: np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
units = []
for i in range(n):
    w = W1[i]; wn = float(np.linalg.norm(w))
    exact = float(np.sum(m * phi((A @ w) / wn)) / wn)
    sg = math.sqrt(w @ C @ w)
    gauss = float(phi(-(w @ mu) / sg) / sg)
    mix = float(sum(c[0] * phi(-(w @ c[1]) / math.sqrt(w @ c[2] @ w)) / math.sqrt(w @ c[2] @ w) for c in comps))
    units.append(dict(exact=exact, gauss=gauss, mix=mix, gauss_rel_err=abs(gauss / exact - 1), mix_rel_err=abs(mix / exact - 1)))
json.dump(dict(n=n, seed=seed, T=a.T, dx=a.dx, units=units), open(a.out, 'w'), indent=1)
print(n, seed, a.dx, [(round(u['gauss_rel_err'], 3), round(u['mix_rel_err'], 3)) for u in units], flush=True)
