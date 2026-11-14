"""Crux check (PREREG section 2): rank of the back value function on its own top-k relevant directions vs k.

usage: python3 scratch/crux_local.py n c Lp [--nb 256] [--P 1024]
For each back b: a reference cut pre-activation z0 (random front, random input), J = d a_L / d z_c at z0, V = right
singular vectors. For k and rho: points z0 + rho |z0| V_k R_b w_i (w_i ~ N(0, I_k)/sqrt(k), common to all backs;
R_b Haar on O(k)). Matrix X[i, (b, j)] = a_L,j at point i. r_loc(eps) = smallest s with uncentred tail <= eps^2 Vloc,
Vloc = mean over columns of the variance over points. Control: V_k -> random k-dim subspace of the active coordinates.
Writes runs/crux_n{n}_c{c}_L{Lp}.json and prints a table.
"""
import argparse, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import crnet as cn

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('c', type=int); ap.add_argument('Lp', type=int)
ap.add_argument('--nb', type=int, default=256); ap.add_argument('--P', type=int, default=1024)
a = ap.parse_args()
n, c, Lp, nb, P = a.n, a.c, a.Lp, a.nb, a.P
tag = cn.cfg_tag(n, c, Lp, 'crux')
KS = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if k <= n // 2]
RHOS = (0.05, 0.2)
EPS = (0.3, 0.1, 0.03, 0.01)


def back_fwd(Z, Wb):  # Z: (m, n) cut pre-activations -> a_L (m, n)
    h = np.maximum(Z, 0)
    for W in Wb:
        h = np.maximum(h @ W.astype(np.float64).T, 0)
    return h


g = cn.rng(tag, 9)
Wcache, z0s, Vs, kst, act = [], [], [], [], []
for b in range(nb):
    Wf = cn.front_weights(tag, n, c, 10**6 + b)
    z0 = cn.forward_pre(cn.directions(n, 1, tag, 8, b), Wf)[0].astype(np.float64)
    Wb = cn.back_weights(tag, n, Lp, b)
    J = cn.jacobian_from_pre(z0, Wb)
    _, sv, Vt = np.linalg.svd(J)
    Wcache.append(Wb); z0s.append(z0); Vs.append(Vt.T); kst.append(cn.n_surv(sv, 0.01)); act.append(np.nonzero(z0 > 0)[0])

res = dict(n=n, c=c, Lp=Lp, nb=nb, P=P, kstar01_mean=float(np.mean(kst)), kstar01_median=float(np.median(kst)), rows=[])
for kind in ('relevant', 'random'):
    for rho in RHOS:
        for k in KS:
            w = cn.rng(tag, 10, k).standard_normal((P, k)) / np.sqrt(k)
            cols = []
            for b in range(nb):
                R = np.linalg.qr(cn.rng(tag, 11, b, k).standard_normal((k, k)))[0]
                if kind == 'relevant':
                    Vk = Vs[b][:, :k]
                else:
                    A = act[b]
                    Q = np.zeros((n, k))
                    kk = min(k, len(A))
                    Q[A, :kk] = np.linalg.qr(cn.rng(tag, 12, b, k).standard_normal((len(A), kk)))[0]
                    Vk = Q
                Z = z0s[b][None, :] + rho * np.linalg.norm(z0s[b]) * (w @ R.T) @ Vk.T
                cols.append(back_fwd(Z, Wcache[b]))
            X = np.concatenate(cols, axis=1)                      # (P, nb*n)
            Vloc = float(np.mean(X.var(0)))
            lam = np.linalg.eigvalsh(X @ X.T / X.size)[::-1]
            tail = np.concatenate([[lam.sum()], lam.sum() - np.cumsum(lam)])
            r = {str(e): int(np.nonzero(tail <= e * e * Vloc)[0][0]) if np.any(tail <= e * e * Vloc) else P + 1 for e in EPS}
            row = dict(kind=kind, rho=rho, k=k, Vloc=Vloc, EM2_over_V=float(lam.sum() / Vloc), r=r)
            res['rows'].append(row)
            print(kind, rho, k, r, f'EM2/V={row["EM2_over_V"]:.1f}', flush=True)
with open(os.path.join(HERE, '..', 'runs', f'crux_n{n}_c{c}_L{Lp}.json'), 'w') as fh:
    json.dump(res, fh, indent=1)
print('k* mean', res['kstar01_mean'])
