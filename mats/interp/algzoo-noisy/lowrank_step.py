"""Question subspaces: how many directions of h_t does the question need?

For each step t: C_t = Cov(h_t), F_t = E[k(m) g g^T] with g = dm/dh_t (m = target margin, k = near-boundary weight,
as in dag.py). Question directions: top eigenvectors of C^{1/2} F C^{1/2} (variance x question sensitivity);
forward-only directions: top eigenvectors of C (PCA). A rank-k description keeps h' = mu + P_k (h - mu) at every step
(P_k the oblique projection onto the kept directions in whitened coordinates), run on the same inputs and noise.
usage: python lowrank_step.py MODEL.npz OUT.json  (rank reduced at one step at a time; review 1, finding 3)
"""
import sys, json, time
import numpy as np
import az, dag

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
W, r, sigma = az.load_npz(path)
net = dag.Net(W, r, sigma); d, n = net.d, net.n
rng = np.random.default_rng(31337)
Nf, Nt = 2 ** 15, 2 ** 16
xf = rng.standard_normal((Nf, n)); ef = rng.standard_normal((Nf, n, d))
xt = rng.standard_normal((Nt, n)); et = rng.standard_normal((Nt, n, d))
lo, hs, _ = net.run(xf, ef)
y = np.argsort(xf, -1)[:, -2]
lo2 = lo.copy(); lo2[np.arange(Nf), y] = -np.inf; jst = lo2.argmax(1)
m = lo[np.arange(Nf), y] - lo[np.arange(Nf), jst]; s = 0.1 * m.std(); k = np.exp(-0.5 * (m / s) ** 2)
g = net.W_oh[y] - net.W_oh[jst]
G = [None] * (n + 1); G[n] = g
for t in range(n - 1, 0, -1):
    zt = hs[t] @ net.W_hh.T + xf[:, t:t + 1] * net.W_hi
    gz = G[t + 1] * (zt > 0); G[t] = gz @ net.W_hh
mu = [h.mean(0) for h in hs]
proj = {}
for t in range(1, n + 1):
    C = np.cov(hs[t].T) + 1e-12 * np.eye(d)
    ev, U = np.linalg.eigh(C); Ch = U @ np.diag(np.sqrt(np.maximum(ev, 0))) @ U.T; Cih = U @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-300))) @ U.T
    F = (k[:, None, None] * G[t][:, :, None] * G[t][:, None, :]).mean(0)
    qe, V = np.linalg.eigh(Ch @ F @ Ch); V = V[:, ::-1]; qe = qe[::-1]
    pe, P = np.linalg.eigh(C); P = P[:, ::-1]; pe = pe[::-1]
    proj[t] = dict(q=[Ch @ V[:, :kk] @ V[:, :kk].T @ Cih for kk in range(d + 1)],
                   f=[P[:, :kk] @ P[:, :kk].T for kk in range(d + 1)],
                   q_spec=(qe / qe.sum()).tolist(), f_spec=(pe / pe.sum()).tolist())


def run_proj_step(x, eps, kind, kk, tstar):
    B = x.shape[0]; h = np.zeros((B, d))
    for t in range(n):
        z = h @ net.W_hh.T + x[:, t:t + 1] * net.W_hi
        h = np.maximum(z, 0) + sigma * r[t] * eps[:, t]
        if t + 1 == tstar:
            Pm = proj[t + 1][kind][kk]
            h = mu[t + 1] + (h - mu[t + 1]) @ Pm.T
    return h @ net.W_oh.T


yt = np.argsort(xt, -1)[:, -2]
lo_t, _, _ = net.run(xt, et); err_t = lo_t.argmax(1) != yt
res = dict(model=path, sigma=sigma, p_net=float(err_t.mean()), rows=[], min_rank={})
for kind in ('q', 'f'):
    for tstar in range(1, n + 1):
        mins = None
        for kk in range(1, d + 1):
            es = run_proj_step(xt, et, kind, kk, tstar).argmax(1) != yt
            fl = float((es != err_t).mean()); dp = float(es.mean() - err_t.mean())
            res['rows'].append(dict(kind=kind, step=tstar, rank=kk, flip=fl, dp=dp))
            if mins is None and fl <= 0.01: mins = kk
        res['min_rank'][f'{kind}_{tstar}'] = mins
        print(kind, tstar, 'min rank for flip <= 1%:', mins, flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
