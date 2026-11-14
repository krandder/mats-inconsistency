"""Question subspaces: how many directions of h_t does the question need?

For each step t: C_t = Cov(h_t), F_t = E[k(m) g g^T] with g = dm/dh_t (m = target margin, k = near-boundary weight,
as in dag.py). Question directions: top eigenvectors of C^{1/2} F C^{1/2} (variance x question sensitivity);
forward-only directions: top eigenvectors of C (PCA). A rank-k description keeps h' = mu + P_k (h - mu) at every step
(P_k the oblique projection onto the kept directions in whitened coordinates), run on the same inputs and noise.
usage: python lowrank_avg.py MODEL.npz OUT.json  (average-question variant, PREREG Amendment 4/4b)
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


def run_proj(x, eps, kind, kk):
    B = x.shape[0]; h = np.zeros((B, d))
    for t in range(n):
        z = h @ net.W_hh.T + x[:, t:t + 1] * net.W_hi
        h = np.maximum(z, 0) + sigma * r[t] * eps[:, t]
        Pm = proj[t + 1][kind][kk]
        h = mu[t + 1] + (h - mu[t + 1]) @ Pm.T
    return h @ net.W_oh.T


yt = np.argsort(xt, -1)[:, -2]
lo_t, _, _ = net.run(xt, et)


def ce_err(lo, x):
    yy = np.argsort(x, -1)[:, -2]
    mm = lo.max(1, keepdims=True); lse = mm[:, 0] + np.log(np.exp(lo - mm).sum(1))
    return lse - lo[np.arange(len(x)), yy], (lo.argmax(1) != yy).astype(float)


xq = np.sort(np.random.default_rng(1).standard_normal((2 ** 20, n)), 1); GQ = np.quantile(xq[:, -1] - xq[:, -2], [0.25, 0.5, 0.75])
srt = np.sort(xt, 1); gap = srt[:, -1] - srt[:, -2]; a_ = np.argmax(xt, 1); b_ = yt
grp = lambda p: np.where(p <= 2, 0, np.where(p <= 6, 1, 2))
St = [np.digitize(gap, GQ) == q for q in range(4)] + [(grp(a_) == i) & (grp(b_) == j) for i in range(3) for j in range(3)]
CEt, ERt = ce_err(lo_t, xt)
sd = dict(dL=CEt.std(), dp=ERt.std()); sdk = dict(dL=[CEt[s_].std() for s_ in St], dp=[ERt[s_].std() for s_ in St]); pik = [s_.mean() for s_ in St]
res = dict(model=path, sigma=sigma, p_net=float(ERt.mean()), L_net=float(CEt.mean()), rows=[])
for kind in ('q', 'f'):
    for kk in range(1, d + 1):
        ce, er = ce_err(run_proj(xt, et, kind, kk), xt)
        row = dict(kind=kind, rank=kk, flip=float((er != ERt).mean()))
        for tg, dv in (('dL', ce - CEt), ('dp', er - ERt)):
            row[tg] = float(dv.mean()); row[tg + '_k'] = [float(dv[s_].mean()) for s_ in St]
            for NL in (10, 14, 18):
                T = sd[tg] / np.sqrt(2 ** NL); Tk = [sdk[tg][k] / np.sqrt(2 ** NL * pik[k]) for k in range(len(St))]
                row[f'{tg}_pass_{NL}'] = bool(abs(row[tg]) <= T and all(abs(row[tg + '_k'][k]) <= Tk[k] for k in range(len(St))))
        res['rows'].append(row)
        print(kind, kk, 'flip %.4f dp %+.5f dL %+.5f' % (row['flip'], row['dp'], row['dL']), {k: v for k, v in row.items() if 'pass' in k}, flush=True)
for kind in ('q', 'f'):
    for tg in ('dp', 'dL'):
        for NL in (10, 14, 18):
            ks = [r_['rank'] for r_ in res['rows'] if r_['kind'] == kind and r_[f'{tg}_pass_{NL}']]
            res[f'min_rank_{kind}_{tg}_{NL}'] = min(ks) if ks else None
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print({k: v for k, v in res.items() if k.startswith('min_rank')})
