"""Linear response along covariant Lyapunov vectors by exact tangent propagation (post hoc correction of the
finite-difference "linear" copies of response_net.py, which hit a double-precision floor near |dh|/|h| ~ 1e-8).
Same nets, noise and CLV construction as response_net.py (bug-fixed version). For each path and CLV k:
v_0 = CLV_k at layer 0, v_t = D_t W_t v_{t-1} along the base path's own gates (log norms accumulated, no underflow);
relative linear response ln R(t) = 2 ln(|v_t| / |v_0|) - 2 ln(|h_t| / |h_0|). Paths with rank loss in the Ginelli
window are flagged; statistics are given over surviving paths only.
usage: tangent_response.py n seed --P 64 --out file"""
import sys, os, math, json, time, argparse
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import numpy as np
import st

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=64); ap.add_argument('--out', required=True)
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
K = min(24, max(2, n // 4))
Jmax, Tf = 2 * n, 2 * n
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=600e6)


def noise(l):
    return np.random.default_rng([4747, seed, n, l + st.OFF]).standard_normal((P, n))


h = np.zeros((P, n))
for l in range(-14 * n, -2 * n + 1):
    h = np.maximum(h @ Wf(l).T, 0.0) + noise(l)
Q = np.linalg.qr(np.random.default_rng([4848, seed, n]).standard_normal((P, n, K)))[0]
Rs = {}; H = {}; Dg = {}
Q0 = None
for l in range(-2 * n + 1, Jmax + Tf + 1):
    W = Wf(l)
    z = h @ W.T
    Dm = (z > 0).astype(float)
    h = np.maximum(z, 0.0) + noise(l)
    Y = Dm[:, :, None] * np.matmul(W[None], Q)
    Q, R = np.linalg.qr(Y)
    sg = np.sign(np.einsum('pkk->pk', R)); sg[sg == 0] = 1
    Q = Q * sg[:, None, :]; R = R * sg[:, :, None]
    if l >= 1:
        Rs[l] = R
    if l == 0:
        Q0 = Q.copy()
    if 0 <= l <= Jmax:
        H[l] = np.linalg.norm(h, axis=1)
        if l >= 1:
            Dg[l] = Dm
rankloss = np.zeros(P, bool)
C = np.triu(np.random.default_rng([4949, seed, n]).standard_normal((P, K, K)))
for l in range(Jmax + Tf, 0, -1):
    Rl = Rs[l].copy()
    dg = np.einsum('pkk->pk', Rl)
    scale = np.maximum(np.abs(Rl).max(axis=(1, 2)), 1e-300)
    small = np.abs(dg) < 1e-10 * scale[:, None]
    rankloss[small.any(1)] = True
    idx = np.where(small)
    Rl[idx[0], idx[1], idx[1]] = np.where(dg[small] >= 0, 1, -1) * 1e-10 * scale[idx[0]]
    C = np.triu(np.linalg.solve(Rl, C))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
V = np.einsum('pnk,pkj->pnj', Q0, C)
V /= np.linalg.norm(V, axis=1, keepdims=True)
logv = np.zeros((P, K))
lnR = np.zeros((Jmax + 1, P, K))
for t in range(1, Jmax + 1):
    W = Wf(t)
    V = Dg[t][:, :, None] * np.matmul(W[None], V)
    nv = np.linalg.norm(V, axis=1)
    dead = nv <= 0
    logv += np.log(np.where(dead, 1e-300, nv))
    V = V / np.where(dead, 1.0, nv)[:, None, :]
    lnR[t] = 2 * logv - 2 * np.log(H[t] / H[0])[:, None]
ok = ~rankloss
TS = sorted(set(int(round(x)) for x in np.geomspace(1, Jmax, 40)))
out = dict(n=n, seed=seed, P=P, K=K, Jmax=Jmax, TS=TS, rankloss_fraction=float(rankloss.mean()), n_ok=int(ok.sum()),
           lnR_median={str(k): [float(np.median(lnR[t, ok, k - 1])) if ok.any() else None for t in TS] for k in range(1, K + 1)},
           lnR_q25={str(k): [float(np.quantile(lnR[t, ok, k - 1], 0.25)) if ok.any() else None for t in TS] for k in range(1, K + 1)},
           lnR_q75={str(k): [float(np.quantile(lnR[t, ok, k - 1], 0.75)) if ok.any() else None for t in TS] for k in range(1, K + 1)},
           lnR_at_n_mean_exp={str(k): float(np.log(np.mean(np.exp(lnR[n, ok, k - 1])))) if ok.any() else None for k in range(1, K + 1)},
           sec=time.time() - t0)
json.dump(out, open(a.out, 'w'))
print(json.dumps(dict(n=n, seed=seed, rankloss=out['rankloss_fraction'], sec=round(out['sec'], 1))), flush=True)
