"""Response kernels of the stationary noisy network along covariant Lyapunov directions, and the flip-response.
Model S (sigma = 1), nets st.W_layer(seed, n, l). P independent stationary paths per net (burn-in 12n layers from 0).
Injection layer l0 = 0. Same noise and weights after the injection for every copy (noise regenerated per layer).
Pass 1: base path with QR (K tangent directions) from -2n (warm-up) to Jmax + Tf, storing R and the base states.
Pass 2: Ginelli backward iteration -> covariant Lyapunov vectors (CLV) v_k at layer 0, k = 1..K.
Pass 3: perturbed copies h'_0 = h_0 + eps |h_0| v_k for k in KSET and eps in (1e-7 [linear], 0.3/sqrt(n), 3/sqrt(n));
  flip copies: at layer 1 one unit with |z_1i| < |w_i| (decided by one noise SD) gets its gate flipped
  (a_i = 0 if on, a_i = z_i if off); a control flip of a unit with |z_1i| in [3, 6] noise SDs.
Recorded for t = 1..Jmax: mean over paths of r_t^2 = |h'_t - h_t|^2/|h_t|^2, its radial part ((|h'|-|h|)/|h|)^2 and
angular part 2(1 - cos), the number of differing gates g_t; for flips also per-path lifetimes (last t with g_t > 0,
and the first t with r_t^2 below 1e-2 r_1^2) and cascade sizes sum_t g_t; the tangent exponents of the CLVs.
usage: response_net.py n seed --P 64 --out file"""
import sys, os, math, json, time, argparse
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import numpy as np
import st

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=64); ap.add_argument('--out', required=True)
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
K = min(24, max(2, n // 4))  # beyond ~n/4 directions, rank loss (few active units) kills CLVs at small n
KSET = [k for k in (1, 2, 3, 4, 6, 8, 12, 16, 24) if k <= K]
EPS = [1e-7, 0.3 / math.sqrt(n), 3.0 / math.sqrt(n)]
Jmax = 2 * n
Tf = 2 * n
t0 = time.time()
Wf = st.WCache(seed, n, max_bytes=600e6)


def noise(l):
    return np.random.default_rng([4747, seed, n, l + st.OFF]).standard_normal((P, n))


# burn-in
h = np.zeros((P, n))
for l in range(-14 * n, -2 * n + 1):
    h = np.maximum(h @ Wf(l).T, 0.0) + noise(l)
# pass 1: QR from -2n+1 .. Jmax+Tf
Q = np.linalg.qr(np.random.default_rng([4848, seed, n]).standard_normal((P, n, K)))[0]
Rs = {}
H = {}
Q0 = None
logdiag = np.zeros((P, K)); cnt = 0
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
        logdiag += np.log(np.maximum(np.abs(np.einsum('pkk->pk', R)), 1e-300)); cnt += 1
    if l == 0:
        Q0 = Q.copy()
    if 0 <= l <= Jmax:
        H[l] = h.copy()
lyap = (logdiag / cnt).mean(0)
# pass 2: Ginelli backward iteration
C = np.triu(np.random.default_rng([4949, seed, n]).standard_normal((P, K, K)))
rankloss = np.zeros(P, bool)
for l in range(Jmax + Tf, 0, -1):
    Rl = Rs[l].copy()
    dg = np.einsum('pkk->pk', Rl)
    scale = np.maximum(np.abs(Rl).max(axis=(1, 2)), 1e-300)  # an all-off layer gives R = 0 (bug fix 1)
    small = np.abs(dg) < 1e-10 * scale[:, None]
    rankloss[small.any(1)] = True
    idx = np.where(small)
    Rl[idx[0], idx[1], idx[1]] = np.where(dg[small] >= 0, 1, -1) * 1e-10 * scale[idx[0]]
    C = np.linalg.solve(Rl, C)
    C = np.triu(C)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
V = np.einsum('pnk,pkj->pnj', Q0, C)  # CLVs at layer 0, (P, n, K), unit columns
V /= np.linalg.norm(V, axis=1, keepdims=True)
del Rs
# pass 3: copies
h0 = H[0]
r0 = np.linalg.norm(h0, axis=1, keepdims=True)
copies = []
names = []
for k in KSET:
    for e in EPS:
        copies.append(h0 + e * r0 * V[:, :, k - 1]); names.append(('clv', k, e))
copies.append(h0.copy()); names.append(('flip', 'nd', 0))
copies.append(h0.copy()); names.append(('flip', 'far', 0))
C3 = np.stack(copies)  # (M, P, n)
M = C3.shape[0]
rec = {nm: dict(r2=np.zeros(Jmax + 1), rad=np.zeros(Jmax + 1), ang=np.zeros(Jmax + 1), g=np.zeros(Jmax + 1)) for nm in names}
TS = sorted(set(int(round(x)) for x in np.geomspace(1, Jmax, 40)))
lr2 = {nm: np.zeros((P, len(TS))) for nm in names}  # per-path log10 r_t^2 at log-spaced t
gpath = {nm: np.zeros((P, Jmax + 1)) for nm in names if nm[0] == 'flip'}
r2path = {nm: np.zeros((P, Jmax + 1)) for nm in names if nm[0] == 'flip'}
flip_info = {}
for l in range(1, Jmax + 1):
    W = Wf(l)
    xi = noise(l)
    hb = H[l]
    zb = H[l - 1] @ W.T
    Z = C3 @ W.T
    A = np.maximum(Z, 0.0)
    if l == 1:
        for mi, nm in enumerate(names):
            if nm[0] != 'flip':
                continue
            wn = np.linalg.norm(W, axis=1)[None]
            marg = np.abs(zb) / wn
            if nm[1] == 'nd':
                cand = marg < 1.0
            else:
                cand = (marg > 3.0) & (marg < 6.0)
            rngf = np.random.default_rng([5050, seed, n, 1 if nm[1] == 'nd' else 2])
            pick = np.array([rngf.choice(np.where(c)[0]) if c.any() else int(np.argmin(np.abs(m - (0.5 if nm[1] == 'nd' else 4.5))))
                             for c, m in zip(cand, marg)])
            idx = (np.arange(P), pick)
            zz = Z[mi][idx]
            A[mi][idx] = np.where(zz > 0, 0.0, zz)
            flip_info[nm[1]] = dict(margin_sd=marg[idx].tolist(), had_candidate=cand.any(1).tolist())
    C3 = A + xi[None]
    gb = zb > 0
    nb = np.linalg.norm(hb, axis=1)
    for mi, nm in enumerate(names):
        d = C3[mi] - hb
        r2 = np.sum(d * d, 1) / nb ** 2
        nc = np.linalg.norm(C3[mi], axis=1)
        rad = ((nc - nb) / nb) ** 2
        cos = np.sum(C3[mi] * hb, 1) / (nc * nb)
        g = np.sum((Z[mi] > 0) != gb, 1)
        rec[nm]['r2'][l] += r2.mean(); rec[nm]['rad'][l] += rad.mean(); rec[nm]['ang'][l] += (2 * (1 - cos)).mean()
        rec[nm]['g'][l] += g.mean()
        if l in TS:
            lr2[nm][:, TS.index(l)] = np.log10(np.maximum(r2, 1e-300))
        if nm[0] == 'flip':
            gpath[nm][:, l] = g
            r2path[nm][:, l] = r2
out = dict(n=n, seed=seed, P=P, K=K, KSET=KSET, EPS=EPS, Jmax=Jmax, lyap_times_n=(lyap * n).tolist(), curves={},
           rankloss_fraction=float(rankloss.mean()))
for nm in names:
    key = '%s_%s_%g' % nm
    out['curves'][key] = {k2: v.tolist() for k2, v in rec[nm].items()}
    out['curves'][key]['log10r2_median'] = np.median(lr2[nm], 0).tolist()
    out['curves'][key]['log10r2_q25'] = np.quantile(lr2[nm], 0.25, axis=0).tolist()
    out['curves'][key]['log10r2_q75'] = np.quantile(lr2[nm], 0.75, axis=0).tolist()
out['TS'] = TS
for nm in gpath:
    g = gpath[nm]; r2 = r2path[nm]
    life = np.array([int(np.max(np.where(row > 0)[0])) if (row > 0).any() else 0 for row in g])
    thr = 1e-2 * r2[:, 1:2]
    below = r2 < thr
    below[:, 0] = False
    lif2 = np.array([int(np.argmax(b)) if b.any() else Jmax + 1 for b in below])
    out['flip_' + nm[1]] = dict(lifetime_gates=life.tolist(), lifetime_r2=lif2.tolist(), cascade=g.sum(1).tolist(),
                               r2_1=r2[:, 1].tolist(), **flip_info.get(nm[1], {}))
out['sec'] = time.time() - t0
json.dump(out, open(a.out, 'w'))
print(json.dumps(dict(n=n, seed=seed, sec=round(out['sec'], 1), lyap=[round(x, 2) for x in out['lyap_times_n'][:6]],
                      flip_nd_mean_cascade=float(np.mean(out['flip_nd']['cascade'])),
                      flip_nd_median_life=float(np.median(out['flip_nd']['lifetime_gates'])))), flush=True)
