"""Amendment 8: sparsity measures for one noisy 2nd-argmax RNN (width sweep).
usage: python sweep_an.py MODEL.npz OUT.json
M1 mean-ablation DAG, M2 resample-ablation DAG, M3 algorithm-relative DAG (fractions of all edges, held-out flip <= 1%),
M4 uniform number of question directions per step (fraction of d), M5 share of edges whose ablation alone flips
>= 0.1% of answers (1,000 sampled edges). Edge set as in dag.py: rec (n-1) d^2, inp n d, out d.
"""
import sys, json, time
import numpy as np
import az, alg

t0 = time.time()
path, out = sys.argv[1], sys.argv[2]
W, r, sigma = az.load_npz(path)
W_hi, W_hh, W_oh = [np.asarray(w, float) for w in W]; r = np.asarray(r, float)
d = W_hi.shape[0]; n = W_oh.shape[0]
rng = np.random.default_rng(8080)
TAU = 0.01
NR, NI = (n - 1) * d * d, n * d
E = NR + NI + d
NFIT, NTEST = 2 ** 13, 2 ** 15


def clean(x, eps):
    B = x.shape[0]; h = np.zeros((B, d)); hs = [h]
    for t in range(n):
        h = np.maximum(h @ W_hh.T + x[:, t:t + 1] * W_hi, 0) + sigma * r[t] * eps[:, t]; hs.append(h)
    return h @ W_oh.T, hs


def split(keep):
    return keep[:NR].reshape(n - 1, d, d), keep[NR:NR + NI].reshape(n, d), keep[NR + NI:]


def run_masked(x, eps, keep, R, Rx):
    """R[t] (t = 1..n): replacement source values for h_t, array (B, d) or (d,); Rx (B, n) replacement inputs or None (0)."""
    kr, ki, ko = split(keep); B = x.shape[0]; h = np.zeros((B, d))
    for t in range(n):
        z = x[:, t:t + 1] * (W_hi * ki[t])
        if Rx is not None: z = z + Rx[:, t:t + 1] * (W_hi * ~ki[t])
        if t >= 1:
            K = kr[t - 1]; z = z + h @ (W_hh * K).T + R[t] @ (W_hh * ~K).T
        h = np.maximum(z, 0) + sigma * r[t] * eps[:, t]
    return h @ (W_oh * ko).T + R[n] @ (W_oh * ~ko).T


def errs(lo, x):
    return lo.argmax(1) != np.argsort(x, -1)[:, -2]


def saliency(x, eps, R, Rx):
    B = x.shape[0]; lo, hs = clean(x, eps); y = np.argsort(x, -1)[:, -2]
    lo2 = lo.copy(); lo2[np.arange(B), y] = -np.inf; jst = lo2.argmax(1)
    m = lo[np.arange(B), y] - lo[np.arange(B), jst]; s = 0.1 * m.std(); k = np.exp(-0.5 * (m / s) ** 2)
    g_h = W_oh[y] - W_oh[jst]
    sal_r = np.zeros((n - 1, d, d)); sal_i = np.zeros((n, d))
    sal_o = (k[:, None] * (g_h * (hs[n] - R[n])) ** 2).mean(0)
    for t in range(n - 1, -1, -1):
        zt = hs[t] @ W_hh.T + x[:, t:t + 1] * W_hi; g_z = g_h * (zt > 0)
        dx = x[:, t:t + 1] - (Rx[:, t:t + 1] if Rx is not None else 0)
        sal_i[t] = (k[:, None] * (g_z * dx * W_hi) ** 2).mean(0)
        if t >= 1:
            dev = hs[t] - R[t]
            sal_r[t - 1] = np.einsum('b,bi,bj->ij', k, g_z ** 2, dev ** 2) / B * W_hh ** 2
        g_h = g_z @ W_hh
    return np.concatenate([sal_r.ravel(), sal_i.ravel(), sal_o])


def dag_size(order, fixed, fit, test):
    """fit/test: (x, eps, R, Rx, base_err). Smallest prefix of `order` (plus `fixed` edges always kept) with fit flip
    <= TAU by binary search, then stepped up by 1% of the ablatable edges until the held-out flip <= TAU."""
    def flip(K, S):
        keep = fixed.copy(); keep[order[:K]] = True
        x, eps, R, Rx, be = S
        return float((errs(run_masked(x, eps, keep, R, Rx), x) != be).mean()), keep
    lo, hi = 0, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if flip(mid, fit)[0] <= TAU: hi = mid
        else: lo = mid + 1
    K = lo; steps = 0
    while True:
        ft, keep = flip(K, test)
        if ft <= TAU or K >= len(order): break
        K = min(len(order), K + max(1, len(order) // 100)); steps += 1
    return dict(K_fit=lo, K=int(K), kept_total=int(keep.sum()), frac=float(keep.sum() / E), test_flip=ft, step_ups=steps), keep


res = dict(model=path, sigma=sigma, d=d, n_edges=E)
xf = rng.standard_normal((NFIT, n)); ef = rng.standard_normal((NFIT, n, d))
xt = rng.standard_normal((NTEST, n)); et = rng.standard_normal((NTEST, n, d))
xm = rng.standard_normal((2 ** 15, n)); em = rng.standard_normal((2 ** 15, n, d))
lom, hsm = clean(xm, em); mu = [h.mean(0) for h in hsm]
lof, hsf = clean(xf, ef); lot, hst = clean(xt, et); bf, bt = errs(lof, xf), errs(lot, xt)
res['p_net'] = float(bt.mean())
res['noise_flip'] = float((errs(clean(xt, rng.standard_normal(et.shape))[0], xt) != bt).mean())
print('p_net %.4f noise_flip %.4f' % (res['p_net'], res['noise_flip']), flush=True)
allfalse = np.zeros(E, bool)

# M1 mean ablation
Rm = [None] + [np.broadcast_to(mu[t], (1, d)) for t in range(1, n + 1)]
order = np.argsort(-saliency(xf, ef, Rm, None))
res['M1_mean'], keep1 = dag_size(order, allfalse, (xf, ef, Rm, None, bf), (xt, et, Rm, None, bt))
print('M1', res['M1_mean'], '%.0fs' % (time.time() - t0), flush=True)

# Amendment 9: redundancy test (forbid M1's kept set), repeat searches on other fit samples, random-edge baseline
def flip_keep(keep, x, eps, R, Rx, be):
    return float((errs(run_masked(x, eps, keep, R, Rx), x) != be).mean())


comp = ~keep1
red = dict(flip_ablate_M1_set=flip_keep(comp, xt, et, Rm, None, bt))
if red['flip_ablate_M1_set'] <= TAU:
    order_c = order[comp[order]]
    info, keep_c = dag_size(order_c, allfalse, (xf, ef, Rm, None, bf), (xt, et, Rm, None, bt))
    red.update(exists=True, **{k: v for k, v in info.items()}, overlap_with_M1=int((keep_c & keep1).sum()))
else:
    red.update(exists=False)
res['R_redundancy'] = red
print('R', red, '%.0fs' % (time.time() - t0), flush=True)
reps_ = []
for rs in (8081, 8082):
    g2 = np.random.default_rng(rs); x2 = g2.standard_normal((NFIT, n)); e2 = g2.standard_normal((NFIT, n, d))
    b2 = errs(clean(x2, e2)[0], x2)
    o2 = np.argsort(-saliency(x2, e2, Rm, None))
    info, k2 = dag_size(o2, allfalse, (x2, e2, Rm, None, b2), (xt, et, Rm, None, bt))
    reps_.append((info, k2))
ks = [keep1] + [k2 for _, k2 in reps_]
jac = [float((ks[a] & ks[b]).sum() / max((ks[a] | ks[b]).sum(), 1)) for a in range(3) for b in range(a + 1, 3)]
res['S_stability'] = dict(sizes=[int(k.sum()) for k in ks], fracs=[float(k.sum() / E) for k in ks], jaccard=jac,
                          test_flips=[res['M1_mean']['test_flip']] + [i['test_flip'] for i, _ in reps_])
print('S', res['S_stability'], '%.0fs' % (time.time() - t0), flush=True)
info, _ = dag_size(rng.permutation(E), allfalse, (xf, ef, Rm, None, bf), (xt, et, Rm, None, bt))
res['B_random'] = info
print('B', info, '%.0fs' % (time.time() - t0), flush=True)

# M2 resample ablation: the source unit's clean value on another random input (a fixed permutation of the batch)
pf, pt = rng.permutation(NFIT), rng.permutation(NTEST)
Rf = [None] + [hsf[t][pf] for t in range(1, n + 1)]; Rt = [None] + [hst[t][pt] for t in range(1, n + 1)]
order = np.argsort(-saliency(xf, ef, Rf, xf[pf]))
res['M2_resample'], _ = dag_size(order, allfalse, (xf, ef, Rf, xf[pf], bf), (xt, et, Rt, xt[pt], bt))
print('M2', res['M2_resample'], '%.0fs' % (time.time() - t0), flush=True)

# M3 algorithm-relative ablation (CL conditional means fitted on 2^17 base samples; input edges always kept)
F_ = 7


def feats(M, S, xl):
    S = np.where(np.isfinite(S), S, 0.0)
    return np.stack([np.ones_like(M), M, S, xl, np.maximum(M, 0), np.maximum(S, 0), np.maximum(xl, 0)], -1)


xc = rng.standard_normal((2 ** 17, n)); hsc = []
for i0 in range(0, 2 ** 17, 2 ** 14):
    _, hh = clean(xc[i0:i0 + 2 ** 14], rng.standard_normal((2 ** 14, n, d))); hsc.append(hh)
Ac = alg.alg_states(xc); CL = {}
for t in range(1, n + 1):
    Ht = np.concatenate([hh[t] for hh in hsc]); key = (Ac['a'][:, t] + 1) * 11 + Ac['b'][:, t] + 1
    Fe = feats(Ac['M'][:, t], Ac['S'][:, t], xc[:, t - 1]); CL[t] = {}
    for kk in np.unique(key):
        s = key == kk; CL[t][int(kk)] = np.linalg.lstsq(Fe[s], Ht[s], rcond=None)[0]


def cond_means(x):
    A = alg.alg_states(x); H = [None]
    for t in range(1, n + 1):
        key = (A['a'][:, t] + 1) * 11 + A['b'][:, t] + 1; Fe = feats(A['M'][:, t], A['S'][:, t], x[:, t - 1])
        Ht = np.zeros((x.shape[0], d))
        for kk in np.unique(key):
            s = key == kk; Ht[s] = Fe[s] @ CL[t].get(int(kk), np.zeros((F_, d)))
        H.append(Ht)
    return H


Hf, Ht_ = cond_means(xf), cond_means(xt)
sal = saliency(xf, ef, Hf, None); fixed = np.zeros(E, bool); fixed[NR:NR + NI] = True
abl = np.flatnonzero(~fixed); order = abl[np.argsort(-sal[abl])]
res['M3_alg'], _ = dag_size(order, fixed, (xf, ef, Hf, None, bf), (xt, et, Ht_, None, bt))
res['M3_alg']['frac_ablatable'] = float((res['M3_alg']['K']) / len(abl))
res['alg_readout_err'] = float(errs(Ht_[n] @ W_oh.T, xt).mean())
print('M3', res['M3_alg'], '%.0fs' % (time.time() - t0), flush=True)

# M4 question directions: uniform rank k per step (lowrank.py's question subspaces)
y = np.argsort(xm, -1)[:, -2]; lo2 = lom.copy(); lo2[np.arange(len(y)), y] = -np.inf; jst = lo2.argmax(1)
m = lom[np.arange(len(y)), y] - lom[np.arange(len(y)), jst]; s = 0.1 * m.std(); kw = np.exp(-0.5 * (m / s) ** 2)
G = [None] * (n + 1); G[n] = W_oh[y] - W_oh[jst]
for t in range(n - 1, 0, -1):
    zt = hsm[t] @ W_hh.T + xm[:, t:t + 1] * W_hi; G[t] = (G[t + 1] * (zt > 0)) @ W_hh
proj = {}
for t in range(1, n + 1):
    C = np.cov(hsm[t].T); C = C + (1e-6 * np.trace(C) / d + 1e-12) * np.eye(d)
    ev, U = np.linalg.eigh(C); Ch = U @ np.diag(np.sqrt(np.maximum(ev, 0))) @ U.T; Cih = U @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-300))) @ U.T
    Fm = (kw[:, None, None] * G[t][:, :, None] * G[t][:, None, :]).mean(0)
    qe, V = np.linalg.eigh(Ch @ Fm @ Ch); V = V[:, ::-1]
    proj[t] = (Ch, V, Cih)


def run_proj(x, eps, kk):
    B = x.shape[0]; h = np.zeros((B, d))
    for t in range(n):
        h = np.maximum(h @ W_hh.T + x[:, t:t + 1] * W_hi, 0) + sigma * r[t] * eps[:, t]
        Ch, V, Cih = proj[t + 1]; Pm = Ch @ V[:, :kk] @ V[:, :kk].T @ Cih
        h = mu[t + 1] + (h - mu[t + 1]) @ Pm.T
    return h @ W_oh.T


lo_, hi_ = 1, d
while lo_ < hi_:
    mid = (lo_ + hi_) // 2
    if (errs(run_proj(xf, ef, mid), xf) != bf).mean() <= TAU: hi_ = mid
    else: lo_ = mid + 1
kk = lo_
while True:
    ft = float((errs(run_proj(xt, et, kk), xt) != bt).mean())
    if ft <= TAU or kk >= d: break
    kk += 1
res['M4_dirs'] = dict(k_fit=lo_, k=kk, frac=kk / d, test_flip=ft)
print('M4', res['M4_dirs'], '%.0fs' % (time.time() - t0), flush=True)

# M5 single-edge mean ablations on 1,000 uniformly sampled edges, 8,192 inputs
xs, es_ = xf, ef; hs5 = hsf; b5 = bf; NS = len(xs)
sample = rng.choice(E, size=1000, replace=False); flips = []
for e in sample:
    if e >= NR + NI:                                             # readout unit i
        i = e - NR - NI; lo = hs5[n] @ W_oh.T + np.outer(mu[n][i] - hs5[n][:, i], W_oh[:, i])
    else:
        if e < NR:
            t = e // (d * d) + 1; i = (e // d) % d; j = e % d       # edge h_t,j -> z_{t+1},i
            z = hs5[t] @ W_hh.T + xs[:, t:t + 1] * W_hi; z[:, i] += W_hh[i, j] * (mu[t][j] - hs5[t][:, j])
        else:
            t = (e - NR) // d; i = (e - NR) % d                     # input x_t -> z_{t+1},i
            z = hs5[t] @ W_hh.T + xs[:, t:t + 1] * W_hi; z[:, i] -= xs[:, t] * W_hi[i]
        h = np.maximum(z, 0) + sigma * r[t] * es_[:, t]
        for tt in range(t + 1, n):
            h = np.maximum(h @ W_hh.T + xs[:, tt:tt + 1] * W_hi, 0) + sigma * r[tt] * es_[:, tt]
        lo = h @ W_oh.T
    flips.append(int((errs(lo, xs) != b5).sum()))
flips = np.array(flips)
kind = np.where(sample < NR, 'rec', np.where(sample < NR + NI, 'inp', 'out'))
res['M5_single'] = dict(n_sampled=1000, n_inputs=NS, share_ge_0p1pct=float((flips >= 0.001 * NS).mean()),
                        share_any=float((flips >= 1).mean()), se=float(np.sqrt((flips >= 0.001 * NS).mean() * (1 - (flips >= 0.001 * NS).mean()) / 1000)),
                        by_kind={k: dict(n=int((kind == k).sum()), share_ge_0p1pct=float((flips[kind == k] >= 0.001 * NS).mean()) if (kind == k).any() else None) for k in ('rec', 'inp', 'out')})
print('M5', res['M5_single'], '%.0fs' % (time.time() - t0), flush=True)
res['seconds'] = time.time() - t0
json.dump(res, open(out, 'w'), indent=1)
print('done %.0fs' % res['seconds'])
