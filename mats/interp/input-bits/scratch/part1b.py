"""Part 1b (PREREG Part 1): interaction structure of the input-to-output map over input coordinates, one net, sigma = 0.
usage: part1b.py n seed [--P 16384] [--PJ 1024] [--m 64] [--out file]
Depths Db = {n/8, n/4, n/2, n, 32} (those <= 2n). Per depth L, on f(x) = a_L(x), x ~ N(0, I_n):
- Hermite shares (P Gaussian inputs, two halves): univariate degrees 1-4 (W1 = degree 1; S1 = sum = first-order share up
  to degree 4) for a and for |a|; degree-2 share (diagonal / off-diagonal) of a (all outputs if n <= 128, else 64 seeded
  outputs) and of the scalar |a|^2;
- Jansen total indices T_j on all coordinates (n <= 64) or m seeded coordinates (scaled): mean dimension D = sum_j T_j;
- even share s_even (x vs -x);
- rotated coordinates: y = V^T x, V = eigenvectors of M = E[|a|^2 (x x^T - I)] from an independent pilot of P inputs;
  univariate shares and off-diagonal degree-2 share of a in y-coordinates (same sample as the unrotated ones).
Seeds: Hermite [7011, seed, n], Jansen [7012, seed, n], pilot [7013, seed, n], even [7014, seed, n], subsets [7015, seed, n]."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=16384); ap.add_argument('--PJ', type=int, default=1024)
ap.add_argument('--m', type=int, default=64); ap.add_argument('--out')
a = ap.parse_args()
n, seed, P = a.n, a.seed, a.P
t0 = time.time()
Db = sorted(L for L in {max(1, n // 8), n // 4, n // 2, n, 32} if L <= 2 * n)
Lmax = max(Db)
net = ib.Net(seed, n, Lmax)


def fwd_all(X):
    """Outputs at every depth in Db: dict L -> a_L(X)."""
    out = {}
    h = X
    for l in range(1, Lmax + 1):
        h = np.maximum(h @ net.W[l - 1].T, 0.0)
        if l in Db:
            out[l] = h
    return out


sub = np.random.default_rng([7015, seed, n])
outs_deg2 = None if n <= 128 else np.sort(sub.choice(n, 64, replace=False))
coords = np.arange(n) if n <= 64 else np.sort(sub.choice(n, a.m, replace=False))
res = dict(n=n, seed=seed, P=P, PJ=a.PJ, depths=Db, coords=coords.tolist(),
           deg2_outputs=None if outs_deg2 is None else outs_deg2.tolist(), by_depth={})
X = np.random.default_rng([7011, seed, n]).standard_normal((P, n))
FA = fwd_all(X)
Xp = np.random.default_rng([7013, seed, n]).standard_normal((P, n))
FP = fwd_all(Xp)
Xe = np.random.default_rng([7014, seed, n]).standard_normal((P // 2, n))
FE, FEm = fwd_all(Xe), fwd_all(-Xe)
for L in Db:
    F = FA[L]
    r = {}
    us, V = ib.univariate_shares(X, F)
    r.update(V=V, W1=us[0], univ=us, S1=sum(us))
    nrm = np.linalg.norm(F, axis=1, keepdims=True)
    usA, VA = ib.univariate_shares(X, nrm)
    r.update(amp_W1=usA[0], amp_S1=sum(usA), amp_univ=usA)
    d2 = ib.deg2_share(X, F, outputs=outs_deg2)
    r.update(W2=d2['W2'], W2_diag=d2['W2_diag'], W2_off=d2['W2_off'])
    d2A = ib.deg2_share(X, nrm ** 2)
    r.update(sq_W2=d2A['W2'], sq_W2_off=d2A['W2_off'])
    # even share
    Ev = 0.5 * (FE[L] + FEm[L])
    Vv = float(np.sum(np.concatenate([FE[L], FEm[L]]).var(0, ddof=1)))
    r['s_even'] = float(np.sum(Ev.var(0, ddof=1))) / Vv
    # rotation from the pilot
    Fp = FP[L]
    q = np.sum(Fp * Fp, 1)
    qc = q - q.mean()
    M = (Xp * qc[:, None]).T @ Xp / P
    ev, Vec = np.linalg.eigh(M)
    order = np.argsort(-np.abs(ev))
    Vec = Vec[:, order]
    Y = X @ Vec
    usR, _ = ib.univariate_shares(Y, F)
    d2R = ib.deg2_share(Y, F, outputs=outs_deg2)
    r.update(rot_W1=usR[0], rot_S1=sum(usR), rot_univ=usR, rot_W2_off=d2R['W2_off'], rot_W2=d2R['W2'],
             rot_eig_top=[float(e) for e in ev[order][:8]])
    res['by_depth'][str(L)] = r
    print(f'L={L}: W1 {r["W1"]:.3f} S1 {r["S1"]:.3f} W2 {r["W2"]:.3f} (off {r["W2_off"]:.3f}) | amp W1 {r["amp_W1"]:.3f} '
          f'S1 {r["amp_S1"]:.3f} | s_even {r["s_even"]:.3f} | rot S1 {r["rot_S1"]:.3f} W2off {r["rot_W2_off"]:.3f} | '
          f'{time.time() - t0:.0f}s', flush=True)
# Jansen total indices (all depths at once)
rj = np.random.default_rng([7012, seed, n])
A_ = rj.standard_normal((a.PJ, n)); B_ = rj.standard_normal((a.PJ, n))
fA = fwd_all(A_)
Tsum = {L: [] for L in Db}
VJ = {L: float(np.sum(np.concatenate([fA[L], fwd_all(B_)[L]]).var(0, ddof=1))) for L in Db}
for j in coords:
    AB = A_.copy(); AB[:, j] = B_[:, j]
    fAB = fwd_all(AB)
    for L in Db:
        Tsum[L].append(0.5 * np.sum((fA[L] - fAB[L]) ** 2, 1).mean() / VJ[L])
for L in Db:
    T = np.array(Tsum[L])
    scale = n / len(coords)
    r = res['by_depth'][str(L)]
    r.update(D=float(scale * T.sum()), D_se=float(scale * math.sqrt(len(T)) * T.std(ddof=1)) if len(coords) < n else None,
             T_max=float(T.max()), T_min=float(T.min()))
    print(f'L={L}: mean dimension D {r["D"]:.2f} (se {r["D_se"]})', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
