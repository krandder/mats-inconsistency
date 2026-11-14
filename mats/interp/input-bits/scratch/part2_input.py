"""Part 2 (PREREG Part 2), input regime: RQMC gains on the Part 1b nets, with the first-order residual decomposition.
usage: part2_input.py n seed [--R 32] [--P 16384] [--out file]
f(x) = a_L(x), x ~ N(0, I_n), sigma = 0, depths Db = {n/8, n/4, n/2, n, 32} (<= 2n), as part1b.
- MC variance V = tr Var f from P fresh inputs (seed [7301, seed, n]).
- First-order projection f1(x) = mean + sum_j sum_{k=1..4} c_{jk} He_k(x_j), coefficients c = E[f He_k(x_j)]/k! estimated on
  an independent pilot of P inputs (seed [7302, seed, n]); S1_hat = Var(f1)/V on the MC sample.
- RQMC (scipy scrambled Sobol', LMS + digital shift, x = Phi^-1(U), plain coordinates) at N = 2^8, 2^10, 2^12 with R
  replicates (seed [7303, seed, n, m, r]): G(N) = V / (N Var_rqmc(mean f)); G_res(N) = V_res / (N Var_rqmc(mean(f - f1)))
  with V_res = tr Var(f - f1) (MC); G_1(N) on f1 alone.
All depths share each forward pass."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--R', type=int, default=32); ap.add_argument('--P', type=int, default=16384)
ap.add_argument('--ms', default='8,10,12'); ap.add_argument('--out')
a = ap.parse_args()
n, seed, P, R = a.n, a.seed, a.P, a.R
ms = [int(x) for x in a.ms.split(',')]
t0 = time.time()
Db = sorted(L for L in {max(1, n // 8), n // 4, n // 2, n, 32} if L <= 2 * n)
Lmax = max(Db)
net = ib.Net(seed, n, Lmax)


def fwd_all(X, chunk=4096):
    out = {L: np.empty((len(X), n)) for L in Db}
    for i in range(0, len(X), chunk):
        h = X[i:i + chunk]
        for l in range(1, Lmax + 1):
            h = np.maximum(h @ net.W[l - 1].T, 0.0)
            if l in Db:
                out[l][i:i + chunk] = h
    return out


def herm_feats(X):
    return [ib.hermite(X, k) for k in (1, 2, 3, 4)]


# pilot: first-order coefficients (n_out x n per degree)
Xp = np.random.default_rng([7302, seed, n]).standard_normal((P, n))
Fp = fwd_all(Xp)
Hp = herm_feats(Xp)
coef = {}
for L in Db:
    Fc = Fp[L] - Fp[L].mean(0)
    coef[L] = (Fp[L].mean(0), [Fc.T @ Hp[k] / P / math.factorial(k + 1) for k in range(4)])


def f1(L, X):
    m, cs = coef[L]
    H = herm_feats(X)
    return m[None, :] + sum(H[k] @ cs[k].T for k in range(4))


Xm = np.random.default_rng([7301, seed, n]).standard_normal((P, n))
Fm = fwd_all(Xm)
res = dict(n=n, seed=seed, R=R, P=P, depths=Db, by_depth={})
for L in Db:
    F = Fm[L]; F1 = f1(L, Xm)
    V = float(np.sum(F.var(0, ddof=1)))
    Vres = float(np.sum((F - F1).var(0, ddof=1)))
    V1 = float(np.sum(F1.var(0, ddof=1)))
    us, _ = ib.univariate_shares(Xm, F)
    S1u = float(sum(us))
    # V_res includes the estimation noise of f1's coefficients (a first-order function, removed by RQMC); the
    # noise-free residual variance is V (1 - S1u) with S1u the unbiased two-halves first-order share.
    res['by_depth'][str(L)] = dict(V=V, V_res=Vres, V_f1=V1, S1_hat=V1 / V, S1_unb=S1u, W1_unb=us[0],
                                   res_share=Vres / V, rows=[])
for m in ms:
    N = 1 << m
    est = {L: [] for L in Db}; estr = {L: [] for L in Db}; est1 = {L: [] for L in Db}
    for r in range(R):
        Xq = ib.sobol_gauss(n, m, [7303, seed, n, m, r])
        Fq = fwd_all(Xq)
        for L in Db:
            F1q = f1(L, Xq)
            est[L].append(Fq[L].mean(0)); estr[L].append((Fq[L] - F1q).mean(0)); est1[L].append(F1q.mean(0))
    for L in Db:
        d = res['by_depth'][str(L)]
        vq = float(np.sum(np.array(est[L]).var(0, ddof=1)))
        vr = float(np.sum(np.array(estr[L]).var(0, ddof=1)))
        v1 = float(np.sum(np.array(est1[L]).var(0, ddof=1)))
        d['rows'].append(dict(N=N, G=d['V'] / N / vq, G_res=d['V_res'] / N / vr, G_res_true=d['V'] * (1 - d['S1_unb']) / N / vr,
                              G_f1=d['V_f1'] / N / max(v1, 1e-300), var_rqmc=vq, var_rqmc_res=vr))
        print(f'L={L} N={N}: G {d["V"] / N / vq:.2f}  G_res {d["V_res"] / N / vr:.2f}  S1_hat {d["S1_hat"]:.3f}  '
              f'G_res/(1-S1) {d["V_res"] / N / vr / (1 - d["S1_hat"]):.2f} | {time.time() - t0:.0f}s', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'])
