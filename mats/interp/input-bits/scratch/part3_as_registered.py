"""Part 3 (PREREG Part 3): structured samplers for the input regime, MSE vs compute, one net and depth, sigma = 0.
usage: part3.py n seed L [--R 12] [--mmax 12] [--Pp 4096] [--out file]
All estimators are unbiased, so MSE = variance; nMSE = variance / V (V = tr Var_x a). Cost in mv (one n x n mat-vec);
a forward pass is L mv. Methods (seeds [7501.., seed, n, L, ...]):
  MC       : V / N at cost N L (analytic, V from P = 16384 fresh inputs).
  RBMC     : V_sph / N (radius exact).
  RQMC     : scrambled Sobol' in the n Gaussian coordinates, N = 2^m, R replicates.
  RQMC-RB  : the same points normalised to the sphere, m_n g(u) (the same forward pass, by homogeneity).
  RQMC-ROT : radius exact, coordinates rotated x = V y with V the eigenvectors (by |eigenvalue|) of
             M = E[|a|^2 (x x^T - I)] from a pilot of Pp inputs; cost N (L + 1) + Pp L + n/2 (pilot charged).
  ML2      : two-level shared-trajectory sampler (unrestricted lane's mlmc.py), prefix r chosen from the pilot by the
             ideal formula; MC levels: nMSE(C) = (sqrt(V1 r) + sqrt(V2 L))^2 / (C V) (setup L and pilot excluded/charged).
  ML2-RQMC : the same two levels, each with scrambled Sobol' directions (independent scrambles), N2 = 2^m, N1 = 2^(m+k),
             k in {0, 2, 4}; envelope over k; cost N1 r + N2 L + L (setup).
  MLD      : dyadic multilevel (levels 0, 1, 2, 4, ..., L/2), MC levels, ideal allocation from the pilot.
  CVQ      : control variate with exactly zero mean, c(x) = G x + qhat (x^T H_k x - tr H_k)/2, G = E[a x^T] (n x n),
             qhat = E a/|E a|, H_k = top-k (by |eigenvalue|, k = 32) part of E[(qhat.a)(x x^T - I)], all from the pilot;
             estimator mean(a - c) on MC inputs: nMSE = Var(a - c)/(N V); cost N (L + 2) + Pp L (pilot charged).
  RQMC-CVQ : the same residual a - c on the RQMC points of method RQMC (plain Gaussian coordinates)."""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/unrestricted/lib')
import ib
from mlmc import ReferencePath, level_samples

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int); ap.add_argument('L', type=int)
ap.add_argument('--R', type=int, default=12); ap.add_argument('--mmax', type=int, default=12); ap.add_argument('--Pp', type=int, default=4096)
ap.add_argument('--out')
a = ap.parse_args()
n, seed, L, R = a.n, a.seed, a.L, a.R
t0 = time.time()
net = ib.Net(seed, n, L)
W = net.W
mn = ib.mean_radius(n)
ms = list(range(6, a.mmax + 1, 2))


def f(X, chunk=4096):
    out = np.empty((len(X), n))
    for i in range(0, len(X), chunk):
        h = X[i:i + chunk]
        for Wl in W:
            h = np.maximum(h @ Wl.T, 0.0)
        out[i:i + chunk] = h
    return out


Xm = np.random.default_rng([7501, seed, n, L]).standard_normal((16384, n))
Fm = f(Xm)
V = float(np.sum(Fm.var(0, ddof=1)))
Gm = Fm / np.linalg.norm(Xm, axis=1, keepdims=True)
Vsph = float(np.sum((mn * Gm).var(0, ddof=1)))
res = dict(n=n, seed=seed, L=L, R=R, V=V, V_sph=Vsph, methods={})
res['methods']['MC'] = [dict(N=1 << m, cost=(1 << m) * L, nmse=1.0 / (1 << m)) for m in ms]
res['methods']['RBMC'] = [dict(N=1 << m, cost=(1 << m) * L, nmse=Vsph / V / (1 << m)) for m in ms]
# pilot (charged where used)
Xp = np.random.default_rng([7502, seed, n, L]).standard_normal((a.Pp, n))
Fp = f(Xp)
q = np.sum(Fp * Fp, 1); qc = q - q.mean()
Mq = (Xp * qc[:, None]).T @ Xp / a.Pp
ev, Vec = np.linalg.eigh(Mq)
Vec = Vec[:, np.argsort(-np.abs(ev))]
pilot_cost = a.Pp * L + n / 2
# quadratic control variate (exact zero mean under x ~ N(0, I))
Fc = Fp - Fp.mean(0)
Gm_ = Fc.T @ Xp / a.Pp                                   # n_out x n
qhat = Fp.mean(0) / max(np.linalg.norm(Fp.mean(0)), 1e-300)
sc = Fp @ qhat; scc = sc - sc.mean()
Hs = (Xp * scc[:, None]).T @ Xp / a.Pp - np.eye(n) * scc.mean()
eh, Vh = np.linalg.eigh(Hs)
idx = np.argsort(-np.abs(eh))[:32]
Hk_val, Hk_vec = eh[idx], Vh[:, idx]


def cvq(X):
    P_ = X @ Hk_vec
    quad = 0.5 * (P_ * P_ - 1.0) @ Hk_val
    return X @ Gm_.T + quad[:, None] * qhat[None, :]


Rm = Fm - cvq(Xm)
Vres_cv = float(np.sum(Rm.var(0, ddof=1)))
res['cvq'] = dict(V_res=Vres_cv, ratio=Vres_cv / V, k=32)
res['methods']['CVQ'] = [dict(N=1 << m, cost=(1 << m) * (L + 2), cost_pilot=(1 << m) * (L + 2) + pilot_cost, nmse=Vres_cv / V / (1 << m)) for m in ms]
print(f'CVQ: residual variance share {Vres_cv / V:.3f}', flush=True)
rq, rb, rot, rqc = [], [], [], []
for m in ms:
    N = 1 << m
    e1, e2, e3, e4 = [], [], [], []
    for r in range(R):
        Xq = ib.sobol_gauss(n, m, [7503, seed, n, L, m, r])
        Fq = f(Xq)
        e1.append(Fq.mean(0))
        e4.append((Fq - cvq(Xq)).mean(0))
        e2.append(mn * (Fq / np.linalg.norm(Xq, axis=1, keepdims=True)).mean(0))
        Yq = ib.sobol_gauss(n, m, [7504, seed, n, L, m, r])
        Xr = Yq @ Vec.T
        Fr = f(Xr)
        e3.append(mn * (Fr / np.linalg.norm(Xr, axis=1, keepdims=True)).mean(0))
    v1 = float(np.sum(np.array(e1).var(0, ddof=1))); v2 = float(np.sum(np.array(e2).var(0, ddof=1))); v3 = float(np.sum(np.array(e3).var(0, ddof=1)))
    v4 = float(np.sum(np.array(e4).var(0, ddof=1)))
    rqc.append(dict(N=N, cost=N * (L + 2), cost_pilot=N * (L + 2) + pilot_cost, nmse=v4 / V))
    rq.append(dict(N=N, cost=N * L, nmse=v1 / V)); rb.append(dict(N=N, cost=N * L, nmse=v2 / V))
    rot.append(dict(N=N, cost=N * (L + 1), cost_pilot=N * (L + 1) + pilot_cost, nmse=v3 / V))
    print(f'N={N}: gain vs MC  RQMC {1 / N / (v1 / V):.2f}  RQMC-RB {1 / N / (v2 / V):.2f}  RQMC-ROT {1 / N / (v3 / V):.2f} '
          f'RQMC-CVQ {1 / N / (v4 / V):.2f} (RBMC {V / Vsph:.3f}, CVQ {V / Vres_cv:.2f}) | {time.time() - t0:.0f}s', flush=True)
res['methods'].update({'RQMC': rq, 'RQMC-RB': rb, 'RQMC-ROT': rot, 'RQMC-CVQ': rqc})
# multilevel: pilot level variances on Pp directions, all prefix depths from one pass per batch.
# Definitions as work/unrestricted/lib/mlmc.py: G_r(U) = m_n s_r |h_r(U)| q_L (s_r = product of the reference path's
# norms after layer r), level j of levels [0 = r_0 < r_1 < ... < r_J]: G_{r_j} - G_{r_{j-1}}, last level Y - G_{r_J}.
ref = ReferencePath(W, np.random.default_rng([7505, seed, n, L]))
Up = ib.sphere(np.random.default_rng([7506, seed, n, L]).standard_normal((a.Pp, n)))
rs_all = [1 << i for i in range(0, 30) if (1 << i) <= L // 2]
norms = {0: np.linalg.norm(Up, axis=1)}
h = Up
for l in range(1, L + 1):
    h = np.maximum(h @ W[l - 1].T, 0.0)
    if l in rs_all:
        norms[l] = np.linalg.norm(h, axis=1)
Yp = mn * h
qL, suf = ref.qL, ref.suffix
Gr = {r: mn * suf[r] * norms[r] for r in [0] + rs_all}     # scalar amplitude of G_r along qL
trv = lambda M: float(np.sum(M.var(0, ddof=1)))
cands = []
for r in rs_all:
    V1 = float(np.var(Gr[r] - Gr[0], ddof=1))                 # |qL| = 1: trace = scalar variance
    V2 = trv(Yp - Gr[r][:, None] * qL[None, :])
    cands.append(dict(r=r, V1=V1, V2=V2, K=(math.sqrt(V1 * r) + math.sqrt(V2 * L)) ** 2))
best = min(cands, key=lambda x: x['K'])
rbest = best['r']
pilot_ml = a.Pp * L
res['ml2_pilot'] = cands
res['methods']['ML2'] = [dict(cost=C, nmse=best['K'] / C / V, cost_charged=C + L + pilot_ml) for C in [(1 << m) * L for m in ms]]
lev = [0] + rs_all
Vj = [float(np.var(Gr[lev[j]] - Gr[lev[j - 1]], ddof=1)) for j in range(1, len(lev))] + [trv(Yp - Gr[lev[-1]][:, None] * qL[None, :])]
Cj = [lev[j] for j in range(1, len(lev))] + [L]
Kd = sum(math.sqrt(v * c) for v, c in zip(Vj, Cj)) ** 2
res['mld'] = dict(levels=lev, V=Vj, C=Cj, K=Kd)
res['methods']['MLD'] = [dict(cost=C, nmse=Kd / C / V, cost_charged=C + L + pilot_ml) for C in [(1 << m) * L for m in ms]]
# two-level with RQMC on both levels
ml2q = []
for m in ms:
    N2 = 1 << m
    v2s = []
    for r_ in range(R):
        U2 = ib.sphere(ib.sobol_gauss(n, m, [7507, seed, n, L, m, r_]))
        v2s.append(level_samples(W, ref, [0, rbest], 2, U2).mean(0))
    var2 = float(np.sum(np.array(v2s).var(0, ddof=1)))
    for k in (0, 2, 4):
        if m + k > 16:
            continue
        N1 = 1 << (m + k)
        v1s = []
        for r_ in range(R):
            U1 = ib.sphere(ib.sobol_gauss(n, m + k, [7508, seed, n, L, m + k, r_]))
            v1s.append(level_samples(W, ref, [0, rbest], 1, U1).mean(0))
        var1 = float(np.sum(np.array(v1s).var(0, ddof=1)))
        ml2q.append(dict(N1=N1, N2=N2, cost=N1 * rbest + N2 * L + L, nmse=(var1 + var2) / V, var1=var1, var2=var2))
    print(f'ML2-RQMC m={m}: ' + ', '.join(f'N1={x["N1"]} gain {(L / x["cost"]) / x["nmse"]:.2f}' for x in ml2q if x['N2'] == N2)
          + f' | {time.time() - t0:.0f}s', flush=True)
res['methods']['ML2-RQMC'] = ml2q
res['r_best'] = rbest
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
print('done', res['sec'], 'r_best', rbest, 'ML2 K/(V L)', best['K'] / V / L, 'MLD K/(V L)', Kd / V / L)
