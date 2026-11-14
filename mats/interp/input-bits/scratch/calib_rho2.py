"""Instrument calibration (no net data): the order-2 RQMC variance ratio rho_2(N, d) = Var_RQMC / Var_MC of the mean of an
exchangeable order-2 function f(x) = x^T M x (M symmetric, zero diagonal, iid N(0,1) entries), x = Phi^-1(U), scipy
scrambled Sobol' (LMS + digital shift) as used in part2_input.py. Average over 4 matrices M; R replicates per (N, d).
Also rho_3 for random triple products (order 3). Output: results/rho_calibration.json.
usage: calib_rho2.py [--R 64]"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

ap = argparse.ArgumentParser(); ap.add_argument('--R', type=int, default=64)
a = ap.parse_args()
out = {}
for d in (32, 64, 128, 256):
    rng = np.random.default_rng([31, d])
    Ms = []
    for i in range(4):
        M = rng.standard_normal((d, d)); M = (M + M.T) / 2; np.fill_diagonal(M, 0.0); Ms.append(M)
    T = rng.integers(0, d, (8 * d, 3))
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    f2 = lambda X: np.stack([np.einsum('pi,ij,pj->p', X, M, X) for M in Ms], 1)
    f3 = lambda X: np.sum(X[:, T[:, 0]] * X[:, T[:, 1]] * X[:, T[:, 2]], 1)[:, None]
    Xmc = rng.standard_normal((1 << 14, d))
    v2 = f2(Xmc).var(0); v3 = f3(Xmc).var(0)
    for m in (8, 10, 12):
        N = 1 << m
        e2, e3 = [], []
        for r in range(a.R):
            X = ib.sobol_gauss(d, m, [32, d, m, r])
            e2.append(f2(X).mean(0)); e3.append(f3(X).mean(0))
        r2 = np.array(e2).var(0, ddof=1) / (v2 / N)
        r3 = float(np.array(e3).var(0, ddof=1)[0] / (v3[0] / N))
        out[f'{d},{N}'] = dict(d=d, N=N, rho2=float(r2.mean()), rho2_each=r2.tolist(), rho3=r3)
        print(d, N, 'rho2', round(float(r2.mean()), 3), np.round(r2, 3).tolist(), 'rho3', round(r3, 3), flush=True)
json.dump(out, open('/home/agent/workspaces/mats/work/input-bits/results/rho_calibration.json', 'w'), indent=1)
