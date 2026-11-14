"""Calibration of the instrument (no net data): how much of an order-k ANOVA term does scrambled Sobol' (scipy, LMS +
digital shift) remove at N points in dimension d? Test functions on x ~ N(0, I_d) (x = Phi^{-1}(U)):
  lin  : sum_j c_j x_j                                   (order 1)
  diag : sum_j c_j (x_j^2 - 1)                          (order 1, degree 2)
  quad : x^T M x - tr M, M symmetric, zero diagonal       (order 2)
  cub  : sum over random triples x_i x_j x_k              (order 3)
  ridge: mean_i relu(w_i.x) - E, w_i dense unit vectors   (Hermite mix)
Coordinates: either all d equally weighted (exchangeable, like the input regime) or with geometric weights 0.999^j
(important coordinates first). Prints ratio rho = Var_RQMC / Var_MC (fraction NOT removed) at N = 2^8, 2^10, 2^12.
usage: calib_rqmc.py d [--R 32]"""
import argparse, math, sys
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri

ap = argparse.ArgumentParser(); ap.add_argument('d', type=int); ap.add_argument('--R', type=int, default=32)
a = ap.parse_args()
d = a.d
rng = np.random.default_rng(7)
c = rng.standard_normal(d)
M = rng.standard_normal((d, d)); M = (M + M.T) / 2; np.fill_diagonal(M, 0.0)
T = rng.integers(0, d, (4 * d, 3))
T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
Wr = rng.standard_normal((64, d)); Wr /= np.linalg.norm(Wr, axis=1, keepdims=True)
fs = {
    'lin': lambda X: X @ c,
    'diag': lambda X: (X * X - 1) @ c,
    'quad': lambda X: np.einsum('pi,ij,pj->p', X, M, X),
    'cub': lambda X: np.sum(X[:, T[:, 0]] * X[:, T[:, 1]] * X[:, T[:, 2]], 1),
    'ridge': lambda X: np.mean(np.maximum(X @ Wr.T, 0.0), 1),
}
Xmc = rng.standard_normal((1 << 15, d))
vmc = {k: float(np.var(f(Xmc))) for k, f in fs.items()}
for k in (8, 10, 12):
    N = 1 << k
    est = {kk: [] for kk in fs}
    for r in range(a.R):
        U = qmc.Sobol(d=d, scramble=True, seed=np.random.default_rng([11, d, k, r])).random_base2(k)
        X = ndtri(np.clip(U, 1e-16, 1 - 1e-16))
        for kk, f in fs.items():
            est[kk].append(float(np.mean(f(X))))
    print(f'd={d} N={N}: ' + ', '.join(f'{kk} {np.var(v, ddof=1) / (vmc[kk] / N):.3f}' for kk, v in est.items()), flush=True)
