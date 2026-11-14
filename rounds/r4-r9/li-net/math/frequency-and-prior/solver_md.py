"""Accurate solver for min over mu in simplex(Mod T) of b0*KL(mu||base) + bP*sum_j KL(Bern (A mu)_j || Bern P_j).
Mirror descent (exponentiated gradient) with Armijo backtracking; forced rows (constant on Mod T) dropped,
since their KL term is constant in mu. Used by the skeptic to separate solver failures from theorem failures."""
import numpy as np
from scipy.special import xlogy, logit

def _bkl(x, p):
    return xlogy(x, x) - xlogy(x, p) + xlogy(1 - x, 1 - x) - xlogy(1 - x, 1 - p)

def solve(A, P, base=None, b0=0.0, bP=1.0, mu0=None, iters=20000, tol=1e-13):
    free = np.ptp(A, axis=1) > 0
    Af, Pf = A[free], P[free]
    n = A.shape[1]
    mu = np.full(n, 1.0 / n) if mu0 is None else 0.5 * mu0 + 0.5 / n
    lb = np.log(base) if b0 > 0 else None
    def F(m):
        x = np.clip(Af @ m, 0.0, 1.0)
        v = bP * _bkl(x, Pf).sum()
        if b0 > 0: v += b0 * float(np.sum(xlogy(m, m)) - m @ lb)
        return v
    def G(m):
        x = np.clip(Af @ m, 1e-300, 1 - 1e-16)
        gr = bP * (Af.T @ (logit(x) - logit(Pf)))
        if b0 > 0: gr = gr + b0 * (np.log(np.maximum(m, 1e-300)) - lb + 1)
        return gr
    lr, f0 = 1.0, F(mu)
    for k in range(iters):
        gr = G(mu)
        while True:
            z = np.log(np.maximum(mu, 1e-300)) - lr * (gr - gr.min())
            z -= z.max(); m2 = np.exp(z); m2 /= m2.sum()
            f2 = F(m2)
            if f2 <= f0 - 1e-4 * (gr @ (mu - m2)) or lr < 1e-12: break
            lr *= 0.5
        done = abs(f0 - f2) < tol * max(1, abs(f0))
        mu, f0 = m2, f2; lr *= 1.5
        if done and k > 50: break
    x = np.clip(A @ mu, 0.0, 1.0)
    val = bP * _bkl(x, P).sum() + (b0 * float(np.sum(xlogy(mu, mu)) - mu @ lb) if b0 > 0 else 0.0)
    return mu, x, val
