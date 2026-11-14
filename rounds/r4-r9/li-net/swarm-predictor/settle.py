"""Settlement: what each chain is paid once the board is revealed (STUDY-LIR 3c).

A claim (i, j) enters the pool as the log-likelihood ratio
    l_ij(z) = log q_ij(f_j(z)) - log p0_j(f_j(z))
(its probability read as a posterior over the rules-only prior; see pool.py). Its settlement is
    s_ij = l_ij(z*) - E_mu[l_ij] = (y_j - mu_j) * (logit q_ij - logit p0_j):
the chain's log-score gain over the prior on that statement, minus the pool's expectation of the
same gain. So E_mu[s_ij] = 0: a claim that agrees with the pool costs nothing and earns nothing in
expectation, and a chain earns only where it moved away from the pool in the direction the board
turned out. -s_ij is exactly the derivative of the pool's log loss -log mu(z*) in the claim's stake
b_ij (grad_check below; STUDY-LIR 3c, dL/dbeta_i = -s_i / beta_0). A chain is paid s_i = sum of its
s_ij. Only this payment reaches the chain sampler.
Logged, never paid:
  s_raw   the same with the raw log score, (y_j - mu_j) * logit q_ij. It differs from s_ij by the
          prior's own bet on the statements the chain chose, which does not depend on what it said.
  pi_i    stake-weighted sum of b_ij * s_ij (first-order leave-one-out, STUDY-LIR section 4).
  loo_i   exact leave-one-out: log mu(z*) - log mu_without_i(z*), stakes recomputed without chain i.
  info_i  oracle informativeness: sum over the chain's claims of E_bayes[l_ij], i.e.
          KL(bayes_j || p0_j) - KL(bayes_j || q_ij); how much better than the prior its claims are,
          measured against exact Bayes and independent of the pool and of the realised board.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from pool import Board, Claims, logit, shift, stakes, theta, world


def settle(board: Board, cl: Claims, b: np.ndarray, mu_marg: np.ndarray, y: np.ndarray,
           n_chains: Optional[int] = None) -> Dict[str, np.ndarray]:
    """mu_marg: pool prices [J] (from the pool the chains are settled against); y: realised truth [J]."""
    n = int(n_chains if n_chains is not None else cl.chain.max() + 1)
    err = y[cl.stmt] - mu_marg[cl.stmt]
    s = err * shift(board, cl)
    s_raw = err * logit(cl.q)
    per = lambda v: np.bincount(cl.chain, weights=v, minlength=n)
    return {"s_claim": s, "s": per(s), "s_raw": per(s_raw), "pi": per(b * s),
            "n_claims": np.bincount(cl.chain, minlength=n)}


def loo_world(board: Board, cl: Claims, zstar: int, kappa: float, alpha: float, n_chains: int) -> np.ndarray:
    """Exact leave-one-out credit for the world pool with stakes b = kappa * w / m_j ** alpha."""
    J = board.J
    d = shift(board, cl)
    S = np.zeros((n_chains, J))
    M = np.zeros((n_chains, J))
    np.add.at(S, (cl.chain, cl.stmt), cl.w * d)
    np.add.at(M, (cl.chain, cl.stmt), 1.0)
    St, Mt = S.sum(0), M.sum(0)

    def logmu_star(Ssum, Msum):
        th = kappa * Ssum * np.where(Msum > 0, Msum, 1.0) ** (-alpha)
        lg = board.logp0[None] + th @ board.F.T if th.ndim == 2 else (board.logp0 + board.F @ th)[None]
        mx = lg.max(1, keepdims=True)
        return lg[:, zstar] - (mx[:, 0] + np.log(np.exp(lg - mx).sum(1)))

    full = logmu_star(St, Mt)[0]
    return full - logmu_star(St[None] - S, Mt[None] - M)


def informativeness(board: Board, cl: Claims, bayes_marg: np.ndarray, n_chains: int) -> np.ndarray:
    bj = bayes_marg[cl.stmt]
    q = np.clip(cl.q, 1e-6, 1 - 1e-6)
    p0 = np.clip(board.p0_marg[cl.stmt], 1e-6, 1 - 1e-6)
    e = bj * np.log(q / p0) + (1 - bj) * np.log((1 - q) / (1 - p0))
    return np.bincount(cl.chain, weights=e, minlength=n_chains)


def grad_check(board: Board, cl: Claims, b: np.ndarray, zstar: int, h: float = 1e-5) -> float:
    """max over claims of |d(-log mu(z*))/d b_ij + s_ij|, by central differences on the world pool."""
    def L(bb):
        mu = world(board, theta(board, cl, bb))
        return -np.log(mu[zstar])
    mu = world(board, theta(board, cl, b))
    y = board.F[zstar]
    s = settle(board, cl, b, mu @ board.F, y)["s_claim"]
    worst = 0.0
    for k in range(len(b)):
        e = np.zeros_like(b)
        e[k] = h
        g = (L(b + e) - L(b - e)) / (2 * h)
        worst = max(worst, abs(g + s[k]))
    return worst


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
