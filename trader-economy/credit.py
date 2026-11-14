"""Coalition values and the four information-pricing arms.

A Game holds the reports (logits) of P players over one epoch and the
outcomes.  v(S) = total nats by which a cross-fitted logistic combiner of the
players in S beats the 0.5 prior on this epoch (fit on even steps, scored on
odd steps and vice versa, so a useless report earns ~0, not an overfit bonus).
An LMSR with liquidity b pays b * v(all) for this epoch: that is the pot.

Arms (payment to player i, in nats; multiply by b for money):
  loo     v(N) - v(N minus i)                   leave-one-out = moving last
  msr1    v(before + i) - v(before), one fixed order   Hanson MSR in sequence
  msrK    msr averaged over K random orders     sampled Shapley
  shapley exact Shapley (all 2^P coalitions)
MSR and Shapley telescope to v(N) (budget balanced); LOO does not.
"""
import itertools
from math import factorial

import numpy as np


def _fit(X, y, ridge=1e-2, iters=10):
    th = np.zeros(X.shape[1])
    R = ridge * np.eye(X.shape[1])
    R[-1, -1] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ th)))
        g = X.T @ (y - p) - R @ th
        H = (X * (p * (1 - p))[:, None]).T @ X + R
        th += np.linalg.solve(H, g)
    return th


def _score(X, y, th):
    p = np.clip(1 / (1 + np.exp(-(X @ th))), 1e-6, 1 - 1e-6)
    return float(np.sum(np.where(y > 0.5, np.log(p), np.log(1 - p))) - len(y) * np.log(0.5))


class Game:
    def __init__(self, Z, y):
        self.Z = np.asarray(Z, float)           # (P, n)
        self.y = np.asarray(y, float)
        self.P = self.Z.shape[0]
        n = len(y)
        self.a = np.arange(n) % 2 == 0
        self.b = ~self.a
        self.memo = {}
        self.evals = 0

    def v(self, mask):
        if mask in self.memo:
            return self.memo[mask]
        idx = [i for i in range(self.P) if mask >> i & 1]
        X = np.vstack([self.Z[idx], np.ones(len(self.y))]).T if idx else np.ones((len(self.y), 1))
        tot = 0.0
        for tr, te in ((self.a, self.b), (self.b, self.a)):
            th = _fit(X[tr], self.y[tr])
            tot += _score(X[te], self.y[te], th)
        self.memo[mask] = tot
        self.evals += 1
        return tot

    def full(self):
        return self.v((1 << self.P) - 1)


def loo(g):
    N = (1 << g.P) - 1
    return np.array([g.full() - g.v(N & ~(1 << i)) for i in range(g.P)])


def msr_order(g, order):
    phi = np.zeros(g.P)
    m = 0
    prev = g.v(0)
    for i in order:
        m |= 1 << i
        cur = g.v(m)
        phi[i] = cur - prev
        prev = cur
    return phi


def msr_k(g, K, rng):
    phi = np.zeros(g.P)
    for _ in range(K):
        phi += msr_order(g, rng.permutation(g.P))
    return phi / K


def shapley_exact_from(vfun, P):
    """Exact Shapley for any value function on bitmasks (P <= ~12)."""
    M = 1 << P
    vals = np.array([vfun(m) for m in range(M)])
    pc = np.array([bin(m).count("1") for m in range(M)])
    wt = np.array([factorial(k) * factorial(P - k - 1) / factorial(P) for k in range(P)])
    masks = np.arange(M)
    phi = np.zeros(P)
    for i in range(P):
        sel = (masks >> i & 1) == 0
        S = masks[sel]
        phi[i] = np.sum(wt[pc[S]] * (vals[S | (1 << i)] - vals[S]))
    return phi


def shapley_exact(g):
    return shapley_exact_from(g.v, g.P)


def arm_payments(g, arm, rng, K=16):
    """Credit in nats per player for one arm, and #coalition evaluations used."""
    e0 = g.evals
    if g.P == 0:
        return np.zeros(0), 0
    if arm == "loo":
        phi = loo(g)
    elif arm == "msr1":
        phi = msr_order(g, range(g.P))
    elif arm == "msrK":
        phi = msr_k(g, K, rng)
    elif arm == "shapley":
        phi = shapley_exact(g)
    elif arm == "equal":
        phi = np.full(g.P, g.full() / g.P)
    else:
        raise ValueError(arm)
    return phi, g.evals - e0
