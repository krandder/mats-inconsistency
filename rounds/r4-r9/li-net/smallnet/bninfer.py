"""Exact single-cell marginals and log-likelihoods of small Bayes nets over the 25 board cells.

A net is given by a topological order (a permutation of the cells), parents par[c] (k <= 2 cells earlier
in the order, -1 = none) and a table cpt[c, v] = P(x_c = 1 | parents' values v), v = x_p1 + 2 x_p2.
Tables must not depend on a missing parent (replicated entries), so -1 parents can be ignored.

k = 1 (a forest): marginals by forward propagation, exact, vectorized (torch).
k = 2: exact variable elimination along the order (keep the joint of the 'frontier': cells already
placed that a later cell still needs). Brute force over 2^25 states (~33M) per (board, stage, order)
is ~1000x slower; simulations with nearest or random parents give max frontier 12 (median), 16 (max).
"""
from __future__ import annotations

import numpy as np
import torch


def k1_marginals(order: torch.Tensor, par: torch.Tensor, cpt: torch.Tensor) -> torch.Tensor:
    """order [B,n] long, par [B,n] long (-1 none), cpt [B,n,2] P(x_c=1 | x_par=v). Returns [B,n]."""
    B, n = order.shape
    m = torch.zeros(B, n, dtype=cpt.dtype, device=cpt.device)
    ar = torch.arange(B, device=cpt.device)
    for t in range(n):
        c = order[:, t]
        p = par[ar, c]
        mp = torch.where(p >= 0, m[ar, p.clamp(min=0)], torch.zeros_like(m[:, 0]))
        t0, t1 = cpt[ar, c, 0], cpt[ar, c, 1]
        m[ar, c] = torch.where(p >= 0, t0 * (1 - mp) + t1 * mp, t0)
    return m


def _last_use(order: np.ndarray, par: np.ndarray) -> np.ndarray:
    n = len(order)
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n)
    last = pos.copy()
    for c in range(n):
        for p in par[c]:
            if p >= 0:
                last[p] = max(last[p], pos[c])
    return last


def k2_marginals_np(order: np.ndarray, par: np.ndarray, cpt: np.ndarray) -> np.ndarray:
    """One net: order [n], par [n,2], cpt [n,4] -> exact marginals [n] (float64)."""
    n = len(order)
    last = _last_use(order, par)
    axes: list = []
    J = np.array(1.0)
    marg = np.zeros(n)
    for t, c in enumerate(order):
        ps = [int(p) for p in par[c] if p >= 0]
        tab = np.asarray(cpt[c], np.float64)
        if len(ps) == 0:
            f = np.array([1 - tab[0], tab[0]])
        elif len(ps) == 1:
            # parent in slot 0 or 1; table replicated over the missing one
            slot = 0 if par[c][0] >= 0 else 1
            t1 = tab[[0, 1]] if slot == 0 else tab[[0, 2]]
            f = np.stack([1 - t1, t1], axis=-1)            # [x_p, x_c]
        else:
            t2 = tab.reshape(2, 2).T                        # t2[a, b] = tab[a + 2b], a = x_p1, b = x_p2
            f = np.stack([1 - t2, t2], axis=-1)            # [x_p1, x_p2, x_c]
        J = np.einsum(J, list(axes), f, ps + [int(c)], list(axes) + [int(c)])
        axes = axes + [int(c)]
        marg[c] = J.reshape(-1, 2).sum(0)[1] if J.ndim > 1 else J[1]
        for v in [int(c)] + ps:
            if last[v] == t and v in axes:
                J = J.sum(axis=axes.index(v))
                axes.remove(v)
    return marg


def k2_marginals(order, par, cpt) -> np.ndarray:
    """Batch wrapper: order [B,n], par [B,n,2], cpt [B,n,4] (numpy or torch) -> [B,n] float64."""
    order = np.asarray(order.cpu() if torch.is_tensor(order) else order)
    par = np.asarray(par.cpu() if torch.is_tensor(par) else par)
    cpt = np.asarray(cpt.detach().cpu().double() if torch.is_tensor(cpt) else cpt, np.float64)
    return np.stack([k2_marginals_np(order[b], par[b], cpt[b]) for b in range(len(order))])


def brute_marginals(par: np.ndarray, cpt: np.ndarray) -> np.ndarray:
    """Reference: enumerate all 2^n states (small n only). par [n,2], cpt [n,4]."""
    n = len(par)
    X = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int64)
    lp = np.zeros(2 ** n)
    for c in range(n):
        a = X[:, par[c][0]] if par[c][0] >= 0 else 0
        b = X[:, par[c][1]] if par[c][1] >= 0 else 0
        p1 = cpt[c][a + 2 * b]
        lp += np.log(np.where(X[:, c] == 1, p1, 1 - p1))
    w = np.exp(lp)
    return w @ X / w.sum()


def bn_loglik(x: torch.Tensor, par: torch.Tensor, cpt: torch.Tensor, k: int) -> torch.Tensor:
    """log P(x) under the net. x [B,n] {0,1}; par [B,n,k] or [B,n]; cpt [B,n,2^k]. Returns [B]."""
    if k == 1:
        par = par.unsqueeze(-1)
    v = torch.zeros_like(x, dtype=torch.long)
    for s in range(par.shape[-1]):
        p = par[..., s]
        v = v + (1 << s) * torch.where(p >= 0, x.long().gather(1, p.clamp(min=0)), torch.zeros_like(p))
    p1 = cpt.gather(-1, v.unsqueeze(-1)).squeeze(-1).clamp(1e-7, 1 - 1e-7)
    return torch.where(x > 0, p1.log(), (1 - p1).log()).sum(-1)


def oracle_cpts(post_w: np.ndarray, P: np.ndarray, order: np.ndarray, par: np.ndarray) -> np.ndarray:
    """Exact conditionals P*(x_c=1 | parents) from posterior weights over placements (diagnostic only)."""
    n = P.shape[1]
    cpt = np.zeros((n, 4))
    for c in range(n):
        ps = [p for p in par[c] if p >= 0]
        for v in range(4):
            a, b = v & 1, v >> 1
            m = np.ones(len(P), bool)
            if par[c][0] >= 0:
                m &= P[:, par[c][0]] == a
            if par[c][1] >= 0:
                m &= P[:, par[c][1]] == b
            den = post_w[m].sum()
            cpt[c, v] = (post_w[m] @ P[m, c]) / den if den > 0 else 0.5
        if not ps:
            cpt[c, :] = cpt[c, 0]
    return cpt


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1) k=2 elimination vs brute force on random 12-cell nets
    for rep in range(30):
        n = 12
        order = rng.permutation(n)
        par = -np.ones((n, 2), np.int64)
        for t, c in enumerate(order):
            if t == 1:
                par[c, rng.integers(2)] = order[0]
            elif t >= 2:
                par[c] = rng.choice(order[:t], 2, replace=False)
        cpt = rng.uniform(0.02, 0.98, (n, 4))
        for c in range(n):  # replicate over missing parents
            if par[c, 0] < 0 and par[c, 1] < 0:
                cpt[c, :] = cpt[c, 0]
            elif par[c, 1] < 0:
                cpt[c, 2:] = cpt[c, :2]
            elif par[c, 0] < 0:
                cpt[c, [1, 3]] = cpt[c, [0, 2]]
        a, b = k2_marginals_np(order, par, cpt), brute_marginals(par, cpt)
        assert np.abs(a - b).max() < 1e-10, (rep, np.abs(a - b).max())
    print("k2 elimination == brute force on 30 random 12-cell nets")
    # 2) k=1 forward == brute force
    for rep in range(30):
        n = 12
        order = rng.permutation(n)
        par1 = -np.ones(n, np.int64)
        for t, c in enumerate(order):
            if t >= 1:
                par1[c] = rng.choice(order[:t])
        cpt1 = rng.uniform(0.02, 0.98, (n, 2))
        for c in range(n):
            if par1[c] < 0:
                cpt1[c, 1] = cpt1[c, 0]
        m = k1_marginals(torch.tensor(order)[None], torch.tensor(par1)[None], torch.tensor(cpt1)[None])[0].numpy()
        par2 = np.stack([par1, -np.ones(n, np.int64)], 1)
        cpt2 = np.concatenate([cpt1, cpt1], 1)
        assert np.abs(m - brute_marginals(par2, cpt2)).max() < 1e-10
    print("k1 forward == brute force on 30 random 12-cell forests")
    # 3) log-lik check
    x = torch.tensor(rng.integers(0, 2, (1, 12)))
    print("ok")
