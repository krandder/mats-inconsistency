"""T1 lemma 6 check: a fund of funds with fixed fractions is a flat mixture; only rebalancing adds power.

Checks (see PREREG.md): C6a buy-and-hold DAG and cyclic cross-holdings == flat Bayes mixture;
C6b constant-rebalanced DAG == flat CRP with path-product weights; C6c endogenous wealth-weighted
price is invariant to buy-and-hold consolidation (and not to rebalancing); C6d growth per family;
C6e the analytic toy. Numpy only. Usage: python3 check_L6.py --seed 1 [--T 6000]
Importable helpers for K1: leaf_returns, flat_weights, bh_fund_values, crp_fund_returns.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
from streams import make_questions, FAMILIES  # noqa: E402

KT_K = [0, 1, 2, 3, 4, 6, 8]
LEAF_NAMES = [f"kt{k}" for k in KT_K] + ["logit16", "const0.9", "const0.1"]
EPS = 1e-4


def leaf_probs(bits):
    """bits [Q,T] -> P [N,Q,T] = each leaf's P(y_t=1 | y_<t), and price pi [Q,T] (order-0 KT)."""
    Q, T = bits.shape
    y = bits.astype(np.int64)
    P = np.empty((len(LEAF_NAMES), Q, T))
    qi = np.arange(Q)
    for n, k in enumerate(KT_K):
        cnt = np.full((Q, 2 ** k, 2), 0.5)
        ctx = np.zeros(Q, dtype=np.int64)
        for t in range(T):
            c = cnt[qi, ctx]
            P[n, :, t] = c[:, 1] / c.sum(1)
            cnt[qi, ctx, y[:, t]] += 1.0
            if k > 0:
                ctx = ((ctx << 1) | y[:, t]) & ((1 << k) - 1)
    W = 16
    w = np.zeros((Q, W)); b = np.zeros(Q); lr = 0.1
    xpad = np.concatenate([np.zeros((Q, W)), 2.0 * y - 1.0], axis=1)
    n = len(KT_K)
    for t in range(T):
        x = xpad[:, t:t + W]
        p = 1.0 / (1.0 + np.exp(-(np.einsum("qw,qw->q", w, x) + b)))
        p = np.clip(p, EPS, 1 - EPS)
        P[n, :, t] = p
        g = y[:, t] - p
        w += lr * g[:, None] * x; b += lr * g
    P[n + 1] = 0.9
    P[n + 2] = 0.1
    price = P[0].copy()  # order-0 KT running base rate = the exogenous market price
    return P, price


def leaf_returns(P, price, bits):
    """Kelly wealth relative per step: x[i,q,t] = p_i(y_t) / pi(y_t)."""
    y = bits[None].astype(bool)
    return np.where(y, P / price[None], (1 - P) / (1 - price[None]))


def random_dag(rng, n_leaf, n1=5, n2=2):
    """Adjacency A[parent, child] of fixed fractions (rows of funds sum to 1). Nodes: leaves, L1, L2, root."""
    n = n_leaf + n1 + n2 + 1
    A = np.zeros((n, n))
    L1 = list(range(n_leaf, n_leaf + n1)); L2 = list(range(n_leaf + n1, n_leaf + n1 + n2)); root = n - 1
    for f in L1:
        ch = rng.choice(n_leaf, size=rng.integers(3, 6), replace=False)
        A[f, ch] = rng.dirichlet(np.ones(len(ch)))
    for f in L2:
        ch = list(rng.choice(L1, size=rng.integers(2, 4), replace=False)) + [int(rng.integers(n_leaf))]
        A[f, ch] = rng.dirichlet(np.ones(len(ch)))
    ch = L2 + [L1[0]]
    A[root, ch] = rng.dirichlet(np.ones(len(ch)))
    return A, root


def flat_weights(A, n_leaf, node):
    """Path-sum (Leontief) weights of each leaf in `node`: row of (I - A)^-1 restricted to leaves."""
    M = np.linalg.inv(np.eye(A.shape[0]) - A)
    return M[node, :n_leaf]


def bh_fund_values(A, n_leaf, leafV, iters=200):
    """Buy-and-hold: fund value = sum_c A[f,c] * (child value / child initial value). Leaves start at 1.
    Solved by fixed-point iteration (works for cyclic cross-holdings with spectral radius < 1).
    leafV [n_leaf, ...] -> V [n, ...]."""
    n = A.shape[0]
    V = np.zeros((n,) + leafV.shape[1:])
    V[:n_leaf] = leafV
    V0 = np.ones(n)
    for _ in range(iters):
        Vn = V.copy()
        Vn[n_leaf:] = np.tensordot(A[n_leaf:] / V0[None, :], V, axes=(1, 0))
        # own leaf holdings are already inside A (children that are leaves)
        if np.max(np.abs(Vn - V)) < 1e-15 * max(1.0, np.max(np.abs(V))):
            V = Vn; break
        V = Vn
    return V


def crp_fund_returns(A, n_leaf, leafX, order):
    """Every fund rebalances to fixed fractions A[f] each step: x_f = sum_c A[f,c] x_c (DAG, bottom-up order)."""
    X = np.zeros((A.shape[0],) + leafX.shape[1:])
    X[:n_leaf] = leafX
    for f in order:
        X[f] = np.tensordot(A[f], X, axes=(0, 0))
    return X


def best_crp(x, iters=500):
    """Log-optimal constant portfolio over leaves for each question. x [N,T] -> alpha [N] (Cover's iteration)."""
    N = x.shape[0]
    a = np.full(N, 0.1 / N); a[np.argmax(np.log(x).sum(1))] += 0.9
    for _ in range(iters):
        d = a @ x
        a = a * (x / d[None]).mean(1)
        a /= a.sum()
    return a


def eg_crp(x, eta=0.05):
    """Online exponentiated-gradient portfolio (Helmbold et al. 1998), causal. x [N,T] -> log wealth."""
    N, T = x.shape
    a = np.full(N, 1.0 / N); lw = 0.0
    for t in range(T):
        d = a @ x[:, t]
        lw += np.log(d)
        a = a * np.exp(eta * x[:, t] / d); a /= a.sum()
    return lw


def run(seed, T):
    rng = np.random.default_rng(1000 + seed)
    bits, fam, _ = make_questions(64, T, seed)
    P, price = leaf_probs(bits)
    X = leaf_returns(P, price, bits)                       # [N,Q,T]
    N, Q, _ = X.shape
    logX = np.log(X)
    LV = np.cumsum(logX, axis=2)                           # log buy-and-hold leaf wealth, W0 = 1
    m = LV.max(0)                                          # common scale per (q,t); fund values are linear
    leafV = np.exp(LV - m[None])                           # scaled leaf wealth (true = leafV * e^m)
    out = {"seed": seed, "T": T, "leaves": LEAF_NAMES}

    # C6a: buy-and-hold DAG == flat Bayes mixture (two independent routes)
    A, root = random_dag(rng, N)
    V = bh_fund_values(A, N, leafV)
    lam = flat_weights(A, N, root)
    flat = np.tensordot(lam, leafV, axes=(0, 0))
    # independent route: root as ONE Kelly trader with Bayes-mixture belief (posterior weights from likelihoods)
    logpost = np.log(lam)[:, None] * np.ones((1, Q))
    lw_bayes = np.zeros(Q); lwpath = np.zeros((Q, T))
    y = bits.astype(bool)
    for t in range(T):
        wpost = np.exp(logpost - logpost.max(0)); wpost /= wpost.sum(0)
        pF = (wpost * P[:, :, t]).sum(0)
        lw_bayes += np.log(np.where(y[:, t], pF / price[:, t], (1 - pF) / (1 - price[:, t])))
        lwpath[:, t] = lw_bayes
        logpost += np.log(np.where(y[None, :, t], P[:, :, t], 1 - P[:, :, t]))
    out["C6a_max_abs_logdiff_dag_vs_flat"] = float(np.max(np.abs(np.log(V[root]) - np.log(flat))))
    out["C6a_max_abs_logdiff_bayes_trader_vs_flat"] = float(np.max(np.abs(lwpath - (np.log(flat) + m))))
    out["C6a_lambda_sum"] = float(lam.sum())

    # C6a (cyclic): two funds holding 30% / 20% of each other plus leaves
    nC = N + 2
    C = np.zeros((nC, nC))
    C[N, :N] = rng.dirichlet(np.ones(N)) * 0.7; C[N, N + 1] = 0.3
    C[N + 1, :N] = rng.dirichlet(np.ones(N)) * 0.8; C[N + 1, N] = 0.2
    # fractions are of initial capital (each fund starts at 1); cross-held shares are valued at market
    Vc = bh_fund_values(C, N, leafV[:, :4, :])
    lamc = flat_weights(C, N, N)
    out["C6a_cyclic_spectral_radius"] = float(np.max(np.abs(np.linalg.eigvals(C[N:, N:]))))
    out["C6a_cyclic_max_abs_logdiff"] = float(np.max(np.abs(np.log(Vc[N]) - np.log(np.tensordot(lamc, leafV[:, :4, :], axes=(0, 0))))))
    out["C6a_cyclic_lambda_sum"] = float(lamc.sum())

    # C6b: constant-rebalanced DAG == flat CRP with the same path-product weights
    order = list(range(N, A.shape[0]))  # L1, L2, root are already in bottom-up order
    Xf = crp_fund_returns(A, N, X, order)
    flatx = np.tensordot(lam, X, axes=(0, 0))
    out["C6b_max_abs_logdiff_crp_dag_vs_flat"] = float(np.max(np.abs(np.cumsum(np.log(Xf[root]), 1) - np.cumsum(np.log(flatx), 1))))

    # C6c: endogenous price pi_t = wealth-weighted mean belief; funds (2 roots) vs flattened leaves
    A2 = np.zeros((N + 2, N + 2))
    A2[N, :N] = rng.dirichlet(np.ones(N)); A2[N + 1, :N] = rng.dirichlet(np.ones(N))
    W_init = np.array([0.6, 0.4])
    lam2 = W_init[:, None] * A2[N:, :N]                  # leaf capital held via each fund
    Qc = 8
    def market(mode):
        pr = np.zeros((Qc, T))
        if mode == "flat":
            w = lam2.sum(0)[:, None] * np.ones((1, Qc))    # [N,Qc]
        else:
            w = lam2[:, :, None] * np.ones((1, 1, Qc))     # per-fund leaf holdings [2,N,Qc]
            fw = W_init[:, None] * np.ones((1, Qc))        # fund wealth for rebalancing
        for t in range(T):
            p = P[:, :Qc, t]
            if mode == "flat":
                pi = (w * p).sum(0) / w.sum(0)
                ret = np.where(y[None, :Qc, t], p / pi, (1 - p) / (1 - pi)); w = w * ret
            elif mode == "bh":
                # each fund is ONE trader whose belief is its Bayes mixture; its wealth is its holdings' value
                Wf = w.sum(1); pf = (w * p[None]).sum(1) / Wf
                pi = (Wf * pf).sum(0) / Wf.sum(0)
                ret = np.where(y[None, None, :Qc, t], p[None] / pi, (1 - p[None]) / (1 - pi)); w = w * ret
            else:  # rebalancing to fixed fractions: belief = fixed linear pool
                pf = np.tensordot(A2[N:, :N], p, axes=(1, 0))  # [2,Qc]
                pi = (fw * pf).sum(0) / fw.sum(0)
                fw = fw * np.where(y[None, :Qc, t], pf / pi, (1 - pf) / (1 - pi))
            pr[:, t] = pi
        return pr
    pflat, pbh, preb = market("flat"), market("bh"), market("reb")
    out["C6c_max_abs_price_diff_bh_vs_flat"] = float(np.max(np.abs(pbh - pflat)))
    out["C6c_max_abs_price_diff_rebalanced_vs_flat"] = float(np.max(np.abs(preb - pflat)))

    # C6d: growth per step vs market, per question, then per family
    g_leaf = logX.sum(2) / T                               # [N,Q]
    g_bh = (np.log(leafV[:, :, -1].mean(0)) + m[:, -1]) / T             # uniform buy-and-hold = Bayes mixture, uniform prior
    g_ucrp = np.log(X.mean(0)).sum(1) / T
    g_eg = np.array([eg_crp(X[:, q, :]) for q in range(Q)]) / T
    g_best = np.zeros(Q); a_best = np.zeros((Q, N))
    for q in range(Q):
        a_best[q] = best_crp(X[:, q, :], iters=2000)
        # a pure leaf is also a CRP; the MM iteration is monotone but slow near vertices
        g_best[q] = max(np.log(a_best[q] @ X[:, q, :]).sum() / T, g_leaf[:, q].max())
    rows = []
    for f, name in enumerate(FAMILIES):
        m = fam == f
        bl = g_leaf[:, m].max(0)
        rows.append({"family": name, "best_leaf": float(bl.mean()), "bh_uniform": float(g_bh[m].mean()),
                     "crp_uniform": float(g_ucrp[m].mean()), "crp_eg_online": float(g_eg[m].mean()),
                     "crp_best_hindsight": float(g_best[m].mean()),
                     "frac_q_eg_beats_best_leaf": float((g_eg[m] > bl).mean()),
                     "frac_q_bestcrp_gt_best_leaf_by_1e-3": float((g_best[m] > bl + 1e-3).mean())})
    out["C6d"] = rows
    bl_all = g_leaf.max(0)
    out["C6d_bh_minus_bestleaf_min"] = float((g_bh - bl_all).min())
    out["C6d_bh_minus_bestleaf_max"] = float((g_bh - bl_all).max())
    out["C6d_bound_ln10_over_T"] = float(np.log(N) / T)
    out["C6d_bestcrp_minus_bestleaf_min"] = float((g_best - bl_all).min())
    return out


def toy():
    pi, truth = 0.3, 0.5
    gr = lambda q: truth * np.log(q / pi) + (1 - truth) * np.log((1 - q) / (1 - pi))
    return {"leaf_0.9": gr(0.9), "leaf_0.1": gr(0.1), "bh_asymptotic": max(gr(0.9), gr(0.1)), "crp_50_50": gr(0.5)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=6000)
    a = ap.parse_args()
    res = run(a.seed, a.T)
    res["C6e_toy"] = toy()
    path = os.path.join(HERE, "out", f"L6-s{a.seed}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "C6d"}, indent=1))
    for r in res["C6d"]:
        print(r)
