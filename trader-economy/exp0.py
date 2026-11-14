"""Experiment 0: no hidden layer.  Agents are weights of a softmax regression.

Kelvin's question: is a hidden unit needed, or can an agent be just some of
the weights from inputs to outputs?  The softmax output is the LMSR: price =
softmax(b0 + x W), the logit sum is the market's share vector, and an agent's
trade is its additive logit contribution x[:, j] * W[j, c] for the entries it
owns.  Zero trade = zero weights.

Granularities (who owns which entries of W, D x C):
  weight  every entry its own agent            (D*C agents)
  column  all C weights out of one input j      (D agents, input "sellers")
  row     all D weights into one class c        (C agents)

Pay per round (fresh batch, before the update, so it is out of sample):
  v(S) = sum over the batch of log p_S(y) - log p0(y), where p_S keeps only
  the entries owned by agents in S (the rest zeroed).  The LMSR subsidy for
  the round is B * v(all).  Rules: loo = v(N) - v(N-i); msr1 = marginal in
  one fixed order; shap = MSR averaged over K random orders (sampled Shapley).
Training arms:
  backprop  every entry follows the gradient of the global log loss
  loo/msr1/shap  each agent follows the gradient of ITS OWN pay under that
            rule.  For loo that gradient is exactly the global one (v(N-i)
            does not depend on agent i's weights).  For shap it is the
            gradient of v(pre + i) for a random predecessor set: a dropout.
Compute cost c per active entry (|w| > 1e-3) per round, plus an L1 prox of
the same size in the update.  Ledger: agents + treasury + burned = start +
subsidy, asserted.  Below DEATH an agent dies (weights zeroed, remainder to
the treasury); a new agent takes the slot when the treasury can pay W0.
True value of an agent: Shapley of the REFIT game (best held-out log score a
softmax regression reaches using only the coalition's entries).
"""
import itertools
import json
import os
import pickle
import sys
import time
from math import factorial
from multiprocessing import Pool

import numpy as np

from core import mean_ci, spearman

HERE = os.path.dirname(os.path.abspath(__file__))
D, C = 8, 3
NAMES = ["x0 strong", "x1 medium", "x2 copy of x0", "x3 comp", "x4 comp", "x5 weak", "x6 noise", "x7 noise"]
BATCH, T, LR = 250, 800, 0.5
B, W0, DEATH, TREAS0 = 0.01, 1.0, 0.1, 5.0
ACTIVE = 1e-3


def make3(n, rng):
    x = rng.standard_normal((n, D))
    x[:, 2] = (x[:, 0] + 0.5 * rng.standard_normal(n)) / np.sqrt(1.25)
    u = rng.standard_normal(n)
    x[:, 3] = (u + 0.4 * rng.standard_normal(n)) / np.sqrt(1.16)
    x[:, 4] = (u + 0.4 * rng.standard_normal(n)) / np.sqrt(1.16)
    diff = (x[:, 3] - x[:, 4]) / (0.4 * np.sqrt(2) / np.sqrt(1.16))
    Z = np.stack([1.5 * x[:, 0] + 0.4 * x[:, 5], 1.2 * x[:, 1] - 0.6 * x[:, 0], 1.0 * diff - 0.4 * x[:, 5]], 1)
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    y = (rng.random(n)[:, None] > np.cumsum(P, 1)).sum(1)
    return x, y, P


def groups(gran):
    if gran == "weight":
        return [np.array([j * C + c]) for j in range(D) for c in range(C)]
    if gran == "column":
        return [np.array([j * C + c for c in range(C)]) for j in range(D)]
    if gran == "row":
        return [np.array([j * C + c for j in range(D)]) for c in range(C)]
    raise ValueError(gran)


def owner_matrix(G):
    M = np.zeros((len(G), D * C))
    for i, g in enumerate(G):
        M[i, g] = 1
    return M


def logp(x, W, b0):
    """log softmax for a stack of weight matrices W (M, D, C) -> (M, n, C)."""
    Z = np.einsum("nd,mdc->mnc", x, W) + b0
    Z -= Z.max(-1, keepdims=True)
    return Z - np.log(np.exp(Z).sum(-1, keepdims=True))


def values(x, y, Wflat, b0, E, lp0):
    """v for a stack of entry masks E (M, D*C): summed nats over the batch."""
    Ws = (E * Wflat).reshape(-1, D, C)
    L = logp(x, Ws, b0)
    return L[:, np.arange(len(y)), y].sum(1) - lp0


def grads(x, y, Wflat, b0, E):
    """gradient of v wrt W at each masked model (M, D*C), masked to E."""
    Ws = (E * Wflat).reshape(-1, D, C)
    L = logp(x, Ws, b0)
    R = -np.exp(L)
    R[:, np.arange(len(y)), y] += 1
    g = np.einsum("nd,mnc->mdc", x, R).reshape(len(E), -1)
    return g * E


def perm_masks(P, K, rng):
    """prefix masks of K random orders: (K*P, P) masks and the agent added."""
    rows, who = [], []
    for _ in range(K):
        o = rng.permutation(P)
        m = np.zeros(P)
        for i in o:
            m = m.copy()
            m[i] = 1
            rows.append(m)
            who.append(i)
    return np.array(rows), np.array(who)


def pay_rule(rule, x, y, W, b0, lp0, O, rng, K=16, order=None):
    P = len(O)
    full = values(x, y, W, b0, np.ones((1, D * C)), lp0)[0]
    if rule == "loo":
        A = 1 - np.eye(P)
        return full - values(x, y, W, b0, A @ O, lp0), full
    if rule == "msr1":
        o = order
        A = np.zeros((P + 1, P))
        for k, i in enumerate(o):
            A[k + 1] = A[k]
            A[k + 1, i] = 1
        v = values(x, y, W, b0, A @ O, lp0)
        phi = np.zeros(P)
        phi[o] = np.diff(v)
        return phi, full
    if rule == "shap":
        A, who = perm_masks(P, K, rng)
        v = values(x, y, W, b0, A @ O, lp0)
        v = v.reshape(K, P)
        prev = np.concatenate([np.zeros((K, 1)), v[:, :-1]], 1)
        d = (v - prev).ravel()
        phi = np.zeros(P)
        np.add.at(phi, who, d)
        return phi / K, full
    if rule.startswith("semi"):
        # semivalue: each other agent present independently with prob pi
        # (pi -> 1 is LOO; uniform coalition size is Shapley); rescaled so the
        # payments sum to the pot (budget balance by normalisation)
        pi = float(rule[4:])
        Q = 16
        who = np.repeat(np.arange(P), Q)
        A = (rng.random((P * Q, P)) < pi).astype(float)
        A[np.arange(P * Q), who] = 0
        Ai = A.copy()
        Ai[np.arange(P * Q), who] = 1
        d = values(x, y, W, b0, Ai @ O, lp0) - values(x, y, W, b0, A @ O, lp0)
        phi = d.reshape(P, Q).mean(1)
        if abs(phi.sum()) > 1e-9:
            phi = phi * full / phi.sum()
        return phi, full
    raise ValueError(rule)


def own_grad(rule, x, y, W, b0, O, rng, Q=8, order=None):
    """each agent's gradient of its own pay, on its own entries (D*C,)."""
    P = len(O)
    if rule in ("backprop", "loo"):
        return grads(x, y, W, b0, np.ones((1, D * C)))[0]
    if rule == "msr1":
        A = np.zeros((P, P))
        m = np.zeros(P)
        for i in order:
            m = m.copy()
            m[i] = 1
            A[i] = m
        g = grads(x, y, W, b0, A @ O)
        return (g * O).sum(0)
    if rule == "shap":
        A = np.zeros((P * Q, P))
        who = np.repeat(np.arange(P), Q)
        for r, i in enumerate(who):
            o = rng.permutation(P)
            pos = int(np.flatnonzero(o == i)[0])
            A[r, o[:pos + 1]] = 1
        g = grads(x, y, W, b0, A @ O)
        g = (g * O[who]).reshape(P, Q, -1).mean(1)
        return g.sum(0)
    if rule.startswith("semi"):
        pi = float(rule[4:])
        who = np.repeat(np.arange(P), Q)
        A = (rng.random((P * Q, P)) < pi).astype(float)
        A[np.arange(P * Q), who] = 1
        g = grads(x, y, W, b0, A @ O)
        g = (g * O[who]).reshape(P, Q, -1).mean(1)
        return g.sum(0)
    raise ValueError(rule)


# ---------------------------------------------------------------- refit game
def fit_softmax(x, y, free, iters=15, ridge=1e-4):
    """Newton for softmax regression with only `free` entries of W (plus
    intercepts) trainable.  Returns W flat and intercepts."""
    n = len(y)
    idx = np.flatnonzero(free)
    k = len(idx)
    th = np.zeros(k + C)
    Y = np.eye(C)[y]
    jj, cc = idx // C, idx % C
    for _ in range(iters):
        W = np.zeros(D * C)
        W[idx] = th[:k]
        Z = x @ W.reshape(D, C) + th[k:]
        Z -= Z.max(1, keepdims=True)
        Pm = np.exp(Z)
        Pm /= Pm.sum(1, keepdims=True)
        R = Y - Pm
        # design: parameter a = (j,c) -> feature x_j on class c; intercepts
        F = np.concatenate([x[:, jj], np.ones((n, C))], 1)   # (n, k+C) feature
        cls = np.concatenate([cc, np.arange(C)])             # class of each param
        g = (F * R[:, cls]).sum(0) - ridge * th
        # Hessian: sum_n F_a F_b (P_ca delta - P_ca P_cb)
        Pa = Pm[:, cls]
        H = (F * Pa).T @ F * (cls[:, None] == cls[None, :]) - (F * Pa).T @ (F * Pa)
        H += ridge * np.eye(k + C)
        th += np.linalg.solve(H, g)
    W = np.zeros(D * C)
    W[idx] = th[:k]
    return W, th[k:]


def refit_value(x, y, xt, yt, free, lp0t):
    W, b = fit_softmax(x, y, free)
    L = logp(xt, W.reshape(1, D, C), b)[0]
    return float(L[np.arange(len(yt)), yt].mean() - lp0t / len(yt))


def shapley_exact_from(vals, P):
    M = 1 << P
    pc = np.array([bin(m).count("1") for m in range(M)])
    wt = np.array([factorial(k) * factorial(P - k - 1) / factorial(P) for k in range(P)])
    masks = np.arange(M)
    phi = np.zeros(P)
    for i in range(P):
        S = masks[(masks >> i & 1) == 0]
        phi[i] = np.sum(wt[pc[S]] * (vals[S | (1 << i)] - vals[S]))
    return phi


def refit_truth():
    """True (information) value per agent at each granularity, nats/example."""
    path = os.path.join(HERE, "results", "exp0_truth.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    rng = np.random.default_rng(777)
    x, y, _ = make3(20000, rng)
    xt, yt, Pt = make3(20000, rng)
    p0 = np.bincount(yt, minlength=C) / len(yt)
    lp0t = float(np.log(p0)[yt].sum())
    out = {"bayes_gain": float(np.log(Pt[np.arange(len(yt)), yt]).mean() - lp0t / len(yt))}
    for gran in ("row", "column"):
        O = owner_matrix(groups(gran))
        P = len(O)
        vals = np.array([refit_value(x, y, xt, yt, (np.array([m >> i & 1 for i in range(P)]) @ O) > 0, lp0t)
                         for m in range(1 << P)])
        out[gran] = shapley_exact_from(vals, P)
        out[gran + "_full"] = float(vals[-1])
    # weight level: sampled Shapley over 60 random orders (24 players)
    O = owner_matrix(groups("weight"))
    P = len(O)
    memo = {}
    phi = np.zeros(P)
    K = 60
    for _ in range(K):
        o = rng.permutation(P)
        m = 0
        prev = 0.0
        for i in o:
            m |= 1 << int(i)
            if m not in memo:
                memo[m] = refit_value(x, y, xt, yt, (np.array([m >> a & 1 for a in range(P)]) @ O) > 0, lp0t)
            phi[i] += memo[m] - prev
            prev = memo[m]
    out["weight"] = phi / K
    pickle.dump(out, open(path, "wb"))
    return out


def prior_logits():
    """market maker's starting price: class frequencies of the generator."""
    yb = make3(100000, np.random.default_rng(99))[1]
    return np.log(np.bincount(yb, minlength=C) / len(yb))


def ig_pay(x, y, W, b0, O, m=50):
    """Aumann-Shapley / integrated-gradients pay per agent (summed nats).
    Additive logits: pay_i = <z_i, e_y - pbar>, pbar = path-averaged price."""
    Wm = W.reshape(D, C)
    q = x @ Wm
    pbar = np.zeros_like(q)
    for a in (np.arange(m) + 0.5) / m:
        Z = b0 + a * q
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        pbar += E / E.sum(1, keepdims=True)
    R = np.eye(C)[y] - pbar / m
    contrib = np.einsum("nd,nc->dc", x, R) * Wm          # per entry
    return O @ contrib.ravel()


# ------------------------------------------------------------------ economy
def run(cfg):
    gran, arm, cost, seed = cfg["gran"], cfg["arm"], cfg["cost"], cfg["seed"]
    Tn = cfg.get("T", T)
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(1000 + seed)          # same data stream for every arm
    G = groups(gran)
    O = owner_matrix(G)
    P = len(G)
    truth = refit_truth()[gran]
    b0 = prior_logits()
    order = np.random.default_rng(seed + 55).permutation(P)
    econ = arm != "backprop"
    rule = arm if econ else "shap"
    W = np.zeros(D * C)
    wealth = np.full(P, W0)
    alive = np.ones(P, bool)
    treasury, burned, subsidy = TREAS0, 0.0, 0.0
    start = wealth.sum() + treasury
    kappa = cost / B / BATCH                               # L1 per unit |w|, mean-nats units
    pay_hist = np.zeros(P)
    pot_hist = 0.0
    deaths = np.zeros(P, int)
    budget = []
    for t in range(Tn):
        x, y, _ = make3(BATCH, drng)
        lp0 = float(b0[y].sum())
        # respawn
        if econ:
            for i in np.flatnonzero(~alive):
                if treasury >= W0:
                    treasury -= W0
                    wealth[i] = W0
                    alive[i] = True
        # pay on the fresh batch (before learning from it)
        phi, full = pay_rule(rule, x, y, W, b0, lp0, O, rng, order=order)
        subsidy += B * full
        treasury += B * full - B * phi.sum()               # regulator keeps / covers the gap (LOO)
        budget.append((phi.sum(), full))
        wealth += B * phi * alive
        treasury += B * (phi * ~alive).sum()                # dead slots have zero weights -> phi ~ 0
        active = (np.abs(W) > ACTIVE) @ O.T
        ccost = np.minimum(cost * active, np.maximum(wealth, 0)) * alive
        wealth -= ccost
        burned += ccost.sum()
        if t >= Tn // 2:
            pay_hist += B * phi
            pot_hist += B * full
        # learning: each agent on its own entries
        g = own_grad(arm, x, y, W, b0, O, rng, order=order) / BATCH
        W = W + LR * g
        W = np.sign(W) * np.maximum(np.abs(W) - LR * kappa, 0)
        # bankruptcy
        if econ:
            dead = alive & (wealth < DEATH * W0)
            for i in np.flatnonzero(dead):
                treasury += wealth[i]
                wealth[i] = 0.0
                alive[i] = False
                W[G[i]] = 0.0
                deaths[i] += 1
    tot = wealth.sum() + treasury + burned
    assert abs(tot - (start + subsidy)) < 1e-6 * max(1.0, abs(tot)), (tot, start + subsidy)
    # evaluation on a test set
    xt, yt, Pt = make3(20000, np.random.default_rng(4242))
    lp0t = float(b0[yt].sum())
    L = logp(xt, W.reshape(1, D, C), b0)[0]
    ll = float(-L[np.arange(len(yt)), yt].mean())
    ll_bayes = float(-np.log(Pt[np.arange(len(yt)), yt]).mean())
    # zero-out Shapley at the final weights on the test set ("unit Shapley")
    if P <= 12:
        A = np.array([[m >> i & 1 for i in range(P)] for m in range(1 << P)], float)
        vals = np.concatenate([values(xt[s:s + 5000], yt[s:s + 5000], W, b0, A @ O, float(b0[yt[s:s + 5000]].sum()))[None]
                               for s in range(0, 20000, 5000)]).sum(0) / len(yt)
        zshap = shapley_exact_from(vals, P)
    else:
        zr = np.random.default_rng(1)
        zshap = sum(pay_rule("shap", xt[:2000], yt[:2000], W, b0, float(b0[yt[:2000]].sum()), O, zr, K=8)[0]
                    for _ in range(12)) / 12 / 2000
    igp = ig_pay(xt, yt, W, b0, O) / len(yt)
    per_round = pay_hist / (Tn - Tn // 2)
    Wm = W.reshape(D, C)
    feat_norm = np.abs(Wm).sum(1)
    bs = np.array(budget[Tn // 2:])
    return dict(cfg=cfg, ll=ll, ll_bayes=ll_bayes, W=W.tolist(), feat_norm=feat_norm.tolist(),
                pay=per_round.tolist(), zshap=zshap.tolist(), truth=truth.tolist(),
                sp_pay_truth=spearman(per_round, truth), sp_zshap_truth=spearman(zshap, truth),
                sp_pay_zshap=spearman(per_round, zshap), ig=igp.tolist(),
                sp_ig_truth=spearman(igp, truth), sp_ig_zshap=spearman(igp, zshap),
                budget_ratio=float(bs[:, 0].sum() / bs[:, 1].sum()),
                deaths=deaths.tolist(), alive_end=int(alive.sum()),
                active_end=int((np.abs(W) > ACTIVE).sum()),
                wealth_end=wealth.tolist(), pot_per_round=pot_hist / (Tn - Tn // 2))


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    refit_truth()
    cfgs = []
    seeds = range(5)
    for gran in ("weight", "column", "row"):
        for arm in ("backprop", "loo", "msr1", "shap"):
            for cost in (0.0, 0.01, 0.05):
                for s in seeds:
                    cfgs.append(dict(gran=gran, arm=arm, cost=cost, seed=s))
    if what == "semi":
        cfgs = [dict(gran=g, arm=f"semi{pi}", cost=0.01, seed=s) for g in ("column", "weight")
                for pi in (0.5, 0.8, 0.95) for s in seeds]
    if what == "quick":
        cfgs = [dict(gran=g, arm=a, cost=0.01, seed=0, T=200) for g in ("weight", "column", "row") for a in ("backprop", "shap")]
    t0 = time.time()
    with Pool(8) as pool:
        rows = pool.map(run, cfgs)
    json.dump(rows, open(os.path.join(HERE, "results", f"exp0_{what}_rows.json"), "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", what, round(time.time() - t0), "s", len(rows))


if __name__ == "__main__":
    main()
