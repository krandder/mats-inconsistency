"""Experiment A: hidden units as agents.

An AGENT is a parameter set: hidden unit h = incoming weights + bias +
outgoing weight (1 hidden layer).  The output sigmoid is the binary LMSR
(b=1); the unit's trade is its additive logit contribution a_h(x) * v_h.
The output bias b0 is fixed at the base-rate log odds (the market maker's
opening price) and belongs to no one.

Pay per round (fresh batch, before learning from it):
  v(S) = sum over batch of log p_S(y) - log p0(y), units outside S ablated
  (zero-out: a_h := 0; the mean-ablation check replaces a_h by its mean on a
  reference batch).  Rules: loo, msr1 (one fixed order), shap (MSR averaged
  over K=16 random orders).  All three are computed every round for
  reporting; the ledger uses the arm's own rule (backprop arms: shap).
Training arms:
  backprop      global log-loss gradient (+ the same L1 prox as the others)
  backprop_alt  same, different init seed (reference for "same function")
  loo           own-pay gradient == backprop, plus ledger/bankruptcy
  msr1          grad of v(prefix up to and incl. h) in a fixed order
  shap          grad of v(S+h) for Q=8 random predecessor sets S (dropout)
Ledger: agents + treasury + burned == start + subsidy (asserted).  Compute
cost per active connection per round (|w| > 1e-3: incoming, bias, outgoing).
Bankruptcy below DEATH*W0 zeroes the unit; respawn = fresh random init paid
by the treasury.  Planted useless units: slots 0..3 may only connect to pure
noise features (a fixed structural mask, kept on respawn).

2-hidden-layer variant (synthetic only): every unit owns its incoming
weights + bias; last-layer units also own their output weight.  Pay =
ablation Shapley of the WHOLE-network log score with all units of both
layers as players (so upstream units are paid their share of the value that
flows through their readers; the pot is split across both layers).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from core import make_data, mean_ci, spearman, sigmoid

HERE = os.path.dirname(os.path.abspath(__file__))
B, W0, DEATH, TREAS0 = 0.01, 1.0, 0.1, 5.0
ACT_T = 1e-3
N_NOISE_UNITS = 4


def logsig(z):
    return -np.logaddexp(0.0, -z)


def score(z, y):
    """log p(y) for logit z (broadcast; y in {0,1})."""
    t = (2 * y - 1) * z
    return -(np.maximum(-t, 0) + np.log1p(np.exp(-np.abs(t))))


def act(pre, kind):
    if kind == "tanh":
        a = np.tanh(pre)
        return a, 1 - a * a
    a = np.maximum(pre, 0)
    return a, (pre > 0).astype(float)


# ------------------------------------------------------------------ data
def load_wdbc(seed, n_noise=4):
    rows = [l.strip().split(",") for l in open(os.path.join(HERE, "data", "wdbc.data")) if l.strip()]
    y = np.array([1.0 if r[1] == "M" else 0.0 for r in rows])
    X = np.array([[float(v) for v in r[2:]] for r in rows])
    rng = np.random.default_rng(500 + seed)
    X = np.hstack([X, rng.standard_normal((len(X), n_noise))])    # appended pure-noise columns
    idx = rng.permutation(len(y))
    ntr = int(0.7 * len(y))
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    X = (X - mu) / sd
    return X[tr], y[tr], X[te], y[te]


def fit_lr(X, y, lam=1.0, iters=30):
    X1 = np.hstack([X, np.ones((len(X), 1))])
    th = np.zeros(X1.shape[1])
    R = lam * np.eye(len(th))
    R[-1, -1] = 1e-6
    for _ in range(iters):
        p = sigmoid(X1 @ th)
        g = X1.T @ (y - p) - R @ th
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + R
        th += np.linalg.solve(H, g)
    return th


def lr_logits(th, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ th


# ------------------------------------------------------ one hidden layer
class Net1:
    def __init__(self, d, H, rng, kind, conn):
        self.d, self.H, self.kind = d, H, kind
        self.conn = conn                                   # (H, d) structural mask
        self.W1 = rng.standard_normal((H, d)) / np.sqrt(conn.sum(1, keepdims=True)) * conn
        self.b1 = np.zeros(H)
        self.v = rng.standard_normal(H) * 0.1
        self.b0 = 0.0

    def respawn(self, h, rng):
        self.W1[h] = rng.standard_normal(self.d) / np.sqrt(self.conn[h].sum()) * self.conn[h]
        self.b1[h] = 0.0
        self.v[h] = rng.standard_normal() * 0.1

    def kill(self, h):
        self.W1[h] = 0.0
        self.b1[h] = 0.0
        self.v[h] = 0.0

    def forward(self, x):
        pre = x @ self.W1.T + self.b1
        a, da = act(pre, self.kind)
        return a, da

    def contrib(self, x):
        a, da = self.forward(x)
        return a * self.v, a, da

    def active(self):
        return ((np.abs(self.W1) > ACT_T).sum(1) + (np.abs(self.b1) > ACT_T) + (np.abs(self.v) > ACT_T))

    def logits(self, x):
        return self.b0 + self.contrib(x)[0].sum(1)


def v_masks(c, y, b0, M, cbar=None):
    """v(S) summed over the batch for masks M (m, H); c: (n, H) contributions."""
    z = b0 + c @ M.T
    if cbar is not None:
        z = z + (1 - M) @ cbar
    return (score(z, y[:, None]) - score(b0, y)[:, None]).sum(0)


def prefix_masks(orders, H):
    rows, who = [], []
    for o in orders:
        m = np.zeros(H)
        for i in o:
            m = m.copy()
            m[i] = 1
            rows.append(m)
            who.append(i)
    return np.array(rows), np.array(who)


def shap_from_orders(vfun, orders, H):
    A, who = prefix_masks(orders, H)
    v = vfun(A).reshape(len(orders), H)
    prev = np.concatenate([np.zeros((len(orders), 1)), v[:, :-1]], 1)
    phi = np.zeros(H)
    np.add.at(phi, who, (v - prev).ravel())
    return phi / len(orders)


def all_pays(c, y, b0, H, rng, order, K=16, cbar=None):
    vf = lambda M: v_masks(c, y, b0, M, cbar)
    full = vf(np.ones((1, H)))[0]
    loo = full - vf(1 - np.eye(H))
    A, who = prefix_masks([order], H)
    vv = vf(A)
    msr1 = np.zeros(H)
    msr1[order] = np.diff(np.concatenate([[0.0], vv]))
    shap = shap_from_orders(vf, [rng.permutation(H) for _ in range(K)], H)
    return dict(loo=loo, msr1=msr1, shap=shap), full


def residuals(arm, c, y, b0, H, rng, order, Q=8):
    """per-unit residual (n, H): d own-pay / d own logit contribution."""
    if arm in ("backprop", "backprop_alt", "loo"):
        r = y - sigmoid(b0 + c.sum(1))
        return np.repeat(r[:, None], H, 1)
    if arm == "msr1":
        A = np.zeros((H, H))
        m = np.zeros(H)
        for i in order:
            m = m.copy()
            m[i] = 1
            A[i] = m
        return y[:, None] - sigmoid(b0 + c @ A.T)
    if arm == "shap":
        A = np.zeros((H * Q, H))
        who = np.repeat(np.arange(H), Q)
        for r_, i in enumerate(who):
            o = rng.permutation(H)
            pos = int(np.flatnonzero(o == i)[0])
            A[r_, o[:pos + 1]] = 1
        R = y[:, None] - sigmoid(b0 + c @ A.T)
        return R.reshape(len(y), H, Q).mean(2)
    raise ValueError(arm)


def neuron_shapley(net, x, y, K, rng, ablate="zero", xref=None):
    """held-out unit Shapley, nats per example (Ghorbani & Zou style sampling)."""
    c, a, _ = net.contrib(x)
    cbar = None
    if ablate == "mean":
        ar, _ = net.forward(xref)
        cbar = ar.mean(0) * net.v
    H = net.H
    orders = [rng.permutation(H) for _ in range(K)]
    phi = np.zeros(H)
    for s in range(0, K, 25):
        phi += shap_from_orders(lambda M: v_masks(c, y, net.b0, M, cbar), orders[s:s + 25], H) * len(orders[s:s + 25])
    return phi / K / len(y)


def dup_test(net, x, y, rng, K=200):
    """copy the most valuable unit, halve v on both copies (function unchanged)."""
    c, _, _ = net.contrib(x)
    H = net.H
    vf = lambda M: v_masks(c, y, net.b0, M)
    phi = shap_from_orders(vf, [rng.permutation(H) for _ in range(K)], H)
    h = int(np.argmax(phi))
    c2 = np.hstack([c, c[:, h:h + 1] / 2])
    c2[:, h] /= 2
    vf2 = lambda M: v_masks(c2, y, net.b0, M)
    phi2 = shap_from_orders(vf2, [rng.permutation(H + 1) for _ in range(K)], H + 1)
    full = vf(np.ones((1, H)))[0]
    loo1 = full - vf((1 - np.eye(H))[h:h + 1])[0]
    loo2 = full - vf2(1 - np.eye(H + 1))
    o = list(rng.permutation(H))          # copy enters right after the original in a fixed order
    o2 = o[:o.index(h) + 1] + [H] + o[o.index(h) + 1:]
    A, _ = prefix_masks([o], H)
    m1 = np.diff(np.concatenate([[0.0], vf(A)]))[o.index(h)]
    A2, _ = prefix_masks([o2], H + 1)
    d2 = np.diff(np.concatenate([[0.0], vf2(A2)]))
    n = len(y)
    return dict(shap_orig=phi[h] / n, shap_a=phi2[h] / n, shap_b=phi2[H] / n,
                loo_orig=loo1 / n, loo_a=loo2[h] / n, loo_b=loo2[H] / n,
                msr1_orig=m1 / n, msr1_a=d2[o2.index(h)] / n, msr1_b=d2[o2.index(H)] / n)


def temp_fit(zr, yr, b0, zt, yt):
    """diagnostic: best global logit scale s (fit on reference/train data,
    applied to test): z -> b0 + s*(z - b0).  Returns (s, test log loss)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(3), 60))
    ls = [-score(b0 + g * (zr - b0), yr).mean() for g in grid]
    g = grid[int(np.argmin(ls))]
    return float(g), float(-score(b0 + g * (zt - b0), yt).mean())


def run1(cfg):
    task, arm, cost, seed = cfg["task"], cfg["arm"], cfg["cost"], cfg["seed"]
    H, kind = cfg.get("H", 32), cfg.get("act", "tanh")
    init_seed = seed + (100 if arm == "backprop_alt" else 0)
    rng = np.random.default_rng(10_000 + seed)            # pay sampling / orders
    irng = np.random.default_rng(init_seed)                 # init + respawn draws
    if task == "synth":
        d, BATCH, T, LR = 12, 250, cfg.get("T", 1200), 0.2
        noise_feats = [8, 9, 10, 11]
        drng = np.random.default_rng(1000 + seed)
        xt, yt, pt = make_data(20000, np.random.default_rng(4242 + seed))
        xref, yref, _ = make_data(2000, np.random.default_rng(77 + seed))
        base = 0.5
        get_batch = lambda t: make_data(BATCH, drng)[:2]
        wd = 0.0
    else:
        Xtr, ytr, xt, yt = load_wdbc(seed)
        d, BATCH, T, LR = Xtr.shape[1], 64, cfg.get("T", 1500), 0.1
        noise_feats = list(range(30, d))
        drng = np.random.default_rng(1000 + seed)
        xref, yref = Xtr, ytr
        base = ytr.mean()
        perm = []

        def get_batch(t):
            nonlocal perm
            if len(perm) < BATCH:
                perm = list(drng.permutation(len(ytr)))
            idx, perm = perm[:BATCH], perm[BATCH:]
            return Xtr[idx], ytr[idx]
        wd = 1e-3
    conn = np.ones((H, d))
    conn[:N_NOISE_UNITS] = 0
    conn[:N_NOISE_UNITS, noise_feats] = 1
    net = Net1(d, H, irng, kind, conn)
    net.b0 = np.log(base / (1 - base))
    order = np.random.default_rng(seed + 55).permutation(H)
    econ = arm in ("loo", "msr1", "shap")
    rule = arm if econ else "shap"
    wealth = np.full(H, W0)
    alive = np.ones(H, bool)
    treasury, burned, subsidy = TREAS0, 0.0, 0.0
    start = wealth.sum() + treasury
    kappa = cost / B / BATCH
    hist = {k: np.zeros(H) for k in ("loo", "msr1", "shap")}
    nh = 0
    deaths = np.zeros(H, int)
    budget = []
    for t in range(T):
        x, y = get_batch(t)
        if econ:
            for i in np.flatnonzero(~alive):
                if treasury >= W0:
                    treasury -= W0
                    wealth[i] = W0
                    alive[i] = True
                    net.respawn(i, irng)
        c, a, da = net.contrib(x)
        pays, full = all_pays(c, y, net.b0, H, rng, order)
        phi = pays[rule]
        subsidy += B * full
        treasury += B * full - B * phi.sum()
        budget.append((phi.sum(), full))
        wealth += B * phi * alive
        treasury += B * (phi * ~alive).sum()
        ccost = np.minimum(cost * net.active(), np.maximum(wealth, 0)) * alive
        wealth -= ccost
        burned += ccost.sum()
        if t >= (3 * T) // 4:
            for k in hist:
                hist[k] += B * pays[k]
            nh += 1
        # learning
        R = residuals(arm, c, y, net.b0, H, rng, order)
        gv = (R * a).sum(0) / BATCH
        gpre = R * net.v * da
        gW = gpre.T @ x / BATCH
        gb = gpre.sum(0) / BATCH
        net.v += LR * (gv - wd * net.v)
        net.W1 += LR * (gW - wd * net.W1) * conn
        net.b1 += LR * gb
        net.W1 = np.sign(net.W1) * np.maximum(np.abs(net.W1) - LR * kappa, 0)
        net.v = np.sign(net.v) * np.maximum(np.abs(net.v) - LR * kappa, 0)
        if econ:
            dead = alive & (wealth < DEATH * W0)
            for i in np.flatnonzero(dead):
                treasury += wealth[i]
                wealth[i] = 0.0
                alive[i] = False
                net.kill(i)
                deaths[i] += 1
    tot = wealth.sum() + treasury + burned
    assert abs(tot - (start + subsidy)) < 1e-6 * max(1.0, abs(tot)), (tot, start + subsidy)
    # ---- evaluation
    zt = net.logits(xt)
    ll = float(-score(zt, yt).mean())
    ll0 = float(-score(net.b0, yt).mean())
    if task == "synth":
        xs, ys, _ = make_data(20000, np.random.default_rng(31 + seed))
        th = fit_lr(xs, ys, lam=1e-3)
    else:
        th = fit_lr(xref, yref, lam=1.0)
    ll_lr = float(-score(lr_logits(th, xt), yt).mean())
    ns = min(len(yt), 3000)
    trng = np.random.default_rng(9)
    truth = neuron_shapley(net, xt[:ns], yt[:ns], 200, trng)
    truth_mean = neuron_shapley(net, xt[:ns], yt[:ns], 200, np.random.default_rng(9), "mean", xref)
    per = {k: (hist[k] / max(nh, 1)).tolist() for k in hist}
    bs = np.array(budget[(3 * T) // 4:])
    useless = np.arange(H) < N_NOISE_UNITS
    tscale, ll_temp = temp_fit(net.logits(xref), yref, net.b0, zt, yt)
    out = dict(cfg=cfg, ll=ll, ll0=ll0, ll_lr=ll_lr, tscale=tscale, ll_temp=ll_temp, zt=zt[:2000].round(4).tolist(),
               pay=per, truth=truth.tolist(), truth_mean=truth_mean.tolist(),
               sp={k: spearman(per[k], truth) for k in per},
               sp_zero_vs_mean=spearman(truth, truth_mean),
               budget_ratio=float(bs[:, 0].sum() / bs[:, 1].sum()),
               deaths=deaths.tolist(), alive_end=int(alive.sum()),
               deaths_useless=int(deaths[useless].sum()), deaths_useful=int(deaths[~useless].sum()),
               vabs_useless=float(np.abs(net.v[useless]).mean()), vabs_useful=float(np.abs(net.v[~useless]).mean()),
               truth_useless=float(truth[useless].sum()),
               pay_useless=float(np.sum(per[rule][:N_NOISE_UNITS])),
               active_conn=int(net.active().sum()), burned=burned, subsidy=subsidy,
               gain=ll0 - ll, dup=dup_test(net, xt[:ns], yt[:ns], np.random.default_rng(3)))
    if task == "synth":
        out["ll_bayes"] = float(-score(np.log(pt / (1 - pt)), yt).mean())
    return out


# ------------------------------------------------------ two hidden layers
class Net2:
    def __init__(self, d, H1, H2, rng, kind, conn):
        self.d, self.H1, self.H2, self.kind, self.conn = d, H1, H2, kind, conn
        self.P = H1 + H2
        self.W1 = rng.standard_normal((H1, d)) / np.sqrt(conn.sum(1, keepdims=True)) * conn
        self.b1 = np.zeros(H1)
        self.W2 = rng.standard_normal((H2, H1)) / np.sqrt(H1)
        self.b2 = np.zeros(H2)
        self.v = rng.standard_normal(H2) * 0.1
        self.b0 = 0.0

    def respawn(self, p, rng):
        if p < self.H1:
            self.W1[p] = rng.standard_normal(self.d) / np.sqrt(self.conn[p].sum()) * self.conn[p]
            self.b1[p] = 0.0
        else:
            j = p - self.H1
            self.W2[j] = rng.standard_normal(self.H1) / np.sqrt(self.H1)
            self.b2[j] = 0.0
            self.v[j] = rng.standard_normal() * 0.1

    def kill(self, p):
        if p < self.H1:
            self.W1[p] = 0.0
            self.b1[p] = 0.0
        else:
            j = p - self.H1
            self.W2[j] = 0.0
            self.b2[j] = 0.0
            self.v[j] = 0.0

    def active(self):
        a1 = (np.abs(self.W1) > ACT_T).sum(1) + (np.abs(self.b1) > ACT_T)
        a2 = (np.abs(self.W2) > ACT_T).sum(1) + (np.abs(self.b2) > ACT_T) + (np.abs(self.v) > ACT_T)
        return np.concatenate([a1, a2])

    def fwd(self, x, M):
        """masked forward for masks M (m, P).  Returns z (m, n) and caches."""
        M1, M2 = M[:, :self.H1], M[:, self.H1:]
        a1, da1 = act(x @ self.W1.T + self.b1, self.kind)          # (n, H1)
        a1m = a1[None] * M1[:, None, :]                              # (m, n, H1)
        a2, da2 = act(a1m @ self.W2.T + self.b2, self.kind)          # (m, n, H2)
        a2m = a2 * M2[:, None, :]
        z = self.b0 + a2m @ self.v
        return z, (a1, da1, a1m, a2m, da2, M2)

    def vals(self, x, y, M, chunk=256):
        out = []
        for s in range(0, len(M), chunk):
            z, _ = self.fwd(x, M[s:s + chunk])
            out.append((score(z, y[None]) - score(self.b0, y)[None]).sum(1))
        return np.concatenate(out)

    def own_grads(self, x, y, M, owner):
        """gradient of v(M_r) wrt the params of owner[r], summed per owner."""
        z, (a1, da1, a1m, a2m, da2, M2) = self.fwd(x, M)
        r = y[None] - sigmoid(z)                                     # (m, n)
        G2 = r[:, :, None] * self.v * da2 * M2[:, None, :]           # d/dpre2 (m, n, H2)
        gW1 = np.zeros_like(self.W1); gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2); gb2 = np.zeros_like(self.b2); gv = np.zeros_like(self.v)
        cnt = np.zeros(self.P)
        for rr, p in enumerate(owner):
            cnt[p] += 1
            if p < self.H1:
                g1 = (G2[rr] @ self.W2[:, p]) * da1[:, p]
                gW1[p] += g1 @ x
                gb1[p] += g1.sum()
            else:
                j = p - self.H1
                gv[j] += r[rr] @ a2m[rr, :, j]
                gW2[j] += G2[rr, :, j] @ a1m[rr]
                gb2[j] += G2[rr, :, j].sum()
        c1, c2 = np.maximum(cnt[:self.H1], 1), np.maximum(cnt[self.H1:], 1)
        return gW1 / c1[:, None], gb1 / c1, gW2 / c2[:, None], gb2 / c2, gv / c2


def run2(cfg):
    arm, cost, seed = cfg["arm"], cfg["cost"], cfg["seed"]
    H1, H2, kind = 16, 16, "tanh"
    d, BATCH, T, LR, Q, K = 12, 250, cfg.get("T", 800), 0.2, 4, 8
    rng = np.random.default_rng(10_000 + seed)
    irng = np.random.default_rng(seed + (100 if arm == "backprop_alt" else 0))
    drng = np.random.default_rng(1000 + seed)
    xt, yt, pt = make_data(20000, np.random.default_rng(4242 + seed))
    conn = np.ones((H1, d))
    conn[:N_NOISE_UNITS] = 0
    conn[:N_NOISE_UNITS, [8, 9, 10, 11]] = 1
    net = Net2(d, H1, H2, irng, kind, conn)
    P = net.P
    econ = arm in ("loo", "shap")
    wealth = np.full(P, W0)
    alive = np.ones(P, bool)
    treasury, burned, subsidy = TREAS0, 0.0, 0.0
    start = wealth.sum() + treasury
    kappa = cost / B / BATCH
    hist = np.zeros(P)
    hist_loo = np.zeros(P)
    nh = 0
    deaths = np.zeros(P, int)
    for t in range(T):
        x, y, _ = make_data(BATCH, drng)
        if econ:
            for i in np.flatnonzero(~alive):
                if treasury >= W0:
                    treasury -= W0
                    wealth[i] = W0
                    alive[i] = True
                    net.respawn(i, irng)
        full = net.vals(x, y, np.ones((1, P)))[0]
        orders = [rng.permutation(P) for _ in range(K)]
        shap = shap_from_orders(lambda M: net.vals(x, y, M), orders, P)
        loo = full - net.vals(x, y, 1 - np.eye(P))
        phi = loo if arm == "loo" else shap
        subsidy += B * full
        treasury += B * full - B * phi.sum()
        wealth += B * phi * alive
        treasury += B * (phi * ~alive).sum()
        ccost = np.minimum(cost * net.active(), np.maximum(wealth, 0)) * alive
        wealth -= ccost
        burned += ccost.sum()
        if t >= (3 * T) // 4:
            hist += B * shap
            hist_loo += B * loo
            nh += 1
        if arm == "shap":
            A = np.zeros((P * Q, P))
            owner = np.repeat(np.arange(P), Q)
            for r_, i in enumerate(owner):
                o = rng.permutation(P)
                pos = int(np.flatnonzero(o == i)[0])
                A[r_, o[:pos + 1]] = 1
        else:
            A = np.ones((P, P))
            owner = np.arange(P)
        gW1, gb1, gW2, gb2, gv = net.own_grads(x, y, A, owner)
        net.W1 += LR * gW1 / BATCH * conn
        net.b1 += LR * gb1 / BATCH
        net.W2 += LR * gW2 / BATCH
        net.b2 += LR * gb2 / BATCH
        net.v += LR * gv / BATCH
        for nm in ("W1", "W2", "v"):
            w = getattr(net, nm)
            setattr(net, nm, np.sign(w) * np.maximum(np.abs(w) - LR * kappa, 0))
        if econ:
            dead = alive & (wealth < DEATH * W0)
            for i in np.flatnonzero(dead):
                treasury += wealth[i]
                wealth[i] = 0.0
                alive[i] = False
                net.kill(i)
                deaths[i] += 1
    tot = wealth.sum() + treasury + burned
    assert abs(tot - (start + subsidy)) < 1e-6 * max(1.0, abs(tot)), (tot, start + subsidy)
    z, _ = net.fwd(xt, np.ones((1, P)))
    zt = z[0]
    ll = float(-score(zt, yt).mean())
    ns = 1000
    trng = np.random.default_rng(9)
    orders = [trng.permutation(P) for _ in range(200)]
    truth = shap_from_orders(lambda M: net.vals(xt[:ns], yt[:ns], M), orders, P) / ns
    per = hist / max(nh, 1)
    useless = np.zeros(P, bool)
    useless[:N_NOISE_UNITS] = True
    xr, yr, _ = make_data(2000, np.random.default_rng(77 + seed))
    tscale, ll_temp = temp_fit(net.fwd(xr, np.ones((1, P)))[0][0], yr, net.b0, zt, yt)
    return dict(cfg=cfg, ll=ll, tscale=tscale, ll_temp=ll_temp, ll_bayes=float(-score(np.log(pt / (1 - pt)), yt).mean()),
                zt=zt[:2000].round(4).tolist(), pay=per.tolist(), pay_loo=(hist_loo / max(nh, 1)).tolist(),
                truth=truth.tolist(), sp_shap=spearman(per, truth), sp_loo=spearman(hist_loo, truth),
                upstream_share_pay=float(per[:H1].sum() / per.sum()),
                upstream_share_truth=float(truth[:H1].sum() / truth.sum()),
                deaths_useless=int(deaths[useless].sum()), deaths_useful=int(deaths[~useless].sum()),
                deaths_l1=int(deaths[:H1].sum()), deaths_l2=int(deaths[H1:].sum()),
                active_conn=int(net.active().sum()), burned=burned)


def run(cfg):
    t0 = time.time()
    r = run2(cfg) if cfg.get("layers", 1) == 2 else run1(cfg)
    r["secs"] = time.time() - t0
    return r


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "summarize":
        return summarize()
    arms = ("backprop", "backprop_alt", "loo", "msr1", "shap")
    cfgs = []
    if what == "quick":
        cfgs = [dict(task=t, arm=a, cost=c, seed=0, T=300) for t, c in (("synth", 5e-4), ("wdbc", 2e-4)) for a in ("backprop", "shap")]
        cfgs += [dict(task="synth", layers=2, arm="shap", cost=5e-4, seed=0, T=100)]
    else:
        seeds = range(5)
        for s in seeds:
            for arm in arms:
                for c in (0.0, 5e-4, 2e-3):
                    cfgs.append(dict(task="synth", arm=arm, cost=c, seed=s))
                for c in (0.0, 2e-4):
                    cfgs.append(dict(task="wdbc", arm=arm, cost=c, seed=s))
            for arm in ("backprop", "loo", "shap"):
                cfgs.append(dict(task="synth", layers=2, arm=arm, cost=5e-4, seed=s))
        cfgs.sort(key=lambda c: -(c.get("layers", 1) * 10 + (c["arm"] == "shap")))
    t0 = time.time()
    with Pool(6) as pool:
        rows = pool.map(run, cfgs, chunksize=1)
    json.dump(rows, open(os.path.join(HERE, "results", f"mlp_{what}_rows.json"), "w"),
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", what, round(time.time() - t0), "s", len(rows))



# ---------------------------------------------------------------- summary
def summarize(path=None):
    rows = json.load(open(path or os.path.join(HERE, "results", "mlp_all_rows.json")))
    ci = lambda v: [round(x, 5) for x in mean_ci([u for u in v if u == u])] if len(v) else None
    out = {"design": __doc__, "n_rows": len(rows)}
    key = lambda c: (c["task"] if c.get("layers", 1) == 1 else "synth2", c["cost"])
    bp = {(key(r["cfg"]), r["cfg"]["seed"]): r for r in rows if r["cfg"]["arm"] == "backprop"}
    groups = {}
    for r in rows:
        groups.setdefault((key(r["cfg"]), r["cfg"]["arm"]), []).append(r)
    for (k, arm), rs in sorted(groups.items(), key=lambda t: str(t[0])):
        rs = sorted(rs, key=lambda r: r["cfg"]["seed"])
        s = {"seeds": len(rs)}
        for m in ("ll", "ll_temp", "tscale", "ll_lr", "ll_bayes", "sp_zero_vs_mean", "budget_ratio",
                  "deaths_useless", "deaths_useful", "deaths_l1", "deaths_l2", "alive_end", "vabs_useless",
                  "vabs_useful", "truth_useless", "pay_useless", "active_conn", "burned", "gain",
                  "sp_shap", "sp_loo", "upstream_share_pay", "upstream_share_truth"):
            if m in rs[0]:
                s[m] = ci([r[m] for r in rs])
        if "sp" in rs[0]:
            for rule in ("loo", "msr1", "shap"):
                s["sp_" + rule] = ci([r["sp"][rule] for r in rs])
        if "ll_lr" in rs[0]:
            s["ll_minus_lr"] = ci([r["ll"] - r["ll_lr"] for r in rs])
        pairs = [(r, bp[(k, r["cfg"]["seed"])]) for r in rs if (k, r["cfg"]["seed"]) in bp]
        if pairs and arm != "backprop":
            s["ll_minus_backprop"] = ci([r["ll"] - b["ll"] for r, b in pairs])
            s["mean_abs_logit_diff_vs_backprop"] = ci([float(np.mean(np.abs(np.array(r["zt"]) - np.array(b["zt"])))) for r, b in pairs])
            if k[0] != "wdbc":   # b0 = 0 on the balanced synthetic task
                s["mean_abs_logit_diff_vs_backprop_after_scale"] = ci([float(np.mean(np.abs(r["tscale"] * np.array(r["zt"]) - np.array(b["zt"])))) for r, b in pairs])
                s["mean_abs_logit_backprop"] = ci([float(np.mean(np.abs(np.array(b["zt"])))) for r, b in pairs])
        if "dup" in rs[0]:
            for rule in ("shap", "loo", "msr1"):
                o = np.array([r["dup"][rule + "_orig"] for r in rs])
                a = np.array([r["dup"][rule + "_a"] for r in rs])
                b = np.array([r["dup"][rule + "_b"] for r in rs])
                ok = np.abs(o) > 1e-6
                s["dup_" + rule] = {"orig_nats": ci(o), "copyA_over_orig": ci(a[ok] / o[ok]),
                                    "copyB_over_orig": ci(b[ok] / o[ok]), "sum_over_orig": ci((a + b)[ok] / o[ok])}
        if "gain" in rs[0]:
            s["gain_per_100_conn"] = ci([100 * r["gain"] / max(r["active_conn"], 1) for r in rs])
        out[f"{k[0]}|cost={k[1]}|{arm}"] = s
    json.dump(out, open(os.path.join(HERE, "results", "mlp_summary.json"), "w"), indent=1)
    # charts
    lab = {"backprop": "Backprop", "backprop_alt": "Backprop, other init", "loo": "LOO pay + ledger",
           "msr1": "MSR one order + ledger", "shap": "Shapley-16 pay + ledger"}
    for task, cost, cid in (("synth", 0.0005, "mlp_logloss_synth"), ("wdbc", 0.0002, "mlp_logloss_wdbc")):
        bars = []
        for arm in ("backprop", "loo", "msr1", "shap"):
            s = out[f"{task}|cost={cost}|{arm}"]
            bars.append({"label": lab[arm], "value": s["ll"][0], "ci": s["ll"][1], "highlight": arm == "shap"})
            if arm == "shap":
                bars.append({"label": "Shapley-16, output rescaled (diagnostic)", "value": s["ll_temp"][0], "ci": s["ll_temp"][1], "highlight": False})
        s = out[f"{task}|cost={cost}|backprop"]
        bars.append({"label": "Logistic regression", "value": s["ll_lr"][0], "ci": s["ll_lr"][1], "highlight": False})
        if "ll_bayes" in s:
            bars.append({"label": "True probability", "value": s["ll_bayes"][0], "ci": s["ll_bayes"][1], "highlight": False})
        json.dump({"id": cid, "kind": "bar", "title": f"MLP agents ({task}, 32 tanh units): test log loss (lower is better)",
                   "unit": "nats", "source": "trader-economy/results/mlp_summary.json",
                   "note": f"compute cost {cost} per connection per round; 5 seeds, 95% t-intervals", "bars": bars},
                  open(os.path.join(HERE, "charts", cid + ".json"), "w"), indent=1)
    groups_ = ["Backprop net", "LOO-trained net", "Shapley-trained net"]
    series = {}
    for rule, nm in (("loo", "LOO pay"), ("msr1", "MSR one order"), ("shap", "Shapley-16 pay")):
        series[nm] = [out[f"synth|cost=0.0005|{a}"]["sp_" + rule][0] for a in ("backprop", "loo", "shap")]
    json.dump({"id": "mlp_pay_vs_value", "kind": "grouped_bar",
               "title": "Does a unit's pay track its held-out Neuron Shapley? Spearman (synthetic, cost 0.0005)",
               "source": "trader-economy/results/mlp_summary.json", "groups": groups_, "series": series},
              open(os.path.join(HERE, "charts", "mlp_pay_vs_value.json"), "w"), indent=1)
    print("summary written")


if __name__ == "__main__":
    main()
