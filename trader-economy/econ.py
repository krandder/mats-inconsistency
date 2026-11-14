"""Trader economy, toys 1 and 2, regulated and unregulated.

Regulated (primary): each agent reports its belief truthfully (its own online
logistic model, trained on its own log loss) and a stake (fixed fraction of
wealth; no strategic sizing).  An agent whose own predictions did not beat
the 0.5 prior last epoch reports "belief = price" (it abstains: no trade, no
pay).  The regulator's output is a logistic combiner of the active reports,
fitted on the previous epoch.  Each epoch the regulator pays out the LMSR
pot b * v(all) using one arm (credit.py).  Negative credit is charged up to
the agent's stake.

Unregulated (comparison): Kelly bettors (log utility, stake = 0.5 * wealth)
trade every step against an LMSR with liquidity b (core.clear/settle).

Everyone pays compute per step and a fee per raw feature read (to the
feature owners, pooled in the treasury, or burned).  Wealth below 10% of the
starting wealth is bankruptcy; the slot is refilled from the treasury.
Ledger per run: agents + treasury + burned == start + money paid in by the
regulator / LMSR (asserted).
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from core import (D, KIND, OnlineMLP, clear, logloss, make_data, mean_ci,
                  settle, sigmoid, spearman, subset_mi_table)
from credit import Game, arm_payments, shapley_exact, shapley_exact_from

EPOCH = 250
W0 = 10.0
DEATH = 0.1
LAM = 0.5
LR = 0.05
ZCLIP = 6.0
B = 0.5
PRIOR_LL = np.log(2)


def rand_mask(rng, lo=1, hi=3, n=D):
    m = np.zeros(n, bool)
    m[rng.choice(n, rng.integers(lo, hi + 1), replace=False)] = True
    return m


def info_of(masks_feat, table):
    f = np.flatnonzero(np.any(masks_feat, axis=0)) if len(masks_feat) else []
    return table[tuple(int(j) for j in f)]


def ideal_shapley(feat_masks, table):
    """True value: Shapley of each agent when a coalition is worth the
    information (nats per example) in the union of its members' features."""
    P = len(feat_masks)
    fm = np.array(feat_masks)

    def v(m):
        idx = [i for i in range(P) if m >> i & 1]
        if not idx:
            return 0.0
        return table[tuple(int(j) for j in np.flatnonzero(fm[idx].any(0)))]
    return shapley_exact_from(v, P)


# ------------------------------------------------------------------ toy 1 --
def toy1(cfg):
    seed, arm = cfg["seed"], cfg["arm"]
    T, N = cfg.get("T", 20000), cfg.get("N", 10)
    fee, comp, fee_mode = cfg.get("fee", 0.004), cfg.get("compute", 0.004), cfg.get("fee_mode", "owner")
    K = cfg.get("K", 16)
    ref_every = cfg.get("ref_every", 2)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 101)
    table, _ = subset_mi_table()
    x, y, ptrue = make_data(T, rng)

    masks = np.array([rand_mask(rng) for _ in range(N)])
    theta = np.zeros((N, D + 1))
    w = np.full(N, W0)
    alive = np.ones(N, bool)
    born = np.zeros(N, int)
    active = np.zeros(N, bool)            # reports a belief this epoch
    own_ll = np.zeros(N)
    own_n = np.zeros(N)
    treasury, burned, paid_in = 10 * W0, 0.0, 0.0
    start = N * W0 + treasury
    comb = None                            # (player slots, weights, bias)
    Zep = np.zeros((N, EPOCH))
    gain_ep = np.zeros(N)
    p_out, p_lr, p_mlp = np.zeros(T), np.zeros(T), np.zeros(T)
    lr_all = np.zeros(D + 1)
    mlp = OnlineMLP(D, 32, np.random.default_rng(seed + 999))
    recs = []            # per agent-epoch: earnings, ideal shapley, mech shapley, info, kind
    budget = []
    evals = []
    lifelog = []
    shares_hist = []
    wealth_hist = []

    for t in range(T):
        if t % EPOCH == 0:
            ep_start = t
        for i in np.flatnonzero(~alive):
            if treasury >= W0:
                treasury -= W0
                masks[i], theta[i], w[i], alive[i], born[i] = rand_mask(rng), 0.0, W0, True, t
                active[i] = False
                own_ll[i] = own_n[i] = 0.0
        xt, yt = x[t], y[t]
        cost = np.where(alive, comp + fee * masks.sum(1), 0.0)
        pay = np.minimum(cost, w)
        w -= pay
        fpart = np.where(cost > 0, pay * fee * masks.sum(1) / np.maximum(cost, 1e-12), 0.0)
        burned += float((pay - fpart).sum())
        if fee_mode == "owner":
            treasury += float(fpart.sum())
        else:
            burned += float(fpart.sum())
        z = np.clip((theta[:, :D] * masks) @ xt + theta[:, D], -ZCLIP, ZCLIP)
        q = sigmoid(z)
        # output
        if arm == "kelly":
            age = t - born
            stake = np.where(alive, LAM * np.minimum(1.0, age / 50) * w, 0.0)
            qc = np.clip(q, 0.01, 0.99)
            p = float(clear(stake, qc, b=B)) if stake.sum() > 0 else 0.5
            g, L = settle(stake, qc, p, yt, b=B)
            w += g
            paid_in += L
            gain_ep += g
        else:
            if comb is not None:
                sl, beta, b0 = comb
                zz = np.where(alive[sl] & active[sl], z[sl], 0.0)
                p = float(sigmoid(beta @ zz + b0))
            else:
                p = 0.5
        p_out[t] = p
        Zep[:, t - ep_start] = np.where(active, z, 0.0)
        # own learning
        own_ll += alive * -np.where(yt > 0.5, np.log(np.clip(q, 1e-6, 1)), np.log(np.clip(1 - q, 1e-6, 1)))
        own_n += alive
        theta[:, :D] += LR * ((yt - q) * alive)[:, None] * (masks * xt)
        theta[:, D] += LR * (yt - q) * alive
        xa = np.append(xt, 1.0)
        pl = sigmoid(lr_all @ xa)
        p_lr[t] = pl
        lr_all += LR * (yt - pl) * xa
        pm = mlp.predict(xt)
        p_mlp[t] = pm
        mlp.update(pm, yt)
        # bankruptcy
        for i in np.flatnonzero(alive & (w < DEATH * W0)):
            lifelog.append((info_of(masks[i:i + 1], table), t - born[i], True))
            treasury += w[i]
            w[i], alive[i], active[i] = 0.0, False, False
        # ---- epoch settlement ----
        if (t + 1) % EPOCH == 0:
            ep = t // EPOCH
            yep = y[ep_start:t + 1]
            if arm == "kelly":
                players = np.flatnonzero(alive & (born <= ep_start))
            else:
                players = np.flatnonzero(alive & active & (born <= ep_start))
            game = Game(Zep[players], yep) if arm != "kelly" else Game(
                np.clip(np.log(np.clip(1e-9 + 0, 1, 1)) + Zep[players] * 0 + 0, -1, 1), yep)
            if arm == "kelly":
                # kelly agents' own logits (not zeroed by 'active') for the reference game
                zk = np.clip((theta[players][:, :D] * masks[players]) @ x[ep_start:t + 1].T
                             + theta[players][:, D:D + 1], -ZCLIP, ZCLIP)
                game = Game(zk, yep)
                earn = gain_ep[players].copy()
                pot = None
            elif len(players):
                phi, ne = arm_payments(game, arm, arng, K)
                evals.append(ne)
                pot = B * game.full()
                money = B * phi
                earn = np.zeros(len(players))
                for k, i in enumerate(players):
                    if money[k] >= 0:
                        w[i] += money[k]
                        earn[k] = money[k]
                    else:
                        c = min(-money[k], LAM * w[i])
                        w[i] -= c
                        earn[k] = -c
                paid_in += float(earn.sum())
                if abs(pot) > 1e-9:
                    budget.append(float(B * phi.sum() / pot))
            else:
                earn = np.zeros(0)
            if len(players) and t >= T // 2:
                ideal = ideal_shapley([masks[i] for i in players], table) * EPOCH * B
                mech = shapley_exact(game) * B if (ep % ref_every == 0 and len(players) <= 12) else None
                for k, i in enumerate(players):
                    recs.append(dict(ep=ep, earn=float(earn[k]), ideal=float(ideal[k]),
                                     mech=None if mech is None else float(mech[k]),
                                     info=info_of(masks[i:i + 1], table)))
            # next epoch's output combiner and who reports
            if arm != "kelly" and len(players):
                from credit import _fit
                X = np.vstack([Zep[players], np.ones(len(yep))]).T
                th = _fit(X, yep)
                comb = (players, th[:-1], th[-1])
            elif arm != "kelly":
                comb = None
            mean_ll = own_ll / np.maximum(own_n, 1)
            active = alive & (own_n >= 100) & (mean_ll < PRIOR_LL - 0.001)
            own_ll[:] = 0.0
            own_n[:] = 0.0
            gain_ep[:] = 0.0
            tw = w[alive].sum()
            shares = np.where(alive, w, 0.0) / max(tw, 1e-9)
            shares_hist.append(shares.tolist())
            wealth_hist.append(float(tw))

    total = w.sum() + treasury + burned
    assert abs(total - (start + paid_in)) < 1e-6 * max(1.0, abs(total)), (total, start + paid_in)
    for i in np.flatnonzero(alive):
        lifelog.append((info_of(masks[i:i + 1], table), T - born[i], False))
    h = slice(T // 2, T)
    sh = np.array(shares_hist)
    drift = np.abs(np.diff(sh, axis=0)).sum(1) / 2   # fraction of wealth share that moved per epoch
    ne = len(drift)
    er = [r["earn"] for r in recs]
    return dict(
        cfg=cfg,
        ll_out=logloss(p_out[h], y[h]), ll_lr=logloss(p_lr[h], y[h]), ll_mlp=logloss(p_mlp[h], y[h]),
        ll_bayes=logloss(ptrue[h], y[h]),
        sp_ideal=spearman(er, [r["ideal"] for r in recs]),
        sp_mech=spearman([r["earn"] for r in recs if r["mech"] is not None],
                         [r["mech"] for r in recs if r["mech"] is not None]),
        sp_info=spearman(er, [r["info"] for r in recs]),
        budget_ratio=float(np.mean(budget)) if budget else float("nan"),
        evals_per_epoch=float(np.mean(evals)) if evals else float("nan"),
        useless_died=[d for inf, l, d in lifelog if inf < 0.002],
        useless_life=[l for inf, l, d in lifelog if inf < 0.002],
        useful_died=[d for inf, l, d in lifelog if inf >= 0.002],
        useful_life=[l for inf, l, d in lifelog if inf >= 0.002],
        drift_early=float(drift[: ne // 4].mean()), drift_late=float(drift[-ne // 4:].mean()),
        wealth_hist=wealth_hist, alive_end=int(alive.sum()),
        conc_end=float(1 / np.sum(sh[-1] ** 2)) if sh[-1].sum() > 0 else 0.0,
        paid_in_per_step=paid_in / T,
    )


# ------------------------------------------------------------ toy 1 audit --
AUDIT_FEATS = [[0], [4], [1], [5], [6], [7], [2], [3], [8], [9]]
AUDIT_NAMES = ["x0", "x4 (copy of x0)", "x1", "x5 (copy of x1)", "x6 (comp)", "x7 (comp)",
               "x2", "x3", "x8 (noise)", "x9 (noise)"]


def audit(cfg):
    """Fixed designed agents, one per feature: substitutes, complements,
    independents, noise.  Every epoch compute all arms on the same game."""
    seed = cfg["seed"]
    T = cfg.get("T", 10000)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 5)
    table, _ = subset_mi_table()
    x, y, _ = make_data(T, rng)
    N = len(AUDIT_FEATS)
    masks = np.zeros((N, D), bool)
    for i, f in enumerate(AUDIT_FEATS):
        masks[i, f] = True
    theta = np.zeros((N, D + 1))
    Z = np.zeros((N, T))
    for t in range(T):
        z = np.clip((theta[:, :D] * masks) @ x[t] + theta[:, D], -ZCLIP, ZCLIP)
        Z[:, t] = z
        q = sigmoid(z)
        theta[:, :D] += LR * (y[t] - q)[:, None] * (masks * x[t])
        theta[:, D] += LR * (y[t] - q)
    ideal = ideal_shapley(list(masks), table) * EPOCH       # nats per epoch
    out = {a: [] for a in ("loo", "msr1", "msr1_rev", "msrK", "shapley")}
    kerr = {K: [] for K in (1, 2, 4, 8, 16, 32, 64)}
    full = []
    for ep in range(T // EPOCH // 2, T // EPOCH):
        s = slice(ep * EPOCH, (ep + 1) * EPOCH)
        g = Game(Z[:, s], y[s])
        sh = shapley_exact(g)
        out["shapley"].append(sh)
        out["loo"].append(arm_payments(g, "loo", arng)[0])
        out["msr1"].append(arm_payments(g, "msr1", arng)[0])
        from credit import msr_order
        out["msr1_rev"].append(msr_order(g, list(range(N))[::-1]))
        out["msrK"].append(arm_payments(g, "msrK", arng, 16)[0])
        full.append(g.full())
        for K in kerr:
            for _ in range(5):
                est = arm_payments(g, "msrK", arng, K)[0]
                kerr[K].append(float(np.abs(est - sh).sum() / max(np.abs(sh).sum(), 1e-9)))
    res = {a: np.mean(v, axis=0).tolist() for a, v in out.items()}
    res["ideal"] = ideal.tolist()
    res["full"] = float(np.mean(full))
    res["loo_budget_ratio"] = float(np.mean([np.sum(v) / f for v, f in zip(out["loo"], full)]))
    res["kerr"] = {K: float(np.mean(v)) for K, v in kerr.items()}
    res["sp_vs_ideal"] = {a: spearman(res[a], ideal) for a in out}
    return res


# ------------------------------------------------------------------ toy 2 --
def toy2(cfg):
    """Layer 1: N1 agents on raw features publish logits.  Layer 2: N2 readers,
    each on 3-4 layer-1 outputs, publish logits; the output combines readers.
    Readers pass rho of their positive earnings to their sources, split by the
    arm run on the reader's own game (its score from its sources)."""
    seed, arm = cfg["seed"], cfg["arm"]
    T, N1, N2 = cfg.get("T", 20000), cfg.get("N1", 8), cfg.get("N2", 4)
    fee, comp, rho = cfg.get("fee", 0.004), cfg.get("compute", 0.004), cfg.get("rho", 0.5)
    K = cfg.get("K", 8)
    rng = np.random.default_rng(seed)
    arng = np.random.default_rng(seed + 101)
    table, _ = subset_mi_table()
    x, y, ptrue = make_data(T, rng)
    from credit import _fit

    m1 = np.array([rand_mask(rng) for _ in range(N1)])
    th1 = np.zeros((N1, D + 1))
    w1 = np.full(N1, W0)
    a1 = np.ones(N1, bool)
    b1 = np.zeros(N1, int)
    act1 = np.zeros(N1, bool)
    ll1 = np.zeros(N1)
    n1 = np.zeros(N1)
    m2 = np.array([rand_mask(rng, 3, 4, N1) for _ in range(N2)])
    th2 = np.zeros((N2, N1 + 1))
    w2 = np.full(N2, W0)
    a2 = np.ones(N2, bool)
    b2 = np.zeros(N2, int)
    act2 = np.zeros(N2, bool)
    ll2 = np.zeros(N2)
    n2 = np.zeros(N2)
    cum2 = np.zeros(N2)
    hwm2 = np.zeros(N2)
    treasury, burned, paid_in = 10 * W0, 0.0, 0.0
    start = (N1 + N2) * W0 + treasury
    comb = None
    Z1 = np.zeros((N1, EPOCH))
    Z2 = np.zeros((N2, EPOCH))
    gain2 = np.zeros(N2)
    earn1 = np.zeros(N1)
    p_out = np.zeros(T)
    recs, budget, lifelog, shares_hist = [], [], [], []
    unread_life = []

    def nll(q, yy):
        return -np.where(yy > 0.5, np.log(np.clip(q, 1e-6, 1)), np.log(np.clip(1 - q, 1e-6, 1)))

    for t in range(T):
        if t % EPOCH == 0:
            ep_start = t
        for i in np.flatnonzero(~a1):
            if treasury >= W0:
                treasury -= W0
                m1[i], th1[i], w1[i], a1[i], b1[i], act1[i], ll1[i], n1[i] = rand_mask(rng), 0.0, W0, True, t, False, 0, 0
        for j in np.flatnonzero(~a2):
            if treasury >= W0:
                treasury -= W0
                m2[j], th2[j], w2[j], a2[j], b2[j], act2[j], ll2[j], n2[j] = rand_mask(rng, 3, 4, N1), 0.0, W0, True, t, False, 0, 0
                cum2[j] = hwm2[j] = 0.0
        xt, yt = x[t], y[t]
        c1 = np.where(a1, comp + fee * m1.sum(1), 0.0)
        p1 = np.minimum(c1, w1)
        w1 -= p1
        fp = np.where(c1 > 0, p1 * fee * m1.sum(1) / np.maximum(c1, 1e-12), 0.0)
        treasury += float(fp.sum())
        burned += float((p1 - fp).sum())
        p2 = np.where(a2, np.minimum(comp, w2), 0.0)
        w2 -= p2
        burned += float(p2.sum())
        z1 = np.clip((th1[:, :D] * m1) @ xt + th1[:, D], -ZCLIP, ZCLIP)
        q1 = sigmoid(z1)
        pub1 = np.where(a1 & act1, z1, 0.0)            # abstainers publish the prior
        inp = m2 * pub1[None, :]
        z2 = np.clip((th2[:, :N1] * inp).sum(1) + th2[:, N1], -ZCLIP, ZCLIP)
        q2 = sigmoid(z2)
        if arm == "kelly":
            age = t - b2
            stake = np.where(a2, LAM * np.minimum(1.0, age / 50) * w2, 0.0)
            qc = np.clip(q2, 0.01, 0.99)
            p = float(clear(stake, qc, b=B)) if stake.sum() > 0 else 0.5
            g, L = settle(stake, qc, p, yt, b=B)
            w2 += g
            paid_in += L
            gain2 += g
        else:
            if comb is not None:
                sl, beta, bb = comb
                zz = np.where(a2[sl] & act2[sl], z2[sl], 0.0)
                p = float(sigmoid(beta @ zz + bb))
            else:
                p = 0.5
        p_out[t] = p
        Z1[:, t - ep_start] = pub1
        Z2[:, t - ep_start] = np.where(act2, z2, 0.0)
        ll1 += a1 * nll(q1, yt)
        n1 += a1
        ll2 += a2 * nll(q2, yt)
        n2 += a2
        th1[:, :D] += LR * ((yt - q1) * a1)[:, None] * (m1 * xt)
        th1[:, D] += LR * (yt - q1) * a1
        th2[:, :N1] += 0.02 * ((yt - q2) * a2)[:, None] * inp
        th2[:, N1] += 0.02 * (yt - q2) * a2
        for i in np.flatnonzero(a1 & (w1 < DEATH * W0)):
            lifelog.append((info_of(m1[i:i + 1], table), t - b1[i], True, bool(m2[a2][:, i].any())))
            treasury += w1[i]
            w1[i], a1[i], act1[i] = 0.0, False, False
        for j in np.flatnonzero(a2 & (w2 < DEATH * W0)):
            treasury += w2[j]
            w2[j], a2[j], act2[j] = 0.0, False, False
        if (t + 1) % EPOCH == 0:
            ep = t // EPOCH
            yep = y[ep_start:t + 1]
            earn1[:] = 0.0
            if arm == "kelly":
                readers = np.flatnonzero(a2 & (born_ok := (b2 <= ep_start)))
                money2 = np.zeros(N2)
                for j in readers:
                    cum2[j] += gain2[j]
                    money2[j] = max(0.0, cum2[j] - hwm2[j])
                    hwm2[j] = max(hwm2[j], cum2[j])
            else:
                readers = np.flatnonzero(a2 & act2 & (b2 <= ep_start))
                money2 = np.zeros(N2)
                if len(readers):
                    g = Game(Z2[readers], yep)
                    phi, _ = arm_payments(g, "msrK" if arm == "royalty" else arm, arng, K)
                    pot = B * g.full()
                    if abs(pot) > 1e-9:
                        budget.append(("out", float(B * phi.sum() / pot)))
                    for k, j in enumerate(readers):
                        mny = B * phi[k]
                        if mny >= 0:
                            w2[j] += mny
                            money2[j] = mny
                        else:
                            c = min(-mny, LAM * w2[j])
                            w2[j] -= c
                            money2[j] = -c
                        paid_in += money2[j] if mny >= 0 else -c
            # readers pay their sources
            for j in readers:
                if money2[j] <= 0:
                    continue
                src = np.flatnonzero(m2[j] & a1 & (b1 <= ep_start))
                if len(src) == 0:
                    continue
                pot_j = rho * money2[j]
                if arm in ("royalty", "kelly"):
                    shares = np.full(len(src), 1.0 / len(src))
                else:
                    gj = Game(Z1[src], yep)
                    cj, _ = arm_payments(gj, arm, arng, K)
                    vj = gj.full()
                    shares = cj / vj if vj > 1e-9 else np.zeros(len(src))
                    budget.append(("src", float(shares.sum())))
                amounts = pot_j * shares
                paid_by_reader = min(pot_j, w2[j])
                w2[j] -= paid_by_reader
                treasury += paid_by_reader
                for k, i in enumerate(src):
                    a = amounts[k]
                    if a >= 0:
                        a = min(a, treasury)
                        treasury -= a
                        w1[i] += a
                        earn1[i] += a
                    else:
                        c = min(-a, LAM * w1[i])
                        w1[i] -= c
                        treasury += c
                        earn1[i] -= c
            # references for layer-1 agents
            players = np.flatnonzero(a1 & (b1 <= ep_start))
            if t >= T // 2 and len(players):
                read = m2[a2].any(0)
                fm = [m1[i] if read[i] else np.zeros(D, bool) for i in players]
                ideal = ideal_shapley(fm, table) * EPOCH * B
                sysv = None
                if ep % 2 == 0 and len(players) <= 10 and arm != "kelly":
                    rd = np.flatnonzero(a2 & act2)
                    Zfull = np.clip((th1[:, :D] * m1) @ x[ep_start:t + 1].T + th1[:, D:D + 1], -ZCLIP, ZCLIP)
                    Zfull = Zfull * (a1 & act1)[:, None]

                    def vsys(mask):
                        keep = np.zeros(N1, bool)
                        for k, i in enumerate(players):
                            if mask >> k & 1:
                                keep[i] = True
                        zin = Zfull * keep[:, None]
                        zr = np.clip(np.einsum("jk,kn->jn", th2[rd][:, :N1] * m2[rd], zin) + th2[rd][:, N1:N1 + 1],
                                     -ZCLIP, ZCLIP)
                        return Game(zr, yep).full() if len(rd) else 0.0
                    sysv = shapley_exact_from(vsys, len(players)) * B
                for k, i in enumerate(players):
                    recs.append(dict(ep=ep, earn=float(earn1[i]), ideal=float(ideal[k]),
                                     sys=None if sysv is None else float(sysv[k]),
                                     info=info_of(m1[i:i + 1], table), read=bool(read[i])))
            if arm != "kelly":
                rs = np.flatnonzero(a2 & act2 & (b2 <= ep_start))
                if len(rs):
                    th = _fit(np.vstack([Z2[rs], np.ones(len(yep))]).T, yep)
                    comb = (rs, th[:-1], th[-1])
                else:
                    comb = None
            act1 = a1 & (n1 >= 100) & (ll1 / np.maximum(n1, 1) < PRIOR_LL - 0.001)
            act2 = a2 & (n2 >= 100) & (ll2 / np.maximum(n2, 1) < PRIOR_LL - 0.001)
            ll1[:] = n1[:] = ll2[:] = n2[:] = 0.0
            gain2[:] = 0.0
            allw = np.concatenate([np.where(a1, w1, 0), np.where(a2, w2, 0)])
            shares_hist.append((allw / max(allw.sum(), 1e-9)).tolist())

    total = w1.sum() + w2.sum() + treasury + burned
    assert abs(total - (start + paid_in)) < 1e-6 * max(1.0, abs(total)), (total, start + paid_in)
    for i in np.flatnonzero(a1):
        lifelog.append((info_of(m1[i:i + 1], table), T - b1[i], False, bool(m2[a2][:, i].any())))
    h = slice(T // 2, T)
    sh = np.array(shares_hist)
    drift = np.abs(np.diff(sh, axis=0)).sum(1) / 2
    ne = len(drift)
    er = [r["earn"] for r in recs]
    rs_ = [r for r in recs if r["sys"] is not None]
    bo = [b for k, b in budget if k == "out"]
    bs = [b for k, b in budget if k == "src"]
    return dict(
        cfg=cfg,
        ll_out=logloss(p_out[h], y[h]), ll_bayes=logloss(ptrue[h], y[h]),
        sp_ideal=spearman(er, [r["ideal"] for r in recs]),
        sp_sys=spearman([r["earn"] for r in rs_], [r["sys"] for r in rs_]),
        earn_useless=float(np.mean([r["earn"] for r in recs if r["info"] < 0.002])) if any(r["info"] < 0.002 for r in recs) else float("nan"),
        earn_useful=float(np.mean([r["earn"] for r in recs if r["info"] >= 0.002])) if recs else float("nan"),
        budget_out=float(np.mean(bo)) if bo else float("nan"),
        budget_src=float(np.mean(bs)) if bs else float("nan"),
        useless_died=[d for inf, l, d, rd in lifelog if inf < 0.002],
        useless_life=[l for inf, l, d, rd in lifelog if inf < 0.002],
        useful_died=[d for inf, l, d, rd in lifelog if inf >= 0.002],
        useful_life=[l for inf, l, d, rd in lifelog if inf >= 0.002],
        drift_early=float(drift[: ne // 4].mean()), drift_late=float(drift[-ne // 4:].mean()),
        l1_alive_end=int(a1.sum()), l2_alive_end=int(a2.sum()), paid_in_per_step=paid_in / T,
    )


# ------------------------------------------------------------------ driver --
def _mc(rows, key):
    v = [r[key] for r in rows if r[key] == r[key]]
    return mean_ci(v) if v else (float("nan"), 0.0)


def summarize(rows, keys):
    out = {k: _mc(rows, k) for k in keys}
    ud = [d for r in rows for d in r["useless_died"]]
    fd = [d for r in rows for d in r["useful_died"]]
    out["useless_died_frac"] = float(np.mean(ud)) if ud else None
    out["useless_n"] = len(ud)
    out["useful_died_frac"] = float(np.mean(fd)) if fd else None
    out["useful_n"] = len(fd)
    ul = [l for r in rows for l in r["useless_life"]]
    fl = [l for r in rows for l in r["useful_life"]]
    out["useless_median_life"] = float(np.median(ul)) if ul else None
    out["useful_median_life"] = float(np.median(fl)) if fl else None
    return out


def _run(job):
    kind, cfg = job
    t0 = time.time()
    r = {"toy1": toy1, "audit": audit, "toy2": toy2}[kind](cfg)
    r["secs"] = time.time() - t0
    return kind, r


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = [0, 1, 2, 3, 4]
    subset_mi_table()
    jobs = []
    if what in ("all", "audit"):
        jobs += [("audit", dict(seed=s)) for s in seeds]
    if what in ("all", "toy1"):
        for arm in ("kelly", "loo", "msr1", "msrK", "shapley"):
            jobs += [("toy1", dict(seed=s, arm=arm)) for s in seeds]
        jobs += [("toy1", dict(seed=s, arm="msrK", fee_mode="burn")) for s in seeds]
    if what in ("all", "sweep"):
        for fee in (0.0, 0.004, 0.012, 0.03):
            for comp in (0.0, 0.004, 0.012, 0.03):
                for arm in ("msrK", "kelly"):
                    jobs += [("toy1", dict(seed=s, arm=arm, fee=fee, compute=comp, ref_every=10 ** 9)) for s in seeds[:3]]
    if what in ("all", "toy2"):
        for arm in ("kelly", "royalty", "loo", "msr1", "msrK", "shapley"):
            jobs += [("toy2", dict(seed=s, arm=arm)) for s in seeds]
    if what == "quick":
        jobs = [("audit", dict(seed=0, T=3000)), ("toy1", dict(seed=0, arm="msrK", T=3000)),
                ("toy1", dict(seed=0, arm="kelly", T=3000)), ("toy2", dict(seed=0, arm="msrK", T=3000)),
                ("toy2", dict(seed=0, arm="kelly", T=3000))]
    t0 = time.time()
    with Pool(15) as pool:
        out = pool.map(_run, jobs, chunksize=1)
    rows = {}
    for kind, r in out:
        rows.setdefault(kind, []).append(r)
    json.dump(rows, open(f"results/econ_{what}_rows.json", "w"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("done", what, round(time.time() - t0), "s", {k: len(v) for k, v in rows.items()})
