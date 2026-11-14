"""Inputs vs computation: how value flowing downwards splits between the
information sources (inputs) and the computing agents (hidden units).

One game: players are inputs (off = replaced by its baseline 0, the mean after
standardising) and hidden units (off = output zeroed).  v(S) = held-out
sum_n log p_S(y_n) - log p_0(y_n), p_0 = the output with every player off.

Part 1  toy games, logit z = a*x * A_in(alpha) * A_proc(g):
        (a) one input, k duplicate processors;  (b) one processor, k duplicate
        inputs;  (c) one of each.  Aggregators: "any" = min(1, sum of gates)
        (any one suffices; an OR realisable by a ReLU 1 - relu(1 - sum)),
        "add" = sum/k (each duplicate carries 1/k: processor out-weight a/k, or
        processor in-weight a/k on each copy).  Exact Shapley + IG/Aumann-Shapley
        (joint straight path, all gates = t, midpoint rule m = 20 and 50,
        gradients by central differences).
Part 2  12-feature synthetic task (core.make_data), relu net 12 -> 8 -> 4 -> 1
        trained by backprop; layer-1 units 6,7 can only read the 4 noise
        features (designed useless units).  Pays:
          flat Shapley over 24 players (12 inputs + 8 + 4 units), 150 random
            orders with antithetic reversal;
          flat IG (all 24 gates on the joint straight path, m = 32);
          nested Owen ("downward flow"): the net is unrolled into a tree, so an
            input read by several units is a separate player per path
            (input d -> unit i -> unit j).  Levels: output -> unions U_j =
            {self_j} + sub-unions U_ij, U_ij = {self_ij} + paths (d,i,j).
            Value by nested random orders (unions ordered, then members inside
            each union, recursively) = Winter's (1989) levels-structure value
            of the unrolled tree game; with one level it is Owen's (1977) value.
            Pays are summed back: input d = its paths, unit i = its self_ij,
            unit j = self_j.  A variant: textbook Owen needs a partition.
        Compared with information value of each input (Shapley of the refit
        logistic information table, and the drop-feature refit value).
Part 3  training on the payoffs, same net, 5 seeds: arms
          backprop  full gradient (ledger paid by flat Shapley, no deaths)
          shap      each unit follows grad of v(pre + u), pre = random
                    predecessor set over all 24 players (Shapley dropout)
          owen      each unit follows the gradient of its nested-Owen pay in
                    one sampled nested order (sum over its self players of
                    grad v(after) - grad v(before)); inputs are paid their
                    path marginals.
        W2 edges are owned by both ends: they move by the mean of the two
        owners' gradients.  Conserved ledger (agents + inputs + treasury +
        burned = start + subsidy, asserted), cost per active connection,
        bankruptcy (weights zeroed) and respawn from the treasury.
Outputs results/split_summary.json, results/split_rows.json, charts/split_*.json.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

import core
from core import mean_ci, spearman, make_data, KIND
from credit import shapley_exact_from

HERE = os.path.dirname(os.path.abspath(__file__))
D, H1, H2 = 12, 8, 4
NOISE_UNITS = [6, 7]
NP = D + H1 + H2
GX = D * H1 * H2
NG = GX + H1 * H2 + H2
M1 = np.ones((H1, D))
M1[NOISE_UNITS] = 0
M1[np.ix_(NOISE_UNITS, [8, 9, 10, 11])] = 1
BATCH, LR = 250, 0.2
B, W0, DEATH, TREAS0, ACTIVE = 0.01, 1.0, 0.1, 5.0, 1e-3
PAYEVERY = 10


def sig(z):
    return 1 / (1 + np.exp(-z))


def logsc(z, y):
    """per-example log score of logits z (..., n) for labels y."""
    return -np.logaddexp(0, np.where(y > 0.5, -z, z))


# ============================================================== Part 1
def toy_v(alpha, g, x, y, a, agg_in, agg_proc):
    def A(v, agg):
        v = np.asarray(v, float)
        return min(1.0, v.sum()) if agg == "any" else v.sum() / len(v)
    z = a * x * A(alpha, agg_in) * A(g, agg_proc)
    return float(logsc(z, y).sum() - logsc(np.zeros_like(z), y).sum())


def part1_case(case, k, variant, seed, a=2.0, n=20000):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = (rng.random(n) < sig(a * x)).astype(float)
    nin, npr = {"a": (1, k), "b": (k, 1), "c": (1, 1)}[case]
    agg_in = variant if case == "b" else "add"
    agg_proc = variant if case == "a" else "add"
    P = nin + npr

    def vg(gv):
        return toy_v(gv[:nin], gv[nin:], x, y, a, agg_in, agg_proc)

    def vmask(m):
        return vg(np.array([m >> i & 1 for i in range(P)], float))
    phi = shapley_exact_from(vmask, P)
    full = vg(np.ones(P))
    out = dict(case=case, k=k, variant=variant, seed=seed, full=full,
               shap_input=float(phi[:nin].sum() / full), shap_proc=float(phi[nin:].sum() / full))
    for m in (20, 50):
        ig = np.zeros(P)
        eps = 1e-4
        for t in (np.arange(m) + 0.5) / m:
            for i in range(P):
                gp = np.full(P, t)
                gm = gp.copy()
                gp[i] += eps
                gm[i] -= eps
                ig[i] += (vg(gp) - vg(gm)) / (2 * eps) / m
        out[f"ig{m}_input"] = float(ig[:nin].sum() / full)
        out[f"ig{m}_proc"] = float(ig[nin:].sum() / full)
        out[f"ig{m}_completeness"] = float(ig.sum() / full)
    return out


# ============================================================== network
def init_params(rng):
    return dict(W1=rng.standard_normal((H1, D)) * np.sqrt(2 / D) * M1, b1=np.zeros(H1),
                W2=rng.standard_normal((H2, H1)) * np.sqrt(2 / H1), b2=np.zeros(H2),
                w3=rng.standard_normal(H2) * 0.3, b3=0.0)


def flat_values(x, y, P, gin, gu1, gu2):
    """v for flat masks: gin (M,D), gu1 (M,H1), gu2 (M,H2).  Returns (M,)."""
    Wm = gin[:, :, None] * P["W1"].T[None]                  # (M,D,H1)
    a1 = np.matmul(x[None], Wm) + P["b1"]                   # (M,n,H1)
    h1 = np.maximum(a1, 0) * gu1[:, None]
    h2 = np.maximum(h1 @ P["W2"].T + P["b2"], 0) * gu2[:, None]
    z = h2 @ P["w3"] + P["b3"]
    return logsc(z, y).sum(1) - logsc(np.full(len(y), P["b3"]), y).sum()


def unroll_gates(G):
    """flat gate vectors (M,NG) -> gx (M,D,H1,H2), g1 (M,H1,H2), g2 (M,H2)."""
    M = len(G)
    return (G[:, :GX].reshape(M, D, H1, H2), G[:, GX:GX + H1 * H2].reshape(M, H1, H2), G[:, GX + H1 * H2:])


def flat_to_gates(F):
    M = len(F)
    gx = np.broadcast_to(F[:, :D, None, None], (M, D, H1, H2)).reshape(M, -1)
    g1 = np.broadcast_to(F[:, D:D + H1, None], (M, H1, H2)).reshape(M, -1)
    return np.concatenate([gx, g1, F[:, D + H1:]], 1)


def unrolled(x, y, P, G, grad=False, chunk=64):
    """Tree-unrolled net: layer-1 unit i is copied once per reader j; each path
    (d,i,j), copy (i,j) and unit j has its own gate.  Returns v (M,), and if
    grad: per-mask parameter gradients and gate gradients."""
    outs = []
    for s in range(0, len(G), chunk):
        outs.append(_unrolled(x, y, P, G[s:s + chunk], grad))
    if not grad:
        return np.concatenate(outs)
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _unrolled(x, y, P, G, grad):
    gx, g1, g2 = unroll_gates(G)
    M, n = len(G), len(y)
    a1 = np.empty((M, n, H1, H2))
    for i in range(H1):
        xi = x * P["W1"][i]                                  # (n,D)
        a1[:, :, i, :] = (xi @ gx[:, :, i, :].transpose(1, 0, 2).reshape(D, M * H2)).reshape(n, M, H2).transpose(1, 0, 2)
    a1 += P["b1"][None, None, :, None]
    h1r = np.maximum(a1, 0)
    h1 = h1r * g1[:, None]
    a2 = np.einsum("mnij,ji->mnj", h1, P["W2"]) + P["b2"]
    h2r = np.maximum(a2, 0)
    h2 = h2r * g2[:, None]
    z = h2 @ P["w3"] + P["b3"]
    v = logsc(z, y).sum(1) - logsc(np.full(n, P["b3"]), y).sum()
    if not grad:
        return v
    dz = y - sig(z)
    dw3 = np.einsum("mn,mnj->mj", dz, h2)
    dh2 = dz[..., None] * P["w3"]
    dg2 = (dh2 * h2r).sum(1)
    da2 = dh2 * g2[:, None] * (a2 > 0)
    dW2 = np.einsum("mnj,mnij->mji", da2, h1)
    db2 = da2.sum(1)
    dh1 = da2[:, :, None, :] * P["W2"].T[None, None]
    dg1 = (dh1 * h1r).sum(1)
    da1 = dh1 * g1[:, None] * (a1 > 0)
    t = np.tensordot(da1, x, axes=([1], [0]))               # (M,H1,H2,D)
    t = t.transpose(0, 3, 1, 2)                              # (M,D,H1,H2)
    dgx = t * P["W1"].T[None, :, :, None]
    dW1 = (t * gx).sum(3).transpose(0, 2, 1)
    db1 = da1.sum((1, 3))
    return dict(v=v, W1=dW1, b1=db1, W2=dW2, b2=db2, w3=dw3,
                gates=np.concatenate([dgx.reshape(M, -1), dg1.reshape(M, -1), dg2], 1))


# ------------------------------------------------------------ nested Owen
def path_idx(d, i, j):
    return (d * H1 + i) * H2 + j


def self1_idx(i, j):
    return GX + i * H2 + j


def self2_idx(j):
    return GX + H1 * H2 + j


def nested_order(rng):
    order = []
    for j in rng.permutation(H2):
        mem = [("s", j)] + [("u", i) for i in range(H1)]
        for k in rng.permutation(len(mem)):
            kind, i = mem[k]
            if kind == "s":
                order.append(self2_idx(j))
            else:
                sub = [self1_idx(i, j)] + [path_idx(d, i, j) for d in range(D)]
                order.extend(sub[q] for q in rng.permutation(len(sub)))
    return np.array(order)


def prefix_masks(order):
    pos = np.empty(len(order), int)
    pos[order] = np.arange(len(order))
    return (pos[None, :] < np.arange(len(order) + 1)[:, None]).astype(float), pos


def owen_pay_from(order, v):
    """marginal per gate, then summed to inputs / L1 units / L2 units."""
    marg = np.zeros(NG)
    marg[order] = np.diff(v)
    px = marg[:GX].reshape(D, H1, H2)
    s1 = marg[GX:GX + H1 * H2].reshape(H1, H2)
    s2 = marg[GX + H1 * H2:]
    return np.concatenate([px.sum((1, 2)), s1.sum(1), s2]), px, s1


def flat_perm_pay(x, y, P, K, rng, antithetic=True):
    phi = np.zeros(NP)
    orders = []
    for _ in range(K // 2 if antithetic else K):
        o = rng.permutation(NP)
        orders.append(o)
        if antithetic:
            orders.append(o[::-1])
    for o in orders:
        pos = np.empty(NP, int)
        pos[o] = np.arange(NP)
        F = (pos[None, :] < np.arange(NP + 1)[:, None]).astype(float)
        v = flat_values(x, y, P, F[:, :D], F[:, D:D + H1], F[:, D + H1:])
        phi[o] += np.diff(v)
    return phi / len(orders)


# ============================================================== training
def unit_grads(g, rows_i, rows_j, sign_i=None, sign_j=None):
    """Assemble the own gradient of every unit from per-mask gradients g.
    rows_i[i]: list of mask rows (with weights) whose grads unit i follows."""
    out = dict(W1=np.zeros((H1, D)), b1=np.zeros(H1), W2a=np.zeros((H2, H1)), W2b=np.zeros((H2, H1)),
               b2=np.zeros(H2), w3=np.zeros(H2))
    for i, rows in enumerate(rows_i):
        for r, w in rows:
            out["W1"][i] += w * g["W1"][r, i]
            out["b1"][i] += w * g["b1"][r, i]
            out["W2a"][:, i] += w * g["W2"][r, :, i]
    for j, rows in enumerate(rows_j):
        for r, w in rows:
            out["W2b"][j] += w * g["W2"][r, j]
            out["b2"][j] += w * g["b2"][r, j]
            out["w3"][j] += w * g["w3"][r, j]
    return dict(W1=out["W1"], b1=out["b1"], W2=0.5 * (out["W2a"] + out["W2b"]), b2=out["b2"], w3=out["w3"])


def train(cfg):
    arm, cost, seed, T = cfg["arm"], cfg["cost"], cfg["seed"], cfg["T"]
    ledger = cfg.get("ledger", True)
    lr = cfg.get("lr", LR)
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(5000 + seed)
    P = init_params(np.random.default_rng(100 + seed))
    ybar = 0.5
    P["b3"] = 0.0
    econ = arm in ("shap", "owen")
    wealth = np.full(H1 + H2, W0)
    inputs = np.zeros(D)
    alive = np.ones(H1 + H2, bool)
    treasury, burned, subsidy = TREAS0, 0.0, 0.0
    start = wealth.sum() + treasury
    kappa = cost / B / BATCH
    pay_hist = np.zeros(NP)
    cost_hist = np.zeros(H1 + H2)
    deaths = np.zeros(H1 + H2, int)
    Q = 2
    for t in range(T):
        x, y, _ = make_data(BATCH, drng)
        if econ:
            for u in np.flatnonzero(~alive):
                if treasury >= W0:
                    treasury -= W0
                    wealth[u] = W0
                    alive[u] = True
                    respawn(P, u, rng)
        # ---- pay on the fresh batch, before learning from it
        if ledger:
            if arm == "owen":
                # nested-Owen pay is 420 forward passes: compute it every
                # PAYEVERY rounds and scale by PAYEVERY (subsidy too)
                if t % PAYEVERY == 0:
                    order = nested_order(rng)
                    Mk, _ = prefix_masks(order)
                    v = unrolled(x, y, P, Mk, chunk=16)
                    phi, _, _ = owen_pay_from(order, v)
                    phi, full = phi * PAYEVERY, v[-1] * PAYEVERY
                else:
                    phi, full = np.zeros(NP), 0.0
            else:
                phi = flat_perm_pay(x, y, P, 4, rng)
                full = flat_values(x, y, P, np.ones((1, D)), np.ones((1, H1)), np.ones((1, H2)))[0]
            subsidy += B * full
            treasury += B * (full - phi.sum())
            inputs += B * phi[:D]
            up = B * phi[D:]
            wealth += up * alive
            treasury += (up * ~alive).sum()
            nconn = np.concatenate([(np.abs(P["W1"]) > ACTIVE).sum(1) + 0.5 * (np.abs(P["W2"]) > ACTIVE).sum(0),
                                    0.5 * (np.abs(P["W2"]) > ACTIVE).sum(1) + (np.abs(P["w3"]) > ACTIVE)])
            cc = np.minimum(cost * nconn, np.maximum(wealth, 0)) * alive
            wealth -= cc
            burned += cc.sum()
            if t >= T // 2:
                pay_hist += B * phi
                cost_hist += cc
        # ---- learning
        if arm == "backprop":
            g = unrolled(x, y, P, np.ones((1, NG)), grad=True)
            gr = unit_grads(g, [[(0, 1.0)]] * H1, [[(0, 1.0)]] * H2)
            gb3 = g_b3(x, y, P, g)
        elif arm == "shap":
            F = np.zeros(((H1 + H2) * Q, NP))
            rows_i, rows_j = [[] for _ in range(H1)], [[] for _ in range(H2)]
            r = 0
            for u in range(H1 + H2):
                for _ in range(Q):
                    o = rng.permutation(NP)
                    pu = int(np.flatnonzero(o == D + u)[0])
                    F[r, o[:pu + 1]] = 1
                    (rows_i[u] if u < H1 else rows_j[u - H1]).append((r, 1.0 / Q))
                    r += 1
            g = unrolled(x, y, P, flat_to_gates(F), grad=True)
            gr = unit_grads(g, rows_i, rows_j)
            gb3 = g_b3(x, y, P)
        else:
            order = nested_order(rng)
            Mk, pos = prefix_masks(order)
            rows_i, rows_j = [[] for _ in range(H1)], [[] for _ in range(H2)]
            sel = []
            for i in range(H1):
                for j in range(H2):
                    p = pos[self1_idx(i, j)]
                    rows_i[i] += [(len(sel), 1.0), (len(sel) + 1, -1.0)]
                    sel += [p + 1, p]
            for j in range(H2):
                p = pos[self2_idx(j)]
                rows_j[j] += [(len(sel), 1.0), (len(sel) + 1, -1.0)]
                sel += [p + 1, p]
            g = unrolled(x, y, P, Mk[sel], grad=True, chunk=34)
            gr = unit_grads(g, rows_i, rows_j)
            gb3 = g_b3(x, y, P)
        live1 = alive[:H1].astype(float) if econ else np.ones(H1)
        live2 = alive[H1:].astype(float) if econ else np.ones(H2)
        P["W1"] += lr * gr["W1"] / BATCH * M1 * live1[:, None]
        P["b1"] += lr * gr["b1"] / BATCH * live1
        P["W2"] += lr * gr["W2"] / BATCH * live1[None] * live2[:, None]
        P["b2"] += lr * gr["b2"] / BATCH * live2
        P["w3"] += lr * gr["w3"] / BATCH * live2
        if cfg.get("train_b3", False):      # market-maker prior fixed at the base-rate logit (0)
            P["b3"] += lr * gb3 / BATCH
        if kappa > 0:
            for k in ("W1", "W2", "w3"):
                P[k] = np.sign(P[k]) * np.maximum(np.abs(P[k]) - lr * kappa, 0)
        if econ:
            dead = alive & (wealth < DEATH * W0)
            for u in np.flatnonzero(dead):
                treasury += wealth[u]
                wealth[u] = 0.0
                alive[u] = False
                kill(P, u)
                deaths[u] += 1
    if ledger:
        tot = wealth.sum() + inputs.sum() + treasury + burned
        assert abs(tot - (start + subsidy)) < 1e-6 * max(1.0, abs(tot)), (tot, start + subsidy)
    xt, yt, pt = make_data(20000, np.random.default_rng(4242))
    z = flat_logits(xt, P)
    ll = float(-logsc(z, yt).mean())
    ll_bayes = float(-logsc(np.log(pt / (1 - pt)), yt).mean())
    half = T - T // 2
    return dict(cfg=cfg, P={k: np.asarray(v).tolist() for k, v in P.items()}, ll=ll, ll_bayes=ll_bayes,
                pay=(pay_hist / half).tolist(), cost=(cost_hist / half).tolist(),
                profit=((pay_hist[D:] - cost_hist) / half).tolist(), deaths=deaths.tolist(),
                alive_end=alive.tolist(), wealth_end=wealth.tolist(), inputs_end=inputs.tolist(),
                treasury_end=treasury, subsidy=subsidy, burned=burned)


def g_b3(x, y, P, g=None):
    return float((y - sig(flat_logits(x, P))).sum())


def flat_logits(x, P):
    h1 = np.maximum(x @ P["W1"].T + P["b1"], 0)
    h2 = np.maximum(h1 @ P["W2"].T + P["b2"], 0)
    return h2 @ P["w3"] + P["b3"]


def kill(P, u):
    if u < H1:
        P["W1"][u] = 0
        P["b1"][u] = 0
        P["W2"][:, u] = 0
    else:
        j = u - H1
        P["W2"][j] = 0
        P["b2"][j] = 0
        P["w3"][j] = 0


def respawn(P, u, rng):
    if u < H1:
        P["W1"][u] = rng.standard_normal(D) * np.sqrt(2 / D) * M1[u]
        P["b1"][u] = 0.0
    else:
        P["W2"][u - H1] = rng.standard_normal(H1) * np.sqrt(2 / H1)
        P["b2"][u - H1] = 0.0


# ============================================================== Part 2
def info_values():
    table, info = core.subset_mi_table()

    def vf(m):
        return table[tuple(i for i in range(D) if m >> i & 1)]
    shap = shapley_exact_from(vf, D)
    allk = tuple(range(D))
    drop = np.array([table[allk] - table[tuple(k for k in allk if k != d)] for d in range(D)])
    return shap, drop, info


def part2(seed, T=3000, n=3000, K=150, KO=40):
    res = train(dict(arm="backprop", cost=0.0, seed=seed, T=T, ledger=False))
    P = {k: (np.array(v) if isinstance(v, list) else v) for k, v in res["P"].items()}
    rng = np.random.default_rng(900 + seed)
    x, y, _ = make_data(n, np.random.default_rng(7000 + seed))
    full = flat_values(x, y, P, np.ones((1, D)), np.ones((1, H1)), np.ones((1, H2)))[0]
    flat = flat_perm_pay(x, y, P, K, rng)
    # flat LOO
    E = np.ones((NP, NP)) - np.eye(NP)
    loo = full - flat_values(x, y, P, E[:, :D], E[:, D:D + H1], E[:, D + H1:])
    # flat IG, joint straight path
    m = 32
    tt = (np.arange(m) + 0.5) / m
    g = unrolled(x, y, P, np.repeat(tt[:, None], NG, 1), grad=True, chunk=8)["gates"].mean(0)
    ig = np.concatenate([g[:GX].reshape(D, H1, H2).sum((1, 2)), g[GX:GX + H1 * H2].reshape(H1, H2).sum(1),
                         g[GX + H1 * H2:]])
    # nested Owen over the unrolled tree
    ow = np.zeros(NP)
    px_t = np.zeros((D, H1, H2))
    s1_t = np.zeros((H1, H2))
    for _ in range(KO):
        order = nested_order(rng)
        Mk, _ = prefix_masks(order)
        v = unrolled(x, y, P, Mk, chunk=16)
        phi, px, s1 = owen_pay_from(order, v)
        ow += phi
        px_t += px
        s1_t += s1
    ow /= KO
    px_t /= KO
    s1_t /= KO
    # unit i's earnings from its readers = its sub-unions' total (self + its paths)
    earn1 = s1_t.sum(1) + px_t.sum((0, 2))
    keep1 = s1_t.sum(1)
    union_j = ow[D + H1:] + s1_t.sum(0) + px_t.sum((0, 1))
    n_ = float(n)
    return dict(seed=seed, ll=res["ll"], ll_bayes=res["ll_bayes"], full=full / n_,
                flat=(flat / n_).tolist(), loo=(loo / n_).tolist(), ig=(ig / n_).tolist(),
                owen=(ow / n_).tolist(), owen_earn1=(earn1 / n_).tolist(), owen_keep1=(keep1 / n_).tolist(),
                owen_union2=(union_j / n_).tolist(), ig_completeness=float(ig.sum() / full))


def part3(cfg):
    return diag(cfg)


def diag(cfg):
    """own-pay arms: is the all-on net just mis-scaled?  Test log loss before
    and after refitting one output scale and bias (1-D logistic on a separate
    calibration set), plus the logit scale."""
    r = train(cfg)
    P = {k: (np.array(v) if isinstance(v, list) else v) for k, v in r["P"].items()}
    xc, yc, _ = make_data(20000, np.random.default_rng(4343))
    xt, yt, _ = make_data(20000, np.random.default_rng(4242))
    zc, zt = flat_logits(xc, P), flat_logits(xt, P)
    best = (np.inf, 1.0, 0.0)
    for sc in np.exp(np.linspace(np.log(1e-3), np.log(10), 121)):
        for b in np.linspace(-2, 2, 41):
            l = -logsc(sc * (zc - P["b3"]) + b, yc).mean()
            if l < best[0]:
                best = (l, sc, b)
    _, sc, b = best
    r.pop("P")
    r.update(ll_rescaled=float(-logsc(sc * (zt - P["b3"]) + b, yt).mean()), scale=float(sc), logit_sd=float(np.std(zt)))
    return r


P2_KW = {}


def job(a):
    kind, arg = a
    t0 = time.time()
    out = part2(arg, **P2_KW) if kind == "p2" else (diag(arg) if kind == "diag" else part3(arg))
    out["secs"] = round(time.time() - t0, 1)
    print(kind, arg if kind == "p2" else (arg["arm"], arg["cost"], arg["seed"]), out["secs"], "s", flush=True)
    return kind, out


# ============================================================== summary
def ci(v):
    m, h = mean_ci(v)
    return [round(m, 5), round(h, 5)]


def summarize(p1, p2, p3, T3):
    S = {"setup": dict(D=D, H1=H1, H2=H2, noise_units=NOISE_UNITS, batch=BATCH, lr=LR, B=B, W0=W0, death=DEATH,
                       treasury0=TREAS0, T_part3=T3, kinds=KIND)}
    # ---- part 1
    rows = {}
    for r in p1:
        rows.setdefault((r["case"], r["k"], r["variant"]), []).append(r)
    S["part1"] = []
    for (case, k, var), rs in sorted(rows.items()):
        S["part1"].append(dict(case=case, k=k, variant=var,
                               **{f: ci([r[f] for r in rs]) for f in
                                  ("shap_input", "shap_proc", "ig20_input", "ig50_input", "ig50_proc",
                                   "ig50_completeness")}))
    S["pay_every_owen"] = PAYEVERY
    # ---- part 2
    shap_info, drop_info, info = info_values()
    S["part2"] = dict(info_shap=shap_info.round(5).tolist(), info_drop=drop_info.round(5).tolist(),
                      info_all=info["all"], bayes_gain=info["bayes"],
                      ll=ci([r["ll"] for r in p2]), ll_bayes=ci([r["ll_bayes"] for r in p2]),
                      full=ci([r["full"] for r in p2]))
    groups = dict(inputs=slice(0, D), L1=slice(D, D + H1), L2=slice(D + H1, NP))
    for meth in ("flat", "ig", "owen", "loo"):
        arr = np.array([r[meth] for r in p2])
        fulls = np.array([r["full"] for r in p2])
        S["part2"][meth] = dict(
            share={g: ci(arr[:, s].sum(1) / fulls) for g, s in groups.items()},
            per_input=[ci(arr[:, d]) for d in range(D)],
            per_kind={kd: ci(arr[:, [d for d in range(D) if KIND[d] == kd]].sum(1)) for kd in ("inf", "sub", "comp", "noise")},
            L1_free=ci(arr[:, D:D + 6].mean(1)), L1_noise_only=ci(arr[:, D + 6:D + 8].mean(1)),
            sp_info_shap=ci([spearman(a[:D], shap_info) for a in arr]),
            sp_info_drop=ci([spearman(a[:D], drop_info) for a in arr]),
            L1_units=[ci(arr[:, D + i]) for i in range(H1)], L2_units=[ci(arr[:, D + H1 + j]) for j in range(H2)])
    S["part2"]["owen_keep_frac_L1"] = ci([np.sum(r["owen_keep1"]) / np.sum(r["owen_earn1"]) for r in p2])
    S["part2"]["owen_keep_frac_L2"] = ci([np.sum(r["owen"][D + H1:]) / np.sum(r["owen_union2"]) for r in p2])
    S["part2"]["ig_completeness"] = ci([r["ig_completeness"] for r in p2])
    # ---- part 3
    S["part3"] = {}
    for key in sorted({(r["cfg"]["arm"], r["cfg"]["cost"]) for r in p3}):
        rs = [r for r in p3 if (r["cfg"]["arm"], r["cfg"]["cost"]) == key]
        prof = np.array([r["profit"] for r in rs])
        dth = np.array([r["deaths"] for r in rs])
        alive = np.array([r["alive_end"] for r in rs], float)
        pay = np.array([r["pay"] for r in rs])
        S["part3"][f"{key[0]}_c{key[1]}"] = dict(
            n=len(rs), ll=ci([r["ll"] for r in rs]), ll_rescaled=ci([r["ll_rescaled"] for r in rs]),
            out_scale_refit=ci([r["scale"] for r in rs]), ll_minus_bayes=ci([r["ll"] - r["ll_bayes"] for r in rs]),
            profit_L1_free=ci(prof[:, :6].mean(1)), profit_L1_noise=ci(prof[:, 6:8].mean(1)),
            profit_L2=ci(prof[:, 8:].mean(1)),
            frac_free_L1_profitable=ci((prof[:, :6] > 0).mean(1)),
            deaths_L1_free=ci(dth[:, :6].sum(1)), deaths_L1_noise=ci(dth[:, 6:8].sum(1)), deaths_L2=ci(dth[:, 8:].sum(1)),
            noise_units_dead_at_end=ci(1 - alive[:, 6:8].mean(1)),
            input_share_of_pay=ci(pay[:, :D].sum(1) / pay.sum(1)), total_pay_per_round=ci(pay.sum(1)),
            input_pay_noise=ci(pay[:, 8:D].sum(1)), input_pay_inf=ci(pay[:, :4].sum(1)))
    return S


def charts(S):
    cdir = os.path.join(HERE, "charts")
    src = "trader-economy/results/split_summary.json"
    bars = []
    for r in S["part1"]:
        if r["case"] == "c" and r["variant"] != "any":
            continue
        lab = {"a": f"1 input, {r['k']} dup processors", "b": f"1 processor, {r['k']} dup inputs",
               "c": "1 input, 1 processor"}[r["case"]] + ("" if r["case"] == "c" else f" ({r['variant']})")
        bars.append(dict(label=lab, value=r["shap_input"][0], ci=r["shap_input"][1], highlight=r["case"] == "c"))
    json.dump(dict(id="split_scarcity", kind="bar", title="Input's Shapley share of the value: scarcity decides",
                   unit="fraction of v(all)", source=src, bars=bars,
                   note="any = one duplicate suffices; add = duplicates add. IG shares in split_summary.json."),
              open(os.path.join(cdir, "split_scarcity.json"), "w"), indent=1)
    p2 = S["part2"]
    json.dump(dict(id="split_layers", kind="grouped_bar", title="Who gets the value: inputs vs layer-1 vs layer-2 units",
                   unit="fraction of v(all)", source=src, groups=["inputs", "layer 1", "layer 2"],
                   series={nm: [p2[m]["share"][g][0] for g in ("inputs", "L1", "L2")]
                           for nm, m in (("flat Shapley", "flat"), ("flat IG", "ig"), ("nested Owen", "owen"))},
                   ci={nm: [p2[m]["share"][g][1] for g in ("inputs", "L1", "L2")]
                       for nm, m in (("flat Shapley", "flat"), ("flat IG", "ig"), ("nested Owen", "owen"))},
                   note="12-8-4 relu net, backprop-trained, 5 seeds, 95% t-intervals"),
              open(os.path.join(cdir, "split_layers.json"), "w"), indent=1)
    bars = []
    for k, r in S["part3"].items():
        bars.append(dict(label=k, value=r["ll"][0], ci=r["ll"][1], highlight=k.startswith("backprop")))
    json.dump(dict(id="split_training", kind="bar", title="Training each unit on its own pay: test log loss",
                   unit="nats/example", source=src, bars=bars,
                   note=f"Bayes log loss {S['part2']['ll_bayes'][0]}; 5 seeds, 95% t-intervals"),
              open(os.path.join(cdir, "split_training.json"), "w"), indent=1)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "p2":
        # rerun of Part 2 with the output bias fixed at the base-rate logit
        # (v1 trained b3, so "all off" was a distorted prior and v(all) was inflated)
        rows = json.load(open(os.path.join(HERE, "results", "split_all_rows.json")))
        t0 = time.time()
        P2_KW.update(T=2000, n=3000, K=100, KO=25)
        with Pool(5) as pool:
            p2 = [o for _, o in pool.map(job, [("p2", s) for s in range(5)], chunksize=1)]
        rows["part2_v1_trained_b3"] = rows.pop("part2")
        rows["part2"] = p2
        json.dump(rows, open(os.path.join(HERE, "results", "split_all_rows.json"), "w"))
        old = json.load(open(os.path.join(HERE, "results", "split_summary.json")))
        S = summarize(rows["part1"], p2, rows["part3"], 1000)
        S["part2"]["settings"] = dict(T=2000, n_eval=3000, flat_orders=100, owen_nested_orders=25, b3="fixed 0")
        S["part2_v1_trained_b3"] = old.get("part2_v1_trained_b3", old["part2"])
        S["part3_v1_trained_b3"] = old["part3_v1_trained_b3"]
        S["runtime_s"] = old["runtime_s"] + round(time.time() - t0)
        json.dump(S, open(os.path.join(HERE, "results", "split_summary.json"), "w"), indent=1)
        charts(S)
        print("done p2", round(time.time() - t0), flush=True)
        return
    if what == "p3":
        rows = json.load(open(os.path.join(HERE, "results", "split_all_rows.json")))
        T3 = 1000
        jobs = [("p3", dict(arm=a, cost=c, seed=s, T=T3)) for a in ("owen", "shap", "backprop")
                for c in (0.0, 0.002) for s in range(5)]
        t0 = time.time()
        with Pool(5) as pool:
            p3 = [o for _, o in pool.map(job, jobs, chunksize=1)]
        rows["part3_v1_trained_b3"] = rows.pop("part3")
        rows["part3"] = p3
        json.dump(rows, open(os.path.join(HERE, "results", "split_all_rows.json"), "w"))
        S = summarize(rows["part1"], rows["part2"], p3, T3)
        old = json.load(open(os.path.join(HERE, "results", "split_summary.json")))
        S["part3_v1_trained_b3"] = old["part3"]
        S["runtime_s"] = old["runtime_s"] + round(time.time() - t0)
        json.dump(S, open(os.path.join(HERE, "results", "split_summary.json"), "w"), indent=1)
        charts(S)
        print("done p3", round(time.time() - t0), flush=True)
        return
    if what == "diag":
        jobs = [("diag", dict(arm=a, cost=0.0, seed=s, T=1500, lr=lr)) for a in ("owen", "shap", "backprop")
                for lr in (0.05,) for s in range(5)] + [("diag", dict(arm="shap", cost=0.0, seed=s, T=1500, lr=0.2)) for s in range(5)]
        with Pool(5) as pool:
            out = [o for _, o in pool.map(job, jobs, chunksize=1)]
        json.dump(out, open(os.path.join(HERE, "results", "split_diag_rows.json"), "w"))
        S = json.load(open(os.path.join(HERE, "results", "split_summary.json")))
        S["part3_diag"] = {}
        for key in sorted({(o["cfg"]["arm"], o["cfg"]["lr"]) for o in out}):
            rs = [o for o in out if (o["cfg"]["arm"], o["cfg"]["lr"]) == key]
            pr = np.array([o["profit"] for o in rs])
            S["part3_diag"][f"{key[0]}_lr{key[1]}"] = dict(
                ll=ci([o["ll"] for o in rs]), ll_rescaled=ci([o["ll_rescaled"] for o in rs]),
                scale=ci([o["scale"] for o in rs]), logit_sd=ci([o["logit_sd"] for o in rs]),
                profit_L1_free=ci(pr[:, :6].mean(1)), profit_L1_noise=ci(pr[:, 6:8].mean(1)),
                deaths_L1_free=ci([sum(o["deaths"][:6]) for o in rs]), deaths_L1_noise=ci([sum(o["deaths"][6:8]) for o in rs]))
        json.dump(S, open(os.path.join(HERE, "results", "split_summary.json"), "w"), indent=1)
        print("done diag", flush=True)
        return
    seeds = range(5) if what == "all" else range(1)
    T3 = 1500 if what == "all" else 100
    T2 = 3000 if what == "all" else 200
    t0 = time.time()
    p1 = [part1_case(c, k, v, s) for c, ks in (("a", (1, 2, 5)), ("b", (1, 2, 5)), ("c", (1,)))
          for k in ks for v in ("any", "add") for s in seeds]
    print("part1", round(time.time() - t0), "s", flush=True)
    jobs = []
    for arm in ("owen", "shap", "backprop"):
        for cost in (0.0, 0.002):
            for s in seeds:
                jobs.append(("p3", dict(arm=arm, cost=cost, seed=s, T=T3)))
    jobs[10:10] = [("p2", s) for s in seeds]
    if what != "all":
        P2_KW.update(T=T2, n=500, K=20, KO=3)
        with Pool(1) as pool:
            out = [job(j) for j in jobs]
    else:
        with Pool(5) as pool:
            out = pool.map(job, jobs, chunksize=1)
    p2 = [o for k, o in out if k == "p2"]
    p3 = [o for k, o in out if k == "p3"]
    json.dump(dict(part1=p1, part2=p2, part3=p3), open(os.path.join(HERE, "results", f"split_{what}_rows.json"), "w"))
    S = summarize(p1, p2, p3, T3)
    S["runtime_s"] = round(time.time() - t0)
    json.dump(S, open(os.path.join(HERE, "results", "split_summary.json" if what == "all" else f"split_{what}_summary.json"), "w"), indent=1)
    if what == "all":
        charts(S)
    print("done", what, S["runtime_s"], "s", flush=True)


if __name__ == "__main__":
    main()
