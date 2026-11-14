"""E-CA data: noisy elementary CA spacetime grids, a FIXED permutation of cell ids, exact posteriors by
forward-backward over the row chain, query-family masks (eurisko, 2026-10-15, design v2).

Grid: T rows x W cols, x[t+1, c] = rule(x[t, c-1], x[t, c], x[t, c+1]) XOR Bernoulli(eps), periodic, row 0 iid Bern(1/2).
Grid cell g = t * W + c. The MODEL sees only permuted token ids: token j holds grid cell PERM[j] (xp = xg[:, PERM]).
Everything grid-structured (families, exact inference, blanket labels) lives here, on the EVALUATION side.
Exact inference is used only for scoring, the D0 gate and stiffness evaluation, never for training or selection.
"""
from __future__ import annotations

import os

import numpy as np

T_DEF, W_DEF, EPS_DEF = 6, 8, 0.015
PERM_SEED = 20261015
TRAIN_BASE, CAL_BASE, VAL_BASE, TEST_BASE = 1_000_000, 4_000_000, 7_000_000, 9_000_000
DATA_DIR = os.environ.get("ECA_DATA", "/tmp/eca/data")


def make_perm(T=T_DEF, W=W_DEF) -> np.ndarray:
    """The fixed permutation: token j <- grid cell PERM[j]. Seed depends only on (T, W), never on rule or run."""
    return np.random.default_rng(PERM_SEED + 1000 * T + W).permutation(T * W)


def rule_table(rule: int) -> np.ndarray:
    return np.array([(rule >> k) & 1 for k in range(8)], np.uint8)   # index 4*l + 2*c + r


def simulate(rule: int, n: int, seed: int, T=T_DEF, W=W_DEF, eps=EPS_DEF) -> np.ndarray:
    """[n, T, W] uint8 grids."""
    rng = np.random.default_rng(seed)
    tab = rule_table(rule)
    x = np.zeros((n, T, W), np.uint8)
    x[:, 0] = rng.integers(0, 2, (n, W))
    for t in range(T - 1):
        row = x[:, t]
        l = np.roll(row, 1, axis=1)     # x[c-1]
        r = np.roll(row, -1, axis=1)    # x[c+1]
        x[:, t + 1] = tab[4 * l + 2 * row + r] ^ (rng.random((n, W)) < eps).astype(np.uint8)
    return x


def bits(W):
    s = np.arange(2 ** W)
    return ((s[:, None] >> np.arange(W)[None, :]) & 1).astype(np.uint8)   # [S, W], bit c of state s


def transition(rule: int, W=W_DEF, eps=EPS_DEF) -> np.ndarray:
    """P[s, s'] = P(row_{t+1} = s' | row_t = s)."""
    B = bits(W)
    tab = rule_table(rule)
    f = tab[4 * np.roll(B, 1, axis=1) + 2 * B + np.roll(B, -1, axis=1)]          # [S, W] deterministic next row
    fs = (f.astype(np.int64) << np.arange(W)).sum(1)                                # next state index
    S = 2 ** W
    d = np.array([bin(v).count("1") for v in range(S)])                             # popcount
    flips = d[fs[:, None] ^ np.arange(S)[None, :]]                                  # [S, S'] number of noise flips
    return (eps ** flips) * ((1 - eps) ** (W - flips))


def row_masks(revealed_g: np.ndarray, xg: np.ndarray, T, W):
    """revealed_g, xg: [N, T*W] (grid order) -> per-row bitmask and value [N, T] int64."""
    rv = revealed_g.reshape(-1, T, W).astype(np.int64)
    xv = (xg.reshape(-1, T, W).astype(np.int64)) * rv
    sh = np.arange(W)
    return (rv << sh).sum(2), (xv << sh).sum(2)


def posterior_marginals(rule, revealed_g, xg, T=T_DEF, W=W_DEF, eps=EPS_DEF, P=None, chunk=4096):
    """Exact P(x_g = 1 | revealed cells) for every grid cell, [N, T*W] float64 (revealed cells -> their value).
    Forward-backward over rows (2^W states), uniform iid row 0."""
    if P is None:
        P = transition(rule, W, eps)
    B = bits(W).astype(np.float64)
    S = 2 ** W
    sidx = np.arange(S)
    N = revealed_g.shape[0]
    out = np.zeros((N, T * W))
    for a in range(0, N, chunk):
        m, v = row_masks(revealed_g[a:a + chunk], xg[a:a + chunk], T, W)
        n = m.shape[0]
        e = (((sidx[None, None, :] ^ v[:, :, None]) & m[:, :, None]) == 0).astype(np.float64)   # [n, T, S]
        al = np.zeros((n, T, S))
        cur = e[:, 0] / S
        cur /= cur.sum(1, keepdims=True)
        al[:, 0] = cur
        for t in range(1, T):
            cur = (cur @ P) * e[:, t]
            cur /= cur.sum(1, keepdims=True)
            al[:, t] = cur
        be = np.ones((n, T, S))
        cur = np.ones((n, S))
        for t in range(T - 2, -1, -1):
            cur = (cur * e[:, t + 1]) @ P.T
            cur /= cur.sum(1, keepdims=True)
            be[:, t] = cur
        g = al * be
        g /= g.sum(2, keepdims=True)
        out[a:a + n] = (g @ B).reshape(n, T * W)                                    # [n, T, W]
    return out


def brute_marginals(rule, revealed_g, xg, T, W, eps):
    """Brute force over all 2^(T*W) grids (T*W <= 16): exact P(x_g = 1 | revealed)."""
    n_cells = T * W
    assert n_cells <= 16
    G = ((np.arange(2 ** n_cells)[:, None] >> np.arange(n_cells)[None, :]) & 1).astype(np.uint8)  # [2^n, n] grid order
    tab = rule_table(rule)
    g3 = G.reshape(-1, T, W)
    lp = np.full(G.shape[0], -W * np.log(2.0))
    for t in range(T - 1):
        row = g3[:, t]
        f = tab[4 * np.roll(row, 1, axis=1) + 2 * row + np.roll(row, -1, axis=1)]
        fl = (f != g3[:, t + 1]).sum(1)
        lp += fl * np.log(eps) + (W - fl) * np.log(1 - eps)
    p = np.exp(lp)
    out = np.zeros(revealed_g.shape, np.float64)
    for i in range(revealed_g.shape[0]):
        r = revealed_g[i].astype(bool)
        ok = (G[:, r] == xg[i, r][None, :]).all(1)
        w = p * ok
        out[i] = (w @ G) / w.sum()
    return out


def binary_entropy(p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def parents(g, T, W):
    t, c = divmod(int(g), W)
    if t == 0:
        return None
    return [(t - 1) * W + (c + d) % W for d in (-1, 0, 1)]


def blankets(g, T, W):
    """Valid deterministic local blankets of grid cell g (sets of 3 cells): its 3 past parents, and for each of its
    3 children, the child plus the child's other two parents (backward/sideways triples). Evaluation-only labels."""
    t, c = divmod(int(g), W)
    out = []
    if t > 0:
        out.append(parents(g, T, W))
    if t < T - 1:
        for d in (-1, 0, 1):                       # child (t+1, c-d) has g as parent at offset d
            cc = (c - d) % W
            child = (t + 1) * W + cc
            co = [t * W + (cc + e) % W for e in (-1, 0, 1) if (cc + e) % W != c]
            out.append([child] + co)
    return out


# ------------------------------------------------------------------------------------------------ query families
FAMILIES = ("F", "B1", "B2", "R", "S1")


def family_masks(fam: str, n: int, seed: int, T=T_DEF, W=W_DEF):
    """Evidence masks (grid order, bool [n, T*W]) and one query cell per instance (grid id [n]).
    F  forward: row 0 revealed, query a cell in rows ceil(T/2)..T-1.
    B1 two-row bridge: rows 0 and T-1 revealed, query a cell in the middle rows (T=6: rows 2-3).
    B2 anchor bridge: row 0 cols 0..W/2-1, row T-1, columns 0 and 1 (all rows) revealed; query an unrevealed interior
       cell (rows 1..T-2, cols 2..W-1).
    R  random: reveal count k ~ U{1..T*W-1}, query a random hidden cell.
    S1 one-step: like R but the query's 3 past parents are revealed (D0 gate's one-step check)."""
    rng = np.random.default_rng(seed)
    N = T * W
    rev = np.zeros((n, T, W), bool)
    q = np.zeros(n, np.int64)
    rows = np.arange(T)
    if fam == "F":
        rev[:, 0, :] = True
        qrows = rows[(T + 1) // 2:]
        for i in range(n):
            q[i] = rng.choice(qrows) * W + rng.integers(W)
    elif fam == "B1":
        rev[:, 0, :] = True
        rev[:, T - 1, :] = True
        mid = [(T - 1) // 2, T // 2] if T % 2 == 0 else [(T - 1) // 2]
        for i in range(n):
            q[i] = rng.choice(mid) * W + rng.integers(W)
    elif fam == "B2":
        rev[:, 0, : W // 2] = True
        rev[:, T - 1, :] = True
        rev[:, :, 0:2] = True
        for i in range(n):
            q[i] = rng.integers(1, T - 1) * W + rng.integers(2, W)
    elif fam in ("R", "S1"):
        for i in range(n):
            k = rng.integers(1, N)
            r = np.zeros(N, bool)
            r[rng.permutation(N)[:k]] = True
            if fam == "R":
                hid = np.flatnonzero(~r)
                q[i] = rng.choice(hid)
            else:
                g = rng.integers(W, N)          # rows >= 1
                r[parents(g, T, W)] = True
                r[g] = False
                q[i] = g
            rev[i] = r.reshape(T, W)
    else:
        raise ValueError(fam)
    rev = rev.reshape(n, N)
    assert not rev[np.arange(n), q].any()
    return rev, q


def random_reveal(n, seed, N, kmin=1, kmax=None):
    """Training-distribution masks: reveal count k ~ U{kmin..kmax} (default 1..N-1), uniform random subset."""
    rng = np.random.default_rng(seed)
    kmax = N - 1 if kmax is None else kmax
    k = rng.integers(kmin, kmax + 1, n)
    order = np.argsort(rng.random((n, N)), 1)
    pos = np.argsort(order, 1)
    return pos < k[:, None]


# ------------------------------------------------------------------------------------------------ cached splits
def split_seed(split, rule):
    base = {"train": TRAIN_BASE, "cal": CAL_BASE, "val": VAL_BASE, "test": TEST_BASE}[split]
    return base + 10_000 * rule


def load_grids(split, rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF):
    """Cached grids in GRID order [n, T*W] uint8 (the caller permutes before anything reaches the model)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"grids-{split}-r{rule}-n{n}-T{T}W{W}-e{eps}.npy")
    if os.path.exists(path):
        return np.load(path)
    x = simulate(rule, n, split_seed(split, rule), T, W, eps).reshape(n, T * W)
    tmp = path + f".tmp{os.getpid()}.npy"
    np.save(tmp, x)
    os.replace(tmp, path)
    return x


def eval_instances(split, rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF, families=FAMILIES):
    """Per family: grids (grid order), evidence masks, query cell, exact posterior marginals (all cells).
    Instance i of every family uses grid i of the split. Cached."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"inst-{split}-r{rule}-n{n}-T{T}W{W}-e{eps}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    xg = load_grids(split, rule, n, T, W, eps)
    P = transition(rule, W, eps)
    out = {"xg": xg}
    fseed = {"val": 70, "test": 90}[split] * 1000 + rule
    for fi, fam in enumerate(families):
        rev, q = family_masks(fam, n, fseed + 100 * fi, T, W)
        post = posterior_marginals(rule, rev, xg, T, W, eps, P)
        out[f"{fam}_rev"] = rev
        out[f"{fam}_q"] = q
        out[f"{fam}_post"] = post
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return out


def es_instances(rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF):
    """Early-stopping set: validation grids with training-distribution masks (k ~ U{1..N-1}). Outcomes only."""
    xg = load_grids("val", rule, n, T, W, eps)
    rev = random_reveal(n, 77_000 + rule, T * W)
    return xg, rev


if __name__ == "__main__":
    import sys
    import time
    # brute-force cross-check of the forward-backward code at W=4, T=4 (eps 0.015 and 0.05)
    worst = 0.0
    for rule in (110, 90, 30):
        for eps in (0.015, 0.05):
            T, W = 4, 4
            xg = simulate(rule, 60, 5 + rule, T, W, eps).reshape(60, T * W)
            rev = random_reveal(60, 6 + rule, T * W, kmin=0, kmax=T * W)
            a = posterior_marginals(rule, rev, xg, T, W, eps)
            b = brute_marginals(rule, rev, xg, T, W, eps)
            d = np.abs(a - b).max()
            worst = max(worst, d)
            print(f"brute-force check rule {rule} eps {eps} W4 T4: max |fb - brute| = {d:.2e}")
    assert worst < 1e-9, worst
    Ts = [int(a) for a in sys.argv[1:]] or [T_DEF]
    for T in Ts:
        W = W_DEF
        p = make_perm(T, W)
        print("T", T, "perm", p.tolist())
        for rule in (110, 90):
            t0 = time.time()
            for split, n in (("train", 50_000), ("cal", 5_000)):
                load_grids(split, rule, n, T, W)
            for split, n in (("val", 3000), ("test", 3000)):
                d = eval_instances(split, rule, n, T, W)
                h = {f: float(binary_entropy(d[f"{f}_post"][np.arange(n), d[f"{f}_q"]]).mean()) for f in FAMILIES}
                print(rule, split, "T", T, "mean exact H(Y|E) nats per family", {k: round(v, 3) for k, v in h.items()},
                      f"{time.time()-t0:.1f}s", flush=True)
