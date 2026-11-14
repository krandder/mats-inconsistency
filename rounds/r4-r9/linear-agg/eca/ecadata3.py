"""E-CA round 3 data (eurisko, 2026-10-15): wider and deeper noisy CA grids with DISJOINT train / cal / val / test row-0
states, and exact posteriors by a bit-factored forward-backward (O(W 2^W) per row step instead of the dense O(4^W)).

Same generative model as round 1 (ecadata.py): x[t+1, c] = rule(x[t, c-1], x[t, c], x[t, c+1]) XOR Bernoulli(eps), periodic,
grid cell g = t * W + c. Round 3 defaults: T = 10, W in {14, 16}, eps = 0.015, rules 110 and 90.

Row-0 split (the round-2 diagnosis: at W = 8 only 256 row-0 states exist and ~82% of test grids also occur in training):
all 2^W row-0 states are permuted once (seed ROW0_SEED + W) and cut into buckets train 80% / cal 5% / val 7.5% / test 7.5%;
every grid of a split draws its row 0 uniformly from that split's bucket. So no test (or val, cal) grid shares its row 0
with any training grid, by construction (`overlap_report` checks it). The exact posterior uses the iid-uniform row-0 prior
(the bucket assignment is a random hash the model cannot know).

Exact inference (`fb_marginals`): forward-backward over row states, but each row-to-row step is a sweep over the W output
cells. The ring is cut at the two input bits (s_{W-1}, s_0) (4 conditioning cases); inside each case the remaining W - 2
input bits form a line that is consumed from the front while output bits are appended at the back, so the working tensor
never exceeds 2^(W+1) entries per grid and each step costs O(W 2^(W+1)). Verified against the dense code (ecadata,
256 states at W = 8) and brute force (W = 4); run `python3 ecadata3.py check`.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ecadata as D1  # noqa: E402  (round-1 module: families, blankets, parents, dense inference for checks)

T_DEF, W_DEF, EPS_DEF = 10, 12, 0.015
ROW0_SEED = 30_015_010
# round-3 spec (gemini, 09:3xZ): at W = 12 the 4,096 row-0 states split 3072 / 384 / 320 / 320 (train / cal / val / test)
BUCKETS = (("train", 0.75), ("cal", 0.09375), ("val", 0.078125), ("test", 0.078125))
DATA_DIR = os.environ.get("ECA_DATA", "/tmp/eca/data3")
FAMILIES = D1.FAMILIES
make_perm = D1.make_perm
random_reveal = D1.random_reveal
binary_entropy = D1.binary_entropy
parents = D1.parents
blankets = D1.blankets
rule_table = D1.rule_table


def family_masks(fam, n, seed, T=T_DEF, W=W_DEF):
    """Round-3 query families (spec, T = 10, W = 12): F row 0 revealed, query rows 6-9 (rows >= 6T/10); B1 rows 0 and T-1,
    query rows 4-5; B2 row 0 cols 0..W/2-1 + row T-1 + columns 0-1, query rows 4-5, cols 3..W-2; R and S1 as in round 1."""
    if fam in ("R", "S1"):
        return D1.family_masks(fam, n, seed, T, W)
    rng = np.random.default_rng(seed)
    N = T * W
    rev = np.zeros((n, T, W), bool)
    q = np.zeros(n, np.int64)
    mid = [(T - 1) // 2, T // 2] if T % 2 == 0 else [(T - 1) // 2]
    if fam == "F":
        rev[:, 0, :] = True
        qrows = np.arange((6 * T) // 10, T)
        for i in range(n):
            q[i] = rng.choice(qrows) * W + rng.integers(W)
    elif fam == "B1":
        rev[:, 0, :] = True
        rev[:, T - 1, :] = True
        for i in range(n):
            q[i] = rng.choice(mid) * W + rng.integers(W)
    elif fam == "B2":
        rev[:, 0, : W // 2] = True
        rev[:, T - 1, :] = True
        rev[:, :, 0:2] = True
        for i in range(n):
            q[i] = rng.choice(mid) * W + rng.integers(3, W - 1)
    else:
        raise ValueError(fam)
    rev = rev.reshape(n, N)
    assert not rev[np.arange(n), q].any()
    return rev, q


# ------------------------------------------------------------------------------------------------ row-0 buckets
def row0_buckets(W):
    """dict split -> sorted array of the row-0 states (ints in [0, 2^W), bit c = cell c) assigned to that split."""
    perm = np.random.default_rng(ROW0_SEED + W).permutation(2 ** W)
    out, a = {}, 0
    for i, (name, frac) in enumerate(BUCKETS):
        b = 2 ** W if i == len(BUCKETS) - 1 else a + int(round(frac * 2 ** W))
        out[name] = np.sort(perm[a:b])
        a = b
    return out


def states_to_rows(states, W):
    return ((np.asarray(states)[:, None] >> np.arange(W)[None, :]) & 1).astype(np.uint8)


def rows_to_states(rows):
    rows = np.asarray(rows, np.int64)
    return (rows << np.arange(rows.shape[-1])).sum(-1)


def simulate(rule, row0, seed, T, eps):
    """row0 [n, W] uint8 -> grids [n, T, W] uint8 (noise from rng(seed))."""
    rng = np.random.default_rng(seed)
    n, W = row0.shape
    tab = rule_table(rule)
    x = np.zeros((n, T, W), np.uint8)
    x[:, 0] = row0
    for t in range(T - 1):
        row = x[:, t]
        x[:, t + 1] = tab[4 * np.roll(row, 1, axis=1) + 2 * row + np.roll(row, -1, axis=1)] ^ (rng.random((n, W)) < eps).astype(np.uint8)
    return x


def split_seed(split, rule, W):
    base = {"train": 1_000_000, "cal": 4_000_000, "val": 7_000_000, "test": 9_000_000}[split]
    return 300_000_000 + base + 10_000 * rule + 100 * W


def load_grids(split, rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF):
    """Cached grids in GRID order [n, T*W] uint8; row 0 drawn uniformly (with replacement) from the split's bucket."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"grids-{split}-r{rule}-n{n}-T{T}W{W}-e{eps}.npy")
    if os.path.exists(path):
        return np.load(path)
    seed = split_seed(split, rule, W)
    rng = np.random.default_rng(seed)
    states = rng.choice(row0_buckets(W)[split], size=n, replace=True)
    x = simulate(rule, states_to_rows(states, W), seed + 1, T, eps).reshape(n, T * W)
    tmp = path + f".tmp{os.getpid()}.npy"
    np.save(tmp, x)
    os.replace(tmp, path)
    return x


# ------------------------------------------------------------------------------------------------ factored exact inference
def _phi(rule, eps, dev):
    tab = torch.tensor(rule_table(rule).astype(np.int64), device=dev).view(2, 2, 2)      # f(l, m, r)
    o = torch.arange(2, device=dev).view(1, 1, 1, 2)
    return torch.where(o == tab.unsqueeze(-1), torch.tensor(1 - eps, dtype=torch.float64, device=dev),
                       torch.tensor(eps, dtype=torch.float64, device=dev))                  # phi[l, m, r, o]


def _fwd_step(A, phi, W):
    """A [N, 2^W] bit-axis order (axis c = bit c, flattened C-order with bit 0 as the SLOWEST axis) -> prior of the next row,
    same layout: out(s') = sum_s A(s) prod_c phi(s_{c-1}, s_c, s_{c+1}, s'_c)."""
    N = A.shape[0]
    A = A.view(N, 2, 2 ** (W - 2), 2)                    # [N, s_0, s_1..s_{W-2}, s_{W-1}]
    out = torch.zeros(N, 2 ** W, dtype=A.dtype, device=A.device)
    for a in (0, 1):
        for b in (0, 1):
            X = A[:, b, :, a].contiguous()               # [N, line = s_1..s_{W-2}]
            L = W - 2
            # c = 0: window (a, b, s_1); append o_0
            X = (X.view(N, 2, -1, 1) * phi[a, b].view(1, 2, 1, 2)).reshape(N, -1)
            # c = 1: window (b, s_1, s_2); append o_1
            X = (X.view(N, 2, 2, -1, 1) * phi[b].view(1, 2, 2, 1, 2)).reshape(N, -1)
            # c = 2 .. W-3: window (s_{c-1}, s_c, s_{c+1}) = first three line axes; sum out s_{c-1}; append o_c
            for c in range(2, W - 2):
                X = torch.einsum("nlmrq,lmro->nmrqo", X.view(N, 2, 2, 2, -1), phi).reshape(N, -1)
            # c = W-2: window (s_{W-3}, s_{W-2}, a); sum out s_{W-3}; append o_{W-2}
            X = torch.einsum("nlmq,lmo->nmqo", X.view(N, 2, 2, -1), phi[:, :, a, :]).reshape(N, -1)
            # c = W-1: window (s_{W-2}, a, b); sum out s_{W-2}; append o_{W-1}
            X = torch.einsum("nlq,lo->nqo", X.view(N, 2, -1), phi[:, a, b, :]).reshape(N, -1)
            out += X                                     # [N, o_0..o_{W-1}] with o_0 slowest
            del L
    return out


def _bwd_step(B, phi, W):
    """B [N, 2^W] over the next row (bit 0 slowest) -> beta(s) = sum_o B(o) prod_c phi(s_{c-1}, s_c, s_{c+1}, o_c)."""
    N = B.shape[0]
    out = torch.zeros(N, 2, 2 ** (W - 2), 2, dtype=B.dtype, device=B.device)     # [N, s_0, s_1..s_{W-2}, s_{W-1}]
    for a in (0, 1):
        for b in (0, 1):
            X = B
            # c = 0: window (a, b, s_1): consume o_0 (front), introduce s_1 (back)
            X = torch.einsum("noq,so->nqs", X.view(N, 2, -1), phi[a, b]).reshape(N, -1)
            # c = 1: window (b, s_1, s_2): consume o_1, introduce s_2; s_1 is the last axis
            X = torch.einsum("noqm,mro->nqmr", X.view(N, 2, -1, 2), phi[b]).reshape(N, -1)
            # c = 2 .. W-3: window (s_{c-1}, s_c, s_{c+1}): consume o_c, introduce s_{c+1}
            for c in range(2, W - 2):
                X = torch.einsum("noqlm,lmro->nqlmr", X.view(N, 2, -1, 2, 2), phi).reshape(N, -1)
            # c = W-2: window (s_{W-3}, s_{W-2}, a): consume o_{W-2}
            X = torch.einsum("noqlm,lmo->nqlm", X.view(N, 2, -1, 2, 2), phi[:, :, a, :]).reshape(N, -1)
            # c = W-1: window (s_{W-2}, a, b): consume o_{W-1}
            X = torch.einsum("noqm,mo->nqm", X.view(N, 2, -1, 2), phi[:, a, b, :]).reshape(N, -1)
            out[:, b, :, a] = X                          # [N, s_1..s_{W-2}]
    return out.reshape(N, -1)


def _emission(rev_row, val_row, W):
    """[N, W] bool / uint8 -> per-bit factors as a [N, 2^W] multiplier (bit 0 slowest)."""
    N = rev_row.shape[0]
    e = torch.ones(N, *([2] * W), dtype=torch.float64, device=rev_row.device)
    for c in range(W):
        f = torch.ones(N, 2, dtype=torch.float64, device=rev_row.device)
        r = rev_row[:, c]
        v = val_row[:, c].long()
        f[r, 1 - v[r]] = 0.0
        shape = [N] + [1] * W
        shape[1 + c] = 2
        e = e * f.view(shape)
    return e.reshape(N, -1)


def _bit_marginals(G, W):
    """G [N, 2^W] normalised (bit 0 slowest) -> P(bit c = 1) [N, W]."""
    N = G.shape[0]
    out = torch.empty(N, W, dtype=G.dtype, device=G.device)
    for c in range(W):
        out[:, c] = G.view(N, 2 ** c, 2, 2 ** (W - 1 - c))[:, :, 1, :].sum((1, 2))
    return out


def next_det_state(rule, W, dev):
    """F(s) for every row state s, in this module's layout (flat index = sum_c s_c 2^(W-1-c), bit 0 slowest)."""
    S = 2 ** W
    idx = np.arange(S)
    B = ((idx[:, None] >> (W - 1 - np.arange(W))[None, :]) & 1).astype(np.int64)      # [S, W], column c = bit c
    f = rule_table(rule)[4 * np.roll(B, 1, axis=1) + 2 * B + np.roll(B, -1, axis=1)].astype(np.int64)
    fs = (f << (W - 1 - np.arange(W))[None, :]).sum(1)
    return torch.tensor(fs, device=dev)


def _channel(X, W, eps):
    """Independent bit-flip channel with flip probability eps on every bit: x_c -> (1 - 2 eps) x_c + eps (x_0 + x_1)."""
    N = X.shape[0]
    Y = X.view(N, *([2] * W))
    a = 1.0 - 2.0 * eps
    for ax in range(1, W + 1):
        Y = a * Y + eps * Y.sum(ax, keepdim=True)
    return Y.reshape(N, -1)


@torch.no_grad()
def fb_marginals(rule, revealed_g, xg, T=T_DEF, W=W_DEF, eps=EPS_DEF, device=None, batch=None):
    """Exact P(x_g = 1 | revealed cells), [N, T*W] float64 numpy, GRID order; uniform iid row 0. Each row step is the
    deterministic map F (scatter forward, gather backward) followed by the independent per-bit flip channel, O(W 2^W)."""
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch is None:
        batch = max(1, int(2 ** 22 // 2 ** W)) if dev.type == "cpu" else max(1, int(2 ** 25 // 2 ** W))
    fs = next_det_state(rule, W, dev)
    N = revealed_g.shape[0]
    out = np.zeros((N, T * W))
    for a0 in range(0, N, batch):
        rv = torch.tensor(np.asarray(revealed_g[a0:a0 + batch], bool).reshape(-1, T, W), device=dev)
        xv = torch.tensor(np.asarray(xg[a0:a0 + batch], np.uint8).reshape(-1, T, W), device=dev)
        n = rv.shape[0]
        em = [_emission(rv[:, t], xv[:, t], W) for t in range(T)]
        alphas = []
        A = em[0] / em[0].sum(1, keepdim=True)
        alphas.append(A)
        for t in range(1, T):
            A = _channel(torch.zeros_like(A).index_add_(1, fs, A), W, eps) * em[t]
            A = A / A.sum(1, keepdim=True)
            alphas.append(A)
        Bt = torch.ones(n, 2 ** W, dtype=torch.float64, device=dev)
        res = torch.empty(n, T, W, dtype=torch.float64, device=dev)
        for t in range(T - 1, -1, -1):
            G = alphas[t] * Bt
            G = G / G.sum(1, keepdim=True)
            res[:, t] = _bit_marginals(G, W)
            if t > 0:
                Bt = _channel(em[t] * Bt, W, eps)[:, fs]
                Bt = Bt / Bt.sum(1, keepdim=True)
        out[a0:a0 + n] = res.reshape(n, T * W).cpu().numpy()
    return out


@torch.no_grad()
def fb_marginals_sweep(rule, revealed_g, xg, T=T_DEF, W=W_DEF, eps=EPS_DEF, device=None, batch=None):
    """Exact P(x_g = 1 | revealed cells) for every grid cell, [N, T*W] float64 numpy (revealed cells -> their value).
    revealed_g, xg: [N, T*W] in GRID order. Uniform iid row 0. Factored forward-backward (see module docstring)."""
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if batch is None:
        batch = max(1, int(2 ** 21 // 2 ** W)) if dev.type == "cpu" else max(1, int(2 ** 24 // 2 ** W))
    phi = _phi(rule, eps, dev)
    N = revealed_g.shape[0]
    out = np.zeros((N, T * W))
    for a0 in range(0, N, batch):
        rv = torch.tensor(np.asarray(revealed_g[a0:a0 + batch], bool).reshape(-1, T, W), device=dev)
        xv = torch.tensor(np.asarray(xg[a0:a0 + batch], np.uint8).reshape(-1, T, W), device=dev)
        n = rv.shape[0]
        alphas = []
        A = _emission(rv[:, 0], xv[:, 0], W)
        A = A / A.sum(1, keepdim=True)
        alphas.append(A)
        for t in range(1, T):
            A = _fwd_step(A, phi, W) * _emission(rv[:, t], xv[:, t], W)
            A = A / A.sum(1, keepdim=True)
            alphas.append(A)
        Bt = torch.ones(n, 2 ** W, dtype=torch.float64, device=dev)
        res = torch.empty(n, T, W, dtype=torch.float64, device=dev)
        for t in range(T - 1, -1, -1):
            G = alphas[t] * Bt
            G = G / G.sum(1, keepdim=True)
            res[:, t] = _bit_marginals(G, W)
            if t > 0:
                Bt = _bwd_step(_emission(rv[:, t], xv[:, t], W) * Bt, phi, W)
                Bt = Bt / Bt.sum(1, keepdim=True)
        out[a0:a0 + n] = res.reshape(n, T * W).cpu().numpy()
    return out


def posterior_marginals(rule, revealed_g, xg, T=T_DEF, W=W_DEF, eps=EPS_DEF, P=None, chunk=None):
    """Same signature as ecadata.posterior_marginals (P, chunk ignored)."""
    return fb_marginals(rule, revealed_g, xg, T, W, eps)


# ------------------------------------------------------------------------------------------------ cached evaluation sets
def eval_instances(split, rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF, families=FAMILIES):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"inst-{split}-r{rule}-n{n}-T{T}W{W}-e{eps}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    xg = load_grids(split, rule, n, T, W, eps)
    out = {"xg": xg}
    fseed = {"val": 70, "test": 90}[split] * 1000 + rule
    for fi, fam in enumerate(families):
        rev, q = family_masks(fam, n, fseed + 100 * fi, T, W)
        out[f"{fam}_rev"] = rev
        out[f"{fam}_q"] = q
        out[f"{fam}_post"] = fb_marginals(rule, rev, xg, T, W, eps)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return out


def es_instances(rule, n, T=T_DEF, W=W_DEF, eps=EPS_DEF):
    xg = load_grids("val", rule, n, T, W, eps)
    rev = random_reveal(n, 77_000 + rule, T * W)
    return xg, rev


def overlap_report(rule, T=T_DEF, W=W_DEF, eps=EPS_DEF, n_train=50_000, n_cal=5_000, n_val=3_000, n_test=3_000):
    """Row-0 and whole-grid overlap between splits (exact match), plus the share of test rows (any t) seen in training rows."""
    g = {s: load_grids(s, rule, n, T, W, eps).reshape(n, T, W) for s, n in
         (("train", n_train), ("cal", n_cal), ("val", n_val), ("test", n_test))}
    st = {s: rows_to_states(g[s][:, 0]) for s in g}
    tr0 = set(st["train"].tolist())
    trg = set(map(bytes, g["train"].reshape(n_train, -1)))
    trrows = set(rows_to_states(g["train"].reshape(-1, W)).tolist())
    rep = {"W": W, "T": T, "rule": rule, "row0_states_total": 2 ** W,
           "bucket_sizes": {k: int(len(v)) for k, v in row0_buckets(W).items()},
           "distinct_row0": {s: int(len(set(st[s].tolist()))) for s in st}}
    for s in ("cal", "val", "test"):
        rep[f"{s}_row0_in_train"] = float(np.mean([v in tr0 for v in st[s].tolist()]))
        rep[f"{s}_grid_in_train"] = float(np.mean([bytes(x) in trg for x in g[s].reshape(len(g[s]), -1)]))
    trows = rows_to_states(g["test"][:, 1:].reshape(-1, W))
    rep["test_rows1plus_seen_as_any_train_row"] = float(np.mean([v in trrows for v in trows.tolist()]))
    return rep


# ------------------------------------------------------------------------------------------------ checks and timing
def _dense_check(n=40, dev="cpu"):
    worst = 0.0
    for rule in (110, 90, 30):
        for (T, W, eps) in ((6, 8, 0.015), (10, 8, 0.015), (7, 8, 0.05), (5, 6, 0.05)):
            xg = D1.simulate(rule, n, 11 + rule + W, T, W, eps).reshape(n, T * W)
            rev = random_reveal(n, 12 + rule + T, T * W, kmin=0, kmax=T * W)
            a = fb_marginals(rule, rev, xg, T, W, eps, device=dev)
            a2 = fb_marginals_sweep(rule, rev, xg, T, W, eps, device=dev)
            b = D1.posterior_marginals(rule, rev, xg, T, W, eps)
            d = max(float(np.abs(a - b).max()), float(np.abs(a2 - b).max()))
            worst = max(worst, d)
            print(f"scatter+channel and sweep vs dense: rule {rule} T{T} W{W} eps {eps}: max |diff| = {d:.2e}", flush=True)
        for (T, W, eps) in ((4, 4, 0.015), (4, 4, 0.05), (3, 5, 0.05)):
            xg = D1.simulate(rule, n, 21 + rule + W, T, W, eps).reshape(n, T * W)
            rev = random_reveal(n, 22 + rule + T, T * W, kmin=0, kmax=T * W)
            a = fb_marginals(rule, rev, xg, T, W, eps, device=dev)
            a2 = fb_marginals_sweep(rule, rev, xg, T, W, eps, device=dev)
            b = D1.brute_marginals(rule, rev, xg, T, W, eps)
            d = max(float(np.abs(a - b).max()), float(np.abs(a2 - b).max()))
            worst = max(worst, d)
            print(f"scatter+channel and sweep vs brute force: rule {rule} T{T} W{W} eps {eps}: max |diff| = {d:.2e}", flush=True)
    return worst


def _time(W, T=T_DEF, n=8, rule=110, dev="cpu", threads=1):
    torch.set_num_threads(threads)
    xg = load_grids("test", rule, 64, T, W)[:n]
    rev = random_reveal(n, 5, T * W)
    fb_marginals(rule, rev[:1], xg[:1], T, W, device=dev, batch=1)            # warm-up
    t0 = time.time()
    fb_marginals(rule, rev, xg, T, W, device=dev)
    if dev != "cpu":
        torch.cuda.synchronize()
    return (time.time() - t0) / n


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "check":
        w = _dense_check()
        print("WORST", w)
        assert w < 1e-10, w
    elif cmd == "time":
        W = int(sys.argv[2])
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        thr = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        print(f"W={W} T={T_DEF}: {_time(W, n=n, threads=thr):.3f} s per grid (CPU, {thr} thread(s), batch default)")
    elif cmd == "cross":                       # scatter+channel vs sweep at W = 12 and 14 (no dense code there)
        for W in (12, 14):
            for rule in (110, 90):
                xg = load_grids("test", rule, 64, T_DEF, W)[:24]
                rev = random_reveal(24, 9 + W, T_DEF * W)
                d = float(np.abs(fb_marginals(rule, rev, xg, T_DEF, W) - fb_marginals_sweep(rule, rev, xg, T_DEF, W)).max())
                print(f"scatter vs sweep W{W} rule {rule}: max |diff| = {d:.2e}", flush=True)
    elif cmd == "overlap":
        import json
        for W in (12, 14, 16):
            for rule in (110, 90):
                print(json.dumps(overlap_report(rule, W=W)), flush=True)
