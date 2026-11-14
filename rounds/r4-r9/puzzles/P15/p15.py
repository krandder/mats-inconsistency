"""P15: local EM (P7 likelihood-message settlement, Bl) vs backprop on one tied, radius-1,
discrete-latent cellular automaton (U=2 units x K=4 values per cell, hid 32).

Everything is stacked over E elements (configs x seeds) with a leading E dimension on every
parameter and a per-element Adam learning rate, so one process trains many seeds with no
interaction between elements (gate G6 checks this). The window is a gather through a neighbour
index table nbr (N, W) whose value N points at an all-zero "null" cell (out of range / padding).

Tasks: prefix parity (mod 2; mod 3 secondary) in 1D (W=3) and Bansal's maze path task on an
n x n cell grid in 2D (W=9, 6 input bits per cell: passages E,S,W,N, start, goal).
The exact radius-1 rules are the systolic XOR z_k[i] = x_i ^ z_{k-1}[i-1] and dead-end filling.
"""
import math
import os
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
U_, K_, H_ = 2, 4, 32
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))          # E, S, W, N
OFF2 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]  # F.unfold order
WIN_E, WIN_S, WIN_W, WIN_N, WIN_C = 5, 7, 3, 1, 4
BIG = 10 ** 6


# ================================================================ neighbour tables
def nbr1d(L):
    i = np.arange(L)[:, None] + np.array([-1, 0, 1])[None]
    i[(i < 0) | (i >= L)] = L
    return torch.tensor(i, dtype=torch.long)


def nbr2d(n):
    r, c = np.divmod(np.arange(n * n), n)
    out = np.full((n * n, 9), n * n, dtype=np.int64)
    for w, (dr, dc) in enumerate(OFF2):
        rr, cc = r + dr, c + dc
        ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
        out[ok, w] = (rr * n + cc)[ok]
    return torch.tensor(out)


# ================================================================ prefix task
def prefix_set(rng, num, lmin, lmax, mod=2, pad_to=None):
    P = pad_to or lmax
    L = rng.integers(lmin, lmax + 1, num)
    x = rng.integers(0, 2, (num, P))
    valid = np.arange(P)[None] < L[:, None]
    x = x * valid
    y = (np.cumsum(x, 1) % mod) * valid
    return {"x": x.astype(np.float32)[..., None], "y": y.astype(np.int64), "valid": valid.astype(np.float32),
            "L": L}


def prefix_wave(x, k, mod=2):
    """g_k[i] = sum(x[i-k+1..i]) mod m; x (..., N) ints."""
    cs = np.cumsum(x, -1)
    sh = np.zeros_like(cs)
    if k == 0:
        sh = cs.copy()                      # empty window: g_0 = 0
    elif k < cs.shape[-1]:
        sh[..., k:] = cs[..., :-k]
    return (cs - sh) % mod


# ================================================================ mazes (cells)
def dfs_maze(n, rng, start=None):
    """Depth-first backtracker from `start` (as easy-to-hard make_mazes.py, which starts the DFS at the start
    cell, so the start is always a dead end)."""
    pas = np.zeros((n, n, 4), np.uint8)
    vis = np.zeros((n, n), bool)
    r0, c0 = (int(rng.integers(n)), int(rng.integers(n))) if start is None else (int(start[0]), int(start[1]))
    st = [(r0, c0)]
    vis[r0, c0] = True
    while st:
        r, c = st[-1]
        nb = [(d, r + dr, c + dc) for d, (dr, dc) in enumerate(DIRS)
              if 0 <= r + dr < n and 0 <= c + dc < n and not vis[r + dr, c + dc]]
        if not nb:
            st.pop()
            continue
        d, rr, cc = nb[int(rng.integers(len(nb)))]
        pas[r, c, d] = 1
        pas[rr, cc, (d + 2) % 4] = 1
        vis[rr, cc] = True
        st.append((rr, cc))
    return pas


def dead_end_fill(pas, end):
    """pas (M,n,n,4) uint8, end (M,n,n) bool -> fill time ft (M,n,n) int (BIG = never filled)."""
    M, n = pas.shape[0], pas.shape[1]
    filled = np.zeros((M, n, n), bool)
    ft = np.full((M, n, n), BIG, np.int64)
    k = 0
    while True:
        k += 1
        fp = np.pad(filled, ((0, 0), (1, 1), (1, 1)), constant_values=True)
        cnt = np.zeros((M, n, n), np.int64)
        for d, (dr, dc) in enumerate(DIRS):
            fn = fp[:, 1 + dr:1 + dr + n, 1 + dc:1 + dc + n]
            cnt += (pas[..., d] == 1) & ~fn
        new = ~filled & ~end & (cnt <= 1)
        if not new.any():
            break
        filled |= new
        ft[new] = k
    return ft


def maze_arrays(pas, s, g):
    """pas (M,n,n,4), s/g (M,2) -> dict with x (M,N,6), y (M,N), ft (M,N), tau (M,)."""
    M, n = pas.shape[0], pas.shape[1]
    st = np.zeros((M, n, n), np.uint8)
    go = np.zeros((M, n, n), np.uint8)
    st[np.arange(M), s[:, 0], s[:, 1]] = 1
    go[np.arange(M), g[:, 0], g[:, 1]] = 1
    ft = dead_end_fill(pas, (st + go) > 0)
    y = (ft >= BIG).astype(np.int64)
    x = np.concatenate([pas, st[..., None], go[..., None]], -1).reshape(M, n * n, 6)
    tau = np.where(ft < BIG, ft, 0).reshape(M, -1).max(1)
    return {"x": x.astype(np.uint8), "y": y.reshape(M, -1).astype(np.uint8),
            "ft": np.minimum(ft, 30000).reshape(M, -1).astype(np.int16), "tau": tau.astype(np.int32), "n": n}


def gen_mazes(num, n, seed, tau_max=None, batch=512):
    rng = np.random.default_rng(seed)
    parts, have = [], 0
    while have < num:
        s = rng.integers(0, n, (batch, 2))
        pas = np.stack([dfs_maze(n, rng, s[i]) for i in range(batch)])
        g = rng.integers(0, n, (batch, 2))
        same = (s == g).all(1)
        while same.any():
            g[same] = rng.integers(0, n, (int(same.sum()), 2))
            same = (s == g).all(1)
        a = maze_arrays(pas, s, g)
        keep = np.ones(batch, bool) if tau_max is None else a["tau"] <= tau_max
        a = {k: (v[keep] if isinstance(v, np.ndarray) else v) for k, v in a.items()}
        parts.append(a)
        have += int(keep.sum())
    out = {k: np.concatenate([p[k] for p in parts])[:num] for k in ("x", "y", "ft", "tau")}
    out["n"] = n
    return out


def maze_cached(num, n, seed, tau_max=None, cache_dir=None):
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, f"maze_n{n}_s{seed}_t{tau_max}_m{num}.npz")
    if os.path.exists(f):
        d = dict(np.load(f))
        d["n"] = int(d["n"])
        return d
    d = gen_mazes(num, n, seed, tau_max)
    tmp = f"{f}.{os.getpid()}.tmp.npz"          # parallel workers may build the same pool
    np.savez_compressed(tmp, **d)
    os.replace(tmp, f)
    return d


# ---------------------------------------------------------------- Bansal pixel format
def cells_to_pixels(x, n):
    """x (M,N,6) -> (M,3,4n+4,4n+4) float32, Bansal/easy-to-hard rendering (2x2 px blocks, 3 px pad)."""
    M = x.shape[0]
    P = 4 * n + 4
    img = np.zeros((M, 3, P, P), np.float32)
    xs = x.reshape(M, n, n, 6)
    blk = np.zeros((M, 2 * n - 1, 2 * n - 1, 3), np.float32)
    blk[:, 0::2, 0::2] = 1.0
    blk[:, 0::2, 1::2] = xs[:, :, :-1, 0:1]          # east passages
    blk[:, 1::2, 0::2] = xs[:, :-1, :, 1:2]          # south passages
    st, go = xs[..., 4] > 0, xs[..., 5] > 0
    cb = blk[:, 0::2, 0::2]
    cb[st] = np.array([0, 1, 0], np.float32)
    cb[go] = np.array([1, 0, 0], np.float32)
    blk[:, 0::2, 0::2] = cb
    up = blk.repeat(2, 1).repeat(2, 2)               # (M, 2(2n-1), 2(2n-1), 3)
    img[:, :, 3:P - 3, 3:P - 3] = up.transpose(0, 3, 1, 2)
    return img


def path_pixels(x, y, n):
    """Solution image (M,P,P): path cells plus the passages between consecutive path cells."""
    M = x.shape[0]
    P = 4 * n + 4
    xs = x.reshape(M, n, n, 6)
    ys = y.reshape(M, n, n).astype(bool)
    blk = np.zeros((M, 2 * n - 1, 2 * n - 1), np.float32)
    blk[:, 0::2, 0::2] = ys
    blk[:, 0::2, 1::2] = ys[:, :, :-1] & ys[:, :, 1:] & (xs[:, :, :-1, 0] > 0)
    blk[:, 1::2, 0::2] = ys[:, :-1, :] & ys[:, 1:, :] & (xs[:, :-1, :, 1] > 0)
    sol = np.zeros((M, P, P), np.float32)
    sol[:, 3:P - 3, 3:P - 3] = blk.repeat(2, 1).repeat(2, 2)
    return sol


def pixels_to_cells(img, sol=None):
    """Inverse of cells_to_pixels for easy-to-hard data: img (M,3,P,P) -> x (M,N,6), n, y (M,N)."""
    M, _, P, _ = img.shape
    n = (P - 4) // 4
    b = img[:, :, 3:P - 3:2, 3:P - 3:2]               # (M,3,2n-1,2n-1) block values
    opn = b.max(1) > 0
    cells = b[:, :, 0::2, 0::2]
    st = (cells[:, 1] > 0) & (cells[:, 0] == 0)
    go = (cells[:, 0] > 0) & (cells[:, 1] == 0)
    pas = np.zeros((M, n, n, 4), np.uint8)
    pas[:, :, :-1, 0] = opn[:, 0::2, 1::2]
    pas[:, :-1, :, 1] = opn[:, 1::2, 0::2]
    pas[:, :, 1:, 2] = pas[:, :, :-1, 0]
    pas[:, 1:, :, 3] = pas[:, :-1, :, 1]
    x = np.concatenate([pas, st[..., None], go[..., None]], -1).reshape(M, n * n, 6).astype(np.uint8)
    y = None if sol is None else (sol[:, 3:P - 3:4, 3:P - 3:4] > 0).reshape(M, -1).astype(np.uint8)
    return x, n, y


# ================================================================ task objects
class Task:
    """kind prefix|maze; D training depth; the training/validation/test distributions."""

    def __init__(self, kind, D, mod=2, n=None):
        self.kind, self.D, self.mod = kind, D, mod
        if kind == "prefix":
            self.Cx, self.Cy, self.W = 1, mod, 3
        else:
            self.Cx, self.Cy, self.W = 6, 2, 9
            self.n = n or {8: 4, 16: 6, 32: 9}[D]
        self.pool = None

    # --- structure
    def nbr(self, size):
        return nbr1d(size) if self.kind == "prefix" else nbr2d(size)

    def size_T(self, size):
        """'size' in T = c x size: L for prefix, n^2 for mazes."""
        return size if self.kind == "prefix" else size * size

    # --- data
    def load_pool(self, num=50000):
        if self.kind == "maze" and self.pool is None:
            self.pool = maze_cached(num, self.n, seed=1000 + self.n, tau_max=self.D)

    def train_batch(self, rng, B):
        if self.kind == "prefix":
            d = prefix_set(rng, B, max(1, self.D // 2), self.D, self.mod)
            d["xi"] = d["x"][..., 0].astype(np.int64)
            return d
        self.load_pool()
        idx = rng.integers(0, len(self.pool["tau"]), B)
        return {"x": self.pool["x"][idx].astype(np.float32), "y": self.pool["y"][idx].astype(np.int64),
                "valid": np.ones(self.pool["y"][idx].shape, np.float32), "ft": self.pool["ft"][idx].astype(np.int64)}

    def wave(self, d, k):
        """True wavefront g_k (B,N) for the oracle arm and the latent diagnostics."""
        if self.kind == "prefix":
            return prefix_wave(d["xi"], k, self.mod) * d["valid"].astype(np.int64)
        return (d["ft"] <= k).astype(np.int64)

    def eval_set(self, which, size=None, num=None):
        """which: indist | val | test.  Fixed seeds; disjoint from training (train uses the rng
        stream seeded by [1000, seed]; mazes use pool seed 1000+n)."""
        if self.kind == "prefix":
            if which == "indist":
                rng = np.random.default_rng([7001, self.D, self.mod])
                d = prefix_set(rng, num or 1000, max(1, self.D // 2), self.D, self.mod)
            else:
                rng = np.random.default_rng([7002 if which == "val" else 7003, size, self.mod])
                d = prefix_set(rng, num, size, size, self.mod)
            d["xi"] = d["x"][..., 0].astype(np.int64)
            d["size"] = size or self.D
            return d
        if which == "indist":
            m = maze_cached(num or 500, self.n, seed=2000 + self.n, tau_max=self.D)
            sz = self.n
        else:
            m = maze_cached(num, size, seed=(2100 if which == "val" else 3000) + size)
            sz = size
        return {"x": m["x"].astype(np.float32), "y": m["y"].astype(np.int64),
                "valid": np.ones(m["y"].shape, np.float32), "ft": m["ft"].astype(np.int64),
                "tau": m["tau"], "size": sz}


def stack_batches(ds, dev, dtype):
    """list over E of per-element batches -> stacked tensors (E,B,N,...)."""
    out = {"x": torch.tensor(np.stack([d["x"] for d in ds]), dtype=dtype, device=dev),
           "y": torch.tensor(np.stack([d["y"] for d in ds]), device=dev),
           "valid": torch.tensor(np.stack([d["valid"] for d in ds]), dtype=dtype, device=dev)}
    return out


# ================================================================ stacked model
def init_params(seeds, Cx, W, Cy, U=U_, K=K_, H=H_, dtype=torch.float32, dev=None):
    dev = dev or DEV
    C = Cx + U * K
    ps = {k: [] for k in ("W1", "b1", "W2", "b2", "hw", "hb")}
    for s in seeds:
        g = torch.Generator().manual_seed(int(s) * 7919 + 17)
        ps["W1"].append(torch.randn(U, H, W, C, generator=g, dtype=torch.float64) / math.sqrt(W * C))
        ps["b1"].append(torch.zeros(U, H, dtype=torch.float64))
        ps["W2"].append(torch.randn(U, K, H, generator=g, dtype=torch.float64) / math.sqrt(H))
        ps["b2"].append(torch.zeros(U, K, dtype=torch.float64))
        ps["hw"].append(torch.randn(U, K, Cy, generator=g, dtype=torch.float64) * 0.1)
        ps["hb"].append(torch.zeros(Cy, dtype=torch.float64))
    return {k: torch.stack(v).to(dev, dtype).requires_grad_(True) for k, v in ps.items()}


def gather(t, nbr, null=None):
    """t (E,B,N,F) -> (E,B,N,W,F); index N = null cell (zeros, or the vector `null`)."""
    E, B, N, Fd = t.shape
    pad = t.new_zeros(E, B, 1, Fd) if null is None else null.to(t).expand(E, B, 1, Fd)
    return torch.cat([t, pad], 2)[:, :, nbr]


def pre_x(P, x, nbr, Cx, xnull=None):
    xw = gather(x, nbr, xnull)
    return torch.einsum("ebnwc,euhwc->ebnuh", xw, P["W1"][..., :Cx]) + P["b1"][:, None, None]


def level(P, px, z, nbr, Cx):
    """one tied trader application: z (E,B,N,U,K) of level k-1 -> pre (E,B,N,U,H), logits (E,B,N,U,K)."""
    E, B, N, U, K = z.shape
    zw = gather(z.reshape(E, B, N, U * K), nbr)
    pre = px + torch.einsum("ebnwc,euhwc->ebnuh", zw, P["W1"][..., Cx:])
    lg = torch.einsum("ebnuh,eukh->ebnuk", F.relu(pre), P["W2"]) + P["b2"][:, None, None]
    return pre, lg


def head_logits(P, z):
    return P["hb"][:, None, None] + torch.einsum("ebnuk,eukc->ebnc", z, P["hw"])


@torch.no_grad()
def head_msg(P, z, y, valid):
    """lam(v) = log P(y | z[c,u] = v, other unit as realised): (E,B,N,U,K)."""
    hl = head_logits(P, z)
    cur = torch.einsum("ebnuk,eukc->ebnuc", z, P["hw"])
    lg2 = hl[:, :, :, None, None, :] - cur[:, :, :, :, None, :] + P["hw"][:, None, None]
    ls = F.log_softmax(lg2, -1)
    U, K = z.shape[3], z.shape[4]
    yi = y[:, :, :, None, None, None].expand(*y.shape, U, K, 1)
    return ls.gather(-1, yi).squeeze(-1) * valid[..., None, None]


@torch.no_grad()
def trader_msg(P, pre1, z0, lam1, nbr, Cx, max_elems=1 << 26, fan=False):
    """Likelihood message from level k+1 to level k (P7 form "l"), for every cell, unit s, value v:
    lam_k[m,s,v] = sum over (target n, window w with nbr[n,w]=m, unit a) of
                   log sum_j P_{k+1}(z[n,a]=j | z_k[m,s]=v, rest realised) exp(lam1[n,a,j]).
    pre1 (E,B,N,U,H) level-(k+1) pre-activations; z0 (E,B,N,U,K) realised level k; lam1 normalised."""
    E, B, N, U, H = pre1.shape
    K = z0.shape[-1]
    Wn = nbr.shape[1]
    A = P["W1"][..., Cx:].reshape(E, U, H, Wn, U, K)                  # e a h w s v
    Aw = A.permute(0, 1, 3, 4, 5, 2).contiguous()                      # e a w s v h
    W2T = P["W2"].transpose(2, 3).reshape(E * U, H, K)                  # (e a) h j
    zi = z0.argmax(-1)                                                  # (E,B,N,U)
    zv = z0.sum(-1)
    ziw = gather(zi.to(z0.dtype), nbr).long()                           # (E,B,N,W,S)
    zvw = gather(zv, nbr)
    out = z0.new_zeros(E, B, N, Wn, U, K)
    per_b = N * Wn * U * K * U * H
    bc = max(1, min(B, max_elems // max(1, E * per_b)))
    ei = torch.arange(E, device=z0.device)[:, None, None, None, None, None]
    ai = torch.arange(U, device=z0.device)[None, :, None, None, None, None]
    wi = torch.arange(Wn, device=z0.device)[None, None, None, None, :, None]
    si = torch.arange(U, device=z0.device)[None, None, None, None, None, :]
    for b0 in range(0, B, bc):
        b1 = min(B, b0 + bc)
        bb = b1 - b0
        zb = ziw[:, b0:b1][:, None]                                     # (E,1,b,N,W,S)
        cur = Aw[ei, ai, wi, si, zb]                                    # (E,A,b,N,W,S,H)
        pr = pre1[:, b0:b1].permute(0, 3, 1, 2, 4).contiguous()         # (E,A,b,N,H)
        p2 = (pr[:, :, :, :, None, None, None, :] - cur[:, :, :, :, :, :, None, :]
              + Aw[:, :, None, None]).relu_()                          # (E,A,b,N,W,S,V,H)
        lg = torch.bmm(p2.reshape(E * U, -1, H), W2T).view(E, U, bb, N, Wn, U, K, K)
        lg = lg + P["b2"][:, :, None, None, None, None, None, :]
        l1 = lam1[:, b0:b1].permute(0, 3, 1, 2, 4)[:, :, :, :, None, None, None, :]
        m = torch.logsumexp(F.log_softmax(lg, -1) + l1, -1).sum(1)     # (E,b,N,W,S,V)
        out[:, b0:b1] = m * zvw[:, b0:b1, :, :, :, None]
    lam0 = z0.new_zeros(E, B, N + 1, U, K)
    lam0.index_add_(2, nbr.reshape(-1), out.reshape(E, B, N * Wn, U, K))
    lam0 = lam0[:, :, :N]
    if fan:  # Bl-fan: divide by fan-out (number of non-null reading windows x U units)
        fo = torch.zeros(N + 1, device=z0.device, dtype=z0.dtype)
        fo.index_add_(0, nbr.reshape(-1), torch.ones(nbr.numel(), device=z0.device, dtype=z0.dtype))
        lam0 = lam0 / (fo[:N] * U).clamp(min=1)[None, None, :, None, None]
    return lam0


def onehot(idx, K, dtype):
    return F.one_hot(idx, K).to(dtype)


def masked_mean(v, valid):
    """v, valid (E,B,N) -> per-element mean over valid cells, summed over E (so grads are per element)."""
    return ((v * valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)).sum()


def head_ce(P, z, y, valid):
    lp = F.log_softmax(head_logits(P, z), -1)
    return masked_mean(-lp.gather(-1, y[..., None]).squeeze(-1), valid)


# ================================================================ settlement (E-step)
@torch.no_grad()
def settle(P, px, zs, pres, lgs, y, valid, nbr, Cx, local=False, fan=False, shuffle=None):
    """zs[0..T] realised one-hot levels (zs[0] = zeros), pres[k]/lgs[k] (k=1..T) of level k.
    Returns qs[k] (k=1..T) and mean |lam| per level."""
    T = len(zs) - 1
    qs, lam_abs = [None] * (T + 1), [0.0] * (T + 1)
    lam = head_msg(P, zs[T], y, valid)
    for k in range(T, 0, -1):
        if k < T:
            if local:
                lam = head_msg(P, zs[k], y, valid)
            else:
                lam = lam - lam.max(-1, keepdim=True).values
                lam = trader_msg(P, pres[k + 1], zs[k], lam, nbr, Cx, fan=fan)
        lam_abs[k] = lam.abs().mean((1, 2, 3, 4))
        la = lam if shuffle is None else lam[:, shuffle]
        qs[k] = F.softmax(F.log_softmax(lgs[k], -1) + la, -1)
    return qs, lam_abs


# ================================================================ optimiser
class SAdam:
    """Adam (torch.optim.Adam math) with a per-element learning rate lr (E,) and frozen params."""

    def __init__(self, P, lr, betas=(0.9, 0.999), eps=1e-8, frozen=()):
        self.P, self.lr, self.b1, self.b2, self.eps = P, lr, betas[0], betas[1], eps
        self.names = [k for k in P if k not in frozen]
        self.m = {k: torch.zeros_like(P[k]) for k in self.names}
        self.v = {k: torch.zeros_like(P[k]) for k in self.names}
        self.t = 0

    @torch.no_grad()
    def step(self, grads):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k in self.names:
            g = grads[k]
            if g is None:
                continue
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            den = (self.v[k].sqrt() / math.sqrt(bc2)).add_(self.eps)
            lr = self.lr.view(-1, *([1] * (g.dim() - 1))).to(g)
            self.P[k].sub_(lr / bc1 * self.m[k] / den)

    def state(self):
        return {"m": self.m, "v": self.v, "t": self.t}

    def load(self, s):
        self.m, self.v, self.t = s["m"], s["v"], s["t"]


# ================================================================ arms (one training step)
BL_ARMS = ("Bl", "Bl-ac", "Bl-local", "Bl-shuffle", "Bl-fan")
A_ARMS = ("A", "A-prog", "A-ds", "A-1step", "A-soft-prog")


def gumbel(u):
    return -torch.log(-torch.log(u.clamp(1e-12, 1 - 1e-7)))


def train_step(arm, P, task, batch, noise, sched, cfg, nbr):
    """One training step for every element. batch: stacked (E,B,N,...) tensors (+ numpy 'np' list for
    the oracle wavefront); noise: (T,E,B,N,U,K) uniforms; sched: shared (n,k) for progressive arms;
    cfg: dict of per-element tensors (tau (E,), eps float, alpha float).
    Returns (loss tensor for backward, stats dict)."""
    x, y, vm = batch["x"], batch["y"], batch["valid"]
    Cx, D = task.Cx, task.D
    E, B, N = vm.shape
    U, K = U_, K_
    dt = x.dtype
    vm5 = vm[..., None, None]
    z0 = x.new_zeros(E, B, N, U, K)
    stats = {}
    if arm in BL_ARMS or arm in ("C", "O"):
        with torch.no_grad():
            px = pre_x(P, x, nbr, Cx)
            zs, pres, lgs = [z0], [None], [None]
            z = z0
            for k in range(1, D + 1):
                if arm == "O" and k > 1:   # teacher forcing: unit 0 = true wavefront g_{k-1}
                    g = batch["wave"][k - 1]
                    z = z.clone()
                    z[:, :, :, 0] = onehot(g, K, dt) * vm[..., None]
                pre, lg = level(P, px, z, nbr, Cx)
                if arm == "Bl-ac":
                    lpm = torch.log((1 - cfg["eps"]) * F.softmax(lg, -1) + cfg["eps"] / K)
                else:
                    lpm = lg
                z = onehot((lpm + gumbel(noise[k - 1])).argmax(-1), K, dt) * vm5
                zs.append(z)
                pres.append(pre)
                lgs.append(lg)
            if arm == "O":
                zT = z.clone()
                zT[:, :, :, 0] = onehot(batch["wave"][D], K, dt) * vm[..., None]
                zs[D] = zT
        if arm == "C":
            return head_ce(P, zs[D], y, vm), stats
        pxg = pre_x(P, x, nbr, Cx)
        # teacher-forced inputs for O: level k reads the forced z_{k-1}
        if arm == "O":
            zin = []
            for k in range(1, D + 1):
                zi = zs[k - 1]
                if k > 1:
                    zi = zi.clone()
                    zi[:, :, :, 0] = onehot(batch["wave"][k - 1], K, dt) * vm[..., None]
                zin.append(zi)
        else:
            zin = zs[:D]
        Z = torch.stack(zin, 1).reshape(E, D * B, N, U, K)
        _, LG = level(P, pxg.repeat(1, D, 1, 1, 1), Z, nbr, Cx)
        LP = F.log_softmax(LG, -1).view(E, D, B, N, U, K)
        vmD = vm[:, None].expand(E, D, B, N)
        if arm == "O":
            G = torch.stack([batch["wave"][k] for k in range(1, D + 1)], 1)   # (E,D,B,N)
            ce = -LP[..., 0, :].gather(-1, G[..., None]).squeeze(-1)
            prior = ((ce * vmD).sum((1, 2, 3)) / vmD.sum((1, 2, 3))).sum()
        else:
            perm = None
            if arm == "Bl-shuffle":
                perm = torch.randperm(B, generator=cfg["gen_perm"]).to(x.device)
            qs, lam_abs = settle(P, None, zs, pres, lgs, y, vm, nbr, Cx, local=(arm == "Bl-local"),
                                 fan=(arm == "Bl-fan"), shuffle=perm)
            Q = torch.stack(qs[1:], 1)
            ce = -(Q * LP).sum(-1).mean(-1)                                   # mean over units
            prior = ((ce * vmD).sum((1, 2, 3)) / vmD.sum((1, 2, 3))).sum()
            stats["lam_abs"] = torch.stack(lam_abs[1:], 1)                   # (E,D)
        return head_ce(P, zs[D], y, vm) + prior, stats

    # ---------------- backprop arms
    px = pre_x(P, x, nbr, Cx)
    tau = cfg["tau"].view(E, 1, 1, 1, 1).to(dt)

    def st_level(z, k, soft=False):
        _, lg = level(P, px, z, nbr, Cx)
        if soft:
            return F.softmax(lg, -1) * vm5
        s = F.softmax((lg + gumbel(noise[k - 1])) / tau, -1)
        h = onehot(s.argmax(-1), K, dt)
        return ((h - s).detach() + s) * vm5

    def roll(z, k0, k1, soft=False, grad=True):
        for k in range(k0, k1):
            if grad:
                z = st_level(z, k + 1, soft)
            else:
                with torch.no_grad():
                    z = st_level(z, k + 1, soft)
        return z

    if arm == "A":
        return head_ce(P, roll(z0, 0, D), y, vm), stats
    if arm == "A-ds":
        z, tot = z0, 0.0
        for k in range(D):
            z = st_level(z, k + 1)
            tot = tot + head_ce(P, z, y, vm)
        return tot / D, stats
    if arm == "A-1step":
        z = roll(z0, 0, D - 1, grad=False).detach()
        return head_ce(P, st_level(z, D), y, vm), stats
    if arm in ("A-prog", "A-soft-prog"):
        soft = arm == "A-soft-prog"
        n, kk = sched
        z = roll(z0, 0, n, soft, grad=False).detach() if n > 0 else z0
        z = roll(z, n, n + kk, soft)
        loss = cfg["alpha"] * head_ce(P, z, y, vm)
        if cfg["alpha"] < 1:
            loss = loss + (1 - cfg["alpha"]) * head_ce(P, roll(z0, 0, D, soft), y, vm)
        return loss, stats
    raise ValueError(arm)


# ================================================================ analytic FLOPs (per element)
def flops(arm, task, B, N, sched=None, alpha=1.0, H=H_, U=U_, K=K_):
    """Analytic training FLOPs of one step for ONE element (validated by gate G4 against an op-level
    dispatch count, elementwise ops included). N = padded cells per instance.
    Backward rules: a matmul costs 2x its forward when both operands need grads, 1x when one does
    (x never needs a grad; a level whose input z is detached - Bl family, O, the first BPTT level and
    every level after a no-grad prefix - only needs the weight grad of its z-matmul)."""
    W, Cx, Cy, D = task.W, task.Cx, task.Cy, task.D
    X = B * N
    UK, UH = U * K, U * H
    f_px = 2 * X * W * Cx * UH + X * UH
    f_z = 2 * X * W * UK * UH                                                  # z-window matmul
    f_lvl = f_z + 2 * X * UH + 2 * X * UH * K + X * UK                         # zmm, add px, relu, W2, b2
    bw_det = f_z + 2 * (f_lvl - f_z)                                           # input z detached
    bw_grad = 2 * f_lvl                                                        # input z needs a grad
    bw_px = f_px
    f_samp = 9 * X * UK                                                        # gumbel(4) + add + argmax + onehot/mask
    f_head = 2 * X * UK * Cy + X * Cy + 6 * X * Cy + 3 * X                     # logits + CE
    Xm = X * W * U * K * U                                                     # message rows per level
    f_msg = 3 * Xm * H + 2 * Xm * H * K + 12 * Xm * K + 3 * Xm                 # p2, bmm, lsm+lse, sums
    f_hmsg = X * UK * (4 * Cy + 3) + 2 * X * UK * Cy
    f_q = 8 * X * UK
    if arm in BL_ARMS or arm in ("C", "O"):
        f = f_px + D * (f_lvl + f_samp)                                        # sampling pass (no grad)
        if arm == "C":
            return f + 3 * f_head
        f += f_px + bw_px + D * (f_lvl + bw_det) + 3 * D * 8 * X * UK + 3 * f_head   # gradient pass
        if arm == "O":
            return f
        if arm == "Bl-local":
            return f + D * (f_hmsg + f_q)
        return f + f_hmsg + (D - 1) * (f_msg + 2 * X * UK) + D * f_q

    def chain(nl, soft=False):
        """nl levels with grad, the first one reading a detached z: (forward, backward)."""
        ex = 6 * X * UK if soft else 8 * X * UK
        fw = nl * (f_lvl + ex + (0 if soft else f_samp))
        bw = (bw_det + (nl - 1) * bw_grad + 2 * nl * ex) if nl > 0 else 0
        return fw, bw

    if arm in ("A", "A-ds"):
        fw, bw = chain(D)
        heads = D if arm == "A-ds" else 1
        return f_px + bw_px + fw + bw + 3 * heads * f_head
    if arm == "A-1step":
        fw_ng = (D - 1) * (f_lvl + 8 * X * UK + f_samp)
        fw, bw = chain(1)
        return f_px + bw_px + fw_ng + fw + bw + 3 * f_head
    if arm in ("A-prog", "A-soft-prog"):
        soft = arm == "A-soft-prog"
        n, k = sched
        per = f_lvl + (6 * X * UK if soft else 8 * X * UK + f_samp)
        fw, bw = chain(k, soft)
        f = f_px + bw_px + n * per + fw + bw + 3 * f_head
        if alpha < 1:
            fw, bw = chain(D, soft)
            f += fw + bw + 3 * f_head
        return f
    raise ValueError(arm)


# ================================================================ evaluation helpers
@torch.no_grad()
def greedy_eval(P, task, d, Ts, cap, soft=False, chunk_cells=400_000, dtype=torch.float32, record=None,
                compact=True):
    """Greedy rollouts of every element on a fixed set d (numpy dict, same data for all elements).
    Ts: sorted list of depths to record. cap: fixed-point halt cap. Returns dict of numpy arrays:
      exact (E,len(Ts)) exact-string accuracy, cellacc (E,len(Ts)), fp_exact (E,), fp_level (E,) mean,
      fp_frac (E,) fraction halted before cap.  record: optional callback(k, z_prev, z, x, vm, sl) for
      rule-recovery window collection.
    compact (hard mode only; exact, not an approximation): a maze/string where every element has reached a
    fixed point (z_k == z_{k-1}) can never change again, because level() depends only on (px, z); it is
    dropped from the batch and its last predictions are carried to every later depth."""
    E = P["W1"].shape[0]
    M, N = d["y"].shape
    nbr = task.nbr(d["size"]).to(DEV)
    Tmax = max(max(Ts), cap)
    bs = max(1, min(M, chunk_cells // (E * N)))
    nch = -(-M // bs)
    bs = -(-M // nch)                     # balanced chunks: no tiny tail chunk running the full cap
    compact = compact and not soft
    ex = np.zeros((E, len(Ts)))
    ca = np.zeros((E, len(Ts)))
    fpx = np.zeros(E)
    fpl = np.zeros(E)
    fpf = np.zeros(E)
    for m0 in range(0, M, bs):
        m1 = min(M, m0 + bs)
        b = m1 - m0
        x = torch.tensor(d["x"][m0:m1], dtype=dtype, device=DEV)[None].expand(E, b, N, task.Cx)
        y = torch.tensor(d["y"][m0:m1], device=DEV)[None].expand(E, b, N)
        vm = torch.tensor(d["valid"][m0:m1], dtype=dtype, device=DEV)[None].expand(E, b, N)
        px = pre_x(P, x, nbr, task.Cx)
        z = x.new_zeros(E, b, N, U_, K_)
        halted = torch.zeros(E, b, dtype=torch.bool, device=DEV)
        hl = torch.full((E, b), float(cap), device=DEV, dtype=torch.float32)
        hc = torch.zeros(E, b, device=DEV, dtype=torch.float32)
        okF = torch.zeros(E, b, device=DEV, dtype=torch.float32)       # latest per item (frozen once dropped)
        cellF = torch.zeros(E, b, device=DEV, dtype=torch.float32)
        act = torch.arange(b, device=DEV)                               # active items of this chunk
        full = True
        ti = 0
        for k in range(1, Tmax + 1):
            _, lg = level(P, px, z, nbr, task.Cx)
            if soft:
                zn = F.softmax(lg, -1) * vm[..., None, None]
            else:
                zn = onehot(lg.argmax(-1), K_, dtype) * vm[..., None, None]
            if record is not None:
                record(k, z, zn, x, vm, (m0, m1))
            pred = head_logits(P, zn).argmax(-1)
            okc = ((pred == y) | (vm == 0)).float()
            ok = okc.min(-1).values
            cellacc = (okc * vm).sum(-1) / vm.sum(-1)
            if soft:
                same = (zn.argmax(-1) == z.argmax(-1)).all(-1).all(-1) & ((zn - z).abs().amax((2, 3, 4)) < 1e-6)
                fixed = same
            else:
                fixed = (zn == z).all(-1).all(-1).all(-1)
            if full:
                newly = fixed & ~halted & (k <= cap)
                hl[newly] = k - 1
                hc[newly] = ok[newly]
                halted |= newly
                okF, cellF = ok, cellacc
            else:
                h_a = halted[:, act]
                newly = fixed & ~h_a & (k <= cap)
                hl[:, act] = torch.where(newly, float(k - 1), hl[:, act])
                hc[:, act] = torch.where(newly, ok, hc[:, act])
                halted[:, act] = h_a | newly
                okF[:, act] = ok
                cellF[:, act] = cellacc
            z = zn
            while ti < len(Ts) and Ts[ti] == k:
                ex[:, ti] += okF.sum(1).cpu().numpy()
                ca[:, ti] += cellF.sum(1).cpu().numpy()
                ti += 1
            if k == cap:
                hc[~halted] = okF[~halted]
            if bool(halted.all()) and not soft and k >= 1:
                # deterministic CA at a global fixed point: every later depth predicts the same
                while ti < len(Ts):
                    ex[:, ti] += okF.sum(1).cpu().numpy()
                    ca[:, ti] += cellF.sum(1).cpu().numpy()
                    ti += 1
                break
            if ti >= len(Ts) and k >= cap:
                break
            if compact and k % 8 == 0:
                done = halted[:, act].all(0)
                nd = int(done.sum())
                if nd > 0 and 4 * nd >= act.numel():
                    keep = ~done
                    if full:
                        okF, cellF = okF.clone(), cellF.clone()
                        full = False
                    act = act[keep]
                    x, y, vm, px, z = x[:, keep], y[:, keep], vm[:, keep], px[:, keep], z[:, keep]
        fpx += hc.sum(1).cpu().numpy()
        fpl += hl.sum(1).cpu().numpy()
        fpf += halted.float().sum(1).cpu().numpy()
    return {"exact": ex / M, "cellacc": ca / M, "fp_exact": fpx / M, "fp_level": fpl / M, "fp_frac": fpf / M}


@torch.no_grad()
def indist_eval(P, task, d, S=8, dtype=torch.float32, gen=None, soft=False, diag=False):
    """In-distribution at T=D: log loss from S sampled rollouts (clamped), greedy exact accuracy,
    and (diag) latent diagnostics per level: nmi(z_k; g_k), used states, KL(q||p), |lam|, H(q)."""
    from common import mutual_info, entropy
    E = P["W1"].shape[0]
    M, N = d["y"].shape
    D = task.D
    nbr = task.nbr(N if task.kind == "prefix" else task.n).to(DEV)
    x = torch.tensor(d["x"], dtype=dtype, device=DEV)[None].expand(E, M, N, task.Cx)
    y = torch.tensor(d["y"], device=DEV)[None].expand(E, M, N)
    vm = torch.tensor(d["valid"], dtype=dtype, device=DEV)[None].expand(E, M, N)
    px = pre_x(P, x, nbr, task.Cx)
    pr = 0
    diagd = None
    for s in range(S):
        z = x.new_zeros(E, M, N, U_, K_)
        zs, pres, lgs = [z], [None], [None]
        for k in range(1, D + 1):
            pre, lg = level(P, px, z, nbr, task.Cx)
            if soft:
                z = F.softmax(lg, -1) * vm[..., None, None]
            else:
                u = torch.rand(lg.shape, generator=gen, device=DEV, dtype=dtype)
                z = onehot((lg + gumbel(u)).argmax(-1), K_, dtype) * vm[..., None, None]
            if diag and s == 0:
                zs.append(z)
                pres.append(pre)
                lgs.append(lg)
        pr = pr + F.softmax(head_logits(P, z), -1)
        if diag and s == 0 and not soft:
            qs, lam_abs = settle(P, None, zs, pres, lgs, y, vm, nbr, task.Cx)
            vmask = d["valid"].reshape(-1) > 0
            diagd = []
            for k in range(1, D + 1):
                p = F.softmax(lgs[k], -1)
                kl = (qs[k] * (torch.log(qs[k] + 1e-12) - torch.log(p + 1e-12))).sum(-1).mean(-1)
                hq = -(qs[k] * torch.log(qs[k] + 1e-12)).sum(-1).mean(-1)
                zi = zs[k].argmax(-1)
                zj = (zi[..., 0] * K_ + zi[..., 1]).cpu().numpy()
                g = task.wave(d, k).reshape(-1)[vmask]
                row = []
                for e in range(E):
                    a = zj[e].reshape(-1)[vmask]
                    hz, hg = entropy(a), entropy(g)
                    mi = mutual_info(a, g)
                    row.append({"nmi_g": mi / max(1e-9, min(hz, hg)) if min(hz, hg) > 1e-9 else 0.0,
                                "states": int(len(np.unique(a))),
                                "kl": float(((kl[e] * vm[e]).sum() / vm[e].sum())),
                                "lam_abs": float(lam_abs[k][e]), "Hq": float(((hq[e] * vm[e]).sum() / vm[e].sum()))})
                diagd.append(row)
    pr = (pr / S).clamp(1e-4, 1 - 1e-4)
    ll = -torch.log(pr.gather(-1, y[..., None]).squeeze(-1))
    ll = ((ll * vm).sum((1, 2)) / vm.sum((1, 2))).cpu().numpy()
    # greedy exact at T = D
    z = x.new_zeros(E, M, N, U_, K_)
    for k in range(1, D + 1):
        _, lg = level(P, px, z, nbr, task.Cx)
        z = (F.softmax(lg, -1) if soft else onehot(lg.argmax(-1), K_, dtype)) * vm[..., None, None]
    pred = head_logits(P, z).argmax(-1)
    ok = ((pred == y) | (vm == 0)).all(-1).float().mean(1).cpu().numpy()
    out = {"logloss": ll.tolist(), "exact": ok.tolist()}
    if diagd is not None:
        out["diag"] = diagd
    return out


# ================================================================ rule recovery (reached windows)
def joint_state(z):
    """z (...,U,K) one-hot or zero -> joint index 0..K^U-1, or K^U for the null/zero state."""
    zi = z.argmax(-1)
    j = zi[..., 0] * K_ + zi[..., 1]
    return torch.where(z.sum((-1, -2)) > 0, j, torch.full_like(j, K_ ** U_))


class WindowRecorder:
    """Collects unique reached windows per element during greedy rollouts.
    prefix: key = (x_i, s_left, s_center, s_right, s_new); maze: (6 input bits, s_c, s_E, s_S, s_W, s_N, s_new).
    States are 5-bit codes (16 = null)."""

    def __init__(self, task, E):
        self.task, self.E = task, E
        self.keys = [set() for _ in range(E)]
        self.nbr = None
        self.buf = []

    def __call__(self, k, zp, zn, x, vm, sl):
        N = zp.shape[2]
        if self.nbr is None or self.nbr.shape[0] != N:
            self.nbr = self.task.nbr(int(round(N ** 0.5)) if self.task.kind == "maze" else N).to(zp.device)
        sp = joint_state(zp)                                            # (E,b,N)
        sn = joint_state(zn)
        spw = gather(sp[..., None].to(torch.float32), self.nbr, torch.full((1,), float(K_ ** U_)))[..., 0].long()
        if self.task.kind == "prefix":
            key = x[..., 0].long()
            for w in (0, 1, 2):
                key = key * 32 + spw[..., w]
        else:
            key = torch.zeros_like(sn)
            for c in range(6):
                key = key * 2 + x[..., c].long()
            for w in (WIN_C, WIN_E, WIN_S, WIN_W, WIN_N):
                key = key * 32 + spw[..., w]
        key = key * 32 + sn
        valid = vm > 0
        # one unique() per level for all elements: element index in the bits above the key (key < 2^36)
        ekey = key + (torch.arange(key.shape[0], device=key.device, dtype=key.dtype) << self.SH)[:, None, None]
        self.buf.append(torch.unique(ekey[valid]))
        if len(self.buf) >= 64:
            self.flush()

    SH = 40

    def flush(self):
        if self.buf:
            u = torch.unique(torch.cat(self.buf)).cpu().numpy()
            self.buf = []
            e, k = u >> self.SH, u & ((1 << self.SH) - 1)
            for i in range(self.E):
                self.keys[i].update(k[e == i].tolist())

    def recover(self, P):
        """Best 2-class quotient phi (phi(null)=0) over visited states; rule holds on all reached windows
        and the head agrees with phi on visited states. Returns per element dict."""
        self.flush()
        out = []
        for e in range(self.E):
            ks = np.array(sorted(self.keys[e]), dtype=np.int64)
            out.append(recover_rule(self.task, ks, P, e))
        return out


def decode_keys(task, ks):
    cols = []
    k = ks.copy()
    nst = 6 if task.kind == "maze" else 4
    for _ in range(nst):
        cols.append(k % 32)
        k //= 32
    st = np.stack(cols[::-1], 1)                                         # prefix: left,center,right,new
    if task.kind == "prefix":
        return k[:, None], st
    bits = np.stack([(k >> (5 - c)) & 1 for c in range(6)], 1)           # E,S,W,N,start,goal
    return bits, st                                                      # st: c,E,S,W,N,new


def recover_rule(task, ks, P, e):
    if len(ks) == 0:
        return {"recovered": False, "n_windows": 0}
    inp, st = decode_keys(task, ks)
    vis = sorted(set(st.reshape(-1).tolist()) - {K_ ** U_})
    S = len(vis)
    if S > 20:
        return {"recovered": False, "n_windows": int(len(ks)), "n_states": S}
    idx = np.full(K_ ** U_ + 1, -1)
    idx[vis] = np.arange(S)
    col = np.where(st == K_ ** U_, S, idx[st])
    # head decode of each visited joint state
    hw = P["hw"][e].detach().double().cpu()
    hb = P["hb"][e].detach().double().cpu()
    dec = []
    for j in vis:
        z = torch.zeros(U_, K_, dtype=torch.float64)
        z[0, j // K_] = 1
        z[1, j % K_] = 1
        dec.append(int((hb + torch.einsum("uk,ukc->c", z, hw)).argmax()))
    dec = np.array(dec)
    if task.kind == "prefix" and task.mod != 2:
        return {"recovered": None, "n_windows": int(len(ks)), "n_states": S}

    def holds(maps):
        """maps (M,S) int8 -> (M,) bool: the rule holds on every reached window under that quotient."""
        phi = np.concatenate([maps, np.zeros((len(maps), 1), np.int8)], 1)               # null -> 0
        ph = phi[:, col]                                                                 # (M,R,nst)
        if task.kind == "prefix":
            return (ph[:, :, 3] == (inp[:, 0][None] ^ ph[:, :, 0])).all(1)
        end = (inp[:, 4] | inp[:, 5])[None]
        cnt = sum(inp[:, d][None] * (1 - ph[:, :, 1 + d]) for d in range(4))
        rule = ph[:, :, 0] | ((1 - end) & (cnt <= 1))
        return (ph[:, :, 5] == rule).all(1)

    # the head fixes the quotient: phi = head class (prefix parity) or 1 - head class (maze: filled = not on path)
    head_map = (dec if task.kind == "prefix" else 1 - dec).astype(np.int8)[None]
    good = bool(holds(head_map)[0])
    # rule under ANY 2-class quotient (head ignored): brute force when small enough, else not evaluated
    rule_only = None
    if good:
        rule_only = True
    elif 2 ** S * len(ks) <= 2 * 10 ** 8:
        rule_only = False
        ch = max(1, int(2 * 10 ** 7 // max(1, len(ks) * col.shape[1])))
        for m0 in range(0, 2 ** S, ch):
            ar = np.arange(m0, min(2 ** S, m0 + ch))
            maps = ((ar[:, None] >> np.arange(S)[None]) & 1).astype(np.int8)
            if holds(maps).any():
                rule_only = True
                break
    return {"recovered": good, "rule_only": rule_only, "n_windows": int(len(ks)), "n_states": S}


# ================================================================ training driver
CS = (0.1, 0.25, 0.5, 1, 2, 4)
TEST = {"prefix": {"mult": (4, 16, 64), "num": (1000, 500, 200)},
        "maze": {"sizes": (19, 37, 59), "num": (200, 100, 100)}}
RESDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")


def budget(task, B=32, steps=6000):
    N = task.D if task.kind == "prefix" else task.n ** 2
    return steps * flops("Bl", task, B, N)


def t_grid(size_T, cs=CS):
    return sorted(set(max(1, int(round(c * size_T))) for c in cs))


def eval_sizes(task, which):
    if task.kind == "prefix":
        if which == "val":
            return [(2 * task.D, 500)]
        return [(m * task.D, n) for m, n in zip(TEST["prefix"]["mult"], TEST["prefix"]["num"])]
    if which == "val":
        return [(2 * task.n, 200)]
    return list(zip(TEST["maze"]["sizes"], TEST["maze"]["num"]))


def run_job(job, log=print):
    """job: {id, arm, kind, D, mod, seeds, configs:[{lr,tau,alpha,eps}], frac, final, extra_T}."""
    torch.backends.cuda.matmul.allow_tf32 = False
    arm, kind, D = job["arm"], job["kind"], job["D"]
    task = Task(kind, D, job.get("mod", 2))
    seeds = job["seeds"]
    cfgs = job["configs"]
    el = [(c, s) for c in range(len(cfgs)) for s in seeds]
    E = len(el)
    B = job.get("B", 32)
    dt = torch.float64 if job.get("f64") else torch.float32
    os.makedirs(RESDIR, exist_ok=True)
    os.makedirs(CKDIR, exist_ok=True)
    ck = os.path.join(CKDIR, job["id"] + ".pt")
    outf = os.path.join(RESDIR, job["id"] + ".json")
    if os.path.exists(outf) and not job.get("force"):
        log(f"[{job['id']}] done already")
        return json.load(open(outf))
    P = init_params([s for _, s in el], task.Cx, task.W, task.Cy, dtype=dt)
    lr = torch.tensor([cfgs[c]["lr"] for c, _ in el], device=DEV, dtype=dt)
    frozen = ("W1", "b1", "W2", "b2") if arm == "C" else ()
    opt = SAdam(P, lr, frozen=frozen)
    cfg = {"tau": torch.tensor([cfgs[c].get("tau", 1.0) for c, _ in el], device=DEV, dtype=dt),
           "eps": job.get("eps", 0.05), "alpha": job.get("alpha", 1.0 if kind == "prefix" else 0.01),
           "gen_perm": torch.Generator().manual_seed(4242)}
    rngs = [np.random.default_rng([1000, s]) for _, s in el]
    gens = [torch.Generator(device=DEV).manual_seed(2000 + s) for _, s in el]
    srng = np.random.default_rng([3000, D])
    N = D if kind == "prefix" else task.n ** 2
    nbr = task.nbr(N if kind == "prefix" else task.n).to(DEV)
    Btot = budget(task) * job.get("frac", 1.0)
    marks = [Btot * (i + 1) / job.get("n_evals", 10) for i in range(job.get("n_evals", 10))]
    st = {"step": 0, "flops": 0.0, "curve": [], "wall": 0.0, "lam_abs": None}
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        for k in P:
            P[k].data.copy_(s["P"][k])
        opt.load(s["opt"])
        st = s["st"]
        for r, bs in zip(rngs, s["rngs"]):
            r.bit_generator.state = bs
        for g, gs in zip(gens, s["gens"]):
            g.set_state(gs.cpu() if hasattr(gs, "cpu") else gs)
        srng.bit_generator.state = s["srng"]
        cfg["gen_perm"].set_state(s["gperm"].cpu())
        log(f"[{job['id']}] resumed at step {st['step']}")
    ind = task.eval_set("indist")
    egen = torch.Generator(device=DEV).manual_seed(99)
    last_ck = time.time()
    t0 = time.time()
    lam_acc = None

    def save_ck():
        torch.save({"P": {k: v.detach() for k, v in P.items()}, "opt": opt.state(), "st": st,
                    "rngs": [r.bit_generator.state for r in rngs], "gens": [g.get_state() for g in gens],
                    "srng": srng.bit_generator.state, "gperm": cfg["gen_perm"].get_state()}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    while len(st["curve"]) < len(marks) and st["step"] < job.get("max_steps", 10 ** 9):
        ds = [task.train_batch(r, B) for r in rngs]
        bt = stack_batches(ds, DEV, dt)
        if arm == "O":
            bt["wave"] = [torch.tensor(np.stack([task.wave(d, k) for d in ds]), device=DEV) for k in range(D + 1)]
        sched = None
        if arm in ("A-prog", "A-soft-prog"):
            n = int(srng.integers(0, D))
            sched = (n, int(srng.integers(1, D - n + 1)))
        noise = torch.stack([torch.rand((D, B, N, U_, K_), generator=g, device=DEV, dtype=dt) for g in gens], 1)
        for k in P:
            P[k].grad = None
        loss, stats = train_step(arm, P, task, bt, noise, sched, cfg, nbr)
        grads = torch.autograd.grad(loss, [P[k] for k in opt.names], allow_unused=True)
        opt.step(dict(zip(opt.names, grads)))
        st["step"] += 1
        st["flops"] += flops(arm, task, B, N, sched, cfg["alpha"])
        if "lam_abs" in stats:
            la = stats["lam_abs"].mean(0).cpu().numpy()
            lam_acc = la if lam_acc is None else 0.99 * lam_acc + 0.01 * la
            st["lam_abs"] = lam_acc.tolist()
        if st["flops"] >= marks[len(st["curve"])] and not job.get("no_curve"):
            r = indist_eval(P, task, ind, S=8, dtype=dt, gen=egen, soft=(arm == "A-soft-prog"))
            r.update({"step": st["step"], "flops": st["flops"], "wall": st["wall"] + time.time() - t0,
                      "loss": float(loss.detach()) / E})
            st["curve"].append(r)
            log(f"[{job['id']}] step {st['step']} flops {st['flops']:.3g} exact {np.round(r['exact'], 2).tolist()} "
                f"ll {np.round(r['logloss'], 3).tolist()}")
        if job.get("no_curve") and st["flops"] >= marks[len(st["curve"])]:
            st["curve"].append({"step": st["step"], "flops": st["flops"]})
        if time.time() - last_ck > 600:
            st["wall"] += time.time() - t0
            t0 = time.time()
            save_ck()
            last_ck = time.time()
    st["wall"] += time.time() - t0
    t0 = time.time()
    save_ck()
    res = {"job": {k: v for k, v in job.items()}, "elements": [{"config": cfgs[c], "seed": s} for c, s in el],
           "steps": st["step"], "flops": st["flops"], "B_D": budget(task), "wall_train": st["wall"],
           "curve": st["curve"], "lam_abs_by_level": st["lam_abs"]}
    soft = arm == "A-soft-prog"
    if job.get("final", True):
        res["indist"] = indist_eval(P, task, ind, S=8, dtype=dt, gen=egen, soft=soft, diag=(not soft))
        for which in job.get("evals", ("val", "test")):
            rs = []
            extra = [tuple(z) for z in job.get("extra_test", [])] if which == "test" else []
            for size, num in eval_sizes(task, which) + extra:
                d = task.eval_set(which, size, num)
                sT = task.size_T(size)
                cs = list(CS) + list(job.get("extra_c", [])) if which == "test" else list(CS)
                Ts = t_grid(sT, cs)
                rec = WindowRecorder(task, E) if (which == "val" or size == eval_sizes(task, "test")[-1][0]) \
                    and not soft else None
                g = greedy_eval(P, task, d, Ts, cap=4 * sT, soft=soft, dtype=dt, record=rec)
                rr = {"size": size, "num": num, "Ts": Ts, "cs": [c for c in cs], **{k: v.tolist() for k, v in g.items()}}
                if rec is not None:
                    rr["rule"] = rec.recover(P)
                rs.append(rr)
                log(f"[{job['id']}] {which} size {size}: exact@T {np.round(g['exact'], 2).tolist()}")
            res[which] = rs
        res["wall_eval"] = time.time() - t0
    res["params"] = {k: v.detach().cpu().tolist() for k, v in P.items()} if job.get("save_params", False) else None
    json.dump(res, open(outf + ".tmp", "w"))
    os.replace(outf + ".tmp", outf)
    log(f"[{job['id']}] finished: {st['step']} steps, train {st['wall']:.0f}s, eval {res.get('wall_eval', 0):.0f}s")
    return res


if __name__ == "__main__":
    import sys
    q = json.load(open(sys.argv[1]))
    only = sys.argv[2:] or None
    for job in q:
        if only and job["id"] not in only:
            continue
        try:
            run_job(job, log=lambda m: print(time.strftime("%H:%M:%S"), m, flush=True))
        except Exception as ex:  # keep the queue going; the error is logged
            import traceback
            traceback.print_exc()
            print("JOB FAILED", job["id"], repr(ex), flush=True)
