"""P5 maze path-parity data: perfect DFS mazes, BFS labels, batch sampler, fixed eval sets.

A maze on an H x W cell grid is stored as open[H, W, 4] (N, E, S, W passages). Perfect (DFS
recursive-backtracker) mazes are trees, so every S-T path is unique.
Input per cell, 7 channels: open N/E/S/W, is_S, is_T, token (Bernoulli 0.5).
MP-sparse: y = XOR of tokens on the S->T path, S excluded, T included; scored only at T.
MP-dense: y_v = XOR of tokens on S->v at every cell v (is_T channel is zero).
"""
import os
from collections import deque

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N E S W
OPP = (2, 3, 0, 1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def dfs_maze(h, w, rng):
    """Iterative randomized DFS (recursive backtracker); returns open[h, w, 4] bool."""
    op = np.zeros((h, w, 4), bool)
    seen = np.zeros((h, w), bool)
    st = [(int(rng.integers(h)), int(rng.integers(w)))]
    seen[st[0]] = True
    while st:
        i, j = st[-1]
        nb = [(d, i + di, j + dj) for d, (di, dj) in enumerate(DIRS)
              if 0 <= i + di < h and 0 <= j + dj < w and not seen[i + di, j + dj]]
        if not nb:
            st.pop()
            continue
        d, a, b = nb[int(rng.integers(len(nb)))]
        op[i, j, d] = op[a, b, OPP[d]] = True
        seen[a, b] = True
        st.append((a, b))
    return op


def bfs(op, s):
    """BFS from s through open passages: dist[h, w], parent index (flat, -1 at root)."""
    h, w, _ = op.shape
    dist = np.full(h * w, -1, np.int64)
    par = np.full(h * w, -1, np.int64)
    s = s[0] * w + s[1]
    dist[s] = 0
    q = deque([s])
    while q:
        v = q.popleft()
        i, j = divmod(v, w)
        for d, (di, dj) in enumerate(DIRS):
            if op[i, j, d]:
                u = (i + di) * w + (j + dj)
                if dist[u] < 0:
                    dist[u] = dist[v] + 1
                    par[u] = v
                    q.append(u)
    return dist, par


def prefix_parity(dist, par, tok):
    """y_v = XOR of tokens on S->v, S excluded, v included (tok flat)."""
    y = np.zeros_like(tok)
    for v in np.argsort(dist, kind="stable"):
        if par[v] >= 0:
            y[v] = y[par[v]] ^ tok[v]
    return y


def pack(ops):
    return np.packbits(ops.reshape(len(ops), -1), axis=1)


def unpack(bits, h, w):
    return np.unpackbits(bits, axis=1, count=h * w * 4).reshape(-1, h, w, 4).astype(bool)


def make_example(op, rng, s=None, t=None, dense=False, target_dist=None):
    """Draw S, T (distinct, uniform unless given), tokens; return x[7,h,w], y, mask, info."""
    h, w, _ = op.shape
    n = h * w
    if s is None:
        s = divmod(int(rng.integers(n)), w)
    dist, par = bfs(op, s)
    if t is None:
        if target_dist is not None:  # curriculum: T uniform among cells at the drawn distance
            dd = min(target_dist, int(dist.max()))
            cand = np.flatnonzero(dist == dd)
            t = divmod(int(cand[rng.integers(len(cand))]), w)
        else:
            while True:
                t = divmod(int(rng.integers(n)), w)
                if t != tuple(s):
                    break
    tok = (rng.random(n) < 0.5).astype(np.int64)
    y = prefix_parity(dist, par, tok)
    x = np.zeros((7, h, w), np.float32)
    x[0:4] = op.transpose(2, 0, 1)
    x[4][s] = 1
    x[6] = tok.reshape(h, w)
    ti = t[0] * w + t[1]
    onp = np.zeros(n, np.float32)  # cells on the S->T path (S excluded)
    v = ti
    while par[v] >= 0:
        onp[v] = 1
        v = par[v]
    extra = np.stack([onp.reshape(h, w), y.reshape(h, w).astype(np.float32)])  # on-path, dense prefix parity
    if dense:
        yy = y.reshape(h, w).astype(np.float32)
        mask = np.ones((h, w), np.float32)
    else:
        x[5][t] = 1
        yy = np.zeros((h, w), np.float32)
        yy[t] = y[ti]
        mask = np.zeros((h, w), np.float32)
        mask[t] = 1
    return x, yy, mask, int(dist[ti]), extra


def make_batch(ops, rng, bs, dense=False, curriculum_max=None):
    idx = rng.integers(len(ops), size=bs)
    xs, ys, ms, ds, es = [], [], [], [], []
    for k in idx:
        td = int(rng.integers(1, curriculum_max + 1)) if curriculum_max else None
        x, y, m, d, e = make_example(ops[k], rng, dense=dense, target_dist=td)
        xs.append(x); ys.append(y); ms.append(m); ds.append(d); es.append(e)
    return np.stack(xs), np.stack(ys), np.stack(ms), np.array(ds), np.stack(es)


def diameter_ends(op, rng):
    h, w, _ = op.shape
    d0, _ = bfs(op, (int(rng.integers(h)), int(rng.integers(w))))
    u = divmod(int(d0.argmax()), w)
    d1, _ = bfs(op, u)
    v = divmod(int(d1.argmax()), w)
    return (u, v) if rng.random() < 0.5 else (v, u)


def make_set(h, w, num, seed, long_path=False, dense=False):
    """Fixed eval set: x[num,7,h,w], y, mask, dist."""
    rng = np.random.default_rng(seed)
    xs, ys, ms, ds, es = [], [], [], [], []
    for _ in range(num):
        op = dfs_maze(h, w, rng)
        s = t = None
        if long_path:
            s, t = diameter_ends(op, rng)
        x, y, m, d, e = make_example(op, rng, s=s, t=t, dense=dense)
        xs.append(x); ys.append(y); ms.append(m); ds.append(d); es.append(e.astype(np.uint8))
    return dict(x=np.stack(xs).astype(np.uint8), y=np.stack(ys).astype(np.uint8), mask=np.stack(ms).astype(np.uint8),
                dist=np.array(ds), extra=np.stack(es))


# Eval sets (name -> (h, w, num, seed, long_path, dense)). Seeds are disjoint from training (seed 1).
def eval_specs(L30, L100):
    return {
        "val9": (9, 9, 1000, 101, False, False), "val13": (13, 13, 500, 102, False, False),
        "valdense9": (9, 9, 1000, 103, False, True),
        "test9": (9, 9, 1000, 201, False, False), "test13": (13, 13, 1000, 202, False, False),
        "test19": (19, 19, 1000, 203, False, False), "test33": (33, 33, 1000, 204, False, False),
        "test59": (59, 59, 500, 205, False, False),
        "long19": (19, 19, 200, 301, True, False), "long33": (33, 33, 200, 302, True, False),
        "long59": (59, 59, 200, 303, True, False),
        "lat30": (9, L30, 50, 401, True, False), "lat100": (9, L100, 50, 402, True, False),
        "stab33": (33, 33, 200, 501, False, False),
        "testdense9": (9, 9, 1000, 601, False, True),
    }


def build_all(n_train=200_000):
    os.makedirs(DATA, exist_ok=True)
    rng = np.random.default_rng(1)
    ops = np.stack([dfs_maze(9, 9, rng) for _ in range(n_train)])
    np.savez_compressed(os.path.join(DATA, "train9.npz"), bits=pack(ops), h=9, w=9)
    # S-T distance histogram on 9x9 (uniform S, T) and L_train = measured max
    rng = np.random.default_rng(7)
    ds = np.array([make_example(ops[i], rng)[3] for i in rng.integers(n_train, size=50_000)])
    diam = np.array([bfs(ops[i], diameter_ends(ops[i], rng)[0])[0].max() for i in range(5000)])
    L_train = int(max(ds.max(), 1))
    # elongated 9 x L lattices: path ~30x and ~100x L_train (tree diameter grows ~linearly in L)
    rng = np.random.default_rng(9)
    per_col = np.mean([bfs(o, diameter_ends(o, rng)[0])[0].max() / 200 for o in (dfs_maze(9, 200, rng) for _ in range(10))])
    L30, L100 = int(np.ceil(30 * L_train / per_col)), int(np.ceil(100 * L_train / per_col))
    meta = dict(L_train=L_train, st_hist=np.bincount(ds).tolist(), st_median=float(np.median(ds)),
                st_p90=float(np.percentile(ds, 90)), st_max=int(ds.max()), diam9_max=int(diam.max()),
                diam9_mean=float(diam.mean()), lat_per_col=float(per_col), L30=L30, L100=L100)
    for name, (h, w, num, seed, lp, dense) in eval_specs(L30, L100).items():
        s = make_set(h, w, num, seed, lp, dense)
        np.savez_compressed(os.path.join(DATA, f"{name}.npz"), **s)
        meta[f"{name}_dist_mean"] = float(s["dist"].mean())
        meta[f"{name}_dist_max"] = int(s["dist"].max())
        print(name, h, w, num, "dist mean %.1f max %d" % (s["dist"].mean(), s["dist"].max()), flush=True)
    import json
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    print({k: v for k, v in meta.items() if k != "st_hist"})


def load_train():
    z = np.load(os.path.join(DATA, "train9.npz"))
    return unpack(z["bits"], int(z["h"]), int(z["w"]))


def load_set(name):
    z = np.load(os.path.join(DATA, f"{name}.npz"))
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build_all()
