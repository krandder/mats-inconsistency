"""P2 data: maze-dataset DFS-percolation mazes drawn on the raw (2n+1)^2 pixel lattice.

  gen.py pilot            tie fraction / contested-region size, 1000 mazes per cell
  gen.py pool  [--num N]  training pool at n=9 (p=0: 40%, 0.05: 30%, 0.1: 30%), dead-end start
  gen.py test             frozen test, dev and n=59 extension sets (sha256 recorded)

Per maze: open pixels, start/goal pixel, BFS distances d_s, d_g, D = d_s[goal],
bitmask of all shortest-path pixels (d_s + d_g == D), canonical path (move order D,R,U,L),
covariates: contested size (pixels on some but not all shortest paths), 8x8 tiles touched,
path length, loop count (E - V + 1 of the open-pixel graph).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MOVES = ((1, 0), (0, 1), (-1, 0), (0, -1))  # D, R, U, L (canonical order)


def lattice(n, p):
    from maze_dataset import LatticeMazeGenerators
    m = LatticeMazeGenerators.gen_dfs_percolation(np.array([n, n]), p=p)
    cl = m.connection_list
    L = 2 * n + 1
    op = np.zeros((L, L), bool)
    op[1::2, 1::2] = True
    op[2:-1:2, 1::2] = cl[0][:-1, :]  # down connections (last row is always False)
    op[1::2, 2:-1:2] = cl[1][:, :-1]  # right connections
    deg = cl[0].astype(int) + cl[1].astype(int)
    deg[1:, :] += cl[0][:-1, :]
    deg[:, 1:] += cl[1][:, :-1]
    return op, deg


def bfs(op, s):
    L = op.shape[0]
    d = np.full((L, L), -1, np.int32)
    d[s] = 0
    q = deque([s])
    while q:
        i, j = q.popleft()
        for di, dj in MOVES:
            a, b = i + di, j + dj
            if 0 <= a < L and 0 <= b < L and op[a, b] and d[a, b] < 0:
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    return d


def canon_path(op, dg, s):
    L = op.shape[0]
    path = np.zeros((L, L), bool)
    cur = s
    path[cur] = True
    while dg[cur] > 0:
        for di, dj in MOVES:
            a, b = cur[0] + di, cur[1] + dj
            if 0 <= a < L and 0 <= b < L and op[a, b] and dg[a, b] == dg[cur] - 1:
                cur = (a, b)
                break
        path[cur] = True
    return path


def maze_record(n, p, dead_end, rng):
    """One maze or None (percolated maze with no dead end when dead_end=True)."""
    op, deg = lattice(n, p)
    cells = [(r, c) for r in range(n) for c in range(n)]
    if dead_end:
        de = [rc for rc in cells if deg[rc] == 1]
        if not de:
            return None
        sc = de[rng.integers(len(de))]
    else:
        sc = cells[rng.integers(len(cells))]
    while True:
        gc = cells[rng.integers(len(cells))]
        if gc != sc:
            break
    s = (2 * sc[0] + 1, 2 * sc[1] + 1)
    g = (2 * gc[0] + 1, 2 * gc[1] + 1)
    ds, dg = bfs(op, s), bfs(op, g)
    D = int(ds[g])
    dag = (ds >= 0) & (dg >= 0) & (ds + dg == D)
    layer_count = np.bincount(ds[dag], minlength=D + 1)
    contested = dag & (layer_count[np.clip(ds, 0, D)] > 1)
    V = int(op.sum())
    E = int((op[1:, :] & op[:-1, :]).sum() + (op[:, 1:] & op[:, :-1]).sum())
    tiles = np.unique((np.argwhere(contested) // 8), axis=0)
    return dict(open=op, s=s, g=g, ds=ds, dg=dg, D=D, dag=dag, canon=canon_path(op, dg, s),
                contested=int(contested.sum()), tiles=int(len(tiles)), loops=E - V + 1,
                tie=bool(contested.any()), p=p, dead_end=dead_end, n=n)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    return np.random.default_rng(seed)


# ---------------------------------------------------------------- pilot
def pilot_cell(args):
    n, p, de, num, seed = args
    rng = seed_all(seed)
    recs, drops, t0 = [], 0, time.time()
    while len(recs) < num:
        r = maze_record(n, p, de, rng)
        if r is None:
            drops += 1
            continue
        recs.append((r["tie"], r["contested"], r["tiles"], r["D"], r["loops"]))
    a = np.array(recs, float)
    tie = a[:, 0] > 0
    cs = a[tie, 1]
    return dict(n=n, p=p, dead_end=de, num=num, drops=drops, tie_frac=float(tie.mean()),
                contested_median=float(np.median(cs)) if tie.any() else 0.0,
                contested_q25=float(np.percentile(cs, 25)) if tie.any() else 0.0,
                contested_q75=float(np.percentile(cs, 75)) if tie.any() else 0.0,
                frac_tie_contested_le8=float((cs <= 8).mean()) if tie.any() else 0.0,
                frac_tie_contested_gt32=float((cs > 32).mean()) if tie.any() else 0.0,
                frac_tie_one_tile=float((a[tie, 2] <= 1).mean()) if tie.any() else 0.0,
                D_mean=float(a[:, 3].mean()), loops_mean=float(a[:, 4].mean()),
                sec_per_maze=(time.time() - t0) / (num + drops))


def cmd_pilot(a):
    jobs, k = [], 0
    for n in (9, 13, 19, 25, 33):
        for p in (0.05, 0.1):
            for de in (True, False):
                jobs.append((n, p, de, a.num, 10_000 + k))
                k += 1
    with Pool(a.procs) as pool:
        out = pool.map(pilot_cell, jobs)
    for r in out:
        print(json.dumps(r))
    with open(os.path.join(HERE, "results", "pilot_ties.json"), "w") as f:
        json.dump(out, f, indent=1)


# ---------------------------------------------------------------- pool
def pool_chunk(args):
    n, p, num, seed = args
    rng = seed_all(seed)
    L = 2 * n + 1
    out = dict(open=[], s=[], g=[], ds=[], dg=[], canon=[], tie=[], contested=[], p=[])
    drops = 0
    while len(out["s"]) < num:
        r = maze_record(n, p, True, rng)
        if r is None:
            drops += 1
            continue
        out["open"].append(np.packbits(r["open"].ravel()))
        out["canon"].append(np.packbits(r["canon"].ravel()))
        out["s"].append(r["s"][0] * L + r["s"][1])
        out["g"].append(r["g"][0] * L + r["g"][1])
        out["ds"].append(np.clip(r["ds"], 0, 255).astype(np.uint8).ravel())
        out["dg"].append(np.clip(r["dg"], 0, 255).astype(np.uint8).ravel())
        out["tie"].append(r["tie"])
        out["contested"].append(r["contested"])
        out["p"].append(p)
    return {k: np.array(v) for k, v in out.items()}, drops


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_pool(a):
    n, chunk = 9, 1000
    nums = {0.0: int(0.4 * a.num), 0.05: int(0.3 * a.num), 0.1: a.num - int(0.4 * a.num) - int(0.3 * a.num)}
    jobs, k = [], 0
    for p, num in nums.items():
        for i in range(0, num, chunk):
            jobs.append((n, p, min(chunk, num - i), 20_000 + k))
            k += 1
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(pool_chunk, jobs)
    parts = [r[0] for r in res]
    drops = {p: 0 for p in nums}
    for j, r in zip(jobs, res):
        drops[j[1]] += r[1]
    data = {key: np.concatenate([q[key] for q in parts]) for key in parts[0]}
    perm = np.random.default_rng(0).permutation(len(data["s"]))
    data = {key: v[perm] for key, v in data.items()}
    path = os.path.join(DATA, "pool_n9.npz")
    np.savez_compressed(path, n=n, **data)
    meta = dict(file="pool_n9.npz", sha256=sha256(path), num=int(len(data["s"])), n=n,
                per_p={str(p): v for p, v in nums.items()},
                dropped_no_dead_end={str(p): v for p, v in drops.items()},
                drop_rate={str(p): drops[p] / (drops[p] + nums[p]) for p in nums},
                tie_frac={str(p): float(data["tie"][data["p"] == p].mean()) for p in nums},
                seconds=time.time() - t0, bytes=os.path.getsize(path))
    json.dump(meta, open(os.path.join(DATA, "pool_n9.meta.json"), "w"), indent=1)
    print(json.dumps(meta))


# ---------------------------------------------------------------- test sets
def test_cell(args):
    n, p, de, n_tie, n_notie, seed = args
    rng = seed_all(seed)
    ties, noties, drops = [], [], 0
    while len(ties) < n_tie or len(noties) < n_notie:
        r = maze_record(n, p, de, rng)
        if r is None:
            drops += 1
            continue
        if r["tie"] and len(ties) < n_tie:
            ties.append(r)
        elif not r["tie"] and len(noties) < n_notie:
            noties.append(r)
    recs = ties + noties
    L = 2 * n + 1
    return dict(
        n=n, p=p, dead_end=de, drops=drops,
        open=np.stack([np.packbits(r["open"].ravel()) for r in recs]),
        dag=np.stack([np.packbits(r["dag"].ravel()) for r in recs]),
        s=np.array([r["s"][0] * L + r["s"][1] for r in recs]),
        g=np.array([r["g"][0] * L + r["g"][1] for r in recs]),
        D=np.array([r["D"] for r in recs]), tie=np.array([r["tie"] for r in recs]),
        contested=np.array([r["contested"] for r in recs]),
        tiles=np.array([r["tiles"] for r in recs]), loops=np.array([r["loops"] for r in recs]))


def cmd_test(a):
    jobs, k = [], 0
    specs = []
    for n in (9, 13, 19, 25, 33):
        for p in (0.0, 0.05, 0.1):
            for de in (True, False):
                nt, nn = (0, 300) if p == 0 else (a.per // 2, a.per // 2)
                specs.append(("test", n, p, de, nt, nn))
                specs.append(("dev", n, p, de, 0, a.dev))
    for p in (0.05, 0.1):
        specs.append(("ext", 59, p, True, a.ext, 0))
    for sp in specs:
        jobs.append(sp[1:] + (30_000 + k,))
        k += 1
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(test_cell, jobs)
    meta = {"seconds": None, "files": {}}
    for split in ("test", "dev", "ext"):
        cells = [r for sp, r in zip(specs, res) if sp[0] == split]
        arrays = {}
        for i, c in enumerate(cells):
            key = f"n{c['n']}_p{c['p']}_de{int(c['dead_end'])}"
            for f in ("open", "dag", "s", "g", "D", "tie", "contested", "tiles", "loops"):
                arrays[f"{key}__{f}"] = c[f]
        path = os.path.join(DATA, f"{split}.npz")
        np.savez_compressed(path, **arrays)
        meta["files"][split] = dict(sha256=sha256(path), bytes=os.path.getsize(path),
                                    cells={f"n{c['n']}_p{c['p']}_de{int(c['dead_end'])}":
                                           dict(num=int(len(c["s"])), ties=int(c["tie"].sum()), drops=c["drops"])
                                           for c in cells})
    meta["seconds"] = time.time() - t0
    json.dump(meta, open(os.path.join(DATA, "sets.meta.json"), "w"), indent=1)
    print(json.dumps({k: (v["sha256"], v["bytes"]) for k, v in meta["files"].items()}))


# ---------------------------------------------------------------- loading
def unpack(bits, L):
    return np.unpackbits(bits, axis=-1, count=L * L).reshape(-1, L, L).astype(bool)


def load_cell(split, n, p, de):
    z = np.load(os.path.join(DATA, f"{split}.npz"))
    key = f"n{n}_p{p}_de{int(de)}"
    L = 2 * n + 1
    out = {f: z[f"{key}__{f}"] for f in ("s", "g", "D", "tie", "contested", "tiles", "loops")}
    out["open"] = unpack(z[f"{key}__open"], L)
    out["dag"] = unpack(z[f"{key}__dag"], L)
    out["n"], out["L"] = n, L
    return out


def maze_hash(op, s, g):
    return hashlib.sha1(np.packbits(op.ravel()).tobytes() + int(s).to_bytes(4, "little") + int(g).to_bytes(4, "little")).hexdigest()


def cmd_overlap(a):
    z = np.load(os.path.join(DATA, "pool_n9.npz"))
    L = 19
    pool = set()
    for o, s, g in zip(z["open"], z["s"], z["g"]):
        pool.add(maze_hash(unpack(o[None], L)[0], s, g))
    hits, tot = 0, 0
    for split in ("test", "dev"):
        for p in (0.0, 0.05, 0.1):
            for de in (True, False):
                c = load_cell(split, 9, p, de)
                for o, s, g in zip(c["open"], c["s"], c["g"]):
                    tot += 1
                    hits += maze_hash(o, s, g) in pool
    r = dict(pool=len(pool), test_dev_n9=tot, overlap=hits)
    json.dump(r, open(os.path.join(HERE, "results", "overlap.json"), "w"))
    print(r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--per", type=int, default=300)
    ap.add_argument("--dev", type=int, default=100)
    ap.add_argument("--ext", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    {"pilot": cmd_pilot, "pool": cmd_pool, "test": cmd_test, "overlap": cmd_overlap}[a.cmd](a)
