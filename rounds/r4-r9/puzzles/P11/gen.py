"""P11 maze data: seeded numpy DFS perfect mazes, rendered exactly like upstream deep-thinking.

Upstream naming: size 9 = 5 cells (24 px), 17 = 9 cells (40 px), 29 = 15 cells (64 px), 59 = 30 cells (124 px).
Grid = (2c+1)^2 array (open cells and knocked-down walls = 1). Image = each grid cell as a 2x2
block plus a 1 px border: side 2(2c+1)+2. Input channels: open, start, goal. Target: on-path
pixels from BFS (vendored P7 bfs). Maze RNG is a numpy Generator seeded per maze, independent of torch.
Seed ranges: train [0, 1e6), validation [1e6, 2e6), test [2e6, 3e6).
"""
import numpy as np

from vendored import bfs

SIZES = {9: 5, 17: 9, 29: 15, 59: 30}          # upstream size name -> cells per side
VAL_BASE, TEST_BASE = 1_000_000, 2_000_000


def side(size):
    c = SIZES[size]
    return 2 * (2 * c + 1) + 2


def dfs_grid(c, ix, iy, rng):
    """Depth-first perfect maze on c x c cells started at (ix, iy); same algorithm as upstream
    Maze.make_maze (neighbour order W, E, S, N; random choice; backtrack on dead end)."""
    n = 2 * c + 1
    A = np.zeros((n, n), np.uint8)
    vis = np.zeros((c, c), bool)
    x, y = ix, iy
    vis[x, y] = True
    A[2 * y + 1, 2 * x + 1] = 1
    stack = []
    nv = 1
    r = rng.random(4 * c * c)
    k = 0
    while nv < c * c:
        nb = []
        for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
            x2, y2 = x + dx, y + dy
            if 0 <= x2 < c and 0 <= y2 < c and not vis[x2, y2]:
                nb.append((dx, dy))
        if not nb:
            x, y = stack.pop()
            continue
        dx, dy = nb[int(r[k] * len(nb))]
        k += 1
        A[2 * y + 1 + dy, 2 * x + 1 + dx] = 1
        stack.append((x, y))
        x, y = x + dx, y + dy
        vis[x, y] = True
        A[2 * y + 1, 2 * x + 1] = 1
        nv += 1
    return A


def one_maze(c, seed):
    """-> grid (2c+1)^2 uint8 open, start (row,col), goal (row,col), onpath grid, near-endpoint distance grid, path length."""
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    s, g = rng.choice(c * c, 2, replace=False)
    ix, iy = int(s % c), int(s // c)
    ex, ey = int(g % c), int(g // c)
    A = dfs_grid(c, ix, iy, rng)
    return grid_targets(A, (2 * iy + 1, 2 * ix + 1), (2 * ey + 1, 2 * ex + 1))


def grid_targets(A, sp, gp):
    walls = A == 0
    ds = bfs(walls, sp)
    dg = bfs(walls, gp)
    L = int(ds[gp])
    onpath = ((ds + dg) == L) & (A > 0)
    near = np.minimum(ds, dg)
    return A, sp, gp, onpath.astype(np.uint8), near, L


def up2(a):
    """(..., n, n) -> (..., 2n+2, 2n+2): 2x2 blocks plus 1 px border (upstream rendering)."""
    b = np.repeat(np.repeat(a, 2, axis=-2), 2, axis=-1)
    pad = [(0, 0)] * (b.ndim - 2) + [(1, 1), (1, 1)]
    return np.pad(b, pad)


def render(A, sp, gp, onpath):
    n = A.shape[0]
    st = np.zeros((n, n), np.uint8)
    go = np.zeros((n, n), np.uint8)
    st[sp] = 1
    go[gp] = 1
    x = up2(np.stack([A, st, go]))
    y = up2(onpath)
    return x, y


def to_rgb(x):
    """our (open, start, goal) -> upstream RGB input (start = [0,1,0], goal = [1,0,0], open = white)."""
    o, s, g = x[0], x[1], x[2]
    return np.stack([o * (1 - s), o * (1 - g), o * (1 - s) * (1 - g)], -1)


def batch(size, seeds, meta=False):
    c = SIZES[size]
    S = side(size)
    X = np.zeros((len(seeds), 3, S, S), np.uint8)
    Y = np.zeros((len(seeds), S, S), np.uint8)
    NEAR = np.zeros((len(seeds), S, S), np.int32) if meta else None
    L = np.zeros(len(seeds), np.int32)
    for i, sd in enumerate(seeds):
        A, sp, gp, op, near, l = one_maze(c, sd)
        X[i], Y[i] = render(A, sp, gp, op)
        L[i] = l
        if meta:
            NEAR[i] = up2(np.where(A > 0, near, -1)).astype(np.int32)
    if meta:
        return X, Y, L, NEAR
    return X, Y


def eval_set(split, size, n):
    base = VAL_BASE if split == "val" else TEST_BASE
    seeds = [base + size * 10_000 + k for k in range(n)]
    X, Y, L, NEAR = batch(size, seeds, meta=True)
    return {"x": X, "y": Y, "len": L, "near": NEAR, "seeds": seeds, "size": size, "split": split}


def train_seeds(run_seed, step, bs):
    off = (run_seed * 104_729) % VAL_BASE
    return [(off + step * bs + j) % VAL_BASE for j in range(bs)]


# ------------------------------------------------------------ training pool (size 9) and GPU rendering
def _pool_part(args):
    lo, hi, c = args
    n = 2 * c + 1
    A = np.zeros((hi - lo, n, n), np.uint8)
    OP = np.zeros((hi - lo, n, n), np.uint8)
    SG = np.zeros((hi - lo, 4), np.int16)
    for i, sd in enumerate(range(lo, hi)):
        a, sp, gp, op, near, L = one_maze(c, sd)
        A[i], OP[i] = a, op
        SG[i] = (sp[0], sp[1], gp[0], gp[1])
    return A, OP, SG


def build_pool(path, n=400_000, size=9, procs=6):
    """Maze seeds [0, n) (inside the train range [0, 1e6)); every run draws a seeded permutation, so no
    maze repeats within a run."""
    from multiprocessing import Pool
    c = SIZES[size]
    parts = [(lo, min(lo + 10_000, n), c) for lo in range(0, n, 10_000)]
    with Pool(procs) as p:
        res = p.map(_pool_part, parts)
    A = np.concatenate([r[0] for r in res])
    OP = np.concatenate([r[1] for r in res])
    SG = np.concatenate([r[2] for r in res])
    np.savez_compressed(path, A=np.packbits(A, axis=-1), OP=np.packbits(OP, axis=-1), SG=SG, n=n, size=size,
                        grid=2 * c + 1)


def load_pool(path, dev):
    import torch
    z = np.load(path)
    g = int(z["grid"])
    A = np.unpackbits(z["A"], axis=-1)[..., :g]
    OP = np.unpackbits(z["OP"], axis=-1)[..., :g]
    return {"A": torch.tensor(A, device=dev), "OP": torch.tensor(OP, device=dev),
            "SG": torch.tensor(z["SG"].astype(np.int64), device=dev), "n": int(z["n"])}


def render_torch(pool, idx):
    """GPU render of pool mazes idx -> x (B,3,S,S) float (open, start, goal), y (B,S,S) float."""
    import torch
    import torch.nn.functional as F
    A = pool["A"][idx].float()
    B, n, _ = A.shape
    SG = pool["SG"][idx]
    ar = torch.arange(B, device=A.device)
    st = torch.zeros_like(A)
    go = torch.zeros_like(A)
    st[ar, SG[:, 0], SG[:, 1]] = 1
    go[ar, SG[:, 2], SG[:, 3]] = 1
    x = torch.stack([A, st, go], 1)
    x = F.pad(x.repeat_interleave(2, -1).repeat_interleave(2, -2), (1, 1, 1, 1))
    y = F.pad(pool["OP"][idx].float().repeat_interleave(2, -1).repeat_interleave(2, -2), (1, 1, 1, 1))
    return x, y


# ------------------------------------------------------------ S0 round-trip test vs upstream gen_dataset
def roundtrip_test(n=200, seed=0):
    """Upstream gen_dataset(maze_size=5) -> downsample to the (2c+1)^2 grid -> our render and our BFS
    on-path must equal upstream inputs and solutions bit for bit."""
    import random
    import vendored_upstream as U
    from vendored import upstream_render
    np.random.seed(seed)
    random.seed(seed)
    big_x, _, se, big_sol = U.gen_dataset(n, 5)
    ok_in = ok_sol = ok_render = 0
    for i in range(n):
        img = big_x[i]                      # (24, 24, 3) upstream RGB
        grid_rgb = img[1:-1:2, 1:-1:2, :]   # downsample: top-left pixel of each 2x2 block
        A = (grid_rgb.max(-1) > 0).astype(np.uint8)
        sp = tuple(int(v) for v in np.argwhere((grid_rgb[..., 1] == 1) & (grid_rgb[..., 0] == 0))[0])
        gp = tuple(int(v) for v in np.argwhere((grid_rgb[..., 0] == 1) & (grid_rgb[..., 1] == 0))[0])
        ix, iy, ex, ey = [int(v) for v in se[i]]
        assert sp == (2 * iy + 1, 2 * ix + 1) and gp == (2 * ey + 1, 2 * ex + 1)
        A2, sp2, gp2, op, near, L = grid_targets(A, sp, gp)
        x, y = render(A, sp, gp, op)
        ok_in += int(np.array_equal(to_rgb(x).astype(np.float64), img))
        ok_sol += int(np.array_equal(y.astype(np.float64), big_sol[i, :, :, 0]))
        # the vendored upstream rendering block applied to our grid gives the same image
        rx, ry = upstream_render(to_rgb(np.stack([A, x[1][1:-1:2, 1:-1:2], x[2][1:-1:2, 1:-1:2]]))[None],
                                 op[None, :, :, None], 5)
        ok_render += int(np.array_equal(rx[0], img) and np.array_equal(ry[0, :, :, 0], big_sol[i, :, :, 0]))
    return {"n": n, "inputs_equal": ok_in, "solutions_equal": ok_sol, "vendored_render_equal": ok_render,
            "pass": ok_in == n and ok_sol == n and ok_render == n}


def own_generator_checks(n=200):
    """Our DFS makes perfect mazes: open cells form a tree (edges = cells - 1), all connected, path unique."""
    bad = 0
    for size in (9, 17):
        c = SIZES[size]
        for k in range(n // 2):
            A, sp, gp, op, near, L = one_maze(c, VAL_BASE + 999_000 + size * 1000 + k)
            cells = c * c
            open_n = int(A.sum())
            edges = open_n - cells
            conn = bool((bfs(A == 0, sp)[A > 0] < 10 ** 6).all())
            bad += int(not (edges == cells - 1 and conn and op.sum() == L + 1))
    return {"n": n, "bad": bad, "pass": bad == 0}


if __name__ == "__main__":
    import json
    import time
    t = time.time()
    print(json.dumps(roundtrip_test()))
    print(json.dumps(own_generator_checks()))
    for size in (9, 17, 29, 59):
        t0 = time.time()
        X, Y = batch(size, train_seeds(0, 0, 64))
        print(size, X.shape, f"{(time.time() - t0) * 1000:.1f} ms / 64 mazes")
