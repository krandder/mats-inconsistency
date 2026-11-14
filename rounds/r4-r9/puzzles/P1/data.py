"""P1 data: T2m (variable plateau + irreducible instances), prefix parity (T1 smoke),
and mazes (easy-to-hard-data format, Bansal et al. 2022).

T2m (--task t2m): bits x, marker channel at positions a and b, blocker channel.
  d ~ U[L/8, L-1], a ~ U[0, L-1-d], b = a + d.
  resolvable (50%): y_i = x_a XOR x_b at every position.
  blocked (50%): one blocker strictly between a and b; y_i = x_a XOR x_b XOR z,
  z ~ Bern(0.5) not in the input, so the Bayes belief is 0.5 forever.
Input tensor: (B, 5, L) float = [onehot(x) (3 ch, PAD unused), marker, blocker].
"""
import math
import os
import random
import numpy as np
import torch

MAZE_SPLITS = {  # size: (dev, pool, eval), disjoint, in this order within the file
    13: (200, 200, 600), 19: (0, 200, 600), 25: (0, 200, 600), 33: (200, 200, 600),
    59: (100, 400, 500), 99: (50, 100, 200),
}


def t_cap_maze(n):
    return max(100, 12 * n)


# ---------------------------------------------------------------- T2m
def t2m_batch(n, L, gen, device="cpu"):
    """Returns xf (n,5,L), y (n,L) float, mask (n,L), meta dict of (n,) tensors."""
    x = torch.randint(0, 2, (n, L), generator=gen)
    lo = max(2, L // 8)
    d = torch.randint(lo, L, (n,), generator=gen)                      # U[L/8, L-1]
    a = (torch.rand(n, generator=gen) * (L - d).float()).long()        # U[0, L-1-d]
    b = a + d
    blocked = torch.rand(n, generator=gen) < 0.5
    # blocker strictly between a and b: a+1 .. b-1
    c = a + 1 + (torch.rand(n, generator=gen) * (d - 1).float()).long()
    z = torch.randint(0, 2, (n,), generator=gen)
    idx = torch.arange(n)
    xa, xb = x[idx, a], x[idx, b]
    ybit = xa ^ xb ^ (z * blocked.long())
    y = ybit.view(n, 1).expand(n, L).float()
    xf = torch.zeros(n, 5, L)
    xf[:, 0:3] = torch.nn.functional.one_hot(x, 3).float().transpose(1, 2)
    xf[idx, 3, a] = 1.0
    xf[idx, 3, b] = 1.0
    xf[idx[blocked], 4, c[blocked]] = 1.0
    mask = torch.ones(n, L)
    meta = {"resolvable": (~blocked).long(), "d": d, "a": a, "b": b}
    return xf.to(device), y.to(device), mask.to(device), {k: v.to(device) for k, v in meta.items()}


def t2m_train_batch(n, gen, device="cpu"):
    L = 64 if torch.rand(1, generator=gen).item() < 0.5 else 128
    return (L,) + t2m_batch(n, L, gen, device)


def t2m_curr_batch(n, it, n_curr, gen, device="cpu"):
    """Warm-up curriculum (deviation 9): short L first, d from 1. Without it the backbone stays at chance
    (diagnostic: plain L=16 and d-ramp at L=32 both flat for 1.3-3k iterations; d-ramp at L=16 learns by ~700).
    Stages by fraction of n_curr: [0, .35) L=16 with d_max ramping 2 -> 15; [.35, .55) L=32;
    [.55, .8) L=64; [.8, 1) L in {64, 128}; d ~ U[1, d_max] (blocked only when d >= 2)."""
    f = it / max(1, n_curr)
    if f < 0.35:
        L, dmax = 16, int(round(2 + 13 * min(1.0, f / 0.25)))
    elif f < 0.55:
        L, dmax = 32, 31
    elif f < 0.8:
        L, dmax = 64, 63
    else:
        L = 64 if torch.rand(1, generator=gen).item() < 0.5 else 128
        dmax = L - 1
    x = torch.randint(0, 2, (n, L), generator=gen)
    d = torch.randint(1, dmax + 1, (n,), generator=gen)
    a = (torch.rand(n, generator=gen) * (L - d).float()).long()
    b = a + d
    blocked = (torch.rand(n, generator=gen) < 0.5) & (d >= 2)
    c = a + 1 + (torch.rand(n, generator=gen) * (d - 1).clamp(min=1).float()).long()
    z = torch.randint(0, 2, (n,), generator=gen)
    idx = torch.arange(n)
    ybit = x[idx, a] ^ x[idx, b] ^ (z * blocked.long())
    y = ybit.view(n, 1).expand(n, L).float()
    xf = torch.zeros(n, 5, L)
    xf[:, 0:3] = torch.nn.functional.one_hot(x, 3).float().transpose(1, 2)
    xf[idx, 3, a] = 1.0
    xf[idx, 3, b] = 1.0
    xf[idx[blocked], 4, c[blocked]] = 1.0
    return L, xf.to(device), y.to(device), torch.ones(n, L, device=device), None


def t2m_dbatch(n, L, dmin, dmax, gen, device="cpu"):
    """T2m batch with d ~ U[dmin, dmax] (blocked only when d >= 2); same layout as t2m_batch."""
    x = torch.randint(0, 2, (n, L), generator=gen)
    d = torch.randint(dmin, dmax + 1, (n,), generator=gen)
    a = (torch.rand(n, generator=gen) * (L - d).float()).long()
    b = a + d
    blocked = (torch.rand(n, generator=gen) < 0.5) & (d >= 2)
    c = a + 1 + (torch.rand(n, generator=gen) * (d - 1).clamp(min=1).float()).long()
    z = torch.randint(0, 2, (n,), generator=gen)
    idx = torch.arange(n)
    ybit = x[idx, a] ^ x[idx, b] ^ (z * blocked.long())
    y = ybit.view(n, 1).expand(n, L).float()
    xf = torch.zeros(n, 5, L)
    xf[:, 0:3] = torch.nn.functional.one_hot(x, 3).float().transpose(1, 2)
    xf[idx, 3, a] = 1.0
    xf[idx, 3, b] = 1.0
    xf[idx[blocked], 4, c[blocked]] = 1.0
    return xf.to(device), y.to(device), torch.ones(n, L, device=device)


# recipe v2 (deviation 9b): gradual L, d from 1 until SPEC_FROM, then the spec distribution
V2_STAGES = [(1000, 16), (1600, 32), (2200, 48), (3000, 64), (3600, 96)]
V2_SPEC_FROM = 6000


def t2m_curr2_batch(n, it, gen, device="cpu"):
    for end, L in V2_STAGES:
        if it < end:
            dmax = int(round(2 + 13 * min(1.0, it / 700))) if L == 16 else L - 1
            return (L,) + t2m_dbatch(n, L, 1, dmax, gen, device) + (None,)
    if it < V2_SPEC_FROM:
        L = 64 if torch.rand(1, generator=gen).item() < 0.5 else 128
        return (L,) + t2m_dbatch(n, L, 1, L - 1, gen, device) + (None,)
    return t2m_train_batch(n, gen, device)


def v2_lr_factor(it, iters):
    """x base lr (1e-3): warm-up 100, 1.0 while L <= 32, then 0.3 with a cosine to 0.3*0.03."""
    if it < 100:
        return (it + 1) / 100
    if it < 1600:
        return 1.0
    f = (it - 1600) / max(1, iters - 1600)
    return 0.3 * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * f)))


def prefix_as_float(x, nin=3):
    """Prefix-parity int input -> one-hot float (pilot backbone takes ints; kept for API)."""
    return torch.nn.functional.one_hot(x, nin).float().transpose(1, 2)


# ---------------------------------------------------------------- mazes
def maze_dir(root, n, split="test"):
    return os.path.join(root, f"maze_data_{split}_{n}")


def load_maze_arrays(root, n):
    d = maze_dir(root, n)
    x = np.load(os.path.join(d, "inputs.npy"), mmap_mode="r")
    y = np.load(os.path.join(d, "solutions.npy"), mmap_mode="r")
    return x, y


def split_indices(n, which):
    dev, pool, ev = MAZE_SPLITS[n]
    off = {"dev": (0, dev), "pool": (dev, dev + pool), "eval": (dev + pool, dev + pool + ev)}[which]
    return np.arange(off[0], off[1])


class LabelledMazes:
    def __init__(self, root, n, which, limit=None):
        self.x, self.y = load_maze_arrays(root, n)
        self.idx = split_indices(n, which)
        if limit:
            self.idx = self.idx[:limit]
        self.n = n

    def __len__(self):
        return len(self.idx)

    def batches(self, bs):
        for s in range(0, len(self.idx), bs):
            ii = self.idx[s:s + bs]
            yield (torch.from_numpy(np.asarray(self.x[ii], dtype=np.float32)),
                   torch.from_numpy(np.asarray(self.y[ii], dtype=np.float32)), ii)


class UnlabelledMazes:
    """Test-time pool: yields inputs only. y is never read (the array is not even opened)."""
    def __init__(self, root, n, limit=None):
        d = maze_dir(root, n)
        self.x = np.load(os.path.join(d, "inputs.npy"), mmap_mode="r")
        self.idx = split_indices(n, "pool")
        if limit:
            self.idx = self.idx[:limit]
        self.n = n

    def __len__(self):
        return len(self.idx)

    def batches(self, bs):
        for s in range(0, len(self.idx), bs):
            ii = self.idx[s:s + bs]
            y = None
            assert y is None
            yield torch.from_numpy(np.asarray(self.x[ii], dtype=np.float32)), y, ii


def gen_mazes(num, n, seed):
    """Generate num mazes of image size for n x n (cells (n+1)/2) with the repo's make_mazes.py.
    Returns inputs (num,3,H,W) uint8 and solutions (num,H,W) uint8."""
    import make_mazes_upstream as mm
    random.seed(seed)
    np.random.seed(seed)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        inp, _, _, sol = mm.gen_dataset(num, (n + 1) // 2)
    return (inp.transpose(0, 3, 1, 2).astype(np.uint8), sol[..., 0].astype(np.uint8))


def _gen_chunk(args):
    num, n, seed = args
    return gen_mazes(num, n, seed)


def write_generated(root, n, num, seed0, procs=4, chunk=25):
    """Deterministic parallel generation: chunk k uses seed seed0 + k."""
    from multiprocessing import Pool
    d = maze_dir(root, n)
    if os.path.exists(os.path.join(d, "solutions.npy")):
        return
    os.makedirs(d, exist_ok=True)
    jobs = [(min(chunk, num - s), n, seed0 + k) for k, s in enumerate(range(0, num, chunk))]
    with Pool(procs) as p:
        outs = p.map(_gen_chunk, jobs)
    x = np.concatenate([o[0] for o in outs])
    y = np.concatenate([o[1] for o in outs])
    np.save(os.path.join(d, "inputs.npy"), x)
    np.save(os.path.join(d, "solutions.npy"), y)


def fresh_train_mazes(num, seed, n=9):
    x, y = gen_mazes(num, n, seed)
    return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
