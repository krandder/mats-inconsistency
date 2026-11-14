"""P9 helpers: frozen P7 generator, vectorised BFS, rank statistics, intervals, checkpoints.

No scipy on this host; every rank statistic here uses tie-averaged ranks (P7/common.spearman
breaks ties by array order and must not be used: ~80% of cells tie at 0).
"""
import hashlib
import inspect
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import p7maze_snapshot as _snap

_src = inspect.getsource(_snap.bfs) + "\n\n" + inspect.getsource(_snap.gen_mazes)
assert hashlib.sha256(_src.encode()).hexdigest() == _snap.FUNC_SHA256, "P7 generator snapshot changed"
bfs = _snap.bfs
gen_mazes = _snap.gen_mazes
INF = 10 ** 6

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833}  # one-sided 95% = two-sided 90%


# ------------------------------------------------------------------ vectorised grid BFS (torch, bool)
def _nbr(r):
    """4-neighbour OR of a (B,n,n) bool tensor."""
    o = torch.zeros_like(r)
    o[:, 1:] |= r[:, :-1]
    o[:, :-1] |= r[:, 1:]
    o[:, :, 1:] |= r[:, :, :-1]
    o[:, :, :-1] |= r[:, :, 1:]
    return o


def reach(open_, src):
    """open_, src (B,n,n) bool -> cells 4-connected to src through open cells (src must be open)."""
    r = src & open_
    while True:
        r2 = (r | _nbr(r)) & open_
        if torch.equal(r2, r):
            return r
        r = r2


def dist(open_, src):
    """BFS distance (int64, INF where unreachable) from src cells through open cells."""
    r = src & open_
    d = torch.full(r.shape, INF, dtype=torch.int64)
    d[r] = 0
    k = 0
    while True:
        k += 1
        r2 = (r | _nbr(r)) & open_
        new = r2 & ~r
        if not bool(new.any()):
            return d
        d[new] = k
        r = r2


def onpath_mask(open_, s, g):
    """on some shortest s->g path; all-zero if disconnected (P7's formula would mark s alone)."""
    ds, dg = dist(open_, s), dist(open_, g)
    dsg = (ds * g.long()).sum((1, 2))  # g one-hot; INF if g unreachable
    ok = (dsg < INF)[:, None, None]
    return ((ds + dg) == dsg[:, None, None]) & (ds < INF) & ok, ds, dg, dsg


# ------------------------------------------------------------------ rank statistics
def rankdata(a):
    """Tie-averaged ranks (1-based), like scipy.stats.rankdata(method='average')."""
    a = np.asarray(a, dtype=np.float64)
    o = np.argsort(a, kind="mergesort")
    s = a[o]
    r = np.empty(len(a))
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def auroc(score, label):
    """Mann-Whitney AUROC with tie-averaged ranks; label bool."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kendall_tau_b(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    den = math.sqrt((conc + disc + ta) * (conc + disc + tb))
    return float((conc - disc) / den) if den > 0 else float("nan")


# ------------------------------------------------------------------ intervals
def tci(xs, level=0.95):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    n = len(xs)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "lo": None, "hi": None, "n": 1, "sd": None}
    sd = float(np.std(xs, ddof=1))
    t = (T975 if level == 0.95 else T95).get(n - 1, 1.96 if level == 0.95 else 1.645)
    h = t * sd / math.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n, "sd": sd}


def binom_ci(k, n):
    """Wilson 95%."""
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "n": 0}
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"p": p, "lo": c - h, "hi": c + h, "n": n}


# ------------------------------------------------------------------ io
def dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_js)
    os.replace(tmp, path)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.tolist()
    raise TypeError(type(o))


def save_partial(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_partial(path):
    if os.path.exists(path):
        return torch.load(path, map_location="cpu", weights_only=False)
    return None


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self):
        return round(time.time() - self.t0, 1)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
