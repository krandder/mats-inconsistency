"""T2 data for wm2: ShipHunt-hard boards with board-dependent probe selection (pre-registered in SPEC-REVIEW C.13).

Board: the harness board for the env seed (ShipHunt(CFG, seed).occ, unchanged harness; same boards as T1).
Probes (separate RNG stream default_rng([seed, 20261011])): for t = 1..6, U = unprobed cells, A = cells of U that are
occupied or edge-adjacent to an occupied cell. With prob Q = 0.7 the cell is uniform on A (uniform on U if A is empty),
else uniform on U. The reading flips with prob eps = 0.2 (as T1).
Exact posterior (EVALUATION ONLY): enumeration over the 11088 placements with the reading likelihood AND the selection
likelihood P(cell_t | board, earlier cells). Also stored: the selection-unaware posterior (T1 formula on T2 data).
Files use smallnet's data format (occ, pc, pr, seeds; post, ltrue for val/test) so round-1 code reads them unchanged:
  OUT/val-n3000.post.npz, OUT/test-n5000.post.npz, OUT/train-n10000-s{0..4}.npz
  python3 t2data.py [OUT=/tmp/wm2/data-t2]
"""
from __future__ import annotations

import os
import sys

import numpy as np

HARNESS = os.environ.get("LIR_HARNESS", "/home/agent/workspaces/eurisko-study/lir-agent/harness")
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)
from env.shiphunt import PRESETS, ShipHunt, placements  # noqa: E402

CFG = PRESETS["hard"]
W, NC, NPROBE, EPS = CFG.W, CFG.n_cells, CFG.n_probes, CFG.eps
Q = 0.7
STREAM = 20261011
STAGES = (0, 2, 4, 6)
TRAIN_BASE, VAL_BASE, TEST_BASE = 1_000_000, 7_000_000, 9_000_000


def near_mask(occ: np.ndarray) -> np.ndarray:
    """occ bool [..., 25] -> occupied or edge-adjacent to an occupied cell, bool [..., 25]."""
    M = occ.reshape(-1, W, W)
    D = M.copy()
    D[:, 1:, :] |= M[:, :-1, :]
    D[:, :-1, :] |= M[:, 1:, :]
    D[:, :, 1:] |= M[:, :, :-1]
    D[:, :, :-1] |= M[:, :, 1:]
    return D.reshape(occ.shape)


def episode(seed: int):
    occ = ShipHunt(CFG, int(seed)).occ.astype(bool)
    near = near_mask(occ[None])[0]
    rng = np.random.default_rng([int(seed), STREAM])
    U = np.ones(NC, bool)
    pc = np.zeros(NPROBE, np.int64)
    pr = np.zeros(NPROBE, np.uint8)
    for t in range(NPROBE):
        A = near & U
        if rng.random() < Q and A.any():
            c = int(rng.choice(np.flatnonzero(A)))
        else:
            c = int(rng.choice(np.flatnonzero(U)))
        r = bool(occ[c]) ^ bool(rng.random() < EPS)
        pc[t], pr[t] = c, r
        U[c] = False
    return occ.astype(np.uint8), pc, pr


def episodes(seeds):
    seeds = np.asarray(seeds)
    occ = np.zeros((len(seeds), NC), np.uint8)
    pc = np.zeros((len(seeds), NPROBE), np.int64)
    pr = np.zeros((len(seeds), NPROBE), np.uint8)
    for i, s in enumerate(seeds):
        occ[i], pc[i], pr[i] = episode(s)
    return {"occ": occ, "pc": pc, "pr": pr, "seeds": seeds}


_P = None
_NM = None


def _tabs():
    global _P, _NM
    if _P is None:
        _P = placements(CFG).astype(bool)
        _NM = near_mask(_P)
    return _P, _NM


def selection_logp(pc_row, P=None, NM=None):
    """log P(cell_t | board, earlier cells) for every placement: [11088, 6]. EVALUATION ONLY."""
    if P is None:
        P, NM = _tabs()
    U = np.ones(NC, bool)
    out = np.zeros((len(P), NPROBE))
    for t in range(NPROBE):
        c = int(pc_row[t])
        nA = (NM & U).sum(1)
        inA = NM[:, c]
        nU = U.sum()
        p = np.where(nA > 0, Q * inA / np.maximum(nA, 1) + (1 - Q) / nU, 1.0 / nU)
        out[:, t] = np.log(p)
        U[c] = False
    return out


def exact_posteriors(d, aware=True):
    """Exact marginals [N,4,25] and log P*(true placement | evidence) [N,4]. EVALUATION ONLY."""
    P, NM = _tabs()
    Pf = P.astype(np.float64)
    N = len(d["occ"])
    marg = np.zeros((N, len(STAGES), NC))
    ltrue = np.zeros((N, len(STAGES)))
    for i in range(N):
        tidx = int(np.flatnonzero((P == d["occ"][i].astype(bool)).all(1))[0])
        lread = np.where(P[:, d["pc"][i]] == d["pr"][i].astype(bool)[None, :], np.log(1 - EPS), np.log(EPS))  # [11088,6]
        lsel = selection_logp(d["pc"][i], P, NM) if aware else np.zeros_like(lread)
        cum = np.concatenate([np.zeros((len(P), 1)), np.cumsum(lread + lsel, 1)], 1)  # [11088,7]
        for k, st in enumerate(STAGES):
            lw = cum[:, st]
            w = np.exp(lw - lw.max())
            w /= w.sum()
            marg[i, k] = w @ Pf
            ltrue[i, k] = np.log(max(w[tidx], 1e-300))
    return marg, ltrue


def seeds_for(split, n, seed=0):
    if split == "train":
        return TRAIN_BASE + 100_000 * seed + np.arange(n)
    return {"val": VAL_BASE, "test": TEST_BASE}[split] + np.arange(n)


def build(out):
    import time
    os.makedirs(out, exist_ok=True)
    for split, n in (("val", 3000), ("test", 5000)):
        t = time.time()
        d = episodes(seeds_for(split, n))
        d["post"], d["ltrue"] = exact_posteriors(d, aware=True)
        d["post_unaware"], d["ltrue_unaware"] = exact_posteriors(d, aware=False)
        p = os.path.join(out, f"{split}-n{n}.post.npz")
        np.savez_compressed(p + ".tmp.npz", **d)
        os.replace(p + ".tmp.npz", p)
        print(split, n, f"{time.time() - t:.1f}s", flush=True)
    for s in range(5):
        t = time.time()
        d = episodes(seeds_for("train", 10_000, s))
        p = os.path.join(out, f"train-n10000-s{s}.npz")
        np.savez_compressed(p + ".tmp.npz", **d)
        os.replace(p + ".tmp.npz", p)
        print("train s", s, f"{time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wm2/data-t2")
