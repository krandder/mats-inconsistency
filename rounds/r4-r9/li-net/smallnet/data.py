"""ShipHunt-hard datasets for the smallnet experiment (eurisko, 2026-10-10).

Imports the harness env unchanged (env/shiphunt.py, preset 'hard': 5x5, ships 3,2,2, 6 probes, eps 0.2).
One episode = one board + ONE observed probe trajectory (6 distinct random cells, each report flipped
with prob eps). Stages = prefixes of that trajectory after 0, 2, 4, 6 probes.

Training arrays carry only what an episode shows: the true board and the observed (cell, report) list.
The exact posterior (enumeration over the 11088 placements) is computed only for VALIDATION and TEST
boards (evaluation), and for training boards only for the evaluation-only reference arm 'refpost'.

Seed ranges (disjoint): train seed s, board i -> 1_000_000 + 100_000*s + i (i < 10_000, nested prefixes);
validation board i -> 7_000_000 + i; test board i -> 9_000_000 + i.
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
W = CFG.W
NC = CFG.n_cells          # 25
NPROBE = CFG.n_probes     # 6
STAGES = (0, 2, 4, 6)
TRAIN_BASE, VAL_BASE, TEST_BASE = 1_000_000, 7_000_000, 9_000_000
DATA_DIR = os.environ.get("SMALLNET_DATA", "/tmp/smallnet/data")


def seeds_for(split: str, n: int, seed: int = 0) -> np.ndarray:
    if split == "train":
        return TRAIN_BASE + 100_000 * seed + np.arange(n)
    if split == "val":
        return VAL_BASE + np.arange(n)
    if split == "test":
        return TEST_BASE + np.arange(n)
    raise ValueError(split)


def episodes(seeds) -> dict:
    """occ uint8 [N,25]; pc int64 [N,6] probe cells in trajectory order; pr uint8 [N,6] reported hit (1) / miss (0)."""
    seeds = np.asarray(seeds)
    occ = np.zeros((len(seeds), NC), np.uint8)
    pc = np.zeros((len(seeds), NPROBE), np.int64)
    pr = np.zeros((len(seeds), NPROBE), np.uint8)
    for i, s in enumerate(seeds):
        e = ShipHunt(CFG, int(s))
        occ[i] = e.occ
        for k, (c, h) in enumerate(e.probes):
            pc[i, k] = c
            pr[i, k] = h
    return {"occ": occ, "pc": pc, "pr": pr, "seeds": seeds}


def exact_posteriors(seeds, stages=STAGES):
    """Exact marginals [N,len(stages),25] and log P*(true placement | evidence) [N,len(stages)], using the
    harness oracle (ShipHunt.posterior / bayes_cell_probs) on the probe prefix of each stage. EVALUATION ONLY."""
    P = placements(CFG).astype(np.float64)
    seeds = np.asarray(seeds)
    marg = np.zeros((len(seeds), len(stages), NC), np.float64)
    ltrue = np.zeros((len(seeds), len(stages)), np.float64)
    for i, s in enumerate(seeds):
        e = ShipHunt(CFG, int(s))
        full = list(e.probes)
        tidx = int(np.flatnonzero((P == e.occ.astype(np.float64)).all(1))[0])
        for k, st in enumerate(stages):
            e.probes = full[:st]
            w = e.posterior()
            marg[i, k] = w @ P
            ltrue[i, k] = np.log(max(w[tidx], 1e-300))
        e.probes = full
    return marg, ltrue


def evidence_states(pc, pr, n_probes):
    """Per-cell evidence state after the first n_probes probes: 0 none, 1 reported miss, 2 reported hit. [N,25] int64."""
    N = pc.shape[0]
    ev = np.zeros((N, NC), np.int64)
    rows = np.arange(N)
    for k in range(n_probes):
        ev[rows, pc[:, k]] = 1 + pr[:, k].astype(np.int64)
    return ev


def d4_perms(w: int = W) -> np.ndarray:
    """The 8 symmetries of the square board as cell permutations: new[i] = old[perm[i]]."""
    idx = np.arange(w * w).reshape(w, w)
    out = []
    for k in range(4):
        r = np.rot90(idx, k)
        out.append(r.reshape(-1))
        out.append(np.fliplr(r).reshape(-1))
    perms = np.stack(out)
    assert len({tuple(p) for p in perms}) == 8
    return perms


SYM = d4_perms()


def load_split(split: str, n: int, seed: int = 0, with_post: bool = False) -> dict:
    """Cached episodes (+ exact posteriors when with_post) as numpy arrays."""
    os.makedirs(DATA_DIR, exist_ok=True)
    tag = f"{split}-n{n}" + (f"-s{seed}" if split == "train" else "")
    path = os.path.join(DATA_DIR, tag + (".post" if with_post else "") + ".npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    # nested prefixes: reuse a bigger cached file if present
    seeds = seeds_for(split, n, seed)
    d = episodes(seeds)
    if with_post:
        d["post"], d["ltrue"] = exact_posteriors(seeds)
    tmp = path + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **d)
    os.replace(tmp, path)
    return d


def prior_marginals() -> np.ndarray:
    return placements(CFG).astype(np.float64).mean(0)


if __name__ == "__main__":
    import time
    t = time.time()
    v = load_split("val", 3000, with_post=True)
    print("val", {k: a.shape for k, a in v.items()}, f"{time.time()-t:.1f}s")
    t = time.time()
    te = load_split("test", 5000, with_post=True)
    print("test", {k: a.shape for k, a in te.items()}, f"{time.time()-t:.1f}s")
    for s in range(5):
        t = time.time()
        tr = load_split("train", 10_000, seed=s)
        print("train s", s, tr["occ"].shape, f"{time.time()-t:.1f}s")
