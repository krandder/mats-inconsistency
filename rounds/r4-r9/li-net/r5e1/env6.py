"""R5-E1 environment: 6x6 coupled multi-ship grid (ShipHunt family) with an exact Bayes oracle.

Board: 3 ships of lengths 4, 3, 2 (straight, horizontal or vertical, no overlap) on 6x6 = 36 cells (row-major, cell = 6*row + col).
Root observation x_root: 6 noisy sonar point-probes at distinct uniformly drawn cells (each report flipped with prob EPS = 0.15)
and one sycophantic commander's note naming a cell: a uniformly drawn ship cell w.p. NOTE_REL = 0.30, else a uniformly drawn
water cell. Prior: uniform over the distinct occupancy patterns (the harness convention, lir-agent/harness/env/shiphunt.py).
Exact Bayes P*(Z | x_root) = softmax_p(log-lik) @ P, an enumeration over all placements: EVALUATION AND GATES ONLY (never a
training target, never used for selection).

Search geometry: 40 patches = 20 horizontal 2x3 + 20 vertical 3x2 rectangles. Halves: NW = rows 0-2 (cells 0..17), SE = rows
3-5 (cells 18..35). A patch belongs to a half when at least 3 of its 6 cells lie in it (the 4 horizontal row-2/3 patches lie
3/3 and belong to both halves).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

W = H = 6
N = 36
SHIPS = (4, 3, 2)
N_SHIP_CELLS = sum(SHIPS)
N_PROBES = 6
EPS = 0.15
NOTE_REL = 0.30
NW_CELLS = np.arange(18)
SE_CELLS = np.arange(18, 36)
LLR_PROBE = math.log((1 - EPS) / EPS)
# log P(note = n | n ship) - log P(note = n | n water) = log((0.30/9) / (0.70/27))
LLR_NOTE = math.log((NOTE_REL / N_SHIP_CELLS) / ((1 - NOTE_REL) / (N - N_SHIP_CELLS)))


def enum_placements() -> np.ndarray:
    """All distinct occupancy patterns of the three ships, bool [Npl, 36], sorted (same algorithm as the harness)."""
    single = []
    for L in SHIPS:
        opts = set()
        for r in range(H):
            for c in range(W):
                if c + L <= W:
                    opts.add(frozenset(r * W + c + k for k in range(L)))
                if L > 1 and r + L <= H:
                    opts.add(frozenset((r + k) * W + c for k in range(L)))
        single.append(sorted(opts, key=sorted))
    out = set()
    for combo in itertools.product(*single):
        cells = set()
        ok = True
        for s in combo:
            if cells & s:
                ok = False
                break
            cells |= s
        if ok:
            out.add(frozenset(cells))
    arr = np.zeros((len(out), N), dtype=bool)
    for i, cells in enumerate(sorted(out, key=sorted)):
        arr[i, list(cells)] = True
    return arr


_P = None


def placements() -> np.ndarray:
    global _P
    if _P is None:
        _P = enum_placements()
    return _P


def make_patches():
    P = []
    for r in range(H - 1):
        for c in range(W - 2):
            P.append([(r + a) * W + c + b for a in range(2) for b in range(3)])
    for r in range(H - 2):
        for c in range(W - 1):
            P.append([(r + a) * W + c + b for a in range(3) for b in range(2)])
    return np.array(P, dtype=np.int64)


PATCHES = make_patches()                                     # [40, 6]
NPATCH = len(PATCHES)
_nw = (PATCHES < 18).sum(1)
IN_NW = _nw >= 3                                             # [40] bool
IN_SE = (6 - _nw) >= 3
SECTOR = np.where(IN_NW & ~IN_SE, 0, np.where(IN_SE & ~IN_NW, 1, 2))   # 0 NW only, 1 SE only, 2 tie (both)
OVERLAP = np.array([[len(set(a) & set(b)) for b in PATCHES] for a in PATCHES], dtype=np.int64)   # [40, 40]
PATCH_MASK = np.zeros((NPATCH, N), dtype=bool)
for _i, _p in enumerate(PATCHES):
    PATCH_MASK[_i, _p] = True
LINES = np.array([[r * W + c for c in range(W)] for r in range(H)] + [[r * W + c for r in range(H)] for c in range(W)])


# ------------------------------------------------------------------------------------------------ boards (numpy, fixed splits)
def make_boards(stream: int, n: int) -> dict:
    """n boards from a fixed numpy stream (calibration / test / RL splits). Evidence ev: 0 none, 1 probe reported miss,
    2 probe reported hit."""
    P = placements()
    rng = np.random.default_rng([stream, 5501])
    occ = P[rng.integers(len(P), size=n)].copy()
    cells = rng.random((n, N)).argsort(1)[:, :N_PROBES]
    flip = rng.random((n, N_PROBES)) < EPS
    rep = np.take_along_axis(occ, cells, 1) ^ flip
    ev = np.zeros((n, N), dtype=np.int64)
    np.put_along_axis(ev, cells, 1 + rep.astype(np.int64), 1)
    true_note = rng.random(n) < NOTE_REL
    sc = rng.random((n, N))
    ship_pick = np.where(occ, sc, -1.0).argmax(1)
    water_pick = np.where(~occ, sc, -1.0).argmax(1)
    note = np.where(true_note, ship_pick, water_pick)
    return {"occ": occ, "ev": ev, "note": note.astype(np.int64), "note_true": occ[np.arange(n), note]}


def boards_to_torch(b: dict, dev) -> dict:
    return {k: torch.as_tensor(v, device=dev) for k, v in b.items()}


def torch_boards(B: int, gen: torch.Generator, Pt: torch.Tensor) -> dict:
    """Same generative process on the device (training streams). Pt: bool placements [Npl, 36] on the device."""
    dev = Pt.device
    occ = Pt[torch.randint(Pt.shape[0], (B,), device=dev, generator=gen)]
    cells = torch.rand(B, N, device=dev, generator=gen).argsort(1)[:, :N_PROBES]
    flip = torch.rand(B, N_PROBES, device=dev, generator=gen) < EPS
    rep = occ.gather(1, cells) ^ flip
    ev = torch.zeros(B, N, dtype=torch.long, device=dev)
    ev.scatter_(1, cells, 1 + rep.long())
    true_note = torch.rand(B, device=dev, generator=gen) < NOTE_REL
    sc = torch.rand(B, N, device=dev, generator=gen)
    ship_pick = torch.where(occ, sc, torch.full_like(sc, -1.0)).argmax(1)
    water_pick = torch.where(~occ, sc, torch.full_like(sc, -1.0)).argmax(1)
    note = torch.where(true_note, ship_pick, water_pick)
    return {"occ": occ, "ev": ev, "note": note, "note_true": occ.gather(1, note[:, None]).squeeze(1)}


# ------------------------------------------------------------------------------------------------ exact Bayes (EVALUATION ONLY)
def loglik_vec(ev: torch.Tensor, note: torch.Tensor) -> torch.Tensor:
    """Per-board vector v [B, 36] with log P(x_root | placement p) = const_b + P[p] . v (ships never overlap, so every
    placement has exactly 9 ship cells and 27 water cells; the note likelihood is 0.30/9 on a ship cell, 0.70/27 on water)."""
    v = torch.zeros(ev.shape, dtype=torch.float64, device=ev.device)
    v = torch.where(ev == 2, torch.full_like(v, LLR_PROBE), v)
    v = torch.where(ev == 1, torch.full_like(v, -LLR_PROBE), v)
    v.scatter_add_(1, note[:, None], torch.full((ev.shape[0], 1), LLR_NOTE, dtype=torch.float64, device=ev.device))
    return v


@torch.no_grad()
def bayes_marginals(ev: torch.Tensor, note: torch.Tensor, Pf: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Exact posterior P*(ship on cell | x_root), float64 [B, 36]. Pf: float64 placements [Npl, 36]. EVALUATION ONLY."""
    out = []
    for a in range(0, ev.shape[0], chunk):
        v = loglik_vec(ev[a:a + chunk], note[a:a + chunk])
        lw = v @ Pf.T
        out.append(torch.softmax(lw, 1) @ Pf)
    return torch.cat(out)


def bayes_bruteforce(ev_row: np.ndarray, note: int) -> np.ndarray:
    """Loop implementation in the harness style (posterior weights per placement), for the unit test only."""
    P = placements()
    logw = np.zeros(len(P))
    for c in range(N):
        if ev_row[c] == 0:
            continue
        h = ev_row[c] == 2
        logw += np.where(P[:, c] == h, np.log(1 - EPS), np.log(EPS))
    k = P.sum(1)
    on = P[:, note]
    logw += np.log(np.where(on, NOTE_REL / k, (1 - NOTE_REL) / (N - k)))
    logw -= logw.max()
    w = np.exp(logw)
    w /= w.sum()
    return w @ P.astype(np.float64)


# ------------------------------------------------------------------------------------------------ training reveal sets
_PATCH_T = {}
_LINE_T = {}


def reveal_masks(B: int, gen: torch.Generator, dev) -> torch.Tensor:
    """Reveal sets S for any-order training of theta_real (true values are shown on S):
    15% root (S empty); 35% uniform random subset with |S| ~ U{1..35}; 35% local 2x3 / 3x2 patch transitions (union of
    m ~ U{1..4} random patches, 4 random cells of each revealed); 15% row/column line scans (m ~ U{1..2} full lines)."""
    if dev not in _PATCH_T:
        _PATCH_T[dev] = torch.as_tensor(PATCHES, device=dev)
        _LINE_T[dev] = torch.as_tensor(LINES, device=dev)
    pt, lt = _PATCH_T[dev], _LINE_T[dev]
    mode = torch.rand(B, device=dev, generator=gen)
    rev = torch.zeros(B, N, dtype=torch.bool, device=dev)
    # uniform
    k = torch.randint(1, N, (B,), device=dev, generator=gen)
    rk = torch.rand(B, N, device=dev, generator=gen).argsort(1).argsort(1)
    uni = rk < k[:, None]
    # patches
    m = torch.randint(1, 5, (B,), device=dev, generator=gen)
    pr = torch.zeros(B, N, dtype=torch.bool, device=dev)
    for j in range(4):
        pid = torch.randint(NPATCH, (B,), device=dev, generator=gen)
        cells = pt[pid]                                                            # [B, 6]
        sel = torch.rand(B, 6, device=dev, generator=gen).argsort(1)[:, :4]
        c4 = cells.gather(1, sel)
        add = torch.zeros(B, N, dtype=torch.bool, device=dev)
        add.scatter_(1, c4, True)
        pr = pr | (add & (j < m)[:, None])
    # lines
    ml = torch.randint(1, 3, (B,), device=dev, generator=gen)
    lr_ = torch.zeros(B, N, dtype=torch.bool, device=dev)
    for j in range(2):
        lid = torch.randint(len(LINES), (B,), device=dev, generator=gen)
        add = torch.zeros(B, N, dtype=torch.bool, device=dev)
        add.scatter_(1, lt[lid], True)
        lr_ = lr_ | (add & (j < ml)[:, None])
    rev = torch.where((mode >= 0.15)[:, None] & (mode < 0.50)[:, None], uni, rev)
    rev = torch.where((mode >= 0.50)[:, None] & (mode < 0.85)[:, None], pr, rev)
    rev = torch.where((mode >= 0.85)[:, None], lr_, rev)
    return rev
