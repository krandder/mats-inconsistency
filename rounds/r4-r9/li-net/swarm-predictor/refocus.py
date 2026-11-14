"""Refocus: which statements each round's chains are asked about. Hard-coded rules, not learned.

Also the shared board factory (the same boards as smoke_pool.py: harness preset plus K_SHOTS
noiseless shots at random unprobed cells), so run_swarm.py and analyze.py rebuild identical boards
from a seed.

Rules (PROPOSAL.md section 3):
  round 1     the open cells next to any reported hit (probe or shot), padded to 12 statements with
              random open statements (a random 12 if there are more);
  rounds 2-3  the 12 claimed open statements with the largest residual: sum over claims of
              conf * (logit q - logit price)^2 / sum of conf, i.e. the confidence-weighted spread of
              the claims around the current price; padded with random unclaimed open statements;
  ablation    round-robin over the open statements within 2 hops (Manhattan) of a hit cell, plus the
              row statements of rows that hold a hit, 12 per round in a fixed order ("the query" =
              the set of hit cells; the whole board if there is no hit);
  assign      each chain gets a random 6 of the round's targets, in random order.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from pool import Board, Claims, logit
from env.shiphunt import EnvConfig, ShipHunt

K_SHOTS = 4
K_TARGETS = 12
K_CLAIMS = 6


def make_board(cfg: EnvConfig, seed: int, k_shots: int = K_SHOTS) -> ShipHunt:
    env = ShipHunt(cfg, seed)
    rng = np.random.default_rng([seed, 11])
    probed = {c for c, _ in env.probes}
    cand = [c for c in range(cfg.n_cells) if c not in probed]
    for c in rng.choice(cand, size=k_shots, replace=False):
        env.fire(int(c))
    return env


def open_mask(board: Board, env: ShipHunt) -> np.ndarray:
    """Statements that are not already revealed (statement index of cell c is c)."""
    m = np.ones(board.J, bool)
    m[[c for c, _ in env.fired]] = False
    return m


def hit_cells(env: ShipHunt) -> List[int]:
    return sorted({c for c, h in env.probes if h} | {c for c, h in env.fired if h})


def _rc(cfg: EnvConfig, c: int):
    return divmod(c, cfg.W)


def neighbours(cfg: EnvConfig, c: int) -> List[int]:
    r, k = _rc(cfg, c)
    out = []
    for dr, dk in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, kk = r + dr, k + dk
        if 0 <= rr < cfg.H and 0 <= kk < cfg.W:
            out.append(rr * cfg.W + kk)
    return out


def stmt_row(board: Board, j: int) -> int:
    """Row of a statement: the cell's row, or r of 'H{L}:r{r}'."""
    n = board.names[j]
    if n.startswith("occ:"):
        return j // board.cfg.W
    return int(n.split(":r")[1])


def _pad(board, env, chosen: List[int], rng, k: int) -> List[int]:
    om = open_mask(board, env)
    rest = [int(j) for j in np.flatnonzero(om) if int(j) not in chosen]
    need = max(0, min(k, len(chosen) + len(rest)) - len(chosen))
    if need:
        chosen = chosen + [int(j) for j in rng.choice(rest, size=need, replace=False)]
    return chosen


def round1_targets(board: Board, env: ShipHunt, rng: np.random.Generator, k: int = K_TARGETS) -> List[int]:
    om = open_mask(board, env)
    adj = sorted({n for h in hit_cells(env) for n in neighbours(board.cfg, h) if om[n]})
    if len(adj) > k:
        adj = sorted(int(j) for j in rng.choice(adj, size=k, replace=False))
    return _pad(board, env, [int(j) for j in adj], rng, k)


def residuals(board: Board, cl: Claims, prices: np.ndarray) -> np.ndarray:
    """Per statement: confidence-weighted mean squared log-odds gap between claims and price (nan if unclaimed)."""
    if len(cl.q) == 0:
        return np.full(board.J, np.nan)
    gap2 = (logit(cl.q) - logit(prices[cl.stmt])) ** 2
    num = np.bincount(cl.stmt, weights=cl.w * gap2, minlength=board.J)
    den = np.bincount(cl.stmt, weights=cl.w, minlength=board.J)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def residual_targets(board: Board, env: ShipHunt, cl: Claims, prices: np.ndarray, rng: np.random.Generator,
                     k: int = K_TARGETS) -> List[int]:
    om = open_mask(board, env)
    r = residuals(board, cl, prices)
    claimed = [int(j) for j in np.flatnonzero(om) if not np.isnan(r[j])]
    top = sorted(claimed, key=lambda j: (-r[j], j))[:k]
    return _pad(board, env, top, rng, k)


def near_statements(board: Board, env: ShipHunt, hops: int = 2) -> List[int]:
    om = open_mask(board, env)
    hits = hit_cells(env)
    if not hits:
        return [int(j) for j in np.flatnonzero(om)]
    rows = {_rc(board.cfg, h)[0] for h in hits}
    out = []
    for j in np.flatnonzero(om):
        j = int(j)
        if board.names[j].startswith("occ:"):
            r, c = _rc(board.cfg, j)
            if min(abs(r - hr) + abs(c - hc) for hr, hc in (_rc(board.cfg, h) for h in hits)) <= hops:
                out.append(j)
        elif stmt_row(board, j) in rows:
            out.append(j)
    return out


def roundrobin_targets(board: Board, env: ShipHunt, rnd: int, k: int = K_TARGETS, hops: int = 2) -> List[int]:
    near = near_statements(board, env, hops)
    if len(near) <= k:
        return near
    start = (rnd * k) % len(near)
    return [near[(start + i) % len(near)] for i in range(k)]


def assign(targets: Sequence[int], n: int, rng: np.random.Generator, k: int = K_CLAIMS) -> List[List[int]]:
    kk = min(k, len(targets))
    return [[int(j) for j in rng.choice(list(targets), size=kk, replace=False)] for _ in range(n)]
