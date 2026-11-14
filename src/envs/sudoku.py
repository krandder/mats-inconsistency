"""6x6 Sudoku & 9x9 `Sudoku-Extreme` Environment (`src/envs/sudoku.py`).

Unifies:
- `Sudoku-6x6` (2x3 boxes, 36 cells, digits 1..6, ambiguous multi-solution & unsolvable twin boards)
- `Sudoku-9x9-Extreme` (3x3 boxes, 81 cells, digits 1..9, 17..24 givens requiring multi-stall search,
  plus unsolvable twin boards with zero direct peer clash).

Uses the compiled C bitmask-DFS solver (`sud9.c` / `libsud9.so`) when available for sub-millisecond
9x9 enumeration and exact marginal computation, with a pure-Python bitmask-MRV solver for 6x6 (and 9x9 fallback).
The model never sees the rules; `check_witness` is a fast vectorized PyTorch row/col/box verifier on CPU/GPU.
"""

from __future__ import annotations

import ctypes
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import (
    EnvBatch,
    GameFamily,
    GameID,
    QueryType,
    UnifiedGameEnv,
    UtilitySpec,
)

_SUD9_C_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../rounds/r4-r9/puzzles/R9-E2/code/sud9.c")
)
_SUD9_SO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../rounds/r4-r9/puzzles/R9-E2/code/libsud9.so")
)
_LIB_SUD9 = None


def _get_libsud9():
    global _LIB_SUD9
    if _LIB_SUD9 is not None:
        return _LIB_SUD9
    if os.path.exists(_SUD9_C_PATH):
        if not os.path.exists(_SUD9_SO_PATH) or os.path.getmtime(_SUD9_SO_PATH) < os.path.getmtime(_SUD9_C_PATH):
            os.system(f"gcc -O2 -shared -fPIC -o {_SUD9_SO_PATH} {_SUD9_C_PATH}")
    if os.path.exists(_SUD9_SO_PATH):
        lib = ctypes.CDLL(_SUD9_SO_PATH)
        lib.enum9.restype = ctypes.c_longlong
        lib.enum9.argtypes = [
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.random_grid9.restype = ctypes.c_int
        lib.random_grid9.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
        _LIB_SUD9 = lib
    return _LIB_SUD9


def _sudoku_geometry(size: int) -> Tuple[int, int, List[int], List[List[int]], List[set]]:
    if size == 6:
        br, bc = 2, 3
    elif size == 9:
        br, bc = 3, 3
    else:
        raise ValueError(f"Unsupported Sudoku size {size}; expected 6 or 9.")
    n_cells = size * size
    box_of = [(i // size // br) * (size // bc) + (i % size) // bc for i in range(n_cells)]
    rows = [[r * size + c for c in range(size)] for r in range(size)]
    cols = [[r * size + c for r in range(size)] for c in range(size)]
    boxes = [[] for _ in range(size)]
    for i, b in enumerate(box_of):
        boxes[b].append(i)
    units = rows + cols + boxes
    peers = [set(j for u in units if i in u for j in u if j != i) for i in range(n_cells)]
    return br, bc, box_of, units, peers


def _solve_all_py(board: List[int], size: int, cap: int, box_of: List[int]) -> List[List[int]]:
    """Bitmask DFS with Minimum Remaining Values (MRV) for 6x6 (and 9x9 fallback)."""
    n_cells = size * size
    full = (1 << size) - 1
    R = [0] * size
    Cm = [0] * size
    B = [0] * size
    g = list(board)
    for i, d in enumerate(g):
        if d:
            b = 1 << (d - 1)
            r, c = divmod(i, size)
            if (R[r] | Cm[c] | B[box_of[i]]) & b:
                return []
            R[r] |= b
            Cm[c] |= b
            B[box_of[i]] |= b
    sols: List[List[int]] = []

    def rec(empt: List[int]) -> None:
        if len(sols) > cap:
            return
        if not empt:
            sols.append(g.copy())
            return
        best = None
        bm = 0
        bn = 99
        for i in empt:
            r, c = divmod(i, size)
            m = full & ~(R[r] | Cm[c] | B[box_of[i]])
            n = m.bit_count()
            if n < bn:
                best, bm, bn = i, m, n
                if n <= 1:
                    break
        if bn == 0 or best is None:
            return
        rest = [j for j in empt if j != best]
        r, c = divmod(best, size)
        bx = box_of[best]
        m = bm
        while m:
            b = m & -m
            m ^= b
            R[r] |= b
            Cm[c] |= b
            B[bx] |= b
            g[best] = b.bit_length()
            rec(rest)
            g[best] = 0
            R[r] ^= b
            Cm[c] ^= b
            B[bx] ^= b
            if len(sols) > cap:
                return

    rec([i for i in range(n_cells) if not g[i]])
    return sols


class SudokuEnv(UnifiedGameEnv):
    """Unified 6x6 and 9x9 (`Sudoku-Extreme`) Environment."""

    game_family = GameFamily.CONSTRAINT_CSP

    def __init__(
        self,
        size: int = 9,
        extreme: bool = True,
        unsolvable_frac: float = 0.0,
        num_queries: int = 8,
    ):
        if size not in (6, 9):
            raise ValueError("SudokuEnv size must be 6 or 9")
        self.size = int(size)
        self.extreme = bool(extreme)
        self.unsolvable_frac = float(unsolvable_frac)
        self.num_queries = int(num_queries)

        self.game_id = GameID.SUDOKU_6X6 if size == 6 else GameID.SUDOKU_9X9_EXTREME
        self.grid_h = self.size
        self.grid_w = self.size
        self.br, self.bc, self.box_of, self.units, self.peers = _sudoku_geometry(self.size)
        self.inventory_spec = self.br * 10 + self.bc
        self.default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH

        self.num_vars = self.size * self.size
        self.domain_size = self.size  # digits 1 .. size
        # Per-cell features: normalized row, col, box_id, is_given, plus one-hot given digit (size)
        self.feat_dim = 4 + self.size

        self._units_tensor_cache: Dict[torch.device, torch.Tensor] = {}

    def _get_units_tensor(self, device: torch.device) -> torch.Tensor:
        if device not in self._units_tensor_cache:
            self._units_tensor_cache[device] = torch.tensor(self.units, dtype=torch.long, device=device)
        return self._units_tensor_cache[device]

    def _enum_board(
        self,
        board: np.ndarray,
        cap: int,
        seed: int,
    ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Return `(num_solutions, marginals [N*N, N], sample_solution [N*N])`."""
        if self.size == 9:
            lib = _get_libsud9()
            if lib is not None:
                b = np.ascontiguousarray(board, dtype=np.int8)
                marg = np.zeros(81 * 9, dtype=np.float32)
                smp = np.zeros(81, dtype=np.int8)
                n = int(lib.enum9(b.ctypes.data, int(cap), int(seed) & ((1 << 63) - 1), marg.ctypes.data, smp.ctypes.data))
                return n, marg.reshape(81, 9), smp

        sols = _solve_all_py(board.tolist(), self.size, cap, self.box_of)
        n = len(sols)
        if n == 0:
            return 0, np.zeros((self.num_vars, self.size), dtype=np.float32), np.zeros(self.num_vars, dtype=np.int8)
        S = np.array(sols[:cap], dtype=np.int8)
        marg = np.stack([(S == d).mean(axis=0) for d in range(1, self.size + 1)], axis=-1).astype(np.float32)
        smp = S[seed % len(S)]
        return n, marg, smp

    def _random_full_grid(self, rng: random.Random) -> np.ndarray:
        if self.size == 9:
            lib = _get_libsud9()
            if lib is not None:
                g = np.zeros(81, dtype=np.int8)
                lib.random_grid9(rng.getrandbits(63), g.ctypes.data)
                g = g.reshape(9, 9)
                perm = np.array([0] + rng.sample(range(1, 10), 9), dtype=np.int8)
                g = perm[g]
                rows = [b * 3 + r for b in rng.sample(range(3), 3) for r in rng.sample(range(3), 3)]
                cols = [b * 3 + c for b in rng.sample(range(3), 3) for c in rng.sample(range(3), 3)]
                g = g[rows][:, cols]
                if rng.random() < 0.5:
                    g = g.T
                return g.reshape(-1).copy()

        # Pure Python generator for 6x6 (and 9x9 fallback)
        N = self.size
        while True:
            g = [0] * (N * N)
            idx = list(range(N * N))
            rng.shuffle(idx)
            for i in idx[:N]:
                d = rng.randint(1, N)
                if all(g[j] != d for j in self.peers[i]):
                    g[i] = d
            s = _solve_all_py(g, N, 1, self.box_of)
            if s:
                grid = np.array(s[0], dtype=np.int8).reshape(N, N)
                break
        perm = np.array([0] + rng.sample(range(1, N + 1), N), dtype=np.int8)
        grid = perm[grid]
        nb, ns = N // self.br, N // self.bc
        rows = [b * self.br + r for b in rng.sample(range(nb), nb) for r in rng.sample(range(self.br), self.br)]
        cols = [s * self.bc + c for s in rng.sample(range(ns), ns) for c in rng.sample(range(self.bc), self.bc)]
        return grid[rows][:, cols].reshape(-1).copy()

    def _naked_singles_fixpoint(self, board: np.ndarray) -> Tuple[np.ndarray, int]:
        """Propagate naked singles to a fixpoint; returns `(grid_after_ns, num_empty_remaining)`."""
        g = board.copy()
        N = self.size
        all_digits = set(range(1, N + 1))
        while True:
            moves = []
            for i in range(self.num_vars):
                if g[i] != 0:
                    continue
                used = {int(g[j]) for j in self.peers[i] if g[j] != 0}
                if len(used) == N - 1:
                    d = next(iter(all_digits - used))
                    moves.append((i, d))
            if not moves:
                break
            for i, d in moves:
                g[i] = d
        return g, int((g == 0).sum())

    def _generate_one(self, rng: random.Random, want_solvable: bool) -> Dict[str, Any]:
        N = self.size
        n_cells = self.num_vars
        while True:
            grid = self._random_full_grid(rng)
            if N == 6:
                kgiv = rng.randint(13, 17) if self.extreme else rng.randint(15, 20)
                cap = 1 if self.extreme else 20
            else:
                # 9x9 Sudoku-Extreme: unique or low-ambiguity multi-stall boards
                kgiv = rng.randint(27, 32) if self.extreme else rng.randint(30, 36)
                cap = 1 if self.extreme else 10

            idx = rng.sample(range(n_cells), kgiv)
            b = np.zeros(n_cells, dtype=np.int8)
            b[idx] = grid[idx]

            if want_solvable:
                nsol, marg, smp = self._enum_board(b, cap, rng.getrandbits(31))
                if nsol < 1 or nsol > cap:
                    continue
                if self.extreme:
                    _, rem_empty = self._naked_singles_fixpoint(b)
                    min_rem = 8 if N == 6 else 20
                    if rem_empty < min_rem:
                        continue
                smp_onehot = np.eye(N, dtype=np.float32)[smp - 1]
                marg = (1.0 - 1e-5) * marg + 1e-5 * smp_onehot
                return {
                    "board": b,
                    "solvable": True,
                    "outcome": smp,
                    "marg": marg,
                    "nsol": nsol,
                }
            else:
                # Unsolvable twin: change 1 or 2 givens without creating a direct peer conflict,
                # verified to have 0 solutions.
                for _ in range(25):
                    bb = b.copy()
                    n_swap = 1 if rng.random() < 0.7 else 2
                    swap_cells = rng.sample(idx, min(n_swap, len(idx)))
                    ok_swap = True
                    for c_idx in swap_cells:
                        cands = [
                            d
                            for d in range(1, N + 1)
                            if d != bb[c_idx] and all(bb[j] != d for j in self.peers[c_idx])
                        ]
                        if not cands:
                            ok_swap = False
                            break
                        bb[c_idx] = rng.choice(cands)
                    if not ok_swap:
                        continue
                    nsol, _, _ = self._enum_board(bb, 1, 0)
                    if nsol == 0:
                        return {
                            "board": bb,
                            "solvable": False,
                            "outcome": grid,
                            "marg": np.zeros((n_cells, N), dtype=np.float32),
                            "nsol": 0,
                        }

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 10_000, "cal": 20_000, "test": 30_000}.get(split, 40_000)
        rng = random.Random(seed * 1009 + split_offset + self.size)

        B = batch_size
        V = self.num_vars
        N = self.size
        Q = self.num_queries

        obs_tokens = np.zeros((B, V), dtype=np.int64)
        obs_mask = np.zeros((B, V), dtype=bool)
        target_tokens = np.zeros((B, V), dtype=np.int64)
        exact_marginals = np.zeros((B, V, N), dtype=np.float32)
        obs_features = np.zeros((B, V, self.feat_dim), dtype=np.float32)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        solvable = np.ones(B, dtype=bool)
        ambig_mask = np.zeros((B, V), dtype=bool)

        rows_norm = np.array([i // N / max(1, N - 1) for i in range(V)], dtype=np.float32)
        cols_norm = np.array([(i % N) / max(1, N - 1) for i in range(V)], dtype=np.float32)
        box_norm = np.array([self.box_of[i] / max(1, N - 1) for i in range(V)], dtype=np.float32)

        for b in range(B):
            want_solv = rng.random() >= self.unsolvable_frac
            item = self._generate_one(rng, want_solv)
            board = item["board"]
            outcome = item["outcome"]
            marg = item["marg"]

            obs_tokens[b] = board
            obs_mask[b] = board > 0
            target_tokens[b] = outcome
            exact_marginals[b] = marg
            solvable[b] = item["solvable"]

            ns_grid, _ = self._naked_singles_fixpoint(board)
            ambig_mask[b] = ns_grid == 0

            obs_features[b, :, 0] = rows_norm
            obs_features[b, :, 1] = cols_norm
            obs_features[b, :, 2] = box_norm
            obs_features[b, :, 3] = obs_mask[b].astype(np.float32)
            for i in range(V):
                if board[i] > 0:
                    obs_features[b, i, 4 + int(board[i]) - 1] = 1.0

            # Sample relational queries Q(A, B) = P(cell_A == cell_B) on non-peer empty cells
            empty_cells = [i for i in range(V) if board[i] == 0]
            if len(empty_cells) < 2:
                empty_cells = list(range(V))
            for q in range(Q):
                ca, cb = rng.sample(empty_cells, 2)
                query_pairs[b, q] = (ca, cb, int(QueryType.EQUAL))
                if item["solvable"]:
                    if item["nsol"] == 1:
                        query_targets[b, q] = 1.0 if outcome[ca] == outcome[cb] else 0.0
                    else:
                        # Dot product of marginals (exact when nsol=1, well-calibrated approximation otherwise)
                        query_targets[b, q] = float(np.dot(marg[ca], marg[cb]))
                else:
                    query_targets[b, q] = 0.0

        header = self.build_batch_header(B, device=dev)
        return EnvBatch(
            header=header,
            obs_tokens=torch.as_tensor(obs_tokens, dtype=torch.long, device=dev),
            obs_mask=torch.as_tensor(obs_mask, dtype=torch.bool, device=dev),
            obs_features=torch.as_tensor(obs_features, dtype=torch.float32, device=dev),
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.as_tensor(solvable, dtype=torch.bool, device=dev),
            meta={"ambig_mask": torch.as_tensor(ambig_mask, dtype=torch.bool, device=dev)},
        )

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        nxt = super().step_unmask(batch, var_idx, val_idx)
        b_arange = torch.arange(nxt.batch_size, device=nxt.device)
        nxt.obs_features[b_arange, var_idx, 3] = 1.0
        nxt.obs_features[b_arange, var_idx, 4:] = 0.0
        nxt.obs_features[b_arange, var_idx, 4 + (val_idx - 1).clamp(0, self.size - 1)] = 1.0
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Fast vectorized PyTorch verifier: checks givens match and all rows/cols/boxes are permutations of `1..N`."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens >= 1) & (candidate_tokens <= self.size)

        units_t = self._get_units_tensor(candidate_tokens.device)  # [3*N, N]
        # Gather candidate digits per unit: [B, 3*N, N]
        B = candidate_tokens.shape[0]
        N = self.size
        unit_vals = candidate_tokens[:, units_t.reshape(-1)].reshape(B, 3 * N, N)
        # Check every digit 1..N appears in every unit
        digits = torch.arange(1, N + 1, device=candidate_tokens.device).view(1, 1, 1, N)
        present = (unit_vals.unsqueeze(-1) == digits).any(dim=2)  # [B, 3*N, N]
        units_ok = present.all(dim=-1).all(dim=-1)
        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & units_ok
