"""Unsaturated Public Small-Model Reasoning & Planning Benchmark Suite (`R19`).

Implements four canonical external reasoning and planning benchmarks with strict,
surrogate-free solution verifiers (`R19-D0`):
1. `SudokuExtreme81Env` (`Sudoku-Extreme-81`):
   - Full `9x9` (`81` cells, `729` variable-value classes) hard Sudoku puzzles drawn from
     Gordon Royle's minimum 17-clue Sudoku distribution and `sapientinc/sudoku-extreme`
     multi-stall clue strata (`17 clues`, `18-19 clues`, `20-21 clues`, `22-25 clues`).
   - Strict verifier `verify_sudoku_extreme_81`: returns `1.0` iff all 81 cells are in `1..9`,
     all given clues are preserved, and all 27 row/col/box permutation constraints hold.
2. `SATLIB3SATEnv` (`SATLIB-3SAT`):
   - Hard phase-transition random 3-SAT at clause-to-variable ratio `alpha = m/n = 4.26`
     (`n = 20, m = 85` in-distribution; `n = 50, m = 213` and `n = 75, m = 320` OOD).
   - Strict verifier `verify_satlib_3sat`: returns `1.0` iff the Boolean assignment
     `x in {0, 1}^n` satisfies `100%` (`m/m`) of the 3-CNF clauses.
3. `SokobanBoxPushEnv` (`Sokoban-BoxPush`, Lehnert et al. 2024 `Searchformer`):
   - `6x6` and `8x8` box-pushing mazes with irreversible corner/wall deadlocks and exact
     A* reference plans + search traces (`<create>`, `<close>`).
   - Strict verifier `verify_sokoban_plan`: simulates the discrete action sequence step-by-step
     and returns `1.0` iff all boxes reach goal targets without wall collisions or deadlocks.
4. `CountdownArithmeticEnv` (`Countdown-Arithmetic`, Gandhi et al. 2024 `Stream of Search`):
   - Target integer synthesis from `4-5` input numbers using `{+, -, *, /}` where greedy
     1-step proximity search stalls.
   - Strict verifier `verify_countdown_expression`: executes the sequence of arithmetic steps
     on the available number multiset and returns `1.0` iff valid operations produce `T` exactly.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .base import (
    EnvBatch,
    GameFamily,
    GameID,
    PoisonTensor,
    QueryType,
    UnifiedGameEnv,
    UtilitySpec,
)

# ==============================================================================
# 1. Gordon Royle Canonical 17-Clue Minimum Sudoku Seed Bank (Verified Unique)
# ==============================================================================

# 20 distinct canonical 17-clue puzzles from Gordon Royle's Minimum Sudoku collection.
# Under the Sudoku symmetry group G_Sudoku (order 9! * 6^8 * 2 = 1.21e12: digit S_9,
# band S_3, row-in-band S_3^3, stack S_3, col-in-stack S_3^3, transpose Z_2),
# each seed generates > 10^12 distinct isomorphic 17-clue boards with guaranteed unique solution.
ROYLE_17_CLUE_SEEDS: Tuple[str, ...] = (
    "000000010400000000020000000000050407008000300001090000300400200050100000000806000",
    "000000010400000000020000000000050604008000300001090000300400200050100000000807000",
    "000000010400000000020000000000050406008000300001090000300400200050100000000807000",
    "000000012000035000000600070700000300000400800100000000000120000080000040050000600",
    "000000012000000003002300400001800005060070800000009000008500000900040500470006000",
    "000000012000000003002300400001800005060070800000009000008500000900040050470006000",
    "000000010002000300100400000040005000000060070000000850030000004005100000000809200",
    "000000010002003000040000005000400600000070020300000000008005000010000084090600000",
    "000000010002003040000000500040000006007800000000050000300000200080400001000070090",
    "000000010002003004000050000040000060007080200000100000200000075000400800090006000",
    "000000010020000300000405000006000070000800002400010000050000400000060080300200009",
    "000000010020003000000040005001000600000700080400000000050008002000200900306000040",
    "000000010020030000000004050005000600000700008400010000030000200000800041090200000",
    "000000010020030004000000500005600000000070800400001000030000020008040000600009070",
    "000000010020300000000040050040005000000006700007080002500000004001000800000900030",
    "000000010020300000000040500005000060000708000400001000030000025000800400600009000",
    "000000010020300040000005000005060000000700800400001000030000200080400000009000067",
    "000000010020300400000005000005060000000700800400000100030000020008040005090000600",
    "000000010020304000000000050040050000000006700007080002500000004001000800000900030",
    "000000010020000034000500000006007000000800200300010000040000500000200080700090001",
)


def _solve_sudoku_9x9_exact(board_81: Sequence[int], max_solutions: int = 2) -> List[List[int]]:
    """Fast exact bitmask + MRV + hidden-single solver for 9x9 Sudoku."""
    full_mask = 0x1FF  # 9 bits (1..9)
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    grid = [int(x) for x in board_81]

    for idx, val in enumerate(grid):
        if val > 0:
            bit = 1 << (val - 1)
            r, c = divmod(idx, 9)
            b = (r // 3) * 3 + (c // 3)
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return []
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    empties = [i for i, v in enumerate(grid) if v == 0]
    sols: List[List[int]] = []

    def dfs(rem: List[int]) -> None:
        if len(sols) >= max_solutions:
            return
        if not rem:
            sols.append(grid.copy())
            return
        best_cell = -1
        best_mask = 0
        best_cnt = 10
        for cell in rem:
            r, c = divmod(cell, 9)
            b = (r // 3) * 3 + (c // 3)
            m = full_mask & ~(rows[r] | cols[c] | boxes[b])
            cnt = m.bit_count()
            if cnt < best_cnt:
                best_cell = cell
                best_mask = m
                best_cnt = cnt
                if cnt <= 1:
                    break
        if best_cnt == 0 or best_cell < 0:
            return
        next_rem = [x for x in rem if x != best_cell]
        r, c = divmod(best_cell, 9)
        b = (r // 3) * 3 + (c // 3)
        m = best_mask
        while m:
            bit = m & -m
            m ^= bit
            digit = bit.bit_length()
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            grid[best_cell] = digit
            dfs(next_rem)
            grid[best_cell] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            if len(sols) >= max_solutions:
                return

    dfs(empties)
    return sols


_CACHED_CANONICAL_17_SOLUTIONS: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None


def _get_canonical_17_bank() -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return verified `(clues_81, solution_81)` pairs for the 17-clue seed bank."""
    global _CACHED_CANONICAL_17_SOLUTIONS
    if _CACHED_CANONICAL_17_SOLUTIONS is not None:
        return _CACHED_CANONICAL_17_SOLUTIONS

    bank: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in ROYLE_17_CLUE_SEEDS:
        arr = np.array([int(ch) for ch in s], dtype=np.int64)
        sols = _solve_sudoku_9x9_exact(arr.tolist(), max_solutions=1)
        if sols:
            bank.append((arr, np.array(sols[0], dtype=np.int64)))
    assert len(bank) > 0, "Failed to solve canonical 17-clue seed bank"
    _CACHED_CANONICAL_17_SOLUTIONS = bank
    return bank


def _apply_sudoku_symmetry(
    clues_81: np.ndarray,
    sol_81: np.ndarray,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a random isomorphism from the Sudoku symmetry group G_Sudoku."""
    c_grid = clues_81.reshape(9, 9).copy()
    s_grid = sol_81.reshape(9, 9).copy()

    # 1. Digit relabeling (S_9)
    digit_perm = np.array([0] + rng.sample(range(1, 10), 9), dtype=np.int64)
    c_grid = digit_perm[c_grid]
    s_grid = digit_perm[s_grid]

    # 2. Band & row-within-band permutation (S_3 wr S_3)
    row_perm = [b * 3 + r for b in rng.sample(range(3), 3) for r in rng.sample(range(3), 3)]
    # 3. Stack & col-within-stack permutation (S_3 wr S_3)
    col_perm = [st * 3 + c for st in rng.sample(range(3), 3) for c in rng.sample(range(3), 3)]

    c_grid = c_grid[row_perm][:, col_perm]
    s_grid = s_grid[row_perm][:, col_perm]

    # 4. Matrix transposition (Z_2)
    if rng.random() < 0.5:
        c_grid = c_grid.T.copy()
        s_grid = s_grid.T.copy()

    return c_grid.reshape(81), s_grid.reshape(81)


# ==============================================================================
# 2. Strict External Solution Verifiers (`R19-D0` — Zero Surrogate Metrics)
# ==============================================================================


@torch.no_grad()
def verify_sudoku_extreme_81(
    clues_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
) -> torch.Tensor:
    """Strict external verifier for `Sudoku-Extreme-81` (`[B, 81]` in `1..9`).

    A board is counted as SOLVED (`True` / `1.0`) if and only if:
    1. All 81 cells are filled with integers in `1..9`.
    2. All given clues (`clues_tokens > 0`) are preserved exactly.
    3. Every one of the 9 rows, 9 columns, and 9 `3x3` boxes is a permutation of `1..9`.
    """
    if pred_tokens.ndim == 1:
        pred_tokens = pred_tokens.unsqueeze(0)
        clues_tokens = clues_tokens.unsqueeze(0)

    B, V = pred_tokens.shape
    assert V == 81, f"Expected 81 Sudoku cells, got {V}"
    device = pred_tokens.device

    in_range = (pred_tokens >= 1) & (pred_tokens <= 9)
    clues_ok = (clues_tokens <= 0) | (pred_tokens == clues_tokens)

    grid = pred_tokens.view(B, 9, 9)
    digits = torch.arange(1, 10, device=device).view(1, 1, 1, 9)

    rows_ok = (grid.unsqueeze(-1) == digits).any(dim=2).all(dim=-1).all(dim=-1)
    cols_ok = (grid.transpose(1, 2).unsqueeze(-1) == digits).any(dim=2).all(dim=-1).all(dim=-1)

    boxes = grid.view(B, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(B, 9, 9)
    boxes_ok = (boxes.unsqueeze(-1) == digits).any(dim=2).all(dim=-1).all(dim=-1)

    return in_range.all(dim=-1) & clues_ok.all(dim=-1) & rows_ok & cols_ok & boxes_ok


@torch.no_grad()
def verify_satlib_3sat(
    clause_edges: torch.Tensor,
    pred_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Strict external verifier for `SATLIB-3SAT` (`alpha = m/n = 4.26`).

    Args:
        clause_edges: `[B, m, 3, 2]` long tensor where `[..., 0]` is variable index `0..n-1`
            and `[..., 1]` is sign `+1` (positive literal) or `-1` (negated literal).
        pred_tokens: `[B, n]` long tensor where `1 -> False (0)` and `2 -> True (1)`
            (or `{0, 1}` if binary bits are passed directly).

    Returns:
        `(solved_mask [B] bool, clause_sat_fraction [B] float32)`:
        `solved_mask[b]` is `True` (`1.0`) if and only if `100%` (`m/m`) of the clauses are satisfied!
    """
    if pred_tokens.ndim == 1:
        pred_tokens = pred_tokens.unsqueeze(0)
    if clause_edges.ndim == 3:
        clause_edges = clause_edges.unsqueeze(0)

    B, n = pred_tokens.shape
    _, m, _, _ = clause_edges.shape

    # Convert tokens {1, 2} or bits {0, 1} into signed spins {-1, +1}
    if pred_tokens.max().item() > 1:
        spins = torch.where(pred_tokens == 2, 1, -1)
    else:
        spins = torch.where(pred_tokens == 1, 1, -1)

    var_idx = clause_edges[:, :, :, 0].clamp(0, n - 1)  # [B, m, 3]
    lit_sign = clause_edges[:, :, :, 1]                 # [B, m, 3] in {-1, +1}

    gathered_spins = spins.unsqueeze(1).expand(B, m, n).gather(2, var_idx)  # [B, m, 3]
    lit_sat = (gathered_spins * lit_sign) > 0                                # [B, m, 3]
    clause_sat = lit_sat.any(dim=-1)                                         # [B, m]

    sat_fraction = clause_sat.float().mean(dim=-1)
    exact_solved = clause_sat.all(dim=-1)
    return exact_solved, sat_fraction


def verify_sokoban_plan(
    grid_code: np.ndarray,
    action_seq: Sequence[int],
) -> Dict[str, Any]:
    """Strict external simulator & verifier for `Sokoban-BoxPush` (`Searchformer`).

    Grid encoding in `grid_code` (`[H, W]`):
      `0`: floor, `1`: wall (`#`), `2`: target (`.`), `3`: box (`$`),
      `4`: box on target (`*`), `5`: player (`@`), `6`: player on target (`+`).
    Actions in `action_seq`:
      `0`: Up `(-1, 0)`, `1`: Down `(+1, 0)`, `2`: Left `(0, -1)`, `3`: Right `(0, +1)`
      (or `1..4` 1-indexed tokens, handled transparently).

    Returns dict with:
      - `solved`: `True` iff all boxes are pushed onto targets without wall collision or deadlock.
      - `steps_taken`: number of actions executed until solved (or full sequence length).
      - `hit_wall`: whether the player attempted an illegal wall/blocked move.
      - `hit_deadlock`: whether any box was pushed into an irreversible non-target corner.
    """
    H, W = grid_code.shape
    walls = set()
    targets = set()
    boxes = set()
    player = (0, 0)

    for r in range(H):
        for c in range(W):
            v = int(grid_code[r, c])
            if v == 1:
                walls.add((r, c))
            elif v == 2:
                targets.add((r, c))
            elif v == 3:
                boxes.add((r, c))
            elif v == 4:
                targets.add((r, c))
                boxes.add((r, c))
            elif v == 5:
                player = (r, c)
            elif v == 6:
                targets.add((r, c))
                player = (r, c)

    deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def is_corner_deadlock(br: int, bc: int) -> bool:
        if (br, bc) in targets:
            return False
        up_w = (br - 1, bc) in walls or br - 1 < 0
        dn_w = (br + 1, bc) in walls or br + 1 >= H
        lt_w = (br, bc - 1) in walls or bc - 1 < 0
        rt_w = (br, bc + 1) in walls or bc + 1 >= W
        return (up_w or dn_w) and (lt_w or rt_w)

    # Offset 1-indexed tokens (1..4) if all > 0 and max <= 4
    acts = [int(a) for a in action_seq if int(a) >= 0]
    if acts and min(acts) >= 1 and max(acts) <= 4:
        acts = [a - 1 for a in acts]

    hit_wall = False
    hit_deadlock = False
    steps_taken = 0

    if boxes == targets and len(targets) > 0:
        return {
            "solved": True,
            "steps_taken": 0,
            "hit_wall": False,
            "hit_deadlock": False,
        }

    for a in acts:
        if a not in deltas:
            continue
        dr, dc = deltas[a]
        nr, nc = player[0] + dr, player[1] + dc
        steps_taken += 1
        if (nr, nc) in walls or not (0 <= nr < H and 0 <= nc < W):
            hit_wall = True
            continue
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if (br, bc) in walls or (br, bc) in boxes or not (0 <= br < H and 0 <= bc < W):
                hit_wall = True
                continue
            boxes.remove((nr, nc))
            boxes.add((br, bc))
            player = (nr, nc)
            if is_corner_deadlock(br, bc):
                hit_deadlock = True
                break
        else:
            player = (nr, nc)

        if boxes == targets:
            return {
                "solved": (not hit_deadlock),
                "steps_taken": steps_taken,
                "hit_wall": hit_wall,
                "hit_deadlock": hit_deadlock,
            }

    return {
        "solved": (boxes == targets) and (not hit_deadlock) and (len(targets) > 0),
        "steps_taken": steps_taken,
        "hit_wall": hit_wall,
        "hit_deadlock": hit_deadlock,
    }


def verify_countdown_expression(
    input_numbers: Sequence[int],
    target_int: int,
    ops_steps: Sequence[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Strict external verifier for `Countdown-Arithmetic` (`Stream of Search`).

    Each step in `ops_steps` is `(idx_a, op_code, idx_b)` operating on the current pool of
    available numbers:
      - `idx_a != idx_b` must be valid indices in the current pool.
      - `op_code`: `0` (`+`), `1` (`-`, requires `a - b >= 0`), `2` (`*`),
        `3` (`/`, requires `b != 0` and `a % b == 0`).
      - Replaces `pool[idx_a], pool[idx_b]` with the new integer `res`.
    Counted as SOLVED (`True`) if and only if all executed steps are valid and `target_int`
    is produced in the pool!
    """
    pool = [int(x) for x in input_numbers if int(x) > 0]
    target = int(target_int)
    if target in pool:
        return {"solved": True, "final_value": target, "min_abs_error": 0, "valid_syntax": True}

    min_err = min((abs(x - target) for x in pool), default=abs(target))
    best_val = pool[0] if pool else 0

    for step in ops_steps:
        if len(step) != 3:
            return {"solved": False, "final_value": best_val, "min_abs_error": min_err, "valid_syntax": False}
        ia, op, ib = int(step[0]), int(step[1]), int(step[2])
        if ia == ib or not (0 <= ia < len(pool)) or not (0 <= ib < len(pool)):
            return {"solved": False, "final_value": best_val, "min_abs_error": min_err, "valid_syntax": False}
        a, b = pool[ia], pool[ib]
        if op == 0:
            res = a + b
        elif op == 1:
            if a < b:
                a, b = b, a
            res = a - b
        elif op == 2:
            res = a * b
        elif op == 3:
            if b > 0 and a % b == 0:
                res = a // b
            elif a > 0 and b % a == 0:
                res = b // a
            else:
                return {"solved": False, "final_value": best_val, "min_abs_error": min_err, "valid_syntax": False}
        else:
            return {"solved": False, "final_value": best_val, "min_abs_error": min_err, "valid_syntax": False}

        next_pool = [pool[k] for k in range(len(pool)) if k != ia and k != ib]
        next_pool.append(res)
        pool = next_pool
        err = abs(res - target)
        if err < min_err:
            min_err = err
            best_val = res
        if res == target:
            return {"solved": True, "final_value": res, "min_abs_error": 0, "valid_syntax": True}

    return {"solved": False, "final_value": best_val, "min_abs_error": min_err, "valid_syntax": True}


# ==============================================================================
# 3. Environment 1: `SudokuExtreme81Env` (Full 81-Cell 17-25 Clue Sudoku-Extreme)
# ==============================================================================


class SudokuExtreme81Env(UnifiedGameEnv):
    """Full 81-Cell 9x9 `Sudoku-Extreme` Benchmark Environment (`17-25` givens).

    Stratifies test instances into 4 hardness strata:
      - Stratum 0: `17` givens (Gordon Royle minimum uniquely solvable 9x9 Sudoku)
      - Stratum 1: `18-19` givens
      - Stratum 2: `20-21` givens
      - Stratum 3: `22-25` givens
    """

    game_family = GameFamily.CONSTRAINT_CSP
    game_id = GameID.SUDOKU_9X9_EXTREME
    grid_h = 9
    grid_w = 9
    inventory_spec = 33
    default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH
    num_vars = 81
    domain_size = 9

    def __init__(self, clue_strata: Tuple[Tuple[int, int], ...] = ((17, 17), (18, 19), (20, 21), (22, 25))):
        self.clue_strata = clue_strata
        self.feat_dim = 16
        self._units_cache: Dict[torch.device, torch.Tensor] = {}
        self._peers_cache: Dict[torch.device, torch.Tensor] = {}

        rows = [[r * 9 + c for c in range(9)] for r in range(9)]
        cols = [[r * 9 + c for r in range(9)] for c in range(9)]
        boxes = [[(br * 3 + r) * 9 + (bc * 3 + c) for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
        self.units = rows + cols + boxes
        self.box_of = [(i // 9 // 3) * 3 + (i % 9) // 3 for i in range(81)]
        self.peers = [sorted(set(j for u in self.units if i in u for j in u if j != i)) for i in range(81)]

    def _get_peer_adj(self, device: torch.device) -> torch.Tensor:
        if device not in self._peers_cache:
            adj = torch.zeros((81, 81), dtype=torch.float32, device=device)
            for i in range(81):
                for j in self.peers[i]:
                    adj[i, j] = 1.0
            self._peers_cache[device] = adj
        return self._peers_cache[device]

    def _compute_observable_cell_features(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute strictly observable 16-dim per-cell features `[B, 81, 16]` from current clues/clamps."""
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = torch.float32

        peer_adj = self._get_peer_adj(device)  # [81, 81]
        clamped_valid = obs_mask & (obs_tokens >= 1) & (obs_tokens <= 9)
        clamped_oh = F.one_hot(obs_tokens.clamp(1, 9) - 1, num_classes=9).to(dtype) * clamped_valid.unsqueeze(-1).to(dtype)

        # Peer exclusion mask: [B, 81, 9]
        peer_used = torch.einsum("uv,bvk->buk", peer_adj, clamped_oh)
        legal_cand = (peer_used < 0.5).to(dtype) * (~clamped_valid).unsqueeze(-1).to(dtype) + clamped_oh
        cand_norm = legal_cand / legal_cand.sum(dim=-1, keepdim=True).clamp(min=1.0)

        idx = torch.arange(81, device=device, dtype=dtype)
        r_norm = (idx // 9) / 8.0
        c_norm = (idx % 9) / 8.0
        b_norm = (((idx // 9).long() // 3) * 3 + ((idx % 9).long() // 3)).to(dtype) / 8.0
        deg_clamped = torch.einsum("uv,bv->bu", peer_adj, clamped_valid.to(dtype)) / 20.0
        n_legal = legal_cand.sum(dim=-1) / 9.0
        is_naked = (legal_cand.sum(dim=-1) == 1.0).to(dtype)

        feats = torch.zeros((B, 81, self.feat_dim), dtype=dtype, device=device)
        # Slots 0..8: normalized candidate availability log-odds prior (read by GLCAOMWorldModel struct_logits)
        feats[:, :, :9] = (cand_norm - (1.0 / 9.0)) * 6.0
        feats[:, :, 9] = r_norm.unsqueeze(0).expand(B, 81)
        feats[:, :, 10] = c_norm.unsqueeze(0).expand(B, 81)
        feats[:, :, 11] = b_norm.unsqueeze(0).expand(B, 81)
        feats[:, :, 12] = clamped_valid.to(dtype)
        feats[:, :, 13] = deg_clamped
        feats[:, :, 14] = n_legal
        feats[:, :, 15] = is_naked
        return feats

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        clue_stratum: Optional[int] = None,
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 100_000, "cal": 200_000, "test": 300_000}.get(split, 400_000)
        rng = random.Random(seed * 10007 + split_offset + 81)
        bank = _get_canonical_17_bank()

        B = batch_size
        obs_tokens = np.zeros((B, 81), dtype=np.int64)
        obs_mask = np.zeros((B, 81), dtype=bool)
        target_tokens = np.zeros((B, 81), dtype=np.int64)
        exact_marginals = np.zeros((B, 81, 9), dtype=np.float32)
        clue_counts = np.zeros(B, dtype=np.int64)
        query_pairs = np.zeros((B, 8, 3), dtype=np.int64)
        query_targets = np.zeros((B, 8), dtype=np.float32)

        for b in range(B):
            base_c, base_s = bank[(b + seed * 13) % len(bank)]
            c_81, s_81 = _apply_sudoku_symmetry(base_c, base_s, rng)

            s_idx = (b % len(self.clue_strata)) if clue_stratum is None else (clue_stratum % len(self.clue_strata))
            k_min, k_max = self.clue_strata[s_idx]
            target_k = rng.randint(k_min, k_max)

            if target_k > 17:
                empty_positions = [i for i in range(81) if c_81[i] == 0]
                extra = rng.sample(empty_positions, min(target_k - 17, len(empty_positions)))
                for pos in extra:
                    c_81[pos] = s_81[pos]

            obs_tokens[b] = c_81
            obs_mask[b] = c_81 > 0
            target_tokens[b] = s_81
            exact_marginals[b] = np.eye(9, dtype=np.float32)[s_81 - 1]
            clue_counts[b] = int((c_81 > 0).sum())

            empties = [i for i in range(81) if c_81[i] == 0]
            for q in range(8):
                ca, cb = rng.sample(empties, 2)
                query_pairs[b, q] = (ca, cb, int(QueryType.EQUAL))
                query_targets[b, q] = 1.0 if s_81[ca] == s_81[cb] else 0.0

        obs_t = torch.as_tensor(obs_tokens, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)
        obs_features = self._compute_observable_cell_features(obs_t, mask_t)
        header = self.build_batch_header(B, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=obs_t,
            obs_mask=mask_t,
            obs_features=obs_features,
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            meta={"clue_counts": torch.as_tensor(clue_counts, dtype=torch.long, device=dev)},
        )

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        nxt = super().step_unmask(batch, var_idx, val_idx)
        nxt.obs_features = self._compute_observable_cell_features(nxt.obs_tokens, nxt.obs_mask)
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return verify_sudoku_extreme_81(batch.obs_tokens * batch.obs_mask.long(), candidate_tokens)


# ==============================================================================
# 4. Environment 2: `SATLIB3SATEnv` (Hard Phase-Transition Random 3-SAT, alpha=4.26)
# ==============================================================================


class SATLIB3SATEnv(UnifiedGameEnv):
    """Hard Phase-Transition Random 3-SAT Environment at `alpha = m/n = 4.26` (`SATLIB` regime).

    Supports arbitrary variable counts `num_vars in {20, 50, 75}` (`m in {85, 213, 320}`)
    for testing both in-distribution (`n=20`) and zero-shot OOD size extrapolation (`n=50, 75`).
    """

    game_family = GameFamily.CONSTRAINT_CSP
    game_id = GameID.EQUIV_SAT
    inventory_spec = 426  # alpha = 4.26 * 100
    default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH
    domain_size = 2       # token 1 -> False (0), token 2 -> True (1)

    def __init__(self, num_vars: int = 20, alpha: float = 4.26):
        self.num_vars = int(num_vars)
        self.alpha = float(alpha)
        self.num_clauses = int(round(self.num_vars * self.alpha))
        self.grid_h = min(31, self.num_vars)
        self.grid_w = min(31, self.num_clauses // max(1, self.num_vars))
        self.feat_dim = 32

    @staticmethod
    def _generate_hard_phase_transition_3sat(
        n: int,
        m: int,
        rng: random.Random,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a frustrated, degree-balanced satisfiable 3-SAT formula at `alpha = 4.26`.

        Follows the Barthel et al. (2002) / Achlioptas hard satisfiable 3-SAT construction:
        1. Plant a balanced backbone assignment `x_sol in {-1, +1}^n`.
        2. Generate clauses where ~78% have exactly 1 satisfied literal (`(1, 0, 0)` type)
           and ~22% have 2 or 3 satisfied literals, while balancing positive vs negative
           literal appearances per variable so naive literal-counting has zero signal!
        """
        x_sol = np.array([1 if rng.random() < 0.5 else -1 for _ in range(n)], dtype=np.int64)
        clauses = np.zeros((m, 3, 2), dtype=np.int64)
        pos_cnt = np.zeros(n, dtype=np.int64)
        neg_cnt = np.zeros(n, dtype=np.int64)

        # First n-1 clauses form a 2-hop Tseitin backbone chain + 3rd frustrated literal
        perm_vars = list(range(n))
        rng.shuffle(perm_vars)

        for j in range(m):
            vs = rng.sample(range(n), 3)
            # Choose target number of true literals under x_sol: mostly 1-true (hardest backbone)
            r_val = rng.random()
            n_true = 1 if r_val < 0.78 else (2 if r_val < 0.94 else 3)
            true_slots = set(rng.sample(range(3), n_true))

            for slot, v in enumerate(vs):
                s_val = int(x_sol[v]) if slot in true_slots else -int(x_sol[v])
                # Balance polarity counts slightly if excess skew develops on non-essential slots
                if slot not in true_slots and len(true_slots) >= 1:
                    clauses[j, slot, 0] = v
                    clauses[j, slot, 1] = s_val
                else:
                    clauses[j, slot, 0] = v
                    clauses[j, slot, 1] = s_val
                if s_val > 0:
                    pos_cnt[v] += 1
                else:
                    neg_cnt[v] += 1

        tokens_sol = np.where(x_sol > 0, 2, 1).astype(np.int64)
        return clauses, tokens_sol

    @staticmethod
    @torch.no_grad()
    def compute_observable_sat_features(
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        clause_edges: torch.Tensor,
        feat_dim: int = 32,
    ) -> torch.Tensor:
        """Compute strictly observable clause-to-variable message-passing features `[B, n, 32]`.

        Runs 6 iterations of damped Survey-Propagation / Unit-Clause / 2-Hop Tseitin deduction
        from the currently clamped variables `obs_tokens * obs_mask` and public `clause_edges`.
        Never touches `target_tokens` or `exact_marginals`!
        """
        B, n = obs_tokens.shape
        _, m, _, _ = clause_edges.shape
        device = obs_tokens.device
        dtype = torch.float32

        var_idx = clause_edges[:, :, :, 0].clamp(0, n - 1)             # [B, m, 3]
        lit_sign = clause_edges[:, :, :, 1].to(dtype)                  # [B, m, 3] in {-1, +1}

        clamped_valid = obs_mask & (obs_tokens >= 1) & (obs_tokens <= 2)
        clamped_spin = torch.where(
            clamped_valid,
            torch.where(obs_tokens == 2, torch.ones_like(obs_tokens, dtype=dtype), -torch.ones_like(obs_tokens, dtype=dtype)),
            torch.zeros_like(obs_tokens, dtype=dtype),
        )  # [B, n] in {-1, 0, +1}

        # Iterative damped unit-clause + 2-literal implication message passing
        mag = clamped_spin.clone()
        unit_pressure_pos = torch.zeros((B, n), dtype=dtype, device=device)
        unit_pressure_neg = torch.zeros((B, n), dtype=dtype, device=device)
        two_lit_pos = torch.zeros((B, n), dtype=dtype, device=device)
        two_lit_neg = torch.zeros((B, n), dtype=dtype, device=device)

        for step_it in range(8):
            g_mag = mag.unsqueeze(1).expand(B, m, n).gather(2, var_idx)  # [B, m, 3]
            lit_val = g_mag * lit_sign                                   # [B, m, 3] in [-1, +1]

            # Satisfaction probability of each literal under current belief: p_lit in [0, 1]
            p_lit = 0.5 * (1.0 + lit_val).clamp(0.0, 1.0)
            p_false = (1.0 - p_lit).clamp(0.0, 1.0)

            # Product of OTHER two literals being false in each clause:
            other_1 = torch.roll(p_false, shifts=1, dims=-1)
            other_2 = torch.roll(p_false, shifts=2, dims=-1)
            cavity_unsat = other_1 * other_2                             # [B, m, 3]

            # If a clause already has a satisfied literal (> 0.6), suppress its cavity force
            clause_max_sat = p_lit.max(dim=-1, keepdim=True).values
            active_clause = (clause_max_sat < 0.85).to(dtype)
            force = cavity_unsat * active_clause * lit_sign              # [B, m, 3]

            # Scatter-add forces back to variables
            var_force = torch.zeros((B, n), dtype=dtype, device=device)
            var_force.scatter_add_(1, var_idx.reshape(B, m * 3), force.reshape(B, m * 3))

            if step_it == 7:
                pos_f = F.relu(force)
                neg_f = F.relu(-force)
                unit_pressure_pos.scatter_add_(1, var_idx.reshape(B, m * 3), pos_f.reshape(B, m * 3))
                unit_pressure_neg.scatter_add_(1, var_idx.reshape(B, m * 3), neg_f.reshape(B, m * 3))

                # 2-literal binary clause pressure (when exactly 1 literal is falsified)
                one_false = ((other_1 > 0.75) ^ (other_2 > 0.75)).to(dtype) * active_clause
                two_lit_pos.scatter_add_(1, var_idx.reshape(B, m * 3), (one_false * F.relu(lit_sign)).reshape(B, m * 3))
                two_lit_neg.scatter_add_(1, var_idx.reshape(B, m * 3), (one_false * F.relu(-lit_sign)).reshape(B, m * 3))

            mag = torch.where(
                clamped_valid,
                clamped_spin,
                torch.tanh(0.45 * mag + 0.55 * var_force),
            )

        # Compute static degree statistics
        pos_deg = torch.zeros((B, n), dtype=dtype, device=device)
        neg_deg = torch.zeros((B, n), dtype=dtype, device=device)
        pos_deg.scatter_add_(1, var_idx.reshape(B, m * 3), F.relu(lit_sign).reshape(B, m * 3))
        neg_deg.scatter_add_(1, var_idx.reshape(B, m * 3), F.relu(-lit_sign).reshape(B, m * 3))

        net_bias = 3.5 * mag + 1.2 * (unit_pressure_pos - unit_pressure_neg) + 0.4 * (two_lit_pos - two_lit_neg)

        feats = torch.zeros((B, n, feat_dim), dtype=dtype, device=device)
        # Slots 0, 1: local log-odds for token 1 (False) and token 2 (True)
        feats[:, :, 0] = -net_bias
        feats[:, :, 1] = net_bias
        feats[:, :, 2] = clamped_valid.to(dtype)
        feats[:, :, 3] = clamped_spin
        feats[:, :, 4] = pos_deg / 15.0
        feats[:, :, 5] = neg_deg / 15.0
        feats[:, :, 6] = (pos_deg - neg_deg) / 10.0
        feats[:, :, 7] = unit_pressure_pos / 5.0
        feats[:, :, 8] = unit_pressure_neg / 5.0
        feats[:, :, 9] = two_lit_pos / 5.0
        feats[:, :, 10] = two_lit_neg / 5.0
        feats[:, :, 11] = mag.abs()
        return feats

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 100_000, "cal": 200_000, "test": 300_000}.get(split, 400_000)
        rng = random.Random(seed * 10009 + split_offset + self.num_vars * 31)

        B = batch_size
        n = self.num_vars
        m = self.num_clauses

        obs_tokens = np.zeros((B, n), dtype=np.int64)
        obs_mask = np.zeros((B, n), dtype=bool)
        target_tokens = np.zeros((B, n), dtype=np.int64)
        exact_marginals = np.zeros((B, n, 2), dtype=np.float32)
        clause_edges = np.zeros((B, m, 3, 2), dtype=np.int64)
        query_pairs = np.zeros((B, 8, 3), dtype=np.int64)
        query_targets = np.zeros((B, 8), dtype=np.float32)

        for b in range(B):
            c_edges, sol_toks = self._generate_hard_phase_transition_3sat(n, m, rng)
            clause_edges[b] = c_edges
            target_tokens[b] = sol_toks
            exact_marginals[b] = np.eye(2, dtype=np.float32)[sol_toks - 1]

            # Anchor 1 variable symmetry breaker so the backbone polarity is well-defined
            obs_tokens[b, 0] = sol_toks[0]
            obs_mask[b, 0] = True

            for q in range(8):
                va, vb = rng.sample(range(1, n), 2)
                query_pairs[b, q] = (va, vb, int(QueryType.EQUAL))
                query_targets[b, q] = 1.0 if sol_toks[va] == sol_toks[vb] else 0.0

        obs_t = torch.as_tensor(obs_tokens, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)
        clauses_t = torch.as_tensor(clause_edges, dtype=torch.long, device=dev)
        obs_features = self.compute_observable_sat_features(obs_t, mask_t, clauses_t, self.feat_dim)
        header = self.build_batch_header(B, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=obs_t,
            obs_mask=mask_t,
            obs_features=obs_features,
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            meta={"clause_edges": clauses_t},
        )

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        nxt = super().step_unmask(batch, var_idx, val_idx)
        if "clause_edges" in nxt.meta:
            nxt.obs_features = self.compute_observable_sat_features(
                nxt.obs_tokens,
                nxt.obs_mask,
                nxt.meta["clause_edges"],
                self.feat_dim,
            )
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        solved, _ = verify_satlib_3sat(batch.meta["clause_edges"], candidate_tokens)
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        return solved & matches_obs.all(dim=-1)


# ==============================================================================
# 5. Environment 3: `SokobanBoxPushEnv` (`Searchformer` Box-Pushing with Deadlocks)
# ==============================================================================


def solve_sokoban_astar(
    grid_code: np.ndarray,
    max_expansions: int = 2500,
) -> Dict[str, Any]:
    """Exact A* solver for `Sokoban-BoxPush` producing optimal plan + Searchformer trace tokens.

    Records `<create>` and `<close>` node operations for training `Searchformer_Distill`
    and `SoS_StreamOfSearch`, and returns exact optimal plan length `len(plan)`.
    """
    H, W = grid_code.shape
    walls = set()
    targets = set()
    boxes_init = []
    player_init = (0, 0)

    for r in range(H):
        for c in range(W):
            v = int(grid_code[r, c])
            if v == 1:
                walls.add((r, c))
            elif v == 2:
                targets.add((r, c))
            elif v == 3:
                boxes_init.append((r, c))
            elif v == 4:
                targets.add((r, c))
                boxes_init.append((r, c))
            elif v == 5:
                player_init = (r, c)
            elif v == 6:
                targets.add((r, c))
                player_init = (r, c)

    targets_list = sorted(targets)
    start_boxes = tuple(sorted(boxes_init))

    def heuristic(b_tup: Tuple[Tuple[int, int], ...], p_pos: Tuple[int, int]) -> int:
        if not targets_list:
            return 0
        h_box = sum(min(abs(bx[0] - tg[0]) + abs(bx[1] - tg[1]) for tg in targets_list) for bx in b_tup)
        unplaced = [bx for bx in b_tup if bx not in targets]
        h_plr = (min(abs(p_pos[0] - bx[0]) + abs(p_pos[1] - bx[1]) for bx in unplaced) - 1) if unplaced else 0
        return h_box * 2 + max(0, h_plr)

    def is_deadlock(br: int, bc: int) -> bool:
        if (br, bc) in targets:
            return False
        up_w = (br - 1, bc) in walls
        dn_w = (br + 1, bc) in walls
        lt_w = (br, bc - 1) in walls
        rt_w = (br, bc + 1) in walls
        return (up_w or dn_w) and (lt_w or rt_w)

    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    start_state = (player_init, start_boxes)
    pq: List[Tuple[int, int, Tuple[int, int], Tuple[Tuple[int, int], ...], List[int]]] = [
        (heuristic(start_boxes, player_init), 0, player_init, start_boxes, [])
    ]
    best_g = {start_state: 0}
    trace_events: List[Tuple[str, int, int, int]] = []
    nodes_expanded = 0

    while pq and nodes_expanded < max_expansions:
        f_val, g_val, plr, bxs, path = heapq.heappop(pq)
        if set(bxs) == targets:
            return {
                "solvable": True,
                "optimal_plan": path,
                "optimal_length": len(path),
                "nodes_expanded": nodes_expanded,
                "trace_events": trace_events,
            }
        nodes_expanded += 1
        trace_events.append(("close", plr[0], plr[1], g_val))

        b_set = set(bxs)
        for act_id, (dr, dc) in enumerate(deltas):
            nr, nc = plr[0] + dr, plr[1] + dc
            if (nr, nc) in walls or not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in b_set:
                br, bc = nr + dr, nc + dc
                if (br, bc) in walls or (br, bc) in b_set or not (0 <= br < H and 0 <= bc < W):
                    continue
                if is_deadlock(br, bc):
                    continue
                nxt_boxes = tuple(sorted((b_set - {(nr, nc)}) | {(br, bc)}))
            else:
                nxt_boxes = bxs

            nxt_state = ((nr, nc), nxt_boxes)
            ng = g_val + 1
            if ng < best_g.get(nxt_state, 1_000_000):
                best_g[nxt_state] = ng
                nf = ng + heuristic(nxt_boxes, (nr, nc))
                heapq.heappush(pq, (nf, ng, (nr, nc), nxt_boxes, path + [act_id]))
                trace_events.append(("create", nr, nc, act_id))

    return {
        "solvable": False,
        "optimal_plan": [],
        "optimal_length": 0,
        "nodes_expanded": nodes_expanded,
        "trace_events": trace_events,
    }


class SokobanBoxPushEnv(UnifiedGameEnv):
    """`Sokoban-BoxPush` (`Searchformer` Lehnert et al. 2024) Planning Environment.

    Represents a `plan_horizon = 16` step action sequence over `{1:Up, 2:Down, 3:Left, 4:Right}`
    on `6x6` box-pushing mazes with 2 boxes, internal obstacles, and corner deadlocks.
    `meta["clause_edges"]` (public) stores the `6x6` initial grid + A* trace statistics.
    """

    game_family = GameFamily.TACTICAL_PERFECT_INFO
    game_id = GameID.HEX_5X5
    grid_h = 6
    grid_w = 6
    inventory_spec = 206
    default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH
    num_vars = 14
    domain_size = 4  # 1: Up, 2: Down, 3: Left, 4: Right

    def __init__(self, plan_horizon: int = 14, num_boxes: int = 2):
        self.num_vars = int(plan_horizon)
        self.num_boxes = int(num_boxes)
        self.feat_dim = 16

    def _generate_solvable_maze(self, rng: random.Random) -> Tuple[np.ndarray, List[int], int]:
        """Generate a `6x6` Sokoban maze with 2 boxes and exact A* plan length in `6..14`."""
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
        while True:
            grid = np.zeros((6, 6), dtype=np.int64)
            grid[0, :] = 1
            grid[5, :] = 1
            grid[:, 0] = 1
            grid[:, 5] = 1

            # Add 1 or 2 internal wall pillars to create deadlock corners and narrow corridors
            inner_cells = [(r, c) for r in range(1, 5) for c in range(1, 5)]
            for wr, wc in rng.sample(inner_cells, rng.randint(1, 2)):
                grid[wr, wc] = 1

            floor_cells = [(r, c) for r in range(1, 5) for c in range(1, 5) if grid[r, c] == 0]
            if len(floor_cells) < 7:
                continue

            # Place 2 boxes on non-corner interior cells and simulate reverse pulls/pushes
            non_corner = [
                (r, c)
                for (r, c) in floor_cells
                if not ((grid[r - 1, c] == 1 or grid[r + 1, c] == 1) and (grid[r, c - 1] == 1 or grid[r, c + 1] == 1))
            ]
            if len(non_corner) < 3:
                continue

            chosen = rng.sample(non_corner, 3)
            b1, b2, plr = chosen[0], chosen[1], chosen[2]
            cur_boxes = {b1, b2}
            cur_plr = plr

            # Simulate 10..18 valid forward pushes from `(plr, cur_boxes)` to reach target configuration
            pushes_made = 0
            for _ in range(28):
                act = rng.randint(0, 3)
                dr, dc = deltas[act]
                nr, nc = cur_plr[0] + dr, cur_plr[1] + dc
                if grid[nr, nc] == 1:
                    continue
                if (nr, nc) in cur_boxes:
                    br, bc = nr + dr, nc + dc
                    if grid[br, bc] == 1 or (br, bc) in cur_boxes:
                        continue
                    # Avoid pushing into a corner unless it becomes the final target
                    if (grid[br - 1, bc] == 1 or grid[br + 1, bc] == 1) and (grid[br, bc - 1] == 1 or grid[br, bc + 1] == 1):
                        continue
                    cur_boxes.remove((nr, nc))
                    cur_boxes.add((br, bc))
                    cur_plr = (nr, nc)
                    pushes_made += 1
                else:
                    cur_plr = (nr, nc)
                if pushes_made >= rng.randint(3, 6):
                    break

            if pushes_made < 2 or cur_boxes == {b1, b2}:
                continue

            # Construct initial grid_code: targets = cur_boxes, initial boxes = {b1, b2}, player = plr
            out_grid = grid.copy()
            for tr, tc in cur_boxes:
                out_grid[tr, tc] = 2
            for br, bc in (b1, b2):
                out_grid[br, bc] = 4 if (br, bc) in cur_boxes else 3
            out_grid[plr[0], plr[1]] = 6 if plr in cur_boxes else 5

            res = solve_sokoban_astar(out_grid, max_expansions=800)
            if res["solvable"] and 5 <= res["optimal_length"] <= self.num_vars:
                plan = list(res["optimal_plan"])
                # Pad plan to `self.num_vars` with no-op / goal-preserving actions
                while len(plan) < self.num_vars:
                    plan.append(plan[-1])
                return out_grid, plan[: self.num_vars], int(res["optimal_length"])

    @torch.no_grad()
    def _compute_observable_sokoban_features(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        packed_grids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute strictly observable step-by-step simulation features `[B, L, 16]`.

        Simulates the currently clamped action prefix `obs_tokens * obs_mask` on the public
        initial grid `packed_grids` (`[B, 36]`) and computes exact 1-step lookahead box-to-target
        Manhattan distance reductions and deadlock indicators for each action `1..4`!
        Never reads `target_tokens` or `exact_marginals`.
        """
        B, L = obs_tokens.shape
        device = obs_tokens.device
        dtype = torch.float32
        feats = torch.zeros((B, L, self.feat_dim), dtype=dtype, device=device)

        grids_np = packed_grids[:, :36].detach().cpu().numpy().reshape(B, 6, 6)
        toks_np = obs_tokens.detach().cpu().numpy()
        mask_np = obs_mask.detach().cpu().numpy()
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))

        feat_np = np.zeros((B, L, self.feat_dim), dtype=np.float32)

        for b in range(B):
            g0 = grids_np[b]
            walls = set()
            targets = set()
            boxes = set()
            plr = (1, 1)
            for r in range(6):
                for c in range(6):
                    v = int(g0[r, c])
                    if v == 1:
                        walls.add((r, c))
                    elif v == 2:
                        targets.add((r, c))
                    elif v == 3:
                        boxes.add((r, c))
                    elif v == 4:
                        targets.add((r, c))
                        boxes.add((r, c))
                    elif v == 5:
                        plr = (r, c)
                    elif v == 6:
                        targets.add((r, c))
                        plr = (r, c)

            t_list = sorted(targets)

            def box_dist(b_set: set, p_cur: Tuple[int, int]) -> float:
                if not t_list:
                    return 0.0
                d_b = sum(min(abs(bx[0] - tg[0]) + abs(bx[1] - tg[1]) for tg in t_list) for bx in b_set)
                unp = [bx for bx in b_set if bx not in targets]
                d_p = min((abs(p_cur[0] - bx[0]) + abs(p_cur[1] - bx[1]) for bx in unp), default=1) - 1
                return float(2.5 * d_b + 0.6 * max(0, d_p))

            for t_step in range(L):
                cur_d = box_dist(boxes, plr)
                # 2-ply lookahead score for each action a in 0..3 from current simulated prefix state
                for a_id, (dr, dc) in enumerate(deltas):
                    nr, nc = plr[0] + dr, plr[1] + dc
                    if (nr, nc) in walls:
                        feat_np[b, t_step, a_id] = -5.0
                        continue
                    nxt_boxes = set(boxes)
                    dead = False
                    if (nr, nc) in nxt_boxes:
                        br, bc = nr + dr, nc + dc
                        if (br, bc) in walls or (br, bc) in nxt_boxes:
                            feat_np[b, t_step, a_id] = -5.0
                            continue
                        nxt_boxes.remove((nr, nc))
                        nxt_boxes.add((br, bc))
                        if (br, bc) not in targets:
                            up_w = (br - 1, bc) in walls
                            dn_w = (br + 1, bc) in walls
                            lt_w = (br, bc - 1) in walls
                            rt_w = (br, bc + 1) in walls
                            if (up_w or dn_w) and (lt_w or rt_w):
                                dead = True
                    if dead:
                        feat_np[b, t_step, a_id] = -6.0
                    else:
                        d1 = box_dist(nxt_boxes, (nr, nc))
                        # Best 2nd ply from (nr, nc)
                        best_d2 = d1
                        for dr2, dc2 in deltas:
                            n2r, n2c = nr + dr2, nc + dc2
                            if (n2r, n2c) in walls:
                                continue
                            b2_set = set(nxt_boxes)
                            if (n2r, n2c) in b2_set:
                                b2r, b2c = n2r + dr2, n2c + dc2
                                if (b2r, b2c) in walls or (b2r, b2c) in b2_set:
                                    continue
                                if (b2r, b2c) not in targets and (
                                    ((b2r - 1, b2c) in walls or (b2r + 1, b2c) in walls)
                                    and ((b2r, b2c - 1) in walls or (b2r, b2c + 1) in walls)
                                ):
                                    continue
                                b2_set.remove((n2r, n2c))
                                b2_set.add((b2r, b2c))
                            best_d2 = min(best_d2, box_dist(b2_set, (n2r, n2c)))
                        gain = (cur_d - 0.5 * (d1 + best_d2))
                        feat_np[b, t_step, a_id] = float(2.2 * gain + (3.0 if nxt_boxes == targets else 0.0))

                feat_np[b, t_step, 4] = float(t_step) / float(max(1, L - 1))
                feat_np[b, t_step, 5] = float(plr[0]) / 5.0
                feat_np[b, t_step, 6] = float(plr[1]) / 5.0
                feat_np[b, t_step, 7] = float(cur_d) / 15.0
                feat_np[b, t_step, 8] = 1.0 if boxes == targets else 0.0

                # Advance simulated state if this step is clamped in obs_mask
                if mask_np[b, t_step] and 1 <= toks_np[b, t_step] <= 4:
                    act_c = int(toks_np[b, t_step]) - 1
                    dr, dc = deltas[act_c]
                    nr, nc = plr[0] + dr, plr[1] + dc
                    if (nr, nc) not in walls:
                        if (nr, nc) in boxes:
                            br, bc = nr + dr, nc + dc
                            if (br, bc) not in walls and (br, bc) not in boxes:
                                boxes.remove((nr, nc))
                                boxes.add((br, bc))
                                plr = (nr, nc)
                        else:
                            plr = (nr, nc)

        feats.copy_(torch.as_tensor(feat_np, dtype=dtype, device=device))
        return feats

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 100_000, "cal": 200_000, "test": 300_000}.get(split, 400_000)
        rng = random.Random(seed * 10037 + split_offset + 606)

        B = batch_size
        L = self.num_vars

        obs_tokens = np.zeros((B, L), dtype=np.int64)
        obs_mask = np.zeros((B, L), dtype=bool)
        target_tokens = np.zeros((B, L), dtype=np.int64)
        exact_marginals = np.zeros((B, L, 4), dtype=np.float32)
        packed_meta = np.zeros((B, 38), dtype=np.int64)  # 36 grid cells + opt_len + astar_nodes
        query_pairs = np.zeros((B, 4, 3), dtype=np.int64)
        query_targets = np.zeros((B, 4), dtype=np.float32)

        for b in range(B):
            grid, plan, opt_len = self._generate_solvable_maze(rng)
            plan_1idx = np.array([a + 1 for a in plan], dtype=np.int64)
            target_tokens[b] = plan_1idx
            exact_marginals[b] = np.eye(4, dtype=np.float32)[plan_1idx - 1]
            packed_meta[b, :36] = grid.reshape(36)
            packed_meta[b, 36] = opt_len
            packed_meta[b, 37] = opt_len * 3

            for q in range(4):
                sa, sb = rng.sample(range(L), 2)
                query_pairs[b, q] = (sa, sb, int(QueryType.EQUAL))
                query_targets[b, q] = 1.0 if plan_1idx[sa] == plan_1idx[sb] else 0.0

        obs_t = torch.as_tensor(obs_tokens, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)
        meta_t = torch.as_tensor(packed_meta, dtype=torch.long, device=dev)
        obs_features = self._compute_observable_sokoban_features(obs_t, mask_t, meta_t)
        header = self.build_batch_header(B, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=obs_t,
            obs_mask=mask_t,
            obs_features=obs_features,
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            meta={"clause_edges": meta_t},
        )

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        nxt = super().step_unmask(batch, var_idx, val_idx)
        if "clause_edges" in nxt.meta:
            nxt.obs_features = self._compute_observable_sokoban_features(
                nxt.obs_tokens,
                nxt.obs_mask,
                nxt.meta["clause_edges"],
            )
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B = candidate_tokens.shape[0]
        device = candidate_tokens.device
        grids_np = batch.meta["clause_edges"][:, :36].detach().cpu().numpy().reshape(B, 6, 6)
        cands_np = candidate_tokens.detach().cpu().numpy()
        out = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            res = verify_sokoban_plan(grids_np[b], cands_np[b].tolist())
            out[b] = bool(res["solved"])
        return out


# ==============================================================================
# 6. Environment 4: `CountdownArithmeticEnv` (`Stream of Search` Target Synthesis)
# ==============================================================================


# Canonical enumeration of all 24 pair-operation choices `(i, j, op)` on a pool of size `<= 4`:
# At step s in {0, 1, 2}, pool has size `4 - s`. We map token `z in 1..12` to `(pair_idx, op_id)`
# where `pair_idx in {0..2}` selects one of the unordered pairs and `op_id in {0:+, 1:-, 2:*, 3:/}`.
COUNTDOWN_PAIRS_BY_STEP: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),  # Step 0 (4 numbers -> 6 pairs, top 3 active + 4 ops)
    ((0, 1), (0, 2), (1, 2)),                          # Step 1 (3 numbers -> 3 pairs x 4 ops = 12)
    ((0, 1), (0, 1), (0, 1)),                          # Step 2 (2 numbers -> 1 pair x 4 ops)
)


def decode_countdown_tokens_to_steps(
    tokens_1idx: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Decode a 3-step token vector `tokens_1idx` (`z in 1..9`) into `[(ia, op, ib), ...]`."""
    steps: List[Tuple[int, int, int]] = []
    for s_idx, tok in enumerate(tokens_1idx[:3]):
        z = max(0, int(tok) - 1)
        pairs = COUNTDOWN_PAIRS_BY_STEP[min(s_idx, 2)]
        p_idx = (z // 3) % len(pairs)
        op_id = z % 3  # 0: +, 1: -, 2: * (plus exact / when applicable)
        ia, ib = pairs[p_idx]
        steps.append((ia, op_id, ib))
    return steps


class CountdownArithmeticEnv(UnifiedGameEnv):
    """`Countdown-Arithmetic` (`Stream of Search` Gandhi et al. 2024) Environment.

    Each instance has 4 input integers `[x_0, x_1, x_2, x_3]` and a target integer `T in [15, 650]`.
    `num_vars = 3` sequential reduction steps, `domain_size = 9` (`3` canonical pair choices x `3` arithmetic ops).
    Filtered so 1-step greedy proximity fails on most instances, requiring multi-step lookahead.
    """

    game_family = GameFamily.CONSTRAINT_CSP
    game_id = GameID.SHIPHUNT_6X6
    grid_h = 4
    grid_w = 3
    inventory_spec = 409
    default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH
    num_vars = 3
    domain_size = 9

    def __init__(self):
        self.feat_dim = 16
        self.number_pool = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25)

    def _generate_countdown_instance(self, rng: random.Random) -> Tuple[List[int], int, List[int]]:
        """Generate `(nums_4, target_T, solution_tokens_3)` requiring multi-step lookahead."""
        while True:
            nums = sorted(rng.sample(self.number_pool, 4))
            sol_toks = [rng.randint(1, 9) for _ in range(3)]
            steps = decode_countdown_tokens_to_steps(sol_toks)

            # Simulate forward to find the resulting target T
            pool = list(nums)
            ok = True
            for ia, op, ib in steps:
                if ia >= len(pool) or ib >= len(pool) or ia == ib:
                    ok = False
                    break
                a, b = pool[ia], pool[ib]
                if op == 0:
                    res = a + b
                elif op == 1:
                    if a <= b:
                        ok = False
                        break
                    res = a - b
                else:
                    res = a * b
                pool = [pool[k] for k in range(len(pool)) if k != ia and k != ib] + [res]
            if not ok or len(pool) != 1:
                continue
            target = pool[0]
            if not (18 <= target <= 650) or target in nums:
                continue

            # Check whether naive 1-step greedy (picking step 0 with value closest to target) solves it;
            # prefer instances where intermediate values non-monotonically overshoot/undershoot
            best_g_tok = 1
            best_g_gap = 10**9
            for z0 in range(1, 10):
                ia, op, ib = decode_countdown_tokens_to_steps([z0, 1, 1])[0]
                a, b = nums[ia], nums[ib]
                val = (a + b) if op == 0 else (abs(a - b) if op == 1 else a * b)
                if abs(val - target) < best_g_gap:
                    best_g_gap = abs(val - target)
                    best_g_tok = z0

            if best_g_tok != sol_toks[0] or rng.random() < 0.25:
                return nums, target, sol_toks

    @torch.no_grad()
    def _compute_observable_countdown_features(
        self,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        packed_nums_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute strictly observable arithmetic lookahead features `[B, 3, 16]`."""
        B = obs_tokens.shape[0]
        device = obs_tokens.device
        dtype = torch.float32

        p_np = packed_nums_target.detach().cpu().numpy()
        toks_np = obs_tokens.detach().cpu().numpy()
        mask_np = obs_mask.detach().cpu().numpy()
        feat_np = np.zeros((B, 3, self.feat_dim), dtype=np.float32)

        for b in range(B):
            nums = [int(p_np[b, k]) for k in range(4)]
            target = int(p_np[b, 4])

            # Evaluate reachability score for each candidate token z in 1..9 at each step s in 0..2
            cur_pool = list(nums)
            for s_idx in range(3):
                pairs = COUNTDOWN_PAIRS_BY_STEP[s_idx]
                for z in range(9):
                    p_idx = (z // 3) % len(pairs)
                    op = z % 3
                    ia, ib = pairs[p_idx]
                    if ia >= len(cur_pool) or ib >= len(cur_pool) or ia == ib:
                        feat_np[b, s_idx, z] = -4.0
                        continue
                    a, b_val = cur_pool[ia], cur_pool[ib]
                    if op == 1 and a == b_val:
                        feat_np[b, s_idx, z] = -3.0
                        continue
                    res = (a + b_val) if op == 0 else (abs(a - b_val) if op == 1 else a * b_val)
                    nxt_pool = [cur_pool[k] for k in range(len(cur_pool)) if k != ia and k != ib] + [res]

                    # Compute min relative error achievable from nxt_pool in remaining steps
                    if s_idx == 2:
                        min_gap = abs(res - target)
                    elif s_idx == 1:
                        # 1 step left on 2 numbers in nxt_pool
                        x0, x1 = nxt_pool[0], nxt_pool[1]
                        cands = [x0 + x1, abs(x0 - x1), x0 * x1]
                        min_gap = min(abs(c - target) for c in cands)
                    else:
                        # 2 steps left on 3 numbers in nxt_pool: check all 3 pairs x 3 ops -> final 3 ops
                        min_gap = abs(res - target)
                        for p1_a, p1_b in COUNTDOWN_PAIRS_BY_STEP[1]:
                            u, v = nxt_pool[p1_a], nxt_pool[p1_b]
                            rem = nxt_pool[3 - p1_a - p1_b]
                            for r1 in (u + v, abs(u - v), u * v):
                                for r2 in (r1 + rem, abs(r1 - rem), r1 * rem):
                                    min_gap = min(min_gap, abs(r2 - target))
                                    if min_gap == 0:
                                        break

                    if min_gap == 0:
                        feat_np[b, s_idx, z] = 4.5
                    else:
                        feat_np[b, s_idx, z] = float(-1.8 * math.log1p(min_gap))

                feat_np[b, s_idx, 9] = float(target) / 500.0
                for k_i in range(min(4, len(cur_pool))):
                    feat_np[b, s_idx, 10 + k_i] = float(cur_pool[k_i]) / 50.0
                feat_np[b, s_idx, 14] = float(s_idx) / 2.0

                if mask_np[b, s_idx] and 1 <= toks_np[b, s_idx] <= 9:
                    z_c = int(toks_np[b, s_idx]) - 1
                    p_idx = (z_c // 3) % len(pairs)
                    op = z_c % 3
                    ia, ib = pairs[p_idx]
                    if ia < len(cur_pool) and ib < len(cur_pool) and ia != ib:
                        a, b_val = cur_pool[ia], cur_pool[ib]
                        res = (a + b_val) if op == 0 else (abs(a - b_val) if op == 1 else a * b_val)
                        cur_pool = [cur_pool[k] for k in range(len(cur_pool)) if k != ia and k != ib] + [res]

        return torch.as_tensor(feat_np, dtype=dtype, device=device)

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 100_000, "cal": 200_000, "test": 300_000}.get(split, 400_000)
        rng = random.Random(seed * 10061 + split_offset + 405)

        B = batch_size
        obs_tokens = np.zeros((B, 3), dtype=np.int64)
        obs_mask = np.zeros((B, 3), dtype=bool)
        target_tokens = np.zeros((B, 3), dtype=np.int64)
        exact_marginals = np.zeros((B, 3, 9), dtype=np.float32)
        packed_meta = np.zeros((B, 5), dtype=np.int64)  # [n0, n1, n2, n3, target_T]
        query_pairs = np.zeros((B, 2, 3), dtype=np.int64)
        query_targets = np.zeros((B, 2), dtype=np.float32)

        for b in range(B):
            nums, target, sol_toks = self._generate_countdown_instance(rng)
            target_tokens[b] = np.array(sol_toks, dtype=np.int64)
            exact_marginals[b] = np.eye(9, dtype=np.float32)[np.array(sol_toks) - 1]
            packed_meta[b, :4] = nums
            packed_meta[b, 4] = target
            query_pairs[b, 0] = (0, 1, int(QueryType.EQUAL))
            query_targets[b, 0] = 1.0 if sol_toks[0] == sol_toks[1] else 0.0
            query_pairs[b, 1] = (1, 2, int(QueryType.EQUAL))
            query_targets[b, 1] = 1.0 if sol_toks[1] == sol_toks[2] else 0.0

        obs_t = torch.as_tensor(obs_tokens, dtype=torch.long, device=dev)
        mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)
        meta_t = torch.as_tensor(packed_meta, dtype=torch.long, device=dev)
        obs_features = self._compute_observable_countdown_features(obs_t, mask_t, meta_t)
        header = self.build_batch_header(B, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=obs_t,
            obs_mask=mask_t,
            obs_features=obs_features,
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            meta={"clause_edges": meta_t},
        )

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        nxt = super().step_unmask(batch, var_idx, val_idx)
        if "clause_edges" in nxt.meta:
            nxt.obs_features = self._compute_observable_countdown_features(
                nxt.obs_tokens,
                nxt.obs_mask,
                nxt.meta["clause_edges"],
            )
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B = candidate_tokens.shape[0]
        device = candidate_tokens.device
        meta_np = batch.meta["clause_edges"].detach().cpu().numpy()
        cands_np = candidate_tokens.detach().cpu().numpy()
        out = torch.zeros(B, dtype=torch.bool, device=device)
        for b in range(B):
            nums = meta_np[b, :4].tolist()
            target = int(meta_np[b, 4])
            steps = decode_countdown_tokens_to_steps(cands_np[b].tolist())
            res = verify_countdown_expression(nums, target, steps)
            out[b] = bool(res["solved"])
        return out
