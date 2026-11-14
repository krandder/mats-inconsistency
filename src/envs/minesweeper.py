"""6x6 Epistemic Minesweeper Environment (`src/envs/minesweeper.py`).

Tests:
1. Multi-step overlap/subset deduction (identifying forced-safe `P*(mine) = 0` and forced-mine `P*(mine) = 1`
   cells along the clue frontier).
2. Epistemic calibration & self-trustworthiness (`theta_trust`): distinguishing irreducible 50/50 risk on
   ambiguous frontier cells from deterministic deductions.
3. Safe survival action selection (`UtilitySpec.SAFE_SURVIVAL`): choosing the minimum-risk unrevealed cell
   when forced moves are exhausted.

Computes the 100% exact Bayesian posterior `P*(Z_i = 1 | clues, M)` over all `C(36, M)` boards in sub-millisecond
time by combining exact frontier constraint DFS with exact interior combinatorics `C(|I|, M - m_F)`.
"""

from __future__ import annotations

import math
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


def _moore_neighbors(size: int) -> List[List[int]]:
    nbrs = []
    for r in range(size):
        for c in range(size):
            cur = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        cur.append(nr * size + nc)
            nbrs.append(cur)
    return nbrs


class MinesweeperEnv(UnifiedGameEnv):
    """6x6 Epistemic Minesweeper with exact frontier-DFS + combinatorics Bayesian oracle."""

    game_family = GameFamily.EPISTEMIC_PROBE
    game_id = GameID.MINESWEEPER_6X6

    def __init__(
        self,
        size: int = 6,
        num_mines: int = 6,
        num_revealed_clues: int = 10,
        num_queries: int = 8,
        utility_spec: UtilitySpec = UtilitySpec.SAFE_SURVIVAL,
    ):
        self.size = int(size)
        self.num_mines = int(num_mines)
        self.num_revealed_clues = int(num_revealed_clues)
        self.num_queries = int(num_queries)

        self.grid_h = self.size
        self.grid_w = self.size
        self.inventory_spec = self.num_mines
        self.default_utility_spec = utility_spec

        self.num_vars = self.size * self.size
        self.domain_size = 2  # token 1 = safe (0), token 2 = mine (1)
        # Per-cell features: [row_norm, col_norm, is_revealed_safe, clue_count_norm, clue_one_hot(0..8)] -> 13 dims
        self.feat_dim = 13
        self.neighbors = _moore_neighbors(self.size)

    def _solve_exact_posterior(
        self,
        revealed_mask: np.ndarray,
        clue_counts: np.ndarray,
        clamped_mine_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Compute exact per-cell mine probabilities `[N]` and pairwise equality `P*(Z_a == Z_b)` `[N, N]`.

        Args:
            revealed_mask: `[N]` bool array of revealed safe cells (which carry clue numbers).
            clue_counts: `[N]` int array of Moore-neighborhood mine counts at `revealed_mask` cells.
            clamped_mine_mask: Optional `[N]` bool array of cells clamped as known mines during search.
        """
        N = self.num_vars
        M = self.num_mines
        if clamped_mine_mask is None:
            clamped_mine_mask = np.zeros(N, dtype=bool)

        # Revealed safe cells cannot be mines
        if np.any(revealed_mask & clamped_mine_mask):
            return np.zeros(N, dtype=np.float32), np.eye(N, dtype=np.float32), False

        clues = [i for i in range(N) if revealed_mask[i]]
        # Unresolved cells: neither revealed safe nor clamped mine
        unresolved_set = set(range(N)) - set(clues) - set(np.where(clamped_mine_mask)[0].tolist())
        m_clamped = int(clamped_mine_mask.sum())
        if m_clamped > M:
            return np.zeros(N, dtype=np.float32), np.eye(N, dtype=np.float32), False

        # Frontier F = unresolved cells adjacent to at least one revealed clue cell
        frontier_set = set()
        clue_reqs: List[Tuple[List[int], int]] = []
        for c in clues:
            rem_need = int(clue_counts[c]) - sum(1 for u in self.neighbors[c] if clamped_mine_mask[u])
            unres_nbrs = [u for u in self.neighbors[c] if u in unresolved_set]
            if rem_need < 0 or rem_need > len(unres_nbrs):
                return np.zeros(N, dtype=np.float32), np.eye(N, dtype=np.float32), False
            frontier_set.update(unres_nbrs)
            clue_reqs.append((unres_nbrs, rem_need))

        frontier = sorted(frontier_set)
        f_idx = {u: idx for idx, u in enumerate(frontier)}
        F_len = len(frontier)
        interior = sorted(unresolved_set - frontier_set)
        I_len = len(interior)
        rem_total_mines = M - m_clamped

        # Map each clue requirement to frontier indices
        mapped_clues = [([f_idx[u] for u in nbrs], need) for nbrs, need in clue_reqs]
        # For each frontier variable k, which clues Finish when variable k is assigned?
        var_clues: List[List[int]] = [[] for _ in range(F_len)]
        for c_i, (f_nbrs, _) in enumerate(mapped_clues):
            for fv in f_nbrs:
                var_clues[fv].append(c_i)

        valid_f_assignments: List[np.ndarray] = []
        valid_weights: List[float] = []

        cur_f = np.zeros(F_len, dtype=np.int8)
        clue_cur_sum = [0] * len(mapped_clues)
        clue_rem_vars = [len(f_nbrs) for f_nbrs, _ in mapped_clues]

        def dfs(k: int, mines_used: int) -> None:
            if mines_used > rem_total_mines:
                return
            if mines_used + (F_len - k) + I_len < rem_total_mines:
                return
            if k == F_len:
                m_int = rem_total_mines - mines_used
                if 0 <= m_int <= I_len:
                    w = float(math.comb(I_len, m_int))
                    if w > 0:
                        valid_f_assignments.append(cur_f.copy())
                        valid_weights.append(w)
                return

            for val in (0, 1):
                cur_f[k] = val
                ok = True
                for c_i in var_clues[k]:
                    clue_cur_sum[c_i] += val
                    clue_rem_vars[c_i] -= 1
                for c_i in var_clues[k]:
                    need = mapped_clues[c_i][1]
                    if clue_cur_sum[c_i] > need or (clue_cur_sum[c_i] + clue_rem_vars[c_i] < need):
                        ok = False
                        break
                if ok:
                    dfs(k + 1, mines_used + val)
                for c_i in var_clues[k]:
                    clue_cur_sum[c_i] -= val
                    clue_rem_vars[c_i] += 1

        dfs(0, 0)

        if not valid_weights:
            return np.zeros(N, dtype=np.float32), np.eye(N, dtype=np.float32), False

        W = np.array(valid_weights, dtype=np.float64)
        W /= W.sum()
        F_mat = np.array(valid_f_assignments, dtype=np.float64)  # [S, F_len]
        m_F_vec = F_mat.sum(axis=1)  # [S]
        m_I_vec = rem_total_mines - m_F_vec  # [S]

        p_mine = np.zeros(N, dtype=np.float64)
        p_mine[clamped_mine_mask] = 1.0
        p_mine[revealed_mask] = 0.0

        if F_len > 0:
            p_f = W @ F_mat
            for idx, u in enumerate(frontier):
                p_mine[u] = p_f[idx]

        p_int_per_sol = (m_I_vec / I_len) if I_len > 0 else np.zeros_like(m_I_vec)
        if I_len > 0:
            p_int = float(W @ p_int_per_sol)
            for u in interior:
                p_mine[u] = p_int

        # Compute exact pairwise equality P*(Z_a == Z_b)
        # First compute P*(Z_u = 1, Z_v = 1) for all pairs
        p_both_1 = np.outer(p_mine, p_mine)
        if F_len > 0:
            f_co = (F_mat.T * W) @ F_mat  # [F_len, F_len]
            for i1, u in enumerate(frontier):
                for i2, v in enumerate(frontier):
                    p_both_1[u, v] = f_co[i1, i2]
            if I_len > 0:
                f_i_co = (F_mat.T @ (W * p_int_per_sol))  # [F_len]
                for i1, u in enumerate(frontier):
                    for v in interior:
                        p_both_1[u, v] = f_i_co[i1]
                        p_both_1[v, u] = f_i_co[i1]
        if I_len >= 2:
            p_ii_co = float(W @ (p_int_per_sol * np.maximum(0.0, m_I_vec - 1.0) / max(1, I_len - 1)))
            for u in interior:
                for v in interior:
                    if u != v:
                        p_both_1[u, v] = p_ii_co
        for u in range(N):
            p_both_1[u, u] = p_mine[u]

        # P(Z_a == Z_b) = P(1, 1) + P(0, 0) = 1 - P(Z_a=1) - P(Z_b=1) + 2 P(1, 1)
        pair_eq = (1.0 - p_mine[:, None] - p_mine[None, :] + 2.0 * p_both_1).clip(0.0, 1.0)
        return p_mine.astype(np.float32), pair_eq.astype(np.float32), True

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        utility_spec: Optional[Union[UtilitySpec, int]] = None,
        **kwargs: Any,
    ) -> EnvBatch:
        dev = torch.device(device)
        uspec = self.default_utility_spec if utility_spec is None else utility_spec
        split_offset = {"train": 0, "val": 10_000, "cal": 20_000, "test": 30_000}.get(split, 40_000)
        rng = random.Random(seed * 1013 + split_offset)

        B = batch_size
        N = self.num_vars
        M = self.num_mines
        Q = self.num_queries

        obs_tokens = np.zeros((B, N), dtype=np.int64)
        obs_mask = np.zeros((B, N), dtype=bool)
        target_tokens = np.zeros((B, N), dtype=np.int64)
        exact_marginals = np.zeros((B, N, 2), dtype=np.float32)
        obs_features = np.zeros((B, N, self.feat_dim), dtype=np.float32)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        clue_Arr = np.full((B, N), -1, dtype=np.int64)

        rows_norm = np.array([i // self.size / (self.size - 1) for i in range(N)], dtype=np.float32)
        cols_norm = np.array([(i % self.size) / (self.size - 1) for i in range(N)], dtype=np.float32)

        for b in range(B):
            mine_cells = set(rng.sample(range(N), M))
            mines = np.array([1 if i in mine_cells else 0 for i in range(N)], dtype=np.int64)
            target_tokens[b] = mines + 1  # 1 = safe, 2 = mine

            full_clues = np.array(
                [sum(mines[u] for u in self.neighbors[i]) for i in range(N)],
                dtype=np.int64,
            )
            safe_cells = [i for i in range(N) if i not in mine_cells]
            rev_cells = rng.sample(safe_cells, min(self.num_revealed_clues, len(safe_cells)))
            rev_mask = np.zeros(N, dtype=bool)
            rev_mask[rev_cells] = True

            obs_mask[b] = rev_mask
            obs_tokens[b, rev_mask] = 1  # known safe
            clue_Arr[b, rev_mask] = full_clues[rev_mask]

            p_mine, pair_eq, _ = self._solve_exact_posterior(rev_mask, full_clues)
            exact_marginals[b, :, 0] = 1.0 - p_mine
            exact_marginals[b, :, 1] = p_mine

            obs_features[b, :, 0] = rows_norm
            obs_features[b, :, 1] = cols_norm
            obs_features[b, :, 2] = rev_mask.astype(np.float32)
            for c in rev_cells:
                k = int(full_clues[c])
                obs_features[b, c, 3] = k / 8.0
                obs_features[b, c, 4 + min(k, 8)] = 1.0

            unrev = [i for i in range(N) if not rev_mask[i]]
            for q in range(Q):
                ca, cb = rng.sample(unrev, 2)
                query_pairs[b, q] = (ca, cb, int(QueryType.EQUAL))
                query_targets[b, q] = float(pair_eq[ca, cb])

        header = self.build_batch_header(B, utility_spec=uspec, device=dev)
        exact_marg_t = torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev)
        obs_mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)

        # Legal actions: choosing any unrevealed cell to probe/step on
        legal_action_mask = ~obs_mask_t
        # Expected survival utility of probing cell i: +1 if safe, -5 if mine -> 1*(1 - p_mine) - 5*p_mine
        p_mine_t = exact_marg_t[:, :, 1]
        utility_vec = (1.0 - 6.0 * p_mine_t).masked_fill(obs_mask_t, -1e4)

        return EnvBatch(
            header=header,
            obs_tokens=torch.as_tensor(obs_tokens, dtype=torch.long, device=dev),
            obs_mask=obs_mask_t,
            obs_features=torch.as_tensor(obs_features, dtype=torch.float32, device=dev),
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=exact_marg_t,
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            legal_action_mask=legal_action_mask,
            utility_vector=utility_vec,
            meta={"clue_counts": torch.as_tensor(clue_Arr, dtype=torch.long, device=dev)},
        )

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify whether `candidate_tokens` (`[B, 36]` in `{1 (safe), 2 (mine)}`) matches all revealed clue sums
        and total mine count `M`."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens == 1) | (candidate_tokens == 2)
        mines = (candidate_tokens == 2).long()  # [B, N]
        count_ok = mines.sum(dim=-1) == self.num_mines

        clues = batch.meta["clue_counts"]  # [B, N], -1 where not a revealed clue
        B, N = mines.shape
        clues_ok = torch.ones(B, dtype=torch.bool, device=candidate_tokens.device)
        for i in range(N):
            active = clues[:, i] >= 0
            if active.any():
                nbr_sum = mines[:, self.neighbors[i]].sum(dim=-1)
                clues_ok = clues_ok & ((~active) | (nbr_sum == clues[:, i]))
        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & count_ok & clues_ok
