"""5x5 Hex & 6x7 Connect-4 Tactical Perfect-Information Environments (`src/envs/hex_c4.py`).

Provides:
- `HexEnv` (`GameID.HEX_5X5 = 50`, 5x5 grid = 25 cells, Red top-to-bottom vs Blue left-to-right).
- `Connect4Env` (`GameID.CONNECT4_6X7 = 51`, 6x7 grid = 42 cells, gravity-drop 4-in-a-row).
- Unified `TacticalGameEnv` wrapper (`variant="hex"` or `"connect4"`).
- Fast tactical negamax / playout oracle for ground-truth move values `Q*(s, a)`, cell ownership marginals
  `exact_marginals`, relational queries `Q(A, B)` (whether cells A and B are claimed by the same player),
  and head-to-head `play_match(agent_a, agent_b)` for Multiplayer Elo leagues across compute budgets `B`.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

# ---------------------------------------------------------------------------
# 5x5 Hex Mechanics
# ---------------------------------------------------------------------------

HEX_DIRS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


def hex_neighbors(size: int = 5) -> List[List[int]]:
    nbrs = []
    for r in range(size):
        for c in range(size):
            cur = []
            for dr, dc in HEX_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    cur.append(nr * size + nc)
            nbrs.append(cur)
    return nbrs


HEX5_NBRS = hex_neighbors(5)


def hex_check_winner(board: np.ndarray, size: int = 5) -> int:
    """Return 1 if Player 0 (Red, top->bottom) has won, 2 if Player 1 (Blue, left->right) has won, 0 otherwise."""
    nbrs = HEX5_NBRS if size == 5 else hex_neighbors(size)
    # Check Player 0 (token 1): row 0 to row size-1
    starts_0 = [c for c in range(size) if board[c] == 1]
    if starts_0:
        vis = set(starts_0)
        q = deque(starts_0)
        while q:
            u = q.popleft()
            if u // size == size - 1:
                return 1
            for v in nbrs[u]:
                if board[v] == 1 and v not in vis:
                    vis.add(v)
                    q.append(v)

    # Check Player 1 (token 2): col 0 to col size-1
    starts_1 = [r * size for r in range(size) if board[r * size] == 2]
    if starts_1:
        vis = set(starts_1)
        q = deque(starts_1)
        while q:
            u = q.popleft()
            if u % size == size - 1:
                return 2
            for v in nbrs[u]:
                if board[v] == 2 and v not in vis:
                    vis.add(v)
                    q.append(v)
    return 0


def hex_shortest_path_heuristic(board: np.ndarray, player_tok: int, size: int = 5) -> float:
    """0-1 BFS shortest bridge distance for `player_tok` (`1` = Red top-bottom, `2` = Blue left-right)."""
    nbrs = HEX5_NBRS if size == 5 else hex_neighbors(size)
    opp_tok = 3 - player_tok
    INF = 99
    dist = [INF] * (size * size)
    dq: deque[int] = deque()

    for i in range(size):
        cell = i if player_tok == 1 else (i * size)
        if board[cell] == opp_tok:
            continue
        cost = 0 if board[cell] == player_tok else 1
        dist[cell] = cost
        if cost == 0:
            dq.appendleft(cell)
        else:
            dq.append(cell)

    while dq:
        u = dq.popleft()
        if (player_tok == 1 and u // size == size - 1) or (player_tok == 2 and u % size == size - 1):
            return float(dist[u])
        for v in nbrs[u]:
            if board[v] == opp_tok:
                continue
            w = 0 if board[v] == player_tok else 1
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return float(INF)


def hex_action_values_and_marginals(
    board: np.ndarray,
    to_move_tok: int,
    rng: random.Random,
    num_rollouts: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tactical 2-ply + rollout move values `[25]`, cell ownership marginals `[25, 2]`, and sample completion."""
    N = len(board)
    empties = [i for i in range(N) if board[i] == 0]
    opp_tok = 3 - to_move_tok
    q_vals = np.full(N, -1e4, dtype=np.float32)

    if not empties:
        marg = np.zeros((N, 2), dtype=np.float32)
        marg[:, 0] = (board == 1).astype(np.float32)
        marg[:, 1] = (board == 2).astype(np.float32)
        return q_vals, marg, board.copy()

    # Score every legal cell by 2-ply shortest-path differential + winning tactic detection
    for move in empties:
        b1 = board.copy()
        b1[move] = to_move_tok
        w = hex_check_winner(b1)
        if w == to_move_tok:
            q_vals[move] = 1.0
            continue
        d_my = hex_shortest_path_heuristic(b1, to_move_tok)
        d_opp = hex_shortest_path_heuristic(b1, opp_tok)
        # Check if opponent has an immediate 1-move win reply
        opp_win = False
        for om in empties:
            if om == move:
                continue
            b1[om] = opp_tok
            if hex_check_winner(b1) == opp_tok:
                opp_win = True
                b1[om] = 0
                break
            b1[om] = 0
        if opp_win:
            q_vals[move] = -0.95
        else:
            q_vals[move] = float(np.tanh(0.45 * (d_opp - d_my)))

    # Rollout completions weighted by softmax move values to estimate future cell ownership marginals
    valid_q = np.array([q_vals[m] for m in empties], dtype=np.float64)
    probs = np.exp((valid_q - valid_q.max()) * 3.0)
    probs /= probs.sum()

    counts = np.zeros((N, 2), dtype=np.float32)
    sample_comp = board.copy()
    for r_idx in range(num_rollouts):
        b_sim = board.copy()
        first_m = empties[int(rng.choices(range(len(empties)), weights=probs.tolist(), k=1)[0])]
        b_sim[first_m] = to_move_tok
        rem = [m for m in empties if m != first_m]
        rng.shuffle(rem)
        cur = opp_tok
        for m in rem:
            b_sim[m] = cur
            cur = 3 - cur
        counts[:, 0] += (b_sim == 1).astype(np.float32)
        counts[:, 1] += (b_sim == 2).astype(np.float32)
        if r_idx == 0:
            sample_comp = b_sim

    marg = counts / float(num_rollouts)
    return q_vals, marg, sample_comp


# ---------------------------------------------------------------------------
# 6x7 Connect-4 Mechanics
# ---------------------------------------------------------------------------

C4_H, C4_W = 6, 7
C4_N = 42


def _build_c4_lines() -> List[Tuple[int, int, int, int]]:
    lines = []
    for r in range(C4_H):
        for c in range(C4_W):
            if c + 3 < C4_W:
                lines.append(tuple(r * C4_W + c + k for k in range(4)))
            if r + 3 < C4_H:
                lines.append(tuple((r + k) * C4_W + c for k in range(4)))
            if r + 3 < C4_H and c + 3 < C4_W:
                lines.append(tuple((r + k) * C4_W + c + k for k in range(4)))
            if r - 3 >= 0 and c + 3 < C4_W:
                lines.append(tuple((r - k) * C4_W + c + k for k in range(4)))
    return lines


C4_LINES = _build_c4_lines()
C4_LINES_ARR = np.array(C4_LINES, dtype=np.int64)  # [69, 4]


def c4_check_winner(board: np.ndarray) -> int:
    """Return 1 if Player 0 won, 2 if Player 1 won, 0 otherwise."""
    vals = board[C4_LINES_ARR]  # [69, 4]
    if np.any(np.all(vals == 1, axis=1)):
        return 1
    if np.any(np.all(vals == 2, axis=1)):
        return 2
    return 0


def c4_legal_cells(board: np.ndarray) -> List[int]:
    """Return the lowest empty cell index `r * 7 + c` in each non-full column `c` (row 0 = bottom)."""
    out = []
    for c in range(C4_W):
        for r in range(C4_H):
            idx = r * C4_W + c
            if board[idx] == 0:
                out.append(idx)
                break
    return out


def c4_heuristic_eval(board: np.ndarray, to_move_tok: int) -> float:
    """Fast window-threat evaluation in `[-1, 1]` from `to_move_tok`'s perspective."""
    opp_tok = 3 - to_move_tok
    vals = board[C4_LINES_ARR]
    my_cnt = (vals == to_move_tok).sum(axis=1)
    opp_cnt = (vals == opp_tok).sum(axis=1)
    my_open = (opp_cnt == 0)
    opp_open = (my_cnt == 0)
    score = (
        0.25 * np.sum(my_open & (my_cnt == 3))
        + 0.04 * np.sum(my_open & (my_cnt == 2))
        - 0.28 * np.sum(opp_open & (opp_cnt == 3))
        - 0.04 * np.sum(opp_open & (opp_cnt == 2))
    )
    return float(np.tanh(score))


def c4_action_values_and_marginals(
    board: np.ndarray,
    to_move_tok: int,
    rng: random.Random,
    num_rollouts: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2-ply tactical move values `[42]`, cell marginals `[42, 2]`, and a completed grid `[42]`."""
    legal = c4_legal_cells(board)
    opp_tok = 3 - to_move_tok
    q_vals = np.full(C4_N, -1e4, dtype=np.float32)

    if not legal:
        marg = np.zeros((C4_N, 2), dtype=np.float32)
        marg[:, 0] = (board == 1).astype(np.float32)
        marg[:, 1] = (board == 2).astype(np.float32)
        return q_vals, marg, board.copy()

    for move in legal:
        b1 = board.copy()
        b1[move] = to_move_tok
        if c4_check_winner(b1) == to_move_tok:
            q_vals[move] = 1.0
            continue
        opp_legal = c4_legal_cells(b1)
        worst_reply = 1.0
        for om in opp_legal:
            b1[om] = opp_tok
            if c4_check_winner(b1) == opp_tok:
                worst_reply = -1.0
                b1[om] = 0
                break
            val = c4_heuristic_eval(b1, to_move_tok)
            if val < worst_reply:
                worst_reply = val
            b1[om] = 0
        q_vals[move] = float(worst_reply)

    valid_q = np.array([q_vals[m] for m in legal], dtype=np.float64)
    probs = np.exp((valid_q - valid_q.max()) * 3.0)
    probs /= probs.sum()

    counts = np.zeros((C4_N, 2), dtype=np.float32)
    sample_comp = board.copy()
    for r_idx in range(num_rollouts):
        b_sim = board.copy()
        first_m = legal[int(rng.choices(range(len(legal)), weights=probs.tolist(), k=1)[0])]
        b_sim[first_m] = to_move_tok
        cur = opp_tok
        # Complete all remaining empty cells respecting gravity so target_tokens is in {1, 2}
        while True:
            rem_leg = c4_legal_cells(b_sim)
            if not rem_leg:
                break
            mv = rem_leg[rng.randrange(len(rem_leg))]
            b_sim[mv] = cur
            cur = 3 - cur
        counts[:, 0] += (b_sim == 1).astype(np.float32)
        counts[:, 1] += (b_sim == 2).astype(np.float32)
        if r_idx == 0:
            sample_comp = b_sim

    marg = counts / float(num_rollouts)
    return q_vals, marg, sample_comp


# ---------------------------------------------------------------------------
# Unified Tactical Perfect-Information Environment
# ---------------------------------------------------------------------------


class TacticalGameEnv(UnifiedGameEnv):
    """Unified 5x5 Hex & 6x7 Connect-4 Environment for tactical search and Elo tournaments."""

    game_family = GameFamily.TACTICAL_PERFECT_INFO

    def __init__(
        self,
        variant: str = "hex",
        num_queries: int = 8,
    ):
        variant = variant.lower().strip()
        if variant not in ("hex", "connect4", "c4"):
            raise ValueError(f"Unknown tactical game variant {variant!r}; expected 'hex' or 'connect4'")
        self.variant = "hex" if variant == "hex" else "connect4"
        self.num_queries = int(num_queries)

        if self.variant == "hex":
            self.game_id = GameID.HEX_5X5
            self.grid_h = 5
            self.grid_w = 5
            self.num_vars = 25
            self.inventory_spec = 25
        else:
            self.game_id = GameID.CONNECT4_6X7
            self.grid_h = 6
            self.grid_w = 7
            self.num_vars = 42
            self.inventory_spec = 42

        self.default_utility_spec = UtilitySpec.PLAYER_0_WIN
        self.domain_size = 2  # token 1 = Player 0, token 2 = Player 1
        # Per-cell features: [row_norm, col_norm, is_p0, is_p1, is_legal_move, to_move_is_p0] -> 6 dims
        self.feat_dim = 6

    def _sample_midgame_board(self, rng: random.Random) -> Tuple[np.ndarray, int]:
        """Generate a non-terminal mid-game board and player to move (`1` or `2`)."""
        while True:
            board = np.zeros(self.num_vars, dtype=np.int64)
            n_plies = rng.randint(4, 12) if self.variant == "hex" else rng.randint(6, 18)
            cur = 1
            ok = True
            for _ in range(n_plies):
                if self.variant == "hex":
                    leg = [i for i in range(self.num_vars) if board[i] == 0]
                else:
                    leg = c4_legal_cells(board)
                if not leg:
                    ok = False
                    break
                mv = rng.choice(leg)
                board[mv] = cur
                w = hex_check_winner(board) if self.variant == "hex" else c4_check_winner(board)
                if w != 0:
                    ok = False
                    break
                cur = 3 - cur
            if ok:
                leg_now = [i for i in range(self.num_vars) if board[i] == 0] if self.variant == "hex" else c4_legal_cells(board)
                if len(leg_now) >= 2:
                    return board, cur

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
        rng = random.Random(seed * 1021 + split_offset + self.num_vars)

        B = batch_size
        N = self.num_vars
        Q = self.num_queries

        obs_tokens = np.zeros((B, N), dtype=np.int64)
        obs_mask = np.zeros((B, N), dtype=bool)
        target_tokens = np.zeros((B, N), dtype=np.int64)
        exact_marginals = np.zeros((B, N, 2), dtype=np.float32)
        obs_features = np.zeros((B, N, self.feat_dim), dtype=np.float32)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        legal_mask = np.zeros((B, N), dtype=bool)
        utility_vec = np.full((B, N), -1e4, dtype=np.float32)
        to_move_arr = np.ones(B, dtype=np.int64)

        rows_norm = np.array([i // self.grid_w / max(1, self.grid_h - 1) for i in range(N)], dtype=np.float32)
        cols_norm = np.array([(i % self.grid_w) / max(1, self.grid_w - 1) for i in range(N)], dtype=np.float32)

        for b in range(B):
            board, to_move = self._sample_midgame_board(rng)
            to_move_arr[b] = to_move
            obs_tokens[b] = board
            obs_mask[b] = board > 0

            if self.variant == "hex":
                q_vals, marg, comp = hex_action_values_and_marginals(board, to_move, rng)
                leg = [i for i in range(N) if board[i] == 0]
            else:
                q_vals, marg, comp = c4_action_values_and_marginals(board, to_move, rng)
                leg = c4_legal_cells(board)

            target_tokens[b] = comp
            exact_marginals[b] = marg
            legal_mask[b, leg] = True
            utility_vec[b] = q_vals

            obs_features[b, :, 0] = rows_norm
            obs_features[b, :, 1] = cols_norm
            obs_features[b, :, 2] = (board == 1).astype(np.float32)
            obs_features[b, :, 3] = (board == 2).astype(np.float32)
            obs_features[b, :, 4] = legal_mask[b].astype(np.float32)
            obs_features[b, :, 5] = 1.0 if to_move == 1 else 0.0

            unobs = [i for i in range(N) if board[i] == 0]
            for q in range(Q):
                ca, cb = rng.sample(unobs, 2)
                query_pairs[b, q] = (ca, cb, int(QueryType.EQUAL))
                query_targets[b, q] = float(marg[ca, 0] * marg[cb, 0] + marg[ca, 1] * marg[cb, 1])

        header = self.build_batch_header(B, device=dev)
        # Set per-row utility_spec to match active player (1 for Player 0, 2 for Player 1)
        header[:, 5] = torch.as_tensor(to_move_arr, dtype=torch.long, device=dev)

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
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            legal_action_mask=torch.as_tensor(legal_mask, dtype=torch.bool, device=dev),
            utility_vector=torch.as_tensor(utility_vec, dtype=torch.float32, device=dev),
            meta={"to_move": torch.as_tensor(to_move_arr, dtype=torch.long, device=dev)},
        )

    def play_match(
        self,
        policy_a: Callable[[np.ndarray, int, List[int]], int],
        policy_b: Callable[[np.ndarray, int, List[int]], int],
        num_games: int = 32,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Play `num_games` alternating-color games between `policy_a` and `policy_b` for Elo rating."""
        rng = random.Random(seed + 771)
        wins_a = 0
        wins_b = 0
        draws = 0

        for g in range(num_games):
            a_is_p0 = (g % 2 == 0)
            board = np.zeros(self.num_vars, dtype=np.int64)
            cur = 1
            winner = 0
            while True:
                leg = [i for i in range(self.num_vars) if board[i] == 0] if self.variant == "hex" else c4_legal_cells(board)
                if not leg:
                    break
                if (cur == 1 and a_is_p0) or (cur == 2 and not a_is_p0):
                    mv = int(policy_a(board.copy(), cur, leg))
                else:
                    mv = int(policy_b(board.copy(), cur, leg))
                if mv not in leg:
                    mv = rng.choice(leg)
                board[mv] = cur
                winner = hex_check_winner(board) if self.variant == "hex" else c4_check_winner(board)
                if winner != 0:
                    break
                cur = 3 - cur

            if winner == 0:
                draws += 1
            elif (winner == 1 and a_is_p0) or (winner == 2 and not a_is_p0):
                wins_a += 1
            else:
                wins_b += 1

        score_a = (wins_a + 0.5 * draws) / max(1, num_games)
        return {
            "wins_a": float(wins_a),
            "wins_b": float(wins_b),
            "draws": float(draws),
            "score_a": float(score_a),
        }


class HexEnv(TacticalGameEnv):
    def __init__(self, num_queries: int = 8):
        super().__init__(variant="hex", num_queries=num_queries)


class Connect4Env(TacticalGameEnv):
    def __init__(self, num_queries: int = 8):
        super().__init__(variant="connect4", num_queries=num_queries)
