"""Galanis (2026) 4-Information-Structure Prediction Market Environment (`src/envs/galanis_market.py`).

Wraps `src/envs/simulations/galanis_market` (`futarchy-fi/simulations` implementation of Galanis 2026,
*Information Aggregation with AI Agents*, arXiv:2604.20050) into the unified `GLC-AOM` token & tensor interface.

Supports all 4 information and payoff structures:
- `t3s111y2` (`GameID.GALANIS_MARKET_T3S111Y2 = 70`): own 1-bit signal `d_i`, `X = 1[d_a + d_b + d_c >= 2]`
- `t3s110`   (`GameID.GALANIS_MARKET_T3S110 = 71`):   own 1-bit signal `d_i`, `X = 1[d_a == d_b == d_c == 1]`
- `t3s111`   (`GameID.GALANIS_MARKET_T3S111 = 72`):   own 1-bit signal `d_i`, `X = 1[d_a == d_b == d_c == 1]`
- `t3s111o2ye2` (`GameID.GALANIS_MARKET_T3S111O2YE2 = 73`): sees OTHER two signals `(d_j)_{j != i}`,
  `X = 1[d_a + d_b + d_c == 2]` (interactive Aumann-style reasoning across rounds).

Domain Variables `V = [d_a, d_b, d_c, X(omega)]` (`N_vars = 4`, domain `{1 (No/0), 2 (Yes/1)}`):
- Observed signals are clamped in `obs_tokens` / `obs_mask`.
- Public LMSR trade history & prior-round public belief over `omega in {a..h}` enter `memory_tokens` (`M_t`).
- Exact Bayesian posterior over `omega` given public price history + private partition cell populates
  `exact_marginals` `[B, 4, 2]`, `query_targets`, and expected LMSR trade utilities `utility_vector` `[B, num_actions]`.
"""

from __future__ import annotations

import os
import sys
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

_SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "simulations"))
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

from galanis_market.lmsr import LMSR  # noqa: E402
from galanis_market.myopic import _posterior_x_given_history  # noqa: E402
from galanis_market.structures import STATES, STATE_LABELS, STRUCTURES, Structure  # noqa: E402

STRUCTURE_TO_GAME_ID = {
    "t3s111y2": GameID.GALANIS_MARKET_T3S111Y2,
    "t3s110": GameID.GALANIS_MARKET_T3S110,
    "t3s111": GameID.GALANIS_MARKET_T3S111,
    "t3s111o2ye2": GameID.GALANIS_MARKET_T3S111O2YE2,
}

STRUCTURE_TO_INV_SPEC = {
    "t3s111y2": 1112,
    "t3s110": 1103,
    "t3s111": 1113,
    "t3s111o2ye2": 2222,
}


class GalanisMarketEnv(UnifiedGameEnv):
    """Unified Galanis (2026) Prediction Market Environment wrapping `futarchy-fi/simulations`."""

    game_family = GameFamily.PREDICTION_MARKET

    def __init__(
        self,
        structure_name: str = "t3s111o2ye2",
        num_rounds: int = 3,
        num_actions: int = 19,
        liquidity_b: float = 0.1,
        manipulator_bonus: float = 0.05,
        utility_spec: UtilitySpec = UtilitySpec.MARKET_HONEST_PROFIT,
        num_queries: int = 6,
    ):
        if structure_name not in STRUCTURES:
            raise ValueError(f"Unknown Galanis structure {structure_name!r}; choose from {list(STRUCTURES)}")
        self.structure_name = structure_name
        self.structure: Structure = STRUCTURES[structure_name]
        self.num_rounds = int(num_rounds)
        self.num_actions = int(num_actions)
        self.liquidity_b = float(liquidity_b)
        self.manipulator_bonus = float(manipulator_bonus)
        self.num_queries = int(num_queries)

        self.game_id = STRUCTURE_TO_GAME_ID[structure_name]
        self.grid_h = 1
        self.grid_w = 4  # 4 variables: [d_a, d_b, d_c, X(omega)]
        self.inventory_spec = STRUCTURE_TO_INV_SPEC[structure_name] * 10 + self.num_rounds
        self.default_utility_spec = utility_spec

        self.num_vars = 4
        self.domain_size = 2  # token 1 -> 0 (No), token 2 -> 1 (Yes)
        # Price grid {0.05, 0.10, ..., 0.95} for num_actions=19
        step = 1.0 / (self.num_actions + 1)
        self.price_grid = np.array([round((i + 1) * step, 6) for i in range(self.num_actions)], dtype=np.float32)
        self.lmsr = LMSR(b=self.liquidity_b)

        # Per-variable features:
        # [var_id_one_hot(4), is_observed, obs_val, active_trader_id_norm, round_norm, current_price, structure_id_one_hot(4)] -> 13 dims
        self.feat_dim = 13
        # Memory tokens M_t: up to num_rounds + 1 tokens, each 12D (8D public belief over omega + round/trader/price_from/price_to)
        self.num_mem_tokens = self.num_rounds + 1
        self.mem_dim = 12

    def _simulate_public_history_to_round(
        self,
        omega_idx: int,
        stop_round: int,
    ) -> Tuple[List[float], List[np.ndarray], Dict[int, float]]:
        """Simulate Bayesian myopic play from round `0` up to `stop_round` (exclusive).

        Returns:
            prices: list of public prices `[p_0=0.5, p_1, ..., p_{stop_round}]`
            public_belief_snapshots: list of `[8]` probability vectors over `STATES` after each round
            current_public_belief: dict `{omega: prob}` at the start of `stop_round`.
        """
        belief: Dict[int, float] = {i: 1.0 / len(STATES) for i in range(len(STATES))}
        prices: List[float] = [0.5]
        snapshots: List[np.ndarray] = [np.full(8, 1.0 / 8.0, dtype=np.float32)]
        cells_for_traders = [self.structure.cell_of(t, STATES[omega_idx]) for t in range(3)]

        for r in range(stop_round):
            trader = r % 3
            cell = cells_for_traders[trader]
            p_t = _posterior_x_given_history(self.structure, trader, cell, belief)
            prices.append(float(p_t))

            cell_to_price = {
                c: _posterior_x_given_history(self.structure, trader, c, belief)
                for c in range(self.structure.cells_per_trader)
            }
            consistent_cells = [c for c, p in cell_to_price.items() if abs(p - p_t) < 1e-6]
            if not consistent_cells:
                consistent_cells = [cell]

            new_belief: Dict[int, float] = {}
            for om, prob in belief.items():
                if self.structure.cell_of(trader, STATES[om]) in consistent_cells:
                    new_belief[om] = prob
            z = sum(new_belief.values())
            if z > 0:
                belief = {k: v / z for k, v in new_belief.items()}
            snap = np.array([belief.get(i, 0.0) for i in range(8)], dtype=np.float32)
            snapshots.append(snap)

        return prices, snapshots, belief

    def _compute_action_utilities(
        self,
        p_from: np.ndarray,
        p_true_x: np.ndarray,
        utility_spec: Union[UtilitySpec, int],
    ) -> np.ndarray:
        """Compute expected LMSR trade payoff (+ optional manipulator bias) for each grid price `p_to`."""
        B = len(p_from)
        A = self.num_actions
        u = np.zeros((B, A), dtype=np.float32)
        uspec = int(utility_spec)

        for b in range(B):
            pf = max(1e-4, min(1.0 - 1e-4, float(p_from[b])))
            px = float(p_true_x[b])
            for a_idx, pt in enumerate(self.price_grid):
                pt_f = max(1e-4, min(1.0 - 1e-4, float(pt)))
                pay_yes = self.lmsr.trade_payoff(pf, pt_f, True)
                pay_no = self.lmsr.trade_payoff(pf, pt_f, False)
                exp_lmsr = px * pay_yes + (1.0 - px) * pay_no
                if uspec == int(UtilitySpec.BIAS_PRO_APPROVE):
                    exp_lmsr += self.manipulator_bonus * (pt_f - 0.5)
                elif uspec == int(UtilitySpec.BIAS_PRO_REJECT):
                    exp_lmsr -= self.manipulator_bonus * (pt_f - 0.5)
                u[b, a_idx] = float(exp_lmsr)
        return u

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
        split_offset = {"train": 0, "val": 1000, "cal": 2000, "test": 3000}.get(split, 4000)
        rng = np.random.default_rng([seed, split_offset, int(self.game_id), 7001])

        B = batch_size
        V = self.num_vars
        Q = self.num_queries
        M_tok = self.num_mem_tokens

        obs_tokens = np.zeros((B, V), dtype=np.int64)
        obs_mask = np.zeros((B, V), dtype=bool)
        target_tokens = np.zeros((B, V), dtype=np.int64)
        exact_marginals = np.zeros((B, V, 2), dtype=np.float32)
        obs_features = np.zeros((B, V, self.feat_dim), dtype=np.float32)
        memory_tokens = np.zeros((B, M_tok, self.mem_dim), dtype=np.float32)
        memory_mask = np.zeros((B, M_tok), dtype=bool)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)

        p_from_arr = np.zeros(B, dtype=np.float32)
        p_true_x_arr = np.zeros(B, dtype=np.float32)
        omega_arr = np.zeros(B, dtype=np.int64)
        round_arr = np.zeros(B, dtype=np.int64)

        struct_idx = ["t3s111y2", "t3s110", "t3s111", "t3s111o2ye2"].index(self.structure_name)
        all_var_pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]

        for b in range(B):
            omega_idx = int(rng.integers(len(STATES)))
            omega_arr[b] = omega_idx
            da, db, dc = STATES[omega_idx]
            x_val = self.structure.x_of(STATES[omega_idx])
            true_vals = [da, db, dc, x_val]
            target_tokens[b] = np.array(true_vals, dtype=np.int64) + 1

            r_now = int(rng.integers(self.num_rounds))
            round_arr[b] = r_now
            trader = r_now % 3
            cell = self.structure.cell_of(trader, STATES[omega_idx])

            prices, snapshots, pub_belief = self._simulate_public_history_to_round(omega_idx, r_now)
            p_curr = prices[-1]
            p_from_arr[b] = p_curr

            # Which variables does `trader` directly observe at the root?
            if self.structure_name == "t3s111o2ye2":
                # Trader observes the OTHER two signal bits
                for j in range(3):
                    if j != trader:
                        obs_mask[b, j] = True
                        obs_tokens[b, j] = true_vals[j] + 1
            else:
                # Trader observes their own signal bit
                obs_mask[b, trader] = True
                obs_tokens[b, trader] = true_vals[trader] + 1

            # Exact posterior over omega given public history AND trader's private cell
            priv_weights = np.array(
                [
                    pub_belief.get(om, 0.0) if self.structure.cell_of(trader, STATES[om]) == cell else 0.0
                    for om in range(8)
                ],
                dtype=np.float64,
            )
            if priv_weights.sum() <= 0:
                priv_weights = np.array(
                    [1.0 if self.structure.cell_of(trader, STATES[om]) == cell else 0.0 for om in range(8)],
                    dtype=np.float64,
                )
            priv_weights /= priv_weights.sum()

            # Compute exact marginals for [d_a, d_b, d_c, X(omega)]
            states_mat = np.array(
                [[STATES[om][0], STATES[om][1], STATES[om][2], self.structure.x_of(STATES[om])] for om in range(8)],
                dtype=np.float64,
            )  # [8, 4]
            p_one = priv_weights @ states_mat  # [4]
            exact_marginals[b, :, 0] = (1.0 - p_one).astype(np.float32)
            exact_marginals[b, :, 1] = p_one.astype(np.float32)
            p_true_x_arr[b] = float(p_one[3])

            # Features per variable
            for v_i in range(V):
                obs_features[b, v_i, v_i] = 1.0
                obs_features[b, v_i, 4] = float(obs_mask[b, v_i])
                obs_features[b, v_i, 5] = float(obs_tokens[b, v_i] - 1) if obs_mask[b, v_i] else 0.5
                obs_features[b, v_i, 6] = float(trader) / 2.0
                obs_features[b, v_i, 7] = float(r_now) / max(1, self.num_rounds - 1)
                obs_features[b, v_i, 8] = float(p_curr)
                obs_features[b, v_i, 9 + struct_idx] = 1.0

            # Populate Memory Tokens M_t from public price trajectory & public belief snapshots
            for m_i in range(r_now + 1):
                memory_mask[b, m_i] = True
                memory_tokens[b, m_i, :8] = snapshots[m_i]
                p_prev = prices[m_i - 1] if m_i > 0 else 0.5
                p_after = prices[m_i]
                memory_tokens[b, m_i, 8:12] = [
                    float(m_i) / max(1, self.num_rounds),
                    float((m_i - 1) % 3) / 2.0 if m_i > 0 else 0.0,
                    float(p_prev),
                    float(p_after),
                ]

            # Relational queries Q(A, B) across the 6 pairs of [d_a, d_b, d_c, X]
            for q_i in range(Q):
                va, vb = all_var_pairs[q_i % len(all_var_pairs)]
                query_pairs[b, q_i] = (va, vb, int(QueryType.EQUAL))
                eq_om = (states_mat[:, va] == states_mat[:, vb]).astype(np.float64)
                query_targets[b, q_i] = float(priv_weights @ eq_om)

        header = self.build_batch_header(B, utility_spec=uspec, device=dev)
        u_vec = self._compute_action_utilities(p_from_arr, p_true_x_arr, uspec)
        legal_mask = torch.ones((B, self.num_actions), dtype=torch.bool, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=torch.as_tensor(obs_tokens, dtype=torch.long, device=dev),
            obs_mask=torch.as_tensor(obs_mask, dtype=torch.bool, device=dev),
            obs_features=torch.as_tensor(obs_features, dtype=torch.float32, device=dev),
            memory_tokens=torch.as_tensor(memory_tokens, dtype=torch.float32, device=dev),
            memory_mask=torch.as_tensor(memory_mask, dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            legal_action_mask=legal_mask,
            utility_vector=torch.as_tensor(u_vec, dtype=torch.float32, device=dev),
            meta={
                "p_from": torch.as_tensor(p_from_arr, dtype=torch.float32, device=dev),
                "p_true_x": torch.as_tensor(p_true_x_arr, dtype=torch.float32, device=dev),
                "omega_idx": torch.as_tensor(omega_arr, dtype=torch.long, device=dev),
                "round_idx": torch.as_tensor(round_arr, dtype=torch.long, device=dev),
            },
        )

    def swap_utility(
        self,
        batch: EnvBatch,
        new_utility_spec: Union[UtilitySpec, int],
    ) -> EnvBatch:
        swapped = super().swap_utility(batch, new_utility_spec)
        p_from = batch.meta["p_from"].detach().cpu().numpy()
        p_true_x = batch.meta["p_true_x"].detach().cpu().numpy()
        u_vec = self._compute_action_utilities(p_from, p_true_x, new_utility_spec)
        swapped.utility_vector = torch.as_tensor(u_vec, dtype=torch.float32, device=batch.device)
        return swapped

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify whether `candidate_tokens` (`[B, 4]` in `{1, 2}`) satisfies `X == payoff_fn(d_a, d_b, d_c)`
        and matches observed private signals."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens == 1) | (candidate_tokens == 2)
        bits = (candidate_tokens - 1).clamp(0, 1)  # [B, 4]
        s_sum = bits[:, 0] + bits[:, 1] + bits[:, 2]
        x_bit = bits[:, 3]

        if self.structure_name == "t3s111y2":
            expected_x = (s_sum >= 2).long()
        elif self.structure_name in ("t3s110", "t3s111"):
            expected_x = (s_sum == 3).long()
        else:
            expected_x = (s_sum == 2).long()

        payoff_consistent = x_bit == expected_x
        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & payoff_consistent

    def play_market_tournament(
        self,
        traders: List[Callable[[int, int, List[float], str], float]],
        num_episodes: int = 64,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """Run `num_episodes` 3-player LMSR prediction market games across `traders[0..2]`.

        Each trader callback receives `(trader_seat, partition_cell, price_history, structure_name)`
        and returns a target price `p_to in (0.01, 0.99)`.
        """
        rng = np.random.default_rng(seed + 7109)
        profits = np.zeros(3, dtype=np.float64)
        final_brier = 0.0

        for _ in range(num_episodes):
            omega_idx = int(rng.integers(len(STATES)))
            omega = STATES[omega_idx]
            outcome_yes = bool(self.structure.x_of(omega) == 1)
            prices = [0.5]
            for r in range(self.num_rounds):
                seat = r % 3
                cell = self.structure.cell_of(seat, omega)
                p_from = prices[-1]
                p_to = float(traders[seat](seat, cell, list(prices), self.structure_name))
                p_to = max(0.01, min(0.99, p_to))
                profits[seat] += self.lmsr.trade_payoff(p_from, p_to, outcome_yes)
                prices.append(p_to)
            final_brier += (prices[-1] - float(outcome_yes)) ** 2

        mean_profits = (profits / max(1, num_episodes)).tolist()
        return {
            "mean_profits": mean_profits,
            "final_brier": float(final_brier / max(1, num_episodes)),
        }
