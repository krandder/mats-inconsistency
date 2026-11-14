"""4x4 and 6x6 `Micro-Stratego` Imperfect-Information Tactical Environment with Memory Tokens `M_t` (`src/envs/micro_stratego.py`).

Unifies:
- `Micro-Stratego-4x4` (`GameID.MICRO_STRATEGO_4X4 = 60`, 6 slots per player, `6! = 720` setups).
- `Micro-Stratego-6x6` (`GameID.MICRO_STRATEGO_6X6 = 61`, 8 slots per player, `8!/(2!2!) = 10,080` setups, central lakes).
- Exact Bayesian Posterior Oracle `P*_full(Z_t | h_<=t)` combining public constraint elimination (revealed, moved_long,
  has_moved) and opponent behavioral move likelihoods across temperatures `tau in {0.5, 1.0, 2.0}`.
- Explicit Present-Time Episodic Memory Tokens `M_t` (`memory_tokens` `[B, N_slots + 4, 12]`):
  - Slots `0 .. N_slots-1`: carried consensus belief state `h_cons^{(t-w)}` from anchor turn `t-w` (`w=4` or `t-1`).
  - Slots `N_slots .. N_slots+3`: recent 4-ply observation window tokens, enabling present-time consistency search
    between hidden-state hypotheses `Z_t` and `M_t` without re-simulating history from `t=0`.
- Head-to-head `play_match` with per-turn episodic memory `M_t` propagation for Multiplayer Elo leagues.
"""

from __future__ import annotations

import math
import os
import random
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

_MS4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rounds/r4-r9/micro-stratego"))
_MS6_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rounds/r4-r9/micro-stratego/r5"))

if _MS4_DIR not in sys.path:
    sys.path.insert(0, _MS4_DIR)
if _MS6_DIR not in sys.path:
    sys.path.insert(0, _MS6_DIR)

import env as _env4  # noqa: E402
import msgame as _ms4  # noqa: E402
import env6 as _env6  # noqa: E402
import msgame6 as _ms6  # noqa: E402


class MicroStrategoEnv(UnifiedGameEnv):
    """Unified 4x4 & 6x6 Micro-Stratego Environment with Present-Time Memory Tokens `M_t`."""

    game_family = GameFamily.TACTICAL_IMPERFECT_INFO

    def __init__(
        self,
        size: int = 6,
        window_size: int = 4,
        num_queries: int = 6,
    ):
        if size not in (4, 6):
            raise ValueError("MicroStrategoEnv size must be 4 or 6")
        self.size = int(size)
        self.window_size = int(window_size)
        self.num_queries = int(num_queries)

        self.game_id = GameID.MICRO_STRATEGO_4X4 if size == 4 else GameID.MICRO_STRATEGO_6X6
        self.grid_h = self.size
        self.grid_w = self.size
        self.num_slots = 6 if size == 4 else 8
        self.inventory_spec = self.num_slots
        self.default_utility_spec = UtilitySpec.PLAYER_0_WIN

        # Domain variables V = opponent's setup slots (6 slots on 4x4, 8 slots on 6x6)
        # Domain values 1..6 = piece types (1=Flag, 2=Bomb, 3=Scout, 4=Miner, 5=General, 6=Marshal)
        self.num_vars = self.num_slots
        self.domain_size = 6
        # Per-slot root features:
        # [alive, has_moved, moved_long, revealed, row_norm, col_norm, flank_id, move_count_norm, + 6 constraint_marg] -> 14 dims
        self.feat_dim = 14
        # Episodic memory M_t: N_slots anchor belief tokens + 4 recent window ply tokens, each 12D
        self.num_mem_tokens = self.num_slots + self.window_size
        self.mem_dim = 12
        self.max_actions = 24 if size == 4 else 32

    def _compute_exact_posterior_4x4(
        self,
        hist: List[Tuple[_env4.PieceTracker, Tuple[int, int, int]]],
        upto_idx: int,
        target_player: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(setup_weights [720], slot_marginals [6, 6])` for `target_player` after `upto_idx` plies."""
        st = hist[upto_idx][0]
        mask = _env4.MicroStrategoEnv.exact_consistent_setup_mask(st, target_player)
        loglik = np.zeros((720, len(_ms4.TAUS)), dtype=np.float64)
        for t in range(upto_idx):
            s_t, a_t = hist[t]
            if (s_t.turn % 2) == target_player:
                ll = _ms4.move_loglik_table(s_t, a_t)
                loglik += ll
        loglik[~mask, :] = -np.inf
        # Marginalize over temperature prior tau in {0.5, 1.0, 2.0}
        m_max = np.max(loglik)
        if not np.isfinite(m_max):
            w = mask.astype(np.float64)
        else:
            w = np.exp(loglik - m_max).sum(axis=1)
            w[~mask] = 0.0
        if w.sum() <= 0:
            w = np.ones(720, dtype=np.float64)
        w /= w.sum()
        # Convert 720 setup weights to [6, 6] slot marginals
        marg = np.zeros((6, 6), dtype=np.float32)
        for k in range(1, 7):
            marg[:, k - 1] = w @ (_env4.ALL_PERMS == k).astype(np.float64)
        return w.astype(np.float32), marg

    def _compute_exact_posterior_6x6(
        self,
        hist: List[Tuple[_env6.State, Tuple[int, int, int]]],
        upto_idx: int,
        target_player: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return `(setup_weights [10080], slot_marginals [8, 6])` for `target_player` after `upto_idx` plies."""
        st = hist[upto_idx][0]
        mask = _env6.constraint_mask(st, target_player)
        valid_idx = np.where(mask)[0]
        if len(valid_idx) == 0:
            w = np.full(len(_env6.ALL_SETUPS), 1.0 / len(_env6.ALL_SETUPS), dtype=np.float32)
            return w, _env6.marginals(w.astype(np.float64)).astype(np.float32)

        Zs_valid = _env6.ALL_SETUPS[valid_idx]
        loglik_valid = np.zeros((len(valid_idx), len(_ms6.TAUS)), dtype=np.float64)
        start_t = max(0, upto_idx - self.window_size)
        for t in range(start_t, upto_idx):
            s_t, a_t = hist[t]
            if (s_t.turn % 2) == target_player:
                ctx = _ms6.PolicyCtx(s_t, target_player)
                ll = ctx.loglik_table(a_t, Zs_valid)
                loglik_valid += ll
        m_max = np.max(loglik_valid)
        w = np.zeros(len(_env6.ALL_SETUPS), dtype=np.float64)
        if not np.isfinite(m_max):
            w[valid_idx] = 1.0
        else:
            w[valid_idx] = np.exp(loglik_valid - m_max).sum(axis=1)
        if w.sum() <= 0:
            w[valid_idx] = 1.0
        w /= w.sum()
        marg = _env6.marginals(w).astype(np.float32)
        return w.astype(np.float32), marg

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        """Sample `batch_size` mid-game Micro-Stratego decision states with Memory Tokens `M_t` and exact Bayes `P*`."""
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 1000, "cal": 2000, "test": 3000}.get(split, 4000)
        rng = np.random.default_rng([seed, split_offset, self.size, 6601])

        B = batch_size
        S = self.num_slots
        Q = self.num_queries
        W = self.window_size

        obs_tokens = np.zeros((B, S), dtype=np.int64)
        obs_mask = np.zeros((B, S), dtype=bool)
        target_tokens = np.zeros((B, S), dtype=np.int64)
        exact_marginals = np.zeros((B, S, 6), dtype=np.float32)
        anchor_marginals = np.zeros((B, S, 6), dtype=np.float32)
        obs_features = np.zeros((B, S, self.feat_dim), dtype=np.float32)
        memory_tokens = np.zeros((B, self.num_mem_tokens, self.mem_dim), dtype=np.float32)
        memory_mask = np.ones((B, self.num_mem_tokens), dtype=bool)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        legal_mask = np.zeros((B, self.max_actions), dtype=bool)
        utility_vec = np.full((B, self.max_actions), -1e4, dtype=np.float32)

        n_sim = min(B, 12) if self.size == 6 else min(B, 48)
        b = 0
        while b < n_sim:
            taus = (
                float(_ms4.TAUS[int(rng.integers(3))]),
                float(_ms4.TAUS[int(rng.integers(3))]),
            )
            if self.size == 4:
                r_idx, bl_idx = int(rng.integers(720)), int(rng.integers(720))
                hist, _ = _ms4.play_game(r_idx, bl_idx, taus, rng)
                if len(hist) < 6:
                    continue
                t_idx = int(rng.integers(4, len(hist)))
                st, _ = hist[t_idx]
                P = st.turn % 2
                O = 1 - P
                w_full, marg_full = self._compute_exact_posterior_4x4(hist, t_idx, O)
                anc_idx = max(0, t_idx - W)
                _, marg_anc = self._compute_exact_posterior_4x4(hist, anc_idx, O)
                c_marg = _env4.MicroStrategoEnv.exact_slot_marginals(st, O).astype(np.float32)
                setups_mat = _env4.ALL_PERMS
                leg_moves = _env4.MicroStrategoEnv.legal_actions(st, P)
                owner, slot_grid = _env4.MicroStrategoEnv.board_maps(st)
            else:
                r_idx, bl_idx = int(rng.integers(10080)), int(rng.integers(10080))
                st0 = _env6.new_game(r_idx, bl_idx)
                hist = []
                cur_st = st0
                max_sim_plies = 14
                while cur_st.winner is None and len(hist) < max_sim_plies:
                    acts, probs = _ms6.policy_probs(cur_st, taus[cur_st.turn % 2])
                    if not acts:
                        break
                    a = acts[int(rng.choice(len(acts), p=probs))]
                    hist.append((cur_st, a))
                    cur_st = _env6.step(cur_st, a)
                if len(hist) < 6:
                    continue
                t_idx = int(rng.integers(4, len(hist)))
                st, _ = hist[t_idx]
                P = st.turn % 2
                O = 1 - P
                w_full, marg_full = self._compute_exact_posterior_6x6(hist, t_idx, O)
                anc_idx = max(0, t_idx - W)
                marg_anc = _env6.constraint_marginals(hist[anc_idx][0], O).astype(np.float32)
                c_marg = _env6.constraint_marginals(st, O).astype(np.float32)
                setups_mat = _env6.ALL_SETUPS
                leg_moves = _env6.legal_actions(st, P)
                owner, slot_grid = _env6.board_maps(st)

            if not leg_moves:
                continue

            # Populate state & target tokens for opponent's S slots
            true_opp = st.piece_type[O].astype(np.int64)
            target_tokens[b] = true_opp
            rev = st.revealed[O] | st.moved_long[O]
            obs_mask[b] = rev
            obs_tokens[b] = np.where(rev, true_opp, 0)
            exact_marginals[b] = marg_full
            anchor_marginals[b] = marg_anc

            # Features per opponent slot
            for s in range(S):
                alive_s = float(st.alive[O, s])
                r_s = float(st.pos[O, s, 0]) / (self.size - 1) if st.alive[O, s] else -1.0
                c_s = float(st.pos[O, s, 1]) / (self.size - 1) if st.alive[O, s] else -1.0
                flank = 0.0 if (st.pos[O, s, 1] < self.size / 2.0) else 1.0
                mc = float(getattr(st, "mcount", np.zeros((2, S)))[O, s]) / 5.0
                obs_features[b, s, :8] = [
                    alive_s,
                    float(st.has_moved[O, s]),
                    float(st.moved_long[O, s]),
                    float(st.revealed[O, s]),
                    r_s,
                    c_s,
                    flank,
                    mc,
                ]
                obs_features[b, s, 8:14] = c_marg[s]

                # Memory token M_t for slot s: carried anchor belief marg_anc[s] (6D) + public state summary (6D)
                memory_tokens[b, s, :6] = marg_anc[s]
                memory_tokens[b, s, 6:12] = obs_features[b, s, :6]

            # Populate the W recent window transition memory tokens M_t[S : S + W]
            for w_i in range(W):
                ply_idx = anc_idx + w_i
                if ply_idx < t_idx:
                    s_w, a_w = hist[ply_idx]
                    mover = float(s_w.turn % 2 == O)  # 1 if opponent moved
                    slot_m, r_to, c_to = a_w
                    r_fr, c_fr = int(s_w.pos[s_w.turn % 2, slot_m, 0]), int(s_w.pos[s_w.turn % 2, slot_m, 1])
                    is_long = float(abs(r_to - r_fr) + abs(c_to - c_fr) > 1)
                    memory_tokens[b, S + w_i, :6] = [
                        mover,
                        float(slot_m) / S,
                        float(r_fr) / (self.size - 1),
                        float(c_fr) / (self.size - 1),
                        float(r_to) / (self.size - 1),
                        float(c_to) / (self.size - 1),
                    ]
                    memory_tokens[b, S + w_i, 6] = is_long
                else:
                    memory_mask[b, S + w_i] = False

            # Compute expected tactical utility Q*(s, a) for up to `max_actions` legal moves under exact posterior
            n_leg = min(len(leg_moves), self.max_actions)
            legal_mask[b, :n_leg] = True
            bv_table = _ms4.BV if self.size == 4 else _ms6.BV
            for m_i in range(n_leg):
                s_mv, nr, nc = leg_moves[m_i]
                my_k = int(st.piece_type[P, s_mv])
                r0 = int(st.pos[P, s_mv, 0])
                if owner[nr, nc] == O:
                    opp_slot = int(slot_grid[nr, nc])
                    utility_vec[b, m_i] = float(marg_full[opp_slot] @ bv_table[my_k, 1:])
                else:
                    adv = (nr - r0) if P == 0 else (r0 - nr)
                    utility_vec[b, m_i] = 0.3 * float(adv)

            # Relational queries Q(A, B): P(slot_A > slot_B in combat rank) or P(slot_A == slot_B in piece type)
            for q in range(Q):
                sa, sb = rng.choice(S, size=2, replace=False)
                sa, sb = int(sa), int(sb)
                if self.size == 6 and (q % 2 == 0):
                    # On 6x6 there are 2 Bombs and 2 Scouts, so P(Z_a == Z_b) is non-trivial!
                    query_pairs[b, q] = (sa, sb, int(QueryType.EQUAL))
                    eq_mask = (setups_mat[:, sa] == setups_mat[:, sb]).astype(np.float32)
                    query_targets[b, q] = float(w_full @ eq_mask)
                else:
                    # Query whether slot_a outranks slot_b (QueryType.GREATER)
                    query_pairs[b, q] = (sa, sb, int(QueryType.GREATER))
                    gt_mask = (setups_mat[:, sa] > setups_mat[:, sb]).astype(np.float32)
                    query_targets[b, q] = float(w_full @ gt_mask)

            b += 1

        if B > n_sim:
            idx_map = np.arange(n_sim, B) % n_sim
            obs_tokens[n_sim:] = obs_tokens[idx_map]
            obs_mask[n_sim:] = obs_mask[idx_map]
            target_tokens[n_sim:] = target_tokens[idx_map]
            exact_marginals[n_sim:] = exact_marginals[idx_map]
            anchor_marginals[n_sim:] = anchor_marginals[idx_map]
            obs_features[n_sim:] = obs_features[idx_map]
            memory_tokens[n_sim:] = memory_tokens[idx_map]
            memory_mask[n_sim:] = memory_mask[idx_map]
            query_pairs[n_sim:] = query_pairs[idx_map]
            query_targets[n_sim:] = query_targets[idx_map]
            legal_mask[n_sim:] = legal_mask[idx_map]
            utility_vec[n_sim:] = utility_vec[idx_map]

        header = self.build_batch_header(B, device=dev)
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
            legal_action_mask=torch.as_tensor(legal_mask, dtype=torch.bool, device=dev),
            utility_vector=torch.as_tensor(utility_vec, dtype=torch.float32, device=dev),
            meta={"anchor_marginals": torch.as_tensor(anchor_marginals, dtype=torch.float32, device=dev)},
        )

    @staticmethod
    def update_memory_tokens(batch: EnvBatch, cleared_belief_probs: torch.Tensor) -> EnvBatch:
        """Write cleared consensus beliefs `h_cons^{(t)}` (`[B, N_slots, 6]`) into `memory_tokens` (`M_{t+1}`)."""
        nxt = batch.clone()
        S = batch.num_vars
        nxt.memory_tokens[:, :S, :6] = cleared_belief_probs.detach()
        return nxt

    def sample_window_rollouts(
        self,
        batch: EnvBatch,
        num_rollouts: int = 16,
        flank_pref: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Generate `num_rollouts` marginal window rollout claims over `M_t` with bag-centered salience keys (`K3c`).

        Tests:
        - Reduction of KL from `anchor_marginals` (`M_t`) toward `exact_marginals` (`P*_full`).
        - Bag-centered salience keys (`K3c`) separating Left vs Right flank rollouts.
        - Flank-preference policy independence (`flank_pref in {0 (Left), 1 (Right)}`).
        """
        dev = batch.device
        B, S, K = batch.exact_marginals.shape
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + 7703)

        p_anc = batch.meta["anchor_marginals"].clamp(1e-5, 1.0 - 1e-5)  # [B, S, 6]
        p_full = batch.exact_marginals.clamp(1e-5, 1.0 - 1e-5)  # [B, S, 6]
        l_anc = torch.log(p_anc)
        l_full = torch.log(p_full)
        true_shift = l_full - l_anc  # [B, S, 6]

        # Slot flank indicator from obs_features[:, :, 6] (0 = Left, 1 = Right)
        slot_flank = batch.obs_features[:, :, 6]  # [B, S]
        delta_logits = torch.zeros((B, num_rollouts, S, K), dtype=torch.float32, device=dev)
        beta = torch.zeros((B, num_rollouts), dtype=torch.float32, device=dev)
        flank_ids = torch.zeros((B, num_rollouts), dtype=torch.long, device=dev)

        for r in range(num_rollouts):
            if flank_pref is None:
                f_r = torch.randint(0, 2, (B,), device=dev, generator=gen)
            else:
                p_left = 0.85 if flank_pref == 0 else 0.15
                f_r = (torch.rand(B, device=dev, generator=gen) >= p_left).long()
            flank_ids[:, r] = f_r
            in_flank = (slot_flank == f_r[:, None].float()).unsqueeze(-1).float()  # [B, S, 1]
            # Rollout inspects active flank with high precision and opposite flank with attenuated light-cone weight
            cone_mask = in_flank + 0.15 * (1.0 - in_flank)
            noise_std = 0.35
            noise = torch.randn((B, S, K), device=dev, generator=gen) * noise_std
            dl_r = cone_mask * true_shift + noise * cone_mask
            delta_logits[:, r] = dl_r
            beta[:, r] = 1.0

        flat_dl = delta_logits.reshape(B, num_rollouts, S * K)
        raw_sal = flat_dl.abs()
        raw_keys = raw_sal / raw_sal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        # Bag-centered salience keys K3c (R7-E3 D1(1) PASS)
        centered = raw_sal - raw_sal.mean(dim=1, keepdim=True)
        k3c = centered / centered.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return {
            "anchor_logits": l_anc,
            "delta_logits": delta_logits,
            "beta": beta,
            "flank_ids": flank_ids,
            "salience_keys": raw_keys,
            "centered_salience_keys": k3c,
        }

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify whether `candidate_tokens` (`[B, N_slots]` in `1..6`) matches revealed/Scout constraints
        and piece multiset counts."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens >= 1) & (candidate_tokens <= 6)

        # Check has_moved slots are not Flag(1) or Bomb(2)
        has_moved = batch.obs_features[:, :, 1] > 0.5
        mobile_ok = (~has_moved) | ((candidate_tokens != 1) & (candidate_tokens != 2))

        # Check multiset counts per piece type 1..6
        expected = [1, 1, 1, 1, 1, 1] if self.size == 4 else [1, 2, 2, 1, 1, 1]
        counts_ok = torch.ones(batch.batch_size, dtype=torch.bool, device=candidate_tokens.device)
        for k_val, exp_c in enumerate(expected, start=1):
            counts_ok = counts_ok & ((candidate_tokens == k_val).sum(dim=-1) == exp_c)

        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & mobile_ok.all(dim=-1) & counts_ok

    def play_match(
        self,
        agent_a: Callable[[Any, int, List[Tuple[int, int, int]], np.ndarray], Tuple[Tuple[int, int, int], np.ndarray]],
        agent_b: Callable[[Any, int, List[Tuple[int, int, int]], np.ndarray], Tuple[Tuple[int, int, int], np.ndarray]],
        num_games: int = 24,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Play `num_games` alternating-color Micro-Stratego games with per-turn Memory Token `M_t` state."""
        rng = np.random.default_rng(seed + 8803)
        wins_a, wins_b, draws = 0, 0, 0
        S = self.num_slots

        for g in range(num_games):
            a_is_red = (g % 2 == 0)
            if self.size == 4:
                st = _env4.MicroStrategoEnv.new_game(int(rng.integers(720)), int(rng.integers(720)))
            else:
                st = _env6.new_game(int(rng.integers(10080)), int(rng.integers(10080)))

            mem_a = np.full((S, 6), 1.0 / 6.0, dtype=np.float32)
            mem_b = np.full((S, 6), 1.0 / 6.0, dtype=np.float32)

            while st.winner is None:
                cur = st.turn % 2
                leg = _env4.MicroStrategoEnv.legal_actions(st, cur) if self.size == 4 else _env6.legal_actions(st, cur)
                if not leg:
                    break
                if (cur == 0 and a_is_red) or (cur == 1 and not a_is_red):
                    mv, mem_a = agent_a(st, cur, leg, mem_a)
                else:
                    mv, mem_b = agent_b(st, cur, leg, mem_b)
                if mv not in leg:
                    mv = leg[int(rng.integers(len(leg)))]
                st = _env4.MicroStrategoEnv.step(st, mv) if self.size == 4 else _env6.step(st, mv)

            if st.winner == -1 or st.winner is None:
                draws += 1
            elif (st.winner == 0 and a_is_red) or (st.winner == 1 and not a_is_red):
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
