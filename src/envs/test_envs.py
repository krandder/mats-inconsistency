"""Unit & Speed Tests for the Unified Multi-Game Environment Suite (`src/envs/test_envs.py`).

Tests all 8 environment families (15 registered configurations):
1. `equiv_sat.py`: Latent Equivalence / Parity & Direct `P(A == B)` Query benchmark
2. `sudoku.py`: 6x6 and 9x9 `Sudoku-Extreme`
3. `shiphunt.py`: 6x6 and 8x8 Coupled `ShipHunt`
4. `minesweeper.py`: 6x6 Epistemic Minesweeper
5. `hex_c4.py`: 5x5 Hex & 6x7 Connect-4
6. `micro_stratego.py`: 4x4 & 6x6 `Micro-Stratego` with Memory Tokens `M_t`
7. `galanis_market.py`: Galanis (2026) 4-information-structure prediction market game
8. `proposal_poker.py`: Futarchy Proposal Poker with private signal `s_i` and swappable utility bias `y_i`
"""

from __future__ import annotations

import os
import sys
import time

if "/tmp/r4e3a/pylib" not in sys.path and os.path.isdir("/tmp/r4e3a/pylib"):
    sys.path.insert(0, "/tmp/r4e3a/pylib")

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.envs import (
    ENV_REGISTRY,
    HEADER_LEN,
    Connect4Env,
    EquivSatEnv,
    GalanisMarketEnv,
    GameFamily,
    GameID,
    HexEnv,
    MicroStrategoEnv,
    MinesweeperEnv,
    ProposalPokerEnv,
    ShipHuntEnv,
    SudokuEnv,
    UtilitySpec,
    list_registered_envs,
    make_env,
)


def test_hierarchical_headers_unique_and_valid():
    """Verify every registered game variant emits a distinct, unambiguous 6-token header."""
    headers = {}
    for name in list_registered_envs():
        env = make_env(name)
        batch = env.sample_batch(batch_size=2, split="test", seed=42)
        assert batch.header.shape == (2, HEADER_LEN)
        h_tuple = tuple(int(x) for x in batch.header[0].tolist())
        # First 5 tokens [GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC] uniquely identify the game structure
        struct_key = h_tuple[:5]
        assert struct_key not in headers, f"Header collision between {name} and {headers[struct_key]}: {struct_key}"
        headers[struct_key] = name

        # Verify basic tensor shapes and probability simplex invariants
        B, V, K = batch.exact_marginals.shape
        assert B == 2
        assert V == env.num_vars
        assert K == env.domain_size
        assert batch.obs_tokens.shape == (B, V)
        assert batch.obs_mask.shape == (B, V)
        assert batch.target_tokens.shape == (B, V)
        assert torch.allclose(
            batch.exact_marginals[batch.solvable].sum(dim=-1),
            torch.ones((int(batch.solvable.sum()), V)),
            atol=1e-4,
        )


def test_equiv_sat_floating_queries_and_unmasking():
    """Verify Equiv-SAT floating components have 0.5 node marginals but exact {0, 1} P(A == B) queries."""
    env = EquivSatEnv(num_vars=16, num_clauses=14, num_queries=8, num_anchored=2)
    batch = env.sample_batch(batch_size=8, split="test", seed=7)

    # Ground-truth target_tokens must be a valid witness on all solvable instances
    is_wit = env.check_witness(batch, batch.target_tokens)
    assert torch.all(is_wit == batch.solvable)

    # Evaluate exact oracle vs independent-marginal fallback on relational queries Q(A, B)
    metrics_exact = env.evaluate_beliefs(batch, batch.exact_marginals, batch.query_targets)
    metrics_marg_only = env.evaluate_beliefs(batch, batch.exact_marginals, None)

    assert metrics_exact["kl"] < 1e-6
    assert metrics_exact["query_kl"] < 1e-6
    assert metrics_exact["query_acc"] == 1.0
    # Independent marginals cannot resolve floating equivalence pairs (P_marg(A==B) = 0.5 -> high query_kl)
    assert metrics_marg_only["query_kl"] > 0.25

    # Clamping variable A in a floating query pair via step_unmask must collapse variable B's conditional marginal!
    va = batch.query_pairs[:, 0, 0]
    true_a = batch.target_tokens.gather(1, va.unsqueeze(1)).squeeze(1)
    clamped_batch = env.step_unmask(batch, va, true_a)
    cond_margs, _, _ = env.exact_posterior_for_batch(clamped_batch)
    vb = batch.query_pairs[:, 0, 1]
    b_arange = torch.arange(batch.batch_size)
    # After clamping A, B's marginal in the same component must be deterministic (0 or 1)!
    max_prob_b = cond_margs[b_arange, vb].max(dim=-1).values
    assert torch.all(max_prob_b > 0.99)


def test_sudoku_6x6_and_9x9_extreme():
    """Verify 6x6 Sudoku and 9x9 Sudoku-Extreme board generation, stall masks, and witness checks."""
    for size in (6, 9):
        env = SudokuEnv(size=size, extreme=True, unsolvable_frac=0.25, num_queries=6)
        batch = env.sample_batch(batch_size=4, split="test", seed=11)
        assert batch.num_vars == size * size
        assert batch.domain_size == size

        # Ground-truth outcome on solvable boards must pass check_witness
        wit = env.check_witness(batch, batch.target_tokens)
        assert torch.all(wit[batch.solvable])

        # Corrupting a cell must fail check_witness
        bad = batch.target_tokens.clone()
        bad[:, 0] = (bad[:, 0] % size) + 1
        assert not torch.any(env.check_witness(batch, bad))

        metrics = env.evaluate_beliefs(batch, batch.exact_marginals, batch.query_targets)
        assert metrics["kl"] < 1e-5
        assert metrics["witness_rate"] > 0.0


def test_shiphunt_6x6_and_8x8_and_patch_rollouts():
    """Verify 6x6 and 8x8 Coupled ShipHunt exact Bayes, 3-tier patch rollouts, and utility swaps."""
    for size in (6, 8):
        env = ShipHuntEnv(size=size, num_queries=6)
        batch = env.sample_batch(batch_size=8, split="test", seed=19, num_root_clamped=2)
        assert batch.num_vars == size * size
        assert torch.all(env.check_witness(batch, batch.target_tokens))

        # Test patch rollout sampling and salience keys
        ro_nw = env.sample_patch_rollout_claims(batch, num_reads=12, preference_sector=0, seed=1)
        ro_se = env.sample_patch_rollout_claims(batch, num_reads=12, preference_sector=1, seed=2)
        assert ro_nw["delta_logits"].shape == (8, 12, size * size)
        assert ro_nw["salience_keys"].shape == (8, 12, size * size)
        assert torch.all(ro_nw["beta"] > 0)

        # Test zero-retraining utility swap between NW and SE sector preference
        swapped = env.swap_utility(batch, UtilitySpec.SECTOR_SE_PREF)
        assert torch.all(swapped.header[:, 5] == int(UtilitySpec.SECTOR_SE_PREF))
        assert not torch.allclose(batch.utility_vector, swapped.utility_vector)
        assert torch.allclose(batch.exact_marginals, swapped.exact_marginals)


def test_minesweeper_6x6_exact_posterior_and_witness():
    """Verify 6x6 Epistemic Minesweeper exact frontier+interior posterior and witness verifier."""
    env = MinesweeperEnv(size=6, num_mines=6, num_revealed_clues=10)
    batch = env.sample_batch(batch_size=8, split="test", seed=23)
    assert batch.num_vars == 36
    assert torch.all(env.check_witness(batch, batch.target_tokens))

    # Expected number of mines under exact_marginals must equal num_mines = 6.0 on every board!
    expected_mines = batch.exact_marginals[:, :, 1].sum(dim=-1)
    assert torch.allclose(expected_mines, torch.full_like(expected_mines, 6.0), atol=1e-3)

    metrics = env.evaluate_beliefs(batch, batch.exact_marginals, batch.query_targets)
    assert metrics["kl"] < 1e-5
    assert metrics["query_kl"] < 1e-5


def test_hex_and_connect4_tactical_and_matches():
    """Verify 5x5 Hex and 6x7 Connect-4 batch generation and head-to-head match execution."""
    for env_cls in (HexEnv, Connect4Env):
        env = env_cls(num_queries=6)
        batch = env.sample_batch(batch_size=6, split="test", seed=31)
        assert batch.legal_action_mask is not None
        assert torch.all(batch.legal_action_mask.any(dim=-1))

        # Greedy tactical policy vs random policy in play_match
        def random_pol(board, cur, leg):
            return leg[0]

        def center_pol(board, cur, leg):
            mid = len(board) // 2
            return min(leg, key=lambda m: abs(m - mid))

        res = env.play_match(center_pol, random_pol, num_games=8, seed=5)
        assert res["wins_a"] + res["wins_b"] + res["draws"] == 8.0
        assert 0.0 <= res["score_a"] <= 1.0


def test_micro_stratego_4x4_and_6x6_memory_and_k3c():
    """Verify 4x4 & 6x6 Micro-Stratego exact Bayes, M_t memory updates, and K3c flank separation."""
    for size in (4, 6):
        env = MicroStrategoEnv(size=size, window_size=4, num_queries=6)
        batch = env.sample_batch(batch_size=6, split="test", seed=37)
        S = env.num_slots
        assert batch.num_vars == S
        assert batch.domain_size == 6
        assert batch.memory_tokens.shape == (6, S + 4, 12)
        assert torch.all(env.check_witness(batch, batch.target_tokens))

        # Test updating M_t with cleared consensus beliefs
        updated = env.update_memory_tokens(batch, batch.exact_marginals)
        assert torch.allclose(updated.memory_tokens[:, :S, :6], batch.exact_marginals)

        # Test window rollouts and bag-centered salience keys K3c
        ro = env.sample_window_rollouts(batch, num_rollouts=16, seed=41)
        k3c = ro["centered_salience_keys"]  # [B, 16, S*6]
        flanks = ro["flank_ids"]  # [B, 16]
        # Cosine similarity between same-flank vs across-flank rollouts under K3c
        gram = torch.bmm(k3c, k3c.transpose(1, 2))  # [B, 16, 16]
        same_f = flanks.unsqueeze(2) == flanks.unsqueeze(1)
        eye = torch.eye(16, dtype=torch.bool, device=batch.device).unsqueeze(0)
        within_sim = gram[same_f & (~eye)].mean().item()
        across_sim = gram[~same_f].mean().item()
        assert (within_sim - across_sim) > 0.30, f"Expected K3c flank cosine gap > 0.30, got {within_sim - across_sim:.3f}"


def test_galanis_market_all_4_structures():
    """Verify all 4 Galanis (2026) prediction market information structures and LMSR tournament play."""
    for struct_name in ("t3s111y2", "t3s110", "t3s111", "t3s111o2ye2"):
        env = GalanisMarketEnv(structure_name=struct_name, num_rounds=3, num_actions=19)
        batch = env.sample_batch(batch_size=8, split="test", seed=43)
        assert batch.num_vars == 4
        assert torch.all(env.check_witness(batch, batch.target_tokens))

        # Verify utility swap shifts optimal price target in the manipulator direction
        b_up = env.swap_utility(batch, UtilitySpec.BIAS_PRO_APPROVE)
        b_dn = env.swap_utility(batch, UtilitySpec.BIAS_PRO_REJECT)
        opt_up = b_up.utility_vector.argmax(dim=-1).float().mean().item()
        opt_dn = b_dn.utility_vector.argmax(dim=-1).float().mean().item()
        assert opt_up >= opt_dn

    # Run a 3-trader LMSR tournament on t3s111o2ye2
    env_hard = GalanisMarketEnv(structure_name="t3s111o2ye2", num_rounds=3)
    res = env_hard.play_market_tournament(
        traders=[
            lambda seat, cell, hist, s: 0.5,
            lambda seat, cell, hist, s: 0.6 if cell in (1, 2) else 0.3,
            lambda seat, cell, hist, s: hist[-1],
        ],
        num_episodes=16,
        seed=9,
    )
    assert len(res["mean_profits"]) == 3
    assert 0.0 <= res["final_brier"] <= 1.0


def test_proposal_poker_honesty_and_utility_swap():
    """Verify Proposal Poker exact Bayes, zero-retraining utility bias swaps, and LIR honesty audit."""
    env = ProposalPokerEnv(signal_precision=0.75, bias_magnitude=2.0)
    batch = env.sample_batch(batch_size=16, split="test", seed=53)
    assert batch.num_vars == 5
    assert torch.all(env.check_witness(batch, batch.target_tokens))

    # 1. LIR-Masked oracle model: uses exact_marginals for beliefs (independent of y_i)
    #    and utility_vector for actions (steered by y_i)
    def lir_masked_model(b):
        return b.exact_marginals, b.utility_vector * 5.0

    # 2. Fused / corrupted model: shifts stated belief on x toward utility_bias_y
    def fused_corrupted_model(b):
        yi = b.meta["utility_bias_y"].unsqueeze(-1)  # [B, 1]
        p_x = (b.exact_marginals[:, 0:1, 1] + 0.20 * torch.tanh(yi)).clamp(0.01, 0.99)
        corrupted = b.exact_marginals.clone()
        corrupted[:, 0:1, 1] = p_x
        corrupted[:, 0:1, 0] = 1.0 - p_x
        return corrupted, b.utility_vector * 5.0

    audit_lir = env.audit_honesty_and_swap(lir_masked_model, batch)
    audit_fused = env.audit_honesty_and_swap(fused_corrupted_model, batch)

    assert audit_lir["belief_kl"] < 1e-6
    assert abs(audit_lir["desire_bias"]) < 1e-7
    assert audit_lir["belief_drift_l1"] < 1e-7
    assert audit_lir["faith_swap_rate"] > 0.50

    assert audit_fused["desire_bias"] > 0.25
    assert audit_fused["belief_drift_l1"] > 0.25


def test_cpu_and_gpu_execution_speed():
    """Verify all 8 environments execute cleanly and fast on CPU and (if available) CUDA."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    for dev in devices:
        t0 = time.perf_counter()
        for name in list_registered_envs():
            env = make_env(name)
            b = env.sample_batch(batch_size=4, split="test", seed=101, device=dev)
            assert b.device.type == torch.device(dev).type
            m = env.evaluate_beliefs(b, b.exact_marginals, b.query_targets)
            assert m["kl"] < 1e-4
        elapsed = time.perf_counter() - t0
        assert elapsed < 15.0, f"Suite execution on {dev} took {elapsed:.2f}s (> 15.0s)"
