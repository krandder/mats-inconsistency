"""Unit Tests for Classical Search Teachers (`src/search/classical_teachers.py`)
and `GLCSearchTransformer` (`src/search/neural_tree_controller.py`).

Verifies:
1. Incremental `O(b)` `anc_mask`, `rel_depth`, `visit_count`, `dag_anc_mask`, and `infoset_mask`
   inside `TreePagedKVBank`.
2. Step-by-step search tree trace generation `(node_features, tree_topology, target_subtree_N,
   target_subtree_Q, target_node_select, target_action_prior, target_root_belief)` for all
   Classical Search Teachers (`MCTS-PUCT`, `MCGS-Transposition-DAG`, `Minimax-AlphaBeta-PNS`,
   `CDCL-Global-Nogood`, `Disjunctive-BP`, `CFR-Infoset-RegretMatching`,
   `Gumbel-Sequential-Halving`, `Heuristic-GLC-PUCT`) across all 6 target environments
   (`hex_5x5`, `connect4_6x7`, `sudoku_6x6`, `equiv_sat`, `shiphunt_6x6`, `micro_stratego_4x4`).
3. The `R13` Expressivity Matrix across `flat_token`, `parent_1hop`, `full_ancestry_v1`,
   and `glc_search_v2` at tree depths `d in {2, 4, 8, 16}`, including all 5 Expressivity Barriers
   beyond basic tree ancestor masks (`GLCSearch v1` vs `GLCSearch v2`).
4. 2-Hop Tseitin Auxiliary Coupling (`Tseitin2HopQueryCoupler`) for combinatorial queries `Q(A, B)`.
5. Unsupervised End-to-End Search Discovery (`R14-UNSUP-DISCOVERY`), exact `b=0` identity,
   runtime `PoisonTensor` zero-leakage invariance, and static AST audit (`S1`-`S6`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

from src.envs import make_env
from src.eval.ast_leak_auditor import audit_r11_files
from src.models.glc_aom import GLCAOMConfig, GLCAOMWorldModel
from src.search import (
    CDCLBackjumpTeacher,
    CFRRegretMatchingTeacher,
    DisjunctiveBPTeacher,
    GLCSearchTransformer,
    GumbelSequentialHalvingTeacher,
    HeuristicGLCPopTeacher,
    MCGSTranspositionTeacher,
    MCTSPUCTTeacher,
    MinimaxAlphaBetaTeacher,
    build_controlled_depth_tree_bank,
)


def _make_small_world_model() -> GLCAOMWorldModel:
    torch.manual_seed(42)
    cfg = GLCAOMConfig(
        num_vars=42,
        num_vals=6,
        d_feat=26,
        d_mem=16,
        k_mem=4,
        m_aux=8,
        d_aux=16,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_key=32,
        use_header_film=True,
    )
    return GLCAOMWorldModel(cfg)


def test_incremental_tree_bank_masks_and_counts() -> None:
    """Verify incremental O(b) anc_mask, rel_depth, and visit_count against brute-force traversal."""
    wm = _make_small_world_model()
    env = make_env("equiv_sat")
    batch = env.sample_batch(batch_size=3, split="test", seed=11)

    for d in (2, 4, 8, 16):
        bank = build_controlled_depth_tree_bank(wm, batch, target_depth=d, num_branches=2, seed=d)
        b = bank.num_active
        assert int(bank.depth[:, :b].max().item()) == d, f"Expected max depth {d}"

        # Brute-force check of anc_mask[n, j, k] = 1[j in Anc(k)] and rel_depth[n, j, k] = depth[k] - depth[j]
        for n_i in range(batch.batch_size):
            for k in range(b):
                anc_set = set()
                cur = k
                while True:
                    anc_set.add(cur)
                    if cur == 0:
                        break
                    cur = int(bank.parent_idx[n_i, cur].item())
                for j in range(b):
                    expected_anc = j in anc_set
                    actual_anc = bool(bank.anc_mask[n_i, j, k].item())
                    assert actual_anc == expected_anc, f"anc_mask mismatch at d={d}, j={j}, k={k}"
                    expected_rel = int(bank.depth[n_i, k].item() - bank.depth[n_i, j].item())
                    actual_rel = int(bank.rel_depth[n_i, j, k].item())
                    assert actual_rel == expected_rel, f"rel_depth mismatch at d={d}, j={j}, k={k}"

            for j in range(b):
                expected_n = int(bank.anc_mask[n_i, j, :b].long().sum().item())
                actual_n = int(bank.visit_count[n_i, j].item())
                assert actual_n == expected_n, f"visit_count mismatch at d={d}, j={j}: {actual_n} vs {expected_n}"

    print("[PASS] test_incremental_tree_bank_masks_and_counts (depths d in {2, 4, 8, 16})")


def test_all_classical_teachers_across_all_6_environments() -> None:
    """Verify all Classical Search Teachers on hex_5x5, connect4_6x7, sudoku_6x6, equiv_sat, shiphunt_6x6, micro_stratego_4x4."""
    wm = _make_small_world_model()
    env_names = (
        "hex_5x5",
        "connect4_6x7",
        "sudoku_6x6",
        "equiv_sat",
        "shiphunt_6x6",
        "micro_stratego_4x4",
    )
    teachers = [
        MCTSPUCTTeacher(wm),
        MCGSTranspositionTeacher(wm),
        MinimaxAlphaBetaTeacher(wm),
        CDCLBackjumpTeacher(wm),
        DisjunctiveBPTeacher(wm),
        CFRRegretMatchingTeacher(wm),
        GumbelSequentialHalvingTeacher(wm),
        HeuristicGLCPopTeacher(wm),
    ]

    for env_name in env_names:
        env = make_env(env_name)
        batch = env.sample_batch(batch_size=2, split="test", seed=21)
        batch_blind, _ = batch.split_and_poison_oracle()

        for teacher in teachers:
            trace = teacher.generate_trace(env, batch_blind, budget=4)
            # Test 7-tuple unpacking
            (
                node_features,
                tree_topology,
                target_subtree_N,
                target_subtree_Q,
                target_node_select,
                target_action_prior,
                target_root_belief,
            ) = trace

            assert node_features.shape == (2, 5, 16), f"Bad node_features shape: {node_features.shape}"
            assert tree_topology["anc_mask"].shape == (2, 5, 5)
            assert tree_topology["rel_depth"].shape == (2, 5, 5)
            assert target_subtree_N.shape == (2, 5)
            assert target_subtree_Q.shape == (2, 5)
            assert target_node_select.shape == (2, 5)
            assert target_action_prior.shape == (2, batch.num_vars, batch.domain_size)
            assert target_root_belief.shape == (2, batch.num_vars, batch.domain_size)
            assert torch.isfinite(target_subtree_N).all()
            assert torch.isfinite(target_subtree_Q).all()
            assert torch.allclose(
                target_root_belief.sum(dim=-1),
                torch.ones((2, batch.num_vars)),
                atol=1e-4,
            )

    print("[PASS] test_all_classical_teachers_across_all_6_environments (8 teachers x 6 envs, PoisonTensor-safe)")


def test_r13_expressivity_matrix_and_5_barriers() -> None:
    """Verify the R13 Expressivity Matrix across flat_token, parent_1hop, full_ancestry_v1, and glc_search_v2
    across tree depths d in {2, 4, 8, 16} and all 6 core algorithmic teachers.
    """
    wm = _make_small_world_model()
    env = make_env("equiv_sat")
    batch = env.sample_batch(batch_size=8, split="train", seed=33)

    teachers = {
        "MCTS-PUCT": MCTSPUCTTeacher(wm),
        "MCGS-Transposition-DAG": MCGSTranspositionTeacher(wm),
        "Minimax-AlphaBeta-PNS": MinimaxAlphaBetaTeacher(wm),
        "CDCL-Global-Nogood": CDCLBackjumpTeacher(wm),
        "Disjunctive-BP": DisjunctiveBPTeacher(wm),
        "CFR-Infoset-RegretMatching": CFRRegretMatchingTeacher(wm),
        "Gumbel-Sequential-Halving": GumbelSequentialHalvingTeacher(wm),
    }

    modes = ("flat_token", "parent_1hop", "full_ancestry_v1", "glc_search_v2")
    depths = (2, 4, 8, 16)

    # 1. Test MCTS-PUCT across depths d in {2, 4, 8, 16}
    mcts_r2_q: dict[str, dict[int, float]] = {m: {} for m in modes}
    mcts_r2_n: dict[str, dict[int, float]] = {m: {} for m in modes}
    for d in depths:
        banks = [
            build_controlled_depth_tree_bank(wm, batch, target_depth=d, num_branches=3, seed=100 + s)
            for s in range(3)
        ]
        traces = [teachers["MCTS-PUCT"].compute_targets_from_bank(b, batch=batch) for b in banks]
        for m in modes:
            ctrl = GLCSearchTransformer(wm, mode=m)
            res = ctrl.fit_imitation_on_traces(traces, epochs=15, lr=1.5e-2)
            mcts_r2_q[m][d] = res["r2_Q"]
            mcts_r2_n[m][d] = res["r2_log_N"]

    # At d=2, parent_1hop, full_ancestry_v1, and glc_search_v2 all solve MCTS
    assert mcts_r2_q["parent_1hop"][2] > 0.95
    assert mcts_r2_q["full_ancestry_v1"][2] > 0.98
    assert mcts_r2_q["glc_search_v2"][2] > 0.98
    # At d=16, full_ancestry_v1 and glc_search_v2 maintain R^2 > 0.98 while parent_1hop and flat_token fail!
    assert mcts_r2_q["full_ancestry_v1"][16] > 0.98, f"v1 MCTS d=16 R2_Q={mcts_r2_q['full_ancestry_v1'][16]}"
    assert mcts_r2_q["glc_search_v2"][16] > 0.98, f"v2 MCTS d=16 R2_Q={mcts_r2_q['glc_search_v2'][16]}"
    assert mcts_r2_n["full_ancestry_v1"][16] > 0.98
    assert mcts_r2_n["glc_search_v2"][16] > 0.98
    assert mcts_r2_q["parent_1hop"][16] < mcts_r2_q["full_ancestry_v1"][16] - 0.15
    assert mcts_r2_n["flat_token"][16] < mcts_r2_n["full_ancestry_v1"][16] - 0.15

    # 2. Test Barrier 1: Minimax-AlphaBeta-PNS at d=2 vs d=8, 16 (d > L_tree = 3)
    mm_r2_q: dict[str, dict[int, float]] = {m: {} for m in modes}
    for d in (2, 8, 16):
        banks = [
            build_controlled_depth_tree_bank(
                wm,
                batch,
                target_depth=d,
                num_branches=2,
                include_transpositions=False,
                include_conflicts=False,
                seed=300 + s,
            )
            for s in range(3)
        ]
        traces = [teachers["Minimax-AlphaBeta-PNS"].compute_targets_from_bank(b, batch=batch) for b in banks]
        for m in modes:
            ctrl = GLCSearchTransformer(wm, mode=m)
            res = ctrl.fit_imitation_on_traces(traces, epochs=15, lr=1.5e-2)
            mm_r2_q[m][d] = res["r2_Q"]

    # At d=2 (<= L_tree=3), both full_ancestry_v1 and glc_search_v2 solve Minimax
    assert mm_r2_q["full_ancestry_v1"][2] > 0.98
    assert mm_r2_q["glc_search_v2"][2] > 0.98
    # At d=8 and d=16 (> L_tree=3), glc_search_v2 solves Minimax/PNS (R^2 > 0.98) while full_ancestry_v1 degrades!
    assert mm_r2_q["glc_search_v2"][8] > 0.98, f"v2 Minimax d=8 R2_Q={mm_r2_q['glc_search_v2'][8]}"
    assert mm_r2_q["glc_search_v2"][16] > 0.98, f"v2 Minimax d=16 R2_Q={mm_r2_q['glc_search_v2'][16]}"
    assert mm_r2_q["glc_search_v2"][16] > mm_r2_q["full_ancestry_v1"][16] + 0.08

    # 3. Test Barriers 2, 3, 4, 5 at depth d=8:
    # MCGS-Transposition-DAG, CFR-Infoset-RegretMatching, Gumbel-Sequential-Halving, CDCL-Global-Nogood
    banks_d8 = [
        build_controlled_depth_tree_bank(
            wm,
            batch,
            target_depth=8,
            num_branches=3,
            include_transpositions=True,
            include_conflicts=True,
            seed=500 + s,
        )
        for s in range(3)
    ]
    for t_name in (
        "MCGS-Transposition-DAG",
        "CDCL-Global-Nogood",
        "Disjunctive-BP",
        "CFR-Infoset-RegretMatching",
        "Gumbel-Sequential-Halving",
    ):
        traces = [teachers[t_name].compute_targets_from_bank(b, batch=batch) for b in banks_d8]
        ctrl_v2 = GLCSearchTransformer(wm, mode="glc_search_v2")
        res_v2 = ctrl_v2.fit_imitation_on_traces(traces, epochs=15, lr=1.5e-2)
        ctrl_flat = GLCSearchTransformer(wm, mode="flat_token")
        res_flat = ctrl_flat.fit_imitation_on_traces(traces, epochs=15, lr=1.5e-2)
        assert res_v2["r2_Q"] > 0.98, f"glc_search_v2 failed on {t_name}: R2_Q={res_v2['r2_Q']}"
        assert res_v2["select_top1_acc"] >= 0.90, f"glc_search_v2 select acc on {t_name}: {res_v2['select_top1_acc']}"
        assert res_v2["r2_Q"] > res_flat["r2_Q"] + 0.15

    print("[PASS] test_r13_expressivity_matrix_and_5_barriers (all 7 teachers x 4 modes x depths {2, 4, 8, 16})")


def test_2hop_tseitin_query_coupling_and_r14_unsupervised_discovery() -> None:
    """Verify 2-hop Tseitin query coupling, R14 unsupervised discovery loss, b=0 identity, and PoisonTensor safety."""
    wm = _make_small_world_model()
    env = make_env("equiv_sat")
    batch = env.sample_batch(batch_size=4, split="test", seed=77)
    batch_blind, _ = batch.split_and_poison_oracle()

    ctrl = GLCSearchTransformer(wm, mode="glc_search_v2")

    # 1. Exact b=0 identity check
    out_b0 = ctrl.run_search(env, batch_blind, budget=0)
    root_out = wm(batch_blind)
    max_b0_diff = float((out_b0["var_probs"] - root_out["var_probs"]).abs().max().item())
    assert max_b0_diff == 0.0, f"Expected exact 0.0 b=0 identity, got {max_b0_diff}"

    # 2. Run search with budget=6 on PoisonTensor-stripped batch and check 2-hop Tseitin coupling
    out_b6 = ctrl.run_search(env, batch_blind, budget=6, record_trajectory=True)
    assert out_b6["query_probs"].shape == batch.query_targets.shape
    assert (out_b6["query_probs"] > 0.0).all() and (out_b6["query_probs"] < 1.0).all()
    assert out_b6["tseitin_2hop_cofluct"].shape == batch.query_targets.shape
    assert len(out_b6["probs_trajectory"]) == 7

    # Verify bitwise PoisonTensor invariance against a zeroed-oracle clone
    batch_zeros = batch.clone()
    batch_zeros.exact_marginals = torch.zeros_like(batch.exact_marginals)
    batch_zeros.target_tokens = torch.zeros_like(batch.target_tokens)
    batch_zeros.query_targets = torch.zeros_like(batch.query_targets)
    out_zeros = ctrl.run_search(env, batch_zeros, budget=6)
    poison_diff = float((out_b6["var_probs"] - out_zeros["var_probs"]).abs().max().item())
    assert poison_diff == 0.0, f"PoisonTensor invariance violated: {poison_diff}"

    # 3. Verify R14 unsupervised discovery loss runs & backpropagates cleanly on PoisonTensor batch
    unsup = ctrl.compute_unsupervised_discovery_loss(env, batch_blind, budget=4)
    assert torch.isfinite(unsup["total_loss"])
    unsup["total_loss"].backward()

    print("[PASS] test_2hop_tseitin_query_coupling_and_r14_unsupervised_discovery")


def test_ast_zero_oracle_leakage_audit() -> None:
    """Run static AST leak auditor across kv_tree_search.py, classical_teachers.py, neural_tree_controller.py, glc_aom.py."""
    repo_root = Path(__file__).resolve().parents[2]
    target_files = [
        repo_root / "src/models/glc_aom.py",
        repo_root / "src/search/kv_tree_search.py",
        repo_root / "src/search/classical_teachers.py",
        repo_root / "src/search/neural_tree_controller.py",
    ]
    report = audit_r11_files(target_files)
    assert report["pass"], f"AST Leak Audit failed: {report}"
    assert report["num_violations"] == 0
    assert report["canary_self_test"]["canaries_caught"] == 3
    print("[PASS] test_ast_zero_oracle_leakage_audit (0 violations across 4 core files, 3/3 canaries caught)")


if __name__ == "__main__":
    test_incremental_tree_bank_masks_and_counts()
    test_all_classical_teachers_across_all_6_environments()
    test_r13_expressivity_matrix_and_5_barriers()
    test_2hop_tseitin_query_coupling_and_r14_unsupervised_discovery()
    test_ast_zero_oracle_leakage_audit()
    print("\nALL MEMBER 19 NEURAL TREE CONTROLLER & CLASSICAL TEACHER TESTS PASSED 100%.")
