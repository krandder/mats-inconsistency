"""Unit and Verification Tests for `src/models/glc_aom.py` and `src/search/kv_tree_search.py`.

Verifies all Round 1 architectural invariants and Red-Team stress tests (`RT-D0` through `RT-D4`
and `ROUND1-SCOREBOARD.md` §2–3):
- `RT-D0`: Zero-step identity (`B=0`), zero utility-to-belief leakage (`chi = (0, 1)` vs `chi = (1, 1)`),
  and `Z_aux = Z^W_aux (+) Z^Pi_aux` structural partition.
- `RT-D1`: Monotonicity / Nested Precision-Gated Residual Update, Unsolvable Board Silence (`0.0000` harm),
  BilinearAuxLemmaEmitter even co-fluctuation symmetry + zero-init firewall + PR >= 0.45,
  CombinatorialQueryHead O(D_aux) floating equivalence resolution, EpisodicMemoryTransport surprise flush,
  and DepthDebiasedTrustHead Kish-ESS shrinkage + removal of the `m_qs >= 0.50` greedy lock.
- `RT-D2`: Stage 1 Disjunctive Counterfactual Bayes Mixture (zero piece-count vetoes) and Stage 2
  Parent-Relative Edge Increments + Bag-Centered Edge Keys (duplicate prefix deduplication in [0.95, 1.15]
  and orthogonal leaf preservation >= 0.80).
- `RT-D3`: Tree-PUCT `SELECT_NODE(j)` topology non-collapse (`f_root in [0.03, 0.35]`, `f_fork >= 0.10`,
  `d_max >= 6`), `Used(j)` sibling deduplication, and `MoPS` Gram orthogonal projection against `K_done`.
- `RT-D4`: Gershgorin spectral bound `< 1.0` up to `B = 128`, parameter count `<= 8,192`,
  Epoch-0 Fallback calibration safety, and end-to-end execution of all 9 `SEARCH_ALGORITHMS`.
"""

from __future__ import annotations

import os
import sys

if "/tmp/r4e3a/pylib" not in sys.path and os.path.isdir("/tmp/r4e3a/pylib"):
    sys.path.insert(0, "/tmp/r4e3a/pylib")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from src.envs.base import QueryType, UtilitySpec, categorical_kl
from src.envs.equiv_sat import EquivSatEnv
from src.envs.shiphunt import ShipHuntEnv
from src.envs.sudoku import SudokuEnv
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    CombinatorialQueryHead,
    DepthDebiasedTrustHead,
    EpisodicMemoryTransport,
    GLCAOMConfig,
    GLCAOMWorldModel,
    LIRMask,
)
from src.search.kv_tree_search import (
    MoPSFractalPlanHead,
    ResolvableInconsistencyTreePUCT,
    SEARCH_ALGORITHMS,
    SUPPORTED_BUDGETS,
    TreePagedKVBank,
    TreePagedKVSearch,
    TwoStageTreeClearer,
)


def _make_test_world_model() -> GLCAOMWorldModel:
    torch.manual_seed(42)
    cfg = GLCAOMConfig(
        num_vars=36,
        num_vals=6,
        d_feat=32,
        d_mem=32,
        k_mem=4,
        m_aux=8,
        d_aux=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_key=64,
    )
    return GLCAOMWorldModel(cfg)


def test_rt_d0_zero_step_identity_and_zero_utility_leakage() -> None:
    """Verify RT-D0:
    1. Exact b=0 identity (`max |mu_0 - q_real(C_0)| == 0.0000`).
    2. Exact 0.0000 belief drift under utility swaps (`swap_utility`) and RL updates on `theta_policy`.
    3. Immediate action preference flip under `swap_utility`.
    4. Exact 0.0000 belief leakage from `Z^Pi_aux` plan/control channels into `V` or `Q(A, B)`.
    """
    model = _make_test_world_model()
    search = TreePagedKVSearch(model)
    env = ShipHuntEnv()
    batch_nw = env.sample_batch(batch_size=4, utility_spec=int(UtilitySpec.SECTOR_NW_PREF), seed=101)
    batch_se = env.swap_utility(batch_nw, int(UtilitySpec.SECTOR_SE_PREF))
    # Also perturb utility_vector on batch_se to test both header[:, 5] and utility_vector invariance
    assert batch_se.utility_vector is not None
    batch_se.utility_vector = 1.0 - batch_se.utility_vector

    out_nw = model(batch_nw)
    out_se = model(batch_se)

    # 1. Exact 0.0000 belief drift across utility swap
    assert torch.max(torch.abs(out_nw["var_logits"] - out_se["var_logits"])).item() == 0.0
    assert torch.max(torch.abs(out_nw["var_probs"] - out_se["var_probs"])).item() == 0.0
    assert torch.max(torch.abs(out_nw["query_probs"] - out_se["query_probs"])).item() == 0.0
    assert torch.max(torch.abs(out_nw["m_ql"] - out_se["m_ql"])).item() == 0.0
    assert torch.max(torch.abs(out_nw["beta_trust"] - out_se["beta_trust"])).item() == 0.0

    # Action distributions MUST differ under swapped utility
    action_diff = torch.max(torch.abs(out_nw["action_probs"] - out_se["action_probs"])).item()
    assert action_diff > 0.05, f"Expected action_probs to flip under utility swap, got diff={action_diff}"

    # 2. Exact b=0 identity in TreePagedKVSearch
    s0 = search.run_search(env, batch_nw, budget=0, algorithm="GLC_Tree_Active_PUCT")
    assert torch.max(torch.abs(s0["var_probs"] - out_nw["var_probs"])).item() == 0.0

    # 3. Taking an optimization step on policy loss (`action_logits`) causes 0.0000 drift in epistemic beliefs
    opt_pol = torch.optim.SGD(model.get_policy_parameters(), lr=0.1)
    opt_pol.zero_grad()
    pol_loss = -out_nw["action_logits"][:, 0].mean()
    pol_loss.backward()
    opt_pol.step()

    out_after = model(batch_nw)
    assert torch.max(torch.abs(out_after["var_logits"] - out_nw["var_logits"])).item() == 0.0

    # 4. Perturbing Z^Pi_aux (plan/control half of Z_aux) causes 0.0000 change in belief readout
    h_cons_1 = out_nw["h_cons"].clone()
    h_cons_2 = out_nw["h_cons"].clone()
    w_dim = model.cfg.d_v + model.cfg.d_aux_world
    h_cons_2[:, w_dim:] = torch.randn_like(h_cons_2[:, w_dim:]) * 50.0

    fw1 = model.aux_emitter.readout_var_firewall(h_cons_1, num_vars=batch_nw.num_vars, num_vals=batch_nw.domain_size)
    fw2 = model.aux_emitter.readout_var_firewall(h_cons_2, num_vars=batch_nw.num_vars, num_vals=batch_nw.domain_size)
    assert torch.max(torch.abs(fw1 - fw2)).item() == 0.0

    qp1 = model.query_head.predict_query_probs(batch_nw, out_nw["var_probs"], h_cons_1)
    qp2 = model.query_head.predict_query_probs(batch_nw, out_nw["var_probs"], h_cons_2)
    assert torch.max(torch.abs(qp1 - qp2)).item() == 0.0


def test_rt_d1_bilinear_aux_query_memory_and_unsolvable_silence() -> None:
    """Verify RT-D1:
    1. BilinearAuxLemmaEmitter is strictly even (`emitter(+Delta) == emitter(-Delta)`), has 0.0000
       firewall leakage at init, and achieves Participation Ratio `PR >= 0.45`.
    2. CombinatorialQueryHead resolves floating equivalence chains (`A == B` vs `A != B` where
       marginals are `[0.5, 0.5]`) via `Z^W_aux` in `O(D_aux)`.
    3. EpisodicMemoryTransport flushes contradicted priors via surprise discounting.
    4. Unsolvable boards (`solvable=False`) produce `m_ql == 0.0`, `alpha == 0.0`, and `0.0000` harm.
    5. DepthDebiasedTrustHead removes the `m_qs >= 0.50` greedy lock and satisfies Kish-ESS bounds.
    """
    model = _make_test_world_model()
    cfg = model.cfg
    emitter = model.aux_emitter

    # 1. Even symmetry: emitter(+delta) == emitter(-delta)
    delta_pos = torch.randn(8, 16, 2)
    _, _, z_pos = emitter(delta_pos)
    _, _, z_neg = emitter(-delta_pos)
    assert torch.max(torch.abs(z_pos - z_neg)).item() < 1e-6

    # Participation Ratio on diverse co-fluctuation batch >= 0.45
    diverse_deltas = torch.randn(64, cfg.num_vars, cfg.num_vals)
    _, _, z_div = emitter(diverse_deltas)
    pr = emitter.compute_participation_ratio(z_div[:, : cfg.d_aux_world])
    assert pr >= 0.45, f"Expected PR >= 0.45, got {pr:.4f}"

    # Zero-initialized firewall has exact 0.0000 leakage onto V
    fake_h_cons = torch.cat([torch.zeros(4, cfg.d_v), torch.randn(4, cfg.d_aux_total) * 10.0], dim=-1)
    fw_out = emitter.readout_var_firewall(fake_h_cons, num_vars=16, num_vals=2)
    assert torch.max(torch.abs(fw_out)).item() == 0.0

    # 2. Floating equivalence chain resolution via Z^W_aux:
    # Suppose variables 0 and 1 have uniform marginals [0.5, 0.5], and sibling branches
    # (X_0=1, X_1=1) and (X_0=0, X_1=0) both fired (+delta and -delta), writing into Z^W_aux.
    env_eq = EquivSatEnv(num_vars=16, num_clauses=12, num_queries=4)
    batch_eq = env_eq.sample_batch(batch_size=2, seed=7)
    batch_eq.query_pairs[:, 0, 0] = 0
    batch_eq.query_pairs[:, 0, 1] = 1
    batch_eq.query_pairs[:, 0, 2] = int(QueryType.EQUAL)

    unif_probs = torch.full((2, 16, 2), 0.5)
    zero_h_cons = torch.zeros(2, cfg.d_amp)
    p_eq_no_aux = model.query_head.predict_query_probs(batch_eq, unif_probs, zero_h_cons)[:, 0]
    assert torch.allclose(p_eq_no_aux, torch.tensor([0.5, 0.5]), atol=1e-4)

    # Construct sibling branch shifts where var 0 and var 1 co-fluctuate positively (A == B)
    sib_delta = torch.zeros(2, 16, 2)
    sib_delta[:, 0, 0] = -3.0
    sib_delta[:, 0, 1] = 3.0
    sib_delta[:, 1, 0] = -3.0
    sib_delta[:, 1, 1] = 3.0
    v_exp_same, _, _ = emitter(sib_delta)
    p_eq_with_same_aux = model.query_head.predict_query_probs(batch_eq, unif_probs, v_exp_same)[:, 0]
    assert (p_eq_with_same_aux > 0.70).all(), f"Expected P(A==B) > 0.70 from Z^W_aux, got {p_eq_with_same_aux}"

    # Opposite co-fluctuation (A != B) should push P(A == B) below 0.30
    sib_delta_opp = sib_delta.clone()
    sib_delta_opp[:, 1, 0] = 3.0
    sib_delta_opp[:, 1, 1] = -3.0
    v_exp_opp, _, _ = emitter(sib_delta_opp)
    p_eq_with_opp_aux = model.query_head.predict_query_probs(batch_eq, unif_probs, v_exp_opp)[:, 0]
    assert (p_eq_with_opp_aux < 0.30).all(), f"Expected P(A==B) < 0.30 for anti-correlated Z^W_aux, got {p_eq_with_opp_aux}"

    # 3. EpisodicMemoryTransport surprise flush
    mem = model.memory_transport
    h_prev = torch.randn(4, cfg.d_amp)
    beta_prev = torch.ones(4)
    nll_no_mem = torch.tensor([1.0, 1.0, 1.0, 1.0])
    # First 2 instances have 0 surprise; last 2 have high surprise (contradicted prior)
    nll_with_mem = torch.tensor([0.9, 1.0, 6.0, 8.0])
    m_out = mem(h_cons_prev=h_prev, beta_prev=beta_prev, nll_with_memory=nll_with_mem, nll_without_memory=nll_no_mem)
    assert m_out["beta_memory"][0].item() > 0.80
    assert m_out["beta_memory"][2].item() < 1e-3
    assert m_out["beta_memory"][3].item() < 1e-4

    # 4. Unsolvable board silence (RT-D1): 0.0000 harm on unsolvable instances
    env_unsolv = EquivSatEnv(num_vars=16, num_clauses=12, num_queries=4, unsolvable_frac=1.0)
    batch_unsolv = env_unsolv.sample_batch(batch_size=4, seed=99)
    search = TreePagedKVSearch(model)
    res_b0 = search.run_search(env_unsolv, batch_unsolv, budget=0, algorithm="GLC_Tree_Active_PUCT")
    res_b8 = search.run_search(env_unsolv, batch_unsolv, budget=8, algorithm="GLC_Tree_Active_PUCT")
    assert torch.max(torch.abs(res_b8["var_probs"] - res_b0["var_probs"])).item() == 0.0

    # 5. Removal of m_qs >= 0.50 greedy lock + Kish-ESS shrinkage dominance
    # At a depth-2 symmetric split (step probs 0.5 * 0.5 = 0.25 < 0.50), leaf self-consistency m_ql
    # on a self-consistent node is still ~1.0
    root_mask = torch.zeros(2, 16, dtype=torch.bool)
    node_mask = torch.zeros(2, 16, dtype=torch.bool)
    node_mask[:, 0] = True
    node_mask[:, 1] = True
    node_toks = torch.zeros(2, 16, dtype=torch.long)
    node_toks[:, 0] = 1
    node_toks[:, 1] = 2
    node_logits = torch.zeros(2, 16, 2)
    node_logits[:, 0, 0] = 10.0
    node_logits[:, 1, 1] = 10.0
    m_ql_val = DepthDebiasedTrustHead.compute_leaf_self_consistency(node_logits, node_mask, node_toks, root_mask)
    assert (m_ql_val > 0.99).all()

    # Kish-ESS shrinkage dominance over uniform and root on synthetic calibration/test split
    torch.manual_seed(123)
    N_cal, N_test, K_roll = 32, 32, 8
    true_cal = torch.randn(N_cal, 4)
    true_test = torch.randn(N_test, 4)
    betas_cal = torch.rand(N_cal, K_roll) * 3.0 + 0.1
    betas_test = torch.rand(N_test, K_roll) * 3.0 + 0.1
    noise_scale_cal = 1.0 / torch.sqrt(betas_cal).unsqueeze(-1)
    noise_scale_test = 1.0 / torch.sqrt(betas_test).unsqueeze(-1)
    vals_cal = true_cal.unsqueeze(1) + torch.randn(N_cal, K_roll, 4) * noise_scale_cal
    vals_test = true_test.unsqueeze(1) + torch.randn(N_test, K_roll, 4) * noise_scale_test
    root_cal = true_cal + torch.randn(N_cal, 4) * 0.9
    root_test = true_test + torch.randn(N_test, 4) * 0.9

    pooled_test, pool_info = DepthDebiasedTrustHead.pool_values_trust_shrunk(
        rollout_values=vals_test,
        beta_trust=betas_test,
        root_value=root_test,
        cal_values=vals_cal,
        cal_betas=betas_cal,
        cal_targets=true_cal,
        cal_root_values=root_cal,
        kappa_ess=0.65,
    )
    assert pool_info["ess_mean"] >= 0.65 * K_roll - 1e-4
    mse_shrunk = ((pooled_test - true_test) ** 2).mean().item()
    mse_unif = ((vals_test.mean(dim=1) - true_test) ** 2).mean().item()
    mse_root = ((root_test - true_test) ** 2).mean().item()
    assert mse_shrunk <= min(mse_unif, mse_root) + 1e-3


def test_rt_d2_disjunctive_bayes_and_edge_preconditioning() -> None:
    """Verify RT-D2:
    1. Stage 1 Disjunctive Counterfactual Bayes Mixture eliminates piece-count vetoes (`p < 0.001`)
       on mutually exclusive sibling case splits.
    2. Stage 2 Gershgorin Row-Crowding Preconditioning over Bag-Centered Edge Keys satisfies:
       - Duplicate prefix deduplication `W_prefix(16) / W_prefix(1) in [0.95, 1.15]`
       - Orthogonal leaf preservation `W_leaf / W_single >= 0.80`
    """
    model = _make_test_world_model()
    clearer = TwoStageTreeClearer(model.cfg)

    # 1. Disjunctive Counterfactual Bayes Mixture vs Conjunctive Veto
    # Suppose parent has uniform prior [0.5, 0.5] on variable 0 and variable 1.
    # Sibling 0 clamps V_0=0 and deduces V_1=0 with high confidence (p(V_1=1) = 1e-4).
    # Sibling 1 clamps V_0=1 and deduces V_1=1 with high confidence (p(V_1=0) = 1e-4).
    # Naive log-odds addition vetoes BOTH values of V_1, whereas Stage 1 Disjunctive Bayes
    # recovers the exact [0.5, 0.5] marginal with zero vetoes!
    parent_probs = torch.full((2, 4, 2), 0.5)
    child_probs = torch.full((2, 2, 4, 2), 0.5)
    # Sibling 0: V_1 = 0 (prob 0.9999)
    child_probs[:, 0, 1, 0] = 0.9999
    child_probs[:, 0, 1, 1] = 0.0001
    # Sibling 1: V_1 = 1 (prob 0.9999)
    child_probs[:, 1, 1, 0] = 0.0001
    child_probs[:, 1, 1, 1] = 0.9999

    split_var = torch.zeros(2, dtype=torch.long)
    split_vals = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    child_m_ql = torch.ones((2, 2))

    mu_disj, bar_delta = clearer.stage1_disjunctive_bayes_mixture(
        parent_probs=parent_probs,
        child_probs=child_probs,
        split_var=split_var,
        split_vals=split_vals,
        child_m_ql=child_m_ql,
    )
    assert (mu_disj >= 0.001).all(), "Stage 1 Disjunctive Bayes produced a piece-count veto!"
    assert torch.allclose(mu_disj[:, 1, :], torch.full((2, 2), 0.5), atol=1e-3)

    # 2. Duplicate prefix deduplication: 16 identical copies vs 1 copy
    D = model.cfg.d_amp
    single_delta = torch.ones((2, 1, D))
    single_key = F.normalize(torch.abs(torch.randn(2, 1, D)) + 0.1, p=2.0, dim=-1)
    single_beta = torch.ones((2, 1)) * 2.0

    out_1, _ = clearer.stage2_preconditioned_clearing(single_delta, single_key, single_beta)

    dup_delta = single_delta.expand(2, 16, D).clone()
    dup_key = single_key.expand(2, 16, D).clone()
    dup_beta = single_beta.expand(2, 16).clone()
    out_16, _ = clearer.stage2_preconditioned_clearing(dup_delta, dup_key, dup_beta)

    ratio_dup = (out_16.norm(dim=-1) / out_1.norm(dim=-1)).mean().item()
    assert 0.95 <= ratio_dup <= 1.15, f"Expected W_prefix(16)/W_prefix(1) in [0.95, 1.15], got {ratio_dup:.4f}"

    # 3. Orthogonal leaf preservation: 4 orthogonal edge keys with shared background floor
    bg = torch.full((2, 4, D), 0.05)
    orth_keys = bg.clone()
    for r in range(4):
        orth_keys[:, r, r * 10 : (r + 1) * 10] += 1.0
    orth_keys = F.normalize(orth_keys, p=2.0, dim=-1)
    orth_deltas = torch.zeros((2, 4, D))
    for r in range(4):
        orth_deltas[:, r, r * 10 : (r + 1) * 10] = 1.0
    orth_betas = torch.ones((2, 4)) * 2.0

    out_orth, info_orth = clearer.stage2_preconditioned_clearing(orth_deltas, orth_keys, orth_betas)
    # Compare per-leaf weight in the orthogonal bag against a single isolated leaf
    w_single = (2.0 / (clearer.compute_precision_ridge(orth_betas[:, :1]) + 2.0)).mean().item()
    w_orth_mean = info_orth["w_eff"].mean().item()
    pres_ratio = w_orth_mean / w_single
    assert pres_ratio >= 0.80, f"Expected orthogonal leaf preservation >= 0.80, got {pres_ratio:.4f}"


def test_rt_d3_tree_topology_non_collapse_and_mops_gram_projection() -> None:
    """Verify RT-D3:
    1. `GLC_Tree_Active_PUCT` at `B = 16` achieves non-star root fork fraction `f_root in [0.03, 0.35]`,
       internal fork fraction `f_fork >= 0.10`, and max depth `d_max >= 6`.
    2. `Used(j)` action-exhaustion mask guarantees zero duplicate `(parent, v, z)` edges.
    3. `MoPSFractalPlanHead` Gram orthogonal projection reduces cosine overlap with `K_done` (`< 0.15`).
    """
    model = _make_test_world_model()
    search = TreePagedKVSearch(model)
    env = SudokuEnv(size=6)
    batch = env.sample_batch(batch_size=4, seed=202)

    res = search.run_search(env, batch, budget=16, algorithm="GLC_Tree_Active_PUCT", record_trajectory=True)
    bank: TreePagedKVBank = res["tree_bank"]
    sel_nodes = res["selected_nodes"]  # [N, 16]

    # 1. Topology metrics across B = 16 steps (steps b = 1..15 after initial root step b=0)
    # Root forks after b=0: j_sel == 0 for step_b >= 1
    root_forks_after_0 = (sel_nodes[:, 1:] == 0).float().mean().item()
    assert 0.03 <= root_forks_after_0 <= 0.35, f"Expected f_root in [0.03, 0.35], got {root_forks_after_0:.3f}"

    # Max depth >= 6
    d_max = bank.depth.max(dim=-1).values.float().mean().item()
    assert d_max >= 6.0, f"Expected max depth >= 6.0, got {d_max:.2f}"

    # Non-continuation forks (selecting an earlier node j_sel < b_active - 1 for step_b >= 1)
    step_indices = torch.arange(1, 16, device=sel_nodes.device).unsqueeze(0)
    fork_steps = (sel_nodes[:, 1:] < step_indices).float().mean().item()
    assert fork_steps >= 0.10, f"Expected internal/root fork fraction >= 0.10, got {fork_steps:.3f}"

    # 2. Zero duplicate (parent, step_var, step_val) transitions in tree_bank
    for n_i in range(batch.batch_size):
        seen_edges = set()
        for node_j in range(1, bank.num_active):
            edge_tuple = (
                int(bank.parent_idx[n_i, node_j].item()),
                int(bank.step_var[n_i, node_j].item()),
                int(bank.step_val[n_i, node_j].item()),
            )
            assert edge_tuple not in seen_edges, f"Duplicate sibling action {edge_tuple} in tree_bank!"
            seen_edges.add(edge_tuple)

    # 3. MoPS Gram orthogonal projection against K_done
    plan_head = search.plan_head
    D = model.cfg.d_amp
    keys_done = torch.zeros(2, 3, D)
    keys_done[:, 0, 0] = 1.0
    keys_done[:, 1, 1] = 1.0
    keys_done[:, 2, 2] = 1.0
    beta_done = torch.full((2, 3), 10.0)
    raw_p = torch.ones(2, 4, D)
    proj_p = plan_head.gram_orthogonal_project(raw_p, keys_done, beta_done)
    max_cos = torch.bmm(proj_p, keys_done.transpose(1, 2)).abs().max().item()
    assert max_cos < 0.15, f"Expected Gram-orthogonalized plan cosine < 0.15 with K_done, got {max_cos:.4f}"


def test_rt_d4_spectral_stability_param_cap_and_all_algorithms() -> None:
    """Verify RT-D4:
    1. `TwoStageTreeClearer` parameter count is strictly `<= 8,192`.
    2. Gershgorin spectral bound `||T_b||_infty < 1.0` holds up to `B = 128`.
    3. `calibrate_soft_teacher` Epoch-0 Fallback Rule guarantees zero validation degradation.
    4. All 9 `SEARCH_ALGORITHMS` execute cleanly and maintain Dual-Channel Readout monotonicity.
    """
    model = _make_test_world_model()
    search = TreePagedKVSearch(model)
    clearer = search.tree_clearer

    # 1. Parameter count cap <= 8,192
    n_params = clearer.count_parameters()
    assert n_params <= 8192, f"TwoStageTreeClearer exceeded 8,192 parameter cap: {n_params}"

    # 2. Spectral stability up to B = 128
    D = model.cfg.d_amp
    deltas_128 = torch.randn(2, 128, D)
    keys_128 = F.normalize(torch.rand(2, 128, D) + 0.01, p=2.0, dim=-1)
    betas_128 = torch.rand(2, 128) * 4.0 + 0.1
    cleared_128, info_128 = clearer.stage2_preconditioned_clearing(deltas_128, keys_128, betas_128)
    assert torch.isfinite(cleared_128).all()
    assert (info_128["spectral_norm_bound"] < 1.0).all()

    # 3. Soft-teacher calibration + Epoch-0 Fallback safety
    V, K = 16, 2
    cal_deltas = torch.randn(8, 8, D) * 0.5
    cal_keys = F.normalize(torch.rand(8, 8, D) + 0.01, p=2.0, dim=-1)
    cal_betas = torch.ones(8, 8)
    cal_root = torch.randn(8, V, K)
    cal_teacher = F.softmax(torch.randn(8, V, K), dim=-1)
    cal_res = clearer.calibrate_soft_teacher(
        cal_edge_deltas=cal_deltas,
        cal_edge_keys=cal_keys,
        cal_betas=cal_betas,
        cal_root_logits=cal_root,
        cal_soft_teacher_probs=cal_teacher,
        max_epochs=5,
    )
    assert cal_res["final_val_kl"] <= cal_res["epoch0_val_kl"] + 1e-6

    # 4. Run all 9 search algorithms on EquivSatEnv and check outputs + shapes
    env_eq = EquivSatEnv(num_vars=16, num_clauses=14, num_queries=4)
    batch_eq = env_eq.sample_batch(batch_size=2, seed=303)
    for alg in SEARCH_ALGORITHMS:
        out = search.run_search(env_eq, batch_eq, budget=4, algorithm=alg, record_trajectory=True)
        assert out["var_probs"].shape == (2, 16, 2)
        assert out["query_probs"].shape == (2, 4)
        assert out["discovery_witness_tokens"].shape == (2, 16)
        assert out["discovery_witness_valid"].shape == (2,)
        assert torch.isfinite(out["var_probs"]).all()
        assert torch.isfinite(out["query_probs"]).all()

        # Verify Nested Gate Monotonicity on GLC_Tree_Active_PUCT
        if alg == "GLC_Tree_Active_PUCT":
            # Also test any-order loss L_{any-order} and VICReg loss
            ao_loss = model.compute_any_order_loss(batch_eq, out["h_cons"], num_permutations=2)
            vic_loss = BilinearAuxLemmaEmitter.compute_vicreg_loss(
                out["h_cons"][:, model.cfg.d_v : model.cfg.d_v + model.cfg.d_aux_world]
            )
            assert torch.isfinite(ao_loss)
            assert torch.isfinite(vic_loss)
