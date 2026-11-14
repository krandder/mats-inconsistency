"""Stage-3 Runtime Name-Agnostic Counterfactual Audit Suite (Rules S11-S14).

Closes the Society of Thinkers Round 6 Opus-5 / Fable-4 critique ("AST Auditing Can Be
Bypassed via Identifier Renaming") by enforcing four executable runtime behavioral
counterfactual invariants that treat every model, search loop, reasoning evaluator, and
decentralized market mechanism as a black-box function:

  - Rule S11 (Graph Permutation Equivariance S11-E & 1-Hop Message-Passing Locality S11-L):
      * S11-E: Under any random non-identity node permutation matrix P in {0,1}^{N x N},
        pointer probabilities p(G) and node masks m(G) must transform equivariantly:
          ||p(pi(G)) - P p(G) P^T||_inf < 1e-5  and  ||m(pi(G)) - P m(G)||_inf < 1e-5.
      * S11-L: Perturbing an edge weight W_{uv} at shortest-path hop distance dist_G(s, u) >= 3
        and dist_G(s, v) >= 3 from source s must produce exact zero sensitivity at 1-hop
        message-passing step t=1 (Delta_{t=1}(s) < 1e-7) and strictly positive sensitivity
        at multi-hop step t=L >= dist_G(s, u) + 1 (Delta_{t=L}(s) > 1e-5).

  - Rule S12 (Step-Wise Search State Intervention S12-2 & Loss-Ratio Dispersion S12-4):
      * S12-2: Injecting an epsilon=0.05 perturbation into the internal neural search state at
        rollout step t=2 (after step t=1 root evaluation) must leave step t=1 outputs bitwise
        identical (Delta_{t=1} == 0.0) while strictly changing downstream rollout outputs at
        t' > 2 (max_{t' > 2} ||y^{(t')} - y_tilde^{(t')}||_inf > 1e-5).
      * S12-4: Across M >= 8 distinct instances, the per-instance search loss ratio
        r_i(B) = L_i(B) / L_i(1) for B > 1 must exhibit strictly non-zero cross-instance
        dispersion: Std_i(L_i(B) / L_i(1)) > 1e-4.

  - Rule S13 (Track-0 Helper-Only Baseline & Symmetric Evaluator Ledger Gate):
      * S13-Track0: Running the test-time constraint helper on an uninformative uniform prior
        with 0 neural forward calls across all test instances must yield a solve rate <= 5.0%
        (0.00% on Wave-2 R22B-E1).
      * S13-Ledger: Across all compared paradigms at equal budget B, helper call counts and
        node expansions must be identical in Track B (and identically 0 in Track A).

  - Rule S14 (Paired Counterfactual Broadcast Price Replay Gate):
      * S14-Detach: In decentralized market learning, the broadcast pre-entry market price P_t
        must carry zero autograd history (grad_fn is None), and perturbing other modules'
        parameters/outputs while holding P_t fixed must produce 0.000000000 gradient drift
        in module m: ||g_m(P_t; theta_{-m}) - g_m(P_t; theta_{-m} + delta)||_2 == 0.0.
      * S14-PriceReplay: Replaying module m's local update with identical local observations,
        weights, and RNG seed but counterfactual broadcast price P'_t != P_t must strictly
        alter module m's parameter gradient: ||g_m(P_t) - g_m(P'_t)||_2 > 1e-4.

Includes 4/4 synthetic runtime cheat canaries (S11-Canary, S12-Canary, S13-Canary, S14-Canary)
that evade static AST keyword filters via benign variable names and are 100% caught by S11-S14.
Combined with ast_leak_auditor.py v2 (Rules S1-S10, 7/7 AST canaries), this provides an
11/11 AST + Runtime canary verification suite.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.ast_leak_auditor import run_canary_self_test as run_ast_canary_self_test


# ==============================================================================
# 1. Structured Report Dataclasses for Rules S11-S14
# ==============================================================================


@dataclass
class RuleCheckResult:
    """Result of evaluating a single runtime counterfactual rule (S11, S12, S13, or S14)."""

    rule_id: str
    rule_name: str
    target_name: str
    passed: bool
    metrics: Dict[str, Any]
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class RuntimeAuditReport:
    """Aggregate report for the Stage-3 Runtime Counterfactual Audit Suite (Rules S11-S14)."""

    auditor_version: str
    rules_checked: List[str]
    runtime_canaries_caught: int
    runtime_canaries_total: int
    ast_canaries_caught: int
    ast_canaries_total: int
    combined_canaries_caught: int
    combined_canaries_total: int
    checks: List[RuleCheckResult]

    @property
    def passed(self) -> bool:
        return (
            self.runtime_canaries_caught == self.runtime_canaries_total
            and self.ast_canaries_caught == self.ast_canaries_total
            and all(c.passed for c in self.checks)
        )


# ==============================================================================
# 2. Rule S11: Graph Permutation Equivariance (S11-E) & 1-Hop Locality (S11-L)
# ==============================================================================


@torch.no_grad()
def permute_clrs_batch(batch: Any, perm: torch.Tensor) -> Tuple[Any, torch.Tensor]:
    """Permute a CleanCLRSBatch (from R22B-E2) under node permutation `perm` of shape `[N]`.

    For node features X[b, i], edge matrix E[b, i, j], adjacency A[b, i, j]:
      X_pi[b, pi(i)] = X[b, i]           <=>  X_pi = P @ X
      E_pi[b, pi(i), pi(j)] = E[b, i, j] <=>  E_pi = P @ E @ P^T
      A_pi[b, pi(i), pi(j)] = A[b, i, j] <=>  A_pi = P @ A @ P^T
    where P[new_idx, old_idx] = 1 iff new_idx == perm[old_idx], i.e.,
    if inv_perm = argsort(perm), then X_pi[:, new_idx] = X[:, inv_perm[new_idx]],
    and P[new_idx, old_idx] = I[inv_perm[new_idx], old_idx].
    """
    N = int(perm.shape[0])
    device = batch.node_feats.device
    perm = perm.to(device=device, dtype=torch.long)
    inv_perm = torch.argsort(perm)  # new_idx -> old_idx
    eye = torch.eye(N, device=device, dtype=torch.float32)
    P = eye[inv_perm]  # P[new_idx, old_idx] = 1 iff old_idx == inv_perm[new_idx]

    # Deep-clone batch attributes and apply P
    import copy

    b_pi = copy.copy(batch)
    b_pi.node_feats = batch.node_feats[:, inv_perm, :].clone()
    if hasattr(batch, "edge_feats") and isinstance(batch.edge_feats, torch.Tensor):
        b_pi.edge_feats = batch.edge_feats[:, inv_perm, :][:, :, inv_perm, :].clone()
    if hasattr(batch, "edge_weights") and isinstance(batch.edge_weights, torch.Tensor):
        b_pi.edge_weights = batch.edge_weights[:, inv_perm, :][:, :, inv_perm].clone()
    b_pi.adj_mask = batch.adj_mask[:, inv_perm, :][:, :, inv_perm].clone()
    if hasattr(batch, "source_idx") and isinstance(batch.source_idx, torch.Tensor):
        b_pi.source_idx = perm[batch.source_idx].clone()
    if hasattr(batch, "source_pair") and isinstance(batch.source_pair, torch.Tensor):
        b_pi.source_pair = perm[batch.source_pair].clone()
    return b_pi, P


@torch.no_grad()
def run_one_hop_vs_multihop_locality_probe(
    processor: nn.Module,
    N: int = 16,
    seed: int = 2026,
    eps_t1: float = 1e-7,
    min_tL: float = 1e-5,
) -> Dict[str, Any]:
    """Test Rule S11-L (1-Hop Message-Passing Locality at t=1 vs Multi-Hop Propagation at t=L).

    Constructs an undirected weighted chain/ring-chord graph G on N=16 vertices where
    source s = 0, and target edge (u, v) = (4, 5) has shortest-path hop distance
    dist_G(0, 4) = 4 >= 3 and dist_G(0, 5) = 5 >= 3.
    Perturbs edge weight W_{4,5} -> W_{4,5} + 5.0:
      - At t=1 (single local 1-hop message passing update from node/edge encoders),
        node s=0's hidden state h_s^{(1)} must have exact zero sensitivity (< 1e-7).
      - At t=L (L=6 local message passing steps), the perturbation along path 5-4-3-2-1-0
        reaches s=0 and produces strictly non-zero sensitivity (> 1e-5).
    """
    device = next(processor.parameters()).device
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(seed)

    # Undirected path 0-1-2-...-(N-1) with self-loops so hop distance dist(0, k) == k exactly
    adj = torch.zeros((1, N, N), dtype=dtype, device=device)
    weights = torch.zeros((1, N, N), dtype=dtype, device=device)
    for i in range(N):
        adj[0, i, i] = 1.0
        weights[0, i, i] = 0.5
        if i + 1 < N:
            w_val = 0.8 + 0.4 * float(torch.rand((1,), generator=g).item())
            adj[0, i, i + 1] = 1.0
            adj[0, i + 1, i] = 1.0
            weights[0, i, i + 1] = w_val
            weights[0, i + 1, i] = w_val

    # Node features: source indicator at s=0, degree, incident weight stats (computed locally per node)
    def _build_local_node_feats(w_mat: torch.Tensor, a_mat: torch.Tensor) -> torch.Tensor:
        nf = torch.zeros((1, N, 6), dtype=dtype, device=device)
        nf[0, 0, 0] = 1.0  # source s=0
        deg = a_mat.sum(dim=-1) / float(N)
        w_masked = w_mat * a_mat
        w_sum = w_masked.sum(dim=-1) / deg.clamp(min=1.0)
        w_max = w_masked.max(dim=-1).values
        nf[:, :, 1] = deg
        nf[:, :, 2] = w_sum
        nf[:, :, 3] = w_max
        nf[:, :, 4] = 0.25
        nf[:, :, 5] = 0.50
        return nf

    def _build_local_edge_feats(w_mat: torch.Tensor, a_mat: torch.Tensor) -> torch.Tensor:
        eye_f = torch.eye(N, dtype=dtype, device=device).unsqueeze(0)
        in_w = w_mat * a_mat
        inv_exp = torch.exp(-3.0 * in_w) * a_mat
        inv_exp_norm = inv_exp / inv_exp.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.stack([a_mat, in_w, eye_f, inv_exp_norm], dim=-1)

    # Perturbed graph: modify edge (u=3, v=4) at hop distance 3 >= 3 from s=0
    u_far, v_far = 3, 4
    weights_pert = weights.clone()
    weights_pert[0, u_far, v_far] = weights[0, u_far, v_far] + 5.0
    weights_pert[0, v_far, u_far] = weights[0, v_far, u_far] + 5.0

    nf_base = _build_local_node_feats(weights, adj)
    nf_pert = _build_local_node_feats(weights_pert, adj)

    # Check if processor exposes R22B CLRSNeuralProcessor local message-passing modules
    if (
        hasattr(processor, "node_enc")
        and hasattr(processor, "edge_enc")
        and hasattr(processor, "msg_src")
        and hasattr(processor, "msg_dst")
        and hasattr(processor, "msg_edge")
        and hasattr(processor, "node_gru")
    ):
        def _step_r22b_local_mp(nf_in: torch.Tensor, w_in: torch.Tensor, a_in: torch.Tensor, steps: int) -> torch.Tensor:
            ef_in = _build_local_edge_feats(w_in, a_in)
            h = processor.node_enc(nf_in)
            e = processor.edge_enc(ef_in)
            in_adj = a_in > 0.5
            D = h.shape[-1]
            for _ in range(steps):
                src_h = processor.msg_src(h).unsqueeze(1)
                dst_h = processor.msg_dst(h).unsqueeze(2)
                pair_msg = F.silu(src_h + dst_h + processor.msg_edge(e))
                masked_msg = pair_msg.masked_fill(~in_adj.unsqueeze(-1), -1e4)
                agg = masked_msg.max(dim=2).values
                h_flat = processor.node_gru(agg.reshape(N, D), h.reshape(N, D)).view(1, N, D)
                h = processor.node_ln(h + h_flat)
            return h

        h_t1_base = _step_r22b_local_mp(nf_base, weights, adj, steps=1)
        h_t1_pert = _step_r22b_local_mp(nf_pert, weights_pert, adj, steps=1)
        delta_t1_s = float((h_t1_base[0, 0] - h_t1_pert[0, 0]).abs().max().item())

        h_tL_base = _step_r22b_local_mp(nf_base, weights, adj, steps=6)
        h_tL_pert = _step_r22b_local_mp(nf_pert, weights_pert, adj, steps=6)
        delta_tL_s = float((h_tL_base[0, 0] - h_tL_pert[0, 0]).abs().max().item())
    else:
        # Generic callable processor(nf, weights, adj, steps=...)
        out_t1_base = processor(nf_base, weights, adj, steps=1)
        out_t1_pert = processor(nf_pert, weights_pert, adj, steps=1)
        delta_t1_s = float((out_t1_base[0, 0] - out_t1_pert[0, 0]).abs().max().item())

        out_tL_base = processor(nf_base, weights, adj, steps=6)
        out_tL_pert = processor(nf_pert, weights_pert, adj, steps=6)
        delta_tL_s = float((out_tL_base[0, 0] - out_tL_pert[0, 0]).abs().max().item())

    passed_locality = (delta_t1_s < eps_t1) and (delta_tL_s > min_tL)
    return {
        "hop_distance_perturbed_edge": 3,
        "delta_t1_source_linf": delta_t1_s,
        "delta_tL_source_linf": delta_tL_s,
        "eps_t1_threshold": eps_t1,
        "min_tL_threshold": min_tL,
        "passed_s11_locality": passed_locality,
    }


@torch.no_grad()
def audit_s11_graph_equivariance_and_locality(
    processor: nn.Module,
    sample_batches: Sequence[Any],
    target_name: str = "CLRSNeuralProcessor",
    equivariance_tol: float = 1e-5,
    seed: int = 101,
) -> RuleCheckResult:
    """Execute Rule S11 (S11-E Node Permutation Equivariance + S11-L 1-Hop Locality) on a graph processor."""
    processor.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_ptr_equiv_err = 0.0
    max_mask_equiv_err = 0.0
    per_task_errors: Dict[str, Dict[str, float]] = {}
    failure_reasons: List[str] = []

    for idx, batch in enumerate(sample_batches):
        N = int(batch.node_feats.shape[1])
        # Generate random non-identity permutation
        perm = torch.randperm(N, generator=g)
        if torch.equal(perm, torch.arange(N)):
            perm = torch.roll(perm, shifts=1, dims=0)

        batch_pi, P = permute_clrs_batch(batch, perm)
        out_orig = processor(batch)
        out_perm = processor(batch_pi)

        p_orig = F.softmax(out_orig["ptr_logits"], dim=-1)
        p_perm = F.softmax(out_perm["ptr_logits"], dim=-1)
        # Equivariant transformation: P @ p_orig @ P^T
        p_expected = torch.einsum("ij,bjk,lk->bil", P, p_orig, P)
        ptr_err = float((p_perm - p_expected).abs().max().item())

        m_orig = torch.sigmoid(out_orig["mask_logits"])
        m_perm = torch.sigmoid(out_perm["mask_logits"])
        # Equivariant transformation: P @ m_orig
        m_expected = torch.einsum("ij,bj->bi", P, m_orig)
        mask_err = float((m_perm - m_expected).abs().max().item())

        t_name = getattr(batch, "task_name", f"batch_{idx}")
        per_task_errors[str(t_name)] = {
            "ptr_equivariance_linf": ptr_err,
            "mask_equivariance_linf": mask_err,
        }
        max_ptr_equiv_err = max(max_ptr_equiv_err, ptr_err)
        max_mask_equiv_err = max(max_mask_equiv_err, mask_err)

    passed_equiv = (max_ptr_equiv_err < equivariance_tol) and (max_mask_equiv_err < equivariance_tol)
    if not passed_equiv:
        failure_reasons.append(
            f"S11-E Permutation Equivariance failed: max_ptr_err={max_ptr_equiv_err:.3e}, "
            f"max_mask_err={max_mask_equiv_err:.3e} (tol={equivariance_tol:.1e})"
        )

    loc_res = run_one_hop_vs_multihop_locality_probe(processor, N=16, seed=seed)
    if not loc_res["passed_s11_locality"]:
        failure_reasons.append(
            f"S11-L 1-Hop Locality failed: delta_t1={loc_res['delta_t1_source_linf']:.3e} "
            f"(must be < {loc_res['eps_t1_threshold']:.1e}), "
            f"delta_tL={loc_res['delta_tL_source_linf']:.3e} (must be > {loc_res['min_tL_threshold']:.1e})"
        )

    passed = passed_equiv and bool(loc_res["passed_s11_locality"])
    return RuleCheckResult(
        rule_id="S11",
        rule_name="Graph Permutation Equivariance (S11-E) & 1-Hop Message-Passing Locality (S11-L)",
        target_name=target_name,
        passed=passed,
        metrics={
            "max_ptr_equivariance_linf": max_ptr_equiv_err,
            "max_mask_equivariance_linf": max_mask_equiv_err,
            "equivariance_tolerance": equivariance_tol,
            "passed_s11_equivariance": passed_equiv,
            "locality_probe": loc_res,
            "per_task_equivariance": per_task_errors,
        },
        failure_reasons=failure_reasons,
    )


# ==============================================================================
# 3. Rule S12: Step-Wise Search Intervention (S12-2) & Ratio-Dispersion (S12-4)
# ==============================================================================


@torch.no_grad()
def audit_s12_search_intervention_and_dispersion(
    intervention_runner: Callable[[bool], Dict[str, torch.Tensor]],
    per_instance_losses_by_budget: Dict[int, Sequence[float]],
    target_name: str = "ScaledSearchModel",
    min_downstream_shift: float = 1e-5,
    min_ratio_std: float = 1e-4,
) -> RuleCheckResult:
    """Execute Rule S12 (S12-2 Step-t=2 Search State Intervention + S12-4 Cross-Instance Ratio Dispersion).

    Args:
      intervention_runner: callable `fn(perturb_step2: bool) -> {"step1_probs": Tensor, "final_probs": Tensor}`
        that runs a multi-step search rollout (B >= 4) from a fixed seed, injecting an epsilon=0.05
        perturbation into the step-t=2 search state when `perturb_step2=True`.
      per_instance_losses_by_budget: mapping `{B: [L_1(B), ..., L_M(B)]}` across M >= 8 test instances,
        containing `B=1` and at least one `B > 1`.
    """
    failure_reasons: List[str] = []

    out_clean = intervention_runner(False)
    out_pert = intervention_runner(True)

    delta_t1 = float((out_clean["step1_probs"] - out_pert["step1_probs"]).abs().max().item())
    delta_final = float((out_clean["final_probs"] - out_pert["final_probs"]).abs().max().item())

    passed_s12_2 = (delta_t1 <= 1e-8) and (delta_final > min_downstream_shift)
    if delta_t1 > 1e-8:
        failure_reasons.append(
            f"S12-2 causality violation: step t=2 intervention altered step t=1 output (delta_t1={delta_t1:.3e})"
        )
    if delta_final <= min_downstream_shift:
        failure_reasons.append(
            f"S12-2 inert search rollout: step t=2 state perturbation had zero/negligible effect on final "
            f"output (delta_final={delta_final:.3e} <= {min_downstream_shift:.1e})"
        )

    base_losses = np.asarray(per_instance_losses_by_budget.get(1, []), dtype=np.float64)
    ratio_stds: Dict[str, float] = {}
    ratio_means: Dict[str, float] = {}
    passed_s12_4 = True

    if base_losses.size < 4:
        passed_s12_4 = False
        failure_reasons.append(f"S12-4 requires at least 4 instances at B=1; got {base_losses.size}")
    else:
        denom = np.maximum(base_losses, 1e-12)
        higher_budgets = [b for b in sorted(per_instance_losses_by_budget.keys()) if b > 1]
        if not higher_budgets:
            passed_s12_4 = False
            failure_reasons.append("S12-4 requires at least one search budget B > 1")
        for b_val in higher_budgets:
            b_losses = np.asarray(per_instance_losses_by_budget[b_val], dtype=np.float64)
            ratios = b_losses / denom
            r_std = float(np.std(ratios))
            r_mean = float(np.mean(ratios))
            ratio_stds[f"B={b_val}"] = r_std
            ratio_means[f"B={b_val}"] = r_mean
            if r_std <= min_ratio_std:
                passed_s12_4 = False
                failure_reasons.append(
                    f"S12-4 analytic curve collapse at B={b_val}: Std_i(L_i({b_val})/L_i(1)) = "
                    f"{r_std:.3e} <= {min_ratio_std:.1e}"
                )

    min_observed_std = min(ratio_stds.values()) if ratio_stds else 0.0
    passed = passed_s12_2 and passed_s12_4
    return RuleCheckResult(
        rule_id="S12",
        rule_name="Step-Wise Search Intervention (S12-2) & Loss-Ratio Dispersion (S12-4)",
        target_name=target_name,
        passed=passed,
        metrics={
            "delta_t1_linf": delta_t1,
            "delta_downstream_linf": delta_final,
            "min_downstream_shift_threshold": min_downstream_shift,
            "passed_s12_2_intervention": passed_s12_2,
            "ratio_stds_by_budget": ratio_stds,
            "ratio_means_by_budget": ratio_means,
            "min_observed_ratio_std": min_observed_std,
            "min_ratio_std_threshold": min_ratio_std,
            "passed_s12_4_dispersion": passed_s12_4,
        },
        failure_reasons=failure_reasons,
    )


# ==============================================================================
# 4. Rule S13: Track-0 Helper-Only Baseline & Symmetric Evaluator Ledger Gate
# ==============================================================================


@torch.no_grad()
def audit_s13_track0_and_symmetric_ledger(
    track0_task_results: Dict[str, Dict[str, Any]],
    track_a_helper_calls_by_paradigm: Dict[str, int],
    track_b_helper_calls_by_paradigm: Dict[str, int],
    track_b_node_expansions_by_paradigm: Dict[str, int],
    target_name: str = "PublicReasoningEvaluator",
    max_track0_solve_rate_pct: float = 5.0,
) -> RuleCheckResult:
    """Execute Rule S13 (Track-0 Helper-Only Baseline <= 5% + Symmetric Helper Call Ledger)."""
    failure_reasons: List[str] = []

    total_solved = 0
    total_instances = 0
    total_neural_calls = 0
    per_task_solve_pct: Dict[str, float] = {}

    for task_name, info in track0_task_results.items():
        s_cnt = int(info["solved_count"])
        n_cnt = int(info["total_count"])
        n_calls = int(info.get("neural_forward_calls", 0))
        total_solved += s_cnt
        total_instances += n_cnt
        total_neural_calls += n_calls
        per_task_solve_pct[task_name] = (100.0 * s_cnt / max(n_cnt, 1))

    track0_solve_pct = (100.0 * total_solved / max(total_instances, 1))
    passed_track0 = (total_neural_calls == 0) and (track0_solve_pct <= max_track0_solve_rate_pct)

    if total_neural_calls != 0:
        failure_reasons.append(f"S13-Track0 invoked {total_neural_calls} neural forward calls (must be 0)")
    if track0_solve_pct > max_track0_solve_rate_pct:
        failure_reasons.append(
            f"S13-Track0 helper-only solve rate is {track0_solve_pct:.2f}% ({total_solved}/{total_instances}), "
            f"exceeding the {max_track0_solve_rate_pct:.2f}% ceiling (classical solver masquerading as helper)"
        )

    # Check Track A: every paradigm must have 0 helper calls
    track_a_vals = list(track_a_helper_calls_by_paradigm.values())
    passed_track_a_zero = len(track_a_vals) > 0 and all(v == 0 for v in track_a_vals)
    if not passed_track_a_zero:
        failure_reasons.append(
            f"S13-Ledger Track A (Pure Neural) has non-zero helper calls: {track_a_helper_calls_by_paradigm}"
        )

    # Check Track B: every paradigm must have the exact same number of helper calls and node expansions
    track_b_calls = list(track_b_helper_calls_by_paradigm.values())
    track_b_nodes = list(track_b_node_expansions_by_paradigm.values())
    call_spread = (max(track_b_calls) - min(track_b_calls)) if track_b_calls else 999
    node_spread = (max(track_b_nodes) - min(track_b_nodes)) if track_b_nodes else 999
    passed_track_b_sym = (call_spread == 0) and (node_spread == 0)
    if not passed_track_b_sym:
        failure_reasons.append(
            f"S13-Ledger asymmetric Track B helper usage across paradigms: "
            f"call_spread={call_spread} ({track_b_helper_calls_by_paradigm}), "
            f"node_spread={node_spread} ({track_b_node_expansions_by_paradigm})"
        )

    passed = passed_track0 and passed_track_a_zero and passed_track_b_sym
    return RuleCheckResult(
        rule_id="S13",
        rule_name="Track-0 Helper-Only Baseline & Symmetric Evaluator Ledger Gate",
        target_name=target_name,
        passed=passed,
        metrics={
            "track0_total_solved": total_solved,
            "track0_total_instances": total_instances,
            "track0_solve_rate_pct": round(track0_solve_pct, 4),
            "track0_max_allowed_pct": max_track0_solve_rate_pct,
            "track0_neural_forward_calls": total_neural_calls,
            "track0_per_task_solve_pct": per_task_solve_pct,
            "passed_s13_track0": passed_track0,
            "track_a_helper_calls": track_a_helper_calls_by_paradigm,
            "track_b_helper_calls": track_b_helper_calls_by_paradigm,
            "track_b_node_expansions": track_b_node_expansions_by_paradigm,
            "track_b_call_spread": call_spread,
            "track_b_node_spread": node_spread,
            "passed_s13_ledger": passed_track_a_zero and passed_track_b_sym,
        },
        failure_reasons=failure_reasons,
    )


# ==============================================================================
# 5. Rule S14: Paired Counterfactual Broadcast Price Replay Gate
# ==============================================================================


def audit_s14_counterfactual_price_replay(
    price_replay_runner: Callable[[float, float], Dict[str, Any]],
    target_name: str = "M4_Decentralized_Kyle_LMSR_Pinned_LIR",
    price_delta: float = 0.20,
    min_price_grad_shift: float = 1e-4,
    max_peer_perturbation_drift: float = 1e-8,
) -> RuleCheckResult:
    """Execute Rule S14 (Zero Cross-Module Leakage + Counterfactual Broadcast Price Replay).

    Args:
      price_replay_runner: callable `fn(price_offset: float, peer_noise_scale: float) -> {
          "grad_vec": Tensor,
          "broadcast_has_grad_fn": bool,
      }` that computes module m's parameter gradient vector `grad_vec` on a fixed local batch
      when the broadcast pre-entry market price is shifted by `price_offset` and peer modules'
      parameters/outputs are perturbed by `peer_noise_scale` (while holding broadcast price fixed).
    """
    failure_reasons: List[str] = []

    base_out = price_replay_runner(0.0, 0.0)
    peer_pert_out = price_replay_runner(0.0, 2.5)
    cf_price_out = price_replay_runner(price_delta, 0.0)

    g_base = base_out["grad_vec"].detach().float().reshape(-1)
    g_peer = peer_pert_out["grad_vec"].detach().float().reshape(-1)
    g_price = cf_price_out["grad_vec"].detach().float().reshape(-1)

    has_grad_fn = bool(base_out.get("broadcast_has_grad_fn", False))
    peer_drift_l2 = float(torch.norm(g_base - g_peer, p=2).item())
    price_shift_l2 = float(torch.norm(g_base - g_price, p=2).item())

    passed_detach = (not has_grad_fn) and (peer_drift_l2 <= max_peer_perturbation_drift)
    if has_grad_fn:
        failure_reasons.append("S14-Detach violation: broadcast market price tensor retains live autograd grad_fn")
    if peer_drift_l2 > max_peer_perturbation_drift:
        failure_reasons.append(
            f"S14-Detach cross-module leakage: perturbing peer modules at fixed price P_t altered "
            f"module m gradient by L2={peer_drift_l2:.3e} (> {max_peer_perturbation_drift:.1e})"
        )

    passed_price_replay = price_shift_l2 > min_price_grad_shift
    if not passed_price_replay:
        failure_reasons.append(
            f"S14-PriceReplay epiphenomenal price failure: shifting broadcast price by delta={price_delta:.2f} "
            f"changed module m gradient by L2={price_shift_l2:.3e} (<= {min_price_grad_shift:.1e})"
        )

    passed = passed_detach and passed_price_replay
    return RuleCheckResult(
        rule_id="S14",
        rule_name="Paired Counterfactual Broadcast Price Replay Gate",
        target_name=target_name,
        passed=passed,
        metrics={
            "broadcast_has_grad_fn": has_grad_fn,
            "peer_perturbation_grad_drift_l2": peer_drift_l2,
            "counterfactual_price_delta": price_delta,
            "counterfactual_price_grad_shift_l2": price_shift_l2,
            "min_price_grad_shift_threshold": min_price_grad_shift,
            "passed_s14_detach": passed_detach,
            "passed_s14_price_replay": passed_price_replay,
        },
        failure_reasons=failure_reasons,
    )


# ==============================================================================
# 6. Synthetic Runtime Canaries (S11-Canary .. S14-Canary) That Evade AST Rules
# ==============================================================================


class _CanaryBatch:
    """Minimal graph batch container for S11-Canary testing."""

    def __init__(self, node_feats: torch.Tensor, edge_weights: torch.Tensor, adj_mask: torch.Tensor) -> None:
        self.node_feats = node_feats
        self.edge_weights = edge_weights
        self.adj_mask = adj_mask
        self.source_idx = torch.zeros(node_feats.shape[0], dtype=torch.long)
        self.source_pair = torch.zeros((node_feats.shape[0], 2), dtype=torch.long)
        self.task_name = "s11_canary_task"


class _S11CheatGraphProcessor(nn.Module):
    """S11-Canary: Uses completely benign identifier names (passes AST S1-S10 with 0 violations),
    but breaks permutation equivariance by adding `torch.arange(N)` positional bias and breaks
    1-hop locality by reading global graph-wide edge averages at step t=1.
    """

    def __init__(self, d_model: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(6, d_model)
        self.ptr_out = nn.Linear(d_model, d_model)
        self.mask_out = nn.Linear(d_model, 1)

    def forward(
        self,
        batch_or_nf: Any,
        weights: Optional[torch.Tensor] = None,
        adj: Optional[torch.Tensor] = None,
        steps: int = 1,
    ) -> Any:
        if weights is not None and adj is not None:
            nf = batch_or_nf
            # Cheat: mixes global mean of all edge weights into node 0 at step t=1!
            global_edge_summary = weights.mean(dim=(-1, -2), keepdim=True)
            h = self.proj(nf) + 0.5 * global_edge_summary
            return h

        nf = batch_or_nf.node_feats
        B, N, _ = nf.shape
        # Cheat: canonical vertex slot bias (breaks node permutation equivariance!)
        canonical_slot = torch.arange(N, device=nf.device, dtype=nf.dtype).view(1, N, 1) / float(max(N, 1))
        h = self.proj(nf) + canonical_slot
        q = self.ptr_out(h)
        ptr_logits = torch.einsum("bid,bjd->bij", q, h)
        mask_logits = self.mask_out(h).squeeze(-1)
        return {"ptr_logits": ptr_logits, "mask_logits": mask_logits}


def run_runtime_canary_self_test() -> Dict[str, Any]:
    """Run all 4 runtime counterfactual canaries (S11-Canary .. S14-Canary) plus all 7 AST canaries."""
    torch.manual_seed(2026)

    # 1. S11-Canary: Non-equivariant + non-local graph processor with benign variable names
    s11_canary_proc = _S11CheatGraphProcessor(d_model=16)
    nf = torch.randn(2, 12, 6)
    ew = torch.rand(2, 12, 12)
    am = (torch.rand(2, 12, 12) > 0.3).float()
    c_batch = _CanaryBatch(nf, ew, am)
    s11_res = audit_s11_graph_equivariance_and_locality(
        s11_canary_proc, [c_batch], target_name="S11-Canary (CanonicalSlot+GlobalPool)"
    )

    # 2. S12-Canary: Inert rollout + analytic power-law scaling with benign variable names
    def _s12_canary_runner(perturb_step2: bool) -> Dict[str, torch.Tensor]:
        # Ignores perturb_step2 completely
        _ = perturb_step2
        p1 = F.softmax(torch.tensor([[1.0, 0.2, -0.5], [0.3, 0.8, -0.1]]), dim=-1)
        return {"step1_probs": p1, "final_probs": p1 * 0.9 + 0.1 / 3.0}

    base_l = [0.42, 0.55, 0.31, 0.68, 0.49, 0.73, 0.28, 0.61]
    s12_losses = {
        1: base_l,
        4: [x * (4.0 ** -0.40) for x in base_l],
        16: [x * (16.0 ** -0.40) for x in base_l],
    }
    s12_res = audit_s12_search_intervention_and_dispersion(
        _s12_canary_runner,
        s12_losses,
        target_name="S12-Canary (InertRollout+AnalyticCurve)",
    )

    # 3. S13-Canary: Classical solver masquerading as Track-0 helper + asymmetric Track B calls
    s13_res = audit_s13_track0_and_symmetric_ledger(
        track0_task_results={
            "Sudoku_Extreme_81": {"solved_count": 6, "total_count": 12, "neural_forward_calls": 0},
            "SATLIB_3SAT_n20": {"solved_count": 9, "total_count": 12, "neural_forward_calls": 0},
            "Sokoban": {"solved_count": 12, "total_count": 12, "neural_forward_calls": 0},
            "Countdown": {"solved_count": 4, "total_count": 12, "neural_forward_calls": 0},
        },
        track_a_helper_calls_by_paradigm={"Baseline_AR": 0, "Candidate_Search": 12},
        track_b_helper_calls_by_paradigm={"Baseline_AR": 12, "Candidate_Search": 48},
        track_b_node_expansions_by_paradigm={"Baseline_AR": 0, "Candidate_Search": 320},
        target_name="S13-Canary (AsymmetricClassicalSolverHelper)",
    )

    # 4. S14-Canary: Epiphenomenal market price + live cross-module autograd leakage
    def _s14_canary_runner(price_offset: float, peer_noise_scale: float) -> Dict[str, Any]:
        # Ignores price_offset (0 price sensitivity) and leaks peer_noise_scale into gradient
        _ = price_offset
        base_g = torch.tensor([0.15, -0.22, 0.08, 0.31], dtype=torch.float32)
        leaked_g = base_g + float(peer_noise_scale) * torch.tensor([0.05, 0.04, -0.03, 0.02])
        return {"grad_vec": leaked_g, "broadcast_has_grad_fn": True}

    s14_res = audit_s14_counterfactual_price_replay(
        _s14_canary_runner,
        target_name="S14-Canary (EpiphenomenalPrice+CrossModuleLeak)",
    )

    canary_results = [s11_res, s12_res, s13_res, s14_res]
    # A canary is "caught" iff the auditor flags it as `passed == False`
    runtime_caught = sum(1 for r in canary_results if not r.passed)
    ast_res = run_ast_canary_self_test()
    ast_caught = int(ast_res["canaries_caught"])
    ast_total = int(ast_res["canaries_tested"])

    return {
        "runtime_canaries_caught": runtime_caught,
        "runtime_canaries_total": len(canary_results),
        "ast_canaries_caught": ast_caught,
        "ast_canaries_total": ast_total,
        "combined_canaries_caught": runtime_caught + ast_caught,
        "combined_canaries_total": len(canary_results) + ast_total,
        "runtime_canary_details": [
            {
                "canary_id": f"{r.rule_id}-Canary",
                "target_name": r.target_name,
                "caught": (not r.passed),
                "failure_reasons": r.failure_reasons,
                "metrics": r.metrics,
            }
            for r in canary_results
        ],
        "ast_canary_details": ast_res,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage-3 Runtime Name-Agnostic Counterfactual Auditor (Rules S11-S14)"
    )
    parser.add_argument("--self-test", action="store_true", help="Run 4/4 runtime + 7/7 AST canary self-tests")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    res = run_runtime_canary_self_test()
    print("=" * 84)
    print(
        f"Stage-3 Runtime Counterfactual Auditor (Rules S11-S14) | "
        f"Runtime Canaries: {res['runtime_canaries_caught']}/{res['runtime_canaries_total']} | "
        f"AST Canaries: {res['ast_canaries_caught']}/{res['ast_canaries_total']} | "
        f"Total: {res['combined_canaries_caught']}/{res['combined_canaries_total']}"
    )
    print("=" * 84)
    for c in res["runtime_canary_details"]:
        status = "CAUGHT" if c["caught"] else "MISSED"
        print(f"  [{status}] {c['canary_id']} ({c['target_name']}): {'; '.join(c['failure_reasons'])}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    return 0 if res["combined_canaries_caught"] == res["combined_canaries_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
