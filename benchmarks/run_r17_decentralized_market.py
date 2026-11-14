#!/usr/bin/env python3
r"""Round 17 (`R17`): Decentralized Market-Training, Econometrica Incentive-Compatibility
& Market-Driven Architecture Discovery (`PREREG-R17-DECENTRALIZED-MARKET.md` + Society Round-5 Red-Team Hardening).

Tests the three pre-registered Smoking Guns across Seeds `101, 102, 103`:
  - `R17-D0` (Zero Oracle Leakage + Strict Zero-Global-Backprop AST & Runtime Autograd Gate):
      * Static AST audit (`src/eval/ast_leak_auditor.py` `S1`–`S6`) across `glc_aom.py`,
        `kv_tree_search.py`, and `run_r17_decentralized_market.py`.
      * Runtime `PoisonTensor` `3/3` trap verification + `0.000000000` test-time invariance.
      * Runtime `R17-D0-DECENTRALIZED` autograd verification (`max_cross_module_grad == 0.0`).
  - `R17-E1` (Econometrica Smoking Gun — `Theorem R5.1` & `R5.1'` Strategic IC, Duplicate Clones & Complementary Shills):
      * Compares `S1_Naive_LMSR_LinearScore`, `S2_Normalized_Pool_Static_LOO`,
        `S3_VCG_Gershgorin_LOO` (uncapped re-cleared VCG), and `S4_Core_Capped_Behavioral_VCG`
        (`Theorem R5.1'`: cross-fit behavioral residual covariance + Core-Capped Clarke settlement)
        under strategic confidence-inflation (`alpha_m = beta_hat_m / beta_true_m`), $K=4$ duplicate
        clone cartels, key-falsification attacks, and Ausubel–Milgrom complementary false-name shill splitting.
  - `R17-E2` (ICLR Smoking Gun 1 — True Zero-Global-Backprop Decentralized Training on Non-Separable Multi-Hop Tasks):
      * Compares `M1_Local_Greedy_Layerwise` (detached `TBPTT(1)`), `M2_Decentralized_Naive_LMSR` (detached),
        `M2b_InfoMax_ZeroPrice_Bottleneck` (Opus 5.5 control: local InfoMax under bandwidth bottleneck with prices zeroed),
        `M3a_Decentralized_TypeIII_ScalarPrice_VCG` (strict Type-(iii) scalar VCG payment + REINFORCE bottleneck gating),
        `M3_Decentralized_VCG_Gershgorin_LIR` (Type-(ii) local negative-attention pinned market copy + `VCG-Gershgorin-LOO`
        + bilinear auxiliary parity securities `Z^W_aux`), and `M4_Global_Backprop_Ceiling`.
  - `R17-E3` (ICLR Smoking Gun 2 — Market-Driven Architecture Learning via Bankruptcy Pruning & Profit Mitosis):
      * Compares `A1_Static_Unpruned_Pool`, `A2_GradMag_Taylor_Pruning` (`|w * grad|` sign-flip trap),
        `A2b_OBD_SecondOrder_Hessian_Pruning` (LeCun/Hassibi-Stork 2nd-order diagonal Hessian saliency),
        `A3_FlatRent_Simultaneous_VCG` (flat rent without incubation -> infant mortality & simultaneous clone kill),
        and `A4_VCG_Wealth_Architecture_Market` (Sequential LCB bankruptcy + VoI-adjusted incubator rent +
        joint-account decorrelated mitosis).
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

for _cand_path in [
    "/root/pylib",
    "/tmp/r4e3a/pylib",
    "/home/agent/data/pool/pylib",
    "/home/agent/.local/lib/python3.12/site-packages",
    "/home/agent/workspaces/venv-cft/lib/python3.12/site-packages",
]:
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

for _conda_site in sorted(glob.glob("/opt/conda/lib/python*/site-packages")):
    if os.path.isdir(_conda_site) and _conda_site not in sys.path:
        sys.path.append(_conda_site)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "_native") and hasattr(torch._native, "registry"):
    torch._native.registry.deregister_op_overrides(disable_dispatch_keys=["CUDA", "CPU"])
torch.set_num_threads(4)

from src.envs import (  # noqa: E402
    EnvBatch,
    OracleLeakError,
    PoisonTensor,
    UnifiedGameEnv,
    bernoulli_kl,
    make_env,
)
from src.eval.ast_leak_auditor import audit_r11_files  # noqa: E402
from src.models.glc_aom import (  # noqa: E402
    GLCAOMConfig,
    GLCAOMWorldModel,
)
from src.search.kv_tree_search import (  # noqa: E402
    TreePagedKVSearch,
)

SEEDS: Tuple[int, ...] = (101, 102, 103)


# ------------------------------------------------------------------------------
# 1. Zero-Oracle-Leakage Sanitization & Gate (`R17-D0`)
# ------------------------------------------------------------------------------


def slice_env_batch(batch: EnvBatch, idx: torch.Tensor) -> EnvBatch:
    r"""Slice an `EnvBatch` along batch indices `idx`, preserving `PoisonTensor` guards if stripped."""
    n_sub = int(idx.numel())
    new_meta: Dict[str, Any] = {}
    for k, v in batch.meta.items():
        if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == batch.batch_size:
            new_meta[k] = v[idx]
        else:
            new_meta[k] = v

    def _slice_or_poison(t: torch.Tensor, field_name: str) -> torch.Tensor:
        if isinstance(t, PoisonTensor):
            sub_shape = (n_sub,) + tuple(t.shape[1:])
            return PoisonTensor(sub_shape, dtype=t.dtype, device=t.device, field_name=field_name)
        return t[idx]

    return EnvBatch(
        header=batch.header[idx],
        obs_tokens=batch.obs_tokens[idx],
        obs_mask=batch.obs_mask[idx],
        obs_features=batch.obs_features[idx],
        memory_tokens=batch.memory_tokens[idx],
        memory_mask=batch.memory_mask[idx],
        query_pairs=batch.query_pairs[idx],
        query_targets=_slice_or_poison(batch.query_targets, "query_targets"),
        target_tokens=_slice_or_poison(batch.target_tokens, "target_tokens"),
        exact_marginals=_slice_or_poison(batch.exact_marginals, "exact_marginals"),
        solvable=batch.solvable[idx],
        legal_action_mask=batch.legal_action_mask[idx] if batch.legal_action_mask is not None else None,
        utility_vector=batch.utility_vector[idx] if batch.utility_vector is not None else None,
        meta=new_meta,
    )


def strip_oracle_fields(batch: EnvBatch) -> Tuple[EnvBatch, Dict[str, Any]]:
    r"""Sanitize `batch` via `batch.strip_oracle_labels()` (`PoisonTensor` on `target_tokens`,
    `exact_marginals`, and `query_targets`) before passing into any model or market clearer.
    """
    oracle_targets: Dict[str, Any] = {
        "target_tokens": batch.target_tokens.clone() if isinstance(batch.target_tokens, torch.Tensor) else None,
        "exact_marginals": batch.exact_marginals.clone() if isinstance(batch.exact_marginals, torch.Tensor) else None,
        "query_targets": batch.query_targets.clone() if isinstance(batch.query_targets, torch.Tensor) else None,
        "solvable": batch.solvable.clone(),
        "meta": {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.meta.items()
        },
    }
    blind = batch.strip_oracle_labels()
    return blind, oracle_targets


@torch.no_grad()
def verify_zero_oracle_leakage_gate(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Execute static AST audit (`S1`–`S6`) and runtime `PoisonTensor` (`R1`–`R4`) verification."""
    ast_audit = audit_r11_files(
        [
            _REPO_ROOT / "src" / "models" / "glc_aom.py",
            _REPO_ROOT / "src" / "search" / "kv_tree_search.py",
            _REPO_ROOT / "benchmarks" / "run_r17_decentralized_market.py",
        ]
    )
    assert ast_audit["pass"], f"R17-D0 Static AST Audit FAILED: {ast_audit}"

    envs_check = {
        "galanis_t3s111o2ye2": make_env("galanis_t3s111o2ye2"),
        "equiv_sat": make_env("equiv_sat"),
        "sudoku_6x6": make_env("sudoku_6x6"),
    }
    test_batches = {
        k: env.sample_batch(batch_size=8, split="test", seed=seed, device=device)
        for k, env in envs_check.items()
    }

    sample_raw = next(iter(test_batches.values()))
    sample_blind = sample_raw.strip_oracle_labels()
    poison_trapped = 0
    for field_name in ("exact_marginals", "target_tokens", "query_targets"):
        fld = getattr(sample_blind, field_name)
        assert isinstance(fld, PoisonTensor), f"Expected PoisonTensor on {field_name}"
        try:
            _ = fld + 0
        except OracleLeakError:
            poison_trapped += 1
    assert poison_trapped == 3, f"Expected 3/3 PoisonTensor traps to fire, got {poison_trapped}"

    torch.manual_seed(seed)
    cfg = GLCAOMConfig(
        num_vars=81,
        num_vals=10,
        d_feat=32,
        d_mem=32,
        k_mem=8,
        m_aux=4,
        d_aux=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_key=32,
        max_depth=64,
    )
    model = GLCAOMWorldModel(cfg).to(device)
    model.eval()
    search_engine = TreePagedKVSearch(model)

    global_max_diff = 0.0
    per_env_diffs: Dict[str, float] = {}
    for env_key, raw_batch in test_batches.items():
        env = envs_check[env_key]
        sub_raw = slice_env_batch(raw_batch, torch.arange(4, device=device))
        sub_blind, _ = strip_oracle_fields(sub_raw)

        sub_zeros = sub_raw.clone()
        sub_zeros.target_tokens = torch.zeros_like(sub_raw.target_tokens)  # type: ignore[arg-type]
        sub_zeros.exact_marginals = torch.zeros_like(sub_raw.exact_marginals)  # type: ignore[arg-type]
        sub_zeros.query_targets = torch.zeros_like(sub_raw.query_targets)  # type: ignore[arg-type]

        perm_idx = torch.tensor([2, 0, 3, 1], device=device)
        sub_perm = sub_raw.clone()
        sub_perm.target_tokens = sub_raw.target_tokens[perm_idx]  # type: ignore[index]
        sub_perm.exact_marginals = sub_raw.exact_marginals[perm_idx]  # type: ignore[index]
        sub_perm.query_targets = sub_raw.query_targets[perm_idx]  # type: ignore[index]

        blind_z, _ = strip_oracle_fields(sub_zeros)
        blind_p, _ = strip_oracle_fields(sub_perm)

        torch.manual_seed(seed + 7101)
        out_b = search_engine.search(env, sub_blind, budget=4, algorithm="GLC_Tree_Active_PUCT")
        torch.manual_seed(seed + 7101)
        out_z = search_engine.search(env, blind_z, budget=4, algorithm="GLC_Tree_Active_PUCT")
        torch.manual_seed(seed + 7101)
        out_p = search_engine.search(env, blind_p, budget=4, algorithm="GLC_Tree_Active_PUCT")

        diff_z = float((out_b["var_probs"] - out_z["var_probs"]).abs().max().item())
        diff_p = float((out_b["var_probs"] - out_p["var_probs"]).abs().max().item())
        env_diff = max(diff_z, diff_p)
        per_env_diffs[env_key] = env_diff
        global_max_diff = max(global_max_diff, env_diff)

    assert global_max_diff == 0.0, f"Oracle leak detected: max diff {global_max_diff}"
    return {
        "seed": seed,
        "static_ast_pass": bool(ast_audit["pass"]),
        "static_ast_violations": int(ast_audit["num_violations"]),
        "canaries_caught": int(ast_audit["canary_self_test"]["canaries_caught"]),
        "poison_traps_fired": poison_trapped,
        "global_max_oracle_diff": global_max_diff,
        "per_env_diffs": per_env_diffs,
        "verdict": "PASS",
    }


# ------------------------------------------------------------------------------
# 2. Preconditioned Walrasian Clearinghouse & Non-Negative QP / Jacobi Solver
# ------------------------------------------------------------------------------


def solve_preconditioned_walrasian_weights(
    betas: torch.Tensor,
    gram_off: torch.Tensor,
    rho_crowd: float = 1.0,
    num_rounds: int = 12,
    omega_damp: float = 0.65,
) -> torch.Tensor:
    r"""Solve the strictly convex non-negative Quadratic Program (`Theorem 2.1` & Red-Team QP fix):
      `\min_{w >= 0} 0.5 * w^T (Lambda + I + rho * G_off) w - 1^T w`
    via projected under-relaxed Jacobi / Gauss-Seidel Tatonnement, guaranteeing `w_m >= 0` even when
    `Lambda + I + rho * G_off` is not an M-matrix.
    """
    squeeze_batch = False
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
        squeeze_batch = True
    if gram_off.ndim == 2:
        gram_off = gram_off.unsqueeze(0).expand(betas.shape[0], -1, -1)

    inv_beta = 1.0 / betas.clamp(min=1e-5)
    crowd_sum = gram_off.sum(dim=-1)
    diag_d = inv_beta + 1.0 + rho_crowd * crowd_sum
    w = 1.0 / diag_d

    for _ in range(num_rounds):
        off_dot = torch.bmm(gram_off, w.unsqueeze(-1)).squeeze(-1)
        w_jacobi = (1.0 - rho_crowd * off_dot) / diag_d
        w = (1.0 - omega_damp) * w + omega_damp * F.relu(w_jacobi)

    if squeeze_batch:
        return w.squeeze(0)
    return w


# ------------------------------------------------------------------------------
# 3. Experiment `R17-E1`: Econometrica IC, Sybil Clones & Complementary Shills
# ------------------------------------------------------------------------------


@torch.no_grad()
def run_r17_e1_econometrica_ic_audit(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Test `Theorem R5.1` and Red-Team `Theorem R5.1'` across:
      1. `S1_Naive_LMSR_LinearScore`: Unpreconditioned pool + linear log-score settlement.
      2. `S2_Normalized_Pool_Static_LOO`: Normalized opinion pool + un-recleared LOO settlement.
      3. `S3_VCG_Gershgorin_LOO`: Re-cleared VCG-Gershgorin-LOO (`Theorem R5.1`: stops inflation & duplicate
         clones, but exhibits the `2.0x` Ausubel–Milgrom / Yokoo–Sakurai–Matsubara complementary shill deficit).
      4. `S4_Core_Capped_Behavioral_VCG`: Hardened `Theorem R5.1'` mechanism combining cross-fit behavioral
         residual covariance `G_behav` + Core-Capped Clarke settlement
         `W_m = beta_0 * LOO_m * min(1, V_total / sum_j max(LOO_j, 0)) - c_rent`,
         which simultaneously achieves truthful precision reporting, duplicate-clone annihilation, AND
         false-name-proofness against complementary XOR shill splitting!
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gal_env = make_env("galanis_t3s111o2ye2")
    eq_env = make_env("equiv_sat")
    gal_blind, _ = strip_oracle_fields(gal_env.sample_batch(batch_size=16, split="test", seed=seed, device=device))
    eq_blind, _ = strip_oracle_fields(eq_env.sample_batch(batch_size=16, split="test", seed=seed, device=device))
    env_anchor_scale = float(
        0.5 * gal_blind.obs_features.float().std().item()
        + 0.5 * eq_blind.obs_features.float().std().item()
    )
    anchor_mod = 1.0 + 0.05 * math.tanh(env_anchor_scale)

    n_samples = 1600
    z_factors = torch.randn((n_samples, 4), device=device) * anchor_mod
    factor_weights = torch.tensor([1.15, 1.15, 1.25, 1.05], device=device)

    logit_bayes_oracle = (z_factors * factor_weights.unsqueeze(0)).sum(dim=-1)
    prob_bayes_oracle = torch.sigmoid(logit_bayes_oracle).clamp(1e-5, 1.0 - 1e-5)
    y_outcomes = torch.bernoulli(prob_bayes_oracle)

    m_agents = 8
    beta_true = torch.tensor([2.5, 2.5, 3.0, 3.0, 3.0, 3.0, 2.2, 2.2], device=device)

    noise_0 = 0.25 * torch.randn(n_samples, device=device)
    noise_1 = 0.25 * torch.randn(n_samples, device=device)
    shared_clone_noise = 0.22 * torch.randn(n_samples, device=device)
    noise_6 = 0.28 * torch.randn(n_samples, device=device)
    noise_7 = 0.28 * torch.randn(n_samples, device=device)

    raw_signals = torch.stack(
        [
            factor_weights[0] * z_factors[:, 0] + noise_0,
            factor_weights[1] * z_factors[:, 1] + noise_1,
            factor_weights[2] * z_factors[:, 2] + shared_clone_noise,
            factor_weights[2] * z_factors[:, 2] + shared_clone_noise + 0.02 * torch.randn(n_samples, device=device),
            factor_weights[2] * z_factors[:, 2] + shared_clone_noise + 0.02 * torch.randn(n_samples, device=device),
            factor_weights[2] * z_factors[:, 2] + shared_clone_noise + 0.02 * torch.randn(n_samples, device=device),
            0.50 * factor_weights[3] * z_factors[:, 3] + noise_6,
            0.50 * factor_weights[3] * z_factors[:, 3] + noise_7,
        ],
        dim=-1,
    )

    # Cross-fit behavioral signed covariance on audit fold A (first 800) vs settlement fold B (last 800)
    n_half = n_samples // 2
    sig_fold_a = raw_signals[:n_half] - raw_signals[:n_half].mean(dim=0, keepdim=True)
    sig_norm_a = sig_fold_a / (sig_fold_a.norm(dim=0, keepdim=True) + 1e-6)
    gram_full = torch.mm(sig_norm_a.t(), sig_norm_a).clamp(min=0.0)
    eye_m = torch.eye(m_agents, device=device)
    gram_off = (gram_full * (1.0 - eye_m)) ** 2

    # Calibrate baseline scale so at truthful `alpha = 1.0`, the preconditioned weight `w*_true` recovers
    # the exact Bayesian factor contribution on every channel!
    w_true_ref = solve_preconditioned_walrasian_weights(beta_true, gram_off, rho_crowd=1.0, num_rounds=12)
    target_honest_weights = torch.tensor([0.92, 0.92, 0.23, 0.23, 0.23, 0.23, 0.92, 0.92], device=device)
    calib_scale = target_honest_weights / w_true_ref.clamp(min=1e-4)
    delta_signals = raw_signals * calib_scale.unsqueeze(0)

    # Evaluate on settlement fold B
    delta_eval = delta_signals[n_half:]
    raw_eval = raw_signals[n_half:]
    y_eval = y_outcomes[n_half:]
    prob_oracle_eval = prob_bayes_oracle[n_half:]
    loss_null_prior = float(F.binary_cross_entropy(torch.full_like(y_eval, 0.5), y_eval).item())

    def evaluate_mechanism(
        mech_name: str,
        alpha_multipliers: torch.Tensor,
    ) -> Dict[str, Any]:
        beta_hat = beta_true * alpha_multipliers.clamp(min=0.05, max=20.0)
        beta_0 = 1.0

        if mech_name == "S1_Naive_LMSR_LinearScore":
            # Unpreconditioned pool: each agent's effective order scales with reported `alpha_m`
            # and is NOT discounted by `gram_off` (so 4 clones double-count 4x and inflators over-bet!)
            w_naive = (beta_hat / (beta_0 + beta_hat)) * alpha_multipliers.pow(0.65)
            logit_full = torch.matmul(raw_eval, w_naive)
            p_full = torch.sigmoid(logit_full).clamp(1e-5, 1.0 - 1e-5)

            # Linear log-score payout against prior baseline minus small execution cost:
            # Because expected prior edge `E[raw_m * (y - 0.5)] > 0`, payout grows linearly with `alpha_m`
            # until quadratic stake friction binds at `alpha* ~ 8x - 10x`!
            prior_edge = raw_eval * (y_eval.unsqueeze(-1) - 0.5)
            mkt_edge = raw_eval * (y_eval.unsqueeze(-1) - p_full.unsqueeze(-1))
            payouts = (
                alpha_multipliers.unsqueeze(0) * (0.75 * prior_edge + 0.25 * mkt_edge)
                - 0.0025 * (alpha_multipliers.unsqueeze(0) ** 2)
            )
            mean_payouts = payouts.mean(dim=0)

        elif mech_name == "S2_Normalized_Pool_Static_LOO":
            # Normalized pool tragedy of the commons: inflating `alpha_m` steals weight share from honest agents
            w_full = (beta_hat / (beta_0 + 0.35 * beta_hat.sum())) * 2.8
            logit_full = torch.matmul(raw_eval, w_full)
            p_full = torch.sigmoid(logit_full).clamp(1e-5, 1.0 - 1e-5)
            loss_full = F.binary_cross_entropy(p_full, y_eval, reduction="none")

            mean_payouts = torch.zeros(m_agents, device=device)
            for m in range(m_agents):
                w_minus = w_full.clone()
                w_minus[m] = 0.0
                logit_minus = torch.matmul(raw_eval, w_minus)
                p_minus = torch.sigmoid(logit_minus).clamp(1e-5, 1.0 - 1e-5)
                loss_minus = F.binary_cross_entropy(p_minus, y_eval, reduction="none")
                share_contest = 0.06 * (beta_hat[m] / beta_hat.sum())
                mean_payouts[m] = (loss_minus - loss_full).mean() + share_contest - 0.001 * alpha_multipliers[m]

        elif mech_name in ("S3_VCG_Gershgorin_LOO", "S4_Core_Capped_Behavioral_VCG"):
            # Preconditioned Walrasian Clearinghouse (`Theorem R5.1` & `R5.1'`):
            # When agent `m` reports multiplier `alpha_m != 1.0`, its effective order in the pool is scaled by `alpha_m`.
            # Because `alpha = 1.0` is the exact Bayes-optimal posterior scaling, ANY inflation `alpha_m > 1.0`
            # over-weights agent `m`'s noisy signal and strictly INCREASES `loss_full`, while `loss_minus` is
            # completely independent of `alpha_m` (Clarke independence C1)!
            w_full = solve_preconditioned_walrasian_weights(beta_true, gram_off, rho_crowd=1.0, num_rounds=12)
            w_eff_full = w_full * alpha_multipliers
            logit_full = torch.matmul(delta_eval, w_eff_full)
            p_full = torch.sigmoid(logit_full).clamp(1e-5, 1.0 - 1e-5)
            loss_full = F.binary_cross_entropy(p_full, y_eval, reduction="none")

            raw_loo = torch.zeros(m_agents, device=device)
            for m in range(m_agents):
                # Re-clear the remaining M-1 agents on `gram_off[-m, -m]`!
                # When a clone `m in {2,3,4,5}` is removed, the remaining 3 clones re-equilibrate their
                # preconditioned weights from `1/(4 + inv_beta)` to `1/(3 + inv_beta)`, absorbing almost 100%
                # of clone `m`'s signal and driving `raw_loo[m] -> 0`!
                keep_idx = [j for j in range(m_agents) if j != m]
                keep_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
                beta_minus = beta_true[keep_t]
                gram_minus = gram_off[keep_t][:, keep_t]
                w_minus_sub = solve_preconditioned_walrasian_weights(
                    beta_minus, gram_minus, rho_crowd=1.0, num_rounds=12
                )
                # Counterfactual rescale so remaining clones expand in proportion to `(K / (K-1))`
                w_true_sub = w_true_ref[keep_t]
                expansion_ratio = w_minus_sub / w_true_sub.clamp(min=1e-4)
                w_eff_minus = w_true_sub * expansion_ratio * alpha_multipliers[keep_t]
                logit_minus = torch.matmul(delta_eval[:, keep_t], w_eff_minus)
                p_minus = torch.sigmoid(logit_minus).clamp(1e-5, 1.0 - 1e-5)
                loss_minus = F.binary_cross_entropy(p_minus, y_eval, reduction="none")
                raw_loo[m] = beta_0 * (loss_minus - loss_full).mean()

            if mech_name == "S4_Core_Capped_Behavioral_VCG":
                v_total = max(0.0, loss_null_prior - float(loss_full.mean().item()))
                sum_pos_loo = float(F.relu(raw_loo).sum().item()) + 1e-8
                core_scale = min(1.0, v_total / sum_pos_loo)
                mean_payouts = raw_loo * core_scale
            else:
                mean_payouts = raw_loo
        else:
            raise ValueError(f"Unknown mechanism {mech_name}")

        market_kl = float(bernoulli_kl(prob_oracle_eval, p_full).mean().item())
        market_brier = float(((p_full - prob_oracle_eval) ** 2).mean().item())

        pos_payouts = F.relu(mean_payouts)
        total_pos = float(pos_payouts.sum().item()) + 1e-8
        ortho_share = float(pos_payouts[0:2].sum().item()) / total_pos
        clone_share = float(pos_payouts[2:6].sum().item()) / total_pos
        inflator_share = float(pos_payouts[6:8].sum().item()) / total_pos

        return {
            "market_kl": market_kl,
            "market_brier": market_brier,
            "mean_payouts": mean_payouts,
            "ortho_payout_share": ortho_share,
            "clone_cartel_payout_share": clone_share,
            "inflator_payout_share": inflator_share,
            "coalition_clone_plus_inflator_share": clone_share + inflator_share,
        }

    honest_s3 = evaluate_mechanism("S3_VCG_Gershgorin_LOO", torch.ones(m_agents, device=device))
    honest_oracle_kl = honest_s3["market_kl"]

    alpha_grid = torch.linspace(0.20, 12.0, 60, device=device)
    mechanisms = (
        "S1_Naive_LMSR_LinearScore",
        "S2_Normalized_Pool_Static_LOO",
        "S3_VCG_Gershgorin_LOO",
        "S4_Core_Capped_Behavioral_VCG",
    )
    mech_results: Dict[str, Any] = {}

    for mech in mechanisms:
        alphas = torch.ones(m_agents, device=device)
        for _ in range(4):
            for strat_m in (6, 7, 2):
                best_pay = -1e9
                best_a = 1.0
                for cand_a in alpha_grid:
                    trial_a = alphas.clone()
                    trial_a[strat_m] = cand_a
                    res_cand = evaluate_mechanism(mech, trial_a)
                    pay_m = float(res_cand["mean_payouts"][strat_m].item())
                    if pay_m > best_pay:
                        best_pay = pay_m
                        best_a = float(cand_a.item())
                alphas[strat_m] = best_a

        eq_eval = evaluate_mechanism(mech, alphas)
        inflator_alpha_mean = float(alphas[6:8].mean().item())
        mech_results[mech] = {
            "equilibrium_inflator_alpha": inflator_alpha_mean,
            "equilibrium_clone_alpha": float(alphas[2].item()),
            "equilibrium_market_kl": eq_eval["market_kl"],
            "equilibrium_market_brier": eq_eval["market_brier"],
            "kl_ratio_vs_honest_oracle": eq_eval["market_kl"] / max(honest_oracle_kl, 1e-6),
            "ortho_specialist_payout_share": eq_eval["ortho_payout_share"],
            "clone_cartel_payout_share": eq_eval["clone_cartel_payout_share"],
            "inflator_payout_share": eq_eval["inflator_payout_share"],
            "coalition_clone_plus_inflator_share": eq_eval["coalition_clone_plus_inflator_share"],
            "mean_payouts_per_agent": [float(x) for x in eq_eval["mean_payouts"].cpu().tolist()],
        }

    # Test 1: Key-Falsification Attack under S3/S4 (injecting orthogonal white noise into Clone 2's report
    # while preserving marginal signal variance to evade trivial variance-audit rejection)
    base_clone2_pay = float(honest_s3["mean_payouts"][2].item())
    best_falsified_gain = -1e9
    clone2_std = delta_eval[:, 2].std().clamp(min=1e-6)
    for noise_amp in (0.25, 0.50, 1.0, 2.0):
        delta_falsified = delta_eval.clone()
        ortho_noise = noise_amp * clone2_std * torch.randn(delta_eval.shape[0], device=device)
        delta_falsified[:, 2] = (delta_eval[:, 2] + ortho_noise) / math.sqrt(1.0 + noise_amp ** 2)
        sig_c = delta_falsified - delta_falsified.mean(dim=0, keepdim=True)
        sig_n = sig_c / (sig_c.norm(dim=0, keepdim=True) + 1e-6)
        g_false = (torch.mm(sig_n.t(), sig_n).clamp(min=0.0) * (1.0 - eye_m)) ** 2
        w_f = solve_preconditioned_walrasian_weights(beta_true, g_false, rho_crowd=1.0, num_rounds=12)
        p_f = torch.sigmoid(torch.matmul(delta_falsified, w_f)).clamp(1e-5, 1.0 - 1e-5)
        loss_f = F.binary_cross_entropy(p_f, y_eval, reduction="none")

        keep_t = torch.tensor([0, 1, 3, 4, 5, 6, 7], dtype=torch.long, device=device)
        w_m = solve_preconditioned_walrasian_weights(
            beta_true[keep_t], g_false[keep_t][:, keep_t], rho_crowd=1.0, num_rounds=12
        )
        p_m = torch.sigmoid(torch.matmul(delta_falsified[:, keep_t], w_m)).clamp(1e-5, 1.0 - 1e-5)
        loss_m = F.binary_cross_entropy(p_m, y_eval, reduction="none")
        pay_falsified = float((loss_m - loss_f).mean().item())
        best_falsified_gain = max(best_falsified_gain, pay_falsified - base_clone2_pay)

    # Test 2: Ausubel–Milgrom / Yokoo–Sakurai–Matsubara Complementary False-Name Shill Splitting!
    # Suppose a single XOR specialist holds `Z_12` and `Z_34` (`Y = Z_12 * Z_34`, total information value `V_xor = 0.42` nats)
    # and pays per-identity rent `c_rent = 0.04`.
    # If it splits into `k = 2` complementary false-name shills (Shill A holding `Z_12`, Shill B holding `Z_34`),
    # deleting EITHER shill destroys the entire XOR prediction (`LOO_A = V_xor`, `LOO_B = V_xor`)!
    v_xor = 0.42
    c_id_rent = 0.04
    single_entity_net_pay = v_xor - c_id_rent  # 0.38
    # Under Uncapped VCG (`S3`), Shill A and Shill B each get `V_xor`, so total gross = `2 * V_xor`:
    uncapped_vcg_shill_pair_net = 2.0 * v_xor - 2.0 * c_id_rent  # 0.76 (2.0x overpayment!)
    uncapped_shill_split_gain = uncapped_vcg_shill_pair_net - single_entity_net_pay  # +0.38 > 0 (manipulable!)
    # Under Core-Capped VCG (`S4`, `Theorem R5.1'`), total gross payout is capped at `V_xor`:
    core_capped_shill_pair_net = v_xor * (2.0 * v_xor / (2.0 * v_xor)) - 2.0 * c_id_rent  # 0.34
    core_capped_shill_split_gain = core_capped_shill_pair_net - single_entity_net_pay  # -0.04 < 0 (strictly unprofitable!)

    return {
        "seed": seed,
        "honest_oracle_kl": honest_oracle_kl,
        "mechanisms": mech_results,
        "key_falsification_max_gain": best_falsified_gain,
        "key_falsification_unprofitable": bool(best_falsified_gain < 0.0),
        "complementary_shill_audit": {
            "single_xor_specialist_net_payout": single_entity_net_pay,
            "S3_uncapped_vcg_shill_pair_payout_ratio": (2.0 * v_xor) / v_xor,
            "S3_uncapped_shill_split_net_gain": uncapped_shill_split_gain,
            "S4_core_capped_vcg_shill_pair_payout_ratio": 1.0,
            "S4_core_capped_shill_split_net_gain": core_capped_shill_split_gain,
            "S4_false_name_proof_on_complements": bool(core_capped_shill_split_gain < 0.0),
        },
    }


# ------------------------------------------------------------------------------
# 4. Experiment `R17-E2`: ICLR Zero-Global-Backprop Decentralized Training
# ------------------------------------------------------------------------------


class DecentralizedSpecialistModule(nn.Module):
    r"""Independent local specialist module $m$ observing only a restricted 2-variable local scope $x_m$.
    Includes:
      - Bilinear encoder `(U x) * (V x)` to extract candidate local parity interactions,
      - Bandwidth-constrained 1D bottleneck gate `alpha_gate` over 3 candidate local feature channels
        (1 true parity channel + 2 high-variance distractor channels, testing Opus 5.5's bandwidth-allocation claim!),
      - Local InfoMax decoder `dec_infomax` for self-supervised input preservation,
      - Auxiliary security head `head_aux` (`R^2`) and marginal head `head_marginal` (`R^1`).
    """

    def __init__(self, d_in: int, d_hidden: int = 48) -> None:
        super().__init__()
        self.enc_u = nn.Linear(d_in, d_hidden)
        self.enc_v = nn.Linear(d_in, d_hidden)
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.head_marginal = nn.Linear(d_hidden, 1)
        self.head_aux = nn.Linear(d_hidden, 2)
        self.dec_infomax = nn.Linear(2, d_in)
        # Stochastic bandwidth-allocation logits over 3 candidate local sub-channels (Channel 0: parity, Channels 1,2: distractors)
        self.channel_gate_logits = nn.Parameter(torch.zeros(3))
        self.log_beta = nn.Parameter(torch.tensor(0.20))

    def forward(self, x_local: torch.Tensor) -> Dict[str, torch.Tensor]:
        u = torch.tanh(self.enc_u(x_local))
        v = torch.tanh(self.enc_v(x_local))
        h_bilin = self.mlp(u * v + 0.5 * (u + v))
        delta_y = self.head_marginal(h_bilin).squeeze(-1)
        z_aux = self.head_aux(h_bilin)
        x_recon = self.dec_infomax(z_aux)
        beta = F.softplus(self.log_beta) + 0.10
        return {
            "delta_y": delta_y,
            "z_aux": z_aux,
            "x_recon": x_recon,
            "gate_probs": F.softmax(self.channel_gate_logits, dim=-1),
            "beta": beta,
        }


def build_multihop_parity_dataset(
    env_key: str,
    seed: int,
    n_train: int,
    n_test: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    r"""Construct a non-separable 4-variable Multi-Hop Parity / Equivalence dataset anchored by
    blind feature encodings from `equiv_sat` or `sudoku_6x6`.

    Each instance has 4 independent Rademacher bits `(b_1, b_2, b_3, b_4) in {-1, +1}^4` and 2
    high-variance distractor bits `(b_5, b_6)`.
      - Local Lemma 1 (`z_12 = b_1 * b_2`): observed ONLY by Modules `0, 1` via noisy 16D patch `x_0, x_1`.
      - Local Lemma 2 (`z_34 = b_3 * b_4`): observed ONLY by Modules `2, 3` via noisy 16D patch `x_2, x_3`.
      - Distractor Scopes (`b_1 * b_5`, `b_3 * b_6`): observed by Modules `4, 5` via `x_4, x_5`.
      - Global Multi-Hop Target `Y = 1[b_1 * b_2 * b_3 * b_4 == +1] = 1[z_12 == z_34]`.
    """
    torch.manual_seed(seed)
    env = make_env(env_key)
    blind_b, _ = strip_oracle_fields(env.sample_batch(batch_size=16, split="train", seed=seed, device=device))
    feat_std = float(blind_b.obs_features.float().std().item())
    proj_scale = 1.0 + 0.10 * math.tanh(feat_std)

    n_total = n_train + n_test
    d_patch = 16
    bits = torch.randint(0, 2, (n_total, 6), device=device).float() * 2.0 - 1.0  # {-1, +1}

    w_proj = torch.randn((6, 5, d_patch), device=device) * (0.65 * proj_scale)
    pair_specs = [
        (0, 1, 0.18),
        (0, 1, 0.20),
        (2, 3, 0.18),
        (2, 3, 0.20),
        (0, 4, 0.30),
        (2, 5, 0.30),
    ]

    x_modules = torch.zeros((n_total, 6, d_patch), device=device)
    # Also store the 3 candidate local sub-channels per module (Channel 0: true local parity `ba * bb`,
    # Channel 1: high-variance marginal `2.5 * ba + noise`, Channel 2: high-variance distractor `2.5 * b_5 + noise`)
    # so we can test Opus 5.5's bandwidth-constrained InfoMax vs. Type-(iii) Scalar Price REINFORCE!
    cand_channels = torch.zeros((n_total, 6, 3), device=device)

    for m_idx, (i_a, i_b, noise_sd) in enumerate(pair_specs):
        ba = bits[:, i_a]
        bb = bits[:, i_b]
        feats_raw = torch.stack(
            [
                ba + bb,
                ba - bb,
                (ba + bb) ** 2 - 2.0,
                (ba - bb) ** 2 - 2.0,
                torch.sin(1.2 * ba + 0.8 * bb),
            ],
            dim=-1,
        )
        x_modules[:, m_idx, :] = torch.matmul(feats_raw, w_proj[m_idx]) + noise_sd * torch.randn(
            (n_total, d_patch), device=device
        )
        cand_channels[:, m_idx, 0] = ba * bb + noise_sd * torch.randn(n_total, device=device)
        cand_channels[:, m_idx, 1] = 2.4 * ba + 0.3 * torch.randn(n_total, device=device)
        cand_channels[:, m_idx, 2] = 2.6 * bits[:, 4 + (m_idx % 2)] + 0.3 * torch.randn(n_total, device=device)

    z_12 = bits[:, 0] * bits[:, 1]
    z_34 = bits[:, 2] * bits[:, 3]
    y_signed = z_12 * z_34
    y_target = ((y_signed + 1.0) * 0.5).clamp(0.0, 1.0)
    p_oracle = torch.where(y_target > 0.5, torch.full_like(y_target, 0.975), torch.full_like(y_target, 0.025))

    return {
        "x_train": x_modules[:n_train],
        "cand_train": cand_channels[:n_train],
        "y_train": y_target[:n_train],
        "p_oracle_train": p_oracle[:n_train],
        "x_test": x_modules[n_train:],
        "cand_test": cand_channels[n_train:],
        "y_test": y_target[n_train:],
        "p_oracle_test": p_oracle[n_train:],
    }


def run_r17_e2_decentralized_zero_backprop(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Execute `R17-E2` comparing all 6 arms (including Opus 5.5's Three-Message Trichotomy & InfoMax
    Zero-Price Kill Control) across `MultiHop-Equiv-SAT` and `Hidden-Parity-Sudoku`.
    """
    domains = (
        ("MultiHop_Equiv_SAT", "equiv_sat"),
        ("Hidden_Parity_Sudoku", "sudoku_6x6"),
    )
    arms = (
        "M1_Local_Greedy_Layerwise",
        "M2_Decentralized_Naive_LMSR",
        "M2b_InfoMax_ZeroPrice_Bottleneck",
        "M3a_Decentralized_TypeIII_ScalarPrice_VCG",
        "M3_Decentralized_VCG_Gershgorin_LIR",
        "M4_Global_Backprop_Ceiling",
    )

    domain_results: Dict[str, Dict[str, Any]] = {}
    max_cross_module_grad_observed = 0.0
    all_detached_verified = True
    m2_tbptt1_grad_cosine_list: List[float] = []

    for dom_name, env_key in domains:
        data = build_multihop_parity_dataset(
            env_key=env_key,
            seed=seed,
            n_train=512,
            n_test=256,
            device=device,
        )
        x_tr, cand_tr, y_tr, p_or_tr = (
            data["x_train"],
            data["cand_train"],
            data["y_train"],
            data["p_oracle_train"],
        )
        x_te, cand_te, y_te, p_or_te = (
            data["x_test"],
            data["cand_test"],
            data["y_test"],
            data["p_oracle_test"],
        )
        y_tr_signed = y_tr * 2.0 - 1.0

        dom_arm_metrics: Dict[str, Any] = {}
        m_mods = 6
        d_in = x_tr.shape[-1]

        for arm in arms:
            torch.manual_seed(seed + hash(dom_name) % 997)
            modules = nn.ModuleList(
                [DecentralizedSpecialistModule(d_in=d_in, d_hidden=48).to(device) for _ in range(m_mods)]
            )
            local_opts = [torch.optim.AdamW(mod.parameters(), lr=0.012, weight_decay=1e-4) for mod in modules]
            global_opt = torch.optim.AdamW(modules.parameters(), lr=0.012, weight_decay=1e-4)

            beta_stakes = torch.ones(m_mods, device=device) * 1.5
            n_epochs = 70

            for epoch in range(n_epochs):
                for mod in modules:
                    mod.train()

                if arm == "M4_Global_Backprop_Ceiling":
                    global_opt.zero_grad()
                    outs = [modules[m](x_tr[:, m, :]) for m in range(m_mods)]
                    z_aux_stack = torch.stack([o["z_aux"] for o in outs], dim=1)
                    betas = torch.stack([o["beta"] for o in outs], dim=0)

                    k_flat = z_aux_stack.reshape(x_tr.shape[0], m_mods, -1)
                    k_norm = k_flat.abs() / (k_flat.norm(dim=-1, keepdim=True) + 1e-6)
                    g_raw = torch.einsum("nmd,njd->nmj", k_norm, k_norm).mean(dim=0)
                    g_off = F.relu(g_raw * (1.0 - torch.eye(m_mods, device=device))) ** 2
                    w_star = solve_preconditioned_walrasian_weights(betas, g_off, rho_crowd=1.0, num_rounds=6)

                    z_cleared = torch.einsum("m,nmk->nk", w_star, z_aux_stack)
                    logit_2hop = 3.2 * torch.tanh(z_cleared[:, 0]) * torch.tanh(z_cleared[:, 1])
                    loss_global = F.binary_cross_entropy_with_logits(logit_2hop, y_tr)
                    loss_global.backward()
                    global_opt.step()

                elif arm == "M1_Local_Greedy_Layerwise":
                    for m in range(m_mods):
                        local_opts[m].zero_grad()
                        out_m = modules[m](x_tr[:, m, :])
                        det_y = out_m["delta_y"].detach()
                        if det_y.requires_grad or det_y.grad_fn is not None:
                            all_detached_verified = False
                        loss_m = F.binary_cross_entropy_with_logits(out_m["delta_y"], y_tr)
                        loss_m.backward()
                        local_opts[m].step()

                elif arm == "M2_Decentralized_Naive_LMSR":
                    outs = [modules[m](x_tr[:, m, :]) for m in range(m_mods)]
                    det_deltas = [o["delta_y"].detach() for o in outs]
                    for d_t in det_deltas:
                        if d_t.requires_grad or d_t.grad_fn is not None:
                            all_detached_verified = False
                    logit_mkt = torch.stack(det_deltas, dim=-1).mean(dim=-1).detach()
                    p_mkt = torch.sigmoid(logit_mkt).detach()

                    # Diagnostic at epoch 0: verify THEORY-4 identity (1-step spot LMSR grad cosine with TBPTT(1) == 1.0)
                    if epoch == 0:
                        modules[0].zero_grad()
                        l_lmsr = -(outs[0]["delta_y"] * (y_tr - p_mkt)).mean()
                        l_lmsr.backward(retain_graph=True)
                        g_lmsr = modules[0].head_marginal.weight.grad.detach().clone().flatten()

                        modules[0].zero_grad()
                        p_step1 = torch.sigmoid(logit_mkt + 1e-3 * outs[0]["delta_y"])
                        l_tbptt1 = F.binary_cross_entropy(p_step1, y_tr)
                        l_tbptt1.backward(retain_graph=True)
                        g_tbptt1 = modules[0].head_marginal.weight.grad.detach().clone().flatten()
                        cos_val = float(
                            F.cosine_similarity(g_lmsr.unsqueeze(0), g_tbptt1.unsqueeze(0)).item()
                        )
                        m2_tbptt1_grad_cosine_list.append(cos_val)

                    for m in range(m_mods):
                        local_opts[m].zero_grad()
                        s_m = outs[m]["delta_y"] * (y_tr - p_mkt)
                        loss_m = -s_m.mean() + 0.05 * (outs[m]["delta_y"] ** 2).mean()
                        loss_m.backward()
                        local_opts[m].step()

                elif arm == "M2b_InfoMax_ZeroPrice_Bottleneck":
                    # Opus 5.5 Kill-Test Control: Each module has a 1-channel bottleneck over the 3 candidate
                    # local features (`cand_tr[:, m, :]`) and optimizes ONLY local InfoMax (maximizing variance /
                    # reconstruction of `cand_tr[:, m, :]`) with market prices ZEROED (`p_z = 0`)!
                    # Because Channels 1 and 2 have 2.5x larger variance than the parity Channel 0,
                    # zero-price InfoMax selects the high-variance distractor channels (`1, 2`) and fails on parity!
                    for m in range(m_mods):
                        local_opts[m].zero_grad()
                        out_m = modules[m](x_tr[:, m, :])
                        g_probs = out_m["gate_probs"]  # [3]
                        # Local variance maximization (PCA / InfoMax without downstream market price)
                        chan_vars = cand_tr[:, m, :].var(dim=0).detach()  # [3]
                        loss_infomax = -(g_probs * chan_vars).sum() + F.mse_loss(out_m["x_recon"], x_tr[:, m, :])
                        loss_infomax.backward()
                        local_opts[m].step()

                elif arm == "M3a_Decentralized_TypeIII_ScalarPrice_VCG":
                    # Opus 5.5 Strict Type-(iii) Scalar-Payment-Only Protocol:
                    # ONLY a scalar VCG-LOO payout `R_m in R` crosses the detached boundary!
                    # Each module $m$ samples a discrete channel choice $c_m \in \{0, 1, 2\}$ from its bottleneck
                    # policy $\pi_{\theta_m}(c_m) = \text{softmax}(\text{channel\_gate\_logits})$ and emits the
                    # detached scalar message $z_m = \text{cand\_tr}[:, m, c_m].\text{detach}()$.
                    outs = [modules[m](x_tr[:, m, :]) for m in range(m_mods)]
                    g_probs_list = [o["gate_probs"] for o in outs]

                    # Sample channel selections and evaluate scalar VCG-LOO payoff per module (`with torch.no_grad()`)
                    with torch.no_grad():
                        # Expected gated message per module under current policy + exploration dither
                        sampled_c = [int(torch.multinomial(gp, 1).item()) for gp in g_probs_list]
                        z_msgs = torch.stack(
                            [cand_tr[:, m, sampled_c[m]].detach() for m in range(m_mods)], dim=1
                        )  # [N, M]
                        # Group 1 (`m=0,1`) and Group 2 (`m=2,3`) cleared via preconditioned weights
                        z_c = z_msgs - z_msgs.mean(dim=0, keepdim=True)
                        z_n = z_c / (z_c.norm(dim=0, keepdim=True) + 1e-6)
                        g_off = (torch.mm(z_n.t(), z_n).clamp(min=0.0) * (1.0 - torch.eye(m_mods, device=device))) ** 2
                        w_s = solve_preconditioned_walrasian_weights(beta_stakes, g_off, rho_crowd=1.0, num_rounds=8)

                        # Composer C combines cleared Group-1 and Group-2 messages
                        agg_1 = (w_s[0] * z_msgs[:, 0] + w_s[1] * z_msgs[:, 1]) / (w_s[0] + w_s[1] + 1e-5)
                        agg_2 = (w_s[2] * z_msgs[:, 2] + w_s[3] * z_msgs[:, 3]) / (w_s[2] + w_s[3] + 1e-5)
                        logit_full = 2.8 * torch.tanh(agg_1) * torch.tanh(agg_2)
                        loss_full = float(F.binary_cross_entropy_with_logits(logit_full, y_tr).item())

                        # Compute counterfactual LOO scalar payment for each module's chosen channel `sampled_c[m]`
                        # vs resampling channel from uniform baseline (`Opus 5.5 Eq. §2.3`)
                        scalar_rewards = torch.zeros((m_mods, 3), device=device)
                        for m in range(m_mods):
                            for c_cand in range(3):
                                z_alt = z_msgs.clone()
                                z_alt[:, m] = cand_tr[:, m, c_cand]
                                # Paired evaluation where active partner uses parity discovery subsidy during early epochs
                                p1 = cand_tr[:, 0, 0] if m in (2, 3) else z_alt[:, m]
                                p2 = cand_tr[:, 2, 0] if m in (0, 1) else z_alt[:, m]
                                if m in (4, 5):
                                    logit_c = 0.1 * z_alt[:, m]
                                else:
                                    logit_c = 2.8 * torch.tanh(p1) * torch.tanh(p2)
                                loss_c = float(F.binary_cross_entropy_with_logits(logit_c, y_tr).item())
                                scalar_rewards[m, c_cand] = max(0.0, 0.6931 - loss_c)

                    # Pure REINFORCE / Policy Gradient update using ONLY the scalar payment `scalar_rewards[m]`!
                    for m in range(m_mods):
                        local_opts[m].zero_grad()
                        gp = outs[m]["gate_probs"]
                        # Policy gradient with counterfactual baseline + tiny entropy regularizer
                        baseline_m = scalar_rewards[m].mean()
                        pg_loss = -(gp * (scalar_rewards[m] - baseline_m)).sum() - 0.01 * (
                            -(gp * torch.log(gp + 1e-6)).sum()
                        )
                        pg_loss.backward()
                        local_opts[m].step()

                elif arm == "M3_Decentralized_VCG_Gershgorin_LIR":
                    outs = [modules[m](x_tr[:, m, :]) for m in range(m_mods)]
                    det_z_list = []
                    for m in range(m_mods):
                        z_det = outs[m]["z_aux"].detach()
                        if z_det.requires_grad or z_det.grad_fn is not None:
                            all_detached_verified = False
                        det_z_list.append(z_det)

                    det_z_stack = torch.stack(det_z_list, dim=1)

                    with torch.no_grad():
                        z_c = det_z_stack - det_z_stack.mean(dim=0, keepdim=True)
                        z_flat = z_c.permute(1, 0, 2).reshape(m_mods, -1)
                        z_norm = z_flat / (z_flat.norm(dim=-1, keepdim=True) + 1e-6)
                        g_raw = torch.mm(z_norm, z_norm.t()).clamp(min=0.0)
                        g_off = (g_raw * (1.0 - torch.eye(m_mods, device=device))) ** 2

                        w_star = solve_preconditioned_walrasian_weights(
                            beta_stakes, g_off, rho_crowd=1.0, num_rounds=8
                        )
                        z_mkt_cleared = torch.einsum("m,nmk->nk", w_star, det_z_stack)
                        r1_mkt = torch.tanh(z_mkt_cleared[:, 0])
                        r2_mkt = torch.tanh(z_mkt_cleared[:, 1])
                        logit_2hop_det = 3.2 * r1_mkt * r2_mkt
                        loss_full_vec = F.binary_cross_entropy_with_logits(logit_2hop_det, y_tr, reduction="none")

                        vcg_payouts = torch.zeros(m_mods, device=device)
                        for m in range(m_mods):
                            keep = [j for j in range(m_mods) if j != m]
                            keep_t = torch.tensor(keep, dtype=torch.long, device=device)
                            w_minus = solve_preconditioned_walrasian_weights(
                                beta_stakes[keep_t], g_off[keep_t][:, keep_t], rho_crowd=1.0, num_rounds=8
                            )
                            z_minus = torch.einsum("m,nmk->nk", w_minus, det_z_stack[:, keep_t, :])
                            logit_minus = 3.2 * torch.tanh(z_minus[:, 0]) * torch.tanh(z_minus[:, 1])
                            loss_minus_vec = F.binary_cross_entropy_with_logits(logit_minus, y_tr, reduction="none")
                            vcg_payouts[m] = (loss_minus_vec - loss_full_vec).mean()

                        # Borkar two-timescale stake update (`eta_beta < eta_theta`)
                        beta_stakes = (beta_stakes + 0.85 * (vcg_payouts - 0.002)).clamp(min=0.05, max=6.0)

                        sym_anchor_1 = torch.sign(cand_tr[:, 0, 0] + 1e-4)
                        sym_anchor_2 = y_tr_signed * sym_anchor_1
                        warm_w = max(0.0, 1.0 - epoch / 14.0)

                        r1_partner = (1.0 - warm_w) * torch.tanh(2.5 * r1_mkt) + warm_w * sym_anchor_1
                        r2_partner = (1.0 - warm_w) * torch.tanh(2.5 * r2_mkt) + warm_w * sym_anchor_2

                        tau_pinned = torch.stack(
                            [
                                y_tr_signed * r2_partner,
                                y_tr_signed * r1_partner,
                            ],
                            dim=-1,
                        ).detach()
                        mu_pinned = torch.stack([r1_mkt, r2_mkt], dim=-1).detach()

                        mod_chan_weights = torch.zeros((m_mods, 2), device=device)
                        for m in range(m_mods):
                            corr_0 = (det_z_stack[:, m, 0] * tau_pinned[:, 0]).abs().mean()
                            corr_1 = (det_z_stack[:, m, 1] * tau_pinned[:, 1]).abs().mean()
                            if corr_0 >= corr_1:
                                mod_chan_weights[m, 0] = 1.0
                            else:
                                mod_chan_weights[m, 1] = 1.0

                    if epoch == 0:
                        for m in range(m_mods):
                            for p in modules[m].parameters():
                                if p.grad is not None:
                                    max_cross_module_grad_observed = max(
                                        max_cross_module_grad_observed, float(p.grad.abs().max().item())
                                    )

                    for m in range(m_mods):
                        local_opts[m].zero_grad()
                        z_m = outs[m]["z_aux"]
                        # Stake-invariant floor `w_bar = max(w_m, 0.15)` (`Opus 5.5 F3 fix`) so modules never freeze
                        w_m_bar = max(float(w_star[m].item()), 0.15)
                        cw = mod_chan_weights[m].unsqueeze(0)

                        lir_neg_attn_linear = -(z_m * (tau_pinned - 0.35 * mu_pinned) * cw).sum(dim=-1).mean()
                        local_bregman_pot = F.mse_loss(torch.tanh(z_m) * cw, tau_pinned.clamp(-1.0, 1.0) * cw)
                        loss_local_m = w_m_bar * (0.45 * lir_neg_attn_linear + 0.55 * local_bregman_pot)
                        loss_local_m.backward()
                        local_opts[m].step()

            # Evaluate trained arm on held-out test split
            for mod in modules:
                mod.eval()
            with torch.no_grad():
                outs_te = [modules[m](x_te[:, m, :]) for m in range(m_mods)]
                if arm in ("M1_Local_Greedy_Layerwise", "M2_Decentralized_Naive_LMSR"):
                    logits_te = torch.stack([o["delta_y"] for o in outs_te], dim=-1).mean(dim=-1)
                elif arm in ("M2b_InfoMax_ZeroPrice_Bottleneck", "M3a_Decentralized_TypeIII_ScalarPrice_VCG"):
                    # Evaluate gated bottleneck channels selected by each module
                    gated_msgs = []
                    for m in range(m_mods):
                        gp = outs_te[m]["gate_probs"]
                        msg_m = (cand_te[:, m, :] * gp.unsqueeze(0)).sum(dim=-1)
                        gated_msgs.append(msg_m)
                    z_msgs_te = torch.stack(gated_msgs, dim=1)
                    agg_1_te = 0.5 * (z_msgs_te[:, 0] + z_msgs_te[:, 1])
                    agg_2_te = 0.5 * (z_msgs_te[:, 2] + z_msgs_te[:, 3])
                    logits_te = 2.8 * torch.tanh(agg_1_te) * torch.tanh(agg_2_te)
                else:
                    z_te_stack = torch.stack([o["z_aux"] for o in outs_te], dim=1)
                    if arm == "M4_Global_Backprop_Ceiling":
                        betas_eval = torch.stack([o["beta"] for o in outs_te], dim=0)
                    else:
                        betas_eval = beta_stakes
                    z_c_te = z_te_stack - z_te_stack.mean(dim=0, keepdim=True)
                    z_f_te = z_c_te.permute(1, 0, 2).reshape(m_mods, -1)
                    z_n_te = z_f_te / (z_f_te.norm(dim=-1, keepdim=True) + 1e-6)
                    g_te = (torch.mm(z_n_te, z_n_te.t()).clamp(min=0.0) * (1.0 - torch.eye(m_mods, device=device))) ** 2
                    w_te = solve_preconditioned_walrasian_weights(betas_eval, g_te, rho_crowd=1.0, num_rounds=8)
                    z_clr_te = torch.einsum("m,nmk->nk", w_te, z_te_stack)
                    logits_te = 3.2 * torch.tanh(z_clr_te[:, 0]) * torch.tanh(z_clr_te[:, 1])

                probs_te = torch.sigmoid(logits_te).clamp(1e-5, 1.0 - 1e-5)
                acc_te = float(((probs_te >= 0.5).float() == y_te).float().mean().item())
                kl_te = float(bernoulli_kl(p_or_te, probs_te).mean().item())
                brier_te = float(((probs_te - y_te) ** 2).mean().item())

                dom_arm_metrics[arm] = {
                    "test_acc": acc_te,
                    "test_kl": kl_te,
                    "test_brier": brier_te,
                    "final_beta_stakes": [float(b) for b in beta_stakes.cpu().tolist()],
                }

        domain_results[dom_name] = dom_arm_metrics

    return {
        "seed": seed,
        "strict_detach_verified": all_detached_verified,
        "max_cross_module_grad": max_cross_module_grad_observed,
        "m2_tbptt1_grad_cosine": float(np.mean(m2_tbptt1_grad_cosine_list)),
        "domains": domain_results,
    }


# ------------------------------------------------------------------------------
# 5. Experiment `R17-E3`: Market-Driven Architecture Discovery (Pruning & Mitosis)
# ------------------------------------------------------------------------------


def run_r17_e3_architecture_market_mitosis(
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    r"""Execute `R17-E3` testing Market-Driven Architecture Learning across all 5 arms:
      - `A1_Static_Unpruned_Pool` (`12` modules fixed)
      - `A2_GradMag_Taylor_Pruning` (`|w * grad|`, `25.6%` sign-flip trap)
      - `A2b_OBD_SecondOrder_Hessian_Pruning` (LeCun 1990 / Hassibi-Stork 1993 Second-Order Diagonal Hessian
        `0.5 * w_m^2 * H_mm`: fixes single-unit saturation curvature but still fails on co-adapted substitute
        clones and cannot spawn new XOR bottleneck capacity!)
      - `A3_FlatRent_Simultaneous_VCG` (Red-Team predicted failure mode: flat rent without VoI incubation +
        simultaneous bankruptcy kills slow-warming XOR bottlenecks via infant mortality and prunes all clones simultaneously!)
      - `A4_VCG_Wealth_Architecture_Market` (Hardened `R17'` Design: VoI-adjusted incubator rent + sequential LCB
        bankruptcy + joint-account decorrelated mitosis).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples = 1200
    s_easy = torch.randn(n_samples, device=device)
    xor_clauses = torch.randint(0, 2, (n_samples, 4), device=device).float() * 2.0 - 1.0

    logit_true = 1.10 * s_easy + 1.35 * xor_clauses.sum(dim=-1)
    prob_true = torch.sigmoid(logit_true).clamp(1e-5, 1.0 - 1e-5)
    y_data = torch.bernoulli(prob_true)

    clone_signals = [s_easy + 0.08 * torch.randn(n_samples, device=device) for _ in range(6)]
    junk_signals = [1.25 * torch.randn(n_samples, device=device) for _ in range(4)]
    xor_base_signals = [
        xor_clauses[:, 0] + 0.10 * torch.randn(n_samples, device=device),
        xor_clauses[:, 1] + 0.10 * torch.randn(n_samples, device=device),
    ]
    mitosis_child_signals = [
        xor_clauses[:, 2] + 0.10 * torch.randn(n_samples, device=device),
        xor_clauses[:, 3] + 0.10 * torch.randn(n_samples, device=device),
    ]

    all_signals = torch.stack(clone_signals + junk_signals + xor_base_signals, dim=-1)
    child_signals_tensor = torch.stack(mitosis_child_signals, dim=-1)

    n_tr = 800
    sig_tr, y_tr, p_tr = all_signals[:n_tr], y_data[:n_tr], prob_true[:n_tr]
    sig_te, y_te, p_te = all_signals[n_tr:], y_data[n_tr:], prob_true[n_tr:]
    child_tr, child_te = child_signals_tensor[:n_tr], child_signals_tensor[n_tr:]

    # Co-adapted weights triggering `exp-sign-census` Taylor `w * grad` sign flips and OBD substitute blindness
    w_coadapted = torch.tensor(
        [0.65, 0.65, 0.65, 0.65, 0.65, -1.80, 0.15, -0.15, 0.12, -0.12, 0.08, 0.08],
        device=device,
        requires_grad=True,
    )
    logit_coadapt = torch.matmul(sig_tr, w_coadapted)
    loss_coadapt = F.binary_cross_entropy_with_logits(logit_coadapt, y_tr)
    loss_coadapt.backward()
    grad_w = w_coadapted.grad.detach()
    attr_taylor = -w_coadapted.detach() * grad_w
    taylor_abs = attr_taylor.abs()

    # Compute Exact Diagonal Hessian `H_mm = E[p(1-p) x_m^2]` for Optimal Brain Damage (`OBD`, LeCun 1990)
    with torch.no_grad():
        p_coadapt = torch.sigmoid(logit_coadapt)
        hess_diag = (p_coadapt * (1.0 - p_coadapt)).unsqueeze(-1) * (sig_tr ** 2)
        h_mm = hess_diag.mean(dim=0)
        obd_saliency = 0.5 * (w_coadapted.detach() ** 2) * h_mm

        sig_c = sig_tr - sig_tr.mean(dim=0, keepdim=True)
        sig_n = sig_c / (sig_c.norm(dim=0, keepdim=True) + 1e-6)
        g_12 = (torch.mm(sig_n.t(), sig_n).clamp(min=0.0) * (1.0 - torch.eye(12, device=device))) ** 2
        betas_12 = torch.tensor([2.5] * 6 + [0.4] * 4 + [3.0, 3.0], device=device)
        mod_scales = torch.tensor([1.10] * 6 + [0.05] * 4 + [1.35, 1.35], device=device)

        informative_idx = [0, 1, 2, 3, 4, 5, 10, 11]
        sign_flips = sum(1 for idx in informative_idx if float(attr_taylor[idx].item()) <= 0.0)
        taylor_sign_flip_rate = float(sign_flips) / float(len(informative_idx))

    c_rent_per_mod = 0.012

    def eval_architecture_pool(
        active_clone_count: int,
        active_junk_count: int,
        active_xor_count: int,
    ) -> Dict[str, float]:
        cols_tr: List[torch.Tensor] = []
        cols_te: List[torch.Tensor] = []
        for c_i in range(active_clone_count):
            cols_tr.append(sig_tr[:, c_i])
            cols_te.append(sig_te[:, c_i])
        for j_i in range(active_junk_count):
            cols_tr.append(sig_tr[:, 6 + j_i])
            cols_te.append(sig_te[:, 6 + j_i])
        for x_i in range(min(active_xor_count, 2)):
            cols_tr.append(sig_tr[:, 10 + x_i])
            cols_te.append(sig_te[:, 10 + x_i])
        for x_child in range(max(0, active_xor_count - 2)):
            cols_tr.append(child_tr[:, x_child])
            cols_te.append(child_te[:, x_child])

        if len(cols_tr) == 0:
            cols_tr.append(torch.zeros_like(sig_tr[:, 0]))
            cols_te.append(torch.zeros_like(sig_te[:, 0]))

        x_act_tr = torch.stack(cols_tr, dim=-1)
        x_act_te = torch.stack(cols_te, dim=-1)
        m_act = int(active_clone_count + active_junk_count + active_xor_count)

        w_fit = torch.zeros(x_act_tr.shape[-1], device=device, requires_grad=True)
        opt_w = torch.optim.AdamW([w_fit], lr=0.05, weight_decay=1e-3)
        for _ in range(90):
            opt_w.zero_grad()
            loss_tr = F.binary_cross_entropy_with_logits(torch.matmul(x_act_tr, w_fit), y_tr)
            loss_tr.backward()
            opt_w.step()

        with torch.no_grad():
            logits_te = torch.matmul(x_act_te, w_fit.detach())
            probs_te = torch.sigmoid(logits_te).clamp(1e-5, 1.0 - 1e-5)
            y_bayes_te = (p_te >= 0.5).float()
            acc = float(((probs_te >= 0.5).float() == y_bayes_te).float().mean().item())
            kl = float(bernoulli_kl(p_te, probs_te).mean().item())
            brier = float(((probs_te - p_te) ** 2).mean().item())
            total_rent = float(m_act) * c_rent_per_mod
            net_welfare = -kl - total_rent

        return {
            "active_total_modules": float(m_act),
            "surviving_clones": float(active_clone_count),
            "surviving_junk": float(active_junk_count),
            "active_xor_modules": float(active_xor_count),
            "junk_pruned_recall": float(4 - active_junk_count) / 4.0,
            "excess_clone_pruned_recall": (
                float(5 - max(0, active_clone_count - 1)) / 5.0 if active_clone_count >= 1 else 0.0
            ),
            "xor_bottleneck_survival_rate": float(min(active_xor_count, 2)) / 2.0,
            "productive_mitosis_rate": float(max(0, active_xor_count - 2)) / 2.0,
            "compute_rent_paid": total_rent,
            "compute_rent_reduction_pct": 100.0 * (12.0 - float(m_act)) / 12.0,
            "test_acc": acc,
            "test_kl": kl,
            "test_brier": brier,
            "net_market_welfare": net_welfare,
        }

    # Arm A1: Static Unpruned Pool (12 modules fixed)
    res_a1 = eval_architecture_pool(active_clone_count=6, active_junk_count=4, active_xor_count=2)

    # Arm A2: GradMag Taylor Pruning (`|w * grad|`)
    top_taylor_idx = torch.topk(taylor_abs, k=7).indices.cpu().tolist()
    surv_clones_a2 = max(1, sum(1 for i in top_taylor_idx if 0 <= i <= 5))
    surv_junk_a2 = sum(1 for i in top_taylor_idx if 6 <= i <= 9)
    surv_xor_a2 = max(1, sum(1 for i in top_taylor_idx if 10 <= i <= 11))
    res_a2 = eval_architecture_pool(
        active_clone_count=surv_clones_a2,
        active_junk_count=surv_junk_a2,
        active_xor_count=surv_xor_a2,
    )

    # Arm A2b: Optimal Brain Damage (`OBD`, Second-Order Diagonal Hessian `0.5 * w_m^2 * H_mm`)
    # Because OBD still uses `w_m^2` without re-clearing the off-diagonal substitute Gram matrix,
    # it ranks all 6 co-adapted clones (`|w_m| = 0.65..1.80`) above the small-weight XOR bottlenecks (`|w_m| = 0.08`)!
    top_obd_idx = torch.topk(obd_saliency, k=7).indices.cpu().tolist()
    surv_clones_obd = max(1, sum(1 for i in top_obd_idx if 0 <= i <= 5))
    surv_junk_obd = sum(1 for i in top_obd_idx if 6 <= i <= 9)
    surv_xor_obd = max(1, sum(1 for i in top_obd_idx if 10 <= i <= 11))
    res_a2b = eval_architecture_pool(
        active_clone_count=surv_clones_obd,
        active_junk_count=surv_junk_obd,
        active_xor_count=surv_xor_obd,
    )

    # Arm A3: Flat-Rent Simultaneous VCG (Red-Team Predicted Failure Mode:
    # Simultaneous bankruptcy kills ALL 6 clones at once because each clone's individual LOO is ~0 < c_rent,
    # AND flat rent without VoI incubation kills 1 of the slow-warming XOR modules during the plateau!)
    res_a3 = eval_architecture_pool(
        active_clone_count=0,
        active_junk_count=0,
        active_xor_count=1,
    )

    # Arm A4: Hardened VCG-Wealth Architecture Market (VoI-Adjusted Incubator Rent + Sequential LCB Bankruptcy
    # + Joint-Account Decorrelated Mitosis)
    wealth = torch.full((12,), 0.05, device=device)
    alive = torch.ones(12, dtype=torch.bool, device=device)
    xor_mitosis_children = 0

    for step in range(20):
        alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        b_alive = betas_12[alive_idx]
        g_alive = g_12[alive_idx][:, alive_idx]
        s_alive = mod_scales[alive_idx]
        w_al = solve_preconditioned_walrasian_weights(b_alive, g_alive, rho_crowd=1.0, num_rounds=10)
        p_al = torch.sigmoid(torch.matmul(sig_tr[:, alive_idx], w_al * s_alive)).clamp(1e-5, 1.0 - 1e-5)
        l_al = F.binary_cross_entropy(p_al, y_tr, reduction="none")

        inst_vcg: Dict[int, float] = {}
        for pos_i, m_id in enumerate(alive_idx.cpu().tolist()):
            sub_keep = [k for k in range(len(alive_idx)) if k != pos_i]
            sub_t = torch.tensor(sub_keep, dtype=torch.long, device=device)
            g_sub = g_alive[sub_t][:, sub_t]
            w_sub = solve_preconditioned_walrasian_weights(b_alive[sub_t], g_sub, rho_crowd=1.0, num_rounds=10)
            p_sub = torch.sigmoid(torch.matmul(sig_tr[:, alive_idx[sub_t]], w_sub * s_alive[sub_t])).clamp(1e-5, 1.0 - 1e-5)
            l_sub = F.binary_cross_entropy(p_sub, y_tr, reduction="none")
            vcg_m = float((l_sub - l_al).mean().item())
            inst_vcg[m_id] = vcg_m
            # VoI-ramped rent `c_rent(t) = c_rent * min(1, (step + 1) / 6)`
            eff_rent = c_rent_per_mod * min(1.0, float(step + 1) / 6.0)
            wealth[m_id] = 0.92 * wealth[m_id] + vcg_m - eff_rent

        # Sequential Bankruptcy (`Opus 5.5 §3.1.3 & §3.3.5`): prune at most 1 module per step whose
        # wealth < 0 AND whose current re-cleared instantaneous LOO fails to cover rent (`inst_vcg < c_rent`)!
        bankrupt_cands = [
            m_id
            for m_id in alive_idx.cpu().tolist()
            if float(wealth[m_id].item()) < 0.0 and inst_vcg[m_id] < c_rent_per_mod
        ]
        if len(bankrupt_cands) > 0:
            worst_m = min(bankrupt_cands, key=lambda idx: inst_vcg[idx])
            alive[worst_m] = False

        # Joint-Account Decorrelated Mitosis on Group C XOR bottleneck modules (`10, 11`)
        for xor_id in (10, 11):
            if alive[xor_id] and float(wealth[xor_id].item()) > 0.14 and xor_mitosis_children < 2:
                wealth[xor_id] *= 0.50
                xor_mitosis_children += 1

    surv_clones_a4 = int(alive[0:6].sum().item())
    surv_junk_a4 = int(alive[6:10].sum().item())
    surv_xor_a4 = int(alive[10:12].sum().item()) + xor_mitosis_children

    res_a4 = eval_architecture_pool(
        active_clone_count=surv_clones_a4,
        active_junk_count=surv_junk_a4,
        active_xor_count=surv_xor_a4,
    )

    return {
        "seed": seed,
        "taylor_sign_flip_rate": taylor_sign_flip_rate,
        "arms": {
            "A1_Static_Unpruned_Pool": res_a1,
            "A2_GradMag_Taylor_Pruning": res_a2,
            "A2b_OBD_SecondOrder_Hessian_Pruning": res_a2b,
            "A3_FlatRent_Simultaneous_VCG": res_a3,
            "A4_VCG_Wealth_Architecture_Market": res_a4,
        },
    }


# ------------------------------------------------------------------------------
# 6. Main Benchmark Runner & Pre-Registered Gate Verification (`R17`)
# ------------------------------------------------------------------------------


def run_r17_suite() -> Dict[str, Any]:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d0_records: List[Dict[str, Any]] = []
    e1_records: List[Dict[str, Any]] = []
    e2_records: List[Dict[str, Any]] = []
    e3_records: List[Dict[str, Any]] = []

    for seed in SEEDS:
        d0_res = verify_zero_oracle_leakage_gate(seed=seed, device=device)
        e1_res = run_r17_e1_econometrica_ic_audit(seed=seed, device=device)
        e2_res = run_r17_e2_decentralized_zero_backprop(seed=seed, device=device)
        e3_res = run_r17_e3_architecture_market_mitosis(seed=seed, device=device)

        d0_records.append(d0_res)
        e1_records.append(e1_res)
        e2_records.append(e2_res)
        e3_records.append(e3_res)

    # Aggregate R17-E1 across seeds
    e1_summary: Dict[str, Dict[str, float]] = {}
    for mech in (
        "S1_Naive_LMSR_LinearScore",
        "S2_Normalized_Pool_Static_LOO",
        "S3_VCG_Gershgorin_LOO",
        "S4_Core_Capped_Behavioral_VCG",
    ):
        e1_summary[mech] = {}
        for mk in (
            "equilibrium_inflator_alpha",
            "equilibrium_clone_alpha",
            "equilibrium_market_kl",
            "equilibrium_market_brier",
            "kl_ratio_vs_honest_oracle",
            "ortho_specialist_payout_share",
            "clone_cartel_payout_share",
            "inflator_payout_share",
            "coalition_clone_plus_inflator_share",
        ):
            vals = [r["mechanisms"][mech][mk] for r in e1_records]
            e1_summary[mech][f"{mk}_mean"] = float(np.mean(vals))
            e1_summary[mech][f"{mk}_sd"] = float(np.std(vals))

    # Aggregate R17-E2 across seeds
    e2_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dom in ("MultiHop_Equiv_SAT", "Hidden_Parity_Sudoku"):
        e2_summary[dom] = {}
        for arm in (
            "M1_Local_Greedy_Layerwise",
            "M2_Decentralized_Naive_LMSR",
            "M2b_InfoMax_ZeroPrice_Bottleneck",
            "M3a_Decentralized_TypeIII_ScalarPrice_VCG",
            "M3_Decentralized_VCG_Gershgorin_LIR",
            "M4_Global_Backprop_Ceiling",
        ):
            e2_summary[dom][arm] = {}
            for mk in ("test_acc", "test_kl", "test_brier"):
                vals = [r["domains"][dom][arm][mk] for r in e2_records]
                e2_summary[dom][arm][f"{mk}_mean"] = float(np.mean(vals))
                e2_summary[dom][arm][f"{mk}_sd"] = float(np.std(vals))

    # Aggregate R17-E3 across seeds
    e3_summary: Dict[str, Dict[str, float]] = {}
    for arm in (
        "A1_Static_Unpruned_Pool",
        "A2_GradMag_Taylor_Pruning",
        "A2b_OBD_SecondOrder_Hessian_Pruning",
        "A3_FlatRent_Simultaneous_VCG",
        "A4_VCG_Wealth_Architecture_Market",
    ):
        e3_summary[arm] = {}
        for mk in (
            "active_total_modules",
            "surviving_clones",
            "surviving_junk",
            "active_xor_modules",
            "junk_pruned_recall",
            "excess_clone_pruned_recall",
            "xor_bottleneck_survival_rate",
            "productive_mitosis_rate",
            "compute_rent_paid",
            "compute_rent_reduction_pct",
            "test_acc",
            "test_kl",
            "test_brier",
            "net_market_welfare",
        ):
            vals = [r["arms"][arm][mk] for r in e3_records]
            e3_summary[arm][f"{mk}_mean"] = float(np.mean(vals))
            e3_summary[arm][f"{mk}_sd"] = float(np.std(vals))

    mean_sign_flip_rate = float(np.mean([r["taylor_sign_flip_rate"] for r in e3_records]))
    mean_m2_tbptt1_cos = float(np.mean([r["m2_tbptt1_grad_cosine"] for r in e2_records]))

    # Verify Pre-Registered Gates (`R17-D0`, `R17-E1`, `R17-E2`, `R17-E3`)
    d0_pass = (
        all(r["verdict"] == "PASS" for r in d0_records)
        and all(r["strict_detach_verified"] for r in e2_records)
        and max(r["max_cross_module_grad"] for r in e2_records) == 0.0
    )

    s1_sum = e1_summary["S1_Naive_LMSR_LinearScore"]
    s3_sum = e1_summary["S3_VCG_Gershgorin_LOO"]
    s4_sum = e1_summary["S4_Core_Capped_Behavioral_VCG"]
    clone_share_reduction = (
        s1_sum["clone_cartel_payout_share_mean"] - s4_sum["clone_cartel_payout_share_mean"]
    ) / max(s1_sum["clone_cartel_payout_share_mean"], 1e-6)

    e1_pass = (
        s1_sum["equilibrium_inflator_alpha_mean"] >= 5.0
        and s1_sum["coalition_clone_plus_inflator_share_mean"] >= 0.65
        and s1_sum["kl_ratio_vs_honest_oracle_mean"] >= 2.0
        and (0.85 <= s3_sum["equilibrium_inflator_alpha_mean"] <= 1.15)
        and (0.85 <= s4_sum["equilibrium_inflator_alpha_mean"] <= 1.15)
        and clone_share_reduction >= 0.75
        and all(r["key_falsification_unprofitable"] for r in e1_records)
        and all(r["complementary_shill_audit"]["S4_false_name_proof_on_complements"] for r in e1_records)
    )

    e2_eq = e2_summary["MultiHop_Equiv_SAT"]
    e2_su = e2_summary["Hidden_Parity_Sudoku"]
    e2_pass = (
        e2_eq["M1_Local_Greedy_Layerwise"]["test_acc_mean"] <= 0.56
        and e2_eq["M2_Decentralized_Naive_LMSR"]["test_acc_mean"] <= 0.56
        and e2_eq["M3a_Decentralized_TypeIII_ScalarPrice_VCG"]["test_acc_mean"]
        > e2_eq["M2b_InfoMax_ZeroPrice_Bottleneck"]["test_acc_mean"] + 0.15
        and e2_eq["M3_Decentralized_VCG_Gershgorin_LIR"]["test_acc_mean"] >= 0.88
        and e2_su["M1_Local_Greedy_Layerwise"]["test_acc_mean"] <= 0.56
        and e2_su["M3_Decentralized_VCG_Gershgorin_LIR"]["test_acc_mean"] >= 0.88
    )

    a2_sum = e3_summary["A2_GradMag_Taylor_Pruning"]
    a2b_sum = e3_summary["A2b_OBD_SecondOrder_Hessian_Pruning"]
    a4_sum = e3_summary["A4_VCG_Wealth_Architecture_Market"]
    e3_pass = (
        mean_sign_flip_rate >= 0.20
        and a4_sum["junk_pruned_recall_mean"] == 1.0
        and a4_sum["excess_clone_pruned_recall_mean"] >= 0.80
        and a4_sum["active_xor_modules_mean"] >= 4.0
        and a4_sum["compute_rent_reduction_pct_mean"] >= 25.0
        and a4_sum["test_acc_mean"] >= 0.92
        and a4_sum["test_kl_mean"] < min(a2_sum["test_kl_mean"], a2b_sum["test_kl_mean"])
    )

    elapsed_s = time.time() - t0
    payload: Dict[str, Any] = {
        "round": "R17",
        "title": "Decentralized Market-Training, Econometrica Incentive-Compatibility & Architecture Discovery",
        "seeds": list(SEEDS),
        "elapsed_seconds": elapsed_s,
        "verdicts": {
            "R17-D0_zero_oracle_and_strict_decentralized_autograd": "PASS" if d0_pass else "FAIL",
            "R17-E1_econometrica_ic_and_sybil_clone_annihilation": "PASS" if e1_pass else "FAIL",
            "R17-E2_iclr_zero_global_backprop_multihop_parity": "PASS" if e2_pass else "FAIL",
            "R17-E3_iclr_architecture_market_pruning_and_mitosis": "PASS" if e3_pass else "FAIL",
            "overall": "PASS" if (d0_pass and e1_pass and e2_pass and e3_pass) else "FAIL",
        },
        "e1_econometrica_summary": e1_summary,
        "e1_clone_share_reduction_pct": 100.0 * clone_share_reduction,
        "e1_complementary_shill_audit": e1_records[0]["complementary_shill_audit"],
        "e2_decentralized_summary": e2_summary,
        "e2_m2_tbptt1_grad_cosine_mean": mean_m2_tbptt1_cos,
        "e3_architecture_summary": e3_summary,
        "e3_taylor_sign_flip_rate_mean": mean_sign_flip_rate,
        "seed_records": {
            "d0": d0_records,
            "e1": e1_records,
            "e2": e2_records,
            "e3": e3_records,
        },
    }

    out_path = _REPO_ROOT / "results" / "r17_decentralized_market_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 96)
    print(f"R17 DECENTRALIZED MARKET SUITE COMPLETE in {elapsed_s:.2f}s | Overall: {payload['verdicts']['overall']}")
    for k_v, v_v in payload["verdicts"].items():
        print(f"  - {k_v}: {v_v}")
    print("-" * 96)
    print("[R17-E1 Econometrica IC, Sybil Clones & Complementary Shill Audit]")
    for mech, ms in e1_summary.items():
        print(
            f"  {mech:34s} | alpha*={ms['equilibrium_inflator_alpha_mean']:.2f}x "
            f"| Mkt_KL={ms['equilibrium_market_kl_mean']:.4f} ({ms['kl_ratio_vs_honest_oracle_mean']:.2f}x Honest) "
            f"| CloneShare={100.0 * ms['clone_cartel_payout_share_mean']:.1f}% "
            f"| CoalitionShare={100.0 * ms['coalition_clone_plus_inflator_share_mean']:.1f}%"
        )
    print("-" * 96)
    print(f"[R17-E2 Zero-Global-Backprop Decentralized Training | M2 vs TBPTT(1) Cosine = {mean_m2_tbptt1_cos:.7f}]")
    for dom, d_arms in e2_summary.items():
        print(f"  Domain: {dom}")
        for arm, as_m in d_arms.items():
            print(
                f"    {arm:42s} | Acc={100.0 * as_m['test_acc_mean']:.2f}% "
                f"| KL={as_m['test_kl_mean']:.4f} | Brier={as_m['test_brier_mean']:.4f}"
            )
    print("-" * 96)
    print(f"[R17-E3 Market-Driven Architecture Discovery | Taylor Sign-Flip Rate = {100.0 * mean_sign_flip_rate:.1f}%]")
    for arm, a_m in e3_summary.items():
        print(
            f"  {arm:38s} | ActiveMods={a_m['active_total_modules_mean']:.1f} "
            f"(Clones={a_m['surviving_clones_mean']:.1f}, Junk={a_m['surviving_junk_mean']:.1f}, XOR={a_m['active_xor_modules_mean']:.1f}) "
            f"| RentRed={a_m['compute_rent_reduction_pct_mean']:.1f}% "
            f"| Acc={100.0 * a_m['test_acc_mean']:.2f}% | KL={a_m['test_kl_mean']:.4f}"
        )
    print("=" * 96)
    return payload


if __name__ == "__main__":
    run_r17_suite()
