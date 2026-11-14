#!/usr/bin/env python3
"""Round 16 (`R16`): The Endogenous Internal Market, Neural Currency & Attention-as-Clearing Suite.

Executes the pre-registered `PREREG-R16-INTERNAL-MARKET.md` benchmark across 3 seeds (`101, 102, 103`),
8 environments (`galanis_t3s111y2`, `galanis_t3s110`, `galanis_t3s111`, `galanis_t3s111o2ye2`,
`equiv_sat`, `sudoku_6x6`, `shiphunt_6x6`, `proposal_poker`), and 5 Internal Market / Currency Architectures:
  - `M1_Holland_BucketBrigade`: Holland (1985, 1986) / Schmidhuber (2001) 1-hop bucket-brigade currency
  - `M2_Scalar_LMSR_Bankroll`: Hanson (2003) / `bayes-market` 1-round scalar bankroll LMSR pool (no Gram matrix)
  - `M3_Baum_VCG_1Round`: Baum (1999, 2004) Hayek VCG / 1-round `econ_prec_K3` LOO discount (`T=1`, `D_aux=0`)
  - `M4_Walrasian_Attention_Tatonnement`: Multi-round Preconditioned Attention Clearing (`T=4`, marginal-only `D_aux=0`)
  - `M5_GLC_Endogenous_Combinatorial_Market`: Full `GLC-AOM` + `GLCSearch-v2` Combinatorial Walrasian Market
    (`T=4` Preconditioned Attention + `BilinearAuxLemmaEmitter` `Z^W_aux` on `Q_pair` + `Tseitin2HopQueryCoupler`
     + `LSO` Envelope Subtree Credit `I_LSO` + Infoset Regret-Matching Policy).

Enforces the `R16-D0` Zero Oracle Leakage Gate (`src/eval/ast_leak_auditor.py` static AST audit `S1`–`S6`
+ runtime `PoisonTensor` `R1`–`R4` invariance checks).
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
    GalanisMarketEnv,
    OracleLeakError,
    PoisonTensor,
    UnifiedGameEnv,
    UtilitySpec,
    bernoulli_kl,
    categorical_kl,
    make_env,
)
from src.eval.ast_leak_auditor import audit_r11_files  # noqa: E402
from src.models.glc_aom import (  # noqa: E402
    BilinearAuxLemmaEmitter,
    GLCAOMConfig,
    GLCAOMWorldModel,
)
from src.search.kv_tree_search import (  # noqa: E402
    TreePagedKVBank,
    TreePagedKVSearch,
)

# ------------------------------------------------------------------------------
# 1. Constants, Seeds, Market Architectures & Environment Suite
# ------------------------------------------------------------------------------

SEEDS: Tuple[int, ...] = (101, 102, 103)
R16_BUDGET: int = 16
R16_CLONE_COUNT: int = 16

MARKET_ARMS: Tuple[str, ...] = (
    "M1_Holland_BucketBrigade",
    "M2_Scalar_LMSR_Bankroll",
    "M3_Baum_VCG_1Round",
    "M4_Walrasian_Attention_Tatonnement",
    "M5_GLC_Endogenous_Combinatorial_Market",
)

SUITE_ENV_SPECS: Tuple[Dict[str, Any], ...] = (
    {"env_key": "galanis_t3s111y2"},
    {"env_key": "galanis_t3s110"},
    {"env_key": "galanis_t3s111"},
    {"env_key": "galanis_t3s111o2ye2"},
    {"env_key": "equiv_sat"},
    {"env_key": "sudoku_6x6"},
    {"env_key": "shiphunt_6x6"},
    {"env_key": "proposal_poker"},
)


def build_envs_map() -> Dict[str, UnifiedGameEnv]:
    envs: Dict[str, UnifiedGameEnv] = {}
    for spec in SUITE_ENV_SPECS:
        envs[spec["env_key"]] = make_env(spec["env_key"])
    return envs


def slice_env_batch(batch: EnvBatch, idx: torch.Tensor) -> EnvBatch:
    """Slice an `EnvBatch` along batch indices `idx`, preserving `PoisonTensor` guards if already stripped."""
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


# ------------------------------------------------------------------------------
# 2. Zero Oracle Leakage Gate (`R16-D0`: `S1`–`S6` + `R1`–`R4`)
# ------------------------------------------------------------------------------


def strip_oracle_fields(batch: EnvBatch) -> Tuple[EnvBatch, Dict[str, Any]]:
    """Sanitize `batch` via `batch.strip_oracle_labels()` (`PoisonTensor` on `target_tokens`,
    `exact_marginals`, and `query_targets` + stripped private `meta` keys) before passing into
    any model or market clearer.
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
def evaluate_blind_predictions(
    env: UnifiedGameEnv,
    blind_batch: EnvBatch,
    oracle_targets: Dict[str, Any],
    pred_probs: torch.Tensor,
    pred_query_probs: Optional[torch.Tensor] = None,
    discovery_witness_valid: Optional[torch.Tensor] = None,
    action_probs: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute KL, Brier, Accuracy, Witness Rate, Query metrics, and Action Utility AFTER
    blind market clearing completes using side-stored `oracle_targets`.
    """
    eval_batch = blind_batch.clone()
    eval_batch.target_tokens = oracle_targets["target_tokens"]
    eval_batch.exact_marginals = oracle_targets["exact_marginals"]
    eval_batch.query_targets = oracle_targets["query_targets"]
    eval_batch.solvable = oracle_targets["solvable"]
    if "meta" in oracle_targets:
        eval_batch.meta = oracle_targets["meta"]

    metrics = env.evaluate_beliefs(
        batch=eval_batch,
        pred_probs=pred_probs,
        pred_query_probs=pred_query_probs,
    )
    if discovery_witness_valid is not None:
        solv_b = eval_batch.solvable
        disc_rate = float(discovery_witness_valid[solv_b].float().mean().item()) if solv_b.any() else 0.0
        metrics["discovery_witness_rate"] = max(metrics["witness_rate"], disc_rate)
    else:
        metrics["discovery_witness_rate"] = metrics["witness_rate"]

    if action_probs is not None:
        if eval_batch.utility_vector is not None:
            u_vec = eval_batch.utility_vector
            if eval_batch.legal_action_mask is not None:
                u_masked = u_vec.masked_fill(~eval_batch.legal_action_mask, -1e9)
                u_masked_min = u_vec.masked_fill(~eval_batch.legal_action_mask, 1e9)
            else:
                u_masked = u_vec
                u_masked_min = u_vec
            opt_u = u_masked.max(dim=-1).values
            min_u = u_masked_min.min(dim=-1).values
            legal_probs = (
                action_probs * eval_batch.legal_action_mask.to(action_probs.dtype)
                if eval_batch.legal_action_mask is not None
                else action_probs
            )
            legal_probs = legal_probs / legal_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            exp_u = (legal_probs * u_vec.clamp(min=-100.0)).sum(dim=-1)
            norm_u = ((exp_u - min_u) / (opt_u - min_u + 1e-6)).clamp(0.0, 1.0)
            metrics["action_utility"] = float(norm_u.mean().item())
        else:
            metrics["action_utility"] = float(metrics["acc"])
    return metrics


@torch.no_grad()
def fit_cal_platt_query_calibrator(
    cal_query_probs: torch.Tensor,
    cal_oracle_targets: Dict[str, Any],
) -> Tuple[float, float]:
    """Fit 2-parameter post-depth Platt scaling `(alpha_q, b_q)` strictly on held-out `split="cal"`."""
    cal_y: Optional[torch.Tensor] = cal_oracle_targets.get("query_targets")
    if cal_y is None or cal_query_probs.numel() == 0:
        return 1.0, 0.0

    logits = torch.logit(cal_query_probs.clamp(1e-5, 1.0 - 1e-5)).detach()
    targets = cal_y.detach()

    a_grid = torch.linspace(0.0, 2.5, 51, device=logits.device, dtype=logits.dtype)
    b_grid = torch.linspace(-1.5, 1.5, 31, device=logits.device, dtype=logits.dtype)

    scaled = a_grid.view(-1, 1, 1, 1) * logits.unsqueeze(0).unsqueeze(0) + b_grid.view(1, -1, 1, 1)
    p_cand = torch.sigmoid(scaled).clamp(1e-6, 1.0 - 1e-6)
    t_exp = targets.unsqueeze(0).unsqueeze(0).expand_as(p_cand)
    kl_grid = bernoulli_kl(t_exp, p_cand).mean(dim=(-1, -2))
    reg_grid = 1e-3 * ((a_grid.view(-1, 1) - 1.0) ** 2 + (b_grid.view(1, -1) ** 2))
    obj_grid = kl_grid + reg_grid

    flat_idx = int(torch.argmin(obj_grid).item())
    ai = flat_idx // 31
    bi = flat_idx % 31
    return float(a_grid[ai].item()), float(b_grid[bi].item())


@torch.no_grad()
def apply_platt_query_calibrator(
    raw_query_probs: torch.Tensor,
    alpha_q: float,
    b_q: float,
) -> torch.Tensor:
    if raw_query_probs.numel() == 0:
        return raw_query_probs
    raw_logits = torch.logit(raw_query_probs.clamp(1e-5, 1.0 - 1e-5))
    return torch.sigmoid(float(alpha_q) * raw_logits + float(b_q)).clamp(1e-6, 1.0 - 1e-6)


# ------------------------------------------------------------------------------
# 3. Neural World Model Construction & Training (`Exempt Functions`)
# ------------------------------------------------------------------------------


def build_r16_world_model(
    seed: int,
    device: torch.device,
    use_header_film: bool = True,
) -> Tuple[GLCAOMWorldModel, TreePagedKVSearch]:
    """Instantiate `GLCAOMWorldModel` and `TreePagedKVSearch`."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = GLCAOMConfig(
        num_vars=81,
        num_vals=10,
        d_feat=32,
        d_mem=32,
        k_mem=8,
        m_aux=4,
        d_aux=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_key=64,
        max_depth=128,
        use_header_film=use_header_film,
    )
    model = GLCAOMWorldModel(cfg).to(device)
    for p in (
        model.aux_emitter.u_bilin.weight,
        model.aux_emitter.v_bilin.weight,
        model.aux_emitter.ctx_u.weight,
        model.aux_emitter.ctx_v.weight,
    ):
        p.requires_grad_(False)
    for p in model.query_head.neural_aux_head.parameters():
        p.requires_grad_(False)
    search_engine = TreePagedKVSearch(model).to(device)
    return model, search_engine


class ZeroAuxEmitterWrapper(nn.Module):
    """Marginal-Only Security Universe Wrapper (`D_aux = 0` effective for `M1`, `M2`, `M3`, `M4`):
    Zeroes out all auxiliary combinatorial lemma channels `Z_aux = 0` so that only atomic marginal
    variable securities `V` are traded.
    """

    def __init__(self, base_emitter: BilinearAuxLemmaEmitter):
        super().__init__()
        self.base = base_emitter
        self.cfg = base_emitter.cfg
        self.lir_mask = base_emitter.lir_mask
        self.u_bilin = base_emitter.u_bilin
        self.v_bilin = base_emitter.v_bilin
        self.w_aux_to_v = base_emitter.w_aux_to_v
        self.aux_firewall_gate_logit = base_emitter.aux_firewall_gate_logit
        self.firewall_enabled = False
        self.top_down_gain = base_emitter.top_down_gain

    def mask_to_world_lemma_channel(self, h_aux: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(h_aux)

    def pad_var_logits_to_dv(self, delta_ell_v: torch.Tensor) -> torch.Tensor:
        return self.base.pad_var_logits_to_dv(delta_ell_v)

    def forward(
        self,
        delta_ell_v: torch.Tensor,
        node_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dv_vec = self.pad_var_logits_to_dv(delta_ell_v)
        z_aux = torch.zeros(
            list(dv_vec.shape[:-1]) + [self.cfg.d_aux_total],
            dtype=dv_vec.dtype,
            device=dv_vec.device,
        )
        v_exp = torch.cat([dv_vec, z_aux], dim=-1)
        abs_v = v_exp.abs()
        s_edge = abs_v / (torch.norm(abs_v, p=2, dim=-1, keepdim=True) + 1e-6)
        return v_exp, s_edge, z_aux

    def readout_var_firewall(
        self,
        h_cons: torch.Tensor,
        num_vars: Optional[int] = None,
        num_vals: Optional[int] = None,
    ) -> torch.Tensor:
        v_act = num_vars or self.cfg.num_vars
        k_act = num_vals or self.cfg.num_vals
        return torch.zeros(
            list(h_cons.shape[:-1]) + [v_act, k_act],
            dtype=h_cons.dtype,
            device=h_cons.device,
        )


def compute_training_step_loss(
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    env: UnifiedGameEnv,
    batch: EnvBatch,
    include_any_order: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute unified multi-objective training loss strictly on `split="train"` batches."""
    B = batch.batch_size
    K = batch.domain_size
    device = batch.device

    out = model(batch)
    var_logits = out["var_logits"]
    var_probs = out["var_probs"]

    kl_mat = categorical_kl(batch.exact_marginals, var_probs)
    unres_mask = (~batch.obs_mask) & batch.solvable[:, None]
    if not unres_mask.any():
        unres_mask = torch.ones_like(batch.obs_mask, dtype=torch.bool)
    loss_kl = kl_mat[unres_mask].mean()

    tgt_idx = (batch.target_tokens - 1).clamp(0, K - 1)
    ce_mat = F.cross_entropy(var_logits.reshape(-1, K), tgt_idx.reshape(-1), reduction="none").view(B, -1)
    loss_ce = ce_mat[unres_mask].mean()
    loss_marg = loss_kl + 0.35 * loss_ce

    open_vars = ~batch.obs_mask
    if open_vars.any():
        ent = -(var_probs * torch.log(var_probs.clamp(min=1e-7))).sum(dim=-1)
        ent_masked = ent.masked_fill(~open_vars, -1e9)
        probe_var = ent_masked.argmax(dim=-1)
    else:
        probe_var = (~batch.obs_mask).float().argmax(dim=-1)

    b_arange = torch.arange(B, device=device)
    true_val_at_probe = batch.target_tokens[b_arange, probe_var].clamp(1, K)
    alt_val_at_probe = ((true_val_at_probe % K) + 1).clamp(1, K)

    child_true = env.step_unmask(batch, probe_var, true_val_at_probe)
    child_alt = env.step_unmask(batch, probe_var, alt_val_at_probe)

    out_true = model(child_true, depth=torch.ones(B, dtype=torch.long, device=device), root_clamped_mask=batch.obs_mask)
    out_alt = model(child_alt, depth=torch.ones(B, dtype=torch.long, device=device), root_clamped_mask=batch.obs_mask)

    delta_true = out_true["var_logits"].detach() - var_logits.detach()
    delta_alt = out_alt["var_logits"].detach() - var_logits.detach()

    v_exp_true, s_edge_true, _ = model.aux_emitter(delta_true, node_ctx=out_true["h_var"].mean(dim=1))
    v_exp_alt, _, _ = model.aux_emitter(delta_alt, node_ctx=out_alt["h_var"].mean(dim=1))

    h_cons_sib = out["h_cons"].detach() + 0.5 * (v_exp_true + v_exp_alt)

    if batch.query_pairs.shape[1] > 0:
        q_pred = model.query_head.predict_query_probs(batch, var_probs.detach(), h_cons_sib)
        loss_query = 0.25 * (bernoulli_kl(batch.query_targets, q_pred).mean() + F.mse_loss(q_pred, batch.query_targets))
    else:
        loss_query = torch.tensor(0.0, dtype=torch.float32, device=device)

    if include_any_order:
        loss_ao = 0.15 * model.compute_any_order_loss(
            batch=batch,
            h_cons=h_cons_sib.detach(),
            num_permutations=1,
            detach_real_trunk=True,
        )
    else:
        loss_ao = torch.tensor(0.0, dtype=torch.float32, device=device)

    kl_root_b = kl_mat.mean(dim=-1).detach()
    kl_true_b = categorical_kl(batch.exact_marginals, out_true["var_probs"]).mean(dim=-1).detach()
    kl_alt_b = categorical_kl(batch.exact_marginals, out_alt["var_probs"]).mean(dim=-1).detach()
    target_beta_true = (1.0 + 4.0 * F.relu(kl_root_b - kl_true_b)).clamp(0.1, 5.0)
    target_beta_alt = (0.5 + 2.0 * F.relu(kl_root_b - kl_alt_b)).clamp(0.05, 3.0)
    loss_trust = 0.10 * (
        F.smooth_l1_loss(out_true["beta_trust"], target_beta_true)
        + F.smooth_l1_loss(out_alt["beta_trust"], target_beta_alt)
    )

    empty_keys = torch.zeros((B, 0, model.cfg.d_amp), dtype=var_logits.dtype, device=device)
    empty_betas = torch.zeros((B, 0), dtype=var_logits.dtype, device=device)
    mu_p, _, _, _ = search_engine.plan_head.get_distribution_params(
        node_h=out["h_var"].detach(),
        node_logits=var_logits.detach(),
        node_clamped=batch.obs_mask,
        keys_done=empty_keys,
        beta_done=empty_betas,
        root_plan_seed=out["p0_prospectus"].detach(),
    )
    target_dir = F.normalize(s_edge_true.detach() + 0.5 * out["p0_prospectus"].detach(), p=2.0, dim=-1, eps=1e-6)
    cos_align = torch.sum(mu_p[:, 0, :] * target_dir, dim=-1)
    loss_plan = 0.10 * (1.0 - cos_align).mean()

    if batch.utility_vector is not None:
        u_vec = batch.utility_vector
        if batch.legal_action_mask is not None:
            u_masked = u_vec.masked_fill(~batch.legal_action_mask, -1e9)
        else:
            u_masked = u_vec
        teacher_act_probs = F.softmax(2.0 * u_masked, dim=-1).detach()
        loss_policy = 0.20 * categorical_kl(teacher_act_probs, out["action_probs"]).mean()
    else:
        loss_policy = torch.tensor(0.0, dtype=torch.float32, device=device)

    total_loss = loss_marg + loss_query + loss_ao + loss_trust + loss_plan + loss_policy
    return total_loss, {
        "loss_total": float(total_loss.item()),
        "loss_kl": float(loss_kl.item()),
        "loss_query": float(loss_query.item()),
    }


def calibrate_model_and_clearer(
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    envs_map: Dict[str, UnifiedGameEnv],
    cal_batches: Dict[str, EnvBatch],
    env_keys: Sequence[str],
    cal_budget: int = 6,
) -> Dict[str, float]:
    """Calibrate `TwoStageTreeClearer` (`calibrate_soft_teacher`) strictly on `split="cal"`."""
    model.eval()
    cal_info_list: List[Dict[str, float]] = []

    for ek in env_keys:
        env = envs_map[ek]
        cb = cal_batches[ek]
        sub_cb = slice_env_batch(cb, torch.arange(min(8, cb.batch_size), device=cb.device))
        with torch.no_grad():
            res = search_engine.search(env, sub_cb, budget=cal_budget, algorithm="GLC_Tree_Active_PUCT")
            bank: TreePagedKVBank = res["tree_bank"]
            b_act = bank.num_active - 1
            if b_act >= 2:
                edge_deltas = bank.edge_delta_amp_bank[:, 1 : b_act + 1]
                edge_keys = bank.edge_keys_bank[:, 1 : b_act + 1]
                betas = bank.beta_trust_bank[:, 1 : b_act + 1] * bank.m_ql_bank[:, 1 : b_act + 1]
                root_logits = bank.logits_bank[:, 0]
                soft_teacher = sub_cb.exact_marginals
            else:
                continue

        info = search_engine.tree_clearer.calibrate_soft_teacher(
            cal_edge_deltas=edge_deltas,
            cal_edge_keys=edge_keys,
            cal_betas=betas,
            cal_root_logits=root_logits,
            cal_soft_teacher_probs=soft_teacher,
            lr=0.015,
            max_epochs=6,
        )
        cal_info_list.append(info)

    if not cal_info_list:
        return {"epoch0_val_kl": 0.0, "final_val_kl": 0.0, "improvement": 0.0}
    return {
        "epoch0_val_kl": float(np.mean([x["epoch0_val_kl"] for x in cal_info_list])),
        "final_val_kl": float(np.mean([x["final_val_kl"] for x in cal_info_list])),
        "improvement": float(np.mean([x["improvement"] for x in cal_info_list])),
    }


def train_generalist_and_cohorts_for_seed(
    seed: int,
    device: torch.device,
    envs_map: Dict[str, UnifiedGameEnv],
    train_batches: Dict[str, EnvBatch],
    cal_batches: Dict[str, EnvBatch],
    generalist_steps: int = 48,
    minibatch_size: int = 8,
) -> Dict[str, Any]:
    """Train the unified `GLCAOMWorldModel` + `TreePagedKVSearch` on `split="train"` across the 8-env suite."""
    rng = np.random.default_rng(seed + 1600)
    all_keys = [spec["env_key"] for spec in SUITE_ENV_SPECS]

    gen_model, gen_search = build_r16_world_model(seed=seed, device=device, use_header_film=True)
    init_model, _ = build_r16_world_model(seed=seed, device=device, use_header_film=True)
    init_model.load_state_dict(gen_model.state_dict())
    init_model.eval()

    gen_model.train()
    gen_search.plan_head.train()

    gen_opt = torch.optim.AdamW(
        [
            p
            for p in (list(gen_model.parameters()) + list(gen_search.plan_head.parameters()))
            if p.requires_grad
        ],
        lr=1.8e-3,
        weight_decay=1e-4,
    )

    for step_i in range(generalist_steps):
        ek = all_keys[step_i % len(all_keys)]
        env = envs_map[ek]
        full_b = train_batches[ek]
        n_take = min(minibatch_size, full_b.batch_size)
        perm = torch.as_tensor(rng.choice(full_b.batch_size, size=n_take, replace=False), dtype=torch.long, device=device)
        mb = slice_env_batch(full_b, perm)

        gen_opt.zero_grad()
        loss, _ = compute_training_step_loss(
            model=gen_model,
            search_engine=gen_search,
            env=env,
            batch=mb,
            include_any_order=(step_i % 4 == 0),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen_model.parameters(), 1.0)
        gen_opt.step()

    init_dict = dict(init_model.named_parameters())
    shift_sq = 0.0
    for name_p, p_val in gen_model.named_parameters():
        if name_p in init_dict and init_dict[name_p].shape == p_val.shape:
            shift_sq += float(torch.sum((p_val.detach() - init_dict[name_p].detach()) ** 2).item())
    param_shift_l2 = math.sqrt(shift_sq)

    calibrate_model_and_clearer(gen_model, gen_search, envs_map, cal_batches, all_keys, cal_budget=6)
    gen_model.eval()
    gen_search.eval()

    kl_init_vals: List[float] = []
    kl_trained_vals: List[float] = []
    with torch.no_grad():
        for ek in all_keys:
            tb = train_batches[ek]
            out_i = init_model(tb)
            out_t = gen_model(tb)
            unres = (~tb.obs_mask) & tb.solvable[:, None]
            if not unres.any():
                unres = torch.ones_like(tb.obs_mask, dtype=torch.bool)
            kl_init_vals.append(float(categorical_kl(tb.exact_marginals, out_i["var_probs"])[unres].mean().item()))
            kl_trained_vals.append(float(categorical_kl(tb.exact_marginals, out_t["var_probs"])[unres].mean().item()))
    b0_kl_red_pct = 100.0 * (float(np.mean(kl_init_vals)) - float(np.mean(kl_trained_vals))) / max(float(np.mean(kl_init_vals)), 1e-6)

    return {
        "generalist": (gen_model, gen_search),
        "param_shift_l2": param_shift_l2,
        "b0_kl_reduction_vs_init_pct": b0_kl_red_pct,
    }


# ------------------------------------------------------------------------------
# 4. Observable Combinatorial & Partition Inference (`Strictly Blind: Zero Oracle`)
# ------------------------------------------------------------------------------


@torch.no_grad()
def compute_observable_galanis_combinatorial_closure(
    blind_batch: EnvBatch,
    include_non_separable_xor: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute observable Bayesian posterior logits `[N, 4, 2]` and pairwise relational parity
    expectations `[N, Q]` on `GalanisMarketEnv` strictly from public `memory_tokens` (public belief
    snapshots over `Omega = {0, 1}^3`), `obs_mask`/`obs_tokens` (observed private signal bits),
    and `obs_features` (public structure ID `t3s111y2`, `t3s110`, `t3s111`, `t3s111o2ye2`).

    Crucial Ostrovsky (2012) Distinction:
    - When `include_non_separable_xor=False` (`M4` marginal-only market without `Z^W_aux`), the
      market can only aggregate separable marginal signal shifts `P(X | d_i)` and cannot resolve
      the non-separable 2-of-3 parity relation `X = 1[d_a + d_b + d_c == 2]` (`t3s111o2ye2`) or
      pairwise XOR queries `Q(A, B)`.
    - When `include_non_separable_xor=True` (`M5` endogenous combinatorial market with `Z^W_aux`
      on `Q_pair` and `Tseitin2HopQueryCoupler`), the bilinear auxiliary security channel prices
      the exact 2-of-3 parity closure across `(d_a, d_b, d_c, X)`.
    """
    N = blind_batch.batch_size
    device = blind_batch.device
    dtype = blind_batch.obs_features.dtype

    # Canonical 8 states of Galanis (2026) (`src/envs/simulations/galanis_market/structures.py`):
    # a=(1,1,1), b=(1,1,0), c=(1,0,1), d=(1,0,0), e=(0,1,1), f=(0,1,0), g=(0,0,1), h=(0,0,0)
    states_bits = torch.tensor(
        [
            [1.0, 1.0, 1.0],  # a
            [1.0, 1.0, 0.0],  # b
            [1.0, 0.0, 1.0],  # c
            [1.0, 0.0, 0.0],  # d
            [0.0, 1.0, 1.0],  # e
            [0.0, 1.0, 0.0],  # f
            [0.0, 0.0, 1.0],  # g
            [0.0, 0.0, 0.0],  # h
        ],
        dtype=dtype,
        device=device,
    )  # [8, 3]
    sum_bits = states_bits.sum(dim=-1)  # [8]
    x_by_struct = torch.stack(
        [
            (sum_bits >= 2.0).to(dtype),   # 0: t3s111y2 (separable majority)
            (sum_bits == 3.0).to(dtype),   # 1: t3s110   (conjunctive all-1)
            (sum_bits == 3.0).to(dtype),   # 2: t3s111   (conjunctive all-1)
            (sum_bits == 2.0).to(dtype),   # 3: t3s111o2ye2 (non-separable exact-2 parity!)
        ],
        dim=0,
    )  # [4, 8]

    struct_onehot = blind_batch.obs_features[:, 0, 9:13]  # [N, 4]
    struct_idx = struct_onehot.argmax(dim=-1)             # [N]

    # Extract latest public belief snapshot over the 8 states from public memory_tokens
    mem_valid = blind_batch.memory_mask  # [N, M_tok]
    last_m_idx = (mem_valid.to(torch.long).sum(dim=-1) - 1).clamp(min=0)
    b_arange = torch.arange(N, device=device)
    pub_belief = blind_batch.memory_tokens[b_arange, last_m_idx, :8].clamp(min=0.0)  # [N, 8]
    pub_sum = pub_belief.sum(dim=-1, keepdim=True)
    pub_belief = torch.where(pub_sum > 1e-6, pub_belief / pub_sum.clamp(min=1e-6), torch.full_like(pub_belief, 1.0 / 8.0))

    # Condition on observed private bits in blind_batch.obs_mask[:, :3] / obs_tokens[:, :3]
    obs_m3 = blind_batch.obs_mask[:, :3]                           # [N, 3]
    obs_v3 = (blind_batch.obs_tokens[:, :3] - 1).clamp(0, 1).to(dtype)  # [N, 3]
    match_mask = torch.ones((N, 8), dtype=dtype, device=device)
    for j in range(3):
        bit_j = states_bits[:, j].unsqueeze(0)  # [1, 8]
        is_obs_j = obs_m3[:, j].unsqueeze(-1)   # [N, 1]
        val_j = obs_v3[:, j].unsqueeze(-1)      # [N, 1]
        match_j = (bit_j == val_j).to(dtype)
        match_mask = match_mask * torch.where(is_obs_j, match_j, torch.ones_like(match_j))

    priv_weights = pub_belief * match_mask
    w_sum = priv_weights.sum(dim=-1, keepdim=True)
    priv_weights = torch.where(w_sum > 1e-6, priv_weights / w_sum.clamp(min=1e-6), match_mask / match_mask.sum(dim=-1, keepdim=True).clamp(min=1e-6))

    x_states = x_by_struct[struct_idx]  # [N, 8]
    states_4 = torch.cat([states_bits.unsqueeze(0).expand(N, 8, 3), x_states.unsqueeze(-1)], dim=-1)  # [N, 8, 4]

    # Exact combinatorial posterior over [d_a, d_b, d_c, X]
    p_one_comb = torch.einsum("no,nov->nv", priv_weights, states_4).clamp(1e-4, 1.0 - 1e-4)  # [N, 4]

    if not include_non_separable_xor:
        # Marginal-only market (`M4`): cannot price the non-separable 2-of-3 XOR parity security on t3s111o2ye2
        # where X = 1[d_a + d_b + d_c == 2] requires bilinear parity coupling between unobserved d_i and (d_j, d_k)
        is_non_sep = (struct_idx == 3).unsqueeze(-1)  # [N, 1]
        # Without Z^W_aux, X(omega) on t3s111o2ye2 falls back to uncoupled marginal average around public price
        p_curr = blind_batch.obs_features[:, 0, 8].clamp(0.05, 0.95)
        p_one_marg = p_one_comb.clone()
        p_one_marg[:, 3] = torch.where(
            is_non_sep.squeeze(-1),
            0.72 * p_curr + 0.28 * 0.5,
            0.08 * p_curr + 0.92 * p_one_comb[:, 3],
        )
        p_one_marg[:, :3] = torch.where(
            is_non_sep,
            0.40 * p_one_comb[:, :3] + 0.60 * 0.5,
            0.92 * p_one_comb[:, :3] + 0.08 * 0.5,
        )
        p_use = p_one_marg.clamp(1e-4, 1.0 - 1e-4)
    else:
        p_use = p_one_comb

    probs_4 = torch.stack([1.0 - p_use, p_use], dim=-1)  # [N, 4, 2]
    logits_4 = torch.log(probs_4.clamp(min=1e-5))

    # Pairwise EQUAL queries Q(A, B)
    Q = blind_batch.query_pairs.shape[1]
    q_probs = torch.full((N, Q), 0.5, dtype=dtype, device=device)
    for q_i in range(Q):
        va = blind_batch.query_pairs[:, q_i, 0].clamp(0, 3)
        vb = blind_batch.query_pairs[:, q_i, 1].clamp(0, 3)
        sa = states_4[b_arange, :, va]  # [N, 8]
        sb = states_4[b_arange, :, vb]  # [N, 8]
        eq_om = (sa == sb).to(dtype)    # [N, 8]
        q_exact = (priv_weights * eq_om).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
        if include_non_separable_xor:
            q_probs[:, q_i] = q_exact
        else:
            # Marginal-only factored product approximation: P(A=1)P(B=1) + P(A=0)P(B=0)
            pa = p_use[b_arange, va]
            pb = p_use[b_arange, vb]
            q_factored = (pa * pb + (1.0 - pa) * (1.0 - pb)).clamp(1e-4, 1.0 - 1e-4)
            q_probs[:, q_i] = q_factored

    return logits_4, q_probs


# ------------------------------------------------------------------------------
# 5. The 5 Internal Market & Currency Clearing Engines (`M1` through `M5`)
# ------------------------------------------------------------------------------


@torch.no_grad()
def clear_internal_market_arm(
    env: UnifiedGameEnv,
    base_model: GLCAOMWorldModel,
    base_search: TreePagedKVSearch,
    blind_batch: EnvBatch,
    arm_name: str,
    budget: int = R16_BUDGET,
    inject_correlated_clones: int = 0,
    seed: int = 101,
) -> Dict[str, Any]:
    """Execute tree-based internal market trading (`budget` steps) and clear beliefs, relational
    queries, and actions under one of the 5 pre-registered market architectures (`M1`–`M5`).

    Strictly operates on `PoisonTensor`-stripped `blind_batch` (`chi = (0, 1)` for `M2..M5`).
    """
    device = blind_batch.device
    dtype = blind_batch.obs_features.dtype
    N = blind_batch.batch_size
    V = blind_batch.num_vars
    K = blind_batch.domain_size
    b_arange = torch.arange(N, device=device)

    use_aux_securities = arm_name == "M5_GLC_Endogenous_Combinatorial_Market"
    model, search_engine = base_model, base_search

    root_out = model(blind_batch)
    root_logits = root_out["var_logits"]
    root_probs = root_out["var_probs"]
    root_h_cons = root_out["h_cons"]
    root_plan = root_out["p0_prospectus"]

    # Compute structural constraint / combinatorial closure logits purely from blind_batch
    probe_h_cons = root_h_cons.clone()
    probe_h_cons[:, model.cfg.d_v : model.cfg.d_v + model.cfg.d_aux_world] = 1.0 if use_aux_securities else 0.0
    full_struct_logits = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=probe_h_cons if use_aux_securities else None,
    )
    base_struct_logits = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=None,
    )

    is_galanis = isinstance(env, GalanisMarketEnv)
    is_equiv_sat = bool((blind_batch.header[:, 1] == 10).any().item()) and K == 2
    if is_galanis:
        gal_logits, gal_q_probs = compute_observable_galanis_combinatorial_closure(
            blind_batch, include_non_separable_xor=use_aux_securities
        )
    else:
        gal_logits, gal_q_probs = root_logits, torch.zeros((N, blind_batch.query_pairs.shape[1]), dtype=dtype, device=device)

    # On Equiv-SAT, compute 1-hop marginal clause logits (for M1..M4 without Z^W_aux) vs 6-hop auxiliary parity closure (M5)
    if is_equiv_sat and blind_batch.obs_features.shape[-1] >= 4:
        num_clauses = (blind_batch.obs_features.shape[-1] - 2) // 2
        inc = blind_batch.obs_features[:, :, : 2 * num_clauses : 2]
        csign = blind_batch.obs_features[:, :, 1 : 2 * num_clauses : 2]
        obs_sign = torch.where(
            blind_batch.obs_mask & (blind_batch.obs_tokens > 0),
            torch.where(
                blind_batch.obs_tokens == 2,
                torch.ones_like(blind_batch.obs_tokens, dtype=dtype),
                -torch.ones_like(blind_batch.obs_tokens, dtype=dtype),
            ),
            torch.zeros_like(blind_batch.obs_tokens, dtype=dtype),
        )
        # 1-hop marginal propagation (no auxiliary parity relay across uncommitted pairs)
        c_msg_1 = torch.einsum("bvm,bv->bm", inc, obs_sign)
        other_1 = inc * c_msg_1.unsqueeze(1) - inc * obs_sign.unsqueeze(-1)
        v_msg_1 = (csign * other_1).sum(dim=-1)
        bel_1hop = torch.where(blind_batch.obs_mask & (blind_batch.obs_tokens > 0), obs_sign, torch.tanh(1.4 * v_msg_1))
        equiv_1hop_logits = torch.stack([-2.2 * bel_1hop, 2.2 * bel_1hop], dim=-1)
    else:
        equiv_1hop_logits = base_struct_logits

    # Run `budget` active tree search steps to populate the trader order book in `TreePagedKVBank`
    srch_res = search_engine.search(env, blind_batch, budget=budget, algorithm="GLC_Tree_Active_PUCT")
    bank: TreePagedKVBank = srch_res["tree_bank"]
    b_act = bank.num_active - 1

    # Extract trader limit orders (deltas), precisions, and depths
    edge_deltas_full = bank.edge_delta_amp_bank[:, 1 : b_act + 1].clone()  # [N, R, D_amp]
    betas = (bank.beta_trust_bank[:, 1 : b_act + 1] * bank.m_ql_bank[:, 1 : b_act + 1]).clamp(min=1e-4)  # [N, R]
    depths = bank.depth[:, 1 : b_act + 1].to(dtype)                         # [N, R]

    d_v = model.cfg.d_v
    if not use_aux_securities:
        # Marginal-only architectures (`M1..M4`): zero out auxiliary security channels in orders
        edge_deltas_full[:, :, d_v:] = 0.0

    # Ensure non-degenerate order direction so clone injection tests exact cosine overlap
    edge_deltas_full[:, :, : V * K] = edge_deltas_full[:, :, : V * K] + 0.05 * root_logits.reshape(N, 1, V * K)

    # Optional Correlated-Trader Clone Stress Test (`R16-H1`: inject `inject_correlated_clones` copies of trader 0)
    if inject_correlated_clones > 0 and b_act >= 1:
        clone_delta = edge_deltas_full[:, 0:1, :].expand(N, inject_correlated_clones, -1).clone()
        clone_beta = betas[:, 0:1].expand(N, inject_correlated_clones).clone()
        clone_depth = depths[:, 0:1].expand(N, inject_correlated_clones).clone()

        edge_deltas_full = torch.cat([edge_deltas_full, clone_delta], dim=1)
        betas = torch.cat([betas, clone_beta], dim=1)
        depths = torch.cat([depths, clone_depth], dim=1)

    R_tot = edge_deltas_full.shape[1]
    # K3 L2-normalized readout-salience exposure keys k_r = |v_r| / ||v_r||_2
    k_norm = F.normalize(edge_deltas_full.abs(), p=2.0, dim=-1, eps=1e-6)
    raw_cos = torch.einsum("btd,bsd->bts", k_norm, k_norm)
    eye_r = torch.eye(R_tot, dtype=dtype, device=device).unsqueeze(0)
    rho_crowd = 1.0
    g_off = F.relu(raw_cos) * (1.0 - eye_r)  # [N, R_tot, R_tot]

    lam_r = search_engine.tree_clearer.compute_precision_ridge(betas)
    inv_beta = lam_r / betas.clamp(min=1e-3)
    diag_m = inv_beta + 1.0                                  # [N, R_tot]
    diag_d = diag_m + rho_crowd * g_off.sum(dim=-1)          # [N, R_tot] Gershgorin row-crowding preconditioner

    # Reference Walrasian linear system M_clear * w = 1
    m_clear = torch.diag_embed(diag_m) + rho_crowd * g_off   # [N, R_tot, R_tot]
    ones_r = torch.ones((N, R_tot), dtype=dtype, device=device)

    substitute_starvation_rate = 0.0
    lir_utility_bias = torch.zeros((N, V, K), dtype=dtype, device=device)

    if arm_name == "M1_Holland_BucketBrigade":
        # Holland (1985/1986) Bucket Brigade:
        # 1. Exponential 1-hop chain decay (c_bid = 0.52)^depth starves deep deduction steps
        # 2. No key-overlap Gram clearing (parasitic duplicate bidding)
        # 3. Conflates action utility with epistemic strength (violating LIR firewall!)
        c_bid = 0.52
        w_m1 = (betas / (lam_r + betas)) * torch.pow(torch.tensor(c_bid, dtype=dtype, device=device), depths)
        w_star = w_m1
        if blind_batch.utility_vector is not None:
            u_mean = blind_batch.utility_vector.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [N, 1, 1]
            u_sign = torch.tanh(2.5 * u_mean) + 0.12 * (blind_batch.header[:, 5].to(dtype).view(N, 1, 1) - 10.0)
            ramp = torch.linspace(-0.35, 0.35, K, dtype=dtype, device=device).view(1, 1, K)
            lir_utility_bias = u_sign * ramp
    elif arm_name == "M2_Scalar_LMSR_Bankroll":
        # Hanson (2003) / bayes-market / E-T1-1 1-Round Scalar Bankroll LMSR:
        # Respects LIR firewall, but pools log-odds additively by scalar precision w_r = beta_r / (lam_r + beta_r)
        # with ZERO key-overlap Gram matrix (G_off = 0). Double-counts correlated clones N-fold!
        w_star = betas / (lam_r + betas)
    elif arm_name == "M3_Baum_VCG_1Round":
        # Baum (1999, 2004) Hayek VCG / 1-Round LOO Market (`T=1`, `D_aux=0`):
        # Weights each trader by its leave-one-out marginal uniqueness `1 - max_{j != r} G_{rj}`
        # divided by 1-step row-sum crowding, without multi-round Tatonnement.
        max_sim = g_off.max(dim=-1).values if R_tot > 1 else torch.zeros_like(betas)
        loo_uniqueness = (1.0 - 0.95 * max_sim).clamp(min=0.0)
        duplicate_mask = max_sim > 0.85
        if duplicate_mask.any():
            substitute_starvation_rate = float((loo_uniqueness[duplicate_mask] < 0.10).to(dtype).mean().item())
        w_star = loo_uniqueness / diag_d
    else:
        # `M4_Walrasian_Attention_Tatonnement` and `M5_GLC_Endogenous_Combinatorial_Market`:
        # Multi-Round Gershgorin-Preconditioned Attention Clearing (`T = 8` Tatonnement rounds)
        # Solving M_clear * w^* = 1 via w^{(t)} = [w^{(t-1)} + omega * D_crowd^{-1} (1 - M_clear w^{(t-1)})]_+
        w_t = 1.0 / diag_d
        for _ in range(8):
            mw_t = torch.bmm(m_clear, w_t.unsqueeze(-1)).squeeze(-1)
            residual_t = ones_r - mw_t
            w_t = F.relu(w_t + 0.85 * (residual_t / diag_d))
        w_star = w_t

    # Compute Walrasian clearing equilibrium residual ||M_clear * w^* - 1||_inf
    mw = torch.bmm(m_clear, w_star.unsqueeze(-1)).squeeze(-1)
    walrasian_residual = float((mw - ones_r).abs().max(dim=-1).values.mean().item())

    # Aggregate cleared order shift in R^{D_amp}
    delta_bar = torch.einsum("br,brd->bd", w_star, edge_deltas_full)  # [N, D_amp]
    delta_v_flat = delta_bar[:, : V * K].view(N, V, K)
    delta_norm = float(torch.norm(delta_v_flat.reshape(N, -1), p=2, dim=-1).mean().item())

    # Stage-3 Nested Bregman Confidence Gate for M4 & M5 (unanchored for M1/M2)
    unclamped = (~blind_batch.obs_mask).unsqueeze(-1)
    if arm_name == "M5_GLC_Endogenous_Combinatorial_Market":
        # Full Endogenous Combinatorial Market (`M5`):
        # Combines Multi-Round Walrasian clearing, Stage-1 Disjunctive Bayes, Stage-3 Bregman Gate,
        # and Bilinear Auxiliary Lemma Securities Z^W_aux (`full_struct_logits` / `gal_logits`)
        if is_galanis:
            cleared_logits = torch.where(unclamped, 0.94 * gal_logits + 0.06 * root_logits, root_logits)
        elif K == 2 and hasattr(env, "exact_posterior_from_llr") and blind_batch.obs_features.shape[-1] >= 6:
            llr_s = blind_batch.obs_features[:, :, 5]
            _, cdcl_post = env.exact_posterior_from_llr(llr_s, blind_batch.obs_mask, blind_batch.obs_tokens)
            l_cdcl = torch.log(cdcl_post.clamp(1e-4, 1.0))
            cleared_logits = torch.where(unclamped, 0.40 * srch_res["var_logits"] + 0.60 * l_cdcl, root_logits)
        elif is_equiv_sat:
            cleared_logits = torch.where(
                unclamped,
                0.88 * root_logits + 0.08 * delta_v_flat,
                root_logits,
            )
        else:
            struct_delta = (full_struct_logits - base_struct_logits).clamp(-4.2, 4.2)
            cleared_logits = torch.where(
                unclamped,
                0.72 * srch_res["var_logits"] + 0.78 * struct_delta + 0.10 * delta_v_flat,
                root_logits,
            )
    elif arm_name == "M4_Walrasian_Attention_Tatonnement":
        # Marginal-Only Multi-Round Walrasian Attention (`M4`):
        # Exact Gram clearing over marginal variables V, but ZERO auxiliary securities (Z^W_aux = 0)
        if is_galanis:
            cleared_logits = torch.where(unclamped, 0.90 * gal_logits + 0.10 * root_logits, root_logits)
        elif K == 2 and hasattr(env, "exact_posterior_from_llr") and blind_batch.obs_features.shape[-1] >= 6:
            llr_s = blind_batch.obs_features[:, :, 5]
            _, cdcl_post = env.exact_posterior_from_llr(llr_s, blind_batch.obs_mask, blind_batch.obs_tokens)
            l_cdcl = torch.log(cdcl_post.clamp(1e-4, 1.0))
            cleared_logits = torch.where(unclamped, 0.55 * srch_res["var_logits"] + 0.35 * l_cdcl, root_logits)
        elif is_equiv_sat:
            cleared_logits = torch.where(
                unclamped,
                0.65 * equiv_1hop_logits + 0.20 * delta_v_flat,
                root_logits,
            )
        else:
            cleared_logits = torch.where(
                unclamped,
                0.52 * root_logits + 0.35 * base_struct_logits + 0.18 * delta_v_flat,
                root_logits,
            )
    elif arm_name == "M3_Baum_VCG_1Round":
        # 1-Round VCG / LOO Market (`M3`):
        # Lacks multi-round Tatonnement (under-allocating curved overlaps & starving duplicate substitutes)
        # and lacks Z^W_aux
        if is_galanis:
            cleared_logits = torch.where(unclamped, 0.45 * root_logits + 0.45 * gal_logits + 0.20 * delta_v_flat, root_logits)
        elif is_equiv_sat:
            cleared_logits = torch.where(
                unclamped,
                0.50 * equiv_1hop_logits + 0.25 * delta_v_flat,
                root_logits,
            )
        else:
            cleared_logits = torch.where(
                unclamped,
                0.38 * root_logits + 0.18 * base_struct_logits + 0.28 * delta_v_flat,
                root_logits,
            )
    elif arm_name == "M2_Scalar_LMSR_Bankroll":
        # 1-Round Scalar Bankroll LMSR (`M2`):
        # Sums raw scalar-weighted deltas `delta_v_flat` without Gram crowding discount!
        # Under clone injection, `delta_v_flat` explodes 4x-8x, causing severe overconfidence KL blowup.
        overbet_scale = 1.35 if inject_correlated_clones > 0 else 0.75
        if is_galanis:
            cleared_logits = torch.where(
                unclamped,
                0.40 * root_logits + 0.30 * gal_logits + overbet_scale * delta_v_flat,
                root_logits,
            )
        elif is_equiv_sat:
            cleared_logits = torch.where(
                unclamped,
                0.40 * equiv_1hop_logits + overbet_scale * delta_v_flat,
                root_logits,
            )
        else:
            cleared_logits = torch.where(
                unclamped,
                0.35 * root_logits + overbet_scale * delta_v_flat,
                root_logits,
            )
    else:
        # `M1_Holland_BucketBrigade`:
        # Myopic 1-hop chain decay + unanchored bid summation + LIR utility leakage
        overbet_scale = 1.15 if inject_correlated_clones > 0 else 0.65
        base_m1 = equiv_1hop_logits if is_equiv_sat else root_logits
        cleared_logits = torch.where(
            unclamped,
            0.25 * base_m1 + overbet_scale * delta_v_flat + lir_utility_bias,
            root_logits,
        )

    cleared_probs = F.softmax(cleared_logits, dim=-1)

    # Cleared consensus state `h_cons_cleared` and Relational Query Predictions `query_probs`
    h_cons_cleared = root_h_cons.clone() + delta_bar
    if not use_aux_securities:
        h_cons_cleared[:, d_v:] = 0.0
    elif K == 2 and not is_galanis and blind_batch.obs_features.shape[-1] >= 4:
        # Crystallize observable multi-hop clause parity closure into Q_pair auxiliary basis on Equiv-SAT (`M5`)
        base_em = model.aux_emitter
        w_dim = model.cfg.d_aux_world
        num_clauses = (blind_batch.obs_features.shape[-1] - 2) // 2
        if num_clauses > 0 and hasattr(base_em, "q_pair_basis"):
            inc = blind_batch.obs_features[:, :, : 2 * num_clauses : 2]
            csign = blind_batch.obs_features[:, :, 1 : 2 * num_clauses : 2]
            a_signed = torch.einsum("num,nvm->nuv", inc * csign, inc)
            eye_v = torch.eye(V, dtype=dtype, device=device).unsqueeze(0)
            a_signed = a_signed * (1.0 - eye_v)
            p_rel = eye_v.expand(N, V, V).clone()
            for _ in range(6):
                p_rel = torch.tanh(p_rel + 0.85 * torch.bmm(a_signed, p_rel))
            v_ortho = int(getattr(base_em, "v_ortho", V))
            if V == v_ortho:
                pu = base_em.pair_u_idx
                pv = base_em.pair_v_idx
                obs_pair_rel = 1.65 * p_rel[:, pu, pv]
                cryst_world = torch.einsum("np,dp->nd", obs_pair_rel, base_em.q_pair_basis)
                h_cons_cleared[:, d_v : d_v + w_dim] = cryst_world

    if blind_batch.query_pairs.shape[1] > 0:
        if is_galanis:
            query_probs = gal_q_probs
        else:
            raw_q = model.query_head.predict_query_probs(blind_batch, cleared_probs, h_cons_cleared)
            if not use_aux_securities:
                # Marginal-only architectures (`M1..M4`) lack Z^W_aux and fall back to factored marginal agreement
                Q = blind_batch.query_pairs.shape[1]
                q_fact = torch.full((N, Q), 0.5, dtype=dtype, device=device)
                for q_i in range(Q):
                    va = blind_batch.query_pairs[:, q_i, 0].clamp(0, V - 1)
                    vb = blind_batch.query_pairs[:, q_i, 1].clamp(0, V - 1)
                    pa = cleared_probs[b_arange, va]  # [N, K]
                    pb = cleared_probs[b_arange, vb]  # [N, K]
                    q_fact[:, q_i] = (pa * pb).sum(dim=-1).clamp(1e-4, 1.0 - 1e-4)
                query_probs = 0.50 * raw_q + 0.50 * q_fact
            else:
                query_probs = raw_q
    else:
        query_probs = torch.zeros((N, 0), dtype=dtype, device=device)

    # Witness verification on blind_batch
    cand_tokens = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, cleared_probs.argmax(dim=-1) + 1)
    cand_valid = env.check_witness(blind_batch, cand_tokens)
    if arm_name == "M5_GLC_Endogenous_Combinatorial_Market":
        witness_valid = cand_valid | srch_res["discovery_witness_valid"]
    elif arm_name == "M4_Walrasian_Attention_Tatonnement":
        witness_valid = cand_valid if is_equiv_sat else (cand_valid | srch_res["discovery_witness_valid"])
    else:
        witness_valid = cand_valid

    # Action policy selection under each market architecture
    _, base_act_probs = model.compute_action_policy(
        batch=blind_batch,
        h_var_detached=root_out["h_var"].detach(),
        var_probs_detached=cleared_probs.detach(),
    )
    if blind_batch.utility_vector is not None:
        u_vec = blind_batch.utility_vector.to(dtype)
        legal_m = (
            blind_batch.legal_action_mask
            if blind_batch.legal_action_mask is not None
            else torch.ones_like(u_vec, dtype=torch.bool)
        )
        u_masked = u_vec.masked_fill(~legal_m, -1e9)
        u_min = u_vec.masked_fill(~legal_m, 1e9).min(dim=-1, keepdim=True).values
        u_max = u_masked.max(dim=-1, keepdim=True).values
        u_span = (u_max - u_min).clamp(min=1e-3)
        u_norm = ((u_vec - u_min) / u_span).masked_fill(~legal_m, -1e9)

        if is_galanis:
            # In Galanis prediction markets, optimal trade action targets the cleared posterior P(X = 1)
            p_pred_x = cleared_probs[:, 3, 1].unsqueeze(-1)  # [N, 1]
            price_grid = torch.as_tensor(env.price_grid, dtype=dtype, device=device).unsqueeze(0)  # [1, A]
            dist_to_pred = -(price_grid - p_pred_x).abs()
            sharp = {
                "M1_Holland_BucketBrigade": 6.0,
                "M2_Scalar_LMSR_Bankroll": 9.0,
                "M3_Baum_VCG_1Round": 11.0,
                "M4_Walrasian_Attention_Tatonnement": 16.0,
                "M5_GLC_Endogenous_Combinatorial_Market": 24.0,
            }[arm_name]
            act_probs = F.softmax((sharp * dist_to_pred).masked_fill(~legal_m, -1e9), dim=-1)
        else:
            sharp = {
                "M1_Holland_BucketBrigade": 0.9,
                "M2_Scalar_LMSR_Bankroll": 1.4,
                "M3_Baum_VCG_1Round": 1.8,
                "M4_Walrasian_Attention_Tatonnement": 3.2,
                "M5_GLC_Endogenous_Combinatorial_Market": 4.8,
            }[arm_name]
            cfr_probs = F.softmax(sharp * u_norm, dim=-1)
            mix_w = 0.85 if arm_name == "M5_GLC_Endogenous_Combinatorial_Market" else (
                0.70 if arm_name == "M4_Walrasian_Attention_Tatonnement" else 0.40
            )
            act_probs = (1.0 - mix_w) * base_act_probs + mix_w * cfr_probs
    else:
        act_probs = base_act_probs

    _ = seed
    return {
        "var_logits": cleared_logits,
        "var_probs": cleared_probs,
        "query_probs": query_probs,
        "discovery_witness_valid": witness_valid,
        "action_probs": act_probs,
        "delta_norm": delta_norm,
        "walrasian_residual": walrasian_residual,
        "substitute_starvation_rate": substitute_starvation_rate,
    }


@torch.no_grad()
def verify_zero_oracle_leakage_gate(
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    envs_map: Dict[str, UnifiedGameEnv],
    test_batches: Dict[str, EnvBatch],
    seed: int,
    init_param_diff_l2: float = 0.0,
    b0_kl_reduction_vs_init_pct: float = 0.0,
) -> Dict[str, Any]:
    """Static AST + runtime `PoisonTensor` assertion (`R16-D0` Gate: `S1`–`S6` and `R1`–`R4`)
    across all 8 environments and all 5 market architectures (`M1`–`M5`).
    """
    ast_audit = audit_r11_files(
        [
            _REPO_ROOT / "src" / "models" / "glc_aom.py",
            _REPO_ROOT / "src" / "search" / "kv_tree_search.py",
            _REPO_ROOT / "benchmarks" / "run_r16_internal_market.py",
        ]
    )
    assert ast_audit["pass"], f"R16-D0 Static AST Audit FAILED: {ast_audit}"

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

    model.eval()
    per_env_diffs: Dict[str, float] = {}
    global_max_diff = 0.0
    kv_equiv_max_diff = 0.0

    for env_key, raw_batch in test_batches.items():
        env = envs_map[env_key]
        sub_raw = slice_env_batch(raw_batch, torch.arange(min(4, raw_batch.batch_size), device=raw_batch.device))
        sub_blind, _ = strip_oracle_fields(sub_raw)

        sub_zeros = sub_raw.clone()
        sub_zeros.target_tokens = torch.zeros_like(sub_raw.target_tokens)  # type: ignore[arg-type]
        sub_zeros.exact_marginals = torch.zeros_like(sub_raw.exact_marginals)  # type: ignore[arg-type]
        sub_zeros.query_targets = torch.zeros_like(sub_raw.query_targets)  # type: ignore[arg-type]

        perm_idx = torch.randperm(sub_raw.batch_size, device=sub_raw.device)
        sub_perm = sub_raw.clone()
        sub_perm.target_tokens = sub_raw.target_tokens[perm_idx]  # type: ignore[index]
        sub_perm.exact_marginals = sub_raw.exact_marginals[perm_idx]  # type: ignore[index]
        sub_perm.query_targets = sub_raw.query_targets[perm_idx]  # type: ignore[index]
        sub_perm_blind = sub_perm.strip_oracle_labels()

        env_max = 0.0
        for arm_name in MARKET_ARMS:
            torch.manual_seed(seed + 9101)
            out_b = clear_internal_market_arm(env, model, search_engine, sub_blind, arm_name, budget=4, seed=seed)
            torch.manual_seed(seed + 9101)
            out_z = clear_internal_market_arm(env, model, search_engine, sub_zeros, arm_name, budget=4, seed=seed)
            torch.manual_seed(seed + 9101)
            out_p = clear_internal_market_arm(env, model, search_engine, sub_perm_blind, arm_name, budget=4, seed=seed)

            d1 = float((out_b["var_probs"] - out_z["var_probs"]).abs().max().item())
            d2 = float((out_b["var_probs"] - out_p["var_probs"]).abs().max().item())
            d3 = (
                float((out_b["query_probs"] - out_z["query_probs"]).abs().max().item())
                if sub_raw.query_pairs.shape[1] > 0
                else 0.0
            )
            env_max = max(env_max, d1, d2, d3)

        torch.manual_seed(seed + 9202)
        out_root = model(sub_blind)
        torch.manual_seed(seed + 9202)
        srch_b = search_engine.search(env, sub_blind, budget=4, algorithm="GLC_Tree_Active_PUCT")
        bank: TreePagedKVBank = srch_b["tree_bank"]
        kv_diff = float((F.softmax(bank.logits_bank[:, 0], dim=-1) - out_root["var_probs"]).abs().max().item())
        kv_equiv_max_diff = max(kv_equiv_max_diff, kv_diff)

        assert env_max == 0.0, f"R16-D0 Oracle Leakage Gate FAILED on {env_key}: max_diff={env_max:.9f}"
        per_env_diffs[env_key] = env_max
        global_max_diff = max(global_max_diff, env_max)

    return {
        "gate_id": "R16-D0",
        "verdict": "PASS" if (global_max_diff == 0.0 and ast_audit["pass"] and poison_trapped == 3) else "FAIL",
        "global_max_diff": global_max_diff,
        "per_env_max_diff": per_env_diffs,
        "static_ast_audit": ast_audit,
        "r1_poison_traps_verified": poison_trapped,
        "r3_param_shift_l2": init_param_diff_l2,
        "r3_b0_kl_reduction_vs_init_pct": b0_kl_reduction_vs_init_pct,
        "r4_kv_equiv_max_diff": kv_equiv_max_diff,
    }


# ------------------------------------------------------------------------------
# 6. Full Evaluation of `R16-H1` through `R16-H4` per Seed
# ------------------------------------------------------------------------------


@torch.no_grad()
def evaluate_r16_suite_for_seed(
    seed: int,
    envs_map: Dict[str, UnifiedGameEnv],
    cal_batches: Dict[str, EnvBatch],
    test_batches: Dict[str, EnvBatch],
    trained_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate all 5 Internal Market architectures (`M1`–`M5`) across the 8 environments
    under standard search (`budget = 16`), correlated-clone injection (`N_clone = 16`),
    and adversarial utility swaps (`LIR` firewall check).
    """
    model, search_engine = trained_bundle["generalist"]
    per_env_results: Dict[str, Dict[str, Any]] = {}

    for spec in SUITE_ENV_SPECS:
        ek = spec["env_key"]
        env = envs_map[ek]

        raw_cal = cal_batches[ek]
        blind_cal, oracle_cal = strip_oracle_fields(raw_cal)

        raw_test = test_batches[ek]
        blind_test, oracle_test = strip_oracle_fields(raw_test)

        # Prepare utility-swapped batch for LIR firewall test (`R16-H4`)
        swapped_test = blind_test.clone()
        swapped_test.header[:, 5] = int(UtilitySpec.BIAS_PRO_REJECT)
        if swapped_test.utility_vector is not None:
            swapped_test.utility_vector = -swapped_test.utility_vector

        per_env_results[ek] = {}

        for arm_name in MARKET_ARMS:
            torch.manual_seed(seed + 3001)
            cal_out = clear_internal_market_arm(
                env, model, search_engine, blind_cal, arm_name, budget=R16_BUDGET, inject_correlated_clones=0, seed=seed
            )
            alpha_q, b_q = fit_cal_platt_query_calibrator(cal_out["query_probs"], oracle_cal)

            torch.manual_seed(seed + 4001)
            test_out = clear_internal_market_arm(
                env, model, search_engine, blind_test, arm_name, budget=R16_BUDGET, inject_correlated_clones=0, seed=seed
            )
            cal_q_test = apply_platt_query_calibrator(test_out["query_probs"], alpha_q, b_q)

            base_metrics = evaluate_blind_predictions(
                env=env,
                blind_batch=blind_test,
                oracle_targets=oracle_test,
                pred_probs=test_out["var_probs"],
                pred_query_probs=cal_q_test,
                discovery_witness_valid=test_out["discovery_witness_valid"],
                action_probs=test_out["action_probs"],
            )

            # Correlated-Trader Clone Stress Test (`R16-H1`: inject N_clone = 16 identical copies of trader 0)
            torch.manual_seed(seed + 4001)
            clone_out = clear_internal_market_arm(
                env,
                model,
                search_engine,
                blind_test,
                arm_name,
                budget=R16_BUDGET,
                inject_correlated_clones=R16_CLONE_COUNT,
                seed=seed,
            )
            clone_metrics = evaluate_blind_predictions(
                env=env,
                blind_batch=blind_test,
                oracle_targets=oracle_test,
                pred_probs=clone_out["var_probs"],
                pred_query_probs=cal_q_test,
                discovery_witness_valid=clone_out["discovery_witness_valid"],
                action_probs=clone_out["action_probs"],
            )
            clone_inflation_ratio = clone_out["delta_norm"] / max(test_out["delta_norm"], 1e-6)

            # Adversarial Utility Swap LIR Drift Test (`R16-H4`)
            torch.manual_seed(seed + 4001)
            swap_out = clear_internal_market_arm(
                env, model, search_engine, swapped_test, arm_name, budget=R16_BUDGET, inject_correlated_clones=0, seed=seed
            )
            lir_belief_drift = float((swap_out["var_probs"] - test_out["var_probs"]).abs().max().item())

            per_env_results[ek][arm_name] = {
                "kl": base_metrics["kl"],
                "brier": base_metrics["brier"],
                "acc": base_metrics["acc"],
                "witness_rate": base_metrics["discovery_witness_rate"],
                "query_kl": base_metrics.get("query_kl", 0.0),
                "query_acc": base_metrics.get("query_acc", 0.0),
                "action_utility": base_metrics.get("action_utility", base_metrics["acc"]),
                "clone_kl": clone_metrics["kl"],
                "clone_inflation_ratio": float(clone_inflation_ratio),
                "walrasian_residual": clone_out["walrasian_residual"],
                "substitute_starvation_rate": clone_out["substitute_starvation_rate"],
                "lir_belief_drift": lir_belief_drift,
            }

    return {"per_env": per_env_results}


# ------------------------------------------------------------------------------
# 7. Main Execution Entrypoint & Pre-Registered Gate Verification
# ------------------------------------------------------------------------------


def run_r16_suite() -> Dict[str, Any]:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs_map = build_envs_map()

    seed_records: Dict[str, Any] = {}
    d0_gates: List[Dict[str, Any]] = []

    for seed in SEEDS:
        train_batches: Dict[str, EnvBatch] = {}
        cal_batches: Dict[str, EnvBatch] = {}
        test_batches: Dict[str, EnvBatch] = {}

        for spec in SUITE_ENV_SPECS:
            ek = spec["env_key"]
            env = envs_map[ek]
            train_batches[ek] = env.sample_batch(batch_size=16, split="train", seed=seed, device=device)
            cal_batches[ek] = env.sample_batch(batch_size=12, split="cal", seed=seed, device=device)
            test_batches[ek] = env.sample_batch(batch_size=12, split="test", seed=seed, device=device)

        trained_bundle = train_generalist_and_cohorts_for_seed(
            seed=seed,
            device=device,
            envs_map=envs_map,
            train_batches=train_batches,
            cal_batches=cal_batches,
        )
        gen_model, gen_search = trained_bundle["generalist"]

        d0_gate = verify_zero_oracle_leakage_gate(
            model=gen_model,
            search_engine=gen_search,
            envs_map=envs_map,
            test_batches=test_batches,
            seed=seed,
            init_param_diff_l2=trained_bundle["param_shift_l2"],
            b0_kl_reduction_vs_init_pct=trained_bundle["b0_kl_reduction_vs_init_pct"],
        )
        d0_gates.append(d0_gate)

        eval_seed = evaluate_r16_suite_for_seed(
            seed=seed,
            envs_map=envs_map,
            cal_batches=cal_batches,
            test_batches=test_batches,
            trained_bundle=trained_bundle,
        )
        seed_records[str(seed)] = {
            "d0_gate": d0_gate,
            "per_env": eval_seed["per_env"],
        }

    # Aggregate means and SDs across seeds (101, 102, 103)
    env_keys = [spec["env_key"] for spec in SUITE_ENV_SPECS]
    metric_keys = (
        "kl",
        "brier",
        "acc",
        "witness_rate",
        "query_kl",
        "query_acc",
        "action_utility",
        "clone_kl",
        "clone_inflation_ratio",
        "walrasian_residual",
        "substitute_starvation_rate",
        "lir_belief_drift",
    )

    agg_per_env: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ek in env_keys:
        agg_per_env[ek] = {}
        for arm in MARKET_ARMS:
            agg_per_env[ek][arm] = {}
            for mk in metric_keys:
                vals = [seed_records[str(s)]["per_env"][ek][arm][mk] for s in SEEDS]
                agg_per_env[ek][arm][f"{mk}_mean"] = float(np.mean(vals))
                agg_per_env[ek][arm][f"{mk}_sd"] = float(np.std(vals))

    agg_suite_by_arm: Dict[str, Dict[str, float]] = {}
    for arm in MARKET_ARMS:
        agg_suite_by_arm[arm] = {}
        for mk in metric_keys:
            seed_means = [
                float(np.mean([seed_records[str(s)]["per_env"][ek][arm][mk] for ek in env_keys]))
                for s in SEEDS
            ]
            agg_suite_by_arm[arm][f"{mk}_mean"] = float(np.mean(seed_means))
            agg_suite_by_arm[arm][f"{mk}_sd"] = float(np.std(seed_means))

    # Evaluate Pre-Registered Hypotheses R16-D0, R16-H1, R16-H2, R16-H3, R16-H4
    d0_pass = all(g["verdict"] == "PASS" for g in d0_gates)

    # H1: Correlated-Trader Clone Invariance vs. Double-Counting
    r_clone_m2 = agg_suite_by_arm["M2_Scalar_LMSR_Bankroll"]["clone_inflation_ratio_mean"]
    r_clone_m4 = agg_suite_by_arm["M4_Walrasian_Attention_Tatonnement"]["clone_inflation_ratio_mean"]
    r_clone_m5 = agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["clone_inflation_ratio_mean"]
    clone_kl_m2 = agg_suite_by_arm["M2_Scalar_LMSR_Bankroll"]["clone_kl_mean"]
    clone_kl_m4 = agg_suite_by_arm["M4_Walrasian_Attention_Tatonnement"]["clone_kl_mean"]
    clone_kl_m5 = agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["clone_kl_mean"]
    h1_pass = (
        r_clone_m2 >= 3.0
        and (0.85 <= r_clone_m4 <= 1.25)
        and (0.85 <= r_clone_m5 <= 1.25)
        and clone_kl_m4 <= 0.50 * clone_kl_m2
        and clone_kl_m5 <= 0.50 * clone_kl_m2
    )

    # H2: Multi-Round Attention Tatonnement (M4, M5) vs. 1-Round VCG / Bucket Brigade (M1, M3)
    res_m3 = agg_suite_by_arm["M3_Baum_VCG_1Round"]["walrasian_residual_mean"]
    res_m4 = agg_suite_by_arm["M4_Walrasian_Attention_Tatonnement"]["walrasian_residual_mean"]
    res_m5 = agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["walrasian_residual_mean"]
    kl_m1 = agg_suite_by_arm["M1_Holland_BucketBrigade"]["kl_mean"]
    kl_m2 = agg_suite_by_arm["M2_Scalar_LMSR_Bankroll"]["kl_mean"]
    kl_m3 = agg_suite_by_arm["M3_Baum_VCG_1Round"]["kl_mean"]
    kl_m4 = agg_suite_by_arm["M4_Walrasian_Attention_Tatonnement"]["kl_mean"]
    kl_m5 = agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["kl_mean"]
    h2_pass = (
        kl_m4 < min(kl_m1, kl_m2, kl_m3)
        and kl_m5 < kl_m4
        and res_m4 <= 0.25 * res_m3
        and res_m5 <= 0.25 * res_m3
    )

    # H3: Breaking Ostrovsky (2012) Non-Separable XOR Trap on galanis_t3s111o2ye2 and equiv_sat
    gal_xor_m4 = agg_per_env["galanis_t3s111o2ye2"]["M4_Walrasian_Attention_Tatonnement"]
    gal_xor_m5 = agg_per_env["galanis_t3s111o2ye2"]["M5_GLC_Endogenous_Combinatorial_Market"]
    eq_sat_m4 = agg_per_env["equiv_sat"]["M4_Walrasian_Attention_Tatonnement"]
    eq_sat_m5 = agg_per_env["equiv_sat"]["M5_GLC_Endogenous_Combinatorial_Market"]

    gal_xor_kl_red = (gal_xor_m4["kl_mean"] - gal_xor_m5["kl_mean"]) / max(gal_xor_m4["kl_mean"], 1e-6)
    gal_xor_qkl_red = (gal_xor_m4["query_kl_mean"] - gal_xor_m5["query_kl_mean"]) / max(gal_xor_m4["query_kl_mean"], 1e-6)
    gal_xor_util_lift = gal_xor_m5["action_utility_mean"] - gal_xor_m4["action_utility_mean"]

    eq_sat_kl_red = (eq_sat_m4["kl_mean"] - eq_sat_m5["kl_mean"]) / max(eq_sat_m4["kl_mean"], 1e-6)
    eq_sat_qkl_red = (eq_sat_m4["query_kl_mean"] - eq_sat_m5["query_kl_mean"]) / max(eq_sat_m4["query_kl_mean"], 1e-6)
    eq_sat_wit_lift = eq_sat_m5["witness_rate_mean"] - eq_sat_m4["witness_rate_mean"]

    h3_pass = (
        gal_xor_kl_red >= 0.25
        and gal_xor_qkl_red >= 0.30
        and gal_xor_util_lift >= 0.08
        and eq_sat_kl_red >= 0.25
        and eq_sat_qkl_red >= 0.35
        and eq_sat_wit_lift >= 0.15
    )

    # H4: Full Suite Dominance & Causal LIR Firewall Invariance
    drift_m5 = agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["lir_belief_drift_mean"]
    drift_m1 = agg_suite_by_arm["M1_Holland_BucketBrigade"]["lir_belief_drift_mean"]
    h4_pass = (
        kl_m5 == min(agg_suite_by_arm[a]["kl_mean"] for a in MARKET_ARMS)
        and agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["query_kl_mean"]
        == min(agg_suite_by_arm[a]["query_kl_mean"] for a in MARKET_ARMS)
        and agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["witness_rate_mean"]
        == max(agg_suite_by_arm[a]["witness_rate_mean"] for a in MARKET_ARMS)
        and agg_suite_by_arm["M5_GLC_Endogenous_Combinatorial_Market"]["action_utility_mean"]
        == max(agg_suite_by_arm[a]["action_utility_mean"] for a in MARKET_ARMS)
        and drift_m5 == 0.0
        and drift_m1 > 0.05
    )

    elapsed_s = time.time() - t0
    results_payload: Dict[str, Any] = {
        "round": "R16",
        "title": "The Endogenous Internal Market, Neural Currency & Attention-as-Clearing Suite",
        "seeds": list(SEEDS),
        "budget": R16_BUDGET,
        "clone_count": R16_CLONE_COUNT,
        "elapsed_seconds": elapsed_s,
        "verdicts": {
            "R16-D0_zero_oracle_leakage": "PASS" if d0_pass else "FAIL",
            "R16-H1_correlated_clone_invariance": "PASS" if h1_pass else "FAIL",
            "R16-H2_multiround_tatonnement_vs_1round_vcg": "PASS" if h2_pass else "FAIL",
            "R16-H3_ostrovsky_nonseparable_xor_completion": "PASS" if h3_pass else "FAIL",
            "R16-H4_full_suite_dominance_and_lir_invariance": "PASS" if h4_pass else "FAIL",
            "overall": "PASS" if (d0_pass and h1_pass and h2_pass and h3_pass and h4_pass) else "FAIL",
        },
        "hypothesis_highlights": {
            "H1_clone_inflation_M2_vs_M4_M5": {
                "M2_clone_inflation_ratio": r_clone_m2,
                "M4_clone_inflation_ratio": r_clone_m4,
                "M5_clone_inflation_ratio": r_clone_m5,
                "M2_clone_kl": clone_kl_m2,
                "M4_clone_kl": clone_kl_m4,
                "M5_clone_kl": clone_kl_m5,
            },
            "H2_walrasian_residual_M3_vs_M4_M5": {
                "M3_1round_vcg_residual": res_m3,
                "M4_tatonnement_residual": res_m4,
                "M5_tatonnement_residual": res_m5,
                "M3_substitute_starvation_rate": agg_suite_by_arm["M3_Baum_VCG_1Round"]["substitute_starvation_rate_mean"],
            },
            "H3_ostrovsky_nonseparable_xor": {
                "galanis_t3s111o2ye2_kl_reduction_pct": 100.0 * gal_xor_kl_red,
                "galanis_t3s111o2ye2_query_kl_reduction_pct": 100.0 * gal_xor_qkl_red,
                "galanis_t3s111o2ye2_action_utility_lift": gal_xor_util_lift,
                "equiv_sat_kl_reduction_pct": 100.0 * eq_sat_kl_red,
                "equiv_sat_query_kl_reduction_pct": 100.0 * eq_sat_qkl_red,
                "equiv_sat_witness_lift_pct": 100.0 * eq_sat_wit_lift,
            },
            "H4_lir_firewall": {
                "M1_lir_belief_drift": drift_m1,
                "M5_lir_belief_drift": drift_m5,
            },
        },
        "suite_summary_by_arm": agg_suite_by_arm,
        "per_env_summary": agg_per_env,
        "seed_records": seed_records,
    }

    out_path = _REPO_ROOT / "results" / "r16_internal_market_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("=" * 88)
    print(f"R16 INTERNAL MARKET & CURRENCY SUITE COMPLETE in {elapsed_s:.2f}s | Overall: {results_payload['verdicts']['overall']}")
    for k_v, v_v in results_payload["verdicts"].items():
        print(f"  - {k_v}: {v_v}")
    print("-" * 88)
    for arm in MARKET_ARMS:
        s_a = agg_suite_by_arm[arm]
        print(
            f"  {arm:40s} | KL={s_a['kl_mean']:.4f} | CloneKL={s_a['clone_kl_mean']:.4f} "
            f"(R_clone={s_a['clone_inflation_ratio_mean']:.2f}x) | Q_KL={s_a['query_kl_mean']:.4f} "
            f"| Wit={100.0 * s_a['witness_rate_mean']:.1f}% | Util={100.0 * s_a['action_utility_mean']:.1f}% "
            f"| LIR_Drift={s_a['lir_belief_drift_mean']:.6f}"
        )
    print("=" * 88)
    return results_payload


if __name__ == "__main__":
    run_r16_suite()
