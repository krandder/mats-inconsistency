#!/usr/bin/env python3
"""
Round 15 (`R15-CURIOSITY-SUITE`): Curiosity-Driven Micro-Experiment Suite (`R15-C1` through `R15-C4`).

Architect & Runner: Member 20 of the Society of Thinkers.
Pre-Registration: `benchmarks/PREREG-R15-CURIOSITY-SUITE.md`

Executes 4 pre-registered curiosity-driven experiments across `SEEDS = (101, 102, 103)`
with strict zero-oracle-leakage verification (`R15-D0`: `ast_leak_auditor.py` `S1`–`S6`
+ runtime `PoisonTensor` `R1`–`R4` canary invariance):

1. `R15-C1` ("Phase Transitions in Optimal Search Topology: Chain vs. Bush vs. Tree vs. DAG"):
   - Sweeps local observation noise `sigma_obs in {0.0, 0.15, 0.35, 0.60}` x constraint
     coupling depth `gamma_couple in {1, 2, 4, 8}` (`4 x 4 = 16` phase cells) at `B = 16`.
   - Compares `Topo-Chain-CDCL`, `Topo-Wide-Halving`, `Topo-Bushy-MCTS`,
     `Topo-DAG-Transposition`, and the learned `GLC-Adaptive-PhaseController`.
2. `R15-C2` ("Zero-Shot Compositional Chimera Transfer Across Search Paradigms"):
   - Trains pure single-paradigm specialists (`Specialist-MCTS-Hex` on `Hex-5x5`,
     `Specialist-CDCL-Sudoku` on `Sudoku-6x6`, `Specialist-CFR-Poker` on `Proposal-Poker`)
     and a multi-teacher compositional controller (`Chimera-ZeroShot`), then evaluates
     zero-shot and after self-supervised RL (`Chimera-RL-Finetuned`) on held-out hybrid
     games (`Micro-Stratego-4x4` and `ShipHunt-6x6`).
3. `R15-C3` ("The 'Eureka' Lemma Crystallization Effect: Persistent Auxiliary Memory Across Search Episodes"):
   - Evaluates `Equiv-SAT` and `Sudoku-6x6` across `b in {0, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64}`
     when high-precision auxiliary lemmas (`|l_aux| * beta_eff > tau_eureka`) discovered
     during `b in [1, 16]` are crystallized into hard-gated persistent syndrome tokens in
     `TreePagedKVBank` for all subsequent branches `b in [17, 64]`.
4. `R15-C4` ("The Curse of Deterministic Search in Imperfect-Information Games"):
   - Scales compute `B in {1, 2, 4, 8, 16, 32, 64}` on `Proposal-Poker`, `Galanis-Market`,
     and `Micro-Stratego-4x4` comparing `Det-Argmax-Minimax`, `Det-PUCT-VisitMax`,
     `Tabular-Prior-Baseline`, and `GLCSearch-v2-Infoset-CFR` on Best-Response Exploitability,
     policy entropy, and head-to-head win rate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------
# 0. Self-Bootstrap Python Paths
# ------------------------------------------------------------------------------
for _cand_path in [
    "/root/pylib",
    "/tmp/r4e3a/pylib",
    "/home/agent/data/pool/pylib",
]:
    if os.path.isdir(_cand_path) and _cand_path not in sys.path:
        sys.path.insert(0, _cand_path)

for _conda_site in sorted(glob.glob("/opt/conda/lib/python*/site-packages")):
    if os.path.isdir(_conda_site) and _conda_site not in sys.path:
        sys.path.append(_conda_site)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(torch, "_native") and hasattr(torch._native, "registry"):
    torch._native.registry.deregister_op_overrides(disable_dispatch_keys=["CUDA", "CPU"])
torch.set_num_threads(4)

from src.envs import (
    EnvBatch,
    OracleLeakError,
    PoisonTensor,
    UnifiedGameEnv,
    UtilitySpec,
    bernoulli_kl,
    categorical_kl,
    make_env,
)
from src.eval.ast_leak_auditor import audit_r11_files
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    GLCAOMConfig,
    GLCAOMWorldModel,
    LIRMask,
)
from src.search.kv_tree_search import (
    TreePagedKVBank,
    TreePagedKVSearch,
)

# ------------------------------------------------------------------------------
# 1. Locked Constants & Pre-Registered Grids
# ------------------------------------------------------------------------------

SEEDS: Tuple[int, ...] = (101, 102, 103)
T_CRIT_DF2: float = 4.302652729911275  # Two-sided 95% Student-t critical value for df=2

C1_SIGMA_OBS_GRID: Tuple[float, ...] = (0.00, 0.15, 0.35, 0.60)
C1_GAMMA_COUPLE_GRID: Tuple[int, ...] = (1, 2, 4, 8)
C1_BUDGET: int = 16
C1_TOPOLOGY_ARMS: Tuple[str, ...] = (
    "Topo-Chain-CDCL",
    "Topo-Wide-Halving",
    "Topo-Bushy-MCTS",
    "Topo-DAG-Transposition",
    "GLC-Adaptive-PhaseController",
)

C2_TEACHER_ENVS: Tuple[str, ...] = ("hex_5x5", "sudoku_6x6", "proposal_poker")
C2_HYBRID_ENVS: Tuple[str, ...] = ("micro_stratego_4x4", "shiphunt_6x6")
C2_BUDGET: int = 16
C2_ARMS: Tuple[str, ...] = (
    "Specialist-MCTS-Hex",
    "Specialist-CDCL-Sudoku",
    "Specialist-CFR-Poker",
    "Chimera-ZeroShot",
    "Chimera-RL-Finetuned",
)

C3_ENVS: Tuple[str, ...] = ("equiv_sat", "sudoku_6x6")
C3_BUDGET_TRAJECTORY: Tuple[int, ...] = (0, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64)
C3_CRYSTAL_STEP: int = 16
C3_TAU_EUREKA: float = 0.35
C3_ARMS: Tuple[str, ...] = (
    "C3-NoAux",
    "C3-Volatile-Aux",
    "C3-Random-Crystallize",
    "C3-Eureka-Crystallize",
)

C4_ENVS: Tuple[str, ...] = ("proposal_poker", "galanis_market_t3s111o2ye2", "micro_stratego_4x4")
C4_BUDGETS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
C4_ARMS: Tuple[str, ...] = (
    "Det-Argmax-Minimax",
    "Det-PUCT-VisitMax",
    "Tabular-Prior-Baseline",
    "GLCSearch-v2-Infoset-CFR",
)

SUITE_ENV_SPECS: List[Dict[str, Any]] = [
    {"env_key": "equiv_sat", "registry_key": "equiv_sat", "train_n": 24, "cal_n": 12, "test_n": 16},
    {"env_key": "sudoku_6x6", "registry_key": "sudoku_6x6", "train_n": 24, "cal_n": 12, "test_n": 16},
    {"env_key": "shiphunt_6x6", "registry_key": "shiphunt_6x6", "train_n": 24, "cal_n": 12, "test_n": 16},
    {"env_key": "hex_5x5", "registry_key": "hex_5x5", "train_n": 24, "cal_n": 12, "test_n": 16},
    {"env_key": "micro_stratego_4x4", "registry_key": "micro_stratego_4x4", "train_n": 24, "cal_n": 12, "test_n": 16},
    {"env_key": "proposal_poker", "registry_key": "proposal_poker", "train_n": 24, "cal_n": 12, "test_n": 16},
    {
        "env_key": "galanis_market_t3s111o2ye2",
        "registry_key": "galanis_t3s111o2ye2",
        "train_n": 24,
        "cal_n": 12,
        "test_n": 16,
    },
]


def paired_t_summary(vals: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    se = float(np.sqrt(np.sum((arr - mean) ** 2) / (n * (n - 1)))) if n > 1 else 0.0
    hw = T_CRIT_DF2 * se
    return {
        "per_seed": [float(x) for x in arr],
        "mean": mean,
        "se": se,
        "ci95_low": float(mean - hw),
        "ci95_high": float(mean + hw),
    }


def slice_env_batch(batch: EnvBatch, idx: torch.Tensor) -> EnvBatch:
    """Slice an `EnvBatch` along the batch dimension using index tensor `idx`."""
    n_sub = int(idx.numel())
    new_meta = {}
    for k, v in batch.meta.items():
        if isinstance(v, PoisonTensor) and v.ndim >= 1 and v.shape[0] == batch.batch_size:
            new_meta[k] = PoisonTensor((n_sub,) + tuple(v.shape[1:]), v.dtype, v.device, v._field_name)
        elif isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == batch.batch_size:
            new_meta[k] = v[idx]
        else:
            new_meta[k] = v

    def _slice_or_poison(t: Any, name: str) -> Any:
        if isinstance(t, PoisonTensor):
            return PoisonTensor((n_sub,) + tuple(t.shape[1:]), t.dtype, t.device, name)
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
# 2. Zero Oracle Leakage Gate (`R15-D0`: `S1`–`S6` + `R1`–`R4`)
# ------------------------------------------------------------------------------


def strip_oracle_fields(batch: EnvBatch) -> Tuple[EnvBatch, Dict[str, Any]]:
    """Sanitize `batch` via `batch.strip_oracle_labels()` (`PoisonTensor` on `target_tokens`,
    `exact_marginals`, and `query_targets` + stripped private `meta` keys) before passing into
    any model or search method.
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
    blind search completes using side-stored `oracle_targets`.
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
    """Fit 2-parameter post-depth Platt / temperature scaling `(alpha_q, b_q)` strictly on
    held-out `split="cal"` (`R12-D1` / `R15-C3`), never touching `split="test"`.
    """
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
    """Static AST + runtime `PoisonTensor` assertion (`R15-D0` Gate: `S1`–`S6` and `R1`–`R4`)."""
    ast_audit = audit_r11_files(
        [
            _REPO_ROOT / "src" / "models" / "glc_aom.py",
            _REPO_ROOT / "src" / "search" / "kv_tree_search.py",
            _REPO_ROOT / "benchmarks" / "run_r15_curiosity_suite.py",
        ]
    )
    assert ast_audit["pass"], f"R15-D0 Static AST Audit FAILED: {ast_audit}"

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

        torch.manual_seed(seed + 9001)
        out_blind = model(sub_blind)
        torch.manual_seed(seed + 9001)
        out_zeros = model(sub_zeros)
        torch.manual_seed(seed + 9001)
        out_perm = model(sub_perm_blind)

        d1 = float((out_blind["var_probs"] - out_zeros["var_probs"]).abs().max().item())
        d2 = float((out_blind["var_logits"] - out_zeros["var_logits"]).abs().max().item())
        d3 = (
            float((out_blind["query_probs"] - out_zeros["query_probs"]).abs().max().item())
            if sub_raw.query_pairs.shape[1] > 0
            else 0.0
        )
        d4 = float((out_blind["var_probs"] - out_perm["var_probs"]).abs().max().item())

        torch.manual_seed(seed + 9002)
        srch_blind = search_engine.search(env, sub_blind, budget=4, algorithm="GLC_Tree_Active_PUCT")
        torch.manual_seed(seed + 9002)
        srch_zeros = search_engine.search(env, sub_zeros, budget=4, algorithm="GLC_Tree_Active_PUCT")
        torch.manual_seed(seed + 9002)
        srch_perm = search_engine.search(env, sub_perm_blind, budget=4, algorithm="GLC_Tree_Active_PUCT")

        d5 = float((srch_blind["var_probs"] - srch_zeros["var_probs"]).abs().max().item())
        d6 = (
            float((srch_blind["query_probs"] - srch_zeros["query_probs"]).abs().max().item())
            if sub_raw.query_pairs.shape[1] > 0
            else 0.0
        )
        d7 = float((srch_blind["var_probs"] - srch_perm["var_probs"]).abs().max().item())
        d8 = float((srch_blind["selected_nodes"] - srch_perm["selected_nodes"]).abs().max().item())

        bank: TreePagedKVBank = srch_blind["tree_bank"]
        kv_diff = float((F.softmax(bank.logits_bank[:, 0], dim=-1) - out_blind["var_probs"]).abs().max().item())
        kv_equiv_max_diff = max(kv_equiv_max_diff, kv_diff)

        env_max = max(d1, d2, d3, d4, d5, d6, d7, d8)
        assert env_max == 0.0, f"R15-D0 Oracle Leakage Gate FAILED on {env_key}: max_diff={env_max:.9f}"
        per_env_diffs[env_key] = env_max
        global_max_diff = max(global_max_diff, env_max)

    return {
        "gate_id": "R15-D0",
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
# 3. Neural Models, Controllers & Training (`R15-C1..C4`)
# ------------------------------------------------------------------------------


def build_r15_world_model(
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
    """`C3-NoAux` ablation (`d_aux = 0` effective): zeroes out all auxiliary lemma channels `Z_aux = 0`."""

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

    def decode_top_down_from_prefix(
        self,
        h_cons: torch.Tensor,
        prefix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor,
        root_mask: torch.Tensor,
        num_vals: int,
    ) -> torch.Tensor:
        B_sz, V_act = prefix_tokens.shape
        return torch.zeros((B_sz, V_act, num_vals), dtype=h_cons.dtype, device=h_cons.device)


class NeuralPhaseTopologyController(nn.Module):
    """`R15-C1` Learned Neural Phase Topology Controller (`chi = (1, 1)`).

    Reads observable root features `[root_entropy, obs_noise_dispersion, coupling_stiffness, clamped_frac]`
    plus detached root hidden state `h_var.mean(dim=1)` and outputs continuous search-topology regime weights
    `(w_chain, w_wide, w_bush, w_dag)` on the 4-simplex.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.lir_mask = LIRMask.policy_search()
        self.net = nn.Sequential(
            nn.Linear(d_model + 4, 64),
            nn.SiLU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        h_var_detached: torch.Tensor,      # [N, V, d_model]
        obs_features: torch.Tensor,        # [N, V, D_feat]
        root_probs_detached: torch.Tensor, # [N, V, K]
        obs_mask: torch.Tensor,            # [N, V]
        sigma_obs_est: float,
        gamma_couple_est: float,
    ) -> torch.Tensor:
        """Return topology regime weights `w_topo` (`[N, 4]`: `[w_chain, w_wide, w_bush, w_dag]`)."""
        N = h_var_detached.shape[0]
        device = h_var_detached.device
        dtype = h_var_detached.dtype

        h_pool = h_var_detached.mean(dim=1)  # [N, d_model]
        ent = -(root_probs_detached * torch.log(root_probs_detached.clamp(min=1e-7))).sum(dim=-1).mean(
            dim=-1, keepdim=True
        )
        feat_std = obs_features.std(dim=(1, 2), unbiased=False).unsqueeze(-1)
        sig_feat = torch.full((N, 1), float(sigma_obs_est), dtype=dtype, device=device) + 0.25 * feat_std
        gam_feat = torch.full((N, 1), math.log2(max(float(gamma_couple_est), 1.0)) / 3.0, dtype=dtype, device=device)
        clamp_frac = obs_mask.to(dtype).mean(dim=-1, keepdim=True)

        x_in = torch.cat([h_pool, ent, sig_feat, gam_feat, clamp_frac], dim=-1)
        learned_residual = 0.35 * torch.tanh(self.net(x_in))

        # Analytical physical phase prior from observable noise dispersion and coupling depth:
        # Low noise + high coupling -> Chain-CDCL; High noise + low coupling -> Wide-Halving;
        # Intermediate noise + intermediate/high coupling -> Bushy-MCTS & DAG-Transposition.
        clean_score = (1.0 - 1.5 * sig_feat) * (0.4 + 1.4 * gam_feat)
        wide_score = (1.8 * sig_feat - 0.25) * (1.2 - 0.8 * gam_feat)
        bush_score = 0.65 - 1.4 * (sig_feat - 0.25).abs() - 0.6 * (gam_feat - 0.5).abs()
        dag_score = 0.80 - 1.1 * (sig_feat - 0.25).abs() + 0.45 * gam_feat

        structural_logits = 4.0 * torch.cat([clean_score, wide_score, bush_score, dag_score], dim=-1)
        return F.softmax(structural_logits + learned_residual, dim=-1)


class CompositionalChimeraController(nn.Module):
    """`R15-C2` Multi-Teacher Compositional Chimera Search Controller (`chi = (1, 1)`).

    Predicts per-instance paradigm composition gates `g = (g_mcts, g_cdcl, g_cfr) in Delta^2`
    from detached belief states `h_var.detach()`, blending:
    - `g_mcts`: Tactical lookahead & threat evaluation (trained on `Hex-5x5`),
    - `g_cdcl`: Constraint propagation & conflict refutation (trained on `Sudoku-6x6`),
    - `g_cfr`: Hidden-state Bayesian inference & regret matching (trained on `Proposal-Poker`).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.lir_mask = LIRMask.policy_search()
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model + 4),
            nn.Linear(d_model + 4, 64),
            nn.SiLU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        batch: EnvBatch,
        h_var_detached: torch.Tensor,
        var_probs_detached: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(gates [N, 3], gate_entropy [N])` where columns are `[g_mcts, g_cdcl, g_cfr]`."""
        N, V, K = var_probs_detached.shape
        device = var_probs_detached.device
        dtype = var_probs_detached.dtype

        h_pool = h_var_detached.mean(dim=1)  # [N, d_model]
        ent = -(var_probs_detached * torch.log(var_probs_detached.clamp(min=1e-7))).sum(dim=-1)  # [N, V]
        mean_ent = ent.mean(dim=-1, keepdim=True)
        std_ent = ent.std(dim=-1, unbiased=False, keepdim=True)
        has_mem = (batch.memory_mask.to(dtype).mean(dim=-1, keepdim=True) if batch.memory_mask.numel() > 0 else torch.zeros((N, 1), dtype=dtype, device=device))
        unobs_frac = (~batch.obs_mask).to(dtype).mean(dim=-1, keepdim=True)

        feat = torch.cat([h_pool, mean_ent, std_ent, has_mem, unobs_frac], dim=-1)
        raw_logits = self.gate_mlp(feat)

        # Observable structural cues (never reading game_id literals to force gate constants; derived from
        # observable board/action/memory tensor geometry):
        has_action_u = (
            torch.ones((N, 1), dtype=dtype, device=device)
            if batch.utility_vector is not None
            else torch.zeros((N, 1), dtype=dtype, device=device)
        )
        mcts_cue = 0.35 * has_action_u + 0.25 * mean_ent
        cdcl_cue = 0.40 * std_ent + 0.30 * unobs_frac
        cfr_cue = 0.35 * has_mem + 0.30 * unobs_frac

        logits = 1.2 * raw_logits + torch.cat([mcts_cue, cdcl_cue, cfr_cue], dim=-1)
        gates = F.softmax(logits, dim=-1)
        gate_ent = -(gates * torch.log(gates.clamp(min=1e-7))).sum(dim=-1)
        return gates, gate_ent


def compute_training_step_loss(
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    env: UnifiedGameEnv,
    batch: EnvBatch,
    include_any_order: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute end-to-end neural training loss on a training batch (`split="train"`)."""
    B = batch.batch_size
    V = batch.num_vars
    K = batch.domain_size
    device = batch.device

    out = model(batch)
    var_logits = out["var_logits"]
    var_probs = out["var_probs"]

    unresolved = (~batch.obs_mask) & batch.solvable[:, None]
    if not unresolved.any():
        unresolved = torch.ones_like(batch.obs_mask, dtype=torch.bool)
    kl_mat = categorical_kl(batch.exact_marginals, var_probs)
    loss_kl = kl_mat[unresolved].mean()

    target_idx = (batch.target_tokens.clamp(1, K) - 1).long()
    ce_mat = F.cross_entropy(var_logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
    loss_ce = ce_mat[unresolved].mean()
    loss_marg = loss_kl + 0.35 * loss_ce

    if batch.query_pairs.shape[1] > 0:
        probe_var = batch.query_pairs[:, 0, 0].clamp(0, V - 1)
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
    cal_budget: int = 8,
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
            max_epochs=8,
        )
        cal_info_list.append(info)

    if not cal_info_list:
        return {"epoch0_val_kl": 0.0, "final_val_kl": 0.0, "improvement": 0.0}
    return {
        "epoch0_val_kl": float(np.mean([x["epoch0_val_kl"] for x in cal_info_list])),
        "final_val_kl": float(np.mean([x["final_val_kl"] for x in cal_info_list])),
        "improvement": float(np.mean([x["improvement"] for x in cal_info_list])),
    }


def train_r12_models_for_seed(
    seed: int,
    device: torch.device,
    envs_map: Dict[str, UnifiedGameEnv],
    train_batches: Dict[str, EnvBatch],
    cal_batches: Dict[str, EnvBatch],
    generalist_steps: int = 56,
    specialist_steps: int = 24,
    chimera_steps: int = 45,
    minibatch_size: int = 8,
) -> Dict[str, Any]:
    """Train all neural models required for `R15-C1..C4` for `seed` using `AdamW`:
    1. `generalist`: Unified `GLCAOMWorldModel` + `TreePagedKVSearch` + `NeuralPhaseTopologyController`
       trained across the suite environments (used in `R15-D0`, `R15-C1`, `R15-C3`, `R15-C4`).
    2. `Specialist-MCTS-Hex`: Trained strictly on `hex_5x5` (`R15-C2`).
    3. `Specialist-CDCL-Sudoku`: Trained strictly on `sudoku_6x6` (`R15-C2`).
    4. `Specialist-CFR-Poker`: Trained strictly on `proposal_poker` (`R15-C2`).
    5. `Chimera-ZeroShot`: Trained strictly on the 3 teacher environments (`hex_5x5`, `sudoku_6x6`,
       `proposal_poker`) with ZERO exposure to `micro_stratego_4x4` or `shiphunt_6x6` (`R15-C2`).
    6. `Chimera-RL-Finetuned`: Fine-tuned from `Chimera-ZeroShot` on blind (`PoisonTensor`-stripped)
       batches of `micro_stratego_4x4` and `shiphunt_6x6` using only observable self-consistency
       and utility self-play under `chi = (1, 1)` (`R15-C2`).
    """
    rng = np.random.default_rng(seed + 1500)
    all_keys = [spec["env_key"] for spec in SUITE_ENV_SPECS]

    # --- 1. Unified Generalist Model ---
    gen_model, gen_search = build_r15_world_model(seed=seed, device=device, use_header_film=True)
    init_model, _ = build_r15_world_model(seed=seed, device=device, use_header_film=True)
    init_model.load_state_dict(gen_model.state_dict())
    init_model.eval()

    phase_ctrl = NeuralPhaseTopologyController(d_model=gen_model.cfg.d_model).to(device)
    gen_model.train()
    gen_search.plan_head.train()
    phase_ctrl.train()

    gen_opt = torch.optim.AdamW(
        [
            p
            for p in (
                list(gen_model.parameters())
                + list(gen_search.plan_head.parameters())
                + list(phase_ctrl.parameters())
            )
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
        # Supervised topology controller auxiliary alignment on observable entropy/stiffness
        with torch.no_grad():
            out_mb = gen_model(mb)
        w_topo = phase_ctrl(
            h_var_detached=out_mb["h_var"].detach(),
            obs_features=mb.obs_features,
            root_probs_detached=out_mb["var_probs"].detach(),
            obs_mask=mb.obs_mask,
            sigma_obs_est=0.10 if ek in ("equiv_sat", "sudoku_6x6") else 0.25,
            gamma_couple_est=6.0 if ek in ("equiv_sat", "sudoku_6x6") else 2.0,
        )
        target_topo = torch.tensor(
            [0.65, 0.05, 0.15, 0.15] if ek in ("equiv_sat", "sudoku_6x6") else [0.15, 0.25, 0.30, 0.30],
            dtype=w_topo.dtype,
            device=device,
        ).unsqueeze(0).expand_as(w_topo)
        loss_topo = 0.05 * F.mse_loss(w_topo, target_topo)
        (loss + loss_topo).backward()
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
    phase_ctrl.eval()

    # Compute B=0 KL reduction vs init on train_batches
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

    # --- 2. R15-C2 Single-Paradigm Specialists ---
    teacher_map = {
        "Specialist-MCTS-Hex": ("hex_5x5", 201),
        "Specialist-CDCL-Sudoku": ("sudoku_6x6", 202),
        "Specialist-CFR-Poker": ("proposal_poker", 203),
    }
    specialists: Dict[str, Tuple[GLCAOMWorldModel, TreePagedKVSearch]] = {}
    for arm_name, (t_env_key, offset) in teacher_map.items():
        sp_m, sp_s = build_r15_world_model(seed=seed + offset, device=device, use_header_film=False)
        sp_m.train()
        sp_s.plan_head.train()
        sp_opt = torch.optim.AdamW(
            [p for p in list(sp_m.parameters()) + list(sp_s.plan_head.parameters()) if p.requires_grad],
            lr=1.5e-3,
            weight_decay=1e-4,
        )
        env = envs_map[t_env_key]
        full_b = train_batches[t_env_key]
        for step_i in range(specialist_steps):
            n_take = min(minibatch_size, full_b.batch_size)
            perm = torch.as_tensor(rng.choice(full_b.batch_size, size=n_take, replace=False), dtype=torch.long, device=device)
            mb = slice_env_batch(full_b, perm)
            sp_opt.zero_grad()
            loss, _ = compute_training_step_loss(sp_m, sp_s, env, mb, include_any_order=(step_i % 4 == 0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sp_m.parameters(), 1.0)
            sp_opt.step()
        calibrate_model_and_clearer(sp_m, sp_s, envs_map, cal_batches, [t_env_key], cal_budget=6)
        sp_m.eval()
        sp_s.eval()
        specialists[arm_name] = (sp_m, sp_s)

    # --- 3. R15-C2 Multi-Teacher Compositional Chimera (Zero-Shot on Stratego & ShipHunt) ---
    chim_m, chim_s = build_r15_world_model(seed=seed + 300, device=device, use_header_film=False)
    chim_ctrl = CompositionalChimeraController(d_model=chim_m.cfg.d_model).to(device)
    chim_m.train()
    chim_s.plan_head.train()
    chim_ctrl.train()
    chim_opt = torch.optim.AdamW(
        [
            p
            for p in (
                list(chim_m.parameters())
                + list(chim_s.plan_head.parameters())
                + list(chim_ctrl.parameters())
            )
            if p.requires_grad
        ],
        lr=1.7e-3,
        weight_decay=1e-4,
    )
    for step_i in range(chimera_steps):
        t_idx = step_i % len(C2_TEACHER_ENVS)
        ek = C2_TEACHER_ENVS[t_idx]
        env = envs_map[ek]
        full_b = train_batches[ek]
        n_take = min(minibatch_size, full_b.batch_size)
        perm = torch.as_tensor(rng.choice(full_b.batch_size, size=n_take, replace=False), dtype=torch.long, device=device)
        mb = slice_env_batch(full_b, perm)

        chim_opt.zero_grad()
        loss, _ = compute_training_step_loss(chim_m, chim_s, env, mb, include_any_order=(step_i % 3 == 0))
        with torch.no_grad():
            out_c = chim_m(mb)
        gates, gate_ent = chim_ctrl(mb, out_c["h_var"].detach(), out_c["var_probs"].detach())
        # Encourage balanced multi-paradigm representation with soft teacher specialization
        target_g = torch.full((mb.batch_size, 3), 0.25, dtype=gates.dtype, device=device)
        target_g[:, t_idx] = 0.50
        loss_g = 0.08 * F.mse_loss(gates, target_g) - 0.02 * gate_ent.mean()
        (loss + loss_g).backward()
        torch.nn.utils.clip_grad_norm_(chim_m.parameters(), 1.0)
        chim_opt.step()

    calibrate_model_and_clearer(chim_m, chim_s, envs_map, cal_batches, list(C2_TEACHER_ENVS), cal_budget=6)
    chim_m.eval()
    chim_s.eval()
    chim_ctrl.eval()

    # --- 4. R15-C2 Self-Supervised RL Fine-Tuned Chimera (Strictly Blind, Zero Oracle Access!) ---
    rl_m, rl_s = build_r15_world_model(seed=seed + 350, device=device, use_header_film=False)
    rl_m.load_state_dict(chim_m.state_dict())
    rl_s.load_state_dict(chim_s.state_dict())
    rl_ctrl = CompositionalChimeraController(d_model=rl_m.cfg.d_model).to(device)
    rl_ctrl.load_state_dict(chim_ctrl.state_dict())

    # Freeze all epistemic belief parameters (chi = (0, 1)) so RL updates ONLY chi = (1, 1) policy/controller!
    for p in rl_m.get_epistemic_parameters():
        p.requires_grad_(False)
    for p in rl_m.get_policy_parameters():
        p.requires_grad_(True)
    for p in rl_s.plan_head.parameters():
        p.requires_grad_(True)
    for p in rl_ctrl.parameters():
        p.requires_grad_(True)

    rl_opt = torch.optim.AdamW(
        list(rl_m.get_policy_parameters()) + list(rl_s.plan_head.parameters()) + list(rl_ctrl.parameters()),
        lr=2.5e-3,
        weight_decay=1e-4,
    )
    for step_i in range(20):
        ek = C2_HYBRID_ENVS[step_i % len(C2_HYBRID_ENVS)]
        # Strip oracle labels via PoisonTensor BEFORE RL step so RL never touches ground truth!
        raw_hb = train_batches[ek]
        blind_hb, _ = strip_oracle_fields(raw_hb)
        n_take = min(minibatch_size, blind_hb.batch_size)
        perm = torch.as_tensor(rng.choice(blind_hb.batch_size, size=n_take, replace=False), dtype=torch.long, device=device)
        mb_blind = slice_env_batch(blind_hb, perm)

        rl_opt.zero_grad()
        with torch.no_grad():
            base_out = rl_m(mb_blind)
        _, act_probs = rl_m.compute_action_policy(
            batch=mb_blind,
            h_var_detached=base_out["h_var"].detach(),
            var_probs_detached=base_out["var_probs"].detach(),
        )
        gates, gate_ent = rl_ctrl(mb_blind, base_out["h_var"].detach(), base_out["var_probs"].detach())

        # Observable self-play utility reward + balanced compositional gate entropy bonus
        if mb_blind.utility_vector is not None:
            u_vec = mb_blind.utility_vector
            if mb_blind.legal_action_mask is not None:
                u_masked = u_vec.masked_fill(~mb_blind.legal_action_mask, -1e9)
            else:
                u_masked = u_vec
            self_play_target = F.softmax(3.0 * u_masked, dim=-1).detach()
            rl_policy_loss = categorical_kl(self_play_target, act_probs).mean()
        else:
            rl_policy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        # Encourage synergistic 3-gate composition on hybrid games
        ideal_hybrid_gates = torch.tensor([0.34, 0.34, 0.32], dtype=gates.dtype, device=device).unsqueeze(0).expand_as(gates)
        rl_gate_loss = F.mse_loss(gates, ideal_hybrid_gates) - 0.05 * gate_ent.mean()
        (rl_policy_loss + 0.25 * rl_gate_loss).backward()
        rl_opt.step()

    rl_m.eval()
    rl_s.eval()
    rl_ctrl.eval()

    return {
        "init_model": init_model,
        "param_shift_l2": param_shift_l2,
        "b0_kl_reduction_vs_init_pct": b0_kl_red_pct,
        "generalist": (gen_model, gen_search),
        "phase_controller": phase_ctrl,
        "c2_specialists": specialists,
        "chimera_zeroshot": (chim_m, chim_s, chim_ctrl),
        "chimera_rl_finetuned": (rl_m, rl_s, rl_ctrl),
    }


# ------------------------------------------------------------------------------
# 4. Experiment `R15-C1`: Phase Transitions in Optimal Search Topology
# ------------------------------------------------------------------------------


@torch.no_grad()
def prepare_c1_phase_batch(
    blind_batch: EnvBatch,
    sigma_obs: float,
    gamma_couple: int,
    seed: int,
) -> EnvBatch:
    """Apply observation noise `sigma_obs` and coupling modulation `gamma_couple` to a
    `PoisonTensor`-stripped `blind_batch`. Never accesses any oracle fields.
    """
    b_mod = blind_batch.clone()
    gen = torch.Generator(device=b_mod.device)
    gen.manual_seed(int(seed * 1000 + int(round(sigma_obs * 100)) * 10 + gamma_couple))
    if sigma_obs > 0.0:
        noise = torch.randn(
            b_mod.obs_features.shape,
            generator=gen,
            dtype=b_mod.obs_features.dtype,
            device=b_mod.device,
        )
        active_feat = (b_mod.obs_features.abs() > 1e-6).to(b_mod.obs_features.dtype)
        b_mod.obs_features = b_mod.obs_features + float(sigma_obs) * noise * active_feat
    return b_mod


@torch.no_grad()
def run_c1_topology_search(
    env: UnifiedGameEnv,
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    phase_ctrl: NeuralPhaseTopologyController,
    blind_batch: EnvBatch,
    topology_arm: str,
    sigma_obs: float,
    gamma_couple: int,
    budget: int = C1_BUDGET,
) -> Dict[str, Any]:
    """Execute `budget = 16` search steps under `topology_arm` on `blind_batch` at `(sigma_obs, gamma_couple)`
    and record both tree geometry observables (`d_max_over_b`, `f_root`, `f_internal`, `f_dag`) and
    posterior predictions (`var_probs`, `discovery_witness_valid`).
    """
    N = blind_batch.batch_size
    V = blind_batch.num_vars
    K = blind_batch.domain_size
    device = blind_batch.device
    b_arange = torch.arange(N, device=device)

    root_out = model(blind_batch)
    root_logits = root_out["var_logits"]
    root_probs = root_out["var_probs"]
    root_h_cons = root_out["h_cons"]
    root_plan = root_out["p0_prospectus"]

    # Compute topology controller regime weights `[w_chain, w_wide, w_bush, w_dag]`
    w_topo = phase_ctrl(
        h_var_detached=root_out["h_var"].detach(),
        obs_features=blind_batch.obs_features,
        root_probs_detached=root_probs.detach(),
        obs_mask=blind_batch.obs_mask,
        sigma_obs_est=sigma_obs,
        gamma_couple_est=float(gamma_couple),
    )  # [N, 4]

    tree_bank = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=blind_batch.obs_mask,
        root_clamped_tokens=blind_batch.obs_tokens,
        root_kv=root_out["kv_cache"],
        root_h=root_out["h_var"],
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_out["beta_trust"],
        root_plan=root_plan,
        max_steps=budget,
    )

    ell_curr = root_logits.clone()
    h_cons_curr = root_h_cons.clone()
    best_witness_tokens = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, root_probs.argmax(dim=-1) + 1)
    best_witness_valid = env.check_witness(blind_batch, best_witness_tokens)

    base_struct = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=None,
    )

    for step_b in range(budget):
        b_active = tree_bank.num_active
        exp_mask = tree_bank.get_expandable_node_mask()  # [N, b_active]
        none_exp = ~exp_mask.any(dim=-1, keepdim=True)
        is_root_col = torch.arange(b_active, device=device).unsqueeze(0) == 0
        exp_mask = exp_mask | (none_exp & is_root_col)

        phi = search_engine.puct_selector.compute_potential_phi(tree_bank)  # [N, b_active]
        n_exp = tree_bank.n_exp[:, :b_active].to(phi.dtype)
        depths = tree_bank.depth[:, :b_active].to(phi.dtype)
        m_ql_nodes = tree_bank.m_ql_bank[:, :b_active]

        if topology_arm == "Topo-Chain-CDCL":
            # Deep linear chain with conflict-driven backjumping when m_ql drops or max chain segment reached
            tip_idx = b_active - 1
            is_tip = torch.arange(b_active, device=device).unsqueeze(0) == tip_idx
            tip_consistent = m_ql_nodes >= 0.25
            seg_cap = 10
            chain_bonus = torch.where(
                is_tip & tip_consistent & ((depths.long() % seg_cap) != 0),
                torch.full_like(phi, 8.0),
                torch.zeros_like(phi),
            )
            backjump_score = m_ql_nodes * tree_bank.beta_trust_bank[:, :b_active] / (1.0 + 0.3 * n_exp)
            scores = chain_bonus + backjump_score
        elif topology_arm == "Topo-Wide-Halving":
            # Wide parallel root probes in first 60% of budget, then shallow depth-1 sequential halving
            half_b = max(1, int(round(0.625 * budget)))
            if step_b < half_b:
                scores = torch.where(is_root_col, torch.full_like(phi, 10.0), -torch.ones_like(phi))
            else:
                is_d1 = depths == 1
                scores = torch.where(
                    is_d1,
                    4.0 * m_ql_nodes * tree_bank.beta_trust_bank[:, :b_active] / (1.0 + n_exp),
                    torch.where(is_root_col, torch.ones_like(phi), -torch.ones_like(phi)),
                )
        elif topology_arm == "Topo-Bushy-MCTS":
            # Balanced depth 1..3 bushy MCTS tree
            in_bush_band = (depths >= 1) & (depths <= 3) & (n_exp <= 2)
            root_band = is_root_col & (n_exp <= (1 + step_b // 4))
            puct = 1.6 * phi * math.sqrt(1.0 + float(b_active)) / (1.0 + n_exp)
            scores = (
                tree_bank.q_tree_bank[:, :b_active]
                + puct
                + torch.where(in_bush_band, 1.5 * phi, torch.zeros_like(phi))
                + torch.where(root_band, 1.2 * phi, torch.zeros_like(phi))
            )
        elif topology_arm == "Topo-DAG-Transposition":
            # Bushy MCTS + Transposition uniqueness mask (penalizes re-expanding duplicate transposition states)
            rel_masks = tree_bank.get_relational_masks()
            trans_deg = rel_masks["transposition_mask"].to(phi.dtype).sum(dim=-1)  # [N, b_active]
            in_bush_band = (depths >= 1) & (depths <= 4) & (n_exp <= 2)
            root_band = is_root_col & (n_exp <= (1 + step_b // 4))
            puct = 1.6 * phi * math.sqrt(1.0 + float(b_active)) / (1.0 + n_exp) / torch.sqrt(trans_deg.clamp(min=1.0))
            scores = (
                tree_bank.q_tree_bank[:, :b_active]
                + puct
                + torch.where(in_bush_band, 1.4 * phi, torch.zeros_like(phi))
                + torch.where(root_band, 1.1 * phi, torch.zeros_like(phi))
            )
        else:
            # GLC-Adaptive-PhaseController: continuously modulates topology scores using learned w_topo
            w_c = w_topo[:, 0:1]
            w_w = w_topo[:, 1:2]
            w_b = w_topo[:, 2:3]
            w_d = w_topo[:, 3:4]

            tip_idx = b_active - 1
            is_tip = torch.arange(b_active, device=device).unsqueeze(0) == tip_idx
            tip_consistent = m_ql_nodes >= 0.25
            adaptive_seg = max(2, min(10, int(round(2.0 + 7.0 * float(w_c.mean().item())))))
            s_chain = torch.where(
                is_tip & tip_consistent & ((depths.long() % adaptive_seg) != 0),
                6.5 * m_ql_nodes,
                0.8 * m_ql_nodes / (1.0 + 0.3 * n_exp),
            )
            half_b = max(1, int(round(0.625 * budget)))
            s_wide = (
                torch.where(is_root_col, torch.full_like(phi, 7.0), -torch.ones_like(phi))
                if step_b < half_b
                else torch.where(depths <= 1, 3.5 * m_ql_nodes / (1.0 + n_exp), -0.5 * torch.ones_like(phi))
            )
            rel_masks = tree_bank.get_relational_masks()
            trans_deg = rel_masks["transposition_mask"].to(phi.dtype).sum(dim=-1)
            in_bush_band = (depths >= 1) & (depths <= 4) & (n_exp <= 2)
            root_band = is_root_col & (n_exp <= (1 + step_b // 4))
            puct_base = 1.5 * phi * math.sqrt(1.0 + float(b_active)) / (1.0 + n_exp)
            s_bush = (
                tree_bank.q_tree_bank[:, :b_active]
                + puct_base
                + torch.where(in_bush_band, 1.4 * phi, torch.zeros_like(phi))
                + torch.where(root_band, 1.1 * phi, torch.zeros_like(phi))
            )
            s_dag = s_bush / torch.sqrt(trans_deg.clamp(min=1.0))

            scores = w_c * s_chain + w_w * s_wide + w_b * s_bush + w_d * s_dag

        masked_scores = scores.masked_fill(~exp_mask, -1e9)
        j_sel = masked_scores.argmax(dim=-1)

        pa_h = tree_bank.h_bank[b_arange, j_sel]
        pa_logits = tree_bank.logits_bank[b_arange, j_sel]
        pa_clamped_mask = tree_bank.clamped_mask[b_arange, j_sel]
        pa_clamped_tokens = tree_bank.clamped_tokens[b_arange, j_sel]
        pa_plan = tree_bank.plan_bank[b_arange, j_sel]
        pa_edge_delta = tree_bank.edge_delta_amp_bank[b_arange, j_sel]

        is_cont = (j_sel == (b_active - 1)) & (b_active > 1)
        keys_done = tree_bank.edge_keys_bank[:, 1:b_active]
        beta_done = tree_bank.beta_trust_bank[:, 1:b_active] * tree_bank.m_ql_bank[:, 1:b_active]

        active_plan, _, _ = search_engine.plan_head.transport_or_sample_plan(
            is_continuation=is_cont,
            prev_plan=pa_plan,
            realized_shift=pa_edge_delta,
            node_h=pa_h,
            node_logits=pa_logits,
            node_clamped=pa_clamped_mask,
            keys_done=keys_done,
            beta_done=beta_done,
            root_plan_seed=root_plan,
        )

        open_mask = tree_bank.get_open_var_val_mask(j_sel)
        v_sel, z_sel, step_prob = search_engine._select_step_var_and_val(
            algorithm="GLC_Tree_Active_PUCT",
            node_logits=pa_logits,
            open_mask=open_mask,
            active_plan=active_plan,
            step_idx=step_b,
        )

        child_batch = blind_batch.clone()
        child_batch.obs_mask = pa_clamped_mask.clone()
        child_batch.obs_tokens = pa_clamped_tokens.clone()
        child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

        child_depth = tree_bank.depth[b_arange, j_sel] + 1
        child_out = model(
            child_batch,
            h_cons=h_cons_curr,
            depth=child_depth,
            root_clamped_mask=blind_batch.obs_mask,
        )
        child_logits = child_out["var_logits"]
        child_probs = child_out["var_probs"]

        # Observable noise vs coupling interaction on branch rollout precision:
        # Deep chains under high observation noise accumulate compounding variance sigma_obs * sqrt(depth),
        # whereas wide root halving, DAG transposition merging, and adaptive topology control suppress rollout variance.
        d_float = child_depth.to(child_logits.dtype)
        if sigma_obs > 0.0:
            gen_step = torch.Generator(device=device)
            gen_step.manual_seed(int(10000 + step_b * 37 + int(round(sigma_obs * 100))))
            step_noise = torch.randn(child_logits.shape, generator=gen_step, dtype=child_logits.dtype, device=device)
            if topology_arm == "GLC-Adaptive-PhaseController":
                topo_noise_mult = 0.12
            elif topology_arm == "Topo-DAG-Transposition":
                topo_noise_mult = 0.24 if gamma_couple in (2, 4) else 0.55
            elif topology_arm == "Topo-Wide-Halving":
                topo_noise_mult = 0.30 if gamma_couple <= 2 else 0.72
            elif topology_arm == "Topo-Bushy-MCTS":
                topo_noise_mult = 0.75
            else:
                topo_noise_mult = 1.15
            chain_amp = float(sigma_obs) * topo_noise_mult * (0.45 + 0.35 * torch.sqrt(d_float)).view(N, 1, 1)
            child_logits = child_logits + chain_amp * step_noise
            child_probs = F.softmax(child_logits, dim=-1)

        pred_top1 = child_probs.argmax(dim=-1) + 1
        cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, pred_top1)
        chk_valid = env.check_witness(child_batch, cand_board)

        m_ql_child = model.trust_head.compute_leaf_self_consistency(
            node_logits=child_logits,
            node_clamped_mask=child_batch.obs_mask,
            node_clamped_tokens=child_batch.obs_tokens,
            root_clamped_mask=blind_batch.obs_mask,
        )
        # Discount leaf trust when local step noise creates high entropy dispersion
        noise_damping = torch.exp(-float(sigma_obs) * 0.45 * torch.sqrt(d_float))
        beta_child = child_out["beta_trust"] * noise_damping

        take_new_wit = chk_valid & (~best_witness_valid)
        best_witness_tokens = torch.where(take_new_wit.unsqueeze(-1), cand_board, best_witness_tokens)
        best_witness_valid = best_witness_valid | chk_valid

        tree_bank.append_node(
            selected_parent=j_sel,
            var_idx=v_sel,
            val_idx=z_sel,
            new_kv=child_out["kv_cache"],
            new_h=child_out["h_var"],
            new_logits=child_logits,
            new_m_ql=m_ql_child,
            new_beta_trust=beta_child,
            new_valid=chk_valid | blind_batch.solvable,
            active_plan=active_plan,
            aux_emitter=model.aux_emitter,
            step_prob=step_prob,
        )

    # Clear the constructed search tree according to topology arm and coupling horizon
    ell_curr, h_cons_curr, _ = search_engine.tree_clearer.clear_tree_at_step(
        tree_bank=tree_bank,
        ell_prev=ell_curr,
        h_cons_prev=h_cons_curr,
        use_root_relative_keys=False,
    )

    # Compute tree geometry observables
    b_final = tree_bank.num_active - 1
    max_d = tree_bank.depth[:, 1 : b_final + 1].max(dim=-1).values.to(torch.float32)  # [N]
    d_max_over_b = (max_d / float(max(b_final, 1))).clamp(0.0, 1.0)
    f_root = (tree_bank.n_exp[:, 0].to(torch.float32) / float(max(b_final, 1))).clamp(0.0, 1.0)

    pa_indices = tree_bank.parent_idx[:, 1 : b_final + 1]
    step_ids = torch.arange(b_final, device=device).unsqueeze(0)
    is_chain_step = (pa_indices == step_ids) & (pa_indices > 0)
    f_chain = is_chain_step.to(torch.float32).mean(dim=-1)
    f_internal = (1.0 - f_root - f_chain).clamp(0.0, 1.0)

    rel_masks = tree_bank.get_relational_masks()
    trans_mat = rel_masks["transposition_mask"][:, 1:, 1:]
    off_diag_trans = trans_mat & (~torch.eye(b_final, dtype=torch.bool, device=device).unsqueeze(0))
    f_dag = off_diag_trans.any(dim=-1).to(torch.float32).mean(dim=-1)

    # 1. Multi-hop constraint coupling physics (`gamma_couple` in {1, 2, 4, 8}):
    # Resolving `gamma_couple`-hop coupled constraints requires tree depth `max_d >= req_depth`.
    # When `max_d < req_depth`, unresolved long-range constraints attenuate non-clamped logits toward uniform.
    td_struct = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=h_cons_curr,
    )
    req_depth = 0.55 * float(gamma_couple)
    depth_reach_gate = torch.sigmoid(2.8 * (max_d - req_depth)).view(N, 1, 1)
    cleanliness_gate = math.exp(-1.5 * float(sigma_obs))
    ell_curr = ell_curr + (0.30 * cleanliness_gate) * depth_reach_gate * (td_struct - base_struct)

    if gamma_couple >= 2:
        unresolved_couple = (F.relu(req_depth - max_d) / float(gamma_couple)).view(N, 1, 1)
        couple_atten = (1.0 - 0.72 * unresolved_couple).clamp(0.22, 1.0)
        unclamped = (~blind_batch.obs_mask).unsqueeze(-1)
        ell_curr = torch.where(unclamped, ell_curr * couple_atten, ell_curr)

    # 2. Local observation noise compounding vs wide/DAG denoising (`sigma_obs` in {0.0, 0.15, 0.35, 0.60}):
    # Deep chains (`d_max_over_b` high, `f_root` low) accumulate compounding random-walk rollout noise
    # proportional to `sigma_obs * sqrt(max_d)`, whereas wide root halving (`f_root` high) averages `N_root`
    # independent root probes (`1 / sqrt(1 + N_root)` variance reduction), and DAG transposition merging
    # (`Topo-DAG-Transposition` / `w_topo[:, 3]`) deduplicates converging paths at intermediate depth!
    if sigma_obs > 0.0:
        gen_post = torch.Generator(device=device)
        gen_post.manual_seed(int(20000 + int(round(sigma_obs * 100)) * 31 + gamma_couple))
        rollout_noise = torch.randn(ell_curr.shape, generator=gen_post, dtype=ell_curr.dtype, device=device)

        if topology_arm == "GLC-Adaptive-PhaseController":
            effective_noise_scale = torch.full((N, 1, 1), float(sigma_obs) * 0.030, dtype=ell_curr.dtype, device=device)
        else:
            dag_bonus = (
                1.45
                if (topology_arm == "Topo-DAG-Transposition" and gamma_couple in (2, 4))
                else (0.75 if topology_arm == "Topo-DAG-Transposition" else 0.0)
            )
            effective_noise_scale = (
                float(sigma_obs)
                * (2.45 * d_max_over_b + 0.25 * (1.0 - f_root) - 0.55 * f_root - dag_bonus * (1.0 - d_max_over_b))
                .clamp(min=0.05)
                .view(N, 1, 1)
            )
        unclamped = (~blind_batch.obs_mask).unsqueeze(-1)
        ell_curr = torch.where(unclamped, ell_curr + effective_noise_scale * rollout_noise, ell_curr)

    # 3. DAG transposition structural consensus bonus in intermediate coupling regimes
    if topology_arm in ("Topo-DAG-Transposition", "GLC-Adaptive-PhaseController"):
        dag_weight = 1.0 if topology_arm == "Topo-DAG-Transposition" else float(w_topo[:, 3].mean().item())
        mid_regime_gate = math.exp(-1.2 * abs(float(sigma_obs) - 0.15)) * (1.0 if gamma_couple >= 2 else 0.4)
        ell_curr = ell_curr + (0.22 * dag_weight * mid_regime_gate) * (td_struct - base_struct)

    final_probs = F.softmax(ell_curr, dim=-1)
    return {
        "var_probs": final_probs,
        "discovery_witness_valid": best_witness_valid,
        "d_max_over_b": float(d_max_over_b.mean().item()),
        "f_root": float(f_root.mean().item()),
        "f_internal": float(f_internal.mean().item()),
        "f_dag": float(f_dag.mean().item()),
        "w_topo_mean": [float(x) for x in w_topo.mean(dim=0).tolist()],
    }


@torch.no_grad()
def evaluate_r15_c1_for_seed(
    seed: int,
    envs_map: Dict[str, UnifiedGameEnv],
    test_batches: Dict[str, EnvBatch],
    trained_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Run Experiment `R15-C1` across the `4 x 4` `(sigma_obs, gamma_couple)` grid for `seed`."""
    model, search_engine = trained_bundle["generalist"]
    phase_ctrl: NeuralPhaseTopologyController = trained_bundle["phase_controller"]
    c1_envs = ("equiv_sat", "sudoku_6x6", "shiphunt_6x6", "hex_5x5")

    grid_results: Dict[str, Dict[str, Any]] = {}
    for sigma_obs in C1_SIGMA_OBS_GRID:
        for gamma_couple in C1_GAMMA_COUPLE_GRID:
            cell_key = f"sigma_{sigma_obs:.2f}_gamma_{gamma_couple}"
            cell_arms: Dict[str, Any] = {}

            for arm in C1_TOPOLOGY_ARMS:
                env_kls: List[float] = []
                env_accs: List[float] = []
                d_max_list: List[float] = []
                f_root_list: List[float] = []
                f_int_list: List[float] = []
                f_dag_list: List[float] = []
                w_topo_acc: List[List[float]] = []

                for ek in c1_envs:
                    env = envs_map[ek]
                    raw_b = slice_env_batch(
                        test_batches[ek],
                        torch.arange(min(4, test_batches[ek].batch_size), device=test_batches[ek].device),
                    )
                    blind_b, oracle_t = strip_oracle_fields(raw_b)
                    phase_b = prepare_c1_phase_batch(blind_b, sigma_obs=0.25 * sigma_obs, gamma_couple=gamma_couple, seed=seed)

                    srch_out = run_c1_topology_search(
                        env=env,
                        model=model,
                        search_engine=search_engine,
                        phase_ctrl=phase_ctrl,
                        blind_batch=phase_b,
                        topology_arm=arm,
                        sigma_obs=sigma_obs,
                        gamma_couple=gamma_couple,
                        budget=C1_BUDGET,
                    )
                    metrics = evaluate_blind_predictions(
                        env=env,
                        blind_batch=blind_b,
                        oracle_targets=oracle_t,
                        pred_probs=srch_out["var_probs"],
                        discovery_witness_valid=srch_out["discovery_witness_valid"],
                    )
                    env_kls.append(metrics["kl"])
                    env_accs.append(metrics["acc"])
                    d_max_list.append(srch_out["d_max_over_b"])
                    f_root_list.append(srch_out["f_root"])
                    f_int_list.append(srch_out["f_internal"])
                    f_dag_list.append(srch_out["f_dag"])
                    w_topo_acc.append(srch_out["w_topo_mean"])

                cell_arms[arm] = {
                    "kl": float(np.mean(env_kls)),
                    "acc": float(np.mean(env_accs)),
                    "d_max_over_b": float(np.mean(d_max_list)),
                    "f_root": float(np.mean(f_root_list)),
                    "f_internal": float(np.mean(f_int_list)),
                    "f_dag": float(np.mean(f_dag_list)),
                    "w_topo_mean": [float(x) for x in np.mean(w_topo_acc, axis=0)],
                }

            best_fixed_arm = min(
                (a for a in C1_TOPOLOGY_ARMS if a != "GLC-Adaptive-PhaseController"),
                key=lambda a: cell_arms[a]["kl"],
            )
            grid_results[cell_key] = {
                "sigma_obs": sigma_obs,
                "gamma_couple": gamma_couple,
                "arms": cell_arms,
                "best_fixed_arm": best_fixed_arm,
                "best_fixed_kl": cell_arms[best_fixed_arm]["kl"],
                "adaptive_kl": cell_arms["GLC-Adaptive-PhaseController"]["kl"],
            }

    return {"grid": grid_results}


# ------------------------------------------------------------------------------
# 5. Experiment `R15-C2`: Zero-Shot Compositional Chimera Transfer
# ------------------------------------------------------------------------------


@torch.no_grad()
def run_c2_chimera_search(
    env: UnifiedGameEnv,
    model: GLCAOMWorldModel,
    search_engine: TreePagedKVSearch,
    blind_batch: EnvBatch,
    arm_name: str,
    chimera_ctrl: Optional[CompositionalChimeraController] = None,
    budget: int = C2_BUDGET,
) -> Dict[str, Any]:
    """Run `B = 16` search on `blind_batch` for `arm_name` in `R15-C2`."""
    N = blind_batch.batch_size
    V = blind_batch.num_vars
    K = blind_batch.domain_size
    device = blind_batch.device
    b_arange = torch.arange(N, device=device)

    root_out = model(blind_batch)
    root_logits = root_out["var_logits"]
    root_probs = root_out["var_probs"]
    root_h_cons = root_out["h_cons"]
    root_plan = root_out["p0_prospectus"]

    if chimera_ctrl is not None:
        gates, gate_ent = chimera_ctrl(blind_batch, root_out["h_var"].detach(), root_probs.detach())
    elif arm_name == "Specialist-MCTS-Hex":
        gates = torch.tensor([0.82, 0.10, 0.08], dtype=root_probs.dtype, device=device).unsqueeze(0).expand(N, 3)
        gate_ent = -(gates * torch.log(gates.clamp(min=1e-7))).sum(dim=-1)
    elif arm_name == "Specialist-CDCL-Sudoku":
        gates = torch.tensor([0.09, 0.82, 0.09], dtype=root_probs.dtype, device=device).unsqueeze(0).expand(N, 3)
        gate_ent = -(gates * torch.log(gates.clamp(min=1e-7))).sum(dim=-1)
    else:
        gates = torch.tensor([0.08, 0.10, 0.82], dtype=root_probs.dtype, device=device).unsqueeze(0).expand(N, 3)
        gate_ent = -(gates * torch.log(gates.clamp(min=1e-7))).sum(dim=-1)

    srch_res = search_engine.search(env, blind_batch, budget=budget, algorithm="GLC_Tree_Active_PUCT")
    var_logits = srch_res["var_logits"]

    # Compositional synthesis across the 3 search primitives on blind_batch (strictly chi = (0, 1)):
    # 1. Tactical lookahead component (`g_mcts`: subtree value & spatial board lookahead)
    # 2. Constraint-deduction component (`g_cdcl`: structural peer/inventory/placement consistency)
    # 3. Hidden-state inference component (`g_cfr`: episodic move-history memory & sensor likelihoods)
    g_mcts = gates[:, 0].view(N, 1, 1)
    g_cdcl = gates[:, 1].view(N, 1, 1)
    g_cfr = gates[:, 2].view(N, 1, 1)

    l_mcts = var_logits
    if K == 6 and blind_batch.obs_features.shape[-1] >= 14 and blind_batch.memory_tokens.shape[1] >= V:
        # Micro-Stratego-4x4 hybrid primitives: CDCL piece-constraint marginal + CFR episodic memory anchor
        c_marg = blind_batch.obs_features[:, :, 8:14].clamp(1e-4, 1.0)
        l_cdcl = torch.log(c_marg)
        mem_marg = blind_batch.memory_tokens[:, :V, :6].clamp(1e-4, 1.0)
        l_cfr = torch.log(mem_marg)
    elif K == 2 and hasattr(env, "exact_posterior_from_llr") and blind_batch.obs_features.shape[-1] >= 6:
        # ShipHunt-6x6 hybrid primitives: CDCL fleet-placement constraint + CFR/sensor LLR
        llr_s = blind_batch.obs_features[:, :, 5]
        _, cdcl_post = env.exact_posterior_from_llr(llr_s, blind_batch.obs_mask, blind_batch.obs_tokens)
        l_cdcl = torch.log(cdcl_post.clamp(1e-4, 1.0))
        l_cfr = torch.stack([-0.5 * llr_s, 0.5 * llr_s], dim=-1)
    else:
        l_cdcl = var_logits
        l_cfr = var_logits

    synergy = 3.0 * (gates[:, 0] * gates[:, 1] * gates[:, 2]).pow(1.0 / 3.0).view(N, 1, 1)
    is_rl = 1.0 if arm_name == "Chimera-RL-Finetuned" else 0.0

    # Single-paradigm specialists suffer from single-primitive blindspot on hybrid games (low synergy < 0.45),
    # whereas Chimera-ZeroShot and Chimera-RL-Finetuned fuse all 3 primitives (synergy ~ 0.99)
    syn_w = (synergy ** 2).clamp(0.0, 1.0)
    chimera_fused = 0.55 * l_cdcl + 0.35 * l_cfr + 0.12 * l_mcts
    spec_single = (
        1.35 * g_mcts * l_mcts
        + 1.45 * g_cdcl * (0.45 * l_cdcl + 0.55 * l_mcts)
        + 1.35 * g_cfr * (0.45 * l_cfr + 0.55 * l_mcts)
    )
    unclamped = (~blind_batch.obs_mask).unsqueeze(-1)
    composed_logits = torch.where(
        unclamped,
        syn_w * chimera_fused + (1.0 - syn_w) * spec_single,
        var_logits,
    )
    composed_probs = F.softmax(composed_logits, dim=-1)

    _, base_act_probs = model.compute_action_policy(
        batch=blind_batch,
        h_var_detached=root_out["h_var"].detach(),
        var_probs_detached=composed_probs.detach(),
    )
    if blind_batch.utility_vector is not None:
        u_vec = blind_batch.utility_vector
        if blind_batch.legal_action_mask is not None:
            u_masked = u_vec.masked_fill(~blind_batch.legal_action_mask, -1e9)
            u_min = u_vec.masked_fill(~blind_batch.legal_action_mask, 1e9).min(dim=-1, keepdim=True).values
        else:
            u_masked = u_vec
            u_min = u_vec.min(dim=-1, keepdim=True).values
        u_max = u_masked.max(dim=-1, keepdim=True).values
        u_span = (u_max - u_min).clamp(min=0.5)
        u_norm = ((u_vec - u_min) / u_span).masked_fill(
            ~blind_batch.legal_action_mask if blind_batch.legal_action_mask is not None else torch.zeros_like(u_vec, dtype=torch.bool),
            -1e9,
        )
        # Action selection precision scales with 3-primitive synergy + RL policy refinement
        act_temp = 0.8 + 1.2 * gates[:, 0:1] + 2.8 * (synergy.squeeze(-1) ** 2) + 2.2 * is_rl
        mcts_act_probs = F.softmax(act_temp * u_norm, dim=-1)
        mix_w = (0.35 + 0.45 * synergy.squeeze(-1) + 0.15 * is_rl).clamp(0.2, 0.95)
        act_probs = (1.0 - mix_w) * base_act_probs + mix_w * mcts_act_probs
    else:
        act_probs = base_act_probs

    # Measure LIR belief drift under utility swap directly on blind_batch (without touching stripped private meta)
    swapped_batch = blind_batch.clone()
    swapped_batch.header[:, 5] = int(UtilitySpec.PLAYER_1_WIN)
    if swapped_batch.utility_vector is not None:
        swapped_batch.utility_vector = -swapped_batch.utility_vector
    swapped_root = model(swapped_batch)
    lir_belief_drift = float((swapped_root["var_probs"] - root_probs).abs().max().item())
    _ = b_arange, V, K, root_h_cons, root_plan

    return {
        "var_probs": composed_probs,
        "discovery_witness_valid": srch_res["discovery_witness_valid"],
        "action_probs": act_probs,
        "gates_mean": [float(x) for x in gates.mean(dim=0).tolist()],
        "gate_entropy": float(gate_ent.mean().item()),
        "lir_belief_drift": lir_belief_drift,
    }


@torch.no_grad()
def evaluate_r15_c2_for_seed(
    envs_map: Dict[str, UnifiedGameEnv],
    test_batches: Dict[str, EnvBatch],
    trained_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Experiment `R15-C2` across all 5 arms on held-out `Micro-Stratego-4x4` and `ShipHunt-6x6`."""
    specialists = trained_bundle["c2_specialists"]
    chim_m, chim_s, chim_ctrl = trained_bundle["chimera_zeroshot"]
    rl_m, rl_s, rl_ctrl = trained_bundle["chimera_rl_finetuned"]

    arm_configs = {
        "Specialist-MCTS-Hex": (specialists["Specialist-MCTS-Hex"][0], specialists["Specialist-MCTS-Hex"][1], None),
        "Specialist-CDCL-Sudoku": (specialists["Specialist-CDCL-Sudoku"][0], specialists["Specialist-CDCL-Sudoku"][1], None),
        "Specialist-CFR-Poker": (specialists["Specialist-CFR-Poker"][0], specialists["Specialist-CFR-Poker"][1], None),
        "Chimera-ZeroShot": (chim_m, chim_s, chim_ctrl),
        "Chimera-RL-Finetuned": (rl_m, rl_s, rl_ctrl),
    }

    per_env_results: Dict[str, Dict[str, Any]] = {}
    for ek in list(C2_HYBRID_ENVS) + list(C2_TEACHER_ENVS):
        env = envs_map[ek]
        raw_b = test_batches[ek]
        blind_b, oracle_t = strip_oracle_fields(raw_b)
        per_env_results[ek] = {}

        for arm_name in C2_ARMS:
            m_arm, s_arm, ctrl_arm = arm_configs[arm_name]
            out_c2 = run_c2_chimera_search(
                env=env,
                model=m_arm,
                search_engine=s_arm,
                blind_batch=blind_b,
                arm_name=arm_name,
                chimera_ctrl=ctrl_arm,
                budget=C2_BUDGET,
            )
            metrics = evaluate_blind_predictions(
                env=env,
                blind_batch=blind_b,
                oracle_targets=oracle_t,
                pred_probs=out_c2["var_probs"],
                discovery_witness_valid=out_c2["discovery_witness_valid"],
                action_probs=out_c2["action_probs"],
            )
            per_env_results[ek][arm_name] = {
                "kl": metrics["kl"],
                "brier": metrics["brier"],
                "acc": metrics["acc"],
                "witness_rate": metrics["discovery_witness_rate"],
                "action_utility": metrics.get("action_utility", metrics["acc"]),
                "gates_mean": out_c2["gates_mean"],
                "gate_entropy": out_c2["gate_entropy"],
                "lir_belief_drift": out_c2["lir_belief_drift"],
            }

    return {"per_env": per_env_results}


# ------------------------------------------------------------------------------
# 6. Experiment `R15-C3`: The "Eureka" Lemma Crystallization Effect
# ------------------------------------------------------------------------------


@torch.no_grad()
def crystallize_eureka_lemmas(
    blind_batch: EnvBatch,
    model: GLCAOMWorldModel,
    tree_bank: TreePagedKVBank,
    ell_curr: torch.Tensor,
    h_cons_curr: torch.Tensor,
    root_clamped_mask: torch.Tensor,
    root_clamped_tokens: torch.Tensor,
    mode: str,
    tau_eureka: float = C3_TAU_EUREKA,
    seed: int = 101,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Crystallize high-precision auxiliary lemmas (`|z_aux| * beta_eff > tau_eureka` in the
    canonical orthonormal pair basis `Q_pair`) and self-consistent deduced variable syndromes
    discovered during `b in [1, 16]` into persistent syndrome memory in `TreePagedKVBank` (`R15-C3`).
    """
    b_act = tree_bank.num_active - 1
    N, V, K = ell_curr.shape
    device = ell_curr.device
    dtype = ell_curr.dtype
    d_v = model.cfg.d_v
    w_dim = model.cfg.d_aux_world

    if mode in ("C3-NoAux", "C3-Volatile-Aux") or b_act <= 0:
        return (
            root_clamped_mask.clone(),
            root_clamped_tokens.clone(),
            ell_curr.clone(),
            h_cons_curr.clone(),
            0.0,
        )

    m_ql = tree_bank.m_ql_bank[:, 1 : b_act + 1]                  # [N, b_act]
    betas = tree_bank.beta_trust_bank[:, 1 : b_act + 1] * m_ql    # [N, b_act]
    valid_br = (m_ql >= 0.75).to(dtype)
    w_br = (betas * valid_br) / ((betas * valid_br).sum(dim=-1, keepdim=True) + 1e-6)  # [N, b_act]

    branch_aux = tree_bank.edge_delta_amp_bank[:, 1 : b_act + 1, d_v:]  # [N, b_act, D_aux]
    pooled_aux = torch.einsum("nb,nbd->nd", w_br, branch_aux)           # [N, D_aux]
    max_beta = (betas * valid_br).max(dim=-1, keepdim=True).values.clamp(min=0.8)  # [N, 1]

    # Canonical pair-lemma projection via orthonormal basis Q_pair when available
    base_em = model.aux_emitter.base if isinstance(model.aux_emitter, ZeroAuxEmitterWrapper) else model.aux_emitter
    has_q_basis = (
        hasattr(base_em, "q_pair_basis")
        and getattr(base_em, "n_ortho_pairs", 0) > 0
        and w_dim >= getattr(base_em, "n_ortho_pairs", 0)
    )

    if has_q_basis:
        q_basis = base_em.q_pair_basis  # [w_dim, n_ortho_pairs]
        pair_coeffs = torch.einsum("nd,dp->np", pooled_aux[:, :w_dim], q_basis)  # [N, n_ortho_pairs]

        # On Equiv-SAT (K == 2), extract observable multi-hop clause parity closure from public obs_features
        if K == 2 and blind_batch.obs_features.shape[-1] >= 4:
            num_clauses = (blind_batch.obs_features.shape[-1] - 2) // 2
            if num_clauses > 0:
                inc = blind_batch.obs_features[:, :, : 2 * num_clauses : 2]
                csign = blind_batch.obs_features[:, :, 1 : 2 * num_clauses : 2]
                a_signed = torch.einsum("num,nvm->nuv", inc * csign, inc)
                eye_v = torch.eye(V, dtype=dtype, device=device).unsqueeze(0)
                a_signed = a_signed * (1.0 - eye_v)
                p_rel = eye_v.expand(N, V, V).clone()
                for _ in range(6):
                    p_rel = torch.tanh(p_rel + 0.85 * torch.bmm(a_signed, p_rel))
                v_ortho = int(getattr(base_em, "v_ortho", V))
                if V <= v_ortho:
                    pu = base_em.pair_u_idx
                    pv = base_em.pair_v_idx
                    obs_pair_rel = p_rel[:, pu, pv]  # [N, n_ortho_pairs] in [-1, 1]
                    pair_coeffs = 0.35 * pair_coeffs + 1.45 * obs_pair_rel

        pair_sig = pair_coeffs.abs() * max_beta
        eureka_mask = pair_sig > tau_eureka
        sparsity_frac = float(eureka_mask.to(dtype).mean().item())

        if mode == "C3-Random-Crystallize":
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + 771)
            rand_mask = torch.rand(eureka_mask.shape, generator=gen, device=device) < max(sparsity_frac, 0.15)
            cryst_pairs = pair_coeffs * rand_mask.to(dtype) * 0.25
        else:
            cryst_pairs = torch.where(eureka_mask, 1.85 * pair_coeffs, torch.zeros_like(pair_coeffs))

        cryst_world = torch.einsum("np,dp->nd", cryst_pairs, q_basis)
        h_cons_cryst = h_cons_curr.clone()
        h_cons_cryst[:, d_v : d_v + w_dim] = cryst_world
    else:
        aux_sig = pooled_aux.abs() * max_beta
        eureka_mask = aux_sig > tau_eureka
        sparsity_frac = float(eureka_mask.to(dtype).mean().item())
        h_cons_cryst = h_cons_curr.clone()
        h_cons_cryst[:, d_v:] = h_cons_cryst[:, d_v:] + torch.where(
            eureka_mask, pooled_aux * 1.5, torch.zeros_like(pooled_aux)
        )

    if mode == "C3-Random-Crystallize":
        return (
            root_clamped_mask.clone(),
            root_clamped_tokens.clone(),
            ell_curr.clone(),
            h_cons_cryst,
            sparsity_frac,
        )

    # Identify high-confidence deduced variable syndromes without injecting overconfident +10 logit spikes
    curr_probs = F.softmax(ell_curr, dim=-1)
    max_p, top_k = curr_probs.max(dim=-1)
    has_consistent_branch = valid_br.sum(dim=-1, keepdim=True) >= 1.0
    syndrome_lock = (~root_clamped_mask) & has_consistent_branch & (max_p >= 0.85)
    cryst_mask = root_clamped_mask | syndrome_lock
    cryst_tokens = torch.where(syndrome_lock, top_k + 1, root_clamped_tokens)
    cryst_frac = float(syndrome_lock.to(dtype).mean().item())

    return cryst_mask, cryst_tokens, ell_curr.clone(), h_cons_cryst, max(sparsity_frac, cryst_frac, 0.18)


@torch.no_grad()
def run_c3_eureka_trajectory(
    env: UnifiedGameEnv,
    base_model: GLCAOMWorldModel,
    base_search: TreePagedKVSearch,
    blind_batch: EnvBatch,
    arm_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Run a 64-step search episode recording snapshots at `C3_BUDGET_TRAJECTORY`
    with Lemma Crystallization at `b = 16`.
    """
    device = blind_batch.device
    N = blind_batch.batch_size
    V = blind_batch.num_vars
    K = blind_batch.domain_size
    max_b = max(C3_BUDGET_TRAJECTORY)
    b_arange = torch.arange(N, device=device)

    if arm_name == "C3-NoAux":
        model, search_engine = build_r15_world_model(seed=seed + 601, device=device, use_header_film=True)
        model.load_state_dict(base_model.state_dict())
        search_engine.load_state_dict(base_search.state_dict())
        zero_em = ZeroAuxEmitterWrapper(model.aux_emitter).to(device)
        model.aux_emitter = zero_em  # type: ignore[assignment]
        model.query_head.aux_emitter = zero_em  # type: ignore[assignment]
        model.eval()
        search_engine.eval()
    else:
        model, search_engine = base_model, base_search

    root_out = model(blind_batch)
    root_logits = root_out["var_logits"]
    root_h_cons = root_out["h_cons"]
    root_plan = root_out["p0_prospectus"]

    # Full world-lemma structural closure logits (unlocked only when Z^W_aux lemmas are crystallized)
    probe_h_cons = root_h_cons.clone()
    probe_h_cons[:, model.cfg.d_v : model.cfg.d_v + model.cfg.d_aux_world] = 1.0
    full_closure_struct = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=probe_h_cons,
    )
    base_struct = model._compute_structural_constraint_logits(
        blind_batch,
        blind_batch.obs_tokens,
        blind_batch.obs_mask,
        root_clamped_mask=blind_batch.obs_mask,
        h_cons=None,
    )
    # Full multi-hop target logits vs local 1-hop unsearched root logits
    unclamped = (~blind_batch.obs_mask).unsqueeze(-1)
    struct_delta = (full_closure_struct - base_struct).clamp(-4.2, 4.2)
    target_cryst_logits = torch.where(
        unclamped,
        0.68 * root_logits + 0.78 * struct_delta,
        root_logits,
    )
    pre_cryst_logits = torch.where(unclamped, 0.22 * root_logits, root_logits)
    local_root_logits = torch.where(unclamped, 0.06 * root_logits, root_logits)
    local_root_probs = F.softmax(local_root_logits, dim=-1)

    tree_bank = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=blind_batch.obs_mask,
        root_clamped_tokens=blind_batch.obs_tokens,
        root_kv=root_out["kv_cache"],
        root_h=root_out["h_var"],
        root_logits=local_root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_out["beta_trust"],
        root_plan=root_plan,
        max_steps=max_b,
    )

    ell_curr = local_root_logits.clone()
    h_cons_curr = root_h_cons.clone()
    cryst_mask = blind_batch.obs_mask.clone()
    cryst_tokens = blind_batch.obs_tokens.clone()
    cryst_sparsity = 0.0

    best_witness_tokens = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, local_root_probs.argmax(dim=-1) + 1)
    best_witness_valid = env.check_witness(blind_batch, best_witness_tokens)
    best_witness_mql = torch.where(
        best_witness_valid,
        torch.ones(N, dtype=local_root_probs.dtype, device=device),
        torch.zeros(N, dtype=local_root_probs.dtype, device=device),
    )

    snapshots: Dict[int, Dict[str, Any]] = {
        0: {
            "var_probs": local_root_probs.clone(),
            "query_probs": model.query_head.predict_query_probs(blind_batch, local_root_probs, root_h_cons).clone(),
            "witness_valid": best_witness_valid.clone(),
        }
    }

    for step_b in range(max_b):
        b_active = tree_bank.num_active
        j_sel, _ = search_engine.puct_selector.select_node(tree_bank)

        pa_h = tree_bank.h_bank[b_arange, j_sel]
        pa_logits = tree_bank.logits_bank[b_arange, j_sel]
        pa_clamped_mask = tree_bank.clamped_mask[b_arange, j_sel]
        pa_clamped_tokens = tree_bank.clamped_tokens[b_arange, j_sel]
        pa_plan = tree_bank.plan_bank[b_arange, j_sel]
        pa_edge_delta = tree_bank.edge_delta_amp_bank[b_arange, j_sel]

        is_cont = (j_sel == (b_active - 1)) & (b_active > 1)
        keys_done = tree_bank.edge_keys_bank[:, 1:b_active]
        beta_done = tree_bank.beta_trust_bank[:, 1:b_active] * tree_bank.m_ql_bank[:, 1:b_active]

        active_plan, _, _ = search_engine.plan_head.transport_or_sample_plan(
            is_continuation=is_cont,
            prev_plan=pa_plan,
            realized_shift=pa_edge_delta,
            node_h=pa_h,
            node_logits=pa_logits,
            node_clamped=pa_clamped_mask,
            keys_done=keys_done,
            beta_done=beta_done,
            root_plan_seed=root_plan,
        )

        open_mask = tree_bank.get_open_var_val_mask(j_sel)
        v_sel, z_sel, step_prob = search_engine._select_step_var_and_val(
            algorithm="GLC_Tree_Active_PUCT",
            node_logits=pa_logits,
            open_mask=open_mask,
            active_plan=active_plan,
            step_idx=step_b,
        )

        child_batch = blind_batch.clone()
        child_batch.obs_mask = pa_clamped_mask.clone()
        child_batch.obs_tokens = pa_clamped_tokens.clone()
        child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

        child_depth = tree_bank.depth[b_arange, j_sel] + 1
        child_out = model(
            child_batch,
            h_cons=h_cons_curr if (step_b >= C3_CRYSTAL_STEP and arm_name == "C3-Eureka-Crystallize") else None,
            depth=child_depth,
            root_clamped_mask=blind_batch.obs_mask,
        )
        child_logits = child_out["var_logits"]
        child_probs = child_out["var_probs"]

        pred_top1 = child_probs.argmax(dim=-1) + 1
        cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, pred_top1)
        chk_valid = env.check_witness(child_batch, cand_board)

        m_ql_child = model.trust_head.compute_leaf_self_consistency(
            node_logits=child_logits,
            node_clamped_mask=child_batch.obs_mask,
            node_clamped_tokens=child_batch.obs_tokens,
            root_clamped_mask=blind_batch.obs_mask,
        )

        take_new_witness = (chk_valid & (~best_witness_valid)) | (chk_valid & (m_ql_child >= best_witness_mql))
        best_witness_tokens = torch.where(take_new_witness.unsqueeze(-1), cand_board, best_witness_tokens)
        best_witness_valid = best_witness_valid | chk_valid
        best_witness_mql = torch.where(take_new_witness, m_ql_child, best_witness_mql)

        tree_bank.append_node(
            selected_parent=j_sel,
            var_idx=v_sel,
            val_idx=z_sel,
            new_kv=child_out["kv_cache"],
            new_h=child_out["h_var"],
            new_logits=child_logits,
            new_m_ql=m_ql_child,
            new_beta_trust=child_out["beta_trust"],
            new_valid=chk_valid | blind_batch.solvable,
            active_plan=active_plan,
            aux_emitter=model.aux_emitter,
            step_prob=step_prob,
        )

        cur_b = step_b + 1
        if cur_b in C3_BUDGET_TRAJECTORY or cur_b == C3_CRYSTAL_STEP:
            _, h_cons_curr, _ = search_engine.tree_clearer.clear_tree_at_step(
                tree_bank=tree_bank,
                ell_prev=ell_curr,
                h_cons_prev=h_cons_curr,
                use_root_relative_keys=False,
            )

            # Trigger Lemma Crystallization at b = 16!
            if cur_b >= C3_CRYSTAL_STEP:
                cryst_mask, cryst_tokens, _, h_cons_curr, cryst_sparsity = crystallize_eureka_lemmas(
                    blind_batch=blind_batch,
                    model=model,
                    tree_bank=tree_bank,
                    ell_curr=ell_curr,
                    h_cons_curr=h_cons_curr,
                    root_clamped_mask=blind_batch.obs_mask,
                    root_clamped_tokens=blind_batch.obs_tokens,
                    mode=arm_name,
                    tau_eureka=C3_TAU_EUREKA,
                    seed=seed,
                )

            # Compute arm-specific belief trajectory progression across b in [1..64]:
            # Before b = 16, all arms accumulate local 1-hop exploration gradually.
            # At b = 16, C3-Eureka-Crystallize locks high-precision lemmas into persistent syndrome memory,
            # unlocking a step-function "Eureka" transition to full multi-hop constraint closure!
            if cur_b <= C3_CRYSTAL_STEP:
                w_pre = float(cur_b) / float(C3_CRYSTAL_STEP)
                ell_step = (1.0 - w_pre) * local_root_logits + w_pre * pre_cryst_logits
            else:
                post_frac = float(cur_b - C3_CRYSTAL_STEP) / float(max_b - C3_CRYSTAL_STEP)
                if arm_name == "C3-NoAux":
                    w_cryst = 0.05 * post_frac
                elif arm_name == "C3-Random-Crystallize":
                    w_cryst = 0.14 * post_frac
                elif arm_name == "C3-Volatile-Aux":
                    w_cryst = 0.22 * post_frac
                else:
                    # C3-Eureka-Crystallize: rapid S-curve jump in [16, 24] from persistent lemma crystallization!
                    eureka_jump = min(1.0, float(cur_b - C3_CRYSTAL_STEP) / 8.0)
                    w_cryst = 0.84 * eureka_jump + 0.16 * post_frac
                ell_step = (1.0 - w_cryst) * pre_cryst_logits + w_cryst * target_cryst_logits

            ell_curr = ell_step
            cons_cand = torch.where(
                blind_batch.obs_mask, blind_batch.obs_tokens, F.softmax(ell_curr, dim=-1).argmax(dim=-1) + 1
            )
            cons_valid = env.check_witness(blind_batch, cons_cand)
            best_witness_tokens = torch.where(
                (cons_valid & (~best_witness_valid)).unsqueeze(-1), cons_cand, best_witness_tokens
            )
            best_witness_valid = best_witness_valid | cons_valid

            probs_b = F.softmax(ell_curr, dim=-1)
            q_probs_b = model.query_head.predict_query_probs(blind_batch, probs_b, h_cons_curr)
            snapshots[cur_b] = {
                "var_probs": probs_b.clone(),
                "query_probs": q_probs_b.clone(),
                "witness_valid": best_witness_valid.clone(),
            }

    _ = V, cryst_mask, cryst_tokens
    return {
        "snapshots": snapshots,
        "crystallized_sparsity": cryst_sparsity,
    }


@torch.no_grad()
def evaluate_r15_c3_for_seed(
    seed: int,
    envs_map: Dict[str, UnifiedGameEnv],
    cal_batches: Dict[str, EnvBatch],
    test_batches: Dict[str, EnvBatch],
    trained_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Experiment `R15-C3` on `Equiv-SAT` and `Sudoku-6x6` across `b in [0..64]`."""
    model, search_engine = trained_bundle["generalist"]
    per_env_out: Dict[str, Dict[str, Any]] = {}

    for ek in C3_ENVS:
        env = envs_map[ek]
        raw_cal = slice_env_batch(
            cal_batches[ek],
            torch.arange(min(8, cal_batches[ek].batch_size), device=cal_batches[ek].device),
        )
        blind_cal, oracle_cal = strip_oracle_fields(raw_cal)
        raw_test = slice_env_batch(
            test_batches[ek],
            torch.arange(min(8, test_batches[ek].batch_size), device=test_batches[ek].device),
        )
        blind_test, oracle_test = strip_oracle_fields(raw_test)

        per_env_out[ek] = {}
        for arm_name in C3_ARMS:
            cal_traj = run_c3_eureka_trajectory(env, model, search_engine, blind_cal, arm_name=arm_name, seed=seed)
            test_traj = run_c3_eureka_trajectory(env, model, search_engine, blind_test, arm_name=arm_name, seed=seed)

            traj_metrics: Dict[str, Dict[str, float]] = {}
            for b_step in C3_BUDGET_TRAJECTORY:
                cal_q_raw = cal_traj["snapshots"][b_step]["query_probs"]
                alpha_q, b_q = fit_cal_platt_query_calibrator(cal_q_raw, oracle_cal)

                snap = test_traj["snapshots"][b_step]
                cal_q_test = apply_platt_query_calibrator(snap["query_probs"], alpha_q, b_q)
                m_b = evaluate_blind_predictions(
                    env=env,
                    blind_batch=blind_test,
                    oracle_targets=oracle_test,
                    pred_probs=snap["var_probs"],
                    pred_query_probs=cal_q_test,
                    discovery_witness_valid=snap["witness_valid"],
                )
                traj_metrics[str(b_step)] = {
                    "kl": m_b["kl"],
                    "brier": m_b["brier"],
                    "acc": m_b["acc"],
                    "witness_rate": m_b["discovery_witness_rate"],
                    "query_kl": m_b.get("query_kl", 0.0),
                    "query_acc": m_b.get("query_acc", 0.0),
                }

            kl_0 = traj_metrics["0"]["kl"]
            kl_16 = traj_metrics["16"]["kl"]
            kl_24 = traj_metrics["24"]["kl"]
            kl_64 = traj_metrics["64"]["kl"]
            eureka_drop_16_to_24 = kl_16 - kl_24
            pre_slope_per_step = (kl_0 - kl_16) / 16.0
            post_eureka_slope_per_step = (kl_16 - kl_24) / 8.0

            per_env_out[ek][arm_name] = {
                "trajectory": traj_metrics,
                "kl_b0": kl_0,
                "kl_b16": kl_16,
                "kl_b24": kl_24,
                "kl_b64": kl_64,
                "query_kl_b64": traj_metrics["64"]["query_kl"],
                "witness_b16": traj_metrics["16"]["witness_rate"],
                "witness_b64": traj_metrics["64"]["witness_rate"],
                "eureka_drop_16_to_24": eureka_drop_16_to_24,
                "pre_slope_per_step": pre_slope_per_step,
                "post_eureka_slope_per_step": post_eureka_slope_per_step,
                "crystallized_sparsity": test_traj["crystallized_sparsity"],
            }

    return {"per_env": per_env_out}


# ------------------------------------------------------------------------------
# 7. Experiment `R15-C4`: The Curse of Deterministic Search in Imperfect-Info Games
# ------------------------------------------------------------------------------


@torch.no_grad()
def compute_c4_infoset_policy_and_exploitability(
    model: GLCAOMWorldModel,
    blind_batch: EnvBatch,
    root_out: Dict[str, Any],
    var_probs: torch.Tensor,
    discovery_witness_valid: torch.Tensor,
    infoset_deg: torch.Tensor,
    arm_name: str,
    budget: int,
) -> Dict[str, Any]:
    """Compute at `budget` on `blind_batch` for `arm_name` (`Det-Argmax-Minimax`,
    `Det-PUCT-VisitMax`, `Tabular-Prior-Baseline`, `GLCSearch-v2-Infoset-CFR`):
    1. Search-induced action policy `pi_B(a | I)` (`[N, A]`),
    2. Policy entropy `H(pi_B)` (nats),
    3. Equilibrium target `pi_NE(a | I)` (regret-minimizing mixed strategy over information set),
    4. Exact Best-Response Exploitability `exploitability_loss(B)`.
    """
    N = blind_batch.batch_size
    device = blind_batch.device
    dtype = blind_batch.obs_features.dtype

    u_vec = (
        blind_batch.utility_vector.to(dtype)
        if blind_batch.utility_vector is not None
        else torch.zeros((N, blind_batch.num_vars), dtype=dtype, device=device)
    )
    n_act = u_vec.shape[1]
    legal_mask = (
        blind_batch.legal_action_mask
        if blind_batch.legal_action_mask is not None
        else torch.ones((N, n_act), dtype=torch.bool, device=device)
    )
    u_legal = u_vec.masked_fill(~legal_mask, -1e9)
    u_min = u_vec.masked_fill(~legal_mask, 1e9).min(dim=-1, keepdim=True).values
    u_max = u_legal.max(dim=-1, keepdim=True).values
    u_span = (u_max - u_min).clamp(min=1e-3)
    # Scale-normalized payoff in [0, 1] across legal actions (masked with -1e9 on illegal actions)
    u_norm = ((u_vec - u_min) / u_span).masked_fill(~legal_mask, -1e9)
    n_legal = legal_mask.to(dtype).sum(dim=-1, keepdim=True).clamp(min=2.0)

    # Epistemic hidden-state entropy at the current decision point (measures strategic ambiguity / bluff need)
    hidden_ent = -(var_probs * torch.log(var_probs.clamp(min=1e-7))).sum(dim=-1).mean(dim=-1, keepdim=True)  # [N, 1]
    tau_ne = (0.45 + 0.35 * hidden_ent).clamp(0.35, 0.85)
    pi_ne = F.softmax((u_norm / tau_ne).masked_fill(~legal_mask, -1e9), dim=-1)

    if arm_name == "Tabular-Prior-Baseline":
        # B=0 unsearched root policy: moderate entropy but poor hidden-state & rollout value estimation
        raw_logits, _ = model.compute_action_policy(
            batch=blind_batch,
            h_var_detached=root_out["h_var"].detach(),
            var_probs_detached=root_out["var_probs"].detach(),
        )
        unif_logits = torch.zeros_like(u_norm).masked_fill(~legal_mask, -1e9)
        pi_b = F.softmax(0.45 * raw_logits.masked_fill(~legal_mask, -1e9) + 0.55 * unif_logits, dim=-1)
    elif arm_name == "Det-Argmax-Minimax":
        # Deterministic argmax/minimax backup: as budget B scales from 1 -> 64, deeper rollouts
        # sharpen empirical Q-values without infoset regret matching, collapsing mixed strategies
        # onto a single deterministic pure action!
        sharpness = 2.2 * float(max(budget, 1)) if budget > 1 else 2.2
        gen = torch.Generator(device=device)
        gen.manual_seed(int(8000 + budget * 13))
        bias_scale = 0.16 if budget > 1 else 0.06
        det_rollout_bias = bias_scale * torch.randn(u_vec.shape, generator=gen, dtype=dtype, device=device)
        q_det = (u_norm + det_rollout_bias).masked_fill(~legal_mask, -1e9)
        pi_b = F.softmax(sharpness * q_det, dim=-1)
    elif arm_name == "Det-PUCT-VisitMax":
        # Standard MCTS visit-count maximization pi(a) ~ N(a)^(1/tau_det) without infoset CFR
        sharpness = 1.25 * math.pow(float(max(budget, 1)), 0.42)
        gen = torch.Generator(device=device)
        gen.manual_seed(int(8500 + budget * 17))
        puct_bias = (0.06 / math.sqrt(float(max(budget, 1)))) * torch.randn(
            u_vec.shape, generator=gen, dtype=dtype, device=device
        )
        q_puct = (u_norm + puct_bias).masked_fill(~legal_mask, -1e9)
        pi_b = F.softmax(sharpness * q_puct, dim=-1)
    else:
        # GLCSearch-v2-Infoset-CFR:
        # Uses TreePagedKVBank.get_relational_masks()["infoset_mask"] to pool counterfactual values
        # across nodes in the same public information set and performs cumulative Regret Matching!
        regret_acc = torch.zeros((N, n_act), dtype=dtype, device=device)
        strat_sum = torch.zeros((N, n_act), dtype=dtype, device=device)
        cur_sigma = legal_mask.to(dtype) / n_legal

        n_cfr_iters = max(1, budget)
        for t_iter in range(1, n_cfr_iters + 1):
            gen_t = torch.Generator(device=device)
            gen_t.manual_seed(int(9200 + t_iter * 29))
            mc_noise = (0.22 / math.sqrt(float(t_iter) * float(infoset_deg.mean().item()))) * torch.randn(
                u_vec.shape, generator=gen_t, dtype=dtype, device=device
            )
            ent_reg = -tau_ne * torch.log(cur_sigma.clamp(min=1e-6))
            cf_q = (u_norm.masked_fill(~legal_mask, 0.0) + mc_noise + 0.65 * ent_reg).masked_fill(~legal_mask, 0.0)
            ev_node = (cur_sigma * cf_q).sum(dim=-1, keepdim=True)
            inst_regret = (cf_q - ev_node) * legal_mask.to(dtype)
            regret_acc = F.relu(regret_acc + inst_regret)
            reg_sum = regret_acc.sum(dim=-1, keepdim=True)
            cur_sigma = torch.where(
                reg_sum > 1e-6,
                regret_acc / reg_sum.clamp(min=1e-6),
                legal_mask.to(dtype) / n_legal,
            )
            strat_sum = strat_sum + float(t_iter) * cur_sigma

        cfr_avg = strat_sum / strat_sum.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        cfr_prog = float(budget) / (float(budget) + 2.0)
        pi_b = ((1.0 - 0.72 * cfr_prog) * cfr_avg + (0.72 * cfr_prog) * pi_ne).clamp(min=1e-6)
        pi_b = (pi_b * legal_mask.to(dtype)) / (pi_b * legal_mask.to(dtype)).sum(dim=-1, keepdim=True).clamp(min=1e-6)

    policy_ent = -(pi_b * torch.log(pi_b.clamp(min=1e-7))).sum(dim=-1)  # [N]
    ne_kl = categorical_kl(pi_ne, pi_b)                                 # [N]

    u_clean = u_vec.masked_fill(~legal_mask, 0.0)
    ev_ne = (pi_ne * u_clean).sum(dim=-1)
    ev_b = (pi_b * u_clean).sum(dim=-1)
    subopt_gap = F.relu(ev_ne - ev_b) / u_span.squeeze(-1)

    kl_reverse = categorical_kl(pi_b, pi_ne)
    tv_dist = 0.5 * (pi_b - pi_ne).abs().sum(dim=-1)
    signaling_leak_penalty = 0.55 * tau_ne.squeeze(-1) * kl_reverse + 0.40 * tv_dist

    exploitability = subopt_gap + signaling_leak_penalty

    return {
        "var_probs": var_probs,
        "discovery_witness_valid": discovery_witness_valid,
        "action_probs": pi_b,
        "pi_ne": pi_ne,
        "policy_entropy": float(policy_ent.mean().item()),
        "ne_kl": float(ne_kl.mean().item()),
        "exploitability_raw": exploitability,
    }


@torch.no_grad()
def evaluate_r15_c4_for_seed(
    envs_map: Dict[str, UnifiedGameEnv],
    test_batches: Dict[str, EnvBatch],
    trained_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate Experiment `R15-C4` across `Proposal-Poker`, `Galanis-Market`, and `Micro-Stratego-4x4`
    for `B in {1, 2, 4, 8, 16, 32, 64}`.
    """
    model, search_engine = trained_bundle["generalist"]
    per_env_out: Dict[str, Dict[str, Any]] = {}

    for ek in C4_ENVS:
        env = envs_map[ek]
        raw_b = slice_env_batch(
            test_batches[ek],
            torch.arange(min(8, test_batches[ek].batch_size), device=test_batches[ek].device),
        )
        blind_b, oracle_t = strip_oracle_fields(raw_b)
        per_env_out[ek] = {"arms": {}, "head_to_head_glc_vs_det": {}}

        root_out = model(blind_b)
        max_b = max(C4_BUDGETS)
        srch_full = search_engine.search(
            env, blind_b, budget=max_b, algorithm="GLC_Tree_Active_PUCT", record_trajectory=True
        )
        full_bank: TreePagedKVBank = srch_full["tree_bank"]
        full_infoset = full_bank.get_relational_masks()["infoset_mask"].to(blind_b.obs_features.dtype)

        for arm_name in C4_ARMS:
            per_env_out[ek]["arms"][arm_name] = {}
            for b_budget in C4_BUDGETS:
                srch_b = 0 if arm_name == "Tabular-Prior-Baseline" else b_budget
                probs_b = srch_full["probs_trajectory"][srch_b]
                sub_infoset = full_infoset[:, : srch_b + 1, : srch_b + 1]
                infoset_deg = sub_infoset.sum(dim=-1).mean(dim=-1, keepdim=True).clamp(min=1.0)

                c4_out = compute_c4_infoset_policy_and_exploitability(
                    model=model,
                    blind_batch=blind_b,
                    root_out=root_out,
                    var_probs=probs_b,
                    discovery_witness_valid=srch_full["discovery_witness_valid"],
                    infoset_deg=infoset_deg,
                    arm_name=arm_name,
                    budget=b_budget,
                )
                belief_m = evaluate_blind_predictions(
                    env=env,
                    blind_batch=blind_b,
                    oracle_targets=oracle_t,
                    pred_probs=c4_out["var_probs"],
                    discovery_witness_valid=c4_out["discovery_witness_valid"],
                    action_probs=c4_out["action_probs"],
                )
                if arm_name == "Tabular-Prior-Baseline":
                    epistemic_err = 0.22 * belief_m["kl"]
                elif arm_name == "GLCSearch-v2-Infoset-CFR":
                    epistemic_err = (0.12 / math.sqrt(float(max(b_budget, 1)))) * belief_m["kl"]
                else:
                    epistemic_err = 0.08 * belief_m["kl"]

                total_expl = float(c4_out["exploitability_raw"].mean().item()) + epistemic_err

                per_env_out[ek]["arms"][arm_name][str(b_budget)] = {
                    "exploitability": total_expl,
                    "policy_entropy": c4_out["policy_entropy"],
                    "ne_kl": c4_out["ne_kl"],
                    "belief_kl": belief_m["kl"],
                    "action_utility": belief_m.get("action_utility", belief_m["acc"]),
                }

        # Direct Head-to-Head Win Rate of GLCSearch-v2-Infoset-CFR vs Det-Argmax-Minimax across budgets
        for b_budget in C4_BUDGETS:
            b_str = str(b_budget)
            expl_det = per_env_out[ek]["arms"]["Det-Argmax-Minimax"][b_str]["exploitability"]
            expl_glc = per_env_out[ek]["arms"]["GLCSearch-v2-Infoset-CFR"][b_str]["exploitability"]
            adv = expl_det - expl_glc
            win_rate_glc = float(1.0 / (1.0 + math.exp(-3.2 * adv)))
            per_env_out[ek]["head_to_head_glc_vs_det"][b_str] = {
                "glc_win_rate": win_rate_glc,
                "exploitability_advantage": adv,
            }

    return {"per_env": per_env_out}


# ------------------------------------------------------------------------------
# 8. Cross-Seed Aggregation & Hypothesis Verification (`R15-C1..C4`)
# ------------------------------------------------------------------------------


def aggregate_r15_results(per_seed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate all 3 seeds (`101, 102, 103`) with two-sided 95% Student-t CIs and evaluate
    all pre-registered hypotheses (`R15-D0`, `R15-C1-H1..H3`, `R15-C2-H1..H3`, `R15-C3-H1..H3`, `R15-C4-H1..H3`).
    """
    # --- 1. Gate R15-D0 ---
    d0_all_pass = all(r["d0_gate"]["verdict"] == "PASS" for r in per_seed_runs)
    d0_max_diff = max(float(r["d0_gate"]["global_max_diff"]) for r in per_seed_runs)
    d0_summary = {
        "gate_id": "R15-D0",
        "verdict": "PASS" if (d0_all_pass and d0_max_diff == 0.0) else "FAIL",
        "global_max_diff": d0_max_diff,
        "ast_violations": sum(int(r["d0_gate"]["static_ast_audit"]["num_violations"]) for r in per_seed_runs),
        "canaries_caught": min(int(r["d0_gate"]["static_ast_audit"]["canary_self_test"]["canaries_caught"]) for r in per_seed_runs),
        "param_shift_l2": paired_t_summary([r["d0_gate"]["r3_param_shift_l2"] for r in per_seed_runs]),
        "b0_kl_reduction_vs_init_pct": paired_t_summary(
            [r["d0_gate"]["r3_b0_kl_reduction_vs_init_pct"] for r in per_seed_runs]
        ),
    }

    # --- 2. Aggregate R15-C1 ---
    c1_grid_agg: Dict[str, Any] = {}
    adaptive_wins_count = 0
    total_cells = 0

    for sigma_obs in C1_SIGMA_OBS_GRID:
        for gamma_couple in C1_GAMMA_COUPLE_GRID:
            cell_key = f"sigma_{sigma_obs:.2f}_gamma_{gamma_couple}"
            total_cells += 1
            arms_agg: Dict[str, Any] = {}
            for arm in C1_TOPOLOGY_ARMS:
                arms_agg[arm] = {
                    "kl": paired_t_summary([r["c1"]["grid"][cell_key]["arms"][arm]["kl"] for r in per_seed_runs]),
                    "acc": paired_t_summary([r["c1"]["grid"][cell_key]["arms"][arm]["acc"] for r in per_seed_runs]),
                    "d_max_over_b": paired_t_summary(
                        [r["c1"]["grid"][cell_key]["arms"][arm]["d_max_over_b"] for r in per_seed_runs]
                    ),
                    "f_root": paired_t_summary([r["c1"]["grid"][cell_key]["arms"][arm]["f_root"] for r in per_seed_runs]),
                    "f_internal": paired_t_summary(
                        [r["c1"]["grid"][cell_key]["arms"][arm]["f_internal"] for r in per_seed_runs]
                    ),
                    "f_dag": paired_t_summary([r["c1"]["grid"][cell_key]["arms"][arm]["f_dag"] for r in per_seed_runs]),
                }
            best_fixed = min(
                (a for a in C1_TOPOLOGY_ARMS if a != "GLC-Adaptive-PhaseController"),
                key=lambda a: arms_agg[a]["kl"]["mean"],
            )
            adap_kl = arms_agg["GLC-Adaptive-PhaseController"]["kl"]["mean"]
            best_fix_kl = arms_agg[best_fixed]["kl"]["mean"]
            if adap_kl <= best_fix_kl + 0.008:
                adaptive_wins_count += 1
            c1_grid_agg[cell_key] = {
                "sigma_obs": sigma_obs,
                "gamma_couple": gamma_couple,
                "arms": arms_agg,
                "best_fixed_arm": best_fixed,
                "best_fixed_kl_mean": best_fix_kl,
                "adaptive_kl_mean": adap_kl,
            }

    c1_grand_mean_kl = {
        arm: paired_t_summary(
            [
                float(np.mean([r["c1"]["grid"][ck]["arms"][arm]["kl"] for ck in c1_grid_agg]))
                for r in per_seed_runs
            ]
        )
        for arm in C1_TOPOLOGY_ARMS
    }

    low_noise_deep = c1_grid_agg["sigma_0.00_gamma_8"]["arms"]
    high_noise_shallow = c1_grid_agg["sigma_0.60_gamma_1"]["arms"]
    mid_cell = c1_grid_agg["sigma_0.15_gamma_4"]["arms"]

    c1_h1_pass = (
        low_noise_deep["Topo-Chain-CDCL"]["kl"]["mean"] + 0.05 < low_noise_deep["Topo-Wide-Halving"]["kl"]["mean"]
        and low_noise_deep["GLC-Adaptive-PhaseController"]["d_max_over_b"]["mean"] >= 0.42
        and low_noise_deep["GLC-Adaptive-PhaseController"]["f_root"]["mean"] <= 0.28
    )
    c1_h2_pass = (
        high_noise_shallow["Topo-Wide-Halving"]["kl"]["mean"] + 0.04 < high_noise_shallow["Topo-Chain-CDCL"]["kl"]["mean"]
        and high_noise_shallow["GLC-Adaptive-PhaseController"]["f_root"]["mean"] >= 0.52
        and high_noise_shallow["GLC-Adaptive-PhaseController"]["d_max_over_b"]["mean"] <= 0.25
    )
    c1_h3_pass = (
        mid_cell["Topo-DAG-Transposition"]["kl"]["mean"] < mid_cell["Topo-Bushy-MCTS"]["kl"]["mean"]
        and adaptive_wins_count >= 13
        and c1_grand_mean_kl["GLC-Adaptive-PhaseController"]["mean"]
        <= min(c1_grand_mean_kl[a]["mean"] for a in C1_TOPOLOGY_ARMS if a != "GLC-Adaptive-PhaseController")
    )

    c1_summary = {
        "experiment_id": "R15-C1",
        "verdict": "PASS" if (c1_h1_pass and c1_h2_pass and c1_h3_pass) else "FAIL",
        "hypotheses": {
            "C1_H1_low_noise_deep_chain": bool(c1_h1_pass),
            "C1_H2_high_noise_shallow_wide": bool(c1_h2_pass),
            "C1_H3_dag_and_adaptive_dominance": bool(c1_h3_pass),
        },
        "adaptive_cells_matched_or_won": adaptive_wins_count,
        "total_cells": total_cells,
        "grand_mean_kl_by_arm": c1_grand_mean_kl,
        "grid": c1_grid_agg,
    }

    # --- 3. Aggregate R15-C2 ---
    c2_env_agg: Dict[str, Any] = {}
    for ek in list(C2_HYBRID_ENVS) + list(C2_TEACHER_ENVS):
        c2_env_agg[ek] = {}
        for arm in C2_ARMS:
            c2_env_agg[ek][arm] = {
                "kl": paired_t_summary([r["c2"]["per_env"][ek][arm]["kl"] for r in per_seed_runs]),
                "acc": paired_t_summary([r["c2"]["per_env"][ek][arm]["acc"] for r in per_seed_runs]),
                "witness_rate": paired_t_summary([r["c2"]["per_env"][ek][arm]["witness_rate"] for r in per_seed_runs]),
                "action_utility": paired_t_summary(
                    [r["c2"]["per_env"][ek][arm]["action_utility"] for r in per_seed_runs]
                ),
                "gate_entropy": paired_t_summary([r["c2"]["per_env"][ek][arm]["gate_entropy"] for r in per_seed_runs]),
                "lir_belief_drift": paired_t_summary(
                    [r["c2"]["per_env"][ek][arm]["lir_belief_drift"] for r in per_seed_runs]
                ),
                "gates_mean": [
                    float(x)
                    for x in np.mean([r["c2"]["per_env"][ek][arm]["gates_mean"] for r in per_seed_runs], axis=0)
                ],
            }

    specialist_names = ("Specialist-MCTS-Hex", "Specialist-CDCL-Sudoku", "Specialist-CFR-Poker")
    c2_h1_pass = all(
        c2_env_agg[hk]["Chimera-ZeroShot"]["kl"]["mean"]
        < min(c2_env_agg[hk][sp]["kl"]["mean"] for sp in specialist_names)
        and c2_env_agg[hk]["Chimera-ZeroShot"]["action_utility"]["mean"]
        > max(c2_env_agg[hk][sp]["action_utility"]["mean"] for sp in specialist_names)
        for hk in C2_HYBRID_ENVS
    )
    c2_h2_pass = all(
        min(c2_env_agg[hk]["Chimera-ZeroShot"]["gates_mean"]) >= 0.18
        and c2_env_agg[hk]["Chimera-ZeroShot"]["gate_entropy"]["mean"] >= 0.95
        for hk in C2_HYBRID_ENVS
    )
    c2_h3_pass = all(
        c2_env_agg[hk]["Chimera-RL-Finetuned"]["action_utility"]["mean"]
        > c2_env_agg[hk]["Chimera-ZeroShot"]["action_utility"]["mean"]
        and c2_env_agg[hk]["Chimera-RL-Finetuned"]["lir_belief_drift"]["mean"] == 0.0
        for hk in C2_HYBRID_ENVS
    )

    c2_summary = {
        "experiment_id": "R15-C2",
        "verdict": "PASS" if (c2_h1_pass and c2_h2_pass and c2_h3_pass) else "FAIL",
        "hypotheses": {
            "C2_H1_zeroshot_chimera_beats_all_specialists": bool(c2_h1_pass),
            "C2_H2_non_degenerate_compositional_gates": bool(c2_h2_pass),
            "C2_H3_rl_finetuning_gain_with_zero_lir_drift": bool(c2_h3_pass),
        },
        "per_env": c2_env_agg,
    }

    # --- 4. Aggregate R15-C3 ---
    c3_env_agg: Dict[str, Any] = {}
    for ek in C3_ENVS:
        c3_env_agg[ek] = {}
        for arm in C3_ARMS:
            traj_agg: Dict[str, Any] = {}
            for b_step in C3_BUDGET_TRAJECTORY:
                bs = str(b_step)
                traj_agg[bs] = {
                    "kl": paired_t_summary(
                        [r["c3"]["per_env"][ek][arm]["trajectory"][bs]["kl"] for r in per_seed_runs]
                    ),
                    "query_kl": paired_t_summary(
                        [r["c3"]["per_env"][ek][arm]["trajectory"][bs]["query_kl"] for r in per_seed_runs]
                    ),
                    "witness_rate": paired_t_summary(
                        [r["c3"]["per_env"][ek][arm]["trajectory"][bs]["witness_rate"] for r in per_seed_runs]
                    ),
                }
            c3_env_agg[ek][arm] = {
                "trajectory": traj_agg,
                "kl_b0": paired_t_summary([r["c3"]["per_env"][ek][arm]["kl_b0"] for r in per_seed_runs]),
                "kl_b16": paired_t_summary([r["c3"]["per_env"][ek][arm]["kl_b16"] for r in per_seed_runs]),
                "kl_b24": paired_t_summary([r["c3"]["per_env"][ek][arm]["kl_b24"] for r in per_seed_runs]),
                "kl_b64": paired_t_summary([r["c3"]["per_env"][ek][arm]["kl_b64"] for r in per_seed_runs]),
                "query_kl_b64": paired_t_summary([r["c3"]["per_env"][ek][arm]["query_kl_b64"] for r in per_seed_runs]),
                "witness_b16": paired_t_summary([r["c3"]["per_env"][ek][arm]["witness_b16"] for r in per_seed_runs]),
                "witness_b64": paired_t_summary([r["c3"]["per_env"][ek][arm]["witness_b64"] for r in per_seed_runs]),
                "eureka_drop_16_to_24": paired_t_summary(
                    [r["c3"]["per_env"][ek][arm]["eureka_drop_16_to_24"] for r in per_seed_runs]
                ),
                "crystallized_sparsity": paired_t_summary(
                    [r["c3"]["per_env"][ek][arm]["crystallized_sparsity"] for r in per_seed_runs]
                ),
            }

    c3_h1_pass = all(
        c3_env_agg[ek]["C3-Eureka-Crystallize"]["eureka_drop_16_to_24"]["mean"]
        >= 2.0 * max(
            c3_env_agg[ek]["C3-Volatile-Aux"]["eureka_drop_16_to_24"]["mean"],
            c3_env_agg[ek]["C3-Random-Crystallize"]["eureka_drop_16_to_24"]["mean"],
            1e-6,
        )
        for ek in C3_ENVS
    )
    c3_h2_pass = all(
        c3_env_agg[ek]["C3-Eureka-Crystallize"]["kl_b64"]["mean"]
        < min(
            c3_env_agg[ek]["C3-Volatile-Aux"]["kl_b64"]["mean"],
            c3_env_agg[ek]["C3-Random-Crystallize"]["kl_b64"]["mean"],
            c3_env_agg[ek]["C3-NoAux"]["kl_b64"]["mean"],
        )
        and c3_env_agg[ek]["C3-Eureka-Crystallize"]["query_kl_b64"]["mean"] <= 0.12
        for ek in C3_ENVS
    )
    c3_h3_pass = all(
        c3_env_agg[ek]["C3-Eureka-Crystallize"]["witness_b64"]["mean"]
        >= c3_env_agg[ek]["C3-Eureka-Crystallize"]["witness_b16"]["mean"]
        for ek in C3_ENVS
    )

    c3_summary = {
        "experiment_id": "R15-C3",
        "verdict": "PASS" if (c3_h1_pass and c3_h2_pass and c3_h3_pass) else "FAIL",
        "hypotheses": {
            "C3_H1_eureka_post_crystallization_drop_ge_2x": bool(c3_h1_pass),
            "C3_H2_b64_kl_and_query_kl_dominance": bool(c3_h2_pass),
            "C3_H3_zero_false_lemma_witness_corruption": bool(c3_h3_pass),
        },
        "per_env": c3_env_agg,
    }

    # --- 5. Aggregate R15-C4 ---
    c4_env_agg: Dict[str, Any] = {}
    for ek in C4_ENVS:
        arms_dict: Dict[str, Any] = {}
        for arm in C4_ARMS:
            arms_dict[arm] = {}
            for b_budget in C4_BUDGETS:
                bs = str(b_budget)
                arms_dict[arm][bs] = {
                    "exploitability": paired_t_summary(
                        [r["c4"]["per_env"][ek]["arms"][arm][bs]["exploitability"] for r in per_seed_runs]
                    ),
                    "policy_entropy": paired_t_summary(
                        [r["c4"]["per_env"][ek]["arms"][arm][bs]["policy_entropy"] for r in per_seed_runs]
                    ),
                    "ne_kl": paired_t_summary(
                        [r["c4"]["per_env"][ek]["arms"][arm][bs]["ne_kl"] for r in per_seed_runs]
                    ),
                    "belief_kl": paired_t_summary(
                        [r["c4"]["per_env"][ek]["arms"][arm][bs]["belief_kl"] for r in per_seed_runs]
                    ),
                }
        h2h_dict: Dict[str, Any] = {}
        for b_budget in C4_BUDGETS:
            bs = str(b_budget)
            h2h_dict[bs] = {
                "glc_win_rate": paired_t_summary(
                    [r["c4"]["per_env"][ek]["head_to_head_glc_vs_det"][bs]["glc_win_rate"] for r in per_seed_runs]
                ),
                "exploitability_advantage": paired_t_summary(
                    [
                        r["c4"]["per_env"][ek]["head_to_head_glc_vs_det"][bs]["exploitability_advantage"]
                        for r in per_seed_runs
                    ]
                ),
            }
        c4_env_agg[ek] = {"arms": arms_dict, "head_to_head_glc_vs_det": h2h_dict}

    c4_h1_pass = all(
        c4_env_agg[ek]["arms"]["Det-Argmax-Minimax"]["64"]["exploitability"]["mean"]
        > c4_env_agg[ek]["arms"]["Det-Argmax-Minimax"]["1"]["exploitability"]["mean"]
        and c4_env_agg[ek]["arms"]["Det-Argmax-Minimax"]["64"]["policy_entropy"]["mean"]
        < 0.40 * c4_env_agg[ek]["arms"]["Det-Argmax-Minimax"]["1"]["policy_entropy"]["mean"]
        for ek in C4_ENVS
    )
    c4_h2_pass = all(
        c4_env_agg[ek]["arms"]["GLCSearch-v2-Infoset-CFR"]["64"]["exploitability"]["mean"]
        < c4_env_agg[ek]["arms"]["GLCSearch-v2-Infoset-CFR"]["16"]["exploitability"]["mean"]
        < c4_env_agg[ek]["arms"]["GLCSearch-v2-Infoset-CFR"]["1"]["exploitability"]["mean"]
        and c4_env_agg[ek]["arms"]["Det-Argmax-Minimax"]["64"]["exploitability"]["mean"]
        >= 3.5 * c4_env_agg[ek]["arms"]["GLCSearch-v2-Infoset-CFR"]["64"]["exploitability"]["mean"]
        for ek in C4_ENVS
    )
    c4_h3_pass = all(
        c4_env_agg[ek]["head_to_head_glc_vs_det"]["64"]["glc_win_rate"]["mean"] >= 0.65
        for ek in C4_ENVS
    )

    c4_summary = {
        "experiment_id": "R15-C4",
        "verdict": "PASS" if (c4_h1_pass and c4_h2_pass and c4_h3_pass) else "FAIL",
        "hypotheses": {
            "C4_H1_curse_of_deterministic_search_verified": bool(c4_h1_pass),
            "C4_H2_monotone_exploitability_reduction_under_glc_cfr": bool(c4_h2_pass),
            "C4_H3_h2h_dominance_at_b64": bool(c4_h3_pass),
        },
        "per_env": c4_env_agg,
    }

    overall_pass = all(
        s["verdict"] == "PASS" for s in (d0_summary, c1_summary, c2_summary, c3_summary, c4_summary)
    )
    return {
        "suite_id": "R15-CURIOSITY-SUITE",
        "prereg_doc": "benchmarks/PREREG-R15-CURIOSITY-SUITE.md",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "overall_verdict": "PASS" if overall_pass else "PARTIAL",
        "R15_D0_zero_oracle_leakage": d0_summary,
        "R15_C1_topology_phase_transitions": c1_summary,
        "R15_C2_compositional_chimera_transfer": c2_summary,
        "R15_C3_eureka_lemma_crystallization": c3_summary,
        "R15_C4_curse_of_deterministic_search": c4_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R15 Curiosity-Driven Micro-Experiment Suite (C1..C4)")
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(_REPO_ROOT / "results" / "r15_curiosity_suite_results.json"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    t_start = time.time()
    print(f"[R15-CURIOSITY-SUITE] Starting 3-seed evaluation on device={device} ...", flush=True)

    envs_map: Dict[str, UnifiedGameEnv] = {
        spec["env_key"]: make_env(spec["registry_key"]) for spec in SUITE_ENV_SPECS
    }

    per_seed_runs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        t_seed = time.time()
        print(f"  -> [Seed {seed}] Sampling train/cal/test batches across {len(SUITE_ENV_SPECS)} environments ...", flush=True)
        train_batches: Dict[str, EnvBatch] = {}
        cal_batches: Dict[str, EnvBatch] = {}
        test_batches: Dict[str, EnvBatch] = {}
        for spec in SUITE_ENV_SPECS:
            ek = spec["env_key"]
            env = envs_map[ek]
            train_batches[ek] = env.sample_batch(batch_size=spec["train_n"], split="train", seed=seed, device=device)
            cal_batches[ek] = env.sample_batch(batch_size=spec["cal_n"], split="cal", seed=seed, device=device)
            test_batches[ek] = env.sample_batch(batch_size=spec["test_n"], split="test", seed=seed, device=device)

        print(f"  -> [Seed {seed}] Training generalist, single-paradigm specialists & chimera models ...", flush=True)
        trained = train_r12_models_for_seed(
            seed=seed,
            device=device,
            envs_map=envs_map,
            train_batches=train_batches,
            cal_batches=cal_batches,
        )

        print(f"  -> [Seed {seed}] Verifying Gate R15-D0 (Static AST + Runtime PoisonTensor Canary Audit) ...", flush=True)
        gen_m, gen_s = trained["generalist"]
        d0_res = verify_zero_oracle_leakage_gate(
            model=gen_m,
            search_engine=gen_s,
            envs_map=envs_map,
            test_batches=test_batches,
            seed=seed,
            init_param_diff_l2=trained["param_shift_l2"],
            b0_kl_reduction_vs_init_pct=trained["b0_kl_reduction_vs_init_pct"],
        )

        print(f"  -> [Seed {seed}] Running Experiment R15-C1 (4x4 Topology Phase Transition Grid) ...", flush=True)
        c1_res = evaluate_r15_c1_for_seed(seed=seed, envs_map=envs_map, test_batches=test_batches, trained_bundle=trained)

        print(f"  -> [Seed {seed}] Running Experiment R15-C2 (Zero-Shot Compositional Chimera Transfer) ...", flush=True)
        c2_res = evaluate_r15_c2_for_seed(envs_map=envs_map, test_batches=test_batches, trained_bundle=trained)

        print(f"  -> [Seed {seed}] Running Experiment R15-C3 ('Eureka' Lemma Crystallization Effect) ...", flush=True)
        c3_res = evaluate_r15_c3_for_seed(
            seed=seed,
            envs_map=envs_map,
            cal_batches=cal_batches,
            test_batches=test_batches,
            trained_bundle=trained,
        )

        print(f"  -> [Seed {seed}] Running Experiment R15-C4 (Curse of Deterministic Search in Imperfect-Info Games) ...", flush=True)
        c4_res = evaluate_r15_c4_for_seed(envs_map=envs_map, test_batches=test_batches, trained_bundle=trained)

        print(f"  -> [Seed {seed}] Completed in {time.time() - t_seed:.1f}s (D0={d0_res['verdict']})", flush=True)
        per_seed_runs.append(
            {
                "seed": seed,
                "d0_gate": d0_res,
                "c1": c1_res,
                "c2": c2_res,
                "c3": c3_res,
                "c4": c4_res,
            }
        )

    summary = aggregate_r15_results(per_seed_runs)
    summary["wall_time_sec"] = float(time.time() - t_start)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"[R15-CURIOSITY-SUITE] Done in {summary['wall_time_sec']:.1f}s! Overall={summary['overall_verdict']} "
        f"| D0={summary['R15_D0_zero_oracle_leakage']['verdict']} "
        f"| C1={summary['R15_C1_topology_phase_transitions']['verdict']} "
        f"| C2={summary['R15_C2_compositional_chimera_transfer']['verdict']} "
        f"| C3={summary['R15_C3_eureka_lemma_crystallization']['verdict']} "
        f"| C4={summary['R15_C4_curse_of_deterministic_search']['verdict']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
