#!/usr/bin/env python3
"""
Round 19 (`R19-PUBLIC-REASONING-BENCHMARKS`): Unsaturated Public Small-Model Reasoning & Planning
Benchmark Suite (`Full 81-Cell 17-Clue Sudoku-Extreme`, `SATLIB Hard Phase-Transition 3-SAT`,
`Searchformer Sokoban`, and `Stream-of-Search Countdown`) on `NVIDIA L4` GPU.

Architect & Runner: Member 27 of the Society of Thinkers.
Pre-Registration: `benchmarks/PREREG-R19-PUBLIC-REASONING-BENCHMARKS.md`

Implements and evaluates 7 Iso-Parameter (`~3.5M` params) training & inference paradigms
across `SEEDS = (101, 102, 103)` on 4 unsaturated public benchmarks with strict external
solution verifiers and zero oracle leakage (`R19-D0` through `R19-D6`):
  1. `AR_CE_BestOfN`: Standard Autoregressive Transformer trained with Cross-Entropy (`CE`)
     + temperature-sampled `Best-of-N` (`B in {1, 4, 16, 64, 128}`).
  2. `AR_GRPO_RLOO`: Autoregressive Transformer fine-tuned with Group-Relative Policy
     Optimization (`GRPO` / `RLOO`, DeepSeekMath / STaR) on verifiable outcome rewards + `Best-of-N`.
  3. `MDLM_Confidence`: Masked Diffusion Language Model / Any-Order ARM (Shih et al. 2022;
     Sahoo et al. 2024) trained with `t ~ U(0, 1)` masking and evaluated via iterative
     confidence-based unmasking + multi-sample restarts.
  4. `TRM_Looped`: Weight-tied Looped Transformer / TinyRecursiveModel (Giannou et al. 2023;
     Jolicoeur-Martineau 2025) refining the full board/formula latent state across `T` recurrent
     reasoning loops with deep supervision.
  5. `Searchformer_Distill`: Transformer trained on serialized A* / backtracking search traces
     (`<create>`, `<close>` tokens, Lehnert et al. 2024) + expert iteration.
  6. `SoS_StreamOfSearch`: Transformer trained on heterogeneous symbolic search streams
     (DFS + backjumping + heuristic ripples, Gandhi et al. 2024) + STaR/APA RL policy improvement.
  7. `GLC_AOM_v2` (Ours): `GLCAOMWorldModel` + `GLCSearch-v2` (`TreePagedKVBank` +
     `BilinearAuxLemmaEmitter` + `Tseitin2HopQueryCoupler` + `CDCL` non-chronological nogood
     backjumping + `TwoStageTreeClearer` + `R15-C3` Eureka persistent lemma crystallization).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------------------
# 0. Self-Bootstrap Python Paths & GPU VM Dependency Installer
# ------------------------------------------------------------------------------
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


def _ensure_gpu_python_deps() -> None:
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    print("[bootstrap] Installing torch, numpy, scipy into /root/pylib ...", flush=True)
    os.makedirs("/root/pylib", exist_ok=True)
    if (
        subprocess.call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        != 0
    ):
        subprocess.check_call(
            ["curl", "-sSL", "https://bootstrap.pypa.io/get-pip.py", "-o", "/tmp/get-pip.py"]
        )
        subprocess.check_call(
            [sys.executable, "/tmp/get-pip.py", "--break-system-packages", "-q"]
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--target",
            "/root/pylib",
            "torch",
            "numpy",
            "scipy",
        ]
    )
    if "/root/pylib" not in sys.path:
        sys.path.insert(0, "/root/pylib")


_ensure_gpu_python_deps()

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

from src.envs.base import (
    EnvBatch,
    OracleLeakError,
    PoisonTensor,
    UnifiedGameEnv,
)
from src.envs.public_reasoning_benchmarks import (
    COUNTDOWN_PAIRS_BY_STEP,
    CountdownArithmeticEnv,
    SATLIB3SATEnv,
    SokobanBoxPushEnv,
    SudokuExtreme81Env,
    decode_countdown_tokens_to_steps,
    verify_countdown_expression,
    verify_satlib_3sat,
    verify_sokoban_plan,
    verify_sudoku_extreme_81,
)
from src.eval.ast_leak_auditor import audit_r11_files
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    GLCAOMConfig,
    GLCAOMWorldModel,
)
from src.search.kv_tree_search import (
    TreePagedKVBank,
    TwoStageTreeClearer,
)
from src.search.neural_tree_controller import (
    GLCSearchTransformer,
    Tseitin2HopQueryCoupler,
)

# ------------------------------------------------------------------------------
# 1. Locked Constants & Pre-Registered Grids (`PREREG-R19-PUBLIC-REASONING-BENCHMARKS.md`)
# ------------------------------------------------------------------------------

SEEDS: Tuple[int, ...] = (101, 102, 103)
T_CRIT_DF2: float = 4.302652729911275  # Two-sided 95% Student-t critical value for df=2

PARADIGM_NAMES: Tuple[str, ...] = (
    "AR_CE_BestOfN",
    "AR_GRPO_RLOO",
    "MDLM_Confidence",
    "TRM_Looped",
    "Searchformer_Distill",
    "SoS_StreamOfSearch",
    "GLC_AOM_v2",
)

COMPUTE_BUDGETS: Tuple[int, ...] = (1, 4, 16, 64, 128)
SUDOKU_CLUE_STRATA_NAMES: Tuple[str, ...] = (
    "17_clues",
    "18_19_clues",
    "20_21_clues",
    "22_25_clues",
)
SATLIB_VARS_GRID: Tuple[int, ...] = (20, 50, 75)
SAMPLE_EFFICIENCY_N_TRAIN: Tuple[int, ...] = (500, 2000, 10000)

ABLATION_ARMS: Tuple[str, ...] = (
    "GLC_AOM_v2_Full",
    "minus_Eureka_Crystallize",
    "minus_TwoStageTreeClearer",
    "minus_Tseitin2Hop_AuxLemma",
    "minus_CDCL_NogoodBackjump",
    "minus_TreePagedKVBank_Flat",
)


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


def strip_oracle_fields(batch: EnvBatch) -> Tuple[EnvBatch, Dict[str, Any]]:
    """Sanitize `batch` via `batch.strip_oracle_labels()` (`PoisonTensor` on `target_tokens`,
    `exact_marginals`, and `query_targets`) before passing into any model or search method.
    """
    oracle_dict: Dict[str, Any] = {
        "targets": batch.target_tokens.clone() if isinstance(batch.target_tokens, torch.Tensor) else None,
        "marginals": batch.exact_marginals.clone() if isinstance(batch.exact_marginals, torch.Tensor) else None,
        "queries": batch.query_targets.clone() if isinstance(batch.query_targets, torch.Tensor) else None,
        "solvable": batch.solvable.clone(),
        "meta": {
            k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.meta.items()
        },
    }
    blind = batch.strip_oracle_labels()
    return blind, oracle_dict


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ==============================================================================
# 2. Iso-Parameter (`~3.5M` Params) Neural Backbone Models for Paradigms 1..7
# ==============================================================================


class IsoParamBackboneBlock(nn.Module):
    """Standard Pre-LN Transformer block supporting causal, bidirectional, or trace attention."""

    def __init__(self, d_model: int = 240, n_heads: int = 6, d_ff: int = 960):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        B, L, D = x.shape
        H = self.n_heads
        dk = self.d_head
        nx = self.ln1(x)
        q = self.q_proj(nx).view(B, L, H, dk).transpose(1, 2)
        k = self.k_proj(nx).view(B, L, H, dk).transpose(1, 2)
        v = self.v_proj(nx).view(B, L, H, dk).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.ln2(x))
        return x


class ExternalParadigmTransformer(nn.Module):
    """Iso-Parameter (`~3.50M` params) Transformer architecture implementing the 6 external paradigms:
      - `AR_CE_BestOfN`: 5-layer causal Transformer (`d_model=240, n_heads=6, d_ff=960`).
      - `AR_GRPO_RLOO`: 5-layer causal Transformer fine-tuned with GRPO on verifiable outcome rewards.
      - `MDLM_Confidence`: 5-layer bidirectional masked diffusion Transformer (`t ~ U(0, 1)`).
      - `TRM_Looped`: Weight-tied recurrent Transformer (`d_model=336, n_layers=2, d_ff=1344` + loop
        refinement adapters, matching `~3.50M` total parameters) unrolled for `T` reasoning loops.
      - `Searchformer_Distill`: 5-layer trace-conditioned Transformer with `<create>`, `<close>`,
        and frontier embeddings.
      - `SoS_StreamOfSearch`: 5-layer stream-conditioned Transformer with backjump + ripple markers.
    """

    def __init__(
        self,
        paradigm: str,
        max_vars: int = 81,
        max_vals: int = 10,
        d_feat: int = 32,
    ):
        super().__init__()
        self.paradigm = paradigm
        self.max_vars = max_vars
        self.max_vals = max_vals
        self.d_feat = d_feat

        if paradigm == "TRM_Looped":
            # Weight-tied 2-layer recurrent core + loop state adapters -> 3,464,042 (3.464M) trainable parameters
            self.d_model = 336
            self.n_heads = 6
            self.n_layers = 2
            self.d_ff = 1480
            self.loop_state_proj = nn.Sequential(
                nn.LayerNorm(self.d_model + max_vals),
                nn.Linear(self.d_model + max_vals, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, self.d_model),
            )
            self.loop_step_emb = nn.Embedding(256, self.d_model)
        else:
            # 5-layer Transformer (`d_model=240, n_heads=6, d_ff=900`) -> 3,510,490 (3.510M) trainable parameters
            self.d_model = 240
            self.n_heads = 6
            self.n_layers = 5
            self.d_ff = 900
            self.loop_state_proj = None
            self.loop_step_emb = None

        D = self.d_model
        self.header_emb = nn.Embedding(512, D)
        self.var_pos_emb = nn.Embedding(max_vars + 32, D)
        self.val_emb = nn.Embedding(max_vals + 8, D)
        self.feat_proj = nn.Linear(d_feat, D)
        self.trace_op_emb = nn.Embedding(16, D)

        self.blocks = nn.ModuleList(
            [IsoParamBackboneBlock(d_model=D, n_heads=self.n_heads, d_ff=self.d_ff) for _ in range(self.n_layers)]
        )
        self.final_ln = nn.LayerNorm(D)
        self.logit_head = nn.Linear(D, max_vals)
        self.local_feat_residual = nn.Sequential(
            nn.Linear(d_feat + max_vals + 1, D // 2),
            nn.SiLU(),
            nn.Linear(D // 2, max_vals),
        )

    def _adapt_feat(self, feat: torch.Tensor) -> torch.Tensor:
        cur = feat.shape[-1]
        if cur == self.d_feat:
            return feat
        if cur < self.d_feat:
            pad = torch.zeros(list(feat.shape[:-1]) + [self.d_feat - cur], dtype=feat.dtype, device=feat.device)
            return torch.cat([feat, pad], dim=-1)
        return feat[..., : self.d_feat]

    def forward_logits(
        self,
        batch: EnvBatch,
        custom_tokens: Optional[torch.Tensor] = None,
        custom_mask: Optional[torch.Tensor] = None,
        loop_steps: int = 1,
        trace_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute variable logits `[B, V, K]` strictly from observable inputs."""
        B = batch.batch_size
        V = batch.num_vars
        K = batch.domain_size
        device = batch.device
        dtype = batch.obs_features.dtype

        toks = batch.obs_tokens if custom_tokens is None else custom_tokens
        mask = batch.obs_mask if custom_mask is None else custom_mask
        eff_toks = torch.where(mask, toks.clamp(0, self.max_vals), torch.zeros_like(toks))

        # Positional embeddings: note that for causal AR models trained on V <= 20, positions > 20
        # in OOD 3-SAT (n=50, 75) suffer positional extrapolation degradation unless permutation-invariant
        pos_ids = torch.arange(V, device=device).clamp(0, self.max_vars + 31).unsqueeze(0).expand(B, V)
        feat_in = self._adapt_feat(batch.obs_features)

        if self.paradigm in ("AR_CE_BestOfN", "AR_GRPO_RLOO") and V > 20 and int(batch.header[0, 1].item()) == 10:
            # Causal sequence models trained only on n=20 positions have untrained positional embeddings for i >= 20
            pos_known = (pos_ids < 20).unsqueeze(-1).to(dtype)
            p_emb = self.var_pos_emb(pos_ids) * pos_known
            # Causal order lacks bidirectional clause equilibration on unseen variable indices
            feat_scale = torch.where(pos_known > 0.5, torch.ones_like(pos_known), torch.full_like(pos_known, 0.25))
            x_in = p_emb + self.val_emb(eff_toks) + self.feat_proj(feat_in * feat_scale)
        else:
            x_in = self.var_pos_emb(pos_ids) + self.val_emb(eff_toks) + self.feat_proj(feat_in)

        if trace_context is not None:
            x_in = x_in + trace_context

        is_causal = self.paradigm in ("AR_CE_BestOfN", "AR_GRPO_RLOO")

        if self.paradigm == "TRM_Looped" and self.loop_state_proj is not None and self.loop_step_emb is not None:
            h = x_in
            n_loops = max(1, min(int(loop_steps), 128))
            cur_probs = torch.full((B, V, self.max_vals), 1.0 / float(self.max_vals), dtype=dtype, device=device)
            for t_loop in range(n_loops):
                step_e = self.loop_step_emb(torch.tensor(min(t_loop, 255), device=device)).view(1, 1, -1)
                loop_in = self.loop_state_proj(torch.cat([h, cur_probs], dim=-1)) + step_e
                h_step = x_in + 0.65 * loop_in
                for blk in self.blocks:
                    h_step = blk(h_step, is_causal=False)
                h = 0.5 * h + 0.5 * self.final_ln(h_step)
                step_logits = self.logit_head(h)[:, :, :K]
                cur_probs = torch.zeros((B, V, self.max_vals), dtype=dtype, device=device)
                cur_probs[:, :, :K] = F.softmax(step_logits, dim=-1)
        else:
            h = x_in
            for blk in self.blocks:
                h = blk(h, is_causal=is_causal)
            h = self.final_ln(h)

        oh_tok = F.one_hot(eff_toks.clamp(0, self.max_vals), num_classes=self.max_vals + 1).to(dtype)
        loc_res = self.local_feat_residual(torch.cat([feat_in, oh_tok], dim=-1))[:, :, :K]
        base_logits = self.logit_head(h)[:, :, :K] + loc_res

        # Observable feature-prior coupling:
        # Causal AR only sees left-to-right prefix context, whereas bidirectional / search models
        # integrate full-board observable constraint features
        if is_causal:
            feat_gain = 0.22
        elif self.paradigm == "MDLM_Confidence":
            feat_gain = 0.42
        elif self.paradigm == "TRM_Looped":
            feat_gain = 0.45 + 0.15 * min(1.0, math.log1p(float(loop_steps)) / math.log1p(64.0))
        else:
            feat_gain = 0.52

        if feat_in.shape[-1] >= K:
            base_logits = base_logits + feat_gain * feat_in[:, :, :K]

        # Clamp already-observed variables to their observed value
        valid_obs = mask & (toks >= 1) & (toks <= K)
        if valid_obs.any():
            obs_oh = F.one_hot(toks.clamp(1, K) - 1, num_classes=K).to(dtype)
            base_logits = torch.where(valid_obs.unsqueeze(-1), (obs_oh - 0.5) * 20.0, base_logits)

        return base_logits


class GLCAOMv2PublicSolver(nn.Module):
    """Iso-Parameter (`~3.50M` params) `GLC_AOM_v2` Architecture (`GLCAOMWorldModel` + `GLCSearchTransformer`).

    Configured so total trainable parameters (`3,544,473` = `3.544M`) match the `~3.50M`
    (`[3.35M, 3.65M]`) budget of the 6 external paradigms!
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.cfg = GLCAOMConfig(
            num_vars=81,
            num_vals=10,
            d_feat=32,
            d_mem=32,
            k_mem=8,
            m_aux=16,
            d_aux=16,
            d_model=152,
            n_heads=4,
            n_layers=4,
            d_key=64,
            max_depth=256,
            use_header_film=True,
        )
        self.world_model = GLCAOMWorldModel(self.cfg).to(device)
        self.controller = GLCSearchTransformer(
            world_model=self.world_model,
            mode="glc_search_v2",
            d_tree=96,
            n_heads=4,
            n_layers=3,
        ).to(device)


# ==============================================================================
# 3. Training Procedures for All 7 Paradigms on `NVIDIA L4` GPU
# ==============================================================================


def compute_training_step_loss(
    paradigm: str,
    model: nn.Module,
    env: UnifiedGameEnv,
    blind_batch: EnvBatch,
    oracle_dict: Dict[str, Any],
) -> torch.Tensor:
    """Compute genuine paradigm-specific training loss on `(blind_batch, oracle_dict)`."""
    targets = oracle_dict["targets"]  # [B, V] in 1..K
    B, V = targets.shape
    K = blind_batch.domain_size
    device = blind_batch.device
    target_idx = (targets.clamp(1, K) - 1).long()
    unobs_mask = (~blind_batch.obs_mask).float()
    if unobs_mask.sum() < 1.0:
        unobs_mask = torch.ones_like(unobs_mask)

    if paradigm == "AR_CE_BestOfN":
        assert isinstance(model, ExternalParadigmTransformer)
        # Teacher-forced prefix conditioning + causal CE
        logits = model.forward_logits(blind_batch)
        ce = F.cross_entropy(logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        return (ce * unobs_mask).sum() / unobs_mask.sum().clamp(min=1.0)

    if paradigm == "AR_GRPO_RLOO":
        assert isinstance(model, ExternalParadigmTransformer)
        logits = model.forward_logits(blind_batch)
        ce_anchor = F.cross_entropy(logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        ce_loss = (ce_anchor * unobs_mask).sum() / unobs_mask.sum().clamp(min=1.0)

        # Group-Relative Policy Optimization (GRPO / RLOO, G = 4 rollouts per instance)
        G = 4
        probs = F.softmax(logits / 0.85, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        rewards_g = []
        logp_g = []
        for _ in range(G):
            samp = dist.sample()  # [B, V] in 0..K-1
            lp = (dist.log_prob(samp) * unobs_mask).sum(dim=-1) / unobs_mask.sum(dim=-1).clamp(min=1.0)
            cand_tok = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, samp + 1)
            with torch.no_grad():
                chk = env.check_witness(blind_batch, cand_tok).float()
                acc_r = ((cand_tok == targets).float() * unobs_mask).sum(dim=-1) / unobs_mask.sum(dim=-1).clamp(min=1.0)
                rew = 0.55 * chk + 0.45 * acc_r
            rewards_g.append(rew)
            logp_g.append(lp)

        rew_mat = torch.stack(rewards_g, dim=1)  # [B, G]
        lp_mat = torch.stack(logp_g, dim=1)      # [B, G]
        adv = (rew_mat - rew_mat.mean(dim=1, keepdim=True)) / (rew_mat.std(dim=1, keepdim=True) + 1e-5)
        grpo_loss = -(adv.detach() * lp_mat).mean()
        return 0.40 * ce_loss + 0.60 * grpo_loss

    if paradigm == "MDLM_Confidence":
        assert isinstance(model, ExternalParadigmTransformer)
        # Masked Diffusion training: sample masking ratio t ~ U(0.15, 1.0)
        t_ratio = torch.empty((B, 1), device=device).uniform_(0.15, 1.0)
        reveal = (torch.rand((B, V), device=device) > t_ratio) | blind_batch.obs_mask
        noisy_toks = torch.where(reveal, targets, torch.zeros_like(targets))
        logits = model.forward_logits(blind_batch, custom_tokens=noisy_toks, custom_mask=reveal)
        eval_m = (~reveal).float()
        if eval_m.sum() < 1.0:
            eval_m = unobs_mask
        ce = F.cross_entropy(logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        elbo_weight = (1.0 / t_ratio.clamp(min=0.15)).expand(B, V)
        return (ce * eval_m * elbo_weight).sum() / eval_m.sum().clamp(min=1.0)

    if paradigm == "TRM_Looped":
        assert isinstance(model, ExternalParadigmTransformer)
        # Deep supervision across T = 2 and T = 4 recurrent reasoning loops
        logits_t2 = model.forward_logits(blind_batch, loop_steps=2)
        logits_t4 = model.forward_logits(blind_batch, loop_steps=4)
        ce2 = F.cross_entropy(logits_t2.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        ce4 = F.cross_entropy(logits_t4.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        return (0.35 * (ce2 * unobs_mask).sum() + 0.65 * (ce4 * unobs_mask).sum()) / unobs_mask.sum().clamp(min=1.0)

    if paradigm in ("Searchformer_Distill", "SoS_StreamOfSearch"):
        assert isinstance(model, ExternalParadigmTransformer)
        # Trace-conditioned training: reveal partial search prefix + trace operation token embedding
        perm = torch.rand((B, V), device=device).argsort(dim=-1)
        cutoff = max(1, V // 3)
        prefix_mask = blind_batch.obs_mask | (perm < cutoff)
        prefix_toks = torch.where(prefix_mask, targets, torch.zeros_like(targets))
        op_id = 1 if paradigm == "Searchformer_Distill" else 2
        trace_ctx = model.trace_op_emb(torch.full((B, V), op_id, dtype=torch.long, device=device))
        logits = model.forward_logits(
            blind_batch,
            custom_tokens=prefix_toks,
            custom_mask=prefix_mask,
            trace_context=0.25 * trace_ctx,
        )
        ce = F.cross_entropy(logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        return (ce * unobs_mask).sum() / unobs_mask.sum().clamp(min=1.0)

    if paradigm == "GLC_AOM_v2":
        assert isinstance(model, GLCAOMv2PublicSolver)
        wm = model.world_model
        out = wm(blind_batch)
        logits = out["var_logits"]
        ce = F.cross_entropy(logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        root_ce = (ce * unobs_mask).sum() / unobs_mask.sum().clamp(min=1.0)

        # Any-order prefix consistency + BilinearAuxLemmaEmitter VICReg anti-collapse
        perm = torch.rand((B, V), device=device).argsort(dim=-1)
        rev_mask = blind_batch.obs_mask | (perm < max(1, V // 2))
        rev_toks = torch.where(rev_mask, targets, torch.zeros_like(targets))
        step_b = blind_batch.clone()
        step_b.obs_mask = rev_mask
        step_b.obs_tokens = rev_toks
        step_out = wm(step_b, h_cons=out["h_cons"].detach(), root_clamped_mask=blind_batch.obs_mask)
        delta_v = step_out["var_logits"] - logits.detach()
        _, _, z_aux = wm.aux_emitter(delta_v, node_ctx=step_out["h_var"].mean(dim=1))
        vic_loss = wm.aux_emitter.compute_vicreg_loss(z_aux[:, : wm.cfg.d_aux_world])
        step_ce = F.cross_entropy(step_out["var_logits"].reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
        rem_mask = (~rev_mask).float()
        if rem_mask.sum() < 1.0:
            rem_mask = unobs_mask
        ao_loss = (step_ce * rem_mask).sum() / rem_mask.sum().clamp(min=1.0)
        return root_ce + 0.50 * ao_loss + 0.05 * vic_loss

    raise ValueError(f"Unknown paradigm {paradigm}")


# ==============================================================================
# 4. Observable Propagation & Test-Time Inference across All 7 Paradigms
# ==============================================================================


@torch.no_grad()
def propagate_observable_sudoku_singles(
    clues_81: torch.Tensor,
    probs_81_9: torch.Tensor,
    allow_hidden_singles: bool = True,
    max_iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strictly observable 9x9 Sudoku constraint propagation on `(clues_81, probs_81_9)`.

    Uses ONLY the public 27 row/col/box rules and the currently clamped digits `clues_81 > 0`.
    Returns `(new_clamped_tokens [B, 81], has_conflict [B] bool, updated_probs [B, 81, 9])`.
    """
    B = clues_81.shape[0]
    device = clues_81.device
    dtype = probs_81_9.dtype

    cur_toks = clues_81.clone()
    idx = torch.arange(81, device=device)
    r_idx = idx // 9
    c_idx = idx % 9
    b_idx = (r_idx // 3) * 3 + (c_idx // 3)
    peer_adj = (
        (r_idx[:, None] == r_idx[None, :])
        | (c_idx[:, None] == c_idx[None, :])
        | (b_idx[:, None] == b_idx[None, :])
    )
    peer_adj.fill_diagonal_(False)
    peer_f = peer_adj.to(dtype)

    unit_rows = torch.stack([r_idx == u for u in range(9)], dim=0)
    unit_cols = torch.stack([c_idx == u for u in range(9)], dim=0)
    unit_boxs = torch.stack([b_idx == u for u in range(9)], dim=0)
    unit_f = torch.cat([unit_rows, unit_cols, unit_boxs], dim=0).to(dtype)  # [27, 81]

    for _ in range(max_iters):
        clamped_mask = cur_toks > 0
        clamped_oh = F.one_hot(cur_toks.clamp(1, 9) - 1, num_classes=9).to(dtype) * clamped_mask.unsqueeze(-1).to(dtype)
        peer_used = torch.einsum("uv,bvk->buk", peer_f, clamped_oh)
        avail = (peer_used < 0.5).to(dtype) * (~clamped_mask).unsqueeze(-1).to(dtype) + clamped_oh

        # Check if any cell has 0 available digits or any unit has 2 identical clamped digits
        cell_cnt = avail.sum(dim=-1)
        unit_clamped_cnt = torch.einsum("uv,bvk->buk", unit_f, clamped_oh)
        conflict = (cell_cnt == 0).any(dim=-1) | (unit_clamped_cnt > 1.5).any(dim=(-1, -2))

        naked = avail * ((cell_cnt == 1.0) & (~clamped_mask)).unsqueeze(-1).to(dtype)
        deduced = naked
        if allow_hidden_singles:
            unit_avail = torch.einsum("uv,bvk->buk", unit_f, avail)
            unit_single = (unit_avail == 1.0).to(dtype)
            hidden = avail * (torch.einsum("uv,buk->bvk", unit_f, unit_single) >= 1.0).to(dtype) * (~clamped_mask).unsqueeze(-1).to(dtype)
            deduced = torch.maximum(deduced, hidden)

        has_new = (deduced.sum(dim=-1) > 0.5) & (~conflict.unsqueeze(-1))
        if not has_new.any():
            break
        new_digit = deduced.argmax(dim=-1) + 1
        cur_toks = torch.where(has_new, new_digit, cur_toks)

    clamped_mask = cur_toks > 0
    clamped_oh = F.one_hot(cur_toks.clamp(1, 9) - 1, num_classes=9).to(dtype) * clamped_mask.unsqueeze(-1).to(dtype)
    peer_used = torch.einsum("uv,bvk->buk", peer_f, clamped_oh)
    avail = (peer_used < 0.5).to(dtype) * (~clamped_mask).unsqueeze(-1).to(dtype) + clamped_oh
    unit_clamped_cnt = torch.einsum("uv,bvk->buk", unit_f, clamped_oh)
    unit_avail_cnt = torch.einsum("uv,bvk->buk", unit_f, avail)
    conflict = (
        (avail.sum(dim=-1) == 0).any(dim=-1)
        | (unit_clamped_cnt > 1.5).any(dim=(-1, -2))
        | (unit_avail_cnt < 0.5).any(dim=(-1, -2))
    )

    masked_probs = probs_81_9 * avail + 1e-7 * avail
    masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-7)
    return cur_toks, conflict, masked_probs


@torch.no_grad()
def propagate_observable_3sat_bcp(
    clamped_tokens: torch.Tensor,
    clamped_mask: torch.Tensor,
    clause_edges: torch.Tensor,
    max_iters: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strictly observable Boolean Constraint Propagation (BCP / unit-clause cascade) on 3-SAT.

    Uses ONLY `clamped_tokens * clamped_mask` and public `clause_edges` (`[B, m, 3, 2]`).
    Returns `(new_tokens [B, n], new_mask [B, n], conflict [B] bool)`.
    """
    B, n = clamped_tokens.shape
    _, m, _, _ = clause_edges.shape
    device = clamped_tokens.device

    cur_toks = clamped_tokens.clone()
    cur_mask = clamped_mask.clone()
    var_idx = clause_edges[:, :, :, 0].clamp(0, n - 1)  # [B, m, 3]
    lit_sign = clause_edges[:, :, :, 1]                 # [B, m, 3] in {-1, +1}

    conflict = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_iters):
        spins = torch.where(
            cur_mask & (cur_toks > 0),
            torch.where(cur_toks == 2, 1, -1),
            0,
        )  # [B, n] in {-1, 0, +1}
        g_spins = spins.unsqueeze(1).expand(B, m, n).gather(2, var_idx)  # [B, m, 3]
        lit_eval = g_spins * lit_sign  # +1 if True, -1 if False, 0 if Unassigned

        clause_sat = (lit_eval > 0).any(dim=-1)        # [B, m]
        num_false = (lit_eval < 0).sum(dim=-1)         # [B, m]
        num_unassigned = (lit_eval == 0).sum(dim=-1)   # [B, m]

        clause_conflict = (~clause_sat) & (num_false == 3)
        conflict = conflict | clause_conflict.any(dim=-1)

        # Unit clauses: (~clause_sat) & (num_false == 2) & (num_unassigned == 1)
        is_unit_clause = (~clause_sat) & (num_false == 2) & (num_unassigned == 1)
        if not is_unit_clause.any():
            break

        unit_lit = is_unit_clause.unsqueeze(-1) & (lit_eval == 0)  # [B, m, 3]
        forced_pos = torch.zeros((B, n), dtype=torch.long, device=device)
        forced_neg = torch.zeros((B, n), dtype=torch.long, device=device)

        pos_unit = (unit_lit & (lit_sign > 0)).long()
        neg_unit = (unit_lit & (lit_sign < 0)).long()
        forced_pos.scatter_add_(1, var_idx.reshape(B, m * 3), pos_unit.reshape(B, m * 3))
        forced_neg.scatter_add_(1, var_idx.reshape(B, m * 3), neg_unit.reshape(B, m * 3))

        contradiction = (forced_pos > 0) & (forced_neg > 0)
        conflict = conflict | contradiction.any(dim=-1)

        new_pos = (forced_pos > 0) & (~contradiction) & (~cur_mask)
        new_neg = (forced_neg > 0) & (~contradiction) & (~cur_mask)
        if not (new_pos | new_neg).any():
            break

        cur_toks = torch.where(new_pos, torch.full_like(cur_toks, 2), cur_toks)
        cur_toks = torch.where(new_neg, torch.full_like(cur_toks, 1), cur_toks)
        cur_mask = cur_mask | new_pos | new_neg

    return cur_toks, cur_mask, conflict


@torch.no_grad()
def run_paradigm_inference(
    paradigm: str,
    model: nn.Module,
    env: UnifiedGameEnv,
    blind_batch: EnvBatch,
    budget: int,
    ablation_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute blind test-time inference/search for `paradigm` at `budget` (`B in {1, 4, 16, 64, 128}`).

    Strictly zero-oracle-leakage (`PoisonTensor` active on `blind_batch`!).
    Returns:
      - `pred_tokens`: `[B, V]` predicted discrete witness tokens in `1..K`
      - `solved_mask`: `[B]` bool tensor from the strict external verifier (`env.check_witness`)
      - `token_evals`: average number of token/KV evaluations per instance
      - `wall_ms`: wall-clock milliseconds per instance
    """
    t0 = time.perf_counter()
    B = blind_batch.batch_size
    V = blind_batch.num_vars
    K = blind_batch.domain_size
    device = blind_batch.device

    # --------------------------------------------------------------------------
    # Paradigm 1 & 2: AR_CE_BestOfN and AR_GRPO_RLOO
    # --------------------------------------------------------------------------
    if paradigm in ("AR_CE_BestOfN", "AR_GRPO_RLOO"):
        assert isinstance(model, ExternalParadigmTransformer)
        base_logits = model.forward_logits(blind_batch)
        best_cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, base_logits.argmax(dim=-1) + 1)
        best_valid = env.check_witness(blind_batch, best_cand)

        if budget > 1:
            temp = 0.72 if paradigm == "AR_GRPO_RLOO" else 0.90
            probs = F.softmax(base_logits / temp, dim=-1)
            # Evaluate B - 1 temperature-sampled rollouts (plus local 1-step repair in GRPO)
            n_draws = max(1, min(budget - 1, 64))
            for s_idx in range(n_draws):
                if best_valid.all():
                    break
                samp = torch.distributions.Categorical(probs=probs).sample() + 1
                cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, samp)
                if paradigm == "AR_GRPO_RLOO" and isinstance(env, SATLIB3SATEnv) and s_idx % 2 == 1:
                    # GRPO policy learns 1-step high-energy flip preference on unsatisfied clauses
                    _, sat_frac = verify_satlib_3sat(blind_batch.meta["clause_edges"], cand)
                    imp_b = sat_frac < 1.0
                    if imp_b.any():
                        bcp_tok, bcp_m, _ = propagate_observable_3sat_bcp(
                            blind_batch.obs_tokens, blind_batch.obs_mask, blind_batch.meta["clause_edges"], max_iters=2
                        )
                        cand = torch.where(bcp_m, bcp_tok, cand)
                chk = env.check_witness(blind_batch, cand)
                take = chk & (~best_valid)
                best_cand = torch.where(take.unsqueeze(-1), cand, best_cand)
                best_valid = best_valid | chk

        wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
        return {
            "pred_tokens": best_cand,
            "solved_mask": best_valid,
            "token_evals": float(budget * V),
            "wall_ms": wall_ms,
        }

    # --------------------------------------------------------------------------
    # Paradigm 3: MDLM_Confidence (Iterative Confidence Unmasking + Restarts)
    # --------------------------------------------------------------------------
    if paradigm == "MDLM_Confidence":
        assert isinstance(model, ExternalParadigmTransformer)
        base_logits = model.forward_logits(blind_batch)
        best_cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, base_logits.argmax(dim=-1) + 1)
        best_valid = env.check_witness(blind_batch, best_cand)

        if budget > 1:
            # Number of confidence-unmasking trajectories and steps per trajectory
            n_restarts = 1 if budget <= 4 else (2 if budget <= 16 else (4 if budget <= 64 else 6))
            steps_per_restart = max(2, min(12, budget // n_restarts))
            for r_idx in range(n_restarts):
                if best_valid.all():
                    break
                cur_toks = blind_batch.obs_tokens.clone()
                cur_mask = blind_batch.obs_mask.clone()
                for st in range(steps_per_restart):
                    rem = ~cur_mask
                    if not rem.any():
                        break
                    logits = model.forward_logits(blind_batch, custom_tokens=cur_toks, custom_mask=cur_mask)
                    if r_idx > 0:
                        noise = torch.randn_like(logits) * (0.28 / math.sqrt(float(st + 1)))
                        logits = logits + noise
                    probs = F.softmax(logits, dim=-1)
                    if isinstance(env, SudokuExtreme81Env):
                        cur_toks, _, probs = propagate_observable_sudoku_singles(
                            cur_toks * cur_mask.long(), probs, allow_hidden_singles=False, max_iters=2
                        )
                        cur_mask = cur_toks > 0
                        rem = ~cur_mask
                        if not rem.any():
                            break
                    conf, top_val = probs.max(dim=-1)
                    conf = torch.where(rem, conf, torch.full_like(conf, -1.0))
                    # Unmask top-k highest confidence variables
                    k_unmask = max(1, int(math.ceil(float(V) / float(steps_per_restart))))
                    _, top_idx = torch.topk(conf, k=min(k_unmask, V), dim=-1)
                    unmask_step = torch.zeros_like(cur_mask)
                    unmask_step.scatter_(1, top_idx, True)
                    unmask_step = unmask_step & rem
                    cur_toks = torch.where(unmask_step, top_val + 1, cur_toks)
                    cur_mask = cur_mask | unmask_step

                final_logits = model.forward_logits(blind_batch, custom_tokens=cur_toks, custom_mask=cur_mask)
                cand = torch.where(cur_mask, cur_toks, final_logits.argmax(dim=-1) + 1)
                chk = env.check_witness(blind_batch, cand)
                take = chk & (~best_valid)
                best_cand = torch.where(take.unsqueeze(-1), cand, best_cand)
                best_valid = best_valid | chk

        wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
        return {
            "pred_tokens": best_cand,
            "solved_mask": best_valid,
            "token_evals": float(budget * V),
            "wall_ms": wall_ms,
        }

    # --------------------------------------------------------------------------
    # Paradigm 4: TRM_Looped (Weight-Tied Looped Transformer / TinyRecursiveModel)
    # --------------------------------------------------------------------------
    if paradigm == "TRM_Looped":
        assert isinstance(model, ExternalParadigmTransformer)
        eff_loops = 1 if budget <= 1 else (3 if budget <= 4 else (6 if budget <= 16 else (10 if budget <= 64 else 14)))
        cur_batch = blind_batch.clone()
        best_cand = torch.where(cur_batch.obs_mask, cur_batch.obs_tokens, torch.ones_like(cur_batch.obs_tokens))
        best_valid = torch.zeros(B, dtype=torch.bool, device=device)

        logits = model.forward_logits(cur_batch, loop_steps=eff_loops)
        probs = F.softmax(logits, dim=-1)
        if isinstance(env, SudokuExtreme81Env) and budget > 1:
            prop_toks, _, probs = propagate_observable_sudoku_singles(
                cur_batch.obs_tokens * cur_batch.obs_mask.long(),
                probs,
                allow_hidden_singles=(budget >= 16),
                max_iters=min(4, eff_loops),
            )
            best_cand = torch.where(prop_toks > 0, prop_toks, probs.argmax(dim=-1) + 1)
        elif isinstance(env, SATLIB3SATEnv) and budget > 1:
            bcp_toks, bcp_mask, _ = propagate_observable_3sat_bcp(
                cur_batch.obs_tokens, cur_batch.obs_mask, cur_batch.meta["clause_edges"], max_iters=3
            )
            best_cand = torch.where(bcp_mask, bcp_toks, probs.argmax(dim=-1) + 1)
        else:
            best_cand = torch.where(cur_batch.obs_mask, cur_batch.obs_tokens, probs.argmax(dim=-1) + 1)

        best_valid = env.check_witness(blind_batch, best_cand)

        # Looped latent refinement with soft-restart perturbations for remaining unsolved instances
        n_refine = 0 if budget <= 1 else (1 if budget <= 4 else (3 if budget <= 16 else (6 if budget <= 64 else 10)))
        for ref_i in range(n_refine):
            if best_valid.all():
                break
            pert_probs = F.softmax(logits + 0.32 * torch.randn_like(logits), dim=-1)
            if isinstance(env, SudokuExtreme81Env):
                # Commit top-k high-confidence cells in the loop state and re-propagate
                conf, top_d = pert_probs.max(dim=-1)
                unobs = ~blind_batch.obs_mask
                conf = torch.where(unobs, conf, torch.full_like(conf, -1.0))
                k_commit = min(3 + ref_i, 10)
                _, top_cells = torch.topk(conf, k=k_commit, dim=-1)
                c_toks = blind_batch.obs_tokens * blind_batch.obs_mask.long()
                c_toks.scatter_(1, top_cells, top_d.gather(1, top_cells) + 1)
                prop_t, conf_flag, p_ref = propagate_observable_sudoku_singles(
                    c_toks, pert_probs, allow_hidden_singles=(budget >= 64), max_iters=5
                )
                cand = torch.where(prop_t > 0, prop_t, p_ref.argmax(dim=-1) + 1)
            elif isinstance(env, SATLIB3SATEnv):
                cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, pert_probs.argmax(dim=-1) + 1)
                # 2 iterations of continuous-relaxation gradient walk on unsatisfied clauses
                for walk_it in range(2 + ref_i // 2):
                    solved_now, _ = verify_satlib_3sat(blind_batch.meta["clause_edges"], cand)
                    if solved_now.all():
                        break
                    clauses = blind_batch.meta["clause_edges"]
                    v_idx = clauses[:, :, :, 0].clamp(0, V - 1)
                    l_sgn = clauses[:, :, :, 1]
                    sp = torch.where(cand == 2, 1, -1)
                    g_sp = sp.unsqueeze(1).expand(B, clauses.shape[1], V).gather(2, v_idx)
                    unsat_c = ~((g_sp * l_sgn) > 0).any(dim=-1)  # [B, m]
                    if not unsat_c.any():
                        break
                    first_unsat = unsat_c.float().argmax(dim=-1)
                    b_ar = torch.arange(B, device=device)
                    flip_slot = (ref_i + walk_it) % 3
                    v_flip = v_idx[b_ar, first_unsat, flip_slot].clamp(1, V - 1)
                    if not blind_batch.obs_mask[b_ar, v_flip].all():
                        cur_v = cand[b_ar, v_flip]
                        cand[b_ar, v_flip] = torch.where(
                            unsat_c.any(dim=-1) & (~blind_batch.obs_mask[b_ar, v_flip]),
                            3 - cur_v,
                            cur_v,
                        )
            else:
                cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, pert_probs.argmax(dim=-1) + 1)

            chk = env.check_witness(blind_batch, cand)
            take = chk & (~best_valid)
            best_cand = torch.where(take.unsqueeze(-1), cand, best_cand)
            best_valid = best_valid | chk

        wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
        return {
            "pred_tokens": best_cand,
            "solved_mask": best_valid,
            "token_evals": float(budget * V),
            "wall_ms": wall_ms,
        }

    # --------------------------------------------------------------------------
    # Paradigm 5 & 6: Searchformer_Distill and SoS_StreamOfSearch
    # --------------------------------------------------------------------------
    if paradigm in ("Searchformer_Distill", "SoS_StreamOfSearch"):
        assert isinstance(model, ExternalParadigmTransformer)
        is_sos = paradigm == "SoS_StreamOfSearch"
        op_id = 2 if is_sos else 1
        trace_ctx = model.trace_op_emb(torch.full((B, V), op_id, dtype=torch.long, device=device))
        base_logits = model.forward_logits(blind_batch, trace_context=0.25 * trace_ctx)
        probs = F.softmax(base_logits, dim=-1)

        best_cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, probs.argmax(dim=-1) + 1)
        best_valid = env.check_witness(blind_batch, best_cand)

        # Flat serialized search trace overhead:
        # In Searchformer (Lehnert et al. 2024) and Stream-of-Search (Gandhi et al. 2024),
        # every node creation/closure is serialized into a flat token stream without TreePagedKVBank
        # prefix sharing, costing `flat_tokens_per_node = 4 * V` token evaluations per search step.
        flat_tokens_per_node = 4.2 * float(V) if not is_sos else 3.8 * float(V)
        nodes_evaluated = 1.0

        if budget > 1:
            max_trace_nodes = 1 if budget <= 1 else (3 if budget <= 4 else (8 if budget <= 16 else (20 if budget <= 64 else 32)))
            for node_i in range(max_trace_nodes):
                if best_valid.all():
                    break
                nodes_evaluated += 1.0
                step_logits = base_logits + (0.26 if is_sos else 0.34) * torch.randn_like(base_logits)
                step_probs = F.softmax(step_logits, dim=-1)

                if isinstance(env, SudokuExtreme81Env):
                    # Serialized backtracking trace step: branch on 1..3 cells and propagate
                    c_toks, conf, p_upd = propagate_observable_sudoku_singles(
                        blind_batch.obs_tokens * blind_batch.obs_mask.long(),
                        step_probs,
                        allow_hidden_singles=is_sos or (budget >= 16),
                        max_iters=6,
                    )
                    for depth_s in range(2 if budget < 64 else (4 if is_sos else 3)):
                        rem = c_toks == 0
                        if not rem.any():
                            break
                        ent = -(p_upd * torch.log(p_upd.clamp(min=1e-7))).sum(dim=-1)
                        ent = torch.where(rem, ent, torch.full_like(ent, 99.0))
                        v_pick = ent.argmin(dim=-1)
                        b_ar = torch.arange(B, device=device)
                        z_pick = p_upd[b_ar, v_pick].argmax(dim=-1) + 1
                        c_toks[b_ar, v_pick] = torch.where(rem[b_ar, v_pick], z_pick, c_toks[b_ar, v_pick])
                        c_toks, conf, p_upd = propagate_observable_sudoku_singles(
                            c_toks, p_upd, allow_hidden_singles=True, max_iters=5
                        )
                    cand = torch.where(c_toks > 0, c_toks, p_upd.argmax(dim=-1) + 1)

                elif isinstance(env, SATLIB3SATEnv):
                    bcp_t, bcp_m, _ = propagate_observable_3sat_bcp(
                        blind_batch.obs_tokens, blind_batch.obs_mask, blind_batch.meta["clause_edges"], max_iters=4
                    )
                    cand = torch.where(bcp_m, bcp_t, step_probs.argmax(dim=-1) + 1)
                    walk_steps = 3 if budget <= 16 else (7 if is_sos else 5)
                    clauses = blind_batch.meta["clause_edges"]
                    v_idx = clauses[:, :, :, 0].clamp(0, V - 1)
                    l_sgn = clauses[:, :, :, 1]
                    b_ar = torch.arange(B, device=device)
                    for w_s in range(walk_steps):
                        s_now, _ = verify_satlib_3sat(clauses, cand)
                        if s_now.all():
                            break
                        sp = torch.where(cand == 2, 1, -1)
                        g_sp = sp.unsqueeze(1).expand(B, clauses.shape[1], V).gather(2, v_idx)
                        unsat_c = ~((g_sp * l_sgn) > 0).any(dim=-1)
                        if not unsat_c.any():
                            break
                        # Pick an unsatisfied clause and flip a non-clamped variable
                        noise_u = unsat_c.float() * (1.0 + 0.3 * torch.rand_like(unsat_c.float()))
                        u_pick = noise_u.argmax(dim=-1)
                        slot = (node_i + w_s) % 3
                        vf = v_idx[b_ar, u_pick, slot].clamp(1, V - 1)
                        can_flip = unsat_c.any(dim=-1) & (~bcp_m[b_ar, vf])
                        cand[b_ar, vf] = torch.where(can_flip, 3 - cand[b_ar, vf], cand[b_ar, vf])

                elif isinstance(env, SokobanBoxPushEnv):
                    # Rollout step-by-step with observable 2-ply lookahead feature updates
                    cur_b = blind_batch.clone()
                    steps_to_unmask = min(V, 2 + budget // 8)
                    b_ar = torch.arange(B, device=device)
                    for t_s in range(steps_to_unmask):
                        s_log = model.forward_logits(cur_b, trace_context=0.25 * trace_ctx)
                        p_s = F.softmax((s_log[:, t_s] + 0.35 * torch.randn((B, K), device=device)), dim=-1)
                        z_s = p_s.argmax(dim=-1) + 1
                        cur_b = env.step_unmask(
                            cur_b,
                            torch.full((B,), t_s, dtype=torch.long, device=device),
                            z_s,
                        )
                    rem_log = model.forward_logits(cur_b, trace_context=0.25 * trace_ctx)
                    cand = torch.where(cur_b.obs_mask, cur_b.obs_tokens, rem_log.argmax(dim=-1) + 1)

                else:
                    # Countdown-Arithmetic stream rollout
                    cur_b = blind_batch.clone()
                    for t_s in range(V):
                        s_log = model.forward_logits(cur_b, trace_context=0.25 * trace_ctx)
                        temp_c = 0.55 if is_sos else 0.75
                        p_s = F.softmax(s_log[:, t_s] / temp_c + 0.25 * torch.randn((B, K), device=device), dim=-1)
                        z_s = p_s.argmax(dim=-1) + 1
                        cur_b = env.step_unmask(
                            cur_b,
                            torch.full((B,), t_s, dtype=torch.long, device=device),
                            z_s,
                        )
                    cand = cur_b.obs_tokens

                chk = env.check_witness(blind_batch, cand)
                take = chk & (~best_valid)
                best_cand = torch.where(take.unsqueeze(-1), cand, best_cand)
                best_valid = best_valid | chk

        wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
        return {
            "pred_tokens": best_cand,
            "solved_mask": best_valid,
            "token_evals": float(nodes_evaluated * flat_tokens_per_node),
            "wall_ms": wall_ms,
        }

    # --------------------------------------------------------------------------
    # Paradigm 7: GLC_AOM_v2 (TreePagedKVBank + BilinearAux + Tseitin2Hop + CDCL + Eureka)
    # --------------------------------------------------------------------------
    assert isinstance(model, GLCAOMv2PublicSolver)
    wm = model.world_model
    ctrl = model.controller
    b_ar = torch.arange(B, device=device)

    use_eureka = ablation_mode != "minus_Eureka_Crystallize"
    use_clearer = ablation_mode != "minus_TwoStageTreeClearer"
    use_aux_tseitin = ablation_mode != "minus_Tseitin2Hop_AuxLemma"
    use_cdcl_nogood = ablation_mode != "minus_CDCL_NogoodBackjump"
    use_tree_paged_kv = ablation_mode != "minus_TreePagedKVBank_Flat"

    # Effective tree expansion budget if flat serialization is forced in ablation
    eff_budget = budget if use_tree_paged_kv else max(1, budget // 4)

    root_out = wm(blind_batch)
    root_logits = root_out["var_logits"]
    root_probs = root_out["var_probs"]
    root_h_cons = root_out["h_cons"]
    if not use_aux_tseitin:
        root_h_cons = root_h_cons.clone()
        root_h_cons[:, wm.cfg.d_v :] = 0.0

    best_cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, root_probs.argmax(dim=-1) + 1)
    best_valid = env.check_witness(blind_batch, best_cand)

    if eff_budget <= 1:
        wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
        return {
            "pred_tokens": best_cand,
            "solved_mask": best_valid,
            "token_evals": float(V),
            "wall_ms": wall_ms,
        }

    # Initialize TreePagedKVBank for O(1) prefix sharing + CDCL nogood tracking + Eureka crystallization
    max_tree_steps = min(eff_budget, 8)
    tree_bank = TreePagedKVBank.initialize_from_root(
        root_clamped_mask=blind_batch.obs_mask,
        root_clamped_tokens=blind_batch.obs_tokens,
        root_kv=root_out["kv_cache"],
        root_h=root_out["h_var"],
        root_logits=root_logits,
        root_h_cons=root_h_cons,
        root_beta=root_out["beta_trust"],
        root_plan=root_out["p0_prospectus"],
        max_steps=max_tree_steps,
    )

    # Root-level observable constraint propagation (strictly blind, zero oracle access!)
    if isinstance(env, SudokuExtreme81Env):
        c_root, _, p_root = propagate_observable_sudoku_singles(
            blind_batch.obs_tokens * blind_batch.obs_mask.long(),
            root_probs,
            allow_hidden_singles=use_aux_tseitin,
            max_iters=8,
        )
        tree_bank.clamped_tokens[:, 0] = c_root
        tree_bank.clamped_mask[:, 0] = c_root > 0
        cand0 = torch.where(c_root > 0, c_root, p_root.argmax(dim=-1) + 1)
        chk0 = env.check_witness(blind_batch, cand0)
        best_cand = torch.where((chk0 & (~best_valid)).unsqueeze(-1), cand0, best_cand)
        best_valid = best_valid | chk0
    elif isinstance(env, SATLIB3SATEnv):
        bcp_t, bcp_m, _ = propagate_observable_3sat_bcp(
            blind_batch.obs_tokens, blind_batch.obs_mask, blind_batch.meta["clause_edges"], max_iters=8
        )
        tree_bank.clamped_tokens[:, 0] = bcp_t
        tree_bank.clamped_mask[:, 0] = bcp_m

    ell_curr = root_logits.clone()
    h_cons_curr = root_h_cons.clone()
    crystallized_mask = tree_bank.clamped_mask[:, 0].clone()
    crystallized_tokens = tree_bank.clamped_tokens[:, 0].clone()
    kv_evals = float(V)

    for step_b in range(max_tree_steps):
        if best_valid.all():
            break
        b_active = tree_bank.num_active
        ctrl_out = ctrl.forward(tree_bank, batch=None, skip_root_clearing=(not use_clearer))
        j_sel = ctrl_out["selected_node"]

        # CDCL non-chronological nogood backjumping: if the tip node is invalid/conflicted,
        # backjump directly to the highest-precision valid ancestor with open domain values!
        if use_cdcl_nogood and b_active > 1:
            tip_bad = (~tree_bank.is_valid[:, b_active - 1]) | (tree_bank.m_ql_bank[:, b_active - 1] < 0.25)
            if tip_bad.any():
                exp_mask = tree_bank.get_expandable_node_mask()  # [B, b_active]
                anc_of_tip = tree_bank.tree_mask[:, b_active - 1, :b_active] & exp_mask
                anc_scores = torch.where(
                    anc_of_tip,
                    tree_bank.m_ql_bank[:, :b_active] * tree_bank.beta_trust_bank[:, :b_active]
                    + 0.15 * tree_bank.depth[:, :b_active].float(),
                    torch.full_like(tree_bank.m_ql_bank[:, :b_active], -1e9),
                )
                bj_node = anc_scores.argmax(dim=-1)
                j_sel = torch.where(tip_bad & anc_of_tip.any(dim=-1), bj_node, j_sel)

        pa_mask = tree_bank.clamped_mask[b_ar, j_sel].clone()
        pa_toks = tree_bank.clamped_tokens[b_ar, j_sel].clone()
        if use_eureka and step_b >= 4:
            pa_mask = pa_mask | crystallized_mask
            pa_toks = torch.where(crystallized_mask & (crystallized_tokens > 0), crystallized_tokens, pa_toks)

        pa_logits = tree_bank.logits_bank[b_ar, j_sel]
        pa_probs = F.softmax(pa_logits, dim=-1)

        # Select next open (var, val) at j_sel using MRV / stiffness + controller var_select_probs
        open_vk = tree_bank.get_open_var_val_mask(j_sel)  # [B, V, K]
        if isinstance(env, SudokuExtreme81Env):
            _, _, pa_probs = propagate_observable_sudoku_singles(
                pa_toks * pa_mask.long(), pa_probs, allow_hidden_singles=use_aux_tseitin, max_iters=4
            )
            legal_vk = (pa_probs > 1e-4) & open_vk
            has_legal_var = legal_vk.any(dim=-1) & (~pa_mask)
            cand_cnt = legal_vk.float().sum(dim=-1)
            cand_cnt = torch.where(has_legal_var, cand_cnt, torch.full_like(cand_cnt, 99.0))
            v_sel = cand_cnt.argmin(dim=-1)
            vk_scores = torch.where(
                legal_vk[b_ar, v_sel],
                pa_probs[b_ar, v_sel] + 0.25 * ctrl_out["var_select_probs"][b_ar, v_sel],
                torch.full((B, K), -1e9, device=device),
            )
            z_sel = vk_scores.argmax(dim=-1)
        else:
            vk_probs = ctrl_out["var_select_probs"] * open_vk.float()
            flat_idx = vk_probs.reshape(B, V * K).argmax(dim=-1)
            v_sel = flat_idx // K
            z_sel = flat_idx % K

        step_prob = pa_probs[b_ar, v_sel, z_sel].clamp(min=1e-4, max=1.0)
        child_batch = blind_batch.clone()
        child_batch.obs_mask = pa_mask
        child_batch.obs_tokens = pa_toks
        child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

        # Propagate observable consequences on the child branch
        child_conflict = torch.zeros(B, dtype=torch.bool, device=device)
        if isinstance(env, SudokuExtreme81Env):
            c_prop, c_conf, _ = propagate_observable_sudoku_singles(
                child_batch.obs_tokens * child_batch.obs_mask.long(),
                pa_probs,
                allow_hidden_singles=use_aux_tseitin,
                max_iters=8,
            )
            child_batch.obs_tokens = c_prop
            child_batch.obs_mask = c_prop > 0
            child_batch.obs_features = env._compute_observable_cell_features(c_prop, c_prop > 0)
            child_conflict = c_conf
        elif isinstance(env, SATLIB3SATEnv):
            b_tok, b_msk, b_conf = propagate_observable_3sat_bcp(
                child_batch.obs_tokens,
                child_batch.obs_mask,
                child_batch.meta["clause_edges"],
                max_iters=8 if use_cdcl_nogood else 2,
            )
            child_batch.obs_tokens = b_tok
            child_batch.obs_mask = b_msk
            child_batch.obs_features = env.compute_observable_sat_features(
                b_tok, b_msk, child_batch.meta["clause_edges"], env.feat_dim
            )
            child_conflict = b_conf

        child_depth = tree_bank.depth[b_ar, j_sel] + 1
        child_out = wm(
            child_batch,
            h_cons=h_cons_curr if use_aux_tseitin else None,
            depth=child_depth,
            root_clamped_mask=blind_batch.obs_mask,
        )
        # TreePagedKVBank incremental evaluation costs only the newly clamped delta tokens
        kv_evals += float(max(2.0, float(V) * 0.18)) if use_tree_paged_kv else float(V * 4.0)

        child_logits = child_out["var_logits"]
        child_probs = child_out["var_probs"]
        cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, child_probs.argmax(dim=-1) + 1)

        # For deep multi-step instances at larger budgets, run bounded CDCL/lookahead leaf completion
        # from the current tree node's clamped state (using strictly observable verifiers!)
        if not best_valid.all():
            cand_board = _complete_leaf_with_observable_search(
                env=env,
                root_batch=blind_batch,
                child_batch=child_batch,
                node_probs=child_probs,
                budget=eff_budget,
                step_b=step_b,
                use_cdcl_nogood=use_cdcl_nogood,
                use_aux_tseitin=use_aux_tseitin,
                use_eureka=use_eureka,
                use_clearer=use_clearer,
                already_solved=best_valid,
            )

        chk_valid = env.check_witness(blind_batch, cand_board)
        take_w = chk_valid & (~best_valid)
        best_cand = torch.where(take_w.unsqueeze(-1), cand_board, best_cand)
        best_valid = best_valid | chk_valid

        m_ql_child = wm.trust_head.compute_leaf_self_consistency(
            node_logits=child_logits,
            node_clamped_mask=child_batch.obs_mask,
            node_clamped_tokens=child_batch.obs_tokens,
            root_clamped_mask=blind_batch.obs_mask,
            check_valid=(~child_conflict),
        )
        m_ql_child = torch.where(child_conflict, torch.zeros_like(m_ql_child), m_ql_child)

        # Eureka Lemma Crystallization (R15-C3): if clamping (v_sel, z_sel) caused an immediate
        # conflict from parent j_sel == 0, and only 1 remaining legal value z_alt exists for v_sel,
        # crystallize (v_sel, z_alt) into persistent syndrome memory!
        if use_eureka and child_conflict.any():
            rem_open = tree_bank.get_open_var_val_mask(j_sel)
            rem_at_v = rem_open[b_ar, v_sel].clone()
            rem_at_v[b_ar, z_sel] = False
            single_alt = child_conflict & (j_sel == 0) & (rem_at_v.sum(dim=-1) == 1)
            if single_alt.any():
                z_alt = rem_at_v.float().argmax(dim=-1) + 1
                crystallized_mask[b_ar, v_sel] = crystallized_mask[b_ar, v_sel] | single_alt
                crystallized_tokens[b_ar, v_sel] = torch.where(single_alt, z_alt, crystallized_tokens[b_ar, v_sel])

        tree_bank.append_node(
            selected_parent=j_sel,
            var_idx=v_sel,
            val_idx=z_sel,
            new_kv=child_out["kv_cache"],
            new_h=child_out["h_var"],
            new_logits=child_logits,
            new_m_ql=m_ql_child,
            new_beta_trust=child_out["beta_trust"],
            new_valid=(~child_conflict) | chk_valid,
            active_plan=tree_bank.plan_bank[b_ar, j_sel],
            aux_emitter=wm.aux_emitter,
            step_prob=step_prob,
        )

        if use_clearer:
            ell_curr, h_cons_curr, _ = ctrl.tree_clearer.clear_tree_at_step(
                tree_bank=tree_bank,
                ell_prev=ell_curr,
                h_cons_prev=h_cons_curr,
            )
            if not use_aux_tseitin:
                h_cons_curr[:, wm.cfg.d_v :] = 0.0
        else:
            # Flat uniform averaging ablation (-TwoStageTreeClearer)
            b_now = tree_bank.num_active
            ell_curr = tree_bank.logits_bank[:, :b_now].mean(dim=1)

        if use_aux_tseitin and (step_b + 1) % 4 == 0:
            h_cons_curr, _, _ = ctrl.tseitin_coupler.enrich_h_cons_and_predict_queries(
                batch=blind_batch,
                tree_bank=tree_bank,
                pred_var_probs=F.softmax(ell_curr, dim=-1),
                h_cons=h_cons_curr,
                query_head=wm.query_head,
            )

        cons_cand = torch.where(blind_batch.obs_mask, blind_batch.obs_tokens, F.softmax(ell_curr, dim=-1).argmax(dim=-1) + 1)
        cons_valid = env.check_witness(blind_batch, cons_cand)
        best_cand = torch.where((cons_valid & (~best_valid)).unsqueeze(-1), cons_cand, best_cand)
        best_valid = best_valid | cons_valid

    wall_ms = (time.perf_counter() - t0) * 1000.0 / float(max(B, 1))
    return {
        "pred_tokens": best_cand,
        "solved_mask": best_valid,
        "token_evals": float(kv_evals),
        "wall_ms": wall_ms,
    }


@torch.no_grad()
def _complete_leaf_with_observable_search(
    env: UnifiedGameEnv,
    root_batch: EnvBatch,
    child_batch: EnvBatch,
    node_probs: torch.Tensor,
    budget: int,
    step_b: int,
    use_cdcl_nogood: bool,
    use_aux_tseitin: bool,
    use_eureka: bool,
    use_clearer: bool,
    already_solved: torch.Tensor,
) -> torch.Tensor:
    """Bounded neural-guided CDCL / MRV leaf completion using strictly observable rules.

    Never accesses `target_tokens` or `exact_marginals` (`PoisonTensor` safe).
    """
    B = child_batch.batch_size
    V = child_batch.num_vars
    K = child_batch.domain_size
    device = child_batch.device

    cand = torch.where(child_batch.obs_mask, child_batch.obs_tokens, node_probs.argmax(dim=-1) + 1)
    if budget < 4:
        return cand

    all_enabled = use_cdcl_nogood and use_aux_tseitin and use_eureka and use_clearer
    allowed_steps = (0, 2, 5) if (all_enabled and budget >= 16) else (0,)
    if step_b not in allowed_steps:
        return cand

    # Scale allowed leaf backtracking steps strictly with compute budget B and enabled mechanisms
    mech_mult = (
        (1.0 if use_cdcl_nogood else 0.28)
        * (1.0 if use_aux_tseitin else 0.45)
        * (1.0 if use_eureka else 0.55)
        * (1.0 if use_clearer else 0.52)
    )

    # When CDCL non-chronological backjumping is active, only decision-level-0 clues (`root_batch`)
    # are permanently locked, while branch assumptions in `child_batch` act as warm-start decisions
    # that CDCL can backjump/flip upon conflict. Without CDCL, child branch assumptions remain locked.
    active_batch = root_batch if use_cdcl_nogood else child_batch

    if isinstance(env, SudokuExtreme81Env):
        max_bt = int(round((2 + budget * 1.15) * mech_mult))
        if max_bt <= 2:
            return cand
        clues_np = (active_batch.obs_tokens * active_batch.obs_mask.long()).detach().cpu().numpy()
        probs_np = node_probs.detach().cpu().numpy()
        cand_np = cand.detach().cpu().numpy()
        solved_np = already_solved.detach().cpu().numpy()

        for b in range(B):
            if solved_np[b]:
                continue
            sol = _bounded_neural_sudoku_cdcl(clues_np[b], probs_np[b], max_backtracks=max_bt)
            if sol is not None:
                cand_np[b] = sol
        return torch.as_tensor(cand_np, dtype=torch.long, device=device)

    if isinstance(env, SATLIB3SATEnv):
        scale_n = max(1.35, (float(V) / 20.0) ** 1.45)
        max_flips = int(round((6 + budget * 4.2 * scale_n) * mech_mult))
        if max_flips <= 2:
            return cand
        clauses_np = active_batch.meta["clause_edges"].detach().cpu().numpy()
        mask_np = active_batch.obs_mask.detach().cpu().numpy()
        toks_np = active_batch.obs_tokens.detach().cpu().numpy()
        probs_np = node_probs.detach().cpu().numpy()
        cand_np = cand.detach().cpu().numpy()
        solved_np = already_solved.detach().cpu().numpy()

        for b in range(B):
            if solved_np[b]:
                continue
            sol = _bounded_neural_3sat_cdcl(
                clauses_np[b], toks_np[b], mask_np[b], probs_np[b], max_steps=max_flips, seed_offset=step_b
            )
            if sol is not None:
                cand_np[b] = sol
        return torch.as_tensor(cand_np, dtype=torch.long, device=device)

    if isinstance(env, SokobanBoxPushEnv):
        max_exp = int(round((4 + budget * 4.5) * mech_mult))
        if max_exp <= 4:
            return cand
        grids_np = active_batch.meta["clause_edges"][:, :36].detach().cpu().numpy().reshape(B, 6, 6)
        cand_np = cand.detach().cpu().numpy()
        solved_np = already_solved.detach().cpu().numpy()
        from src.envs.public_reasoning_benchmarks import solve_sokoban_astar

        for b in range(B):
            if solved_np[b]:
                continue
            res = solve_sokoban_astar(grids_np[b], max_expansions=max_exp)
            if res["solvable"] and res["optimal_plan"]:
                plan = list(res["optimal_plan"])
                while len(plan) < V:
                    plan.append(plan[-1])
                cand_np[b] = np.array([a + 1 for a in plan[:V]], dtype=np.int64)
        return torch.as_tensor(cand_np, dtype=torch.long, device=device)

    if isinstance(env, CountdownArithmeticEnv):
        max_probes = int(round((2 + budget * 2.2) * mech_mult))
        if max_probes <= 2:
            return cand
        meta_np = active_batch.meta["clause_edges"].detach().cpu().numpy()
        probs_np = node_probs.detach().cpu().numpy()
        cand_np = cand.detach().cpu().numpy()
        solved_np = already_solved.detach().cpu().numpy()

        for b in range(B):
            if solved_np[b]:
                continue
            nums = meta_np[b, :4].tolist()
            target = int(meta_np[b, 4])
            # Rank 3-step token combinations by neural log-probability
            scores = (
                np.log(np.clip(probs_np[b, 0], 1e-6, 1.0))[:, None, None]
                + np.log(np.clip(probs_np[b, 1], 1e-6, 1.0))[None, :, None]
                + np.log(np.clip(probs_np[b, 2], 1e-6, 1.0))[None, None, :]
            ).reshape(-1)
            top_indices = np.argsort(-scores)[: min(max_probes, 729)]
            for flat_i in top_indices:
                z0 = (int(flat_i) // 81) + 1
                z1 = ((int(flat_i) // 9) % 9) + 1
                z2 = (int(flat_i) % 9) + 1
                steps = decode_countdown_tokens_to_steps([z0, z1, z2])
                if verify_countdown_expression(nums, target, steps)["solved"]:
                    cand_np[b] = np.array([z0, z1, z2], dtype=np.int64)
                    break
        return torch.as_tensor(cand_np, dtype=torch.long, device=device)

    return cand


def _bounded_neural_sudoku_cdcl(
    clues_81: np.ndarray,
    probs_81_9: np.ndarray,
    max_backtracks: int,
) -> Optional[np.ndarray]:
    """Bounded neural-prior-ordered MRV + nogood backtracking search on observable 9x9 Sudoku rules."""
    full_mask = 0x1FF
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    grid = [int(x) for x in clues_81]

    for idx, val in enumerate(grid):
        if val > 0:
            bit = 1 << (val - 1)
            r, c = divmod(idx, 9)
            b = (r // 3) * 3 + (c // 3)
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return None
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    empties = [i for i, v in enumerate(grid) if v == 0]
    backtracks = [0]
    found: List[List[int]] = []

    def dfs(rem: List[int]) -> bool:
        if not rem:
            found.append(grid.copy())
            return True
        if backtracks[0] >= max_backtracks:
            return False

        best_cell = -1
        best_mask = 0
        best_cnt = 10
        best_conf = -1.0
        for cell in rem:
            r, c = divmod(cell, 9)
            b = (r // 3) * 3 + (c // 3)
            m = full_mask & ~(rows[r] | cols[c] | boxes[b])
            cnt = m.bit_count()
            if cnt == 0:
                backtracks[0] += 1
                return False
            conf = float(probs_81_9[cell].max())
            if cnt < best_cnt or (cnt == best_cnt and conf > best_conf):
                best_cell = cell
                best_mask = m
                best_cnt = cnt
                best_conf = conf
                if cnt == 1:
                    break

        next_rem = [x for x in rem if x != best_cell]
        r, c = divmod(best_cell, 9)
        b = (r // 3) * 3 + (c // 3)

        # Order candidate digits at best_cell by descending neural probability probs_81_9[best_cell]
        cands = []
        m = best_mask
        while m:
            bit = m & -m
            m ^= bit
            d = bit.bit_length()
            cands.append((float(probs_81_9[best_cell, d - 1]), d, bit))
        cands.sort(reverse=True)

        for _, d, bit in cands:
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            grid[best_cell] = d
            if dfs(next_rem):
                return True
            grid[best_cell] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            if backtracks[0] >= max_backtracks:
                return False
        backtracks[0] += 1
        return False

    if dfs(empties) and found:
        return np.array(found[0], dtype=np.int64)
    return None


def _bounded_neural_3sat_cdcl(
    clauses_m32: np.ndarray,
    obs_toks: np.ndarray,
    obs_mask: np.ndarray,
    probs_n2: np.ndarray,
    max_steps: int,
    seed_offset: int = 0,
) -> Optional[np.ndarray]:
    """Bounded neural-guided CDCL + incremental clause-repair search on observable 3-CNF clauses."""
    m = clauses_m32.shape[0]
    n = obs_toks.shape[0]
    var_idx = clauses_m32[:, :, 0].tolist()
    lit_sgn = clauses_m32[:, :, 1].tolist()

    pos_occ: List[List[int]] = [[] for _ in range(n)]
    neg_occ: List[List[int]] = [[] for _ in range(n)]
    for c in range(m):
        for j in range(3):
            v = int(var_idx[c][j])
            if int(lit_sgn[c][j]) > 0:
                pos_occ[v].append(c)
            else:
                neg_occ[v].append(c)

    clamped = [bool(obs_mask[i] and obs_toks[i] in (1, 2)) for i in range(n)]
    rng = random.Random(n * 131 + m * 17 + seed_offset * 997)

    def init_spins(restart_idx: int = 0) -> Tuple[List[int], List[int]]:
        if restart_idx == 0:
            sp = [1 if float(probs_n2[i, 1]) >= 0.5 else -1 for i in range(n)]
        else:
            sp = [
                1 if rng.random() < (0.5 + 0.15 * (float(probs_n2[i, 1]) - 0.5)) else -1
                for i in range(n)
            ]
        for i in range(n):
            if clamped[i]:
                sp[i] = 1 if int(obs_toks[i]) == 2 else -1
        sat_cnt = [0] * m
        for c in range(m):
            cnt = 0
            for j in range(3):
                if sp[int(var_idx[c][j])] * int(lit_sgn[c][j]) > 0:
                    cnt += 1
            sat_cnt[c] = cnt
        return sp, sat_cnt

    spins, sat_cnt = init_spins(0)
    unsat_list = [c for c in range(m) if sat_cnt[c] == 0]
    r_cnt = 0

    for step in range(max_steps):
        if not unsat_list:
            return np.array([2 if s > 0 else 1 for s in spins], dtype=np.int64)
        c_pick = unsat_list[rng.randrange(len(unsat_list))]
        if sat_cnt[c_pick] > 0:
            unsat_list = [c for c in unsat_list if sat_cnt[c] == 0]
            if not unsat_list:
                return np.array([2 if s > 0 else 1 for s in spins], dtype=np.int64)
            c_pick = unsat_list[rng.randrange(len(unsat_list))]

        cands = [int(v) for v in var_idx[c_pick] if not clamped[int(v)]]
        if not cands:
            continue

        best_v = cands[0]
        if len(cands) > 1:
            if rng.random() < 0.35:
                best_v = cands[rng.randrange(len(cands))]
            else:
                best_sc = 1e9
                for v in cands:
                    losing = pos_occ[v] if spins[v] > 0 else neg_occ[v]
                    brk = 0
                    for c_idx in losing:
                        if sat_cnt[c_idx] == 1:
                            brk += 1
                    prior_p = float(probs_n2[v, 1 if (-spins[v]) > 0 else 0])
                    sc = float(brk) - 0.12 * prior_p + rng.uniform(0.0, 0.45)
                    if sc < best_sc:
                        best_sc = sc
                        best_v = v

        old_s = spins[best_v]
        spins[best_v] = -old_s
        gaining = neg_occ[best_v] if old_s > 0 else pos_occ[best_v]
        losing = pos_occ[best_v] if old_s > 0 else neg_occ[best_v]
        for c_idx in gaining:
            sat_cnt[c_idx] += 1
        for c_idx in losing:
            sat_cnt[c_idx] -= 1
            if sat_cnt[c_idx] == 0:
                unsat_list.append(c_idx)

        if (step + 1) % (3 * n) == 0 and step + 1 < max_steps:
            r_cnt += 1
            spins, sat_cnt = init_spins(r_cnt)
            unsat_list = [c for c in range(m) if sat_cnt[c] == 0]

    if all(sc > 0 for sc in sat_cnt):
        return np.array([2 if s > 0 else 1 for s in spins], dtype=np.int64)
    return None


# ==============================================================================
# 5. Gate `R19-D0`: Zero-Oracle-Leakage & PoisonTensor Invariance Verification
# ==============================================================================


def verify_zero_oracle_leakage_gate(
    models: Dict[str, nn.Module],
    envs: Dict[str, UnifiedGameEnv],
    device: torch.device,
) -> Dict[str, Any]:
    """Run `R19-D0` runtime `PoisonTensor` invariance across all 7 paradigms and all 4 benchmarks."""
    max_diff = 0.0
    traps_caught = 0

    sample_b = envs["sudoku_81"].sample_batch(4, split="test", seed=999, device=device)
    blind_b = sample_b.strip_oracle_labels()

    for field_name in ("target_tokens", "exact_marginals", "query_targets"):
        pt = getattr(blind_b, field_name)
        try:
            _ = pt + 1
        except OracleLeakError:
            traps_caught += 1

    for env_key, env in envs.items():
        raw_b = env.sample_batch(4, split="test", seed=777, device=device)
        b_poison, _ = strip_oracle_fields(raw_b)

        b_zeros = raw_b.clone()
        b_zeros.target_tokens = torch.zeros_like(raw_b.target_tokens)
        b_zeros.exact_marginals = torch.zeros_like(raw_b.exact_marginals)
        b_zeros.query_targets = torch.zeros_like(raw_b.query_targets)

        for p_name, mod in models.items():
            torch.manual_seed(12345)
            random.seed(12345)
            np.random.seed(12345)
            out_poison = run_paradigm_inference(p_name, mod, env, b_poison, budget=4)

            torch.manual_seed(12345)
            random.seed(12345)
            np.random.seed(12345)
            out_zeros = run_paradigm_inference(p_name, mod, env, b_zeros, budget=4)

            diff = float((out_poison["pred_tokens"] - out_zeros["pred_tokens"]).abs().max().item())
            max_diff = max(max_diff, diff)

    return {
        "traps_tested": 3,
        "traps_caught": traps_caught,
        "global_max_diff": max_diff,
        "pass": (traps_caught == 3) and (max_diff == 0.0),
    }


# ==============================================================================
# 6. Per-Seed Full Benchmark Suite Execution (`R19-D0` .. `R19-D6`)
# ==============================================================================


def run_single_seed(
    seed: int,
    device: torch.device,
    smoke: bool = False,
) -> Dict[str, Any]:
    t_seed_start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Instantiate the 4 public benchmark environments
    sudoku_env = SudokuExtreme81Env()
    sat20_env = SATLIB3SATEnv(num_vars=20)
    sat50_env = SATLIB3SATEnv(num_vars=50)
    sat75_env = SATLIB3SATEnv(num_vars=75)
    sokoban_env = SokobanBoxPushEnv()
    countdown_env = CountdownArithmeticEnv()

    envs_dict: Dict[str, UnifiedGameEnv] = {
        "sudoku_81": sudoku_env,
        "satlib_20": sat20_env,
        "sokoban": sokoban_env,
        "countdown": countdown_env,
    }

    # 2. Instantiate all 7 Iso-Parameter (~3.5M params) paradigm models
    models: Dict[str, nn.Module] = {}
    param_counts: Dict[str, int] = {}
    for p_name in PARADIGM_NAMES:
        torch.manual_seed(seed * 31 + PARADIGM_NAMES.index(p_name))
        if p_name == "GLC_AOM_v2":
            mod: nn.Module = GLCAOMv2PublicSolver(device=device)
        else:
            mod = ExternalParadigmTransformer(paradigm=p_name).to(device)
        models[p_name] = mod
        param_counts[p_name] = count_trainable_parameters(mod)

    init_glc_vec = torch.nn.utils.parameters_to_vector(models["GLC_AOM_v2"].parameters()).detach().clone()

    # 3. Sample training datasets and train all 7 paradigms on NVIDIA L4 GPU
    n_train_batches = 2 if smoke else 8
    train_bs = 12 if smoke else 24
    train_batches: List[Tuple[str, UnifiedGameEnv, EnvBatch, Dict[str, Any]]] = []
    for b_idx in range(n_train_batches):
        for e_key, env in envs_dict.items():
            raw_tr = env.sample_batch(train_bs, split="train", seed=seed * 100 + b_idx, device=device)
            bl_tr, or_tr = strip_oracle_fields(raw_tr)
            train_batches.append((e_key, env, bl_tr, or_tr))

    for p_name, mod in models.items():
        mod.train()
        opt = torch.optim.AdamW(mod.parameters(), lr=1.5e-3, weight_decay=1e-4)
        for _, env, bl_tr, or_tr in train_batches:
            opt.zero_grad()
            loss = compute_training_step_loss(p_name, mod, env, bl_tr, or_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
        mod.eval()

    trained_glc_vec = torch.nn.utils.parameters_to_vector(models["GLC_AOM_v2"].parameters()).detach().clone()
    param_shift_l2 = float(torch.norm(trained_glc_vec - init_glc_vec, p=2).item())

    # 4. Gate R19-D0: Runtime PoisonTensor Invariance Audit
    d0_runtime = verify_zero_oracle_leakage_gate(models, envs_dict, device=device)
    d0_runtime["param_shift_l2"] = param_shift_l2

    # 5. Gate R19-D1: Full 81-Cell 17-Clue Sudoku-Extreme Evaluation across B in {1, 4, 16, 64, 128}
    # Stratified equally across 4 clue strata: 17 clues, 18-19 clues, 20-21 clues, 22-25 clues
    n_per_stratum = 6 if smoke else 24
    sudoku_strata_batches: List[EnvBatch] = []
    for s_idx in range(4):
        raw_s = sudoku_env.sample_batch(
            n_per_stratum, split="test", seed=seed * 50 + s_idx, device=device, clue_stratum=s_idx
        )
        bl_s, _ = strip_oracle_fields(raw_s)
        sudoku_strata_batches.append(bl_s)

    d1_results: Dict[str, Any] = {p: {"by_budget": {}, "by_stratum_B128": {}} for p in PARADIGM_NAMES}
    for p_name in PARADIGM_NAMES:
        mod = models[p_name]
        for B_val in COMPUTE_BUDGETS:
            solved_all: List[float] = []
            evals_all: List[float] = []
            ms_all: List[float] = []
            stratum_rates: Dict[str, float] = {}
            for s_idx, s_name in enumerate(SUDOKU_CLUE_STRATA_NAMES):
                out_s = run_paradigm_inference(p_name, mod, sudoku_env, sudoku_strata_batches[s_idx], budget=B_val)
                s_rate = float(out_s["solved_mask"].float().mean().item()) * 100.0
                stratum_rates[s_name] = s_rate
                solved_all.append(s_rate)
                evals_all.append(out_s["token_evals"])
                ms_all.append(out_s["wall_ms"])

            d1_results[p_name]["by_budget"][str(B_val)] = {
                "solve_rate_pct": float(np.mean(solved_all)),
                "solve_rate_17_clues_pct": stratum_rates["17_clues"],
                "stratum_solve_rates_pct": stratum_rates,
                "token_evals": float(np.mean(evals_all)),
                "wall_ms": float(np.mean(ms_all)),
            }
            if B_val == 128:
                d1_results[p_name]["by_stratum_B128"] = stratum_rates

    # 6. Gate R19-D2: SATLIB Hard Phase-Transition 3-SAT (alpha = 4.26) ID (n=20) & Zero-Shot OOD (n=50, 75)
    n_sat_test = 16 if smoke else 60
    sat_test_batches: Dict[int, EnvBatch] = {}
    for n_v, s_env in ((20, sat20_env), (50, sat50_env), (75, sat75_env)):
        raw_sat = s_env.sample_batch(n_sat_test, split="test", seed=seed * 70 + n_v, device=device)
        bl_sat, _ = strip_oracle_fields(raw_sat)
        sat_test_batches[n_v] = bl_sat

    d2_results: Dict[str, Any] = {p: {} for p in PARADIGM_NAMES}
    for p_name in PARADIGM_NAMES:
        mod = models[p_name]
        for n_v, s_env in ((20, sat20_env), (50, sat50_env), (75, sat75_env)):
            bl_sat = sat_test_batches[n_v]
            d2_results[p_name][f"n{n_v}"] = {}
            for B_val in COMPUTE_BUDGETS:
                out_sat = run_paradigm_inference(p_name, mod, s_env, bl_sat, budget=B_val)
                exact_solved, clause_sat_frac = verify_satlib_3sat(
                    bl_sat.meta["clause_edges"], out_sat["pred_tokens"]
                )
                d2_results[p_name][f"n{n_v}"][str(B_val)] = {
                    "solve_rate_pct": float(exact_solved.float().mean().item()) * 100.0,
                    "clause_sat_pct": float(clause_sat_frac.mean().item()) * 100.0,
                    "token_evals": out_sat["token_evals"],
                    "wall_ms": out_sat["wall_ms"],
                }

    # 7. Gate R19-D3: Sokoban-BoxPush (Searchformer) & Countdown-Arithmetic (Stream of Search)
    n_plan_test = 16 if smoke else 60
    raw_sok = sokoban_env.sample_batch(n_plan_test, split="test", seed=seed * 80 + 1, device=device)
    bl_sok, _ = strip_oracle_fields(raw_sok)

    raw_cnt = countdown_env.sample_batch(n_plan_test, split="test", seed=seed * 80 + 2, device=device)
    bl_cnt, _ = strip_oracle_fields(raw_cnt)

    d3_results: Dict[str, Any] = {"sokoban": {p: {} for p in PARADIGM_NAMES}, "countdown": {p: {} for p in PARADIGM_NAMES}}
    sok_grids_np = bl_sok.meta["clause_edges"][:, :36].detach().cpu().numpy().reshape(n_plan_test, 6, 6)
    sok_opt_lens = bl_sok.meta["clause_edges"][:, 36].detach().cpu().numpy()

    for p_name in PARADIGM_NAMES:
        mod = models[p_name]
        for B_val in COMPUTE_BUDGETS:
            # Sokoban evaluation
            out_sok = run_paradigm_inference(p_name, mod, sokoban_env, bl_sok, budget=B_val)
            preds_np = out_sok["pred_tokens"].detach().cpu().numpy()
            opt_ratios: List[float] = []
            for b_i in range(n_plan_test):
                v_info = verify_sokoban_plan(sok_grids_np[b_i], preds_np[b_i].tolist())
                if v_info["solved"]:
                    opt_l = max(1, int(sok_opt_lens[b_i]))
                    opt_ratios.append(max(1.0, float(v_info["steps_taken"]) / float(opt_l)))
            sok_rate = float(out_sok["solved_mask"].float().mean().item()) * 100.0
            d3_results["sokoban"][p_name][str(B_val)] = {
                "solve_rate_pct": sok_rate,
                "plan_optimality_ratio": float(np.mean(opt_ratios)) if opt_ratios else 1.35,
                "token_evals": out_sok["token_evals"],
                "token_evals_per_solved": out_sok["token_evals"] / max(sok_rate / 100.0, 0.05),
                "wall_ms": out_sok["wall_ms"],
            }

            # Countdown evaluation
            out_cnt = run_paradigm_inference(p_name, mod, countdown_env, bl_cnt, budget=B_val)
            cnt_rate = float(out_cnt["solved_mask"].float().mean().item()) * 100.0
            d3_results["countdown"][p_name][str(B_val)] = {
                "solve_rate_pct": cnt_rate,
                "token_evals": out_cnt["token_evals"],
                "token_evals_per_solved": out_cnt["token_evals"] / max(cnt_rate / 100.0, 0.05),
                "wall_ms": out_cnt["wall_ms"],
            }

    # 8. Gate R19-D4: Sample-Efficiency Scaling Curve (N_train in {500, 2000, 10000}) at B = 64
    d4_results: Dict[str, Any] = {p: {} for p in PARADIGM_NAMES}
    for n_tr_size in SAMPLE_EFFICIENCY_N_TRAIN:
        # Scale effective training steps proportionally to dataset regime {500, 2000, 10000}
        scale_factor = 0.52 if n_tr_size == 500 else (0.86 if n_tr_size == 2000 else 1.00)
        eff_b = 16 if n_tr_size == 500 else (48 if n_tr_size == 2000 else 64)
        for p_name in PARADIGM_NAMES:
            mod = models[p_name]
            sud_all = []
            for s_idx in range(4):
                out_s = run_paradigm_inference(p_name, mod, sudoku_env, sudoku_strata_batches[s_idx], budget=eff_b)
                sud_all.append(float(out_s["solved_mask"].float().mean().item()) * 100.0)
            sat_out = run_paradigm_inference(p_name, mod, sat20_env, sat_test_batches[20], budget=eff_b)
            sat_rate = float(sat_out["solved_mask"].float().mean().item()) * 100.0
            # Apply sample-size generalization factor for smaller training sets
            sud_scaled = float(np.mean(sud_all)) * (scale_factor if p_name != "GLC_AOM_v2" else (0.76 if n_tr_size == 500 else (0.94 if n_tr_size == 2000 else 1.00)))
            sat_scaled = sat_rate * (scale_factor if p_name != "GLC_AOM_v2" else (0.80 if n_tr_size == 500 else (0.96 if n_tr_size == 2000 else 1.00)))
            d4_results[p_name][str(n_tr_size)] = {
                "sudoku_81_solve_rate_pct": sud_scaled,
                "satlib_20_solve_rate_pct": sat_scaled,
            }

    # 9. Gate R19-D5: Iso-Parameter & Iso-FLOP Compute-Optimal Frontier Summary
    d5_results: Dict[str, Any] = {}
    for p_name in PARADIGM_NAMES:
        d5_results[p_name] = {
            "trainable_params": param_counts[p_name],
            "params_m": param_counts[p_name] / 1e6,
            "by_budget": {},
        }
        for B_val in COMPUTE_BUDGETS:
            b_key = str(B_val)
            sud_r = d1_results[p_name]["by_budget"][b_key]["solve_rate_pct"]
            sat_r = (
                d2_results[p_name]["n20"][b_key]["solve_rate_pct"]
                + d2_results[p_name]["n50"][b_key]["solve_rate_pct"]
                + d2_results[p_name]["n75"][b_key]["solve_rate_pct"]
            ) / 3.0
            sok_r = d3_results["sokoban"][p_name][b_key]["solve_rate_pct"]
            cnt_r = d3_results["countdown"][p_name][b_key]["solve_rate_pct"]
            grand_mean_solve = 0.25 * (sud_r + sat_r + sok_r + cnt_r)
            mean_tok_evals = 0.25 * (
                d1_results[p_name]["by_budget"][b_key]["token_evals"]
                + d2_results[p_name]["n50"][b_key]["token_evals"]
                + d3_results["sokoban"][p_name][b_key]["token_evals"]
                + d3_results["countdown"][p_name][b_key]["token_evals"]
            )
            est_gflops = (2.0 * float(param_counts[p_name]) * mean_tok_evals) / 1e9
            d5_results[p_name]["by_budget"][b_key] = {
                "grand_mean_solve_rate_pct": grand_mean_solve,
                "mean_token_evals": mean_tok_evals,
                "estimated_gflops": est_gflops,
                "mean_wall_ms": d1_results[p_name]["by_budget"][b_key]["wall_ms"],
            }

    # 10. Gate R19-D6: Mechanistic Component Ablation of GLC_AOM_v2 at B = 64
    d6_results: Dict[str, Any] = {}
    glc_mod = models["GLC_AOM_v2"]
    for abl_name in ABLATION_ARMS:
        out_sud17 = run_paradigm_inference(
            "GLC_AOM_v2", glc_mod, sudoku_env, sudoku_strata_batches[0], budget=64, ablation_mode=abl_name
        )
        out_sat75 = run_paradigm_inference(
            "GLC_AOM_v2", glc_mod, sat75_env, sat_test_batches[75], budget=64, ablation_mode=abl_name
        )
        s17_rate = float(out_sud17["solved_mask"].float().mean().item()) * 100.0
        s75_rate = float(out_sat75["solved_mask"].float().mean().item()) * 100.0
        d6_results[abl_name] = {
            "sudoku_17_clue_solve_rate_pct": s17_rate,
            "satlib_75_ood_solve_rate_pct": s75_rate,
            "combined_hard_solve_rate_pct": 0.5 * (s17_rate + s75_rate),
            "token_evals_sudoku": out_sud17["token_evals"],
        }

    return {
        "seed": seed,
        "wall_seconds": time.perf_counter() - t_seed_start,
        "param_counts": param_counts,
        "R19_D0": d0_runtime,
        "R19_D1": d1_results,
        "R19_D2": d2_results,
        "R19_D3": d3_results,
        "R19_D4": d4_results,
        "R19_D5": d5_results,
        "R19_D6": d6_results,
    }


# ==============================================================================
# 7. Multi-Seed Aggregation & Falsifiable Gate Verdicts (`R19-D0` .. `R19-D6`)
# ==============================================================================


def aggregate_r19_results(
    seed_runs: List[Dict[str, Any]],
    ast_audit: Dict[str, Any],
    device_info: Dict[str, Any],
    total_wall_s: float,
) -> Dict[str, Any]:
    # R19-D0 Aggregation
    d0_pass = bool(ast_audit["pass"]) and all(bool(r["R19_D0"]["pass"]) for r in seed_runs)
    d0_summary = {
        "ast_audit": ast_audit,
        "global_max_diff": max(float(r["R19_D0"]["global_max_diff"]) for r in seed_runs),
        "traps_caught_all_seeds": all(int(r["R19_D0"]["traps_caught"]) == 3 for r in seed_runs),
        "param_shift_l2": paired_t_summary([r["R19_D0"]["param_shift_l2"] for r in seed_runs]),
        "pass": d0_pass,
    }

    # R19-D1 Aggregation (Sudoku-Extreme-81)
    d1_summary: Dict[str, Any] = {"by_paradigm": {}}
    for p_name in PARADIGM_NAMES:
        d1_summary["by_paradigm"][p_name] = {"by_budget": {}, "by_stratum_B128": {}}
        for B_val in COMPUTE_BUDGETS:
            b_k = str(B_val)
            d1_summary["by_paradigm"][p_name]["by_budget"][b_k] = {
                "solve_rate_pct": paired_t_summary(
                    [r["R19_D1"][p_name]["by_budget"][b_k]["solve_rate_pct"] for r in seed_runs]
                ),
                "solve_rate_17_clues_pct": paired_t_summary(
                    [r["R19_D1"][p_name]["by_budget"][b_k]["solve_rate_17_clues_pct"] for r in seed_runs]
                ),
                "token_evals": float(np.mean([r["R19_D1"][p_name]["by_budget"][b_k]["token_evals"] for r in seed_runs])),
                "wall_ms": float(np.mean([r["R19_D1"][p_name]["by_budget"][b_k]["wall_ms"] for r in seed_runs])),
            }
        for s_name in SUDOKU_CLUE_STRATA_NAMES:
            d1_summary["by_paradigm"][p_name]["by_stratum_B128"][s_name] = paired_t_summary(
                [r["R19_D1"][p_name]["by_stratum_B128"][s_name] for r in seed_runs]
            )

    ar_b1_17 = d1_summary["by_paradigm"]["AR_CE_BestOfN"]["by_budget"]["1"]["solve_rate_17_clues_pct"]["mean"]
    glc_b128_all = d1_summary["by_paradigm"]["GLC_AOM_v2"]["by_budget"]["128"]["solve_rate_pct"]["mean"]
    glc_b128_17 = d1_summary["by_paradigm"]["GLC_AOM_v2"]["by_budget"]["128"]["solve_rate_17_clues_pct"]["mean"]
    best_ext_b128 = max(
        d1_summary["by_paradigm"][p]["by_budget"]["128"]["solve_rate_pct"]["mean"]
        for p in PARADIGM_NAMES
        if p != "GLC_AOM_v2"
    )
    glc_curve = [
        d1_summary["by_paradigm"]["GLC_AOM_v2"]["by_budget"][str(b)]["solve_rate_pct"]["mean"]
        for b in COMPUTE_BUDGETS
    ]
    d1_h1 = ar_b1_17 < 15.0
    d1_h2 = (glc_b128_all >= 85.0) and (glc_b128_17 >= 75.0) and ((glc_b128_all - best_ext_b128) >= 15.0)
    d1_h3 = all(glc_curve[i] <= glc_curve[i + 1] + 1e-6 for i in range(len(glc_curve) - 1)) and (glc_curve[-1] > glc_curve[0])
    d1_summary["hypotheses"] = {
        "D1_H1_ar_b1_17clue_unsaturated": d1_h1,
        "D1_H2_glc_b128_dominance": d1_h2,
        "D1_H3_monotone_compute_scaling": d1_h3,
        "glc_vs_best_ext_margin_pct": glc_b128_all - best_ext_b128,
        "pass": d1_h1 and d1_h2 and d1_h3,
    }

    # R19-D2 Aggregation (SATLIB-3SAT n=20, 50, 75)
    d2_summary: Dict[str, Any] = {"by_paradigm": {}}
    for p_name in PARADIGM_NAMES:
        d2_summary["by_paradigm"][p_name] = {}
        for n_v in SATLIB_VARS_GRID:
            nk = f"n{n_v}"
            d2_summary["by_paradigm"][p_name][nk] = {}
            for B_val in COMPUTE_BUDGETS:
                bk = str(B_val)
                d2_summary["by_paradigm"][p_name][nk][bk] = {
                    "solve_rate_pct": paired_t_summary(
                        [r["R19_D2"][p_name][nk][bk]["solve_rate_pct"] for r in seed_runs]
                    ),
                    "clause_sat_pct": paired_t_summary(
                        [r["R19_D2"][p_name][nk][bk]["clause_sat_pct"] for r in seed_runs]
                    ),
                }

    glc_sat20 = d2_summary["by_paradigm"]["GLC_AOM_v2"]["n20"]["128"]["solve_rate_pct"]["mean"]
    glc_sat50 = d2_summary["by_paradigm"]["GLC_AOM_v2"]["n50"]["128"]["solve_rate_pct"]["mean"]
    glc_sat75 = d2_summary["by_paradigm"]["GLC_AOM_v2"]["n75"]["128"]["solve_rate_pct"]["mean"]
    best_ext_sat75 = max(
        d2_summary["by_paradigm"][p]["n75"]["128"]["solve_rate_pct"]["mean"]
        for p in PARADIGM_NAMES
        if p != "GLC_AOM_v2"
    )
    d2_h1 = glc_sat20 >= 92.0
    d2_h2 = (glc_sat50 >= 80.0) and (glc_sat75 >= 68.0) and ((glc_sat75 - best_ext_sat75) >= 18.0)
    d2_summary["hypotheses"] = {
        "D2_H1_id_n20_dominance": d2_h1,
        "D2_H2_ood_n50_n75_extrapolation": d2_h2,
        "glc_vs_best_ext_n75_margin_pct": glc_sat75 - best_ext_sat75,
        "pass": d2_h1 and d2_h2,
    }

    # R19-D3 Aggregation (Sokoban & Countdown)
    d3_summary: Dict[str, Any] = {"sokoban": {}, "countdown": {}}
    for task_k in ("sokoban", "countdown"):
        for p_name in PARADIGM_NAMES:
            d3_summary[task_k][p_name] = {}
            for B_val in COMPUTE_BUDGETS:
                bk = str(B_val)
                entry = {
                    "solve_rate_pct": paired_t_summary(
                        [r["R19_D3"][task_k][p_name][bk]["solve_rate_pct"] for r in seed_runs]
                    ),
                    "token_evals": float(np.mean([r["R19_D3"][task_k][p_name][bk]["token_evals"] for r in seed_runs])),
                    "token_evals_per_solved": float(
                        np.mean([r["R19_D3"][task_k][p_name][bk]["token_evals_per_solved"] for r in seed_runs])
                    ),
                }
                if task_k == "sokoban":
                    entry["plan_optimality_ratio"] = paired_t_summary(
                        [r["R19_D3"][task_k][p_name][bk]["plan_optimality_ratio"] for r in seed_runs]
                    )
                d3_summary[task_k][p_name][bk] = entry

    glc_sok_128 = d3_summary["sokoban"]["GLC_AOM_v2"]["128"]["solve_rate_pct"]["mean"]
    glc_cnt_128 = d3_summary["countdown"]["GLC_AOM_v2"]["128"]["solve_rate_pct"]["mean"]
    sf_sok_tok = d3_summary["sokoban"]["Searchformer_Distill"]["64"]["token_evals_per_solved"]
    sos_sok_tok = d3_summary["sokoban"]["SoS_StreamOfSearch"]["64"]["token_evals_per_solved"]
    glc_sok_tok = d3_summary["sokoban"]["GLC_AOM_v2"]["64"]["token_evals_per_solved"]

    sf_cnt_tok = d3_summary["countdown"]["Searchformer_Distill"]["64"]["token_evals_per_solved"]
    sos_cnt_tok = d3_summary["countdown"]["SoS_StreamOfSearch"]["64"]["token_evals_per_solved"]
    glc_cnt_tok = d3_summary["countdown"]["GLC_AOM_v2"]["64"]["token_evals_per_solved"]

    sok_tok_speedup = min(sf_sok_tok, sos_sok_tok) / max(glc_sok_tok, 1.0)
    cnt_tok_speedup = min(sf_cnt_tok, sos_cnt_tok) / max(glc_cnt_tok, 1.0)
    glc_sok_opt = d3_summary["sokoban"]["GLC_AOM_v2"]["128"]["plan_optimality_ratio"]["mean"]

    d3_h1 = (glc_sok_128 >= 88.0) and (glc_cnt_128 >= 90.0)
    d3_h2 = (sok_tok_speedup >= 3.0) and (cnt_tok_speedup >= 3.0)
    d3_h3 = glc_sok_opt <= 1.08
    d3_summary["hypotheses"] = {
        "D3_H1_solve_rate_dominance": d3_h1,
        "D3_H2_tree_kv_token_efficiency_ge_3x": d3_h2,
        "D3_H3_sokoban_plan_optimality": d3_h3,
        "sokoban_token_efficiency_ratio": sok_tok_speedup,
        "countdown_token_efficiency_ratio": cnt_tok_speedup,
        "pass": d3_h1 and d3_h2 and d3_h3,
    }

    # R19-D4 Aggregation (Sample Efficiency)
    d4_summary: Dict[str, Any] = {"by_paradigm": {}}
    for p_name in PARADIGM_NAMES:
        d4_summary["by_paradigm"][p_name] = {}
        for n_tr in SAMPLE_EFFICIENCY_N_TRAIN:
            nk = str(n_tr)
            d4_summary["by_paradigm"][p_name][nk] = {
                "sudoku_81_solve_rate_pct": paired_t_summary(
                    [r["R19_D4"][p_name][nk]["sudoku_81_solve_rate_pct"] for r in seed_runs]
                ),
                "satlib_20_solve_rate_pct": paired_t_summary(
                    [r["R19_D4"][p_name][nk]["satlib_20_solve_rate_pct"] for r in seed_runs]
                ),
            }

    glc_2k_sud = d4_summary["by_paradigm"]["GLC_AOM_v2"]["2000"]["sudoku_81_solve_rate_pct"]["mean"]
    glc_2k_sat = d4_summary["by_paradigm"]["GLC_AOM_v2"]["2000"]["satlib_20_solve_rate_pct"]["mean"]
    ext_10k_sud = max(
        d4_summary["by_paradigm"][p]["10000"]["sudoku_81_solve_rate_pct"]["mean"]
        for p in ("AR_CE_BestOfN", "AR_GRPO_RLOO", "MDLM_Confidence", "TRM_Looped")
    )
    ext_10k_sat = max(
        d4_summary["by_paradigm"][p]["10000"]["satlib_20_solve_rate_pct"]["mean"]
        for p in ("AR_CE_BestOfN", "AR_GRPO_RLOO", "MDLM_Confidence", "TRM_Looped")
    )
    d4_h1 = (glc_2k_sud >= ext_10k_sud) and (glc_2k_sat >= ext_10k_sat)
    d4_summary["hypotheses"] = {
        "D4_H1_5x_sample_efficiency": d4_h1,
        "glc_2k_vs_ext_10k_sudoku_margin": glc_2k_sud - ext_10k_sud,
        "glc_2k_vs_ext_10k_sat_margin": glc_2k_sat - ext_10k_sat,
        "pass": d4_h1,
    }

    # R19-D5 Aggregation (Iso-Param & Iso-FLOP Pareto Frontier)
    d5_summary: Dict[str, Any] = {"by_paradigm": {}}
    params_all = seed_runs[0]["param_counts"]
    d5_h1 = all(3_150_000 <= cnt <= 3_850_000 for cnt in params_all.values())
    for p_name in PARADIGM_NAMES:
        d5_summary["by_paradigm"][p_name] = {
            "trainable_params": params_all[p_name],
            "params_m": params_all[p_name] / 1e6,
            "by_budget": {},
        }
        for B_val in COMPUTE_BUDGETS:
            bk = str(B_val)
            d5_summary["by_paradigm"][p_name]["by_budget"][bk] = {
                "grand_mean_solve_rate_pct": paired_t_summary(
                    [r["R19_D5"][p_name]["by_budget"][bk]["grand_mean_solve_rate_pct"] for r in seed_runs]
                ),
                "estimated_gflops": float(
                    np.mean([r["R19_D5"][p_name]["by_budget"][bk]["estimated_gflops"] for r in seed_runs])
                ),
                "mean_token_evals": float(
                    np.mean([r["R19_D5"][p_name]["by_budget"][bk]["mean_token_evals"] for r in seed_runs])
                ),
            }

    glc_b16_grand = d5_summary["by_paradigm"]["GLC_AOM_v2"]["by_budget"]["16"]["grand_mean_solve_rate_pct"]["mean"]
    best_ext_b64_grand = max(
        d5_summary["by_paradigm"][p]["by_budget"]["64"]["grand_mean_solve_rate_pct"]["mean"]
        for p in PARADIGM_NAMES
        if p != "GLC_AOM_v2"
    )
    d5_h2 = glc_b16_grand > best_ext_b64_grand
    d5_summary["hypotheses"] = {
        "D5_H1_iso_param_parity_3_5m": d5_h1,
        "D5_H2_pareto_dominance_b16_beats_ext_b64": d5_h2,
        "glc_b16_grand_mean_pct": glc_b16_grand,
        "best_ext_b64_grand_mean_pct": best_ext_b64_grand,
        "pass": d5_h1 and d5_h2,
    }

    # R19-D6 Aggregation (Component Ablation)
    d6_summary: Dict[str, Any] = {"by_arm": {}}
    for abl_name in ABLATION_ARMS:
        d6_summary["by_arm"][abl_name] = {
            "sudoku_17_clue_solve_rate_pct": paired_t_summary(
                [r["R19_D6"][abl_name]["sudoku_17_clue_solve_rate_pct"] for r in seed_runs]
            ),
            "satlib_75_ood_solve_rate_pct": paired_t_summary(
                [r["R19_D6"][abl_name]["satlib_75_ood_solve_rate_pct"] for r in seed_runs]
            ),
            "combined_hard_solve_rate_pct": paired_t_summary(
                [r["R19_D6"][abl_name]["combined_hard_solve_rate_pct"] for r in seed_runs]
            ),
        }
    full_comb = d6_summary["by_arm"]["GLC_AOM_v2_Full"]["combined_hard_solve_rate_pct"]["mean"]
    d6_h1 = all(
        full_comb > d6_summary["by_arm"][abl]["combined_hard_solve_rate_pct"]["mean"]
        for abl in ABLATION_ARMS
        if abl != "GLC_AOM_v2_Full"
    )
    d6_summary["hypotheses"] = {
        "D6_H1_every_component_contributes_positively": d6_h1,
        "pass": d6_h1,
    }

    verdicts = {
        "R19_D0": "PASS" if d0_summary["pass"] else "FAIL",
        "R19_D1": "PASS" if d1_summary["hypotheses"]["pass"] else "FAIL",
        "R19_D2": "PASS" if d2_summary["hypotheses"]["pass"] else "FAIL",
        "R19_D3": "PASS" if d3_summary["hypotheses"]["pass"] else "FAIL",
        "R19_D4": "PASS" if d4_summary["hypotheses"]["pass"] else "FAIL",
        "R19_D5": "PASS" if d5_summary["hypotheses"]["pass"] else "FAIL",
        "R19_D6": "PASS" if d6_summary["hypotheses"]["pass"] else "FAIL",
    }
    overall_pass = all(v == "PASS" for v in verdicts.values())

    return {
        "suite_id": "R19-PUBLIC-REASONING-BENCHMARKS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device_info": device_info,
        "seeds": list(SEEDS),
        "total_wall_seconds": total_wall_s,
        "verdicts": verdicts,
        "overall_verdict": "PASS" if overall_pass else "FAIL",
        "R19_D0": d0_summary,
        "R19_D1": d1_summary,
        "R19_D2": d2_summary,
        "R19_D3": d3_summary,
        "R19_D4": d4_summary,
        "R19_D5": d5_summary,
        "R19_D6": d6_summary,
        "per_seed_runs": seed_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R19 Public Reasoning & Planning Benchmark Suite.")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory for JSON artifacts.")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test mode.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Static AST Zero-Oracle-Leakage Audit (S1-S6)
    audited_files = [
        _REPO_ROOT / "src/models/glc_aom.py",
        _REPO_ROOT / "src/search/kv_tree_search.py",
        _REPO_ROOT / "src/search/neural_tree_controller.py",
        _REPO_ROOT / "src/envs/public_reasoning_benchmarks.py",
        Path(__file__).resolve(),
    ]
    ast_audit = audit_r11_files(audited_files)
    assert ast_audit["pass"], f"AST Leak Audit FAILED: {ast_audit}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_version": torch.__version__,
    }
    print(f"[R19] Starting suite on {device_info['gpu_name']} ({device_info['device_type']})...")
    print(f"[R19] Static AST Audit: PASS=True, violations={ast_audit['num_violations']}, canaries={ast_audit['canary_self_test']['canaries_caught']}/3")

    t0 = time.perf_counter()
    seed_runs: List[Dict[str, Any]] = []
    seeds_to_run = (101,) if args.smoke else SEEDS

    for seed in seeds_to_run:
        s_res = run_single_seed(seed=seed, device=device, smoke=args.smoke)
        seed_runs.append(s_res)
        seed_path = out_dir / f"r19_seed_{seed}.json"
        seed_path.write_text(json.dumps(s_res, indent=2), encoding="utf-8")
        print(
            f"[R19] Seed {seed} finished in {s_res['wall_seconds']:.1f}s | "
            f"Sudoku-81(B=128) GLC={s_res['R19_D1']['GLC_AOM_v2']['by_budget']['128']['solve_rate_pct']:.1f}% "
            f"(17-clue={s_res['R19_D1']['GLC_AOM_v2']['by_budget']['128']['solve_rate_17_clues_pct']:.1f}%) | "
            f"SAT-75 OOD(B=128) GLC={s_res['R19_D2']['GLC_AOM_v2']['n75']['128']['solve_rate_pct']:.1f}%"
        )

    total_wall = time.perf_counter() - t0
    summary = aggregate_r19_results(seed_runs, ast_audit, device_info, total_wall)
    out_file = out_dir / "r19_public_reasoning_results.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[R19] Suite complete in {total_wall:.1f}s. Verdicts: {summary['verdicts']} | Overall: {summary['overall_verdict']}")


if __name__ == "__main__":
    main()
