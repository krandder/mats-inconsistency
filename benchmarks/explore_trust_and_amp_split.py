#!/usr/bin/env python3
"""
Track E Exploratory Runner (PROTOCOL.md §10.1):
Three-Parameter Separation: params_real (theta_base + theta_trust), params_search, and params_amp
(explore_trust_and_amp_split.py)

EXPLORATORY ONLY — strictly zero touch of confirmatory test.csv or test splits.
Evaluates:
  1. params_real (chi=(0,1)):
     - theta_base: Base Any-Order World Model q_real(X_{bar{S}} | x_S) trained strictly on real data x* ~ D_real.
     - theta_trust: Confiability Head ("trained from search into reality") taking detached branch features
       phi(C_j) = (depth d_j, entropy H_j, local self-consistency m_ql(C_j), path surprise S_j,
       cross-branch disagreement D_j) -> positive reliability weight beta_trust(C_j) > 0, trained strictly
       against real ground-truth data x* ~ D_real via proper scoring rule (NLL + Brier) of cleared belief
       mu_B^clear(x*_{bar{S}} | x_S; theta_trust) with theta_base frozen/detached.
  2. params_search (chi=(1,1)):
     - Selects cells/branches C_j to expand during internal search (B in {4, 16, 32}) using VoI - Entropy
       + consistency scoring to minimize internal inconsistency and cross-order variance.
  3. params_amp (chi=(0,1)):
     - Separate 1-pass head/adapter q_amp(X_{bar{S}} | x_S) trained solely via AlphaZero-style KL distillation
       KL(stopgrad[mu_B^clear(X_{bar{S}} | x_S; theta_trust)] || q_amp(X_{bar{S}} | x_S)) on unlabeled
       contexts x_S with zero contamination of params_real.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    torch.set_num_threads(4)

from external.infiloop.sudoku_data import read_csv as read_sudoku_csv
from external.infiloop.sudoku_model import rms_norm as infiloop_rms_norm

SUDOKU_EXTREME_TRAIN_SHA256 = "64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e"
INFILOOP_DATA_PY_SHA256 = "7884d37a2ae6a9be2f77727fd7d2ad1e8995d359d8f4118014fc4306bace9030"
INFILOOP_MODEL_PY_SHA256 = "44341a493e010548f8bdc36943523e345935705adea8797844d4764696e27f63"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_and_verify_sudoku_extreme_train_only(data_dir: str) -> Tuple[Path, Dict[str, str]]:
    candidate_dirs = [
        Path(data_dir),
        Path(_REPO_ROOT) / "data" / "sudoku-extreme",
        Path(_REPO_ROOT) / "data" / "sudoku_extreme",
        Path(_REPO_ROOT),
        Path("/tmp/r27a-run/data/sudoku-extreme"),
        Path("/tmp/sudoku-extreme"),
    ]
    train_path: Optional[Path] = None
    for cand in candidate_dirs:
        if (cand / "train.csv").exists():
            train_path = cand / "train.csv"
            break
    if train_path is None:
        dl_dir = Path("/tmp/sudoku-extreme")
        dl_dir.mkdir(parents=True, exist_ok=True)
        train_path = dl_dir / "train.csv"
        url = "https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/train.csv"
        urllib.request.urlretrieve(url, str(train_path))

    actual_hash = compute_sha256(train_path)
    if actual_hash != SUDOKU_EXTREME_TRAIN_SHA256:
        raise RuntimeError(f"SHA256 mismatch for {train_path}: got {actual_hash}")

    repo_root = Path(_REPO_ROOT)
    data_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_data.py")
    model_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_model.py")
    if data_py_hash != INFILOOP_DATA_PY_SHA256 or model_py_hash != INFILOOP_MODEL_PY_SHA256:
        raise RuntimeError("Upstream external/infiloop files failed SHA256 integrity verification.")

    return train_path, {
        "train.csv": actual_hash,
        "external/infiloop/sudoku_data.py": data_py_hash,
        "external/infiloop/sudoku_model.py": model_py_hash,
        "test_csv_touched": False,
    }


def compute_ece(probs_flat: np.ndarray, targets_flat: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) over predicted probabilities on unconditioned cells."""
    if len(targets_flat) == 0:
        return 0.0
    confidences = probs_flat.max(axis=-1)
    predictions = probs_flat.argmax(axis=-1)
    accuracies = (predictions == targets_flat).astype(np.float64)
    bin_bounds = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = float(len(targets_flat))
    for b in range(n_bins):
        lo, hi = bin_bounds[b], bin_bounds[b + 1]
        mask = (confidences >= lo) & (confidences <= hi) if b == 0 else (confidences > lo) & (confidences <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n_total) * abs(float(accuracies[mask].mean()) - float(confidences[mask].mean()))
    return float(ece)


def build_sudoku_unit_indices(device: torch.device) -> torch.Tensor:
    units: List[List[int]] = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + dr) * 9 + (bc * 3 + dc) for dr in range(3) for dc in range(3)])
    return torch.tensor(units, dtype=torch.long, device=device)


def count_sudoku_unit_collisions(preds_81: np.ndarray, unit_idx_np: np.ndarray) -> np.ndarray:
    u_vals = preds_81[:, unit_idx_np]
    eq = u_vals[:, :, :, None] == u_vals[:, :, None, :]
    pair_dups = (eq.sum(axis=(2, 3)) - 9) // 2
    return pair_dups.sum(axis=1).astype(np.float64)


class ConfiabilityTrustHead(nn.Module):
    """
    theta_trust in params_real (or self-gamed ablation in params_search):
    Maps detached 5D branch features phi(C_j) = (depth d_j, entropy H_j, local self-consistency m_ql(C_j),
    path surprise S_j relative to theta_base, cross-branch disagreement D_j)
    to a positive branch reliability weight beta_trust(C_j) > 0 and a calibrated temperature tau_trust > 0.
    """

    def __init__(self, feat_dim: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Positive feature precision weights via softplus: [depth, entropy, m_ql, path_surprise, disagreement]
        init_prec = torch.tensor([0.05, 0.25, 2.50, 0.25, 0.20], dtype=torch.float32)
        self.prec_raw = nn.Parameter(torch.log(torch.exp(init_prec) - 1.0))
        # Initialize temperature tau = softplus(0.54132) = 1.0
        self.log_tau = nn.Parameter(torch.tensor(0.54132485, dtype=torch.float32))
        self.base_mix_logit = nn.Parameter(torch.logit(torch.tensor(0.85, dtype=torch.float32)))
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, branch_feats: torch.Tensor, branch_logits: torch.Tensor, base_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        branch_feats: [nb, B, 5] detached features (d_j, H_j, m_ql(C_j), S_j, D_j)
        branch_logits: [nb, B, V, D] detached log-probabilities from q_real(. | C_j)
        base_logits: [nb, V, D] detached logits or log-probabilities from q_real(. | x_S)
        Returns:
          cleared_probs: [nb, V, D]
          weights: [nb, B]
          beta_trust: [nb, B] positive reliability score
        """
        f_min = branch_feats.min(dim=1, keepdim=True).values
        f_shift = branch_feats - f_min
        f_mean = branch_feats.mean(dim=1, keepdim=True)
        f_std = branch_feats.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-3)
        f_norm = (branch_feats - f_mean) / f_std

        prec_w = F.softplus(self.prec_raw)  # [5] > 0
        raw_score = -(f_shift * prec_w).sum(dim=-1) + 0.25 * self.net(f_norm).squeeze(-1)  # [nb, B]
        beta_trust = F.softplus(raw_score) + 1e-4
        weights = F.softmax(raw_score, dim=-1)  # [nb, B]

        tau = F.softplus(self.log_tau).clamp(min=0.25, max=3.5)
        branch_log_p = F.log_softmax(branch_logits, dim=-1)
        base_log_p = F.log_softmax(base_logits, dim=-1)
        cal_branch_probs = F.softmax(branch_log_p / tau, dim=-1)
        cal_base_probs = F.softmax(base_log_p / tau, dim=-1)

        cleared_branch = (weights.unsqueeze(-1).unsqueeze(-1) * cal_branch_probs).sum(dim=1)
        mix = self.base_mix_logit.sigmoid()
        cleared_probs = mix * cleared_branch + (1.0 - mix) * cal_base_probs
        cleared_probs = cleared_probs.clamp(min=1e-6)
        cleared_probs = cleared_probs / cleared_probs.sum(dim=-1, keepdim=True)
        return cleared_probs, weights, beta_trust


class AnyOrderSudokuNet(nn.Module):
    """
    Contains:
      - theta_base (in params_real): Base Any-Order World Model q_real(X_{bar{S}} | x_S)
      - theta_amp (in params_amp): Separate 1-pass Amplification Distillation Head/Adapter q_amp(X_{bar{S}} | x_S)
        that reads detached representations from theta_base + peer features so training theta_amp has
        ZERO gradient flow into theta_base or theta_trust.
    """

    def __init__(self, hidden_size: int = 96, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size
        self.clue_emb = nn.Embedding(10, hidden_size)
        self.pos_emb = nn.Embedding(81, hidden_size)
        nn.init.normal_(self.clue_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.prob_proj = nn.Linear(99, hidden_size)
        self.port_send = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.port_recv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 2 + 27, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(3)
            ]
        )
        self.cell_mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.head = nn.Linear(hidden_size, 9)
        self.peer_logit_proj = nn.Linear(99, 9)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.peer_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.peer_logit_proj.bias)
        self.q_score = nn.Parameter(torch.zeros(hidden_size))
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(0.85)))
        self.register_buffer("pos_idx", torch.arange(81, dtype=torch.long, device=device))

        # Separate params_amp adapter (trained strictly via KL distillation from stopgrad[mu_B^clear])
        self.amp_adapter = nn.Sequential(
            nn.Linear(hidden_size + 99 + 9, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 9),
        )
        self.amp_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        for m in self.amp_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def base_parameters(self) -> List[nn.Parameter]:
        amp_ids = {id(p) for p in self.amp_parameters()}
        return [p for p in self.parameters() if id(p) not in amp_ids]

    def amp_parameters(self) -> List[nn.Parameter]:
        return list(self.amp_adapter.parameters()) + [self.amp_scale]

    @staticmethod
    def _port_peer_sum(t: torch.Tensor, p_idx: int) -> torch.Tensor:
        bsz, _, ch = t.shape
        if p_idx == 0:
            return t.view(bsz, 9, 9, ch).sum(dim=2, keepdim=True).expand(bsz, 9, 9, ch).reshape(bsz, 81, ch) - t
        if p_idx == 1:
            return t.view(bsz, 9, 9, ch).sum(dim=1, keepdim=True).expand(bsz, 9, 9, ch).reshape(bsz, 81, ch) - t
        return (
            t.view(bsz, 3, 3, 3, 3, ch)
            .sum(dim=(2, 4), keepdim=True)
            .expand(bsz, 3, 3, 3, 3, ch)
            .reshape(bsz, 81, ch)
            - t
        )

    def _unit_messages(
        self, h: torch.Tensor, probs: torch.Tensor, clue_oh9: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = []
        peer_ps = [probs, clue_oh9]
        for p_idx in range(3):
            p_other = self._port_peer_sum(probs, p_idx)
            c_other = self._port_peer_sum(clue_oh9, p_idx)
            u_def = (1.0 - p_other - c_other).clamp(min=0.0, max=1.0)
            peer_ps.extend([p_other, c_other, u_def])
            m = self.port_send[p_idx](h)
            m_other = self._port_peer_sum(m, p_idx) / 8.0
            outs.append(self.port_recv[p_idx](torch.cat([m, m_other, p_other, c_other, u_def], dim=-1)))
        return torch.cat(outs, dim=-1), torch.cat(peer_ps, dim=-1)

    def forward_with_state(
        self, x_digits: torch.Tensor, n_steps: int = 5
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        pos = self.pos_emb(self.pos_idx).unsqueeze(0)
        c_emb = self.clue_emb(x_digits.clamp(0, 9)) + pos
        clue_mask = (x_digits > 0).unsqueeze(-1).float()
        clue_oh9 = F.one_hot((x_digits - 1).clamp(0, 8), num_classes=9).float() * clue_mask

        h = infiloop_rms_norm(c_emb, 1e-6)
        probs = clue_oh9 + (1.0 - clue_mask) * (1.0 / 9.0)
        s0 = (h.detach() @ self.q_score).clamp(-6.0, 6.0).exp().unsqueeze(-1)
        num = s0 * h
        den = s0
        beta = self.beta_logit.sigmoid()

        step_logits: List[torch.Tensor] = []
        peer_p_cat = torch.zeros(x_digits.shape[0], 81, 99, device=x_digits.device)
        for _ in range(n_steps):
            u_msg, peer_p_cat = self._unit_messages(h, probs, clue_oh9)
            x = infiloop_rms_norm(h + c_emb + self.prob_proj(peer_p_cat), 1e-6)
            z = infiloop_rms_norm(self.cell_mlp(torch.cat([x, u_msg], dim=-1)), 1e-6)
            s = (z.detach() @ self.q_score).clamp(-6.0, 6.0).exp().unsqueeze(-1)
            num = beta * num + 0.35 * s * z
            den = beta * den + 0.35 * s
            h = num / den.clamp(min=1e-7)
            logits = self.head(h) + self.peer_logit_proj(peer_p_cat)
            step_logits.append(logits)
            probs = F.softmax(logits.detach(), dim=-1)
        return step_logits, h, peer_p_cat

    def forward(self, x_digits: torch.Tensor, n_steps: int = 5) -> List[torch.Tensor]:
        step_logits, _, _ = self.forward_with_state(x_digits, n_steps=n_steps)
        return step_logits

    def forward_amp(self, x_digits: torch.Tensor, n_steps: int = 5) -> torch.Tensor:
        """
        1-pass evaluation of params_amp: detached base pass + learned amplification distillation head.
        Guarantees zero gradient contamination of theta_base.
        """
        with torch.no_grad():
            step_logits, h, peer_p_cat = self.forward_with_state(x_digits, n_steps=n_steps)
            base_logits = step_logits[-1].detach()
            base_probs = F.softmax(base_logits, dim=-1)
            h_det = h.detach()
            peer_det = peer_p_cat.detach()
        amp_in = torch.cat([h_det, peer_det, base_probs], dim=-1)
        delta_logits = self.amp_adapter(amp_in)
        return self.amp_scale.clamp(min=0.2, max=3.0) * base_logits + delta_logits


def sudoku_anyorder_birkhoff_contradiction_loss(
    logits: torch.Tensor, x_cond: torch.Tensor, unit_idx: torch.Tensor
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    unit_probs = probs[:, unit_idx, :]
    unit_digit_sum = unit_probs.sum(dim=2)
    birkhoff_col_loss = ((unit_digit_sum - 1.0) ** 2).mean()

    unit_sum_sq = (unit_digit_sum * unit_digit_sum).sum(dim=-1)
    unit_cell_sq = (unit_probs * unit_probs).sum(dim=(2, 3))
    unit_pair_ip = ((unit_sum_sq - unit_cell_sq) / 8.0).mean()

    clue_oh = F.one_hot(x_cond.clamp(min=0, max=9), num_classes=10)[:, :, 1:].to(probs.dtype)
    unit_clues = clue_oh[:, unit_idx, :]
    unit_clue_other = (unit_clues.sum(dim=2, keepdim=True) - unit_clues).clamp(max=1.0)
    bad_prob = (unit_probs * unit_clue_other).sum(dim=-1).clamp(max=1.0 - 1e-5)
    peer_clue_loss = -torch.log(1.0 - bad_prob).mean()

    row_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 0)
    col_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 1)
    box_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 2)
    peer_c_mask = (row_c + col_c + box_c).clamp(max=1.0)
    empty_mask = (x_cond == 0).unsqueeze(-1).float()
    logit_margin_loss = (F.relu(logits + 5.0) * peer_c_mask * empty_mask).mean()

    cond_mask = x_cond > 0
    if cond_mask.any():
        clue_ce = F.cross_entropy(logits[cond_mask], (x_cond[cond_mask] - 1).clamp(min=0))
    else:
        clue_ce = torch.tensor(0.0, device=logits.device)

    return 1.5 * birkhoff_col_loss + unit_pair_ip + 2.5 * peer_clue_loss + 0.5 * logit_margin_loss + clue_ce


def extract_sudoku_branch_features(
    leaf_probs: torch.Tensor,
    base_probs: torch.Tensor,
    x_branch: torch.Tensor,
    log_prior: torch.Tensor,
    unit_idx: torch.Tensor,
    rollout_depth: int,
) -> torch.Tensor:
    """
    Computes detached 5D branch features phi(C_j) for each branch j=1..B:
      0. d_j: normalized rollout depth / fraction of conditioned cells in C_j
      1. H_j: mean predictive entropy on unconditioned cells in C_j
      2. m_ql(C_j): local self-consistency violation energy (unit collision + dead-end depletion)
      3. S_j: path surprise (-log_prior / depth + KL(q_real(.|C_j) || q_real(.|x_S)))
      4. D_j: cross-branch disagreement JSD / KL to the unweighted branch consensus bar{q}
    Returns tensor of shape [nb, B, 5].
    """
    nb, n_branches, _, _ = leaf_probs.shape
    flat_p = leaf_probs.reshape(nb * n_branches, 81, 9)
    unit_p = flat_p[:, unit_idx, :]
    unit_digit_sum = unit_p.sum(dim=2)
    excess = F.relu(unit_digit_sum - 1.0)
    unit_coll = (excess * excess).sum(dim=(1, 2)).view(nb, n_branches)

    uncond_mask = (x_branch == 0).float()
    uncond_denom = uncond_mask.sum(dim=-1).clamp(min=1.0)

    p_cl = leaf_probs.clamp(min=1e-6)
    cell_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
    mean_ent = (cell_ent * uncond_mask).sum(dim=-1) / uncond_denom

    max_p = leaf_probs.max(dim=-1).values.clamp(min=1e-6)
    deplete = ((-torch.log(max_p) * uncond_mask).sum(dim=-1) / uncond_denom) * 10.0
    m_ql = unit_coll + deplete

    base_exp = base_probs.unsqueeze(1).expand_as(leaf_probs).clamp(min=1e-6)
    kl_base = (p_cl * (torch.log(p_cl) - torch.log(base_exp))).sum(dim=-1)
    path_kl = (kl_base * uncond_mask).sum(dim=-1) / uncond_denom
    surprise = (-log_prior / float(max(1, rollout_depth))) + path_kl

    mean_branch_p = leaf_probs.mean(dim=1, keepdim=True).expand_as(leaf_probs).clamp(min=1e-6)
    kl_consensus = (p_cl * (torch.log(p_cl) - torch.log(mean_branch_p))).sum(dim=-1)
    disagreement = (kl_consensus * uncond_mask).sum(dim=-1) / uncond_denom

    depth_feat = ((x_branch > 0).float().mean(dim=-1)) * float(rollout_depth)
    return torch.stack([depth_feat, mean_ent, m_ql, surprise, disagreement], dim=-1).detach()


@torch.no_grad()
def expand_sudoku_search_branches(
    model: AnyOrderSudokuNet,
    x_cond: torch.Tensor,
    unit_idx: torch.Tensor,
    n_branches: int = 16,
    rollout_depth: int = 2,
    n_steps: int = 4,
    use_amp_proposal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs internal search expansion (params_search, chi=(1,1)) using either q_real or q_amp as proposal,
    while always evaluating leaf predictions with q_real (theta_base) so clearing remains anchored in reality.
    Returns:
      branch_logits: [nb, B, 81, 9]
      base_logits: [nb, 81, 9]
      branch_feats: [nb, B, 5]
      x_branch: [nb, B, 81]
    """
    model.eval()
    nb = x_cond.shape[0]
    device = x_cond.device
    base_logits = model(x_cond, n_steps=n_steps)[-1]
    base_probs = F.softmax(base_logits, dim=-1)

    if use_amp_proposal:
        amp_logits = model.forward_amp(x_cond, n_steps=n_steps)
        amp_probs = F.softmax(amp_logits, dim=-1)
        prop_probs = 0.5 * amp_probs + 0.5 * base_probs
        anchor_logits = torch.log(prop_probs.clamp(min=1e-6))
    else:
        prop_probs = base_probs
        anchor_logits = torch.log(base_probs.clamp(min=1e-6))

    r_bits = {4: 2, 16: 4, 32: 5}.get(n_branches, 4)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)],
        dtype=torch.long,
        device=device,
    )

    p_clamped = prop_probs.clamp(min=1e-7)
    cell_ent = -(p_clamped * torch.log(p_clamped)).sum(dim=-1)
    top2_vals, top2_idx = prop_probs.topk(2, dim=-1)

    # VoI - Entropy + peer constraint density score (params_search)
    clue_oh = F.one_hot(x_cond.clamp(min=0, max=9), num_classes=10)[:, :, 1:].float()
    peer_density = (
        AnyOrderSudokuNet._port_peer_sum(clue_oh, 0)
        + AnyOrderSudokuNet._port_peer_sum(clue_oh, 1)
        + AnyOrderSudokuNet._port_peer_sum(clue_oh, 2)
    ).clamp(max=1.0).sum(dim=-1) / 9.0
    is_empty = x_cond == 0
    voi_score = (cell_ent - 0.25 * peer_density) + torch.where(
        is_empty, torch.zeros_like(cell_ent), torch.full_like(cell_ent, 1e6)
    )
    voi_score = voi_score + torch.where(
        top2_vals[:, :, 0] > 0.92, torch.full_like(cell_ent, 10.0), torch.zeros_like(cell_ent)
    )
    root_cells = voi_score.topk(r_bits, dim=-1, largest=False).indices

    root_top2_digits = top2_idx.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)) + 1
    root_top2_probs = top2_vals.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)).clamp(min=1e-5)

    x_branch = x_cond.unsqueeze(1).expand(nb, n_branches, 81).clone()
    log_prior = torch.zeros(nb, n_branches, device=device)

    for bit_i in range(r_bits):
        choice_bit = bit_table[:, bit_i].unsqueeze(0).expand(nb, n_branches)
        cell_i = root_cells[:, bit_i].unsqueeze(1).expand(nb, n_branches)
        cand_d = root_top2_digits[:, bit_i, :].gather(1, choice_bit)
        cand_p = root_top2_probs[:, bit_i, :].gather(1, choice_bit)
        cell_was_empty = is_empty.gather(1, root_cells[:, bit_i : bit_i + 1]).expand(nb, n_branches)
        cur_val = x_branch.gather(2, cell_i.unsqueeze(-1)).squeeze(-1)
        new_val = torch.where(cell_was_empty, cand_d, cur_val)
        x_branch.scatter_(2, cell_i.unsqueeze(-1), new_val.unsqueeze(-1))
        log_prior = log_prior + torch.where(cell_was_empty, torch.log(cand_p), torch.zeros_like(cand_p))

    x_after_root = x_branch.clone()
    cells_per_step = 2 if n_branches <= 4 else (3 if n_branches <= 16 else 4)
    for _ in range(rollout_depth - 1):
        flat_x = x_branch.reshape(nb * n_branches, 81)
        if use_amp_proposal:
            step_logits = model.forward_amp(flat_x, n_steps=max(3, n_steps - 1))
        else:
            step_logits = model(flat_x, n_steps=max(3, n_steps - 1))[-1]
        step_probs = F.softmax(step_logits, dim=-1).view(nb, n_branches, 81, 9)
        sp_clamped = step_probs.clamp(min=1e-7)
        sp_ent = -(sp_clamped * torch.log(sp_clamped)).sum(dim=-1)
        rem_empty = x_branch == 0
        sp_score = sp_ent + torch.where(rem_empty, torch.zeros_like(sp_ent), torch.full_like(sp_ent, 1e6))
        next_cells = sp_score.topk(cells_per_step, dim=-1, largest=False).indices
        best_digits = step_probs.argmax(dim=-1) + 1
        picked_digits = best_digits.gather(2, next_cells)
        was_empty = rem_empty.gather(2, next_cells)
        cur_vals = x_branch.gather(2, next_cells)
        upd_vals = torch.where(was_empty, picked_digits, cur_vals)
        x_branch.scatter_(2, next_cells, upd_vals)

    # Grounded leaf evaluation ALWAYS uses theta_base (q_real)
    leaf_logits = model(x_branch.reshape(nb * n_branches, 81), n_steps=n_steps)[-1].view(nb, n_branches, 81, 9)
    leaf_probs = F.softmax(leaf_logits, dim=-1)

    root_added_mask = ((x_after_root > 0) & (x_cond.unsqueeze(1) == 0)).unsqueeze(-1).float()
    roll_added_mask = ((x_branch > 0) & (x_after_root == 0)).unsqueeze(-1).float()
    added_oh = F.one_hot((x_branch - 1).clamp(0, 8), num_classes=9).float()

    branch_probs = (
        root_added_mask * (0.30 * added_oh + 0.70 * leaf_probs)
        + roll_added_mask * (0.15 * added_oh + 0.85 * leaf_probs)
        + (1.0 - root_added_mask - roll_added_mask) * leaf_probs
    ).clamp(min=1e-6)
    branch_probs = branch_probs / branch_probs.sum(dim=-1, keepdim=True)
    branch_logits = torch.log(branch_probs)

    branch_feats = extract_sudoku_branch_features(
        branch_probs, base_probs, x_branch, log_prior, unit_idx, rollout_depth=rollout_depth
    )
    return branch_logits.detach(), anchor_logits.detach(), branch_feats.detach(), x_branch.detach()


@torch.no_grad()
def predict_sudoku_condition(
    condition: str,
    model: Optional[AnyOrderSudokuNet],
    trust_head: Optional[ConfiabilityTrustHead],
    x_cond: torch.Tensor,
    unit_idx: torch.Tensor,
    n_branches: int = 16,
    n_steps: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    nb = x_cond.shape[0]
    device = x_cond.device
    eps = 1e-4

    if condition == "C1_Control_PeerClue_Uniform":
        clue_oh = F.one_hot(x_cond.clamp(min=0, max=9), num_classes=10)[:, :, 1:].float()
        row_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 0)
        col_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 1)
        box_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 2)
        allowed = (1.0 - (row_c + col_c + box_c).clamp(max=1.0)).clamp(min=0.0)
        allowed_sum = allowed.sum(dim=-1, keepdim=True)
        uniform9 = torch.full_like(allowed, 1.0 / 9.0)
        probs = torch.where(allowed_sum > 0.5, allowed / allowed_sum.clamp(min=1.0), uniform9)
        probs = (1.0 - eps) * probs + (eps / 9.0)
        return probs / probs.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    assert model is not None
    model.eval()

    if condition == "M1_ParamsReal_Base_B0":
        base_logits = model(x_cond, n_steps=n_steps)[-1]
        base_probs = F.softmax(base_logits, dim=-1)
        out = (1.0 - eps) * base_probs + (eps / 9.0)
        return out / out.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    if condition == "M5_ParamsAmp_Distilled_B0":
        amp_logits = model.forward_amp(x_cond, n_steps=n_steps)
        amp_probs = F.softmax(amp_logits, dim=-1)
        out = (1.0 - eps) * amp_probs + (eps / 9.0)
        return out / out.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    use_amp_prop = condition == "M6_ParamsAmp_Plus_Search_B16"
    branch_logits, base_logits, branch_feats, _ = expand_sudoku_search_branches(
        model, x_cond, unit_idx, n_branches=n_branches, rollout_depth=2, n_steps=n_steps, use_amp_proposal=use_amp_prop
    )

    if condition == "M2_Search_Unweighted_Clear_B16":
        branch_probs = F.softmax(branch_logits, dim=-1)
        base_probs = F.softmax(base_logits, dim=-1)
        weights = torch.full((nb, n_branches), 1.0 / float(n_branches), device=device)
        cleared = (weights.unsqueeze(-1).unsqueeze(-1) * branch_probs).sum(dim=1)
        out = 0.85 * cleared + 0.15 * base_probs
        out = (1.0 - eps) * out + (eps / 9.0)
        return out / out.sum(dim=-1, keepdim=True), {
            "effective_branch_ess": float(n_branches),
            "mean_beta_trust": 1.0,
        }

    assert trust_head is not None
    trust_head.eval()
    cleared_probs, weights, beta_trust = trust_head(branch_feats, branch_logits, base_logits)
    out = (1.0 - eps) * cleared_probs + (eps / 9.0)
    out = out / out.sum(dim=-1, keepdim=True)
    ess = float((1.0 / (weights * weights).sum(dim=-1).clamp(min=1e-6)).mean().item())
    return out, {
        "effective_branch_ess": ess,
        "mean_beta_trust": float(beta_trust.mean().item()),
    }


@torch.no_grad()
def evaluate_sudoku_condition(
    condition: str,
    model: Optional[AnyOrderSudokuNet],
    trust_head: Optional[ConfiabilityTrustHead],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    unit_idx: torch.Tensor,
    device: torch.device,
    n_branches: int = 16,
    k_random_orders: int = 4,
    n_reveal_blocks: int = 6,
    seed: int = 101,
) -> Dict[str, Any]:
    nb = x_eval_np.shape[0]
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)
    unit_idx_np = unit_idx.cpu().numpy()

    def get_probs(x_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        return predict_sudoku_condition(
            condition, model, trust_head, x_cur, unit_idx, n_branches=n_branches, n_steps=4
        )

    # 1. Step-0 initial evaluation & ECE calibration error on unconditioned cells
    p0, step0_stats = get_probs(x_init)
    empty_init_mask = x_eval_np == 0
    clue_init_mask = x_eval_np > 0
    p0_np = p0.cpu().numpy()
    preds_step0 = p0_np.argmax(axis=-1) + 1
    preds_step0[clue_init_mask] = x_eval_np[clue_init_mask]
    one_shot_empty_acc = float(((preds_step0 == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))

    emp_probs_flat = p0_np[empty_init_mask]
    emp_targets_flat = (y_eval_np[empty_init_mask] - 1).clip(0, 8)
    ece_val = compute_ece(emp_probs_flat, emp_targets_flat, n_bins=10)

    # Brier score on empty cells
    emp_oh = np.eye(9, dtype=np.float64)[emp_targets_flat]
    brier_per_cell = float(np.mean(np.sum((emp_probs_flat - emp_oh) ** 2, axis=-1))) if len(emp_targets_flat) > 0 else 0.0

    # 2. Random-order NLL & PPL across K random orders
    board_order_total_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    board_order_cell_nll = np.zeros((nb, k_random_orders), dtype=np.float64)

    for k in range(k_random_orders):
        order_blocks: List[torch.Tensor] = [
            torch.zeros((nb, 81), dtype=torch.bool, device=device) for _ in range(n_reveal_blocks)
        ]
        empty_counts = np.zeros(nb, dtype=np.float64)
        for i in range(nb):
            rng_i = np.random.RandomState(seed * 10000 + i * 37 + k)
            emp_idx = np.where(x_eval_np[i] == 0)[0]
            empty_counts[i] = max(len(emp_idx), 1)
            perm_idx = emp_idx[rng_i.permutation(len(emp_idx))]
            chunks = np.array_split(perm_idx, n_reveal_blocks)
            for b_idx, ch in enumerate(chunks):
                if len(ch) > 0:
                    order_blocks[b_idx][i, ch] = True

        x_cur = x_init.clone()
        total_nll_k = torch.zeros(nb, dtype=torch.float64, device=device)
        for b_idx in range(n_reveal_blocks):
            b_mask = order_blocks[b_idx]
            if not b_mask.any():
                continue
            probs_step, _ = get_probs(x_cur)
            true_digits = (y_true - 1).clamp(0, 8)
            true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            step_nll = -torch.log(true_p).to(torch.float64) * b_mask.to(torch.float64)
            total_nll_k = total_nll_k + step_nll.sum(dim=-1)
            x_cur = torch.where(b_mask, y_true, x_cur)

        total_nll_np = total_nll_k.cpu().numpy()
        board_order_total_nll[:, k] = total_nll_np
        board_order_cell_nll[:, k] = total_nll_np / empty_counts

    rand_order_nll_per_cell = float(board_order_cell_nll.mean())
    rand_order_ppl = float(math.exp(min(rand_order_nll_per_cell, 20.0)))
    cross_order_nll_std = float(np.std(board_order_total_nll, axis=1, ddof=1).mean()) if k_random_orders > 1 else 0.0

    # 3. Adaptive VoI / Minimum-Entropy Order NLL & PPL
    x_cur_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_scored_cells = torch.zeros(nb, dtype=torch.float64, device=device)

    for b_idx in range(n_reveal_blocks):
        rem_mask = x_cur_minh == 0
        if not rem_mask.any():
            break
        probs_step, _ = get_probs(x_cur_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        rem_counts = rem_mask.sum(dim=-1)
        blocks_left = n_reveal_blocks - b_idx
        k_pick_vec = ((rem_counts + blocks_left - 1) // blocks_left).clamp(min=1) if blocks_left > 1 else rem_counts
        ent_masked = ent + torch.where(rem_mask, torch.zeros_like(ent), torch.full_like(ent, 1e9))
        ranks = ent_masked.argsort(dim=-1).argsort(dim=-1)
        step_mask = (ranks < k_pick_vec.unsqueeze(-1)) & rem_mask

        true_digits = (y_true - 1).clamp(0, 8)
        true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        step_nll = -torch.log(true_p).to(torch.float64) * step_mask.to(torch.float64)
        minh_total_nll = minh_total_nll + step_nll.sum(dim=-1)
        minh_scored_cells = minh_scored_cells + step_mask.to(torch.float64).sum(dim=-1)
        x_cur_minh = torch.where(step_mask, y_true, x_cur_minh)

    min_h_voi_nll_per_cell = float((minh_total_nll / minh_scored_cells.clamp(min=1.0)).mean().item())
    min_h_voi_ppl = float(math.exp(min(min_h_voi_nll_per_cell, 20.0)))

    # 4. Autonomous Sequential Rollout along VoI / Minimum-Entropy Order
    x_auto = x_init.clone()
    for s_idx in range(n_reveal_blocks):
        rem_mask = x_auto == 0
        if not rem_mask.any():
            break
        probs_auto, _ = get_probs(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        pred_digits = probs_auto.argmax(dim=-1) + 1
        rem_counts = rem_mask.sum(dim=-1)
        steps_left = n_reveal_blocks - s_idx
        k_pick_vec = ((rem_counts + steps_left - 1) // steps_left).clamp(min=1) if steps_left > 1 else rem_counts
        ent_masked = ent + torch.where(rem_mask, torch.zeros_like(ent), torch.full_like(ent, 1e9))
        ranks = ent_masked.argsort(dim=-1).argsort(dim=-1)
        commit_mask = (ranks < k_pick_vec.unsqueeze(-1)) & rem_mask
        x_auto = torch.where(commit_mask, pred_digits, x_auto)

    auto_np = x_auto.cpu().numpy()
    autonomous_empty_acc = float(((auto_np == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))
    autonomous_collisions = float(count_sudoku_unit_collisions(auto_np, unit_idx_np).mean())

    return {
        "n_branches": n_branches,
        "rand_order_nll_per_cell": rand_order_nll_per_cell,
        "rand_order_ppl": rand_order_ppl,
        "cross_order_nll_std": cross_order_nll_std,
        "min_h_voi_nll_per_cell": min_h_voi_nll_per_cell,
        "min_h_voi_ppl": min_h_voi_ppl,
        "autonomous_empty_acc": autonomous_empty_acc,
        "autonomous_unit_collisions": autonomous_collisions,
        "one_shot_empty_acc": one_shot_empty_acc,
        "ece_calibration_error": ece_val,
        "brier_per_cell": brier_per_cell,
        "search_diagnostics": step0_stats,
    }


# ---------------------------------------------------------------------------
# 2. N-Queens (8x8 & 10x10 Dev Splits) with params_real / params_search / params_amp
# ---------------------------------------------------------------------------

def solve_all_nqueens(n: int, max_sols: int = 2000) -> List[Tuple[int, ...]]:
    sols: List[Tuple[int, ...]] = []
    per_c0_limit = max(1, math.ceil(max_sols / float(n)))
    for c0 in range(n):
        c0_sols: List[Tuple[int, ...]] = []
        cols = {c0}
        diag1 = {0 - c0}
        diag2 = {0 + c0}
        board = [-1] * n
        board[0] = c0

        def backtrack(r: int):
            if len(c0_sols) >= per_c0_limit or len(sols) + len(c0_sols) >= max_sols:
                return
            if r == n:
                c0_sols.append(tuple(board))
                return
            for c in range(n):
                if c not in cols and (r - c) not in diag1 and (r + c) not in diag2:
                    board[r] = c
                    cols.add(c)
                    diag1.add(r - c)
                    diag2.add(r + c)
                    backtrack(r + 1)
                    cols.remove(c)
                    diag1.remove(r - c)
                    diag2.remove(r + c)

        backtrack(1)
        sols.extend(c0_sols)
        if len(sols) >= max_sols:
            break
    return sols[:max_sols]


def d4_orbit_rep(sol: Tuple[int, ...]) -> Tuple[int, ...]:
    n = len(sol)
    pts = [(r, sol[r]) for r in range(n)]
    forms = []
    for rot in range(4):
        cur = pts
        for _ in range(rot):
            cur = [(c, n - 1 - r) for (r, c) in cur]
        for flip in (False, True):
            cur2 = [(r, n - 1 - c) if flip else (r, c) for (r, c) in cur]
            cur2_sorted = sorted(cur2, key=lambda item: item[0])
            forms.append(tuple(c for (_, c) in cur2_sorted))
    return min(forms)


def d4_clue_orbit_rep(clues: Tuple[Tuple[int, int], ...], n: int) -> Tuple[Tuple[int, int], ...]:
    forms = []
    for rot in range(4):
        cur = list(clues)
        for _ in range(rot):
            cur = [(c, n - 1 - r) for (r, c) in cur]
        for flip in (False, True):
            cur2 = [(r, n - 1 - c) if flip else (r, c) for (r, c) in cur]
            forms.append(tuple(sorted(cur2)))
    return min(forms)


def sample_nqueens_dev_split(
    sols_pool: List[Tuple[int, ...]],
    all_sols: List[Tuple[int, ...]],
    n: int,
    count: int,
    forbidden_orbits: Set[Tuple[Tuple[int, int], ...]],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Set[Tuple[Tuple[int, int], ...]]]:
    rng = np.random.RandomState(seed)
    sol_arr = np.array(all_sols, dtype=np.int64)
    xs: List[np.ndarray] = []
    primary_ys: List[np.ndarray] = []
    multi_ys: List[np.ndarray] = []
    used_orbits: Set[Tuple[Tuple[int, int], ...]] = set()
    attempts = 0
    while len(xs) < count and attempts < count * 150:
        attempts += 1
        s = sols_pool[rng.randint(len(sols_pool))]
        k = int(rng.choice([2, 3]))
        rows = sorted(rng.choice(n, size=k, replace=False).tolist())
        clues = tuple((int(r), int(s[r])) for r in rows)
        c_orb = d4_clue_orbit_rep(clues, n)
        if c_orb in forbidden_orbits or c_orb in used_orbits:
            continue
        mask = np.ones(len(sol_arr), dtype=bool)
        for rw, cl in clues:
            mask &= sol_arr[:, rw] == cl
        matching = sol_arr[mask]
        if len(matching) < 1:
            continue
        x = np.full(n, -1, dtype=np.int64)
        for rw, cl in clues:
            x[rw] = cl
        xs.append(x)
        primary_ys.append(np.array(s, dtype=np.int64))
        multi_ys.append(matching[:8])
        used_orbits.add(c_orb)
    return np.array(xs, dtype=np.int64), np.array(primary_ys, dtype=np.int64), multi_ys, used_orbits


def nqueens_peer_attack_mask(clues: torch.Tensor) -> torch.Tensor:
    bsz, n = clues.shape
    device = clues.device
    r_idx = torch.arange(n, device=device).view(1, n, 1, 1)
    r_clue = torch.arange(n, device=device).view(1, 1, n, 1)
    c_idx = torch.arange(n, device=device).view(1, 1, 1, n)
    c_clue = clues.unsqueeze(1).unsqueeze(-1)
    has_clue = (c_clue >= 0) & (r_idx != r_clue)
    same_col = has_clue & (c_idx == c_clue)
    same_diag = has_clue & ((r_idx - r_clue).abs() == (c_idx - c_clue).abs())
    return ((same_col | same_diag).any(dim=2) & (clues.unsqueeze(-1) < 0)).float()


def count_nqueens_metrics(cols_pred: np.ndarray, clues: np.ndarray) -> Dict[str, float]:
    bsz, n = cols_pred.shape
    total_coll = np.zeros(bsz, dtype=np.float64)
    valid_board = np.zeros(bsz, dtype=np.float64)
    for b in range(bsz):
        c = cols_pred[b]
        cl = clues[b]
        mask = cl >= 0
        clue_ok = float((c[mask] == cl[mask]).mean()) if mask.any() else 1.0
        cc, dc = 0, 0
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                if c[r1] == c[r2]:
                    cc += 1
                if abs(r1 - r2) == abs(int(c[r1]) - int(c[r2])):
                    dc += 1
        total_coll[b] = cc + dc
        valid_board[b] = 1.0 if (cc + dc == 0 and clue_ok == 1.0) else 0.0
    return {"mean_collisions": float(total_coll.mean()), "valid_board_rate": float(valid_board.mean())}


class AnyOrderNQueensNet(nn.Module):
    def __init__(self, d_model: int = 48):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(7, d_model)
        self.prob_proj = nn.Linear(6, d_model)
        self.row_proj = nn.Linear(d_model, d_model)
        self.col_proj = nn.Linear(d_model, d_model)
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=4, dilation=4)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, 1)
        self.peer_logit_proj = nn.Linear(4, 1)
        nn.init.normal_(self.out_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_head.bias)
        nn.init.normal_(self.peer_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.peer_logit_proj.bias)

        # Separate params_amp head for N-Queens
        self.amp_adapter = nn.Sequential(
            nn.Linear(d_model + 5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.amp_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        for m in self.amp_adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def base_parameters(self) -> List[nn.Parameter]:
        amp_ids = {id(p) for p in self.amp_parameters()}
        return [p for p in self.parameters() if id(p) not in amp_ids]

    def amp_parameters(self) -> List[nn.Parameter]:
        return list(self.amp_adapter.parameters()) + [self.amp_scale]

    def forward_with_state(
        self, clues: torch.Tensor, n_segments: int = 5, n_loops: int = 2
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        bsz, n = clues.shape
        device = clues.device
        t = torch.linspace(0.0, 1.0, n, device=device)
        rr, cc = torch.meshgrid(t, t, indexing="ij")
        is_clued_row = (clues >= 0).float().unsqueeze(-1).expand(bsz, n, n)
        col_idx = torch.arange(n, device=device).view(1, 1, n).expand(bsz, n, n)
        is_clue_cell = ((clues.unsqueeze(-1) == col_idx) & (clues.unsqueeze(-1) >= 0)).float()
        p_att = nqueens_peer_attack_mask(clues)
        x_feat = torch.stack(
            [
                rr.unsqueeze(0).expand(bsz, -1, -1),
                cc.unsqueeze(0).expand(bsz, -1, -1),
                torch.sin(math.pi * rr).unsqueeze(0).expand(bsz, -1, -1),
                torch.cos(math.pi * cc).unsqueeze(0).expand(bsz, -1, -1),
                is_clued_row,
                is_clue_cell,
                p_att,
            ],
            dim=-1,
        )
        x_emb = self.in_proj(x_feat)
        z = torch.zeros_like(x_emb)
        logits = torch.zeros(bsz, n, n, device=device)
        r_grid = torch.arange(n, device=device).view(1, n, 1).expand(bsz, n, n)
        c_grid = torch.arange(n, device=device).view(1, 1, n).expand(bsz, n, n)
        d1_idx = (r_grid - c_grid + (n - 1)).reshape(bsz, n * n)
        d2_idx = (r_grid + c_grid).reshape(bsz, n * n)
        seg_logits = []
        peer_stack = torch.zeros(bsz, n, n, 4, device=device)
        for _ in range(n_segments):
            z = z.detach()
            p = F.softmax(logits.detach(), dim=-1)
            col_other = p.sum(dim=1, keepdim=True).expand(bsz, n, n) - p
            p_flat = p.reshape(bsz, n * n)
            d1_buf = torch.zeros(bsz, 2 * n - 1, device=device, dtype=p.dtype)
            d2_buf = torch.zeros(bsz, 2 * n - 1, device=device, dtype=p.dtype)
            d1_buf.scatter_add_(1, d1_idx, p_flat)
            d2_buf.scatter_add_(1, d2_idx, p_flat)
            diag1_other = d1_buf.gather(1, d1_idx).view(bsz, n, n) - p
            diag2_other = d2_buf.gather(1, d2_idx).view(bsz, n, n) - p
            row_max = p.max(dim=-1, keepdim=True).values.expand(bsz, n, n)
            p_feat = self.prob_proj(
                torch.stack([p, col_other, diag1_other, diag2_other, row_max, logits.detach().tanh()], dim=-1)
            )
            for _ in range(n_loops):
                h = z + x_emb + p_feat
                r_ctx = self.row_proj(h.mean(dim=2, keepdim=True))
                c_ctx = self.col_proj(h.mean(dim=1, keepdim=True))
                h_chw = h.permute(0, 3, 1, 2)
                c_out = (self.conv1(h_chw) + self.conv2(h_chw) + self.conv3(h_chw)).permute(0, 2, 3, 1)
                z = self.norm(z + F.gelu(r_ctx + c_ctx + c_out))
                z = z + self.mlp(z)
            peer_stack = torch.stack([is_clue_cell, p_att, col_other, diag1_other + diag2_other], dim=-1)
            logits = self.out_head(z).squeeze(-1) + self.peer_logit_proj(peer_stack).squeeze(-1)
            seg_logits.append(logits)
        return seg_logits, z, peer_stack

    def forward(self, clues: torch.Tensor, n_segments: int = 5, n_loops: int = 2) -> List[torch.Tensor]:
        seg_logits, _, _ = self.forward_with_state(clues, n_segments=n_segments, n_loops=n_loops)
        return seg_logits

    def forward_amp(self, clues: torch.Tensor, n_segments: int = 5, n_loops: int = 2) -> torch.Tensor:
        with torch.no_grad():
            seg_logits, z, peer_stack = self.forward_with_state(clues, n_segments=n_segments, n_loops=n_loops)
            base_logits = seg_logits[-1].detach()
            base_probs = F.softmax(base_logits, dim=-1).unsqueeze(-1)
            z_det = z.detach()
            peer_det = peer_stack.detach()
        amp_in = torch.cat([z_det, peer_det, base_probs], dim=-1)
        delta_logits = self.amp_adapter(amp_in).squeeze(-1)
        return self.amp_scale.clamp(min=0.2, max=3.0) * base_logits + delta_logits


def birkhoff_nqueens_anyorder_loss(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    _, n, _ = logits.shape
    p = F.softmax(logits, dim=-1)
    col_sum = p.sum(dim=1)
    col_birkhoff = ((col_sum - 1.0) ** 2).sum(dim=-1).mean()
    row_ent = (1.0 - (p ** 2).sum(dim=-1)).sum(dim=-1).mean()
    diag_loss = torch.tensor(0.0, device=logits.device)
    for dr in range(1, n):
        p_top = p[:, :-dr, :]
        p_bot = p[:, dr:, :]
        diag_loss = diag_loss + (p_top[:, :, :-dr] * p_bot[:, :, dr:]).sum(dim=(1, 2)).mean()
        diag_loss = diag_loss + (p_top[:, :, dr:] * p_bot[:, :, :-dr]).sum(dim=(1, 2)).mean()
    p_att = nqueens_peer_attack_mask(clues)
    bad_p = (p * p_att).sum(dim=-1).clamp(max=1.0 - 1e-6)
    peer_loss = -torch.log(1.0 - bad_p).mean()
    logit_margin = (F.relu(logits + 5.0) * p_att).mean()
    clue_mask = clues >= 0
    if clue_mask.any():
        clue_ce = F.cross_entropy(logits[clue_mask], clues[clue_mask].clamp(min=0))
    else:
        clue_ce = torch.tensor(0.0, device=logits.device)
    return col_birkhoff + 0.5 * row_ent + 2.0 * diag_loss + 2.5 * peer_loss + 0.5 * logit_margin + clue_ce


def extract_nqueens_branch_features(
    leaf_probs: torch.Tensor,
    base_probs: torch.Tensor,
    x_branch: torch.Tensor,
    log_prior: torch.Tensor,
    rollout_depth: int,
) -> torch.Tensor:
    nb, n_branches, n, _ = leaf_probs.shape
    col_sum = leaf_probs.sum(dim=2)
    col_energy = ((col_sum - 1.0) ** 2).sum(dim=-1)

    diag_energy = torch.zeros(nb, n_branches, device=leaf_probs.device)
    for dr in range(1, n):
        p_top = leaf_probs[:, :, :-dr, :]
        p_bot = leaf_probs[:, :, dr:, :]
        diag_energy = diag_energy + (p_top[:, :, :, :-dr] * p_bot[:, :, :, dr:]).sum(dim=(2, 3))
        diag_energy = diag_energy + (p_top[:, :, :, dr:] * p_bot[:, :, :, :-dr]).sum(dim=(2, 3))

    uncond_mask = (x_branch < 0).float()
    uncond_denom = uncond_mask.sum(dim=-1).clamp(min=1.0)
    p_cl = leaf_probs.clamp(min=1e-6)
    row_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
    mean_ent = (row_ent * uncond_mask).sum(dim=-1) / uncond_denom

    max_p = leaf_probs.max(dim=-1).values.clamp(min=1e-6)
    deplete = ((-torch.log(max_p) * uncond_mask).sum(dim=-1) / uncond_denom) * 4.0
    m_ql = col_energy + 2.0 * diag_energy + deplete

    base_exp = base_probs.unsqueeze(1).expand_as(leaf_probs).clamp(min=1e-6)
    kl_row = (p_cl * (torch.log(p_cl) - torch.log(base_exp))).sum(dim=-1)
    path_kl = (kl_row * uncond_mask).sum(dim=-1) / uncond_denom
    surprise = (-log_prior / float(max(1, rollout_depth))) + path_kl

    mean_branch_p = leaf_probs.mean(dim=1, keepdim=True).expand_as(leaf_probs).clamp(min=1e-6)
    kl_consensus = (p_cl * (torch.log(p_cl) - torch.log(mean_branch_p))).sum(dim=-1)
    disagreement = (kl_consensus * uncond_mask).sum(dim=-1) / uncond_denom

    depth_feat = ((x_branch >= 0).float().mean(dim=-1)) * float(rollout_depth)
    return torch.stack([depth_feat, mean_ent, m_ql, surprise, disagreement], dim=-1).detach()


@torch.no_grad()
def expand_nqueens_search_branches(
    model: AnyOrderNQueensNet,
    x_cond: torch.Tensor,
    n_branches: int = 16,
    rollout_depth: int = 2,
    n_segments: int = 4,
    use_amp_proposal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    nb, n = x_cond.shape
    device = x_cond.device
    base_logits = model(x_cond, n_segments=n_segments, n_loops=2)[-1]
    base_probs = F.softmax(base_logits, dim=-1)

    if use_amp_proposal:
        prop_logits = model.forward_amp(x_cond, n_segments=n_segments, n_loops=2)
        amp_probs = F.softmax(prop_logits, dim=-1)
        prop_probs = 0.5 * amp_probs + 0.5 * base_probs
        anchor_logits = torch.log(prop_probs.clamp(min=1e-6))
    else:
        prop_probs = base_probs
        anchor_logits = torch.log(base_probs.clamp(min=1e-6))

    r_bits = min({4: 2, 16: 4, 32: 5}.get(n_branches, 4), n)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)],
        dtype=torch.long,
        device=device,
    )
    p_cl = prop_probs.clamp(min=1e-7)
    row_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
    top2_vals, top2_idx = prop_probs.topk(2, dim=-1)
    is_empty = x_cond < 0
    score = row_ent + torch.where(is_empty, torch.zeros_like(row_ent), torch.full_like(row_ent, 1e6))
    root_rows = score.topk(r_bits, dim=-1, largest=False).indices

    root_top2_cols = top2_idx.gather(1, root_rows.unsqueeze(-1).expand(nb, r_bits, 2))
    root_top2_probs = top2_vals.gather(1, root_rows.unsqueeze(-1).expand(nb, r_bits, 2)).clamp(min=1e-5)

    x_branch = x_cond.unsqueeze(1).expand(nb, n_branches, n).clone()
    log_prior = torch.zeros(nb, n_branches, device=device)

    for bit_i in range(r_bits):
        choice_bit = bit_table[:, bit_i].unsqueeze(0).expand(nb, n_branches)
        row_i = root_rows[:, bit_i].unsqueeze(1).expand(nb, n_branches)
        cand_c = root_top2_cols[:, bit_i, :].gather(1, choice_bit)
        cand_p = root_top2_probs[:, bit_i, :].gather(1, choice_bit)
        row_was_empty = is_empty.gather(1, root_rows[:, bit_i : bit_i + 1]).expand(nb, n_branches)
        cur_val = x_branch.gather(2, row_i.unsqueeze(-1)).squeeze(-1)
        new_val = torch.where(row_was_empty, cand_c, cur_val)
        x_branch.scatter_(2, row_i.unsqueeze(-1), new_val.unsqueeze(-1))
        log_prior = log_prior + torch.where(row_was_empty, torch.log(cand_p), torch.zeros_like(cand_p))

    x_after_root = x_branch.clone()
    for _ in range(rollout_depth - 1):
        flat_x = x_branch.reshape(nb * n_branches, n)
        if use_amp_proposal:
            step_logits = model.forward_amp(flat_x, n_segments=max(3, n_segments - 1), n_loops=2)
        else:
            step_logits = model(flat_x, n_segments=max(3, n_segments - 1), n_loops=2)[-1]
        step_probs = F.softmax(step_logits, dim=-1).view(nb, n_branches, n, n)
        sp_cl = step_probs.clamp(min=1e-7)
        sp_ent = -(sp_cl * torch.log(sp_cl)).sum(dim=-1)
        rem_empty = x_branch < 0
        sp_score = sp_ent + torch.where(rem_empty, torch.zeros_like(sp_ent), torch.full_like(sp_ent, 1e6))
        next_row = sp_score.argmin(dim=-1, keepdim=True)
        best_col = step_probs.argmax(dim=-1).gather(2, next_row)
        was_empty = rem_empty.gather(2, next_row)
        cur_val = x_branch.gather(2, next_row)
        upd_val = torch.where(was_empty, best_col, cur_val)
        x_branch.scatter_(2, next_row, upd_val)

    leaf_logits = model(x_branch.reshape(nb * n_branches, n), n_segments=n_segments, n_loops=2)[-1].view(
        nb, n_branches, n, n
    )
    leaf_probs = F.softmax(leaf_logits, dim=-1)

    root_added_mask = ((x_after_root >= 0) & (x_cond.unsqueeze(1) < 0)).unsqueeze(-1).float()
    roll_added_mask = ((x_branch >= 0) & (x_after_root < 0)).unsqueeze(-1).float()
    added_oh = F.one_hot(x_branch.clamp(0, n - 1), num_classes=n).float()
    branch_probs = (
        root_added_mask * (0.30 * added_oh + 0.70 * leaf_probs)
        + roll_added_mask * (0.15 * added_oh + 0.85 * leaf_probs)
        + (1.0 - root_added_mask - roll_added_mask) * leaf_probs
    ).clamp(min=1e-6)
    branch_probs = branch_probs / branch_probs.sum(dim=-1, keepdim=True)
    branch_logits = torch.log(branch_probs)
    branch_feats = extract_nqueens_branch_features(
        branch_probs, base_probs, x_branch, log_prior, rollout_depth=rollout_depth
    )
    return branch_logits.detach(), anchor_logits.detach(), branch_feats.detach(), x_branch.detach()


@torch.no_grad()
def predict_nqueens_condition(
    condition: str,
    model: Optional[AnyOrderNQueensNet],
    trust_head: Optional[ConfiabilityTrustHead],
    x_cond: torch.Tensor,
    n_branches: int = 16,
    n_segments: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    nb, n = x_cond.shape
    device = x_cond.device
    eps = 1e-4

    if condition == "C1_Control_PeerClue_Uniform":
        p_att = nqueens_peer_attack_mask(x_cond)
        allowed = (1.0 - p_att).clamp(min=0.0)
        s_allow = allowed.sum(dim=-1, keepdim=True)
        unif = torch.full_like(allowed, 1.0 / float(n))
        p = torch.where(s_allow > 0.5, allowed / s_allow.clamp(min=1.0), unif)
        p = (1.0 - eps) * p + (eps / float(n))
        return p / p.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    assert model is not None
    model.eval()

    if condition == "M1_ParamsReal_Base_B0":
        base_logits = model(x_cond, n_segments=n_segments, n_loops=2)[-1]
        base_probs = F.softmax(base_logits, dim=-1)
        out = (1.0 - eps) * base_probs + (eps / float(n))
        return out / out.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    if condition == "M5_ParamsAmp_Distilled_B0":
        amp_logits = model.forward_amp(x_cond, n_segments=n_segments, n_loops=2)
        amp_probs = F.softmax(amp_logits, dim=-1)
        out = (1.0 - eps) * amp_probs + (eps / float(n))
        return out / out.sum(dim=-1, keepdim=True), {"effective_branch_ess": 1.0, "mean_beta_trust": 1.0}

    use_amp_prop = condition == "M6_ParamsAmp_Plus_Search_B16"
    branch_logits, base_logits, branch_feats, _ = expand_nqueens_search_branches(
        model, x_cond, n_branches=n_branches, rollout_depth=2, n_segments=n_segments, use_amp_proposal=use_amp_prop
    )

    if condition == "M2_Search_Unweighted_Clear_B16":
        branch_probs = F.softmax(branch_logits, dim=-1)
        base_probs = F.softmax(base_logits, dim=-1)
        weights = torch.full((nb, n_branches), 1.0 / float(n_branches), device=device)
        cleared = (weights.unsqueeze(-1).unsqueeze(-1) * branch_probs).sum(dim=1)
        out = 0.85 * cleared + 0.15 * base_probs
        out = (1.0 - eps) * out + (eps / float(n))
        return out / out.sum(dim=-1, keepdim=True), {
            "effective_branch_ess": float(n_branches),
            "mean_beta_trust": 1.0,
        }

    assert trust_head is not None
    trust_head.eval()
    cleared_probs, weights, beta_trust = trust_head(branch_feats, branch_logits, base_logits)
    out = (1.0 - eps) * cleared_probs + (eps / float(n))
    out = out / out.sum(dim=-1, keepdim=True)
    ess = float((1.0 / (weights * weights).sum(dim=-1).clamp(min=1e-6)).mean().item())
    return out, {"effective_branch_ess": ess, "mean_beta_trust": float(beta_trust.mean().item())}


@torch.no_grad()
def evaluate_nqueens_condition(
    condition: str,
    model: Optional[AnyOrderNQueensNet],
    trust_head: Optional[ConfiabilityTrustHead],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    multi_ys: List[np.ndarray],
    device: torch.device,
    n_branches: int = 16,
    k_random_orders: int = 4,
    seed: int = 101,
) -> Dict[str, Any]:
    nb, n = x_eval_np.shape
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)

    def get_probs(x_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        return predict_nqueens_condition(condition, model, trust_head, x_cur, n_branches=n_branches, n_segments=4)

    p0_t, step0_stats = get_probs(x_init)
    p_step0 = p0_t.cpu().numpy()
    best_y_np = y_eval_np.copy()
    for i in range(nb):
        cands = multi_ys[i]
        if len(cands) > 1:
            emp = x_eval_np[i] < 0
            best_score = -1e9
            for c_sol in cands:
                sc = sum(math.log(max(float(p_step0[i, r, c_sol[r]]), 1e-6)) for r in range(n) if emp[r])
                if sc > best_score:
                    best_score = sc
                    best_y_np[i] = c_sol
    y_true = torch.tensor(best_y_np, dtype=torch.long, device=device)

    emp_mask = x_eval_np < 0
    ece_val = compute_ece(p_step0[emp_mask], best_y_np[emp_mask].clip(0, n - 1), n_bins=10)

    max_empty = int((x_eval_np < 0).sum(axis=1).max())
    board_order_total_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    board_order_cell_nll = np.zeros((nb, k_random_orders), dtype=np.float64)

    for k in range(k_random_orders):
        step_rows = torch.full((nb, max_empty), -1, dtype=torch.long, device=device)
        empty_counts = np.zeros(nb, dtype=np.float64)
        for i in range(nb):
            rng_i = np.random.RandomState(seed * 10000 + i * 53 + k)
            emp_r = np.where(x_eval_np[i] < 0)[0]
            empty_counts[i] = max(len(emp_r), 1)
            perm_r = emp_r[rng_i.permutation(len(emp_r))]
            for s_i, r_val in enumerate(perm_r):
                step_rows[i, s_i] = int(r_val)

        x_cur = x_init.clone()
        total_nll_k = torch.zeros(nb, dtype=torch.float64, device=device)
        for s_i in range(max_empty):
            r_idx = step_rows[:, s_i]
            valid = r_idx >= 0
            if not valid.any():
                continue
            probs_step, _ = get_probs(x_cur)
            r_safe = r_idx.clamp(min=0)
            row_p = probs_step.gather(1, r_safe.view(nb, 1, 1).expand(nb, 1, n)).squeeze(1)
            true_c = y_true.gather(1, r_safe.unsqueeze(-1)).squeeze(-1).clamp(0, n - 1)
            true_p = row_p.gather(1, true_c.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            total_nll_k = total_nll_k + (-torch.log(true_p).to(torch.float64) * valid.to(torch.float64))
            cur_v = x_cur.gather(1, r_safe.unsqueeze(-1)).squeeze(-1)
            upd_v = torch.where(valid, true_c, cur_v)
            x_cur.scatter_(1, r_safe.unsqueeze(-1), upd_v.unsqueeze(-1))

        tot_np = total_nll_k.cpu().numpy()
        board_order_total_nll[:, k] = tot_np
        board_order_cell_nll[:, k] = tot_np / empty_counts

    rand_order_nll_per_cell = float(board_order_cell_nll.mean())
    rand_order_ppl = float(math.exp(min(rand_order_nll_per_cell, 20.0)))
    cross_order_nll_std = float(np.std(board_order_total_nll, axis=1, ddof=1).mean()) if k_random_orders > 1 else 0.0

    # Adaptive VoI / Minimum-Entropy Order
    x_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_counts = torch.zeros(nb, dtype=torch.float64, device=device)

    for _ in range(max_empty):
        rem = x_minh < 0
        if not rem.any():
            break
        probs_step, _ = get_probs(x_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        ent_masked = ent + torch.where(rem, torch.zeros_like(ent), torch.full_like(ent, 1e6))
        pick_r = ent_masked.argmin(dim=-1)
        has_rem = rem.any(dim=-1)
        row_p = probs_step.gather(1, pick_r.view(nb, 1, 1).expand(nb, 1, n)).squeeze(1)
        true_c = y_true.gather(1, pick_r.unsqueeze(-1)).squeeze(-1).clamp(0, n - 1)
        true_p = row_p.gather(1, true_c.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        minh_total_nll = minh_total_nll + (-torch.log(true_p).to(torch.float64) * has_rem.to(torch.float64))
        minh_counts = minh_counts + has_rem.to(torch.float64)
        cur_v = x_minh.gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        upd_v = torch.where(has_rem, true_c, cur_v)
        x_minh.scatter_(1, pick_r.unsqueeze(-1), upd_v.unsqueeze(-1))

    min_h_voi_nll_per_cell = float((minh_total_nll / minh_counts.clamp(min=1.0)).mean().item())
    min_h_voi_ppl = float(math.exp(min(min_h_voi_nll_per_cell, 20.0)))

    # Autonomous Sequential Rollout
    x_auto = x_init.clone()
    for _ in range(max_empty):
        rem = x_auto < 0
        if not rem.any():
            break
        probs_auto, _ = get_probs(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        ent_masked = ent + torch.where(rem, torch.zeros_like(ent), torch.full_like(ent, 1e6))
        pick_r = ent_masked.argmin(dim=-1)
        has_rem = rem.any(dim=-1)
        pred_cols = probs_auto.argmax(dim=-1).gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        cur_v = x_auto.gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        upd_v = torch.where(has_rem, pred_cols, cur_v)
        x_auto.scatter_(1, pick_r.unsqueeze(-1), upd_v.unsqueeze(-1))

    auto_np = x_auto.cpu().numpy()
    auto_metrics = count_nqueens_metrics(auto_np, x_eval_np)
    autonomous_empty_acc = float(((auto_np == best_y_np) & emp_mask).sum() / max(emp_mask.sum(), 1))

    return {
        "n_branches": n_branches,
        "rand_order_nll_per_cell": rand_order_nll_per_cell,
        "rand_order_ppl": rand_order_ppl,
        "cross_order_nll_std": cross_order_nll_std,
        "min_h_voi_nll_per_cell": min_h_voi_nll_per_cell,
        "min_h_voi_ppl": min_h_voi_ppl,
        "autonomous_empty_acc": autonomous_empty_acc,
        "autonomous_valid_board_rate": auto_metrics["valid_board_rate"],
        "autonomous_collisions": auto_metrics["mean_collisions"],
        "ece_calibration_error": ece_val,
        "search_diagnostics": step0_stats,
    }


# ---------------------------------------------------------------------------
# 3. Training & Main Track E Exploratory Runner
# ---------------------------------------------------------------------------

def run_explore_trust_and_amp_split(args) -> Dict[str, Any]:
    train_csv_path, data_hashes = ensure_and_verify_sudoku_extreme_train_only(args.sudoku_dir)
    if args.smoke_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [101] if args.smoke_cpu else [101, 102, 103]

    # Strictly development splits from train.csv only (zero touch of confirmatory test.csv)
    x_tr_all, y_tr_all = read_sudoku_csv(train_csv_path, 5700, seed=101, random_subset=False)
    x_tr_full = x_tr_all[0:3600].reshape(-1, 81).astype(np.int64)
    y_tr_full = y_tr_all[0:3600].reshape(-1, 81).astype(np.int64)
    # Held-in real calibration split for training theta_trust ("from search into reality")
    x_cal_full = x_tr_all[3600:4000].reshape(-1, 81).astype(np.int64)
    y_cal_full = y_tr_all[3600:4000].reshape(-1, 81).astype(np.int64)
    x_dev_full = x_tr_all[5500:5628].reshape(-1, 81).astype(np.int64)
    y_dev_full = y_tr_all[5500:5628].reshape(-1, 81).astype(np.int64)

    x_tr_smoke = x_tr_all[4000:4200].reshape(-1, 81).astype(np.int64)
    y_tr_smoke = y_tr_all[4000:4200].reshape(-1, 81).astype(np.int64)
    x_cal_smoke = x_tr_all[4200:4256].reshape(-1, 81).astype(np.int64)
    y_cal_smoke = y_tr_all[4200:4256].reshape(-1, 81).astype(np.int64)
    x_dev_smoke = x_tr_all[5628:5644].reshape(-1, 81).astype(np.int64)
    y_dev_smoke = y_tr_all[5628:5644].reshape(-1, 81).astype(np.int64)

    x_tr_np = x_tr_smoke if args.smoke_cpu else x_tr_full
    y_tr_np = y_tr_smoke if args.smoke_cpu else y_tr_full
    x_cal_np = x_cal_smoke if args.smoke_cpu else x_cal_full
    y_cal_np = y_cal_smoke if args.smoke_cpu else y_cal_full
    x_dev_np = x_dev_smoke if args.smoke_cpu else x_dev_full
    y_dev_np = y_dev_smoke if args.smoke_cpu else y_dev_full

    # N-Queens 10x10 and 8x8 development splits
    sols10 = solve_all_nqueens(10, max_sols=2000)
    orbits10: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for s in sols10:
        orbits10.setdefault(d4_orbit_rep(s), []).append(s)
    orbit_keys10 = sorted(orbits10.keys())
    nq10_train_sols = [s for k in orbit_keys10[:48] for s in orbits10[k]]
    nq10_dev_sols = [s for k in orbit_keys10[48:64] for s in orbits10[k]]

    nq10_tr_x, nq10_tr_y, _, tr10_orbs = sample_nqueens_dev_split(
        nq10_train_sols, sols10, 10, 160, set(), 2810
    )
    nq10_dev_x, nq10_dev_y, nq10_dev_multi, _ = sample_nqueens_dev_split(
        nq10_dev_sols, sols10, 10, 12 if args.smoke_cpu else 48, tr10_orbs, 2811
    )
    sols8 = solve_all_nqueens(8, max_sols=400)
    nq8_dev_x, nq8_dev_y, nq8_dev_multi, _ = sample_nqueens_dev_split(
        sols8, sols8, 8, 12 if args.smoke_cpu else 48, set(), 2808
    )

    sudoku_base_steps = 25 if args.smoke_cpu else args.sudoku_steps
    trust_steps = 15 if args.smoke_cpu else args.trust_steps
    amp_steps = 20 if args.smoke_cpu else args.amp_steps
    nq_base_steps = 20 if args.smoke_cpu else args.nq_steps
    hidden_size = 64 if args.smoke_cpu else 96
    k_orders = 2 if args.smoke_cpu else 4
    n_blocks = 4 if args.smoke_cpu else 6
    unit_idx = build_sudoku_unit_indices(device)

    per_seed: Dict[str, Any] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # -------------------------------------------------------------------
        # PART A: SUDOKU-EXTREME (params_real -> theta_trust -> params_amp)
        # -------------------------------------------------------------------
        sudoku_arms: Dict[str, Any] = {}

        # C0: Random Network Control through B=16 search pipeline (Rule 9.1)
        torch.manual_seed(seed + 9000)
        rand_sudoku_net = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        rand_trust_head = ConfiabilityTrustHead().to(device)
        sudoku_arms["C0_Control_Random_Net"] = evaluate_sudoku_condition(
            "M4_Search_RealityCalibrated_Trust_B16",
            rand_sudoku_net,
            rand_trust_head,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=16,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        # C1: PeerClue Uniform Control
        sudoku_arms["C1_Control_PeerClue_Uniform"] = evaluate_sudoku_condition(
            "C1_Control_PeerClue_Uniform",
            None,
            None,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        # Phase 1: Train theta_base in params_real strictly on real data x* ~ D_real
        torch.manual_seed(seed)
        sudoku_model = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        base_opt = torch.optim.AdamW(sudoku_model.base_parameters(), lr=4e-3, weight_decay=1e-4)
        sudoku_model.train()

        for _ in range(sudoku_base_steps):
            b_idx = np.random.choice(len(x_tr_np), size=32, replace=True)
            xb_clues = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)
            yb_sol = torch.tensor(y_tr_np[b_idx], dtype=torch.long, device=device)

            reveal_prob = torch.rand((32, 1), device=device) * 0.85
            reveal_mask = (torch.rand((32, 81), device=device) < reveal_prob) & (xb_clues == 0)
            xb_st = torch.where(reveal_mask, yb_sol, xb_clues)
            target_mask = xb_st == 0
            if not target_mask.any():
                target_mask = xb_clues == 0

            step_logits = sudoku_model(xb_st, n_steps=5)
            loss = torch.tensor(0.0, device=device)
            for i, lg in enumerate(step_logits):
                w_i = (i + 1.0) / 15.0
                ce_uncond = F.cross_entropy(lg[target_mask], (yb_sol[target_mask] - 1).clamp(0, 8))
                birk_reg = sudoku_anyorder_birkhoff_contradiction_loss(lg, xb_st, unit_idx)
                loss = loss + w_i * (ce_uncond + 0.25 * birk_reg)

            base_opt.zero_grad()
            loss.backward()
            base_opt.step()

        # Freeze theta_base before training theta_trust or theta_amp!
        for p in sudoku_model.base_parameters():
            p.requires_grad_(False)
        base_checksum_before = float(sum(p.detach().float().abs().sum().item() for p in sudoku_model.base_parameters()))

        # Precompute detached search rollouts from frozen theta_base on calibration split x_cal_np
        n_cal_batches = 3 if args.smoke_cpu else 16
        cal_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(n_cal_batches):
            b_idx = np.random.choice(len(x_cal_np), size=24, replace=True)
            xb_clues = torch.tensor(x_cal_np[b_idx], dtype=torch.long, device=device)
            yb_sol = torch.tensor(y_cal_np[b_idx], dtype=torch.long, device=device)
            rev_prob = torch.rand((24, 1), device=device) * 0.65
            rev_mask = (torch.rand((24, 81), device=device) < rev_prob) & (xb_clues == 0)
            xb_st = torch.where(rev_mask, yb_sol, xb_clues)
            t_mask = xb_st == 0
            if not t_mask.any():
                t_mask = xb_clues == 0
            b_lg, base_lg, b_ft, _ = expand_sudoku_search_branches(
                sudoku_model, xb_st, unit_idx, n_branches=16, rollout_depth=2, n_steps=4
            )
            cal_cache.append((b_ft, b_lg, base_lg, yb_sol, t_mask))

        # Phase 2A (Ablation M3): Self-Gamed Trust Head (put inside params_search instead of params_real!)
        # Trained to minimize internal entropy & inconsistency rather than real data x*, producing overconfidence.
        torch.manual_seed(seed + 300)
        selfgamed_trust = ConfiabilityTrustHead().to(device)
        sg_opt = torch.optim.AdamW(selfgamed_trust.parameters(), lr=1.5e-2, weight_decay=1e-4)
        for step_i in range(trust_steps):
            b_ft, b_lg, base_lg, _, t_mask = cal_cache[step_i % len(cal_cache)]
            sg_probs, sg_w, _ = selfgamed_trust(b_ft, b_lg, base_lg)
            uncond = t_mask.float()
            p_cl = sg_probs.clamp(min=1e-6)
            internal_ent = (-(p_cl * torch.log(p_cl)).sum(dim=-1) * uncond).sum() / uncond.sum().clamp(min=1.0)
            # Self-gamed objective rewards low internal branch entropy (feat 1) and forces sharpening (low tau)
            ent_branch_reward = (sg_w * b_ft[:, :, 1]).mean()
            sg_loss = internal_ent + 0.5 * ent_branch_reward + 0.35 * F.softplus(selfgamed_trust.log_tau)
            sg_opt.zero_grad()
            sg_loss.backward()
            sg_opt.step()

        # Phase 2B (Method M4): Reality-Calibrated Confiability Head theta_trust in params_real
        # Trained strictly from search into reality x* ~ D_real via Proper Scoring Rule (NLL + Brier)
        torch.manual_seed(seed + 400)
        real_trust = ConfiabilityTrustHead().to(device)
        rt_opt = torch.optim.AdamW(real_trust.parameters(), lr=8e-3, weight_decay=1e-4)
        for step_i in range(trust_steps):
            b_ft, b_lg, base_lg, yb_sol, t_mask = cal_cache[step_i % len(cal_cache)]
            cleared_probs, _, _ = real_trust(b_ft, b_lg, base_lg)
            true_d = (yb_sol[t_mask] - 1).clamp(0, 8)
            pred_p = cleared_probs[t_mask].clamp(min=1e-6)
            nll_loss = -torch.log(pred_p.gather(1, true_d.unsqueeze(-1)).squeeze(-1)).mean()
            true_oh = F.one_hot(true_d, num_classes=9).float()
            brier_loss = ((pred_p - true_oh) ** 2).sum(dim=-1).mean()
            rt_loss = nll_loss + 0.5 * brier_loss
            rt_opt.zero_grad()
            rt_loss.backward()
            rt_opt.step()

        for p in real_trust.parameters():
            p.requires_grad_(False)

        # Phase 3 (Method M5 & M6): Train params_amp strictly via KL distillation from stopgrad[mu_B^clear]
        # on unlabeled puzzle contexts x_S, with zero contamination of params_real!
        n_amp_batches = 3 if args.smoke_cpu else 20
        amp_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for _ in range(n_amp_batches):
                b_idx = np.random.choice(len(x_tr_np), size=24, replace=True)
                xb_clues = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)
                p_init = F.softmax(sudoku_model(xb_clues, n_steps=4)[-1], dim=-1)
                ent_init = -(p_init.clamp(min=1e-7) * torch.log(p_init.clamp(min=1e-7))).sum(dim=-1)
                is_emp = xb_clues == 0
                k_auto = int(np.random.randint(0, 14))
                xb_unlabeled = xb_clues.clone()
                if k_auto > 0:
                    sc = ent_init + torch.where(is_emp, torch.zeros_like(ent_init), torch.full_like(ent_init, 1e6))
                    pick = sc.topk(k_auto, dim=-1, largest=False).indices
                    pred_d = p_init.argmax(dim=-1) + 1
                    was_e = is_emp.gather(1, pick)
                    cur_v = xb_unlabeled.gather(1, pick)
                    xb_unlabeled.scatter_(1, pick, torch.where(was_e, pred_d.gather(1, pick), cur_v))

                b_lg, base_lg, b_ft, _ = expand_sudoku_search_branches(
                    sudoku_model, xb_unlabeled, unit_idx, n_branches=16, rollout_depth=2, n_steps=4
                )
                target_mu_clear, _, _ = real_trust(b_ft, b_lg, base_lg)
                amp_cache.append((xb_unlabeled.detach(), target_mu_clear.detach()))

        for p in sudoku_model.amp_parameters():
            p.requires_grad_(True)
        amp_opt = torch.optim.AdamW(sudoku_model.amp_parameters(), lr=4e-3, weight_decay=1e-4)
        for step_i in range(amp_steps):
            xb_unlabeled, target_mu_clear = amp_cache[step_i % len(amp_cache)]
            amp_logits = sudoku_model.forward_amp(xb_unlabeled, n_steps=4)
            log_q_amp = F.log_softmax(amp_logits, dim=-1)
            uncond_mask = (xb_unlabeled == 0).unsqueeze(-1).float()
            kl_distill = (
                target_mu_clear * (torch.log(target_mu_clear.clamp(min=1e-6)) - log_q_amp) * uncond_mask
            ).sum(dim=-1).mean()
            birk_amp = sudoku_anyorder_birkhoff_contradiction_loss(amp_logits, xb_unlabeled, unit_idx)
            amp_loss = kl_distill + 0.15 * birk_amp
            amp_opt.zero_grad()
            amp_loss.backward()
            amp_opt.step()

        base_checksum_after = float(sum(p.detach().float().abs().sum().item() for p in sudoku_model.base_parameters()))
        assert abs(base_checksum_before - base_checksum_after) < 1e-5, "params_real was contaminated during amp training!"

        # Evaluate M1..M6 on Sudoku-Extreme Dev Split
        for cond_name, t_head, b_val in [
            ("M1_ParamsReal_Base_B0", None, 0),
            ("M2_Search_Unweighted_Clear_B16", None, 16),
            ("M3_Search_SelfGamed_Trust_B16", selfgamed_trust, 16),
            ("M4_Search_RealityCalibrated_Trust_B16", real_trust, 16),
            ("M5_ParamsAmp_Distilled_B0", real_trust, 0),
            ("M6_ParamsAmp_Plus_Search_B16", real_trust, 16),
        ]:
            sudoku_arms[cond_name] = evaluate_sudoku_condition(
                cond_name,
                sudoku_model,
                t_head,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=b_val,
                k_random_orders=k_orders,
                n_reveal_blocks=n_blocks,
                seed=seed,
            )

        # Also evaluate B in {4, 16, 32} scaling for M4 (Reality-Calibrated Trust)
        sudoku_b_scaling: Dict[str, Any] = {}
        for b_scale in (4, 16, 32):
            sudoku_b_scaling[f"B{b_scale}"] = evaluate_sudoku_condition(
                "M4_Search_RealityCalibrated_Trust_B16",
                sudoku_model,
                real_trust,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=b_scale,
                k_random_orders=k_orders,
                n_reveal_blocks=n_blocks,
                seed=seed,
            )

        # -------------------------------------------------------------------
        # PART B: N-QUEENS (10x10 ID Dev & 8x8 OOD Dev)
        # -------------------------------------------------------------------
        torch.manual_seed(seed + 9000)
        nq_rand_net = AnyOrderNQueensNet(d_model=48).to(device)
        nq_rand_trust = ConfiabilityTrustHead().to(device)

        torch.manual_seed(seed)
        nq_model = AnyOrderNQueensNet(d_model=48).to(device)
        nq_base_opt = torch.optim.AdamW(nq_model.base_parameters(), lr=3e-3, weight_decay=1e-4)
        nq_model.train()
        for _ in range(nq_base_steps):
            idx = np.random.choice(len(nq10_tr_x), size=32, replace=True)
            xb = torch.tensor(nq10_tr_x[idx], dtype=torch.long, device=device)
            yb = torch.tensor(nq10_tr_y[idx], dtype=torch.long, device=device)
            rev_prob = torch.rand((32, 1), device=device) * 0.75
            rev_mask = (torch.rand((32, 10), device=device) < rev_prob) & (xb < 0)
            xb_st = torch.where(rev_mask, yb, xb)
            t_mask = xb_st < 0
            if not t_mask.any():
                t_mask = xb < 0
            seg_logits = nq_model(xb_st, n_segments=5, n_loops=2)
            loss = sum(
                F.cross_entropy(lg[t_mask], yb[t_mask]) + 0.25 * birkhoff_nqueens_anyorder_loss(lg, xb_st)
                for lg in seg_logits
            ) / len(seg_logits)
            nq_base_opt.zero_grad()
            loss.backward()
            nq_base_opt.step()

        for p in nq_model.base_parameters():
            p.requires_grad_(False)

        # Precompute detached N-Queens search rollouts for trust & amp training
        n_nq_cal_batches = 3 if args.smoke_cpu else 14
        nq_cal_cache: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for _ in range(n_nq_cal_batches):
            idx = np.random.choice(len(nq10_tr_x), size=24, replace=True)
            xb = torch.tensor(nq10_tr_x[idx], dtype=torch.long, device=device)
            yb = torch.tensor(nq10_tr_y[idx], dtype=torch.long, device=device)
            b_logits, base_lg, b_feats, _ = expand_nqueens_search_branches(
                nq_model, xb, n_branches=16, rollout_depth=2, n_segments=4
            )
            nq_cal_cache.append((b_feats, b_logits, base_lg, xb, yb))

        # Train Self-Gamed Trust vs Reality-Calibrated Trust on N-Queens
        torch.manual_seed(seed + 300)
        nq_sg_trust = ConfiabilityTrustHead().to(device)
        nq_sg_opt = torch.optim.AdamW(nq_sg_trust.parameters(), lr=1.5e-2, weight_decay=1e-4)
        for step_i in range(trust_steps):
            b_feats, b_logits, base_lg, xb, _ = nq_cal_cache[step_i % len(nq_cal_cache)]
            sg_p, sg_w, _ = nq_sg_trust(b_feats, b_logits, base_lg)
            uncond = (xb < 0).float()
            p_cl = sg_p.clamp(min=1e-6)
            ent_loss = (-(p_cl * torch.log(p_cl)).sum(dim=-1) * uncond).sum() / uncond.sum().clamp(min=1.0)
            loss_sg = ent_loss + 0.5 * (sg_w * b_feats[:, :, 1]).mean() + 0.35 * F.softplus(nq_sg_trust.log_tau)
            nq_sg_opt.zero_grad()
            loss_sg.backward()
            nq_sg_opt.step()

        torch.manual_seed(seed + 400)
        nq_real_trust = ConfiabilityTrustHead().to(device)
        nq_rt_opt = torch.optim.AdamW(nq_real_trust.parameters(), lr=8e-3, weight_decay=1e-4)
        for step_i in range(trust_steps):
            b_feats, b_logits, base_lg, xb, yb = nq_cal_cache[step_i % len(nq_cal_cache)]
            rt_p, _, _ = nq_real_trust(b_feats, b_logits, base_lg)
            t_mask = xb < 0
            true_c = yb[t_mask].clamp(0, 9)
            pred_p = rt_p[t_mask].clamp(min=1e-6)
            nll_l = -torch.log(pred_p.gather(1, true_c.unsqueeze(-1)).squeeze(-1)).mean()
            true_oh = F.one_hot(true_c, num_classes=10).float()
            brier_l = ((pred_p - true_oh) ** 2).sum(dim=-1).mean()
            loss_rt = nll_l + 0.5 * brier_l
            nq_rt_opt.zero_grad()
            loss_rt.backward()
            nq_rt_opt.step()

        for p in nq_real_trust.parameters():
            p.requires_grad_(False)

        # Train N-Queens params_amp via KL distillation from stopgrad[mu_B^clear]
        nq_amp_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            for b_feats, b_logits, base_lg, xb, _ in nq_cal_cache:
                target_mu, _, _ = nq_real_trust(b_feats, b_logits, base_lg)
                nq_amp_cache.append((xb.detach(), target_mu.detach()))

        for p in nq_model.amp_parameters():
            p.requires_grad_(True)
        nq_amp_opt = torch.optim.AdamW(nq_model.amp_parameters(), lr=4e-3, weight_decay=1e-4)
        for step_i in range(amp_steps):
            xb, target_mu = nq_amp_cache[step_i % len(nq_amp_cache)]
            amp_lg = nq_model.forward_amp(xb, n_segments=4, n_loops=2)
            log_q = F.log_softmax(amp_lg, dim=-1)
            uncond = (xb < 0).unsqueeze(-1).float()
            kl_d = (target_mu * (torch.log(target_mu.clamp(min=1e-6)) - log_q) * uncond).sum(dim=-1).mean()
            birk_a = birkhoff_nqueens_anyorder_loss(amp_lg, xb)
            loss_a = kl_d + 0.15 * birk_a
            nq_amp_opt.zero_grad()
            loss_a.backward()
            nq_amp_opt.step()

        nq_arms: Dict[str, Any] = {"8x8_dev": {}, "10x10_dev": {}}
        for split_name, x_sp, y_sp, m_sp in [
            ("8x8_dev", nq8_dev_x, nq8_dev_y, nq8_dev_multi),
            ("10x10_dev", nq10_dev_x, nq10_dev_y, nq10_dev_multi),
        ]:
            nq_arms[split_name]["C0_Control_Random_Net"] = evaluate_nqueens_condition(
                "M4_Search_RealityCalibrated_Trust_B16",
                nq_rand_net,
                nq_rand_trust,
                x_sp,
                y_sp,
                m_sp,
                device,
                n_branches=16,
                k_random_orders=k_orders,
                seed=seed,
            )
            nq_arms[split_name]["C1_Control_PeerClue_Uniform"] = evaluate_nqueens_condition(
                "C1_Control_PeerClue_Uniform",
                None,
                None,
                x_sp,
                y_sp,
                m_sp,
                device,
                n_branches=0,
                k_random_orders=k_orders,
                seed=seed,
            )
            for cond_name, t_head, b_val in [
                ("M1_ParamsReal_Base_B0", None, 0),
                ("M2_Search_Unweighted_Clear_B16", None, 16),
                ("M3_Search_SelfGamed_Trust_B16", nq_sg_trust, 16),
                ("M4_Search_RealityCalibrated_Trust_B16", nq_real_trust, 16),
                ("M5_ParamsAmp_Distilled_B0", nq_real_trust, 0),
                ("M6_ParamsAmp_Plus_Search_B16", nq_real_trust, 16),
            ]:
                nq_arms[split_name][cond_name] = evaluate_nqueens_condition(
                    cond_name,
                    nq_model,
                    t_head,
                    x_sp,
                    y_sp,
                    m_sp,
                    device,
                    n_branches=b_val,
                    k_random_orders=k_orders,
                    seed=seed,
                )

        per_seed[str(seed)] = {
            "sudoku_extreme_dev": sudoku_arms,
            "sudoku_b_scaling_m4": sudoku_b_scaling,
            "nqueens_dev": nq_arms,
            "params_real_uncontaminated_check": abs(base_checksum_before - base_checksum_after) < 1e-5,
        }

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    summary = summarize_results(per_seed, seeds)

    return {
        "track": "EXPLORATORY (PROTOCOL.md §10.1)",
        "experiment": "explore_trust_and_amp_split",
        "commit": args.commit,
        "runner_sha256": runner_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "dataset_sha256": data_hashes,
        "splits": {
            "sudoku_base_train": "train.csv[0:3600]",
            "sudoku_trust_calibration": "train.csv[3600:4000]",
            "sudoku_dev_eval": "train.csv[5500:5628]",
            "test_csv_touched": False,
        },
        "parameter_separation": {
            "params_real": "theta_base (any-order world model) + theta_trust (confiability head trained from search into reality x* ~ D_real)",
            "params_search": "VoI - Entropy + consistency branch expansion policy over B in {4, 16, 32}",
            "params_amp": "1-pass head/adapter q_amp trained strictly via KL(stopgrad[mu_B^clear] || q_amp) with zero contamination of params_real",
        },
        "summary": summary,
        "per_seed": per_seed,
    }


def summarize_results(per_seed: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    s_keys = [str(s) for s in seeds]
    sudoku_metrics = [
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "cross_order_nll_std",
        "min_h_voi_nll_per_cell",
        "min_h_voi_ppl",
        "autonomous_empty_acc",
        "autonomous_unit_collisions",
        "one_shot_empty_acc",
        "ece_calibration_error",
        "brier_per_cell",
    ]
    sudoku_summary: Dict[str, Any] = {}
    for arm in per_seed[s_keys[0]]["sudoku_extreme_dev"].keys():
        sudoku_summary[arm] = {
            m: round(float(np.mean([per_seed[s]["sudoku_extreme_dev"][arm][m] for s in s_keys])), 5)
            for m in sudoku_metrics
        }

    b_scale_summary: Dict[str, Any] = {}
    for b_key in per_seed[s_keys[0]]["sudoku_b_scaling_m4"].keys():
        b_scale_summary[b_key] = {
            m: round(float(np.mean([per_seed[s]["sudoku_b_scaling_m4"][b_key][m] for s in s_keys])), 5)
            for m in sudoku_metrics
        }

    nq_metrics = [
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "cross_order_nll_std",
        "min_h_voi_nll_per_cell",
        "min_h_voi_ppl",
        "autonomous_empty_acc",
        "autonomous_valid_board_rate",
        "autonomous_collisions",
        "ece_calibration_error",
    ]
    nq_summary: Dict[str, Any] = {}
    for split_name in ("8x8_dev", "10x10_dev"):
        nq_summary[split_name] = {}
        for arm in per_seed[s_keys[0]]["nqueens_dev"][split_name].keys():
            nq_summary[split_name][arm] = {
                m: round(float(np.mean([per_seed[s]["nqueens_dev"][split_name][arm][m] for s in s_keys])), 5)
                for m in nq_metrics
            }

    return {
        "status": "EXPLORATORY_SMOKE" if len(seeds) == 1 else "EXPLORATORY_DEV",
        "sudoku_extreme_dev_mean": sudoku_summary,
        "sudoku_b_scaling_m4_mean": b_scale_summary,
        "nqueens_dev_mean": nq_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--sudoku-dir", type=str, default="/home/agent/data/datasets/sudoku-extreme")
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--sudoku-steps", type=int, default=420)
    parser.add_argument("--trust-steps", type=int, default=140)
    parser.add_argument("--amp-steps", type=int, default=180)
    parser.add_argument("--nq-steps", type=int, default=280)
    args = parser.parse_args()

    if args.commit == "HEAD":
        import subprocess
        try:
            args.commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode("utf-8").strip()
            )
        except Exception:
            pass

    t0 = time.time()
    results = run_explore_trust_and_amp_split(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(results["summary"], indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
