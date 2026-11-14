#!/usr/bin/env python3
"""
Track E Exploratory Runner 2 (PROTOCOL.md §10.1):
Low-Data Cross-Permutation Variance ("Zero-Curl" Commutator Regularization) &
Unified Stiffness-Entropy VoI Composition Law (explore_lowdata_order_variance.py)

EXPLORATORY ONLY — strictly zero touch of confirmatory test.csv or test splits.
Tests two foundational hypotheses raised by Kelvin:

1. The Low-Data Cross-Permutation Variance Hypothesis ("Bitter-Lesson Data Multiplier"):
   - A true joint distribution P(x) satisfies path-independence across all M! reveal
     permutations: Var_{sigma ~ S_M}[NLL(x; sigma)] = 0, or locally the 2-Step Commutator
     (Zero-Curl) Identity for every pair of variables (i, j) and any partial context x_S:
       log q_theta(X_i=a | x_S) + log q_theta(X_j=b | x_S, X_i=a)
         == log q_theta(X_j=b | x_S) + log q_theta(X_i=a | x_S, X_j=b).
   - Crucially, this identity must hold for ANY (a, b) and ANY partial context x_S without
     requiring ground-truth labels.
   - Across low-data regimes N_train in {32, 128, 512} on Sudoku-Extreme (train.csv[0:512]
     evaluated on development slice train.csv[6000:6128], zero touch of test.csv) and
     Correlated-Parity-Coupled-Grid (synthetic 6x6 Z_4 modular parity grid), compares:
       * Control_Random_Network (run through the exact same B=8 search pipeline per Rule 9.1)
       * Control_Untrained_Init (untrained model at initialization)
       * Control_ClueCopyFloor & Control_PeerClue_Uniform
       * Standard_AO_CE (standard Any-Order autoregressive cross-entropy on N_train boards)
       * AO_CE_Plus_OrderVar_ZeroCurl_SameBoards (adds self-supervised 2-step commutator
         zero-curl loss + cross-order path consistency on the exact same N_train boards,
         0 extra labels)
       * AO_CE_Plus_OrderVar_ZeroCurl_UnlabelledClues (also applies zero-curl + Birkhoff
         consistency on unlabelled clue-only boards from train.csv[512:1536], verified
         0 extra label reads via WatchedLabelArray)
       * AO_CE_Plus_ZeroCurl_And_SearchB8 (combines Zero-Curl training with B=8 internal
         inconsistency-minimizing lookahead search at inference).

2. Stiffness vs. Entropy: Unified Composition Law of External Reveal Order & Internal VoI Search:
   - When an environment has genuine aleatoric coin flips / high-stiffness root variables Z_r
     (high marginal entropy H(Z_r) = log 4, but massive downstream mutual information
     VoI(Z_r) = sum_{w != Z_r} I(Z_r; X_w | x_{S_t}) >> H(Z_r)) alongside biased low-entropy
     uncoupled distractors Y_m (H(Y_m) < H(Z_r), VoI(Y_m) = 0), forced deterministic cells F_m
     (H(F_m) ~ 0, VoI(F_m) = 0), and high-entropy weakly-coupled variables W_m (H(W_m) = log 4,
     small VoI(W_m) > 0):
       * Greedy Min-Entropy-First (gamma = 0) gets trapped picking the uninformative biased
         distractors Y_m before the stiff root coin flips Z_r.
       * Pure Max-VoI picks Z_r first, but then picks noisy high-entropy W_m before locking in
         zero-risk forced cells F_m.
       * The Unified Information-Bellman Composition Law selects:
           v_t^* = argmin_{v notin S_t} [ H(q_{theta,B}^{aug}(X_v | x_{S_t}))
                                          - gamma * sum_{w notin S_t u {v}} I_theta(X_v; X_w | x_{S_t}) ]
         with gamma = 1, prioritizing stiff root coin flips first (where VoI > H), then locking
         in deterministic forced cells (H ~ 0), and deferring noisy guesses until last.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

T_CRIT_DF2 = 4.302652729911275
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
    """
    Verifies public sapientinc/sudoku-extreme train.csv ONLY by exact SHA256.
    Strictly never opens or reads test.csv (Track E Exploratory rule).
    """
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
        raise RuntimeError(
            f"SHA256 mismatch for {train_path}: expected {SUDOKU_EXTREME_TRAIN_SHA256}, got {actual_hash}. "
            "Refusing fallback per PROTOCOL.md Section 9.4."
        )

    repo_root = Path(_REPO_ROOT)
    data_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_data.py")
    model_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_model.py")
    if data_py_hash != INFILOOP_DATA_PY_SHA256 or model_py_hash != INFILOOP_MODEL_PY_SHA256:
        raise RuntimeError("Upstream external/infiloop files failed SHA256 integrity verification.")

    verified = {
        "train.csv": actual_hash,
        "external/infiloop/sudoku_data.py": data_py_hash,
        "external/infiloop/sudoku_model.py": model_py_hash,
        "test_csv_touched": False,
    }
    return train_path, verified


class WatchedLabelArray:
    """Wraps ground-truth label arrays and increments access_count on every __getitem__ read."""

    def __init__(self, array: Any):
        self._array = array
        self.access_count = 0

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            self.access_count += 1
        elif isinstance(idx, (list, tuple, np.ndarray)):
            self.access_count += len(idx)
        elif isinstance(idx, torch.Tensor):
            self.access_count += int(idx.numel())
        else:
            self.access_count += 1
        return self._array[idx]

    def __len__(self) -> int:
        return len(self._array)


def paired_ci95(vals_a: List[float], vals_b: List[float]) -> Dict[str, float]:
    diffs = np.array(vals_a, dtype=np.float64) - np.array(vals_b, dtype=np.float64)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    if n <= 1:
        return {"mean_diff": mean_d, "ci95_low": mean_d, "ci95_high": mean_d}
    sd = float(np.std(diffs, ddof=1))
    se = sd / math.sqrt(n)
    return {
        "mean_diff": round(mean_d, 5),
        "ci95_low": round(float(mean_d - T_CRIT_DF2 * se), 5),
        "ci95_high": round(float(mean_d + T_CRIT_DF2 * se), 5),
    }


# ---------------------------------------------------------------------------
# 1. Sudoku-Extreme Any-Order World Model, Zero-Curl Commutator & VoI Search
# ---------------------------------------------------------------------------

def build_sudoku_unit_indices(device: torch.device) -> torch.Tensor:
    units: List[List[int]] = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + dr) * 9 + (bc * 3 + dc) for dr in range(3) for dc in range(3)])
    return torch.tensor(units, dtype=torch.long, device=device)  # [27, 9]


def count_sudoku_unit_collisions(preds_81: np.ndarray, unit_idx_np: np.ndarray) -> np.ndarray:
    bsz = preds_81.shape[0]
    collisions = np.zeros(bsz, dtype=np.float64)
    for b in range(bsz):
        board = preds_81[b]
        c_cnt = 0
        for u in range(27):
            u_vals = board[unit_idx_np[u]]
            for i in range(9):
                for j in range(i + 1, 9):
                    if u_vals[i] == u_vals[j]:
                        c_cnt += 1
        collisions[b] = c_cnt
    return collisions


class AnyOrderSudokuNet(nn.Module):
    """
    Any-Order Conditional World Model q_theta(X_v | x_{S_t}) on 9x9 Sudoku.
    All weights initialized from N(0, 0.02^2); zero hardcoded solver rules.
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

    def forward(self, x_digits: torch.Tensor, n_steps: int = 5) -> List[torch.Tensor]:
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
        for _ in range(n_steps):
            u_msg, peer_p_cat = self._unit_messages(h, probs, clue_oh9)
            x = infiloop_rms_norm(h + c_emb + self.prob_proj(peer_p_cat), 1e-6)
            z = infiloop_rms_norm(self.cell_mlp(torch.cat([x, u_msg], dim=-1)), 1e-6)
            s = (z.detach() @ self.q_score).clamp(-6.0, 6.0).exp().unsqueeze(-1)
            num = beta * num + 0.35 * s * z
            den = beta * den + 0.35 * s
            h = num / den.clamp(min=1e-7)
            logits = 2.0 * (self.head(h) + self.peer_logit_proj(peer_p_cat))
            step_logits.append(logits)
            probs = F.softmax(logits.detach(), dim=-1)
        return step_logits


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

    cond_mask = x_cond > 0
    if cond_mask.any():
        clue_ce = F.cross_entropy(logits[cond_mask], (x_cond[cond_mask] - 1).clamp(min=0))
    else:
        clue_ce = torch.tensor(0.0, device=logits.device)

    return 1.2 * birkhoff_col_loss + 0.8 * unit_pair_ip + 2.0 * peer_clue_loss + 0.5 * clue_ce


def compute_sudoku_2step_commutator_and_path_variance_loss(
    model: AnyOrderSudokuNet,
    x_s: torch.Tensor,
    logits_s: torch.Tensor,
    unit_idx: torch.Tensor,
    n_steps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Self-Supervised 2-Step Commutator ("Zero-Curl") + Cross-Permutation Path Consistency Loss.
    Uses ZERO ground-truth labels:
      For each partial board x_S in the batch:
        1. Picks a pair of distinct unconditioned cells (i, j) notin S and synthetic actions
           (a, b) with a != b from the model's own predictive distribution q_theta(. | x_S).
        2. Evaluates the two 1-step conditioned states x_{S u {i:a}} and x_{S u {j:b}} in a
           single batched forward pass.
        3. Enforces the exact 2-Step Commutator Identity (Zero-Curl):
             C_theta(i,a; j,b | x_S) = [log q_theta(X_i=a | x_S) + log q_theta(X_j=b | x_S, X_i=a)]
                                     - [log q_theta(X_j=b | x_S) + log q_theta(X_i=a | x_S, X_j=b)] == 0,
           plus cross-branch conditional JSD consistency on remaining unconditioned cells k notin S u {i,j}.
    Returns:
      (total_self_sup_loss, mean_abs_curl_detached)
    """
    bsz = x_s.shape[0]
    device = x_s.device
    log_q_s = F.log_softmax(logits_s, dim=-1)  # [B, 81, 9]
    q_s = F.softmax(logits_s.detach(), dim=-1)  # [B, 81, 9]

    # Select two distinct unconditioned cells (i, j) per board
    is_empty = x_s == 0  # [B, 81]
    noise = torch.rand_like(q_s[:, :, 0])
    pick_score = noise + torch.where(is_empty, torch.zeros_like(noise), torch.full_like(noise, 1e6))
    top2_cells = pick_score.topk(2, dim=-1, largest=False).indices  # [B, 2]
    cell_i = top2_cells[:, 0]  # [B]
    cell_j = top2_cells[:, 1]  # [B]

    batch_idx = torch.arange(bsz, device=device)
    q_i = q_s[batch_idx, cell_i]  # [B, 9]
    q_j = q_s[batch_idx, cell_j]  # [B, 9]
    act_a = q_i.argmax(dim=-1)  # [B]
    # Ensure act_b != act_a so (i:a, j:b) is always a valid non-colliding joint assignment
    q_j_masked = q_j.clone()
    q_j_masked[batch_idx, act_a] = -1.0
    act_b = q_j_masked.argmax(dim=-1)  # [B]

    # Construct x_{S u {i:a}} and x_{S u {j:b}}
    x_si = x_s.clone()
    x_sj = x_s.clone()
    i_was_empty = is_empty[batch_idx, cell_i]
    j_was_empty = is_empty[batch_idx, cell_j]
    x_si[batch_idx, cell_i] = torch.where(i_was_empty, act_a + 1, x_si[batch_idx, cell_i])
    x_sj[batch_idx, cell_j] = torch.where(j_was_empty, act_b + 1, x_sj[batch_idx, cell_j])

    # Single batched forward pass on [x_si; x_sj]
    x_cat = torch.cat([x_si, x_sj], dim=0)  # [2B, 81]
    logits_cat = model(x_cat, n_steps=n_steps)[-1]
    logits_si, logits_sj = logits_cat[:bsz], logits_cat[bsz:]
    log_q_si = F.log_softmax(logits_si, dim=-1)  # [B, 81, 9]
    log_q_sj = F.log_softmax(logits_sj, dim=-1)  # [B, 81, 9]

    # 1. Exact 2-Step Commutator (Zero-Curl) on (i, a; j, b):
    log_qi_a = log_q_s[batch_idx, cell_i, act_a]
    log_qj_b = log_q_s[batch_idx, cell_j, act_b]
    log_qj_b_given_ia = log_q_si[batch_idx, cell_j, act_b]
    log_qi_a_given_jb = log_q_sj[batch_idx, cell_i, act_a]

    valid_pair = (i_was_empty & j_was_empty).float()
    curl_diff = ((log_qi_a + log_qj_b_given_ia) - (log_qj_b + log_qi_a_given_jb)).clamp(-6.0, 6.0)
    curl_loss = ((curl_diff ** 2) * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)
    mean_abs_curl = (curl_diff.detach().abs() * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)

    # 2. Symmetric Cross-Order Conditional Consistency on remaining cells k notin S u {i,j}
    rem_mask = ((x_si == 0) & (x_sj == 0)).unsqueeze(-1).float()  # [B, 81, 1]
    q_si_det = F.softmax(logits_si.detach(), dim=-1).clamp(min=1e-6)
    q_sj_det = F.softmax(logits_sj.detach(), dim=-1).clamp(min=1e-6)
    q_s_det = q_s.clamp(min=1e-6)
    rem_denom = rem_mask.sum().clamp(min=1.0)
    cross_kl = (
        0.5 * ((q_si_det * (torch.log(q_si_det) - log_q_sj) * rem_mask).sum() / rem_denom)
        + 0.5 * ((q_sj_det * (torch.log(q_sj_det) - log_q_si) * rem_mask).sum() / rem_denom)
        + 0.5 * ((q_s_det * (torch.log(q_s_det) - 0.5 * (log_q_si + log_q_sj)) * rem_mask).sum() / rem_denom)
    )

    # 3. Structural Birkhoff consistency on the conditioned rollout states
    birk_cat = sudoku_anyorder_birkhoff_contradiction_loss(logits_cat, x_cat, unit_idx)

    total_reg = 0.60 * curl_loss + 0.40 * cross_kl + 0.25 * birk_cat
    return total_reg, mean_abs_curl


@torch.no_grad()
def build_sudoku_amplified_teacher_target(
    q_real: AnyOrderSudokuNet,
    x_s: torch.Tensor,
    unit_idx: torch.Tensor,
    n_branches: int = 4,
    beta_clear: float = 2.5,
) -> torch.Tensor:
    """
    Uses the FROZEN real-data-only parameter set q_real (theta_real) to compute the
    search-amplified target distribution q_{theta_real, B}^{aug}(. | x_S) for distilling
    into the second parameter set q_amp (theta_amp).
    Also applies a 1-step peer-clue mask projection so the amplified teacher is strictly
    consistent with all conditioned clues in x_S.
    """
    q_real.eval()
    aug_p, _, _ = predict_sudoku_anyorder_search_and_voi(
        q_real, x_s, unit_idx, n_branches=n_branches, beta_clear=beta_clear, rollout_depth=2, n_steps=4, compute_voi=False
    )
    peer_p = peer_clue_uniform_sudoku_probs(x_s, unit_idx)
    # Multiplicative product-of-experts between search-amplified q_real and peer-clue support
    prod_p = (aug_p * peer_p).clamp(min=1e-7)
    prod_p = prod_p / prod_p.sum(dim=-1, keepdim=True)
    return prod_p.detach()


def compute_sudoku_twoparam_distill_and_zerocurl_loss(
    q_amp: AnyOrderSudokuNet,
    q_real_frozen: AnyOrderSudokuNet,
    x_s: torch.Tensor,
    logits_amp_s: torch.Tensor,
    unit_idx: torch.Tensor,
    n_steps: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-Parameter Split Distillation & Teacher-Anchored Zero-Curl Loss for theta_amp (q_amp),
    holding theta_real (q_real_frozen) strictly frozen:
      1. Amplification Distillation: KL( q_{theta_real, B=4}^{aug}(. | x_S) || q_{theta_amp}(. | x_S) )
      2. Teacher-Anchored 2-Step Commutator (One-Way Zero-Curl):
         Evaluates the more-informed 1-step conditioned states x_{S u {i:a}} and x_{S u {j:b}}
         using the FROZEN real-data teacher q_real_frozen (and detached q_amp), so gradients
         flow strictly into the sparser state x_S of q_amp and NEVER blur dense states!
      3. Birkhoff unit doubly-stochastic consistency on q_amp.
    """
    bsz = x_s.shape[0]
    device = x_s.device
    q_real_frozen.eval()

    log_q_amp_s = F.log_softmax(logits_amp_s, dim=-1)  # [B, 81, 9]
    q_amp_s_det = F.softmax(logits_amp_s.detach(), dim=-1)

    # 1. Amplify frozen q_real on x_s and distill into q_amp
    with torch.no_grad():
        teacher_aug_p = build_sudoku_amplified_teacher_target(q_real_frozen, x_s, unit_idx, n_branches=4, beta_clear=2.5)
    is_empty = x_s == 0  # [B, 81]
    emp_f = is_empty.unsqueeze(-1).float()
    distill_kl = (
        (teacher_aug_p * (torch.log(teacher_aug_p.clamp(min=1e-7)) - log_q_amp_s) * emp_f).sum()
        / emp_f.sum().clamp(min=1.0)
    )

    # 2. Teacher-Anchored 2-Step Commutator (Zero-Curl) on pairs (i, a; j, b) with a != b
    noise = torch.rand_like(q_amp_s_det[:, :, 0])
    pick_score = noise + torch.where(is_empty, torch.zeros_like(noise), torch.full_like(noise, 1e6))
    top2_cells = pick_score.topk(2, dim=-1, largest=False).indices  # [B, 2]
    cell_i = top2_cells[:, 0]
    cell_j = top2_cells[:, 1]

    batch_idx = torch.arange(bsz, device=device)
    act_a = teacher_aug_p[batch_idx, cell_i].argmax(dim=-1)
    t_j_masked = teacher_aug_p[batch_idx, cell_j].clone()
    t_j_masked[batch_idx, act_a] = -1.0
    act_b = t_j_masked.argmax(dim=-1)

    x_si = x_s.clone()
    x_sj = x_s.clone()
    i_was_empty = is_empty[batch_idx, cell_i]
    j_was_empty = is_empty[batch_idx, cell_j]
    x_si[batch_idx, cell_i] = torch.where(i_was_empty, act_a + 1, x_si[batch_idx, cell_i])
    x_sj[batch_idx, cell_j] = torch.where(j_was_empty, act_b + 1, x_sj[batch_idx, cell_j])

    x_cat = torch.cat([x_si, x_sj], dim=0)
    with torch.no_grad():
        # Frozen teacher evaluates the more-conditioned states x_{S u {i:a}} and x_{S u {j:b}}
        lg_real_cat = q_real_frozen(x_cat, n_steps=n_steps)[-1]
        lq_real_si = F.log_softmax(lg_real_cat[:bsz], dim=-1)
        lq_real_sj = F.log_softmax(lg_real_cat[bsz:], dim=-1)

    lg_amp_cat = q_amp(x_cat, n_steps=n_steps)[-1]
    lq_amp_si = F.log_softmax(lg_amp_cat[:bsz], dim=-1)
    lq_amp_sj = F.log_softmax(lg_amp_cat[bsz:], dim=-1)

    # One-way teacher-anchored curl + self-curl with detached 1-step targets so dense states never blur
    log_qi_a = log_q_amp_s[batch_idx, cell_i, act_a]
    log_qj_b = log_q_amp_s[batch_idx, cell_j, act_b]
    target_step_diff = 0.5 * (
        (lq_real_sj[batch_idx, cell_i, act_a] - lq_real_si[batch_idx, cell_j, act_b])
        + (lq_amp_sj[batch_idx, cell_i, act_a].detach() - lq_amp_si[batch_idx, cell_j, act_b].detach())
    )
    valid_pair = (i_was_empty & j_was_empty).float()
    curl_diff = ((log_qi_a - log_qj_b) - target_step_diff).clamp(-5.0, 5.0)
    curl_loss = ((curl_diff ** 2) * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)
    mean_abs_curl = (curl_diff.detach().abs() * valid_pair).sum() / valid_pair.sum().clamp(min=1.0)

    birk_s = sudoku_anyorder_birkhoff_contradiction_loss(logits_amp_s, x_s, unit_idx)
    total_loss = 0.85 * distill_kl + 0.45 * curl_loss + 0.30 * birk_s
    return total_loss, mean_abs_curl


@torch.no_grad()
def evaluate_sudoku_2step_commutator_curl(
    model: Optional[AnyOrderSudokuNet],
    arm_name: str,
    x_eval_np: np.ndarray,
    unit_idx: torch.Tensor,
    device: torch.device,
    q_real_anchor: Optional[AnyOrderSudokuNet] = None,
    n_pairs_per_board: int = 4,
    seed: int = 101,
) -> Dict[str, float]:
    """
    Evaluates the empirical 2-Step Commutator ("Curl") violation on held-out dev boards:
      | [log q(X_i=a | x_S) + log q(X_j=b | x_S, X_i=a)]
        - [log q(X_j=b | x_S) + log q(X_i=a | x_S, X_j=b)] |
    across random pairs of empty cells (i, j) and distinct candidate digits (a != b).
    """
    nb = x_eval_np.shape[0]
    x_s = torch.tensor(x_eval_np, dtype=torch.long, device=device)

    def base_log_probs(x_in: torch.Tensor) -> torch.Tensor:
        if arm_name == "Control_ClueCopyFloor":
            return torch.full((x_in.shape[0], 81, 9), -math.log(9.0), device=device)
        if arm_name == "Control_PeerClue_Uniform":
            p = peer_clue_uniform_sudoku_probs(x_in, unit_idx)
            return torch.log(p.clamp(min=1e-6))
        assert model is not None
        model.eval()
        lg = model(x_in, n_steps=5)[-1]
        p = F.softmax(lg, dim=-1)
        if q_real_anchor is not None:
            q_real_anchor.eval()
            p_real = F.softmax(q_real_anchor(x_in, n_steps=5)[-1], dim=-1)
            p_peer = peer_clue_uniform_sudoku_probs(x_in, unit_idx)
            p = (0.70 * p + 0.30 * p_real) * p_peer
            p = p.clamp(min=1e-7)
            p = p / p.sum(dim=-1, keepdim=True)
        eps = 1e-4
        p = (1.0 - eps) * p + (eps / 9.0)
        return torch.log(p / p.sum(dim=-1, keepdim=True))

    log_q_s = base_log_probs(x_s)  # [nb, 81, 9]
    q_s = torch.exp(log_q_s)

    abs_curls: List[float] = []
    sq_curls: List[float] = []
    rng = np.random.RandomState(seed + 777)

    for p_idx in range(n_pairs_per_board):
        cell_i = torch.zeros(nb, dtype=torch.long, device=device)
        cell_j = torch.zeros(nb, dtype=torch.long, device=device)
        for b in range(nb):
            emp = np.where(x_eval_np[b] == 0)[0]
            if len(emp) >= 2:
                chosen = rng.choice(emp, size=2, replace=False)
                cell_i[b] = int(chosen[0])
                cell_j[b] = int(chosen[1])

        batch_idx = torch.arange(nb, device=device)
        # Pick top-1 predicted digit at cell_i and top distinct digit (b != a) at cell_j
        act_a = q_s[batch_idx, cell_i].argmax(dim=-1)
        q_j_m = q_s[batch_idx, cell_j].clone()
        q_j_m[batch_idx, act_a] = -1.0
        act_b = q_j_m.argmax(dim=-1)

        x_si = x_s.clone()
        x_sj = x_s.clone()
        x_si[batch_idx, cell_i] = act_a + 1
        x_sj[batch_idx, cell_j] = act_b + 1

        log_q_cat = base_log_probs(torch.cat([x_si, x_sj], dim=0))
        log_q_si, log_q_sj = log_q_cat[:nb], log_q_cat[nb:]

        path_ij = log_q_s[batch_idx, cell_i, act_a] + log_q_si[batch_idx, cell_j, act_b]
        path_ji = log_q_s[batch_idx, cell_j, act_b] + log_q_sj[batch_idx, cell_i, act_a]
        diff = (path_ij - path_ji).cpu().numpy()
        abs_curls.extend(np.abs(diff).tolist())
        sq_curls.extend((diff ** 2).tolist())

    return {
        "commutator_curl_mae": float(np.mean(abs_curls)),
        "commutator_curl_rmse": float(math.sqrt(float(np.mean(sq_curls)))),
    }


def compute_pure_neural_sudoku_inconsistency_energy(
    leaf_probs: torch.Tensor,
    base_probs: torch.Tensor,
    x_branch: torch.Tensor,
    unit_idx: torch.Tensor,
) -> torch.Tensor:
    nb, n_branches, _, _ = leaf_probs.shape
    flat_p = leaf_probs.reshape(nb * n_branches, 81, 9)
    unit_p = flat_p[:, unit_idx, :]
    unit_digit_sum = unit_p.sum(dim=2)
    excess = F.relu(unit_digit_sum - 1.0)
    unit_coll_energy = (excess * excess).sum(dim=(1, 2)).view(nb, n_branches)

    uncond_mask = (x_branch == 0).float()
    uncond_denom = uncond_mask.sum(dim=-1).clamp(min=1.0)
    max_p = leaf_probs.max(dim=-1).values.clamp(min=1e-6)
    deplete_energy = ((-torch.log(max_p) * uncond_mask).sum(dim=-1) / uncond_denom) * 10.0

    base_exp = base_probs.unsqueeze(1).expand_as(leaf_probs).clamp(min=1e-6)
    leaf_clamped = leaf_probs.clamp(min=1e-6)
    kl_cell = (leaf_clamped * (torch.log(leaf_clamped) - torch.log(base_exp))).sum(dim=-1)
    path_kl_energy = (kl_cell * uncond_mask).sum(dim=-1) / uncond_denom

    return unit_coll_energy + deplete_energy + 0.25 * path_kl_energy


@torch.no_grad()
def predict_sudoku_anyorder_search_and_voi(
    model: AnyOrderSudokuNet,
    x_cond: torch.Tensor,
    unit_idx: torch.Tensor,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    rollout_depth: int = 2,
    n_steps: int = 4,
    compute_voi: bool = False,
    leaf_judge_model: Optional[AnyOrderSudokuNet] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes Search-Augmented Any-Order distribution q_{theta, B}^{aug}(X_v | x_{S_t})
    and (when compute_voi=True or n_branches > 0) estimates the Downstream Mutual Information
    (Stiffness / VoI) for unconditioned cells v notin S_t:
      VoI(v | x_{S_t}) = sum_{w notin S_t u {v}} I_theta(X_v; X_w | x_{S_t})
    When leaf_judge_model (frozen q_real) is provided alongside model (q_amp):
      - q_amp proposes the branches and amortized lookahead prior,
      - frozen q_real serves as the incorruptible leaf consistency judge and dense-state anchor!
    """
    model.eval()
    nb = x_cond.shape[0]
    device = x_cond.device
    base_logits = model(x_cond, n_steps=n_steps)[-1]
    base_probs = F.softmax(base_logits, dim=-1)  # [nb, 81, 9]
    eps = 1e-4

    if leaf_judge_model is not None:
        leaf_judge_model.eval()
        real_logits = leaf_judge_model(x_cond, n_steps=n_steps)[-1]
        real_probs = F.softmax(real_logits, dim=-1)
        # Blend q_amp (global lookahead on sparse states) with frozen q_real (sharp local conditional on dense states)
        emp_frac = (x_cond == 0).float().mean(dim=-1, keepdim=True).unsqueeze(-1)  # [nb, 1, 1] in [0, 1]
        w_amp = (0.35 + 0.65 * emp_frac).clamp(0.35, 0.90)
        peer_p = peer_clue_uniform_sudoku_probs(x_cond, unit_idx)
        base_probs = (w_amp * base_probs + (1.0 - w_amp) * real_probs) * peer_p
        base_probs = base_probs.clamp(min=1e-7)
        base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)

    p_cl = base_probs.clamp(min=1e-7)
    base_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)  # [nb, 81]
    is_empty = x_cond == 0

    # Estimate downstream stiffness / mutual information VoI(v) for candidate empty cells
    voi_scores = torch.zeros((nb, 81), device=device)
    if compute_voi:
        # Select K_cand = 6 candidate uncertain empty cells per board to probe 2-way counterfactuals
        k_cand = 6
        top2_vals, top2_idx = base_probs.topk(2, dim=-1)  # [nb, 81, 2]
        probe_rank = base_ent + torch.where(is_empty, torch.zeros_like(base_ent), torch.full_like(base_ent, 1e6))
        probe_rank = probe_rank + torch.where(
            top2_vals[:, :, 0] > 0.95, torch.full_like(base_ent, 5.0), torch.zeros_like(base_ent)
        )
        cand_cells = probe_rank.topk(k_cand, dim=-1, largest=False).indices  # [nb, k_cand]

        x_cf = x_cond.unsqueeze(1).unsqueeze(2).expand(nb, k_cand, 2, 81).clone()
        cand_d = top2_idx.gather(1, cand_cells.unsqueeze(-1).expand(nb, k_cand, 2)) + 1  # [nb, k_cand, 2]
        cand_p = top2_vals.gather(1, cand_cells.unsqueeze(-1).expand(nb, k_cand, 2)).clamp(min=1e-5)
        cand_p_norm = cand_p / cand_p.sum(dim=-1, keepdim=True)  # [nb, k_cand, 2]

        c_idx_exp = cand_cells.unsqueeze(-1).unsqueeze(-1).expand(nb, k_cand, 2, 1)
        x_cf.scatter_(3, c_idx_exp, cand_d.unsqueeze(-1))

        judge_net = leaf_judge_model if leaf_judge_model is not None else model
        cf_logits = judge_net(x_cf.reshape(nb * k_cand * 2, 81), n_steps=max(3, n_steps - 1))[-1]
        cf_probs = F.softmax(cf_logits, dim=-1).view(nb, k_cand, 2, 81, 9).clamp(min=1e-7)
        cf_ent = -(cf_probs * torch.log(cf_probs)).sum(dim=-1)  # [nb, k_cand, 2, 81]
        exp_post_ent = (cand_p_norm.unsqueeze(-1) * cf_ent).sum(dim=2)  # [nb, k_cand, 81]

        mi_vw = F.relu(base_ent.unsqueeze(1) - exp_post_ent) * is_empty.unsqueeze(1).float()  # [nb, k_cand, 81]
        self_mask = F.one_hot(cand_cells, num_classes=81).float()
        mi_vw = mi_vw * (1.0 - self_mask)
        cand_voi = mi_vw.sum(dim=-1)  # [nb, k_cand]
        voi_scores.scatter_(1, cand_cells, cand_voi)

    if n_branches <= 0:
        out_probs = (1.0 - eps) * base_probs + (eps / 9.0)
        return out_probs / out_probs.sum(dim=-1, keepdim=True), voi_scores, {
            "mean_branch_energy": 0.0,
            "min_branch_energy": 0.0,
        }

    r_bits = {4: 2, 8: 3, 16: 4}.get(n_branches, 3)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)],
        dtype=torch.long,
        device=device,
    )
    top2_vals, top2_idx = base_probs.topk(2, dim=-1)
    uncertain_score = (
        base_ent
        - 0.25 * voi_scores
        + torch.where(is_empty, torch.zeros_like(base_ent), torch.full_like(base_ent, 1e6))
        + torch.where(top2_vals[:, :, 0] > 0.92, torch.full_like(base_ent, 10.0), torch.zeros_like(base_ent))
    )
    root_cells = uncertain_score.topk(r_bits, dim=-1, largest=False).indices  # [nb, r_bits]

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

    cells_per_step = 3
    for _ in range(rollout_depth - 1):
        flat_x = x_branch.reshape(nb * n_branches, 81)
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

    # Frozen q_real (leaf_judge_model) judges the leaf rollout states when provided!
    judge_net = leaf_judge_model if leaf_judge_model is not None else model
    leaf_logits = judge_net(x_branch.reshape(nb * n_branches, 81), n_steps=n_steps)[-1]
    leaf_probs = F.softmax(leaf_logits, dim=-1).view(nb, n_branches, 81, 9)
    if leaf_judge_model is not None:
        amp_leaf_probs = F.softmax(model(x_branch.reshape(nb * n_branches, 81), n_steps=n_steps)[-1], dim=-1).view(
            nb, n_branches, 81, 9
        )
        leaf_probs = 0.50 * leaf_probs + 0.50 * amp_leaf_probs
    e_inc = compute_pure_neural_sudoku_inconsistency_energy(leaf_probs, base_probs, x_branch, unit_idx)

    added_mask = ((x_branch > 0) & (x_cond.unsqueeze(1) == 0)).unsqueeze(-1).float()
    added_oh = F.one_hot((x_branch - 1).clamp(0, 8), num_classes=9).float()
    branch_pred = added_mask * (0.65 * added_oh + 0.35 * leaf_probs) + (1.0 - added_mask) * leaf_probs

    if beta_clear <= 0.0:
        weights = torch.full((nb, n_branches), 1.0 / float(n_branches), device=device)
    else:
        e_shift = e_inc - e_inc.min(dim=-1, keepdim=True).values
        log_w = 0.25 * log_prior - beta_clear * e_shift
        weights = F.softmax(log_w, dim=-1)

    cleared_probs = (weights.unsqueeze(-1).unsqueeze(-1) * branch_pred).sum(dim=1)
    aug_probs = 0.85 * cleared_probs + 0.15 * base_probs
    if leaf_judge_model is not None:
        aug_probs = aug_probs * peer_clue_uniform_sudoku_probs(x_cond, unit_idx)
        aug_probs = aug_probs.clamp(min=1e-7)
        aug_probs = aug_probs / aug_probs.sum(dim=-1, keepdim=True)
    aug_probs = (1.0 - eps) * aug_probs + (eps / 9.0)
    aug_probs = aug_probs / aug_probs.sum(dim=-1, keepdim=True)

    return aug_probs, voi_scores, {
        "mean_branch_energy": float(e_inc.mean().item()),
        "min_branch_energy": float(e_inc.min(dim=-1).values.mean().item()),
    }


def peer_clue_uniform_sudoku_probs(x_cond: torch.Tensor, unit_idx: torch.Tensor) -> torch.Tensor:
    clue_oh = F.one_hot(x_cond.clamp(min=0, max=9), num_classes=10)[:, :, 1:].float()
    row_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 0)
    col_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 1)
    box_c = AnyOrderSudokuNet._port_peer_sum(clue_oh, 2)
    allowed = (1.0 - (row_c + col_c + box_c).clamp(max=1.0)).clamp(min=0.0)
    allowed_sum = allowed.sum(dim=-1, keepdim=True)
    uniform9 = torch.full_like(allowed, 1.0 / 9.0)
    probs = torch.where(allowed_sum > 0.5, allowed / allowed_sum.clamp(min=1.0), uniform9)
    eps = 1e-4
    probs = (1.0 - eps) * probs + (eps / 9.0)
    return probs / probs.sum(dim=-1, keepdim=True)


@torch.no_grad()
def evaluate_sudoku_lowdata_arm(
    arm_name: str,
    model: Optional[AnyOrderSudokuNet],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    unit_idx: torch.Tensor,
    device: torch.device,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    k_random_orders: int = 4,
    n_reveal_blocks: int = 9,
    seed: int = 101,
    leaf_judge_model: Optional[AnyOrderSudokuNet] = None,
) -> Dict[str, Any]:
    nb = x_eval_np.shape[0]
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)
    unit_idx_np = unit_idx.cpu().numpy()

    def get_probs(x_cur: torch.Tensor) -> torch.Tensor:
        if arm_name == "Control_ClueCopyFloor":
            return torch.full((nb, 81, 9), 1.0 / 9.0, device=device)
        if arm_name == "Control_PeerClue_Uniform":
            return peer_clue_uniform_sudoku_probs(x_cur, unit_idx)
        assert model is not None
        p, _, _ = predict_sudoku_anyorder_search_and_voi(
            model,
            x_cur,
            unit_idx,
            n_branches=n_branches,
            beta_clear=beta_clear,
            compute_voi=False,
            leaf_judge_model=leaf_judge_model,
        )
        return p

    p0 = get_probs(x_init)
    empty_init_mask = x_eval_np == 0
    clue_init_mask = x_eval_np > 0
    preds_step0 = (p0.argmax(dim=-1) + 1).cpu().numpy()
    preds_step0[clue_init_mask] = x_eval_np[clue_init_mask]
    one_shot_empty_acc = float(((preds_step0 == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))

    board_order_total_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    board_order_cell_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    early_step_nlls: List[float] = []
    late_step_nlls: List[float] = []

    for k in range(k_random_orders):
        order_blocks = [torch.zeros((nb, 81), dtype=torch.bool, device=device) for _ in range(n_reveal_blocks)]
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
            probs_step = get_probs(x_cur)
            true_digits = (y_true - 1).clamp(0, 8)
            true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            step_nll = -torch.log(true_p).to(torch.float64) * b_mask.to(torch.float64)
            total_nll_k = total_nll_k + step_nll.sum(dim=-1)
            blk_mean_nll = float((step_nll.sum() / b_mask.to(torch.float64).sum().clamp(min=1.0)).item())
            if b_idx == 0:
                early_step_nlls.append(blk_mean_nll)
            if b_idx == n_reveal_blocks - 1:
                late_step_nlls.append(blk_mean_nll)
            x_cur = torch.where(b_mask, y_true, x_cur)

        total_nll_np = total_nll_k.cpu().numpy()
        board_order_total_nll[:, k] = total_nll_np
        board_order_cell_nll[:, k] = total_nll_np / empty_counts

    random_order_joint_nll = float(board_order_total_nll.mean())
    rand_order_nll_per_cell = float(board_order_cell_nll.mean())
    rand_order_ppl = float(math.exp(min(rand_order_nll_per_cell, 20.0)))
    cross_order_nll_std = float(np.std(board_order_total_nll, axis=1, ddof=1).mean())
    cross_order_cell_nll_std = float(np.std(board_order_cell_nll, axis=1, ddof=1).mean())

    # Adaptive Min-Entropy Order NLL
    x_cur_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_scored_cells = torch.zeros(nb, dtype=torch.float64, device=device)

    for b_idx in range(n_reveal_blocks):
        rem_mask = x_cur_minh == 0
        if not rem_mask.any():
            break
        probs_step = get_probs(x_cur_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        rem_counts = rem_mask.sum(dim=-1)
        blocks_left = n_reveal_blocks - b_idx
        step_mask = torch.zeros_like(rem_mask)
        for i in range(nb):
            r_cnt = int(rem_counts[i].item())
            if r_cnt == 0:
                continue
            k_pick = max(1, math.ceil(r_cnt / float(blocks_left))) if blocks_left > 1 else r_cnt
            ent_i = ent[i].clone()
            ent_i[~rem_mask[i]] = 1e9
            pick_idx = ent_i.topk(k_pick, largest=False).indices
            step_mask[i, pick_idx] = True

        true_digits = (y_true - 1).clamp(0, 8)
        true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        step_nll = -torch.log(true_p).to(torch.float64) * step_mask.to(torch.float64)
        minh_total_nll = minh_total_nll + step_nll.sum(dim=-1)
        minh_scored_cells = minh_scored_cells + step_mask.to(torch.float64).sum(dim=-1)
        x_cur_minh = torch.where(step_mask, y_true, x_cur_minh)

    adaptive_order_joint_nll = float(minh_total_nll.mean().item())
    min_h_nll_per_cell = float((minh_total_nll / minh_scored_cells.clamp(min=1.0)).mean().item())
    min_h_ppl = float(math.exp(min(min_h_nll_per_cell, 20.0)))

    # Autonomous Sequential Rollout
    x_auto = x_init.clone()
    for s_idx in range(n_reveal_blocks):
        rem_mask = x_auto == 0
        if not rem_mask.any():
            break
        probs_auto = get_probs(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        pred_digits = probs_auto.argmax(dim=-1) + 1
        rem_counts = rem_mask.sum(dim=-1)
        steps_left = n_reveal_blocks - s_idx
        commit_mask = torch.zeros_like(rem_mask)
        for i in range(nb):
            r_cnt = int(rem_counts[i].item())
            if r_cnt == 0:
                continue
            k_pick = max(1, math.ceil(r_cnt / float(steps_left))) if steps_left > 1 else r_cnt
            ent_i = ent[i].clone()
            ent_i[~rem_mask[i]] = 1e9
            pick_idx = ent_i.topk(k_pick, largest=False).indices
            commit_mask[i, pick_idx] = True
        x_auto = torch.where(commit_mask, pred_digits, x_auto)

    auto_np = x_auto.cpu().numpy()
    auto_empty_acc = float(((auto_np == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))
    auto_collisions = float(count_sudoku_unit_collisions(auto_np, unit_idx_np).mean())

    curl_metrics = evaluate_sudoku_2step_commutator_curl(
        model,
        arm_name,
        x_eval_np,
        unit_idx,
        device,
        q_real_anchor=leaf_judge_model,
        n_pairs_per_board=3,
        seed=seed,
    )

    return {
        "n_branches": n_branches,
        "random_order_joint_nll": random_order_joint_nll,
        "rand_order_nll_per_cell": rand_order_nll_per_cell,
        "rand_order_ppl": rand_order_ppl,
        "adaptive_order_joint_nll": adaptive_order_joint_nll,
        "min_h_nll_per_cell": min_h_nll_per_cell,
        "min_h_ppl": min_h_ppl,
        "cross_order_nll_std": cross_order_nll_std,
        "cross_order_cell_nll_std": cross_order_cell_nll_std,
        "commutator_curl_mae": curl_metrics["commutator_curl_mae"],
        "commutator_curl_rmse": curl_metrics["commutator_curl_rmse"],
        "early_step_nll_per_cell": float(np.mean(early_step_nlls)),
        "late_step_nll_per_cell": float(np.mean(late_step_nlls)),
        "one_shot_empty_acc": one_shot_empty_acc,
        "autonomous_empty_acc": auto_empty_acc,
        "autonomous_unit_collisions": auto_collisions,
    }


# ---------------------------------------------------------------------------
# 2. Correlated-Parity-Coupled-Grid (Synthetic 6x6 Z_4 Modular Constraint Grid)
#    for Low-Data Cross-Permutation Variance & Zero-Curl Testing
# ---------------------------------------------------------------------------

def generate_correlated_parity_grid_dataset(
    n_samples: int, n_clues: int = 16, seed: int = 2026
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Explicitly labelled synthetic environment (Rule 3):
    Generates overconstrained 6x6 grids with entries in {1, 2, 3, 4} (representing Z_4 residues 0..3)
    consisting of four coupled 3x3 zero-sum modular subgrids:
      1. Every 3-cell half-row sums to 0 mod 4:
         X_{r,0} + X_{r,1} + X_{r,2} == 0 (mod 4),  X_{r,3} + X_{r,4} + X_{r,5} == 0 (mod 4)
      2. Every 3-cell half-column sums to 0 mod 4:
         X_{0,c} + X_{1,c} + X_{2,c} == 0 (mod 4),  X_{3,c} + X_{4,c} + X_{5,c} == 0 (mod 4)
      3. Periodic quadrant parity coupling:
         X_{r, c+3} == X_{r, c} (mod 4),  X_{r+3, c} == X_{r, c} (mod 4)
    With n_clues = 16 revealed cells out of 36, the remaining 20 empty cells are 100% uniquely
    determined by the overlapping 3-cell row/col triad sums and quadrant couplings.
    """
    rng = np.random.RandomState(seed)
    xs = np.zeros((n_samples, 36), dtype=np.int64)
    ys = np.zeros((n_samples, 36), dtype=np.int64)

    for i in range(n_samples):
        sub = np.zeros((3, 3), dtype=np.int64)
        # Sample top-left 2x2 freely in 0..3
        sub[:2, :2] = rng.randint(0, 4, size=(2, 2))
        for r in range(2):
            sub[r, 2] = (-int(sub[r, :2].sum())) % 4
        for c in range(3):
            sub[2, c] = (-int(sub[:2, c].sum())) % 4
        grid = np.tile(sub, (2, 2))
        y_flat = grid.reshape(36) + 1  # values in 1..4 (0 = unconditioned)
        clue_idx = rng.choice(36, size=n_clues, replace=False)
        x_flat = np.zeros(36, dtype=np.int64)
        x_flat[clue_idx] = y_flat[clue_idx]
        xs[i] = x_flat
        ys[i] = y_flat

    row_units = np.array([[r * 6 + c for c in range(6)] for r in range(6)], dtype=np.int64)
    col_units = np.array([[r * 6 + c for r in range(6)] for c in range(6)], dtype=np.int64)
    return xs, ys, row_units, col_units


class AnyOrderParityGridNet(nn.Module):
    """
    Any-Order Conditional World Model q_theta(X_v | x_{S_t}) on the 6x6 Z_4 Parity Grid.
    All weights initialized from N(0, 0.02^2).
    """

    def __init__(self, hidden_size: int = 64, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size
        self.val_emb = nn.Embedding(5, hidden_size)
        self.pos_emb = nn.Embedding(36, hidden_size)
        nn.init.normal_(self.val_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # Per-cell message channels: probs(4) + row_triad_conv(4) + col_triad_conv(4) + quad_peer_avg(4) + r_emp(1) + c_emp(1) = 18
        self.msg_proj = nn.Linear(18, hidden_size)
        self.row_mlp = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.col_mlp = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.cell_mlp = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.head = nn.Linear(hidden_size, 4)
        self.peer_logit_proj = nn.Linear(18, 4)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.peer_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.peer_logit_proj.bias)
        self.register_buffer("pos_idx", torch.arange(36, dtype=torch.long, device=device))

    @staticmethod
    def _triad_zero_sum_conv(p_triad: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor [..., 3, 4] of Z_4 distributions across a 3-cell triad (u, v, w),
        computes for each position the implied distribution of (-X_other1 - X_other2) mod 4
        via exact 4-point circular convolution.
        """
        p0 = p_triad[..., 0, :]
        p1 = p_triad[..., 1, :]
        p2 = p_triad[..., 2, :]

        def conv_neg(pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
            # P(d) = sum_{a=0..3} pa(a) * pb((-d - a) mod 4)
            outs = []
            for d in range(4):
                s = torch.zeros_like(pa[..., 0])
                for a in range(4):
                    b = (-d - a) % 4
                    s = s + pa[..., a] * pb[..., b]
                outs.append(s)
            return torch.stack(outs, dim=-1)

        c0 = conv_neg(p1, p2)
        c1 = conv_neg(p0, p2)
        c2 = conv_neg(p0, p1)
        return torch.stack([c0, c1, c2], dim=-2)

    @classmethod
    def _parity_peer_messages(cls, probs: torch.Tensor, is_empty: torch.Tensor) -> torch.Tensor:
        bsz = probs.shape[0]
        p_grid = probs.view(bsz, 6, 6, 4)
        # 1. Row 3-cell triad convolution: [B, 6, 2, 3, 4] -> [B, 36, 4]
        r_triads = p_grid.view(bsz, 6, 2, 3, 4)
        r_conv = cls._triad_zero_sum_conv(r_triads).reshape(bsz, 36, 4)

        # 2. Column 3-cell triad convolution: transpose (row, col), apply, transpose back
        c_triads = p_grid.permute(0, 2, 1, 3).contiguous().view(bsz, 6, 2, 3, 4)
        c_conv = cls._triad_zero_sum_conv(c_triads).view(bsz, 6, 6, 4).permute(0, 2, 1, 3).reshape(bsz, 36, 4)

        # 3. Periodic quadrant peer average across (r +/- 3, c +/- 3)
        q_grid = p_grid.view(bsz, 2, 3, 2, 3, 4)
        q_sum = q_grid.sum(dim=(1, 3), keepdim=True).expand_as(q_grid).reshape(bsz, 36, 4)
        q_peer = (q_sum - probs) / 3.0

        emp_f = is_empty.float().view(bsz, 6, 6)
        r_emp = (emp_f.sum(dim=2, keepdim=True).expand(bsz, 6, 6).reshape(bsz, 36) - emp_f.reshape(bsz, 36)) / 5.0
        c_emp = (emp_f.sum(dim=1, keepdim=True).expand(bsz, 6, 6).reshape(bsz, 36) - emp_f.reshape(bsz, 36)) / 5.0
        return torch.cat(
            [probs, r_conv, c_conv, q_peer, r_emp.unsqueeze(-1), c_emp.unsqueeze(-1)],
            dim=-1,
        )  # [B, 36, 18]

    def forward(self, x_cond: torch.Tensor, n_steps: int = 4) -> List[torch.Tensor]:
        bsz = x_cond.shape[0]
        pos = self.pos_emb(self.pos_idx).unsqueeze(0)
        v_emb = self.val_emb(x_cond.clamp(0, 4)) + pos
        is_cond = (x_cond > 0).unsqueeze(-1).float()
        cond_oh = F.one_hot((x_cond - 1).clamp(0, 3), num_classes=4).float() * is_cond
        probs = cond_oh + (1.0 - is_cond) * 0.25
        h = infiloop_rms_norm(v_emb, 1e-6)

        step_logits: List[torch.Tensor] = []
        for _ in range(n_steps):
            m_in = self._parity_peer_messages(probs, x_cond == 0)
            h_in = infiloop_rms_norm(h + v_emb + self.msg_proj(m_in), 1e-6)
            h_grid = h_in.view(bsz, 6, 6, self.hidden_size)
            r_ctx = (h_grid.sum(dim=2, keepdim=True).expand_as(h_grid) - h_grid).reshape(bsz, 36, self.hidden_size) / 5.0
            c_ctx = (h_grid.sum(dim=1, keepdim=True).expand_as(h_grid) - h_grid).reshape(bsz, 36, self.hidden_size) / 5.0
            r_msg = self.row_mlp(torch.cat([h_in, r_ctx], dim=-1))
            c_msg = self.col_mlp(torch.cat([h_in, c_ctx], dim=-1))
            h = infiloop_rms_norm(h + 0.5 * self.cell_mlp(torch.cat([h_in, r_msg, c_msg], dim=-1)), 1e-6)
            logits = 2.0 * (self.head(h) + self.peer_logit_proj(m_in))
            step_logits.append(logits)
            probs = cond_oh + (1.0 - is_cond) * F.softmax(logits.detach(), dim=-1)
        return step_logits


def parity_grid_modular_consistency_loss(logits: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
    """
    Label-Free Modular Triad & Quadrant Consistency Loss on the 6x6 Z_4 grid:
    Enforces that each cell's predictive distribution agrees with the circular convolution
    of its row-triad peers, column-triad peers, and periodic quadrant peers without using any labels.
    """
    is_cond = (x_cond > 0).unsqueeze(-1).float()
    cond_oh = F.one_hot((x_cond - 1).clamp(0, 3), num_classes=4).float() * is_cond
    probs = cond_oh + (1.0 - is_cond) * F.softmax(logits, dim=-1)  # [B, 36, 4]
    p_det = probs.detach()
    m_in = AnyOrderParityGridNet._parity_peer_messages(p_det, x_cond == 0)
    r_conv = m_in[:, :, 4:8]
    c_conv = m_in[:, :, 8:12]
    q_peer = m_in[:, :, 12:16]

    loss_triad = ((probs - r_conv) ** 2).mean() + ((probs - c_conv) ** 2).mean() + 0.5 * ((probs - q_peer) ** 2).mean()
    cond_mask = x_cond > 0
    clue_ce = F.cross_entropy(logits[cond_mask], (x_cond[cond_mask] - 1).clamp(0, 3)) if cond_mask.any() else 0.0
    return 2.0 * loss_triad + 0.5 * clue_ce


@torch.no_grad()
def predict_parity_grid_anyorder_search(
    model: AnyOrderParityGridNet,
    x_cond: torch.Tensor,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    leaf_judge_model: Optional[AnyOrderParityGridNet] = None,
) -> torch.Tensor:
    """
    Computes 1-pass or B-branch search-amplified distribution on the 6x6 Z_4 Parity Grid.
    When leaf_judge_model (frozen q_real) is provided, model (q_amp) proposes the branches
    and leaf_judge_model evaluates leaf modular consistency and anchors dense states.
    """
    model.eval()
    nb = x_cond.shape[0]
    device = x_cond.device
    is_cond = (x_cond > 0).unsqueeze(-1).float()
    cond_oh = F.one_hot((x_cond - 1).clamp(0, 3), num_classes=4).float() * is_cond

    base_logits = model(x_cond, n_steps=4)[-1]
    if leaf_judge_model is not None:
        leaf_judge_model.eval()
        real_logits = leaf_judge_model(x_cond, n_steps=4)[-1]
        emp_frac = (x_cond == 0).float().mean(dim=-1, keepdim=True).unsqueeze(-1)
        w_amp = (0.35 + 0.50 * emp_frac).clamp(0.35, 0.85)
        comb_logits = w_amp * base_logits + (1.0 - w_amp) * real_logits
        base_probs = F.softmax(comb_logits, dim=-1)
        # Multiplicative (log-space) Bayesian fusion with 1-step triad/quadrant peer convolution support:
        # uniform peer support [0.25, 0.25, 0.25, 0.25] is a no-op in log-space, while resolved peers sharpen!
        p_full = cond_oh + (1.0 - is_cond) * base_probs
        m_in = AnyOrderParityGridNet._parity_peer_messages(p_full, x_cond == 0)
        log_sup = (
            0.60 * torch.log(m_in[:, :, 4:8].clamp(min=1e-4))
            + 0.60 * torch.log(m_in[:, :, 8:12].clamp(min=1e-4))
            + 0.30 * torch.log(m_in[:, :, 12:16].clamp(min=1e-4))
        )
        base_probs = F.softmax(comb_logits + log_sup, dim=-1)
    else:
        base_probs = F.softmax(base_logits, dim=-1)

    eps = 1e-4
    if n_branches <= 0:
        out_p = (1.0 - eps) * base_probs + (eps / 4.0)
        return out_p / out_p.sum(dim=-1, keepdim=True)

    r_bits = {4: 2, 8: 3}.get(n_branches, 3)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)], dtype=torch.long, device=device
    )
    p_cl = base_probs.clamp(min=1e-7)
    base_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
    is_empty = x_cond == 0
    top2_vals, top2_idx = base_probs.topk(2, dim=-1)
    score = (
        base_ent
        + torch.where(is_empty, torch.zeros_like(base_ent), torch.full_like(base_ent, 1e6))
        + torch.where(top2_vals[:, :, 0] > 0.92, torch.full_like(base_ent, 10.0), torch.zeros_like(base_ent))
    )
    root_cells = score.topk(r_bits, dim=-1, largest=False).indices
    root_d = top2_idx.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)) + 1
    root_p = top2_vals.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)).clamp(min=1e-5)

    x_branch = x_cond.unsqueeze(1).expand(nb, n_branches, 36).clone()
    log_prior = torch.zeros(nb, n_branches, device=device)
    for bit_i in range(r_bits):
        choice_bit = bit_table[:, bit_i].unsqueeze(0).expand(nb, n_branches)
        cell_i = root_cells[:, bit_i].unsqueeze(1).expand(nb, n_branches)
        cand_d = root_d[:, bit_i, :].gather(1, choice_bit)
        cand_p = root_p[:, bit_i, :].gather(1, choice_bit)
        was_emp = is_empty.gather(1, root_cells[:, bit_i : bit_i + 1]).expand(nb, n_branches)
        cur_val = x_branch.gather(2, cell_i.unsqueeze(-1)).squeeze(-1)
        x_branch.scatter_(2, cell_i.unsqueeze(-1), torch.where(was_emp, cand_d, cur_val).unsqueeze(-1))
        log_prior = log_prior + torch.where(was_emp, torch.log(cand_p), torch.zeros_like(cand_p))

    judge_net = leaf_judge_model if leaf_judge_model is not None else model
    flat_x = x_branch.reshape(nb * n_branches, 36)
    leaf_logits = judge_net(flat_x, n_steps=4)[-1]
    if leaf_judge_model is not None:
        amp_leaf_logits = model(flat_x, n_steps=4)[-1]
        leaf_logits = 0.50 * leaf_logits + 0.50 * amp_leaf_logits
    leaf_probs = F.softmax(leaf_logits, dim=-1).view(nb, n_branches, 36, 4)

    # Pure-neural triad inconsistency energy on leaf branches
    is_c_flat = (flat_x > 0).unsqueeze(-1).float()
    c_oh_flat = F.one_hot((flat_x - 1).clamp(0, 3), num_classes=4).float() * is_c_flat
    p_flat = c_oh_flat + (1.0 - is_c_flat) * leaf_probs.reshape(nb * n_branches, 36, 4)
    m_flat = AnyOrderParityGridNet._parity_peer_messages(p_flat, flat_x == 0)
    triad_err = ((p_flat - m_flat[:, :, 4:8]) ** 2 + (p_flat - m_flat[:, :, 8:12]) ** 2).sum(dim=(1, 2)).view(nb, n_branches)
    uncond_b = (x_branch == 0).float()
    deplete = ((-torch.log(leaf_probs.max(dim=-1).values.clamp(min=1e-6)) * uncond_b).sum(dim=-1)) / uncond_b.sum(dim=-1).clamp(min=1.0)
    e_inc = triad_err + 2.0 * deplete

    e_shift = e_inc - e_inc.min(dim=-1, keepdim=True).values
    weights = F.softmax(0.50 * log_prior - beta_clear * e_shift, dim=-1)
    added_mask = ((x_branch > 0) & (x_cond.unsqueeze(1) == 0)).unsqueeze(-1).float()
    branch_pred = added_mask * base_probs.unsqueeze(1) + (1.0 - added_mask) * leaf_probs
    cleared_probs = (weights.unsqueeze(-1).unsqueeze(-1) * branch_pred).sum(dim=1)
    aug_probs = 0.65 * cleared_probs + 0.35 * base_probs
    aug_probs = (1.0 - eps) * aug_probs + (eps / 4.0)
    return aug_probs / aug_probs.sum(dim=-1, keepdim=True)


def compute_parity_grid_twoparam_distill_and_zerocurl_loss(
    q_amp: AnyOrderParityGridNet,
    q_real_frozen: AnyOrderParityGridNet,
    x_s: torch.Tensor,
    logits_amp_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-Parameter Split Distillation & Teacher-Anchored Zero-Curl Loss on the 6x6 Parity Grid:
    Holds q_real_frozen (theta_real) strictly frozen and trains q_amp (theta_amp) via
    search-amplified teacher distillation + one-way teacher-anchored 2-step commutator consistency.
    """
    bsz = x_s.shape[0]
    device = x_s.device
    q_real_frozen.eval()

    log_q_amp_s = F.log_softmax(logits_amp_s, dim=-1)  # [B, 36, 4]
    q_amp_s_det = F.softmax(logits_amp_s.detach(), dim=-1)

    with torch.no_grad():
        teacher_aug_p = predict_parity_grid_anyorder_search(
            q_real_frozen, x_s, n_branches=4, beta_clear=2.5, leaf_judge_model=q_real_frozen
        )
    is_empty = x_s == 0
    emp_f = is_empty.unsqueeze(-1).float()
    distill_kl = (
        (teacher_aug_p * (torch.log(teacher_aug_p.clamp(min=1e-7)) - log_q_amp_s) * emp_f).sum()
        / emp_f.sum().clamp(min=1.0)
    )

    noise = torch.rand_like(q_amp_s_det[:, :, 0])
    pick_score = noise + torch.where(is_empty, torch.zeros_like(noise), torch.full_like(noise, 1e6))
    top2 = pick_score.topk(2, dim=-1, largest=False).indices
    cell_i, cell_j = top2[:, 0], top2[:, 1]

    batch_idx = torch.arange(bsz, device=device)
    act_a = teacher_aug_p[batch_idx, cell_i].argmax(dim=-1)
    act_b = teacher_aug_p[batch_idx, cell_j].argmax(dim=-1)

    x_si = x_s.clone()
    x_sj = x_s.clone()
    i_emp = is_empty[batch_idx, cell_i]
    j_emp = is_empty[batch_idx, cell_j]
    x_si[batch_idx, cell_i] = torch.where(i_emp, act_a + 1, x_si[batch_idx, cell_i])
    x_sj[batch_idx, cell_j] = torch.where(j_emp, act_b + 1, x_sj[batch_idx, cell_j])

    x_cat = torch.cat([x_si, x_sj], dim=0)
    with torch.no_grad():
        lg_real_cat = q_real_frozen(x_cat, n_steps=4)[-1]
        lq_real_si = F.log_softmax(lg_real_cat[:bsz], dim=-1)
        lq_real_sj = F.log_softmax(lg_real_cat[bsz:], dim=-1)

    lg_amp_cat = q_amp(x_cat, n_steps=4)[-1]
    lq_amp_si = F.log_softmax(lg_amp_cat[:bsz], dim=-1)
    lq_amp_sj = F.log_softmax(lg_amp_cat[bsz:], dim=-1)

    log_qi_a = log_q_amp_s[batch_idx, cell_i, act_a]
    log_qj_b = log_q_amp_s[batch_idx, cell_j, act_b]
    target_step_diff = 0.5 * (
        (lq_real_sj[batch_idx, cell_i, act_a] - lq_real_si[batch_idx, cell_j, act_b])
        + (lq_amp_sj[batch_idx, cell_i, act_a].detach() - lq_amp_si[batch_idx, cell_j, act_b].detach())
    )
    valid = (i_emp & j_emp).float()
    curl_diff = ((log_qi_a - log_qj_b) - target_step_diff).clamp(-5.0, 5.0)
    curl_loss = ((curl_diff ** 2) * valid).sum() / valid.sum().clamp(min=1.0)
    mean_abs_curl = (curl_diff.detach().abs() * valid).sum() / valid.sum().clamp(min=1.0)

    mod_loss = parity_grid_modular_consistency_loss(logits_amp_s, x_s)
    return 0.85 * distill_kl + 0.45 * curl_loss + 0.35 * mod_loss, mean_abs_curl


@torch.no_grad()
def evaluate_parity_grid_arm(
    arm_name: str,
    model: Optional[AnyOrderParityGridNet],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    device: torch.device,
    n_branches: int = 0,
    k_random_orders: int = 4,
    n_reveal_blocks: int = 6,
    seed: int = 101,
    leaf_judge_model: Optional[AnyOrderParityGridNet] = None,
) -> Dict[str, Any]:
    nb = x_eval_np.shape[0]
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)

    def get_probs(x_cur: torch.Tensor) -> torch.Tensor:
        if arm_name == "Control_ClueCopyFloor":
            return torch.full((x_cur.shape[0], 36, 4), 0.25, device=device)
        assert model is not None
        return predict_parity_grid_anyorder_search(
            model, x_cur, n_branches=n_branches, beta_clear=2.5, leaf_judge_model=leaf_judge_model
        )

    p0 = get_probs(x_init)
    empty_init_mask = x_eval_np == 0
    preds0 = (p0.argmax(dim=-1) + 1).cpu().numpy()
    one_shot_acc = float(((preds0 == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))

    board_order_total_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    board_order_cell_nll = np.zeros((nb, k_random_orders), dtype=np.float64)
    early_step_nlls: List[float] = []
    late_step_nlls: List[float] = []

    for k in range(k_random_orders):
        order_blocks = [torch.zeros((nb, 36), dtype=torch.bool, device=device) for _ in range(n_reveal_blocks)]
        empty_counts = np.zeros(nb, dtype=np.float64)
        for i in range(nb):
            rng_i = np.random.RandomState(seed * 1000 + i * 29 + k)
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
            probs_step = get_probs(x_cur)
            true_d = (y_true - 1).clamp(0, 3)
            true_p = probs_step.gather(2, true_d.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            step_nll = -torch.log(true_p).to(torch.float64) * b_mask.to(torch.float64)
            total_nll_k = total_nll_k + step_nll.sum(dim=-1)
            blk_mean = float((step_nll.sum() / b_mask.to(torch.float64).sum().clamp(min=1.0)).item())
            if b_idx == 0:
                early_step_nlls.append(blk_mean)
            if b_idx == n_reveal_blocks - 1:
                late_step_nlls.append(blk_mean)
            x_cur = torch.where(b_mask, y_true, x_cur)

        tot_np = total_nll_k.cpu().numpy()
        board_order_total_nll[:, k] = tot_np
        board_order_cell_nll[:, k] = tot_np / empty_counts

    random_order_joint_nll = float(board_order_total_nll.mean())
    rand_nll = float(board_order_cell_nll.mean())
    rand_ppl = float(math.exp(min(rand_nll, 20.0)))
    cross_std = float(np.std(board_order_total_nll, axis=1, ddof=1).mean())

    # Adaptive Min-Entropy Order NLL
    x_cur_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_scored_cells = torch.zeros(nb, dtype=torch.float64, device=device)
    for b_idx in range(n_reveal_blocks):
        rem_mask = x_cur_minh == 0
        if not rem_mask.any():
            break
        probs_step = get_probs(x_cur_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        rem_counts = rem_mask.sum(dim=-1)
        blocks_left = n_reveal_blocks - b_idx
        step_mask = torch.zeros_like(rem_mask)
        for i in range(nb):
            r_cnt = int(rem_counts[i].item())
            if r_cnt == 0:
                continue
            k_pick = max(1, math.ceil(r_cnt / float(blocks_left))) if blocks_left > 1 else r_cnt
            ent_i = ent[i].clone()
            ent_i[~rem_mask[i]] = 1e9
            pick_idx = ent_i.topk(k_pick, largest=False).indices
            step_mask[i, pick_idx] = True
        true_d = (y_true - 1).clamp(0, 3)
        true_p = probs_step.gather(2, true_d.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        step_nll = -torch.log(true_p).to(torch.float64) * step_mask.to(torch.float64)
        minh_total_nll = minh_total_nll + step_nll.sum(dim=-1)
        minh_scored_cells = minh_scored_cells + step_mask.to(torch.float64).sum(dim=-1)
        x_cur_minh = torch.where(step_mask, y_true, x_cur_minh)

    adaptive_order_joint_nll = float(minh_total_nll.mean().item())
    min_h_nll_per_cell = float((minh_total_nll / minh_scored_cells.clamp(min=1.0)).mean().item())

    # 2-Step Commutator Curl MAE on held-out dev boards (averaged across 3 random cell pairs per board)
    p0_base = get_probs(x_init) if n_branches == 0 else predict_parity_grid_anyorder_search(
        model, x_init, n_branches=0, leaf_judge_model=leaf_judge_model
    ) if model is not None else p0
    log_q_s = torch.log(p0_base.clamp(min=1e-6))
    rng = np.random.RandomState(seed + 555)
    abs_curls: List[float] = []
    for _ in range(3):
        cell_i = torch.zeros(nb, dtype=torch.long, device=device)
        cell_j = torch.zeros(nb, dtype=torch.long, device=device)
        for b in range(nb):
            emp = np.where(x_eval_np[b] == 0)[0]
            chosen = rng.choice(emp, size=2, replace=False)
            cell_i[b], cell_j[b] = int(chosen[0]), int(chosen[1])
        b_idx = torch.arange(nb, device=device)
        act_a = p0_base[b_idx, cell_i].argmax(dim=-1)
        act_b = p0_base[b_idx, cell_j].argmax(dim=-1)
        x_si, x_sj = x_init.clone(), x_init.clone()
        x_si[b_idx, cell_i] = act_a + 1
        x_sj[b_idx, cell_j] = act_b + 1
        x_cat = torch.cat([x_si, x_sj], dim=0)
        p_cat = (
            torch.full((x_cat.shape[0], 36, 4), 0.25, device=device)
            if model is None
            else predict_parity_grid_anyorder_search(model, x_cat, n_branches=0, leaf_judge_model=leaf_judge_model)
        )
        log_q_si = torch.log(p_cat[:nb].clamp(min=1e-6))
        log_q_sj = torch.log(p_cat[nb:].clamp(min=1e-6))
        curl_diff = (log_q_s[b_idx, cell_i, act_a] + log_q_si[b_idx, cell_j, act_b]) - (
            log_q_s[b_idx, cell_j, act_b] + log_q_sj[b_idx, cell_i, act_a]
        )
        abs_curls.extend(curl_diff.abs().cpu().numpy().tolist())
    curl_mae = float(np.mean(abs_curls))

    return {
        "n_branches": n_branches,
        "random_order_joint_nll": random_order_joint_nll,
        "rand_order_nll_per_cell": rand_nll,
        "rand_order_ppl": rand_ppl,
        "adaptive_order_joint_nll": adaptive_order_joint_nll,
        "min_h_nll_per_cell": min_h_nll_per_cell,
        "cross_order_nll_std": cross_std,
        "commutator_curl_mae": curl_mae,
        "early_step_nll_per_cell": float(np.mean(early_step_nlls)),
        "late_step_nll_per_cell": float(np.mean(late_step_nlls)),
        "one_shot_empty_acc": one_shot_acc,
    }


# ---------------------------------------------------------------------------
# 3. Stiff-CoinFlip-Coupled-Grid (Synthetic 24-Variable Z_4 Environment)
#    Testing Stiffness vs. Entropy & Unified Composition Law
# ---------------------------------------------------------------------------
# Board layout of 24 variables in Z_4 (values 1..4 when conditioned, 0 when unconditioned):
#   - Indices 0..2   (R=3): Stiff Root Coin-Flip variables Z_0, Z_1, Z_2 ~ Uniform({0,1,2,3})
#                           (genuine aleatoric coin flips: H = log(4) = 1.3863 nats, high VoI ~ 3.8 nats)
#   - Indices 3..14  (K=12): Coupled Downstream Block variables X_0..X_11 (4 per root r=0,1,2):
#                            X_{4r+0} = (z_r + offset_{4r+0}) mod 4
#                            X_{4r+1} = (z_r + offset_{4r+1}) mod 4
#                            X_{4r+2} = (z_r + offset_{4r+2}) mod 4
#                            X_{4r+3} = (z_r + z_{(r+1)%3} + offset_{4r+3}) mod 4
#                            with per-cell observation fidelity 1 - eps_couple (eps_couple = 0.03).
#   - Indices 15..17 (F=3): Forced Deterministic cells F_0..F_2 (peaked prior p_mode = 0.985, H ~ 0.095 nats, VoI = 0)
#   - Indices 18..23 (W=6): 3 High-Entropy Weakly-Coupled Distractor Pairs (18,19), (20,21), (22,23):
#                            W_{2m} ~ Uniform({0,1,2,3}) (H = 1.3863 nats), W_{2m+1} = (w_m + offset) mod 4 with
#                            weak coupling p_match = 0.73 (so VoI(W_j) ~ 0.39 nats > 0, but H(W_j) - VoI(W_j) = +0.99 >> H(F_m)).


def generate_stiff_coinflip_coupled_dataset(
    n_boards: int, seed: int = 3030
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      context_offsets: [n_boards, 24] integer offsets in 0..3 known as board context features
      y_true:          [n_boards, 24] realized ground-truth values in 1..4
    """
    rng = np.random.RandomState(seed)
    offsets = rng.randint(0, 4, size=(n_boards, 24), dtype=np.int64)
    ys = np.zeros((n_boards, 24), dtype=np.int64)

    for b in range(n_boards):
        off = offsets[b]
        # 1. Sample R=3 genuine aleatoric root coin flips z_0, z_1, z_2 ~ Uniform(0..3)
        z = rng.randint(0, 4, size=3)
        for r in range(3):
            ys[b, r] = ((int(z[r]) + int(off[r])) % 4) + 1

        # 2. Sample K=12 coupled downstream block variables X_0..X_11
        for r in range(3):
            z_r = int(z[r])
            z_next = int(z[(r + 1) % 3])
            det_vals = [
                (z_r + int(off[3 + 4 * r + 0])) % 4,
                (z_r + int(off[3 + 4 * r + 1])) % 4,
                (z_r + int(off[3 + 4 * r + 2])) % 4,
                (z_r + z_next + int(off[3 + 4 * r + 3])) % 4,
            ]
            for k_i, dv in enumerate(det_vals):
                if rng.rand() < 0.97:
                    val = dv
                else:
                    val = int(rng.randint(0, 4))
                ys[b, 3 + 4 * r + k_i] = val + 1

        # 3. Sample F=3 Forced cells (p_mode = 0.985 on offset[idx])
        for idx in range(15, 18):
            probs = np.full(4, 0.005, dtype=np.float64)
            probs[int(off[idx])] = 0.985
            ys[b, idx] = int(rng.choice(4, p=probs)) + 1

        # 4. Sample W=6 High-Entropy Weakly-Coupled Distractor Pairs (18,19), (20,21), (22,23)
        for m in range(3):
            idx0 = 18 + 2 * m
            idx1 = 18 + 2 * m + 1
            wm = int(rng.randint(0, 4))
            ys[b, idx0] = ((wm + int(off[idx0])) % 4) + 1
            target_w1 = (wm + int(off[idx1])) % 4
            probs_w1 = np.full(4, 0.09, dtype=np.float64)
            probs_w1[target_w1] = 0.73
            ys[b, idx1] = int(rng.choice(4, p=probs_w1)) + 1

    return offsets, ys


_Z64_GRID = np.array(
    [[z0, z1, z2] for z0 in range(4) for z1 in range(4) for z2 in range(4)], dtype=np.int64
)  # [64, 3]


def exact_stiff_coinflip_oracle_posterior_and_voi(
    x_cond_row: np.ndarray, offset_row: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the exact Bayesian marginal posteriors P^*(X_v | x_{S_t}) [24, 4]
    and exact one-step Mutual Information VoI^*(v | x_{S_t}) = sum_{w notin S_t u {v}} I^*(X_v; X_w | x_{S_t}) [24]
    by exact enumeration over the 4^3 = 64 root configurations (z_0, z_1, z_2) and the W-pairs (4^2 = 16 each).
    """
    p_match = 0.9775
    p_other = 0.0075

    # Precompute deterministic target table [64, 15] for indices 0..14 given canonical (z0, z1, z2)
    det_table = np.zeros((64, 15), dtype=np.int64)
    det_table[:, 0] = (_Z64_GRID[:, 0] + int(offset_row[0])) % 4
    det_table[:, 1] = (_Z64_GRID[:, 1] + int(offset_row[1])) % 4
    det_table[:, 2] = (_Z64_GRID[:, 2] + int(offset_row[2])) % 4
    for r in range(3):
        zr = _Z64_GRID[:, r]
        zn = _Z64_GRID[:, (r + 1) % 3]
        det_table[:, 3 + 4 * r + 0] = (zr + int(offset_row[3 + 4 * r + 0])) % 4
        det_table[:, 3 + 4 * r + 1] = (zr + int(offset_row[3 + 4 * r + 1])) % 4
        det_table[:, 3 + 4 * r + 2] = (zr + int(offset_row[3 + 4 * r + 2])) % 4
        det_table[:, 3 + 4 * r + 3] = (zr + zn + int(offset_row[3 + 4 * r + 3])) % 4

    # Precompute [64, 15, 4] emission table: P(X_idx = d | z_state)
    emit_15 = np.full((64, 15, 4), p_other, dtype=np.float64)
    s_idx = np.arange(64)[:, None]
    v_idx = np.arange(15)[None, :]
    emit_15[s_idx, v_idx, det_table] = p_match
    for r in range(3):
        emit_15[:, r, :] = 0.0
        emit_15[np.arange(64), r, det_table[:, r]] = 1.0

    def compute_block15(x_c: np.ndarray) -> np.ndarray:
        w64 = np.ones(64, dtype=np.float64)
        cond_indices = np.where(x_c[:15] > 0)[0]
        for idx in cond_indices:
            w64 *= emit_15[:, idx, int(x_c[idx]) - 1]
        w_sum = w64.sum()
        if w_sum <= 1e-30:
            w64 = np.full(64, 1.0 / 64.0, dtype=np.float64)
        else:
            w64 = w64 / w_sum
        m15 = np.einsum("s,svd->vd", w64, emit_15)
        for idx in cond_indices:
            m15[idx, :] = 0.0
            m15[idx, int(x_c[idx]) - 1] = 1.0
        return np.clip(m15, 1e-12, 1.0)

    marg = np.zeros((24, 4), dtype=np.float64)
    marg[:15] = compute_block15(x_cond_row)

    # 2. Forced cells 15..17
    for idx in range(15, 18):
        if x_cond_row[idx] > 0:
            marg[idx, int(x_cond_row[idx]) - 1] = 1.0
        else:
            marg[idx, :] = 0.005
            marg[idx, int(offset_row[idx])] = 0.985

    # 3. Weakly-coupled pairs (18,19), (20,21), (22,23)
    w_joints = []
    for m in range(3):
        idx0 = 18 + 2 * m
        idx1 = 18 + 2 * m + 1
        w_joint = np.full((4, 4), 0.25 * 0.09, dtype=np.float64)
        for d0 in range(4):
            t1 = ((d0 - int(offset_row[idx0])) + int(offset_row[idx1])) % 4
            w_joint[d0, t1] = 0.25 * 0.73
        w_joints.append(w_joint.copy())
        wj = w_joint.copy()
        if x_cond_row[idx0] > 0:
            obs0 = int(x_cond_row[idx0]) - 1
            mask0 = np.arange(4) != obs0
            wj[mask0, :] = 0.0
        if x_cond_row[idx1] > 0:
            obs1 = int(x_cond_row[idx1]) - 1
            mask1 = np.arange(4) != obs1
            wj[:, mask1] = 0.0
        wj /= max(wj.sum(), 1e-30)
        marg[idx0, :] = wj.sum(axis=1)
        marg[idx1, :] = wj.sum(axis=0)

    base_marg = np.clip(marg, 1e-12, 1.0)
    base_ent = -(base_marg * np.log(base_marg)).sum(axis=-1)  # [24]
    is_emp = x_cond_row == 0

    voi = np.zeros(24, dtype=np.float64)
    emp_15 = is_emp[:15].astype(np.float64)
    for v in range(15):
        if not is_emp[v]:
            continue
        exp_ent15 = np.zeros(15, dtype=np.float64)
        for d in range(4):
            p_vd = base_marg[v, d]
            if p_vd < 1e-6:
                continue
            x_next = x_cond_row.copy()
            x_next[v] = d + 1
            m15_next = compute_block15(x_next)
            exp_ent15 += p_vd * (-(m15_next * np.log(m15_next)).sum(axis=-1))
        mi_vw = np.maximum(0.0, base_ent[:15] - exp_ent15) * emp_15
        mi_vw[v] = 0.0
        voi[v] = float(mi_vw.sum())

    # Forced cells 15..17 are independent (voi = 0.0)
    # Weakly-coupled pairs (18,19), (20,21), (22,23) only couple within each pair
    for m in range(3):
        idx0 = 18 + 2 * m
        idx1 = 18 + 2 * m + 1
        if is_emp[idx0] and is_emp[idx1]:
            # Exact mutual information I(W_{2m}; W_{2m+1})
            wj = np.clip(w_joints[m], 1e-12, 1.0)
            h_joint = float(-(wj * np.log(wj)).sum())
            mi_pair = max(0.0, float(base_ent[idx0] + base_ent[idx1] - h_joint))
            voi[idx0] = mi_pair
            voi[idx1] = mi_pair

    return base_marg, voi


def verify_stiff_coinflip_oracle_against_bruteforce(
    offsets: np.ndarray, ys: np.ndarray, n_episodes: int = 24
) -> Dict[str, Any]:
    """
    Rule 9.6 verification: checks exact_stiff_coinflip_oracle_posterior_and_voi against
    full brute-force enumeration over the joint configuration space on 24 episodes.
    """
    max_abs_diff = 0.0
    rng = np.random.RandomState(906)
    for ep in range(min(n_episodes, len(offsets))):
        x_c = np.zeros(24, dtype=np.int64)
        rev_idx = rng.choice(24, size=5, replace=False)
        x_c[rev_idx] = ys[ep, rev_idx]
        fast_marg, _ = exact_stiff_coinflip_oracle_posterior_and_voi(x_c, offsets[ep])

        # Independent brute-force enumeration of all 4^3 = 64 root assignments by explicit loop
        bf_num = np.zeros((15, 4), dtype=np.float64)
        bf_den = 0.0
        for z_flat in range(64):
            z0 = z_flat & 3
            z1 = (z_flat >> 2) & 3
            z2 = (z_flat >> 4) & 3
            zs = [z0, z1, z2]
            dets = [
                (z0 + int(offsets[ep, 0])) % 4,
                (z1 + int(offsets[ep, 1])) % 4,
                (z2 + int(offsets[ep, 2])) % 4,
            ]
            ok = True
            for r_i in range(3):
                if x_c[r_i] > 0 and (x_c[r_i] - 1) != dets[r_i]:
                    ok = False
            if not ok:
                continue
            w = 1.0 / 64.0
            for r in range(3):
                zr, zn = zs[r], zs[(r + 1) % 3]
                dets.extend([
                    (zr + int(offsets[ep, 3 + 4 * r + 0])) % 4,
                    (zr + int(offsets[ep, 3 + 4 * r + 1])) % 4,
                    (zr + int(offsets[ep, 3 + 4 * r + 2])) % 4,
                    (zr + zn + int(offsets[ep, 3 + 4 * r + 3])) % 4,
                ])
            for idx in range(3, 15):
                if x_c[idx] > 0:
                    w *= 0.9775 if (x_c[idx] - 1) == dets[idx] else 0.0075
            bf_den += w
            for idx in range(3):
                bf_num[idx, dets[idx]] += w
            for idx in range(3, 15):
                if x_c[idx] > 0:
                    bf_num[idx, x_c[idx] - 1] += w
                else:
                    for d in range(4):
                        bf_num[idx, d] += w * (0.9775 if d == dets[idx] else 0.0075)
        bf_marg = np.clip(bf_num / max(bf_den, 1e-30), 1e-12, 1.0)
        diff = float(np.max(np.abs(fast_marg[:15] - bf_marg)))
        if diff > max_abs_diff:
            max_abs_diff = diff

    return {
        "episodes_checked": int(min(n_episodes, len(offsets))),
        "max_abs_diff": max_abs_diff,
        "verified": bool(max_abs_diff < 1e-9),
    }


class AnyOrderStiffCoupledNet(nn.Module):
    """
    Pure-Neural Any-Order Conditional World Model q_theta(X_v | x_{S_t}, offsets)
    on the 24-variable Z_4 Stiff-CoinFlip-Coupled-Grid.
    Learns the coupling structure directly from masked training boards; all weights initialized
    from N(0, 0.02^2).
    """

    def __init__(self, hidden_size: int = 64, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size
        self.can_prior = nn.Parameter(torch.randn(24, 4, device=device) * 0.02)
        self.var_couple = nn.Parameter(torch.randn(24, 24, device=device) * 0.02)
        self.conv_sum_couple = nn.Parameter(torch.randn(24, 24, device=device) * 0.02)
        self.conv_diff_couple = nn.Parameter(torch.randn(24, 24, device=device) * 0.02)
        self.cell_mlp = nn.Sequential(
            nn.Linear(12, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 4),
        )
        for mod in self.cell_mlp:
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, mean=0.0, std=0.02)
                nn.init.zeros_(mod.bias)
        self.register_buffer("d_idx", torch.arange(4, dtype=torch.long, device=device))

    def forward(self, x_cond: torch.Tensor, offsets: torch.Tensor, n_steps: int = 4) -> List[torch.Tensor]:
        bsz = x_cond.shape[0]
        d_grid = self.d_idx.view(1, 1, 4)
        shift_idx = (d_grid + offsets.unsqueeze(-1)) % 4  # [B, 24, 4]
        unshift_idx = (d_grid - offsets.unsqueeze(-1)) % 4  # [B, 24, 4]
        is_cond = (x_cond > 0).unsqueeze(-1).float()
        cond_oh = F.one_hot((x_cond - 1).clamp(0, 3), num_classes=4).float() * is_cond
        probs = cond_oh + (1.0 - is_cond) * 0.25
        eye_mask = 1.0 - torch.eye(24, device=x_cond.device)

        step_logits: List[torch.Tensor] = []
        for step_i in range(n_steps):
            p_shift = probs.gather(-1, shift_idx)  # [B, 24, 4]
            if step_i == 0:
                # Step 0 is the local unsearched pass (used when n_steps=1, B=0): no cross-variable message passing
                lg_can = self.can_prior.unsqueeze(0).expand(bsz, 24, 4)
            else:
                p_cent = p_shift - 0.25
                p_triples = p_shift.view(bsz, 8, 3, 4)
                p_next = torch.roll(p_triples, shifts=-1, dims=2)
                c_sum = (
                    torch.stack(
                        [sum(p_triples[:, :, :, a] * p_next[:, :, :, (d - a) % 4] for a in range(4)) for d in range(4)],
                        dim=-1,
                    ).reshape(bsz, 24, 4)
                    - 0.25
                )
                c_diff = (
                    torch.stack(
                        [sum(p_triples[:, :, :, a] * p_next[:, :, :, (d + a) % 4] for a in range(4)) for d in range(4)],
                        dim=-1,
                    ).reshape(bsz, 24, 4)
                    - 0.25
                )
                w_u = self.var_couple * eye_mask
                msg_u = torch.einsum("vu,bud->bvd", w_u, p_cent)
                msg_s = torch.einsum("vu,bud->bvd", self.conv_sum_couple, c_sum)
                msg_d = torch.einsum("vu,bud->bvd", self.conv_diff_couple, c_diff)
                mlp_in = torch.cat([p_cent, msg_u, msg_s + msg_d], dim=-1)
                lg_can = self.can_prior.unsqueeze(0) + msg_u + msg_s + msg_d + 0.30 * self.cell_mlp(mlp_in)
            logits = lg_can.gather(-1, unshift_idx)
            step_logits.append(logits)
            probs = cond_oh + (1.0 - is_cond) * F.softmax(logits.detach(), dim=-1)
        return step_logits


@torch.no_grad()
def predict_stiff_coupled_neural_search_and_voi(
    model: AnyOrderStiffCoupledNet,
    x_cond: torch.Tensor,
    offsets: torch.Tensor,
    n_branches: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the neural conditional distribution q_{theta, B}^{aug}(X_v | x_{S_t}) [nb, 24, 4]
    and the neural one-step expected downstream entropy reduction (VoI / Stiffness):
      VoI_theta(v | x_{S_t}) = sum_{w notin S_t u {v}} I_theta(X_v; X_w | x_{S_t}) [nb, 24].
    When n_branches == 0 (NoSearch_B0), uses the local unsearched pass (n_steps=1) with no multi-step
    counterfactual propagation; when n_branches == 8 (Compose_B8), uses deep recurrent refinement (n_steps=4)
    plus B=8 counterfactual branch rollouts cleared by pure-neural inconsistency energy.
    """
    model.eval()
    nb = x_cond.shape[0]
    device = x_cond.device
    base_steps = 1 if n_branches <= 0 else 4
    base_logits = model(x_cond, offsets, n_steps=base_steps)[-1]
    is_cond = (x_cond > 0).unsqueeze(-1).float()
    cond_oh = F.one_hot((x_cond - 1).clamp(0, 3), num_classes=4).float() * is_cond
    base_probs = cond_oh + (1.0 - is_cond) * F.softmax(base_logits, dim=-1)
    eps = 1e-4
    base_probs = ((1.0 - eps) * base_probs + (eps / 4.0))
    base_probs = base_probs / base_probs.sum(dim=-1, keepdim=True)

    p_cl = base_probs.clamp(min=1e-7)
    base_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)  # [nb, 24]
    is_empty = x_cond == 0  # [nb, 24]

    if n_branches <= 0:
        # At B=0 (no internal counterfactual search), no counterfactual rollouts are executed
        return base_probs, torch.zeros_like(base_ent)

    # Compute neural counterfactual entropy reduction across all 24 variables and 4 values d in 0..3
    x_cf = x_cond.unsqueeze(1).unsqueeze(2).expand(nb, 24, 4, 24).clone()
    v_idx = torch.arange(24, device=device).view(1, 24, 1, 1).expand(nb, 24, 4, 1)
    d_vals = (torch.arange(4, device=device).view(1, 1, 4, 1) + 1).expand(nb, 24, 4, 1)
    x_cf.scatter_(3, v_idx, d_vals)
    off_cf = offsets.unsqueeze(1).unsqueeze(2).expand(nb, 24, 4, 24).reshape(nb * 96, 24)
    cf_logits = model(x_cf.reshape(nb * 96, 24), off_cf, n_steps=4)[-1].view(nb, 24, 4, 24, 4)
    cf_is_cond = (x_cf > 0).unsqueeze(-1).float()
    cf_oh = F.one_hot((x_cf - 1).clamp(0, 3), num_classes=4).float() * cf_is_cond
    cf_probs = (cf_oh + (1.0 - cf_is_cond) * F.softmax(cf_logits, dim=-1)).clamp(min=1e-7)
    cf_ent = -(cf_probs * torch.log(cf_probs)).sum(dim=-1)  # [nb, 24, 4, 24]
    exp_post_ent = (base_probs.unsqueeze(-1) * cf_ent).sum(dim=2)  # [nb, 24, 24]

    # Filter small neural noise below 0.08 nats per pair so uncoupled variables have clean 0 mutual information
    mi_vw = F.relu(base_ent.unsqueeze(1) - exp_post_ent - 0.08) * is_empty.unsqueeze(1).float()
    eye_mask = torch.eye(24, device=device).unsqueeze(0)
    mi_vw = mi_vw * (1.0 - eye_mask)
    voi_scores = mi_vw.sum(dim=-1) * is_empty.float()  # [nb, 24]

    # Internal B=8 Search: branch on the top R=3 highest-stiffness uncertain empty variables
    r_bits = 3  # 2^3 = 8 branches
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(8)], dtype=torch.long, device=device
    )
    top2_p, top2_d = base_probs.topk(2, dim=-1)  # [nb, 24, 2]
    branch_rank = (
        voi_scores
        + 0.2 * base_ent
        + torch.where(is_empty, torch.zeros_like(base_ent), torch.full_like(base_ent, -1e6))
    )
    root_vars = branch_rank.topk(r_bits, dim=-1, largest=True).indices  # [nb, 3]
    root_d = top2_d.gather(1, root_vars.unsqueeze(-1).expand(nb, r_bits, 2)) + 1
    root_p = top2_p.gather(1, root_vars.unsqueeze(-1).expand(nb, r_bits, 2)).clamp(min=1e-5)

    x_branch = x_cond.unsqueeze(1).expand(nb, 8, 24).clone()
    log_prior = torch.zeros(nb, 8, device=device)
    for bit_i in range(r_bits):
        choice_bit = bit_table[:, bit_i].unsqueeze(0).expand(nb, 8)
        var_i = root_vars[:, bit_i].unsqueeze(1).expand(nb, 8)
        cand_d = root_d[:, bit_i, :].gather(1, choice_bit)
        cand_p = root_p[:, bit_i, :].gather(1, choice_bit)
        was_emp = is_empty.gather(1, root_vars[:, bit_i : bit_i + 1]).expand(nb, 8)
        cur_val = x_branch.gather(2, var_i.unsqueeze(-1)).squeeze(-1)
        x_branch.scatter_(2, var_i.unsqueeze(-1), torch.where(was_emp, cand_d, cur_val).unsqueeze(-1))
        log_prior = log_prior + torch.where(was_emp, torch.log(cand_p), torch.zeros_like(cand_p))

    leaf_logits = model(
        x_branch.reshape(nb * 8, 24),
        offsets.unsqueeze(1).expand(nb, 8, 24).reshape(nb * 8, 24),
        n_steps=4,
    )[-1].view(nb, 8, 24, 4)
    leaf_probs = F.softmax(leaf_logits, dim=-1).clamp(min=1e-6)

    # Pure-neural branch inconsistency energy: log-loss on already-conditioned variables + unconditioned depletion
    cond_mask_b = (x_cond > 0).unsqueeze(1).expand(nb, 8, 24)
    true_cond_d = (x_cond - 1).clamp(0, 3).unsqueeze(1).expand(nb, 8, 24)
    cond_p = leaf_probs.gather(3, true_cond_d.unsqueeze(-1)).squeeze(-1)
    cond_nll = (-torch.log(cond_p) * cond_mask_b.float()).sum(dim=-1)
    uncond_b = (x_branch == 0).float()
    deplete = ((-torch.log(leaf_probs.max(dim=-1).values) * uncond_b).sum(dim=-1)) / uncond_b.sum(dim=-1).clamp(min=1.0)
    e_inc = cond_nll + 1.5 * deplete

    e_shift = e_inc - e_inc.min(dim=-1, keepdim=True).values
    weights = F.softmax(0.5 * log_prior - 2.5 * e_shift, dim=-1)  # [nb, 8]

    # Gate branch sharpening by evidence spread across branches so unrevealed uniform coin flips stay calibrated
    e_spread = (e_inc.max(dim=-1).values - e_inc.min(dim=-1).values).unsqueeze(-1).unsqueeze(-1)
    alpha_ev = torch.tanh(1.5 * e_spread) * 0.50

    added_mask = ((x_branch > 0) & (x_cond.unsqueeze(1) == 0)).unsqueeze(-1).float()
    added_oh = F.one_hot((x_branch - 1).clamp(0, 3), num_classes=4).float()
    branch_pred = added_mask * (0.50 * added_oh + 0.50 * leaf_probs) + (1.0 - added_mask) * leaf_probs
    cleared_probs = (weights.unsqueeze(-1).unsqueeze(-1) * branch_pred).sum(dim=1)
    aug_probs = (1.0 - alpha_ev) * base_probs + alpha_ev * cleared_probs
    aug_probs = cond_oh + (1.0 - is_cond) * aug_probs
    aug_probs = ((1.0 - eps) * aug_probs + (eps / 4.0))
    aug_probs = aug_probs / aug_probs.sum(dim=-1, keepdim=True)
    return aug_probs, voi_scores


@torch.no_grad()
def evaluate_stiff_coinflip_ordering_arm(
    arm_name: str,
    order_policy: str,
    model: Optional[AnyOrderStiffCoupledNet],
    offsets_np: np.ndarray,
    y_eval_np: np.ndarray,
    device: torch.device,
    n_branches: int = 0,
    gamma_voi: float = 1.0,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    Evaluates an external reveal ordering / internal VoI search composition arm on Stiff-CoinFlip-Coupled-Grid:
      order_policy in {"random", "min_entropy", "pure_max_voi", "unified_voi_entropy"}
    Measures:
      1. post_q3_nll_per_cell, post_q3_ppl, post_q3_empty_acc (remaining board NLL & accuracy after Q=3 active reveals)
      2. post_q6_nll_per_cell, post_q6_empty_acc (remaining board NLL & accuracy after Q=6 active reveals)
      3. root_coinflip_recall_at_q3 (fraction of the R=3 stiff root coin-flip variables Z_0, Z_1, Z_2 revealed in the first 3 queries)
      4. forced_committed_before_noisy_w (fraction of the F=3 forced deterministic cells F_0..2 selected before the high-entropy noisy W_0)
      5. au_remaining_nll_curve (Area Under the Remaining-Board NLL Curve across steps t=0..12)
      6. core18_trajectory_nll_per_cell, core18_trajectory_ppl (Prefix trajectory NLL across the 18 solvable-core steps t=0..17)
      7. post_q3_autonomous_commit_acc (accuracy on the 15autonomous commitments made in steps t=3..17 following Q=3 oracle queries under order_policy!)
    """
    nb = offsets_np.shape[0]
    x_cur = torch.zeros((nb, 24), dtype=torch.long, device=device)
    offsets = torch.tensor(offsets_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)

    def get_probs_and_voi(x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if arm_name == "Control_UniformFloor":
            p = torch.full((nb, 24, 4), 0.25, device=device)
            v = torch.zeros((nb, 24), device=device)
            return p, v
        if arm_name == "Ceiling_Exact_Oracle_Unified_VoI_Entropy":
            x_np = x_in.cpu().numpy()
            p_list, v_list = [], []
            for b in range(nb):
                pm, vm = exact_stiff_coinflip_oracle_posterior_and_voi(x_np[b], offsets_np[b])
                p_list.append(pm)
                v_list.append(vm)
            return (
                torch.tensor(np.array(p_list), dtype=torch.float32, device=device),
                torch.tensor(np.array(v_list), dtype=torch.float32, device=device),
            )
        assert model is not None
        return predict_stiff_coupled_neural_search_and_voi(model, x_in, offsets, n_branches=n_branches)

    rng = np.random.RandomState(seed + 404)
    rng_auto = np.random.RandomState(seed + 808)
    traj_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    core18_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    rem_nll_curve: List[float] = []
    chosen_history = np.zeros((nb, 24), dtype=np.int64)

    post_q3_nll = 0.0
    post_q3_acc = 0.0
    post_q3_auto_acc = 0.0
    post_q6_nll = 0.0
    post_q6_acc = 0.0

    def select_next_var(rem_m: torch.Tensor, ent_m: torch.Tensor, voi_m: torch.Tensor, r_gen: np.random.RandomState) -> torch.Tensor:
        pick_out = torch.zeros(nb, dtype=torch.long, device=device)
        for b in range(nb):
            emp_b = torch.where(rem_m[b])[0]
            if len(emp_b) == 0:
                continue
            if order_policy == "random":
                chosen_idx = int(emp_b[r_gen.randint(len(emp_b))].item())
            elif order_policy == "min_entropy":
                scores = ent_m[b, emp_b]
                chosen_idx = int(emp_b[scores.argmin()].item())
            elif order_policy == "pure_max_voi":
                scores = -voi_m[b, emp_b] + torch.tensor(r_gen.uniform(0, 1e-3, size=len(emp_b)), device=device, dtype=voi_m.dtype)
                chosen_idx = int(emp_b[scores.argmin()].item())
            elif order_policy == "unified_voi_entropy":
                scores = ent_m[b, emp_b] - gamma_voi * voi_m[b, emp_b]
                chosen_idx = int(emp_b[scores.argmin()].item())
            else:
                raise ValueError(f"Unknown order_policy: {order_policy}")
            pick_out[b] = chosen_idx
        return pick_out

    for step_t in range(24):
        probs, voi = get_probs_and_voi(x_cur)
        rem_mask = x_cur == 0  # [nb, 24]
        p_cl = probs.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)  # [nb, 24]

        # Record remaining-board NLL & accuracy at steps t=0..12
        true_d = (y_true - 1).clamp(0, 3)
        true_p = probs.gather(2, true_d.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        cell_nll = -torch.log(true_p).to(torch.float64)
        rem_f = rem_mask.to(torch.float64)
        rem_cnt = rem_f.sum(dim=-1).clamp(min=1.0)
        mean_rem_nll = float(((cell_nll * rem_f).sum(dim=-1) / rem_cnt).mean().item())
        if step_t <= 12:
            rem_nll_curve.append(mean_rem_nll)

        if step_t == 3:
            post_q3_nll = mean_rem_nll
            preds_q3 = (probs.argmax(dim=-1) + 1)
            acc_q3 = ((preds_q3 == y_true) & rem_mask).float().sum(dim=-1) / rem_mask.float().sum(dim=-1).clamp(min=1.0)
            post_q3_acc = float(acc_q3.mean().item())
            # Autonomous commit rollout from Q=3 state for the 15 solvable-core commitments (steps 3..17)
            # following the arm's own order_policy (in 3 blocks of 5 variables):
            x_auto = x_cur.clone()
            committed_mask = torch.zeros((nb, 24), dtype=torch.bool, device=device)
            for _ in range(3):
                p_au, v_au = get_probs_and_voi(x_auto)
                rem_au = x_auto == 0
                if not rem_au.any():
                    break
                ent_au = -(p_au.clamp(min=1e-7) * torch.log(p_au.clamp(min=1e-7))).sum(dim=-1)
                for _ in range(5):
                    rem_au = x_auto == 0
                    if not rem_au.any():
                        break
                    p_pick = select_next_var(rem_au, ent_au, v_au, rng_auto)
                    b_i = torch.arange(nb, device=device)
                    pred_d = p_au[b_i, p_pick].argmax(dim=-1) + 1
                    x_auto[b_i, p_pick] = pred_d
                    committed_mask[b_i, p_pick] = True
                    ent_au[b_i, p_pick] = 1e6
                    v_au[b_i, p_pick] = -1e6
            auto_correct = ((x_auto == y_true) & committed_mask).float().sum(dim=-1) / committed_mask.float().sum(dim=-1).clamp(min=1.0)
            post_q3_auto_acc = float(auto_correct.mean().item())

        if step_t == 6:
            post_q6_nll = mean_rem_nll
            preds_q6 = (probs.argmax(dim=-1) + 1)
            acc_q6 = ((preds_q6 == y_true) & rem_mask).float().sum(dim=-1) / rem_mask.float().sum(dim=-1).clamp(min=1.0)
            post_q6_acc = float(acc_q6.mean().item())

        pick_v = select_next_var(rem_mask, ent, voi, rng)
        chosen_history[:, step_t] = pick_v.cpu().numpy()

        b_idx = torch.arange(nb, device=device)
        picked_nll = cell_nll[b_idx, pick_v]
        traj_total_nll = traj_total_nll + picked_nll
        if step_t < 18:
            core18_total_nll = core18_total_nll + picked_nll
        x_cur[b_idx, pick_v] = y_true[b_idx, pick_v]

    traj_nll_per_cell = float((core18_total_nll / 18.0).mean().item())
    traj_ppl = float(math.exp(min(traj_nll_per_cell, 20.0)))
    full24_nll_per_cell = float((traj_total_nll / 24.0).mean().item())

    # Diagnostic 1: fraction of R=3 stiff root coin-flip blocks r in {0, 1, 2} resolved in first 3 queries
    first3 = chosen_history[:, :3]
    stiff_blocks = [{r, 3 + 4 * r + 0, 3 + 4 * r + 1, 3 + 4 * r + 2} for r in range(3)]
    root_recall_q3 = float(
        np.mean([sum(1 for blk in stiff_blocks if len(set(first3[b].tolist()) & blk) > 0) / 3.0 for b in range(nb)])
    )

    # Diagnostic 2: fraction of F=3 forced deterministic cells (indices 15, 16, 17) chosen before noisy W cells (18..23)
    forced_before_w0 = []
    for b in range(nb):
        order_b = chosen_history[b].tolist()
        w_positions = [order_b.index(w_idx) for w_idx in range(18, 24) if w_idx in order_b]
        pos_w0 = min(w_positions) if w_positions else 24
        cnt_f = sum(1 for f_idx in (15, 16, 17) if (order_b.index(f_idx) if f_idx in order_b else 24) < pos_w0)
        forced_before_w0.append(cnt_f / 3.0)

    return {
        "order_policy": order_policy,
        "n_branches": n_branches,
        "gamma_voi": gamma_voi,
        "post_q3_nll_per_cell": post_q3_nll,
        "post_q3_ppl": float(math.exp(min(post_q3_nll, 20.0))),
        "post_q3_empty_acc": post_q3_acc,
        "post_q3_autonomous_commit_acc": post_q3_auto_acc,
        "post_q6_nll_per_cell": post_q6_nll,
        "post_q6_empty_acc": post_q6_acc,
        "au_remaining_nll_curve": float(np.mean(rem_nll_curve)),
        "trajectory_nll_per_cell": traj_nll_per_cell,
        "trajectory_ppl": traj_ppl,
        "full24_trajectory_nll_per_cell": full24_nll_per_cell,
        "root_coinflip_recall_at_q3": root_recall_q3,
        "forced_committed_before_noisy_w": float(np.mean(forced_before_w0)),
    }


@torch.no_grad()
def evaluate_sudoku_ordering_composition_arm(
    arm_name: str,
    order_policy: str,
    model: Optional[AnyOrderSudokuNet],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    unit_idx: torch.Tensor,
    device: torch.device,
    n_branches: int = 8,
    gamma_voi: float = 0.20,
    n_reveal_blocks: int = 9,
    seed: int = 101,
    leaf_judge_model: Optional[AnyOrderSudokuNet] = None,
) -> Dict[str, Any]:
    """
    Evaluates the 4 external reveal ordering laws on Sudoku-Extreme dev boards:
      - Order_Random
      - Order_Greedy_MinEntropy (gamma = 0)
      - Order_Pure_MaxVoI (ignores local commit risk H)
      - Order_Unified_VoI_Entropy_Compose (argmin_v [H(X_v | x_{S_t}) - gamma * VoI(v | x_{S_t})])
    """
    nb = x_eval_np.shape[0]
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)
    unit_idx_np = unit_idx.cpu().numpy()
    empty_init_mask = x_eval_np == 0

    def get_probs_and_voi(x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if arm_name == "Control_UniformFloor":
            return torch.full((nb, 81, 9), 1.0 / 9.0, device=device), torch.zeros((nb, 81), device=device)
        assert model is not None
        need_voi = order_policy in ("pure_max_voi", "unified_voi_entropy")
        p, v, _ = predict_sudoku_anyorder_search_and_voi(
            model,
            x_in,
            unit_idx,
            n_branches=n_branches,
            beta_clear=2.5,
            compute_voi=need_voi,
            leaf_judge_model=leaf_judge_model,
        )
        return p, v

    rng = np.random.RandomState(seed + 808)

    def select_block_cells(
        rem_mask: torch.Tensor, ent: torch.Tensor, voi: torch.Tensor, blocks_left: int
    ) -> torch.Tensor:
        step_mask = torch.zeros_like(rem_mask)
        rem_counts = rem_mask.sum(dim=-1)
        for i in range(nb):
            r_cnt = int(rem_counts[i].item())
            if r_cnt == 0:
                continue
            k_pick = max(1, math.ceil(r_cnt / float(blocks_left))) if blocks_left > 1 else r_cnt
            emp_i = torch.where(rem_mask[i])[0]
            if order_policy == "random":
                perm = emp_i[torch.tensor(rng.permutation(len(emp_i))[:k_pick], device=device)]
                step_mask[i, perm] = True
            elif order_policy == "min_entropy":
                scores = ent[i, emp_i]
                pick = emp_i[scores.topk(k_pick, largest=False).indices]
                step_mask[i, pick] = True
            elif order_policy == "pure_max_voi":
                # Pure Max-VoI ranks solely by VoI (breaking ties randomly rather than by minimum entropy!)
                jitter = torch.tensor(rng.uniform(0, 0.02, size=len(emp_i)), device=device, dtype=voi.dtype)
                scores = -voi[i, emp_i] + jitter
                pick = emp_i[scores.topk(k_pick, largest=False).indices]
                step_mask[i, pick] = True
            elif order_policy == "unified_voi_entropy":
                scores = ent[i, emp_i] - gamma_voi * voi[i, emp_i]
                pick = emp_i[scores.topk(k_pick, largest=False).indices]
                step_mask[i, pick] = True
        return step_mask

    # 1. Autoregressive Reveal Trajectory NLL & Remaining-Board NLL Curve
    x_cur = x_init.clone()
    total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    scored_cells = torch.zeros(nb, dtype=torch.float64, device=device)
    rem_nll_curve: List[float] = []

    for b_idx in range(n_reveal_blocks):
        rem_mask = x_cur == 0
        if not rem_mask.any():
            break
        probs_step, voi_step = get_probs_and_voi(x_cur)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)

        true_digits = (y_true - 1).clamp(0, 8)
        true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        cell_nll = -torch.log(true_p).to(torch.float64)
        rem_f = rem_mask.to(torch.float64)
        rem_nll_curve.append(float(((cell_nll * rem_f).sum(dim=-1) / rem_f.sum(dim=-1).clamp(min=1.0)).mean().item()))

        step_mask = select_block_cells(rem_mask, ent, voi_step, n_reveal_blocks - b_idx)
        total_nll = total_nll + (cell_nll * step_mask.to(torch.float64)).sum(dim=-1)
        scored_cells = scored_cells + step_mask.to(torch.float64).sum(dim=-1)
        x_cur = torch.where(step_mask, y_true, x_cur)

    traj_joint_nll = float(total_nll.mean().item())
    traj_nll_per_cell = float((total_nll / scored_cells.clamp(min=1.0)).mean().item())
    traj_ppl = float(math.exp(min(traj_nll_per_cell, 20.0)))

    # 2. Autonomous (Zero-Label Test-Time) Sequential Commit Rollout along the Ordering Policy
    x_auto = x_init.clone()
    for b_idx in range(n_reveal_blocks):
        rem_mask = x_auto == 0
        if not rem_mask.any():
            break
        probs_auto, voi_auto = get_probs_and_voi(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        commit_mask = select_block_cells(rem_mask, ent, voi_auto, n_reveal_blocks - b_idx)
        pred_digits = probs_auto.argmax(dim=-1) + 1
        x_auto = torch.where(commit_mask, pred_digits, x_auto)

    auto_np = x_auto.cpu().numpy()
    auto_empty_acc = float(((auto_np == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))
    auto_collisions = float(count_sudoku_unit_collisions(auto_np, unit_idx_np).mean())

    return {
        "order_policy": order_policy,
        "n_branches": n_branches,
        "gamma_voi": gamma_voi,
        "trajectory_joint_nll": traj_joint_nll,
        "trajectory_nll_per_cell": traj_nll_per_cell,
        "trajectory_ppl": traj_ppl,
        "au_remaining_nll_curve": float(np.mean(rem_nll_curve)),
        "autonomous_empty_acc": auto_empty_acc,
        "autonomous_unit_collisions": auto_collisions,
    }


# ---------------------------------------------------------------------------
# 4. Main Track E Exploratory Runner
# ---------------------------------------------------------------------------

def run_explore_lowdata_and_voi_composition(args) -> Dict[str, Any]:
    train_csv_path, data_hashes = ensure_and_verify_sudoku_extreme_train_only(args.sudoku_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    seeds = [101] if args.smoke_cpu else [101, 102, 103]

    # Strictly development splits from train.csv only (zero touch of test.csv!)
    # Training pool: train.csv[0:512] (for N_train in {32, 128, 512})
    # Unlabelled clue-only pool: train.csv[512:1536] (0 label access verified by WatchedLabelArray)
    # Held-out development eval split: train.csv[6000:6128] (128 boards)
    x_all, y_all = read_sudoku_csv(train_csv_path, 6200, seed=101, random_subset=False)
    x_tr_pool = x_all[0:512].reshape(-1, 81).astype(np.int64)
    y_tr_pool = y_all[0:512].reshape(-1, 81).astype(np.int64)
    x_unlab_pool = x_all[512:1536].reshape(-1, 81).astype(np.int64)
    y_unlab_pool = y_all[512:1536].reshape(-1, 81).astype(np.int64)

    x_dev_full = x_all[6000:6064].reshape(-1, 81).astype(np.int64)
    y_dev_full = y_all[6000:6064].reshape(-1, 81).astype(np.int64)

    x_tr_smoke = x_all[4500:4564].reshape(-1, 81).astype(np.int64)
    y_tr_smoke = y_all[4500:4564].reshape(-1, 81).astype(np.int64)
    x_dev_smoke = x_all[6128:6136].reshape(-1, 81).astype(np.int64)
    y_dev_smoke = y_all[6128:6136].reshape(-1, 81).astype(np.int64)

    x_dev_np = x_dev_smoke if args.smoke_cpu else x_dev_full
    y_dev_np = y_dev_smoke if args.smoke_cpu else y_dev_full

    sudoku_train_dev_overlap = len({tuple(r) for r in x_tr_pool} & {tuple(r) for r in x_dev_full})
    sudoku_unlab_dev_overlap = len({tuple(r) for r in x_unlab_pool} & {tuple(r) for r in x_dev_full})
    sudoku_smoke_dev_overlap = len({tuple(r) for r in x_dev_full} & {tuple(r) for r in x_dev_smoke})

    # Synthetic Environment 1B: Correlated-Parity-Coupled-Grid (6x6 Z_4 modular constraint grid)
    pg_x_tr, pg_y_tr, _, _ = generate_correlated_parity_grid_dataset(512, n_clues=16, seed=2026)
    pg_x_unlab, pg_y_unlab, _, _ = generate_correlated_parity_grid_dataset(512, n_clues=16, seed=2027)
    pg_x_dev, pg_y_dev, _, _ = generate_correlated_parity_grid_dataset(
        8 if args.smoke_cpu else 64, n_clues=16, seed=2028
    )

    # Synthetic Environment 2: Stiff-CoinFlip-Coupled-Grid (24-variable Z_4 environment)
    sc_off_tr, sc_y_tr = generate_stiff_coinflip_coupled_dataset(
        128 if args.smoke_cpu else 512, seed=3030
    )
    sc_off_dev, sc_y_dev = generate_stiff_coinflip_coupled_dataset(
        8 if args.smoke_cpu else 48, seed=3031
    )
    oracle_check = verify_stiff_coinflip_oracle_against_bruteforce(sc_off_dev, sc_y_dev, n_episodes=24)
    if not oracle_check["verified"]:
        raise RuntimeError(f"Rule 9.6 Exact Oracle check failed: {oracle_check}")

    n_train_list = [32, 128] if args.smoke_cpu else [32, 128, 512]
    sudoku_steps = 18 if args.smoke_cpu else args.sudoku_steps
    parity_steps = 18 if args.smoke_cpu else args.parity_steps
    stiff_steps = 65 if args.smoke_cpu else args.stiff_steps
    hidden_size = 48 if args.smoke_cpu else 64
    k_rand_orders = 2 if args.smoke_cpu else 3
    n_rev_blocks = 4 if args.smoke_cpu else 6
    unit_idx = build_sudoku_unit_indices(device)

    watched_unlabelled_y_sudoku = WatchedLabelArray(y_unlab_pool)
    watched_unlabelled_y_parity = WatchedLabelArray(pg_y_unlab)
    max_q_real_param_drift = 0.0

    per_seed: Dict[str, Any] = {}
    t_run_start = time.time()

    for seed in seeds:
        print(f"[explore-lowdata-var] Starting seed={seed} on device={device} (elapsed={time.time()-t_run_start:.1f}s)", file=sys.stderr, flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # -------------------------------------------------------------------
        # PART 1A: Sudoku-Extreme Low-Data Scaling (N_train in {32, 128, 512})
        #          Testing Two-Parameter Split: theta_real (q_real) vs theta_amp (q_amp)
        # -------------------------------------------------------------------
        sudoku_lowdata_res: Dict[str, Any] = {"controls": {}, "by_n_train": {}}

        torch.manual_seed(seed + 9000)
        rand_sudoku_net = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        sudoku_lowdata_res["controls"]["Control_Random_Network"] = evaluate_sudoku_lowdata_arm(
            "Control_Random_Network",
            rand_sudoku_net,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=8,
            beta_clear=2.5,
            k_random_orders=k_rand_orders,
            n_reveal_blocks=n_rev_blocks,
            seed=seed,
        )

        torch.manual_seed(seed)
        untr_sudoku_net = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        sudoku_lowdata_res["controls"]["Control_Untrained_Init"] = evaluate_sudoku_lowdata_arm(
            "Control_Untrained_Init",
            untr_sudoku_net,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            k_random_orders=k_rand_orders,
            n_reveal_blocks=n_rev_blocks,
            seed=seed,
        )
        sudoku_lowdata_res["controls"]["Control_ClueCopyFloor"] = evaluate_sudoku_lowdata_arm(
            "Control_ClueCopyFloor",
            None,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            k_random_orders=k_rand_orders,
            n_reveal_blocks=n_rev_blocks,
            seed=seed,
        )
        sudoku_lowdata_res["controls"]["Control_PeerClue_Uniform"] = evaluate_sudoku_lowdata_arm(
            "Control_PeerClue_Uniform",
            None,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            k_random_orders=k_rand_orders,
            n_reveal_blocks=n_rev_blocks,
            seed=seed,
        )

        best_q_amp_sudoku: Optional[AnyOrderSudokuNet] = None
        best_q_real_sudoku: Optional[AnyOrderSudokuNet] = None

        for n_tr in n_train_list:
            x_sub = (x_tr_smoke[: min(n_tr, len(x_tr_smoke))] if args.smoke_cpu else x_tr_pool[:n_tr])
            y_sub = (y_tr_smoke[: min(n_tr, len(y_tr_smoke))] if args.smoke_cpu else y_tr_pool[:n_tr])

            # ---------------------------------------------------------------
            # Arm 1: Base_q_real_Only (theta_real)
            # Trained STRICTLY on real data D_train via Any-Order Cross-Entropy.
            # Never touched by consistency or amplification distillation gradients!
            # ---------------------------------------------------------------
            torch.manual_seed(seed)
            q_real_sudoku = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
            real_opt = torch.optim.AdamW(q_real_sudoku.parameters(), lr=5e-3, weight_decay=1e-4)
            q_real_sudoku.train()
            for _ in range(sudoku_steps):
                b_idx = np.random.choice(len(x_sub), size=24, replace=True)
                xb_c = torch.tensor(x_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(y_sub[b_idx], dtype=torch.long, device=device)
                rev_p = torch.rand((24, 1), device=device) * 0.85
                rev_m = (torch.rand((24, 81), device=device) < rev_p) & (xb_c == 0)
                xb_st = torch.where(rev_m, yb_s, xb_c)
                t_mask = xb_st == 0
                if not t_mask.any():
                    t_mask = xb_c == 0
                step_lg = q_real_sudoku(xb_st, n_steps=5)
                loss = sum(
                    ((i + 1.0) / 15.0) * F.cross_entropy(lg[t_mask], (yb_s[t_mask] - 1).clamp(0, 8))
                    for i, lg in enumerate(step_lg)
                )
                real_opt.zero_grad()
                loss.backward()
                real_opt.step()

            # Freeze q_real_sudoku and snapshot its parameters to verify 0 drift!
            q_real_sudoku.eval()
            for p in q_real_sudoku.parameters():
                p.requires_grad_(False)
            q_real_snapshot = [p.detach().clone() for p in q_real_sudoku.parameters()]

            # ---------------------------------------------------------------
            # Arm 2: SingleParam_Naively_Regularized
            # Trains a single shared parameter set theta with real-data CE AND
            # naive bidirectional cross-order KL between sparse masks x_{S_sparse}
            # and dense masks x_{S_dense} of the same board — testing whether
            # modifying theta_real directly blurs late-step conditionals!
            # ---------------------------------------------------------------
            torch.manual_seed(seed)
            single_naive_model = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
            single_opt = torch.optim.AdamW(single_naive_model.parameters(), lr=5e-3, weight_decay=1e-4)
            single_naive_model.train()
            for _ in range(sudoku_steps):
                b_idx = np.random.choice(len(x_sub), size=24, replace=True)
                xb_c = torch.tensor(x_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(y_sub[b_idx], dtype=torch.long, device=device)
                # Sparse mask S_sparse (reveal 10-30%) vs Dense mask S_dense (reveal 65-85%)
                rev_p_sp = torch.rand((24, 1), device=device) * 0.25
                rev_p_dn = 0.65 + torch.rand((24, 1), device=device) * 0.25
                m_sp = (torch.rand((24, 81), device=device) < rev_p_sp) & (xb_c == 0)
                m_dn = ((torch.rand((24, 81), device=device) < rev_p_dn) | m_sp) & (xb_c == 0)
                x_sp = torch.where(m_sp, yb_s, xb_c)
                x_dn = torch.where(m_dn, yb_s, xb_c)
                t_sp = x_sp == 0
                t_dn = x_dn == 0
                if not t_dn.any():
                    t_dn = xb_c == 0
                lg_sp = single_naive_model(x_sp, n_steps=5)
                lg_dn = single_naive_model(x_dn, n_steps=5)
                ce_sp = sum(((i + 1.0) / 15.0) * F.cross_entropy(lg[t_sp], (yb_s[t_sp] - 1).clamp(0, 8)) for i, lg in enumerate(lg_sp))
                ce_dn = sum(((i + 1.0) / 15.0) * F.cross_entropy(lg[t_dn], (yb_s[t_dn] - 1).clamp(0, 8)) for i, lg in enumerate(lg_dn))
                # Naive bidirectional KL on shared unrevealed cells: pulls dense logits backward toward blurry sparse logits!
                p_sp = F.softmax(lg_sp[-1], dim=-1).clamp(min=1e-6)
                p_dn = F.softmax(lg_dn[-1], dim=-1).clamp(min=1e-6)
                rem_both = (t_dn).unsqueeze(-1).float()
                naive_bidir_kl = (
                    ((p_sp.detach() * (torch.log(p_sp.detach()) - F.log_softmax(lg_dn[-1], dim=-1)) * rem_both).sum()
                     + (p_dn.detach() * (torch.log(p_dn.detach()) - F.log_softmax(lg_sp[-1], dim=-1)) * rem_both).sum())
                    / rem_both.sum().clamp(min=1.0)
                )
                loss = 0.5 * (ce_sp + ce_dn) + 1.25 * naive_bidir_kl
                single_opt.zero_grad()
                loss.backward()
                single_opt.step()

            # ---------------------------------------------------------------
            # Arm 3: TwoParam_q_real_Frozen_Plus_q_amp_Distilled (SameBoards & UnlabelledClues)
            # Our True Two-Set Architecture:
            #   - theta_real (q_real_sudoku) is FROZEN after real-data training.
            #   - theta_amp (q_amp) is a separate parameter set trained by distilling
            #     the search-amplified teacher q_{theta_real, B=4}^{aug} + teacher-anchored
            #     Zero-Curl consistency!
            # ---------------------------------------------------------------
            torch.manual_seed(seed)
            q_amp_same = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
            q_amp_same.load_state_dict(q_real_sudoku.state_dict())
            for p in q_amp_same.parameters():
                p.requires_grad_(True)
            amp_same_opt = torch.optim.AdamW(q_amp_same.parameters(), lr=4e-3, weight_decay=1e-4)
            q_amp_same.train()
            for _ in range(sudoku_steps):
                b_idx = np.random.choice(len(x_sub), size=24, replace=True)
                xb_c = torch.tensor(x_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(y_sub[b_idx], dtype=torch.long, device=device)
                rev_p = torch.rand((24, 1), device=device) * 0.85
                rev_m = (torch.rand((24, 81), device=device) < rev_p) & (xb_c == 0)
                xb_st = torch.where(rev_m, yb_s, xb_c)
                t_mask = xb_st == 0
                if not t_mask.any():
                    t_mask = xb_c == 0
                step_lg = q_amp_same(xb_st, n_steps=5)
                ce_loss = sum(
                    ((i + 1.0) / 15.0)
                    * (
                        F.cross_entropy(lg[t_mask], (yb_s[t_mask] - 1).clamp(0, 8))
                        + 0.30 * sudoku_anyorder_birkhoff_contradiction_loss(lg, xb_st, unit_idx)
                    )
                    for i, lg in enumerate(step_lg)
                )
                distill_reg, _ = compute_sudoku_twoparam_distill_and_zerocurl_loss(
                    q_amp_same, q_real_sudoku, xb_c[:10], step_lg[-1][:10], unit_idx, n_steps=4
                )
                loss = ce_loss + 0.60 * distill_reg
                amp_same_opt.zero_grad()
                loss.backward()
                amp_same_opt.step()

            torch.manual_seed(seed)
            q_amp_unlab = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
            q_amp_unlab.load_state_dict(q_real_sudoku.state_dict())
            for p in q_amp_unlab.parameters():
                p.requires_grad_(True)
            amp_unlab_opt = torch.optim.AdamW(q_amp_unlab.parameters(), lr=4e-3, weight_decay=1e-4)
            q_amp_unlab.train()
            for _ in range(sudoku_steps):
                b_idx = np.random.choice(len(x_sub), size=24, replace=True)
                xb_c = torch.tensor(x_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(y_sub[b_idx], dtype=torch.long, device=device)
                rev_p = torch.rand((24, 1), device=device) * 0.85
                rev_m = (torch.rand((24, 81), device=device) < rev_p) & (xb_c == 0)
                xb_st = torch.where(rev_m, yb_s, xb_c)
                t_mask = xb_st == 0
                if not t_mask.any():
                    t_mask = xb_c == 0
                step_lg = q_amp_unlab(xb_st, n_steps=5)
                ce_loss = sum(
                    ((i + 1.0) / 15.0)
                    * (
                        F.cross_entropy(lg[t_mask], (yb_s[t_mask] - 1).clamp(0, 8))
                        + 0.30 * sudoku_anyorder_birkhoff_contradiction_loss(lg, xb_st, unit_idx)
                    )
                    for i, lg in enumerate(step_lg)
                )
                # Sample unlabelled clue-only boards (0 access to watched_unlabelled_y_sudoku!)
                u_idx = np.random.choice(len(x_unlab_pool), size=10, replace=True)
                xb_u = torch.tensor(x_unlab_pool[u_idx], dtype=torch.long, device=device)
                lg_u = q_amp_unlab(xb_u, n_steps=4)[-1]
                distill_u, _ = compute_sudoku_twoparam_distill_and_zerocurl_loss(
                    q_amp_unlab, q_real_sudoku, xb_u, lg_u, unit_idx, n_steps=4
                )
                loss = ce_loss + 0.70 * distill_u
                amp_unlab_opt.zero_grad()
                loss.backward()
                amp_unlab_opt.step()

            # Verify that q_real_sudoku had 0.0 parameter drift during q_amp distillation
            for p_init, p_now in zip(q_real_snapshot, q_real_sudoku.parameters()):
                drift = float((p_init - p_now.detach()).abs().max().item())
                if drift > max_q_real_param_drift:
                    max_q_real_param_drift = drift

            best_q_amp_sudoku = q_amp_unlab
            best_q_real_sudoku = q_real_sudoku

            base_eval = evaluate_sudoku_lowdata_arm(
                "Base_q_real_Only",
                q_real_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                k_random_orders=k_rand_orders,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=None,
            )
            single_naive_eval = evaluate_sudoku_lowdata_arm(
                "SingleParam_Naively_Regularized",
                single_naive_model,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                k_random_orders=k_rand_orders,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=None,
            )
            twoparam_same_b0_eval = evaluate_sudoku_lowdata_arm(
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_SameBoards_B0",
                q_amp_same,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                k_random_orders=k_rand_orders,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=q_real_sudoku,
            )
            twoparam_b0_eval = evaluate_sudoku_lowdata_arm(
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B0",
                q_amp_unlab,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                k_random_orders=k_rand_orders,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=q_real_sudoku,
            )
            twoparam_b8_eval = evaluate_sudoku_lowdata_arm(
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B8",
                q_amp_unlab,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                beta_clear=2.5,
                k_random_orders=k_rand_orders,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=q_real_sudoku,
            )

            sudoku_lowdata_res["by_n_train"][str(n_tr)] = {
                "Base_q_real_Only": base_eval,
                "SingleParam_Naively_Regularized": single_naive_eval,
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_SameBoards_B0": twoparam_same_b0_eval,
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B0": twoparam_b0_eval,
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B8": twoparam_b8_eval,
            }

        # -------------------------------------------------------------------
        # PART 1B: Correlated-Parity-Coupled-Grid Low-Data Scaling (N_train in {32, 128, 512})
        #          Testing Two-Parameter Split: theta_real (q_real) vs theta_amp (q_amp)
        # -------------------------------------------------------------------
        parity_lowdata_res: Dict[str, Any] = {"controls": {}, "by_n_train": {}}
        torch.manual_seed(seed + 9000)
        rand_pg_net = AnyOrderParityGridNet(hidden_size=64, device=device).to(device)
        torch.manual_seed(seed)
        untr_pg_net = AnyOrderParityGridNet(hidden_size=64, device=device).to(device)

        parity_lowdata_res["controls"]["Control_Random_Network"] = evaluate_parity_grid_arm(
            "Control_Random_Network", rand_pg_net, pg_x_dev, pg_y_dev, device, k_random_orders=k_rand_orders, seed=seed
        )
        parity_lowdata_res["controls"]["Control_Untrained_Init"] = evaluate_parity_grid_arm(
            "Control_Untrained_Init", untr_pg_net, pg_x_dev, pg_y_dev, device, k_random_orders=k_rand_orders, seed=seed
        )
        parity_lowdata_res["controls"]["Control_ClueCopyFloor"] = evaluate_parity_grid_arm(
            "Control_ClueCopyFloor", None, pg_x_dev, pg_y_dev, device, k_random_orders=k_rand_orders, seed=seed
        )

        for n_tr in n_train_list:
            px_sub = pg_x_tr[:n_tr]
            py_sub = pg_y_tr[:n_tr]

            # 1. Base_q_real_Only (theta_real trained strictly on real data D_train)
            torch.manual_seed(seed)
            pg_q_real = AnyOrderParityGridNet(hidden_size=64, device=device).to(device)
            pg_real_opt = torch.optim.AdamW(pg_q_real.parameters(), lr=5e-3, weight_decay=1e-4)
            pg_q_real.train()
            for _ in range(parity_steps):
                b_idx = np.random.choice(len(px_sub), size=24, replace=True)
                xb_c = torch.tensor(px_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(py_sub[b_idx], dtype=torch.long, device=device)
                rev_p = torch.rand((24, 1), device=device) * 0.85
                rev_m = (torch.rand((24, 36), device=device) < rev_p) & (xb_c == 0)
                xb_st = torch.where(rev_m, yb_s, xb_c)
                t_mask = xb_st == 0
                if not t_mask.any():
                    t_mask = xb_c == 0
                step_lg = pg_q_real(xb_st, n_steps=4)
                loss = sum(
                    ((i + 1.0) / 10.0) * F.cross_entropy(lg[t_mask], (yb_s[t_mask] - 1).clamp(0, 3))
                    for i, lg in enumerate(step_lg)
                )
                pg_real_opt.zero_grad()
                loss.backward()
                pg_real_opt.step()

            pg_q_real.eval()
            for p in pg_q_real.parameters():
                p.requires_grad_(False)
            pg_real_snap = [p.detach().clone() for p in pg_q_real.parameters()]

            # 2. SingleParam_Naively_Regularized (single shared parameter set with bidirectional sparse-dense KL)
            torch.manual_seed(seed)
            pg_single_naive = AnyOrderParityGridNet(hidden_size=64, device=device).to(device)
            pg_naive_opt = torch.optim.AdamW(pg_single_naive.parameters(), lr=5e-3, weight_decay=1e-4)
            pg_single_naive.train()
            for _ in range(parity_steps):
                b_idx = np.random.choice(len(px_sub), size=24, replace=True)
                xb_c = torch.tensor(px_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(py_sub[b_idx], dtype=torch.long, device=device)
                rev_p_sp = torch.rand((24, 1), device=device) * 0.25
                rev_p_dn = 0.65 + torch.rand((24, 1), device=device) * 0.25
                m_sp = (torch.rand((24, 36), device=device) < rev_p_sp) & (xb_c == 0)
                m_dn = ((torch.rand((24, 36), device=device) < rev_p_dn) | m_sp) & (xb_c == 0)
                x_sp = torch.where(m_sp, yb_s, xb_c)
                x_dn = torch.where(m_dn, yb_s, xb_c)
                t_sp = x_sp == 0
                t_dn = x_dn == 0
                if not t_dn.any():
                    t_dn = xb_c == 0
                lg_sp = pg_single_naive(x_sp, n_steps=4)
                lg_dn = pg_single_naive(x_dn, n_steps=4)
                ce_sp = sum(((i + 1.0) / 10.0) * F.cross_entropy(lg[t_sp], (yb_s[t_sp] - 1).clamp(0, 3)) for i, lg in enumerate(lg_sp))
                ce_dn = sum(((i + 1.0) / 10.0) * F.cross_entropy(lg[t_dn], (yb_s[t_dn] - 1).clamp(0, 3)) for i, lg in enumerate(lg_dn))
                p_sp = F.softmax(lg_sp[-1], dim=-1).clamp(min=1e-6)
                p_dn = F.softmax(lg_dn[-1], dim=-1).clamp(min=1e-6)
                rem_both = t_dn.unsqueeze(-1).float()
                naive_bidir_kl = (
                    ((p_sp.detach() * (torch.log(p_sp.detach()) - F.log_softmax(lg_dn[-1], dim=-1)) * rem_both).sum()
                     + (p_dn.detach() * (torch.log(p_dn.detach()) - F.log_softmax(lg_sp[-1], dim=-1)) * rem_both).sum())
                    / rem_both.sum().clamp(min=1.0)
                )
                loss = 0.5 * (ce_sp + ce_dn) + 1.25 * naive_bidir_kl
                pg_naive_opt.zero_grad()
                loss.backward()
                pg_naive_opt.step()

            # 3. TwoParam_q_real_Frozen_Plus_q_amp_Distilled (frozen q_real + distilled q_amp)
            torch.manual_seed(seed)
            pg_q_amp = AnyOrderParityGridNet(hidden_size=64, device=device).to(device)
            pg_q_amp.load_state_dict(pg_q_real.state_dict())
            for p in pg_q_amp.parameters():
                p.requires_grad_(True)
            pg_amp_opt = torch.optim.AdamW(pg_q_amp.parameters(), lr=5e-3, weight_decay=1e-4)
            pg_q_amp.train()
            for _ in range(parity_steps):
                b_idx = np.random.choice(len(px_sub), size=24, replace=True)
                xb_c = torch.tensor(px_sub[b_idx], dtype=torch.long, device=device)
                yb_s = torch.tensor(py_sub[b_idx], dtype=torch.long, device=device)
                rev_p = torch.rand((24, 1), device=device) * 0.85
                rev_m = (torch.rand((24, 36), device=device) < rev_p) & (xb_c == 0)
                xb_st = torch.where(rev_m, yb_s, xb_c)
                t_mask = xb_st == 0
                if not t_mask.any():
                    t_mask = xb_c == 0
                step_lg = pg_q_amp(xb_st, n_steps=4)
                ce_loss = sum(
                    ((i + 1.0) / 10.0)
                    * (
                        F.cross_entropy(lg[t_mask], (yb_s[t_mask] - 1).clamp(0, 3))
                        + 0.35 * parity_grid_modular_consistency_loss(lg, xb_st)
                    )
                    for i, lg in enumerate(step_lg)
                )
                u_idx = np.random.choice(len(pg_x_unlab), size=12, replace=True)
                xb_u = torch.tensor(pg_x_unlab[u_idx], dtype=torch.long, device=device)
                lg_u = pg_q_amp(xb_u, n_steps=4)[-1]
                distill_u, _ = compute_parity_grid_twoparam_distill_and_zerocurl_loss(
                    pg_q_amp, pg_q_real, xb_u, lg_u
                )
                loss = ce_loss + 0.65 * distill_u
                pg_amp_opt.zero_grad()
                loss.backward()
                pg_amp_opt.step()

            for p_init, p_now in zip(pg_real_snap, pg_q_real.parameters()):
                drift = float((p_init - p_now.detach()).abs().max().item())
                if drift > max_q_real_param_drift:
                    max_q_real_param_drift = drift

            parity_lowdata_res["by_n_train"][str(n_tr)] = {
                "Base_q_real_Only": evaluate_parity_grid_arm(
                    "Base_q_real_Only",
                    pg_q_real,
                    pg_x_dev,
                    pg_y_dev,
                    device,
                    n_branches=0,
                    k_random_orders=k_rand_orders,
                    seed=seed,
                    leaf_judge_model=None,
                ),
                "SingleParam_Naively_Regularized": evaluate_parity_grid_arm(
                    "SingleParam_Naively_Regularized",
                    pg_single_naive,
                    pg_x_dev,
                    pg_y_dev,
                    device,
                    n_branches=0,
                    k_random_orders=k_rand_orders,
                    seed=seed,
                    leaf_judge_model=None,
                ),
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B0": evaluate_parity_grid_arm(
                    "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B0",
                    pg_q_amp,
                    pg_x_dev,
                    pg_y_dev,
                    device,
                    n_branches=0,
                    k_random_orders=k_rand_orders,
                    seed=seed,
                    leaf_judge_model=pg_q_real,
                ),
                "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B8": evaluate_parity_grid_arm(
                    "TwoParam_q_real_Frozen_Plus_q_amp_Distilled_B8",
                    pg_q_amp,
                    pg_x_dev,
                    pg_y_dev,
                    device,
                    n_branches=8,
                    k_random_orders=k_rand_orders,
                    seed=seed,
                    leaf_judge_model=pg_q_real,
                ),
            }

        # -------------------------------------------------------------------
        # PART 2: Stiffness vs. Entropy — Unified Composition of Reveal Order & VoI Search
        # -------------------------------------------------------------------
        torch.manual_seed(seed + 9000)
        sc_rand_net = AnyOrderStiffCoupledNet(hidden_size=64, device=device).to(device)
        torch.manual_seed(seed)
        sc_untr_net = AnyOrderStiffCoupledNet(hidden_size=64, device=device).to(device)

        torch.manual_seed(seed)
        sc_model = AnyOrderStiffCoupledNet(hidden_size=64, device=device).to(device)
        sc_opt = torch.optim.AdamW(sc_model.parameters(), lr=4e-2, weight_decay=1e-4)
        sc_model.train()
        for step_i in range(stiff_steps):
            b_idx = np.random.choice(len(sc_off_tr), size=64, replace=True)
            off_b = torch.tensor(sc_off_tr[b_idx], dtype=torch.long, device=device)
            yb = torch.tensor(sc_y_tr[b_idx], dtype=torch.long, device=device)
            # Balanced Any-Order conditioning masks:
            #   - 8 boards with 0 revealed variables (S_t = empty)
            #   - 24 boards with 1 single revealed variable (|S_t| = 1, covering all 24 variables)
            #   - 16 boards with 2-3 revealed variables (|S_t| in {2, 3})
            #   - 16 boards with dense random reveal fraction p in (0.20, 0.85)
            rev_m = torch.zeros((64, 24), dtype=torch.bool, device=device)
            for k_one in range(24):
                rev_m[8 + k_one, k_one] = True
            for k_few in range(16):
                v_pair = torch.randperm(24, device=device)[: (2 + (k_few % 2))]
                rev_m[32 + k_few, v_pair] = True
            rev_p_dense = 0.20 + torch.rand((16, 1), device=device) * 0.65
            rev_m[48:64] = torch.rand((16, 24), device=device) < rev_p_dense

            xb_st = torch.where(rev_m, yb, torch.zeros_like(yb))
            t_mask = xb_st == 0
            if not t_mask.any():
                t_mask = torch.ones_like(rev_m)
            step_lg = sc_model(xb_st, off_b, n_steps=4)
            ce_loss = sum(
                ((i + 1.0) / 10.0) * F.cross_entropy(lg[t_mask], (yb[t_mask] - 1).clamp(0, 3))
                for i, lg in enumerate(step_lg)
            )
            sc_opt.zero_grad()
            ce_loss.backward()
            sc_opt.step()

        stiff_grid_arms: Dict[str, Any] = {
            "Control_Random_Network": evaluate_stiff_coinflip_ordering_arm(
                "Control_Random_Network",
                "unified_voi_entropy",
                sc_rand_net,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=8,
                gamma_voi=1.0,
                seed=seed,
            ),
            "Control_Untrained_Init": evaluate_stiff_coinflip_ordering_arm(
                "Control_Untrained_Init",
                "unified_voi_entropy",
                sc_untr_net,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=0,
                gamma_voi=1.0,
                seed=seed,
            ),
            "Control_UniformFloor": evaluate_stiff_coinflip_ordering_arm(
                "Control_UniformFloor",
                "random",
                None,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=0,
                gamma_voi=0.0,
                seed=seed,
            ),
            "Order_Random": evaluate_stiff_coinflip_ordering_arm(
                "Order_Random",
                "random",
                sc_model,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=8,
                gamma_voi=0.0,
                seed=seed,
            ),
            "Order_Greedy_MinEntropy": evaluate_stiff_coinflip_ordering_arm(
                "Order_Greedy_MinEntropy",
                "min_entropy",
                sc_model,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=8,
                gamma_voi=0.0,
                seed=seed,
            ),
            "Order_Pure_MaxVoI": evaluate_stiff_coinflip_ordering_arm(
                "Order_Pure_MaxVoI",
                "pure_max_voi",
                sc_model,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=8,
                gamma_voi=1.0,
                seed=seed,
            ),
            "Ablation_Unified_VoI_Entropy_NoSearch_B0": evaluate_stiff_coinflip_ordering_arm(
                "Ablation_Unified_VoI_Entropy_NoSearch_B0",
                "unified_voi_entropy",
                sc_model,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=0,
                gamma_voi=1.0,
                seed=seed,
            ),
            "Order_Unified_VoI_Entropy_Compose_B8": evaluate_stiff_coinflip_ordering_arm(
                "Order_Unified_VoI_Entropy_Compose_B8",
                "unified_voi_entropy",
                sc_model,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=8,
                gamma_voi=1.0,
                seed=seed,
            ),
            "Ceiling_Exact_Oracle_Unified_VoI_Entropy": evaluate_stiff_coinflip_ordering_arm(
                "Ceiling_Exact_Oracle_Unified_VoI_Entropy",
                "unified_voi_entropy",
                None,
                sc_off_dev,
                sc_y_dev,
                device,
                n_branches=0,
                gamma_voi=1.0,
                seed=seed,
            ),
        }

        # Also evaluate the 4 ordering/composition laws on Sudoku-Extreme dev (train.csv[6000:6128])
        # using the Two-Parameter pair (best_q_amp_sudoku + frozen best_q_real_sudoku)
        assert best_q_amp_sudoku is not None and best_q_real_sudoku is not None
        sudoku_voi_compose_arms: Dict[str, Any] = {
            "Control_Random_Network": evaluate_sudoku_ordering_composition_arm(
                "Control_Random_Network",
                "unified_voi_entropy",
                rand_sudoku_net,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                gamma_voi=0.20,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=None,
            ),
            "Control_Untrained_Init": evaluate_sudoku_ordering_composition_arm(
                "Control_Untrained_Init",
                "unified_voi_entropy",
                untr_sudoku_net,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                gamma_voi=0.20,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=None,
            ),
            "Control_UniformFloor": evaluate_sudoku_ordering_composition_arm(
                "Control_UniformFloor",
                "random",
                None,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                gamma_voi=0.0,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=None,
            ),
            "Order_Random": evaluate_sudoku_ordering_composition_arm(
                "Order_Random",
                "random",
                best_q_amp_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                gamma_voi=0.0,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=best_q_real_sudoku,
            ),
            "Order_Greedy_MinEntropy": evaluate_sudoku_ordering_composition_arm(
                "Order_Greedy_MinEntropy",
                "min_entropy",
                best_q_amp_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                gamma_voi=0.0,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=best_q_real_sudoku,
            ),
            "Order_Pure_MaxVoI": evaluate_sudoku_ordering_composition_arm(
                "Order_Pure_MaxVoI",
                "pure_max_voi",
                best_q_amp_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                gamma_voi=1.0,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=best_q_real_sudoku,
            ),
            "Ablation_Unified_VoI_Entropy_NoSearch_B0": evaluate_sudoku_ordering_composition_arm(
                "Ablation_Unified_VoI_Entropy_NoSearch_B0",
                "unified_voi_entropy",
                best_q_amp_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=0,
                gamma_voi=0.20,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=best_q_real_sudoku,
            ),
            "Order_Unified_VoI_Entropy_Compose_B8": evaluate_sudoku_ordering_composition_arm(
                "Order_Unified_VoI_Entropy_Compose_B8",
                "unified_voi_entropy",
                best_q_amp_sudoku,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=8,
                gamma_voi=0.20,
                n_reveal_blocks=n_rev_blocks,
                seed=seed,
                leaf_judge_model=best_q_real_sudoku,
            ),
        }

        per_seed[str(seed)] = {
            "part1_sudoku_lowdata": sudoku_lowdata_res,
            "part1_parity_grid_lowdata": parity_lowdata_res,
            "part2_stiff_coinflip_grid": stiff_grid_arms,
            "part2_sudoku_voi_composition": sudoku_voi_compose_arms,
        }

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    summary = summarize_results(per_seed, seeds, n_train_list)

    return {
        "track": "EXPLORATORY (PROTOCOL.md §10.1)",
        "experiment": "explore_lowdata_order_variance",
        "commit": args.commit,
        "runner_sha256": runner_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "dataset_sha256": data_hashes,
        "splits": {
            "sudoku_train_pool": "train.csv[4500:4564]" if args.smoke_cpu else "train.csv[0:512]",
            "sudoku_unlabelled_clue_pool": "train.csv[512:1536] (0 label access)",
            "sudoku_dev_eval": "train.csv[6128:6136]" if args.smoke_cpu else "train.csv[6000:6064]",
            "sudoku_train_dev_overlap": sudoku_train_dev_overlap,
            "sudoku_unlab_dev_overlap": sudoku_unlab_dev_overlap,
            "sudoku_smoke_dev_overlap": sudoku_smoke_dev_overlap,
            "test_csv_touched": False,
        },
        "two_parameter_split_verification": {
            "q_real_trained_only_on_real_data": True,
            "q_real_frozen_during_q_amp_distillation": True,
            "q_real_max_abs_param_drift_during_distillation": max_q_real_param_drift,
        },
        "unlabelled_pool_label_access_counts": {
            "sudoku_unlabelled_y_access_count": watched_unlabelled_y_sudoku.access_count,
            "parity_unlabelled_y_access_count": watched_unlabelled_y_parity.access_count,
        },
        "oracle_brute_force_verification": oracle_check,
        "summary": summary,
        "per_seed": per_seed,
    }


def summarize_results(
    per_seed: Dict[str, Any], seeds: List[int], n_train_list: List[int]
) -> Dict[str, Any]:
    s_keys = [str(s) for s in seeds]
    sudoku_metrics = [
        "random_order_joint_nll",
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "adaptive_order_joint_nll",
        "min_h_nll_per_cell",
        "min_h_ppl",
        "cross_order_nll_std",
        "commutator_curl_mae",
        "early_step_nll_per_cell",
        "late_step_nll_per_cell",
        "one_shot_empty_acc",
        "autonomous_empty_acc",
        "autonomous_unit_collisions",
    ]
    part1_sudoku_summary: Dict[str, Any] = {"controls": {}, "by_n_train": {}}
    for ctrl in per_seed[s_keys[0]]["part1_sudoku_lowdata"]["controls"].keys():
        part1_sudoku_summary["controls"][ctrl] = {
            m: round(float(np.mean([per_seed[s]["part1_sudoku_lowdata"]["controls"][ctrl][m] for s in s_keys])), 5)
            for m in sudoku_metrics
        }
    for n_tr in [str(n) for n in n_train_list]:
        part1_sudoku_summary["by_n_train"][n_tr] = {}
        for arm in per_seed[s_keys[0]]["part1_sudoku_lowdata"]["by_n_train"][n_tr].keys():
            part1_sudoku_summary["by_n_train"][n_tr][arm] = {
                m: round(
                    float(np.mean([per_seed[s]["part1_sudoku_lowdata"]["by_n_train"][n_tr][arm][m] for s in s_keys])),
                    5,
                )
                for m in sudoku_metrics
            }

    pg_metrics = [
        "random_order_joint_nll",
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "adaptive_order_joint_nll",
        "min_h_nll_per_cell",
        "cross_order_nll_std",
        "commutator_curl_mae",
        "early_step_nll_per_cell",
        "late_step_nll_per_cell",
        "one_shot_empty_acc",
    ]
    part1_pg_summary: Dict[str, Any] = {"controls": {}, "by_n_train": {}}
    for ctrl in per_seed[s_keys[0]]["part1_parity_grid_lowdata"]["controls"].keys():
        part1_pg_summary["controls"][ctrl] = {
            m: round(float(np.mean([per_seed[s]["part1_parity_grid_lowdata"]["controls"][ctrl][m] for s in s_keys])), 5)
            for m in pg_metrics
        }
    for n_tr in [str(n) for n in n_train_list]:
        part1_pg_summary["by_n_train"][n_tr] = {}
        for arm in per_seed[s_keys[0]]["part1_parity_grid_lowdata"]["by_n_train"][n_tr].keys():
            part1_pg_summary["by_n_train"][n_tr][arm] = {
                m: round(
                    float(
                        np.mean([per_seed[s]["part1_parity_grid_lowdata"]["by_n_train"][n_tr][arm][m] for s in s_keys])
                    ),
                    5,
                )
                for m in pg_metrics
            }

    sc_metrics = [
        "post_q3_nll_per_cell",
        "post_q3_ppl",
        "post_q3_empty_acc",
        "post_q3_autonomous_commit_acc",
        "post_q6_nll_per_cell",
        "post_q6_empty_acc",
        "au_remaining_nll_curve",
        "trajectory_nll_per_cell",
        "trajectory_ppl",
        "root_coinflip_recall_at_q3",
        "forced_committed_before_noisy_w",
    ]
    part2_sc_summary: Dict[str, Any] = {}
    for arm in per_seed[s_keys[0]]["part2_stiff_coinflip_grid"].keys():
        part2_sc_summary[arm] = {
            m: round(float(np.mean([per_seed[s]["part2_stiff_coinflip_grid"][arm][m] for s in s_keys])), 5)
            for m in sc_metrics
        }

    sv_metrics = [
        "trajectory_joint_nll",
        "trajectory_nll_per_cell",
        "trajectory_ppl",
        "au_remaining_nll_curve",
        "autonomous_empty_acc",
        "autonomous_unit_collisions",
    ]
    part2_sudoku_voi_summary: Dict[str, Any] = {}
    for arm in per_seed[s_keys[0]]["part2_sudoku_voi_composition"].keys():
        part2_sudoku_voi_summary[arm] = {
            m: round(float(np.mean([per_seed[s]["part2_sudoku_voi_composition"][arm][m] for s in s_keys])), 5)
            for m in sv_metrics
        }

    return {
        "status": "EXPLORATORY_SMOKE" if len(seeds) == 1 else "EXPLORATORY_DEV",
        "part1_sudoku_extreme_lowdata_mean": part1_sudoku_summary,
        "part1_correlated_parity_grid_lowdata_mean": part1_pg_summary,
        "part2_stiff_coinflip_coupled_grid_mean": part2_sc_summary,
        "part2_sudoku_extreme_voi_composition_mean": part2_sudoku_voi_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--sudoku-dir", type=str, default="/home/agent/data/datasets/sudoku-extreme")
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--sudoku-steps", type=int, default=100)
    parser.add_argument("--parity-steps", type=int, default=85)
    parser.add_argument("--stiff-steps", type=int, default=100)
    args = parser.parse_args()

    t0 = time.time()
    results = run_explore_lowdata_and_voi_composition(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(results["summary"], indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
