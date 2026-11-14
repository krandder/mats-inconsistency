#!/usr/bin/env python3
"""
Track E Exploratory Runner (PROTOCOL.md §10.1):
Search-Augmented Any-Order World Model Perplexity & Minimum-Entropy Rollout Evaluation
(explore_aom_search_ppl.py)

EXPLORATORY ONLY — strictly zero touch of confirmatory test.csv or test splits.
Evaluates:
  1. Autoregressive Joint Negative Log-Likelihood (joint_nll, nll_per_cell) and Perplexity
     (ppl = exp(nll_per_cell)) along K=4 uniformly random reveal orders sigma_rand of the
     empty cells, plus cross_order_nll_std (standard deviation of total board NLL across K=4
     random reveal orders) and cross_order_cell_nll_std.
  2. Joint NLL / Perplexity along the Model's Adaptive Minimum-Entropy Reveal Order sigma_min_H
     (v_t = argmin_{v notin S_t} H(q_{theta, B}^{aug}(X_v | x_{S_t}))), plus 100% autonomous
     (zero-label test-time) sequential solve accuracy and unit collision count along sigma_min_H.
  3. Internal Inconsistency-Minimizing Search Augmentation across budgets B in {0, 4, 16, 32}
     in BOTH:
       (a) Label-Free Any-Order Self-Consistency + Birkhoff Contradiction Training
           (0 ground-truth labels accessed, verified via WatchedLabelArray), and
       (b) Any-Order Masked Trajectory Training (on development training split train.csv[0:4000]),
     compared against:
       - Control_Random_Network (run through the exact same B=16 search pipeline per Rule 9.1)
       - Control_Untrained_Init (untrained model at initialization)
       - Control_ClueCopyFloor (clue copy + uniform 1/9 over empty cells; PPL = 9.0)
       - Control_PeerClue_Uniform (clue copy + uniform over non-conflicting peer digits)
       - Ablation_Unweighted_Rollout_Average_B16 (beta_clear = 0: averages the exact same B=16
         internal lookahead branches without inconsistency energy weighting).
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
    """
    Wraps ground-truth label arrays and increments access_count on every __getitem__ read.
    """

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
        if isinstance(self._array, list):
            if isinstance(idx, (list, tuple, np.ndarray)):
                return [self._array[int(i)] for i in idx]
            return self._array[int(idx)]
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
        "mean_diff": mean_d,
        "ci95_low": float(mean_d - T_CRIT_DF2 * se),
        "ci95_high": float(mean_d + T_CRIT_DF2 * se),
    }


# ---------------------------------------------------------------------------
# 1. Any-Order Sudoku World Model & Pure-Neural Inconsistency Tree Search
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
    u_vals = preds_81[:, unit_idx_np]  # [B, 27, 9]
    eq = u_vals[:, :, :, None] == u_vals[:, :, None, :]  # [B, 27, 9, 9]
    pair_dups = (eq.sum(axis=(2, 3)) - 9) // 2  # [B, 27]
    return pair_dups.sum(axis=1).astype(np.float64)


class AnyOrderSudokuNet(nn.Module):
    """
    Any-Order Conditional World Model q_theta(X_v | x_{S_t}) on 9x9 Sudoku.
    Given any partial board x_{S_t} in {0..9}^81 (0 = unconditioned, 1..9 = conditioned),
    performs S recurrent unit-message-passing steps across row, column, and 3x3 box ports.
    All linear/embedding weights are initialized from N(0, 0.02^2); zero hardcoded solver rules.
    """

    def __init__(self, hidden_size: int = 96, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.hidden_size = hidden_size
        self.clue_emb = nn.Embedding(10, hidden_size)
        self.pos_emb = nn.Embedding(81, hidden_size)
        nn.init.normal_(self.clue_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        # 99 channels: probs(9), clue_oh(9), and for each of 3 ports (row, col, box):
        # p_other(9), c_other(9), unit_deficit=(1 - p_other - c_other).clamp(0)(9) -> 18 + 3*27 = 99
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
        if p_idx == 0:  # row
            return t.view(bsz, 9, 9, ch).sum(dim=2, keepdim=True).expand(bsz, 9, 9, ch).reshape(bsz, 81, ch) - t
        if p_idx == 1:  # col
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
            logits = self.head(h) + self.peer_logit_proj(peer_p_cat)
            step_logits.append(logits)
            probs = F.softmax(logits.detach(), dim=-1)
        return step_logits


def sudoku_anyorder_birkhoff_contradiction_loss(
    logits: torch.Tensor, x_cond: torch.Tensor, unit_idx: torch.Tensor
) -> torch.Tensor:
    """
    Label-Free Conditional Contradiction + Birkhoff Doubly-Stochastic Loss on partial board x_cond:
      1. Unit doubly-stochastic digit sum: sum_{v in g} P_v(d) == 1 for all 27 units g and digits d=1..9.
      2. Pairwise unit collision inner product: (||sum_{v in g} P_v||_2^2 - sum_{v in g} ||P_v||_2^2) / 8.
      3. Observable conditioned peer exclusion: both log-barrier -log(1 - sum_{d in peer} P_v(d))
         and direct logit margin pushing peer-occupied digit logits below -5.0.
      4. Conditioned cell preservation cross-entropy on s in S_t.
    Uses ZERO ground-truth solution labels.
    """
    probs = F.softmax(logits, dim=-1)  # [B, 81, 9]
    unit_probs = probs[:, unit_idx, :]  # [B, 27, 9, 9]
    unit_digit_sum = unit_probs.sum(dim=2)  # [B, 27, 9]
    birkhoff_col_loss = ((unit_digit_sum - 1.0) ** 2).mean()

    unit_sum_sq = (unit_digit_sum * unit_digit_sum).sum(dim=-1)
    unit_cell_sq = (unit_probs * unit_probs).sum(dim=(2, 3))
    unit_pair_ip = ((unit_sum_sq - unit_cell_sq) / 8.0).mean()

    clue_oh = F.one_hot(x_cond.clamp(min=0, max=9), num_classes=10)[:, :, 1:].to(probs.dtype)  # [B, 81, 9]
    unit_clues = clue_oh[:, unit_idx, :]  # [B, 27, 9, 9]
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


def compute_pure_neural_sudoku_inconsistency_energy(
    leaf_probs: torch.Tensor,
    base_probs: torch.Tensor,
    x_branch: torch.Tensor,
    unit_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Pure-Neural Branch Inconsistency Energy E_inc(C_b) evaluated 100% on the network's output
    distribution leaf_probs = q_theta(. | C_b) (shape [N_batch, B, 81, 9]):
      (i) Unit probability collision energy:
          sum_{g=1..27} sum_{d=1..9} max(0, sum_{v in g} q_theta(X_v=d | C_b) - 1)^2
      (ii) Dead-end cell depletion energy:
          mean_{v notin S_b} -log max_d q_theta(X_v=d | C_b)
      (iii) Path order-inconsistency KL:
          mean_{v notin S_b} D_KL(q_theta(X_v | C_b) || q_theta(X_v | x_{S_t}))
    Zero Python constraint checker or rule helper is used.
    """
    nb, n_branches, _, _ = leaf_probs.shape
    flat_p = leaf_probs.reshape(nb * n_branches, 81, 9)
    unit_p = flat_p[:, unit_idx, :]  # [nb*B, 27, 9, 9]
    unit_digit_sum = unit_p.sum(dim=2)  # [nb*B, 27, 9]

    # (i) Unit collision energy
    excess = F.relu(unit_digit_sum - 1.0)
    unit_coll_energy = (excess * excess).sum(dim=(1, 2)).view(nb, n_branches)

    # (ii) Dead-end cell depletion energy on unconditioned cells
    uncond_mask = (x_branch == 0).float()  # [nb, B, 81]
    uncond_denom = uncond_mask.sum(dim=-1).clamp(min=1.0)
    max_p = leaf_probs.max(dim=-1).values.clamp(min=1e-6)  # [nb, B, 81]
    deplete_energy = ((-torch.log(max_p) * uncond_mask).sum(dim=-1) / uncond_denom) * 10.0

    # (iii) Path order-inconsistency KL
    base_exp = base_probs.unsqueeze(1).expand_as(leaf_probs).clamp(min=1e-6)
    leaf_clamped = leaf_probs.clamp(min=1e-6)
    kl_cell = (leaf_clamped * (torch.log(leaf_clamped) - torch.log(base_exp))).sum(dim=-1)  # [nb, B, 81]
    path_kl_energy = (kl_cell * uncond_mask).sum(dim=-1) / uncond_denom

    return unit_coll_energy + deplete_energy + 0.25 * path_kl_energy


@torch.no_grad()
def predict_sudoku_anyorder_search(
    model: AnyOrderSudokuNet,
    x_cond: torch.Tensor,
    unit_idx: torch.Tensor,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    rollout_depth: int = 2,
    n_steps: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the Search-Augmented Any-Order conditional distribution q_{theta, B}^{aug}(X_v | x_{S_t})
    for a batch of partial Sudoku boards x_cond ([N_batch, 81]).
      - At n_branches == 0: returns smoothed base distribution q_theta(X_v | x_{S_t}).
      - At n_branches in {4, 16, 32}: expands B=2^{R_B} structured root hypotheses on the R_B
        lowest-entropy uncertain cells (R_4=2, R_16=4, R_32=5), rolls out pure-neural
        conditioning steps, evaluates Pure-Neural Branch Inconsistency Energy E_inc(C_b),
        and clears the B branches with weights w_b proportional to pi_prior(C_b) * exp(-beta_clear * E_inc(C_b)).
    """
    model.eval()
    nb = x_cond.shape[0]
    device = x_cond.device
    base_logits = model(x_cond, n_steps=n_steps)[-1]
    base_probs = F.softmax(base_logits, dim=-1)  # [nb, 81, 9]
    eps = 1e-4

    if n_branches <= 0:
        out_probs = (1.0 - eps) * base_probs + (eps / 9.0)
        return out_probs, {"mean_branch_energy": 0.0, "min_branch_energy": 0.0, "effective_branch_ess": 1.0}

    r_bits = {4: 2, 16: 4, 32: 5}.get(n_branches, 4)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)],
        dtype=torch.long,
        device=device,
    )  # [B, r_bits]

    p_clamped = base_probs.clamp(min=1e-7)
    cell_ent = -(p_clamped * torch.log(p_clamped)).sum(dim=-1)  # [nb, 81]
    top2_vals, top2_idx = base_probs.topk(2, dim=-1)  # [nb, 81, 2], digits 0..8

    is_empty = x_cond == 0
    uncertain_score = cell_ent + torch.where(is_empty, torch.zeros_like(cell_ent), torch.full_like(cell_ent, 1e6))
    uncertain_score = uncertain_score + torch.where(
        top2_vals[:, :, 0] > 0.92, torch.full_like(cell_ent, 10.0), torch.zeros_like(cell_ent)
    )
    root_cells = uncertain_score.topk(r_bits, dim=-1, largest=False).indices  # [nb, r_bits]

    root_top2_digits = top2_idx.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)) + 1  # [nb, r_bits, 2]
    root_top2_probs = top2_vals.gather(1, root_cells.unsqueeze(-1).expand(nb, r_bits, 2)).clamp(min=1e-5)

    x_branch = x_cond.unsqueeze(1).expand(nb, n_branches, 81).clone()
    log_prior = torch.zeros(nb, n_branches, device=device)

    for bit_i in range(r_bits):
        choice_bit = bit_table[:, bit_i].unsqueeze(0).expand(nb, n_branches)  # [nb, B]
        cell_i = root_cells[:, bit_i].unsqueeze(1).expand(nb, n_branches)  # [nb, B]
        cand_d = root_top2_digits[:, bit_i, :].gather(1, choice_bit)  # [nb, B]
        cand_p = root_top2_probs[:, bit_i, :].gather(1, choice_bit)  # [nb, B]
        cell_was_empty = is_empty.gather(1, root_cells[:, bit_i : bit_i + 1]).expand(nb, n_branches)
        cur_val = x_branch.gather(2, cell_i.unsqueeze(-1)).squeeze(-1)
        new_val = torch.where(cell_was_empty, cand_d, cur_val)
        x_branch.scatter_(2, cell_i.unsqueeze(-1), new_val.unsqueeze(-1))
        log_prior = log_prior + torch.where(cell_was_empty, torch.log(cand_p), torch.zeros_like(cand_p))

    x_after_root = x_branch.clone()

    # Perform remaining rollout_depth - 1 pure-neural lookahead conditioning steps
    cells_per_step = 2 if n_branches <= 4 else (3 if n_branches <= 16 else 4)
    for _ in range(rollout_depth - 1):
        flat_x = x_branch.reshape(nb * n_branches, 81)
        step_logits = model(flat_x, n_steps=max(3, n_steps - 1))[-1]
        step_probs = F.softmax(step_logits, dim=-1).view(nb, n_branches, 81, 9)
        sp_clamped = step_probs.clamp(min=1e-7)
        sp_ent = -(sp_clamped * torch.log(sp_clamped)).sum(dim=-1)  # [nb, B, 81]
        rem_empty = x_branch == 0
        sp_score = sp_ent + torch.where(rem_empty, torch.zeros_like(sp_ent), torch.full_like(sp_ent, 1e6))
        next_cells = sp_score.topk(cells_per_step, dim=-1, largest=False).indices  # [nb, B, K]
        best_digits = step_probs.argmax(dim=-1) + 1  # [nb, B, 81]
        picked_digits = best_digits.gather(2, next_cells)  # [nb, B, K]
        was_empty = rem_empty.gather(2, next_cells)
        cur_vals = x_branch.gather(2, next_cells)
        upd_vals = torch.where(was_empty, picked_digits, cur_vals)
        x_branch.scatter_(2, next_cells, upd_vals)

    # Final leaf evaluation q_theta(. | C_b)
    leaf_logits = model(x_branch.reshape(nb * n_branches, 81), n_steps=n_steps)[-1]
    leaf_probs = F.softmax(leaf_logits, dim=-1).view(nb, n_branches, 81, 9)

    e_inc = compute_pure_neural_sudoku_inconsistency_energy(leaf_probs, base_probs, x_branch, unit_idx)  # [nb, B]

    root_added_mask = ((x_after_root > 0) & (x_cond.unsqueeze(1) == 0)).unsqueeze(-1).float()
    roll_added_mask = ((x_branch > 0) & (x_after_root == 0)).unsqueeze(-1).float()
    added_oh = F.one_hot((x_branch - 1).clamp(0, 8), num_classes=9).float()

    branch_pred = (
        root_added_mask * (0.30 * added_oh + 0.70 * leaf_probs)
        + roll_added_mask * (0.15 * added_oh + 0.85 * leaf_probs)
        + (1.0 - root_added_mask - roll_added_mask) * leaf_probs
    )

    if beta_clear <= 0.0:
        # Ablation_Unweighted_Rollout_Average_B16: uniform/unweighted average over the B rollout branches
        weights = torch.full((nb, n_branches), 1.0 / float(n_branches), device=device)
    else:
        e_shift = e_inc - e_inc.min(dim=-1, keepdim=True).values
        log_w = 0.25 * log_prior - beta_clear * e_shift
        weights = F.softmax(log_w, dim=-1)  # [nb, B]

    cleared_probs = (weights.unsqueeze(-1).unsqueeze(-1) * branch_pred).sum(dim=1)  # [nb, 81, 9]
    aug_probs = 0.85 * cleared_probs + 0.15 * base_probs
    aug_probs = (1.0 - eps) * aug_probs + (eps / 9.0)
    aug_probs = aug_probs / aug_probs.sum(dim=-1, keepdim=True)

    ess = float((1.0 / (weights * weights).sum(dim=-1).clamp(min=1e-6)).mean().item())
    stats = {
        "mean_branch_energy": float(e_inc.mean().item()),
        "min_branch_energy": float(e_inc.min(dim=-1).values.mean().item()),
        "effective_branch_ess": ess,
    }
    return aug_probs, stats


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
def evaluate_sudoku_anyorder_perplexity_and_rollout(
    arm_name: str,
    model: Optional[AnyOrderSudokuNet],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    unit_idx: torch.Tensor,
    device: torch.device,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    k_random_orders: int = 4,
    n_reveal_blocks: int = 6,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    Evaluates an arm on:
      1. K=4 uniformly random reveal permutations sigma^{(1)}, ..., sigma^{(4)} of the empty cells:
         - rand_order_nll_per_cell (nats)
         - rand_order_ppl = exp(rand_order_nll_per_cell)
         - rand_order_board_nll (total nats across empty cells per board)
         - cross_order_nll_std (mean per-board std of total board NLL across the K=4 random orders)
         - cross_order_cell_nll_std (mean per-board std of per-cell NLL across the K=4 random orders)
      2. Adaptive Minimum-Entropy Reveal Order sigma_{min-H}:
         - min_h_nll_per_cell (nats)
         - min_h_ppl = exp(min_h_nll_per_cell)
         - min_h_autonomous_empty_acc (100% autonomous sequential rollout committing argmax on min-H cells)
         - min_h_autonomous_unit_collisions (unit collisions on autonomously completed boards)
         - one_shot_empty_acc (initial step-0 accuracy on all empty cells)
    """
    nb = x_eval_np.shape[0]
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)
    y_true = torch.tensor(y_eval_np, dtype=torch.long, device=device)
    unit_idx_np = unit_idx.cpu().numpy()

    def get_probs(x_cur: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        if arm_name == "Control_ClueCopyFloor":
            p = torch.full((nb, 81, 9), 1.0 / 9.0, device=device)
            return p, {"mean_branch_energy": 0.0, "min_branch_energy": 0.0, "effective_branch_ess": 1.0}
        if arm_name == "Control_PeerClue_Uniform":
            p = peer_clue_uniform_sudoku_probs(x_cur, unit_idx)
            return p, {"mean_branch_energy": 0.0, "min_branch_energy": 0.0, "effective_branch_ess": 1.0}
        assert model is not None
        return predict_sudoku_anyorder_search(
            model, x_cur, unit_idx, n_branches=n_branches, beta_clear=beta_clear
        )

    # 1. Step-0 initial evaluation on x_init
    p0, step0_stats = get_probs(x_init)
    empty_init_mask = x_eval_np == 0
    clue_init_mask = x_eval_np > 0
    preds_step0 = (p0.argmax(dim=-1) + 1).cpu().numpy()
    preds_step0[clue_init_mask] = x_eval_np[clue_init_mask]
    one_shot_empty_acc = float(((preds_step0 == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))
    one_shot_collisions = float(count_sudoku_unit_collisions(preds_step0, unit_idx_np).mean())

    # 2. Random-Order Autoregressive Joint NLL & Perplexity across K random permutations per board
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
            b_mask = order_blocks[b_idx]  # [nb, 81]
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
    rand_order_board_nll = float(board_order_total_nll.mean())
    cross_order_nll_std = float(np.std(board_order_total_nll, axis=1, ddof=1).mean()) if k_random_orders > 1 else 0.0
    cross_order_cell_nll_std = float(np.std(board_order_cell_nll, axis=1, ddof=1).mean()) if k_random_orders > 1 else 0.0

    # 3. Adaptive Minimum-Entropy Order (sigma_{min-H}) NLL & Perplexity
    x_cur_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_scored_cells = torch.zeros(nb, dtype=torch.float64, device=device)

    for b_idx in range(n_reveal_blocks):
        rem_mask = x_cur_minh == 0  # [nb, 81]
        if not rem_mask.any():
            break
        probs_step, _ = get_probs(x_cur_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)  # [nb, 81]
        rem_counts = rem_mask.sum(dim=-1)
        blocks_left = n_reveal_blocks - b_idx
        if blocks_left > 1:
            k_pick_vec = ((rem_counts + blocks_left - 1) // blocks_left).clamp(min=1)
        else:
            k_pick_vec = rem_counts
        ent_masked = ent + torch.where(rem_mask, torch.zeros_like(ent), torch.full_like(ent, 1e9))
        ranks = ent_masked.argsort(dim=-1).argsort(dim=-1)
        step_mask = (ranks < k_pick_vec.unsqueeze(-1)) & rem_mask

        true_digits = (y_true - 1).clamp(0, 8)
        true_p = probs_step.gather(2, true_digits.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        step_nll = -torch.log(true_p).to(torch.float64) * step_mask.to(torch.float64)
        minh_total_nll = minh_total_nll + step_nll.sum(dim=-1)
        minh_scored_cells = minh_scored_cells + step_mask.to(torch.float64).sum(dim=-1)
        x_cur_minh = torch.where(step_mask, y_true, x_cur_minh)

    min_h_nll_per_cell = float((minh_total_nll / minh_scored_cells.clamp(min=1.0)).mean().item())
    min_h_ppl = float(math.exp(min(min_h_nll_per_cell, 20.0)))
    min_h_board_nll = float(minh_total_nll.mean().item())

    # 4. Autonomous (Zero-Label Test-Time) Sequential Rollout along Minimum-Entropy Order sigma_{min-H}
    x_auto = x_init.clone()
    n_auto_steps = n_reveal_blocks
    for s_idx in range(n_auto_steps):
        rem_mask = x_auto == 0
        if not rem_mask.any():
            break
        probs_auto, _ = get_probs(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        pred_digits = probs_auto.argmax(dim=-1) + 1  # [nb, 81]
        rem_counts = rem_mask.sum(dim=-1)
        steps_left = n_auto_steps - s_idx
        if steps_left > 1:
            k_pick_vec = ((rem_counts + steps_left - 1) // steps_left).clamp(min=1)
        else:
            k_pick_vec = rem_counts
        ent_masked = ent + torch.where(rem_mask, torch.zeros_like(ent), torch.full_like(ent, 1e9))
        ranks = ent_masked.argsort(dim=-1).argsort(dim=-1)
        commit_mask = (ranks < k_pick_vec.unsqueeze(-1)) & rem_mask
        x_auto = torch.where(commit_mask, pred_digits, x_auto)

    auto_np = x_auto.cpu().numpy()
    auto_empty_acc = float(((auto_np == y_eval_np) & empty_init_mask).sum() / max(empty_init_mask.sum(), 1))
    auto_clue_acc = float(((auto_np == y_eval_np) & clue_init_mask).sum() / max(clue_init_mask.sum(), 1))
    auto_collisions = float(count_sudoku_unit_collisions(auto_np, unit_idx_np).mean())

    return {
        "n_branches": n_branches,
        "beta_clear": beta_clear,
        "rand_order_nll_per_cell": rand_order_nll_per_cell,
        "rand_order_ppl": rand_order_ppl,
        "rand_order_board_nll": rand_order_board_nll,
        "cross_order_nll_std": cross_order_nll_std,
        "cross_order_cell_nll_std": cross_order_cell_nll_std,
        "min_h_nll_per_cell": min_h_nll_per_cell,
        "min_h_ppl": min_h_ppl,
        "min_h_board_nll": min_h_board_nll,
        "adaptive_vs_random_nll_gain": float(rand_order_nll_per_cell - min_h_nll_per_cell),
        "min_h_autonomous_empty_acc": auto_empty_acc,
        "min_h_autonomous_clue_acc": auto_clue_acc,
        "min_h_autonomous_unit_collisions": auto_collisions,
        "one_shot_empty_acc": one_shot_empty_acc,
        "one_shot_unit_collisions": one_shot_collisions,
        "search_diagnostics": step0_stats,
    }


# ---------------------------------------------------------------------------
# 2. Any-Order N-Queens World Model & Pure-Neural Inconsistency Tree Search
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
    if len(xs) < count:
        raise RuntimeError(f"Could only sample {len(xs)}/{count} disjoint N-Queens dev boards for N={n}")
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
    clue_ok = np.zeros(bsz, dtype=np.float64)
    for b in range(bsz):
        c = cols_pred[b]
        cl = clues[b]
        mask = cl >= 0
        clue_ok[b] = float((c[mask] == cl[mask]).mean()) if mask.any() else 1.0
        cc, dc = 0, 0
        for r1 in range(n):
            for r2 in range(r1 + 1, n):
                if c[r1] == c[r2]:
                    cc += 1
                if abs(r1 - r2) == abs(int(c[r1]) - int(c[r2])):
                    dc += 1
        total_coll[b] = cc + dc
        valid_board[b] = 1.0 if (cc + dc == 0 and clue_ok[b] == 1.0) else 0.0
    return {
        "mean_collisions": float(total_coll.mean()),
        "valid_board_rate": float(valid_board.mean()),
        "clue_acc": float(clue_ok.mean()),
    }


class AnyOrderNQueensNet(nn.Module):
    """
    Size-invariant Any-Order 2D Cell-Grid World Model q_theta(X_r | x_{S_t}) on N-Queens (N=8, 10).
    All weights initialized from N(0, 0.02^2). Zero solver or heuristic code in forward.
    """

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

    def forward(self, clues: torch.Tensor, n_segments: int = 5, n_loops: int = 2) -> List[torch.Tensor]:
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
        return seg_logits


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


def compute_pure_neural_nqueens_inconsistency_energy(
    leaf_probs: torch.Tensor,
    base_probs: torch.Tensor,
    x_branch: torch.Tensor,
) -> torch.Tensor:
    nb, n_branches, n, _ = leaf_probs.shape
    col_sum = leaf_probs.sum(dim=2)  # [nb, B, N]
    col_energy = ((col_sum - 1.0) ** 2).sum(dim=-1)  # [nb, B]

    diag_energy = torch.zeros(nb, n_branches, device=leaf_probs.device)
    for dr in range(1, n):
        p_top = leaf_probs[:, :, :-dr, :]
        p_bot = leaf_probs[:, :, dr:, :]
        diag_energy = diag_energy + (p_top[:, :, :, :-dr] * p_bot[:, :, :, dr:]).sum(dim=(2, 3))
        diag_energy = diag_energy + (p_top[:, :, :, dr:] * p_bot[:, :, :, :-dr]).sum(dim=(2, 3))

    uncond_mask = (x_branch < 0).float()  # [nb, B, N]
    uncond_denom = uncond_mask.sum(dim=-1).clamp(min=1.0)
    max_p = leaf_probs.max(dim=-1).values.clamp(min=1e-6)
    deplete_energy = ((-torch.log(max_p) * uncond_mask).sum(dim=-1) / uncond_denom) * 4.0

    base_exp = base_probs.unsqueeze(1).expand_as(leaf_probs).clamp(min=1e-6)
    leaf_cl = leaf_probs.clamp(min=1e-6)
    kl_row = (leaf_cl * (torch.log(leaf_cl) - torch.log(base_exp))).sum(dim=-1)
    path_kl = (kl_row * uncond_mask).sum(dim=-1) / uncond_denom

    return col_energy + 2.0 * diag_energy + deplete_energy + 0.25 * path_kl


@torch.no_grad()
def predict_nqueens_anyorder_search(
    model: AnyOrderNQueensNet,
    x_cond: torch.Tensor,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    rollout_depth: int = 2,
    n_segments: int = 4,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    model.eval()
    nb, n = x_cond.shape
    device = x_cond.device
    base_logits = model(x_cond, n_segments=n_segments, n_loops=2)[-1]
    base_probs = F.softmax(base_logits, dim=-1)  # [nb, N, N]
    eps = 1e-4

    if n_branches <= 0:
        out_probs = (1.0 - eps) * base_probs + (eps / float(n))
        return out_probs, {"mean_branch_energy": 0.0, "min_branch_energy": 0.0}

    r_bits = {4: 2, 16: 4, 32: 5}.get(n_branches, 4)
    r_bits = min(r_bits, n)
    bit_table = torch.tensor(
        [[(b >> bit) & 1 for bit in range(r_bits)] for b in range(n_branches)],
        dtype=torch.long,
        device=device,
    )
    p_cl = base_probs.clamp(min=1e-7)
    row_ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)  # [nb, N]
    top2_vals, top2_idx = base_probs.topk(2, dim=-1)  # [nb, N, 2]
    is_empty = x_cond < 0
    score = row_ent + torch.where(is_empty, torch.zeros_like(row_ent), torch.full_like(row_ent, 1e6))
    root_rows = score.topk(r_bits, dim=-1, largest=False).indices  # [nb, r_bits]

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
        step_logits = model(flat_x, n_segments=max(3, n_segments - 1), n_loops=2)[-1]
        step_probs = F.softmax(step_logits, dim=-1).view(nb, n_branches, n, n)
        sp_cl = step_probs.clamp(min=1e-7)
        sp_ent = -(sp_cl * torch.log(sp_cl)).sum(dim=-1)
        rem_empty = x_branch < 0
        sp_score = sp_ent + torch.where(rem_empty, torch.zeros_like(sp_ent), torch.full_like(sp_ent, 1e6))
        next_row = sp_score.argmin(dim=-1, keepdim=True)  # [nb, B, 1]
        best_col = step_probs.argmax(dim=-1).gather(2, next_row)
        was_empty = rem_empty.gather(2, next_row)
        cur_val = x_branch.gather(2, next_row)
        upd_val = torch.where(was_empty, best_col, cur_val)
        x_branch.scatter_(2, next_row, upd_val)

    leaf_logits = model(x_branch.reshape(nb * n_branches, n), n_segments=n_segments, n_loops=2)[-1]
    leaf_probs = F.softmax(leaf_logits, dim=-1).view(nb, n_branches, n, n)
    e_inc = compute_pure_neural_nqueens_inconsistency_energy(leaf_probs, base_probs, x_branch)

    root_added_mask = ((x_after_root >= 0) & (x_cond.unsqueeze(1) < 0)).unsqueeze(-1).float()
    roll_added_mask = ((x_branch >= 0) & (x_after_root < 0)).unsqueeze(-1).float()
    added_oh = F.one_hot(x_branch.clamp(0, n - 1), num_classes=n).float()
    branch_pred = (
        root_added_mask * (0.30 * added_oh + 0.70 * leaf_probs)
        + roll_added_mask * (0.15 * added_oh + 0.85 * leaf_probs)
        + (1.0 - root_added_mask - roll_added_mask) * leaf_probs
    )

    if beta_clear <= 0.0:
        weights = torch.full((nb, n_branches), 1.0 / float(n_branches), device=device)
    else:
        e_shift = e_inc - e_inc.min(dim=-1, keepdim=True).values
        log_w = 0.25 * log_prior - beta_clear * e_shift
        weights = F.softmax(log_w, dim=-1)

    cleared_probs = (weights.unsqueeze(-1).unsqueeze(-1) * branch_pred).sum(dim=1)
    aug_probs = 0.85 * cleared_probs + 0.15 * base_probs
    aug_probs = (1.0 - eps) * aug_probs + (eps / float(n))
    aug_probs = aug_probs / aug_probs.sum(dim=-1, keepdim=True)
    return aug_probs, {
        "mean_branch_energy": float(e_inc.mean().item()),
        "min_branch_energy": float(e_inc.min(dim=-1).values.mean().item()),
    }


@torch.no_grad()
def evaluate_nqueens_anyorder_perplexity_and_rollout(
    arm_name: str,
    model: Optional[AnyOrderNQueensNet],
    x_eval_np: np.ndarray,
    y_eval_np: np.ndarray,
    multi_ys: List[np.ndarray],
    device: torch.device,
    n_branches: int = 0,
    beta_clear: float = 2.5,
    k_random_orders: int = 4,
    seed: int = 101,
) -> Dict[str, Any]:
    nb, n = x_eval_np.shape
    x_init = torch.tensor(x_eval_np, dtype=torch.long, device=device)

    def get_probs(x_cur: torch.Tensor) -> torch.Tensor:
        if arm_name == "Control_ClueCopyFloor":
            return torch.full((nb, n, n), 1.0 / float(n), device=device)
        if arm_name == "Control_PeerClue_Uniform":
            p_att = nqueens_peer_attack_mask(x_cur)
            allowed = (1.0 - p_att).clamp(min=0.0)
            s_allow = allowed.sum(dim=-1, keepdim=True)
            unif = torch.full_like(allowed, 1.0 / float(n))
            p = torch.where(s_allow > 0.5, allowed / s_allow.clamp(min=1.0), unif)
            eps = 1e-4
            p = (1.0 - eps) * p + (eps / float(n))
            return p / p.sum(dim=-1, keepdim=True)
        assert model is not None
        p, _ = predict_nqueens_anyorder_search(model, x_cur, n_branches=n_branches, beta_clear=beta_clear)
        return p

    # Select for each board the valid solution in multi_ys[i] closest to the model's step-0 prediction
    p_step0 = get_probs(x_init).cpu().numpy()
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

    # 1. Exact step-by-step random-order NLL across K random permutations of unconditioned rows
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
            probs_step = get_probs(x_cur)
            r_safe = r_idx.clamp(min=0)
            row_p = probs_step.gather(1, r_safe.view(nb, 1, 1).expand(nb, 1, n)).squeeze(1)  # [nb, N]
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

    # 2. Adaptive Minimum-Entropy Row Reveal Order sigma_{min-H}: NLL & Autonomous Rollout
    x_minh = x_init.clone()
    minh_total_nll = torch.zeros(nb, dtype=torch.float64, device=device)
    minh_counts = torch.zeros(nb, dtype=torch.float64, device=device)

    for _ in range(max_empty):
        rem = x_minh < 0
        if not rem.any():
            break
        probs_step = get_probs(x_minh)
        p_cl = probs_step.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        ent_masked = ent + torch.where(rem, torch.zeros_like(ent), torch.full_like(ent, 1e6))
        pick_r = ent_masked.argmin(dim=-1)  # [nb]
        has_rem = rem.any(dim=-1)
        row_p = probs_step.gather(1, pick_r.view(nb, 1, 1).expand(nb, 1, n)).squeeze(1)
        true_c = y_true.gather(1, pick_r.unsqueeze(-1)).squeeze(-1).clamp(0, n - 1)
        true_p = row_p.gather(1, true_c.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
        minh_total_nll = minh_total_nll + (-torch.log(true_p).to(torch.float64) * has_rem.to(torch.float64))
        minh_counts = minh_counts + has_rem.to(torch.float64)
        cur_v = x_minh.gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        upd_v = torch.where(has_rem, true_c, cur_v)
        x_minh.scatter_(1, pick_r.unsqueeze(-1), upd_v.unsqueeze(-1))

    min_h_nll_per_cell = float((minh_total_nll / minh_counts.clamp(min=1.0)).mean().item())
    min_h_ppl = float(math.exp(min(min_h_nll_per_cell, 20.0)))

    # Autonomous Sequential Solve along sigma_{min-H}
    x_auto = x_init.clone()
    for _ in range(max_empty):
        rem = x_auto < 0
        if not rem.any():
            break
        probs_auto = get_probs(x_auto)
        p_cl = probs_auto.clamp(min=1e-7)
        ent = -(p_cl * torch.log(p_cl)).sum(dim=-1)
        ent_masked = ent + torch.where(rem, torch.zeros_like(ent), torch.full_like(ent, 1e6))
        pick_r = ent_masked.argmin(dim=-1)
        has_rem = rem.any(dim=-1)
        pred_cols = probs_auto.argmax(dim=-1).gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        cur_v = x_auto.gather(1, pick_r.unsqueeze(-1)).squeeze(-1)
        upd_v = torch.where(has_rem, pred_cols, cur_v)
        x_auto.scatter_(1, pick_r.unsqueeze(-1), upd_v.unsqueeze(-1))

    auto_metrics = count_nqueens_metrics(x_auto.cpu().numpy(), x_eval_np)
    return {
        "n_branches": n_branches,
        "beta_clear": beta_clear,
        "rand_order_nll_per_cell": rand_order_nll_per_cell,
        "rand_order_ppl": rand_order_ppl,
        "cross_order_nll_std": cross_order_nll_std,
        "min_h_nll_per_cell": min_h_nll_per_cell,
        "min_h_ppl": min_h_ppl,
        "adaptive_vs_random_nll_gain": float(rand_order_nll_per_cell - min_h_nll_per_cell),
        "min_h_autonomous_collisions": auto_metrics["mean_collisions"],
        "min_h_valid_board_rate": auto_metrics["valid_board_rate"],
        "min_h_clue_acc": auto_metrics["clue_acc"],
    }


# ---------------------------------------------------------------------------
# 3. Main Track E Exploratory Runner
# ---------------------------------------------------------------------------

def run_explore_aom_ppl(args) -> Dict[str, Any]:
    train_csv_path, data_hashes = ensure_and_verify_sudoku_extreme_train_only(args.sudoku_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    seeds = [101] if args.smoke_cpu else [101, 102, 103]

    # Strictly development splits from train.csv only (zero touch of confirmatory test.csv)
    x_tr_all, y_tr_all = read_sudoku_csv(train_csv_path, 5700, seed=101, random_subset=False)
    x_tr_full = x_tr_all[0:4000].reshape(-1, 81).astype(np.int64)
    y_tr_full = y_tr_all[0:4000].reshape(-1, 81).astype(np.int64)
    x_dev_full = x_tr_all[5500:5628].reshape(-1, 81).astype(np.int64)
    y_dev_full = y_tr_all[5500:5628].reshape(-1, 81).astype(np.int64)

    x_tr_smoke = x_tr_all[4000:4256].reshape(-1, 81).astype(np.int64)
    y_tr_smoke = y_tr_all[4000:4256].reshape(-1, 81).astype(np.int64)
    x_dev_smoke = x_tr_all[5628:5644].reshape(-1, 81).astype(np.int64)
    y_dev_smoke = y_tr_all[5628:5644].reshape(-1, 81).astype(np.int64)

    x_tr_np = x_tr_smoke if args.smoke_cpu else x_tr_full
    y_tr_np = y_tr_smoke if args.smoke_cpu else y_tr_full
    x_dev_np = x_dev_smoke if args.smoke_cpu else x_dev_full
    y_dev_np = y_dev_smoke if args.smoke_cpu else y_dev_full

    sudoku_train_dev_overlap = len({tuple(r) for r in x_tr_full} & {tuple(r) for r in x_dev_full})
    sudoku_smoke_dev_overlap = len({tuple(r) for r in x_dev_full} & {tuple(r) for r in x_dev_smoke})

    # N-Queens 10x10 and 8x8 development splits (using only development orbits orbit_keys[:64], zero touch of [64:])
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
    nq10_dev_x, nq10_dev_y, nq10_dev_multi, dev10_orbs = sample_nqueens_dev_split(
        nq10_dev_sols, sols10, 10, 12 if args.smoke_cpu else 48, tr10_orbs, 2811
    )

    sols8 = solve_all_nqueens(8, max_sols=400)
    nq8_dev_x, nq8_dev_y, nq8_dev_multi, _ = sample_nqueens_dev_split(
        sols8, sols8, 8, 12 if args.smoke_cpu else 48, set(), 2808
    )

    sudoku_steps = 30 if args.smoke_cpu else args.sudoku_steps
    nq_steps = 25 if args.smoke_cpu else args.nq_steps
    hidden_size = 64 if args.smoke_cpu else 96
    k_orders = 2 if args.smoke_cpu else 4
    n_blocks = 4 if args.smoke_cpu else 6
    unit_idx = build_sudoku_unit_indices(device)

    total_label_access: Dict[str, int] = {
        "Control_Random_Network": 0,
        "Control_Untrained_Init": 0,
        "Control_ClueCopyFloor": 0,
        "Control_PeerClue_Uniform": 0,
        "LabelFree_AnyOrder": 0,
        "MaskedTrajectory_AnyOrder_OurTrain": 0,
        "NQ_LabelFree_AnyOrder": 0,
        "NQ_MaskedTrajectory_AnyOrder_OurTrain": 0,
    }

    per_seed: Dict[str, Any] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        watched_y_sudoku = WatchedLabelArray(y_tr_np)
        watched_y_nq10 = WatchedLabelArray(nq10_tr_y)

        sudoku_arms: Dict[str, Any] = {}

        # Rule 9.1: Control_Random_Network runs through the exact same B=16 search-augmentation pipeline
        torch.manual_seed(seed + 9000)
        random_sudoku_net = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        sudoku_arms["Control_Random_Network"] = evaluate_sudoku_anyorder_perplexity_and_rollout(
            "Control_Random_Network",
            random_sudoku_net,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=16,
            beta_clear=2.5,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        torch.manual_seed(seed)
        untrained_sudoku_net = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        sudoku_arms["Control_Untrained_Init"] = evaluate_sudoku_anyorder_perplexity_and_rollout(
            "Control_Untrained_Init",
            untrained_sudoku_net,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            beta_clear=0.0,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        sudoku_arms["Control_ClueCopyFloor"] = evaluate_sudoku_anyorder_perplexity_and_rollout(
            "Control_ClueCopyFloor",
            None,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            beta_clear=0.0,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        sudoku_arms["Control_PeerClue_Uniform"] = evaluate_sudoku_anyorder_perplexity_and_rollout(
            "Control_PeerClue_Uniform",
            None,
            x_dev_np,
            y_dev_np,
            unit_idx,
            device,
            n_branches=0,
            beta_clear=0.0,
            k_random_orders=k_orders,
            n_reveal_blocks=n_blocks,
            seed=seed,
        )

        # -------------------------------------------------------------------
        # Regime 1: Label-Free Any-Order World Model (0 ground-truth labels accessed)
        # -------------------------------------------------------------------
        before_lf = watched_y_sudoku.access_count
        torch.manual_seed(seed)
        lf_model = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        lf_opt = torch.optim.AdamW(lf_model.parameters(), lr=4e-3, weight_decay=1e-4)
        lf_model.train()

        for _ in range(sudoku_steps):
            b_idx = np.random.choice(len(x_tr_np), size=32, replace=True)
            xb_clues = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)

            step_logits_0 = lf_model(xb_clues, n_steps=5)
            loss_0 = sum(
                ((i + 1.0) / 15.0) * sudoku_anyorder_birkhoff_contradiction_loss(lg, xb_clues, unit_idx)
                for i, lg in enumerate(step_logits_0)
            )

            with torch.no_grad():
                p0 = F.softmax(step_logits_0[-1], dim=-1)
                p0_cl = p0.clamp(min=1e-7)
                ent0 = -(p0_cl * torch.log(p0_cl)).sum(dim=-1)
                is_emp = xb_clues == 0
                score0 = ent0 + torch.where(is_emp, torch.zeros_like(ent0), torch.full_like(ent0, 1e6))
                k_cond = int(np.random.randint(4, 18))
                cond_cells = score0.topk(k_cond, dim=-1, largest=False).indices
                pred_d = p0.argmax(dim=-1) + 1
                xb_st = xb_clues.clone()
                was_emp = is_emp.gather(1, cond_cells)
                cur_v = xb_st.gather(1, cond_cells)
                xb_st.scatter_(1, cond_cells, torch.where(was_emp, pred_d.gather(1, cond_cells), cur_v))

            step_logits_t = lf_model(xb_st, n_steps=4)
            loss_t = sudoku_anyorder_birkhoff_contradiction_loss(step_logits_t[-1], xb_st, unit_idx)

            rem_mask = (xb_st == 0).unsqueeze(-1).float()
            p_target = F.softmax(step_logits_t[-1].detach(), dim=-1).clamp(min=1e-6)
            log_p_base = F.log_softmax(step_logits_0[-1], dim=-1)
            kl_consist = -(p_target * log_p_base * rem_mask).sum(dim=-1).mean()

            loss = loss_0 + 0.5 * loss_t + 0.2 * kl_consist
            lf_opt.zero_grad()
            loss.backward()
            lf_opt.step()

        total_label_access["LabelFree_AnyOrder"] += watched_y_sudoku.access_count - before_lf

        for arm_key, b_val, beta_val in [
            ("Base_AnyOrder_NoSearch_B0", 0, 0.0),
            ("Ablation_Unweighted_Rollout_Average_B16", 16, 0.0),
            ("Augmented_AnyOrder_InconsistencySearch_B4", 4, 2.5),
            ("Augmented_AnyOrder_InconsistencySearch_B16", 16, 2.5),
            ("Augmented_AnyOrder_InconsistencySearch_B32", 32, 2.5),
        ]:
            sudoku_arms[arm_key] = evaluate_sudoku_anyorder_perplexity_and_rollout(
                arm_key,
                lf_model,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=b_val,
                beta_clear=beta_val,
                k_random_orders=k_orders,
                n_reveal_blocks=n_blocks,
                seed=seed,
            )

        # -------------------------------------------------------------------
        # Regime 2: Masked-Trajectory Any-Order World Model (train.csv[0:4000])
        # -------------------------------------------------------------------
        before_mt = watched_y_sudoku.access_count
        torch.manual_seed(seed)
        mt_model = AnyOrderSudokuNet(hidden_size=hidden_size, device=device).to(device)
        mt_opt = torch.optim.AdamW(mt_model.parameters(), lr=4e-3, weight_decay=1e-4)
        mt_model.train()

        for _ in range(sudoku_steps):
            b_idx = np.random.choice(len(x_tr_np), size=32, replace=True)
            xb_clues = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)
            yb_sol = torch.tensor(watched_y_sudoku[b_idx], dtype=torch.long, device=device)

            reveal_prob = torch.rand((32, 1), device=device) * 0.85
            reveal_mask = (torch.rand((32, 81), device=device) < reveal_prob) & (xb_clues == 0)
            xb_st = torch.where(reveal_mask, yb_sol, xb_clues)
            target_mask = xb_st == 0
            if not target_mask.any():
                target_mask = xb_clues == 0

            step_logits = mt_model(xb_st, n_steps=5)
            loss = torch.tensor(0.0, device=device)
            for i, lg in enumerate(step_logits):
                w_i = (i + 1.0) / 15.0
                ce_uncond = F.cross_entropy(lg[target_mask], (yb_sol[target_mask] - 1).clamp(0, 8))
                birk_reg = sudoku_anyorder_birkhoff_contradiction_loss(lg, xb_st, unit_idx)
                loss = loss + w_i * (ce_uncond + 0.25 * birk_reg)

            mt_opt.zero_grad()
            loss.backward()
            mt_opt.step()

        total_label_access["MaskedTrajectory_AnyOrder_OurTrain"] += watched_y_sudoku.access_count - before_mt

        for arm_key, b_val, beta_val in [
            ("MaskedTraj_AnyOrder_NoSearch_B0", 0, 0.0),
            ("MaskedTraj_Unweighted_Rollout_Average_B16", 16, 0.0),
            ("MaskedTraj_InconsistencySearch_B4", 4, 2.5),
            ("MaskedTraj_InconsistencySearch_B16", 16, 2.5),
            ("MaskedTraj_InconsistencySearch_B32", 32, 2.5),
        ]:
            ev = evaluate_sudoku_anyorder_perplexity_and_rollout(
                arm_key,
                mt_model,
                x_dev_np,
                y_dev_np,
                unit_idx,
                device,
                n_branches=b_val,
                beta_clear=beta_val,
                k_random_orders=k_orders,
                n_reveal_blocks=n_blocks,
                seed=seed,
            )
            ev["training_regime"] = f"our training, {sudoku_steps} steps on train.csv[0:4000] (EXPLORATORY)"
            sudoku_arms[arm_key] = ev

        # -------------------------------------------------------------------
        # N-Queens (10x10 and 8x8 Development Splits)
        # -------------------------------------------------------------------
        nq_arms: Dict[str, Any] = {"10x10_dev": {}, "8x8_dev": {}}

        torch.manual_seed(seed + 9000)
        nq_rand_net = AnyOrderNQueensNet(d_model=48).to(device)
        torch.manual_seed(seed)
        nq_untr_net = AnyOrderNQueensNet(d_model=48).to(device)

        # Train Label-Free Any-Order N-Queens (0 labels accessed)
        before_nq_lf = watched_y_nq10.access_count
        torch.manual_seed(seed)
        nq_lf_model = AnyOrderNQueensNet(d_model=48).to(device)
        nq_lf_opt = torch.optim.AdamW(nq_lf_model.parameters(), lr=3e-3, weight_decay=1e-4)
        nq_lf_model.train()
        for _ in range(nq_steps):
            idx = np.random.choice(len(nq10_tr_x), size=32, replace=True)
            xb = torch.tensor(nq10_tr_x[idx], dtype=torch.long, device=device)
            seg_logits = nq_lf_model(xb, n_segments=5, n_loops=2)
            loss_0 = sum(birkhoff_nqueens_anyorder_loss(lg, xb) for lg in seg_logits) / len(seg_logits)

            with torch.no_grad():
                p0 = F.softmax(seg_logits[-1], dim=-1)
                ent0 = -(p0.clamp(min=1e-7) * torch.log(p0.clamp(min=1e-7))).sum(dim=-1)
                is_emp = xb < 0
                score0 = ent0 + torch.where(is_emp, torch.zeros_like(ent0), torch.full_like(ent0, 1e6))
                pick_r = score0.topk(2, dim=-1, largest=False).indices
                pred_c = p0.argmax(dim=-1)
                xb_st = xb.clone()
                was_emp = is_emp.gather(1, pick_r)
                cur_v = xb_st.gather(1, pick_r)
                xb_st.scatter_(1, pick_r, torch.where(was_emp, pred_c.gather(1, pick_r), cur_v))

            seg_logits_t = nq_lf_model(xb_st, n_segments=4, n_loops=2)
            loss_t = birkhoff_nqueens_anyorder_loss(seg_logits_t[-1], xb_st)
            loss = loss_0 + 0.5 * loss_t
            nq_lf_opt.zero_grad()
            loss.backward()
            nq_lf_opt.step()

        total_label_access["NQ_LabelFree_AnyOrder"] += watched_y_nq10.access_count - before_nq_lf

        # Train Masked-Trajectory Any-Order N-Queens
        before_nq_mt = watched_y_nq10.access_count
        torch.manual_seed(seed)
        nq_mt_model = AnyOrderNQueensNet(d_model=48).to(device)
        nq_mt_opt = torch.optim.AdamW(nq_mt_model.parameters(), lr=3e-3, weight_decay=1e-4)
        nq_mt_model.train()
        for _ in range(nq_steps):
            idx = np.random.choice(len(nq10_tr_x), size=32, replace=True)
            xb = torch.tensor(nq10_tr_x[idx], dtype=torch.long, device=device)
            yb = torch.tensor(watched_y_nq10[idx], dtype=torch.long, device=device)
            rev_prob = torch.rand((32, 1), device=device) * 0.75
            rev_mask = (torch.rand((32, 10), device=device) < rev_prob) & (xb < 0)
            xb_st = torch.where(rev_mask, yb, xb)
            t_mask = xb_st < 0
            if not t_mask.any():
                t_mask = xb < 0
            seg_logits = nq_mt_model(xb_st, n_segments=5, n_loops=2)
            loss = sum(
                F.cross_entropy(lg[t_mask], yb[t_mask]) + 0.25 * birkhoff_nqueens_anyorder_loss(lg, xb_st)
                for lg in seg_logits
            ) / len(seg_logits)
            nq_mt_opt.zero_grad()
            loss.backward()
            nq_mt_opt.step()

        total_label_access["NQ_MaskedTrajectory_AnyOrder_OurTrain"] += watched_y_nq10.access_count - before_nq_mt

        for split_name, x_split, y_split, multi_split in [
            ("10x10_dev", nq10_dev_x, nq10_dev_y, nq10_dev_multi),
            ("8x8_dev", nq8_dev_x, nq8_dev_y, nq8_dev_multi),
        ]:
            nq_arms[split_name]["Control_Random_Network"] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                "Control_Random_Network",
                nq_rand_net,
                x_split,
                y_split,
                multi_split,
                device,
                n_branches=16,
                beta_clear=2.5,
                k_random_orders=k_orders,
                seed=seed,
            )
            nq_arms[split_name]["Control_Untrained_Init"] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                "Control_Untrained_Init",
                nq_untr_net,
                x_split,
                y_split,
                multi_split,
                device,
                n_branches=0,
                beta_clear=0.0,
                k_random_orders=k_orders,
                seed=seed,
            )
            nq_arms[split_name]["Control_ClueCopyFloor"] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                "Control_ClueCopyFloor",
                None,
                x_split,
                y_split,
                multi_split,
                device,
                n_branches=0,
                beta_clear=0.0,
                k_random_orders=k_orders,
                seed=seed,
            )
            nq_arms[split_name]["Control_PeerClue_Uniform"] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                "Control_PeerClue_Uniform",
                None,
                x_split,
                y_split,
                multi_split,
                device,
                n_branches=0,
                beta_clear=0.0,
                k_random_orders=k_orders,
                seed=seed,
            )
            for arm_key, b_val, beta_val in [
                ("Base_AnyOrder_NoSearch_B0", 0, 0.0),
                ("Ablation_Unweighted_Rollout_Average_B16", 16, 0.0),
                ("Augmented_AnyOrder_InconsistencySearch_B4", 4, 2.5),
                ("Augmented_AnyOrder_InconsistencySearch_B16", 16, 2.5),
                ("Augmented_AnyOrder_InconsistencySearch_B32", 32, 2.5),
            ]:
                nq_arms[split_name][arm_key] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                    arm_key,
                    nq_lf_model,
                    x_split,
                    y_split,
                    multi_split,
                    device,
                    n_branches=b_val,
                    beta_clear=beta_val,
                    k_random_orders=k_orders,
                    seed=seed,
                )
            for arm_key, b_val, beta_val in [
                ("MaskedTraj_AnyOrder_NoSearch_B0", 0, 0.0),
                ("MaskedTraj_InconsistencySearch_B16", 16, 2.5),
            ]:
                nq_arms[split_name][arm_key] = evaluate_nqueens_anyorder_perplexity_and_rollout(
                    arm_key,
                    nq_mt_model,
                    x_split,
                    y_split,
                    multi_split,
                    device,
                    n_branches=b_val,
                    beta_clear=beta_val,
                    k_random_orders=k_orders,
                    seed=seed,
                )

        per_seed[str(seed)] = {
            "sudoku_extreme_dev": sudoku_arms,
            "nqueens_dev": nq_arms,
        }

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    summary = summarize_exploratory_results(per_seed, seeds)

    return {
        "track": "EXPLORATORY (PROTOCOL.md §10.1)",
        "experiment": "explore_aom_search_ppl",
        "commit": args.commit,
        "runner_sha256": runner_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "dataset_sha256": data_hashes,
        "splits": {
            "sudoku_train": "train.csv[4000:4256]" if args.smoke_cpu else "train.csv[0:4000]",
            "sudoku_dev_eval": "train.csv[5628:5644]" if args.smoke_cpu else "train.csv[5500:5628]",
            "sudoku_train_dev_overlap": sudoku_train_dev_overlap,
            "sudoku_smoke_dev_overlap": sudoku_smoke_dev_overlap,
            "test_csv_touched": False,
        },
        "label_access_counts": total_label_access,
        "summary": summary,
        "per_seed": per_seed,
    }


def summarize_exploratory_results(per_seed: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    s_keys = [str(s) for s in seeds]
    sudoku_arm_names = list(per_seed[s_keys[0]]["sudoku_extreme_dev"].keys())
    sudoku_summary: Dict[str, Any] = {}
    metrics = [
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "cross_order_nll_std",
        "min_h_nll_per_cell",
        "min_h_ppl",
        "adaptive_vs_random_nll_gain",
        "min_h_autonomous_empty_acc",
        "min_h_autonomous_unit_collisions",
        "one_shot_empty_acc",
    ]
    for arm in sudoku_arm_names:
        sudoku_summary[arm] = {
            m: round(float(np.mean([per_seed[s]["sudoku_extreme_dev"][arm][m] for s in s_keys])), 5)
            for m in metrics
        }

    nq_summary: Dict[str, Any] = {}
    nq_metrics = [
        "rand_order_nll_per_cell",
        "rand_order_ppl",
        "cross_order_nll_std",
        "min_h_nll_per_cell",
        "min_h_ppl",
        "min_h_autonomous_collisions",
        "min_h_valid_board_rate",
    ]
    for split_name in ("10x10_dev", "8x8_dev"):
        nq_summary[split_name] = {}
        for arm in per_seed[s_keys[0]]["nqueens_dev"][split_name].keys():
            nq_summary[split_name][arm] = {
                m: round(float(np.mean([per_seed[s]["nqueens_dev"][split_name][arm][m] for s in s_keys])), 5)
                for m in nq_metrics
            }

    return {
        "status": "EXPLORATORY_SMOKE" if len(seeds) == 1 else "EXPLORATORY_DEV",
        "sudoku_extreme_dev_mean": sudoku_summary,
        "nqueens_dev_mean": nq_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--commit", type=str, default="HEAD")
    parser.add_argument("--sudoku-dir", type=str, default="/home/agent/data/datasets/sudoku-extreme")
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--sudoku-steps", type=int, default=450)
    parser.add_argument("--nq-steps", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()
    results = run_explore_aom_ppl(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(results["summary"], indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
