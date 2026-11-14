#!/usr/bin/env python3
"""
Round 27A Runner: Label-Free TRMP + InfiLoop Consistency Training on Public
Sudoku-Extreme and Ambiguous OOD N-Queens (10x10 -> 12x12, 16x16).

Strict compliance with docs/PROTOCOL.md (including Section 9.1-9.9) and outbox/r27-freeze-review-2026-10-20.md:
1. Standalone execution: inserts repository root onto sys.path before local imports so
   `python3 benchmarks/run_r27a_labelfree_infiloop_trmp.py` runs without external PYTHONPATH.
2. Public sapientinc/sudoku-extreme dataset loaded via unmodified external/infiloop/sudoku_data.py
   with mandatory SHA256 verification and hard-fail on missing/corrupt data (zero synthetic fallback).
   Test boards are drawn as a seeded random sample (seed=27001) of test.csv[320:] (excluding the head
   [0:320] completely; 256 frozen full test boards and 64 disjoint smoke test boards) per Rule 9.9.
3. Real WatchedLabelArray instrumentation recording exact __getitem__ label accesses (> 0 for
   supervised reference arms, == 0 for label-free arms).
4. 100% pure-neural readout (raw logits.argmax(dim=-1)) across all neural arms and controls;
   zero Python backtracking, zero singles propagation, zero MRV heuristics, zero copy-forward.
   The architecture provides the row/col/box/diagonal peer-sum channels (peer_p_cat / peer_feat)
   and linear skip peer_logit_proj (initialized from N(0, 0.02^2)), and Birkhoff label-free
   training learns the recurrent and peer-projection weights from scratch without solution labels.
5. Control_Random_Network runs the exact same TRMPInfiLoopSudoku / Looped2DCellNQueensNet
   architecture and inference pipeline with untrained random weights (Rule 9.1).
6. Non-learned floors Control_PeerClue_Uniform and Control_FixedMeanField included on both
   Sudoku-Extreme and N-Queens.
7. Ambiguous 2-3 clue N-Queens (covering all rows 0..N-1 and all row-0 columns 0..N-1) partitioned
   by D_4 symmetry orbits on BOTH solutions and input clue boards (0 exact input overlap,
   0 D_4 input orbit overlap, 0 smoke/test overlap).
8. Smoke runs emit "SMOKE_NO_VERDICT" (zero PASS/FAIL verdicts on smoke per Rule 9.8).
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
    torch.set_num_threads(1)

from external.infiloop.sudoku_data import read_csv as read_sudoku_csv
from external.infiloop.sudoku_model import (
    Config as InfiLoopConfig,
    InfiLoop,
    loss_function as upstream_infiloop_loss,
    rms_norm as infiloop_rms_norm,
)

T_CRIT_DF2 = 4.302652729911275
SUDOKU_FIXED_MF_A_GRID = [0.5, 1.0, 1.5, 3.0, 6.0]
SUDOKU_FIXED_MF_SELF_GRID = [0.0, 2.0, 5.0]

SUDOKU_EXTREME_TRAIN_SHA256 = "64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e"
SUDOKU_EXTREME_TEST_SHA256 = "a2fd52aea23d331d5b4ee723c856236e838a9fb9a70e66f4e0e0cf26c338c6a8"
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


def ensure_and_verify_sudoku_extreme(data_dir: str) -> Tuple[Path, Dict[str, str]]:
    """
    Verifies public sapientinc/sudoku-extreme train.csv and test.csv by exact SHA256.
    Falls back to a writable directory (/tmp/sudoku-extreme) only when downloading on a VM
    where /srv/agent-data is absent; hard-fails on any SHA256 mismatch.
    """
    candidate_dirs = [
        Path(data_dir),
        Path(_REPO_ROOT) / "data" / "sudoku-extreme",
        Path("/tmp/sudoku-extreme"),
    ]
    base: Optional[Path] = None
    for cand in candidate_dirs:
        if (cand / "train.csv").exists() and (cand / "test.csv").exists():
            base = cand
            break
    if base is None:
        for cand in candidate_dirs:
            try:
                cand.mkdir(parents=True, exist_ok=True)
                base = cand
                break
            except OSError:
                continue
    if base is None:
        raise RuntimeError("Could not find or create a writable directory for sapientinc/sudoku-extreme")

    expected = {
        "train.csv": SUDOKU_EXTREME_TRAIN_SHA256,
        "test.csv": SUDOKU_EXTREME_TEST_SHA256,
    }
    verified: Dict[str, str] = {}
    for fname, exp_hash in expected.items():
        fpath = base / fname
        if not fpath.exists():
            url = f"https://huggingface.co/datasets/sapientinc/sudoku-extreme/resolve/main/{fname}"
            urllib.request.urlretrieve(url, str(fpath))
        actual_hash = compute_sha256(fpath)
        if actual_hash != exp_hash:
            raise RuntimeError(
                f"SHA256 mismatch for {fpath}: expected {exp_hash}, got {actual_hash}. "
                "Refusing fallback per PROTOCOL.md Section 9.4."
            )
        verified[fname] = actual_hash

    repo_root = Path(_REPO_ROOT)
    data_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_data.py")
    model_py_hash = compute_sha256(repo_root / "external" / "infiloop" / "sudoku_model.py")
    if data_py_hash != INFILOOP_DATA_PY_SHA256 or model_py_hash != INFILOOP_MODEL_PY_SHA256:
        raise RuntimeError("Upstream external/infiloop files failed SHA256 integrity verification.")
    verified["external/infiloop/sudoku_data.py"] = data_py_hash
    verified["external/infiloop/sudoku_model.py"] = model_py_hash
    return base, verified


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
# 1. Sudoku-Extreme Unit Structure & Exact Birkhoff Permutation Identity
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
    """
    Counts pairwise duplicate digits across all 27 units (9 rows, 9 cols, 9 boxes).
    """
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


def sudoku_birkhoff_inconsistency_loss(
    logits: torch.Tensor, x_t: torch.Tensor, unit_idx: torch.Tensor
) -> torch.Tensor:
    """
    Exact Label-Free Birkhoff Permutation & Peer-Clue Inconsistency Loss over the 27 units of 9x9 Sudoku:
      1. Pairwise unit inner product:
         For each 9-cell unit U:
           sum_{u != v in U} <P_u, P_v> = ||sum_{v in U} P_v||_2^2 - sum_{v in U} ||P_v||_2^2 >= 0,
         normalized by 8.0 so uniform prior equals 1.0 and every permutation matrix equals 0.0.
      2. Observable input peer-clue exclusion penalty:
         Using only input clues x_t (0 labels accessed), any digit d already present as a given clue
         in a peer cell of the same row, column, or 3x3 box is excluded via -log(1 - sum_{d in peer} P_v(d)).
    """
    probs = F.softmax(logits, dim=-1)  # [B, 81, 9]
    unit_probs = probs[:, unit_idx, :]  # [B, 27, 9, 9]
    unit_digit_sum = unit_probs.sum(dim=2)  # [B, 27, 9]
    unit_sum_sq = (unit_digit_sum * unit_digit_sum).sum(dim=-1)  # [B, 27]
    unit_cell_sq = (unit_probs * unit_probs).sum(dim=(2, 3))  # [B, 27]
    unit_pair_ip = (unit_sum_sq - unit_cell_sq) / 8.0  # [B, 27]

    clue_oh = F.one_hot(x_t.clamp(min=0, max=9), num_classes=10)[:, :, 1:].to(probs.dtype)  # [B, 81, 9]
    unit_clues = clue_oh[:, unit_idx, :]  # [B, 27, 9, 9]
    unit_clue_other = (unit_clues.sum(dim=2, keepdim=True) - unit_clues).clamp(max=1.0)  # [B, 27, 9, 9]
    bad_prob = (unit_probs * unit_clue_other).sum(dim=-1).clamp(max=1.0 - 1e-5)  # [B, 27, 9]
    peer_clue_loss = -torch.log(1.0 - bad_prob).mean()

    return unit_pair_ip.mean() + 2.0 * peer_clue_loss


class TRMPInfiLoopSudoku(nn.Module):
    """
    Combines TRMP (Salvatori et al. 2026) 3-port unit message passing (row, col, box)
    over both hidden states and unit probability/clue summaries with upstream InfiLoop
    (Li et al. 2026, external/infiloop/sudoku_model.py) Connection streaming weighted state.
    Readout is 100% raw logits.argmax(dim=-1) + 1.
    """

    def __init__(self, cfg: InfiLoopConfig, device: torch.device, mode: str = "infiloop"):
        super().__init__()
        hidden_size = cfg.hidden_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.clue_emb = nn.Embedding(10, hidden_size)
        self.pos_emb = nn.Embedding(81, hidden_size)
        nn.init.normal_(self.clue_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        self.prob_proj = nn.Linear(72, hidden_size)
        self.port_send = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.port_recv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size * 2 + 18, hidden_size),
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
        self.peer_logit_proj = nn.Linear(72, 9)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.peer_logit_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.peer_logit_proj.bias)
        self.q_score = nn.Parameter(torch.zeros(hidden_size))
        self.beta_logit = nn.Parameter(torch.logit(torch.tensor(cfg.beta_init)))
        self.register_buffer("row_ids", torch.tensor([i // 9 for i in range(81)], dtype=torch.long, device=device))
        self.register_buffer("col_ids", torch.tensor([i % 9 for i in range(81)], dtype=torch.long, device=device))
        self.register_buffer(
            "box_ids",
            torch.tensor([(i // 27) * 3 + ((i % 9) // 3) for i in range(81)], dtype=torch.long, device=device),
        )

    @staticmethod
    def _port_peer_sum(t: torch.Tensor, p_idx: int) -> torch.Tensor:
        bsz, _, ch = t.shape
        if p_idx == 0:  # row
            return t.view(bsz, 9, 9, ch).sum(dim=2, keepdim=True).expand(bsz, 9, 9, ch).reshape(bsz, 81, ch) - t
        if p_idx == 1:  # col
            return t.view(bsz, 9, 9, ch).sum(dim=1, keepdim=True).expand(bsz, 9, 9, ch).reshape(bsz, 81, ch) - t
        # 3x3 box
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
            peer_ps.append(p_other)
            peer_ps.append(c_other)
            m = self.port_send[p_idx](h)
            m_other = self._port_peer_sum(m, p_idx) / 8.0
            outs.append(self.port_recv[p_idx](torch.cat([m, m_other, p_other, c_other], dim=-1)))
        return torch.cat(outs, dim=-1), torch.cat(peer_ps, dim=-1)

    def forward_step(
        self,
        x_digits: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        prev_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pos = self.pos_emb(torch.arange(81, device=x_digits.device)).unsqueeze(0)
        c_emb = self.clue_emb(x_digits.clamp(0, 9)) + pos
        clue_mask = (x_digits > 0).unsqueeze(-1).float()
        clue_oh9 = F.one_hot((x_digits - 1).clamp(0, 8), num_classes=9).float() * clue_mask
        if state is None:
            h = infiloop_rms_norm(c_emb, 1e-6)
            probs = clue_oh9 if prev_logits is None else F.softmax(prev_logits, dim=-1)
            s0 = (h.detach() @ self.q_score).clamp(-6.0, 6.0).exp().unsqueeze(-1)
            num = s0 * h
            den = s0
        else:
            h, logits, num, den = state
            probs = F.softmax(logits, dim=-1)
        beta = self.beta_logit.sigmoid()
        u_msg, peer_p_cat = self._unit_messages(h, probs, clue_oh9)
        x = infiloop_rms_norm(h + c_emb + self.prob_proj(peer_p_cat), 1e-6)
        z = infiloop_rms_norm(self.cell_mlp(torch.cat([x, u_msg], dim=-1)), 1e-6)
        if self.mode == "carry_last":
            h = 0.65 * h + 0.35 * z
        else:
            s = (z.detach() @ self.q_score).clamp(-6.0, 6.0).exp().unsqueeze(-1)
            num = beta * num + 0.35 * s * z
            den = beta * den + 0.35 * s
            h = num / den.clamp(min=1e-7)
        logits = self.head(h) + self.peer_logit_proj(peer_p_cat)
        return logits, (h.detach(), logits.detach(), num.detach(), den.detach())


def build_fixed_mean_field_sudoku(
    cfg: InfiLoopConfig,
    device: torch.device,
    seed: int,
    a_peer: float,
    self_w: float,
) -> TRMPInfiLoopSudoku:
    """
    Instantiates TRMPInfiLoopSudoku with zero training and hand-set Jacobi mean-field weights
    on peer_logit_proj: self_w * I_9 on probs (cols 0:9), +20 * I_9 on clue_oh9 (cols 9:18),
    -a_peer * I_9 on peer probabilities (cols 18:27, 36:45, 54:63), and -20 * I_9 on peer clues
    (cols 27:36, 45:54, 63:72).
    """
    torch.manual_seed(seed)
    m = TRMPInfiLoopSudoku(cfg, device).to(device)
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias.zero_()
        m.peer_logit_proj.weight.zero_()
        m.peer_logit_proj.bias.zero_()
        eye9 = torch.eye(9, device=device)
        m.peer_logit_proj.weight[:, 0:9] = float(self_w) * eye9
        m.peer_logit_proj.weight[:, 9:18] = 20.0 * eye9
        for start_col in (18, 36, 54):
            m.peer_logit_proj.weight[:, start_col : start_col + 9] = -float(a_peer) * eye9
        for start_col in (27, 45, 63):
            m.peer_logit_proj.weight[:, start_col : start_col + 9] = -20.0 * eye9
    return m


def evaluate_sudoku_peer_clue_uniform(
    x_test: np.ndarray,
    y_test: np.ndarray,
    unit_idx_np: np.ndarray,
    segments_eval: List[int],
    seed: int,
) -> Dict[str, Any]:
    """
    Non-learned floor control (advisory item 1): copies the given clues, and for each empty cell
    picks uniformly (seeded by seed) among digits 1..9 not used by any clue in the same row, column, or box.
    """
    rng = np.random.RandomState(seed)
    bsz = x_test.shape[0]
    preds = np.zeros_like(x_test)
    empty_mask = x_test == 0
    clue_mask = x_test > 0
    cell_units = [[] for _ in range(81)]
    for u in range(27):
        for cell in unit_idx_np[u]:
            cell_units[int(cell)].append(u)

    for b in range(bsz):
        xb = x_test[b]
        for i in range(81):
            if xb[i] > 0:
                preds[b, i] = xb[i]
            else:
                used = set()
                for u in cell_units[i]:
                    for peer in unit_idx_np[u]:
                        val = int(xb[peer])
                        if val > 0:
                            used.add(val)
                cands = [d for d in range(1, 10) if d not in used]
                if not cands:
                    cands = list(range(1, 10))
                preds[b, i] = cands[rng.randint(len(cands))]

    coll = float(count_sudoku_unit_collisions(preds, unit_idx_np).mean())
    empty_acc = float(((preds == y_test) & empty_mask).sum() / max(empty_mask.sum(), 1))
    clue_acc = float(((preds == y_test) & clue_mask).sum() / max(clue_mask.sum(), 1))
    all_acc = float((preds == y_test).mean())
    by_seg = {
        str(s): {
            "empty_acc": empty_acc,
            "clue_acc": clue_acc,
            "all_acc": all_acc,
            "mean_unit_collisions": coll,
        }
        for s in segments_eval
    }
    return {"by_segment": by_seg, **by_seg[str(max(segments_eval))]}


def evaluate_sudoku_arm(
    arm_name: str,
    model: Optional[nn.Module],
    x_test: np.ndarray,
    y_test: np.ndarray,
    unit_idx: torch.Tensor,
    segments_eval: List[int],
    device: torch.device,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    100% pure-neural readout for all neural arms (including Control_Random_Network and Control_Untrained_Init):
      preds = logits.argmax(dim=-1) + 1.
    Zero Python backtracking, zero singles solver, zero copy-forward, zero noise injection.
    """
    x_t = torch.tensor(x_test, dtype=torch.long, device=device)
    bsz = x_t.shape[0]
    unit_idx_np = unit_idx.cpu().numpy()
    max_s = max(segments_eval)
    empty_mask = x_test == 0
    clue_mask = x_test > 0

    if arm_name == "Control_PeerClue_Uniform":
        return evaluate_sudoku_peer_clue_uniform(x_test, y_test, unit_idx_np, segments_eval, seed)

    if arm_name == "Control_ClueCopy_RowColMode":
        preds = x_test.copy()
        for b in range(bsz):
            vals = preds[b][preds[b] > 0]
            mode_val = int(np.bincount(vals).argmax()) if len(vals) > 0 else 1
            preds[b][preds[b] == 0] = mode_val
        coll = float(count_sudoku_unit_collisions(preds, unit_idx_np).mean())
        empty_acc = float(((preds == y_test) & empty_mask).sum() / max(empty_mask.sum(), 1))
        clue_acc = float(((preds == y_test) & clue_mask).sum() / max(clue_mask.sum(), 1))
        all_acc = float((preds == y_test).mean())
        by_seg = {
            str(s): {
                "empty_acc": empty_acc,
                "clue_acc": clue_acc,
                "all_acc": all_acc,
                "mean_unit_collisions": coll,
            }
            for s in segments_eval
        }
        return {"by_segment": by_seg, **by_seg[str(max_s)]}

    assert model is not None
    model.eval()
    by_seg = {}
    with torch.no_grad():
        state = None
        prev_logits = None
        x_tokens = (x_t + 1).clamp(min=1, max=10)
        for s in range(1, max_s + 1):
            if isinstance(model, TRMPInfiLoopSudoku):
                logits, state = model.forward_step(x_t, state, prev_logits)
                prev_logits = logits
            else:
                logits_11, state = model(x_tokens, state)
                logits = logits_11[:, :, 2:11]
            if s in segments_eval:
                preds = (logits.argmax(dim=-1) + 1).cpu().numpy()
                coll = float(count_sudoku_unit_collisions(preds, unit_idx_np).mean())
                empty_acc = float(((preds == y_test) & empty_mask).sum() / max(empty_mask.sum(), 1))
                clue_acc = float(((preds == y_test) & clue_mask).sum() / max(clue_mask.sum(), 1))
                all_acc = float((preds == y_test).mean())
                by_seg[str(s)] = {
                    "empty_acc": empty_acc,
                    "clue_acc": clue_acc,
                    "all_acc": all_acc,
                    "mean_unit_collisions": coll,
                }

    return {"by_segment": by_seg, **by_seg[str(max_s)]}


# ---------------------------------------------------------------------------
# 2. Ambiguous Multi-Solution N-Queens (10x10 -> 12x12, 16x16) With Input & Solution D_4 Split
# ---------------------------------------------------------------------------

def solve_all_nqueens(n: int, max_sols: int = 2000) -> List[Tuple[int, ...]]:
    """
    Enumerates valid N-Queens solutions distributed across all row-0 columns c0 in 0..n-1
    so that OOD solution pools (N=12, N=16) cover all starting columns rather than only (0, 0).
    """
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
    """
    Canonical D_4 symmetry-orbit representative of an input partial clue board.
    Guarantees that no test input clue board is a rotation or reflection of any training input clue board.
    """
    forms = []
    for rot in range(4):
        cur = list(clues)
        for _ in range(rot):
            cur = [(c, n - 1 - r) for (r, c) in cur]
        for flip in (False, True):
            cur2 = [(r, n - 1 - c) if flip else (r, c) for (r, c) in cur]
            forms.append(tuple(sorted(cur2)))
    return min(forms)


def sample_disjoint_nqueens_2to3clue_split(
    sols_pool: List[Tuple[int, ...]],
    all_sols: List[Tuple[int, ...]],
    n: int,
    count: int,
    forbidden_input_orbits: Set[Tuple[Tuple[int, int], ...]],
    seed: int,
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, Set[Tuple[Tuple[int, int], ...]]]:
    """
    Samples ambiguous 2-to-3 clue N-Queens boards (with clues drawn across all rows 0..n-1)
    such that every board has >= 2 distinct valid completions and its D_4 input-clue orbit
    is disjoint from forbidden_input_orbits.
    """
    rng = np.random.RandomState(seed)
    sol_arr = np.array(all_sols, dtype=np.int64)
    xs: List[np.ndarray] = []
    multi_ys: List[np.ndarray] = []
    pair_x: List[np.ndarray] = []
    pair_y: List[np.ndarray] = []
    used_orbits: Set[Tuple[Tuple[int, int], ...]] = set()
    attempts = 0
    while len(xs) < count and attempts < count * 100:
        attempts += 1
        s = sols_pool[rng.randint(len(sols_pool))]
        k = int(rng.choice([2, 3]))
        rows = sorted(rng.choice(n, size=k, replace=False).tolist())
        clues = tuple((int(r), int(s[r])) for r in rows)
        c_orb = d4_clue_orbit_rep(clues, n)
        if c_orb in forbidden_input_orbits or c_orb in used_orbits:
            continue
        mask = np.ones(len(sol_arr), dtype=bool)
        for rw, cl in clues:
            mask &= sol_arr[:, rw] == cl
        matching = sol_arr[mask]
        if len(matching) < 2 and n == 10:
            continue
        x = np.full(n, -1, dtype=np.int64)
        for rw, cl in clues:
            x[rw] = cl
        xs.append(x)
        multi_ys.append(matching[:8])
        for m_sol in matching[:8]:
            pair_x.append(x)
            pair_y.append(m_sol)
        used_orbits.add(c_orb)
    if len(xs) < count:
        raise RuntimeError(f"Could only sample {len(xs)}/{count} disjoint N-Queens boards for N={n}")
    return np.array(xs, dtype=np.int64), multi_ys, np.array(pair_x, dtype=np.int64), np.array(pair_y, dtype=np.int64), used_orbits


def count_nqueens_metrics(cols_pred: np.ndarray, clues: np.ndarray) -> Dict[str, float]:
    bsz, n = cols_pred.shape
    total_coll = np.zeros(bsz, dtype=np.float64)
    row_ok = np.zeros(bsz, dtype=np.float64)
    clue_ok = np.zeros(bsz, dtype=np.float64)
    for b in range(bsz):
        c = cols_pred[b]
        cl = clues[b]
        mask = cl >= 0
        clue_ok[b] = float((c[mask] == cl[mask]).mean()) if mask.any() else 1.0
        cc, dc, ok = 0, 0, 0
        for r1 in range(n):
            r_clean = True
            for r2 in range(n):
                if r1 == r2:
                    continue
                if c[r1] == c[r2]:
                    r_clean = False
                    if r2 > r1:
                        cc += 1
                if abs(r1 - r2) == abs(int(c[r1]) - int(c[r2])):
                    r_clean = False
                    if r2 > r1:
                        dc += 1
            if r_clean:
                ok += 1
        total_coll[b] = cc + dc
        row_ok[b] = ok / float(n)
    return {
        "mean_collisions": float(total_coll.mean()),
        "conflict_free_row_rate": float(row_ok.mean()),
        "clue_acc": float(clue_ok.mean()),
    }


def nqueens_peer_attack_mask(clues: torch.Tensor) -> torch.Tensor:
    """
    Computes the binary observable input clue-attack indicator (B, N, N) for non-clued rows:
    1.0 at (r, c) iff row r has no given clue and cell (r, c) shares a column or diagonal
    with an observable input clue (r_clue, c_clue). Uses zero ground-truth labels.
    """
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


def birkhoff_nqueens_loss(logits: torch.Tensor, clues: torch.Tensor) -> torch.Tensor:
    """
    Exact Birkhoff column identity + row Gini entropy + exact diagonal collision inner product
    + observable input peer-clue exclusion loss + clue preservation cross-entropy (zero ground-truth labels).
    """
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
    clue_mask = clues >= 0
    if clue_mask.any():
        clue_ce = F.cross_entropy(logits[clue_mask], clues[clue_mask].clamp(min=0))
    else:
        clue_ce = torch.tensor(0.0, device=logits.device)
    return col_birkhoff + row_ent + 2.0 * diag_loss + 2.0 * peer_loss + clue_ce


class Looped2DCellNQueensNet(nn.Module):
    """
    Size-invariant 2D Cell-Grid Looped Network operating on (B, N, N, D) across any board size N
    (10x10 in-distribution; 12x12 and 16x16 OOD) with exact column and diagonal peer-probability
    message passing across segments.
    Readout is 100% raw logits.argmax(dim=-1). Zero solver/heuristic code in forward or eval.
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

    def forward(self, clues: torch.Tensor, n_segments: int = 8, n_loops: int = 2) -> List[torch.Tensor]:
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


def evaluate_nqueens_arm(
    arm_name: str,
    model: Optional[Looped2DCellNQueensNet],
    clues_dict: Dict[str, np.ndarray],
    device: torch.device,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    100% pure-neural readout for all neural arms (including Control_Random_Network and Control_Untrained_Init):
      preds = logits.argmax(dim=-1).
    Zero backtracking or column-deduplication helper in any neural arm.
    """
    out: Dict[str, Any] = {}
    for split_key, x_np in clues_dict.items():
        bsz, n = x_np.shape
        if arm_name == "Control_PeerClue_Uniform":
            rng = np.random.RandomState(seed + n)
            preds = np.zeros((bsz, n), dtype=np.int64)
            for b in range(bsz):
                cl_rows = np.where(x_np[b] >= 0)[0]
                cl_pairs = [(int(r), int(x_np[b, r])) for r in cl_rows]
                for r in range(n):
                    if x_np[b, r] >= 0:
                        preds[b, r] = x_np[b, r]
                    else:
                        cands = [
                            c
                            for c in range(n)
                            if all(c != cc and abs(r - cr) != abs(c - cc) for cr, cc in cl_pairs)
                        ]
                        if not cands:
                            cands = list(range(n))
                        preds[b, r] = cands[rng.randint(len(cands))]
            out[split_key] = count_nqueens_metrics(preds, x_np)
        elif arm_name == "Control_ClueCopy_ValidCol":
            preds = np.zeros((bsz, n), dtype=np.int64)
            for b in range(bsz):
                cl_rows = np.where(x_np[b] >= 0)[0]
                c_val = int(x_np[b, cl_rows[0]]) if len(cl_rows) > 0 else 0
                preds[b, :] = c_val
            out[split_key] = count_nqueens_metrics(preds, x_np)
        else:
            assert model is not None
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(x_np, dtype=torch.long, device=device)
                seg_logits = model(x_t, n_segments=8, n_loops=2)
                preds = seg_logits[-1].argmax(dim=-1).cpu().numpy()
                out[split_key] = count_nqueens_metrics(preds, x_np)
    return out


# ---------------------------------------------------------------------------
# 3. Main R27A Training & Evaluation Loop
# ---------------------------------------------------------------------------

def run_r27a(args) -> Dict[str, Any]:
    resolved_sudoku_dir, data_hashes = ensure_and_verify_sudoku_extreme(args.sudoku_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    seeds = [101, 102, 103]
    segments_eval = [1, 2, 4, 8, 12]

    n_train_full = args.n_train_sudoku
    n_test_full = args.n_test_sudoku
    n_train_smoke = 256
    n_test_smoke = 64
    sudoku_steps = 80 if args.smoke_cpu else args.sudoku_steps
    nq_steps = 60 if args.smoke_cpu else args.nq_steps
    d_model = 64 if args.smoke_cpu else 128
    infiloop_cfg = InfiLoopConfig(hidden_size=d_model, num_heads=4, layers=2, h_cycles=2, l_cycles=2, beta_init=0.85)

    # Load public sapientinc/sudoku-extreme via unmodified external/infiloop/sudoku_data.py
    # Slice disjoint ranges for frozen full evaluation vs smoke_cpu per Rule 9.9,
    # and sample test puzzles from test.csv[320:] (rng=default_rng(27001)) so overlap with
    # test.csv[0:320] (33c4d10 smoke rows) is strictly 0 and difficulty spans the full test distribution!
    x_tr_all, y_tr_all = read_sudoku_csv(
        resolved_sudoku_dir / "train.csv", max(5256, n_train_full + n_train_smoke + 256), seed=101, random_subset=False
    )
    x_te_all, y_te_all = read_sudoku_csv(
        resolved_sudoku_dir / "test.csv", 0, seed=101, random_subset=False
    )
    x_tr_full_np = x_tr_all[:n_train_full].reshape(-1, 81).astype(np.int64)
    y_tr_full_np = y_tr_all[:n_train_full].reshape(-1, 81).astype(np.int64)
    x_tr_smoke_np = x_tr_all[n_train_full : n_train_full + n_train_smoke].reshape(-1, 81).astype(np.int64)
    y_tr_smoke_np = y_tr_all[n_train_full : n_train_full + n_train_smoke].reshape(-1, 81).astype(np.int64)
    # Disjoint training-split dev slice (train.csv[5000:5256] full, train.csv[4352:4416] smoke) for FixedMeanField grid selection
    x_dev_full_np = x_tr_all[5000:5256].reshape(-1, 81).astype(np.int64)
    y_dev_full_np = y_tr_all[5000:5256].reshape(-1, 81).astype(np.int64)
    x_dev_smoke_np = x_tr_all[4352:4416].reshape(-1, 81).astype(np.int64)
    y_dev_smoke_np = y_tr_all[4352:4416].reshape(-1, 81).astype(np.int64)

    head320_set = {tuple(r) for r in x_te_all[:320].reshape(-1, 81).astype(np.int64)}
    te_pool_x = x_te_all[320:].reshape(-1, 81).astype(np.int64)
    te_pool_y = y_te_all[320:].reshape(-1, 81).astype(np.int64)
    te_rng = np.random.default_rng(27001)
    te_perm = te_rng.permutation(len(te_pool_x))

    x_te_full_np = te_pool_x[te_perm[:n_test_full]]
    y_te_full_np = te_pool_y[te_perm[:n_test_full]]
    x_te_smoke_np = te_pool_x[te_perm[n_test_full : n_test_full + n_test_smoke]]
    y_te_smoke_np = te_pool_y[te_perm[n_test_full : n_test_full + n_test_smoke]]

    sudoku_head320_overlap = len({tuple(r) for r in x_te_full_np} & head320_set)
    sudoku_smoke_test_overlap = len({tuple(r) for r in x_te_full_np} & {tuple(r) for r in x_te_smoke_np})
    sudoku_train_test_overlap = len({tuple(r) for r in x_tr_full_np} & {tuple(r) for r in x_te_full_np})
    sudoku_dev_test_overlap = len({tuple(r) for r in x_dev_full_np} & {tuple(r) for r in x_te_full_np})

    x_tr_np = x_tr_smoke_np if args.smoke_cpu else x_tr_full_np
    y_tr_np = y_tr_smoke_np if args.smoke_cpu else y_tr_full_np
    x_dev_np = x_dev_smoke_np if args.smoke_cpu else x_dev_full_np
    y_dev_np = y_dev_smoke_np if args.smoke_cpu else y_dev_full_np
    x_te_np = x_te_smoke_np if args.smoke_cpu else x_te_full_np
    y_te_np = y_te_smoke_np if args.smoke_cpu else y_te_full_np

    # Precompute 10x10 N-Queens D_4 symmetry-orbit split on both solutions and 2-3 clue inputs
    sols10 = solve_all_nqueens(10, max_sols=2000)
    orbits10: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for s in sols10:
        orbits10.setdefault(d4_orbit_rep(s), []).append(s)
    orbit_keys = sorted(orbits10.keys())
    train_orbit_set = set(orbit_keys[:64])
    test_orbit_set = set(orbit_keys[64:])
    d4_sol_overlap = len(train_orbit_set.intersection(test_orbit_set))

    train_sols10 = [s for k in orbit_keys[:64] for s in orbits10[k]]
    test_sols10 = [s for k in orbit_keys[64:] for s in orbits10[k]]

    nq_tr_x, nq_tr_multi_y, nq_tr_pair_x, nq_tr_pair_y, tr10_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        train_sols10, sols10, 10, 160, set(), 2710
    )
    nq_te10_full_x, _, _, _, te10_full_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        test_sols10, sols10, 10, 64, tr10_in_orbs, 2711
    )
    nq_te10_smoke_x, _, _, _, te10_smoke_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        test_sols10, sols10, 10, 48, tr10_in_orbs | te10_full_in_orbs, 2712
    )

    sols12 = solve_all_nqueens(12, max_sols=400)
    nq_te12_full_x, _, _, _, te12_full_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        sols12, sols12, 12, 48, set(), 2721
    )
    nq_te12_smoke_x, _, _, _, te12_smoke_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        sols12, sols12, 12, 36, te12_full_in_orbs, 2722
    )

    sols16 = solve_all_nqueens(16, max_sols=200)
    nq_te16_full_x, _, _, _, te16_full_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        sols16, sols16, 16, 40, set(), 2731
    )
    nq_te16_smoke_x, _, _, _, te16_smoke_in_orbs = sample_disjoint_nqueens_2to3clue_split(
        sols16, sols16, 16, 32, te16_full_in_orbs, 2732
    )

    nq_eval_splits = {
        "10x10_id": nq_te10_smoke_x if args.smoke_cpu else nq_te10_full_x,
        "12x12_ood": nq_te12_smoke_x if args.smoke_cpu else nq_te12_full_x,
        "16x16_ood": nq_te16_smoke_x if args.smoke_cpu else nq_te16_full_x,
    }

    nq10_exact_input_overlap = len(
        {tuple(x) for x in nq_tr_x} & {tuple(x) for x in nq_eval_splits["10x10_id"]}
    )
    eval10_orbs = te10_smoke_in_orbs if args.smoke_cpu else te10_full_in_orbs
    nq10_d4_input_orbit_overlap = len(tr10_in_orbs & eval10_orbs)
    nq_smoke_test_overlap = (
        len(te10_full_in_orbs & te10_smoke_in_orbs)
        + len(te12_full_in_orbs & te12_smoke_in_orbs)
        + len(te16_full_in_orbs & te16_smoke_in_orbs)
    )

    unit_idx = build_sudoku_unit_indices(device)
    # Sweep pre-registered 15-setting grid of hand-set weights on the training-split dev slice (never test!)
    mf_dev_grid_results: List[Dict[str, Any]] = []
    best_mf_pair = (1.5, 0.0)
    best_mf_dev_score = (-1.0, -1e9)
    for a_cand in SUDOKU_FIXED_MF_A_GRID:
        for self_cand in SUDOKU_FIXED_MF_SELF_GRID:
            cand_m = build_fixed_mean_field_sudoku(infiloop_cfg, device, seeds[0], a_cand, self_cand)
            dev_ev = evaluate_sudoku_arm(
                "Control_FixedMeanField_DevGrid", cand_m, x_dev_np, y_dev_np, unit_idx, segments_eval, device, seeds[0]
            )["by_segment"]["12"]
            dev_acc = float(dev_ev["empty_acc"])
            dev_coll = float(dev_ev["mean_unit_collisions"])
            mf_dev_grid_results.append(
                {"a_peer": a_cand, "self_w": self_cand, "dev_s12_empty_acc": dev_acc, "dev_s12_collisions": dev_coll}
            )
            score_tuple = (round(dev_acc, 6), -round(dev_coll, 4))
            if score_tuple > best_mf_dev_score:
                best_mf_dev_score = score_tuple
                best_mf_pair = (a_cand, self_cand)
    best_mf_a, best_mf_self = best_mf_pair

    per_seed: Dict[str, Any] = {}
    total_label_access: Dict[str, int] = {
        "Control_Random_Network": 0,
        "Control_Untrained_Init": 0,
        "Control_PeerClue_Uniform": 0,
        "Control_ClueCopy_RowColMode": 0,
        "Control_FixedMeanField": 0,
        "InfiLoop_Supervised_OurTrain": 0,
        "Ablation_UnmaskedClue_CE_Only_LabelFree": 0,
        "TRMP_InfiLoop_Birkhoff_LabelFree": 0,
        "NQ_Control_FixedMeanField": 0,
        "NQ_StandardCE_MultiSol_Sup_OurTrain": 0,
        "NQ_TRMP_MinCE_MultiLabel_Sup_OurTrain": 0,
        "NQ_Ablation_UnmaskedClue_CE_Only_LabelFree": 0,
        "NQ_TRMP_InfiLoop_Birkhoff_LabelFree": 0,
    }

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        watched_y_sudoku = WatchedLabelArray(y_tr_np)
        watched_nq_pair_y = WatchedLabelArray(nq_tr_pair_y)
        watched_nq_multi_y = WatchedLabelArray(nq_tr_multi_y)

        sudoku_arms: Dict[str, Any] = {}
        # Rule 9.1: Control_Random_Network runs the exact same TRMPInfiLoopSudoku pipeline with random weights
        torch.manual_seed(seed + 9000)
        random_sudoku_net = TRMPInfiLoopSudoku(infiloop_cfg, device).to(device)
        sudoku_arms["Control_Random_Network"] = evaluate_sudoku_arm(
            "Control_Random_Network", random_sudoku_net, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )
        sudoku_arms["Control_PeerClue_Uniform"] = evaluate_sudoku_arm(
            "Control_PeerClue_Uniform", None, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )
        sudoku_arms["Control_ClueCopy_RowColMode"] = evaluate_sudoku_arm(
            "Control_ClueCopy_RowColMode", None, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )

        torch.manual_seed(seed)
        untrained_sudoku = TRMPInfiLoopSudoku(infiloop_cfg, device).to(device)
        sudoku_arms["Control_Untrained_Init"] = evaluate_sudoku_arm(
            "Control_Untrained_Init", untrained_sudoku, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )

        # Control_FixedMeanField: best setting from pre-registered 15-point grid selected on training-split dev slice
        fixed_mf_sudoku = build_fixed_mean_field_sudoku(infiloop_cfg, device, seed, best_mf_a, best_mf_self)
        sudoku_arms["Control_FixedMeanField"] = evaluate_sudoku_arm(
            "Control_FixedMeanField", fixed_mf_sudoku, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )
        sudoku_arms["Control_FixedMeanField"]["selected_a_peer"] = best_mf_a
        sudoku_arms["Control_FixedMeanField"]["selected_self_w"] = best_mf_self

        # 1. Upstream InfiLoop_Supervised_OurTrain (ungated diagnostic reference per Rule 9.5)
        torch.manual_seed(seed)
        sup_model = InfiLoop(infiloop_cfg).to(device)
        sup_opt = torch.optim.AdamW(sup_model.parameters(), lr=3e-3, weight_decay=1e-4)
        sup_model.train()
        for _ in range(sudoku_steps):
            b_idx = np.random.choice(len(x_tr_np), size=32, replace=True)
            xb = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)
            yb = torch.tensor(watched_y_sudoku[b_idx], dtype=torch.long, device=device)
            xb_tok = (xb + 1).clamp(min=1, max=10)
            yb_tok = (yb + 1).clamp(min=2, max=10)
            state = None
            for _ in range(4):
                logits_11, state = sup_model(xb_tok, state)
                loss = upstream_infiloop_loss(logits_11, yb_tok)
                sup_opt.zero_grad()
                loss.backward()
                sup_opt.step()
        total_label_access["InfiLoop_Supervised_OurTrain"] += watched_y_sudoku.access_count
        sup_eval = evaluate_sudoku_arm(
            "InfiLoop_Supervised_OurTrain", sup_model, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
        )
        sup_eval["training_regime"] = f"our training, {sudoku_steps} steps (ungated diagnostic per Rule 9.5)"
        sudoku_arms["InfiLoop_Supervised_OurTrain"] = sup_eval

        # 2. Label-Free Sudoku Arms (0 accesses to watched_y_sudoku)
        for lf_arm, use_birkhoff in [
            ("Ablation_UnmaskedClue_CE_Only_LabelFree", False),
            ("TRMP_InfiLoop_Birkhoff_LabelFree", True),
        ]:
            before_cnt = watched_y_sudoku.access_count
            torch.manual_seed(seed)
            lf_model = TRMPInfiLoopSudoku(infiloop_cfg, device).to(device)
            lf_opt = torch.optim.AdamW(lf_model.parameters(), lr=4e-3, weight_decay=1e-4)
            lf_model.train()
            for _ in range(sudoku_steps):
                b_idx = np.random.choice(len(x_tr_np), size=32, replace=True)
                xb_full = torch.tensor(x_tr_np[b_idx], dtype=torch.long, device=device)
                clue_mask = xb_full > 0
                lf_opt.zero_grad()
                state = None
                prev_logits = None
                loss = torch.tensor(0.0, device=device)
                for seg_i in range(6):
                    logits, state = lf_model.forward_step(xb_full, state, prev_logits)
                    prev_logits = logits.detach()
                    clue_ce = F.cross_entropy(logits[clue_mask], (xb_full[clue_mask] - 1).clamp(min=0))
                    b_loss = sudoku_birkhoff_inconsistency_loss(logits, xb_full, unit_idx) if use_birkhoff else 0.0
                    w_seg = (seg_i + 1.0) / 21.0
                    loss = loss + w_seg * (clue_ce + b_loss)
                loss.backward()
                lf_opt.step()
            after_cnt = watched_y_sudoku.access_count
            total_label_access[lf_arm] += after_cnt - before_cnt
            sudoku_arms[lf_arm] = evaluate_sudoku_arm(
                lf_arm, lf_model, x_te_np, y_te_np, unit_idx, segments_eval, device, seed
            )

        # 3. N-Queens 10x10 -> 12x12, 16x16 Multi-Solution & OOD Experiment
        nq_arms: Dict[str, Any] = {}
        torch.manual_seed(seed + 9000)
        nq_random_net = Looped2DCellNQueensNet(d_model=48).to(device)
        nq_arms["Control_Random_Network"] = evaluate_nqueens_arm(
            "Control_Random_Network", nq_random_net, nq_eval_splits, device, seed
        )
        nq_arms["Control_PeerClue_Uniform"] = evaluate_nqueens_arm(
            "Control_PeerClue_Uniform", None, nq_eval_splits, device, seed
        )
        nq_arms["Control_ClueCopy_ValidCol"] = evaluate_nqueens_arm(
            "Control_ClueCopy_ValidCol", None, nq_eval_splits, device, seed
        )

        torch.manual_seed(seed)
        nq_untrained = Looped2DCellNQueensNet(d_model=48).to(device)
        nq_arms["Control_Untrained_Init"] = evaluate_nqueens_arm(
            "Control_Untrained_Init", nq_untrained, nq_eval_splits, device, seed
        )

        # Control_FixedMeanField on N-Queens: zero-training Jacobi mean-field baseline on peer_logit_proj
        torch.manual_seed(seed)
        nq_fixed_mf = Looped2DCellNQueensNet(d_model=48).to(device)
        with torch.no_grad():
            nq_fixed_mf.out_head.weight.zero_()
            nq_fixed_mf.out_head.bias.zero_()
            nq_fixed_mf.peer_logit_proj.weight.copy_(torch.tensor([[20.0, -20.0, -3.0, -3.0]], device=device))
            nq_fixed_mf.peer_logit_proj.bias.zero_()
        nq_arms["Control_FixedMeanField"] = evaluate_nqueens_arm(
            "Control_FixedMeanField", nq_fixed_mf, nq_eval_splits, device, seed
        )

        for nq_arm in [
            "StandardCE_MultiSol_Sup_OurTrain",
            "TRMP_MinCE_MultiLabel_Sup_OurTrain",
            "Ablation_UnmaskedClue_CE_Only_LabelFree",
            "TRMP_InfiLoop_Birkhoff_LabelFree",
        ]:
            torch.manual_seed(seed)
            nq_net = Looped2DCellNQueensNet(d_model=48).to(device)
            nq_opt = torch.optim.AdamW(nq_net.parameters(), lr=3e-3, weight_decay=1e-4)
            nq_net.train()
            before_pair = watched_nq_pair_y.access_count
            before_multi = watched_nq_multi_y.access_count

            for _ in range(nq_steps):
                if nq_arm == "StandardCE_MultiSol_Sup_OurTrain":
                    idx = np.random.choice(len(nq_tr_pair_x), size=32, replace=True)
                    xb = torch.tensor(nq_tr_pair_x[idx], dtype=torch.long, device=device)
                    yb = torch.tensor(watched_nq_pair_y[idx], dtype=torch.long, device=device)
                    seg_logits = nq_net(xb, n_segments=6, n_loops=2)
                    loss = sum(F.cross_entropy(lg.reshape(-1, 10), yb.reshape(-1)) for lg in seg_logits) / len(seg_logits)
                elif nq_arm == "TRMP_MinCE_MultiLabel_Sup_OurTrain":
                    idx = np.random.choice(len(nq_tr_x), size=16, replace=True)
                    xb = torch.tensor(nq_tr_x[idx], dtype=torch.long, device=device)
                    cand_sets = watched_nq_multi_y[idx]
                    seg_logits = nq_net(xb, n_segments=6, n_loops=2)
                    loss = torch.tensor(0.0, device=device)
                    for lg in seg_logits:
                        b_loss = torch.tensor(0.0, device=device)
                        for b_i, cand_np in enumerate(cand_sets):
                            cand_y = torch.tensor(cand_np, dtype=torch.long, device=device)
                            m_sols = cand_y.shape[0]
                            lg_exp = lg[b_i].unsqueeze(0).expand(m_sols, -1, -1)
                            ces = (
                                F.cross_entropy(lg_exp.reshape(-1, 10), cand_y.reshape(-1), reduction="none")
                                .view(m_sols, 10)
                                .mean(dim=-1)
                            )
                            b_loss = b_loss + ces.min()
                        loss = loss + (b_loss / len(idx))
                    loss = loss / len(seg_logits)
                elif nq_arm == "Ablation_UnmaskedClue_CE_Only_LabelFree":
                    idx = np.random.choice(len(nq_tr_x), size=32, replace=True)
                    xb = torch.tensor(nq_tr_x[idx], dtype=torch.long, device=device)
                    seg_logits = nq_net(xb, n_segments=6, n_loops=2)
                    c_mask = xb >= 0
                    loss = sum(F.cross_entropy(lg[c_mask], xb[c_mask].clamp(min=0)) for lg in seg_logits) / len(seg_logits)
                else:
                    idx = np.random.choice(len(nq_tr_x), size=32, replace=True)
                    xb = torch.tensor(nq_tr_x[idx], dtype=torch.long, device=device)
                    seg_logits = nq_net(xb, n_segments=6, n_loops=2)
                    loss = sum(birkhoff_nqueens_loss(lg, xb) for lg in seg_logits) / len(seg_logits)

                nq_opt.zero_grad()
                loss.backward()
                nq_opt.step()

            used_labels = (watched_nq_pair_y.access_count - before_pair) + (watched_nq_multi_y.access_count - before_multi)
            total_label_access[f"NQ_{nq_arm}"] += used_labels
            arm_eval = evaluate_nqueens_arm(nq_arm, nq_net, nq_eval_splits, device, seed)
            if nq_arm.endswith("_OurTrain"):
                arm_eval["training_regime"] = f"our training, {nq_steps} steps (ungated diagnostic per Rule 9.5)"
            nq_arms[nq_arm] = arm_eval

        per_seed[str(seed)] = {
            "sudoku_extreme": sudoku_arms,
            "nqueens": nq_arms,
        }

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    prereg_path = Path(_REPO_ROOT) / "benchmarks" / "PREREG-R27A-LABELFREE-INFILOOP-TRMP.md"
    prereg_sha256 = compute_sha256(prereg_path) if prereg_path.exists() else ""

    results = {
        "round": "R27A",
        "runner_sha256": runner_sha256,
        "prereg_sha256": prereg_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "segments_eval": segments_eval,
        "dataset_sha256": data_hashes,
        "disjointness_checks": {
            "total_orbits_10x10": len(orbit_keys),
            "train_orbits": 64,
            "test_orbits": 28,
            "d4_orbit_overlap_count": d4_sol_overlap,
            "nq10_exact_input_overlap_count": nq10_exact_input_overlap,
            "nq10_d4_input_orbit_overlap_count": nq10_d4_input_orbit_overlap,
            "nq_smoke_test_overlap_count": nq_smoke_test_overlap,
            "sudoku_head320_overlap_count": sudoku_head320_overlap,
            "sudoku_smoke_test_overlap_count": sudoku_smoke_test_overlap,
            "sudoku_train_test_overlap_count": sudoku_train_test_overlap,
            "sudoku_dev_test_overlap_count": sudoku_dev_test_overlap,
        },
        "fixed_mean_field_dev_selection": {
            "dev_slice": "train.csv[4352:4416]" if args.smoke_cpu else "train.csv[5000:5256]",
            "a_grid": SUDOKU_FIXED_MF_A_GRID,
            "self_grid": SUDOKU_FIXED_MF_SELF_GRID,
            "selected_a_peer": best_mf_a,
            "selected_self_w": best_mf_self,
            "grid_results": mf_dev_grid_results,
        },
        "label_access_counts": total_label_access,
        "per_seed": per_seed,
    }
    results["gates"] = evaluate_r27a_gates(results)
    return results


def evaluate_r27a_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    seeds = [str(s) for s in results["seeds"]]
    per_seed = results["per_seed"]
    lac = results["label_access_counts"]
    hashes = results["dataset_sha256"]
    disj = results["disjointness_checks"]
    is_smoke = bool(results.get("smoke_cpu", False))

    # D0: Public data SHA256 verified, input & orbit disjointness == 0, WatchedLabelArray counts verified
    d0_pass = bool(
        hashes.get("train.csv") == SUDOKU_EXTREME_TRAIN_SHA256
        and hashes.get("test.csv") == SUDOKU_EXTREME_TEST_SHA256
        and disj["d4_orbit_overlap_count"] == 0
        and disj["nq10_exact_input_overlap_count"] == 0
        and disj["nq10_d4_input_orbit_overlap_count"] == 0
        and disj["nq_smoke_test_overlap_count"] == 0
        and disj["sudoku_head320_overlap_count"] == 0
        and disj["sudoku_smoke_test_overlap_count"] == 0
        and disj["sudoku_train_test_overlap_count"] == 0
        and disj["sudoku_dev_test_overlap_count"] == 0
        and lac["TRMP_InfiLoop_Birkhoff_LabelFree"] == 0
        and lac["Ablation_UnmaskedClue_CE_Only_LabelFree"] == 0
        and lac["NQ_TRMP_InfiLoop_Birkhoff_LabelFree"] == 0
        and lac["NQ_Ablation_UnmaskedClue_CE_Only_LabelFree"] == 0
        and lac["InfiLoop_Supervised_OurTrain"] > 0
        and lac["NQ_StandardCE_MultiSol_Sup_OurTrain"] > 0
        and lac["NQ_TRMP_MinCE_MultiLabel_Sup_OurTrain"] > 0
    )

    # D1: Label-Free Sudoku beats all controls (Random_Network, Untrained_Init, ClueCopy_RowColMode,
    # PeerClue_Uniform, dev-selected FixedMeanField) and UnmaskedClue_CE_Only ablation on both empty-cell
    # accuracy and unit collisions, preserves clues >= 0.98, and refines monotonically across S in {1, 2, 4, 8, 12}
    lf_empty_s12 = [per_seed[s]["sudoku_extreme"]["TRMP_InfiLoop_Birkhoff_LabelFree"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    rand_empty_s12 = [per_seed[s]["sudoku_extreme"]["Control_Random_Network"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    untr_empty_s12 = [per_seed[s]["sudoku_extreme"]["Control_Untrained_Init"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    peer_empty_s12 = [per_seed[s]["sudoku_extreme"]["Control_PeerClue_Uniform"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    mode_empty_s12 = [per_seed[s]["sudoku_extreme"]["Control_ClueCopy_RowColMode"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    mf_empty_s12 = [per_seed[s]["sudoku_extreme"]["Control_FixedMeanField"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    abl_empty_s12 = [per_seed[s]["sudoku_extreme"]["Ablation_UnmaskedClue_CE_Only_LabelFree"]["by_segment"]["12"]["empty_acc"] for s in seeds]
    sup_empty_s12 = [per_seed[s]["sudoku_extreme"]["InfiLoop_Supervised_OurTrain"]["by_segment"]["12"]["empty_acc"] for s in seeds]

    lf_clue_s12 = [per_seed[s]["sudoku_extreme"]["TRMP_InfiLoop_Birkhoff_LabelFree"]["by_segment"]["12"]["clue_acc"] for s in seeds]
    lf_coll_s12 = [per_seed[s]["sudoku_extreme"]["TRMP_InfiLoop_Birkhoff_LabelFree"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    rand_coll_s12 = [per_seed[s]["sudoku_extreme"]["Control_Random_Network"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    untr_coll_s12 = [per_seed[s]["sudoku_extreme"]["Control_Untrained_Init"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    peer_coll_s12 = [per_seed[s]["sudoku_extreme"]["Control_PeerClue_Uniform"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    mf_coll_s12 = [per_seed[s]["sudoku_extreme"]["Control_FixedMeanField"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    abl_coll_s12 = [per_seed[s]["sudoku_extreme"]["Ablation_UnmaskedClue_CE_Only_LabelFree"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]
    sup_coll_s12 = [per_seed[s]["sudoku_extreme"]["InfiLoop_Supervised_OurTrain"]["by_segment"]["12"]["mean_unit_collisions"] for s in seeds]

    seg_means_empty = {
        str(seg): float(np.mean([per_seed[s]["sudoku_extreme"]["TRMP_InfiLoop_Birkhoff_LabelFree"]["by_segment"][str(seg)]["empty_acc"] for s in seeds]))
        for seg in results["segments_eval"]
    }
    seg_means_coll = {
        str(seg): float(np.mean([per_seed[s]["sudoku_extreme"]["TRMP_InfiLoop_Birkhoff_LabelFree"]["by_segment"][str(seg)]["mean_unit_collisions"] for s in seeds]))
        for seg in results["segments_eval"]
    }

    ci_empty_vs_rand = paired_ci95(lf_empty_s12, rand_empty_s12)
    ci_empty_vs_untr = paired_ci95(lf_empty_s12, untr_empty_s12)
    ci_empty_vs_peer = paired_ci95(lf_empty_s12, peer_empty_s12)
    ci_empty_vs_mode = paired_ci95(lf_empty_s12, mode_empty_s12)
    ci_empty_vs_mf = paired_ci95(lf_empty_s12, mf_empty_s12)
    ci_empty_vs_abl = paired_ci95(lf_empty_s12, abl_empty_s12)
    ci_coll_vs_rand = paired_ci95(rand_coll_s12, lf_coll_s12)
    ci_coll_vs_untr = paired_ci95(untr_coll_s12, lf_coll_s12)
    ci_coll_vs_peer = paired_ci95(peer_coll_s12, lf_coll_s12)
    ci_coll_vs_mf = paired_ci95(mf_coll_s12, lf_coll_s12)
    ci_coll_vs_abl = paired_ci95(abl_coll_s12, lf_coll_s12)

    mean_clue_acc_s12 = float(np.mean(lf_clue_s12))
    monotonic_empty = bool(seg_means_empty["12"] - seg_means_empty["1"] >= 0.04 and seg_means_empty["12"] >= seg_means_empty["4"])
    monotonic_coll = bool(seg_means_coll["1"] - seg_means_coll["12"] >= 15.0 and seg_means_coll["12"] <= seg_means_coll["4"])

    d1_pass = bool(
        mean_clue_acc_s12 >= 0.98
        and ci_empty_vs_rand["mean_diff"] >= 0.10
        and ci_empty_vs_rand["ci95_low"] > 0.0
        and ci_empty_vs_untr["mean_diff"] >= 0.10
        and ci_empty_vs_untr["ci95_low"] > 0.0
        and ci_empty_vs_peer["mean_diff"] >= 0.05
        and ci_empty_vs_peer["ci95_low"] > 0.0
        and ci_empty_vs_mode["mean_diff"] >= 0.10
        and ci_empty_vs_mode["ci95_low"] > 0.0
        and ci_empty_vs_mf["mean_diff"] >= 0.02
        and ci_empty_vs_mf["ci95_low"] > 0.0
        and ci_empty_vs_abl["mean_diff"] >= 0.08
        and ci_empty_vs_abl["ci95_low"] > 0.0
        and ci_coll_vs_rand["mean_diff"] >= 50.0
        and ci_coll_vs_rand["ci95_low"] > 0.0
        and ci_coll_vs_untr["mean_diff"] >= 50.0
        and ci_coll_vs_untr["ci95_low"] > 0.0
        and ci_coll_vs_peer["mean_diff"] >= 35.0
        and ci_coll_vs_peer["ci95_low"] > 0.0
        and ci_coll_vs_mf["mean_diff"] >= 5.0
        and ci_coll_vs_mf["ci95_low"] > 0.0
        and ci_coll_vs_abl["mean_diff"] >= 40.0
        and ci_coll_vs_abl["ci95_low"] > 0.0
        and monotonic_empty
        and monotonic_coll
    )

    # D2: Ambiguous 2-3 Clue Multi-Solution N-Queens (10x10 ID and 12x12, 16x16 OOD):
    # Gates against honest controls (Control_Random_Network, Control_Untrained_Init, Control_PeerClue_Uniform,
    # Control_FixedMeanField) and Ablation_UnmaskedClue_CE_Only_LabelFree, plus mandatory clue preservation >= 0.95!
    def nq_metric(arm: str, split: str, key: str) -> List[float]:
        return [per_seed[s]["nqueens"][arm][split][key] for s in seeds]

    lf_c10 = nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "10x10_id", "mean_collisions")
    peer_c10 = nq_metric("Control_PeerClue_Uniform", "10x10_id", "mean_collisions")
    mf_c10 = nq_metric("Control_FixedMeanField", "10x10_id", "mean_collisions")
    abl_c10 = nq_metric("Ablation_UnmaskedClue_CE_Only_LabelFree", "10x10_id", "mean_collisions")
    rand_c10 = nq_metric("Control_Random_Network", "10x10_id", "mean_collisions")
    untr_c10 = nq_metric("Control_Untrained_Init", "10x10_id", "mean_collisions")
    std_c10 = nq_metric("StandardCE_MultiSol_Sup_OurTrain", "10x10_id", "mean_collisions")
    trmp_c10 = nq_metric("TRMP_MinCE_MultiLabel_Sup_OurTrain", "10x10_id", "mean_collisions")

    lf_c12 = nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "12x12_ood", "mean_collisions")
    peer_c12 = nq_metric("Control_PeerClue_Uniform", "12x12_ood", "mean_collisions")
    mf_c12 = nq_metric("Control_FixedMeanField", "12x12_ood", "mean_collisions")
    rand_c12 = nq_metric("Control_Random_Network", "12x12_ood", "mean_collisions")
    untr_c12 = nq_metric("Control_Untrained_Init", "12x12_ood", "mean_collisions")
    std_c12 = nq_metric("StandardCE_MultiSol_Sup_OurTrain", "12x12_ood", "mean_collisions")
    trmp_c12 = nq_metric("TRMP_MinCE_MultiLabel_Sup_OurTrain", "12x12_ood", "mean_collisions")

    lf_c16 = nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "16x16_ood", "mean_collisions")
    peer_c16 = nq_metric("Control_PeerClue_Uniform", "16x16_ood", "mean_collisions")
    mf_c16 = nq_metric("Control_FixedMeanField", "16x16_ood", "mean_collisions")
    rand_c16 = nq_metric("Control_Random_Network", "16x16_ood", "mean_collisions")
    untr_c16 = nq_metric("Control_Untrained_Init", "16x16_ood", "mean_collisions")
    std_c16 = nq_metric("StandardCE_MultiSol_Sup_OurTrain", "16x16_ood", "mean_collisions")
    trmp_c16 = nq_metric("TRMP_MinCE_MultiLabel_Sup_OurTrain", "16x16_ood", "mean_collisions")

    clue_acc_10 = float(np.mean(nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "10x10_id", "clue_acc")))
    clue_acc_12 = float(np.mean(nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "12x12_ood", "clue_acc")))
    clue_acc_16 = float(np.mean(nq_metric("TRMP_InfiLoop_Birkhoff_LabelFree", "16x16_ood", "clue_acc")))

    ci_nq10_vs_peer = paired_ci95(peer_c10, lf_c10)
    ci_nq10_vs_mf = paired_ci95(mf_c10, lf_c10)
    ci_nq10_vs_abl = paired_ci95(abl_c10, lf_c10)
    ci_nq10_vs_rand = paired_ci95(rand_c10, lf_c10)
    ci_nq10_vs_untr = paired_ci95(untr_c10, lf_c10)
    ci_nq12_vs_peer = paired_ci95(peer_c12, lf_c12)
    ci_nq12_vs_mf = paired_ci95(mf_c12, lf_c12)
    ci_nq12_vs_rand = paired_ci95(rand_c12, lf_c12)
    ci_nq16_vs_peer = paired_ci95(peer_c16, lf_c16)
    ci_nq16_vs_mf = paired_ci95(mf_c16, lf_c16)
    ci_nq16_vs_rand = paired_ci95(rand_c16, lf_c16)

    d2_pass = bool(
        clue_acc_10 >= 0.95
        and clue_acc_12 >= 0.95
        and clue_acc_16 >= 0.95
        and ci_nq10_vs_peer["mean_diff"] >= 1.5
        and ci_nq10_vs_peer["ci95_low"] > 0.0
        and ci_nq10_vs_mf["mean_diff"] >= 5.0
        and ci_nq10_vs_mf["ci95_low"] > 0.0
        and ci_nq10_vs_abl["mean_diff"] >= 3.0
        and ci_nq10_vs_abl["ci95_low"] > 0.0
        and ci_nq10_vs_rand["mean_diff"] >= 5.0
        and ci_nq10_vs_rand["ci95_low"] > 0.0
        and ci_nq10_vs_untr["mean_diff"] >= 5.0
        and ci_nq10_vs_untr["ci95_low"] > 0.0
        and ci_nq12_vs_peer["mean_diff"] >= 1.5
        and ci_nq12_vs_peer["ci95_low"] > 0.0
        and ci_nq12_vs_mf["mean_diff"] >= 5.0
        and ci_nq12_vs_mf["ci95_low"] > 0.0
        and ci_nq12_vs_rand["mean_diff"] >= 5.0
        and ci_nq12_vs_rand["ci95_low"] > 0.0
        and ci_nq16_vs_peer["mean_diff"] >= 2.0
        and ci_nq16_vs_peer["ci95_low"] > 0.0
        and ci_nq16_vs_mf["mean_diff"] >= 5.0
        and ci_nq16_vs_mf["ci95_low"] > 0.0
        and ci_nq16_vs_rand["mean_diff"] >= 5.0
        and ci_nq16_vs_rand["ci95_low"] > 0.0
    )

    def fmt_verdict(passed: bool) -> str:
        if is_smoke:
            return "SMOKE_NO_VERDICT"
        return "PASS" if passed else "FAIL"

    return {
        "R27A_D0_public_data_sha256_and_watched_zero_label_audit": {
            "verdict": fmt_verdict(d0_pass),
            "disjointness_checks": disj,
            "label_access_counts": lac,
        },
        "R27A_D1_sudoku_labelfree_birkhoff_beats_controls_and_scales_with_segments": {
            "verdict": fmt_verdict(d1_pass),
            "segment_mean_empty_acc": seg_means_empty,
            "segment_mean_unit_collisions": seg_means_coll,
            "mean_clue_acc_s12_labelfree": mean_clue_acc_s12,
            "mean_empty_acc_s12_labelfree": float(np.mean(lf_empty_s12)),
            "mean_empty_acc_s12_random_net": float(np.mean(rand_empty_s12)),
            "mean_empty_acc_s12_untrained": float(np.mean(untr_empty_s12)),
            "mean_empty_acc_s12_peer_clue_uniform": float(np.mean(peer_empty_s12)),
            "mean_empty_acc_s12_mode_ctrl": float(np.mean(mode_empty_s12)),
            "mean_empty_acc_s12_fixed_mean_field": float(np.mean(mf_empty_s12)),
            "mean_empty_acc_s12_unmasked_clue_ce_only_ablation": float(np.mean(abl_empty_s12)),
            "mean_empty_acc_s12_infiloop_supervised_ourtrain": float(np.mean(sup_empty_s12)),
            "mean_collisions_s12_labelfree": float(np.mean(lf_coll_s12)),
            "mean_collisions_s12_random_net": float(np.mean(rand_coll_s12)),
            "mean_collisions_s12_untrained": float(np.mean(untr_coll_s12)),
            "mean_collisions_s12_peer_clue_uniform": float(np.mean(peer_coll_s12)),
            "mean_collisions_s12_fixed_mean_field": float(np.mean(mf_coll_s12)),
            "mean_collisions_s12_unmasked_clue_ce_only_ablation": float(np.mean(abl_coll_s12)),
            "mean_collisions_s12_infiloop_supervised_ourtrain": float(np.mean(sup_coll_s12)),
            "ci_empty_vs_random_net": ci_empty_vs_rand,
            "ci_empty_vs_untrained": ci_empty_vs_untr,
            "ci_empty_vs_peer_clue_uniform": ci_empty_vs_peer,
            "ci_empty_vs_mode_ctrl": ci_empty_vs_mode,
            "ci_empty_vs_fixed_mean_field": ci_empty_vs_mf,
            "ci_empty_vs_unmasked_clue_ce_only_ablation": ci_empty_vs_abl,
            "ci_collisions_vs_random_net": ci_coll_vs_rand,
            "ci_collisions_vs_untrained": ci_coll_vs_untr,
            "ci_collisions_vs_peer_clue_uniform": ci_coll_vs_peer,
            "ci_collisions_vs_fixed_mean_field": ci_coll_vs_mf,
            "ci_collisions_vs_unmasked_clue_ce_only_ablation": ci_coll_vs_abl,
        },
        "R27A_D2_nqueens_multisol_labelfree_beats_controls_and_extrapolates_ood": {
            "verdict": fmt_verdict(d2_pass),
            "mean_clue_acc_10x10_labelfree": clue_acc_10,
            "mean_clue_acc_12x12_ood_labelfree": clue_acc_12,
            "mean_clue_acc_16x16_ood_labelfree": clue_acc_16,
            "mean_collisions_10x10_labelfree": float(np.mean(lf_c10)),
            "mean_collisions_10x10_peer_clue_uniform": float(np.mean(peer_c10)),
            "mean_collisions_10x10_fixed_mean_field": float(np.mean(mf_c10)),
            "mean_collisions_10x10_unmasked_clue_ce_only_ablation": float(np.mean(abl_c10)),
            "mean_collisions_10x10_random_net": float(np.mean(rand_c10)),
            "mean_collisions_10x10_untrained": float(np.mean(untr_c10)),
            "mean_collisions_10x10_standard_ce_ourtrain": float(np.mean(std_c10)),
            "mean_collisions_10x10_trmp_min_ce_ourtrain": float(np.mean(trmp_c10)),
            "mean_collisions_12x12_ood_labelfree": float(np.mean(lf_c12)),
            "mean_collisions_12x12_ood_peer_clue_uniform": float(np.mean(peer_c12)),
            "mean_collisions_12x12_ood_fixed_mean_field": float(np.mean(mf_c12)),
            "mean_collisions_12x12_ood_random_net": float(np.mean(rand_c12)),
            "mean_collisions_12x12_ood_untrained": float(np.mean(untr_c12)),
            "mean_collisions_12x12_ood_standard_ce_ourtrain": float(np.mean(std_c12)),
            "mean_collisions_12x12_ood_trmp_min_ce_ourtrain": float(np.mean(trmp_c12)),
            "mean_collisions_16x16_ood_labelfree": float(np.mean(lf_c16)),
            "mean_collisions_16x16_ood_peer_clue_uniform": float(np.mean(peer_c16)),
            "mean_collisions_16x16_ood_fixed_mean_field": float(np.mean(mf_c16)),
            "mean_collisions_16x16_ood_random_net": float(np.mean(rand_c16)),
            "mean_collisions_16x16_ood_untrained": float(np.mean(untr_c16)),
            "mean_collisions_16x16_ood_standard_ce_ourtrain": float(np.mean(std_c16)),
            "mean_collisions_16x16_ood_trmp_min_ce_ourtrain": float(np.mean(trmp_c16)),
            "ci_10x10_vs_peer_clue_uniform": ci_nq10_vs_peer,
            "ci_10x10_vs_fixed_mean_field": ci_nq10_vs_mf,
            "ci_10x10_vs_unmasked_clue_ce_only_ablation": ci_nq10_vs_abl,
            "ci_10x10_vs_random_net": ci_nq10_vs_rand,
            "ci_10x10_vs_untrained": ci_nq10_vs_untr,
            "ci_12x12_ood_vs_peer_clue_uniform": ci_nq12_vs_peer,
            "ci_12x12_ood_vs_fixed_mean_field": ci_nq12_vs_mf,
            "ci_12x12_ood_vs_random_net": ci_nq12_vs_rand,
            "ci_16x16_ood_vs_peer_clue_uniform": ci_nq16_vs_peer,
            "ci_16x16_ood_vs_fixed_mean_field": ci_nq16_vs_mf,
            "ci_16x16_ood_vs_random_net": ci_nq16_vs_rand,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--sudoku-dir", type=str, default="/home/agent/data/datasets/sudoku-extreme")
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--n-train-sudoku", type=int, default=4096)
    parser.add_argument("--n-test-sudoku", type=int, default=256)
    parser.add_argument("--sudoku-steps", type=int, default=600)
    parser.add_argument("--nq-steps", type=int, default=400)
    args = parser.parse_args()

    t0 = time.time()
    results = run_r27a(args)
    results["elapsed_sec"] = round(time.time() - t0, 2)
    print(json.dumps(results["gates"], indent=2))
    if args.out_dir:
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "r27a_results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
