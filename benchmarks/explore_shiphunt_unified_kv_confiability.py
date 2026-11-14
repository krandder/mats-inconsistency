#!/usr/bin/env python3
"""
Track E Exploratory Runner (PROTOCOL.md §10.1):
Unified Compute, Data, and Memory (CDM) via Key-Query-Value Preconditioned Attention
and Confiability Head on ShipHunt-6x6-Unified-CDM
(explore_shiphunt_unified_kv_confiability.py)

EXPLORATORY ONLY — development splits only, zero touch of confirmatory test splits.
Evaluates Kelvin & opus-mats's unified formulation (research-orientation-2026-10-20-addendum.md):
  Any operation j that returns evidence is a rollout returning (Value v_j = Delta_ell_j, Key k_j, Cost c_j):
    1. Compute (THINK): simulating a world-model hypothesis / row-col-strip deduction (cost c_compute = 0.25)
    2. Data (LOOK): directly probing a board square (r, c) (cost c_data = 1.00)
    3. Memory (REMEMBER): retrieving a similar past episode from an episodic memory bank (cost c_memory = 0.40)

  Two Key-Query Attention Heads with strict parameter separation (params_search vs theta_trust in params_real):
    1. A Priori VoI Attention Selector (params_search, chi=(1,1) — Before running operation j):
       Estimates expected usefulness per unit cost:
         VoI_hat(j | Q, K_collected) = (Relevance(Q, k_hat_j) - Redundancy(k_hat_j, K_collected)) / c_j
       Compared against Exact Bayesian VoI*(j) across cost regimes and post-redundancy steps.
    2. A Posteriori Confiability Attention Head (theta_trust in params_real, chi=(0,1) — After operation j):
       Evaluates how trustworthy key k_j is specifically for answering query q_i:
         beta_{ij} = softplus(q_i^T W_trust k_j + b_trust(q_i, k_j))
       combined with Gram key-overlap de-duplication (G + lambda_0 I)^{-1},
       trained strictly against real ground-truth boards x* ~ D_real.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available():
    torch.set_num_threads(1)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between 1D arrays x and y with average tie handling."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 2:
        return 0.0

    def _ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(a[order[j + 1]] - a[order[i]]) <= 1e-12:
                j += 1
            avg_r = 0.5 * (i + j)
            for k in range(i, j + 1):
                ranks[order[k]] = avg_r
            i = j + 1
        return ranks

    rx = _ranks(x)
    ry = _ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((rx * ry).sum() / denom)


def compute_binary_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error for binary probabilities."""
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    y = np.asarray(targets, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return 0.0
    conf = np.maximum(p, 1.0 - p)
    pred = (p >= 0.5).astype(np.float64)
    acc = (pred == y).astype(np.float64)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n_tot = float(len(y))
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n_tot) * abs(float(acc[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


class ShipHuntUnifiedCDMEnv:
    """
    Exhaustively enumerated 6x6 ShipHunt environment with 4 Fleet Doctrine Rules (R in {0, 1, 2, 3})
    and Unified Compute (THINK), Data (LOOK), and Memory (REMEMBER) actions.

    Action Space (56 candidate operations total):
      - Data (LOOK): actions 0..35 (probe cell u in {0..35}, exact observation of (z_u, c_u))
      - Compute (THINK): actions 36..47 (12 strip consistency / world-model completion rollouts):
          * 36..41: Shallow deductive strip rollouts (depth d=1, high reliability 0.92 on strip s in {0..5},
            plus ungrounded board-completion speculation on cells outside strip s).
          * 42..47: Deep speculative multi-step rollouts (depth d=3, lower reliability 0.65 on strip s in {0..5},
            plus heavy uncalibrated speculative noise across non-strip cells).
      - Memory (REMEMBER): actions 48..55 (8 episodic memory bank retrievals):
          * 48..51: Fresh doctrine-matched memory banks on Quadrants Q_0..Q_3 (reliability 0.88 on
            quadrant density + doctrine rule signature, plus past-episode interference outside Q_b).
          * 52..55: Stale/shifted secondary memory banks on Quadrants Q_0..Q_3 (coupled to same quadrant
            feature, reliability 0.56, high retrieval surprise and heavy interference outside Q_b).

    Rule 9.6 Exact Bayesian Oracle:
      All H = 802 valid configurations across the 4 rules are enumerated in closed form with zero truncation.
      Every action a in {0..55} has an exact observation likelihood model P(O_a = o | h), making both the
      posterior P(h | H_t) and the Exact Bayesian Value of Information VoI*(a | H_t) closed-form and exact.
    """

    def __init__(self):
        self.n_cells = 36
        self.num_rules = 4
        self.num_data_actions = 36
        self.num_compute_actions = 12
        self.num_memory_actions = 8
        self.num_actions = 56
        self.max_outcomes = 10

        self.orth_mask = self._build_orthogonal_mask()
        self.strips = self._build_strips()  # 6 strips of 6 cells each
        self.quadrants = self._build_quadrants()  # 4 quadrants of 9 cells each
        self.cell_region_encoding = self._build_cell_region_encoding()  # [36, 10] (6 strips + 4 quads)

        (
            self.hypotheses_z,
            self.hypotheses_clue,
            self.hypotheses_rule,
            self.prior_w,
        ) = self._enumerate_all_hypotheses()
        self.num_hypotheses = len(self.prior_w)
        self.hypotheses_z_f64 = self.hypotheses_z.astype(np.float64)
        self.rule_onehot_f64 = np.eye(4, dtype=np.float64)[self.hypotheses_rule]
        self.pz_by_rule_prior = np.stack(
            [self.hypotheses_z[self.hypotheses_rule == r].mean(axis=0) for r in range(4)], axis=0
        ).astype(np.float32)

        (
            self.action_modalities,
            self.action_strips_or_quads,
            self.action_footprints,
            self.action_spillover_footprints,
            self.action_key_footprints,
            self.action_meta_static,
            self.channel_true_outcome,
            self.channel_num_outcomes,
            self.channel_likelihoods,
            self.channel_value_templates,
        ) = self._build_unified_cdm_channels()

    def _build_orthogonal_mask(self) -> np.ndarray:
        mask = np.zeros((36, 36), dtype=np.float32)
        for r1 in range(6):
            for c1 in range(6):
                i = r1 * 6 + c1
                for r2 in range(6):
                    for c2 in range(6):
                        j = r2 * 6 + c2
                        if abs(r1 - r2) + abs(c1 - c2) == 1:
                            mask[i, j] = 1.0
        return mask

    def _build_strips(self) -> List[List[int]]:
        strips: List[List[int]] = []
        for s in range(3):
            cells_h = [((2 * s) * 6 + c) for c in range(3)] + [((2 * s + 1) * 6 + c) for c in range(3)]
            strips.append(cells_h)
        for s in range(3):
            cells_v = [((2 * s) * 6 + c) for c in range(3, 6)] + [((2 * s + 1) * 6 + c) for c in range(3, 6)]
            strips.append(cells_v)
        return strips

    def _build_quadrants(self) -> List[List[int]]:
        quads: List[List[int]] = []
        for qr in range(2):
            for qc in range(2):
                cells = [(qr * 3 + dr) * 6 + (qc * 3 + dc) for dr in range(3) for dc in range(3)]
                quads.append(cells)
        return quads

    def _build_cell_region_encoding(self) -> np.ndarray:
        enc = np.zeros((36, 10), dtype=np.float32)
        for s, cells in enumerate(self.strips):
            for c in cells:
                enc[c, s] = 1.0
        for q, cells in enumerate(self.quadrants):
            for c in cells:
                enc[c, 6 + q] = 1.0
        return enc

    def _enumerate_all_hypotheses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_z, all_clues, all_r, all_w = [], [], [], []
        for rule_id in range(4):
            configs = []
            if rule_id == 0:
                for r1 in range(6):
                    for r2 in range(r1 + 1, 6):
                        for c1 in range(4):
                            for c2 in range(4):
                                z = np.zeros((6, 6), dtype=np.float32)
                                z[r1, c1 : c1 + 3] = 1.0
                                z[r2, c2 : c2 + 3] = 1.0
                                configs.append(z.reshape(36))
            elif rule_id == 1:
                for c1 in range(6):
                    for c2 in range(c1 + 1, 6):
                        for r1 in range(4):
                            for r2 in range(4):
                                z = np.zeros((6, 6), dtype=np.float32)
                                z[r1 : r1 + 3, c1] = 1.0
                                z[r2 : r2 + 3, c2] = 1.0
                                configs.append(z.reshape(36))
            elif rule_id == 2:
                for r1 in range(2):
                    for c1 in range(5):
                        for r2 in range(3, 6):
                            for c2 in range(5):
                                z = np.zeros((6, 6), dtype=np.float32)
                                z[r1 : r1 + 2, c1 : c1 + 2] = 1.0
                                z[r2, c2 : c2 + 2] = 1.0
                                configs.append(z.reshape(36))
            else:
                for r1 in range(4):
                    for c1 in range(4):
                        for r2 in range(4):
                            for c2 in range(2, 6):
                                z = np.zeros((6, 6), dtype=np.float32)
                                for k in range(3):
                                    z[r1 + k, c1 + k] = 1.0
                                    z[r2 + k, c2 - k] = 1.0
                                if z.sum() == 6:
                                    configs.append(z.reshape(36))

            z_arr = np.stack(configs, axis=0)
            clue_arr = z_arr @ self.orth_mask.T
            all_z.append(z_arr)
            all_clues.append(clue_arr)
            all_r.append(np.full(len(configs), rule_id, dtype=np.int64))
            all_w.append(np.full(len(configs), 0.25 / len(configs), dtype=np.float64))

        return (
            np.concatenate(all_z, axis=0),
            np.concatenate(all_clues, axis=0),
            np.concatenate(all_r, axis=0),
            np.concatenate(all_w, axis=0),
        )

    def _build_unified_cdm_channels(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        H = self.num_hypotheses
        A = self.num_actions
        K_max = self.max_outcomes

        modalities = np.zeros(A, dtype=np.int64)  # 0: Data, 1: Compute, 2: Memory
        group_id = np.zeros(A, dtype=np.int64)
        footprints = np.zeros((A, 36), dtype=np.float32)
        spillover_footprints = np.zeros((A, 36), dtype=np.float32)
        key_footprints = np.zeros((A, 46), dtype=np.float32)
        meta_static = np.zeros((A, 17), dtype=np.float32)

        true_outcome = np.zeros((A, H), dtype=np.int64)
        num_outcomes = np.zeros(A, dtype=np.int64)
        likelihoods = np.zeros((A, H, K_max), dtype=np.float64)

        for u in range(36):
            modalities[u] = 0
            group_id[u] = u
            footprints[u, u] = 1.0
            footprints[u] = np.maximum(footprints[u], 0.35 * self.orth_mask[u])
            spillover_footprints[u] = footprints[u].copy()
            key_footprints[u, :36] = footprints[u]
            meta_static[u, :7] = np.array(
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, footprints[u].sum() / 36.0], dtype=np.float32
            )
            meta_static[u, 7:] = self.cell_region_encoding[u]

            z_u = np.round(self.hypotheses_z[:, u]).astype(np.int64)
            c_u = np.clip(np.round(self.hypotheses_clue[:, u]).astype(np.int64), 0, 4)
            out_u = z_u * 5 + c_u
            true_outcome[u] = out_u
            num_outcomes[u] = 10
            for h_idx in range(H):
                likelihoods[u, h_idx, out_u[h_idx]] = 1.0

        for m in range(12):
            a = 36 + m
            s = m % 6
            is_deep = m >= 6
            modalities[a] = 1
            group_id[a] = 36 + s
            strip_cells = self.strips[s]
            for c_idx in strip_cells:
                footprints[a, c_idx] = 1.0
            spillover_footprints[a] = np.maximum(footprints[a], 0.90 if is_deep else 0.55)

            key_footprints[a, :36] = footprints[a]
            key_footprints[a, 36 + s] = 1.5
            depth_norm = 1.0 if is_deep else 0.20
            exp_cons = 0.50 if is_deep else 0.92
            meta_static[a, :7] = np.array(
                [1.0, 0.0, 0.0, depth_norm, exp_cons, 0.0 if is_deep else 1.0, footprints[a].sum() / 36.0],
                dtype=np.float32,
            )
            meta_static[a, 7 + s] = 1.0

            strip_sum = np.clip(np.round(self.hypotheses_z[:, strip_cells].sum(axis=1)).astype(np.int64), 0, 6)
            true_outcome[a] = strip_sum
            K_c = 7
            num_outcomes[a] = K_c
            acc = 0.65 if is_deep else 0.92
            err_p = (1.0 - acc) / float(K_c - 1)
            likelihoods[a, :, :K_c] = err_p
            for h_idx in range(H):
                likelihoods[a, h_idx, strip_sum[h_idx]] = acc

        for b in range(8):
            a = 48 + b
            q = b % 4
            is_stale = b >= 4
            modalities[a] = 2
            group_id[a] = 42 + q
            quad_cells = self.quadrants[q]
            for c_idx in quad_cells:
                footprints[a, c_idx] = 1.0
            spillover_footprints[a] = np.maximum(footprints[a], 0.95 if is_stale else 0.60)

            key_footprints[a, :36] = footprints[a]
            key_footprints[a, 42 + q] = 1.5
            exp_cons = 0.48 if is_stale else 0.88
            meta_static[a, :7] = np.array(
                [0.0, 0.0, 1.0, 0.65 if is_stale else 0.15, exp_cons, 0.0 if is_stale else 1.0, footprints[a].sum() / 36.0],
                dtype=np.float32,
            )
            meta_static[a, 13 + q] = 1.0

            quad_dense = (self.hypotheses_z[:, quad_cells].sum(axis=1) >= 2.0).astype(np.int64)
            mem_out = self.hypotheses_rule * 2 + quad_dense
            true_outcome[a] = mem_out
            K_m = 8
            num_outcomes[a] = K_m
            acc = 0.56 if is_stale else 0.88
            err_p = (1.0 - acc) / float(K_m - 1)
            likelihoods[a, :, :K_m] = err_p
            for h_idx in range(H):
                likelihoods[a, h_idx, mem_out[h_idx]] = acc

        norms = np.linalg.norm(key_footprints, axis=1, keepdims=True)
        key_footprints = key_footprints / np.maximum(norms, 1e-8)

        value_templates = np.zeros((A, K_max, 36), dtype=np.float32)
        base_p = (self.prior_w[:, None] * self.hypotheses_z).sum(axis=0)
        base_logit = np.log(np.clip(base_p, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - base_p, 1e-5, 1.0 - 1e-5))
        for a in range(A):
            K_a = int(num_outcomes[a])
            for o in range(K_a):
                w_o = self.prior_w * likelihoods[a, :, o]
                if w_o.sum() > 1e-12:
                    w_o = w_o / w_o.sum()
                    p_o = (w_o[:, None] * self.hypotheses_z).sum(axis=0)
                else:
                    p_o = base_p
                logit_o = np.log(np.clip(p_o, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - p_o, 1e-4, 1.0 - 1e-4))
                value_templates[a, o] = (logit_o - base_logit).astype(np.float32)

        return (
            modalities,
            group_id,
            footprints,
            spillover_footprints,
            key_footprints,
            meta_static,
            true_outcome,
            num_outcomes,
            likelihoods,
            value_templates,
        )

    def sample_episode(self, rng: np.random.RandomState, ep_id: int) -> Dict[str, Any]:
        h_star = int(rng.randint(0, self.num_hypotheses))
        z_star = self.hypotheses_z[h_star].copy()
        c_star = self.hypotheses_clue[h_star].copy()
        r_star = int(self.hypotheses_rule[h_star])

        realizations = np.zeros(self.num_actions, dtype=np.int64)
        for a in range(self.num_actions):
            probs = self.channel_likelihoods[a, h_star, : self.channel_num_outcomes[a]]
            probs = probs / probs.sum()
            realizations[a] = int(rng.choice(self.channel_num_outcomes[a], p=probs))

        # Also sample concrete rollouts/retrieved boards per Compute/Memory channel for this episode:
        # On the primary footprint (strip s or quad q), the rollout/memory emits strong cell-level log-odds
        # (accurate with probability matching the channel's precision, or inverted on noisy realizations),
        # while outside the primary footprint it emits ungrounded completion/past-episode interference!
        rollout_v = np.zeros((self.num_actions, 36), dtype=np.float32)
        z_signed = 2.0 * z_star - 1.0  # in {-1, +1}^36
        for a in range(self.num_actions):
            o = int(realizations[a])
            fp = self.action_footprints[a]
            non_fp = 1.0 - fp
            if a < 36:
                rollout_v[a] = 1.35 * self.channel_value_templates[a, o] * fp
            elif 36 <= a < 42:
                # Shallow deductive compute: high accuracy on strip s, ungrounded completion outside strip s
                is_correct = o == int(self.channel_true_outcome[a, h_star])
                sign_fp = z_signed if is_correct else -z_signed
                spill = rng.normal(0.0, 1.45, size=36).astype(np.float32) * non_fp
                rollout_v[a] = (1.85 * sign_fp + 0.8 * self.channel_value_templates[a, o]) * fp + spill
            elif 42 <= a < 48:
                # Deep speculative compute: noisy on strip s, strong uncalibrated speculation outside strip s
                is_correct = o == int(self.channel_true_outcome[a, h_star])
                sign_fp = z_signed if is_correct else -z_signed
                spill = rng.normal(0.0, 2.35, size=36).astype(np.float32) * non_fp
                rollout_v[a] = (1.75 * sign_fp + 0.6 * self.channel_value_templates[a, o]) * fp + spill
            elif 48 <= a < 52:
                # Fresh memory bank: reliable on quadrant q, past-episode interference outside quadrant q
                is_correct = o == int(self.channel_true_outcome[a, h_star])
                sign_fp = z_signed if is_correct else -z_signed
                spill = rng.normal(0.0, 1.50, size=36).astype(np.float32) * non_fp
                rollout_v[a] = (1.70 * sign_fp + 0.8 * self.channel_value_templates[a, o]) * fp + spill
            else:
                # Stale memory bank: noisy on quadrant q, heavy stale prototype bias outside quadrant q
                is_correct = o == int(self.channel_true_outcome[a, h_star])
                sign_fp = z_signed if is_correct else -z_signed
                spill = rng.normal(0.0, 2.40, size=36).astype(np.float32) * non_fp
                rollout_v[a] = (1.80 * sign_fp + 0.6 * self.channel_value_templates[a, o]) * fp + spill

        # Start with 1 revealed cell so the board takes 5+ steps to resolve
        zeros = np.where(z_star < 0.5)[0]
        init_cells = [int(rng.choice(zeros))]

        focus_quad = int(rng.randint(0, 4))
        adj_quad = (focus_quad + 1) % 4
        u_weights = np.full(36, 0.35, dtype=np.float32)
        for idx in self.quadrants[focus_quad]:
            u_weights[idx] = 2.8
        for idx in self.quadrants[adj_quad]:
            u_weights[idx] = 1.15
        u_weights = u_weights / u_weights.mean()

        return {
            "ep_id": ep_id,
            "h_star": h_star,
            "z_star": z_star,
            "c_star": c_star,
            "r_star": r_star,
            "realizations": realizations,
            "rollout_v": rollout_v,
            "init_cells": init_cells,
            "u_weights": u_weights,
            "focus_quad": focus_quad,
        }

    def exact_posterior_from_history(
        self, executed_actions: List[int], realized_outcomes: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = self.prior_w.copy()
        seen_actions = set()
        for a, o in zip(executed_actions, realized_outcomes):
            if a in seen_actions:
                continue
            seen_actions.add(a)
            w = w * self.channel_likelihoods[a, :, o]
        w_sum = w.sum()
        if w_sum <= 1e-15:
            w = self.prior_w.copy()
            w_sum = w.sum()
        w = w / w_sum
        p_z = (w @ self.hypotheses_z_f64).astype(np.float64)
        p_rule = (w @ self.rule_onehot_f64).astype(np.float64)
        return w, p_z, p_rule

    @staticmethod
    def _binary_entropy_vec(p: np.ndarray) -> np.ndarray:
        p_c = np.clip(p, 1e-12, 1.0 - 1e-12)
        return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))

    def utility_weighted_entropy(self, w: np.ndarray, u_weights: np.ndarray) -> float:
        p_z = w @ self.hypotheses_z_f64
        h_cells = float((u_weights.astype(np.float64) * self._binary_entropy_vec(p_z)).sum())
        p_r = w @ self.rule_onehot_f64
        p_r_c = np.clip(p_r, 1e-12, 1.0)
        h_rule = -float(np.where(p_r > 1e-12, p_r_c * np.log(p_r_c), 0.0).sum())
        return h_cells + 0.5 * h_rule

    def exact_bayesian_voi_vector(
        self,
        w_curr: np.ndarray,
        executed_actions: List[int],
        u_weights: np.ndarray,
        action_costs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_ent = self.utility_weighted_entropy(w_curr, u_weights)
        seen_set = set(executed_actions)
        seen_groups = {int(self.action_strips_or_quads[a]) for a in seen_set}

        w_unnorm = self.channel_likelihoods * w_curr[None, :, None]
        p_out = w_unnorm.sum(axis=1)
        p_out_safe = np.maximum(p_out, 1e-15)
        w_cond = w_unnorm / p_out_safe[:, None, :]

        w_cond_flat = np.ascontiguousarray(np.transpose(w_cond, (0, 2, 1)).reshape(-1, self.num_hypotheses))
        pz_cond = (w_cond_flat @ self.hypotheses_z_f64).reshape(self.num_actions, self.max_outcomes, 36)
        pr_cond = (w_cond_flat @ self.rule_onehot_f64).reshape(self.num_actions, self.max_outcomes, 4)

        u_w64 = u_weights.astype(np.float64)
        ent_cells_cond = (self._binary_entropy_vec(pz_cond) * u_w64[None, None, :]).sum(axis=-1)
        pr_c = np.clip(pr_cond, 1e-12, 1.0)
        ent_rule_cond = -np.where(pr_cond > 1e-12, pr_c * np.log(pr_c), 0.0).sum(axis=-1)

        tot_ent_cond = ent_cells_cond + 0.5 * ent_rule_cond
        exp_post_ent = (p_out * tot_ent_cond).sum(axis=-1)
        ig_exact = np.maximum(0.0, base_ent - exp_post_ent)

        for a in range(self.num_actions):
            if a in seen_set:
                ig_exact[a] = 0.0
            elif int(self.action_strips_or_quads[a]) in seen_groups:
                ig_exact[a] *= 0.12

        voi_exact = ig_exact / np.maximum(action_costs.astype(np.float64), 1e-6)
        return ig_exact, voi_exact

    def verify_oracle_brute_force(self, num_episodes: int = 24) -> Dict[str, Any]:
        rng = np.random.RandomState(906)
        max_post_diff = 0.0
        max_voi_diff = 0.0
        costs = self.get_cost_regime("Regime_Balanced")

        for ep_i in range(num_episodes):
            ep = self.sample_episode(rng, ep_i)
            exec_a = [ep["init_cells"][0], 36 + (ep_i % 6), 48 + (ep_i % 4)]
            exec_o = [int(ep["realizations"][a]) for a in exec_a]

            w_vec, pz_vec, pr_vec = self.exact_posterior_from_history(exec_a, exec_o)
            _, voi_vec = self.exact_bayesian_voi_vector(w_vec, exec_a, ep["u_weights"], costs)

            w_bf = np.zeros(self.num_hypotheses, dtype=np.float64)
            seen = set()
            unique_pairs = []
            for a, o in zip(exec_a, exec_o):
                if a not in seen:
                    seen.add(a)
                    unique_pairs.append((a, o))
            for h_idx in range(self.num_hypotheses):
                lh = float(self.prior_w[h_idx])
                for a, o in unique_pairs:
                    lh *= float(self.channel_likelihoods[a, h_idx, o])
                w_bf[h_idx] = lh
            w_bf /= w_bf.sum()
            pz_bf = np.zeros(36, dtype=np.float64)
            pr_bf = np.zeros(4, dtype=np.float64)
            for h_idx in range(self.num_hypotheses):
                pz_bf += w_bf[h_idx] * self.hypotheses_z[h_idx]
                pr_bf[int(self.hypotheses_rule[h_idx])] += w_bf[h_idx]

            post_diff = max(
                float(np.max(np.abs(w_vec - w_bf))),
                float(np.max(np.abs(pz_vec - pz_bf))),
                float(np.max(np.abs(pr_vec - pr_bf))),
            )
            max_post_diff = max(max_post_diff, post_diff)

            base_ent_bf = self.utility_weighted_entropy(w_bf, ep["u_weights"])
            seen_groups = {int(self.action_strips_or_quads[a]) for a in seen}
            for cand_a in [0, 15, 37, 43, 49, 53]:
                if cand_a in seen:
                    voi_bf = 0.0
                else:
                    K_a = int(self.channel_num_outcomes[cand_a])
                    exp_ent = 0.0
                    for o in range(K_a):
                        w_o = np.zeros(self.num_hypotheses, dtype=np.float64)
                        for h_idx in range(self.num_hypotheses):
                            w_o[h_idx] = w_bf[h_idx] * float(self.channel_likelihoods[cand_a, h_idx, o])
                        p_o = float(w_o.sum())
                        if p_o > 1e-15:
                            w_o /= p_o
                            exp_ent += p_o * self.utility_weighted_entropy(w_o, ep["u_weights"])
                    ig_bf = max(0.0, base_ent_bf - exp_ent)
                    if int(self.action_strips_or_quads[cand_a]) in seen_groups:
                        ig_bf *= 0.12
                    voi_bf = ig_bf / float(costs[cand_a])
                max_voi_diff = max(max_voi_diff, abs(float(voi_vec[cand_a]) - float(voi_bf)))

        max_diff = max(max_post_diff, max_voi_diff)
        if max_diff >= 1e-12:
            raise RuntimeError(f"Rule 9.6 Oracle verification failed: max_diff={max_diff}")
        return {
            "episodes_checked": num_episodes,
            "max_posterior_abs_diff": float(max_post_diff),
            "max_voi_abs_diff": float(max_voi_diff),
            "max_abs_diff": float(max_diff),
            "verified": True,
        }

    def get_cost_regime(self, regime_name: str) -> np.ndarray:
        costs = np.zeros(self.num_actions, dtype=np.float32)
        if regime_name == "Regime_Balanced":
            c_data, c_comp, c_mem = 1.00, 0.25, 0.40
        elif regime_name == "Regime_CheapComputeMemory":
            c_data, c_comp, c_mem = 1.20, 0.12, 0.20
        elif regime_name == "Regime_CheapData":
            c_data, c_comp, c_mem = 0.35, 0.60, 0.70
        else:
            raise ValueError(f"Unknown cost regime: {regime_name}")
        costs[:36] = c_data
        costs[36:48] = c_comp
        costs[48:56] = c_mem
        return costs

    def build_query_features(
        self,
        p_z: np.ndarray,
        p_rule: np.ndarray,
        executed_actions: List[int],
        u_weights: np.ndarray,
    ) -> np.ndarray:
        p_c = np.clip(p_z, 1e-6, 1.0 - 1e-6)
        ent_i = -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))
        var_i = p_c * (1.0 - p_c)

        mean_pz_r = (p_rule[:, None] * self.pz_by_rule_prior).sum(axis=0)
        rule_disagree_i = ((p_rule[:, None] * (self.pz_by_rule_prior - mean_pz_r[None, :]) ** 2).sum(axis=0)).astype(
            np.float32
        )

        unrev_i = np.ones(36, dtype=np.float32)
        for a in executed_actions:
            if a < 36:
                unrev_i[a] = 0.0

        pr_c = np.clip(p_rule, 1e-6, 1.0)
        rule_ent = float(-(pr_c * np.log(pr_c)).sum())
        rows = np.array([i // 6 for i in range(36)], dtype=np.float32) / 5.0
        cols = np.array([i % 6 for i in range(36)], dtype=np.float32) / 5.0

        q_epistemic = np.stack(
            [
                ent_i.astype(np.float32),
                var_i.astype(np.float32),
                rule_disagree_i,
                unrev_i,
                u_weights.astype(np.float32),
                (u_weights * ent_i).astype(np.float32),
                (u_weights * rule_disagree_i).astype(np.float32),
                np.full(36, rule_ent, dtype=np.float32),
                rows,
                cols,
            ],
            axis=-1,
        )
        return np.concatenate([q_epistemic, self.cell_region_encoding], axis=-1)

    def execute_operation_and_build_evidence(
        self,
        ep: Dict[str, Any],
        action_a: int,
        p_z_prior: np.ndarray,
    ) -> Dict[str, Any]:
        outcome_o = int(ep["realizations"][action_a])
        v_j = ep["rollout_v"][action_a].copy()
        fp_true = self.action_footprints[action_a].copy()
        fp_spill = self.action_spillover_footprints[action_a].copy()
        k_fp = self.action_key_footprints[action_a].copy()
        static_m = self.action_meta_static[action_a].copy()

        prior_logit = np.log(np.clip(p_z_prior, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - p_z_prior, 1e-4, 1.0 - 1e-4))
        post_p = 1.0 / (1.0 + np.exp(-(prior_logit + v_j)))
        kl_shift = np.abs(post_p - p_z_prior)
        surprise_j = float((kl_shift * fp_spill).sum() / max(float(fp_spill.sum()), 1.0))

        spill_only = np.maximum(0.0, fp_spill - fp_true)
        spill_energy = float(np.abs(v_j * spill_only).mean())
        primary_energy = float(np.abs(v_j * fp_true).mean()) + 1e-4
        m_ql = float(static_m[4]) / (1.0 + 1.5 * (spill_energy / primary_energy))
        val_salience = float(np.linalg.norm(v_j) / 6.0)

        k_meta_realized = np.concatenate(
            [
                static_m,
                np.array([m_ql, surprise_j, val_salience], dtype=np.float32),
            ],
            axis=0,
        )

        return {
            "action": action_a,
            "outcome": outcome_o,
            "v_j": v_j,
            "fp_true": fp_true,
            "fp_spill": fp_spill,
            "k_fp": k_fp,
            "k_meta": k_meta_realized,
        }


class UnifiedKeyQueryVoIAttentionNet(nn.Module):
    """
    Head 1: A Priori VoI Attention Selector (params_search, chi=(1,1) — Before running operation j):
      VoI_hat(j | Q, K_collected) = [ Relevance(Q, k_hat_j) - Redundancy(k_hat_j, K_collected) ]_+ / c_j
    All relevance, redundancy, and cost-scaling projections are learned parameters in params_search
    with zero hardcoded feature shortcuts, so Control_Random_Network is a true untrained control.
    """

    def __init__(self, q_dim: int = 20, k_meta_dim: int = 17, d_model: int = 48):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(k_meta_dim + 36, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.w_voi = nn.Linear(d_model, d_model, bias=False)
        self.action_mlp = nn.Sequential(
            nn.Linear(d_model + k_meta_dim + 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.cost_exp = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.red_weight = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))
        self.rep_weight = nn.Parameter(torch.tensor(-2.5, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.12)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.05)

    def forward(
        self,
        q_feat: torch.Tensor,
        k_meta_cand: torch.Tensor,
        cand_footprints: torch.Tensor,
        gram_overlap_precomp: torch.Tensor,
        repeat_mask: torch.Tensor,
        action_costs: torch.Tensor,
        mode: str = "full_unified",
    ) -> torch.Tensor:
        """
        q_feat: [B, 36, 20]
        k_meta_cand: [56, 17]
        cand_footprints: [56, 36]
        gram_overlap_precomp: [B, 56] Gram projection overlap of cand_key_fp onto K_collected
        repeat_mask: [B, 56]
        action_costs: [B, 56]
        """
        bsz = q_feat.shape[0]
        device = q_feat.device

        k_in = torch.cat([k_meta_cand, cand_footprints], dim=-1)  # [56, 53]
        k_emb = self.k_proj(k_in)  # [56, d]

        if mode == "key_only_no_query":
            # Query-blind key coverage (ignores q_feat!)
            q_pooled = torch.zeros((bsz, 56, self.d_model), device=device)
            att_summary = cand_footprints.sum(dim=-1).unsqueeze(0).expand(bsz, 56) / 6.0
        else:
            q_emb = self.q_proj(q_feat)  # [B, 36, d]
            wk = self.w_voi(k_emb)  # [56, d]
            bilin = torch.einsum("bid,jd->bij", q_emb, wk) / math.sqrt(float(self.d_model))  # [B, 36, 56]
            att_weights = F.softmax(bilin, dim=1)  # [B, 36, 56]
            q_pooled = torch.einsum("bij,bid->bjd", att_weights, q_emb)  # [B, 56, d]
            att_summary = (F.softplus(bilin) * cand_footprints.T.unsqueeze(0)).mean(dim=1)  # [B, 56]

        if mode == "no_redundancy":
            overlap = torch.zeros((bsz, 56), device=device)
            rep_eff = torch.zeros((bsz, 56), device=device)
        else:
            overlap = gram_overlap_precomp
            rep_eff = repeat_mask

        k_meta_exp = k_meta_cand.unsqueeze(0).expand(bsz, 56, -1)
        aux = torch.stack([att_summary, overlap, rep_eff], dim=-1)
        mlp_in = torch.cat([q_pooled, k_meta_exp, aux], dim=-1)
        relevance = F.softplus(self.action_mlp(mlp_in).squeeze(-1))

        if mode == "no_redundancy":
            net_ig = relevance
        else:
            red_pen = (torch.sigmoid(self.red_weight) * overlap + torch.sigmoid(self.rep_weight) * rep_eff).clamp(
                min=0.0, max=1.0
            )
            net_ig = relevance * (1.0 - red_pen)

        # Learned cost exponent: initialized at 0.0 so Control_Random_Network does not get free cost-division ranking,
        # while trained params_search learns cost_exp -> 1.0!
        c_pow = torch.pow(action_costs.clamp(min=1e-4), torch.sigmoid(self.cost_exp) * 1.3)
        return net_ig / c_pow


def compute_gram_overlap_with_collected(
    collected_key_fp: np.ndarray, cand_key_fp: np.ndarray, lam_red: float = 0.12
) -> np.ndarray:
    """
    Computes Gram subspace projection overlap k_j^T K_col^T (K_col K_col^T + lam_red I)^{-1} K_col k_j
    for candidate keys cand_key_fp [56, 46] against collected keys collected_key_fp [M_t, 46].
    """
    if collected_key_fp is None or len(collected_key_fp) == 0:
        return np.zeros(len(cand_key_fp), dtype=np.float32)
    m_t = len(collected_key_fp)
    gram_col = collected_key_fp @ collected_key_fp.T + lam_red * np.eye(m_t, dtype=np.float32)
    proj_rhs = collected_key_fp @ cand_key_fp.T  # (M_t, 56)
    sol = np.linalg.solve(gram_col, proj_rhs)
    overlap = np.clip((proj_rhs * sol).sum(axis=0), 0.0, 1.0)
    return overlap.astype(np.float32)


class ConfiabilityAttentionClearingHead(nn.Module):
    """
    Head 2: A Posteriori Confiability Attention Head (theta_trust in params_real, chi=(0,1) — After operation j):
      Evaluates how trustworthy key k_j is specifically for answering query q_i:
        beta_{ij} = softplus(q_i^T W_trust k_j + b_trust(q_i, k_j))
      combined with Gram key-overlap de-duplication:
        c_dedup = (1 + lambda_0) (G + lambda_0 I_M)^{-1} 1_M
        Delta_ell_i^clear = sum_{j=1}^M beta_{ij} c_{dedup, j} v_{j, i}
    """

    def __init__(self, q_dim: int = 20, k_meta_dim: int = 20, d_model: int = 48):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(k_meta_dim + 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.w_trust = nn.Linear(d_model, d_model, bias=False)
        self.pair_trust_mlp = nn.Sequential(
            nn.Linear(q_dim + k_meta_dim + 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.scalar_branch_mlp = nn.Sequential(
            nn.Linear(k_meta_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.log_lambda_0 = nn.Parameter(torch.tensor(-2.8, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.08)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        prior_logits: torch.Tensor,
        q_feat: torch.Tensor,
        k_meta: torch.Tensor,
        fp_true: torch.Tensor,
        fp_spill: torch.Tensor,
        k_fp: torch.Tensor,
        v_updates: torch.Tensor,
        mode: str = "query_key_gram",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, m_len, _ = v_updates.shape
        device = v_updates.device

        if mode == "unweighted_uniform":
            cleared_logits = prior_logits + v_updates.sum(dim=1)
            beta_uniform = torch.ones((bsz, 36, m_len), device=device)
            return torch.sigmoid(cleared_logits), beta_uniform

        if mode == "scalar_branch_trust":
            beta_j = F.softplus(self.scalar_branch_mlp(k_meta).squeeze(-1) - 1.2)  # [B, M]
            cleared_logits = prior_logits + (beta_j.unsqueeze(-1) * v_updates).sum(dim=1)
            return torch.sigmoid(cleared_logits), beta_j.unsqueeze(1).expand(bsz, 36, m_len)

        fp_t_ij = fp_true.permute(0, 2, 1).unsqueeze(-1)  # [B, 36, M, 1]
        spill_only_ij = (fp_spill - fp_true).clamp(min=0.0).permute(0, 2, 1).unsqueeze(-1)  # [B, 36, M, 1]
        k_meta_exp = k_meta.unsqueeze(1).expand(bsz, 36, m_len, -1)  # [B, 36, M, 20]
        k_in = torch.cat([k_meta_exp, fp_t_ij, spill_only_ij], dim=-1)  # [B, 36, M, 22]

        q_emb = self.q_proj(q_feat)  # [B, 36, d]
        k_emb = self.k_proj(k_in)  # [B, 36, M, d]
        wk = self.w_trust(k_emb)  # [B, 36, M, d]
        bilin = (q_emb.unsqueeze(2) * wk).sum(dim=-1) / math.sqrt(float(self.d_model))  # [B, 36, M]

        q_exp = q_feat.unsqueeze(2).expand(bsz, 36, m_len, -1)
        pair_in = torch.cat([q_exp, k_in], dim=-1)
        raw_ij = bilin + self.pair_trust_mlp(pair_in).squeeze(-1) - 1.2
        beta_ij = F.softplus(raw_ij)  # [B, 36, M]

        v_ij = v_updates.permute(0, 2, 1)  # [B, 36, M]

        if mode == "query_key_nogram":
            cleared_logits = prior_logits + (beta_ij * v_ij).sum(dim=-1)
            return torch.sigmoid(cleared_logits), beta_ij

        b_mean_j = (beta_ij * fp_t_ij.squeeze(-1)).sum(dim=1) / fp_t_ij.squeeze(-1).sum(dim=1).clamp(min=1.0)
        gamma_j = (b_mean_j / b_mean_j.max(dim=1, keepdim=True).values.clamp(min=1e-4)).clamp(min=0.0, max=1.0)
        raw_gram = torch.bmm(k_fp, k_fp.transpose(1, 2)).clamp(min=0.0)
        eye_m = torch.eye(m_len, device=device).unsqueeze(0).expand(bsz, m_len, m_len)
        off_diag = 1.0 - eye_m
        gamma_pair = torch.sqrt((gamma_j.unsqueeze(2) * gamma_j.unsqueeze(1)).clamp(min=1e-8))
        gram_eff = eye_m + off_diag * gamma_pair * raw_gram

        lam_0 = F.softplus(self.log_lambda_0).clamp(min=0.02, max=0.35)
        ones_m = torch.ones((bsz, m_len, 1), device=device)
        c_dedup = (1.0 + lam_0) * torch.linalg.solve(gram_eff + lam_0 * eye_m, ones_m).squeeze(-1)
        c_dedup = c_dedup.clamp(min=0.05, max=1.05)

        w_clear_ij = beta_ij * c_dedup.unsqueeze(1)
        cleared_logits = prior_logits + (w_clear_ij * v_ij).sum(dim=-1)
        return torch.sigmoid(cleared_logits), beta_ij


def build_confiability_training_step_batch(
    env: ShipHuntUnifiedCDMEnv,
    episodes: List[Dict[str, Any]],
    rng: np.random.RandomState,
    device: torch.device,
    stress_duplicates: bool = False,
) -> Dict[str, Any]:
    prior_logits_list = []
    q_feat_list = []
    k_meta_list = []
    fp_true_list = []
    fp_spill_list = []
    k_fp_list = []
    v_upd_list = []
    z_target_list = []
    exact_post_list = []
    total_cost_list = []

    costs = env.get_cost_regime("Regime_Balanced")
    base_p0 = (env.prior_w[:, None] * env.hypotheses_z).sum(axis=0).astype(np.float32)
    base_logit0 = np.log(np.clip(base_p0, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - base_p0, 1e-4, 1.0 - 1e-4))

    for ep in episodes:
        init_a = list(ep["init_cells"])
        init_o = [int(ep["realizations"][a]) for a in init_a]
        _, pz_init, pr_init = env.exact_posterior_from_history(init_a, init_o)
        q_feat = env.build_query_features(pz_init, pr_init, init_a, ep["u_weights"])

        s1, s2 = int(rng.randint(0, 3)), int(rng.randint(3, 6))
        q1, q2 = int(rng.randint(0, 2)), int(rng.randint(2, 4))
        extra_data = int(rng.choice([u for u in range(36) if u not in init_a]))

        cand_seq = [
            init_a[0],
            extra_data,
            36 + s1,              # shallow compute 1
            36 + s2,              # shallow compute 2
            42 + s1,              # deep speculative compute 1 (overlaps strip s1)
            42 + ((s2 + 1) % 6),  # deep speculative compute 2
            48 + q1,              # fresh memory 1
            48 + q2,              # fresh memory 2
            52 + q1,              # stale memory 1 (overlaps quad q1)
        ]
        if stress_duplicates:
            # Add 6 redundant executions of shallow/deep compute & memory to test Gram de-duplication
            cand_seq.extend([36 + s1, 36 + s1, 36 + s2, 48 + q1, 48 + q1, 48 + q2])

        ev_k_meta, ev_fp_t, ev_fp_s, ev_k_fp, ev_v = [], [], [], [], []
        exec_a_all, exec_o_all = [], []
        tot_c = 0.0
        for a in cand_seq:
            o = int(ep["realizations"][a])
            exec_a_all.append(a)
            exec_o_all.append(o)
            tot_c += float(costs[a])
            ev = env.execute_operation_and_build_evidence(ep, a, base_p0)
            ev_k_meta.append(ev["k_meta"])
            ev_fp_t.append(ev["fp_true"])
            ev_fp_s.append(ev["fp_spill"])
            ev_k_fp.append(ev["k_fp"])
            ev_v.append(ev["v_j"])

        _, pz_exact, _ = env.exact_posterior_from_history(exec_a_all, exec_o_all)

        prior_logits_list.append(base_logit0)
        q_feat_list.append(q_feat)
        k_meta_list.append(np.stack(ev_k_meta, axis=0))
        fp_true_list.append(np.stack(ev_fp_t, axis=0))
        fp_spill_list.append(np.stack(ev_fp_s, axis=0))
        k_fp_list.append(np.stack(ev_k_fp, axis=0))
        v_upd_list.append(np.stack(ev_v, axis=0))
        z_target_list.append(ep["z_star"])
        exact_post_list.append(pz_exact.astype(np.float32))
        total_cost_list.append(tot_c)

    return {
        "prior_logits": torch.tensor(np.stack(prior_logits_list), dtype=torch.float32, device=device),
        "q_feat": torch.tensor(np.stack(q_feat_list), dtype=torch.float32, device=device),
        "k_meta": torch.tensor(np.stack(k_meta_list), dtype=torch.float32, device=device),
        "fp_true": torch.tensor(np.stack(fp_true_list), dtype=torch.float32, device=device),
        "fp_spill": torch.tensor(np.stack(fp_spill_list), dtype=torch.float32, device=device),
        "k_fp": torch.tensor(np.stack(k_fp_list), dtype=torch.float32, device=device),
        "v_updates": torch.tensor(np.stack(v_upd_list), dtype=torch.float32, device=device),
        "z_target": torch.tensor(np.stack(z_target_list), dtype=torch.float32, device=device),
        "exact_post": torch.tensor(np.stack(exact_post_list), dtype=torch.float32, device=device),
        "total_cost": np.array(total_cost_list, dtype=np.float64),
    }


def train_voi_and_confiability_heads(
    env: ShipHuntUnifiedCDMEnv,
    train_episodes: List[Dict[str, Any]],
    seed: int,
    device: torch.device,
    smoke_cpu: bool = False,
) -> Tuple[UnifiedKeyQueryVoIAttentionNet, ConfiabilityAttentionClearingHead, Dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed + 1000)

    voi_net = UnifiedKeyQueryVoIAttentionNet().to(device)
    trust_net = ConfiabilityAttentionClearingHead().to(device)

    trust_init_snap = {k: v.detach().clone() for k, v in trust_net.named_parameters()}
    for p in trust_net.parameters():
        p.requires_grad_(False)

    regimes = ["Regime_Balanced", "Regime_CheapComputeMemory", "Regime_CheapData"]
    q_feat_samples, gram_ov_samples, rep_mask_samples, cost_samples, target_voi_samples = [], [], [], [], []
    n_voitrain = len(train_episodes)

    for idx in range(n_voitrain):
        ep = train_episodes[idx]
        for reg_name in regimes:
            costs = env.get_cost_regime(reg_name)
            for state_type in (0, 1):
                exec_a = list(ep["init_cells"])
                if state_type == 1:
                    exec_a.extend([36 + (ep["ep_id"] % 6), 48 + (ep["ep_id"] % 4)])
                    if idx % 2 == 0:
                        exec_a.append(36 + ((ep["ep_id"] + 1) % 6))
                exec_o = [int(ep["realizations"][a]) for a in exec_a]
                w_curr, pz_curr, pr_curr = env.exact_posterior_from_history(exec_a, exec_o)
                q_f = env.build_query_features(pz_curr, pr_curr, exec_a, ep["u_weights"])
                _, voi_star = env.exact_bayesian_voi_vector(w_curr, exec_a, ep["u_weights"], costs)

                col_k = np.stack([env.action_key_footprints[a_prev] for a_prev in exec_a], axis=0)
                gram_ov = compute_gram_overlap_with_collected(col_k, env.action_key_footprints)

                rep_m = np.zeros(56, dtype=np.float32)
                for a_prev in exec_a:
                    rep_m[a_prev] = 1.0

                q_feat_samples.append(q_f)
                gram_ov_samples.append(gram_ov)
                rep_mask_samples.append(rep_m)
                cost_samples.append(costs)
                target_voi_samples.append(voi_star.astype(np.float32))

    q_feat_t = torch.tensor(np.stack(q_feat_samples), dtype=torch.float32, device=device)
    gram_ov_t = torch.tensor(np.stack(gram_ov_samples), dtype=torch.float32, device=device)
    rep_mask_t = torch.tensor(np.stack(rep_mask_samples), dtype=torch.float32, device=device)
    costs_t = torch.tensor(np.stack(cost_samples), dtype=torch.float32, device=device)
    target_voi_t = torch.tensor(np.stack(target_voi_samples), dtype=torch.float32, device=device)

    k_meta_cand_t = torch.tensor(env.action_meta_static, dtype=torch.float32, device=device)
    cand_fp_t = torch.tensor(env.action_footprints, dtype=torch.float32, device=device)

    opt_voi = torch.optim.AdamW(voi_net.parameters(), lr=5e-3, weight_decay=1e-4)
    voi_epochs = 60 if smoke_cpu else 180
    for _ in range(voi_epochs):
        opt_voi.zero_grad()
        pred_voi = voi_net(
            q_feat_t,
            k_meta_cand_t,
            cand_fp_t,
            gram_ov_t,
            rep_mask_t,
            costs_t,
            mode="full_unified",
        )
        mse_loss = F.smooth_l1_loss(pred_voi, target_voi_t)
        p_targ = F.softmax(target_voi_t * 8.0, dim=-1)
        log_p_pred = F.log_softmax(pred_voi * 8.0, dim=-1)
        kl_loss = F.kl_div(log_p_pred, p_targ, reduction="batchmean")
        loss = mse_loss + 0.6 * kl_loss
        loss.backward()
        opt_voi.step()

    max_trust_drift = max(
        float((v.detach() - trust_init_snap[k]).abs().max().item()) for k, v in trust_net.named_parameters()
    )

    voi_snap = {k: v.detach().clone() for k, v in voi_net.named_parameters()}
    for p in voi_net.parameters():
        p.requires_grad_(False)
    for p in trust_net.parameters():
        p.requires_grad_(True)

    # Train trust_net on standard non-duplicate multi-modal pools from D_real (and Gram de-duplication
    # automatically generalizes to duplicate stress pools via the Gram inverse!)
    batch_std = build_confiability_training_step_batch(env, train_episodes, rng, device, stress_duplicates=False)

    opt_trust = torch.optim.AdamW(trust_net.parameters(), lr=5e-3, weight_decay=1e-4)
    trust_epochs = 60 if smoke_cpu else 180
    for _ in range(trust_epochs):
        opt_trust.zero_grad()
        probs_qg, beta_qg = trust_net(
            batch_std["prior_logits"],
            batch_std["q_feat"],
            batch_std["k_meta"],
            batch_std["fp_true"],
            batch_std["fp_spill"],
            batch_std["k_fp"],
            batch_std["v_updates"],
            mode="query_key_gram",
        )
        probs_sc, _ = trust_net(
            batch_std["prior_logits"],
            batch_std["q_feat"],
            batch_std["k_meta"],
            batch_std["fp_true"],
            batch_std["fp_spill"],
            batch_std["k_fp"],
            batch_std["v_updates"],
            mode="scalar_branch_trust",
        )
        p_qg_c = probs_qg.clamp(1e-6, 1.0 - 1e-6)
        p_sc_c = probs_sc.clamp(1e-6, 1.0 - 1e-6)
        y = batch_std["z_target"]
        loss_qg = F.binary_cross_entropy(p_qg_c, y) + 0.5 * ((p_qg_c - y) ** 2).mean()
        loss_sc = F.binary_cross_entropy(p_sc_c, y) + 0.5 * ((p_sc_c - y) ** 2).mean()
        l1_reg = 0.002 * beta_qg.mean()
        (loss_qg + 0.5 * loss_sc + l1_reg).backward()
        opt_trust.step()

    max_voi_drift = max(
        float((v.detach() - voi_snap[k]).abs().max().item()) for k, v in voi_net.named_parameters()
    )

    firewall_check = {
        "params_search_trained_with_theta_trust_frozen": True,
        "theta_trust_max_abs_drift_during_search_training": max_trust_drift,
        "theta_trust_trained_strictly_on_real_boards_z_star": True,
        "params_search_max_abs_drift_during_trust_training": max_voi_drift,
    }
    return voi_net, trust_net, firewall_check


@torch.no_grad()
def evaluate_part1_voi_metareasoning(
    env: ShipHuntUnifiedCDMEnv,
    dev_episodes: List[Dict[str, Any]],
    voi_net: UnifiedKeyQueryVoIAttentionNet,
    rand_voi_net: UnifiedKeyQueryVoIAttentionNet,
    trust_net: ConfiabilityAttentionClearingHead,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    regimes = ["Regime_Balanced", "Regime_CheapComputeMemory", "Regime_CheapData"]
    policies = [
        "Control_Random_Network",
        "Control_Uniform_Random_Action",
        "Policy_Always_Data_Only",
        "Policy_Always_Compute_Only",
        "Policy_Always_Memory_Only",
        "Policy_KeyOnly_NoQuery_PUCT",
        "Policy_NoRedundancy_Penalty",
        "Policy_Unified_KeyQuery_VoI_Attention",
        "Ceiling_Exact_Bayesian_VoI_Oracle",
    ]

    k_meta_cand_t = torch.tensor(env.action_meta_static, dtype=torch.float32, device=device)
    cand_fp_t = torch.tensor(env.action_footprints, dtype=torch.float32, device=device)
    base_p0 = (env.prior_w[:, None] * env.hypotheses_z).sum(axis=0).astype(np.float32)
    base_logit0 = np.log(np.clip(base_p0, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - base_p0, 1e-4, 1.0 - 1e-4))

    def score_candidates_for_policy(
        pol_name: str,
        q_feat_np: np.ndarray,
        q_feat_init_np: np.ndarray,
        exec_actions: List[int],
        costs_np: np.ndarray,
        voi_exact_np: np.ndarray,
        rng_pol: np.random.RandomState,
    ) -> np.ndarray:
        if pol_name == "Ceiling_Exact_Bayesian_VoI_Oracle":
            return voi_exact_np.copy()
        if pol_name == "Control_Uniform_Random_Action":
            return rng_pol.uniform(0.0, 1.0, size=56).astype(np.float32)

        q_t = torch.tensor(q_feat_np[None, :, :], dtype=torch.float32, device=device)
        q_init_t = torch.tensor(q_feat_init_np[None, :, :], dtype=torch.float32, device=device)
        col_k_np = np.stack([env.action_key_footprints[a] for a in exec_actions], axis=0)
        gram_ov_np = compute_gram_overlap_with_collected(col_k_np, env.action_key_footprints)
        gram_ov_t = torch.tensor(gram_ov_np[None, :], dtype=torch.float32, device=device)
        rep_m = torch.zeros((1, 56), dtype=torch.float32, device=device)
        for a_prev in exec_actions:
            rep_m[0, a_prev] = 1.0
        c_t = torch.tensor(costs_np[None, :], dtype=torch.float32, device=device)

        if pol_name == "Control_Random_Network":
            return (
                rand_voi_net(q_t, k_meta_cand_t, cand_fp_t, gram_ov_t, rep_m, c_t, mode="full_unified")[0]
                .cpu()
                .numpy()
            )

        if pol_name == "Policy_Unified_KeyQuery_VoI_Attention":
            return (
                voi_net(q_t, k_meta_cand_t, cand_fp_t, gram_ov_t, rep_m, c_t, mode="full_unified")[0]
                .cpu()
                .numpy()
            )

        if pol_name == "Policy_NoRedundancy_Penalty":
            scores = (
                voi_net(q_init_t, k_meta_cand_t, cand_fp_t, gram_ov_t * 0.0, rep_m * 0.0, c_t, mode="no_redundancy")[0]
                .cpu()
                .numpy()
            )
            for a_init in exec_actions[:1]:
                scores[a_init] = -1e9
            return scores

        if pol_name == "Policy_KeyOnly_NoQuery_PUCT":
            return (
                voi_net(q_t, k_meta_cand_t, cand_fp_t, gram_ov_t, rep_m, c_t, mode="key_only_no_query")[0]
                .cpu()
                .numpy()
            )

        base_scores = (
            voi_net(q_t, k_meta_cand_t, cand_fp_t, gram_ov_t, rep_m, c_t, mode="full_unified")[0]
            .cpu()
            .numpy()
        )
        masked = np.full(56, -1e9, dtype=np.float32)
        if pol_name == "Policy_Always_Data_Only":
            masked[:36] = base_scores[:36]
        elif pol_name == "Policy_Always_Compute_Only":
            masked[36:48] = base_scores[36:48]
        elif pol_name == "Policy_Always_Memory_Only":
            masked[48:56] = base_scores[48:56]
        return masked

    regime_results: Dict[str, Any] = {}
    agg_by_policy: Dict[str, Dict[str, List[float]]] = {
        p: {
            "spearman_rho_all": [],
            "spearman_rho_post_redundancy": [],
            "top1_action_match": [],
            "top1_modality_match": [],
            "nats_per_unit_cost": [],
            "total_ig_nats": [],
            "redundant_action_rate": [],
            "final_nll_per_cell": [],
            "final_brier": [],
            "final_cell_acc": [],
            "compute_share": [],
            "data_share": [],
            "memory_share": [],
        }
        for p in policies
    }

    for reg_idx, regime_name in enumerate(regimes):
        costs_np = env.get_cost_regime(regime_name)
        reg_pol_metrics: Dict[str, Any] = {}

        ep_precomp = []
        for ep in dev_episodes:
            init_a = list(ep["init_cells"])
            init_o = [int(ep["realizations"][a]) for a in init_a]
            w_init, pz_init, pr_init = env.exact_posterior_from_history(init_a, init_o)
            q_feat_init = env.build_query_features(pz_init, pr_init, init_a, ep["u_weights"])
            ent_init = env.utility_weighted_entropy(w_init, ep["u_weights"])
            _, voi_exact_init = env.exact_bayesian_voi_vector(w_init, init_a, ep["u_weights"], costs_np)

            red_probe_a = init_a + [36 + (ep["ep_id"] % 6), 48 + (ep["ep_id"] % 4)]
            red_probe_o = [int(ep["realizations"][a]) for a in red_probe_a]
            w_red, pz_red, pr_red = env.exact_posterior_from_history(red_probe_a, red_probe_o)
            q_feat_red = env.build_query_features(pz_red, pr_red, red_probe_a, ep["u_weights"])
            _, voi_exact_red = env.exact_bayesian_voi_vector(w_red, red_probe_a, ep["u_weights"], costs_np)

            ep_precomp.append(
                {
                    "ep": ep,
                    "w_init": w_init,
                    "q_feat_init": q_feat_init,
                    "ent_init": ent_init,
                    "voi_exact_init": voi_exact_init,
                    "red_probe_a": red_probe_a,
                    "q_feat_red": q_feat_red,
                    "voi_exact_red": voi_exact_red,
                }
            )

        for pol_idx, pol_name in enumerate(policies):
            rng_pol = np.random.RandomState(seed * 100 + reg_idx * 20 + pol_idx)
            rho_all_list = []
            rho_post_red_list = []
            top1_act_list = []
            top1_mod_list = []
            nats_per_cost_list = []
            tot_ig_list = []
            red_rate_list = []
            nll_list = []
            brier_list = []
            acc_list = []
            mod_counts = np.zeros(3, dtype=np.float64)

            for item in ep_precomp:
                ep = item["ep"]
                exec_a = list(ep["init_cells"])
                exec_o = [int(ep["realizations"][a]) for a in exec_a]
                q_feat_init = item["q_feat_init"]
                ent_init = item["ent_init"]

                scores_red = score_candidates_for_policy(
                    pol_name,
                    item["q_feat_red"],
                    q_feat_init,
                    item["red_probe_a"],
                    costs_np,
                    item["voi_exact_red"],
                    rng_pol,
                )
                rho_post_red_list.append(spearman_rank_corr(scores_red, item["voi_exact_red"]))

                budget_spent = 0.0
                max_budget = 2.20
                redundant_steps = 0
                total_steps = 0

                while budget_spent < max_budget and total_steps < 5:
                    if total_steps == 0:
                        w_curr = item["w_init"]
                        q_feat_curr = q_feat_init
                        voi_exact_curr = item["voi_exact_init"]
                    else:
                        w_curr, pz_curr, pr_curr = env.exact_posterior_from_history(exec_a, exec_o)
                        if env.utility_weighted_entropy(w_curr, ep["u_weights"]) <= 1e-3:
                            break
                        q_feat_curr = env.build_query_features(pz_curr, pr_curr, exec_a, ep["u_weights"])
                        _, voi_exact_curr = env.exact_bayesian_voi_vector(w_curr, exec_a, ep["u_weights"], costs_np)

                    if float(np.max(voi_exact_curr)) <= 1e-6:
                        break

                    scores_curr = score_candidates_for_policy(
                        pol_name, q_feat_curr, q_feat_init, exec_a, costs_np, voi_exact_curr, rng_pol
                    )
                    rho_all_list.append(spearman_rank_corr(scores_curr, voi_exact_curr))
                    best_pred_a = int(np.argmax(scores_curr))
                    best_ora_a = int(np.argmax(voi_exact_curr))
                    top1_act_list.append(1.0 if best_pred_a == best_ora_a else 0.0)
                    top1_mod_list.append(
                        1.0 if int(env.action_modalities[best_pred_a]) == int(env.action_modalities[best_ora_a]) else 0.0
                    )

                    seen_groups = {int(env.action_strips_or_quads[a]) for a in exec_a}
                    if best_pred_a in exec_a or int(env.action_strips_or_quads[best_pred_a]) in seen_groups:
                        redundant_steps += 1

                    mod_counts[int(env.action_modalities[best_pred_a])] += 1.0
                    exec_a.append(best_pred_a)
                    exec_o.append(int(ep["realizations"][best_pred_a]))
                    budget_spent += float(costs_np[best_pred_a])
                    total_steps += 1

                w_final, _, _ = env.exact_posterior_from_history(exec_a, exec_o)
                ent_final = env.utility_weighted_entropy(w_final, ep["u_weights"])
                ig_total = max(0.0, ent_init - ent_final)
                nats_per_cost = ig_total / max(budget_spent, 1e-6)
                tot_ig_list.append(ig_total)
                nats_per_cost_list.append(nats_per_cost)
                red_rate_list.append(float(redundant_steps) / float(max(total_steps, 1)))

                ev_k_meta, ev_fp_t, ev_fp_s, ev_k_fp, ev_v = [], [], [], [], []
                for a_ev in exec_a:
                    ev = env.execute_operation_and_build_evidence(ep, a_ev, base_p0)
                    ev_k_meta.append(ev["k_meta"])
                    ev_fp_t.append(ev["fp_true"])
                    ev_fp_s.append(ev["fp_spill"])
                    ev_k_fp.append(ev["k_fp"])
                    ev_v.append(ev["v_j"])

                p_clear, _ = trust_net(
                    torch.tensor(base_logit0[None, :], dtype=torch.float32, device=device),
                    torch.tensor(q_feat_init[None, :, :], dtype=torch.float32, device=device),
                    torch.tensor(np.stack(ev_k_meta)[None, :, :], dtype=torch.float32, device=device),
                    torch.tensor(np.stack(ev_fp_t)[None, :, :], dtype=torch.float32, device=device),
                    torch.tensor(np.stack(ev_fp_s)[None, :, :], dtype=torch.float32, device=device),
                    torch.tensor(np.stack(ev_k_fp)[None, :, :], dtype=torch.float32, device=device),
                    torch.tensor(np.stack(ev_v)[None, :, :], dtype=torch.float32, device=device),
                    mode="query_key_gram",
                )
                p_np = np.clip(p_clear[0].cpu().numpy(), 1e-6, 1.0 - 1e-6)
                z_gt = ep["z_star"]
                nll = float(-(z_gt * np.log(p_np) + (1.0 - z_gt) * np.log(1.0 - p_np)).mean())
                brier = float(((p_np - z_gt) ** 2).mean())
                acc = float(((p_np >= 0.5).astype(np.float32) == z_gt).mean())
                nll_list.append(nll)
                brier_list.append(brier)
                acc_list.append(acc)

            mod_tot = max(float(mod_counts.sum()), 1.0)
            metrics = {
                "spearman_rho_all": float(np.mean(rho_all_list)),
                "spearman_rho_post_redundancy": float(np.mean(rho_post_red_list)),
                "top1_action_match": float(np.mean(top1_act_list)),
                "top1_modality_match": float(np.mean(top1_mod_list)),
                "nats_per_unit_cost": float(np.mean(nats_per_cost_list)),
                "total_ig_nats": float(np.mean(tot_ig_list)),
                "redundant_action_rate": float(np.mean(red_rate_list)),
                "final_nll_per_cell": float(np.mean(nll_list)),
                "final_brier": float(np.mean(brier_list)),
                "final_cell_acc": float(np.mean(acc_list)),
                "data_share": float(mod_counts[0] / mod_tot),
                "compute_share": float(mod_counts[1] / mod_tot),
                "memory_share": float(mod_counts[2] / mod_tot),
            }
            reg_pol_metrics[pol_name] = metrics
            for k_m, v_m in metrics.items():
                agg_by_policy[pol_name][k_m].append(v_m)

        regime_results[regime_name] = reg_pol_metrics

    mean_across_regimes = {
        pol_name: {k_m: float(np.mean(v_list)) for k_m, v_list in m_dict.items()}
        for pol_name, m_dict in agg_by_policy.items()
    }
    return {
        "by_cost_regime": regime_results,
        "mean_across_regimes": mean_across_regimes,
    }


@torch.no_grad()
def evaluate_part2_confiability_ablation(
    env: ShipHuntUnifiedCDMEnv,
    dev_episodes: List[Dict[str, Any]],
    trust_net: ConfiabilityAttentionClearingHead,
    rand_trust_net: ConfiabilityAttentionClearingHead,
    device: torch.device,
    seed: int,
) -> Dict[str, Any]:
    rng_eval = np.random.RandomState(seed + 5000)
    batch_std = build_confiability_training_step_batch(env, dev_episodes, rng_eval, device, stress_duplicates=False)
    batch_dup = build_confiability_training_step_batch(env, dev_episodes, rng_eval, device, stress_duplicates=True)

    base_p0 = (env.prior_w[:, None] * env.hypotheses_z).sum(axis=0).astype(np.float64)
    base_ent0 = float(env._binary_entropy_vec(base_p0).mean())

    arms = [
        ("Control_Random_Trust_Network", rand_trust_net, "query_key_gram"),
        ("Clear_Unweighted_Uniform", trust_net, "unweighted_uniform"),
        ("Clear_Scalar_Branch_Trust", trust_net, "scalar_branch_trust"),
        ("Clear_QueryKey_NoGram_Dedup", trust_net, "query_key_nogram"),
        ("Clear_QueryKey_Attention_Confiability", trust_net, "query_key_gram"),
        ("Ceiling_Exact_Bayesian_Posterior", None, "oracle"),
    ]

    results: Dict[str, Any] = {}
    for arm_name, net_obj, mode_str in arms:
        arm_metrics: Dict[str, float] = {}
        for pool_prefix, b_data in [("std", batch_std), ("stress_dup", batch_dup)]:
            if mode_str == "oracle":
                probs = b_data["exact_post"].cpu().numpy()
                beta_mean_primary = 1.0
                beta_mean_spill = 0.0
            else:
                probs_t, beta_t = net_obj(
                    b_data["prior_logits"],
                    b_data["q_feat"],
                    b_data["k_meta"],
                    b_data["fp_true"],
                    b_data["fp_spill"],
                    b_data["k_fp"],
                    b_data["v_updates"],
                    mode=mode_str,
                )
                probs = probs_t.cpu().numpy()
                fp_t_ij = b_data["fp_true"].permute(0, 2, 1).cpu().numpy()
                spill_ij = np.maximum(0.0, (b_data["fp_spill"] - b_data["fp_true"]).permute(0, 2, 1).cpu().numpy())
                b_np = beta_t.cpu().numpy()
                beta_mean_primary = float((b_np * fp_t_ij).sum() / max(float(fp_t_ij.sum()), 1.0))
                beta_mean_spill = float((b_np * spill_ij).sum() / max(float(spill_ij.sum()), 1.0))

            p_c = np.clip(probs, 1e-6, 1.0 - 1e-6)
            y_gt = b_data["z_target"].cpu().numpy()
            nll_cell = float(-(y_gt * np.log(p_c) + (1.0 - y_gt) * np.log(1.0 - p_c)).mean())
            ppl_cell = float(math.exp(min(nll_cell, 10.0)))
            brier = float(((p_c - y_gt) ** 2).mean())
            ece = compute_binary_ece(p_c, y_gt)
            acc = float(((p_c >= 0.5).astype(np.float32) == y_gt).mean())
            eff_ig_per_cell = max(0.0, base_ent0 - nll_cell)
            eff_ig_total = eff_ig_per_cell * 36.0
            mean_cost = float(b_data["total_cost"].mean())
            nats_per_cost = eff_ig_total / max(mean_cost, 1e-6)

            if pool_prefix == "std":
                arm_metrics.update(
                    {
                        "nll_per_cell": nll_cell,
                        "ppl_per_cell": ppl_cell,
                        "brier_score": brier,
                        "ece": ece,
                        "cell_acc": acc,
                        "nats_per_unit_cost": nats_per_cost,
                        "beta_primary_footprint": beta_mean_primary,
                        "beta_spillover_footprint": beta_mean_spill,
                    }
                )
            else:
                arm_metrics.update(
                    {
                        "stress_dup_nll_per_cell": nll_cell,
                        "stress_dup_ppl_per_cell": ppl_cell,
                        "stress_dup_brier": brier,
                        "stress_dup_ece": ece,
                        "stress_dup_cell_acc": acc,
                        "stress_dup_nats_per_unit_cost": nats_per_cost,
                    }
                )
        results[arm_name] = arm_metrics
    return results


def average_nested_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not dict_list:
        return {}
    first = dict_list[0]
    out: Dict[str, Any] = {}
    for k, v in first.items():
        if isinstance(v, dict):
            out[k] = average_nested_dicts([d[k] for d in dict_list])
        elif isinstance(v, (int, float, np.floating)):
            vals = [float(d[k]) for d in dict_list]
            out[k] = round(float(np.mean(vals)), 5)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Track E Exploratory Runner: Unified Compute, Data, and Memory via Key-Query VoI + Confiability Attention"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=str, default="101,102,103")
    parser.add_argument("--commit", type=str, default="")
    parser.add_argument("--smoke-cpu", action="store_true", help="Fast smoke test on CPU (produces no verdicts)")
    parser.add_argument(
        "--out",
        type=str,
        default="results/explore/expshiphuntkv/explore_shiphunt_unified_kv_confiability.json",
    )
    args = parser.parse_args()

    t_start = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    runner_path = Path(__file__).resolve()
    runner_sha256 = compute_sha256(runner_path)
    git_commit = args.commit if args.commit else get_git_commit(repo_root)

    device = torch.device("cpu" if args.smoke_cpu else args.device)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.smoke_cpu:
        seeds = seeds[:1]

    print(
        f"[EXPLORATORY] Running explore_shiphunt_unified_kv_confiability on device={device}, "
        f"seeds={seeds}, smoke_cpu={args.smoke_cpu}"
    )

    env = ShipHuntUnifiedCDMEnv()
    oracle_check = env.verify_oracle_brute_force(num_episodes=24)
    print(f"[Rule 9.6 Oracle Verification] verified={oracle_check['verified']}, max_abs_diff={oracle_check['max_abs_diff']:.3e}")

    per_seed_results: Dict[str, Any] = {}
    part1_list = []
    part2_list = []
    firewall_checks = []

    n_train = 40 if args.smoke_cpu else 180
    n_dev = 24 if args.smoke_cpu else 80

    for seed in seeds:
        t_seed = time.time()
        rng_train = np.random.RandomState(seed * 1000 + 11)
        rng_dev = np.random.RandomState(seed * 1000 + 777)

        train_episodes = [env.sample_episode(rng_train, i) for i in range(n_train)]
        dev_episodes = [env.sample_episode(rng_dev, 10000 + i) for i in range(n_dev)]

        torch.manual_seed(seed + 9999)
        rand_voi_net = UnifiedKeyQueryVoIAttentionNet().to(device)
        rand_trust_net = ConfiabilityAttentionClearingHead().to(device)
        for p in rand_voi_net.parameters():
            p.requires_grad_(False)
        for p in rand_trust_net.parameters():
            p.requires_grad_(False)

        voi_net, trust_net, fw_check = train_voi_and_confiability_heads(
            env, train_episodes, seed=seed, device=device, smoke_cpu=args.smoke_cpu
        )
        firewall_checks.append(fw_check)

        p1_res = evaluate_part1_voi_metareasoning(
            env, dev_episodes, voi_net, rand_voi_net, trust_net, device=device, seed=seed
        )
        p2_res = evaluate_part2_confiability_ablation(
            env, dev_episodes, trust_net, rand_trust_net, device=device, seed=seed
        )

        part1_list.append(p1_res)
        part2_list.append(p2_res)
        per_seed_results[f"seed_{seed}"] = {
            "seed": seed,
            "elapsed_sec": round(time.time() - t_seed, 2),
            "firewall_verification": fw_check,
            "part1_voi_metareasoning": p1_res,
            "part2_confiability_ablation": p2_res,
        }
        print(
            f"  [Seed {seed}] done in {time.time() - t_seed:.1f}s | "
            f"VoI Spearman rho={p1_res['mean_across_regimes']['Policy_Unified_KeyQuery_VoI_Attention']['spearman_rho_all']:.4f} "
            f"(Rand={p1_res['mean_across_regimes']['Control_Random_Network']['spearman_rho_all']:.4f}, "
            f"PostRed={p1_res['mean_across_regimes']['Policy_Unified_KeyQuery_VoI_Attention']['spearman_rho_post_redundancy']:.4f} vs "
            f"NoRed={p1_res['mean_across_regimes']['Policy_NoRedundancy_Penalty']['spearman_rho_post_redundancy']:.4f}) | "
            f"Clear NLL={p2_res['Clear_QueryKey_Attention_Confiability']['nll_per_cell']:.4f} "
            f"(Scalar={p2_res['Clear_Scalar_Branch_Trust']['nll_per_cell']:.4f}, "
            f"StressDup={p2_res['Clear_QueryKey_Attention_Confiability']['stress_dup_nll_per_cell']:.4f} vs "
            f"NoGram={p2_res['Clear_QueryKey_NoGram_Dedup']['stress_dup_nll_per_cell']:.4f})"
        )

    summary_p1 = average_nested_dicts(part1_list)
    summary_p2 = average_nested_dicts(part2_list)

    out_payload = {
        "track": "EXPLORATORY (PROTOCOL.md §10.1)",
        "experiment": "explore_shiphunt_unified_kv_confiability",
        "commit": git_commit,
        "runner_sha256": runner_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "seeds": seeds,
        "environment": {
            "name": "ShipHunt-6x6-Unified-CDM",
            "num_hypotheses_exact": env.num_hypotheses,
            "num_rules": env.num_rules,
            "actions_total": env.num_actions,
            "actions_data_look": env.num_data_actions,
            "actions_compute_think": env.num_compute_actions,
            "actions_memory_remember": env.num_memory_actions,
            "train_episodes_per_seed": n_train,
            "dev_episodes_per_seed": n_dev,
            "train_dev_overlap": 0,
            "test_splits_touched": False,
        },
        "oracle_brute_force_verification": oracle_check,
        "two_parameter_firewall_verification": {
            "params_search_and_theta_trust_separated": True,
            "max_theta_trust_drift_during_search_training": max(
                f["theta_trust_max_abs_drift_during_search_training"] for f in firewall_checks
            ),
            "max_params_search_drift_during_trust_training": max(
                f["params_search_max_abs_drift_during_trust_training"] for f in firewall_checks
            ),
        },
        "summary": {
            "status": "SMOKE_CPU_NO_VERDICT" if args.smoke_cpu else "EXPLORATORY_DEV",
            "part1_voi_metareasoning_mean": summary_p1,
            "part2_confiability_ablation_mean": summary_p2,
        },
        "per_seed": per_seed_results,
        "elapsed_total_sec": round(time.time() - t_start, 2),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"[EXPLORATORY] Saved output JSON to {out_path} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
