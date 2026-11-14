#!/usr/bin/env python3
"""
Round 27B Runner: Self-Supervised on Simulator Hidden State & Rule-Label-Free World Model,
Learned Neural Value-of-Information Probing & Zero-Drift Utility Swaps.
"""

import argparse
import hashlib
import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_CRIT_DF4 = 2.7764451051977987
T_CRIT_DF2 = 4.302652729911275


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class WatchedLabelArray:
    """Wraps ground-truth rule arrays and counts every __getitem__ access during SGD training."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr
        self.access_count = 0

    def __getitem__(self, item):
        self.access_count += 1
        return self._arr[item]

    def __len__(self) -> int:
        return len(self._arr)


def paired_ci95(vals_a: List[float], vals_b: List[float]) -> Dict[str, float]:
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    d = a - b
    mean_d = float(d.mean())
    if len(d) < 2:
        return {"mean_diff": mean_d, "se": 0.0, "ci95_low": mean_d, "ci95_high": mean_d}
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    t_val = T_CRIT_DF4 if len(d) >= 5 else T_CRIT_DF2
    return {
        "mean_diff": mean_d,
        "se": se,
        "ci95_low": float(mean_d - t_val * se),
        "ci95_high": float(mean_d + t_val * se),
    }


def binary_kl(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p = np.clip(p_true, 1e-6, 1.0 - 1e-6)
    q = np.clip(p_pred, 1e-6, 1.0 - 1e-6)
    kl = p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))
    return float(kl.mean())


def categorical_kl(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p = np.clip(p_true, 1e-7, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = np.clip(p_pred, 1e-7, 1.0)
    q = q / q.sum(axis=-1, keepdims=True)
    return float((p * np.log(p / q)).sum(axis=-1).mean())


def binary_jsd(p1: np.ndarray, p2: np.ndarray) -> float:
    p = np.clip(p1, 1e-7, 1.0 - 1e-7)
    q = np.clip(p2, 1e-7, 1.0 - 1e-7)
    m = 0.5 * (p + q)
    return 0.5 * binary_kl(p, m) + 0.5 * binary_kl(q, m)


class ExhaustiveUncertainRuleEnv:
    """
    Exhaustively enumerated 6x6 synthetic environment with 4 distinct latent rules R_rule in {0, 1, 2, 3}.
    Zero truncation, zero random subsampling in _enumerate_all_hypotheses.
    """

    def __init__(self, env_name: str):
        self.env_name = env_name
        self.num_rules = 4
        self.n_cells = 36
        self.neigh_masks = self._build_rule_neighborhoods()
        (
            self.hypotheses_z,
            self.hypotheses_clue,
            self.hypotheses_rule,
            self.prior_w,
        ) = self._enumerate_all_hypotheses()

    def _build_rule_neighborhoods(self) -> np.ndarray:
        masks = np.zeros((4, 36, 36), dtype=np.float32)
        for r in range(6):
            for c in range(6):
                i = r * 6 + c
                for r2 in range(6):
                    for c2 in range(6):
                        j = r2 * 6 + c2
                        if i == j:
                            continue
                        dr, dc = abs(r - r2), abs(c - c2)
                        if self.env_name == "shiphunt_6x6":
                            if dr == 0 and dc in (1, 2):
                                masks[0, i, j] = 1.0
                            if dc == 0 and dr in (1, 2):
                                masks[1, i, j] = 1.0
                            if max(dr, dc) == 1:
                                masks[2, i, j] = 1.0
                            if dr == dc and dr in (1, 2):
                                masks[3, i, j] = 1.0
                        else:
                            if dr + dc == 1:
                                masks[0, i, j] = 1.0
                            if max(dr, dc) == 1:
                                masks[1, i, j] = 1.0
                            if (dr == 1 and dc == 2) or (dr == 2 and dc == 1):
                                masks[2, i, j] = 1.0
                            if (dr == 1 and dc == 1) or (dr + dc == 2 and (dr == 0 or dc == 0)):
                                masks[3, i, j] = 1.0
        return masks

    def _enumerate_all_hypotheses(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_z, all_clues, all_r, all_w = [], [], [], []
        for rule_id in range(4):
            configs = []
            if self.env_name == "shiphunt_6x6":
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
            else:
                for c0 in range(6):
                    for c1 in range(6):
                        if abs(c1 - c0) < 2:
                            continue
                        for c2 in range(6):
                            if abs(c2 - c1) < 2 or c2 == c0:
                                continue
                            for c3 in range(6):
                                if abs(c3 - c2) < 2 or c3 in (c0, c1):
                                    continue
                                z = np.zeros((6, 6), dtype=np.float32)
                                z[0, c0] = 1.0
                                z[1, c1] = 1.0
                                z[2, c2] = 1.0
                                z[3, c3] = 1.0
                                rem = [c for c in range(6) if c not in (c0, c1, c2, c3)]
                                z[4, rem[0]] = 1.0
                                z[5, rem[1]] = 1.0
                                configs.append(z.reshape(36))

            z_arr = np.stack(configs, axis=0)
            clue_arr = z_arr @ self.neigh_masks[rule_id].T
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

    def exact_joint_bayes(
        self, rev_mask: np.ndarray, obs_z: np.ndarray, obs_clue: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        rev_idx = np.where(rev_mask > 0.5)[0]
        if len(rev_idx) == 0:
            w = self.prior_w.copy()
        else:
            z_match = np.all(np.abs(self.hypotheses_z[:, rev_idx] - obs_z[None, rev_idx]) < 1e-4, axis=1)
            c_match = np.all(np.abs(self.hypotheses_clue[:, rev_idx] - obs_clue[None, rev_idx]) < 1e-4, axis=1)
            w = self.prior_w * (z_match & c_match).astype(np.float64)
            if w.sum() <= 0.0:
                w = self.prior_w.copy()
        w = w / w.sum()
        p_z = (w[:, None] * self.hypotheses_z).sum(axis=0).astype(np.float32)
        p_rule = np.array([w[self.hypotheses_rule == r].sum() for r in range(4)], dtype=np.float32)
        return p_z, p_rule

    def exact_bayes_voi_probe(self, rev_mask: np.ndarray, obs_z: np.ndarray, obs_clue: np.ndarray) -> int:
        rev_idx = np.where(rev_mask > 0.5)[0]
        unrev = [i for i in range(36) if rev_mask[i] < 0.5]
        if not unrev:
            return 0
        if len(rev_idx) == 0:
            surv = np.arange(len(self.prior_w))
        else:
            z_match = np.all(np.abs(self.hypotheses_z[:, rev_idx] - obs_z[None, rev_idx]) < 1e-4, axis=1)
            c_match = np.all(np.abs(self.hypotheses_clue[:, rev_idx] - obs_clue[None, rev_idx]) < 1e-4, axis=1)
            surv = np.where(z_match & c_match)[0]
            if len(surv) == 0:
                surv = np.arange(len(self.prior_w))
        w_surv = self.prior_w[surv]
        w_surv = w_surv / w_surv.sum()
        r_surv = self.hypotheses_rule[surv]

        best_v = unrev[0]
        best_exp_ent = 1e9
        for v in unrev:
            zv = self.hypotheses_z[surv, v]
            cv = self.hypotheses_clue[surv, v]
            outcomes: Dict[Tuple[int, int], np.ndarray] = {}
            for idx_s in range(len(surv)):
                key = (int(round(float(zv[idx_s]))), int(round(float(cv[idx_s]))))
                if key not in outcomes:
                    outcomes[key] = np.zeros(4, dtype=np.float64)
                outcomes[key][r_surv[idx_s]] += w_surv[idx_s]
            exp_ent = 0.0
            for r_vec in outcomes.values():
                p_out = r_vec.sum()
                if p_out > 1e-12:
                    p_r_cond = r_vec / p_out
                    ent_cond = -sum(x * math.log(x) for x in p_r_cond if x > 1e-12)
                    exp_ent += p_out * ent_cond
            p_zv = float((w_surv * zv).sum())
            ent_zv = -(p_zv * math.log(max(p_zv, 1e-9)) + (1.0 - p_zv) * math.log(max(1.0 - p_zv, 1e-9)))
            score_ent = exp_ent - 0.05 * ent_zv
            if score_ent < best_exp_ent:
                best_exp_ent = score_ent
                best_v = v
        return int(best_v)

    def helper_stencil_control_beliefs(
        self, rev_mask: np.ndarray, obs_z: np.ndarray, obs_clue: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        rev_idx = np.where(rev_mask > 0.5)[0]
        p0 = 6.0 / 36.0
        log_l = np.zeros(4, dtype=np.float64)
        p_z_r = np.zeros((36, 4), dtype=np.float32)
        for r in range(4):
            m_r = self.neigh_masks[r]
            known_1 = m_r @ (obs_z * rev_mask)
            unrev_d = np.maximum(m_r @ (1.0 - rev_mask), 1.0)
            rem = (obs_clue - known_1) * rev_mask
            dens_u = np.clip(rem, 0.0, None) / unrev_d * rev_mask
            rev_cnt = np.maximum(m_r.T @ rev_mask, 1.0)
            avg_dens_v = (m_r.T @ dens_u) / rev_cnt
            p_v = 1.0 / (1.0 + np.exp(-2.5 * (avg_dens_v - p0)))
            p_z_r[:, r] = rev_mask * obs_z + (1.0 - rev_mask) * p_v
            for u in rev_idx:
                rem_i = int(round(float(obs_clue[u] - known_1[u])))
                ud_i = int(round(float((m_r @ (1.0 - rev_mask))[u])))
                if rem_i < 0 or rem_i > ud_i:
                    log_l[r] -= 10.0
                else:
                    c_p = math.comb(ud_i, rem_i) * (p0 ** rem_i) * ((1.0 - p0) ** (ud_i - rem_i))
                    log_l[r] += math.log(max(c_p, 1e-9))
        log_l = log_l - log_l.max()
        mu_rule = (np.exp(log_l) / np.exp(log_l).sum()).astype(np.float32)
        mu_z = (p_z_r * mu_rule[None, :]).sum(axis=1).astype(np.float32)
        mu_z = rev_mask * obs_z + (1.0 - rev_mask) * mu_z
        var_v = ((p_z_r - mu_z[:, None]) ** 2 * mu_rule[None, :]).sum(axis=1)
        score = var_v - 1e6 * rev_mask
        return mu_z, mu_rule, int(np.argmax(score))


def verify_oracle_exactness(envs: Dict[str, ExhaustiveUncertainRuleEnv]) -> Dict[str, Any]:
    max_tv_z = 0.0
    max_tv_rule = 0.0
    checked = 0
    for env in envs.values():
        for idx in range(0, min(len(env.prior_w), 300), 10):
            if checked >= 60:
                break
            tz = env.hypotheses_z[idx]
            tc = env.hypotheses_clue[idx]
            rev = np.zeros(36, dtype=np.float32)
            rev[[0, 7, 14, 21, 28]] = 1.0
            pz_fast, pr_fast = env.exact_joint_bayes(rev, tz * rev, tc * rev)
            num_z = np.zeros(36, dtype=np.float64)
            num_r = np.zeros(4, dtype=np.float64)
            den = 0.0
            for h_i in range(len(env.prior_w)):
                ok = True
                for u in (0, 7, 14, 21, 28):
                    if abs(float(env.hypotheses_z[h_i, u]) - float(tz[u])) > 1e-4:
                        ok = False
                        break
                    if abs(float(env.hypotheses_clue[h_i, u]) - float(tc[u])) > 1e-4:
                        ok = False
                        break
                if ok:
                    wh = float(env.prior_w[h_i])
                    den += wh
                    num_z += wh * env.hypotheses_z[h_i]
                    num_r[int(env.hypotheses_rule[h_i])] += wh
            pz_bf = (num_z / den).astype(np.float32)
            pr_bf = (num_r / den).astype(np.float32)
            tv_z = float(np.max(np.abs(pz_fast - pz_bf)))
            tv_r = float(0.5 * np.sum(np.abs(pr_fast - pr_bf)))
            max_tv_z = max(max_tv_z, tv_z)
            max_tv_rule = max(max_tv_rule, tv_r)
            checked += 1
    return {"episodes_checked": checked, "max_tv_z": max_tv_z, "max_tv_rule": max_tv_rule}


class PureNeuralRuleWorldNet(nn.Module):
    """
    100% Pure-Neural 2D Relative-Offset Kernel & Bethe/IPF Log-Odds Propagation World Model,
    with Learned Neural VoI Head and Utility-Conditioned Action Policy Head.
    Zero environment stencils (neigh_masks), zero base-rate constants (6/36 or -1.6),
    zero hand-coded test-time VoI formulas, and zero hardcoded utility shifts.
    """

    def __init__(self, hidden_size: int = 48, mode: str = "lir_separated"):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode  # "lir_separated", "no_rule_slots", "fused_utility", "token_masked_utility"
        in_dim = 4 if mode == "fused_utility" else 3
        self.cell_in = nn.Linear(in_dim, hidden_size)
        self.pos_emb = nn.Embedding(36, hidden_size)
        self.u_tok_proj = nn.Linear(1, hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
        self.trunk_mlp = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.z_res_head = nn.Linear(hidden_size, 4)

        # 4 latent rule slots: 2D relative-offset sensor kernels (4, 11, 11),
        # 2D relative-offset slot occupancy geometry (4, 11, 11),
        # per-slot cell priors (4, 36), conditional clue distribution P(c_u | z_u, r) (4, 2, 9),
        # learned Bethe/IPF log-odds step gain, and learned rule consistency weights.
        self.rel_kernel_logits = nn.Parameter(torch.empty(4, 11, 11))
        self.rel_pair_pos = nn.Parameter(torch.zeros(4, 11, 11))
        self.rel_pair_neg = nn.Parameter(torch.zeros(4, 11, 11))
        self.slot_cell_logits = nn.Parameter(torch.zeros(4, 36))
        self.clue_cond_logits = nn.Parameter(torch.zeros(4, 2, 9))
        self.ipf_step_gain = nn.Parameter(torch.zeros(4))
        self.rule_weights = nn.Parameter(torch.zeros(4))

        # Learned Neural VoI Head & Learned Utility-Conditioned Action Policy Head
        self.voi_head = nn.Sequential(
            nn.Linear(20, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.pol_head = nn.Sequential(
            nn.Linear(15, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

        dr_idx = torch.zeros(36, 36, dtype=torch.long)
        dc_idx = torch.zeros(36, 36, dtype=torch.long)
        off_diag = torch.ones(36, 36, dtype=torch.float32) - torch.eye(36, dtype=torch.float32)
        for i in range(36):
            r1, c1 = divmod(i, 6)
            for j in range(36):
                r2, c2 = divmod(j, 6)
                dr_idx[i, j] = (r2 - r1) + 5
                dc_idx[i, j] = (c2 - c1) + 5
        self.register_buffer("dr_idx", dr_idx)
        self.register_buffer("dc_idx", dc_idx)
        self.register_buffer("off_diag", off_diag)
        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.rel_kernel_logits, mean=0.0, std=0.15)
        nn.init.zeros_(self.rel_pair_pos)
        nn.init.zeros_(self.rel_pair_neg)
        nn.init.zeros_(self.slot_cell_logits)
        nn.init.zeros_(self.clue_cond_logits)
        nn.init.zeros_(self.ipf_step_gain)
        nn.init.zeros_(self.rule_weights)

    def get_learned_rule_kernels(self) -> torch.Tensor:
        rel_sym = 0.5 * (self.rel_kernel_logits + torch.flip(self.rel_kernel_logits, dims=[1, 2]))
        rel_sym = 0.5 * (rel_sym + torch.flip(rel_sym, dims=[1]))
        k_36 = torch.sigmoid(rel_sym - 1.5)[:, self.dr_idx, self.dc_idx] * self.off_diag.unsqueeze(0)
        if self.mode == "no_rule_slots":
            return k_36[:1].expand(4, -1, -1)
        return k_36

    @torch.no_grad()
    def permute_slots(self, perm: Tuple[int, ...]) -> None:
        idx = list(perm)
        self.rel_kernel_logits.copy_(self.rel_kernel_logits[idx])
        self.rel_pair_pos.copy_(self.rel_pair_pos[idx])
        self.rel_pair_neg.copy_(self.rel_pair_neg[idx])
        self.slot_cell_logits.copy_(self.slot_cell_logits[idx])
        self.clue_cond_logits.copy_(self.clue_cond_logits[idx])
        self.ipf_step_gain.copy_(self.ipf_step_gain[idx])
        self.z_res_head.weight.copy_(self.z_res_head.weight[idx])
        if self.z_res_head.bias is not None:
            self.z_res_head.bias.copy_(self.z_res_head.bias[idx])

    def forward(
        self,
        rev_mask: torch.Tensor,
        obs_z: torch.Tensor,
        obs_clue: torch.Tensor,
        u_sign_tensor: torch.Tensor,
        ipf_iters: int = 3,
    ) -> Dict[str, torch.Tensor]:
        bsz = rev_mask.shape[0]
        device = rev_mask.device
        obs_z_m = obs_z * rev_mask
        obs_c_m = obs_clue * rev_mask
        unrev_mask = 1.0 - rev_mask
        raw_3 = torch.stack([rev_mask, obs_z_m, obs_c_m], dim=-1)  # (B, 36, 3)

        if self.mode == "fused_utility":
            u_ch = u_sign_tensor.view(bsz, 1, 1).expand(bsz, 36, 1)
            x_in = torch.cat([raw_3, u_ch], dim=-1)
        else:
            x_in = raw_3

        pos_ids = torch.arange(36, device=device).unsqueeze(0).expand(bsz, 36)
        h = self.cell_in(x_in) + self.pos_emb(pos_ids)
        if self.mode == "token_masked_utility":
            u_tok = self.u_tok_proj(u_sign_tensor.view(bsz, 1, 1)).expand(bsz, 4, self.hidden_size)
            seq = torch.cat([h, u_tok], dim=1)
            attn_out, _ = self.attn(seq, seq, seq)
            h = h + attn_out[:, :36, :]
        else:
            attn_out, _ = self.attn(h, h, h)
            h = h + attn_out
        h = h + self.trunk_mlp(h)

        k_r = self.get_learned_rule_kernels()  # (4, 36, 36)
        w_p1 = 0.5 * (self.rel_pair_pos + torch.flip(self.rel_pair_pos, dims=[1, 2]))[:, self.dr_idx, self.dc_idx] * self.off_diag.unsqueeze(0)
        w_p0 = 0.5 * (self.rel_pair_neg + torch.flip(self.rel_pair_neg, dims=[1, 2]))[:, self.dr_idx, self.dc_idx] * self.off_diag.unsqueeze(0)
        slot_cell = self.slot_cell_logits
        clue_cond = self.clue_cond_logits
        step_gain = F.softplus(self.ipf_step_gain)  # (4,)
        if self.mode == "no_rule_slots":
            w_p1 = torch.zeros_like(w_p1)
            w_p0 = torch.zeros_like(w_p0)
            slot_cell = slot_cell[:1].expand(4, -1)
            clue_cond = clue_cond[:1].expand(4, -1, -1)
            step_gain = step_gain[:1].expand(4)

        ctx_pair = torch.einsum("ruv,bv->bur", w_p1, obs_z_m) + torch.einsum("ruv,bv->bur", w_p0, (1.0 - obs_z_m) * rev_mask)
        known_1 = torch.einsum("ruv,bv->bur", k_r, obs_z_m)
        raw_unrev_d = torch.einsum("ruv,bv->bur", k_r, unrev_mask)
        rev_deg_v = torch.einsum("ruv,bu->bvr", k_r, rev_mask)
        rem_c_raw = (obs_c_m.unsqueeze(-1) - known_1) * rev_mask.unsqueeze(-1)
        rem_c = rem_c_raw.clamp(min=0.0)
        empty_c = (raw_unrev_d - rem_c).clamp(min=0.0) * rev_mask.unsqueeze(-1)

        z_res = self.z_res_head(h)
        if self.mode == "no_rule_slots":
            z_res = z_res[:, :, :1].expand(-1, -1, 4)

        # Identical residual scale across all modes (no ablation-specific multiplier)
        z_logits_r = slot_cell.T.unsqueeze(0).expand(bsz, -1, -1) + ctx_pair + 0.60 * z_res

        # 3-step Bethe / Iterative Proportional Fitting log-odds propagation over learned kernel k_r
        for _ in range(ipf_iters):
            p_un = torch.sigmoid(z_logits_r) * unrev_mask.unsqueeze(-1)
            mu_rem_step = torch.einsum("ruv,bvr->bur", k_r, p_un)
            mu_emp_step = (raw_unrev_d - mu_rem_step).clamp(min=0.0)
            ll_step = (
                torch.log((rem_c + 0.05) / (mu_rem_step + 0.05))
                - torch.log((empty_c + 0.05) / (mu_emp_step + 0.05))
            ) * rev_mask.unsqueeze(-1)
            back_ll = torch.einsum("ruv,bur->bvr", k_r, ll_step)
            z_logits_r = z_logits_r + step_gain.view(1, 1, 4) * back_ll

        p_z_r = torch.sigmoid(z_logits_r)
        p_z_eff = rev_mask.unsqueeze(-1) * obs_z_m.unsqueeze(-1) + unrev_mask.unsqueeze(-1) * p_z_r
        pred_c_r = torch.einsum("ruv,bvr->bur", k_r, p_z_eff)

        p_unrev = p_z_r * unrev_mask.unsqueeze(-1)
        mu_rem = torch.einsum("ruv,bvr->bur", k_r, p_unrev)
        var_rem = torch.einsum("ruv,bvr->bur", k_r, p_unrev * (1.0 - p_z_r)).clamp(min=0.08)
        nll_rem = (0.5 * (rem_c_raw - mu_rem).square() / var_rem) * rev_mask.unsqueeze(-1)
        viol_rem = (F.relu(-rem_c_raw - 0.15) + F.relu(rem_c_raw - raw_unrev_d - 0.15)) * rev_mask.unsqueeze(-1)
        bce_cell = (
            F.binary_cross_entropy_with_logits(
                slot_cell.T.unsqueeze(0).expand(bsz, -1, -1),
                obs_z_m.unsqueeze(-1).expand(-1, -1, 4),
                reduction="none",
            )
            * rev_mask.unsqueeze(-1)
        )
        log_pc = F.log_softmax(clue_cond, dim=-1)  # (4, 2, 9)
        z_obs_int = torch.clamp(torch.round(obs_z_m).long(), 0, 1)
        c_obs_int = torch.clamp(torch.round(obs_c_m).long(), 0, 8)
        ll_pc = log_pc[:, z_obs_int, c_obs_int].permute(1, 2, 0) * rev_mask.unsqueeze(-1)

        if self.mode == "no_rule_slots":
            rule_logits = torch.zeros(bsz, 4, device=device)
        else:
            rw = F.softplus(self.rule_weights)
            rule_logits = (
                -rw[0] * bce_cell.sum(dim=1)
                + rw[1] * ll_pc.sum(dim=1)
                - rw[2] * nll_rem.sum(dim=1)
                - rw[3] * 10.0 * viol_rem.sum(dim=1)
            )

        mu_rule = F.softmax(rule_logits, dim=-1)
        mu_z_raw = (p_z_r * mu_rule.unsqueeze(1)).sum(dim=-1)
        mu_z = rev_mask * obs_z_m + unrev_mask * mu_z_raw

        # LIR Firewall: in lir_separated (chi_world=(1,0), chi_policy=(0,1)), detach all world states/beliefs
        if self.mode == "lir_separated":
            z_res_pol = z_res.detach()
            p_z_r_pol = p_z_eff.detach()
            pred_c_r_pol = pred_c_r.detach()
            rev_deg_pol = rev_deg_v.detach()
            mu_rule_pol = mu_rule.detach()
            mu_z_pol = mu_z.detach()
            k_r_pol = k_r.detach()
        else:
            z_res_pol = z_res
            p_z_r_pol = p_z_eff
            pred_c_r_pol = pred_c_r
            rev_deg_pol = rev_deg_v
            mu_rule_pol = mu_rule
            mu_z_pol = mu_z
            k_r_pol = k_r

        # Stencil & belief disagreement variance features under current rule posterior (Law of Total Variance)
        mean_pz_pol = (mu_rule_pol.unsqueeze(1) * p_z_r_pol).sum(dim=-1, keepdim=True)
        mean_c_pol = (mu_rule_pol.unsqueeze(1) * pred_c_r_pol).sum(dim=-1, keepdim=True)
        var_z_slot = (mu_rule_pol.unsqueeze(1) * (p_z_r_pol - mean_pz_pol).square()).sum(dim=-1)
        var_c_slot = (mu_rule_pol.unsqueeze(1) * (pred_c_r_pol - mean_c_pol).square()).sum(dim=-1)
        k_bar = torch.einsum("br,ruv->buv", mu_rule_pol, k_r_pol)  # (B, 36, 36)
        var_k = torch.einsum("br,bruv->buv", mu_rule_pol, (k_r_pol.unsqueeze(0) - k_bar.unsqueeze(1)).square())
        u_var = (mu_z_pol * (1.0 - mu_z_pol) * unrev_mask).unsqueeze(-1)
        stencil_var = torch.bmm(var_k, u_var).squeeze(-1)  # (B, 36)
        rev_stencil_var = torch.bmm(var_k, ((obs_c_m + 3.0 * obs_z_m + 1.0) * rev_mask).unsqueeze(-1)).squeeze(-1)  # (B, 36)

        # Learned Neural VoI Head (translation-equivariant epistemic disagreement & belief features; no pos_emb bias)
        rule_exp = mu_rule_pol.unsqueeze(1).expand(bsz, 36, 4)
        voi_in = torch.cat(
            [
                p_z_r_pol,
                pred_c_r_pol,
                rev_deg_pol,
                rule_exp,
                var_c_slot.unsqueeze(-1),
                var_z_slot.unsqueeze(-1),
                stencil_var.unsqueeze(-1),
                rev_stencil_var.unsqueeze(-1),
            ],
            dim=-1,
        )  # (B, 36, 20)
        voi_logits = self.voi_head(voi_in).squeeze(-1)  # (B, 36)
        probe_score = voi_logits - 1e6 * rev_mask

        # Learned Utility-Conditioned Action Policy Head (translation-equivariant belief & residual features)
        u_exp = u_sign_tensor.view(bsz, 1, 1).expand(bsz, 36, 1)
        pol_in = torch.cat(
            [
                z_res_pol,
                p_z_r_pol,
                mu_z_pol.unsqueeze(-1),
                (p_z_r_pol - 0.5) * u_exp,
                (mu_z_pol.unsqueeze(-1) - 0.5) * u_exp,
                u_exp,
            ],
            dim=-1,
        )  # (B, 36, 15)
        act_logits = self.pol_head(pol_in).squeeze(-1)  # (B, 36)

        return {
            "mu_z": mu_z,
            "mu_rule": mu_rule,
            "rule_logits": rule_logits,
            "z_logits_r": z_logits_r,
            "z_res": z_res,
            "p_z_r": p_z_r,
            "pred_c_r": pred_c_r,
            "stencil_var": stencil_var,
            "rev_stencil_var": rev_stencil_var,
            "ll_pc": ll_pc,
            "bce_cell": bce_cell,
            "voi_logits": voi_logits,
            "probe_score": probe_score,
            "act_logits": act_logits,
            "raw_input_channels": int(raw_3.shape[-1]),
        }


def sinkhorn_assign(err_bk: torch.Tensor, temp: float = 0.08, n_iters: int = 8) -> torch.Tensor:
    logits = -err_bk.detach() / temp
    logits = logits - logits.max(dim=-1, keepdim=True).values
    q = torch.exp(logits) + 1e-6
    for _ in range(n_iters):
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-8) * (err_bk.shape[0] / 4.0)
    return q / q.sum(dim=1, keepdim=True).clamp(min=1e-8)


def train_model_rule_label_free(
    net: PureNeuralRuleWorldNet,
    z_tr_np: np.ndarray,
    c_tr_np: np.ndarray,
    z_cal_np: np.ndarray,
    c_cal_np: np.ndarray,
    watched_rules: WatchedLabelArray,
    n_steps: int,
    device: torch.device,
) -> None:
    """
    Trains PureNeuralRuleWorldNet self-supervised on simulator hidden state & clue trajectories (z_tr_np, c_tr_np)
    and rule-label-free during SGD, plus 4 post-EM slot-indexing calibration boards (z_cal_np, c_cal_np)
    to break S_4 slot permutation symmetry. Asserts watched_rules.access_count == 0.
    """
    if watched_rules.access_count != 0:
        raise RuntimeError("Rule labels were accessed prior to training!")
    z_tr = torch.tensor(z_tr_np, device=device)
    c_tr = torch.tensor(c_tr_np, device=device)
    n_tr = len(z_tr_np)

    # Phase 1: Rule-label-free Sinkhorn EM discovery of the 4 2D relative-offset kernels on partial 28-cell trajectories
    opt_k = torch.optim.Adam([net.rel_kernel_logits], lr=0.15)
    k_steps = max(180, n_steps // 2)
    for _ in range(k_steps):
        ep_mask = (torch.rand(n_tr, 36, device=device) < 0.80).float()
        k_36 = net.get_learned_rule_kernels()
        z_est = ep_mask * z_tr + (1.0 - ep_mask) * (z_tr * ep_mask).mean(dim=1, keepdim=True)
        pred_c = torch.einsum("ruv,bv->bur", k_36, z_est)
        err = ((pred_c - c_tr.unsqueeze(-1)).square() * ep_mask.unsqueeze(-1)).mean(dim=1)
        q = (
            torch.full((n_tr, 4), 0.25, device=device)
            if net.mode == "no_rule_slots"
            else sinkhorn_assign(err, temp=0.08, n_iters=8)
        )
        k_flat = F.normalize(k_36.reshape(4, -1), dim=-1)
        gram = torch.mm(k_flat, k_flat.T)
        rep = 0.0 if net.mode == "no_rule_slots" else ((gram - torch.eye(4, device=device)).square()).mean()
        loss_k = (q * err).sum(dim=-1).mean() + 0.015 * k_36.mean() + 0.08 * rep
        opt_k.zero_grad()
        loss_k.backward()
        opt_k.step()

    # Post-EM canonical slot permutation alignment on the 4 training calibration episodes (0 gradient steps)
    if net.mode != "no_rule_slots":
        with torch.no_grad():
            k_learned = net.get_learned_rule_kernels()
            z_cal = torch.tensor(z_cal_np, device=device)
            c_cal = torch.tensor(c_cal_np, device=device)
            pred_cal = torch.einsum("ruv,bv->bur", k_learned, z_cal)
            err_cal = (pred_cal - c_cal.unsqueeze(-1)).square().mean(dim=1).cpu().numpy()
            best_perm = min(itertools.permutations(range(4)), key=lambda p: sum(err_cal[r, p[r]] for r in range(4)))
            net.permute_slots(best_perm)

    # Compute detached rule-label-free slot responsibilities on 28-cell partial training episodes
    with torch.no_grad():
        ep_mask_28 = (torch.rand(n_tr, 36, device=device) < 0.80).float()
        k_aligned = net.get_learned_rule_kernels()
        z_est_28 = ep_mask_28 * z_tr + (1.0 - ep_mask_28) * (z_tr * ep_mask_28).mean(dim=1, keepdim=True)
        pred_tr = torch.einsum("ruv,bv->bur", k_aligned, z_est_28)
        err_tr = ((pred_tr - c_tr.unsqueeze(-1)).square() * ep_mask_28.unsqueeze(-1)).mean(dim=1)
        w_tr_all = (
            torch.full((n_tr, 4), 0.25, device=device)
            if net.mode == "no_rule_slots"
            else F.softmax(-err_tr / 0.04, dim=-1)
        )

    # Phase 2: Joint training of world-model beliefs, rule calibration, neural VoI head, and action policy head
    net.rel_kernel_logits.requires_grad_(False)
    prior_params = (
        [
            net.slot_cell_logits,
            net.rel_pair_pos,
            net.rel_pair_neg,
            net.clue_cond_logits,
            net.ipf_step_gain,
            net.rule_weights,
        ]
        + list(net.voi_head.parameters())
        + list(net.pol_head.parameters())
    )
    trunk_params = [
        p
        for n, p in net.named_parameters()
        if p.requires_grad
        and not any(
            n.startswith(prefix)
            for prefix in (
                "slot_cell_logits",
                "rel_pair_pos",
                "rel_pair_neg",
                "clue_cond_logits",
                "ipf_step_gain",
                "rule_weights",
                "voi_head",
                "pol_head",
            )
        )
    ]
    opt = torch.optim.AdamW(
        [
            {"params": prior_params, "lr": 0.04, "weight_decay": 0.0},
            {"params": trunk_params, "lr": 6e-3, "weight_decay": 1e-4},
        ]
    )

    bsz = 64
    for _ in range(n_steps):
        b_loc = np.random.choice(n_tr, size=bsz, replace=True)
        z_b = z_tr[b_loc]
        c_b = c_tr[b_loc]
        w_b = w_tr_all[b_loc]
        u_signs = torch.where(torch.rand(bsz, device=device) < 0.5, 1.0, -1.0)

        rev_m = (torch.rand(bsz, 36, device=device) < (5.0 / 36.0)).float()
        held_m = (1.0 - rev_m) * (torch.rand(bsz, 36, device=device) < 0.50).float()

        with torch.no_grad():
            u_logit_sel = 1.5 * u_signs.unsqueeze(1) * (2.0 * z_b - 1.0)
            u_prob_sel = torch.sigmoid(u_logit_sel) * (1.0 - rev_m)
            u_sel_m = (torch.rand(bsz, 36, device=device) < u_prob_sel).float()

        out = net(rev_m, z_b * rev_m, c_b * rev_m, u_signs)

        # 1) Unbiased world-model state prediction on uniform held-out cells held_m
        bce_z = F.binary_cross_entropy_with_logits(
            out["z_logits_r"], z_b.unsqueeze(-1).expand(-1, -1, 4), reduction="none"
        )
        loss_z = ((bce_z * held_m.unsqueeze(-1)).sum(dim=1) / held_m.sum(dim=1, keepdim=True).clamp(min=1.0) * w_b).sum(dim=-1).mean()

        # 2) On-policy trajectory residual loss on u_sel_m (drives trunk z_res)
        bce_res = F.binary_cross_entropy_with_logits(
            out["z_res"], z_b.unsqueeze(-1).expand(-1, -1, 4), reduction="none"
        )
        loss_u_traj = ((bce_res * u_sel_m.unsqueeze(-1)).sum(dim=1) / u_sel_m.sum(dim=1, keepdim=True).clamp(min=1.0) * w_b).sum(dim=-1).mean()

        # 3) Revealed-cell prior & conditional clue likelihood calibration
        loss_rev_cell = (out["bce_cell"].sum(dim=1) * w_b).sum(dim=-1).mean()
        loss_pc = -(out["ll_pc"].sum(dim=1) * w_b).sum(dim=-1).mean()

        # 4) Hindsight prefix-to-episode rule calibration loss
        loss_r = (
            torch.tensor(0.0, device=device)
            if net.mode == "no_rule_slots"
            else -(w_b * F.log_softmax(out["rule_logits"], dim=-1)).sum(dim=-1).mean()
        )

        # 5) Self-Supervised Law-of-Total-Variance Neural VoI Head Training
        with torch.no_grad():
            p_z_d = out["p_z_r"].detach()
            c_d = out["pred_c_r"].detach()
            r_d = out["mu_rule"].detach().unsqueeze(1)
            st_var_d = out["stencil_var"].detach()
            rev_st_var_d = out["rev_stencil_var"].detach()
            mean_pz = (r_d * p_z_d).sum(dim=-1, keepdim=True)
            mean_c = (r_d * c_d).sum(dim=-1, keepdim=True)
            var_z_slot = (r_d * (p_z_d - mean_pz).square()).sum(dim=-1)
            var_c_slot = (r_d * (c_d - mean_c).square()).sum(dim=-1)
            voi_target = var_c_slot + 2.0 * var_z_slot + 2.5 * st_var_d + 2.5 * rev_st_var_d
        loss_voi = ((out["voi_logits"] - voi_target).square() * (1.0 - rev_m)).sum() / (1.0 - rev_m).sum().clamp(min=1.0)

        # 6) Utility-Conditioned Action Policy Head Training on held_m
        u_target = torch.where(u_signs.unsqueeze(1) > 0, z_b, 1.0 - z_b)
        loss_pol = (
            F.binary_cross_entropy_with_logits(out["act_logits"], u_target, reduction="none") * held_m
        ).sum() / held_m.sum().clamp(min=1.0)

        loss = loss_z + 0.4 * loss_u_traj + 0.2 * loss_rev_cell + 0.2 * loss_pc + 0.5 * loss_r + loss_voi + loss_pol
        opt.zero_grad()
        loss.backward()
        opt.step()

    if watched_rules.access_count != 0:
        raise RuntimeError(f"Rule labels were accessed during training: {watched_rules.access_count}")


def run_r27b(args) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke_cpu else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(1)
    seeds = [101, 102, 103, 104, 105]
    n_test_full = args.n_test
    n_test_smoke = 40
    n_steps = 260 if args.smoke_cpu else args.steps
    k_probes = 4

    envs = {
        "shiphunt_6x6": ExhaustiveUncertainRuleEnv("shiphunt_6x6"),
        "minesweeper_6x6": ExhaustiveUncertainRuleEnv("minesweeper_6x6"),
    }
    oracle_check = verify_oracle_exactness(envs)
    if oracle_check["max_tv_z"] > 1e-6 or oracle_check["max_tv_rule"] > 1e-6:
        raise RuntimeError(f"Exact Bayesian oracle check failed: {oracle_check}")

    env_splits: Dict[str, Dict[str, Any]] = {}
    disjointness_checks: Dict[str, int] = {}
    split_rng = np.random.default_rng(27200)
    for env_name, env in envs.items():
        per_rule_full = max(1, n_test_full // 4)
        per_rule_smoke = max(1, n_test_smoke // 4)
        if env_name == "minesweeper_6x6":
            # Partition the 90 underlying 6x6 mine boards first so zero z-boards are shared across train/smoke/test
            b_perm = split_rng.permutation(90)
            tr_b = b_perm[: 90 - per_rule_smoke - per_rule_full]
            sm_b = b_perm[90 - per_rule_smoke - per_rule_full : 90 - per_rule_full]
            te_b = b_perm[90 - per_rule_full :]
            tr_list, sm_list, te_list, cal_list = [], [], [], []
            for r_id in range(4):
                tr_part = np.array([r_id * 90 + int(b) for b in tr_b], dtype=np.int64)
                sm_part = np.array([r_id * 90 + int(b) for b in sm_b], dtype=np.int64)
                te_part = np.array([r_id * 90 + int(b) for b in te_b], dtype=np.int64)
                tr_list.append(tr_part)
                sm_list.append(sm_part)
                te_list.append(te_part)
                cal_list.append(int(tr_part[0]))
            tr_idx = np.concatenate(tr_list)
            sm_idx = np.concatenate(sm_list)
            te_idx = np.concatenate(te_list)
        else:
            # Exclude any cross-rule colliding z-boards from smoke_idx and test_idx so board_z_*_overlap == 0
            z_tups = [tuple(row.tolist()) for row in env.hypotheses_z]
            z_counts: Dict[Tuple[float, ...], int] = {}
            for t in z_tups:
                z_counts[t] = z_counts.get(t, 0) + 1
            tr_list, sm_list, te_list, cal_list = [], [], [], []
            for r_id in range(4):
                r_indices = np.where(env.hypotheses_rule == r_id)[0].copy()
                split_rng.shuffle(r_indices)
                uniq_r = np.array([i for i in r_indices if z_counts[z_tups[int(i)]] == 1], dtype=np.int64)
                coll_r = np.array([i for i in r_indices if z_counts[z_tups[int(i)]] > 1], dtype=np.int64)
                n_tr = int(0.55 * len(r_indices))
                n_tr_u = max(1, n_tr - len(coll_r))
                tr_part = np.concatenate([coll_r, uniq_r[:n_tr_u]])
                sm_part = uniq_r[n_tr_u : n_tr_u + per_rule_smoke]
                te_part = uniq_r[n_tr_u + per_rule_smoke : n_tr_u + per_rule_smoke + per_rule_full]
                tr_list.append(tr_part)
                sm_list.append(sm_part)
                te_list.append(te_part)
                cal_list.append(int(tr_part[0]))
            tr_idx = np.concatenate(tr_list)
            sm_idx = np.concatenate(sm_list)
            te_idx = np.concatenate(te_list)

        disjointness_checks[f"{env_name}_train_test_overlap"] = len(set(tr_idx.tolist()) & set(te_idx.tolist()))
        disjointness_checks[f"{env_name}_smoke_test_overlap"] = len(set(sm_idx.tolist()) & set(te_idx.tolist()))
        disjointness_checks[f"{env_name}_train_smoke_overlap"] = len(set(tr_idx.tolist()) & set(sm_idx.tolist()))
        tr_z_set = {tuple(env.hypotheses_z[i].tolist()) for i in tr_idx}
        sm_z_set = {tuple(env.hypotheses_z[i].tolist()) for i in sm_idx}
        te_z_set = {tuple(env.hypotheses_z[i].tolist()) for i in te_idx}
        disjointness_checks[f"{env_name}_board_z_train_test_overlap"] = len(tr_z_set & te_z_set)
        disjointness_checks[f"{env_name}_board_z_smoke_test_overlap"] = len(sm_z_set & te_z_set)
        disjointness_checks[f"{env_name}_board_z_train_smoke_overlap"] = len(tr_z_set & sm_z_set)
        env_splits[env_name] = {
            "train_idx": tr_idx,
            "smoke_idx": sm_idx,
            "test_idx": te_idx,
            "cal_idx": cal_list,
        }

    per_seed: Dict[str, Any] = {}
    oracle_access_in_forward = 0
    sgd_rule_label_access_counts = {
        "Control_Random_Network": 0,
        "Control_Untrained_Init": 0,
        "Ablation_NoRuleSlots": 0,
        "Ablation_Fused_Utility": 0,
        "Ablation_Token_Masked_Utility": 0,
        "Inconsistency_ActiveVoI_LIR": 0,
    }

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        env_results: Dict[str, Any] = {}
        for env_name, env in envs.items():
            sp = env_splits[env_name]
            train_idx = sp["train_idx"]
            eval_idx = sp["smoke_idx"] if args.smoke_cpu else sp["test_idx"]
            cal_idx = sp["cal_idx"]

            watched_rules = WatchedLabelArray(env.hypotheses_rule)
            train_mean_z = env.hypotheses_z[train_idx].mean(axis=0)

            generic_king = np.zeros((36, 36), dtype=np.float32)
            for i in range(36):
                r1, c1 = divmod(i, 6)
                for j in range(36):
                    if i != j and max(abs(r1 - j // 6), abs(c1 - j % 6)) == 1:
                        generic_king[i, j] = 1.0

            models: Dict[str, PureNeuralRuleWorldNet] = {}
            torch.manual_seed(seed + 9000)
            models["Control_Random_Network"] = PureNeuralRuleWorldNet(hidden_size=48, mode="lir_separated").to(device)
            torch.manual_seed(seed)
            models["Control_Untrained_Init"] = PureNeuralRuleWorldNet(hidden_size=48, mode="lir_separated").to(device)

            arm_configs = [
                ("Ablation_NoRuleSlots", "no_rule_slots"),
                ("Ablation_Fused_Utility", "fused_utility"),
                ("Ablation_Token_Masked_Utility", "token_masked_utility"),
                ("Inconsistency_ActiveVoI_LIR", "lir_separated"),
            ]

            z_tr_np = env.hypotheses_z[train_idx]
            c_tr_np = env.hypotheses_clue[train_idx]
            z_cal_np = env.hypotheses_z[cal_idx]
            c_cal_np = env.hypotheses_clue[cal_idx]

            for arm_name, mode in arm_configs:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                net = PureNeuralRuleWorldNet(hidden_size=48, mode=mode).to(device)
                before_rule_acc = watched_rules.access_count
                train_model_rule_label_free(net, z_tr_np, c_tr_np, z_cal_np, c_cal_np, watched_rules, n_steps, device)
                after_rule_acc = watched_rules.access_count
                sgd_rule_label_access_counts[arm_name] += after_rule_acc - before_rule_acc
                models[arm_name] = net

            def eval_policy_and_beliefs(arm_key: str) -> Dict[str, float]:
                kl_z_list, kl_rule_list = [], []
                rule_ent_after_k = []
                rule_top1_acc = []
                util_seek_list, util_avoid_list, util_rev_gap_list = [], [], []
                drift_jsd_list = []

                for idx_pos, idx in enumerate(eval_idx):
                    true_z = env.hypotheses_z[idx]
                    true_c = env.hypotheses_clue[idx]
                    true_r = int(env.hypotheses_rule[idx])

                    # D2: Start with 1 initial revealed cell; make K=4 active probes
                    rev = np.zeros(36, dtype=np.float32)
                    rev[(idx_pos * 7 + 14) % 36] = 1.0

                    for step_p in range(k_probes):
                        obs_z = true_z * rev
                        obs_c = true_c * rev
                        if arm_key in ("Control_Uniform", "Control_TrainMean", "Control_ZeroParam_LLR"):
                            unrev = [i for i in range(36) if rev[i] < 0.5]
                            next_cell = int(unrev[(idx_pos * 7 + step_p * 5) % len(unrev)])
                        elif arm_key == "Control_Helper_Stencil_Only":
                            _, _, next_cell = env.helper_stencil_control_beliefs(rev, obs_z, obs_c)
                        elif arm_key == "Ceiling_Exact_Joint_Bayes":
                            next_cell = env.exact_bayes_voi_probe(rev, obs_z, obs_c)
                        else:
                            net = models[arm_key]
                            net.eval()
                            with torch.no_grad():
                                out = net(
                                    torch.tensor(rev[None, :], device=device),
                                    torch.tensor(obs_z[None, :], device=device),
                                    torch.tensor(obs_c[None, :], device=device),
                                    torch.ones(1, device=device),
                                )
                                next_cell = int(torch.argmax(out["probe_score"][0]).item())
                        rev[next_cell] = 1.0

                    obs_z = true_z * rev
                    obs_c = true_c * rev
                    _, p_rule_star_probe = env.exact_joint_bayes(rev, obs_z, obs_c)
                    p_r_clip = np.clip(p_rule_star_probe, 1e-7, 1.0)
                    rule_ent_after_k.append(float(-(p_r_clip * np.log(p_r_clip)).sum()))

                    if arm_key in ("Control_Uniform", "Control_TrainMean", "Control_ZeroParam_LLR"):
                        mu_rule_probe = np.full(4, 0.25, dtype=np.float32)
                    elif arm_key == "Control_Helper_Stencil_Only":
                        _, mu_rule_probe, _ = env.helper_stencil_control_beliefs(rev, obs_z, obs_c)
                    elif arm_key == "Ceiling_Exact_Joint_Bayes":
                        mu_rule_probe = p_rule_star_probe
                    else:
                        net = models[arm_key]
                        net.eval()
                        with torch.no_grad():
                            out_pr = net(
                                torch.tensor(rev[None, :], device=device),
                                torch.tensor(obs_z[None, :], device=device),
                                torch.tensor(obs_c[None, :], device=device),
                                torch.ones(1, device=device),
                            )
                            mu_rule_probe = out_pr["mu_rule"][0].cpu().numpy()
                    rule_top1_acc.append(float(int(np.argmax(mu_rule_probe)) == true_r))

                    # Evaluate D1 & D3 beliefs and policy actions on matched 5-cell context mask across all arms
                    rev_fixed = np.zeros(36, dtype=np.float32)
                    for c_off in (0, 7, 14, 21, 28):
                        rev_fixed[(idx_pos * 5 + c_off) % 36] = 1.0
                    obs_z_f = true_z * rev_fixed
                    obs_c_f = true_c * rev_fixed
                    p_z_star, p_rule_star = env.exact_joint_bayes(rev_fixed, obs_z_f, obs_c_f)

                    if arm_key == "Control_Uniform":
                        mu_z = rev_fixed * obs_z_f + (1.0 - rev_fixed) * 0.5
                        mu_rule = np.full(4, 0.25, dtype=np.float32)
                        mu_z_neg = mu_z.copy()
                        act_pos = np.linspace(0.0, 1e-3, 36, dtype=np.float32)
                        act_neg = act_pos.copy()
                    elif arm_key == "Control_TrainMean":
                        mu_z = rev_fixed * obs_z_f + (1.0 - rev_fixed) * train_mean_z
                        mu_rule = np.full(4, 0.25, dtype=np.float32)
                        mu_z_neg = mu_z.copy()
                        act_pos = train_mean_z.copy()
                        act_neg = train_mean_z.copy()
                    elif arm_key == "Control_ZeroParam_LLR":
                        local_llr = (generic_king @ (obs_c_f * rev_fixed)) / np.maximum(generic_king @ rev_fixed, 1.0)
                        prob_llr = 1.0 / (1.0 + np.exp(-(local_llr - 1.0)))
                        mu_z = rev_fixed * obs_z_f + (1.0 - rev_fixed) * prob_llr.astype(np.float32)
                        mu_rule = np.full(4, 0.25, dtype=np.float32)
                        mu_z_neg = mu_z.copy()
                        act_pos = mu_z.copy()
                        act_neg = 1.0 - mu_z.copy()
                    elif arm_key == "Control_Helper_Stencil_Only":
                        mu_z, mu_rule, _ = env.helper_stencil_control_beliefs(rev_fixed, obs_z_f, obs_c_f)
                        mu_z_neg = mu_z.copy()
                        act_pos = mu_z.copy()
                        act_neg = 1.0 - mu_z.copy()
                    elif arm_key == "Ceiling_Exact_Joint_Bayes":
                        mu_z = p_z_star
                        mu_rule = p_rule_star
                        mu_z_neg = p_z_star.copy()
                        act_pos = p_z_star.copy()
                        act_neg = 1.0 - p_z_star.copy()
                    else:
                        net = models[arm_key]
                        net.eval()
                        with torch.no_grad():
                            out_pos = net(
                                torch.tensor(rev_fixed[None, :], device=device),
                                torch.tensor(obs_z_f[None, :], device=device),
                                torch.tensor(obs_c_f[None, :], device=device),
                                torch.ones(1, device=device),
                            )
                            out_neg = net(
                                torch.tensor(rev_fixed[None, :], device=device),
                                torch.tensor(obs_z_f[None, :], device=device),
                                torch.tensor(obs_c_f[None, :], device=device),
                                -torch.ones(1, device=device),
                            )
                            mu_z = out_pos["mu_z"][0].cpu().numpy()
                            mu_rule = out_pos["mu_rule"][0].cpu().numpy()
                            mu_z_neg = out_neg["mu_z"][0].cpu().numpy()
                            act_pos = out_pos["act_logits"][0].cpu().numpy()
                            act_neg = out_neg["act_logits"][0].cpu().numpy()

                    kl_z_list.append(binary_kl(p_z_star, mu_z))
                    kl_rule_list.append(categorical_kl(p_rule_star[None, :], mu_rule[None, :]))
                    drift_jsd_list.append(binary_jsd(mu_z, mu_z_neg))

                    unrev_idx = np.where(rev_fixed < 0.5)[0]
                    top_seek = unrev_idx[np.argsort(-act_pos[unrev_idx])[:3]]
                    top_avoid = unrev_idx[np.argsort(-act_neg[unrev_idx])[:3]]
                    seek_u_pos = float(true_z[top_seek].mean())
                    seek_u_neg = float(true_z[top_avoid].mean())
                    util_seek_list.append(seek_u_pos)
                    util_avoid_list.append(float(1.0 - seek_u_neg))
                    util_rev_gap_list.append(seek_u_pos - seek_u_neg)

                return {
                    "kl_z": float(np.mean(kl_z_list)),
                    "kl_rule": float(np.mean(kl_rule_list)),
                    "true_rule_entropy_after_probes": float(np.mean(rule_ent_after_k)),
                    "rule_top1_acc": float(np.mean(rule_top1_acc)),
                    "utility_seek_score": float(np.mean(util_seek_list)),
                    "utility_avoid_score": float(np.mean(util_avoid_list)),
                    "utility_action_reversal_gap": float(np.mean(util_rev_gap_list)),
                    "utility_swap_belief_jsd": float(np.mean(drift_jsd_list)),
                }

            all_arm_keys = [
                "Control_Uniform",
                "Control_TrainMean",
                "Control_ZeroParam_LLR",
                "Control_Helper_Stencil_Only",
                "Control_Random_Network",
                "Control_Untrained_Init",
                "Ablation_NoRuleSlots",
                "Ablation_Fused_Utility",
                "Ablation_Token_Masked_Utility",
                "Inconsistency_ActiveVoI_LIR",
                "Ceiling_Exact_Joint_Bayes",
            ]
            arms = {k: eval_policy_and_beliefs(k) for k in all_arm_keys}
            env_results[env_name] = arms
        per_seed[str(seed)] = env_results

    runner_sha256 = compute_sha256(Path(__file__).resolve())
    prereg_path = Path(__file__).resolve().parents[1] / "benchmarks" / "PREREG-R27B-UNCERTAIN-RULES-ACTIVE-VOI.md"
    prereg_sha256 = compute_sha256(prereg_path) if prereg_path.exists() else ""

    results = {
        "round": "R27B",
        "runner_sha256": runner_sha256,
        "prereg_sha256": prereg_sha256,
        "device": str(device),
        "smoke_cpu": bool(args.smoke_cpu),
        "raw_input_channels": 3,
        "oracle_access_in_forward": oracle_access_in_forward,
        "sgd_rule_label_access_counts": sgd_rule_label_access_counts,
        "post_em_slot_alignment_episodes_per_rule": 1,
        "disjointness_checks": disjointness_checks,
        "oracle_check": oracle_check,
        "seeds": seeds,
        "per_seed": per_seed,
    }
    results["gates"] = evaluate_r27b_gates(results)
    return results


def evaluate_r27b_gates(results: Dict[str, Any]) -> Dict[str, Any]:
    is_smoke = bool(results.get("smoke_cpu", False))
    seeds = [str(s) for s in results["seeds"]]
    per_seed = results["per_seed"]
    envs = ["shiphunt_6x6", "minesweeper_6x6"]

    req_arms = [
        "Control_Uniform",
        "Control_TrainMean",
        "Control_ZeroParam_LLR",
        "Control_Helper_Stencil_Only",
        "Control_Random_Network",
        "Control_Untrained_Init",
        "Ablation_NoRuleSlots",
        "Ablation_Fused_Utility",
        "Ablation_Token_Masked_Utility",
        "Inconsistency_ActiveVoI_LIR",
        "Ceiling_Exact_Joint_Bayes",
    ]
    controls_ok = all(all(c in per_seed[s][e] for c in req_arms) for s in seeds for e in envs)
    oc = results.get("oracle_check", {})
    sgd_acc = results.get("sgd_rule_label_access_counts", {})
    disj = results.get("disjointness_checks", {})
    d0_pass = bool(
        results["raw_input_channels"] == 3
        and results["oracle_access_in_forward"] == 0
        and len(sgd_acc) == 6
        and all(v == 0 for v in sgd_acc.values())
        and results.get("post_em_slot_alignment_episodes_per_rule") == 1
        and len(disj) == 12
        and all(v == 0 for v in disj.values())
        and oc.get("episodes_checked", 0) >= 20
        and oc.get("max_tv_z", 1.0) <= 1e-6
        and oc.get("max_tv_rule", 1.0) <= 1e-6
        and controls_ok
    )

    def env_metric(arm: str, env_n: str, key: str) -> List[float]:
        return [float(per_seed[s][env_n][arm][key]) for s in seeds]

    def avg_metric(arm: str, key: str) -> List[float]:
        return [float(np.mean([per_seed[s][e][arm][key] for e in envs])) for s in seeds]

    # D1: Checked BOTH overall AND per-environment (shiphunt_6x6 and minesweeper_6x6 separately)
    ours_z = avg_metric("Inconsistency_ActiveVoI_LIR", "kl_z")
    ci_z_unif = paired_ci95(avg_metric("Control_Uniform", "kl_z"), ours_z)
    ci_z_tmean = paired_ci95(avg_metric("Control_TrainMean", "kl_z"), ours_z)
    ci_z_llr = paired_ci95(avg_metric("Control_ZeroParam_LLR", "kl_z"), ours_z)
    ci_z_help = paired_ci95(avg_metric("Control_Helper_Stencil_Only", "kl_z"), ours_z)
    ci_z_rand = paired_ci95(avg_metric("Control_Random_Network", "kl_z"), ours_z)
    ci_z_untr = paired_ci95(avg_metric("Control_Untrained_Init", "kl_z"), ours_z)
    ci_z_norule = paired_ci95(avg_metric("Ablation_NoRuleSlots", "kl_z"), ours_z)

    ours_r = avg_metric("Inconsistency_ActiveVoI_LIR", "kl_rule")
    ci_r_unif = paired_ci95(avg_metric("Control_Uniform", "kl_rule"), ours_r)
    ci_r_help = paired_ci95(avg_metric("Control_Helper_Stencil_Only", "kl_rule"), ours_r)
    ci_r_rand = paired_ci95(avg_metric("Control_Random_Network", "kl_rule"), ours_r)
    ci_r_untr = paired_ci95(avg_metric("Control_Untrained_Init", "kl_rule"), ours_r)
    ci_r_norule = paired_ci95(avg_metric("Ablation_NoRuleSlots", "kl_rule"), ours_r)

    per_env_d1 = {}
    for env_n in envs:
        oz = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "kl_z")
        or_ = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "kl_rule")
        per_env_d1[env_n] = {
            "kl_z_diff_vs_train_mean": paired_ci95(env_metric("Control_TrainMean", env_n, "kl_z"), oz),
            "kl_z_diff_vs_helper": paired_ci95(env_metric("Control_Helper_Stencil_Only", env_n, "kl_z"), oz),
            "kl_z_diff_vs_no_rule_slots": paired_ci95(env_metric("Ablation_NoRuleSlots", env_n, "kl_z"), oz),
            "kl_rule_diff_vs_helper": paired_ci95(env_metric("Control_Helper_Stencil_Only", env_n, "kl_rule"), or_),
            "kl_rule_diff_vs_no_rule_slots": paired_ci95(env_metric("Ablation_NoRuleSlots", env_n, "kl_rule"), or_),
        }

    d1_pass = bool(
        all(
            ci["mean_diff"] >= 0.04 and ci["ci95_low"] > 0.0
            for ci in [ci_z_unif, ci_z_tmean, ci_z_llr, ci_z_help, ci_z_rand, ci_z_untr, ci_z_norule]
        )
        and all(
            ci["mean_diff"] >= 0.25 and ci["ci95_low"] > 0.0
            for ci in [ci_r_unif, ci_r_help, ci_r_rand, ci_r_untr, ci_r_norule]
        )
        and all(
            per_env_d1[e]["kl_z_diff_vs_train_mean"]["mean_diff"] >= 0.04
            and per_env_d1[e]["kl_z_diff_vs_train_mean"]["ci95_low"] > 0.0
            and per_env_d1[e]["kl_z_diff_vs_helper"]["mean_diff"] >= 0.04
            and per_env_d1[e]["kl_z_diff_vs_helper"]["ci95_low"] > 0.0
            and per_env_d1[e]["kl_z_diff_vs_no_rule_slots"]["mean_diff"] >= 0.04
            and per_env_d1[e]["kl_z_diff_vs_no_rule_slots"]["ci95_low"] > 0.0
            and per_env_d1[e]["kl_rule_diff_vs_helper"]["mean_diff"] >= 0.25
            and per_env_d1[e]["kl_rule_diff_vs_helper"]["ci95_low"] > 0.0
            and per_env_d1[e]["kl_rule_diff_vs_no_rule_slots"]["mean_diff"] >= 0.25
            and per_env_d1[e]["kl_rule_diff_vs_no_rule_slots"]["ci95_low"] > 0.0
            for e in envs
        )
    )

    # D2: Strategic Active Rule VoI Probing (gated BOTH overall AND per-environment)
    ours_ent = avg_metric("Inconsistency_ActiveVoI_LIR", "true_rule_entropy_after_probes")
    unif_ent = avg_metric("Control_Uniform", "true_rule_entropy_after_probes")
    help_ent = avg_metric("Control_Helper_Stencil_Only", "true_rule_entropy_after_probes")
    rand_ent = avg_metric("Control_Random_Network", "true_rule_entropy_after_probes")
    untr_ent = avg_metric("Control_Untrained_Init", "true_rule_entropy_after_probes")
    norule_ent = avg_metric("Ablation_NoRuleSlots", "true_rule_entropy_after_probes")
    ours_top1 = avg_metric("Inconsistency_ActiveVoI_LIR", "rule_top1_acc")
    rand_top1 = avg_metric("Control_Random_Network", "rule_top1_acc")
    untr_top1 = avg_metric("Control_Untrained_Init", "rule_top1_acc")

    ci_ent_unif = paired_ci95(unif_ent, ours_ent)
    ci_ent_help = paired_ci95(help_ent, ours_ent)
    ci_ent_rand = paired_ci95(rand_ent, ours_ent)
    ci_ent_untr = paired_ci95(untr_ent, ours_ent)
    ci_ent_norule = paired_ci95(norule_ent, ours_ent)
    rel_red_unif = float((np.mean(unif_ent) - np.mean(ours_ent)) / max(np.mean(unif_ent), 1e-8))
    rel_red_norule = float((np.mean(norule_ent) - np.mean(ours_ent)) / max(np.mean(norule_ent), 1e-8))
    rel_red_help = float((np.mean(help_ent) - np.mean(ours_ent)) / max(np.mean(help_ent), 1e-8))

    per_env_d2 = {}
    for env_n in envs:
        oe = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "true_rule_entropy_after_probes")
        ue = env_metric("Control_Uniform", env_n, "true_rule_entropy_after_probes")
        he = env_metric("Control_Helper_Stencil_Only", env_n, "true_rule_entropy_after_probes")
        re_ = env_metric("Control_Random_Network", env_n, "true_rule_entropy_after_probes")
        ute = env_metric("Control_Untrained_Init", env_n, "true_rule_entropy_after_probes")
        nre = env_metric("Ablation_NoRuleSlots", env_n, "true_rule_entropy_after_probes")
        ot1 = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "rule_top1_acc")
        rt1 = env_metric("Control_Random_Network", env_n, "rule_top1_acc")
        ut1 = env_metric("Control_Untrained_Init", env_n, "rule_top1_acc")
        per_env_d2[env_n] = {
            "entropy_ours_mean": float(np.mean(oe)),
            "rel_reduction_vs_uniform": float((np.mean(ue) - np.mean(oe)) / max(np.mean(ue), 1e-8)),
            "rel_reduction_vs_helper": float((np.mean(he) - np.mean(oe)) / max(np.mean(he), 1e-8)),
            "rel_reduction_vs_no_rule_slots": float((np.mean(nre) - np.mean(oe)) / max(np.mean(nre), 1e-8)),
            "ci_entropy_diff_vs_uniform": paired_ci95(ue, oe),
            "ci_entropy_diff_vs_helper": paired_ci95(he, oe),
            "ci_entropy_diff_vs_random_net": paired_ci95(re_, oe),
            "ci_entropy_diff_vs_untrained": paired_ci95(ute, oe),
            "ci_entropy_diff_vs_no_rule_slots": paired_ci95(nre, oe),
            "rule_top1_acc_ours_mean": float(np.mean(ot1)),
            "rule_top1_acc_random_net_mean": float(np.mean(rt1)),
            "rule_top1_acc_untrained_mean": float(np.mean(ut1)),
        }

    d2_pass = bool(
        rel_red_unif >= 0.25
        and ci_ent_unif["ci95_low"] > 0.0
        and rel_red_norule >= 0.15
        and ci_ent_norule["ci95_low"] > 0.0
        and rel_red_help >= 0.15
        and ci_ent_help["ci95_low"] > 0.0
        and ci_ent_rand["ci95_low"] > 0.0
        and ci_ent_untr["ci95_low"] > 0.0
        and float(np.mean(ours_top1)) >= 0.80
        and float(np.mean(rand_top1)) <= 0.35
        and float(np.mean(untr_top1)) <= 0.35
        and all(
            per_env_d2[e]["rel_reduction_vs_uniform"] >= 0.25
            and per_env_d2[e]["ci_entropy_diff_vs_uniform"]["ci95_low"] > 0.0
            and per_env_d2[e]["rel_reduction_vs_no_rule_slots"] >= 0.15
            and per_env_d2[e]["ci_entropy_diff_vs_no_rule_slots"]["ci95_low"] > 0.0
            and per_env_d2[e]["rel_reduction_vs_helper"] >= 0.15
            and per_env_d2[e]["ci_entropy_diff_vs_helper"]["ci95_low"] > 0.0
            and per_env_d2[e]["ci_entropy_diff_vs_random_net"]["ci95_low"] > 0.0
            and per_env_d2[e]["ci_entropy_diff_vs_untrained"]["ci95_low"] > 0.0
            and per_env_d2[e]["rule_top1_acc_ours_mean"] >= 0.80
            and per_env_d2[e]["rule_top1_acc_random_net_mean"] <= 0.35
            and per_env_d2[e]["rule_top1_acc_untrained_mean"] <= 0.35
            for e in envs
        )
    )

    # D3: By-construction LIR firewall vs genuine learned ablation drift + policy action reversal (gated BOTH overall AND per-env)
    ours_jsd = avg_metric("Inconsistency_ActiveVoI_LIR", "utility_swap_belief_jsd")
    fused_jsd = avg_metric("Ablation_Fused_Utility", "utility_swap_belief_jsd")
    tmask_jsd = avg_metric("Ablation_Token_Masked_Utility", "utility_swap_belief_jsd")

    ours_u_seek = avg_metric("Inconsistency_ActiveVoI_LIR", "utility_seek_score")
    unif_u_seek = avg_metric("Control_Uniform", "utility_seek_score")
    tmean_u_seek = avg_metric("Control_TrainMean", "utility_seek_score")
    rand_u_seek = avg_metric("Control_Random_Network", "utility_seek_score")
    untr_u_seek = avg_metric("Control_Untrained_Init", "utility_seek_score")
    norule_u_seek = avg_metric("Ablation_NoRuleSlots", "utility_seek_score")

    ours_u_avoid = avg_metric("Inconsistency_ActiveVoI_LIR", "utility_avoid_score")
    unif_u_avoid = avg_metric("Control_Uniform", "utility_avoid_score")
    ceil_u_avoid = avg_metric("Ceiling_Exact_Joint_Bayes", "utility_avoid_score")

    ours_rev_gap = avg_metric("Inconsistency_ActiveVoI_LIR", "utility_action_reversal_gap")
    tmean_rev_gap = avg_metric("Control_TrainMean", "utility_action_reversal_gap")
    rand_rev_gap = avg_metric("Control_Random_Network", "utility_action_reversal_gap")
    untr_rev_gap = avg_metric("Control_Untrained_Init", "utility_action_reversal_gap")
    norule_rev_gap = avg_metric("Ablation_NoRuleSlots", "utility_action_reversal_gap")

    ci_u_seek_unif = paired_ci95(ours_u_seek, unif_u_seek)
    ci_u_seek_tmean = paired_ci95(ours_u_seek, tmean_u_seek)
    ci_u_seek_rand = paired_ci95(ours_u_seek, rand_u_seek)
    ci_u_seek_untr = paired_ci95(ours_u_seek, untr_u_seek)
    ci_u_seek_norule = paired_ci95(ours_u_seek, norule_u_seek)
    ci_u_avoid = paired_ci95(ours_u_avoid, unif_u_avoid)
    ci_rev_norule = paired_ci95(ours_rev_gap, norule_rev_gap)

    per_env_d3 = {}
    for env_n in envs:
        o_jsd = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "utility_swap_belief_jsd")
        f_jsd = env_metric("Ablation_Fused_Utility", env_n, "utility_swap_belief_jsd")
        t_jsd = env_metric("Ablation_Token_Masked_Utility", env_n, "utility_swap_belief_jsd")
        o_sk = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "utility_seek_score")
        u_sk = env_metric("Control_Uniform", env_n, "utility_seek_score")
        tm_sk = env_metric("Control_TrainMean", env_n, "utility_seek_score")
        rn_sk = env_metric("Control_Random_Network", env_n, "utility_seek_score")
        ut_sk = env_metric("Control_Untrained_Init", env_n, "utility_seek_score")
        nr_sk = env_metric("Ablation_NoRuleSlots", env_n, "utility_seek_score")
        o_av = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "utility_avoid_score")
        u_av = env_metric("Control_Uniform", env_n, "utility_avoid_score")
        c_av = env_metric("Ceiling_Exact_Joint_Bayes", env_n, "utility_avoid_score")
        o_rg = env_metric("Inconsistency_ActiveVoI_LIR", env_n, "utility_action_reversal_gap")
        nr_rg = env_metric("Ablation_NoRuleSlots", env_n, "utility_action_reversal_gap")
        per_env_d3[env_n] = {
            "max_belief_jsd_ours": float(max(o_jsd)),
            "mean_belief_jsd_fused_ablation": float(np.mean(f_jsd)),
            "mean_belief_jsd_token_masked_ablation": float(np.mean(t_jsd)),
            "utility_seek_ours_mean": float(np.mean(o_sk)),
            "utility_avoid_ours_mean": float(np.mean(o_av)),
            "utility_avoid_ceiling_mean": float(np.mean(c_av)),
            "utility_action_reversal_gap_ours_mean": float(np.mean(o_rg)),
            "ci_seek_vs_uniform": paired_ci95(o_sk, u_sk),
            "ci_seek_vs_train_mean": paired_ci95(o_sk, tm_sk),
            "ci_seek_vs_random_net": paired_ci95(o_sk, rn_sk),
            "ci_seek_vs_untrained": paired_ci95(o_sk, ut_sk),
            "ci_seek_vs_no_rule_slots": paired_ci95(o_sk, nr_sk),
            "ci_avoid_vs_uniform": paired_ci95(o_av, u_av),
            "ci_rev_gap_vs_no_rule_slots": paired_ci95(o_rg, nr_rg),
        }

    d3_pass = bool(
        max(ours_jsd) <= 1e-7
        and float(np.mean(fused_jsd)) >= 0.004
        and float(np.mean(tmask_jsd)) >= 0.004
        and float(np.mean(ours_u_seek)) >= 0.35
        and all(
            ci["ci95_low"] > 0.0
            for ci in [ci_u_seek_unif, ci_u_seek_tmean, ci_u_seek_rand, ci_u_seek_untr, ci_u_seek_norule]
        )
        and float(np.mean(ours_u_avoid)) >= 0.80
        and float(np.mean(ours_u_avoid)) >= 0.85 * float(np.mean(ceil_u_avoid))
        and ci_u_avoid["ci95_low"] > 0.0
        and float(np.mean(ours_rev_gap)) >= 0.30
        and ci_rev_norule["ci95_low"] > 0.0
        and abs(float(np.mean(tmean_rev_gap))) <= 1e-6
        and abs(float(np.mean(rand_rev_gap))) <= 0.05
        and abs(float(np.mean(untr_rev_gap))) <= 0.05
        and all(
            per_env_d3[e]["max_belief_jsd_ours"] <= 1e-7
            and per_env_d3[e]["mean_belief_jsd_fused_ablation"] >= 0.004
            and per_env_d3[e]["mean_belief_jsd_token_masked_ablation"] >= 0.004
            and per_env_d3[e]["utility_seek_ours_mean"] >= 0.35
            and per_env_d3[e]["ci_seek_vs_uniform"]["ci95_low"] > 0.0
            and per_env_d3[e]["ci_seek_vs_train_mean"]["ci95_low"] > 0.0
            and per_env_d3[e]["ci_seek_vs_random_net"]["ci95_low"] > 0.0
            and per_env_d3[e]["ci_seek_vs_untrained"]["ci95_low"] > 0.0
            and per_env_d3[e]["ci_seek_vs_no_rule_slots"]["ci95_low"] > 0.0
            and per_env_d3[e]["utility_avoid_ours_mean"] >= 0.80
            and per_env_d3[e]["utility_avoid_ours_mean"] >= 0.85 * per_env_d3[e]["utility_avoid_ceiling_mean"]
            and per_env_d3[e]["ci_avoid_vs_uniform"]["ci95_low"] > 0.0
            and per_env_d3[e]["utility_action_reversal_gap_ours_mean"] >= 0.30
            and per_env_d3[e]["ci_rev_gap_vs_no_rule_slots"]["ci95_low"] > 0.0
            for e in envs
        )
    )

    def fmt_verdict(passed: bool) -> str:
        if is_smoke:
            return "SMOKE_NO_VERDICT"
        return "PASS" if passed else "FAIL"

    return {
        "R27B_D0_leak_free_exact_oracle_and_controls_present": {
            "verdict": fmt_verdict(d0_pass),
            "pass": d0_pass,
            "raw_input_channels": results["raw_input_channels"],
            "oracle_access_in_forward": results["oracle_access_in_forward"],
            "sgd_rule_label_access_counts": sgd_acc,
            "post_em_slot_alignment_episodes_per_rule": results.get("post_em_slot_alignment_episodes_per_rule"),
            "disjointness_checks": disj,
            "oracle_check": oc,
        },
        "R27B_D1_joint_state_and_rule_belief_beats_controls_and_no_rule_ablation": {
            "verdict": fmt_verdict(d1_pass),
            "pass": d1_pass,
            "kl_z_ours_mean": float(np.mean(ours_z)),
            "kl_z_diff_vs_uniform": ci_z_unif,
            "kl_z_diff_vs_train_mean": ci_z_tmean,
            "kl_z_diff_vs_llr": ci_z_llr,
            "kl_z_diff_vs_helper": ci_z_help,
            "kl_z_diff_vs_random_net": ci_z_rand,
            "kl_z_diff_vs_untrained": ci_z_untr,
            "kl_z_diff_vs_no_rule_slots": ci_z_norule,
            "kl_rule_ours_mean": float(np.mean(ours_r)),
            "kl_rule_diff_vs_uniform": ci_r_unif,
            "kl_rule_diff_vs_helper": ci_r_help,
            "kl_rule_diff_vs_random_net": ci_r_rand,
            "kl_rule_diff_vs_untrained": ci_r_untr,
            "kl_rule_diff_vs_no_rule_slots": ci_r_norule,
            "per_env": per_env_d1,
        },
        "R27B_D2_strategic_active_rule_voi_probing": {
            "verdict": fmt_verdict(d2_pass),
            "pass": d2_pass,
            "entropy_ours_mean": float(np.mean(ours_ent)),
            "entropy_uniform_mean": float(np.mean(unif_ent)),
            "entropy_helper_mean": float(np.mean(help_ent)),
            "entropy_random_net_mean": float(np.mean(rand_ent)),
            "entropy_untrained_mean": float(np.mean(untr_ent)),
            "entropy_no_rule_slots_mean": float(np.mean(norule_ent)),
            "rel_reduction_vs_uniform": rel_red_unif,
            "rel_reduction_vs_no_rule_slots": rel_red_norule,
            "rel_reduction_vs_helper": rel_red_help,
            "ci_entropy_diff_vs_uniform": ci_ent_unif,
            "ci_entropy_diff_vs_helper": ci_ent_help,
            "ci_entropy_diff_vs_random_net": ci_ent_rand,
            "ci_entropy_diff_vs_untrained": ci_ent_untr,
            "ci_entropy_diff_vs_no_rule_slots": ci_ent_norule,
            "rule_top1_acc_ours_mean": float(np.mean(ours_top1)),
            "rule_top1_acc_random_net_mean": float(np.mean(rand_top1)),
            "rule_top1_acc_untrained_mean": float(np.mean(untr_top1)),
            "per_env": per_env_d2,
        },
        "R27B_D3_lir_structural_invariance_vs_learned_ablation_drift": {
            "verdict": fmt_verdict(d3_pass),
            "pass": d3_pass,
            "max_belief_jsd_ours": float(max(ours_jsd)),
            "mean_belief_jsd_fused_ablation": float(np.mean(fused_jsd)),
            "mean_belief_jsd_token_masked_ablation": float(np.mean(tmask_jsd)),
            "utility_seek_ours_mean": float(np.mean(ours_u_seek)),
            "utility_avoid_ours_mean": float(np.mean(ours_u_avoid)),
            "utility_avoid_ceiling_mean": float(np.mean(ceil_u_avoid)),
            "utility_action_reversal_gap_ours_mean": float(np.mean(ours_rev_gap)),
            "utility_action_reversal_gap_no_rule_slots_mean": float(np.mean(norule_rev_gap)),
            "ci_seek_vs_uniform": ci_u_seek_unif,
            "ci_seek_vs_train_mean": ci_u_seek_tmean,
            "ci_seek_vs_random_net": ci_u_seek_rand,
            "ci_seek_vs_untrained": ci_u_seek_untr,
            "ci_seek_vs_no_rule_slots": ci_u_seek_norule,
            "ci_avoid_vs_uniform": ci_u_avoid,
            "ci_rev_gap_vs_no_rule_slots": ci_rev_norule,
            "per_env": per_env_d3,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--n-test", type=int, default=80)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--out", type=str, default="results/r27/r27b_results.json")
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()
    t0 = time.time()
    res = run_r27b(args)
    res["elapsed_sec"] = round(time.time() - t0, 2)
    out_p = (Path(args.out_dir) / "r27b_results.json") if args.out_dir else Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w") as f:
        json.dump(res, f, indent=2)
    print(f"[R27B] Saved to {out_p} ({res['elapsed_sec']}s)")
    for k, v in res["gates"].items():
        print(f"  {k}: pass={v['pass']} verdict={v['verdict']}")
