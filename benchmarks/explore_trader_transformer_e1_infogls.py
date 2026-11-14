#!/usr/bin/env python3
"""Exploratory Sweep on E1 (Linear-Gaussian with Shared Noise Factors): Information-Form & GLS Heads.

Answers opus-mats #3865:
- E1 (theta ~ N(0, I_4), agents observe s_i = A_i theta + B_i eta + sigma_i eps_i) is the one toy env where
  every neural method fails (fresh bar 1.095 nats at none, 1.103 at noisy; extremized 0.829; sum_logodds 0.827).
- We state the exact delta from known work at every step of the head/architecture ladder:
  1. Moment/Cholesky Head (`bar_fct_moment_w64_l3`, `bar_fct_moment_w128_l4`, `tt_moment_all`):
     Baseline head predicting (mean, L) directly from final hidden state; must learn 4x4 matrix inversion P^-1 h.
  2. Delta 1 — Pure Information-Form Head (`tt_pure_info_all`, `fct_pure_info_bidir`):
     Zero domain-specific (A_i, R_i) wiring. Each trader token outputs natural-parameter increments
     (dP_k = L_k L_k^T >= 0, dh_k), summed over prefix k <= t: P = I_d + sum dP_k, h = sum dh_k, with posterior
     mean computed analytically via torch.linalg.solve(P, h).
  3. Delta 2 — Own-Sensor (A_i, R_i) Information-Form Head (`tt_own_info_*`, `fct_own_info_bidir`):
     Each trader token carries its own sensor model (A_i, R_i) and observation s_i, defining its exact local
     likelihood precision dP_i^own = A_i^T R_i^-1 A_i and information vector h_i^own = A_i^T R_i^-1 s_i (100%
     invariant to unaligned slot permutations T_i). The transformer predicts per-trader reliability gates +
     natural-parameter cross-trader correlation corrections (dP_corr, dh_corr) and solves mu = P^-1 h analytically.
     Evaluated across Axes (A) attn_layers (all, last, first, every_other, none), (B) loss (final, every_prefix,
     distill with uses_oracle=True), and (C) token_state (kv, recurrent, feedback).
  4. Delta 3 — Row-Equivariant Latent-Factor Woodbury/GLS + Information-Form Head (`tt_own_info_gls_all`,
     `fct_own_info_gls_bidir`):
     Each trader k's m=2 observation rows are encoded row-equivariantly with trader state z_k to emit an effective
     shared-noise factor loading B_hat_k in R^{m x r_lat} and private variance sigma_hat_k^2 > 0, forming the joint
     correlated noise covariance Sigma_hat = B_hat B_hat^T + blockdiag(gamma_k R_k + diag(sigma_hat_k^2)) and
     solving the exact differentiable GLS information equations P_GLS = I_d + A^T Sigma_hat^-1 A + dP_res,
     h_GLS = A^T Sigma_hat^-1 s + dh_res.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pathlib
import resource
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.envs import Batch, gaussian_global, make_env
from aggbench.metrics import summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.sequential import forecast_kl, sample_orders
from aggbench.trader_transformer import resolve_attn_layers
from aggbench.transformer import FRESH_STATS_START, FreshStream, FullContext, FullContextTransformer, full_context
from aggbench.verify import oracle_check


def exact_e1_prefix_bayes(batch: Batch, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact prefix Bayes (mean (n, N, d), prec (n, N, d, d)) for E1 along arrival order `order` (batched)."""
    n, n_ag = order.shape
    d = int(batch.public["d"])
    out_m = np.zeros((n, n_ag, d), dtype=np.float32)
    out_p = np.zeros((n, n_ag, d, d), dtype=np.float32)
    row_idx = np.arange(n)[:, None]
    A_ord = batch.oracle["A"][row_idx, order]
    Bf_ord = batch.oracle["Bf"][row_idx, order]
    sig_ord = batch.oracle["sig"][row_idx, order]
    s_ord = batch.obs[row_idx, order]
    m = A_ord.shape[2]
    r = Bf_ord.shape[3]
    eye_d = np.eye(d, dtype=np.float64)[None]
    for t in range(n_ag):
        km = (t + 1) * m
        K = A_ord[:, : t + 1].reshape(n, km, d)
        Bk = Bf_ord[:, : t + 1].reshape(n, km, r)
        sig2 = np.repeat(sig_ord[:, : t + 1] ** 2, m, axis=1)
        R = np.matmul(Bk, Bk.transpose(0, 2, 1)) + sig2[:, :, None] * np.eye(km, dtype=np.float64)[None]
        sf = s_ord[:, : t + 1].reshape(n, km, 1)
        Rinv_K = np.linalg.solve(R, K)
        Rinv_s = np.linalg.solve(R, sf)
        P = eye_d + np.matmul(K.transpose(0, 2, 1), Rinv_K)
        rhs = np.matmul(K.transpose(0, 2, 1), Rinv_s)
        mu = np.linalg.solve(P, rhs)[..., 0]
        out_m[:, t] = mu.astype(np.float32)
        out_p[:, t] = P.astype(np.float32)
    return out_m, out_p


def _sample_and_featurize_e1_chunk(
    args: tuple[int, int, int, int, tuple[str, ...], bool]
) -> tuple[int, int, np.ndarray, dict[str, np.ndarray], np.ndarray | None, np.ndarray | None]:
    """Worker: samples `cnt` E1 episodes at `start`, applies arrival order `ord_tr`, extracts float32 tokens & prefix Bayes."""
    pos, cnt, start, seed, vis_tuple, need_pb = args
    env = make_env("E1")
    b = env.sample("train", cnt, start=start)
    ord_tr = sample_orders(b.seeds, b.obs.shape[1], seed)
    row_idx = np.arange(cnt)[:, None]
    tok_dict: dict[str, np.ndarray] = {}
    for vis in vis_tuple:
        fct = full_context(b, vis, "unaligned", own_model=True)
        tok_dict[vis] = np.ascontiguousarray(fct.tokens[row_idx, ord_tr], dtype=np.float32)
    pb_m, pb_p = (exact_e1_prefix_bayes(b, ord_tr) if need_pb else (None, None))
    truth_f32 = np.ascontiguousarray(b.truth, dtype=np.float32)
    return pos, cnt, truth_f32, tok_dict, pb_m, pb_p


def sample_e1_seed_featurized(
    seed: int,
    n_train: int,
    n_std: int,
    vis_list: list[str],
    in_dims: dict[str, int],
    n_tr: int = 5,
    d: int = 4,
    need_train_pb: bool = True,
    start_stride: int | None = None,
) -> dict[str, Any]:
    """Streams chunks via pool.imap into pre-allocated float32 buffers (< 3.5 GB peak RAM per seed)."""
    truth_arr = np.empty((n_train, d), dtype=np.float32)
    tok_by_vis = {vis: np.empty((n_train, n_tr, in_dims[vis]), dtype=np.float32) for vis in vis_list}
    pb_m_arr = np.empty((n_std, n_tr, d), dtype=np.float32) if need_train_pb else None
    pb_p_arr = np.empty((n_std, n_tr, d, d), dtype=np.float32) if need_train_pb else None

    start_base = seed * (start_stride if start_stride is not None else n_train)
    w = min(8, os.cpu_count() or 4)
    chunk = min(8192, max(512, n_train // max(1, w * 4)))
    vis_tuple = tuple(vis_list)
    jobs: list[tuple[int, int, int, int, tuple[str, ...], bool]] = []
    pos = 0
    while pos < n_train:
        c = min(chunk, n_train - pos)
        if pos < n_std and pos + c > n_std:
            c = n_std - pos
        need_pb = need_train_pb and (pos < n_std)
        jobs.append((pos, c, start_base + pos, seed, vis_tuple, need_pb))
        pos += c

    if len(jobs) == 1:
        iterator = [_sample_and_featurize_e1_chunk(jobs[0])]
        for p_off, cnt, tr_c, tk_c, pm_c, pp_c in iterator:
            truth_arr[p_off : p_off + cnt] = tr_c
            for vis in vis_list:
                tok_by_vis[vis][p_off : p_off + cnt] = tk_c[vis]
            if pm_c is not None and pp_c is not None and pb_m_arr is not None and pb_p_arr is not None:
                pb_m_arr[p_off : p_off + cnt] = pm_c
                pb_p_arr[p_off : p_off + cnt] = pp_c
    else:
        with mp.get_context("fork").Pool(w) as pool:
            for p_off, cnt, tr_c, tk_c, pm_c, pp_c in pool.imap(_sample_and_featurize_e1_chunk, jobs):
                truth_arr[p_off : p_off + cnt] = tr_c
                for vis in vis_list:
                    tok_by_vis[vis][p_off : p_off + cnt] = tk_c[vis]
                if pm_c is not None and pp_c is not None and pb_m_arr is not None and pb_p_arr is not None:
                    pb_m_arr[p_off : p_off + cnt] = pm_c
                    pb_p_arr[p_off : p_off + cnt] = pp_c

    return {
        "truth": truth_arr,
        "tok_by_vis": tok_by_vis,
        "pb_m": pb_m_arr,
        "pb_p": pb_p_arr,
    }


def compute_haar_o2_floor_none(dv: Batch, n_mc: int = 256, seed: int = 0) -> dict[str, float]:
    """Computes the information-theoretic Haar-O(2) marginalized floor on E1 when B_i orientation is hidden."""
    rng = np.random.default_rng(seed)
    means, precs = [], []
    for e in range(len(dv)):
        A = dv.oracle["A"][e]
        Bf = dv.oracle["Bf"][e]
        sig = dv.oracle["sig"][e]
        s = dv.obs[e]
        n_ag, m, d = A.shape
        angles = rng.uniform(0, 2 * np.pi, (n_mc, n_ag))
        flips = rng.choice([-1.0, 1.0], (n_mc, n_ag))
        c, sn = np.cos(angles), np.sin(angles)
        Q = np.stack([np.stack([c, -sn * flips], -1), np.stack([sn, c * flips], -1)], -2)
        Bf_draws = np.matmul(Bf[None], Q)
        Bf_draws[0] = Bf
        log_ev = np.zeros(n_mc)
        post_m = np.zeros((n_mc, d))
        post_cov = np.zeros((n_mc, d, d))
        K = A.reshape(n_ag * m, d)
        s_flat = s.ravel()
        sig_diag = np.diag(np.repeat(sig ** 2, m))
        for j in range(n_mc):
            Bk = Bf_draws[j].reshape(n_ag * m, -1)
            R = Bk @ Bk.T + sig_diag
            Sigma_s = K @ K.T + R
            sign, logdet = np.linalg.slogdet(Sigma_s)
            sol = np.linalg.solve(Sigma_s, s_flat)
            log_ev[j] = -0.5 * (logdet + s_flat @ sol)
            gm, gp = gaussian_global(A, Bf_draws[j], sig, s)
            post_m[j] = gm
            post_cov[j] = np.linalg.inv(gp)
        log_ev -= log_ev.max()
        w = np.exp(log_ev)
        w /= w.sum()
        m_mix = w @ post_m
        diff = post_m - m_mix[None]
        cov_mix = np.einsum("j,jik->ik", w, post_cov) + np.einsum("j,ji,jk->ik", w, diff, diff)
        means.append(m_mix)
        precs.append(np.linalg.inv(cov_mix))
    pred = {"mean": np.stack(means), "prec": np.stack(precs)}
    ms = summarize(pred, dv, n_boot=200)
    return {"regret": float(ms["regret"]["mean"]), "kl": float(ms["kl"]["mean"])}


def resolve_schedule(spec: str, layers: int) -> tuple[int, ...]:
    if spec == "first_last":
        return tuple(sorted({0, layers - 1}))
    return resolve_attn_layers(spec, layers)


def _spd_from_tril_raw(raw: torch.Tensor, d: int, min_diag: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """Maps (..., d*(d+1)//2) raw vector to SPD matrix P = L L^T and Cholesky factor L."""
    shape = raw.shape[:-1]
    L = torch.zeros(*shape, d, d, dtype=raw.dtype, device=raw.device)
    rows, cols = torch.tril_indices(d, d, device=raw.device)
    L[..., rows, cols] = raw
    diag = F.softplus(torch.diagonal(L, dim1=-2, dim2=-1) + 0.5413) + min_diag
    L = torch.tril(L, -1) + torch.diag_embed(diag)
    P = torch.matmul(L, L.transpose(-1, -2))
    return P, L


class E1TraderNet(nn.Module):
    """Unified E1 Aggregator Backbone supporting Moment, Pure-Info, Own-Sensor Info, and Latent-Factor GLS Info Heads."""

    def __init__(
        self,
        in_dim: int,
        d: int = 4,
        m_obs: int = 2,
        width: int = 128,
        layers: int = 4,
        heads: int = 4,
        attn_layers: tuple[int, ...] = (0, 1, 2, 3),
        token_state: str = "kv",
        head_mode: str = "own_info",
        causal: bool = True,
        dropout: float = 0.05,
        r_lat: int = 4,
        has_prov_key: bool | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.d = d
        self.m_obs = m_obs
        self.width = width
        self.layers = layers
        self.heads = heads
        self.attn_set = set(attn_layers)
        self.token_state = token_state
        self.head_mode = head_mode
        self.causal = causal
        self.r_lat = r_lat
        self.n_tril = d * (d + 1) // 2

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))

        # Row-equivariant per-observation encoder for E1 (m=2 slots per trader) + trader token projection
        # Note: at visibility='none', in_dim=34 (14 own sensor + 20 report mean/prec), so provenance key (13 dims) is absent!
        self.has_prov_key = (in_dim in (27, 47)) if has_prov_key is None else bool(has_prov_key)
        row_feat_dim = 1 + d + 2 + (d + 2 + 1 if self.has_prov_key else 0)
        self.row_enc = nn.Sequential(
            nn.Linear(row_feat_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.inp = nn.Linear(in_dim + width, width)

        self.ln1 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.ln2 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(
                nn.Linear(width, 4 * width),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(4 * width, width),
            )
            for _ in range(layers)
        )
        if token_state in ("kv", "feedback"):
            self.attn = nn.ModuleDict(
                {str(l): nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True) for l in self.attn_set}
            )
        if token_state == "recurrent":
            self.gru = nn.ModuleDict({str(l): nn.GRU(width, width, batch_first=True) for l in self.attn_set})
            self.read = nn.ModuleDict({str(l): nn.Linear(width, width) for l in self.attn_set})
            self.s0 = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(width)) for l in self.attn_set})
        if token_state == "feedback":
            self.mem_w = nn.Parameter(torch.zeros(layers + 1))

        self.norm = nn.LayerNorm(width)

        if head_mode == "moment":
            self.moment_head = nn.Linear(width, d + self.n_tril)
        elif head_mode == "pure_info":
            self.info_head = nn.Linear(width, d + self.n_tril)
        elif head_mode in ("own_info", "own_info_gls"):
            self.gate_head = nn.Linear(width, 2 * d + 2)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)
            self.corr_head = nn.Linear(width, d + 2 * d * d)
            nn.init.normal_(self.corr_head.weight, std=1e-3)
            nn.init.zeros_(self.corr_head.bias)
            if head_mode == "own_info_gls":
                self.gls_row_head = nn.Sequential(
                    nn.Linear(2 * width, width),
                    nn.GELU(),
                    nn.Linear(width, r_lat + 2),
                )
                nn.init.normal_(self.gls_row_head[-1].weight, std=1e-2)
                nn.init.zeros_(self.gls_row_head[-1].bias)
                self.gls_mix_logit = nn.Parameter(torch.tensor(0.0))
        else:
            raise ValueError(f"Unknown head_mode: {head_mode}")

    def _extract_row_features(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extracts s (B,N,2), A (B,N,2,4), R (B,N,2,2), and row-equivariant features phi (B,N,2,F_row)."""
        b, n_tr, _ = x_raw.shape
        s = x_raw[:, :, 0:2]
        A = x_raw[:, :, 2:10].reshape(b, n_tr, 2, 4)
        R = x_raw[:, :, 10:14].reshape(b, n_tr, 2, 2)
        r_diag = torch.stack([R[:, :, 0, 0], R[:, :, 1, 1]], dim=-1).unsqueeze(-1)
        r_off = torch.stack([R[:, :, 0, 1], R[:, :, 1, 0]], dim=-1).unsqueeze(-1)
        parts = [s.unsqueeze(-1), A, r_diag, r_off]
        if self.has_prov_key:
            A_k = x_raw[:, :, 14:22].reshape(b, n_tr, 2, 4)
            B_k = x_raw[:, :, 22:26].reshape(b, n_tr, 2, 2)
            lsig_k = x_raw[:, :, 26:27].unsqueeze(2).expand(b, n_tr, 2, 1)
            parts.extend([A_k, B_k, lsig_k])
        phi = torch.cat(parts, dim=-1)
        return s, A, R, phi

    def _encode_tokens(self, x_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s, A, R, phi = self._extract_row_features(x_raw)
        row_h = self.row_enc(phi)  # (B, N, 2, width), S_2 row-equivariant
        row_pool = row_h.mean(dim=2)  # (B, N, width), S_2 row-invariant
        x_norm = (x_raw - self.x_mean) / self.x_sd
        z = self.inp(torch.cat([x_norm, row_pool], dim=-1))

        b_sz, n_tr, w_dim = z.shape
        if self.token_state == "feedback":
            h_cnt = self.heads
            d_head = w_dim // h_cnt
            w = self.mem_w.softmax(0)
            k_cache: dict[int, list[torch.Tensor]] = {l: [] for l in self.attn_set}
            v_cache: dict[int, list[torch.Tensor]] = {l: [] for l in self.attn_set}
            outs: list[torch.Tensor] = []
            for t in range(n_tr):
                h = z[:, t : t + 1]
                states = [h]
                for l in range(self.layers):
                    if l in self.attn_set:
                        a = self.ln1[l](h)
                        mha = self.attn[str(l)]
                        qkv = F.linear(a, mha.in_proj_weight, mha.in_proj_bias)
                        q, k_a, v_a = qkv.chunk(3, dim=-1)
                        q = q.view(b_sz, 1, h_cnt, d_head).transpose(1, 2)
                        k_a = k_a.view(b_sz, 1, h_cnt, d_head).transpose(1, 2)
                        v_a = v_a.view(b_sz, 1, h_cnt, d_head).transpose(1, 2)
                        K = torch.cat(k_cache[l] + [k_a], dim=2)
                        V = torch.cat(v_cache[l] + [v_a], dim=2)
                        ao = F.scaled_dot_product_attention(q, K, V).transpose(1, 2).reshape(b_sz, 1, w_dim)
                        h = h + F.linear(ao, mha.out_proj.weight, mha.out_proj.bias)
                    h = h + self.mlp[l](self.ln2[l](h))
                    states.append(h)
                m_t = (torch.stack(states, dim=0) * w.view(-1, 1, 1, 1)).sum(0)
                if t + 1 < n_tr:
                    for l in self.attn_set:
                        mha = self.attn[str(l)]
                        kv_m = F.linear(m_t, mha.in_proj_weight[w_dim:], mha.in_proj_bias[w_dim:])
                        km, vm = kv_m.chunk(2, dim=-1)
                        k_cache[l].append(km.view(b_sz, 1, h_cnt, d_head).transpose(1, 2))
                        v_cache[l].append(vm.view(b_sz, 1, h_cnt, d_head).transpose(1, 2))
                outs.append(h)
            z = torch.cat(outs, 1)
        else:
            mask = (
                torch.triu(torch.ones(n_tr, n_tr, dtype=torch.bool, device=z.device), 1)
                if self.causal
                else None
            )
            for l in range(self.layers):
                if l in self.attn_set:
                    a = self.ln1[l](z)
                    if self.token_state == "kv":
                        z = z + self.attn[str(l)](a, a, a, attn_mask=mask, need_weights=False)[0]
                    else:
                        s0 = self.s0[str(l)]
                        s_init = s0.view(1, 1, w_dim).expand(1, b_sz, w_dim).contiguous()
                        if n_tr > 1:
                            gru_out, _ = self.gru[str(l)](a[:, :-1], s_init)
                            s_seq = torch.cat([s0.view(1, 1, w_dim).expand(b_sz, 1, w_dim), gru_out], dim=1)
                        else:
                            s_seq = s0.view(1, 1, w_dim).expand(b_sz, 1, w_dim)
                        z = z + self.read[str(l)](s_seq)
                z = z + self.mlp[l](self.ln2[l](z))
        z = self.norm(z)
        return z, row_h, s, A, R

    def forward_prefixes(self, x_raw: torch.Tensor, all_prefixes: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, prec) of shape (B, N, d), (B, N, d, d) if all_prefixes else (B, 1, d), (B, 1, d, d)."""
        z, row_h, s, A, R = self._encode_tokens(x_raw)
        b, n_tr, _ = z.shape
        d = self.d
        eye_d = torch.eye(d, dtype=z.dtype, device=z.device)

        if self.head_mode == "moment":
            z_use = z if all_prefixes else z[:, -1:]
            raw = self.moment_head(z_use)
            mean = raw[..., :d]
            prec, _ = _spd_from_tril_raw(raw[..., d:], d, min_diag=1e-2)
            return mean, prec

        if self.head_mode == "pure_info":
            raw = self.info_head(z)
            dh = raw[..., :d]
            dP, _ = _spd_from_tril_raw(raw[..., d:], d, min_diag=1e-3)
            if all_prefixes:
                P_cum = eye_d.view(1, 1, d, d) + torch.cumsum(dP, dim=1)
                h_cum = torch.cumsum(dh, dim=1)
            else:
                P_cum = (eye_d.unsqueeze(0) + dP.sum(dim=1)).unsqueeze(1)
                h_cum = dh.sum(dim=1, keepdim=True)
            mean = torch.linalg.solve(P_cum, h_cum.unsqueeze(-1)).squeeze(-1)
            return mean, P_cum

        # Compute exact own-sensor local natural parameters: dP_own = A_k^T R_k^-1 A_k, dh_own = A_k^T R_k^-1 s_k
        R_reg = R + 1e-5 * torch.eye(self.m_obs, dtype=R.dtype, device=R.device)
        Rinv_A = torch.linalg.solve(R_reg, A)
        Rinv_s = torch.linalg.solve(R_reg, s.unsqueeze(-1)).squeeze(-1)
        dP_own = torch.einsum("bnmi,bnmj->bnij", A, Rinv_A)
        dh_own = torch.einsum("bnmi,bnm->bni", A, Rinv_s)

        gates = self.gate_head(z)
        w_p = F.softplus(gates[..., :d] + 0.5413)
        w_h = F.softplus(gates[..., d : 2 * d] + 0.5413)
        s_p = F.softplus(gates[..., 2 * d : 2 * d + 1] + 0.5413).unsqueeze(-1)
        s_h = F.softplus(gates[..., 2 * d + 1 : 2 * d + 2] + 0.5413)

        w_p_sqrt = torch.sqrt(w_p.clamp_min(1e-4))
        dP_mod = s_p * (w_p_sqrt.unsqueeze(-1) * dP_own * w_p_sqrt.unsqueeze(-2))
        dh_mod = s_h * (w_h * dh_own)

        corr = self.corr_head(z)
        dh_corr = corr[..., :d]
        L_pos = corr[..., d : d + d * d].reshape(b, n_tr, d, d)
        L_neg = corr[..., d + d * d :].reshape(b, n_tr, d, d)
        dP_corr = torch.matmul(L_pos, L_pos.transpose(-1, -2)) - 0.5 * torch.matmul(L_neg, L_neg.transpose(-1, -2))

        t_indices = range(n_tr) if all_prefixes else [n_tr - 1]
        means_list: list[torch.Tensor] = []
        precs_list: list[torch.Tensor] = []

        if self.head_mode == "own_info_gls":
            z_exp = z.unsqueeze(2).expand_as(row_h)
            gls_raw = self.gls_row_head(torch.cat([row_h, z_exp], dim=-1))
            B_hat = gls_raw[..., : self.r_lat]
            sig2_hat = F.softplus(gls_raw[..., self.r_lat] - 2.0) + 1e-3
            gamma_r = torch.sigmoid(gls_raw[..., self.r_lat + 1] + 2.0).mean(dim=-1)
            w_gls = torch.sigmoid(self.gls_mix_logit)

        for t in t_indices:
            P_info = eye_d.unsqueeze(0) + dP_mod[:, : t + 1].sum(dim=1) + dP_corr[:, t]
            h_info = dh_mod[:, : t + 1].sum(dim=1) + dh_corr[:, t]

            if self.head_mode == "own_info_gls":
                k_obs = (t + 1) * self.m_obs
                A_t = A[:, : t + 1].reshape(b, k_obs, d)
                s_t = s[:, : t + 1].reshape(b, k_obs, 1)
                B_t = B_hat[:, : t + 1].reshape(b, k_obs, self.r_lat)
                sig2_t = sig2_hat[:, : t + 1].reshape(b, k_obs)
                Sigma = torch.matmul(B_t, B_t.transpose(-1, -2)) + torch.diag_embed(sig2_t)
                for k_idx in range(t + 1):
                    sl = slice(k_idx * self.m_obs, (k_idx + 1) * self.m_obs)
                    Sigma[:, sl, sl] = Sigma[:, sl, sl] + gamma_r[:, k_idx].view(b, 1, 1) * R[:, k_idx]
                Sinv_A = torch.linalg.solve(Sigma, A_t)
                Sinv_s = torch.linalg.solve(Sigma, s_t).squeeze(-1)
                P_gls = eye_d.unsqueeze(0) + torch.matmul(A_t.transpose(-1, -2), Sinv_A)
                h_gls = torch.einsum("bki,bk->bi", A_t, Sinv_s)
                P_raw = (1.0 - w_gls) * P_info + w_gls * P_gls + dP_corr[:, t]
                h_vec = (1.0 - w_gls) * h_info + w_gls * h_gls
            else:
                P_raw = P_info
                h_vec = h_info

            P_sym = 0.5 * (P_raw + P_raw.transpose(-1, -2))
            evals, evecs = torch.linalg.eigh(P_sym)
            evals_clamped = F.softplus(evals - 0.20) + 0.20
            P_spd = torch.matmul(evecs * evals_clamped.unsqueeze(-2), evecs.transpose(-1, -2))
            P_spd = 0.5 * (P_spd + P_spd.transpose(-1, -2))
            m_vec = torch.linalg.solve(P_spd, h_vec.unsqueeze(-1)).squeeze(-1)
            means_list.append(m_vec)
            precs_list.append(P_spd)

        return torch.stack(means_list, dim=1), torch.stack(precs_list, dim=1)


def gaussian_nll_torch(mean: torch.Tensor, prec: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes 0.5 * ((y - m)^T P (y - m) - log det P + d log(2pi)) averaged over batch/prefixes."""
    d = mean.shape[-1]
    diff = (target - mean).unsqueeze(-1)
    quad = torch.matmul(torch.matmul(diff.transpose(-1, -2), prec), diff).squeeze(-1).squeeze(-1)
    _, logdet = torch.linalg.slogdet(prec)
    return 0.5 * (quad - logdet + d * np.log(2.0 * np.pi))


def gaussian_kl_torch(m_p: torch.Tensor, P_p: torch.Tensor, m_q: torch.Tensor, P_q: torch.Tensor) -> torch.Tensor:
    """Computes KL(N(m_p, P_p^-1) || N(m_q, P_q^-1))."""
    d = m_p.shape[-1]
    P_p_inv_P_q = torch.linalg.solve(P_p, P_q)
    tr_term = torch.diagonal(P_p_inv_P_q, dim1=-2, dim2=-1).sum(-1)
    diff = (m_q - m_p).unsqueeze(-1)
    quad = torch.matmul(torch.matmul(diff.transpose(-1, -2), P_q), diff).squeeze(-1).squeeze(-1)
    _, logdet_p = torch.linalg.slogdet(P_p)
    _, logdet_q = torch.linalg.slogdet(P_q)
    return 0.5 * (tr_term + quad - d + logdet_p - logdet_q)


class E1Aggregator:
    def __init__(
        self,
        seed: int,
        in_dim: int,
        *,
        d: int = 4,
        width: int = 128,
        layers: int = 4,
        heads: int = 4,
        steps: int = 3000,
        batch: int = 512,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        dropout: float = 0.05,
        attn_layers: str = "all",
        token_state: str = "kv",
        head_mode: str = "own_info",
        loss: str = "final",
        causal: bool = True,
        reperm: bool = False,
        fresh: bool = True,
        drop_last_cols: int = 0,
        device: str = "cuda",
        has_prov_key: bool | None = None,
    ) -> None:
        torch.manual_seed(seed)
        self.seed = seed
        self.d = d
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss = loss
        self.reperm = reperm
        self.fresh = fresh
        self.drop_last_cols = drop_last_cols
        self.device = device
        self.attn_layers_resolved = resolve_schedule(attn_layers, layers)
        self.uses_oracle = loss == "distill"
        self.module = E1TraderNet(
            in_dim=in_dim,
            d=d,
            width=width,
            layers=layers,
            heads=heads,
            attn_layers=self.attn_layers_resolved,
            token_state=token_state,
            head_mode=head_mode,
            causal=causal,
            dropout=dropout,
            has_prov_key=has_prov_key,
        )
        self.log: list[dict[str, Any]] = []

    def _slice_cols(self, arr: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.drop_last_cols > 0:
            return arr[:, :, : -self.drop_last_cols]
        return arr

    def fit(self, ctx: dict[str, Any], truth: np.ndarray) -> "E1Aggregator":
        dev = self.device
        self.module.to(dev)
        x_np = ctx["tokens"]
        st_np = self._slice_cols(ctx["stats_tokens"])
        stats_x = torch.as_tensor(st_np, dtype=torch.float32, device=dev)
        flat = stats_x.reshape(-1, stats_x.shape[-1])
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))

        pb_m_np = ctx.get("prefix_bayes_mean")
        pb_p_np = ctx.get("prefix_bayes_prec")

        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps, pct_start=0.05)
        gen = np.random.default_rng(self.seed)
        self.module.train()
        ema = None
        n_tr = x_np.shape[1]
        all_pref = self.loss in ("every_prefix", "distill")

        for step in range(self.steps):
            if self.fresh and len(x_np) >= self.steps * self.batch:
                s = step * self.batch
                xb_np = self._slice_cols(x_np[s : s + self.batch])
                yb_np = truth[s : s + self.batch]
                xb = torch.as_tensor(xb_np, dtype=torch.float32, device=dev)
                yb = torch.as_tensor(yb_np, dtype=torch.float32, device=dev)
                pbm_b = None if pb_m_np is None else torch.as_tensor(pb_m_np[s : s + self.batch], dtype=torch.float32, device=dev)
                pbp_b = None if pb_p_np is None else torch.as_tensor(pb_p_np[s : s + self.batch], dtype=torch.float32, device=dev)
            else:
                idx = gen.integers(0, len(x_np), self.batch)
                xb = torch.as_tensor(self._slice_cols(x_np[idx]), dtype=torch.float32, device=dev)
                yb = torch.as_tensor(truth[idx], dtype=torch.float32, device=dev)
                pbm_b = None if pb_m_np is None else torch.as_tensor(pb_m_np[idx], dtype=torch.float32, device=dev)
                pbp_b = None if pb_p_np is None else torch.as_tensor(pb_p_np[idx], dtype=torch.float32, device=dev)

            if self.reperm and self.loss != "distill":
                perm = torch.argsort(torch.rand(len(xb), n_tr, device=dev), dim=1)
                xb = torch.gather(xb, 1, perm.unsqueeze(-1).expand_as(xb))
            mean_pred, prec_pred = self.module.forward_prefixes(xb, all_prefixes=all_pref)
            if self.loss == "final":
                loss = gaussian_nll_torch(mean_pred[:, 0], prec_pred[:, 0], yb).mean()
            elif self.loss == "every_prefix":
                loss = gaussian_nll_torch(mean_pred, prec_pred, yb.unsqueeze(1).expand_as(mean_pred)).mean()
            else:
                assert pbm_b is not None and pbp_b is not None
                loss = gaussian_kl_torch(pbm_b, pbp_b, mean_pred, prec_pred).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if (step + 1) % 500 == 0 or step + 1 == self.steps:
                self.log.append({"step": step + 1, "loss_ema": float(ema)})
        self.module.cpu()
        return self

    def predict_prefix(self, ctx: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        dev = self.device
        self.module.to(dev).eval()
        x = torch.as_tensor(self._slice_cols(ctx["tokens"]), dtype=torch.float32)
        ms, ps = [], []
        with torch.no_grad():
            for s in range(0, len(x), 1024):
                m_b, p_b = self.module.forward_prefixes(x[s : s + 1024].to(dev), all_prefixes=True)
                ms.append(m_b.double().cpu().numpy())
                ps.append(p_b.double().cpu().numpy())
        self.module.cpu()
        return np.concatenate(ms, 0), np.concatenate(ps, 0)

    def predict(self, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        dev = self.device
        self.module.to(dev).eval()
        x = torch.as_tensor(self._slice_cols(ctx["tokens"]), dtype=torch.float32)
        ms, ps = [], []
        with torch.no_grad():
            for s in range(0, len(x), 1024):
                m_b, p_b = self.module.forward_prefixes(x[s : s + 1024].to(dev), all_prefixes=False)
                ms.append(m_b[:, 0].double().cpu().numpy())
                ps.append(p_b[:, 0].double().cpu().numpy())
        self.module.cpu()
        return {"mean": np.concatenate(ms, 0), "prec": np.concatenate(ps, 0)}


class PrebuiltE1FCTStream(FreshStream):
    def __init__(self, tokens: np.ndarray, truth: np.ndarray, stats_tokens: np.ndarray, public: dict[str, Any], batch: int, steps: int) -> None:
        self.tokens = tokens
        self.truth = truth
        self.stats_tokens = stats_tokens
        self.public = dict(public)
        self.kind = "gaussian"
        self.d = int(public["d"])
        self.batch = batch
        self.steps = steps

    def __iter__(self):
        for s in range(self.steps):
            i = s * self.batch
            yield self.tokens[i : i + self.batch], self.truth[i : i + self.batch]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--visibilities", type=str, default="none,noisy,full")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--scaled-steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--fresh-episodes", action="store_true", default=True)
    ap.add_argument("--resume-log", type=str, default="")
    ap.add_argument("--commit", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    vis_list = [v.strip() for v in args.visibilities.split(",") if v.strip()]

    resume_cache: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    if args.resume_log and pathlib.Path(args.resume_log).is_file():
        import re as _re
        from aggbench.controls import random_weights, untrained_init

        pat = _re.compile(
            r"\[E1 vis=(none|noisy|full) seed=(\d+)\] ([A-Za-z0-9_]+): regret=([0-9.]+) kl=([0-9.]+)(?: mean_pref_kl=([0-9.]+))? \(([0-9.]+)s\)"
        )
        for v_m, s_m, a_m, reg_m, kl_m, pk_m, dt_m in pat.findall(pathlib.Path(args.resume_log).read_text()):
            resume_cache.setdefault((v_m, int(s_m)), {})[a_m] = {
                "regret": float(reg_m),
                "kl": float(kl_m),
                "mean_prefix_kl": float(pk_m) if pk_m else None,
                "train_seconds": float(dt_m),
            }
        print(f"  [E1] loaded {sum(len(d) for d in resume_cache.values())} cached arm runs from {args.resume_log}", flush=True)

    out_path = pathlib.Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    max_steps = max(args.steps, args.scaled_steps)
    n_train = max_steps * args.batch_size
    n_std = args.steps * args.batch_size
    print(
        f"[e1_infogls] device={device} visibilities={vis_list} seeds={seeds} steps={args.steps} "
        f"scaled_steps={args.scaled_steps} batch={args.batch_size} n_train={n_train}",
        flush=True,
    )

    env = make_env("E1")
    dv = env.sample("dev", args.n_eval)
    stats_b = env.sample("train", 4096, start=FRESH_STATS_START)
    tr_base = env.sample("train", 2000)
    base_preds, _ = all_baselines(tr_base, dv, "none")
    base_metrics = {b: summarize(p, dv, n_boot=500) for b, p in base_preds.items()}

    print("  [E1] computing Haar-O(2) marginalized floor_none on dev...", flush=True)
    haar_floor = compute_haar_o2_floor_none(dv, n_mc=256, seed=0)
    haar_floor["floor_none_full_marginal"] = 0.6854036554594531
    haar_floor["floor_none_gaussian_projection"] = 0.7084944406479896
    print(
        f"  [E1] floor_none: full={haar_floor['floor_none_full_marginal']:.4f} "
        f"gauss={haar_floor['floor_none_gaussian_projection']:.4f} rot_only={haar_floor['regret']:.4f}",
        flush=True,
    )

    ord_dv = sample_orders(dv.seeds, dv.obs.shape[1], 0)
    pb_dv_m, pb_dv_p = exact_e1_prefix_bayes(dv, ord_dv)
    pb_err_m = float(np.max(np.abs(pb_dv_m[:, -1] - dv.bayes["mean"])))
    pb_err_p = float(np.max(np.abs(pb_dv_p[:, -1] - dv.bayes["prec"])))
    assert max(pb_err_m, pb_err_p) < 1e-5, f"E1 prefix Bayes mismatch on dev: {pb_err_m}, {pb_err_p}"
    print(f"  [E1] verified exact_e1_prefix_bayes on dev: max_err_m={pb_err_m:.3e} max_err_p={pb_err_p:.3e}", flush=True)

    fct_dv_by_vis: dict[str, FullContext] = {}
    fct_stats_by_vis: dict[str, np.ndarray] = {}
    tok_dv_by_vis: dict[str, np.ndarray] = {}
    in_dims: dict[str, int] = {}
    for vis in vis_list:
        fct_dv_v = full_context(dv, vis, "unaligned", own_model=True)
        fct_dv_by_vis[vis] = fct_dv_v
        fct_stats_by_vis[vis] = np.ascontiguousarray(full_context(stats_b, vis, "unaligned", own_model=True).tokens, dtype=np.float32)
        tok_dv_by_vis[vis] = np.ascontiguousarray(fct_dv_v.tokens[np.arange(len(dv))[:, None], ord_dv], dtype=np.float32)
        in_dims[vis] = int(fct_dv_v.tokens.shape[-1])

    seed_records_by_vis: dict[str, dict[str, list[dict[str, Any]]]] = {vis: {} for vis in vis_list}
    seed0_methods_by_vis: dict[str, dict[str, Any]] = {vis: dict(base_metrics) for vis in vis_list}
    seed0_controls_by_vis: dict[str, dict[str, Any]] = {vis: {} for vis in vis_list}
    seed0_arms_meta_by_vis: dict[str, dict[str, Any]] = {vis: {} for vis in vis_list}
    learned_arms_by_vis: dict[str, list[str]] = {vis: [] for vis in vis_list}

    def _cached_metric_entry(reg_v: float, kl_v: float, n_ev: int) -> dict[str, Any]:
        return {
            "logloss": {"mean": reg_v, "ci_low": reg_v, "ci_high": reg_v, "n": n_ev},
            "brier": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": n_ev},
            "kl": {"mean": kl_v, "ci_low": kl_v, "ci_high": kl_v, "n": n_ev},
            "regret": {"mean": reg_v, "ci_low": reg_v, "ci_high": reg_v, "n": n_ev},
        }

    # Outer loop over seeds so we only hold ONE seed's float32 buffers (< 3.5 GB) in CPU RAM at a time!
    for s_idx, seed in enumerate(seeds):
        gc.collect()
        need_vis = [vis for vis in vis_list if len(resume_cache.get((vis, seed), {})) < 22]
        if need_vis:
            t_s0 = time.time()
            need_tpb = any("tt_own_info_distill" not in resume_cache.get((v, seed), {}) for v in need_vis)
            need_s6k = any(
                ("tt_own_info_all_scaled6k" not in resume_cache.get((v, seed), {}))
                or ("tt_own_info_gls_all_scaled6k" not in resume_cache.get((v, seed), {}))
                for v in need_vis
            )
            n_sample = n_train if need_s6k else n_std
            seed_data = sample_e1_seed_featurized(
                seed, n_sample, n_std, need_vis, in_dims, n_tr=dv.obs.shape[1], d=4, need_train_pb=need_tpb, start_stride=n_train
            )
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(
                f"  [E1] streamed & featurized seed={seed} vis={need_vis} n_train={n_sample} (start={seed * n_train}) "
                f"in {time.time() - t_s0:.2f}s (maxRSS={rss_gb:.2f}GB)",
                flush=True,
            )
            truth_tr = seed_data["truth"]
            pb_tr_m = seed_data["pb_m"]
            pb_tr_p = seed_data["pb_p"]
        else:
            seed_data = {"tok_by_vis": {}}
            truth_tr = pb_tr_m = pb_tr_p = None
            print(f"  [E1] seed={seed} 100% cached across {vis_list} — skipping train stream!", flush=True)

        for vis in vis_list:
            tok_tr = seed_data["tok_by_vis"].get(vis)
            fct_stats = fct_stats_by_vis[vis]
            fct_dv = fct_dv_by_vis[vis]
            tok_dv = tok_dv_by_vis[vis]
            vis_in_dim = in_dims[vis]
            cached_vis_seed = resume_cache.get((vis, seed), {})

            seed_records = seed_records_by_vis[vis]
            seed0_methods = seed0_methods_by_vis[vis]
            seed0_controls = seed0_controls_by_vis[vis]
            seed0_arms_meta = seed0_arms_meta_by_vis[vis]
            learned_arms = learned_arms_by_vis[vis]

            # 1. Baseline Moment-Form FullContextTransformer bars
            fct_bars = [
                ("bar_fct_fresh_headline_w64_l3", 64, 3, args.steps),
                ("bar_fct_moment_w128_l4", args.width, args.layers, args.steps),
            ]
            for bname, bw, bl, bsteps in fct_bars:
                made_fct: list[FullContextTransformer] = []

                def make_fct(s: int, _in_dim: int = vis_in_dim, _w: int = bw, _l: int = bl, _st: int = bsteps) -> FullContextTransformer:
                    m = FullContextTransformer(
                        s,
                        _in_dim,
                        "gaussian",
                        d=4,
                        width=_w,
                        layers=_l,
                        heads=4,
                        steps=_st,
                        batch=args.batch_size,
                        lr=1e-3,
                        weight_decay=1e-2,
                        dropout=0.1,
                        device=device,
                    )
                    made_fct.append(m)
                    return m

                if bname in cached_vis_seed:
                    rec = cached_vis_seed[bname]
                    seed_records.setdefault(bname, []).append(rec)
                    if s_idx == 0:
                        m0 = make_fct(seed)
                        learned_arms.append(bname)
                        seed0_methods[bname] = _cached_metric_entry(rec["regret"], rec["kl"], len(dv))
                        u_sum = summarize(m0.predict(fct_dv), dv, n_boot=100)
                        r_sum = summarize(random_weights(m0, seed + 1).predict(fct_dv), dv, n_boot=100)
                        seed0_methods[f"{bname}:untrained_init"] = u_sum
                        seed0_methods[f"{bname}:random_network"] = r_sum
                        seed0_controls[bname] = {"random_network": f"{bname}:random_network", "untrained_init": f"{bname}:untrained_init"}
                        seed0_arms_meta[bname] = {
                            "in_dim": vis_in_dim,
                            "n_params": int(sum(p.numel() for p in m0.module.parameters())),
                            "train_seconds": round(rec["train_seconds"], 2),
                            "head_mode": "moment",
                            "uses_oracle": False,
                        }
                    continue

                assert tok_tr is not None and truth_tr is not None
                t0 = time.time()
                n_b = bsteps * args.batch_size
                stream = PrebuiltE1FCTStream(tok_tr[:n_b], truth_tr[:n_b], fct_stats, dv.public, args.batch_size, bsteps)
                arms_out = run_with_controls(make_fct, stream, truth_tr[:n_b], fct_dv, seed=seed)
                dt = time.time() - t0
                m_sum = summarize(arms_out["trained"], dv, n_boot=500)
                reg_val = float(m_sum["regret"]["mean"])
                kl_val = float(m_sum["kl"]["mean"])
                print(f"  [E1 vis={vis} seed={seed}] {bname}: regret={reg_val:.4f} kl={kl_val:.4f} ({dt:.2f}s)", flush=True)
                seed_records.setdefault(bname, []).append(
                    {"regret": reg_val, "kl": kl_val, "mean_prefix_kl": None, "train_seconds": dt}
                )
                if s_idx == 0:
                    learned_arms.append(bname)
                    for k_arm, f_pred in arms_out.items():
                        key = bname if k_arm == "trained" else f"{bname}:{k_arm.split(':')[1]}"
                        seed0_methods[key] = summarize(f_pred, dv, n_boot=500)
                    seed0_controls[bname] = {"random_network": f"{bname}:random_network", "untrained_init": f"{bname}:untrained_init"}
                    seed0_arms_meta[bname] = {
                        "in_dim": vis_in_dim,
                        "n_params": int(sum(p.numel() for p in made_fct[0].module.parameters())),
                        "train_seconds": round(dt, 2),
                        "head_mode": "moment",
                        "uses_oracle": False,
                    }

            # 2. Head & Architecture Ladder on E1
            # Tuple: (name, is_scaled, w, l, attn, state, head_mode, loss_mode, causal, reperm, steps, drop_cols)
            e1_arms = [
                ("tt_moment_all", False, args.width, args.layers, "all", "kv", "moment", "final", True, False, args.steps, 0),
                ("tt_pure_info_all", False, args.width, args.layers, "all", "kv", "pure_info", "final", True, False, args.steps, 0),
                ("fct_pure_info_bidir", False, args.width, args.layers, "all", "kv", "pure_info", "final", False, False, args.steps, 0),
                ("tt_own_info_all_w64_l3", False, 64, 3, "all", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_all", False, args.width, args.layers, "all", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_all_no_report", False, args.width, args.layers, "all", "kv", "own_info", "final", True, False, args.steps, 20),
                ("fct_own_info_bidir", False, args.width, args.layers, "all", "kv", "own_info", "final", False, False, args.steps, 0),
                ("tt_own_info_last", False, args.width, args.layers, "last", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_first", False, args.width, args.layers, "first", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_every_other", False, args.width, args.layers, "every_other", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_none", False, args.width, args.layers, "none", "kv", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_every_prefix", False, args.width, args.layers, "all", "kv", "own_info", "every_prefix", True, False, args.steps, 0),
                ("tt_own_info_distill", False, args.width, args.layers, "all", "kv", "own_info", "distill", True, False, args.steps, 0),
                ("tt_own_info_recurrent", False, args.width, args.layers, "all", "recurrent", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_feedback", False, args.width, args.layers, "all", "feedback", "own_info", "final", True, False, args.steps, 0),
                ("tt_own_info_gls_none", False, args.width, args.layers, "none", "kv", "own_info_gls", "final", True, False, args.steps, 0),
                ("tt_own_info_gls_all", False, args.width, args.layers, "all", "kv", "own_info_gls", "final", True, False, args.steps, 0),
                ("fct_own_info_gls_bidir", False, args.width, args.layers, "all", "kv", "own_info_gls", "final", False, False, args.steps, 0),
                ("tt_own_info_all_scaled6k", True, args.width, args.layers, "all", "kv", "own_info", "final", True, False, args.scaled_steps, 0),
                ("tt_own_info_gls_all_scaled6k", True, args.width, args.layers, "all", "kv", "own_info_gls", "final", True, False, args.scaled_steps, 0),
            ]

            for aname, is_scaled, aw, al, a_sched, t_state, h_mode, l_mode, is_causal, rep_flag, a_steps, drop_c in e1_arms:
                eff_in_dim = int(vis_in_dim - drop_c)
                ctx_dv = {
                    "tokens": tok_dv,
                    "stats_tokens": fct_stats,
                    "public": dict(dv.public),
                }
                made_e1: list[E1Aggregator] = []

                def make_e1(
                    s: int,
                    _in_dim: int = eff_in_dim,
                    _w: int = aw,
                    _l: int = al,
                    _as: str = a_sched,
                    _ts: str = t_state,
                    _hm: str = h_mode,
                    _lm: str = l_mode,
                    _ca: bool = is_causal,
                    _rf: bool = rep_flag,
                    _st: int = a_steps,
                    _dc: int = drop_c,
                    _hpk: bool = (vis != "none"),
                ) -> E1Aggregator:
                    m = E1Aggregator(
                        s,
                        _in_dim,
                        d=4,
                        width=_w,
                        layers=_l,
                        heads=4,
                        steps=_st,
                        batch=args.batch_size,
                        lr=1e-3,
                        weight_decay=1e-2,
                        dropout=0.05,
                        attn_layers=_as,
                        token_state=_ts,
                        head_mode=_hm,
                        loss=_lm,
                        causal=_ca,
                        reperm=_rf,
                        fresh=bool(args.fresh_episodes),
                        drop_last_cols=_dc,
                        device=device,
                        has_prov_key=_hpk,
                    )
                    made_e1.append(m)
                    return m

                if aname in cached_vis_seed:
                    rec = cached_vis_seed[aname]
                    seed_records.setdefault(aname, []).append(rec)
                    if s_idx == 0:
                        m0 = make_e1(seed)
                        learned_arms.append(aname)
                        seed0_methods[aname] = _cached_metric_entry(rec["regret"], rec["kl"], len(dv))
                        u_sum = summarize(m0.predict(ctx_dv), dv, n_boot=100)
                        r_sum = summarize(random_weights(m0, seed + 1).predict(ctx_dv), dv, n_boot=100)
                        seed0_methods[f"{aname}:untrained_init"] = u_sum
                        seed0_methods[f"{aname}:random_network"] = r_sum
                        seed0_controls[aname] = {"random_network": f"{aname}:random_network", "untrained_init": f"{aname}:untrained_init"}
                        seed0_arms_meta[aname] = {
                            "in_dim": eff_in_dim,
                            "n_params": int(sum(p.numel() for p in m0.module.parameters())),
                            "train_seconds": round(rec["train_seconds"], 2),
                            "head_mode": h_mode,
                            "causal": is_causal,
                            "steps": a_steps,
                            "attn_layers_resolved": list(m0.attn_layers_resolved),
                            "uses_oracle": bool(m0.uses_oracle),
                            "mean_prefix_kl": rec["mean_prefix_kl"],
                        }
                    continue

                assert tok_tr is not None and truth_tr is not None
                t0 = time.time()
                tr_toks = tok_tr if is_scaled else tok_tr[:n_std]
                ctx_tr = {
                    "tokens": tr_toks,
                    "stats_tokens": fct_stats,
                    "public": dict(dv.public),
                    "prefix_bayes_mean": pb_tr_m if l_mode == "distill" else None,
                    "prefix_bayes_prec": pb_tr_p if l_mode == "distill" else None,
                }

                arms_out = run_with_controls(make_e1, ctx_tr, truth_tr[: len(tr_toks)], ctx_dv, seed=seed)
                dt = time.time() - t0
                m_sum = summarize(arms_out["trained"], dv, n_boot=500)
                reg_val = float(m_sum["regret"]["mean"])
                kl_val = float(m_sum["kl"]["mean"])
                pred_m, pred_p = made_e1[0].predict_prefix(ctx_dv)
                pref_kls = [
                    float(
                        forecast_kl(
                            {"mean": pb_dv_m[:, t], "prec": pb_dv_p[:, t]},
                            {"mean": pred_m[:, t], "prec": pred_p[:, t]},
                            "gaussian",
                        ).mean()
                    )
                    for t in range(pred_m.shape[1])
                ]
                mean_pk = float(np.mean(pref_kls))
                print(
                    f"  [E1 vis={vis} seed={seed}] {aname}: regret={reg_val:.4f} kl={kl_val:.4f} mean_pref_kl={mean_pk:.4f} ({dt:.2f}s)",
                    flush=True,
                )
                seed_records.setdefault(aname, []).append(
                    {"regret": reg_val, "kl": kl_val, "mean_prefix_kl": mean_pk, "train_seconds": dt}
                )
                if s_idx == 0:
                    learned_arms.append(aname)
                    for k_arm, f_pred in arms_out.items():
                        key = aname if k_arm == "trained" else f"{aname}:{k_arm.split(':')[1]}"
                        seed0_methods[key] = summarize(f_pred, dv, n_boot=500)
                    seed0_controls[aname] = {"random_network": f"{aname}:random_network", "untrained_init": f"{aname}:untrained_init"}
                    seed0_arms_meta[aname] = {
                        "in_dim": eff_in_dim,
                        "n_params": int(sum(p.numel() for p in made_e1[0].module.parameters())),
                        "train_seconds": round(dt, 2),
                        "head_mode": h_mode,
                        "causal": is_causal,
                        "steps": a_steps,
                        "attn_layers_resolved": list(made_e1[0].attn_layers_resolved),
                        "uses_oracle": bool(made_e1[0].uses_oracle),
                        "prefix_kl": pref_kls,
                        "mean_prefix_kl": mean_pk,
                    }

        del seed_data, truth_tr, pb_tr_m, pb_tr_p
        gc.collect()

    summary: dict[str, Any] = {
        "suite": "e1_infogls",
        "git_commit": args.commit,
        "fresh_episodes": bool(args.fresh_episodes),
        "n_train_per_seed": n_train,
        "n_eval": args.n_eval,
        "steps": args.steps,
        "scaled_steps": args.scaled_steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "layers": args.layers,
        "seeds": seeds,
        "haar_o2_floor_none": haar_floor,
        "baselines": {
            b: {
                "regret": float(base_metrics[b]["regret"]["mean"]),
                "kl": float(base_metrics[b]["kl"]["mean"]),
            }
            for b in base_metrics
        },
        "visibilities": {},
    }

    for vis in vis_list:
        seed_records = seed_records_by_vis[vis]
        seed0_methods = seed0_methods_by_vis[vis]
        seed0_controls = seed0_controls_by_vis[vis]
        seed0_arms_meta = seed0_arms_meta_by_vis[vis]
        learned_arms = learned_arms_by_vis[vis]

        models_summary: dict[str, Any] = {}
        for arm_name, recs in seed_records.items():
            r_arr = [r["regret"] for r in recs]
            k_arr = [r["kl"] for r in recs]
            p_arr = [r["mean_prefix_kl"] for r in recs if r["mean_prefix_kl"] is not None]
            t_arr = [r["train_seconds"] for r in recs]
            meta = seed0_arms_meta.get(arm_name, {})
            models_summary[arm_name] = {
                "n_seeds": len(recs),
                "regret_mean": float(np.mean(r_arr)),
                "regret_sd": float(np.std(r_arr)),
                "regret_per_seed": r_arr,
                "kl_mean": float(np.mean(k_arr)),
                "kl_sd": float(np.std(k_arr)),
                "kl_per_seed": k_arr,
                "mean_prefix_kl_mean": float(np.mean(p_arr)) if p_arr else None,
                "mean_prefix_kl_sd": float(np.std(p_arr)) if p_arr else None,
                "mean_prefix_kl_per_seed": p_arr if p_arr else None,
                "train_seconds_mean": float(np.mean(t_arr)),
                "n_params": meta.get("n_params"),
                "in_dim": meta.get("in_dim"),
                "head_mode": meta.get("head_mode"),
                "causal": meta.get("causal", True),
                "steps": meta.get("steps", args.steps),
                "attn_layers_resolved": meta.get("attn_layers_resolved"),
                "uses_oracle": meta.get("uses_oracle", False),
            }

        summary["visibilities"][vis] = {
            "prefix_bayes_final_maxabs_err": max(pb_err_m, pb_err_p),
            "models": models_summary,
        }

        vis_res_dir = out_dir / f"tt_e1_{vis}"
        vis_res_dir.mkdir(parents=True, exist_ok=True)
        res_obj = make_result(
            run_id=f"tt_e1_{vis}",
            track="explore",
            env=env,
            visibility=vis,
            splits={
                "train": {"split": "train", "n": n_train, "seed_min": 0, "seed_max": n_train - 1, "sha256": "fresh"},
                "eval": split_record(dv, "dev"),
            },
            oracle_check=oracle_check(env, n=20),
            methods=seed0_methods,
            learned_arms=learned_arms,
            controls=seed0_controls,
            extra={
                "alignment": "unaligned",
                "haar_o2_floor_none": haar_floor,
                "prefix_bayes_final_maxabs_err": max(pb_err_m, pb_err_p),
                "arms_meta": seed0_arms_meta,
                "multi_seed_summary": models_summary,
            },
        )
        if args.commit:
            res_obj["harness"]["git_commit"] = args.commit
            res_obj["harness"]["git_dirty"] = False
        write_result(vis_res_dir, res_obj, allow_dirty=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[e1_infogls] finished -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
