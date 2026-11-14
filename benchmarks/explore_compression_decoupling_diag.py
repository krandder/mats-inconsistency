#!/usr/bin/env python3
"""Decoupling Diagnostic for Learned Compression (#3887: Z9-p2 & E2 at b=2, Label-Only, 3 Seeds).

Isolates encoder vs. decoder vs. structural-information bottlenecks between closed-form
compression (`raw_code` on `Z9-p2`: 0.0000; `llr_quantizer` on `E2`: 0.0331) and learned
compression at b = 2 bits/agent, label-only (Y labels only, 3 seeds 0, 1, 2):

  (i)   Encoder FIXED to the closed-form b=2 code (`raw_code` on Z9-p2, `llr_quantizer` on E2),
        train ONLY the decoder:
        - `i_enc_cf_dec_learned_msg_only`: standard set decoder reading ONLY the 2-bit messages `m`
          (bounded by the message-only Bayes floor: 0.3225 on Z9-p2, 0.0555 on E2).
        - `i_enc_cf_dec_learned_with_keys`: set decoder reading `[m, key_structure]`, receiving the
          same per-episode structure keys (`key.full` on Z9-p2; `own_model` on E2) that the
          closed-form exact decoder receives.
  (ii)  Decoder FIXED to the exact Bayes decoder for that code family (`tabular_message_decoder`
        on Z9-p2; `gaussian_cell_decoder` on E2), train ONLY the encoder on Y labels:
        - `ii_enc_learned_dec_fixed_bayes`: encoder reads full agent token `x`.
        - `ii_enc_learned_obs_only_dec_fixed_bayes`: encoder reads only private observation/stat
          (`obs` on Z9-p2; `[local_logit, scale]` on E2) since the fixed Bayes decoder already
          has the structure keys.
  (iii) Both encoder and decoder learned end-to-end on Y labels:
        - `iii_both_learned_msg_only`: current `aware` arm from `aggbench/compression.py`
          (message-only set decoder `decode(m)`).
        - `iii_both_learned_with_keys`: encoder sends explicit b=2-bit message `m` (verified by
          `Channel(2).send`), and the learned set decoder receives `[m, key_structure]`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import re
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.compression import (
    CLIP,
    ESCAPE_FRAC,
    GAUSS_SIGMA2,
    Channel,
    _chol,
    _kl_logits,
    _qmc,
    agent_statistic,
    agent_tokens,
    companded_thresholds,
    gain,
    gaussian_report_model,
    gumbel_code,
    llr_quantizer_forecast,
    make_cenv,
    raw_code_forecast,
    tune_llr_quantizer,
)
from aggbench.metrics import bernoulli_logloss, summarize, summarize_values
from aggbench.parity_learn import NV, Z9Spec
from aggbench.report import achievable_floor, prior_forecast
from aggbench.results import harness_info


def sample_z9_fresh_batch(
    spec: Z9Spec,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draws `n` fresh `Z9-p2` episodes on `device`: returns `(X_full_17d, K_14d, Y, sym_cf)`."""
    R = spec.n_readers
    k = spec.k
    n_par = k * spec.readers_per_bit
    par = torch.arange(k, device=device).repeat_interleave(spec.readers_per_bit).unsqueeze(0).expand(n, n_par)
    dist = torch.randint(4, 6, (n, spec.n_dist_agents), generator=gen, device=device)
    var = torch.cat([par, dist], dim=1)
    perm = torch.rand((n, R), generator=gen, device=device).argsort(dim=1)
    var = torch.gather(var, 1, perm)
    acc = spec.acc[0] + (spec.acc[1] - spec.acc[0]) * torch.rand((n, R), generator=gen, device=device)
    bits = (torch.rand((n, NV), generator=gen, device=device) < spec.bit_prob).long()
    truth = torch.gather(bits, 1, var)
    read = torch.where(torch.rand((n, R), generator=gen, device=device) < acc, truth, 1 - truth)
    y = (bits[:, :k].sum(dim=1) % 2).float()
    N = R // spec.p
    onehot = F.one_hot(var, num_classes=NV).float()
    keys = torch.cat([onehot, acc.unsqueeze(-1)], dim=-1).reshape(n, N, spec.p * (NV + 1))
    obs = read.float().reshape(n, N, spec.p)
    zero_loc = torch.zeros((n, N, 1), dtype=torch.float32, device=device)
    tok = torch.cat([obs, keys, zero_loc], dim=-1)
    sym_cf = read[:, 0::2] * 2 + read[:, 1::2]
    return tok, keys, y, sym_cf.long()


def compute_z9_msg_only_bayes_floor(env: Any, tr: Any, dv: Any, b: int = 2) -> dict[str, Any]:
    """Exact Bayes decoder given ONLY the b-bit raw_code messages (without oracle['var']).

    Because episodes permute variable assignments randomly and agents are exchangeable when
    `oracle['var']` is withheld from the decoder, the distribution of `sym = (m_1, ..., m_7)`
    depends only on the empirical histogram of the 4 symbols `{0, 1, 2, 3}` (plus exact Laplace-
    smoothed joint contingency table lookup from 200,000 train episodes). We also verify analytically
    by averaging `P(m | w, var)` over random variable permutations `var`: for every single episode,
    because each agent reads `p = 2` of the `k = 4` parity bits (or distractors), any marginal over
    `< 4` parity bits is uniform regardless of `Y`, and without `var` the probability that a random
    subset of readings covers all 4 parity bits an odd number of times is small, bounding the
    message-only Bayes decoder at ~0.3225 regret.
    """
    sym_tr = tr.oracle["sym"]  # (N_tr, 7) in {0,1,2,3}
    y_tr = np.asarray(tr.truth, dtype=np.float64)
    sym_dv = dv.oracle["sym"]  # (N_dv, 7)

    # Sorted 7-tuple of symbols in {0,1,2,3} is a sufficient statistic under agent exchangeability
    # (there are C(7+4-1, 4-1) = 120 distinct multisets of size 7 over 4 symbols!).
    s_tr = np.sort(sym_tr, axis=1)
    s_dv = np.sort(sym_dv, axis=1)
    powers = (4 ** np.arange(7, dtype=np.int64))
    code_tr = (s_tr * powers[None, :]).sum(axis=1)
    code_dv = (s_dv * powers[None, :]).sum(axis=1)

    l0 = float(dv.public["prior_logit"])
    p0_prior = 1.0 / (1.0 + math.exp(-l0))
    logits_dv = np.full(len(dv), l0, dtype=np.float64)
    uniq = np.unique(code_dv)
    for u in uniq:
        mask_tr = (code_tr == u)
        n1 = float(y_tr[mask_tr].sum()) + 1.0 * p0_prior
        n0 = float((1.0 - y_tr[mask_tr]).sum()) + 1.0 * (1.0 - p0_prior)
        logits_dv[code_dv == u] = math.log(n1 / n0)
    return {"logit": np.clip(logits_dv, -CLIP, CLIP)}


def compute_e2_msg_only_bayes_floor(env: Any, tr: Any, dv: Any, quant: Any) -> dict[str, Any]:
    """Plug-in tabulated message-only floor on E2 given ONLY the 2-bit llr_quantizer symbols `(m_1, ..., m_5)`
    without per-episode `(cov, mu)` at the decoder.
    """
    r_tr, sc_tr = agent_statistic(tr)
    sym_tr = quant.encode(r_tr, sc_tr, float(tr.public["prior_logit"])).astype(np.int64)
    r_dv, sc_dv = agent_statistic(dv)
    sym_dv = quant.encode(r_dv, sc_dv, float(dv.public["prior_logit"])).astype(np.int64)
    y_tr = np.asarray(tr.truth, dtype=np.float64)

    # Agents in E2 are exchangeable when their per-episode cov/mu keys are withheld;
    # sorted 5-tuple over 4 symbols has C(5+4-1, 3) = 56 distinct multisets.
    s_tr = np.sort(sym_tr, axis=1)
    s_dv = np.sort(sym_dv, axis=1)
    powers = (4 ** np.arange(s_tr.shape[1], dtype=np.int64))
    code_tr = (s_tr * powers[None, :]).sum(axis=1)
    code_dv = (s_dv * powers[None, :]).sum(axis=1)

    l0 = float(dv.public["prior_logit"])
    p0_prior = 1.0 / (1.0 + math.exp(-l0))
    logits_dv = np.full(len(dv), l0, dtype=np.float64)
    for u in np.unique(code_dv):
        mask_tr = (code_tr == u)
        n1 = float(y_tr[mask_tr].sum()) + 1.0 * p0_prior
        n0 = float((1.0 - y_tr[mask_tr]).sum()) + 1.0 * (1.0 - p0_prior)
        logits_dv[code_dv == u] = math.log(n1 / n0)
    return {"logit": np.clip(logits_dv, -CLIP, CLIP)}


def make_z9_lik_tensor(batch: Any, device: torch.device) -> torch.Tensor:
    """Vectorized exact per-episode likelihood table `(B, 7, 64, 4)` for Z9-p2 matching `_tabular_tables`."""
    w_idx = np.arange(64)[:, None]
    w_bits = (w_idx >> np.arange(6)[None, :]) & 1  # (64, 6)
    var_all = batch.oracle["var"].reshape(-1, 7, 2)  # (B, 7, 2)
    acc_all = batch.oracle["acc"].reshape(-1, 7, 2)  # (B, 7, 2)
    w_read = np.transpose(w_bits[:, var_all], (1, 2, 0, 3))  # (B, 7, 64, 2)
    acc_b = acc_all[:, :, None, :]  # (B, 7, 1, 2)
    p_o1 = np.where(w_read == 1, acc_b, 1.0 - acc_b)
    p_o0 = 1.0 - p_o1
    lik = np.stack(
        [
            p_o0[..., 0] * p_o0[..., 1],
            p_o0[..., 0] * p_o1[..., 1],
            p_o1[..., 0] * p_o0[..., 1],
            p_o1[..., 0] * p_o1[..., 1],
        ],
        axis=-1,
    ).astype(np.float32)
    return torch.tensor(lik, dtype=torch.float32, device=device)


def make_z9_target_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    w_bits = (np.arange(64)[:, None] >> np.arange(6)[None, :]) & 1
    target_w = (w_bits[:, :4].sum(-1) % 2).astype(bool)
    return (
        torch.tensor(np.where(target_w)[0], dtype=torch.long, device=device),
        torch.tensor(np.where(~target_w)[0], dtype=torch.long, device=device),
    )


def decode_fixed_bayes_z9(
    p_msg: torch.Tensor,
    lik: torch.Tensor,
    idx_y1: torch.Tensor,
    idx_y0: torch.Tensor,
) -> torch.Tensor:
    """Exact Bayes log-odds on Z9-p2 given per-agent symbol probabilities `p_msg` `(B, 7, 4)` and `lik` `(B, 7, 64, 4)`."""
    pm_w = (lik * p_msg[:, :, None, :]).sum(dim=-1).clamp_min(1e-12)  # (B, 7, 64)
    log_p_w = pm_w.log().sum(dim=1)  # (B, 64)
    lp1 = torch.logsumexp(log_p_w.index_select(1, idx_y1), dim=-1)
    lp0 = torch.logsumexp(log_p_w.index_select(1, idx_y0), dim=-1)
    return (lp1 - lp0).clamp(-CLIP, CLIP)


def torch_box_logprob(
    mean_t: torch.Tensor,
    L_t: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """PyTorch Genz separation-of-variables box log-probability matching `aggbench.compression.box_logprob`."""
    n, N = mean_t.shape
    M = u.shape[0]
    z = torch.zeros((n, M, N), dtype=mean_t.dtype, device=mean_t.device)
    logp = torch.zeros((n, M), dtype=mean_t.dtype, device=mean_t.device)
    for i in range(N):
        if i == 0:
            shift = mean_t[:, 0:1].expand(n, M)
        else:
            shift = mean_t[:, i : i + 1] + torch.einsum("nk,nmk->nm", L_t[:, i, :i], z[:, :, :i])
        d = L_t[:, i, i : i + 1]
        det = d <= 1e-12
        dd = torch.where(det, torch.ones_like(d), d)
        a = (lo[:, i : i + 1] - shift) / dd
        b = (hi[:, i : i + 1] - shift) / dd
        flip = (a + b) > 0
        a2 = torch.where(flip, -b, a)
        b2 = torch.where(flip, -a, b)
        la = torch.special.log_ndtr(a2)
        lb = torch.special.log_ndtr(b2)
        r = torch.exp((la - lb).clamp(max=0.0)).clamp(max=1.0 - 1e-7)
        lm = lb + torch.log1p(-r)
        p_a = torch.special.ndtr(a2)
        p_b = torch.special.ndtr(b2)
        p_u = (p_a + u[None, :, i] * (p_b - p_a)).clamp(1e-7, 1.0 - 1e-7)
        zz = torch.special.ndtri(p_u)
        zz = torch.where(flip, -zz, zz)
        zz = torch.minimum(torch.maximum(zz, torch.minimum(a, b)), torch.maximum(a, b))
        logp = logp + torch.where(det, torch.zeros_like(lm), lm)
        z[:, :, i] = torch.where(det, torch.zeros_like(zz), zz)
    mx = logp.max(dim=1, keepdim=True).values
    return torch.log(torch.exp(logp - mx).mean(dim=1)) + mx[:, 0]


def make_e2_exact_tensors(batch: Any, quant: Any, device: torch.device) -> dict[str, torch.Tensor]:
    """Precomputes exact Gaussian report model and cell bounds/centroids for E2 at b=2 on `device`."""
    r, scale = agent_statistic(batch)
    l0 = float(batch.public["prior_logit"])
    c, h, S = gaussian_report_model(batch)
    L = _chol(S)
    th = quant.thresholds(scale, l0)  # (n, 5, 3)
    PAD = 30.0
    pad_th = np.concatenate(
        [np.full((len(batch), 5, 1), -PAD), th, np.full((len(batch), 5, 1), PAD)],
        axis=-1,
    ).astype(np.float32)

    t_std = np.concatenate([[-np.inf], companded_thresholds(2, quant.s), [np.inf]])
    centroids_std = np.array(
        [(norm.pdf(t_std[j]) - norm.pdf(t_std[j + 1])) / (norm.cdf(t_std[j + 1]) - norm.cdf(t_std[j])) for j in range(4)],
        dtype=np.float32,
    )
    var_rem = float(1.0 - np.mean(centroids_std ** 2))
    sc = scale if quant.family == "own" else np.full_like(scale, quant.pooled)
    cell_vals = (l0 + sc[..., None] * centroids_std[None, None, :]).astype(np.float32)  # (n, 5, 4)
    S_q = S + np.eye(5)[None, :, :] * (var_rem * (sc ** 2))[:, :, None]
    w_lin = (2.0 * np.einsum("nij,nj->ni", np.linalg.pinv(S_q, rcond=1e-10, hermitian=True), h)).astype(np.float32)

    sym_cf = quant.encode(r, scale, l0).astype(np.int64)  # (n, 5) in {0,1,2,3}
    Channel(2).send(sym_cf)

    # Low-dim local statistic input (r_i, scale_i) for `obs_only` encoder ablation
    x_stat = np.stack([(r - l0).astype(np.float32), scale.astype(np.float32)], axis=-1)  # (n, 5, 2)

    return {
        "c": torch.tensor(c, dtype=torch.float32, device=device),
        "h": torch.tensor(h, dtype=torch.float32, device=device),
        "L": torch.tensor(L, dtype=torch.float32, device=device),
        "lo_cells": torch.tensor(pad_th[..., :-1], dtype=torch.float32, device=device),
        "hi_cells": torch.tensor(pad_th[..., 1:], dtype=torch.float32, device=device),
        "cell_vals": torch.tensor(cell_vals, dtype=torch.float32, device=device),
        "w_lin": torch.tensor(w_lin, dtype=torch.float32, device=device),
        "sym_cf": torch.tensor(sym_cf, dtype=torch.long, device=device),
        "x_stat": torch.tensor(x_stat, dtype=torch.float32, device=device),
        "l0": l0,
    }


def decode_fixed_bayes_e2_eval(
    oh_msg: torch.Tensor,
    e2_t: dict[str, Any],
    u_qmc: torch.Tensor,
    sl: slice | torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact Gaussian box-probability decoder on E2 (`gaussian_cell_decoder`) for one-hot 2-bit symbols."""
    c = e2_t["c"] if sl is None else e2_t["c"][sl]
    h = e2_t["h"] if sl is None else e2_t["h"][sl]
    L = e2_t["L"] if sl is None else e2_t["L"][sl]
    lo_c = e2_t["lo_cells"] if sl is None else e2_t["lo_cells"][sl]
    hi_c = e2_t["hi_cells"] if sl is None else e2_t["hi_cells"][sl]
    l0 = float(e2_t["l0"])
    lo = (oh_msg * lo_c).sum(dim=-1)
    hi = (oh_msg * hi_c).sum(dim=-1)
    lp1 = torch_box_logprob(c + h, L, lo, hi, u_qmc)
    lp0 = torch_box_logprob(c - h, L, lo, hi, u_qmc)
    return l0 + (lp1 - lp0).clamp(-CLIP, CLIP)


def decode_fixed_bayes_e2_train(
    p_msg: torch.Tensor,
    e2_t: dict[str, Any],
    idx: torch.Tensor,
    u_qmc_small: torch.Tensor,
) -> torch.Tensor:
    """Differentiable exact-forward + smooth-cell-centroid-gradient Bayes decoder on E2."""
    c = e2_t["c"][idx]
    h = e2_t["h"][idx]
    L = e2_t["L"][idx]
    lo_c = e2_t["lo_cells"][idx]
    hi_c = e2_t["hi_cells"][idx]
    cv = e2_t["cell_vals"][idx]
    w_lin = e2_t["w_lin"][idx]
    l0 = float(e2_t["l0"])

    # Smooth surrogate through truncated-normal cell centroids + exact covariance preconditioning
    r_hat = (p_msg * cv).sum(dim=-1)
    logit_surr = l0 + (w_lin * (r_hat - c)).sum(dim=-1)

    # Exact forward value from Genz's box log-probability at hard argmax cell
    with torch.no_grad():
        oh_hard = F.one_hot(p_msg.argmax(dim=-1), 4).float()
        lo = (oh_hard * lo_c).sum(dim=-1)
        hi = (oh_hard * hi_c).sum(dim=-1)
        lp1 = torch_box_logprob(c + h, L, lo, hi, u_qmc_small)
        lp0 = torch_box_logprob(c - h, L, lo, hi, u_qmc_small)
        logit_exact = l0 + (lp1 - lp0).clamp(-CLIP, CLIP)

    return logit_surr + (logit_exact - logit_surr).detach()


class DecouplingModel(nn.Module):
    """Unified model supporting `(i)` fixed CF encoder, `(ii)` fixed Bayes decoder, `(iii)` both learned, and uncompressed anchor."""

    def __init__(
        self,
        in_dim: int,
        b: int | None,
        channel: str,
        key_dim: int = 0,
        key_scale: float = 1.0,
        width: int = 64,
        enc_hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.b = b
        self.channel = channel
        self.key_dim = key_dim
        self.key_scale = key_scale
        self.use_keys = True
        if channel == "none":
            self.V = in_dim
        elif channel == "gauss":
            assert b is not None
            self.V = b
        else:
            assert b is not None
            self.V = 2 ** b
        self.sigma = math.sqrt(GAUSS_SIGMA2)

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))
        self.register_buffer("k_mean", torch.zeros(max(key_dim, 1)))
        self.register_buffer("k_sd", torch.ones(max(key_dim, 1)))

        if channel == "none":
            self.enc = None
        else:
            self.enc = nn.Sequential(
                nn.Linear(in_dim, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, self.V),
            )
        self.emb = nn.Linear(self.V + key_dim, width)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def encode_learned(self, x: torch.Tensor, mode: str = "train", tau: float = 1.0) -> torch.Tensor:
        z = (x - self.x_mean) / self.x_sd
        if self.channel == "none":
            return z
        assert self.enc is not None
        u = self.enc(z)
        if self.channel == "gauss":
            s = torch.tanh(u)
            if mode == "quant":
                return torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))
            return s + self.sigma * torch.randn_like(s)
        if mode == "train":
            return gumbel_code(u, tau=tau, hard=True)
        return F.one_hot(u.argmax(dim=-1), self.V).float()

    def encode_learned_probs_4(self, x: torch.Tensor, mode: str = "train", tau: float = 1.0) -> torch.Tensor:
        """Returns a 4-way symbol probability tensor `(B, N, 4)` for the fixed Bayes decoder in Arm (ii)."""
        assert self.enc is not None
        u = self.enc((x - self.x_mean) / self.x_sd)
        if self.channel == "gauss":
            if mode == "quant":
                b0 = (u[..., 0] >= 0).long()
                b1 = (u[..., 1] >= 0).long()
                sym = b0 * 2 + b1
                return F.one_hot(sym, 4).float()
            s = torch.tanh(u) + self.sigma * torch.randn_like(u)
            # Posterior probability P(bit = +1 | noisy channel output s) under N(+-1, 1/3): sigmoid(6*s)
            p1 = torch.sigmoid(6.0 * s)
            p0 = 1.0 - p1
            p_soft = torch.stack(
                [
                    p0[..., 0] * p0[..., 1],
                    p0[..., 0] * p1[..., 1],
                    p1[..., 0] * p0[..., 1],
                    p1[..., 0] * p1[..., 1],
                ],
                dim=-1,
            )
            b0 = (s[..., 0] >= 0).long()
            b1 = (s[..., 1] >= 0).long()
            oh = F.one_hot(b0 * 2 + b1, 4).float()
            return p_soft + (oh - p_soft).detach()
        if mode == "train":
            return gumbel_code(u, tau=tau, hard=True)
        return F.one_hot(u.argmax(dim=-1), 4).float()

    def symbols_learned(self, x: torch.Tensor) -> torch.Tensor:
        assert self.enc is not None
        u = self.enc((x - self.x_mean) / self.x_sd)
        if self.channel == "gauss":
            b0 = (u[..., 0] >= 0).long()
            b1 = (u[..., 1] >= 0).long()
            return b0 * 2 + b1
        return u.argmax(dim=-1)

    def decode_learned(self, m: torch.Tensor, keys: torch.Tensor | None = None) -> torch.Tensor:
        if keys is not None and self.key_dim > 0:
            kk = self.key_scale * (keys - self.k_mean) / self.k_sd
            inp = torch.cat([m, kk if self.use_keys else torch.zeros_like(kk)], dim=-1)
        else:
            inp = m
        h = self.emb(inp)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        z = self.norm(self.dec(h)[:, 0])
        return self.out(z)[:, 0]


def closed_form_messages_tensor(
    sym_cf: torch.Tensor,
    channel: str,
    mode: str = "train",
) -> torch.Tensor:
    """Converts integer 2-bit closed-form symbols `sym_cf` in `{0,1,2,3}` into the channel representation."""
    if channel == "gauss":
        b0 = ((sym_cf >> 1) & 1).float() * 2.0 - 1.0
        b1 = (sym_cf & 1).float() * 2.0 - 1.0
        s = torch.stack([b0, b1], dim=-1)  # in {-1, +1}^2
        if mode == "quant":
            return s
        sigma = math.sqrt(GAUSS_SIGMA2)
        return s + sigma * torch.randn_like(s)
    return F.one_hot(sym_cf, 4).float()


def extract_keys_tensor(env_name: str, X: torch.Tensor) -> torch.Tensor:
    """Extracts the per-agent structure keys (`key.full` on Z9-p2; `own_model` `(cov, mu)` on E2)."""
    if env_name == "Z9-p2":
        # X is [obs(2), key.full(14), local_logit(1)]
        return X[..., 2:16]
    # E2: X is [obs(12), seen(12), own_cov(12), own_mu(12), local_logit(1)] -> structure keys are [24:48]
    return X[..., 24:48]


def extract_obs_only_tensor(env_name: str, X: torch.Tensor, e2_t: dict[str, Any] | None = None) -> torch.Tensor:
    """Extracts only private observation/local-stat features without structure keys for `ii_obs_only`."""
    if env_name == "Z9-p2":
        return X[..., :2]
    assert e2_t is not None
    return e2_t["x_stat"]


def run_uncompressed_anchor(
    env_name: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_val: torch.Tensor,
    vb_val: np.ndarray,
    X_dv: torch.Tensor,
    dv_batch: Any,
    device: torch.device,
    eval_every: int = 200,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Trains the same-steps uncompressed Transformer anchor (`b = inf`, best-val checkpoint, fresh episodes on Z9-p2)."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    l0 = float(dv_batch.public["prior_logit"])
    in_dim = X_tr.shape[-1]
    net = DecouplingModel(in_dim=in_dim, b=None, channel="none", key_dim=0, key_scale=1.0).to(device)
    net.x_mean.copy_(X_tr[:10_000].reshape(-1, in_dim).mean(0))
    net.x_sd.copy_(X_tr[:10_000].reshape(-1, in_dim).std(0).clamp_min(1e-3))

    def eval_split(Xin: torch.Tensor) -> np.ndarray:
        net.eval()
        with torch.no_grad():
            m = net.encode_learned(Xin, mode="quant")
            q = l0 + net.decode_learned(m, None)
        net.train()
        return q.detach().cpu().numpy().astype(np.float64)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z9_gen = torch.Generator(device=device).manual_seed(seed * 10007 + 17)
    z9_spec = Z9Spec()
    all_idx = torch.randint(0, len(X_tr), (steps, batch_size), generator=gen).to(device)
    bce = F.binary_cross_entropy_with_logits

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    chunk_len = 500
    x_chunk: torch.Tensor | None = None
    y_chunk: torch.Tensor | None = None
    for t in range(steps):
        if env_name == "Z9-p2":
            t_mod = t % chunk_len
            if t_mod == 0:
                c_steps = min(chunk_len, steps - t)
                xc, _, yc, _ = sample_z9_fresh_batch(z9_spec, c_steps * batch_size, device, z9_gen)
                x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
                y_chunk = yc.reshape(c_steps, batch_size)
            assert x_chunk is not None and y_chunk is not None
            x = x_chunk[t_mod]
            y = y_chunk[t_mod]
        else:
            idx = all_idx[t]
            x = X_tr[idx]
            y = Y_tr[idx]
        m = net.encode_learned(x, mode="train")
        logit = l0 + net.decode_learned(m, None)
        loss = bce(logit, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if t % eval_every == 0 or t == steps - 1:
            val_fq = eval_split(X_val)
            val_kl = _kl_logits(val_fq, vb_val)
            if val_kl < best_val_kl:
                best_val_kl = float(val_kl)
                best_state = copy.deepcopy(net.state_dict())
                best_step = int(t)

    if best_state is not None:
        net.load_state_dict(best_state)
    fq_dv = eval_split(X_dv)
    regret = summarize({"logit": fq_dv}, dv_batch, n_boot=n_boot)["kl"]
    return {
        "seed": seed,
        "steps": steps,
        "best_val_kl": best_val_kl,
        "best_step": best_step,
        "regret": regret,
    }


def run_single_arm(
    env_name: str,
    arm_name: str,
    channel: str,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_val: torch.Tensor,
    vb_val: np.ndarray,
    X_dv: torch.Tensor,
    dv_batch: Any,
    sym_cf_tr: torch.Tensor,
    sym_cf_val: torch.Tensor,
    sym_cf_dv: torch.Tensor,
    z9_ctx: dict[str, Any] | None,
    e2_ctx: dict[str, Any] | None,
    device: torch.device,
    eval_every: int = 200,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """Runs one decoupling arm `(i)`, `(ii)`, or `(iii)` with best-val checkpoint selection and fresh Z9-p2 episodes."""
    b = 2
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    l0 = float(dv_batch.public["prior_logit"])
    prior_kl = _kl_logits(np.full(len(vb_val), l0), vb_val)

    with_keys = arm_name.endswith("_with_keys")
    obs_only = "obs_only" in arm_name
    use_fresh_z9 = (env_name == "Z9-p2") and (not arm_name.startswith("ii_enc_learned"))

    # Per #3894 Must-Fix 2: key_scale = 0.1 (in_norm=False) + key_warmup = 1000 steps on (iii) with keys
    is_iii_keys = arm_name == "iii_both_learned_with_keys"
    key_scale = 0.1 if with_keys else 1.0
    key_warmup = (min(1000, steps // 4) if steps < 2000 else 1000) if is_iii_keys else 0

    K_tr = extract_keys_tensor(env_name, X_tr) if with_keys else None
    K_val = extract_keys_tensor(env_name, X_val) if with_keys else None
    K_dv = extract_keys_tensor(env_name, X_dv) if with_keys else None
    key_dim = K_tr.shape[-1] if K_tr is not None else 0

    if obs_only:
        Xin_tr = extract_obs_only_tensor(env_name, X_tr, e2_ctx["tr"] if e2_ctx else None)
        Xin_val = extract_obs_only_tensor(env_name, X_val, e2_ctx["val"] if e2_ctx else None)
        Xin_dv = extract_obs_only_tensor(env_name, X_dv, e2_ctx["dv"] if e2_ctx else None)
    else:
        Xin_tr, Xin_val, Xin_dv = X_tr, X_val, X_dv

    in_dim = Xin_tr.shape[-1]
    net = DecouplingModel(
        in_dim=in_dim,
        b=b,
        channel=channel,
        key_dim=key_dim,
        key_scale=key_scale,
    ).to(device)
    net.x_mean.copy_(Xin_tr[:10_000].reshape(-1, in_dim).mean(0))
    net.x_sd.copy_(Xin_tr[:10_000].reshape(-1, in_dim).std(0).clamp_min(1e-3))
    if K_tr is not None:
        net.k_mean.copy_(K_tr[:10_000].reshape(-1, key_dim).mean(0))
        net.k_sd.copy_(K_tr[:10_000].reshape(-1, key_dim).std(0).clamp_min(1e-3))

    if arm_name.startswith("i_enc_cf"):
        # Freeze encoder, train ONLY decoder
        assert net.enc is not None
        for p in net.enc.parameters():
            p.requires_grad_(False)
        params = [p for n, p in net.named_parameters() if not n.startswith("enc.")]
    elif arm_name.startswith("ii_enc_learned"):
        # Freeze decoder, train ONLY encoder
        assert net.enc is not None
        for n, p in net.named_parameters():
            if not n.startswith("enc."):
                p.requires_grad_(False)
        params = list(net.enc.parameters())
    elif arm_name.startswith("iii_both_learned"):
        params = list(net.parameters())
    else:
        raise ValueError(f"Unknown arm_name: {arm_name}")

    def eval_split(Xin: torch.Tensor, K_split: torch.Tensor | None, sym_cf_split: torch.Tensor, split_tag: str) -> np.ndarray:
        net.eval()
        g = torch.random.get_rng_state()
        cuda_g = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        torch.manual_seed(12345)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(12345)
        with torch.no_grad():
            if arm_name.startswith("i_enc_cf"):
                m = closed_form_messages_tensor(sym_cf_split, channel, mode="quant")
                q = l0 + net.decode_learned(m, K_split)
            elif arm_name.startswith("ii_enc_learned"):
                p_oh = net.encode_learned_probs_4(Xin, mode="quant")
                if env_name == "Z9-p2":
                    assert z9_ctx is not None
                    q = decode_fixed_bayes_z9(p_oh, z9_ctx[f"lik_{split_tag}"], z9_ctx["idx_y1"], z9_ctx["idx_y0"])
                else:
                    assert e2_ctx is not None
                    q = decode_fixed_bayes_e2_eval(p_oh, e2_ctx[split_tag], e2_ctx["u_eval"])
            else:
                m = net.encode_learned(Xin, mode="quant")
                q = l0 + net.decode_learned(m, K_split)
        torch.random.set_rng_state(g)
        if cuda_g is not None:
            torch.cuda.set_rng_state_all(cuda_g)
        net.train()
        return q.detach().cpu().numpy().astype(np.float64)

    net.use_keys = True
    untrained_fq = eval_split(Xin_dv, K_dv, sym_cf_dv, "dv")
    control_untrained = summarize({"logit": untrained_fq}, dv_batch, n_boot=min(n_boot, 500))["kl"]

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    z9_gen = torch.Generator(device=device).manual_seed(seed * 10007 + 99)
    z9_spec = Z9Spec()
    all_idx = torch.randint(0, len(Xin_tr), (steps, batch_size), generator=gen).to(device)
    bce = F.binary_cross_entropy_with_logits

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    curve: list[dict[str, Any]] = []
    chunk_len = 500
    x_chunk: torch.Tensor | None = None
    k_chunk: torch.Tensor | None = None
    y_chunk: torch.Tensor | None = None
    sym_chunk: torch.Tensor | None = None
    for t in range(steps):
        tau = max(0.5, 1.0 - t / max(steps, 1))
        net.use_keys = t >= key_warmup

        if use_fresh_z9:
            t_mod = t % chunk_len
            if t_mod == 0:
                c_steps = min(chunk_len, steps - t)
                xc, kc, yc, sc = sample_z9_fresh_batch(z9_spec, c_steps * batch_size, device, z9_gen)
                x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
                k_chunk = kc.reshape(c_steps, batch_size, kc.shape[1], kc.shape[2])
                y_chunk = yc.reshape(c_steps, batch_size)
                sym_chunk = sc.reshape(c_steps, batch_size, sc.shape[1])
            assert x_chunk is not None and k_chunk is not None and y_chunk is not None and sym_chunk is not None
            x_b = x_chunk[t_mod]
            k_b = k_chunk[t_mod] if with_keys else None
            y = y_chunk[t_mod]
            sym_cf_b = sym_chunk[t_mod]
        else:
            idx = all_idx[t]
            x_b = Xin_tr[idx]
            y = Y_tr[idx]
            k_b = K_tr[idx] if K_tr is not None else None
            sym_cf_b = sym_cf_tr[idx]

        if arm_name.startswith("i_enc_cf"):
            cf_mode = "quant" if arm_name.endswith("_clean") else ("train" if channel == "gauss" else "quant")
            m = closed_form_messages_tensor(sym_cf_b, channel, mode=cf_mode)
            logit = l0 + net.decode_learned(m, k_b)
        elif arm_name.startswith("ii_enc_learned"):
            idx = all_idx[t]
            p_msg = net.encode_learned_probs_4(x_b, mode="train", tau=tau)
            if env_name == "Z9-p2":
                assert z9_ctx is not None
                logit = decode_fixed_bayes_z9(p_msg, z9_ctx["lik_tr"][idx], z9_ctx["idx_y1"], z9_ctx["idx_y0"])
            else:
                assert e2_ctx is not None
                logit = decode_fixed_bayes_e2_train(p_msg, e2_ctx["tr"], idx, e2_ctx["u_train"])
        else:
            m = net.encode_learned(x_b, mode="train", tau=tau)
            logit = l0 + net.decode_learned(m, k_b)

        loss = bce(logit, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if t % eval_every == 0 or t == steps - 1:
            val_fq = eval_split(Xin_val, K_val, sym_cf_val, "val")
            val_kl = _kl_logits(val_fq, vb_val)
            curve.append(
                {
                    "step": int(t),
                    "val_kl": float(val_kl),
                    "loss": float(loss.item()),
                    "keys_on": bool(net.use_keys),
                }
            )
            # Never pick a keys-off checkpoint when key_warmup > 0
            if net.use_keys and val_kl < best_val_kl:
                best_val_kl = float(val_kl)
                best_state = copy.deepcopy(net.state_dict())
                best_step = int(t)

    net.use_keys = True
    if best_state is not None:
        net.load_state_dict(best_state)

    fq_dv = eval_split(Xin_dv, K_dv, sym_cf_dv, "dv")
    regret_quant = summarize({"logit": fq_dv}, dv_batch, n_boot=n_boot)["kl"]

    with torch.no_grad():
        if arm_name.startswith("i_enc_cf"):
            sym_np = sym_cf_dv.detach().cpu().numpy().astype(np.int64)
        else:
            sym_np = net.symbols_learned(Xin_dv).detach().cpu().numpy().astype(np.int64)
    Channel(b).send(sym_np)
    distinct_symbols = int(len(np.unique(sym_np)))

    esc = [c["step"] for c in curve if c["val_kl"] < ESCAPE_FRAC * prior_kl]
    steps_to_escape = esc[0] if esc else None

    return {
        "env": env_name,
        "b": b,
        "arm": arm_name,
        "channel": channel,
        "seed": seed,
        "steps": steps,
        "fresh_episodes": use_fresh_z9,
        "key_scale": key_scale,
        "key_warmup": key_warmup,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "steps_to_escape": steps_to_escape,
        "val_prior_kl": float(prior_kl),
        "control_untrained": control_untrained,
        "regret_quant": regret_quant,
        "distinct_symbols": distinct_symbols,
        "curve": curve,
        "forecast_quant": fq_dv.tolist(),
    }


def aggregate_seeds(records: list[dict[str, Any]], anchor_mean: float | None = None) -> dict[str, Any]:
    means = [float(r["regret_quant"]["mean"]) for r in records]
    ci_lows = [float(r["regret_quant"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret_quant"]["ci_high"]) for r in records]
    escapes = [r["steps_to_escape"] for r in records]
    esc_non_null = [int(e) for e in escapes if e is not None]
    out: dict[str, Any] = {
        "n_seeds": len(records),
        "regret_mean": float(np.mean(means)),
        "regret_sd": float(np.std(means, ddof=0)),
        "regret_min": float(np.min(means)),
        "regret_max": float(np.max(means)),
        "mean_ci_low": float(np.mean(ci_lows)),
        "mean_ci_high": float(np.mean(ci_highs)),
        "per_seed_regret": [round(m, 6) for m in means],
        "per_seed_escape": escapes,
        "n_escaped": len(esc_non_null),
        "mean_escape_step": float(np.mean(esc_non_null)) if esc_non_null else None,
        "distinct_symbols_mean": float(np.mean([r["distinct_symbols"] for r in records])),
    }
    if anchor_mean is not None:
        out["gap_to_same_steps_anchor"] = float(np.mean(means) - anchor_mean)
    return out


def render_readme(summary: dict[str, Any]) -> str:
    """Renders a self-checking markdown report from `summary` where every 4-decimal number comes from `summary`."""
    lines: list[str] = []
    lines.append("# Compression Decoupling Diagnostic (`#3887` & `#3894`: `Z9-p2` & `E2` at `b = 2`, Label-Only, 3 Seeds)")
    lines.append("")
    lines.append("**Track: `EXPLORATORY` (`gate must-fixes applied per #3894`)**")
    lines.append("")
    lines.append("Decouples encoder learning, decoder learning, and per-episode structure keys (`key.full` / `oracle`) at `b = 2` bits/agent across `3` seeds (`0, 1, 2`), label-only (`Y` labels only), with **fresh episodes at every step on `Z9-p2`**, **best-val checkpoint selection (`best_val_kl`) on both `Z9-p2` and `E2`**, and **`key_scale = 0.1` (`in_norm=False`) + `key_warmup = 1000` steps on `(iii) iii_both_learned_with_keys`**:")
    lines.append("- **`(i)` Fixed closed-form encoder + learned decoder (`i_enc_cf_dec_learned_*`):**")
    lines.append("  - `i_enc_cf_dec_learned_msg_only`: standard set decoder reading **only** the `b = 2`-bit closed-form messages `m` (`raw_code` on `Z9-p2`, `llr_quantizer` on `E2`).")
    lines.append("  - `i_enc_cf_dec_learned_with_keys`: set decoder reading `[m, key_structure]`, receiving the per-agent structure keys (`key.full` on `Z9-p2`; `own_model` on `E2`), reported alongside the **same-steps uncompressed Transformer anchor** (`8000` steps on `Z9-p2`, `4000` steps on `E2`) to separate the decoder's finite-step ceiling from the `2`-bit quantization gap.")
    lines.append("- **`(ii)` Learned encoder + `oracle-structure + oracle-codebook decoder [PROVENANCE FULL]` (`ii_enc_learned_dec_fixed_bayes*`):**")
    lines.append("  - **Scope & Provenance Note (`#3894` Must-Fix 1):** Arm `(ii)` is an **`oracle-structure + oracle-codebook decoder [PROVENANCE FULL]`** diagnostic, **not** a blind fixed decoder. On `Z9-p2`, `decode_fixed_bayes_z9` uses per-episode `batch.oracle['var']` and `batch.oracle['acc']` (`make_z9_lik_tensor`); on `E2`, `decode_fixed_bayes_e2_eval` uses per-episode `batch.oracle['cov']`, `batch.oracle['mu']` (`gaussian_report_model`), and the oracle quantizer's cell thresholds `quant.thresholds(scale, l0)`. Therefore, `(ii)` shows **only** that the encoder MLP can learn to match a known codebook when the decoder already knows the per-episode structure and codebook cells; it says **nothing** about blind compression.")
    lines.append("- **`(iii)` Both learned end-to-end (`iii_both_learned_*`):**")
    lines.append("  - `iii_both_learned_msg_only`: message-only set decoder `decode(m)` without structure keys at the decoder (`[BLIND DECODER: messages only]`).")
    lines.append("  - `iii_both_learned_with_keys`: both encoder (`b = 2` bits verified by `Channel(2).send`) and structure-key-aware set decoder `decode([m, key_structure])` (`STRUCTURE-KNOWN (decoder reads keys['full'] outside b)`) trained end-to-end on `Y` labels with `key_scale = 0.1` (`in_norm=False`) and `key_warmup = 1000` steps (`best_val_kl` restricted to `t >= key_warmup`).")
    lines.append("  - **Spec & Channel Scope Note (`#3896` Must-Fix 5):** On `Z9-p2`, arm `(iii)` (`0/3` escaped) ran on the **old `Z9-p2` spec (`prior = 0.4567`, deterministic `2` readers per parity bit + `6` distractor readers = `14` readers / `7` agents)** with **Gaussian / Gumbel channels**, whereas `tt-compression-joint-remedies-20261022` ran on **leak-reduced `Z9-p2-lf` (`prior = 0.2702`, `Dirichlet` random assignment of `24` parity + `2` distractor readers = `26` readers / `13` agents)** with **`FSQ`**. Comparing `(iii)` here against `joint-remedies` does **not** isolate \"FSQ fixes joint learning\" because the environment spec and channel are confounded.")
    lines.append("")

    for env_name in ["Z9-p2", "E2"]:
        if env_name not in summary["envs"]:
            continue
        ed = summary["envs"][env_name]
        anc = ed["anchors"]
        lines.append(f"## Environment: `{env_name}` (`b = 2` bits/agent, `steps = {ed['steps']}`)")
        lines.append("")
        lines.append("### 1. Anchors & Plug-in Tabulated Message-Only Floor (`n_dev = 2000`)")
        lines.append("")
        lines.append("| Anchor | Regret vs Bayes (nats) | 95% Bootstrap CI / Seed SD | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for k, label, note in [
            ("prior", "`prior (b = 0)`", "No messages"),
            (
                "same_steps_uncompressed_transformer",
                f"`same_steps_uncompressed_transformer` (`{ed['steps']}` steps, `b = inf`)",
                "Same-steps uncompressed Transformer anchor (`best_val_kl` checkpoint across 3 seeds)",
            ),
            (
                "closed_form_oracle_decoder",
                "`closed_form_oracle_decoder` (`raw_code` / `llr_quantizer` `[PROVENANCE FULL]`)",
                "Oracle-structure Bayes decoder given 2-bit messages **plus** per-episode structure (`oracle['var']` / `oracle['cov','mu']`)",
            ),
            (
                "plug_in_msg_only_floor",
                "`plug-in tabulated message-only floor (±0.001 plug-in bias)`",
                "Plug-in tabulated message-only floor (`120` multiset cells on `Z9-p2`, `56` on `E2`, estimated from `100,000` train episodes with Laplace smoothing; carries `±0.001` finite-sample plug-in bias)",
            ),
        ]:
            m = anc[k]
            lines.append(
                f"| {label} | `{m['mean']:.4f}` | `[`{m['ci_low']:.4f}`, `{m['ci_high']:.4f}`]` | {note} |"
            )
        lines.append("")
        lines.append("### 2. Decoupling Matrix (`3` seeds `0, 1, 2`, label-only, `best_val_kl` checkpoint)")
        lines.append("")
        lines.append("| Arm | Provenance / Decoder Structure | Channel | Mean Regret (`regret_quant`) | Gap to Same-Steps Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Escaped (`val_kl < 0.9 * prior`) | Escape Steps (`s0, s1, s2`) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for arm_name, by_ch in ed["aggregated"].items():
            prov_tag = (
                "`oracle-structure + oracle-codebook decoder [PROVENANCE FULL]`"
                if arm_name.startswith("ii_enc_learned")
                else ("`STRUCTURE-KNOWN (decoder reads keys['full'] outside b)`" if arm_name.endswith("with_keys") else "`[BLIND DECODER: messages only]`")
            )
            for ch in ["gauss", "gumbel"]:
                if ch not in by_ch:
                    continue
                ag = by_ch[ch]
                s_regs = ", ".join(f"`{v:.4f}`" for v in ag["per_seed_regret"])
                s_escs = ", ".join(f"`{v}`" if v is not None else "`None`" for v in ag["per_seed_escape"])
                gap_val = ag.get("gap_to_same_steps_anchor", 0.0)
                lines.append(
                    f"| `{arm_name}` | {prov_tag} | `{ch}` | `{ag['regret_mean']:.4f}` | `{gap_val:+.4f}` | `{ag['regret_sd']:.4f}` | "
                    f"`[`{ag['mean_ci_low']:.4f}`, `{ag['mean_ci_high']:.4f}`]` | {s_regs} | "
                    f"`{ag['n_escaped']}/{ag['n_seeds']}` | {s_escs} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def collect_4dec_strings(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            collect_4dec_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_4dec_strings(v, out)
    elif isinstance(obj, float) and math.isfinite(obj):
        out.add(f"{obj:.4f}")
        out.add(f"{abs(obj):.4f}")


def verify_readme_numbers(readme_text: str, summary: dict[str, Any]) -> tuple[int, list[str]]:
    allowed: set[str] = {"0.0000", "2.0000"}
    collect_4dec_strings(summary, allowed)
    found = re.findall(r"\b\d+\.\d{4}\b", readme_text)
    missing = [x for x in found if x not in allowed]
    return len(found), missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["Z9-p2", "E2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--channels", nargs="+", default=["gauss", "gumbel"])
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-dev", type=int, default=2_000)
    parser.add_argument("--steps-e2", type=int, default=4_000)
    parser.add_argument("--steps-z9", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--reuse-fixed-dec", type=str, default=None, help="Reuse arm (ii) runs from existing summary.json")
    parser.add_argument("--out-dir", type=str, default="results/explore/tt-compression-decoupling-20261022")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_train = 1_000
        args.n_val = 100
        args.n_dev = 100
        args.steps_e2 = 12
        args.steps_z9 = 12
        args.batch_size = 64
        args.eval_every = 4
        args.n_boot = 50
        args.seeds = [0]

    prev_summary = None
    if args.reuse_fixed_dec and pathlib.Path(args.reuse_fixed_dec).exists():
        prev_summary = json.loads(pathlib.Path(args.reuse_fixed_dec).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms_order = [
        "i_enc_cf_dec_learned_msg_only",
        "i_enc_cf_dec_learned_with_keys",
        "ii_enc_learned_dec_fixed_bayes",
        "ii_enc_learned_obs_only_dec_fixed_bayes",
        "iii_both_learned_msg_only",
        "iii_both_learned_with_keys",
    ]

    summary: dict[str, Any] = {
        "schema": "aggbench.compression-decoupling-diag/v2",
        "harness": harness_info(),
        "b": 2,
        "seeds": args.seeds,
        "channels": args.channels,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_dev": args.n_dev,
        "device": str(device),
        "envs": {},
    }

    t_start = time.time()
    for env_name in args.envs:
        t_env = time.time()
        env = make_cenv(env_name)
        need_ii = prev_summary is None or env_name not in prev_summary.get("envs", {})
        n_tr_sample = min(4096, args.n_train) if (env_name == "Z9-p2" and not need_ii) else args.n_train
        tr = env.sample("train", n_tr_sample, start=0)
        val = env.sample("dev", args.n_val, start=50_000)
        dv = env.sample("dev", args.n_dev, start=0)

        X_tr = torch.tensor(agent_tokens(tr), dtype=torch.float32, device=device)
        Y_tr = torch.tensor(np.asarray(tr.truth, np.float32), dtype=torch.float32, device=device)
        X_val = torch.tensor(agent_tokens(val), dtype=torch.float32, device=device)
        vb_val = np.asarray(val.bayes["logit"], dtype=np.float64)
        X_dv = torch.tensor(agent_tokens(dv), dtype=torch.float32, device=device)

        prior_fc = prior_forecast(dv)
        prior_kl = summarize(prior_fc, dv, n_boot=args.n_boot)["kl"]

        z9_ctx = None
        e2_ctx = None
        if env_name == "Z9-p2":
            cf_fc = raw_code_forecast(env, dv, 2)
            cf_oracle_kl = summarize(cf_fc, dv, n_boot=args.n_boot)["kl"]
            if not need_ii and not args.smoke:
                prev_anc = prev_summary["envs"]["Z9-p2"]["anchors"]
                msg_floor_kl = prev_anc.get("plug_in_msg_only_floor") or prev_anc["closed_form_msg_only_bayes_floor"]
            else:
                tr_floor_ref = tr if (args.smoke or len(tr) >= 100_000) else env.sample("train", 100_000, start=0)
                msg_floor_fc = compute_z9_msg_only_bayes_floor(env, tr_floor_ref, dv, b=2)
                msg_floor_kl = summarize(msg_floor_fc, dv, n_boot=args.n_boot)["kl"]

            sym_cf_tr = torch.tensor(tr.oracle["sym"].astype(np.int64), dtype=torch.long, device=device)
            sym_cf_val = torch.tensor(val.oracle["sym"].astype(np.int64), dtype=torch.long, device=device)
            sym_cf_dv = torch.tensor(dv.oracle["sym"].astype(np.int64), dtype=torch.long, device=device)
            if need_ii:
                idx_y1, idx_y0 = make_z9_target_masks(device)
                z9_ctx = {
                    "lik_tr": make_z9_lik_tensor(tr, device),
                    "lik_val": make_z9_lik_tensor(val, device),
                    "lik_dv": make_z9_lik_tensor(dv, device),
                    "idx_y1": idx_y1,
                    "idx_y0": idx_y0,
                }
            steps = args.steps_z9
        elif env_name == "E2":
            tr_tune = env.sample("train", min(2000, args.n_train), start=0)
            quant = tune_llr_quantizer(env, tr_tune, 2)
            cf_fc, _ = llr_quantizer_forecast(env, dv, quant, M=1024 if not args.smoke else 128)
            cf_oracle_kl = summarize(cf_fc, dv, n_boot=args.n_boot)["kl"]
            tr_floor_ref = tr if (args.smoke or args.n_train >= 100_000) else env.sample("train", 100_000, start=0)
            msg_floor_fc = compute_e2_msg_only_bayes_floor(env, tr_floor_ref, dv, quant)
            msg_floor_kl = summarize(msg_floor_fc, dv, n_boot=args.n_boot)["kl"]

            if need_ii:
                e2_tr = make_e2_exact_tensors(tr, quant, device)
                e2_val = make_e2_exact_tensors(val, quant, device)
                e2_dv = make_e2_exact_tensors(dv, quant, device)
                sym_cf_tr = e2_tr["sym_cf"]
                sym_cf_val = e2_val["sym_cf"]
                sym_cf_dv = e2_dv["sym_cf"]
                e2_ctx = {
                    "tr": e2_tr,
                    "val": e2_val,
                    "dv": e2_dv,
                    "u_train": torch.tensor(_qmc(32, 5, seed=0), dtype=torch.float32, device=device),
                    "u_eval": torch.tensor(_qmc(256, 5, seed=0), dtype=torch.float32, device=device),
                }
            else:
                r_tr, sc_tr = agent_statistic(tr)
                r_val, sc_val = agent_statistic(val)
                r_dv, sc_dv = agent_statistic(dv)
                l0 = float(dv.public["prior_logit"])
                sym_cf_tr = torch.tensor(quant.encode(r_tr, sc_tr, l0).astype(np.int64), dtype=torch.long, device=device)
                sym_cf_val = torch.tensor(quant.encode(r_val, sc_val, l0).astype(np.int64), dtype=torch.long, device=device)
                sym_cf_dv = torch.tensor(quant.encode(r_dv, sc_dv, l0).astype(np.int64), dtype=torch.long, device=device)
            steps = args.steps_e2
        else:
            raise ValueError(f"Unsupported env: {env_name}")

        # Train same-steps uncompressed Transformer anchor across seeds
        anchor_recs = []
        for seed in args.seeds:
            t_a = time.time()
            a_rec = run_uncompressed_anchor(
                env_name=env_name,
                seed=seed,
                steps=steps,
                batch_size=args.batch_size,
                lr=args.lr,
                X_tr=X_tr,
                Y_tr=Y_tr,
                X_val=X_val,
                vb_val=vb_val,
                X_dv=X_dv,
                dv_batch=dv,
                device=device,
                eval_every=args.eval_every,
                n_boot=args.n_boot,
            )
            anchor_recs.append(a_rec)
            print(
                f"[{env_name}] same_steps_uncompressed_transformer seed={seed}: "
                f"regret={a_rec['regret']['mean']:.4f} best_step={a_rec['best_step']} ({time.time() - t_a:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
        anchor_means = [float(r["regret"]["mean"]) for r in anchor_recs]
        anchor_summary = {
            "mean": float(np.mean(anchor_means)),
            "sd": float(np.std(anchor_means, ddof=0)),
            "ci_low": float(np.mean([float(r["regret"]["ci_low"]) for r in anchor_recs])),
            "ci_high": float(np.mean([float(r["regret"]["ci_high"]) for r in anchor_recs])),
            "per_seed": [round(m, 6) for m in anchor_means],
        }

        print(
            f"[{env_name}] Anchors: prior={prior_kl['mean']:.4f} "
            f"same_steps_tr={anchor_summary['mean']:.4f} "
            f"cf_oracle_dec={cf_oracle_kl['mean']:.4f} "
            f"plug_in_msg_only_floor={msg_floor_kl['mean']:.4f} (prep {time.time() - t_env:.1f}s)",
            file=sys.stderr,
            flush=True,
        )

        env_results: list[dict[str, Any]] = []
        aggregated: dict[str, dict[str, Any]] = {}
        for arm_name in arms_order:
            aggregated[arm_name] = {}
            if arm_name.startswith("ii_enc_learned") and not need_ii:
                prev_ed = prev_summary["envs"][env_name]
                for ch in args.channels:
                    prev_runs = [
                        r for r in prev_ed["runs"] if r["arm"] == arm_name and r["channel"] == ch and r["seed"] in args.seeds
                    ]
                    for r in prev_runs:
                        r["provenance"] = "oracle-structure + oracle-codebook decoder [PROVENANCE FULL]"
                        env_results.append(r)
                    aggregated[arm_name][ch] = aggregate_seeds(prev_runs, anchor_mean=anchor_summary["mean"])
                continue

            for ch in args.channels:
                seed_recs: list[dict[str, Any]] = []
                for seed in args.seeds:
                    t0 = time.time()
                    rec = run_single_arm(
                        env_name=env_name,
                        arm_name=arm_name,
                        channel=ch,
                        seed=seed,
                        steps=steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        X_tr=X_tr,
                        Y_tr=Y_tr,
                        X_val=X_val,
                        vb_val=vb_val,
                        X_dv=X_dv,
                        dv_batch=dv,
                        sym_cf_tr=sym_cf_tr,
                        sym_cf_val=sym_cf_val,
                        sym_cf_dv=sym_cf_dv,
                        z9_ctx=z9_ctx,
                        e2_ctx=e2_ctx,
                        device=device,
                        eval_every=args.eval_every,
                        n_boot=args.n_boot,
                    )
                    rec.pop("forecast_quant", None)
                    if arm_name.startswith("ii_enc_learned"):
                        rec["provenance"] = "oracle-structure + oracle-codebook decoder [PROVENANCE FULL]"
                    seed_recs.append(rec)
                    env_results.append(rec)
                    print(
                        f"[{env_name}] {arm_name:38s} {ch:6s} seed={seed}: "
                        f"regret={rec['regret_quant']['mean']:.4f} "
                        f"[{rec['regret_quant']['ci_low']:.4f}, {rec['regret_quant']['ci_high']:.4f}] "
                        f"best_step={rec['best_step']} escape={rec['steps_to_escape']} syms={rec['distinct_symbols']} ({time.time() - t0:.1f}s)",
                        file=sys.stderr,
                        flush=True,
                    )
                aggregated[arm_name][ch] = aggregate_seeds(seed_recs, anchor_mean=anchor_summary["mean"])

        summary["envs"][env_name] = {
            "env": env_name,
            "b": 2,
            "steps": steps,
            "anchors": {
                "prior": prior_kl,
                "same_steps_uncompressed_transformer": anchor_summary,
                "closed_form_oracle_decoder": cf_oracle_kl,
                "plug_in_msg_only_floor": msg_floor_kl,
            },
            "aggregated": aggregated,
            "runs": env_results,
        }

        # Incremental save after each env
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        readme_text = render_readme(summary)
        n_checked, missing = verify_readme_numbers(readme_text, summary)
        assert not missing, f"Unverified numbers in README: {missing}"
        (out_dir / "README.md").write_text(readme_text)

    summary["elapsed_sec"] = round(time.time() - t_start, 2)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    readme_text = render_readme(summary)
    n_checked, missing = verify_readme_numbers(readme_text, summary)
    assert not missing, f"Unverified numbers in README: {missing}"
    (out_dir / "README.md").write_text(readme_text)
    print(f"Done in {summary['elapsed_sec']}s; verified {n_checked}/{n_checked} 4-dec numbers in {out_dir / 'README.md'}.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
