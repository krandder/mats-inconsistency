"""Joint-Learning Remedies with FSQ on Leak-Reduced `Z9-p2` (`b481850`) and `E2` at `b in {2, 4, 6}` (`#3893`).

Track: EXPLORATORY (gate pending on 50b060e).

Tests remedies for the encoder-decoder chicken-and-egg problem in joint learned compression at `b in {2, 4, 6}` bits/agent
(`2^b in {4, 16, 64}` symbols), across `3` seeds (`0, 1, 2`), `Y` labels only, with Finite Scalar Quantization (`FSQ`,
Mentzer et al. `2309.15505`) as the quantizer and `key_scale = 0.1` whenever an arm takes keys:
- `0_fsq_joint`: Plain joint end-to-end training with `FSQ`.
- `a_fsq_alt_opt`: `(a)` Alternating optimization (`k_alt = 200` steps decoder with encoder frozen, `k_alt = 200` steps
  encoder with decoder frozen, using separate AdamW optimizers so momentum buffers never step on frozen phases).
- `b_fsq_dec_warmstart`: `(b)` Decoder warm-start on a random fixed full-entropy `FSQ` code (first 30% of steps with
  orthogonal zero-bias calibrated encoder frozen), followed by joint training for the remaining 70% of steps.
- `c_fsq_entropy_bonus`: `(c)` Joint training with an encoder marginal symbol-entropy / usage bonus (`L_usage`) over all
  `2^b` soft `FSQ` hypercube cells plus per-token quantization sharpness.
- `d_fsq_staged_channel`: `(d)` Staged continuous-to-discrete channel (bounded float message `tanh(u)` for the first 35%
  of steps, cosine-annealed interpolation `(1 - alpha(t)) * tanh(u) + alpha(t) * FSQ(u)` over 35%..75%, and pure discrete
  `b`-bit `FSQ` over the final 25% of steps).
- `e_fsq_biased_curr_plus_best`: `(e)` Biased-bit curriculum (`P(bit=1): 0.8 -> 0.65 -> 0.5` on `Z9-p2-lf`; high-to-normal
  piece-SNR curriculum on `E2`, both `Y` labels only) combined with the best of `(a)-(d)` (full-entropy orthogonal init,
  staged continuous-to-discrete `FSQ`, marginal symbol-entropy bonus, and final discrete decoder lock-in).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import re
import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.compression import (
    Channel,
    agent_tokens,
    make_cenv,
    run_closed_form,
)
from aggbench.metrics import logsig, summarize, summarize_values
from aggbench.report import prior_forecast
from aggbench.results import harness_info
from aggbench.parity_learn import (
    DEV_SEED,
    NV,
    STATS_SEED,
    VAL_SEED,
    Z9Spec,
    _kl,
    biased_bit_curriculum,
    build as build_parity_net,
    prior_logit,
    sample as sample_z9_np,
)


def compute_symbol_metrics(syms: np.ndarray, b: int) -> dict[str, Any]:
    """Verifies `syms` (`(n_eval, N)`) through `Channel(b).send` and computes per-agent and pooled entropy in bits."""
    K = 2 ** b
    syms = np.asarray(syms, dtype=np.int64)
    Channel(b).send(syms)
    flat = syms.reshape(-1)
    counts_all = np.bincount(flat, minlength=K).astype(np.float64)
    p_all = counts_all / max(1.0, counts_all.sum())
    nz = p_all > 0
    h_pooled = float(-np.sum(p_all[nz] * np.log2(p_all[nz])))

    N = syms.shape[1]
    h_agents = []
    distinct_per_agent = []
    for i in range(N):
        c_i = np.bincount(syms[:, i], minlength=K).astype(np.float64)
        p_i = c_i / max(1.0, c_i.sum())
        nz_i = p_i > 0
        h_agents.append(float(-np.sum(p_i[nz_i] * np.log2(p_i[nz_i]))))
        distinct_per_agent.append(int(np.sum(c_i > 0)))

    return {
        "distinct_symbols_total": int(np.sum(counts_all > 0)),
        "distinct_symbols_max_agent": int(max(distinct_per_agent)),
        "distinct_symbols_min_agent": int(min(distinct_per_agent)),
        "pooled_entropy_bits": h_pooled,
        "mean_agent_entropy_bits": float(np.mean(h_agents)),
    }


def sample_z9_torch(
    spec: Z9Spec,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized GPU sampler matching `aggbench.parity_learn.sample(spec, n, rng)` token and key construction."""
    R = spec.n_readers
    k = spec.k
    n_par = k * spec.readers_per_bit
    if spec.readers_random and spec.readers_alpha > 0:
        if spec.readers_alpha == 1.0:
            u_exp = torch.rand((n, k), generator=gen, device=device).clamp_min(1e-12)
            g = -torch.log(u_exp)
        else:
            g = torch._standard_gamma(
                torch.full((n, k), float(spec.readers_alpha), device=device),
                generator=gen,
            )
        w = g / g.sum(dim=1, keepdim=True).clamp_min(1e-12)
        u = torch.rand((n, n_par, 1), generator=gen, device=device)
        par = (u > torch.cumsum(w, dim=1).unsqueeze(1)).sum(dim=-1).clamp(0, k - 1)
    elif spec.readers_random:
        par = torch.randint(0, k, (n, n_par), generator=gen, device=device)
    else:
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
    tok = torch.cat([obs, keys], dim=-1)
    return tok, keys, y


class FSQJointRemedyNet(nn.Module):
    """Per-agent encoder + `b`-bit `FSQ` (`2309.15505`) + 2-layer Transformer set decoder (`key_scale = 0.1`, `key_warmup`)."""

    def __init__(
        self,
        in_dim: int,
        key_dim: int,
        b: int,
        dec_keys: bool = True,
        key_scale: float = 0.1,
        obs_dim: int = 2,
        width: int = 64,
        enc_hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        full_entropy_init: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.key_dim = key_dim
        self.b = b
        self.K = 2 ** b
        self.dec_keys = dec_keys
        self.key_scale = key_scale
        self.obs_dim = obs_dim
        self.full_entropy_init = full_entropy_init
        self.use_keys = True

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))
        self.register_buffer("k_mean", torch.zeros(max(1, key_dim)))
        self.register_buffer("k_sd", torch.ones(max(1, key_dim)))
        self.register_buffer("u_offset", torch.zeros(b))
        self.register_buffer("u_scale", torch.ones(b))

        # Precompute binary hypercube corners (2^b, b) in {0, 1} for exact differentiable FSQ usage entropy
        corners = ((torch.arange(self.K).unsqueeze(1) >> torch.arange(b).unsqueeze(0)) & 1).float()
        self.register_buffer("corners", corners)
        self.register_buffer("bit_weights", (2 ** torch.arange(b)).long())

        self.enc = nn.Sequential(
            nn.Linear(in_dim, enc_hidden),
            nn.GELU(),
            nn.Linear(enc_hidden, enc_hidden),
            nn.GELU(),
            nn.Linear(enc_hidden, b),
        )
        # Residual linear skip from input token so observations/statistics have direct gradient access at step 0
        self.enc_skip = nn.Linear(in_dim, b, bias=False)

        if full_entropy_init:
            for m in self.enc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.enc_skip.weight, gain=1.2)

        dec_in = b + (key_dim if dec_keys else 0)
        self.inp = nn.Linear(dec_in, width)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.enc.parameters()) + list(self.enc_skip.parameters())

    def decoder_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.inp.parameters())
            + [self.cls]
            + list(self.dec.parameters())
            + list(self.norm.parameters())
            + list(self.out.parameters())
        )

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.x_mean) / self.x_sd
        if self.obs_dim < self.in_dim:
            # Scale key coordinates inside the encoder token by key_scale (0.1) so 14 key dims do not swamp 2 obs dims
            z = torch.cat([z[..., : self.obs_dim], self.key_scale * z[..., self.obs_dim :]], dim=-1)
        return z

    def raw_u(self, x: torch.Tensor) -> torch.Tensor:
        z = self.normalize_x(x)
        raw = self.enc(z) + 0.5 * self.enc_skip(z)
        return (raw - self.u_offset) / self.u_scale

    @torch.no_grad()
    def calibrate_full_entropy(self, x_cal: torch.Tensor) -> None:
        """Centers and unit-scales `u` on calibration tokens so all `2^b` FSQ symbols start at full entropy."""
        z = self.normalize_x(x_cal)
        raw = (self.enc(z) + 0.5 * self.enc_skip(z)).reshape(-1, self.b)
        self.u_offset.copy_(raw.mean(dim=0))
        self.u_scale.copy_(raw.std(dim=0).clamp_min(1e-2))

    def encode_fsq(
        self,
        x: torch.Tensor,
        quant_alpha: float = 1.0,
        compute_usage_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantizes via `FSQ` (`2309.15505`) with staged homotopy weight `quant_alpha in [0, 1]`."""
        u = self.raw_u(x)
        # Continuous bounded value in (-1, +1)^b
        h_cont = torch.tanh(u)
        # FSQ L=2 per coordinate: z_tilde in (0, 1), rounded to {0, 1} with STE in training, exact z_hard at eval
        z_tilde = 0.5 * (1.0 + h_cont)
        z_hard = torch.round(z_tilde).clamp(0.0, 1.0)
        syms = (z_hard.long() * self.bit_weights).sum(dim=-1)
        z_ste = z_tilde + (z_hard - z_tilde).detach() if self.training else z_hard
        m_fsq = 2.0 * z_ste - 1.0

        if not self.training or quant_alpha >= 1.0:
            m = m_fsq
        elif quant_alpha <= 0.0:
            m = h_cont
        else:
            m = (1.0 - quant_alpha) * h_cont + quant_alpha * m_fsq

        usage_loss = torch.zeros((), dtype=x.dtype, device=x.device)
        if compute_usage_loss:
            p1 = torch.sigmoid(2.0 * u).clamp(1e-6, 1.0 - 1e-6)  # (B, N, b)
            log_p1 = torch.log(p1)
            log_p0 = torch.log(1.0 - p1)
            log_p_corners = torch.einsum("mnd,kd->mnk", log_p1, self.corners) + torch.einsum(
                "mnd,kd->mnk", log_p0, 1.0 - self.corners
            )
            p_corners = torch.exp(log_p_corners)
            p_bar = p_corners.reshape(-1, self.K).mean(dim=0).clamp_min(1e-8)
            kl_to_unif = math.log(float(self.K)) + (p_bar * torch.log(p_bar)).sum()
            cond_ent = -(p_corners * log_p_corners).sum(dim=-1).mean()
            scale_reg = (u.pow(2).mean() - 1.0).pow(2)
            usage_loss = kl_to_unif + 0.08 * cond_ent + 0.05 * scale_reg

        return m, syms, usage_loss

    def decode(self, m: torch.Tensor, keys: torch.Tensor | None = None) -> torch.Tensor:
        if self.dec_keys:
            assert keys is not None
            kk = self.key_scale * (keys - self.k_mean) / self.k_sd
            h_in = torch.cat([m, kk if self.use_keys else torch.zeros_like(kk)], dim=-1)
        else:
            h_in = m
        h = self.inp(h_in)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        o = self.out(self.norm(self.dec(h)[:, 0]))
        return o[:, 0]


def run_z9_lf_uncompressed_anchor(
    decoder_name: str,
    stages: list[tuple[Z9Spec, int]],
    seed: int,
    lf_spec: Z9Spec,
    X_val: torch.Tensor,
    K_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    K_dv: torch.Tensor,
    bayes_dv: np.ndarray,
    X_cal: torch.Tensor,
    K_cal: torch.Tensor,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> dict[str, Any]:
    """Trains a same-steps uncompressed reader (`isab`, `transformer`, or `deepsets`) on `Z9-p2-lf` (`best_val_kl` checkpoint)."""
    device = X_val.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    F_dim = X_val.shape[-1]
    Fk_dim = K_val.shape[-1]
    net = build_parity_net(F_dim, Fk_dim, None, "none", decoder_name, False).to(device)
    with torch.no_grad():
        flat_x = X_cal.reshape(-1, F_dim)
        flat_k = K_cal.reshape(-1, Fk_dim)
        net.x_mean.copy_(flat_x.mean(dim=0))
        net.x_sd.copy_(flat_x.std(dim=0).clamp_min(1e-2))
        net.k_mean.copy_(flat_k.mean(dim=0))
        net.k_sd.copy_(flat_k.std(dim=0).clamp_min(1e-2))

    @torch.no_grad()
    def eval_split(X_s: torch.Tensor, K_s: torch.Tensor, bayes_s: np.ndarray, l0_s: float) -> tuple[float, np.ndarray]:
        net.eval()
        out = []
        for s in range(0, len(X_s), 2048):
            xb = X_s[s : s + 2048]
            kb = K_s[s : s + 2048]
            out.append(net.decode(net.message(xb, "train"), kb)[0].double().cpu().numpy())
        net.train()
        q = l0_s + np.concatenate(out, axis=0)
        kl_v = _kl(q, bayes_s)
        return float(kl_v.mean()), kl_v

    total_steps = sum(n_s for _, n_s in stages)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_steps)
    gen = torch.Generator(device=device).manual_seed(seed * 10007 + 31)

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    t = 0
    chunk_len = 500
    for stage_spec, n_stage_steps in stages:
        l0 = prior_logit(stage_spec)
        is_final = stage_spec == lf_spec
        x_chunk: torch.Tensor | None = None
        k_chunk: torch.Tensor | None = None
        y_chunk: torch.Tensor | None = None
        for s_step in range(n_stage_steps):
            t_mod = s_step % chunk_len
            if t_mod == 0:
                c_steps = min(chunk_len, n_stage_steps - s_step)
                xc, kc, yc = sample_z9_torch(stage_spec, c_steps * batch_size, device, gen)
                x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
                k_chunk = kc.reshape(c_steps, batch_size, kc.shape[1], kc.shape[2])
                y_chunk = yc.reshape(c_steps, batch_size)
            assert x_chunk is not None and k_chunk is not None and y_chunk is not None
            xb = x_chunk[t_mod]
            kb = k_chunk[t_mod]
            yb = y_chunk[t_mod]
            lg = l0 + net.decode(net.message(xb, "train"), kb)[0]
            loss = F.binary_cross_entropy_with_logits(lg, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if (t + 1) % eval_every == 0 or (t + 1) == total_steps:
                v_kl, _ = eval_split(X_val, K_val, bayes_val, prior_logit(lf_spec))
                if is_final and v_kl < best_val_kl:
                    best_val_kl = v_kl
                    best_state = copy.deepcopy(net.state_dict())
                    best_step = t + 1
            t += 1

    if best_state is not None:
        net.load_state_dict(best_state)
    _, kl_dv = eval_split(X_dv, K_dv, bayes_dv, prior_logit(lf_spec))
    reg = summarize_values(kl_dv, n_boot)
    return {
        "decoder": decoder_name,
        "seed": seed,
        "steps": total_steps,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "regret": reg,
    }


@torch.no_grad()
def eval_model_z9(
    net: FSQJointRemedyNet,
    X: torch.Tensor,
    K: torch.Tensor,
    bayes: np.ndarray,
    prior_l0: float,
    b: int,
    check_channel: bool = False,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluates strictly on discrete `b`-bit `FSQ` messages (`quant_alpha = 1.0`)."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        net.eval()
        preds = []
        syms_list = []
        for s in range(0, len(X), 2048):
            xb = X[s : s + 2048]
            kb = K[s : s + 2048]
            m, syms, _ = net.encode_fsq(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = net.decode(m, kb)
            preds.append(lg.double().cpu().numpy())
            syms_list.append(syms.cpu().numpy())
        net.train()
        q = prior_l0 + np.concatenate(preds, axis=0)
        all_syms = np.concatenate(syms_list, axis=0)
        kl_vals = _kl(q, bayes)
        sym_metrics = compute_symbol_metrics(all_syms, b=b) if check_channel else {}
        return float(kl_vals.mean()), kl_vals, sym_metrics
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_z9_lf_arm(
    arm_name: str,
    b: int,
    seed: int,
    lf_spec: Z9Spec,
    X_val: torch.Tensor,
    K_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    K_dv: torch.Tensor,
    bayes_dv: np.ndarray,
    X_cal: torch.Tensor,
    K_cal: torch.Tensor,
    prior_kl: float,
    same_stage_anchor_regret: float,
    best_uncompressed_regret: float,
    steps: int = 8_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> dict[str, Any]:
    """Runs one joint-learning remedy arm on leak-reduced `Z9-p2` (`LF_SPEC`) at bit budget `b`."""
    device = X_val.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    dec_keys = not arm_name.endswith("_keys0")
    use_full_ent_init = arm_name in (
        "a_fsq_alt_opt",
        "b_fsq_dec_warmstart",
        "c_fsq_entropy_bonus",
        "d_fsq_staged_channel",
        "e_fsq_biased_curr_plus_best",
        "e_fsq_biased_curr_plus_best_keys0",
    )
    net = FSQJointRemedyNet(
        in_dim=X_val.shape[-1],
        key_dim=K_val.shape[-1] if dec_keys else 0,
        b=b,
        dec_keys=dec_keys,
        key_scale=0.1,
        obs_dim=2,
        full_entropy_init=use_full_ent_init,
    ).to(device)

    with torch.no_grad():
        flat_x = X_cal.reshape(-1, X_cal.shape[-1])
        flat_k = K_cal.reshape(-1, K_cal.shape[-1])
        net.x_mean.copy_(flat_x.mean(dim=0))
        net.x_sd.copy_(flat_x.std(dim=0).clamp_min(1e-2))
        if dec_keys:
            net.k_mean.copy_(flat_k.mean(dim=0))
            net.k_sd.copy_(flat_k.std(dim=0).clamp_min(1e-2))
        if use_full_ent_init:
            net.calibrate_full_entropy(X_cal[:1024])

    is_biased_curr = arm_name in ("e_fsq_biased_curr_plus_best", "e_fsq_biased_curr_plus_best_keys0")
    if is_biased_curr:
        stages = biased_bit_curriculum(lf_spec, steps=(steps // 4, steps // 4, steps - 2 * (steps // 4)), probs=(0.8, 0.65))
    else:
        stages = [(lf_spec, steps)]

    total_steps = sum(n_s for _, n_s in stages)
    # Per #3894 Must-Fix 5: key_warmup = 1000 steps whenever dec_keys = True
    key_warmup = (min(1000, total_steps // 4) if total_steps < 2000 else 1000) if dec_keys else 0

    if arm_name == "a_fsq_alt_opt":
        opt_enc = torch.optim.AdamW(net.encoder_parameters(), lr=lr, weight_decay=1e-4)
        opt_dec = torch.optim.AdamW(net.decoder_parameters(), lr=lr, weight_decay=1e-4)
        sched_enc = torch.optim.lr_scheduler.OneCycleLR(opt_enc, max_lr=lr, total_steps=max(2, total_steps // 2))
        sched_dec = torch.optim.lr_scheduler.OneCycleLR(opt_dec, max_lr=lr, total_steps=max(2, total_steps // 2))
    else:
        opt_all = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched_all = torch.optim.lr_scheduler.OneCycleLR(opt_all, max_lr=lr, total_steps=total_steps)

    gen = torch.Generator(device=device).manual_seed(seed * 10007 + 99)
    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    escape_step: int | None = None
    k_alt = max(1, min(200, total_steps // 10))
    warm_steps = int(0.30 * total_steps)
    t_float_end = int(0.35 * total_steps)
    t_anneal_end = int(0.75 * total_steps)

    t = 0
    chunk_len = 500
    for stage_spec, n_stage_steps in stages:
        l0 = prior_logit(stage_spec)
        is_final_stage = stage_spec == lf_spec
        x_chunk: torch.Tensor | None = None
        k_chunk: torch.Tensor | None = None
        y_chunk: torch.Tensor | None = None
        for s_step in range(n_stage_steps):
            t_mod = s_step % chunk_len
            if t_mod == 0:
                c_steps = min(chunk_len, n_stage_steps - s_step)
                xc, kc, yc = sample_z9_torch(stage_spec, c_steps * batch_size, device, gen)
                x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
                k_chunk = kc.reshape(c_steps, batch_size, kc.shape[1], kc.shape[2])
                y_chunk = yc.reshape(c_steps, batch_size)
            assert x_chunk is not None and k_chunk is not None and y_chunk is not None
            xb = x_chunk[t_mod]
            kb = k_chunk[t_mod]
            yb = y_chunk[t_mod]
            net.use_keys = (t >= key_warmup) if dec_keys else False

            # Single-factor isolation: usage_w > 0 ONLY in (c) and (e*)
            if arm_name in ("0_fsq_joint", "a_fsq_alt_opt", "b_fsq_dec_warmstart"):
                quant_alpha = 1.0
                usage_w = 0.0
            elif arm_name == "c_fsq_entropy_bonus":
                quant_alpha = 1.0
                usage_w = 0.10
            elif arm_name == "d_fsq_staged_channel":
                if t < t_float_end:
                    quant_alpha = 0.0
                elif t < t_anneal_end:
                    prog = (t - t_float_end) / max(1, t_anneal_end - t_float_end)
                    quant_alpha = 0.5 * (1.0 - math.cos(math.pi * prog))
                else:
                    quant_alpha = 1.0
                usage_w = 0.0
            elif is_biased_curr:
                if not is_final_stage:
                    prog = t / max(1, total_steps // 2)
                    quant_alpha = 0.25 + 0.55 * prog
                else:
                    prog_final = (t - total_steps // 2) / max(1, total_steps // 4)
                    quant_alpha = min(1.0, 0.80 + 0.20 * max(0.0, prog_final))
                usage_w = 0.08
            else:
                raise ValueError(f"Unknown arm: {arm_name}")

            if arm_name == "a_fsq_alt_opt":
                dec_phase = (t // k_alt) % 2 == 0
                for p in net.encoder_parameters():
                    p.requires_grad_(not dec_phase)
                for p in net.decoder_parameters():
                    p.requires_grad_(dec_phase)
                m, _, _ = net.encode_fsq(xb, quant_alpha=quant_alpha, compute_usage_loss=False)
                lg = l0 + net.decode(m, kb)
                loss = F.binary_cross_entropy_with_logits(lg, yb)
                if dec_phase:
                    opt_dec.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.decoder_parameters(), 1.0)
                    opt_dec.step()
                    sched_dec.step()
                else:
                    opt_enc.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.encoder_parameters(), 1.0)
                    opt_enc.step()
                    sched_enc.step()
            elif arm_name == "b_fsq_dec_warmstart":
                in_warm = t < warm_steps
                for p in net.encoder_parameters():
                    p.requires_grad_(not in_warm)
                m, _, _ = net.encode_fsq(xb, quant_alpha=1.0, compute_usage_loss=False)
                lg = l0 + net.decode(m, kb)
                loss = F.binary_cross_entropy_with_logits(lg, yb)
                opt_all.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt_all.step()
                sched_all.step()
            else:
                m, _, u_loss = net.encode_fsq(xb, quant_alpha=quant_alpha, compute_usage_loss=(usage_w > 0))
                lg = l0 + net.decode(m, kb)
                loss = F.binary_cross_entropy_with_logits(lg, yb) + usage_w * u_loss
                opt_all.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt_all.step()
                sched_all.step()

            if (t + 1) % eval_every == 0 or (t + 1) == total_steps:
                val_kl, _, _ = eval_model_z9(net, X_val, K_val, bayes_val, prior_logit(lf_spec), b=b, check_channel=False)
                if is_final_stage and escape_step is None and val_kl < 0.9 * prior_kl:
                    escape_step = t + 1
                # Never pick a keys-off checkpoint when dec_keys=True, and only pick discrete-mode checkpoints on final stage
                keys_ready = (not dec_keys) or net.use_keys
                if is_final_stage and keys_ready and (quant_alpha >= 0.999 or (t + 1) == total_steps) and val_kl < best_val_kl:
                    best_val_kl = val_kl
                    best_state = copy.deepcopy(net.state_dict())
                    best_step = t + 1
            t += 1

    net.use_keys = dec_keys
    for p in net.parameters():
        p.requires_grad_(True)
    if best_state is not None:
        net.load_state_dict(best_state)

    # For (d) and (e*), run a brief 400-step discrete decoder lock-in on frozen discrete FSQ symbols (B8)
    if arm_name in ("d_fsq_staged_channel", "e_fsq_biased_curr_plus_best", "e_fsq_biased_curr_plus_best_keys0"):
        ft_steps = min(400, max(10, total_steps // 10))
        for p in net.encoder_parameters():
            p.requires_grad_(False)
        opt_ft = torch.optim.AdamW(net.decoder_parameters(), lr=5e-4, weight_decay=1e-4)
        l0_final = prior_logit(lf_spec)
        xc_ft, kc_ft, yc_ft = sample_z9_torch(lf_spec, ft_steps * batch_size, device, gen)
        x_ft = xc_ft.reshape(ft_steps, batch_size, xc_ft.shape[1], xc_ft.shape[2])
        k_ft = kc_ft.reshape(ft_steps, batch_size, kc_ft.shape[1], kc_ft.shape[2])
        y_ft = yc_ft.reshape(ft_steps, batch_size)
        for ft_i in range(ft_steps):
            xb, kb, yb = x_ft[ft_i], k_ft[ft_i], y_ft[ft_i]
            with torch.no_grad():
                m, _, _ = net.encode_fsq(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = l0_final + net.decode(m, kb)
            loss_ft = F.binary_cross_entropy_with_logits(lg, yb)
            opt_ft.zero_grad()
            loss_ft.backward()
            torch.nn.utils.clip_grad_norm_(net.decoder_parameters(), 1.0)
            opt_ft.step()
        val_kl_ft, _, _ = eval_model_z9(net, X_val, K_val, bayes_val, prior_logit(lf_spec), b=b, check_channel=False)
        if val_kl_ft < best_val_kl:
            best_val_kl = val_kl_ft
            if escape_step is None and val_kl_ft < 0.9 * prior_kl:
                escape_step = total_steps
        elif best_state is not None:
            net.load_state_dict(best_state)
        for p in net.parameters():
            p.requires_grad_(True)

    _, kl_dv, sym_metrics = eval_model_z9(net, X_dv, K_dv, bayes_dv, prior_logit(lf_spec), b=b, check_channel=True)
    regret_summary = summarize_values(kl_dv, n_boot)
    gap_to_same_stage = float(regret_summary["mean"] - same_stage_anchor_regret)
    gap_to_best_unc = float(regret_summary["mean"] - best_uncompressed_regret)

    return {
        "arm": arm_name,
        "b": b,
        "seed": seed,
        "dec_keys": dec_keys,
        "key_scale": 0.1 if dec_keys else 0.0,
        "key_warmup": key_warmup,
        "regret_quant": regret_summary,
        "gap_to_anchor": gap_to_same_stage,
        "gap_to_best_uncompressed": gap_to_best_unc,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "escape_step": escape_step,
        "escaped": escape_step is not None,
        "symbol_metrics": sym_metrics,
    }


def run_e2_uncompressed_anchor(
    curr_mode: str,
    seed: int,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    dv_batch: Any,
    steps: int = 4_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> dict[str, Any]:
    """Trains the same-data (`n_train = 100,000`, `4000` steps) uncompressed Transformer anchor on `E2` (`std` or `curr`)."""
    device = X_tr.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    in_dim = X_tr.shape[-1]
    width = 64
    net = nn.ModuleDict(
        {
            "inp": nn.Linear(in_dim, width),
            "dec": nn.TransformerEncoder(
                nn.TransformerEncoderLayer(width, 4, 4 * width, 0.0, batch_first=True, norm_first=True),
                2,
                enable_nested_tensor=False,
            ),
            "norm": nn.LayerNorm(width),
            "out": nn.Linear(width, 1),
        }
    ).to(device)
    cls_tok = nn.Parameter(0.02 * torch.randn(1, 1, width, device=device))
    params = list(net.parameters()) + [cls_tok]
    x_mean = X_tr[:20_000].reshape(-1, in_dim).mean(dim=0)
    x_sd = X_tr[:20_000].reshape(-1, in_dim).std(dim=0).clamp_min(1e-2)

    def forward_unc(xb: torch.Tensor) -> torch.Tensor:
        z = (xb - x_mean) / x_sd
        h = net["inp"](z)
        h = torch.cat([cls_tok.expand(len(h), -1, -1), h], dim=1)
        return net["out"](net["norm"](net["dec"](h)[:, 0]))[:, 0]

    @torch.no_grad()
    def eval_split(X_s: torch.Tensor, bayes_s: np.ndarray) -> tuple[float, np.ndarray]:
        net.eval()
        preds = []
        for s in range(0, len(X_s), 2048):
            preds.append(forward_unc(X_s[s : s + 2048]).double().cpu().numpy())
        net.train()
        q = np.concatenate(preds, axis=0)
        kl_v = _kl(q, bayes_s)
        return float(kl_v.mean()), kl_v

    abs_ev = X_tr[:, :, -1].abs().mean(dim=1)
    q50 = float(torch.quantile(abs_ev[:20_000], 0.50).item())
    q25 = float(torch.quantile(abs_ev[:20_000], 0.25).item())
    high_idx = torch.nonzero(abs_ev >= q50, as_tuple=False).squeeze(-1).cpu().numpy()
    mid_idx = torch.nonzero(abs_ev >= q25, as_tuple=False).squeeze(-1).cpu().numpy()

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best_val_kl = float("inf")
    best_state = None
    best_cls = None
    best_step = -1
    n_train = len(X_tr)

    for t in range(steps):
        if curr_mode == "curr" and t < steps // 4 and len(high_idx) >= batch_size:
            pick = high_idx[rng.integers(0, len(high_idx), batch_size)]
            idx = torch.as_tensor(pick, device=device)
        elif curr_mode == "curr" and t < steps // 2 and len(mid_idx) >= batch_size:
            pick = mid_idx[rng.integers(0, len(mid_idx), batch_size)]
            idx = torch.as_tensor(pick, device=device)
        else:
            idx = torch.as_tensor(rng.integers(0, n_train, batch_size), device=device)
        lg = forward_unc(X_tr[idx])
        loss = F.binary_cross_entropy_with_logits(lg, Y_tr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (t + 1) % eval_every == 0 or (t + 1) == steps:
            v_kl, _ = eval_split(X_val, bayes_val)
            if v_kl < best_val_kl:
                best_val_kl = v_kl
                best_state = copy.deepcopy(net.state_dict())
                best_cls = cls_tok.detach().clone()
                best_step = t + 1

    if best_state is not None and best_cls is not None:
        net.load_state_dict(best_state)
        with torch.no_grad():
            cls_tok.copy_(best_cls)
    _, kl_dv = eval_split(X_dv, np.asarray(dv_batch.bayes["logit"], dtype=np.float64))
    reg = summarize_values(kl_dv, n_boot)
    return {
        "mode": curr_mode,
        "seed": seed,
        "steps": steps,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "regret": reg,
    }


@torch.no_grad()
def eval_model_e2(
    net: FSQJointRemedyNet,
    X: torch.Tensor,
    bayes: np.ndarray,
    b: int,
    check_channel: bool = False,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        net.eval()
        preds = []
        syms_list = []
        for s in range(0, len(X), 2048):
            xb = X[s : s + 2048]
            m, syms, _ = net.encode_fsq(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = net.decode(m, None)
            preds.append(lg.double().cpu().numpy())
            syms_list.append(syms.cpu().numpy())
        net.train()
        q = np.concatenate(preds, axis=0)
        all_syms = np.concatenate(syms_list, axis=0)
        kl_vals = _kl(q, bayes)
        sym_metrics = compute_symbol_metrics(all_syms, b=b) if check_channel else {}
        return float(kl_vals.mean()), kl_vals, sym_metrics
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_e2_arm(
    arm_name: str,
    b: int,
    seed: int,
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    dv_batch: Any,
    prior_kl: float,
    same_stage_anchor_regret: float,
    best_uncompressed_regret: float,
    steps: int = 4_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> dict[str, Any]:
    """Runs one joint-learning remedy arm on `E2` at bit budget `b`."""
    device = X_tr.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    use_full_ent_init = arm_name in (
        "a_fsq_alt_opt",
        "b_fsq_dec_warmstart",
        "c_fsq_entropy_bonus",
        "d_fsq_staged_channel",
        "e_fsq_biased_curr_plus_best",
    )
    net = FSQJointRemedyNet(
        in_dim=X_tr.shape[-1],
        key_dim=0,
        b=b,
        dec_keys=False,
        key_scale=0.1,
        obs_dim=X_tr.shape[-1],
        full_entropy_init=use_full_ent_init,
    ).to(device)

    with torch.no_grad():
        flat_x = X_tr[:20_000].reshape(-1, X_tr.shape[-1])
        net.x_mean.copy_(flat_x.mean(dim=0))
        net.x_sd.copy_(flat_x.std(dim=0).clamp_min(1e-2))
        if use_full_ent_init:
            net.calibrate_full_entropy(X_tr[:1024])

    if arm_name == "a_fsq_alt_opt":
        opt_enc = torch.optim.AdamW(net.encoder_parameters(), lr=lr, weight_decay=1e-4)
        opt_dec = torch.optim.AdamW(net.decoder_parameters(), lr=lr, weight_decay=1e-4)
        sched_enc = torch.optim.lr_scheduler.OneCycleLR(opt_enc, max_lr=lr, total_steps=max(2, steps // 2))
        sched_dec = torch.optim.lr_scheduler.OneCycleLR(opt_dec, max_lr=lr, total_steps=max(2, steps // 2))
    else:
        opt_all = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched_all = torch.optim.lr_scheduler.OneCycleLR(opt_all, max_lr=lr, total_steps=steps)

    abs_ev = X_tr[:, :, -1].abs().mean(dim=1)
    q50 = float(torch.quantile(abs_ev[:20_000], 0.50).item())
    q25 = float(torch.quantile(abs_ev[:20_000], 0.25).item())
    high_idx = torch.nonzero(abs_ev >= q50, as_tuple=False).squeeze(-1).cpu().numpy()
    mid_idx = torch.nonzero(abs_ev >= q25, as_tuple=False).squeeze(-1).cpu().numpy()

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    escape_step: int | None = None
    k_alt = max(1, min(100, steps // 10))
    warm_steps = int(0.30 * steps)
    t_float_end = int(0.35 * steps)
    t_anneal_end = int(0.75 * steps)
    n_train = len(X_tr)

    for t in range(steps):
        if arm_name == "e_fsq_biased_curr_plus_best" and t < steps // 4 and len(high_idx) >= batch_size:
            pick = high_idx[rng.integers(0, len(high_idx), batch_size)]
            idx = torch.as_tensor(pick, device=device)
        elif arm_name == "e_fsq_biased_curr_plus_best" and t < steps // 2 and len(mid_idx) >= batch_size:
            pick = mid_idx[rng.integers(0, len(mid_idx), batch_size)]
            idx = torch.as_tensor(pick, device=device)
        else:
            idx = torch.as_tensor(rng.integers(0, n_train, batch_size), device=device)

        xb = X_tr[idx]
        yb = Y_tr[idx]

        # Single-factor isolation: usage_w > 0 ONLY in (c) and (e)
        if arm_name in ("0_fsq_joint", "a_fsq_alt_opt", "b_fsq_dec_warmstart"):
            quant_alpha = 1.0
            usage_w = 0.0
        elif arm_name == "c_fsq_entropy_bonus":
            quant_alpha = 1.0
            usage_w = 0.08
        elif arm_name == "d_fsq_staged_channel":
            if t < t_float_end:
                quant_alpha = 0.0
            elif t < t_anneal_end:
                prog = (t - t_float_end) / max(1, t_anneal_end - t_float_end)
                quant_alpha = 0.5 * (1.0 - math.cos(math.pi * prog))
            else:
                quant_alpha = 1.0
            usage_w = 0.0
        elif arm_name == "e_fsq_biased_curr_plus_best":
            if t < steps // 2:
                prog = t / max(1, steps // 2)
                quant_alpha = 0.30 + 0.50 * prog
            else:
                prog_final = (t - steps // 2) / max(1, steps // 4)
                quant_alpha = min(1.0, 0.80 + 0.20 * max(0.0, prog_final))
            usage_w = 0.08
        else:
            raise ValueError(f"Unknown arm: {arm_name}")

        if arm_name == "a_fsq_alt_opt":
            dec_phase = (t // k_alt) % 2 == 0
            for p in net.encoder_parameters():
                p.requires_grad_(not dec_phase)
            for p in net.decoder_parameters():
                p.requires_grad_(dec_phase)
            m, _, _ = net.encode_fsq(xb, quant_alpha=quant_alpha, compute_usage_loss=False)
            lg = net.decode(m, None)
            loss = F.binary_cross_entropy_with_logits(lg, yb)
            if dec_phase:
                opt_dec.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.decoder_parameters(), 1.0)
                opt_dec.step()
                sched_dec.step()
            else:
                opt_enc.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.encoder_parameters(), 1.0)
                opt_enc.step()
                sched_enc.step()
        elif arm_name == "b_fsq_dec_warmstart":
            in_warm = t < warm_steps
            for p in net.encoder_parameters():
                p.requires_grad_(not in_warm)
            m, _, _ = net.encode_fsq(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = net.decode(m, None)
            loss = F.binary_cross_entropy_with_logits(lg, yb)
            opt_all.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt_all.step()
            sched_all.step()
        else:
            m, _, u_loss = net.encode_fsq(xb, quant_alpha=quant_alpha, compute_usage_loss=(usage_w > 0))
            lg = net.decode(m, None)
            loss = F.binary_cross_entropy_with_logits(lg, yb) + usage_w * u_loss
            opt_all.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt_all.step()
            sched_all.step()

        if (t + 1) % eval_every == 0 or (t + 1) == steps:
            val_kl, _, _ = eval_model_e2(net, X_val, bayes_val, b=b, check_channel=False)
            if escape_step is None and val_kl < 0.9 * prior_kl:
                escape_step = t + 1
            if (quant_alpha >= 0.999 or (t + 1) == steps) and val_kl < best_val_kl:
                best_val_kl = val_kl
                best_state = copy.deepcopy(net.state_dict())
                best_step = t + 1

    for p in net.parameters():
        p.requires_grad_(True)
    if best_state is not None:
        net.load_state_dict(best_state)

    _, kl_dv, sym_metrics = eval_model_e2(net, X_dv, np.asarray(dv_batch.bayes["logit"], dtype=np.float64), b=b, check_channel=True)
    regret_summary = summarize_values(kl_dv, n_boot)
    gap_to_same_stage = float(regret_summary["mean"] - same_stage_anchor_regret)
    gap_to_best_unc = float(regret_summary["mean"] - best_uncompressed_regret)

    return {
        "arm": arm_name,
        "b": b,
        "seed": seed,
        "regret_quant": regret_summary,
        "gap_to_anchor": gap_to_same_stage,
        "gap_to_best_uncompressed": gap_to_best_unc,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "escape_step": escape_step,
        "escaped": escape_step is not None,
        "symbol_metrics": sym_metrics,
    }


def aggregate_remedy_seeds(
    records: list[dict[str, Any]],
    plain_tr_regret: float | None = None,
    isab_regret: float | None = None,
) -> dict[str, Any]:
    means = [float(r["regret_quant"]["mean"]) for r in records]
    gaps = [float(r["gap_to_anchor"]) for r in records]
    gaps_best = [float(r["gap_to_best_uncompressed"]) for r in records]
    ci_lows = [float(r["regret_quant"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret_quant"]["ci_high"]) for r in records]
    escs = [r["escape_step"] for r in records]
    b_steps = [int(r.get("best_step", -1)) for r in records]
    h_agents = [float(r["symbol_metrics"]["mean_agent_entropy_bits"]) for r in records]
    h_pools = [float(r["symbol_metrics"]["pooled_entropy_bits"]) for r in records]
    d_syms = [int(r["symbol_metrics"]["distinct_symbols_total"]) for r in records]
    reg_mean = float(np.mean(means))
    out: dict[str, Any] = {
        "n_seeds": len(records),
        "regret_mean": reg_mean,
        "regret_sd": float(np.std(means)),
        "gap_to_anchor_mean": float(np.mean(gaps)),
        "gap_to_anchor_sd": float(np.std(gaps)),
        "gap_to_best_uncompressed_mean": float(np.mean(gaps_best)),
        "gap_to_best_uncompressed_sd": float(np.std(gaps_best)),
        "mean_ci_low": float(np.mean(ci_lows)),
        "mean_ci_high": float(np.mean(ci_highs)),
        "per_seed_regret": [round(m, 4) for m in means],
        "per_seed_gap_to_anchor": [round(g, 4) for g in gaps],
        "per_seed_best_step": b_steps,
        "n_escaped": int(sum(1 for e in escs if e is not None)),
        "per_seed_escape": escs,
        "mean_agent_entropy_bits": float(np.mean(h_agents)),
        "mean_agent_entropy_sd": float(np.std(h_agents)),
        "per_seed_agent_entropy": [round(h, 4) for h in h_agents],
        "pooled_entropy_bits_mean": float(np.mean(h_pools)),
        "per_seed_distinct_symbols": d_syms,
    }
    if plain_tr_regret is not None:
        out["gap_to_plain_tr_mean"] = float(reg_mean - plain_tr_regret)
    if isab_regret is not None:
        out["gap_to_isab_mean"] = float(reg_mean - isab_regret)
    return out


def aggregate_anchor_seeds(records: list[dict[str, Any]]) -> dict[str, Any]:
    means = [float(r["regret"]["mean"]) for r in records]
    ci_lows = [float(r["regret"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret"]["ci_high"]) for r in records]
    b_steps = [int(r.get("best_step", -1)) for r in records]
    return {
        "n_seeds": len(records),
        "regret_mean": float(np.mean(means)),
        "regret_sd": float(np.std(means)),
        "mean_ci_low": float(np.mean(ci_lows)),
        "mean_ci_high": float(np.mean(ci_highs)),
        "per_seed_regret": [round(m, 4) for m in means],
        "per_seed_best_step": b_steps,
    }


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
    allowed: set[str] = {"0.0000", "2.0000", "4.0000", "6.0000"}
    collect_4dec_strings(summary, allowed)
    found = re.findall(r"\b\d+\.\d{4}\b", readme_text)
    missing = [x for x in found if x not in allowed]
    return len(found), missing


def render_readme(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Joint-Learning Remedies with `FSQ` on Leak-Reduced `Z9-p2` (`b481850`) and `E2` at `b in {2, 4, 6}` (`#3893`, `#3894` & `#3896`)")
    lines.append("")
    lines.append("**Track: `EXPLORATORY` (`gate must-fixes applied per #3894 & #3896`)**")
    lines.append("")
    lines.append("Tests remedies for the joint encoder-decoder chicken-and-egg problem at `b in {2, 4, 6}` bits/agent (`2^b in {4, 16, 64}` symbols, `3` seeds `0, 1, 2`) with `FSQ` (`2309.15505`) as the quantizer, `key_scale = 0.1` (`in_norm=False`), and `key_warmup = 1000` steps (`best_val_kl` restricted to `t >= key_warmup`) whenever an arm takes keys (`dec_keys = True`):")
    lines.append("- `0_fsq_joint`: Plain joint end-to-end training with `FSQ` (`usage_w = 0.0`, `Y labels only`).")
    lines.append("- `a_fsq_alt_opt`: `(a)` Alternating optimization (`k_alt` steps decoder with encoder frozen, `k_alt` steps encoder with decoder frozen; `usage_w = 0.0`, `Y labels only`).")
    lines.append("- `b_fsq_dec_warmstart`: `(b)` Decoder warm-start on a random fixed full-entropy `FSQ` code (first 30% of steps with encoder frozen), then joint end-to-end training (`usage_w = 0.0`, `Y labels only`).")
    lines.append("- `c_fsq_entropy_bonus`: `(c)` Encoder marginal symbol-entropy / usage bonus (`L_usage`, `usage_w = 0.10` on `Z9-p2-lf`, `0.08` on `E2`, `Y labels only`).")
    lines.append("- `d_fsq_staged_channel`: `(d)` Staged continuous-to-discrete channel (`tanh(u)` float message first, cosine-annealed quantization `alpha(t): 0 -> 1` to `b`-bit `FSQ`, then pure discrete `b`-bit `FSQ` + discrete decoder lock-in; `usage_w = 0.0`, `Y labels only`).")
    lines.append("- `e_fsq_biased_curr_plus_best`: `(e)` Biased-bit curriculum on `Z9-p2-lf` (**`structure-informed (simulator-side: sets latent-bit priors to 0.8 -> 0.65 -> 0.5 and uses k in prior_logit)`**, **not** `Y labels only`) or high-to-normal piece-SNR curriculum on `E2`, combined with the best of `(a)-(d)` (`dec_keys = True`, `STRUCTURE-KNOWN (decoder reads keys['full'] outside b)` on `Z9-p2-lf`, `key_scale = 0.1`, `key_warmup = 1000`).")
    lines.append("- `e_fsq_biased_curr_plus_best_keys0`: `(e)` on `Z9-p2-lf` with `dec_keys = False` (**`BLIND DECODER (messages only; curriculum still simulator-side structure-informed)`**), compared directly against the blind 4-bit searched reference (`0.2160`) and the blind `5.49`-bit `bit_id+reading` reference (`0.0060`).")
    lines.append("")
    lines.append("### Scope & Interpretation Notes (`#3896` Must-Fixes 1–5)")
    lines.append("- **Structure-Known vs. Blind Decoder (`#3896` Must-Fixes 1 & 2):** On `Z9-p2-lf`, every keyed arm (`dec_keys = True`) is **`STRUCTURE-KNOWN (decoder reads keys['full'] outside b)`**: the `14`-float per-agent bit-identity and accuracy key `keys['full']` is passed directly to the decoder outside the `b`-bit channel (`FSQJointRemedyNet.decode`: `h_in = torch.cat([m, kk], dim=-1)`), so the `b`-bit bottleneck only compresses the agent's `2` binary readings. At `b = 2`, those `2` binary readings fit **losslessly** into `2` bits (the `4` reading pairs `(0,0),(0,1),(1,0),(1,1)`), so `0.0085` at `b = 2` (`+0.0018` vs plain 8k Transformer `0.0068`, `+0.0065` vs 8k ISAB `0.0020`) **only shows that `FSQ` learns the 2-bit identity map and tests nothing about compression**.")
    lines.append("- **Blind Decoder Headline (`#3896` Must-Fix 2):** On `Z9-p2-lf`, the blind arm (`e_fsq_biased_curr_plus_best_keys0`, `BLIND DECODER (messages only; curriculum still simulator-side structure-informed)`) reaches `0.2601` at `b = 2` (`0/3` escaped), `0.2188` at `b = 4` (tying the searched 4-bit blind code `0.2160`), and `0.0813` at `b = 6` (**`+0.0793` behind 8k ISAB `0.0020` and `+0.0745` behind plain 8k Transformer `0.0068`**, whereas the blind hand code `bit_id+reading` reaches `0.0060` at `5.49` bits — so **Kelvin's blind challenge is not met**).")
    lines.append("- **Spec & Channel Comparison Note (`#3896` Must-Fix 5):** The `0/3` escape result for `(iii)` in `tt-compression-decoupling-20261022` ran on the **old `Z9-p2` spec (`prior = 0.4567`, deterministic `2` readers/bit + `6` distractors = `7` agents)** with **Gaussian / Gumbel channels**, whereas this table runs on **leak-reduced `Z9-p2-lf` (`prior = 0.2702`, `Dirichlet` random assignment = `13` agents)** with **`FSQ`**. Comparing them does **not** isolate \"FSQ fixes joint learning\" because the spec and channel are confounded.")
    lines.append("")

    for env_name in ["Z9-p2-lf", "E2"]:
        if env_name not in summary["envs"]:
            continue
        ed = summary["envs"][env_name]
        anc = ed["anchors"]
        lines.append(f"## Environment: `{env_name}`")
        lines.append("")
        if env_name == "Z9-p2-lf":
            lines.append(
                "- **Spec:** `Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)` (`13` agents, `2` readings/agent, fresh episodes every step, `key_scale = 0.1`, `key_warmup = 1000` when `dec_keys = True`)"
            )
            lines.append(
                f"- **Same-Steps (`8000` steps) Uncompressed Anchors & Controls (`n_dev = 4000`, `3` seeds `0, 1, 2`; note `best_step = [8000, 8000, 8000]` means the Z9 anchors were still improving at 8k steps so all gaps are lower bounds):**\n"
                f"  - `prior (b=0)` = `{anc['prior_kl']:.4f}`\n"
                f"  - `anchor_isab_8k_std (b=inf, Y labels only)` = `{anc['anchor_isab_8k_std']['regret_mean']:.4f}` ± `{anc['anchor_isab_8k_std']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_isab_8k_std'].get('per_seed_best_step', [8000, 8000, 8000])}`)\n"
                f"  - `anchor_isab_8k_biased_curr (b=inf, structure-informed curriculum)` = `{anc['anchor_isab_8k_biased_curr']['regret_mean']:.4f}` ± `{anc['anchor_isab_8k_biased_curr']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_isab_8k_biased_curr'].get('per_seed_best_step', [8000, 8000, 8000])}`)\n"
                f"  - `anchor_transformer_8k_std (b=inf, Y labels only)` = `{anc['anchor_transformer_8k_std']['regret_mean']:.4f}` ± `{anc['anchor_transformer_8k_std']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_transformer_8k_std'].get('per_seed_best_step', [7600, 8000, 8000])}`)\n"
                f"  - `anchor_transformer_8k_biased_curr (b=inf, structure-informed curriculum)` = `{anc['anchor_transformer_8k_biased_curr']['regret_mean']:.4f}` ± `{anc['anchor_transformer_8k_biased_curr']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_transformer_8k_biased_curr'].get('per_seed_best_step', [8000, 8000, 8000])}`)\n"
                f"  - `control_deepsets_8k_std (known-incapable reader control)` = `{anc['control_deepsets_8k_std']['regret_mean']:.4f}` ± `{anc['control_deepsets_8k_std']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['control_deepsets_8k_std'].get('per_seed_best_step', [400, 7200, 1600])}`)\n"
                f"  - `blind_refs`: `searched b=2` = `{anc['blind_refs']['b2']:.4f}`, `searched b=4` = `{anc['blind_refs']['b4']:.4f}`, `bit_id+reading (5.49b)` = `{anc['blind_refs']['bit_id_pair']:.4f}`"
            )
            lines.append("")
            lines.append(
                "| `b` | Remedy Arm | Decoder / Curriculum Provenance | Mean Regret (`regret_quant`) | `gap(b)` vs Same-Stage Tr | `gap(b)` vs Plain 8k Tr (`0.0068`) | `gap(b)` vs 8k ISAB (`0.0020`) | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Escaped (`val_kl < 0.9 * prior`) | Escape Steps (`s0, s1, s2`) | Mean Agent Entropy `H(M_i)` (bits) | Distinct Symbols (`s0, s1, s2`) |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for b_str, by_arm in ed["aggregated"].items():
                for arm_name, ag in by_arm.items():
                    s_regs = ", ".join(f"`{v:.4f}`" for v in ag["per_seed_regret"])
                    s_escs = ", ".join(f"`{v}`" if v is not None else "`None`" for v in ag["per_seed_escape"])
                    s_syms = ", ".join(f"`{v}`" for v in ag["per_seed_distinct_symbols"])
                    if arm_name.endswith("_keys0"):
                        prov_str = "`BLIND DECODER (messages only; curriculum still simulator-side structure-informed)`"
                    elif arm_name.startswith("e_fsq_biased_curr"):
                        prov_str = "`STRUCTURE-KNOWN (decoder reads keys['full'] outside b) + structure-informed (simulator-side curriculum)`"
                    else:
                        prov_str = "`STRUCTURE-KNOWN (decoder reads keys['full'] outside b), Y labels only`"
                    gap_plain_tr = ag.get("gap_to_plain_tr_mean", ag["gap_to_anchor_mean"])
                    gap_isab = ag.get("gap_to_isab_mean", ag["gap_to_best_uncompressed_mean"])
                    lines.append(
                        f"| `{b_str}` | `{arm_name}` | {prov_str} | `{ag['regret_mean']:.4f}` | `{ag['gap_to_anchor_mean']:+.4f}` | `{gap_plain_tr:+.4f}` | `{gap_isab:+.4f}` | "
                        f"`{ag['regret_sd']:.4f}` | `[`{ag['mean_ci_low']:.4f}`, `{ag['mean_ci_high']:.4f}`]` | "
                        f"{s_regs} | `{ag['n_escaped']}/{ag['n_seeds']}` | {s_escs} | "
                        f"`{ag['mean_agent_entropy_bits']:.4f} ± {ag['mean_agent_entropy_sd']:.4f}` | {s_syms} |"
                    )
            lines.append("")
        else:
            lines.append(
                "- **Scope & Weak-Anchor Note (`#3896` Must-Fix 3):** In this table, `E2` was **not** trained on fresh episodes (`n_train = 100,000` fixed pool, `4000 x 512` steps $\\approx 20$ epochs), and its agent-token uncompressed Transformer anchor (`0.0415`) is weak (worse than blind closed-form `extremized_q` at `b = 6`, `0.0403`, and far above the per-piece bidirectional Transformer on fresh episodes, which reaches `0.0026`). At `b = 6`, `0_fsq_joint` (`0.0459`, 95% CI `[0.0421, 0.0498]`, gap `+0.0043` vs `0.0415`) lies above `0.0415` and does **not** beat the weak anchor. (See `results/explore/tt-compression-e2-e3-z8-e4-20261022/` for fresh-episode training against strong per-piece anchors.)"
            )
            lines.append(
                f"- **Same-Data (`n_train = 100,000` pool, `4000` steps) Uncompressed Anchors & Closed-Form References (`n_dev = 2000`, `3` seeds `0, 1, 2`):**\n"
                f"  - `prior (b=0)` = `{anc['prior_kl']:.4f}`\n"
                f"  - `anchor_transformer_4k_std (b=inf, static pool)` = `{anc['anchor_transformer_4k_std']['regret_mean']:.4f}` ± `{anc['anchor_transformer_4k_std']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_transformer_4k_std'].get('per_seed_best_step', [1000, 1000, 1000])}`)\n"
                f"  - `anchor_transformer_4k_curr (b=inf, static pool)` = `{anc['anchor_transformer_4k_curr']['regret_mean']:.4f}` ± `{anc['anchor_transformer_4k_curr']['regret_sd']:.4f}` (`per_seed_best_step` = `{anc['anchor_transformer_4k_curr'].get('per_seed_best_step', [1200, 1200, 400])}`)\n"
                f"  - `blind extremized_q (b=2,4,6)` = `{anc['extremized_q']['2']:.4f}`, `{anc['extremized_q']['4']:.4f}`, `{anc['extremized_q']['6']:.4f}`\n"
                f"  - `llr_quantizer [PROVENANCE FULL] (b=2,4,6)` = `{anc['llr_quantizer']['2']:.4f}`, `{anc['llr_quantizer']['4']:.4f}`, `{anc['llr_quantizer']['6']:.4f}`"
            )
            lines.append("")
            lines.append(
                "| `b` | Remedy Arm | Mean Regret (`regret_quant`) | `gap(b)` to Same-Stage Anchor | `gap(b)` to Static-Pool Anchor (`0.0415`) | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Escaped (`val_kl < 0.9 * prior`) | Escape Steps (`s0, s1, s2`) | Mean Agent Entropy `H(M_i)` (bits) | Distinct Symbols (`s0, s1, s2`) |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for b_str, by_arm in ed["aggregated"].items():
                for arm_name, ag in by_arm.items():
                    s_regs = ", ".join(f"`{v:.4f}`" for v in ag["per_seed_regret"])
                    s_escs = ", ".join(f"`{v}`" if v is not None else "`None`" for v in ag["per_seed_escape"])
                    s_syms = ", ".join(f"`{v}`" for v in ag["per_seed_distinct_symbols"])
                    lines.append(
                        f"| `{b_str}` | `{arm_name}` | `{ag['regret_mean']:.4f}` | `{ag['gap_to_anchor_mean']:+.4f}` | `{ag['gap_to_best_uncompressed_mean']:+.4f}` | "
                        f"`{ag['regret_sd']:.4f}` | `[`{ag['mean_ci_low']:.4f}`, `{ag['mean_ci_high']:.4f}`]` | "
                        f"{s_regs} | `{ag['n_escaped']}/{ag['n_seeds']}` | {s_escs} | "
                        f"`{ag['mean_agent_entropy_bits']:.4f} ± {ag['mean_agent_entropy_sd']:.4f}` | {s_syms} |"
                    )
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["Z9-p2-lf", "E2"])
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "0_fsq_joint",
            "a_fsq_alt_opt",
            "b_fsq_dec_warmstart",
            "c_fsq_entropy_bonus",
            "d_fsq_staged_channel",
            "e_fsq_biased_curr_plus_best",
        ],
    )
    parser.add_argument("--steps-z9", type=int, default=8_000)
    parser.add_argument("--steps-e2", type=int, default=4_000)
    parser.add_argument("--n-train-e2", type=int, default=100_000)
    parser.add_argument("--n-val-e2", type=int, default=1_000)
    parser.add_argument("--n-dev-e2", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--merge-dirs", nargs="+", default=None, help="Merge summary.json from per-env/per-bit directories")
    parser.add_argument("--reuse-z9-from", type=str, default=None, help="Reuse Z9-p2-lf anchors from an existing summary.json")
    parser.add_argument("--out-dir", type=str, default="results/explore/tt-compression-joint-remedies-20261022")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_dirs:
        merged: dict[str, Any] | None = None
        total_elapsed = 0.0
        part_harnesses: list[dict[str, Any]] = []
        for d_str in args.merge_dirs:
            p_path = pathlib.Path(d_str)
            if p_path.is_dir():
                p_path = p_path / "summary.json"
            part = json.loads(p_path.read_text())
            total_elapsed += float(part.get("elapsed_sec", 0.0))
            if "harness" in part:
                part_harnesses.append({"source": str(p_path), "harness": part["harness"]})
            if merged is None:
                merged = part
            else:
                for env_k, env_v in part.get("envs", {}).items():
                    if env_k not in merged["envs"]:
                        merged["envs"][env_k] = env_v
                    else:
                        merged["envs"][env_k]["anchors"].update(env_v.get("anchors", {}))
                        for b_k, b_v in env_v.get("aggregated", {}).items():
                            merged["envs"][env_k]["aggregated"].setdefault(b_k, {}).update(b_v)
                        for b_k, b_v in env_v.get("per_seed", {}).items():
                            merged["envs"][env_k]["per_seed"].setdefault(b_k, {}).update(b_v)
        assert merged is not None
        merged["bits"] = [2, 4, 6]
        merged["elapsed_sec"] = round(total_elapsed, 2)
        merged["part_harnesses"] = part_harnesses
        (out_dir / "summary.json").write_text(json.dumps(merged, indent=2) + "\n")
        readme_text = render_readme(merged)
        n_checked, missing = verify_readme_numbers(readme_text, merged)
        if missing:
            raise RuntimeError(f"Unverified 4-decimal numbers in README.md: {missing}")
        (out_dir / "README.md").write_text(readme_text)
        print(f"Merged into {out_dir}; verified {n_checked}/{n_checked} 4-dec numbers.", flush=True)
        return

    if args.smoke:
        args.steps_z9 = 20
        args.steps_e2 = 20
        args.n_train_e2 = 1_000
        args.n_val_e2 = 100
        args.n_dev_e2 = 100
        args.batch_size = 64
        args.eval_every = 10
        args.n_boot = 50
        args.seeds = [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": "aggbench.compression-joint-remedies-fsq/v2",
        "harness": harness_info(),
        "track": "EXPLORATORY",
        "bits": args.bits,
        "seeds": args.seeds,
        "arms": args.arms,
        "key_scale": 0.1,
        "key_warmup": 1000 if not args.smoke else 5,
        "device": str(device),
        "envs": {},
    }

    t_start = time.time()
    for env_name in args.envs:
        if env_name == "Z9-p2-lf":
            lf_spec = Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)
            n_val = 200 if args.smoke else 2_000
            n_dev = 200 if args.smoke else 4_000
            val_d = sample_z9_np(lf_spec, n_val, np.random.default_rng(VAL_SEED))
            dev_d = sample_z9_np(lf_spec, n_dev, np.random.default_rng(DEV_SEED))
            cal_d = sample_z9_np(lf_spec, 4096, np.random.default_rng([STATS_SEED, 0]))

            X_val = torch.tensor(val_d["tokens"], dtype=torch.float32, device=device)
            K_val = torch.tensor(val_d["keys"], dtype=torch.float32, device=device)
            bayes_val = np.asarray(val_d["bayes"], dtype=np.float64)

            X_dv = torch.tensor(dev_d["tokens"], dtype=torch.float32, device=device)
            K_dv = torch.tensor(dev_d["keys"], dtype=torch.float32, device=device)
            bayes_dv = np.asarray(dev_d["bayes"], dtype=np.float64)

            X_cal = torch.tensor(cal_d["tokens"], dtype=torch.float32, device=device)
            K_cal = torch.tensor(cal_d["keys"], dtype=torch.float32, device=device)

            prior_kl = float(_kl(np.full(len(bayes_dv), dev_d["prior_logit"]), bayes_dv).mean())
            std_stages = [(lf_spec, args.steps_z9)]
            curr_stages = biased_bit_curriculum(
                lf_spec,
                steps=(args.steps_z9 // 4, args.steps_z9 // 4, args.steps_z9 - 2 * (args.steps_z9 // 4)),
                probs=(0.8, 0.65),
            )

            # Train same-steps (8000 steps) uncompressed ISAB, Transformer, and DeepSets controls (#3894 Must-Fix 4)
            z9_anc_specs = [
                ("anchor_isab_8k_std", "isab", std_stages),
                ("anchor_isab_8k_biased_curr", "isab", curr_stages),
                ("anchor_transformer_8k_std", "transformer", std_stages),
                ("anchor_transformer_8k_biased_curr", "transformer", curr_stages),
                ("control_deepsets_8k_std", "deepsets", std_stages),
            ]
            z9_anc_out: dict[str, Any] = {}
            reuse_z9_anchors = None
            if args.reuse_z9_from:
                rz_data = json.loads(pathlib.Path(args.reuse_z9_from).read_text())
                reuse_z9_anchors = rz_data.get("envs", {}).get("Z9-p2-lf", {}).get("anchors", {})
            for a_key, dec_name, a_stages in z9_anc_specs:
                if reuse_z9_anchors and a_key in reuse_z9_anchors:
                    z9_anc_out[a_key] = reuse_z9_anchors[a_key]
                    print(
                        f"[Z9-p2-lf anchor] {a_key:32s} REUSED: regret={z9_anc_out[a_key]['regret_mean']:.4f}",
                        flush=True,
                    )
                    continue
                a_recs = []
                for seed in args.seeds:
                    t_a = time.time()
                    ar = run_z9_lf_uncompressed_anchor(
                        decoder_name=dec_name,
                        stages=a_stages,
                        seed=seed,
                        lf_spec=lf_spec,
                        X_val=X_val,
                        K_val=K_val,
                        bayes_val=bayes_val,
                        X_dv=X_dv,
                        K_dv=K_dv,
                        bayes_dv=bayes_dv,
                        X_cal=X_cal,
                        K_cal=K_cal,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        eval_every=args.eval_every,
                        n_boot=args.n_boot,
                    )
                    a_recs.append(ar)
                    print(
                        f"[Z9-p2-lf anchor] {a_key:32s} seed={seed}: regret={ar['regret']['mean']:.4f} "
                        f"best_step={ar['best_step']} ({time.time() - t_a:.1f}s)",
                        flush=True,
                    )
                z9_anc_out[a_key] = aggregate_anchor_seeds(a_recs)

            best_uncompressed_regret = float(
                min(
                    z9_anc_out["anchor_isab_8k_std"]["regret_mean"],
                    z9_anc_out["anchor_isab_8k_biased_curr"]["regret_mean"],
                )
            )
            anchors = {
                "prior_kl": prior_kl,
                "best_uncompressed_name": "ISAB (8k steps on Z9-p2-lf)",
                "best_uncompressed_regret": best_uncompressed_regret,
                **z9_anc_out,
                "blind_refs": {"b2": 0.2540, "b4": 0.2160, "bit_id_pair": 0.0060},
            }

            z9_arms = list(args.arms)
            if "e_fsq_biased_curr_plus_best" in z9_arms and "e_fsq_biased_curr_plus_best_keys0" not in z9_arms:
                z9_arms.append("e_fsq_biased_curr_plus_best_keys0")

            per_seed_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
            aggregated: dict[str, dict[str, Any]] = {}

            for b in args.bits:
                b_key = str(b)
                per_seed_records[b_key] = {}
                aggregated[b_key] = {}
                for arm_name in z9_arms:
                    per_seed_records[b_key][arm_name] = []
                    same_stage_anc = (
                        z9_anc_out["anchor_transformer_8k_biased_curr"]["regret_mean"]
                        if arm_name.startswith("e_fsq_biased_curr")
                        else z9_anc_out["anchor_transformer_8k_std"]["regret_mean"]
                    )
                    for seed in args.seeds:
                        t0 = time.time()
                        rec = run_z9_lf_arm(
                            arm_name=arm_name,
                            b=b,
                            seed=seed,
                            lf_spec=lf_spec,
                            X_val=X_val,
                            K_val=K_val,
                            bayes_val=bayes_val,
                            X_dv=X_dv,
                            K_dv=K_dv,
                            bayes_dv=bayes_dv,
                            X_cal=X_cal,
                            K_cal=K_cal,
                            prior_kl=prior_kl,
                            same_stage_anchor_regret=same_stage_anc,
                            best_uncompressed_regret=best_uncompressed_regret,
                            steps=args.steps_z9,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            eval_every=args.eval_every,
                            n_boot=args.n_boot,
                        )
                        per_seed_records[b_key][arm_name].append(rec)
                        print(
                            f"[Z9-p2-lf b={b}] {arm_name:34s} seed={seed}: regret={rec['regret_quant']['mean']:.4f} "
                            f"gap_stage={rec['gap_to_anchor']:+.4f} gap_best={rec['gap_to_best_uncompressed']:+.4f} "
                            f"esc={rec['escape_step']} H={rec['symbol_metrics']['mean_agent_entropy_bits']:.4f}b "
                            f"syms={rec['symbol_metrics']['distinct_symbols_total']}/{2**b} ({time.time()-t0:.1f}s)",
                            flush=True,
                        )
                    aggregated[b_key][arm_name] = aggregate_remedy_seeds(
                        per_seed_records[b_key][arm_name],
                        plain_tr_regret=z9_anc_out["anchor_transformer_8k_std"]["regret_mean"],
                        isab_regret=z9_anc_out["anchor_isab_8k_std"]["regret_mean"],
                    )
                    summary["envs"]["Z9-p2-lf"] = {
                        "anchors": anchors,
                        "aggregated": aggregated,
                        "per_seed": per_seed_records,
                    }
                    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

            summary["envs"]["Z9-p2-lf"] = {
                "anchors": anchors,
                "aggregated": aggregated,
                "per_seed": per_seed_records,
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        elif env_name == "E2":
            env = make_cenv("E2")
            tr = env.sample("train", args.n_train_e2, start=0)
            val = env.sample("dev", args.n_val_e2, start=50_000)
            dv = env.sample("dev", args.n_dev_e2, start=0)

            X_tr = torch.tensor(agent_tokens(tr), dtype=torch.float32, device=device)
            Y_tr = torch.tensor(np.asarray(tr.truth, np.float32), dtype=torch.float32, device=device)
            X_val = torch.tensor(agent_tokens(val), dtype=torch.float32, device=device)
            bayes_val = np.asarray(val.bayes["logit"], dtype=np.float64)
            X_dv = torch.tensor(agent_tokens(dv), dtype=torch.float32, device=device)

            prior_kl = float(summarize(prior_forecast(dv), dv, n_boot=args.n_boot)["kl"]["mean"])

            # Train same-data uncompressed Transformer anchors on E2 (standard & SNR-curriculum)
            e2_anc_out: dict[str, Any] = {}
            for a_key, c_mode in [("anchor_transformer_4k_std", "std"), ("anchor_transformer_4k_curr", "curr")]:
                a_recs = []
                for seed in args.seeds:
                    t_a = time.time()
                    ar = run_e2_uncompressed_anchor(
                        curr_mode=c_mode,
                        seed=seed,
                        X_tr=X_tr,
                        Y_tr=Y_tr,
                        X_val=X_val,
                        bayes_val=bayes_val,
                        X_dv=X_dv,
                        dv_batch=dv,
                        steps=args.steps_e2,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        eval_every=args.eval_every,
                        n_boot=args.n_boot,
                    )
                    a_recs.append(ar)
                    print(
                        f"[E2 anchor] {a_key:28s} seed={seed}: regret={ar['regret']['mean']:.4f} "
                        f"best_step={ar['best_step']} ({time.time() - t_a:.1f}s)",
                        flush=True,
                    )
                e2_anc_out[a_key] = aggregate_anchor_seeds(a_recs)

            best_uncompressed_e2 = float(
                min(
                    e2_anc_out["anchor_transformer_4k_std"]["regret_mean"],
                    e2_anc_out["anchor_transformer_4k_curr"]["regret_mean"],
                )
            )
            cf_res = run_closed_form(
                "E2",
                bits=tuple(args.bits),
                n_train=min(2000, args.n_train_e2),
                n_dev=args.n_dev_e2,
                M=128 if args.smoke else 512,
                n_boot=args.n_boot,
                verbose=False,
                arms=("llr_quantizer", "extremized_q"),
            )
            ext_q_refs = {str(b): float(cf_res["arms"]["extremized_q"][str(b)]["regret"]["mean"]) for b in args.bits}
            llr_q_refs = {str(b): float(cf_res["arms"]["llr_quantizer"][str(b)]["regret"]["mean"]) for b in args.bits}

            anchors = {
                "prior_kl": prior_kl,
                "best_uncompressed_name": "same_data_full_context_transformer (4k steps)",
                "best_uncompressed_regret": best_uncompressed_e2,
                **e2_anc_out,
                "extremized_q": ext_q_refs,
                "llr_quantizer": llr_q_refs,
            }

            per_seed_records = {}
            aggregated = {}
            for b in args.bits:
                b_key = str(b)
                per_seed_records[b_key] = {}
                aggregated[b_key] = {}
                for arm_name in args.arms:
                    per_seed_records[b_key][arm_name] = []
                    same_stage_anc = (
                        e2_anc_out["anchor_transformer_4k_curr"]["regret_mean"]
                        if arm_name == "e_fsq_biased_curr_plus_best"
                        else e2_anc_out["anchor_transformer_4k_std"]["regret_mean"]
                    )
                    for seed in args.seeds:
                        t0 = time.time()
                        rec = run_e2_arm(
                            arm_name=arm_name,
                            b=b,
                            seed=seed,
                            X_tr=X_tr,
                            Y_tr=Y_tr,
                            X_val=X_val,
                            bayes_val=bayes_val,
                            X_dv=X_dv,
                            dv_batch=dv,
                            prior_kl=prior_kl,
                            same_stage_anchor_regret=same_stage_anc,
                            best_uncompressed_regret=best_uncompressed_e2,
                            steps=args.steps_e2,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            eval_every=args.eval_every,
                            n_boot=args.n_boot,
                        )
                        per_seed_records[b_key][arm_name].append(rec)
                        print(
                            f"[E2 b={b}] {arm_name:28s} seed={seed}: regret={rec['regret_quant']['mean']:.4f} "
                            f"gap_stage={rec['gap_to_anchor']:+.4f} gap_best={rec['gap_to_best_uncompressed']:+.4f} "
                            f"esc={rec['escape_step']} H={rec['symbol_metrics']['mean_agent_entropy_bits']:.4f}b "
                            f"syms={rec['symbol_metrics']['distinct_symbols_total']}/{2**b} ({time.time()-t0:.1f}s)",
                            flush=True,
                        )
                    aggregated[b_key][arm_name] = aggregate_remedy_seeds(
                        per_seed_records[b_key][arm_name],
                        plain_tr_regret=e2_anc_out["anchor_transformer_4k_std"]["regret_mean"],
                    )

            summary["envs"]["E2"] = {
                "anchors": anchors,
                "aggregated": aggregated,
                "per_seed": per_seed_records,
            }

    summary["elapsed_sec"] = round(time.time() - t_start, 2)
    summary_path = out_dir / "summary.json"
    readme_path = out_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if "Z9-p2-lf" in summary["envs"] and "E2" in summary["envs"] and set(args.bits) >= {2, 4, 6}:
        readme_text = render_readme(summary)
        n_checked, missing = verify_readme_numbers(readme_text, summary)
        if missing:
            raise RuntimeError(f"Unverified 4-decimal numbers in README.md: {missing}")
        readme_path.write_text(readme_text)
        print(f"Done in {summary['elapsed_sec']}s; verified {n_checked}/{n_checked} 4-dec numbers in {readme_path}.", flush=True)
    else:
        print(f"Saved partial summary in {summary['elapsed_sec']}s to {summary_path}.", flush=True)


if __name__ == "__main__":
    main()
