#!/usr/bin/env python3
"""Step 6 Diagnostic (#3889): FSQ (2309.15505) & VQ with Usage Regularization + Dead-Code Reset (1711.00937, 2305.08842).

Tests quantizers designed to use all 2^b = 4 code cells at b = 2 bits/agent on `E2` and `Z8`
across 3 seeds (0, 1, 2), label-only (`n_train = 100,000`, `n_dev = 2,000`, `steps = 4,000`),
logging per-agent and pooled symbol entropy (in bits, max = 2.0000) and distinct symbols alongside
`regret_quant` and `steps_to_escape`:

  1. `gauss_noisy_ste` (Baseline v2 Gaussian channel: tanh + N(0, 1/3) in train, sign at eval)
  2. `gumbel_st` (Baseline v2 Gumbel-Softmax straight-through over 2^b = 4 symbols)
  3. `fsq_1d_4level` (Mentzer et al. 2309.15505 1D 4-level Finite Scalar Quantization `L=[4]`:
     monotone scalar rounding into 4 ordered levels `{-1, -1/3, +1/3, +1}` with STE)
  4. `fsq_2d_2x2` (Mentzer et al. 2309.15505 2D binary FSQ `L=[2, 2]` with STE + usage entropy term)
  5. `vq_ema_reset` (Vector Quantizer with `K = 4` codewords, EMA updates, dead-code reset when
     code usage < 5%, and batch marginal entropy regularization `log(4) - H(p_bar)`)
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.compression import (
    ESCAPE_FRAC,
    GAUSS_SIGMA2,
    Channel,
    _kl_logits,
    agent_tokens,
    gain,
    gumbel_code,
    llr_quantizer_forecast,
    make_cenv,
    run_closed_form,
    tune_llr_quantizer,
)
from aggbench.metrics import summarize
from aggbench.report import achievable_floor, prior_forecast
from aggbench.results import harness_info


def compute_symbol_entropy_stats(syms: np.ndarray, b: int = 2) -> dict[str, Any]:
    """Computes pooled and per-agent symbol entropy in bits (max = b bits) from integer symbols `(n, N)` in `[0, 2^b)`."""
    Channel(b).send(syms)
    K = 2 ** b
    flat = syms.ravel()
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
        "min_agent_entropy_bits": float(np.min(h_agents)),
        "max_agent_entropy_bits": float(np.max(h_agents)),
        "symbol_hist": [round(float(x), 4) for x in p_all.tolist()],
    }


class QuantizedCompressionNet(nn.Module):
    """Shared encoder + set decoder supporting matched controls (`noiseless_sign_ste`, `*_no_usage`), `FSQ`, `VQ-EMA`, and uncompressed anchor."""

    def __init__(
        self,
        in_dim: int,
        b: int = 2,
        quantizer: str = "fsq_1d_4level",
        width: int = 64,
        enc_hidden: int = 128,
        layers: int = 2,
        heads: int = 4,
        vq_dim: int = 8,
    ) -> None:
        super().__init__()
        assert b == 2, "Step 6 diagnostic targets b = 2 bits/agent (4 symbols)"
        self.in_dim = in_dim
        self.b = b
        self.K = 2 ** b
        self.quantizer = quantizer
        self.sigma = math.sqrt(GAUSS_SIGMA2)
        self.vq_dim = vq_dim

        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))

        if quantizer in ("gauss_noisy_ste", "noiseless_sign_ste"):
            enc_out = b
            dec_in = b
        elif quantizer == "gumbel_st":
            enc_out = self.K
            dec_in = self.K
        elif quantizer in ("fsq_1d_4level", "fsq_1d_4level_no_usage"):
            enc_out = 1
            dec_in = 1
        elif quantizer in ("fsq_2d_2x2", "fsq_2d_2x2_no_usage"):
            enc_out = 2
            dec_in = 2
        elif quantizer == "vq_ema_reset":
            enc_out = vq_dim
            dec_in = vq_dim
        elif quantizer == "uncompressed":
            enc_out = in_dim
            dec_in = in_dim
        else:
            raise ValueError(f"Unknown quantizer: {quantizer}")

        if quantizer == "uncompressed":
            self.enc = None
            self.llr_skip = None
        else:
            self.enc = nn.Sequential(
                nn.Linear(in_dim, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, enc_hidden),
                nn.GELU(),
                nn.Linear(enc_hidden, enc_out),
            )
            # Skip from own local log-odds (last token feature) for 1D/2D scalar quantizers
            self.llr_skip = nn.Linear(1, enc_out)

        if quantizer == "vq_ema_reset":
            # Initialize 4 distinct codewords on unit sphere in R^vq_dim
            init_cb = F.normalize(torch.randn(self.K, vq_dim), dim=-1)
            self.register_buffer("codebook", init_cb)
            self.register_buffer("ema_count", torch.ones(self.K))
            self.register_buffer("ema_weight", init_cb.clone())

        self.emb = nn.Linear(dec_in, width)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    def _raw_enc(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_mean) / self.x_sd
        if self.quantizer == "uncompressed":
            return xn
        assert self.enc is not None and self.llr_skip is not None
        return self.enc(xn) + 0.5 * self.llr_skip(xn[..., -1:])

    def encode_and_quantize(
        self,
        x: torch.Tensor,
        mode: str = "train",
        tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(m_for_decoder, int_symbols, aux_reg_loss)`."""
        u = self._raw_enc(x)
        zero_loss = torch.zeros((), dtype=x.dtype, device=x.device)

        if self.quantizer == "uncompressed":
            syms = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.long, device=x.device)
            return u, syms, zero_loss

        if self.quantizer == "gauss_noisy_ste":
            s = torch.tanh(u)
            b0 = (u[..., 0] >= 0).long()
            b1 = (u[..., 1] >= 0).long()
            syms = b0 * 2 + b1
            if mode == "quant":
                m = torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))
            else:
                m = s + self.sigma * torch.randn_like(s)
            return m, syms, zero_loss

        if self.quantizer == "noiseless_sign_ste":
            # Matched control (#3894 Must-Fix 3): tanh(u) + noiseless sign STE, zero usage loss
            # (isolates removing the B8 train/score mismatch in gauss_noisy_ste)
            s = torch.tanh(u)
            b0 = (u[..., 0] >= 0).long()
            b1 = (u[..., 1] >= 0).long()
            syms = b0 * 2 + b1
            s_hard = torch.where(s >= 0, torch.ones_like(s), -torch.ones_like(s))
            m = s_hard if mode == "quant" else (s + (s_hard - s).detach())
            return m, syms, zero_loss

        if self.quantizer == "gumbel_st":
            syms = u.argmax(dim=-1)
            if mode == "train":
                m = gumbel_code(u, tau=tau, hard=True)
            else:
                m = F.one_hot(syms, self.K).float()
            return m, syms, zero_loss

        if self.quantizer in ("fsq_1d_4level", "fsq_1d_4level_no_usage"):
            # Mentzer et al. (2309.15505) L = 4 levels in {0, 1, 2, 3}:
            z_tilde = 1.5 * (1.0 + torch.tanh(u / 1.5))  # (B, N, 1) in (0, 3)
            z_hard = torch.round(z_tilde).clamp(0.0, 3.0)
            syms = z_hard[..., 0].long()
            z_ste = z_tilde if mode == "noisy" else (z_tilde + (z_hard - z_tilde).detach())
            m = (z_ste - 1.5) / 1.5
            if self.quantizer == "fsq_1d_4level_no_usage":
                return m, syms, zero_loss
            centers = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=x.dtype, device=x.device)
            p_soft = F.softmax(-2.0 * (z_tilde - centers) ** 2, dim=-1)  # (B, N, 4)
            p_bar = p_soft.reshape(-1, 4).mean(dim=0).clamp_min(1e-8)
            usage_loss = 0.05 * (math.log(4.0) + (p_bar * p_bar.log()).sum())
            return m, syms, usage_loss

        if self.quantizer in ("fsq_2d_2x2", "fsq_2d_2x2_no_usage"):
            # Mentzer et al. (2309.15505) L = [2, 2]: each coordinate in (0, 1) -> rounded to {0, 1} with STE
            z_tilde = 0.5 * (1.0 + torch.tanh(u))  # (B, N, 2) in (0, 1)
            z_hard = torch.round(z_tilde).clamp(0.0, 1.0)
            syms = (z_hard[..., 0].long() * 2) + z_hard[..., 1].long()
            z_ste = z_tilde + (z_hard - z_tilde).detach()
            m = 2.0 * z_ste - 1.0  # in {-1, +1}^2
            if self.quantizer == "fsq_2d_2x2_no_usage":
                return m, syms, zero_loss
            p1 = z_tilde
            p0 = 1.0 - p1
            p_joint = torch.stack(
                [
                    p0[..., 0] * p0[..., 1],
                    p0[..., 0] * p1[..., 1],
                    p1[..., 0] * p0[..., 1],
                    p1[..., 0] * p1[..., 1],
                ],
                dim=-1,
            )
            p_bar = p_joint.reshape(-1, 4).mean(dim=0).clamp_min(1e-8)
            usage_loss = 0.10 * (math.log(4.0) + (p_bar * p_bar.log()).sum())
            return m, syms, usage_loss

        if self.quantizer == "vq_ema_reset":
            # L2-normalized VQ (Huh et al. 2305.08842) + EMA update + dead-code reset + entropy usage term
            z_e = F.normalize(u, dim=-1)  # (B, N, D)
            cb = F.normalize(self.codebook, dim=-1)  # (4, D)
            flat_z = z_e.reshape(-1, self.vq_dim)  # (M, D)
            dists = torch.cdist(flat_z, cb, p=2.0) ** 2  # (M, 4)
            idx = dists.argmin(dim=-1)  # (M,)
            syms = idx.view(x.shape[0], x.shape[1])

            if mode == "train" and self.training:
                with torch.no_grad():
                    oh = F.one_hot(idx, self.K).float()  # (M, 4)
                    batch_cnt = oh.sum(dim=0)  # (4,)
                    batch_sum = oh.t() @ flat_z  # (4, D)
                    decay = 0.98
                    self.ema_count.mul_(decay).add_(batch_cnt, alpha=1.0 - decay)
                    self.ema_weight.mul_(decay).add_(batch_sum, alpha=1.0 - decay)
                    n_tot = self.ema_count.sum().clamp_min(1.0)
                    dead_mask = (self.ema_count / n_tot) < 0.05
                    if dead_mask.any():
                        rand_ids = torch.randint(0, flat_z.shape[0], (self.K,), device=x.device)
                        repl = F.normalize(flat_z[rand_ids] + 0.05 * torch.randn_like(self.codebook), dim=-1)
                        self.ema_weight[dead_mask] = repl[dead_mask] * (n_tot / self.K)
                        self.ema_count[dead_mask] = n_tot / self.K
                    new_cb = F.normalize(self.ema_weight / self.ema_count[:, None].clamp_min(1e-5), dim=-1)
                    self.codebook.copy_(new_cb)

            cb_cur = F.normalize(self.codebook, dim=-1)
            z_q = cb_cur[syms]  # (B, N, D)
            commit_loss = 0.25 * F.mse_loss(z_e, z_q.detach())
            p_soft = F.softmax(-dists / 0.25, dim=-1)
            p_bar = p_soft.mean(dim=0).clamp_min(1e-8)
            entropy_loss = 0.15 * (math.log(float(self.K)) + (p_bar * p_bar.log()).sum())
            m = z_e + (z_q - z_e).detach()
            return m, syms, commit_loss + entropy_loss

        raise ValueError(f"Unhandled quantizer: {self.quantizer}")

    def decode(self, m: torch.Tensor) -> torch.Tensor:
        h = self.emb(m)
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        z = self.norm(self.dec(h)[:, 0])
        return self.out(z)[:, 0]


def run_one_quantizer_arm(
    env_name: str,
    quantizer: str,
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
    b = 2
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    in_dim = X_tr.shape[-1]
    l0 = float(dv_batch.public["prior_logit"])
    prior_kl = _kl_logits(np.full(len(vb_val), l0), vb_val)

    net = QuantizedCompressionNet(in_dim=in_dim, b=b, quantizer=quantizer).to(device)
    net.x_mean.copy_(X_tr[:20_000].reshape(-1, in_dim).mean(0))
    net.x_sd.copy_(X_tr[:20_000].reshape(-1, in_dim).std(0).clamp_min(1e-3))

    def evaluate(X_split: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        net.eval()
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        torch.manual_seed(12345)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(12345)
        preds = []
        syms_list = []
        with torch.no_grad():
            for s in range(0, len(X_split), 1024):
                xb = X_split[s : s + 1024]
                m, syms, _ = net.encode_and_quantize(xb, mode="quant")
                q = l0 + net.decode(m)
                preds.append(q.detach().cpu().numpy().astype(np.float64))
                syms_list.append(syms.detach().cpu().numpy().astype(np.int64))
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        net.train()
        return np.concatenate(preds, axis=0), np.concatenate(syms_list, axis=0)

    fq_untrained, _ = evaluate(X_dv)
    control_untrained = summarize({"logit": fq_untrained}, dv_batch, n_boot=min(n_boot, 500))["kl"]

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    all_idx = torch.randint(0, len(X_tr), (steps, batch_size), generator=gen).to(device)
    bce = F.binary_cross_entropy_with_logits

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    curve: list[dict[str, Any]] = []
    for t in range(steps):
        idx = all_idx[t]
        xb = X_tr[idx]
        yb = Y_tr[idx]
        tau = max(0.5, 1.0 - t / max(steps, 1))
        m, _, reg_loss = net.encode_and_quantize(xb, mode="train", tau=tau)
        logit = l0 + net.decode(m)
        loss = bce(logit, yb) + reg_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if t % eval_every == 0 or t == steps - 1:
            val_fq, val_syms = evaluate(X_val)
            val_kl = _kl_logits(val_fq, vb_val)
            if val_kl < best_val_kl:
                best_val_kl = float(val_kl)
                best_state = copy.deepcopy(net.state_dict())
                best_step = int(t)
            ent_v = compute_symbol_entropy_stats(val_syms, b=b) if quantizer != "uncompressed" else {
                "mean_agent_entropy_bits": 0.0,
                "distinct_symbols_total": 1,
            }
            curve.append(
                {
                    "step": int(t),
                    "val_kl": float(val_kl),
                    "loss": float(loss.item()),
                    "val_entropy_bits": ent_v["mean_agent_entropy_bits"],
                    "val_distinct_symbols": ent_v["distinct_symbols_total"],
                }
            )

    if best_state is not None:
        net.load_state_dict(best_state)

    fq_dv, syms_dv = evaluate(X_dv)
    regret_quant = summarize({"logit": fq_dv}, dv_batch, n_boot=n_boot)["kl"]
    ent_stats = compute_symbol_entropy_stats(syms_dv, b=b) if quantizer != "uncompressed" else {
        "distinct_symbols_total": 1,
        "distinct_symbols_max_agent": 1,
        "distinct_symbols_min_agent": 1,
        "pooled_entropy_bits": 0.0,
        "mean_agent_entropy_bits": 0.0,
        "min_agent_entropy_bits": 0.0,
        "max_agent_entropy_bits": 0.0,
        "symbol_hist": [1.0, 0.0, 0.0, 0.0],
    }
    esc = [c["step"] for c in curve if c["val_kl"] < ESCAPE_FRAC * prior_kl]

    return {
        "env": env_name,
        "b": b if quantizer != "uncompressed" else None,
        "quantizer": quantizer,
        "seed": seed,
        "steps": steps,
        "best_val_kl": best_val_kl,
        "best_step": best_step,
        "steps_to_escape": esc[0] if esc else None,
        "control_untrained": control_untrained,
        "regret_quant": regret_quant,
        "entropy": ent_stats,
        "curve": curve,
        "forecast_quant": fq_dv.tolist(),
    }


def aggregate_quantizer_seeds(records: list[dict[str, Any]], anchor_mean: float | None = None) -> dict[str, Any]:
    means = [float(r["regret_quant"]["mean"]) for r in records]
    ci_lows = [float(r["regret_quant"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret_quant"]["ci_high"]) for r in records]
    h_pooled = [float(r["entropy"]["pooled_entropy_bits"]) for r in records]
    h_agent = [float(r["entropy"]["mean_agent_entropy_bits"]) for r in records]
    h_min = [float(r["entropy"]["min_agent_entropy_bits"]) for r in records]
    d_tot = [int(r["entropy"]["distinct_symbols_total"]) for r in records]
    escapes = [r["steps_to_escape"] for r in records]
    esc_non_null = [int(e) for e in escapes if e is not None]
    out: dict[str, Any] = {
        "n_seeds": len(records),
        "regret_mean": float(np.mean(means)),
        "regret_sd": float(np.std(means, ddof=0)),
        "mean_ci_low": float(np.mean(ci_lows)),
        "mean_ci_high": float(np.mean(ci_highs)),
        "per_seed_regret": [round(m, 6) for m in means],
        "pooled_entropy_bits_mean": float(np.mean(h_pooled)),
        "pooled_entropy_bits_sd": float(np.std(h_pooled, ddof=0)),
        "mean_agent_entropy_bits": float(np.mean(h_agent)),
        "mean_agent_entropy_sd": float(np.std(h_agent, ddof=0)),
        "min_agent_entropy_bits": float(np.mean(h_min)),
        "per_seed_agent_entropy_bits": [round(h, 6) for h in h_agent],
        "distinct_symbols_mean": float(np.mean(d_tot)),
        "per_seed_distinct_symbols": d_tot,
        "n_escaped": len(esc_non_null),
        "per_seed_escape": escapes,
    }
    if anchor_mean is not None:
        out["gap_to_same_data_anchor"] = float(np.mean(means) - anchor_mean)
    return out


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
    allowed: set[str] = {"2.0000", "0.0000"}
    collect_4dec_strings(summary, allowed)
    found = re.findall(r"\b\d+\.\d{4}\b", readme_text)
    missing = [x for x in found if x not in allowed]
    return len(found), missing


def render_readme(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Step 6 Quantizer Symbol-Usage Study (`#3889` & `#3894`: `FSQ`, `VQ-EMA-Reset`, and Matched Controls at `b = 2` on `E2` & `Z8`)")
    lines.append("")
    lines.append("**Track: `EXPLORATORY` (`gate must-fixes applied per #3894`)**")
    lines.append("")
    lines.append("Tests quantizers and matched controls at `b = 2` bits/agent (`2^b = 4` symbols) on `E2` and `Z8` (`3` seeds `0, 1, 2`, label-only, `n_train = 100,000`, `n_dev = 2,000`, `steps = 4,000`, `best_val_kl` checkpoint selection):")
    lines.append("- **Baselines:** `gauss_noisy_ste` (`tanh + N(0, 1/3)` in train, `sign` at eval) and `gumbel_st` (Gumbel-Softmax STE over 4 symbols).")
    lines.append("- **Matched Controls (`#3894` Must-Fix 3 — isolating `B8` train/score mismatch vs rounding geometry vs usage loss):**")
    lines.append("  - `noiseless_sign_ste`: `tanh(u)` + noiseless `sign` STE (`m = s + (sign(s) - s).detach()`) with **zero usage loss** (isolates removing the `N(0, 1/3)` train/score mismatch `B8` in `gauss_noisy_ste`).")
    lines.append("  - `fsq_1d_4level_no_usage`: 1D 4-level `FSQ` (`L=[4]`) with **zero usage loss**.")
    lines.append("  - `fsq_2d_2x2_no_usage`: 2D binary `FSQ` (`L=[2, 2]`) with **zero usage loss**.")
    lines.append("- **Usage-Regularized Quantizers:** `fsq_1d_4level` (`L=[4]` + soft usage loss), `fsq_2d_2x2` (`L=[2, 2]` + soft usage loss), and `vq_ema_reset` (`K=4` EMA codebook + dead-code reset + usage entropy loss).")
    lines.append("- **Scope & Statistical-Tie Note (`#3894` Must-Fix 3):** On `Z8`, `FSQ` vs blind `extremized_q` is a **statistical tie** within the 95% bootstrap CI shown below, and on `E2`, `FSQ` narrows the gap to `extremized_q` but does **not** beat `extremized_q`. All `gap(b=2)` values below are scored against the **same-data (`n_train = 100,000`, `4000`-step) uncompressed Transformer anchor** (`same_data_full_context_transformer`).")
    lines.append("")
    for env_name in summary["envs"]:
        ed = summary["envs"][env_name]
        anc = ed["anchors"]
        lines.append(f"## Environment: `{env_name}` (`b = 2` bits/agent, `2^b = 4` symbols)")
        lines.append("")
        lines.append(
            f"- **Anchors (`n_dev = 2000`):** `prior (b=0)` = `{anc['prior']['mean']:.4f}`, "
            f"`same_data_full_context_transformer (n_train=100k, 4000 steps, b=inf)` = `{anc['same_data_full_context_transformer']['regret_mean']:.4f}`, "
            f"`llr_quantizer [PROVENANCE FULL] (b=2)` = `{anc['llr_quantizer_b2']['mean']:.4f}`, "
            f"`extremized_q (blind, b=2)` = `{anc['extremized_q_b2']['mean']:.4f}` "
            f"(CI `[`{anc['extremized_q_b2']['ci_low']:.4f}`, `{anc['extremized_q_b2']['ci_high']:.4f}`]`), "
            f"`run_closed_form 2k-train full_context_transformer` = `{anc['full_context_transformer_regret']:.4f}`"
        )
        lines.append("")
        lines.append("| Quantizer Arm | Mean Regret (`regret_quant`) | Gap to Same-Data Anchor | Seed SD | Mean 95% CI | Per-Seed Regrets (`s0, s1, s2`) | Mean Agent Entropy `H(M_i)` (bits / `2.0000`) | Pooled Entropy `H(M)` (bits) | Distinct Symbols (`s0, s1, s2`) | Paired Gain vs `gauss_noisy_ste` |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for q_name, ag in ed["aggregated"].items():
            s_regs = ", ".join(f"`{v:.4f}`" for v in ag["per_seed_regret"])
            s_syms = ", ".join(f"`{v}`" for v in ag["per_seed_distinct_symbols"])
            g_entry = ed["gains_vs_gauss"].get(q_name)
            g_str = "`0.0000` (ref)" if g_entry is None else f"`{g_entry['gain_mean']:+.4f} ± {g_entry['gain_sd']:.4f}`"
            gap_a = ag.get("gap_to_same_data_anchor", 0.0)
            lines.append(
                f"| `{q_name}` | `{ag['regret_mean']:.4f}` | `{gap_a:+.4f}` | `{ag['regret_sd']:.4f}` | "
                f"`[`{ag['mean_ci_low']:.4f}`, `{ag['mean_ci_high']:.4f}`]` | {s_regs} | "
                f"`{ag['mean_agent_entropy_bits']:.4f} ± {ag['mean_agent_entropy_sd']:.4f}` | "
                f"`{ag['pooled_entropy_bits_mean']:.4f}` | {s_syms} | {g_str} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["E2", "Z8"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--quantizers",
        nargs="+",
        default=[
            "gauss_noisy_ste",
            "noiseless_sign_ste",
            "gumbel_st",
            "fsq_1d_4level_no_usage",
            "fsq_1d_4level",
            "fsq_2d_2x2_no_usage",
            "fsq_2d_2x2",
            "vq_ema_reset",
        ],
    )
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-dev", type=int, default=2_000)
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--out-dir", type=str, default="results/explore/tt-compression-fsq-vq-20261022")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_train = 1_000
        args.n_val = 100
        args.n_dev = 100
        args.steps = 10
        args.batch_size = 64
        args.eval_every = 5
        args.n_boot = 50
        args.seeds = [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": "aggbench.compression-fsq-vq-step6/v2",
        "harness": harness_info(),
        "b": 2,
        "seeds": args.seeds,
        "quantizers": args.quantizers,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_dev": args.n_dev,
        "steps": args.steps,
        "device": str(device),
        "envs": {},
    }

    t_start = time.time()
    for env_name in args.envs:
        env = make_cenv(env_name)
        tr = env.sample("train", args.n_train, start=0)
        val = env.sample("dev", args.n_val, start=50_000)
        dv = env.sample("dev", args.n_dev, start=0)

        cf_res = run_closed_form(
            env_name,
            bits=(2,),
            n_train=min(2000, args.n_train),
            n_dev=args.n_dev,
            M=256 if args.smoke else 1024,
            n_boot=args.n_boot,
            verbose=False,
            arms=("llr_quantizer", "extremized_q"),
        )

        X_tr = torch.tensor(agent_tokens(tr), dtype=torch.float32, device=device)
        Y_tr = torch.tensor(np.asarray(tr.truth, np.float32), dtype=torch.float32, device=device)
        X_val = torch.tensor(agent_tokens(val), dtype=torch.float32, device=device)
        vb_val = np.asarray(val.bayes["logit"], dtype=np.float64)
        X_dv = torch.tensor(agent_tokens(dv), dtype=torch.float32, device=device)

        # Train same-data uncompressed Transformer anchor (n_train=100k, 4000 steps, best_val_kl)
        unc_recs = []
        for seed in args.seeds:
            t_u = time.time()
            u_rec = run_one_quantizer_arm(
                env_name=env_name,
                quantizer="uncompressed",
                seed=seed,
                steps=args.steps,
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
            u_rec.pop("forecast_quant", None)
            unc_recs.append(u_rec)
            print(
                f"[{env_name}] same_data_uncompressed seed={seed}: regret={u_rec['regret_quant']['mean']:.4f} "
                f"best_step={u_rec['best_step']} ({time.time() - t_u:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
        same_data_anchor = aggregate_quantizer_seeds(unc_recs)

        by_q_recs: dict[str, list[dict[str, Any]]] = {q: [] for q in args.quantizers}
        aggregated: dict[str, Any] = {}
        runs: list[dict[str, Any]] = []

        for q_name in args.quantizers:
            for seed in args.seeds:
                t0 = time.time()
                rec = run_one_quantizer_arm(
                    env_name=env_name,
                    quantizer=q_name,
                    seed=seed,
                    steps=args.steps,
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
                by_q_recs[q_name].append(rec)
                print(
                    f"[{env_name}] {q_name:24s} seed={seed}: regret={rec['regret_quant']['mean']:.4f} "
                    f"best_step={rec['best_step']} H_agent={rec['entropy']['mean_agent_entropy_bits']:.4f}b "
                    f"H_pool={rec['entropy']['pooled_entropy_bits']:.4f}b "
                    f"syms={rec['entropy']['distinct_symbols_total']}/4 ({time.time() - t0:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
            aggregated[q_name] = aggregate_quantizer_seeds(
                by_q_recs[q_name],
                anchor_mean=same_data_anchor["regret_mean"],
            )

        gains_vs_gauss: dict[str, Any] = {}
        if "gauss_noisy_ste" in by_q_recs:
            for q_name in args.quantizers:
                if q_name == "gauss_noisy_ste":
                    continue
                g_seeds = []
                for s_i in range(len(args.seeds)):
                    g_i = gain(
                        {"logit": np.asarray(by_q_recs["gauss_noisy_ste"][s_i]["forecast_quant"], dtype=np.float64)},
                        {"logit": np.asarray(by_q_recs[q_name][s_i]["forecast_quant"], dtype=np.float64)},
                        dv,
                        n_boot=min(args.n_boot, 500),
                    )
                    g_seeds.append(float(g_i["mean"]))
                gains_vs_gauss[q_name] = {
                    "gain_mean": float(np.mean(g_seeds)),
                    "gain_sd": float(np.std(g_seeds, ddof=0)),
                    "gain_per_seed": [round(g, 6) for g in g_seeds],
                }

        for q_name in args.quantizers:
            for rec in by_q_recs[q_name]:
                rec.pop("forecast_quant", None)
                runs.append(rec)

        summary["envs"][env_name] = {
            "env": env_name,
            "b": 2,
            "anchors": {
                "prior": cf_res["anchors"]["prior"],
                "floor": cf_res["anchors"]["floor"],
                "same_data_full_context_transformer": same_data_anchor,
                "full_context_transformer_regret": cf_res["anchors"]["full_context_transformer_regret"],
                "llr_quantizer_b2": cf_res["arms"]["llr_quantizer"]["2"]["regret"],
                "extremized_q_b2": cf_res["arms"]["extremized_q"]["2"]["regret"],
            },
            "aggregated": aggregated,
            "gains_vs_gauss": gains_vs_gauss,
            "runs": runs,
        }

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        readme_text = render_readme(summary)
        n_checked, missing = verify_readme_numbers(readme_text, summary)
        assert not missing, f"Unverified numbers in README: {missing}"
        (out_dir / "README.md").write_text(readme_text)

    summary["elapsed_sec"] = round(time.time() - t_start, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    readme_text = render_readme(summary)
    n_checked, missing = verify_readme_numbers(readme_text, summary)
    assert not missing, f"Unverified numbers in README: {missing}"
    (out_dir / "README.md").write_text(readme_text)
    print(f"Done in {summary['elapsed_sec']}s; verified {n_checked}/{n_checked} 4-dec numbers.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
