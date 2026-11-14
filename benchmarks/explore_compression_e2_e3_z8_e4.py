"""Learned Compression on `E2`, `E3-sybil`, `Z8`, and `E4` Against Strong Per-Piece Anchors (`#3896`).

Track: EXPLORATORY (`#3896` compression lane on `E2`, `E3-sybil`, `Z8`, `E4`).

Design & Protocol (`#3896`):
1. **Fresh simulator episodes at every step** (`steps = 8,000`, `batch_size = 512` -> `4.096M` fresh episodes per run;
   zero static-pool memorization). `val` (`n_val = 1,000`, `start = 50,000`) and `dev` (`n_dev = 2,000`, `start = 0`)
   are drawn directly from `aggbench.compression.make_cenv(env_name).sample("dev", ...)`.
2. **Strong Uncompressed Anchors at Matched Steps (`8,000` steps, `3` seeds `0, 1, 2`, recording `best_step`):**
   - `anchor_per_piece_fct_fresh`: Per-piece bidirectional Transformer (`width = 128, layers = 4, heads = 4`, `CLS`
     readout over `piece_cells`, matching branch `nearbayes`).
   - `anchor_slot_enc_tr_fresh`: Uncompressed (`b = inf`, `16`-dim float message per agent) slot-invariant agent
     encoder + 3-layer Transformer set decoder at the same `8,000` fresh-episode steps.
   - Closed-form references: `prior (b = 0)`, `extremized_q (blind closed-form)`,
     `llr_quantizer [PROVENANCE FULL: oracle-structure decoder]`,
     `queried_llr_qinf [PROVENANCE FULL: oracle-structure decoder]`, and on `E4` `achievable_trace_floor`.
3. **Quantizer:** Finite Scalar Quantization (`FSQ`, Mentzer et al. `2309.15505`) at `b in {1, 2, 4, 6}` bits/agent
   (`2^b in {2, 4, 16, 64}` symbols). At eval (`not self.training`), `m = 2 * z_hard - 1` directly (zero float32 STE
   cancellation leak), verified through `Channel(b).send` and `check_message_locality`.
4. **Arms (`b in {1, 2, 4, 6}`, `3` seeds `0, 1, 2`, `8,000` fresh-episode steps, `BLIND DECODER` at `visibility='none'`):**
   - `plain_fsq`: One-shot joint end-to-end `FSQ` training (`Y` labels only, usage bonus + full-entropy init).
   - `masked_pretrain_fsq`: Phase-1 label-free masked-agent + local summary pretraining (`2,000` steps, `NO Y labels`,
     predicting held-out agent's reading from other agents' `b`-bit `FSQ` messages as in `7b33d6e`) followed by
     Phase-2 joint `Y`-label fine-tuning (`6,000` steps, `8,000` total).
   - `queried_fsq` (**Query-before-compression — Kelvin's essential arm**): Implements `QueryProtocol` (`q = None`
     unbounded read-only query from the consensus state of earlier `b`-bit messages, `aggbench/compression.py:750`).
     Agents speak in index order `t = 0..N-1`; before agent `t` compresses its own input `x_t` into `b` bits,
     `ConsensusQueryNet` reads ONLY the earlier agents' discrete `b`-bit `FSQ` messages `m_{<t}` and sends a read-only
     query `q_t` to agent `t`'s encoder `E(x_t, q_t) -> b-bit FSQ`. Total consensus state is strictly `N x b` bits.
   - `queried_masked_pretrain_fsq`: Combines Phase-1 label-free masked-agent pretraining (`2,000` steps, `NO Y labels`)
     with Phase-2 `queried_fsq` query-before-compression fine-tuning (`6,000` steps, `Y` labels only, `8,000` total).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import time
from typing import Any

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.compression import (
    Channel,
    agent_tokens,
    check_message_locality,
    make_cenv,
    run_closed_form,
)
from aggbench.metrics import logsig, summarize, summarize_values
from aggbench.report import achievable_floor, prior_forecast
from aggbench.results import harness_info


def benchmark_harness_info() -> dict[str, Any]:
    """Extends `aggbench.results.harness_info()` with the SHA256 of this benchmark script so both `aggbench/`
    and `benchmarks/explore_compression_e2_e3_z8_e4.py` are cryptographically pinned."""
    info = harness_info()
    script_path = pathlib.Path(__file__).resolve()
    script_sha = hashlib.sha256(script_path.read_bytes()).hexdigest()
    info["script_path"] = "benchmarks/explore_compression_e2_e3_z8_e4.py"
    info["script_sha256"] = script_sha
    expect_script = os.environ.get("AGGBENCH_EXPECT_SCRIPT_SHA256", "")
    if expect_script:
        if script_sha != expect_script:
            info["git_dirty"] = True
        info["script_verified"] = script_sha == expect_script
    return info


def _kl_logits_vec(q: np.ndarray, b: np.ndarray) -> np.ndarray:
    pb = 1.0 / (1.0 + np.exp(-b))
    return pb * (logsig(b) - logsig(q)) + (1.0 - pb) * (logsig(-b) - logsig(-q))


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


def extract_local_slot_cells(x: torch.Tensor, env_name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts per-slot cells `(B, N, K_slots, F_slot)`, active mask `(B, N, K_slots)`, and canonical local summary
    `(B, N, 5)` strictly from each agent's OWN unaligned token `x` (`(B, N, F_tok)`).

    Guarantees 100% agent locality: agent `i`'s slot cells and summary depend ONLY on `x[:, i, :]`.
    """
    if env_name in ("E2", "E3-sybil", "Z8"):
        # Token layout (49 dims): [obs(12), present(12), cov(12), cov_mu(12), rep_div4(1)]
        K = 12
        obs = x[..., 0:12]
        pres = x[..., 12:24]
        cov = x[..., 24:36]
        mu = x[..., 36:48]
        rep = x[..., 48:49]  # report / 4.0

        tau2 = 0.25 if env_name == "Z8" else 0.0
        scale = 1.0 / (1.0 + tau2)
        piece_llr = 2.0 * mu * obs * scale * pres
        piece_info = 2.0 * (mu ** 2) * scale * pres
        n_seen = pres.sum(dim=-1, keepdim=True).clamp_min(1.0) / float(K)
        rep_exp = rep.expand_as(obs).unsqueeze(-1)
        n_seen_exp = n_seen.expand_as(obs).unsqueeze(-1)

        cells = torch.cat(
            [
                pres.unsqueeze(-1),
                obs.unsqueeze(-1),
                cov.unsqueeze(-1),
                mu.unsqueeze(-1),
                piece_llr.unsqueeze(-1),
                piece_info.unsqueeze(-1),
                rep_exp,
                n_seen_exp,
            ],
            dim=-1,
        )  # (B, N, 12, 8)
        sum_llr = piece_llr.sum(dim=-1)
        sum_info = piece_info.sum(dim=-1)
        max_abs_llr = piece_llr.abs().amax(dim=-1)
        max_mu = mu.amax(dim=-1)
        summary = torch.stack([sum_llr / 4.0, sum_info, max_abs_llr / 4.0, max_mu, rep.squeeze(-1)], dim=-1)
        return cells, pres, summary

    if env_name == "E4":
        # Token layout (39 dims): [obs(7), present(7), cov(8), recent(8), cov_mu(8), rep_div4(1)]
        K = 8
        obs = x[..., 0:7]
        cov = x[..., 14:22]
        rec = x[..., 22:30]
        mu = x[..., 30:38]
        rep = x[..., 38:39]

        L = cov.sum(dim=-1, keepdim=True)
        # Step index t in 0..L-1 from recent = (L - 1 - t) / 8
        t_idx = torch.round(L - 1.0 - 8.0 * rec).long().clamp(0, 5)
        inc_s = torch.gather(obs[..., 1:7], -1, t_idx) * cov
        w_s = 1.5 * (0.7 ** (8.0 * rec)) * cov
        debiased_llr = (inc_s / w_s.clamp_min(0.15)) * cov
        piece_info = 2.0 * (mu ** 2) * cov
        obs0 = obs[..., 0:1]
        obs0_exp = obs0.expand_as(cov).unsqueeze(-1)
        rep_exp = rep.expand_as(cov).unsqueeze(-1)

        cells = torch.cat(
            [
                cov.unsqueeze(-1),
                inc_s.unsqueeze(-1),
                rec.unsqueeze(-1),
                mu.unsqueeze(-1),
                w_s.unsqueeze(-1),
                debiased_llr.unsqueeze(-1),
                obs0_exp,
                rep_exp,
            ],
            dim=-1,
        )  # (B, N, 8, 8)
        sum_debiased = debiased_llr.sum(dim=-1)
        sum_info = piece_info.sum(dim=-1)
        max_abs_inc = inc_s.abs().amax(dim=-1)
        summary = torch.stack(
            [sum_debiased / 4.0, sum_info, max_abs_inc / 4.0, obs0.squeeze(-1), rep.squeeze(-1)],
            dim=-1,
        )
        return cells, cov, summary

    raise ValueError(f"Unsupported env_name: {env_name}")


def sample_env_torch(
    env_name: str,
    n: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized GPU fresh-episode generator matching `aggbench`'s exact generative process for `E2`, `E3-sybil`,
    `Z8`, and `E4` at `visibility='none', alignment='unaligned'`.
    Returns `(tokens, y, bayes_logit)`.
    """
    if env_name in ("E2", "E3-sybil"):
        K = 12
        N0 = 5
        n_syb = 3 if env_name == "E3-sybil" else 0
        N = N0 + n_syb
        mu = 0.1 + 0.5 * torch.rand((n, K), generator=gen, device=device)
        y = (torch.rand((n,), generator=gen, device=device) < 0.5).float()
        x = (2.0 * y - 1.0).unsqueeze(1) * mu + torch.randn((n, K), generator=gen, device=device)
        lam = 2.0 * mu * x

        sizes = torch.randint(2, 7, (n, N0), generator=gen, device=device)
        ranks = torch.rand((n, N0, K), generator=gen, device=device).argsort(dim=-1)
        cov = ranks < sizes.unsqueeze(-1)
        if n_syb > 0:
            cov = torch.cat([cov, cov[:, N0 - 1 : N0, :].expand(n, n_syb, K)], dim=1)
        perm_ag = torch.rand((n, N), generator=gen, device=device).argsort(dim=-1)
        cov = torch.gather(cov, 1, perm_ag.unsqueeze(-1).expand(n, N, K))

        cov_f = cov.float()
        local = (cov_f * lam.unsqueeze(1)).sum(dim=-1)
        bayes = (cov.any(dim=1).float() * lam).sum(dim=-1)

        # Unaligned per-agent slot permutation
        perm_slot = torch.rand((n, N, K), generator=gen, device=device).argsort(dim=-1)
        obs_raw = cov_f * x.unsqueeze(1).expand(n, N, K)
        mu_raw = cov_f * mu.unsqueeze(1).expand(n, N, K)

        obs_s = torch.gather(obs_raw, 2, perm_slot)
        pres_s = torch.gather(cov_f, 2, perm_slot)
        cov_s = pres_s
        mu_s = torch.gather(mu_raw, 2, perm_slot)
        tok = torch.cat([obs_s, pres_s, cov_s, mu_s, (local / 4.0).unsqueeze(-1)], dim=-1)
        return tok, y, bayes

    if env_name == "Z8":
        K = 12
        N = 5
        tau = 0.5
        levels = torch.tensor([0.2, 0.4, 0.6], device=device)
        lvl_idx = torch.randint(0, 3, (n, K), generator=gen, device=device)
        mu = levels[lvl_idx]
        y = (torch.rand((n,), generator=gen, device=device) < 0.5).float()
        x = (2.0 * y - 1.0).unsqueeze(1) * mu + torch.randn((n, K), generator=gen, device=device)

        sizes = torch.randint(2, 7, (n, N), generator=gen, device=device)
        ranks = torch.rand((n, N, K), generator=gen, device=device).argsort(dim=-1)
        cov = ranks < sizes.unsqueeze(-1)
        cov_f = cov.float()

        reads = cov_f * (x.unsqueeze(1) + tau * torch.randn((n, N, K), generator=gen, device=device))
        w = cov_f * (2.0 * mu.unsqueeze(1) / (1.0 + tau * tau))
        local = (w * reads).sum(dim=-1)
        n_k = cov_f.sum(dim=1)
        bayes = torch.where(n_k > 0, 2.0 * mu * reads.sum(dim=1) / (n_k + tau * tau), torch.zeros_like(n_k)).sum(dim=-1)

        perm_slot = torch.rand((n, N, K), generator=gen, device=device).argsort(dim=-1)
        obs_s = torch.gather(reads, 2, perm_slot)
        pres_s = torch.gather(cov_f, 2, perm_slot)
        cov_s = pres_s
        mu_s = torch.gather(cov_f * mu.unsqueeze(1).expand(n, N, K), 2, perm_slot)
        tok = torch.cat([obs_s, pres_s, cov_s, mu_s, (local / 4.0).unsqueeze(-1)], dim=-1)
        return tok, y, bayes

    if env_name == "E4":
        K = 8
        N = 6
        Lmax = 6
        mu = 0.1 + 0.5 * torch.rand((n, K), generator=gen, device=device)
        y = (torch.rand((n,), generator=gen, device=device) < 0.5).float()
        x = (2.0 * y - 1.0).unsqueeze(1) * mu + torch.randn((n, K), generator=gen, device=device)
        e = 0.3 * torch.randn((n, K), generator=gen, device=device)
        b_bias = 0.3 * torch.randn((n,), generator=gen, device=device)
        nu = 0.3 * torch.randn((n, N, Lmax), generator=gen, device=device)
        lam = 2.0 * mu * x
        lam_plus_e = lam + e

        lengths = torch.randint(2, 7, (n, N), generator=gen, device=device)
        perm_k = torch.rand((n, N, K), generator=gen, device=device).argsort(dim=-1)
        order = perm_k[:, :, :Lmax]  # (n, N, 6)
        t_grid = torch.arange(Lmax, device=device).view(1, 1, Lmax)
        valid_step = t_grid < lengths.unsqueeze(-1)  # (n, N, 6)

        w_step = 1.5 * (0.7 ** (lengths.unsqueeze(-1).float() - 1.0 - t_grid.float())) * valid_step.float()
        gathered_le = torch.gather(lam_plus_e.unsqueeze(1).expand(n, N, K), 2, order)
        step_inc = (w_step * gathered_le + nu) * valid_step.float()

        obs = torch.cat([b_bias.unsqueeze(1).expand(n, N).unsqueeze(-1), step_inc], dim=-1)  # (n, N, 7)
        pres = torch.cat([torch.ones((n, N, 1), device=device), valid_step.float()], dim=-1)  # (n, N, 7)
        report = obs.sum(dim=-1)
        bayes = lam.sum(dim=-1)

        # Build cov (n, N, 8) and recent (n, N, 8)
        # In perm_k, piece perm_k[e, i, r] is at rank r (0..7); it is active iff r < lengths[e, i]
        r_grid = torch.arange(K, device=device).view(1, 1, K).expand(n, N, K)
        active_r = (r_grid < lengths.unsqueeze(-1)).float()
        rec_r = ((lengths.unsqueeze(-1).float() - 1.0 - r_grid.float()) / float(K)) * active_r

        cov = torch.zeros((n, N, K), device=device)
        recent = torch.zeros((n, N, K), device=device)
        cov.scatter_(2, perm_k, active_r)
        recent.scatter_(2, perm_k, rec_r)
        cov_mu = cov * mu.unsqueeze(1)

        # Apply unaligned slot permutation to the 3 key blocks (cov, recent, cov_mu)
        perm_slot = torch.rand((n, N, K), generator=gen, device=device).argsort(dim=-1)
        cov_s = torch.gather(cov, 2, perm_slot)
        rec_s = torch.gather(recent, 2, perm_slot)
        mu_s = torch.gather(cov_mu, 2, perm_slot)

        tok = torch.cat([obs, pres, cov_s, rec_s, mu_s, (report / 4.0).unsqueeze(-1)], dim=-1)
        return tok, y, bayes

    raise ValueError(f"Unsupported env_name: {env_name}")


class PerPieceFCTNet(nn.Module):
    """Strong per-piece bidirectional Transformer (`fct` over packed per-piece cells, matching `nearbayes`)."""

    def __init__(self, env_name: str, cell_dim: int = 8, width: int = 128, layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.env_name = env_name
        self.cell_dim = cell_dim
        self.register_buffer("c_mean", torch.zeros(cell_dim))
        self.register_buffer("c_sd", torch.ones(cell_dim))
        self.inp = nn.Linear(cell_dim, width)
        # Only E4 (shared per-rollout trace bias b + step noise) uses a tiny agent-origin indicator; E2/E3/Z8 are pure sets of pieces
        self.use_agent_id = env_name == "E4"
        self.agent_emb = nn.Embedding(16, width) if self.use_agent_id else None
        if self.agent_emb is not None:
            nn.init.normal_(self.agent_emb.weight, mean=0.0, std=0.02)
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)

    @torch.no_grad()
    def calibrate(self, x_cal: torch.Tensor) -> None:
        cells, pres, _ = extract_local_slot_cells(x_cal, self.env_name)
        active = cells[pres > 0.5]
        if len(active) > 0:
            self.c_mean.copy_(active.mean(dim=0))
            self.c_sd.copy_(active.std(dim=0).clamp_min(1e-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cells, pres, _ = extract_local_slot_cells(x, self.env_name)
        B, N, K, _ = cells.shape
        z = ((cells - self.c_mean) / self.c_sd) * pres.unsqueeze(-1)
        h = self.inp(z)
        if self.use_agent_id and self.agent_emb is not None:
            ag_ids = torch.arange(N, device=x.device).view(1, N, 1)
            h = h + self.agent_emb(ag_ids) * pres.unsqueeze(-1)
        # Pack active cells per agent (at most 6 active pieces per agent in E2/E3/Z8/E4)
        _, top_idx = torch.topk(pres, k=min(6, K), dim=2)
        h_packed = torch.gather(h, 2, top_idx.unsqueeze(-1).expand(B, N, min(6, K), h.shape[-1]))
        h_flat = h_packed.reshape(B, N * min(6, K), -1)
        tok = torch.cat([self.cls.expand(B, 1, -1), h_flat], dim=1)
        out = self.out(self.norm(self.enc(tok)[:, 0]))
        return out[:, 0]


class SlotInvariantAgentEncoder(nn.Module):
    """Permutation-invariant per-piece encoder inside each agent (`100%` local to agent `i`'s own token `x_i` and
    optional incoming query `q_i`)."""

    def __init__(self, cell_dim: int = 8, summary_dim: int = 5, query_dim: int = 0, hidden: int = 128, out_dim: int = 4) -> None:
        super().__init__()
        self.cell_dim = cell_dim
        self.summary_dim = summary_dim
        self.query_dim = query_dim
        self.slot_embed = nn.Sequential(
            nn.Linear(cell_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.slot_ctx = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.slot_norm = nn.LayerNorm(hidden)
        self.slot_gate = nn.Linear(hidden, 2)

        in_head = 3 * hidden + summary_dim + query_dim
        self.head = nn.Sequential(
            nn.Linear(in_head, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Linear(summary_dim + query_dim, out_dim, bias=False)

    def pool_slots(self, cells_norm: torch.Tensor, pres: torch.Tensor) -> torch.Tensor:
        """Pools `(B, N, K, F_c)` across the `K` local piece slots into `(B, N, 3 * hidden)` strictly per agent."""
        mask_f = pres.unsqueeze(-1)  # (B, N, K, 1)
        h = self.slot_embed(cells_norm) * mask_f
        denom = mask_f.sum(dim=2).clamp_min(1.0)
        mean_init = h.sum(dim=2) / denom
        h = self.slot_norm(h + self.slot_ctx(mean_init).unsqueeze(2)) * mask_f

        pool_mean = h.sum(dim=2) / denom
        gates = self.slot_gate(h).masked_fill(mask_f == 0, -1e4)
        w = torch.softmax(gates, dim=2) * mask_f
        pool_g0 = (h * w[..., 0:1]).sum(dim=2)
        pool_g1 = (h * w[..., 1:2]).sum(dim=2)
        return torch.cat([pool_mean, pool_g0, pool_g1], dim=-1)

    def apply_head(
        self,
        pooled: torch.Tensor,
        summary_norm: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query is not None and self.query_dim > 0:
            sq = torch.cat([summary_norm, query], dim=-1)
        else:
            sq = summary_norm
        feat = torch.cat([pooled, sq], dim=-1)
        raw = self.head(feat) + 0.5 * self.skip(sq)
        return raw, feat

    def forward_features(
        self,
        cells_norm: torch.Tensor,
        pres: torch.Tensor,
        summary_norm: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pool_slots(cells_norm, pres)
        return self.apply_head(pooled, summary_norm, query)


class ConsensusQueryNet(nn.Module):
    """Lightweight causal consensus-query module reading ONLY the discrete `b`-bit `FSQ` messages `m_{<t}` received
    so far (`QueryProtocol` with `q = None`)."""

    def __init__(self, b: int, query_dim: int = 16, hidden: int = 64) -> None:
        super().__init__()
        self.b = b
        self.query_dim = query_dim
        self.msg_in = nn.Sequential(
            nn.Linear(b + query_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.to_query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, query_dim),
            nn.Tanh(),
        )
        self.prefix_logit = nn.Linear(hidden, 1)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        h_state = torch.zeros((batch_size, self.gru.hidden_size), device=device, dtype=dtype)
        q0 = torch.zeros((batch_size, self.query_dim), device=device, dtype=dtype)
        return h_state, q0

    def step(
        self,
        h_state: torch.Tensor,
        q_prev: torch.Tensor,
        m_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Updates consensus state using ONLY the `b`-bit message `m_t` and previous consensus query `q_prev`."""
        inp = self.msg_in(torch.cat([m_t, q_prev], dim=-1))
        h_next = self.gru(inp, h_state)
        q_next = self.to_query(h_next)
        p_logit = self.prefix_logit(h_next)[:, 0]
        return h_next, q_next, p_logit


class FSQEnvCompressionNet(nn.Module):
    """Supports:
    - One-shot `FSQ` compression (`queried = False`)
    - Sequential consensus-queried `FSQ` compression (`queried = True`, `QueryProtocol`)
    - Masked-agent + local-summary pretraining (`no Y labels`)
    - Uncompressed continuous slot-encoder anchor (`b = None`).
    """

    def __init__(
        self,
        env_name: str,
        b: int | None,
        n_agents: int,
        queried: bool = False,
        query_dim: int = 16,
        width: int = 128,
        enc_hidden: int = 128,
        dec_layers: int = 3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.env_name = env_name
        self.b = b
        self.n_agents = n_agents
        self.queried = queried
        self.query_dim = query_dim if queried else 0
        self.out_dim = 16 if b is None else b
        self.K = 1 if b is None else (2 ** b)

        self.register_buffer("c_mean", torch.zeros(8))
        self.register_buffer("c_sd", torch.ones(8))
        self.register_buffer("s_mean", torch.zeros(5))
        self.register_buffer("s_sd", torch.ones(5))
        self.register_buffer("u_offset", torch.zeros(self.out_dim))
        self.register_buffer("u_scale", torch.ones(self.out_dim))

        if b is not None:
            corners = ((torch.arange(self.K).unsqueeze(1) >> torch.arange(b).unsqueeze(0)) & 1).float()
            self.register_buffer("corners", corners)
            self.register_buffer("bit_weights", (2 ** torch.arange(b)).long())

        self.encoder = SlotInvariantAgentEncoder(
            cell_dim=8,
            summary_dim=5,
            query_dim=self.query_dim,
            hidden=enc_hidden,
            out_dim=self.out_dim,
        )
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.15)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.query_net = ConsensusQueryNet(b=self.out_dim, query_dim=query_dim, hidden=64) if queried else None

        # Pretraining heads (used ONLY during Phase-1 no-Y pretraining)
        self.self_recon_head = nn.Sequential(
            nn.Linear(self.out_dim, 64),
            nn.GELU(),
            nn.Linear(64, 5),
        )
        self.masked_agent_query_proj = nn.Linear(4, width)
        pre_layer = nn.TransformerEncoderLayer(width, heads, 2 * width, 0.0, batch_first=True, norm_first=True)
        self.masked_agent_tr = nn.TransformerEncoder(pre_layer, 2, enable_nested_tensor=False)
        self.masked_agent_msg_emb = nn.Linear(self.out_dim, width)
        self.masked_agent_out = nn.Linear(width, 1)

        # Main set decoder (reads ONLY the b-bit messages; in queried mode also knows the arrival step index 0..N-1
        # since consensus queries are indexed by arrival order t = 0..N-1)
        self.dec_inp = nn.Linear(self.out_dim, width)
        self.step_emb = nn.Embedding(16, width) if queried else None
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        dec_layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(dec_layer, dec_layers, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(width)
        self.dec_out = nn.Linear(width, 1)

    def normalize_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cells, pres, summary = extract_local_slot_cells(x, self.env_name)
        cells_norm = ((cells - self.c_mean) / self.c_sd) * pres.unsqueeze(-1)
        summary_norm = (summary - self.s_mean) / self.s_sd
        return cells_norm, pres, summary_norm

    @torch.no_grad()
    def calibrate(self, x_cal: torch.Tensor) -> None:
        cells, pres, summary = extract_local_slot_cells(x_cal, self.env_name)
        flat_c = cells.reshape(-1, cells.shape[-1])
        flat_s = summary.reshape(-1, summary.shape[-1])
        self.c_mean.copy_(flat_c.mean(dim=0))
        self.c_sd.copy_(flat_c.std(dim=0).clamp_min(1e-2))
        self.s_mean.copy_(flat_s.mean(dim=0))
        self.s_sd.copy_(flat_s.std(dim=0).clamp_min(1e-2))

        cells_norm, pres_c, summary_norm = self.normalize_inputs(x_cal[:1024])
        q_zero = (
            torch.zeros((len(cells_norm), cells_norm.shape[1], self.query_dim), device=x_cal.device, dtype=x_cal.dtype)
            if self.queried
            else None
        )
        raw, _ = self.encoder.forward_features(cells_norm, pres_c, summary_norm, q_zero)
        flat_u = raw.reshape(-1, self.out_dim)
        self.u_offset.copy_(flat_u.mean(dim=0))
        self.u_scale.copy_(flat_u.std(dim=0).clamp_min(1e-2))

    def _quantize_u(
        self,
        u: torch.Tensor,
        quant_alpha: float = 1.0,
        compute_usage_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.b is None:
            return u, torch.zeros(u.shape[:-1], dtype=torch.long, device=u.device), torch.zeros((), device=u.device)

        h_cont = torch.tanh(u)
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

        usage_loss = torch.zeros((), dtype=u.dtype, device=u.device)
        if compute_usage_loss:
            p1 = torch.sigmoid(2.0 * u).clamp(1e-6, 1.0 - 1e-6)
            log_p1 = torch.log(p1)
            log_p0 = torch.log(1.0 - p1)
            log_p_corners = torch.einsum("...d,kd->...k", log_p1, self.corners) + torch.einsum(
                "...d,kd->...k", log_p0, 1.0 - self.corners
            )
            p_corners = torch.exp(log_p_corners)
            p_bar = p_corners.reshape(-1, self.K).mean(dim=0).clamp_min(1e-8)
            kl_to_unif = math.log(float(self.K)) + (p_bar * torch.log(p_bar)).sum()
            cond_ent = -(p_corners * log_p_corners).sum(dim=-1).mean()
            scale_reg = (u.pow(2).mean() - 1.0).pow(2)
            usage_loss = kl_to_unif + 0.06 * cond_ent + 0.04 * scale_reg

        return m, syms, usage_loss

    def encode(
        self,
        x: torch.Tensor,
        quant_alpha: float = 1.0,
        compute_usage_loss: bool = False,
        external_queries: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Encodes `(B, N, F_tok)` into `b`-bit `FSQ` messages `(B, N, b)`, integer symbols `(B, N)`, `usage_loss`,
        and optional prefix logits `(B, N)` from `ConsensusQueryNet`."""
        cells_norm, pres, summary_norm = self.normalize_inputs(x)
        B, N, _, _ = cells_norm.shape

        if not self.queried:
            raw, _ = self.encoder.forward_features(cells_norm, pres, summary_norm, None)
            u = (raw - self.u_offset) / self.u_scale
            m, syms, u_loss = self._quantize_u(u, quant_alpha=quant_alpha, compute_usage_loss=compute_usage_loss)
            return m, syms, u_loss, None

        assert self.query_net is not None
        if external_queries is not None:
            # Used by check_message_locality when `received` (queries) is held fixed externally
            raw, _ = self.encoder.forward_features(cells_norm, pres, summary_norm, external_queries)
            u = (raw - self.u_offset) / self.u_scale
            m, syms, u_loss = self._quantize_u(u, quant_alpha=quant_alpha, compute_usage_loss=compute_usage_loss)
            return m, syms, u_loss, None

        # Pre-pool local slots once across all N agents in parallel (strictly per-agent, no cross-agent interaction)
        pooled_all = self.encoder.pool_slots(cells_norm, pres)
        h_state, q_t = self.query_net.init_state(B, x.device, x.dtype)
        m_list = []
        sym_list = []
        u_list = []
        prefix_logits = []
        for t in range(N):
            p_t = pooled_all[:, t : t + 1]
            s_t = summary_norm[:, t : t + 1]
            raw_t, _ = self.encoder.apply_head(p_t, s_t, q_t.unsqueeze(1))
            u_t = (raw_t[:, 0] - self.u_offset) / self.u_scale
            m_t, sym_t, _ = self._quantize_u(u_t, quant_alpha=quant_alpha, compute_usage_loss=False)
            h_state, q_t, p_logit = self.query_net.step(h_state, q_t, m_t)
            m_list.append(m_t)
            sym_list.append(sym_t)
            u_list.append(u_t)
            prefix_logits.append(p_logit)

        u_all = torch.stack(u_list, dim=1)
        m_all = torch.stack(m_list, dim=1)
        syms_all = torch.stack(sym_list, dim=1)
        _, _, u_loss = self._quantize_u(u_all, quant_alpha=quant_alpha, compute_usage_loss=compute_usage_loss)
        return m_all, syms_all, u_loss, torch.stack(prefix_logits, dim=1)

    def decode(self, m: torch.Tensor) -> torch.Tensor:
        B, N, _ = m.shape
        h = self.dec_inp(m)
        if self.queried and self.step_emb is not None:
            steps = torch.arange(N, device=m.device).unsqueeze(0)
            h = h + self.step_emb(steps)
        tok = torch.cat([self.cls.expand(B, 1, -1), h], dim=1)
        return self.dec_out(self.dec_norm(self.dec(tok)[:, 0]))[:, 0]

    def masked_pretrain_loss(self, x: torch.Tensor, quant_alpha: float = 1.0) -> torch.Tensor:
        """Phase-1 label-free pretraining (`NO Y labels`):
        1. Self-summary reconstruction: each agent's `b`-bit `FSQ` message `m_i` reconstructs its own canonical
           slot-invariant summary `summary_norm[:, i]`.
        2. Masked-agent cross-prediction (`7b33d6e`): from the other `N-1` agents' `b`-bit `FSQ` messages `m_{-i}`
           and agent `i`'s non-sign footprint `(sum_info, max_abs_llr, max_mu, n_seen)`, predict held-out agent `i`'s
           signed local reading `summary_norm[:, i, 0]` (`local_llr`) and `summary_norm[:, i, 4]` (`report`).
        """
        cells_norm, pres, summary_norm = self.normalize_inputs(x)
        B, N, _, _ = cells_norm.shape
        m, _, u_loss, prefix_logits = self.encode(x, quant_alpha=quant_alpha, compute_usage_loss=True)

        # 1. Self-reconstruction loss
        pred_self = self.self_recon_head(m)
        loss_self = F.smooth_l1_loss(pred_self, summary_norm)
        if prefix_logits is not None:
            # Label-free prefix consensus target: running cumulative sum of local reports (no Y!)
            running_rep = torch.cumsum(summary_norm[..., 4], dim=1)
            loss_self = loss_self + 0.25 * F.smooth_l1_loss(prefix_logits, running_rep)

        # 2. Masked-agent cross-prediction loss across all N agents (batched across i=0..N-1)
        msg_toks = self.masked_agent_msg_emb(m)  # (B, N, width)
        n_seen_frac = pres.mean(dim=-1, keepdim=True)
        footprint = torch.cat([summary_norm[..., 1:4], n_seen_frac], dim=-1)  # (B, N, 4)
        q_toks = self.masked_agent_query_proj(footprint)  # (B, N, width)

        others_idx = torch.tensor([[j for j in range(N) if j != i] for i in range(N)], device=x.device)  # (N, N-1)
        other_msgs = msg_toks[:, others_idx, :]  # (B, N, N-1, width)
        seq = torch.cat([q_toks.unsqueeze(2), other_msgs], dim=2).reshape(B * N, N, -1)
        h_all = self.masked_agent_tr(seq)[:, 0, :].reshape(B, N, -1)
        pred_cross = self.masked_agent_out(h_all).squeeze(-1)
        loss_cross = F.smooth_l1_loss(pred_cross, summary_norm[..., 4])

        return loss_self + 0.5 * loss_cross + 0.08 * u_loss


def verify_arm_locality(net: FSQEnvCompressionNet, dv_batch: Any, env_name: str, b: int, device: torch.device) -> bool:
    """Runs BOTH:
    1. `aggbench.compression.check_message_locality` on `64` dev episodes (at the actual sequential queries `q_seq`
       for `queried` arms) to prove that agent `i`'s `b`-bit message depends ONLY on `(obs_i, own_i)` and its received
       query `q_i`, and that perturbing `q_i` changes ONLY agent `i`'s message;
    2. Live sequential `ConsensusQueryNet` rollout causality verification across all `N` agents (`t = 0..N-1`),
       proving that (a) perturbing future agents `j > t` has zero effect on `m_{<=t}`, `syms_{<=t}`, or `q_{<=t}`,
       and (b) perturbing past agents `j < t` while clamping their discrete `b`-bit messages `m_{<t}` has zero effect
       on `q_t` or `m_t` (zero continuous side-channel leak)."""
    sub_obs, _, public, sub_own = dv_batch.agent_inputs("none", alignment="unaligned", own_model=True)
    n_sub = min(64, len(sub_obs))
    obs_s = sub_obs[:n_sub].copy()
    own_s = sub_own[:n_sub].copy()
    K_slots = obs_s.shape[-1]

    def build_tok_from_obs_own(o_arr: np.ndarray, w_arr: np.ndarray) -> torch.Tensor:
        o_clean = np.nan_to_num(o_arr).astype(np.float32)
        pres = np.isfinite(o_arr).astype(np.float32)
        if env_name in ("E2", "E3-sybil"):
            mu_part = w_arr[..., K_slots : 2 * K_slots]
            loc = np.sum(2.0 * mu_part * o_clean * pres, axis=-1, keepdims=True) / 4.0
        elif env_name == "Z8":
            mu_part = w_arr[..., K_slots : 2 * K_slots]
            loc = np.sum((2.0 * mu_part / 1.25) * o_clean * pres, axis=-1, keepdims=True) / 4.0
        else:
            loc = np.sum(o_clean * pres, axis=-1, keepdims=True) / 4.0
        tok_np = np.concatenate([o_clean, pres, w_arr.astype(np.float32), loc.astype(np.float32)], axis=-1)
        return torch.tensor(tok_np, dtype=torch.float32, device=device)

    net.eval()
    with torch.no_grad():
        if not net.queried:
            def encode_oneshot(o_a: np.ndarray, w_a: np.ndarray, received: Any = None) -> np.ndarray:
                t_in = build_tok_from_obs_own(o_a, w_a)
                _, syms, _, _ = net.encode(t_in, quant_alpha=1.0, compute_usage_loss=False)
                return syms.cpu().numpy().astype(np.int64)

            return bool(check_message_locality(encode_oneshot, obs_s, own_s, received=None, trials=5, seed=0))
        else:
            assert net.query_net is not None
            t_base = build_tok_from_obs_own(obs_s, own_s)
            cells_b, pres_b, sum_b = net.normalize_inputs(t_base)
            pooled_b = net.encoder.pool_slots(cells_b, pres_b)
            B_sub, N_ag, _, _ = cells_b.shape
            h_st, q_cur = net.query_net.init_state(B_sub, device, t_base.dtype)
            q_live_list = []
            m_live_list = []
            for t_idx in range(N_ag):
                q_live_list.append(q_cur)
                raw_t, _ = net.encoder.apply_head(pooled_b[:, t_idx : t_idx + 1], sum_b[:, t_idx : t_idx + 1], q_cur.unsqueeze(1))
                u_t = (raw_t[:, 0] - net.u_offset) / net.u_scale
                m_t, _, _ = net._quantize_u(u_t, quant_alpha=1.0, compute_usage_loss=False)
                m_live_list.append(m_t)
                h_st, q_cur, _ = net.query_net.step(h_st, q_cur, m_t)
            q_seq_actual = torch.stack(q_live_list, dim=1).cpu().numpy().astype(np.float32)
            m_seq_actual = torch.stack(m_live_list, dim=1)

            def encode_queried(o_a: np.ndarray, w_a: np.ndarray, received: Any = None) -> np.ndarray:
                t_in = build_tok_from_obs_own(o_a, w_a)
                q_t = torch.tensor(received, dtype=torch.float32, device=device)
                _, syms, _, _ = net.encode(t_in, quant_alpha=1.0, compute_usage_loss=False, external_queries=q_t)
                return syms.cpu().numpy().astype(np.int64)

            fixed_ok = bool(check_message_locality(encode_queried, obs_s, own_s, received=q_seq_actual, trials=5, seed=0))
            if not fixed_ok:
                return False

            # Live sequential GRU rollout causality check across all N_ag steps
            m_base, sym_base, _, pref_base = net.encode(t_base, quant_alpha=1.0, compute_usage_loss=False)
            if not torch.equal(m_base, m_seq_actual):
                return False
            rng = np.random.default_rng(42)
            for t_idx in range(N_ag):
                if t_idx + 1 < N_ag:
                    o_fut = obs_s.copy()
                    w_fut = own_s.copy()
                    fin = np.isfinite(o_fut[:, t_idx + 1 :])
                    o_fut[:, t_idx + 1 :] = np.where(fin, o_fut[:, t_idx + 1 :] + rng.normal(size=fin.shape), o_fut[:, t_idx + 1 :])
                    w_fut[:, t_idx + 1 :] = w_fut[:, t_idx + 1 :] * np.exp(0.3 * rng.normal(size=w_fut[:, t_idx + 1 :].shape))
                    t_fut = build_tok_from_obs_own(o_fut, w_fut)
                    m_f, sym_f, _, pref_f = net.encode(t_fut, quant_alpha=1.0, compute_usage_loss=False)
                    if not (
                        torch.equal(m_f[:, : t_idx + 1], m_base[:, : t_idx + 1])
                        and torch.equal(sym_f[:, : t_idx + 1], sym_base[:, : t_idx + 1])
                        and (pref_f is None or pref_base is None or torch.allclose(pref_f[:, : t_idx + 1], pref_base[:, : t_idx + 1], atol=1e-6))
                    ):
                        return False
                if t_idx > 0:
                    # Perturb past agents < t_idx continuously while clamping discrete prefix messages m_{<t_idx}
                    o_past = obs_s.copy()
                    w_past = own_s.copy()
                    fin_p = np.isfinite(o_past[:, :t_idx])
                    o_past[:, :t_idx] = np.where(fin_p, o_past[:, :t_idx] + rng.normal(size=fin_p.shape), o_past[:, :t_idx])
                    w_past[:, :t_idx] = w_past[:, :t_idx] * np.exp(0.3 * rng.normal(size=w_past[:, :t_idx].shape))
                    t_past = build_tok_from_obs_own(o_past, w_past)
                    c_p, pr_p, s_p = net.normalize_inputs(t_past)
                    pool_p = net.encoder.pool_slots(c_p, pr_p)
                    h_p, q_p = net.query_net.init_state(B_sub, device, t_past.dtype)
                    for j_idx in range(t_idx):
                        h_p, q_p, _ = net.query_net.step(h_p, q_p, m_base[:, j_idx])
                    raw_p, _ = net.encoder.apply_head(pool_p[:, t_idx : t_idx + 1], s_p[:, t_idx : t_idx + 1], q_p.unsqueeze(1))
                    u_p = (raw_p[:, 0] - net.u_offset) / net.u_scale
                    m_p, _, _ = net._quantize_u(u_p, quant_alpha=1.0, compute_usage_loss=False)
                    if not torch.equal(m_p, m_base[:, t_idx]):
                        return False
            return True


def verify_torch_vs_aggbench_generator(env_name: str, device: torch.device, n_check: int = 4096) -> dict[str, Any]:
    """Verifies that the vectorized `sample_env_torch` fresh-episode generator matches `aggbench`'s reference
    `make_cenv(env_name).sample('train', n_check)` in token shape, base rate, Bayes logit moments, and token channel moments."""
    env = make_cenv(env_name)
    ab_batch = env.sample("train", n_check, start=7_000_000)
    ab_tok = np.asarray(agent_tokens(ab_batch), dtype=np.float64)
    ab_y = np.asarray(ab_batch.truth, dtype=np.float64)
    ab_bayes = np.asarray(ab_batch.bayes["logit"], dtype=np.float64)

    gen = torch.Generator(device=device).manual_seed(20261022)
    th_tok_t, th_y_t, th_bayes_t = sample_env_torch(env_name, n_check, device, gen)
    th_tok = th_tok_t.double().cpu().numpy()
    th_y = th_y_t.double().cpu().numpy()
    th_bayes = th_bayes_t.double().cpu().numpy()

    shape_match = list(ab_tok.shape) == list(th_tok.shape)
    tok_mean_max_diff = float(np.max(np.abs(ab_tok.mean(axis=(0, 1)) - th_tok.mean(axis=(0, 1)))))
    tok_std_max_diff = float(np.max(np.abs(ab_tok.std(axis=(0, 1)) - th_tok.std(axis=(0, 1)))))
    bayes_std_diff = float(abs(ab_bayes.std() - th_bayes.std()))
    verified = bool(shape_match and tok_mean_max_diff < 0.05 and tok_std_max_diff < 0.05 and bayes_std_diff < 0.15)
    if not verified:
        raise RuntimeError(f"Generator mismatch on {env_name}: mean_diff={tok_mean_max_diff}, std_diff={tok_std_max_diff}")
    return {
        "n_check": int(n_check),
        "shape": list(ab_tok.shape),
        "shape_match": shape_match,
        "aggbench_y_mean": float(ab_y.mean()),
        "torch_y_mean": float(th_y.mean()),
        "aggbench_bayes_std": float(ab_bayes.std()),
        "torch_bayes_std": float(th_bayes.std()),
        "tok_channel_mean_max_diff": tok_mean_max_diff,
        "tok_channel_std_max_diff": tok_std_max_diff,
        "verified": verified,
    }


def run_per_piece_fct_anchor(
    env_name: str,
    seed: int,
    X_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    bayes_dv: np.ndarray,
    floor_dv: np.ndarray | None,
    X_cal: torch.Tensor,
    l0: float,
    steps: int = 8_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> dict[str, Any]:
    """Trains the strong per-piece bidirectional Transformer (`PerPieceFCTNet`) on fresh simulator episodes."""
    device = X_val.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    net = PerPieceFCTNet(env_name=env_name, cell_dim=8, width=128, layers=4, heads=4).to(device)
    net.calibrate(X_cal)

    @torch.no_grad()
    def eval_split(X_s: torch.Tensor, target_b: np.ndarray) -> tuple[float, np.ndarray]:
        net.eval()
        preds = []
        for s in range(0, len(X_s), 1024):
            preds.append((l0 + net(X_s[s : s + 1024])).double().cpu().numpy())
        net.train()
        q = np.concatenate(preds, axis=0)
        kl_v = _kl_logits_vec(q, target_b)
        return float(kl_v.mean()), kl_v

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed * 10007 + 17)

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    chunk_len = 250
    x_chunk: torch.Tensor | None = None
    y_chunk: torch.Tensor | None = None

    for t in range(steps):
        t_mod = t % chunk_len
        if t_mod == 0:
            c_steps = min(chunk_len, steps - t)
            xc, yc, _ = sample_env_torch(env_name, c_steps * batch_size, device, gen)
            x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
            y_chunk = yc.reshape(c_steps, batch_size)
        assert x_chunk is not None and y_chunk is not None
        xb = x_chunk[t_mod]
        yb = y_chunk[t_mod]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lg = l0 + net(xb)
            loss = F.binary_cross_entropy_with_logits(lg, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (t + 1) % eval_every == 0 or (t + 1) == steps:
            v_kl, _ = eval_split(X_val, bayes_val)
            if v_kl < best_val_kl:
                best_val_kl = v_kl
                best_state = copy.deepcopy(net.state_dict())
                best_step = t + 1

    if best_state is not None:
        net.load_state_dict(best_state)
    _, kl_dv = eval_split(X_dv, bayes_dv)
    reg = summarize_values(kl_dv, n_boot)
    out = {
        "anchor": "anchor_per_piece_fct_fresh",
        "seed": seed,
        "steps": steps,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "regret": reg,
    }
    if floor_dv is not None:
        _, kl_fl = eval_split(X_dv, floor_dv)
        out["regret_to_floor"] = summarize_values(kl_fl, n_boot)
    return out


def run_slot_enc_uncompressed_anchor(
    env_name: str,
    seed: int,
    X_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    bayes_dv: np.ndarray,
    floor_dv: np.ndarray | None,
    X_cal: torch.Tensor,
    l0: float,
    steps: int = 8_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the uncompressed (`b = None`, `16`-dim float per agent) slot-invariant encoder + 3-layer Transformer
    set decoder on fresh episodes, plus evaluates the untrained random-weights control."""
    device = X_val.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    net = FSQEnvCompressionNet(
        env_name=env_name,
        b=None,
        n_agents=X_val.shape[1],
        queried=False,
        width=128,
        enc_hidden=128,
        dec_layers=3,
        heads=4,
    ).to(device)
    net.calibrate(X_cal)

    @torch.no_grad()
    def eval_split(X_s: torch.Tensor, target_b: np.ndarray) -> tuple[float, np.ndarray]:
        net.eval()
        preds = []
        for s in range(0, len(X_s), 1024):
            m, _, _, _ = net.encode(X_s[s : s + 1024], quant_alpha=1.0, compute_usage_loss=False)
            preds.append((l0 + net.decode(m)).double().cpu().numpy())
        net.train()
        q = np.concatenate(preds, axis=0)
        kl_v = _kl_logits_vec(q, target_b)
        return float(kl_v.mean()), kl_v

    # Evaluate untrained control first
    _, kl_untrained = eval_split(X_dv, bayes_dv)
    ctrl_rec = {
        "anchor": "control_untrained_slot_enc",
        "seed": seed,
        "steps": 0,
        "best_val_kl": float(kl_untrained.mean()),
        "best_step": 0,
        "regret": summarize_values(kl_untrained, n_boot),
    }

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    gen = torch.Generator(device=device).manual_seed(seed * 10007 + 53)

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    chunk_len = 250
    x_chunk: torch.Tensor | None = None
    y_chunk: torch.Tensor | None = None

    for t in range(steps):
        t_mod = t % chunk_len
        if t_mod == 0:
            c_steps = min(chunk_len, steps - t)
            xc, yc, _ = sample_env_torch(env_name, c_steps * batch_size, device, gen)
            x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
            y_chunk = yc.reshape(c_steps, batch_size)
        assert x_chunk is not None and y_chunk is not None
        xb = x_chunk[t_mod]
        yb = y_chunk[t_mod]

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            m, _, _, _ = net.encode(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = l0 + net.decode(m)
            loss = F.binary_cross_entropy_with_logits(lg, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (t + 1) % eval_every == 0 or (t + 1) == steps:
            v_kl, _ = eval_split(X_val, bayes_val)
            if v_kl < best_val_kl:
                best_val_kl = v_kl
                best_state = copy.deepcopy(net.state_dict())
                best_step = t + 1

    if best_state is not None:
        net.load_state_dict(best_state)
    _, kl_dv = eval_split(X_dv, bayes_dv)
    out = {
        "anchor": "anchor_slot_enc_tr_fresh",
        "seed": seed,
        "steps": steps,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "regret": summarize_values(kl_dv, n_boot),
    }
    if floor_dv is not None:
        _, kl_fl = eval_split(X_dv, floor_dv)
        out["regret_to_floor"] = summarize_values(kl_fl, n_boot)
    return out, ctrl_rec


@torch.no_grad()
def eval_compression_net(
    net: FSQEnvCompressionNet,
    X: torch.Tensor,
    target_b: np.ndarray,
    l0: float,
    b: int,
    check_channel: bool = False,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Evaluates strictly on discrete `b`-bit `FSQ` messages (`not net.training` -> `m = 2 * z_hard - 1`)."""
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        net.eval()
        preds = []
        syms_list = []
        for s in range(0, len(X), 1024):
            xb = X[s : s + 1024]
            m, syms, _, _ = net.encode(xb, quant_alpha=1.0, compute_usage_loss=False)
            lg = l0 + net.decode(m)
            preds.append(lg.double().cpu().numpy())
            syms_list.append(syms.cpu().numpy())
        net.train()
        q = np.concatenate(preds, axis=0)
        all_syms = np.concatenate(syms_list, axis=0)
        kl_vals = _kl_logits_vec(q, target_b)
        sym_metrics = compute_symbol_metrics(all_syms, b=b) if check_channel else {}
        return float(kl_vals.mean()), kl_vals, sym_metrics
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_compression_arm(
    env_name: str,
    arm_name: str,
    b: int,
    seed: int,
    X_val: torch.Tensor,
    bayes_val: np.ndarray,
    X_dv: torch.Tensor,
    bayes_dv: np.ndarray,
    floor_dv: np.ndarray | None,
    X_cal: torch.Tensor,
    dv_batch: Any,
    l0: float,
    prior_kl: float,
    per_piece_fct_regret: float,
    slot_enc_tr_regret: float,
    steps: int = 8_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    eval_every: int = 200,
    n_boot: int = 2_000,
    strict_step_budget: bool = False,
) -> dict[str, Any]:
    """Runs one learned compression arm (`plain_fsq`, `masked_pretrain_fsq`, `queried_fsq`,
    `queried_masked_pretrain_fsq`) on fresh episodes at bit budget `b`."""
    device = X_val.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    is_queried = arm_name in ("queried_fsq", "queried_masked_pretrain_fsq")
    is_pretrain = arm_name in ("masked_pretrain_fsq", "queried_masked_pretrain_fsq")

    net = FSQEnvCompressionNet(
        env_name=env_name,
        b=b,
        n_agents=X_val.shape[1],
        queried=is_queried,
        query_dim=16,
        width=128,
        enc_hidden=128,
        dec_layers=3,
        heads=4,
    ).to(device)
    net.calibrate(X_cal)

    gen = torch.Generator(device=device).manual_seed(seed * 10007 + 99 + b * 31)
    chunk_len = 250

    ft_steps = min(300, max(10, steps // 20))
    pretrain_steps = (steps // 4) if is_pretrain else 0
    if strict_step_budget:
        finetune_steps = max(10, steps - pretrain_steps - ft_steps)
    else:
        finetune_steps = steps - pretrain_steps

    # Phase 1: Masked-agent + local summary pretraining (NO Y labels!)
    if pretrain_steps > 0:
        pre_params = (
            list(net.encoder.parameters())
            + (list(net.query_net.parameters()) if net.query_net is not None else [])
            + list(net.self_recon_head.parameters())
            + list(net.masked_agent_query_proj.parameters())
            + list(net.masked_agent_tr.parameters())
            + list(net.masked_agent_msg_emb.parameters())
            + list(net.masked_agent_out.parameters())
        )
        opt_pre = torch.optim.AdamW(pre_params, lr=lr, weight_decay=1e-4)
        sched_pre = torch.optim.lr_scheduler.OneCycleLR(opt_pre, max_lr=lr, total_steps=pretrain_steps)
        x_chunk_p: torch.Tensor | None = None
        for t_p in range(pretrain_steps):
            t_mod = t_p % chunk_len
            if t_mod == 0:
                c_steps = min(chunk_len, pretrain_steps - t_p)
                xc, _, _ = sample_env_torch(env_name, c_steps * batch_size, device, gen)
                x_chunk_p = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
            assert x_chunk_p is not None
            xb = x_chunk_p[t_mod]
            q_alpha = min(1.0, 0.30 + 0.70 * (t_p / max(1, pretrain_steps - 1)))
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                loss_p = net.masked_pretrain_loss(xb, quant_alpha=q_alpha)
            opt_pre.zero_grad(set_to_none=True)
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(pre_params, 1.0)
            opt_pre.step()
            sched_pre.step()

    # Phase 2: Joint Y-label fine-tuning on fresh episodes
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=finetune_steps)

    best_val_kl = float("inf")
    best_state = None
    best_step = -1
    escape_step: int | None = None
    x_chunk: torch.Tensor | None = None
    y_chunk: torch.Tensor | None = None

    for t_f in range(finetune_steps):
        global_step = pretrain_steps + t_f + 1
        t_mod = t_f % chunk_len
        if t_mod == 0:
            c_steps = min(chunk_len, finetune_steps - t_f)
            xc, yc, _ = sample_env_torch(env_name, c_steps * batch_size, device, gen)
            x_chunk = xc.reshape(c_steps, batch_size, xc.shape[1], xc.shape[2])
            y_chunk = yc.reshape(c_steps, batch_size)
        assert x_chunk is not None and y_chunk is not None
        xb = x_chunk[t_mod]
        yb = y_chunk[t_mod]

        # Gentle homotopy in first 30% of finetune, then pure discrete FSQ
        if t_f < int(0.30 * finetune_steps):
            prog = t_f / max(1, int(0.30 * finetune_steps))
            quant_alpha = (0.60 + 0.40 * prog) if is_pretrain else (0.30 + 0.70 * prog)
        else:
            quant_alpha = 1.0

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            m, _, u_loss, prefix_logits = net.encode(xb, quant_alpha=quant_alpha, compute_usage_loss=True)
            lg = l0 + net.decode(m)
            loss = F.binary_cross_entropy_with_logits(lg, yb) + 0.05 * u_loss
            if prefix_logits is not None:
                # Supervise ConsensusQueryNet's prefix log-odds readout strictly with the episode outcome Y (no oracle!)
                y_exp = yb.unsqueeze(1).expand_as(prefix_logits)
                loss = loss + 0.15 * F.binary_cross_entropy_with_logits(l0 + prefix_logits, y_exp)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()

        if (t_f + 1) % eval_every == 0 or (t_f + 1) == finetune_steps:
            val_kl, _, _ = eval_compression_net(net, X_val, bayes_val, l0=l0, b=b, check_channel=False)
            if escape_step is None and val_kl < 0.9 * prior_kl:
                escape_step = global_step
            if (quant_alpha >= 0.999 or (t_f + 1) == finetune_steps) and val_kl < best_val_kl:
                best_val_kl = val_kl
                best_state = copy.deepcopy(net.state_dict())
                best_step = global_step

    if best_state is not None:
        net.load_state_dict(best_state)

    # Brief 300-step discrete decoder lock-in on frozen discrete FSQ messages
    dec_params = (
        list(net.dec_inp.parameters())
        + ([net.cls] if net.cls is not None else [])
        + (list(net.step_emb.parameters()) if net.step_emb is not None else [])
        + list(net.dec.parameters())
        + list(net.dec_norm.parameters())
        + list(net.dec_out.parameters())
    )
    for p in net.parameters():
        p.requires_grad_(False)
    for p in dec_params:
        p.requires_grad_(True)
    opt_ft = torch.optim.AdamW(dec_params, lr=3e-4, weight_decay=1e-4)
    xc_ft, yc_ft, _ = sample_env_torch(env_name, ft_steps * batch_size, device, gen)
    x_ft = xc_ft.reshape(ft_steps, batch_size, xc_ft.shape[1], xc_ft.shape[2])
    y_ft = yc_ft.reshape(ft_steps, batch_size)
    for ft_i in range(ft_steps):
        xb, yb = x_ft[ft_i], y_ft[ft_i]
        with torch.no_grad():
            net.eval()
            m, _, _, _ = net.encode(xb, quant_alpha=1.0, compute_usage_loss=False)
        net.train()
        lg = l0 + net.decode(m)
        loss_ft = F.binary_cross_entropy_with_logits(lg, yb)
        opt_ft.zero_grad(set_to_none=True)
        loss_ft.backward()
        torch.nn.utils.clip_grad_norm_(dec_params, 1.0)
        opt_ft.step()
    for p in net.parameters():
        p.requires_grad_(True)
    val_kl_ft, _, _ = eval_compression_net(net, X_val, bayes_val, l0=l0, b=b, check_channel=False)
    total_batches = pretrain_steps + finetune_steps + ft_steps
    if val_kl_ft < best_val_kl:
        best_val_kl = val_kl_ft
        best_step = total_batches if strict_step_budget else steps
    elif best_state is not None:
        net.load_state_dict(best_state)

    locality_ok = verify_arm_locality(net, dv_batch, env_name=env_name, b=b, device=device)
    if not locality_ok:
        raise RuntimeError(f"Locality check failed for {env_name} {arm_name} b={b}")

    _, kl_dv, sym_metrics = eval_compression_net(net, X_dv, bayes_dv, l0=l0, b=b, check_channel=True)
    regret_summary = summarize_values(kl_dv, n_boot)
    gap_to_per_piece_fct = float(regret_summary["mean"] - per_piece_fct_regret)
    gap_to_slot_enc_tr = float(regret_summary["mean"] - slot_enc_tr_regret)

    out = {
        "env": env_name,
        "arm": arm_name,
        "b": b,
        "seed": seed,
        "queried": is_queried,
        "masked_pretrain": is_pretrain,
        "locality_verified": locality_ok,
        "strict_step_budget": strict_step_budget,
        "total_batches": int(total_batches),
        "regret_quant": regret_summary,
        "gap_to_per_piece_fct": gap_to_per_piece_fct,
        "gap_to_slot_enc_tr": gap_to_slot_enc_tr,
        "best_val_kl": float(best_val_kl),
        "best_step": int(best_step),
        "escape_step": escape_step,
        "escaped": escape_step is not None,
        "symbol_metrics": sym_metrics,
    }
    if floor_dv is not None:
        _, kl_fl, _ = eval_compression_net(net, X_dv, floor_dv, l0=l0, b=b, check_channel=False)
        out["regret_to_floor"] = summarize_values(kl_fl, n_boot)
    return out


def _sample_sd_and_t_ci(values: list[float]) -> tuple[float, float, float]:
    """Returns `(sd_ddof1, ci_low, ci_high)` across seeds using `ddof=1` and Student-t at `0.975`."""
    n = len(values)
    mean = float(np.mean(values))
    if n <= 1:
        return 0.0, mean, mean
    sd = float(np.std(values, ddof=1))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    se = sd / math.sqrt(n)
    return sd, float(mean - t_crit * se), float(mean + t_crit * se)


def aggregate_arm_seeds(
    records: list[dict[str, Any]],
    per_piece_fct_regret: float,
    slot_enc_tr_regret: float,
    per_piece_fct_floor_regret: float | None = None,
    ext_b_regret: float | None = None,
    ext_inf_regret: float | None = None,
) -> dict[str, Any]:
    records = sorted(records, key=lambda r: int(r["seed"]))
    means = [float(r["regret_quant"]["mean"]) for r in records]
    val_kls = [float(r["best_val_kl"]) for r in records]
    ci_lows = [float(r["regret_quant"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret_quant"]["ci_high"]) for r in records]
    b_steps = [int(r["best_step"]) for r in records]
    escs = [r["escape_step"] for r in records]
    h_agents = [float(r["symbol_metrics"]["mean_agent_entropy_bits"]) for r in records]
    h_pools = [float(r["symbol_metrics"]["pooled_entropy_bits"]) for r in records]
    d_syms = [int(r["symbol_metrics"]["distinct_symbols_total"]) for r in records]
    reg_mean = float(np.mean(means))
    reg_sd, seed_ci_low, seed_ci_high = _sample_sd_and_t_ci(means)
    val_kl_mean = float(np.mean(val_kls))
    val_kl_sd, _, _ = _sample_sd_and_t_ci(val_kls)
    h_sd, _, _ = _sample_sd_and_t_ci(h_agents)

    out: dict[str, Any] = {
        "n_seeds": len(records),
        "seeds_list": [int(r["seed"]) for r in records],
        "regret_mean": reg_mean,
        "regret_sd": reg_sd,
        "seed_ci_low": seed_ci_low,
        "seed_ci_high": seed_ci_high,
        "val_kl_mean": val_kl_mean,
        "val_kl_sd": val_kl_sd,
        "per_seed_val_kl": [round(v, 4) for v in val_kls],
        "gap_to_per_piece_fct_mean": float(reg_mean - per_piece_fct_regret),
        "gap_to_slot_enc_tr_mean": float(reg_mean - slot_enc_tr_regret),
        "mean_episode_boot_ci_low": float(np.mean(ci_lows)),
        "mean_episode_boot_ci_high": float(np.mean(ci_highs)),
        "per_seed_regret": [round(m, 4) for m in means],
        "per_seed_best_step": b_steps,
        "n_escaped": int(sum(1 for e in escs if e is not None)),
        "per_seed_escape": escs,
        "mean_agent_entropy_bits": float(np.mean(h_agents)),
        "mean_agent_entropy_sd": h_sd,
        "pooled_entropy_bits_mean": float(np.mean(h_pools)),
        "per_seed_distinct_symbols": d_syms,
        "locality_verified_all": all(bool(r.get("locality_verified", False)) for r in records),
    }
    if ext_b_regret is not None:
        wins_b = int(sum(1 for m in means if m < ext_b_regret))
        diff_b = float(reg_mean - ext_b_regret)
        diff_b_lo = float(seed_ci_low - ext_b_regret)
        diff_b_hi = float(seed_ci_high - ext_b_regret)
        if diff_b_hi < 0.0:
            verdict_b = "WIN"
        elif diff_b_lo > 0.0 or wins_b == 0:
            verdict_b = "LOSS"
        else:
            verdict_b = "TIE"
        out["wins_vs_ext_b"] = wins_b
        out["diff_vs_ext_b_mean"] = diff_b
        out["diff_vs_ext_b_ci_low"] = diff_b_lo
        out["diff_vs_ext_b_ci_high"] = diff_b_hi
        out["verdict_vs_ext_b"] = verdict_b
    if ext_inf_regret is not None:
        wins_inf = int(sum(1 for m in means if m < ext_inf_regret))
        diff_inf = float(reg_mean - ext_inf_regret)
        diff_inf_lo = float(seed_ci_low - ext_inf_regret)
        diff_inf_hi = float(seed_ci_high - ext_inf_regret)
        if diff_inf_hi < 0.0:
            verdict_inf = "WIN"
        elif diff_inf_lo > 0.0 or wins_inf == 0:
            verdict_inf = "LOSS"
        else:
            verdict_inf = "TIE"
        out["wins_vs_ext_inf"] = wins_inf
        out["diff_vs_ext_inf_mean"] = diff_inf
        out["diff_vs_ext_inf_ci_low"] = diff_inf_lo
        out["diff_vs_ext_inf_ci_high"] = diff_inf_hi
        out["verdict_vs_ext_inf"] = verdict_inf
    if "regret_to_floor" in records[0]:
        fl_means = [float(r["regret_to_floor"]["mean"]) for r in records]
        fl_mean = float(np.mean(fl_means))
        fl_sd, fl_lo, fl_hi = _sample_sd_and_t_ci(fl_means)
        out["regret_to_floor_mean"] = fl_mean
        out["regret_to_floor_sd"] = fl_sd
        out["regret_to_floor_seed_ci_low"] = fl_lo
        out["regret_to_floor_seed_ci_high"] = fl_hi
        out["per_seed_regret_to_floor"] = [round(m, 4) for m in fl_means]
        if per_piece_fct_floor_regret is not None:
            out["gap_to_per_piece_fct_floor_mean"] = float(fl_mean - per_piece_fct_floor_regret)
    return out


def aggregate_anchor_seeds(records: list[dict[str, Any]]) -> dict[str, Any]:
    records = sorted(records, key=lambda r: int(r["seed"]))
    means = [float(r["regret"]["mean"]) for r in records]
    ci_lows = [float(r["regret"]["ci_low"]) for r in records]
    ci_highs = [float(r["regret"]["ci_high"]) for r in records]
    b_steps = [int(r["best_step"]) for r in records]
    reg_sd, seed_ci_low, seed_ci_high = _sample_sd_and_t_ci(means)
    out: dict[str, Any] = {
        "n_seeds": len(records),
        "seeds_list": [int(r["seed"]) for r in records],
        "regret_mean": float(np.mean(means)),
        "regret_sd": reg_sd,
        "seed_ci_low": seed_ci_low,
        "seed_ci_high": seed_ci_high,
        "mean_ci_low": float(np.mean(ci_lows)),
        "mean_ci_high": float(np.mean(ci_highs)),
        "raw_means": means,
        "raw_ci_lows": ci_lows,
        "raw_ci_highs": ci_highs,
        "per_seed_regret": [round(m, 4) for m in means],
        "per_seed_best_step": b_steps,
    }
    if "regret_to_floor" in records[0]:
        fl_means = [float(r["regret_to_floor"]["mean"]) for r in records]
        fl_sd, fl_lo, fl_hi = _sample_sd_and_t_ci(fl_means)
        out["regret_to_floor_mean"] = float(np.mean(fl_means))
        out["regret_to_floor_sd"] = fl_sd
        out["regret_to_floor_seed_ci_low"] = fl_lo
        out["regret_to_floor_seed_ci_high"] = fl_hi
        out["raw_floor_means"] = fl_means
        out["per_seed_regret_to_floor"] = [round(m, 4) for m in fl_means]
    return out


def recompute_anchor_dict_ddof1(anc_dict: dict[str, Any]) -> dict[str, Any]:
    """Ensures an existing anchor dict uses `ddof=1` sample SD and includes `seed_ci_low` / `seed_ci_high`."""
    means = [float(x) for x in anc_dict.get("raw_means", anc_dict.get("per_seed_regret", []))]
    if not means:
        return anc_dict
    reg_sd, s_lo, s_hi = _sample_sd_and_t_ci(means)
    anc_dict["n_seeds"] = len(means)
    if "seeds_list" not in anc_dict:
        anc_dict["seeds_list"] = list(range(len(means)))
    if "regret_mean" not in anc_dict or "raw_means" in anc_dict:
        anc_dict["regret_mean"] = float(np.mean(means))
    anc_dict["regret_sd"] = reg_sd
    anc_dict["seed_ci_low"] = s_lo
    anc_dict["seed_ci_high"] = s_hi
    anc_dict["per_seed_regret"] = [round(m, 4) for m in means]
    if "per_seed_regret_to_floor" in anc_dict or "raw_floor_means" in anc_dict:
        fl_means = [float(x) for x in anc_dict.get("raw_floor_means", anc_dict["per_seed_regret_to_floor"])]
        fl_sd, fl_lo, fl_hi = _sample_sd_and_t_ci(fl_means)
        if "raw_floor_means" in anc_dict:
            anc_dict["regret_to_floor_mean"] = float(np.mean(fl_means))
        anc_dict["regret_to_floor_sd"] = fl_sd
        anc_dict["regret_to_floor_seed_ci_low"] = fl_lo
        anc_dict["regret_to_floor_seed_ci_high"] = fl_hi
        anc_dict["per_seed_regret_to_floor"] = [round(m, 4) for m in fl_means]
    return anc_dict


def merge_two_anchor_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merges two anchor dicts by `seeds_list` when non-overlapping seeds are combined."""
    seeds_a = a.get("seeds_list", list(range(len(a.get("per_seed_regret", [])))))
    seeds_b = b.get("seeds_list", list(range(len(b.get("per_seed_regret", [])))))
    means_a = [float(x) for x in a.get("raw_means", a.get("per_seed_regret", []))]
    means_b = [float(x) for x in b.get("raw_means", b.get("per_seed_regret", []))]
    steps_a = [int(x) for x in a.get("per_seed_best_step", [])]
    steps_b = [int(x) for x in b.get("per_seed_best_step", [])]
    by_seed: dict[int, tuple[float, int]] = {}
    for s, m, st in zip(seeds_a, means_a, steps_a):
        by_seed[int(s)] = (m, st)
    for s, m, st in zip(seeds_b, means_b, steps_b):
        by_seed[int(s)] = (m, st)
    sorted_s = sorted(by_seed.keys())
    comb_means = [by_seed[s][0] for s in sorted_s]
    comb_steps = [by_seed[s][1] for s in sorted_s]
    out = copy.deepcopy(a)
    out["seeds_list"] = sorted_s
    out["raw_means"] = comb_means
    out["per_seed_best_step"] = comb_steps
    if "per_seed_regret_to_floor" in a and "per_seed_regret_to_floor" in b:
        fl_a = [float(x) for x in a.get("raw_floor_means", a["per_seed_regret_to_floor"])]
        fl_b = [float(x) for x in b.get("raw_floor_means", b["per_seed_regret_to_floor"])]
        fl_by_seed = {int(s): v for s, v in zip(seeds_a, fl_a)}
        for s, v in zip(seeds_b, fl_b):
            fl_by_seed[int(s)] = v
        out["raw_floor_means"] = [fl_by_seed[s] for s in sorted_s]
    return recompute_anchor_dict_ddof1(out)


def _paired_seed_comparison(
    recs_queried: list[dict[str, Any]],
    recs_oneshot: list[dict[str, Any]],
) -> dict[str, Any] | None:
    q_by_s = {int(r["seed"]): float(r["regret_quant"]["mean"]) for r in recs_queried}
    o_by_s = {int(r["seed"]): float(r["regret_quant"]["mean"]) for r in recs_oneshot}
    common = sorted(set(q_by_s.keys()) & set(o_by_s.keys()))
    if not common:
        return None
    diffs = [q_by_s[s] - o_by_s[s] for s in common]
    mean_d = float(np.mean(diffs))
    sd_d, ci_lo, ci_hi = _sample_sd_and_t_ci(diffs)
    if ci_hi < 0.0:
        verdict = "SIGNIFICANT_GAIN"
    elif ci_lo > 0.0:
        verdict = "SIGNIFICANT_HURT"
    else:
        verdict = "NS"
    return {
        "n_pairs": len(common),
        "seeds": common,
        "mean_diff": mean_d,
        "sd_ddof1": sd_d,
        "ci_low": ci_lo,
        "ci_high": ci_hi,
        "n_queried_better": int(sum(1 for d in diffs if d < 0.0)),
        "per_seed_diff": [round(d, 4) for d in diffs],
        "verdict": verdict,
    }


def compute_cell_summary(
    recs_by_arm: dict[str, list[dict[str, Any]]],
    aggs_by_arm: dict[str, dict[str, Any]],
    ext_b_regret: float,
    ext_inf_regret: float,
) -> dict[str, Any]:
    val_selected_arm = min(aggs_by_arm.keys(), key=lambda a: float(aggs_by_arm[a]["val_kl_mean"]))
    dev_min_arm = min(aggs_by_arm.keys(), key=lambda a: float(aggs_by_arm[a]["regret_mean"]))
    val_ag = aggs_by_arm[val_selected_arm]
    dev_ag = aggs_by_arm[dev_min_arm]
    out: dict[str, Any] = {
        "val_selected_arm": val_selected_arm,
        "val_selected_val_kl_mean": float(val_ag["val_kl_mean"]),
        "val_selected_dev_regret_mean": float(val_ag["regret_mean"]),
        "val_selected_dev_regret_sd": float(val_ag["regret_sd"]),
        "val_selected_seed_ci_low": float(val_ag["seed_ci_low"]),
        "val_selected_seed_ci_high": float(val_ag["seed_ci_high"]),
        "val_selected_wins_vs_ext_b": int(val_ag.get("wins_vs_ext_b", 0)),
        "val_selected_wins_vs_ext_inf": int(val_ag.get("wins_vs_ext_inf", 0)),
        "val_selected_diff_vs_ext_b": float(val_ag.get("diff_vs_ext_b_mean", 0.0)),
        "val_selected_verdict_vs_ext_b": str(val_ag.get("verdict_vs_ext_b", "TIE")),
        "val_selected_verdict_vs_ext_inf": str(val_ag.get("verdict_vs_ext_inf", "TIE")),
        "dev_min_arm": dev_min_arm,
        "dev_min_regret_mean": float(dev_ag["regret_mean"]),
        "val_matches_dev_min": bool(val_selected_arm == dev_min_arm),
        "ext_b_regret": float(ext_b_regret),
        "ext_inf_regret": float(ext_inf_regret),
    }
    if "queried_fsq" in recs_by_arm and "plain_fsq" in recs_by_arm:
        out["queried_minus_plain"] = _paired_seed_comparison(recs_by_arm["queried_fsq"], recs_by_arm["plain_fsq"])
    if "queried_masked_pretrain_fsq" in recs_by_arm and "masked_pretrain_fsq" in recs_by_arm:
        out["queried_masked_minus_masked"] = _paired_seed_comparison(
            recs_by_arm["queried_masked_pretrain_fsq"],
            recs_by_arm["masked_pretrain_fsq"],
        )
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
    allowed: set[str] = {"0.0000", "1.0000", "2.0000", "4.0000", "6.0000"}
    collect_4dec_strings(summary, allowed)
    found = re.findall(r"\b\d+\.\d{4}\b", readme_text)
    missing = [x for x in found if x not in allowed]
    return len(found), missing


def render_readme(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "# Learned Compression on `E2`, `E3-sybil`, `Z8`, and `E4` at `b in {1, 2, 4, 6}` Against Strong Per-Piece Anchors (`#3896` & `#3898`)"
    )
    lines.append("")
    lines.append("**Track: `EXPLORATORY` (`#3896` / `#3898` compression lane on `E2`, `E3-sybil`, `Z8`, `E4`)**")
    lines.append("")
    h_info = summary.get("harness", {})
    lines.append("Protocol, Provenance & Gate `#3898` Fixes:")
    lines.append(
        f"- **Code & Script Provenance:** `git_commit = {h_info.get('git_commit', 'unknown')}` (`git_dirty = {h_info.get('git_dirty', False)}`), `aggbench_sha256 = {h_info.get('sha256', 'unknown')}`, `script_sha256 = {h_info.get('script_sha256', 'unknown')}` (`{h_info.get('script_path', 'benchmarks/explore_compression_e2_e3_z8_e4.py')}`). All per-part slice summaries are committed under `results/explore/tt-compression-e2-e3-z8-e4-20261022/parts/`."
    )
    lines.append(
        "- **Fresh Episodes & Generator Match Verification:** All learned anchors and compression arms train on **fresh simulator episodes at every step** (`batch_size = 512`; zero fixed-pool memorization), with `best_val_kl` selected on disjoint `val` (`n_val = 1000`, `start = 50,000`) and scored on `dev` (`n_dev = 2000`, `start = 0`). `verify_torch_vs_aggbench_generator` verifies on `4,096` episodes per environment that `sample_env_torch` matches `aggbench.compression.make_cenv(env).sample('train', 4096)` in token shape, base rate, Bayes logit SD, and per-channel token moments."
    )
    lines.append(
        "- **Step Budget Disclosure (`#3898`):** In the `8,000`-step sweep below, each compressed arm ran `8,000` joint/pretrain steps (`512` episodes/step) plus `300` frozen-encoder discrete-decoder lock-in batches (`8,300` total batches vs `8,000` for the anchors, `+3.75%`). In the long-horizon convergence study (`24,000` and `48,000` steps), `--strict-step-budget` deducts the `300` lock-in batches from `finetune_steps` so compressed arms and anchors use **literally identical total batch counts** (`24,000` and `48,000`). All gaps against `8,000`-step anchors are **same-budget gaps** (lower bounds against converged `24k`/`48k` anchors, which are reported separately below)."
    )
    lines.append(
        "- **What `extremized_q` Gets vs the Learned Per-Piece Encoder (`#3898` Fix 1):** `extremized_q` (fit on `n_train = 2,000` at the same `b` and at `b = inf`) pools each agent's scalar report `local_statistic`. By design, the learned encoder reads the agent's own raw per-piece cells (`obs`, `mu`, `coverage`, `sum_info`, `n_seen`), so it can weight observations by per-piece precision (`mu^2`). On `E4`, `llr_quantizer` and `extremized_q` quantize the raw rollout sum (which retains drift bias `b` and recency weighting `1.5 * 0.7^(L-1-t)`), whereas the learned encoder computes debiased per-step increments (`inc_s / w_s`). Thus `llr_quantizer` has **decoder-side oracle structure only** (`[DECODER-ORACLE STRUCTURE ONLY; raw rollout sum on E4]`), and beating it on `E4` comes from debiasing the per-step local statistic rather than beating an optimal encoder+decoder oracle."
    )
    lines.append(
        "- **Quantizer & Causal Sequential Locality Verification (`#3898` Fix 4):** Finite Scalar Quantization (`FSQ`, Mentzer et al. `2309.15505`) at `b in {1, 2, 4, 6}` bits/agent (`2^b in {2, 4, 16, 64}` symbols). At eval (`not self.training`), `m = 2 * z_hard - 1` directly (`b` values of `+-1`, zero float32 STE rounding leak), verified on `dev` by `Channel(b).send`, `check_message_locality` on `64` dev episodes at the live sequential queries, AND `verify_arm_locality`'s sequential `ConsensusQueryNet` rollout causality check (proving that perturbing future agents `j > t` has zero effect on `m_{<=t}` or `q_{<=t}`, and perturbing past agents `j < t` while clamping `m_{<t}` has zero effect on `q_t` or `m_t`)."
    )
    lines.append(
        "- **Sample Statistics & Arm Selection (`#3898` Fix 2 & Fix 3):** All across-seed standard deviations (`Seed SD`) use **`ddof = 1`** (`np.std(..., ddof=1)`), and all `95%` intervals (`Across-Seed 95% t-CI`) are across-seed Student-$t$ intervals (`mean +- t_{0.975, n-1} * sd_ddof1 / sqrt(n)`). Best-arm selection per `(env, b)` cell is performed **strictly on `val` (`val_kl_mean`)**, never on `dev`."
    )
    lines.append("")

    # Headline Summary Table 1: Per-Cell Win/Loss/Tie vs extremized_q(b) and extremized_q(b=inf), using Val-Selected Arm
    lines.append("## Headline Summary 1: Per-Cell Verdict vs `extremized_q(b)` and `extremized_q(b=inf)` (`Val`-Selected Arm, `#3898` Fix 1 & Fix 3)")
    lines.append("")
    lines.append(
        "- **Summary of Bar 1 (Beat Extremizing):** Learned compression **wins on `E3-sybil` at every `b in {1, 2, 4, 6}` and on `E4` at every `b in {1, 2, 4, 6}`** (and at `b >= 2` also beats unquantized `extremized_q(b=inf)`), **wins on `E2` at `b >= 4`**, **loses on `E2` at `b = 1`**, **ties on `E2` at `b = 2`**, **loses on `Z8` at `b = 1`**, and **ties on `Z8` at `b in {2, 4, 6}`**."
    )
    b6_parts: list[str] = []
    for env_name in ["E2", "E3-sybil", "Z8", "E4"]:
        ed_tmp = summary.get("envs", {}).get(env_name, {})
        cs6 = ed_tmp.get("cell_summaries", {}).get("6")
        pfct_tmp = ed_tmp.get("anchors", {}).get("anchor_per_piece_fct_fresh")
        if cs6 and pfct_tmp:
            b6_parts.append(
                f"`{env_name}`: `{cs6['val_selected_dev_regret_mean']:.4f}` vs `{pfct_tmp['regret_mean']:.4f}` (`n={pfct_tmp['n_seeds']}`)"
            )
    b6_summary_str = "; ".join(b6_parts) if b6_parts else "see per-environment tables below"
    lines.append(
        f"- **Summary of Bar 2 (Match Uncompressed Anchor):** **Not met.** At `b = 6`, the `val`-selected learned arm remains behind the same-budget Per-Piece FCT anchor ({b6_summary_str}), and behind converged `24k`/`48k` anchors."
    )
    lines.append("")
    lines.append(
        "| Env | `b` | `n` Seeds | `Val`-Selected Arm | `Dev`-Min Arm (`Val == Dev?`) | `Val`-Selected `Dev` Regret ± SD (`ddof=1`) | Across-Seed `95% t-CI` | `extremized_q(b)` | `extremized_q(b=inf)` | Diff vs `ext_q(b)` `[95% t-CI]` | Seed Wins vs `ext_q(b)` | Verdict vs `ext_q(b)` | Verdict vs `ext_q(b=inf)` |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for env_name in ["E2", "E3-sybil", "Z8", "E4"]:
        if env_name not in summary["envs"]:
            continue
        ed = summary["envs"][env_name]
        for b_str, cs in ed.get("cell_summaries", {}).items():
            v_arm = cs["val_selected_arm"]
            d_arm = cs["dev_min_arm"]
            v_ag = ed["aggregated"][b_str][v_arm]
            match_str = "YES" if cs["val_matches_dev_min"] else f"NO (`{d_arm}`: `{cs['dev_min_regret_mean']:.4f}`)"
            lines.append(
                f"| `{env_name}` | `{b_str}` | `{v_ag['n_seeds']}` | `{v_arm}` | {match_str} | "
                f"`{cs['val_selected_dev_regret_mean']:.4f}` ± `{cs['val_selected_dev_regret_sd']:.4f}` | "
                f"`[`{cs['val_selected_seed_ci_low']:.4f}`, `{cs['val_selected_seed_ci_high']:.4f}`]` | "
                f"`{cs['ext_b_regret']:.4f}` | `{cs['ext_inf_regret']:.4f}` | "
                f"`{v_ag['diff_vs_ext_b_mean']:+.4f}` `[`{v_ag['diff_vs_ext_b_ci_low']:+.4f}`, `{v_ag['diff_vs_ext_b_ci_high']:+.4f}`]` | "
                f"`{cs['val_selected_wins_vs_ext_b']}/{v_ag['n_seeds']}` | **{cs['val_selected_verdict_vs_ext_b']}** | "
                f"**{cs['val_selected_verdict_vs_ext_inf']}** |"
            )
    lines.append("")

    # Headline Summary Table 2: Paired-by-Seed Query Gain CIs
    lines.append("## Headline Summary 2: Paired-by-Seed Query Gain (`queried_fsq - plain_fsq` & `queried_masked - masked`, `#3898` Fix 2)")
    lines.append("")
    lines.append(
        "- Within each `(env, b)` cell and seed `s`, `queried_fsq` and `plain_fsq` share the exact same fresh-episode training stream (`seed * 10007 + 99 + b * 31`). The table below reports the paired-by-seed difference `d_s = regret(queried, s) - regret(one_shot, s)` (negative = query helps), sample SD (`ddof = 1`), and paired Student-$t$ `95% CI`."
    )
    lines.append("")
    lines.append(
        "| Env | `b` | `n` Pairs | `queried_fsq - plain_fsq` Mean ± SD (`ddof=1`) | Paired `95% t-CI` (`queried - plain`) | Per-Seed Diffs | Verdict (`queried vs plain`) | `queried_masked - masked` Mean ± SD (`ddof=1`) | Paired `95% t-CI` (`q_masked - masked`) | Verdict (`q_masked vs masked`) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for env_name in ["E2", "E3-sybil", "Z8", "E4"]:
        if env_name not in summary["envs"]:
            continue
        ed = summary["envs"][env_name]
        for b_str, cs in ed.get("cell_summaries", {}).items():
            qp = cs.get("queried_minus_plain")
            qm = cs.get("queried_masked_minus_masked")
            if qp is None or qm is None:
                continue
            qp_seeds = ", ".join(f"`{v:+.4f}`" for v in qp["per_seed_diff"])
            lines.append(
                f"| `{env_name}` | `{b_str}` | `{qp['n_pairs']}` | "
                f"`{qp['mean_diff']:+.4f}` ± `{qp['sd_ddof1']:.4f}` | "
                f"`[`{qp['ci_low']:+.4f}`, `{qp['ci_high']:+.4f}`]` | {qp_seeds} | **{qp['verdict']}** | "
                f"`{qm['mean_diff']:+.4f}` ± `{qm['sd_ddof1']:.4f}` | "
                f"`[`{qm['ci_low']:+.4f}`, `{qm['ci_high']:+.4f}`]` | **{qm['verdict']}** |"
            )
    lines.append("")

    # Per-Environment Detailed Tables
    for env_name in ["E2", "E3-sybil", "Z8", "E4"]:
        if env_name not in summary["envs"]:
            continue
        ed = summary["envs"][env_name]
        anc = ed["anchors"]
        lines.append(f"## Environment: `{env_name}` (`N = {ed['n_agents']}` agents, `steps = {ed['steps']}` fresh)")
        lines.append("")
        pfct = anc["anchor_per_piece_fct_fresh"]
        senc = anc["anchor_slot_enc_tr_fresh"]
        ctrl = anc["control_untrained_slot_enc"]
        lines.append(
            f"- **Same-Budget (`{ed['steps']}` fresh-episode steps) Uncompressed Anchors & Controls (`n_dev = {ed['n_dev']}`, `n_seeds = {pfct['n_seeds']}`, `ddof = 1`):**"
        )
        lines.append(f"  - `prior (b=0)` = `{anc['prior_kl']:.4f}`")
        if "achievable_trace_floor_kl" in anc:
            lines.append(
                f"  - `achievable_trace_floor (exact trace ceiling vs Bayes)` = `{anc['achievable_trace_floor_kl']:.4f}` (on `E4`, rollouts observe noisy traces so even the exact trace ceiling has `{anc['achievable_trace_floor_kl']:.4f}` KL to full-context Bayes; both regret vs Bayes and regret vs trace floor are reported below)"
            )
            lines.append(
                f"  - `anchor_per_piece_fct_fresh (b=inf, same-budget {ed['steps']} steps)` = `{pfct['regret_mean']:.4f}` ± `{pfct['regret_sd']:.4f}` (`95% t-CI` `[`{pfct['seed_ci_low']:.4f}`, `{pfct['seed_ci_high']:.4f}`]`; `{pfct['regret_to_floor_mean']:.4f}` ± `{pfct['regret_to_floor_sd']:.4f}` vs trace floor; `per_seed_best_step` = `{pfct['per_seed_best_step']}`)"
            )
            lines.append(
                f"  - `anchor_slot_enc_tr_fresh (b=inf, same-budget {ed['steps']} steps)` = `{senc['regret_mean']:.4f}` ± `{senc['regret_sd']:.4f}` (`95% t-CI` `[`{senc['seed_ci_low']:.4f}`, `{senc['seed_ci_high']:.4f}`]`; `{senc['regret_to_floor_mean']:.4f}` ± `{senc['regret_to_floor_sd']:.4f}` vs trace floor; `per_seed_best_step` = `{senc['per_seed_best_step']}`)"
            )
        else:
            lines.append(
                f"  - `anchor_per_piece_fct_fresh (b=inf, same-budget {ed['steps']} steps)` = `{pfct['regret_mean']:.4f}` ± `{pfct['regret_sd']:.4f}` (`95% t-CI` `[`{pfct['seed_ci_low']:.4f}`, `{pfct['seed_ci_high']:.4f}`]`; `per_seed_best_step` = `{pfct['per_seed_best_step']}`)"
            )
            lines.append(
                f"  - `anchor_slot_enc_tr_fresh (b=inf, same-budget {ed['steps']} steps)` = `{senc['regret_mean']:.4f}` ± `{senc['regret_sd']:.4f}` (`95% t-CI` `[`{senc['seed_ci_low']:.4f}`, `{senc['seed_ci_high']:.4f}`]`; `per_seed_best_step` = `{senc['per_seed_best_step']}`)"
            )
        lines.append(
            f"  - `control_untrained_slot_enc (random weights control)` = `{ctrl['regret_mean']:.4f}` ± `{ctrl['regret_sd']:.4f}`"
        )
        ext_s = ", ".join(f"`b={b_k}`: `{v:.4f}`" for b_k, v in anc["extremized_q"].items())
        llr_s = ", ".join(f"`b={b_k}`: `{v:.4f}`" for b_k, v in anc["llr_quantizer_oracle"].items())
        qllr_s = ", ".join(f"`b={b_k}`: `{v:.4f}`" for b_k, v in anc["queried_llr_qinf_oracle"].items())
        lines.append(f"  - `extremized_q (blind closed-form on scalar report)`: {ext_s}")
        if env_name == "E4":
            lines.append(
                f"  - `llr_quantizer [DECODER-ORACLE STRUCTURE ONLY; quantizes raw biased rollout sum, not per-step debiased]`: {llr_s}"
            )
            lines.append(
                f"  - `queried_llr_qinf [DECODER-ORACLE STRUCTURE ONLY; quantizes raw biased rollout sum, not per-step debiased]`: {qllr_s}"
            )
        else:
            lines.append(f"  - `llr_quantizer [DECODER-ORACLE STRUCTURE ONLY]`: {llr_s}")
            lines.append(f"  - `queried_llr_qinf [DECODER-ORACLE STRUCTURE ONLY]`: {qllr_s}")
        lines.append("")

        if env_name == "E4":
            lines.append(
                "| `b` | Learned Arm | `Val` KL Mean | Mean `Dev` Regret vs Bayes | Mean Regret vs Trace Floor | Same-Budget `gap(b)` vs FCT (`"
                + f"{pfct['regret_mean']:.4f}"
                + "`) | Same-Budget `gap(b)` vs Slot-Enc (`"
                + f"{senc['regret_mean']:.4f}"
                + "`) | Seed SD (`ddof=1`) | Across-Seed `95% t-CI` | Per-Seed Regrets | Best Steps | Mean Agent Entropy `H(M_i)` (bits) | Distinct Symbols |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        else:
            lines.append(
                "| `b` | Learned Arm | `Val` KL Mean | Mean `Dev` Regret (`regret_quant`) | Same-Budget `gap(b)` vs FCT (`"
                + f"{pfct['regret_mean']:.4f}"
                + "`) | Same-Budget `gap(b)` vs Slot-Enc (`"
                + f"{senc['regret_mean']:.4f}"
                + "`) | Seed SD (`ddof=1`) | Across-Seed `95% t-CI` | Per-Seed Regrets | Best Steps | Escaped (`val_kl < 0.9 * prior`) | Mean Agent Entropy `H(M_i)` (bits) | Distinct Symbols |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        for b_str, by_arm in ed["aggregated"].items():
            val_sel = ed["cell_summaries"][b_str]["val_selected_arm"]
            for arm_name, ag in by_arm.items():
                arm_label = f"**`{arm_name}` (val-best)**" if arm_name == val_sel else f"`{arm_name}`"
                s_regs = ", ".join(f"`{v:.4f}`" for v in ag["per_seed_regret"])
                s_bsteps = ", ".join(f"`{v}`" for v in ag["per_seed_best_step"])
                s_syms = ", ".join(f"`{v}`" for v in ag["per_seed_distinct_symbols"])
                if env_name == "E4":
                    lines.append(
                        f"| `{b_str}` | {arm_label} | `{ag['val_kl_mean']:.4f}` | `{ag['regret_mean']:.4f}` | `{ag['regret_to_floor_mean']:.4f}` | "
                        f"`{ag['gap_to_per_piece_fct_mean']:+.4f}` | `{ag['gap_to_slot_enc_tr_mean']:+.4f}` | "
                        f"`{ag['regret_sd']:.4f}` | `[`{ag['seed_ci_low']:.4f}`, `{ag['seed_ci_high']:.4f}`]` | "
                        f"{s_regs} | {s_bsteps} | `{ag['mean_agent_entropy_bits']:.4f} ± {ag['mean_agent_entropy_sd']:.4f}` | {s_syms} |"
                    )
                else:
                    lines.append(
                        f"| `{b_str}` | {arm_label} | `{ag['val_kl_mean']:.4f}` | `{ag['regret_mean']:.4f}` | "
                        f"`{ag['gap_to_per_piece_fct_mean']:+.4f}` | `{ag['gap_to_slot_enc_tr_mean']:+.4f}` | "
                        f"`{ag['regret_sd']:.4f}` | `[`{ag['seed_ci_low']:.4f}`, `{ag['seed_ci_high']:.4f}`]` | "
                        f"{s_regs} | {s_bsteps} | `{ag['n_escaped']}/{ag['n_seeds']}` | "
                        f"`{ag['mean_agent_entropy_bits']:.4f} ± {ag['mean_agent_entropy_sd']:.4f}` | {s_syms} |"
                    )
        lines.append("")

    # Long-Horizon Convergence Section (24k / 48k steps, if present)
    if summary.get("long_horizon"):
        lines.append("## Long-Horizon Convergence & Strictly Matched Step-Budget Study (`24,000` & `48,000` Steps, `#3898` Fix 3)")
        lines.append("")
        lines.append(
            "- In all long-horizon runs below, `--strict-step-budget` is enabled so the `300` discrete-decoder lock-in batches are deducted from `finetune_steps` (`total_batches == steps` identically for both anchors and compressed arms)."
        )
        lines.append("")
        for h_key, h_block in sorted(summary["long_horizon"].items(), key=lambda kv: int(kv[0].split("_")[-1])):
            h_steps = int(h_key.split("_")[-1])
            for env_name, ed_h in h_block.get("envs", {}).items():
                anc_h = ed_h["anchors"]
                pfct_h = anc_h.get("anchor_per_piece_fct_fresh")
                senc_h = anc_h.get("anchor_slot_enc_tr_fresh")
                lines.append(f"### `{env_name}` at `steps = {h_steps}` (`{h_steps * 512:,}` fresh episodes, strictly matched budget)")
                lines.append("")
                if pfct_h:
                    lines.append(
                        f"- `anchor_per_piece_fct_fresh ({h_steps} steps, n={pfct_h['n_seeds']})` = `{pfct_h['regret_mean']:.4f}` ± `{pfct_h['regret_sd']:.4f}` (`95% t-CI` `[`{pfct_h['seed_ci_low']:.4f}`, `{pfct_h['seed_ci_high']:.4f}`]`; `per_seed_regret` = `{[round(v, 4) for v in pfct_h['per_seed_regret']]}`, `per_seed_best_step` = `{pfct_h['per_seed_best_step']}`)"
                    )
                if senc_h:
                    lines.append(
                        f"- `anchor_slot_enc_tr_fresh ({h_steps} steps, n={senc_h['n_seeds']})` = `{senc_h['regret_mean']:.4f}` ± `{senc_h['regret_sd']:.4f}` (`95% t-CI` `[`{senc_h['seed_ci_low']:.4f}`, `{senc_h['seed_ci_high']:.4f}`]`; `per_seed_regret` = `{[round(v, 4) for v in senc_h['per_seed_regret']]}`, `per_seed_best_step` = `{senc_h['per_seed_best_step']}`)"
                    )
                if ed_h.get("aggregated"):
                    lines.append("")
                    lines.append(
                        "| `b` | Learned Arm | `n` Seeds | `Val` KL Mean | Mean `Dev` Regret ± SD (`ddof=1`) | Across-Seed `95% t-CI` | `gap(b)` vs Same-Step FCT | Per-Seed Regrets | Best Steps | Paired `queried - plain` (`[95% t-CI]`) |"
                    )
                    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                    for b_str, by_arm_h in ed_h["aggregated"].items():
                        cs_h = ed_h.get("cell_summaries", {}).get(b_str, {})
                        qp_h = cs_h.get("queried_minus_plain")
                        qp_str = (
                            f"`{qp_h['mean_diff']:+.4f}` `[`{qp_h['ci_low']:+.4f}`, `{qp_h['ci_high']:+.4f}`]` (**{qp_h['verdict']}**)"
                            if qp_h
                            else "n/a"
                        )
                        for arm_name, ag_h in by_arm_h.items():
                            s_regs = ", ".join(f"`{v:.4f}`" for v in ag_h["per_seed_regret"])
                            s_bsteps = ", ".join(f"`{v}`" for v in ag_h["per_seed_best_step"])
                            lines.append(
                                f"| `{b_str}` | `{arm_name}` | `{ag_h['n_seeds']}` | `{ag_h['val_kl_mean']:.4f}` | "
                                f"`{ag_h['regret_mean']:.4f}` ± `{ag_h['regret_sd']:.4f}` | "
                                f"`[`{ag_h['seed_ci_low']:.4f}`, `{ag_h['seed_ci_high']:.4f}`]` | "
                                f"`{ag_h['gap_to_per_piece_fct_mean']:+.4f}` | {s_regs} | {s_bsteps} | {qp_str} |"
                            )
                lines.append("")

    return "\n".join(lines) + "\n"


def _merge_part_dicts(dir_paths: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    merged: dict[str, Any] | None = None
    total_elapsed = 0.0
    part_harnesses: list[dict[str, Any]] = []
    for d_str in dir_paths:
        p_path = pathlib.Path(d_str)
        if p_path.is_dir():
            p_path = p_path / "summary.json"
        part = json.loads(p_path.read_text())
        total_elapsed += float(part.get("elapsed_sec", 0.0))
        if "harness" in part:
            part_harnesses.append({"source": str(p_path), "harness": part["harness"]})
        if merged is None:
            merged = copy.deepcopy(part)
        else:
            for env_k, env_v in part.get("envs", {}).items():
                if env_k not in merged["envs"]:
                    merged["envs"][env_k] = copy.deepcopy(env_v)
                else:
                    target_env = merged["envs"][env_k]
                    for ak, av in env_v.get("anchors", {}).items():
                        if av is None:
                            continue
                        if ak in ("extremized_q", "llr_quantizer_oracle", "queried_llr_qinf_oracle") and isinstance(av, dict):
                            target_env["anchors"].setdefault(ak, {}).update(av)
                        elif ak in ("anchor_per_piece_fct_fresh", "anchor_slot_enc_tr_fresh", "control_untrained_slot_enc") and isinstance(av, dict):
                            if target_env["anchors"].get(ak) is None:
                                target_env["anchors"][ak] = copy.deepcopy(av)
                            else:
                                target_env["anchors"][ak] = merge_two_anchor_dicts(target_env["anchors"][ak], av)
                        else:
                            target_env["anchors"][ak] = copy.deepcopy(av)
                    for b_k, b_v in env_v.get("per_seed", {}).items():
                        target_b = target_env["per_seed"].setdefault(b_k, {})
                        for arm_k, recs in b_v.items():
                            if arm_k not in target_b:
                                target_b[arm_k] = copy.deepcopy(recs)
                            else:
                                by_s = {int(r["seed"]): r for r in target_b[arm_k]}
                                for r in recs:
                                    by_s[int(r["seed"])] = copy.deepcopy(r)
                                target_b[arm_k] = [by_s[s] for s in sorted(by_s.keys())]
    assert merged is not None
    return merged, part_harnesses, total_elapsed


def _reconcile_merged_envs(merged: dict[str, Any], verify_generators: bool = True) -> None:
    device = torch.device("cpu")
    for env_k, ed in merged["envs"].items():
        anc = ed["anchors"]
        # Ensure closed-form b=inf references are present
        if "inf" not in anc.get("extremized_q", {}) or "inf" not in anc.get("llr_quantizer_oracle", {}):
            cf_inf = run_closed_form(
                env_k,
                bits=tuple(int(b) for b in ed.get("per_seed", {}).keys()) or (1, 2, 4, 6),
                n_train=2000,
                n_dev=ed.get("n_dev", 2000),
                M=128,
                n_boot=200,
                verbose=False,
                arms=("llr_quantizer", "extremized_q"),
            )
            anc.setdefault("extremized_q", {})["inf"] = float(cf_inf["arms"]["extremized_q"]["inf"]["regret"]["mean"])
            llr_inf = float(cf_inf["arms"]["llr_quantizer"]["inf"]["regret"]["mean"])
            anc.setdefault("llr_quantizer_oracle", {})["inf"] = llr_inf
            anc.setdefault("queried_llr_qinf_oracle", {})["inf"] = llr_inf

        # Sort bit keys in closed-form dicts
        for cf_k in ("extremized_q", "llr_quantizer_oracle", "queried_llr_qinf_oracle"):
            if isinstance(anc.get(cf_k), dict):
                items = anc[cf_k]
                sorted_keys = sorted([k for k in items.keys() if k != "inf"], key=lambda s: int(s))
                if "inf" in items:
                    sorted_keys.append("inf")
                anc[cf_k] = {k: float(items[k]) for k in sorted_keys}

        for ak in ("anchor_per_piece_fct_fresh", "anchor_slot_enc_tr_fresh", "control_untrained_slot_enc"):
            if isinstance(anc.get(ak), dict):
                anc[ak] = recompute_anchor_dict_ddof1(anc[ak])

        pfct = anc.get("anchor_per_piece_fct_fresh")
        senc = anc.get("anchor_slot_enc_tr_fresh")
        pfct_mean = float(pfct["regret_mean"]) if pfct else 0.0
        senc_mean = float(senc["regret_mean"]) if senc else 0.0
        pfct_fl = pfct.get("regret_to_floor_mean") if pfct else None
        ext_inf = float(anc["extremized_q"]["inf"])

        if verify_generators and "generator_check" not in ed:
            ed["generator_check"] = verify_torch_vs_aggbench_generator(env_k, device=device, n_check=4096)

        sorted_b = sorted(ed.get("per_seed", {}).keys(), key=lambda s: int(s))
        new_agg: dict[str, Any] = {}
        new_ps: dict[str, Any] = {}
        cell_summaries: dict[str, Any] = {}
        for b_k in sorted_b:
            new_ps[b_k] = ed["per_seed"][b_k]
            new_agg[b_k] = {}
            ext_b = float(anc["extremized_q"][b_k])
            for arm_name, recs in new_ps[b_k].items():
                recs_sorted = sorted(recs, key=lambda r: int(r["seed"]))
                for r in recs_sorted:
                    r["gap_to_per_piece_fct"] = float(r["regret_quant"]["mean"] - pfct_mean)
                    r["gap_to_slot_enc_tr"] = float(r["regret_quant"]["mean"] - senc_mean)
                new_ps[b_k][arm_name] = recs_sorted
                new_agg[b_k][arm_name] = aggregate_arm_seeds(
                    recs_sorted,
                    per_piece_fct_regret=pfct_mean,
                    slot_enc_tr_regret=senc_mean,
                    per_piece_fct_floor_regret=pfct_fl,
                    ext_b_regret=ext_b,
                    ext_inf_regret=ext_inf,
                )
            cell_summaries[b_k] = compute_cell_summary(
                new_ps[b_k],
                new_agg[b_k],
                ext_b_regret=ext_b,
                ext_inf_regret=ext_inf,
            )
        ed["per_seed"] = new_ps
        ed["aggregated"] = new_agg
        ed["cell_summaries"] = cell_summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", default=["E2", "E3-sybil", "Z8", "E4"])
    parser.add_argument("--bits", nargs="+", type=int, default=[1, 2, 4, 6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "plain_fsq",
            "masked_pretrain_fsq",
            "queried_fsq",
            "queried_masked_pretrain_fsq",
        ],
    )
    parser.add_argument("--steps", type=int, default=8_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-dev", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2_000)
    parser.add_argument("--merge-dirs", nargs="+", default=None)
    parser.add_argument("--long-horizon-dirs", nargs="+", default=None)
    parser.add_argument("--skip-anchors", action="store_true")
    parser.add_argument("--only-anchors", action="store_true")
    parser.add_argument("--strict-step-budget", action="store_true")
    parser.add_argument("--out-dir", type=str, default="results/explore/tt-compression-e2-e3-z8-e4-20261022")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_dirs:
        merged, part_harnesses, total_elapsed = _merge_part_dicts(args.merge_dirs)
        _reconcile_merged_envs(merged, verify_generators=True)

        if args.long_horizon_dirs:
            lh_by_steps: dict[int, list[str]] = {}
            for d_str in args.long_horizon_dirs:
                p_path = pathlib.Path(d_str)
                if p_path.is_dir():
                    p_path = p_path / "summary.json"
                part_lh = json.loads(p_path.read_text())
                st_val = int(part_lh.get("steps", 24000))
                lh_by_steps.setdefault(st_val, []).append(str(p_path))
                if "harness" in part_lh:
                    part_harnesses.append({"source": str(p_path), "harness": part_lh["harness"]})
            merged["long_horizon"] = {}
            for st_val, paths_st in sorted(lh_by_steps.items()):
                m_lh, _, el_lh = _merge_part_dicts(paths_st)
                _reconcile_merged_envs(m_lh, verify_generators=False)
                total_elapsed += el_lh
                merged["long_horizon"][f"steps_{st_val}"] = m_lh

        merged["harness"] = benchmark_harness_info()
        merged["envs_list"] = list(merged["envs"].keys())
        merged["bits"] = [1, 2, 4, 6]
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
        args.steps = 20
        args.n_val = 100
        args.n_dev = 100
        args.batch_size = 64
        args.eval_every = 10
        args.n_boot = 50
        args.seeds = [0, 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(2)
    else:
        torch.set_float32_matmul_precision("high")

    summary: dict[str, Any] = {
        "schema": "aggbench.compression-e2-e3-z8-e4/v2",
        "harness": benchmark_harness_info(),
        "track": "EXPLORATORY",
        "regime": "fresh",
        "visibility": "none",
        "alignment": "unaligned",
        "bits": args.bits,
        "seeds": args.seeds,
        "arms": [] if args.only_anchors else args.arms,
        "steps": args.steps,
        "strict_step_budget": bool(args.strict_step_budget),
        "batch_size": args.batch_size,
        "device": str(device),
        "envs": {},
    }

    t_start = time.time()
    for env_name in args.envs:
        env = make_cenv(env_name)
        val_b = env.sample("dev", args.n_val, start=50_000)
        dv_b = env.sample("dev", args.n_dev, start=0)
        cal_b = env.sample("train", 512 if args.smoke else 4096, start=9_000_000)

        X_val = torch.tensor(agent_tokens(val_b), dtype=torch.float32, device=device)
        bayes_val = np.asarray(val_b.bayes["logit"], dtype=np.float64)
        X_dv = torch.tensor(agent_tokens(dv_b), dtype=torch.float32, device=device)
        bayes_dv = np.asarray(dv_b.bayes["logit"], dtype=np.float64)
        X_cal = torch.tensor(agent_tokens(cal_b), dtype=torch.float32, device=device)

        l0 = float(dv_b.public["prior_logit"])
        prior_kl = float(summarize(prior_forecast(dv_b), dv_b, n_boot=args.n_boot)["kl"]["mean"])
        floor_dv: np.ndarray | None = None
        floor_kl: float | None = None
        if env_name == "E4":
            fl_fc = achievable_floor(dv_b)
            floor_dv = np.asarray(fl_fc["logit"], dtype=np.float64)
            floor_kl = float(summarize(fl_fc, dv_b, n_boot=args.n_boot)["kl"]["mean"])

        gen_check = verify_torch_vs_aggbench_generator(env_name, device=device, n_check=4096)

        # 1. Closed-form references on dev (including b = inf)
        cf_res = run_closed_form(
            env_name,
            bits=tuple(args.bits),
            n_train=200 if args.smoke else 2000,
            n_dev=args.n_dev,
            M=64 if args.smoke else 512,
            n_boot=args.n_boot,
            verbose=False,
            arms=("llr_quantizer", "queried_llr_qinf", "extremized_q"),
        )
        b_keys_cf = [str(b) for b in args.bits] + ["inf"]
        ext_q_refs = {k: float(cf_res["arms"]["extremized_q"][k]["regret"]["mean"]) for k in b_keys_cf}
        llr_q_refs = {k: float(cf_res["arms"]["llr_quantizer"][k]["regret"]["mean"]) for k in b_keys_cf}
        qllr_refs = {str(b): float(cf_res["arms"]["queried_llr_qinf"][str(b)]["regret"]["mean"]) for b in args.bits}
        qllr_refs["inf"] = llr_q_refs["inf"]

        # 2. Train same-steps Uncompressed Anchors (Per-Piece FCT + Slot-Enc Transformer + Untrained Control)
        if not args.skip_anchors:
            pfct_recs = []
            senc_recs = []
            ctrl_recs = []
            for seed in args.seeds:
                t_a = time.time()
                r_pfct = run_per_piece_fct_anchor(
                    env_name=env_name,
                    seed=seed,
                    X_val=X_val,
                    bayes_val=bayes_val,
                    X_dv=X_dv,
                    bayes_dv=bayes_dv,
                    floor_dv=floor_dv,
                    X_cal=X_cal,
                    l0=l0,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    eval_every=args.eval_every,
                    n_boot=args.n_boot,
                )
                pfct_recs.append(r_pfct)
                print(
                    f"[{env_name} anchor] anchor_per_piece_fct_fresh seed={seed}: regret={r_pfct['regret']['mean']:.4f} "
                    f"best_step={r_pfct['best_step']} ({time.time() - t_a:.1f}s)",
                    flush=True,
                )

                t_b = time.time()
                r_senc, r_ctrl = run_slot_enc_uncompressed_anchor(
                    env_name=env_name,
                    seed=seed,
                    X_val=X_val,
                    bayes_val=bayes_val,
                    X_dv=X_dv,
                    bayes_dv=bayes_dv,
                    floor_dv=floor_dv,
                    X_cal=X_cal,
                    l0=l0,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    eval_every=args.eval_every,
                    n_boot=args.n_boot,
                )
                senc_recs.append(r_senc)
                ctrl_recs.append(r_ctrl)
                print(
                    f"[{env_name} anchor] anchor_slot_enc_tr_fresh   seed={seed}: regret={r_senc['regret']['mean']:.4f} "
                    f"best_step={r_senc['best_step']} ({time.time() - t_b:.1f}s)",
                    flush=True,
                )

            pfct_agg: dict[str, Any] | None = aggregate_anchor_seeds(pfct_recs)
            senc_agg: dict[str, Any] | None = aggregate_anchor_seeds(senc_recs)
            ctrl_agg: dict[str, Any] | None = aggregate_anchor_seeds(ctrl_recs)
            pfct_reg_val = float(pfct_agg["regret_mean"])
            senc_reg_val = float(senc_agg["regret_mean"])
            pfct_fl_val = pfct_agg.get("regret_to_floor_mean")
        else:
            pfct_agg = None
            senc_agg = None
            ctrl_agg = None
            pfct_reg_val = 0.0
            senc_reg_val = 0.0
            pfct_fl_val = None

        anchors: dict[str, Any] = {
            "prior_kl": prior_kl,
            "anchor_per_piece_fct_fresh": pfct_agg,
            "anchor_slot_enc_tr_fresh": senc_agg,
            "control_untrained_slot_enc": ctrl_agg,
            "extremized_q": ext_q_refs,
            "llr_quantizer_oracle": llr_q_refs,
            "queried_llr_qinf_oracle": qllr_refs,
        }
        if floor_kl is not None:
            anchors["achievable_trace_floor_kl"] = floor_kl

        per_seed_records: dict[str, dict[str, list[dict[str, Any]]]] = {}
        aggregated: dict[str, dict[str, Any]] = {}
        cell_summaries: dict[str, Any] = {}

        if args.only_anchors:
            summary["envs"][env_name] = {
                "n_agents": int(X_val.shape[1]),
                "steps": args.steps,
                "n_val": args.n_val,
                "n_dev": args.n_dev,
                "generator_check": gen_check,
                "anchors": anchors,
                "aggregated": aggregated,
                "per_seed": per_seed_records,
                "cell_summaries": cell_summaries,
            }
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            continue

        # 3. Learned compression arms across b in args.bits and args.seeds
        for b in args.bits:
            b_key = str(b)
            per_seed_records[b_key] = {}
            aggregated[b_key] = {}
            for arm_name in args.arms:
                per_seed_records[b_key][arm_name] = []
                for seed in args.seeds:
                    t0 = time.time()
                    rec = run_compression_arm(
                        env_name=env_name,
                        arm_name=arm_name,
                        b=b,
                        seed=seed,
                        X_val=X_val,
                        bayes_val=bayes_val,
                        X_dv=X_dv,
                        bayes_dv=bayes_dv,
                        floor_dv=floor_dv,
                        X_cal=X_cal,
                        dv_batch=dv_b,
                        l0=l0,
                        prior_kl=prior_kl,
                        per_piece_fct_regret=pfct_reg_val,
                        slot_enc_tr_regret=senc_reg_val,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        eval_every=args.eval_every,
                        n_boot=args.n_boot,
                        strict_step_budget=args.strict_step_budget,
                    )
                    per_seed_records[b_key][arm_name].append(rec)
                    print(
                        f"[{env_name} b={b}] {arm_name:28s} seed={seed}: regret={rec['regret_quant']['mean']:.4f} "
                        f"gap_pfct={rec['gap_to_per_piece_fct']:+.4f} gap_senc={rec['gap_to_slot_enc_tr']:+.4f} "
                        f"best_step={rec['best_step']} H={rec['symbol_metrics']['mean_agent_entropy_bits']:.4f}b "
                        f"syms={rec['symbol_metrics']['distinct_symbols_total']}/{2**b} ({time.time()-t0:.1f}s)",
                        flush=True,
                    )
                aggregated[b_key][arm_name] = aggregate_arm_seeds(
                    per_seed_records[b_key][arm_name],
                    per_piece_fct_regret=pfct_reg_val,
                    slot_enc_tr_regret=senc_reg_val,
                    per_piece_fct_floor_regret=pfct_fl_val,
                    ext_b_regret=ext_q_refs[b_key],
                    ext_inf_regret=ext_q_refs["inf"],
                )
                summary["envs"][env_name] = {
                    "n_agents": int(X_val.shape[1]),
                    "steps": args.steps,
                    "n_val": args.n_val,
                    "n_dev": args.n_dev,
                    "generator_check": gen_check,
                    "anchors": anchors,
                    "aggregated": aggregated,
                    "per_seed": per_seed_records,
                    "cell_summaries": cell_summaries,
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            cell_summaries[b_key] = compute_cell_summary(
                per_seed_records[b_key],
                aggregated[b_key],
                ext_b_regret=ext_q_refs[b_key],
                ext_inf_regret=ext_q_refs["inf"],
            )
            summary["envs"][env_name]["cell_summaries"] = cell_summaries
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    summary["elapsed_sec"] = round(time.time() - t_start, 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.skip_anchors and not args.only_anchors:
        readme_text = render_readme(summary)
        n_checked, missing = verify_readme_numbers(readme_text, summary)
        if missing:
            raise RuntimeError(f"Unverified 4-decimal numbers in README.md: {missing}")
        (out_dir / "README.md").write_text(readme_text)
        print(f"Done in {summary['elapsed_sec']}s; verified {n_checked}/{n_checked} 4-dec numbers.", flush=True)
    else:
        print(f"Done partial in {summary['elapsed_sec']}s.", flush=True)


if __name__ == "__main__":
    main()
