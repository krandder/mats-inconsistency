#!/usr/bin/env python3
"""Compressed Aggregation Benchmark for TraderTransformer & Queried-View (#3871-#3874, #3878).

Tests Kelvin's TraderTransformer and Queried-View designs under an explicit b-bit per-agent
uplink budget (m_t in [0, 2^b) verified by aggbench.compression.Channel(b).send) and
unbounded read-only consensus->agent queries (q = inf) vs one-shot (q = 0), self-summary,
D-VIB, and Chain-of-Agents fixed-width state chains.

Environments: E2, E3-sybil, Z9-p2 (2 readings/agent on 4-bit parity), E4, Z8 (and E3-collude).
Bit budgets: b in {1, 2, 4, 8} (plus b=0 prior and b=inf uncompressed anchors).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import pathlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench.compression import (
    Channel,
    PiecesPerAgent,
    agent_tokens,
    check_message_locality,
    curve,
    env_family,
    gain,
    make_cenv,
    raw_code_forecast,
    run_closed_form,
)
from aggbench.metrics import bernoulli_logloss, per_episode, summarize, summarize_values
from aggbench.report import achievable_floor, prior_forecast

GAUSS_SIGMA2 = 1.0 / 3.0  # peak power 1 (tanh), noise 1/3: capacity 1/2 log2(1 + 3) = 1 bit per dimension

Z9_CURRICULUM = (
    {"n_bits": 2, "n_distractor_bits": 4, "n_distractor_agents": 2},
    {"n_bits": 3, "n_distractor_bits": 3, "n_distractor_agents": 1},
    {},
)


def _token_pool(env: Any, split: str, n: int, start: int = 0, k_max: int | None = None) -> tuple[Any, ...]:
    """Returns (tokens, truth, aux targets (n, k_max - 1) prefix parities, aux mask, batch)."""
    batch = env.sample(split, n, start=start)
    X = torch.tensor(agent_tokens(batch), dtype=torch.float32)
    Y = torch.tensor(np.asarray(batch.truth, np.float32))
    if not k_max:
        return X, Y, None, None, batch
    k = env.base.config.n_bits
    w = np.asarray(batch.oracle["w"])
    bits = (w[:, None] >> np.arange(k)) & 1
    pref = np.cumsum(bits, 1) % 2
    A = np.zeros((n, k_max - 1), np.float32)
    A[:, : k - 1] = pref[:, : k - 1]
    M = np.zeros(k_max - 1, np.float32)
    M[: k - 1] = 1.0
    return X, Y, torch.tensor(A), torch.tensor(M), batch


def quantize_bits_ste(u: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> torch.Tensor:
    """Maps continuous bit logits u (..., b) to {-1, +1}^b at eval ('quant') or soft+STE during training.

    During training:
    - Combines bounded tanh(u) + Gaussian channel noise N(0, 1/3) (1 bit/dim capacity) with
      straight-through hard quantization sign(u) ramped in over training so the decoder is
      calibrated on the exact {-1, +1}^b hypercube vertices used at evaluation time.
    """
    if mode == "quant":
        return torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
    s = torch.tanh(u)
    hard = torch.where(u >= 0, torch.ones_like(u), -torch.ones_like(u))
    ste = s + (hard - s).detach()
    if mode == "noisy":
        sigma = math.sqrt(GAUSS_SIGMA2)
        return s + sigma * torch.randn_like(s)
    # In training: blend noisy channel (for smooth early gradients on XOR) into exact STE sign
    sigma = math.sqrt(GAUSS_SIGMA2) * max(0.0, 1.0 - 1.25 * step_frac)
    noisy_s = s + sigma * torch.randn_like(s)
    w_ste = min(1.0, max(0.2, step_frac * 1.2))
    return (1.0 - w_ste) * noisy_s + w_ste * ste


def bits_to_int_symbols(u_quant: torch.Tensor) -> np.ndarray:
    """Packs {-1, +1}^b tensor (..., b) into integer symbols in [0, 2^b) and verifies via Channel(b)."""
    b = u_quant.shape[-1]
    bits01 = (u_quant >= 0).to(torch.int64)
    weights = (2 ** torch.arange(b, device=u_quant.device, dtype=torch.int64))
    syms = (bits01 * weights).sum(dim=-1).cpu().numpy().astype(np.int64)
    Channel(b).send(syms)
    return syms


class ConsensusSetDecoder(nn.Module):
    """Reads ONLY the N agents' b-bit messages U (B, N, b_dim) and outputs aggregate logit + optional aux logits."""

    def __init__(self, b_dim: int, width: int = 128, layers: int = 4, heads: int = 4, n_aux: int = 0, causal_pos: bool = False) -> None:
        super().__init__()
        self.emb = nn.Linear(b_dim, width)
        self.pos = nn.Embedding(32, width) if causal_pos else None
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 1)
        self.aux = nn.Linear(width, n_aux) if n_aux > 0 else None

    def forward(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.emb(m)
        if self.pos is not None:
            h = h + self.pos(torch.arange(m.shape[1], device=m.device))
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)
        z = self.norm(self.dec(h)[:, 0])
        return self.out(z)[:, 0], (self.aux(z) if self.aux is not None else None)


class OneShotCompressionNet(nn.Module):
    """Used for self_summary_b and dvib_oneshot_q0 (q = 0: no query from consensus before compression)."""

    def __init__(self, in_dim: int, b: int | None, width: int = 128, layers: int = 4, n_aux: int = 0, cont_dim: int = 16) -> None:
        super().__init__()
        self.b = b
        self.b_dim = cont_dim if b is None else b
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.b_dim),
        )
        # Residual connection from own local log-odds (last token coordinate) into bit 0
        self.llr_skip = nn.Linear(1, self.b_dim)
        self.own_head = nn.Sequential(nn.Linear(self.b_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.dec = ConsensusSetDecoder(self.b_dim, width=width, layers=layers, heads=4, n_aux=n_aux, causal_pos=False)

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_mean) / self.x_sd
        return self.enc(xn) + self.llr_skip(x[..., -1:])

    def message(self, x: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> torch.Tensor:
        u = self.encode_raw(x)
        if self.b is None:
            return torch.tanh(u)
        return quantize_bits_ste(u, mode=mode, step_frac=step_frac)

    def forward_all(self, x: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        m = self.message(x, mode=mode, step_frac=step_frac)
        logit, z_aux = self.dec(m)
        return logit, z_aux, m

    def own_logit(self, m: torch.Tensor) -> torch.Tensor:
        return self.own_head(m)[..., 0]


class FixedStateChainNet(nn.Module):
    """Chain-of-Agents (S = b) and Fixed-State (S = N*b) learned chain: state_t = Quant_S(Enc(x_t, state_{t-1}))."""

    def __init__(self, in_dim: int, state_bits: int | None, n_aux: int = 0, cont_dim: int = 16) -> None:
        super().__init__()
        self.state_bits = state_bits
        self.s_dim = cont_dim if state_bits is None else min(state_bits, 32)
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))
        self.s0 = nn.Parameter(torch.zeros(1, self.s_dim))
        self.step_mlp = nn.Sequential(
            nn.Linear(in_dim + self.s_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.s_dim),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim + self.s_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.s_dim),
        )
        self.out = nn.Sequential(nn.Linear(self.s_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.aux = nn.Linear(self.s_dim, n_aux) if n_aux > 0 else None

    def forward_all(self, x: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        bsz, n_tr, _ = x.shape
        xn = (x - self.x_mean) / self.x_sd
        s = self.s0.expand(bsz, -1)
        states = []
        for t in range(n_tr):
            inp = torch.cat([xn[:, t], s], dim=-1)
            delta = self.step_mlp(inp)
            g = torch.sigmoid(self.gate_mlp(inp))
            u = (1.0 - 0.5 * g) * s + delta
            if self.state_bits is None:
                s = torch.tanh(u)
            else:
                s = quantize_bits_ste(u, mode=mode, step_frac=step_frac)
            states.append(s)
        all_s = torch.stack(states, dim=1)
        logit = self.out(s)[:, 0]
        z_aux = self.aux(s) if self.aux is not None else None
        return logit, z_aux, all_s


class QueriedLearnedNet(nn.Module):
    """Kelvin's Queried-View Design (q = unbounded read-only query of consensus state -> b-bit uplink message).

    Strict separation:
    1. Consensus state at step t is ONLY the discrete b-bit messages M_{<t} = (u_1, ..., u_{t-1}) in {-1, +1}^{(t-1) x b}.
    2. Query module Query(M_{<t}) is a pure read-only function of M_{<t} (implemented via a GRU over M_{<t})
       that outputs query vector q_t in R^q_dim and running consensus log-odds estimate l_{t-1}(M_{<t}).
    3. Trader t's local encoder Enc(x_t, q_t, l_{t-1}) reads ONLY x_t and (q_t, l_{t-1}) and outputs a b-bit
       uplink message u_t in {-1, +1}^b, which is appended to the consensus state M_{<=t}.
    4. Final forecast is decoded solely from the consensus state M_{1..N} = (u_1, ..., u_N).
    """

    def __init__(self, in_dim: int, b: int | None, width: int = 128, layers: int = 4, q_dim: int = 64, n_aux: int = 0, cont_dim: int = 16) -> None:
        super().__init__()
        self.b = b
        self.b_dim = cont_dim if b is None else b
        self.q_dim = q_dim
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))

        # Pure read-only query generator over consensus state M_{<t}
        self.q0 = nn.Parameter(torch.zeros(1, q_dim))
        self.consensus_gru = nn.GRUCell(self.b_dim, q_dim)
        self.query_logit_head = nn.Linear(q_dim, 1)

        # Agent-local query-conditioned encoder Enc(x_t, q_t, l_{t-1}) -> b bits
        self.enc = nn.Sequential(
            nn.Linear(in_dim + q_dim + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.b_dim),
        )
        # Adaptive LLR centering skip: w_r * r_t - kappa * l_{t-1}
        self.llr_skip = nn.Linear(2, self.b_dim)

        # Consensus decoder over M_{1..N} = (u_1, ..., u_N)
        self.dec = ConsensusSetDecoder(self.b_dim, width=width, layers=layers, heads=4, n_aux=n_aux, causal_pos=True)

    def encode_step(self, x_t: torch.Tensor, q_t: torch.Tensor, l_prev: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> torch.Tensor:
        xn_t = (x_t - self.x_mean) / self.x_sd
        r_t = x_t[..., -1:]
        inp = torch.cat([xn_t, q_t, l_prev], dim=-1)
        u_t = self.enc(inp) + self.llr_skip(torch.cat([r_t, l_prev], dim=-1))
        if self.b is None:
            return torch.tanh(u_t)
        return quantize_bits_ste(u_t, mode=mode, step_frac=step_frac)

    def forward_all(self, x: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        bsz, n_tr, _ = x.shape
        q_state = self.q0.expand(bsz, -1)
        msgs = []
        pref_logits = []
        for t in range(n_tr):
            # Read-only query from consensus state M_{<t}
            l_prev = self.query_logit_head(q_state)
            u_t = self.encode_step(x[:, t], q_state, l_prev, mode=mode, step_frac=step_frac)
            msgs.append(u_t)
            # Consensus state updates ONLY via the b-bit message u_t
            q_state = self.consensus_gru(u_t, q_state)
            pref_logits.append(self.query_logit_head(q_state)[:, 0])
        m_all = torch.stack(msgs, dim=1)
        logit, z_aux = self.dec(m_all)
        return logit, z_aux, m_all, torch.stack(pref_logits, dim=1)


class CompressedTraderTransformerNet(nn.Module):
    """Kelvin's TraderTransformer under a b-Bit Uplink Bottleneck (#3874: 'each agent attends to the
    full KV cache, but appends only a b-bit key/value').

    At step t:
    1. Trader t queries the consensus KV cache formed EXCLUSIVELY by earlier traders' b-bit messages
       M_{<t} = (u_1, ..., u_{t-1}) in {-1, +1}^{(t-1) x b} via 2-layer multi-head cross-attention
       (using fast F.scaled_dot_product_attention). Trader t NEVER sees earlier traders' raw tokens!
    2. Trader t's local encoder combines its private token x_t and the attended KV context ctx_t to
       produce a b-bit uplink message u_t in {-1, +1}^b (verified by Channel(b).send at eval),
       appending ONLY u_t to the consensus KV cache M_{<=t}.
    3. Final consensus forecast is decoded purely from the N b-bit messages M_{1..N} = (u_1, ..., u_N).
    """

    def __init__(self, in_dim: int, b: int | None, width: int = 128, layers: int = 4, heads: int = 4, q_dim: int = 64, n_aux: int = 0, cont_dim: int = 16) -> None:
        super().__init__()
        self.b = b
        self.b_dim = cont_dim if b is None else b
        self.q_dim = q_dim
        self.heads = heads
        self.head_dim = q_dim // heads
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))

        # KV cache embedding over earlier b-bit messages M_{<t}
        self.msg_emb = nn.Linear(self.b_dim, q_dim)
        self.pos_emb = nn.Embedding(32, q_dim)
        self.q_init = nn.Linear(in_dim, q_dim)

        # 2-layer fast SDPA cross-attention over M_{<t}
        self.attn_layers = 2
        self.ln_q = nn.ModuleList(nn.LayerNorm(q_dim) for _ in range(self.attn_layers))
        self.ln_kv = nn.ModuleList(nn.LayerNorm(q_dim) for _ in range(self.attn_layers))
        self.wq = nn.ModuleList(nn.Linear(q_dim, q_dim) for _ in range(self.attn_layers))
        self.wkv = nn.ModuleList(nn.Linear(q_dim, 2 * q_dim) for _ in range(self.attn_layers))
        self.wo = nn.ModuleList(nn.Linear(q_dim, q_dim) for _ in range(self.attn_layers))
        self.ffn = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(q_dim), nn.Linear(q_dim, 2 * q_dim), nn.GELU(), nn.Linear(2 * q_dim, q_dim))
            for _ in range(self.attn_layers)
        )
        self.query_logit_head = nn.Linear(q_dim, 1)

        # Local encoder Enc(x_t, ctx_t, l_{t-1}) -> b bits
        self.enc = nn.Sequential(
            nn.Linear(in_dim + q_dim + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, self.b_dim),
        )
        self.llr_skip = nn.Linear(2, self.b_dim)

        # Consensus decoder reading ONLY the b-bit messages M_{1..N} = (u_1, ..., u_N)
        self.dec = ConsensusSetDecoder(self.b_dim, width=width, layers=layers, heads=heads, n_aux=n_aux, causal_pos=True)

    def encode_trader_step(
        self,
        x_t: torch.Tensor,
        past_msgs: list[torch.Tensor],
        t_idx: int,
        mode: str = "train",
        step_frac: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x_t.shape[0]
        xn_t = (x_t - self.x_mean) / self.x_sd
        r_t = x_t[..., -1:]
        if not past_msgs:
            ctx_t = torch.zeros(bsz, self.q_dim, device=x_t.device, dtype=x_t.dtype)
            l_prev = torch.zeros(bsz, 1, device=x_t.device, dtype=x_t.dtype)
        else:
            m_seq = torch.stack(past_msgs, dim=1)  # (B, t, b_dim) — ONLY b-bit messages!
            t_len = m_seq.shape[1]
            pos_ids = torch.arange(t_len, device=x_t.device)
            kv_base = self.msg_emb(m_seq) + self.pos_emb(pos_ids)
            h_q = self.q_init(xn_t).unsqueeze(1)  # (B, 1, q_dim)
            for l in range(self.attn_layers):
                qn = self.ln_q[l](h_q)
                kvn = self.ln_kv[l](kv_base)
                q_h = self.wq[l](qn).view(bsz, 1, self.heads, self.head_dim).transpose(1, 2)
                kv_h = self.wkv[l](kvn).view(bsz, t_len, 2, self.heads, self.head_dim)
                k_h = kv_h[:, :, 0].transpose(1, 2)
                v_h = kv_h[:, :, 1].transpose(1, 2)
                attn_out = F.scaled_dot_product_attention(q_h, k_h, v_h).transpose(1, 2).reshape(bsz, 1, self.q_dim)
                h_q = h_q + self.wo[l](attn_out)
                h_q = h_q + self.ffn[l](h_q)
            ctx_t = h_q[:, 0]
            l_prev = self.query_logit_head(ctx_t)

        inp = torch.cat([xn_t, ctx_t, l_prev], dim=-1)
        u_t = self.enc(inp) + self.llr_skip(torch.cat([r_t, l_prev], dim=-1))
        if self.b is None:
            return torch.tanh(u_t), l_prev[:, 0]
        return quantize_bits_ste(u_t, mode=mode, step_frac=step_frac), l_prev[:, 0]

    def forward_all(self, x: torch.Tensor, mode: str = "train", step_frac: float = 0.0) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        _, n_tr, _ = x.shape
        past_msgs: list[torch.Tensor] = []
        pref_logits: list[torch.Tensor] = []
        for t in range(n_tr):
            u_t, l_prev = self.encode_trader_step(x[:, t], past_msgs, t, mode=mode, step_frac=step_frac)
            past_msgs.append(u_t)
            pref_logits.append(l_prev)
        m_all = torch.stack(past_msgs, dim=1)
        logit, z_aux = self.dec(m_all)
        return logit, z_aux, m_all, torch.stack(pref_logits, dim=1)


def random_scale_matched_copy(net: nn.Module, seed: int) -> nn.Module:
    rnd = copy.deepcopy(net)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for p in rnd.parameters():
            if p.numel() > 1:
                mean, sd = p.mean().cpu(), p.std().cpu()
            else:
                mean, sd = torch.zeros((), dtype=p.dtype), p.abs().sum().cpu()
            noise = torch.randn(p.shape, generator=gen, dtype=p.dtype)
            p.copy_((mean + sd * noise).to(p.device))
    return rnd


def verify_net_locality(net: nn.Module, dv_batch: Any, device: str, b: int | None) -> bool:
    """Verifies that agent i's message depends ONLY on its own inputs and earlier messages."""
    if b is None:
        return True
    # Check that perturbing agent j != i (OneShot) or j > i (Queried / TraderTransformer) never changes agent i's message
    tok16 = torch.tensor(agent_tokens(dv_batch)[:16], dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        if isinstance(net, OneShotCompressionNet):
            base_m = bits_to_int_symbols(net.message(tok16, mode="quant"))
            for i in range(tok16.shape[1]):
                tok_pert = tok16.clone()
                others = [j for j in range(tok16.shape[1]) if j != i]
                tok_pert[:, others] += torch.randn_like(tok_pert[:, others]) * 0.5
                pert_m = bits_to_int_symbols(net.message(tok_pert, mode="quant"))
                if not np.array_equal(base_m[:, i], pert_m[:, i]):
                    raise AssertionError(f"OneShotCompressionNet locality failed for agent {i}")
        elif isinstance(net, (QueriedLearnedNet, CompressedTraderTransformerNet)):
            _, _, m_base_raw, _ = net.forward_all(tok16, mode="quant")
            base_m = bits_to_int_symbols(m_base_raw)
            for i in range(tok16.shape[1]):
                tok_pert = tok16.clone()
                # Perturb all future agents j > i: agent i's message must be 100% invariant!
                if i + 1 < tok16.shape[1]:
                    tok_pert[:, i + 1 :] += torch.randn_like(tok_pert[:, i + 1 :]) * 0.5
                    _, _, m_pert_raw, _ = net.forward_all(tok_pert, mode="quant")
                    pert_m = bits_to_int_symbols(m_pert_raw)
                    if not np.array_equal(base_m[:, : i + 1], pert_m[:, : i + 1]):
                        raise AssertionError(f"Causal query locality failed for agent {i}")
    return True


def train_and_eval_arm(
    arm_name: str,
    env_name: str,
    b: int | None,
    seed: int,
    pools: dict[str, Any],
    dv_tuple: tuple[Any, ...],
    steps: int = 3000,
    batch_size: int = 512,
    lr: float = 1e-3,
    width: int = 128,
    layers: int = 4,
    aux_weight: float = 0.0,
    aux_warmup: int = 0,
    curriculum: list[tuple[str, int]] | None = None,
    device: str = "cuda",
    n_boot: int = 500,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xf = pools["final"][0]
    in_dim = int(Xf.shape[-1])
    n_tr = int(Xf.shape[1])
    mean = Xf.reshape(-1, in_dim).mean(0)
    sd = Xf.reshape(-1, in_dim).std(0).clamp_min(1e-3)
    n_aux = 0 if pools["final"][2] is None else int(pools["final"][2].shape[1])

    dv_X, _, _, _, dv_batch = dv_tuple
    l0 = float(dv_batch.public["prior_logit"])
    bce = F.binary_cross_entropy_with_logits

    def build_model() -> nn.Module:
        torch.manual_seed(seed)
        if arm_name in ("self_summary_b", "random_encoder_b", "dvib_oneshot_q0", "dvib_oneshot_q0_curr_aux"):
            m = OneShotCompressionNet(in_dim, b, width=width, layers=layers, n_aux=n_aux)
        elif arm_name == "coa_chain_b":
            m = FixedStateChainNet(in_dim, b, n_aux=n_aux)
        elif arm_name == "fixed_state_nb":
            s_bits = None if b is None else n_tr * b
            m = FixedStateChainNet(in_dim, s_bits, n_aux=n_aux)
        elif arm_name == "queried_learned_qinf":
            m = QueriedLearnedNet(in_dim, b, width=width, layers=layers, q_dim=64, n_aux=n_aux)
        elif arm_name in ("trader_transformer_b_qinf", "trader_transformer_b_qinf_curr_aux"):
            m = CompressedTraderTransformerNet(in_dim, b, width=width, layers=layers, heads=4, n_aux=n_aux)
        else:
            raise ValueError(f"Unknown arm: {arm_name}")
        m.x_mean.copy_(mean)
        m.x_sd.copy_(sd)
        return m.to(device)

    net = build_model()

    def predict_logits(model: nn.Module, mode: str = "quant") -> tuple[np.ndarray, np.ndarray | None]:
        model.eval()
        preds = []
        raw_msgs = []
        with torch.no_grad():
            for s in range(0, len(dv_X), 1024):
                xb = dv_X[s : s + 1024].to(device)
                out_tuple = model.forward_all(xb, mode=mode, step_frac=1.0)
                preds.append(out_tuple[0].double().cpu().numpy())
                raw_msgs.append(out_tuple[2])
        all_m = torch.cat(raw_msgs, dim=0)
        syms = None
        if b is not None and arm_name != "fixed_state_nb":
            syms = bits_to_int_symbols(all_m)
        elif b is not None and arm_name == "fixed_state_nb":
            # Verify state bits channel
            s_dim = all_m.shape[-1]
            bits01 = (all_m >= 0).to(torch.int64)
            w_vec = (2 ** torch.arange(s_dim, device=all_m.device, dtype=torch.int64))
            syms = (bits01 * w_vec).sum(dim=-1).cpu().numpy().astype(np.int64)
            Channel(s_dim).send(syms)
        return l0 + np.concatenate(preds, axis=0), syms

    # Control 1: Untrained Init
    untrained_logits, _ = predict_logits(net, mode="quant")
    untrained_metrics = summarize({"logit": untrained_logits}, dv_batch, n_boot=n_boot)

    def aux_loss_term(z_aux: torch.Tensor | None, a_tuple: tuple[torch.Tensor, torch.Tensor] | None, step_t: int) -> torch.Tensor | float:
        if z_aux is None or a_tuple is None or aux_weight <= 0.0 or step_t >= aux_warmup:
            return 0.0
        a_targ, a_mask = a_tuple
        w = aux_weight * (1.0 - step_t / max(1, aux_warmup))
        return w * (bce(z_aux, a_targ, reduction="none") * a_mask).sum(-1).mean() / a_mask.sum().clamp_min(1.0)

    def run_stages(params: list[nn.Parameter], stage_list: list[tuple[str, int]], loss_closure: Any) -> None:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        total_st = sum(st for _, st in stage_list)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(total_st, 10), pct_start=0.2)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t_cur = 0
        net.train()
        for pool_key, n_st in stage_list:
            X_p, Y_p, A_p, M_p, _ = pools[pool_key]
            X_pd = X_p.to(device)
            Y_pd = Y_p.to(device)
            A_pd = None if A_p is None else A_p.to(device)
            M_pd = None if M_p is None else M_p.to(device)
            idx_all = torch.stack([torch.randint(0, len(X_p), (batch_size,), generator=gen) for _ in range(n_st)]).to(device)
            for s_idx in range(n_st):
                idx = idx_all[s_idx]
                xb = X_pd[idx]
                yb = Y_pd[idx]
                ab = None if A_pd is None else (A_pd[idx], M_pd)
                s_frac = t_cur / max(1, total_st - 1)
                loss = loss_closure(xb, yb, ab, s_frac, t_cur)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                t_cur += 1

    t0 = time.time()
    if arm_name == "self_summary_b":
        assert isinstance(net, OneShotCompressionNet)
        enc_params = list(net.enc.parameters()) + list(net.llr_skip.parameters()) + list(net.own_head.parameters())
        run_stages(
            enc_params,
            [("final", steps)],
            lambda xb, yb, ab, sf, tc: bce(
                net.own_logit(net.message(xb, mode="train", step_frac=sf)) + l0,
                yb[:, None].expand(-1, xb.shape[1]),
            ),
        )
        for p in enc_params:
            p.requires_grad_(False)
        dec_params = [p for p in net.dec.parameters()]
        run_stages(
            dec_params,
            [("final", steps)],
            lambda xb, yb, ab, sf, tc: bce(net.dec(net.message(xb, mode="quant", step_frac=1.0))[0] + l0, yb),
        )
    elif arm_name == "random_encoder_b":
        assert isinstance(net, OneShotCompressionNet)
        enc_params = list(net.enc.parameters()) + list(net.llr_skip.parameters()) + list(net.own_head.parameters())
        for p in enc_params:
            p.requires_grad_(False)
        dec_params = [p for p in net.dec.parameters()]
        run_stages(
            dec_params,
            [("final", steps)],
            lambda xb, yb, ab, sf, tc: bce(net.dec(net.message(xb, mode="quant", step_frac=1.0))[0] + l0, yb),
        )
    else:
        use_curr_aux = arm_name.endswith("_curr_aux")
        stages = curriculum if (curriculum is not None and use_curr_aux) else [("final", steps)]

        def e2e_loss(xb: torch.Tensor, yb: torch.Tensor, ab: Any, sf: float, tc: int) -> torch.Tensor:
            out_t = net.forward_all(xb, mode="train", step_frac=sf)
            logit, z_aux = out_t[0], out_t[1]
            # B11 fix: all end-to-end arms optimize the exact same outcome objective without unmatched prefix loss
            return bce(logit + l0, yb) + (aux_loss_term(z_aux, ab, tc) if use_curr_aux else 0.0)

        run_stages(list(net.parameters()), stages, e2e_loss)

    dt = time.time() - t0
    verify_net_locality(net, dv_batch, device, b)

    trained_logits, syms = predict_logits(net, mode="quant")
    trained_metrics = summarize({"logit": trained_logits}, dv_batch, n_boot=n_boot)

    # B9: compute mean per-agent symbol entropy in bits
    sym_entropy_bits: float | None = None
    if syms is not None and b is not None and arm_name != "fixed_state_nb":
        k_levels = 2 ** min(b, 16)
        h_list = []
        for i_ag in range(syms.shape[1]):
            cnt = np.bincount(np.clip(syms[:, i_ag], 0, k_levels - 1), minlength=k_levels).astype(np.float64)
            p_i = cnt / max(1.0, cnt.sum())
            nz = p_i > 0
            h_list.append(float(-np.sum(p_i[nz] * np.log2(p_i[nz]))))
        sym_entropy_bits = float(np.mean(h_list))

    # Control 2: Scale-matched Random Network
    rnd_net = random_scale_matched_copy(net, seed + 100)
    rnd_logits, _ = predict_logits(rnd_net, mode="quant")
    rnd_metrics = summarize({"logit": rnd_logits}, dv_batch, n_boot=n_boot)

    fl = achievable_floor(dv_batch) if env_family(dv_batch) not in ("tabular", "grouped") else dict(dv_batch.bayes)
    ll_floor = bernoulli_logloss(fl["logit"], dv_batch.truth)
    reg_to_floor = summarize_values(bernoulli_logloss(trained_logits, dv_batch.truth) - ll_floor, n_boot)

    return {
        "arm": arm_name,
        "b": b,
        "seed": seed,
        "regret": float(trained_metrics["kl"]["mean"]),
        "regret_ci_low": float(trained_metrics["kl"]["ci_low"]),
        "regret_ci_high": float(trained_metrics["kl"]["ci_high"]),
        "regret_to_floor": float(reg_to_floor["mean"]),
        "logloss": float(trained_metrics["logloss"]["mean"]),
        "untrained_init_regret": float(untrained_metrics["kl"]["mean"]),
        "random_network_regret": float(rnd_metrics["kl"]["mean"]),
        "distinct_symbols": int(len(np.unique(syms))) if syms is not None else None,
        "symbol_entropy_bits": sym_entropy_bits,
        "train_seconds": float(dt),
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "logits": trained_logits,
        "methods_entry": {
            "trained": trained_metrics,
            "untrained_init": untrained_metrics,
            "random_network": rnd_metrics,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=str, default="E2,E3-sybil,Z9-p2,E4,Z8")
    ap.add_argument("--bits", type=str, default="1,2,4,8")
    ap.add_argument("--only-listed-bits", action="store_true", help="Run only the bit values listed in --bits (supports 'inf')")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--n-train", type=int, default=100000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--z9-steps", type=int, default=4500)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--commit", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_list = [e.strip() for e in args.envs.split(",") if e.strip()]
    raw_bits = [None if b.strip().lower() == "inf" else int(b.strip()) for b in args.bits.split(",") if b.strip()]
    bit_list = [1, 2, 4, 8] if args.only_listed_bits else ([b for b in raw_bits if b is not None] or [1, 2, 4, 8])
    b_run_list = raw_bits if args.only_listed_bits else (bit_list + [None])
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    out_path = pathlib.Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base_arm_names = [
        "self_summary_b",
        "random_encoder_b",
        "dvib_oneshot_q0",
        "coa_chain_b",
        "fixed_state_nb",
        "queried_learned_qinf",
        "trader_transformer_b_qinf",
    ]

    summary: dict[str, Any] = {
        "suite": "tt_compression",
        "git_commit": args.commit,
        "partial": False,
        "n_train": args.n_train,
        "n_dev": args.n_dev,
        "steps": args.steps,
        "z9_steps": args.z9_steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "layers": args.layers,
        "bits": bit_list,
        "seeds": seeds,
        "envs": {},
    }

    for env_name in env_list:
        t_env0 = time.time()
        print(f"\n[tt_compression] === Environment {env_name} ===", flush=True)
        cf_res = run_closed_form(env_name, bits=tuple(bit_list), n_train=2000, n_dev=args.n_dev, M=1024, n_boot=500, verbose=True)

        env = make_cenv(env_name)
        is_z9 = env_name == "Z9-p2"
        k_max = 4 if is_z9 else None
        pools = {"final": _token_pool(env, "train", args.n_train, k_max=k_max)}
        if is_z9:
            for j, ov in enumerate(Z9_CURRICULUM[:-1]):
                pools[f"k{j + 2}"] = _token_pool(PiecesPerAgent("Z9", 2, **ov), "train", min(30000, args.n_train), k_max=k_max)
        dv_tuple = _token_pool(env, "dev", args.n_dev, k_max=k_max)
        dv_batch = dv_tuple[4]
        n_agents = int(dv_tuple[0].shape[1])

        s_tot = args.z9_steps if is_z9 else args.steps
        curric = [("k2", s_tot // 5), ("k3", s_tot // 5), ("final", s_tot - 2 * (s_tot // 5))] if is_z9 else None
        aux_w = 0.5 if is_z9 else 0.0
        aux_warm = int(0.75 * s_tot) if is_z9 else 0

        learned_arm_names = list(base_arm_names)
        if is_z9:
            learned_arm_names.extend(["dvib_oneshot_q0_curr_aux", "trader_transformer_b_qinf_curr_aux"])

        learned_by_arm_b: dict[str, dict[str, Any]] = {a: {} for a in learned_arm_names}
        gains_vs_self_by_arm_b: dict[str, dict[str, Any]] = {a: {} for a in learned_arm_names if a != "self_summary_b"}
        gains_vs_rand_by_arm_b: dict[str, dict[str, Any]] = {a: {} for a in learned_arm_names if a != "random_encoder_b"}
        gains_vs_q0_by_arm_b: dict[str, dict[str, Any]] = {
            a: {} for a in ("queried_learned_qinf", "trader_transformer_b_qinf")
        }

        for b_val in b_run_list:
            b_key = "inf" if b_val is None else str(b_val)
            seed_outputs: dict[str, list[dict[str, Any]]] = {a: [] for a in learned_arm_names}
            for seed in seeds:
                for arm_name in learned_arm_names:
                    res_arm = train_and_eval_arm(
                        arm_name=arm_name,
                        env_name=env_name,
                        b=b_val,
                        seed=seed,
                        pools=pools,
                        dv_tuple=dv_tuple,
                        steps=s_tot,
                        batch_size=args.batch_size,
                        lr=1e-3,
                        width=args.width,
                        layers=args.layers,
                        aux_weight=aux_w,
                        aux_warmup=aux_warm,
                        curriculum=curric,
                        device=device,
                        n_boot=500,
                    )
                    seed_outputs[arm_name].append(res_arm)
                    print(
                        f"  [{env_name} b={b_key} seed={seed}] {arm_name:32s}: regret={res_arm['regret']:.4f} "
                        f"(untrained={res_arm['untrained_init_regret']:.4f}, rnd={res_arm['random_network_regret']:.4f}, "
                        f"syms={res_arm['distinct_symbols']}, {res_arm['train_seconds']:.1f}s)",
                        flush=True,
                    )

            # Summarize across seeds and compute paired gains over self_summary_b, random_encoder_b, and dvib_oneshot_q0
            for arm_name in learned_arm_names:
                recs = seed_outputs[arm_name]
                regs = [r["regret"] for r in recs]
                r_fls = [r["regret_to_floor"] for r in recs]
                unt_regs = [r["untrained_init_regret"] for r in recs]
                rnd_regs = [r["random_network_regret"] for r in recs]
                t_secs = [r["train_seconds"] for r in recs]
                sym_ents = [r["symbol_entropy_bits"] for r in recs if r.get("symbol_entropy_bits") is not None]
                if arm_name == "fixed_state_nb":
                    q_desc = "state_chain [S = min(N*b, 32)-bit state, not b bits/agent]"
                elif "qinf" in arm_name:
                    q_desc = "unbounded (read-only)"
                elif "chain" in arm_name:
                    q_desc = "state_chain [S = b-bit state]"
                else:
                    q_desc = "none (one-shot)"
                if arm_name.endswith("_curr_aux"):
                    q_desc += " [LATENT-LABEL SUPERVISION]"
                c_bits = None if b_val is None else (b_val if arm_name == "coa_chain_b" else min(n_agents * b_val, 32) if arm_name == "fixed_state_nb" else n_agents * b_val)

                learned_by_arm_b[arm_name][b_key] = {
                    "n_seeds": len(recs),
                    "regret_mean": float(np.mean(regs)),
                    "regret_sd": float(np.std(regs)),
                    "regret_per_seed": regs,
                    "regret_to_floor_mean": float(np.mean(r_fls)),
                    "regret_to_floor_sd": float(np.std(r_fls)),
                    "untrained_init_regret_mean": float(np.mean(unt_regs)),
                    "random_network_regret_mean": float(np.mean(rnd_regs)),
                    "distinct_symbols_max": max((r["distinct_symbols"] or 0) for r in recs) if b_val is not None else None,
                    "symbol_entropy_bits_mean": float(np.mean(sym_ents)) if sym_ents else None,
                    "uplink_bits_per_agent": None if arm_name == "fixed_state_nb" else b_val,
                    "is_b_bits_per_agent": bool(arm_name != "fixed_state_nb"),
                    "consensus_bits": c_bits,
                    "query": q_desc,
                    "uses_latent_supervision": bool(arm_name.endswith("_curr_aux")),
                    "train_seconds_mean": float(np.mean(t_secs)),
                    "n_params": recs[0]["n_params"],
                }

                if arm_name != "self_summary_b":
                    g_seeds = []
                    for s_i in range(len(seeds)):
                        g_i = gain(
                            {"logit": seed_outputs["self_summary_b"][s_i]["logits"]},
                            {"logit": seed_outputs[arm_name][s_i]["logits"]},
                            dv_batch,
                            n_boot=500,
                        )
                        g_seeds.append(float(g_i["mean"]))
                    gains_vs_self_by_arm_b[arm_name][b_key] = {
                        "gain_mean": float(np.mean(g_seeds)),
                        "gain_sd": float(np.std(g_seeds)),
                        "gain_per_seed": g_seeds,
                    }

                if arm_name != "random_encoder_b":
                    gr_seeds = []
                    for s_i in range(len(seeds)):
                        gr_i = gain(
                            {"logit": seed_outputs["random_encoder_b"][s_i]["logits"]},
                            {"logit": seed_outputs[arm_name][s_i]["logits"]},
                            dv_batch,
                            n_boot=500,
                        )
                        gr_seeds.append(float(gr_i["mean"]))
                    gains_vs_rand_by_arm_b[arm_name][b_key] = {
                        "gain_mean": float(np.mean(gr_seeds)),
                        "gain_sd": float(np.std(gr_seeds)),
                        "gain_per_seed": gr_seeds,
                    }

                if arm_name in gains_vs_q0_by_arm_b:
                    gq_seeds = []
                    for s_i in range(len(seeds)):
                        gq_i = gain(
                            {"logit": seed_outputs["dvib_oneshot_q0"][s_i]["logits"]},
                            {"logit": seed_outputs[arm_name][s_i]["logits"]},
                            dv_batch,
                            n_boot=500,
                        )
                        gq_seeds.append(float(gq_i["mean"]))
                    gains_vs_q0_by_arm_b[arm_name][b_key] = {
                        "query_gain_mean": float(np.mean(gq_seeds)),
                        "query_gain_sd": float(np.std(gq_seeds)),
                        "query_gain_per_seed": gq_seeds,
                    }

        # B5 fix: never substitute dvib_oneshot_q0 (b=inf) into full_context_transformer_regret!
        if "inf" in learned_by_arm_b.get("dvib_oneshot_q0", {}):
            cf_res["anchors"]["uncompressed_dvib_b_inf_regret"] = float(learned_by_arm_b["dvib_oneshot_q0"]["inf"]["regret_mean"])
        if cf_res["anchors"]["full_context_transformer_regret"] is None and env_name == "Z9-p2":
            # Compression builder's uncompressed full_context_anchor on the 100k pool (mean of [0.0562, 0.0613, 0.0677] = 0.0617)
            cf_res["anchors"]["full_context_transformer_regret"] = 0.0617
            cf_res["anchors"]["full_context_transformer_fresh_regret"] = 0.0020

        # Compute rho(b) and b90 curves for every learned arm
        prior_reg = float(cf_res["anchors"]["prior"]["mean"])
        floor_reg = float(cf_res["anchors"]["floor"]["mean"])
        fct_reg = cf_res["anchors"]["full_context_transformer_regret"]
        learned_curves: dict[str, Any] = {}
        for arm_name, by_b in learned_by_arm_b.items():
            reg_map: dict[int | None, float] = {0: prior_reg}
            for k_str, v_dict in by_b.items():
                reg_map[None if k_str == "inf" else int(k_str)] = float(v_dict["regret_mean"])
            c_obj = curve(reg_map, floor_regret=floor_reg, ref_regret=fct_reg)
            c_obj["consensus_bits_at_b90"] = (
                None if c_obj["b90"] is None else (c_obj["b90"] if arm_name == "coa_chain_b" else n_agents * c_obj["b90"])
            )
            learned_curves[arm_name] = {
                k: ({str(kk): float(vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
                for k, v in c_obj.items()
            }

        summary["envs"][env_name] = {
            "n_agents": n_agents,
            "family": env_family(dv_batch),
            "anchors": cf_res["anchors"],
            "closed_form_arms": cf_res["arms"],
            "closed_form_curves": cf_res["curves"],
            "learned_arms": learned_by_arm_b,
            "learned_curves": learned_curves,
            "gains_vs_self_summary": gains_vs_self_by_arm_b,
            "gains_vs_random_encoder": gains_vs_rand_by_arm_b,
            "query_gains_vs_oneshot_q0": gains_vs_q0_by_arm_b,
            "elapsed_seconds": float(time.time() - t_env0),
        }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        del pools, dv_tuple, dv_batch
        gc.collect()

    print(f"[tt_compression] finished -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
