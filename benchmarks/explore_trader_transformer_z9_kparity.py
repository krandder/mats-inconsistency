#!/usr/bin/env python3
"""Exploratory Sweep of Trader-Transformer on Z9 k-Bit Parity (k in {3, 4, 6} + Z9-hard).

Answers opus-mats #3867:
- Z6 is 2-bit parity (solved by 1 attention layer).
- Z9 (main b80a5f5) is k-bit parity (k=3, 4, 6) + distractors, where every individual reading AND every
  pair of readings (for k >= 3) is independent of Y.
- Evaluates Axes (A) attn_layers (all, every_other, first_last, last, first, none), (B) loss (final,
  every_prefix, distill with uses_oracle=True), and (C) token_state (kv, recurrent, feedback) against
  FullContextTransformer bars at 3000x512 fresh episodes (both w=64,L=3 and w=128,L=4).
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
from aggbench.envs import Batch, make_env
from aggbench.metrics import summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.sequential import exact_inputs, exact_sequential_bayes, forecast_kl, sample_orders
from aggbench.trader_transformer import resolve_attn_layers
from aggbench.transformer import FRESH_STATS_START, FreshStream, FullContext, FullContextTransformer, full_context
from aggbench.verify import oracle_check
from aggbench.zoo import none_floor_logit


def exact_prefix_bayes_batch(batch: Batch, order: np.ndarray) -> np.ndarray:
    """Computes exact sequential prefix Bayes logits (n, N) in float32."""
    p, inp = exact_sequential_bayes(batch), exact_inputs(batch)
    n, n_ag = order.shape
    state, out = p.init(n), np.zeros((n, n_ag), dtype=np.float32)
    for t in range(n_ag):
        i = order[:, t]
        state = p.update(state, p.encode(inp, i, p.view(state, None, t), t), t)
        out[:, t] = p.readout(state)["logit"]
    return out


_Z9_WORKER_SPEC: tuple[str, dict[str, Any]] | None = None


def _init_z9_worker(spec: tuple[str, dict[str, Any]]) -> None:
    global _Z9_WORKER_SPEC
    _Z9_WORKER_SPEC = spec


def _sample_and_featurize_z9_chunk(
    args: tuple[int, int, int, int]
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Worker: samples `cnt` Z9 episodes, applies arrival order `ord_tr`, computes float32 tokens & prefix Bayes."""
    pos, cnt, start, seed = args
    assert _Z9_WORKER_SPEC is not None
    env_name, overrides = _Z9_WORKER_SPEC
    env = make_env(env_name, **overrides)
    b = env.sample("train", cnt, start=start)
    ord_tr = sample_orders(b.seeds, b.obs.shape[1], seed)
    row_idx = np.arange(cnt)[:, None]

    fct_none = full_context(b, "none", "unaligned", own_model=True)
    fct_full = full_context(b, "full", "unaligned", own_model=True)
    tok_none = np.ascontiguousarray(fct_none.tokens[row_idx, ord_tr], dtype=np.float32)
    tok_full = np.ascontiguousarray(fct_full.tokens[row_idx, ord_tr], dtype=np.float32)

    pb = exact_prefix_bayes_batch(b, ord_tr)
    max_err = float(np.max(np.abs(pb[:, -1] - b.bayes["logit"])))
    truth_f32 = np.ascontiguousarray(b.truth, dtype=np.float32)
    return pos, cnt, truth_f32, tok_none, tok_full, pb, max_err


def sample_z9_seed_featurized(
    env_name: str,
    overrides: dict[str, Any],
    seed: int,
    n_train: int,
    n_tr: int,
    in_dim_none: int,
    in_dim_full: int,
) -> dict[str, Any]:
    """Streams chunks via pool.imap into pre-allocated float32 buffers (< 2.5 GB peak RAM)."""
    truth_arr = np.empty((n_train,), dtype=np.float32)
    tok_none_arr = np.empty((n_train, n_tr, in_dim_none), dtype=np.float32)
    tok_full_arr = np.empty((n_train, n_tr, in_dim_full), dtype=np.float32)
    pb_arr = np.empty((n_train, n_tr), dtype=np.float32)

    start_base = seed * n_train
    w = min(8, os.cpu_count() or 4)
    chunk = min(8192, max(512, n_train // max(1, w * 4)))
    jobs: list[tuple[int, int, int, int]] = []
    pos = 0
    while pos < n_train:
        c = min(chunk, n_train - pos)
        jobs.append((pos, c, start_base + pos, seed))
        pos += c

    max_err = 0.0
    if len(jobs) == 1:
        _init_z9_worker((env_name, overrides))
        p_off, cnt, tr_c, tn_c, tf_c, pb_c, err_c = _sample_and_featurize_z9_chunk(jobs[0])
        truth_arr[p_off : p_off + cnt] = tr_c
        tok_none_arr[p_off : p_off + cnt] = tn_c
        tok_full_arr[p_off : p_off + cnt] = tf_c
        pb_arr[p_off : p_off + cnt] = pb_c
        max_err = max(max_err, err_c)
    else:
        with mp.get_context("fork").Pool(w, initializer=_init_z9_worker, initargs=((env_name, overrides),)) as pool:
            for p_off, cnt, tr_c, tn_c, tf_c, pb_c, err_c in pool.imap(_sample_and_featurize_z9_chunk, jobs):
                truth_arr[p_off : p_off + cnt] = tr_c
                tok_none_arr[p_off : p_off + cnt] = tn_c
                tok_full_arr[p_off : p_off + cnt] = tf_c
                pb_arr[p_off : p_off + cnt] = pb_c
                max_err = max(max_err, err_c)

    return {
        "truth": truth_arr,
        "tok_none": tok_none_arr,
        "tok_full": tok_full_arr,
        "pb": pb_arr,
        "max_err": max_err,
    }


def resolve_schedule(spec: str, layers: int) -> tuple[int, ...]:
    if spec == "first_last":
        return tuple(sorted({0, layers - 1}))
    return resolve_attn_layers(spec, layers)


class Z9TraderNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int,
        layers: int,
        heads: int,
        attn_layers: tuple[int, ...],
        token_state: str = "kv",
        pos: str = "none",
        max_traders: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.attn_set = set(attn_layers)
        self.token_state = token_state
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_sd", torch.ones(in_dim))
        self.inp = nn.Linear(in_dim, width)
        self.pos = nn.Embedding(max_traders, width) if pos == "learned" else None
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
        self.head = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.inp((x - self.x_mean) / self.x_sd)
        if self.pos is not None:
            z = z + self.pos(torch.arange(z.shape[1], device=z.device))
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
            causal = torch.triu(torch.ones(n_tr, n_tr, dtype=torch.bool, device=z.device), 1)
            for l in range(self.layers):
                if l in self.attn_set:
                    a = self.ln1[l](z)
                    if self.token_state == "kv":
                        z = z + self.attn[str(l)](a, a, a, attn_mask=causal, need_weights=False)[0]
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
        return self.head(self.norm(z))[..., 0]


class Z9TraderAggregator:
    def __init__(
        self,
        seed: int,
        in_dim: int,
        *,
        width: int = 64,
        layers: int = 3,
        heads: int = 4,
        steps: int = 3000,
        batch: int = 512,
        lr: float = 2e-3,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        attn_layers: str = "all",
        token_state: str = "kv",
        loss: str = "final",
        pos: str = "none",
        reperm: bool = False,
        fresh: bool = True,
        drop_last_cols: int = 0,
        device: str = "cuda",
    ) -> None:
        torch.manual_seed(seed)
        self.seed = seed
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
        self.module = Z9TraderNet(
            in_dim=in_dim,
            width=width,
            layers=layers,
            heads=heads,
            attn_layers=self.attn_layers_resolved,
            token_state=token_state,
            pos=pos,
            dropout=dropout,
        )
        self.log: list[dict[str, Any]] = []

    def _slice_cols(self, arr: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if self.drop_last_cols > 0:
            return arr[:, :, : -self.drop_last_cols]
        return arr

    def _loss_fn(self, logits: torch.Tensor, y: torch.Tensor, pb: torch.Tensor | None) -> torch.Tensor:
        if self.loss == "final":
            return F.binary_cross_entropy_with_logits(logits[:, -1], y)
        if self.loss == "every_prefix":
            return F.binary_cross_entropy_with_logits(logits, y[:, None].expand_as(logits))
        assert pb is not None
        return F.binary_cross_entropy_with_logits(logits, torch.sigmoid(pb))

    def fit(self, ctx: dict[str, Any], truth: np.ndarray) -> "Z9TraderAggregator":
        dev = self.device
        self.module.to(dev)
        x_np = ctx["tokens"]
        st_np = self._slice_cols(ctx["stats_tokens"])
        stats_x = torch.as_tensor(st_np, dtype=torch.float32, device=dev)
        flat = stats_x.reshape(-1, stats_x.shape[-1])
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))
        pb_np = ctx.get("prefix_bayes")
        l0 = float(ctx["public"].get("prior_logit", 0.0))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps, pct_start=0.05)
        gen = np.random.default_rng(self.seed)
        self.module.train()
        ema = None
        n_tr = x_np.shape[1]
        for step in range(self.steps):
            if self.fresh and len(x_np) >= self.steps * self.batch:
                s = step * self.batch
                xb = torch.as_tensor(self._slice_cols(x_np[s : s + self.batch]), dtype=torch.float32, device=dev)
                yb = torch.as_tensor(truth[s : s + self.batch], dtype=torch.float32, device=dev)
                pbb = None if pb_np is None else torch.as_tensor(pb_np[s : s + self.batch], dtype=torch.float32, device=dev)
            else:
                idx = gen.integers(0, len(x_np), self.batch)
                xb = torch.as_tensor(self._slice_cols(x_np[idx]), dtype=torch.float32, device=dev)
                yb = torch.as_tensor(truth[idx], dtype=torch.float32, device=dev)
                pbb = None if pb_np is None else torch.as_tensor(pb_np[idx], dtype=torch.float32, device=dev)
            if self.reperm and self.loss != "distill":
                perm = torch.argsort(torch.rand(len(xb), n_tr, device=dev), dim=1)
                xb = torch.gather(xb, 1, perm.unsqueeze(-1).expand_as(xb))
            logits = l0 + self.module(xb)
            loss = self._loss_fn(logits, yb, pbb)
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

    def predict_prefix(self, ctx: dict[str, Any]) -> np.ndarray:
        dev = self.device
        self.module.to(dev).eval()
        x = torch.as_tensor(self._slice_cols(ctx["tokens"]), dtype=torch.float32)
        with torch.no_grad():
            out = torch.cat([self.module(x[s : s + 2048].to(dev)).double().cpu() for s in range(0, len(x), 2048)])
        self.module.cpu()
        return float(ctx["public"].get("prior_logit", 0.0)) + out.numpy()

    def predict(self, ctx: dict[str, Any]) -> dict[str, np.ndarray]:
        return {"logit": self.predict_prefix(ctx)[:, -1]}


class PrebuiltFCTStream(FreshStream):
    def __init__(
        self,
        tokens: np.ndarray,
        truth: np.ndarray,
        stats_tokens: np.ndarray,
        public: dict[str, Any],
        batch: int,
        steps: int,
        drop_last_cols: int = 0,
    ) -> None:
        self.tokens = tokens
        self.truth = truth
        self.drop_last_cols = drop_last_cols
        self.stats_tokens = stats_tokens[:, :, : -drop_last_cols] if drop_last_cols > 0 else stats_tokens
        self.public = dict(public)
        self.kind = "bernoulli"
        self.d = None
        self.batch = batch
        self.steps = steps

    def __iter__(self):
        for s in range(self.steps):
            i = s * self.batch
            xb = self.tokens[i : i + self.batch]
            if self.drop_last_cols > 0:
                xb = xb[:, :, : -self.drop_last_cols]
            yield xb, self.truth[i : i + self.batch]


def build_z9_specs() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        ("Z9-k3", "Z9", {"n_bits": 3}),
        ("Z9-k4", "Z9", {"n_bits": 4}),
        ("Z9-k6", "Z9", {"n_bits": 6}),
        ("Z9-hard-k4", "Z9-hard", {"n_bits": 4}),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=str, default="Z9-k3,Z9-k4,Z9-k6,Z9-hard-k4")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--fresh-episodes", action="store_true", default=True)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--commit", type=str, default="")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    req_envs = [e.strip() for e in args.envs.split(",") if e.strip()]
    all_specs = {tag: (ename, ov) for tag, ename, ov in build_z9_specs()}

    out_path = pathlib.Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n_train = args.steps * args.batch_size if args.fresh_episodes else args.n_train
    print(
        f"[z9_kparity] device={device} envs={req_envs} seeds={seeds} n_train={n_train} "
        f"steps={args.steps} batch={args.batch_size} width={args.width} layers={args.layers} fresh={args.fresh_episodes}",
        flush=True,
    )

    summary: dict[str, Any] = {
        "suite": "z9_kparity",
        "git_commit": args.commit,
        "fresh_episodes": bool(args.fresh_episodes),
        "n_train_per_seed": n_train,
        "n_eval": args.n_eval,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "layers": args.layers,
        "seeds": seeds,
        "envs": {},
    }

    for tag in req_envs:
        if tag not in all_specs:
            raise ValueError(f"Unknown Z9 spec tag: {tag}")
        ename, overrides = all_specs[tag]
        env = make_env(ename, **overrides)

        dv = env.sample("dev", args.n_eval)
        stats_b = env.sample("train", 4096, start=FRESH_STATS_START)
        tr_base = env.sample("train", 2000)
        base_preds, _ = all_baselines(tr_base, dv, "none")
        base_metrics = {b: summarize(p, dv, n_boot=500) for b, p in base_preds.items()}
        fl_logit = none_floor_logit(env, dv)
        base_metrics["floor_none"] = summarize({"logit": fl_logit}, dv, n_boot=500)

        ord_dv = sample_orders(dv.seeds, dv.obs.shape[1], 0)
        pb_dv = exact_prefix_bayes_batch(dv, ord_dv)
        pb_err_dv = float(np.max(np.abs(pb_dv[:, -1] - dv.bayes["logit"])))
        assert pb_err_dv < 1e-6, f"{tag} prefix Bayes mismatch on dev: {pb_err_dv}"
        print(f"  [{tag}] verified exact_prefix_bayes_logits on dev: max_abs_err={pb_err_dv:.3e}", flush=True)

        fct_dv_none = full_context(dv, "none", "unaligned", own_model=True)
        fct_dv_full = full_context(dv, "full", "unaligned", own_model=True)
        fct_stats_none = np.ascontiguousarray(full_context(stats_b, "none", "unaligned", own_model=True).tokens, dtype=np.float32)
        fct_stats_full = np.ascontiguousarray(full_context(stats_b, "full", "unaligned", own_model=True).tokens, dtype=np.float32)
        tok_dv_none = np.ascontiguousarray(fct_dv_none.tokens[np.arange(len(dv))[:, None], ord_dv], dtype=np.float32)
        tok_dv_full = np.ascontiguousarray(fct_dv_full.tokens[np.arange(len(dv))[:, None], ord_dv], dtype=np.float32)

        in_dim_none = int(fct_dv_none.tokens.shape[-1])
        in_dim_full = int(fct_dv_full.tokens.shape[-1])
        n_tr = int(dv.obs.shape[1])

        seed_records: dict[str, list[dict[str, Any]]] = {}
        seed0_methods: dict[str, Any] = dict(base_metrics)
        seed0_controls: dict[str, Any] = {}
        seed0_arms_meta: dict[str, Any] = {}
        learned_arms: list[str] = []

        for s_idx, seed in enumerate(seeds):
            gc.collect()
            t_s0 = time.time()
            start_idx = seed * n_train
            seed_data = sample_z9_seed_featurized(
                ename, overrides, seed, n_train, n_tr=n_tr, in_dim_none=in_dim_none, in_dim_full=in_dim_full
            )
            pb_err_tr = seed_data["max_err"]
            assert pb_err_tr < 1e-6, f"{tag} prefix Bayes mismatch on train: {pb_err_tr}"
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
            print(
                f"  [{tag} seed={seed}] streamed & featurized n_train={n_train} (start={start_idx}) + exact prefix Bayes "
                f"(max_err={pb_err_tr:.3e}) in {time.time() - t_s0:.2f}s (maxRSS={rss_gb:.2f}GB)",
                flush=True,
            )

            truth_tr = seed_data["truth"]
            tok_tr_none = seed_data["tok_none"]
            tok_tr_full = seed_data["tok_full"]
            pb_tr = seed_data["pb"]

            # 1. FCT bars: headline w64_l3 and matched w128_l4 (plus full and no_report)
            fct_bars = [
                ("bar_fct_fresh_headline_w64_l3", tok_tr_none, fct_stats_none, fct_dv_none, 64, 3, 0),
                ("full_context_transformer_none", tok_tr_none, fct_stats_none, fct_dv_none, args.width, args.layers, 0),
                ("full_context_transformer_full", tok_tr_full, fct_stats_full, fct_dv_full, args.width, args.layers, 0),
                (
                    "bar_fct_no_report",
                    tok_tr_none,
                    fct_stats_none,
                    FullContext("bernoulli", fct_dv_none.tokens[:, :, :-1], fct_dv_none.public),
                    args.width,
                    args.layers,
                    1,
                ),
            ]
            for bname, tr_toks, st_toks, dv_ctx, bw, bl, drop_c in fct_bars:
                t0 = time.time()
                eff_in_dim = int(tr_toks.shape[-1] - drop_c)
                stream = PrebuiltFCTStream(tr_toks, truth_tr, st_toks, dv.public, args.batch_size, args.steps, drop_last_cols=drop_c)
                made_fct: list[FullContextTransformer] = []

                def make_fct(s: int, _in_dim: int = eff_in_dim, _w: int = bw, _l: int = bl) -> FullContextTransformer:
                    m = FullContextTransformer(
                        s,
                        _in_dim,
                        "bernoulli",
                        width=_w,
                        layers=_l,
                        heads=4,
                        steps=args.steps,
                        batch=args.batch_size,
                        lr=1e-3,
                        weight_decay=1e-2,
                        dropout=0.1 if args.fresh_episodes else 0.0,
                        device=device,
                    )
                    made_fct.append(m)
                    return m

                arms_out = run_with_controls(
                    make_fct,
                    stream if args.fresh_episodes else FullContext("bernoulli", tr_toks[:, :, :-drop_c] if drop_c else tr_toks, dv.public),
                    truth_tr,
                    dv_ctx,
                    seed=seed,
                )
                dt = time.time() - t0
                m_sum = summarize(arms_out["trained"], dv, n_boot=500)
                reg_val = float(m_sum["regret"]["mean"])
                kl_val = float(m_sum["kl"]["mean"])
                print(f"  [{tag} seed={seed}] {bname}: regret={reg_val:.4f} kl={kl_val:.4f} ({dt:.2f}s)", flush=True)
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
                        "in_dim": eff_in_dim,
                        "n_params": int(sum(p.numel() for p in made_fct[0].module.parameters())),
                        "train_seconds": round(dt, 2),
                        "uses_oracle": False,
                    }

            # 2. Trader-Transformer Arms across Axes (A), (B), (C)
            tt_arms = [
                ("A_attn_all_w64_l3", tok_tr_none, fct_stats_none, tok_dv_none, 64, 3, "all", "kv", "final", "none", False, 0),
                ("A_attn_last_w64_l3", tok_tr_none, fct_stats_none, tok_dv_none, 64, 3, "last", "kv", "final", "none", False, 0),
                ("A_attn_all", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "final", "none", False, 0),
                ("A_attn_all_reperm", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "final", "none", True, 0),
                ("A_attn_all_no_report", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "final", "none", False, 1),
                ("A_attn_first_last", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "first_last", "kv", "final", "none", False, 0),
                ("A_attn_every_other", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "every_other", "kv", "final", "none", False, 0),
                ("A_attn_last", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "last", "kv", "final", "none", False, 0),
                ("A_attn_first", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "first", "kv", "final", "none", False, 0),
                ("A_attn_none", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "none", "kv", "final", "none", False, 0),
                ("B_loss_every_prefix", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "every_prefix", "none", False, 0),
                ("B_loss_every_prefix_reperm", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "every_prefix", "none", True, 0),
                ("B_loss_every_prefix_pos", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "every_prefix", "learned", False, 0),
                ("B_loss_distill", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "kv", "distill", "none", False, 0),
                ("C_state_recurrent_final", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "recurrent", "final", "none", False, 0),
                ("C_state_recurrent_prefix", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "recurrent", "every_prefix", "none", False, 0),
                ("C_state_feedback_final", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "feedback", "final", "none", False, 0),
                ("C_state_feedback_prefix", tok_tr_none, fct_stats_none, tok_dv_none, args.width, args.layers, "all", "feedback", "every_prefix", "none", False, 0),
                ("V_vis_full", tok_tr_full, fct_stats_full, tok_dv_full, args.width, args.layers, "all", "kv", "final", "none", False, 0),
            ]

            for aname, tr_toks, st_toks, dv_toks, aw, al, a_sched, t_state, l_mode, pos_mode, rep_flag, drop_c in tt_arms:
                t0 = time.time()
                eff_in_dim = int(tr_toks.shape[-1] - drop_c)
                ctx_tr = {
                    "tokens": tr_toks,
                    "stats_tokens": st_toks,
                    "public": dict(dv.public),
                    "prefix_bayes": pb_tr if l_mode == "distill" else None,
                }
                ctx_dv = {
                    "tokens": dv_toks,
                    "stats_tokens": st_toks,
                    "public": dict(dv.public),
                    "prefix_bayes": None,
                }
                made_tt: list[Z9TraderAggregator] = []

                def make_tt(
                    s: int,
                    _in_dim: int = eff_in_dim,
                    _w: int = aw,
                    _l: int = al,
                    _as: str = a_sched,
                    _ts: str = t_state,
                    _lm: str = l_mode,
                    _pm: str = pos_mode,
                    _rf: bool = rep_flag,
                    _dc: int = drop_c,
                ) -> Z9TraderAggregator:
                    m = Z9TraderAggregator(
                        s,
                        _in_dim,
                        width=_w,
                        layers=_l,
                        heads=4,
                        steps=args.steps,
                        batch=args.batch_size,
                        lr=1e-3 if args.fresh_episodes else 2e-3,
                        weight_decay=1e-2 if args.fresh_episodes else 1e-4,
                        dropout=0.05 if args.fresh_episodes else 0.0,
                        attn_layers=_as,
                        token_state=_ts,
                        loss=_lm,
                        pos=_pm,
                        reperm=_rf,
                        fresh=bool(args.fresh_episodes),
                        drop_last_cols=_dc,
                        device=device,
                    )
                    made_tt.append(m)
                    return m

                arms_out = run_with_controls(make_tt, ctx_tr, truth_tr, ctx_dv, seed=seed)
                dt = time.time() - t0
                m_sum = summarize(arms_out["trained"], dv, n_boot=500)
                reg_val = float(m_sum["regret"]["mean"])
                kl_val = float(m_sum["kl"]["mean"])
                pred_pref = made_tt[0].predict_prefix(ctx_dv)
                pref_kls = [
                    float(forecast_kl({"logit": pb_dv[:, t]}, {"logit": pred_pref[:, t]}, "bernoulli").mean())
                    for t in range(pred_pref.shape[1])
                ]
                mean_pk = float(np.mean(pref_kls))
                print(
                    f"  [{tag} seed={seed}] {aname}: regret={reg_val:.4f} kl={kl_val:.4f} mean_pref_kl={mean_pk:.4f} ({dt:.2f}s)",
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
                        "n_params": int(sum(p.numel() for p in made_tt[0].module.parameters())),
                        "train_seconds": round(dt, 2),
                        "attn_layers_resolved": list(made_tt[0].attn_layers_resolved),
                        "uses_oracle": bool(made_tt[0].uses_oracle),
                        "prefix_kl": pref_kls,
                        "mean_prefix_kl": mean_pk,
                    }

            del seed_data, truth_tr, tok_tr_none, tok_tr_full, pb_tr
            gc.collect()

        # Write per-env result.json and update summary
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
                "attn_layers_resolved": meta.get("attn_layers_resolved"),
                "uses_oracle": meta.get("uses_oracle", False),
            }

        summary["envs"][tag] = {
            "env_name": ename,
            "overrides": overrides,
            "n_agents": int(dv.obs.shape[1]),
            "prefix_bayes_final_maxabs_err": pb_err_dv,
            "baselines": {
                b: {
                    "regret": float(base_metrics[b]["regret"]["mean"]),
                    "kl": float(base_metrics[b]["kl"]["mean"]),
                }
                for b in base_metrics
            },
            "models": models_summary,
        }

        env_res_dir = out_dir / f"tt_z9_{tag.lower()}"
        env_res_dir.mkdir(parents=True, exist_ok=True)
        res_obj = make_result(
            run_id=f"tt_z9_{tag.lower()}",
            track="explore",
            env=env,
            visibility="none",
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
                "prefix_bayes_final_maxabs_err": pb_err_dv,
                "arms_meta": seed0_arms_meta,
                "multi_seed_summary": models_summary,
            },
        )
        if args.commit:
            res_obj["harness"]["git_commit"] = args.commit
            res_obj["harness"]["git_dirty"] = False
        write_result(env_res_dir / "result.json", res_obj, allow_dirty=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    print(f"[z9_kparity] finished -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
