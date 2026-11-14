#!/usr/bin/env python3
"""Spot-L4 Exploratory Sweep of Kelvin's Trader-Transformer Architecture (Axes A, B, C, and D Grid).

Includes all 7 Must-Fixes from gate-gemini-tt-2026-10-21.md (#3859):
- Must-Fix 2: Unconfounded FullContextTransformer bars (`bar_fct_deepsets_bidir`, `bar_fct_piece_bidir`,
  `bar_fct_no_report`, and `bar_fct_fresh_headline_w64_l3` at 3000x512).
- Must-Fix 4: Exact Z8 prefix Bayes formula from 7379c5d (`sum / (n_k + tau^2)`) + mandatory assertion
  `max |prefix_bayes[:, -1] - batch.bayes["logit"]| < 1e-8` on every environment before training.
- Must-Fix 5: `--commit` CLI flag recorded in `harness["git_commit"]`, per-seed resampled train pool
  (`start = seed * n_train`), and distinct `permute_traders_train` handling in `--fresh-episodes` mode.
- Must-Fix 6: Clean 2x2 factorial separating rank-free mask (`r' <= r` vs `t' <= t and r' <= r`) from
  readout (`cell_T1` vs `pooled` over present cells) in `--suite grid` and `--suite gate_fixes`.
- Must-Fix 7: Full `3000 x 512` fresh-episode support for XOR-type environments (`Z5`, `Z6`) and `E2, Z8`.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import time
import types
from typing import Any

import numpy as np
import torch

from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.envs import VISIBILITY, Batch, make_env
from aggbench.metrics import paired_difference, summarize
from aggbench.results import harness_info, make_result, split_record, write_result
from aggbench.sequential import forecast_kl, sample_orders
from aggbench.verify import oracle_check
from aggbench.zoo import PrivateReadConfig, none_floor_logit
from aggbench.trader_transformer import (
    DEFAULTS as TT_DEFAULTS,
    TraderContext,
    TraderTransformer,
    grid_masks,
    prefix_bayes_logits,
)
from aggbench.transformer import FreshStream, FullContext, FullContextTransformer, full_context


_WORKER_ENV = None


def _init_sample_worker(env_name: str) -> None:
    global _WORKER_ENV
    _WORKER_ENV = make_env(env_name)


def _sample_chunk_worker(args: tuple[str, int, int]) -> Batch:
    split, cnt, start = args
    assert _WORKER_ENV is not None
    return _WORKER_ENV.sample(split, cnt, start=start)


def sample_batch_parallel(env, split: str, n: int, start: int = 0, workers: int = 0) -> Batch:
    """Samples n episodes from `split` starting at `start`, parallelized across CPU workers when n > 4096."""
    if n <= 4096:
        return env.sample(split, n, start=start)
    w = workers or min(8, os.cpu_count() or 4)
    chunk = max(2048, min(16384, (n + w * 4 - 1) // (w * 4)))
    jobs: list[tuple[str, int, int]] = []
    pos = 0
    while pos < n:
        c = min(chunk, n - pos)
        jobs.append((split, c, start + pos))
        pos += c
    with mp.get_context("fork").Pool(w, initializer=_init_sample_worker, initargs=(env.name,)) as pool:
        batches = pool.map(_sample_chunk_worker, jobs)
    return Batch(
        env.name,
        asdict(env.config),
        env.kind,
        np.concatenate([b.seeds for b in batches], 0),
        np.concatenate([b.truth for b in batches], 0),
        np.concatenate([b.obs for b in batches], 0),
        {k: np.concatenate([b.local[k] for b in batches], 0) for k in batches[0].local},
        {k: np.concatenate([b.reports[k] for b in batches], 0) for k in batches[0].reports},
        {v: np.concatenate([b.keys[v] for b in batches], 0) for v in VISIBILITY},
        {k: np.concatenate([b.bayes[k] for b in batches], 0) for k in batches[0].bayes},
        {
            k: (
                np.concatenate([b.oracle[k] for b in batches], 0)
                if np.asarray(batches[0].oracle[k]).ndim > 0
                else batches[0].oracle[k]
            )
            for k in batches[0].oracle
        },
        env.public(),
    )


def _slice_batch(batch: Batch, s: int, e: int) -> Batch:
    return Batch(
        batch.env,
        dict(batch.config),
        batch.kind,
        batch.seeds[s:e],
        batch.truth[s:e],
        batch.obs[s:e],
        {k: v[s:e] for k, v in batch.local.items()},
        {k: v[s:e] for k, v in batch.reports.items()},
        {k: v[s:e] for k, v in batch.keys.items()},
        {k: v[s:e] for k, v in batch.bayes.items()},
        {k: (v[s:e] if np.asarray(v).ndim > 0 else v) for k, v in batch.oracle.items()},
        dict(batch.public),
    )


_PB_BATCH: Batch | None = None
_PB_ORDER: np.ndarray | None = None


def _pb_chunk_worker(se: tuple[int, int]) -> np.ndarray:
    s, e = se
    assert _PB_BATCH is not None and _PB_ORDER is not None
    return prefix_bayes_logits(_slice_batch(_PB_BATCH, s, e), _PB_ORDER[s:e])


def exact_prefix_bayes_logits(batch: Batch, order: np.ndarray) -> tuple[np.ndarray, float]:
    """Computes exact prefix Bayes log-odds (n, N) including the 7379c5d Z8 fix, asserting final == bayes."""
    global _PB_BATCH, _PB_ORDER
    if batch.env == "Z8":
        tau2 = float(PrivateReadConfig(**batch.config).read_noise) ** 2
        cov = batch.oracle["cov"].astype(np.float64)
        reads = np.nan_to_num(batch.obs).astype(np.float64)
        mu = batch.oracle["mu"].astype(np.float64)
        l0 = float(batch.public.get("prior_logit", 0.0))
        idx = np.arange(len(batch))[:, None]
        cov_ord = cov[idx, order]
        r_ord = reads[idx, order] * cov_ord
        cnt = np.cumsum(cov_ord, axis=1)
        sr = np.cumsum(r_ord, axis=1)
        pb = l0 + np.where(cnt > 0, 2.0 * mu[:, None, :] * sr / (cnt + tau2), 0.0).sum(axis=-1)
    elif len(batch) <= 8192:
        pb = prefix_bayes_logits(batch, order)
    else:
        w = min(8, os.cpu_count() or 4)
        _PB_BATCH = batch
        _PB_ORDER = order
        ranges = [(s, min(len(batch), s + 8192)) for s in range(0, len(batch), 8192)]
        try:
            with mp.get_context("fork").Pool(w) as pool:
                parts = pool.map(_pb_chunk_worker, ranges)
        finally:
            _PB_BATCH = None
            _PB_ORDER = None
        pb = np.concatenate(parts, axis=0)
    max_err = float(np.max(np.abs(pb[:, -1] - batch.bayes["logit"])))
    assert max_err < 1e-8, f"prefix_bayes_logits final step mismatch on {batch.env}: max_err={max_err}"
    return pb, max_err


class PrecomputedFreshStream(FreshStream):
    """Wraps a pre-sampled (steps * batch) fresh episode context for 1-pass zero-reuse _fit_stream."""

    def __init__(self, ctx, truth: np.ndarray, batch: int, steps: int, seed: int = 0):
        self.kind = ctx.kind
        self.public = dict(ctx.public)
        self.batch = batch
        self.steps = steps
        self.tokens = ctx.tokens
        self.truth = np.asarray(truth, np.float32)
        self.stats_tokens = ctx.tokens[: min(4096, len(ctx.tokens))]
        self.d = self.public.get("d")
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1337)
        perm = rng.permutation(len(self.tokens))
        for s in range(self.steps):
            idx = perm[s * self.batch : (s + 1) * self.batch]
            yield self.tokens[idx], self.truth[idx]


def _make_matched_fct_net(
    in_dim: int,
    out_dim: int,
    width: int,
    layers: int,
    heads: int,
    dropout: float = 0.1,
    encoder_mode: str = "linear",
) -> torch.nn.Module:
    nn = torch.nn

    class MatchedFCTNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_mode = encoder_mode
            self.register_buffer("x_mean", torch.zeros(in_dim))
            self.register_buffer("x_sd", torch.ones(in_dim))
            if encoder_mode == "deepsets":
                self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width))
                self.rho = nn.Sequential(nn.GELU(), nn.Linear(width, width))
            else:
                self.inp = nn.Linear(in_dim, width)
                self.rho = None
            self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
            layer = nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=4 * width,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.out = nn.Linear(width, out_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.encoder_mode == "deepsets":
                z = self.inp((x - self.x_mean) / self.x_sd)
                assert self.rho is not None
                z = self.rho((z * x[..., :1]).sum(2))
            elif self.encoder_mode == "piece_bidir":
                b, n_tr, n_p, f_dim = x.shape
                x_flat = x.reshape(b, n_tr * n_p, f_dim)
                z = self.inp((x_flat - self.x_mean) / self.x_sd)
            else:
                z = self.inp((x - self.x_mean) / self.x_sd)
            z = torch.cat([self.cls.expand(len(z), 1, -1), z], 1)
            return self.out(self.norm(self.enc(z)[:, 0]))

    return MatchedFCTNet()


class MatchedFullContextTransformer(FullContextTransformer):
    """FullContextTransformer supporting matched encoders (`linear`, `no_report`, `deepsets`, `piece_bidir`)."""

    def __init__(
        self,
        seed: int,
        in_dim: int,
        kind: str,
        *,
        encoder_mode: str = "linear",
        d: int | None = None,
        width: int = 64,
        layers: int = 3,
        heads: int = 4,
        steps: int = 3000,
        batch: int = 512,
        lr: float = 1e-3,
        threads: int | None = None,
        device: str = "cpu",
        val_frac: float = 0.1,
        eval_every: int = 100,
        weight_decay: float = 1e-2,
        dropout: float = 0.1,
        patience: int | None = 800,
    ) -> None:
        super().__init__(
            seed,
            in_dim,
            kind,
            d=d,
            width=width,
            layers=layers,
            heads=heads,
            steps=steps,
            batch=batch,
            lr=lr,
            threads=threads,
            device=device,
            val_frac=val_frac,
            eval_every=eval_every,
            weight_decay=weight_decay,
            dropout=dropout,
            patience=patience,
        )
        self.encoder_mode = encoder_mode
        out_dim = 1 if kind == "bernoulli" else (d + d * (d + 1) // 2)  # type: ignore[operator]
        torch.manual_seed(seed)
        self.module = _make_matched_fct_net(in_dim, out_dim, width, layers, heads, dropout, encoder_mode)

    def predict(self, ctx) -> dict[str, np.ndarray]:
        dev = getattr(self, "device", "cpu")
        self.module.to(dev).eval()
        outs = []
        with torch.no_grad():
            x = torch.as_tensor(ctx.tokens, dtype=torch.float32)
            for s in range(0, len(x), 2048):
                outs.append(self.module(x[s : s + 2048].to(dev)).double().cpu())
        self.module.cpu()
        h = torch.cat(outs)
        if self.kind == "bernoulli":
            return {"logit": float(ctx.public.get("prior_logit", 0.0)) + h[:, 0].numpy()}
        from aggbench.transformer import _gauss_from_head
        mean, L = _gauss_from_head(torch, h, self.d)
        return {"mean": mean.numpy(), "prec": (L @ L.transpose(1, 2)).numpy()}


class CudaTraderTransformer(TraderTransformer):
    """GPU-accelerated TraderTransformer wrapper with clean separation of `rankfree_grid` and `grid_readout`."""

    def __init__(
        self,
        seed: int,
        in_dim: int,
        *,
        device: str | None = None,
        rankfree_grid: bool = False,
        grid_readout: str = "cell_T1",
        val_frac: float = 0.1,
        eval_every: int = 100,
        weight_decay: float = 1e-2,
        patience: int | None = 800,
        permute_traders_train: bool = False,
        fresh_episodes: bool = False,
        **kwargs: Any,
    ):
        super().__init__(seed, in_dim, device=device, **kwargs)
        if grid_readout not in ("cell_T1", "pooled"):
            raise ValueError("grid_readout must be 'cell_T1' or 'pooled'")
        self.val_frac = 0.0 if fresh_episodes else val_frac
        self.eval_every = eval_every
        self.weight_decay = weight_decay
        self.patience = None if fresh_episodes else patience
        self.permute_traders_train = permute_traders_train
        self.fresh_episodes = fresh_episodes
        self.best_step: int | None = None
        self.rankfree_grid = rankfree_grid
        self.grid_readout = grid_readout
        if self.layout != "trader":
            width = kwargs.get("width", TT_DEFAULTS["width"])
            layers = kwargs.get("layers", TT_DEFAULTS["layers"])
            heads = kwargs.get("heads", TT_DEFAULTS["heads"])
            token_heads = kwargs.get("token_heads", TT_DEFAULTS["token_heads"])
            layout = self.layout
            A = set(self.attn_layers)

            def _grid_cells(mod_self, x: torch.Tensor) -> torch.Tensor:
                n, N, T, _ = x.shape
                z = mod_self.embed(x).reshape(n, N * T, width)
                if rankfree_grid:
                    r, _ = np.divmod(np.arange(N * T), T)
                    m_rf = torch.as_tensor(~(r[None, :] <= r[:, None]), device=z.device)
                    masks = [m_rf] * layers
                else:
                    masks = [
                        torch.as_tensor(m, device=z.device)
                        for m in grid_masks(N, T, layout, heads, token_heads, layers)
                    ]
                for l in range(layers):
                    if l in A:
                        a = mod_self.ln1[l](z)
                        m = masks[l] if masks[l].ndim == 2 else masks[l].repeat(n, 1, 1)
                        z = z + mod_self.attn[str(l)](a, a, a, attn_mask=m, need_weights=False)[0]
                    z = z + mod_self.mlp[l](mod_self.ln2[l](z))
                cell_logits = mod_self.head(mod_self.norm(z))[..., 0].reshape(n, N, T)
                if grid_readout == "pooled":
                    present = x[..., 0]
                    denom = present.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    pooled = (cell_logits * present).sum(dim=-1, keepdim=True) / denom
                    return 0.5 * cell_logits + 0.5 * pooled.expand_as(cell_logits)
                return cell_logits

            self.module.cells = types.MethodType(_grid_cells, self.module)

    def fit(self, ctx: TraderContext, truth: np.ndarray) -> "CudaTraderTransformer":
        import copy
        self._check(ctx)
        dev = torch.device(self.device)
        self.module.to(dev)
        gen = np.random.default_rng(self.seed)
        perm = gen.permutation(len(ctx.tokens))
        n_val = int(round(self.val_frac * len(perm))) if self.val_frac else 0
        val_idx, fit_idx = perm[:n_val], perm[n_val:]

        # Keep training tokens in CPU pinned/pageable memory if > 200k episodes to avoid GPU VRAM pressure,
        # or on GPU when <= 200k episodes.
        keep_on_gpu = len(fit_idx) <= 250_000
        tok_dev = dev if keep_on_gpu else torch.device("cpu")
        x = torch.as_tensor(ctx.tokens[fit_idx], dtype=torch.float32, device=tok_dev)
        y = torch.as_tensor(np.asarray(truth)[fit_idx], dtype=torch.float32, device=tok_dev)
        pb = (
            None
            if (ctx.prefix_bayes is None or self.loss != "distill")
            else torch.as_tensor(ctx.prefix_bayes[fit_idx], dtype=torch.float32, device=tok_dev)
        )

        xv = torch.as_tensor(ctx.tokens[val_idx], dtype=torch.float32, device=dev) if n_val else None
        yv = torch.as_tensor(np.asarray(truth)[val_idx], dtype=torch.float32, device=dev) if n_val else None
        pbv = (
            torch.as_tensor(ctx.prefix_bayes[val_idx], dtype=torch.float32, device=dev)
            if (n_val and ctx.prefix_bayes is not None and self.loss == "distill")
            else None
        )

        stats_slice = x[: min(8192, len(x))].to(dev)
        flat = stats_slice.reshape(-1, stats_slice.shape[-1])
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))

        l0 = float(ctx.public.get("prior_logit", 0.0))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps, pct_start=0.05)
        idx_gen = torch.Generator(device=tok_dev).manual_seed(int(self.seed) + 1337)
        gpu_gen = torch.Generator(device=dev).manual_seed(int(self.seed) + 2774)

        best, best_state, best_step = float("inf"), None, 0
        ema = None
        n_traders = x.shape[1]
        stream_perm = torch.randperm(len(x), generator=idx_gen, device=tok_dev) if self.fresh_episodes else None
        for step in range(self.steps):
            self.module.train()
            if stream_perm is not None and (step + 1) * self.batch <= len(x):
                idx = stream_perm[step * self.batch : (step + 1) * self.batch]
            else:
                idx = torch.randint(0, len(x), (self.batch,), generator=idx_gen, device=tok_dev)
            xb = x[idx].to(dev, non_blocking=True)
            yb = y[idx].to(dev, non_blocking=True)
            pbb = None if pb is None else pb[idx].to(dev, non_blocking=True)
            if self.permute_traders_train and self.loss != "distill":
                t_ord = torch.randperm(n_traders, generator=gpu_gen, device=dev)
                xb = xb[:, t_ord]
            loss = self.loss_fn(l0 + self.module(xb), yb, pbb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if n_val and ((step + 1) % self.eval_every == 0 or step + 1 == self.steps):
                self.module.eval()
                with torch.no_grad():
                    assert xv is not None and yv is not None
                    vl = float(self.loss_fn(l0 + self.module(xv), yv, pbv).item())
                if vl < best:
                    best, best_step = vl, step + 1
                    best_state = copy.deepcopy(self.module.state_dict())
            if (step + 1) % 500 == 0 or step + 1 == self.steps:
                self.log.append({"step": step + 1, "loss_ema": ema, "best_val": best, "best_step": best_step})
            if n_val and self.patience and step + 1 - best_step >= self.patience:
                self.log.append({"step": step + 1, "stopped_early": True, "best_val": best, "best_step": best_step})
                break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.best_step = best_step
        self.module.cpu()
        return self

    def predict_prefix(self, ctx: TraderContext) -> np.ndarray:
        self._check(ctx)
        dev = torch.device(self.device)
        self.module.to(dev)
        self.module.eval()
        with torch.no_grad():
            x = torch.as_tensor(ctx.tokens, dtype=torch.float32)
            out = torch.cat([self.module(x[s : s + 2048].to(dev)).double().cpu() for s in range(0, len(x), 2048)])
        self.module.cpu()
        return float(ctx.public.get("prior_logit", 0.0)) + out.numpy()


def build_trader_context_ext(
    batch: Batch,
    visibility: str = "none",
    alignment: str = "unaligned",
    order_seed: int = 0,
    with_prefix_bayes: bool = False,
    layout: str = "trader",
    own_model: bool = True,
    trader_encoder: str = "linear",
    precomputed_pb: np.ndarray | None = None,
) -> TraderContext:
    """Builds TraderContext, supporting trader_encoder='deepsets', Z8 3*K keys, and exact Z8 prefix Bayes."""
    if layout == "trader" and trader_encoder == "linear":
        tok = full_context(batch, visibility, alignment, own_model=own_model).tokens
    else:
        obs, keys, _, own = batch.agent_inputs(visibility, alignment=alignment, own_model=True)
        n, N, K = obs.shape
        feat = [obs[..., None]]
        if own_model:
            feat.append(own.reshape(n, N, -1, K).transpose(0, 1, 3, 2))
        if keys.shape[-1]:
            feat.append(keys.reshape(n, N, -1, K).transpose(0, 1, 3, 2))
        elif not own_model:
            feat.append(np.zeros((n, N, K, 2), np.float32))
        feat_arr = np.concatenate(feat, -1)
        present = np.isfinite(obs)
        cells = np.zeros((n, N, K, feat_arr.shape[-1] + 1), np.float32)
        rank = np.cumsum(present, -1) - 1
        e, i, k = np.nonzero(present)
        cells[e, i, rank[e, i, k], 0] = 1.0
        cells[e, i, rank[e, i, k], 1:] = feat_arr[e, i, k]
        tok = cells
    n, N = tok.shape[:2]
    order = sample_orders(batch.seeds, N, order_seed)
    tokens = tok[np.arange(n)[:, None], order]
    if with_prefix_bayes:
        pb = precomputed_pb if precomputed_pb is not None else exact_prefix_bayes_logits(batch, order)[0]
    else:
        pb = None
    return TraderContext(batch.kind, tokens, order, dict(batch.public), pb, layout)


def run_fct_bar_arm(
    bar_name: str,
    ctr_view,
    cdv_view,
    tr_batch: Batch,
    dv_batch: Batch,
    seed: int,
    n_boot: int,
    device: str,
    *,
    encoder_mode: str = "linear",
    width: int = 128,
    layers: int = 4,
    heads: int = 4,
    steps: int = 2000,
    batch_size: int = 512,
    lr: float = 1e-3,
    fresh_stream: FreshStream | None = None,
    fresh_episodes: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    t0 = time.time()
    in_dim = int(cdv_view.tokens.shape[-1])
    if fresh_stream is not None:
        tr_input: Any = fresh_stream
        tr_truth = None
        is_fresh = True
    elif fresh_episodes:
        tr_input = PrecomputedFreshStream(ctr_view, tr_batch.truth, batch_size, steps, seed=seed)
        tr_truth = None
        is_fresh = True
    else:
        tr_input = ctr_view
        tr_truth = tr_batch.truth
        is_fresh = False

    made: list[MatchedFullContextTransformer] = []

    def make_bar(s: int) -> MatchedFullContextTransformer:
        m = MatchedFullContextTransformer(
            s,
            in_dim,
            tr_batch.kind,
            encoder_mode=encoder_mode,
            d=tr_batch.public.get("d"),
            width=width,
            layers=layers,
            heads=heads,
            steps=steps,
            batch=batch_size,
            lr=lr,
            weight_decay=1e-2,
            dropout=0.1,
            patience=None if is_fresh else 800,
            val_frac=0.0 if is_fresh else 0.1,
            device=device,
        )
        made.append(m)
        return m

    arms = run_with_controls(make_bar, tr_input, tr_truth, cdv_view, seed=seed)
    elapsed = time.time() - t0
    trained_model = made[0]
    n_params = sum(p.numel() for p in trained_model.module.parameters())
    forecasts = {
        bar_name: arms["trained"],
        f"{bar_name}:untrained_init": arms["control:untrained_init"],
        f"{bar_name}:random_network": arms["control:random_network"],
    }
    controls_map = {
        bar_name: {
            "untrained_init": f"{bar_name}:untrained_init",
            "random_network": f"{bar_name}:random_network",
        }
    }
    meta = {
        "in_dim": in_dim,
        "n_params": int(n_params),
        "train_seconds": round(elapsed, 2),
        "attn_layers_resolved": list(range(layers)),
        "uses_oracle": False,
        "encoder_mode": encoder_mode,
        "prefix_kl": None,
        "mean_prefix_kl": None,
        "final_prefix_kl": None,
        "log": trained_model.log,
    }
    return forecasts, controls_map, meta


def run_single_arm(
    arm_name: str,
    ctr: TraderContext,
    cdv: TraderContext,
    tr_batch: Batch,
    dv_batch: Batch,
    pb_eval: np.ndarray | None,
    seed: int,
    n_boot: int,
    device: str,
    hp: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Runs one TraderTransformer configuration through run_with_controls and computes prefix KL curve."""
    t0 = time.time()
    made: list[CudaTraderTransformer] = []

    def make(s: int) -> CudaTraderTransformer:
        m = CudaTraderTransformer(s, ctr.tokens.shape[-1], threads=None, device=device, **hp)
        made.append(m)
        return m

    arms = run_with_controls(make, ctr, tr_batch.truth, cdv, seed=seed)
    elapsed = time.time() - t0

    trained_model = made[0]
    n_params = sum(p.numel() for p in trained_model.module.parameters())
    prefix_kl = None
    if pb_eval is not None:
        pred_pref = trained_model.predict_prefix(cdv)
        prefix_kl = [
            float(forecast_kl({"logit": pb_eval[:, t]}, {"logit": pred_pref[:, t]}, "bernoulli").mean())
            for t in range(pred_pref.shape[1])
        ]

    forecasts = {
        arm_name: arms["trained"],
        f"{arm_name}:untrained_init": arms["control:untrained_init"],
        f"{arm_name}:random_network": arms["control:random_network"],
    }
    controls_map = {
        arm_name: {
            "untrained_init": f"{arm_name}:untrained_init",
            "random_network": f"{arm_name}:random_network",
        }
    }
    meta = {
        "in_dim": int(ctr.tokens.shape[-1]),
        "n_params": int(n_params),
        "train_seconds": round(elapsed, 2),
        "attn_layers_resolved": list(trained_model.attn_layers),
        "uses_oracle": bool(trained_model.uses_oracle),
        "rankfree_grid": bool(trained_model.rankfree_grid),
        "grid_readout": str(trained_model.grid_readout),
        "prefix_kl": prefix_kl,
        "mean_prefix_kl": float(np.mean(prefix_kl)) if prefix_kl is not None else None,
        "final_prefix_kl": float(prefix_kl[-1]) if prefix_kl is not None else None,
        "log": trained_model.log,
    }
    return forecasts, controls_map, meta


def run_suite(args: argparse.Namespace, git_commit_sha: str) -> dict[str, Any]:
    envs = [s.strip() for s in args.envs.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[{args.suite}] device={device} envs={envs} seeds={seeds} "
        f"n_train={args.n_train} n_eval={args.n_eval} steps={args.steps} batch={args.batch_size} "
        f"width={args.width} layers={args.layers} fresh={args.fresh_episodes}",
        flush=True,
    )

    results_by_env: dict[str, Any] = {}
    for env_name in envs:
        env = make_env(env_name)
        oracle_chk = oracle_check(env)
        assert oracle_chk["verified"], f"Bayes oracle failed on {env_name}: {oracle_chk}"

        dv_batch = env.sample("dev", args.n_eval)
        # Mandatory Must-Fix 4 assertion on dev batch before any training
        dv_order_chk = sample_orders(dv_batch.seeds, dv_batch.obs.shape[1], 0)
        _, dv_pb_max_err = exact_prefix_bayes_logits(dv_batch, dv_order_chk)
        print(f"  [{env_name}] verified exact_prefix_bayes_logits on dev: max_abs_err={dv_pb_max_err:.3e}", flush=True)

        seed_records: dict[str, Any] = {}
        for seed in seeds:
            t_sample0 = time.time()
            # Must-Fix 5: Per-seed distinct training episodes (`start = seed * args.n_train`)
            tr_start = seed * args.n_train
            tr_batch = sample_batch_parallel(env, "train", args.n_train, start=tr_start)
            tr_order_seed = sample_orders(tr_batch.seeds, tr_batch.obs.shape[1], seed)
            tr_pb_exact, tr_pb_max_err = exact_prefix_bayes_logits(tr_batch, tr_order_seed)
            dv_order_seed = sample_orders(dv_batch.seeds, dv_batch.obs.shape[1], seed)
            dv_pb_exact, _ = exact_prefix_bayes_logits(dv_batch, dv_order_seed)
            print(
                f"  [{env_name} seed={seed}] sampled n_train={args.n_train} (start={tr_start}) + exact prefix Bayes "
                f"(max_err={tr_pb_max_err:.3e}) in {time.time() - t_sample0:.2f}s",
                flush=True,
            )

            methods_summary: dict[str, Any] = {}
            all_forecasts: dict[str, Any] = {}
            learned_arms: list[str] = []
            controls_map: dict[str, Any] = {}
            arms_meta: dict[str, Any] = {}

            # 1. Baselines on dev batch + floor_none (eb3cd7d)
            tr_base_slice = _slice_batch(tr_batch, 0, min(20000, len(tr_batch)))
            base_forecasts, _ = all_baselines(tr_base_slice, dv_batch, "full")
            for b_name, b_f in base_forecasts.items():
                all_forecasts[b_name] = b_f
                methods_summary[b_name] = summarize(b_f, dv_batch, n_boot=args.n_boot, seed=seed)
            nf_logit = none_floor_logit(env, dv_batch)
            if nf_logit is not None:
                nf_f = {"logit": nf_logit}
                all_forecasts["floor_none"] = nf_f
                methods_summary["floor_none"] = summarize(nf_f, dv_batch, n_boot=args.n_boot, seed=seed)

            # 2. Reference Bars (Must-Fix 2):
            # 2a. Standard FullContextTransformer (none & full)
            ctr_fct_none = full_context(tr_batch, "none", "unaligned")
            cdv_fct_none = full_context(dv_batch, "none", "unaligned")
            ctr_fct_full = full_context(tr_batch, "full", "unaligned")
            cdv_fct_full = full_context(dv_batch, "full", "unaligned")

            bar_runs: list[tuple[str, Any, Any, dict[str, Any]]] = [
                (
                    "full_context_transformer_none",
                    ctr_fct_none,
                    cdv_fct_none,
                    dict(
                        encoder_mode="linear",
                        width=args.width,
                        layers=args.layers,
                        heads=args.heads,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        fresh_episodes=args.fresh_episodes,
                    ),
                ),
                (
                    "full_context_transformer_full",
                    ctr_fct_full,
                    cdv_fct_full,
                    dict(
                        encoder_mode="linear",
                        width=args.width,
                        layers=args.layers,
                        heads=args.heads,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        fresh_episodes=args.fresh_episodes,
                    ),
                ),
                # 2b. Unconfounded bar without scalar report_logit / 4 (`bar_fct_no_report`)
                (
                    "bar_fct_no_report",
                    FullContext(ctr_fct_none.kind, ctr_fct_none.tokens[..., :-1], dict(ctr_fct_none.public)),
                    FullContext(cdv_fct_none.kind, cdv_fct_none.tokens[..., :-1], dict(cdv_fct_none.public)),
                    dict(
                        encoder_mode="no_report",
                        width=args.width,
                        layers=args.layers,
                        heads=args.heads,
                        steps=args.steps,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        fresh_episodes=args.fresh_episodes,
                    ),
                ),
                # 2c. Main 0d94ef1 fresh headline bar (`width=64, layers=3, steps=3000, batch=512` FreshStream)
                (
                    "bar_fct_fresh_headline_w64_l3",
                    ctr_fct_none,
                    cdv_fct_none,
                    dict(
                        encoder_mode="linear",
                        width=64,
                        layers=3,
                        heads=4,
                        steps=3000,
                        batch_size=512,
                        lr=1e-3,
                        fresh_stream=(
                            PrecomputedFreshStream(ctr_fct_none, tr_batch.truth, 512, 3000, seed=seed)
                            if len(ctr_fct_none.tokens) >= 3000 * 512
                            else FreshStream(env, "none", "unaligned", 512, 3000, start=seed * 1_536_000, samplers=4)
                        ),
                    ),
                ),
            ]

            # 2d. Matched-encoder bars for piece-structured envs (`bar_fct_deepsets_bidir` & `bar_fct_piece_bidir`)
            ctr_ds_shared: TraderContext | None = None
            cdv_ds_shared: TraderContext | None = None
            if "cov" in tr_batch.oracle:
                ctr_ds_shared = build_trader_context_ext(
                    tr_batch,
                    "none",
                    "unaligned",
                    order_seed=seed,
                    with_prefix_bayes=True,
                    layout="trader",
                    trader_encoder="deepsets",
                    precomputed_pb=tr_pb_exact,
                )
                cdv_ds_shared = build_trader_context_ext(
                    dv_batch,
                    "none",
                    "unaligned",
                    order_seed=seed,
                    with_prefix_bayes=False,
                    layout="trader",
                    trader_encoder="deepsets",
                )
                bar_runs.extend([
                    (
                        "bar_fct_deepsets_bidir",
                        FullContext(ctr_ds_shared.kind, ctr_ds_shared.tokens, dict(ctr_ds_shared.public)),
                        FullContext(cdv_ds_shared.kind, cdv_ds_shared.tokens, dict(cdv_ds_shared.public)),
                        dict(
                            encoder_mode="deepsets",
                            width=args.width,
                            layers=args.layers,
                            heads=args.heads,
                            steps=args.steps,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            fresh_episodes=args.fresh_episodes,
                        ),
                    ),
                    (
                        "bar_fct_piece_bidir",
                        FullContext(ctr_ds_shared.kind, ctr_ds_shared.tokens, dict(ctr_ds_shared.public)),
                        FullContext(cdv_ds_shared.kind, cdv_ds_shared.tokens, dict(cdv_ds_shared.public)),
                        dict(
                            encoder_mode="piece_bidir",
                            width=args.grid_width,
                            layers=args.layers,
                            heads=args.heads,
                            steps=args.steps,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            fresh_episodes=args.fresh_episodes,
                        ),
                    ),
                ])

            for b_name, b_ctr, b_cdv, b_kw in bar_runs:
                f_dict, c_map, meta = run_fct_bar_arm(
                    b_name, b_ctr, b_cdv, tr_batch, dv_batch, seed, args.n_boot, device, **b_kw
                )
                for k_m, f_v in f_dict.items():
                    all_forecasts[k_m] = f_v
                    methods_summary[k_m] = summarize(f_v, dv_batch, n_boot=args.n_boot, seed=seed)
                learned_arms.append(b_name)
                controls_map.update(c_map)
                arms_meta[b_name] = meta
                print(
                    f"  [{env_name} seed={seed}] {b_name}: regret={methods_summary[b_name]['regret']['mean']:.4f} "
                    f"kl={methods_summary[b_name]['kl']['mean']:.4f} ({meta['train_seconds']}s)",
                    flush=True,
                )

            # 3. Precompute TraderContexts needed for this suite
            base_hp = dict(
                width=args.width,
                layers=args.layers,
                heads=args.heads,
                steps=args.steps,
                batch=args.batch_size,
                lr=args.lr,
                fresh_episodes=args.fresh_episodes,
            )

            if args.suite in ("abc", "gate_fixes"):
                ctx_cache: dict[tuple[str, bool, bool], tuple[TraderContext, TraderContext, np.ndarray | None]] = {}
                vis_list = [("none", True, True), ("full", True, False)]
                if args.suite == "abc":
                    vis_list.extend([
                        ("noisy", True, False),
                        ("none", False, False),
                        ("noisy", False, False),
                    ])
                for vis, own_m, need_pb in vis_list:
                    ctr_c = build_trader_context_ext(
                        tr_batch,
                        vis,
                        "unaligned",
                        order_seed=seed,
                        with_prefix_bayes=need_pb,
                        layout="trader",
                        own_model=own_m,
                        precomputed_pb=tr_pb_exact if need_pb else None,
                    )
                    cdv_c = build_trader_context_ext(
                        dv_batch,
                        vis,
                        "unaligned",
                        order_seed=seed,
                        with_prefix_bayes=False,
                        layout="trader",
                        own_model=own_m,
                    )
                    ctx_cache[(vis, own_m, need_pb)] = (ctr_c, cdv_c, dv_pb_exact)

                fl_tuple = (0, args.layers - 1) if args.layers > 1 else (0,)
                arm_specs = [
                    ("A_attn_all", ("none", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="kv")),
                    ("A_attn_all_reperm", ("none", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="kv", permute_traders_train=True)),
                    ("A_attn_first_last", ("none", True, True), dict(**base_hp, attn_layers=fl_tuple, loss="final", token_state="kv")),
                    ("A_attn_every_other", ("none", True, True), dict(**base_hp, attn_layers="every_other", loss="final", token_state="kv")),
                    ("A_attn_last", ("none", True, True), dict(**base_hp, attn_layers="last", loss="final", token_state="kv")),
                    ("A_attn_first", ("none", True, True), dict(**base_hp, attn_layers="first", loss="final", token_state="kv")),
                    ("A_attn_none", ("none", True, True), dict(**base_hp, attn_layers="none", loss="final", token_state="kv")),
                    ("B_loss_every_prefix", ("none", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv")),
                    ("B_loss_every_prefix_reperm", ("none", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv", permute_traders_train=True)),
                    ("B_loss_distill", ("none", True, True), dict(**base_hp, attn_layers="all", loss="distill", token_state="kv")),
                    ("B_loss_every_prefix_pos", ("none", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv", pos="learned")),
                    ("C_state_recurrent_final", ("none", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="recurrent")),
                    ("C_state_recurrent_prefix", ("none", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="recurrent")),
                    ("C_state_feedback_final", ("none", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="feedback")),
                    ("C_state_feedback_prefix", ("none", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="feedback")),
                    ("V_vis_full", ("full", True, False), dict(**base_hp, attn_layers="all", loss="final", token_state="kv")),
                ]
                if not args.fresh_episodes:
                    arm_specs.insert(
                        2,
                        ("A_attn_all_no_es", ("none", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="kv", val_frac=0.0, patience=None, weight_decay=1e-4)),
                    )
                if args.suite == "abc":
                    arm_specs.extend([
                        ("V_vis_noisy", ("noisy", True, False), dict(**base_hp, attn_layers="all", loss="final", token_state="kv")),
                        ("V_vis_none_no_own", ("none", False, False), dict(**base_hp, attn_layers="all", loss="final", token_state="kv")),
                        ("V_vis_noisy_no_own", ("noisy", False, False), dict(**base_hp, attn_layers="all", loss="final", token_state="kv")),
                    ])

                if "cov" in tr_batch.oracle and ctr_ds_shared is not None and cdv_ds_shared is not None:
                    ctx_cache[("none_ds", True, True)] = (ctr_ds_shared, cdv_ds_shared, dv_pb_exact)
                    arm_specs.extend([
                        ("A_attn_all_deepsets", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="kv", trader_encoder="deepsets")),
                        ("A_attn_all_deepsets_reperm", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="kv", trader_encoder="deepsets", permute_traders_train=True)),
                        ("A_attn_last_deepsets", ("none_ds", True, True), dict(**base_hp, attn_layers="last", loss="final", token_state="kv", trader_encoder="deepsets")),
                        ("B_loss_every_prefix_deepsets", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv", trader_encoder="deepsets")),
                        ("B_loss_every_prefix_deepsets_reperm", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv", trader_encoder="deepsets", permute_traders_train=True)),
                        ("B_loss_distill_deepsets", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="distill", token_state="kv", trader_encoder="deepsets")),
                        ("C_state_recurrent_deepsets", ("none_ds", True, True), dict(**base_hp, attn_layers="all", loss="final", token_state="recurrent", trader_encoder="deepsets")),
                    ])
                    ctr_ds_full = build_trader_context_ext(
                        tr_batch, "full", "unaligned", order_seed=seed, with_prefix_bayes=False, layout="trader", trader_encoder="deepsets"
                    )
                    cdv_ds_full = build_trader_context_ext(
                        dv_batch, "full", "unaligned", order_seed=seed, with_prefix_bayes=False, layout="trader", trader_encoder="deepsets"
                    )
                    ctx_cache[("full_ds", True, False)] = (ctr_ds_full, cdv_ds_full, dv_pb_exact)
                    arm_specs.extend([
                        ("V_vis_full_deepsets", ("full_ds", True, False), dict(**base_hp, attn_layers="all", loss="final", token_state="kv", trader_encoder="deepsets")),
                        ("V_vis_full_deepsets_prefix", ("full_ds", True, False), dict(**base_hp, attn_layers="all", loss="every_prefix", token_state="kv", trader_encoder="deepsets")),
                    ])

                for arm_name, ctx_key, hp in arm_specs:
                    ctr_c, cdv_c, pb_ev = ctx_cache[ctx_key]
                    f_dict, c_map, meta = run_single_arm(
                        arm_name, ctr_c, cdv_c, tr_batch, dv_batch, pb_ev, seed, args.n_boot, device, hp
                    )
                    for k_m, f_v in f_dict.items():
                        all_forecasts[k_m] = f_v
                        methods_summary[k_m] = summarize(f_v, dv_batch, n_boot=args.n_boot, seed=seed)
                    learned_arms.append(arm_name)
                    controls_map.update(c_map)
                    arms_meta[arm_name] = meta
                    print(
                        f"  [{env_name} seed={seed}] {arm_name}: regret={methods_summary[arm_name]['regret']['mean']:.4f} "
                        f"kl={methods_summary[arm_name]['kl']['mean']:.4f} mean_pref_kl={meta['mean_prefix_kl']:.4f} "
                        f"({meta['train_seconds']}s)",
                        flush=True,
                    )

            if args.suite == "grid" or (args.suite == "gate_fixes" and "cov" in tr_batch.oracle):
                grid_hp = dict(
                    width=args.grid_width,
                    layers=args.layers,
                    heads=args.heads,
                    steps=args.steps,
                    batch=args.batch_size,
                    lr=args.lr,
                    fresh_episodes=args.fresh_episodes,
                )
                ctx_trader = build_trader_context_ext(
                    tr_batch,
                    "none",
                    "unaligned",
                    order_seed=seed,
                    with_prefix_bayes=True,
                    layout="trader",
                    own_model=True,
                    precomputed_pb=tr_pb_exact,
                )
                cdv_trader = build_trader_context_ext(
                    dv_batch, "none", "unaligned", order_seed=seed, with_prefix_bayes=False, layout="trader", own_model=True
                )

                grid_ctx_cache: dict[tuple[str, str, bool], tuple[TraderContext, TraderContext, np.ndarray | None]] = {}
                for lay in ("grid", "axial", "grid_full"):
                    for vis, own_m in [("none", True), ("noisy", False)]:
                        if vis == "none" and own_m and ctr_ds_shared is not None and cdv_ds_shared is not None:
                            ctr_g = TraderContext(ctr_ds_shared.kind, ctr_ds_shared.tokens, ctr_ds_shared.order, dict(ctr_ds_shared.public), None, lay)
                            cdv_g = TraderContext(cdv_ds_shared.kind, cdv_ds_shared.tokens, cdv_ds_shared.order, dict(cdv_ds_shared.public), None, lay)
                        else:
                            ctr_g = build_trader_context_ext(
                                tr_batch, vis, "unaligned", order_seed=seed, with_prefix_bayes=False, layout=lay, own_model=own_m
                            )
                            cdv_g = build_trader_context_ext(
                                dv_batch, vis, "unaligned", order_seed=seed, with_prefix_bayes=False, layout=lay, own_model=own_m
                            )
                        grid_ctx_cache[(lay, vis, own_m)] = (ctr_g, cdv_g, dv_pb_exact)

                assert ctr_ds_shared is not None and cdv_ds_shared is not None
                grid_specs = [
                    # Must-Fix 6: Clean 2x2 factorial (mask: grid_full cone vs grid_rankfree) x (readout: cell_T1 vs pooled)
                    ("D_layout_grid_full_cellT1_final", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", rankfree_grid=False, grid_readout="cell_T1")),
                    ("D_layout_grid_full_pooled_final", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", rankfree_grid=False, grid_readout="pooled")),
                    ("D_layout_grid_rankfree_cellT1_final", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", rankfree_grid=True, grid_readout="cell_T1")),
                    ("D_layout_grid_rankfree_pooled_final", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", rankfree_grid=True, grid_readout="pooled")),
                    ("D_layout_grid_full_cellT1_prefix", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="every_prefix", token_state="kv", rankfree_grid=False, grid_readout="cell_T1")),
                    ("D_layout_grid_full_pooled_prefix", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="every_prefix", token_state="kv", rankfree_grid=False, grid_readout="pooled")),
                    ("D_layout_grid_rankfree_cellT1_prefix", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="every_prefix", token_state="kv", rankfree_grid=True, grid_readout="cell_T1")),
                    ("D_layout_grid_rankfree_pooled_prefix", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="every_prefix", token_state="kv", rankfree_grid=True, grid_readout="pooled")),
                    # Per-head axial ("grid") and alternating-layer ("axial") with both readouts
                    ("D_layout_grid_cellT1_final", grid_ctx_cache[("grid", "none", True)], dict(**grid_hp, layout="grid", attn_layers="all", loss="final", token_state="kv", grid_readout="cell_T1")),
                    ("D_layout_grid_pooled_final", grid_ctx_cache[("grid", "none", True)], dict(**grid_hp, layout="grid", attn_layers="all", loss="final", token_state="kv", grid_readout="pooled")),
                    ("D_layout_grid_cellT1_prefix", grid_ctx_cache[("grid", "none", True)], dict(**grid_hp, layout="grid", attn_layers="all", loss="every_prefix", token_state="kv", grid_readout="cell_T1")),
                    ("D_layout_axial_cellT1_final", grid_ctx_cache[("axial", "none", True)], dict(**grid_hp, layout="axial", attn_layers="all", loss="final", token_state="kv", grid_readout="cell_T1")),
                    ("D_layout_axial_pooled_final", grid_ctx_cache[("axial", "none", True)], dict(**grid_hp, layout="axial", attn_layers="all", loss="final", token_state="kv", grid_readout="pooled")),
                    ("D_layout_grid_full_pos", grid_ctx_cache[("grid_full", "none", True)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", pos="learned", grid_readout="cell_T1")),
                ]
                if args.suite == "grid":
                    grid_specs = [
                        ("D_layout_trader_final", (ctx_trader, cdv_trader, dv_pb_exact), dict(**base_hp, layout="trader", attn_layers="all", loss="final", token_state="kv")),
                        ("D_layout_trader_prefix", (ctx_trader, cdv_trader, dv_pb_exact), dict(**base_hp, layout="trader", attn_layers="all", loss="every_prefix", token_state="kv")),
                        ("D_layout_trader_deepsets_final", (ctr_ds_shared, cdv_ds_shared, dv_pb_exact), dict(**base_hp, layout="trader", attn_layers="all", loss="final", token_state="kv", trader_encoder="deepsets")),
                        ("D_layout_trader_deepsets_prefix", (ctr_ds_shared, cdv_ds_shared, dv_pb_exact), dict(**base_hp, layout="trader", attn_layers="all", loss="every_prefix", token_state="kv", trader_encoder="deepsets")),
                    ] + grid_specs + [
                        ("D_layout_grid_full_noisy_no_own", grid_ctx_cache[("grid_full", "noisy", False)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", grid_readout="cell_T1")),
                        ("D_layout_grid_rankfree_noisy_no_own", grid_ctx_cache[("grid_full", "noisy", False)], dict(**grid_hp, layout="grid_full", attn_layers="all", loss="final", token_state="kv", rankfree_grid=True, grid_readout="pooled")),
                    ]

                for arm_name, (ctr_g, cdv_g, pb_ev_g), hp in grid_specs:
                    f_dict, c_map, meta = run_single_arm(
                        arm_name, ctr_g, cdv_g, tr_batch, dv_batch, pb_ev_g, seed, args.n_boot, device, hp
                    )
                    for k_m, f_v in f_dict.items():
                        all_forecasts[k_m] = f_v
                        methods_summary[k_m] = summarize(f_v, dv_batch, n_boot=args.n_boot, seed=seed)
                    learned_arms.append(arm_name)
                    controls_map.update(c_map)
                    arms_meta[arm_name] = meta
                    print(
                        f"  [{env_name} seed={seed}] {arm_name}: regret={methods_summary[arm_name]['regret']['mean']:.4f} "
                        f"kl={methods_summary[arm_name]['kl']['mean']:.4f} mean_pref_kl={meta['mean_prefix_kl']:.4f} "
                        f"({meta['train_seconds']}s)",
                        flush=True,
                    )

            # Compute paired differences against all bars and oracle_clearing
            paired_vs_fct: dict[str, Any] = {}
            paired_vs_fct_fresh: dict[str, Any] = {}
            paired_vs_fct_ds: dict[str, Any] = {}
            paired_vs_fct_piece: dict[str, Any] = {}
            paired_vs_oc: dict[str, Any] = {}
            for arm_name in learned_arms:
                paired_vs_fct[arm_name] = paired_difference(
                    all_forecasts[arm_name],
                    all_forecasts["full_context_transformer_none"],
                    dv_batch,
                    metric="regret",
                    n_boot=args.n_boot,
                    seed=seed,
                )
                if "bar_fct_fresh_headline_w64_l3" in all_forecasts:
                    paired_vs_fct_fresh[arm_name] = paired_difference(
                        all_forecasts[arm_name],
                        all_forecasts["bar_fct_fresh_headline_w64_l3"],
                        dv_batch,
                        metric="regret",
                        n_boot=args.n_boot,
                        seed=seed,
                    )
                if "bar_fct_deepsets_bidir" in all_forecasts:
                    paired_vs_fct_ds[arm_name] = paired_difference(
                        all_forecasts[arm_name],
                        all_forecasts["bar_fct_deepsets_bidir"],
                        dv_batch,
                        metric="regret",
                        n_boot=args.n_boot,
                        seed=seed,
                    )
                if "bar_fct_piece_bidir" in all_forecasts:
                    paired_vs_fct_piece[arm_name] = paired_difference(
                        all_forecasts[arm_name],
                        all_forecasts["bar_fct_piece_bidir"],
                        dv_batch,
                        metric="regret",
                        n_boot=args.n_boot,
                        seed=seed,
                    )
                if "oracle_clearing" in all_forecasts:
                    paired_vs_oc[arm_name] = paired_difference(
                        all_forecasts[arm_name],
                        all_forecasts["oracle_clearing"],
                        dv_batch,
                        metric="regret",
                        n_boot=args.n_boot,
                        seed=seed,
                    )

            if seed == seeds[0]:
                res_obj = make_result(
                    run_id=f"tt-{args.suite}-{env_name.lower()}-20261021",
                    track="EXPLORATORY",
                    env=env,
                    visibility="none",
                    splits={"train": split_record(tr_batch, "train"), "eval": split_record(dv_batch, "dev")},
                    oracle_check=oracle_chk,
                    methods=methods_summary,
                    learned_arms=learned_arms,
                    controls=controls_map,
                    smoke=False,
                    extra={
                        "alignment": "unaligned",
                        "prefix_bayes_final_maxabs_err": float(max(tr_pb_max_err, dv_pb_max_err)),
                        "arms_meta": arms_meta,
                        "paired_vs_fct_none": paired_vs_fct,
                        "paired_vs_fct_fresh_headline": paired_vs_fct_fresh,
                        "paired_vs_fct_deepsets_bidir": paired_vs_fct_ds,
                        "paired_vs_fct_piece_bidir": paired_vs_fct_piece,
                        "paired_vs_oracle_clearing": paired_vs_oc,
                    },
                )
                if git_commit_sha and res_obj.get("harness", {}).get("git_commit") == "unknown":
                    res_obj["harness"]["git_commit"] = git_commit_sha
                out_dir = pathlib.Path(args.out).parent / f"tt_{args.suite}_{env_name.lower()}"
                write_result(out_dir, res_obj)

            seed_records[str(seed)] = {
                "train_start": tr_start,
                "prefix_bayes_final_maxabs_err": float(max(tr_pb_max_err, dv_pb_max_err)),
                "methods": methods_summary,
                "arms_meta": arms_meta,
                "paired_vs_fct_none": paired_vs_fct,
                "paired_vs_fct_fresh_headline": paired_vs_fct_fresh,
                "paired_vs_fct_deepsets_bidir": paired_vs_fct_ds,
                "paired_vs_fct_piece_bidir": paired_vs_fct_piece,
                "paired_vs_oracle_clearing": paired_vs_oc,
            }

        # Aggregate 3-seed summary
        agg_methods: dict[str, Any] = {}
        s0 = seed_records[str(seeds[0])]["methods"]
        for m_name in s0:
            regs = [seed_records[str(s)]["methods"][m_name]["regret"]["mean"] for s in seeds]
            kls = [seed_records[str(s)]["methods"][m_name]["kl"]["mean"] for s in seeds]
            briers = [seed_records[str(s)]["methods"][m_name]["brier"]["mean"] for s in seeds]
            agg_methods[m_name] = {
                "regret_mean_3seed": float(np.mean(regs)),
                "regret_sd_3seed": float(np.std(regs, ddof=1)) if len(seeds) > 1 else 0.0,
                "regret_per_seed": [float(r) for r in regs],
                "kl_mean_3seed": float(np.mean(kls)),
                "kl_sd_3seed": float(np.std(kls, ddof=1)) if len(seeds) > 1 else 0.0,
                "brier_mean_3seed": float(np.mean(briers)),
            }

        agg_prefix: dict[str, Any] = {}
        s0_meta = seed_records[str(seeds[0])]["arms_meta"]
        for arm_name in s0_meta:
            p_kls = [
                seed_records[str(s)]["arms_meta"][arm_name]["prefix_kl"]
                for s in seeds
                if seed_records[str(s)]["arms_meta"][arm_name]["prefix_kl"] is not None
            ]
            mean_pk = np.mean(p_kls, axis=0).tolist() if p_kls else None
            agg_prefix[arm_name] = {
                "prefix_kl_3seed": mean_pk,
                "mean_prefix_kl_3seed": float(np.mean(mean_pk)) if mean_pk else None,
                "final_prefix_kl_3seed": float(mean_pk[-1]) if mean_pk else None,
                "in_dim": s0_meta[arm_name]["in_dim"],
                "n_params": s0_meta[arm_name]["n_params"],
                "attn_layers_resolved": s0_meta[arm_name]["attn_layers_resolved"],
                "uses_oracle": s0_meta[arm_name]["uses_oracle"],
                "rankfree_grid": s0_meta[arm_name].get("rankfree_grid"),
                "grid_readout": s0_meta[arm_name].get("grid_readout"),
                "encoder_mode": s0_meta[arm_name].get("encoder_mode"),
            }

        results_by_env[env_name] = {
            "oracle_check": oracle_chk,
            "prefix_bayes_final_maxabs_err": float(
                max(seed_records[str(s)]["prefix_bayes_final_maxabs_err"] for s in seeds)
            ),
            "agg_methods_3seed": agg_methods,
            "agg_prefix_3seed": agg_prefix,
            "per_seed": seed_records,
        }
        ckpt_path = pathlib.Path(args.out)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_path.write_text(
            json.dumps({"partial": True, "git_commit": git_commit_sha, "results_by_env": results_by_env}, indent=2),
            encoding="utf-8",
        )

    return results_by_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Trader-Transformer (A)/(B)/(C) and (D) Grid on CUDA.")
    parser.add_argument("--suite", choices=["abc", "grid", "gate_fixes"], required=True)
    parser.add_argument("--envs", type=str, default="E2,Z8,Z5,Z6")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--grid-width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fresh-episodes", action="store_true", help="1-pass no-reuse fresh episodes per step (#3856)")
    parser.add_argument("--commit", type=str, default="", help="Explicit 40-char git commit SHA for provenance")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    if args.fresh_episodes and args.n_train < args.steps * args.batch_size:
        args.n_train = args.steps * args.batch_size

    t_start = time.time()
    runner_path = pathlib.Path(__file__).resolve()
    runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    h_info = harness_info()
    git_commit_sha = args.commit or h_info.get("git_commit", "unknown")
    if git_commit_sha and h_info.get("git_commit") == "unknown":
        h_info["git_commit"] = git_commit_sha

    gate_disclosure = {
        "a_what_a_token_is": (
            "In layout='trader' (linear): one token per trader i in {0..N-1} produced by aggbench.transformer.full_context "
            "([obs_i (NaN->0), obs_present_mask_i, own_sensor_model_i, provenance_key_i at visibility, report_logit_i / 4]). "
            "In trader_encoder='deepsets' and layout in {'grid', 'axial', 'grid_full'}: piece cells [present, obs_k, own_cov_k, "
            "own_mu_k, (key_cov_k, key_mu_k)] with no scalar report."
        ),
        "b_what_a_layer_is": (
            "Pre-norm Transformer layer in TraderNet: LayerNorm -> causal Cross-Trader Self-Attention (or GRUCell in "
            "token_state='recurrent', or Feedback memory attention in token_state='feedback', active on attn_layers) -> "
            "residual + LayerNorm -> 2-layer GELU MLP (width -> 4*width -> width) -> residual."
        ),
        "c_what_crosses_between_traders_and_width": (
            f"Full d_model = {args.width} (layout='trader') or {args.grid_width} (layout='grid*') hidden state vectors "
            "cross between traders via causal self-attention K/V states, GRU recurrent state s_t, or Feedback memory."
        ),
        "d_regret_bars_and_alignment": (
            "Evaluated on alignment='unaligned' dev splits (n_eval=2000, test untouched) against exact Bayes (0.0000), "
            "oracle_clearing, extremized_log_pool, FullContextTransformer (linear, no_report, deepsets_bidir, piece_bidir, "
            "and 3000x512 fresh_headline_w64_l3) across 3 seeds with per-seed resampled train pools and controls."
        ),
    }

    suite_res = run_suite(args, git_commit_sha)
    payload = {
        "track": "EXPLORATORY",
        "suite": args.suite,
        "alignment": "unaligned",
        "test_split_touched": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "harness": h_info,
        "runner": {
            "script": "benchmarks/explore_trader_transformer_abc_grid.py",
            "sha256": runner_sha256,
            "git_commit": git_commit_sha,
            "args": vars(args),
            "wall_seconds": round(time.time() - t_start, 2),
        },
        "gate_a_to_d": gate_disclosure,
        "results_by_env": suite_res,
    }

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] wrote {out_path} in {payload['runner']['wall_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
