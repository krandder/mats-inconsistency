"""L6 (EXPLORATORY): can softmax attention do Bayesian aggregation, or must the weights be additive?

Every learned arm outputs a weighted combination of the agents' reports with per-agent weights a_i >= 0
(E2: logit = l0 + sum_i a_i (l_i - l0); E1: precision I + sum_i a_i (P_i - I), information sum_i a_i h_i).
The arms differ only in how the weights a_i are produced:

  softmax        2 pre-LN softmax self-attention blocks, then a softmax readout: a = softmax(q.k_i)   sum a = 1
  softmax_T      as softmax, times one learned scalar T (global total weight)                         sum a = T
  softmax_gain   as softmax, times a per-agent learned gain g_i = softplus(MLP(h_i)) (free value path)
  sigmoid        sigmoid attention in the blocks and the readout: a_i = sigmoid(q.k_i + b)            sum a in (0, N)
  linear         unnormalized linear attention (phi = elu + 1) in blocks and readout                  sum a in (0, inf)
  clearing_learned  no blocks; phi_i = MLP(token_i); a = (Phi Phi^T + eps I)^-1 diag(Phi Phi^T)
                    (the clearing solve with a learned kernel; E1 clamps a >= 0 so precision stays PD)
  clearing_exact    not learned: the clearing solve with the kernel read from the full keys (by construction)

Tokens (all learned arms): report features, full key, log(N)/log(32) (public n_agents). One model per
(env, arm, seed) is trained on a mixture of 15 cells (5 agent counts x 3 overlap levels) on the train split
with the truth as label (no distillation) and evaluated per cell on the dev split. Test split never used.

Run: python l6_attention.py --out DIR [--quick]
"""
import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.envs import View, gaussian_global
from aggbench.metrics import summarize, summarize_values, paired_difference, per_episode
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

NS = (2, 4, 8, 16, 32)
# E2 overlap: n_pieces K per cell (expected agents per piece ~ 4N/K = 0.25, 1, 4); strengths scale as sqrt(2/N)
E2_OVERLAP = {"low": lambda n: 16 * n, "mid": lambda n: 4 * n, "high": lambda n: max(n, 6)}  # n_pieces K
E1_OVERLAP = {"low": 0.0, "mid": 1.0, "high": 3.0}                                         # factor_scale
K_MAX = 16 * max(NS)
LEARNED = ("softmax", "softmax_T", "softmax_gain", "sigmoid", "linear", "clearing_learned")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cell_env(env_name, n, ov):
    if env_name == "E2":  # piece strengths scaled by sqrt(2/N) so the Bayes posterior is not saturated at large N
        sc = math.sqrt(2 / n)
        return make_env("E2", n_agents=n, n_pieces=E2_OVERLAP[ov](n), strength_range=(0.1 * sc, 0.6 * sc))
    return make_env("E1", n_agents=n, factor_scale=E1_OVERLAP[ov])


def sample_view(env, split, n, chunk=500):
    """Concatenate chunked batch.view('full') objects (keeps memory low for large K)."""
    views, truth = [], []
    for s in range(0, n, chunk):
        b = env.sample(split, min(chunk, n - s), start=s)
        views.append(b.view("full"))
        truth.append(b.truth)
        del b
    v0 = views[0]
    rep = {k: np.concatenate([v.reports[k] for v in views]) for k in v0.reports}
    return View(v0.kind, rep, np.concatenate([v.keys for v in views]).astype(np.float32), v0.public), \
        np.concatenate(truth)


# ----------------------------------------------------------------------------- tokens and values (view only)

def tokens_and_values(view):
    """tokens (n, N, D) float32; values for the weighted combination (numpy float64)."""
    N = view.public["n_agents"]
    logn = np.full(view.keys.shape[:2] + (1,), math.log(N) / math.log(32), np.float32)
    if view.kind == "bernoulli":
        K = view.public["n_pieces"]
        l = view.reports["logit"] - view.public["prior_logit"]
        cov, cmu = view.keys[..., :K], view.keys[..., K:]
        pad = np.zeros(view.keys.shape[:2] + (K_MAX - K,), np.float32)
        tok = np.concatenate([(l / 4)[..., None].astype(np.float32), logn, cov, pad, 2 * cmu, pad], -1)
        return tok.astype(np.float32), {"dl": l}
    P = view.reports["prec"]
    h = np.einsum("naij,naj->nai", P, view.reports["mean"])
    eye = np.eye(P.shape[-1])
    Pm = P - eye
    feat = np.concatenate([np.log1p(np.abs(Pm)).reshape(*P.shape[:2], -1) * np.sign(Pm).reshape(*P.shape[:2], -1),
                           np.sign(h) * np.log1p(np.abs(h))], -1).astype(np.float32)
    tok = np.concatenate([feat, logn, view.keys.astype(np.float32)], -1)
    return tok.astype(np.float32), {"Pm": Pm, "h": h}


def combine_np(view, vals, a):
    """Forecast dict from per-agent weights a (n, N), float64."""
    a = np.asarray(a, np.float64)
    if view.kind == "bernoulli":
        return {"logit": view.public["prior_logit"] + (a * vals["dl"]).sum(1)}
    d = vals["Pm"].shape[-1]
    prec = np.eye(d) + np.einsum("na,naij->nij", a, vals["Pm"])
    h = np.einsum("na,nai->ni", a, vals["h"])
    return {"mean": np.linalg.solve(prec, h[..., None])[..., 0], "prec": prec}


# ----------------------------------------------------------------------------- model

def attn_weights(s, variant, bias=None):
    """s: scores (..., N_keys). Returns non-negative weights over the last axis."""
    if variant.startswith("softmax"):
        return torch.softmax(s, -1)
    if variant == "sigmoid":
        return torch.sigmoid(s + bias)
    if variant == "linear":
        return s  # already phi(q).phi(k) >= 0
    raise ValueError(variant)


class Block(nn.Module):
    def __init__(self, w, heads, variant):
        super().__init__()
        self.h, self.dh, self.variant = heads, w // heads, variant
        self.ln1, self.ln2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.qkv, self.o = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))
        self.bias = nn.Parameter(torch.full((heads, 1, 1), -2.0))

    def forward(self, x):
        n, N, w = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(n, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.variant == "linear":
            s = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-1, -2) / self.dh
        else:
            s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        a = attn_weights(s, self.variant, self.bias)
        x = x + self.o((a @ v).transpose(1, 2).reshape(n, N, w))
        return x + self.mlp(self.ln2(x))


class Net(nn.Module):
    def __init__(self, in_dim, variant, w=128, layers=2, heads=4):
        super().__init__()
        self.variant = variant
        self.embed = nn.Linear(in_dim, w)
        if variant == "clearing_learned":
            self.phi = nn.Sequential(nn.Linear(in_dim, w), nn.GELU(), nn.Linear(w, w))
            self.log_eps = nn.Parameter(torch.tensor(-2.0))
            return
        self.blocks = nn.ModuleList([Block(w, heads, variant) for _ in range(layers)])
        self.ln = nn.LayerNorm(w)
        self.q, self.k = nn.Parameter(torch.randn(w) / math.sqrt(w)), nn.Linear(w, w)
        self.b = nn.Parameter(torch.zeros(()))
        self.logT = nn.Parameter(torch.zeros(()))
        if variant == "softmax_gain":
            self.gain = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1))

    def forward(self, tok):
        """Per-agent weights a (n, N)."""
        if self.variant == "clearing_learned":
            phi = self.phi(tok)
            G = phi @ phi.transpose(-1, -2)
            c = torch.diagonal(G, dim1=-2, dim2=-1)
            eye = torch.eye(G.shape[-1], device=G.device)
            A = (G + (F.softplus(self.log_eps) + 1e-3) * eye).double()
            return torch.linalg.solve(A, c.double()[..., None])[..., 0].float()
        x = self.embed(tok)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        k = self.k(x)
        if self.variant == "linear":
            s = (F.elu(k) + 1) @ (F.elu(self.q) + 1) / k.shape[-1]
        else:
            s = k @ self.q / math.sqrt(k.shape[-1])
        a = attn_weights(s, self.variant, self.b)
        if self.variant == "softmax_T":
            a = a * F.softplus(self.logT) / math.log(2.0)
        if self.variant == "softmax_gain":
            a = a * F.softplus(self.gain(x)[..., 0])
        return a


def torch_loss(kind, a, vals, truth):
    if kind == "bernoulli":
        logit = vals["l0"] + (a * vals["dl"]).sum(1)
        return F.binary_cross_entropy_with_logits(logit, truth)
    a = a.double()
    if vals.get("clamp"):
        a = a.clamp(min=0)
    d = vals["Pm"].shape[-1]
    prec = torch.eye(d, device=a.device, dtype=torch.float64) + torch.einsum("na,naij->nij", a, vals["Pm"])
    h = torch.einsum("na,nai->ni", a, vals["h"])
    L = torch.linalg.cholesky(prec)
    mean = torch.cholesky_solve(h[..., None], L)[..., 0]
    diff = truth - mean
    quad = torch.einsum("ni,nij,nj->n", diff, prec, diff)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    return (0.5 * (quad - logdet)).mean()


@dataclass
class TrainCfg:
    steps: int = 6000
    batch: int = 128
    lr: float = 1e-3


class Aggregator:
    """aggbench learned-aggregator interface. fit/predict take a View or a list of Views (one per cell)."""

    def __init__(self, seed, variant, in_dim, cfg):
        torch.manual_seed(seed)
        self.seed, self.variant, self.cfg = seed, variant, cfg
        # the module lives on the CPU between calls: aggbench.controls.random_weights draws CPU tensors
        self.module = Net(in_dim, variant)
        self.log = []

    def _weights(self, view):
        tok, vals = tokens_and_values(view)
        out = []
        net = copy.deepcopy(self.module).to(DEV).eval()
        with torch.no_grad():
            for s in range(0, len(tok), 512):
                out.append(net(torch.tensor(tok[s:s + 512], device=DEV)).double().cpu().numpy())
        a = np.concatenate(out)
        if view.kind == "gaussian" and self.variant == "clearing_learned":
            a = np.clip(a, 0, None)
        return a, vals

    def fit(self, views, truths):
        views, truths = (views, truths) if isinstance(views, list) else ([views], [truths])
        cells = []
        for v, y in zip(views, truths):
            tok, vals = tokens_and_values(v)
            tv = {k: torch.tensor(x, device=DEV, dtype=torch.float64 if v.kind == "gaussian" else torch.float32)
                  for k, x in vals.items()}
            tv["l0"] = v.public.get("prior_logit", 0.0)
            tv["clamp"] = self.variant == "clearing_learned"
            cells.append((v.kind, torch.tensor(tok, device=DEV),
                          torch.tensor(y, device=DEV, dtype=torch.float64 if v.kind == "gaussian" else torch.float32),
                          tv))
        self.module.to(DEV)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.cfg.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.cfg.lr, total_steps=self.cfg.steps, pct_start=0.05)
        gen = np.random.default_rng(self.seed)
        self.module.train()
        run = 0.0
        for step in range(self.cfg.steps):
            kind, tok, y, tv = cells[gen.integers(len(cells))]
            idx = torch.tensor(gen.integers(0, len(tok), self.cfg.batch), device=DEV)
            vb = {k: (x[idx] if torch.is_tensor(x) else x) for k, x in tv.items()}
            loss = torch_loss(kind, self.module(tok[idx]), vb, y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            run = 0.98 * run + 0.02 * loss.item()
            if (step + 1) % 500 == 0:
                self.log.append({"step": step + 1, "loss_ema": run})
        self.module.cpu()
        return self

    def predict(self, views):
        if isinstance(views, list):
            return [self.predict(v) for v in views]
        a, vals = self._weights(views)
        f = combine_np(views, vals, a)
        f["_total_weight"] = a.sum(1)
        return f


# ----------------------------------------------------------------------------- exact clearing from keys

def clearing_exact(view):
    """Per-agent weights from the clearing solve with the kernel read off the full keys (no learning)."""
    if view.kind == "bernoulli":
        K = view.public["n_pieces"]
        cov = view.keys[..., :K].astype(np.float64)
        mu = view.keys[..., K:].astype(np.float64).max(1)            # each covered piece's strength
        S = np.einsum("nak,nk,nbk->nab", cov, 2 * mu ** 2, cov)
        diag = np.diagonal(S, axis1=1, axis2=2)
        beta = np.einsum("nab,nb->na", np.linalg.pinv(S, rcond=1e-10, hermitian=True), diag)
        f = {"logit": view.public["prior_logit"] + (beta * (view.reports["logit"] - view.public["prior_logit"])).sum(1)}
        f["_total_weight"] = beta.sum(1)
        return f
    # E1: keys = [A (m*d), Bf (m*r), log sigma]; recover s_i from the report as in oracle_clearing
    n, N, _ = view.keys.shape
    d = view.public["d"]
    keys = view.keys.astype(np.float64)
    m = 2
    r = (keys.shape[-1] - 1 - m * d) // m
    means, precs = [], []
    for e in range(n):
        A = keys[e, :, :m * d].reshape(N, m, d)
        Bf = keys[e, :, m * d:m * d + m * r].reshape(N, m, r)
        sig = np.exp(keys[e, :, -1])
        s = np.zeros((N, m))
        for i in range(N):
            R = Bf[i] @ Bf[i].T + sig[i] ** 2 * np.eye(m)
            hh = view.reports["prec"][e, i] @ view.reports["mean"][e, i]
            s[i] = np.linalg.lstsq(A[i].T @ np.linalg.inv(R), hh, rcond=None)[0]
        mean, prec = gaussian_global(A, Bf, sig, s)
        means.append(mean)
        precs.append(prec)
    return {"mean": np.array(means), "prec": np.array(precs)}


def bayes_total_weight(view, bayes):
    """Total weight Bayes puts on the reports. E2: sum of clearing weights; E1: tr(P_bayes - I) / tr(mean_i P_i - I)."""
    if view.kind == "bernoulli":
        return clearing_exact(view)["_total_weight"]
    d = bayes["prec"].shape[-1]
    num = np.trace(bayes["prec"] - np.eye(d), axis1=1, axis2=2)
    den = np.trace(view.reports["prec"].mean(1) - np.eye(d), axis1=1, axis2=2)
    return num / den


def strip(f):
    return {k: v for k, v in f.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", default="E2,E1")
    ap.add_argument("--arms", default=",".join(LEARNED))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--commit", default="unknown", help="repo commit the run was launched from (.git is not synced to the VM)")
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a local check (SMOKE, no verdicts)")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.n_dev, args.steps, args.seeds = 300, 200, 200, "0"
    torch.set_num_threads(2 if DEV.type == "cpu" else 8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    track = "SMOKE" if args.quick else "EXPLORATORY"
    cfg = TrainCfg(steps=args.steps)
    summary = {"track": track, "device": str(DEV), "args": vars(args), "cells": {}}

    for env_name in args.envs.split(","):
        t0 = time.time()
        overlaps = E2_OVERLAP if env_name == "E2" else E1_OVERLAP
        cells = [(n, ov) for ov in overlaps for n in NS]
        data = {}
        for n, ov in cells:
            env = cell_env(env_name, n, ov)
            trv, try_ = sample_view(env, "train", args.n_train)
            dvb = env.sample("dev", args.n_dev)
            data[(n, ov)] = (env, trv, try_, dvb)
        print(f"[{env_name}] data ready {time.time() - t0:.0f}s", flush=True)
        train_views = [data[c][1] for c in cells]
        train_truth = [data[c][2] for c in cells]
        dev_views = [data[c][3].view("full") for c in cells]
        in_dim = tokens_and_values(dev_views[0])[0].shape[-1]

        preds = {}  # (arm, seed, ctl) -> list of forecasts per cell
        logs = {}
        for arm in args.arms.split(","):
            for seed in seeds:
                t1 = time.time()
                holder = {}

                def make(s, arm=arm, holder=holder):
                    agg = Aggregator(s, arm, in_dim, cfg)
                    holder.setdefault("first", agg)
                    return agg
                res = run_with_controls(make, train_views, train_truth, dev_views, seed=seed)
                logs[f"{arm}:seed{seed}"] = holder["first"].log
                for ctl, fl in res.items():
                    preds[(arm, seed, ctl)] = fl
                print(f"[{env_name}] {arm} seed {seed} {time.time() - t1:.0f}s loss {holder['first'].log[-1:]}",
                      flush=True)

        for ci, (n, ov) in enumerate(cells):
            env, trv, try_, dvb = data[(n, ov)]
            dv = dev_views[ci]
            tr_small = env.sample("train", min(1000, args.n_train))  # extremizing alpha fitted on train
            forecasts, alpha = all_baselines(tr_small, dvb, "full")
            ce = clearing_exact(dv)
            forecasts["clearing_exact"] = strip(ce)
            totw = {"bayes": summarize_values(bayes_total_weight(dv, dvb.bayes), n_boot=500)}
            if "_total_weight" in ce:
                totw["clearing_exact"] = summarize_values(ce["_total_weight"], n_boot=500)
            learned, controls = [], {}
            for arm in args.arms.split(","):
                for seed in seeds:
                    base = f"{arm}:s{seed}"
                    for ctl in ("trained", "control:untrained_init", "control:random_network"):
                        name = base if ctl == "trained" else f"{base}:{ctl}"
                        f = preds[(arm, seed, ctl)][ci]
                        forecasts[name] = strip(f)
                        totw[name] = summarize_values(f["_total_weight"], n_boot=500)
                    learned.append(base)
                    controls[base] = {"random_network": f"{base}:control:random_network",
                                      "untrained_init": f"{base}:control:untrained_init"}
            methods = {k: summarize(f, dvb, n_boot=1000) for k, f in forecasts.items()}
            paired = {}
            for name in learned + ["clearing_exact", "sum_logodds", "log_pool"]:
                paired[name] = {ref: paired_difference(forecasts[name], forecasts[ref], dvb, n_boot=1000)
                                for ref in ("extremized_log_pool", "oracle_clearing")}
            # seed-mean regret per arm, CI over episodes of the per-episode seed mean
            seed_mean = {}
            for arm in args.arms.split(","):
                reg = np.mean([per_episode(forecasts[f"{arm}:s{s}"], dvb)["regret"] for s in seeds], 0)
                seed_mean[arm] = summarize_values(reg, n_boot=1000)
            run_id = f"l6-softmax-vs-additive-20261020-{env_name}-N{n}-{ov}"
            result = make_result(
                run_id=run_id, track=track, smoke=args.quick, env=env, visibility="full",
                splits={"train": {"split": "train", "first_seed": 0, "n": args.n_train},
                        "eval": split_record(dvb, "dev")},
                oracle_check=oracle_check(env), methods=methods, learned_arms=learned, controls=controls,
                extra={"cell": {"env": env_name, "n_agents": n, "overlap": ov},
                       "note": "learned arms trained once per (env, arm, seed) on the mixture of all 15 cells of "
                               "this env (train split, truth labels, no distillation); evaluated here on this "
                               "cell's dev split. clearing_exact is computed from the full keys (by construction).",
                       "total_weight": totw, "paired_difference_regret": paired, "seed_mean_regret": seed_mean,
                       "extremize_alpha": alpha, "train_cfg": vars(cfg), "seeds": seeds,
                       "train_loss_log": logs if ci == 0 else "see the N2-low cell of this env",
                       "device": str(DEV), "runner_commit": args.commit,
                       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            write_result(out / "json", result, name=f"{env_name}-N{n}-{ov}.json")
            summary["cells"][f"{env_name}-N{n}-{ov}"] = {
                "regret": {k: m["regret"] for k, m in methods.items() if ":control" not in k},
                "controls_regret": {k: m["regret"] for k, m in methods.items() if ":control" in k},
                "seed_mean_regret": seed_mean,
                "total_weight": {k: v for k, v in totw.items() if ":control" not in k}}
        print(f"[{env_name}] done {time.time() - t0:.0f}s", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
