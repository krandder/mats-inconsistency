"""EXPLORATORY lane L2 (stage 1): vector messages from local encoders, aggregated by attention.

Each agent i runs the same encoder on its OWN inputs only (batch.agent_inputs("full"): its raw
observation and its key) and sends a message (key k_i in R^m, value v_i in R^m). An attention
aggregator (2 self-attention blocks + attention pooling with learned seeds) sees only the messages.
Encoder and aggregator are trained end to end on the train split's truth labels (no distillation to
bayes). Message dimension m is swept. Evaluation on the dev split only; the test split is never sampled.

Usage (torch venv, CPU):
    PYTHONPATH=. python3 results/explore/l2-vector-messages-20261020/run_l2.py --env E2 --dims 1 2 4 8 16
"""
import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.messages import check_locality
from aggbench.metrics import paired_difference, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

HERE = Path(__file__).resolve().parent
RUN_ID = "l2-vector-messages-20261020"
VIS = "full"


# ----------------------------------------------------------------------------- agent-side inputs

class AgentInputs:
    """What the pipeline sees: each agent's own (obs, key) plus public constants. Passed to
    run_with_controls in place of a View; the aggregator part only ever sees encoder outputs."""

    def __init__(self, kind, obs, keys, public):
        self.kind, self.obs, self.keys, self.public = kind, obs, keys, public


def biased_agent_obs(batch):
    """E3-biased: the environment's distortion applied on the agent side, before encoding.

    The biased agent's raw observations are honest (README). We replace its piece values so that each
    piece LLR becomes lam'_k = a lam_k + b / n_i (n_i = pieces it sees). Its local log-odds are then
    exactly the environment's distorted report l0 + a (l - l0) + b, i.e. it is an agent that perceives
    its evidence through the same distortion. This is part of simulating the agent (like the env
    computing reports from oracle a, b); the aggregator never sees roles, a or b."""
    obs = batch.obs.copy()
    a, b = batch.oracle["a"], batch.oracle["b"]
    mu = batch.oracle["mu"]
    for e, i in zip(*np.nonzero(batch.oracle["role"] == 2)):
        seen = np.isfinite(obs[e, i])
        lam = 2 * mu[e] * obs[e, i]
        lam_new = a[e, i] * lam + b[e, i] / seen.sum()
        obs[e, i] = np.where(seen, lam_new / (2 * mu[e]), np.nan)
    return obs


def agent_inputs(batch, env_name):
    obs, keys, public = batch.agent_inputs(VIS)
    if env_name == "E3-biased":
        obs = biased_agent_obs(batch)
    return AgentInputs(batch.kind, obs, keys, public)


def local_features(obs, keys, kind):
    """Per-agent features from agent i's own obs and key only (elementwise along the agent axis)."""
    if kind == "bernoulli":
        K = obs.shape[-1]
        seen = np.isfinite(obs)
        x = np.where(seen, obs, 0.0)
        mu = keys[..., K:]                       # key "full" = [coverage bits, coverage * mu]
        llr = 2 * mu * x                         # local preprocessing: own piece LLRs
        return np.concatenate([x, seen.astype(float), keys, llr / 2], axis=-1)
    # E1: s_i and key (A_i, B_i, log sigma_i), plus the agent's own local posterior in information
    # form (h_i = A^T R^-1 s, J_i = A^T R^-1 A, R = B B^T + sigma^2 I), computed from its own inputs only.
    m = obs.shape[-1]
    d = (keys.shape[-1] - 1 - m * m) // m        # key = [A (m*d), B (m*r), log sigma]; here r = m = 2
    r = (keys.shape[-1] - 1 - m * d) // m
    A = keys[..., :m * d].reshape(*keys.shape[:-1], m, d)
    B = keys[..., m * d:m * d + m * r].reshape(*keys.shape[:-1], m, r)
    sig2 = np.exp(2 * keys[..., -1])[..., None, None]
    R = B @ np.swapaxes(B, -1, -2) + sig2 * np.eye(m)
    Rinv_A = np.linalg.solve(R, A)
    J = np.swapaxes(A, -1, -2) @ Rinv_A
    h = np.einsum("...md,...m->...d", Rinv_A, obs)
    symlog = lambda z: np.sign(z) * np.log1p(np.abs(z))
    return np.concatenate([obs, keys, h / 5, symlog(h), J.reshape(*J.shape[:-2], -1) / 10,
                           symlog(J.reshape(*J.shape[:-2], -1))], axis=-1)


# ----------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, w, heads=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(w), nn.LayerNorm(w)
        self.attn = nn.MultiheadAttention(w, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class Pipeline(nn.Module):
    """encoder (per agent, shared weights) -> messages (key, value in R^m) -> attention aggregator."""

    def __init__(self, in_dim, m, out_dim, width=64, enc_width=128, seeds=4):
        super().__init__()
        self.m = m
        self.encoder = nn.Sequential(nn.Linear(in_dim, enc_width), nn.GELU(), nn.Linear(enc_width, enc_width),
                                     nn.GELU(), nn.Linear(enc_width, 2 * m))
        self.embed = nn.Linear(2 * m, width)
        self.blocks = nn.ModuleList([Block(width), Block(width)])
        self.seeds = nn.Parameter(torch.randn(seeds, width) / np.sqrt(width))
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(seeds * width), nn.Linear(seeds * width, 128), nn.GELU(),
                                  nn.Linear(128, out_dim))

    def encode(self, feats):              # (n, agents, in_dim) -> (n, agents, 2m); row-wise, no mixing
        return self.encoder(feats)

    def aggregate(self, msgs):            # sees messages only
        z = self.embed(msgs)
        for b in self.blocks:
            z = b(z)
        q = self.seeds.unsqueeze(0).expand(z.shape[0], -1, -1)
        pooled = self.pool(q, z, z, need_weights=False)[0]
        return self.head(pooled.flatten(1))

    def forward(self, feats):
        return self.aggregate(self.encode(feats))


def tril_indices(d):
    return torch.tril_indices(d, d)


class VectorMessageAggregator:
    """.module / .fit(inputs, truth) / .predict(inputs), as aggbench.controls expects."""

    def __init__(self, seed, kind, in_dim, m, d=4, steps=3000, batch=512, lr=2e-3, log=None):
        torch.manual_seed(seed)
        self.kind, self.d, self.steps, self.bs, self.lr, self.log = kind, d, steps, batch, lr, log
        out_dim = 1 if kind == "bernoulli" else d + d * (d + 1) // 2
        self.module = Pipeline(in_dim, m, out_dim)
        self.fit_info = {}

    def _feats(self, inp):
        return torch.tensor(local_features(inp.obs, inp.keys, self.kind), dtype=torch.float32)

    def _decode(self, out):
        if self.kind == "bernoulli":
            return out[:, 0]
        d = self.d
        mean = out[:, :d]
        L = torch.zeros(out.shape[0], d, d, dtype=out.dtype)
        ti = tril_indices(d)
        L[:, ti[0], ti[1]] = out[:, d:]
        prec = torch.eye(d, dtype=out.dtype) + L @ L.transpose(1, 2)  # posterior precision >= prior (I)
        return mean, prec

    def _loss(self, out, y, l0):
        if self.kind == "bernoulli":
            return F.binary_cross_entropy_with_logits(l0 + self._decode(out), y)
        mean, prec = self._decode(out)
        diff = (y - mean).unsqueeze(-1)
        quad = (diff.transpose(1, 2) @ prec @ diff)[:, 0, 0]
        return (0.5 * (quad - torch.linalg.slogdet(prec)[1])).mean()

    def fit(self, inp, truth):
        X = self._feats(inp)
        Y = torch.tensor(truth, dtype=torch.float32)
        l0 = float(inp.public.get("prior_logit", 0.0))
        n_val = min(10000, len(X) // 10)       # held-out slice of TRAIN for checkpoint choice
        Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.05)
        g = torch.Generator().manual_seed(1234)
        best, best_state, t0, hist = float("inf"), None, time.time(), []
        for step in range(1, self.steps + 1):
            idx = torch.randint(0, len(Xt), (self.bs,), generator=g)
            self.module.train()
            loss = self._loss(self.module(Xt[idx]), Yt[idx], l0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 250 == 0 or step == self.steps:
                self.module.eval()
                with torch.no_grad():
                    vl = float(np.mean([self._loss(self.module(Xv[j:j + 2500]), Yv[j:j + 2500], l0).item()
                                        for j in range(0, n_val, 2500)]))
                hist.append((step, round(vl, 5)))
                if vl < best:
                    best = vl
                    best_state = {k: v.clone() for k, v in self.module.state_dict().items()}
                if self.log:
                    self.log(f"  step {step} train_loss {loss.item():.4f} val_loss {vl:.4f} ({time.time() - t0:.0f}s)")
        self.module.load_state_dict(best_state)
        self.module.eval()
        self.fit_info = {"best_val_loss": best, "val_history": hist, "seconds": round(time.time() - t0, 1)}

    def predict(self, inp):
        self.module.eval()
        X = self._feats(inp)
        with torch.no_grad():
            outs = torch.cat([self.module(X[j:j + 4096]) for j in range(0, len(X), 4096)]).double()
        if self.kind == "bernoulli":
            return {"logit": float(inp.public["prior_logit"]) + self._decode(outs).numpy()}
        mean, prec = self._decode(outs)
        return {"mean": mean.numpy(), "prec": prec.numpy()}

    def encode_numpy(self, obs, keys):
        with torch.no_grad():
            f = torch.tensor(local_features(obs, keys, self.kind), dtype=torch.float32)
            return self.module.encode(f).double().numpy()


# ----------------------------------------------------------------------------- run

def cached_sample(env, split, n, cache):
    path = Path(cache) / f"{env.name}_{split}_{n}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    b = env.sample(split, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(b))
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["E1", "E2", "E3-sybil", "E3-biased", "E3-collude"])
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-dev", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="/tmp/l2-vecmsg-cache")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    logf = open(Path(args.out) / f"log_{args.env}.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    env = make_env(args.env)
    t0 = time.time()
    tr = cached_sample(env, "train", args.n_train, args.cache)
    dv = cached_sample(env, "dev", args.n_dev, args.cache)
    log(f"[{args.env}] sampled train {len(tr)} dev {len(dv)} in {time.time() - t0:.0f}s")
    tr_in, dv_in = agent_inputs(tr, args.env), agent_inputs(dv, args.env)
    in_dim = local_features(dv_in.obs[:1], dv_in.keys[:1], dv.kind).shape[-1]

    forecasts, alpha = all_baselines(tr, dv, VIS)
    learned, controls, locality, fit_info, paired = [], {}, {}, {}, {}
    for m in args.dims:
        name = f"vecmsg_m{m}"
        log(f"[{args.env}] m={m} (message = key R^{m} + value R^{m})")
        built = {}

        def make(seed, m=m):
            agg = VectorMessageAggregator(seed, dv.kind, in_dim, m, steps=args.steps, log=log)
            built.setdefault("first", agg)
            return agg

        arms = run_with_controls(make, tr_in, tr.truth, dv_in, seed=args.seed)
        trained = built["first"]
        for arm, f in arms.items():
            forecasts[f"{name}:{arm}"] = f
        learned.append(f"{name}:trained")
        controls[f"{name}:trained"] = {"random_network": f"{name}:control:random_network",
                                       "untrained_init": f"{name}:control:untrained_init"}
        sl = slice(0, 500)
        try:
            ok = check_locality(trained.encode_numpy, dv_in.obs[sl], dv_in.keys[sl], trials=5)
            locality[name] = {"passed": bool(ok), "episodes": 500, "trials": 5, "atol": 1e-9,
                              "encoder": "trained encoder, message = [key, value] in R^(2m)"}
        except AssertionError as err:
            locality[name] = {"passed": False, "error": str(err)}
        fit_info[name] = trained.fit_info
        paired[name] = {
            f"trained_minus_{b}": paired_difference(arms["trained"], forecasts[b], dv)
            for b in ("oracle_clearing", "extremized_log_pool")}
        r = summarize(arms["trained"], dv, n_boot=500)["regret"]
        log(f"[{args.env}] m={m} regret {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]; "
            f"minus oracle_clearing {paired[name]['trained_minus_oracle_clearing']['mean']:+.4f} "
            f"locality {locality[name].get('passed')}")

    methods = {k: summarize(f, dv) for k, f in forecasts.items()}
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extra = {
        "lane": "L2", "question": "vector messages from local encoders + attention aggregator vs scalar reports",
        "runner": str(Path(__file__).relative_to(HERE.parents[2])), "runner_sha256": runner_sha,
        "message_dims": args.dims, "message_definition": "per agent: learned key in R^m and learned value in R^m (2m floats)",
        "training": {"labels": "truth (no distillation to bayes)", "steps": args.steps, "batch": 512, "lr": 2e-3,
                     "optimizer": "AdamW wd 1e-4, one-cycle", "checkpoint_choice": "best loss on first 10k train episodes held out",
                     "train_seed": args.seed},
        "inputs": "batch.agent_inputs('full'): own obs + own key; E2/E3 features add own piece LLRs mu*x",
        "e3_biased_handling": ("distortion applied agent-side: biased agent's piece LLRs become a*lam + b/n_i so its "
                               "local log-odds equal the environment's distorted report") if args.env == "E3-biased" else None,
        "locality_check": locality, "paired_differences": paired, "fit_info": fit_info,
        "extremize_alpha": alpha, "torch_threads": args.threads,
    }
    result = make_result(run_id=f"{RUN_ID}-{args.env}", track="EXPLORATORY", env=env, visibility=VIS,
                         splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
                         oracle_check=oracle_check(env), methods=methods, learned_arms=learned,
                         controls=controls, extra=extra)
    p = write_result(args.out, result, name=f"result_{args.env}.json")
    log(f"[{args.env}] wrote {p} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
