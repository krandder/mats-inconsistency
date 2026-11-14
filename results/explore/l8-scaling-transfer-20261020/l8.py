"""L8 (EXPLORATORY): scaling and transfer of permutation-invariant aggregators on aggbench E2.

Train on few agents (N = 4..8, K = 12 pieces, 2-6 pieces per agent) and test zero-shot on more agents,
more evidence pieces and other overlap levels. Three learned arms, all invariant to agent order and piece
order, and all defined for any N and K:

  attn1      one attention layer over agents. Agent token = report + sum over its pieces of phi(key piece).
             Attention logits = content term + per-head key overlap S_ij = sum_k f(key_ik) . g(key_jk);
             unnormalized overlap mass sum_j S_ij is also fed in (so the layer can count redundancy).
             Readout: logit = l0 + sum_i a_i (l_i - l0) + b, a_i and b from the tokens.
  attnloop   the same layer applied T times with shared weights (looped transformer), T = 4 in training;
             attnloop_T8 is the same trained weights run for 8 loops at test time.
  clearloop  learned clearing: piece estimates w_k start at 0; each loop every agent's residual
             e_i = (l_i - l0) - sum_k c_ik w_k is split over its claimed pieces by learned weights
             beta(key_ik) and averaged per piece (Cimmino-style), w <- w + gamma * update. Readout
             l0 + s sum_k u_k w_k with a learned per-piece gate u_k. T = 16 in training; clearloop_T64 runs
             64 loops at test time. With full keys and beta ~ mu^2 it can represent oracle_clearing.

Aggregators see batch.view(visibility) only (reports, keys, public). Labels: truth on the train split.
Every learned arm goes through aggbench.controls.run_with_controls; fit() is memoized per (arm, seed,
width, data, visibility) so one trained model is evaluated on every test condition (controls are rebuilt
from it each time exactly as run_with_controls does).

Usage:  python l8.py --out DIR [--quick]   (writes DIR/preds_<vis>_<cond>.npz, DIR/meta.json)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from aggbench import make_env  # noqa: E402
from aggbench.baselines import all_baselines, extremized, fit_extremize  # noqa: E402
from aggbench.controls import run_with_controls  # noqa: E402
from aggbench.results import harness_info  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_NS = (4, 5, 6, 7, 8)
TRAIN_CFG = dict(n_pieces=12, pieces_per_agent=(2, 6))

# test conditions: name -> make_env("E2", **overrides)
CONDITIONS = {
    "N4": dict(n_agents=4),
    "N8": dict(n_agents=8),
    "N16": dict(n_agents=16),
    "N32": dict(n_agents=32),
    "N64": dict(n_agents=64),
    "N8_K24_ppa4-12": dict(n_agents=8, n_pieces=24, pieces_per_agent=(4, 12)),
    "N8_K48_ppa8-24": dict(n_agents=8, n_pieces=48, pieces_per_agent=(8, 24)),
    "N8_K24_ppa2-6": dict(n_agents=8, n_pieces=24, pieces_per_agent=(2, 6)),
    "N8_K48_ppa2-6": dict(n_agents=8, n_pieces=48, pieces_per_agent=(2, 6)),
    "N8_ppa1-2": dict(n_agents=8, pieces_per_agent=(1, 2)),
    "N8_ppa6-10": dict(n_agents=8, pieces_per_agent=(6, 10)),
    "N8_ppa10-12": dict(n_agents=8, pieces_per_agent=(10, 12)),
    "N64_K48_ppa8-24": dict(n_agents=64, n_pieces=48, pieces_per_agent=(8, 24)),
}
CURVE_CONDS = ("N8", "N64")   # learning-curve models are evaluated here only


# ----------------------------------------------------------------------------- features

def view_tensors(view):
    """reports minus prior (n,N), per agent-piece features (n,N,K,3), coverage bits (n,N,K)."""
    l0 = view.public["prior_logit"]
    r = view.reports["logit"] - l0
    n, N, kd = view.keys.shape
    K = kd // 2
    c = view.keys[..., :K]
    cm = view.keys[..., K:]
    mu = np.where(c > 0, cm / np.maximum(c, 1e-9), 0.0)
    x = np.stack([c, mu, 2 * mu ** 2], -1)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return t(r), t(x), t(c)


def mlp(i, w, o):
    return nn.Sequential(nn.Linear(i, w), nn.GELU(), nn.Linear(w, w), nn.GELU(), nn.Linear(w, o))


# ----------------------------------------------------------------------------- networks

class AttnLayer(nn.Module):
    def __init__(self, w, heads=4):
        super().__init__()
        self.h, self.dh = heads, max(w // heads, 4)
        self.q, self.k, self.v = nn.Linear(w, heads * self.dh), nn.Linear(w, heads * self.dh), nn.Linear(w, w)
        self.f, self.g = nn.Linear(w, heads), nn.Linear(w, heads)
        self.upd = mlp(2 * w + 2 * heads, w, w)

    def forward(self, h, P, c):
        n, N, w = h.shape
        q = self.q(h).view(n, N, self.h, self.dh).transpose(1, 2)
        k = self.k(h).view(n, N, self.h, self.dh).transpose(1, 2)
        S = torch.einsum("nikh,njkh->nhij", self.f(P) * c[..., None], self.g(P) * c[..., None]) / 4
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh) + S, -1)
        out = torch.einsum("nhij,njd->nid", att, self.v(h)) / self.h
        mass = S.sum(-1).transpose(1, 2) / 8                        # (n,N,H): unnormalized overlap mass
        own = torch.diagonal(S, dim1=-2, dim2=-1).transpose(1, 2)   # (n,N,H): own key "self-overlap"
        return h + self.upd(torch.cat([h, out, mass, own], -1))


class AttnNet(nn.Module):
    def __init__(self, w, loops):
        super().__init__()
        self.loops = loops
        self.phi = mlp(3, w, w)
        self.emb = mlp(1 + w, w, w)
        self.layer = AttnLayer(w)
        self.a, self.b = nn.Linear(w, 1), nn.Linear(w, 1)
        with torch.no_grad():
            self.a.bias.fill_(0.25)

    def forward(self, r, x, c, loops=None):
        P = self.phi(x) * c[..., None]
        h = self.emb(torch.cat([r[..., None] / 4, P.sum(2) / 4], -1))
        for _ in range(loops or self.loops):
            h = self.layer(h, P, c)
        a = self.a(h)[..., 0]
        return (a * r).sum(1) + self.b(h.mean(1))[:, 0]


class ClearNet(nn.Module):
    def __init__(self, w, loops):
        super().__init__()
        self.loops = loops
        self.beta = mlp(3, w, 1)
        self.gate = mlp(3, w, 1)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, r, x, c, loops=None):
        beta = F.softplus(self.beta(x)[..., 0]) * c + 1e-6 * c        # (n,N,K)
        split = beta / (beta.sum(-1, keepdim=True) + 1e-6)             # agent's residual -> its pieces
        deg = c.sum(1)                                                 # (n,K) agents claiming piece k
        w = torch.zeros_like(deg)
        g = 2 * torch.sigmoid(self.gamma)
        for _ in range(loops or self.loops):
            e = r - (c * w[:, None, :]).sum(-1)                        # (n,N)
            w = w + g * (split * e[..., None]).sum(1) / deg.clamp(min=1)
        mmu = (x[..., 1] * c).sum(1) / deg.clamp(min=1)
        u = torch.sigmoid(self.gate(torch.stack([torch.log1p(deg), mmu, (deg > 0).float()], -1))[..., 0] + 2)
        return self.scale * (u * w).sum(-1)


# ----------------------------------------------------------------------------- aggregator (harness interface)

_FIT_CACHE = {}


class Aggregator:
    """.module, .fit(view(s), truth(s)), .predict(view). fit accepts a list of views (one per training N)."""

    def __init__(self, arm, seed, width, n_train, steps, test_loops=None):
        self.arm, self.seed, self.width, self.n_train, self.steps = arm, seed, width, n_train, steps
        self.test_loops = test_loops
        torch.manual_seed(seed)
        base = arm.split("_T")[0]
        if base == "attn1":
            self.module = AttnNet(width, 1)
        elif base == "attnloop":
            self.module = AttnNet(width, 4)
        elif base == "clearloop":
            self.module = ClearNet(width, 16)
        else:
            raise ValueError(arm)
        self.cache_key = (base, seed, width, n_train, steps)
        self.log = []

    def fit(self, views, truths):
        key = self.cache_key + (id(views),)
        if key in _FIT_CACHE:
            self.module.load_state_dict(_FIT_CACHE[key][0])
            self.log = _FIT_CACHE[key][1]
            return
        m = self.module.to(DEV)
        data = [tuple(t.to(DEV) for t in view_tensors(v)) + (torch.tensor(y, dtype=torch.float32, device=DEV),)
                for v, y in zip(views, truths)]
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=self.steps, pct_start=0.05)
        gen = torch.Generator().manual_seed(self.seed + 7)
        bs, t0 = 256, time.time()
        for step in range(self.steps):
            r, x, c, y = data[int(torch.randint(len(data), (1,), generator=gen))]
            idx = torch.randint(len(y), (bs,), generator=gen).to(DEV)
            loss = F.binary_cross_entropy_with_logits(m(r[idx], x[idx], c[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 500 == 0 or step == self.steps - 1:
                self.log.append((step, float(loss)))
        m.cpu()
        self.train_seconds = time.time() - t0
        _FIT_CACHE[key] = ({k: v.detach().clone() for k, v in m.state_dict().items()}, self.log)

    def predict(self, view):
        m = self.module.to(DEV).eval()
        r, x, c = view_tensors(view)
        out = []
        with torch.no_grad():
            for s in range(0, len(r), 250):
                sl = slice(s, s + 250)
                out.append(m(r[sl].to(DEV), x[sl].to(DEV), c[sl].to(DEV), loops=self.test_loops).double().cpu())
        m.cpu().train()
        return {"logit": view.public["prior_logit"] + torch.cat(out).numpy()}


# ----------------------------------------------------------------------------- run

def train_views(vis, n_total):
    """Train split, N = 4..8 at the training config; disjoint seed ranges per N (start offsets)."""
    per = n_total // len(TRAIN_NS)
    views, truths, recs = [], [], []
    for j, N in enumerate(TRAIN_NS):
        b = make_env("E2", n_agents=N, **TRAIN_CFG).sample("train", per, start=j * 1_000_000)
        views.append(b.view(vis))
        truths.append(b.truth)
        recs.append({"split": "train", "first_seed": int(b.seeds[0]), "n": int(len(b)), "n_agents": N})
    return views, truths, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vis", default="full,noisy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    q = a.quick
    n_eval = 300 if q else 2000
    steps = 200 if q else 6000
    main_width, main_data = 32, (2000 if q else 50000)
    seeds = (0,) if q else (0, 1, 2)
    curve = [] if q else [(w, main_data) for w in (8, 16, 64, 128)] + [(main_width, d) for d in (1000, 3000, 10000)]

    # arm spec: (method name, arm, seed, width, n_train, test_loops)
    specs = []
    for s in seeds:
        for arm, tl in (("attn1", None), ("attnloop", None), ("attnloop_T8", 8), ("clearloop", None),
                        ("clearloop_T64", 64)):
            specs.append((f"{arm}:s{s}", arm, s, main_width, main_data, tl, None))
    for w, d in curve:
        for arm in ("attn1", "attnloop", "clearloop"):
            specs.append((f"{arm}:w{w}:n{d}", arm, 0, w, d, None, CURVE_CONDS))

    meta = {"harness": harness_info(), "device": str(DEV), "torch": torch.__version__, "steps": steps,
            "batch_size": 256, "n_eval": n_eval, "train_ns": TRAIN_NS, "train_cfg": TRAIN_CFG,
            "conditions": CONDITIONS, "curve_conditions": CURVE_CONDS, "specs": specs, "train": {},
            "train_loss_log": {}, "alpha_train_cfg": {}}
    t_start = time.time()
    for vis in a.vis.split(","):
        trains = {}
        for d in sorted({sp[4] for sp in specs}):
            trains[d] = train_views(vis, d)
            meta["train"][f"{vis}:n{d}"] = trains[d][2]
        # extremizing alpha fitted on the training config (pooled N = 4..8), applied unchanged at test
        tv, tt, _ = trains[main_data]
        alphas = [fit_extremize(v, y) for v, y in zip(tv, tt)]
        alpha_tr = float(np.mean(alphas))
        meta["alpha_train_cfg"][vis] = {"per_N": alphas, "used": alpha_tr}
        for cname, ov in CONDITIONS.items():
            env = make_env("E2", **ov)
            dv = env.sample("dev", n_eval)
            tr_target = env.sample("train", 2000, start=8_000_000)   # only to fit alpha at the target config
            fc, alpha_tgt = all_baselines(tr_target, dv, vis)
            fc["extremized_log_pool@train_alpha"] = extremized(dv.view(vis), alpha_tr)
            fc = {"baseline:" + k: v["logit"] for k, v in fc.items()}
            for name, arm, s, w, d, tl, only in specs:
                if only and cname not in only:
                    continue
                views, truths, _ = trains[d]
                make = lambda seed, arm=arm, w=w, d=d, tl=tl: Aggregator(arm, seed, w, d, steps, tl)
                t0 = time.time()
                arms = run_with_controls(make, views, truths, dv.view(vis), seed=s)
                for k, f in arms.items():
                    fc[f"{name}:{k}"] = f["logit"]
                key = f"{vis}:{name}"
                if key not in meta["train_loss_log"]:
                    cache = [v for kk, v in _FIT_CACHE.items() if kk[:5] == (arm.split('_T')[0], s, w, d, steps)]
                    meta["train_loss_log"][key] = cache[0][1] if cache else None
                print(f"[{time.time() - t_start:7.0f}s] {vis} {cname} {name} {time.time() - t0:.1f}s", flush=True)
            np.savez_compressed(os.path.join(a.out, f"preds_{vis}_{cname}.npz"), seeds=dv.seeds,
                                bayes=dv.bayes["logit"], truth=dv.truth, **fc)
            meta.setdefault("alpha_target_cfg", {})[f"{vis}:{cname}"] = alpha_tgt
        _FIT_CACHE.clear()   # cache keys use id(views); clear between visibilities
    meta["param_counts"] = {f"{arm}:w{w}": sum(p.numel() for p in Aggregator(arm, 0, w, 1, 1).module.parameters())
                            for arm in ("attn1", "attnloop", "clearloop") for w in (8, 16, 32, 64, 128)}
    meta["wall_seconds"] = time.time() - t_start
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)
    print("DONE", meta["wall_seconds"])


if __name__ == "__main__":
    main()
