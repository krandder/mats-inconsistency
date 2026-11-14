"""EXPLORATORY lane L5 (stage 1, robustness): Sybil copies, colluders and a biased source with provenance
hidden (visibility "none") or noisy ("noisy").

Question: can a learned aggregator detect and discount duplicated sources, colluding sources and a biased
source from the content of their reports alone, and how much regret remains against oracle_clearing?

Arms (all evaluated on the dev split only; the test split is never sampled):
  set:<train>      pairwise-relation set network over agent tokens [(l_i - l0)/4, key_i]; pair features
                   are the report difference d_ij = l_i - l_j as (d/4, |d|/4, log(|d| + 1e-3)/7) and, when
                   keys exist, the mean absolute key difference. Output logit = l0 + sum_i w_i (l_i - l0)
                   + sum_i c_i with (w_i, c_i) read out per agent. Trained on truth labels (no distillation).
                   <train> is "clean" (E2 with the same total number of agents, no adversaries),
                   "matched" (the evaluation environment's own train split) or "pooled" (all seven
                   evaluation environments mixed, different train seed ranges per environment).
  market_wealth:slot      the "traders" baseline: each source's wealth is exp(eta * its cumulative log score
                   over the training episodes) (Hedge; eta = 1 is the Kelly/Bayes wealth update), weights are
                   normalised wealth, aggregate = l0 + alpha sum_i w_i (l_i - l0); eta chosen on the last 20%
                   of train, alpha fitted on train. Sources are the view's agent slots, which the harness
                   shuffles every episode, so a slot is not a persistent trader.
  market_wealth:identity  the same market, but the slots are persistent trader identities reconstructed from
                   the oracle roles (colluders, biased, honest, Sybil source, Sybil copies). REFERENCE ONLY:
                   the view has no identities; this arm shows what a market with persistent identities does.
  logpool_fit:identity    per-identity weights w_i and offsets c_i of a log pool fitted by gradient descent on
                   train log loss (the best fixed per-source weighting). REFERENCE ONLY, as above.
Controls: every torch arm goes through aggbench.controls.run_with_controls (untrained init, random network).

Stage "train" (torch; run on the L4) writes forecasts to <out>/; stage "assemble" (numpy + jsonschema, local)
recomputes baselines and writes the harness-validated JSONs. Usage:
  python3 run_l5.py train --out /root/runs/<id> --vis none --seed 0 --device cuda
  python3 run_l5.py market --out /root/runs/<id>
  python3 run_l5.py assemble --inp /tmp/.../runs/<id>
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from aggbench import make_env
from aggbench.envs import View

HERE = Path(__file__).resolve().parent
RUN_ID = "l5-sybil-collusion-20261020"
N_TRAIN = 40000
N_DEV = 2000
N_MAX = 12

# evaluation environments: name -> (preset, overrides); total agents in brackets
EVAL_ENVS = {
    "sybil0": ("E2", {}),                                   # [5] no adversary
    "sybil1": ("E3-sybil", {"n_sybil": 1}),                 # [6]
    "sybil3": ("E3-sybil", {"n_sybil": 3}),                 # [8]
    "sybil7": ("E3-sybil", {"n_sybil": 7}),                 # [12]
    "collude3": ("E3-collude", {}),                         # [5] 3 colluders
    "biased1": ("E3-biased", {}),                           # [5] 1 biased
    "mixed": ("E3-sybil", {"n_sybil": 3, "n_colluders": 3, "n_biased": 1}),  # [8] all three
}
CLEAN_FOR = {"sybil0": "clean5", "sybil1": "clean6", "sybil3": "clean8", "sybil7": "clean12",
             "collude3": "clean5", "biased1": "clean5", "mixed": "clean8"}
CLEAN_ENVS = {"clean5": 5, "clean6": 6, "clean8": 8, "clean12": 12}


def env_of(name):
    if name in CLEAN_ENVS:
        return make_env("E2", n_agents=CLEAN_ENVS[name])
    preset, kw = EVAL_ENVS[name]
    return make_env(preset, **kw)


def train_batch(name, n=N_TRAIN, slot=0):
    """Train split; `slot` offsets the seed range so pooled training does not reuse seeds across envs."""
    return env_of(name).sample("train", n, start=slot * n)


def pad_view(view, n_max=N_MAX):
    """Pad agents to n_max with NaN (an absent agent)."""
    l = view.reports["logit"]
    n, N = l.shape
    lp = np.full((n, n_max), np.nan)
    lp[:, :N] = l
    kp = np.full((n, n_max, view.keys.shape[2]), np.nan)
    kp[:, :N] = view.keys
    return View(view.kind, {"logit": lp}, kp, dict(view.public))


def concat_views(views):
    views = [pad_view(v) for v in views]
    return View(views[0].kind, {"logit": np.concatenate([v.reports["logit"] for v in views])},
                np.concatenate([v.keys for v in views]), dict(views[0].public))


def canonical_order(batch):
    """Per episode, agent positions in a persistent identity order (uses oracle roles: reference arms only).

    Order: colluders, biased, honest agents other than the Sybil source, the Sybil source, Sybil copies.
    Agents inside one group are exchangeable (their coverage is redrawn every episode)."""
    role, cov = batch.oracle["role"], batch.oracle["cov"]
    orders = []
    for e in range(len(batch)):
        r = role[e]
        col, bia, syb = (list(np.flatnonzero(r == k)) for k in (1, 2, 3))
        honest = list(np.flatnonzero(r == 0))
        if syb:
            src = next(i for i in honest if np.array_equal(cov[e, i], cov[e, syb[0]]))
            honest = [i for i in honest if i != src] + [src]
        orders.append(col + bia + honest + syb)
    return np.array(orders)


def identity_view(batch):
    order = canonical_order(batch)
    v = batch.view("none")
    v.reports["logit"] = np.take_along_axis(v.reports["logit"], order, axis=1)
    return v


def identity_groups(name):
    """Identity index -> group label, in canonical order (for weight diagnostics)."""
    c = env_of(name).config
    n_honest = c.n_agents - c.n_colluders - c.n_biased
    g = ["colluder"] * c.n_colluders + ["biased"] * c.n_biased
    if c.n_sybil:
        g += ["honest"] * (n_honest - 1) + ["sybil_source"] + ["sybil_copy"] * c.n_sybil
    else:
        g += ["honest"] * n_honest
    return g


# ============================================================================= torch arms

def _torch():
    import torch
    return torch


def build_setnet(key_dim, width=64, layers=2, heads=4):
    torch = _torch()
    nn = torch.nn

    class PairBlock(nn.Module):
        def __init__(self, n_pair):
            super().__init__()
            self.heads, self.dh = heads, width // heads
            self.rel = nn.Sequential(nn.Linear(2 * width + n_pair, width), nn.GELU(),
                                     nn.Linear(width, heads + width))
            self.out = nn.Linear(width, width)
            self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

        def forward(self, h, pf, mask):
            B, N, W = h.shape
            z = self.ln1(h)
            pair = torch.cat([z[:, :, None].expand(B, N, N, W), z[:, None].expand(B, N, N, W), pf], -1)
            r = self.rel(pair)
            logits = r[..., :self.heads].masked_fill(~mask[:, None, :, None], -1e9)
            att = logits.softmax(dim=2)                                   # over j
            vals = r[..., self.heads:].reshape(B, N, N, self.heads, self.dh)
            msg = (att[..., None] * vals).sum(2).reshape(B, N, W)
            h = h + self.out(msg)
            return h + self.ff(self.ln2(h))

    class SetNet(nn.Module):
        def __init__(self):
            super().__init__()
            n_pair = 3 + (1 if key_dim else 0)
            self.embed = nn.Sequential(nn.Linear(1 + key_dim, width), nn.GELU(), nn.Linear(width, width))
            self.blocks = nn.ModuleList([PairBlock(n_pair) for _ in range(layers)])
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))

        def per_agent(self, x, keys, mask):
            """x: (B,N) centred log-odds (0 where absent); keys (B,N,k); mask (B,N) bool -> w, c (B,N)."""
            d = x[:, :, None] - x[:, None, :]
            pf = [d / 4, d.abs() / 4, torch.log(d.abs() + 1e-3) / 7]
            if key_dim:
                pf.append((keys[:, :, None] - keys[:, None, :]).abs().mean(-1))
            pm = (mask[:, :, None] & mask[:, None, :]).float()
            pf = torch.stack(pf, -1) * pm[..., None]
            h = self.embed(torch.cat([x[..., None] / 4, keys], -1))
            for blk in self.blocks:
                h = blk(h, pf, mask)
            o = self.head(h) * mask[..., None].float()
            return o[..., 0], o[..., 1]

        def forward(self, x, keys, mask):
            w, c = self.per_agent(x, keys, mask)
            return (w * x).sum(1) + c.sum(1)

    return SetNet()


class SetAggregator:
    """Learned aggregator (aggbench interface). Module lives on CPU between calls so controls work."""

    def __init__(self, seed, key_dim, steps, device, batch=512, lr=2e-3):
        torch = _torch()
        torch.manual_seed(seed)
        self.seed, self.steps, self.device, self.bs, self.lr = seed, steps, device, batch, lr
        self.module = build_setnet(key_dim)

    def _tensors(self, view):
        torch = _torch()
        l = view.reports["logit"]
        mask = np.isfinite(l)
        x = np.where(mask, l - view.public["prior_logit"], 0.0)
        keys = np.nan_to_num(view.keys, nan=0.0)
        f = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt)
        return f(x), f(keys), f(mask, torch.bool)

    def fit(self, view, truth):
        torch = _torch()
        x, k, m = self._tensors(view)
        y = torch.tensor(truth, dtype=torch.float32)
        net = self.module.to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.05)
        gen = torch.Generator().manual_seed(self.seed + 12345)
        n = len(y)
        for step in range(self.steps):
            idx = torch.randint(0, n, (self.bs,), generator=gen)
            out = net(x[idx].to(self.device), k[idx].to(self.device), m[idx].to(self.device))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y[idx].to(self.device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        self.module = net.cpu()
        self.final_loss = float(loss.detach())

    def _run(self, view, fn):
        torch = _torch()
        x, k, m = self._tensors(view)
        net = self.module.to(self.device)
        outs = []
        with torch.no_grad():
            for s in range(0, len(x), 4096):
                outs.append(fn(net, x[s:s + 4096].to(self.device), k[s:s + 4096].to(self.device),
                               m[s:s + 4096].to(self.device)))
        self.module = net.cpu()
        return outs

    def predict(self, view):
        outs = self._run(view, lambda net, x, k, m: net(x, k, m).double().cpu().numpy())
        return {"logit": view.public["prior_logit"] + np.concatenate(outs)}

    def weights(self, view):
        outs = self._run(view, lambda net, x, k, m: [t.cpu().numpy() for t in net.per_agent(x, k, m)])
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def _logsig(x):
    return -np.logaddexp(0.0, -x)


def _logloss(logit, y):
    return -(y * _logsig(logit) + (1 - y) * _logsig(-logit))


def _fit_scale(z, l0, y, lo=0.05, hi=20.0):
    """alpha minimising mean log loss of l0 + alpha z (log grid then golden section, as fit_extremize)."""
    loss = lambda la: _logloss(l0 + np.exp(la) * z, y).mean()
    grid = np.linspace(np.log(lo), np.log(hi), 61)
    j = int(np.argmin([loss(g) for g in grid]))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if loss(c) < loss(d):
            b = d
        else:
            a = c
    return float((a + b) / 2)


ETAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


class WealthMarket:
    """Log pool weighted by per-source wealth = exp(eta * cumulative own log score on train)."""

    def __init__(self, seed, n_slots):
        torch = _torch()
        torch.manual_seed(seed)
        self.module = torch.nn.Module()
        self.module.logw = torch.nn.Parameter(torch.zeros(n_slots, dtype=torch.float64))
        self.module.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def _agg(self, l, l0, logw, la):
        w = np.exp(logw - logw.max())
        w /= w.sum()
        return l0 + np.exp(la) * ((l - l0) * w).sum(1)

    def fit(self, view, truth):
        torch = _torch()
        l, l0, y = view.reports["logit"], view.public["prior_logit"], truth
        score = -_logloss(l, y[:, None])                   # each source's own log score per episode
        cut = int(0.8 * len(y))
        best = None
        for eta in ETAS:
            logw = eta * score[:cut].sum(0)
            w = np.exp(logw - logw.max()); w /= w.sum()
            z = ((l - l0) * w).sum(1)
            la = _fit_scale(z[:cut], l0, y[:cut])
            hold = _logloss(l0 + np.exp(la) * z[cut:], y[cut:]).mean()
            if best is None or hold < best[0]:
                best = (hold, eta)
        self.eta = best[1]
        logw = self.eta * score.sum(0)
        logw = logw - logw.max()
        w = np.exp(logw); w /= w.sum()
        la = _fit_scale(((l - l0) * w).sum(1), l0, y)
        with torch.no_grad():
            self.module.logw.copy_(torch.tensor(logw))
            self.module.log_alpha.copy_(torch.tensor([la]))

    def predict(self, view):
        p = self.module
        return {"logit": self._agg(view.reports["logit"], view.public["prior_logit"],
                                   p.logw.detach().numpy(), p.log_alpha.detach()[0].item())}

    def weight_vector(self):
        lw = self.module.logw.detach().numpy()
        w = np.exp(lw - lw.max())
        return w / w.sum() * float(np.exp(self.module.log_alpha.detach()[0].item()))


class LogPoolFit:
    """logit = l0 + sum_i w_i (l_i - l0) + sum_i c_i with per-source (w, c) fitted by L-BFGS on train."""

    def __init__(self, seed, n_slots):
        torch = _torch()
        torch.manual_seed(seed)
        self.module = torch.nn.Module()
        self.module.w = torch.nn.Parameter(torch.full((n_slots,), 1.0 / n_slots, dtype=torch.float64))
        self.module.c = torch.nn.Parameter(torch.zeros(n_slots, dtype=torch.float64))

    def fit(self, view, truth):
        torch = _torch()
        x = torch.tensor(view.reports["logit"] - view.public["prior_logit"])
        y = torch.tensor(truth, dtype=torch.float64)
        opt = torch.optim.LBFGS(self.module.parameters(), max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                (x * self.module.w).sum(1) + self.module.c.sum(), y)
            loss.backward()
            return loss
        opt.step(closure)

    def predict(self, view):
        w, c = self.module.w.detach().numpy(), self.module.c.detach().numpy()
        x = view.reports["logit"] - view.public["prior_logit"]
        return {"logit": view.public["prior_logit"] + (x * w).sum(1) + c.sum()}


# ============================================================================= stage: train (VM)

def train_configs():
    """train config -> list of eval env names it is evaluated on."""
    cfg = {"matched:" + e: [e] for e in EVAL_ENVS if e != "sybil0"}
    for c in CLEAN_ENVS:
        cfg["clean:" + c] = [e for e, cl in CLEAN_FOR.items() if cl == c]
    cfg["pooled:all7"] = list(EVAL_ENVS)
    return cfg


def stage_train(args):
    from aggbench.controls import run_with_controls
    torch = _torch()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = args.vis
    dev = {e: env_of(e).sample("dev", args.n_dev) for e in EVAL_ENVS}
    for e, b in dev.items():
        np.save(out / f"devseeds_{e}.npy", b.seeds)
    meta = {}
    for cfg, targets in train_configs().items():
        if args.only and cfg not in args.only:
            continue
        t0 = time.time()
        kind, src = cfg.split(":")
        if kind == "pooled":
            trs = [train_batch(e, args.n_train, slot=j + 1) for j, e in enumerate(EVAL_ENVS)]
            tv = concat_views([b.view(vis) for b in trs])
            ty = np.concatenate([b.truth for b in trs])
            steps = args.steps * 2
        else:
            tb = train_batch(src, args.n_train)
            tv, ty, steps = pad_view(tb.view(vis)), tb.truth, args.steps
        ev = concat_views([dev[e].view(vis) for e in targets])
        key_dim = tv.keys.shape[2]
        holder = {}

        def make(seed):
            agg = SetAggregator(seed, key_dim, steps, args.device)
            holder.setdefault("first", agg)
            return agg
        arms = run_with_controls(make, tv, ty, ev, seed=args.seed)
        trained = holder["first"]
        w, c = trained.weights(ev)
        save = {}
        for j, e in enumerate(targets):
            sl = slice(j * args.n_dev, (j + 1) * args.n_dev)
            n_ag = dev[e].reports["logit"].shape[1]
            for arm, f in arms.items():
                save[f"{e}|{arm}"] = f["logit"][sl]
            save[f"{e}|w"] = w[sl, :n_ag]
            save[f"{e}|c"] = c[sl, :n_ag]
        tag = f"set_{vis}_s{args.seed}_{cfg.replace(':', '-')}"
        np.savez_compressed(out / f"{tag}.npz", **save)
        meta[cfg] = {"targets": targets, "steps": steps, "n_train_episodes": int(len(ty)),
                     "final_minibatch_loss": trained.final_loss, "seconds": round(time.time() - t0, 1)}
        print(tag, meta[cfg], flush=True)
    (out / f"meta_set_{vis}_s{args.seed}.json").write_text(json.dumps(meta, indent=2))


def stage_market(args):
    from aggbench.controls import run_with_controls
    torch = _torch()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save, meta = {}, {}
    for e in EVAL_ENVS:
        tb, db = train_batch(e, args.n_train), env_of(e).sample("dev", args.n_dev)
        N = tb.reports["logit"].shape[1]
        variants = {"market_wealth:slot": (WealthMarket, tb.view("none"), db.view("none")),
                    "market_wealth:identity": (WealthMarket, identity_view(tb), identity_view(db)),
                    "logpool_fit:identity": (LogPoolFit, identity_view(tb), identity_view(db))}
        meta[e] = {}
        for name, (cls, tv, dv) in variants.items():
            holder = {}

            def make(seed, cls=cls):
                agg = cls(seed, N)
                holder.setdefault("first", agg)
                return agg
            arms = run_with_controls(make, tv, tb.truth, dv, seed=0)
            for arm, f in arms.items():
                save[f"{e}|{name}|{arm}"] = f["logit"]
            agg = holder["first"]
            if cls is WealthMarket:
                meta[e][name] = {"eta": agg.eta, "weights_times_alpha": agg.weight_vector().tolist()}
            else:
                meta[e][name] = {"w": agg.module.w.detach().numpy().tolist(),
                                 "c": agg.module.c.detach().numpy().tolist()}
        print(e, json.dumps(meta[e])[:300], flush=True)
    np.savez_compressed(out / "market.npz", **save)
    (out / "meta_market.json").write_text(json.dumps(meta, indent=2))


# ============================================================================= stage: assemble (local)

def stage_assemble(args):
    from aggbench.baselines import all_baselines
    from aggbench.metrics import paired_difference, summarize, per_episode
    from aggbench.results import make_result, split_record, write_result
    from aggbench.verify import oracle_check
    inp = Path(args.inp)
    outdir = Path(args.dest)
    market = dict(np.load(inp / "market.npz"))
    meta_market = json.loads((inp / "meta_market.json").read_text())
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    seeds = args.seeds
    summary = {}
    for e in EVAL_ENVS:
        env = env_of(e)
        tb, db = train_batch(e, args.n_train), env.sample("dev", args.n_dev)
        assert np.array_equal(np.load(inp / f"devseeds_{e}.npy"), db.seeds), "dev seeds differ from the VM's"
        ocheck = oracle_check(env)
        roles = db.oracle["role"]
        for vis in args.vis:
            base, alpha = all_baselines(tb, db, vis)
            fc = dict(base)
            controls, learned, diag = {}, [], {}
            for name in ("market_wealth:slot", "market_wealth:identity", "logpool_fit:identity"):
                for arm in ("trained", "control:untrained_init", "control:random_network"):
                    fc[name if arm == "trained" else f"{name}:{arm}"] = {"logit": market[f"{e}|{name}|{arm}"]}
                learned.append(name)
                controls[name] = {"random_network": f"{name}:control:random_network",
                                  "untrained_init": f"{name}:control:untrained_init"}
            for cfg, targets in train_configs().items():
                if e not in targets:
                    continue
                kind = cfg.split(":")[0]
                for s in seeds:
                    z = np.load(inp / f"set_{vis}_s{s}_{cfg.replace(':', '-')}.npz")
                    nm = f"set:{kind}:s{s}"
                    fc[nm] = {"logit": z[f"{e}|trained"]}
                    fc[f"{nm}:control:untrained_init"] = {"logit": z[f"{e}|control:untrained_init"]}
                    fc[f"{nm}:control:random_network"] = {"logit": z[f"{e}|control:random_network"]}
                    learned.append(nm)
                    controls[nm] = {"random_network": f"{nm}:control:random_network",
                                    "untrained_init": f"{nm}:control:untrained_init"}
                    if s == seeds[0]:
                        diag[f"set:{kind}"] = role_weights(z[f"{e}|w"], z[f"{e}|c"], roles, db)
                # follow-up: clean/matched training with the pooled arm's data size (7 x N_TRAIN, 2x steps)
                big = Path(args.big_inp) / f"set_{vis}_s0_{cfg.replace(':', '-')}.npz" if args.big_inp else None
                if big is not None and big.exists() and kind != "pooled":
                    z = np.load(big)
                    nm = f"set:{kind}-280k:s0"
                    fc[nm] = {"logit": z[f"{e}|trained"]}
                    fc[f"{nm}:control:untrained_init"] = {"logit": z[f"{e}|control:untrained_init"]}
                    fc[f"{nm}:control:random_network"] = {"logit": z[f"{e}|control:random_network"]}
                    learned.append(nm)
                    controls[nm] = {"random_network": f"{nm}:control:random_network",
                                    "untrained_init": f"{nm}:control:untrained_init"}
                    diag[f"set:{kind}-280k"] = role_weights(z[f"{e}|w"], z[f"{e}|c"], roles, db)
            methods = {k: summarize(f, db) for k, f in fc.items()}
            paired = {}
            for k in learned + ["extremized_log_pool", "log_pool"]:
                paired[k] = {"minus_extremized_log_pool": paired_difference(fc[k], fc["extremized_log_pool"], db),
                             "minus_oracle_clearing": paired_difference(fc[k], fc["oracle_clearing"], db)}
            seed_spread = {}
            for kind in ("matched", "clean", "pooled"):
                vals = [methods[f"set:{kind}:s{s}"]["regret"]["mean"] for s in seeds if f"set:{kind}:s{s}" in methods]
                if vals:
                    seed_spread[f"set:{kind}"] = {"seeds": seeds, "regret_means": vals,
                                                  "mean": float(np.mean(vals)), "min": min(vals), "max": max(vals)}
            preset, kw = EVAL_ENVS[e]
            run_id = f"{RUN_ID}-{e}-{vis}"
            extra = {
                "lane": "L5", "question": "detect and discount Sybil copies, colluders and a biased source "
                                          "from report content with provenance hidden or noisy",
                "eval_env_label": e, "preset": preset, "overrides": kw,
                "train_env_clean": CLEAN_FOR[e], "fitted_extremizing_alpha": alpha,
                "runner": "results/explore/l5-sybil-collusion-20261020/run_l5.py", "runner_sha256": runner_sha,
                "vm_run_commit": args.vm_commit, "vm_run_id": args.vm_run_id,
                "train_splits": {"matched_and_markets": split_record(tb, "train"),
                                 "pooled": "train seeds start at (j+1)*N_TRAIN for env j of EVAL_ENVS order",
                                 "n_train_per_env": args.n_train},
                "paired_regret": paired, "seed_spread": seed_spread,
                "set_weight_diagnostics_seed0": diag,
                "market_fits": meta_market[e], "identity_groups": identity_groups(e),
                "reference_only_arms": ["market_wealth:identity", "logpool_fit:identity"],
                "reference_only_note": "identity arms see persistent trader identities rebuilt from oracle roles; "
                                       "the view has none. They are references, not fair arms.",
                "note_sybil0": "for sybil0 the matched and clean training envs coincide (E2, 5 agents); "
                               "only set:clean is reported" if e == "sybil0" else "",
            }
            result = make_result(run_id=run_id, track="EXPLORATORY", env=env, visibility=vis,
                                 splits={"train": split_record(tb, "train"), "eval": split_record(db, "dev")},
                                 oracle_check=ocheck, methods=methods, learned_arms=learned, controls=controls,
                                 extra=extra)
            write_result(outdir / "json", result, name=f"{e}-{vis}.json")
            summary[f"{e}|{vis}"] = {"methods": {k: v["regret"] for k, v in methods.items()},
                                     "paired": paired, "seed_spread": seed_spread, "diag": diag,
                                     "kl": {k: v["kl"]["mean"] for k, v in methods.items()}}
            print(e, vis, "done", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))


def role_weights(w, c, roles, db):
    """Mean per-agent weight w_i and offset c_i by oracle role, and the total weight on the duplicated
    content (Sybil source + copies, or the colluding group). Diagnostics only (oracle used after predict)."""
    out = {}
    names = {0: "honest", 1: "colluder", 2: "biased", 3: "sybil_copy"}
    for k, nm in names.items():
        m = roles == k
        if m.any():
            out[f"mean_w_{nm}"] = float(w[m].mean())
            out[f"mean_c_{nm}"] = float(c[m].mean())
    cov = db.oracle["cov"]
    if (roles == 3).any():
        tot, single = [], []
        for e in range(len(roles)):
            syb = np.flatnonzero(roles[e] == 3)
            grp = [i for i in range(roles.shape[1]) if np.array_equal(cov[e, i], cov[e, syb[0]])]
            tot.append(w[e, grp].sum())
            single.append(w[e, roles[e] == 0].mean())
        out["mean_total_w_on_duplicated_group"] = float(np.mean(tot))
        out["mean_w_per_honest_agent"] = float(np.mean(single))
    if (roles == 1).any():
        out["mean_total_w_on_colluding_group"] = float(w[roles == 1].reshape(len(roles), -1).sum(1).mean())
    return out


def stage_tables(args):
    """Markdown tables from the result JSONs, written between the TABLES markers of README.md."""
    jdir = Path(args.dest) / "json"
    R = {p.stem: json.loads(p.read_text()) for p in sorted(jdir.glob("*.json"))}
    envs = list(EVAL_ENVS)
    ci = lambda m: f"{m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"
    lines = []
    cols = [("oracle_clearing", "oracle_clearing"), ("extremized_log_pool", "extremized"),
            ("market_wealth:slot", "market: slot wealth"), ("set:clean:s0", "set: clean-trained"),
            ("set:matched:s0", "set: matched-trained"), ("set:pooled:s0", "set: pooled-trained"),
            ("market_wealth:identity", "REF market: identity wealth"),
            ("logpool_fit:identity", "REF identity log pool")]
    for vis in args.vis:
        lines += [f"\n**Regret to Bayes (nats/episode), visibility `{vis}`**, mean [95% bootstrap CI] over 2000 dev "
                  "episodes; set arms are training seed 0." + (" The clean- and matched-trained columns here are the 40k runs, "
                  "which overfit; see the equal-data follow-up table." if vis == "noisy" else "") + "\n",
                  "| env | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for e in envs:
            m = R[f"{e}-{vis}"]["methods"]
            lines.append(f"| {e} | " + " | ".join(ci(m[k]["regret"]) if k in m else "= clean" for k, _ in cols) + " |")
    for vis in args.vis:
        lines += [f"\n**Set arms over 3 training seeds, visibility `{vis}`**: mean of seed means (min, max); "
                  "paired regret difference of seed 0 against oracle_clearing and extremized [95% CI].\n",
                  "| env | train | regret 3 seeds | minus oracle_clearing | minus extremized | "
                  "untrained ctl | random-net ctl |", "|---|---|---|---|---|---|---|"]
        for e in envs:
            r = R[f"{e}-{vis}"]
            for kind in ("clean", "matched", "pooled"):
                s = r["extra"]["seed_spread"].get(f"set:{kind}")
                if not s or (e == "sybil0" and kind == "matched"):
                    continue
                p = r["extra"]["paired_regret"][f"set:{kind}:s0"]
                ctl = r["controls"][f"set:{kind}:s0"]
                lines.append(f"| {e} | {kind} | {s['mean']:.3f} ({s['min']:.3f}, {s['max']:.3f}) | "
                             f"{ci(p['minus_oracle_clearing'])} | {ci(p['minus_extremized_log_pool'])} | "
                             f"{r['methods'][ctl['untrained_init']]['regret']['mean']:.2f} | "
                             f"{r['methods'][ctl['random_network']]['regret']['mean']:.2f} |")
    for vis in args.vis:
        rows = []
        for e in envs:
            r = R[f"{e}-{vis}"]
            for kind in ("clean-280k", "matched-280k", "pooled"):
                nm = f"set:{kind}:s0"
                if nm not in r["methods"] or (kind == "pooled" and "set:clean-280k:s0" not in r["methods"]):
                    continue
                p, ctl = r["extra"]["paired_regret"][nm], r["controls"][nm]
                rows.append(f"| {e} | {kind} | {ci(r['methods'][nm]['regret'])} | {ci(p['minus_oracle_clearing'])} | "
                            f"{ci(p['minus_extremized_log_pool'])} | "
                            f"{r['methods'][ctl['untrained_init']]['regret']['mean']:.2f} | "
                            f"{r['methods'][ctl['random_network']]['regret']['mean']:.2f} |")
        if rows:
            lines += [f"\n**Follow-up, equal data, visibility `{vis}`**: clean and matched training with the pooled "
                      "arm's data size (280,000 train episodes, 8000 steps; seed 0), next to the pooled arm (seed 0). "
                      "Regret [95% CI]; paired differences [95% CI]; control regrets.\n",
                      "| env | train | regret | minus oracle_clearing | minus extremized | untrained ctl | "
                      "random-net ctl |", "|---|---|---|---|---|---|---|"] + rows
    lines += ["\n**Market and reference arms**: paired regret difference against extremized [95% CI], "
              "visibility `none` (these arms ignore keys); controls are the regret of the untrained and random-"
              "weight versions.\n",
              "| env | slot wealth (eta) | REF identity wealth (eta) | REF identity log pool | ctl (untrained / random), "
              "slot | ctl, identity log pool |", "|---|---|---|---|---|---|"]
    for e in envs:
        r = R[f"{e}-none"]
        p, mf, c = r["extra"]["paired_regret"], r["extra"]["market_fits"], r["controls"]
        reg = lambda k: r["methods"][k]["regret"]["mean"]
        lines.append(
            f"| {e} | {ci(p['market_wealth:slot']['minus_extremized_log_pool'])} ({mf['market_wealth:slot']['eta']:g}) | "
            f"{ci(p['market_wealth:identity']['minus_extremized_log_pool'])} ({mf['market_wealth:identity']['eta']:g}) | "
            f"{ci(p['logpool_fit:identity']['minus_extremized_log_pool'])} | "
            f"{reg(c['market_wealth:slot']['untrained_init']):.3f} / {reg(c['market_wealth:slot']['random_network']):.3f} | "
            f"{reg(c['logpool_fit:identity']['untrained_init']):.3f} / {reg(c['logpool_fit:identity']['random_network']):.3f} |")
    lines += ["\n**Does the set network count duplicated content once?** Total weight w on the duplicated group "
              "(Sybil source + copies) divided by the mean weight per honest agent, seed 0 (1 = counted once, "
              "1 + n_sybil = counted per copy); and the identity log pool's fitted weights for comparison. "
              "Diagnostic only: roles come from the oracle after prediction.\n",
              "| env | vis | set clean | set matched | set pooled | REF identity log pool |", "|---|---|---|---|---|---|"]
    for e in ("sybil1", "sybil3", "sybil7"):  # mixed has no honest agent besides the Sybil source
        for vis in args.vis:
            r = R[f"{e}-{vis}"]
            d = r["extra"]["set_weight_diagnostics_seed0"]
            ratio = lambda k: (f"{d[k]['mean_total_w_on_duplicated_group'] / d[k]['mean_w_per_honest_agent']:.2f}"
                               if k in d and abs(d[k]['mean_w_per_honest_agent']) > 1e-6 else "n/a")
            lw = r["extra"]["market_fits"]["logpool_fit:identity"]["w"]
            g = r["extra"]["identity_groups"]
            dup = sum(w for w, gg in zip(lw, g) if gg in ("sybil_source", "sybil_copy"))
            hon = [w for w, gg in zip(lw, g) if gg == "honest"]
            ref = f"{dup / np.mean(hon):.2f}" if hon else f"dup total w {dup:.2f} (no other honest agent)"
            lines.append(f"| {e} | {vis} | {ratio('set:clean')} | {ratio('set:matched')} | {ratio('set:pooled')} | {ref} |")
    text = "\n".join(lines) + "\n"
    readme = Path(args.dest) / "README.md"
    body = readme.read_text()
    a, b = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
    body = body.split(a)[0] + a + "\n" + text + b + body.split(b)[1]
    readme.write_text(body)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True)
    t.add_argument("--vis", required=True, choices=["none", "noisy", "full"])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--n-train", type=int, default=N_TRAIN)
    t.add_argument("--n-dev", type=int, default=N_DEV)
    t.add_argument("--device", default="cpu")
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--only", nargs="*")
    m = sub.add_parser("market")
    m.add_argument("--out", required=True)
    m.add_argument("--n-train", type=int, default=N_TRAIN)
    m.add_argument("--n-dev", type=int, default=N_DEV)
    m.add_argument("--threads", type=int, default=1)
    a = sub.add_parser("assemble")
    a.add_argument("--inp", required=True)
    a.add_argument("--dest", default=str(HERE))
    a.add_argument("--vis", nargs="+", default=["none", "noisy"])
    a.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a.add_argument("--n-train", type=int, default=N_TRAIN)
    a.add_argument("--n-dev", type=int, default=N_DEV)
    a.add_argument("--vm-commit", default="")
    a.add_argument("--vm-run-id", default="")
    a.add_argument("--big-inp", default="", help="dir with the follow-up 280k-episode clean/matched runs")
    tb = sub.add_parser("tables")
    tb.add_argument("--dest", default=str(HERE))
    tb.add_argument("--vis", nargs="+", default=["none", "noisy"])
    args = ap.parse_args()
    {"train": stage_train, "market": stage_market, "assemble": stage_assemble,
     "tables": stage_tables}[args.stage](args)


if __name__ == "__main__":
    main()
