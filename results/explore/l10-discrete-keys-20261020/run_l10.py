"""EXPLORATORY lane L10 (stage 1): forecasters that also say what they know, with short discrete keys.

Each agent runs the same small network on its OWN inputs only (batch.agent_inputs("full"): its raw
observation and its own key = coverage bits and strengths of the pieces it saw) and sends
    forecast f_i   its own log-odds of Y, trained only as a forecaster (BCE against Y; the aggregator's
                   gradient does not reach it, so it stays a forecast and not a code)
    key k_i        a short description of what it knows, learned end to end with the aggregator.
Key arms:
    none           no key (forecast only)
    bits<L>        L binary symbols, straight-through Gumbel-sigmoid in training, hard threshold at eval
    bag<T>x<V>     T tokens from a vocabulary of V, straight-through Gumbel-softmax; the aggregator sees the
                   bag (token counts), so token order carries nothing
    cont<m>        unconstrained real vector in R^m (continuous reference)
    prov_bits      ground-truth provenance: the agent's 12 coverage bits (not learned)
    prov_full      ground-truth provenance with strengths: the env's full key [coverage, coverage x mu]
The aggregator (2 self-attention blocks + attention pooling) sees only the messages [f_i, k_i].
Training labels are truth on the train split (no distillation). Evaluation on dev only; test is never sampled.

Usage:
    python3 run_l10.py data --env E2 --cache DIR                  # sample train/dev once
    python3 run_l10.py train --env E2 --arms none bits8 --seeds 0 1 --cache DIR --out DIR
    python3 run_l10.py finalize --out DIR                          # local: JSON (aggbench.results), tables
"""
import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aggbench import make_env
from aggbench.messages import check_locality

HERE = Path(__file__).resolve().parent
RUN_ID = "l10-discrete-keys-20261020"
VIS = "full"
ENVS = ("E2", "E3-collude")
DEFAULT_ARMS = (["none"] + [f"bits{L}" for L in (1, 2, 3, 4, 6, 8, 12, 16, 24)]
                + [f"bag{T}x16" for T in (1, 2, 4, 8)] + [f"cont{m}" for m in (1, 2, 4, 8, 16)]
                + ["prov_bits", "prov_full"])
# Follow-up diagnostics (seed 0): other ways to train bits, and 3x longer training.
FOLLOWUP_ARMS = ([f"stbits{L}" for L in (2, 4, 8, 16)] + [f"wbits{L}" for L in (2, 4, 8, 16)]
                 + [f"{a}@12000" for a in ("none", "bits8", "wbits8", "cont4", "prov_bits", "prov_full")]
                 + [f"stbits{L}" for L in (24, 32, 64)] + [f"stbag{T}x16" for T in (2, 4, 8)] + ["stbits16@12000"])
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- inputs

class AgentInputs:
    """Each agent's own (obs, key) plus public constants; passed to run_with_controls in place of a View."""

    def __init__(self, kind, obs, keys, public):
        self.kind, self.obs, self.keys, self.public = kind, obs, keys, public


def local_features(obs, keys):
    """Agent i's features from its own obs and key only (elementwise along the agent axis)."""
    K = obs.shape[-1]
    x = np.where(np.isfinite(obs), obs, 0.0)
    mu = keys[..., K:]
    return np.concatenate([x, keys, mu * x], axis=-1).astype(np.float32)


def arm_steps(arm, default):
    """'<arm>@<steps>' trains for <steps> steps instead of the default."""
    return int(arm.split("@")[1]) if "@" in arm else default


def bit_mode(arm):
    """bits: Gumbel-sigmoid straight-through; stbits: deterministic straight-through (no noise);
    wbits: warm start, soft sigmoid bits for the first half of training, then deterministic straight-through."""
    a = arm.split("@")[0]
    return "st" if a.startswith("st") else "warm" if a.startswith("wbits") else "gumbel"


def parse_arm(arm):
    arm = arm.split("@")[0]
    if arm.startswith("stbits") or arm.startswith("wbits"):
        L = int(arm.lstrip("stwbi"))
        return ("bits", L, L)
    if arm == "none":
        return ("none", 0, 0)
    if arm.startswith("bits"):
        L = int(arm[4:])
        return ("bits", L, L)
    if arm.startswith("bag") or arm.startswith("stbag"):
        T, V = map(int, arm.split("bag")[1].split("x"))
        return ("bag", (T, V), V)
    if arm.startswith("cont"):
        m = int(arm[4:])
        return ("cont", m, m)
    if arm == "prov_bits":
        return ("prov_bits", 0, 12)
    if arm == "prov_full":
        return ("prov_full", 0, 24)
    raise ValueError(arm)


def capacity_bits(arm):
    kind, p, _ = parse_arm(arm)
    if kind == "bits":
        return float(p)
    if kind == "bag":
        T, V = p
        return math.log2(math.comb(V + T - 1, T))
    if kind == "prov_bits":
        return 12.0
    if kind == "none":
        return 0.0
    return None  # continuous


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
    """agent net (shared weights, per agent) -> (forecast, key) -> attention aggregator over messages."""

    def __init__(self, in_dim, arm, width=64, agent_width=128, seeds=4, K=12):
        super().__init__()
        self.kind, self.p, self.kdim = parse_arm(arm)
        self.mode = bit_mode(arm)
        self.progress = 1.0   # fraction of training done; set by fit (wbits: soft bits while < 0.5)
        self.K = K
        self.tau = 1.0
        self.agent = nn.Sequential(nn.Linear(in_dim, agent_width), nn.GELU(),
                                   nn.Linear(agent_width, agent_width), nn.GELU())
        self.f_head = nn.Linear(agent_width, 1)
        n_logits = {"bits": self.p if self.kind == "bits" else 0, "cont": self.p if self.kind == "cont" else 0,
                    "bag": self.p[0] * self.p[1] if self.kind == "bag" else 0}.get(self.kind, 0)
        self.k_head = nn.Linear(agent_width, n_logits) if n_logits else None
        self.embed = nn.Linear(1 + self.kdim, width)
        self.blocks = nn.ModuleList([Block(width), Block(width)])
        self.seeds = nn.Parameter(torch.randn(seeds, width) / np.sqrt(width))
        self.pool = nn.MultiheadAttention(width, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(seeds * width), nn.Linear(seeds * width, 128), nn.GELU(),
                                  nn.Linear(128, 1))

    def key(self, h, feats):
        """Key from agent i's own hidden state (or own provenance). Returns (key for aggregator, raw symbols)."""
        K = self.K
        if self.kind == "none":
            return feats[..., :0], None
        if self.kind == "prov_bits":
            k = feats[..., K:2 * K]
            return 2 * k - 1, k
        if self.kind == "prov_full":
            return feats[..., K:3 * K] * torch.tensor([1.0] * K + [4.0] * K, device=feats.device), None
        z = self.k_head(h)
        if self.kind == "cont":
            return z, z
        if self.kind == "bits":
            if self.training and self.mode != "gumbel":
                s = torch.sigmoid(z)
                if self.mode == "warm" and self.progress < 0.5:
                    b = s
                else:
                    b = (z > 0).float() + s - s.detach()
            elif self.training:
                u = torch.rand_like(z).clamp(1e-6, 1 - 1e-6)
                soft = torch.sigmoid((z + torch.log(u) - torch.log1p(-u)) / self.tau)
                hard = (soft > 0.5).float()
                b = hard - soft.detach() + soft
            else:
                b = (z > 0).float()
            return 2 * b - 1, b
        T, V = self.p
        z = z.view(*z.shape[:-1], T, V)
        if self.training and self.mode == "st":
            soft = torch.softmax(z, -1)
            oh = F.one_hot(z.argmax(-1), V).float() + soft - soft.detach()
        elif self.training:
            oh = F.gumbel_softmax(z, tau=self.tau, hard=True, dim=-1)
        else:
            oh = F.one_hot(z.argmax(-1), V).float()
        return oh.sum(-2), oh.argmax(-1)

    def encode(self, feats):
        """(n, agents, in_dim) -> forecast (n, agents), key (n, agents, kdim), symbols. Row-wise; no mixing."""
        h = self.agent(feats)
        f = self.f_head(h)[..., 0]
        k, sym = self.key(h, feats)
        return f, k, sym

    def aggregate(self, f, k):
        z = self.embed(torch.cat([f.detach().unsqueeze(-1) / 4, k], -1))
        for b in self.blocks:
            z = b(z)
        q = self.seeds.unsqueeze(0).expand(z.shape[0], -1, -1)
        pooled = self.pool(q, z, z, need_weights=False)[0]
        return self.head(pooled.flatten(1))[:, 0]

    def forward(self, feats):
        f, k, _ = self.encode(feats)
        return self.aggregate(f, k), f


class KeyedForecasters:
    """.module / .fit(inputs, truth) / .predict(inputs), as aggbench.controls expects."""

    def __init__(self, seed, in_dim, arm, steps=4000, batch=512, lr=2e-3, tau=(1.0, 0.25), log=None):
        torch.manual_seed(seed)
        steps = arm_steps(arm, steps)
        self.arm, self.steps, self.bs, self.lr, self.tau, self.log = arm, steps, batch, lr, tau, log
        self.module = Pipeline(in_dim, arm).to(DEV)
        self.fit_info = {}

    def _device(self):
        return next(self.module.parameters()).device

    def _feats(self, inp):
        return torch.tensor(local_features(inp.obs, inp.keys), device=self._device())

    def _loss(self, X, y, l0):
        agg, f = self.module(X)
        la = F.binary_cross_entropy_with_logits(l0 + agg, y)
        lf = F.binary_cross_entropy_with_logits(l0 + f, y.unsqueeze(1).expand_as(f))
        return la, lf

    def fit(self, inp, truth):
        X = self._feats(inp)
        Y = torch.tensor(truth, dtype=torch.float32, device=DEV)
        l0 = float(inp.public["prior_logit"])
        n_val = min(10000, len(X) // 10)        # held-out slice of TRAIN for checkpoint choice
        Xv, Yv, Xt, Yt = X[:n_val], Y[:n_val], X[n_val:], Y[n_val:]
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.05)
        g = torch.Generator(device=DEV).manual_seed(1234)
        best, best_state, t0, hist = float("inf"), None, time.time(), []
        t_hi, t_lo = self.tau
        for step in range(1, self.steps + 1):
            self.module.tau = t_hi * (t_lo / t_hi) ** (step / self.steps)
            self.module.progress = step / self.steps
            idx = torch.randint(0, len(Xt), (self.bs,), generator=g, device=DEV)
            self.module.train()
            la, lf = self._loss(Xt[idx], Yt[idx], l0)
            opt.zero_grad()
            (la + lf).backward()
            nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % 250 == 0 or step == self.steps:
                self.module.eval()
                with torch.no_grad():
                    vl = [self._loss(Xv[j:j + 5000], Yv[j:j + 5000], l0) for j in range(0, n_val, 5000)]
                    va = float(np.mean([a.item() for a, _ in vl]))
                    vf = float(np.mean([b.item() for _, b in vl]))
                hist.append((step, round(va, 5), round(vf, 5)))
                hard_phase = not (self.module.mode == "warm" and self.module.progress <= 0.5)
                if va < best and hard_phase:  # checkpoint chosen in eval mode (hard keys), as at test time
                    best = va
                    best_state = {k: v.clone() for k, v in self.module.state_dict().items()}
                if self.log and step % 1000 == 0:
                    self.log(f"  {self.arm} step {step} tau {self.module.tau:.2f} val agg {va:.4f} "
                             f"own-forecast {vf:.4f} ({time.time() - t0:.0f}s)")
        self.module.load_state_dict(best_state)
        self.module.eval()
        self.module.cpu()   # aggbench.controls.random_weights draws its noise on CPU; inference runs on CPU
        self.fit_info = {"best_val_loss": best, "val_history": hist, "seconds": round(time.time() - t0, 1)}

    def _run(self, inp):
        self.module.eval()
        X = self._feats(inp)
        outs, fs, ks, syms = [], [], [], []
        with torch.no_grad():
            for j in range(0, len(X), 4096):
                f, k, sym = self.module.encode(X[j:j + 4096])
                outs.append(self.module.aggregate(f, k))
                fs.append(f)
                ks.append(k)
                syms.append(sym if sym is not None else k)
        cat = lambda xs: torch.cat(xs).double().cpu().numpy()
        return cat(outs), cat(fs), cat(ks), cat(syms)

    def predict(self, inp):
        out = self._run(inp)[0]
        return {"logit": float(inp.public["prior_logit"]) + out}

    def messages(self, inp):
        """Dev messages for analysis: own forecasts (log-odds), keys as the aggregator sees them, raw symbols."""
        _, f, k, sym = self._run(inp)
        return float(inp.public["prior_logit"]) + f, k, sym

    def encode_numpy(self, obs, keys):
        self.module.eval()
        with torch.no_grad():
            f, k, _ = self.module.encode(torch.tensor(local_features(obs, keys), device=self._device()))
            return torch.cat([f.unsqueeze(-1), k], -1).double().cpu().numpy()


# ----------------------------------------------------------------------------- interpretability

def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def plugin_mi(a, b):
    """Plug-in mutual information (bits) between two discrete integer arrays."""
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(joint, (ai, bi), 1)
    joint /= joint.sum()
    return entropy_bits(joint.sum(1)) + entropy_bits(joint.sum(0)) - entropy_bits(joint.ravel())


def unit_matrix(arm, k, sym, seen):
    """Per-unit MI (bits) with each piece's seen bit: units are key bits, vocab tokens (present in the bag
    or not), or continuous dims cut into 8 quantile bins. Returns (units, K) and the unit labels."""
    kind = parse_arm(arm)[0]
    if kind in ("bits", "prov_bits"):
        units = sym.reshape(-1, sym.shape[-1]).astype(int)
    elif kind == "bag":
        units = (k.reshape(-1, k.shape[-1]) > 0).astype(int)
    else:
        kk = k.reshape(-1, k.shape[-1])
        units = np.stack([np.searchsorted(np.quantile(c, np.linspace(0, 1, 9)[1:-1]), c) for c in kk.T], 1)
    S = seen.reshape(-1, seen.shape[-1]).astype(int)
    return np.array([[plugin_mi(units[:, u], S[:, j]) for j in range(S.shape[1])] for u in range(units.shape[1])])


def probe_mi(k, target, n_classes, seed=0, steps=600):
    """Held-out lower bound on I(key; target_j) for each piece j, in bits: H(target_j) - CE of an MLP probe
    trained on episodes of the first half, evaluated on the second half (a variational bound)."""
    n = k.shape[0]
    half = n // 2
    X = torch.tensor(k.reshape(n, -1, k.shape[-1]) if k.ndim == 3 else k, dtype=torch.float32)
    X = X.reshape(-1, X.shape[-1])
    T = torch.tensor(target.reshape(-1, target.shape[-1]), dtype=torch.long)
    n_agents = k.shape[1]
    tr, te = slice(0, half * n_agents), slice(half * n_agents, None)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd
    Kp = T.shape[1]
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                        nn.Linear(128, Kp * n_classes))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    Xtr, Ttr = X[tr], T[tr]
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (1024,), generator=g)
        loss = F.cross_entropy(net(Xtr[idx]).view(-1, Kp, n_classes).transpose(1, 2), Ttr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ce = F.cross_entropy(net(X[te]).view(-1, Kp, n_classes).transpose(1, 2), T[te], reduction="none")
    ce_bits = ce.mean(0).numpy() / np.log(2)
    Tte = T[te].numpy()
    H = np.array([entropy_bits(np.bincount(Tte[:, j], minlength=n_classes) / len(Tte)) for j in range(Kp)])
    return H - ce_bits, H


def interpretability(arm, k, sym, dev):
    """Does the key say which evidence pieces the agent saw?"""
    seen = dev.oracle["cov"].astype(int)                    # (n, agents, K) true provenance
    lam_pos = (dev.oracle["mu"] * dev.oracle["x"] > 0)[:, None, :]
    signed = np.where(seen == 1, np.where(lam_pos, 1, 2), 0)  # 0 unseen, 1 seen & LLR>0, 2 seen & LLR<0
    out = {"capacity_bits": capacity_bits(arm), "key_dim": int(k.shape[-1])}
    if k.shape[-1] == 0:
        out.update({"probe_seen_bits_total": 0.0, "probe_signed_bits_total": 0.0})
        return out
    flat = (k.reshape(-1, k.shape[-1]) if parse_arm(arm)[0] in ("cont", "prov_full")
            else np.round(k.reshape(-1, k.shape[-1])).astype(int))
    _, inv = np.unique(flat, axis=0, return_inverse=True)
    out["distinct_keys_used"] = int(inv.max() + 1)
    sym_ids = inv.reshape(-1)
    out["key_entropy_bits"] = (entropy_bits(np.bincount(sym_ids) / len(sym_ids))
                               if parse_arm(arm)[0] not in ("cont", "prov_full") else None)
    mi_s, H_s = probe_mi(k, seen, 2)
    mi_v, H_v = probe_mi(k, signed, 3)
    rng = np.random.default_rng(0)
    shuf = k[rng.permutation(len(k))]
    mi_null, _ = probe_mi(shuf, seen, 2)
    out.update({
        "H_seen_bits_per_piece": H_s.tolist(), "probe_seen_bits": mi_s.tolist(),
        "probe_seen_bits_total": float(mi_s.sum()), "H_seen_total": float(H_s.sum()),
        "probe_seen_bits_total_shuffled_key": float(mi_null.sum()),
        "probe_signed_bits_total": float(mi_v.sum()), "H_signed_total": float(H_v.sum()),
        "probe_value_bits_total": float(mi_v.sum() - mi_s.sum()),
    })
    M = unit_matrix(arm, k, sym, seen)
    norm = M / H_s.mean()
    best_per_piece = norm.max(0)
    out.update({
        "unit_mi_seen_matrix_bits": np.round(M, 4).tolist(),
        "alignment_mean_best_unit": float(best_per_piece.mean()),
        "pieces_with_a_dedicated_unit": int((best_per_piece > 0.5).sum()),
        "units_dedicated_to_one_piece": int((norm.max(1) > 0.5).sum()),
    })
    return out


def key_content(arm, key, dv):
    """What a discrete key says (plug-in MI in bits, minus the same estimate with keys shuffled across rows):
    n_seen = number of pieces the agent saw; strength = sum of mu_k^2 over its pieces (8 quantile bins);
    sign / size = sign and |.| (8 quantile bins) of its exact local log-odds; Y = the outcome."""
    if key.size == 0 or parse_arm(arm)[0] in ("cont", "prov_full", "none"):
        return None
    flat = np.round(key.reshape(-1, key.shape[-1])).astype(int)
    _, sym = np.unique(flat, axis=0, return_inverse=True)
    sym = sym.reshape(-1)
    cov = dv.oracle["cov"]
    local = dv.local["logit"] - dv.public["prior_logit"]

    def q8(v):
        return np.searchsorted(np.quantile(v, np.linspace(0, 1, 9)[1:-1]), v)
    targets = {"n_seen": cov.sum(-1).ravel(),
               "strength": q8((cov * dv.oracle["mu"][:, None, :] ** 2).sum(-1).ravel()),
               "sign_local": (local > 0).astype(int).ravel(), "size_local": q8(np.abs(local).ravel()),
               "Y": np.repeat(dv.truth, cov.shape[1])}
    rng = np.random.default_rng(0)
    shuf = sym[rng.permutation(len(sym))]
    out = {}
    for name, t in targets.items():
        out[name] = round(plugin_mi(sym, t) - plugin_mi(shuf, t), 4)
        out[f"H_{name}"] = round(entropy_bits(np.bincount(t) / len(t)), 4)
    return out


# ----------------------------------------------------------------------------- data

def cached_sample(env, split, n, cache):
    path = Path(cache) / f"{env.name}_{split}_{n}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    b = env.sample(split, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(b))
    tmp.rename(path)
    return b


def to_inputs(batch):
    obs, keys, public = batch.agent_inputs(VIS)
    return AgentInputs(batch.kind, obs, keys, public)


# ----------------------------------------------------------------------------- commands

def cmd_data(args):
    env = make_env(args.env)
    t0 = time.time()
    cached_sample(env, "train", args.n_train, args.cache)
    cached_sample(env, "dev", args.n_dev, args.cache)
    print(f"[{args.env}] data ready in {time.time() - t0:.0f}s", flush=True)


def cmd_train(args):
    from aggbench.controls import run_with_controls
    torch.set_num_threads(args.threads)
    out = Path(args.out) / "arms"
    out.mkdir(parents=True, exist_ok=True)
    logf = open(Path(args.out) / f"log_{args.env}_{args.tag}.txt", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    env = make_env(args.env)
    tr = cached_sample(env, "train", args.n_train, args.cache)
    dv = cached_sample(env, "dev", args.n_dev, args.cache)
    tr_in, dv_in = to_inputs(tr), to_inputs(dv)
    in_dim = local_features(dv_in.obs[:1], dv_in.keys[:1]).shape[-1]
    for seed in args.seeds:
        for arm in args.arms:
            stem = f"{args.env}__{arm}__s{seed}"
            if (out / f"{stem}.json").exists():
                continue
            t0 = time.time()
            built = {}

            def make(s, arm=arm):
                agg = KeyedForecasters(s, in_dim, arm, steps=args.steps, log=log)
                built.setdefault("first", agg)
                return agg

            arms = run_with_controls(make, tr_in, tr.truth, dv_in, seed=seed)
            trained = built["first"]
            sl = slice(0, 500)
            try:
                ok = check_locality(trained.encode_numpy, dv_in.obs[sl], dv_in.keys[sl], trials=5)
                loc = {"passed": bool(ok), "episodes": 500, "trials": 5, "atol": 1e-9,
                       "encoder": "trained agent net; message = [forecast, key]"}
            except AssertionError as err:
                loc = {"passed": False, "error": str(err)}
            f_own, k, sym = trained.messages(dv_in)
            interp = interpretability(arm, k, sym, dv) if seed == args.seeds[0] else None
            np.savez_compressed(out / f"{stem}.npz", **{f"logit_{a.replace(':', '_')}": v["logit"]
                                                        for a, v in arms.items()},
                                own_forecast=f_own.astype(np.float32),
                                key=(k if interp is not None else k[:0]).astype(np.float32))
            rec = {"env": args.env, "arm": arm, "seed": seed, "locality": loc, "fit_info": trained.fit_info,
                   "interpretability": interp, "seconds": round(time.time() - t0, 1), "device": DEV,
                   "n_params": sum(p.numel() for p in trained.module.parameters())}
            (out / f"{stem}.json").write_text(json.dumps(rec) + "\n")
            ll = F.binary_cross_entropy_with_logits(torch.tensor(arms["trained"]["logit"]),
                                                    torch.tensor(dv.truth, dtype=torch.float64)).item()
            llb = F.binary_cross_entropy_with_logits(torch.tensor(dv.bayes["logit"]),
                                                     torch.tensor(dv.truth, dtype=torch.float64)).item()
            log(f"[{args.env}] {arm} s{seed}: dev regret {ll - llb:.4f} locality {loc['passed']} "
                f"({time.time() - t0:.0f}s)"
                + (f" probe-seen {interp['probe_seen_bits_total']:.2f} bits" if interp else ""))


def fmt(r, d=3):
    return f"{r['mean']:.{d}f} [{r['ci_low']:.{d}f}, {r['ci_high']:.{d}f}]"


def cmd_finalize(args):
    from aggbench.baselines import all_baselines
    from aggbench.metrics import paired_difference, per_episode, summarize
    from aggbench.results import make_result, split_record, write_result
    from aggbench.verify import oracle_check
    torch.set_num_threads(2)
    outdir = Path(args.out)
    recs = [json.loads(p.read_text()) for p in sorted((outdir / "arms").glob("*.json"))]
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tables = {}
    for env_name in ENVS:
        env = make_env(env_name)
        er = [r for r in recs if r["env"] == env_name]
        if not er:
            continue
        tr = env.sample("train", args.n_train_baselines)   # only for the extremizing alpha of the baseline
        dv = env.sample("dev", args.n_dev)
        forecasts, alpha = all_baselines(tr, dv, VIS)
        learned, controls, extra_arms, paired = [], {}, {}, {}
        for r in er:
            name = f"{r['arm']}:s{r['seed']}"
            z = np.load(outdir / "arms" / f"{env_name}__{r['arm']}__s{r['seed']}.npz")
            for a in ("trained", "control_untrained_init", "control_random_network"):
                forecasts[f"{name}:{a.replace('control_', 'control:')}"] = {"logit": z[f"logit_{a}"].astype(float)}
            learned.append(f"{name}:trained")
            controls[f"{name}:trained"] = {"random_network": f"{name}:control:random_network",
                                           "untrained_init": f"{name}:control:untrained_init"}
            tf = forecasts[f"{name}:trained"]
            paired[name] = {f"trained_minus_{b}": paired_difference(tf, forecasts[b], dv)
                            for b in ("oracle_clearing", "extremized_log_pool")}
            own = z["own_forecast"].astype(float)
            extra_arms[name] = {k: r[k] for k in ("locality", "fit_info", "interpretability", "seconds", "device",
                                                   "n_params")}
            extra_arms[name]["key_content_bits"] = key_content(r["arm"], z["key"], dv)  # None if key not saved
            extra_arms[name]["own_forecast_kl_to_local_posterior"] = float(np.mean(
                [per_episode({"logit": own[:, i]}, _local_batch(dv, i))["kl"].mean() for i in range(own.shape[1])]))
            extra_arms[name]["own_forecast_kl_to_global_bayes"] = float(np.mean(
                [per_episode({"logit": own[:, i]}, dv)["kl"].mean() for i in range(own.shape[1])]))
        for name in paired:  # does the key help? key arm minus the no-key arm, same seed, same episodes
            seed = name.split(":")[1]
            if f"none:{seed}:trained" in forecasts:
                paired[name]["trained_minus_none"] = paired_difference(
                    forecasts[f"{name}:trained"], forecasts[f"none:{seed}:trained"], dv)
        methods = {k: summarize(f, dv) for k, f in forecasts.items()}
        extra = {
            "lane": "L10", "question": ("forecasters that also output a short discrete key describing what they know, "
                                        "learned end to end with an attention aggregator; regret vs key length"),
            "runner": str(Path(__file__).relative_to(HERE.parents[2])), "runner_sha256": runner_sha,
            "runner_commit": args.runner_commit,
            "arms_definition": {"none": "forecast only", "bits<L>": "L straight-through Gumbel-sigmoid bits",
                                "bag<T>x<V>": "bag of T tokens from vocab V (straight-through Gumbel-softmax)",
                                "cont<m>": "real vector R^m", "prov_bits": "true coverage bits (12)",
                                "prov_full": "env key 'full': coverage bits and coverage x mu (24 floats)"},
            "message_definition": ("per agent: forecast (own log-odds, trained only by its own BCE; gradient from "
                                   "the aggregator stopped) + key"),
            "training": {"labels": "truth (no distillation to bayes)", "steps": args.steps, "batch": 512, "lr": 2e-3,
                         "optimizer": "AdamW wd 1e-4, one-cycle", "gumbel_tau": "1.0 -> 0.25 geometric",
                         "checkpoint_choice": "best aggregator loss (eval mode, hard keys) on first 10k train episodes held out",
                         "n_train": args.n_train},
            "inputs": "batch.agent_inputs('full'): own obs + own key (coverage, coverage x mu); features add mu x",
            "e3_biased": "excluded (lane uses E2 and E3-collude)",
            "interpretability_method": ("probe_seen_bits: held-out MLP-probe lower bound on sum_k I(key; seen_k) in bits "
                                        "(probe trained on first half of dev episodes, scored on second half); "
                                        "probe_signed: same for the 3-valued (unseen, seen&LLR>0, seen&LLR<0); "
                                        "unit matrix: plug-in MI per key unit (bit / vocab token present / continuous dim "
                                        "in 8 quantile bins) and piece; alignment = max over units of MI / H(seen), "
                                        "averaged over pieces"),
            "baseline_extremize_alpha": alpha, "baseline_alpha_train_episodes": args.n_train_baselines,
            "arms": extra_arms, "paired_differences": paired,
            "compute": "spot L4 via work/run_on_l4.sh (training); finalize on CPU",
        }
        result = make_result(run_id=f"{RUN_ID}-{env_name}", track="EXPLORATORY", env=env, visibility=VIS,
                             splits={"train": {"split": "train", "first_seed": 0, "n": args.n_train},
                                     "eval": split_record(dv, "dev")},
                             oracle_check=oracle_check(env), methods=methods, learned_arms=learned,
                             controls=controls, extra=extra)
        p = write_result(outdir, result, name=f"result_{env_name}.json")
        print("wrote", p, flush=True)
        tables[env_name] = (methods, paired, extra_arms)
    (outdir / "tables.md").write_text(render_tables(tables))
    print((outdir / "tables.md").read_text())


def _local_batch(dv, i):
    """A shallow stand-in batch whose 'bayes' is agent i's exact local posterior (to score own forecasts)."""
    class B:
        pass
    b = B()
    b.kind, b.truth, b.bayes = "bernoulli", dv.truth, {"logit": dv.local["logit"][:, i]}
    return b


def render_tables(tables):
    lines = []
    for env_name, (methods, paired, arms) in tables.items():
        lines += [f"### {env_name}", "",
                  "| arm | capacity (bits) | regret to Bayes, seed 0 [95% CI] | seed 1 | minus no-key arm [95% CI] "
                  "| minus oracle_clearing [95% CI] | random-net control | key entropy (bits) "
                  "| probe I(key; seen) bits | probe I(key; signed evidence) bits | alignment | locality |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in ("bayes", "oracle_clearing", "extremized_log_pool", "log_pool"):
            lines.append(f"| {b} (baseline) | - | {fmt(methods[b]['regret'])} |" + " - |" * 9)
        for arm in DEFAULT_ARMS + FOLLOWUP_ARMS:
            n0 = f"{arm}:s0"
            if n0 not in arms:
                continue
            m0 = methods[f"{n0}:trained"]["regret"]
            m1 = methods.get(f"{arm}:s1:trained", {}).get("regret")
            d = paired[n0]["trained_minus_oracle_clearing"]
            rn = methods[f"{n0}:control:random_network"]["regret"]["mean"]
            it = arms[n0]["interpretability"] or {}
            cap = capacity_bits(arm)
            dn = paired[n0].get("trained_minus_none")
            ent = it.get("key_entropy_bits")
            cells = [arm, "-" if cap is None else f"{cap:.1f}", fmt(m0),
                     "-" if m1 is None else f"{m1['mean']:.3f}", "-" if arm == "none" or dn is None else fmt(dn, 4),
                     fmt(d), f"{rn:.3f}", "-" if ent is None else f"{ent:.2f}",
                     f"{it.get('probe_seen_bits_total', 0):.2f}", f"{it.get('probe_signed_bits_total', 0):.2f}",
                     "-" if "alignment_mean_best_unit" not in it else f"{it['alignment_mean_best_unit']:.2f}",
                     "pass" if arms[n0]["locality"]["passed"] else "FAIL"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "train", "finalize"])
    ap.add_argument("--env", choices=ENVS)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-train-baselines", type=int, default=20000)
    ap.add_argument("--n-dev", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--runner-commit", default="unknown")
    ap.add_argument("--cache", default="/tmp/l10-cache")
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    {"data": cmd_data, "train": cmd_train, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
