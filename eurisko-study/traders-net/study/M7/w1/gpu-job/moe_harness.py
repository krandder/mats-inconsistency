"""M7 shared Stage-1 harness: a small MoE transformer on the S0 streams, trained prequentially.

M7 worker 1 (traders-net study, 2026-09-28). Siblings: import this file and pass your own
Policy subclass; everything else (stream, model, optimiser, metrics, CSV schema) stays shared, so
all arms are comparable.

Model. The S0 substrate Net (d=64, H=4, L=2, one token per question, W=32 window) with each
dense MLP replaced by an MoE layer: E experts, each a ReLU MLP d -> h_e -> d (h_e = 64), top-k=2.
Active FLOPs per token are fixed by design (k * expert FLOPs) for every token-choice arm; expert
choice has the same *average* (capacity C = N*k/E per expert).

Gate convention (fixed by Stage 0, check_stage0.py): router logits z = W_r x (no bias in z);
selection S = top-k(z + b_sel) where b_sel is a non-gradient selection bias (DeepSeek's
loss-free bias lives there, zero for other arms); gates g = softmax(z_S) renormalised over S
(T1 Lemma 7 convention (i)). Then -dL/dz_i = g_i (r_i - R) for i in S, 0 outside, with
r_i = <E_i(x), -dL/dh> and R = sum_S g_i r_i.

Policy hooks (subclass Policy):
  select(layer, z, ctx) -> combine matrix G [N,E] (zeros off the routed set). Default: top-k
      over z + b_sel with softmax renormalised over S.
  aux_loss(layer, z, G, ctx) -> scalar tensor added to the training loss (default 0).
  after_step(layer, rec, ctx): called after each optimiser step with rec = dict(z, G, load,
      rev_all, train=True). rev_all [N,E] is r_i for EVERY expert (all experts are computed
      densely here); a censored arm must only read rev_all where G > 0 (or its own sampled
      shadow expert). Using rev_all elsewhere is full-information credit: say so.
  ctx: dict(fam [N] long, qid [N] long, step int, train bool, pos [N] long). fam is the
      question's nominal family fam0 (static metadata, label-free); never a label.

Prequential: S0 prequential.run_prequential (test-then-train, asserted). Routing at test time
uses the current policy state; only the train pass calls after_step.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUB = os.environ.get("M7_SUBSTRATE", "/home/agent/workspaces/eurisko-study/traders-net/study/S0/substrate")
if SUB not in sys.path:
    sys.path.insert(0, SUB)
from model import LN  # noqa: E402
import prequential  # noqa: E402
import streams_ext  # noqa: E402

FAMILIES = streams_ext.FAMILIES


# ------------------------------------------------------------------ policies
class Policy:
    name = "none"

    def __init__(self, E, k=2, **kw):
        self.E, self.k = E, k
        self.kw = kw

    def init_layer(self, layer):
        pass

    def b_sel(self, layer):
        return layer.b_sel

    def select(self, layer, z, ctx):
        zs = z + self.b_sel(layer)
        idx = zs.topk(self.k, dim=-1).indices                       # [N,k]
        g = torch.softmax(z.gather(-1, idx), dim=-1)                 # renormalised over S
        return torch.zeros_like(z).scatter(-1, idx, g)

    def aux_loss(self, layer, z, G, ctx):
        return z.new_zeros(())

    def after_step(self, layer, rec, ctx):
        pass


class SwitchAux(Policy):
    """Switch-Transformer auxiliary load loss alpha * E * sum_i f_i P_i (arXiv 2101.03961)."""
    name = "switch"

    def aux_loss(self, layer, z, G, ctx):
        a = self.kw.get("alpha", 1e-2)
        f = (G > 0).float().mean(0) / self.k                       # fraction of assignments
        P = torch.softmax(z, -1).mean(0)
        return a * self.E * (f.detach() * P).sum()


class DeepSeekBias(Policy):
    """Auxiliary-loss-free balancing (arXiv 2408.15664): b_i += u * sign(mean load - load_i),
    b enters only the top-k selection."""
    name = "deepseek"

    def after_step(self, layer, rec, ctx):
        u = self.kw.get("u", 1e-3)
        load = rec["load"]
        layer.b_sel += u * torch.sign(load.mean() - load)


class ExpertChoice(Policy):
    """Expert choice routing (arXiv 2202.09368): each expert takes its top-C tokens by
    softmax affinity, C = N*k/E; gate = the affinity."""
    name = "expertchoice"

    def select(self, layer, z, ctx):
        N = z.shape[0]
        C = max(1, int(round(N * self.k / self.E)))
        s = torch.softmax(z, dim=-1)                                # [N,E]
        idx = s.topk(min(C, N), dim=0).indices                      # [C,E]
        G = torch.zeros_like(s)
        G.scatter_(0, idx, s.gather(0, idx))
        return G


POLICIES = {"none": Policy, "switch": SwitchAux, "deepseek": DeepSeekBias,
            "expertchoice": ExpertChoice}


# ------------------------------------------------------------------ model
class MoE(nn.Module):
    def __init__(self, d, E, h, policy):
        super().__init__()
        self.E, self.h = E, h
        self.router = nn.Linear(d, E, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.w1 = nn.Parameter(torch.randn(E, d, h) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(E, h))
        self.w2 = nn.Parameter(torch.randn(E, h, d) / math.sqrt(h) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(E, d))
        self.register_buffer("b_sel", torch.zeros(E))
        self.policy = policy
        self.last = None
        policy.init_layer(self)

    def experts(self, x):                                            # x [N,d] -> [N,E,d]
        a = F.relu(torch.einsum("nd,edh->neh", x, self.w1) + self.b1)
        return torch.einsum("neh,ehd->ned", a, self.w2) + self.b2

    def forward(self, x, ctx):
        shp = x.shape
        x = x.reshape(-1, shp[-1])
        z = self.router(x)
        G = self.policy.select(self, z, ctx)
        Eo = self.experts(x)
        out = torch.einsum("ne,ned->nd", G, Eo)
        if ctx.get("train") and out.requires_grad:
            out.retain_grad()
        self.last = dict(z=z, G=G, Eo=Eo, out=out)
        return out.reshape(shp), self.policy.aux_loss(self, z, G, ctx)


class MoENet(nn.Module):
    def __init__(self, Q, W, E, policy_cls, d=64, H=4, L=2, h_e=64, k=2, dense=False, pkw=None):
        super().__init__()
        self.Q, self.W, self.d, self.H, self.L, self.E, self.k = Q, W, d, H, L, E, k
        self.dense = dense
        self.win = nn.Linear(W, d)
        self.qid = nn.Parameter(torch.randn(Q, d) * 0.1)
        self.ln1 = nn.ModuleList([LN(d) for _ in range(L)])
        self.ln2 = nn.ModuleList([LN(d) for _ in range(L)])
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d) for _ in range(L)])
        self.o = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        self.policy = policy_cls(E, k=k, **(pkw or {}))
        if dense:     # same active FLOPs as top-k of h_e: one MLP of width k*h_e
            self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, k * h_e), nn.ReLU(), nn.Linear(k * h_e, d))
                                      for _ in range(L)])
        else:
            self.moe = nn.ModuleList([MoE(d, E, h_e, self.policy) for _ in range(L)])
            for i, m in enumerate(self.moe):
                m.li = i
        self.lnf = LN(d)
        self.rw = nn.Parameter(torch.randn(Q, d) / math.sqrt(d))
        self.rb = nn.Parameter(torch.zeros(Q))
        self.ctx = {}
        self.aux = None

    def forward(self, win):
        B = win.shape[0]
        x = self.win(2.0 * win - 1.0) + self.qid.unsqueeze(0)
        dh = self.d // self.H
        aux = x.new_zeros(())
        for l in range(self.L):
            h = self.ln1[l](x)
            q, k, v = self.qkv[l](h).view(B, self.Q, 3, self.H, dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
            x = x + self.o[l]((A @ v).transpose(1, 2).reshape(B, self.Q, self.d))
            h2 = self.ln2[l](x)
            if self.dense:
                x = x + self.mlp[l](h2)
            else:
                m, a = self.moe[l](h2, self.ctx)
                x = x + m
                aux = aux + a
        self.aux = aux
        return (self.lnf(x) * self.rw).sum(-1) + self.rb

    def active_flops_per_token(self):
        """Multiply-adds x2 per token (one question at one position), forward only."""
        d, L, Q = self.d, self.L, self.Q
        attn = 2 * (3 * d * d + d * d) + 2 * 2 * Q * d               # projections + scores/mix
        ff = 2 * 2 * d * (self.k * (self.moe[0].h if not self.dense else self.mlp[0][0].out_features // self.k))
        router = 0 if self.dense else 2 * d * self.E
        return int(L * (attn + ff + router) + 2 * 32 * d + 2 * d)


# ------------------------------------------------------------------ metrics
def mi_bits(counts):
    """I(A;B) in bits from a joint count table."""
    p = counts / max(counts.sum(), 1)
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())


class Tracker:
    """Accumulates routing statistics per layer from the test (trade-time) forward passes."""

    def __init__(self, L, E, n_fam, n_steps, window_frac=0.2):
        self.L, self.E, self.n_fam = L, E, n_fam
        self.w0 = int(n_steps * (1 - window_frac))
        self.load_win = np.zeros((L, E))
        self.fam_ex = np.zeros((L, n_fam, E))
        self.step_max = [[] for _ in range(L)]
        self.load_trace = []                       # (step, layer, max_share, dead_frac) every 50 steps
        self.cum = np.zeros((L, E))

    def update(self, s, model, fam_tok):
        for l, m in enumerate(model.moe):
            G = m.last["G"].detach()
            routed = (G > 0).float()
            share = routed.sum(0).cpu().numpy()     # tokens routed to each expert
            ntok = routed.shape[0]
            self.cum[l] += share
            self.step_max[l].append(share.max() / ntok)
            if s >= self.w0:
                self.load_win[l] += share
                ft = fam_tok.cpu().numpy()
                for f in range(self.n_fam):
                    sel = ft == f
                    if sel.any():
                        self.fam_ex[l, f] += routed[torch.as_tensor(sel, device=routed.device)].sum(0).cpu().numpy()
            if s % 50 == 0:
                self.load_trace.append((s, l, float(share.max() / ntok), float((share == 0).mean())))

    def summary(self, k):
        out = {}
        for l in range(self.L):
            lw = self.load_win[l]
            ntok = lw.sum() / k
            frac = lw / max(ntok, 1)                                  # fraction of tokens per expert
            fair = k / self.E
            out[f"L{l}.max_load"] = float(frac.max())
            out[f"L{l}.max_load_stepmean"] = float(np.mean(self.step_max[l][self.w0:]))
            out[f"L{l}.dead_frac"] = float((frac < 0.1 * fair).mean())
            out[f"L{l}.cv_load"] = float(frac.std() / max(frac.mean(), 1e-12))
            fe = self.fam_ex[l]
            fe = fe[fe.sum(1) > 0]
            mi = mi_bits(fe)
            pf = fe.sum(1) / fe.sum()
            hf = float(-(pf * np.log2(pf)).sum())
            out[f"L{l}.mi_expert_family_bits"] = mi
            out[f"L{l}.mi_norm"] = mi / hf if hf > 0 else float("nan")
        for key in ["max_load", "max_load_stepmean", "dead_frac", "cv_load", "mi_expert_family_bits", "mi_norm"]:
            out[key] = float(np.mean([out[f"L{l}.{key}"] for l in range(self.L)]))
        out["max_load_worst_layer"] = float(max(out[f"L{l}.max_load"] for l in range(self.L)))
        return out


# ------------------------------------------------------------------ run
def run(arm, E, seed, T=24000, Q=64, B=8, lr=1e-3, device="cpu", out_dir=None, policy_cls=None,
        pkw=None, dense=False, regime_every=10000, log_every=500, tag=None):
    """One prequential run. Returns a summary dict; writes <out_dir>/<run_id>.json and a trace CSV.
    Pass policy_cls for a sibling's arm (arm is then just its label)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    s = streams_ext.make_streams(Q=Q, T=T, seed=seed, regime_every=regime_every)
    st = streams_ext.to_array_stream(s)
    pc = policy_cls or POLICIES.get(arm, Policy)
    model = MoENet(st.Q, st.W, E, pc, dense=dense, pkw=pkw).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_steps = math.ceil(st.n_pos / B)
    trk = None if dense else Tracker(model.L, E, st.n_fam, n_steps)
    fam_q = torch.as_tensor(st.fam, device=device)
    qid = torch.arange(st.Q, device=device)
    state = {"s": 0}
    t0 = time.time()

    class Dev:  # stream wrapper moving tensors to the device and setting ctx before each forward
        def __init__(self, inner):
            self.inner = inner
            for a in ("Q", "W", "n_pos", "fam", "delay", "n_fam", "desc"):
                setattr(self, a, getattr(inner, a))

        def window(self, idx):
            return self.inner.window(idx).to(device)

        def label(self, idx):   # stays on CPU (run_prequential calls .numpy()); bce_keep moves it
            return self.inner.label(idx)

        def label_mask(self, idx):
            return self.inner.label_mask(idx)

    class Wrapped(nn.Module):
        """Sets ctx, adds aux loss on the train pass, runs after_step hooks, records stats."""

        def __init__(self):
            super().__init__()
            self.m = model

        def forward(self, win):
            n = win.shape[0]
            model.ctx = dict(fam=fam_q.repeat(n), qid=qid.repeat(n), step=state["s"], train=self.training)
            z = model(win)
            if not self.training and trk is not None:
                trk.update(state["s"], model, model.ctx["fam"])
            return z if self.training else z.cpu()   # run_prequential calls .numpy() on test logits

    wrapped = Wrapped()

    class Opt:
        """Optimizer shim for run_prequential: step() also runs the policy's after_step hooks.
        The aux loss is folded into the BCE by bce_keep below (value unchanged, gradient added)."""

        def zero_grad(self, set_to_none=True):
            opt.zero_grad(set_to_none=set_to_none)

        def step(self):
            opt.step()
            if not dense:
                for m in model.moe:
                    L_ = m.last
                    gout = L_["out"].grad
                    rev_all = -torch.einsum("ned,nd->ne", L_["Eo"].detach(), gout) if gout is not None else None
                    load = (L_["G"].detach() > 0).float().sum(0)
                    with torch.no_grad():
                        model.policy.after_step(m, dict(z=L_["z"].detach(), G=L_["G"].detach(), load=load,
                                                        rev_all=rev_all, train=True), model.ctx)
            state["s"] += 1

    # loss.backward() in run_prequential frees the graph; keep it for the aux backward.
    orig_bce = prequential._masked_bce

    def bce_keep(z, y, m):
        l = orig_bce(z, y.to(z.device), m.to(z.device))
        if not dense and model.aux is not None and model.aux.requires_grad:
            return l + model.aux - model.aux.detach()   # same value, aux gradient included
        return l

    prequential._masked_bce = bce_keep
    model.aux = None
    try:
        res = prequential.run_prequential(wrapped, Dev(st), Opt(), B=B)
    finally:
        prequential._masked_bce = orig_bce
    wall = time.time() - t0

    z, y, m = res["z"], res["y"], res["mask"]
    half = st.n_pos // 2
    bce_all = np.logaddexp(0, z.astype(np.float64)) - y * z
    fb, fr = prequential.per_family(z[half:], y[half:], m[half:], st.fam, st.n_fam)
    out = dict(arm=tag or arm, E=E, seed=seed, T=T, B=B, Q=st.Q, steps=res["steps"], wall_s=wall,
               bce_all=float(bce_all[m].mean()), bce_2nd_half=float(bce_all[half:][m[half:]].mean()),
               active_flops_per_token=model.active_flops_per_token(),
               n_params=int(sum(p.numel() for p in model.parameters())))
    # post-shift window: 2000 positions after each regime boundary (position = time - W)
    post = []
    for b0 in range(regime_every, T, regime_every) if regime_every else []:
        lo, hi = b0 - st.W, min(b0 - st.W + 2000, st.n_pos)
        post.append(float(bce_all[lo:hi][m[lo:hi]].mean()))
    out["bce_post_shift_2k"] = float(np.mean(post)) if post else float("nan")
    for f, name in enumerate(FAMILIES):
        out[f"bce2h.{name}"] = float(fb[f])
    out["loss_per_active_mflop"] = out["bce_2nd_half"] / (out["active_flops_per_token"] / 1e6)
    if trk is not None:
        out.update(trk.summary(model.k))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        rid = f"{tag or arm}-E{E}-s{seed}"
        with open(os.path.join(out_dir, rid + ".json.tmp"), "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(os.path.join(out_dir, rid + ".json.tmp"), os.path.join(out_dir, rid + ".json"))
        if trk is not None:
            with open(os.path.join(out_dir, rid + ".trace.csv"), "w") as fh:
                fh.write("step,layer,max_share,dead_frac\n")
                for r in trk.load_trace:
                    fh.write(",".join(str(v) for v in r) + "\n")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="none")
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--u", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pkw = {}
    if a.u is not None:
        pkw["u"] = a.u
    if a.alpha is not None:
        pkw["alpha"] = a.alpha
    dense = a.arm == "dense"
    r = run(a.arm, a.E, a.seed, T=a.T, B=a.B, device=a.device, out_dir=a.out, pkw=pkw, dense=dense, tag=a.tag)
    print(json.dumps(r))
