"""M7 w5: bankruptcy-and-birth MoE arm on w1's shared harness (M7/w1/moe_harness.py).

Currency (M1/CURRENCY.lock, committed-trader settlement only): each routed expert's committed
position is its gated slot in the MoE output. At TEST (trade) time we rerun the net with that slot
removed (remaining top-k gates renormalised) and settle, when the label is revealed, the exact
log-score difference  pay = log p_full(y) - log p_without_slot(y)  (prequential: computed before
the model trains on that label). Last MoE layer: exact per-token leave-one-out (no cross-token
path after it). Layer 0: the slot is removed for all tokens jointly (attention spill from other
tokens' removals is included; approximation, stated in the deliverable).

Wealth (relative, per layer): W_i += P_i - rent, rent = max(sum_i P_i, 0)/E (market pays out what
it earns; unrouted experts pay rent and earn nothing). rent_ema = EMA(rent, 0.01). After a warmup
of H steps, the most negative expert with W_i < -H * rent_ema goes bankrupt (at most one per layer
per `cool` steps). Birth: `split` (default) = child is a copy of the richest expert (weights +
N(0, 0.01 * std) noise, router row copied exactly, Adam moments copied), parent and child share the
parent's wealth half/half. `fresh` = child freshly initialised, router row = mean router row, Adam
moments zeroed, W = 0.

Hooks used (no harness edits): labels of the train batch via prequential._masked_bce (delay 0, so
it is the just-tested block), counterfactual forwards via a MoENet.forward wrapper active only on
the test pass, the optimizer via a torch.optim.Adam wrapper (for Adam-state surgery at birth).
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

H1 = os.environ.get("M7_HARNESS_DIR", "/home/agent/workspaces/eurisko-study/traders-net/study/M7/w1")
if H1 not in sys.path:
    sys.path.insert(0, H1)
import moe_harness as MH  # noqa: E402
import prequential  # noqa: E402  (S0 substrate, on path via the harness)

_CUR = {"policy": None, "opt": None}


class BankruptcyBirth(MH.Policy):
    name = "bb"

    def __init__(self, E, k=2, **kw):
        super().__init__(E, k=k, **kw)
        self.H = kw.get("H", 200)
        self.cool = kw.get("cool", 50)
        self.birth = kw.get("birth", "split")
        self.noise = kw.get("noise", 0.01)
        self.layers = []
        self.W, self.inc, self.rent_ema, self.last_birth = {}, {}, {}, {}
        self.cf = {}          # layer index -> (expert ids [N,k], pay-able logits [k,B,Q]) from the test pass
        self.z_full = None
        self.lab = None       # (y, m) of the just-tested block
        self.events = []      # (step, layer, dead, parent, W_dead, thresh)
        self.n_births = {}
        self.pay_log = []     # (step, layer, total_pay, rent)
        _CUR["policy"] = self

    def init_layer(self, layer):
        li = len(self.layers)
        self.layers.append(layer)
        self.W[li] = torch.zeros(self.E, dtype=torch.float64)
        self.inc[li] = torch.zeros(self.E, dtype=torch.float64)
        self.rent_ema[li] = None
        self.last_birth[li] = -10 ** 9
        self.n_births[li] = 0

    # --------------------------------------------------------------- settlement
    def settle(self, li, step):
        if li not in self.cf or self.lab is None or self.z_full is None:
            return
        ids, zcf = self.cf.pop(li)
        y, m = self.lab
        s = 2.0 * y.double() - 1.0                                   # [B,Q]
        lf = F.logsigmoid(s * self.z_full.double())                  # log p_full(y)
        mk = m.double().reshape(-1)
        P = torch.zeros(self.E, dtype=torch.float64)
        for j in range(ids.shape[1]):
            lc = F.logsigmoid(s * zcf[j].double())
            pay = (lf - lc).reshape(-1) * mk                         # [N]
            P.index_add_(0, ids[:, j].cpu(), pay.cpu())
        tot = float(P.sum())
        rent = max(tot, 0.0) / self.E
        self.inc[li] += P
        self.W[li] += P - rent
        re = self.rent_ema[li]
        self.rent_ema[li] = rent if re is None else 0.99 * re + 0.01 * rent
        if step % 50 == 0:
            self.pay_log.append((step, li, tot, rent))

    def maybe_birth(self, layer, li, step):
        if step < self.H or step - self.last_birth[li] < self.cool:
            return
        thresh = -self.H * self.rent_ema[li]
        W = self.W[li]
        dead = int(torch.argmin(W))
        if not (W[dead] < thresh):
            return
        order = torch.argsort(W, descending=True)
        parent = int(order[0]) if int(order[0]) != dead else int(order[1])
        self.events.append((step, li, dead, parent, float(W[dead]), float(thresh), self.birth))
        opt = _CUR["opt"]
        with torch.no_grad():
            per_exp = [layer.w1, layer.b1, layer.w2, layer.b2]
            if self.birth == "split":
                for p in per_exp:
                    p[dead] = p[parent] + self.noise * p[parent].std() * torch.randn_like(p[parent])
                layer.router.weight[dead] = layer.router.weight[parent]
                for p in per_exp + [layer.router.weight]:
                    stt = opt.state.get(p, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in stt:
                            stt[key][dead] = stt[key][parent]
                half = max(float(W[parent]), 0.0) / 2
                W[parent] = W[parent] - half if W[parent] > 0 else W[parent]
                W[dead] = half
            else:
                d, h = layer.w1.shape[1], layer.w1.shape[2]
                layer.w1[dead] = torch.randn(d, h, device=layer.w1.device) / math.sqrt(d)
                layer.b1[dead] = 0
                layer.w2[dead] = torch.randn(h, d, device=layer.w2.device) / math.sqrt(h) * 0.5
                layer.b2[dead] = 0
                layer.router.weight[dead] = layer.router.weight.mean(0)
                for p in per_exp + [layer.router.weight]:
                    stt = opt.state.get(p, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in stt:
                            stt[key][dead] = 0
                W[dead] = 0.0
        self.inc[li][dead] = 0.0
        self.last_birth[li] = step
        self.n_births[li] += 1

    def after_step(self, layer, rec, ctx):
        li = self.layers.index(layer)
        step = ctx["step"]
        self.settle(li, step)
        self.maybe_birth(layer, li, step)


# ------------------------------------------------------------------ counterfactual test pass
def cf_forward(net, win, drop_layer, G_drop):
    """MoENet.forward with the combine matrix of `drop_layer` replaced by G_drop; no side effects."""
    B = win.shape[0]
    x = net.win(2.0 * win - 1.0) + net.qid.unsqueeze(0)
    dh = net.d // net.H
    for l in range(net.L):
        h = net.ln1[l](x)
        q, k, v = net.qkv[l](h).view(B, net.Q, 3, net.H, dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        x = x + net.o[l]((A @ v).transpose(1, 2).reshape(B, net.Q, net.d))
        h2 = net.ln2[l](x)
        mo = net.moe[l]
        xf = h2.reshape(-1, net.d)
        if l == drop_layer:
            G = G_drop
        else:
            G = net.policy.select(mo, mo.router(xf), net.ctx)
        out = torch.einsum("ne,ned->nd", G, mo.experts(xf))
        x = x + out.reshape(B, net.Q, net.d)
    return (net.lnf(x) * net.rw).sum(-1) + net.rb


_orig_forward = MH.MoENet.forward


def _forward_with_cf(self, win):
    z = _orig_forward(self, win)
    pol = self.policy
    if isinstance(pol, BankruptcyBirth) and not self.ctx.get("train", True):
        with torch.no_grad():
            pol.z_full = z.detach()
            for l, mo in enumerate(self.moe):
                G = mo.last["G"].detach()
                ids = G.topk(pol.k, dim=-1).indices                  # slot order: by gate
                zc = []
                for j in range(pol.k):
                    Gd = G.clone()
                    Gd.scatter_(-1, ids[:, j:j + 1], 0.0)
                    Gd = Gd / Gd.sum(-1, keepdim=True).clamp_min(1e-12)
                    zc.append(cf_forward(self, win, l, Gd))
                pol.cf[l] = (ids, torch.stack(zc))
    return z


MH.MoENet.forward = _forward_with_cf

_orig_adam = torch.optim.Adam


class _AdamCap(_orig_adam):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        _CUR["opt"] = self


MH.torch.optim.Adam = _AdamCap   # harness calls torch.optim.Adam (same torch module)

_orig_bce = prequential._masked_bce


def _bce_capture(z, y, m):
    pol = _CUR["policy"]
    if pol is not None:
        pol.lab = (y.detach(), m.detach() if torch.is_tensor(m) else torch.as_tensor(m))
    return _orig_bce(z, y, m)


def run_bb(E, seed, out_dir, T=24000, B=8, device="cpu", tag="bb", **pkw):
    prequential._masked_bce = _bce_capture   # harness wraps whatever is installed at call time
    try:
        r = MH.run(tag, E, seed, T=T, B=B, device=device, out_dir=None, policy_cls=BankruptcyBirth,
                   pkw=pkw, tag=tag)
    finally:
        prequential._masked_bce = _orig_bce
    pol = _CUR["policy"]
    L = len(pol.layers)
    r["n_births"] = int(sum(pol.n_births.values()))
    for li in range(L):
        r[f"L{li}.n_births"] = pol.n_births[li]
        inc = pol.inc[li].clamp_min(0).numpy()
        r[f"L{li}.gini_income"] = float(_gini(inc))
    r["gini_income"] = float(np.mean([r[f"L{li}.gini_income"] for li in range(L)]))
    r["pkw"] = {k: v for k, v in pkw.items()}
    _write(r, out_dir, f"{tag}-E{E}-s{seed}", pol)
    return r


def _gini(w):
    w = np.sort(np.asarray(w, float))
    if w.sum() <= 0:
        return float("nan")
    n = len(w)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))


def _write(r, out_dir, rid, pol):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, rid + ".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(r, fh, indent=1)
    os.replace(tmp, os.path.join(out_dir, rid + ".json"))
    if pol is not None:
        with open(os.path.join(out_dir, rid + ".births.csv"), "w") as fh:
            fh.write("step,layer,dead,parent,W_dead,thresh,birth\n")
            for e in pol.events:
                fh.write(",".join(str(v) for v in e) + "\n")
        with open(os.path.join(out_dir, rid + ".pay.csv"), "w") as fh:
            fh.write("step,layer,total_pay,rent\n")
            for e in pol.pay_log:
                fh.write(",".join(str(v) for v in e) + "\n")


def run_ctrl(arm, E, seed, out_dir, T=24000, B=8, device="cpu", **pkw):
    r = MH.run(arm, E, seed, T=T, B=B, device=device, out_dir=None, pkw=pkw, tag=arm)
    _write(r, out_dir, f"{arm}-E{E}-s{seed}", None)
    return r


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bb")        # bb | bb-fresh | none | deepseek
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=200)
    ap.add_argument("--u", type=float, default=1e-3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.arm.startswith("bb"):
        r = run_bb(a.E, a.seed, a.out, T=a.T, B=a.B, device=a.device, tag=a.arm, H=a.H,
                   birth="fresh" if a.arm == "bb-fresh" else "split")
    else:
        kw = {"u": a.u} if a.arm == "deepseek" else {}
        r = run_ctrl(a.arm, a.E, a.seed, a.out, T=a.T, B=a.B, device=a.device, **kw)
        if a.tag:
            os.replace(os.path.join(a.out, f"{a.arm}-E{a.E}-s{a.seed}.json"), os.path.join(a.out, f"{a.tag}-E{a.E}-s{a.seed}.json"))
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("bce2h")}))
