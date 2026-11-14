"""M7 w3: Exp3 shadow credit for a top-k MoE router (see PREREG.md).

Imports w1's shared harness (../w1/moe_harness.py). Arms: exp3 (one uniformly sampled unrouted expert per
token, importance-weighted), fullinfo (every unrouted expert, q=1; diagnostic), plus the harness's
none / deepseek controls.

Usage: python exp3.py --arm exp3 --E 8 --seed 1 --out results/
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# harness: a snapshot next to this file (GPU job dir) wins, else w1's live copy
_LOCAL = os.path.join(HERE, "moe_harness.py")
HARNESS = _LOCAL if os.path.exists(_LOCAL) else os.path.join(HERE, "..", "w1", "moe_harness.py")
sys.path.insert(0, os.path.dirname(HARNESS))
import moe_harness as H  # noqa: E402

REGISTRY = []


class Exp3Shadow(H.Policy):
    """Shadow credit for unrouted experts. After each train step, the censored router gradient
    (-dL/dz_i = g_i (r_i - R) for i in S, 0 outside) is completed with an importance-weighted estimate for
    the unrouted experts: c_e = ghat_e (r_e - R) / q_e, ghat_e = the gate e would get if added to S.
    The policy reads rev_all only at the routed set and at the sampled shadow expert (exp3); fullinfo reads
    every unrouted expert (q=1). The extra router gradient -sum_n c[n,e] x_n enters on the next step."""
    name = "exp3"
    full = False

    def __init__(self, E, k=2, **kw):
        super().__init__(E, k=k, **kw)
        self.layers = []
        self.gen = torch.Generator().manual_seed(int(kw.get("sample_seed", 12345)))
        self.n_updates = 0
        self.pend_norm = []
        REGISTRY.append(self)

    def init_layer(self, layer):
        layer._pend = None
        layer._x = None
        layer.wealth = torch.zeros(self.E, dtype=torch.float64)
        layer.shadow_credit = torch.zeros(self.E, dtype=torch.float64)
        layer.router.register_forward_hook(lambda mod, inp, out: setattr(layer, "_x", inp[0].detach()))
        self.layers.append(layer)

    def aux_loss(self, layer, z, G, ctx):
        if ctx.get("train") and layer._pend is not None:
            # value is irrelevant (the harness adds aux - aux.detach()); gradient wrt router.weight = _pend
            return (layer.router.weight * layer._pend).sum()
        return z.new_zeros(())

    def after_step(self, layer, rec, ctx):
        rev, G, z, x = rec["rev_all"], rec["G"], rec["z"], layer._x
        if rev is None or x is None:
            layer._pend = None
            return
        routed = G > 0
        R = (G * rev).sum(1)                                              # [N] realised revenue of the mix
        layer.wealth += (G * rev).sum(0).double().cpu()
        N, E = z.shape
        if self.full:
            mask = (~routed).float()
            q = 1.0
        else:
            u = torch.rand(N, E, generator=self.gen).to(z.device)
            u[routed] = -1.0
            e = u.argmax(1)
            mask = torch.zeros_like(z).scatter(1, e[:, None], 1.0)
            n_unr = (~routed).sum(1, keepdim=True).float()
            q = 1.0 / n_unr                                               # [N,1] uniform over unrouted
        zS = z.masked_fill(~routed, float("-inf"))
        lseS = torch.logsumexp(zS, 1, keepdim=True)                       # [N,1]
        ghat = torch.exp(z - torch.logaddexp(z, lseS))                    # gate if e were added to S
        c = mask * ghat * (rev - R[:, None]) / q                          # [N,E] credit (= -dL/dz_e estimate)
        layer.shadow_credit += c.sum(0).double().cpu()
        layer._pend = -(c.t() @ x)                                        # dL/dW_router, [E,d]
        self.n_updates += 1


class FullInfo(Exp3Shadow):
    name = "fullinfo"
    full = True


POL = {"exp3": Exp3Shadow, "fullinfo": FullInfo}


def gini(w):
    w = np.sort(np.asarray(w, float))
    if w.min() < 0:
        w = w - w.min()
    n = len(w)
    if w.sum() <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(w) / (n * w.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="exp3")
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=int, default=24000)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sha = hashlib.sha256(open(HARNESS, "rb").read()).hexdigest()
    pc = POL.get(a.arm)
    pkw = {"sample_seed": 1000 + a.seed}
    tag = f"w3{a.arm}"
    r = H.run(a.arm, a.E, a.seed, T=a.T, B=a.B, device=a.device, out_dir=a.out, policy_cls=pc, pkw=pkw, tag=tag)
    r["harness_sha256"] = sha
    r["exp3_sha256"] = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    k = 2
    n_exp_train = (k + 1) if a.arm == "exp3" else (a.E if a.arm == "fullinfo" else k)
    r["train_experts_evaluated_per_token"] = n_exp_train
    r["train_expert_flop_ratio_vs_topk"] = n_exp_train / k
    if pc is not None and REGISTRY:
        p = REGISTRY[-1]
        for l, lay in enumerate(p.layers):
            r[f"L{l}.wealth_gini"] = gini(lay.wealth.numpy())
            r[f"L{l}.wealth"] = [float(v) for v in lay.wealth.numpy()]
            r[f"L{l}.shadow_credit"] = [float(v) for v in lay.shadow_credit.numpy()]
        r["shadow_updates"] = p.n_updates
    rid = f"{tag}-E{a.E}-s{a.seed}"
    path = os.path.join(a.out, rid + ".json")
    with open(path + ".tmp", "w") as fh:
        json.dump(r, fh, indent=1)
    os.replace(path + ".tmp", path)
    print(json.dumps({kk: v for kk, v in r.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    main()
