#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written after the registered R8-E3 verdicts were seen; NOT a rule test). The registered E3-D1
fits let attn_L3-K3c and looped_attn_T3-K3c memorise the calibration positions (calibration cross-entropy 1.10 -> 0.97,
test KL 0.049 -> 0.64). This rerun changes ONE thing: every D1 aggregator is fitted on 1,500 calibration roots with
early stopping on 500 further, held-out calibration roots (validation cross-entropy against the true hidden types
every 50 steps; the best step is kept). Everything else (rollouts, bags, test positions, generators, scoring) is the
registered r8/run_r8e3.py code, so the 500 test bags are the registered ones.
   python3 r8/expl_d1_es.py --seed 101 --ckpt /tmp/r5e2/ckpt/s101/primary --out OUT [--smoke]"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_r8e3 as RR  # noqa: E402
import r8lib as Y  # noqa: E402
import r7lib as X  # noqa: E402
import e3lib as L  # noqa: E402
import eval6 as EV  # noqa: E402
import train6 as T  # noqa: E402


def gather(Mo, Rc, roots, gen, dev):
    PL, PR, L0, HID, TRUE = [], [], [], [], []
    for c0 in range(0, roots.shape[0], 128):
        ix = roots[c0:c0 + 128]
        rt, pL, pR = RR.d1_pools(Mo, Rc, ix, gen, dev)
        PL.append(pL)
        PR.append(pR)
        L0.append(rt["p0b"])
        HID.append(rt["hidden"])
        TRUE.append(Rc["slot_true"][ix].long())
    cat = lambda xs: {k: torch.cat([x[k] for x in xs]) for k in xs[0]}
    return cat(PL), cat(PR), torch.cat(L0), torch.cat(HID), torch.cat(TRUE)


def fit_d1_es(a, Mo, Rc, dev):
    gen = torch.Generator(device=dev).manual_seed(a.seed + 810)
    roots = T.two_flank_roots(Rc, a.n_train + a.n_val, gen, dev)
    tr = gather(Mo, Rc, roots[:a.n_train], gen, dev)
    va = gather(Mo, Rc, roots[a.n_train:], gen, dev)
    res = {"n_train": int(tr[2].shape[0]), "n_val": int(va[2].shape[0]),
           "cal_ce_anchor_train": float(Y.ce_true(tr[2], tr[4], tr[3])), "val_ce_anchor": float(Y.ce_true(va[2], va[4], va[3]))}

    def mk_bag(D, skew, g):
        poolL, poolR, l0 = D[0], D[1], D[2]
        N = l0.shape[0]
        pL = torch.full((N,), 0.5, device=dev) if skew == "unskewed" else \
            torch.distributions.Beta(0.5, 0.5).sample((N,)).to(dev)
        s = Y.seq_bag(poolL, poolR, pL, g)
        K = torch.randint(1, Y.BAG + 1, (N,), generator=g, device=dev)
        return Y.prefix(s, K, l0)

    vgen = torch.Generator(device=dev).manual_seed(a.seed + 815)
    torch.manual_seed(a.seed + 816)
    vbags = {sk: [mk_bag(va, sk, vgen) for _ in range(2)] for sk in ("unskewed", "beta")}

    def vce(fn, skew):
        with torch.no_grad():
            return float(sum(Y.ce_true(fn(b), va[4], va[3]) for b in vbags[skew]) / len(vbags[skew]))

    def fit(name, n_steps, fn, modules, scalars, lr, skew="beta", extra=None):
        params = [p for m in modules for p in m.parameters() if p.requires_grad] + scalars
        opt = torch.optim.Adam(params, lr=lr)
        best = (vce(fn, skew), -1, [copy.deepcopy(m.state_dict()) for m in modules], [s.detach().clone() for s in scalars])
        hist = [[-1, None, best[0]]]
        for st in range(n_steps):
            b = mk_bag(tr, skew, gen)
            loss = Y.ce_true(fn(b), tr[4], tr[3])
            tot = loss + (extra(b) if extra is not None else 0.0)
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
            if (st + 1) % 50 == 0:
                v = vce(fn, skew)
                hist.append([st + 1, float(loss), v])
                if v < best[0]:
                    best = (v, st + 1, [copy.deepcopy(m.state_dict()) for m in modules], [s.detach().clone() for s in scalars])
        for m, sd in zip(modules, best[2]):
            m.load_state_dict(sd)
        with torch.no_grad():
            for s, v in zip(scalars, best[3]):
                s.copy_(v)
        res[name + "_hist"] = hist
        res[name + "_best_step"] = best[1]
        res[name + "_best_val_ce"] = best[0]

    la = torch.tensor(math.log(0.1), device=dev, requires_grad=True)
    fit("scaled_mix", a.d1_steps, lambda b: Y.agg_scaled(b, la.exp()), [], [la], 1e-2, skew="unskewed")
    res["alpha"] = float(la.exp())
    le = torch.tensor(math.log(0.02), device=dev, requires_grad=True)
    fit("mmd_fit", a.d1_steps, lambda b: Y.agg_mmd(b, le.exp()), [], [le], 1e-2)
    res["eta_fit"] = float(le.exp())
    torch.manual_seed(a.seed + 820)
    key = X.KeyK3c().to(dev)
    lu = torch.tensor(math.log(0.1), device=dev, requires_grad=True)
    fit("kc_K3c", a.d1_steps, lambda b: Y.agg_kc(b, key(b), lu.exp()), [key], [lu], 3e-3,
        extra=lambda b: L.key_align_loss(key(b), key.target(b), b["valid"]))
    res["u"] = float(lu.exp())
    for p in key.parameters():
        p.requires_grad_(False)
    nets = {}
    for nm, kw, so in (("attn_L3_K3c", dict(layers=3), 830), ("looped_attn_T3_K3c", dict(loops=3), 840)):
        torch.manual_seed(a.seed + so)
        m = Y.AnchorAttn(**kw).to(dev)
        fit(nm, a.d1_steps, lambda b, m=m: m(b, key(b)), [m], [], 1e-3)
        T.freeze(m)
        nets[nm] = m
    with torch.no_grad():
        for nm, fn in RR.d1_fns(res, key, nets).items():
            res[f"val_ce_K16_{nm}"] = float(Y.ce_true(fn(Y.prefix(Y.seq_bag(va[0], va[1], torch.full((va[2].shape[0],), 0.5, device=dev), vgen), 16, va[2])), va[4], va[3]))
    return key, nets, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--d1-steps", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_train, a.n_val, a.d1_steps = 32, 16, 100
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    eo = os.path.join(a.out, "eval")
    os.makedirs(eo, exist_ok=True)
    Mo = EV.Models(a.ckpt, "primary", dev)
    h0 = T.param_hash(Mo.net)
    assert h0 == Mo.meta["param_hash_final"]
    R = T.load_split("test", dev)
    Rc = T.load_split("cal", dev)
    t0 = time.time()
    key, nets, fres = fit_d1_es(a, Mo, Rc, dev)
    fres["sec"] = time.time() - t0
    json.dump(fres, open(os.path.join(eo, "fit_es.json"), "w"), indent=1, default=float)
    RR.log("fit_es", json.dumps({k: v for k, v in fres.items() if not k.endswith("hist")}, default=float)[:1500])
    t1 = time.time()
    r = RR.part_d1(a, Mo, key, nets, fres, R, dev, eo)
    r["sec"] = time.time() - t1
    json.dump(r, open(os.path.join(eo, "d1.json"), "w"), indent=1, default=float)
    RR.log("d1", json.dumps({k: v for k, v in r.items() if k in ("attn_L3_K3c", "looped_attn_T3_K3c", "scaled_mix")}, default=float)[:1500])
    assert T.param_hash(Mo.net) == h0
    json.dump({"param_hash": h0, "unchanged": True, "sec": time.time() - t0}, open(os.path.join(eo, "done.json"), "w"))


if __name__ == "__main__":
    main()
