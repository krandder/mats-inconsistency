#!/usr/bin/env python3
"""R9-E1 audit, checks 5 and 8: paired comparisons behind RESULTS prose claims (summary.json per-seed values, seed-t 4.303),
E1-D1.2 re-decided at each strong-root candidate M_0 in {30, 32, 34}, econ_prec's ridge above 2, the other 2-scalar comparator.
Run: python3 check8_extra.py > check8_extra.txt"""
import json
import math

import numpy as np

RES = "/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main"
T = {s: json.load(open(f"{RES}/s{s}/summary.json"))["test"] for s in (101, 102, 103)}
F = {s: json.load(open(f"{RES}/s{s}/summary.json"))["fit"] for s in T}
KS = (1, 2, 4, 8, 16, 24)
la = "looped_attn"


def ci(x, t=4.303):
    x = np.asarray(x, dtype=float)
    m = x.mean()
    h = t * x.std(ddof=1) / math.sqrt(len(x))
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "per_seed": [round(float(v), 6) for v in x]}


f5 = lambda c: f"{c['mean']:.5f} [{c['lo']:.5f}, {c['hi']:.5f}] per seed {c['per_seed']}"
print("precise: s101 srB34 attn_K3 K24-K0 =", T[101]["srB34_kl_attn_K3_K24"] - T[101]["srB34_kl_attn_K3_K0"])
print("precise: s102 srB34 looped_attn K2-K0 =", T[102]["srB34_kl_looped_attn_K2"] - T[102]["srB34_kl_looped_attn_K0"])
print("precise: s102 srB32 looped_attn K1-K0 =", T[102]["srB32_kl_looped_attn_K1"] - T[102]["srB32_kl_looped_attn_K0"])
print("econ_R7 mode B above the root at K24 in every seed:", all(T[s]["srB32_kl_econ_R7_K24"] > T[s]["srB32_kl_econ_R7_K0"] for s in T),
      "| econ_R7 mode A above the root at any K in any seed:", any(T[s][f"srA32_kl_econ_R7_K{K}"] > T[s]["srA32_kl_econ_R7_K0"] for s in T for K in KS))
for anc in ("looped_attn", "attn_K3"):
    for b in ("econ_D3", "econ_prec", "econ_R7", "kc_K3", "jacobi_prec", "jacobi", "attn_K1", "looped_lin", "kc_K1"):
        d16 = ci([T[s][f"srA32_kl_{b}_K16"] - T[s][f"srA32_kl_{anc}_K16"] for s in T])
        d24 = ci([T[s][f"srA32_kl_{b}_K24"] - T[s][f"srA32_kl_{anc}_K24"] for s in T])
        print(f"mode A KL({b}) - KL({anc}): K16 {f5(d16)} | K24 {f5(d24)}")
C = {s: T[s]["controlled"] for s in T}
r = lambda s, a, t: C[s][f"t{t}_N16"][f"W_{a}"] / C[s][f"t{t}_N1"][f"W_{a}"]
for t in ("0.3", "2.5"):
    for b in ("econ_D3", "econ_R7"):
        print(f"W16/W1 {b} - econ_prec at tau {t}:", f5(ci([r(s, b, t) - r(s, "econ_prec", t) for s in T])))
mse = lambda s, a: np.mean([np.mean([C[s][f"t{t}_N{N}"][f"mse_{a}"] for N in (1, 2, 4, 8, 16)]) for t in ("0.3", "1.5", "2.5")])
for b in ("econ_D3", "econ_R7"):
    print(f"D3 MSE({b}) - MSE(econ_prec), all tiers:", f5(ci([mse(s, b) - mse(s, "econ_prec") for s in T])))
sp_inv2 = math.log(math.exp(2) - 1)
for s in F:
    a, b = F[s]["econ_prec"]["a"], F[s]["econ_prec"]["b"]
    thr = (a - sp_inv2) / b if a > sp_inv2 else float("nan")
    print(f"s{s} econ_prec lambda(beta=0) = {math.log1p(math.exp(a)):.3f}; lambda > 2 iff beta < {thr:.3f}")
print("fs KL(32) looped_attn per seed:", [round(T[s]["fs_kl_looped_attn_K32"], 5) for s in T])
print("E1-D1.2 re-decided at each strong-root candidate (registered rule uses M_0_strong = 32):")
for M in (30, 32, 34):
    p = {"KL0<=0.050": ci([T[s][f"srA{M}_kl_{la}_K0"] for s in T])["mean"] <= 0.050}
    for a in ("mix", "mmd"):
        d = ci([T[s][f"srB{M}_kl_{a}_K16"] - T[s][f"srB{M}_kl_{a}_K0"] for s in T])
        p[f"B_{a}_degrades"] = d["mean"] >= 0.015 and d["lo"] > 0
    det = {}
    for md in "AB":
        mx = {s: round(max(T[s][f"sr{md}{M}_kl_{la}_K{K}"] - T[s][f"sr{md}{M}_kl_{la}_K0"] for K in KS), 6) for s in T}
        g = ci([T[s][f"sr{md}{M}_kl_{la}_K0"] - T[s][f"sr{md}{M}_kl_{la}_K16"] for s in T])
        p[f"{md}_never_degrades"] = all(v <= 0.0005 for v in mx.values())
        p[f"{md}_K0-K16_ci>0"] = g["mean"] > 0 and g["lo"] > 0
        for b in ("mmd", "scaled_mix"):
            for K in (16, 24):
                d = ci([T[s][f"sr{md}{M}_kl_{b}_K{K}"] - T[s][f"sr{md}{M}_kl_{la}_K{K}"] for s in T])
                p[f"{md}_{b}_K{K}_ci>0"] = d["mean"] > 0 and d["lo"] > 0
        det[md] = {"max_rise": mx, "K0-K16": f5(g)}
    fails = [k for k, v in p.items() if not v]
    print(f"  M_0 = {M}: {'PASS' if not fails else 'FAIL'}; failing parts {fails}; {det}")
