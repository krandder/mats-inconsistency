#!/usr/bin/env python3
"""R8-E1 audit, check 5: re-decide E1-D0, E1-D1.1, E1-D1.2, E1-D2.1, E1-D2.2 with PREREG-R8-E1 section 3, from the auditor's own
re-derived numbers (a4_rederive.json, a4b_oracle_root.json). Inputs that raw.json does not contain (P-FullSupport KL of jacobi_prec,
looped_lin and kc_K1; the K = 0 identity at full precision) are taken from summary.json and marked "summary". Compares with
analysis/results.json verdicts. Run after a4_rederive.py and a4b_oracle_root.py: python3 a5_rules.py"""
import json
import math
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R = "/home/agent/workspaces/eurisko-study/li-net/r8e1/results/main"
SEEDS = ("101", "102", "103")
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
T2 = math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2))
A = json.load(open(os.path.join(OUT, "a4_rederive.json")))["per_seed"]
B = json.load(open(os.path.join(OUT, "a4b_oracle_root.json")))["per_seed"]
S = {s: json.load(open(f"{R}/s{s}/summary.json")) for s in SEEDS}
T = {s: S[s]["test"] for s in SEEDS}
theirs = json.load(open(f"{R}/analysis/results.json"))["verdicts"]


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = T2 * x.std(ddof=1) / math.sqrt(len(x))
    return {"mean": round(float(m), 6), "lo": round(float(m - h), 6), "hi": round(float(m + h), 6), "per_seed": [round(float(v), 6) for v in x]}


V = {}
# E1-D0
pear = ci([B[s]["clause1_pearson_cpu"] for s in SEEDS])
ident = max(max(abs(v) for v in T[s]["identity_K0"].values()) for s in SEEDS)
p = {"sha256_match_all_seeds (own sha256 of the npz vs .json)": all(B[s]["ckpt_sha256_matches_json"] for s in SEEDS),
     "mean_pearson>=0.999 (own CPU recompute)": pear["mean"] >= 0.999,
     "K0 |KL diff| < 1e-7 every seed, M_0 2 and 18 (summary full precision; raw rounded arrays equal)":
         ident < 1e-7 and all(A[s]["K0_equal_root_fs"] and A[s]["K0_equal_root_sr18"] for s in SEEDS)}
V["E1-D0"] = {"pass": all(p.values()), "parts": p, "stats": {"pearson": pear, "max_identity_value_in_summary": ident}}
# E1-D1.1
la = "looped_attn"
steps = {s: [A[s][f"fs_kl_{la}_K{KS_FS[i + 1]}"] - A[s][f"fs_kl_{la}_K{KS_FS[i]}"] for i in range(len(KS_FS) - 1)] for s in SEEDS}
kl32, jsd24 = ci([A[s][f"fs_kl_{la}_K32"] for s in SEEDS]), ci([A[s][f"fs_jsd_{la}_K24"] for s in SEEDS])
p = {"every step <= +0.0005 every seed": all(d <= 0.0005 for s in SEEDS for d in steps[s]), "mean KL(32) <= 0.033": kl32["mean"] <= 0.033,
     "mean JSD(24) <= 0.0075": jsd24["mean"] <= 0.0075}
V["E1-D1.1"] = {"pass": all(p.values()), "parts": p, "stats": {"max_step": round(max(d for s in SEEDS for d in steps[s]), 6),
                "largest_step_per_seed": {s: round(max(steps[s]), 6) for s in SEEDS}, "KL32": kl32, "JSD24": jsd24,
                "KL_by_K_mean": {K: round(float(np.mean([A[s][f'fs_kl_{la}_K{K}'] for s in SEEDS])), 5) for K in KS_FS}}}
# E1-D1.2 (sr18 = tau 50/50 over 0.3/1.5)
g = lambda tg, a, K: [A[s][f"{tg}_kl_{a}_K{K}"] for s in SEEDS]
kl0 = ci(g("sr18", la, 0))
mmd = ci([A[s]["sr18_kl_mmd_K16"] - A[s]["sr18_kl_mmd_K0"] for s in SEEDS])
mix = ci([A[s]["sr18_kl_mix_K16"] - A[s]["sr18_kl_mix_K0"] for s in SEEDS])
maxdeg = {s: max(A[s][f"sr18_kl_{la}_K{K}"] - A[s][f"sr18_kl_{la}_K0"] for K in KS_SR[1:]) for s in SEEDS}
gain = ci([A[s][f"sr18_kl_{la}_K0"] - A[s][f"sr18_kl_{la}_K16"] for s in SEEDS])
vs = ci([A[s]["sr18_kl_mmd_K16"] - A[s][f"sr18_kl_{la}_K16"] for s in SEEDS])
p = {"mean KL(0) <= 0.050": kl0["mean"] <= 0.050, "mmd KL(16)-KL(0) >= 0.015, CI > 0": mmd["mean"] >= 0.015 and mmd["lo"] > 0,
     "looped_attn max_K>=1 KL(K)-KL(0) <= 0.0005 every seed": all(v <= 0.0005 for v in maxdeg.values()),
     "looped_attn KL(0)-KL(16) > 0, CI > 0": gain["mean"] > 0 and gain["lo"] > 0, "mmd(16)-looped(16) > 0, CI > 0": vs["mean"] > 0 and vs["lo"] > 0}
t3 = {"looped_attn_max_increase_per_seed": {s: round(max(A[s][f"sr18t3_kl_{la}_K{K}"] - A[s][f"sr18t3_kl_{la}_K0"] for K in KS_SR[1:]), 6) for s in SEEDS},
      "mmd_K16_minus_K0": ci([A[s]["sr18t3_kl_mmd_K16"] - A[s]["sr18t3_kl_mmd_K0"] for s in SEEDS]),
      "mmd_K24_minus_K0": ci([A[s]["sr18t3_kl_mmd_K24"] - A[s]["sr18t3_kl_mmd_K0"] for s in SEEDS]),
      "mix_K16_minus_K0": ci([A[s]["sr18t3_kl_mix_K16"] - A[s]["sr18t3_kl_mix_K0"] for s in SEEDS])}
mmd_curve = {K: round(float(np.mean([A[s][f"sr18_kl_mmd_K{K}"] - A[s]["sr18_kl_mmd_K0"] for s in SEEDS])), 5) for K in KS_SR}
V["E1-D1.2"] = {"pass": all(p.values()), "parts": p, "stats": {"KL0": kl0, "mmd_K16_minus_K0": mmd, "mix_K16_minus_K0": mix,
                "looped_attn_max_increase_per_seed": {s: round(v, 6) for s, v in maxdeg.items()}, "KL0_minus_KL16": gain, "mmd16_minus_looped16": vs,
                "mmd_dKL_by_K_mean": mmd_curve, "tier3_reported": t3}}
V["E1-D1"] = {"pass": V["E1-D1.1"]["pass"] and V["E1-D1.2"]["pass"]}
# E1-D2.1 (jacobi_prec): FS KL inputs from summary.json (raw.json does not hold jacobi_prec, looped_lin, kc_K1 on the FS stream)
W = lambda s, a, t, N: A[s][f"W_{a}_t{t}_N{N}"]
jp = "jacobi_prec"
fs = ci([T[s]["fs_kl_jacobi_prec_K24"] for s in SEEDS])
beat = {}
for b in ("looped_lin", "kc_K1"):
    d = ci([T[s][f"fs_kl_{b}_K24"] - T[s]["fs_kl_jacobi_prec_K24"] for s in SEEDS])
    beat[b] = {"diff": d, "lower_every_seed": all(T[s]["fs_kl_jacobi_prec_K24"] < T[s][f"fs_kl_{b}_K24"] for s in SEEDS),
               "own_kl": ci([T[s][f"fs_kl_{b}_K24"] for s in SEEDS])}
r16 = ci([W(s, jp, 0.3, 16) / W(s, jp, 0.3, 1) for s in SEEDS])
mono = [W(s, jp, 0.3, 8) / W(s, jp, 0.3, 1) < W(s, jp, 1.5, 8) / W(s, jp, 1.5, 1) < W(s, jp, 2.5, 8) / W(s, jp, 2.5, 1) for s in SEEDS]
r25 = ci([W(s, jp, 2.5, 16) / W(s, jp, 2.5, 1) for s in SEEDS])
p = {"mean FS KL (K = 24) <= 0.040 [summary]": fs["mean"] <= 0.040,
     "below looped_lin every seed, CI > 0 [summary]": beat["looped_lin"]["lower_every_seed"] and beat["looped_lin"]["diff"]["lo"] > 0,
     "below kc_K1 every seed, CI > 0 [summary]": beat["kc_K1"]["lower_every_seed"] and beat["kc_K1"]["diff"]["lo"] > 0,
     "mean W16/W1 tau 0.3 <= 1.25": r16["mean"] <= 1.25, "CI upper W16/W1 tau 0.3 <= 1.30": r16["hi"] <= 1.30,
     "W8/W1 strictly increasing in tau every seed": all(mono), "mean W16/W1 tau 2.5 >= 2.0": r25["mean"] >= 2.0}
V["E1-D2.1"] = {"pass": all(p.values()), "parts": p, "stats": {"fs_kl": fs, "beats": beat, "W16/W1_t0.3": r16, "W16/W1_t2.5": r25,
                "W8/W1_by_tau": {t: ci([W(s, jp, t, 8) / W(s, jp, t, 1) for s in SEEDS]) for t in (0.3, 1.5, 2.5)}, "monotone_per_seed": mono,
                "R7_jacobi_same_run_W16/W1_t0.3 [summary]": ci([T[s]["controlled"]["t0.3_N16"]["W_jacobi"] / T[s]["controlled"]["t0.3_N1"]["W_jacobi"] for s in SEEDS])}}
# E1-D2.2 (econ_D3)
ed = "econ_D3"
e03, e25 = ci([W(s, ed, 0.3, 16) / W(s, ed, 0.3, 1) for s in SEEDS]), ci([W(s, ed, 2.5, 16) / W(s, ed, 2.5, 1) for s in SEEDS])
mse = lambda s, a: float(np.mean([A[s][f"mse_{a}_t{t}_N{N}"] for t in (0.3, 1.5, 2.5) for N in (1, 2, 4, 8, 16)]))
dm = ci([mse(s, "kc_K3") - mse(s, ed) for s in SEEDS])
cells = ("K1_R6", "K1_R7", "K3_R6", "K3_R7")
p = {"mean W16/W1 tau 0.3 <= 1.25": e03["mean"] <= 1.25, "mean W16/W1 tau 2.5 >= 2.2": e25["mean"] >= 2.2,
     "MSE(kc_K3) - MSE(econ_D3) > 0, CI > 0": dm["mean"] > 0 and dm["lo"] > 0,
     "2x2 PUCT reported for all 4 cells": all(f"puct_{c}_coverage" in A[s] for c in cells for s in SEEDS)}
V["E1-D2.2"] = {"pass": all(p.values()), "parts": p, "stats": {"W16/W1_t0.3": e03, "W16/W1_t2.5": e25, "mse_kc_K3_minus_econ_D3": dm,
                "econ_R7_W16/W1_t0.3": ci([W(s, "econ_R7", 0.3, 16) / W(s, "econ_R7", 0.3, 1) for s in SEEDS]),
                "puct_coverage": {c: ci([A[s][f"puct_{c}_coverage"] for s in SEEDS]) for c in cells}}}
V["E1-D2"] = {"pass": V["E1-D2.1"]["pass"] and V["E1-D2.2"]["pass"]}
print(f"{'rule':10s} {'auditor':8s} {'analyze8':8s} agree")
agree = True
for k in ("E1-D0", "E1-D1.1", "E1-D1.2", "E1-D1", "E1-D2.1", "E1-D2.2", "E1-D2"):
    a, b = V[k]["pass"], theirs[k]["pass"]
    agree &= a == b
    print(f"{k:10s} {'PASS' if a else 'FAIL':8s} {'PASS' if b else 'FAIL':8s} {a == b}")
    if "parts" in V[k]:
        tp = list(theirs[k]["parts"].values())
        mp = list(V[k]["parts"].values())
        print("   parts (auditor):", {kk: vv for kk, vv in V[k]["parts"].items()})
        print("   parts identical to analyze8 part by part:", tp == mp)
print("ALL VERDICTS AGREE:", agree)
print(json.dumps({k: v.get("stats") for k, v in V.items()}, indent=1)[:12000])
json.dump(V, open(os.path.join(OUT, "a5_rules.json"), "w"), indent=1)
