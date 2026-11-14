#!/usr/bin/env python3
"""R9-E1: print every number RESULTS-R9-E1.md quotes, each rounded ONCE from full precision (results.json / summary.json),
plus derived descriptive numbers. Written before any main result existed.
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 numbers9.py main > results/main/analysis/numbers.txt"""
import json
import os
import sys

import numpy as np

R9 = os.environ.get("R9DIR", os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
SEEDS = tuple(int(x) for x in os.environ.get("R9SEEDS", "101,102,103").split(","))
r = json.load(open(f"{R9}/results/{tag}/analysis/results.json"))
S = {s: json.load(open(f"{R9}/results/{tag}/s{s}/summary.json")) for s in SEEDS}
TS = {s: S[s]["test"] for s in S}
st, V = r["stats"], r["verdicts"]
Ms = {s: int(S[s]["fit"]["M_strong"]) for s in SEEDS}
T975 = 4.302652729911275
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
SR = (18, 30, 32, 34)
POOLS = ("looped_attn", "attn_K3", "jacobi_prec", "jacobi", "econ_prec", "econ_D3", "econ_R7", "kc_K3", "kc_K1", "looped_lin",
         "scaled_mix", "mmd", "mix", "attn_K1")


def c(x, nd=4):
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}] (" + " / ".join(f"{v:.{nd}f}" for v in x["per_seed"]) + ")"


def d(name, xs, nd=4):
    x = np.asarray(xs, float)
    m = x.mean()
    h = T975 * x.std(ddof=1) / np.sqrt(len(x))
    print(f"{name}: {m:.{nd}f} [{m - h:.{nd}f}, {m + h:.{nd}f}] per seed " + " / ".join(f"{v:.{nd}f}" for v in x))


print("== verdicts")
for k, v in V.items():
    print(k, "PASS" if v["pass"] else "FAIL", v.get("parts", ""))
print("== E1-D0")
print("sha256", V["E1-D0"]["stats"]["sha256"])
print("clause-1 Pearson", c(st["clause1_corr_std6"], 5))
print("K=0 max KL diff (M_0 = 2 and M_0_strong)", V["E1-D0"]["stats"]["K0_max_kl_diff"], "max logit diff",
      V["E1-D0"]["stats"]["K0_max_abs_logit_diff"], "every M_0", V["E1-D0"]["stats"]["K0_max_kl_diff_every_M0 (reported)"])
print("== strong-root selection (calibration boards)")
for s in SEEDS:
    sel = S[s]["fit"]["strong_selection"]
    print(s, "M_strong", Ms[s], "KL(ref36||root)", {k: round(v, 5) for k, v in sel["kl_ref36_root"].items()}, "KL(root||ref36)",
          {k: round(v, 5) for k, v in sel["kl_root_ref36_reported"].items()})
print("== root KL by M_0 (test, exact Bayes)")
for M, x in st["kl_root"].items():
    print("root KL M", M, c(x, 5))
print("== sparse-root FS curve (KL, JSD) per pool")
for a in POOLS:
    print(a, "KL", [f"{st[f'fs_kl_{a}_K{K}']['mean']:.4f}" for K in KS_FS], "JSD", [f"{st[f'fs_jsd_{a}_K{K}']['mean']:.5f}" for K in KS_FS])
print("looped_attn step increases per seed", V["E1-D1.1"]["stats"]["step_increase_per_seed"])
for K in KS_FS:
    print(f"looped_attn FS K{K}", c(st[f"fs_kl_looped_attn_K{K}"]), "JSD", c(st[f"fs_jsd_looped_attn_K{K}"], 5))
for a in ("jacobi_prec", "jacobi", "looped_lin", "kc_K1", "kc_K3", "econ_prec", "econ_D3", "looped_attn", "attn_K3"):
    print(f"FS KL K24 {a}", c(st[f"fs_kl_{a}_K24"]))
for b in ("looped_lin", "kc_K1", "kc_K3", "econ_prec"):
    d(f"FS KL K24 {b} - jacobi_prec", [TS[s][f"fs_kl_{b}_K24"] - TS[s][f"fs_kl_jacobi_prec_K24"] for s in SEEDS])
print("FS union K24 KL", np.mean([TS[s]["fs_kl_union_K24"] for s in SEEDS]), "JSD", np.mean([TS[s]["fs_jsd_union_K24"] for s in SEEDS]))
print("== strong roots: KL(K) - KL(0) per pool, by mode and root (3-seed means)")
for md in ("A", "B"):
    for M in list(SR) + ["S"]:
        tg = f"sr{md}{M}" if M != "S" else f"srS{md}"
        print(f"-- mode {md} M_0 {M} KL(0)", c(st[f"{tg}_kl_looped_attn_K0"], 5))
        for a in POOLS:
            print(f"{tg} {a} dKL", [f"{st[f'{tg}_dkl_{a}_K{K}']['mean']:+.5f}" for K in KS_SR])
    for K in KS_SR[1:]:
        print(f"srS{md} K{K} looped_attn dKL", c(st[f"srS{md}_dkl_looped_attn_K{K}"], 5), "mmd", c(st[f"srS{md}_dkl_mmd_K{K}"], 5),
              "mix", c(st[f"srS{md}_dkl_mix_K{K}"], 5), "scaled_mix", c(st[f"srS{md}_dkl_scaled_mix_K{K}"], 5))
print("E1-D1.2 stats", json.dumps(V["E1-D1.2"]["stats"], default=float)[:6000])
print("== anytime pools per seed: largest step change and largest change from K = 0")
for a in ("looped_attn", "attn_K3"):
    curves = [("fs", KS_FS, lambda s: "fs")] + [(f"sr{md}{M}", KS_SR, None) for md in ("A", "B") for M in SR]
    for tg, KS, _ in curves:
        for s in SEEDS:
            steps = [TS[s][f"{tg}_kl_{a}_K{KS[i + 1]}"] - TS[s][f"{tg}_kl_{a}_K{KS[i]}"] for i in range(len(KS) - 1)]
            vs0 = [TS[s][f"{tg}_kl_{a}_K{K}"] - TS[s][f"{tg}_kl_{a}_K0"] for K in KS[1:]]
            i = int(np.argmax(steps))
            print(a, tg, s, f"largest step {steps[i]:+.5f} (K {KS[i]} -> {KS[i + 1]}); largest change from K0 {max(vs0):+.5f}")
print("== D3 weights")
w = st["weights"]
for a in ("econ_prec", "econ_D3", "econ_R7", "jacobi_prec", "jacobi", "kc_K3", "kc_K1", "looped_attn", "attn_K3", "attn_K1", "mmd"):
    for t in ("0.3", "1.5", "2.5"):
        print(a, t, [round(w[f"{a}_W_t{t}"][str(N)]["mean"], 3) for N in (1, 2, 4, 8, 16)], "W8/W1", c(w[f"{a}_W8/W1_t{t}"], 3),
              "W16/W1", c(w[f"{a}_W16/W1_t{t}"], 3))
print("== D3 MSE (mean over N) per tier and over tiers")
for a in POOLS:
    print(a, [f"{st['mse'][f'{a}_t{t}']['mean']:.5f}" for t in ("0.3", "1.5", "2.5")], "all", c(st["mse_all_tiers"][a], 6))
print("E1-D2.1 stats", json.dumps(V["E1-D2.1"]["stats"], default=float)[:4000])
print("E1-D2.2 stats", json.dumps(V["E1-D2.2"]["stats"], default=float)[:4000])
print("== fit")
for s, f in r["fit"].items():
    print(s, "scalars", {k: round(v, 4) for k, v in f["scalars"].items()}, "beta_clean", round(f["beta_clean"], 3))
    print(s, "beta by root", {M: {t: round(v, 3) for t, v in x.items()} for M, x in f["beta_by_root"].items()})
    jp = f["jacobi_prec"]
    print(s, "jacobi_prec a", round(jp["a"], 3), "b", round(jp["b"], 3), "lambda at tier beta", {k: round(v, 3) for k, v in jp["lambda_mean_at_beta"].items()},
          "eta", [round(x, 3) for x in jp["eta"]], "s", round(jp["s_mean"], 3))
    ep = f["econ_prec"]
    print(s, "econ_prec u", round(ep["u"], 4), "a", round(ep["a"], 4), "b", round(ep["b"], 4), "lambda at tier beta",
          {k: round(v, 4) for k, v in ep["lambda_at_beta"].items()})
    print(s, "jacobi lambda mean", round(float(np.mean(f["jacobi"]["lambda"])), 3), "params", f["params"])
    print(s, "anytime held-out BCE to ref", {k: round(v, 4) for k, v in f["heldout_any_bce_to_ref"].items()})
    print(s, "jacobi/econ held-out", {k: round(v, 5) for k, v in f["heldout_jacobi_econ"].items()})
for key in r["fit"][str(SEEDS[0])]["scalars"]:
    vals = [r["fit"][str(s)]["scalars"][key] for s in SEEDS]
    print("mean", key, round(float(np.mean(vals)), 4), [round(v, 4) for v in vals])
print("== compute", {s: {k: round(v, 1) for k, v in x.items()} for s, x in r["compute"].items()})
