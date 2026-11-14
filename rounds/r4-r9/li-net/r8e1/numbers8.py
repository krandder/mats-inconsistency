#!/usr/bin/env python3
"""R8-E1: print every number RESULTS-R8-E1.md quotes, each rounded ONCE from full precision (results.json / summary.json),
plus the derived descriptive numbers. Written before any main result existed (PREREG deviation 2).
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 numbers8.py main > results/main/analysis/numbers.txt"""
import json
import os
import sys

import numpy as np

R8 = os.environ.get("R8DIR", os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
r = json.load(open(f"{R8}/results/{tag}/analysis/results.json"))
S = {s: json.load(open(f"{R8}/results/{tag}/s{s}/summary.json")) for s in (101, 102, 103)}
TS = {s: S[s]["test"] for s in S}
st, V = r["stats"], r["verdicts"]
T975 = 4.302652729911275
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)


def c(x, nd=4):
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}] (" + " / ".join(f"{v:.{nd}f}" for v in x["per_seed"]) + ")"


def d(name, xs, scale=1.0, nd=4):
    x = np.asarray(xs, float) * scale
    m = x.mean()
    h = T975 * x.std(ddof=1) / np.sqrt(3)
    print(f"{name}: {m:.{nd}f} [{m - h:.{nd}f}, {m + h:.{nd}f}] per seed " + " / ".join(f"{v:.{nd}f}" for v in x))


g = lambda k: np.array([TS[s][k] for s in TS])
print("== verdicts")
for k, v in V.items():
    print(k, "PASS" if v["pass"] else "FAIL", v.get("parts", ""))
print("== E1-D0")
print("sha256", V["E1-D0"]["stats"]["sha256"])
print("clause-1 Pearson", c(st["clause1_corr_std6"], 5))
print("K=0 max KL diff", V["E1-D0"]["stats"]["K0_max_kl_diff"], "max logit diff", V["E1-D0"]["stats"]["K0_max_abs_logit_diff"])
print("== root KL by M_0", {M: round(x["mean"], 5) for M, x in st["kl_root"].items()})
for M, x in st["kl_root"].items():
    print("root KL M", M, c(x, 5))
print("== sparse-root FS curve (KL, JSD) per pool")
for a in ("looped_attn", "attn_K3", "jacobi_prec", "jacobi", "econ_D3", "kc_K3", "kc_K1", "looped_lin", "scaled_mix", "mmd", "mix",
          "attn_K1", "econ_R7"):
    print(a, "KL", [f"{st[f'fs_kl_{a}_K{K}']['mean']:.4f}" for K in KS_FS], "JSD", [f"{st[f'fs_jsd_{a}_K{K}']['mean']:.5f}" for K in KS_FS])
print("looped_attn step increases per seed", V["E1-D1.1"]["stats"]["step_increase_per_seed"])
for K in KS_FS:
    print(f"looped_attn FS K{K}", c(st[f"fs_kl_looped_attn_K{K}"]), "JSD", c(st[f"fs_jsd_looped_attn_K{K}"], 5))
print("FS union JSD K24", c({"mean": float(g("fs_jsd_union_K24").mean()), "lo": 0, "hi": 0, "per_seed": g("fs_jsd_union_K24").tolist()}, 5))
print("== strong roots: KL(K) - KL(0) per pool")
for tagr in sorted({k.split("_")[0] for k in st if k.startswith("sr") and "_dkl_" in k}):
    print("--", tagr, "KL(0)", c(st[f"{tagr}_kl_looped_attn_K0"], 5))
    for a in ("looped_attn", "attn_K3", "mmd", "mix", "scaled_mix", "jacobi_prec", "kc_K3", "econ_D3"):
        print(tagr, a, "dKL", [f"{st[f'{tagr}_dkl_{a}_K{K}']['mean']:+.5f}" for K in KS_SR])
    for K in KS_SR[1:]:
        print(tagr, f"looped_attn K{K} dKL", c(st[f"{tagr}_dkl_looped_attn_K{K}"], 5), "mmd", c(st[f"{tagr}_dkl_mmd_K{K}"], 5),
              "mix", c(st[f"{tagr}_dkl_mix_K{K}"], 5))
print("E1-D1.2 stats", json.dumps(V["E1-D1.2"]["stats"], default=float)[:3000])
print("== anytime pools per seed: largest step change and largest change from K = 0 (deviation 6)")
for a in ("looped_attn", "attn_K3"):
    for tg, KS in (("fs", KS_FS), ("sr6", KS_SR), ("sr12", KS_SR), ("sr18", KS_SR), ("sr18t3", KS_SR)):
        for s in TS:
            steps = [TS[s][f"{tg}_kl_{a}_K{KS[i + 1]}"] - TS[s][f"{tg}_kl_{a}_K{KS[i]}"] for i in range(len(KS) - 1)]
            vs0 = [TS[s][f"{tg}_kl_{a}_K{K}"] - TS[s][f"{tg}_kl_{a}_K0"] for K in KS[1:]]
            i = int(np.argmax(steps))
            print(a, tg, s, f"largest step {steps[i]:+.5f} (K {KS[i]} -> {KS[i + 1]}); largest change from K0 {max(vs0):+.5f}")
print("== D3 weights")
w = st["weights"]
for a in ("jacobi_prec", "jacobi", "econ_D3", "econ_R7", "kc_K3", "kc_K1", "looped_attn", "attn_K3", "attn_K1", "mmd"):
    for t in ("0.3", "1.5", "2.5"):
        print(a, t, [round(w[f"{a}_W_t{t}"][str(N)]["mean"], 3) for N in (1, 2, 4, 8, 16)], "W8/W1", c(w[f"{a}_W8/W1_t{t}"], 3),
              "W16/W1", c(w[f"{a}_W16/W1_t{t}"], 3))
print("== D3 MSE (mean over N) per tier and over tiers")
for a in ("looped_attn", "attn_K3", "attn_K1", "jacobi_prec", "jacobi", "econ_D3", "econ_R7", "kc_K3", "kc_K1", "looped_lin", "mmd",
          "scaled_mix", "mix"):
    print(a, [f"{st['mse'][f'{a}_t{t}']['mean']:.5f}" for t in ("0.3", "1.5", "2.5")], "all", c(st["mse_all_tiers"][a], 6))
print("E1-D2.1 stats", json.dumps(V["E1-D2.1"]["stats"], default=float)[:3000])
print("E1-D2.2 stats", json.dumps(V["E1-D2.2"]["stats"], default=float)[:3000])
print("== 2x2 PUCT")
for cell in ("K1_R6", "K1_R7", "K3_R6", "K3_R7"):
    print(cell, "coverage", c(st[f"puct_{cell}_coverage"]), "wave-1 NW", c(st[f"puct_{cell}_wave1_pref_share_NW"], 3), "strict",
          c(st[f"puct_{cell}_wave1_strict_share_NW"], 3), "attn_K3 JSD", c(st[f"puct_{cell}_jsd_attn_K3"], 5), "KL",
          c(st[f"puct_{cell}_kl_attn_K3"]), "looped_attn JSD", c(st[f"puct_{cell}_jsd_looped_attn"], 5), "KL", c(st[f"puct_{cell}_kl_looped_attn"]),
          "union JSD", c(st[f"puct_{cell}_jsd_union"], 5))
cov = {cell: g(f"puct_{cell}_coverage") for cell in ("K1_R6", "K1_R7", "K3_R6", "K3_R7")}
d("coverage gain of the R7 score with K1 keys", cov["K1_R7"] - cov["K1_R6"])
d("coverage gain of the R7 score with K3 keys", cov["K3_R7"] - cov["K3_R6"])
d("coverage gain of K3 keys with the R6 score", cov["K3_R6"] - cov["K1_R6"])
d("coverage gain of K3 keys with the R7 score", cov["K3_R7"] - cov["K1_R7"])
d("interaction (K3_R7 - K3_R6) - (K1_R7 - K1_R6)", (cov["K3_R7"] - cov["K3_R6"]) - (cov["K1_R7"] - cov["K1_R6"]))
print("== fit")
for s, f in r["fit"].items():
    print(s, "scalars", {k: round(v, 4) for k, v in f["scalars"].items()}, "beta_clean", round(f["beta_clean"], 3))
    print(s, "beta by root", {M: {t: round(v, 3) for t, v in x.items()} for M, x in f["beta_by_root"].items()})
    jp = f["jacobi_prec"]
    print(s, "jacobi_prec a", round(jp["a"], 3), "b", round(jp["b"], 3), "lambda at tier beta", {k: round(v, 3) for k, v in jp["lambda_mean_at_beta"].items()},
          "eta", [round(x, 3) for x in jp["eta"]], "s", round(jp["s_mean"], 3))
    print(s, "jacobi lambda mean", round(float(np.mean(f["jacobi"]["lambda"])), 3), "params", f["params"])
    print(s, "anytime held-out BCE to ref", {k: round(v, 4) for k, v in f["heldout_any_bce_to_ref"].items()})
for key in r["fit"]["101"]["scalars"]:
    vals = [r["fit"][s]["scalars"][key] for s in ("101", "102", "103")]
    print("mean", key, round(float(np.mean(vals)), 4), [round(v, 4) for v in vals])
print("== compute", {s: {k: round(v, 1) for k, v in x.items()} for s, x in r["compute"].items()})
