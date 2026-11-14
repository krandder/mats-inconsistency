#!/usr/bin/env python3
"""R7-E1: print every number RESULTS-R7-E1.md quotes, each rounded ONCE from full precision (results.json / summary.json),
plus the derived descriptive numbers (added after scoring at the audit's request; PREREG deviation 2).
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 numbers7.py main > results/main/analysis/numbers.txt"""
import json
import sys

import numpy as np

import os
R7 = os.environ.get("R7DIR", "/home/agent/workspaces/eurisko-study/li-net/r7e1")
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
r = json.load(open(f"{R7}/results/{tag}/analysis/results.json"))
S = {s: json.load(open(f"{R7}/results/{tag}/s{s}/summary.json")) for s in (101, 102, 103)}
st, V = r["stats"], r["verdicts"]


def c(x, nd=4):
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}] (" + " / ".join(f"{v:.{nd}f}" for v in x["per_seed"]) + ")"


print("== verdicts")
for k, v in V.items():
    print(k, "PASS" if v["pass"] else "FAIL", v.get("parts", ""))
print("== gate")
print("clause1", c(st["clause1_corr_std6"])); print("fs cut scaled_mix", c(st["fs_cut_scaled_mix"], 4)); print("kl_p0", c(st["kl_p0"]))
print("== P-FullSupport KL / JSD")
for a in ("mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "looped_lin", "jacobi", "looped_attn", "attn_K1",
          "attn_K3", "attn_K4", "union"):
    print(f"{a:12s} KL {c(st[f'fs_kl_{a}'])}  JSD {c(st[f'fs_jsd_{a}'], 5)}")
d11 = V["E1-D1.1"]["stats"]
print("== D1.1", "params", d11["params_looped_attn"], d11["params_attn_K3"], f"{d11['params_ratio']:.4f}")
print("kl ratio", c(d11["kl_ratio_to_attn_K3"], 4))
for b, x in d11["jsd_ratio"].items():
    print("jsd ratio", b, c(x, 3))
for b, x in d11["beats"].items():
    print("beats", b, "KLdiff", c(x["kl_diff"], 4), "JSDdiff", c(x["jsd_diff"], 5), x["pass"])
d12 = V["E1-D1.2"]["stats"]
print("== D1.2 params", d12["params_jacobi"])
for b, x in d12["lower"].items():
    print("lower vs", b, c(x["kl_diff"], 4), "every seed", x["lower_every_seed"])
for t, x in d12["W8/W1_by_tau"].items():
    print("jacobi W8/W1 tau", t, c(x, 3))
print("jacobi W16/W1 tau0.3", c(d12["W16/W1_t0.3"], 3), "mono", d12["monotone_per_seed"])
print("== D2.1")
for k, x in V["E1-D2.1"]["stats"].items():
    print(k, c(x, 4) if isinstance(x, dict) else x)
print("== D2.2")
for stg, d in V["E1-D2.2"]["stats"].items():
    for k, x in d.items():
        print(stg, k, c(x, 4))
print("== PUCT table")
for stg in ("mixed", "clean"):
    for kind in ("open", "puct", "greedy", "open30", "puct30"):
        print(stg, kind, "cov", c(st[f"ap_{stg}_coverage_{kind}"], 4), "attnK3 JSD", c(st[f"ap_{stg}_{kind}_jsd_attn_K3"], 5),
              "KL", c(st[f"ap_{stg}_{kind}_kl_attn_K3"], 4), "union JSD", c(st[f"ap_{stg}_{kind}_jsd_union"], 5), "KL",
              c(st[f"ap_{stg}_{kind}_kl_union"], 4))
    for kind in ("puct", "greedy"):
        for h in ("NW", "SE"):
            print(stg, kind, h, "wave1 pref", c(st[f"ap_{stg}_{kind}_wave1_pref_share_{h}"], 3), "strict",
                  c(st[f"ap_{stg}_{kind}_wave1_strict_share_{h}"], 3), "cov by wave",
                  [round(x, 4) for x in st[f"ap_{stg}_{kind}_coverage_by_wave_{h}"]])
print("== D3")
d3 = V["E1-D3"]["stats"]
for k in ("W16/W1_t0.3", "W8/W1_t2.5", "W16/W1_t2.5"):
    print(k, c(d3[k], 3))
for t, x in d3["W8/W1_by_tau"].items():
    print("attnK3 W8/W1 tau", t, c(x, 3))
print("monotone", d3["monotone_per_seed"])
for t in ("0.3", "1.5", "2.5"):
    for b in ("mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "jacobi", "looped_attn", "mix"):
        print(f"tau {t} cut vs {b}", c(d3[f"msecut_vs_{b}_t{t}"], 4), "diff", c(d3[f"msediff_{b}_minus_attnK3_t{t}"], 6))
print("== W(N) means")
for a in ("attn_K3", "jacobi", "looped_attn", "econ_K3", "kc_K3", "kc_K1", "attn_K4"):
    for t in ("0.3", "1.5", "2.5"):
        print(a, t, [round(st["weights"][f"{a}_W_t{t}"][str(N)]["mean"], 3) for N in (1, 2, 4, 8, 16)],
              "W8/W1", round(st["weights"][f"{a}_W8/W1_t{t}"]["mean"], 3), "W16/W1", round(st["weights"][f"{a}_W16/W1_t{t}"]["mean"], 3))
print("== MSE means")
for a in ("attn_K3", "attn_K4", "attn_K1", "looped_attn", "jacobi", "econ_K3", "kc_K3", "kc_K4", "kc_K1", "mmd", "mix", "scaled_mix", "looped_lin"):
    print(a, [f"{st['mse'][f'{a}_t{t}']['mean']:.5f}" for t in ("0.3", "1.5", "2.5")])
print("== keys")
for v, x in st["keys"].items():
    print(v, "pearson", c(x["pearson"], 3), "gap", c(x["gap"], 3), "cos", [round(y, 3) for y in x["cos_by_overlap_mean"]])
print("== fit")
for s, f in r["fit"].items():
    print(s, "scalars", {k: round(v, 4) for k, v in f["scalars"].items()}, "beta", {k: round(v, 3) for k, v in f["beta_by_tau"].items()})
    print(s, "params", f["params"])
    print(s, "jacobi lam mean", round(float(np.mean(f["jacobi"]["lambda"])), 3), "eta", [round(x, 3) for x in f["jacobi"]["eta"]],
          "P mean", [round(x, 3) for x in f["jacobi"]["P_mean"]], "s mean", round(float(np.mean(f["jacobi"]["s"])), 3))
for k in ("scalars",):
    for key in r["fit"]["101"][k]:
        vals = [r["fit"][s][k][key] for s in ("101", "102", "103")]
        print("mean", key, round(float(np.mean(vals)), 4), [round(v, 4) for v in vals])
for t in ("0.3", "1.5", "2.5"):
    vals = [r["fit"][s]["beta_by_tau"][t] for s in ("101", "102", "103")]
    print("beta mean tau", t, round(float(np.mean(vals)), 3), [round(v, 3) for v in vals])
print("== compute")
for s, x in r["compute"].items():
    print(s, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in x.items() if k not in ("ckpt", "loaded")}, x.get("ckpt"), x.get("loaded"))
for s in (101, 102, 103):
    print(s, "train curve last", S[s]["train_curve"][-1] if S[s].get("train_curve") else None)


# ---------------------------------------------------------------------------------------------- derived (descriptive) numbers
# Added after scoring at the audit's request (deviation 2): every derived number RESULTS quotes, from summary.json, per seed
# first and then 3-seed mean [seed-t 95% CI]. Nothing here enters a registered rule.
T975 = 4.302652729911275
TS = {s: S[s]["test"] for s in (101, 102, 103)}


def d(name, xs, scale=1.0, nd=4):
    x = np.asarray(xs, float) * scale
    m = x.mean()
    h = T975 * x.std(ddof=1) / np.sqrt(3)
    print(f"{name}: {m:.{nd}f} [{m - h:.{nd}f}, {m + h:.{nd}f}] per seed " + " / ".join(f"{v:.{nd}f}" for v in x))


g = lambda k: np.array([TS[s][k] for s in TS])
mse = lambda t, a: np.array([np.mean([TS[s]["controlled"][f"t{t}_N{N}"][f"mse_{a}"] for N in (1, 2, 4, 8, 16)]) for s in TS])
print("== derived: P-FullSupport")
d("scaled_mix KL cut % vs root", 1 - g("fs_kl_scaled_mix") / g("kl_p0"), 100, 2)
d("jacobi KL cut % vs looped_lin", 1 - g("fs_kl_jacobi") / g("fs_kl_looped_lin"), 100, 1)
d("KL looped_attn - attn_K3", g("fs_kl_looped_attn") - g("fs_kl_attn_K3"), 1, 4)
d("JSD looped_attn - attn_K3", g("fs_jsd_looped_attn") - g("fs_jsd_attn_K3"), 1, 5)
d("KL jacobi - econ_K3", g("fs_kl_jacobi") - g("fs_kl_econ_K3"), 1, 4)
print("== derived: P-ActivePUCT")
for stg in ("mixed", "clean"):
    d(f"{stg} KL cut % puct vs open (attn_K3)", 1 - g(f"ap_{stg}_puct_kl_attn_K3") / g(f"ap_{stg}_open_kl_attn_K3"), 100, 2)
    d(f"{stg} union JSD open / puct", g(f"ap_{stg}_open_jsd_union") / g(f"ap_{stg}_puct_jsd_union"), 1, 1)
    d(f"{stg} attn_K3 JSD / union JSD after PUCT", g(f"ap_{stg}_puct_jsd_attn_K3") / g(f"ap_{stg}_puct_jsd_union"), 1, 2)
d("attn_K3 JSD after PUCT, mixed / clean", g("ap_mixed_puct_jsd_attn_K3") / g("ap_clean_puct_jsd_attn_K3"), 1, 2)
print("== derived: controlled-bag MSE cuts of attn_L3-K3 (%)")
for t in ("0.3", "1.5", "2.5"):
    for b in ("mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "jacobi", "looped_attn"):
        d(f"tau {t} cut vs {b}", 1 - mse(t, "attn_K3") / mse(t, b), 100, 2)
print("== derived: key attribution in both orders (share of the kc_K1 -> attn_L3-K3 gap)")
for t in ("0.3", "1.5", "2.5"):
    a, k3, a1, a3 = mse(t, "kc_K1"), mse(t, "kc_K3"), mse(t, "attn_K1"), mse(t, "attn_K3")
    d(f"MSE tau {t}, keys swapped inside kc_dedup (%)", (a - k3) / (a - a3), 100, 1)
    d(f"MSE tau {t}, keys swapped inside attn_L3 (%)", (a1 - a3) / (a - a3), 100, 1)
for m in ("kl", "jsd"):
    a, k3, a1, a3 = g(f"fs_{m}_kc_K1"), g(f"fs_{m}_kc_K3"), g(f"fs_{m}_attn_K1"), g(f"fs_{m}_attn_K3")
    d(f"P-FullSupport {m}, keys swapped inside kc_dedup (%)", (a - k3) / (a - a3), 100, 1)
    d(f"P-FullSupport {m}, keys swapped inside attn_L3 (%)", (a1 - a3) / (a - a3), 100, 1)
print("== derived: jacobi_lin_T4 closed form for N identical reads, 16 (lambda + beta) / (lambda + 16 beta)")
for s in ("101", "102", "103"):
    lam = float(np.mean(r["fit"][s]["jacobi"]["lambda"]))
    b = r["fit"][s]["beta_by_tau"]["0.3"]
    print(s, f"lambda cell mean {lam:.3f}, beta tau0.3 {b:.3f}, closed form {16 * (lam + b) / (lam + 16 * b):.3f}, measured "
          f"{st['weights']['jacobi_W16/W1_t0.3']['per_seed'][int(s) - 101]:.3f}, bar needs lambda <= {4 * b / 14.75:.3f}")
