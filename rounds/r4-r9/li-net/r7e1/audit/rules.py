#!/usr/bin/env python3
"""R7-E1 audit, check 5: re-decide every rule of PREREG-R7-E1 section 3 from MY numbers (rederive.json from raw.json,
ckpt-output.json for the clause-1 Pearson recomputed on CPU), with t = 4.303 (and t = 4.302653 as a check), and compare with
analysis/results.json and verdicts.md. Parameter counts come from check 6 (params_jacobi.json; instantiated modules).
Usage: python3 rules.py > rules-output.txt"""
import json
import math
import time

R7 = "/home/agent/workspaces/eurisko-study/li-net/r7e1"
S = (101, 102, 103)
M = {int(k): v for k, v in json.load(open("rederive.json")).items()}
CK = {int(k): v for k, v in json.load(open("ckpt-output.json")).items()}
PJ = json.load(open("params_jacobi.json"))
RES = json.load(open(f"{R7}/results/main/analysis/results.json"))
OV = RES["verdicts"]
T_LIST = (4.303, 4.302652729911275)


def ci(xs, t):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    h = t * sd / math.sqrt(len(xs))
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": list(xs)}


def f(c, nd=4):
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({', '.join(f'{x:.{nd}f}' for x in c['per_seed'])})"


def decide(t):
    V, X = {}, {}
    g = lambda k: [M[s][k] for s in S]
    # E1-D0
    X["pearson"] = ci([CK[s]["pearson_cpu"] for s in S], t)
    X["fs_cut_scaled_mix"] = ci([1 - M[s]["fs_kl_scaled_mix"] / M[s]["kl_p0"] for s in S], t)
    V["E1-D0"] = {"pearson>=0.99": X["pearson"]["mean"] >= 0.99, "cut>=0.40": X["fs_cut_scaled_mix"]["mean"] >= 0.40}
    # E1-D1.1
    pr = PJ["counted"]["looped_attn_T3+K3head"] / PJ["counted"]["attn_L3+K3head"]
    X["params_ratio"] = pr
    X["kl_ratio"] = ci([M[s]["fs_kl_looped_attn"] / M[s]["fs_kl_attn_K3"] for s in S], t)
    X["kl_looped_attn"] = ci(g("fs_kl_looped_attn"), t)
    d11 = {"params<=0.38": pr <= 0.38, "kl_ratio<=1.10": X["kl_ratio"]["mean"] <= 1.10}
    for b in ("scaled_mix", "mmd"):
        X[f"jsd_ratio_{b}"] = ci([M[s][f"fs_jsd_{b}"] / M[s]["fs_jsd_looped_attn"] for s in S], t)
        d11[f"jsd_ratio>=4.5_{b}"] = X[f"jsd_ratio_{b}"]["mean"] >= 4.5
    for b in ("scaled_mix", "mmd", "kc_K1", "kc_K3"):
        X[f"kl_diff_{b}"] = ci([M[s][f"fs_kl_{b}"] - M[s]["fs_kl_looped_attn"] for s in S], t)
        X[f"jsd_diff_{b}"] = ci([M[s][f"fs_jsd_{b}"] - M[s]["fs_jsd_looped_attn"] for s in S], t)
        d11[f"beats_{b}"] = X[f"kl_diff_{b}"]["lo"] > 0 and X[f"jsd_diff_{b}"]["lo"] > 0
    V["E1-D1.1"] = d11
    # E1-D1.2
    d12 = {"params<3000": PJ["counted"]["jacobi_lin_T4+K3head"] < 3000}
    X["kl_jacobi"] = ci(g("fs_kl_jacobi"), t)
    for b in ("looped_lin", "kc_K1"):
        X[f"kl_{b}"] = ci(g(f"fs_kl_{b}"), t)
        d12[f"beats_{b}"] = X["kl_jacobi"]["mean"] < X[f"kl_{b}"]["mean"] and all(M[s]["fs_kl_jacobi"] < M[s][f"fs_kl_{b}"] for s in S)
    for tau in ("0.3", "1.5", "2.5"):
        X[f"jac_W8/W1_t{tau}"] = ci(g(f"jacobi_W8/W1_t{tau}"), t)
        X[f"jac_W16/W1_t{tau}"] = ci(g(f"jacobi_W16/W1_t{tau}"), t)
    d12["W16/W1_t0.3<=1.25"] = X["jac_W16/W1_t0.3"]["mean"] <= 1.25
    d12["W8/W1_t2.5>=2.0"] = X["jac_W8/W1_t2.5"]["mean"] >= 2.0
    d12["monotone_every_seed"] = all(M[s]["jacobi_W8/W1_t0.3"] < M[s]["jacobi_W8/W1_t1.5"] < M[s]["jacobi_W8/W1_t2.5"] for s in S)
    V["E1-D1.2"] = d12
    # E1-D2.1
    X["cov_puct_mixed"] = ci(g("ap_mixed_coverage_puct"), t)
    X["cov_open_mixed"] = ci(g("ap_mixed_coverage_open"), t)
    X["cov_puct_clean"] = ci(g("ap_clean_coverage_puct"), t)
    X["wave1_NW"] = ci(g("ap_mixed_puct_wave1_pref_share_NW"), t)
    X["wave1_SE"] = ci(g("ap_mixed_puct_wave1_pref_share_SE"), t)
    X["wave1_strict_NW"] = ci(g("ap_mixed_puct_wave1_strict_share_NW"), t)
    V["E1-D2.1"] = {"coverage>=0.955": X["cov_puct_mixed"]["mean"] >= 0.955, "coverage_ci_lo>0.950": X["cov_puct_mixed"]["lo"] > 0.950,
                    "wave1>=0.75": X["wave1_NW"]["mean"] >= 0.75}
    # E1-D2.2
    d22 = {}
    for st in ("mixed", "clean"):
        X[f"{st}_jsd_ratio"] = ci([M[s][f"ap_{st}_open_jsd_attn_K3"] / M[s][f"ap_{st}_puct_jsd_attn_K3"] for s in S], t)
        X[f"{st}_jsd_diff"] = ci([M[s][f"ap_{st}_open_jsd_attn_K3"] - M[s][f"ap_{st}_puct_jsd_attn_K3"] for s in S], t)
        X[f"{st}_kl_cut"] = ci([1 - M[s][f"ap_{st}_puct_kl_attn_K3"] / M[s][f"ap_{st}_open_kl_attn_K3"] for s in S], t)
        d22.update({f"{st}_jsd_ratio>=2.0": X[f"{st}_jsd_ratio"]["mean"] >= 2.0, f"{st}_jsd_ci>0": X[f"{st}_jsd_diff"]["lo"] > 0,
                    f"{st}_kl_cut>=0.35": X[f"{st}_kl_cut"]["mean"] >= 0.35})
    V["E1-D2.2"] = d22
    # E1-D3 (attn_L3-K3)
    for tau in ("0.3", "1.5", "2.5"):
        X[f"a3_W8/W1_t{tau}"] = ci(g(f"attn_K3_W8/W1_t{tau}"), t)
        X[f"a3_W16/W1_t{tau}"] = ci(g(f"attn_K3_W16/W1_t{tau}"), t)
        for b in ("mmd", "kc_K1", "kc_K3", "jacobi"):
            X[f"cut_{b}_t{tau}"] = ci([1 - M[s][f"mse_attn_K3_t{tau}"] / M[s][f"mse_{b}_t{tau}"] for s in S], t)
            X[f"diff_{b}_t{tau}"] = ci([M[s][f"mse_{b}_t{tau}"] - M[s][f"mse_attn_K3_t{tau}"] for s in S], t)
    c1 = {"W16/W1<=1.25": X["a3_W16/W1_t0.3"]["mean"] <= 1.25, "cut_vs_mmd>=0.35": X["cut_mmd_t0.3"]["mean"] >= 0.35,
          "ci_vs_mmd>0": X["diff_mmd_t0.3"]["lo"] > 0, "cut_vs_kc_K1>=0.20": X["cut_kc_K1_t0.3"]["mean"] >= 0.20,
          "ci_vs_kc_K1>0": X["diff_kc_K1_t0.3"]["lo"] > 0}
    c2 = {"W8/W1>=2.20": X["a3_W8/W1_t2.5"]["mean"] >= 2.20, "W8/W1_ci_lo>1.90": X["a3_W8/W1_t2.5"]["lo"] > 1.90,
          "W16/W1>=2.80": X["a3_W16/W1_t2.5"]["mean"] >= 2.80, "cut_vs_kc_K1>=0.20": X["cut_kc_K1_t2.5"]["mean"] >= 0.20,
          "cut_vs_mmd>=0.20": X["cut_mmd_t2.5"]["mean"] >= 0.20, "beats_kc_K3_ci>0": X["diff_kc_K3_t2.5"]["lo"] > 0}
    c3 = {"monotone_every_seed": all(M[s]["attn_K3_W8/W1_t0.3"] < M[s]["attn_K3_W8/W1_t1.5"] < M[s]["attn_K3_W8/W1_t2.5"] for s in S)}
    V["E1-D3"] = {**{f"D3.1 {k}": v for k, v in c1.items()}, **{f"D3.2 {k}": v for k, v in c2.items()}, **{f"D3.3 {k}": v for k, v in c3.items()}}
    return V, X


print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
V, X = decide(4.303)
V2, X2 = decide(T_LIST[1])
order = ("E1-D0", "E1-D1.1", "E1-D1.2", "E1-D2.1", "E1-D2.2", "E1-D3")
mine = {r: all(V[r].values()) for r in order}
mine["E1-D1"] = mine["E1-D1.1"] and mine["E1-D1.2"]
mine["E1-D2"] = mine["E1-D2.1"] and mine["E1-D2.2"]
print("== verdicts: mine (t=4.303) | mine (t=4.302653) | analysis/results.json")
for r in ("E1-D0", "E1-D1.1", "E1-D1.2", "E1-D1", "E1-D2.1", "E1-D2.2", "E1-D2", "E1-D3"):
    m2 = all(V2[r].values()) if r in V2 else (all(V2["E1-D1.1"].values()) and all(V2["E1-D1.2"].values()) if r == "E1-D1" else all(V2["E1-D2.1"].values()) and all(V2["E1-D2.2"].values()))
    print(f"{r:8s} {'PASS' if mine[r] else 'FAIL'} | {'PASS' if m2 else 'FAIL'} | {'PASS' if OV[r]['pass'] else 'FAIL'}"
          + ("" if mine[r] == OV[r]["pass"] else "   <-- DIFFERENT"))
print("== failing clauses (mine):", {r: [k for k, v in V[r].items() if not v] for r in order})
print("== my statistics (3-seed mean [seed-t 95% CI, t=4.303] (per seed))")
for k, c in X.items():
    print(f"  {k}: " + (f"{c:.5f}" if isinstance(c, float) else f(c, 5)))

# ---- compare my 3-seed stats with analysis/results.json
st, vv = RES["stats"], RES["verdicts"]
pairs = {
    "pearson": st["clause1_corr_std6"], "fs_cut_scaled_mix": st["fs_cut_scaled_mix"],
    "kl_ratio": vv["E1-D1.1"]["stats"]["kl_ratio_to_attn_K3"], "kl_looped_attn": st["fs_kl_looped_attn"],
    "jsd_ratio_scaled_mix": vv["E1-D1.1"]["stats"]["jsd_ratio"]["scaled_mix"], "jsd_ratio_mmd": vv["E1-D1.1"]["stats"]["jsd_ratio"]["mmd"],
    **{f"kl_diff_{b}": vv["E1-D1.1"]["stats"]["beats"][b]["kl_diff"] for b in ("scaled_mix", "mmd", "kc_K1", "kc_K3")},
    **{f"jsd_diff_{b}": vv["E1-D1.1"]["stats"]["beats"][b]["jsd_diff"] for b in ("scaled_mix", "mmd", "kc_K1", "kc_K3")},
    "kl_jacobi": st["fs_kl_jacobi"], "kl_looped_lin": st["fs_kl_looped_lin"], "kl_kc_K1": st["fs_kl_kc_K1"],
    **{f"jac_W8/W1_t{t}": st["weights"][f"jacobi_W8/W1_t{t}"] for t in ("0.3", "1.5", "2.5")},
    **{f"jac_W16/W1_t{t}": st["weights"][f"jacobi_W16/W1_t{t}"] for t in ("0.3", "1.5", "2.5")},
    "cov_puct_mixed": st["ap_mixed_coverage_puct"], "cov_open_mixed": st["ap_mixed_coverage_open"], "cov_puct_clean": st["ap_clean_coverage_puct"],
    "wave1_NW": st["ap_mixed_puct_wave1_pref_share_NW"], "wave1_SE": st["ap_mixed_puct_wave1_pref_share_SE"],
    "wave1_strict_NW": st["ap_mixed_puct_wave1_strict_share_NW"],
    **{f"{s_}_jsd_ratio": vv["E1-D2.2"]["stats"][s_]["jsd_ratio_open_over_puct"] for s_ in ("mixed", "clean")},
    **{f"{s_}_jsd_diff": vv["E1-D2.2"]["stats"][s_]["jsd_diff_open_minus_puct"] for s_ in ("mixed", "clean")},
    **{f"{s_}_kl_cut": vv["E1-D2.2"]["stats"][s_]["kl_cut_puct_vs_open"] for s_ in ("mixed", "clean")},
    **{f"a3_W8/W1_t{t}": st["weights"][f"attn_K3_W8/W1_t{t}"] for t in ("0.3", "1.5", "2.5")},
    **{f"a3_W16/W1_t{t}": st["weights"][f"attn_K3_W16/W1_t{t}"] for t in ("0.3", "1.5", "2.5")},
    **{f"cut_{b}_t{t}": vv["E1-D3"]["stats"][f"msecut_vs_{b}_t{t}"] for b in ("mmd", "kc_K1", "kc_K3", "jacobi") for t in ("0.3", "1.5", "2.5")},
    **{f"diff_{b}_t{t}": vv["E1-D3"]["stats"][f"msediff_{b}_minus_attnK3_t{t}"] for b in ("mmd", "kc_K1", "kc_K3", "jacobi") for t in ("0.3", "1.5", "2.5")},
}
worst = (0, None)
print("== 3-seed comparison with analysis/results.json (max |difference| over mean, lo, hi and per-seed values)")
for k, ref in pairs.items():
    c = X[k]
    d = max([abs(c["mean"] - ref["mean"]), abs(c["lo"] - ref["lo"]), abs(c["hi"] - ref["hi"])]
            + [abs(a - b) for a, b in zip(c["per_seed"], ref["per_seed"])])
    if d > worst[0]:
        worst = (d, k)
    print(f"  {k:22s} max diff {d:.2g}")
print("LARGEST 3-seed-statistic difference vs results.json (t=4.303 vs owner's 4.302653 included):", worst)
_, X3 = decide(T_LIST[1])
w3 = max((max([abs(X3[k]["mean"] - r["mean"]), abs(X3[k]["lo"] - r["lo"]), abs(X3[k]["hi"] - r["hi"])]), k) for k, r in pairs.items())
print("LARGEST 3-seed-statistic difference vs results.json with the owner's t = 4.302653:", w3)
json.dump({"verdicts_mine": mine, "clauses": V, "stats": X}, open("rules.json", "w"), indent=1)
