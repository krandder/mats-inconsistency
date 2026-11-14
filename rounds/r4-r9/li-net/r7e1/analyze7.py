#!/usr/bin/env python3
"""R7-E1 analysis: applies PREREG-R7-E1.md section 3 mechanically to results/<tag>/s<seed>/summary.json (3 seeds); writes
results/<tag>/analysis/{results.json, verdicts.md, *.csv, *.png}. Written before any main result existed.
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 analyze7.py main"""
import csv
import json
import math
import os
import sys

import numpy as np

R7 = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 4.302652729911275, 1: 12.706204736174705}
SEEDS = (101, 102, 103)
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)
LEVELS = (0, 1, 2, 3, 4, 6)
POOLS = ("mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "looped_lin", "jacobi", "looped_attn", "attn_K1",
         "attn_K3", "attn_K4")
WPOOLS = ("attn_K1", "attn_K3", "attn_K4", "looped_attn", "jacobi", "econ_K3", "kc_K1", "kc_K3", "kc_K4", "mmd")
KINDS = ("open", "puct", "greedy", "open30", "puct30")
COL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4", "gray": "#8a8984",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surf": "#fcfcfb"}


def ci(xs):
    x = np.asarray(xs, dtype=float)
    m = float(x.mean())
    h = T975.get(len(x) - 1, 1.96) * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": x.tolist()}


def fci(c, nd=4):
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({', '.join(f'{v:.{nd}f}' for v in c['per_seed'])})"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    S = {s: json.load(open(os.path.join(R7, "results", tag, f"s{s}", "summary.json"))) for s in SEEDS}
    T = {s: S[s]["test"] for s in SEEDS}
    C = {s: T[s]["controlled"] for s in SEEDS}
    out = {"tag": tag, "stats": {}, "verdicts": {}}
    st, V = out["stats"], out["verdicts"]
    PAR = S[101]["fit"]["params"]

    # ---- E1-D0 gate
    st["clause1_corr_std6"] = ci([T[s]["clause1_corr_std6"] for s in SEEDS])
    st["fs_cut_scaled_mix"] = ci([1 - T[s]["fs_kl_scaled_mix"] / T[s]["kl_p0"] for s in SEEDS])
    st["kl_p0"] = ci([T[s]["kl_p0"] for s in SEEDS])
    for a in POOLS + ("union",):
        st[f"fs_kl_{a}"] = ci([T[s][f"fs_kl_{a}"] for s in SEEDS])
        st[f"fs_jsd_{a}"] = ci([T[s][f"fs_jsd_{a}"] for s in SEEDS])
    d0 = st["clause1_corr_std6"]["mean"] >= 0.99 and st["fs_cut_scaled_mix"]["mean"] >= 0.40
    V["E1-D0"] = {"pass": d0, "stats": {"clause1_pearson": st["clause1_corr_std6"], "fullsupport_cut_scaled_mix": st["fs_cut_scaled_mix"]}}

    # ---- E1-D1.1 looped_attn_T3 on P-FullSupport
    la = "looped_attn"
    p_ratio = PAR["looped_attn"] / PAR["attn_K3"]
    kl_ratio = ci([T[s][f"fs_kl_{la}"] / T[s]["fs_kl_attn_K3"] for s in SEEDS])
    jr = {b: ci([T[s][f"fs_jsd_{b}"] / T[s][f"fs_jsd_{la}"] for s in SEEDS]) for b in ("scaled_mix", "mmd")}
    beats = {}
    for b in ("scaled_mix", "mmd", "kc_K1", "kc_K3"):
        dk = ci([T[s][f"fs_kl_{b}"] - T[s][f"fs_kl_{la}"] for s in SEEDS])
        dj = ci([T[s][f"fs_jsd_{b}"] - T[s][f"fs_jsd_{la}"] for s in SEEDS])
        beats[b] = {"kl_diff": dk, "jsd_diff": dj, "pass": dk["lo"] > 0 and dj["lo"] > 0}
    parts11 = {"params<=0.38": p_ratio <= 0.38, "kl_ratio<=1.10": kl_ratio["mean"] <= 1.10,
               "jsd_ratio>=4.5_scaled_mix": jr["scaled_mix"]["mean"] >= 4.5, "jsd_ratio>=4.5_mmd": jr["mmd"]["mean"] >= 4.5,
               **{f"beats_{b}": beats[b]["pass"] for b in beats}}
    V["E1-D1.1"] = {"pass": all(parts11.values()), "parts": parts11,
                    "stats": {"params_looped_attn": PAR["looped_attn"], "params_attn_K3": PAR["attn_K3"], "params_ratio": p_ratio,
                              "kl_ratio_to_attn_K3": kl_ratio, "kl_looped_attn": st[f"fs_kl_{la}"],
                              "kl_looped_attn<=0.038 (reported)": st[f"fs_kl_{la}"]["mean"] <= 0.038,
                              "jsd_ratio": jr, "beats": beats}}

    # ---- controlled-bag helpers
    W = lambda s, t, N, a: C[s][f"t{t}_N{N}"][f"W_{a}"]
    mse = lambda s, t, a: float(np.mean([C[s][f"t{t}_N{N}"][f"mse_{a}"] for N in NS]))
    wr = {}
    for a in WPOOLS:
        for t in TAUS:
            wr[f"{a}_W8/W1_t{t}"] = ci([W(s, t, 8, a) / W(s, t, 1, a) for s in SEEDS])
            wr[f"{a}_W16/W1_t{t}"] = ci([W(s, t, 16, a) / W(s, t, 1, a) for s in SEEDS])
            wr[f"{a}_W_t{t}"] = {N: ci([W(s, t, N, a) for s in SEEDS]) for N in NS}
    mono = lambda a: [W(s, "0.3", 8, a) / W(s, "0.3", 1, a) < W(s, "1.5", 8, a) / W(s, "1.5", 1, a)
                      < W(s, "2.5", 8, a) / W(s, "2.5", 1, a) for s in SEEDS]
    st["weights"] = wr
    st["mse"] = {f"{a}_t{t}": ci([mse(s, t, a) for s in SEEDS]) for a in POOLS for t in TAUS}

    # ---- E1-D1.2 jacobi_lin_T4
    jb = "jacobi"
    lower = {}
    for b in ("looped_lin", "kc_K1"):
        d = ci([T[s][f"fs_kl_{b}"] - T[s][f"fs_kl_{jb}"] for s in SEEDS])
        every = all(T[s][f"fs_kl_{jb}"] < T[s][f"fs_kl_{b}"] for s in SEEDS)
        lower[b] = {"kl_diff": d, "lower_every_seed": every, "pass": d["mean"] > 0 and every}
    jmono = mono(jb)
    parts12 = {"params<3000": PAR["jacobi"] < 3000, "beats_looped_lin": lower["looped_lin"]["pass"], "beats_kc_K1": lower["kc_K1"]["pass"],
               "W16/W1_t0.3<=1.25": wr[f"{jb}_W16/W1_t0.3"]["mean"] <= 1.25, "W8/W1_t2.5>=2.0": wr[f"{jb}_W8/W1_t2.5"]["mean"] >= 2.0,
               "monotone_every_seed": all(jmono)}
    V["E1-D1.2"] = {"pass": all(parts12.values()), "parts": parts12,
                    "stats": {"params_jacobi": PAR["jacobi"], "kl_jacobi": st[f"fs_kl_{jb}"], "kl_looped_lin": st["fs_kl_looped_lin"],
                              "kl_kc_K1": st["fs_kl_kc_K1"], "R6_looped_lin_kl (reported)": 0.069, "lower": lower,
                              "W16/W1_t0.3": wr[f"{jb}_W16/W1_t0.3"], "W8/W1_t2.5": wr[f"{jb}_W8/W1_t2.5"],
                              "W8/W1_by_tau": {t: wr[f"{jb}_W8/W1_t{t}"] for t in TAUS}, "monotone_per_seed": jmono}}
    V["E1-D1"] = {"pass": V["E1-D1.1"]["pass"] and V["E1-D1.2"]["pass"]}

    # ---- E1-D2 P-ActivePUCT (attn_L3-K3)
    for stg in ("mixed", "clean"):
        for kind in KINDS:
            st[f"ap_{stg}_coverage_{kind}"] = ci([T[s][f"ap_{stg}_coverage_{kind}"] for s in SEEDS])
            for a in POOLS + ("union",):
                st[f"ap_{stg}_{kind}_kl_{a}"] = ci([T[s][f"ap_{stg}_{kind}_kl_{a}"] for s in SEEDS])
                st[f"ap_{stg}_{kind}_jsd_{a}"] = ci([T[s][f"ap_{stg}_{kind}_jsd_{a}"] for s in SEEDS])
        for kind in ("puct", "greedy"):
            for h in ("NW", "SE"):
                st[f"ap_{stg}_{kind}_wave1_pref_share_{h}"] = ci([T[s][f"ap_{stg}_{kind}_wave1_pref_share_{h}"] for s in SEEDS])
                st[f"ap_{stg}_{kind}_wave1_strict_share_{h}"] = ci([T[s][f"ap_{stg}_{kind}_wave1_strict_share_{h}"] for s in SEEDS])
                st[f"ap_{stg}_{kind}_coverage_by_wave_{h}"] = [
                    float(np.mean([T[s][f"ap_{stg}_{kind}_coverage_by_wave_{h}"][i] for s in SEEDS]))
                    for i in range(len(T[101][f"ap_{stg}_{kind}_coverage_by_wave_{h}"]))]
    cov = st["ap_mixed_coverage_puct"]
    w1 = st["ap_mixed_puct_wave1_pref_share_NW"]
    parts21 = {"coverage>=0.955": cov["mean"] >= 0.955, "coverage_ci_lo>0.950": cov["lo"] > 0.950, "wave1>=0.75": w1["mean"] >= 0.75}
    V["E1-D2.1"] = {"pass": all(parts21.values()), "parts": parts21,
                    "stats": {"coverage_puct_mixed": cov, "coverage_open_mixed": st["ap_mixed_coverage_open"],
                              "coverage_puct_clean (reported)": st["ap_clean_coverage_puct"], "wave1_pref_share_NW": w1,
                              "wave1_strict_share_NW (reported)": st["ap_mixed_puct_wave1_strict_share_NW"]}}
    d22, parts22 = {}, {}
    for stg in ("mixed", "clean"):
        r = ci([T[s][f"ap_{stg}_open_jsd_attn_K3"] / T[s][f"ap_{stg}_puct_jsd_attn_K3"] for s in SEEDS])
        d = ci([T[s][f"ap_{stg}_open_jsd_attn_K3"] - T[s][f"ap_{stg}_puct_jsd_attn_K3"] for s in SEEDS])
        kc_ = ci([1 - T[s][f"ap_{stg}_puct_kl_attn_K3"] / T[s][f"ap_{stg}_open_kl_attn_K3"] for s in SEEDS])
        d22[stg] = {"jsd_ratio_open_over_puct": r, "jsd_diff_open_minus_puct": d, "kl_cut_puct_vs_open": kc_}
        parts22.update({f"{stg}_jsd_ratio>=2.0": r["mean"] >= 2.0, f"{stg}_jsd_ci>0": d["lo"] > 0, f"{stg}_kl_cut>=0.35": kc_["mean"] >= 0.35})
    V["E1-D2.2"] = {"pass": all(parts22.values()), "parts": parts22, "stats": d22}
    V["E1-D2"] = {"pass": V["E1-D2.1"]["pass"] and V["E1-D2.2"]["pass"]}

    # ---- E1-D3 controlled bags (attn_L3-K3)
    a3 = "attn_K3"
    cut = lambda t, b: ci([1 - mse(s, t, a3) / mse(s, t, b) for s in SEEDS])
    dif = lambda t, b: ci([mse(s, t, b) - mse(s, t, a3) for s in SEEDS])
    r3 = {"W16/W1_t0.3": wr[f"{a3}_W16/W1_t0.3"], "W8/W1_t2.5": wr[f"{a3}_W8/W1_t2.5"], "W16/W1_t2.5": wr[f"{a3}_W16/W1_t2.5"],
          "W8/W1_by_tau": {t: wr[f"{a3}_W8/W1_t{t}"] for t in TAUS}}
    for t in TAUS:
        for b in ("mmd", "kc_K1", "kc_K3", "kc_K4", "econ_K3", "jacobi", "looped_attn", "mix", "scaled_mix"):
            r3[f"msecut_vs_{b}_t{t}"] = cut(t, b)
            r3[f"msediff_{b}_minus_attnK3_t{t}"] = dif(t, b)
    c1 = {"W16/W1<=1.25": r3["W16/W1_t0.3"]["mean"] <= 1.25,
          "cut_vs_mmd>=0.35": r3["msecut_vs_mmd_t0.3"]["mean"] >= 0.35, "ci_vs_mmd>0": r3["msediff_mmd_minus_attnK3_t0.3"]["lo"] > 0,
          "cut_vs_kc_K1>=0.20": r3["msecut_vs_kc_K1_t0.3"]["mean"] >= 0.20, "ci_vs_kc_K1>0": r3["msediff_kc_K1_minus_attnK3_t0.3"]["lo"] > 0}
    c2 = {"W8/W1>=2.20": r3["W8/W1_t2.5"]["mean"] >= 2.20, "W8/W1_ci_lo>1.90": r3["W8/W1_t2.5"]["lo"] > 1.90,
          "W16/W1>=2.80": r3["W16/W1_t2.5"]["mean"] >= 2.80, "cut_vs_kc_K1>=0.20": r3["msecut_vs_kc_K1_t2.5"]["mean"] >= 0.20,
          "cut_vs_mmd>=0.20": r3["msecut_vs_mmd_t2.5"]["mean"] >= 0.20, "beats_kc_K3_ci>0": r3["msediff_kc_K3_minus_attnK3_t2.5"]["lo"] > 0}
    amono = mono(a3)
    r3["monotone_per_seed"] = amono
    V["E1-D3"] = {"pass": all(c1.values()) and all(c2.values()) and all(amono),
                  "parts": {"D3.1_clean_dedup": all(c1.values()), "D3.2_high_noise_support": all(c2.values()),
                            "D3.3_monotone": all(amono), "D3.1": c1, "D3.2": c2}, "stats": r3}
    if not d0:
        for k in V:
            if k != "E1-D0":
                V[k]["note"] = "gate failed: descriptive"

    # ---- descriptive: keys, union decomposition, scalars
    keyres = {}
    for v in ("K1", "K3", "K4", "jacobi", "looped_attn", "looped_lin"):
        keyres[v] = {"pearson": ci([T[s][f"key_{v}_pearson"] for s in SEEDS]), "gap": ci([T[s][f"key_{v}_gap"] for s in SEEDS]),
                     "cos_by_overlap_mean": [float(np.mean([T[s][f"key_{v}_by_overlap"][str(o)] for s in SEEDS])) for o in LEVELS]}
    st["keys"] = keyres
    out["fit"] = {s: {k: S[s]["fit"][k] for k in ("scalars", "beta_by_tau", "params", "heldout_bag_bce_to_ref", "jacobi",
                                                  "looped_lin", "looped_attn_step_emb_norm", "ref_coverage_mean")} for s in SEEDS}
    out["compute"] = {s: {"train_sec": S[s]["train_sec"], "fit_sec": S[s]["fit"]["sec"], "test_sec": T[s]["sec"],
                          "total_sec": S[s]["total_sec"], "ckpt": S[s].get("ckpt"), "loaded": S[s].get("loaded_theta_real")} for s in SEEDS}
    od = os.path.join(R7, "results", tag, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=1, default=float)
    P_ = lambda k: "PASS" if V[k]["pass"] else "FAIL"
    L = ["| rule | verdict | key statistics: 3-seed mean [seed-t 95% CI] (per seed) |", "|---|---|---|",
         f"| E1-D0 | {P_('E1-D0')} | clause-1 Pearson {fci(st['clause1_corr_std6'])}; P-FullSupport scaled_mix KL cut "
         f"{fci(st['fs_cut_scaled_mix'])} |",
         f"| E1-D1.1 looped_attn_T3 | {P_('E1-D1.1')} | params {PAR['looped_attn']} / {PAR['attn_K3']} = {p_ratio:.3f}; KL ratio to "
         f"attn_L3-K3 {fci(kl_ratio, 3)}; KL {fci(st['fs_kl_looped_attn'])}; JSD ratio vs scaled_mix {fci(jr['scaled_mix'], 2)}, vs mmd "
         f"{fci(jr['mmd'], 2)}; " + "; ".join(f"{b}: KL diff {fci(beats[b]['kl_diff'])}, JSD diff {fci(beats[b]['jsd_diff'], 5)}"
                                              for b in beats) + f"; parts {parts11} |",
         f"| E1-D1.2 jacobi_lin_T4 | {P_('E1-D1.2')} | params {PAR['jacobi']}; KL {fci(st['fs_kl_jacobi'])} vs looped_lin "
         f"{fci(st['fs_kl_looped_lin'])} vs kc_K1 {fci(st['fs_kl_kc_K1'])}; W16/W1 tau0.3 {fci(wr['jacobi_W16/W1_t0.3'], 3)}; W8/W1 "
         f"by tau " + ", ".join(fci(wr[f'jacobi_W8/W1_t{t}'], 3) for t in TAUS) + f"; monotone per seed {jmono}; parts {parts12} |",
         f"| E1-D1 (both) | {P_('E1-D1')} | |",
         f"| E1-D2.1 | {P_('E1-D2.1')} | coverage puct (mixed) {fci(cov)} vs open {fci(st['ap_mixed_coverage_open'], 3)}; clean puct "
         f"{fci(st['ap_clean_coverage_puct'])}; wave-1 NW share {fci(w1, 3)} (wholly in half {fci(st['ap_mixed_puct_wave1_strict_share_NW'], 3)}) |",
         f"| E1-D2.2 | {P_('E1-D2.2')} | " + "; ".join(
             f"{stg}: JSD ratio {fci(d22[stg]['jsd_ratio_open_over_puct'], 2)}, diff {fci(d22[stg]['jsd_diff_open_minus_puct'], 5)}, KL cut "
             f"{fci(d22[stg]['kl_cut_puct_vs_open'], 3)}" for stg in ("mixed", "clean")) + f"; parts {parts22} |",
         f"| E1-D2 (both) | {P_('E1-D2')} | |",
         f"| E1-D3 | {P_('E1-D3')} | tau0.3: W16/W1 {fci(r3['W16/W1_t0.3'], 3)}, MSE cut vs mmd {fci(r3['msecut_vs_mmd_t0.3'], 3)}, vs "
         f"kc_K1 {fci(r3['msecut_vs_kc_K1_t0.3'], 3)}; tau2.5: W8/W1 {fci(r3['W8/W1_t2.5'], 3)}, W16/W1 {fci(r3['W16/W1_t2.5'], 3)}, MSE "
         f"cut vs kc_K1 {fci(r3['msecut_vs_kc_K1_t2.5'], 3)}, vs mmd {fci(r3['msecut_vs_mmd_t2.5'], 3)}, MSE(kc_K3) - MSE(attn_K3) "
         f"{fci(r3['msediff_kc_K3_minus_attnK3_t2.5'], 6)}; monotone per seed {amono}; parts {V['E1-D3']['parts']} |"]
    open(os.path.join(od, "verdicts.md"), "w").write("\n".join(L) + "\n")
    plots(st, wr, r3, od)
    print("\n".join(L))


def plots(st, wr, r3, od):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["ink2"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                         "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": COL["surf"], "axes.facecolor": COL["surf"],
                         "lines.linewidth": 2.0, "legend.frameon": False})
    # P-FullSupport pools
    names = ["scaled_mix", "mmd", "kc_K1", "kc_K3", "econ_K3", "looped_lin", "jacobi", "looped_attn", "attn_K1", "attn_K3", "attn_K4"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for i, (m, ttl, c) in enumerate((("kl", "P-FullSupport: KL to exact Bayes", COL["s1"]),
                                     ("jsd", "P-FullSupport: JSD between U_NW and U_SE", COL["s2"]))):
        ax[i].bar(range(len(names)), [st[f"fs_{m}_{a}"]["mean"] for a in names], color=c, width=0.6)
        ax[i].set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax[i].set(title=ttl)
        ax[i].set_axisbelow(True)
        ax[i].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "fullsupport_pools.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "fullsupport_pools.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "kl_mean", "kl_lo", "kl_hi", "jsd_mean", "jsd_lo", "jsd_hi"])
        for a in list(POOLS) + ["union"]:
            k_, j_ = st[f"fs_kl_{a}"], st[f"fs_jsd_{a}"]
            w.writerow([a, k_["mean"], k_["lo"], k_["hi"], j_["mean"], j_["lo"], j_["hi"]])
    # P-ActivePUCT
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    kinds = ["open", "puct", "greedy", "open30", "puct30"]
    xs = np.arange(len(kinds))
    for j, (stg, c) in enumerate((("mixed", COL["s1"]), ("clean", COL["s3"]))):
        ax[0].bar(xs + (j - 0.5) * 0.38, [st[f"ap_{stg}_coverage_{k}"]["mean"] for k in kinds], width=0.38, color=c, label=stg)
    ax[0].axhline(0.955, color=COL["ink2"], lw=1, ls="--")
    ax[0].set_ylim(0.7, 1.0)
    ax[0].set_xticks(xs, ["open K24", "PUCT K24", "greedy K24", "open K30", "PUCT K30"], fontsize=8)
    ax[0].set(title="coverage (bar: 0.955)")
    ax[0].legend(fontsize=7)
    for j, stg in enumerate(("mixed", "clean")):
        a = ax[1 + j]
        a.bar(np.arange(3) - 0.2, [st[f"ap_{stg}_{k}_jsd_attn_K3"]["mean"] for k in ("open", "puct", "greedy")], width=0.4,
              color=COL["s2"], label="attn_L3-K3")
        a.bar(np.arange(3) + 0.2, [st[f"ap_{stg}_{k}_jsd_union"]["mean"] for k in ("open", "puct", "greedy")], width=0.4,
              color=COL["gray"], label="union call (clean readings)")
        a.set_xticks(range(3), ["open_U", "PUCT", "greedy"])
        a.set(title=f"{stg} reads: JSD between U_NW and U_SE")
        a.legend(fontsize=7)
    for a in ax:
        a.set_axisbelow(True)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "puct.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "puct.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "search", "coverage", "jsd_attn_K3", "kl_attn_K3", "jsd_union", "kl_union"])
        for stg in ("mixed", "clean"):
            for k in kinds:
                w.writerow([stg, k, st[f"ap_{stg}_coverage_{k}"]["mean"], st[f"ap_{stg}_{k}_jsd_attn_K3"]["mean"],
                            st[f"ap_{stg}_{k}_kl_attn_K3"]["mean"], st[f"ap_{stg}_{k}_jsd_union"]["mean"],
                            st[f"ap_{stg}_{k}_kl_union"]["mean"]])
    # controlled bags
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    tc = {"0.3": COL["s1"], "1.5": COL["s2"], "2.5": COL["s3"]}
    rows = []
    for i, a in enumerate(("attn_K3", "jacobi")):
        for t in TAUS:
            m = [wr[f"{a}_W_t{t}"][N]["mean"] for N in NS]
            ax[i].plot(NS, m, color=tc[t], marker="o", ms=4, label=f"tau {t}")
            rows += [[a, t, N, wr[f"{a}_W_t{t}"][N]["mean"]] for N in NS]
        ax[i].set(title=f"{a} cluster weight W(N)", xlabel="N same-patch reads", xscale="log")
        ax[i].set_xticks(NS, [str(x) for x in NS])
        ax[i].legend(fontsize=8)
    bs = ["attn_K3", "looped_attn", "jacobi", "econ_K3", "kc_K3", "kc_K1", "mmd"]
    for j, t in enumerate(("0.3", "2.5")):
        ax[2].bar(np.arange(len(bs)) + (j - 0.5) * 0.38, [st["mse"][f"{b}_t{t}"]["mean"] for b in bs], width=0.38,
                  color=tc[t], label=f"tau {t}")
    ax[2].set_xticks(range(len(bs)), bs, rotation=45, ha="right", fontsize=8)
    ax[2].set(title="controlled bags: 36-cell MSE to exact Bayes (mean over N)")
    ax[2].legend(fontsize=8)
    ax[2].set_axisbelow(True)
    ax[2].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "controlled_bags.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "controlled_bags.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "tau", "N", "W_mean"])
        w.writerows(rows)
        for a in WPOOLS:
            if a in ("attn_K3", "jacobi"):
                continue
            for t in TAUS:
                w.writerows([[a, t, N, wr[f"{a}_W_t{t}"][N]["mean"]] for N in NS])
    with open(os.path.join(od, "controlled_mse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "tau", "mse_mean", "mse_lo", "mse_hi"])
        for b in POOLS:
            for t in TAUS:
                c = st["mse"][f"{b}_t{t}"]
                w.writerow([b, t, c["mean"], c["lo"], c["hi"]])


if __name__ == "__main__":
    main()
