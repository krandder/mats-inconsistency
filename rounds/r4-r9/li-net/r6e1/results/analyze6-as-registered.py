#!/usr/bin/env python3
"""R6-E1 analysis: applies PREREG-R6-E1.md section 6 mechanically to results/<tag>/s<seed>/summary.json (3 seeds); writes
results/<tag>/analysis/{results.json, verdicts.md, *.csv, *.png}. Written before any main result existed.
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 analyze6.py main"""
import csv
import json
import math
import os
import sys

import numpy as np

R6 = os.path.dirname(os.path.abspath(__file__))
T975 = {2: 4.302652729911275, 1: 12.706204736174705}
SEEDS = (101, 102, 103)
VARS = ("K1", "K2", "K3", "K4", "looped")
LEVELS = (0, 1, 2, 3, 4, 6)
NS = (1, 2, 4, 8, 16)
COL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4", "gray": "#8a8984",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df", "surf": "#fcfcfb"}


def ci(xs):
    x = np.asarray(xs, dtype=float)
    m = float(x.mean())
    h = T975.get(len(x) - 1, 1.96) * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": x.tolist()}


def fci(c, nd=4):
    return f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({', '.join(f'{v:.{nd}f}' for v in c['per_seed'])})"


def spearman_levels(vals):
    """Spearman rank correlation between overlap level index (0..5) and the cosine values (ties averaged)."""
    v = np.asarray(vals, float)
    r = v.argsort().argsort().astype(float)
    for u in np.unique(v):
        m = v == u
        r[m] = r[m].mean()
    x = np.arange(len(v), dtype=float)
    return float(np.corrcoef(x, r)[0, 1])


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "main"
    S = {s: json.load(open(os.path.join(R6, "results", tag, f"s{s}", "summary.json"))) for s in SEEDS}
    T = {s: S[s]["test"] for s in SEEDS}
    out = {"tag": tag, "stats": {}, "verdicts": {}}
    st = out["stats"]
    # ---- E1-D0 gate
    st["clause1_corr_std6"] = ci([T[s]["clause1_corr_std6"] for s in SEEDS])
    st["fs_cut_scaled_mix"] = ci([1 - T[s]["fs_kl_scaled_mix"] / T[s]["kl_p0"] for s in SEEDS])
    for a in ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K2", "attn_K3", "attn_K4", "union"):
        st[f"fs_kl_{a}"] = ci([T[s][f"fs_kl_{a}"] for s in SEEDS])
        st[f"fs_jsd_{a}"] = ci([T[s][f"fs_jsd_{a}"] for s in SEEDS])
    st["kl_p0"] = ci([T[s]["kl_p0"] for s in SEEDS])
    d0 = st["clause1_corr_std6"]["mean"] >= 0.99 and st["fs_cut_scaled_mix"]["mean"] >= 0.40
    out["verdicts"]["E1-D0"] = {"pass": d0, "stats": {"clause1_pearson": st["clause1_corr_std6"],
                                                      "fullsupport_cut_scaled_mix": st["fs_cut_scaled_mix"]}}
    # ---- E1-D1 keys
    keyres = {}
    for v in VARS:
        p = ci([T[s][f"key_{v}_pearson"] for s in SEEDS])
        gp = ci([T[s][f"key_{v}_gap"] for s in SEEDS])
        curve = [float(np.mean([T[s][f"key_{v}_by_overlap"][str(o)] for s in SEEDS])) for o in LEVELS]
        sp_mean = spearman_levels(curve)
        sp_seed = [spearman_levels([T[s][f"key_{v}_by_overlap"][str(o)] for o in LEVELS]) for s in SEEDS]
        c1 = p["mean"] >= 0.60 and p["lo"] > 0.50 and gp["mean"] >= 0.50
        c2 = sp_mean >= 0.94
        keyres[v] = {"pearson": p, "gap": gp, "cos_by_overlap_mean": curve, "spearman_levels_mean_curve": sp_mean,
                     "spearman_levels_per_seed": sp_seed, "clause1": c1, "clause2": c2,
                     "same": ci([T[s][f"key_{v}_same"] for s in SEEDS]), "across": ci([T[s][f"key_{v}_across"] for s in SEEDS])}
    st["keys"] = keyres
    d1 = any(keyres[v]["clause1"] and keyres[v]["clause2"] for v in ("K4", "K3"))
    out["verdicts"]["E1-D1"] = {"pass": d1, "stats": {v: {k: keyres[v][k] for k in ("pearson", "gap", "spearman_levels_mean_curve",
                                                                                  "clause1", "clause2")} for v in ("K4", "K3")}}
    # ---- E1-D2.1 P-FullSupport
    r21 = {}
    ok = True
    for b in ("scaled_mix", "mmd", "kc"):
        ratio = ci([T[s][f"fs_jsd_{b}"] / T[s]["fs_jsd_attn_K4"] for s in SEEDS])
        djsd = ci([T[s][f"fs_jsd_{b}"] - T[s]["fs_jsd_attn_K4"] for s in SEEDS])
        dkl = ci([T[s][f"fs_kl_{b}"] - T[s]["fs_kl_attn_K4"] for s in SEEDS])
        r21[b] = {"jsd_ratio": ratio, "jsd_diff": djsd, "kl_diff": dkl}
        ok = ok and ratio["mean"] >= 2.5 and djsd["lo"] > 0 and dkl["lo"] > 0
    lp_ratio = ci([T[s]["fs_kl_looped"] / T[s]["fs_kl_attn_K4"] for s in SEEDS])
    params = S[101]["fit"]["params"]
    r21["looped_secondary"] = {"kl_ratio_to_attn": lp_ratio, "params_looped": params["looped"], "params_attn_K4": params["attn_K4"],
                               "pass": lp_ratio["mean"] <= 1.15 and params["looped"] < params["attn_K4"] / 5}
    out["verdicts"]["E1-D2.1"] = {"pass": ok, "stats": r21}
    # ---- E1-D2.2 P-ActivePUCT
    cov_p = ci([T[s]["ap_coverage_puct"] for s in SEEDS])
    cov_o = ci([T[s]["ap_coverage_open"] for s in SEEDS])
    jr = ci([T[s]["ap_open_jsd_attn_K4"] / T[s]["ap_puct_jsd_attn_K4"] for s in SEEDS])
    jd = ci([T[s]["ap_open_jsd_attn_K4"] - T[s]["ap_puct_jsd_attn_K4"] for s in SEEDS])
    kc_ = ci([1 - T[s]["ap_puct_kl_attn_K4"] / T[s]["ap_open_kl_attn_K4"] for s in SEEDS])
    w1 = ci([T[s]["ap_wave1_pref_share_NW"] for s in SEEDS])
    d22 = cov_p["mean"] >= 0.95 and jr["mean"] >= 1.8 and jd["lo"] > 0 and kc_["mean"] >= 0.20 and w1["mean"] >= 0.75
    out["verdicts"]["E1-D2.2"] = {"pass": d22, "stats": {"coverage_puct": cov_p, "coverage_open": cov_o, "jsd_ratio_open_over_puct": jr,
                                                         "jsd_diff_open_minus_puct": jd, "kl_cut_puct_vs_open": kc_,
                                                         "wave1_nw_share": w1,
                                                         "parts": {"coverage>=0.95": cov_p["mean"] >= 0.95, "jsd_ratio>=1.8": jr["mean"] >= 1.8,
                                                                   "jsd_ci>0": jd["lo"] > 0, "kl_cut>=0.20": kc_["mean"] >= 0.20,
                                                                   "wave1>=0.75": w1["mean"] >= 0.75}}}
    for a in ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K4"):
        for kind in ("open", "puct"):
            st[f"ap_{kind}_kl_{a}"] = ci([T[s][f"ap_{kind}_kl_{a}"] for s in SEEDS])
            st[f"ap_{kind}_jsd_{a}"] = ci([T[s][f"ap_{kind}_jsd_{a}"] for s in SEEDS])
    for kind in ("open", "puct"):
        st[f"ap_{kind}_jsd_union"] = ci([T[s][f"ap_{kind}_jsd_union"] for s in SEEDS])
        st[f"ap_{kind}_kl_union"] = ci([T[s][f"ap_{kind}_kl_union"] for s in SEEDS])
    # ---- E1-D3 controlled bags (attn_L3 with K4 keys)
    C = {s: T[s]["controlled"] for s in SEEDS}
    W = lambda s, t, N, v="K4": C[s][f"t{t}_N{N}"][f"W_{v}"]
    mse = lambda s, t, a: float(np.mean([C[s][f"t{t}_N{N}"][f"mse_{a}"] for N in NS]))
    r = {}
    for t in ("0.3", "1.5", "2.5"):
        r[f"W8/W1_t{t}"] = ci([W(s, t, 8) / W(s, t, 1) for s in SEEDS])
        r[f"W16/W1_t{t}"] = ci([W(s, t, 16) / W(s, t, 1) for s in SEEDS])
        r[f"W_t{t}"] = {N: ci([W(s, t, N) for s in SEEDS]) for N in NS}
        for b in ("mmd", "kc", "mix", "scaled_mix"):
            r[f"msecut_vs_{b}_t{t}"] = ci([1 - mse(s, t, "attn_K4") / mse(s, t, b) for s in SEEDS])
    c1 = r["W16/W1_t0.3"]["mean"] <= 1.25 and r["msecut_vs_mmd_t0.3"]["mean"] >= 0.35
    c2 = (r["W8/W1_t2.5"]["mean"] >= 2.20 and r["W8/W1_t2.5"]["lo"] > 1.90 and r["W16/W1_t2.5"]["mean"] >= 2.80
          and r["msecut_vs_kc_t2.5"]["mean"] >= 0.15)
    mono = [W(s, "2.5", 8) / W(s, "2.5", 1) > W(s, "1.5", 8) / W(s, "1.5", 1) > W(s, "0.3", 8) / W(s, "0.3", 1) for s in SEEDS]
    c3 = all(mono)
    r["monotone_noise_per_seed"] = mono
    out["verdicts"]["E1-D3"] = {"pass": c1 and c2 and c3, "parts": {"clean_dedup": c1, "high_noise_support": c2, "monotone": c3},
                                "stats": r}
    out["fit"] = {s: {k: S[s]["fit"][k] for k in ("eta", "u", "alpha", "beta_by_tau", "params", "heldout_bag_bce_to_ref",
                                                  "looped_eta", "looped_s", "ref_coverage_mean")} for s in SEEDS}
    out["compute"] = {s: {"train_sec": S[s]["train_sec"], "fit_sec": S[s]["fit"]["sec"], "test_sec": T[s]["sec"],
                          "total_sec": S[s]["total_sec"], "resumed": S[s].get("resumed_from_saved_theta_real")} for s in SEEDS}
    od = os.path.join(R6, "results", tag, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=1, default=float)
    lines = ["| rule | verdict | key statistics: 3-seed mean [seed-t 95% CI] (per seed) |", "|---|---|---|"]
    V = out["verdicts"]
    lines.append(f"| E1-D0 | {'PASS' if V['E1-D0']['pass'] else 'FAIL'} | clause-1 Pearson {fci(st['clause1_corr_std6'])}; "
                 f"P-FullSupport scaled_mix KL cut {fci(st['fs_cut_scaled_mix'])} |")
    for v in ("K4", "K3"):
        k = keyres[v]
        lines.append(f"| E1-D1 ({v}) | {'PASS' if (k['clause1'] and k['clause2']) else 'FAIL'} | Pearson {fci(k['pearson'])}; gap "
                     f"{fci(k['gap'])}; cos by overlap {', '.join(f'{x:.3f}' for x in k['cos_by_overlap_mean'])}; Spearman "
                     f"{k['spearman_levels_mean_curve']:.3f} |")
    lines.append(f"| E1-D1 (rule: K4 or K3) | {'PASS' if V['E1-D1']['pass'] else 'FAIL'} | |")
    lines.append(f"| E1-D2.1 | {'PASS' if V['E1-D2.1']['pass'] else 'FAIL'} | " + "; ".join(
        f"{b}: JSD ratio {fci(r21[b]['jsd_ratio'], 2)}, KL diff {fci(r21[b]['kl_diff'])}" for b in ("scaled_mix", "mmd", "kc"))
                 + f"; looped (secondary) KL ratio {fci(lp_ratio, 3)}, params {params['looped']} vs {params['attn_K4']} |")
    lines.append(f"| E1-D2.2 | {'PASS' if V['E1-D2.2']['pass'] else 'FAIL'} | coverage puct {fci(cov_p, 3)} vs open {fci(cov_o, 3)}; "
                 f"JSD ratio {fci(jr, 2)}; KL cut {fci(kc_, 3)}; wave-1 NW share {fci(w1, 3)} |")
    lines.append(f"| E1-D3 | {'PASS' if V['E1-D3']['pass'] else 'FAIL'} | W16/W1 tau0.3 {fci(r['W16/W1_t0.3'], 3)}; MSE cut vs mmd "
                 f"tau0.3 {fci(r['msecut_vs_mmd_t0.3'], 3)}; W8/W1 tau2.5 {fci(r['W8/W1_t2.5'], 3)}; W16/W1 tau2.5 "
                 f"{fci(r['W16/W1_t2.5'], 3)}; MSE cut vs kc tau2.5 {fci(r['msecut_vs_kc_t2.5'], 3)}; monotone per seed {mono} |")
    open(os.path.join(od, "verdicts.md"), "w").write("\n".join(lines) + "\n")
    plots(S, T, od, st, keyres, r)
    print("\n".join(lines))


def plots(S, T, od, st, keyres, r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["ink2"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                         "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": COL["surf"], "axes.facecolor": COL["surf"],
                         "lines.linewidth": 2.0, "legend.frameon": False})
    vc = {"K1": COL["s1"], "K2": COL["s2"], "K3": COL["s3"], "K4": COL["s4"], "looped": COL["gray"]}
    # keys by overlap
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    rows = []
    for v in VARS:
        ax[0].plot(range(len(LEVELS)), keyres[v]["cos_by_overlap_mean"], color=vc[v], marker="o", ms=4,
                   label=f"{v} (r {keyres[v]['pearson']['mean']:.2f})")
        rows.append([v] + keyres[v]["cos_by_overlap_mean"] + [keyres[v]["pearson"]["mean"], keyres[v]["gap"]["mean"],
                                                              keyres[v]["spearman_levels_mean_curve"]])
    ax[0].set_xticks(range(len(LEVELS)), [str(o) for o in LEVELS])
    ax[0].set(title="mean key cosine by shared cells", xlabel="cells shared by the two reads' patches")
    ax[0].legend(fontsize=7)
    xs = np.arange(len(VARS))
    ax[1].bar(xs - 0.2, [keyres[v]["pearson"]["mean"] for v in VARS], width=0.4, color=COL["s1"], label="Pearson with overlap")
    ax[1].bar(xs + 0.2, [keyres[v]["gap"]["mean"] for v in VARS], width=0.4, color=COL["s2"], label="same - across cosine")
    ax[1].axhline(0.60, color=COL["s1"], lw=1, ls="--")
    ax[1].axhline(0.50, color=COL["s2"], lw=1, ls=":")
    ax[1].set_xticks(xs, list(VARS))
    ax[1].set(title="key geometry (bars: 0.60 Pearson, 0.50 gap)")
    ax[1].set_axisbelow(True)
    ax[1].grid(axis="x", visible=False)
    ax[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(od, "keys.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "keys.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + [f"cos_overlap_{o}" for o in LEVELS] + ["pearson", "gap", "spearman_levels"])
        w.writerows(rows)
    # P-FullSupport and P-ActivePUCT
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    names = ["scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K4"]
    ax[0].bar(range(len(names)), [st[f"fs_jsd_{a}"]["mean"] for a in names], color=COL["s2"], width=0.6)
    ax[0].set_xticks(range(len(names)), [a.replace("attn_", "attn ") for a in names], fontsize=8)
    ax[0].set(title="P-FullSupport: JSD between U_NW and U_SE")
    ax[1].bar(range(len(names)), [st[f"fs_kl_{a}"]["mean"] for a in names], color=COL["s1"], width=0.6)
    ax[1].set_xticks(range(len(names)), [a.replace("attn_", "attn ") for a in names], fontsize=8)
    ax[1].set(title="P-FullSupport: KL to exact Bayes")
    kinds = ["open", "puct"]
    ax[2].bar(np.arange(2) - 0.2, [st[f"ap_{k}_jsd_attn_K4"]["mean"] for k in kinds], width=0.4, color=COL["s2"], label="JSD (attn K4)")
    ax2 = ax[2]
    ax2.bar(np.arange(2) + 0.2, [st[f"ap_{k}_kl_attn_K4"]["mean"] / 10 for k in kinds], width=0.4, color=COL["s1"],
            label="KL (attn K4) / 10")
    ax2.set_xticks(range(2), ["open_U", "puct_key_U"])
    ax2.set(title="P-ActivePUCT")
    ax2.legend(fontsize=7)
    for a in ax:
        a.set_axisbelow(True)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "policy_independence.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "policy_independence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "pool", "kl_mean", "jsd_mean"])
        for a in names + ["mix", "attn_K2", "attn_K3", "union"]:
            w.writerow(["fullsupport", a, st[f"fs_kl_{a}"]["mean"], st[f"fs_jsd_{a}"]["mean"]])
        for k in kinds:
            for a in ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K4", "union"):
                w.writerow([k, a, st[f"ap_{k}_kl_{a}"]["mean"], st[f"ap_{k}_jsd_{a}"]["mean"]])
    # controlled bags
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    tc = {"0.3": COL["s1"], "1.5": COL["s2"], "2.5": COL["s3"]}
    rows = []
    for t in ("0.3", "1.5", "2.5"):
        m = [r[f"W_t{t}"][N]["mean"] for N in NS]
        ax[0].plot(NS, m, color=tc[t], marker="o", ms=4, label=f"tau {t}")
        rows += [[t, N, r[f"W_t{t}"][N]["mean"]] for N in NS]
    ax[0].set(title="attn_L3 (K4) cluster weight W(N)", xlabel="N same-patch reads", xscale="log")
    ax[0].set_xticks(NS, [str(x) for x in NS])
    ax[0].legend(fontsize=8)
    ax[1].bar(np.arange(3) - 0.2, [r[f"W8/W1_t{t}"]["mean"] for t in ("0.3", "1.5", "2.5")], width=0.4, color=COL["s1"], label="W(8)/W(1)")
    ax[1].bar(np.arange(3) + 0.2, [r[f"W16/W1_t{t}"]["mean"] for t in ("0.3", "1.5", "2.5")], width=0.4, color=COL["s2"], label="W(16)/W(1)")
    ax[1].set_xticks(range(3), ["tau 0.3", "tau 1.5", "tau 2.5"])
    ax[1].set(title="weight growth by noise tier")
    ax[1].set_axisbelow(True)
    ax[1].grid(axis="x", visible=False)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "controlled_bags.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "controlled_bags.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "N", "W_attn_K4_mean"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
