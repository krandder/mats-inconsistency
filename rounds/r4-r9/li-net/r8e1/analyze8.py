#!/usr/bin/env python3
"""R8-E1 analysis: applies PREREG-R8-E1.md section 3 mechanically to results/<tag>/s<seed>/summary.json (3 seeds); writes
results/<tag>/analysis/{results.json, verdicts.md, *.csv, *.png}. Written before any main result existed.
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 analyze8.py main"""
import csv
import json
import math
import os
import sys

import numpy as np

R8 = os.environ.get("R8DIR", os.path.dirname(os.path.abspath(__file__)))
T975 = {2: 4.302652729911275, 1: 12.706204736174705}
SEEDS = (101, 102, 103)
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
POOLS = ("looped_attn", "attn_K3", "mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "econ_R7", "econ_D3", "looped_lin", "jacobi",
         "jacobi_prec", "attn_K1")
WPOOLS = ("looped_attn", "attn_K3", "attn_K1", "jacobi", "jacobi_prec", "econ_R7", "econ_D3", "kc_K1", "kc_K3", "mmd")
CELLS = ("K1_R6", "K1_R7", "K3_R6", "K3_R7")
COL = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4", "s6": "#7a5cc7", "gray": "#8a8984",
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
    S = {s: json.load(open(os.path.join(R8, "results", tag, f"s{s}", "summary.json"))) for s in SEEDS}
    T = {s: S[s]["test"] for s in SEEDS}
    C = {s: T[s]["controlled"] for s in SEEDS}
    out = {"tag": tag, "stats": {}, "verdicts": {}}
    st, V = out["stats"], out["verdicts"]

    # ---- E1-D0: checkpoint reload, world model, anchor identity at K = 0
    ck = {s: S[s]["checkpoint"] for s in SEEDS}
    st["clause1_corr_std6"] = ci([T[s]["clause1_corr_std6"] for s in SEEDS])
    idk = [T[s]["identity_K0"][f"looped_attn_M{M}_kl_diff"] for s in SEEDS for M in (2, 18)]
    idl = [T[s]["identity_K0"][f"looped_attn_M{M}_max_abs_logit_diff"] for s in SEEDS for M in (2, 18)]
    p0 = {"sha256_match_all_seeds": all(ck[s]["match"] and ck[s]["sha256"] == ck[s]["json_sha256"] for s in SEEDS),
          "pearson>=0.999": st["clause1_corr_std6"]["mean"] >= 0.999, "K0_kl_identity<1e-7": max(idk) < 1e-7}
    V["E1-D0"] = {"pass": all(p0.values()), "parts": p0,
                  "stats": {"sha256": {s: ck[s]["sha256"] for s in SEEDS}, "clause1_pearson": st["clause1_corr_std6"],
                            "K0_max_kl_diff": max(idk), "K0_max_abs_logit_diff": max(idl)}}

    # ---- curves
    for a in POOLS:
        for K in KS_FS:
            st[f"fs_kl_{a}_K{K}"] = ci([T[s][f"fs_kl_{a}_K{K}"] for s in SEEDS])
            st[f"fs_jsd_{a}_K{K}"] = ci([T[s][f"fs_jsd_{a}_K{K}"] for s in SEEDS])
    roots = sorted({k.split("_")[0] for k in T[101] if k.startswith("sr") and "_kl_" in k})   # sr6, sr12, sr18 (tau 0.3/1.5), sr18t3
    for M in roots:
        for a in POOLS:
            for K in KS_SR:
                st[f"{M}_kl_{a}_K{K}"] = ci([T[s][f"{M}_kl_{a}_K{K}"] for s in SEEDS])
                st[f"{M}_dkl_{a}_K{K}"] = ci([T[s][f"{M}_kl_{a}_K{K}"] - T[s][f"{M}_kl_{a}_K0"] for s in SEEDS])
                st[f"{M}_jsd_{a}_K{K}"] = ci([T[s][f"{M}_jsd_{a}_K{K}"] for s in SEEDS])
    st["kl_root"] = {M: ci([T[s]["kl_root"][M] for s in SEEDS]) for M in T[101]["kl_root"]}

    # ---- E1-D1.1: sparse-root anytime curve (looped_attn, M_0 = 2, P-FullSupport prefixes)
    la = "looped_attn"
    steps = {s: [T[s][f"fs_kl_{la}_K{KS_FS[i + 1]}"] - T[s][f"fs_kl_{la}_K{KS_FS[i]}"] for i in range(len(KS_FS) - 1)] for s in SEEDS}
    p11 = {"monotone_every_step_every_seed": all(d <= 0.0005 for s in SEEDS for d in steps[s]),
           "KL(32)<=0.033": st[f"fs_kl_{la}_K32"]["mean"] <= 0.033, "JSD(24)<=0.0075": st[f"fs_jsd_{la}_K24"]["mean"] <= 0.0075}
    V["E1-D1.1"] = {"pass": all(p11.values()), "parts": p11,
                    "stats": {"step_increase_per_seed": steps, "max_step_increase": max(d for s in SEEDS for d in steps[s]),
                              "KL_by_K": {K: st[f"fs_kl_{la}_K{K}"] for K in KS_FS}, "KL32": st[f"fs_kl_{la}_K32"],
                              "JSD24": st[f"fs_jsd_{la}_K24"]}}

    # ---- E1-D1.2: strong-root safety (M_0 = 18, reads tau 50/50 over 0.3 / 1.5: amended reading 6)
    M = 18
    kl0 = ci([T[s][f"sr{M}_kl_{la}_K0"] for s in SEEDS])
    mmd_deg = st[f"sr{M}_dkl_mmd_K16"]
    mix_deg = st[f"sr{M}_dkl_mix_K16"]
    maxdeg = {s: max(T[s][f"sr{M}_kl_{la}_K{K}"] - T[s][f"sr{M}_kl_{la}_K0"] for K in KS_SR[1:]) for s in SEEDS}
    gain16 = ci([T[s][f"sr{M}_kl_{la}_K0"] - T[s][f"sr{M}_kl_{la}_K16"] for s in SEEDS])
    vs_mmd = ci([T[s][f"sr{M}_kl_mmd_K16"] - T[s][f"sr{M}_kl_{la}_K16"] for s in SEEDS])
    p12 = {"KL(0)<=0.050": kl0["mean"] <= 0.050, "mmd_degrades>=0.015_ci>0": mmd_deg["mean"] >= 0.015 and mmd_deg["lo"] > 0,
           "looped_attn_never_degrades_every_seed": all(v <= 0.0005 for v in maxdeg.values()),
           "KL(0)-KL(16)>0_ci>0": gain16["mean"] > 0 and gain16["lo"] > 0, "mmd(16)-looped(16)>0_ci>0": vs_mmd["mean"] > 0 and vs_mmd["lo"] > 0}
    V["E1-D1.2"] = {"pass": all(p12.values()), "parts": p12,
                    "stats": {"KL0": kl0, "mmd_K16_minus_K0": mmd_deg, "mix_K16_minus_K0 (reported)": mix_deg,
                              "looped_attn_max_increase_per_seed": maxdeg, "looped_attn_K0_minus_K16": gain16,
                              "mmd_minus_looped_attn_K16": vs_mmd,
                              "tier3_reported": ({"looped_attn_max_increase_per_seed": {s: max(T[s][f"sr18t3_kl_{la}_K{K}"] - T[s][f"sr18t3_kl_{la}_K0"] for K in KS_SR[1:]) for s in SEEDS}, "mmd_K16_minus_K0": st["sr18t3_dkl_mmd_K16"], "mix_K16_minus_K0": st["sr18t3_dkl_mix_K16"]} if "sr18t3" in roots else None),
                              "looped_attn_KL_by_K": {K: st[f"sr{M}_kl_{la}_K{K}"] for K in KS_SR}}}
    V["E1-D1"] = {"pass": V["E1-D1.1"]["pass"] and V["E1-D1.2"]["pass"]}

    # ---- controlled bags
    W = lambda s, t, N, a: C[s][f"t{t}_N{N}"][f"W_{a}"]
    mse = lambda s, t, a: float(np.mean([C[s][f"t{t}_N{N}"][f"mse_{a}"] for N in NS]))
    mse_all = lambda s, a: float(np.mean([mse(s, t, a) for t in TAUS]))
    wr = {}
    for a in WPOOLS:
        for t in TAUS:
            wr[f"{a}_W8/W1_t{t}"] = ci([W(s, t, 8, a) / W(s, t, 1, a) for s in SEEDS])
            wr[f"{a}_W16/W1_t{t}"] = ci([W(s, t, 16, a) / W(s, t, 1, a) for s in SEEDS])
            wr[f"{a}_W_t{t}"] = {N: ci([W(s, t, N, a) for s in SEEDS]) for N in NS}
    st["weights"] = wr
    st["mse"] = {f"{a}_t{t}": ci([mse(s, t, a) for s in SEEDS]) for a in POOLS for t in TAUS}
    st["mse_all_tiers"] = {a: ci([mse_all(s, a) for s in SEEDS]) for a in POOLS}
    mono = lambda a: [W(s, "0.3", 8, a) / W(s, "0.3", 1, a) < W(s, "1.5", 8, a) / W(s, "1.5", 1, a)
                      < W(s, "2.5", 8, a) / W(s, "2.5", 1, a) for s in SEEDS]

    # ---- E1-D2.1: jacobi_lin_T4_prec
    jp = "jacobi_prec"
    fsj = st[f"fs_kl_{jp}_K24"]
    beat = {}
    for b in ("looped_lin", "kc_K1"):
        d = ci([T[s][f"fs_kl_{b}_K24"] - T[s][f"fs_kl_{jp}_K24"] for s in SEEDS])
        every = all(T[s][f"fs_kl_{jp}_K24"] < T[s][f"fs_kl_{b}_K24"] for s in SEEDS)
        beat[b] = {"kl_diff": d, "lower_every_seed": every, "pass": every and d["lo"] > 0}
    jm = mono(jp)
    p21 = {"FS_KL<=0.040": fsj["mean"] <= 0.040, "beats_looped_lin": beat["looped_lin"]["pass"], "beats_kc_K1": beat["kc_K1"]["pass"],
           "W16/W1_t0.3<=1.25": wr[f"{jp}_W16/W1_t0.3"]["mean"] <= 1.25, "W16/W1_t0.3_ci_hi<=1.30": wr[f"{jp}_W16/W1_t0.3"]["hi"] <= 1.30,
           "W8/W1_monotone_every_seed": all(jm), "W16/W1_t2.5>=2.0": wr[f"{jp}_W16/W1_t2.5"]["mean"] >= 2.0}
    V["E1-D2.1"] = {"pass": all(p21.values()), "parts": p21,
                    "stats": {"fs_kl_K24": fsj, "beats": beat, "W16/W1_t0.3": wr[f"{jp}_W16/W1_t0.3"], "W16/W1_t2.5": wr[f"{jp}_W16/W1_t2.5"],
                              "W8/W1_by_tau": {t: wr[f"{jp}_W8/W1_t{t}"] for t in TAUS}, "monotone_per_seed": jm,
                              "R7_jacobi_same_run_W16/W1_t0.3": wr["jacobi_W16/W1_t0.3"]}}

    # ---- E1-D2.2: econ_kc_K3 fitted on mixed-tier D3 probes; 2x2 PUCT table
    ed = "econ_D3"
    dm = ci([mse_all(s, "kc_K3") - mse_all(s, ed) for s in SEEDS])
    for cell in CELLS:
        st[f"puct_{cell}_coverage"] = ci([T[s][f"puct_{cell}_coverage"] for s in SEEDS])
        st[f"puct_{cell}_wave1_pref_share_NW"] = ci([T[s][f"puct_{cell}_wave1_pref_share_NW"] for s in SEEDS])
        st[f"puct_{cell}_wave1_strict_share_NW"] = ci([T[s][f"puct_{cell}_wave1_strict_share_NW"] for s in SEEDS])
        for a in ("attn_K3", "looped_attn", "union"):
            st[f"puct_{cell}_kl_{a}"] = ci([T[s][f"puct_{cell}_kl_{a}"] for s in SEEDS])
            st[f"puct_{cell}_jsd_{a}"] = ci([T[s][f"puct_{cell}_jsd_{a}"] for s in SEEDS])
    reported = all(f"puct_{c}_coverage" in st for c in CELLS)
    p22 = {"W16/W1_t0.3<=1.25": wr[f"{ed}_W16/W1_t0.3"]["mean"] <= 1.25, "W16/W1_t2.5>=2.2": wr[f"{ed}_W16/W1_t2.5"]["mean"] >= 2.2,
           "D3_MSE_below_kc_K3_ci>0": dm["lo"] > 0, "PUCT_2x2_reported": reported}
    V["E1-D2.2"] = {"pass": all(p22.values()), "parts": p22,
                    "stats": {"W16/W1_t0.3": wr[f"{ed}_W16/W1_t0.3"], "W16/W1_t2.5": wr[f"{ed}_W16/W1_t2.5"],
                              "mse_kc_K3_minus_econ_D3_all_tiers": dm,
                              "per_tier_mse_diff": {t: ci([mse(s, t, "kc_K3") - mse(s, t, ed) for s in SEEDS]) for t in TAUS},
                              "econ_R7_W16/W1_t0.3": wr["econ_R7_W16/W1_t0.3"], "econ_R7_W16/W1_t2.5": wr["econ_R7_W16/W1_t2.5"],
                              "puct_coverage": {c: st[f"puct_{c}_coverage"] for c in CELLS}}}
    V["E1-D2"] = {"pass": V["E1-D2.1"]["pass"] and V["E1-D2.2"]["pass"]}
    if not V["E1-D0"]["pass"]:
        for k in V:
            if k != "E1-D0":
                V[k]["note"] = "gate failed: descriptive"
    out["fit"] = {s: {k: S[s]["fit"][k] for k in ("scalars", "beta_by_tau", "beta_clean", "beta_by_root", "params", "heldout_bag_bce_to_ref",
                                                  "heldout_any_bce_to_ref", "jacobi", "jacobi_prec", "trust_heldout_mse")} for s in SEEDS}
    out["compute"] = {s: {"fit_sec": S[s]["fit"]["sec"], "test_sec": T[s]["sec"], "total_sec": S[s]["total_sec"]} for s in SEEDS}
    od = os.path.join(R8, "results", tag, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=1, default=float)
    P_ = lambda k: "PASS" if V[k]["pass"] else "FAIL"
    L = ["| rule | verdict | key statistics: 3-seed mean [seed-t 95% CI] (per seed) |", "|---|---|---|",
         f"| E1-D0 | {P_('E1-D0')} | sha256 match {p0['sha256_match_all_seeds']}; Pearson {fci(st['clause1_corr_std6'])}; K = 0 max KL diff "
         f"{max(idk):.2e}, max logit diff {max(idl):.2e} |",
         f"| E1-D1.1 | {P_('E1-D1.1')} | looped_attn_T3 KL by K " + ", ".join(f"{K}: {st[f'fs_kl_{la}_K{K}']['mean']:.4f}" for K in KS_FS)
         + f"; max step increase {V['E1-D1.1']['stats']['max_step_increase']:.5f}; KL(32) {fci(st[f'fs_kl_{la}_K32'])}; JSD(24) "
           f"{fci(st[f'fs_jsd_{la}_K24'], 5)}; parts {p11} |",
         f"| E1-D1.2 | {P_('E1-D1.2')} | KL(0) {fci(kl0)}; mmd KL(16)-KL(0) {fci(mmd_deg)}; mix {fci(mix_deg)}; looped_attn max increase per "
         f"seed {[round(v, 5) for v in maxdeg.values()]}; KL(0)-KL(16) {fci(gain16)}; mmd(16)-looped(16) {fci(vs_mmd)}; parts {p12} |",
         f"| E1-D1 (both) | {P_('E1-D1')} | |",
         f"| E1-D2.1 | {P_('E1-D2.1')} | FS KL {fci(fsj)}; vs looped_lin {fci(beat['looped_lin']['kl_diff'])}; vs kc_K1 "
         f"{fci(beat['kc_K1']['kl_diff'])}; W16/W1 tau0.3 {fci(wr[f'{jp}_W16/W1_t0.3'], 3)}; W8/W1 by tau "
         + ", ".join(fci(wr[f'{jp}_W8/W1_t{t}'], 3) for t in TAUS) + f"; W16/W1 tau2.5 {fci(wr[f'{jp}_W16/W1_t2.5'], 3)}; monotone {jm}; "
           f"parts {p21} |",
         f"| E1-D2.2 | {P_('E1-D2.2')} | econ_D3 W16/W1 tau0.3 {fci(wr[f'{ed}_W16/W1_t0.3'], 3)}, tau2.5 {fci(wr[f'{ed}_W16/W1_t2.5'], 3)}; "
         f"MSE(kc_K3)-MSE(econ_D3) {fci(dm, 6)}; PUCT coverage " + ", ".join(f"{c} {st[f'puct_{c}_coverage']['mean']:.4f}" for c in CELLS)
         + f"; parts {p22} |",
         f"| E1-D2 (both) | {P_('E1-D2')} | |"]
    open(os.path.join(od, "verdicts.md"), "w").write("\n".join(L) + "\n")
    plots(st, wr, roots, od)
    print("\n".join(L))


def plots(st, wr, roots, od):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["ink2"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                         "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": COL["surf"], "axes.facecolor": COL["surf"],
                         "lines.linewidth": 2.0, "legend.frameon": False})
    pc = {"looped_attn": COL["s1"], "attn_K3": COL["s3"], "mmd": COL["s2"], "mix": COL["gray"], "scaled_mix": COL["s4"],
          "jacobi_prec": COL["s6"], "kc_K3": COL["s5"]}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    xs = list(range(len(KS_FS)))
    rows = []
    for a in ("looped_attn", "attn_K3", "jacobi_prec", "kc_K3", "scaled_mix", "mmd"):
        ax[0].plot(xs, [st[f"fs_kl_{a}_K{K}"]["mean"] for K in KS_FS], color=pc[a], marker="o", ms=3, label=a)
        rows += [["sparse_root_M2", a, K, st[f"fs_kl_{a}_K{K}"]["mean"], st[f"fs_jsd_{a}_K{K}"]["mean"]] for K in KS_FS]
    ax[0].set_xticks(xs, [str(K) for K in KS_FS])
    ax[0].set(title="sparse root (2 anchors): KL to exact Bayes", xlabel="reads K (P-FullSupport prefix)")
    ax[0].legend(fontsize=7)
    xs2 = list(range(len(KS_SR)))
    if "sr18" in roots:
        for a in ("looped_attn", "attn_K3", "jacobi_prec", "scaled_mix", "mmd", "mix"):
            ax[1].plot(xs2, [st[f"sr18_dkl_{a}_K{K}"]["mean"] for K in KS_SR], color=pc[a], marker="o", ms=3, label=a)
        ax[1].axhline(0, color=COL["ink2"], lw=1)
        ax[1].set_xticks(xs2, [str(K) for K in KS_SR])
        ax[1].set(title="strong root (18 anchors): KL(K) - KL(0)", xlabel="reads K (90/10 skew, tau 0.3/1.5)")
        ax[1].legend(fontsize=7)
    for M in roots:
        for a in POOLS:
            rows += [[f"strong_root_{M}", a, K, st[f"{M}_kl_{a}_K{K}"]["mean"], st[f"{M}_jsd_{a}_K{K}"]["mean"]] for K in KS_SR]
    fig.tight_layout()
    fig.savefig(os.path.join(od, "anytime.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "anytime_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "pool", "K", "kl_mean", "jsd_mean"])
        w.writerows(rows)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    tc = {"0.3": COL["s1"], "1.5": COL["s2"], "2.5": COL["s3"]}
    rows = []
    for i, a in enumerate(("jacobi_prec", "econ_D3")):
        for t in TAUS:
            ax[i].plot(NS, [wr[f"{a}_W_t{t}"][N]["mean"] for N in NS], color=tc[t], marker="o", ms=4, label=f"tau {t}")
        ax[i].set(title=f"{a} cluster weight W(N)", xlabel="N same-patch reads", xscale="log")
        ax[i].set_xticks(NS, [str(x) for x in NS])
        ax[i].legend(fontsize=8)
    bs = ["jacobi", "jacobi_prec", "econ_R7", "econ_D3", "kc_K3", "looped_attn"]
    for j, t in enumerate(("0.3", "2.5")):
        ax[2].bar(np.arange(len(bs)) + (j - 0.5) * 0.38, [wr[f"{b}_W16/W1_t{t}"]["mean"] for b in bs], width=0.38, color=tc[t],
                  label=f"tau {t}")
    ax[2].axhline(1.25, color=COL["s1"], lw=1, ls="--")
    ax[2].set_xticks(range(len(bs)), bs, rotation=40, ha="right", fontsize=8)
    ax[2].set(title="W(16)/W(1) (bar: 1.25 at tau 0.3)")
    ax[2].legend(fontsize=8)
    ax[2].set_axisbelow(True)
    ax[2].grid(axis="x", visible=False)
    for a in WPOOLS:
        for t in TAUS:
            rows += [[a, t, N, wr[f"{a}_W_t{t}"][N]["mean"]] for N in NS]
    fig.tight_layout()
    fig.savefig(os.path.join(od, "d3.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "d3_weights.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "tau", "N", "W_mean"])
        w.writerows(rows)
    with open(os.path.join(od, "d3_mse.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pool", "tau", "mse_mean", "mse_lo", "mse_hi"])
        for a in POOLS:
            for t in TAUS:
                c = st["mse"][f"{a}_t{t}"]
                w.writerow([a, t, c["mean"], c["lo"], c["hi"]])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(range(4), [st[f"puct_{c}_coverage"]["mean"] for c in CELLS], color=[COL["s2"], COL["s4"], COL["s1"], COL["s3"]], width=0.6)
    ax[0].set_ylim(0.85, 1.0)
    ax[0].set_xticks(range(4), ["K1 keys\nR6 score", "K1 keys\nR7 score", "K3 keys\nR6 score", "K3 keys\nR7 score"], fontsize=8)
    ax[0].set(title="PUCT coverage after 4 waves")
    ax[1].bar(range(4), [st[f"puct_{c}_jsd_attn_K3"]["mean"] for c in CELLS], color=[COL["s2"], COL["s4"], COL["s1"], COL["s3"]], width=0.6)
    ax[1].set_xticks(range(4), ["K1, R6", "K1, R7", "K3, R6", "K3, R7"], fontsize=8)
    ax[1].set(title="attn_L3-K3 JSD between U_NW and U_SE")
    for a in ax:
        a.set_axisbelow(True)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "puct2x2.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(od, "puct2x2.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "coverage", "wave1_pref_share_NW", "wave1_strict_share_NW", "jsd_attn_K3", "kl_attn_K3", "jsd_looped_attn",
                    "kl_looped_attn", "jsd_union"])
        for c in CELLS:
            w.writerow([c, st[f"puct_{c}_coverage"]["mean"], st[f"puct_{c}_wave1_pref_share_NW"]["mean"],
                        st[f"puct_{c}_wave1_strict_share_NW"]["mean"], st[f"puct_{c}_jsd_attn_K3"]["mean"], st[f"puct_{c}_kl_attn_K3"]["mean"],
                        st[f"puct_{c}_jsd_looped_attn"]["mean"], st[f"puct_{c}_kl_looped_attn"]["mean"], st[f"puct_{c}_jsd_union"]["mean"]])


if __name__ == "__main__":
    main()
