#!/usr/bin/env python3
"""R9-E1 analysis: applies PREREG-R9-E1.md section 3 mechanically to results/<tag>/s<seed>/summary.json (3 seeds); writes
results/<tag>/analysis/{results.json, verdicts.md, *.csv, *.png}. Written before any main result existed.
Usage: PYTHONPATH=/tmp/r4e3a/pylib python3 analyze9.py main"""
import csv
import json
import math
import os
import sys

import numpy as np

R9 = os.environ.get("R9DIR", os.path.dirname(os.path.abspath(__file__)))
T975 = {2: 4.302652729911275, 1: 12.706204736174705}
SEEDS = tuple(int(x) for x in os.environ.get("R9SEEDS", "101,102,103").split(","))
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
SR = (18, 30, 32, 34)
POOLS = ("looped_attn", "attn_K3", "mix", "scaled_mix", "mmd", "kc_K1", "kc_K3", "econ_R7", "econ_D3", "looped_lin", "jacobi",
         "jacobi_prec", "attn_K1", "econ_prec")
WPOOLS = ("looped_attn", "attn_K3", "attn_K1", "jacobi", "jacobi_prec", "econ_R7", "econ_D3", "kc_K1", "kc_K3", "mmd", "econ_prec")
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
    S = {s: json.load(open(os.path.join(R9, "results", tag, f"s{s}", "summary.json"))) for s in SEEDS}
    T = {s: S[s]["test"] for s in SEEDS}
    C = {s: T[s]["controlled"] for s in SEEDS}
    Ms = {s: int(S[s]["fit"]["M_strong"]) for s in SEEDS}
    out = {"tag": tag, "stats": {}, "verdicts": {}, "M_strong": Ms}
    st, V = out["stats"], out["verdicts"]
    la = "looped_attn"

    # ---- E1-D0: checkpoint reload and Pearson; anchor identity at K = 0 at M_0 = 2 and M_0_strong (both anytime pools)
    ck = {s: S[s]["checkpoint"] for s in SEEDS}
    st["clause1_corr_std6"] = ci([T[s]["clause1_corr_std6"] for s in SEEDS])
    idk = [T[s]["identity_K0"][f"{a}_M{M}_kl_diff"] for s in SEEDS for a in ("looped_attn", "attn_K3") for M in (2, Ms[s])]
    idl = [T[s]["identity_K0"][f"{a}_M{M}_max_abs_logit_diff"] for s in SEEDS for a in ("looped_attn", "attn_K3") for M in (2, Ms[s])]
    idk_all = max(v for s in SEEDS for k, v in T[s]["identity_K0"].items() if k.endswith("kl_diff"))
    p0 = {"sha256_match_all_seeds": all(ck[s]["match"] and ck[s]["sha256"] == ck[s]["json_sha256"] for s in SEEDS),
          "pearson>=0.999": st["clause1_corr_std6"]["mean"] >= 0.999, "K0_kl_identity<1e-7": max(idk) < 1e-7}
    V["E1-D0"] = {"pass": all(p0.values()), "parts": p0,
                  "stats": {"sha256": {s: ck[s]["sha256"] for s in SEEDS}, "clause1_pearson": st["clause1_corr_std6"],
                            "K0_max_kl_diff": max(idk), "K0_max_abs_logit_diff": max(idl), "K0_max_kl_diff_every_M0 (reported)": idk_all,
                            "M_strong": Ms}}

    # ---- curves
    for a in POOLS:
        for K in KS_FS:
            st[f"fs_kl_{a}_K{K}"] = ci([T[s][f"fs_kl_{a}_K{K}"] for s in SEEDS])
            st[f"fs_jsd_{a}_K{K}"] = ci([T[s][f"fs_jsd_{a}_K{K}"] for s in SEEDS])
    for md in ("A", "B"):
        for M in SR:
            tg = f"sr{md}{M}"
            for a in POOLS:
                for K in KS_SR:
                    st[f"{tg}_kl_{a}_K{K}"] = ci([T[s][f"{tg}_kl_{a}_K{K}"] for s in SEEDS])
                    st[f"{tg}_dkl_{a}_K{K}"] = ci([T[s][f"{tg}_kl_{a}_K{K}"] - T[s][f"{tg}_kl_{a}_K0"] for s in SEEDS])
                    st[f"{tg}_jsd_{a}_K{K}"] = ci([T[s][f"{tg}_jsd_{a}_K{K}"] for s in SEEDS])
        for a in POOLS:                                                 # each seed at its own M_0_strong ("S")
            for K in KS_SR:
                st[f"srS{md}_kl_{a}_K{K}"] = ci([T[s][f"sr{md}{Ms[s]}_kl_{a}_K{K}"] for s in SEEDS])
                st[f"srS{md}_dkl_{a}_K{K}"] = ci([T[s][f"sr{md}{Ms[s]}_kl_{a}_K{K}"] - T[s][f"sr{md}{Ms[s]}_kl_{a}_K0"] for s in SEEDS])
                st[f"srS{md}_jsd_{a}_K{K}"] = ci([T[s][f"sr{md}{Ms[s]}_jsd_{a}_K{K}"] for s in SEEDS])
    st["kl_root"] = {M: ci([T[s]["kl_root"][M] for s in SEEDS]) for M in T[SEEDS[0]]["kl_root"]}
    st["strong_selection"] = {s: S[s]["fit"]["strong_selection"] for s in SEEDS}

    # ---- E1-D1.1: sparse-root anytime curve (looped_attn, M_0 = 2, interleaved P-FullSupport prefixes), as R8
    steps = {s: [T[s][f"fs_kl_{la}_K{KS_FS[i + 1]}"] - T[s][f"fs_kl_{la}_K{KS_FS[i]}"] for i in range(len(KS_FS) - 1)] for s in SEEDS}
    p11 = {"monotone_every_step_every_seed": all(d <= 0.0005 for s in SEEDS for d in steps[s]),
           "KL(32)<=0.033": st[f"fs_kl_{la}_K32"]["mean"] <= 0.033, "JSD(24)<=0.0075": st[f"fs_jsd_{la}_K24"]["mean"] <= 0.0075}
    V["E1-D1.1"] = {"pass": all(p11.values()), "parts": p11,
                    "stats": {"step_increase_per_seed": steps, "max_step_increase": max(d for s in SEEDS for d in steps[s]),
                              "KL_by_K": {K: st[f"fs_kl_{la}_K{K}"] for K in KS_FS}, "KL32": st[f"fs_kl_{la}_K32"],
                              "JSD24": st[f"fs_jsd_{la}_K24"]}}

    # ---- E1-D1.2: genuine strong-root safety at each seed's M_0_strong
    kl0 = ci([T[s][f"srA{Ms[s]}_kl_{la}_K0"] for s in SEEDS])
    deg = {md: {a: st[f"srS{md}_dkl_{a}_K16"] for a in ("mix", "mmd")} for md in ("A", "B")}
    p12 = {"KL(0)<=0.050": kl0["mean"] <= 0.050}
    for a in ("mix", "mmd"):
        p12[f"modeB_{a}_degrades_K16>=0.015_ci>0"] = deg["B"][a]["mean"] >= 0.015 and deg["B"][a]["lo"] > 0
    c3 = {}
    for md in ("A", "B"):
        tg = lambda s: f"sr{md}{Ms[s]}"
        maxdeg = {s: max(T[s][f"{tg(s)}_kl_{la}_K{K}"] - T[s][f"{tg(s)}_kl_{la}_K0"] for K in KS_SR[1:]) for s in SEEDS}
        gain16 = ci([T[s][f"{tg(s)}_kl_{la}_K0"] - T[s][f"{tg(s)}_kl_{la}_K16"] for s in SEEDS])
        beats = {f"{b}_K{K}": ci([T[s][f"{tg(s)}_kl_{b}_K{K}"] - T[s][f"{tg(s)}_kl_{la}_K{K}"] for s in SEEDS])
                 for b in ("mmd", "scaled_mix") for K in (16, 24)}
        c3[md] = {"looped_attn_max_increase_per_seed": maxdeg, "looped_attn_K0_minus_K16": gain16, "beats": beats}
        p12[f"mode{md}_looped_attn_never_degrades_every_seed"] = all(v <= 0.0005 for v in maxdeg.values())
        p12[f"mode{md}_KL(0)-KL(16)>0_ci>0"] = gain16["mean"] > 0 and gain16["lo"] > 0
        for k_, d_ in beats.items():
            p12[f"mode{md}_{k_}_minus_looped_ci>0"] = d_["mean"] > 0 and d_["lo"] > 0
    rep = {md: {"attn_K3_max_increase_per_seed": {s: max(T[s][f"sr{md}{Ms[s]}_kl_attn_K3_K{K}"] - T[s][f"sr{md}{Ms[s]}_kl_attn_K3_K0"]
                                                          for K in KS_SR[1:]) for s in SEEDS},
                "scaled_mix_K16_minus_K0": st[f"srS{md}_dkl_scaled_mix_K16"]} for md in ("A", "B")}
    V["E1-D1.2"] = {"pass": all(p12.values()), "parts": p12,
                    "stats": {"M_strong": Ms, "KL0": kl0, "K16_minus_K0": deg, "clause3": c3, "reported": rep,
                              "looped_attn_KL_by_K": {md: {K: st[f"srS{md}_kl_{la}_K{K}"] for K in KS_SR} for md in ("A", "B")}}}
    V["E1-D1"] = {"pass": V["E1-D1.1"]["pass"] and V["E1-D1.2"]["pass"]}

    # ---- controlled bags (D3)
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

    # ---- E1-D2.1: econ_prec_K3 (3 scalars) on D3 controlled bags
    ep = "econ_prec"
    em = mono(ep)
    dm = {b: ci([mse_all(s, b) - mse_all(s, ep) for s in SEEDS]) for b in ("kc_K3", "econ_D3")}
    p21 = {"W16/W1_t0.3<=1.25": wr[f"{ep}_W16/W1_t0.3"]["mean"] <= 1.25, "W16/W1_t0.3_ci_hi<=1.30": wr[f"{ep}_W16/W1_t0.3"]["hi"] <= 1.30,
           "W8/W1_monotone_every_seed": all(em), "W16/W1_t2.5>=2.20": wr[f"{ep}_W16/W1_t2.5"]["mean"] >= 2.20,
           "D3_MSE_below_kc_K3_ci>0": dm["kc_K3"]["lo"] > 0, "D3_MSE_below_econ_kc_K3_ci>0": dm["econ_D3"]["lo"] > 0}
    V["E1-D2.1"] = {"pass": all(p21.values()), "parts": p21,
                    "stats": {"W16/W1_t0.3": wr[f"{ep}_W16/W1_t0.3"], "W16/W1_t2.5": wr[f"{ep}_W16/W1_t2.5"],
                              "W8/W1_by_tau": {t: wr[f"{ep}_W8/W1_t{t}"] for t in TAUS}, "monotone_per_seed": em,
                              "mse_minus_econ_prec_all_tiers": dm,
                              "per_tier_mse_diff": {b: {t: ci([mse(s, t, b) - mse(s, t, ep) for s in SEEDS]) for t in TAUS}
                                                    for b in ("kc_K3", "econ_D3")},
                              "econ_D3_W16/W1_t0.3 (reported)": wr["econ_D3_W16/W1_t0.3"],
                              "econ_D3_W16/W1_t2.5 (reported)": wr["econ_D3_W16/W1_t2.5"]}}

    # ---- E1-D2.2: jacobi_lin_T4_prec weight law, P-FullSupport KL (K = 24) and the 4 comparisons
    jp = "jacobi_prec"
    jm = mono(jp)
    fsj = st[f"fs_kl_{jp}_K24"]
    beat = {}
    for b in ("looped_lin", "kc_K1", "kc_K3", "econ_prec"):
        d = ci([T[s][f"fs_kl_{b}_K24"] - T[s][f"fs_kl_{jp}_K24"] for s in SEEDS])
        beat[b] = {"kl_diff": d, "lower_every_seed (reported)": all(T[s][f"fs_kl_{jp}_K24"] < T[s][f"fs_kl_{b}_K24"] for s in SEEDS),
                   "pass": d["lo"] > 0}
    p22 = {"W16/W1_t0.3<=1.25": wr[f"{jp}_W16/W1_t0.3"]["mean"] <= 1.25, "W8/W1_monotone_every_seed": all(jm),
           "W16/W1_t2.5>=2.20": wr[f"{jp}_W16/W1_t2.5"]["mean"] >= 2.20, "FS_KL_K24<=0.0420": fsj["mean"] <= 0.0420}
    for b in beat:
        p22[f"beats_{b}_ci>0"] = beat[b]["pass"]
    V["E1-D2.2"] = {"pass": all(p22.values()), "parts": p22,
                    "stats": {"fs_kl_K24": fsj, "beats": beat, "W16/W1_t0.3": wr[f"{jp}_W16/W1_t0.3"], "W16/W1_t2.5": wr[f"{jp}_W16/W1_t2.5"],
                              "W8/W1_by_tau": {t: wr[f"{jp}_W8/W1_t{t}"] for t in TAUS}, "monotone_per_seed": jm,
                              "R7_jacobi_same_run_fs_kl_K24 (reported)": st["fs_kl_jacobi_K24"]}}
    V["E1-D2"] = {"pass": V["E1-D2.1"]["pass"] and V["E1-D2.2"]["pass"]}
    if not V["E1-D0"]["pass"]:
        for k in V:
            if k != "E1-D0":
                V[k]["note"] = "gate failed: descriptive"
    out["fit"] = {s: {k: S[s]["fit"][k] for k in ("M_strong", "strong_selection", "scalars", "beta_by_tau", "beta_clean", "beta_by_root",
                                                  "params", "heldout_bag_bce_to_ref", "heldout_any_bce_to_ref", "heldout_jacobi_econ",
                                                  "jacobi", "jacobi_prec", "econ_prec", "trust_heldout_mse")} for s in SEEDS}
    out["compute"] = {s: {"fit_sec": S[s]["fit"]["sec"], "test_sec": T[s]["sec"], "total_sec": S[s]["total_sec"]} for s in SEEDS}
    od = os.path.join(R9, "results", tag, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "results.json"), "w"), indent=1, default=float)
    P_ = lambda k: "PASS" if V[k]["pass"] else "FAIL"
    c3s = lambda md: (f"mode {md}: looped_attn max increase per seed {[round(v, 5) for v in c3[md]['looped_attn_max_increase_per_seed'].values()]}"
                      f"; KL(0)-KL(16) {fci(c3[md]['looped_attn_K0_minus_K16'])}; "
                      + "; ".join(f"{k_}-looped {fci(d_)}" for k_, d_ in c3[md]["beats"].items()))
    L = ["| rule | verdict | key statistics: 3-seed mean [seed-t 95% CI] (per seed) |", "|---|---|---|",
         f"| E1-D0 | {P_('E1-D0')} | sha256 match {p0['sha256_match_all_seeds']}; Pearson {fci(st['clause1_corr_std6'])}; K = 0 max KL diff "
         f"{max(idk):.2e}, max logit diff {max(idl):.2e} (M_0 = 2 and M_0_strong {list(Ms.values())}) |",
         f"| E1-D1.1 | {P_('E1-D1.1')} | looped_attn_T3 KL by K " + ", ".join(f"{K}: {st[f'fs_kl_{la}_K{K}']['mean']:.4f}" for K in KS_FS)
         + f"; max step increase {V['E1-D1.1']['stats']['max_step_increase']:.5f}; KL(32) {fci(st[f'fs_kl_{la}_K32'])}; JSD(24) "
           f"{fci(st[f'fs_jsd_{la}_K24'], 5)}; parts {p11} |",
         f"| E1-D1.2 | {P_('E1-D1.2')} | M_0_strong {list(Ms.values())}; KL(0) {fci(kl0)}; mode B KL(16)-KL(0): mix {fci(deg['B']['mix'])}, "
         f"mmd {fci(deg['B']['mmd'])}; mode A (reported): mix {fci(deg['A']['mix'])}, mmd {fci(deg['A']['mmd'])}; {c3s('A')}; {c3s('B')}; "
         f"parts {p12} |",
         f"| E1-D1 (both) | {P_('E1-D1')} | |",
         f"| E1-D2.1 | {P_('E1-D2.1')} | econ_prec W16/W1 tau0.3 {fci(wr[f'{ep}_W16/W1_t0.3'], 3)}; W8/W1 by tau "
         + ", ".join(fci(wr[f'{ep}_W8/W1_t{t}'], 3) for t in TAUS) + f"; monotone {em}; W16/W1 tau2.5 {fci(wr[f'{ep}_W16/W1_t2.5'], 3)}; "
           f"MSE(kc_K3)-MSE(econ_prec) {fci(dm['kc_K3'], 6)}; MSE(econ_kc_K3)-MSE(econ_prec) {fci(dm['econ_D3'], 6)}; parts {p21} |",
         f"| E1-D2.2 | {P_('E1-D2.2')} | jacobi_prec W16/W1 tau0.3 {fci(wr[f'{jp}_W16/W1_t0.3'], 3)}; W8/W1 by tau "
         + ", ".join(fci(wr[f'{jp}_W8/W1_t{t}'], 3) for t in TAUS) + f"; monotone {jm}; W16/W1 tau2.5 {fci(wr[f'{jp}_W16/W1_t2.5'], 3)}; "
           f"FS KL(24) {fci(fsj)}; " + "; ".join(f"{b}-jacobi_prec {fci(beat[b]['kl_diff'])}" for b in beat) + f"; parts {p22} |",
         f"| E1-D2 (both) | {P_('E1-D2')} | |"]
    open(os.path.join(od, "verdicts.md"), "w").write("\n".join(L) + "\n")
    plots(st, wr, od)
    print("\n".join(L))


def plots(st, wr, od):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": COL["ink2"], "xtick.color": COL["ink2"], "ytick.color": COL["ink2"],
                         "axes.grid": True, "grid.color": COL["grid"], "grid.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": COL["surf"], "axes.facecolor": COL["surf"],
                         "lines.linewidth": 2.0, "legend.frameon": False})
    pc = {"looped_attn": COL["s1"], "attn_K3": COL["s3"], "mmd": COL["s2"], "mix": COL["gray"], "scaled_mix": COL["s4"],
          "jacobi_prec": COL["s6"], "kc_K3": COL["s5"], "econ_prec": COL["ink2"]}
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9))
    xs = list(range(len(KS_FS)))
    rows = []
    for a in ("looped_attn", "attn_K3", "jacobi_prec", "econ_prec", "scaled_mix", "mmd"):
        ax[0].plot(xs, [st[f"fs_kl_{a}_K{K}"]["mean"] for K in KS_FS], color=pc[a], marker="o", ms=3, label=a)
    for a in POOLS:
        rows += [["sparse_root_M2", a, K, st[f"fs_kl_{a}_K{K}"]["mean"], st[f"fs_jsd_{a}_K{K}"]["mean"]] for K in KS_FS]
    ax[0].set_xticks(xs, [str(K) for K in KS_FS])
    ax[0].set(title="sparse root (2 anchors): KL to exact Bayes", xlabel="reads K (P-FullSupport prefix)")
    ax[0].legend(fontsize=7)
    xs2 = list(range(len(KS_SR)))
    for i, (md, lab) in enumerate((("A", "mode A: cumulative-context reads, 3 tiers"), ("B", "mode B: bounded local reads, tau 0.3/1.5"))):
        for a in ("looped_attn", "attn_K3", "scaled_mix", "mmd", "mix"):
            ax[i + 1].plot(xs2, [st[f"srS{md}_dkl_{a}_K{K}"]["mean"] for K in KS_SR], color=pc[a], marker="o", ms=3, label=a)
        ax[i + 1].axhline(0, color=COL["ink2"], lw=1)
        ax[i + 1].set_xticks(xs2, [str(K) for K in KS_SR])
        ax[i + 1].set(title=f"strong root: KL(K) - KL(0), {lab}", xlabel="reads K (90/10 skew)")
        if md == "B":
            ax[i + 1].set_yscale("symlog", linthresh=0.01)
        ax[i + 1].legend(fontsize=7)
        ax[i + 1].title.set_fontsize(8)
    for md in ("A", "B"):
        for M in list(SR) + ["S"]:
            tg = f"sr{md}{M}" if M != "S" else f"srS{md}"
            for a in POOLS:
                rows += [[f"strong_root_mode{md}_M{M}", a, K, st[f"{tg}_kl_{a}_K{K}"]["mean"], st[f"{tg}_jsd_{a}_K{K}"]["mean"]] for K in KS_SR]
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
    for i, a in enumerate(("econ_prec", "jacobi_prec")):
        for t in TAUS:
            ax[i].plot(NS, [wr[f"{a}_W_t{t}"][N]["mean"] for N in NS], color=tc[t], marker="o", ms=4, label=f"tau {t}")
        ax[i].set(title=f"{a} cluster weight W(N)", xlabel="N same-patch reads", xscale="log")
        ax[i].set_xticks(NS, [str(x) for x in NS])
        ax[i].legend(fontsize=8)
    bs = ["econ_prec", "econ_D3", "kc_K3", "jacobi_prec", "jacobi", "looped_attn"]
    for j, t in enumerate(("0.3", "2.5")):
        ax[2].bar(np.arange(len(bs)) + (j - 0.5) * 0.38, [wr[f"{b}_W16/W1_t{t}"]["mean"] for b in bs], width=0.38, color=tc[t],
                  label=f"tau {t}")
    ax[2].axhline(1.25, color=COL["s1"], lw=1, ls="--")
    ax[2].axhline(2.20, color=COL["s3"], lw=1, ls="--")
    ax[2].set_xticks(range(len(bs)), bs, rotation=40, ha="right", fontsize=8)
    ax[2].set(title="W(16)/W(1) (bars: <= 1.25 at tau 0.3, >= 2.20 at tau 2.5)")
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


if __name__ == "__main__":
    main()
