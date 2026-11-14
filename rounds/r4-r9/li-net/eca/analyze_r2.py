"""E-CA round 2 analysis (eurisko, 2026-10-15): applies PREREG-R2.md's rules R2-D2a/b and R2-D3a/b mechanically.

  PYTHONPATH=/tmp/smallnet/pylib python3 analyze_r2.py [R2A_DIR] [R2B_DIR] [R1_PRIMARY_DIR]

Defaults: results/raw/r2a, results/raw/r2b, results/raw/main2 (round-1 primary, for the descriptive voi comparison).
Test KL / log loss are recomputed from test-estimates.csv (float64 logits). Writes results/r2/summary.json, tables.md.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

E = os.path.dirname(os.path.abspath(__file__))
R2A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(E, "results/raw/r2a")
R2B = sys.argv[2] if len(sys.argv) > 2 else os.path.join(E, "results/raw/r2b")
R1P = sys.argv[3] if len(sys.argv) > 3 else os.path.join(E, "results/raw/main2")
OUT = os.environ.get("ECA_OUT_R2", os.path.join(E, "results/r2"))
SEEDS = (0, 1, 2)
FAMS = ("F", "B1", "B2", "R")
MS = (0, 1, 2, 4, 8, 16)
T975 = {1: 12.706, 2: 4.303}


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def kl(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def ci(v):
    v = np.asarray([x for x in v if x is not None and math.isfinite(float(x))], float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(v.mean())
    if n < 2:
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": n}
    h = T975.get(n - 1, 1.96) * float(v.std(ddof=1)) / math.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


def load_run(d):
    res = json.load(open(os.path.join(d, "result.json")))
    rows = list(csv.reader(open(os.path.join(d, "test-estimates.csv"))))
    hdr, body = rows[0], rows[1:]
    cols = {h: i for i, h in enumerate(hdr)}
    est = {}
    for fam in FAMS:
        fr = [r for r in body if r[0] == fam]
        est[fam] = {"y": np.array([float(r[cols["y"]]) for r in fr]), "pstar": np.array([float(r[cols["pstar"]]) for r in fr]),
                    "cols": {h: np.array([float(r[i]) for r in fr]) for h, i in cols.items() if i >= 4}}
    return res, est


def kl_col(est, fam, col):
    e = est[fam]
    return float(kl(e["pstar"], sig(e["cols"][col])).mean())


f3 = lambda c: f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c["n"] > 1 else f"{c['mean']:.4f}"
f2 = lambda c: f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if c["n"] > 1 else f"{c['mean']:.3f}"
vd = lambda x: "NOT APPLICABLE" if x is None else ("PASS" if x else "FAIL")


def main():
    os.makedirs(OUT, exist_ok=True)
    S, tabs = {}, []
    # ---------------- R2a: 1-layer, rules 110 and 90
    for rule in (110, 90):
        runs = {s: load_run(os.path.join(R2A, f"r{rule}-T1L1-s{s}")) for s in SEEDS
                if os.path.exists(os.path.join(R2A, f"r{rule}-T1L1-s{s}", "result.json"))}
        if not runs:
            continue
        seeds = sorted(runs)
        key = f"r2a-r{rule}"
        R = {"seeds": seeds}
        val = {f: ci([runs[s][0]["P0"]["val"][f]["kl"] for s in seeds]) for f in ("F", "B1", "B2", "R", "S1")}
        nlong = sum(1 for f in ("F", "B1", "B2") if val[f]["mean"] >= 0.05)
        R["D2a"] = {"val_kl": val, "n_long_ge_0.05": nlong, "pass": bool(val["S1"]["mean"] <= 0.025 and nlong >= 2)}
        p0t = {f: ci([kl_col(runs[s][1], f, "P0_logit") for s in seeds]) for f in FAMS}
        R["P0_test_kl"] = p0t
        fams_q = [f for f in ("F", "B1", "B2") if p0t[f]["mean"] >= 0.05]
        d2b = {"qualifying": fams_q, "families": {}}
        allpass = True if fams_q else None
        for fam in fams_q:
            k0 = [kl_col(runs[s][1], fam, "P0_logit") for s in seeds]
            fam_ok = False
            det = {}
            for sel in ("stiff", "conf"):
                ok_both = True
                for pool in ("mix", "sw"):
                    k16 = [kl_col(runs[s][1], fam, f"{sel}|{pool}|16") for s in seeds]
                    cut = ci([1 - b / a for a, b in zip(k0, k16)])
                    dif = ci([a - b for a, b in zip(k0, k16)])
                    pa = cut["mean"] >= 0.25 and dif["lo"] > 0
                    pb = False
                    vs = {}
                    for M in (8, 16):
                        ks = [kl_col(runs[s][1], fam, f"{sel}|{pool}|{M}") for s in seeds]
                        kr = [kl_col(runs[s][1], fam, f"rand|{pool}|{M}") for s in seeds]
                        dd = ci([b - a for a, b in zip(ks, kr)])
                        vs[M] = dd
                        pb = pb or dd["lo"] > 0
                    det[f"{sel}|{pool}"] = {"kl_M16": ci(k16), "cut_vs_P0_M16": cut, "diff_vs_P0": dif, "pass_cut": pa,
                                            "rand_minus_sel_M8": vs[8], "rand_minus_sel_M16": vs[16], "pass_vs_rand": pb}
                    ok_both = ok_both and pa and pb
                fam_ok = fam_ok or ok_both
            d2b["families"][fam] = {"detail": det, "pass": fam_ok}
            allpass = allpass and fam_ok
        d2b["pass"] = allpass
        R["D2b"] = d2b
        curves = {fam: {"P0": p0t[fam]} for fam in FAMS}
        for fam in FAMS:
            for sel in ("stiff", "conf", "rand"):
                for pool in ("mix", "sw", "lw"):
                    curves[fam][f"{sel}|{pool}"] = [ci([kl_col(runs[s][1], fam, f"{sel}|{pool}|{M}") for s in seeds]) for M in MS]
        R["curves_kl"] = curves
        R["D1"] = {k: ci([runs[s][0]["D1"][k] for s in seeds]) for k in ("spearman_beta_vs_info", "auc_blanket", "spearman_selfentropy_vs_info")}
        R["sw_fit_M16"] = {sel: [runs[s][0]["pool_sw"][f"{sel}|sw|16"] for s in seeds] for sel in ("stiff", "conf", "rand")}
        R["training"] = [{k: runs[s][0][k] for k in ("steps_run", "best_step", "stop_reason", "train_sec", "best_val_outcome_logloss")} for s in seeds]
        S[key] = R
        tabs.append(f"\n### {key} (1 layer, T1, seeds {seeds})\n")
        tabs.append("R2-D2a gate (validation P0 KL): " + "; ".join(f"{f} {f3(val[f])}" for f in ("F", "B1", "B2", "R", "S1"))
                    + f" -> **{vd(R['D2a']['pass'])}** ({nlong} of F/B1/B2 >= 0.05; S1 <= 0.025: {val['S1']['mean'] <= 0.025})\n")
        tabs.append("Test P0 KL (1,000 chain instances): " + "; ".join(f"{f} {f3(p0t[f])}" for f in FAMS) + "\n")
        tabs.append(f"R2-D2b (qualifying families, test P0 KL >= 0.05: {fams_q or 'none'}) -> **{vd(d2b['pass'])}**\n")
        for fam in fams_q:
            tabs.append(f"\n| {fam} | KL M=16 | cut vs P0 M=16 | (cut) | KL(rand) - KL(sel) M=8 | M=16 | (vs rand) |\n|---|---|---|---|---|---|---|\n")
            for k_, v in d2b["families"][fam]["detail"].items():
                tabs.append(f"| {k_} | {f3(v['kl_M16'])} | {f2(v['cut_vs_P0_M16'])} | {'PASS' if v['pass_cut'] else 'FAIL'} | "
                            f"{f3(v['rand_minus_sel_M8'])} | {f3(v['rand_minus_sel_M16'])} | {'PASS' if v['pass_vs_rand'] else 'FAIL'} |\n")
            tabs.append(f"Family {fam}: **{'PASS' if d2b['families'][fam]['pass'] else 'FAIL'}**\n")
        tabs.append("\nKL vs M (3-seed means):\n\n| family | selector, pool | " + " | ".join(f"M={m}" for m in MS) + " | P0 |\n|---|---|" + "---|" * (len(MS) + 1) + "\n")
        for fam in FAMS:
            for sel in ("stiff", "conf", "rand"):
                for pool in ("mix", "sw"):
                    c = curves[fam][f"{sel}|{pool}"]
                    tabs.append(f"| {fam} | {sel}, {pool} | " + " | ".join(f"{x['mean']:.4f}" for x in c) + f" | {p0t[fam]['mean']:.4f} |\n")
        tabs.append(f"\nD1 at 1 layer: Spearman {f3(R['D1']['spearman_beta_vs_info'])}, AUC {f3(R['D1']['auc_blanket'])}\n")
    # ---------------- R2b: rule 110, 4 layers, v2 keys, voiu
    runs = {s: load_run(os.path.join(R2B, f"r110-T1-s{s}")) for s in SEEDS if os.path.exists(os.path.join(R2B, f"r110-T1-s{s}", "result.json"))}
    if runs:
        seeds = sorted(runs)
        R = {"seeds": seeds}
        d3b_cos = {f: ci([runs[s][0]["D3b"][f"{f}|cos_fwd_bwd"] for s in seeds]) for f in ("B1", "B2")}
        d3b_acos = {f: ci([runs[s][0]["D3b"].get(f"{f}|attncos_fwd_bwd", float("nan")) for s in seeds]) for f in ("B1", "B2")}
        red = ci([runs[s][0]["D3a"]["mean_offdiag_cos_redundant"] for s in seeds])
        red_a = ci([runs[s][0]["D3a"].get("mean_offdiag_attncos_redundant", float("nan")) for s in seeds])
        r5 = ci([runs[s][0]["D3a"]["ratio_m5"] for s in seeds])
        r2 = ci([runs[s][0]["D3a"]["ratio_m2"] for s in seeds])
        p3a = bool(all(d3b_cos[f]["mean"] <= 0.35 for f in ("B1", "B2")) and red["mean"] >= 0.85 and r5["mean"] < 0.20)
        R["D3a"] = {"cos_fwd_bwd": d3b_cos, "attncos_fwd_bwd": d3b_acos, "redundant_key_cos": red, "redundant_attn_cos": red_a,
                    "ratio_m5": r5, "ratio_m2": r2, "pass": p3a}
        d3bb = {}
        for fam in ("B1", "B2"):
            for k_ in ("both", "fwd", "bwd"):
                d3bb[f"{fam}|{k_}"] = ci([runs[s][0]["D3b"][f"{fam}|{k_}|dd"]["kl"] for s in seeds])
        R["D3b_pool_kl"] = d3bb
        p3b = True
        det = {}
        r1 = {}
        if os.path.isdir(R1P):
            for s in SEEDS:
                d = os.path.join(R1P, f"r110-T1-s{s}")
                if os.path.exists(os.path.join(d, "result.json")):
                    r1[s] = load_run(d)
        for fam in ("B1", "B2"):
            k0 = [kl_col(runs[s][1], fam, "P0_logit") for s in seeds]
            for pool in ("mix", "sw"):
                ku = [kl_col(runs[s][1], fam, f"voiu|{pool}|8") for s in seeds]
                kv = [kl_col(runs[s][1], fam, f"voi|{pool}|8") for s in seeds]
                c_v = ci([1 - b / a for a, b in zip(kv, ku)])
                d_v = ci([a - b for a, b in zip(kv, ku)])
                c_0 = ci([1 - b / a for a, b in zip(k0, ku)])
                d_0 = ci([a - b for a, b in zip(k0, ku)])
                ok = c_v["mean"] >= 0.30 and d_v["lo"] > 0 and c_0["mean"] >= 0.35 and d_0["lo"] > 0
                r1v = ci([kl_col(r1[s][1], fam, f"voi|{pool}|8") for s in r1]) if r1 and pool != "sw" else None
                det[f"{fam}|{pool}"] = {"kl_voiu_M8": ci(ku), "kl_voi_M8": ci(kv), "kl_P0": ci(k0), "cut_vs_voi": c_v, "diff_vs_voi": d_v,
                                        "cut_vs_P0": c_0, "diff_vs_P0": d_0, "pass": bool(ok), "round1_voi_M8_kl": r1v}
                p3b = p3b and ok
        R["D3b"] = {"detail": det, "pass": bool(p3b)}
        curves = {fam: {"P0": ci([kl_col(runs[s][1], fam, "P0_logit") for s in seeds])} for fam in FAMS}
        for fam in FAMS:
            for sel in ("stiff", "voi", "voiu", "conf", "rand"):
                for pool in ("mix", "sw", "lw"):
                    curves[fam][f"{sel}|{pool}"] = [ci([kl_col(runs[s][1], fam, f"{sel}|{pool}|{M}") for s in seeds]) for M in MS]
        R["curves_kl"] = curves
        R["training"] = [{k: runs[s][0][k] for k in ("steps_run", "best_step", "stop_reason", "train_sec", "best_val_outcome_logloss")} for s in seeds]
        S["r2b-r110"] = R
        tabs.append(f"\n### r2b-r110 (4 layers, T1, v2 keys, seeds {seeds})\n")
        tabs.append(f"R2-D3a: cos(k_fwd, k_bwd) B1 {f3(d3b_cos['B1'])}, B2 {f3(d3b_cos['B2'])} (bar <= 0.35); redundant-view key cos "
                    f"{f3(red)} (bar >= 0.85); drift ratio m=5 {f3(r5)} (bar < 0.20), m=2 {f3(r2)}; attention-vector cosines: fwd/bwd B1 "
                    f"{f3(d3b_acos['B1'])}, B2 {f3(d3b_acos['B2'])} (disjoint supports, 0 by construction), redundant views {f3(red_a)} "
                    f"-> **{vd(p3a)}**\n")
        tabs.append("D3b pools with v2 keys (KL, dd pool): " + "; ".join(f"{k_} {f3(v)}" for k_, v in d3bb.items()) + "\n")
        tabs.append(f"\nR2-D3b (voiu vs voi at M = 8, same models) -> **{vd(p3b)}**\n\n| family, pool | KL P0 | KL voi M=8 | KL voiu M=8 | cut vs voi | cut vs P0 | rule | round-1 voi M=8 (other models) |\n|---|---|---|---|---|---|---|---|\n")
        for k_, v in det.items():
            tabs.append(f"| {k_} | {f3(v['kl_P0'])} | {f3(v['kl_voi_M8'])} | {f3(v['kl_voiu_M8'])} | {f2(v['cut_vs_voi'])} | {f2(v['cut_vs_P0'])} | "
                        f"{'PASS' if v['pass'] else 'FAIL'} | {f3(v['round1_voi_M8_kl']) if v['round1_voi_M8_kl'] else '-'} |\n")
        tabs.append("\nKL vs M (3-seed means):\n\n| family | selector, pool | " + " | ".join(f"M={m}" for m in MS) + " | P0 |\n|---|---|" + "---|" * (len(MS) + 1) + "\n")
        for fam in FAMS:
            for sel in ("stiff", "voi", "voiu", "conf", "rand"):
                for pool in ("mix", "sw"):
                    c = curves[fam][f"{sel}|{pool}"]
                    tabs.append(f"| {fam} | {sel}, {pool} | " + " | ".join(f"{x['mean']:.4f}" for x in c) + f" | {curves[fam]['P0']['mean']:.4f} |\n")
    # secondary readout (added 05:25Z, after the round-2 results were read): every selector under sw_mix and mix vs P0
    tabs.append("\n### Secondary: each selector vs P0 at M = 16 under sw_mix and the plain mixture (cut = 1 - KL/KL(P0), 3-seed mean, paired 95% CI)\n\n| run set | family | " + " | ".join(f"{sel} {pool}" for sel in ("stiff", "voi", "voiu", "conf", "rand") for pool in ("sw", "mix")) + " |\n|---|---|" + "---|" * 10 + "\n")
    for key, rdir, pat in (("r2a-r110", R2A, "r110-T1L1-s{}"), ("r2a-r90", R2A, "r90-T1L1-s{}"), ("r2b-r110", R2B, "r110-T1-s{}")):
        rr = {s_: load_run(os.path.join(rdir, pat.format(s_))) for s_ in SEEDS if os.path.exists(os.path.join(rdir, pat.format(s_), "result.json"))}
        if not rr:
            continue
        S.setdefault(key, {})["sw_mix_vs_P0_M16"] = {}
        for fam in FAMS:
            k0 = [kl_col(rr[s_][1], fam, "P0_logit") for s_ in sorted(rr)]
            cells = []
            for sel in ("stiff", "voi", "voiu", "conf", "rand"):
                for pool in ("sw", "mix"):
                    col = f"{sel}|{pool}|16"
                    if col not in rr[sorted(rr)[0]][1][fam]["cols"]:
                        cells.append("-")
                        continue
                    km = [kl_col(rr[s_][1], fam, col) for s_ in sorted(rr)]
                    c = ci([1 - b_ / a_ for a_, b_ in zip(k0, km)])
                    d = ci([a_ - b_ for a_, b_ in zip(k0, km)])
                    S[key]["sw_mix_vs_P0_M16"][f"{fam}|{sel}|{pool}"] = {"cut": c, "diff": d}
                    star = "*" if d["lo"] > 0 else ""
                    cells.append(f"{c['mean']:+.2f}{star}")
            tabs.append(f"| {key} | {fam} | " + " | ".join(cells) + " |\n")
    tabs.append("\n(* = paired 95% CI of KL(P0) - KL(arm) above 0.)\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"stiff": "#1f6feb", "voi": "#d1242f", "voiu": "#8250df", "conf": "#9a6700", "rand": "#57606a"}
        for key, sels in (("r2a-r110", ("stiff", "conf", "rand")), ("r2a-r90", ("stiff", "conf", "rand")),
                          ("r2b-r110", ("stiff", "voi", "voiu", "conf", "rand"))):
            if key not in S:
                continue
            R = S[key]
            fig, axs2 = plt.subplots(2, 4, figsize=(16, 7.6))
            for row, pool in enumerate(("mix", "sw")):
                for ax, fam in zip(axs2[row], FAMS):
                    for sel in sels:
                        c = R["curves_kl"][fam][f"{sel}|{pool}"]
                        xs = np.arange(len(MS))
                        ax.plot(xs, [x["mean"] for x in c], "-o", color=cols[sel], ms=3, label=sel)
                        ax.fill_between(xs, [x["lo"] for x in c], [x["hi"] for x in c], color=cols[sel], alpha=0.12)
                    ax.axhline(R["curves_kl"][fam]["P0"]["mean"], color="k", ls="--", lw=1, label="P0 one pass")
                    ax.set_xticks(np.arange(len(MS)))
                    ax.set_xticklabels([str(m) for m in MS])
                    ax.set_yscale("log")
                    ax.set_xlabel("M (cells materialised per chain; K = 16)")
                    ax.set_ylabel("KL(exact || estimate), nats")
                    ax.set_title(f"{key} {fam}: {'plain mixture' if pool == 'mix' else 'sw_mix (anchored at P0)'}", fontsize=8)
                axs2[row][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"kl-vs-M-{key}.png"), dpi=110)
            plt.close(fig)
    except Exception as ex:
        print("figures skipped:", ex)
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(OUT, "tables.md"), "w").write("".join(tabs))
    print("".join(tabs)[:5000])


if __name__ == "__main__":
    main()
