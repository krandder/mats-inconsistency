"""E-CA round 3 analysis (eurisko, 2026-10-15): applies PREREG-R3.md Sec. 3 and 5 mechanically.

  PYTHONPATH=/tmp/smallnet/pylib python3 analyze_r3.py [RUNS_DIR]     (default results/raw/r3)

Run dirs: RUNS_DIR/r{rule}-L{L}-s{seed}/ (result.json, test-estimates.csv, d1-pairs.csv). Test KL is recomputed from the
per-instance logits (float64) and exact posteriors in test-estimates.csv. Writes results/r3/{summary.json, tables.md, *.png}.
Written 09:5xZ, before any round-3 result existed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

E = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(E, "results/raw/r3")
OUT = os.environ.get("ECA_OUT_R3", os.path.join(E, "results/r3"))
RULES, LS, SEEDS = (110, 90), (2, 3, 4), (0, 1, 2)
FAMS = ("F", "B1", "B2")
SELS = ("stiff", "conf", "voi", "voiu", "hyb", "rand")
POOLS = ("sw", "mix", "dd")
MS = (0, 1, 2, 4, 8, 16)
THR = {110: 0.08, 90: 0.05}
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
vd = lambda x: "GATE FAILED (descriptive)" if x is None else ("PASS" if x else "FAIL")


def gate_ok(val, thr):
    return val["S1"] <= 0.015 and sum(1 for f in FAMS if val[f] >= thr) >= 2


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for rule in RULES:
        for L in LS:
            for s in SEEDS:
                d = os.path.join(RUNS, f"r{rule}-L{L}-s{s}")
                if os.path.exists(os.path.join(d, "result.json")):
                    runs[(rule, L, s)] = load_run(d)
    S = {"runs_found": sorted(f"r{r}-L{L}-s{s}" for (r, L, s) in runs)}
    tabs = []
    # ---------------- gate and depth selection (seed 0 only)
    gate = {}
    for rule in RULES:
        rows = {}
        for L in LS:
            if (rule, L, 0) in runs:
                v = {f: runs[(rule, L, 0)][0]["P0"]["val"][f]["kl"] for f in ("F", "B1", "B2", "R", "S1")}
                rows[L] = {"val_kl": v, "pass": gate_ok(v, THR[rule])}
        passing = [L for L in rows if rows[L]["pass"]]
        gate[rule] = {"seed0": rows, "selected_L": max(passing) if passing else None}
    S["gate"] = gate
    tabs.append("### Depth gate (seed 0, validation one-pass KL, nats)\n\n| rule | L | F | B1 | B2 | R | S1 | gate |\n|---|---|---|---|---|---|---|---|\n")
    for rule in RULES:
        for L, v in sorted(gate[rule]["seed0"].items()):
            vv = v["val_kl"]
            tabs.append(f"| {rule} | {L} | {vv['F']:.4f} | {vv['B1']:.4f} | {vv['B2']:.4f} | {vv['R']:.4f} | {vv['S1']:.4f} | {'pass' if v['pass'] else 'fail'} |\n")
        tabs.append(f"\nRule {rule}: selected L = {gate[rule]['selected_L']} (largest passing; bar S1 <= 0.015 and >= {THR[rule]} on 2 of F/B1/B2)\n\n")
    # ---------------- per rule at the selected L (or L = 4 descriptive if the gate failed)
    for rule in RULES:
        Lsel = gate[rule]["selected_L"]
        Luse = Lsel if Lsel is not None else 4
        seeds = [s for s in SEEDS if (rule, Luse, s) in runs]
        if not seeds:
            continue
        key = f"r{rule}"
        R = {"L": Luse, "gate_passed": Lsel is not None, "seeds": seeds}
        rr = {s: runs[(rule, Luse, s)] for s in seeds}
        val = {f: ci([rr[s][0]["P0"]["val"][f]["kl"] for s in seeds]) for f in ("F", "B1", "B2", "R", "S1")}
        d0 = val["S1"]["mean"] <= 0.015 and sum(1 for f in FAMS if val[f]["mean"] >= THR[rule]) >= 2
        R["R3D0"] = {"val_kl": val, "pass": bool(d0) if Lsel is not None else False}
        k0 = {fam: [kl_col(rr[s][1], fam, "P0_logit") for s in seeds] for fam in FAMS}
        R["P0_test_kl"] = {fam: ci(k0[fam]) for fam in FAMS}
        cur = lambda fam, sel, pool, M: [kl_col(rr[s][1], fam, f"{sel}|{pool}|{M}") for s in seeds]
        # R3-D2a: F, M = 16, sw; stiff, conf, hyb
        d2a = {}
        ok_all = True
        for sel in ("stiff", "conf", "hyb"):
            ka = cur("F", sel, "sw", 16)
            kr = cur("F", "rand", "sw", 16)
            c0 = ci([1 - b / a for a, b in zip(k0["F"], ka)])
            d0_ = ci([a - b for a, b in zip(k0["F"], ka)])
            cr = ci([1 - b / a for a, b in zip(kr, ka)])
            dr = ci([a - b for a, b in zip(kr, ka)])
            p = (c0["mean"] >= 0.25 and d0_["lo"] > 0) and (cr["mean"] >= 0.15 and dr["lo"] > 0)
            d2a[sel] = {"kl": ci(ka), "cut_vs_P0": c0, "diff_vs_P0": d0_, "cut_vs_rand": cr, "diff_vs_rand": dr, "pass": bool(p)}
            ok_all = ok_all and p
        R["R3D2a"] = {"arms": d2a, "pass": bool(ok_all) if Lsel is not None else None}
        # R3-D2b: hyb under sw at M = 16 (rule 110 is the rule; rule 90 descriptive)
        d2b = {}
        okb = True
        for fam in FAMS:
            kh = cur(fam, "hyb", "sw", 16)
            c0 = ci([1 - b / a for a, b in zip(k0[fam], kh)])
            d0_ = ci([a - b for a, b in zip(k0[fam], kh)])
            ent = {"kl_hyb": ci(kh), "cut_vs_P0": c0, "diff_vs_P0": d0_, "pass_P0": bool(c0["mean"] >= 0.25 and d0_["lo"] > 0)}
            okb = okb and ent["pass_P0"]
            if fam in ("B1", "B2"):
                ks = cur(fam, "stiff", "sw", 16)
                ds = ci([a - b for a, b in zip(ks, kh)])
                ent.update({"kl_stiff": ci(ks), "stiff_minus_hyb": ds, "pass_vs_stiff": bool(ds["lo"] > 0)})
                okb = okb and ent["pass_vs_stiff"]
            else:
                kr = cur(fam, "rand", "sw", 16)
                dr = ci([a - b for a, b in zip(kr, kh)])
                ent.update({"kl_rand": ci(kr), "rand_minus_hyb": dr, "pass_vs_rand": bool(dr["lo"] > 0)})
                okb = okb and ent["pass_vs_rand"]
            d2b[fam] = ent
        R["R3D2b"] = {"families": d2b, "pass": (bool(okb) if rule == 110 else "descriptive") if Lsel is not None else None}
        # R3-D3
        sp = {k_: ci([rr[s][0]["R3D3"]["spearman_vs_negKL"][k_] for s in seeds]) for k_ in ("beta", "confidence", "selfcons", "neg_surprise")}
        em = {k_: ci([rr[s][0]["R3D3"]["entropy_matched_spearman_vs_negKL"][k_] for s in seeds]) for k_ in ("beta", "confidence", "selfcons")}
        gap = max(sp["beta"]["mean"], sp["selfcons"]["mean"]) - sp["confidence"]["mean"]
        R["R3D3"] = {"spearman": sp, "entropy_matched": em, "best_minus_confidence": gap,
                     "pass": bool(gap >= 0.05) if Lsel is not None else None,
                     "mean_kl_epi": ci([rr[s][0]["R3D3"]["mean_kl_epi"] for s in seeds])}
        # curves, compute, misc
        R["curves_kl"] = {fam: {f"{sel}|{pool}": [ci(cur(fam, sel, pool, M)) for M in MS] for sel in SELS for pool in POOLS} for fam in FAMS}
        R["hyb_branch_share"] = {k_: ci([rr[s][0]["hyb_branch_share"].get(k_, float("nan")) for s in seeds])
                                 for k_ in ("test|F", "test|B1", "test|B2")}
        R["compute_per_query"] = {k_: ci([rr[s][0]["compute_per_query"][k_]["passes_per_query"] for s in seeds])
                                  for k_ in rr[seeds[0]][0]["compute_per_query"]}
        R["D1"] = {k_: ci([rr[s][0]["D1"][k_] for s in seeds]) for k_ in ("spearman_beta_vs_info", "auc_blanket", "spearman_selfentropy_vs_info")}
        R["D3"] = {"ratio_m5": ci([rr[s][0]["D3a"]["ratio_m5"] for s in seeds]),
                   "redundant_cos": ci([rr[s][0]["D3a"]["mean_offdiag_cos_redundant"] for s in seeds]),
                   "cos_fwd_bwd_B1": ci([rr[s][0]["D3b"]["B1|cos_fwd_bwd"] for s in seeds]),
                   "cos_fwd_bwd_B2": ci([rr[s][0]["D3b"]["B2|cos_fwd_bwd"] for s in seeds])}
        R["training"] = [{k_: rr[s][0][k_] for k_ in ("steps_run", "best_step", "stop_reason", "train_sec", "best_val_outcome_logloss")} for s in seeds]
        S[key] = R
        # tables
        tabs.append(f"\n### Rule {rule} at L = {Luse} ({'selected by the gate' if Lsel is not None else 'GATE FAILED: L = 4 shown, descriptive'}; seeds {seeds})\n\n")
        tabs.append("R3-D0 (3-seed validation one-pass KL): " + "; ".join(f"{f} {f3(val[f])}" for f in ("F", "B1", "B2", "R", "S1"))
                    + f" -> **{vd(R['R3D0']['pass'] if Lsel is not None else None)}**\n\n")
        tabs.append("Test one-pass KL (500 chain instances): " + "; ".join(f"{f} {f3(R['P0_test_kl'][f])}" for f in FAMS) + "\n\n")
        tabs.append(f"R3-D2a (F, M = 16, sw) -> **{vd(R['R3D2a']['pass'])}**\n\n| arm | KL | cut vs P0 | cut vs rand | KL(rand) - KL(arm) | pass |\n|---|---|---|---|---|---|\n")
        for sel, v in d2a.items():
            tabs.append(f"| {sel} | {f3(v['kl'])} | {f2(v['cut_vs_P0'])} | {f2(v['cut_vs_rand'])} | {f3(v['diff_vs_rand'])} | {'yes' if v['pass'] else 'no'} |\n")
        lab = R["R3D2b"]["pass"]
        tabs.append(f"\nR3-D2b (hyb, sw, M = 16) -> **{vd(lab) if lab != 'descriptive' else 'descriptive (rule 90)'}**\n\n| family | KL hyb | cut vs P0 | comparison | pass |\n|---|---|---|---|---|\n")
        for fam, v in d2b.items():
            comp = (f"stiff - hyb {f3(v['stiff_minus_hyb'])}" if fam != "F" else f"rand - hyb {f3(v['rand_minus_hyb'])}")
            pss = v["pass_P0"] and (v.get("pass_vs_stiff", True) and v.get("pass_vs_rand", True))
            tabs.append(f"| {fam} | {f3(v['kl_hyb'])} | {f2(v['cut_vs_P0'])} | {comp} | {'yes' if pss else 'no'} |\n")
        tabs.append(f"\nR3-D3 (Spearman with -KL(P* || q) over 2,000 pairs): beta {f3(sp['beta'])}, confidence {f3(sp['confidence'])}, "
                    f"self-consistency {f3(sp['selfcons'])} (-s_hat {f3(sp['neg_surprise'])}); best minus confidence {gap:+.3f} -> "
                    f"**{vd(R['R3D3']['pass'])}**. Entropy-matched: beta {f3(em['beta'])}, confidence {f3(em['confidence'])}, "
                    f"self-consistency {f3(em['selfcons'])}\n")
        tabs.append("\nKL vs M (3-seed means; P0 is M = 0 for sw):\n\n| family | selector, pool | " + " | ".join(f"M={m}" for m in MS) + " |\n|---|---|" + "---|" * len(MS) + "\n")
        for fam in FAMS:
            for sel in SELS:
                for pool in POOLS:
                    c = R["curves_kl"][fam][f"{sel}|{pool}"]
                    tabs.append(f"| {fam} | {sel}, {pool} | " + " | ".join(f"{x['mean']:.4f}" for x in c) + " |\n")
        tabs.append("\nhyb branch share (test): " + "; ".join(f"{k_} {f2(v)}" for k_, v in R["hyb_branch_share"].items())
                    + "\n\nForward passes per query (K = 16) at M = 16: " + "; ".join(
                        f"{sel} {R['compute_per_query'][f'{sel}|16']['mean']:.0f}" for sel in SELS) + "\n")
        tabs.append(f"\nD1: Spearman(beta, ln2 - H*) {f3(R['D1']['spearman_beta_vs_info'])}, AUC {f3(R['D1']['auc_blanket'])}; "
                    f"D3a ratio m=5 {f3(R['D3']['ratio_m5'])}, redundant cos {f3(R['D3']['redundant_cos'])}; cos(k_fwd, k_bwd) "
                    f"B1 {f3(R['D3']['cos_fwd_bwd_B1'])}, B2 {f3(R['D3']['cos_fwd_bwd_B2'])}\n")
    # ---------------- all depths (secondary): P0 val KL and F/B1/B2 sw M=16 best selector
    tabs.append("\n### All depths (secondary): 3-seed validation one-pass KL and test KL under sw at M = 16\n\n| rule | L | seeds | val F | val B1 | val B2 | val S1 | test P0 F/B1/B2 | stiff | conf | hyb | rand |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for rule in RULES:
        for L in LS:
            seeds = [s for s in SEEDS if (rule, L, s) in runs]
            if not seeds:
                continue
            rr = {s: runs[(rule, L, s)] for s in seeds}
            v = {f: np.mean([rr[s][0]["P0"]["val"][f]["kl"] for s in seeds]) for f in ("F", "B1", "B2", "S1")}
            t0 = "/".join(f"{np.mean([kl_col(rr[s][1], f, 'P0_logit') for s in seeds]):.3f}" for f in FAMS)
            sel_c = []
            for sel in ("stiff", "conf", "hyb", "rand"):
                sel_c.append("/".join(f"{np.mean([kl_col(rr[s][1], f, f'{sel}|sw|16') for s in seeds]):.3f}" for f in FAMS))
            tabs.append(f"| {rule} | {L} | {seeds} | {v['F']:.4f} | {v['B1']:.4f} | {v['B2']:.4f} | {v['S1']:.4f} | {t0} | " + " | ".join(sel_c) + " |\n")
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open(os.path.join(OUT, "tables.md"), "w").write("".join(tabs))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"stiff": "#1f6feb", "voi": "#d1242f", "voiu": "#8250df", "conf": "#9a6700", "rand": "#57606a", "hyb": "#1a7f37"}
        for rule in RULES:
            if f"r{rule}" not in S:
                continue
            R = S[f"r{rule}"]
            fig, axs2 = plt.subplots(2, 3, figsize=(13.5, 7.6))
            for row, pool in enumerate(("sw", "mix")):
                for ax, fam in zip(axs2[row], FAMS):
                    for sel in SELS:
                        c = R["curves_kl"][fam][f"{sel}|{pool}"]
                        xs = np.arange(len(MS))
                        ax.plot(xs, [x["mean"] for x in c], "-o", color=cols[sel], ms=3, label=sel)
                        ax.fill_between(xs, [x["lo"] for x in c], [x["hi"] for x in c], color=cols[sel], alpha=0.10)
                    ax.axhline(R["P0_test_kl"][fam]["mean"], color="k", ls="--", lw=1, label="P0 one pass")
                    ax.set_xticks(np.arange(len(MS)))
                    ax.set_xticklabels([str(m) for m in MS])
                    ax.set_yscale("log")
                    ax.set_xlabel("M (cells materialised per chain; K = 16)")
                    ax.set_ylabel("KL(exact || estimate), nats")
                    ax.set_title(f"rule {rule}, L = {R['L']}, {fam}: {'sw_mix (anchored)' if pool == 'sw' else 'plain mixture'}", fontsize=8)
                axs2[row][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"kl-vs-M-r3-r{rule}.png"), dpi=110)
            plt.close(fig)
    except Exception as ex:
        print("figures skipped:", ex)
    print("".join(tabs)[:6000])


if __name__ == "__main__":
    main()
