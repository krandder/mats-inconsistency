"""E-CA analysis (eurisko, 2026-10-15): applies PREREG.md Sec. 7 mechanically to the 12 main runs.

  PYTHONPATH=/tmp/smallnet/pylib python3 analyze.py [RUNS_DIR]   (default /tmp/eca/runs/main)

Reads RUNS_DIR/r{rule}-{arm}-s{seed}/result.json and test-estimates.csv. Test KL / log loss are recomputed here from the
per-instance logits and exact posteriors in test-estimates.csv (same instances in every seed). Writes
results/summary.json, results/tables.md and PNG figures under results/.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

E = os.path.dirname(os.path.abspath(__file__))
# run directories in precedence order (PREREG Sec. 12): the float64 crash-rerun main2/ first, then the original main/.
# Every run read from a directory other than main2/ is "capped": its CSV probabilities are limited to [6e-8, 1 - 6e-8]
# (float32 resolution; its +inf logits came from the float32 pooling bug).
RUN_DIRS = sys.argv[1:] or ["/tmp/eca/runs/main2", "/tmp/eca/runs/main"]
RUNS = RUN_DIRS[0]
CAP_LOGIT = math.log((1 - 6e-8) / 6e-8)
OUT = os.environ.get("ECA_OUT", os.path.join(E, "results"))
RULES, ARMS, SEEDS = (110, 90), ("T1", "T2"), (0, 1, 2)
FAMS = ("F", "B1", "B2", "R")
SELS = ("stiff", "voi", "conf", "rand")
MS = (0, 1, 2, 4, 8, 16)
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def kl(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def ll(q, y):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


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


def _rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[o]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def _auc(score, label):
    label = np.asarray(label, bool)
    r = _rank(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load():
    runs = {}
    for rule in RULES:
        for arm in ARMS:
            for s in SEEDS:
                d = None
                for base in RUN_DIRS:
                    if os.path.exists(os.path.join(base, f"r{rule}-{arm}-s{s}", "result.json")):
                        d = os.path.join(base, f"r{rule}-{arm}-s{s}")
                        break
                if d is None:
                    continue
                rj = os.path.join(d, "result.json")
                capped = os.path.basename(os.path.dirname(d)) != "main2"
                res = json.load(open(rj))
                rows = list(csv.reader(open(os.path.join(d, "test-estimates.csv"))))
                hdr, body = rows[0], rows[1:]
                cols = {h: i for i, h in enumerate(hdr)}
                est = {}
                for fam in FAMS:
                    fr = [r for r in body if r[0] == fam]
                    y = np.array([float(r[cols["y"]]) for r in fr])
                    ps = np.array([float(r[cols["pstar"]]) for r in fr])
                    cc = {h: np.array([float(r[i]) for r in fr]) for h, i in cols.items() if i >= 4}
                    if capped:
                        cc = {h: np.clip(v, -CAP_LOGIT, CAP_LOGIT) for h, v in cc.items()}
                    est[fam] = {"y": y, "pstar": ps, "cols": cc}
                res["_source"] = d
                res["_capped"] = capped
                runs[(rule, arm, s)] = (res, est)
    return runs


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    S = {"runs_found": sorted(f"r{r}-{a}-s{s}" for (r, a, s) in runs), "run_dirs": RUN_DIRS,
         "sources": {f"r{r}-{a}-s{s}": runs[(r, a, s)][0]["_source"] for (r, a, s) in runs}}
    tabs = []

    # ---------------- per-run test KL / logloss from the per-instance file
    def metric(rule, arm, s, fam, col, what="kl"):
        _, est = runs[(rule, arm, s)]
        e = est[fam]
        q = sig(e["cols"][col])
        return float((kl(e["pstar"], q) if what == "kl" else ll(q, e["y"])).mean())

    for rule in RULES:
        for arm in ARMS:
            seeds = [s for s in SEEDS if (rule, arm, s) in runs]
            key = f"r{rule}-{arm}"
            if not seeds:
                continue
            R = {"seeds": seeds, "sources": [runs[(rule, arm, s)][0]["_source"] for s in seeds],
                 "capped_seeds": [s for s in seeds if runs[(rule, arm, s)][0]["_capped"]]}
            # D0 (validation P0, recorded per run) + test P0 on all 3000
            d0 = {}
            for fam in ("F", "B1", "B2", "R", "S1"):
                d0[f"val_kl_{fam}"] = ci([runs[(rule, arm, s)][0]["P0"]["val"][fam]["kl"] for s in seeds])
                d0[f"test_kl_{fam}"] = ci([runs[(rule, arm, s)][0]["P0"]["test"][fam]["kl"] for s in seeds])
            d0["pass"] = all(d0[f"val_kl_{f}"]["mean"] >= 0.05 for f in ("F", "B1", "B2")) and d0["val_kl_S1"]["mean"] <= 0.02
            d0["solved_lt_0.01"] = all(d0[f"val_kl_{f}"]["mean"] < 0.01 for f in ("F", "B1", "B2"))
            d0["per_run_pass"] = [runs[(rule, arm, s)][0]["D0_gate_val"]["pass"] for s in seeds]
            R["D0"] = d0
            # D1
            d1 = {k: ci([runs[(rule, arm, s)][0]["D1"][k] for s in seeds]) for k in
                  ("spearman_beta_vs_info", "auc_blanket", "auc_parents_only", "spearman_neg_surprise_vs_info",
                   "spearman_selfentropy_vs_info", "frac_beta_zero", "frac_blanket")}
            orc = []
            for s_ in seeds:
                rows = list(csv.DictReader(open(os.path.join(runs[(rule, arm, s_)][0]["_source"], "d1-pairs.csv"))))
                bl = np.array([int(r_["blanket"]) for r_ in rows]).astype(bool)
                ig = np.log(2) - np.array([float(r_["exact_H"]) for r_ in rows])
                orc.append(_auc(ig, bl))
            d1["oracle_auc_exact_info_gain"] = ci(orc)   # secondary (03:50Z): ceiling of any entropy-based score on this label
            d1["expected_spearman"] = ci([runs[(rule, arm, s)][0]["D1"]["expected_version"]["spearman"] for s in seeds])
            d1["pass"] = d1["spearman_beta_vs_info"]["mean"] >= 0.60 and d1["auc_blanket"]["mean"] >= 0.85
            d1["pass_strict"] = d1["spearman_beta_vs_info"]["lo"] >= 0.60 and d1["auc_blanket"]["lo"] >= 0.85
            R["D1"] = d1
            # D2: declare the P2 variant from VALIDATION outcome log loss (lw, M=16, F/B1/B2), then test
            vll = {sel: float(np.mean([np.mean([runs[(rule, arm, s)][0]["chains"]["val"][f"{fam}|{sel}|lw|16"]["logloss"]
                                                for fam in ("F", "B1", "B2")]) for s in seeds])) for sel in ("stiff", "voi")}
            p2 = min(vll, key=vll.get)
            d2 = {"declared_variant": p2, "val_logloss_lw16": vll, "families": {}}
            allpass = True
            for fam in ("F", "B1", "B2"):
                kp0 = [metric(rule, arm, s, fam, "P0_logit") for s in seeds]
                kp2 = [metric(rule, arm, s, fam, f"{p2}|lw|16") for s in seeds]
                kp4 = [metric(rule, arm, s, fam, "rand|lw|16") for s in seeds]
                kp2m0 = [metric(rule, arm, s, fam, f"{p2}|lw|0") for s in seeds]
                cut0 = ci([1 - b / a for a, b in zip(kp0, kp2)])
                cut4 = ci([1 - b / a for a, b in zip(kp4, kp2)])
                cutm0 = ci([1 - b / a for a, b in zip(kp2m0, kp2)])
                diff0 = ci([a - b for a, b in zip(kp0, kp2)])
                diff4 = ci([a - b for a, b in zip(kp4, kp2)])
                pa = cut0["mean"] >= 0.30 and diff0["lo"] > 0
                pb = cut4["mean"] >= 0.20 and diff4["lo"] > 0
                allpass = allpass and pa and pb
                d2["families"][fam] = {"kl_P0": ci(kp0), "kl_P2_M16": ci(kp2), "kl_P4_M16": ci(kp4), "kl_P2_M0": ci(kp2m0),
                                       "cut_vs_P0": cut0, "cut_vs_P4": cut4, "cut_vs_P2M0": cutm0, "diff_vs_P0": diff0,
                                       "diff_vs_P4": diff4, "pass_a": pa, "pass_b": pb,
                                       "strict_a": cut0["lo"] >= 0.30, "strict_b": cut4["lo"] >= 0.20}
            d2["pass"] = allpass
            R["D2"] = d2
            # base-design rules kept as SECONDARY readouts (v2's D0-D3 are the verdicts): VoI vs Conf / Rand at equal M on
            # bridge queries, and P2-VoI at M = 8 vs P0 (base D1 used M = 8)
            sec = {}
            for fam in ("B1", "B2"):
                for M in (8, 16):
                    kv = [metric(rule, arm, s, fam, f"voi|lw|{M}") for s in seeds]
                    kc = [metric(rule, arm, s, fam, f"conf|lw|{M}") for s in seeds]
                    kr = [metric(rule, arm, s, fam, f"rand|lw|{M}") for s in seeds]
                    k0 = [metric(rule, arm, s, fam, "P0_logit") for s in seeds]
                    sec[f"{fam}|M{M}"] = {"voi_minus_conf": ci([a - b for a, b in zip(kv, kc)]),
                                          "voi_minus_rand": ci([a - b for a, b in zip(kv, kr)]),
                                          "voi_cut_vs_P0": ci([1 - a / b for a, b in zip(kv, k0)])}
            R["secondary_base_rules"] = sec
            # P1 arm (random-order chains + plain mixture; PREREG Sec. 5) and the other selectors under the mixture, vs P0
            p1 = {}
            for fam in ("F", "B1", "B2", "R"):
                k0 = [metric(rule, arm, s, fam, "P0_logit") for s in seeds]
                for sel in SELS:
                    for M in (2, 8, 16):
                        km = [metric(rule, arm, s, fam, f"{sel}|mix|{M}") for s in seeds]
                        p1[f"{fam}|{sel}|mix|{M}"] = {"kl": ci(km), "cut_vs_P0": ci([1 - a / b for a, b in zip(km, k0)]),
                                                      "diff_vs_P0": ci([b - a for a, b in zip(km, k0)])}
            R["mixture_vs_P0"] = p1
            # D3
            d3a = {k: ci([runs[(rule, arm, s)][0]["D3a"][k] for s in seeds]) for k in
                   ("ratio_m2", "ratio_m5", "lw|drift_m2", "lw|drift_m5", "dd|drift_m2", "dd|drift_m5", "mean_offdiag_cos_redundant")}
            d3a["pass"] = d3a["ratio_m5"]["mean"] < 0.20 and d3a["ratio_m2"]["mean"] < 0.20
            d3b = {}
            pb_all = True
            for fam in ("B1", "B2"):
                g = lambda c, m: [runs[(rule, arm, s)][0]["D3b"][f"{fam}|{c}|dd"][m] for s in seeds]
                both_kl, f_kl, b_kl = g("both", "kl"), g("fwd", "kl"), g("bwd", "kl")
                both_ll, f_ll, b_ll = g("both", "logloss"), g("fwd", "logloss"), g("bwd", "logloss")
                best_kl = [min(a, b) for a, b in zip(f_kl, b_kl)]
                best_ll = [min(a, b) for a, b in zip(f_ll, b_ll)]
                allv = both_kl + f_kl + b_kl + both_ll + f_ll + b_ll
                if all(math.isfinite(float(x)) for x in allv):
                    ok = bool(np.mean(both_kl) < np.mean(f_kl) and np.mean(both_kl) < np.mean(b_kl) and
                              np.mean(both_ll) < np.mean(f_ll) and np.mean(both_ll) < np.mean(b_ll))
                else:
                    ok = None                       # non-finite (float32 pooling bug in an original run): not evaluable
                pb_all = (pb_all and ok) if ok is not None and pb_all is not None else None
                d3b[fam] = {"kl_both": ci(both_kl), "kl_fwd": ci(f_kl), "kl_bwd": ci(b_kl), "ll_both": ci(both_ll),
                            "ll_fwd": ci(f_ll), "ll_bwd": ci(b_ll),
                            "diff_kl_bestsingle_minus_both": ci([a - b for a, b in zip(best_kl, both_kl)]),
                            "diff_ll_bestsingle_minus_both": ci([a - b for a, b in zip(best_ll, both_ll)]),
                            "cos_fwd_bwd": ci([runs[(rule, arm, s)][0]["D3b"][f"{fam}|cos_fwd_bwd"] for s in seeds]),
                            "pass": ok}
            d3b["pass"] = pb_all
            R["D3"] = {"a": d3a, "b": d3b, "pass": (d3a["pass"] and pb_all) if pb_all is not None else None}
            # KL vs M per family x selector (lw pool), P6 aggregators, dedup pool, compute
            curves = {}
            for fam in FAMS:
                curves[fam] = {"P0": ci([metric(rule, arm, s, fam, "P0_logit") for s in seeds])}
                for sel in SELS:
                    for pool in ("lw", "dd", "mix", "g0", "g1"):
                        curves[fam][f"{sel}|{pool}"] = [ci([metric(rule, arm, s, fam, f"{sel}|{pool}|{M}") for s in seeds])
                                                        for M in MS]
            R["curves_kl"] = curves
            R["frac_lw0_equals_prior"] = {fam: ci([float(np.mean(np.abs(runs[(rule, arm, s)][1][fam]["cols"]["voi|lw|0"]
                                                                      - runs[(rule, arm, s)][1][fam]["cols"]["p0_logit"]) < 1e-4))
                                                   for s in seeds]) for fam in FAMS}
            R["logloss_M16"] = {fam: {f"{sel}|{pool}": ci([metric(rule, arm, s, fam, f"{sel}|{pool}|16", "ll") for s in seeds])
                                      for sel in SELS for pool in ("lw", "dd", "mix", "g0", "g1")} for fam in FAMS}
            R["logloss_P0"] = {fam: ci([metric(rule, arm, s, fam, "P0_logit", "ll") for s in seeds]) for fam in FAMS}
            r0 = runs[(rule, arm, seeds[0])][0]
            R["compute_per_query"] = r0["compute_per_query"]
            R["training"] = [{k: runs[(rule, arm, s)][0][k] for k in ("steps_run", "best_step", "best_val_outcome_logloss",
                                                                   "stop_reason", "train_sec", "train_seqs")} for s in seeds]
            S[key] = R
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---------------- tables
    vd = lambda x: "NOT EVALUABLE" if x is None else ("PASS" if x else "FAIL")
    f3 = lambda c: f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c["n"] > 1 else f"{c['mean']:.4f}"
    f2 = lambda c: f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]" if c["n"] > 1 else f"{c['mean']:.3f}"
    for key in [k for k in S if k[1:2].isdigit()]:
        R = S[key]
        tabs.append(f"\n### {key} (seeds {R['seeds']}; float32-capped seeds: {R['capped_seeds'] or 'none'})\n")
        d0 = R["D0"]
        tabs.append("D0 gate (validation P0 KL, nats; 3-seed mean [95% CI]): " + "; ".join(
            f"{f} {f3(d0[f'val_kl_{f}'])}" for f in ("F", "B1", "B2", "R", "S1")) + f" -> **{'PASS' if d0['pass'] else 'FAIL'}**\n")
        d1 = R["D1"]
        tabs.append(f"D1: Spearman {f3(d1['spearman_beta_vs_info'])}, AUC {f3(d1['auc_blanket'])} -> **{'PASS' if d1['pass'] else 'FAIL'}**"
                    f" (strict {'PASS' if d1['pass_strict'] else 'FAIL'}); expected-entropy Spearman {f3(d1['expected_spearman'])};"
                    f" AUC parents-only {f3(d1['auc_parents_only'])}; oracle AUC of the exact information gain {f3(d1['oracle_auc_exact_info_gain'])}; Spearman(-s_hat) {f3(d1['spearman_neg_surprise_vs_info'])};"
                    f" Spearman(ln2 - h(q)) {f3(d1['spearman_selfentropy_vs_info'])}; frac beta = 0 {f3(d1['frac_beta_zero'])}\n")
        d2 = R["D2"]
        tabs.append(f"D2 (declared P2 = {d2['declared_variant']}, val log loss {json.dumps({k: round(v, 4) for k, v in d2['val_logloss_lw16'].items()})})"
                    f" -> **{'PASS' if d2['pass'] else 'FAIL'}**\n")
        tabs.append("| family | KL P0 | KL P2 M=0 | KL P2 M=16 | KL P4-Rand M=16 | cut vs P0 | cut vs P4 | cut vs P2 M=0 | (a) | (b) |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for fam, v in d2["families"].items():
            tabs.append(f"| {fam} | {f3(v['kl_P0'])} | {f3(v['kl_P2_M0'])} | {f3(v['kl_P2_M16'])} | {f3(v['kl_P4_M16'])} | {f2(v['cut_vs_P0'])} | "
                        f"{f2(v['cut_vs_P4'])} | {f2(v['cut_vs_P2M0'])} | {'PASS' if v['pass_a'] else 'FAIL'} | {'PASS' if v['pass_b'] else 'FAIL'} |\n")
        d3 = R["D3"]
        tabs.append(f"D3a: drift ratio m=2 {f3(d3['a']['ratio_m2'])}, m=5 {f3(d3['a']['ratio_m5'])} (scalar drift m=5 {f3(d3['a']['lw|drift_m5'])},"
                    f" dedup {f3(d3['a']['dd|drift_m5'])}; mean cos between redundant keys {f3(d3['a']['mean_offdiag_cos_redundant'])})"
                    f" -> **{'PASS' if d3['a']['pass'] else 'FAIL'}**\n")
        for fam in ("B1", "B2"):
            v = d3["b"][fam]
            tabs.append(f"D3b {fam}: KL both {f3(v['kl_both'])} vs fwd {f3(v['kl_fwd'])} / bwd {f3(v['kl_bwd'])}; log loss both {f3(v['ll_both'])}"
                        f" vs fwd {f3(v['ll_fwd'])} / bwd {f3(v['ll_bwd'])}; cos(k_fwd, k_bwd) {f3(v['cos_fwd_bwd'])} -> {vd(v['pass'])}\n")
        tabs.append(f"D3 overall -> **{vd(d3['pass'])}**\n")
        tabs.append("\nP1 arm (random-order chains + plain mixture) and the other selectors under the mixture: test KL and cut vs P0 "
                    "(3-seed mean [95% CI]; a positive cut = better than P0):\n\n| family | selector | KL M=2 | cut M=2 | KL M=8 | cut M=8 | KL M=16 | cut M=16 |\n|---|---|---|---|---|---|---|---|\n")
        for fam in ("F", "B1", "B2", "R"):
            for sel in SELS:
                cells = []
                for M in (2, 8, 16):
                    v = R["mixture_vs_P0"][f"{fam}|{sel}|mix|{M}"]
                    cells += [f"{v['kl']['mean']:.4f}", f2(v["cut_vs_P0"])]
                tabs.append(f"| {fam} | {sel} | " + " | ".join(cells) + " |\n")
        tabs.append("Secondary (base design rules; KL differences, negative = VoI better): " + "; ".join(
            f"{k}: VoI-Conf {f3(v['voi_minus_conf'])}, VoI-Rand {f3(v['voi_minus_rand'])}, VoI cut vs P0 {f2(v['voi_cut_vs_P0'])}"
            for k, v in R["secondary_base_rules"].items()) + "\n")
        tabs.append("\nKL vs M (lw pool; 3-seed mean):\n\n| family | selector | " + " | ".join(f"M={m}" for m in MS) + " | P0 |\n|---|---|" + "---|" * (len(MS) + 1) + "\n")
        for fam in FAMS:
            for sel in SELS:
                c = R["curves_kl"][fam][f"{sel}|lw"]
                tabs.append(f"| {fam} | {sel} | " + " | ".join(f"{x['mean']:.4f}" for x in c) + f" | {R['curves_kl'][fam]['P0']['mean']:.4f} |\n")
        tabs.append("\nKL vs M, mix aggregator (plain Monte Carlo average of the K = 16 chain claims; 3-seed mean):\n\n| family | selector | " + " | ".join(f"M={m}" for m in MS) + " |\n|---|---|" + "---|" * len(MS) + "\n")
        for fam in FAMS:
            for sel in SELS:
                c = R["curves_kl"][fam][f"{sel}|mix"]
                tabs.append(f"| {fam} | {sel} | " + " | ".join(f"{x['mean']:.4f}" for x in c) + " |\n")
        tabs.append("\nFraction of test queries whose lw pool at M = 0 returns exactly the network prior p0 (beta_int(Y | E) = 0): "
                    + "; ".join(f"{fam} {f2(R['frac_lw0_equals_prior'][fam])}" for fam in FAMS) + "\n")
        tabs.append("\nP6 aggregators at M=16 on the declared P2 claims (test KL / log loss, 3-seed mean):\n\n| family | mix (SNIS, unit w) | g0 plain | lw learned-w | dd (P5) | g1 product |\n|---|---|---|---|---|---|\n")
        p2 = d2["declared_variant"]
        for fam in FAMS:
            cells = []
            for pool in ("mix", "g0", "lw", "dd", "g1"):
                cells.append(f"{R['curves_kl'][fam][f'{p2}|{pool}'][-1]['mean']:.4f} / {R['logloss_M16'][fam][f'{p2}|{pool}']['mean']:.4f}")
            tabs.append(f"| {fam} | " + " | ".join(cells) + "|\n")
    first = next((k for k in S if k[1:2].isdigit()), None)
    if first:
        cq = S[first]["compute_per_query"]
        fp = cq["P0"]["flops_per_query"]
        tabs.append(f"\n### Compute per query (forward passes; FLOPs = passes x {fp:.3g} per forward pass; K = 16 chains)\n\n| selector | "
                    + " | ".join(f"M={m}" for m in MS) + " |\n|---|" + "---|" * len(MS) + "\n")
        tabs.append("| P0 | 1 | | | | | |\n")
        for sel in SELS:
            tabs.append(f"| {sel} | " + " | ".join(str(int(cq[f'{sel}|{m}']['passes_per_query'])) for m in MS) + " |\n")
    open(os.path.join(OUT, "tables.md"), "w").write("".join(tabs))
    # compact verdict table (one row per rule x training arm)
    vt = ["| rule, arm | D0 gate (val P0 KL F / B1 / B2; S1) | D1 (Spearman; AUC) | D2 (declared P2; cut vs P0 at M=16, F / B1 / B2) | "
          "D3a (drift ratio m=5; m=2) | D3b (KL both vs best single, B1; B2) | D3 |",
          "|---|---|---|---|---|---|---|"]
    for key in [k for k in S if k[1:2].isdigit()]:
        R = S[key]
        d0, d1, d2, d3 = R["D0"], R["D1"], R["D2"], R["D3"]
        c0 = " / ".join(f"{d0[f'val_kl_{f}']['mean']:.3f}" for f in ("F", "B1", "B2"))
        cuts = " / ".join(f"{d2['families'][f]['cut_vs_P0']['mean']:+.2f}" for f in ("F", "B1", "B2"))
        b3 = []
        for fam in ("B1", "B2"):
            v = d3["b"][fam]
            best = min(v["kl_fwd"]["mean"], v["kl_bwd"]["mean"]) if v["kl_fwd"]["n"] and v["kl_bwd"]["n"] else float("nan")
            b3.append(f"{v['kl_both']['mean']:.3f} vs {best:.3f}")
        vt.append(f"| {key} | **{vd(d0['pass'])}** ({c0}; {d0['val_kl_S1']['mean']:.3f}) | **{vd(d1['pass'])}** "
                  f"({d1['spearman_beta_vs_info']['mean']:.3f}; {d1['auc_blanket']['mean']:.3f}) | **{vd(d2['pass'])}** "
                  f"({d2['declared_variant']}; {cuts}) | **{vd(d3['a']['pass'])}** ({d3['a']['ratio_m5']['mean']:.3f}; "
                  f"{d3['a']['ratio_m2']['mean']:.3f}) | **{vd(d3['b']['pass'])}** ({'; '.join(b3)}) | **{vd(d3['pass'])}** |")
    open(os.path.join(OUT, "verdicts.md"), "w").write("\n".join(vt) + "\n")

    # ---------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = {"stiff": "#1f6feb", "voi": "#d1242f", "conf": "#9a6700", "rand": "#57606a"}
        for key in [k for k in S if k[1:2].isdigit()]:
            R = S[key]
            fig, axs2 = plt.subplots(2, 4, figsize=(16, 7.6))
            for row, (pool, pname) in enumerate((("lw", "stiffness-weighted pool (pre-registered)"), ("mix", "plain mixture (P1 aggregator)"))):
                axs = axs2[row]
                for ax, fam in zip(axs, FAMS):
                    for sel in SELS:
                        c = R["curves_kl"][fam][f"{sel}|{pool}"]
                        m = [x["mean"] for x in c]
                        lo = [x["lo"] for x in c]
                        hi = [x["hi"] for x in c]
                        xs = np.arange(len(MS))
                        ax.plot(xs, m, "-o", color=cols[sel], label=sel, ms=3)
                        ax.fill_between(xs, lo, hi, color=cols[sel], alpha=0.12)
                    ax.axhline(R["curves_kl"][fam]["P0"]["mean"], color="k", ls="--", lw=1, label="P0 one pass")
                    ax.set_xticks(np.arange(len(MS)))
                    ax.set_xticklabels([str(m) for m in MS])
                    ax.set_xlabel("M (cells materialised per chain; K = 16 chains)")
                    ax.set_title(f"{key} {fam}: {pname}", fontsize=8)
                    ax.set_ylabel("KL(exact || estimate), nats")
                    ax.set_yscale("log")
                axs[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"kl-vs-M-{key}.png"), dpi=110)
            plt.close(fig)
        # KL vs forward passes per query (compute axis), lw pool
        for key in [k for k in S if k[1:2].isdigit()]:
            R = S[key]
            cq = R["compute_per_query"]
            fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
            for ax, fam in zip(axs, FAMS):
                for sel in SELS:
                    xs = [cq[f"{sel}|{M}"]["passes_per_query"] for M in MS]
                    ax.plot(xs, [x["mean"] for x in R["curves_kl"][fam][f"{sel}|lw"]], "-o", color=cols[sel], ms=3, label=sel)
                ax.plot([1], [R["curves_kl"][fam]["P0"]["mean"]], "k*", ms=10, label="P0")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("forward passes per query")
                ax.set_ylabel("KL, nats")
                ax.set_title(f"{key} {fam}")
            axs[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"kl-vs-compute-{key}.png"), dpi=110)
            plt.close(fig)
        # D1 scatter (seed 0 of each rule x arm): beta_int vs exact information gain
        fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
        for ax, (rule, arm) in zip(axs, [(r, a) for r in RULES for a in ARMS]):
            f = next((os.path.join(b_, f"r{rule}-{arm}-s0", "d1-pairs.csv") for b_ in RUN_DIRS
                      if os.path.exists(os.path.join(b_, f"r{rule}-{arm}-s0", "d1-pairs.csv"))), None)
            if f is None:
                continue
            rows = list(csv.DictReader(open(f)))
            b = np.array([float(r["beta"]) for r in rows])
            ig = np.log(2) - np.array([float(r["exact_H"]) for r in rows])
            bl = np.array([int(r["blanket"]) for r in rows]).astype(bool)
            ax.scatter(ig[~bl], b[~bl], s=4, alpha=0.4, color="#57606a", label="no local blanket in S")
            ax.scatter(ig[bl], b[bl], s=4, alpha=0.4, color="#1f6feb", label="S holds a local blanket")
            ax.set_xlabel("ln 2 - H_exact(X_i | x_S), nats")
            ax.set_ylabel("beta_int(i | S)")
            ax.set_title(f"r{rule}-{arm} seed 0")
        axs[0].legend(fontsize=7, markerscale=3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "d1-stiffness-scatter.png"), dpi=110)
        plt.close(fig)
    except Exception as ex:  # figures are optional; tables carry every number
        print("figures skipped:", ex)
    if not os.environ.get("ECA_QUIET"):
        print(open(os.path.join(OUT, "tables.md")).read()[:6000])


if __name__ == "__main__":
    main()
