#!/usr/bin/env python3
"""E-MS report: reads runs/s<seed>/eval/*.json|npz and train_meta.json for seeds 101-103, applies the PREREG-MS.md
decision rules (3-seed mean + 95% t-interval, df = 2), writes report/verdicts.csv, report/per_seed.csv,
report/summary.json, report/tables.md, report/audit.md and PNG curves. The audit re-derives every headline number
from the saved per-item arrays (npz), independently of the json summaries.
   python3 report_ms.py [--seeds 101,102,103]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("MS_RUNS", os.path.join(HERE, "runs"))
OUT = os.environ.get("MS_REPORT", os.path.join(HERE, "report"))
T975 = 4.302652729911275


def tci(x):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = T975 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h


def fmt(m, lo, hi, p=3):
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def load(seed, part):
    j = os.path.join(RUNS, f"s{seed}", "eval", f"{part}.json")
    z = os.path.join(RUNS, f"s{seed}", "eval", f"{part}.npz")
    return (json.load(open(j)) if os.path.exists(j) else None), (np.load(z) if os.path.exists(z) else None)


def kl_rows(p, q, eps=1e-9):
    q = np.clip(q, eps, 1)
    return np.where(p > 0, p * (np.log(np.clip(p, eps, 1)) - np.log(q)), 0.0).sum(-1)


def jsd_bern(a, b, eps=1e-9):
    a, b = np.clip(a, eps, 1 - eps), np.clip(b, eps, 1 - eps)
    m = 0.5 * (a + b)
    kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="101,102,103")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(OUT, exist_ok=True)
    V = []          # verdict rows: rule, criterion, per-seed values, mean, lo, hi, threshold, pass
    EXPLO = {"on": False}  # set once the D0 gate is known to have failed: later sections are exploratory
    PS = []         # per-seed rows
    md = []
    audit = []

    def verdict(rule, crit, vals, thr, null, direction=">=", p=3):
        m, lo, hi = tci(vals)
        if direction == ">=":
            ok = m >= thr and lo > null
        elif direction == "<=":
            ok = m <= thr and hi < null if null is not None else m <= thr
        else:
            ok = m > thr and lo > null
        if rule != "E1-D0" and EXPLO["on"]:
            V.append({"rule": rule + " (exploratory)", "criterion": crit, "per_seed": [round(float(v), 5) for v in vals],
                      "mean": m, "lo": lo, "hi": hi, "threshold": f"{direction} {thr}",
                      "pass": f"n/a: exploratory, bar {'met' if ok else 'not met'}"})
            return ok
        V.append({"rule": rule, "criterion": crit, "per_seed": [round(float(v), 5) for v in vals], "mean": m, "lo": lo,
                  "hi": hi, "threshold": f"{direction} {thr}", "pass": bool(ok)})
        return ok

    def vword(ok):
        if EXPLO["on"]:
            return "pre-registered bar " + ("met" if ok else "not met") + " (EXPLORATORY: not a rule test, the D0 gate failed)"
        return "**PASS**" if ok else "**FAIL**"

    def head(txt):
        return txt + (" -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)\n"
                      if EXPLO["on"] else "\n")

    # ------------------------------------------------------------------ D0
    d0 = {s: load(s, "d0") for s in seeds}
    have_d0 = all(d0[s][0] is not None for s in seeds)
    if have_d0:
        acc = [d0[s][0]["battle_acc_ply0"] for s in seeds]
        nll = [d0[s][0]["nll_5known"] for s in seeds]
        kl0 = [d0[s][0]["kl0_full"] for s in seeds]
        klc = [d0[s][0]["kl0_constraint"] for s in seeds]
        g1 = verdict("E1-D0", "1-step battle accuracy (ply 0, Z materialised)", acc, 0.97, -1)
        g2 = tci(nll)[0] <= 0.03
        V.append({"rule": "E1-D0", "criterion": "5-known-slot NLL (nats)", "per_seed": [round(v, 5) for v in nll],
                  "mean": tci(nll)[0], "lo": tci(nll)[1], "hi": tci(nll)[2], "threshold": "<= 0.03", "pass": bool(g2)})
        g3 = verdict("E1-D0", "0-step belief KL to P*_full per hidden slot (nats)", kl0, 0.12, -1)
        gate = bool(g1 and g2 and g3)
        md.append("### E1-D0 (gate)\n")
        md.append("| seed | battle acc ply 0 | battle acc all plies | 5-known NLL | 0-step KL vs P*_full | 0-step KL vs P*_c | n positions |")
        md.append("|---|---|---|---|---|---|---|")
        for s in seeds:
            r = d0[s][0]
            md.append(f"| {s} | {r['battle_acc_ply0']:.4f} | {r['battle_acc_allply']:.4f} | {r['nll_5known']:.4f} | "
                      f"{r['kl0_full']:.4f} | {r['kl0_constraint']:.4f} | {r['n_pos']} |")
        md.append(f"| mean [95% CI] | {fmt(*tci(acc), 4)} | | {fmt(*tci(nll), 4)} | {fmt(*tci(kl0), 4)} | {fmt(*tci(klc), 4)} | |")
        md.append(f"\nGate: battle {g1}, five-known {g2}, belief gap {g3} -> **{'PASS' if gate else 'FAIL'}**. "
                  f"KL(P*_full || P*_c) on these positions (data property): {d0[seeds[0]][0]['kl_full_vs_constraint_oracle']:.4f} nats.\n")
        for s in seeds:
            z = d0[s][1]
            audit.append(("D0 kl0_full", s, d0[s][0]["kl0_full"], float(z["kl0_full"].mean())))
            audit.append(("D0 battle acc", s, d0[s][0]["battle_acc_ply0"], float(z["acc0"].mean())))
            audit.append(("D0 5-known NLL", s, d0[s][0]["nll_5known"], float(z["nll5"].mean())))
    else:
        gate = None
    EXPLO["on"] = gate is False
    # ------------------------------------------------------------------ D1
    d1 = {s: load(s, "d1") for s in seeds}
    if all(d1[s][0] is not None for s in seeds):
        red = [d1[s][0]["kl_reduction_vs_row_major"] for s in seeds]
        invr = [d1[s][0]["row_major"]["invalid_rate"] for s in seeds]
        md.append(head("### E1-D1 (any-order belief unmasking, K = 8)"))
        md.append("| seed | KL 0-step | KL row_major | KL random | KL stiff_voi | invalid row_major | invalid random | invalid stiff_voi |")
        md.append("|---|---|---|---|---|---|---|---|")
        for s in seeds:
            r = d1[s][0]
            md.append(f"| {s} | {r['zero_step_kl']:.4f} | {r['row_major']['kl']:.4f} | {r['random']['kl']:.4f} | "
                      f"{r['stiff_voi']['kl']:.4f} | {r['row_major']['invalid_rate']:.4f} | {r['random']['invalid_rate']:.4f} | "
                      f"{r['stiff_voi']['invalid_rate']:.4f} |")
        ok1 = verdict("E1-D1", "KL reduction stiff_voi vs row_major", red, 0.20, 0)
        floor = all(v < 0.005 for v in invr)
        if floor:
            V.append({"rule": "E1-D1", "criterion": "invalid-rate reduction", "per_seed": [], "mean": float("nan"),
                      "lo": float("nan"), "hi": float("nan"), "threshold": ">= 0.30", "pass": None,
                      "note": "row_major invalid rate < 0.5% in every seed: at floor, not testable"})
            ok2 = True
        else:
            ired = [d1[s][0]["invalid_reduction_vs_row_major"] for s in seeds]
            ok2 = verdict("E1-D1", "invalid-rate reduction stiff_voi vs row_major", ired, 0.30, 0)
        md.append(f"\nKL reduction stiff_voi vs row_major: {fmt(*tci(red))}. {vword(ok1 and ok2)}"
                  f"{' (invalid-rate part at floor)' if floor else ''}.\n")
        for s in seeds:
            z = d1[s][1]
            for o in ("row_major", "random", "stiff_voi"):
                audit.append((f"D1 KL {o}", s, d1[s][0][o]["kl"], float(z[f"kl_{o}"].mean())))
                audit.append((f"D1 invalid {o}", s, d1[s][0][o]["invalid_rate"], float(z[f"invalid_{o}"].mean())))
        PS += [{"rule": "E1-D1", "seed": s, **{f"kl_{o}": d1[s][0][o]["kl"] for o in ("row_major", "random", "stiff_voi")}} for s in seeds]
    # ------------------------------------------------------------------ D2
    d2 = {s: load(s, "d2") for s in seeds}
    if all(d2[s][0] is not None for s in seeds):
        aggs = ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3")
        md.append(head("### E1-D2 (policy independence under flank skew, K = 24; reference K = 128)"))
        md.append("| seed | " + " | ".join(f"JSD {g}" for g in aggs) + " | " + " | ".join(f"MSE {g}" for g in aggs) +
                  " | flank gap | key cos within / across |")
        md.append("|---|" + "---|" * 10)
        for s in seeds:
            r = d2[s][0]
            md.append(f"| {s} | " + " | ".join(f"{r[g]['jsd']:.2e}" for g in aggs) + " | " +
                      " | ".join(f"{r[g]['mse']:.2e}" for g in aggs) +
                      f" | {r['flank_gap_mean']:.3f} | {r['key_cos_within']:.3f} / {r['key_cos_across']:.3f} |")
        allok = True
        for ours in ("kc_dedup", "attn_L3"):
            for base in ("mix", "ataraxos_mmd"):
                rj = [1 - d2[s][0][ours]["jsd"] / d2[s][0][base]["jsd"] for s in seeds]
                rm = [1 - d2[s][0][ours]["mse"] / d2[s][0][base]["mse"] for s in seeds]
                allok &= verdict("E1-D2", f"JSD reduction {ours} vs {base}", rj, 0.50, 0)
                allok &= verdict("E1-D2", f"value-MSE reduction {ours} vs {base}", rm, 0.25, 0)
                md.append(f"- {ours} vs {base}: JSD reduction {fmt(*tci(rj))}, MSE reduction {fmt(*tci(rm))}")
        md.append(f"\n{vword(allok)} (all 8 comparisons would have to meet it).\n")
        for s in seeds:
            z = d2[s][1]
            for g in aggs:
                audit.append((f"D2 JSD {g}", s, d2[s][0][g]["jsd"], float(jsd_bern(z[f"muL_{g}"], z[f"muR_{g}"]).mean())))
                audit.append((f"D2 MSE {g}", s, d2[s][0][g]["mse"],
                              float((0.5 * ((z[f"muL_{g}"] - z["vref"]) ** 2 + (z[f"muR_{g}"] - z["vref"]) ** 2)).mean())))
    # ------------------------------------------------------------------ D3
    d3 = {s: load(s, "d3") for s in seeds}
    if all(d3[s][0] is not None for s in seeds):
        md.append(head("### E1-D3 (theta_trust vs output entropy)"))
        md.append("| seed | Spearman trust | Spearman stiff | strata trust | strata stiff | delusional trust | delusional stiff | delusional uniform | n rollouts |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for s in seeds:
            r = d3[s][0]
            md.append(f"| {s} | {r['spearman_trust']:.3f} | {r['spearman_stiff']:.3f} | {r['strata_trust']:.3f} | {r['strata_stiff']:.3f} | "
                      f"{r['delusional_trust']:.4f} | {r['delusional_stiff']:.4f} | {r['delusional_uniform']:.4f} | {r['n_rollouts']} |")
        dov = [d3[s][0]["spearman_trust"] - d3[s][0]["spearman_stiff"] for s in seeds]
        dst = [d3[s][0]["strata_trust"] - d3[s][0]["strata_stiff"] for s in seeds]
        o1 = verdict("E1-D3", "Spearman(trust) - Spearman(stiff), overall", dov, 0.10, 0)
        o2 = verdict("E1-D3", "same, mean within 10 entropy strata", dst, 0.15, 0)
        nst = sum(int(d3[s][1]["delusional_stiff"].sum()) for s in seeds)
        if nst < 10:
            V.append({"rule": "E1-D3", "criterion": "delusional-rate cut trust vs stiff", "per_seed": [], "mean": float("nan"),
                      "lo": float("nan"), "hi": float("nan"), "threshold": ">= 0.40", "pass": None,
                      "note": f"stiff made {nst} delusional choices over 3 seeds (< 10): at floor, not testable"})
            o3 = None
        else:
            cut = [1 - d3[s][0]["delusional_trust"] / d3[s][0]["delusional_stiff"] if d3[s][0]["delusional_stiff"] > 0 else float("nan") for s in seeds]
            o3 = verdict("E1-D3", "delusional-rate cut trust vs stiff", cut, 0.40, 0)
        md.append(f"\nSpearman gain overall {fmt(*tci(dov))}, within strata {fmt(*tci(dst))}; stiff delusional choices over 3 seeds: {nst}. "
                  f"Part 1: {vword(o1 and o2)}; part 2: {'at floor, not testable' if o3 is None else vword(o3)}.\n")
        for s in seeds:
            z = d3[s][1]
            y = -z["kl"]
            audit.append(("D3 Spearman trust", s, d3[s][0]["spearman_trust"], spearman(z["beta"], y)))
            audit.append(("D3 Spearman stiff", s, d3[s][0]["spearman_stiff"], spearman(-z["H"], y)))
            audit.append(("D3 delusional stiff", s, d3[s][0]["delusional_stiff"], float(z["delusional_stiff"].mean())))
            audit.append(("D3 delusional trust", s, d3[s][0]["delusional_trust"], float(z["delusional_trust"].mean())))
    # ------------------------------------------------------------------ D4
    d4 = {s: load(s, "d4") for s in seeds}
    if all(d4[s][0] is not None for s in seeds):
        md.append(head("### E1-D4.1 (LIR mask vs fused distillation)"))
        md.append("| seed | theta_real hash unchanged | belief KL LIR | belief KL fused | change fused | battle acc LIR / fused | value MSE theta_real | value MSE theta_amp | value MSE fused |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for s in seeds:
            r = d4[s][0]
            md.append(f"| {s} | {r['lir_param_hash_identical'] and r['lir_saved_equals_used']} | {r['lir_metrics']['kl0_full']:.4f} | "
                      f"{r['fused_metrics']['kl0_full']:.4f} | {r['belief_kl_change_fused']:+.3f} | "
                      f"{r['lir_metrics']['battle_acc_ply0']:.4f} / {r['fused_metrics']['battle_acc_ply0']:.4f} | "
                      f"{r['value_mse_real']:.4f} | {r['value_mse_amp']:.4f} | {r['value_mse_fused']:.4f} |")
        zero = all(d4[s][0]["lir_param_hash_identical"] and d4[s][0]["lir_saved_equals_used"] and
                   abs(d4[s][0]["lir_metrics"]["kl0_full"] - d0[s][0]["kl0_full"]) < 1e-9 and
                   abs(d4[s][0]["lir_metrics"]["battle_acc_ply0"] - d0[s][0]["battle_acc_ply0"]) < 1e-12 for s in seeds) if have_d0 else False
        drift = [abs(d4[s][0]["lir_metrics"]["kl0_full"] - d0[s][0]["kl0_full"]) for s in seeds] if have_d0 else []
        md.append(f"\nLIR belief-KL drift (D4 recomputation minus D0) per seed: {drift}.")
        V.append({"rule": "E1-D4.1", "criterion": "LIR: theta_real bit-identical, belief KL drift 0.000", "per_seed": [],
                  "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "threshold": "identical", "pass": bool(zero)})
        vr = [d4[s][0]["value_mse_reduction_amp"] for s in seeds]
        o1 = verdict("E1-D4.1", "theta_amp 0-step value-MSE reduction", vr, 0.20, 0)
        fc = [d4[s][0]["belief_kl_change_fused"] for s in seeds]
        o2 = verdict("E1-D4.1", "fused belief-KL increase", fc, 0.20, 0, direction=">")
        md.append(f"\nValue-MSE reduction {fmt(*tci(vr))}; fused belief-KL change {fmt(*tci(fc))}. "
                  f"Verdict: **{'PASS' if (zero and o1 and o2) else 'FAIL'}** (zero drift {zero}, value {o1}, fused {o2}).\n")
        for s in seeds:
            z = d4[s][1]
            audit.append(("D4 value MSE amp", s, d4[s][0]["value_mse_amp"], float(((z["v_amp"] - z["v_true"]) ** 2).mean())))
            audit.append(("D4 value MSE real", s, d4[s][0]["value_mse_real"], float(((z["v_real"] - z["v_true"]) ** 2).mean())))
    d4u = {s: load(s, "d4u") for s in seeds}
    if all(d4u[s][0] is not None for s in seeds):
        md.append(head("### E1-D4.2 (zero-retraining utility swap)"))
        md.append("| seed | Miner capture U_flag | Miner capture U_miner | ratio | own Miner lost U_flag / U_miner | eff. win U_flag / U_miner |")
        md.append("|---|---|---|---|---|---|")
        for s in seeds:
            r = d4u[s][0]
            md.append(f"| {s} | {r['flag']['miner_capture_rate']:.3f} | {r['miner']['miner_capture_rate']:.3f} | {r['capture_ratio']:.2f} | "
                      f"{r['flag']['own_miner_lost_rate']:.3f} / {r['miner']['own_miner_lost_rate']:.3f} | "
                      f"{r['flag']['effective_win']:.3f} / {r['miner']['effective_win']:.3f} |")
        ra = [d4u[s][0]["capture_ratio"] for s in seeds]
        o = verdict("E1-D4.2", "Miner-capture ratio U_miner / U_flag", ra, 2.0, 1.0)
        md.append(f"\nRatio {fmt(*tci(ra), 2)}. {vword(o)}.\n")
        for s in seeds:
            z = d4u[s][1]
            audit.append(("D4.2 capture U_flag", s, d4u[s][0]["flag"]["miner_capture_rate"], float(z["cap_flag"].mean())))
            audit.append(("D4.2 capture U_miner", s, d4u[s][0]["miner"]["miner_capture_rate"], float(z["cap_miner"].mean())))
    # ------------------------------------------------------------------ D5
    d5 = {s: load(s, "d5") for s in seeds}
    if all(d5[s][0] is not None for s in seeds):
        md.append(head("### E1-D5 (tournaments, 400 games per pair per seed, K = 16)"))
        md.append("| seed | vs P0 | vs Ataraxos-1Step | vs Tabular-MCTS |")
        md.append("|---|---|---|---|")
        opps = ("P0", "Ataraxos-1Step", "Tabular-MCTS")
        for s in seeds:
            r = d5[s][0]
            md.append(f"| {s} | " + " | ".join(f"{r[o]['effective_win']:.3f} ({r[o]['wins']}/{r[o]['draws']}/{r[o]['losses']})" for o in opps) + " |")
        allok = True
        for o in opps:
            w = [d5[s][0][o]["effective_win"] for s in seeds]
            allok &= verdict("E1-D5", f"Unified-LIR effective win vs {o}", w, 0.55, 0.5)
            md.append(f"- vs {o}: {fmt(*tci(w))}")
        md.append(f"\n{vword(allok)}. Cells: effective win (wins/draws/losses).\n")
        for s in seeds:
            z = d5[s][1]
            for o in opps:
                audit.append((f"D5 vs {o}", s, d5[s][0][o]["effective_win"], float(z[f"res_{o}"].mean())))
    # ------------------------------------------------------------------ outputs
    with open(os.path.join(OUT, "verdicts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "criterion", "per_seed", "mean", "ci_lo", "ci_hi", "threshold", "pass", "note"])
        for v in V:
            w.writerow([v["rule"], v["criterion"], " ".join(str(x) for x in v["per_seed"]), f"{v['mean']:.5f}",
                        f"{v['lo']:.5f}", f"{v['hi']:.5f}", v["threshold"], v["pass"], v.get("note", "")])
    vt = ["| rule | criterion | per seed | mean [95% CI] | threshold | pass |", "|---|---|---|---|---|---|"]
    for v in V:
        ps = ", ".join(f"{x:.4g}" for x in v["per_seed"]) if v["per_seed"] else v.get("note", "")
        mm = fmt(v["mean"], v["lo"], v["hi"], 4) if not math.isnan(v["mean"]) else "n/a"
        vt.append(f"| {v['rule']} | {v['criterion']} | {ps} | {mm} | {v['threshold']} | {v['pass']} |")
    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(vt) + "\n\n" + "\n".join(md) + "\n")
    aud = ["| quantity | seed | json summary | re-derived from npz | abs diff |", "|---|---|---|---|---|"]
    worst = 0.0
    for q, s, j, z in audit:
        d = abs(j - z) if (j == j and z == z) else float("nan")
        worst = max(worst, d if d == d else 0)
        aud.append(f"| {q} | {s} | {j:.6f} | {z:.6f} | {d:.2e} |")
    aud.append(f"\nLargest absolute difference: {worst:.2e} over {len(audit)} re-derived numbers.")
    open(os.path.join(OUT, "audit.md"), "w").write("\n".join(aud) + "\n")
    json.dump({"verdicts": V, "gate": gate}, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    # training curves
    rows = []
    for s in seeds:
        p = os.path.join(RUNS, f"s{s}", "train_meta.json")
        if not os.path.exists(p):
            continue
        tm = json.load(open(p))
        for h in tm["real"]["hist"]:
            rows.append([s, h["step"], *h["train"], *h["cal"]])
    with open(os.path.join(OUT, "train_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "step", "train_belief", "train_policy", "train_transition", "train_value", "cal_belief",
                    "cal_policy", "cal_transition", "cal_value"])
        w.writerows(rows)
    try:
        sys.path.insert(0, "/tmp/ms-pylib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if rows:
            R = np.array(rows, float)
            fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
            names = ["belief", "policy", "transition", "value"]
            for k in range(4):
                for s in seeds:
                    m = R[:, 0] == s
                    ax[k].plot(R[m, 1], R[m, 6 + k], label=f"seed {s} (cal)")
                ax[k].set_title(f"theta_real {names[k]} CE (cal split)")
                ax[k].set_xlabel("step")
            ax[0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "train_curves.png"), dpi=110)
            plt.close(fig)
        if all(d1[s][0] is not None for s in seeds):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            orders = ("0-step", "row_major", "random", "stiff_voi")
            vals = [[d1[s][0]["zero_step_kl"]] + [d1[s][0][o]["kl"] for o in orders[1:]] for s in seeds]
            vals = np.array(vals)
            ax.bar(range(4), vals.mean(0), yerr=[vals.mean(0) - np.array([tci(vals[:, i])[1] for i in range(4)]),
                                                 np.array([tci(vals[:, i])[2] for i in range(4)]) - vals.mean(0)], capsize=4)
            ax.set_xticks(range(4))
            ax.set_xticklabels(orders)
            ax.set_ylabel("KL(P*_full || belief) per hidden slot")
            ax.set_title("D1 belief KL (3 seeds, 95% CI)")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "d1_kl.png"), dpi=110)
            plt.close(fig)
            with open(os.path.join(OUT, "d1_kl.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["seed"] + list(orders))
                for s, v in zip(seeds, vals):
                    w.writerow([s] + [f"{x:.6f}" for x in v])
        if all(d3[s][0] is not None for s in seeds):
            fig, ax = plt.subplots(figsize=(5, 3.2))
            st = np.array([d3[s][0]["strata_trust_each"] for s in seeds], float)
            ss = np.array([d3[s][0]["strata_stiff_each"] for s in seeds], float)
            ax.plot(range(10), np.nanmean(st, 0), "o-", label="theta_trust beta")
            ax.plot(range(10), np.nanmean(ss, 0), "s-", label="-H(q) (stiff)")
            ax.axhline(0, color="gray", lw=0.5)
            ax.set_xlabel("entropy stratum (decile of H(q_r))")
            ax.set_ylabel("Spearman with -KL(Q* || q_r)")
            ax.set_title("D3 within-stratum Spearman (3-seed mean)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "d3_strata.png"), dpi=110)
            plt.close(fig)
            with open(os.path.join(OUT, "d3_strata.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stratum", "trust_mean", "stiff_mean"])
                for i in range(10):
                    w.writerow([i, f"{np.nanmean(st[:, i]):.5f}", f"{np.nanmean(ss[:, i]):.5f}"])
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print(open(os.path.join(OUT, "tables.md")).read())
    print("audit worst diff", worst)


if __name__ == "__main__":
    main()
