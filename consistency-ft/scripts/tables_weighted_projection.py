"""Markdown tables + flat chart rows for scripts/weighted_projection.py output.

Usage: tables_weighted_projection.py /home/agent/data/weighted-projection-20260924/full.json \
           docs/weighted-projection-20260924-charts.json > tables.md
Writes the chart JSON: a flat "rows" list (dataset, job, stratum, method, joint/all12 LL, deltas + CIs,
recovery) plus fitted parameters, dependence captured and control log odds ratios.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MODEL_ONLY = ["raw", "eq_kl", "eq_lsl", "bc_kl", "rw_kl", "rw_lsl", "rwg_kl", "bcx_kl", "rwx_kl",
              "fill_and3", "fill_and4", "fill_litand"]
HYB = ["h_eq", "h_bc", "h_rw", "h_rwg", "h_bcx", "h_rwx", "h_and3", "h_and4", "h_litand", "h_and3_s", "h_and4_s",
       "h_litand_s"]


def f(x):
    return f"{x:+.3f}"


def ci(v):
    return f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"


def main():
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    rows = []
    for j in d["jobs"]:
        for s, e in j["by"].items():
            for m, v in e["ll"].items():
                r = {"dataset": j["dataset"], "job": j["name"], "stratum": s, "pairs": e["pairs"], "method": m,
                     "joint_ll": v["joint"], "all12_ll": v["all12"]}
                for ref, dd in e["vs"].items():
                    if m in dd:
                        r[f"joint_minus_{ref}"] = dd[m]["joint"]
                        r[f"all12_minus_{ref}"] = dd[m]["all12"]
                if "recovery" in e:
                    r["recovery_of_known_gain"] = e["recovery"].get(m)
                rows.append(r)
    charts = {"description": "docs/weighted-projection-20260924.md; code scripts/weighted_projection.py",
              "rows_units": ("mean per-pair log loss (joint = 8 AND/OR forms; all12), clip 0.01; *_minus_* = [diff, ci lo, "
                             "ci hi], negative = method better; cluster bootstrap 2,000; tuned parameters cross-fitted (5 folds)"),
              "external_params": d["external_params"],
              "jobs": [{k: j[k] for k in j if k != "by"} for j in d["jobs"]], "rows": rows}
    Path(sys.argv[2]).write_text(json.dumps(charts, separators=(",", ":")))

    out = []
    P = out.append
    for ds in ("structured", "market_context", "old_1066"):
        jobs = [j for j in d["jobs"] if j["dataset"] == ds]
        if not jobs:
            continue
        P(f"\n### {ds}: model-only methods, Δ vs equal-weight KL projection (joint / all-12)\n")
        P("| job | stratum | eq_kl joint | " + " | ".join(MODEL_ONLY[2:]) + " |")
        P("|" + "---|" * (3 + len(MODEL_ONLY) - 2))
        for j in jobs:
            for s, e in j["by"].items():
                cells = []
                for m in MODEL_ONLY[2:]:
                    if m in e["vs"]["eq_kl"]:
                        v = e["vs"]["eq_kl"][m]
                        cells.append(f"{ci(v['joint'])} / {f(v['all12'][0])}")
                    else:
                        cells.append("—")
                P(f"| {j['name']} | {s} ({e['pairs']}) | {e['ll']['eq_kl']['joint']:.3f} | " + " | ".join(cells) + " |")
        P(f"\n### {ds}: Δ vs raw (joint / all-12)\n")
        P("| job | stratum | raw joint | raw all12 | " + " | ".join(MODEL_ONLY[1:]) + " |")
        P("|" + "---|" * (4 + len(MODEL_ONLY) - 1))
        for j in jobs:
            e = j["by"]["all"]
            cells = [f"{f(e['vs']['raw'][m]['joint'][0])} / {f(e['vs']['raw'][m]['all12'][0])}" if m in e["vs"]["raw"] else "—"
                     for m in MODEL_ONLY[1:]]
            P(f"| {j['name']} | all ({e['pairs']}) | {e['ll']['raw']['joint']:.3f} | {e['ll']['raw']['all12']:.3f} | "
              + " | ".join(cells) + " |")
        if ds == "old_1066":
            continue
        P(f"\n### {ds}: market variants, joint Δ vs market + independence [95% CI]\n")
        refcols = ["market_known"] if ds == "structured" else []
        P("| job | stratum | market indep | " + " | ".join(refcols + HYB) + " |")
        P("|" + "---|" * (3 + len(refcols) + len(HYB)))
        for j in jobs:
            for s, e in j["by"].items():
                cells = [ci(e["vs"]["market_indep"][m]["joint"]) if m in e["vs"]["market_indep"] else "—" for m in refcols + HYB]
                P(f"| {j['name']} | {s} ({e['pairs']}) | {e['ll']['market_indep']['joint']:.3f} | " + " | ".join(cells) + " |")
        P(f"\n### {ds}: market variants, joint Δ vs the old hybrid h_eq (paired) [95% CI]\n")
        P("| job | stratum | " + " | ".join(HYB[1:]) + " |")
        P("|" + "---|" * (2 + len(HYB) - 1))
        for j in jobs:
            for s, e in j["by"].items():
                if "h_eq" not in e["vs"]:
                    continue
                P(f"| {j['name']} | {s} ({e['pairs']}) | " + " | ".join(
                    ci(e["vs"]["h_eq"][m]["joint"]) if m in e["vs"]["h_eq"] else "—" for m in HYB[1:]) + " |")
        P(f"\n### {ds}: market variants, all-12 Δ vs market + independence\n")
        P("| job | stratum | " + " | ".join(refcols + HYB) + " |")
        P("|" + "---|" * (2 + len(refcols) + len(HYB)))
        for j in jobs:
            for s, e in j["by"].items():
                cells = [ci(e["vs"]["market_indep"][m]["all12"]) if m in e["vs"]["market_indep"] else "—" for m in refcols + HYB]
                P(f"| {j['name']} | {s} ({e['pairs']}) | " + " | ".join(cells) + " |")
        if ds == "structured":
            P("\n### structured: share of the known-relation gain recovered (joint; (indep − method) / (indep − known))\n")
            P("| job | stratum | known gain | " + " | ".join(MODEL_ONLY[:1] + HYB) + " |")
            P("|" + "---|" * (3 + 1 + len(HYB)))
            for j in jobs:
                for s, e in j["by"].items():
                    if "recovery" not in e:
                        continue
                    gain = e["ll"]["market_indep"]["joint"] - e["ll"]["market_known"]["joint"]
                    P(f"| {j['name']} | {s} | {gain:.3f} | " + " | ".join(
                        f"{100 * e['recovery'][m]:.0f}%" if m in e["recovery"] else "—" for m in MODEL_ONLY[:1] + HYB) + " |")
            P("\n### structured: joint Δ vs market + known relation [95% CI]\n")
            P("| job | stratum | " + " | ".join(["eq_kl", "rw_kl", "fill_and3", "fill_and4"] + HYB) + " |")
            P("|" + "---|" * (2 + 4 + len(HYB)))
            for j in jobs:
                for s, e in j["by"].items():
                    P(f"| {j['name']} | {s} | " + " | ".join(ci(e["vs"]["market_known"][m]["joint"]) for m in
                                                           ["eq_kl", "rw_kl", "fill_and3", "fill_and4"] + HYB) + " |")
            P("\n### structured: dependence captured D (median / share ≥ 0.5), own marginals\n")
            ms = list(jobs[0]["dependence_captured"])
            P("| job | type | " + " | ".join(ms) + " |")
            P("|" + "---|" * (2 + len(ms)))
            for j in jobs:
                for t in ("threshold", "date", "mc_excl"):
                    P(f"| {j['name']} | {t} | " + " | ".join(
                        f"{j['dependence_captured'][m][t]['median']:.2f} / {j['dependence_captured'][m][t]['share_ge_half']:.2f}"
                        for m in ms) + " |")
            P("\n### structured controls: log odds ratio (median / share |log OR| > 1)\n")
            ms = list(jobs[0]["control_log_or"])
            P("| job | " + " | ".join(ms) + " |")
            P("|" + "---|" * (1 + len(ms)))
            for j in jobs:
                P(f"| {j['name']} | " + " | ".join(
                    f"{j['control_log_or'][m]['median']:+.2f} / {j['control_log_or'][m]['share_abs_gt_1']:.2f}" for m in ms) + " |")
    P("\n### fitted parameters (mean over the 5 training folds)\n")
    from training.supervised_pairs import FORM_ORDER  # noqa: E402
    P("| dataset | job | param | " + " | ".join(FORM_ORDER) + " |")
    P("|" + "---|" * (3 + len(FORM_ORDER)))
    P("| external | 32B old test | bias | " + " | ".join(f"{d['external_params']['bias'][x]:+.2f}" for x in FORM_ORDER) + " |")
    P("| external | 32B old test | σ² | " + " | ".join(f"{d['external_params']['sigma2'][x]:.2f}" for x in FORM_ORDER) + " |")
    for j in d["jobs"]:
        pr = j["params"]
        P(f"| {j['dataset']} | {j['name']} | bias | " + " | ".join(f"{pr['bias_mean'][x]:+.2f}" for x in FORM_ORDER) + " |")
        P(f"| {j['dataset']} | {j['name']} | weight | " + " | ".join(f"{pr['weights_mean'][x]:.2f}" for x in FORM_ORDER) + " |")
    P("\n| dataset | job | γ by fold | λ fill (and3 / and4 / litand) | λ hybrid (and3 / and4 / litand) |")
    P("|---|---|---|---|---|")
    for j in d["jobs"]:
        pr = j["params"]
        lf = " / ".join(",".join(f"{x:g}" for x in pr["lambda_fill_by_fold"][k]) for k in ("and3", "and4", "litand"))
        lh = " / ".join(",".join(f"{x:g}" for x in pr["lambda_hybrid_by_fold"][k]) for k in ("and3", "and4", "litand")) \
            if pr["lambda_hybrid_by_fold"] else "—"
        P(f"| {j['dataset']} | {j['name']} | {','.join(f'{x:g}' for x in pr['gamma_by_fold'])} | {lf} | {lh} |")
    print("\n".join(out))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
