"""Chart JSON for docs/confidence-weighting-20260927.md from the analysis outputs (CPU only).

Inputs (in --dir): step1.json (scripts/confidence_weighting.py), step2-only.json (--step2 ... --step2-only),
structured JSON (scripts/confidence_weighting_structured.py).  Output: one row per (signal, arm) with the
Spearman error checks and the cross-fitted all-12 log loss, equal vs weighted, paired bootstrap 95% CI.

Usage: tables_confidence_weighting.py --dir /home/agent/data/confidence-weighting-20260927 \
         --structured structured2.json --out docs/confidence-weighting-20260927-charts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def arm_rows(block, source, kind):
    out = []
    arms = block["arms"]
    for name, a in arms.items():
        if not a.get("ref"):
            continue
        d, lo, hi = a["vs_ref"]["all12"]
        out.append({"source": source, "kind": kind, "pairs": block["pairs"], "arm": name, "ref": a["ref"],
                    "ll_ref": arms[a["ref"]]["all12_ll"], "ll_arm": a["all12_ll"],
                    "diff": d, "ci95": [lo, hi], "tau_by_fold": a.get("tau_by_fold")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--structured", default="structured2.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s1 = json.loads((a.dir / "step1.json").read_text())
    s2 = json.loads((a.dir / "step2-only.json").read_text())["step2"]
    st = json.loads((a.dir / a.structured).read_text())
    err = []
    for sub, sig in s1["error_checks"].items():
        for name, e in sig.items():
            err.append({"source": f"step1 {sub}", "signal": name, "fit_error": e["fit_error"],
                        "outcome_logloss": e["outcome_logloss"]})
    for name, e in s2["error_checks"].items():
        err.append({"source": "step2 subset_300", "signal": name, "fit_error": e["fit_error"],
                    "outcome_logloss": e["outcome_logloss"], "rho_with_extremeness": e.get("rho_with_extremeness")})
    proj = []
    for k, b in s1["projection"].items():
        sub, kind = k.split()
        proj += arm_rows(b, f"step1 {sub}", kind)
    for kind, b in s2["projection"].items():
        proj += arm_rows(b, "step2 subset_300", kind)
    struct = []
    for j in st["jobs"]:
        e = j["by"]["all"]
        struct.append({"job": j["name"], "pairs": j["pairs"], "recovery": e.get("recovery"),
                       "vs_eq": e["vs_eq"], "chosen_weights_by_fold": j["chosen_weights_by_fold"]})
    out = {"description": __doc__,
           "rows_units": "Spearman rho; log loss in nats per answer (all 12 forms), lower is better; diff = arm - ref "
                         "with paired bootstrap 95% CI over pairs (2,000 resamples)",
           "error_checks": err, "projection": proj,
           "raw_ll_step2": s2["raw_ll"], "raw_vs_lp": s2["raw_vs_lp"], "prob_shift": s2["prob_shift"],
           "structured": struct}
    a.out.write_text(json.dumps(out, indent=1))
    print(f"{len(err)} error rows, {len(proj)} projection rows, {len(struct)} structured jobs -> {a.out}")


if __name__ == "__main__":
    main()
