"""Eliciting projection weights on the structured pairs (CPU only).

Point: Qwen3-32B question-only T=0 logprob rerun (confidence-weighting step2-structured).  Signals from
scripts/run_eliciting_weights.py --data structured (critique trust, critique revision distance, predicted revision).
Scoring as scripts/confidence_weighting_structured.py (recovery of the known relation's joint-LL gain by the hybrid
= market marginals + projection log odds ratio; clustered bootstrap CIs; tau cross-fitted by cluster, picked by the
model's all-12 log loss "_m" or by the hybrid's joint log loss "_h").

Usage: eliciting_weights_structured.py --log DIR/log.jsonl --out FILE.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.batch_projection import log_or, project  # noqa: E402
from scripts.confidence_weighting import logit  # noqa: E402
from scripts.confidence_weighting_structured import arm, cands, external_form_var, lp_job, score  # noqa: E402
from scripts.eliciting_weights import load_signals  # noqa: E402
from scripts.weighted_projection import forms, load_structured, prep, with_or  # noqa: E402

LP = Path("/home/agent/data/confidence-weighting-20260927/step2-structured/log.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--think-subset", action="store_true", help="only pairs with all 12 thinking answers")
    a = ap.parse_args()
    Vf = external_form_var()
    base = next(j for j in (prep(j) for j in load_structured()) if j["name"] == "32B Q")
    job, Vdig = lp_job(base, LP)
    M = load_signals(a.log, job["pids"], LP)
    if a.think_subset:
        keep = [i for i in range(len(job["pids"])) if (~np.isnan(M["p_think"][i])).sum() == 12]
        n0 = len(job["pids"])
        job = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n0 else v) for k, v in job.items()}
        job["pids"] = [job["pids"][i] for i in keep]
        job["pairs_meta"] = [job["pairs_meta"][i] for i in keep]
        Vdig = Vdig[keep]
        M = {k: v[keep] for k, v in M.items()}
        job["name"] += f" (think subset, {len(keep)} pairs)"
    T = job["T"]
    p = 1 / (1 + np.exp(-T))
    specs = {"form": (Vf, None), "digit": (None, Vdig)}
    fillm = lambda V: np.where(np.isnan(V), np.nanmedian(V), V)  # noqa: E731
    if (~np.isnan(M["trust"])).mean() > 0.5:
        specs["critique trust"] = (None, fillm(5 - M["trust"]))
        specs["form x critique trust"] = (Vf, fillm(5 - M["trust"]))
        specs["critique revision dist"] = (None, fillm((logit(M["p_revised"]) - T) ** 2))
    if (~np.isnan(M["change"])).mean() > 0.5:
        specs["predicted change"] = (None, fillm(M["change"] ** 2))
        specs["form x predicted change"] = (Vf, fillm(M["change"] ** 2))
    if (~np.isnan(M["p_think"])).mean() > 0.5:
        specs["measured revision (think)"] = (None, fillm((logit(M["p_think"]) - T) ** 2))
        specs["form x measured revision"] = (Vf, fillm((logit(M["p_think"]) - T) ** 2))
    cov = {k: float((~np.isnan(v)).mean()) for k, v in M.items()}
    print("coverage", cov, flush=True)
    W0 = job["W0"]
    preds = {"eq": forms(project(T, W0, "kl"))}
    hyb = {"eq": with_or(job["ma"], job["mb"], log_or(project(T, W0, "kl")))}
    chosen = {}
    for name, (vf, va) in specs.items():
        for pick, (pr, hy, lt, ch) in arm(job, T, W0, cands(vf, va)).items():
            preds[f"{name} _{pick}"], hyb[f"{name} _{pick}"], chosen[f"{name} _{pick}"] = pr, hy, ch
    if cov["p_revised"] > 0.5:
        Tr = logit(np.where(np.isnan(M["p_revised"]), p, M["p_revised"]))
        q = project(Tr, W0, "kl")
        preds["point: critique revised"] = forms(q)
        hyb["point: critique revised"] = with_or(job["ma"], job["mb"], log_or(q))
    pts = {}
    if cov["p_think"] > 0.5:
        Tt = logit(np.where(np.isnan(M["p_think"]), p, M["p_think"]))
        pts["point: think"] = Tt
        pts["point: mean(quick, think)"] = (T + Tt) / 2
    if cov["p_revised"] > 0.5:
        pts["point: mean(quick, critique revised)"] = (T + logit(np.where(np.isnan(M["p_revised"]), p, M["p_revised"]))) / 2
    for name, Tp in pts.items():
        q = project(Tp, W0, "kl")
        preds[name] = forms(q)
        hyb[name] = with_or(job["ma"], job["mb"], log_or(q))
    r = score(job, preds, hyb)
    r["chosen_weights_by_fold"] = chosen
    r["coverage"] = cov
    e = r["by"]["all"]
    print({k: round(v, 3) for k, v in e.get("recovery", {}).items()}, flush=True)
    for m, v in e["vs_eq"].items():
        print(f"   {m:34s} joint {v['joint'][0]:+.4f} [{v['joint'][1]:+.4f}, {v['joint'][2]:+.4f}]  "
              f"all12 {v['all12'][0]:+.4f} [{v['all12'][1]:+.4f}, {v['all12'][2]:+.4f}]", flush=True)
    a.out.write_text(json.dumps({"description": __doc__, "job": r}, indent=1, default=float))


if __name__ == "__main__":
    main()
