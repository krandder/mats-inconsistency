"""Confidence-weighted projection on the structured pairs (CPU only; step 3 of confidence weighting).

Does weighting the 12 answers by confidence keep the known relations (implication ladders,
exclusive options) better than equal weights?  Structured pairs: docs/structured-pairs-20260924.md;
methods and scoring follow scripts/weighted_projection.py (market hybrid = market marginals + the
projection's log odds ratio; recovery = share of the known relation's joint-LL gain).

Weights (w = 1 / (v + tau), tau cross-fitted on the other 4 of 5 folds by cluster; tau = inf is
equal weights):
  form   v = per-form mean sampling variance of Qwen3-32B's log-odds on the OLD 1,066-pair test
         (s0..s4; docs/confidence-weighting-20260927.md).  External, label-free, available for every run.
  digit  v = variance of the implied number under the probability-digit logprobs (32B Q only: a
         temperature-0 rerun of the same prompts with logprobs, scripts/run_confidence_signals.py
         --data structured --variants lp; its answers are the point for all 32B Q-lp arms).
  form x digit  the product of both.
tau is tuned two ways: by the model-only all-12 log loss ("_m") and by the hybrid's joint log loss ("_h").

Usage: confidence_weighting_structured.py --out /home/agent/data/confidence-weighting-20260927/structured.json
       [--lp-log /home/agent/data/confidence-weighting-20260927/step2-structured/log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.batch_projection import log_or, project  # noqa: E402
from scripts.confidence_weighting import TAUS, digit_signal, load_dec  # noqa: E402
from scripts.confidence_weighting import logit as clogit  # noqa: E402
from scripts.weighted_projection import (K, SEED, boot, dep_stats, forms, jl, ll, load_structured, prep,  # noqa: E402
                                         with_or)
from training.supervised_pairs import FORM_ORDER  # noqa: E402

FI = {f: i for i, f in enumerate(FORM_ORDER)}


def external_form_var():
    pids, Y, P, sub = load_dec()
    S5 = np.stack([clogit(P[f"s{i}"]) for i in range(5)], 0)
    return np.nanmean(S5.var(0, ddof=1), 0)


def folds(job):
    cls = sorted(set(job["cluster"]))
    rng = np.random.default_rng(SEED)
    f = dict(zip(cls, rng.permutation(len(cls)) % K))
    return np.array([f[c] for c in job["cluster"]])


def cands(Vf, Va):
    out = {"eq": None}
    for t in TAUS:
        if np.isinf(t):
            continue
        if Vf is not None:
            out[f"f{t}"] = 1 / (Vf + t)
        if Va is not None:
            out[f"a{t}"] = 1 / (Va + t)
    if Vf is not None and Va is not None:
        for t1 in TAUS:
            for t2 in TAUS:
                if np.isinf(t1) or np.isinf(t2):
                    continue
                out[f"f{t1}a{t2}"] = 1 / (Vf + t1) / (Va + t2)
    return out


def arm(job, T, W0, cand):
    """Cross-fitted: choose among cand on the training folds, by the model-only all-12 LL ('m') and by the
    hybrid's joint LL ('h').  Returns {pick: (proj preds, hybrid preds, log OR, chosen keys)}."""
    n = len(T)
    Y = job["Y"]
    ma, mb = job["ma"], job["mb"]
    cache = {}
    for k, w in cand.items():
        W = W0 if w is None else W0 * np.broadcast_to(w, T.shape)
        q = project(T, W, "kl")
        pr = forms(q)
        lt = log_or(q)
        hy = with_or(ma, mb, lt)
        cache[k] = (pr, hy, lt)
    fold = folds(job)
    res = {}
    for pick in ("m", "h"):
        pr = np.full((n, 12), np.nan); hy = np.full((n, 12), np.nan); lt = np.full(n, np.nan); chosen = []
        for f in range(K):
            tr, te = fold != f, fold == f
            if pick == "m":
                key = min(cache, key=lambda k: ll(cache[k][0][tr], Y[tr])[:, 1].mean())
            else:
                key = min(cache, key=lambda k: ll(cache[k][1][tr], Y[tr])[:, 0].mean())
            chosen.append(key)
            pr[te], hy[te], lt[te] = cache[key][0][te], cache[key][1][te], cache[key][2][te]
        res[pick] = (pr, hy, lt, chosen)
    return res


def lp_job(base, lp_log):
    """32B Q with the logprob rerun's answers as the point and its digit variance."""
    byp = defaultdict(dict)
    for r in jl(lp_log):
        a, v, _ = digit_signal(r.get("logprobs_retry") if r.get("retried") else r.get("logprobs"))
        byp[r["pair_id"]][r["form"]] = (r.get("p"), v)
    keep = [i for i, p in enumerate(base["pids"]) if len(byp.get(p, {})) == 12]
    job = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(base["pids"]) else v) for k, v in base.items()}
    job["pids"] = [base["pids"][i] for i in keep]
    job["pairs_meta"] = [base["pairs_meta"][i] for i in keep]
    job["S"] = np.array([[np.nan if byp[p][f][0] is None else byp[p][f][0] for f in FORM_ORDER] for p in job["pids"]], float)
    Va = np.array([[np.nan if byp[p][f][1] is None else byp[p][f][1] for f in FORM_ORDER] for p in job["pids"]], float)
    job["name"] = base["name"] + " (logprob rerun)"
    job.pop("T", None); job.pop("W0", None)
    prep(job)
    return job, np.nan_to_num(Va, nan=np.nanmedian(Va))


def score(job, preds, hyb):
    Y = job["Y"]
    known, mi = job["known"], None
    ma, mb = job["ma"], job["mb"]
    mi = forms(np.stack([ma * mb, ma - ma * mb, mb - ma * mb, 1 - ma - mb + ma * mb], 1))
    L = {"market_indep": ll(mi, Y), "market_known": ll(known, Y)}
    for k, v in preds.items():
        L[f"proj {k}"] = ll(v, Y)
    for k, v in hyb.items():
        L[f"hybrid {k}"] = ll(v, Y)
    res = {"name": job["name"], "pairs": len(Y), "by": {}}
    for s in ("all", "threshold", "date", "mc_excl", "control"):
        idx = np.arange(len(Y)) if s == "all" else np.where(job["strata"] == s)[0]
        cls = job["cluster"][idx]
        e = {"pairs": int(len(idx)), "ll_joint": {m: float(v[idx, 0].mean()) for m, v in L.items()},
             "ll_all12": {m: float(v[idx, 1].mean()) for m, v in L.items()}, "vs_eq": {}}
        gap = L["market_indep"][idx, 0].mean() - L["market_known"][idx, 0].mean()
        if s != "control" and gap > 1e-9:
            e["recovery"] = {m: float((L["market_indep"][idx, 0].mean() - v[idx, 0].mean()) / gap) for m, v in L.items()
                             if m.startswith("hybrid")}
        for m in L:
            for ref in ("proj eq", "hybrid eq"):
                if m != ref and m.split()[0] == ref.split()[0]:
                    b = boot(L[m][idx] - L[ref][idx], cls)
                    e["vs_eq"][m] = {"joint": b[0], "all12": b[1]}
        res["by"][s] = e
    res["dependence_captured"] = {k: dep_stats(job, v, list(range(len(Y)))) for k, v in preds.items()}
    res["dependence_captured"]["raw"] = dep_stats(job, job["S"], list(range(len(Y))))
    return res


def run(job, Vf, Va):
    T, W0 = job["T"], job["W0"]
    preds, hyb, chosen = {}, {}, {}
    preds["eq"] = forms(project(T, W0, "kl"))
    hyb["eq"] = with_or(job["ma"], job["mb"], log_or(project(T, W0, "kl")))
    specs = {"form": (Vf, None)}
    if Va is not None:
        specs.update({"digit": (None, Va), "form x digit": (Vf, Va)})
    for name, (vf, va) in specs.items():
        both = arm(job, T, W0, cands(vf, va))
        for pick, (pr, hy, lt, ch) in both.items():
            preds[f"{name} _{pick}"], hyb[f"{name} _{pick}"], chosen[f"{name} _{pick}"] = pr, hy, ch
    r = score(job, preds, hyb)
    r["chosen_weights_by_fold"] = chosen
    e = r["by"]["all"]
    print(job["name"], {k: round(v, 3) for k, v in e.get("recovery", {}).items()}, flush=True)
    for m, v in e["vs_eq"].items():
        print(f"   {m:28s} joint {v['joint'][0]:+.4f} [{v['joint'][1]:+.4f}, {v['joint'][2]:+.4f}]  "
              f"all12 {v['all12'][0]:+.4f} [{v['all12'][1]:+.4f}, {v['all12'][2]:+.4f}]", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lp-log", type=Path)
    a = ap.parse_args()
    Vf = external_form_var()
    print("external per-form sampling variance:", dict(zip(FORM_ORDER, Vf.round(3))), flush=True)
    out = {"description": __doc__, "external_form_var": dict(zip(FORM_ORDER, Vf.round(4).tolist())), "jobs": []}
    jobs = [prep(j) for j in load_structured()]
    for job in jobs:
        out["jobs"].append(run(job, Vf, None))
        a.out.write_text(json.dumps(out, indent=1))
    if a.lp_log and a.lp_log.exists():
        base = next(j for j in jobs if j["name"] == "32B Q")
        job, Va = lp_job(base, a.lp_log)
        out["jobs"].append(run(job, Vf, Va))
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
