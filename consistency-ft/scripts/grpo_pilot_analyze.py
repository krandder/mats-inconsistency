"""Pre-registered analysis of the GRPO consistency pilot (docs/grpo-consistency-pilot-20261008.md).

Input: an eval directory with one subdirectory per model tag (base, arm-acc, arm-acc_cons, arm-cons), each with
test.jsonl (all 12 forms of the 200 test pairs) and singles-external.jsonl (449 questions of 2026), as written
by training/grpo_pilot_eval.py.

Primary: single-forecast log loss on the P and Q rows of test.jsonl plus singles-external (849 rows); paired
bootstrap clustered by question (parent market for Manifold options), 10,000 resamples, seed 20261008.
Secondary on the test pairs: negation correlation, AND/OR bound violations, mean incoherence, joint (AND) and
all-12 log loss, kl projection gain; bootstrap clustered by pair.

Usage: grpo_pilot_analyze.py --eval DIR --out docs/grpo-consistency-pilot-20261008.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coherence_projection import FORM_ATOMS, fit  # noqa: E402

LO = 0.01
SEED = 20261008
B = 10_000
TAGS = ["base", "arm-acc", "arm-acc_cons", "arm-cons"]
CONTRASTS = [("arm-acc_cons", "arm-acc"), ("arm-acc_cons", "base"), ("arm-acc", "base"), ("arm-cons", "base"),
             ("arm-cons", "arm-acc")]
ANDS = {"(P) AND (Q)": ("P", "Q"), "(P) AND (NOT Q)": ("P", "NOT Q"), "(NOT P) AND (Q)": ("NOT P", "Q"),
        "(NOT P) AND (NOT Q)": ("NOT P", "NOT Q")}
ORS = {"(P) OR (Q)": ("P", "Q"), "(P) OR (NOT Q)": ("P", "NOT Q"), "(NOT P) OR (Q)": ("NOT P", "Q"),
       "(NOT P) OR (NOT Q)": ("NOT P", "NOT Q")}


def clip(p):
    return 0.5 if p is None else min(1 - LO, max(LO, p))


def ll(p, y):
    p = clip(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def load(d: Path, tag: str, name: str) -> list[dict]:
    return [json.loads(l) for l in (d / tag / f"{name}.jsonl").read_text().splitlines() if l.strip()]


def cluster_boot(values: dict[str, np.ndarray], clusters: list[str], rng_idx) -> dict:
    """values: tag -> per-row metric; clusters: per-row cluster id. Returns point + CI for every contrast."""
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    cl = np.array([pos[c] for c in clusters])
    sums = {t: np.bincount(cl, weights=v, minlength=len(ids)) for t, v in values.items()}
    counts = np.bincount(cl, minlength=len(ids)).astype(float)
    idx = rng_idx(len(ids))
    den = counts[idx].sum(1)
    out = {"n_rows": len(clusters), "n_clusters": len(ids), "mean": {t: float(v.mean()) for t, v in values.items()},
           "contrasts": {}}
    for a, b in CONTRASTS:
        if a in values and b in values:
            point = float(values[a].mean() - values[b].mean())
            d = (sums[a][idx].sum(1) - sums[b][idx].sum(1)) / den
            lo, hi = np.percentile(d, [2.5, 97.5])
            out["contrasts"][f"{a} - {b}"] = {"diff": point, "ci95": [float(lo), float(hi)],
                                              "reading": "lower" if hi < 0 else "higher" if lo > 0 else "no detectable difference"}
    return out


def project(fm: dict[str, float]) -> dict[str, float]:
    forms = list(fm)
    q = fit(forms, np.array([fm[f] for f in forms]), "kl")
    return {f: float(np.dot(FORM_ATOMS[f], q)) for f in FORM_ATOMS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tags = [t for t in TAGS if (args.eval / t / "DONE").exists()]
    rng = np.random.default_rng(SEED)
    cache = {}

    def rng_idx(n):
        if n not in cache:
            cache[n] = rng.integers(0, n, size=(B, n))
        return cache[n]

    report = {"tags": tags, "seed": SEED, "resamples": B}
    # ---- primary: singles
    singles = {}
    for t in tags:
        rows = [r for r in load(args.eval, t, "test") if r["form"] in ("P", "Q")]
        rows += load(args.eval, t, "singles-external")
        singles[t] = rows
    ref = singles[tags[0]]
    key = [(r.get("pair_id"), r["form"], r.get("qid")) for r in ref]
    for t in tags:
        assert [(r.get("pair_id"), r["form"], r.get("qid")) for r in singles[t]] == key, t
    clusters = [r["cluster"] if r.get("cluster") else (r["p_parent"] if r["form"] == "P" else r["q_parent"]) for r in ref]
    report["parse_failures"] = {t: {"singles": sum(r["p"] is None for r in singles[t])} for t in tags}
    report["primary_single_log_loss"] = cluster_boot(
        {t: np.array([ll(r["p"], r["label"]) for r in singles[t]]) for t in tags}, clusters, rng_idx)
    report["single_brier"] = cluster_boot(
        {t: np.array([(clip(r["p"]) - r["label"]) ** 2 for r in singles[t]]) for t in tags}, clusters, rng_idx)
    by_source = {}
    for src in sorted({r["source"] for r in ref}):
        sel = [i for i, r in enumerate(ref) if r["source"] == src]
        by_source[src] = cluster_boot({t: np.array([ll(singles[t][i]["p"], singles[t][i]["label"]) for i in sel])
                                       for t in tags}, [clusters[i] for i in sel], rng_idx)
    report["single_log_loss_by_source"] = by_source
    # ---- secondary: the 12 forms of the test pairs
    sec = defaultdict(dict)
    pair_rows = {t: load(args.eval, t, "test") for t in tags}
    for t in tags:
        report["parse_failures"][t]["test_all12"] = sum(r["p"] is None for r in pair_rows[t])
    pairs = sorted({r["pair_id"] for r in pair_rows[tags[0]]})
    metrics = {}
    for t in tags:
        bp = defaultdict(dict)
        lab = defaultdict(dict)
        for r in pair_rows[t]:
            bp[r["pair_id"]][r["form"]] = r["p"]
            lab[r["pair_id"]][r["form"]] = r["label"]
        m = defaultdict(list)
        neg_a, neg_b = [], []
        for pid in pairs:
            fm, y = bp[pid], lab[pid]
            c = {f: clip(p) for f, p in fm.items()}
            raw12 = np.mean([ll(fm[f], y[f]) for f in fm])
            have = {f: clip(p) for f, p in fm.items() if p is not None}
            proj = project(have) if len(have) >= 2 else c
            m["all12_log_loss"].append(raw12)
            m["projected_all12_log_loss"].append(np.mean([ll(proj[f], y[f]) for f in fm]))
            m["projection_gain"].append(m["projected_all12_log_loss"][-1] - raw12)
            m["joint_and_log_loss"].append(np.mean([ll(fm[f], y[f]) for f in ANDS]))
            m["and_violation_share"].append(np.mean([c[f] > min(c[a], c[b]) + 0.1 for f, (a, b) in ANDS.items()]))
            m["or_violation_share"].append(np.mean([c[f] < max(c[a], c[b]) - 0.1 for f, (a, b) in ORS.items()]))
            s = np.array([have[f] for f in have])
            A = np.array([FORM_ATOMS[f] for f in have], dtype=float)
            q = fit(list(have), s, "kl") if len(have) >= 2 else None
            m["incoherence"].append(float(np.sum(s * np.log(s / np.clip(A @ q, 1e-6, 1)) +
                                                (1 - s) * np.log((1 - s) / np.clip(1 - A @ q, 1e-6, 1))))
                                    if q is not None else 0.0)
            for a, na in (("P", "NOT P"), ("Q", "NOT Q")):
                neg_a.append(c[a]); neg_b.append(c[na])
        metrics[t] = {k: np.array(v) for k, v in m.items()}
        sec["negation_corr"][t] = float(np.corrcoef(neg_a, neg_b)[0, 1])
    for name in metrics[tags[0]]:
        sec[name] = cluster_boot({t: metrics[t][name] for t in tags}, pairs, rng_idx)
    report["secondary_test_pairs"] = sec
    args.out.write_text(json.dumps(report, indent=1))
    p = report["primary_single_log_loss"]
    print("primary single log loss:", json.dumps(p["mean"]))
    for k, v in p["contrasts"].items():
        print(f"  {k:28s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] {v['reading']}")


if __name__ == "__main__":
    main()
