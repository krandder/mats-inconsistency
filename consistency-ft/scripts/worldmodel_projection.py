"""Coherence projection over 2^n atoms for the world-model clusters (CPU only, no API calls).

For each cluster (scripts/worldmodel_clusters.py), fit a distribution q over the 2^n
atoms (n <= 8, so <= 256 atoms) to the stated probabilities of the queries asked, then
read every query from q (a conditional "B | A" is implied as P(A and B) / P(A)).
Two fits:
  kl        minimize sum_i KL(stated_i || implied_i), Bernoulli KL per query
  ls_logit  minimize sum_i (logit implied_i - logit stated_i)^2
q = softmax(z), L-BFGS with analytic gradients, two starts (uniform; independence
product of the stated singles), plus a 1e-4 * sum q log q tie-break (max-entropy
among equally good fits; unasked joint structure defaults toward independence).
Stated values are clipped to [0.01, 0.99]; missing answers are left out of the fit and
out of scoring.  The fit never sees labels.

Scoring: clip 0.01; log loss and Brier on every query with a known truth value
(conditionals only where the condition resolved YES).  Gains are projected minus raw
(negative = better), pooled over queries; 95% CIs from a paired bootstrap over
clusters (2,000 resamples, seed 20260924).

Query-count curve: fit on nested query sets (singles; + pairwise ANDs; + conditionals;
+ 3-way ANDs/ORs (n=2: the pairwise OR); + negations = all), plus a side branch
singles + negations (rephrasing only, no joint structure) and score (a) the singles (a fixed target) and
(b) every scoreable query (queries outside the fit are read from q, i.e. the fitted
world model's prediction replaces the raw answer).
Shuffled-structure control: the same fit with the formulas permuted at random within
the cluster (5 permutations); query i is then read as formula pi(i).

Usage: worldmodel_projection.py --data /home/agent/data/worldmodel-clusters-20260924 \
          --out-json docs/worldmodel-clusters-20260924.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

LO = 0.01
PEPS = 1e-9
LAM = 1e-4
STAGES = [("singles", {"single"}), ("+and2", {"single", "and2"}), ("+cond", {"single", "and2", "cond"}),
          ("+3way/or", {"single", "and2", "cond", "and3", "or3", "or"}),
          ("singles+neg", {"single", "neg"}),  # side branch: rephrasing only, no joint structure
          ("all", {"single", "and2", "cond", "and3", "or3", "or", "neg"})]
METHODS = ("kl", "ls_logit")
N_PERM = 5
SEED = 20260924


def atoms_truth(f, n: int) -> np.ndarray:
    idx = np.arange(2 ** n)
    op = f["op"]
    if op == "var":
        return ((idx >> f["i"]) & 1).astype(bool)
    if op == "not":
        return ~atoms_truth(f["a"], n)
    if op == "and":
        return np.logical_and.reduce([atoms_truth(a, n) for a in f["args"]])
    if op == "or":
        return np.logical_or.reduce([atoms_truth(a, n) for a in f["args"]])
    raise ValueError(op)


def vectors(f, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(numerator, denominator) indicator vectors: implied = a.q / b.q."""
    if f["op"] == "cond":
        g = atoms_truth(f["given"], n)
        return (g & atoms_truth(f["then"], n)).astype(float), g.astype(float)
    return atoms_truth(f, n).astype(float), np.ones(2 ** n)


def logit(x):
    return np.log(x / (1 - x))


def fit(n: int, formulas: list, stated: np.ndarray, method: str, singles: dict[int, float]) -> np.ndarray:
    K = 2 ** n
    A = np.array([vectors(f, n)[0] for f in formulas])
    B = np.array([vectors(f, n)[1] for f in formulas])
    s = np.clip(stated, LO, 1 - LO)
    ls = logit(s)

    def fg(z):
        z = z - z.max()
        q = np.exp(z); q /= q.sum()
        num, den = A @ q, np.maximum(B @ q, 1e-300)
        p = np.clip(num / den, PEPS, 1 - PEPS)
        if method == "kl":
            loss = float(np.sum(s * np.log(s / p) + (1 - s) * np.log((1 - s) / (1 - p))))
            dp = -s / p + (1 - s) / (1 - p)
        else:
            r = logit(p) - ls
            loss = float(np.sum(r ** 2))
            dp = 2 * r / (p * (1 - p))
        gq = ((A - p[:, None] * B) / den[:, None] * dp[:, None]).sum(0)
        lq = np.log(np.maximum(q, 1e-300))
        loss += LAM * float(q @ lq)
        gq = gq + LAM * (lq + 1)
        gz = q * (gq - q @ gq)
        return loss, gz

    starts = [np.zeros(K)]
    if singles:
        bits = (np.arange(K)[:, None] >> np.arange(n)[None, :]) & 1
        pm = np.array([np.clip(singles.get(i, 0.5), LO, 1 - LO) for i in range(n)])
        starts.append((bits * np.log(pm) + (1 - bits) * np.log(1 - pm)).sum(1))
    best = None
    for z0 in starts:
        r = minimize(fg, z0, jac=True, method="L-BFGS-B", options={"maxiter": 3000, "gtol": 1e-9, "ftol": 1e-13})
        if best is None or r.fun < best.fun:
            best = r
    z = best.x - best.x.max()
    q = np.exp(z)
    return q / q.sum()


def implied(q: np.ndarray, f, n: int) -> float:
    a, b = vectors(f, n)
    return float((a @ q) / max(b @ q, 1e-300))


def ll(p, y):
    p = min(1 - LO, max(LO, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def br(p, y):
    p = min(1 - LO, max(LO, p))
    return (p - y) ** 2


def load(data: Path):
    clusters = {c["cluster_id"]: c for c in map(json.loads, (data / "clusters.jsonl").read_text().splitlines())}
    queries = [json.loads(l) for l in (data / "queries.jsonl").read_text().splitlines()]
    log = {}
    for l in (data / "log.jsonl").read_text().splitlines():
        r = json.loads(l); log[r["query_id"]] = r
    by = defaultdict(list)
    for q in queries:
        r = log.get(q["query_id"])
        q["p"] = None if r is None else r["p"]
        q["asked"] = r is not None
        by[q["cluster_id"]].append(q)
    return clusters, by, log


def analyse_cluster(c: dict, qs: list[dict], rng: np.random.Generator) -> dict:
    """Per-query records: raw and projected predictions for every fit variant."""
    n = c["n"]
    present = [q for q in qs if q["p"] is not None]
    singles = {q["formula"]["i"]: q["p"] for q in present if q["kind"] == "single"}
    out = {q["query_id"]: {"kind": q["kind"], "label": q["label"], "raw": q["p"], "formula": q["formula_str"]}
           for q in qs}
    for method in METHODS:
        for stage, kinds in STAGES:
            use = [q for q in present if q["kind"] in kinds]
            if not use:
                continue
            qd = fit(n, [q["formula"] for q in use], np.array([q["p"] for q in use]), method, singles)
            for q in qs:
                out[q["query_id"]][f"{method}|{stage}"] = implied(qd, q["formula"], n)
                out[q["query_id"]][f"{method}|{stage}|in_fit"] = q in use
        # shuffled-structure control (all queries)
        if len(present) > 1:
            forms = [q["formula"] for q in present]
            acc = defaultdict(list)
            for _ in range(N_PERM):
                perm = rng.permutation(len(present))
                pf = [forms[j] for j in perm]
                qd = fit(n, pf, np.array([q["p"] for q in present]), method, singles)
                for k, q in enumerate(present):
                    acc[q["query_id"]].append(implied(qd, pf[k], n))
            for qid, v in acc.items():
                out[qid][f"{method}|shuffled"] = v  # list over permutations
    return out


def bootstrap(per_cluster: list[tuple[float, float, int]], rng, reps=2000):
    """per_cluster: (sum raw loss, sum proj loss, count).  Returns pooled raw, proj, gain, CI."""
    arr = np.array(per_cluster, dtype=float)
    arr = arr[arr[:, 2] > 0]
    if not len(arr):
        return None
    raw, proj, cnt = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
    idx = rng.integers(0, len(arr), size=(reps, len(arr)))
    g = (arr[idx, 1].sum(1) - arr[idx, 0].sum(1)) / arr[idx, 2].sum(1)
    return {"clusters": int(len(arr)), "queries": int(cnt), "raw": raw / cnt, "proj": proj / cnt,
            "gain": (proj - raw) / cnt, "ci": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}


def summarize(clusters, recs, rng, key: str, select, metric=ll, shuffled=False):
    """Pooled raw vs `key` over queries passing `select`, grouped per cluster."""
    per = []
    for cid, qr in recs.items():
        s_raw = s_proj = cnt = 0.0
        for qid, r in qr.items():
            if r["raw"] is None or r["label"] is None or not select(r) or key not in r:
                continue
            p = r[key]
            pv = float(np.mean([metric(x, r["label"]) for x in p])) if shuffled else metric(p, r["label"])
            s_raw += metric(r["raw"], r["label"]); s_proj += pv; cnt += 1
        per.append((s_raw, s_proj, cnt))
    return bootstrap(per, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/agent/data/worldmodel-clusters-20260924"))
    ap.add_argument("--out-json", type=Path, required=True)
    a = ap.parse_args()
    clusters, by, log = load(a.data)
    rng = np.random.default_rng(SEED)
    recs = {}
    for cid in sorted(by):
        recs[cid] = analyse_cluster(clusters[cid], by[cid], rng)
    with (a.data / "projections.jsonl").open("w") as f:
        for cid, qr in recs.items():
            for qid, r in qr.items():
                f.write(json.dumps({"cluster_id": cid, "query_id": qid, **r}) + "\n")

    groups = [("all", lambda c: True)] + [(f"{t}", lambda c, t=t: c["type"] == t) for t in ("mc", "group")]
    sizes = sorted({c["n"] for c in clusters.values()})
    full = STAGES[-1][0]
    res = {"model": "qwen3-32b (DashScope, thinking off, v6_direct-style one query per call)",
           "calls": {"asked": sum(1 for qs in by.values() for q in qs if q["asked"]),
                     "missing": sum(1 for qs in by.values() for q in qs if q["asked"] and q["p"] is None),
                     "not_asked": sum(1 for qs in by.values() for q in qs if not q["asked"]),
                     "cost_usd_cap_price": sum(r["cost"] for r in log.values()),
                     "prompt_tokens": sum((r["usage"] or {}).get("prompt_tokens", 0) + ((r.get("usage_retry") or {}).get("prompt_tokens", 0)) for r in log.values()),
                     "completion_tokens": sum((r["usage"] or {}).get("completion_tokens", 0) + ((r.get("usage_retry") or {}).get("completion_tokens", 0)) for r in log.values())},
           "main": [], "curve": [], "by_kind": [], "control": []}
    for gname, gsel in groups:
        for n in sizes:
            sub = {cid: r for cid, r in recs.items() if clusters[cid]["n"] == n and gsel(clusters[cid])}
            qpc = np.mean([len(by[cid]) for cid in sub])
            for method in METHODS:
                key = f"{method}|{full}"
                for mname, metric in (("log_loss", ll), ("brier", br)):
                    s = summarize(clusters, sub, rng, key, lambda r: True, metric)
                    res["main"].append({"type": gname, "n": n, "method": method, "metric": mname,
                                        "queries_per_cluster": float(qpc), **(s or {})})
                    sh = summarize(clusters, sub, rng, f"{method}|shuffled", lambda r: True, metric, shuffled=True)
                    res["control"].append({"type": gname, "n": n, "method": method, "metric": mname, **(sh or {})})
                for stage, kinds in STAGES:
                    key = f"{method}|{stage}"
                    nq = float(np.mean([sum(1 for q in by[cid] if q["kind"] in kinds) for cid in sub]))
                    for target, sel in (("singles", lambda r: r["kind"] == "single"), ("all_queries", lambda r: True),
                                        ("fitted_queries", lambda r, key=key: r.get(key + "|in_fit", False))):
                        s = summarize(clusters, sub, rng, key, sel, ll)
                        res["curve"].append({"type": gname, "n": n, "method": method, "stage": stage,
                                             "queries_in_fit_per_cluster": nq, "target": target, **(s or {})})
                for kind in sorted({r["kind"] for qr in sub.values() for r in qr.values()}):
                    s = summarize(clusters, sub, rng, f"{method}|{full}", lambda r, k=kind: r["kind"] == k, ll)
                    if s:
                        res["by_kind"].append({"type": gname, "n": n, "method": method, "kind": kind, **s})
    a.out_json.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["calls"]))
    for r in res["main"]:
        if r["metric"] == "log_loss" and "raw" in r:
            print(f"{r['type']:5} n={r['n']} {r['method']:8} q/cl={r['queries_per_cluster']:.0f} raw {r['raw']:.4f} "
                  f"proj {r['proj']:.4f} gain {r['gain']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")


if __name__ == "__main__":
    main()
