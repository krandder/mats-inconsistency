"""Project per-form predictions onto the nearest coherent set (CPU only, no API calls).

For each pair, fit a distribution q over the 4 atoms (P&Q, P&~Q, ~P&Q, ~P&~Q) to the
stated probabilities of the 12 forms, then read all 12 forms from q. Three fits:
  ls_prob   least squares on probabilities
  ls_logit  least squares on log-odds
  kl        minimize sum_i KL(stated_i || implied_i), Bernoulli KL per form
Stated probabilities are clipped to [0.01, 0.99] first (the scorer's clip). Missing
forms are left out of the fit (the projection still fills them).

Scoring uses scripts/score_verbalized_baseline.score (clip 0.01, missing imputed at
0.5). Paired bootstrap over pairs (2,000 resamples) for the gain vs raw and the
difference vs a reference run (the 235B all-12-in-one-call thinking-off run).

Usage:
  coherence_projection.py --out docs/coherence-projection-20260924.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.score_verbalized_baseline import score  # noqa: E402

# atoms: a=P&Q, b=P&~Q, c=~P&Q, d=~P&~Q
FORM_ATOMS = {
    "P": (1, 1, 0, 0), "NOT P": (0, 0, 1, 1), "Q": (1, 0, 1, 0), "NOT Q": (0, 1, 0, 1),
    "(P) AND (Q)": (1, 0, 0, 0), "(P) OR (Q)": (1, 1, 1, 0),
    "(P) AND (NOT Q)": (0, 1, 0, 0), "(P) OR (NOT Q)": (1, 1, 0, 1),
    "(NOT P) AND (Q)": (0, 0, 1, 0), "(NOT P) OR (Q)": (1, 0, 1, 1),
    "(NOT P) AND (NOT Q)": (0, 0, 0, 1), "(NOT P) OR (NOT Q)": (0, 1, 1, 1),
}
LO = 0.01
EPS = 1e-6  # atom floor inside the optimizer (keeps log-odds finite)

P235 = "/home/agent/data/precutoff-baseline-20260923"
Q06 = "/home/agent/data/qwen06b-fixed-20260923/results"
MODELS = {
    "qwen3-235b per-form (thinking off)": (f"{P235}/per-form/predictions-test.jsonl", "p"),
    "qwen3-32b v6_direct per-form": ("/home/agent/data/qwen32b-baseline-20260923/remote-out/predictions-test-v6_direct.jsonl", "p"),
    **{f"0.6B {a}": (f"{Q06}/{a}/predictions-test-best.jsonl", "probability") for a in (
        "A8_full_consistency_l1", "A7_full_accuracy", "A4_one_consistency_l3", "A3_one_consistency_l1",
        "A1_one_accuracy")},
    # the reference itself, projected: does all-in-one keep an edge once both are coherent?
    "qwen3-235b all-12-in-one-call (thinking off)": (f"{P235}/qwen3-235b-a22b-nothink/predictions-test.jsonl", "p"),
}
REFERENCE = ("qwen3-235b all-12-in-one-call (thinking off)", f"{P235}/qwen3-235b-a22b-nothink/predictions-test.jsonl", "p")


def logit(x):
    return np.log(x / (1 - x))


def fit(forms: list[str], stated: np.ndarray, method: str) -> np.ndarray:
    A = np.array([FORM_ATOMS[f] for f in forms], dtype=float)
    s = np.clip(stated, LO, 1 - LO)

    def implied(q):
        return np.clip(A @ q, EPS, 1 - EPS)

    if method == "ls_prob":
        f = lambda q: float(np.sum((A @ q - s) ** 2))
    elif method == "ls_logit":
        f = lambda q: float(np.sum((logit(implied(q)) - logit(s)) ** 2))
    elif method == "kl":
        f = lambda q: float(np.sum(s * np.log(s / implied(q)) + (1 - s) * np.log((1 - s) / (1 - implied(q)))))
    else:
        raise ValueError(method)
    cons = ({"type": "eq", "fun": lambda q: q.sum() - 1, "jac": lambda q: np.ones(4)},)
    bounds = [(EPS, 1)] * 4
    # starts: uniform, and the independence product of the stated marginals
    pP = s[forms.index("P")] if "P" in forms else 0.5
    pQ = s[forms.index("Q")] if "Q" in forms else 0.5
    starts = [np.full(4, 0.25), np.array([pP * pQ, pP * (1 - pQ), (1 - pP) * pQ, (1 - pP) * (1 - pQ)])]
    best = None
    for x0 in starts:
        r = minimize(f, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-12, "maxiter": 500})
        q = np.clip(r.x, 0, None); q = q / q.sum()
        v = f(np.clip(q, EPS, None) / np.clip(q, EPS, None).sum())
        if best is None or v < best[0]:
            best = (v, q)
    return best[1]


def load(path: str, key: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        rows.append({"pair_id": r["pair_id"], "query_id": r["query_id"], "form": r["form"], "label": r["label"],
                     "observed": r["observed"], "p": r[key]})
    return rows


def project(rows: list[dict], method: str) -> list[dict]:
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)
    out = []
    for pid, rs in by_pair.items():
        have = [r for r in rs if r["p"] is not None]
        q = fit([r["form"] for r in have], np.array([r["p"] for r in have], dtype=float), method)
        for r in rs:
            out.append(dict(r, p=float(np.dot(FORM_ATOMS[r["form"]], q))))
    return out


def per_pair_loss(rows: list[dict]) -> dict:
    """pair -> (sum all-12 log loss, n all, sum observed log loss, n observed, sum brier, sum observed brier)."""
    acc = defaultdict(lambda: np.zeros(6))
    for r in rows:
        p = 0.5 if r["p"] is None else min(1 - LO, max(LO, r["p"]))
        y = r["label"]
        ll = -(y * math.log(p) + (1 - y) * math.log(1 - p)); br = (p - y) ** 2
        v = acc[r["pair_id"]]
        v[0] += ll; v[1] += 1; v[4] += br
        if r["observed"]:
            v[2] += ll; v[3] += 1; v[5] += br
    return acc


def summary(rows):
    a = score(rows, impute=True); o = score([r for r in rows if r["observed"]], impute=True)
    return {"all12_log_loss": a["log_loss"], "all12_brier": a["brier"], "observed_log_loss": o["log_loss"],
            "observed_brier": o["brier"]}


def boot(x: dict, y: dict, pairs: list[str], rng_seed: int = 20260924, n: int = 2000) -> dict:
    """Paired bootstrap of metric(x) - metric(y) over pairs. Metric = ratio of sums (matches pooled scoring)."""
    X = np.array([x[p] for p in pairs]); Y = np.array([y[p] for p in pairs])
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, len(pairs), size=(n, len(pairs)))
    res = {}
    for name, (num, den) in {"all12_log_loss": (0, 1), "observed_log_loss": (2, 3), "all12_brier": (4, 1),
                             "observed_brier": (5, 3)}.items():
        point = X[:, num].sum() / X[:, den].sum() - Y[:, num].sum() / Y[:, den].sum()
        bx = X[idx, num].sum(1) / X[idx, den].sum(1); by = Y[idx, num].sum(1) / Y[idx, den].sum(1)
        d = bx - by
        res[name] = {"diff": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ref_name, ref_path, ref_key = REFERENCE
    ref_rows = load(ref_path, ref_key)
    ref_pp = per_pair_loss(ref_rows)
    out = {"reference": {"name": ref_name, "path": ref_path, **summary(ref_rows)}, "models": {}}
    for name, (path, key) in MODELS.items():
        rows = load(path, key)
        raw_pp = per_pair_loss(rows)
        pairs = sorted(raw_pp)
        assert set(pairs) == set(ref_pp), name
        m = {"path": path, "raw": {**summary(rows), "vs_reference": boot(raw_pp, ref_pp, pairs)}}
        for method in ("ls_prob", "ls_logit", "kl"):
            proj = project(rows, method)
            pp = per_pair_loss(proj)
            m[method] = {**summary(proj), "gain_vs_raw": boot(pp, raw_pp, pairs), "vs_reference": boot(pp, ref_pp, pairs)}
            print(f"{name:40s} {method:9s} all12 {m[method]['all12_log_loss']:.4f} (raw {m['raw']['all12_log_loss']:.4f}) "
                  f"d {m[method]['gain_vs_raw']['all12_log_loss']['diff']:+.4f} {m[method]['gain_vs_raw']['all12_log_loss']['ci95']}",
                  flush=True)
        out["models"][name] = m
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
