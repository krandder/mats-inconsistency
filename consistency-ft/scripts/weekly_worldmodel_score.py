"""Score the weekly world-model fits (scripts/weekly_worldmodel_fit.py) and write the summary + chart JSON.

Rows: one per asked query (week, kind, events, held-out flag, outcome label, stated answer, fitted or predicted
value of M1/M2/M3; BP beliefs, with the Gibbs-sampled values of the same fitted models as a check).

(a) held-out answer prediction: mean KL(stated || predicted) on the held-out "A and B" / "A and B and C"
    answers; M0 = product of the stated single answers (no fit) as reference.
(b) accuracy vs real outcomes: log loss and Brier of the single events at each week (raw stated vs fitted
    marginals) and of every queried joint event (fitted for training answers, predicted for held-out ones).
(c) per Luna relation type of the link ("A and B" queries).
Paired bootstrap (2,000 draws) over events: each draw resamples the events with replacement and weights each
query by the mean multiplicity of its events (a single event's rows over all weeks move together); a
week-block bootstrap is reported for the main differences as well.
Trajectories: single-event log loss and |p - y| by weeks left until the event's end, and the share of
week-to-week moves that go toward the outcome.

Usage: weekly_worldmodel_score.py [--data DIR]   ->  <data>/summary.json, <data>/charts.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA = Path("/home/agent/data/weekly-worldmodel-20260927")
LO = 0.01
B = 2000
SEED = 20260927
WEEK_MS = 7 * 86_400_000
MODELS = ["M1", "M2", "M3", "M2b", "M3b"]
BUCKETS = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]


def clip(p):
    return np.clip(np.asarray(p, dtype=float), LO, 1 - LO)


def logloss(p, y):
    p = clip(p); return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def kl(s, p):
    s, p = clip(s), clip(p)
    return s * np.log(s / p) + (1 - s) * np.log((1 - s) / (1 - p))


def load(d: Path, fits_dir="fits", lam_file=None):
    summ = json.loads((d / "build-summary.json").read_text())
    log = {}
    for l in (d / "log.jsonl").read_text().splitlines():
        r = json.loads(l); log[r["query_id"]] = r
    lam = json.loads((Path(lam_file) if lam_file else d / "lambda.json").read_text())["chosen"]
    keys = {"M1": "M1@0.0"} | {m: f"{m}@{lam[m]}" for m in MODELS[1:]}
    rows, weeks, fits = [], {}, {}
    for w in summ["weeks"]:
        ev = [json.loads(l) for l in (d / "weeks" / w / "events.jsonl").read_text().splitlines()]
        g = json.loads((d / "weeks" / w / "graph.json").read_text())
        weeks[w] = {"events": ev, "graph": g}
        fits[w] = json.loads((d / fits_dir / f"{w}.json").read_text())
    for l in (d / "queries.jsonl").read_text().splitlines():
        q = json.loads(l)
        w = q["week"]
        if w not in weeks:
            continue
        r = log.get(q["query_id"])
        if r is None or r["p"] is None:
            continue
        ev = weeks[w]["events"]
        pr = fits[w]["preds"]
        row = {"qid": q["query_id"], "week": w, "kind": q["kind"], "held": q["heldout"], "y": q["label"], "raw": r["p"],
               "eids": [ev[i]["id"] for i in q["events"]], "idx": q["events"]}
        for m, k in keys.items():
            row[m] = pr[k][q["query_id"]]
            row[m + "g"] = pr[k + ":gibbs"][q["query_id"]]
        if q["kind"] == "and2":
            i, j = sorted(q["events"])
            row["rel"] = weeks[w]["graph"]["rel"][f"{i}-{j}"]["type"]
        if q["kind"] == "single":
            e = ev[q["events"][0]]
            row["weeks_left"] = (e["end_time"] - _ms(w)) / WEEK_MS
        rows.append(row)
    # M0: product of the stated singles of the same week
    single = {(r["week"], r["idx"][0]): r["raw"] for r in rows if r["kind"] == "single"}
    for r in rows:
        ps = [single.get((r["week"], i)) for i in r["idx"]]
        r["M0"] = float(np.prod(clip(ps))) if all(p is not None for p in ps) else None
    return summ, rows, fits, keys, lam


def _ms(week):
    import datetime as dt
    return int(dt.datetime.fromisoformat(week).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


class Boot:
    """Event-cluster bootstrap weights for a list of rows (same draws for every statistic)."""

    def __init__(self, all_rows, seed=SEED):
        ids = sorted({e for r in all_rows for e in r["eids"]})
        self.eix = {e: k for k, e in enumerate(ids)}
        rng = np.random.default_rng(seed)
        self.C = rng.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=B).astype(float)  # (B, n_ev)
        self.C = np.concatenate([self.C, np.zeros((B, 1))], 1)
        wk = sorted({r["week"] for r in all_rows})
        self.wix = {w: k for k, w in enumerate(wk)}
        self.W = rng.multinomial(len(wk), np.ones(len(wk)) / len(wk), size=B).astype(float)

    def weights(self, rows):
        pad = len(self.eix)
        idx = np.array([[self.eix[e] for e in r["eids"]] + [pad] * (3 - len(r["eids"])) for r in rows])
        k = np.array([len(r["eids"]) for r in rows], dtype=float)
        return self.C[:, idx].sum(2) / k  # (B, n_rows)

    def week_weights(self, rows):
        return self.W[:, [self.wix[r["week"]] for r in rows]]


def stat_ci(loss, Wt):
    """loss (n,), Wt (B, n) -> mean, 95% CI of the weighted mean."""
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = (Wt @ loss) / Wt.sum(1)
    return {"mean": float(loss.mean()), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5))}


def compare(rows, boot, fn, cols, pairs, week_block=False):
    """fn(row_values, rows) -> per-row loss.  Returns per-column means with CIs and paired differences."""
    if not rows:
        return {"n": 0}
    Wt = boot.weights(rows)
    out = {"n": len(rows), "events": len({e for r in rows for e in r["eids"]})}
    L = {}
    for c in cols:
        vals = np.array([r[c] for r in rows], dtype=float)
        L[c] = fn(vals, rows)
        out[c] = stat_ci(L[c], Wt)
    for a, b in pairs:
        d = L[a] - L[b]
        out[f"{a}-{b}"] = stat_ci(d, Wt)
        if week_block:
            out[f"{a}-{b}"]["week_block"] = stat_ci(d, boot.week_weights(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--fits-dir", default="fits")
    ap.add_argument("--lambda-file")
    ap.add_argument("--tag", default="", help="suffix for summary/charts file names")
    a = ap.parse_args()
    d = a.data
    summ, rows, fits, keys, lam = load(d, a.fits_dir, a.lambda_file)
    boot = Boot(rows)
    ys = lambda rs: np.array([r["y"] for r in rs], dtype=float)
    LL = lambda v, rs: logloss(v, ys(rs))
    BR = lambda v, rs: (clip(v) - ys(rs)) ** 2
    KL = lambda v, rs: kl(np.array([r["raw"] for r in rs]), v)
    models = MODELS
    mpairs = [("M2", "M1"), ("M3", "M2"), ("M3", "M1"), ("M2b", "M2"), ("M3b", "M3"), ("M3b", "M2b")]
    res = {"weeks": summ["weeks"], "lambda": lam, "fit_keys": keys}

    held = [r for r in rows if r["held"]]
    res["heldout_kl"] = {k: compare([r for r in held if (k == "all" or r["kind"] == k)], boot, KL,
                                    ["M0"] + models + [m + "g" for m in models], [("M1", "M0")] + mpairs, week_block=True)
                         for k in ("all", "and2", "and3")}
    raw_pairs = [(m, "raw") for m in models] + mpairs + [(m, "M0") for m in models] + [("M0", "raw")]
    sing = [r for r in rows if r["kind"] == "single"]
    linked = {(w, i) for w in fits for c in [None] for i in
              {v for e in json.loads((d / "weeks" / w / "graph.json").read_text())["links"] for v in e}}
    sing_l = [r for r in sing if (r["week"], r["idx"][0]) in linked]
    joint_tr = [r for r in rows if r["kind"] != "single" and not r["held"]]
    joint_ho = [r for r in rows if r["kind"] != "single" and r["held"]]
    joint = joint_tr + joint_ho
    cols = ["raw", "M0"] + models + [m + "g" for m in models]
    res["outcome_logloss"] = {name: compare(rs, boot, LL, cols, raw_pairs, week_block=True) for name, rs in
                              (("single", sing), ("single_linked", sing_l), ("joint_all", joint),
                               ("joint_train", joint_tr), ("joint_heldout", joint_ho),
                               ("and2", [r for r in joint if r["kind"] == "and2"]),
                               ("and3", [r for r in joint if r["kind"] == "and3"]))}
    res["outcome_brier"] = {name: compare(rs, boot, BR, ["raw", "M0"] + models, raw_pairs) for name, rs in
                            (("single", sing), ("single_linked", sing_l), ("joint_all", joint), ("joint_heldout", joint_ho))}
    # (c) relation types (and2)
    rels = sorted({r["rel"] for r in rows if r["kind"] == "and2"})
    res["by_relation"] = {}
    for t in rels:
        rr = [r for r in rows if r["kind"] == "and2" and r["rel"] == t]
        res["by_relation"][t] = {
            "heldout_kl": compare([r for r in rr if r["held"]], boot, KL, ["M0"] + models, [("M1", "M0")] + mpairs),
            "outcome_logloss_all": compare(rr, boot, LL, ["raw", "M0"] + models, raw_pairs),
            "outcome_logloss_heldout": compare([r for r in rr if r["held"]], boot, LL, ["raw", "M0"] + models, raw_pairs),
            "yes_rate": float(np.mean(ys(rr))), "mean_stated": float(np.mean([r["raw"] for r in rr])),
            "mean_M0": float(np.mean([r["M0"] for r in rr]))}
    # AND bias: how the stated joints sit against the stated singles, and where the fits move the singles
    ab = {}
    single = {(r["week"], r["idx"][0]): r["raw"] for r in rows if r["kind"] == "single"}
    for k in ("and2", "and3"):
        rr = [r for r in rows if r["kind"] == k and r["M0"] is not None]
        mins = np.array([min(single[(r["week"], i)] for i in r["idx"]) for r in rr])
        st, m0 = np.array([r["raw"] for r in rr]), np.array([r["M0"] for r in rr])
        ab[k] = {"n": len(rr), "mean_stated": float(st.mean()), "mean_M0": float(m0.mean()), "yes_rate": float(ys(rr).mean()),
                 "mean_stated_minus_M0": float((st - m0).mean()), "share_stated_above_min_single": float((st > mins + 1e-9).mean()),
                 "mean_excess_over_min_single": float(np.clip(st - mins, 0, None).mean())}
    sy = ys(sing)
    ab["singles"] = {"n": len(sing), "yes_rate": float(sy.mean()),
                     **{c: {"mean": float(np.mean([r[c] for r in sing])),
                            "mean_shift_vs_raw": float(np.mean([r[c] - r["raw"] for r in sing])),
                            "mean_abs_shift_vs_raw": float(np.mean([abs(r[c] - r["raw"]) for r in sing]))} for c in ["raw"] + models}}
    res["and_bias"] = ab
    # trajectories
    traj = {}
    for lo, hi in BUCKETS:
        rr = [r for r in sing if lo <= r["weeks_left"] < hi]
        if not rr:
            continue
        name = f"{lo}-{hi if hi < 1e8 else 'inf'}"
        y = ys(rr)
        traj[name] = {"n": len(rr), "yes_rate": float(y.mean()),
                      **{c: {"logloss": float(logloss(np.array([r[c] for r in rr]), y).mean()),
                             "abs_err": float(np.abs(clip([r[c] for r in rr]) - y).mean())} for c in ["raw"] + models}}
    by_ev = defaultdict(list)
    for r in sing:
        by_ev[r["eids"][0]].append(r)
    moves = {c: [] for c in ["raw"] + models}
    for e, rr in by_ev.items():
        rr.sort(key=lambda r: r["week"])
        for r0, r1 in zip(rr, rr[1:]):
            s = 2 * r0["y"] - 1
            for c in moves:
                moves[c].append(s * (r1[c] - r0[c]))
    res["trajectory_by_weeks_left"] = traj
    res["moves_toward_outcome"] = {c: {"n": len(v), "share_toward": float(np.mean(np.array(v) > 0)),
                                       "share_away": float(np.mean(np.array(v) < 0)), "mean_signed": float(np.mean(v))}
                                   for c, v in moves.items() if v}
    ev_multi = [rr for rr in by_ev.values() if len(rr) >= 3]
    res["events_asked_3plus_weeks"] = len(ev_multi)
    # per-week table
    pw = []
    for w in summ["weeks"]:
        f = fits[w]
        wr = [r for r in rows if r["week"] == w]
        hw = [r for r in wr if r["held"]]
        sw = [r for r in wr if r["kind"] == "single"]
        comps = f["components"]
        pw.append({"week": w, "events": f["events"], "links": f["links"], "triangles": f["triangles"],
                   "answers": len(wr), "missing": f["missing"], "largest_component": max((c["n"] for c in comps), default=0),
                   "max_treewidth": max((c["tw"] for c in comps), default=0),
                   "heldout_kl": {c: float(kl(np.array([r["raw"] for r in hw]), np.array([r[c] for r in hw])).mean())
                                  for c in ["M0"] + models} if hw else None,
                   "single_logloss": {c: float(logloss(np.array([r[c] for r in sw]), ys(sw)).mean()) for c in ["raw"] + models},
                   "bp": {k: [{kk: ci.get(kk) for kk in ("nodes", "tw", "bp_residual", "bp_iters", "bp_converged", "refit_damp", "gibbs")}
                              for ci in f["fits"][keys[k]]["components"] if ci["how"] == "bp"] for k in MODELS[1:]},
                   "exact_components": sum(ci["how"] == "exact" for ci in f["fits"][keys["M3"]]["components"]),
                   "J_abs_mean_M2": _wmean(f["fits"][keys["M2"]]["components"], "J_abs_mean"),
                   "K_abs_mean_M3": _wmean(f["fits"][keys["M3"]]["components"], "K_abs_mean")})
    res["per_week"] = pw
    res["answers"] = {"rows": len(rows), "by_kind": {k: sum(r["kind"] == k for r in rows) for k in ("single", "and2", "and3")},
                      "heldout": len(held)}
    (d / f"summary{a.tag}.json").write_text(json.dumps(res, indent=1))
    charts = {
        "heldout_kl_by_week": [{"week": p["week"], **(p["heldout_kl"] or {})} for p in pw],
        "single_logloss_by_week": [{"week": p["week"], **p["single_logloss"]} for p in pw],
        "trajectory_logloss_by_weeks_left": [{"bucket": k, "n": v["n"], **{c: v[c]["logloss"] for c in ["raw"] + models}}
                                             for k, v in traj.items()],
        "main_table": {name: {c: res["outcome_logloss"][name][c] for c in ["raw"] + models}
                       for name in ("single", "joint_all", "joint_heldout")} |
                      {"heldout_kl": {c: res["heldout_kl"]["all"][c] for c in ["M0"] + models}},
        "by_relation_heldout_kl": {t: {c: v["heldout_kl"].get(c, {}).get("mean") for c in ["M0"] + models} | {"n": v["heldout_kl"]["n"]}
                                   for t, v in res["by_relation"].items()},
    }
    (d / f"charts{a.tag}.json").write_text(json.dumps(charts, indent=1))
    print(json.dumps({k: res[k] for k in ("heldout_kl", "outcome_logloss")}, indent=1)[:6000])


def _wmean(comps, key):
    xs = [(c.get(key), c["links"]) for c in comps if c.get(key) is not None]
    tot = sum(n for _, n in xs)
    return sum(v * n for v, n in xs) / tot if tot else None


if __name__ == "__main__":
    main()
