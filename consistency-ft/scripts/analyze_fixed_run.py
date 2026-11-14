#!/usr/bin/env python3
"""Offline analysis of the fixed-prompt Qwen3-0.6B run (2026-09-23).

Reads every arm's per-epoch prediction dumps (validation from training, test
from rescore_checkpoints.py), and writes one JSON summary with, per arm and
epoch: observed-form and all-12-form log loss / Brier / accuracy / ECE,
consistency MSE, SD of p(AND) across pairs, corr(p(P), p(NOT P)), per-form log
loss and a multiple-choice vs binary split.  Baselines are scored on the same
rows: constant 0.5, Boolean prior, per-form train base rate (one-label and
all-12 tables) and, when given, the Qwen3-32B zero-shot verbalized predictions.
The headline epoch is the validation-selected one (best.txt).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from training.supervised_pairs import (FORM_ORDER, FORM_PRIOR, LABEL_FILES, load_split, probability_metrics,
                                       relation_penalties)

EPS = 1e-7


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def ll(p: float, y: int) -> float:
    p = min(1 - EPS, max(EPS, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or statistics.pstdev(x) == 0 or statistics.pstdev(y) == 0:
        return None
    return statistics.correlation(x, y)


def score(pred: dict[str, float], rows: list[dict], mc_pairs: set[str]) -> dict:
    """pred maps query_id -> p; rows are the reference (pair_id, query_id, form, label, observed)."""
    obs = [(pred[r["query_id"]], r["label"]) for r in rows if r["observed"]]
    full = [(pred[r["query_id"]], r["label"]) for r in rows if r["label"] is not None]
    out = {"observed": probability_metrics([p for p, _ in obs], [y for _, y in obs]),
           "all12": probability_metrics([p for p, _ in full], [y for _, y in full])}
    by_pair: dict[str, dict[str, float]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["form"]] = pred[r["query_id"]]
    out["consistency_mse"] = statistics.fmean(relation_penalties(v)["total"] for v in by_pair.values())
    ands = [v["(P) AND (Q)"] for v in by_pair.values()]
    out["sd_p_and"] = statistics.pstdev(ands)
    out["sd_p_and_4forms_mean"] = statistics.fmean(
        statistics.pstdev([v[f] for v in by_pair.values()]) for f in FORM_ORDER if "AND" in f)
    out["corr_p_notp"] = corr([v["P"] for v in by_pair.values()], [v["NOT P"] for v in by_pair.values()])
    out["corr_q_notq"] = corr([v["Q"] for v in by_pair.values()], [v["NOT Q"] for v in by_pair.values()])
    out["per_form_log_loss"] = {f: statistics.fmean(ll(pred[r["query_id"]], r["label"]) for r in rows
                                                    if r["form"] == f and r["label"] is not None) for f in FORM_ORDER}
    for name, keep in (("multiple_choice", True), ("binary", False)):
        sub = [r for r in rows if (r["pair_id"] in mc_pairs) == keep]
        so = [r for r in sub if r["observed"]]
        out[name] = {"pairs": len({r["pair_id"] for r in sub}),
                     "observed_log_loss": statistics.fmean(ll(pred[r["query_id"]], r["label"]) for r in so) if so else None,
                     "all12_log_loss": statistics.fmean(ll(pred[r["query_id"]], r["label"]) for r in sub
                                                        if r["label"] is not None) if sub else None}
    return out


def pair_losses(pred: dict[str, float], rows: list[dict], observed: bool) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        if (r["observed"] if observed else r["label"] is not None):
            acc.setdefault(r["pair_id"], []).append(ll(pred[r["query_id"]], r["label"]))
    return {k: statistics.fmean(v) for k, v in acc.items()}


def paired_bootstrap(a: dict[str, float], b: dict[str, float], n: int = 2000, seed: int = 0) -> dict:
    """Mean(a - b) over pairs with a pair-level (not family-clustered) 95% interval."""
    keys = sorted(a)
    d = [a[k] - b[k] for k in keys]
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(n))
    return {"mean_diff": statistics.fmean(d), "ci95": [boots[int(.025 * n)], boots[int(.975 * n)]]}


def reference(data_root: Path, split: str) -> tuple[list[dict], set[str]]:
    one = load_split(data_root, split, labels_file=LABEL_FILES["one"])
    full = load_split(data_root, split, labels_file=LABEL_FILES["full"])
    rows, mc = [], set()
    for e, f in zip(one, full, strict=True):
        if any(str(e.propositions[e.pair[k]].get("answer_text") or "").strip() for k in ("p_id", "q_id")):
            mc.add(e.pair["pair_id"])
        for form in FORM_ORDER:
            qid = e.queries[form]["query_id"]
            rows.append({"pair_id": e.pair["pair_id"], "query_id": qid, "form": form,
                         "label": f.labels.get(qid), "observed": form == e.observed_form})
    return rows, mc


def base_rate_tables(data_root: Path) -> dict[str, dict[str, float]]:
    one = load_split(data_root, "train", labels_file=LABEL_FILES["one"])
    full = load_split(data_root, "train", labels_file=LABEL_FILES["full"])
    obs: dict[str, list[int]] = {f: [] for f in FORM_ORDER}
    for e in one:
        obs[e.observed_form].append(e.labels[e.pair["observed_query_id"]])
    allf: dict[str, list[int]] = {f: [] for f in FORM_ORDER}
    for e in full:
        for f, y in e.labeled_forms.items():
            allf[f].append(y)
    return {"base_rate_one_label": {f: statistics.fmean(v) if v else FORM_PRIOR[f] for f, v in obs.items()},
            "base_rate_all12": {f: statistics.fmean(v) for f, v in allf.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--qwen32b-test", help="predictions-test-*.jsonl from the 32B verbalized baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results, data_root = Path(a.results), Path(a.data_root)
    refs = {s: reference(data_root, s) for s in ("validation", "test")}
    tables = base_rate_tables(data_root)
    summary: dict = {"baselines": {}, "arms": {}, "base_rate_tables": tables}
    for split, (rows, mc) in refs.items():
        consts = {"constant_0.5": {f: .5 for f in FORM_ORDER}, "boolean_prior": FORM_PRIOR, **tables}
        for name, table in consts.items():
            summary["baselines"].setdefault(name, {})[split] = score({r["query_id"]: table[r["form"]] for r in rows}, rows, mc)
    test_rows, test_mc = refs["test"]
    if a.qwen32b_test and Path(a.qwen32b_test).exists():
        raw = {r["query_id"]: r.get("p") for r in jsonl(Path(a.qwen32b_test))}
        missing = sum(raw.get(r["query_id"]) is None for r in test_rows)
        pred = {r["query_id"]: .5 if raw.get(r["query_id"]) is None else float(raw[r["query_id"]]) for r in test_rows}
        summary["baselines"]["qwen3_32b_zero_shot"] = {"test": {**score(pred, test_rows, test_mc), "missing_imputed_0.5": missing}}
    test_pair_losses = {}
    for arm_dir in sorted(p for p in results.iterdir() if p.is_dir() and (p / "run-manifest.json").exists()):
        best = int((arm_dir / "best.txt").read_text()) if (arm_dir / "best.txt").exists() else None
        arm = {"best_epoch": best, "epochs": {}}
        for epoch_dir in sorted(arm_dir.glob("epoch-*")):
            ep = int(epoch_dir.name.split("-")[1]); rec = {}
            for split, path in (("validation", arm_dir / f"predictions-validation-epoch-{ep:02d}.jsonl"),
                                ("test", arm_dir / f"predictions-test-epoch-{ep:02d}.jsonl")):
                if path.exists():
                    pred = {r["query_id"]: r["probability"] for r in jsonl(path)}
                    rows, mc = refs[split]
                    rec[split] = score(pred, rows, mc)
                    if split == "test" and ep == best:
                        test_pair_losses[arm_dir.name] = {k: pair_losses(pred, rows, k == "observed")
                                                          for k in ("observed", "all12")}
            arm["epochs"][ep] = rec
        arm["headline"] = arm["epochs"].get(best, {})
        summary["arms"][arm_dir.name] = arm
    comparisons = {}
    for treat in [k for k in test_pair_losses if "consistency" in k]:
        for ctrl in [k for k in test_pair_losses if k != treat]:
            comparisons[f"{treat} - {ctrl}"] = {k: paired_bootstrap(test_pair_losses[treat][k], test_pair_losses[ctrl][k])
                                                for k in ("observed", "all12")}
    summary["paired_test_log_loss_differences"] = comparisons
    Path(a.out).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{'arm':28s} {'ep':>3s} {'obsLL':>7s} {'allLL':>7s} {'allBr':>6s} {'cons':>7s} {'sdAND':>6s} {'rPnP':>6s}")
    def line(name, s, ep=""):
        c = s.get("corr_p_notp")
        print(f"{name:28s} {str(ep):>3s} {s['observed']['log_loss']:7.4f} {s['all12']['log_loss']:7.4f} "
              f"{s['all12']['brier']:6.4f} {s['consistency_mse']:7.4f} {s['sd_p_and']:6.3f} {c if c is None else round(c, 3)!s:>6s}")
    for name, b in summary["baselines"].items():
        line(name, b["test"])
    for name, arm in summary["arms"].items():
        if "test" in arm["headline"]:
            line(name, arm["headline"]["test"], arm["best_epoch"])


if __name__ == "__main__":
    main()
