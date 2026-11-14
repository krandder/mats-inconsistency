"""Score verbalized-probability predictions (runs on General, CPU only).

Usage:
  score_verbalized_baseline.py --data-root D --pred predictions-<split>-<variant>.jsonl --out scores.json
      [--calibrate-from predictions-validation-<variant>.jsonl]

Every metric is reported with missing predictions excluded and with missing
imputed at 0.5; probabilities are clipped to [0.01, 0.99] (the range the prompt
asks for) and also scored raw (eps 1e-7).  Baselines are computed on the same rows.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from training.supervised_pairs import FORM_ORDER, relation_penalties
from training import verbalized_baseline as vb

PRIOR = {"lit": 0.5, "AND": 0.25, "OR": 0.75}


def op_of(form: str) -> str:
    return "AND" if "AND" in form else "OR" if "OR" in form else "lit"


def train_form_rates(root: Path, labels_file: str = "full-label-targets.jsonl") -> dict[str, float]:
    q = {r["query_id"]: r["form"] for r in map(json.loads, (root / "train/queries.jsonl").read_text().splitlines())}
    c = defaultdict(lambda: [0, 0])
    for r in map(json.loads, (root / "train" / labels_file).read_text().splitlines()):
        c[q[r["query_id"]]][0] += r["label"]; c[q[r["query_id"]]][1] += 1
    return {f: y / n for f, (y, n) in c.items()}


def proposition_strata(root: Path, split: str) -> tuple[set, set]:
    options, edited = set(), set()
    for p in map(json.loads, (root / split / "propositions.jsonl").read_text().splitlines()):
        if p.get("answer_text"):
            options.add(p["id"])
        if vb.strip_creator_edits(p.get("criteria_text") or "")[1]:
            edited.add(p["id"])
    return options, edited


def involved(row: dict) -> list[str]:
    f = row["form"]
    if f in ("P", "NOT P"):
        return [row["p_id"]]
    if f in ("Q", "NOT Q"):
        return [row["q_id"]]
    return [row["p_id"], row["q_id"]]


def score(rows: list[dict], key: str = "p", *, impute: bool, clip_lo: float = 0.01) -> dict:
    ps, ys = [], []
    for r in rows:
        p = r[key]
        if p is None:
            if not impute:
                continue
            p = 0.5
        ps.append(vb.clip(p, clip_lo)); ys.append(r["label"])
    return vb.metrics(ps, ys)


def both(rows, key="p"):
    return {"missing_excluded": score(rows, key, impute=False), "imputed_0.5": score(rows, key, impute=True),
            "raw_eps_1e-7_missing_excluded": {k: v for k, v in score(rows, key, impute=False, clip_lo=1e-7).items()
                                               if k != "reliability"}}


def consistency(rows: list[dict], key: str = "p") -> dict:
    by_pair = defaultdict(dict)
    for r in rows:
        by_pair[r["pair_id"]][r["form"]] = r[key]
    out = {}
    for mode in ("missing_excluded", "imputed_0.5"):
        totals, terms = [], defaultdict(list)
        for probs in by_pair.values():
            if len(probs) != 12:
                continue
            if any(v is None for v in probs.values()):
                if mode == "missing_excluded":
                    continue
                probs = {k: 0.5 if v is None else v for k, v in probs.items()}
            t = relation_penalties({k: vb.clip(v) for k, v in probs.items()})
            totals.append(t["total"])
            for k, v in t.items():
                terms[k].append(v)
        out[mode] = {"pairs": len(totals), "mean_squared_violation": sum(totals) / len(totals) if totals else None,
                     "by_term": {k: sum(v) / len(v) for k, v in terms.items() if k != "total"}}
    return out


def missing_report(rows: list[dict]) -> dict:
    by_form = defaultdict(lambda: [0, 0])
    reasons = defaultdict(int)
    for r in rows:
        by_form[r["form"]][0] += r["p"] is None; by_form[r["form"]][1] += 1
        if r["p"] is None:
            reasons[r["reason"]] += 1
    n = len(rows)
    return {"queries": n, "first_try_unparseable": sum(r["parsed_first"] is None for r in rows),
            "missing_after_retry": sum(r["p"] is None for r in rows),
            "missing_rate": sum(r["p"] is None for r in rows) / n,
            "by_form": {f: {"missing": m, "n": k} for f, (m, k) in by_form.items()}, "reasons": dict(reasons)}


def fit_platt(rows: list[dict]) -> dict:
    """Fit p' = sigmoid(a*logit(p)+b) by Newton on non-missing rows."""
    xs = [(math.log(vb.clip(r["p"]) / (1 - vb.clip(r["p"]))), r["label"]) for r in rows if r["p"] is not None]
    a, b = 1.0, 0.0
    for _ in range(50):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in xs:
            q = 1 / (1 + math.exp(-(a * x + b))); d = q - y; w = q * (1 - q)
            ga += d * x; gb += d; haa += w * x * x; hab += w * x; hbb += w
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det; db = (haa * gb - hab * ga) / det
        a -= da; b -= db
        if abs(da) + abs(db) < 1e-10:
            break
    return {"a": a, "b": b, "n": len(xs)}


def apply_platt(p, c):
    if p is None:
        return None
    x = math.log(vb.clip(p) / (1 - vb.clip(p)))
    return 1 / (1 + math.exp(-(c["a"] * x + c["b"])))


def report(rows: list[dict], root: Path, split: str) -> dict:
    rates = train_form_rates(root)
    rates_one = train_form_rates(root, "one-label-targets.jsonl")  # reproduces the 0.578 observed-form baseline
    options, edited = proposition_strata(root, split)
    for r in rows:
        r["b_const"] = 0.5; r["b_prior"] = PRIOR[op_of(r["form"])]; r["b_rate"] = rates[r["form"]]; r["b_rate1"] = rates_one[r["form"]]
        ids = involved(r)
        r["_type"] = "option" if all(i in options for i in ids) else "binary" if not any(i in options for i in ids) else "mixed"
        r["_edited"] = "edited" if any(i in edited for i in ids) else "unedited"
    obs = [r for r in rows if r["observed"]]
    res = {"missing": missing_report(rows), "missing_observed": missing_report(obs),
           "all_forms": both(rows), "observed": both(obs),
           "per_form": {f: both([r for r in rows if r["form"] == f]) for f in FORM_ORDER},
           "consistency": consistency(rows),
           "baselines": {name: {"all_forms": score(rows, key, impute=False), "observed": score(obs, key, impute=False),
                                "consistency": consistency(rows, key)["missing_excluded"]["mean_squared_violation"]}
                         for name, key in (("constant_0.5", "b_const"), ("boolean_prior", "b_prior"), ("train_form_rate_full_labels", "b_rate"), ("train_form_rate_one_label", "b_rate1"))}}
    for strat in ("_type", "_edited"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[strat]].append(r)
        res["by" + strat] = {g: {"missing": missing_report(v)["missing_rate"], "n": len(v), **both(v),
                                 "train_form_rate_baseline": score(v, "b_rate", impute=False)} for g, v in groups.items()}
    for r in rows:
        for k in ("b_const", "b_prior", "b_rate", "b_rate1", "_type", "_edited"):
            r.pop(k)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate-from")
    args = ap.parse_args()
    root = Path(args.data_root)
    rows = [json.loads(l) for l in Path(args.pred).read_text().splitlines()]
    split = rows[0]["pair_id"].split("-pair-")[0]
    res = {"pred": args.pred, "split": split, **report(rows, root, split)}
    if args.calibrate_from:
        cal_rows = [json.loads(l) for l in Path(args.calibrate_from).read_text().splitlines()]
        c = fit_platt(cal_rows)
        cal = [dict(r, p=apply_platt(r["p"], c)) for r in rows]
        res["platt"] = {"fit_on": args.calibrate_from, **c, "all_forms": both(cal),
                        "observed": both([r for r in cal if r["observed"]]), "consistency": consistency(cal)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    a, o = res["all_forms"]["imputed_0.5"], res["observed"]["imputed_0.5"]
    print(f"{Path(args.pred).name}: missing {res['missing']['missing_rate']:.2%} | all-12 LL {a['log_loss']:.4f} Brier {a['brier']:.4f} "
          f"acc {a['accuracy']:.3f} ECE {a['ece']:.3f} | observed LL {o['log_loss']:.4f} | consistency "
          f"{res['consistency']['missing_excluded']['mean_squared_violation']:.4f}")


if __name__ == "__main__":
    main()
