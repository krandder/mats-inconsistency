"""Score the market-context zero-shot runs (scripts/market_context_zeroshot.py). CPU only.

For each job (model x arm x split) and method:
  raw        the 12 stated probabilities (missing scored at 0.5)
  projected  coherence projection, KL fit over the 4 atoms (scripts/coherence_projection.py)
  hybrid     the market marginals p(P), p(Q) at t (clipped to [0.01, 0.99]) joined with the
             model's projected odds ratio  theta = (a d) / (b c)  of its fitted atoms; singles
             equal the market.  Does the model's dependence beat independence when the
             marginals come from the market?
Baseline: market independence (price at t, clip 0.01; AND = x y, OR = x + y - x y).

Form groups: joint (the 8 AND/OR forms incl. signed variants; main result), single (P, NOT P,
Q, NOT Q), all12.  Strata: same_topic, other.  Log loss and Brier, scores clipped to
[0.01, 0.99].  Paired bootstrap over pairs (2,000 resamples, seed 20260924), metric = mean
over the pairs' forms.  Labels come from full-label-targets.jsonl, strata from
pair-outcomes.jsonl (scoring only).

Usage: score_market_context_zeroshot.py [--out-json docs/market-context-zeroshot-20260924.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coherence_projection import FORM_ATOMS, fit  # noqa: E402
from scripts.market_context_baselines import form_probs  # noqa: E402
from scripts.score_verbalized_baseline import consistency  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

DATA = Path("/home/agent/data/jev-market-context-open-20260923")
OUT = Path("/home/agent/data/market-context-zeroshot-20260924")
LO = 0.01
SINGLE = ("P", "NOT P", "Q", "NOT Q")
JOINT = tuple(f for f in FORM_ORDER if f not in SINGLE)
GROUPS = {"joint": JOINT, "single": SINGLE, "all12": FORM_ORDER}
STRATA = ("all", "same_topic", "other")
ATOM_FLOOR = 1e-6
B = 2000


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_split(split: str):
    d = DATA / split
    qs = {q["query_id"]: q for q in jl(d / "queries.jsonl")}
    labels = {}
    for r in jl(d / "full-label-targets.jsonl"):
        q = qs[r["query_id"]]
        labels[(q["pair_id"], q["form"])] = r["label"]
    strata = {r["pair_id"]: r["stratum"] for r in jl(d / "pair-outcomes.jsonl")}
    market = {}
    for p in jl(d / "pairs.jsonl"):
        market[p["pair_id"]] = {"probs": form_probs(p["p_price_t"], p["q_price_t"], LO)["indep"],
                                "a": min(1 - LO, max(LO, p["p_price_t"])), "b": min(1 - LO, max(LO, p["q_price_t"]))}
    return labels, strata, market


def joint_with_or(a: float, b: float, log_theta: float) -> np.ndarray:
    """Atoms (PQ, P~Q, ~PQ, ~P~Q) with marginals a, b and log odds ratio log_theta (bisection on p11)."""
    lo, hi = max(0.0, a + b - 1), min(a, b)
    for _ in range(200):
        m = (lo + hi) / 2
        cells = np.array([m, a - m, b - m, 1 - a - b + m])
        cells = np.clip(cells, 1e-300, None)
        lt = math.log(cells[0]) + math.log(cells[3]) - math.log(cells[1]) - math.log(cells[2])
        if lt < log_theta:
            lo = m
        else:
            hi = m
    m = (lo + hi) / 2
    return np.array([m, a - m, b - m, 1 - a - b + m])


def methods_for_pair(stated: dict[str, float | None], mk: dict) -> tuple[dict, dict, float]:
    have = [f for f in FORM_ORDER if stated.get(f) is not None]
    q = fit(have, np.array([stated[f] for f in have], dtype=float), "kl") if have else np.full(4, 0.25)
    proj = {f: float(np.dot(FORM_ATOMS[f], q)) for f in FORM_ORDER}
    qa = np.clip(q, ATOM_FLOOR, None)
    log_theta = float(math.log(qa[0]) + math.log(qa[3]) - math.log(qa[1]) - math.log(qa[2]))
    atoms = joint_with_or(mk["a"], mk["b"], log_theta)
    hyb = {f: float(np.dot(FORM_ATOMS[f], atoms)) for f in FORM_ORDER}
    return proj, hyb, log_theta


def loss(p: float | None, y: int) -> tuple[float, float]:
    p = 0.5 if p is None else min(1 - LO, max(LO, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p)), (p - y) ** 2


def per_pair(preds: dict[str, dict[str, float | None]], labels) -> dict[str, dict[str, np.ndarray]]:
    """group -> pair -> [mean log loss, mean brier] over the group's forms."""
    out = {g: {} for g in GROUPS}
    for pid, fp in preds.items():
        for g, forms in GROUPS.items():
            ls = [loss(fp[f], labels[(pid, f)]) for f in forms]
            out[g][pid] = np.array([sum(x[0] for x in ls) / len(ls), sum(x[1] for x in ls) / len(ls)])
    return out


def boot(x: dict, y: dict, pairs: list[str], seed: int = 20260924) -> dict | None:
    if not pairs:
        return None
    X = np.array([x[p] for p in pairs]); Y = np.array([y[p] for p in pairs])
    D = X - Y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(B, len(pairs)))
    res = {}
    for k, name in enumerate(("log_loss", "brier")):
        d = D[idx, k].mean(1)
        res[name] = {"diff": float(D[:, k].mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


def metric(pp: dict, pairs: list[str]) -> dict:
    X = np.array([pp[p] for p in pairs])
    return {"log_loss": float(X[:, 0].mean()), "brier": float(X[:, 1].mean()), "pairs": len(pairs)}


def score_split(split: str) -> dict:
    labels, strata, market = load_split(split)
    jobs = {}
    for log in sorted((OUT / "logs").glob(f"*-{split}.jsonl")):
        recs = jl(log)
        model, arm = recs[0]["model"], recs[0]["arm"]
        by_pair = defaultdict(dict)
        for r in recs:
            by_pair[r["pair_id"]][r["form"]] = r
        complete = {p: v for p, v in by_pair.items() if len(v) == 12}
        jobs[f"{model}|{arm}"] = {"model": model, "arm": arm, "recs": recs, "by_pair": complete}
    mk_preds = {p: dict(market[p]["probs"]) for p in market}
    mk_pp = per_pair(mk_preds, labels)
    res = {"jobs": {}, "comparisons": []}
    pp_store = {}
    for name, j in jobs.items():
        recs = j["recs"]
        pairs = sorted(j["by_pair"])
        raw = {p: {f: j["by_pair"][p][f]["p"] for f in FORM_ORDER} for p in pairs}
        proj, hyb, lt = {}, {}, {}
        for p in pairs:
            proj[p], hyb[p], lt[p] = methods_for_pair(raw[p], market[p])
        rows_for_cons = [{"pair_id": p, "form": f, "p": raw[p][f]} for p in pairs for f in FORM_ORDER]
        cons_raw = consistency(rows_for_cons)
        missing = {"calls": len(recs), "api_fail_first": sum(r["raw"] is None for r in recs),
                   "unparsed_first": sum(r["parsed_first"] is None for r in recs),
                   "missing_after_retry": sum(r["p"] is None for r in recs),
                   "reasons": dict(Counter(r["reason"] for r in recs if r["p"] is None)),
                   "cost_usd": sum(r["cost"] for r in recs),
                   "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in recs),
                   "served_models": dict(Counter(r["served_model"] for r in recs))}
        missing["missing_rate"] = missing["missing_after_retry"] / max(1, missing["calls"])
        missing["unparsed_first_rate"] = missing["unparsed_first"] / max(1, missing["calls"])
        methods = {"raw": per_pair(raw, labels), "projected": per_pair(proj, labels), "hybrid": per_pair(hyb, labels)}
        pp_store[name] = (pairs, methods)
        out = {"model": j["model"], "arm": j["arm"], "pairs": len(pairs), "missing": missing,
               "consistency_raw_msv": cons_raw["imputed_0.5"]["mean_squared_violation"], "scores": {}, "vs_market": {},
               "projected_vs_raw": {}, "log_odds_ratio": {}}
        for s in STRATA:
            sp = [p for p in pairs if s == "all" or strata[p] == s]
            if not sp:
                continue
            ltv = np.array([lt[p] for p in sp])
            out["log_odds_ratio"][s] = {"mean": float(ltv.mean()), "median": float(np.median(ltv)),
                                        "share_positive": float((ltv > 0).mean())}
            out["scores"][s] = {"market_indep": {g: metric(mk_pp[g], sp) for g in GROUPS}}
            for mname, m in methods.items():
                out["scores"][s][mname] = {g: metric(m[g], sp) for g in GROUPS}
                out["vs_market"].setdefault(s, {})[mname] = {g: boot(m[g], mk_pp[g], sp) for g in GROUPS}
            out["projected_vs_raw"][s] = {g: boot(methods["projected"][g], methods["raw"][g], sp) for g in GROUPS}
        res["jobs"][name] = out
    # arm and model comparisons on common pairs
    names = sorted(pp_store)
    wanted = []
    for model in sorted({n.split("|")[0] for n in names}):
        for a, b in (("L", "P"), ("L", "Q"), ("P", "Q")):
            wanted.append((f"{model}|{a}", f"{model}|{b}"))
    for arm in ("L", "Q", "P"):
        wanted.append((f"qwen3-235b-a22b|{arm}", f"qwen3-32b|{arm}"))
    for x, y in wanted:
        if x not in pp_store or y not in pp_store:
            continue
        common = sorted(set(pp_store[x][0]) & set(pp_store[y][0]))
        comp = {"x": x, "y": y, "pairs": len(common), "by": {}}
        for s in STRATA:
            sp = [p for p in common if s == "all" or strata[p] == s]
            comp["by"][s] = {m: {g: boot(pp_store[x][1][m][g], pp_store[y][1][m][g], sp) for g in GROUPS}
                             for m in ("raw", "projected", "hybrid")}
        res["comparisons"].append(comp)
    return res


def fmt(v: dict) -> str:
    return f"{v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"


def chart_rows(res: dict) -> dict:
    """Flat rows for charts: one per split x job x method x group x stratum, with the market difference."""
    rows = []
    for split, r in res.items():
        for name, j in r["jobs"].items():
            for st, sc in j["scores"].items():
                for m in ("market_indep", "raw", "projected", "hybrid"):
                    for g in GROUPS:
                        if m == "hybrid" and g == "single":
                            continue
                        v = j["vs_market"][st][m][g]["log_loss"] if m != "market_indep" else None
                        vb_ = j["vs_market"][st][m][g]["brier"] if m != "market_indep" else None
                        rows.append({"split": split, "model": j["model"], "arm": j["arm"], "method": m, "forms": g,
                                     "stratum": st, "pairs": sc[m][g]["pairs"], "log_loss": sc[m][g]["log_loss"],
                                     "brier": sc[m][g]["brier"],
                                     "log_loss_minus_market": v["diff"] if v else 0.0,
                                     "log_loss_minus_market_ci95": v["ci95"] if v else None,
                                     "brier_minus_market": vb_["diff"] if vb_ else 0.0,
                                     "brier_minus_market_ci95": vb_["ci95"] if vb_ else None})
    comps = []
    for split, r in res.items():
        for c in r["comparisons"]:
            for st, by in c["by"].items():
                for m, gs in by.items():
                    for g, v in gs.items():
                        if v:
                            comps.append({"split": split, "x": c["x"], "y": c["y"], "method": m, "forms": g, "stratum": st,
                                          "log_loss_diff": v["log_loss"]["diff"], "ci95": v["log_loss"]["ci95"],
                                          "brier_diff": v["brier"]["diff"], "brier_ci95": v["brier"]["ci95"]})
    return {"units": "mean log loss / Brier per query, clip 0.01, missing scored 0.5; diff = x - y, negative = x better; "
                     "ci95 = paired bootstrap over pairs (2,000)", "scores": rows, "comparisons": comps}


def f3(v: dict | None, k: str = "log_loss") -> str:
    if not v:
        return "n/a"
    x = v[k]
    return f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def tables(res: dict) -> str:
    """Markdown tables for the doc."""
    out = []
    for split, r in res.items():
        J = r["jobs"]
        out.append(f"## {split}\n")
        out.append("| model | arm | pairs | calls | unparsed first try | missing after retry | cost $ | consistency MSV (raw) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for name, j in J.items():
            m = j["missing"]
            out.append(f"| {j['model']} | {j['arm']} | {j['pairs']} | {m['calls']} | {m['unparsed_first']} ({m['unparsed_first_rate']:.2%}) "
                       f"| {m['missing_after_retry']} ({m['missing_rate']:.2%}) | {m['cost_usd']:.2f} | {j['consistency_raw_msv']:.4f} |")
        for g, title in (("joint", "Joint forms (8 AND/OR incl. signed)"), ("single", "Single forms (P, NOT P, Q, NOT Q)"),
                         ("all12", "All 12 forms")):
            for st in STRATA:
                out.append(f"\n### {split}: {title}, stratum {st}\n")
                out.append("| model | arm | method | pairs | log loss | Brier | market log loss (same pairs) | Δ log loss vs market [95% CI] | Δ Brier vs market [95% CI] |")
                out.append("|---|---|---|---|---|---|---|---|---|")
                for name, j in J.items():
                    if st not in j["scores"]:
                        continue
                    sc = j["scores"][st]
                    for meth in ("raw", "projected", "hybrid"):
                        if meth == "hybrid" and g == "single":
                            continue
                        v = j["vs_market"][st][meth][g]
                        out.append(f"| {j['model']} | {j['arm']} | {meth} | {sc[meth][g]['pairs']} | {sc[meth][g]['log_loss']:.3f} | "
                                   f"{sc[meth][g]['brier']:.3f} | {sc['market_indep'][g]['log_loss']:.3f} | {f3(v)} | {f3(v, 'brier')} |")
                if g != "joint":
                    break  # strata only for joints
        out.append(f"\n### {split}: projected minus raw\n")
        out.append("| model | arm | joint Δ log loss | single Δ log loss | all-12 Δ log loss |")
        out.append("|---|---|---|---|---|")
        for name, j in J.items():
            pv = j["projected_vs_raw"]["all"]
            out.append(f"| {j['model']} | {j['arm']} | {f3(pv['joint'])} | {f3(pv['single'])} | {f3(pv['all12'])} |")
        out.append(f"\n### {split}: arm and model comparisons (x − y, joint forms)\n")
        out.append("| x | y | stratum | pairs | raw | projected | hybrid |")
        out.append("|---|---|---|---|---|---|---|")
        for c in r["comparisons"]:
            for st in STRATA:
                by = c["by"][st]
                out.append(f"| {c['x']} | {c['y']} | {st} | {c['pairs'] if st == 'all' else ''} | {f3(by['raw']['joint'])} | "
                           f"{f3(by['projected']['joint'])} | {f3(by['hybrid']['joint'])} |")
        out.append(f"\n### {split}: the model's dependence (log odds ratio of the projected atoms)\n")
        out.append("| model | arm | stratum | mean | median | share > 0 |")
        out.append("|---|---|---|---|---|---|")
        for name, j in J.items():
            for st, v in j["log_odds_ratio"].items():
                out.append(f"| {j['model']} | {j['arm']} | {st} | {v['mean']:+.2f} | {v['median']:+.2f} | {v['share_positive']:.2f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT / "scores.json")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--charts", type=Path)
    ap.add_argument("--tables", type=Path)
    a = ap.parse_args()
    res = {}
    for split in a.splits.split(","):
        res[split] = score_split(split)
        for name, j in res[split]["jobs"].items():
            s = j["scores"]["all"]
            v = j["vs_market"]["all"]
            print(f"{split:10s} {name:22s} pairs {j['pairs']:4d} miss {j['missing']['missing_rate']:.4f} "
                  f"mkt joint {s['market_indep']['joint']['log_loss']:.4f} | "
                  + " | ".join(f"{m} {s[m]['joint']['log_loss']:.4f} d {fmt(v[m]['joint']['log_loss'])}"
                               for m in ("raw", "projected", "hybrid")), flush=True)
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(res, indent=1))
    if a.tables:
        a.tables.write_text(tables(res))
    if a.charts:
        a.charts.write_text(json.dumps(chart_rows(res), indent=1))


if __name__ == "__main__":
    main()
