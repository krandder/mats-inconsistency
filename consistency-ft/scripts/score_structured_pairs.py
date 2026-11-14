"""Score the structured-pair runs (scripts/structured_pairs.py). CPU only.

Methods (per model x arm):
  raw        the 12 stated probabilities (missing scored at 0.5)
  projected  KL coherence projection over the 4 atoms (scripts/coherence_projection.py)
  hybrid     market marginals + the model's projected log odds ratio (as in the market-context test)
  hybrid_and3  market marginals + the odds ratio of the model's RAW stated P, Q, "P and Q" (3 forms,
             no projection; stated AND clipped into its Frechet range).  Uses no known relation.
Baselines:
  market_indep   price at t, independence
  market_oracle  price at t + the KNOWN relation: implication -> P(A and B) = min(a, b) (upper
                 Frechet bound); exclusivity -> P(A and B) = max(0, a + b - 1) (lower bound);
                 control -> independence.
Market prices: clipped to [0.01, 0.99].  For mc_excl the per-option last-bet prices of the store
are stale (no arbitrage fills), so the main market uses prices normalized by the sum over open
options at t; the raw last-bet version is reported too ("market_*_rawprice").

CIs: paired cluster bootstrap (2,000 resamples, seed 20260924) over groups (ladder group,
MC market, control pair), metric = mean over pairs of the mean over the group's forms.

Structure respect (raw stated answers vs projected atoms), with A = antecedent:
  implication: order p(A) <= p(B) + 0.01; zero cell p(A and not B) <= 0.05;
               AND ~ antecedent |p(A and B) - p(A)| <= 0.05;
               dependence captured D = (p11 - p_A p_B) / (min(p_A, p_B) - p_A p_B), own marginals.
  exclusivity: p(A and B) <= 0.05; p(A) + p(B) <= 1.01; D = (p_A p_B - p11) / (p_A p_B - max(0, p_A + p_B - 1)).
Market check at t: implication violated if price(A) > price(B); exclusivity if price(A) + price(B) > 1.

Usage: score_structured_pairs.py [--out-json docs/structured-pairs-20260924-charts.json]
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
from scripts.score_market_context_zeroshot import joint_with_or  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

OUT = Path("/home/agent/data/structured-pairs-20260924")
LO = 0.01
SINGLE = ("P", "NOT P", "Q", "NOT Q")
JOINT = tuple(f for f in FORM_ORDER if f not in SINGLE)
GROUPS = {"joint": JOINT, "single": SINGLE, "all12": FORM_ORDER}
TYPES = ("threshold", "date", "mc_excl", "control")
B = 2000
A_MAT = np.array([FORM_ATOMS[f] for f in FORM_ORDER], dtype=float)


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def cl(x: float) -> float:
    return min(1 - LO, max(LO, x))


def atoms_from(a: float, b: float, p11: float) -> np.ndarray:
    return np.array([p11, a - p11, b - p11, 1 - a - b + p11])


def forms_of(atoms: np.ndarray) -> dict[str, float]:
    return dict(zip(FORM_ORDER, (A_MAT @ atoms).tolist()))


def orient(pair: dict, x: dict) -> tuple[float, float, float, float]:
    """(p(A), p(B), p(A and B), p(A and not B)) with A the antecedent, from form values x."""
    if pair["relation"].get("implies") == "Q->P":
        return x["Q"], x["P"], x["(P) AND (Q)"], x["(NOT P) AND (Q)"]
    return x["P"], x["Q"], x["(P) AND (Q)"], x["(P) AND (NOT Q)"]


def dep_captured(kind: str, a: float, b: float, p11: float) -> float | None:
    if kind in ("threshold", "date"):
        den = min(a, b) - a * b
        return None if den < 0.01 else (p11 - a * b) / den
    if kind == "mc_excl":
        den = a * b - max(0.0, a + b - 1)
        return None if den < 0.01 else (a * b - p11) / den
    return None


def respect(kind: str, pair: dict, x: dict) -> dict:
    if kind in ("threshold", "date"):
        a, b, ab, anb = orient(pair, x)
        return {"order": a <= b + 0.01, "zero_cell": anb <= 0.05, "and_eq_antecedent": abs(ab - a) <= 0.05,
                "dep": dep_captured(kind, a, b, ab)}
    if kind == "mc_excl":
        a, b, ab = x["P"], x["Q"], x["(P) AND (Q)"]
        return {"and_zero": ab <= 0.05, "sum_le_1": a + b <= 1.01, "dep": dep_captured(kind, a, b, ab)}
    return {}


def loss(p: float | None, y: int) -> tuple[float, float]:
    p = 0.5 if p is None else cl(p)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p)), (p - y) ** 2


def per_pair(preds: dict, labels: dict) -> dict:
    out = {g: {} for g in GROUPS}
    for pid, fp in preds.items():
        for g, forms in GROUPS.items():
            ls = [loss(fp[f], labels[pid][f]) for f in forms]
            out[g][pid] = np.array([np.mean([l[0] for l in ls]), np.mean([l[1] for l in ls])])
    return out


def boot(x: dict, y: dict, pairs: list[str], cluster: dict, seed: int = 20260924) -> dict | None:
    if not pairs:
        return None
    cl_ids = sorted({cluster[p] for p in pairs})
    idx_of = defaultdict(list)
    for i, p in enumerate(pairs):
        idx_of[cluster[p]].append(i)
    D = np.array([x[p] - y[p] for p in pairs])
    S = np.array([D[idx_of[c]].sum(0) for c in cl_ids]); N = np.array([len(idx_of[c]) for c in cl_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cl_ids), size=(B, len(cl_ids)))
    res = {}
    for k, name in enumerate(("log_loss", "brier")):
        d = S[draws, k].sum(1) / N[draws].sum(1)
        res[name] = {"diff": float(D[:, k].mean()), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return res


def metric(pp: dict, pairs: list[str]) -> dict:
    X = np.array([pp[p] for p in pairs])
    return {"log_loss": float(X[:, 0].mean()), "brier": float(X[:, 1].mean()), "pairs": len(pairs)}


def market_preds(pair: dict, raw_price: bool = False) -> tuple[dict, dict]:
    if pair["type"] == "mc_excl" and not raw_price:
        a, b = cl(pair["p_price_norm"]), cl(pair["q_price_norm"])
    else:
        a, b = cl(pair["p_price_t"]), cl(pair["q_price_t"])
    indep = forms_of(atoms_from(a, b, a * b))
    kind = pair["type"]
    if kind in ("threshold", "date"):
        p11 = min(a, b)
    elif kind == "mc_excl":
        p11 = max(0.0, a + b - 1)
    else:
        p11 = a * b
    return indep, forms_of(atoms_from(a, b, p11))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--out-json", type=Path, default=Path("docs/structured-pairs-20260924-charts.json"))
    a = ap.parse_args()
    pairs = {p["pair_id"]: p for p in jl(a.out / "pairs.jsonl")}
    lab = {r["pair_id"]: (r["p"], r["q"]) for r in jl(a.out / "labels.jsonl")}
    labels = {}
    for pid, (yp, yq) in lab.items():
        cell = [(1, 1), (1, 0), (0, 1), (0, 0)].index((yp, yq))
        labels[pid] = {f: FORM_ATOMS[f][cell] for f in FORM_ORDER}
    cluster = {pid: p["group"] for pid, p in pairs.items()}
    by_type = {t: sorted(pid for pid, p in pairs.items() if p["type"] == t) for t in TYPES}
    res = {"counts": {t: len(v) for t, v in by_type.items()},
           "groups": {t: len({cluster[p] for p in v}) for t, v in by_type.items()},
           "outcome_check": {}, "market_check": {}, "baselines": {}, "jobs": {}}

    # rule validity on outcomes (reported, never used for selection)
    for t in ("threshold", "date"):
        bad = sum(1 for pid in by_type[t] if orient(pairs[pid], {"P": lab[pid][0], "Q": lab[pid][1],
                  "(P) AND (Q)": lab[pid][0] * lab[pid][1], "(P) AND (NOT Q)": lab[pid][0] * (1 - lab[pid][1]),
                  "(NOT P) AND (Q)": (1 - lab[pid][0]) * lab[pid][1]})[3] == 1)
        res["outcome_check"][t] = {"violations": bad, "pairs": len(by_type[t])}
    res["outcome_check"]["mc_excl"] = {"violations": sum(lab[p] == (1, 1) for p in by_type["mc_excl"]),
                                       "pairs": len(by_type["mc_excl"])}
    res["base_rates"] = {t: {"both_yes": float(np.mean([lab[p] == (1, 1) for p in v])),
                             "p_yes": float(np.mean([lab[p][0] for p in v] + [lab[p][1] for p in v]))}
                         for t, v in by_type.items()}

    # market check at t
    mviol = {}
    for t in ("threshold", "date"):
        rows = []
        for pid in by_type[t]:
            p = pairs[pid]
            pa, pb = (p["q_price_t"], p["p_price_t"]) if p["relation"]["implies"] == "Q->P" else (p["p_price_t"], p["q_price_t"])
            rows.append(pa - pb)
            mviol[pid] = pa > pb
        r = np.array(rows)
        res["market_check"][t] = {"violations": int((r > 0).sum()), "violations_gt_0.02": int((r > 0.02).sum()),
                                  "violations_gt_0.05": int((r > 0.05).sum()), "ties": int((r == 0).sum()),
                                  "pairs": len(r), "mean_excess_when_violated": float(r[r > 0].mean()) if (r > 0).any() else 0.0}
    s = np.array([pairs[p]["p_price_t"] + pairs[p]["q_price_t"] for p in by_type["mc_excl"]])
    tot = np.array([pairs[p]["relation"]["sum_open_prices"] for p in by_type["mc_excl"]])
    for pid in by_type["mc_excl"]:
        mviol[pid] = pairs[pid]["p_price_t"] + pairs[pid]["q_price_t"] > 1
    res["market_check"]["mc_excl"] = {"violations_rawprice": int((s > 1).sum()), "violations_gt_1.02": int((s > 1.02).sum()),
                                      "pairs": len(s), "median_sum_open_options": float(np.median(tot)),
                                      "note": "normalized prices cannot violate; raw last-bet prices are stale in the store"}

    mk = {}; orc = {}; mk_raw = {}; orc_raw = {}
    for pid, p in pairs.items():
        mk[pid], orc[pid] = market_preds(p)
        mk_raw[pid], orc_raw[pid] = market_preds(p, raw_price=True)
    base = {"market_indep": per_pair(mk, labels), "market_oracle": per_pair(orc, labels),
            "market_indep_rawprice": per_pair(mk_raw, labels), "market_oracle_rawprice": per_pair(orc_raw, labels)}
    for t, v in by_type.items():
        res["baselines"][t] = {m: {g: metric(base[m][g], v) for g in GROUPS} for m in base}
        res["baselines"][t]["oracle_vs_indep"] = {g: boot(base["market_oracle"][g], base["market_indep"][g], v, cluster)
                                                  for g in GROUPS}

    store = {}
    for log in sorted((a.out / "logs").glob("*-structured.jsonl")):
        recs = jl(log)
        model, arm = recs[0]["model"], recs[0]["arm"]
        byp = defaultdict(dict)
        for r in recs:
            byp[r["pair_id"]][r["form"]] = r["p"]
        done = sorted(p for p, v in byp.items() if len(v) == 12)
        raw, proj, hyb, hyb3, patoms = {}, {}, {}, {}, {}
        for pid in done:
            st = byp[pid]
            have = [f for f in FORM_ORDER if st[f] is not None]
            q = fit(have, np.array([st[f] for f in have], dtype=float), "kl") if have else np.full(4, .25)
            patoms[pid] = q
            raw[pid] = {f: st[f] for f in FORM_ORDER}
            proj[pid] = forms_of(q)
            qa = np.clip(q, 1e-6, None)
            lt = math.log(qa[0]) + math.log(qa[3]) - math.log(qa[1]) - math.log(qa[2])
            p = pairs[pid]
            if p["type"] == "mc_excl":
                ma, mb = cl(p["p_price_norm"]), cl(p["q_price_norm"])
            else:
                ma, mb = cl(p["p_price_t"]), cl(p["q_price_t"])
            hyb[pid] = forms_of(joint_with_or(ma, mb, lt))
            # hybrid_and3: odds ratio from the model's raw stated P, Q and "P and Q" only (no projection);
            # the stated AND is clipped into the Frechet range of the stated marginals.
            sp_, sq_, sa_ = (st["P"], st["Q"], st["(P) AND (Q)"])
            if None in (sp_, sq_, sa_):
                lt3 = lt
            else:
                sp_, sq_ = cl(sp_), cl(sq_)
                s11 = min(min(sp_, sq_), max(max(0.0, sp_ + sq_ - 1), sa_))
                c = np.clip(atoms_from(sp_, sq_, s11), 1e-6, None)
                lt3 = math.log(c[0]) + math.log(c[3]) - math.log(c[1]) - math.log(c[2])
            hyb3[pid] = forms_of(joint_with_or(ma, mb, lt3))
        meths = {"raw": per_pair(raw, labels), "projected": per_pair(proj, labels), "hybrid": per_pair(hyb, labels),
                 "hybrid_and3": per_pair(hyb3, labels)}
        job = {"model": model, "arm": arm, "pairs": len(done), "calls": len(recs),
               "missing": sum(r["p"] is None for r in recs), "cost_usd": sum(r["cost"] for r in recs),
               "served_models": dict(Counter(r["served_model"] for r in recs)), "by_type": {}}
        for t in TYPES + ("all",):
            v = [p for p in done if t == "all" or pairs[p]["type"] == t]
            if not v:
                continue
            e = {"scores": {m: {g: metric(meths[m][g], v) for g in GROUPS} for m in meths},
                 "market_indep": {g: metric(base["market_indep"][g], v) for g in GROUPS},
                 "market_oracle": {g: metric(base["market_oracle"][g], v) for g in GROUPS},
                 "vs_market_indep": {m: {g: boot(meths[m][g], base["market_indep"][g], v, cluster) for g in GROUPS}
                                     for m in meths},
                 "vs_market_oracle": {m: {g: boot(meths[m][g], base["market_oracle"][g], v, cluster) for g in GROUPS}
                                      for m in meths},
                 "projected_vs_raw": {g: boot(meths["projected"][g], meths["raw"][g], v, cluster) for g in GROUPS}}
            if t in ("threshold", "date", "mc_excl"):
                rr = [respect(t, pairs[p], raw[p]) for p in v if all(raw[p][f] is not None for f in FORM_ORDER)]
                pr = [respect(t, pairs[p], proj[p]) for p in v]
                keys = [k for k in rr[0] if k != "dep"]

                def summ(rows):
                    d = [r["dep"] for r in rows if r["dep"] is not None]
                    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
                    out.update(dep_median=float(np.median(d)) if d else None,
                               dep_mean_clipped=float(np.mean(np.clip(d, -1, 1))) if d else None,
                               dep_share_ge_half=float(np.mean(np.array(d) >= 0.5)) if d else None, dep_n=len(d))
                    return out
                e["respect"] = {"raw": summ(rr), "projected": summ(pr)}
                # where the market violates the relation at t
                vv = [p for p in v if mviol.get(p)]
                if vv:
                    e["market_violation_subset"] = {
                        "pairs": len(vv),
                        "market_indep": {g: metric(base["market_indep"][g], vv) for g in GROUPS},
                        "market_oracle": {g: metric(base["market_oracle"][g], vv) for g in GROUPS},
                        **{m: {g: metric(meths[m][g], vv) for g in GROUPS} for m in meths},
                        "projected_vs_market_indep": {g: boot(meths["projected"][g], base["market_indep"][g], vv, cluster)
                                                      for g in GROUPS},
                        "hybrid_vs_market_indep": {g: boot(meths["hybrid"][g], base["market_indep"][g], vv, cluster)
                                                   for g in GROUPS}}
            if t == "control" or t == "all":
                lts = []
                for p in v:
                    qa = np.clip(patoms[p], 1e-6, None)
                    lts.append(math.log(qa[0]) + math.log(qa[3]) - math.log(qa[1]) - math.log(qa[2]))
                lts = np.array(lts)
                e["log_odds_ratio"] = {"median": float(np.median(lts)), "share_abs_gt_1": float((abs(lts) > 1).mean())}
            job["by_type"][t] = e
        res["jobs"][f"{model}|{arm}"] = job
        store[f"{model}|{arm}"] = (done, meths)
    res["comparisons"] = []
    for x, y in (("qwen3-32b|P", "qwen3-32b|Q"), ("qwen3-235b-a22b|Q", "qwen3-32b|Q"), ("qwen3-235b-a22b|P", "qwen3-235b-a22b|Q"),
                 ("qwen3-235b-a22b|P", "qwen3-32b|P")):
        if x not in store or y not in store:
            continue
        common = sorted(set(store[x][0]) & set(store[y][0]))
        comp = {"x": x, "y": y, "by_type": {}}
        for t in TYPES + ("all",):
            v = [p for p in common if t == "all" or pairs[p]["type"] == t]
            if v:
                comp["by_type"][t] = {"pairs": len(v), **{m: boot(store[x][1][m]["joint"], store[y][1][m]["joint"], v, cluster)
                                                        for m in store[x][1]}}
        res["comparisons"].append(comp)
    (a.out / "scores.json").write_text(json.dumps(res, indent=1))
    rows = []
    for name, j in res["jobs"].items():
        for t, e in j["by_type"].items():
            for g in GROUPS:
                rows.append({"model": j["model"], "arm": j["arm"], "type": t, "forms": g, "method": "market_indep",
                             "log_loss": e["market_indep"][g]["log_loss"], "brier": e["market_indep"][g]["brier"]})
                rows.append({"model": j["model"], "arm": j["arm"], "type": t, "forms": g, "method": "market_oracle",
                             "log_loss": e["market_oracle"][g]["log_loss"], "brier": e["market_oracle"][g]["brier"]})
                for m in ("raw", "projected", "hybrid", "hybrid_and3"):
                    vi, vo = e["vs_market_indep"][m][g], e["vs_market_oracle"][m][g]
                    rows.append({"model": j["model"], "arm": j["arm"], "type": t, "forms": g, "method": m,
                                 "pairs": e["scores"][m][g]["pairs"], "log_loss": e["scores"][m][g]["log_loss"],
                                 "brier": e["scores"][m][g]["brier"],
                                 "ll_minus_market_indep": vi["log_loss"]["diff"], "ll_minus_market_indep_ci95": vi["log_loss"]["ci95"],
                                 "ll_minus_market_oracle": vo["log_loss"]["diff"], "ll_minus_market_oracle_ci95": vo["log_loss"]["ci95"]})
    a.out_json.write_text(json.dumps({"units": "mean log loss / Brier per query, clip 0.01, missing 0.5; diff negative = "
                                      "method better; ci95 = cluster bootstrap over groups (2,000)",
                                      "counts": res["counts"], "market_check": res["market_check"],
                                      "outcome_check": res["outcome_check"], "rows": rows,
                                      "respect": {n: {t: e.get("respect") for t, e in j["by_type"].items()}
                                                  for n, j in res["jobs"].items()}}, indent=1))
    print(json.dumps({k: res[k] for k in ("counts", "groups", "outcome_check", "market_check", "base_rates")}, indent=1))


if __name__ == "__main__":
    main()
