"""Analysis of the bias-balanced triangle run (scripts/balanced_triangles.py).

(a) Checks on the raw stated answers, per triangle:
    complement      mirT + T - 1;  mirP + pair_k - 1                         (0 if coherent)
    partition       none + one + two + all - 1                                (0)
    additivity      pair_k - T - sib_k  (3 per triangle)                      (0)
    single-2pairs   single_i - pair_j - pair_k + T  (pairs that contain i)    (>= 0; = P(L_i, not L_j, not L_k))
    order           T <= pair_k, sib_k <= pair_k, pair_k <= each of its singles (share of 12 comparisons violated)
    wording         all-three (counting words) - T (and-words)                (0)
    Bias: mean signed (stated - outcome) per question type and per type x number of negated literals.
(b) Fits over the 8 outcomes of each triangle, q = softmax(theta), minimising sum_forms KL(stated || implied)
    with equal weights, stated clipped to [0.01, 0.99]:
      fit16     all 16 answers
      fit16bc   all 16 answers after subtracting the per-type mean signed bias estimated on the OTHER weeks
      fitAND    the weekly run's 7 answers on the same triangle (3 singles, 3 "A and B", "A and B and C"),
                i.e. the AND-only design, projected the same way
      indep     product of the model's stated signed singles (this run)
      raw       each stated answer as is
      M3global / M3bglobal  the weekly run's global week fits (AND-only design, M3 and singles-anchored M3b),
                read from its fits/<week>.json, scored on the YES singles and the 7 AND forms only
    Outcome log loss per form (mean within triangle, then over triangles) on: singles (3 signed), the 3
    YES singles (raw = the weekly run's stated answers), all 16
    forms, the sign-balanced joints (T, 3 siblings, 3 pairs), the weekly run's 7 AND-only forms, and the
    8-outcome joint log loss -log q(outcome).  Fitted and independence q are mixed with 1% uniform and every
    binary prediction is clipped to [0.01, 0.99] (as the stated answers are) before scoring.  Paired bootstrap 95% CIs over triangles (2,000 resamples).
Writes <data>/summary.json, <data>/charts.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DATA = Path("/home/agent/data/balanced-triangles-20260927")
WEEKLY = Path("/home/agent/data/weekly-worldmodel-20260927/sixweek")
LO, HI = 0.01, 0.99
B = 2000
SEED = 20260927
GLOBAL = {"M3global": "M3@1.0", "M3bglobal": "M3b@0.3"}  # weekly_worldmodel_fit.py fits (lambda as tuned there)
TYPES = ["single", "pair", "T", "sib", "count0", "count1", "count2", "count3", "mirT", "mirP"]


def ftype(form):
    return form if form.startswith("count") else form.rstrip("0123456789")


def masks(atom_lists):
    M = np.zeros((len(atom_lists), 8))
    for r, at in enumerate(atom_lists):
        M[r, at] = 1
    return M


def project(p, M):
    """argmin_q sum_r KL(p_r || M_r q), q on the 8 outcomes."""
    p = np.clip(np.asarray(p, float), LO, HI)

    def f(th):
        q = np.exp(th - th.max()); q /= q.sum()
        m = np.clip(M @ q, 1e-12, 1 - 1e-12)
        loss = np.sum(p * np.log(p / m) + (1 - p) * np.log((1 - p) / (1 - m)))
        dm = -p / m + (1 - p) / (1 - m)
        g = q * (M.T @ dm) - q * np.dot(dm, M @ q)
        return loss, g
    r = minimize(f, np.zeros(8), jac=True, method="L-BFGS-B", options={"maxiter": 2000, "gtol": 1e-10})
    q = np.exp(r.x - r.x.max())
    return q / q.sum()


EPS_MIX = 0.01  # every fitted / independence distribution is mixed with 1% uniform before scoring


def ll(p, y):
    p = min(max(p, LO), HI)  # every variant's binary predictions clipped like the stated answers
    return -np.log(p if y else 1 - p)


def pos_atoms(kind, ks):
    return [a for a, x in enumerate(itertools.product((0, 1), repeat=3)) if all(x[k] for k in ks)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    a = ap.parse_args()
    D = a.data
    tris = [json.loads(l) for l in (D / "triangles.jsonl").read_text().splitlines()]
    ans = {}
    for l in (D / "log.jsonl").read_text().splitlines():
        r = json.loads(l)
        ans[r["prompt_sha256"]] = r["p"]
    # weekly AND-only answers for the same triangles
    need = defaultdict(dict)
    for t in tris:
        e = t["events"]
        need[t["week"]][("single", (e[0],))] = None
        for k in range(3):
            need[t["week"]][("single", (e[k],))] = None
        for i, j in itertools.combinations(range(3), 2):
            need[t["week"]][("and2", tuple(sorted((e[i], e[j]))))] = None
        need[t["week"]][("and3", tuple(sorted(e)))] = None
    qid, need_qid = {}, {}
    with (WEEKLY / "queries.jsonl").open() as f:
        for line in f:
            head = line[:200]
            wk = head.split('"week": "')[1][:10]
            if wk not in need:
                continue
            q = json.loads(line)
            key = (q["kind"], tuple(sorted(q["events"])))
            if key in need[wk]:
                qid[q["query_id"]] = (wk, key)
    with (WEEKLY / "log.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r["query_id"] in qid:
                wk, key = qid[r["query_id"]]
                need[wk][key] = r["p"]
                need_qid[(wk, key)] = r["query_id"]
    # the weekly run's global fits (M3 and singles-anchored M3b) on the same YES singles / ANDs
    gpred = {}
    for wk in need:
        fp = WEEKLY / "fits" / f"{wk}.json"
        if fp.exists():
            pr = json.loads(fp.read_text())["preds"]
            for m in GLOBAL:
                gpred[(wk, m)] = pr[GLOBAL[m]]

    keep, dropped = [], 0
    for t in tris:
        p = {f: ans.get(v["sha"]) for f, v in t["forms"].items()}
        e = t["events"]
        wkly = [need[t["week"]][("single", (e[k],))] for k in range(3)] + \
               [need[t["week"]][("and2", tuple(sorted((e[i], e[j]))))] for i, j in itertools.combinations(range(3), 2)] + \
               [need[t["week"]][("and3", tuple(sorted(e)))]]
        if any(v is None for v in p.values()) or any(v is None for v in wkly):
            dropped += 1; continue
        t["p"] = p; t["wkly"] = wkly
        keep.append(t)
    tris = keep
    n = len(tris)
    weeks = sorted({t["week"] for t in tris})
    print(f"{n} triangles with every answer ({dropped} dropped)", flush=True)

    # ---- (a) bias and checks ---------------------------------------------------------------
    bias_rows = defaultdict(list)       # type -> [(p - y)]
    bias_neg = defaultdict(list)        # (type, negated literals in the statement) -> ...
    for t in tris:
        for f, v in t["forms"].items():
            d = min(max(t["p"][f], LO), HI) - v["label"]
            bias_rows[ftype(f)].append((d, t["p"][f], v["label"], t["week"]))
            bias_neg[(ftype(f), v["formula"].count("NOT"))].append((d, v["label"]))
    rng = np.random.default_rng(SEED)

    def boot_mean(x):
        x = np.asarray(x); idx = rng.integers(0, len(x), (B, len(x)))
        m = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    bias = {ty: {"n": len(bias_rows[ty]), "mean_stated": float(np.mean([r[1] for r in bias_rows[ty]])),
                 "yes_rate": float(np.mean([r[2] for r in bias_rows[ty]])),
                 "bias": boot_mean([r[0] for r in bias_rows[ty]])} for ty in TYPES}
    bias_by_neg = {ty: {k: {"n": len(bias_neg[(ty, k)]), "bias": float(np.mean([d for d, _ in bias_neg[(ty, k)]])),
                            "yes_rate": float(np.mean([y for _, y in bias_neg[(ty, k)]]))}
                        for k in range(4) if bias_neg[(ty, k)]} for ty in TYPES}

    chk = defaultdict(list)
    for t in tris:
        P = {f: min(max(v, LO), HI) for f, v in t["p"].items()}
        chk["complement_T"].append(P["mirT"] + P["T"] - 1)
        chk["complement_pair"].append(P["mirP"] + P[f"pair{t['mir_k']}"] - 1)
        chk["partition"].append(sum(P[f"count{c}"] for c in range(4)) - 1)
        for k in range(3):
            chk["additivity_pair-T-sib"].append(P[f"pair{k}"] - P["T"] - P[f"sib{k}"])
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            chk["single-2pairs+T"].append(P[f"single{i}"] - P[f"pair{j}"] - P[f"pair{k}"] + P["T"])
        viol = 0
        for k in range(3):
            viol += P["T"] > P[f"pair{k}"] + 1e-9
            viol += P[f"sib{k}"] > P[f"pair{k}"] + 1e-9
            for i in range(3):
                if i != k:
                    viol += P[f"pair{k}"] > P[f"single{i}"] + 1e-9
        chk["order_violation_share"].append(viol / 12)
        chk["wording_all3-T"].append(P["count3"] - P["T"])
        for k in range(3):
            if not t["signs"][k]:  # stated "k resolves NO" + weekly stated "k resolves YES" - 1
                chk["complement_single_NO+YES"].append(P[f"single{k}"] + min(max(t["wkly"][k], LO), HI) - 1)
    checks = {}
    for k, v in chk.items():
        v = np.asarray(v)
        c = {"n": len(v), "mean": boot_mean(v), "mean_abs": float(np.abs(v).mean()),
             "share_abs_gt_0.05": float((np.abs(v) > 0.05).mean())}
        if k == "single-2pairs+T":
            c["share_negative"] = float((v < -1e-9).mean())
        if k == "order_violation_share":
            c = {"n": len(v), "share_of_comparisons_violated": boot_mean(v),
                 "share_triangles_any": float((v > 0).mean())}
        checks[k] = c

    # ---- (b) fits ----------------------------------------------------------------------------
    by_week_type = defaultdict(list)
    for ty, rows in bias_rows.items():
        for d, _, _, wk in rows:
            by_week_type[(wk, ty)].append(d)

    def loo_bias(wk, ty):
        xs = [d for w in weeks if w != wk for d in by_week_type[(w, ty)]]
        return float(np.mean(xs))
    and_atoms = [pos_atoms("s", (k,)) for k in range(3)] + \
                [pos_atoms("p", (i, j)) for i, j in itertools.combinations(range(3), 2)] + [pos_atoms("t", (0, 1, 2))]
    M_and = masks(and_atoms)
    VARS = ["raw", "indep", "fitAND", "fit16", "fit16bc"] + list(GLOBAL)
    SETS = ["singles", "pos_singles3", "all16", "joints_balanced", "and_only7", "outcome8"]
    L = {v: {s: np.full(n, np.nan) for s in SETS} for v in VARS}
    fitted_single_shift = defaultdict(list)
    for r, t in enumerate(tris):
        forms = list(t["forms"])
        M = masks([t["forms"][f]["atoms"] for f in forms])
        p = np.array([t["p"][f] for f in forms])
        pbc = np.array([t["p"][f] - loo_bias(t["week"], ftype(f)) for f in forms])
        s = np.clip([t["p"][f"single{k}"] for k in range(3)], LO, HI)
        q_ind = np.array([np.prod([s[k] if int(x[k] == t["signs"][k]) else 1 - s[k] for k in range(3)])
                          for x in itertools.product((0, 1), repeat=3)])
        qs = {"indep": q_ind, "fitAND": project(t["wkly"], M_and), "fit16": project(p, M), "fit16bc": project(pbc, M)}
        qs = {v: (1 - EPS_MIX) * q + EPS_MIX / 8 for v, q in qs.items()}
        y = t["outcome"]
        lab = {f: t["forms"][f]["label"] for f in forms}
        and_lab = [int(y in at) for at in and_atoms]
        e = t["events"]
        and_keys = [("single", (e[k],)) for k in range(3)] + \
                   [("and2", tuple(sorted((e[i], e[j])))) for i, j in itertools.combinations(range(3), 2)] + \
                   [("and3", tuple(sorted(e)))]
        for v in GLOBAL:
            pa = [gpred[(t["week"], v)][need_qid[(t["week"], k)]] for k in and_keys]
            L[v]["and_only7"][r] = np.mean([ll(pr, yy) for pr, yy in zip(pa, and_lab)])
            L[v]["pos_singles3"][r] = np.mean([ll(pr, yy) for pr, yy in zip(pa[:3], and_lab[:3])])
        for v in VARS[:5]:
            if v == "raw":
                pred = {f: min(max(t["p"][f], LO), HI) for f in forms}
                pred_and = np.clip(t["wkly"], LO, HI)
            else:
                q = qs[v]
                pred = {f: float(q[t["forms"][f]["atoms"]].sum()) for f in forms}
                pred_and = M_and @ q
                L[v]["outcome8"][r] = -np.log(max(q[y], 1e-6))
                for k in range(3):
                    fitted_single_shift[v].append(pred[f"single{k}"] - min(max(t["p"][f"single{k}"], LO), HI))
            L[v]["singles"][r] = np.mean([ll(pred[f], lab[f]) for f in forms if f.startswith("single")])
            L[v]["all16"][r] = np.mean([ll(pred[f], lab[f]) for f in forms])
            L[v]["joints_balanced"][r] = np.mean([ll(pred[f], lab[f]) for f in forms if ftype(f) in ("T", "sib", "pair")])
            L[v]["and_only7"][r] = np.mean([ll(pr, yy) for pr, yy in zip(pred_and, and_lab)])
            L[v]["pos_singles3"][r] = np.mean([ll(pr, yy) for pr, yy in zip(pred_and[:3], and_lab[:3])])
    idx = rng.integers(0, n, (B, n))

    def ci(x):
        m = x[idx].mean(1)
        return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
    table = {}
    for s in SETS:
        table[s] = {}
        for v in VARS:
            x = L[v][s]
            if np.isnan(x).all():
                continue
            row = {"ll": ci(x)}
            for ref in ("raw", "indep", "fitAND", "fit16"):
                if ref != v and not np.isnan(L[ref][s]).all():
                    row[f"minus_{ref}"] = ci(x - L[ref][s])
            table[s][v] = row
    shift = {v: float(np.mean(x)) for v, x in fitted_single_shift.items()}
    summary = {"triangles": n, "dropped_missing": dropped, "weeks": weeks,
               "per_week": {w: sum(t["week"] == w for t in tris) for w in weeks},
               "bias_by_type": bias, "bias_by_type_and_negations": bias_by_neg, "checks": checks,
               "loo_bias_used": {w: {ty: loo_bias(w, ty) for ty in TYPES} for w in weeks},
               "logloss": table, "mean_fitted_minus_stated_single": shift,
               "notes": {"bias": "mean(stated - outcome); stated clipped to [0.01, 0.99]",
                         "outcome8": "-log q(realised outcome of the three events); raw has no joint distribution",
                         "scoring": "fitted/independence q mixed with 1% uniform; every binary prediction clipped to [0.01, 0.99]",
                         "bias_by_type_and_negations": "key = number of NOT literals in the statement as written",
                         "pos_singles3": "the 3 events' YES singles; raw = weekly stated YES answers",
                         "and_only7": "the weekly run's 3 positive singles, 3 positive pairs and positive triple; raw = weekly stated answers"}}
    (D / "summary.json").write_text(json.dumps(summary, indent=1))
    charts = {
        "bias_by_type": [{"type": ty, "bias": bias[ty]["bias"][0], "lo": bias[ty]["bias"][1], "hi": bias[ty]["bias"][2],
                          "mean_stated": bias[ty]["mean_stated"], "yes_rate": bias[ty]["yes_rate"], "n": bias[ty]["n"]}
                         for ty in TYPES],
        "bias_by_type_and_negations": [{"type": ty, "n_neg": k, **v} for ty in TYPES for k, v in bias_by_neg[ty].items()],
        "checks": [{"check": k, **v} for k, v in checks.items()],
        "logloss": [{"set": s, "variant": v, "ll": r["ll"][0], "lo": r["ll"][1], "hi": r["ll"][2]}
                    for s, rows in table.items() for v, r in rows.items()],
    }
    (D / "charts.json").write_text(json.dumps(charts, indent=1))

    print("bias (stated - outcome) by type:")
    for ty in TYPES:
        b = bias[ty]
        print(f"  {ty:7s} n={b['n']:5d} stated={b['mean_stated']:.3f} yes={b['yes_rate']:.3f} "
              f"bias={b['bias'][0]:+.3f} [{b['bias'][1]:+.3f},{b['bias'][2]:+.3f}]  by #NOT in statement: "
              + " ".join(f"{k}:{v['bias']:+.3f}(y{v['yes_rate']:.2f})" for k, v in bias_by_neg[ty].items()))
    print("checks:")
    for k, v in checks.items():
        print(f"  {k}: {json.dumps(v)}")
    print("log loss:")
    for s in SETS:
        for v, r in table[s].items():
            extra = " ".join(f"{k}={x[0]:+.4f}[{x[1]:+.4f},{x[2]:+.4f}]" for k, x in r.items() if k != "ll")
            print(f"  {s:16s} {v:8s} {r['ll'][0]:.4f} [{r['ll'][1]:.4f},{r['ll'][2]:.4f}] {extra}")
    print("mean fitted - stated single:", json.dumps(shift))


if __name__ == "__main__":
    main()
