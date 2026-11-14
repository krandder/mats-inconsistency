"""Better coherence projections, cross-fitted (CPU only, no API calls).

Finding being fixed (docs/structured-pairs-20260924.md): the raw one-form answers respect known
relations, but the equal-weight KL projection over all 12 forms washes the dependence out.
Hypothesis: systematically biased forms (AND too high, OR too low;
docs/averaging-vs-structure-20260924.md) and noisy forms overwhelm the informative ones.

Methods (model-only; T = logit(clip(stated, 0.01)), missing forms get weight 0):
  raw          stated answers (missing scored 0.5)
  eq_kl        equal-weight KL projection (the current method; == coherence_projection.fit "kl")
  eq_lsl       equal-weight log-odds least squares
  bc_kl        bias-corrected: T - b, b = per-form log-odds bias, then eq KL
  rw_kl        bias-corrected + reliability weights w_f = 1 / sigma_f^2 (KL)
  rw_lsl       the same with log-odds least squares
  rwg_kl       weights w_f^gamma, gamma in {0, .5, 1, 2, 3} tuned on the training folds (joint LL)
  fill_<src>   dependence-preserving: marginals of rw_kl + log odds ratio from <src>, scaled by
               lambda (tuned on the training folds, grid 0..3 step 0.1; lambda = 1 is "no shrink")
     src and3  raw P, Q, P-and-Q (AND clipped into its Frechet range)       (hybrid_and3's source)
     src and4  the 4 bias-corrected AND answers, normalized to sum 1
     src litand  rw_kl projection fitted on the 4 literals + 4 ANDs only
Market variants (structured pairs and market-context only): market marginals (price at t, clip
0.01; normalized for mc_excl) + a log odds ratio:
  h_eq, h_bc, h_rw, h_rwg        from the corresponding projection (h_eq == the old "hybrid")
  h_and3, h_and4, h_litand       from the DP sources, lambda = 1 (h_and3 == the old "hybrid_and3")
  h_and3_s, h_and4_s, h_litand_s the same with lambda tuned on the training folds (hybrid joint LL)
Baselines: market_indep; market_known (structured only: min(a, b) for implication,
max(0, a + b - 1) for exclusivity, independence for controls).

Parameter fitting.  b and sigma^2 are LABEL-FREE: iterate b <- b + mean residual of the
log-odds LS projection of T - b (4 rounds), sigma_f^2 = mean squared residual.  gamma and lambda
use labels.  Every parameter is fitted on the other 4 of 5 folds (folds by cluster: ladder group /
MC market / control pair for structured pairs, pair otherwise) and applied to the held-out fold:
cross-fitted, never tuned on the pairs it scores.  "x" variants (bcx_kl, rwx_kl, h_bcx, h_rwx) use
b and sigma^2 estimated once on a different data set: Qwen3-32B's t0 answers on the old 1,066-pair
test (the averaging-vs-structure run); they are not reported on that data set itself.

Metrics: mean per-pair log loss over the joint forms (8 AND/OR) and all 12, clip 0.01.  CIs:
paired cluster bootstrap, 2,000 resamples, seed 20260924.  Recovery = (market_indep - method) /
(market_indep - market_known) on the joint forms.

Usage: weighted_projection.py [--out-json /home/agent/data/weighted-projection-20260924/full.json]
       then scripts/tables_weighted_projection.py (tables + docs/weighted-projection-20260924-charts.json)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.batch_projection import A, joint_with_or_batch, log_or, project  # noqa: E402
from scripts.score_structured_pairs import dep_captured  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

LO = 0.01
SEED = 20260924
B = 2000
K = 5
SINGLE = ("P", "NOT P", "Q", "NOT Q")
JI = np.array([i for i, f in enumerate(FORM_ORDER) if f not in SINGLE])
AND_F = ("(P) AND (Q)", "(P) AND (NOT Q)", "(NOT P) AND (Q)", "(NOT P) AND (NOT Q)")  # atom order
AND_I = np.array([FORM_ORDER.index(f) for f in AND_F])
LITAND = np.array([f in SINGLE or f in AND_F for f in FORM_ORDER], float)
iP, iQ = FORM_ORDER.index("P"), FORM_ORDER.index("Q")
GAMMAS = (0.0, 0.5, 1.0, 2.0, 3.0)
LAMBDAS = tuple(np.round(np.arange(0, 3.01, 0.1), 2))
SP = Path("/home/agent/data/structured-pairs-20260924")
MC = Path("/home/agent/data/jev-market-context-open-20260923/test")
MCL = Path("/home/agent/data/market-context-zeroshot-20260924/logs")
OLD = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919/test")


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def cl(x):
    return np.clip(x, LO, 1 - LO)


def logit(x):
    return np.log(x / (1 - x))


def forms(atoms):
    return atoms @ A.T


def atoms_from(a, b, p11):
    return np.stack([p11, a - p11, b - p11, 1 - a - b + p11], 1)


def ll(P, Y):
    """per pair: (joint LL, all12 LL); P may contain nan (scored 0.5)."""
    p = cl(np.where(np.isnan(P), 0.5, P))
    L = -(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return np.stack([L[:, JI].mean(1), L.mean(1)], 1)


# ------------------------------------------------------------------ data

def job_from_stated(name, pids, byp, labels, extra):
    S = np.array([[byp[p].get(f) if byp[p].get(f) is not None else np.nan for f in FORM_ORDER] for p in pids], float)
    Y = np.array([[labels[p][f] for f in FORM_ORDER] for p in pids], float)
    return {"name": name, "pids": pids, "S": S, "Y": Y, **extra}


def load_structured():
    pairs = {p["pair_id"]: p for p in jl(SP / "pairs.jsonl")}
    lab = {r["pair_id"]: (r["p"], r["q"]) for r in jl(SP / "labels.jsonl")}
    labels = {}
    for pid, (yp, yq) in lab.items():
        cell = [(1, 1), (1, 0), (0, 1), (0, 0)].index((yp, yq))
        labels[pid] = {f: A[FORM_ORDER.index(f), cell] for f in FORM_ORDER}
    jobs = []
    for log in sorted((SP / "logs").glob("*-structured.jsonl")):
        recs = jl(log)
        byp = defaultdict(dict)
        for r in recs:
            byp[r["pair_id"]][r["form"]] = r["p"]
        pids = sorted(p for p, v in byp.items() if len(v) == 12)
        ps = [pairs[p] for p in pids]
        a = cl(np.array([p["p_price_norm"] if p["type"] == "mc_excl" else p["p_price_t"] for p in ps]))
        b = cl(np.array([p["q_price_norm"] if p["type"] == "mc_excl" else p["q_price_t"] for p in ps]))
        typ = np.array([p["type"] for p in ps])
        p11 = np.where(np.isin(typ, ["threshold", "date"]), np.minimum(a, b),
                       np.where(typ == "mc_excl", np.maximum(0, a + b - 1), a * b))
        m = recs[0]["model"].replace("qwen3-235b-a22b", "235B").replace("qwen3-32b", "32B")
        jobs.append(job_from_stated(f"{m} {recs[0]['arm']}", pids, byp, labels, {
            "dataset": "structured", "cluster": np.array([p["group"] for p in ps]), "strata": typ,
            "ma": a, "mb": b, "known": forms(atoms_from(a, b, p11)), "pairs_meta": ps}))
    return jobs


def load_market_context():
    qs = {q["query_id"]: q for q in jl(MC / "queries.jsonl")}
    labels = defaultdict(dict)
    for r in jl(MC / "full-label-targets.jsonl"):
        q = qs[r["query_id"]]
        labels[q["pair_id"]][q["form"]] = r["label"]
    strata = {r["pair_id"]: r["stratum"] for r in jl(MC / "pair-outcomes.jsonl")}
    prices = {p["pair_id"]: (p["p_price_t"], p["q_price_t"]) for p in jl(MC / "pairs.jsonl")}
    jobs = []
    for log in sorted(MCL.glob("*-test.jsonl")):
        recs = jl(log)
        byp = defaultdict(dict)
        for r in recs:
            byp[r["pair_id"]][r["form"]] = r["p"]
        pids = sorted(p for p, v in byp.items() if len(v) == 12)
        m = recs[0]["model"].replace("qwen3-235b-a22b", "235B").replace("qwen3-32b", "32B")
        jobs.append(job_from_stated(f"{m} {recs[0]['arm']}", pids, byp, labels, {
            "dataset": "market_context", "cluster": np.array(pids), "strata": np.array([strata[p] for p in pids]),
            "ma": cl(np.array([prices[p][0] for p in pids])), "mb": cl(np.array([prices[p][1] for p in pids]))}))
    return jobs


def load_old():
    qs = {q["query_id"]: q for q in jl(OLD / "queries.jsonl")}
    labels = defaultdict(dict)
    for r in jl(OLD / "full-label-targets.jsonl"):
        q = qs[r["query_id"]]
        labels[q["pair_id"]][q["form"]] = r["label"]
    jobs = []
    for name, path, filt in (
            ("32B Q (DashScope t0)", "/home/agent/data/decomposition-32b-20260924/log.jsonl", lambda r: r["sample"] == "t0"),
            ("235B Q", "/home/agent/data/precutoff-baseline-20260923/per-form/predictions-test.jsonl", lambda r: True)):
        byp = defaultdict(dict)
        for r in jl(path):
            if filt(r):
                byp[r["pair_id"]][r["form"]] = r["p"]
        pids = sorted(p for p, v in byp.items() if len(v) == 12)
        jobs.append(job_from_stated(name, pids, byp, labels, {
            "dataset": "old_1066", "cluster": np.array(pids), "strata": np.array(["all"] * len(pids))}))
    return jobs


# ------------------------------------------------------------------ fitting

def prep(job):
    S = job["S"]
    W0 = (~np.isnan(S)).astype(float)
    T = logit(cl(np.nan_to_num(S, nan=0.5)))
    job["W0"], job["T"] = W0, T
    return job


def est_bias_var(T, W0, rounds=4):
    b = np.zeros(12)
    for _ in range(rounds):
        q = project(T - b, W0, "lsl")
        m = np.clip(forms(q), 1e-6, 1 - 1e-6)
        e = (T - b) - logit(m)
        b = b + (e * W0).sum(0) / W0.sum(0)
    q = project(T - b, W0, "lsl")
    e = (T - b) - logit(np.clip(forms(q), 1e-6, 1 - 1e-6))
    s2 = (e ** 2 * W0).sum(0) / W0.sum(0)
    return b, s2


def weights(s2, gamma=1.0):
    w = (1 / s2) ** gamma
    return w / w.mean()


def or_and3(S):
    sp, sq, sa = cl(S[:, iP]), cl(S[:, iQ]), S[:, FORM_ORDER.index("(P) AND (Q)")]
    s11 = np.minimum(np.minimum(sp, sq), np.maximum(np.maximum(0, sp + sq - 1), sa))
    return log_or(atoms_from(sp, sq, s11)), np.isnan(sa) | np.isnan(S[:, iP]) | np.isnan(S[:, iQ])


def or_and4(T, W0, b):
    s = 1 / (1 + np.exp(-(T[:, AND_I] - b[AND_I])))
    q = s / s.sum(1, keepdims=True)
    return log_or(q), (W0[:, AND_I] == 0).any(1)


def marg(atoms):
    return cl(atoms[:, 0] + atoms[:, 1]), cl(atoms[:, 0] + atoms[:, 2])


def with_or(a, b, lt):
    return forms(joint_with_or_batch(a, b, lt))


def pair_ll_joint(P, Y):
    return ll(P, Y)[:, 0].mean()


def fit_params(job, idx, ext):
    """All parameters from the rows idx (training folds)."""
    T, W0, S, Y = job["T"][idx], job["W0"][idx], job["S"][idx], job["Y"][idx]
    b, s2 = est_bias_var(T, W0)
    par = {"b": b, "s2": s2}
    # gamma (labels): joint LL of the weighted KL projection
    best = None
    for g in GAMMAS:
        q = project(T - b, W0 * weights(s2, g), "kl")
        v = pair_ll_joint(forms(q), Y)
        if best is None or v < best[0]:
            best = (v, g)
    par["gamma"] = best[1]
    # lambda per DP source, model fill and market hybrid (labels)
    q_rw = project(T - b, W0 * weights(s2), "kl")
    pa, pb = marg(q_rw)
    srcs = dp_sources(T, W0, S, b, s2, q_rw)
    par["lam_fill"], par["lam_h"] = {}, {}
    for name, lt in srcs.items():
        par["lam_fill"][name] = min(LAMBDAS, key=lambda l: pair_ll_joint(with_or(pa, pb, l * lt), Y))
        if "ma" in job:
            ma, mb = job["ma"][idx], job["mb"][idx]
            par["lam_h"][name] = min(LAMBDAS, key=lambda l: pair_ll_joint(with_or(ma, mb, l * lt), Y))
    return par


def dp_sources(T, W0, S, b, s2, q_rw):
    lt_rw = log_or(q_rw)
    l3, miss3 = or_and3(S)
    l4, miss4 = or_and4(T, W0, b)
    q_la = project(T - b, W0 * weights(s2) * LITAND, "kl")
    return {"and3": np.where(miss3, lt_rw, l3), "and4": np.where(miss4, lt_rw, l4), "litand": log_or(q_la)}


def apply(job, idx, par, ext):
    T, W0, S = job["T"][idx], job["W0"][idx], job["S"][idx]
    b, s2 = par["b"], par["s2"]
    out, lts = {}, {}
    q = {"eq_kl": project(T, W0, "kl"), "eq_lsl": project(T, W0, "lsl"), "bc_kl": project(T - b, W0, "kl"),
         "rw_kl": project(T - b, W0 * weights(s2), "kl"), "rw_lsl": project(T - b, W0 * weights(s2), "lsl"),
         "rwg_kl": project(T - b, W0 * weights(s2, par["gamma"]), "kl")}
    if ext is not None and job["name"] != ext["skip"]:
        q["bcx_kl"] = project(T - ext["b"], W0, "kl")
        q["rwx_kl"] = project(T - ext["b"], W0 * weights(ext["s2"]), "kl")
    out["raw"] = S
    for k, v in q.items():
        out[k] = forms(v)
        lts[k] = log_or(v)
    pa, pb = marg(q["rw_kl"])
    srcs = dp_sources(T, W0, S, b, s2, q["rw_kl"])
    for name, lt in srcs.items():
        out[f"fill_{name}"] = with_or(pa, pb, par["lam_fill"][name] * lt)
        lts[f"fill_{name}"] = par["lam_fill"][name] * lt
    if "ma" in job:
        ma, mb = job["ma"][idx], job["mb"][idx]
        out["market_indep"] = forms(atoms_from(ma, mb, ma * mb))
        if "known" in job:
            out["market_known"] = job["known"][idx]
        for k, tag in (("eq_kl", "h_eq"), ("bc_kl", "h_bc"), ("rw_kl", "h_rw"), ("rwg_kl", "h_rwg"),
                       ("bcx_kl", "h_bcx"), ("rwx_kl", "h_rwx")):
            if k in lts:
                out[tag] = with_or(ma, mb, lts[k])
        for name, lt in srcs.items():
            out[f"h_{name}"] = with_or(ma, mb, lt)
            out[f"h_{name}_s"] = with_or(ma, mb, par["lam_h"][name] * lt)
    return out, lts


def crossfit(job, ext):
    n = len(job["pids"])
    cls = sorted(set(job["cluster"]))
    rng = np.random.default_rng(SEED)
    fold_of_cl = dict(zip(cls, rng.permutation(len(cls)) % K))
    fold = np.array([fold_of_cl[c] for c in job["cluster"]])
    preds, lts, pars = {}, {}, []
    for k in range(K):
        te, tr = np.where(fold == k)[0], np.where(fold != k)[0]
        par = fit_params(job, tr, ext)
        pars.append(par)
        o, l = apply(job, te, par, ext)
        for m, v in o.items():
            preds.setdefault(m, np.full((n, 12), np.nan))[te] = v
        for m, v in l.items():
            lts.setdefault(m, np.full(n, np.nan))[te] = v
    return preds, lts, pars


# ------------------------------------------------------------------ scoring

def boot(D, clusters, seed=SEED):
    """D: n x k per-pair differences. Cluster bootstrap of the mean over pairs."""
    cls = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(cls)}
    ci = np.array([pos[c] for c in clusters])
    Sm = np.zeros((len(cls), D.shape[1])); Nc = np.zeros(len(cls))
    np.add.at(Sm, ci, D); np.add.at(Nc, ci, 1)
    rng = np.random.default_rng(seed)
    dr = rng.integers(0, len(cls), size=(B, len(cls)))
    cnt = np.zeros((B, len(cls)))
    for r in range(B):
        cnt[r] = np.bincount(dr[r], minlength=len(cls))
    num = cnt @ Sm; den = cnt @ Nc
    d = num / den[:, None]
    return [[float(D[:, j].mean()), float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))]
            for j in range(D.shape[1])]


def dep_stats(job, lts_or_forms, idx):
    """median D and share D >= 0.5 for structured implication / exclusivity pairs, from 12-form preds."""
    out = {}
    for t in ("threshold", "date", "mc_excl"):
        ii = [i for i in idx if job["strata"][i] == t]
        ds = []
        for i in ii:
            x = dict(zip(FORM_ORDER, lts_or_forms[i]))
            p = job["pairs_meta"][i]
            if t == "mc_excl":
                a, b_, ab = x["P"], x["Q"], x["(P) AND (Q)"]
            elif p["relation"].get("implies") == "Q->P":
                a, b_, ab = x["Q"], x["P"], x["(P) AND (Q)"]
            else:
                a, b_, ab = x["P"], x["Q"], x["(P) AND (Q)"]
            d = dep_captured(t, a, b_, ab)
            if d is not None:
                ds.append(d)
        out[t] = {"median": float(np.median(ds)), "share_ge_half": float(np.mean(np.array(ds) >= 0.5)), "n": len(ds)}
    return out


def score_job(job, preds, lts, pars):
    Y = job["Y"]
    L = {m: ll(P, Y) for m, P in preds.items()}
    strata = ["all"] + sorted(set(job["strata"]) - {"all"})
    res = {"name": job["name"], "dataset": job["dataset"], "pairs": len(job["pids"]), "by": {}}
    refs = ["raw", "eq_kl"] + [r for r in ("market_indep", "market_known", "h_eq") if r in L]
    for s in strata:
        idx = np.arange(len(Y)) if s == "all" else np.where(job["strata"] == s)[0]
        cls = job["cluster"][idx]
        e = {"pairs": int(len(idx)), "ll": {m: {"joint": float(L[m][idx, 0].mean()), "all12": float(L[m][idx, 1].mean())}
                                            for m in L}, "vs": {}}
        for r in refs:
            e["vs"][r] = {}
            for m in L:
                if m == r:
                    continue
                bj = boot(L[m][idx] - L[r][idx], cls)
                e["vs"][r][m] = {"joint": bj[0], "all12": bj[1]}
        if "market_known" in L and s in ("threshold", "date", "mc_excl", "all"):
            gap = L["market_indep"][idx, 0].mean() - L["market_known"][idx, 0].mean()
            if gap > 1e-9:
                e["recovery"] = {m: float((L["market_indep"][idx, 0].mean() - L[m][idx, 0].mean()) / gap) for m in L}
        res["by"][s] = e
    if job["dataset"] == "structured":
        idx = list(range(len(Y)))
        res["dependence_captured"] = {m: dep_stats(job, preds[m], idx)
                                      for m in ("raw", "eq_kl", "bc_kl", "rw_kl", "rwg_kl", "fill_and3", "fill_and4",
                                                "fill_litand")}
        ctrl = job["strata"] == "control"
        res["control_log_or"] = {m: {"median": float(np.median(v[ctrl])), "share_abs_gt_1": float((abs(v[ctrl]) > 1).mean())}
                                 for m, v in lts.items()}
    res["params"] = {
        "bias_mean": dict(zip(FORM_ORDER, np.mean([p["b"] for p in pars], 0).round(3).tolist())),
        "sigma2_mean": dict(zip(FORM_ORDER, np.mean([p["s2"] for p in pars], 0).round(3).tolist())),
        "weights_mean": dict(zip(FORM_ORDER, np.mean([weights(p["s2"]) for p in pars], 0).round(3).tolist())),
        "gamma_by_fold": [p["gamma"] for p in pars],
        "lambda_fill_by_fold": {k: [p["lam_fill"][k] for p in pars] for k in pars[0]["lam_fill"]},
        "lambda_hybrid_by_fold": {k: [p["lam_h"][k] for p in pars] for k in pars[0]["lam_h"]}}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=Path("/home/agent/data/weighted-projection-20260924/full.json"))
    ap.add_argument("--only", default=None, help="dataset filter for quick runs")
    a = ap.parse_args()
    old = [prep(j) for j in load_old()]
    ext_job = old[0]
    b_ext, s2_ext = est_bias_var(ext_job["T"], ext_job["W0"])
    ext = {"b": b_ext, "s2": s2_ext, "skip": ext_job["name"]}
    print("external bias (32B old test):", dict(zip(FORM_ORDER, b_ext.round(3))), flush=True)
    print("external sigma2:", dict(zip(FORM_ORDER, s2_ext.round(3))), flush=True)
    jobs = load_structured() + load_market_context() + old
    results = {"description": __doc__, "external_params": {
        "source": "Qwen3-32B DashScope t0, old 1,066-pair test (docs/averaging-vs-structure-20260924.md)",
        "bias": dict(zip(FORM_ORDER, b_ext.round(4).tolist())), "sigma2": dict(zip(FORM_ORDER, s2_ext.round(4).tolist()))},
        "jobs": []}
    for job in jobs:
        if a.only and job["dataset"] != a.only:
            continue
        prep(job)
        preds, lts, pars = crossfit(job, ext)
        r = score_job(job, preds, lts, pars)
        results["jobs"].append(r)
        e = r["by"]["all"]
        line = " ".join(f"{m} {e['ll'][m]['joint']:.3f}" for m in ("raw", "eq_kl", "bc_kl", "rw_kl", "rwg_kl", "fill_and4",
                                                                   "fill_litand", "market_indep", "market_known", "h_eq",
                                                                   "h_rw", "h_and3", "h_and4_s", "h_litand_s") if m in e["ll"])
        print(f"[{job['dataset']}] {job['name']}: {line}", flush=True)
        a.out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
