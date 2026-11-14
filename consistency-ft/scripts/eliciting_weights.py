"""Eliciting projection weights: analysis on the 300-pair subset (CPU only).

Signals (docs/eliciting-weights-ideas-20260927.md), from scripts/run_eliciting_weights.py logs:
  rev       predicted revision "Change: c" (how far the model expects to move if it thought much longer)
  think     measured revision: (logit p_think - logit p_quick)^2, thinking on, budget 1,024
  critique  E[trust] (1-5, logprobs) when the model reviews all 12 of its quick answers together; also its
            revised probabilities
  evidence  E[directness] (1-5, logprobs), the 12 statements rated side by side without answers
  truth discovery (label-free): per-form noise variance from residuals to the (iterated) consensus; per-answer
            Huber reweighting (IRLS)
  stacking  w = exp(theta . x), theta fitted on the training folds by post-projection log loss (Powell)
Quick point: p_lp, the T=0 v6_direct answer from the confidence-weighting logprob run (same model, same prompts).

Every weighted arm is cross-fitted over 5 folds by pair (seed 20260924), tau/theta chosen on the 4 training folds by
all-12 log loss; paired bootstrap (2,000) CIs vs equal weights and vs the per-form weights.

Usage: eliciting_weights.py --log DIR/log.jsonl --out FILE.json [--no-stack] [--model-name 32B]
       [--quick-log .../step2-old/log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.confidence_weighting import (FI, K, TAUS, boot, cl, digit_signal, error_checks, folds_of,  # noqa: E402
                                          form_control, jl, load_dec, logit, pair_ll, partial_spearman, proj, sig)
from training.supervised_pairs import FORM_ORDER  # noqa: E402

QUICK = Path("/home/agent/data/confidence-weighting-20260927/step2-old/log.jsonl")
CHANGE = re.compile(r"Change:\s*\**\s*(\d*\.?\d+)\s*(%?)", re.I)
CRIT = re.compile(r"S(\d+)\s*:\s*trust\s*=\s*(\d)\D+?revised\s*=\s*(\d*\.?\d+)", re.I)
EVID = re.compile(r"S(\d+)\s*:\s*\**\s*(\d)")


def expected_digits(lps, after: re.Pattern):
    """Expected value (1-5) of each digit token whose preceding text matches `after`, from its top-5 logprobs."""
    out, text = [], ""
    for t in lps or []:
        tok = t["token"]
        if tok.strip() in {"1", "2", "3", "4", "5"} and after.search(text):
            d = {}
            for c in t["top_logprobs"]:
                k = c["token"].strip()
                if k in {"1", "2", "3", "4", "5"}:
                    d[int(k)] = d.get(int(k), 0) + np.exp(c["logprob"])
            z = sum(d.values())
            out.append(sum(k * v for k, v in d.items()) / z if z > 0 else float(tok))
        text += tok
    return out


def load_signals(log: Path, pids: list[str], quick_log: Path):
    idx = {p: i for i, p in enumerate(pids)}
    n = len(pids)
    M = {k: np.full((n, 12), np.nan) for k in ("p_lp", "v_digit", "p_rev", "change", "p_think", "think_tokens",
                                                "trust", "trust_stated", "p_revised", "evid", "evid_stated")}
    for r in jl(quick_log):
        if r["variant"] == "lp" and r["pair_id"] in idx:
            i, j = idx[r["pair_id"]], FI[r["form"]]
            M["p_lp"][i, j] = np.nan if r["p"] is None else r["p"]
            _, v, _ = digit_signal(r.get("logprobs_retry") if r.get("retried") else r.get("logprobs"))
            M["v_digit"][i, j] = np.nan if v is None else v
    for r in jl(log):
        if r["pair_id"] not in idx:
            continue
        i, v = idx[r["pair_id"]], r["variant"]
        if v == "lp":
            M["p_lp"][i, FI[r["form"]]] = np.nan if r["p"] is None else r["p"]
        elif v == "rev":
            j = FI[r["form"]]
            M["p_rev"][i, j] = np.nan if r["p"] is None else r["p"]
            m = CHANGE.search((r.get("raw_retry") if r.get("retried") else r["raw"]) or "")
            if m:
                c = float(m.group(1)) / (100 if m.group(2) or float(m.group(1)) > 1 else 1)
                M["change"][i, j] = min(c, 1.0)
        elif v == "think":
            j = FI[r["form"]]
            M["p_think"][i, j] = np.nan if r["p"] is None else r["p"]
            M["think_tokens"][i, j] = (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", np.nan)
        elif v in ("critique", "evidence"):
            order = r["order"]
            raw = r["raw"] or ""
            if v == "critique":
                found = {int(a): (int(b), float(c)) for a, b, c in CRIT.findall(raw)}
                ex = expected_digits(r["logprobs"], re.compile(r"trust\s*=\s*$"))
            else:
                found = {int(a): (int(b), None) for a, b in EVID.findall(raw)}
                ex = expected_digits(r["logprobs"], re.compile(r"S?\d+\s*:\s*$"))
            ok_ex = len(ex) == len(order) == len(found)
            for k, f in enumerate(order, 1):
                if k not in found:
                    continue
                j = FI[f]
                s, pr = found[k]
                if v == "critique":
                    M["trust_stated"][i, j] = s
                    M["trust"][i, j] = ex[k - 1] if ok_ex else s
                    M["p_revised"][i, j] = min(max(pr if pr <= 1 else pr / 100, 0.01), 0.99)
                else:
                    M["evid_stated"][i, j] = s
                    M["evid"][i, j] = ex[k - 1] if ok_ex else s
    return M


def fill(V):
    return np.where(np.isnan(V), np.nanmedian(V), V)


def crossfit(T, Y, cands):
    """cands: key -> weight matrix (n x 12).  tau/theta chosen on 4 training folds by all-12 LL."""
    n = len(T)
    ok = ~np.isnan(T)
    T0 = np.where(ok, T, 0.0)
    fold = folds_of(n)
    cache = {k: proj(T0, W * ok, "kl") for k, W in cands.items()}
    out = np.full((n, 12), np.nan)
    chosen = []
    for f in range(K):
        tr, te = fold != f, fold == f
        best = min(cache, key=lambda k: pair_ll(cache[k][tr], Y[tr])[:, 0].mean())
        chosen.append(str(best))
        out[te] = cache[best][te]
    return out, chosen


def inv_cands(V, Vf=None):
    """1/(V + tau) (and x 1/(Vf + t1) when Vf is given); 'eq' included so tau=inf stays possible."""
    Vn = fill(V)
    c = {"eq": np.ones_like(Vn)}
    for t in TAUS:
        if not np.isinf(t):
            c[f"a{t}"] = 1 / (Vn + t)
    if Vf is not None:
        for t1 in TAUS:
            for t2 in TAUS:
                if not (np.isinf(t1) and np.isinf(t2)):
                    w1 = 1 if np.isinf(t1) else 1 / (Vf + t1)
                    w2 = 1 if np.isinf(t2) else 1 / (Vn + t2)
                    c[f"f{t1}a{t2}"] = w1 * w2 * np.ones_like(Vn)
    return c


def truth_discovery(T, iters):
    """Label-free per-form noise variance: residuals to the weighted consensus, iterated."""
    ok = ~np.isnan(T)
    T0 = np.where(ok, T, 0.0)
    W = ok.astype(float)
    for _ in range(iters):
        fit = logit(proj(T0, W, "kl"))
        r2 = np.where(ok, (T0 - fit) ** 2, np.nan)
        s2 = np.nanmean(r2, 0)
        W = ok / s2[None, :]
    return np.broadcast_to(s2, T.shape).copy()


def huber_cands(T):
    ok = ~np.isnan(T)
    T0 = np.where(ok, T, 0.0)
    c = {"eq": np.ones_like(T0)}
    for cc in (0.25, 0.5, 1.0, 2.0):
        W = ok.astype(float)
        for _ in range(5):
            r = np.abs(T0 - logit(proj(T0, W, "kl")))
            W = ok * np.minimum(1, cc / np.maximum(r, 1e-9))
        c[f"huber{cc}"] = W
    return c


def stack(T, Y, feats, lam=0.01, maxfev=300):
    """w = exp(form effect + theta . x); theta fitted on the 4 training folds (Powell), applied to the held-out fold."""
    n = len(T)
    ok = ~np.isnan(T)
    T0 = np.where(ok, T, 0.0)
    X = np.stack([fill(f) for f in feats], -1) if feats else np.zeros((n, 12, 0))
    X = (X - X.reshape(-1, X.shape[-1]).mean(0)) / (X.reshape(-1, X.shape[-1]).std(0) + 1e-9) if feats else X
    fold = folds_of(n)
    out = np.full((n, 12), np.nan)
    thetas = []

    def W_of(th, rows):
        fe = np.concatenate([[0.0], th[:11]])
        z = fe[None, :] + (X[rows] @ th[11:] if X.shape[-1] else 0)
        return np.exp(np.clip(z, -6, 6)) * ok[rows]

    for f in range(K):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]

        def obj(th):
            return pair_ll(proj(T0[tr], W_of(th, tr), "kl"), Y[tr])[:, 0].mean() + lam * (th ** 2).mean()
        res = minimize(obj, np.zeros(11 + X.shape[-1]), method="Powell",
                       options={"maxfev": maxfev, "xtol": 1e-3, "ftol": 1e-6})
        thetas.append(res.x.round(3).tolist())
        out[te] = proj(T0[te], W_of(res.x, te), "kl")
    return out, thetas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--quick-log", type=Path, default=QUICK)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--think-subset", action="store_true", help="only pairs with all 12 thinking answers")
    a = ap.parse_args()
    pids_all, Y_all, P, sub = load_dec()
    ii = np.where(sub)[0]
    pids = [pids_all[i] for i in ii]
    Y = Y_all[ii]
    S5 = np.stack([logit(P[f"s{k}"][ii]) for k in range(5)], 0)
    W3 = np.stack([logit(P[f"w{k}"][ii]) for k in range(1, 4)], 0)
    v_samp, v_word = S5.var(0, ddof=1), W3.var(0, ddof=1)
    M = load_signals(a.log, pids, a.quick_log)
    if a.think_subset:
        k = np.where((~np.isnan(M["p_think"])).sum(1) == 12)[0]
        M = {kk: v[k] for kk, v in M.items()}
        Y, v_samp, v_word, pids = Y[k], v_samp[k], v_word[k], [pids[i] for i in k]
        print(f"think subset: {len(k)} pairs", flush=True)
    have = {k: float((~np.isnan(v)).mean()) for k, v in M.items()}
    print("coverage", {k: round(v, 3) for k, v in have.items()}, flush=True)
    T = logit(M["p_lp"])
    n = len(T)
    fitT = logit(proj(np.nan_to_num(T), (~np.isnan(T)).astype(float), "kl"))
    Vf = form_control(v_samp, n)
    p = sig(T)
    sig_V = {}
    if have["change"] > 0.5:
        sig_V["rev: change^2"] = M["change"] ** 2
        sig_V["rev: (change / p(1-p))^2 (log-odds scale)"] = (M["change"] / (cl(p) * (1 - cl(p)))) ** 2
    if have["p_think"] > 0.5:
        sig_V["think: (logit think - logit quick)^2"] = (logit(M["p_think"]) - T) ** 2
    if have["trust"] > 0.5:
        sig_V["critique: 5 - E[trust]"] = 5 - M["trust"]
        sig_V["critique: (logit revised - logit quick)^2"] = (logit(M["p_revised"]) - T) ** 2
    if have["evid"] > 0.5:
        sig_V["evidence: 5 - E[directness]"] = 5 - M["evid"]
    out = {"description": __doc__, "coverage": have, "error_checks": {}, "arms": {}, "extra": {}}

    # ---- per-answer error prediction
    for name, V in sig_V.items():
        r = error_checks(T, V, Y, fitT)
        out["error_checks"][name] = r
        print(f"[err] {name:45s} fit {r['fit_error']['pooled']:+.3f}/{r['fit_error']['partial_on_extremeness']:+.3f}"
              f"  out {r['outcome_logloss']['pooled']:+.3f}/{r['outcome_logloss']['within_form_mean']:+.3f}/"
              f"{r['outcome_logloss']['partial_on_extremeness']:+.3f}  rho_ext {r['rho_with_extremeness']:+.3f}", flush=True)

    # ---- (a) predicts (b)?
    if "rev: change^2" in sig_V and "think: (logit think - logit quick)^2" in sig_V:
        c = M["change"]
        d = np.abs(M["p_think"] - M["p_lp"])
        dl = np.abs(logit(M["p_think"]) - T)
        m = ~np.isnan(c) & ~np.isnan(d)
        wf = [spearmanr(c[m[:, j], j], d[m[:, j], j])[0] for j in range(12) if np.std(c[m[:, j], j]) > 0]
        out["extra"]["predicted_vs_measured_revision"] = {
            "n": int(m.sum()), "spearman_prob": float(spearmanr(c[m], d[m])[0]),
            "spearman_logit": float(spearmanr(c[m], dl[m])[0]), "within_form_mean": float(np.mean(wf)),
            "partial_on_extremeness": partial_spearman(c[m], d[m], np.abs(T[m])),
            "mean_predicted": float(np.nanmean(c)), "mean_measured_abs": float(np.nanmean(d)),
            "share_moved_gt_0.05": float((d[m] > 0.05).mean()), "median_think_tokens": float(np.nanmedian(M["think_tokens"])),
            "share_think_budget_hit": float((M["think_tokens"][~np.isnan(M["think_tokens"])] >= 1000).mean())}
        cp = np.nanmean(c, 1); dp = np.nanmean(d, 1)
        out["extra"]["predicted_vs_measured_revision"]["per_pair_spearman"] = float(spearmanr(cp, dp, nan_policy="omit")[0])
        print("[a->b]", out["extra"]["predicted_vs_measured_revision"], flush=True)
        # per-form means: does predicted change rank forms like measured change?
        out["extra"]["per_form_means"] = {f: {"predicted_change": float(np.nanmean(c[:, j])),
                                              "measured_abs_change": float(np.nanmean(d[:, j])),
                                              "samp_var": float(np.nanmean(v_samp[:, j]))} for j, f in enumerate(FORM_ORDER)}

    # ---- raw points
    raw = {"quick (lp)": M["p_lp"]}
    for k, lab in (("p_rev", "rev own answer"), ("p_think", "think answer"), ("p_revised", "critique revised")):
        if have[k] > 0.5:
            raw[lab] = M[k]
    if have["p_think"] > 0.5:
        raw["mean(quick, think) log-odds"] = sig((T + logit(M["p_think"])) / 2)
    base_raw = pair_ll(M["p_lp"], Y)
    out["raw_ll"] = {k: {"all12": float(pair_ll(v, Y)[:, 0].mean()), "vs_quick": boot(pair_ll(v, Y) - base_raw)[0]}
                     for k, v in raw.items()}
    for k, v in out["raw_ll"].items():
        print(f"[raw] {k:30s} {v['all12']:.4f}  vs quick {v['vs_quick'][0]:+.4f} [{v['vs_quick'][1]:+.4f}, {v['vs_quick'][2]:+.4f}]",
              flush=True)

    # ---- projections
    L, info = {}, {}

    def add(name, pred, chosen=None):
        L[name] = pair_ll(pred, Y)
        info[name] = chosen

    Tn = np.nan_to_num(T)
    okw = (~np.isnan(T)).astype(float)
    add("eq", proj(Tn, okw, "kl"))
    add("form (per-form sampling var)", *crossfit(T, Y, inv_cands(Vf)))
    for name, V in sig_V.items():
        add(name, *crossfit(T, Y, inv_cands(V)))
        add(name + " x form", *crossfit(T, Y, inv_cands(V, Vf)))
    for k, lab in (("p_think", "think answer"), ("p_revised", "critique revised"), ("p_rev", "rev own answer")):
        if have[k] > 0.5:
            Tk = logit(M[k])
            add(f"point: {lab}, eq", proj(np.nan_to_num(Tk), (~np.isnan(Tk)).astype(float), "kl"))
    if have["p_think"] > 0.5:
        Tm = np.where(np.isnan(M["p_think"]), T, (T + logit(M["p_think"])) / 2)
        add("point: mean(quick, think), eq", proj(np.nan_to_num(Tm), okw, "kl"))
        add("point: mean(quick, think), form", *crossfit(Tm, Y, inv_cands(Vf)))
        kth = "think: (logit think - logit quick)^2"
        if kth in sig_V:
            add("point: mean(quick, think), measured-revision weights", *crossfit(Tm, Y, inv_cands(sig_V[kth])))
            Tk = np.where(np.isnan(M["p_think"]), T, logit(M["p_think"]))
            add("point: think answer, measured-revision weights", *crossfit(Tk, Y, inv_cands(sig_V[kth])))
    if have["p_revised"] > 0.5:
        Tm = np.where(np.isnan(M["p_revised"]), T, (T + logit(M["p_revised"])) / 2)
        add("point: mean(quick, critique revised), eq", proj(np.nan_to_num(Tm), okw, "kl"))
    for it in (1, 5):
        add(f"truth discovery: per-form residual var ({it} iter)", *crossfit(T, Y, inv_cands(truth_discovery(T, it))))
    add("truth discovery: Huber per-answer", *crossfit(T, Y, huber_cands(T)))
    # ---- oracle headroom (uses test labels; an upper bound, not a method): per pair, the best of 25 weight vectors
    #      (equal; each form x5; each form x0.2)
    cand = [np.ones(12)]
    for j in range(12):
        for m in (5.0, 0.2):
            w = np.ones(12); w[j] = m; cand.append(w)
    LL = np.stack([pair_ll(proj(Tn, okw * w[None, :], "kl"), Y)[:, 0] for w in cand], 1)
    out["extra"]["oracle_headroom"] = {"eq": float(LL[:, 0].mean()), "per_pair_best_of_25": float(LL.min(1).mean()),
                                       "best_single_vector_insample": float(LL.mean(0).min())}
    print("[oracle]", out["extra"]["oracle_headroom"], flush=True)
    if not a.no_stack:
        free = [np.abs(T), np.log(v_samp + 0.01), np.log(v_word + 0.01), np.log(fill(M["v_digit"]) + 0.01),
                np.abs(T - fitT)]
        paid = [np.log(fill(V) + 0.01) for k, V in sig_V.items() if "log-odds scale" not in k]
        specs = {"stacking: form only": [], "stacking: form + free features": free}
        tk = "think: (logit think - logit quick)^2"
        if tk in sig_V:
            specs["stacking: form + measured revision"] = [np.log(fill(sig_V[tk]) + 0.01)]
        if paid:
            specs["stacking: form + free + elicited"] = free + paid
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(len(specs)) as ex:
            futs = {k: ex.submit(stack, T, Y, f) for k, f in specs.items()}
            for k, fu in futs.items():
                add(k, *fu.result())
    for name in L:
        r = {"all12_ll": float(L[name][:, 0].mean()), "andor_ll": float(L[name][:, 1].mean()), "chosen": info[name]}
        if name != "eq":
            r["vs_eq"] = boot(L[name] - L["eq"])[0]
            if name != "form (per-form sampling var)":
                r["vs_form"] = boot(L[name] - L["form (per-form sampling var)"])[0]
        out["arms"][name] = r
        ve, vf = r.get("vs_eq"), r.get("vs_form")
        print(f"[proj] {name:55s} {r['all12_ll']:.4f}" + (f"  vs eq {ve[0]:+.4f} [{ve[1]:+.4f}, {ve[2]:+.4f}]" if ve else "")
              + (f"  vs form {vf[0]:+.4f} [{vf[1]:+.4f}, {vf[2]:+.4f}]" if vf else ""), flush=True)
        a.out.write_text(json.dumps(out, indent=1, default=float))
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
