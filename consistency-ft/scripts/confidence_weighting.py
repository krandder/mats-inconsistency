"""Per-answer confidence weighting of the coherence projection (CPU only).

Question (Kelvin, 27 Sep 2026): the projection weights all 12 answers equally, but the model may
know some answers well and guess on others.  Can a per-answer confidence signal (i) predict the
answer's error and (ii) improve accuracy when the projection is weighted by it?

Data: Qwen3-32B on the old 1,066-pair test, question only (docs/averaging-vs-structure-20260924.md):
t0 (temperature 0), s0..s4 (temperature 0.7), and on the 300-pair subset w1..w3 (paraphrases).
Step-2 signals (verbalized second-order confidence, digit logprobs, P(YES) token) are added by
--step2 DIR (the log of scripts/run_confidence_signals.py).

Signals (per answer, a variance v on the log-odds scale; larger = less confident):
  samp      var of logit over the 5 samples s0..s4 (or the samples not used for the point)
  word      var of logit over w1..w3 (subset)
  sampword  var over s0..s4 and w1..w3 (subset)
  form      control: the cross-fitted mean samp variance of the answer's form (per-form, not per-answer)
  step 2: range (90% range width in log-odds), know (1-5 self-rated knowledge, v = 6 - k),
          lp (variance of the implied number under the digit-token distribution, log-odds),
          yes (the P(YES) token probability as a point, with v = its Bernoulli-entropy proxy)

Error checks (per answer): Spearman of v with
  fit error  |logit(stated) - logit(equal-weight KL fit)|  (label-free)
  outcome    per-answer log loss of the stated answer
pooled, within form (mean of the 12 per-form rhos), and partial on |logit(stated)| (extremeness).

Weighted projection: w = 1 / (v + tau), tau in TAUS (tau = inf is equal weights), chosen on the
4 training folds (5 folds by pair) by all-12 log loss, applied to the held-out fold.  KL and
log-odds least squares.  Points: t0, s0 (weights from s1..s4: separate samples), mean of s0..s4
in log-odds (weights from the same samples, and a split: point = mean s0..s2, weights = s3, s4).
Paired bootstrap over pairs (2,000, seed 20260924) of the mean per-pair log loss.

Usage: confidence_weighting.py [--step2 DIR] --out /home/agent/data/confidence-weighting-20260927/step1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.batch_projection import A, project  # noqa: E402
from training.supervised_pairs import FORM_ORDER  # noqa: E402

LO = 0.01
SEED = 20260924
NB = 2000
K = 5
TAUS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, np.inf)
DEC = Path("/home/agent/data/decomposition-32b-20260924")
OLD = Path("/home/agent/data/jev-disjoint-v1-filtered-20260919/test")
SINGLE = ("P", "NOT P", "Q", "NOT Q")
JI = np.array([i for i, f in enumerate(FORM_ORDER) if f not in SINGLE])
FI = {f: i for i, f in enumerate(FORM_ORDER)}


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def cl(x):
    return np.clip(x, LO, 1 - LO)


def logit(x):
    x = cl(x)
    return np.log(x / (1 - x))


def sig(t):
    return 1 / (1 + np.exp(-t))


def load_dec():
    """pids, Y (N x 12), P[sample] (N x 12, nan if missing), subset mask."""
    byp = defaultdict(lambda: defaultdict(dict))
    lab = defaultdict(dict)
    for r in jl(DEC / "log.jsonl"):
        byp[r["pair_id"]][r["sample"]][r["form"]] = r["p"]
        lab[r["pair_id"]][r["form"]] = r["label"]
    pids = sorted(byp)
    samples = ["t0", "s0", "s1", "s2", "s3", "s4", "w1", "w2", "w3"]
    P = {s: np.array([[byp[p][s].get(f, np.nan) if byp[p][s].get(f) is not None else np.nan for f in FORM_ORDER]
                      for p in pids], float) for s in samples}
    Y = np.array([[lab[p][f] for f in FORM_ORDER] for p in pids], float)
    sub = set(json.loads((DEC / "subset.json").read_text())["pair_ids"])
    return pids, Y, P, np.array([p in sub for p in pids])


def pair_ll(P, Y):
    """N x 2: mean per-pair log loss over all 12 and over the 8 AND/OR forms."""
    p = cl(np.where(np.isnan(P), 0.5, P))
    L = -(Y * np.log(p) + (1 - Y) * np.log(1 - p))
    return np.stack([L.mean(1), L[:, JI].mean(1)], 1)


def proj(T, W, kind):
    return project(T, W, kind) @ A.T


def folds_of(n):
    return np.random.default_rng(SEED).permutation(n) % K


def grid_weights(V, taus=TAUS, Vf=None):
    """Candidate weight matrices: 1/(V + tau); with Vf also the product 1/(Vf + t1) * 1/(V + t2)."""
    Vn = np.nan_to_num(V, nan=np.nanmedian(V))
    cand = {("eq",): np.ones_like(Vn)}
    for t in taus:
        if not np.isinf(t):
            cand[("a", t)] = 1 / (Vn + t)
    if Vf is not None:
        for t1 in taus:
            for t2 in taus:
                if not (np.isinf(t1) and np.isinf(t2)):
                    w1 = 1 if np.isinf(t1) else 1 / (Vf + t1)
                    w2 = 1 if np.isinf(t2) else 1 / (Vn + t2)
                    cand[("f", t1, t2)] = w1 * w2 * np.ones_like(Vn)
    return cand


def weighted_crossfit(T, V, Y, kind, taus=TAUS, Vf=None):
    """Cross-fitted choice among grid_weights by all-12 log loss on the training folds."""
    n = len(T)
    ok = ~np.isnan(T)
    T0 = np.where(ok, T, 0.0)
    fold = folds_of(n)
    cache = {key: proj(T0, W * ok, kind) for key, W in grid_weights(V, taus, Vf).items()}
    out = np.full((n, 12), np.nan)
    chosen = []
    for k in range(K):
        tr, te = fold != k, fold == k
        best = min(cache, key=lambda key: pair_ll(cache[key][tr], Y[tr])[:, 0].mean())
        chosen.append(str(best))
        out[te] = cache[best][te]
    return out, chosen, {str(key): float(pair_ll(v, Y)[:, 0].mean()) for key, v in cache.items()}


def boot(D):
    """D: n x m per-pair diffs -> [[mean, lo, hi], ...]."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(D), size=(NB, len(D)))
    m = D[idx].mean(1)
    return [[float(D[:, j].mean()), float(np.percentile(m[:, j], 2.5)), float(np.percentile(m[:, j], 97.5))]
            for j in range(D.shape[1])]


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(a) for a in (x, y, z))

    def res(a):
        b = np.polyfit(rz, a, 1)
        return a - np.polyval(b, rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def error_checks(T, V, Y, fitT, mask=None):
    """Spearman of confidence variance V with the answer's fit error and outcome log loss."""
    p = sig(T)
    e_fit = np.abs(T - fitT)
    e_out = -(Y * np.log(cl(p)) + (1 - Y) * np.log(1 - cl(p)))
    ext = np.abs(T)
    M = np.ones_like(T, bool) if mask is None else np.repeat(mask[:, None], 12, 1)
    M = M & ~np.isnan(V) & ~np.isnan(T)
    res = {}
    for name, E in (("fit_error", e_fit), ("outcome_logloss", e_out)):
        v, e, x = V[M], E[M], ext[M]
        within = []
        for j in range(12):
            mj = M[:, j]
            if mj.sum() > 20 and np.std(V[mj, j]) > 0:
                within.append(spearmanr(V[mj, j], E[mj, j])[0])
        res[name] = {"pooled": float(spearmanr(v, e)[0]), "within_form_mean": float(np.mean(within)),
                     "partial_on_extremeness": partial_spearman(v, e, x)}
    res["n_answers"] = int(M.sum())
    res["share_zero_variance"] = float((V[M] == 0).mean())
    res["rho_with_extremeness"] = float(spearmanr(V[M], ext[M])[0])
    return res


def form_control(Vs, n):
    """Per-form mean variance, cross-fitted (training folds only)."""
    fold = folds_of(n)
    out = np.zeros_like(Vs)
    for k in range(K):
        tr = fold != k
        out[fold == k] = np.nanmean(Vs[tr], 0)
    return out


def run_arms(arms, Y, label):
    """arms: list of (name, T point, V or None for equal).  Returns results incl. CIs vs the matching equal arm."""
    res = {"pairs": int(len(Y)), "arms": {}}
    L = {}
    for arm in arms:
        name, T, V, ref, kind = arm[:5]
        Vf = arm[5] if len(arm) > 5 else None
        if V is None:
            pred = proj(np.nan_to_num(T), (~np.isnan(T)).astype(float), kind)
            info = {}
        else:
            pred, chosen, grid = weighted_crossfit(T, V, Y, kind, Vf=Vf)
            info = {"tau_by_fold": chosen, "all12_ll_by_weights_insample": grid}
        L[name] = pair_ll(pred, Y)
        res["arms"][name] = {"all12_ll": float(L[name][:, 0].mean()), "andor_ll": float(L[name][:, 1].mean()),
                             "ref": ref, **info}
        print(f"[{label}] {name:48s} all12 {L[name][:, 0].mean():.4f} andor {L[name][:, 1].mean():.4f} "
              f"{info.get('tau_by_fold', '')}", flush=True)
    for arm in arms:
        name, ref = arm[0], arm[3]
        if ref:
            b = boot(L[name] - L[ref])
            res["arms"][name]["vs_ref"] = {"all12": b[0], "andor": b[1]}
            print(f"    {name} - {ref}: all12 {b[0][0]:+.4f} [{b[0][1]:+.4f}, {b[0][2]:+.4f}]", flush=True)
    return res, L


# ------------------------------------------------------------------ step 2 signals

DIGITS = set("0123456789")


def _tops(tok):
    return {t["token"]: np.exp(t["logprob"]) for t in tok["top_logprobs"]}


def digit_signal(lps):
    """Implied-number distribution from the logprobs of 'Probability: 0.d1d2'.
    d1 from the top-5 at the first decimal, d2 from the top-5 at the second (non-digit = answer ended, d2 = 0);
    d2's distribution is assumed the same for every d1.  Returns (mean prob, var of logit, entropy of d1) or Nones."""
    if not lps:
        return None, None, None
    toks = [t["token"] for t in lps]
    for j in range(1, len(toks) - 1):
        if toks[j] == "." and toks[j - 1].strip() == "0" and toks[j + 1] in DIGITS:
            break
    else:
        return None, None, None
    t1 = {k: v for k, v in _tops(lps[j + 1]).items() if k in DIGITS}
    z1 = sum(t1.values())
    d1 = {int(k): v / z1 for k, v in t1.items()}
    if j + 2 < len(lps):
        t2 = _tops(lps[j + 2])
        d2 = defaultdict(float)
        for k, v in t2.items():
            d2[int(k) if k in DIGITS else 0] += v
        z2 = sum(d2.values()); d2 = {k: v / z2 for k, v in d2.items()}
    else:
        d2 = {0: 1.0}
    vals, ws = [], []
    for a, pa in d1.items():
        for b, pb in d2.items():
            vals.append(a / 10 + b / 100); ws.append(pa * pb)
    vals, ws = np.array(vals), np.array(ws)
    lv = logit(vals)
    m = float((ws * lv).sum())
    return float((ws * cl(vals)).sum()), float((ws * (lv - m) ** 2).sum()), float(-(np.array(list(d1.values())) *
                                                                                   np.log(list(d1.values()))).sum())


def know_signal(lps):
    """Expected self-rated knowledge (1-5) from the logprobs of the digit after 'Knowledge:'."""
    if not lps:
        return None
    toks = [t["token"] for t in lps]
    for j, t in enumerate(toks):
        if t.strip() in {"1", "2", "3", "4", "5"} and "Knowledge" in "".join(toks[max(0, j - 4):j]):
            d = {int(k.strip()): v for k, v in _tops(lps[j]).items() if k.strip() in {"1", "2", "3", "4", "5"}}
            z = sum(d.values())
            return float(sum(k * v for k, v in d.items()) / z)
    return None


RANGE = __import__("re").compile(r"Range:\s*\**\s*(\d*\.?\d+)\s*%?\s*(?:-|–|to)\s*(\d*\.?\d+)", __import__("re").I)


def range_signal(text):
    """sd on the log-odds scale implied by a 90% range: (logit hi - logit lo) / 3.29."""
    m = RANGE.search(text or "")
    if not m:
        return None
    lo, hi = float(m.group(1)), float(m.group(2))
    if lo > 1 or hi > 1:
        lo, hi = lo / 100, hi / 100
    if hi < lo:
        lo, hi = hi, lo
    return float((logit(hi) - logit(lo)) / 3.29)


def yes_signal(lps):
    if not lps:
        return None
    for t in reversed(lps):
        y = n = 0.0
        for k, v in _tops(t).items():
            kk = k.strip(" .:_=").upper()
            y += v if kk == "YES" else 0; n += v if kk == "NO" else 0
        if y + n > 0.5:
            return float(y / (y + n))
    return None


def load_step2(d, pids):
    idx = {p: i for i, p in enumerate(pids)}
    n = len(pids)
    M = {k: np.full((n, 12), np.nan) for k in ("p_lp", "lpmean", "v_lp", "h_lp", "p_range", "sd_range", "p_know",
                                                "know", "pyes")}
    for r in jl(Path(d) / "log.jsonl"):
        i, j = idx[r["pair_id"]], FI[r["form"]]
        v = r["variant"]
        lps = r.get("logprobs_retry") if r.get("retried") else r.get("logprobs")
        if v == "lp":
            M["p_lp"][i, j] = np.nan if r["p"] is None else r["p"]
            a, b, c = digit_signal(lps)
            if a is not None:
                M["lpmean"][i, j], M["v_lp"][i, j], M["h_lp"][i, j] = a, b, c
        elif v == "range":
            M["p_range"][i, j] = np.nan if r["p"] is None else r["p"]
            x = range_signal(r.get("raw_retry") if r.get("retried") else r["raw"])
            M["sd_range"][i, j] = np.nan if x is None else x
        elif v == "know":
            M["p_know"][i, j] = np.nan if r["p"] is None else r["p"]
            x = know_signal(lps)
            M["know"][i, j] = np.nan if x is None else x
        elif v == "yes":
            x = yes_signal(lps)
            M["pyes"][i, j] = np.nan if x is None else x
    return M


def agreement(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    return {"n": int(ok.sum()), "same": float((np.abs(a[ok] - b[ok]) < 1e-9).mean()),
            "within_0.05": float((np.abs(a[ok] - b[ok]) <= 0.05 + 1e-9).mean()),
            "mean_abs": float(np.abs(a[ok] - b[ok]).mean()), "mean_signed": float((a[ok] - b[ok]).mean()),
            "r_logit": float(np.corrcoef(logit(a[ok]), logit(b[ok]))[0, 1])}


def step2(d, pids, Y, P, sub, Tl, v_samp, n):
    ii = np.where(sub)[0]
    S = load_step2(d, [pids[i] for i in ii])
    Ys = Y[ii]
    t0 = P["t0"][ii]
    res = {"coverage": {k: float((~np.isnan(v)).mean()) for k, v in S.items()}, "prob_shift": {}, "error_checks": {},
           "raw_ll": {}, "projection": {}}
    for k in ("p_lp", "p_range", "p_know", "pyes"):
        res["prob_shift"][f"{k} vs t0 (Sep 24 run)"] = agreement(S[k], t0)
    for k in ("p_range", "p_know"):
        res["prob_shift"][f"{k} vs p_lp (same day)"] = agreement(S[k], S["p_lp"])
    for k in ("p_lp", "lpmean", "p_range", "p_know", "pyes"):
        res["raw_ll"][k] = {"all12": float(pair_ll(S[k], Ys)[:, 0].mean())}
    res["raw_ll"]["t0"] = {"all12": float(pair_ll(t0, Ys)[:, 0].mean())}
    base = pair_ll(S["p_lp"], Ys)
    res["raw_vs_lp"] = {k: boot(pair_ll(S[k] if k in S else t0, Ys) - base)[0]
                        for k in ("p_range", "p_know", "lpmean", "pyes", "t0")}
    T = logit(S["p_lp"])
    fitT = logit(proj(np.nan_to_num(T), (~np.isnan(T)).astype(float), "kl"))
    Tr, Tk, Ty = logit(S["p_range"]), logit(S["p_know"]), logit(S["pyes"])
    fitTr = logit(proj(np.nan_to_num(Tr), (~np.isnan(Tr)).astype(float), "kl"))
    fitTk = logit(proj(np.nan_to_num(Tk), (~np.isnan(Tk)).astype(float), "kl"))
    sig_defs = {
        "lp: var of implied number (digit logprobs)": (T, S["v_lp"], fitT),
        "lp: entropy of first digit": (T, S["h_lp"], fitT),
        "range: (90% width / 3.29)^2": (Tr, S["sd_range"] ** 2, fitTr),
        "know: 6 - E[knowledge]": (Tk, 6 - S["know"], fitTk),
        "yes: P(YES) Bernoulli var": (T, S["pyes"] * (1 - S["pyes"]), fitT),
        "yes: (logit P(YES) - logit stated)^2": (T, (logit(S["pyes"]) - T) ** 2, fitT),
        "samp (s0..s4), for reference": (T, v_samp[ii], fitT),
    }
    for name, (Tp, V, ft) in sig_defs.items():
        res["error_checks"][name] = error_checks(Tp, V, Ys, ft)
        r = res["error_checks"][name]
        print(f"[err step2] {name}: fit {r['fit_error']['pooled']:+.3f}/{r['fit_error']['within_form_mean']:+.3f}/"
              f"{r['fit_error']['partial_on_extremeness']:+.3f}  out {r['outcome_logloss']['pooled']:+.3f}/"
              f"{r['outcome_logloss']['within_form_mean']:+.3f}/{r['outcome_logloss']['partial_on_extremeness']:+.3f}",
              flush=True)
    m = len(ii)
    for kind in ("kl", "lsl"):
        fc = form_control(v_samp[ii], m)
        arms = [
            ("lp eq", T, None, None, kind),
            ("lp w=digit var", T, S["v_lp"], "lp eq", kind),
            ("lp w=digit entropy", T, S["h_lp"], "lp eq", kind),
            ("lp w=P(YES) var", T, S["pyes"] * (1 - S["pyes"]), "lp eq", kind),
            ("lp w=disagreement with P(YES)", T, (logit(S["pyes"]) - T) ** 2, "lp eq", kind),
            ("lp w=samp", T, v_samp[ii], "lp eq", kind),
            ("lp w=form-mean samp", T, fc, "lp eq", kind),
            ("lp w=form-mean samp x digit var", T, S["v_lp"], "lp w=form-mean samp", kind, fc),
            ("lpmean (digit-mean point) eq", logit(S["lpmean"]), None, "lp eq", kind),
            ("range eq", Tr, None, None, kind),
            ("range w=range width", Tr, S["sd_range"] ** 2, "range eq", kind),
            ("range w=form-mean samp x range width", Tr, S["sd_range"] ** 2, "range eq", kind, fc),
            ("know eq", Tk, None, None, kind),
            ("know w=6-E[k]", Tk, 6 - S["know"], "know eq", kind),
            ("know w=form-mean samp x 6-E[k]", Tk, 6 - S["know"], "know eq", kind, fc),
            ("P(YES) token eq", Ty, None, "lp eq", kind),
        ]
        res["projection"][kind], _ = run_arms(arms, Ys, f"step2 300 {kind}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--step2", type=Path, help="run_confidence_signals.py output dir (old subset)")
    ap.add_argument("--step2-only", action="store_true")
    a = ap.parse_args()
    pids, Y, P, sub = load_dec()
    Tl = {s: logit(P[s]) for s in P}
    S5 = np.stack([Tl[f"s{i}"] for i in range(5)], 0)
    W3 = np.stack([Tl[f"w{i}"] for i in range(1, 4)], 0)
    v_samp = S5.var(0, ddof=1)
    v_samp14 = S5[1:].var(0, ddof=1)
    v_samp34 = S5[3:].var(0, ddof=1)
    mean5 = S5.mean(0)
    mean3 = S5[:3].mean(0)
    v_word = W3.var(0, ddof=1)
    v_sw = np.concatenate([S5, W3], 0).var(0, ddof=1)
    n = len(pids)
    fitT0 = logit(proj(Tl["t0"], np.ones((n, 12)), "kl"))
    out = {"description": __doc__, "error_checks": {}, "projection": {}}
    if a.step2:
        out["step2"] = step2(a.step2, pids, Y, P, sub, Tl, v_samp, n)
        if a.step2_only:
            a.out.write_text(json.dumps(out, indent=1))
            return

    # ---- error checks (point = t0)
    out["error_checks"]["all_1066"] = {"samp (sd over s0..s4)": error_checks(Tl["t0"], v_samp, Y, fitT0)}
    out["error_checks"]["subset_300"] = {
        "samp (sd over s0..s4)": error_checks(Tl["t0"], v_samp, Y, fitT0, sub),
        "word (sd over w1..w3)": error_checks(Tl["t0"], v_word, Y, fitT0, sub),
        "sampword (s0..s4 + w1..w3)": error_checks(Tl["t0"], v_sw, Y, fitT0, sub)}
    for k, d in out["error_checks"].items():
        for s, r in d.items():
            print(f"[err {k}] {s}: fit {r['fit_error']}  out {r['outcome_logloss']}  zero-var {r['share_zero_variance']:.2f}",
                  flush=True)

    # ---- projections, all 1,066
    for kind in ("kl", "lsl"):
        arms = [
            ("t0 eq", Tl["t0"], None, None, kind),
            ("t0 w=samp(s0..s4)", Tl["t0"], v_samp, "t0 eq", kind),
            ("t0 w=form-mean samp (control)", Tl["t0"], form_control(v_samp, n), "t0 eq", kind),
            ("t0 w=form-mean x samp", Tl["t0"], v_samp, "t0 w=form-mean samp (control)", kind, form_control(v_samp, n)),
            ("s0 eq", Tl["s0"], None, None, kind),
            ("s0 w=samp(s1..s4)", Tl["s0"], v_samp14, "s0 eq", kind),
            ("mean5 eq", mean5, None, None, kind),
            ("mean5 w=samp(same s0..s4)", mean5, v_samp, "mean5 eq", kind),
            ("mean3 eq", mean3, None, None, kind),
            ("mean3(s0..s2) w=samp(s3,s4)", mean3, v_samp34, "mean3 eq", kind),
        ]
        out["projection"][f"all_1066 {kind}"], _ = run_arms(arms, Y, f"1066 {kind}")
        # subset: wording signals
        ii = np.where(sub)[0]
        arms = [
            ("t0 eq", Tl["t0"][ii], None, None, kind),
            ("t0 w=samp", Tl["t0"][ii], v_samp[ii], "t0 eq", kind),
            ("t0 w=word", Tl["t0"][ii], v_word[ii], "t0 eq", kind),
            ("t0 w=sampword", Tl["t0"][ii], v_sw[ii], "t0 eq", kind),
            ("mean5 eq", mean5[ii], None, None, kind),
            ("mean5 w=word (separate samples)", mean5[ii], v_word[ii], "mean5 eq", kind),
        ]
        out["projection"][f"subset_300 {kind}"], _ = run_arms(arms, Y[ii], f"300 {kind}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
