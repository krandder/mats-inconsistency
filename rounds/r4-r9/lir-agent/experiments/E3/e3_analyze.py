"""E3 CPU-side scoring (written before the runs; PREREG.md). Reads pair_*.json / probe_*.json from
e3_eval.py, computes the exact Bayes P(success) of every candidate by enumeration (env/shiphunt.py
placements + posterior), and writes per-pair metrics JSON:
  gold(n), oracle(n), fsel(n), raw gap(n), excess gap(n) (isotonic f->Bayes fit on calibration states of
  the same pair; selection stays argmax raw f), excess_ref gap(n) for frozen-forecaster pairs (isotonic
  from the frozen-proposer pair c000:c000), Smith-Winkler linear-shrinkage gap(n), InferenceTimePessimism
  gold(n) (Huang et al. 2025 Alg. 2-3, beta tuned on calibration states), distinct plan sets per 64,
  dedup-n gold/gap, realised-vs-Bayes n=1 check, probe Brier/ECE vs Bayes and vs realised.
n = 1..64 by the unbiased subset estimator (Nakano et al. / Gao et al. 2022): the item ranked i of N by
forecast is the argmax of a uniform n-subset with prob C(N-1-i, n-1)/C(N, n); tied forecasts share.
Test states = even indices, calibration states = odd indices.
Usage: python3 e3_analyze.py RUNDIR [RUNDIR ...]   (each RUNDIR has eval/ and summary.json)
"""
import glob, json, math, os, sys
from functools import lru_cache

import numpy as np

H = os.environ.get("LIR_HARNESS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../harness"))
sys.path.insert(0, H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import shiphunt as SH  # noqa: E402
from dataclasses import replace  # noqa: E402
SH.PRESETS.setdefault("hardsyco", replace(SH.PRESETS["hard"], note=True, note_rel=0.4))
NS = list(range(1, 65))
KEY_N = (1, 4, 16, 64)


@lru_cache(maxsize=None)
def _env(preset, seed):
    e = SH.ShipHunt(SH.PRESETS[preset], seed)
    return e, e.posterior(), SH.placements(e.cfg)


def bayes_and_real(preset, seed, plans):
    e, w, P = _env(preset, seed)
    pl = np.asarray(plans)                                  # [K, L]
    occ = P[:, pl]                                          # [nP, K, L]
    good = occ if e.goal == "hit" else ~occ
    succ = good.sum(-1) >= e.cfg.need_hits                  # order-free success
    b = w @ succ
    real = (e.occ[pl] if e.goal == "hit" else ~e.occ[pl]).sum(-1) >= e.cfg.need_hits
    ent = float(-(w[w > 0] * np.log(w[w > 0])).sum())
    return b, real.astype(float), ent


def subset_weights(f, n):
    """P(item is the argmax of a uniform n-subset), ties shared equally."""
    N = len(f)
    if n > N:
        return None
    o = np.argsort(-f, kind="stable")
    i = np.arange(N)
    lw = np.array([math.lgamma(N - k) - math.lgamma(n) - math.lgamma(N - k - n + 1) if N - 1 - k >= n - 1 else -np.inf
                   for k in i]) - (math.lgamma(N + 1) - math.lgamma(n + 1) - math.lgamma(N - n + 1))
    wr = np.exp(lw)
    w = np.zeros(N)
    w[o] = wr
    fs = f[o]
    # share within tie groups
    start = 0
    while start < N:
        end = start + 1
        while end < N and fs[end] == fs[start]:
            end += 1
        if end - start > 1:
            idx = o[start:end]
            w[idx] = w[idx].mean()
        start = end
    return w


def pav(x, y):
    """Isotonic (non-decreasing) regression of y on x; returns predict function."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(float)
    vals, wts, xr = [], [], []
    for xi, yi in zip(xs, ys):
        vals.append(yi); wts.append(1.0); xr.append([xi, xi])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; vals[-2] = v; xr[-2][1] = xr[-1][1]
            vals.pop(); wts.pop(); xr.pop()
    lo = np.array([r[0] for r in xr]); hi = np.array([r[1] for r in xr]); vals = np.array(vals)
    mids_x = np.concatenate([[lo[0]], (hi[:-1] + lo[1:]) / 2, [hi[-1]]]) if len(vals) > 1 else None

    def pred(z):
        z = np.asarray(z, float)
        if len(vals) == 1:
            return np.full_like(z, vals[0])
        k = np.searchsorted(lo, z, side="right") - 1
        return vals[np.clip(k, 0, len(vals) - 1)]
    return pred


def ece_eqmass(p, y, bins=10):
    o = np.argsort(p, kind="stable")
    tot = 0.0
    for ch in np.array_split(o, bins):
        if len(ch):
            tot += len(ch) * abs(p[ch].mean() - y[ch].mean())
    return tot / len(p)


def itp_value(f, b, n, beta, rng, reps=64):
    """InferenceTimePessimism (Huang et al. 2025, Alg. 2-3) on an n-subset: lambda solves
    mean relu((r-lambda)/beta) = 1 over the subset; output ~ subset item with prob prop. to relu(.)
    (the exact target of the rejection sampler). Returns E[Bayes value], E[forecast] of the output."""
    N = len(f)
    gs, fs = [], []
    for _ in range(reps if n < N else 1):
        idx = rng.choice(N, n, replace=False) if n < N else np.arange(N)
        r = f[idx]
        rs = np.sort(r)
        lam = None
        J, Z = rs.sum() / n, 1.0       # Alg. 3 over sorted unique values (ascending)
        u, cnt = np.unique(rs, return_counts=True)
        prev = -np.inf
        for i in range(len(u)):
            l = (J - beta) / Z
            if (prev <= l < u[i]) or i == len(u) - 1:
                lam = l
                break
            J -= u[i] * cnt[i] / n
            Z -= cnt[i] / n
            prev = u[i]
        wt = np.maximum((r - lam) / beta, 0)
        if wt.sum() <= 0:
            wt = np.ones_like(r)
        wt = wt / wt.sum()
        gs.append(float(wt @ b[idx])); fs.append(float(wt @ r))
    return float(np.mean(gs)), float(np.mean(fs))


def curves(F, B, ns=NS):
    """F, B: lists per state of arrays (forecast, Bayes). Returns per-n arrays [S] of fsel and gold."""
    S = len(F)
    fsel = np.full((len(ns), S), np.nan)
    gold = np.full((len(ns), S), np.nan)
    orac = np.full((len(ns), S), np.nan)
    for s in range(S):
        f, b = F[s], B[s]
        for j, n in enumerate(ns):
            w = subset_weights(f, n)
            if w is None:
                continue
            fsel[j, s] = w @ f
            gold[j, s] = w @ b
            wo = subset_weights(b, n)
            orac[j, s] = wo @ b
    return fsel, gold, orac


def mean_se(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))] if len(x) > 1 else [float("nan")] * 2


def analyze_pair(path, iso_ref=None):
    d = json.load(open(path))
    preset = d["preset"]
    MAXS = int(os.environ.get("E3_MAX_STATES", "120"))   # homogenise: 240-state pilot files truncated to the first 120 states
    seeds, plans, f = d["state_seeds"][:MAXS], d["plans"][:MAXS], d["f"][:MAXS]
    B, R, F, ENT, DIST, DIST_ORD, DD = [], [], [], [], [], [], []
    for sd, pl, fs in zip(seeds, plans, f):
        b, r, ent = bayes_and_real(preset, sd, pl)
        B.append(b); R.append(r); F.append(np.asarray(fs)); ENT.append(ent)
        sets = [frozenset(p) for p in pl]
        DIST.append(len(set(sets))); DIST_ORD.append(len(set(tuple(p) for p in pl)))
        seen, keep = set(), []
        for k, s_ in enumerate(sets):
            if s_ not in seen:
                seen.add(s_); keep.append(k)
        DD.append(keep)
    S = len(seeds)
    test = [s for s in range(S) if s % 2 == 0]
    cal = [s for s in range(S) if s % 2 == 1]
    fc = np.concatenate([F[s] for s in cal]); bc = np.concatenate([B[s] for s in cal])
    iso = pav(fc, bc)
    # Smith-Winkler-style linear posterior-mean shrinkage fitted on calibration candidates
    # alpha = var(v) / (var(v) + var(error)), error = (f - mean f) - (v - mean v): classic shrinkage toward the prior mean
    m_f, m_b = float(fc.mean()), float(bc.mean())
    var_v = float(np.var(bc, ddof=1)); var_e = float(np.var((fc - m_f) - (bc - m_b), ddof=1))
    k_sw = var_v / max(var_v + var_e, 1e-12)
    sw = lambda z: m_b + k_sw * (np.asarray(z) - m_f)
    Ft, Bt = [F[s] for s in test], [B[s] for s in test]
    fsel, gold, orac = curves(Ft, Bt)
    # excess: select by raw f, value by iso(f)
    exc = np.full_like(fsel, np.nan); swg = np.full_like(fsel, np.nan); exc_ref = np.full_like(fsel, np.nan)
    for j, n in enumerate(NS):
        for t, s in enumerate(test):
            w = subset_weights(F[s], n)
            exc[j, t] = w @ iso(F[s]) - w @ B[s]
            swg[j, t] = w @ sw(F[s]) - w @ B[s]
            if iso_ref is not None:
                exc_ref[j, t] = w @ iso_ref(F[s]) - w @ B[s]
    gap = fsel - gold
    # POST-HOC (added 15:55Z after seeing that a near-constant forecaster makes the single-split excess gap equal to the
    # calibration-minus-test mean difference): 2-fold cross-fitted excess gap over ALL states (fit on odd -> score even,
    # fit on even -> score odd), and gold / raw gap over all states.
    ft = np.concatenate([F[s] for s in test]); bt_ = np.concatenate([B[s] for s in test])
    iso_t = pav(ft, bt_)
    allst = list(range(S))
    xf = np.full((len(NS), S), np.nan); ga = np.full((len(NS), S), np.nan); gp = np.full((len(NS), S), np.nan)
    for j, n in enumerate(NS):
        for s_ in allst:
            w = subset_weights(F[s_], n)
            isof = iso if s_ % 2 == 0 else iso_t
            xf[j, s_] = w @ isof(F[s_]) - w @ B[s_]
            ga[j, s_] = w @ B[s_]
            gp[j, s_] = w @ F[s_] - w @ B[s_]
    out_x = {"excess_gap_xfit": [mean_se(xf[j]) for j in range(len(NS))],
             "gold_all": [mean_se(ga[j]) for j in range(len(NS))], "gap_all": [mean_se(gp[j]) for j in range(len(NS))],
             "n_states_all": S}
    out = {"pair": os.path.basename(path), "proposer": d["proposer"], "forecaster": d["forecaster"], "preset": preset,
           "n_states_test": len(test), "n_states_cal": len(cal), "ns": NS}
    for name, arr in [("gold", gold), ("oracle", orac), ("fsel", fsel), ("gap", gap), ("excess_gap", exc),
                      ("sw_gap", swg), ("excess_ref_gap", exc_ref)]:
        ms = [mean_se(arr[j]) for j in range(len(NS))]
        out[name] = [m[0] for m in ms]
        out[name + "_se"] = [m[1] for m in ms]
    out["gold_deficit"] = [o - g for o, g in zip(out["oracle"], out["gold"])]
    out["posthoc_all_states"] = out_x
    # realised check at n=1 (random candidate): mean realised vs mean Bayes over all test candidates
    rt = np.concatenate([R[s] for s in test]); btt = np.concatenate([B[s] for s in test]); ftt = np.concatenate([F[s] for s in test])
    out["n1_realised_mean"], out["n1_bayes_mean"], out["n1_forecast_mean"] = float(rt.mean()), float(btt.mean()), float(ftt.mean())
    # realised outcome of the argmax-f candidate at n=64 (the full pool), as a check on gold(64)
    out["n64_realised_of_argmax"] = float(np.mean([R[s][int(np.argmax(F[s]))] for s in test]))
    out["cand_brier_vs_bayes_test"] = float(np.mean((ftt - btt) ** 2))
    out["distinct_sets_per64_mean"] = float(np.mean(DIST)); out["distinct_ordered_per64_mean"] = float(np.mean(DIST_ORD))
    out["dup_rate_sets"] = float(1 - np.mean(DIST) / len(plans[0]))
    # dedup-n: n counts distinct plan sets (first occurrence kept)
    dd = {}
    for n in (4, 16, 32):
        g_, gp_, ok = [], [], 0
        for s in test:
            k = DD[s]
            if len(k) < n:
                continue
            ok += 1
            w = subset_weights(F[s][k], n)
            g_.append(w @ B[s][k]); gp_.append(w @ F[s][k] - w @ B[s][k])
        dd[str(n)] = {"states_with_enough_distinct": ok, "gold": mean_se(g_), "gap": mean_se(gp_)}
    out["dedup"] = dd
    # InferenceTimePessimism: beta tuned on calibration states at n=64 (max gold), then test curve
    rng = np.random.default_rng(0)
    betas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    calg = {bt: np.mean([itp_value(F[s], B[s], 64, bt, rng)[0] for s in cal]) for bt in betas}
    bstar = max(calg, key=calg.get)
    itp = {}
    for n in (1, 4, 16, 64):
        v = [itp_value(F[s], B[s], n, bstar, rng, reps=32) for s in test]
        itp[str(n)] = {"gold": mean_se([x[0] for x in v]), "gap": mean_se([x[1] - x[0] for x in v])}
    out["itp"] = {"beta": bstar, "cal_gold_by_beta": {str(k): float(v) for k, v in calg.items()}, "test": itp}
    out["sw_k"], out["sw_mf"], out["sw_mb"] = k_sw, m_f, m_b
    # entropy strata (exogenous n): gap/gold at n=16 by posterior-entropy tercile
    ent_t = np.array([ENT[s] for s in test]); j16 = NS.index(16)
    q = np.quantile(ent_t, [1 / 3, 2 / 3])
    strata = {}
    for nm, m in [("low", ent_t <= q[0]), ("mid", (ent_t > q[0]) & (ent_t <= q[1])), ("high", ent_t > q[1])]:
        strata[nm] = {"gold16": mean_se(gold[j16][m]), "gap16": mean_se(gap[j16][m]), "excess16": mean_se(exc[j16][m])}
    out["entropy_strata"] = strata
    return out, iso


def analyze_probe(path):
    d = json.load(open(path))
    F, B, R = [], [], []
    MAXP = int(os.environ.get("E3_MAX_PROBE_STATES", "100"))
    for sd, pl, fs in list(zip(d["state_seeds"], d["plans"], d["f"]))[:MAXP]:
        b, r, _ = bayes_and_real(d["preset"], sd, pl)
        F.append(np.asarray(fs)); B.append(b); R.append(r)
    F, B, R = np.concatenate(F), np.concatenate(B), np.concatenate(R)
    return {"forecaster": d["forecaster"], "n": int(len(F)), "brier_vs_bayes": float(np.mean((F - B) ** 2)),
            "brier_realised": float(np.mean((F - R) ** 2)), "bayes_brier_floor": float(np.mean((B - R) ** 2)),
            "ece_realised": float(ece_eqmass(F, R)), "ece_vs_bayes": float(ece_eqmass(F, B)),
            "mean_f": float(F.mean()), "mean_bayes": float(B.mean()), "corr_f_bayes": float(np.corrcoef(F, B)[0, 1])}


def main():
    for rd in sys.argv[1:]:
        ev = os.path.join(rd, "eval")
        if not os.path.isdir(ev):
            print(rd, "no eval dir, skipped")
            continue
        res = {"run": rd, "pairs": {}, "probes": {}}
        ref = None
        p0 = os.path.join(ev, "pair_c000_c000.json")
        if os.path.exists(p0):
            o, ref = analyze_pair(p0)
            res["pairs"]["c000:c000"] = o
        else:   # RAPc / RAPf share P's warm start: the frozen-proposer reference is P's c000:c000
            pp = os.path.join(os.path.dirname(os.path.normpath(rd)), "P", "eval", "pair_c000_c000.json")
            if os.path.exists(pp):
                _, ref = analyze_pair(pp)
                res["ref_pair"] = pp
        frozen_fc = os.path.basename(os.path.normpath(rd)) == "RAPf"
        for p in sorted(glob.glob(os.path.join(ev, "pair_*.json"))):
            k = os.path.basename(p)[5:-5].replace("_", ":")
            if k in res["pairs"]:
                continue
            o, _ = analyze_pair(p, iso_ref=ref if (k.endswith(":c000") or frozen_fc) else None)
            res["pairs"][k] = o
        for p in sorted(glob.glob(os.path.join(ev, "probe_*.json"))):
            res["probes"][os.path.basename(p)[6:-5]] = analyze_probe(p)
        json.dump(res, open(os.path.join(rd, "e3_metrics.json"), "w"), indent=1, default=float)
        print(rd, "pairs", len(res["pairs"]), "probes", len(res["probes"]))


if __name__ == "__main__":
    main()
