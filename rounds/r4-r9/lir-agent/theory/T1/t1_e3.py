"""T1 on E3's frozen-checkpoint forecasts (CPU only, no new GPU). Inputs (read-only):
  ../../experiments/E3/runs/sS/ARM/eval/pair_cP_cF.json   64 policy-proposed candidates per state, forecaster E[P(success)]
  ../../experiments/E3/runs/sS/ARM/eval/probe_cF.json     16 uniform-random plans per state (same plans for every arm/seed)
Truth: exact Bayes P* per candidate (e3_analyze.bayes_and_real: enumeration over ShipHunt placements); epsilon = F - P*.
Outputs t1_e3.json:
  curse(n) = gap(argmax F) - gap(random pick) on nested pools of DISTINCT plans, n in {1,2,4,8,16} (ties in F averaged);
  fused-vs-protected curse difference at n=8 vs seed sd;
  Spearman(stated uncertainty proxy, |eps|) overall and at the argmax: proxies are f(1-f) (the only uncertainty the fixed
  format states) and two-seed disagreement |F_s - F_s'| on the shared probe plans;
  stopping rules at cost c per distinct forecast call, scored by P*_committed - c*calls, calibration = odd states, test = even.
Usage: python3 t1_e3.py   (single process)."""
import glob, json, os, sys, time
import numpy as np
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.normpath(os.path.join(HERE, "../../experiments/E3"))
sys.path.insert(0, E3)
import e3_analyze as EA  # noqa: E402  (adds harness to path, registers hardsyco)

NS = [1, 2, 4, 8, 16]; CS = [0.005, 0.01, 0.02]
t0 = time.time()


def load(path):
    d = json.load(open(path))
    F, B, R, P = [], [], [], []
    for seed, plans, f in zip(d["state_seeds"], d["plans"], d["f"]):
        b, r, _ = EA.bayes_and_real(d["preset"], seed, plans)
        F.append(np.asarray(f, float)); B.append(b); R.append(r); P.append([tuple(sorted(p)) for p in plans])
    return d, F, B, R, P


def distinct_pool(F, B, R, P, K=16):
    seen, idx = set(), []
    for i, p in enumerate(P):
        if p not in seen:
            seen.add(p); idx.append(i)
        if len(idx) == K:
            break
    idx = np.array(idx)
    return F[idx], B[idx], R[idx]


def argmax_tie_avg(f, x):
    m = f == f.max()
    return float(x[m].mean())


def curse_curves(F, B, R, P, states):
    res = {n: dict(curse=[], gap=[], gold=[], oracle=[]) for n in NS}; nd = []
    for s in states:
        f, b, _ = distinct_pool(F[s], B[s], R[s], P[s]); nd.append(len(f)); e = f - b
        for n in NS:
            if len(f) < n:
                continue
            ff, ee, bb = f[:n], e[:n], b[:n]
            g = argmax_tie_avg(ff, ee)
            res[n]["curse"].append(g - ee.mean()); res[n]["gap"].append(g)
            res[n]["gold"].append(argmax_tie_avg(ff, bb)); res[n]["oracle"].append(bb.max())
    out = {str(n): {k: (float(np.mean(v)) if v else None) for k, v in d.items()} | {"n_states": len(d["curse"]),
           "curse_se": float(np.std(d["curse"]) / np.sqrt(max(len(d["curse"]), 1)))} for n, d in res.items()}
    out["mean_distinct_per64"] = float(np.mean(nd))
    return out


# ---------- stopping rules on distinct pools ----------
def pools(F, B, R, P, states):
    return [distinct_pool(F[s], B[s], R[s], P[s]) for s in states]


def fixed_val(pl, n, score=None):
    v = []
    for i, (f, b, r) in enumerate(pl):
        k = min(n, len(f)); sc = (f if score is None else score(f))[:k]
        v.append((argmax_tie_avg(sc, b[:k]), k))
    return v


def best_n(pl, c, score=None):
    vals = {n: np.mean([g - c * k for g, k in fixed_val(pl, n, score)]) for n in range(1, 17)}
    return max(vals, key=vals.get)


def hedgetune_n(pl, c):
    """Khalaf et al. 2025 Thm 3 (BoN) with a cost: root of mean_t E_{u~Beta(n,1)}[r_t(u)] gradient = c."""
    rows = [(f, b) for f, b, _ in pl if len(f) >= 2]
    if not rows:
        return 1
    def resid(n):
        tot = []
        for f, b in rows:
            K = len(f); u = (stats.rankdata(f, method="average") - 0.5) / K
            tot.append(np.sum(b * (1 / n + np.log(u)) * n * u ** (n - 1)) / K)
        return float(np.mean(tot)) - c
    if resid(1.0) <= 0: return 1
    if resid(16.0) >= 0: return 16
    return int(round(optimize.brentq(resid, 1.0, 16.0)))


def iso_fit(pl):
    f = np.concatenate([p[0] for p in pl]); r = np.concatenate([p[2] for p in pl])
    return EA.pav(f, r)                          # learned gap model on realised 0/1 outcomes (GOAL signal 4)


def reservation_rel(pl, m, c):
    parts = [m(f) - m(f).mean() for f, _, _ in pl if len(f) > 1]
    if not parts:
        return np.inf
    d = np.concatenate(parts)
    g = lambda z: np.maximum(d - z, 0).mean() - c
    lo, hi = d.min() - 1, d.max() + 1
    if g(lo) * g(hi) > 0:
        return hi
    return optimize.brentq(g, lo, hi)


def pandora_rel(pl, m, zd, kmin):
    v = []
    for f, b, _ in pl:
        h = m(f)
        for k in range(len(h)):
            seen = h[:k + 1]
            if (k + 1 >= kmin and seen.max() - seen.mean() >= zd) or k == len(h) - 1:
                v.append((argmax_tie_avg(seen, b[:k + 1]), k + 1)); break
    return v


def tuned_pandora(cal, test, m, c):
    zd = reservation_rel(cal, m, c)
    km = max((2, 3, 4), key=lambda kk: np.mean([g - c * k for g, k in pandora_rel(cal, m, zd, kk)]))
    return pandora_rel(test, m, zd, km)


def score(v, c):
    return float(np.mean([g - c * k for g, k in v])), float(np.mean([k for _, k in v]))


def stopping(F, B, R, P, S):
    cal = pools(F, B, R, P, [s for s in range(S) if s % 2 == 1]); te = pools(F, B, R, P, [s for s in range(S) if s % 2 == 0])
    m = iso_fit(cal); ident = lambda f: f
    mu0 = np.mean(np.concatenate([p[0] for p in cal])); vb = np.concatenate([p[1] for p in cal])
    s0sq = max(np.var(vb), 1e-4)
    sw = lambda f: mu0 + s0sq / (s0sq + f * (1 - f)) * (f - mu0)   # Smith-Winkler with the Bernoulli-implied sd
    out = {}
    for c in CS:
        r = {}
        n_o = best_n([(b, b, rr) for _, b, rr in cal], c)
        r["oracle_fixed_n"] = score(fixed_val([(b, b, rr) for _, b, rr in te], n_o), c)
        n_f = best_n(cal, c); r["fixed_n"] = score(fixed_val(te, n_f), c)
        n_h = hedgetune_n(cal, c); r["hedgetune_n"] = score(fixed_val(te, n_h), c)
        n_g = best_n(cal, c, m); r["gap_model_fixed_n"] = score(fixed_val(te, n_g, m), c)
        r["gap_model_pandora_rel"] = score(tuned_pandora(cal, te, m, c), c)
        r["raw_pandora_rel"] = score(tuned_pandora(cal, te, ident, c), c)
        n_s = best_n(cal, c, sw); r["smith_winkler_bernoulli_sd_fixed_n"] = score(fixed_val(te, n_s, sw), c)
        r["n"] = dict(oracle=n_o, fixed=n_f, hedgetune=n_h, gap_model=n_g, sw=n_s)
        out[str(c)] = r
    return out


def slope_and_recal(F, B, R, P, S):
    """Within-pool linear decomposition F = a_s + beta*P* + eta and the P1b prediction of curse(8);
    cross-fitted recalibrated curse: m = isotonic(F -> realised) fitted on the other half of states (GOAL signal 4)."""
    pl = [distinct_pool(F[s], B[s], R[s], P[s]) for s in range(S)]
    fd = np.concatenate([f - f.mean() for f, b, _ in pl if len(f) > 1]); bd = np.concatenate([b - b.mean() for f, b, _ in pl if len(f) > 1])
    beta = float(np.dot(fd, bd) / np.dot(bd, bd)); tau2 = float(np.var(fd - beta * bd)); sV2 = float(np.var(bd))
    from scipy import integrate
    e8 = integrate.quad(lambda x: x * 8 * stats.norm.pdf(x) * stats.norm.cdf(x) ** 7, -12, 12)[0]
    pred8 = ((beta - 1) * beta * sV2 + tau2) / np.sqrt(beta ** 2 * sV2 + tau2) * e8 if beta ** 2 * sV2 + tau2 > 0 else 0.0
    rec = {n: [] for n in NS}
    for half in (0, 1):
        fit = [pl[s] for s in range(S) if s % 2 != half]; m = iso_fit(fit)
        for s in range(S):
            if s % 2 != half:
                continue
            f, b, _ = pl[s]; e = m(f) - b
            for n in NS:
                if len(f) >= n:
                    rec[n].append(argmax_tie_avg(f[:n], e[:n]) - e[:n].mean())
    return dict(beta=beta, tau2=tau2, sV2_within=sV2, curse8_gaussian_pred=float(pred8),
                curse_recal={str(n): (float(np.mean(v)) if v else None) for n, v in rec.items()},
                gap_recal_at_argmax16=None)


def spearman_f1f(F, B, P, S):
    f = np.concatenate(F); b = np.concatenate(B); u = f * (1 - f); e = f - b
    at_f, at_e = [], []
    for s in range(S):
        fs, bs, _ = distinct_pool(F[s], B[s], B[s], P[s])
        i = int(np.argmax(fs)); at_f.append(fs[i] * (1 - fs[i])); at_e.append(fs[i] - bs[i])
    sp = lambda a, c: (float(stats.spearmanr(a, c).statistic) if np.std(a) > 0 and np.std(c) > 0 else None)
    return dict(all=sp(u, np.abs(e)), at_argmax_abs=sp(at_f, np.abs(at_e)), at_argmax_signed=sp(at_f, at_e),
                f_sd=float(np.std(f)), f_n_unique=int(len(np.unique(np.round(f, 4)))))


out = {"runs": {}, "probe_ensemble": {}}
runs = sorted(glob.glob(os.path.join(E3, "runs/s*/*/eval/pair_*.json")))
for path in runs:
    seed, arm = path.split("/")[-4], path.split("/")[-3]
    key = f"{arm}/{seed}/{os.path.basename(path)[5:-5]}"
    d, F, B, R, P = load(path); S = len(F)
    out["runs"][key] = dict(arm=arm, seed=seed, pair=os.path.basename(path)[5:-5], preset=d["preset"], n_states=S,
                            curse=curse_curves(F, B, R, P, range(S)), spearman_f1f=spearman_f1f(F, B, P, S),
                            stopping=stopping(F, B, R, P, S), slope=slope_and_recal(F, B, R, P, S))
    c8 = out["runs"][key]["curse"]["8"]["curse"]
    print(key, "S", S, "distinct/64 %.1f" % out["runs"][key]["curse"]["mean_distinct_per64"],
          "curse n8 %s" % (None if c8 is None else round(c8, 4)),
          "f unique", out["runs"][key]["spearman_f1f"]["f_n_unique"], flush=True)

# ---------- probe-set two-seed ensembles (shared uniform-random plans) ----------
probes = {}
for path in sorted(glob.glob(os.path.join(E3, "runs/s*/*/eval/probe_*.json"))):
    seed, arm = path.split("/")[-4], path.split("/")[-3]; ck = os.path.basename(path)[6:-5]
    probes.setdefault((arm, ck), {})[seed] = path
for (arm, ck), by in sorted(probes.items()):
    seeds = sorted(by)
    if len(seeds) < 2:
        continue
    L = {s: load(by[s]) for s in seeds}
    S = min(len(L[s][1]) for s in seeds)
    rows = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            a, b_ = seeds[i], seeds[j]
            Fa = np.stack(L[a][1][:S]); Fb = np.stack(L[b_][1][:S]); Bt = np.stack(L[a][2][:S])
            assert all(L[a][4][k] == L[b_][4][k] for k in range(S)), "probe plans differ"
            dis = np.abs(Fa - Fb); ea = Fa - Bt
            sp = lambda x, y: (float(stats.spearmanr(x, y).statistic) if np.std(x) > 0 and np.std(y) > 0 else None)
            ia = Fa.argmax(1); idx = np.arange(S)
            curse = {}
            for n in NS:
                ga = [argmax_tie_avg(Fa[s, :n], ea[s, :n]) - ea[s, :n].mean() for s in range(S)]
                wco = np.minimum(Fa, Fb)
                gw = [argmax_tie_avg(wco[s, :n], Bt[s, :n]) for s in range(S)]
                gs = [argmax_tie_avg(Fa[s, :n], Bt[s, :n]) for s in range(S)]
                curse[str(n)] = dict(curse_single=float(np.mean(ga)), gold_single=float(np.mean(gs)), gold_wco=float(np.mean(gw)))
            # equal-call comparison at cost c: WCO pays 2 calls per candidate
            eq = {}
            for c in CS:
                best_single = max(np.mean([argmax_tie_avg(Fa[s, :n], Bt[s, :n]) for s in range(S)]) - c * n for n in range(1, 17))
                best_wco = max(np.mean([argmax_tie_avg(np.minimum(Fa, Fb)[s, :n], Bt[s, :n]) for s in range(S)]) - 2 * c * n for n in range(1, 9))
                eq[str(c)] = dict(single_best_n_in_sample=float(best_single), wco_best_n_in_sample=float(best_wco))
            rows[f"{a}-{b_}"] = dict(spearman_disagree_abs_eps=sp(dis.ravel(), np.abs(ea).ravel()),
                                     spearman_disagree_at_argmax_abs=sp(dis[idx, ia], np.abs(ea[idx, ia])),
                                     spearman_disagree_at_argmax_signed=sp(dis[idx, ia], ea[idx, ia]),
                                     corr_Fa_Fb=sp(Fa.ravel(), Fb.ravel()), curves=curse, equal_calls=eq, n_states=S)
    out["probe_ensemble"][f"{arm}/{ck}"] = rows
    print("probe", arm, ck, {k: (v["spearman_disagree_abs_eps"], v["corr_Fa_Fb"]) for k, v in rows.items()}, flush=True)


# ---------- aggregate over seeds ----------
def agg(vals):
    v = [x for x in vals if x is not None]
    return dict(mean=float(np.mean(v)) if v else None, sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n_seeds=len(v), per_seed=v)


A = {}
groups = {}
for k, r in out["runs"].items():
    groups.setdefault((r["arm"], r["pair"]), []).append(r)
for (arm, pair), rs in sorted(groups.items()):
    g = {"curse": {n: agg([r["curse"][n]["curse"] for r in rs]) for n in map(str, NS)},
         "gold": {n: agg([r["curse"][n]["gold"] for r in rs]) for n in map(str, NS)},
         "beta": agg([r["slope"]["beta"] for r in rs]),
         "curse8_gaussian_pred": agg([r["slope"]["curse8_gaussian_pred"] for r in rs]),
         "curse_recal": {n: agg([r["slope"]["curse_recal"][n] for r in rs]) for n in map(str, NS)},
         "spearman_f1f_all": agg([r["spearman_f1f"]["all"] for r in rs]),
         "spearman_f1f_at_argmax_abs": agg([r["spearman_f1f"]["at_argmax_abs"] for r in rs]),
         "stopping": {}}
    for c in map(str, CS):
        g["stopping"][c] = {rule: agg([r["stopping"][c][rule][0] for r in rs]) for rule in rs[0]["stopping"][c] if rule != "n"}
        for rule in ("hedgetune_n", "gap_model_fixed_n", "gap_model_pandora_rel", "raw_pandora_rel", "smith_winkler_bernoulli_sd_fixed_n"):
            g["stopping"][c][f"paired_{rule}_minus_fixed_n"] = agg([r["stopping"][c][rule][0] - r["stopping"][c]["fixed_n"][0] for r in rs])
    A[f"{arm}/{pair}"] = g
out["aggregate"] = A
# fused vs protected at n=8 (curse), per checkpoint pair
fvp = {}
for pair in ("c000_c000", "c100_c100"):
    P_ = A.get(f"P/{pair}");
    for fused in ("B0s", "M0g"):
        Fz = A.get(f"{fused}/{pair}")
        if P_ and Fz and P_["curse"]["8"]["mean"] is not None and Fz["curse"]["8"]["mean"] is not None:
            sds = [x for x in (P_["curse"]["8"]["sd"], Fz["curse"]["8"]["sd"]) if x is not None]
            fvp[f"{fused}-P/{pair}"] = dict(diff=Fz["curse"]["8"]["mean"] - P_["curse"]["8"]["mean"],
                                            max_seed_sd=max(sds) if sds else None,
                                            pooled_seed_sd=float(np.sqrt(np.mean(np.square(sds)))) if sds else None)
out["fused_vs_protected_curse_n8"] = fvp
out["seconds"] = time.time() - t0
json.dump(out, open(os.path.join(HERE, "t1_e3.json"), "w"), indent=1)
for k, g in A.items():
    print(k, "curse", {n: (round(v["mean"], 4) if v["mean"] is not None else None) for n, v in g["curse"].items()},
          "sp_f1f", g["spearman_f1f_all"]["mean"], "beta", round(g["beta"]["mean"], 3),
          "pred8", round(g["curse8_gaussian_pred"]["mean"], 4),
          "recal", {n: (round(v["mean"], 4) if v["mean"] is not None else None) for n, v in g["curse_recal"].items()})
    for c, s in g["stopping"].items():
        print("   c", c, " ".join(f"{rule}={v['mean']:+.4f}±{(v['sd'] or 0):.4f}" for rule, v in s.items() if v["mean"] is not None))
print("fused_vs_protected", fvp)
print("seconds", round(out["seconds"], 1))
