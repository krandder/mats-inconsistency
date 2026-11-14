"""P1 analysis (CPU): halting rules applied post hoc to the stored pools, bootstrap CIs, criteria.

  python analyze.py --res results/vm/trader-cat-p1-7 --out results --nboot 2000 --procs 5

Writes <out>/p1_results.json (everything) and <out>/p1_chart.json (curves and bars for the write-up).
Point estimates average the per-unit values (unit = backbone seed x head seed on T2m, head seed on mazes).
CIs: bootstrap resamples, each drawing one unit and multinomial example weights; a*, the envelope and
the oracle are recomputed inside every resample.
"""
import argparse
import glob
import json
import os
import time
import numpy as np
from acore import load_pool, obs_view, rule_knobs
from halting import SIMPLE_RULES, HEAD_RULES_T2M, HEAD_RULES_MAZE, stops_for

HORIZONS = [1, 2, 4, 8, 16, 32, 64, "cap"]
HK = [2, 4, 8, 16, 32, 64, "cap"]
HEAD_ARM = {"R4": "R4", "R4c": "R7", "R4e": "R4", "R7": "R7", "R7abl": "R7abl", "R8": "R8", "R9": "R9",
            "R9s": "R9s", "R7TT": "R7TT", "R11": "R11", "R12": "R12", "R12TT": "R12TT", "R13": "R13",
            "R5_30": "R5_30", "R5_100": "R5_100"}
R3S = ["R3sum", "R3min", "R3ent"]
T2M_COMP = SIMPLE_RULES + ["R4", "R4c", "R4e", "R5_30", "R5_100", "R11", "R12", "R13"]
BUDGETS = (0.5, 1.0, 2.0)
VOC_MAZE = ["R7", "R8", "R9", "R9s", "R7TT"]


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [None, None]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def fnum(v):
    v = float(v)
    return v if np.isfinite(v) else None


PROCS = 1          # --procs; the maze bootstrap and calibration blocks run in a fork pool (results do not depend on it)
_CTX = {}


def pmap(f, jobs):
    if PROCS <= 1 or len(jobs) <= 1:
        return [f(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(PROCS, len(jobs))) as pool:
        return pool.map(f, jobs, chunksize=1)


def _ru_job(job):
    units, kind, comp, slices_of = _CTX["ru"]
    i, W = job
    return unit_stats(units[i], W, kind, comp, slices_of(units[i]))


def _cal_job(job):
    O, E, info, hs = _CTX["cal"]
    arm, settle, seed = job
    return calib_block(O, E, info, arm, settle, hs, np.random.default_rng(seed))


def rankavg(x):
    s = np.argsort(x, kind="mergesort")
    xs = x[s]
    _, idx, cnt = np.unique(xs, return_index=True, return_counts=True)
    r = np.empty(len(x))
    r[s] = np.repeat(idx + (cnt - 1) / 2.0, cnt)
    return r


def spearman(a, b):
    ra, rb = rankavg(a), rankavg(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


# ------------------------------------------------------------------ one pool through one head seed
class Unit:
    def __init__(self, O, E, M, info, dev_obs, hseed, rules, metric, grid=1, npix=None):
        self.info, self.Tc, self.hseed = info, int(info["Tc"]), hseed
        ob = obs_view(O, info, hseed)
        self.acc_t = E["S"].astype(float) if metric == "score" else E["exact"].astype(float)
        self.exact_t = E["exact"].astype(float)
        self.N = self.acc_t.shape[0]
        self.M = M
        self.rules = [r for r in rules if r in SIMPLE_RULES or ("h_" + HEAD_ARM[r] in ob.d or
                                                                 "h_R13_bits" in ob.d and r == "R13")]
        self.stops, self.A, self.C, self.Cf, self.X = {}, {}, {}, {}, {}
        mb = info["macs_backbone_per_pos"] if "macs_backbone_per_pos" in info else 1.0
        self.rh = {}
        for r in self.rules:
            kn = rule_knobs(dev_obs, r, self.Tc)
            st = stops_for(ob, r, kn)
            self.stops[r] = st
            self.A[r] = np.take_along_axis(self.acc_t, st, 1)
            self.X[r] = np.take_along_axis(self.exact_t, st, 1)
            rh = 0.0 if r in SIMPLE_RULES else info["macs_head_per_pos"][HEAD_ARM[r]] / mb
            self.rh[r] = rh
            if grid == 1:        # T2m: steps; FLOP-adjusted = steps x (1 + r_head)
                self.C[r] = st.astype(float)
                self.Cf[r] = st * (1.0 + rh)
            else:                # mazes: pixel-steps / npix = t + r_head x floor(t / grid)
                self.C[r] = st + rh * (st // grid)
                self.Cf[r] = self.C[r]
        self.npix = npix


def wmean(W, Mx):
    return (W @ Mx) / W.sum(1, keepdims=True)


def cost_at(ACC, COST, a_star):
    ok = ACC >= a_star[:, None] - 1e-12
    cm = np.where(ok, COST, np.inf)
    k = cm.argmin(1)
    return cm[np.arange(len(k)), k], k


def acc_at_budget(ACC, COST, budget):
    ok = COST <= budget[:, None] + 1e-9
    a = np.where(ok, ACC, -np.inf).max(1)
    return np.where(np.isfinite(a), a, np.nan)


def oracle_t2m(u, W, a_star):
    """Lagrangian oracle on the score: each example stops at argmax_t S_i(t) - lam t; smallest mean t
    whose mean score reaches a* (lam over a log grid)."""
    if not hasattr(u, "_orc"):
        S = u.acc_t[:, 1:]
        t = np.arange(1, u.Tc + 1, dtype=float)
        lams = np.concatenate([[0.0], np.geomspace(1e-5, 10.0, 300)])
        st = np.stack([np.argmax(S - l * t[None], 1) for l in lams], 1)          # (N, nl)
        u._orc = (np.take_along_axis(S, st, 1), st + 1.0)
    Sv, tv = u._orc
    SC, TC = wmean(W, Sv), wmean(W, tv)
    c, _ = cost_at(SC, TC, a_star)
    return c


def oracle_maze(u, W, a_star, cost_of):
    """Cheapest subset of solvable examples, each at its earliest exact step (weighted)."""
    ex = u.exact_t[:, 1:] > 0
    first = np.where(ex.any(1), ex.argmax(1) + 1, -1)
    sol = np.where(first > 0)[0]
    c = cost_of(first[sol].astype(float))
    order = np.argsort(c, kind="mergesort")
    cs, ss = c[order], sol[order]
    Wt = W.sum(1)
    cw = np.cumsum(W[:, ss], 1)
    cc = np.cumsum(W[:, ss] * cs[None], 1)
    need = a_star * Wt
    idx = (cw < need[:, None] - 1e-9).sum(1)
    out = np.full(W.shape[0], np.nan)
    okk = idx < len(ss)
    out[okk] = cc[np.arange(W.shape[0])[okk], idx[okk]] / Wt[okk]
    return out


def unit_stats(u, W, kind, comp=None, slices=()):
    """Everything needed per resample for one unit. W (b, N). Returns dict of (b,) arrays."""
    b = W.shape[0]
    fixed = wmean(W, u.acc_t[:, 1:])
    bf = fixed.max(1)
    a_star = bf - 0.02 * np.abs(bf)      # = 0.98 x best when best > 0
    out = {"a_star": a_star, "best_fixed": fixed.max(1)}
    cost, kst, ACCs, COSTs = {}, {}, {}, {}
    for r in u.rules:
        ACC, COST = wmean(W, u.A[r]), wmean(W, u.C[r])
        c, k = cost_at(ACC, COST, a_star)
        cost[r], kst[r], ACCs[r], COSTs[r] = c, k, ACC, COST
        out["cost:" + r] = c
        out["kstar:" + r] = np.where(np.isfinite(c), k, -1)
        if kind == "t2m":
            cf, _ = cost_at(ACC, wmean(W, u.Cf[r]), a_star)
            out["costf:" + r] = cf
    simple = [r for r in SIMPLE_RULES if r in cost]
    env = np.min(np.stack([cost[r] for r in simple]), 0)
    out["cost:envelope"] = env
    if kind == "t2m":
        orc = oracle_t2m(u, W, a_star)
    else:
        orc = oracle_maze(u, W, a_star, lambda t: t)
    out["cost:oracle"] = orc
    out["headroom"] = 1 - orc / env
    # matched compute: budgets relative to the oracle's mean cost
    for bm in BUDGETS:
        bud = bm * orc
        best_env = np.full(b, -np.inf)
        for r in u.rules:
            a = acc_at_budget(ACCs[r], COSTs[r], bud)
            out[f"acc@{bm}:{r}"] = a
            if r in simple:
                best_env = np.fmax(best_env, a)
        out[f"acc@{bm}:envelope"] = best_env
    # slices at each rule's a* knob
    for sname, m in slices:
        Wm = W * m[None]
        den = Wm.sum(1)
        for r in u.rules:
            k = kst[r]
            st = (Wm @ u.stops[r]) / den[:, None]
            xx = (Wm @ u.X[r]) / den[:, None]
            out[f"steps[{sname}]:{r}"] = st[np.arange(b), k]
            out[f"exact[{sname}]:{r}"] = xx[np.arange(b), k]
    if comp is not None:
        comp(u, out, cost)
    return out


def t2m_comp(u, out, cost):
    comps = [r for r in T2M_COMP if r in cost]
    best = np.min(np.stack([cost[r] for r in comps]), 0)
    out["cost:best_comp"] = best
    for arm in ("R7", "R8", "R7abl"):
        if arm not in cost:
            continue
        out[f"sav:{arm}_vs_best"] = 1 - cost[arm] / best
        out[f"sav:{arm}_vs_envelope"] = 1 - cost[arm] / out["cost:envelope"]
        for x in comps:
            out[f"sav:{arm}_vs_{x}"] = 1 - cost[arm] / cost[x]
    fb = np.min(np.stack([out["costf:" + r] for r in comps]), 0)
    out["savf:R7_vs_best"] = 1 - out["costf:R7"] / fb
    if "R8" in cost:
        out["ratio:R8_over_R7_minus1"] = cost["R8"] / cost["R7"] - 1
    # S2 slices: blocked steps (best R3 variant at a*, by cost) and resolvable exact vs R4
    r3 = np.stack([cost[r] for r in R3S])
    r3best = np.array(R3S)[r3.argmin(0)]
    bl = np.stack([out[f"steps[blocked]:{r}"] for r in R3S])
    out["blocked_steps:R3best"] = bl[r3.argmin(0), np.arange(bl.shape[1])]
    out["diff:blocked_steps_R3best_minus_R7"] = out["blocked_steps:R3best"] - out["steps[blocked]:R7"]
    out["diff:res_exact_R7_minus_R4"] = out["exact[resolvable]:R7"] - out["exact[resolvable]:R4"]
    out["_r3best"] = r3best


def maze_comp(u, out, cost):
    env = out["cost:envelope"]
    for arm in [r for r in HEAD_RULES_MAZE if r in cost]:
        out[f"sav:{arm}_vs_envelope"] = 1 - cost[arm] / env
    for x in ("R12TT", "R11", "R13", "R7TT", "R8"):
        if "R9" in cost and x in cost:
            out[f"sav:R9_vs_{x}"] = 1 - cost["R9"] / cost[x]
    for bm in BUDGETS:
        if "R9" not in cost:
            break
        a9 = out[f"acc@{bm}:R9"]
        out[f"diff@{bm}:R9_minus_envelope"] = a9 - out[f"acc@{bm}:envelope"]
        if "R7TT" in cost:
            out[f"diff@{bm}:R9_minus_R7TT"] = a9 - out[f"acc@{bm}:R7TT"]
        if "R12TT" in cost:
            out[f"diff@{bm}:R12TT_minus_R9"] = out[f"acc@{bm}:R12TT"] - a9
        best_voc = np.max(np.stack([out[f"acc@{bm}:{r}"] for r in VOC_MAZE if r in cost]), 0)
        out[f"diff@{bm}:bestVOC_minus_envelope"] = best_voc - out[f"acc@{bm}:envelope"]


def run_units(units, nboot, rng, kind, comp, slices_of):
    """Point estimates (mean over units, uniform weights) and bootstrap draws (one unit per resample)."""
    pts = []
    for u in units:
        pts.append(unit_stats(u, np.ones((1, u.N)), kind, comp, slices_of(u)))
    assign = rng.integers(len(units), size=nboot)
    jobs = []
    for i, u in enumerate(units):
        nb = int((assign == i).sum())
        if nb == 0:
            continue
        W = rng.multinomial(u.N, np.full(u.N, 1.0 / u.N), size=nb).astype(float)
        jobs += [(i, w) for w in np.array_split(W, max(1, min(PROCS, nb)))]
    _CTX["ru"] = (units, kind, comp, slices_of)
    draws = pmap(_ru_job, jobs)
    keys = [k for k in pts[0] if not k.startswith("_")]
    D = {k: np.concatenate([d[k] for d in draws]).astype(float) for k in keys}
    point = {}
    for k in keys:
        v = np.array([p[k][0] for p in pts], float)
        if k.startswith(("sav", "savf")):      # unreachable a*: 1 - inf/x = -inf -> -10 ; x/inf -> 1
            v, D[k] = np.clip(v, -10, 1), np.clip(D[k], -10, 1)
        point[k] = float(np.nanmean(np.where(np.isfinite(v), v, np.nan))) if np.isfinite(v).any() else float("nan")
    point_inf = {k: bool(any(not np.isfinite(p[k][0]) for p in pts)) for k in keys}
    return point, point_inf, D, pts


def summarize(point, point_inf, D, prefixes):
    res = {}
    for k in point:
        if any(k.startswith(p) for p in prefixes):
            res[k] = {"point": fnum(point[k]), "ci": ci(D[k]), "some_unit_unreached": point_inf[k]}
    return res


def p_one_sided(x):
    x = np.asarray(x, float)
    return float((1 + np.sum(~(x > 0))) / (1 + len(x)))


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def curves_for_chart(u, rules, maxpts=60):
    out = {}
    for r in rules:
        if r not in u.A:
            continue
        acc, cost = u.A[r].mean(0), u.C[r].mean(0)
        o = np.argsort(cost)
        acc, cost = acc[o], cost[o]
        # Pareto hull (max accuracy at <= cost), thinned
        acc = np.maximum.accumulate(acc)
        idx = np.unique(np.linspace(0, len(cost) - 1, min(maxpts, len(cost))).astype(int))
        out[r] = {"cost": [round(float(c), 3) for c in cost[idx]], "acc": [round(float(a), 5) for a in acc[idx]]}
    return out


# ------------------------------------------------------------------ T2m
def analyze_t2m(res_dir, nboot, rng):
    d = os.path.join(res_dir, "t2m")
    bseeds = sorted({int(p.split("_b")[-1].split(".")[0]) for p in glob.glob(os.path.join(d, "obs_L64_b*.npz"))})
    if not bseeds:
        return None
    out = {"bseeds": bseeds, "pools": {}, "gates": {}, "capacity": {}}
    for b in bseeds:
        for gp in glob.glob(os.path.join(d, f"t2m_bb_s{b}*.gate.json")):
            out["gates"][os.path.basename(gp)] = json.load(open(gp))
        cp = os.path.join(d, f"capacity_b{b}.json")
        if os.path.exists(cp):
            out["capacity"][f"b{b}"] = json.load(open(cp))
    dev = {b: [load_pool(d, f"dev64_b{b}"), load_pool(d, f"dev128_b{b}")] for b in bseeds}
    chart = {}
    for pool in ("L64", "L128", "L256"):
        units = []
        for b in bseeds:
            if not os.path.exists(os.path.join(d, f"obs_{pool}_b{b}.npz")):
                continue
            O, E, M, info = load_pool(d, f"{pool}_b{b}")
            for h in info["head_seeds"]:
                dv = [obs_view(x[0], x[3], h) for x in dev[b]]
                units.append(Unit(O, E, M, info, dv, h, SIMPLE_RULES + HEAD_RULES_T2M, "score", grid=1))
        if not units:
            continue
        t0 = time.time()

        def slices_of(u):
            res = u.M["resolvable"].astype(bool)
            return [("blocked", (~res).astype(float)), ("resolvable", res.astype(float))]
        point, pinf, D, pts = run_units(units, nboot, rng, "t2m", t2m_comp, slices_of)
        summ = summarize(point, pinf, D, ("a_star", "best_fixed", "cost", "costf", "sav", "savf", "ratio",
                                           "diff", "headroom", "acc@", "steps[", "exact[", "blocked_steps"))
        # d bins on resolvable instances (point, mean over units, at each unit's a* knob)
        dbins = [(1 / 8, 1 / 4), (1 / 4, 1 / 2), (1 / 2, 3 / 4), (3 / 4, 1.01)]
        dtab = {}
        for lo, hi in dbins:
            row = {}
            for r in units[0].rules:
                vals = []
                for u, p in zip(units, pts):
                    fr = u.M["d"] / float(u.info.get("L", u.Tc))
                    m = u.M["resolvable"].astype(bool) & (fr >= lo) & (fr < hi)
                    k = int(p["kstar:" + r][0])
                    if m.any() and k >= 0:
                        vals.append(u.stops[r][m, k].mean())
                row[r] = fnum(np.mean(vals)) if vals else None
            dtab[f"{lo:.3f}-{min(hi, 1):.3f}"] = row
        # area under accuracy vs log cost between 0.5x and 2x the oracle (point)
        auc = {}
        for r in units[0].rules:
            v = []
            for u, p in zip(units, pts):
                o = p["cost:oracle"][0]
                if not np.isfinite(o):
                    continue
                bud = np.geomspace(0.5 * o, 2 * o, 25)
                ACC, COST = u.A[r].mean(0), u.C[r].mean(0)
                v.append(np.nanmean([acc_at_budget(ACC[None], COST[None], np.array([x]))[0] for x in bud]))
            auc[r] = fnum(np.nanmean(v)) if v else None
        r3b = [p["_r3best"][0] for p in pts]
        out["pools"][pool] = {"n_units": len(units), "N_per_unit": [u.N for u in units], "summary": summ,
                              "d_bins_resolvable_steps": dtab, "auc_logcost_0.5x_2x": auc,
                              "r3best_per_unit": r3b, "r_head": units[0].rh,
                              "p_one_sided": {k: p_one_sided(D[k]) for k in D if k.startswith("sav:R8_vs_best")
                                              or k.startswith("sav:R7_vs_best")},
                              "seconds": round(time.time() - t0, 1)}
        chart[pool] = curves_for_chart(units[0], ["R0", "R2:1:1", "R3ent", "R4", "R4c", "R4e", "R7", "R8",
                                                  "R11", "R12", "R13", "R5_30"])
        chart[pool + "_bars"] = {r: summ.get("cost:" + r) for r in units[0].rules + ["envelope", "oracle"]}
    # criteria
    crit = {}
    P = out["pools"]
    pv = {f"R8:{pl}": P[pl]["p_one_sided"].get("sav:R8_vs_best", 1.0) for pl in ("L64", "L128") if pl in P}
    crit["holm_R8_vs_best"] = holm(pv) if pv else {}
    s2 = {}
    for pl in ("L64", "L128"):
        if pl not in P:
            continue
        S = P[pl]["summary"]
        g = lambda k: S.get(k, {"point": None, "ci": [None, None]})  # noqa: E731
        sv = g("sav:R7_vs_best")
        s2[pl] = {
            "saving_vs_best": sv,
            "saving_ge_20pct_strict": bool(sv["ci"][0] is not None and sv["ci"][0] > 0.20),
            "saving_ge_20pct_lenient": bool(sv["point"] is not None and sv["point"] >= 0.20 and sv["ci"][0] > 0),
            "blocked_fewer_than_R3": g("diff:blocked_steps_R3best_minus_R7"),
            "blocked_ok": bool((g("diff:blocked_steps_R3best_minus_R7")["ci"][0] or -1) > 0),
            "res_exact_above_R4": g("diff:res_exact_R7_minus_R4"),
            "res_exact_ok": bool((g("diff:res_exact_R7_minus_R4")["ci"][0] or -1) > 0),
            "R8_over_R7_minus1": g("ratio:R8_over_R7_minus1"),
            "R8_within_10pct": bool(g("ratio:R8_over_R7_minus1")["ci"][1] is not None and
                                    g("ratio:R8_over_R7_minus1")["ci"][1] < 0.10),
        }
        s2[pl]["pass_strict"] = all(s2[pl][k] for k in ("saving_ge_20pct_strict", "blocked_ok", "res_exact_ok",
                                                         "R8_within_10pct"))
        s2[pl]["pass_lenient"] = all(s2[pl][k] for k in ("saving_ge_20pct_lenient", "blocked_ok", "res_exact_ok",
                                                          "R8_within_10pct"))
    crit["S2"] = s2
    crit["S2_pass"] = bool(s2) and all(v["pass_strict"] for v in s2.values()) and len(s2) == 2
    f2 = {}
    for x in ("R4c", "R4e", "R13", "R11", "R3sum", "R3min", "R3ent"):
        f2[x] = {pl: P[pl]["summary"].get(f"sav:R7_vs_{x}", {}).get("point") for pl in ("L64", "L128") if pl in P}
    within = lambda x: all(v is not None and v < 0.10 for v in f2[x].values()) and len(f2[x]) == 2  # noqa: E731
    crit["F2_savings_R7_vs"] = f2
    crit["F2_nonmyopia_kill"] = bool(within("R4c") or within("R4e"))
    crit["F2_R13_within10"] = within("R13")
    crit["F2_R3_or_R11_within10"] = bool(within("R11") or within("R3sum") or within("R3min") or within("R3ent"))
    out["criteria"] = crit
    return out, chart


# ------------------------------------------------------------------ mazes
def calib_block(O, E, info, arm, settle, hseeds, rng, nres=200):
    """Predicted vs realized totals per horizon on eval decision points t < Tc."""
    ht = np.asarray(info["head_times"])
    Tc = int(info["Tc"])
    keep = ht < Tc
    t = ht[keep]
    S = E["S"].astype(float)
    Sp = E["Sp"].astype(float)
    ent = O["ent"].astype(float)
    K = E["K"].astype(float)[:, keep]
    hl = HK if settle == "K" else HORIZONS
    res = {}
    for j, H in enumerate(hl):
        tH = np.array([Tc if H == "cap" else min(int(x) + H, Tc) for x in t])
        jj = HORIZONS.index(H)
        Kr = K[:, :, jj]
        G = S[:, tH] - S[:, t]
        Pg = Sp[:, tH] - Sp[:, t]
        dent = ent[:, t] - ent[:, tH]
        real = {"K": Kr, "G": G, "P": Pg}[settle]
        per = []
        for s in hseeds:
            key = f"h{s}_{arm}"
            if key not in O:
                continue
            pred = O[key].astype(float)[:, keep, j]
            per.append((pred, real))
        if not per:
            continue

        def stats(pairs_idx=None):
            sp, ratios, suff, mart = [], [], [], []
            for pred, rl in per:
                p, r_ = (pred, rl) if pairs_idx is None else (pred[pairs_idx], rl[pairs_idx])
                pf, rf = p.ravel(), r_.ravel()
                sp.append(spearman(pf, rf))
                edges = np.quantile(pf, [0.2, 0.4, 0.6, 0.8])
                bi = np.searchsorted(edges, pf, side="right")
                ratios.append([pf[bi == q].mean() / rf[bi == q].mean() if rf[bi == q].mean() != 0 else np.nan
                               for q in range(5)])
            return float(np.mean(sp)), np.nanmean(np.array(ratios), 0)
        sp0, rat0 = stats()
        boots_sp, boots_rat = [], []
        N = per[0][0].shape[0]
        for _ in range(nres):
            idx = rng.integers(N, size=N)
            s_, r_ = stats(idx)
            boots_sp.append(s_)
            boots_rat.append(r_)
        boots_rat = np.array(boots_rat)
        # sufficiency / martingale per predicted-K bin (seed 0 bins), and harmful movement share
        pred0 = per[0][0].ravel()
        edges = np.quantile(pred0, [0.2, 0.4, 0.6, 0.8])
        bi = np.searchsorted(edges, pred0, side="right")
        Kf, Gf, Df = Kr.ravel(), G.ravel(), dent.ravel()
        suff = [{"K": float(Kf[bi == q].mean()), "G": float(Gf[bi == q].mean()),
                 "ent_drop": float(Df[bi == q].mean()), "pred": float(pred0[bi == q].mean())} for q in range(5)]
        res[str(H)] = {"spearman": sp0, "spearman_ci": ci(boots_sp),
                       "bin_ratio_pred_over_real": [fnum(x) for x in rat0],
                       "bin_ratio_ci": [ci(boots_rat[:, q]) for q in range(5)],
                       "pred_mean": float(np.mean([p.mean() for p, _ in per])), "real_mean": float(real.mean()),
                       "bins_K_G_entdrop": suff if settle == "K" else None}
    return res


def harmful_share(E, info, rng, nres=200):
    ht = np.asarray(info["head_times"])
    Tc = int(info["Tc"])
    keep = ht < Tc
    t = ht[keep]
    S = E["S"].astype(float)
    K = E["K"].astype(float)[:, keep]
    out = {}
    for H in HK:
        tH = np.array([Tc if H == "cap" else min(int(x) + H, Tc) for x in t])
        Kr = K[:, :, HORIZONS.index(H)]
        G = S[:, tH] - S[:, t]
        neg = (G < 0)
        per_ex_num, per_ex_den = (Kr * neg).sum(1), Kr.sum(1)
        pt = per_ex_num.sum() / max(per_ex_den.sum(), 1e-12)
        bs = []
        N = len(per_ex_den)
        for _ in range(nres):
            i = rng.integers(N, size=N)
            bs.append(per_ex_num[i].sum() / max(per_ex_den[i].sum(), 1e-12))
        out[str(H)] = {"share": float(pt), "ci": ci(bs)}
    return out


def analyze_maze(res_dir, nboot, rng):
    d = os.path.join(res_dir, "maze")
    if not os.path.isdir(d):
        return None, {}
    out = {"validity": {}, "sizes": {}}
    for f in ("repro.json", "headroom_gate.json", "capacity_maze.json"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            out["validity"][f] = json.load(open(p))
    rp = out["validity"].get("repro.json")
    if rp:
        paper = {"13": 0.9988, "59": 0.9730}
        out["validity"]["repro_gate"] = {n: {"paper": paper[n], "ours_best": rp[n]["best"],
                                             "pass": bool(rp[n]["best"] >= paper[n] - 0.03)} for n in paper if n in rp}
    chart = {}
    sizes = sorted(int(p.split("_n")[-1].split(".")[0]) for p in glob.glob(os.path.join(d, "obs_eval_n*.npz")))
    pvals = {}
    for n in sizes:
        O, E, M, info = load_pool(d, f"eval_n{n}")
        Od, Ed, Md, infod = load_pool(d, f"dev_n{n}")
        units = []
        for h in info["head_seeds"]:
            units.append(Unit(O, E, M, info, [obs_view(Od, infod, h)], h, SIMPLE_RULES + HEAD_RULES_MAZE, "exact",
                              grid=int(info["grid"])))
        t0 = time.time()
        Tc = int(info["Tc"])
        exT = E["exact"].astype(bool)
        never = ~exT[:, 1:].any(1)
        kl1, kl2 = O["kl1"][:, Tc].astype(float), O["kl2"][:, Tc].astype(float)
        cyc2 = (kl1 > 1e-3) & (kl2 < 0.1 * kl1)
        over = exT[:, 1:].any(1) & ~exT[:, Tc]

        def slices_of(u, never=never, cyc2=cyc2, over=over):
            return [("never_solved", never.astype(float)), ("two_cycle", cyc2.astype(float)),
                    ("overthinking", over.astype(float)), ("all", np.ones(u.N))]
        point, pinf, D, pts = run_units(units, nboot, rng, "maze", maze_comp, slices_of)
        summ = summarize(point, pinf, D, ("a_star", "best_fixed", "cost", "sav", "diff", "headroom", "acc@",
                                           "steps[", "exact["))
        for arm in ("R8", "R9s", "R7"):
            k = f"sav:{arm}_vs_envelope"
            if k in D:
                pvals[f"{arm}:{n}"] = p_one_sided(D[k])
        hs = info["head_seeds"]
        arms = (("R8", "K"), ("R9", "K"), ("R7", "G"), ("R7TT", "G"), ("R9s", "P"))
        _CTX["cal"] = (O, E, info, hs)
        cal = dict(zip([a for a, _ in arms],
                       pmap(_cal_job, [(a, st, int(rng.integers(2 ** 62))) for a, st in arms])))
        out["sizes"][str(n)] = {"N": units[0].N, "Tc": Tc, "npix": int(M["npix"][0]) if "npix" in M else None,
                                "slice_sizes": {"never_solved": int(never.sum()), "two_cycle": int(cyc2.sum()),
                                                "overthinking": int(over.sum())},
                                "summary": summ, "calibration": cal, "harmful_K_share": harmful_share(E, info, rng),
                                "r_head": units[0].rh, "seconds": round(time.time() - t0, 1)}
        chart[f"maze{n}"] = curves_for_chart(units[0], ["R0", "R2:2:1", "R3ent", "R4", "R7", "R8", "R9", "R7TT",
                                                        "R11", "R12TT", "R13"])
        chart[f"maze{n}_bars"] = {r: summ.get("cost:" + r) for r in units[0].rules + ["envelope", "oracle"]}
    out["holm_secondary_vs_envelope"] = holm(pvals) if pvals else {}
    # S3 / F3 / F4 at 59 and 99
    crit = {}
    for n in ("59", "99"):
        if n not in out["sizes"]:
            continue
        Z = out["sizes"][n]
        c9 = Z["calibration"].get("R9", {})
        sp_ok = all(v["spearman_ci"][0] is not None and v["spearman_ci"][0] >= 0.8 for v in c9.values()) and bool(c9)
        bins_ok = all(all(c is not None and c[0] is not None and 0.8 <= c[0] and c[1] <= 1.25
                          for c in v["bin_ratio_ci"]) for v in c9.values()) and bool(c9)
        harm_ok = all(v["ci"][1] is not None and v["ci"][1] < 0.20 for v in Z["harmful_K_share"].values())
        S = Z["summary"]
        r7tt_ok = all((S.get(f"diff@{b}:R9_minus_R7TT", {}).get("ci", [None])[0] or -1) > -0.02 for b in BUDGETS)
        crit[n] = {"S3_spearman_ge_0.8": sp_ok, "S3_bins_in_0.8_1.25": bins_ok, "S3_harmful_lt_20pct": harm_ok,
                   "S3_within_2pp_of_R7TT": r7tt_ok,
                   "S3_pass": bool(sp_ok and bins_ok and harm_ok and r7tt_ok),
                   "F3_R9_5pp_worse_than_R7TT": any((S.get(f"diff@{b}:R9_minus_R7TT", {}).get("point") or 0) < -0.05
                                                    for b in BUDGETS),
                   "F3_spearman_below_0.5_any_H": any(v["spearman"] < 0.5 for v in c9.values()),
                   "F3_bin_off_2x": any(any(r is not None and (r > 2 or r < 0.5) for r in v["bin_ratio_pred_over_real"])
                                        for v in c9.values()),
                   "F3_harmful_ge_20pct": any(v["share"] >= 0.20 for v in Z["harmful_K_share"].values()),
                   "F4_R12TT_within_2pp_all_budgets": all(abs(S.get(f"diff@{b}:R12TT_minus_R9", {}).get("point") or 9)
                                                          <= 0.02 for b in BUDGETS),
                   "F4_R12TT_cost_within_10pct": abs(S.get("sav:R9_vs_R12TT", {}).get("point") or 9) < 0.10}
        crit[n]["F4_kill"] = bool(crit[n]["F4_R12TT_within_2pp_all_budgets"] and crit[n]["F4_R12TT_cost_within_10pct"])
        crit[n]["F1_voc_beats_env_2pp"] = any((S.get(f"diff@{b}:bestVOC_minus_envelope", {}).get("point") or 0) > 0.02
                                              for b in BUDGETS)
        crit[n]["F1_voc_beats_env_10pct_cost"] = any((S.get(f"sav:{a}_vs_envelope", {}).get("point") or 0) > 0.10
                                                     for a in VOC_MAZE)
    crit["S3_pass"] = bool(crit) and all(crit[n]["S3_pass"] for n in ("59", "99") if n in crit) and \
        all(n in crit for n in ("59", "99"))
    out["criteria"] = crit
    return out, chart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/vm/trader-cat-p1-7")
    ap.add_argument("--out", default="results")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="t2m,maze")
    ap.add_argument("--procs", type=int, default=1, help="worker processes for the maze blocks (<= 6)")
    a = ap.parse_args()
    global PROCS
    PROCS = max(1, min(6, a.procs))
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    R, C = {"nboot": a.nboot, "res": os.path.abspath(a.res)}, {}
    if "t2m" in a.only:
        r = analyze_t2m(a.res, a.nboot, rng)
        if r:
            R["t2m"], C["t2m"] = r
    if "maze" in a.only:
        r, c = analyze_maze(a.res, a.nboot, rng)
        if r:
            R["maze"], C["maze"] = r, c
    json.dump(R, open(os.path.join(a.out, "p1_results.json"), "w"), indent=1, default=lambda o: o.tolist()
              if hasattr(o, "tolist") else str(o))
    json.dump(C, open(os.path.join(a.out, "p1_chart.json"), "w"), default=lambda o: o.tolist()
              if hasattr(o, "tolist") else str(o))
    print("wrote", os.path.join(a.out, "p1_results.json"))


if __name__ == "__main__":
    main()
