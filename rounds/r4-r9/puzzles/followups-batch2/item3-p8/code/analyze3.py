"""Item 3 analysis (PREREG.md). python analyze3.py RESULTS_DIR [P8_RESULTS_DIR] -> RESULTS_DIR/../item3_summary.json.
Uses every seed whose price, truth (2 replicates) and B2 files exist (PREREG contingency: >= 3 seeds, taken in order)."""
import json, math, os, sys

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, t as tdist

POOL = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "m2", "m4", "m6", "m8", "nz05", "nz20", "d2", "d4", "d6", "reach"]
CTX = {"empty": [], "g2g4": ["g2", "g4"]}
ITEMS = [f"{c}:{z}" for c, S in CTX.items() for z in POOL if z not in S]
LAM_PAR, LAM_ORACLE = 0.0063, 0.0631  # fixed before the run (P8 parity fit; P8 maze in-sample argmax at N=1024)
PRIMARY = ["R4", "R8"]
MARGIN = 0.05
RULES = ["R1", "R2", "R3", "R4", "R5", "R7b", "R8", "R8o"]


def tau(a, b):
    v = kendalltau(a, b, variant="b").statistic
    return float(v) if v == v else 0.0


def tci(xs, level):
    xs = np.asarray(xs, float)
    n = len(xs)
    m = float(xs.mean())
    if n < 2:
        return dict(mean=m, lo=None, hi=None, n=n)
    h = float(tdist.ppf(1 - (1 - level) / 2, n - 1) * xs.std(ddof=1) / math.sqrt(n))
    return dict(mean=m, lo=m - h, hi=m + h, half=h, sd=float(xs.std(ddof=1)), n=n)


def load(R, n):
    p = os.path.join(R, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def rule_prices(tab, its, rule):
    if rule == "R2":  # P8 convention: learnability among items with positive importance
        lo = min(tab[i]["R2"] for i in its) - 1e9
        return [tab[i]["R2"] if tab[i]["R1"] > 0 else lo for i in its]
    if rule == "R8":
        return [tab[i]["R1"] - LAM_PAR * tab[i]["L_Z"] for i in its]
    if rule == "R8o":
        return [tab[i]["R1"] - LAM_ORACLE * tab[i]["L_Z"] for i in its]
    return [tab[i][rule] if tab[i].get(rule) is not None else np.nan for i in its]


def table(p, r7):
    tab = {}
    for i in ITEMS:
        v = p["prices"][i]
        tab[i] = {k: v[k] for k in ("L_Z", "R1", "R2", "R3", "R4", "R5")}
        tab[i]["R7b"] = r7["prices"][i]["R7b"] if r7 else None
    return tab


def perm_test(pairs, n=4000, seed=0):
    """P8 perm_test: pairs = [(prices_a, prices_b, truth)] per seed; swap the two rules' within-seed ranks per item.
    Returns the observed mean tau difference a - b and one-sided p for a > b."""
    rng = np.random.default_rng(seed)
    data = [(rankdata(a), rankdata(b), np.asarray(t)) for a, b, t in pairs]
    stat = lambda d: np.mean([tau(a, t) - tau(b, t) for a, b, t in d])
    obs = stat(data)
    cnt = 0
    for _ in range(n):
        pp = []
        for a, b, t in data:
            m = rng.random(len(a)) < 0.5
            pp.append((np.where(m, b, a), np.where(m, a, b), t))
        cnt += stat(pp) >= obs - 1e-12
    return dict(diff=float(obs), p_one_sided=(cnt + 1) / (n + 1))


def main():
    R = sys.argv[1]
    P8R = sys.argv[2] if len(sys.argv) > 2 else None
    seeds, tabs, truth, ctl_truth, reps, pe, b2 = [], {}, {}, {}, {}, {}, {}
    for s in range(100):
        p, r7 = load(R, f"price_maze_s{s}"), load(R, f"r7b_maze_s{s}")
        t0, t1, bb = load(R, f"truth_maze_s{s}_r0"), load(R, f"truth_maze_s{s}_r1"), load(R, f"b2_s{s}")
        if s == 99 or not (p and t0 and t1 and bb):
            continue
        seeds.append(s)
        tabs[s] = table(p, r7)
        reps[s] = [[t["items"][i]["held"]["gain_vs_base"] for i in ITEMS] for t in (t0, t1)]
        truth[s] = dict(zip(ITEMS, np.mean(reps[s], 0)))
        ctl_truth[s] = {i: np.mean([t["items"][i]["held"]["truth"] for t in (t0, t1)]) for i in ITEMS}
        pe[s] = {k: np.mean([np.asarray(t["held_per_example"][k]) for t in (t0, t1)], 0) for k in t0["held_per_example"]}
        b2[s] = bb["B2"]["held"]["acc"]
    out = dict(seeds=seeds, n=len(seeds), lam_par=LAM_PAR, lam_oracle=LAM_ORACLE, margin=MARGIN)
    if len(seeds) < 3:
        out["status"] = "Problem: fewer than 3 complete seeds"
        print(json.dumps(out))
        return
    tv = lambda s: [truth[s][i] for i in ITEMS]
    per_seed = {s: {r: tau(rule_prices(tabs[s], ITEMS, r), tv(s)) for r in RULES
                    if not np.isnan(rule_prices(tabs[s], ITEMS, r)).any()} for s in seeds}
    out["tau_per_seed"] = per_seed
    have = {r: [s for s in seeds if r in per_seed[s]] for r in RULES}  # R7b may be missing for late seeds
    out["tau"] = {r: dict(tci([per_seed[s][r] for s in have[r]], 0.95), seeds=have[r]) for r in RULES if len(have[r]) >= 2}
    # primary: paired differences vs R1 over seeds, 97.5% t-CI (Bonferroni over the two contrasts)
    prim = {}
    for r in PRIMARY:
        d = [per_seed[s][r] - per_seed[s]["R1"] for s in seeds]
        prim[r] = dict(per_seed=d, ci975=tci(d, 0.975), ci95=tci(d, 0.95),
                       perm_vs_R1=perm_test([(rule_prices(tabs[s], ITEMS, r), rule_prices(tabs[s], ITEMS, "R1"), tv(s))
                                             for s in seeds]),
                       perm_R1_vs=perm_test([(rule_prices(tabs[s], ITEMS, "R1"), rule_prices(tabs[s], ITEMS, r), tv(s))
                                             for s in seeds]))
    out["primary"] = prim
    gain = [r for r in PRIMARY if prim[r]["ci975"]["mean"] >= MARGIN and prim[r]["ci975"]["lo"] > 0]
    nogain = all(prim[r]["ci975"]["hi"] < MARGIN for r in PRIMARY)
    out["status"] = "Gain" if gain else ("No gain" if nogain else "Informative")
    out["gain_rules"] = gain
    out["R4_hurts"] = bool(prim["R4"]["ci975"]["hi"] < 0)
    # pre-registered noisy-copy check (empty context): d = bits(nz05 pipeline) - bits(g2 pipeline) per held maze
    nc = {}
    for s in seeds:
        d = pe[s]["empty:nz05"] - pe[s]["empty:g2"]
        h = 1.96 * d.std(ddof=1) / math.sqrt(len(d))
        right = truth[s]["empty:g2"] > truth[s]["empty:nz05"]
        nc[s] = dict(gain_g2=truth[s]["empty:g2"], gain_nz05=truth[s]["empty:nz05"], g2_minus_nz05=float(d.mean()),
                     ci95=float(h), rules_g2_above_nz05={r: bool(rule_prices(tabs[s], ["empty:g2"], r)[0] >
                                                               rule_prices(tabs[s], ["empty:nz05"], r)[0])
                                                          for r in RULES if r in per_seed[s]},
                     truth_g2_above=bool(right))
    ms = tci([nc[s]["g2_minus_nz05"] for s in seeds], 0.95)
    out["noisy_copy"] = dict(per_seed=nc, across=ms,
                             verdict="g2 > nz05" if ms["lo"] > 0 else ("nz05 > g2" if ms["hi"] < 0 else "tie"),
                             rule_correct_share={r: float(np.mean([nc[s]["rules_g2_above_nz05"][r] == nc[s]["truth_g2_above"]
                                                                   for s in have[r]])) for r in RULES
                                                 if have[r]})
    # gates / validity context
    out["truth_reliability_rho"] = {s: float(spearmanr(*reps[s]).statistic) for s in seeds}
    out["b2_acc"] = b2
    out["truth_items"] = {i: tci([truth[s][i] for s in seeds], 0.95) for i in ITEMS}
    # secondary / exploratory
    sec = {}
    sec["per_context"] = {c: {r: tci([tau(rule_prices(tabs[s], [i for i in ITEMS if i.startswith(c + ":")], r),
                                          [truth[s][i] for i in ITEMS if i.startswith(c + ":")]) for s in have[r]], 0.95)
                              for r in RULES if len(have[r]) >= 2} for c in CTX}
    grid = np.concatenate([[0], np.logspace(-3, 1, 41)])
    curve = [float(np.mean([tau([tabs[s][i]["R1"] - lam * tabs[s][i]["L_Z"] for i in ITEMS], tv(s)) for s in seeds]))
             for lam in grid]
    j = int(np.argmax(curve))
    sec["lambda_curve_in_sample"] = dict(grid=grid.tolist(), tau=curve, argmax=float(grid[j]), tau_max=curve[j])
    sec["control_truth_tau"] = {r: tci([tau(rule_prices(tabs[s], ITEMS, r), [ctl_truth[s][i] for i in ITEMS])
                                        for s in have[r]], 0.95) for r in RULES if len(have[r]) >= 2}
    if P8R:
        s3 = {}
        for s in seeds:
            p = load(P8R, f"price_maze_s{s}")
            if p is None:
                continue
            tb = table(p, load(P8R, f"r7b_maze_s{s}"))
            s3[s] = {r: tau(rule_prices(tb, ITEMS, r), tv(s)) for r in RULES}
        if s3:
            ks = sorted(s3)
            sec["p8_n1024_prices_vs_n4096_truth"] = dict(
                seeds=ks, tau={r: tci([s3[s][r] for s in ks], 0.95) for r in RULES},
                R4_minus_R1=tci([s3[s]["R4"] - s3[s]["R1"] for s in ks], 0.95),
                R8_minus_R1=tci([s3[s]["R8"] - s3[s]["R1"] for s in ks], 0.95))
    out["secondary"] = sec
    path = os.path.join(os.path.dirname(os.path.abspath(R)), "item3_summary.json")
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("status", out["status"], "seeds", seeds)
    for r, v in out["tau"].items():
        print(f"tau {r:4s} {v['mean']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]")
    for r in PRIMARY:
        c = prim[r]["ci975"]
        print(f"{r}-R1 {c['mean']:+.3f} 97.5% [{c['lo']:+.3f}, {c['hi']:+.3f}] perm p(R>R1)={prim[r]['perm_vs_R1']['p_one_sided']:.4f}"
              f" p(R1>R)={prim[r]['perm_R1_vs']['p_one_sided']:.4f}")
    print("noisy copy", out["noisy_copy"]["verdict"], out["noisy_copy"]["across"])
    print("wrote", path)


if __name__ == "__main__":
    main()
