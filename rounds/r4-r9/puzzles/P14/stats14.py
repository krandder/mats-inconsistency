"""P14 analysis: pre-registered tests (PREREG.md), secondary metrics, chart JSON.
Reads results/runs/*_eval_*.json, results/hp.json (BP*, endpoint switch) and results/none_loss_4102.json.
Writes results/summary.json and charts/*.json; prints the verdict table. Usage: stats14.py [--runs DIR]."""
import argparse
import glob
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMARY = ["V", "V-noD", "BP-D", "BP", "V-L1", "BP-L1-cm"]
SECONDARY = ["BP-L0-cm", "BP-L1-D-cm", "DisARM-L1-cm", "STG-L1-cm", "V-W", "V-local", "N-sel", "N0"]
PLANTED = ["P-LOCO", "P-W"]
ALL = PRIMARY + SECONDARY + PLANTED + ["ORACLE"]
SUFF = 0.02
PLANTED_NAMES = ["g2", "g2dup", "g1", "ring", "wallcopy", "random"]


# ------------------------------------------------------------------------------------------ helpers
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    """Newcombe hybrid score interval (method 10) for p1 - p2."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def boot_median(x, B=10000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = np.median(rng.choice(x, (B, len(x))), axis=1)
    return float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else float("nan"), float("nan"), float("nan")
    m, s = x.mean(), x.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return float(m), float(m - t * s), float(m + t * s)


def holm(ps):
    """ps: dict name -> p. Returns name -> adjusted p."""
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def mw_less(x, y):
    """One-sided Mann-Whitney: H1 x tends to be lower than y."""
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    return float(stats.mannwhitneyu(x, y, alternative="less").pvalue)


def van_elteren(strata):
    """strata: list of (x, y) samples. Two-sided van Elteren test (stratified Wilcoxon rank-sum, weights
    1/(N_s + 1)), normal approximation with tie-corrected variances. Positive z: x tends to be larger."""
    num, var = 0.0, 0.0
    for x, y in strata:
        x, y = np.asarray(x, float), np.asarray(y, float)
        n1, n2 = len(x), len(y)
        if n1 == 0 or n2 == 0:
            continue
        N = n1 + n2
        r = stats.rankdata(np.concatenate([x, y]))
        W = r[:n1].sum()
        EW = n1 * (N + 1) / 2
        _, cnt = np.unique(r, return_counts=True)
        tie = (cnt ** 3 - cnt).sum()
        VW = n1 * n2 / 12 * ((N + 1) - tie / (N * (N - 1)))
        w = 1.0 / (N + 1)
        num += w * (W - EW)
        var += w * w * VW
    if var <= 0:
        return float("nan"), float("nan")
    z = num / math.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def logrank_less(tx, ex, ty, ey):
    """One-sided log-rank: H1 group x reaches the event sooner (times tx, event flags ex)."""
    tx, ex, ty, ey = map(np.asarray, (tx, ex, ty, ey))
    times = np.unique(np.concatenate([tx[ex == 1], ty[ey == 1]]))
    O, E, V = 0.0, 0.0, 0.0
    for t in times:
        nx, ny = (tx >= t).sum(), (ty >= t).sum()
        dx, dy = ((tx == t) & (ex == 1)).sum(), ((ty == t) & (ey == 1)).sum()
        n, d = nx + ny, dx + dy
        if n < 2:
            continue
        O += dx
        E += d * nx / n
        V += d * (nx / n) * (1 - nx / n) * (n - d) / (n - 1)
    if V <= 0:
        return float("nan"), float("nan")
    z = (O - E) / math.sqrt(V)
    return float(z), float(stats.norm.sf(z))


def km(times, events, grid):
    """Kaplan-Meier fraction that has reached the event by each grid point (1 - S(t))."""
    times, events = np.asarray(times, float), np.asarray(events, int)
    S, out = 1.0, []
    ut = np.unique(times[events == 1])
    surv = {}
    for t in ut:
        n = (times >= t).sum()
        d = ((times == t) & (events == 1)).sum()
        S *= 1 - d / n
        surv[t] = S
    for g in grid:
        s = 1.0
        for t in ut:
            if t <= g:
                s = surv[t]
        out.append(1 - s)
    return out


def ami(a, b):
    """Adjusted mutual information (arithmetic-mean normalisation, as sklearn's default)."""
    a = np.unique(a, return_inverse=True)[1]
    b = np.unique(b, return_inverse=True)[1]
    n = len(a)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(ct, (a, b), 1)
    ai, bj = ct.sum(1), ct.sum(0)

    def H(c):
        p = c[c > 0] / n
        return float(-(p * np.log(p)).sum())

    nz = ct > 0
    MI = float((ct[nz] / n * np.log(ct[nz] * n / np.outer(ai, bj)[nz])).sum())
    emi = 0.0
    for x in ai:
        for y in bj:
            lo, hi = max(1, int(x + y - n)), int(min(x, y))
            if hi < lo:
                continue
            nij = np.arange(lo, hi + 1, dtype=float)
            lg = (gammaln(x + 1) + gammaln(y + 1) + gammaln(n - x + 1) + gammaln(n - y + 1) - gammaln(n + 1)
                  - gammaln(nij + 1) - gammaln(x - nij + 1) - gammaln(y - nij + 1) - gammaln(n - x - y + nij + 1))
            emi += float((nij / n * np.log(nij * n / (x * y)) * np.exp(lg)).sum())
    ha, hb = H(ai), H(bj)
    den = 0.5 * (ha + hb) - emi
    if abs(den) < 1e-12:
        return 1.0 if abs(MI - emi) < 1e-12 else 0.0
    return float((MI - emi) / den)


# ------------------------------------------------------------------------------------------ loading
def load_runs(rdir, task="reach"):
    runs = defaultdict(dict)
    for f in glob.glob(os.path.join(rdir, f"*_{task}_eval_s*.json")):
        base = os.path.basename(f)[:-5]
        parts = base.split("_")
        if len(parts) != 4:  # tagged runs (smoke, reruns) are not part of the analysis
            continue
        d = json.load(open(f))
        runs[d["arm"]][d["seed"]] = d
    return runs


def gap_or_floor(fin, key, floor):
    g = fin.get(key)
    return floor if g is None else g


def per_seed(runs, arm, floor, view="final"):
    out = {}
    for s, d in sorted(runs.get(arm, {}).items()):
        fin = d["final"] if view == "final" else (d["step6000"] or d["final"])
        out[s] = {
            "gap_1": gap_or_floor(fin, "gap_1", floor), "suff1": bool(fin.get("suff1")),
            "gap_1_s8": gap_or_floor(fin, "gap_1_sample8", floor), "suff1_s8": bool(fin.get("suff1_sample8")),
            "gap_all": gap_or_floor(fin, "gap_all", floor), "suffall": bool(fin.get("suffall")),
            "n_live": fin.get("n_live"), "collapse": bool(fin.get("collapse")),
            "ll9": fin["output"]["9"]["greedy"]["logloss"] if fin.get("output") else None,
            "ll15": fin["output"]["15"]["greedy"]["logloss"] if fin.get("output") and "15" in fin["output"] else None,
            "ll21": fin["output"]["21"]["greedy"]["logloss"] if fin.get("output") and "21" in fin["output"] else None,
            "acc9": fin["output"]["9"]["greedy"]["acc"] if fin.get("output") else None,
            "acc21": fin["output"]["21"]["greedy"]["acc"] if fin.get("output") and "21" in fin["output"] else None,
            "ll9_s8": fin["output"]["9"]["sample8"]["logloss"] if fin.get("output") else None,
            "entropy": fin.get("mean_bit_entropy"), "fires_live": fin.get("fires_per_maze_live"),
            "false_del": d.get("false_delistings"), "delistings": d.get("delistings"),
            "flops": d["flops"], "steps": d["steps"], "wall_s": d["wall_s"],
        }
    return out


def flops_to_suff1(d, B):
    for c in d["checkpoints"]:
        if c.get("suff1"):
            return c["i"] * B / 12.0, 1
    return B, 0


def planted_pass(d):
    """(a) no consequence-defined false delisting; (b) final live set sufficient (joint gap <= 0.02) with
    <= 2 slots; (c) the random (5) and wall-copy (4) slots are not live at the end.
    Pre-registered reading: for (b) and (c) a planted slot counts as live while it is LISTED (the wall copy is
    constant on open cells, so the entropy-filtered live set would pass (c) trivially); the entropy-filtered
    reading is reported alongside as b_open / c_open."""
    fin = d["final"]
    live = fin.get("live") or []
    listed = [j for j, x in enumerate(fin.get("listed") or []) if x]
    a = (d.get("false_delistings") or 0) == 0
    suff = fin.get("gap_all") is not None and fin["gap_all"] <= SUFF
    b = suff and len(listed) <= 2
    c = 4 not in listed and 5 not in listed
    return {"a": bool(a), "b": bool(b), "c": bool(c), "pass": bool(a and b and c), "live": live, "listed": listed,
            "b_open": bool(suff and len(live) <= 2), "c_open": bool(4 not in live and 5 not in live),
            "order": [PLANTED_NAMES[e["slot"]] for e in d.get("market", {}).get("events", [])]}


def planted_spearman(d, contrib):
    """Spearman over the 6 planted slots between the final market value (LOCO: final VoI EMA; wealth: wealth)
    and the slot's joint-probe Shapley contribution."""
    m = d.get("market", {})
    val = m.get("final_voi") if "final_voi" in m else m.get("wealth")
    if val is None or contrib is None:
        return None
    c = [contrib[nm] for nm in PLANTED_NAMES]
    rho = stats.spearmanr(val, c).statistic
    return None if rho is None or math.isnan(rho) else float(rho)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "results", "runs"))
    ap.add_argument("--hp", default=os.path.join(HERE, "results", "hp.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "summary.json"))
    ap.add_argument("--charts", default=os.path.join(HERE, "charts"))
    a = ap.parse_args()
    hp = json.load(open(a.hp)) if os.path.exists(a.hp) else {"BP*": "BP-L1-cm", "endpoint_switch": False}
    bpstar = hp["BP*"]
    switch = bool(hp.get("endpoint_switch"))
    nl = os.path.join(HERE, "results", "none_loss.json")
    floor = json.load(open(nl))["4102"]["reach"]["walls_only_gap"] if os.path.exists(nl) else 0.5614
    runs = load_runs(a.runs)
    S = {arm: per_seed(runs, arm, floor) for arm in ALL}
    S6 = {arm: per_seed(runs, arm, floor, view="step6000") for arm in ALL}
    summ = {"BP*": bpstar, "endpoint_switch": switch, "floor_gap_no_live_bit": floor,
            "n_seeds": {arm: len(S[arm]) for arm in ALL}}

    def vals(arm, key, view=S, seeds=None):
        return [v[key] for s, v in sorted(view[arm].items()) if (seeds is None or s in seeds) and v[key] is not None]

    # ---- per-arm table
    table = {}
    for arm in ALL:
        if not S[arm]:
            continue
        if arm == "ORACLE":
            table[arm] = {"ll9": mean_ci(vals(arm, "ll9")), "ll21": mean_ci(vals(arm, "ll21")),
                          "acc21": mean_ci(vals(arm, "acc21"))}
            continue
        n = len(S[arm])
        k = sum(v["suff1"] for v in S[arm].values())
        k8 = sum(v["suff1_s8"] for v in S[arm].values())
        kall = sum(v["suffall"] for v in S[arm].values())
        table[arm] = {
            "n": n, "gap_1_median": boot_median(vals(arm, "gap_1")),
            "suff1": [k, n, *wilson(k, n)[1:]], "suffall": [kall, n, *wilson(kall, n)[1:]],
            "gap_1_s8_median": boot_median(vals(arm, "gap_1_s8")), "suff1_s8": [k8, n, *wilson(k8, n)[1:]],
            "gap_all_median": boot_median(vals(arm, "gap_all")),
            "n_live_mean": mean_ci(vals(arm, "n_live")), "collapse": sum(v["collapse"] for v in S[arm].values()),
            "ll9": mean_ci(vals(arm, "ll9")), "ll15": mean_ci(vals(arm, "ll15")), "ll21": mean_ci(vals(arm, "ll21")),
            "acc9": mean_ci(vals(arm, "acc9")), "ll9_s8": mean_ci(vals(arm, "ll9_s8")),
            "entropy": mean_ci(vals(arm, "entropy")), "fires_live": mean_ci(vals(arm, "fires_live")),
            "delistings_mean": mean_ci(vals(arm, "delistings")) if vals(arm, "delistings") else None,
            "false_del_runs": sum(1 for v in S[arm].values() if (v["false_del"] or 0) > 0),
            "false_del_median": float(np.median(vals(arm, "false_del"))) if vals(arm, "false_del") else None,
            "steps_mean": float(np.mean(vals(arm, "steps"))), "flops_mean": float(np.mean(vals(arm, "flops"))),
            "wall_min_mean": float(np.mean(vals(arm, "wall_s"))) / 60,
            "step6000": {"gap_1_median": boot_median(vals(arm, "gap_1", S6)),
                         "suff1": [sum(v["suff1"] for v in S6[arm].values()), len(S6[arm])]},
        }
    summ["table"] = table

    # ---- primary family {S1, S2}
    V = vals("V", "gap_1")
    prim = {}
    B = next(iter(runs["V"].values()))["B"] if runs.get("V") else None
    if switch and B:
        tv = [flops_to_suff1(d, B) for d in runs["V"].values()]
        tb = [flops_to_suff1(d, B) for d in runs[bpstar].values()]
        z, p = logrank_less([t for t, _ in tv], [e for _, e in tv], [t for t, _ in tb], [e for _, e in tb])
        prim["S1"] = {"test": "log-rank one-sided, FLOPs to SUFF1 (switch fired)", "z": z, "p": p}
    else:
        prim["S1"] = {"test": "Mann-Whitney one-sided, gap_1 V < BP*", "p": mw_less(V, vals(bpstar, "gap_1"))}
    prim["S2"] = {"test": "Mann-Whitney one-sided, gap_1 V < BP-D", "p": mw_less(V, vals("BP-D", "gap_1"))}
    adj = holm({k: v["p"] for k, v in prim.items() if not math.isnan(v["p"])})
    for k in prim:
        prim[k]["p_holm"] = adj.get(k)
        prim[k]["pass"] = bool(adj.get(k) is not None and adj[k] < 0.05)
    # paired Wilcoxon sensitivity (seed pairs)
    for k, comp in (("S1", bpstar), ("S2", "BP-D")):
        common = sorted(set(S["V"]) & set(S[comp]))
        dv = [S["V"][s]["gap_1"] - S[comp][s]["gap_1"] for s in common]
        if len(common) >= 5 and any(abs(x) > 0 for x in dv):
            prim[k]["wilcoxon_paired_p_less"] = float(stats.wilcoxon(dv, alternative="less").pvalue)
        prim[k]["median_diff"] = float(np.median(dv)) if dv else None
    summ["primary"] = prim

    # ---- attribution family (van Elteren, two-sided, Holm over 3)
    def g(arm):
        return vals(arm, "gap_1")
    attr = {}
    z, p = van_elteren([(g("V-noD"), g("BP")), (g("V"), g("BP-D"))])
    attr["estimator_A"] = {"z_LEG_minus_ST": z, "p": p, "note": "z > 0: LEG has larger gap_1 (worse)"}
    z, p = van_elteren([(g("BP-D"), g("BP")), (g("V"), g("V-noD"))])
    attr["listing_A"] = {"z_on_minus_off": z, "p": p, "note": "z > 0: listing has larger gap_1 (worse)"}
    z, p = van_elteren([(g("BP-D"), g("BP-L1-cm")), (g("V"), g("V-L1"))])
    attr["penalty_B"] = {"z_fee_minus_L1": z, "p": p, "note": "z > 0: slot fee has larger gap_1 (worse)"}
    adj = holm({k: v["p"] for k, v in attr.items() if not math.isnan(v["p"])})
    for k in attr:
        attr[k]["p_holm"] = adj.get(k)
    # descriptive interaction: difference of median differences
    attr["interaction_A_desc"] = (float(np.median(g("V")) - np.median(g("BP-D")))
                                  - float(np.median(g("V-noD")) - np.median(g("BP")))) if all(
        S[x] for x in ("V", "BP-D", "V-noD", "BP")) else None
    attr["interaction_B_desc"] = (float(np.median(g("V")) - np.median(g("BP-D")))
                                  - float(np.median(g("V-L1")) - np.median(g("BP-L1-cm")))) if all(
        S[x] for x in ("V", "BP-D", "V-L1", "BP-L1-cm")) else None
    summ["attribution"] = attr

    # ---- secondary: SUFF1 rate differences (Newcombe) and one-sided Fisher (V higher)
    sec = {}
    kv, nv = table["V"]["suff1"][:2] if "V" in table else (0, 0)
    for comp in [x for x in PRIMARY + SECONDARY if x != "V" and x in table]:
        kc, nc = table[comp]["suff1"][:2]
        d, lo, hi = newcombe(kv, nv, kc, nc)
        p = float(stats.fisher_exact([[kv, nv - kv], [kc, nc - kc]], alternative="greater")[1])
        sec[comp] = {"V_minus": [d, lo, hi], "fisher_p_V_greater": p}
    summ["suff1_vs_V"] = sec

    # ---- S3 non-inferiority
    s3 = {}
    common = sorted(set(S["V"]) & set(S[bpstar]))
    d9 = [S["V"][s]["ll9"] - S[bpstar][s]["ll9"] for s in common]
    s3["ll9_V_minus_BPstar"] = mean_ci(d9)
    common_o = sorted(set(S["V"]) & set(S["ORACLE"]))
    d21 = [S["V"][s]["ll21"] - S["ORACLE"][s]["ll21"] for s in common_o]
    s3["ll21_V_minus_oracle"] = mean_ci(d21)
    s3["pass"] = bool(d9 and d21 and s3["ll9_V_minus_BPstar"][2] <= SUFF and s3["ll21_V_minus_oracle"][2] <= SUFF)
    summ["S3"] = s3

    # ---- S4/S5 planted and money
    pl = {}
    shp = os.path.join(HERE, "results", "planted_shapley_reach.json")
    contrib = json.load(open(shp))["shapley_loss_reduction"] if os.path.exists(shp) else None
    for arm in PLANTED:
        pp = {s: planted_pass(d) for s, d in sorted(runs.get(arm, {}).items())}
        rhos = [planted_spearman(d, contrib) for d in runs.get(arm, {}).values()]
        rhos = [r for r in rhos if r is not None]
        pl[arm] = {"per_seed": pp, "n_pass": sum(v["pass"] for v in pp.values()), "n": len(pp),
                   "a": sum(v["a"] for v in pp.values()), "b": sum(v["b"] for v in pp.values()),
                   "c": sum(v["c"] for v in pp.values()), "b_open": sum(v["b_open"] for v in pp.values()),
                   "c_open": sum(v["c_open"] for v in pp.values()),
                   "spearman_value_vs_shapley": boot_median(rhos) if rhos else None}
    summ["planted"] = pl
    vfd = vals("V", "false_del")
    summ["S4"] = {"P-LOCO_pass": pl["P-LOCO"]["n_pass"], "n": pl["P-LOCO"]["n"],
                  "V_false_del_median": float(np.median(vfd)) if vfd else None,
                  "pass": bool(pl["P-LOCO"]["n"] and pl["P-LOCO"]["n_pass"] >= 8 and vfd and np.median(vfd) == 0)}
    s10 = set(range(10))
    kvw = sum(v["suff1"] for s, v in S["V-W"].items() if s in s10)
    kv10 = sum(v["suff1"] for s, v in S["V"].items() if s in s10)
    summ["S5"] = {"V-W_suff1_seeds0_9": kvw, "V_suff1_seeds0_9": kv10, "P-W_pass": pl["P-W"]["n_pass"],
                  "pass": bool(S["V-W"] and abs(kvw - kv10) <= 2 and pl["P-W"]["n_pass"] >= 8)}

    # ---- falsifiers
    F = {}
    if "V" in table and bpstar in table:
        F["F1"] = bool(not prim["S1"]["pass"] and table["V"]["suff1"][0] / max(1, table["V"]["suff1"][1])
                       <= table[bpstar]["suff1"][0] / max(1, table[bpstar]["suff1"][1]))
        F["F2"] = bool(np.median(V) >= np.median(g("BP-D")) and table["V"]["suff1"][0] / table["V"]["suff1"][1]
                       <= table["BP-D"]["suff1"][0] / table["BP-D"]["suff1"][1])
        F["F3"] = not s3["pass"]
        F["F4"] = bool(table["V"]["collapse"] >= 5)
        F["F5"] = bool(table["V"]["false_del_runs"] >= 3 or (pl["P-LOCO"]["n"] - pl["P-LOCO"]["n_pass"]) >= 3)
        adv_g = np.median(g(bpstar)) - np.median(V)
        adv_g8 = np.median(vals(bpstar, "gap_1_s8")) - np.median(vals("V", "gap_1_s8"))
        adv_r = table["V"]["suff1"][0] / table["V"]["suff1"][1] - table[bpstar]["suff1"][0] / table[bpstar]["suff1"][1]
        adv_r8 = (table["V"]["suff1_s8"][0] / table["V"]["suff1_s8"][1]
                  - table[bpstar]["suff1_s8"][0] / table[bpstar]["suff1_s8"][1])
        F["F6"] = bool((adv_g > 0 and adv_g8 <= 0) or (adv_r > 0 and adv_r8 <= 0))
        F["F6_detail"] = {"median_gap_adv_greedy": float(adv_g), "median_gap_adv_s8": float(adv_g8),
                          "suff1_adv_greedy": float(adv_r), "suff1_adv_s8": float(adv_r8)}
        F["V_local_ge_V"] = bool(np.median(g("V-local")) <= np.median(V)) if S["V-local"] else None
    summ["falsifiers"] = F

    # ---- KM curves for FLOPs to SUFF1 (every non-oracle arm)
    grid = [B * i / 12 for i in range(1, 13)] if B else []
    kmc = {}
    for arm in PRIMARY + SECONDARY:
        if not runs.get(arm):
            continue
        tt = [flops_to_suff1(d, B) for d in runs[arm].values()]
        kmc[arm] = km([t for t, _ in tt], [e for _, e in tt], grid)
    summ["km_suff1"] = {"grid_flops": grid, "curves": kmc}

    # ---- code identity: which g6 code the best bit matches (median NMI)
    ident = {}
    fam = ["g2", "d02", "d12", "ring", "g1", "g0", "dist_class", "onpath", "wall_nbrs"]
    for arm in PRIMARY + SECONDARY:
        rows = []
        for d in runs.get(arm, {}).values():
            fin = d["final"]
            jb = fin.get("best_slot")
            if jb is None or str(jb) not in fin.get("identity", {}):
                continue
            rows.append(fin["identity"][str(jb)])
        if rows:
            ident[arm] = {c: float(np.median([r[c] for r in rows])) for c in fam}
            ident[arm]["nmi_g2_ge"] = {t: int(sum(r["g2"] >= t for r in rows)) for t in (0.7, 0.8, 0.9)}
            ident[arm]["n"] = len(rows)
            ident[arm]["best_code_counts"] = {}
            for r in rows:
                c = max(["g2", "d02", "d12", "ring", "g1", "g0"], key=lambda c: r[c])
                ident[arm]["best_code_counts"][c] = ident[arm]["best_code_counts"].get(c, 0) + 1
    summ["identity"] = ident

    # ---- cross-seed reproducibility (AMI of the live-bit joint partitions on probe-test open cells)
    try:
        import p14
        te = p14.EvalSet(300, 9, 4102)
        om = te.open
        n_cells = te.num * 81
        rep = {}
        for arm in PRIMARY + SECONDARY:
            parts = {}
            for s, d in runs.get(arm, {}).items():
                lb = d["final"].get("live_bits_te") or {}
                if not lb:
                    continue
                lab = np.zeros(int(om.sum()), np.int64)
                for j in sorted(lb, key=int):
                    bits = np.unpackbits(np.array(lb[j], np.uint8))[:n_cells].reshape(te.num, 9, 9)
                    lab = lab * 2 + bits[om].astype(np.int64)
                parts[s] = lab
            vs = [ami(parts[x], parts[y]) for x, y in itertools.combinations(sorted(parts), 2)]
            if vs:
                rep[arm] = {"ami_median": float(np.median(vs)), "pairs": len(vs)}
        summ["reproducibility_ami"] = rep
    except Exception as e:  # noqa: BLE001
        summ["reproducibility_ami"] = {"error": repr(e)}

    # ---- wealth ledgers
    wl = {}
    for arm in ("V-W", "P-W"):
        rows = [d["market"] for d in runs.get(arm, {}).values() if "market" in d]
        if rows:
            wl[arm] = {k: float(np.mean([r[k] for r in rows])) for k in ("subsidies", "minted", "fees", "bankruptcies")}
            wl[arm]["max_rel_conservation_err"] = float(max(r["max_rel_conservation_err"] for r in rows))
    summ["wealth"] = wl

    # ---- delisted-slot identity (V): entropy-normalised NMI with the family
    dq = []
    for d in runs.get("V", {}).values():
        for r in d.get("delisting_quality") or []:
            dq.append(max(r["nmi_min_family"].values()))
    summ["V_delisted_max_nmi_min_family"] = boot_median(dq) if dq else None

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(summ, open(a.out, "w"), indent=1, default=float)
    write_charts(summ, a.charts, bpstar)
    print_report(summ)


def write_charts(summ, cdir, bpstar):
    os.makedirs(cdir, exist_ok=True)
    t = summ["table"]
    arms = [a for a in PRIMARY + SECONDARY if a in t]

    def pts(key, idx=None):
        out = []
        for a in arms:
            v = t[a][key]
            if key.startswith("suff"):
                k, n, lo, hi = v
                y = k / n
                out.append({"x": a + (" (BP*)" if a == bpstar else ""), "y": y, "ci95": max(y - lo, hi - y),
                            "lo": lo, "hi": hi})
            else:
                m, lo, hi = v
                out.append({"x": a + (" (BP*)" if a == bpstar else ""), "y": m, "ci95": max(m - lo, hi - m),
                            "lo": lo, "hi": hi})
        return out
    json.dump({"title": "P14: best live bit's probe gap to the g2 oracle (median over seeds, 95% bootstrap CI)",
               "x_label": "arm", "y_label": "gap_1 (nats/cell; SUFF1 needs <= 0.02)",
               "series": [{"name": "greedy bits", "points": pts("gap_1_median")},
                          {"name": "8 sampled bits", "points": pts("gap_1_s8_median")}]},
              open(os.path.join(cdir, "gap1_by_arm.json"), "w"), indent=1)
    json.dump({"title": "P14: runs ending with one sufficient live bit (SUFF1), 95% Wilson CI",
               "x_label": "arm", "y_label": "fraction of seeds",
               "series": [{"name": "SUFF1 greedy", "points": pts("suff1")},
                          {"name": "SUFF1 8 sampled", "points": pts("suff1_s8")}]},
              open(os.path.join(cdir, "suff1_by_arm.json"), "w"), indent=1)
    k = summ["km_suff1"]
    json.dump({"title": "P14: Kaplan-Meier fraction of runs with SUFF1 by FLOPs (12 checkpoints, censored at B)",
               "x_label": "counted FLOPs", "y_label": "fraction reached SUFF1",
               "series": [{"name": a, "points": [{"x": x, "y": y} for x, y in zip(k["grid_flops"], c)]}
                          for a, c in k["curves"].items()]},
              open(os.path.join(cdir, "km_suff1.json"), "w"), indent=1)


def print_report(s):
    print(f"BP* = {s['BP*']}  endpoint switch = {s['endpoint_switch']}")
    print(f"{'arm':14s} {'n':>3s} {'gap_1 med [95%]':>26s} {'SUFF1':>7s} {'SUFF1 s8':>8s} {'live':>5s} {'ll9':>7s} {'ll21':>7s} {'ent':>5s}")
    for arm, r in s["table"].items():
        if arm == "ORACLE":
            print(f"{'ORACLE':14s} ll9 {r['ll9'][0]:.4f} ll21 {r['ll21'][0]:.4f}")
            continue
        g = r["gap_1_median"]
        print(f"{arm:14s} {r['n']:3d} {g[0]:8.4f} [{g[1]:.4f},{g[2]:.4f}] {r['suff1'][0]:3d}/{r['suff1'][1]:<3d}"
              f" {r['suff1_s8'][0]:3d}/{r['suff1_s8'][1]:<3d} {r['n_live_mean'][0]:5.2f} {r['ll9'][0]:7.4f}"
              f" {r['ll21'][0]:7.4f} {r['entropy'][0]:5.3f}")
    print("primary:", json.dumps(s["primary"], default=float))
    print("attribution:", json.dumps(s["attribution"], default=float))
    print("S3:", json.dumps(s["S3"], default=float))
    print("S4:", json.dumps(s["S4"], default=float), " S5:", json.dumps(s["S5"], default=float))
    print("planted:", {k: (v["n_pass"], v["n"], v["a"], v["b"], v["c"]) for k, v in s["planted"].items()})
    print("falsifiers:", json.dumps(s["falsifiers"], default=float))


if __name__ == "__main__":
    main()
