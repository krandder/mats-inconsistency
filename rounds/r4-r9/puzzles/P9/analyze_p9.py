"""P9 analysis: reads results/ JSONs, applies the spec bars, writes results/summary_<stage>.json and results/charts.json.

    python analyze_p9.py [p9a|gate|aa|p9bT|p9bE|all]
Intervals: t over seeds (95%, or 90% for TOST); paired differences are per seed (same seed = same data/RNG streams).
"""
import glob
import json
import os
import sys

import numpy as np

import p9util as U

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def J(path):
    with open(path) as f:
        return json.load(f)


def seeds_of(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        b = os.path.basename(f)[:-5]
        arm, s = b.rsplit("_s", 1)
        if not s.isdigit():
            continue
        out.setdefault(arm, {})[int(s)] = J(f)
    return out


def rnd(o, k=4):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float):
        return round(o, k)
    if isinstance(o, dict):
        return {a: rnd(b, k) for a, b in o.items()}
    if isinstance(o, list):
        return [rnd(b, k) for b in o]
    return o


# ------------------------------------------------------------------ P9a
def p9a(sub="p9a", tag=""):
    R = seeds_of(os.path.join(RES, sub, "*.json"))
    out = {"stage": "P9a", "source": f"results/{sub}", "arms": {}, "bars": {}}
    M = ("kl", "voi_spearman", "kendall_median", "sign_acc", "brier")
    for arm, byseed in R.items():
        out["arms"][arm] = {sz: {m: U.tci([byseed[s]["eval"][sz][m] for s in sorted(byseed)]) for m in M}
                            for sz in ("9", "15")}
        out["arms"][arm]["seeds"] = sorted(byseed)
    x1 = R.get("X1", {}).get(0)
    b = out["bars"]
    if x1:
        b["X1_selftest"] = all(abs(x1["eval"][sz][m] - (0.0 if m in ("kl", "brier") else 1.0)) < 1e-5  # KL floor 6.7e-7 from the 1e-6 clip
                               for sz in ("9", "15") for m in M)

    def arm_pass(arm):
        if arm not in R:
            return None
        e = out["arms"][arm]["9"]
        return {"kl<=0.02": e["kl"]["mean"] <= 0.02, "voi_spearman>=0.9": e["voi_spearman"]["mean"] >= 0.9,
                "kendall>=0.67": e["kendall_median"]["mean"] >= 0.67, "sign_acc>=0.9": e["sign_acc"]["mean"] >= 0.9}

    for arm in ("X3", "X2", "X2p", "X0", "X4"):
        pa = arm_pass(arm)
        if pa is not None:
            b[f"{arm}_bars"] = pa
            b[f"{arm}_all"] = all(pa.values())
    if "X0" in R and "X4" in R:
        b["sanity_X0_X4_fail"] = (not b["X0_all"]) and (not b["X4_all"])
    # TOST: paired by seed at matched settled examples
    if "X3" in R and "X2" in R:
        common = sorted(set(R["X3"]) & set(R["X2"]))
        for sz in ("9", "15"):
            d = [R["X3"][s]["eval"][sz]["kl"] - R["X2"][s]["eval"][sz]["kl"] for s in common]
            ci90 = U.tci(d, level=0.90)
            ci95 = U.tci(d)
            b[f"tost_{sz}"] = {"diff_kl_X3_minus_X2": ci90, "ci95": ci95, "band": 0.005,
                               "equivalent": ci90["lo"] is not None and ci90["lo"] >= -0.005 and ci90["hi"] <= 0.005,
                               "X3_not_worse_by_0.005": ci90["hi"] is not None and ci90["hi"] <= 0.005}
        w2 = [R["X3"][s]["is_diag"]["w2_per_drawn"] for s in R["X3"]]
        b["w2"] = {"per_seed": w2, "analytic": 5.0625,
                   "within_10pct": all(abs(w / 5.0625 - 1) <= 0.10 for w in w2)}
        b["settle_rate"] = [R["X3"][s]["is_diag"]["settle_rate"] for s in R["X3"]]
        b["ess_ratio"] = [R["X3"][s]["is_diag"]["ess_ratio"] for s in R["X3"]]
    # falsifiers
    f = {}
    if "X3" in R and "X2" in R:
        f["D1_w2_off_by_10pct"] = not b["w2"]["within_10pct"]
        f["void_efficiency (X3 misses KL 0.02 while X2 meets it)"] = (not b["X3_bars"]["kl<=0.02"]) and b["X2_bars"]["kl<=0.02"]
        if "X2p" in R:
            f["D1_finite_capacity (X2p misses while X2, X3 pass)"] = (not b["X2p_bars"]["kl<=0.02"]) and \
                b["X2_bars"]["kl<=0.02"] and b["X3_bars"]["kl<=0.02"]
        f["few_pass_readout (VOI spearman < 0.8 or sign < 0.75 for X2 or X3)"] = any(
            out["arms"][a]["9"]["voi_spearman"]["mean"] < 0.8 or out["arms"][a]["9"]["sign_acc"]["mean"] < 0.75
            for a in ("X2", "X3"))
    out["falsifiers_triggered"] = f
    return rnd(out)


# ------------------------------------------------------------------ gate
def gate(sub=""):
    base = os.path.join(RES, sub) if sub else RES
    G = seeds_of(os.path.join(base, "eval", "gate_*.json"))
    G = {a.replace("gate_", ""): v for a, v in G.items()}
    out = {"stage": "gate", "source": base.replace(HERE + "/", ""), "arms": {}}
    for arm, byseed in G.items():
        ss = sorted(byseed)
        out["arms"][arm] = {"seeds": ss}
        for dset, Ts in (("dev9", ("T20", "T40")), ("dev25", ("T50", "T100"))):
            for T in Ts:
                out["arms"][arm][f"{dset}_{T}"] = {
                    "em": U.tci([byseed[s][dset][T]["em"] for s in ss]),
                    "cell_acc": U.tci([byseed[s][dset][T]["cell_acc"] for s in ss]),
                    "per_seed_em": [byseed[s][dset][T]["em"] for s in ss]}
    if "N1" in G and len(G["N1"]) > 3:  # the gate decision uses the 3 pre-registered N1 seeds; all seeds shown above
        ss = [s for s in sorted(G["N1"]) if s < 3]
        out["arms"]["N1_gate3"] = {"seeds": ss}
        for dset, Ts in (("dev9", ("T20", "T40")), ("dev25", ("T50", "T100"))):
            for T in Ts:
                out["arms"]["N1_gate3"][f"{dset}_{T}"] = {
                    "em": U.tci([G["N1"][s][dset][T]["em"] for s in ss]),
                    "cell_acc": U.tci([G["N1"][s][dset][T]["cell_acc"] for s in ss]),
                    "per_seed_em": [G["N1"][s][dset][T]["em"] for s in ss]}
    if "N1" in out["arms"]:
        a = out["arms"]["N1_gate3" if "N1_gate3" in out["arms"] else "N1"]
        out["N1_gate_9x9"] = {"cell_acc>=0.98": a["dev9_T20"]["cell_acc"]["mean"] >= 0.98,
                              "em>=0.80": a["dev9_T20"]["em"]["mean"] >= 0.80}
        out["N1_gate_pass"] = all(out["N1_gate_9x9"].values())
        sd = a["dev25_T100"]["em"]["sd"]
        out["power"] = {"sigma_pts_N1_25x25": None if sd is None else 100 * sd,
                        "seeds": 5 if sd is not None and 100 * sd > 2 else 3,
                        "bar_i_underpowered": sd is not None and 100 * sd > 4}
        hr = {}
        for arm in ("N0", "N1"):
            if arm in out["arms"]:
                m = out["arms"]["N1_gate3" if arm == "N1" and "N1_gate3" in out["arms"] else arm]["dev25_T100"]["em"]["mean"]
                hr[arm] = {"em": m, "in_[0.05,0.90]": 0.05 <= m <= 0.90}
        out["headroom"] = hr
        out["headroom_pass"] = all(v["in_[0.05,0.90]"] for v in hr.values())
    return rnd(out)


# ------------------------------------------------------------------ P9b-T (test JSONs)
def test_table():
    Rt = seeds_of(os.path.join(RES, "eval", "test_*.json"))
    return {a.replace("test_", ""): v for a, v in Rt.items()}


def em33(r, T="T150"):
    return r["test33"][T]["em"]


def paired(Rt, a, b, f):
    common = sorted(set(Rt.get(a, {})) & set(Rt.get(b, {})))
    d = [100 * (f(Rt[a][s]) - f(Rt[b][s])) for s in common]
    return {"pts": U.tci(d), "per_seed": d, "seeds": common}


def aa():
    Rt = test_table()
    G = seeds_of(os.path.join(RES, "eval", "gate_*.json"))
    G = {a.replace("gate_", ""): v for a, v in G.items()}
    out = {"stage": "A/A (N2 clamp-and-void vs N4' reveal-true 0.25 x2)"}
    if "N2" in Rt and "N4p" in Rt:
        d_em = 100 * (em33(Rt["N2"][0]) - em33(Rt["N4p"][0]))
        d_acc = Rt["N2"][0]["test33"]["T150"]["cell_acc"] - Rt["N4p"][0]["test33"]["T150"]["cell_acc"]
        sig = 100 * np.std([em33(Rt["N1"][s]) for s in Rt.get("N1", {})], ddof=1) if len(Rt.get("N1", {})) > 1 else 0
        band = max(3.0, 2 * sig)
        out.update({"em33_T150": {"N2": em33(Rt["N2"][0]), "N4p": em33(Rt["N4p"][0])}, "diff_pts": d_em,
                    "band_pts": band, "sigma_pts_N1_test33": sig, "diff_cell_acc": d_acc,
                    "pass": abs(d_em) <= band and abs(d_acc) <= 0.02})
    for a in ("N2", "N4p"):
        if a in G:
            out[f"gate_{a}"] = {"dev9_T20_em": G[a][0]["dev9"]["T20"]["em"], "dev25_T100_em": G[a][0]["dev25"]["T100"]["em"]}
    tr = {}
    for a in ("N2", "N4p"):
        f = os.path.join(RES, "train", f"{a}_s0.json")
        if os.path.exists(f):
            tr[a] = J(f)["acct"]["clamp"]
    out["clamp_counts"] = tr
    out["all_sizes_T150"] = {a: {n: {"em": Rt[a][0][f"test{n}"]["T150"]["em"], "cell_acc": Rt[a][0][f"test{n}"]["T150"]["cell_acc"]}
                                 for n in (9, 15, 21, 33)} for a in ("N2", "N4p") if a in Rt}
    # A/A debug (plans/aa2.json): N4' seed 0 rerun with fresh scratch, and a second pair (N2, N4') at seed 1
    d2 = os.path.join(RES, "aa2", "eval")
    if os.path.isdir(d2):
        R2 = {a.replace("test_", ""): v for a, v in seeds_of(os.path.join(d2, "test_*.json")).items()}
        dbg = {}

        def pair(ra, rb):
            return {"em33_T150": [em33(ra), em33(rb)], "diff_em33_pts": 100 * (em33(ra) - em33(rb)),
                    "diff_cell_acc33": ra["test33"]["T150"]["cell_acc"] - rb["test33"]["T150"]["cell_acc"],
                    "diff_by_size": {n: {"em_pts": 100 * (ra[f"test{n}"]["T150"]["em"] - rb[f"test{n}"]["T150"]["em"]),
                                         "cell_acc": ra[f"test{n}"]["T150"]["cell_acc"] - rb[f"test{n}"]["T150"]["cell_acc"]}
                                     for n in (9, 15, 21, 33)}}
        if 0 in R2.get("N4p", {}) and "N2" in Rt:
            dbg["seed0: N2 - N4' rerun"] = pair(Rt["N2"][0], R2["N4p"][0])
        for s_ in (1, 2, 3):
            if s_ in R2.get("N4p", {}) and s_ in R2.get("N2", {}):
                dbg[f"seed{s_}: N2 - N4'"] = pair(R2["N2"][s_], R2["N4p"][s_])
        band = out.get("band_pts", 3.0)
        for k, v in dbg.items():
            v["pass"] = abs(v["diff_em33_pts"]) <= band and abs(v["diff_cell_acc33"]) <= 0.02
        # replicate noise: the same arm and seed trained twice (GPU nondeterminism only)
        if 0 in R2.get("N4p", {}) and "N4p" in Rt:
            dbg["replicate: N4' s0 original - rerun"] = pair(Rt["N4p"][0], R2["N4p"][0])
        out["debug_aa2"] = dbg
        # paired mean over seeds (declared deviation: the pre-registered A/A is one seed). "fresh" = seeds 1-3 from
        # the debug runs; "all" adds seed 0 with the original N4' (the pre-registered pair).
        pm = {}
        fresh = [s_ for s_ in (1, 2, 3) if s_ in R2.get("N4p", {}) and s_ in R2.get("N2", {})]
        pairs_fresh = [(R2["N2"][s_], R2["N4p"][s_]) for s_ in fresh]
        sets = {"fresh seeds 1-3": pairs_fresh}
        if "N2" in Rt and "N4p" in Rt:
            sets["all 4 seeds (seed 0 original N4')"] = [(Rt["N2"][0], Rt["N4p"][0])] + pairs_fresh
        if 0 in R2.get("N4p", {}) and "N2" in Rt:
            sets["all 4 seeds (seed 0 rerun N4')"] = [(Rt["N2"][0], R2["N4p"][0])] + pairs_fresh
        for k, P in sets.items():
            if len(P) < 2:
                continue
            e = {"n": len(P)}
            for n in (9, 15, 21, 33):
                e[f"em{n}_pts"] = U.tci([100 * (a[f"test{n}"]["T150"]["em"] - b[f"test{n}"]["T150"]["em"]) for a, b in P])
                e[f"cell_acc{n}"] = U.tci([a[f"test{n}"]["T150"]["cell_acc"] - b[f"test{n}"]["T150"]["cell_acc"] for a, b in P])
            m_em, m_acc = e["em33_pts"]["mean"], e["cell_acc33"]["mean"]
            e["pass_on_mean"] = abs(m_em) <= band and abs(m_acc) <= 0.02
            e["cell_acc33_ci_inside_0.02"] = (e["cell_acc33"]["lo"] is not None and e["cell_acc33"]["lo"] >= -0.02
                                              and e["cell_acc33"]["hi"] <= 0.02)
            pm[k] = e
        out["debug_paired_mean"] = pm
        tr2 = {}
        for a, s_ in (("N4p", 0), ("N2", 1), ("N4p", 1), ("N2", 2), ("N4p", 2), ("N2", 3), ("N4p", 3)):
            f = os.path.join(RES, "aa2", "train", f"{a}_s{s_}.json")
            if os.path.exists(f):
                L = J(f)["log"]
                tr2[f"{a}_s{s_}"] = {"clamp": J(f)["acct"]["clamp"], "final_l_nat_last5": float(np.mean([x["l_nat"] for x in L[-5:]])),
                                     "max_l_nat_2nd_half": max(x["l_nat"] for x in L[len(L) // 2:])}
        out["debug_aa2_train"] = tr2
    f = os.path.join(RES, "train", "N4p_s0.json")
    if os.path.exists(f):
        L = J(f)["log"]
        out["N4p_s0_divergence"] = {"l_nat_by_step": {x["step"]: round(x["l_nat"], 3) for x in L[::4]}}
    return rnd(out)


def p9bT():
    Rt = test_table()
    out = {"stage": "P9b-T", "em33_T150": {}, "em33_T300": {}, "em_by_size_T150": {}}
    for a, byseed in Rt.items():
        out["em33_T150"][a] = U.tci([em33(byseed[s]) for s in sorted(byseed)])
        out["em33_T150"][a]["per_seed"] = [em33(byseed[s]) for s in sorted(byseed)]
        out["em33_T300"][a] = U.tci([byseed[s]["test33_T300_200mazes"]["T300"]["em"] for s in sorted(byseed)])
        out["em_by_size_T150"][a] = {n: U.tci([byseed[s][f"test{n}"]["T150"]["em"] for s in sorted(byseed)])["mean"]
                                     for n in (9, 15, 21, 33)}
    c = {}
    if "N4a" in Rt and "N1c" in Rt:
        c["i_N4a_minus_N1c"] = paired(Rt, "N4a", "N1c", em33)
        p = c["i_N4a_minus_N1c"]["pts"]
        c["i_pass"] = p["mean"] is not None and p["mean"] >= 5 and p["lo"] is not None and p["lo"] > 0
    if "N4a" in Rt and "N3" in Rt:
        c["ii_N4a_minus_N3"] = paired(Rt, "N4a", "N3", em33)
        p = c["ii_N4a_minus_N3"]["pts"]
        c["ii_pass"] = p["mean"] is not None and p["mean"] >= 2 and p["lo"] is not None and p["lo"] > 0
    if "N4a" in Rt and "N0" in Rt:
        c["iii_N4a_minus_N0"] = paired(Rt, "N4a", "N0", em33)
        p = c["iii_N4a_minus_N0"]["pts"]
        c["iii_pass"] = p["mean"] is not None and p["mean"] >= -2
    if "N4a" in Rt and "N3b" in Rt:  # Stage 2b rival
        c["ii_b_N4a_minus_N3b"] = paired(Rt, "N4a", "N3b", em33)
        p = c["ii_b_N4a_minus_N3b"]["pts"]
        c["ii_b_pass"] = p["mean"] is not None and p["mean"] >= 2 and p["lo"] is not None and p["lo"] > 0
    if "N3" in Rt and "N1c" in Rt:
        c["N3_minus_N1c"] = paired(Rt, "N3", "N1c", em33)
        p = c["N3_minus_N1c"]["pts"]
        c["stage2b_trigger_N3"] = p["mean"] is not None and p["mean"] >= 2 and p["lo"] is not None and p["lo"] > 0
    if "N1" in Rt and "N1c" in Rt:
        c["N1c_minus_N1"] = paired(Rt, "N1c", "N1", em33)
    out["claims"] = c
    # descriptive (not pre-registered): the same paired differences at the smaller test sizes, where exact match
    # is off the floor (33x33 is near 0 for every binary-state arm)
    ds = {}
    for n in (15, 21):
        fn = (lambda r, n=n: r[f"test{n}"]["T150"]["em"])
        for a, b in (("N4a", "N1c"), ("N4a", "N3"), ("N4a", "N0"), ("N1c", "N1"), ("N3", "N1c")):
            if a in Rt and b in Rt:
                ds[f"{a}-{b} @{n}x{n}"] = paired(Rt, a, b, fn)
    out["descriptive_by_size"] = ds
    f = {}
    if "i_N4a_minus_N1c" in c:
        f["self_reveal_aid (N4a - N1c < 2)"] = c["i_N4a_minus_N1c"]["pts"]["mean"] < 2
    if "iii_N4a_minus_N0" in c:
        f["N4a below N0 by > 5"] = c["iii_N4a_minus_N0"]["pts"]["mean"] < -5
    if "ii_N4a_minus_N3" in c:
        f["revealing_not_noise (N3 >= N4a - 1)"] = c["ii_N4a_minus_N3"]["pts"]["mean"] <= 1
    if "ii_b_N4a_minus_N3b" in c:
        f["revealing_not_noise (N3b >= N4a - 1)"] = c["ii_b_N4a_minus_N3b"]["pts"]["mean"] <= 1
    out["falsifiers_triggered"] = f
    return rnd(out)


# ------------------------------------------------------------------ P9b-E (imp JSONs)
def boot_auroc_diff(cells_by_seed, est_a, others, B=2000, seed=0):
    """Hierarchical bootstrap: resample seeds, then mazes within each seed; statistic = mean over seeds of
    AUROC(est_a) - max over others of AUROC(other) (the max is taken on the pooled-over-seeds means)."""
    rng = np.random.default_rng(seed)
    S = list(cells_by_seed)
    prep = {}
    for s in S:
        c = cells_by_seed[s]
        mz = np.array(c["maze"])
        um = np.unique(mz)
        prep[s] = (mz, um, {k: np.array(c[k], float) for k in [est_a, *others]}, np.array(c["must"]))

    def stat(pick):
        acc = {k: [] for k in [est_a, *others]}
        for s, mazes in pick:
            mz, um, sc, must = prep[s]
            rows = np.concatenate([np.flatnonzero(mz == m) for m in mazes])
            for k in acc:
                acc[k].append(U.auroc(sc[k][rows], must[rows]))
        m = {k: np.nanmean(v) for k, v in acc.items()}
        return m[est_a] - max(m[k] for k in others), m

    point, m0 = stat([(s, prep[s][1]) for s in S])
    bs = []
    for _ in range(B):
        ss = rng.choice(S, len(S), replace=True)
        pick = [(s, rng.choice(prep[s][1], len(prep[s][1]), replace=True)) for s in ss]
        bs.append(stat(pick)[0])
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"point": point, "lo95": float(lo), "hi95": float(hi), "B": B, "auroc_means": m0}


def p9bE(arm="N1", B=1000):
    R = seeds_of(os.path.join(RES, "eval", "imp_*.json"))
    R = {a.replace("imp_", ""): v for a, v in R.items()}
    out = {"stage": "P9b-E", "arm": arm}
    if arm not in R:
        return out
    by = R[arm]
    for st in ("imp9", "imp15"):
        E = {}
        for k in ("BS", "BS_T4", "BS_T2", "AtP", "FCI", "H0", "LW"):
            E[k] = U.tci([by[s][st]["estimators"][k]["auroc_must_onpath"] for s in sorted(by)])
            E[k]["spearman_graded"] = U.tci([by[s][st]["estimators"][k]["spearman_graded_onpath"] for s in sorted(by)])["mean"]
        o = {"auroc_must_onpath": E}
        if all("cells_onpath" in by[s][st] for s in by):
            o["boot_BS_minus_max(H0,FCI,AtP)"] = boot_auroc_diff({s: by[s][st]["cells_onpath"] for s in by}, "BS",
                                                                 ["H0", "FCI", "AtP"], B=B)
            bt = o["boot_BS_minus_max(H0,FCI,AtP)"]
            o["pass_auroc>=0.85"] = E["BS"]["mean"] >= 0.85
            o["pass_margin>=0.05_ci_excl_0"] = bt["point"] >= 0.05 and bt["lo95"] > 0
            o["falsified (<= max + 0.02)"] = bt["point"] <= 0.02
        o["iia"] = {k: U.tci([by[s][st]["iia"][k]["whole_maze"] for s in sorted(by)])
                    for k in ("LW", "best_kv_crossfit", "random_kv", "none")}
        o["iia_changed_cell_acc"] = {k: U.tci([by[s][st]["iia"][k]["changed_cell_acc"] for s in sorted(by)])["mean"]
                                     for k in ("LW", "best_kv_crossfit", "random_kv", "none")}
        out[st] = o
    al = {}
    for n in (15, 33):
        key = f"test{n}_T150_200mazes"
        rows = [by[s]["allocation"][key] for s in sorted(by)]
        d_top_atp = [100 * (r["drop"]["pay_top2"] - r["drop"]["atp_top2"]) for r in rows]
        d_top_rnd = [100 * (r["drop"]["pay_top2"] - r["drop_random2_mean"]) for r in rows]
        bot = [100 * r["drop"]["pay_bottom2"] for r in rows]
        al[n] = {"em_none": U.tci([r["em_none"] for r in rows]),
                 "drop_pts": {g: U.tci([100 * r["drop"][g] for r in rows])["mean"] for g in rows[0]["drop"]},
                 "pay_top2_minus_atp_top2": U.tci(d_top_atp), "pay_top2_minus_random2": U.tci(d_top_rnd),
                 "pay_bottom2_drop": U.tci(bot)}
    a33 = al[33]
    al["pass_33"] = {"top2 > atp_top2": a33["pay_top2_minus_atp_top2"]["mean"] > 0,
                     "top2 > random2": a33["pay_top2_minus_random2"]["mean"] > 0,
                     "bottom2 <= 2pts": a33["pay_bottom2_drop"]["mean"] <= 2}
    al["falsified_33 (top2 <= random2)"] = a33["pay_top2_minus_random2"]["mean"] <= 0
    out["allocation"] = al
    return rnd(out)


def p9bE_iia():
    """(iv): IIA(N4a) >= IIA(N1) + 0.1 for both LW and the best channel, both above the nulls."""
    R = seeds_of(os.path.join(RES, "eval", "imp_*.json"))
    R = {a.replace("imp_", ""): v for a, v in R.items()}
    out = {}
    if "N1" in R and "N4a" in R:
        for st in ("imp9", "imp15"):
            o = {}
            for k in ("LW", "best_kv_crossfit"):
                common = sorted(set(R["N1"]) & set(R["N4a"]))
                d = [R["N4a"][s][st]["iia"][k]["whole_maze"] - R["N1"][s][st]["iia"][k]["whole_maze"] for s in common]
                o[k] = U.tci(d)
                o[k]["pass>=0.1"] = o[k]["mean"] >= 0.1
                o[k]["N4a_above_nulls"] = all(
                    np.mean([R["N4a"][s][st]["iia"][k]["whole_maze"] for s in R["N4a"]]) >
                    np.mean([R["N4a"][s][st]["iia"][nl]["whole_maze"] for s in R["N4a"]]) for nl in ("random_kv", "none"))
            o["iv_pass"] = all(o[k]["pass>=0.1"] and o[k]["N4a_above_nulls"] for k in ("LW", "best_kv_crossfit"))
            o["falsified (<= +0.05)"] = any(o[k]["mean"] <= 0.05 for k in ("LW", "best_kv_crossfit"))
            out[st] = o
    return rnd(out)


def p9bE_desc():
    """Descriptive, not pre-registered (from the imp npz files, which are trashed after this runs): can each
    estimator tell path cells from off-path cells, and must-pass cells among ALL open cells?"""
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "eval", "imp_*_s*.npz"))):
        arm, s = os.path.basename(f)[4:-4].rsplit("_s", 1)
        z = np.load(f)
        for st in ("imp9", "imp15"):
            on, must = z[f"{st}_onpath"].astype(bool), z[f"{st}_must"].astype(bool)
            est = {"BS": z[f"{st}_BS"][:, :, 0].astype(float).max(1), "AtP": z[f"{st}_AtP"][:, :, 0].astype(float).max(1),
                   "LW": z[f"{st}_LW"].astype(float), "FCI": z[f"{st}_FCI"].astype(float), "H0": z[f"{st}_H0"].astype(float)}
            o = out.setdefault(arm, {}).setdefault(st, {})
            for k, v in est.items():
                o.setdefault(k, {"auroc_onpath_vs_offpath": [], "auroc_must_all_open": [], "seeds": []})
                o[k]["auroc_onpath_vs_offpath"].append(U.auroc(v, on))
                o[k]["auroc_must_all_open"].append(U.auroc(v, must))
                o[k]["seeds"].append(int(s))
    for arm in out:
        for st in out[arm]:
            for k, v in out[arm][st].items():
                for m in ("auroc_onpath_vs_offpath", "auroc_must_all_open"):
                    v[m] = U.tci(v[m])
    return rnd(out)


def charts(S):
    """Chart JSON: each chart = {id, title, type, x, series:[{name, y, lo, hi}]}. S = the stage summaries."""
    C = []
    for key in ("p9a", "p9a_rt"):
      if key in S:
        A = S[key]["arms"]
        arms = [a for a in ("X0", "X4", "X2p", "X2", "X3", "X1") if a in A]
        lab = " (retuned recipe)" if key == "p9a_rt" else " (pre-registered 1.5k steps)"
        for sz in ("9", "15"):
            C.append({"id": f"{key}_kl_{sz}", "title": f"P9a KL per door-dependent cell, {sz}x{sz} (bar 0.02){lab}",
                      "type": "bar", "x": arms, "series": [{"name": "KL (nats)",
                                                           "y": [A[a][sz]["kl"]["mean"] for a in arms],
                                                           "lo": [A[a][sz]["kl"]["lo"] for a in arms],
                                                           "hi": [A[a][sz]["kl"]["hi"] for a in arms]}]})
        C.append({"id": f"{key}_voi", "title": f"P9a VOI Spearman and sign accuracy, 9x9 (bars 0.9 / 0.9){lab}", "type": "bar",
                  "x": arms, "series": [{"name": m, "y": [A[a]["9"][m]["mean"] for a in arms],
                                         "lo": [A[a]["9"][m]["lo"] for a in arms], "hi": [A[a]["9"][m]["hi"] for a in arms]}
                                        for m in ("voi_spearman", "sign_acc")]})
    if "gate" in S:
        G = S["gate"]["arms"]
        arms = [a for a in sorted(G) if a != "N1_gate3"]
        C.append({"id": "gate_em", "title": "Gate: exact match, dev 9x9 T=20 and dev 25x25 T=100", "type": "bar",
                  "x": arms, "series": [{"name": k, "y": [G[a][k]["em"]["mean"] for a in arms],
                                         "lo": [G[a][k]["em"]["lo"] for a in arms], "hi": [G[a][k]["em"]["hi"] for a in arms]}
                                        for k in ("dev9_T20", "dev25_T100")]})
    Rt = test_table()
    if Rt:
        arms = [a for a in ("N0", "N1", "N1c", "N3", "N3b", "N4a", "N2", "N4p") if a in Rt]
        C.append({"id": "test33_curve", "title": "33x33 exact match vs iterations (mean over seeds)", "type": "line",
                  "x": list(range(1, 151)),
                  "series": [{"name": f"{a} (n={len(Rt[a])})",
                              "y": list(np.mean([Rt[a][s]["test33"]["em_curve"] for s in Rt[a]], 0).round(4))} for a in arms]})
        if "p9bT" in S:
            E = S["p9bT"]["em33_T150"]
            C.append({"id": "test33_T150", "title": "33x33 exact match at T=150 (95% t-CI over seeds)", "type": "bar",
                      "x": arms, "series": [{"name": "em", "y": [E[a]["mean"] for a in arms], "lo": [E[a]["lo"] for a in arms],
                                             "hi": [E[a]["hi"] for a in arms]}]})
            B = S["p9bT"]["em_by_size_T150"]
            C.append({"id": "em_by_size", "title": "Exact match at T=150 by maze size", "type": "line", "x": [9, 15, 21, 33],
                      "series": [{"name": a, "y": [B[a][str(n)] if str(n) in B[a] else B[a][n] for n in (9, 15, 21, 33)]}
                                 for a in arms]})
    if "p9bE" in S and "imp9" in S["p9bE"]:
        E = S["p9bE"]
        ests = ["BS", "BS_T4", "BS_T2", "AtP", "FCI", "LW", "H0"]
        C.append({"id": "p9bE_auroc", "title": f"AUROC(must-pass | onpath) on {E['arm']} (bar 0.85)", "type": "bar",
                  "x": ests, "series": [{"name": st, "y": [E[st]["auroc_must_onpath"][k]["mean"] for k in ests],
                                         "lo": [E[st]["auroc_must_onpath"][k]["lo"] for k in ests],
                                         "hi": [E[st]["auroc_must_onpath"][k]["hi"] for k in ests]} for st in ("imp9", "imp15")]})
        al = E["allocation"]
        gs = list(al["33"]["drop_pts"]) if "33" in al else list(al[33]["drop_pts"])
        a33 = al["33"] if "33" in al else al[33]
        C.append({"id": "allocation33", "title": "Exact-match drop (points) when two channels are held, 33x33 T=150",
                  "type": "bar", "x": gs, "series": [{"name": "drop", "y": [a33["drop_pts"][g] for g in gs]}]})
    return C


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    done = {}
    if what in ("p9a", "all"):
        done["p9a"] = p9a()
        if os.path.isdir(os.path.join(RES, "rt", "p9a")):  # rerun at the pilot-retuned recipe (declared deviation)
            done["p9a_rt"] = p9a(sub="rt/p9a")
    if what in ("gate", "all"):
        done["gate"] = gate()
    if what in ("aa", "all"):
        done["aa"] = aa()
    if what in ("p9bT", "all"):
        done["p9bT"] = p9bT()
        done["p9bT"]["iv_iia"] = p9bE_iia()
    if what in ("p9bE", "all"):
        done["p9bE"] = p9bE()
        done["p9bE_N4a"] = p9bE("N4a")  # descriptive (the metrics list asks for N1 and N4a)
        if glob.glob(os.path.join(RES, "eval", "imp_*.npz")):
            done["p9bE_desc"] = p9bE_desc()
    for k, v in done.items():
        U.dump(os.path.join(RES, f"summary_{k}.json"), v)
    if what == "all":
        U.dump(os.path.join(RES, "charts.json"), charts(done))
    print("wrote", ", ".join(f"summary_{k}.json" for k in done))


if __name__ == "__main__":
    main()
