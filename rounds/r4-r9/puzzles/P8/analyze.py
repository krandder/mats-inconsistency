"""P8 analysis: price tables, pricing accuracy vs deployed truth (S1/F1/F1'), unit tests U1-U8, noise floor, towers
(S5/F2), C2 economy (S2/F3), R6 surrogate (S3/F4), parity search (S4). Writes results/p8_summary.json,
results/p8_chart.json and prints a compact report (RESULTS.md is written by hand from it)."""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P8_RES", os.path.join(HERE, "results"))
sys.path.insert(0, HERE)
from p8lib import POOL_MAZE, POOL_PAR, auc, ci, kendall_b, spearman, jsonable  # noqa: E402

CTX_M = {"empty": [], "g2g4": ["g2", "g4"]}
CTX_P = {"empty": [], "p1234": ["p1", "p2", "p3", "p4"]}
F_MAZE = 8 * 121


def load(pat):
    out = {}
    for p in sorted(glob.glob(os.path.join(RES, pat))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    return out


def by_seed(pat):
    return {v["seed"]: v for v in load(pat).values()}


def items(kind):
    ctx, pool = (CTX_M, POOL_MAZE) if kind == "maze" else (CTX_P, POOL_PAR)
    return [f"{c}:{z}" for c in ctx for z in pool if z not in ctx[c]]


# ------------------------------------------------------------ noise floor (Phase 0)
KREP = int(os.environ.get("P8_KREP", "12"))
KREP_T = int(os.environ.get("P8_KREP_TOWER", "4"))


def noise():
    """SD of single-run codelengths/savings over inits x data orders; price noise = SD / sqrt(KREP) (prices are
    means of KREP replicates, each with its own init and order)."""
    out = {}
    for kind in ("maze", "par"):
        runs = [v for k, v in sorted(load(f"noise_{kind}_s*_o*.json").items())]
        if not runs:
            continue
        bits = {}
        for r in runs:
            for t, v in r["bits"].items():
                bits.setdefault(t, []).extend(v)
        sd = {t: float(np.std(v, ddof=1)) for t, v in bits.items()}
        o = dict(sd=sd, mean={t: float(np.mean(v)) for t, v in bits.items()}, n_runs=len(next(iter(bits.values()))),
                 n_orders=len(runs))
        if kind == "maze":
            A = np.array([np.array(r["bits"]["L(Y|X)"]) - np.array(r["bits"]["L(Y|X,g2)"]) - np.array(r["bits"]["L(g2|X)"])
                          for r in runs])
            sav = A.ravel()
            s1 = float(np.std(sav, ddof=1))
            o["saving_g2"] = dict(mean=float(np.mean(sav)), sd_single=s1, sd_K=s1 / math.sqrt(KREP),
                                  sd4_latin=float(np.std([np.mean([A[o, (g - o) % A.shape[1]] for o in range(A.shape[0])])
                                                          for g in range(A.shape[1])], ddof=1)),
                                  order_means=A.mean(1).tolist())
            o["gate_single"] = dict(ratio=3 * s1 / np.mean(sav), fired=bool(3 * s1 > 0.1 * np.mean(sav)))
            o["gate_K"] = dict(ratio=3 * s1 / math.sqrt(KREP) / np.mean(sav),
                               fired=bool(3 * s1 / math.sqrt(KREP) > 0.1 * np.mean(sav)))
            o["price_sd"] = s1 / math.sqrt(KREP)
            o["sd_single"] = s1
        else:
            # rough: unpaired sum of the three codelength variances of a saving, / sqrt(K)
            v = sd.get("L(Y|X)", 0) ** 2 + sd.get("L(Y|X,q1,q2)", 0) ** 2 + sd.get("L(p1|X)", 0) ** 2
            o["price_sd"] = float(math.sqrt(v) / math.sqrt(KREP))
        out[kind] = o
    return out


# ------------------------------------------------------------ prices + truth
def price_table(kind, seed, lcap):
    p = load(f"price_{kind}_s{seed}.json").get(f"price_{kind}_s{seed}")
    if p is None:
        return None
    r7b = load(f"r7b_{kind}_s{seed}.json").get(f"r7b_{kind}_s{seed}")
    r6 = load(f"r6_{kind}_s{seed}.json").get(f"r6_{kind}_s{seed}")
    tab = {}
    for it in items(kind) + [k for k in p["prices"] if k not in items(kind)]:
        v = p["prices"][it]
        c = it.split(":")[0]
        row = {k: v[k] for k in ("L_Y", "L_Z", "L_YZ", "R1", "R2", "R3", "R4", "R5", "EDL") if k in v}
        row["R7a"] = v.get("R7a")
        row["R7b"] = r7b["prices"][it]["R7b"] if r7b and it in r7b["prices"] else None
        row["R6"] = r6["prices"].get(it) if r6 else None
        row["pct_local"] = 100 * v["R4"] / v["L_Y"]
        row["pct_cap"] = 100 * v["R4"] / lcap if lcap else None
        row["curve_R4"] = v.get("curve_R4")
        row["R4_sd"] = v.get("R4_sd")
        tab[it] = row
    tab["_krep"] = p.get("krep", 1)
    w = {"work_R4": sum(p["members"][k]["work"] for k in p["members"] if k.startswith(("Y|", "Z|", "YZ|"))
                        and (k.startswith("Y|") or k.split("|")[1] in items(kind)))}
    return tab, w, p


def truth_of(kind, seed, ev="held"):
    t = load(f"truth_{kind}_s{seed}.json").get(f"truth_{kind}_s{seed}")
    if t is None:
        return None
    return {it: t["items"][it][ev] for it in items(kind) if it in t["items"]}


def rule_prices(tab, its, rule, lam=None, rng=None):
    if rule == "R0":
        return list(rng.random(len(its)))
    if rule == "R2":
        lo = min(tab[i]["R2"] for i in its) - 1e9
        return [tab[i]["R2"] if tab[i]["R1"] > 0 else lo for i in its]
    if rule == "R8":
        return [tab[i]["R1"] - lam * tab[i]["L_Z"] for i in its]
    return [tab[i][rule] if tab[i][rule] is not None else np.nan for i in its]


RULES = ["R0", "R1", "R2", "R3", "R4", "R5", "R7a", "R7b", "R6"]


def score(prices, truth, cls):
    pr = np.asarray(prices, float)
    ok = ~np.isnan(pr)
    if ok.sum() < 3:
        return None
    return dict(tau=kendall_b(pr[ok], np.asarray(truth)[ok]), rho=spearman(pr[ok], np.asarray(truth)[ok]),
                auc=auc(pr[ok], np.asarray(cls)[ok] > 0))


def lam_fit(tabs, truths, its_of, grid):
    """argmax over grid of mean tau (over seeds) of R1 - lam*L_Z."""
    curve = []
    for lam in grid:
        taus = []
        for s in tabs:
            its = its_of(s)
            if not its:
                continue
            taus.append(kendall_b(rule_prices(tabs[s], its, "R8", lam), [truths[s][i]["truth"] for i in its]))
        curve.append(float(np.mean(taus)) if taus else float("nan"))
    j = int(np.nanargmax(curve)) if any(c == c for c in curve) else 0
    return float(grid[j]), curve


def perm_test(tabs, truths, its_of, ra, rb, lam=None, n=4000, seed=0):
    """paired permutation over items: swap the two rules' within-seed ranks per item. H1: tau(ra) > tau(rb)."""
    rng = np.random.default_rng(seed)
    from scipy.stats import rankdata
    data = []
    for s in tabs:
        its = its_of(s)
        a = np.asarray(rule_prices(tabs[s], its, ra, lam), float)
        b = np.asarray(rule_prices(tabs[s], its, rb, lam), float)
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        data.append((rankdata(a), rankdata(b), np.asarray([truths[s][i]["truth"] for i in its])))
    if not data:
        return None

    def stat(pairs):
        return np.mean([kendall_b(a, t) - kendall_b(b, t) for a, b, t in pairs])

    obs = stat(data)
    cnt = 0
    for _ in range(n):
        pp = []
        for a, b, t in data:
            m = rng.random(len(a)) < 0.5
            pp.append((np.where(m, b, a), np.where(m, a, b), t))
        cnt += stat(pp) >= obs - 1e-12
    return dict(diff=float(obs), p=(cnt + 1) / (n + 1))


def pricing(kind, noise_sd, lcaps):
    seeds = sorted(by_seed(f"truth_{kind}_s*.json"))
    tabs, truths, work = {}, {}, {}
    for s in seeds:
        pt = price_table(kind, s, lcaps.get(s))
        tr = truth_of(kind, s)
        if pt and tr:
            tabs[s], truths[s], work[s] = pt[0], tr, pt[1]
    if not tabs:
        return None
    ctxs = CTX_M if kind == "maze" else CTX_P
    rng = np.random.default_rng(1)
    out = dict(seeds=sorted(tabs), per_seed={}, pooled={}, per_context={})
    grid = np.concatenate([[0], np.logspace(-3, 1, 41)])
    for s in tabs:
        its = [i for i in items(kind) if i in truths[s]]
        tv = [truths[s][i]["truth"] for i in its]
        cl = [truths[s][i]["cls"] for i in its]
        out["per_seed"][s] = {r: score(rule_prices(tabs[s], its, r, rng=rng), tv, cl) for r in RULES}
    for r in RULES:
        taus = [out["per_seed"][s][r]["tau"] for s in tabs if out["per_seed"][s][r]]
        out["pooled"][r] = dict(tau=ci(taus), rho=ci([out["per_seed"][s][r]["rho"] for s in tabs if out["per_seed"][s][r]]),
                                auc=ci([out["per_seed"][s][r]["auc"] for s in tabs if out["per_seed"][s][r]]))
    for c in ctxs:
        its_c = lambda s, c=c: [i for i in items(kind) if i.startswith(c + ":") and i in truths[s]]
        pc = {}
        for r in RULES:
            taus = []
            for s in tabs:
                its = its_c(s)
                sc = score(rule_prices(tabs[s], its, r, rng=rng), [truths[s][i]["truth"] for i in its],
                           [truths[s][i]["cls"] for i in its])
                if sc:
                    taus.append(sc["tau"])
            pc[r] = ci(taus)
        out["per_context"][c] = pc
    its_all = lambda s: [i for i in items(kind) if i in truths[s]]
    lam_self, curve = lam_fit(tabs, truths, its_all, grid)
    out["lambda_curve"] = dict(grid=grid, tau=curve, argmax=lam_self)
    # other-context lambda (maze): fit on one context, apply to the other
    cs = list(ctxs)
    other = {}
    for a, b in ((cs[0], cs[1]), (cs[1], cs[0])):
        la, _ = lam_fit(tabs, truths, lambda s: [i for i in items(kind) if i.startswith(a + ":") and i in truths[s]], grid)
        taus = [kendall_b(rule_prices(tabs[s], [i for i in items(kind) if i.startswith(b + ":")], "R8", la),
                          [truths[s][i]["truth"] for i in items(kind) if i.startswith(b + ":")]) for s in tabs]
        other[b] = dict(lam=la, tau=ci(taus))
    out["lambda_other_context"] = other
    out["_tabs"], out["_truths"], out["_work"] = tabs, truths, work
    # F1': noisy copy / decoy / Manhattan paid above +3 SD where truth <= 0; or g2 at S=empty priced <= 0
    if kind == "maze":
        bad = {"nz05", "nz20", "d2", "d4", "d6", "m2", "m4", "m6", "m8"}
        fired = []
        for s in tabs:
            hits = [i for i in items(kind) if i.split(":")[1] in bad and tabs[s][i]["R4"] > 3 * noise_sd
                    and truths[s][i]["truth"] <= 0]
            g2bad = tabs[s]["empty:g2"]["R4"] <= 0
            fired.append(dict(seed=s, hits=hits, g2_nonpositive=bool(g2bad), fired=bool(hits or g2bad)))
        out["F1prime"] = dict(per_seed=fired, fired=sum(f["fired"] for f in fired) >= 2)
    return out


def finish_pricing(pm, pp):
    """S1 and F1 on mazes, with R8 lambda fit on parity (pre-registered) and on the other maze context."""
    if pm is None:
        return None
    res = {}
    lam_par = pp["lambda_curve"]["argmax"] if pp else None
    tabs, truths = pm["_tabs"], pm["_truths"]
    its_all = lambda s: [i for i in items("maze") if i in truths[s]]
    r8 = {}
    if lam_par is not None:
        taus = [kendall_b(rule_prices(tabs[s], its_all(s), "R8", lam_par), [truths[s][i]["truth"] for i in its_all(s)])
                for s in tabs]
        r8["parity_fit"] = dict(lam=lam_par, tau=ci(taus))
    r8["other_context"] = pm["lambda_other_context"]
    r8["in_sample_oracle"] = dict(lam=pm["lambda_curve"]["argmax"], tau=float(np.nanmax(pm["lambda_curve"]["tau"])))
    res["R8"] = r8
    t4 = pm["pooled"]["R4"]["tau"]["mean"]
    tests = {}
    for r in ("R1", "R2", "R5", "R7a", "R7b", "R6"):
        tests[r] = perm_test(tabs, truths, its_all, "R4", r)
    if lam_par is not None:
        tests["R8_parity_fit"] = perm_test(tabs, truths, its_all, "R4", "R8", lam=lam_par)
    res["perm_R4_vs"] = tests
    comp = [pm["pooled"]["R7a"]["tau"]["mean"], pm["pooled"]["R7b"]["tau"]["mean"]]
    if "parity_fit" in r8:
        comp.append(r8["parity_fit"]["tau"]["mean"])
    comp = [c for c in comp if c is not None]
    best_cheap = max(comp) if comp else None
    res["tau_R4"] = t4
    res["best_cheap_tau"] = best_cheap
    res["S1"] = bool(t4 is not None and best_cheap is not None and t4 - best_cheap >= 0.05 and
                     all(v and v["p"] < 0.05 for k, v in tests.items() if k != "R6"))
    res["F1_novelty_nil"] = bool(best_cheap is not None and t4 is not None and best_cheap >= t4 - 0.05)
    return res


# ------------------------------------------------------------ unit tests
def unit_tests(nsd):
    U = {}
    p = load("price_maze_s0.json").get("price_maze_s0")
    sd = nsd.get("maze", {}).get("price_sd", float("nan"))
    if p:
        pr = p["prices"]
        N, cells = p["N"], 121
        U["U1_copyY"] = dict(obs=pr["empty:copyY"]["R4"], pred=f"[-2F, 0] = [{-2 * F_MAZE}, 0]",
                             ok=bool(-2 * F_MAZE <= pr["empty:copyY"]["R4"] <= 0))
        ub = N * cells
        U["U2_random"] = dict(obs=pr["empty:random"]["R4"], pred=-ub, ok=bool(abs(pr["empty:random"]["R4"] + ub) <= 0.05 * ub))
        U["U3_null"] = dict(obs=pr["empty:null"]["R4"], pred=-F_MAZE, sd=sd,
                            ok=bool(abs(pr["empty:null"]["R4"] + F_MAZE) <= 3 * sd),
                            note="prediction is -F (uniform first block), R1 = 0 exactly by pairing")
        U["U4_open_start"] = dict(obs=dict(open=pr["empty:open"]["R4"], start=pr["empty:startZ"]["R4"]), limit=3 * sd,
                                  ok=bool(pr["empty:open"]["R4"] <= 3 * sd and pr["empty:startZ"]["R4"] <= 3 * sd))
        U["U8_R3_final_block_importance"] = {k: v["R3"] for k, v in pr.items() if k.startswith("empty:g")}
    q = load("price_par_s0.json").get("price_par_s0")
    if q:
        e = {k: v["R4"] for k, v in q["prices"].items() if k.startswith("empty:") and k.split(":")[1] in POOL_PAR}
        U["U5_parity_greedy_lists_nothing"] = dict(max_R4=max(e.values()), argmax=max(e, key=e.get),
                                                   ok=bool(max(e.values()) <= 0))
    u6 = load("u6_maze_s0.json").get("u6_maze_s0")
    if u6:
        U["U6_start_first"] = dict(first=u6["first"], price=u6["R4"]["startZ"],
                                   pred=u6["L_Y_walls"] - u6["L_start"] - u6["L_Y_walls_start"],
                                   ok=u6["first"] == "startZ")
    to = load("toracle_s0.json").get("toracle_s0")
    if to:
        pay = to["ladder_analysis"]["pay"]["g2"]
        U["U7_g2_indirect_share"] = dict(direct=pay["direct"], total=pay["total"], indirect=pay["indirect"],
                                         share=pay["indirect"] / pay["total"] if pay["total"] else None,
                                         ok=bool(pay["total"] > 0 and pay["indirect"] / pay["total"] >= 0.9))
    return U


# ------------------------------------------------------------ towers
def used_by_depth(an, sd1):
    """sd1 = single-run SD of a saving; the analysis is a mean of an['krep'] replicates."""
    sd = sd1 / math.sqrt(an.get("krep", 1))
    S = an["S"]
    E = [(e["p"], e["W"]) for e in an["edges"] if e["delta"] > 3 * sd]
    depth = {W: 1 for W in S}
    for W in S:  # S is in listing order, parents precede children
        for p, w in E:
            if w == W:
                depth[W] = max(depth[W], depth[p] + 1)
    return max(depth.values()) if depth else 0, E


def towers(nsd, lcaps):
    sd = nsd.get("maze", {}).get("sd_single", 0.0)
    out = {}
    tor = by_seed("toracle_s*.json")
    rows = []
    for s, d in sorted(tor.items()):
        L = d["T1"]["listed"]
        dep = d["deployed"]
        an = d.get("T1_analysis")
        ubd, E = used_by_depth(an, sd) if an else (0, [])
        lad = d["ladder_analysis"]
        rows.append(dict(seed=s, listed=L, used_by_depth=ubd, edges=E, L_S=an["L_S"] if an else None,
                         L_S_ladder=lad["L_S"], L_cap=lcaps.get(s),
                         n_bfs=sum(x.startswith("g") for x in L), n_manhattan=sum(x.startswith("m") for x in L),
                         pay=an["pay"] if an else None,
                         acc_T1=dep["T1oracle"]["held"]["acc"], acc_B4=dep["B4"]["held"]["acc"],
                         acc15_T1=dep["T1oracle"]["f15"]["acc"], acc15_B4=dep["B4"]["f15"]["acc"],
                         bits_T1=dep["T1oracle"]["held"]["bits"], bits_B4=dep["B4"]["held"]["bits"]))
    if rows:
        out["toracle"] = rows
        out["S5"] = dict(bfs_ladder_depth3=[r["used_by_depth"] >= 3 and r["n_bfs"] > r["n_manhattan"] for r in rows],
                         within_1pt_B4=[abs(r["acc_T1"] - r["acc_B4"]) <= 0.01 for r in rows],
                         gap_acc=ci([r["acc_T1"] - r["acc_B4"] for r in rows]))
        f2 = [r["L_cap"] is not None and r["L_cap"] <= 1.1 * r["L_S_ladder"] for r in rows]
        out["F2_receptive_field"] = dict(per_seed=f2, ratio=[r["L_cap"] / r["L_S_ladder"] if r["L_cap"] else None
                                                             for r in rows])
    c2 = by_seed("c2_s*.json")
    rows = []
    for s, d in sorted(c2.items()):
        dep = d["deployed"]
        an = d.get("T1_analysis")
        ubd, E = used_by_depth(an, sd) if an else (0, [])
        arms = {k: dict(acc=v["held"]["acc"], bits=v["held"]["bits"], acc15=v["f15"]["acc"], bits15=v["f15"]["bits"],
                        work=v.get("work")) for k, v in dep.items()}
        big = {k: dict(acc=v["held"]["acc"], bits=v["held"]["bits"], acc15=v["f15"]["acc"], work=v.get("work"))
               for k, v in d.get("deployed_4k", {}).items()}
        rows.append(dict(seed=s, T1=d["T1"]["listed"], T7=d["T7"]["listed"], T0=d["T0_listed"], used_by_depth=ubd,
                         arms=arms, arms_4k=big, paired=d.get("paired_vs_T1"),
                         L_S=an["L_S"] if an else None, B2=dep["B2"].get("config"), B3=dep["B3"].get("config")))
    if rows:
        out["c2"] = rows
        wins = []
        for r in rows:
            a = r["arms"]
            t1 = a["T1"]
            w = all(t1["acc"] >= a[b]["acc"] + 0.02 for b in ("B2", "B3", "B5")) and \
                all(t1["acc15"] >= a[b]["acc15"] for b in ("B2", "B3", "B5")) and t1["acc"] > a["T0"]["acc"]
            wins.append(bool(w))
        out["S2"] = dict(per_seed=wins, n_win=sum(wins), n=len(wins), holds=len(wins) == 5 and sum(wins) >= 4)
        out["F3"] = dict(fired=not out["S2"]["holds"],
                         margin_T1_minus_best_baseline=ci([r["arms"]["T1"]["acc"] - max(r["arms"][b]["acc"] for b in
                                                                                       ("B2", "B3", "B5")) for r in rows]),
                         margin_T1_minus_T7=ci([r["arms"]["T1"]["acc"] - r["arms"]["T7"]["acc"] for r in rows]))
        out["F2_c2_depth_le1"] = [r["used_by_depth"] <= 1 for r in rows]
        out["c2_arm_means"] = {k: dict(acc=ci([r["arms"][k]["acc"] for r in rows]),
                                       acc15=ci([r["arms"][k]["acc15"] for r in rows]),
                                       bits=ci([r["arms"][k]["bits"] for r in rows]),
                                       work=ci([r["arms"][k]["work"] for r in rows if r["arms"][k].get("work")]))
                               for k in rows[0]["arms"]}
    return out


# ------------------------------------------------------------ R6 surrogate
def r6(pm):
    if pm is None:
        return None
    tabs = pm["_tabs"]
    rows = []
    for s, d in sorted(by_seed("r6_maze_s*.json").items()):
        if s not in tabs:
            continue
        its = items("maze")
        R4 = [tabs[s][i]["R4"] for i in its]
        R5 = [tabs[s][i]["R5"] for i in its]
        wR4 = pm["_work"][s]["work_R4"] / pm["_tabs"][s]["_krep"]  # one replicate = the registered single-run R4
        frac = (np.asarray(d["cum_work"]) / wR4).tolist()
        rho4 = [spearman([d["curves"][i][k] for i in its], R4) for k in range(len(frac))]
        rho5 = [spearman([d["curves"][i][k] for i in its], R5) for k in range(len(frac))]
        reach = [f for f, r in zip(frac, rho4) if r >= 0.8]
        rows.append(dict(seed=s, frac=frac, rho_R4=rho4, rho_R5=rho5, frac_at_08=reach[0] if reach else None,
                         final_rho_R4=rho4[-1], final_rho_R5=rho5[-1]))
    if not rows:
        return None
    s3 = [r["frac_at_08"] is not None and r["frac_at_08"] <= 0.2 and r["final_rho_R4"] > r["final_rho_R5"] for r in rows]
    f4 = [(r["frac_at_08"] is None or r["frac_at_08"] >= 0.5) or (r["final_rho_R5"] > r["final_rho_R4"]) for r in rows]
    return dict(rows=rows, S3=s3, F4=f4)


def r6b():
    rows = []
    for s, d in sorted(by_seed("r6b_par_s*.json").items()):
        p = load(f"price_par_s{s}.json").get(f"price_par_s{s}")
        its = items("par")
        lit = [d["prices_literal"][i] for i in its]
        mod = [d["prices"][i] for i in its]
        R4 = [p["prices"][i]["R4"] for i in its] if p else None
        ess = {k: v for k, v in d["n_matched"].items()}
        rows.append(dict(seed=s, rho_literal_R4=spearman(lit, R4) if R4 else None,
                         rho_model_R4=spearman(mod, R4) if R4 else None, n_scored=d["n_scored"],
                         ess_empty=ess.get("empty"), ess_min=min(ess.values()),
                         sd_literal_minus_R4=float(np.std(np.asarray(lit) - np.asarray(R4))) if R4 else None))
    return rows


def parity_search():
    rows = []
    for s, d in sorted(by_seed("r6_par_s*.json").items()):
        t5 = d.get("t5")
        bf = load(f"bf_par_s{s}.json").get(f"bf_par_s{s}")
        if not t5:
            continue
        row = dict(seed=s, recovered=t5["recovered"], best_dag=t5["best_dag"], level1=t5["level1"],
                   t5_work=t5["r6_work"] + t5["exact"]["work"],
                   L_S={json.dumps(e["dag"], sort_keys=True): e["L_S"] for e in t5["exact"]["dags"]})
        if bf:
            row.update(bf_best=bf["best_no_sanity"], bf_best_any=bf["best"], bf_rank_q1q2=bf["rank_q1q2"],
                       bf_work=bf["work"], level1_matches_bf=sorted(bf["best_no_sanity"].split(",")) == sorted(t5["level1"]),
                       speedup=bf["work"] / (t5["r6_work"] + t5["exact"]["work"]))
        rows.append(row)
    return dict(rows=rows, S4=sum(r["recovered"] for r in rows) >= 2 if len(rows) >= 3 else None)


def cpu_total():
    tot, n = 0.0, 0
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("cpu_secs"):
            tot += d["cpu_secs"]
            n += 1
    return dict(cpu_secs=tot, jobs=n)


def main():
    nsd = noise()
    lc = {k: by_seed(f"capref_{k}_s*.json") for k in ("maze", "par")}
    lcaps = {k: {s: v["L_cap"] for s, v in d.items()} for k, d in lc.items()}
    sd_m = nsd.get("maze", {}).get("price_sd", 0.0)
    pm = pricing("maze", sd_m, lcaps["maze"])
    pp = pricing("par", nsd.get("par", {}).get("price_sd", 0.0), lcaps["par"])
    fin = finish_pricing(pm, pp)
    sched = load("sched_maze_s0.json").get("sched_maze_s0")
    sch = None
    if sched and pm and 0 in pm["_tabs"]:
        z = list(sched["R4_fixed250"])
        sch = dict(rho=spearman([sched["R4_fixed250"][k] for k in z], [pm["_tabs"][0][f"empty:{k}"]["R4"] for k in z]),
                   tau=kendall_b([sched["R4_fixed250"][k] for k in z], [pm["_tabs"][0][f"empty:{k}"]["R4"] for k in z]))
    pc = load("pcalib_s0.json").get("pcalib_s0")
    summ = dict(noise=nsd, schedule_sensitivity=sch, parity_calibration=pc, L_cap=lcaps,
                capref={k: {s: {a: v["bits"] for a, v in d["arms"].items()} for s, d in lc[k].items()} for k in lc},
                unit_tests=unit_tests(nsd), towers=towers(nsd, lcaps["maze"]), r6=r6(pm), r6b=r6b(),
                parity_search=parity_search(), cpu=cpu_total())
    for name, pr in (("maze", pm), ("par", pp)):
        if pr:
            summ[f"pricing_{name}"] = {k: v for k, v in pr.items() if not k.startswith("_")}
            summ[f"price_table_{name}"] = {s: {i: {k: v for k, v in r.items() if k != "curve_R4"} for i, r in t.items()
                                              if not i.startswith("_")} for s, t in pr["_tabs"].items()}
            summ[f"truth_{name}"] = pr["_truths"]
    summ["maze_S1_F1"] = fin
    json.dump(jsonable(summ), open(os.path.join(RES, "p8_summary.json"), "w"), indent=1)
    chart = {}
    if pm:
        chart["tau_by_rule_maze"] = {r: pm["pooled"][r]["tau"] for r in RULES}
        chart["lambda_curve_maze"] = pm["lambda_curve"]
        s0 = sorted(pm["_tabs"])[0]
        chart["price_vs_truth_maze_seed%d" % s0] = [dict(item=i, R4=pm["_tabs"][s0][i]["R4"], R7a=pm["_tabs"][s0][i]["R7a"],
                                                        truth=pm["_truths"][s0][i]["truth"]) for i in items("maze")]
    if pp:
        chart["tau_by_rule_par"] = {r: pp["pooled"][r]["tau"] for r in RULES}
    if summ["r6"]:
        chart["r6_rho_vs_cpu_frac"] = [dict(seed=r["seed"], frac=r["frac"], rho_R4=r["rho_R4"], rho_R5=r["rho_R5"])
                                       for r in summ["r6"]["rows"]]
    if summ["towers"].get("c2_arm_means"):
        chart["c2_acc"] = summ["towers"]["c2_arm_means"]
    json.dump(jsonable(chart), open(os.path.join(RES, "p8_chart.json"), "w"), indent=1)
    brief = dict(noise={k: {kk: v[kk] for kk in v if kk in ("price_sd", "gate_3sd_over_10pct")} for k, v in nsd.items()},
                 sched=sch, U={k: v.get("ok") for k, v in summ["unit_tests"].items()},
                 tau_maze={r: pm["pooled"][r]["tau"] for r in RULES} if pm else None,
                 tau_ctx=pm["per_context"] if pm else None,
                 tau_par={r: pp["pooled"][r]["tau"] for r in RULES} if pp else None, fin=fin,
                 F1p=pm.get("F1prime") if pm else None,
                 towers={k: v for k, v in summ["towers"].items() if k not in ("toracle", "c2")},
                 r6={k: v for k, v in (summ["r6"] or {}).items() if k != "rows"},
                 r6_rows=[{k: r[k] for k in ("seed", "frac_at_08", "final_rho_R4", "final_rho_R5")} for r in (summ["r6"] or {}).get("rows", [])],
                 r6b=summ["r6b"], search=[{k: r.get(k) for k in ("seed", "recovered", "level1", "bf_best", "speedup")}
                                          for r in summ["parity_search"]["rows"]], cpu=summ["cpu"])
    print(json.dumps(jsonable(brief), indent=1))


if __name__ == "__main__":
    main()
