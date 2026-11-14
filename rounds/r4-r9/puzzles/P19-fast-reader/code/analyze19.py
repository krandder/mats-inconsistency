"""P19 analysis (pre-registered in ../PREREG.md). python analyze19.py RES_DIR [SEEDS] -> RES_DIR/../p19_summary.json,
p19_chart.json. RES_DIR holds truth_maze_s*_r*.json, pricerep_maze_s*_r*.json, r7brep_maze_s*_r*.json, frfo_maze_s*.json."""
import json
import math
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

RES = sys.argv[1]
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
OUT = os.path.dirname(os.path.abspath(RES.rstrip("/")))
KREP_P, KREP_W = int(os.environ.get("P19_KREP_P", "8")), int(os.environ.get("P19_KREP_W", "12"))
TREPS = (0, 1)
# primary FR configuration, fixed by the truth-free pilot rule in ../PREREG.md (stream orders used, prequential steps)
K_USE, T_USE = int(os.environ.get("P19_KUSE", "2")), int(os.environ.get("P19_TUSE", "64"))
HINDSIGHT = ["nz05", "nz20"]  # functions of Y: excluded in the pre-registered secondary "forward" subset
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
XOR = ["x2", "x4", "x6"]
PRIMARY = ["FR", "FO", "R4", "R1", "R7b"]


def J(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def tau(a, b):
    t = kendalltau(a, b, variant="b").statistic
    return float(t) if t == t else 0.0


def rho(a, b):
    r = spearmanr(a, b).statistic
    return float(r) if r == r else 0.0


def ci(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return dict(mean=None, ci95=None, n=0)
    m = float(np.mean(xs))
    h = T975.get(n - 1, 1.96) * float(np.std(xs, ddof=1)) / math.sqrt(n) if n > 1 else None
    return dict(mean=round(m, 4), ci95=None if h is None else round(h, 4), n=n)


def load_seed(s):
    tr = [J(f"truth_maze_s{s}_r{r}") for r in TREPS]
    pr = [J(f"pricerep_maze_s{s}_r{r}") for r in range(KREP_P)]
    wr = [J(f"r7brep_maze_s{s}_r{r}") for r in range(KREP_W)]
    fr = J(f"frfo_maze_s{s}")
    missing = dict(truth=sum(x is None for x in tr), price=sum(x is None for x in pr), r7b=sum(x is None for x in wr),
                   frfo=int(fr is None))
    if any(missing.values()):
        return None, missing
    its = list(tr[0]["items"].keys())
    truth_reps = [[t["items"][i]["held"]["gain_vs_base"] for i in its] for t in tr]
    truth = list(np.mean(truth_reps, 0))
    truth_ctl = list(np.mean([[t["items"][i]["held"]["truth"] for i in its] for t in tr], 0))
    rules, reps = {}, {}

    def r1r4(M, it):
        c, z = it.split(":")
        LY, LZ, LYZ = M[f"Y|{c}"]["bits"], M[f"Z|{c}:{z}"]["bits"], M[f"YZ|{c}:{z}"]["bits"]
        return LY - LYZ, LY - LYZ - LZ

    R = [[r1r4(p["members"], i) for i in its] for p in pr]  # (rep, item, 2)
    reps["R1"] = [[x[0] for x in rr] for rr in R]
    reps["R4"] = [[x[1] for x in rr] for rr in R]
    reps["R7b"] = [[w["res"][i][0] for i in its] for w in wr]
    assert K_USE <= fr["K_FR"] and T_USE in fr["budgets"]
    reps["FR"] = [[fr["items"][i][f"FR_T{T_USE}_reps"][k] for i in its] for k in range(K_USE)]
    reps["FR_allK"] = [[fr["items"][i]["FR_reps"][k] for i in its] for k in range(fr["K_FR"])]
    for k in ("R1", "R4", "R7b", "FR", "FR_allK"):
        rules[k] = list(np.mean(reps[k], 0))
    for k in ("FO", "FO1") + tuple(f"FR_T{b}" for b in fr["budgets"]):
        rules[k if k.startswith("FO") else k + "_allK"] = [fr["items"][i][k] for i in its]
    rules["FR_1rep"] = reps["FR"][0]
    rules["R7b_1rep"] = reps["R7b"][0]
    rules["R1_1rep"] = reps["R1"][0]
    rules["R4_1rep"] = reps["R4"][0]
    # compute per ranked item (P8 convention: training forward MACs = stop steps x batch x cells x MACs/cell;
    # FR also counts its coding passes and its share of the current readers)
    n = len(its)
    w7 = float(np.mean([w["work"] for w in wr])) / n
    wk = fr["work"]
    readers = sum(v["reader"] for v in wk.values())
    T = 1024 // 16
    fr_one = sum((v["n_rank"] + 1) * v["fr_member_rep"] for v in wk.values())  # one order, whole stream
    fr_all = readers + K_USE * (T_USE / T) * fr_one
    fo_all = readers + sum(v["n_rank"] * v["fo_item"] for v in wk.values())
    wp = float(np.sum([sum(m["work"] for k, m in p["members"].items() if k.startswith("Y|") or k.split("|", 1)[1] in its)
                       for p in pr]))
    comp = dict(R7b_1rep_item=w7, R7b_item=w7 * KREP_W, FR_item=fr_all / n, FO_item=fo_all / n, R1R4_item=wp / n,
                readers_item=readers / n, FR_ratio_vs_R7b_1rep=fr_all / n / w7, FO_ratio_vs_R7b_1rep=fo_all / n / w7,
                FR_ratio_vs_R7b=fr_all / n / (w7 * KREP_W), R1R4_ratio_vs_R7b_1rep=wp / n / w7)
    for k in fr["budgets"]:
        comp[f"FR_T{k}_allK_ratio_vs_R7b_1rep"] = (readers + fr["K_FR"] * min(k, T) / T * fr_one) / n / w7
    comp["FR_allK_ratio_vs_R7b_1rep"] = (readers + fr["K_FR"] * fr_one) / n / w7
    comp["FR_1rep_ratio_vs_R7b_1rep"] = (readers + T_USE / T * fr_one) / n / w7
    comp["FR_ratio_vs_R7b_1rep_excl_readers"] = (fr_all - readers) / n / w7
    san = {k: dict(v, FR=float(np.mean(v[f"FR_T{T_USE}_reps"][:K_USE])))
           for k, v in fr["items"].items() if k.split(":")[1] in ("copyY", "random", "null", "open", "startZ")}
    return dict(items=its, truth=truth, truth_reps=truth_reps, truth_ctl=truth_ctl, rules=rules, reps=reps, comp=comp,
                sanity=san, fr=fr), missing


def per_seed(d):
    its, truth = d["items"], d["truth"]
    out = dict(taus={k: tau(v, truth) for k, v in d["rules"].items()},
               taus_ctl={k: tau(v, d["truth_ctl"]) for k, v in d["rules"].items()},
               aa_spearman=rho(d["truth_reps"][0], d["truth_reps"][1]),
               aa_tau=tau(d["truth_reps"][0], d["truth_reps"][1]))
    for c in ("empty", "g2g4"):
        ix = [j for j, i in enumerate(its) if i.startswith(c + ":")]
        out[f"taus_{c}"] = {k: tau([v[j] for j in ix], [truth[j] for j in ix]) for k, v in d["rules"].items()}
    ix = [j for j, i in enumerate(its) if i.split(":")[1] not in HINDSIGHT]
    out["taus_forward"] = {k: tau([v[j] for j in ix], [truth[j] for j in ix]) for k, v in d["rules"].items()}
    # tau against the truth with each replicate count (compute curve)
    curve = {}
    for k, reps in d["reps"].items():
        curve[k] = [tau(list(np.mean(reps[:m], 0)), truth) for m in range(1, len(reps) + 1)]
    out["curve"] = curve
    # XOR items: rank displacement (rank by rule vs rank by truth, 1 = best) and sign agreement
    rk = lambda v: (-np.asarray(v)).argsort().argsort() + 1
    rt = rk(truth)
    xo = {}
    for k in PRIMARY:
        rr = rk(d["rules"][k])
        ix = [j for j, i in enumerate(its) if i.split(":")[1] in XOR]
        xo[k] = dict(mean_abs_rank_err=float(np.mean([abs(rr[j] - rt[j]) for j in ix])),
                     values={its[j]: d["rules"][k][j] for j in ix})
    out["xor"] = xo
    out["xor_truth"] = {its[j]: truth[j] for j, i in enumerate(its) if i.split(":")[1] in XOR + ["xb4", "g2", "g4", "g6"]}
    # fg4 vs g4 (empty context): relative gap
    j4, jf = its.index("empty:g4"), its.index("empty:fg4")
    out["fg4_gap"] = {k: (v[jf] - v[j4]) / (abs(v[j4]) + 1e-9) for k, v in d["rules"].items()}
    out["fg4_gap"]["truth"] = (truth[jf] - truth[j4]) / (abs(truth[j4]) + 1e-9)
    ranked_empty = [d["rules"]["FR"][j] for j, i in enumerate(its) if i.startswith("empty:")]
    out["gates"] = dict(G1_aa=bool(out["aa_spearman"] >= 0.80),
                        G2_null=bool(abs(d["sanity"]["empty:null"]["FR"]) < 1e-3 and abs(d["sanity"]["empty:null"]["FO"]) < 1e-3),
                        G3_copyY=bool(d["sanity"]["empty:copyY"]["FR"] > max(ranked_empty)))
    return out


def main():
    S, miss = {}, {}
    for s in SEEDS:
        d, m = load_seed(s)
        miss[s] = m
        if d is not None:
            S[s] = (d, per_seed(d))
    seeds = sorted(S)
    summ = dict(seeds=seeds, missing=miss)
    if not seeds:
        print(json.dumps(summ, indent=1))
        return
    rules = list(S[seeds[0]][1]["taus"].keys())
    summ["tau"] = {k: ci([S[s][1]["taus"][k] for s in seeds]) for k in rules}
    summ["tau_vs_control_truth"] = {k: ci([S[s][1]["taus_ctl"][k] for s in seeds]) for k in rules}
    for c in ("empty", "g2g4"):
        summ[f"tau_{c}"] = {k: ci([S[s][1][f"taus_{c}"][k] for s in seeds]) for k in PRIMARY}
    summ["tau_forward"] = {k: ci([S[s][1]["taus_forward"][k] for s in seeds]) for k in rules}
    summ["config"] = dict(K_USE=K_USE, T_USE=T_USE, KREP_P=KREP_P, KREP_W=KREP_W, hindsight=HINDSIGHT)
    summ["tau_per_seed"] = {k: [round(S[s][1]["taus"][k], 4) for s in seeds] for k in rules}
    summ["paired"] = {f"FR-{k}": ci([S[s][1]["taus"]["FR"] - S[s][1]["taus"][k] for s in seeds]) for k in PRIMARY if k != "FR"}
    summ["paired_forward"] = {f"FR-{k}": ci([S[s][1]["taus_forward"]["FR"] - S[s][1]["taus_forward"][k] for s in seeds])
                              for k in PRIMARY if k != "FR"}
    comp_keys = list(S[seeds[0]][0]["comp"].keys())
    summ["compute"] = {k: ci([S[s][0]["comp"][k] for s in seeds]) for k in comp_keys}
    summ["aa_spearman"] = [round(S[s][1]["aa_spearman"], 4) for s in seeds]
    summ["gates"] = {s: S[s][1]["gates"] for s in seeds}
    summ["xor"] = {k: ci([S[s][1]["xor"][k]["mean_abs_rank_err"] for s in seeds]) for k in PRIMARY}
    summ["xor_values"] = {s: {k: S[s][1]["xor"][k]["values"] for k in PRIMARY} for s in seeds}
    summ["xor_truth"] = {s: S[s][1]["xor_truth"] for s in seeds}
    summ["fg4_gap"] = {k: ci([S[s][1]["fg4_gap"][k] for s in seeds]) for k in list(rules) + ["truth"]}
    summ["curve"] = {k: [ci([S[s][1]["curve"][k][m] for s in seeds])["mean"] for m in range(len(S[seeds[0]][1]["curve"][k]))]
                     for k in S[seeds[0]][1]["curve"]}
    summ["sanity"] = {s: {k: dict(FR=v["FR"], FO=v["FO"]) for k, v in S[s][0]["sanity"].items()} for s in seeds}
    # ---- pre-registered verdict
    g1_fail = sum(not S[s][1]["gates"]["G1_aa"] for s in seeds)
    g2_fail = sum(not S[s][1]["gates"]["G2_null"] for s in seeds)
    g3_fail = sum(not S[s][1]["gates"]["G3_copyY"] for s in seeds)
    t_fr = summ["tau"]["FR"]["mean"]
    ratio = summ["compute"]["FR_ratio_vs_R7b_1rep"]["mean"]
    dfo = summ["paired"]["FR-FO"]["mean"]
    k1 = t_fr < 0.70 or ratio > 0.05
    k2 = dfo < 0.05
    v = dict(n_seeds=len(seeds), G1_fail_seeds=g1_fail, G2_fail_seeds=g2_fail, G3_fail_seeds=g3_fail,
             tau_FR=t_fr, FR_compute_ratio=ratio, tau_FR_minus_FO=dfo, K1=bool(k1), K2=bool(k2))
    if len(seeds) < 5:
        v["status"] = "Incomplete"
    elif g2_fail or g1_fail >= 2:
        v["status"] = "Inconclusive (gate)"
    elif g3_fail >= 2:
        v["status"] = "Inconclusive (FR positive control)"
    elif k1 and k2:
        v["status"] = "Killed (K1 and K2)"
    elif k1:
        v["status"] = "Killed (K1)"
    elif k2:
        v["status"] = "Killed (K2: first-order matches the fast reader)"
    else:
        v["status"] = "Pass"
    summ["verdict"] = v
    json.dump(summ, open(os.path.join(OUT, "p19_summary.json"), "w"), indent=1, default=float)
    chart = dict(
        title="P19: Kendall tau-b vs deployed gain over base (44 maze items, mean over seeds) against compute per item",
        rules={k: dict(tau=summ["tau"][k]["mean"], ci95=summ["tau"][k]["ci95"],
                       compute_item=summ["compute"].get(f"{k}_item", {}).get("mean")) for k in PRIMARY},
        curve=summ["curve"], per_seed=summ["tau_per_seed"], kill=dict(tau=0.70, compute_ratio=0.05, fr_minus_fo=0.05))
    json.dump(chart, open(os.path.join(OUT, "p19_chart.json"), "w"), indent=1)
    print(json.dumps(dict(verdict=v, tau=summ["tau"], paired=summ["paired"], aa=summ["aa_spearman"],
                          compute={k: summ["compute"][k] for k in ("FR_ratio_vs_R7b_1rep", "FO_ratio_vs_R7b_1rep", "R7b_item")}),
                     indent=1))


if __name__ == "__main__":
    main()
