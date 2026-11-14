"""P11 analysis: selection (validation only), gates, paired-by-seed CIs, hierarchical bootstrap, mechanism metrics.

  python3 analyze.py [RESULTS_DIR]      default results/vm -> results/summary.json, results/chart.json
Readouts for R-prog, DT-L and R-final are chosen once per arm on validation (mean over the arm's seeds of the
size-17 validation score at 10x and 100x, select_p11 rule). M arms are scored on b ('inst').
"""
import glob
import json
import math
import os
import sys

import numpy as np

from select_p11 import best_readout, family, hpkey, table
from vendored import ci

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = {"E1": ("test17", 1000), "E2": ("test29", 100), "E3": ("test59", 100)}
CORE = ("R-prog", "DT-L", "M-B", "M-budget", "M-TV")
SECONDARY = ("R-final", "M-cap", "M-clip")
R_FAM = ("R-prog", "DT-L", "R-final")
B = 10_000


def bits(h, n):
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), np.uint8))[:n].astype(bool)


def acc_at(setres, readout, ratio):
    return table(setres)[readout][ratio]


def bits_at(setres, readout, ratio):
    n = setres["n"]
    ratios = [c["ratio"] for c in setres["checkpoints"]]
    i = ratios.index(ratio)
    if readout in setres["checkpoints"][i]["bits"]:
        return bits(setres["checkpoints"][i]["bits"][readout], n)
    return bits(setres["derived"][readout]["bits"][i], n)


def load_stage(res, stage, arm, hp):
    out = {}
    for d in sorted(glob.glob(os.path.join(res, stage, f"{arm}_{hpkey(hp)}_s*"))):
        p = os.path.join(d, "eval_core.json")
        if os.path.exists(p):
            e = json.load(open(p))
            e["train"] = json.load(open(os.path.join(d, "train.json")))
            out[e["seed"]] = e
    return out


def pick_readout(evs, allowed=None):
    tabs = [table(e["sets"]["val17"]) for e in evs.values()]
    common = set.intersection(*[set(t) for t in tabs])
    mean_tab = {r: {k: float(np.mean([t[r][k] for t in tabs])) for k in (10, 100)} for r in common}
    return best_readout(mean_tab, allowed)


def endpoints(e, ro):
    v = {k: acc_at(e["sets"][s], ro, r) for k, (s, r) in CELLS.items()}
    v["mean"] = float(np.mean([v[k] for k in CELLS]))
    return v


def paired(a, b, key):
    """a, b: {seed: {key: value}} -> paired t-CI of a - b and the sign count."""
    seeds = sorted(set(a) & set(b))
    d = [a[s][key] - b[s][key] for s in seeds]
    c = ci(d)
    return {"seeds": seeds, "diffs": d, "mean": c["mean"] if c else None, "ci95": c["ci95"] if c else None,
            "lo": (c["mean"] - c["ci95"]) if c and c["ci95"] is not None else None,
            "hi": (c["mean"] + c["ci95"]) if c and c["ci95"] is not None else None,
            "n_pos": int(sum(x > 0 for x in d)), "n_neg": int(sum(x < 0 for x in d)), "n": len(d)}


def hboot(evs_a, ro_a, evs_b, ro_b, cells=CELLS, seed=0):
    """Hierarchical bootstrap of mean over cells of acc(a) - acc(b): resample seeds, then mazes within each
    set (same mazes for both readouts: paired by maze)."""
    rng = np.random.default_rng(seed)
    seeds = sorted(set(evs_a) & set(evs_b))
    D = {}
    for s in seeds:
        D[s] = {k: bits_at(evs_a[s]["sets"][st], ro_a, r).astype(float) - bits_at(evs_b[s]["sets"][st], ro_b, r).astype(float)
                for k, (st, r) in cells.items()}
    point = float(np.mean([np.mean([D[s][k].mean() for k in cells]) for s in seeds]))
    stats = np.empty(B)
    for i in range(B):
        ss = rng.choice(seeds, len(seeds))
        vals = []
        for s in ss:
            vals.append(np.mean([D[s][k][rng.integers(0, len(D[s][k]), len(D[s][k]))].mean() for k in cells]))
        stats[i] = np.mean(vals)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n_seeds": len(seeds)}


def persistence(setres, ro):
    a = bits_at(setres, ro, 10)
    b = bits_at(setres, ro, 1000)
    return float((a & b).sum() / max(1, a.sum())), int(a.sum())


def curve(evs, ro, setname):
    rat = [c["ratio"] for c in next(iter(evs.values()))["sets"][setname]["checkpoints"]]
    out = {}
    for r in rat:
        c = ci([acc_at(e["sets"][setname], ro, r) for e in evs.values()])
        out[r] = [c["mean"], c["ci95"]]
    return out


def late_corr(e, stream="inst", sets=("test17", "test29", "test59")):
    num = sum(e["sets"][s]["late_correction"][stream]["num"] for s in sets)
    den = sum(e["sets"][s]["late_correction"][stream]["den"] for s in sets)
    return num / den if den else float("nan"), num, den


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "vm")
    P = json.load(open(os.path.join(res, "params.json")))
    S = {"params": {k: P[k] for k in ("hp", "gates", "steps", "width", "core_seeds", "secondary_seeds", "flags")},
         "pilot_scores": {a: {k: {kk: vv for kk, vv in v.items() if kk in ("score", "readout", "inst_score",
                                                                        "val9_1x_inst", "val17_peak_inst", "val29_peak_inst")}
                              for k, v in d["rows"].items()} for a, d in P["arms"].items()}}
    hr = os.path.join(res, "headroom.json")
    S["headroom"] = json.load(open(hr)) if os.path.exists(hr) else None
    EV, RO = {}, {}
    for arm in CORE + SECONDARY:
        stage = "core" if arm in CORE else "secondary"
        evs = load_stage(res, stage, arm, P["hp"][arm])
        if not evs:
            continue
        EV[arm] = evs
        if arm in R_FAM:
            RO[arm] = {"best": pick_readout(evs), "lmsr": pick_readout(evs, ("lmsr",)), "tv": pick_readout(evs, ("tv",)),
                       "inst": ("inst", None)}
        else:
            RO[arm] = {"best": ("inst", None)}
    S["readouts"] = {a: {k: v[0] for k, v in d.items()} for a, d in RO.items()}
    S["readout_val_scores"] = {a: {k: v[1] for k, v in d.items()} for a, d in RO.items()}
    # ------------------------------------------------------------------ endpoints per arm and seed
    END = {a: {s: endpoints(e, RO[a]["best"][0]) for s, e in evs.items()} for a, evs in EV.items()}
    S["endpoints"] = {a: {"per_seed": {str(s): v for s, v in d.items()},
                          **{k: ci([v[k] for v in d.values()]) for k in list(CELLS) + ["mean"]}} for a, d in END.items()}
    for a in R_FAM:
        if a in EV:
            S["endpoints"][a]["inst"] = {k: ci([endpoints(e, "inst")[k] for e in EV[a].values()])
                                         for k in list(CELLS) + ["mean"]}
            for tr in ("lmsr", "tv"):
                S["endpoints"][a][tr] = {k: ci([endpoints(e, RO[a][tr][0])[k] for e in EV[a].values()])
                                         for k in list(CELLS) + ["mean"]}
    # in-distribution (size 9 at T_train) with each arm's scored readout
    IND = {a: {s: {"ind": acc_at(e["sets"]["test9"], RO[a]["best"][0], 1)} for s, e in evs.items()} for a, evs in EV.items()}
    S["in_dist_test9_1x"] = {a: ci([v["ind"] for v in d.values()]) for a, d in IND.items()}
    # ------------------------------------------------------------------ paired comparisons
    cmp = {}
    if "M-budget" in END:
        for other in ("M-TV", "R-prog", "DT-L", "M-B"):
            if other in END:
                cmp[f"M-budget - {other}"] = {k: paired(END["M-budget"], END[other], k) for k in list(CELLS) + ["mean"]}
        if "R-prog" in IND:
            cmp["in_dist: M-budget - R-prog"] = paired(IND["M-budget"], IND["R-prog"], "ind")
    if "M-cap" in END and "M-clip" in END:
        cmp["M-cap - M-clip"] = {k: paired(END["M-cap"], END["M-clip"], k) for k in list(CELLS) + ["mean"]}
    S["paired"] = cmp
    # readout comparisons within a model (hierarchical bootstrap)
    hb = {}
    for a in R_FAM:
        if a in EV:
            hb[f"{a}: lmsr - tv tracker"] = hboot(EV[a], RO[a]["lmsr"][0], EV[a], RO[a]["tv"][0])
            hb[f"{a}: best - inst"] = hboot(EV[a], RO[a]["best"][0], EV[a], "inst")
    S["hboot"] = hb
    # ------------------------------------------------------------------ mechanism: late correction
    LC = {}
    for a, evs in EV.items():
        LC[a] = {s: {"rate": late_corr(e)[0]} for s, e in evs.items()}
    S["late_correction_rate"] = {a: ci([v["rate"] for v in d.values()]) for a, d in LC.items()}
    for a in R_FAM:
        if a in EV:
            for tr in ("lmsr", "tv"):
                S["late_correction_rate"][f"{a}:{RO[a][tr][0]}"] = ci([late_corr(e, RO[a][tr][0])[0] for e in EV[a].values()])
    if "M-budget" in LC and "M-TV" in LC:
        cmp["late_correction: M-budget - M-TV"] = paired(LC["M-budget"], LC["M-TV"], "rate")
    # ------------------------------------------------------------------ budget use
    BU = {}
    for a, evs in EV.items():
        names = ["inst"] if a not in R_FAM else [RO[a]["lmsr"][0], RO[a]["tv"][0]]
        for nm in names:
            rows = {}
            for st in ("test17", "test29", "test59"):
                bl = [e["sets"][st].get("budget", {}).get(nm) for e in evs.values()]
                bl = [b for b in bl if b]
                if not bl:
                    continue
                rows[st] = {"exhausted_while_wrong": ci([b["onpath_exhausted_while_wrong"] for b in bl]),
                            "exhausted": ci([b["onpath_exhausted"] for b in bl]),
                            "spearman_t50_near": ci([b["spearman_t50_vs_near_dist"] for b in bl]),
                            "mean_remaining": ci([b["mean_remaining_frac_open"] for b in bl]),
                            "by_len_bin": [{"exact_end": ci([b["by_path_len_bin"][i]["exact_end"] for b in bl]),
                                            "exhaust": ci([b["by_path_len_bin"][i]["exhaust_frac_onpath"] for b in bl]),
                                            "len_range": bl[0]["by_path_len_bin"][i]["len_range"]} for i in range(3)]}
            if rows:
                BU[f"{a}:{nm}"] = rows
    S["budget_use"] = BU
    # ------------------------------------------------------------------ stability
    ST = {}
    for a, evs in EV.items():
        ro = RO[a]["best"][0]
        row = {}
        for st in ("test9", "test17"):
            per = [persistence(e["sets"][st], ro) for e in evs.values()]
            row[f"persistence_{st}"] = ci([p[0] for p in per])
            if a in R_FAM and ro != "inst":
                row[f"persistence_{st}_inst"] = ci([persistence(e["sets"][st], "inst")[0] for e in evs.values()])
            ck = {c["ratio"]: c for c in next(iter(evs.values()))["sets"][st]["checkpoints"]}
            for key in ("flips_drift_per_px_per_1k", "flips_p2_per_px_per_1k"):
                row[f"{key}_{st}_1000x"] = ci([{c["ratio"]: c for c in e["sets"][st]["checkpoints"]}[1000][key]["inst"]
                                               for e in evs.values()])
            for key in ("dphi1", "dphi2", "late_velocity"):
                row[f"{key}_{st}_1000x"] = ci([{c["ratio"]: c for c in e["sets"][st]["checkpoints"]}[1000].get(key)
                                               for e in evs.values()])
        ST[a] = row
    S["stability"] = ST
    # sensitivity gate: R-final inst or M-B loses >= 10 points 10x -> 1000x on size 9 or 17
    sens = {}
    for a in ("R-final", "M-B"):
        if a in EV:
            for st in ("test9", "test17"):
                c10 = np.mean([acc_at(e["sets"][st], "inst", 10) for e in EV[a].values()])
                c1k = np.mean([acc_at(e["sets"][st], "inst", 1000) for e in EV[a].values()])
                sens[f"{a} {st}"] = {"10x": float(c10), "1000x": float(c1k), "drop": float(c10 - c1k)}
    S["sensitivity_gate"] = {"rows": sens, "pass": any(v["drop"] >= 0.10 - 1e-9 for v in sens.values())}
    # path independence
    S["path_independence"] = {a: {"spread_exact": ci([e["path_independence"]["spread_exact"] for e in evs.values()
                                                      if "path_independence" in e]),
                                  "rel_state_dist": ci([float(np.mean(e["path_independence"]["rel_state_dist"]))
                                                        for e in evs.values() if "path_independence" in e])}
                              for a, evs in EV.items()}
    # health
    S["health"] = {a: {"sat_frac_test17": ci([e["sets"]["test17"].get("sat_frac") for e in evs.values()]),
                       "jump_frac_test17": ci([e["sets"]["test17"].get("jump_frac_u_ge_2_after_Ttrain") for e in evs.values()]),
                       "nonfinite_total": int(sum(c["nonfinite"] for e in evs.values() for st in e["sets"].values()
                                                  for c in st["checkpoints"])),
                       "maxabs_inst_test17_1000x": max({c["ratio"]: c for c in e["sets"]["test17"]["checkpoints"]}[1000]["maxabs"]["inst"]
                                                       for e in evs.values()),
                       "vendored_ok": all(e["train"].get("vendored") for e in evs.values()),
                       "final_loss_finite": all(e["train"]["final_loss_finite"] for e in evs.values()),
                       "params": next(iter(evs.values()))["train"]["params"],
                       "train_sec": ci([e["train"]["train_sec"] for e in evs.values()]),
                       "eval_sec": ci([e["eval_sec"] for e in evs.values()])}
                   for a, evs in EV.items()}
    # velocity law: pilot M-cap models, size 9 validation to 1000x
    vl = {}
    for eps in (0.1, 0.3, 1.0):
        p = os.path.join(res, "pilot", f"M-cap_{hpkey({'eps': eps})}_s0", "eval_pilot.json")
        if os.path.exists(p):
            e = json.load(open(p))
            v9 = e["sets"]["val9"]
            ck = {c["ratio"]: c for c in v9["checkpoints"]}
            vl[str(eps)] = {"acc": {r: ck[r]["acc"]["inst"] for r in ck},
                            "flips_per_px_per_1k_1000x": ck[1000]["flips_drift_per_px_per_1k"]["inst"] + ck[1000]["flips_p2_per_px_per_1k"]["inst"],
                            "flips_by_ratio": {r: ck[r]["flips_drift_per_px_per_1k"]["inst"] + ck[r]["flips_p2_per_px_per_1k"]["inst"] for r in ck},
                            "jump_frac": v9.get("jump_frac_u_ge_2_after_Ttrain")}
    S["velocity_law_pilot_mcap"] = vl
    # ------------------------------------------------------------------ floor + verdict
    floor = {}
    for k in CELLS:
        best = max((S["endpoints"][a][k]["mean"], a) for a in END)
        floor[k] = {"best_arm": best[1], "best": best[0], "inconclusive": best[0] < 0.20}
    S["floor"] = floor
    gates_ok = bool(P["gates"]["training"]["pass"] and P["gates"]["floor"]["pass"])
    above_floor = not any(v["inconclusive"] for v in floor.values())
    v = {}
    if all(a in END for a in ("M-budget", "M-TV", "R-prog", "DT-L")):
        c_tv = cmp["M-budget - M-TV"]["mean"]
        hb_rp = hb.get("R-prog: lmsr - tv tracker")
        v["S-A"] = bool(c_tv["lo"] is not None and c_tv["lo"] > 0 and c_tv["n_pos"] >= 4 and hb_rp and hb_rp["lo"] > 0)
        sb = []
        for o in ("R-prog", "DT-L"):
            c = cmp[f"M-budget - {o}"]["mean"]
            sb.append(c["lo"] is not None and c["lo"] > 0 and c["n_pos"] >= 4)
        ind = cmp["in_dist: M-budget - R-prog"]
        v["S-B"] = bool(all(sb) and ind["mean"] >= -0.02)
        lc = cmp.get("late_correction: M-budget - M-TV")
        v["S-C"] = bool(lc and lc["lo"] is not None and lc["lo"] > 0)
        f1a = c_tv["mean"] is not None and (c_tv["mean"] <= 0 or (c_tv["lo"] is not None and c_tv["lo"] <= 0 <= c_tv["hi"]
                                                                 and abs(c_tv["mean"]) < 0.03))
        f1b = not (hb_rp and hb_rp["lo"] > 0)
        v["F1"] = bool(f1a and f1b)
        f2 = 0
        f2_cells = {}
        for k in CELLS:
            mb = S["endpoints"]["M-budget"][k]["mean"]
            bar = max(S["endpoints"]["R-prog"][k]["mean"], S["endpoints"]["DT-L"][k]["mean"])
            f2_cells[k] = {"M-budget": mb, "best_bar": bar, "bar_wins": bar >= mb}
            f2 += bar >= mb
        v["F2_cells"] = f2_cells
        v["F2"] = bool(f2 >= 2 or (S["headroom"] or {}).get("skip_s3", False))
        v["gates_ok"] = gates_ok
        v["above_floor"] = above_floor
        if not (gates_ok and above_floor):
            v["verdict"] = "INCONCLUSIVE"
        elif v["S-A"] and v["S-B"] and v["S-C"]:
            v["verdict"] = "SUPPORTED"
        elif v["F1"] or v["F2"]:
            v["verdict"] = "FALSIFIED" + (" (F1)" if v["F1"] else "") + (" (F2)" if v["F2"] else "")
        else:
            v["verdict"] = "MIXED / NOT SUPPORTED"
        mixed = []
        if v["S-A"] and v["F2"]:
            mixed.append("LMSR beats TV, but budgets lose to known fixes")
        if v["F1"] and all(sb):
            mixed.append("a spending budget helps, the market shape adds nothing")
        if family(RO["R-prog"]["best"][0]) in ("lmsr", "tv"):
            c = cmp["M-budget - R-prog"]["mean"]
            if not (c["lo"] is not None and c["lo"] > 0):
                mixed.append("inference-time effect only")
        v["mixed"] = mixed
    S["verdict"] = v
    # ------------------------------------------------------------------ chart data: accuracy vs depth ratio
    chart = {"ratios": [1, 3, 10, 30, 100, 300, 1000], "series": {}}
    for a, evs in EV.items():
        ros = {"best": RO[a]["best"][0]}
        if a in R_FAM:
            ros.update({"inst": "inst", "lmsr": RO[a]["lmsr"][0], "tv": RO[a]["tv"][0]})
        for tag, ro in ros.items():
            for st in ("test9", "test17", "test29", "test59"):
                chart["series"][f"{a}|{tag}={ro}|{st}"] = curve(evs, ro, st)
    chart["velocity_law"] = vl
    json.dump(chart, open(os.path.join(os.path.dirname(res.rstrip("/")), "chart.json"), "w"), indent=1)
    json.dump(S, open(os.path.join(os.path.dirname(res.rstrip("/")), "summary.json"), "w"), indent=1)
    # console digest
    print("readouts", json.dumps(S["readouts"]))
    for a in END:
        e = S["endpoints"][a]
        print(f"{a:9s} " + " ".join(f"{k}={e[k]['mean']:.3f}±{(e[k]['ci95'] or 0):.3f}" for k in list(CELLS) + ["mean"]))
    for k, c in cmp.items():
        c = c["mean"] if "mean" in c and isinstance(c["mean"], dict) else c
        if c["mean"] is None:
            print(f"{k:40s} n/a (no finite paired differences)")
            continue
        print(f"{k:40s} {c['mean']:+.3f} [{c['lo'] if c['lo'] is None else round(c['lo'], 3)}, "
              f"{c['hi'] if c['hi'] is None else round(c['hi'], 3)}] +{c['n_pos']}/-{c['n_neg']} of {c['n']}")
    for k, c in hb.items():
        print(f"{k:40s} {c['mean']:+.3f} [{c['lo']:.3f}, {c['hi']:.3f}]")
    print("sensitivity", json.dumps(S["sensitivity_gate"]))
    print("floor", json.dumps(floor))
    print("verdict", json.dumps({k: v for k, v in S["verdict"].items() if k != "F2_cells"}))


if __name__ == "__main__":
    main()
