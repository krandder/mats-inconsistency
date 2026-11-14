"""P0 analysis: gates R1-R3, findings F1-F3, falsifiers, shift grid, halting Pareto, costs.

  python analyze.py            reads results/vm/<lane>/{eval,ckpt,markers,logs}, writes results/p0_results.json
                               and results/chart_p0.json, prints a short table.
Stats: per-seed values; success = seed >= 0.90; median; mean with a 95% t-interval; paired bootstrap over instances
(2000 resamples) for per-seed differences; one-sided Fisher exact on success counts.
"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")
MRPILOT = os.path.join(HERE, "..", "..", "market-resnet", "results", "pilot")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
rng = np.random.default_rng(0)


def load_correct(npz):
    z = np.load(npz)
    return np.unpackbits(z["correct_bits"], axis=1, count=int(z["T_cap"])).astype(bool)


EV = {}
for sp in glob.glob(os.path.join(RES, "vm", "*", "eval", "*", "summary.json")):
    EV[os.path.basename(os.path.dirname(sp))] = (os.path.dirname(sp), json.load(open(sp)))
# per-set directory; the re-planned lanes put m59 / shift / stability / stretch sets in tail jobs of their own
# (e.g. M1_s0_m59, Mref_shift): merge those sets into the arm's main eval so every lookup sees one set list
SETDIR = {(e, st): d for e, (d, sm) in EV.items() for st in sm["sets"]}
for e in sorted(EV):
    for suf in ("_m59", "_shift", "_stab", "_stretch"):
        if e.endswith(suf):
            base = e[:-len(suf)]
            base = base if base == "Mref" or base.endswith("_best") else base + "_best"
            if base in EV:
                for st, v in EV[e][1]["sets"].items():
                    EV[base][1]["sets"].setdefault(st, v)
                    SETDIR.setdefault((base, st), EV[e][0])


def acc(eid, st, t):
    if eid not in EV or st not in EV[eid][1]["sets"]:
        return None
    s = EV[eid][1]["sets"][st]
    if t in s["grid"]:
        return s["acc"][s["grid"].index(t)]
    c = corr(eid, st)                       # off-grid iteration: read it from the per-iteration correctness bits
    return float(c[:, t - 1].mean()) if c is not None and 1 <= t <= c.shape[1] else None


def corr(eid, st):
    p = os.path.join(SETDIR.get((eid, st), ""), st + ".npz")
    return load_correct(p) if os.path.exists(p) else None


def ci(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    m = float(np.mean(v))
    h = T975.get(len(v) - 1, 1.96) * float(np.std(v, ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else None
    return {"mean": round(m, 4), "ci95": round(h, 4) if h is not None else None, "median": round(float(np.median(v)), 4),
            "n": len(v), "success": int(sum(x >= 0.90 for x in v)), "per_seed": [round(x, 4) for x in v]}


def fisher_less(a, n1, b, n2):
    """one-sided p that arm 1 (a of n1 successes) has fewer successes than arm 2 (b of n2), hypergeometric"""
    K, N = a + b, n1 + n2
    p = 0.0
    for x in range(0, a + 1):
        if K - x <= n2 and x <= n1:
            p += math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    return p


def paired_boot(e1, e2, st, t, B=2000):
    c1, c2 = corr(e1, st), corr(e2, st)
    if c1 is None or c2 is None or t > c1.shape[1] or t > c2.shape[1]:
        return None
    d = c1[:, t - 1].astype(float) - c2[:, t - 1].astype(float)
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(1)
    return {"diff": round(float(d.mean()), 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
            "hi": round(float(np.percentile(bs, 97.5)), 4)}


OUT = {"evals_found": len(EV)}
MAZE_ARMS = {"M1": 3, "Mprog": 3, "M4": 3, "M4cm": 1, "M5": 1, "M1long": 1}   # M5 one seed (spec cut 3)
PRE_ARMS = {"P1": 8, "P4": 8, "P5": 4, "P4cm": 3, "PL": 3, "P6": 3, "MRP": 3}   # P6 cut (spec cut 2); stays absent
MAZE_EP = [("m33", 100), ("m33", 300), ("m33", 1000), ("m59", 3000), ("m33h", 3000)]
PRE_EP = [("fL128", 300), ("fL256", 300), ("fL512", 300), ("fL512", 3000), ("e2h512", 300)]


def mid(arm, s, which="best"):
    if arm.startswith(("P", "MR")):
        return f"{arm}_s{s}"
    return f"{arm}_s{s}_{which}"


# ------------------------------------------------------------------ endpoints
ep = {}
for arm, ns in MAZE_ARMS.items():
    for which in ("best", "final"):
        for st, t in MAZE_EP:
            v = [acc(mid(arm, s, which), st, t) for s in range(ns)]
            if any(x is not None for x in v):
                ep[f"{arm}/{which}/{st}@{t}"] = ci(v)
for st, t in MAZE_EP + [("m9", 300), ("m13", 300), ("m201", 3000)]:
    a_ = acc("Mref", st, t)
    if a_ is not None:
        ep[f"Mref/{st}@{t}"] = round(a_, 4)
for arm, ns in PRE_ARMS.items():
    for st, t in PRE_EP:
        v = [acc(mid(arm, s), st, t) for s in range(ns)]
        if any(x is not None for x in v):
            ep[f"{arm}/{st}@{t}"] = ci(v)
OUT["endpoints"] = ep


# ------------------------------------------------------------------ R1 harness
def r1():
    r = {}
    ut = glob.glob(os.path.join(RES, "vm", "*", "unit_tests.json"))
    if ut:
        u = json.load(open(ut[0]))
        r["unit_tests"] = {"all_ok": u.get("all_ok"), "dtnet_max_abs_diff": u["dtnet_repro"]["max_abs_diff"],
                           "params_ok": all(v["ok"] for v in u["params"].values()), "dtl_params": u["dtl_params"]}
    rc = glob.glob(os.path.join(RES, "vm", "*", "resume_check.json"))
    if rc:
        r["resume_check"] = json.load(open(rc[0]))
    src = "Mref" if "Mref" in EV else "Mref_r1"
    m59 = acc("Mref", "m59", 3000)
    r["mref_m59_3000"] = {"acc": m59, "ok": m59 is not None and m59 >= 0.97}
    sh = {}
    for e in ("Mref_r1", "Mref"):
        if e in EV and "sh_n9_p0_de0" in EV[e][1]["sets"]:
            s = EV[e][1]["sets"]["sh_n9_p0_de0"]
            sh[e] = {"acc_at_Tcap": s["acc"][-1], "acc_30": acc(e, "sh_n9_p0_de0", 30), "peak": max(s["acc"])}
    a9 = next(iter(sh.values()))["acc_at_Tcap"] if sh else None
    r["mref_sh_n9_de0"] = {"detail": sh, "target": "0.72 +- 0.08", "ok": a9 is not None and abs(a9 - 0.72) <= 0.08}
    e2h = []
    for e in ("Mref_r1", "Mref"):
        if e in EV and "sh_n33_p0_de1" in EV[e][1]["sets"]:
            e33 = "m33r1" if "m33r1" in EV[e][1]["sets"] else "m33"
            for t in [g for g in EV[e][1]["sets"]["sh_n33_p0_de1"]["grid"] if 10 <= g <= 100]:
                a1, a2 = acc(e, e33, t), acc(e, "sh_n33_p0_de1", t)
                if a1 is not None and a2 is not None:
                    e2h.append({"eval": e, "t": t, "e2h": a1, "maze_dataset": a2, "diff": round(a1 - a2, 4)})
    r["n33_e2h_vs_mazedataset"] = {"points": e2h, "max_abs_diff": max((abs(x["diff"]) for x in e2h), default=None),
                                   "ok": bool(e2h) and max(abs(x["diff"]) for x in e2h) <= 0.05}
    # structure: dead-end-filling rounds per maze and M-ref's first stably-correct iteration per round
    struct = {}
    for e, st, f, key, n in (("Mref_r1", "m33r1", "mazes_e2h.npz", "m33", 100),
                             ("Mref_r1", "sh_n33_p0_de1", "mazes_shift.npz", "sh_n33_p0_de1", 100)):
        fz = os.path.join(HERE, "frozen", f)
        p = os.path.join(EV[e][0], st + ".npz") if e in EV else ""
        if not (os.path.exists(fz) and os.path.exists(p)):
            continue
        z = np.load(fz)
        dr, di, sp = (z[f"{key}/{k}"][:n].astype(float) for k in ("deadend_rounds", "depth_iters", "sp_len"))
        fs = np.load(p)["first_stable"].astype(float)
        ok = fs > 0
        struct[st] = {"deadend_rounds_mean": round(dr.mean(), 1), "depth_iters_mean": round(di.mean(), 1),
                      "sp_len_mean": round(sp.mean(), 1),
                      "first_stable_median": float(np.median(fs[ok])) if ok.any() else None,
                      "first_stable_per_deadend_round_median": round(float(np.median(fs[ok] / dr[ok])), 3) if ok.any() else None,
                      "corr_first_stable_deadend_rounds": round(float(np.corrcoef(fs[ok], dr[ok])[0, 1]), 3) if ok.sum() > 2 else None}
    r["n33_e2h_vs_mazedataset"]["structure"] = struct
    rr = [v["first_stable_per_deadend_round_median"] for v in struct.values()]
    r["n33_e2h_vs_mazedataset"]["ok_per_deadend_round"] = (len(rr) == 2 and None not in rr
                                                           and abs(rr[0] / rr[1] - 1) <= 0.05)
    dis = {}
    for e, (_, sm) in EV.items():
        for st, s in sm["sets"].items():
            if "check_vs_exact_disagree" in s and (st.startswith(("m9", "m13", "m33", "m59", "v9", "v33", "m201")) or "_p0_" in st):
                dis[f"{e}/{st}"] = s["check_vs_exact_disagree"]
    r["check_vs_exact_on_p0"] = {"n_evals": len(dis), "total_disagree": sum(dis.values()),
                                 "nonzero": {k: v for k, v in dis.items() if v}, "ok": bool(dis) and sum(dis.values()) == 0}
    rep = {}
    for a in ("A", "B", "Bds", "C", "D"):
        e = f"pilotMR_{a}"
        pj = os.path.join(MRPILOT, f"pilot_{a}.eval.json")
        if e not in EV or not os.path.exists(pj):
            continue
        ref = json.load(open(pj))["sets"]
        worst = 0.0
        for k, rs in ref.items():
            s = EV[e][1]["sets"].get("legacy_" + k)
            if s is None:
                continue
            for t, a_ in zip(s["grid"], s["acc"]):
                if t < len(rs["fixed"]):
                    worst = max(worst, abs(a_ - rs["fixed"][t]["acc"]))
        rep[a] = round(worst, 5)
    r["mr_pilot_repro"] = {"max_abs_diff": rep, "ok": len(rep) == 5 and max(rep.values()) <= 0.005}
    bw = {e: sm.get("bitwise_repeat", {}).get("identical") for e, (_, sm) in EV.items() if "bitwise_repeat" in sm}
    r["bitwise_repeat"] = {"evals": bw, "ok": bool(bw) and all(bw.values())}
    tf = {}
    for m in ("fp32", "tf32"):
        e = f"Mref_m59x20_{m}"
        if e in EV:
            s = EV[e][1]["sets"]["m59x20"]
            tf[m] = {"acc_3000": s["acc"][-1], "sec": EV[e][1]["sec"]}
    r["tf32_vs_fp32_m59x20"] = tf
    keys = ["mref_m59_3000", "mref_sh_n9_de0", "n33_e2h_vs_mazedataset", "check_vs_exact_on_p0", "mr_pilot_repro",
            "bitwise_repeat"]
    r["ok"] = all(r[k]["ok"] for k in keys) and r.get("unit_tests", {}).get("all_ok") is True
    # same gate with the n=33 comparison made per dead-end-filling round (the two generators differ in structure)
    r["ok_with_n33_per_round"] = (all(r[k]["ok"] for k in keys if k != "n33_e2h_vs_mazedataset")
                                  and r["n33_e2h_vs_mazedataset"].get("ok_per_deadend_round") is True
                                  and r.get("unit_tests", {}).get("all_ok") is True)
    return r


OUT["R1"] = r1()


# ------------------------------------------------------------------ R2 maze bar, overthinking
def chosen_iter(eid, vset):
    if eid not in EV or vset not in EV[eid][1]["sets"]:
        return None
    s = EV[eid][1]["sets"][vset]
    return s["grid"][int(np.argmax(s["acc"]))]


def overthink(eid, vset, tset):
    t = chosen_iter(eid, vset)
    if t is None or acc(eid, tset, 3000) is None or acc(eid, tset, t) is None:
        return None
    return {"chosen_t": t, "acc_chosen": acc(eid, tset, t), "acc_3000": acc(eid, tset, 3000),
            "drop": round(acc(eid, tset, t) - acc(eid, tset, 3000), 4)}


ot = {}
for arm, ns in MAZE_ARMS.items():
    for s in range(ns):
        o = overthink(mid(arm, s), "v33", "m33h")
        if o:
            ot[mid(arm, s)] = o
o = overthink("Mref", "v33", "m33h")
if o:
    ot["Mref"] = o
for arm, ns in PRE_ARMS.items():
    for s in range(ns):
        o = overthink(mid(arm, s), "vL512", "fL512")
        if o:
            ot[mid(arm, s)] = o
OUT["overthinking"] = ot
m1 = [max(acc(mid("M1", s), "m33", 300) or 0, acc(mid("M1", s), "m33", 1000) or 0) for s in range(3)]
m1_300 = ep.get("M1/best/m33@300", {}) or {}
m1_1000 = ep.get("M1/best/m33@1000", {}) or {}
lm = acc("M1long_s0_best", "m59", 3000)
drops = [ot[mid("M1", s)]["drop"] for s in range(3) if mid("M1", s) in ot]
OUT["R2"] = {"M1_median_m33_300": m1_300.get("median"), "M1_median_m33_1000": m1_1000.get("median"),
             "M1long_m59_3000": lm, "M1_overthinking_drops": drops,
             "M1long_m33_300": acc("M1long_s0_best", "m33", 300),
             "ok_median": max(m1_300.get("median") or 0, m1_1000.get("median") or 0) >= 0.90,
             "ok_long": lm is not None and lm >= 0.70,
             "ok_overthink": bool(drops) and float(np.median(drops)) <= 0.05}
OUT["R2"]["ok"] = OUT["R2"]["ok_median"] and OUT["R2"]["ok_long"] and OUT["R2"]["ok_overthink"]
# falsifier 2 branch (spec): (a) probe still rising at 15k and M1-long passes -> under-training, use 40k steps;
# (b) M1-long < 0.50 on 33x33 while M-ref passes -> small width fails, the maze bar becomes M-ref (w=128);
# (c) M1-long in [0.50, 0.90) -> weak bar. M1-long here stopped at 30k steps (cut 4), not 40k.
_l33 = max(acc("M1long_s0_best", "m33", 300) or 0, acc("M1long_s0_best", "m33", 1000) or 0) \
    if "M1long_s0_best" in EV else None
_r33 = acc("Mref", "m33", 300)
OUT["R2"]["M1long_m33_best_of_300_1000"] = _l33
OUT["R2"]["Mref_m33_300"] = _r33
OUT["R2"]["M1_val9_best_per_seed"] = None   # filled from the training logs below
if not OUT["R2"]["ok"] and _l33 is not None:
    OUT["R2"]["falsify2_branch"] = ("b: small width fails at this budget; maze bar = M-ref (w=128)"
                                    if _l33 < 0.50 and (_r33 or 0) >= 0.90 else
                                    "c: weak bar (keep w=64, later gains must also beat M1-long)" if _l33 < 0.90 else
                                    "a: under-training at 15k (M1-long passes); later maze experiments use the long schedule")
probe = {}
for arm in ("M1long", "M1", "Mprog", "M4", "M4cm", "M5"):
    for s in range(MAZE_ARMS[arm]):
        cps = glob.glob(os.path.join(RES, "vm", "*", "ckpt", f"{arm}_s{s}.pt"))
        if cps:
            import torch
            lg = torch.load(cps[0], map_location="cpu", weights_only=False)["log"]
            probe[f"{arm}_s{s}"] = {"probe33_acc300": {e["step"]: e["probe33_acc300"] for e in lg if "probe33_acc300" in e},
                                    "best_val9": max((e.get("val9_acc30", 0) for e in lg), default=None),
                                    "best_step": max((e for e in lg if e.get("best")), key=lambda e: e["step"],
                                                     default={}).get("step")}
OUT["maze_training_probe"] = probe
OUT["R2"]["M1_val9_best_per_seed"] = {k: v["best_val9"] for k, v in probe.items() if k.startswith("M1")}

# ------------------------------------------------------------------ R3 prefix bar
p1_512 = ep.get("P1/fL512@300") or {}
p1_128 = ep.get("P1/fL128@300") or {}
pl_512 = ep.get("PL/fL512@300") or {}
OUT["R3"] = {"P1_fL512_300": p1_512, "P1_fL128_300": p1_128, "PL_fL512_300": pl_512,
             "ok": (p1_512.get("success", 0) >= 4) and ((p1_128.get("median") or 0) >= 0.90)}


# ------------------------------------------------------------------ F1 credit horizon
def arm_seeds(arm, n, st, t):
    return [acc(mid(arm, s), st, t) for s in range(n)]


f1 = {}
P1v, P4v = arm_seeds("P1", 8, "fL512", 300), arm_seeds("P4", 8, "fL512", 300)
if all(v is not None for v in P1v + P4v):
    a, b = sum(v >= 0.9 for v in P4v), sum(v >= 0.9 for v in P1v)
    f1["prefix"] = {"P1": ci(P1v), "P4": ci(P4v), "P5": ep.get("P5/fL512@300"), "P4cm": ep.get("P4cm/fL512@300"),
                    "fisher_p_P4_fewer": round(fisher_less(a, 8, b, 8), 4),
                    "median_gap_pp": round(100 * (np.median(P1v) - np.median(P4v)), 2)}
    g = np.median(P1v) - np.median(P4v)
    p4cm = [x for x in arm_seeds("P4cm", 3, "fL512", 300) if x is not None]
    if p4cm:
        f1["prefix"]["P4cm_gap_closed_frac"] = round(float((np.median(p4cm) - np.median(P4v)) / g), 3) if abs(g) > 1e-9 else None
    f1["prefix"]["paired_boot_P1_minus_P4"] = [paired_boot(mid("P1", s), mid("P4", s), "fL512", 300) for s in range(8)]
mz = []
for s in range(3):
    a_, b_ = acc(mid("M4", s), "m33", 300), acc(mid("Mprog", s), "m33", 300)
    if a_ is not None and b_ is not None:
        mz.append({"seed": s, "M4": a_, "Mprog": b_, "diff_pp": round(100 * (a_ - b_), 2),
                   "boot_M4_minus_Mprog": paired_boot(mid("M4", s), mid("Mprog", s), "m33", 300)})
f1["maze"] = {"pairs": mz, "M4cm": acc(mid("M4cm", 0), "m33", 300), "M5": arm_seeds("M5", 1, "m33", 300)}
# the pre-registered 33x33 comparison is uninformative when every arm is near 0 there (w=64, 15k steps);
# the same arms on smaller mazes (training size 9 at 30 iterations, and 13x13 at 300) are reported as secondary
f1["maze"]["at_floor_m33"] = bool(mz) and max(max(x["M4"], x["Mprog"]) for x in mz) <= 0.05
sec = {}
for arm, n in (("M1", 3), ("Mprog", 3), ("M4", 3)):
    for st, t in (("m9", 30), ("m9", 300), ("m13", 300), ("m33", 300)):
        v = arm_seeds(arm, n, st, t)
        if any(x is not None for x in v):
            sec[f"{arm}/{st}@{t}"] = {"per_seed": [None if x is None else round(x, 4) for x in v],
                                      **ci([x for x in v if x is not None])}
f1["maze"]["secondary"] = sec
f1["maze"]["secondary_boot_M4_minus_Mprog"] = {st: [paired_boot(mid("M4", s), mid("Mprog", s), st, t) for s in range(3)
                                                    if acc(mid("M4", s), st, t) is not None
                                                    and acc(mid("Mprog", s), st, t) is not None]
                                               for st, t in (("m9", 30), ("m13", 300))}
if "prefix" in f1 and len(mz) == 3:
    pr = f1["prefix"]
    pred_prefix = pr["fisher_p_P4_fewer"] < 0.05 and pr["median_gap_pp"] >= 10 and (pr.get("P4cm_gap_closed_frac") or 0) < 0.5
    pred_maze = all(x["diff_pp"] <= -10 and x["boot_M4_minus_Mprog"] and x["boot_M4_minus_Mprog"]["hi"] < 0 for x in mz)
    p4cm_med = (ep.get("P4cm/fL512@300") or {}).get("median")
    p1_med, p4_med = np.median(P1v), np.median(P4v)
    succ = lambda v: sum(x >= 0.9 for x in v)
    refuted = (abs(p1_med - p4_med) <= 0.03 and abs(succ(P1v) - succ(P4v)) <= 1
               and (p4cm_med is None or abs(p1_med - p4cm_med) <= 0.03)
               and all(abs(x["diff_pp"]) <= 3 for x in mz))
    f1["prefix_prediction_holds"], f1["maze_prediction_holds"], f1["refuted"] = pred_prefix, pred_maze, refuted
    f1["verdict"] = ("confirmed" if pred_prefix and pred_maze else "refuted" if refuted else "inconclusive")
    _pre_ref = (abs(p1_med - p4_med) <= 0.03 and abs(succ(P1v) - succ(P4v)) <= 1
                and (p4cm_med is None or abs(p1_med - p4cm_med) <= 0.03))
    _mz_ref = all(abs(x["diff_pp"]) <= 3 for x in mz)
    f1["verdict_by_task"] = ("prefix: " + ("confirmed" if pred_prefix else "refuted" if _pre_ref else "inconclusive")
                             + "; maze: " + ("confirmed" if pred_maze else "refuted" if _mz_ref else "inconclusive")
                             + " (M4 minus M-prog on 33x33@300 per seed, pp: "
                             + ", ".join(f"{x['diff_pp']:+.1f}" for x in mz) + ")")
    if f1["maze"]["at_floor_m33"]:
        # every maze arm is ~0 on 33x33, so "M4 within 3 pp of M-prog" there holds only vacuously
        pre_ref = (abs(p1_med - p4_med) <= 0.03 and abs(succ(P1v) - succ(P4v)) <= 1
                   and (p4cm_med is None or abs(p1_med - p4cm_med) <= 0.03))
        f1["verdict"] = ("prefix: " + ("confirmed" if pred_prefix else "refuted" if pre_ref else "inconclusive")
                         + "; maze: at floor on 33x33 (uninformative), see secondary sets")
if "prefix" in f1 and "verdict" not in f1:
    pr = f1["prefix"]
    p4cm_med = (ep.get("P4cm/fL512@300") or {}).get("median")
    p1_med, p4_med = np.median(P1v), np.median(P4v)
    succ = lambda v: sum(x >= 0.9 for x in v)
    pre_ref = (abs(p1_med - p4_med) <= 0.03 and abs(succ(P1v) - succ(P4v)) <= 1
               and (p4cm_med is None or abs(p1_med - p4cm_med) <= 0.03))
    f1["verdict"] = "prefix: " + ("refuted" if pre_ref else "inconclusive") + "; maze: incomplete"
OUT["F1"] = f1

# ------------------------------------------------------------------ F2 market-resnet control
f2 = {}
for st in ("fL128", "fL256", "fL512"):
    for t in (24, 100, 300):
        v = arm_seeds("MRP", 3, st, t)
        if any(x is not None for x in v):
            f2[f"MRP/{st}@{t}"] = ci(v)
    for e in sorted(k for k in EV if k.startswith("PMR_")):
        for t in (24, 100, 300):
            a_ = acc(e, st, t)
            if a_ is not None:
                f2[f"{e[4:]}/{st}@{t}"] = round(a_, 4)
    for arm in ("P1", "P6"):
        v = arm_seeds(arm, PRE_ARMS[arm], st, 300)
        if any(x is not None for x in v):
            f2[f"{arm}/{st}@300"] = ci(v)
# fallback while the market-resnet final grid has no GRID_DONE: MR-P (ours, 3 seeds) against the pilot's single-seed
# arms on the pilot's own legacy sets (weaker: one pilot seed per arm, no final-grid checkpoints)
leg = {}
for st in ("legacy_L64", "legacy_L128", "legacy_mix_8_64"):
    for t in (24, 100, 256):
        v = arm_seeds("MRP", 3, st, t)
        if any(x is not None for x in v):
            leg[f"MRP/{st}@{t}"] = ci([x for x in v if x is not None])
        for a in ("A", "B", "Bds", "C", "D"):
            a_ = acc(f"pilotMR_{a}", st, t)
            if a_ is not None:
                leg[f"pilot_{a}/{st}@{t}"] = round(a_, 4)
f2["legacy_fallback"] = leg
# the spec's F2 sets (fL256, fL512 at T=24) for the pilot's single-seed arms, evaluated after the GPU run on
# General's CPU (strict fp32, same harness): results/vm/general/eval/pilotMRfL_<arm>
pf = {}
for st in ("fL256", "fL512"):
    for t in (24, 50, 100):
        v = arm_seeds("MRP", 3, st, t)
        if any(x is not None for x in v):
            pf[f"MRP/{st}@{t}"] = ci([x for x in v if x is not None])
        for a in ("A", "B", "Bds"):
            a_ = acc(f"pilotMRfL_{a}", st, t)
            if a_ is None:                      # the CPU evals ran the set as "fL256@100" (no alias)
                a_ = acc(f"pilotMRfL_{a}", f"{st}@100", t)
            if a_ is not None:
                pf[f"pilot_{a}/{st}@{t}"] = round(a_, 4)
if pf:
    f2["pilot_on_spec_sets"] = pf
    b24 = [pf.get(f"pilot_B/{st}@24") for st in ("fL256", "fL512")]
    p24 = [(pf.get(f"MRP/{st}@24") or {}).get("mean") for st in ("fL256", "fL512")]
    if None not in b24 and None not in p24:
        # falsifier 5: MR-P matches or beats MR-B (within 1 pp or better on fL256 and fL512)
        f2["falsifier5_MRP_matches_or_beats_B"] = all(p >= b - 0.01 for p, b in zip(p24, b24))
        f2["note"] = "MR-B here is the single pilot seed, not the final grid (no GRID_DONE)"
OUT["F2"] = f2


# ------------------------------------------------------------------ F3 halting Pareto
_FZ = {}
VSET = {"m33": "v33", "m33h": "v33", "fL512": "vL512"}          # halting constants are tuned on these validation sets only
CGRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def depth_of(st):
    """required depth per instance: maze dead-end-filling iterations, prefix ceil(L/5) (DT backbone)"""
    base = {"m33h": "m33"}.get(st, st).split("[")[0]      # m33h = the first 100 of m33, run to 3000
    for f, key in (("mazes_e2h.npz", f"{base}/depth_iters"), ("prefix.npz", f"{base}/depth_dt")):
        if f not in _FZ:
            _FZ[f] = np.load(os.path.join(HERE, "frozen", f))
        if key in _FZ[f].files:
            return _FZ[f][key].astype(float)
    return None


def halting(eid, st):
    p = os.path.join(SETDIR.get((eid, st), ""), st + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    C = load_correct(p)
    B, Tc = C.shape
    grid = z["grid"].tolist()
    rows = np.arange(B)
    pts = {"fixed_T": [(t, float(C[:, t - 1].mean())) for t in grid]}
    kl = z["kl_first"]
    from p0eval import EPS
    pts["kl"] = []
    for j, e in enumerate(EPS):
        stop = np.where(kl[:, j] > 0, kl[:, j], Tc)
        pts["kl"].append((float(stop.mean()), float(C[rows, stop - 1].mean()), e))
    stop = np.where(z["mc_stop"] > 0, z["mc_stop"], Tc)
    pick = np.where(z["mc_stop"] > 0, z["mc_pick"], z["best_idx"][:, -1])
    pts["maxconf"] = [(float(stop.mean()), float(C[rows, np.maximum(pick, 1) - 1].mean()))]
    if "probe" in z.files and float(np.abs(z["probe"].astype(np.float32)).sum()) > 0:
        pr = z["probe"].astype(np.float32)
        pts["probe"] = []
        for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
            hit = pr > thr
            gi = np.where(hit.any(1), hit.argmax(1), len(grid) - 1)
            ts = np.array(grid)[gi]
            pts["probe"].append((float(ts.mean()), float(C[rows, ts - 1].mean()), thr))
    # size-aware T = ceil(c * required depth); the Pareto over c, plus the c chosen on the validation set
    dep, vdep = depth_of(st), depth_of(VSET.get(st, ""))
    if dep is not None:
        pts["size_aware"] = []
        for c in CGRID:
            ts = np.clip(np.ceil(c * dep[:B]), 1, Tc).astype(int)
            pts["size_aware"].append((float(ts.mean()), float(C[rows, ts - 1].mean()), c))
        cv = corr(eid, VSET[st]) if vdep is not None else None
        if cv is not None:
            va = [float(cv[np.arange(len(cv)), np.clip(np.ceil(c * vdep[:len(cv)]), 1, cv.shape[1]).astype(int) - 1].mean())
                  for c in CGRID]
            c_star = next(c for c, a_ in zip(CGRID, va) if a_ >= max(va) - 0.01)
            ts = np.clip(np.ceil(c_star * dep[:B]), 1, Tc).astype(int)
            pts["size_aware_valtuned"] = [(float(ts.mean()), float(C[rows, ts - 1].mean()), c_star)]
    fs = z["first_stable"]
    o_it = float(np.where(fs > 0, fs, 1).mean())
    o_acc = float((fs > 0).mean())
    cand = [(it, a_) for r in ("kl", "maxconf", "probe", "size_aware") for it, a_, *_ in pts.get(r, [])]
    cand += [(t, a_) for t, a_ in pts["fixed_T"]]
    at_matched = max([a_ for it, a_ in cand if it <= o_it] + [0.0])
    reach = [it for it, a_ in cand if a_ >= o_acc - 0.01]
    ratio = (min(reach) / o_it) if reach else None
    # accuracy is a cliff at the required depth, so the gap at exactly matched iterations can be large while a rule
    # that spends a few percent more iterations is within 1 pp; report the gap with 5% slack as well
    at_105 = max([a_ for it, a_ in cand if it <= 1.05 * o_it] + [0.0])
    return {"oracle": {"mean_iters": round(o_it, 2), "acc": round(o_acc, 4)},
            "best_rule_acc_at_oracle_iters": round(at_matched, 4), "gap_pp": round(100 * (o_acc - at_matched), 2),
            "gap_pp_at_1.05x_oracle_iters": round(100 * (o_acc - at_105), 2),
            "iters_ratio_to_within_1pp": round(ratio, 3) if ratio else None,
            "kl_never_frac": {str(e): round(float((kl[:, j] == 0).mean()), 4) for j, e in enumerate(EPS)},
            "points": {k: [[round(x, 4) if isinstance(x, float) else x for x in p_] for p_ in v] for k, v in pts.items()}}


f3 = {}
for eid, st in ([(mid("M1", s), "m33") for s in range(3)] + [("M1long_s0_best", "m33"), ("Mref", "m33"), ("Mref", "m33h")] +
                [(mid("P1", s), "fL512") for s in range(8)]):
    h = halting(eid, st)
    if h:
        f3[f"{eid}/{st}"] = h
for kind, st in (("maze", "m33"), ("prefix", "fL512")):
    hs = [v for k, v in f3.items() if k.endswith("/" + st) and not k.startswith("Mref")]
    subject = "M1 seeds" if kind == "maze" else "P1 seeds"
    if kind == "maze" and not [h for h in hs if h["oracle"]["acc"] >= 0.5]:
        # M1 (w=64) never solves 33x33, so its oracle is ~0 and there is nothing to halt; use M-ref (w=128)
        hs = [f3[k] for k in ("Mref/m33",) if k in f3]
        subject = "M-ref (M1 at floor)"
    if hs:
        f3[f"summary_{kind}"] = {"subject": subject, "median_gap_pp": float(np.median([h["gap_pp"] for h in hs])),
                                 "median_ratio": float(np.median([h["iters_ratio_to_within_1pp"] or 99 for h in hs])),
                                 "headroom": float(np.median([h["gap_pp"] for h in hs])) >= 2 or
                                 float(np.median([h["iters_ratio_to_within_1pp"] or 99 for h in hs])) >= 1.3,
                                 "median_gap_pp_at_1.05x": float(np.median([h["gap_pp_at_1.05x_oracle_iters"] for h in hs])),
                                 "no_room": float(np.median([h["gap_pp"] for h in hs])) <= 1 and
                                 float(np.median([h["iters_ratio_to_within_1pp"] or 99 for h in hs])) <= 1.2}
OUT["F3"] = f3

# ------------------------------------------------------------------ shift grid (descriptive)
sh = {}
for eid in ["Mref", "Mref_r1"] + [mid("M1", s) for s in range(3)] + ["M1long_s0_best"]:
    if eid not in EV:
        continue
    for st, s in EV[eid][1]["sets"].items():
        if st.startswith("sh_"):
            sh.setdefault(st, {})[eid] = {"exact@300": acc(eid, st, 300), "check@300": s.get("check_acc", [None])[-1],
                                          "knutson@300": s.get("knutson_acc", [None])[-1], "peak_exact": max(s["acc"])}
OUT["shift"] = sh

# ------------------------------------------------------------------ stretch, stability, end states, cost
OUT["stretch_m201"] = {e: EV[e][1]["sets"]["m201"]["acc"][-1] for e in EV if "m201" in EV[e][1]["sets"]}
OUT["stability_30k"] = {f"{e}/{st}": {"acc_30000": s["acc"][-1], "end_state": s["end_state"]}
                        for e, (_, sm) in EV.items() for st, s in sm["sets"].items() if "stab" in st}
OUT["end_states"] = {f"{e}/{st}": s["end_state"] for e, (_, sm) in EV.items() for st, s in sm["sets"].items()
                     if st in ("m33", "m59", "fL512", "m33h")}
cost = {}
for mk in glob.glob(os.path.join(RES, "vm", "*", "markers", "*.done")):
    jid = os.path.basename(mk)[:-5]
    info = json.load(open(mk))
    c = {"sec": info.get("sec")}
    lg = os.path.join(os.path.dirname(os.path.dirname(mk)), "logs", jid + ".log")
    if os.path.exists(lg):
        for line in open(lg):
            if line.startswith('{"done"') or line.startswith('{"timing"'):
                try:
                    d = json.loads(line)
                    c.update({k: d[k] for k in ("peak_mem_mb", "sec_per_1k") if k in d})
                except ValueError:
                    pass
    cost[jid] = c
OUT["cost"] = cost
OUT["flops_per_pos"] = {e: sm.get("flops") for e, (_, sm) in EV.items() if e in ("Mref", "M1_s0_best", "P1_s0", "PL_s0", "MRP_s0")}

json.dump(OUT, open(os.path.join(RES, "p0_results.json"), "w"), indent=1, default=float)

# ------------------------------------------------------------------ chart: per-seed endpoint accuracy by arm
series = []
for arm, ns in list(MAZE_ARMS.items()):
    v = ep.get(f"{arm}/best/m33@300")
    if v:
        series.append({"arm": arm, "task": "maze 33x33 @300", **v})
for arm in PRE_ARMS:
    v = ep.get(f"{arm}/fL512@300")
    if v:
        series.append({"arm": arm, "task": "prefix fL512 @300", **v})
chart = {"type": "dot-interval", "title": "P0: exact accuracy per seed at the primary endpoints",
         "y": "exact accuracy", "note": "dots = seeds, bar = mean with 95% t-interval; 0.90 = success line",
         "series": series,
         "curves": {e: {"set": st, "grid": EV[e][1]["sets"][st]["grid"], "acc": EV[e][1]["sets"][st]["acc"]}
                    for e, st in [("Mref", "m33"), ("M1_s0_best", "m33"), ("Mprog_s0_best", "m33"), ("M4_s0_best", "m33"),
                                  ("P1_s0", "fL512"), ("P4_s0", "fL512"), ("MRP_s0", "fL512")]
                    if e in EV and st in EV[e][1]["sets"]}}
json.dump(chart, open(os.path.join(RES, "chart_p0.json"), "w"), indent=1, default=float)
print(json.dumps({"evals": len(EV), "R1_ok": OUT["R1"]["ok"], "R2": {k: v for k, v in OUT["R2"].items()},
                  "R3_ok": OUT["R3"]["ok"], "F1": OUT["F1"].get("verdict"), "F1_by_task": OUT["F1"].get("verdict_by_task")}, default=float, indent=1))
