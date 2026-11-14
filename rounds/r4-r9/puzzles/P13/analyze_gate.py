"""P13 analysis when the learnability gate decides the run (PREREG F0-gate: no main phase), or as a partial view before the
main phase. Reads the pilot evals and training logs (0.5 h and, if run, 1.0 h), the M-ref gate eval, and whatever DTR-eq
and M-ref evals have synced. Writes results/results_gate.json and results/chart_gate.json (analyze.py writes the
main-phase results.json and chart.json; RESULTS.md says which one is the record).
Usage: analyze_gate.py"""
import glob, json, os
import numpy as np
from scipy import stats
from pilot_select import cand, ARMS, FIXES, D
from analyze import Ev

TRAINED = (0.95, 0.5)          # PREREG: trained if 9x9 val >= 0.95 and v33 >= 0.5
GATE = (0.9, 0.5)              # PREREG: learnability gate 9x9 val >= 0.9 and 13 dev >= 0.5 at 1.0 L4-h


def ci(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    m = float(np.mean(v))
    if len(v) < 2:
        return {"mean": m, "n": len(v), "lo": None, "hi": None, "seeds": v}
    h = float(stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)))
    return {"mean": m, "n": len(v), "lo": m - h, "hi": m + h, "seeds": v}


def summary(ev, sets, which="12n"):
    r = {}
    for s in sets:
        if ev.has(s):
            S = ev.S[s]
            r[f"{s}_exact@{which}"] = S[f"exact@{which}"]
            r[f"{s}_ll_raw@{which}"] = S[f"ll_raw@{which}"]
            r[f"{s}_best_over_t"] = S["best_over_t_exact"]
            r[f"{s}_cycle@{which}"] = S[f"cycle@{which}"]
            r[f"{s}_maxb@{which}"] = S[f"maxb@{which}"]
            if "Tmax" in S and S["Tmax"] > S["T12"]:
                r[f"{s}_exact@Tmax"] = S["exact@Tmax"]
    return r


def main():
    out = {"pilots": {}, "gate": {}, "mref": {}, "dtr_eq": {"seeds": {}, "arm": {}}}
    for tag, suf in [("0p5", ""), ("1p0", "_x")]:
        C = [cand(a, f, suf) for a in ARMS for f in FIXES]
        out["pilots"][tag] = C
        have = [c for c in C if not c.get("missing")]
        out["gate"][tag] = {
            "n_evaluated": len(have),
            "pass": sorted(c["name"] for c in have if c["v9_exact"] >= GATE[0] and c["m13_exact"] >= GATE[1]),
            "best_v9": max(((c["v9_exact"], c["name"]) for c in have), default=None),
            "best_m13": max(((c["m13_exact"], c["name"]) for c in have), default=None),
            "max_frac_capped_last25_econ": max((c.get("frac_capped_last25", 0.0) for c in have if c["arm"] == "econ"), default=None)}
    g1 = out["gate"]["1p0"]
    # the pre-registered gate is on the SELECTED fix (pilot_select.py --tag 1p0); "pass" above is per candidate, descriptive
    ps = os.path.join(D, "results", "pilot_selection_1p0.json")
    if os.path.exists(ps):
        sel = json.load(open(ps))
        out["selection_1p0"] = {k: sel.get(k) for k in ["selected", "gate", "pool_econ", "pool_nocap", "mean_rank", "tie_rule_used", "econ_sel_frac_capped_last25", "econ_fix_active", "F2_recorded", "post_hoc_ctrl_selected"]}
        out["F0_gate"] = not sel["gate"]
    else:
        out["F0_gate"] = None
    gm = Ev("ev_mref_gate")
    if gm.ok:
        out["mref"]["gate"] = {"m13": gm.S["m13"]["exact@12n"], "m59": gm.S["m59"]["exact@12n"],
                               "pass": gm.S["m13"]["exact@12n"] >= 1.0 and gm.S["m59"]["exact@12n"] >= 0.99}
    mr = {}
    for n in ["ev59_mref_none", "evT_mref", "ev99_mref"]:
        mr.update(summary(Ev(n), ["v9", "v33", "m59", "e19", "e33", "e59", "e99"]))
    out["mref"]["evals"] = mr
    for s in range(4):
        n = f"m_dtr_s{s}"
        r = {}
        for e in [f"ev59_{n}_none", f"evT_{n}", f"ev99_{n}"]:
            r.update(summary(Ev(e), ["v9", "v33", "m59", "e19", "e33", "e59", "e99"]))
        if "v9_exact@12n" in r and "v33_exact@12n" in r:
            r["trained"] = r["v9_exact@12n"] >= TRAINED[0] and r["v33_exact@12n"] >= TRAINED[1]
        out["dtr_eq"]["seeds"][n] = r
    keys = sorted({k for r in out["dtr_eq"]["seeds"].values() for k in r if k.endswith("@12n") or k.endswith("@Tmax")})
    out["dtr_eq"]["arm"] = {k: ci([r.get(k) for r in out["dtr_eq"]["seeds"].values() if k in r]) for k in keys}
    out["dtr_eq"]["n_trained"] = sum(bool(r.get("trained")) for r in out["dtr_eq"]["seeds"].values())
    # exploratory (23:45Z addendum, not pre-registered): DT-recall-eq val9_acc30 from its training log at the step counts
    # that equal 0.5 and 1.0 L4-h under the Phase-1 budget (30456 steps = 3.0 h), latest logged value at or below each
    B = json.load(open(os.path.join(D, "gpu", "budgets.json")))
    dsteps = int(B.get("main_steps", {}).get("dtr", 30456))
    eqt = {}
    for s in range(4):
        g = sorted(glob.glob(os.path.join(D, "results", "vm", "*", "out", "logs", f"m_dtr_s{s}.log")))
        if not g:
            continue
        L = [json.loads(l) for l in open(g[-1]) if l.startswith("{") and "val9_acc30" in l]
        for h in [0.5, 1.0, 2.0, 3.0]:
            lim = dsteps * h / 3.0
            v = [l for l in L if l["step"] <= lim]
            if v and max(l["step"] for l in L) >= lim - 800:
                eqt.setdefault(f"{h}h", {})[f"m_dtr_s{s}"] = {"step": v[-1]["step"], "val9_acc30": v[-1]["val9_acc30"]}
    out["exploratory_equal_time_v9"] = {"dtr_eq_val9_acc30": eqt, "dtr_steps_3h": dsteps,
        "pilots_v9_exact_12n": {tag: {c["name"]: c.get("v9_exact") for c in out["pilots"][tag] if not c.get("missing")}
                                for tag in ["0p5", "1p0"]}}
    json.dump(out, open(os.path.join(D, "results", "results_gate.json"), "w"), indent=1)
    chart = {"title": "P13 pilots: exact accuracy at 12n after 0.5 and 1.0 L4-h (gate: v9 >= 0.9 and m13 >= 0.5)",
             "x_hours": [0.5, 1.0], "pilots": [], "dtr_eq_by_n": {}, "mref_by_n": {}}
    for a in ARMS:
        for f in FIXES:
            ys = {}
            for tag in ["0p5", "1p0"]:
                c = next(c for c in out["pilots"][tag] if c.get("name") == f"pil_{a}_{f}")
                for s in ["v9", "m13", "v33", "m59"]:
                    ys.setdefault(s, []).append(None if c.get("missing") else c[f"{s}_exact"])
            chart["pilots"].append({"arm": a, "fix": f, **ys})
    for s, n in [("v9", 9), ("e19", 19), ("e33", 33), ("e59", 59), ("e99", 99)]:
        d = out["dtr_eq"]["arm"].get(f"{s}_exact@12n")
        if d:
            chart["dtr_eq_by_n"][n] = {k: d[k] for k in ["mean", "lo", "hi", "n"]}
        if f"{s}_exact@12n" in mr:
            chart["mref_by_n"][n] = mr[f"{s}_exact@12n"]
    json.dump(chart, open(os.path.join(D, "results", "chart_gate.json"), "w"), indent=1)
    print(json.dumps({"gate": out["gate"], "F0_gate": out["F0_gate"], "mref_gate": out["mref"].get("gate"),
                      "dtr_n_trained": out["dtr_eq"]["n_trained"]}, indent=1))


if __name__ == "__main__":
    main()
