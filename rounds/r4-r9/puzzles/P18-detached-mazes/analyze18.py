"""P18 pre-registered analysis (PREREG section 7). Reads the synced evals under results/vm/*/out/ and gpu/main_plan.json;
writes results/analysis.json and charts/*.json. Run on General with the trader-cat venv:
  python analyze18.py            (optionally results/references.json from refs18.py is merged into the charts)

Primary: d59 = mean(A1) - mean(A4*) of seed-level e59 exact@12n; Welch one-sided 95% bounds LB59 / UB59.
  survives   iff LB59 > 0 and d33 >= -0.05
  killed     otherwise: "clearly worse" if UB59 < 0, else "no gain shown"
  inconclusive (runs missing) if A1 or A4* has fewer than 3 evaluated seeds
Secondary: Welch two-sided 95% intervals on e59 and e33 for A1-A2, A1-A3, A3-A4*, A2-A4*, A1 vs every alpha.
"""
import json, math, os, re, sys
import numpy as np
from scipy import stats

D = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(D, "results", "vm")
SETS = ["v9", "m13", "m59", "e19", "e33", "e59"]
ARMS = ["a1", "a2", "a3", "p0", "p05", "p1", "a5"]
LABEL = {"a1": "A1 detached, random order (K=4)", "a2": "A2 detached, round robin (K=4)",
         "a3": "A3 full BPTT, random order (K=4)", "p0": "A4 DT-Recall progressive alpha 0",
         "p05": "A4 DT-Recall progressive alpha 0.5", "p1": "A4 DT-Recall progressive alpha 1",
         "a5": "A5 detached, K=1 (exploratory)"}
TRAINED = 0.9          # v9 exact@12n at or above: the seed "trained"


def find_evals():
    """{run name: (summary json, npz path, slot)} for every eval with a .done marker"""
    out = {}
    if not os.path.isdir(RES):
        return out
    for slot in sorted(os.listdir(RES)):
        o = os.path.join(RES, slot, "out")
        if not os.path.isdir(o):
            continue
        for f in os.listdir(o):
            m = re.fullmatch(r"ev_(.+)\.json", f)
            if m and os.path.exists(os.path.join(o, f"ev_{m.group(1)}.done")):
                out[m.group(1)] = (json.load(open(os.path.join(o, f))), os.path.join(o, f"ev_{m.group(1)}.npz"), slot)
    return out


def train_info(name, slot):
    f = os.path.join(RES, slot, "out", f"{name}.json")
    if not os.path.exists(f):
        return {}
    j = json.load(open(f))
    last = j["log"][-1] if j.get("log") else {}
    return {"steps": j.get("steps"), "train_sec": j.get("train_sec"), "diverged": j.get("diverged"),
            "final_log": last, "cnt": j.get("cnt")}


def welch(a, b, conf=0.95, one_sided=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    d = a.mean() - b.mean()
    if n1 < 2 or n2 < 2:
        return {"d": d, "n": [n1, n2], "lo": None, "hi": None, "df": None}
    va, vb = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    se = math.sqrt(va + vb)
    if se == 0:
        return {"d": d, "n": [n1, n2], "lo": d, "hi": d, "df": None, "se": 0.0}
    df = (va + vb) ** 2 / (va ** 2 / (n1 - 1) + vb ** 2 / (n2 - 1))
    q = stats.t.ppf(conf if one_sided else (1 + conf) / 2, df)
    return {"d": d, "n": [n1, n2], "lo": d - q * se, "hi": d + q * se, "df": df, "se": se}


def main():
    ev = find_evals()
    plan_f = os.path.join(D, "gpu", "main_plan.json")
    plan = json.load(open(plan_f)) if os.path.exists(plan_f) else {}
    best = plan.get("best_alpha")
    runs = {}
    for name, (s, npz, slot) in ev.items():
        m = re.fullmatch(r"(a1|a2|a3|p0|p05|p1|a5)_s(\d+)", name)
        if not m:
            continue
        runs[name] = {"arm": m.group(1), "seed": int(m.group(2)), "slot": slot,
                      "sets": {k: s["sets"][k] for k in SETS if k in s["sets"]}, **train_info(name, slot)}
    by = {a: sorted([r for r in runs.values() if r["arm"] == a], key=lambda r: r["seed"]) for a in ARMS}

    def col(arm, st, key="exact@12n"):
        return [r["sets"][st][key] for r in by[arm] if st in r["sets"] and key in r["sets"][st]]

    table = {}
    for a in ARMS:
        if not by[a]:
            continue
        table[a] = {"label": LABEL[a], "seeds": [r["seed"] for r in by[a]],
                    "trained": [r["sets"]["v9"]["exact@12n"] >= TRAINED for r in by[a] if "v9" in r["sets"]],
                    "diverged": [bool(r.get("diverged")) for r in by[a]]}
        for st in SETS:
            for key in ["exact@12n", "exact@30", "best_over_t_exact"]:
                v = col(a, st, key)
                if v:
                    table[a][f"{st} {key}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                               "values": v}
    res = {"best_alpha": best, "alpha_scores_m59_m13": plan.get("alpha_scores_m59_m13"), "B_hours": plan.get("B_hours"),
           "steps": plan.get("steps"), "table": table, "runs": runs}

    # primary (PREREG 7)
    if best is None:
        res["verdict"] = "inconclusive (A4* not selected: alpha sweep incomplete)"
    else:
        a1, a4 = col("a1", "e59"), col(best, "e59")
        a1_33, a4_33 = col("a1", "e33"), col(best, "e33")
        if len(a1) < 3 or len(a4) < 3:
            res["verdict"] = f"inconclusive (runs missing: A1 {len(a1)}, A4* {len(a4)} evaluated seeds)"
        else:
            lb = welch(a1, a4, 0.95, one_sided=True)
            ub = welch(a4, a1, 0.95, one_sided=True)          # lower bound of A4* - A1 = -(upper bound of A1 - A4*)
            d33 = float(np.mean(a1_33) - np.mean(a4_33))
            LB59, UB59 = lb["lo"], -ub["lo"]
            res["primary"] = {"d59": lb["d"], "LB59": LB59, "UB59": UB59, "df": lb["df"], "d33": d33,
                              "n_a1": len(a1), "n_a4star": len(a4), "a4star": best}
            if LB59 > 0 and d33 >= -0.05:
                v = "A1 SURVIVES"
                tr1 = np.mean(table["a1"]["trained"]); tr4 = np.mean(table[best]["trained"])
                if tr4 < tr1:
                    v += f" (at B = {plan.get('B_hours')} L4-hours only: A4* trained rate {tr4:.2f} < A1 {tr1:.2f})"
            else:
                v = "A1 KILLED: " + ("clearly worse" if UB59 < 0 else "no gain shown")
            res["verdict"] = v
    # secondary contrasts
    sec = {}
    pairs = [("a1", "a2"), ("a1", "a3"), ("a3", best), ("a2", best)] + [("a1", p) for p in ["p0", "p05", "p1"]]
    for x, y in pairs:
        if y is None:
            continue
        for st in ["e59", "e33", "e19", "m59"]:
            u, w = col(x, st), col(y, st)
            if u and w:
                sec[f"{x}-{y} {st}"] = welch(u, w, 0.95)
    res["secondary"] = sec
    os.makedirs(os.path.join(D, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(D, "results", "analysis.json"), "w"), indent=1, default=float)

    # charts
    ch = os.path.join(D, "charts"); os.makedirs(ch, exist_ok=True)
    refs_f = os.path.join(D, "results", "references.json")
    refs = json.load(open(refs_f)) if os.path.exists(refs_f) else {}
    sizes = {"v9": 9, "m13": 13, "e19": 19, "e33": 33, "m59": 59, "e59": 59}
    bars = {"title": "P18: exact maze accuracy at T = 12n by maze size (seed means, points = seeds)",
            "x": ["e19", "e33", "e59"], "x_size": [19, 33, 59], "series": [], "references": refs.get("exact12n", {})}
    for a in ARMS:
        if a in table:
            bars["series"].append({"arm": a, "label": LABEL[a] + (" = A4*" if a == best else ""),
                                   "seeds": table[a]["seeds"],
                                   "values": [col(a, st) for st in bars["x"]],
                                   "mean": [float(np.mean(col(a, st))) if col(a, st) else None for st in bars["x"]]})
    json.dump(bars, open(os.path.join(ch, "exact12n_by_size.json"), "w"), indent=1)
    curves = {"title": "P18: exact accuracy on e59 (59x59, 200 mazes) against test steps t, one line per run",
              "set": "e59", "series": []}
    for name, r in sorted(runs.items()):
        npz = ev[name][1]
        if os.path.exists(npz):
            z = np.load(npz)
            if "e59/t" in z.files:
                curves["series"].append({"run": name, "arm": r["arm"], "t": z["e59/t"].tolist(),
                                         "exact": [round(float(x), 4) for x in z["e59/exact"].mean(1)]})
    json.dump(curves, open(os.path.join(ch, "e59_exact_vs_t.json"), "w"), indent=1)
    print(json.dumps({"verdict": res["verdict"], "primary": res.get("primary"),
                      "arms": {a: {k: (v["mean"] if isinstance(v, dict) else v) for k, v in t.items()
                                   if k in ("seeds", "e59 exact@12n", "e33 exact@12n", "e19 exact@12n", "trained")}
                               for a, t in table.items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
