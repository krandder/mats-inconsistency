"""Item 1 analysis, as pre-registered in PREREG.md (1a) and PREREG-1b.md (1b). Reads the synced raw outputs only.

  python analyze1.py            -> results1.json (+ printed summary)
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "vmout")
E1A, E1B = os.path.join(OUT, "eval1a"), os.path.join(OUT, "eval1b")
DATA = os.path.join(HERE, "code", "data")
DTB = os.path.join(HERE, "..", "..", "..", "dt-baseline")
PRIMARY = ["test_W33", "test_W49", "test_W65"]; LP99 = 21.0
SEEDS_1A = list(range(100, 132)); SEEDS_1B = list(range(200, 220))
P12_S0 = os.path.join(HERE, "..", "..", "P12", "results", "eval", "CostNet+BB_s0_w32.npz")
LS = {s: np.load(os.path.join(DATA, f"{s}.npz"))["L"] for s in PRIMARY}


def ci(xs):
    x = np.asarray([v for v in xs if v is not None], float)
    if len(x) == 0:
        return None
    m = float(x.mean()); sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    h = float(stats.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return {"n": int(len(x)), "mean": m, "sd": sd, "lo": m - h, "hi": m + h, "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max())}


def primary(npz):
    d = np.load(npz)
    return float(np.concatenate([d[f"{s}/exact"][LS[s] > LP99] for s in PRIMARY]).astype(float).mean())


def a1():
    r = {}
    g = os.path.join(E1A, "GATE_1a.json")
    r["gate"] = json.load(open(g)) if os.path.exists(g) else None
    r["control_uniform_primary"] = primary(os.path.join(E1A, "control_uniform.npz")) if os.path.exists(os.path.join(E1A, "control_uniform.npz")) else None
    r["control_oracle_primary"] = primary(os.path.join(E1A, "control_oracle.npz")) if os.path.exists(os.path.join(E1A, "control_oracle.npz")) else None
    r["p12_seed0_step9000_primary_recomputed"] = primary(P12_S0)
    fin, s9 = {}, {}
    for s in SEEDS_1A:
        f = os.path.join(E1A, f"CostNet+BB_s{s}.npz"); f9 = os.path.join(E1A, f"CostNet+BB_s{s}_step9000.npz")
        failed = bool(glob.glob(os.path.join(E1A, f"*FAILED_s{s}")))
        fin[s] = primary(f) if os.path.exists(f) else (0.0 if failed else None)   # prereg: a failed seed counts 0
        s9[s] = primary(f9) if os.path.exists(f9) else (0.0 if failed else None)
    r["per_seed_final"] = fin; r["per_seed_step9000"] = s9
    r["missing_final"] = [s for s, v in fin.items() if v is None]
    r["final"] = ci(list(fin.values())); r["step9000"] = ci(list(s9.values()))
    if r["final"] and r["control_uniform_primary"] is not None:
        lo, hi = r["final"]["lo"], r["final"]["hi"]
        b1 = lo > r["control_uniform_primary"] + 0.05
        b2 = lo > 0.0806
        r["bar1_beats_uniform_plus5"] = bool(b1); r["bar2_beats_invalidDT_plus5"] = bool(b2)
        r["verdict"] = "replicates" if (b1 and b2) else ("does not replicate" if hi < 0.0806 else "inconclusive")
        if r["missing_final"]:
            r["verdict"] += f" (INCOMPLETE: {len(r['missing_final'])} seeds missing)"
    v9 = np.array([v for v in s9.values() if v is not None], float)
    if len(v9):
        r["bar3_frac_step9000_ge_27"] = float((v9 >= 0.27 - 1e-12).mean())
        r["bar3_p12_s0_percentile_among_new_step9000"] = float((v9 < 0.27).mean() * 100)
    # secondaries: per-set means over seeds (final checkpoint)
    per = {}
    for s in SEEDS_1A:
        f = os.path.join(E1A, f"CostNet+BB_s{s}.json")
        if os.path.exists(f):
            for k, v in json.load(open(f))["means"].items():
                per.setdefault(k, []).append(v)
    r["secondary_means_final"] = {k: ci(v) for k, v in sorted(per.items()) if k.endswith("/exact") or k.endswith("/valid")}
    for kind in ("uniform", "oracle"):
        f = os.path.join(E1A, f"control_{kind}.json")
        if os.path.exists(f):
            r[f"control_{kind}_means"] = {k: v for k, v in json.load(open(f))["means"].items() if k.endswith("/exact")}
    return r


def dt_released():
    """released DT-Recall maze checkpoint results from dt-baseline (dt_eval.py JSONs), if present."""
    out = {}
    for s in ("e2h_maze_13", "e2h_maze_33", "e2h_maze_59", "e2h_maze_201"):
        fs = sorted(glob.glob(os.path.join(DTB, "results", "**", "released_maze", f"{s}.json"), recursive=True))
        if fs:
            d = json.load(open(fs[-1]))
            out[s] = {"file": os.path.relpath(fs[-1], DTB), "n": d.get("n"), "T": d.get("T"), "peak_acc": d.get("peak_acc"),
                      "peak_iter": d.get("peak_iter"), "final_acc": d["acc"][-1] if d.get("acc") else None}
    return out


def a1b():
    r = {}
    g = os.path.join(E1B, "GATE_1b.json")
    r["gate"] = json.load(open(g)) if os.path.exists(g) else None
    r["selftest_ok"] = os.path.exists(os.path.join(E1B, "SELFTEST_OK"))
    per = {}
    for s in SEEDS_1B:
        f = os.path.join(E1B, "test", f"bansal_s{s}.json")
        failed = bool(glob.glob(os.path.join(E1B, f"*FAILED_s{s}")))
        if os.path.exists(f):
            d = json.load(open(f))["sets"]
            per[s] = {k: {"exact": v["exact"], "valid": v["valid"], "ratio": v["wall_open_cost_ratio_median"]} for k, v in d.items()}
        elif failed:
            per[s] = "failed"
    r["per_seed"] = per
    sets = ("e2h_maze_13", "e2h_maze_33", "e2h_maze_59", "e2h_maze_201")
    r["costnet"] = {}
    for st in sets:
        xs = [0.0 if v == "failed" else v[st]["exact"] for v in per.values() if v == "failed" or st in v]
        r["costnet"][st] = ci(xs)
        r["costnet"][st + "/valid"] = ci([v[st]["valid"] for v in per.values() if v != "failed" and st in v])
    r["dt_released"] = dt = dt_released()
    r["dt_gate_ok"] = (dt.get("e2h_maze_13", {}).get("peak_acc") or 0) >= 0.95 if "e2h_maze_13" in dt else None
    c = r["costnet"].get("e2h_maze_59")
    if c and "e2h_maze_59" in dt and dt["e2h_maze_59"]["peak_acc"] is not None and r["dt_gate_ok"]:
        D = dt["e2h_maze_59"]["peak_acc"]
        r["primary_verdict"] = ("superior" if c["lo"] > D + 0.05 else "non-inferior" if c["lo"] > D - 0.05
                                else "worse" if c["hi"] < D - 0.05 else "inconclusive")
    else:
        r["primary_verdict"] = "DT side pending" if r["dt_gate_ok"] is None or "e2h_maze_59" not in dt else "DT gate failed"
    return r


def main():
    res = {"item1a": a1(), "item1b": a1b()}
    json.dump(res, open(os.path.join(HERE, "results1.json"), "w"), indent=1, default=str)
    a, b = res["item1a"], res["item1b"]
    print("1a gate", (a["gate"] or {}).get("ok"), "| final", a["final"], "| step9000", a["step9000"])
    print("1a uniform", a["control_uniform_primary"], "oracle", a["control_oracle_primary"], "| verdict", a.get("verdict"),
          "| P12 s0 recomputed", a["p12_seed0_step9000_primary_recomputed"])
    print("1b gate", (b["gate"] or {}).get("ok"), "| costnet", {k: (v or {}).get("mean") for k, v in b["costnet"].items()})
    print("1b DT released", b["dt_released"], "| verdict", b["primary_verdict"])


if __name__ == "__main__":
    main()
