"""M7 w3: aggregate the stage-1 runs (results/gpu/runs/*.json) into results/summary.csv and results/decision.json.
Rules are the PREREG ones. Re-run: python3 analyze.py"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "gpu", "runs")
OUT = os.path.join(HERE, "results")
ARMS = ["none", "deepseek", "exp3", "fullinfo"]
KEYS = ["bce_all", "bce_2nd_half", "bce_post_shift_2k", "bce2h.periodic_ho", "max_load", "max_load_worst_layer",
        "dead_frac", "cv_load", "mi_expert_family_bits", "mi_norm", "L0.wealth_gini", "L1.wealth_gini", "wall_s"]


def load():
    R = {}
    for p in glob.glob(os.path.join(RUNS, "w3*-E*-s*.json")):
        r = json.load(open(p))
        arm = r["arm"][2:]
        R[(arm, r["E"], r["seed"])] = r
    return R


def ms(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v)


def main():
    R = load()
    rows = []
    for E in (8, 64):
        for a in ARMS:
            for k in KEYS:
                vals = [R[(a, E, s)].get(k) for s in (1, 2, 3) if (a, E, s) in R]
                m, sd, n = ms(vals)
                rows.append(dict(E=E, arm=a, metric=k, mean=m, sd=sd, n=n,
                                 per_seed=";".join(f"{x:.5g}" for x in vals if x is not None)))
    with open(os.path.join(OUT, "summary.csv"), "w") as fh:
        fh.write("E,arm,metric,mean,sd,n,per_seed\n")
        for r in rows:
            fh.write(f"{r['E']},{r['arm']},{r['metric']},{r['mean']:.6g},{r['sd']:.3g},{r['n']},{r['per_seed']}\n")
    dec = {"source": "M7/w3/results/gpu/runs/*.json", "rules": "PREREG.md"}
    for arm in ("exp3", "fullinfo"):
        per_E = {}
        for E in (8, 64):
            seeds = [s for s in (1, 2, 3) if (arm, E, s) in R and ("deepseek", E, s) in R]
            if not seeds:
                continue
            d = [R[(arm, E, s)]["bce_2nd_half"] - R[("deepseek", E, s)]["bce_2nd_half"] for s in seeds]
            sds = [ms([R[(x, E, s)]["bce_2nd_half"] for s in seeds])[1] for x in (arm, "deepseek")]
            ml = [R[(arm, E, s)]["max_load_worst_layer"] for s in seeds]
            rel = [R[(arm, E, s)]["bce_2nd_half"] / R[("deepseek", E, s)]["bce_2nd_half"] - 1 for s in seeds]
            beats = len(seeds) == 3 and all(x < 0 for x in d) and -np.mean(d) > max(sds)
            per_E[E] = dict(n=len(seeds), diff_vs_deepseek=d, mean_diff=float(np.mean(d)), max_sd=float(max(sds)),
                            rel_gap=rel, max_load_worst=ml, maxload_gt50_seeds=int(sum(x > 0.5 for x in ml)),
                            beats=bool(beats and max(ml) <= 0.5), worse_2pct=bool(np.mean(rel) > 0.02))
        stop = any(v["maxload_gt50_seeds"] >= 2 for v in per_E.values()) or (
            len(per_E) == 2 and all(v["worse_2pct"] for v in per_E.values()))
        verdict = "beats" if len(per_E) == 2 and all(v["beats"] for v in per_E.values()) else (
            "partial" if any(v["beats"] for v in per_E.values()) else "no")
        dec[arm] = dict(per_E=per_E, stop_rule_hit=bool(stop), verdict_vs_deepseek=verdict)
    dec["wealth_beats_deepseek"] = dec.get("exp3", {}).get("verdict_vs_deepseek") == "beats"
    json.dump(dec, open(os.path.join(OUT, "decision.json"), "w"), indent=1)
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
