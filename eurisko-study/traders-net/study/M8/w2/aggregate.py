"""Aggregate results/run_*.json -> results/summary.csv, results/per_seed.csv, results/decision.json."""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ARMS = ["absag", "taylor", "movement", "magnitude", "cbp", "random", "static128", "dense256"]
MET = ["bce_all", "bce_postshift_switched", "bce_postshift_other", "bce_last10k", "births",
       "births_survived", "dead_frac_mean"]
runs = {}
for p in sorted(glob.glob(os.path.join(RES, "run_*.json"))):
    r = json.load(open(p)); runs[(r["arm"], r["seed"])] = r
seeds = sorted({s for _, s in runs})
EXTRA = ["lfsr16_learnt", "bce_all_nolfsr", "postshift_switched_nolfsr", "lfsr16", "prng", "ca30"]
for p in sorted(glob.glob(os.path.join(RES, "family_s*.csv"))):
    for row in csv.DictReader(open(p)):
        k = (row["arm"], int(row["seed"]))
        if k in runs:
            for m in EXTRA:
                runs[k][m] = float(row[m]) if row[m] != "" else None
MET = MET + [m for m in EXTRA if all(m in r for r in runs.values())]
with open(os.path.join(RES, "per_seed.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "seed"] + MET)
    for (a, s), r in sorted(runs.items()):
        w.writerow([a, s] + [r[m] for m in MET])
with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["arm", "n"] + [f"{m}_{x}" for m in MET for x in ("mean", "sd")])
    for a in ARMS:
        rs = [runs[(a, s)] for s in seeds if (a, s) in runs]
        if not rs:
            continue
        row = [a, len(rs)]
        for m in MET:
            v = np.array([r[m] if r[m] is not None else np.nan for r in rs], float)
            row += [round(float(np.nanmean(v)), 5), round(float(np.nanstd(v, ddof=1)), 5) if len(v) > 1 else ""]
        w.writerow(row)


def cmp(x, y, k2="bce_postshift_switched", k1="bce_all"):
    """x vs y per seed: deltas (x - y) on the two primaries."""
    d2 = [runs[(x, s)][k2] - runs[(y, s)][k2] for s in seeds]
    d1 = [runs[(x, s)][k1] - runs[(y, s)][k1] for s in seeds]
    return {"d_postshift_switched": [round(v, 5) for v in d2], "d_all": [round(v, 5) for v in d1],
            "gain_3of3": bool(all(v < 0 for v in d2) and all(v <= 0.005 for v in d1))}


dec = {"stop_rule": "absag gains over random iff postshift-switched BCE lower in 3/3 seeds and bce_all not higher by >0.005 in any seed (PREREG.md)",
       "seeds": seeds}
if all((a, s) in runs for a in ARMS for s in seeds) and len(seeds) == 3:
    dec["absag_vs_random"] = cmp("absag", "random")
    dec["stop"] = not dec["absag_vs_random"]["gain_3of3"]
    dec["absag_vs"] = {b: cmp("absag", b) for b in ARMS if b not in ("absag",)}
    dec["random_vs_static128"] = cmp("random", "static128")
    if "bce_all_nolfsr" in MET:
        dec["nolfsr_note"] = "same comparisons with lfsr16 targets removed (lfsr16 learning is a bimodal lottery)"
        dec["absag_vs_nolfsr"] = {b: cmp("absag", b, "postshift_switched_nolfsr", "bce_all_nolfsr") for b in ARMS if b != "absag"}
        dec["lfsr16_learnt_by_arm"] = {a: [int(runs[(a, s)]["lfsr16_learnt"]) for s in seeds] for a in ARMS}
json.dump(dec, open(os.path.join(RES, "decision.json"), "w"), indent=1)
print(open(os.path.join(RES, "summary.csv")).read()); print(json.dumps(dec, indent=1))
