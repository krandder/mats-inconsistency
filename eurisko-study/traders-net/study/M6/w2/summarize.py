"""Aggregate M6/w2 results -> summary.csv, per_seed.csv, verdict.json (re-run to re-check)."""
import json, glob, os, math, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); R = f"{D}/results"
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
LEARN = [FAM.index(f) for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]]
NOISE = [FAM.index("ca30"), FAM.index("prng")]
ARMS = ["a", "cg", "cq", "d0", "dfs"]
rows, per = {}, []
for fn in sorted(glob.glob(f"{R}/*-s*.json")):
    r = json.load(open(fn)); arm, s = r["arm"], r["seed"]
    rec = {"arm": arm, "seed": s, "file": os.path.relpath(fn, D)}
    for h in ["all", "late"]:
        b = np.array(r[f"bce_{h}"]); b0 = np.array(r[f"bce_l0_{h}"])
        rec[f"learn_{h}"] = float(np.nanmean(b[LEARN])); rec[f"noise_{h}"] = float(np.max(b[NOISE] - math.log(2)))
        rec[f"learn_l0_{h}"] = float(np.nanmean(b0[LEARN])); rec[f"noise_l0_{h}"] = float(np.max(b0[NOISE] - math.log(2)))
    rec["lfsr16_all"] = r["bce_all"][1]
    tr = r.get("k_eff_n_trace") or []
    rec["k_eff_final"] = tr[-1][1] if tr else float("nan")
    rec["wealth_top_share"] = r.get("wealth_top_share_mean", r.get("k_top_share_mean", float("nan")))
    per.append(rec); rows.setdefault(arm, {})[s] = rec
keys = ["learn_all", "learn_late", "noise_all", "noise_late", "learn_l0_all", "lfsr16_all", "k_eff_final", "wealth_top_share"]
with open(f"{D}/per_seed.csv", "w") as f:
    f.write(",".join(["arm", "seed"] + keys + ["file"]) + "\n")
    for r in per: f.write(",".join([r["arm"], str(r["seed"])] + [f"{r[k]:.5f}" for k in keys] + [r["file"]]) + "\n")
with open(f"{D}/summary.csv", "w") as f:
    f.write("arm,n," + ",".join(f"{k}_mean,{k}_sd" for k in keys) + "\n")
    for a in ARMS:
        if a not in rows: continue
        v = list(rows[a].values())
        f.write(f"{a},{len(v)}," + ",".join(f"{np.mean([x[k] for x in v]):.5f},{np.std([x[k] for x in v], ddof=1) if len(v) > 1 else float('nan'):.5f}" for k in keys) + "\n")
verdict = {}
for a in ARMS[1:]:
    if a not in rows or "a" not in rows: continue
    seeds = sorted(set(rows[a]) & set(rows["a"]))
    rel = [rows[a][s]["learn_all"] / rows["a"][s]["learn_all"] - 1 for s in seeds]
    rel0 = [rows[a][s]["learn_all"] / rows[a][s]["learn_l0_all"] - 1 for s in seeds]
    verdict[a] = {"seeds": seeds, "rel_vs_a": rel, "fails_stop_rule": len(seeds) == 3 and all(x >= 0.01 for x in rel),
                  "beats_a_by_1pct_3of3": len(seeds) == 3 and all(x <= -0.01 for x in rel),
                  "rel_vs_l0": rel0, "beats_l0_3of3": len(seeds) == 3 and all(x < 0 for x in rel0)}
verdict["base_arms_survive_w2"] = any(not v["fails_stop_rule"] for k, v in verdict.items() if isinstance(v, dict)) if verdict else None
json.dump(verdict, open(f"{D}/verdict.json", "w"), indent=1)
print(open(f"{D}/summary.csv").read()); print(json.dumps(verdict, indent=1))
# sensitivity: learnable BCE without lfsr16 (bimodal under prequential training, S0)
NL = [i for i in LEARN if FAM[i] != "lfsr16"]
sens = {}
for a in ARMS:
    v = [np.nanmean(np.array(json.load(open(f"{R}/{a}-s{s}.json"))["bce_all"])[NL]) for s in (1, 2, 3)]
    sens[a] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
for a in ARMS[1:]:
    sens[a]["rel_vs_a"] = [x / y - 1 for x, y in zip(sens[a]["per_seed"], sens["a"]["per_seed"])]
json.dump(sens, open(f"{D}/sensitivity_no_lfsr.json", "w"), indent=1)
print({a: (round(sens[a]["mean"], 4), round(sens[a]["sd"], 4), [round(x, 3) for x in sens[a].get("rel_vs_a", [])]) for a in ARMS})
