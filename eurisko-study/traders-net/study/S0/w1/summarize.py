"""Aggregate w1/results/*.json into w1/SUMMARY-w1.csv (mean and sd over seeds, per arm and family)."""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for p in sorted(glob.glob(os.path.join(HERE, "results", "*-s[0-9].json"))):
    j = json.load(open(p)); R.setdefault(j["arm"], []).append(j)
fams = next(iter(R.values()))[0]["families"]
rows = ["arm,metric,family,mean,sd,n,seeds"]
for arm, js in sorted(R.items()):
    keys = [k for k in ("bce", "brier", "in_bce", "bce_log", "in_bce_log") if k in js[0]]
    if arm == "ME":
        for j in js:
            j["gap"] = (np.array(j["bce"]) - np.array(j["in_bce"])).tolist()
        keys.append("gap")
    for k in keys:
        A = np.array([j[k] for j in js], float)
        for fi, fn in enumerate(fams):
            sd = A[:, fi].std(ddof=1) if len(js) > 1 else float("nan")
            rows.append(f"{arm},{k},{fn},{A[:, fi].mean():.4f},{sd:.4f},{len(js)},{'/'.join(str(j['seed']) for j in js)}")
    if arm == "ME":
        for fn in fams:
            v = [j["overlap"].get(fn, np.nan) for j in js]
            rows.append(f"{arm},overlap49,{fn},{np.mean(v):.4f},{np.std(v, ddof=1):.4f},{len(js)},{'/'.join(str(j['seed']) for j in js)}")
    rows.append(f"{arm},wall_s,all,{np.mean([j['wall_s'] for j in js]):.0f},{np.std([j['wall_s'] for j in js], ddof=1) if len(js)>1 else 0:.0f},{len(js)},")
open(os.path.join(HERE, "SUMMARY-w1.csv"), "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
