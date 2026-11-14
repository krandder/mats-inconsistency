"""Per-head-index means of K (book window), Delta (LOO, future) and Dalone -> results/byhead.csv."""
import numpy as np, csv, os
H = os.path.dirname(os.path.abspath(__file__))
rows = []
for arm in ["U", "Cs", "Cm", "Cw"]:
    for s in [1, 2, 3]:
        d = np.load(os.path.join(H, "results", f"{arm}-s{s}.npz")); l = d["live"]
        for i in range(d["K"].shape[0]):
            rows.append(dict(arm=arm, seed=s, head=i, K=d["K"][i, l].mean(), Delta_x1e3=1e3 * d["Delta"][i, l].mean(),
                             Dalone_x1e3=1e3 * d["Dalone"][i, l].mean(), G=d["G"][i, l].mean(),
                             frac_useful=(d["Delta"][i, l] > 0).mean()))
with open(os.path.join(H, "results", "byhead.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
import collections
agg = collections.defaultdict(list)
for r in rows:
    agg[(r["arm"], "head0" if r["head"] == 0 else "heads1-15")].append(r)
for k, v in sorted(agg.items()):
    print(k, {m: round(float(np.mean([x[m] for x in v])), 3) for m in ["K", "Delta_x1e3", "Dalone_x1e3", "G", "frac_useful"]})
