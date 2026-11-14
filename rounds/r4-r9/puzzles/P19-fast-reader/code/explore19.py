"""P19 exploratory item table (not pre-registered). python explore19.py RES_DIR -> ../results/p19_items.json:
per item, the mean over seeds 0-4 of each rule's value and its rank among the 44 items (1 = highest)."""
import json
import os
import sys

import numpy as np

os.environ.setdefault("P19_KREP_W", "4")
os.environ.setdefault("P19_KREP_P", "2")
sys.argv = [sys.argv[0], sys.argv[1]]
import analyze19 as A  # noqa: E402

rows = {}
for s in range(5):
    d, _ = A.load_seed(s)
    its = d["items"]
    vals = dict(truth=d["truth"], **{k: d["rules"][k] for k in A.PRIMARY})
    for k, v in vals.items():
        rk = (-np.asarray(v)).argsort().argsort() + 1
        for j, i in enumerate(its):
            r = rows.setdefault(i, {}).setdefault(k, dict(v=[], rank=[]))
            r["v"].append(float(v[j]))
            r["rank"].append(int(rk[j]))
out = {i: {k: dict(mean=round(float(np.mean(r["v"])), 2), mean_rank=round(float(np.mean(r["rank"])), 1)) for k, r in d.items()}
       for i, d in rows.items()}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "p19_items.json"), "w"), indent=1)
worst = sorted(out, key=lambda i: -abs(out[i]["FR"]["mean_rank"] - out[i]["truth"]["mean_rank"]))[:10]
for i in worst:
    print(i, {k: out[i][k]["mean_rank"] for k in ("truth", "FR", "FO", "R1", "R7b")})
