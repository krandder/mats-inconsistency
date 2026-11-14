"""EXPLORATORY (post hoc): combine report/expl_routeb_s101_102.json and report/expl_routeb_s103.json (written by
expl_routeb.py with SEEDS/TAG) into 3-seed means and seed-t 95% intervals: report/expl_routeb_summary.json|csv.
   python3 expl_routeb_summary.py"""
import csv
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
a = json.load(open(os.path.join(H, "report", "expl_routeb_s101_102.json")))
a.update(json.load(open(os.path.join(H, "report", "expl_routeb_s103.json"))))


def ci(v):
    v = np.array(v)
    m = v.mean()
    h = 4.302652729749464 * v.std(ddof=1) / math.sqrt(len(v))
    return m, m - h, m + h


out, rows = {}, []
for split in ("cal", "test"):
    for arch in ("primary", "fallback"):
        for q in ("kl_p0_t", "kl_p0_t0", "kl_routeb", "routeb_reduction_vs_p0"):
            v = [a[f"{split}/s{s}/{arch}"][q] for s in (101, 102, 103)]
            m, lo, hi = ci(v)
            out[f"{split}/{arch}/{q}"] = {"per_seed": v, "mean": m, "ci95": [lo, hi]}
            rows.append([split, arch, q] + [f"{x:.5f}" for x in v] + [f"{m:.5f}", f"{lo:.5f}", f"{hi:.5f}"])
json.dump(out, open(os.path.join(H, "report", "expl_routeb_summary.json"), "w"), indent=1)
with open(os.path.join(H, "report", "expl_routeb_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "arch", "quantity", "s101", "s102", "s103", "mean", "ci_lo", "ci_hi"])
    w.writerows(rows)
print(len(rows), "rows")
