"""Mean +- sd over seeds of w3/baseline-s*.json -> w3/baseline-summary.csv"""
import glob, json, os, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "baseline-s[0-9].json")))]
rows = []
for model in ("net", "logistic"):
    for k in runs[0][model]:
        v = np.array([r[model][k] for r in runs], float)
        rows.append([model, k, len(v), np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else np.nan])
with open(os.path.join(HERE, "baseline-summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "metric", "n_seeds", "mean", "sd"])
    for r in rows:
        w.writerow(r[:3] + [f"{r[3]:.4f}", f"{r[4]:.4f}"])
for r in rows:
    if r[1].endswith(":bce") or (r[1].startswith("viol") and r[1].endswith(":mean")):
        print(f"{r[0]:9s} {r[1]:28s} {r[3]:.3f} +- {r[4]:.3f}")
