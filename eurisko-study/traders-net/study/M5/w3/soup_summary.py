"""Soup reference: sh_pat{2,4,8} one-share cumulative profit slopes, per question per epoch.
Reads logical-soup/runs/phase3/<cell>-s<seed>/metrics.csv (read-only). Q taken as 2^F (upper
bound on live questions, so per-question slopes are lower bounds in magnitude)."""
import csv, json, os, re
import numpy as np
ROOT = "/home/agent/workspaces/eurisko-study/logical-soup/runs/phase3"
HERE = os.path.dirname(os.path.abspath(__file__))

def ls(c):
    t = np.arange(len(c), dtype=float); t -= t.mean()
    return float((t * (c - c.mean())).sum() / (t * t).sum())

rows = []
for cell in ["k12-f8", "k12-f10", "k12-f12", "k14-f8", "k14-f10", "k14-f12", "k16-f12"]:
    K, F = map(int, re.findall(r"\d+", cell))
    Q = 2 ** F
    for s in (1, 2, 3):
        with open(f"{ROOT}/{cell}-s{s}/metrics.csv") as fh:
            r = list(csv.DictReader(fh))
        for k in (2, 4, 8):
            c = np.array([float(x[f"sh_pat{k}"]) for x in r]) / 65536.0 / Q
            n = len(c); h = n // 2
            rows.append({"cell": cell, "K_over_Q": 2 ** K / Q, "seed": s, "k": k, "epochs": n,
                         "slope_h1": ls(c[:h]), "slope_h2": ls(c[h:]),
                         "slope_last1000": ls(c[-1000:]), "phi_last20": ls(c[int(n * 0.8):]),
                         "E_end": float(c[-1])})
os.makedirs(f"{HERE}/results", exist_ok=True)
with open(f"{HERE}/results/soup_pat.csv", "w") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
agg = {}
for cell in sorted({r["cell"] for r in rows}):
    for k in (2, 4, 8):
        sel = [r for r in rows if r["cell"] == cell and r["k"] == k]
        for m in ("slope_h1", "slope_h2", "phi_last20"):
            v = np.array([r[m] for r in sel])
            agg[f"{cell}:pat{k}:{m}"] = [float(v.mean()), float(v.std(ddof=1)), int((v > 0).sum())]
json.dump(agg, open(f"{HERE}/results/soup_pat_agg.json", "w"), indent=1)
for k, v in agg.items():
    if "phi" in k: print(k, "%.4f ± %.4f  pos %d/3" % tuple(v))
