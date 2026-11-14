"""Summarise results/frontier-s*.csv: dBCE vs fraction pruned, AUC over 0..40% (trapezoid,
lower = better), mean +- sd over seeds; plot results/frontier-<ckpt>.png."""
import csv, glob, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results")
FINAL = {1: 31100, 2: 31200, 3: 31100}
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "frontier-s*.csv"))):
    rows += list(csv.DictReader(open(f)))
cur = defaultdict(lambda: defaultdict(list))   # (seed,ckpt,gran,pred) -> frac -> [dbce over orders]
base = {}
for r in rows:
    s, st = int(r["seed"]), int(r["step"]); c = "final" if st == FINAL[s] else str(st)
    fr, b = float(r["frac"]), float(r["bce"])
    if int(r["k"]) == 0:
        base[(s, c, r["gran"])] = b
    cur[(s, c, r["gran"], r["predictor"])][fr].append(b)
auc, curve = [], []
for (s, c, g, p), d in cur.items():
    b0 = base[(s, c, g)]
    fr = np.array(sorted(d)); y = np.array([np.mean(d[x]) - b0 for x in fr])
    for x, v in zip(fr, y):
        curve.append({"seed": s, "ckpt": c, "gran": g, "predictor": p, "frac": x, "dbce": round(float(v), 6)})
    m = fr <= 0.4 + 1e-9
    auc.append({"seed": s, "ckpt": c, "gran": g, "predictor": p, "auc_0_40": float(np.trapezoid(y[m], fr[m])) / fr[m].max(),
                "dbce_at": {round(float(x), 4): float(v) for x, v in zip(fr, y)}})
summ = []
grp = defaultdict(list)
for a in auc:
    grp[(a["ckpt"], a["gran"], a["predictor"])].append(a)
for (c, g, p), v in sorted(grp.items()):
    A = [x["auc_0_40"] for x in v]
    row = {"ckpt": c, "gran": g, "predictor": p, "n_seeds": len(v), "auc_mean": round(np.mean(A), 5),
           "auc_sd": round(np.std(A, ddof=1), 5) if len(A) > 1 else ""}
    n = {"neuron": 512, "head": 8, "readout": 64}[g]
    for fr in ([0.01, 0.05, 0.2, 0.4] if g != "head" else [0.125, 0.25, 0.375, 0.5]):
        key = round(round(fr * n) / n, 4)
        xs = [x["dbce_at"].get(key) for x in v if key in x["dbce_at"]]
        row[f"dbce@{fr}"] = f"{np.mean(xs):.4f}+-{np.std(xs, ddof=1):.4f}" if len(xs) > 1 else ""
    summ.append(row)
for name, rs in (("frontier_curves.csv", curve), ("frontier_summary.csv", summ)):
    keys = []
    for r in rs:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rs)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    allp = sorted(set(p for (_, _, p) in grp))
    cmap = plt.get_cmap("tab20")
    col = {p: cmap(i % 20) for i, p in enumerate(allp)}
    col["random"] = "black"; col["oracle_zero"] = "tab:gray"
    for c in sorted(set(a["ckpt"] for a in auc)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, g in zip(axs, ("neuron", "head", "readout")):
            preds = sorted(set(p for (cc, gg, p) in grp if cc == c and gg == g))
            best = sorted(preds, key=lambda p: np.mean([x["auc_0_40"] for x in grp[(c, g, p)]]))
            show = [p for p in best if p in ("random", "oracle_zero")] + [p for p in best if p not in ("random", "oracle_zero")][:6] \
                + [p for p in best if p.startswith(("v1_wealth", "wealth_v1", "taylor", "movement"))]
            for p in dict.fromkeys(show):
                fr = sorted(set(x for s in (1, 2, 3) for x in cur.get((s, c, g, p), {})))
                ys = np.array([[np.mean(cur[(s, c, g, p)][x]) - base[(s, c, g)] for x in fr] for s in (1, 2, 3) if (s, c, g, p) in cur])
                ax.errorbar(fr, ys.mean(0), yerr=ys.std(0, ddof=1) if len(ys) > 1 else None, label=p, capsize=2, color=col[p],
                            lw=2.5 if p in ("random", "oracle_zero") else 1.2, ls="--" if p == "random" else "-")
            ax.set_title(f"{g}, ckpt {c}"); ax.set_xlabel("fraction pruned (least important first)")
            ax.set_ylabel("held-out dBCE (T+6144..T+8191)"); ax.set_xlim(0, 0.62); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"frontier-{c}.png"), dpi=110); plt.close(fig)
except ImportError as e:
    print("no matplotlib:", e)
print("wrote", len(summ), "summary rows")
