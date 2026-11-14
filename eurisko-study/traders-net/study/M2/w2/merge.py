"""Merge results/pred-s*-*-step*.csv into results/predictors.csv and write a coverage table plus a
preview Spearman vs the M2/w1 oracle (dloss_zero, dloss_mean) where available. Worker 3 owns the
real scoring; the preview only checks sign conventions and that ids line up."""
import csv, glob, json, os
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = []
for f in sorted(glob.glob(os.path.join(R, "pred-s*-*-step*.csv"))):
    rows += list(csv.DictReader(open(f)))
cols = ["seed", "step", "final", "gran", "flat", "unit", "predictor", "value"]
with open(os.path.join(R, "predictors.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
tab = {}
for r in rows:
    tab.setdefault((r["predictor"], r["gran"], int(r["seed"]), int(r["step"])), {})[int(r["flat"])] = float(r["value"])
cov = {}
for (p, g, s, st) in tab:
    cov.setdefault((p, g), set()).add((s, st))
with open(os.path.join(R, "coverage.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["predictor", "gran", "n_seed_step", "seed_steps"])
    for (p, g), v in sorted(cov.items()):
        w.writerow([p, g, len(v), " ".join(f"s{a}@{b}" for a, b in sorted(v))])
orc = glob.glob(os.path.join(HERE, "..", "w1", "results", "oracle_units*.csv"))
out = []
if orc:
    O = {}
    for f in orc:
        for r in csv.DictReader(open(f)):
            gg = {"neuron": "neuron", "head": "head", "readout": "readout", "out": "readout"}.get(r["gran"], r["gran"])
            for k in ("dloss_zero", "dloss_mean"):
                if r.get(k, "") != "":
                    O.setdefault((k, gg, int(r["seed"]), int(r["step"])), {})[int(r["flat"])] = float(r[k])
    for (p, g, s, st), d in tab.items():
        for k in ("dloss_zero", "dloss_mean"):
            o = O.get((k, g, s, st))
            if not o:
                continue
            ids = sorted(set(d) & set(o))
            if len(ids) < 5:
                continue
            x = np.array([d[i] for i in ids]); y = np.array([o[i] for i in ids])
            rho = spearmanr(x, y).correlation
            out.append(dict(predictor=p, gran=g, seed=s, step=st, oracle=k, n=len(ids), spearman=round(float(rho), 4),
                            spearman_abs=round(float(spearmanr(np.abs(x), y).correlation), 4)))
    with open(os.path.join(R, "preview_spearman.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    # mean +- sd over seeds per predictor x gran x step x oracle
    agg = {}
    for r in out:
        agg.setdefault((r["predictor"], r["gran"], r["step"] if r["step"] < 31000 else "final", r["oracle"]), []).append(r["spearman"])
    with open(os.path.join(R, "preview_spearman_agg.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["predictor", "gran", "step", "oracle", "n_seeds", "mean", "sd"])
        for k, v in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][3], str(kv[0][2]), -np.nanmean(kv[1]))):
            w.writerow(list(k) + [len(v), round(float(np.nanmean(v)), 4), round(float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan"), 4)])
print(len(rows), "rows;", len(cov), "predictor x gran;", len(out), "preview correlations")

# Proxy target available without w1: abl_val = exact zero-ablation dBCE on held-out positions
# T+8192..T+8703 at the final checkpoint (same quantity as w1's dloss_zero, different segment, n=512).
px = []
for (p, g, s, st), d in tab.items():
    t = tab.get(("abl_val", g, s, st))
    if not t or p == "abl_val":
        continue
    ids = sorted(set(d) & set(t))
    if len(ids) < 5:
        continue
    x = np.array([d[i] for i in ids]); y = np.array([t[i] for i in ids])
    px.append(dict(predictor=p, gran=g, seed=s, step=st, n=len(ids), spearman=round(float(spearmanr(x, y).correlation), 4),
                   spearman_abs=round(float(spearmanr(np.abs(x), y).correlation), 4)))
if px:
    agg = {}
    for r in px:
        agg.setdefault((r["predictor"], r["gran"]), []).append((r["spearman"], r["spearman_abs"]))
    with open(os.path.join(R, "proxy_spearman_ablval_final.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["predictor", "gran", "n_seeds", "mean", "sd", "mean_abs", "sd_abs"])
        for (p, g), v in sorted(agg.items(), key=lambda kv: (kv[0][1], -np.nanmean([a for a, _ in kv[1]]))):
            a = np.array(v, float)
            sd = lambda c: round(float(np.nanstd(c, ddof=1)), 4) if len(c) > 1 else float("nan")
            w.writerow([p, g, len(v), round(float(np.nanmean(a[:, 0])), 4), sd(a[:, 0]), round(float(np.nanmean(a[:, 1])), 4), sd(a[:, 1])])
    print(len(px), "proxy correlations vs abl_val")
