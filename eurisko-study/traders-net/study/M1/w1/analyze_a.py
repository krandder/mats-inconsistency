"""Summarise units-s*.csv -> results/fits.csv (per seed) and results/fits_summary.csv (mean, sd over seeds);
frozen control -> results/frozen_control.csv."""
import csv, json, sys
import numpy as np
from scipy.stats import spearmanr

WN = "/home/agent/workspaces/eurisko-study/wealth-net"
R = "results"
PRED = ["X_norm_out", "X_norm_in", "X_move_out", "X_move_in", "X_wdisp_out", "X_grad_live_out", "X_grad_books_out", "X_normT_out"]


def fit(y, x):
    X = np.c_[np.ones_like(x), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # through-origin scale (the identity predicts slope 1, intercept 0)
    k = (x @ y) / max(1e-30, x @ x)
    rel0 = np.linalg.norm(y - k * x) / np.linalg.norm(y)
    rel1 = np.linalg.norm(y - x) / np.linalg.norm(y)          # residual of the identity itself (slope 1)
    return dict(r2=r2, slope=b[1], k0=k, relres_fit0=rel0, relres_slope1=rel1,
                spearman=spearmanr(y, x)[0], sign_agree=float(np.mean(np.sign(y) == np.sign(x))))


rows = []
for s in (1, 2, 3):
    D = list(csv.DictReader(open(f"{R}/units-s{s}.csv")))
    final = json.load(open(f"{WN}/results/train-s{s}.json"))["steps_done"]
    sets = {"pooled_2k_intervals": lambda r: int(r["t0"]) >= 8000 and int(r["t1"]) - int(r["t0"]) <= 2000,
            "cum_8k_to_final": lambda r: int(r["t0"]) == 8000 and int(r["t1"]) == final,
            "cum_0_to_final(recon_init)": lambda r: int(r["t0"]) == 0 and int(r["t1"]) == final,
            "interval_0_to_8k(recon_init)": lambda r: int(r["t0"]) == 0 and int(r["t1"]) == 8000}
    for name, sel in sets.items():
        for g in ["neuron", "head", "out"]:
            sub = [r for r in D if r["gran"] == g and sel(r)]
            y = np.array([float(r["Y"]) for r in sub])
            for p in PRED:
                x = np.array([float(r[p]) for r in sub])
                rows.append(dict(seed=s, set=name, gran=g, pred=p, n=len(y), **fit(y, x)))
keys = ["r2", "slope", "k0", "relres_fit0", "relres_slope1", "spearman", "sign_agree"]
with open(f"{R}/fits.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
summ = []
for (st, g, p) in sorted({(r["set"], r["gran"], r["pred"]) for r in rows}):
    sub = [r for r in rows if (r["set"], r["gran"], r["pred"]) == (st, g, p)]
    d = dict(set=st, gran=g, pred=p, n=sub[0]["n"])
    for k in keys:
        v = np.array([r[k] for r in sub]); d[k + "_mean"] = round(float(v.mean()), 4); d[k + "_sd"] = round(float(v.std(ddof=1)), 4)
    summ.append(d)
with open(f"{R}/fits_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

# frozen control: income per step with zero weight movement, vs train-run income per step at the end
fr = []
for s in (1, 2, 3):
    z = np.load(f"{WN}/results/frozen-s{s}.npz")
    nsteps = int(z["step"][-1])
    frz = z["neuron"][-1].sum(-1).ravel() / nsteps
    final = json.load(open(f"{WN}/results/train-s{s}.json"))["steps_done"]
    a = json.load(open(f"{WN}/results/train-s{s}/snapshots/step-30000.json"))["wealth"]["neuron"]
    b = json.load(open(f"{WN}/results/train-s{s}/snapshots/step-{final:05d}.json"))["wealth"]["neuron"]
    tr = (np.array(b) - np.array(a)) / (final - 30000)
    fr.append(dict(seed=s, frozen_steps=nsteps, mean_abs_frozen=float(np.abs(frz).mean()), mean_abs_train=float(np.abs(tr).mean()),
                   ratio=float(np.abs(frz).mean() / np.abs(tr).mean()), sum_frozen=float(frz.sum()), sum_train=float(tr.sum()),
                   spearman=float(spearmanr(frz, tr)[0])))
with open(f"{R}/frozen_control.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(fr[0].keys())); w.writeheader(); w.writerows(fr)
print("ok")
