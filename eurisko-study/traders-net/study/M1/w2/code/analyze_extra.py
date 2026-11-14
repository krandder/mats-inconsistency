"""M1(b) extras: Spearman of wealth W with the cumulative squared-gradient term (second),
time course of R^2(T,N), and a W-vs-N scatter figure. Usage: analyze_extra.py RUN_DIR OUT_DIR"""
import sys, os, glob
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rd, od = sys.argv[1], sys.argv[2]; os.makedirs(od, exist_ok=True)
rows = []; data = {}
for f in sorted(glob.glob(os.path.join(rd, "*.units.csv"))):
    tag = os.path.basename(f).replace(".units.csv", ""); arm, seed = tag.rsplit("-s", 1)
    d = pd.read_csv(f); data[tag] = d
    for step in sorted(d.step.unique()):
        for gname, g in d[d.step == step].groupby("gran"):
            rows.append(dict(arm=arm, seed=int(seed), step=int(step), gran=gname,
                             rho_W_second=spearmanr(g.W, g.second).correlation,
                             rho_W_sqnorm=spearmanr(g.W, g.sqnorm).correlation,
                             r2_T_N=np.corrcoef(g["T"], g.N)[0, 1] ** 2,
                             r2_W_N=np.corrcoef(g.W, g.N)[0, 1] ** 2,
                             second_share=np.abs(g.second).sum() / np.abs(g["T"]).sum()))
ex = pd.DataFrame(rows); ex.to_csv(os.path.join(od, "extra_per_run.csv"), index=False)
agg = ex.groupby(["arm", "gran", "step"])[["rho_W_second", "rho_W_sqnorm", "r2_T_N", "r2_W_N", "second_share"]].agg(["mean", "std"])
agg.columns = [f"{a}_{b}" for a, b in agg.columns]; agg.reset_index().to_csv(os.path.join(od, "extra_summary.csv"), index=False)
last = ex.step.max()
print(agg.reset_index().query("step in [1000, 5000, @last] and gran in ['out','neuron','head']").round(3).to_string(index=False))
arms = sorted({t.rsplit("-s", 1)[0] for t in data})
fig, axs = plt.subplots(2, len(arms), figsize=(3.4 * len(arms), 6.4))
for j, arm in enumerate(arms):
    d = data[f"{arm}-s1"]; d = d[d.step == d.step.max()]
    for i, gr in enumerate(["out", "neuron"]):
        g = d[d.gran == gr]; ax = axs[i, j]
        ax.scatter(g.N, g.W, s=10, color="#2a78d6", alpha=0.6, linewidths=0)
        ax.axhline(0, color="#c3c2b7", lw=0.8); ax.axvline(0, color="#c3c2b7", lw=0.8)
        r2 = np.corrcoef(g.N, g.W)[0, 1] ** 2
        ax.set_title(f"{arm}, {gr} (seed 1): R²={r2:.2f}", fontsize=9, color="#0b0b0b")
        ax.set_xlabel("norm growth N = QΔ|θ|²/2η", fontsize=8, color="#52514e"); ax.set_ylabel("books wealth W", fontsize=8, color="#52514e")
        ax.tick_params(labelsize=7, colors="#52514e")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(od, "wealth_vs_normgrowth.png"), dpi=110)
