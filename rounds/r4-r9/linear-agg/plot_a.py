"""Readable Experiment A figure from runs/a-eval/summary.json: adjusted MSE (MSE - oracle) by noise level, 3-seed MEDIAN
(robust to the one exploding seed some arms have), y-axis clipped; off-scale points drawn as triangles at the top edge.
The pre-registered statistics (3-seed means) are in the tables.   python3 plot_a.py"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
s = json.load(open(os.path.join(HERE, "runs", "a-eval", "summary.json")))
mse = s["mse"]
SIG = (0.1, 0.5, 1.0, 2.0, 3.0)
xs = np.array(SIG)
orc = np.array([mse[f"oracle|s{g}"] for g in SIG])
TOP = 0.8
fig, axs = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
for ax, L in zip(axs, (1, 2, 4, 6)):
    ax.plot(xs, np.array([mse[f"adarr|s{g}"] for g in SIG]) - orc, "k--", lw=1.5, label="AdaRR (closed form)")
    for b, st in (("mix", ":"), ("sw", "-."), ("dd", (0, (1, 3)))):
        ax.plot(xs, np.array([np.mean([mse[f"{b}|s{k}|s{g}"] for k in (0, 1, 2)]) for g in SIG]) - orc, color="0.55", ls=st,
                label=f"{b} (1-step)")
    for kind, col, nm in (("gdpp", "#c0392b", "GD++"), ("diag", "#2471a3", "Diag"), ("full", "#1e8449", "Full")):
        per = np.array([[mse[f"{kind}-L{L}-s{k}|s{g}"] for k in (0, 1, 2)] for g in SIG]) - orc[:, None]
        med = np.median(per, 1)
        ok = med < TOP
        ax.plot(xs[ok], med[ok], color=col, marker="o", ms=4, label=f"{nm} (median of 3 seeds)")
        ax.scatter(xs, np.minimum(per.min(1), TOP), color=col, marker="_", s=60)
        ax.scatter(xs, np.minimum(per.max(1), TOP), color=col, marker="_", s=60)
        off = per.max(1) >= TOP
        if off.any():
            ax.scatter(xs[off], np.full(off.sum(), TOP * 0.985), color=col, marker="^", s=40)
    ax.set_title(f"L = {L}")
    ax.set_xlabel("episode noise sigma")
    ax.set_ylim(-0.02, TOP)
    ax.axhline(0, color="0.8", lw=0.8)
axs[0].set_ylabel("MSE - oracle MSE")
axs[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Experiment A: query MSE above the exact posterior. Lines: 3-seed median; bars: seed min / max; "
             "triangles: a seed off the scale (exploding predictions)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "runs", "a-eval", "a-adjusted-mse.png"), dpi=120)
print("ok")
