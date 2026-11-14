"""Max expert share per step (layer 1) for each control arm, E=8 and E=64, seed 1 (GPU runs)."""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__))
arms = ["none", "switch", "deepseek-u1e-3", "deepseek-u1e-2", "expertchoice"]
cols = ["#444444", "#1f77b4", "#d62728", "#ff9896", "#2ca02c"]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, E in zip(axs, (8, 64)):
    for a, c in zip(arms, cols):
        f = os.path.join(H, f"results/gpu/runs/{a}-E{E}-s1.trace.csv")
        rows = [r for r in csv.DictReader(open(f)) if r["layer"] == "1"]
        ax.plot([int(r["step"]) for r in rows], [float(r["max_share"]) for r in rows], label=a, color=c, lw=1.2)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.axhline(2 / E, color="gray", ls="--", lw=0.8)
    ax.set_title(f"E={E}, top-2, layer 1, seed 1 (dashed = fair share 2/E)")
    ax.set_xlabel("prequential step")
axs[0].set_ylabel("max fraction of tokens on one expert")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(H, "results/max_load_trace.png"), dpi=110)
