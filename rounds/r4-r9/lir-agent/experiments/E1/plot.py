"""Calibration-under-RL curves from results.json -> calib_under_rl.png (small multiples, one axis each)."""
import json, os, sys
sys.path.insert(0, "/tmp/lir-agent/E1/pylib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))["arms"]
TAGS, X = ["pre_rl", "rl30", "post_rl"], [0, 30, 60]
ARMS = [("P", "#2a78d6", "P-role"), ("D", "#eb6834", "D (token mask)"), ("F", "#1baf7a", "F (fused)"),
        ("R", "#eda100", "R (random reward)"), ("B0floor", "#e87ba4", "B0-floor"), ("Pall", "#008300", "P-all")]
M = [("fc_committed_ece_tok", "committed P(success) ECE"), ("fc_committed_opt_vs_real_stated", "signed optimism (stated)"),
     ("agree_false_note_stated", "agree with false note"), ("env_reward", "env reward")]
fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for row, ch in enumerate(("on", "off")):
    for col, (m, title) in enumerate(M):
        ax = axes[row, col]
        for arm, c, lab in ARMS:
            a = R.get(f"{arm}-{ch}")
            if not a:
                continue
            mu = [a[t][m]["mean"] for t in TAGS]; sd = [a[t][m]["sd"] or 0 for t in TAGS]
            if None in mu:
                continue
            ax.plot(X, mu, color=c, lw=2, marker="o", ms=8, label=f"{lab} (n={a[TAGS[-1]][m]['n']})")
            ax.fill_between(X, [u - s for u, s in zip(mu, sd)], [u + s for u, s in zip(mu, sd)], color=c, alpha=0.12, lw=0)
        ax.set_title(f"{title}, channel {ch.upper()}", fontsize=10, color="#333")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        if row == 1: ax.set_xlabel("RL step")
axes[0, 0].legend(fontsize=8, frameon=False); axes[1, 0].legend(fontsize=8, frameon=False)
fig.suptitle("E1: calibration under RL, mean ± sd over seeds (held-out syco eval, 128 episodes)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "calib_under_rl.png"), dpi=110)
print("wrote calib_under_rl.png")
