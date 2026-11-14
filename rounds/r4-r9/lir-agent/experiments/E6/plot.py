import json, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
col = {"c": "#1f77b4", "b_c": "#9ecae1", "d": "#d62728", "b_d": "#fc9272", "a_c": "#777777", "a_d": "#bbbbbb"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for name, per in R["arms"].items():
    for i, (s, rows) in enumerate(per.items()):
        st = [r["step"] for r in rows]
        kw = dict(color=col.get(name, "k"), alpha=0.8, label=name if i == 0 else None)
        ax[0].plot(st, [r["ssi"] for r in rows], **kw)
        ax[1].plot(st, [r["gold_acc"] for r in rows], **kw)
        if not name.startswith("a_"):
            ax[2].plot([r["norm"] for r in rows], [r["ssi"] for r in rows], marker=".", **kw)
ax[0].set(xlabel="RL step", ylabel="SSI (reward units)", title="Self-serving index")
ax[1].set(xlabel="RL step", ylabel="gold pairwise accuracy (non-tied)", title="Gold accuracy of proxy")
ax[2].set(xlabel="proxy-adapter update norm", ylabel="SSI", title="SSI vs update norm")
for a in ax:
    a.axhline(0, color="k", lw=0.5); a.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "e6_ssi.png"), dpi=110)
print("wrote e6_ssi.png")
