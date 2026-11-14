# figs.py OUT_DIR FIG_DIR : three plain figures from OUT_DIR/RESULT-work.json (numbers only from that file).
import sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT, FIG = sys.argv[1], sys.argv[2]
R = json.load(open(OUT + "/RESULT-work.json"))
KS = [1, 2, 4, 8, 16, 32, 64]
FAM = {"arithL": "arithmetic chain", "vars": "variable tracking"}
AGG = {"vote": "(a) plain vote", "dedup": "(b) each distinct demonstration once, weight = M's score",
       "snis": "(c) weight = M's score / writer's probability"}
AGS = {"vote": "(a) plain vote", "dedup": "(b) distinct, score-weighted", "snis": "(c) importance-weighted"}
# fixed colour per writer (validated set, see NOTE); line style repeats the identity
WR = [("W1", "M itself, T=1", "#2a78d6", "-", "o"), ("W3", "biased LoRA (wrong last step)", "#eb6834", "-", "s"),
      ("W3s", "biased LoRA (prefers wrong answers)", "#1baf7a", "--", "^"), ("W3p", "biased prompt", "#eda100", "-.", "v"),
      ("W4", "helpful LoRA", "#e87ba4", ":", "D")]
INK, MUTE = "#0b0b0b", "#52514e"

def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=MUTE, labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])

def grid(R, KS, metric, ylabel, fname, title, ylim, direct=False, writers=None):
    fig, axs = plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for i, fam in enumerate(("arithL", "vars")):
        for j, g in enumerate(("vote", "dedup", "snis")):
            ax = axs[i][j]; style(ax)
            ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
            for w, lab, col, ls, mk in WR:
                if writers and w not in writers:
                    continue
                y = [R["curves"]["%s|%s|%s|%d" % (fam, w, g, K)][metric] for K in KS]
                ax.plot(KS, y, color=col, linestyle=ls, marker=mk, markersize=4, linewidth=1.8, label=lab)
            if direct:
                d = R["direct"][fam]["greedy_acc"]
                ax.axhline(d, color=MUTE, linewidth=1, linestyle=(0, (1, 2)))
                ax.text(KS[-1], d + 0.02, "M's direct answer", ha="right", va="bottom", fontsize=7.5, color=MUTE)
            ax.set_ylim(*ylim)
            if i == 0:
                ax.set_title(AGS[g], fontsize=9.5, color=INK)
            if j == 0:
                ax.set_ylabel("%s\n%s" % (FAM[fam], ylabel), fontsize=8.5, color=INK)
            if i == 1:
                ax.set_xlabel("K rollouts per problem", fontsize=8.5, color=INK)
    h, l = axs[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, title="who writes the rollouts", title_fontsize=8)
    fig.suptitle(title, fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(FIG + "/" + fname, dpi=150, facecolor="white"); plt.close(fig)

grid(R, KS, "acc", "accuracy", "fig1_accuracy.png",
     "Accuracy against the exact answers, by who writes the rollouts and how they are aggregated (150 problems per family)", (0, 1.02), direct=True)
grid(R, KS, "tv_ref", "distance to target", "fig2_distance.png",
     "Distance (total variation) from the aggregated answer distribution to the target: M's own answer distribution, from 256 rollouts of M", (0, 0.7))
import os
if os.path.exists(OUT + "/RESULT-ext.json"):
    E = json.load(open(OUT + "/RESULT-ext.json"))
    grid(E, [1, 2, 4, 8, 16, 32, 64, 128, 256], "tv_ref", "distance to target", "fig3_extension.png",
         "The same distance with up to 256 rollouts (32 problems per family; target from 1,024 rollouts of M)", (0, 0.7))
print("figures written")
