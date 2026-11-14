"""'Decayed wealth' page section, built from results/decay_*_s*.csv (see decay.py)."""
import csv, glob, os
import numpy as np
import matplotlib.pyplot as plt

HALF = [100, 300, 1000, 3000, 10000]
WIN = [100, 500, 2000]
BLUES = ["#9ec5f4", "#6ba6ea", "#2a78d6", "#1c5aa6", "#123f78"]   # light -> dark by half-life
ORNG = ["#f4b08f", "#eb6834", "#a8431c"]


def _read(res, name):
    rows = []
    for f in sorted(glob.glob(os.path.join(res, f"decay_{name}_s*.csv"))):
        rows += list(csv.DictReader(open(f)))
    return rows


def section(res, svg, LIGHT):
    A, S = _read(res, "ablation"), _read(res, "spearman")
    if not A or not S:
        return ""
    for r in A:
        r["seed"], r["step"], r["final"] = int(r["seed"]), int(r["step"]), int(r["final"])
        r["frac"], r["bce"] = float(r["frac"]), float(r["bce"])
    seeds = sorted({r["seed"] for r in A})
    base = {(r["seed"], r["final"]): r["bce"] for r in A if r["criterion"] == "none"}

    def curve(c, final):
        xs = sorted({r["frac"] for r in A if r["criterion"] == c})
        ys = [np.mean([r["bce"] - base[(r["seed"], final)] for r in A
                       if r["criterion"] == c and r["final"] == final and r["frac"] == x]) for x in xs]
        return [100 * x for x in xs], ys

    fig, axs = plt.subplots(2, 2, figsize=(9, 6.4), sharey="col")
    for j, final in enumerate([1, 0]):
        for i, group in enumerate(["decay", "window"]):
            ax = axs[i, j]
            items = ([(f"hl{h}", f"half-life {h}", BLUES[k]) for k, h in enumerate(HALF)] + [("cum", "all-time wealth", "#123456")]
                     if group == "decay" else [(f"win{w}", f"last {w} steps", ORNG[k]) for k, w in enumerate(WIN)] + [("cum", "all-time wealth", "#123456")])
            for c, lab, col in items:
                x, y = curve(c, final)
                ax.plot(x, y, color=col, marker="o", ms=3, label=lab)
            for c, lab, col, ls in (("random", "random", LIGHT[3], "--"), ("gradnorm", "smallest grad norm", LIGHT[2], ":")):
                x, y = curve(c, final)
                ax.plot(x, y, color=col, ls=ls, marker="o", ms=3, label=lab)
            ax.set_title(f"{'final' if final else 'step 20000'} checkpoint", fontsize=9)
            ax.set_xlabel("% of neurons removed (lowest currency first)"); ax.set_ylabel("held-out BCE increase")
            if j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); f1 = svg(fig)

    sp = {(int(r["seed"]), r["currency"]): float(r["spearman_vs_oracle"]) for r in S}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    xs = HALF + [30000]
    for k, sd in enumerate(seeds):
        axs[0].plot(xs, [sp[(sd, f"hl{h}")] for h in HALF] + [sp[(sd, "cum")]], color=LIGHT[k], marker="o", ms=3, label=f"seed {sd}")
        axs[1].plot(WIN, [sp[(sd, f"win{w}")] for w in WIN], color=LIGHT[k], marker="o", ms=3, label=f"seed {sd}")
    axs[0].set_xscale("log"); axs[0].set_xticks(xs); axs[0].set_xticklabels([str(h) for h in HALF] + ["none"])
    axs[0].set_xlabel("half-life (steps); 'none' = all-time wealth"); axs[0].set_ylabel("Spearman vs single-neuron ablation")
    axs[1].set_xscale("log"); axs[1].set_xticks(WIN); axs[1].set_xticklabels([str(w) for w in WIN]); axs[1].set_xlabel("income window (steps)")
    for ax in axs:
        ax.axhline(0, color="#654321", lw=0.8)
    axs[0].legend(fontsize=7); fig.tight_layout(); f2 = svg(fig)

    cur = ["cum"] + [f"hl{h}" for h in HALF] + [f"win{w}" for w in WIN] + ["gradnorm"]
    name = {"cum": "all-time wealth", "gradnorm": "loss-gradient norm (control)", **{f"hl{h}": f"decayed, half-life {h}" for h in HALF},
            **{f"win{w}": f"income, last {w} steps" for w in WIN}}
    fr = {(int(r["seed"]), r["currency"]): r["frac_negative"] for r in S}
    trs = ""
    for c in cur:
        v = [sp[(sd, c)] for sd in seeds]
        a5 = [np.mean([r["bce"] - base[(r["seed"], 1)] for r in A if r["criterion"] == c and r["final"] == 1 and abs(r["frac"] - p) < 1e-9]) for p in (0.02, 0.05)]
        trs += (f"<tr><td>{name[c]}</td>" + "".join(f"<td>{x:+.3f}</td>" for x in v) + f"<td><b>{np.mean(v):+.3f}</b></td>"
                + "".join(f"<td>{float(fr[(sd, c)]):.2f}</td>" if fr[(sd, c)] != "" else "<td></td>" for sd in seeds)
                + "".join(f"<td>{x:+.3f}</td>" for x in a5) + "</tr>")
    ohd = {int(r["seed"]): r["oracle_n_harmful"] for r in S}
    return (f"<h2>Decayed wealth</h2><p>Kelvin's question: all-time wealth is history; is recent income the better currency? "
            f"Each neuron's income per 100 steps (from the snapshots) is turned into a decayed wealth "
            f"W<sub>t</sub> = &lambda;W<sub>t-1</sub> + income<sub>t</sub> with half-lives 100 to 10000 steps, and into income over the last 100, 500 and 2000 steps. "
            f"The oracle is the real effect of zeroing one neuron at a time on held-out BCE (final checkpoint; positive = the neuron helps). "
            f"Oracle-harmful neurons (removing one alone lowers held-out BCE): "
            + ", ".join(f"seed {sd}: {ohd[sd]}" for sd in seeds) + " of 512. "
            f"No training; the held-out segment is the one used above, subsampled (every 4th position for the curves, every 8th for the oracle) to fit the CPU budget. "
            f"Data: <code>decay_ablation_s*.csv</code>, <code>decay_spearman_s*.csv</code>.</p>{f2}"
            f"<div class=tw><table><tr><th>currency</th>" + "".join(f"<th>Spearman s{sd}</th>" for sd in seeds) + "<th>mean</th>"
            + "".join(f"<th>share negative s{sd}</th>" for sd in seeds) + "<th>BCE +, 2% removed</th><th>BCE +, 5% removed</th></tr>"
            f"{trs}</table></div><p>Held-out BCE increase when the lowest-currency neurons are removed, mean of seeds. Random: dashed; smallest gradient norm: dotted.</p>{f1}")
