"""'Exact settlement' page section, from results/settle_*_s*.csv and decay_ablation (see settle.py)."""
import csv, glob, os
import numpy as np
import matplotlib.pyplot as plt

NAME = {"loo": "leave-one-out", "seq": "layer by layer (logit lens)", "shap": "sampled Shapley"}


def _read(res, pat):
    return [r for f in sorted(glob.glob(os.path.join(res, pat))) for r in csv.DictReader(open(f))]


def section(res, svg, LIGHT):
    S, A = _read(res, "settle_spearman_s*.csv"), _read(res, "settle_ablation_s*.csv") + _read(res, "decay_ablation_s*.csv")
    if not S:
        return ""
    base = {(r["seed"], r["final"]): float(r["bce"]) for r in A if r["criterion"] == "none"}
    seeds = sorted({r["seed"] for r in S})

    def curve(c, fin):
        xs = [0.01, 0.02, 0.05, 0.1, 0.2]
        ys = [np.mean([float(r["bce"]) - base[(r["seed"], fin)] for r in A
                       if r["criterion"] == c and r["final"] == fin and abs(float(r["frac"]) - x) < 1e-9]) for x in xs]
        return [100 * x for x in xs], ys

    col = {"loo": LIGHT[0], "seq": LIGHT[6], "shap": LIGHT[4]}
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    for ax, fin, ttl in ((axs[0], "1", "final checkpoint"), (axs[1], "0", "step 20000 checkpoint")):
        for k in ("loo", "seq", "shap"):
            for seg, ls in (("heldout", "-"), ("train", "--")):
                x, y = curve(f"{k}_{seg}", fin)
                ax.plot(x, y, color=col[k], ls=ls, marker="o", ms=3, label=f"{NAME[k]}, {'held-out' if seg == 'heldout' else 'training'} data")
        for c, lab, cl, ls in (("cum", "all-time gradient wealth", "#123456", "-"), ("random", "random", LIGHT[3], "--"), ("gradnorm", "smallest grad norm", LIGHT[2], ":")):
            x, y = curve(c, fin)
            ax.plot(x, y, color=cl, ls=ls, marker="o", ms=3, label=lab)
        ax.axhline(0, color="#654321", lw=0.8)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("% of neurons removed (lowest settlement first)")
    axs[0].set_ylabel("held-out BCE increase")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.2, 1, 1)); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7); f1 = svg(fig)
    trs = ""
    for seg in ("heldout", "train"):
        for k in ("loo", "seq", "shap"):
            rr = [r for r in S if r["currency"] == f"{k}_{seg}"]
            mean = lambda f: np.mean([float(r[f]) for r in rr])
            trs += (f"<tr><td>{NAME[k]}</td><td>{'held-out' if seg == 'heldout' else 'training'}</td>"
                    + "".join(f"<td>{float(r['spearman_vs_oracle']):+.3f}</td>" for r in rr)
                    + f"<td><b>{mean('spearman_vs_oracle'):+.3f}</b></td><td>{mean('spearman_layer0'):+.3f}</td><td>{mean('spearman_layer1'):+.3f}</td>"
                    f"<td>{mean('top5_in_top10'):.2f}</td><td>{mean('top5_in_bottom10'):.2f}</td><td>{mean('frac_negative'):.2f}</td></tr>")
    return (f"<h2>Exact settlement</h2><p>Gradient credit replaced by the market's own settlement at the output. A neuron's direct logit contribution "
            f"c<sub>i</sub> is its output vector read through the final LayerNorm (scale held fixed) and readout, so removing it costs no extra forward pass. "
            f"It earns the log-score gain it caused: <b>leave-one-out</b> log p(y) &minus; log p<sub>&minus;i</sub>(y); <b>layer by layer</b>: each MLP layer earns the "
            f"logit-lens gain across it, split inside the layer by leave-one-out, with the leftover spread equally so the layer's neurons sum exactly to its earnings; "
            f"<b>sampled Shapley</b>: 8 random neuron orders per example over the 512 direct contributions (sums to the total gain within 0.005). "
            f"Summed over the held-out segment (the labels the oracle is measured on, so an upper bound) and over 4096 training positions (what a market would have seen). "
            f"Oracle: the single-neuron ablation from the section above. Data: <code>settle_spearman_s*.csv</code>, <code>settle_ablation_s*.csv</code>.</p>"
            f"<div class=tw><table><tr><th>settlement</th><th>data</th>" + "".join(f"<th>Spearman s{sd}</th>" for sd in seeds)
            + "<th>mean</th><th>layer 0 only</th><th>layer 1 only</th><th>oracle top 5% in its top 10%</th><th>in its bottom 10%</th><th>share negative</th></tr>"
            f"{trs}</table></div><p>Held-out BCE increase when the lowest-settlement neurons are removed, mean of 3 seeds. Below zero = removal helps.</p>{f1}")
