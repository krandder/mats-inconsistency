"""'Block-local settlement' page section, from results/blocklocal/{A,B,C}-s*.csv and -log.json (see blocklocal.py)."""
import csv, glob, json, os
import numpy as np
import matplotlib.pyplot as plt
from streams import FAMILIES

ARM = {"A": "A: end-to-end", "B": "B: block-local", "C": "C: block-local + market",
       "R": "C-random: random skip + rebirth", "W": "C-wealth only: no skip, no rebirth"}


def rows(res):
    return [r for f in sorted(glob.glob(os.path.join(res, "blocklocal", "[ABCRW]-s*.csv"))) for r in csv.DictReader(open(f))]


def table(R):
    """Per-arm means (and seed spread) of every numeric column."""
    out = {}
    for a in "ABCRW":
        rr = [r for r in R if r["arm"] == a]
        if not rr:
            continue
        d = {"n": len(rr)}
        for k in rr[0]:
            if k in ("arm", "seed"):
                continue
            v = [float(r[k]) for r in rr if r[k] not in ("", None)]
            if v:
                d[k] = (float(np.mean(v)), float(np.std(v)))
        out[a] = d
    return out


def section(res, svg, LIGHT):
    R = rows(res)
    if not R:
        return ""
    T = table(R)
    base = T.get("A")
    fam_cols = [f for f in FAMILIES if f"bce_{f}" in R[0]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(fam_cols))
    bars = [a for a in "BCRW" if a in T]
    bw = 0.8 / max(1, len(bars))
    for j, a in enumerate(bars):
        if base:
            gap = [T[a][f"bce_{f}"][0] - base[f"bce_{f}"][0] for f in fam_cols]
            axs[0].bar(x + (j - (len(bars) - 1) / 2) * bw, gap, bw, color=LIGHT[2 * j], label=ARM[a])
    axs[0].axhline(0, color="#654321", lw=0.8); axs[0].set_xticks(x); axs[0].set_xticklabels(fam_cols, fontsize=8)
    axs[0].set_ylabel("held-out BCE minus arm A"); axs[0].set_title("price of anarchy by family (mean of seeds)", fontsize=9)
    for a, c in (("A", "#123456"), ("B", LIGHT[0]), ("C", LIGHT[2]), ("R", LIGHT[4]), ("W", LIGHT[6])):
        for f in sorted(glob.glob(os.path.join(res, "blocklocal", f"{a}-s*-log.json"))):
            lg = json.load(open(f))
            lg = lg["log"] if isinstance(lg, dict) else lg
            axs[1].plot([e["step"] for e in lg], [e["loss"] for e in lg], color=c, lw=0.8, alpha=0.8,
                        label=ARM[a] if f.endswith("s1-log.json") else None)
    axs[1].set_xlabel("step"); axs[1].set_ylabel("training BCE (final readout, one batch)"); axs[1].set_title("training", fontsize=9)
    h, l = [], []
    for ax in axs:
        hh, ll = ax.get_legend_handles_labels(); h += hh; l += ll
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=7)
    f1 = svg(fig)

    def cell(a, k, fmt="{:.4f}"):
        if k not in T[a]:
            return "<td>&ndash;</td>"
        m, s = T[a][k]
        return f"<td>{fmt.format(m)}<span style=\"opacity:.6;font-size:85%\"> &plusmn;{fmt.format(s).lstrip('+')}</span></td>"

    cols = [("steps", "steps", "{:.0f}"), ("bce", "held-out BCE", "{:.4f}"), ("gap", "price of anarchy (BCE &minus; A)", "{:+.4f}"),
            ("spearman_settlement_oracle", "Spearman settlement vs oracle", "{:+.3f}"),
            ("spearman_final_price_oracle", "same, at the final price", "{:+.3f}"),
            ("spearman_wealth_oracle", "Spearman wealth vs oracle", "{:+.3f}"),
            ("gini_settlement", "Gini of settlement", "{:.3f}"), ("gini_wealth", "Gini of wealth", "{:.3f}"),
            ("share_negative_settlement", "share negative settlement", "{:.2f}"), ("alive", "units alive", "{:.0f}"), ("births", "births", "{:.0f}"),
            ("max_single_damage", "max single-unit damage", "{:.4f}"),
            ("top1pct_removed_bce_increase", "remove top 1% (5 units)", "{:+.4f}"),
            ("random1pct_removed_bce_increase", "remove random 1%", "{:+.4f}")]
    for a in T:
        if base:
            T[a]["gap"] = (T[a]["bce"][0] - base["bce"][0], 0.0)
    trs = ""
    for k, lab, fmt in cols:
        trs += f"<tr><td>{lab}</td>" + "".join(cell(a, k, fmt) for a in T) + "</tr>"
    frs = "".join(f"<tr><td>{f}</td>" + "".join(cell(a, f"bce_{f}") for a in T)
                  + "".join(f"<td>{T[a][f'bce_{f}'][0] - base[f'bce_{f}'][0]:+.4f}</td>" for a in T if a != "A") + "</tr>" for f in fam_cols) if base else ""
    return (f"<h2>Block-local settlement</h2><p>Traders as blocks. Each MLP block has its own loss: the cross-entropy of the logit-lens prediction after that block. "
            f"The residual entering a block is detached for that loss, so no gradient crosses a block boundary; the block still sees upstream activations as input. "
            f"Arm A is ordinary end-to-end training of the same network. Arm C adds a market inside each block: every neuron's output is scaled by N&middot;w<sub>i</sub>, "
            f"where w is updated each step by w<sub>i</sub> &larr; w<sub>i</sub>&middot;exp(&eta;&middot;s<sub>i</sub>) from its leave-one-out settlement s<sub>i</sub> "
            f"at that block's own price (the logit lens after the block), clipped to &plusmn;0.5 per step, bounded to [0.001, 8]/N and renormalised per block; "
            f"neurons with N&middot;w below 0.1 are skipped, and every 200 steps each skipped neuron is reborn as a noisy copy of a wealth-sampled parent that splits its wealth with it. "
            f"Two controls for C: <b>C-random</b> skips and rebirths the same number of neurons per step as C did (read from C's log for that seed), "
            f"but picks them uniformly at random and has no wealth scaling; <b>C-wealth only</b> keeps the wealth scaling but never skips or rebirths. "
            f"Same streams, model and sampler as above; 3 seeds per arm on one L4 GPU. Settlement is measured on 4096 training positions at the arm's own price "
            f"(final readout for A, block lens for B and C); the oracle is the single-neuron ablation on the held-out segment. Mean &plusmn; std over seeds. "
            f"Data: <code>results/blocklocal/</code>.</p>"
            f"<div class=tw><table><tr><th>measure</th>" + "".join(f"<th>{ARM[a]}</th>" for a in T) + f"</tr>{trs}</table></div>"
            f"<p>Held-out BCE by family.</p><div class=tw><table><tr><th>family</th>" + "".join(f"<th>{ARM[a]}</th>" for a in T)
            + "".join(f"<th>{a} &minus; A</th>" for a in T if a != "A") + f"</tr>{frs}</table></div>{f1}" + OBS + leak(res, svg, LIGHT, R, T))


OBS = ("<h3>Observations</h3><ol>"
       "<li><b>The price of anarchy is negative here, and it is regularisation.</b> On the learnable families (periodic, lfsr16, ca110) "
       "all arms are within 0.02. The gain is on families nobody can predict (pi, ca30, prime, and, or), where end-to-end A memorises "
       "training windows and is confidently wrong on fresh data (pi 5.4 nats against 0.69 for a coin flip). BCE rises steadily as the "
       "boundary leaks (table below).</li>"
       "<li><b>Block-local prices make settlement honest.</b> Settlement at the block's own lens matches the ablation oracle at Spearman "
       "+0.86 in B, against +0.40 for A. The market arms lose this because their units become nearly interchangeable.</li>"
       "<li><b>Settlement adds little beyond dropout-like skipping.</b> C-random (same skip and rebirth rate, units chosen at random, no wealth) "
       "gets most of C's held-out gain (0.335 against 0.299; the gap is almost all pi) and is as robust to losing its top 1% in two seeds "
       "of three (per seed, C 0.007 / 0.010 / 0.023, C-random 0.263 / 0.020 / 0.0003). Wealth scaling without skipping regularises "
       "(0.306) but keeps strong over-reliance (top 1% removal 0.67 to 0.93). Three seeds cannot separate C from C-random.</li></ol>")


def leak(res, svg, LIGHT, R, T):
    """Leaky-boundary sweep: arm B with a fraction alpha of the gradient crossing each boundary."""
    Lr = [r for f in sorted(glob.glob(os.path.join(res, "blocklocal", "L*-s*.csv"))) for r in csv.DictReader(open(f))]
    if not Lr or "A" not in T or "B" not in T:
        return ""
    for r in Lr:
        r["alpha"] = float(r["arm"][1:])
    for r in R:
        if r["arm"] == "B":
            Lr.append(dict(r, alpha=0.0))
    al = sorted({r["alpha"] for r in Lr})
    fam_cols = [f for f in FAMILIES if f"bce_{f}" in R[0]]
    A = T["A"]

    def m(a, k):
        v = [float(r[k]) for r in Lr if r["alpha"] == a and r.get(k) not in ("", None)]
        return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float("nan"), 0.0, 0)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    cols = plt.cm.tab10.colors
    for i, f in enumerate(fam_cols):
        axs[0].plot(al, [m(a, f"bce_{f}")[0] - A[f"bce_{f}"][0] for a in al], marker="o", ms=3, color=cols[i % 10], label=f)
    axs[0].plot(al, [m(a, "bce")[0] - A["bce"][0] for a in al], marker="o", color="#123456", lw=2.2, label="all families")
    axs[0].axhline(0, color="#654321", lw=0.8)
    axs[0].set_xlabel("alpha (share of gradient crossing each boundary)"); axs[0].set_ylabel("held-out BCE minus arm A")
    axs[0].set_title("price of anarchy vs leak", fontsize=9)
    for k, lab, ls in (("spearman_settlement_oracle", "Spearman settlement vs oracle", "-"), ("share_negative_settlement", "share negative settlement", ":"),
                       ("gini_settlement", "Gini of settlement", "--")):
        axs[1].plot(al, [m(a, k)[0] for a in al], marker="s", ms=3, color="#444444", ls=ls, label=lab)
    axs[1].set_xlabel("alpha"); axs[1].set_title("ecology vs leak", fontsize=9)
    h0, l0 = axs[0].get_legend_handles_labels(); h1, l1 = axs[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.18, 1, 1)); fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=5, fontsize=7)
    f2 = svg(fig)
    ks = [("n", "seeds", None), ("bce", "held-out BCE", "{:.4f}"), ("gap", "BCE &minus; A", "{:+.4f}"),
          ("spearman_settlement_oracle", "Spearman settlement vs oracle", "{:+.3f}"), ("spearman_final_price_oracle", "same, final price", "{:+.3f}"),
          ("gini_settlement", "Gini of settlement", "{:.3f}"), ("share_negative_settlement", "share negative", "{:.2f}"),
          ("max_single_damage", "max single-unit damage", "{:.4f}"), ("top1pct_removed_bce_increase", "remove top 1%", "{:+.4f}"),
          ("random1pct_removed_bce_increase", "remove random 1%", "{:+.4f}")]
    trs = ""
    for a in al:
        cells = ""
        for k, _, fmt in ks:
            if k == "n":
                cells += f"<td>{m(a, 'bce')[2]}</td>"
            elif k == "gap":
                cells += f"<td>{m(a, 'bce')[0] - A['bce'][0]:+.4f}</td>"
            else:
                mu, sd, _ = m(a, k)
                cells += f"<td>{fmt.format(mu)}<span style=\"opacity:.6;font-size:85%\"> &plusmn;{fmt.format(sd).lstrip('+')}</span></td>"
        trs += f"<tr><td>{a:g}{' (= B)' if a == 0 else ''}</td>{cells}</tr>"
    frs = "".join(f"<tr><td>{f}</td>" + "".join(f"<td>{m(a, f'bce_{f}')[0] - A[f'bce_{f}'][0]:+.4f}</td>" for a in al) + "</tr>" for f in fam_cols)
    return (f"<h3>Leaky boundary</h3><p>Arm B again, but a fraction &alpha; of the downstream gradient crosses each block boundary "
            f"(the residual entering a block is x.detach() + &alpha;(x &minus; x.detach()), so the forward pass is unchanged and the backward pass is scaled by &alpha;). "
            f"&alpha; = 0 is arm B (the same runs); &alpha; = 1 is end-to-end training on the sum of the block losses, which differs from arm A, "
            f"whose only loss is the final readout. Arm A is the reference for every gap. Mean &plusmn; std over seeds.</p>"
            f"<div class=tw><table><tr><th>&alpha;</th>" + "".join(f"<th>{lab}</th>" for _, lab, _ in ks) + f"</tr>{trs}</table></div>"
            f"<p>Held-out BCE minus arm A, by family.</p><div class=tw><table><tr><th>family</th>" + "".join(f"<th>&alpha; = {a:g}</th>" for a in al)
            + f"</tr>{frs}</table></div>{f2}")
