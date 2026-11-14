#!/usr/bin/env python3
"""Plot a soup run: one PNG per metric group, plus overview.png.

usage: PYTHONPATH=.pylib python3 scripts/plot.py runs/<id>/ [cols...]
       (a path to metrics.csv also works; with cols, one PNG per column)
Groups: calibration (Brier per family), log_score, coherence, shadow
(cumulative profit per reference trader), population (alive, lineages,
genotypes, births, deaths), inequality (active fraction, Gini), wealth,
flows (subsidy, trader pnl, volume, burn),
compute (instructions per epoch and cumulative). Missing values ("nan", the
family has no live question yet) leave gaps. Held-out release, when set in
config.json, is drawn as a dashed line.
"""
import csv, json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ["periodic", "lfsr", "ca30", "ca110", "prime", "pi", "and", "or"]
GROUPS = [
    ("calibration", "mean Brier (ppm, lower is better)", ["brier_ppm"] + [f"brier_{f}" for f in FAMS]),
    ("log_score", "mean log loss (micro-nats)", [f"log_{f}" for f in FAMS]),
    ("coherence", "mean violation (ppm of 1.0)", ["coh_and_ppm", "coh_or_ppm"]),
    ("shadow", "cumulative profit (Q16 ticks)", ["sh_c50", "sh_mom", "sh_pat2", "sh_pat4", "sh_pat8", "sh_arb"]),
    ("population", "cells", ["alive", "lineages", "genotypes", "births", "deaths"]),
    ("inequality", "ppm", ["active_ppm", "gini_ppm"]),
    ("wealth", "Q16 cash", ["wealth"]),
    ("flows", "Q16 cash", ["cum_subsidy", "trader_pnl", "volume", "burned"]),
    ("compute", "instructions", ["ops", "cum_ops"]),
]

def num(s):
    try:
        return float(s)
    except ValueError:
        return math.nan

def load(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r]
    head, data = rows[0], [r for r in rows[1:] if r[0][:1].isdigit()]
    return head, {h: [num(r[i]) for r in data] for i, h in enumerate(head)}

def release(run):
    try:
        v = json.load(open(os.path.join(run, "config.json"))).get("release", "none")
        return float(v) if v not in ("none", None) else None
    except (OSError, ValueError):
        return None

def style(ax, rel):
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    if rel is not None:
        ax.axvline(rel, color="#999", linestyle="--", linewidth=1)

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arg = sys.argv[1]
    run = arg if os.path.isdir(arg) else os.path.dirname(arg)
    path = os.path.join(run, "metrics.csv") if os.path.isdir(arg) else arg
    head, cols = load(path)
    x = cols[head[0]]; rel = release(run)
    out = os.path.join(run, "plots"); os.makedirs(out, exist_ok=True)
    if sys.argv[2:]:
        groups = [(c, c, [c]) for c in sys.argv[2:]]
    else:
        groups = [(n, y, [c for c in cs if c in cols]) for n, y, cs in GROUPS]
        groups = [g for g in groups if g[2]]
    for name, ylabel, cs in groups:
        fig, ax = plt.subplots(figsize=(7, 4))
        for c in cs:
            ax.plot(x, cols[c], linewidth=1.4, label=c)
        ax.set_xlabel(head[0]); ax.set_ylabel(ylabel); ax.set_title(name)
        if len(cs) > 1:
            ax.legend(fontsize=7, ncol=2)
        style(ax, rel)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"{name}.png"), dpi=110); plt.close(fig)
    ncol = 2; nrow = (len(groups) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (name, _, cs) in zip(axes.flat, groups):
        for c in cs:
            ax.plot(x, cols[c], linewidth=1.1, label=c)
        ax.set_title(name, fontsize=9); ax.tick_params(labelsize=7); ax.legend(fontsize=6, ncol=2)
        style(ax, rel)
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "overview.png"), dpi=110); plt.close(fig)
    print(f"wrote {len(groups) + 1} plots to {out}")

if __name__ == "__main__":
    main()
