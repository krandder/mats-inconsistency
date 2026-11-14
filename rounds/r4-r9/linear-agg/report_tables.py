"""Markdown tables for RESULTS-LA.md from runs/a-eval/summary.json and runs/b-eval/summary.json (no hand-copied numbers).
Writes runs/report-tables.md.   python3 report_tables.py"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = 4.302652729911275
SIG = (0.1, 0.5, 1.0, 2.0, 3.0)


def ci(x):
    x = np.asarray(x, float)
    h = T * x.std(ddof=1) / math.sqrt(len(x))
    return x.mean(), x.mean() - h, x.mean() + h


def f3(t, d=4):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def main():
    out = []
    a = json.load(open(os.path.join(HERE, "runs", "a-eval", "summary.json")))
    mse = a["mse"]
    s3 = lambda arm, k: [mse[f"{arm}-s{s}|{k}"] for s in (0, 1, 2)]
    p3 = lambda kind, k: [mse[f"{kind}|s{s}|{k}"] for s in (0, 1, 2)]
    out.append("### A: test MSE of the query (3-seed mean [95% t-CI]; pooled = 50,000 episodes, sigma ~ U(0.1, 3))\n")
    out.append("| arm | params | pooled MSE | adjusted (MSE - oracle) | " + " | ".join(f"sigma {s}" for s in SIG) + " |")
    out.append("|---|---|---|---|" + "---|" * len(SIG))
    orp = mse["oracle|pooled"]
    out.append(f"| oracle (exact posterior; reference) | - | {orp:.4f} | 0 | " + " | ".join(f"{mse[f'oracle|s{s}']:.4f}" for s in SIG) + " |")
    out.append(f"| AdaRR (closed form) | - | {mse['adarr|pooled']:.4f} | {mse['adarr|pooled'] - orp:+.4f} | "
               + " | ".join(f"{mse[f'adarr|s{s}']:.4f}" for s in SIG) + " |")
    out.append(f"| prior mean 0 | - | {mse['prior0|pooled']:.4f} | {mse['prior0|pooled'] - orp:+.4f} | "
               + " | ".join(f"{mse[f'prior0|s{s}']:.4f}" for s in SIG) + " |")
    for kind, nm in (("mix", "mix (1-step)"), ("sw", "sw_mix (1-step)"), ("dd", "dd (1-step Gram dedup)")):
        t = ci(p3(kind, "pooled"))
        out.append(f"| {nm} | 1-3 | {f3(t)} | {t[0] - orp:+.4f} | " + " | ".join(f"{np.mean(p3(kind, f's{s}')):.4f}" for s in SIG) + " |")
    for L in (1, 2, 4, 6):
        for kind, nm in (("gdpp", "GD++"), ("diag", "Diag"), ("full", "Full")):
            arm = f"{kind}-L{L}"
            try:
                t = ci(s3(arm, "pooled"))
            except KeyError:
                continue
            par = a["train"][f"{arm}-s0"]["params"]
            out.append(f"| {nm} L={L} | {par} | {f3(t)} | {t[0] - orp:+.4f} | " + " | ".join(f"{np.mean(s3(arm, f's{s}')):.4f}" for s in SIG) + " |")
    out.append("")
    out.append("### A: training (per model: restarts, best validation MSE, seconds)\n")
    out.append("| model | seed 0 | seed 1 | seed 2 |")
    out.append("|---|---|---|---|")
    for L in (1, 2, 4, 6):
        for kind in ("gdpp", "diag", "full"):
            cells = []
            for s in (0, 1, 2):
                tr = a["train"].get(f"{kind}-L{L}-s{s}")
                cells.append("missing" if tr is None else f"{tr['restarts']} r, {tr['best_val_mse']:.4f}, {tr['sec']:.0f}s")
            out.append(f"| {kind} L={L} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("### A2: mechanism diagnostics (first 5,000 pooled test episodes)\n")
    out.append("| model | kappa(Sigma^L)/kappa(Sigma^0), median, per seed | median kappa per layer (seed 0) | influence ratio bad/clean per seed |")
    out.append("|---|---|---|---|")
    for L in (2, 4, 6):
        for kind in ("gdpp", "diag", "full"):
            try:
                ds = [a["diag"][f"{kind}-L{L}-s{s}"] for s in (0, 1, 2)]
            except KeyError:
                continue
            out.append(f"| {kind} L={L} | " + ", ".join(f"{d['kappa_ratio_median']:.3g}" for d in ds) + " | "
                       + ", ".join(f"{k:.3g}" for k in ds[0]["kappa_median_per_layer"]) + " | "
                       + ", ".join(f"{d['influence_ratio_bad_clean']:.3f}" for d in ds) + " |")
    out.append(f"| AdaRR | - | - | {a['diag']['influence_ratio|adarr']:.3f} |")
    out.append(f"| oracle | - | - | {a['diag']['influence_ratio|oracle']:.3f} |")
    out.append("")
    out.append("Learned Diag scalars (omega_xx, omega_xy, omega_yx, omega_yy) per layer:\n")
    for L in (4, 6):
        for s in (0, 1, 2):
            om = a["diag"].get(f"diag-L{L}-s{s}", {}).get("omegas_xx_xy_yx_yy")
            if om:
                out.append(f"- Diag L={L} seed {s}: " + "; ".join("(" + ", ".join(f"{x:+.3g}" for x in row) + ")" for row in om))
    out.append("")
    out.append("### A3: probe selection on the pooled test set (50,000 episodes)\n")
    out.append("| selector | VoI efficiency (picked pair / best pair) | noisy-TV share of picks | both bridges picked |")
    out.append("|---|---|---|---|")
    for nm in ("exact_voi", "raw_cosine", "uncertainty", "random"):
        r = a["select"][nm]
        out.append(f"| {nm.replace('_', ' ')} | {r['eff']:.3f} | {r['tv_share']:.3f} | {r['both_bridges']:.3f} |")
    for L in (1, 2, 4, 6):
        for kind in ("gdpp", "diag", "full"):
            try:
                rs = [a["select"][f"{kind}-L{L}-s{s}"] for s in (0, 1, 2)]
            except KeyError:
                continue
            out.append(f"| {kind} L={L} score <x_q^L, x_z^L>^2 (3 seeds) | " + ", ".join(f"{r['eff']:.3f}" for r in rs) + " | "
                       + ", ".join(f"{r['tv_share']:.3f}" for r in rs) + " | " + ", ".join(f"{r['both_bridges']:.3f}" for r in rs) + " |")
    out.append("")
    b = json.load(open(os.path.join(HERE, "runs", "b-eval", "summary.json")))
    out.append("### B: test KL(P* || pool) on round 3's 500 test instances per family (3-seed mean [95% t-CI])\n")
    out.append("| arm | B1 | B2 |")
    out.append("|---|---|---|")
    names = {"P0": "one pass P0", "mix": "mix (same rollouts)", "sw": "sw_mix (same rollouts)", "dd": "dd (same rollouts)",
             "mix_tau1": "mix of the 4 tau=1 rollouts", "LT4": "LT-4 (primary)", "LT1": "LT-1 (1 layer)",
             "LT4_noconf": "LT-4 without the conflict feature", "rep_P0": "replication: one pass",
             "rep_mix": "replication: R3 mix (tau 1)", "rep_sw": "replication: R3 sw_mix", "rep_dd": "replication: R3 dd"}
    for arm, nm in names.items():
        out.append(f"| {nm} | " + " | ".join(f3(b["kl_3seed"][f"{arm}|{f}"]) for f in ("B1", "B2")) + " |")
    out.append("| published round 3 (rand, M = 16): mix / sw_mix / dd | "
               + " | ".join(" / ".join(f"{b['published_r3'][f][p]:.4f}" for p in ("mix", "sw", "dd")) for f in ("B1", "B2")) + " |")
    out.append("")
    out.append("LT training (per seed: epochs run, restarts, best validation log loss, parameters):\n")
    for s, r in b["per_seed"].items():
        out.append(f"- seed {s}: " + "; ".join(f"{k} {v['epochs_run']} ep, {v['restarts']} r, {v['best_val_logloss']:.4f}, {v['params']} p"
                                               for k, v in r["lt"].items())
                   + f"; sw fit lam {r['pools']['sw']['lam']:.3g}, eta {r['pools']['sw']['eta']}; dd u {r['pools']['dd_u']:.3g}")
    open(os.path.join(HERE, "runs", "report-tables.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()


def curves():
    import csv
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
    for ax, L in zip(axs, (1, 2, 4, 6)):
        for kind, col in (("gdpp", "#c0392b"), ("diag", "#2471a3"), ("full", "#1e8449")):
            for s in (0, 1, 2):
                f = os.path.join(HERE, "runs", "a", f"{kind}-L{L}-s{s}", "train-history.csv")
                if not os.path.exists(f):
                    continue
                r = list(csv.DictReader(open(f)))
                st = [int(x["step"]) for x in r]
                v = [min(float(x["val_mse"]), 2.0) if x["val_mse"] not in ("nan", "inf") else 2.0 for x in r]
                ax.plot(st, v, color=col, lw=1, alpha=0.8, label=kind.replace("gdpp", "GD++") if s == 0 else None)
        ax.set_title(f"L = {L}")
        ax.set_xlabel("training step")
        ax.set_ylim(0.6, 1.1)
    axs[0].set_ylabel("validation MSE (capped at 2 for display)")
    axs[-1].legend(fontsize=8)
    fig.suptitle("Experiment A training curves (3 seeds each)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "a-eval", "a-train-curves.png"), dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for f in sorted(glob.glob(os.path.join(HERE, "runs", "b-eval", "train-*-s*.csv"))):
        nm = os.path.basename(f)[6:-4]
        r = list(csv.DictReader(open(f)))
        col = {"LT4": "#2471a3", "LT1": "#c0392b", "LT4_noconf": "#1e8449"}[nm.rsplit("-s", 1)[0]]
        ax.plot([int(x["epoch"]) for x in r], [min(float(x["val_logloss"]), 1.0) for x in r], color=col, lw=1,
                label=nm.rsplit("-s", 1)[0] if nm.endswith("s0") else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation outcome log loss")
    ax.legend(fontsize=8)
    ax.set_title("Experiment B aggregator training (3 seeds each)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "runs", "b-eval", "b-train-curves.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    curves()
