#!/usr/bin/env python3
"""Exploratory D2/D3/D5 summary (run after the D0 gate failed, on the coordinator's request; NOT rule tests): 3-seed means
and 95% t-intervals of the raw quantities from runs/s<seed>/eval/d{2,3,5}.json and the key diagnostics from d2.npz.
Writes report/exploratory_summary.md and report/exploratory_summary.csv."""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
T975 = 4.302652729911275


def tci(x):
    x = np.asarray(x, float)
    m = float(x.mean())
    h = T975 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - h, m + h


J = {p: [json.load(open(os.path.join(HERE, "runs", f"s{s}", "eval", f"{p}.json"))) for s in SEEDS] for p in ("d2", "d3", "d5")}
Z2 = [np.load(os.path.join(HERE, "runs", f"s{s}", "eval", "d2.npz")) for s in SEEDS]
rows = []


def add(group, name, vals, p=4, pct=False):
    m, lo, hi = tci(vals)
    f = (lambda v: f"{100 * v:.2f}%") if pct else (lambda v: f"{v:.{p}g}" if abs(v) < 1e-2 and v != 0 else f"{v:.{p}f}")
    rows.append((group, name, ", ".join(f(v) for v in vals), f"{f(m)} [{f(lo)}, {f(hi)}]"))


for g in ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3"):
    add("D2", f"policy-sensitivity JSD, {g}", [r[g]["jsd"] for r in J["d2"]], 3)
for g in ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3"):
    add("D2", f"value MSE vs K=128 reference, {g}", [r[g]["mse"] for r in J["d2"]], 3)
for g in ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3"):
    add("D2", f"mean |mu*(pi_Left) - mu*(pi_Right)|, {g}", [float(np.abs(z[f"muL_{g}"] - z[f"muR_{g}"]).mean()) for z in Z2], 3)
add("D2", "key cosine within flank", [float(np.nanmean(z["cos_within_L"])) for z in Z2])
add("D2", "key cosine across flanks", [float(np.nanmean(z["cos_across_L"])) for z in Z2])
add("D2", "flank gap |mean q_L - mean q_R|", [r["flank_gap_mean"] for r in J["d2"]])
add("D3", "Spearman(beta_trust, -KL) overall", [r["spearman_trust"] for r in J["d3"]])
add("D3", "Spearman(-H(q), -KL) overall", [r["spearman_stiff"] for r in J["d3"]])
add("D3", "Spearman(beta_trust, -KL) within entropy strata", [r["strata_trust"] for r in J["d3"]])
add("D3", "Spearman(-H(q), -KL) within entropy strata", [r["strata_stiff"] for r in J["d3"]])
add("D3", "Spearman trust, 30% checkpoint rollouts", [r["ckpt0"]["spearman_trust"] for r in J["d3"]])
add("D3", "Spearman trust, 100% checkpoint rollouts", [r["ckpt1"]["spearman_trust"] for r in J["d3"]])
for w in ("trust", "stiff", "uniform"):
    add("D3", f"delusional-move rate, {w} weights", [r[f"delusional_{w}"] for r in J["d3"]], pct=True)
Z3 = [np.load(os.path.join(HERE, "runs", f"s{s}", "eval", "d3.npz")) for s in SEEDS]
# rollouts that ended in a predicted terminal (flag capture): leaf claim 0.98 one-hot + 0.02/3 -> q_eff 0.99 or 0.01
TERM = [(np.abs(z["qeff"] - 0.99) < 1e-3) | (np.abs(z["qeff"] - 0.01) < 1e-3) for z in Z3]
add("D3", "KL(Q*||q_r), lowest-entropy tenth of rollouts", [float(z["kl"][z["strata"] == 0].mean()) for z in Z3], 3)
add("D3", "KL(Q*||q_r), other nine tenths", [float(z["kl"][z["strata"] != 0].mean()) for z in Z3], 3)
add("D3", "share of rollouts ending in a predicted flag capture", [float(t.mean()) for t in TERM], 3)
add("D3", "KL(Q*||q_r), rollouts ending in a predicted flag capture", [float(z["kl"][t].mean()) for z, t in zip(Z3, TERM)], 3)
add("D3", "KL(Q*||q_r), other rollouts", [float(z["kl"][~t].mean()) for z, t in zip(Z3, TERM)], 3)
for o in ("P0", "Ataraxos-1Step", "Tabular-MCTS"):
    add("D5", f"Unified-LIR effective win vs {o}", [r[o]["effective_win"] for r in J["d5"]], 3)
n_del = {w: int(sum(np.load(os.path.join(HERE, "runs", f"s{s}", "eval", "d3.npz"))[f"delusional_{w}"].sum() for s in SEEDS))
         for w in ("trust", "stiff", "uniform")}
os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
with open(os.path.join(HERE, "report", "exploratory_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["part", "quantity", "per_seed_101_102_103", "mean_95ci"])
    w.writerows(rows)
md = ["| part | quantity (EXPLORATORY, not a rule test) | seeds 101, 102, 103 | mean [95% CI] |", "|---|---|---|---|"]
md += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
md.append(f"\nDelusional choices over the 3 seeds (600 decisions per seed): trust {n_del['trust']}, stiff {n_del['stiff']}, "
          f"uniform {n_del['uniform']}.")
open(os.path.join(HERE, "report", "exploratory_summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
