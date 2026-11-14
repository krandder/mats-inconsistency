"""P17 analysis: headline, pre-registered kill rule and secondary test, robustness lines, chart JSON.

  python analyze.py --res results --charts charts
Reads results/per_puzzle.npz + results/choices.json (sim.py). Writes results/summary.json, charts/p17_charts.json.
"""
import argparse, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--res", default="results"); ap.add_argument("--charts", default="charts")
ap.add_argument("--boot", type=int, default=10000)
a = ap.parse_args()
Z = np.load(os.path.join(a.res, "per_puzzle.npz"))
ch = json.load(open(os.path.join(a.res, "choices.json")))
B = ch["budgets"]; PB = ch["primary_b"]; sec = ch["secondary_baseline"]
N = len(Z[f"SR|{PB}"])
rng = np.random.default_rng(2026)
IDX = rng.integers(0, N, (a.boot, N))
NBLK = N // 256

methods = sorted({k.split("|")[0] for k in Z.files if not k.startswith("nfe::")})


def acc(m, b):
    return float(Z[f"{m}|{b}"].mean())


def diff(m1, m2, b):
    d = Z[f"{m1}|{b}"].astype(np.int8) - Z[f"{m2}|{b}"].astype(np.int8)
    bs = d[IDX].mean(1)
    blk = d.reshape(NBLK, -1).mean(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
            "block_wins": int((blk > 0).sum()), "block_ties": int((blk == 0).sum()), "n_blocks": NBLK}


def ci_acc(m, b):
    x = Z[f"{m}|{b}"].astype(np.float64)
    bs = x[IDX].mean(1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


out = {"n_test": N, "primary_b": PB, "secondary_baseline": sec, "acc": {}, "nfe": {}, "comparisons": {}}
for m in methods:
    out["acc"][m] = {b: acc(m, b) for b in B if f"{m}|{b}" in Z.files}
    out["nfe"][m] = {b: {"mean": float(Z[f"nfe::{m}|{b}"].mean()), "max_block_mean":
                         float(Z[f"nfe::{m}|{b}"].reshape(NBLK, -1).mean(1).max())} for b in B if f"nfe::{m}|{b}" in Z.files}
pairs = [("Mchk", "SR"), ("Mchk", "MV"), ("Mchk", sec), ("Mchk", "SEQchk"), ("Mchk", "SRchk"),
         ("Mchk", "SR_hindsight"), ("Mchk", "MV_hindsight"), ("Mchk", "FD"), ("Mchk", "FR"), ("Mchk", "RS"),
         ("Mchk", "Mchk_single"), ("Mchk", "Mchk_random"), ("Mchk", "Mchk_local"),
         ("Mres", "SR"), ("Mres", "MV"), ("Mres", "RT"), ("RT", "SR"), ("SEQchk", "SR"), ("SEQchk", "MV")]
for b in B:
    out["comparisons"][b] = {f"{x}-{y}": diff(x, y, b) for x, y in pairs}

p = out["comparisons"][PB]
kill_pass = p["Mchk-SR"]["lo"] > 0 and p["Mchk-MV"]["lo"] > 0
sec_pass = p[f"Mchk-{sec}"]["lo"] > 0
status = "No gain" if not kill_pass else ("Market gain" if sec_pass else "Checker gain, not a market gain")
# equal-NFE fairness: no method above its budget (pooled: per block mean <= b)
fair = {m: all(out["nfe"][m][b]["max_block_mean"] <= b + 1e-9 for b in out["nfe"][m]) for m in methods}
out.update(kill_rule_pass=kill_pass, secondary_pass=sec_pass, status=status, equal_nfe_ok=fair)
json.dump(out, open(os.path.join(a.res, "summary.json"), "w"), indent=1)

os.makedirs(a.charts, exist_ok=True)
show = ["Mchk", "Mres", "SEQchk", "SRchk", "RT", "SR", "MV", "FD", "FR", "RS", "Mchk_local", "Mchk_random", "oracle"]
charts = {"acc_vs_budget": {}, "diff_vs_budget": {}, "nfe_used_vs_budget": {}}
for m in show:
    if m not in methods:
        continue
    r = [ci_acc(m, b) for b in B]
    charts["acc_vs_budget"][m] = {"b": B, "mean": [x[0] for x in r], "lo": [x[1] for x in r], "hi": [x[2] for x in r]}
    charts["nfe_used_vs_budget"][m] = {"b": B, "mean": [out["nfe"][m][b]["mean"] for b in B]}
for key in ["Mchk-SR", "Mchk-MV", f"Mchk-{sec}", "Mres-SR", "Mres-MV", "Mchk-Mchk_random", "Mchk-Mchk_local"]:
    charts["diff_vs_budget"][key] = {"b": B, "mean": [out["comparisons"][b][key]["diff"] for b in B],
                                     "lo": [out["comparisons"][b][key]["lo"] for b in B],
                                     "hi": [out["comparisons"][b][key]["hi"] for b in B]}
charts["meta"] = {"n_test_puzzles": N, "primary_b": PB, "status": status,
                  "note": "accuracy on 2,048 Sudoku-Extreme test puzzles vs average NFE budget per puzzle; "
                          "95% paired bootstrap CIs over puzzles"}
json.dump(charts, open(os.path.join(a.charts, "p17_charts.json"), "w"), indent=1)

print(f"status: {status}  (kill rule pass={kill_pass}, secondary pass={sec_pass}, equal-NFE ok={all(fair.values())})")
print(f"{'method':14s}" + "".join(f"{b:>9d}" for b in B))
for m in show:
    if m in methods:
        print(f"{m:14s}" + "".join(f"{100 * out['acc'][m][b]:9.2f}" for b in B))
for key in ["Mchk-SR", "Mchk-MV", f"Mchk-{sec}", "Mres-SR", "Mres-MV", "Mchk-Mchk_random", "Mchk-Mchk_single", "Mchk-Mchk_local"]:
    print(f"{key:18s}" + "  ".join(f"b{b}:{100 * out['comparisons'][b][key]['diff']:+.2f}[{100 * out['comparisons'][b][key]['lo']:+.2f},{100 * out['comparisons'][b][key]['hi']:+.2f}]" for b in B))
