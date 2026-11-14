"""P17 gates, run on the VM after group 0 of val blocks 0-1 (512 puzzles x 2 restarts, depth 1024). Exit 0 = pass.

G1 reproduction: single-restart accuracy (mean over restarts 0 and 1) at depth 16 within 86.4 +- 5 and at depth 64
   within 93.0 +- 5 (EqR paper Table 4, Sudoku-Extreme: D16B1 86.4, D64B1 93.0).
G2 checker soundness: no (row, step) whose argmax grid is a valid completion (0 violations) but differs from the label.
   Validity is what the market pays for; if valid != correct, the premise is broken.
G3 cost: the measured throughput projects the full bank (12 blocks x 256 puzzles x 8192 NFE) at the $0.71/h
   ceiling, plus 3 VM startups, under $20 (task cap $30, watchdog $27).
"""
import argparse, glob, json, os, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--bank", required=True); ap.add_argument("--log", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--quick", type=int, default=0, help="PREREG amendment 1: depth-64 pre-check of G1/G2 only; G3 is checked by the full gate")
a = ap.parse_args()

corr, viol = [], []
for b in (0, 1):
    z = np.load(os.path.join(a.bank, f"b{b:02d}_g0.npz"))
    corr.append(z["corr"]); viol.append(z["viol"])
corr, viol = np.concatenate(corr), np.concatenate(viol)          # (P, 2, D)
D = corr.shape[-1]
acc16 = float(corr[:, :, 15].mean()) * 100
acc64 = float(corr[:, :, 63].mean()) * 100
acc_last = float(corr[:, :, -1].mean()) * 100
vnc = int(((viol == 0) & ~corr).sum())
valid_steps = int((viol == 0).sum())

rates = []
for line in open(a.log):
    line = line.strip()
    if line.startswith("{"):
        try:
            rates.append(json.loads(line)["row_nfe_per_s"])
        except Exception:
            pass
rate = min(rates) if rates else 0.0
total = 12 * 256 * 8192
gpu_h = total / rate / 3600 if rate else float("inf")
proj = gpu_h * 0.71 + 3 * 0.35 * 0.71

g = {
    "G1_acc_d16": {"value": round(acc16, 2), "band": [81.4, 91.4], "pass": 81.4 <= acc16 <= 91.4},
    "G1_acc_d64": {"value": round(acc64, 2), "band": [88.0, 98.0], "pass": 88.0 <= acc64 <= 98.0},
    "G2_valid_not_correct": {"value": vnc, "valid_row_steps": valid_steps, "pass": vnc == 0},
    "G3_projected_usd": {"value": round(proj, 2), "row_nfe_per_s_min": rate, "gpu_h": round(gpu_h, 2),
                         "limit": 20.0, "pass": proj < 20.0},
    "info": {"P": int(corr.shape[0]), "depth": int(D), "acc_last_pct": round(acc_last, 2)},
}
if a.quick:
    g["G3_projected_usd"] = {"deferred": "checked by the full gate after group 0 at depth 1024", "pass": True,
                             "row_nfe_per_s_min_quick": rate}
ok = all(v["pass"] for k, v in g.items() if k.startswith("G"))
g["PASS"] = ok
json.dump(g, open(a.out, "w"), indent=1)
print(json.dumps(g))
sys.exit(0 if ok else 1)
