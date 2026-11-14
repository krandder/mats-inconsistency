#!/usr/bin/env python3
"""R6-E1 audit: are the summary.json KL values consistent with the 5-decimal rounding of the beliefs saved in raw.json?

For every KL statistic that can be recomputed from raw.json, the true belief q of each cell lies in [q_r - 5e-6, q_r + 5e-6]
(then clamped to [1e-6, 1 - 1e-6] like the experiment's metric). KL(p || q) is convex in q with its minimum at q = p, so the
per-cell minimum is at clip(p, lo, hi) and the maximum at an interval end. p = the auditor's own exact Bayes (float64, then
float32 like the experiment), so the target carries no rounding. Prints, per statistic and seed, whether summary.json's value
lies inside [mean of per-cell minima, mean of per-cell maxima].
Run: OMP_NUM_THREADS=2 PYTHONPATH=/tmp/r4e3a/pylib python3 kl_rounding_bounds.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams_bayes import placements, boards, exact_bayes_full  # noqa: E402  (auditor's own code)
from rederive import bern_kl  # noqa: E402  (auditor's own code)

R6 = "/home/agent/workspaces/eurisko-study/li-net/r6e1"
H = 5e-6


def kl_bounds(p, qr):
    lo_q = np.clip(qr - H, 1e-6, 1 - 1e-6)
    hi_q = np.clip(qr + H, 1e-6, 1 - 1e-6)
    qmin = np.clip(p, lo_q, hi_q)
    kmin = bern_kl(p, qmin)
    kmax = np.maximum(bern_kl(p, lo_q), bern_kl(p, hi_q))
    return float(kmin.mean()), float(kmax.mean()), float(bern_kl(p, qr).mean())


def main():
    P = placements()
    T = boards(P, 22, 500)
    p = exact_bayes_full(P, T["s"], T["note"]).astype(np.float32).astype(np.float64)
    n_in, n_all, worst = 0, 0, (0, None)
    for s in (101, 102, 103):
        raw = json.load(open(os.path.join(R6, "results", "main", f"s{s}", "raw.json")))
        summ = json.load(open(os.path.join(R6, "results", "main", f"s{s}", "summary.json")))["test"]
        items = [("kl_p0", [raw["mu0"]])]
        items += [(f"fs_kl_{a}", [raw["fs_NW"][a], raw["fs_SE"][a]]) for a in
                  ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K2", "attn_K3", "attn_K4", "union")]
        items += [(f"ap_{k}_kl_{a}", [raw["ap_NW"][f"{k}_{a}"], raw["ap_SE"][f"{k}_{a}"]]) for k in ("open", "puct")
                  for a in ("attn_K4", "mix", "scaled_mix", "mmd", "kc")]
        for name, arrs in items:
            b = [kl_bounds(p, np.asarray(a, dtype=np.float64)) for a in arrs]
            lo, hi, pt = (float(np.mean([x[i] for x in b])) for i in range(3))
            ref = summ[name]
            inside = lo - 1e-7 <= ref <= hi + 1e-7          # 1e-7 slack for float32 accumulation on the GPU
            n_all += 1
            n_in += inside
            if abs(pt - ref) > worst[0]:
                worst = (abs(pt - ref), f"s{s} {name}")
            if not inside or name.startswith("fs_kl_mmd"):
                print(f"s{s} {name}: summary {ref:.7f}; rounding interval [{lo:.7f}, {hi:.7f}] width {hi - lo:.2e}; "
                      f"point (exact p, rounded q) {pt:.7f}; inside: {inside}")
    print(f"{n_in} of {n_all} recomputable KL statistics lie inside their rounding interval; largest |point - summary| "
          f"{worst[0]:.2e} ({worst[1]})")


if __name__ == "__main__":
    main()
