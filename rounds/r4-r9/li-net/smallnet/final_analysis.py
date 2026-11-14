"""TEST analysis for the final phase (run once, after the champion was fixed on validation).
Writes results/test-tables-test.md, results/test-decision.json, results/test-curve_kl_vs_{boards,compute}.png,
results/test-summary.json (per-arm per-N mean / CI / paired differences incl. N=30000)."""
import json
import math
import os
import subprocess
import sys

import numpy as np

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
import analyze as A  # noqa: E402

CH, K0, KC = "F-E17-rbtd8", "F-k0", "F-k0cell"
rows = A.load("/tmp/smallnet/runs", "test")
rows = [r for r in rows if r["tag"] in (CH, K0, KC)]
out = {"n_runs": {t: sum(r["tag"] == t for r in rows) for t in (CH, K0, KC)}}
NS = [100, 300, 1000, 3000, 10000, 30000]
tab = {}
for t in (CH, K0, KC):
    tab[t] = {}
    for n in NS:
        v = [r["kl"] for r in rows if r["tag"] == t and r["n"] == n]
        if v:
            m, h = A.tci(v)
            tab[t][n] = {"kl_mean": m, "ci95_half": h, "seeds": len(v),
                         "kl_stage": [float(np.mean([r[f"kl_s{i}"] for r in rows if r["tag"] == t and r["n"] == n])) for i in range(4)],
                         "logloss": float(np.mean([r["logloss"] for r in rows if r["tag"] == t and r["n"] == n])),
                         "ece": float(np.mean([r["ece"] for r in rows if r["tag"] == t and r["n"] == n])),
                         "flops_per_board": float(np.mean([r["flops_per_board"] for r in rows if r["tag"] == t and r["n"] == n])),
                         "flops_at_best": float(np.mean([r["flops_at_best"] for r in rows if r["tag"] == t and r["n"] == n]))}
out["per_arm"] = tab
pair = {}
for base in (K0, KC):
    pair[base] = {}
    for n in NS:
        A_ = {r["seed"]: r["kl"] for r in rows if r["tag"] == CH and r["n"] == n}
        B_ = {r["seed"]: r["kl"] for r in rows if r["tag"] == base and r["n"] == n}
        s = sorted(set(A_) & set(B_))
        if len(s) >= 2:
            d = [A_[k] - B_[k] for k in s]
            m, h = A.tci(d)
            lr = [math.log(A_[k] / B_[k]) for k in s]
            mr, hr = A.tci(lr)
            pair[base][n] = {"diff_mean": m, "diff_ci95_half": h, "ratio_geo": math.exp(mr),
                             "ratio_ci95": [math.exp(mr - hr), math.exp(mr + hr)], "seeds": s}
out["paired_vs"] = pair
dec = {}
for base in (K0, KC):
    r2 = [r for r in rows if r["n"] <= 10000]
    dec[base] = A.decision(r2, CH, base)
out["decision"] = dec
json.dump(out, open(os.path.join(S, "results", "test-summary.json"), "w"), indent=1)
json.dump(dec, open(os.path.join(S, "results", "test-decision.json"), "w"), indent=1)
# markdown tables (N <= 10000 via analyze.md_tables; 30k added below)
md = A.md_tables("/tmp/smallnet/runs", "test", [K0, KC, CH], K0, os.path.join(S, "results", "test-tables-test.md"))
extra = ["", "N = 30000 (3 seeds; the 'eventually more accurate' check), TEST KL x 1e3:", "", "| arm | KL | paired vs F-k0 |", "|---|---|---|"]
for t in (K0, CH):
    if 30000 in tab[t]:
        e = tab[t][30000]
        pv = pair[K0].get(30000) if t == CH else None
        extra.append(f"| {t} | {1000*e['kl_mean']:.3f} +- {1000*e['ci95_half']:.3f} [{e['seeds']}] | " +
                     (f"{1000*pv['diff_mean']:+.3f} +- {1000*pv['diff_ci95_half']:.3f} (x{pv['ratio_geo']:.2f})" if pv else "") + " |")
open(os.path.join(S, "results", "test-tables-test.md"), "a").write("\n".join(extra) + "\n")
A.plot(rows, None, [K0, KC, CH], os.path.join(S, "results"), "test-", "test", label_tags=[K0, KC, CH], title="TEST (5 seeds; 3 at N=30k)")
print(json.dumps({"n_runs": out["n_runs"], "paired_vs_k0": {n: (round(1000 * v["diff_mean"], 3), round(1000 * v["diff_ci95_half"], 3), round(v["ratio_geo"], 3)) for n, v in pair[K0].items()},
                  "decision_vs_k0": {k: dec[K0][k] for k in ("frac_boards_to_reach_base10k", "frac_ci95", "rule_a_pass", "rule_b_pass")},
                  "decision_vs_k0cell": {k: dec[KC][k] for k in ("frac_boards_to_reach_base10k", "frac_ci95", "rule_a_pass", "rule_b_pass")}}, indent=1))
