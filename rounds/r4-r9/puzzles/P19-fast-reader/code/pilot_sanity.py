"""P19 pilot sanity on seed 99 (run AFTER the freeze; disclosed; seed 99 is never a test seed).
python pilot_sanity.py PILOT_RES_DIR -> pilot_sanity.json: truth A/A, XOR/noise truths, and pilot-only taus
(FR at LR* with the pilot's 2 orders, FO, and single-replicate R1/R4/R7b)."""
import json
import os
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr

R = sys.argv[1]
J = lambda n: json.load(open(os.path.join(R, n + ".json")))
tr = [J(f"truth_maze_s99_r{r}") for r in (0, 1)]
pl = J("pilotlr_maze_s99")
pr = J("pricerep_maze_s99_r0") if os.path.exists(os.path.join(R, "pricerep_maze_s99_r0.json")) else None
w7 = J("r7brep_maze_s99_r0")
its = list(tr[0]["items"].keys())
reps = [[t["items"][i]["held"]["gain_vs_base"] for i in its] for t in tr]
truth = np.mean(reps, 0)
lr = str(json.load(open(os.path.join(R, "pilot_rule.json")))["lr_star"])
rules = dict(FR=[float(np.mean(pl["grid"][lr]["prices"][i])) for i in its], FO=[pl["fo"][i]["FO"] for i in its],
             FO1=[pl["fo"][i]["FO1"] for i in its], R7b_1rep=[w7["res"][i][0] for i in its])
if pr:
    M = pr["members"]
    for i in its:
        c, z = i.split(":")
        LY, LZ, LYZ = M[f"Y|{c}"]["bits"], M[f"Z|{c}:{z}"]["bits"], M[f"YZ|{c}:{z}"]["bits"]
        rules.setdefault("R1_1rep", []).append(LY - LYZ)
        rules.setdefault("R4_1rep", []).append(LY - LYZ - LZ)
tau = lambda a, b: float(kendalltau(a, b, variant="b").statistic)
keys = ["x2", "x4", "x6", "xb4", "pg4", "fg4", "g2", "g4", "g6", "nz05", "nz20"]
out = dict(aa_spearman=float(spearmanr(reps[0], reps[1]).statistic), aa_tau=tau(reps[0], reps[1]),
           truth={i: round(float(truth[j]), 1) for j, i in enumerate(its) if i.split(":")[1] in keys},
           rule_values={k: {i: round(float(v[j]), 1) for j, i in enumerate(its) if i.split(":")[1] in keys}
                        for k, v in rules.items()},
           pilot_taus={k: round(tau(v, truth), 3) for k, v in rules.items()},
           pilot_taus_forward={k: round(tau([v[j] for j, i in enumerate(its) if i.split(":")[1] not in ("nz05", "nz20")],
                                            [truth[j] for j, i in enumerate(its) if i.split(":")[1] not in ("nz05", "nz20")]), 3)
                               for k, v in rules.items()})
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(R, "pilot_sanity.json"), "w"), indent=1)
