#!/usr/bin/env python3
"""EXPLORATORY summary (post hoc; NOT a rule test) of the early-stopping rerun r8/runs_es (r8/expl_d1_es.py): the
E3-D1 statistics recomputed for the early-stopped aggregators, with the registered run's numbers next to them.
Writes r8/report/expl_es.json.   python3 r8/expl_es_summary.py"""
import json
import math
import os

import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
T975 = 4.302652729749464
SEEDS = (101, 102, 103)
KS = (0, 1, 2, 4, 8, 16)


def ci(v):
    v = np.asarray(v, float)
    m = float(v.mean())
    h = T975 * v.std(ddof=1) / math.sqrt(len(v))
    return [m, m - h, m + h]


out = {}
for tag, d in (("registered", "runs"), ("early_stopped", "runs_es")):
    D = [json.load(open(os.path.join(H, d, f"s{s}", "eval", "d1.json"))) for s in SEEDS]
    kl = lambda g, K: np.array([x[g][f"K{K}"]["kl"] for x in D])
    js = lambda g, K: np.array([x[g][f"K{K}"]["jsd"] for x in D])
    r = {}
    for g in ("raw_count", "union_t", "scaled_mix", "ataraxos_mmd", "ataraxos_mmd_fit", "kc_dedup_K3c", "attn_L3_K3c",
              "looped_attn_T3_K3c"):
        r[g] = {"kl": {f"K{K}": ci(kl(g, K)) for K in KS}, "jsd_K16": ci(js(g, 16)),
                "max_dK_per_seed": np.max(np.stack([kl(g, K) - kl(g, 0) for K in KS[1:]]), 0).tolist(),
                "kl16_minus_kl0": ci(kl(g, 16) - kl(g, 0))}
    a16 = js("attn_L3_K3c", 16)
    r["attn_vs"] = {X: {"jsd_ratio": ci(js(X, 16) / np.maximum(a16, 1e-30)), "jsd_diff": ci(js(X, 16) - a16),
                        "kl_diff": ci(kl(X, 16) - kl("attn_L3_K3c", 16))}
                    for X in ("scaled_mix", "ataraxos_mmd", "kc_dedup_K3c")}
    r["attn_equals_looped_max_abs"] = float(max(abs(kl("attn_L3_K3c", K) - kl("looped_attn_T3_K3c", K)).max() for K in KS))
    out[tag] = r
F = [json.load(open(os.path.join(H, "runs_es", f"s{s}", "eval", "fit_es.json"))) for s in SEEDS]
out["early_stopping"] = {k: {"best_step": [f[k + "_best_step"] for f in F], "best_val_ce": [f[k + "_best_val_ce"] for f in F]}
                         for k in ("scaled_mix", "mmd_fit", "kc_K3c", "attn_L3_K3c", "looped_attn_T3_K3c")}
out["early_stopping"]["val_ce_anchor"] = [f["val_ce_anchor"] for f in F]
out["early_stopping"]["alpha"] = [f["alpha"] for f in F]
json.dump(out, open(os.path.join(H, "report", "expl_es.json"), "w"), indent=1)
for tag in ("registered", "early_stopped"):
    r = out[tag]
    print(tag)
    for g in ("scaled_mix", "attn_L3_K3c", "looped_attn_T3_K3c", "kc_dedup_K3c"):
        print("  ", g.ljust(20), " ".join(f"{r[g]['kl'][f'K{K}'][0]:.4f}" for K in KS), "max dK", [round(x, 5) for x in r[g]["max_dK_per_seed"]],
              "KL16-KL0 %.5f [%.5f, %.5f]" % tuple(r[g]["kl16_minus_kl0"]))
    for X, v in r["attn_vs"].items():
        print("   attn vs", X, "JSD ratio %.2f [%.2f, %.2f]" % tuple(v["jsd_ratio"]), "KL diff %.5f [%.5f, %.5f]" % tuple(v["kl_diff"]))
    print("   attn == looped (max abs KL diff)", r["attn_equals_looped_max_abs"])
print(out["early_stopping"])
