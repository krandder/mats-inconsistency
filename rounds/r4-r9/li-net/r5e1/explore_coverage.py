#!/usr/bin/env python3
"""EXPLORATORY (post hoc, written after the primary results; not a rule test). Why does E1-D1.1 fail in Part A v2?
Splits the per-cell JSD between the merged beliefs of the U_NW and U_SE bags (same board) by evidence coverage: a cell is
'both' if at least one read in each bag covered it, 'one' if only one bag read it, 'none' if neither did. For 'both' cells it
also bins by how unequal the two bags' read counts are (the skew a deduplicating pool should absorb). Reads only the raw
arrays results/v2/s*/partA_raw.json (offline, CPU). Output: results/v2/analysis/explore_coverage.{json,csv}.
"""
import csv
import json
import math
import os

import numpy as np

R5 = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
POOLS = ("mix", "mmd", "kc", "attn", "union_call")


def patches():
    P = []
    for r in range(5):
        for c in range(4):
            P.append([(r + a) * 6 + c + b for a in range(2) for b in range(3)])
    for r in range(4):
        for c in range(5):
            P.append([(r + a) * 6 + c + b for a in range(3) for b in range(2)])
    return np.array(P)


def ent(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def jsd(p, q):
    return ent(0.5 * (p + q)) - 0.5 * (ent(p) + ent(q))


def kl(P, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    P = np.clip(P, 0, 1)
    a = np.where(P > 0, P * (np.log(np.clip(P, 1e-300, 1)) - np.log(q)), 0.0)
    b = np.where(P < 1, (1 - P) * (np.log(np.clip(1 - P, 1e-300, 1)) - np.log(1 - q)), 0.0)
    return a + b


def main():
    PM = np.zeros((40, 36), dtype=int)
    for i, p in enumerate(patches()):
        PM[i, p] = 1
    out = {}
    rows = []
    for s in SEEDS:
        d = json.load(open(os.path.join(R5, "results", "v2", f"s{s}", "partA_raw.json")))
        Pst = np.array(d["pstar"])
        cnt = {b: PM[np.array(d[f"bag_{b}"]["patch"])].sum(1) for b in ("NW", "SE")}       # [n, 36] reads per cell
        cov_nw, cov_se = cnt["NW"] > 0, cnt["SE"] > 0
        cats = {"both": cov_nw & cov_se, "one": cov_nw ^ cov_se, "none": ~(cov_nw | cov_se)}
        skew = np.abs(cnt["NW"] - cnt["SE"])
        for b_lo, b_hi, name in ((0, 1, "both_skew0"), (1, 3, "both_skew1-2"), (3, 99, "both_skew3+")):
            cats[name] = cats["both"] & (skew >= b_lo) & (skew < b_hi)
        # reads of each cell by corrupted (tau 1.5) reads, per bag: separates read noise from search skew
        cor = {b: (PM[np.array(d[f"bag_{b}"]["patch"])] * (np.array(d[f"bag_{b}"]["tau"]) > 1.0)[..., None]).sum(1)
               for b in ("NW", "SE")}
        cats["both_clean_only"] = cats["both"] & (cor["NW"] == 0) & (cor["SE"] == 0)
        cats["both_some_corrupted"] = cats["both"] & ((cor["NW"] > 0) | (cor["SE"] > 0))
        res = {"cell_share": {k: float(v.mean()) for k, v in cats.items()}}
        for pool in POOLS:
            mN = np.array(d["bag_NW"][pool])
            mS = np.array(d["bag_SE"][pool])
            J = jsd(mN, mS)
            K = 0.5 * (kl(Pst, mN) + kl(Pst, mS))
            res[pool] = {k: {"jsd": float(J[v].mean()), "kl": float(K[v].mean())} for k, v in cats.items() if v.any()}
            res[pool]["all"] = {"jsd": float(J.mean()), "kl": float(K.mean())}
        out[s] = res
        for pool in POOLS:
            for k in list(cats) + ["all"]:
                if k in res[pool]:
                    rows.append([s, pool, k, res[pool][k]["jsd"], res[pool][k]["kl"], res["cell_share"].get(k, 1.0)])
    od = os.path.join(R5, "results", "v2", "analysis")
    json.dump(out, open(os.path.join(od, "explore_coverage.json"), "w"), indent=1)
    with open(os.path.join(od, "explore_coverage.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "pool", "cells", "jsd_NW_vs_SE", "kl_to_bayes", "cell_share"])
        w.writerows(rows)
    keys = ["all", "both", "both_skew0", "both_skew1-2", "both_skew3+", "both_clean_only", "both_some_corrupted", "one", "none"]
    print("3-seed mean JSD (KL) by cell category; share of cells:",
          {k: round(np.mean([out[s]["cell_share"].get(k, 1.0) for s in SEEDS]), 3) for k in keys})
    for pool in POOLS:
        print(f"{pool:11s}", " ".join(f"{k}: {np.mean([out[s][pool][k]['jsd'] for s in SEEDS]):.5f} "
                                      f"({np.mean([out[s][pool][k]['kl'] for s in SEEDS]):.4f})" for k in keys))
    for k in keys:
        r = [out[s]["attn"][k]["jsd"] / out[s]["mix"][k]["jsd"] for s in SEEDS]
        print(f"JSD attn/mix on {k}: {np.mean(r):.3f} (per seed {', '.join(f'{x:.3f}' for x in r)})")


if __name__ == "__main__":
    main()
