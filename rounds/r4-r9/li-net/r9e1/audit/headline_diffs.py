#!/usr/bin/env python3
"""R9-E1 audit, check 4: largest |mine - summary.json| and |mine 3-seed mean - results.json| restricted to the brief's headline
numbers. Reads rederive_out.json (written by rederive.py). Run: python3 headline_diffs.py > headline_diffs.txt"""
import json
import os

o = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rederive_out.json")))
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
head = [f"kl_root_{M}" for M in (2, 6, 12, 18, 30, 32, 34)]
head += [f"fs_kl_looped_attn_K{K}" for K in KS_FS] + ["fs_jsd_looped_attn_K24"]
head += [f"fs_kl_{a}_K24" for a in ("jacobi_prec", "looped_lin", "kc_K1", "kc_K3", "econ_prec")]
head += [f"sr{md}_kl_{a}_K{K}" for md in "AB" for a in ("looped_attn", "attn_K3", "mmd", "mix", "scaled_mix") for K in KS_SR]
head += [f"W_{a}_t{t}_N{N}" for a in ("econ_prec", "jacobi_prec", "econ_D3", "kc_K3", "looped_attn") for t in ("0.3", "1.5", "2.5")
         for N in (1, 2, 4, 8, 16)]
head += [f"mse_{a}_t{t}_N{N}" for a in ("jacobi_prec", "econ_prec", "econ_D3", "kc_K3") for t in ("0.3", "1.5", "2.5")
         for N in (1, 2, 4, 8, 16)]


def cat(k):
    for p in ("kl_root", "fs_kl_looped_attn", "fs_jsd", "fs_kl", "srA", "srB", "W_", "mse_"):
        if k.startswith(p):
            return p


best, overall = {}, (0, None)
for s, d in o["diff_vs_summary"].items():
    for k in head:
        v, c = abs(d[k]), cat(k)
        if v > best.get(c, (0,))[0]:
            best[c] = (v, f"s{s} {k}")
        if v > overall[0]:
            overall = (v, f"s{s} {k}")
print("headline per-seed numbers compared:", 3 * len(head))
print("largest |mine - summary.json| among headline numbers:", overall)
for c, v in best.items():
    print("  ", c, f"{v[0]:.2e}", v[1])
dm = o["diff_vs_results_json"]
hm = [k for k in head if k in dm] + [k for k in dm if k.startswith("mse_")]
w = max((abs(dm[k]), k) for k in hm)
print("largest |mine 3-seed mean - results.json mean| among headline numbers:", f"{w[0]:.2e}", w[1], "n =", len(hm))
m = o["means"]
for k in ("srB_kl_scaled_mix_K24", "srB_kl_mmd_K24", "srB_kl_mmd_K16", "fs_kl_looped_attn_K32", "fs_jsd_looped_attn_K24"):
    print(k, "mine", round(m[k]["mean"], 6), "per seed", [round(x, 6) for x in m[k]["per_seed"]], "diff vs results.json", f"{dm[k]:.2e}")
