#!/usr/bin/env python3
"""R9-E1 audit, check 8: compare the tables in RESULTS-R9-E1.md with analysis/results.json at the printed precision, and print
the source values behind the prose numbers (mapped by hand in the audit). Read-only. Run: python3 check8_tables.py"""
import json
import re

import numpy as np

R9 = "/home/agent/workspaces/eurisko-study/li-net/r9e1"
RES = f"{R9}/results/main"
an = json.load(open(f"{RES}/analysis/results.json"))
st, V = an["stats"], an["verdicts"]
S = {s: json.load(open(f"{RES}/s{s}/summary.json")) for s in (101, 102, 103)}
T = {s: S[s]["test"] for s in S}
txt = open(f"{R9}/RESULTS-R9-E1.md").read().splitlines()
NAME = {"looped_attn_T3 (anchor)": "looped_attn", "attn_L3-K3 (anchor)": "attn_K3", "jacobi_lin_T4_prec": "jacobi_prec",
        "jacobi_lin_T4 (R7)": "jacobi", "econ_prec_K3": "econ_prec", "econ_kc_K3 (D3 fit)": "econ_D3", "kc_dedup K3": "kc_K3",
        "kc_dedup K1": "kc_K1", "looped_lin_T4 (R6)": "looped_lin", "scaled_mix": "scaled_mix", "ataraxos_mmd": "mmd", "mix": "mix",
        "attn_L3-K1": "attn_K1"}


def table_after(header_prefix, start=0):
    i = next(j for j in range(start, len(txt)) if txt[j].startswith(header_prefix))
    rows = []
    for l in txt[i + 2:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip("|").split("|")])
    return i, txt[i], rows


def fmt(x, nd, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s


bad, n = [], 0
# section 4 table: sparse-root KL by K
i4, hdr, rows = table_after("| pool | K = 0 | 1 | 2 | 4 | 8 | 16 | 24 | 32 |")
for r in rows:
    a = NAME[r[0]]
    for K, cell in zip((0, 1, 2, 4, 8, 16, 24, 32), r[1:]):
        n += 1
        want = fmt(st[f"fs_kl_{a}_K{K}"]["mean"], 4)
        if cell != want:
            bad.append(("sec4", r[0], K, cell, want))
# section 5 tables: Mode A then Mode B, KL(K) - KL(0)
start = 0
for md in ("A", "B"):
    i5, hdr, rows = table_after("| pool | K = 1 | 2 | 4 | 8 | 16 | 24 |", start)
    start = i5 + 1
    for r in rows:
        a = NAME[r[0]]
        for K, cell in zip((1, 2, 4, 8, 16, 24), r[1:]):
            n += 1
            want = fmt(st[f"srS{md}_dkl_{a}_K{K}"]["mean"], 4, sign=True)
            if cell != want:
                bad.append((f"sec5 mode {md}", r[0], K, cell, want))
# section 6 table: W(N), W8/W1, W16/W1 [CI]
i6, hdr, rows = table_after("| pool, tier | W(1) |")
for r in rows:
    nm, t = r[0].rsplit(", tau ", 1)
    a = NAME[nm]
    w = st["weights"]
    for N, cell in zip((1, 2, 4, 8, 16), r[1:6]):
        n += 1
        want = fmt(w[f"{a}_W_t{t}"][str(N)]["mean"], 3)
        if cell != want:
            bad.append(("sec6 W", r[0], N, cell, want))
    n += 2
    want8 = fmt(w[f"{a}_W8/W1_t{t}"]["mean"], 3)
    c16 = w[f"{a}_W16/W1_t{t}"]
    want16 = f"{c16['mean']:.3f} [{c16['lo']:.3f}, {c16['hi']:.3f}]"
    if r[6] != want8:
        bad.append(("sec6 W8/W1", r[0], "", r[6], want8))
    if r[7] != want16:
        bad.append(("sec6 W16/W1", r[0], "", r[7], want16))
print(f"tables: {n} cells compared with analysis/results.json at the printed precision; mismatches: {len(bad)}")
for b in bad:
    print("   MISMATCH", b)

# ---- prose sources (printed for the hand mapping)
m = lambda k: st[k]["mean"]
pr = {}
pr["TLDR mode B dKL K1 mix, kc_K1"] = (m("srSB_dkl_mix_K1"), m("srSB_dkl_kc_K1_K1"))
for a in ("jacobi_prec", "econ_prec", "kc_K3", "attn_K1", "mmd", "scaled_mix", "looped_lin"):
    pr[f"mode B dKL {a} K1/K16"] = (m(f"srSB_dkl_{a}_K1"), m(f"srSB_dkl_{a}_K16"))
pr["mode B, pools above root at K24 (3-seed mean)"] = sorted(a for a in NAME.values() if m(f"srSB_dkl_{a}_K24") > 0)
pr["mode B, pools above root at K24 in every seed"] = sorted(
    a for a in NAME.values() if all(T[s][f"srB32_kl_{a}_K24"] > T[s][f"srB32_kl_{a}_K0"] for s in T))
pr["mode B, first K where mean dKL > 0"] = {a: next((K for K in (1, 2, 4, 8, 16, 24) if m(f"srSB_dkl_{a}_K{K}") > 0), None)
                                           for a in NAME.values()}
pr["mode A, pools above root at any K (3-seed mean)"] = sorted(
    a for a in NAME.values() if any(m(f"srSA_dkl_{a}_K{K}") > 0 for K in (1, 2, 4, 8, 16, 24)))
pr["mode A, pools above root at any K in any seed"] = sorted(
    a for a in NAME.values() if any(T[s][f"srA32_kl_{a}_K{K}"] > T[s][f"srA32_kl_{a}_K0"] for s in T for K in (1, 2, 4, 8, 16, 24)))
pr["mode A dKL K16 learned pools"] = {a: round(m(f"srSA_dkl_{a}_K16"), 4) for a in NAME.values()}
pr["mode A turn back up after K16"] = {a: (round(m(f"srSA_dkl_{a}_K16"), 4), round(m(f"srSA_dkl_{a}_K24"), 4)) for a in ("mmd", "scaled_mix", "looped_lin")}
pr["looped_attn per-seed max increase A/B"] = {md: V["E1-D1.2"]["stats"]["clause3"][md]["looped_attn_max_increase_per_seed"] for md in "AB"}
pr["looped_attn s102 mode B dKL by K"] = {K: round(T[102][f"srB32_kl_looped_attn_K{K}"] - T[102]["srB32_kl_looped_attn_K0"], 6) for K in (1, 2, 4, 8, 16, 24)}
pr["attn_K3 per-seed max increase A/B (reported)"] = {md: V["E1-D1.2"]["stats"]["reported"][md]["attn_K3_max_increase_per_seed"] for md in "AB"}
for M in (18, 30, 34):
    pr[f"M{M} KL0"] = m(f"srA{M}_kl_looped_attn_K0")
    for md in ("A", "B"):
        pr[f"M{M} mode {md} dKL K1/K16/K24"] = {a: [round(m(f"sr{md}{M}_dkl_{a}_K{K}"), 4) for K in (1, 16, 24)]
                                               for a in ("looped_attn", "attn_K3", "mmd", "mix", "jacobi_prec", "econ_prec")}
    pr[f"M{M} mode B per-seed max rise looped_attn / attn_K3"] = {
        a: {s: round(max(T[s][f"srB{M}_kl_{a}_K{K}"] - T[s][f"srB{M}_kl_{a}_K0"] for K in (1, 2, 4, 8, 16, 24)), 5) for s in T}
        for a in ("looped_attn", "attn_K3")}
    pr[f"M{M} mode A below root at K16 and K24 (means) for the 6 named pools"] = all(
        m(f"srA{M}_dkl_{a}_K{K}") < 0 for a in ("looped_attn", "attn_K3", "mmd", "mix", "jacobi_prec", "econ_prec") for K in (16, 24))
pr["strong selection per seed"] = {s: (S[s]["fit"]["strong_selection"]["kl_ref36_root"], S[s]["fit"]["strong_selection"]["kl_root_ref36_reported"]["32"]) for s in S}
pr["sparse root step increases per seed (looped_attn)"] = V["E1-D1.1"]["stats"]["step_increase_per_seed"]
pr["attn_K3 sparse step K24->32 per seed"] = {s: T[s]["fs_kl_attn_K3_K32"] - T[s]["fs_kl_attn_K3_K24"] for s in T}
pr["econ_prec scalars u/a/b"] = {s: (S[s]["fit"]["econ_prec"]["u"], S[s]["fit"]["econ_prec"]["a"], S[s]["fit"]["econ_prec"]["b"]) for s in S}
pr["econ_prec lambda at beta"] = {s: S[s]["fit"]["econ_prec"]["lambda_at_beta"] for s in S}
pr["econ_prec lambda means"] = {t: float(np.mean([S[s]["fit"]["econ_prec"]["lambda_at_beta"][t] for s in S])) for t in ("0.3", "1.5", "2.5")}
pr["econ_D3 lambda"] = {s: S[s]["fit"]["scalars"]["econ_D3_lam"] for s in S}
pr["econ_R7 W16/W1 t0.3 (other one-ridge fit)"] = st["weights"]["econ_R7_W16/W1_t0.3"]["mean"]
pr["per-tier MSE diff econ_D3"] = {t: (round(c["mean"], 5), round(c["lo"], 5), round(c["hi"], 5)) for t, c in V["E1-D2.1"]["stats"]["per_tier_mse_diff"]["econ_D3"].items()}
pr["W16/W1 econ_prec per seed t0.3"] = st["weights"]["econ_prec_W16/W1_t0.3"]["per_seed"]
pr["jacobi_prec fitted a/b and lambda at beta"] = {s: (S[s]["fit"]["jacobi_prec"]["a"], S[s]["fit"]["jacobi_prec"]["b"], S[s]["fit"]["jacobi_prec"]["lambda_mean_at_beta"]) for s in S}
pr["jacobi_prec lambda means"] = {t: float(np.mean([S[s]["fit"]["jacobi_prec"]["lambda_mean_at_beta"][t] for s in S])) for t in ("0.3", "1.5", "2.5")}
pr["jacobi (R7) FS KL24, W16/W1 t0.3"] = (m("fs_kl_jacobi_K24"), st["weights"]["jacobi_W16/W1_t0.3"]["mean"])
pr["FS KL24 looped_attn / attn_K3"] = (m("fs_kl_looped_attn_K24"), m("fs_kl_attn_K3_K24"))
pr["jacobi_prec FS KL24 per seed"] = st["fs_kl_jacobi_prec_K24"]["per_seed"]
pr["lower every seed (D2.2)"] = {b: d["lower_every_seed (reported)"] for b, d in V["E1-D2.2"]["stats"]["beats"].items()}
pr["compute sec"] = an["compute"]
pr["clause1 per seed"] = st["clause1_corr_std6"]["per_seed"]
pr["KL0 per seed"] = V["E1-D1.2"]["stats"]["KL0"]["per_seed"]
pr["mode B anchor pools at K16/K24"] = {a: (m(f"srSB_dkl_{a}_K16"), m(f"srSB_dkl_{a}_K24")) for a in ("looped_attn", "attn_K3")}
pr["mode B anchor pools below root at every K (means)"] = all(m(f"srSB_dkl_{a}_K{K}") < 0 for a in ("looped_attn", "attn_K3") for K in (1, 2, 4, 8, 16, 24))
pr["mode B anchor pools below root at every K in every seed"] = {a: all(T[s][f"srB32_kl_{a}_K{K}"] < T[s][f"srB32_kl_{a}_K0"] for s in T for K in (1, 2, 4, 8, 16, 24)) for a in ("looped_attn", "attn_K3")}
for k, v in pr.items():
    print(f"{k}: {v}")
