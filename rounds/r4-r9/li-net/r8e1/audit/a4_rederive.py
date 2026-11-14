#!/usr/bin/env python3
"""R8-E1 audit, check 4: re-derive the headline numbers from raw.json with the auditor's own code.
Imports only numpy/json (no e1v5, e1v4, e1v3, analyze8, wm, env6). Compares with summary.json and analysis/results.json.
Usage: OMP_NUM_THREADS=2 python3 a4_rederive.py  -> prints a report and writes a4_rederive.json next to this script."""
import json
import math
import os
import sys

import numpy as np

R = "/home/agent/workspaces/eurisko-study/li-net/r8e1/results/main"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
TAUS = (0.3, 1.5, 2.5)
NS = (1, 2, 4, 8, 16)
CELLS = ("K1_R6", "K1_R7", "K3_R6", "K3_R7")
# seed-t 97.5% quantile with 2 df in closed form: F(t) = 1/2 + t / (2 sqrt(2 + t^2))  =>  t^2 = 2 q^2 / (1 - q^2), q = 0.95
T2 = math.sqrt(2 * 0.95 ** 2 / (1 - 0.95 ** 2))


def ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = T2 * x.std(ddof=1) / math.sqrt(len(x))
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "per_seed": [float(v) for v in x]}


def bkl(p, q):
    """KL(Bern(p) || Bern(q)) per element; q clipped to [1e-6, 1 - 1e-6] as the experiment's metric does."""
    p = np.clip(np.asarray(p, float), 0, 1)
    q = np.clip(np.asarray(q, float), 1e-6, 1 - 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1)) - np.log(q)), 0.0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1)) - np.log1p(-q)), 0.0)
    return a + b


def ent(p):
    p = np.clip(np.asarray(p, float), 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0.0)
        t2 = np.where(p < 1, (1 - p) * np.log(np.where(p < 1, 1 - p, 1)), 0.0)
    return -(t1 + t2)


def jsd(p, q):
    return ent(0.5 * (p + q)) - 0.5 * (ent(p) + ent(q))


def patches():
    P = []
    for r in range(5):
        for c in range(4):
            P.append([(r + a) * 6 + c + b for a in range(2) for b in range(3)])
    for r in range(4):
        for c in range(5):
            P.append([(r + a) * 6 + c + b for a in range(3) for b in range(2)])
    return np.array(P)


PT = patches()
PM = np.zeros((40, 36), bool)
for i, p in enumerate(PT):
    PM[i, p] = True
NWC = (PT < 18).sum(1)
IN_NW, IN_SE = NWC >= 3, (6 - NWC) >= 3
TILES = (0, 3, 8, 11, 16, 19)

diffs = {}          # family -> list of (key, mine, theirs)


def cmp(fam, key, mine, theirs):
    diffs.setdefault(fam, []).append((key, float(mine), float(theirs), abs(float(mine) - float(theirs))))


mine = {}
for s in SEEDS:
    raw = json.load(open(f"{R}/s{s}/raw.json"))
    T = json.load(open(f"{R}/s{s}/summary.json"))["test"]
    ps = np.asarray(raw["pstar"])
    kl = lambda m: float(bkl(ps, np.asarray(m)).mean())
    M_ = mine.setdefault(s, {})
    # root KL by M_0
    for M in ("2", "6", "12", "18"):
        M_[f"kl_root_{M}"] = kl(raw["root_mu"][M])
        cmp("root KL by M_0", f"s{s} M{M}", M_[f"kl_root_{M}"], T["kl_root"][M])
    # K = 0 identity on the rounded arrays (check 3)
    M_["K0_equal_root_fs"] = all(raw[f"fs_{h}"][f"K0_{a}"] == raw["root_mu"]["2"] for h in ("NW", "SE") for a in ("looped_attn", "attn_K3", "mmd", "scaled_mix"))
    M_["K0_equal_root_sr18"] = all(raw[f"{t}_{h}"][f"K0_{a}"] == raw["root_mu"]["18"] for t in ("sr18", "sr18t3") for h in ("NW", "SE")
                                   for a in ("looped_attn", "attn_K3", "mmd", "mix"))
    # sparse-root curve
    for a in ("looped_attn", "attn_K3", "mmd", "scaled_mix"):
        for K in KS_FS:
            mN, mS = np.asarray(raw["fs_NW"][f"K{K}_{a}"]), np.asarray(raw["fs_SE"][f"K{K}_{a}"])
            M_[f"fs_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
            M_[f"fs_jsd_{a}_K{K}"] = float(jsd(mN, mS).mean())
            cmp("sparse-root FS curve KL", f"s{s} {a} K{K}", M_[f"fs_kl_{a}_K{K}"], T[f"fs_kl_{a}_K{K}"])
            cmp("sparse-root FS curve JSD", f"s{s} {a} K{K}", M_[f"fs_jsd_{a}_K{K}"], T[f"fs_jsd_{a}_K{K}"])
    # FS stream structure (amended reading 7)
    pN, pS = np.asarray(raw["fs_NW"]["patch"]), np.asarray(raw["fs_SE"]["patch"])
    tl = pN[:, 0::4]
    M_["fs_tiles_shared_NW_SE"] = bool((pN[:, 0::4] == pS[:, 0::4]).all())
    M_["fs_first6_tiles_are_partition"] = bool(all(sorted(r[:6].tolist()) == list(TILES) for r in tl))
    M_["fs_slots_24_28_repeat_first2"] = bool((tl[:, 6:8] == tl[:, :2]).all())
    skN = pN[:, [i for i in range(32) if i % 4]]
    skS = pS[:, [i for i in range(32) if i % 4]]
    M_["fs_skewed_reads_in_pref_half_NW"] = float(IN_NW[skN].mean())
    M_["fs_skewed_reads_in_pref_half_SE"] = float(IN_SE[skS].mean())
    M_["fs_first24_cover_all_cells"] = float(PM[pN[:, :24]].any(1).mean())
    # strong root M_0 = 18 (mixed tau = decisive; tier3 reported)
    for tg in ("sr18", "sr18t3"):
        tN, tS = np.asarray(raw[f"{tg}_NW"]["tau"]), np.asarray(raw[f"{tg}_SE"]["tau"])
        vals = np.unique(np.round(np.concatenate([tN.ravel(), tS.ravel()]), 3))
        M_[f"{tg}_tau_values"] = vals.tolist()
        M_[f"{tg}_tau_share"] = {str(v): float((np.round(np.concatenate([tN.ravel(), tS.ravel()]), 3) == v).mean()) for v in vals}
        pN18 = np.asarray(raw[f"{tg}_NW"]["patch"])
        M_[f"{tg}_reads_in_pref_half_NW"] = float(IN_NW[pN18].mean())
        for a in ("looped_attn", "attn_K3", "mmd", "mix"):
            for K in KS_SR:
                mN, mS = np.asarray(raw[f"{tg}_NW"][f"K{K}_{a}"]), np.asarray(raw[f"{tg}_SE"][f"K{K}_{a}"])
                M_[f"{tg}_kl_{a}_K{K}"] = 0.5 * (kl(mN) + kl(mS))
                M_[f"{tg}_jsd_{a}_K{K}"] = float(jsd(mN, mS).mean())
                cmp(f"strong-root {tg} KL", f"s{s} {a} K{K}", M_[f"{tg}_kl_{a}_K{K}"], T[f"{tg}_kl_{a}_K{K}"])
                cmp(f"strong-root {tg} JSD", f"s{s} {a} K{K}", M_[f"{tg}_jsd_{a}_K{K}"], T[f"{tg}_jsd_{a}_K{K}"])
    # D3: W(N) and MSE
    C, CT = raw["controlled"], T["controlled"]
    astar = np.asarray(C["astar"])
    bgp = np.asarray(C["bg_patch"])
    M_["d3_bg_disjoint_from_astar"] = bool((~(PM[bgp] & PM[astar][:, None, :]).any(2)).all())
    for tv in TAUS:
        for N in NS:
            k = f"t{tv}_N{N}"
            for a in ("jacobi_prec", "econ_D3", "econ_R7", "kc_K3", "looped_attn"):
                M_[f"W_{a}_{k}"] = float(np.mean(C[f"{k}_W_{a}"]))
                cmp("D3 W(N)", f"s{s} {a} {k}", M_[f"W_{a}_{k}"], CT[k][f"W_{a}"])
            for a in ("jacobi_prec", "econ_D3", "kc_K3", "attn_K3"):
                M_[f"mse_{a}_{k}"] = float(((np.asarray(C[k][a]) - ps) ** 2).mean())
                cmp("D3 MSE", f"s{s} {a} {k}", M_[f"mse_{a}_{k}"], CT[k][f"mse_{a}"])
    # PUCT 2x2 from the saved patch lists
    for cell in CELLS:
        cv = {}
        for h, half in (("NW", IN_NW), ("SE", IN_SE)):
            p = np.asarray(raw[f"puct_{cell}_{h}"]["patch"])
            assert p.shape[1] == 24
            cv[h] = float(PM[p].any(1).mean(1).mean())
            M_[f"puct_{cell}_coverage_{h}"] = cv[h]
            cmp("PUCT coverage (per preference)", f"s{s} {cell} {h}", cv[h], T[f"puct_{cell}_coverage_{h}"])
            M_[f"puct_{cell}_wave1_pref_share_{h}"] = float(half[p[:, :6]].mean())
            cmp("PUCT wave-1 preferred-half share", f"s{s} {cell} {h}", M_[f"puct_{cell}_wave1_pref_share_{h}"], T[f"puct_{cell}_wave1_pref_share_{h}"])
            M_[f"puct_{cell}_distinct_within_wave_{h}"] = bool(all(len(set(r[6 * w:6 * w + 6].tolist())) == 6 for r in p for w in range(4)))
            M_[f"puct_{cell}_coverage_by_wave_{h}"] = [float(PM[p[:, :6 * (w + 1)]].any(1).mean(1).mean()) for w in range(4)]
        M_[f"puct_{cell}_coverage"] = 0.5 * (cv["NW"] + cv["SE"])
        cmp("PUCT coverage (cell)", f"s{s} {cell}", M_[f"puct_{cell}_coverage"], T[f"puct_{cell}_coverage"])
        mN, mS = np.asarray(raw[f"puct_{cell}_NW"]["attn_K3"]), np.asarray(raw[f"puct_{cell}_SE"]["attn_K3"])
        M_[f"puct_{cell}_kl_attn_K3"] = 0.5 * (kl(mN) + kl(mS))
        M_[f"puct_{cell}_jsd_attn_K3"] = float(jsd(mN, mS).mean())
        cmp("PUCT attn_K3 KL", f"s{s} {cell}", M_[f"puct_{cell}_kl_attn_K3"], T[f"puct_{cell}_kl_attn_K3"])
        cmp("PUCT attn_K3 JSD", f"s{s} {cell}", M_[f"puct_{cell}_jsd_attn_K3"], T[f"puct_{cell}_jsd_attn_K3"])
    M_["anchor_order18"] = raw["anchor_order18"]
    del raw

# anchors identical in every seed
same_anchor = all(mine[s]["anchor_order18"] == mine[101]["anchor_order18"] for s in SEEDS)
for s in SEEDS:
    mine[s].pop("anchor_order18")

# 3-seed statistics vs results.json
res = json.load(open(f"{R}/analysis/results.json"))
st = res["stats"]
agg = {}


def agg_cmp(fam, key, xs, theirs):
    c = ci(xs)
    agg[key] = c
    for f in ("mean", "lo", "hi"):
        cmp(fam, f"{key} {f}", c[f], theirs[f])


for a in ("looped_attn", "attn_K3", "mmd", "scaled_mix"):
    for K in KS_FS:
        agg_cmp("results.json FS (3-seed mean/CI)", f"fs_kl_{a}_K{K}", [mine[s][f"fs_kl_{a}_K{K}"] for s in SEEDS], st[f"fs_kl_{a}_K{K}"])
        agg_cmp("results.json FS (3-seed mean/CI)", f"fs_jsd_{a}_K{K}", [mine[s][f"fs_jsd_{a}_K{K}"] for s in SEEDS], st[f"fs_jsd_{a}_K{K}"])
for tg in ("sr18", "sr18t3"):
    for a in ("looped_attn", "attn_K3", "mmd", "mix"):
        for K in KS_SR:
            agg_cmp("results.json strong root (3-seed mean/CI)", f"{tg}_kl_{a}_K{K}", [mine[s][f"{tg}_kl_{a}_K{K}"] for s in SEEDS], st[f"{tg}_kl_{a}_K{K}"])
            agg_cmp("results.json strong root (3-seed mean/CI)", f"{tg}_dkl_{a}_K{K}",
                    [mine[s][f"{tg}_kl_{a}_K{K}"] - mine[s][f"{tg}_kl_{a}_K0"] for s in SEEDS], st[f"{tg}_dkl_{a}_K{K}"])
for M in ("2", "6", "12", "18"):
    agg_cmp("results.json root KL", f"kl_root_{M}", [mine[s][f"kl_root_{M}"] for s in SEEDS], st["kl_root"][M])
for cell in CELLS:
    agg_cmp("results.json PUCT", f"puct_{cell}_coverage", [mine[s][f"puct_{cell}_coverage"] for s in SEEDS], st[f"puct_{cell}_coverage"])
    agg_cmp("results.json PUCT", f"puct_{cell}_kl_attn_K3", [mine[s][f"puct_{cell}_kl_attn_K3"] for s in SEEDS], st[f"puct_{cell}_kl_attn_K3"])
    agg_cmp("results.json PUCT", f"puct_{cell}_jsd_attn_K3", [mine[s][f"puct_{cell}_jsd_attn_K3"] for s in SEEDS], st[f"puct_{cell}_jsd_attn_K3"])
    agg_cmp("results.json PUCT", f"puct_{cell}_wave1_pref_share_NW", [mine[s][f"puct_{cell}_wave1_pref_share_NW"] for s in SEEDS],
            st[f"puct_{cell}_wave1_pref_share_NW"])
W = st["weights"]
for a in ("jacobi_prec", "econ_D3", "econ_R7", "kc_K3", "looped_attn"):
    for tv in TAUS:
        for N in NS:
            agg_cmp("results.json D3 W(N)", f"W_{a}_t{tv}_N{N}", [mine[s][f"W_{a}_t{tv}_N{N}"] for s in SEEDS], W[f"{a}_W_t{tv}"][str(N)])
        agg_cmp("results.json D3 W ratios", f"{a}_W8/W1_t{tv}", [mine[s][f"W_{a}_t{tv}_N8"] / mine[s][f"W_{a}_t{tv}_N1"] for s in SEEDS],
                W[f"{a}_W8/W1_t{tv}"])
        agg_cmp("results.json D3 W ratios", f"{a}_W16/W1_t{tv}", [mine[s][f"W_{a}_t{tv}_N16"] / mine[s][f"W_{a}_t{tv}_N1"] for s in SEEDS],
                W[f"{a}_W16/W1_t{tv}"])
for a in ("jacobi_prec", "econ_D3", "kc_K3", "attn_K3"):
    for tv in TAUS:
        agg_cmp("results.json D3 MSE", f"mse_{a}_t{tv}", [np.mean([mine[s][f"mse_{a}_t{tv}_N{N}"] for N in NS]) for s in SEEDS], st["mse"][f"{a}_t{tv}"])
    agg_cmp("results.json D3 MSE", f"mse_all_{a}", [np.mean([mine[s][f"mse_{a}_t{tv}_N{N}"] for tv in TAUS for N in NS]) for s in SEEDS],
            st["mse_all_tiers"][a])

# report
print(f"t(0.975, 2 df) closed form = {T2:.12f}")
print("anchor_order18 identical in all 3 seeds:", same_anchor)
for s in SEEDS:
    m = mine[s]
    print(f"s{s}: K0==root (FS, 4 pools, rounded arrays) {m['K0_equal_root_fs']}; K0==root (M_0=18, both tau modes) {m['K0_equal_root_sr18']}; "
          f"FS tiles shared NW/SE {m['fs_tiles_shared_NW_SE']}, first 6 tile slots = the partition {m['fs_first6_tiles_are_partition']}, "
          f"slots 24/28 repeat first 2 {m['fs_slots_24_28_repeat_first2']}; first-24 coverage {m['fs_first24_cover_all_cells']:.4f}; "
          f"skewed reads in preferred half NW {m['fs_skewed_reads_in_pref_half_NW']:.4f} SE {m['fs_skewed_reads_in_pref_half_SE']:.4f}")
    print(f"s{s}: sr18 tau {m['sr18_tau_share']}; sr18t3 tau {m['sr18t3_tau_share']}; sr18 reads in NW half under U_NW "
          f"{m['sr18_reads_in_pref_half_NW']:.4f}; D3 background disjoint from A* {m['d3_bg_disjoint_from_astar']}; PUCT 6 distinct per wave "
          f"{all(m[f'puct_{c}_distinct_within_wave_{h}'] for c in CELLS for h in ('NW', 'SE'))}")
print()
print(f"{'family':48s} {'n':>5s} {'max |mine - theirs|':>20s}  worst key")
worst_all = (None, 0.0)
summ = {}
for fam, L in diffs.items():
    k, a, b, d = max(L, key=lambda x: x[3])
    summ[fam] = {"n": len(L), "max_abs_diff": d, "worst": k, "mine": a, "theirs": b}
    print(f"{fam:48s} {len(L):5d} {d:20.3e}  {k} (mine {a:.7g}, theirs {b:.7g})")
    if d > worst_all[1]:
        worst_all = ((fam, k, a, b), d)
print("\nLARGEST absolute difference over all", sum(len(L) for L in diffs.values()), "comparisons:", worst_all)
json.dump({"t975_df2": T2, "anchor_order18_same_all_seeds": same_anchor, "per_seed": mine, "three_seed": agg, "diff_summary": summ,
           "largest": {"where": worst_all[0], "abs_diff": worst_all[1]}}, open(os.path.join(OUT, "a4_rederive.json"), "w"), indent=1)
