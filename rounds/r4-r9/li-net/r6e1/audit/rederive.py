#!/usr/bin/env python3
"""R6-E1 audit: independent re-derivation of the headline numbers from results/main/s*/raw.json.

Auditor's own code. It imports neither e1v3.py nor analyze6.py nor any li-net/r5e1 module; patch geometry, KL, JSD, coverage,
key geometry, W(N), MSE, CIs and the decision rules are re-implemented here from PREREG-R6-E1.md sections 1-3.
Run (CPU, 2 threads, about 1 min):
  OMP_NUM_THREADS=2 PYTHONPATH=/tmp/r4e3a/pylib python3 rederive.py
Writes rederive-output.json next to this file and prints a text report.
"""
import json
import math
import os
import sys

import numpy as np

R6 = "/home/agent/workspaces/eurisko-study/li-net/r6e1"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
T_EXACT = 4.302652729911275          # t_{0.975, df = 2}
T_PREREG = 4.303                     # the value written in PREREG-R6-E1.md section 3
LEVELS = (0, 1, 2, 3, 4, 6)
NS = (1, 2, 4, 8, 16)
TAUS = ("0.3", "1.5", "2.5")
POOLS = ("mix", "scaled_mix", "mmd", "kc", "looped", "attn_K1", "attn_K2", "attn_K3", "attn_K4")
AP_POOLS = ("attn_K4", "mix", "scaled_mix", "mmd", "kc")
TILES = (0, 3, 8, 11, 16, 19)


# ------------------------------------------------------------------ geometry (own implementation of the brief's definition)
def make_patches():
    """20 horizontal 2x3 (top-left row 0..4, col 0..3, row-major) then 20 vertical 3x2 (row 0..3, col 0..4)."""
    P = []
    for r in range(5):
        for c in range(4):
            P.append([6 * (r + a) + (c + b) for a in range(2) for b in range(3)])
    for r in range(4):
        for c in range(5):
            P.append([6 * (r + a) + (c + b) for a in range(3) for b in range(2)])
    return np.array(P)


PAT = make_patches()
MASK = np.zeros((40, 36), dtype=bool)
for i, p in enumerate(PAT):
    MASK[i, p] = True
OVL = MASK.astype(np.int64) @ MASK.T.astype(np.int64)            # shared cells between two patches
NWC = MASK[:, :18].sum(1)                                         # cells in rows 0-2
SECTOR = np.where(NWC >= 4, 0, np.where(6 - NWC >= 4, 1, 2))      # 0 NW only, 1 SE only, 2 tie (3/3)
IN_NW = NWC >= 3                                                  # half membership, ties count for both (registered)
IN_SE = (6 - NWC) >= 3


# ------------------------------------------------------------------ metrics (float64)
def bern_kl(p, q, eps=1e-6):
    """KL(Bern(p) || Bern(q)) per cell; q clamped to [eps, 1 - eps] as in the experiment's metric."""
    p = np.clip(p, 0.0, 1.0)
    q = np.clip(q, eps, 1 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1.0)) - np.log1p(-q)), 0.0)
    return a + b


def ent(p):
    p = np.clip(p, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        t2 = np.where(p < 1, (1 - p) * np.log(np.where(p < 1, 1 - p, 1.0)), 0.0)
    return -(t1 + t2)


def jsd(p, q):
    return ent(0.5 * (p + q)) - 0.5 * (ent(p) + ent(q))


def ci(xs, t=T_EXACT):
    x = np.asarray(xs, dtype=float)
    m = float(x.mean())
    h = t * float(x.std(ddof=1)) / math.sqrt(len(x))
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": [float(v) for v in x]}


def spearman(vals):
    v = np.asarray(vals, dtype=float)
    r = np.empty(len(v))
    order = np.argsort(v, kind="mergesort")
    r[order] = np.arange(len(v), dtype=float)
    for u in np.unique(v):                       # average ranks for ties
        m = v == u
        r[m] = r[m].mean()
    x = np.arange(len(v), dtype=float)
    return float(np.corrcoef(x, r)[0, 1])


def coverage(patch):
    """share of the 36 cells covered by at least one patch, per board; patch [n, k]."""
    return MASK[patch].any(1).mean(1)


def A(x):
    return np.asarray(x, dtype=np.float64)


# ------------------------------------------------------------------ per-seed re-derivation
def seed_numbers(s):
    raw = json.load(open(os.path.join(R6, "results", "main", f"s{s}", "raw.json")))
    summ = json.load(open(os.path.join(R6, "results", "main", f"s{s}", "summary.json")))
    T = summ["test"]
    pstar, mu0 = A(raw["pstar"]), A(raw["mu0"])
    kl = lambda m: float(bern_kl(pstar, m).mean())
    mine, theirs, checks = {}, {}, {}

    def put(name, val, ref):
        mine[name] = val
        theirs[name] = ref

    put("kl_p0", kl(mu0), T["kl_p0"])
    # ---- P-FullSupport
    fsN, fsS = raw["fs_NW"], raw["fs_SE"]
    pN, pS = np.asarray(fsN["patch"]), np.asarray(fsS["patch"])
    checks["fs_first6_are_tiles"] = bool((pN[:, :6] == np.array(TILES)).all() and (pS[:, :6] == np.array(TILES)).all())
    checks["fs_reads_per_bag"] = [int(pN.shape[1]), int(pS.shape[1])]
    put("fs_coverage_NW", float(coverage(pN).mean()), T["fs_coverage_NW"])
    put("fs_coverage_SE", float(coverage(pS).mean()), T["fs_coverage_SE"])
    # share of the 18 skewed reads lying in the preferred half (>= 3 of 6 cells; ties count for both): descriptive
    checks["fs_skewed_in_pref_half_NW"] = float(IN_NW[pN[:, 6:]].mean())
    checks["fs_skewed_in_pref_half_SE"] = float(IN_SE[pS[:, 6:]].mean())
    checks["fs_skewed_strict_NWonly_under_NW"] = float((SECTOR[pN[:, 6:]] == 0).mean())
    checks["fs_skewed_strict_SEonly_under_SE"] = float((SECTOR[pS[:, 6:]] == 1).mean())
    for a in POOLS + ("union",):
        mN, mS = A(fsN[a]), A(fsS[a])
        put(f"fs_kl_{a}", 0.5 * (kl(mN) + kl(mS)), T[f"fs_kl_{a}"])
        put(f"fs_jsd_{a}", float(jsd(mN, mS).mean()), T[f"fs_jsd_{a}"])
    uN, uS = A(fsN["union"]), A(fsS["union"])
    checks["fs_union_arrays_identical"] = bool(np.array_equal(uN, uS))
    checks["fs_union_max_abs_diff_NW_SE"] = float(np.abs(uN - uS).max())
    mine["fs_cut_scaled_mix"] = 1 - mine["fs_kl_scaled_mix"] / mine["kl_p0"]
    # ---- P-ActivePUCT
    for name in ("NW", "SE"):
        ap = raw[f"ap_{name}"]
        op, pp = np.asarray(ap["open_patch"]), np.asarray(ap["puct_patch"])
        put(f"ap_coverage_open_{name}", float(coverage(op).mean()), T[f"ap_coverage_open_{name}"])
        put(f"ap_coverage_puct_{name}", float(coverage(pp).mean()), T[f"ap_coverage_puct_{name}"])
        half = IN_NW if name == "NW" else IN_SE
        strict = (SECTOR == 0) if name == "NW" else (SECTOR == 1)
        put(f"ap_wave1_pref_share_{name}", float(half[pp[:, :6]].mean()), T[f"ap_wave1_pref_share_{name}"])
        for w in range(4):
            put(f"ap_wave{w + 1}_pref_share_{name}", float(half[pp[:, 6 * w:6 * w + 6]].mean()),
                T[f"ap_wave_pref_share_{name}"][w])
        put(f"ap_open_pref_share_{name}", float(half[op].mean()), T[f"ap_open_pref_share_{name}"])
        checks[f"ap_wave1_strict_share_{name}"] = float(strict[pp[:, :6]].mean())
        checks[f"puct_waves_have_6_distinct_patches_{name}"] = bool(all(
            (np.sort(pp[:, 6 * w:6 * w + 6], 1)[:, 1:] != np.sort(pp[:, 6 * w:6 * w + 6], 1)[:, :-1]).all() for w in range(4)))
        checks[f"puct_reads_{name}"] = int(pp.shape[1])
        checks[f"open_reads_{name}"] = int(op.shape[1])
        # sensitivity (not the registered definition): count the 2 root anchor cells as covered evidence
        anc = np.asarray(raw["anchors"])
        am = np.zeros((anc.shape[0], 36), dtype=bool)
        am[np.arange(anc.shape[0])[:, None], anc] = True
        checks[f"evidence_coverage_incl_anchors_puct_{name}"] = float((MASK[pp].any(1) | am).mean(1).mean())
        checks[f"evidence_coverage_incl_anchors_open_{name}"] = float((MASK[op].any(1) | am).mean(1).mean())
    for kind in ("open", "puct"):
        put(f"ap_coverage_{kind}", 0.5 * (mine[f"ap_coverage_{kind}_NW"] + mine[f"ap_coverage_{kind}_SE"]),
            T[f"ap_coverage_{kind}"])
        checks[f"evidence_coverage_incl_anchors_{kind}"] = 0.5 * (checks[f"evidence_coverage_incl_anchors_{kind}_NW"]
                                                                   + checks[f"evidence_coverage_incl_anchors_{kind}_SE"])
        for a in AP_POOLS:
            mN, mS = A(raw["ap_NW"][f"{kind}_{a}"]), A(raw["ap_SE"][f"{kind}_{a}"])
            put(f"ap_{kind}_kl_{a}", 0.5 * (kl(mN) + kl(mS)), T[f"ap_{kind}_kl_{a}"])
            put(f"ap_{kind}_jsd_{a}", float(jsd(mN, mS).mean()), T[f"ap_{kind}_jsd_{a}"])
    # ---- key geometry (K3, K4) from the saved keys and patches
    kb = raw["key_bags"]
    kpatch = np.asarray(kb["patch"])                                              # [3, 500, 24]
    checks["keybag0_is_open_NW_bag"] = bool(np.array_equal(kpatch[0], np.asarray(raw["ap_NW"]["open_patch"])))
    checks["keybag1_is_open_SE_bag"] = bool(np.array_equal(kpatch[1], np.asarray(raw["ap_SE"]["open_patch"])))
    iu = np.triu_indices(kpatch.shape[2], 1)
    for v in ("K3", "K4"):
        K = np.asarray(kb[f"keys_{v}"], dtype=np.float64)                         # [3, 500, 24, 16], rounded to 4 dp
        cs_l, ov_l, sm_l, ac_l, csn_l = [], [], [], [], []
        Kn = K / np.linalg.norm(K, axis=-1, keepdims=True)
        for j in range(3):
            G = np.einsum("bik,bjk->bij", K[j], K[j])[:, iu[0], iu[1]]
            Gn = np.einsum("bik,bjk->bij", Kn[j], Kn[j])[:, iu[0], iu[1]]
            p1, p2 = kpatch[j][:, iu[0]], kpatch[j][:, iu[1]]
            ov = OVL[p1, p2]
            s1, s2 = SECTOR[p1], SECTOR[p2]
            cs_l.append(G.ravel())
            csn_l.append(Gn.ravel())
            ov_l.append(ov.ravel())
            sm_l.append(G[p1 == p2])
            ac_l.append(G[((s1 == 0) & (s2 == 1)) | ((s1 == 1) & (s2 == 0))])
        cs, csn, ov = np.concatenate(cs_l), np.concatenate(csn_l), np.concatenate(ov_l).astype(float)
        put(f"key_{v}_pearson", float(np.corrcoef(cs, ov / 6)[0, 1]), T[f"key_{v}_pearson"])
        put(f"key_{v}_same", float(np.concatenate(sm_l).mean()), T[f"key_{v}_same"])
        put(f"key_{v}_across", float(np.concatenate(ac_l).mean()), T[f"key_{v}_across"])
        put(f"key_{v}_gap", mine[f"key_{v}_same"] - mine[f"key_{v}_across"], T[f"key_{v}_gap"])
        for o in LEVELS:
            put(f"key_{v}_cos_ov{o}", float(cs[ov == o].mean()), T[f"key_{v}_by_overlap"][str(o)])
        checks[f"key_{v}_pearson_renormalised_keys"] = float(np.corrcoef(csn, ov / 6)[0, 1])
        checks[f"key_{v}_npairs"] = int(cs.size)
        checks[f"key_{v}_n_same_pairs"] = int(np.concatenate(sm_l).size)
        checks[f"key_{v}_n_across_pairs"] = int(np.concatenate(ac_l).size)
        checks[f"key_{v}_pairs_by_level"] = {str(o): int((ov == o).sum()) for o in LEVELS}
        checks[f"key_{v}_overlap5_pairs"] = int((ov == 5).sum())
    # ---- controlled bags (E1-D3)
    C = raw["controlled"]
    astar, bgp = np.asarray(C["astar"]), np.asarray(C["bg_patch"])
    checks["ctrl_bg_never_overlaps_astar"] = bool((OVL[astar[:, None], bgp] == 0).all())
    checks["ctrl_bg_reads"] = int(bgp.shape[1])
    pm = MASK[astar]
    for t in TAUS:
        for N in NS:
            row = T["controlled"][f"t{t}_N{N}"]
            put(f"W_t{t}_N{N}", float(A(C[f"t{t}_N{N}_W_K4"]).mean()), row["W_K4"])
            for a in ("attn_K4", "mmd", "kc", "mix"):
                put(f"mse_{a}_t{t}_N{N}", float(((A(C[f"t{t}_N{N}"][a]) - pstar) ** 2).mean()), row[f"mse_{a}"])
                mine[f"mse_patch_{a}_t{t}_N{N}"] = float((((A(C[f"t{t}_N{N}"][a]) - pstar) ** 2) * pm).sum(1).mean() / 6)
    return mine, theirs, checks


def main():
    per = {s: seed_numbers(s) for s in SEEDS}
    M = {s: per[s][0] for s in SEEDS}
    TH = {s: per[s][1] for s in SEEDS}
    CK = {s: per[s][2] for s in SEEDS}
    # ---- discrepancy vs summary.json (per seed)
    diffs = []
    for s in SEEDS:
        for k, ref in TH[s].items():
            diffs.append((abs(M[s][k] - ref), s, k, M[s][k], ref))
    diffs.sort(reverse=True)
    cat = lambda k: ("fs" if k.startswith("fs_") or k == "kl_p0" else "ap" if k.startswith("ap_") else
                     "keys" if k.startswith("key_") else "W" if k.startswith("W_") else "mse" if k.startswith("mse_") else "other")
    by_cat = {}
    for d in diffs:
        c = cat(d[2])
        if c not in by_cat or d[0] > by_cat[c][0]:
            by_cat[c] = d
    # ---- seed-level derived quantities and rules (PREREG-R6-E1 section 3)
    R = {}
    t_used = {}
    for tname, tval in (("t_exact", T_EXACT), ("t_prereg_4.303", T_PREREG)):
        st = {}
        f = lambda xs: ci(xs, tval)
        st["fs_cut_scaled_mix"] = f([M[s]["fs_cut_scaled_mix"] for s in SEEDS])
        st["clause1_pearson_from_summary"] = f([json.load(open(os.path.join(R6, "results", "main", f"s{s}", "summary.json")))
                                               ["test"]["clause1_corr_std6"] for s in SEEDS])
        d0 = st["clause1_pearson_from_summary"]["mean"] >= 0.99 and st["fs_cut_scaled_mix"]["mean"] >= 0.40
        keys = {}
        for v in ("K3", "K4"):
            p = f([M[s][f"key_{v}_pearson"] for s in SEEDS])
            g = f([M[s][f"key_{v}_gap"] for s in SEEDS])
            curve = [float(np.mean([M[s][f"key_{v}_cos_ov{o}"] for s in SEEDS])) for o in LEVELS]
            sp = spearman(curve)
            sp_seed = [spearman([M[s][f"key_{v}_cos_ov{o}"] for o in LEVELS]) for s in SEEDS]
            c1 = p["mean"] >= 0.60 and p["lo"] > 0.50 and g["mean"] >= 0.50
            keys[v] = {"pearson": p, "gap": g, "curve": curve, "spearman_mean_curve": sp, "spearman_per_seed": sp_seed,
                       "same": f([M[s][f"key_{v}_same"] for s in SEEDS]), "across": f([M[s][f"key_{v}_across"] for s in SEEDS]),
                       "clause1": c1, "clause2": sp >= 0.94, "pass": c1 and sp >= 0.94}
        d1 = keys["K4"]["pass"] or keys["K3"]["pass"]
        r21, ok21 = {}, True
        for b in ("scaled_mix", "mmd", "kc"):
            ratio = f([M[s][f"fs_jsd_{b}"] / M[s]["fs_jsd_attn_K4"] for s in SEEDS])
            dj = f([M[s][f"fs_jsd_{b}"] - M[s]["fs_jsd_attn_K4"] for s in SEEDS])
            dk = f([M[s][f"fs_kl_{b}"] - M[s]["fs_kl_attn_K4"] for s in SEEDS])
            ratio_of_means = float(np.mean([M[s][f"fs_jsd_{b}"] for s in SEEDS]) / np.mean([M[s]["fs_jsd_attn_K4"] for s in SEEDS]))
            r21[b] = {"jsd_ratio": ratio, "jsd_diff": dj, "kl_diff": dk, "jsd_ratio_of_means": ratio_of_means,
                      "pass": ratio["mean"] >= 2.5 and dj["lo"] > 0 and dk["lo"] > 0}
            ok21 = ok21 and r21[b]["pass"]
        lp = f([M[s]["fs_kl_looped"] / M[s]["fs_kl_attn_K4"] for s in SEEDS])
        r21["looped_secondary_kl_ratio"] = lp
        r21["attn_K4_lowest_kl_among_all_pools_per_seed"] = [min(POOLS, key=lambda a: M[s][f"fs_kl_{a}"]) for s in SEEDS]
        cov_p = f([M[s]["ap_coverage_puct"] for s in SEEDS])
        cov_o = f([M[s]["ap_coverage_open"] for s in SEEDS])
        jr = f([M[s]["ap_open_jsd_attn_K4"] / M[s]["ap_puct_jsd_attn_K4"] for s in SEEDS])
        jd = f([M[s]["ap_open_jsd_attn_K4"] - M[s]["ap_puct_jsd_attn_K4"] for s in SEEDS])
        kc_ = f([1 - M[s]["ap_puct_kl_attn_K4"] / M[s]["ap_open_kl_attn_K4"] for s in SEEDS])
        w1 = f([M[s]["ap_wave1_pref_share_NW"] for s in SEEDS])
        parts22 = {"coverage>=0.95": cov_p["mean"] >= 0.95, "jsd_ratio>=1.8": jr["mean"] >= 1.8, "jsd_diff_ci_lo>0": jd["lo"] > 0,
                   "kl_cut>=0.20": kc_["mean"] >= 0.20, "wave1_share>=0.75": w1["mean"] >= 0.75}
        d22 = all(parts22.values())
        W = lambda s, t, N: M[s][f"W_t{t}_N{N}"]
        mse = lambda s, t, a: float(np.mean([M[s][f"mse_{a}_t{t}_N{N}"] for N in NS]))
        r3 = {}
        for t in TAUS:
            r3[f"W8/W1_t{t}"] = f([W(s, t, 8) / W(s, t, 1) for s in SEEDS])
            r3[f"W16/W1_t{t}"] = f([W(s, t, 16) / W(s, t, 1) for s in SEEDS])
            r3[f"W_t{t}"] = {str(N): f([W(s, t, N) for s in SEEDS]) for N in NS}
            for b in ("mmd", "kc", "mix"):
                r3[f"msecut_vs_{b}_t{t}"] = f([1 - mse(s, t, "attn_K4") / mse(s, t, b) for s in SEEDS])
                # sensitivities (not registered): mean over N of per-N cuts; N = 16 only; A* cells only
                r3[f"msecut_meanOfPerN_vs_{b}_t{t}"] = f([float(np.mean([1 - M[s][f"mse_attn_K4_t{t}_N{N}"] / M[s][f"mse_{b}_t{t}_N{N}"]
                                                                         for N in NS])) for s in SEEDS])
                r3[f"msecut_N16_vs_{b}_t{t}"] = f([1 - M[s][f"mse_attn_K4_t{t}_N16"] / M[s][f"mse_{b}_t{t}_N16"] for s in SEEDS])
                r3[f"msecut_Astar_cells_vs_{b}_t{t}"] = f([1 - np.mean([M[s][f"mse_patch_attn_K4_t{t}_N{N}"] for N in NS])
                                                          / np.mean([M[s][f"mse_patch_{b}_t{t}_N{N}"] for N in NS]) for s in SEEDS])
        c1 = r3["W16/W1_t0.3"]["mean"] <= 1.25 and r3["msecut_vs_mmd_t0.3"]["mean"] >= 0.35
        c2 = (r3["W8/W1_t2.5"]["mean"] >= 2.20 and r3["W8/W1_t2.5"]["lo"] > 1.90 and r3["W16/W1_t2.5"]["mean"] >= 2.80
              and r3["msecut_vs_kc_t2.5"]["mean"] >= 0.15)
        mono = [W(s, "2.5", 8) / W(s, "2.5", 1) > W(s, "1.5", 8) / W(s, "1.5", 1) > W(s, "0.3", 8) / W(s, "0.3", 1) for s in SEEDS]
        d3 = c1 and c2 and all(mono)
        R[tname] = {"E1-D0": {"pass": d0, "fs_cut_scaled_mix": st["fs_cut_scaled_mix"],
                              "clause1_pearson_from_summary": st["clause1_pearson_from_summary"]},
                    "E1-D1": {"pass": d1, "keys": keys},
                    "E1-D2.1": {"pass": ok21, "stats": r21},
                    "E1-D2.2": {"pass": d22, "parts": parts22, "coverage_puct": cov_p, "coverage_open": cov_o, "jsd_ratio": jr,
                                "jsd_diff": jd, "kl_cut": kc_, "wave1_nw_share": w1,
                                "evidence_coverage_incl_anchors_puct": f([CK[s]["evidence_coverage_incl_anchors_puct"] for s in SEEDS]),
                                "evidence_coverage_incl_anchors_open": f([CK[s]["evidence_coverage_incl_anchors_open"] for s in SEEDS]),
                                "wave1_strict_NWonly_share": f([CK[s]["ap_wave1_strict_share_NW"] for s in SEEDS])},
                    "E1-D3": {"pass": d3, "parts": {"clean_dedup": c1, "high_noise_support": c2, "monotone": all(mono)},
                              "monotone_per_seed": mono, "stats": r3}}
    # ---- compare with results.json
    RJ = json.load(open(os.path.join(R6, "results", "main", "analysis", "results.json")))
    S = RJ["stats"]
    V = RJ["verdicts"]
    mine_rules = R["t_exact"]
    cmp_ = []

    def cmpci(label, a, b):
        for k in ("mean", "lo", "hi"):
            cmp_.append((abs(a[k] - b[k]), label + "." + k, a[k], b[k]))

    cmpci("fs_cut_scaled_mix", mine_rules["E1-D0"]["fs_cut_scaled_mix"], S["fs_cut_scaled_mix"])
    for v in ("K3", "K4"):
        cmpci(f"keys.{v}.pearson", mine_rules["E1-D1"]["keys"][v]["pearson"], S["keys"][v]["pearson"])
        cmpci(f"keys.{v}.gap", mine_rules["E1-D1"]["keys"][v]["gap"], S["keys"][v]["gap"])
        for i, o in enumerate(LEVELS):
            cmp_.append((abs(mine_rules["E1-D1"]["keys"][v]["curve"][i] - S["keys"][v]["cos_by_overlap_mean"][i]),
                         f"keys.{v}.cos_ov{o}", mine_rules["E1-D1"]["keys"][v]["curve"][i], S["keys"][v]["cos_by_overlap_mean"][i]))
    for b in ("scaled_mix", "mmd", "kc"):
        for k in ("jsd_ratio", "jsd_diff", "kl_diff"):
            cmpci(f"D2.1.{b}.{k}", mine_rules["E1-D2.1"]["stats"][b][k], V["E1-D2.1"]["stats"][b][k])
    cmpci("D2.1.looped_kl_ratio", mine_rules["E1-D2.1"]["stats"]["looped_secondary_kl_ratio"],
          V["E1-D2.1"]["stats"]["looped_secondary"]["kl_ratio_to_attn"])
    for mk, tk in (("coverage_puct", "coverage_puct"), ("coverage_open", "coverage_open"), ("jsd_ratio", "jsd_ratio_open_over_puct"),
                   ("jsd_diff", "jsd_diff_open_minus_puct"), ("kl_cut", "kl_cut_puct_vs_open"), ("wave1_nw_share", "wave1_nw_share")):
        cmpci(f"D2.2.{mk}", mine_rules["E1-D2.2"][mk], V["E1-D2.2"]["stats"][tk])
    for k in ("W8/W1_t0.3", "W16/W1_t0.3", "W8/W1_t1.5", "W16/W1_t1.5", "W8/W1_t2.5", "W16/W1_t2.5", "msecut_vs_mmd_t0.3",
              "msecut_vs_kc_t2.5", "msecut_vs_mmd_t2.5", "msecut_vs_mix_t2.5"):
        cmpci(f"D3.{k}", mine_rules["E1-D3"]["stats"][k], V["E1-D3"]["stats"][k])
    cmp_.sort(reverse=True)
    verdict_cmp = {r: {"mine": mine_rules[r]["pass"], "results_json": V[r]["pass"],
                       "mine_with_t4.303": R["t_prereg_4.303"][r]["pass"]} for r in ("E1-D0", "E1-D1", "E1-D2.1", "E1-D2.2", "E1-D3")}
    out = {"per_seed_mine": M, "per_seed_summary": TH, "checks": CK, "rules": R,
           "largest_seed_level_diffs_vs_summary": [list(d) for d in diffs[:15]],
           "largest_diff_by_category": {c: list(d) for c, d in by_cat.items()},
           "n_seed_level_values_compared": len(diffs),
           "largest_stat_diffs_vs_results_json": [list(d) for d in cmp_[:15]],
           "n_stat_values_compared": len(cmp_), "verdicts": verdict_cmp}
    json.dump(out, open(os.path.join(OUT, "rederive-output.json"), "w"), indent=1, default=float)
    # ---- text report
    p = print
    p(f"# R6-E1 audit re-derivation ({len(diffs)} seed-level values compared with summary.json; {len(cmp_)} 3-seed statistics "
      f"compared with analysis/results.json)")
    p("\n## largest |mine - summary.json| (seed-level)")
    for d in diffs[:12]:
        p(f"  {d[0]:.3e}  s{d[1]}  {d[2]}: mine {d[3]:.7f}  summary {d[4]:.7f}")
    p("\n## largest per category")
    for c, d in sorted(by_cat.items()):
        p(f"  {c:6s} {d[0]:.3e}  s{d[1]} {d[2]}")
    p("\n## largest |mine - results.json| (3-seed mean / CI bounds)")
    for d in cmp_[:10]:
        p(f"  {d[0]:.3e}  {d[1]}: mine {d[2]:.7f}  results.json {d[3]:.7f}")
    p("\n## verdicts (mine with t = 4.30265 / mine with t = 4.303 / results.json)")
    for r, v in verdict_cmp.items():
        p(f"  {r}: {'PASS' if v['mine'] else 'FAIL'} / {'PASS' if v['mine_with_t4.303'] else 'FAIL'} / "
          f"{'PASS' if v['results_json'] else 'FAIL'}")
    rr = R["t_exact"]
    fm = lambda c, nd=4: f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({', '.join(f'{x:.{nd}f}' for x in c['per_seed'])})"
    p("\n## rule statistics (mine)")
    p(f"  E1-D0: fs_cut_scaled_mix {fm(rr['E1-D0']['fs_cut_scaled_mix'])}; clause-1 Pearson (summary.json only) "
      f"{fm(rr['E1-D0']['clause1_pearson_from_summary'])}")
    for v in ("K4", "K3"):
        k = rr["E1-D1"]["keys"][v]
        p(f"  E1-D1 {v}: Pearson {fm(k['pearson'])}; gap {fm(k['gap'])}; same {k['same']['mean']:.4f} across "
          f"{k['across']['mean']:.4f}; curve {', '.join(f'{x:.4f}' for x in k['curve'])}; Spearman mean-curve "
          f"{k['spearman_mean_curve']:.3f}, per seed {k['spearman_per_seed']}; clause1 {k['clause1']} clause2 {k['clause2']}")
    for b in ("scaled_mix", "mmd", "kc"):
        k = rr["E1-D2.1"]["stats"][b]
        p(f"  E1-D2.1 {b}: JSD ratio {fm(k['jsd_ratio'], 3)}; JSD diff {fm(k['jsd_diff'], 5)}; KL diff {fm(k['kl_diff'], 4)}; "
          f"ratio of means {k['jsd_ratio_of_means']:.3f}")
    p(f"  E1-D2.1 looped KL ratio {fm(rr['E1-D2.1']['stats']['looped_secondary_kl_ratio'], 3)}; lowest-KL pool per seed "
      f"{rr['E1-D2.1']['stats']['attn_K4_lowest_kl_among_all_pools_per_seed']}")
    k = rr["E1-D2.2"]
    p(f"  E1-D2.2: coverage puct {fm(k['coverage_puct'], 5)}; open {fm(k['coverage_open'], 4)}; JSD ratio {fm(k['jsd_ratio'], 3)}; "
      f"JSD diff {fm(k['jsd_diff'], 5)}; KL cut {fm(k['kl_cut'], 4)}; wave-1 NW share {fm(k['wave1_nw_share'], 4)}; parts {k['parts']}")
    p(f"    sensitivity (not registered): evidence coverage incl. anchors puct {fm(k['evidence_coverage_incl_anchors_puct'], 5)}, "
      f"open {fm(k['evidence_coverage_incl_anchors_open'], 4)}; wave-1 strict NW-only share {fm(k['wave1_strict_NWonly_share'], 4)}")
    k = rr["E1-D3"]
    for key in ("W16/W1_t0.3", "W8/W1_t0.3", "W8/W1_t1.5", "W16/W1_t1.5", "W8/W1_t2.5", "W16/W1_t2.5", "msecut_vs_mmd_t0.3",
                "msecut_vs_kc_t2.5", "msecut_vs_mmd_t2.5", "msecut_vs_mix_t2.5", "msecut_meanOfPerN_vs_kc_t2.5",
                "msecut_N16_vs_kc_t2.5", "msecut_Astar_cells_vs_kc_t2.5", "msecut_meanOfPerN_vs_mmd_t0.3"):
        p(f"  E1-D3 {key}: {fm(k['stats'][key], 4)}")
    p(f"  E1-D3 W(N) means: " + "; ".join(f"tau {t}: " + ", ".join(f"{k['stats'][f'W_t{t}'][str(N)]['mean']:.4f}" for N in NS)
                                          for t in TAUS))
    p(f"  E1-D3 parts {k['parts']}; monotone per seed {k['monotone_per_seed']}")
    p("\n## structural checks (per seed)")
    for s in SEEDS:
        p(f"  s{s}: " + json.dumps(CK[s], default=float))


if __name__ == "__main__":
    main()
