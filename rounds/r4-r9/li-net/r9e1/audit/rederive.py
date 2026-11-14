#!/usr/bin/env python3
"""R9-E1 audit, checks 3-5: independent re-derivation of the headline numbers from results/main/s<seed>/raw.json.
Auditor's own code: imports json, math, hashlib and numpy only (no e1v6/e1v5/e1v4/e1v3/analyze9).
Metric definitions re-implemented from PREREG section 1.4: KL = mean over boards and cells of Bernoulli KL(exact Bayes || pool),
pool clamped to [1e-6, 1 - 1e-6] (as the registered metric), averaged over the U_NW and U_SE bags; JSD between the two bags.
Usage: OMP_NUM_THREADS=2 python3 rederive.py  -> rederive_out.json and stdout report (saved as rederive.txt)."""
import hashlib
import json
import math
import os

import numpy as np

R9 = "/home/agent/workspaces/eurisko-study/li-net/r9e1"
RES = f"{R9}/results/main"
OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = (101, 102, 103)
M0S = (2, 6, 12, 18, 30, 32, 34)
KS_FS = (0, 1, 2, 4, 8, 16, 24, 32)
KS_SR = (0, 1, 2, 4, 8, 16, 24)
TAUS = ("0.3", "1.5", "2.5")
NS = (1, 2, 4, 8, 16)
FS_CURVE = ("looped_attn", "attn_K3", "mmd", "scaled_mix")
FS_K24 = ("jacobi_prec", "jacobi", "looped_lin", "kc_K1", "kc_K3", "econ_prec", "econ_D3")
SR_POOLS = ("looped_attn", "attn_K3", "mmd", "mix", "scaled_mix")
W_POOLS = ("looped_attn", "jacobi_prec", "econ_R7", "econ_D3", "kc_K3", "econ_prec")
MSE_POOLS = ("jacobi_prec", "econ_prec", "econ_D3", "kc_K3")
T_REG = 4.303                      # PREREG 1.5 (analyze9 uses 4.302652729911275; both are checked)
T_EXACT = 4.302652729911275


def kl(p, q):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-6, 1 - 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - np.log(q)), 0.0)
        b = np.where(p < 1, (1 - p) * (np.log(np.where(p < 1, 1 - p, 1.0)) - np.log1p(-q)), 0.0)
    return float((a + b).mean())


def ent(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0) + np.where(p < 1, (1 - p) * np.log(np.where(p < 1, 1 - p, 1.0)), 0.0)
    return -t


def jsd(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    return float((ent(0.5 * (p + q)) - 0.5 * (ent(p) + ent(q))).mean())


def ci(xs, t=T_REG):
    x = np.asarray(xs, dtype=np.float64)
    m = float(x.mean())
    h = t * float(x.std(ddof=1)) / math.sqrt(len(x))
    return {"mean": m, "lo": m - h, "hi": m + h, "per_seed": [float(v) for v in x]}


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def per_seed(s):
    raw = json.load(open(f"{RES}/s{s}/raw.json"))
    S = json.load(open(f"{RES}/s{s}/summary.json"))
    T = S["test"]
    ps = np.asarray(raw["pstar"], dtype=np.float64)
    A = lambda x: np.asarray(x, dtype=np.float64)
    mine, ref, extra = {}, {}, {}
    Ms = int(raw["M_strong"])
    extra["M_strong_raw"] = Ms
    extra["M_strong_summary_fit"] = int(S["fit"]["M_strong"])
    # ---- roots
    for M in M0S:
        mine[f"kl_root_{M}"] = kl(ps, raw["root_mu"][str(M)])
        ref[f"kl_root_{M}"] = T["kl_root"][str(M)]
    # ---- sparse-root curve (M_0 = 2, interleaved P-FullSupport stream)
    for K in KS_FS:
        for a in FS_CURVE:
            nw, se = A(raw["fs_NW"][f"K{K}_{a}"]), A(raw["fs_SE"][f"K{K}_{a}"])
            mine[f"fs_kl_{a}_K{K}"] = 0.5 * (kl(ps, nw) + kl(ps, se))
            mine[f"fs_jsd_{a}_K{K}"] = jsd(nw, se)
            ref[f"fs_kl_{a}_K{K}"] = T[f"fs_kl_{a}_K{K}"]
            ref[f"fs_jsd_{a}_K{K}"] = T[f"fs_jsd_{a}_K{K}"]
    for a in FS_K24:
        nw, se = A(raw["fs_NW"][f"K24_{a}"]), A(raw["fs_SE"][f"K24_{a}"])
        mine[f"fs_kl_{a}_K24"] = 0.5 * (kl(ps, nw) + kl(ps, se))
        mine[f"fs_jsd_{a}_K24"] = jsd(nw, se)
        ref[f"fs_kl_{a}_K24"] = T[f"fs_kl_{a}_K24"]
        ref[f"fs_jsd_{a}_K24"] = T[f"fs_jsd_{a}_K24"]
    # ---- strong-root curves at M_0_strong, Mode A and Mode B
    for md in ("A", "B"):
        for K in KS_SR:
            for a in SR_POOLS:
                nw, se = A(raw[f"srS{md}_NW"][f"K{K}_{a}"]), A(raw[f"srS{md}_SE"][f"K{K}_{a}"])
                mine[f"sr{md}_kl_{a}_K{K}"] = 0.5 * (kl(ps, nw) + kl(ps, se))
                mine[f"sr{md}_jsd_{a}_K{K}"] = jsd(nw, se)
                ref[f"sr{md}_kl_{a}_K{K}"] = T[f"sr{md}{Ms}_kl_{a}_K{K}"]
                ref[f"sr{md}_jsd_{a}_K{K}"] = T[f"sr{md}{Ms}_jsd_{a}_K{K}"]
    # ---- D3 controlled bags
    C = raw["controlled"]
    for t in TAUS:
        for N in NS:
            row = T["controlled"][f"t{t}_N{N}"]
            for a in W_POOLS:
                mine[f"W_{a}_t{t}_N{N}"] = float(A(C[f"t{t}_N{N}_W_{a}"]).mean())
                ref[f"W_{a}_t{t}_N{N}"] = row[f"W_{a}"]
            for a in MSE_POOLS:
                mine[f"mse_{a}_t{t}_N{N}"] = float(((A(C[f"t{t}_N{N}"][a]) - ps) ** 2).mean())
                ref[f"mse_{a}_t{t}_N{N}"] = row[f"mse_{a}"]
    # ---- check 3 from raw: K = 0 identity, paired patches, tau tiers, Mode A / Mode B token arithmetic
    r2, rS = A(raw["root_mu"]["2"]), A(raw["root_mu"][str(Ms)])
    k0 = {}
    for h in ("NW", "SE"):
        for a in FS_CURVE:
            k0[f"fs_{h}_{a}"] = float(np.abs(A(raw[f"fs_{h}"][f"K0_{a}"]) - r2).max())
        for md in ("A", "B"):
            for a in SR_POOLS:
                k0[f"sr{md}_{h}_{a}"] = float(np.abs(A(raw[f"srS{md}_{h}"][f"K0_{a}"]) - rS).max())
    extra["K0_max_abs_belief_diff_vs_root"] = k0
    extra["K0_kl_diff_vs_root"] = {
        "fs_looped_attn": abs(0.5 * (kl(ps, raw["fs_NW"]["K0_looped_attn"]) + kl(ps, raw["fs_SE"]["K0_looped_attn"])) - kl(ps, r2)),
        "fs_attn_K3": abs(0.5 * (kl(ps, raw["fs_NW"]["K0_attn_K3"]) + kl(ps, raw["fs_SE"]["K0_attn_K3"])) - kl(ps, r2)),
        **{f"sr{md}_{a}": abs(0.5 * (kl(ps, raw[f"srS{md}_NW"][f"K0_{a}"]) + kl(ps, raw[f"srS{md}_SE"][f"K0_{a}"])) - kl(ps, rS))
           for md in ("A", "B") for a in ("looped_attn", "attn_K3")}}
    extra["summary_identity_K0"] = {k: v for k, v in T["identity_K0"].items() if f"_M2_" in k or f"_M{Ms}_" in k}
    pair, taus = {}, {}
    for h in ("NW", "SE"):
        pa, pb = np.asarray(raw[f"srSA_{h}"]["patch"]), np.asarray(raw[f"srSB_{h}"]["patch"])
        pair[h] = bool((pa == pb).all())
        for md in ("A", "B"):
            tv = np.round(A(raw[f"srS{md}_{h}"]["tau"]), 3)
            taus[f"{md}_{h}"] = {str(x): float((tv == x).mean()) for x in np.unique(tv)}
    extra["same_patches_modeA_modeB"] = pair
    extra["tau_shares"] = taus
    sc = S["fit"]["scalars"]
    alpha, eta = sc["alpha"], sc["eta"]
    lS, l2 = logit(rS), logit(r2)
    tok = {}
    for md in ("A", "B"):
        for h in ("NW", "SE"):
            R_ = raw[f"srS{md}_{h}"]
            m1, m2 = A(R_["K1_mix"]), A(R_["K2_mix"])
            s1, s2 = A(R_["K1_scaled_mix"]), A(R_["K2_scaled_mix"])
            q2 = 2 * m2 - m1                                   # read 2's belief from the K = 2 mix (mix = mean of beliefs)
            ok1 = (np.minimum.reduce([rS, r2, m1, s1]) > 1e-3) & (np.maximum.reduce([rS, r2, m1, s1]) < 1 - 1e-3)
            ok2 = ok1 & (np.minimum(q2, s2) > 1e-2) & (np.maximum(q2, s2) < 1 - 1e-2)
            l1, lq2 = logit(m1), logit(q2)
            # hypothesis B: token = l_r - l0(2) added to l0(M_0); hypothesis A: token = l_r - l0(M_0)
            for hyp, base in (("over_l0_2", l2), ("over_l0_Ms", lS)):
                p1 = sig(lS + alpha * np.clip(l1 - base, -8, 8))
                p2 = sig(lS + alpha * (np.clip(l1 - base, -8, 8) + np.clip(lq2 - base, -8, 8)))
                tok[f"{md}_{h}_K1_scaled_mix_{hyp}"] = float(np.abs(p1 - s1)[ok1].max())
                tok[f"{md}_{h}_K2_scaled_mix_{hyp}"] = float(np.abs(p2 - s2)[ok2].max())
            # anytime pools and mmd at K = 1: implied per-cell weight (logit(out) - l0(M_0)) / Delta l, Delta l over l0(2) in B
            base = l2 if md == "B" else lS
            dl1 = np.clip(l1 - base, -8, 8)
            big = ok1 & (np.abs(dl1) > 0.05)
            for a in ("looped_attn", "attn_K3", "mmd"):
                o1 = A(R_[f"K1_{a}"])
                okp = big & (o1 > 1e-3) & (o1 < 1 - 1e-3)
                w = (logit(o1) - lS) / np.where(dl1 == 0, 1.0, dl1)
                tok[f"{md}_{h}_K1_{a}_implied_w_neg_share"] = float((w[okp] < -0.02).mean())
                tok[f"{md}_{h}_K1_{a}_implied_w_median"] = float(np.median(w[okp]))
                if a == "mmd":                                  # mmd: one scalar eta*beta_1 per board (same for every cell)
                    cv = []
                    for b in range(w.shape[0]):
                        wb = w[b][okp[b] & (np.abs(dl1[b]) > 0.5)]
                        if wb.size >= 4:
                            cv.append(float(wb.std() / abs(wb.mean())))
                    tok[f"{md}_{h}_K1_mmd_within_board_cv_median"] = float(np.median(cv))
                    tok[f"{md}_{h}_K1_mmd_implied_beta1_median"] = float(np.median(w[okp]) / eta)
    extra["token_checks"] = tok
    # ---- check 6: econ_prec closed form on identical reads (tau 0.3): N / W(N) is linear in N with slope 1 / u per board
    u = S["fit"]["econ_prec"]["u"]
    Wm = np.stack([A(C[f"t0.3_N{N}_W_econ_prec"]) for N in NS], 1)          # [500, 5]
    Ns = np.asarray(NS, dtype=np.float64)
    y = Ns[None] / Wm
    X = np.stack([np.ones(5), Ns], 1)
    coef, *_ = np.linalg.lstsq(X, y.T, rcond=None)                           # [2, 500]
    resid = y - (X @ coef).T
    extra["econ_prec_closed_form_t0.3"] = {
        "u_fitted": u, "one_over_u": 1.0 / u, "slope_per_board_median": float(np.median(coef[1])),
        "slope_per_board_max_abs_dev_from_1/u": float(np.abs(coef[1] - 1.0 / u).max()),
        "max_abs_residual_N_over_W": float(np.abs(resid).max()),
        "implied_ridge_plus_bg_over_u_beta_median": float(np.median(coef[0])),
        "lambda_at_beta_t0.3_fit": S["fit"]["econ_prec"]["lambda_at_beta"]["0.3"], "beta_t0.3_cal": S["fit"]["beta_by_tau"]["0.3"]}
    lam = S["fit"]["econ_prec"]["lambda_at_beta"]
    bt = S["fit"]["beta_by_tau"]
    extra["econ_prec_closed_form_no_background_W16/W1"] = {
        t: 16 * (lam[t] + bt[t]) / (lam[t] + 16 * bt[t]) for t in TAUS}
    extra["fit_scalars"] = sc
    extra["params"] = S["fit"]["params"]
    extra["clause1_corr_std6_summary"] = T["clause1_corr_std6"]
    extra["checkpoint_summary"] = S["checkpoint"]
    return mine, ref, extra


def main():
    res = {s: per_seed(s) for s in SEEDS}
    an = json.load(open(f"{RES}/analysis/results.json"))
    st = an["stats"]
    out = {"per_seed": {}, "diff_vs_summary": {}, "means": {}, "diff_vs_results_json": {}}
    worst = (0.0, None)
    cat_worst = {}

    def cat(k):
        for p in ("kl_root", "fs_kl", "fs_jsd", "srA_kl", "srB_kl", "srA_jsd", "srB_jsd", "W_", "mse_"):
            if k.startswith(p):
                return p
        return "other"

    for s in SEEDS:
        mine, ref, extra = res[s]
        out["per_seed"][s] = {"mine": mine, "extra": extra}
        d = {k: mine[k] - ref[k] for k in mine}
        out["diff_vs_summary"][s] = d
        for k, v in d.items():
            c = cat(k)
            if abs(v) > cat_worst.get(c, (0.0, None))[0]:
                cat_worst[c] = (abs(v), f"s{s} {k}: mine {mine[k]:.7f} summary {ref[k]:.7f}")
            if abs(v) > worst[0]:
                worst = (abs(v), f"s{s} {k}: mine {mine[k]:.7f} summary {ref[k]:.7f}")
    keys = list(res[SEEDS[0]][0].keys())
    means = {k: ci([res[s][0][k] for s in SEEDS]) for k in keys}
    out["means"] = means

    def st_key(k):                                     # map my key -> analysis/results.json stats entry
        if k.startswith("kl_root_"):
            return st["kl_root"][k.split("_")[-1]]
        if k.startswith("fs_"):
            return st[k]
        if k.startswith(("srA_", "srB_")):
            md = k[2]
            return st[f"srS{md}_" + k[4:]]
        if k.startswith("W_"):
            a, t, N = k[2:].rsplit("_t", 1)[0], k.split("_t")[-1].split("_N")[0], k.split("_N")[-1]
            return st["weights"][f"{a}_W_t{t}"][N]
        if k.startswith("mse_"):
            return None
        return None

    worst_m = (0.0, None)
    dm = {}
    for k in keys:
        e = st_key(k)
        if e is None:
            continue
        dm[k] = means[k]["mean"] - e["mean"]
        if abs(dm[k]) > worst_m[0]:
            worst_m = (abs(dm[k]), f"{k}: mine {means[k]['mean']:.7f} results.json {e['mean']:.7f}")
    # MSE averaged over N per tier and over tiers (results.json stats "mse" / "mse_all_tiers")
    mse_t = {a: {t: [float(np.mean([res[s][0][f"mse_{a}_t{t}_N{N}"] for N in NS])) for s in SEEDS] for t in TAUS} for a in MSE_POOLS}
    mse_all = {a: [float(np.mean([mse_t[a][t][i] for t in TAUS])) for i in range(3)] for a in MSE_POOLS}
    for a in MSE_POOLS:
        for t in TAUS:
            k = f"mse_{a}_t{t}"
            dm[k] = float(np.mean(mse_t[a][t])) - st["mse"][f"{a}_t{t}"]["mean"]
        dm[f"mse_all_{a}"] = float(np.mean(mse_all[a])) - st["mse_all_tiers"][a]["mean"]
        for k in (f"mse_all_{a}",) + tuple(f"mse_{a}_t{t}" for t in TAUS):
            if abs(dm[k]) > worst_m[0]:
                worst_m = (abs(dm[k]), k)
    out["diff_vs_results_json"] = dm
    out["mse_by_tier_per_seed"] = mse_t
    out["mse_all_tiers_per_seed"] = mse_all
    out["largest_abs_diff_vs_summary"] = worst
    out["largest_abs_diff_by_category_vs_summary"] = cat_worst
    out["largest_abs_diff_vs_results_json_means"] = worst_m

    # ---------------------------------------------------------------- check 5: re-decide every rule from my numbers
    P = lambda k, s: res[s][0][k]
    V = {}
    # E1-D0
    shas = {}
    for s in SEEDS:
        for root in ("/home/agent/workspaces/eurisko-study/li-net/r7e1/ckpt", "/tmp/r9e1/src/ckpt"):
            npz = f"{root}/s{s}/theta_real_v2.npz"
            js = json.load(open(f"{root}/s{s}/theta_real_v2.json"))
            shas[f"{root}/s{s}"] = {"file": sha256(npz), "json": js.get("sha256"),
                                    "summary": res[s][2]["checkpoint_summary"]["sha256"]}
    sha_ok = all(v["file"] == v["json"] == v["summary"] for v in shas.values())
    pear = ci([res[s][2]["clause1_corr_std6_summary"] for s in SEEDS])
    k0_raw = max(max(res[s][2]["K0_max_abs_belief_diff_vs_root"][f"{p}_{h}_{a}"] for p in ("fs", "srA", "srB")
                     for h in ("NW", "SE") for a in ("looped_attn", "attn_K3")) for s in SEEDS)
    k0_kl_raw = max(max(res[s][2]["K0_kl_diff_vs_root"].values()) for s in SEEDS)
    k0_sum = max(v for s in SEEDS for k, v in res[s][2]["summary_identity_K0"].items() if k.endswith("kl_diff"))
    V["E1-D0"] = {"sha256_file==json==summary": sha_ok, "pearson_mean (summary.json; not re-derivable from raw)": pear,
                  "K0_raw_max_abs_belief_diff": k0_raw, "K0_raw_max_kl_diff": k0_kl_raw, "K0_summary_max_kl_diff": k0_sum,
                  "pass": sha_ok and pear["mean"] >= 0.999 and k0_kl_raw < 1e-7 and k0_sum < 1e-7, "sha256": shas}
    # E1-D1.1
    la = "looped_attn"
    steps = {s: [P(f"fs_kl_{la}_K{KS_FS[i + 1]}", s) - P(f"fs_kl_{la}_K{KS_FS[i]}", s) for i in range(len(KS_FS) - 1)] for s in SEEDS}
    kl32, jsd24 = ci([P(f"fs_kl_{la}_K32", s) for s in SEEDS]), ci([P(f"fs_jsd_{la}_K24", s) for s in SEEDS])
    p11 = {"monotone": all(d <= 0.0005 for s in SEEDS for d in steps[s]), "KL32<=0.033": kl32["mean"] <= 0.033,
           "JSD24<=0.0075": jsd24["mean"] <= 0.0075}
    V["E1-D1.1"] = {"parts": p11, "pass": all(p11.values()), "max_step": max(d for s in SEEDS for d in steps[s]),
                    "steps": steps, "KL32": kl32, "JSD24": jsd24,
                    "per_seed_reading (not registered): KL32<=0.033 every seed": all(v <= 0.033 for v in kl32["per_seed"]),
                    "per_seed_reading (not registered): JSD24<=0.0075 every seed": all(v <= 0.0075 for v in jsd24["per_seed"])}
    # E1-D1.2 (every seed's M_0_strong is the raw's M_strong)
    kl0 = ci([P(f"srA_kl_{la}_K0", s) for s in SEEDS])
    p12 = {"KL0<=0.050": kl0["mean"] <= 0.050}
    deg = {}
    for md in ("A", "B"):
        for a in ("mix", "mmd"):
            deg[f"{md}_{a}"] = ci([P(f"sr{md}_kl_{a}_K16", s) - P(f"sr{md}_kl_{a}_K0", s) for s in SEEDS])
    for a in ("mix", "mmd"):
        p12[f"B_{a}_K16-K0>=0.015_lo>0"] = deg[f"B_{a}"]["mean"] >= 0.015 and deg[f"B_{a}"]["lo"] > 0
    c3 = {}
    for md in ("A", "B"):
        mx = {s: max(P(f"sr{md}_kl_{la}_K{K}", s) - P(f"sr{md}_kl_{la}_K0", s) for K in KS_SR[1:]) for s in SEEDS}
        g16 = ci([P(f"sr{md}_kl_{la}_K0", s) - P(f"sr{md}_kl_{la}_K16", s) for s in SEEDS])
        bt = {f"{b}_K{K}": ci([P(f"sr{md}_kl_{b}_K{K}", s) - P(f"sr{md}_kl_{la}_K{K}", s) for s in SEEDS])
              for b in ("mmd", "scaled_mix") for K in (16, 24)}
        c3[md] = {"max_increase": mx, "K0-K16": g16, "beats": bt}
        p12[f"{md}_never_degrades"] = all(v <= 0.0005 for v in mx.values())
        p12[f"{md}_K0-K16_lo>0"] = g16["mean"] > 0 and g16["lo"] > 0
        for k_, d_ in bt.items():
            p12[f"{md}_{k_}_lo>0"] = d_["mean"] > 0 and d_["lo"] > 0
    V["E1-D1.2"] = {"parts": p12, "pass": all(p12.values()), "KL0": kl0, "deg": deg, "clause3": c3}
    V["E1-D1"] = {"pass": V["E1-D1.1"]["pass"] and V["E1-D1.2"]["pass"]}
    # E1-D2.1 / E1-D2.2
    W = lambda s, t, N, a: res[s][0][f"W_{a}_t{t}_N{N}"]
    r16 = lambda a, t: ci([W(s, t, 16, a) / W(s, t, 1, a) for s in SEEDS])
    r8 = lambda a, t: ci([W(s, t, 8, a) / W(s, t, 1, a) for s in SEEDS])
    mono = lambda a: [W(s, "0.3", 8, a) / W(s, "0.3", 1, a) < W(s, "1.5", 8, a) / W(s, "1.5", 1, a) < W(s, "2.5", 8, a) / W(s, "2.5", 1, a)
                      for s in SEEDS]
    ep, jp = "econ_prec", "jacobi_prec"
    dmse = {b: ci([mse_all[b][i] - mse_all[ep][i] for i in range(3)]) for b in ("kc_K3", "econ_D3")}
    dmse_tier = {b: {t: ci([mse_t[b][t][i] - mse_t[ep][t][i] for i in range(3)]) for t in TAUS} for b in ("kc_K3", "econ_D3")}
    p21 = {"W16/W1_t0.3<=1.25": r16(ep, "0.3")["mean"] <= 1.25, "W16/W1_t0.3_hi<=1.30": r16(ep, "0.3")["hi"] <= 1.30,
           "monotone": all(mono(ep)), "W16/W1_t2.5>=2.20": r16(ep, "2.5")["mean"] >= 2.20,
           "MSE_kc_K3_lo>0": dmse["kc_K3"]["lo"] > 0, "MSE_econ_D3_lo>0": dmse["econ_D3"]["lo"] > 0}
    V["E1-D2.1"] = {"parts": p21, "pass": all(p21.values()), "W16/W1_t0.3": r16(ep, "0.3"), "W16/W1_t2.5": r16(ep, "2.5"),
                    "W8/W1": {t: r8(ep, t) for t in TAUS}, "mono": mono(ep), "dmse": dmse, "dmse_tier": dmse_tier,
                    "econ_D3_W16/W1": {t: r16("econ_D3", t) for t in TAUS}, "kc_K3_W16/W1": {t: r16("kc_K3", t) for t in TAUS}}
    fsj = ci([P(f"fs_kl_{jp}_K24", s) for s in SEEDS])
    beats = {b: ci([P(f"fs_kl_{b}_K24", s) - P(f"fs_kl_{jp}_K24", s) for s in SEEDS]) for b in ("looped_lin", "kc_K1", "kc_K3", ep)}
    p22 = {"W16/W1_t0.3<=1.25": r16(jp, "0.3")["mean"] <= 1.25, "monotone": all(mono(jp)), "W16/W1_t2.5>=2.20": r16(jp, "2.5")["mean"] >= 2.20,
           "FS_KL24<=0.0420": fsj["mean"] <= 0.0420, **{f"beats_{b}_lo>0": beats[b]["mean"] > 0 and beats[b]["lo"] > 0 for b in beats}}
    V["E1-D2.2"] = {"parts": p22, "pass": all(p22.values()), "W16/W1_t0.3": r16(jp, "0.3"), "W16/W1_t2.5": r16(jp, "2.5"),
                    "W8/W1": {t: r8(jp, t) for t in TAUS}, "mono": mono(jp), "fs_kl_K24": fsj, "beats": beats,
                    "lower_every_seed": {b: all(P(f"fs_kl_{jp}_K24", s) < P(f"fs_kl_{b}_K24", s) for s in SEEDS) for b in beats},
                    "looped_attn_fs_kl_K24": ci([P(f"fs_kl_{la}_K24", s) for s in SEEDS]),
                    "jacobi_R8_fs_kl_K24": ci([P("fs_kl_jacobi_K24", s) for s in SEEDS])}
    V["E1-D2"] = {"pass": V["E1-D2.1"]["pass"] and V["E1-D2.2"]["pass"]}
    # does t = 4.303 vs 4.302652729911275 change any CI-based part? (recompute the CI parts with the exact t)
    out["verdicts_mine"] = V
    out["analysis_verdicts"] = {k: an["verdicts"][k]["pass"] for k in an["verdicts"]}
    json.dump(out, open(os.path.join(OUT, "rederive_out.json"), "w"), indent=1, default=float)

    # ---------------------------------------------------------------- report
    f4 = lambda c, nd=4: f"{c['mean']:.{nd}f} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}] ({', '.join(f'{v:.{nd}f}' for v in c['per_seed'])})"
    print("== check 4: largest |mine - summary.json| over", sum(len(out["diff_vs_summary"][s]) for s in SEEDS), "per-seed numbers:",
          f"{worst[0]:.2e}", worst[1])
    for c_, (v, w) in sorted(cat_worst.items()):
        print(f"   {c_:8s} {v:.2e}  {w}")
    print("== check 4: largest |mine 3-seed mean - results.json mean| over", len(dm), "numbers:", f"{worst_m[0]:.2e}", worst_m[1])
    print("== root KL by M_0 (mine):", ", ".join(f"{M}: {means[f'kl_root_{M}']['mean']:.5f}" for M in M0S))
    print("== sparse root looped_attn KL by K:", ", ".join(f"{K}: {means[f'fs_kl_looped_attn_K{K}']['mean']:.5f}" for K in KS_FS),
          "| JSD(24)", f4(means["fs_jsd_looped_attn_K24"], 5))
    print("== K24 KL:", ", ".join(f"{a} {means[f'fs_kl_{a}_K24']['mean']:.5f}" for a in ("looped_attn",) + FS_K24))
    for md in ("A", "B"):
        for a in SR_POOLS:
            print(f"== strong root M_0_strong mode {md} {a} KL by K:", ", ".join(f"{K}: {means[f'sr{md}_kl_{a}_K{K}']['mean']:.5f}" for K in KS_SR))
    for a in W_POOLS:
        for t in TAUS:
            print(f"== W(N) {a} tau {t}:", ", ".join(f"{N}: {means[f'W_{a}_t{t}_N{N}']['mean']:.4f}" for N in NS))
    for a in MSE_POOLS:
        print(f"== D3 MSE {a}: all tiers {np.mean(mse_all[a]):.6f} per seed {[round(x, 6) for x in mse_all[a]]}; by tier",
              {t: round(float(np.mean(mse_t[a][t])), 6) for t in TAUS})
    print("== check 5 (mine) vs analysis/results.json verdicts:")
    for k in ("E1-D0", "E1-D1.1", "E1-D1.2", "E1-D1", "E1-D2.1", "E1-D2.2", "E1-D2"):
        print(f"   {k}: mine {'PASS' if V[k]['pass'] else 'FAIL'}; analysis {'PASS' if an['verdicts'][k]['pass'] else 'FAIL'}",
              {kk: vv for kk, vv in V[k].get("parts", {}).items()} if "parts" in V[k] else "")
    print("   D0:", {k: V["E1-D0"][k] for k in V["E1-D0"] if k not in ("sha256", "pass")})
    print("   D1.1: max step", f"{V['E1-D1.1']['max_step']:.6f}", "KL32", f4(kl32), "JSD24", f4(jsd24, 5),
          "| per-seed reading KL32 every seed:", V["E1-D1.1"]["per_seed_reading (not registered): KL32<=0.033 every seed"])
    print("   D1.2: KL0", f4(kl0), {k: f4(v) for k, v in deg.items()})
    for md in ("A", "B"):
        print(f"   D1.2 mode {md}: max increase", {s: round(v, 6) for s, v in c3[md]["max_increase"].items()}, "K0-K16", f4(c3[md]["K0-K16"]),
              {k: f4(v) for k, v in c3[md]["beats"].items()})
    print("   D2.1: W16/W1 t0.3", f4(r16(ep, "0.3"), 4), "t2.5", f4(r16(ep, "2.5"), 3), "W8/W1", {t: round(r8(ep, t)["mean"], 4) for t in TAUS},
          "mono", mono(ep), "dMSE kc_K3", f4(dmse["kc_K3"], 6), "dMSE econ_D3", f4(dmse["econ_D3"], 6))
    print("   D2.2: W16/W1 t0.3", f4(r16(jp, "0.3"), 4), "t2.5", f4(r16(jp, "2.5"), 3), "mono", mono(jp), "FS KL24", f4(fsj),
          {b: f4(v) for b, v in beats.items()})
    print("== check 3 (raw): K0 max |belief - root|", k0_raw, "; K0 max KL diff", k0_kl_raw)
    for s in SEEDS:
        e = res[s][2]
        print(f"   s{s}: same patches A/B {e['same_patches_modeA_modeB']}; tau shares {e['tau_shares']}")
        print(f"   s{s}: token checks", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e["token_checks"].items()})
        print(f"   s{s}: econ_prec closed form", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in e["econ_prec_closed_form_t0.3"].items()},
              "no-bg W16/W1", {t: round(v, 4) for t, v in e["econ_prec_closed_form_no_background_W16/W1"].items()})


if __name__ == "__main__":
    main()
