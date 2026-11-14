"""P10 evaluation (numpy only; run with system python3).

  python3 eval.py g1       gate G1 on S4: within-state tie-aware Spearman of the value head vs the realised commit gain,
                           against the free proxies; partial Spearman controlling for entropy; vs exact I*(z_c; z_q|X)
  python3 eval.py s4read   S4 readouts: void-rule variance ratio and Huber bias; joint NLL vs passes (ARDM frontier)
  python3 eval.py pilot    S6 T rule (smallest T in {1,2} with B2-reg 2T-pass held-out loss >= max(0.05, 10 x fee))
  python3 eval.py m2       S-B (i)-(vi), F3, F3b, F4 from the M2 auction / cvae / reg runs
  python3 eval.py all

Reads results/*.json (copied from vm_out/results) and the S4 npz scratch (vm_out/tmp or /tmp/p10).
Writes results/{g1,s4_readouts,s6_T_choice,m2_summary}.json and charts/*.json.
"""
import glob
import json
import math
import os
import sys
import numpy as np

import stats as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P10_EVAL_OUT", os.path.join(HERE, "results"))
CH = os.environ.get("P10_EVAL_CHARTS", os.path.join(HERE, "charts"))
TMPDIRS = [d for d in (os.environ.get("P10_EVAL_TMP"), os.path.join(HERE, "vm_out", "tmp"), "/tmp/p10") if d]
RESDIRS = [d for d in (os.environ.get("P10_EVAL_RES"), RES, os.path.join(HERE, "vm_out", "results")) if d]
sys.path.insert(0, os.path.join(HERE, "..", "P7"))
from common import ci as p7_ci, mutual_info as p7_mi  # read-only import: ci and mutual_info only

FEE = 0.005
GATE_PROXIES = ["entropy", "one_minus_pmax", "masked_peers"]
ALL_PROXIES = GATE_PROXIES + ["mrv_count"]


def npz_path(name):
    for d in TMPDIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def jload(name):
    for d in RESDIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def dump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


# ------------------------------------------------------------------ G1
def per_state_rho(score, target, mask, min_n=3):
    """Within-state Spearman over masked cells, one value per state. States whose target is constant are NaN
    (dropped for every score, so comparisons stay paired); a constant score in a valid state counts as 0."""
    out = np.full(len(score), np.nan)
    for b in range(len(score)):
        m = mask[b]
        if m.sum() < min_n:
            continue
        t = target[b, m]
        if np.all(t == t[0]):
            continue
        s = score[b, m]
        out[b] = 0.0 if np.all(s == s[0]) else S.spearman(s, t)
    return out


def partial_blocks(x, y, z, mask):
    """Within-state ranks (scaled) of x, y and control z over masked cells, returned per state for bootstrapping."""
    blocks = []
    for b in range(len(x)):
        m = mask[b]
        if m.sum() < 3:
            continue
        g = np.zeros(m.sum(), int)
        blocks.append(np.stack([S.within_ranks(x[b, m], g), S.within_ranks(y[b, m], g), S.within_ranks(z[b, m], g)], 1))
    return blocks


def partial_from_blocks(blocks):
    a = np.concatenate(blocks)
    Z = np.column_stack([np.ones(len(a)), a[:, 2]])
    rx = a[:, 0] - Z @ np.linalg.lstsq(Z, a[:, 0], rcond=None)[0]
    ry = a[:, 1] - Z @ np.linalg.lstsq(Z, a[:, 1], rcond=None)[0]
    return S.pearson(rx, ry)


def centre(G, M):
    Mf = M.astype(float)
    return np.where(M, G - (G * Mf).sum(1, keepdims=True) / np.maximum(Mf.sum(1, keepdims=True), 1), 0.0)


def lin_proxy(z, split):
    """Extra baseline (not in the gate): least-squares combination of the four free proxies, fitted on train masked
    cells to the within-state centred gain, scored on the split."""
    Ftr = np.stack([z[f"prox_tr_{k}"] for k in ALL_PROXIES], -1)
    Mtr = z["mtr"]
    Ftc = np.stack([centre(Ftr[..., i], Mtr) for i in range(Ftr.shape[-1])], -1)
    w = np.linalg.lstsq(Ftc[Mtr], centre(z["G_tr"], Mtr)[Mtr], rcond=None)[0]
    return np.stack([z[f"prox_{split}_{k}"] for k in ALL_PROXIES], -1) @ w


def g1():
    seeds = sorted({int(os.path.basename(p)[7:-4]) for d in TMPDIRS for p in glob.glob(os.path.join(d, "s4_seed*.npz"))
                    if os.path.basename(p)[7:-4].isdigit() and int(os.path.basename(p)[7:-4]) < 50})
    per = {}
    for s in seeds:
        z = np.load(npz_path(f"s4_seed{s}.npz"))
        mt, G, v = z["mt"], z["G_te"], z["v_te"]
        # proxy sign chosen on train (within-state rho with the train gain)
        sign = {}
        for k in ALL_PROXIES:
            r = per_state_rho(z[f"prox_tr_{k}"][:1500], z["G_tr"][:1500], z["mtr"][:1500])
            sign[k] = 1.0 if np.nanmean(r) >= 0 else -1.0
        sc = {"head": v, **{k: sign[k] * z[f"prox_te_{k}"] for k in ALL_PROXIES}}
        sc["lin_proxies"] = lin_proxy(z, "te")
        rho = {k: per_state_rho(val, G, mt) for k, val in sc.items()}
        ok = ~np.isnan(rho["head"])
        rho = {k: r[ok] for k, r in rho.items()}
        rho_I = {k: np.nanmean(per_state_rho(val, z["Iex"], mt)) for k, val in sc.items()}
        rho_I["G_target"] = float(np.nanmean(per_state_rho(G, z["Iex"], mt)))
        pooled = {k: S.spearman(val[mt], G[mt]) for k, val in sc.items()}
        H = z["prox_te_entropy"]
        blocks = partial_blocks(v, G, H, mt)
        pb = partial_from_blocks(blocks)
        rng = np.random.default_rng(s)
        boots = [partial_from_blocks([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]) for _ in range(300)]
        # partial controlling for entropy and MRV count (secondary)
        per[s] = {"rho": rho, "sign": sign, "rho_vs_Iex": rho_I, "pooled": pooled, "partial_entropy": pb,
                  "partial_boot": np.array(boots), "n_states": int(ok.sum()),
                  "partial_entropy_mrv": S.partial_spearman(v[mt], G[mt], np.column_stack([H[mt], z["prox_te_mrv_count"][mt]]))}
    if not per:
        print("no S4 seeds found")
        return None
    names = list(next(iter(per.values()))["rho"].keys())
    mean_rho = {k: S.tci([per[s]["rho"][k].mean() for s in per]) for k in names}
    best = max(GATE_PROXIES, key=lambda k: np.mean([per[s]["rho"][k].mean() for s in per]))
    diffs = [per[s]["rho"]["head"] - per[s]["rho"][best] for s in per]
    hb = S.hier_boot(diffs, B=4000)
    seed_d = [d.mean() for d in diffs]
    pe = [per[s]["partial_entropy"] for s in per]
    pboot = np.mean([per[s]["partial_boot"] for s in per], 0)
    diffs_lin = [per[s]["rho"]["head"] - per[s]["rho"]["lin_proxies"] for s in per] if "lin_proxies" in names else None
    res = {
        "seeds": list(per), "n_states": {s: per[s]["n_states"] for s in per},
        "within_state_spearman_vs_G": mean_rho,
        "per_seed_within_state_spearman": {s: {k: float(per[s]["rho"][k].mean()) for k in names} for s in per},
        "proxy_sign_from_train": {s: per[s]["sign"] for s in per},
        "best_free_proxy": best,
        "head_minus_best_proxy": {"mean": hb["mean"], "hier_boot_lo": hb["lo"], "hier_boot_hi": hb["hi"],
                                  "seed_t": S.tci(seed_d)},
        "head_minus_lin_proxies": (S.hier_boot(diffs_lin, B=2000) if diffs_lin else None),
        "partial_spearman_head_G_given_entropy": {"per_seed": pe, "mean": float(np.mean(pe)),
                                                  "boot_lo": float(np.quantile(pboot, 0.025)),
                                                  "boot_hi": float(np.quantile(pboot, 0.975))},
        "partial_spearman_head_G_given_entropy_mrv": {s: per[s]["partial_entropy_mrv"] for s in per},
        "pooled_spearman_vs_G": {k: float(np.mean([per[s]["pooled"][k] for s in per])) for k in names},
        "within_state_spearman_vs_exact_I": {k: float(np.mean([per[s]["rho_vs_Iex"][k] for s in per]))
                                             for k in per[next(iter(per))]["rho_vs_Iex"]},
        "p7_ci_check": p7_ci(seed_d),
    }
    passed = (hb["mean"] >= 0.1 and hb["lo"] > 0 and res["partial_spearman_head_G_given_entropy"]["mean"] > 0)
    res["G1_pass"] = bool(passed)
    res["G1_rule"] = ("head - best free proxy (within-state Spearman with realised G) >= 0.1, hierarchical-bootstrap "
                      "95% CI (seeds x states) excludes 0, partial Spearman | entropy > 0")
    diag = {}
    for s in seeds:
        try:
            zc = np.load(npz_path(f"s4c_seed{s}.npz"))
        except FileNotFoundError:
            continue
        mt, G = zc["mt"], zc["G_te"]
        sign = {k: (1.0 if np.nanmean(per_state_rho(zc[f"prox_tr_{k}"][:1500], zc["G_tr"][:1500], zc["mtr"][:1500])) >= 0
                    else -1.0) for k in ALL_PROXIES}
        sc = {"head_prereg": zc["v_te"], "head_centred": zc["v_te_c"], "lin_proxies": lin_proxy(zc, "te"),
              **{k: sign[k] * zc[f"prox_te_{k}"] for k in ALL_PROXIES}}
        rr = {k: per_state_rho(v, G, mt) for k, v in sc.items()}
        ok = ~np.isnan(rr["head_centred"])
        diag[s] = {k: r[ok] for k, r in rr.items()}
        diag[s]["_partial"] = partial_from_blocks(partial_blocks(zc["v_te_c"], G, zc["prox_te_entropy"], mt))
        diag[s]["_train_within"] = float(np.nanmean(per_state_rho(zc["v_tr_c"][:2000], zc["G_tr"][:2000], zc["mtr"][:2000])))
    if diag:
        ks = [k for k in next(iter(diag.values())) if not k.startswith("_")]
        bestc = max(GATE_PROXIES, key=lambda k: np.mean([diag[s][k].mean() for s in diag]))
        hbc = S.hier_boot([diag[s]["head_centred"] - diag[s][bestc] for s in diag], B=4000)
        res["posthoc_centred_head"] = {
            "note": "post-hoc diagnostic, not the gate: head trained 10k steps on the within-state centred gain; "
                    "separate B2 retrain per seed (tag s4c)",
            "within_state_spearman_vs_G": {k: S.tci([diag[s][k].mean() for s in diag]) for k in ks},
            "best_free_proxy": bestc, "head_centred_minus_best_proxy": hbc,
            "partial_given_entropy": S.tci([diag[s]["_partial"] for s in diag]),
            "train_within_state_spearman": S.tci([diag[s]["_train_within"] for s in diag]),
            "would_pass_G1_rule": bool(hbc["mean"] >= 0.1 and hbc["lo"] > 0 and np.mean([diag[s]["_partial"] for s in diag]) > 0)}
        print("post-hoc centred head:", json.dumps(res["posthoc_centred_head"], default=float)[:1200])
    dump(res, os.path.join(RES, "g1.json"))
    chart = {"title": "S4 gate G1: within-state Spearman with realised commit gain (mean over seeds, 95% t-interval)",
             "kind": "bar", "x": names,
             "series": [{"name": "within-state Spearman", "y": [mean_rho[k]["mean"] for k in names],
                         "ci95": [mean_rho[k].get("ci95") for k in names]}],
             "note": f"best free proxy = {best}; head - best = {hb['mean']:.3f} [{hb['lo']:.3f}, {hb['hi']:.3f}]"}
    dump(chart, os.path.join(CH, "s4_g1.json"))
    print(json.dumps({k: res[k] for k in ("best_free_proxy", "head_minus_best_proxy", "G1_pass")}, default=float, indent=1))
    print("mean within-state rho:", {k: round(v["mean"], 3) for k, v in mean_rho.items()})
    print("partial | entropy:", res["partial_spearman_head_G_given_entropy"])
    return res


# ------------------------------------------------------------------ S4 readouts
def huber_location(x1, p1, delta, iters=80):
    """Huber M-estimate of location for the two-point law {x1 w.p. p1, 0 w.p. 1-p1} (vectorised bisection)."""
    lo, hi = np.minimum(0, x1), np.maximum(0, x1)
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        f = p1 * np.clip(x1 - m, -delta, delta) + (1 - p1) * np.clip(-m, -delta, delta)
        lo = np.where(f > 0, m, lo)
        hi = np.where(f > 0, hi, m)
    return 0.5 * (lo + hi)


def s4read():
    out = {"void_rule": {}, "joint_nll": {}}
    vr = []
    seen = set()
    for p in sorted(p for d in TMPDIRS for p in glob.glob(os.path.join(d, "s4_seed[0-9].npz"))):
        if os.path.basename(p) in seen:
            continue
        seen.add(os.path.basename(p))
        z = np.load(p)
        D, y, pt, mt, delta = z["D_te"], z["yt"], z["p_te"], z["mt"], float(z["delta"])
        Dy = np.nan_to_num(np.take_along_axis(D, y[..., None], -1)[..., 0][mt])
        py = np.take_along_axis(pt, y[..., None], -1)[..., 0][mt]
        am = (pt.argmax(-1) == y)[mt]
        rec = {"seed": int(os.path.basename(p)[7:-4]), "huber_delta": delta, "var_signal": float(Dy.var()),
               "mean_abs_Delta_y": float(np.abs(Dy).mean())}
        for name, pi in (("eps_greedy", np.where(am, 0.9 + 0.1 / 4, 0.1 / 4)), ("mixture", 0.9 * py + 0.1 / 4)):
            var = Dy ** 2 * (1 / pi - 1)
            eclip = pi * np.clip(Dy / pi, -delta, delta)
            m_h = huber_location(Dy / pi, pi, delta)
            rec[name] = {"mean_IS_variance": float(var.mean()), "variance_ratio_IS_over_signal": float(var.mean() / Dy.var()),
                         "clip_bias_mean": float((eclip - Dy).mean()), "clip_bias_mean_abs": float(np.abs(eclip - Dy).mean()),
                         "huber_M_bias_mean": float((m_h - Dy).mean()), "huber_M_bias_mean_abs": float(np.abs(m_h - Dy).mean()),
                         "huber_M_bias_rel": float(np.abs(m_h - Dy).mean() / np.abs(Dy).mean())}
        vr.append(rec)
    out["void_rule"] = {"per_seed": vr}
    if vr:
        for name in ("eps_greedy", "mixture"):
            out["void_rule"][name] = {k: S.tci([r[name][k] for r in vr]) for k in vr[0][name]}
    js = [j for j in (jload(f"s4_seed{s}.json") for s in range(3)) if j]
    if js:
        H = S.tci([j["joint_nll"]["H_Y_given_X"] for j in js])
        series = []
        keys = [k for k in js[0]["joint_nll"] if k not in ("H_Y_given_X", "ARDM_DP")]
        pts = {k: (np.mean([j["joint_nll"][k]["passes"] for j in js]), S.tci([j["joint_nll"][k]["nll"] for j in js])) for k in keys}
        ardm = sorted(js[0]["joint_nll"]["ARDM_DP"], key=int)
        series.append({"name": "ARDM DP schedule", "x": [js[0]["joint_nll"]["ARDM_DP"][r]["passes"] for r in ardm],
                       "y": [np.mean([j["joint_nll"]["ARDM_DP"][r]["nll_expected"] for j in js]) for r in ardm]})
        for k in keys:
            series.append({"name": k, "x": [pts[k][0]], "y": [pts[k][1]["mean"]], "ci95": [pts[k][1].get("ci95")]})
        series.append({"name": "exact H(Y|X)", "x": [0, 64], "y": [H["mean"], H["mean"]]})
        out["joint_nll"] = {"H_Y_given_X": H, "decoders": {k: {"passes": pts[k][0], "nll": pts[k][1]} for k in keys},
                            "ARDM_DP": {r: {"passes": js[0]["joint_nll"]["ARDM_DP"][r]["passes"],
                                            "nll": S.tci([j["joint_nll"]["ARDM_DP"][r]["nll_expected"] for j in js])} for r in ardm},
                            "fit": {k: S.tci([j["fit"][k] for j in js]) for k in js[0]["fit"]}}
        dump({"title": "S4 joint NLL of the true grid (nats) vs passes; exact H(Y|X) is the floor",
              "kind": "scatter-line", "xlabel": "passes", "ylabel": "joint NLL (nats)", "series": series},
             os.path.join(CH, "s4_joint_nll.json"))
    dump(out, os.path.join(RES, "s4_readouts.json"))
    print(json.dumps({k: out["void_rule"].get(k) for k in ("eps_greedy", "mixture")}, default=float)[:1500])
    return out


# ------------------------------------------------------------------ S6 pilot / T rule
def pilot():
    rows = {T: jload(f"s6_pilot_T{T}_seed0.json") for T in (1, 2)}
    thr = max(0.05, 10 * FEE)
    ok = [T for T in (1, 2) if rows[T] and rows[T]["nll_per_cell"] >= thr]
    T = min(ok) if ok else 1
    res = {"threshold_nats_per_cell": thr, "pilots": rows, "T_chosen": T,
           "rule": "smallest T in {1,2} with B2-reg (2T passes) held-out per-cell NLL >= max(0.05, 10 x fee)",
           "rule_met": bool(ok)}
    dump(res, os.path.join(RES, "s6_T_choice.json"))
    print(json.dumps(res, default=float, indent=1))
    return res


# ------------------------------------------------------------------ M2
def m2(T=None, auc="auction", reg="reg2", tag=""):
    """auc: auction arm name; reg: the B2-reg control (reg2 = phase-distinct constants, the compute-matched control;
    reg = the same constant in both phases, kept as a diagnostic)."""
    T = T or (jload("s6_T_choice.json") or {}).get("T_chosen", 1)
    arms = (auc, "cvae", reg, "reg")
    runs = {arm: {s: jload(f"m2_{arm}_T{T}_seed{s}.json") for s in range(5)} for arm in arms}
    runs = {a: {s: r for s, r in d.items() if r} for a, d in runs.items()}
    A, Cv, Rg = runs[auc], runs["cvae"], runs[reg]
    seeds = sorted(set(A) & set(Rg))
    if not seeds:
        print("no M2 runs")
        return None
    slots = list(A[seeds[0]]["test_royalty"])
    learned = [n for n in slots if n.startswith("learned")]
    surv = {n: sum(A[s]["listed_final"][n] for s in seeds) for n in slots}
    G = {n: S.tci([A[s]["test_royalty"][n]["G"] for s in seeds]) for n in slots}
    # (ii) learned slot above fee with CI above 0, per seed
    ii = []
    for s in seeds:
        tr = A[s]["test_royalty"]
        ii.append(any(A[s]["listed_final"][n] and tr[n]["G"] > FEE and tr[n]["G_ci"]["lo"] > 0 for n in learned))
    # (iii) end-to-end vs B2-reg, paired per example within seed
    e2e, e2e_pooled = [], []
    for s in seeds:
        pa = next((os.path.join(d, f"m2_{auc}_T{T}_s{s}_nll.npy") for d in RESDIRS
                   if os.path.exists(os.path.join(d, f"m2_{auc}_T{T}_s{s}_nll.npy"))), "")
        pr = next((os.path.join(d, f"m2_{reg}_T{T}_s{s}_nll.npy") for d in RESDIRS
                   if os.path.exists(os.path.join(d, f"m2_{reg}_T{T}_s{s}_nll.npy"))), "")
        if os.path.exists(pa) and os.path.exists(pr):
            d = np.load(pr) - np.load(pa)
            e2e_pooled.append(d)
        e2e.append(Rg[s]["test_nll_per_cell"] - A[s]["test_nll_ehat_per_cell"])
    iii = {"gain_per_cell_seed_t": S.tci(e2e), "hier_boot": S.hier_boot(e2e_pooled, B=2000) if e2e_pooled else None}
    iii["pass"] = bool(iii["gain_per_cell_seed_t"]["n"] > 1 and iii["gain_per_cell_seed_t"]["lo"] > 0)
    # (iv) controls de-listed
    iv = {n: len(seeds) - surv[n] for n in ("copyhash", "trivial", "random")}
    # (v) surviving learned slot NMI with a non-cell atom >= 0.5 and > its NMI with any cell atom
    v_hits = []
    for s in seeds:
        for r in A[s]["nmi"]["live"]:
            if r["listed"] and r["max_nmi_noncell"] >= 0.5 and r["max_nmi_noncell"] > r["max_nmi_cell"]:
                v_hits.append({"seed": s, **r})
    nmi_tab = {grp: {"noncell": S.tci([r["max_nmi_noncell"] for s in seeds for r in A[s]["nmi"][grp]]),
                     "cell": S.tci([r["max_nmi_cell"] for s in seeds for r in A[s]["nmi"][grp]])}
               for grp in ("live", "dead", "init")}
    # (vi) auction surviving G vs cvae
    cs = sorted(set(seeds) & set(Cv))
    dvi = [A[s]["surviving_G_total"] - Cv[s]["surviving_G_total"] for s in cs]
    vi = {"auction": S.tci([A[s]["surviving_G_total"] for s in cs]), "cvae": S.tci([Cv[s]["surviving_G_total"] for s in cs]),
          "diff": S.tci(dvi)}
    def lsum(r):
        return sum(r["test_royalty"][n]["G"] for n in learned if r["listed_final"][n])
    dl = [lsum(A[s]) - lsum(Cv[s]) for s in cs]
    vi["learned_only"] = {"auction": S.tci([lsum(A[s]) for s in cs]), "cvae": S.tci([lsum(Cv[s]) for s in cs]),
                          "diff": S.tci(dl)}
    vi["pass_geq"] = bool(dvi and np.mean(dvi) >= 0)
    vi["strict_ci_above_0"] = bool(len(dvi) > 1 and vi["diff"]["lo"] > 0)
    cv_e2e = S.tci([Rg[s]["test_nll_per_cell"] - Cv[s]["test_nll_ehat_per_cell"] for s in cs])
    # copy index and triviality per slot
    extra = {}
    for n in slots:
        ci_, tv = [], []
        for s in seeds:
            r = A[s]["test_royalty"][n]
            gvc = r["gain_vs_const"]
            ci_.append(r["rate"] / gvc if gvc > 1e-6 else float("nan"))
            tv.append(r["G"] / Rg[s]["test_nll_per_cell"])
        extra[n] = {"copy_index": S.tci(ci_), "triviality": S.tci(tv),
                    "gain_vs_const": S.tci([A[s]["test_royalty"][n]["gain_vs_const"] for s in seeds]),
                    "rate": S.tci([A[s]["test_royalty"][n]["rate"] for s in seeds])}
    # sanity: NMI numerator equals P7 mutual_info on a toy pair
    rng = np.random.default_rng(0)
    a_, b_ = rng.integers(0, 2, 1000), rng.integers(0, 2, 1000)
    Rs = runs["reg"]
    phase_signal = S.tci([Rs[s]["test_nll_per_cell"] - Rg[s]["test_nll_per_cell"] for s in sorted(set(Rs) & set(Rg))]) \
        if reg != "reg" else None
    res = {
        "auction_arm": auc, "control_arm": reg, "T": T, "fee": FEE, "seeds": seeds,
        "reg_same_const_minus_control_nll_per_cell": phase_signal,
        "cvae_minus_control_nll_per_cell": S.tci([Cv[s]["test_nll_ehat_per_cell"] - Rg[s]["test_nll_per_cell"] for s in cs])
        if (cs := sorted(set(seeds) & set(Cv))) else None, "survival_counts": surv, "test_G_per_slot": G, "slot_extra": extra,
        "S-B(i)_planted_survives_5of5": surv.get("planted", 0) == len(seeds) and len(seeds) == 5,
        "S-B(ii)_learned_above_fee_seeds": int(sum(ii)), "S-B(ii)_pass": int(sum(ii)) >= 4,
        "S-B(iii)_end_to_end": iii,
        "S-B(iv)_delisted_counts": iv, "S-B(iv)_pass": all(c >= 4 for c in iv.values()),
        "S-B(v)_hits": v_hits, "S-B(v)_pass": bool(v_hits), "nmi": nmi_tab,
        "S-B(vi)": vi, "cvae_end_to_end_gain_per_cell": cv_e2e,
        "cvae_pruned": {s: Cv[s].get("cvae_pruned") for s in cs},
        "reg_test_nll_per_cell": S.tci([Rg[s]["test_nll_per_cell"] for s in seeds]),
        "auction_logs_respawns": {s: len(A[s]["log"]["respawns"]) for s in seeds},
        "books": {s: A[s].get("books") for s in seeds},
        "p7_mutual_info_check": float(p7_mi(a_, b_)),
    }
    res["F3b_planted_delisted"] = surv.get("planted", 0) < len(seeds)
    res["F3_no_learned_above_fee"] = (not res["F3b_planted_delisted"]) and (int(sum(ii)) == 0 or not v_hits)
    res["F4_auction_not_above_cvae"] = not vi["strict_ci_above_0"]
    dump(res, os.path.join(RES, f"m2_summary{tag}.json"))
    dump({"title": f"M2 held-out royalty G_k per slot (nats/cell), T={T}, mean over seeds with 95% t-interval; fee line",
          "kind": "bar", "x": slots, "series": [{"name": "G_k", "y": [G[n]["mean"] for n in slots],
                                                   "ci95": [G[n].get("ci95") for n in slots]}],
          "hline": {"name": "fee", "y": FEE}, "survival": surv}, os.path.join(CH, f"m2_royalties{tag}.json"))
    traj = []
    for s in seeds:
        for rec in A[s]["log"]["auctions"]:
            traj.append({"seed": s, "step": rec["step"], "G": rec["G"], "listed": rec["listed"]})
    dump({"title": "M2 auction: royalty per slot at each auction (held-out val batch)", "kind": "lines",
          "slots": slots, "points": traj}, os.path.join(CH, f"m2_auction_traj{tag}.json"))
    print(json.dumps({k: res[k] for k in res if k.startswith("S-B") or k.startswith("F")}, default=float, indent=1)[:3000])
    return res


def m2_arms(T=1):
    """Post-hoc tables: per-arm test NLL (all 2T-pass models) and the fixed-cell positive control (cvae_fix14)."""
    arms = ("auction", "auction_g5000", "cvae", "reg", "reg2", "cvae_fix14", "auction_g5000_fix14")
    tab = {}
    for a in arms:
        rs = [r for r in (jload(f"m2_{a}_T{T}_seed{s}.json") for s in range(5)) if r]
        if not rs:
            continue
        v = [r.get("test_nll_ehat_per_cell", r.get("test_nll_per_cell")) for r in rs]
        tab[a] = {"per_seed": v, "tci": S.tci(v), "seeds_below_0.6": int(sum(x < 0.6 for x in v)), "n": len(v)}
    fx = [r for r in (jload(f"m2_cvae_fix14_T{T}_seed{s}.json") for s in range(5)) if r]
    ctrl = None
    if fx:
        ctrl = {"slot": "planted = [y_14 <= 3] (fixed cell)",
                **{k: S.tci([r["test_royalty"]["planted"][k] for r in fx]) for k in ("gain_resample", "gain_vs_const", "rate", "G", "p1")},
                "teacher_minus_ehat_nll": S.tci([r["test_nll_teacher_per_cell"] - r["test_nll_ehat_per_cell"] for r in fx]),
                "note": "gain_resample ~ rate means the decoder reads the slot; ~0 means it ignores it. At Bayes a cell bit "
                        "has gain = rate (G = 0), so even a readable cell bit cannot clear the fee."}
    res = {"T": T, "test_nll_per_cell_by_arm": tab, "fixed_cell_positive_control": ctrl}
    dump(res, os.path.join(RES, "m2_arms_posthoc.json"))
    dump({"title": f"M2 test NLL per cell by arm (T={T}, 2T passes each); dots = seeds", "kind": "strip",
          "x": list(tab), "points": {a: tab[a]["per_seed"] for a in tab}}, os.path.join(CH, "m2_arm_nll.json"))
    print(json.dumps(res, default=float, indent=1)[:2500])
    return res


if __name__ == "__main__":
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CH, exist_ok=True)
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("g1", "all"):
        g1()
    if w in ("s4read", "all"):
        s4read()
    if w in ("pilot", "all"):
        pilot()
    if w in ("m2", "all"):
        T = int(sys.argv[2]) if len(sys.argv) > 2 else None
        m2(T)
        if jload(f"m2_auction_g5000_T{T or 1}_seed0.json"):
            m2(T, auc="auction_g5000", tag="_g5000")
        m2(T, reg="reg", tag="_regsame")
        if jload(f"m2_auction_g5000_fix14_T{T or 1}_seed0.json"):
            m2(T, auc="auction_g5000_fix14", tag="_fix14")
        m2_arms(T or 1)
