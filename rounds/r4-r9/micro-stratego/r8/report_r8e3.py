#!/usr/bin/env python3
"""R8-E3 report: applies the pre-registered E3-D1 and E3-D2 rules (PREREG-R8-E3.md section 2) to the synced per-seed
files r8/runs/s<seed>/eval/*.json, re-derives headline numbers from the per-item CSVs (self-check), and writes
r8/report/verdicts.csv, tables.md, summary.json, selfcheck.csv and figures (PNG + CSV). Seed-t 95% intervals (df = 2).
   python3 r8/report_r8e3.py [--runs DIR]"""
import argparse
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.join(HERE, "report")
SEEDS = (101, 102, 103)
T975 = 4.302652729749464
KS = (0, 1, 2, 4, 8, 16)
AGGS = ("raw_count", "union_t", "scaled_mix", "ataraxos_mmd", "ataraxos_mmd_fit", "kc_dedup_K3c", "attn_L3_K3c",
        "looped_attn_T3_K3c")
RUNS = os.path.join(HERE, "runs")


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = T975 * x.std(ddof=1) / math.sqrt(len(x))
    return m, m - h, m + h


def J(seed, part):
    p = os.path.join(RUNS, f"s{seed}", "eval", f"{part}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def C(seed, name):
    p = os.path.join(RUNS, f"s{seed}", "eval", name)
    with open(p) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]}


def per(part, f):
    return np.array([f(J(s, part)) if J(s, part) is not None else float("nan") for s in SEEDS], float)


def fmt(x, d=4):
    return "nan" if not np.isfinite(x) else f"{x:.{d}f}"


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    RUNS = ap.parse_args().runs
    os.makedirs(REP, exist_ok=True)
    rows, checks = [], []

    def add(rule, crit, vals, thr, passed, note=""):
        m, lo, hi = tci(vals)
        rows.append({"rule": rule, "criterion": crit, "per_seed": ", ".join(fmt(v) for v in vals), "mean": fmt(m),
                     "ci95": f"[{fmt(lo)}, {fmt(hi)}]", "threshold": thr, "pass": passed, "note": note})

    def vrow(rule, crit, passed):
        rows.append({"rule": rule, "criterion": crit, "per_seed": "", "mean": "", "ci95": "", "threshold": "all",
                     "pass": passed, "note": ""})

    kl = lambda g, K: per("d1", lambda j: j[g][f"K{K}"]["kl"])
    js = lambda g, K: per("d1", lambda j: j[g][f"K{K}"]["jsd"])
    # ---------------- D1 (1)
    ok1 = []
    for g in ("raw_count", "ataraxos_mmd"):
        d = kl(g, 16) - kl(g, 0)
        ok = bool(np.mean(d) > 0.010 and tci(d)[1] > 0)
        ok1.append(ok)
        add("E3-D1 (1)", f"{g}{' (gamma 0.25)' if g == 'ataraxos_mmd' else ''}: KL(16) - KL(0)", d, "> 0.010, CI > 0", ok,
            "KL(0) " + ", ".join(fmt(x) for x in kl(g, 0)) + "; KL(16) " + ", ".join(fmt(x) for x in kl(g, 16)))
    d1a = all(ok1)
    vrow("E3-D1 (1)", "both naive baselines degrade the anchor", d1a)
    # ---------------- D1 (2)
    ok2 = []
    for g, full in (("attn_L3_K3c", True), ("looped_attn_T3_K3c", False)):
        mx = np.max(np.stack([kl(g, K) - kl(g, 0) for K in KS[1:]]), 0)
        a_ = bool(np.all(mx <= 0.0005))
        ok2.append(a_)
        add("E3-D1 (2)", f"{g}: max over K >= 1 of KL(K) - KL(0), every seed", mx, "<= 0.0005 in every seed", a_,
            "per K (3-seed means): " + ", ".join(f"K{K} {np.mean(kl(g, K) - kl(g, 0)):+.5f}" for K in KS[1:]))
        d = kl(g, 16) - kl(g, 0)
        b_ = bool(tci(d)[2] < 0)
        ok2.append(b_)
        add("E3-D1 (2)", f"{g}: KL(16) - KL(0)", d, "CI < 0", b_)
        if not full:
            continue
        ja = js(g, 16)
        for Xn in ("scaled_mix", "ataraxos_mmd"):
            jx = js(Xn, 16)
            ok = bool(np.mean(jx / np.maximum(ja, 1e-30)) >= 8.0 and tci(jx - ja)[1] > 0)
            ok2.append(ok)
            m, lo, hi = tci(jx - ja)
            add("E3-D1 (2)", f"K = 16 JSD ratio {Xn} / {g}", jx / np.maximum(ja, 1e-30), ">= 8, CI(diff) > 0", ok,
                f"JSD {fmt(np.mean(jx), 6)} vs {fmt(np.mean(ja), 6)}; diff {m:.6f} [{lo:.6f}, {hi:.6f}]")
        ka = kl(g, 16)
        for Xn in ("scaled_mix", "ataraxos_mmd", "kc_dedup_K3c"):
            kx = kl(Xn, 16)
            ok = bool(tci(kx - ka)[1] > 0)
            ok2.append(ok)
            add("E3-D1 (2)", f"K = 16 KL to P*_full: {Xn} - {g}", kx - ka, "> 0, CI > 0", ok)
    d1b = all(ok2)
    vrow("E3-D1 (2)", "attn_L3-K3c (a)-(d) and looped_attn_T3-K3c (a), (b)", d1b)
    d1 = d1a and d1b
    vrow("E3-D1", "(1) and (2)", d1)
    # ---------------- D2 (1)
    m1 = lambda k: per("d2", lambda j: j[f"clean_{k}"]["mse"])
    b1 = lambda k: per("d2", lambda j: j[f"clean_{k}"]["bias2"])
    mu, ms, mt, mc = m1("unif"), m1("stiff2"), m1("trust_pool"), m1("cons_pool")
    c_a = bool(tci(mu - mc)[1] > 0)
    add("E3-D2 (1)", "clean bags: MSE(unif_pool) - MSE(cons_pool)", mu - mc, "> 0, CI > 0", c_a,
        f"MSE unif {fmt(np.mean(mu), 5)}, cons {fmt(np.mean(mc), 5)}")
    rr = 1 - mc / ms
    c_b = bool(tci(ms - mc)[1] > 0 and np.mean(rr) >= 0.60)
    m, lo, hi = tci(ms - mc)
    add("E3-D2 (1)", "clean bags: relative MSE cut of cons_pool vs stiff_pool", rr, ">= 0.60, CI(diff) > 0", c_b,
        f"MSE stiff {fmt(np.mean(ms), 5)}; diff {m:.5f} [{lo:.5f}, {hi:.5f}]")
    c_c = bool(np.mean(mc) < np.mean(mt))
    add("E3-D2 (1)", "clean bags: MSE(trust_pool) - MSE(cons_pool)", mt - mc, "> 0 (3-seed mean)", c_c)
    bc = b1("cons_pool")
    c_d = []
    for k in ("unif", "stiff2", "trust_pool"):
        ok = bool(np.mean(b1(k)) > np.mean(bc))
        c_d.append(ok)
        add("E3-D2 (1)", f"clean bags: Bias^2({k}) - Bias^2(cons_pool)", b1(k) - bc, "> 0 (3-seed mean)", ok,
            f"Bias^2 {fmt(np.mean(b1(k)), 5)} vs {fmt(np.mean(bc), 5)}")
    d2a = c_a and c_b and c_c and all(c_d)
    vrow("E3-D2 (1)", "all of the above", d2a)
    # ---------------- D2 (2a)
    st, ss = per("d2", lambda j: j["spearman_trust"]), per("d2", lambda j: j["spearman_stiff"])
    wt = per("d2", lambda j: j["within_trust"])
    tt, ts = per("d2", lambda j: j["strata_trust"]), per("d2", lambda j: j["strata_stiff"])
    r_ = [bool(np.mean(st) >= 0.40 and tci(st)[1] > 0), bool(np.mean(wt) >= 0.40 and tci(wt)[1] > 0),
          bool(np.mean(st - ss) >= 0.60 and tci(st - ss)[1] > 0), bool(np.mean(tt - ts) >= 0.20 and tci(tt - ts)[1] > 0)]
    add("E3-D2 (2a)", "Spearman(beta, -KL) overall, mixed bags", st, ">= 0.40, CI > 0", r_[0],
        "-H " + ", ".join(fmt(x) for x in ss))
    add("E3-D2 (2a)", "same, within (s, a) pairs", wt, ">= 0.40, CI > 0", r_[1],
        "-H within " + ", ".join(fmt(x) for x in per("d2", lambda j: j["within_stiff"])))
    add("E3-D2 (2a)", "Spearman(beta) - Spearman(-H) overall", st - ss, ">= 0.60, CI > 0", r_[2])
    add("E3-D2 (2a)", "same, within 10 equal-count entropy strata", tt - ts, ">= 0.20, CI > 0", r_[3])
    d2b = all(r_)
    vrow("E3-D2 (2a)", "all four", d2b)
    # ---------------- D2 (2b)
    m2 = lambda k: per("d2", lambda j: j[f"mixed_{k}"]["mse"])
    xu, xs = m2("unif"), m2("stiff2")
    oks = []
    for Xn in ("trust_gate", "cons_trust_pool"):
        mx_ = m2(Xn)
        r1 = 1 - mx_ / xu
        ok = bool(np.mean(r1) >= 0.20 and tci(xu - mx_)[1] > 0)
        oks.append(ok)
        m, lo, hi = tci(xu - mx_)
        add("E3-D2 (2b)", f"mixed bags: relative MSE cut of {Xn} vs unif_pool", r1, ">= 0.20, CI(diff) > 0", ok,
            f"MSE {fmt(np.mean(mx_), 5)} vs unif {fmt(np.mean(xu), 5)}; diff {m:.5f} [{lo:.5f}, {hi:.5f}]")
        r2 = 1 - mx_ / xs
        ok = bool(np.mean(r2) >= 0.60 and tci(xs - mx_)[1] > 0)
        oks.append(ok)
        m, lo, hi = tci(xs - mx_)
        add("E3-D2 (2b)", f"mixed bags: relative MSE cut of {Xn} vs stiff_pool", r2, ">= 0.60, CI(diff) > 0", ok,
            f"MSE stiff {fmt(np.mean(xs), 5)}; diff {m:.5f} [{lo:.5f}, {hi:.5f}]")
    cu, cct = m1("unif"), m1("cons_trust_pool")
    ok = bool(np.all(cct <= cu))
    oks.append(ok)
    add("E3-D2 (2b)", "clean bags: MSE(cons_trust_pool) - MSE(unif_pool), every seed", cct - cu, "<= 0 in every seed", ok)
    d2c = all(oks)
    vrow("E3-D2 (2b)", "all of the above", d2c)
    d2 = d2a and d2b and d2c
    vrow("E3-D2", "(1) and (2a) and (2b)", d2)
    S = {"rules": {"D1": d1, "D1(1)": d1a, "D1(2)": d1b, "D2": d2, "D2(1)": d2a, "D2(2a)": d2b, "D2(2b)": d2c}}
    # ---------------- self-check from the per-item CSVs
    from scipy.stats import spearmanr
    for s in SEEDS:
        if J(s, "d1") is None:
            continue
        d1j = J(s, "d1")
        c1 = C(s, "d1_positions.csv")
        for g in AGGS:
            for K in KS:
                checks.append((f"d1 kl {g} K{K}", s, d1j[g][f"K{K}"]["kl"], c1[f"kl_{g}_K{K}"].mean()))
                checks.append((f"d1 jsd {g} K{K}", s, d1j[g][f"K{K}"]["jsd"], c1[f"jsd_{g}_K{K}"].mean()))
        d2j = J(s, "d2")
        c2 = C(s, "d2_rollouts.csv")
        mx = c2["in_mixed"] == 1
        checks.append(("d2 spearman trust (mixed)", s, d2j["spearman_trust"], spearmanr(c2["beta"][mx], -c2["kl"][mx]).correlation))
        ca = C(s, "d2_actions.csv")
        for k in ("clean_unif", "clean_stiff2", "clean_trust_pool", "clean_cons_pool", "clean_cons_trust_pool", "mixed_unif",
                  "mixed_stiff2", "mixed_trust_gate", "mixed_cons_trust_pool"):
            checks.append((f"d2 mse {k}", s, d2j[k]["mse"], float(((ca[f"q_{k}"] - ca["qstar_eff"]) ** 2).mean())))
    S["selfcheck_max_abs_diff"] = max([abs(a - b) for _, _, a, b in checks] or [float("nan")])
    with open(os.path.join(REP, "verdicts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REP, "selfcheck.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "seed", "json_summary", "rederived_from_csv", "abs_diff"])
        for q, s, a, b in checks:
            w.writerow([q, s, f"{a:.8g}", f"{b:.8g}", f"{abs(a - b):.3g}"])
    # ---------------- secondary numbers
    sec = {}
    for g in AGGS:
        for K in KS:
            sec[f"d1.{g}.K{K}.kl"] = kl(g, K).tolist()
            sec[f"d1.{g}.K{K}.jsd"] = js(g, K).tolist()
    for k in ("kl_anchor_t4", "kl_routeb_all", "mean_nL_L", "mean_nL_R"):
        sec["d1." + k] = per("d1", lambda j, k=k: j[k]).tolist()
    for k in ("alpha", "eta_fit", "u", "cal_ce_anchor", "cal_ce_K16_attn_L3_K3c", "cal_ce_K16_looped_attn_T3_K3c",
              "cal_ce_K16_scaled_mix", "cal_ce_K16_raw_count", "n_params_attn_L3_K3c", "n_params_looped_attn_T3_K3c"):
        sec["fit." + k] = per("fit", lambda j, k=k: j[k]).tolist()
    for pref in ("clean", "mixed"):
        ks = ("unif", "stiff2", "stiff_fit", "trust_pool", "cons_pool", "cons_trust_pool", "cons_tau1") if pref == "clean" else \
            ("unif", "stiff2", "trust_gate", "cons_trust_pool", "cons_pool", "trust_pool", "clean10")
        for k in ks:
            for m_ in ("mse", "noise", "bias2", "neff", "delusional"):
                sec[f"d2.{pref}_{k}.{m_}"] = per("d2", lambda j, k=k, m_=m_, pref=pref: j[f"{pref}_{k}"][m_]).tolist()
    for k in ("spearman_trust", "spearman_stiff", "spearman_trust_r5head", "spearman_cons", "within_trust", "within_stiff",
              "within_cons", "strata_trust", "strata_stiff", "gate_dropped_noisy_share", "n_pairs"):
        sec["d2." + k] = per("d2", lambda j, k=k: j[k]).tolist()
    for tn in ("clean", "noisy_myopic", "noisy_tau35"):
        for k in ("kl", "sq_err", "terminal_like", "beta", "beta_r5", "ll", "cone_lp", "ply_lp", "flag_end", "H", "offsup_per_move"):
            sec[f"d2.tier_{tn}.{k}"] = per("d2", lambda j, tn=tn, k=k: j[f"tier_{tn}"][k]).tolist()
    for k in ("tau_cons", "alpha_trust", "M_gate", "tau_ct", "alpha_ct", "gamma"):
        sec["sel." + k] = per("trust", lambda j, k=k: j["selected"][k]).tolist()
    sec["trust.train_spearman_beta_neg_target"] = per("trust", lambda j: j["train_spearman_beta_neg_target"]).tolist()
    for i, tn in enumerate(("clean", "noisy_myopic", "noisy_tau35")):
        sec[f"trust.target_mean_{tn}"] = per("trust", lambda j, i=i: j["target_mean_by_tier"][i]).tolist()
    S["secondary"] = sec
    json.dump(S, open(os.path.join(REP, "summary.json"), "w"), indent=1, default=str)
    lines = ["| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule']} | {r['criterion']} | {r['per_seed']} | {r['mean']} | {r['ci95']} | {r['threshold']} | {r['pass']} | {r['note']} |"
              for r in rows]
    lines += ["", f"Self-check: {len(checks)} numbers re-derived from the per-item CSVs; largest |difference| "
              f"{S['selfcheck_max_abs_diff']:.3g} (r8/report/selfcheck.csv).", "", "| quantity | per seed | mean [95% CI] |", "|---|---|---|"]
    for k, v in sec.items():
        m, lo, hi = tci(v)
        lines.append(f"| {k} | {', '.join(fmt(float(x)) for x in v)} | {fmt(m)} [{fmt(lo)}, {fmt(hi)}] |")
    open(os.path.join(REP, "tables.md"), "w").write("\n".join(lines) + "\n")
    figures(sec)
    print(json.dumps({"rules": S["rules"], "selfcheck": S["selfcheck_max_abs_diff"]}))


def figures(sec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for g, c in (("attn_L3_K3c", "#c64"), ("looped_attn_T3_K3c", "#d96"), ("scaled_mix", "#68a"), ("kc_dedup_K3c", "#6a8"),
                 ("union_t", "#888")):
        v = [tci(sec[f"d1.{g}.K{K}.kl"]) for K in KS]
        ax[0].errorbar(range(len(KS)), [x[0] for x in v], yerr=[x[0] - x[1] if np.isfinite(x[1]) else 0 for x in v], label=g,
                       color=c, marker="o", ms=3, capsize=2)
        out += [["d1", f"{g} KL K{K}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for K, x in zip(KS, v)]
    ax[0].set_xticks(range(len(KS)))
    ax[0].set_xticklabels([str(K) for K in KS])
    ax[0].set_xlabel("rollouts K")
    ax[0].set_title("E3-D1: KL to P*_full, anchored at the current-turn belief (K = 0)", fontsize=8.5)
    ax[0].legend(fontsize=7)
    for g, c in (("raw_count", "#a5c"), ("ataraxos_mmd", "#b44")):
        v = [tci(sec[f"d1.{g}.K{K}.kl"]) for K in KS]
        ax[1].plot(range(len(KS)), [x[0] for x in v], label=g + (" (gamma 0.25)" if g == "ataraxos_mmd" else ""), color=c, marker="o")
        out += [["d1", f"{g} KL K{K}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for K, x in zip(KS, v)]
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(KS)))
    ax[1].set_xticklabels([str(K) for K in KS])
    ax[1].set_xlabel("rollouts K")
    ax[1].set_title("E3-D1 (1): naive baselines (log scale)", fontsize=8.5)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e8_d1.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    for i, (pref, ks) in enumerate((("clean", ("unif", "stiff2", "trust_pool", "cons_pool", "cons_trust_pool")),
                                    ("mixed", ("unif", "stiff2", "trust_gate", "cons_trust_pool", "clean10")))):
        v = [tci(sec[f"d2.{pref}_{k}.mse"]) for k in ks]
        ax[i].bar(range(len(ks)), [x[0] for x in v], yerr=[x[0] - x[1] if np.isfinite(x[1]) else 0 for x in v], capsize=3)
        ax[i].set_xticks(range(len(ks)))
        ax[i].set_xticklabels(ks, fontsize=7.5, rotation=12)
        ax[i].set_title(f"E3-D2: action-value MSE vs Q*, {pref} bags", fontsize=8.5)
        out += [["d2", f"{pref} {k} mse", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for k, x in zip(ks, v)]
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e8_d2.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(REP, "figures.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "series", "mean", "ci_lo", "ci_hi"])
        w.writerows(out)


if __name__ == "__main__":
    main()
