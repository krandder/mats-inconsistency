#!/usr/bin/env python3
"""R7-E3 report: applies the pre-registered E3-D0..D2 rules (PREREG-R7-E3.md section 2) to the synced per-seed files
r7/runs/s<seed>/eval/*.json, re-derives headline numbers from the per-item CSVs (self-check), and writes
r7/report/verdicts.csv, tables.md, summary.json, selfcheck.csv and figures (PNG + CSV). Seed-t 95% intervals (df = 2).
   python3 r7/report_r7e3.py [--runs DIR]"""
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
HEADS = ("K3", "K3c")
AGGS = ("union", "mix", "scaled_mix", "ataraxos_mmd", "kc_dedup_K3", "kc_dedup_K3c", "attn_L3_K3", "attn_L3_K3c")
POOLS = ("unif", "stiff2", "stiff_fit", "trust_pool", "trust_gate", "trust_resample", "trust_pool_r5head", "clean8",
         "explore8", "clean16", "trust_trim12_clean16", "trust_trim12_clean16_r5head", "trust_pool_clean16")
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

    def verdict_row(rule, crit, passed):
        rows.append({"rule": rule, "criterion": crit, "per_seed": "", "mean": "", "ci95": "", "threshold": "all",
                     "pass": passed, "note": ""})

    # ---------------- D0 (inherited from R6-E3)
    anc = per("d0", lambda j: j["cal"]["kl_anchor"])
    bat = per("d0", lambda j: j["battle_acc_ply0"])
    n7 = per("d0", lambda j: j["nll_7known"])
    red = per("d0", lambda j: j["cal"]["reduction_pool16_vs_anchor"])
    p0 = bool(np.mean(anc) >= 0.12)
    p1 = bool(np.mean(bat) >= 0.99 and np.mean(n7) <= 0.01)
    p2 = bool(np.mean(red) >= 0.40)
    add("E3-D0 (0)", "anchor KL(P*_full ‖ P_anchor) per hidden slot, calibration", anc, ">= 0.12", p0)
    add("E3-D0 (1)", "battle accuracy (ply-0 battles, Z materialised), test", bat, ">= 0.99", bool(np.mean(bat) >= 0.99))
    add("E3-D0 (1)", "7-known-slot NLL (nats), test", n7, "<= 0.01", bool(np.mean(n7) <= 0.01))
    add("E3-D0 (2)", "16 marginal cone rollouts (union call): KL cut vs the t-4 anchor, calibration", red, ">= 0.40", p2,
        "KL anchor -> pooled: " + ", ".join(f"{a:.4f}->{b:.4f}" for a, b in zip(anc, per("d0", lambda j: j["cal"]["kl_pool16"]))) +
        "; one pass at t: " + ", ".join(f"{x:.4f}" for x in per("d0", lambda j: j["cal"]["kl_p0"])))
    gate = p0 and p1 and p2
    verdict_row("E3-D0 gate", "(0) and (1) and (2)", gate)
    tag = "" if gate else " (EXPLORATORY: gate failed)"
    shown = lambda ok: ok if gate else f"n/a (would be {ok})"
    # ---------------- D1
    kg = {h: per("d1", lambda j, h=h: j[f"key_gap_{h}"]) for h in HEADS}
    calg = {h: per("fit", lambda j, h=h: j[f"cal_key_gap_{h}"]) for h in HEADS}
    passh = {h: bool(np.mean(kg[h]) >= 0.28 and tci(kg[h])[1] > 0.25) for h in HEADS}
    for h in HEADS:
        add("E3-D1 (1)" + tag, f"{h} key cosine within-flank minus across-flank", kg[h], ">= 0.28, CI low > 0.25",
            shown(passh[h]), f"within {fmt(np.mean(per('d1', lambda j, h=h: j[f'key_cos_within_{h}'])))}, across "
            f"{fmt(np.mean(per('d1', lambda j, h=h: j[f'key_cos_across_{h}'])))}; calibration gap {fmt(np.mean(calg[h]))}")
    d1a = passh["K3"] or passh["K3c"]
    verdict_row("E3-D1 (1)" + tag, "K3 or K3c", shown(d1a))
    if passh["K3"] != passh["K3c"]:
        head, why = ("K3" if passh["K3"] else "K3c"), "the head that passes (1)"
    else:
        head = max(HEADS, key=lambda h: float(np.mean(calg[h])))
        why = ("both pass (1)" if d1a else "neither passes (1)") + "; larger calibration key gap"
    oks = [d1a]
    ja = per("d1", lambda j: j[f"attn_L3_{head}"]["jsd"])
    for Xn in ("scaled_mix", "ataraxos_mmd"):
        jx = per("d1", lambda j, Xn=Xn: j[Xn]["jsd"])
        ok = bool(np.mean(jx / ja) >= 5.0 and tci(jx - ja)[1] > 0)
        oks.append(ok)
        m, lo, hi = tci(jx - ja)
        add("E3-D1 (2)" + tag, f"belief JSD ratio {Xn} / attn_L3 ({head})", jx / ja, ">= 5.0, CI(diff) > 0", shown(ok),
            f"JSD diff {m:.6f} [{lo:.6f}, {hi:.6f}]")
    ka = per("d1", lambda j: j[f"attn_L3_{head}"]["kl"])
    for Xn in ("scaled_mix", "ataraxos_mmd", f"kc_dedup_{head}"):
        kx = per("d1", lambda j, Xn=Xn: j[Xn]["kl"])
        ok = bool(tci(kx - ka)[1] > 0)
        oks.append(ok)
        add("E3-D1 (2)" + tag, f"8-slot belief KL to P*_full: {Xn} - attn_L3 ({head})", kx - ka, "> 0, CI > 0", shown(ok))
    d1 = all(oks)
    verdict_row("E3-D1" + tag, f"(1) and all five of (2); key head for (2): {head} ({why})", shown(d1))
    # ---------------- D2
    st = per("d2", lambda j: j["spearman_trust"])
    ss = per("d2", lambda j: j["spearman_stiff"])
    wt = per("d2", lambda j: j["within_trust"])
    tt = per("d2", lambda j: j["strata_trust"])
    ts = per("d2", lambda j: j["strata_stiff"])
    a1 = bool(np.mean(st) >= 0.30 and tci(st)[1] > 0)
    a2 = bool(np.mean(wt) >= 0.25 and tci(wt)[1] > 0)
    a3 = bool(np.mean(st - ss) >= 0.55 and tci(st - ss)[1] > 0)
    a4 = bool(np.mean(tt - ts) >= 0.20 and tci(tt - ts)[1] > 0)
    add("E3-D2 (1)" + tag, "Spearman(beta_trust, -KL(Q* ‖ q_r)) overall, mixed bags", st, ">= 0.30, CI > 0", shown(a1),
        "Spearman(-H) " + ", ".join(fmt(x) for x in ss))
    add("E3-D2 (1)" + tag, "same, within (s, a) pairs (mean of per-pair Spearman)", wt, ">= 0.25, CI > 0", shown(a2),
        "-H within pairs " + ", ".join(fmt(x) for x in per("d2", lambda j: j["within_stiff"])))
    add("E3-D2 (1)" + tag, "Spearman(beta) - Spearman(-H) overall", st - ss, ">= 0.55, CI > 0", shown(a3))
    add("E3-D2 (1)" + tag, "same, mean within 10 equal-count entropy strata", tt - ts, ">= 0.20, CI > 0", shown(a4),
        "beta " + ", ".join(fmt(x) for x in tt) + "; -H " + ", ".join(fmt(x) for x in ts))
    d2a = a1 and a2 and a3 and a4
    verdict_row("E3-D2 (1)" + tag, "all four", shown(d2a))
    mse = {k: per("d2", lambda j, k=k: j[f"mse_{k}"]) for k in POOLS}
    crit = {}
    for Xn in ("trust_gate", "trust_pool", "trust_resample"):
        rr = 1 - mse[Xn] / mse["stiff2"]
        c1 = bool(np.mean(rr) >= 0.20 and tci(mse["stiff2"] - mse[Xn])[1] > 0)
        c2 = bool(tci(mse["unif"] - mse[Xn])[1] > 0)
        crit[Xn] = c1 and c2
        m, lo, hi = tci(mse["stiff2"] - mse[Xn])
        add("E3-D2 (2)" + tag, f"relative MSE reduction {Xn} vs stiff_pool (gamma = 2)", rr, ">= 0.20, CI(diff) > 0", shown(c1),
            f"MSE {fmt(np.mean(mse[Xn]), 5)} vs {fmt(np.mean(mse['stiff2']), 5)}; diff {m:.5f} [{lo:.5f}, {hi:.5f}]")
        m, lo, hi = tci(mse["unif"] - mse[Xn])
        add("E3-D2 (2)" + tag, f"MSE(unif_pool) - MSE({Xn})", mse["unif"] - mse[Xn], "> 0, CI > 0", shown(c2),
            f"MSE unif {fmt(np.mean(mse['unif']), 5)}")
    d2b = (crit["trust_gate"] or crit["trust_pool"]) and crit["trust_resample"]
    verdict_row("E3-D2 (2)" + tag, "(trust_gate or trust_pool) and trust_resample, each vs stiff_pool and unif_pool", shown(d2b))
    d2 = d2a and d2b
    verdict_row("E3-D2" + tag, "(1) and (2)", shown(d2))
    S = {"gate": gate, "rules": {"D0": gate, "D1": d1, "D1(1)": d1a, "D2": d2, "D2(1)": d2a, "D2(2)": d2b},
         "d1_head": head, "d1_head_reason": why, "d2_criteria": crit}
    # ---------------- self-check from the per-item CSVs
    from scipy.stats import spearmanr
    for s in SEEDS:
        if J(s, "d0") is None:
            continue
        d0 = J(s, "d0")
        checks.append(("d0 battle", s, d0["battle_acc_ply0"], C(s, "d0_battles.csv")["correct"].mean()))
        checks.append(("d0 nll7", s, d0["nll_7known"], C(s, "d0_nll7.csv")["nll7"].mean()))
        g = C(s, "d0_gate_cal.csv")
        checks.append(("d0 anchor kl cal", s, d0["cal"]["kl_anchor"], g["kl_anchor"].mean()))
        checks.append(("d0 pool16 reduction cal", s, d0["cal"]["reduction_pool16_vs_anchor"],
                       1 - g["kl_pool16"].mean() / g["kl_anchor"].mean()))
        d1j = J(s, "d1")
        c1 = C(s, "d1_positions.csv")
        for a in AGGS:
            checks.append((f"d1 jsd {a}", s, d1j[a]["jsd"], c1[f"jsd_{a}"].mean()))
            checks.append((f"d1 kl {a}", s, d1j[a]["kl"], c1[f"kl_{a}"].mean()))
        for h in HEADS:
            checks.append((f"d1 key gap {h}", s, d1j[f"key_gap_{h}"], float(np.nanmean(c1[f"key_gap_{h}"]))))
        d2j = J(s, "d2")
        c2 = C(s, "d2_rollouts.csv")
        mx = c2["in_mixed"] == 1
        checks.append(("d2 spearman trust (mixed)", s, d2j["spearman_trust"], spearmanr(c2["beta"][mx], -c2["kl"][mx]).correlation))
        checks.append(("d2 spearman stiff (mixed; H rounded)", s, d2j["spearman_stiff"], spearmanr(-c2["H"][mx], -c2["kl"][mx]).correlation))
        pr = c2["pair"][mx].astype(int)
        b_ = c2["beta"][mx].reshape(-1, 16)
        k_ = c2["kl"][mx].reshape(-1, 16)
        assert (pr.reshape(-1, 16) == pr.reshape(-1, 16)[:, :1]).all()
        wv = [spearmanr(b_[i], -k_[i]).correlation for i in range(b_.shape[0]) if np.std(b_[i]) > 0 and np.std(k_[i]) > 0]
        checks.append(("d2 within-pair spearman trust", s, d2j["within_trust"], float(np.nanmean(wv))))
        ca = C(s, "d2_actions.csv")
        for k in POOLS:
            checks.append((f"d2 mse {k}", s, d2j[f"mse_{k}"], float(((ca[f"q_{k}"] - ca["qstar_eff"]) ** 2).mean())))
        e_ = c2["e"][mx].reshape(-1, 16)
        checks.append(("d2 unif from rollouts", s, d2j["mse_unif"], float(((e_.mean(1) - ca["qstar_eff"]) ** 2).mean())))
    S["selfcheck_max_abs_diff"] = max([abs(a - b) for _, _, a, b in checks] or [float("nan")])
    S["selfcheck_max_abs_diff_excl_rounded_H"] = max([abs(a - b) for q, _, a, b in checks if "rounded" not in q] or [float("nan")])
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
    for k, f in (("d0.cal.kl_anchor", lambda j: j["cal"]["kl_anchor"]), ("d0.cal.kl_pool16", lambda j: j["cal"]["kl_pool16"]),
                 ("d0.cal.kl_routeb_all", lambda j: j["cal"]["kl_routeb_all"]), ("d0.cal.kl_p0", lambda j: j["cal"]["kl_p0"]),
                 ("d0.cal.kl_scaled16", lambda j: j["cal"]["kl_scaled16"]), ("d0.test.kl_anchor", lambda j: j["test"]["kl_anchor"]),
                 ("d0.test.kl_pool16", lambda j: j["test"]["kl_pool16"]), ("d0.test.kl_p0", lambda j: j["test"]["kl_p0"]),
                 ("d0.test.reduction_pool16", lambda j: j["test"]["reduction_pool16_vs_anchor"])):
        sec[k] = per("d0", f).tolist()
    for h in HEADS:
        for k in (f"key_gap_{h}", f"key_cos_within_{h}", f"key_cos_across_{h}"):
            sec["d1." + k] = per("d1", lambda j, k=k: j[k]).tolist()
        sec[f"fit.cal_key_gap_{h}"] = per("fit", lambda j, h=h: j[f"cal_key_gap_{h}"]).tolist()
    for k in ("kl_anchor", "kl_p0", "kl_routeb_all", "covered_L", "covered_R", "mean_nL_L", "mean_nL_R"):
        sec["d1." + k] = per("d1", lambda j, k=k: j[k]).tolist()
    for a in AGGS:
        for m_ in ("jsd", "kl"):
            sec[f"d1.{a}.{m_}"] = per("d1", lambda j, a=a, m_=m_: j[a][m_]).tolist()
    for k in ("spearman_trust", "spearman_stiff", "spearman_trust_r5head", "within_trust", "within_stiff", "within_trust_r5head",
              "strata_trust", "strata_stiff", "neff_trust_pool", "neff_trust_gate", "neff_stiff2", "resample_pruned_explore_share",
              "clean_terminal_like_kl", "clean_other_kl", "explore_terminal_like_kl", "explore_other_kl", "n_pairs"):
        sec["d2." + k] = per("d2", lambda j, k=k: j[k]).tolist()
    for k in POOLS:
        sec[f"d2.mse_{k}"] = per("d2", lambda j, k=k: j[f"mse_{k}"]).tolist()
        sec[f"d2.delusional_{k}"] = per("d2", lambda j, k=k: j[f"delusional_{k}"]).tolist()
    for tier in ("clean", "explore"):
        for k in ("kl", "abs_err", "sq_err", "terminal_like_H_lt_0.15", "flag_end", "count_violation", "offsup_per_move", "cone_lp",
                  "ply_lp", "beta", "beta_r5", "H"):
            sec[f"d2.tier_{tier}.{k}"] = per("d2", lambda j, tier=tier, k=k: j[f"tier_{tier}"][k]).tolist()
    for k in ("gamma", "alpha", "gate_q", "gate_tau", "target_mean_clean", "target_mean_explore", "train_beta_mean_clean",
              "train_beta_mean_explore", "train_spearman_beta_neg_target", "sel_beta_mean_clean", "sel_beta_mean_explore"):
        sec["trust." + k] = per("trust", lambda j, k=k: j[k]).tolist()
    for k in ("unif", "stiff2", "trust_pool", "trust_gate", "trust_resample", "clean8", "trust_pool_r5head"):
        sec["trust.logloss_" + k] = per("trust", lambda j, k=k: j["logloss"][k]).tolist()
    for k in ("alpha_b", "eta_b", "u_b_K3", "u_b_K3c"):
        sec["fit." + k] = per("fit", lambda j, k=k: j[k]).tolist()
    S["secondary"] = sec
    json.dump(S, open(os.path.join(REP, "summary.json"), "w"), indent=1, default=str)
    lines = ["| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule']} | {r['criterion']} | {r['per_seed']} | {r['mean']} | {r['ci95']} | {r['threshold']} | {r['pass']} | {r['note']} |"
              for r in rows]
    lines += ["", f"Self-check: {len(checks)} numbers re-derived from the per-item CSVs; largest |difference| "
              f"{S['selfcheck_max_abs_diff']:.3g}, {S['selfcheck_max_abs_diff_excl_rounded_H']:.3g} without the "
              "Spearman(-H) check, whose H column is rounded to 6 significant digits (r7/report/selfcheck.csv).", "",
              "| quantity | per seed | mean [95% CI] |", "|---|---|---|"]
    for k, v in sec.items():
        m, lo, hi = tci(v)
        lines.append(f"| {k} | {', '.join(fmt(float(x)) for x in v)} | {fmt(m)} [{fmt(lo)}, {fmt(hi)}] |")
    open(os.path.join(REP, "tables.md"), "w").write("\n".join(lines) + "\n")
    figures(sec, head)
    print(json.dumps({"gate": gate, "rules": S["rules"], "head": head, "selfcheck": S["selfcheck_max_abs_diff"],
                      "selfcheck_excl_H": S["selfcheck_max_abs_diff_excl_rounded_H"]}))


def figures(sec, head):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    hv = [("K4 (R6-E3)", None), ("K3", "d1.key_gap_K3"), ("K3c", "d1.key_gap_K3c")]
    v = [(0.2276, 0.2109, 0.2443) if k is None else tci(sec[k]) for _, k in hv]
    ax[0].bar(range(3), [x[0] for x in v], yerr=[x[0] - x[1] for x in v], capsize=3, color=["#bbb", "#c64", "#68a"])
    ax[0].axhline(0.28, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels([l for l, _ in hv], fontsize=8)
    ax[0].set_title("E3-D1 (1): key cosine gap within vs across flanks (bar 0.28)", fontsize=8.5)
    out += [["d1", f"key gap {l}", f"{x[0]:.5f}", f"{x[1]:.5f}", f"{x[2]:.5f}"] for (l, _), x in zip(hv, v)]
    aggs = ("scaled_mix", "ataraxos_mmd", f"kc_dedup_{head}", f"attn_L3_{head}", "union")
    for i, m_ in enumerate(("jsd", "kl")):
        v = [tci(sec[f"d1.{a}.{m_}"]) for a in aggs]
        ax[1 + i].bar(range(len(aggs)), [x[0] for x in v], yerr=[x[0] - x[1] if np.isfinite(x[1]) else 0 for x in v], capsize=3)
        ax[1 + i].set_xticks(range(len(aggs)))
        ax[1 + i].set_xticklabels(aggs, fontsize=7, rotation=20)
        ax[1 + i].set_title("E3-D1 (2): " + ("belief JSD U_Left vs U_Right" if m_ == "jsd" else "belief KL to P*_full"), fontsize=8.5)
        out += [["d1", f"{a} {m_}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for a, x in zip(aggs, v)]
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e7_d1.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ks = ("unif", "stiff2", "trust_pool", "trust_gate", "trust_resample", "clean8")
    v = [tci(sec[f"d2.mse_{k}"]) for k in ks]
    ax.bar(range(len(ks)), [x[0] for x in v], yerr=[x[0] - x[1] for x in v], capsize=3,
           color=["#888", "#6a8", "#c64", "#d96", "#a5c", "#bbb"])
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(["unif_pool", "stiff_pool (g=2)", "trust_pool", "trust_gate", "trust_resample", "clean only (ref.)"],
                       fontsize=7.5, rotation=12)
    ax.set_title("E3-D2 (2): action-value MSE vs Q*, mixed-reliability bags (16 rollouts per action)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e7_d2.png"), dpi=110)
    plt.close(fig)
    out += [["d2", f"mse {k}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for k, x in zip(ks, v)]
    with open(os.path.join(REP, "figures.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "series", "mean", "ci_lo", "ci_hi"])
        w.writerows(out)


if __name__ == "__main__":
    main()
