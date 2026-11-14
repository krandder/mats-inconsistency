#!/usr/bin/env python3
"""R6-E3 report: applies the pre-registered E3-D0..D2 rules (PREREG-R6-E3.md section 2) to the synced per-seed files
r6/runs/s<seed>/eval/*.json, re-derives the headline numbers from the per-item CSVs (self-check), and writes
r6/report/verdicts.csv, tables.md, summary.json, selfcheck.csv and figures (PNG + CSV). Seed-t 95% intervals (df = 2).
   python3 r6/report_e3.py"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
REP = os.path.join(HERE, "report")
SEEDS = (101, 102, 103)
T975 = 4.302652729749464
AGGS = ("union", "mix", "scaled_mix", "ataraxos_mmd", "kc_dedup", "attn_L3")


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
    os.makedirs(REP, exist_ok=True)
    rows, checks = [], []

    def add(rule, crit, vals, thr, passed, note=""):
        m, lo, hi = tci(vals)
        rows.append({"rule": rule, "criterion": crit, "per_seed": ", ".join(fmt(v) for v in vals), "mean": fmt(m),
                     "ci95": f"[{fmt(lo)}, {fmt(hi)}]", "threshold": thr, "pass": passed, "note": note})

    # ---------------- D0
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
    add("E3-D0 (2)", "16 unskewed marginal cone rollouts (union call): KL cut vs the anchor, calibration", red, ">= 0.40", p2,
        "KL anchor -> pooled: " + ", ".join(f"{a:.4f}->{b:.4f}" for a, b in zip(anc, per("d0", lambda j: j["cal"]["kl_pool16"]))))
    gate = p0 and p1 and p2
    rows.append({"rule": "E3-D0 gate", "criterion": "(0) and (1) and (2)", "per_seed": "", "mean": "", "ci95": "",
                 "threshold": "all", "pass": gate, "note": ""})
    tag = "" if gate else " (EXPLORATORY: gate failed)"
    shown = lambda ok: ok if gate else f"n/a (would be {ok})"
    # ---------------- D1
    kg = per("d1", lambda j: j["key_gap"])
    d1a = bool(np.mean(kg) >= 0.25 and tci(kg)[1] > 0)
    add("E3-D1 (1)" + tag, "K4 key cosine within-flank minus across-flank", kg, ">= 0.25, CI > 0", shown(d1a))
    oks = [d1a]
    ja = per("d1", lambda j: j["attn_L3"]["jsd"])
    for X in ("ataraxos_mmd", "scaled_mix"):
        jx = per("d1", lambda j, X=X: j[X]["jsd"])
        ok = bool(np.mean(jx / ja) >= 2.5 and tci(jx - ja)[1] > 0)
        oks.append(ok)
        m, lo, hi = tci(jx - ja)
        add("E3-D1 (2)" + tag, f"belief JSD ratio {X} / attn_L3", jx / ja, ">= 2.5, CI(diff) > 0", shown(ok),
            f"JSD diff {m:.6f} [{lo:.6f}, {hi:.6f}]")
    ka = per("d1", lambda j: j["attn_L3"]["kl"])
    for X in ("scaled_mix", "ataraxos_mmd", "kc_dedup"):
        kx = per("d1", lambda j, X=X: j[X]["kl"])
        ok = bool(tci(kx - ka)[1] > 0)
        oks.append(ok)
        add("E3-D1 (2)" + tag, f"8-slot belief KL to P*_full: {X} - attn_L3", kx - ka, "> 0, CI > 0", shown(ok))
    d1 = all(oks)
    rows.append({"rule": "E3-D1" + tag, "criterion": "(1) and all five of (2)", "per_seed": "", "mean": "", "ci95": "",
                 "threshold": "all", "pass": shown(d1), "note": ""})
    # ---------------- D2
    st = per("d2", lambda j: j["spearman_trust"])
    ss = per("d2", lambda j: j["spearman_stiff"])
    tt = per("d2", lambda j: j["strata_trust"])
    ts = per("d2", lambda j: j["strata_stiff"])
    a1 = bool(np.mean(st) >= 0.20 and tci(st)[1] > 0)
    a2 = bool(np.mean(st - ss) >= 0.45 and tci(st - ss)[1] > 0)
    a3 = bool(np.mean(tt - ts) >= 0.15 and tci(tt - ts)[1] > 0)
    add("E3-D2 (1)" + tag, "Spearman(beta_trust, -KL(Q* ‖ q_r)) overall", st, ">= 0.20, CI > 0", shown(a1))
    add("E3-D2 (1)" + tag, "Spearman(beta) - Spearman(-H) overall", st - ss, ">= 0.45, CI > 0", shown(a2))
    add("E3-D2 (1)" + tag, "same, mean within 10 equal-count entropy strata", tt - ts, ">= 0.15, CI > 0", shown(a3))
    ms = per("d2", lambda j: j["mse_stiff"])
    mt = per("d2", lambda j: j["mse_trust"])
    rr = 1 - mt / ms
    b1 = bool(np.mean(rr) >= 0.20 and tci(ms - mt)[1] > 0)
    m, lo, hi = tci(ms - mt)
    add("E3-D2 (2)" + tag, "relative action-value MSE reduction trust_pool vs stiff_pool", rr, ">= 0.20, CI(diff) > 0", shown(b1),
        f"MSE stiff - trust {m:.5f} [{lo:.5f}, {hi:.5f}]")
    dt = per("d2", lambda j: j["delusional_trust"])
    ds = per("d2", lambda j: j["delusional_stiff"])
    b2 = bool(np.mean(dt) < np.mean(ds))
    add("E3-D2 (2)" + tag, "delusional-move rate: stiff_pool - trust_pool", ds - dt, "> 0 (strictly lower)", shown(b2),
        "counts trust " + ", ".join(str(int((J(s, 'd2') or {}).get('delusional_count_trust', -1))) for s in SEEDS) +
        "; stiff " + ", ".join(str(int((J(s, 'd2') or {}).get('delusional_count_stiff', -1))) for s in SEEDS))
    d2 = a1 and a2 and a3 and b1 and b2
    rows.append({"rule": "E3-D2" + tag, "criterion": "all of (1) and (2)", "per_seed": "", "mean": "", "ci95": "",
                 "threshold": "all", "pass": shown(d2), "note": ""})
    S = {"gate": gate, "rules": {"D0": gate, "D1": d1, "D2": d2}}
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
        checks.append(("d0 pool16 reduction cal", s, d0["cal"]["reduction_pool16_vs_anchor"], 1 - g["kl_pool16"].mean() / g["kl_anchor"].mean()))
        d1j = J(s, "d1")
        c1 = C(s, "d1_positions.csv")
        for a in AGGS:
            checks.append((f"d1 jsd {a}", s, d1j[a]["jsd"], c1[f"jsd_{a}"].mean()))
            checks.append((f"d1 kl {a}", s, d1j[a]["kl"], c1[f"kl_{a}"].mean()))
        checks.append(("d1 key gap", s, d1j["key_gap"], float(np.nanmean(c1["key_gap"]))))
        d2j = J(s, "d2")
        c2 = C(s, "d2_rollouts.csv")
        checks.append(("d2 spearman trust", s, d2j["spearman_trust"], spearmanr(c2["beta"], -c2["kl"]).correlation))
        checks.append(("d2 spearman stiff", s, d2j["spearman_stiff"], spearmanr(-c2["H"], -c2["kl"]).correlation))
        ca = C(s, "d2_actions.csv")
        for k in ("unif", "stiff", "trust"):
            checks.append((f"d2 mse {k}", s, d2j[f"mse_{k}"], float(((ca[f"q_{k}"] - ca["qstar_eff"]) ** 2).mean())))
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
    # secondary numbers
    sec = {}
    for k, f in (("d0.cal.kl_anchor", lambda j: j["cal"]["kl_anchor"]), ("d0.cal.kl_pool16", lambda j: j["cal"]["kl_pool16"]),
                 ("d0.cal.kl_routeb_all", lambda j: j["cal"]["kl_routeb_all"]), ("d0.cal.kl_p0", lambda j: j["cal"]["kl_p0"]),
                 ("d0.cal.kl_scaled16", lambda j: j["cal"]["kl_scaled16"]), ("d0.cal.covered_share", lambda j: j["cal"]["covered_share"]),
                 ("d0.test.kl_anchor", lambda j: j["test"]["kl_anchor"]), ("d0.test.kl_pool16", lambda j: j["test"]["kl_pool16"]),
                 ("d0.test.reduction_pool16", lambda j: j["test"]["reduction_pool16_vs_anchor"]),
                 ("d0.test.kl_p0", lambda j: j["test"]["kl_p0"]), ("d0.test.kl_routeb_all", lambda j: j["test"]["kl_routeb_all"])):
        sec[k] = per("d0", f).tolist()
    for k in ("kl_anchor", "kl_p0", "kl_routeb_all", "key_cos_within", "key_cos_across", "covered_L", "covered_R",
              "mean_nL_L", "mean_nL_R"):
        sec["d1." + k] = per("d1", lambda j, k=k: j[k]).tolist()
    for a in AGGS:
        for m_ in ("jsd", "kl"):
            sec[f"d1.{a}.{m_}"] = per("d1", lambda j, a=a, m_=m_: j[a][m_]).tolist()
    for k in ("spearman_trust", "spearman_stiff", "strata_trust", "strata_stiff", "mse_unif", "mse_stiff", "mse_trust",
              "delusional_unif", "delusional_stiff", "delusional_trust", "terminal_like_share", "gamma", "n_rollouts"):
        sec["d2." + k] = per("d2", lambda j, k=k: j[k]).tolist()
    for k in ("alpha_b", "eta_b", "u_b"):
        sec["fit." + k] = per("fit", lambda j, k=k: j[k]).tolist()
    S["secondary"] = sec
    json.dump(S, open(os.path.join(REP, "summary.json"), "w"), indent=1, default=str)
    lines = ["| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['rule']} | {r['criterion']} | {r['per_seed']} | {r['mean']} | {r['ci95']} | {r['threshold']} | {r['pass']} | {r['note']} |"
              for r in rows]
    lines += ["", f"Self-check: {len(checks)} numbers re-derived from the per-item CSVs; largest |difference| "
              f"{S['selfcheck_max_abs_diff']:.3g} (r6/report/selfcheck.csv).", "",
              "| quantity | per seed | mean [95% CI] |", "|---|---|---|"]
    for k, v in sec.items():
        m, lo, hi = tci(v)
        lines.append(f"| {k} | {', '.join(fmt(float(x)) for x in v)} | {fmt(m)} [{fmt(lo)}, {fmt(hi)}] |")
    open(os.path.join(REP, "tables.md"), "w").write("\n".join(lines) + "\n")
    figures(sec)
    print(json.dumps({"gate": gate, "rules": S["rules"], "selfcheck": S["selfcheck_max_abs_diff"]}))


def figures(sec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    labels = [("anchor (t-4)", "d0.cal.kl_anchor"), ("16 marginal rollouts", "d0.cal.kl_pool16"),
              ("all hidden slots scored", "d0.cal.kl_routeb_all"), ("one pass at t", "d0.cal.kl_p0")]
    ms = [tci(sec[k]) for _, k in labels]
    ax[0].bar(range(4), [m[0] for m in ms], yerr=[m[0] - m[1] for m in ms], capsize=3, color=["#888", "#c64", "#d96", "#68a"])
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([l for l, _ in labels], fontsize=7, rotation=12)
    ax[0].set_title("E3-D0: KL to P*_full per hidden slot (calibration)", fontsize=9)
    out += [["d0", l, f"{m[0]:.5f}", f"{m[1]:.5f}", f"{m[2]:.5f}"] for (l, _), m in zip(labels, ms)]
    aggs = ("union", "mix", "scaled_mix", "ataraxos_mmd", "kc_dedup", "attn_L3")
    for i, m_ in enumerate(("jsd", "kl")):
        v = [tci(sec[f"d1.{a}.{m_}"]) for a in aggs]
        ax[1 + i].bar(range(len(aggs)), [x[0] for x in v], yerr=[x[0] - x[1] if np.isfinite(x[1]) else 0 for x in v], capsize=3)
        ax[1 + i].set_xticks(range(len(aggs)))
        ax[1 + i].set_xticklabels(aggs, fontsize=7, rotation=20)
        ax[1 + i].set_title("E3-D1: " + ("belief JSD U_Left vs U_Right" if m_ == "jsd" else "belief KL to P*_full"), fontsize=9)
        out += [["d1", f"{a} {m_}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for a, x in zip(aggs, v)]
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e3_d0_d1.png"), dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    v = [tci(sec[f"d2.mse_{k}"]) for k in ("unif", "stiff", "trust")]
    ax.bar(range(3), [x[0] for x in v], yerr=[x[0] - x[1] for x in v], capsize=3, color=["#888", "#6a8", "#c64"])
    ax.set_xticks(range(3))
    ax.set_xticklabels(["unif_pool", "stiff_pool", "trust_pool"])
    ax.set_title("E3-D2: action-value MSE vs Q* (16 rollouts/action)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "e3_d2.png"), dpi=110)
    plt.close(fig)
    out += [["d2", f"mse {k}", f"{x[0]:.6f}", f"{x[1]:.6f}", f"{x[2]:.6f}"] for k, x in zip(("unif", "stiff", "trust"), v)]
    with open(os.path.join(REP, "figures.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["part", "series", "mean", "ci_lo", "ci_hi"])
        w.writerows(out)


if __name__ == "__main__":
    main()
