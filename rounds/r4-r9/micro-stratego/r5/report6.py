#!/usr/bin/env python3
"""R5-E2 report: applies the pre-registered capacity rule and every E2-D0..D3 rule to the synced per-seed summaries
(runs/s<seed>/<arch>/eval/*.json), re-derives each headline number from the per-item CSVs (self-check), and writes
report/verdicts.csv, report/summary.json, report/tables.md and the figures (PNG + CSV). Statistics: per-seed values,
3-seed mean and seed-t 95% interval (df = 2, t = 4.303).
   python3 report6.py"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
REP = os.path.join(HERE, "report")
SEEDS = (101, 102, 103)
AGGS = ("mix", "ataraxos_mmd", "kc_dedup", "attn_L3")
T975 = {2: 4.302652729749464, 1: 12.706204736174698}


def tci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan")
    h = T975.get(n - 1, 4.303) * x.std(ddof=1) / math.sqrt(n)
    return m, m - h, m + h


def J(seed, arch, part):
    p = os.path.join(RUNS, f"s{seed}", arch, "eval", f"{part}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def C(seed, arch, name):
    p = os.path.join(RUNS, f"s{seed}", arch, "eval", name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = list(csv.DictReader(f))
    return {k: np.array([float(x[k]) for x in r]) for k in r[0]} if r else {}


def fmt(x, d=4):
    return "nan" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def ci_str(v, d=4):
    m, lo, hi = tci(v)
    return f"{fmt(m, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def main():
    os.makedirs(REP, exist_ok=True)
    rows = []          # verdict rows
    checks = []        # (quantity, seed, json value, re-derived value)
    S = {"seeds": SEEDS}

    def per(arch, part, f):
        out = []
        for s in SEEDS:
            j = J(s, arch, part)
            out.append(f(j) if j is not None else float("nan"))
        return np.array(out, float)

    # ---------------- capacity rule on the calibration split
    sel = {}
    for arch in ("primary", "fallback"):
        b = per(arch, "d0cal", lambda j: j["battle_acc_ply0"])
        n7 = per(arch, "d0cal", lambda j: j["nll_7known"])
        kl = per(arch, "d0cal", lambda j: j["kl0_full_per_hidden_slot"])
        ok = bool(np.nanmean(b) >= 0.97 and np.nanmean(n7) <= 0.05 and np.nanmean(kl) >= 0.05)
        sel[arch] = {"cal_battle": b.tolist(), "cal_nll7": n7.tolist(), "cal_kl": kl.tolist(), "meets_1_2_3a_on_cal": ok}
    chosen = "primary" if sel["primary"]["meets_1_2_3a_on_cal"] else ("fallback" if sel["fallback"]["meets_1_2_3a_on_cal"] else None)
    S["capacity_rule"] = sel
    S["selected_arch"] = chosen
    A = chosen or "primary"
    S["reported_arch"] = A

    def add(rule, crit, vals, thr, passed, null_note=""):
        m, lo, hi = tci(vals)
        rows.append({"rule": rule, "criterion": crit, "per_seed": ", ".join(fmt(v) for v in vals), "mean": fmt(m),
                     "ci95": f"[{fmt(lo)}, {fmt(hi)}]", "threshold": thr, "pass": passed, "note": null_note})
        return passed

    # ---------------- D0 (test split for 1, 2, 3a; calibration split for 3b)
    bat = per(A, "d0", lambda j: j["battle_acc_ply0"])
    n7 = per(A, "d0", lambda j: j["nll_7known"])
    kl = per(A, "d0", lambda j: j["kl0_full_per_hidden_slot"])
    klt = per(A, "d0", lambda j: j["kl0_full_total_per_position"])
    p1 = add("E2-D0 (1)", "1-step battle accuracy (ply-0 battles, Z materialised), test", bat, ">= 0.97", bool(np.nanmean(bat) >= 0.97))
    p2 = add("E2-D0 (2)", "7-known-slot NLL (nats), test", n7, "<= 0.05", bool(np.nanmean(n7) <= 0.05))
    p3a = add("E2-D0 (3a)", "0-step KL(P*_full || P_0) per hidden slot (nats), test", kl, ">= 0.05", bool(np.nanmean(kl) >= 0.05))
    add("E2-D0 (3a, reported)", "same, total over hidden slots per position (nats)", klt, "(>= 0.40 noted in spec)", "n/a")
    rb = per(A, "d0b", lambda j: j["belief_kl_reduction"])
    ra = per(A, "d0b", lambda j: j["action_mse_reduction"])
    p3b_b = bool(np.nanmean(rb) >= 0.10)
    p3b_a = bool(np.nanmean(ra) >= 0.10)
    add("E2-D0 (3b-i)", "16 unskewed rollouts: belief KL reduction vs P_0, calibration split", rb, ">= 0.10 (or 3b-ii)", p3b_b)
    add("E2-D0 (3b-ii)", "16 unskewed rollouts: action-value MSE reduction vs 0-step, calibration split", ra, ">= 0.10 (or 3b-i)", p3b_a)
    p3b = p3b_b or p3b_a
    gate = bool(chosen is not None and p1 and p2 and p3a and p3b)
    rows.append({"rule": "E2-D0 gate", "criterion": f"capacity rule -> {chosen}; (1) and (2) and (3a) and (3b)", "per_seed": "",
                 "mean": "", "ci95": "", "threshold": "all", "pass": gate, "note": ""})
    S["gate"] = gate
    tag = "" if gate else " (EXPLORATORY: gate failed)"

    # ---------------- D1
    kg = per(A, "d1", lambda j: j["key_gap"])
    d1_1 = bool(np.nanmean(kg) >= 0.25 and tci(kg)[1] > 0)
    add("E2-D1 (1)" + tag, "key cosine within-flank minus across-flank", kg, ">= 0.25, CI > 0", d1_1 if gate else "n/a")
    d1_ok = [d1_1]
    for comp, key in (("belief", "jsd_belief"), ("value", "jsd_value")):
        jm = per(A, "d1", lambda j: j["mix"][key])
        ja = per(A, "d1", lambda j: j["attn_L3"][key])
        ratio = jm / ja
        ok = bool(np.nanmean(ratio) >= 2.5 and tci(jm - ja)[1] > 0)
        d1_ok.append(ok)
        add("E2-D1 (2)" + tag, f"JSD_mix / JSD_attn_L3, {comp} (CI on JSD_mix - JSD_attn)", ratio, ">= 2.5, CI(diff) > 0",
            ok if gate else "n/a", f"diff {ci_str(jm - ja, 6)}")
    for base in ("mix", "ataraxos_mmd", "kc_dedup"):
        for comp, key in (("belief KL to ref", "kl_ref_belief"), ("value MSE to ref", "mse_ref_value")):
            xb = per(A, "d1", lambda j: j[base][key])
            xa = per(A, "d1", lambda j: j["attn_L3"][key])
            ok = bool(tci(xb - xa)[1] > 0)
            d1_ok.append(ok)
            add("E2-D1 (2)" + tag, f"{comp}: {base} - attn_L3", xb - xa, "> 0, CI > 0", ok if gate else "n/a")
    d1 = all(d1_ok)
    rows.append({"rule": "E2-D1" + tag, "criterion": "(1) and all of (2)", "per_seed": "", "mean": "", "ci95": "",
                 "threshold": "all", "pass": d1 if gate else f"n/a (would be {d1})", "note": ""})
    # ---------------- D2
    st = per(A, "d2", lambda j: j["spearman_trust"])
    ss = per(A, "d2", lambda j: j["spearman_stiff"])
    tt = per(A, "d2", lambda j: j["strata_trust"])
    ts = per(A, "d2", lambda j: j["strata_stiff"])
    a1 = bool(np.nanmean(st) >= 0.25 and tci(st)[1] > 0)
    a2 = bool(np.nanmean(st - ss) >= 0.20 and tci(st - ss)[1] > 0)
    a3 = bool(np.nanmean(tt - ts) >= 0.15 and tci(tt - ts)[1] > 0)
    add("E2-D2" + tag, "Spearman(beta_trust, -KL(Q*||q_r)) overall", st, ">= 0.25, CI > 0", a1 if gate else "n/a")
    add("E2-D2" + tag, "Spearman(beta) - Spearman(-H) overall", st - ss, ">= 0.20, CI > 0", a2 if gate else "n/a")
    add("E2-D2" + tag, "same, mean within 10 entropy strata", tt - ts, ">= 0.15, CI > 0", a3 if gate else "n/a")
    d2 = a1 and a2 and a3
    rows.append({"rule": "E2-D2" + tag, "criterion": "all three", "per_seed": "", "mean": "", "ci95": "", "threshold": "all",
                 "pass": d2 if gate else f"n/a (would be {d2})", "note": ""})
    # ---------------- D3
    sr = per(A, "d3a", lambda j: j["shift_rate_relevant"])
    hashok = all(bool((J(s, A, "d3a") or {}).get("param_hash_identical", False)) for s in SEEDS)
    drift = max(float((J(s, A, "d3a") or {}).get("belief_max_abs_change", float("nan"))) for s in SEEDS)
    b1 = bool(np.nanmean(sr) >= 0.75 and tci(sr)[1] > 0.5 and hashok and drift == 0.0)
    add("E2-D3 (a)" + tag, "utility swap: shift rate toward higher true Miner utility (relevant boards)", sr,
        ">= 0.75, CI lower > 0.5; hash identical; belief change 0.0000", b1 if gate else "n/a",
        f"hash identical all seeds: {hashok}; max belief change {drift:.4g}; relevant boards per seed: "
        + ", ".join(str((J(s, A, 'd3a') or {}).get('n_relevant', 'nan')) for s in SEEDS))
    ew = per(A, "d3b", lambda j: j["effective_win"])
    b2 = bool(np.nanmean(ew) > 0.52 and tci(ew)[1] > 0.5)
    add("E2-D3 (b)" + tag, "Unified-LIR effective win vs P0 (400 duplicate games/seed)", ew, "> 0.52, CI lower > 0.5",
        b2 if gate else "n/a")
    d3 = b1 and b2
    rows.append({"rule": "E2-D3" + tag, "criterion": "(a) and (b)", "per_seed": "", "mean": "", "ci95": "", "threshold": "all",
                 "pass": d3 if gate else f"n/a (would be {d3})", "note": ""})
    S["rules"] = {"D0": gate, "D1": d1, "D2": d2, "D3": d3}

    # ---------------- self-check: re-derive headline numbers from the per-item CSVs
    for s in SEEDS:
        for arch in ("primary", "fallback"):
            d0 = J(s, arch, "d0")
            c = C(s, arch, "d0_test_positions.csv")
            if d0 and c:
                checks.append((f"D0 kl0 {arch}", s, d0["kl0_full_per_hidden_slot"], float(c["kl0_full"].mean())))
                n7c = C(s, arch, "d0_test_nll7.csv")
                checks.append((f"D0 nll7 {arch}", s, d0["nll_7known"], float(n7c["nll7"].mean())))
                bc = C(s, arch, "d0_test_battles.csv")
                checks.append((f"D0 battle {arch}", s, d0["battle_acc_ply0"], float(bc["correct"].mean())))
        d1 = J(s, A, "d1")
        c = C(s, A, "d1_positions.csv")
        if d1 and c:
            for agg in AGGS:
                checks.append((f"D1 jsd_b {agg}", s, d1[agg]["jsd_belief"], float(c[f"jsd_b_{agg}"].mean())))
                checks.append((f"D1 klref {agg}", s, d1[agg]["kl_ref_belief"], float(c[f"klref_b_{agg}"].mean())))
                checks.append((f"D1 mse_v {agg}", s, d1[agg]["mse_ref_value"], float(c[f"mse_v_{agg}"].mean())))
            checks.append(("D1 key gap", s, d1["key_gap"], float(np.nanmean(c["key_gap"]))))
        d2 = J(s, A, "d2")
        c = C(s, A, "d2_rollouts.csv")
        if d2 and c:
            from scipy.stats import spearmanr
            checks.append(("D2 spearman trust", s, d2["spearman_trust"], float(spearmanr(c["beta"], -c["kl"]).correlation)))
            checks.append(("D2 spearman stiff", s, d2["spearman_stiff"], float(spearmanr(-c["H"], -c["kl"]).correlation)))
        d3 = J(s, A, "d3a")
        c = C(s, A, "d3a_positions.csv")
        if d3 and c:
            rel = c["relevant"] > 0
            checks.append(("D3a shift rate", s, d3["shift_rate_relevant"], float(c["success"][rel].mean()) if rel.any() else float("nan")))
        d3b = J(s, A, "d3b")
        c = C(s, A, "d3b_games.csv")
        if d3b and c:
            checks.append(("D3b effective win", s, d3b["effective_win"], float(c["result"].mean())))
    S["selfcheck_max_abs_diff"] = max([abs(a - b) for _, _, a, b in checks if math.isfinite(a) and math.isfinite(b)] or [float("nan")])
    # ---------------- write
    with open(os.path.join(REP, "verdicts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(REP, "selfcheck.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "seed", "json_summary", "rederived_from_csv", "abs_diff"])
        for q, s, a, b in checks:
            w.writerow([q, s, f"{a:.8g}", f"{b:.8g}", f"{abs(a - b):.3g}"])
    # secondary tables (both architectures)
    sec = {}
    for arch in ("primary", "fallback"):
        sec[arch] = {}
        for part, keys in (("d0", ("battle_acc_ply0", "nll_7known", "kl0_full_per_hidden_slot", "kl0_constraint",
                                   "kl_full_vs_constraint_oracle")),
                           ("d0cal", ("battle_acc_ply0", "nll_7known", "kl0_full_per_hidden_slot")),
                           ("d0b", ("kl0", "kl16", "belief_kl_reduction", "mse0", "mse16", "action_mse_reduction")),
                           ("d1", ("key_gap", "key_cos_within", "key_cos_across", "ref_cov_per_hidden_slot", "ref_uncovered_share")),
                           ("d2", ("spearman_trust", "spearman_stiff", "strata_trust", "strata_stiff", "kl_lowest_entropy_tenth",
                                   "kl_other", "terminal_share")),
                           ("d3a", ("shift_rate_relevant", "n_relevant", "changed_action_rate", "shift_rate_all",
                                    "mean_miner_std", "mean_miner_miner", "belief_max_abs_change")),
                           ("d3b", ("effective_win", "wins", "draws", "losses", "mean_len"))):
            for k in keys:
                sec[arch][f"{part}.{k}"] = per(arch, part, lambda j, k=k: float(j[k])).tolist()
        for agg in AGGS:
            for k in ("jsd_belief", "jsd_value", "kl_ref_belief", "mse_ref_value"):
                sec[arch][f"d1.{agg}.{k}"] = per(arch, "d1", lambda j, agg=agg, k=k: float(j[agg][k])).tolist()
        tm = []
        for s in SEEDS:
            p = os.path.join(RUNS, f"s{s}", arch, "train_meta.json")
            tm.append(json.load(open(p)) if os.path.exists(p) else {})
        sec[arch]["params"] = [x.get("real", {}).get("params") for x in tm]
        sec[arch]["train_sec"] = [x.get("sec") for x in tm]
        sec[arch]["agg_fit"] = [{k: v for k, v in x.get("agg", {}).items() if not k.endswith("hist")} for x in tm]
        sec[arch]["lir_hash_identical"] = [x.get("lir_param_hash_before") == x.get("lir_param_hash_after") if x else None for x in tm]
        sec[arch]["audit_vm"] = [J(s, arch, "audit_vm") for s in SEEDS]
    S["secondary"] = sec
    json.dump(S, open(os.path.join(REP, "summary.json"), "w"), indent=1, default=str)
    lines = ["| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['rule']} | {r['criterion']} | {r['per_seed']} | {r['mean']} | {r['ci95']} | {r['threshold']} | "
                     f"{r['pass']} | {r['note']} |")
    lines += ["", f"Capacity rule (calibration split, 3-seed means): primary meets (1),(2),(3a): "
                  f"{sel['primary']['meets_1_2_3a_on_cal']}; fallback: {sel['fallback']['meets_1_2_3a_on_cal']}; selected: {chosen}.",
              f"Self-check: {len(checks)} headline numbers re-derived from the per-item CSVs; largest |difference| "
              f"{S['selfcheck_max_abs_diff']:.3g} (report/selfcheck.csv)."]
    lines += ["", "| architecture | quantity | per seed | mean [95% CI] |", "|---|---|---|---|"]
    for arch in ("primary", "fallback"):
        for k, v in sec[arch].items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) and x is not None for x in v):
                lines.append(f"| {arch} | {k} | {', '.join(fmt(float(x)) for x in v)} | {ci_str(v)} |")
    open(os.path.join(REP, "tables.md"), "w").write("\n".join(lines) + "\n")
    figures(A)
    print(json.dumps({"selected": chosen, "gate": gate, "rules": S["rules"], "selfcheck": S["selfcheck_max_abs_diff"]}))


def figures(A):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # training curves (calibration-split losses)
    out = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for arch, ls in (("primary", "-"), ("fallback", "--")):
        for s in SEEDS:
            p = os.path.join(RUNS, f"s{s}", arch, "train_curves.csv")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = list(csv.DictReader(f))
            st = [int(x["step"]) for x in r]
            for i, k in enumerate(("cal_belief", "cal_policy")):
                ax[i].plot(st, [float(x[k]) for x in r], ls, label=f"{arch} s{s}")
            for x in r:
                out.append([arch, s, x["step"], x["cal_belief"], x["cal_policy"], x["cal_trans"], x["cal_value"]])
    for i, t in enumerate(("calibration belief CE (masked slots)", "calibration policy CE")):
        ax[i].set_title(t)
        ax[i].set_xlabel("step")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "train_curves.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(REP, "train_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arch", "seed", "step", "cal_belief", "cal_policy", "cal_trans", "cal_value"])
        w.writerows(out)
    # D0 by turn
    rows = []
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = [(6, 12), (12, 18), (18, 24), (24, 30), (30, 37)]
    for arch, mk in (("primary", "o-"), ("fallback", "s--")):
        ys = []
        for s in SEEDS:
            c = C(s, arch, "d0_test_positions.csv")
            if c is None:
                continue
            ys.append([c["kl0_full"][(c["t"] >= lo) & (c["t"] < hi)].mean() for lo, hi in bins])
        if ys:
            m = np.mean(ys, 0)
            ax.plot([np.mean(b) for b in bins], m, mk, label=f"{arch}: KL(P*_full || P_0)")
            for b, v in zip(bins, m):
                rows.append([arch, f"{b[0]}-{b[1] - 1}", f"{v:.5f}"])
    c = C(SEEDS[0], "primary", "d0_test_positions.csv")
    if c is not None:
        g = [c["kl_pfull_pc"][(c["t"] >= lo) & (c["t"] < hi)].mean() for lo, hi in bins]
        ax.plot([np.mean(b) for b in bins], g, "^:", label="data: KL(P*_full || P*_c)")
        for b, v in zip(bins, g):
            rows.append(["constraint-only oracle", f"{b[0]}-{b[1] - 1}", f"{v:.5f}"])
    ax.axhline(0.05, color="k", lw=0.8, ls=":", label="gate 0.05")
    ax.set_xlabel("turn t")
    ax.set_ylabel("nats per hidden slot")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "d0_by_turn.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(REP, "d0_by_turn.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "turn_bin", "mean_over_seeds"])
        w.writerows(rows)
    # D1 bars
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    rows = []
    for i, (k, t) in enumerate((("jsd_belief", "policy sensitivity JSD, 8-slot belief"), ("jsd_value", "policy sensitivity JSD, value"),
                                ("kl_ref_belief", "belief KL to K=128 reference"))):
        ms, es = [], []
        for agg in AGGS:
            v = [J(s, A, "d1")[agg][k] for s in SEEDS if J(s, A, "d1")]
            m, lo, hi = tci(v)
            ms.append(m)
            es.append(0 if not math.isfinite(lo) else m - lo)
            rows.append([k, agg, f"{m:.6g}", f"{lo:.6g}", f"{hi:.6g}"])
        ax[i].bar(range(4), ms, yerr=es, capsize=3, color=["#888", "#6a8", "#68a", "#c64"])
        ax[i].set_xticks(range(4))
        ax[i].set_xticklabels(AGGS, fontsize=8)
        ax[i].set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REP, "d1_aggregators.png"), dpi=110)
    plt.close(fig)
    with open(os.path.join(REP, "d1_aggregators.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "aggregator", "mean", "ci_lo", "ci_hi"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
