"""P3 analysis: T1 tables + tests, M0 reading, Stage 4 gate, chart JSON.  Reads results/t1/*.json, results/m0.json,
results/unit_tests.json; writes results/summary.json and results/charts.json; prints the tables used in RESULTS.md."""
import glob
import json
import math
import os
from collections import Counter

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
T1 = os.path.join(RES, "t1")


def load(name):
    p = os.path.join(T1, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def pooled(names):
    reps, runs = [], []
    for nm in names:
        d = load(nm)
        if d is not None:
            reps += d["reps"]
            runs.append(d)
    return reps, runs


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def tci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    h = stats.t.ppf(0.975, len(x) - 1) * se
    return float(m), float(m - h), float(m + h)


def newcombe(k1, n1, k2, n2):
    """95% CI for p1 - p2 (Newcombe hybrid score)."""
    l1, u1 = wilson(k1, n1); l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def arm_row(reps, runs):
    n = len(reps)
    k = sum(r["aligned"] for r in reps)
    lo, hi = wilson(k, n)
    live = [sum(t["H_bits"] > 0.1 for t in r["traders"] if not t["decoy"]) for r in reps]
    const = [sum(t["H_bits"] <= 0.1 for t in r["traders"] if not t["decoy"]) for r in reps]
    nat1 = [any(t["nmi_nat1"] >= 0.8 for t in r["traders"] if not t["decoy"]) for r in reps]
    nat2 = [any(t["nmi_nat2"] >= 0.8 for t in r["traders"] if not t["decoy"]) for r in reps]
    sg = tci([r["suff_gap"] for r in reps]); mg = tci([r["model_gap"] for r in reps])
    row = dict(n=n, aligned=k, rate=k / n if n else float("nan"), ci95=[lo, hi],
               has_nat1=float(np.mean(nat1)), has_nat2=float(np.mean(nat2)),
               both_nat=float(np.mean([a and b for a, b in zip(nat1, nat2)])),
               rot_sufficient=float(np.mean([r["rot_trader"] and r["suff_gap"] <= 0.02 for r in reps])),
               rot_trader=float(np.mean([r["rot_trader"] for r in reps])), dup=float(np.mean([r["dup"] for r in reps])),
               oracle_nat_dup=float(np.mean([r.get("profile") in ("b1|b1|b2", "b1|b2|b2") for r in reps])),
               oracle_all3=float(np.mean([r.get("profile") == "b1|b2|xor" for r in reps])),
               aligned_but_gap_gt_0p05=int(sum(r["aligned"] and r["suff_gap"] > 0.05 for r in reps)),
               live_traders_mean=float(np.mean(live)), const_traders_mean=float(np.mean(const)),
               suff_gap=sg, model_gap=mg, suff_gap_le_0p02=float(np.mean([r["suff_gap"] <= 0.02 for r in reps])),
               ece=float(np.mean([r["ece"] for r in reps])),
               profiles=Counter(r["profile"] for r in reps if r.get("profile")).most_common(6),
               train_sec=float(np.mean([d["train_sec"] for d in runs])) if runs else None)
    led = [r["ledger"] for r in reps if "ledger" in r]
    if led:
        row["as_err_abs_mean_per_output"] = float(np.mean([l["as_err_abs_mean"] for l in led]))
        row["as_err_abs_mean_joint"] = float(np.mean([l["joint_as_err_abs_mean"] for l in led]))
        row["total_pay_per_output"] = float(np.mean([l["total"] for l in led]))
    return row


def traj_mean(runs):
    if not runs:
        return None
    steps = [r["step"] for r in runs[0]["traj"]]
    arr = np.array([[r["max_nmi_nat1_nat2_rot"] for r in d["traj"]] for d in runs])  # runs, steps, 3
    return dict(step=steps, nat1=arr[:, :, 0].mean(0).round(3).tolist(), nat2=arr[:, :, 1].mean(0).round(3).tolist(),
                rot=arr[:, :, 2].mean(0).round(3).tolist())


def main():
    out = {}
    ARMS_S = [("loo", "LOO / backprop"), ("add", "additive reader"), ("mob1e-3", "Moebius L1 1e-3"),
              ("mob1e-2", "Moebius L1 1e-2"), ("shap", "exact Shapley"), ("banz", "Banzhaf"),
              ("asig", "AS-IG per-output"), ("asigj", "AS-IG joint (neg. control)")]
    S = {}
    runs_by = {}
    for tag, _ in ARMS_S:
        reps, runs = pooled([f"S_N_{tag}_s0", f"S_N_{tag}_s1"])
        if reps:
            S[tag] = arm_row(reps, runs)
            runs_by[tag] = runs
    # Moebius lambda chosen by validation log loss (model_gap on 100k fresh examples, mean over the 20 seeds)
    mob_pick = None
    if "mob1e-3" in S and "mob1e-2" in S:
        mob_pick = min(("mob1e-3", "mob1e-2"), key=lambda t: S[t]["model_gap"][0])
        S["mob"] = dict(S[mob_pick], picked=mob_pick)
    out["S"] = S
    out["moebius_lambda_pick"] = mob_pick

    # criterion 1: AS-IG vs LOO, additive, Moebius (picked), Banzhaf; one-sided Fisher, Holm
    tests = {}
    if "asig" in S:
        a = S["asig"]
        raw = []
        for b in ("loo", "add", "mob", "banz"):
            if b not in S:
                continue
            B_ = S[b]
            p = stats.fisher_exact([[a["aligned"], a["n"] - a["aligned"]], [B_["aligned"], B_["n"] - B_["aligned"]]],
                                   alternative="greater").pvalue
            raw.append((b, p, a["rate"] - B_["rate"], newcombe(a["aligned"], a["n"], B_["aligned"], B_["n"])))
        order = sorted(range(len(raw)), key=lambda i: raw[i][1])
        m = len(raw)
        adj = [0.0] * m
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * raw[i][1]))
            adj[i] = run
        for i, (b, p, d, ci) in enumerate(raw):
            tests[b] = dict(p_one_sided=p, p_holm=adj[i], diff=d, diff_ci95=list(ci))
        crit = dict(asig_aligned_ge_16_of_20=a["aligned"] >= 16 and a["n"] >= 20,
                    diff_ge_0p30_all=all(t["diff"] >= 0.30 for t in tests.values()),
                    holm_all_lt_0p05=all(t["p_holm"] < 0.05 for t in tests.values()),
                    bayes_gap_le_0p02=a["suff_gap"][0] <= 0.02)
        out["tests_asig_vs"] = tests
        # extra (reported, not in the pre-registered family): Shapley vs additive, AS-IG vs Shapley
        ex = {}
        for x, y in (("shap", "add"), ("asig", "shap"), ("shap", "loo"), ("add", "loo")):
            if x in S and y in S:
                X, Y = S[x], S[y]
                ex[f"{x}_vs_{y}"] = dict(diff=X["rate"] - Y["rate"], p_one_sided=stats.fisher_exact(
                    [[X["aligned"], X["n"] - X["aligned"]], [Y["aligned"], Y["n"] - Y["aligned"]]], alternative="greater").pvalue)
        out["tests_extra"] = ex
    else:
        crit = {}

    # controls: prequential pay, 95% t-CI upper bound < 0.005 (nats/example)
    ctl = {}
    d = load("ctl_shuffle_asig")
    if d:
        pp = np.array(d["prequential_pay"])  # R, n
        tot = pp.sum(1)
        ctl["shuffle"] = dict(per_trader_mean=pp.mean(0).tolist(), total_pay=tci(tot),
                              max_trader_upper=max(tci(pp[:, j])[2] for j in range(pp.shape[1])))
        ctl["shuffle"]["pass"] = bool(ctl["shuffle"]["total_pay"][2] < 0.005 and ctl["shuffle"]["max_trader_upper"] < 0.005)
    for nm, key in (("ctl_decoy_asig", "decoy"), ("ctl_decoyreal_asig", "decoy_real_x")):
        d = load(nm)
        if d:
            pp = np.array(d["prequential_pay"])
            dec = tci(pp[:, -1])
            ctl[key] = dict(decoy_pay=dec, trader_pays_mean=pp[:, :-1].mean(0).tolist(), passed=bool(dec[2] < 0.005),
                            aligned=sum(r["aligned"] for r in d["reps"]), n=len(d["reps"]))
    out["controls"] = ctl
    crit["controls_pass"] = bool(ctl.get("shuffle", {}).get("pass") and ctl.get("decoy", {}).get("passed"))
    crit["all"] = bool(crit and all(v for k, v in crit.items() if k != "all"))
    out["criterion1"] = crit

    # pre-registered falsifiers (a)-(d)
    fz = {}
    if all(t in S for t in ("asig", "shap", "add", "mob", "loo", "banz", "asigj")):
        near = {x: {b: abs(S[x]["rate"] - S[b]["rate"]) < 0.30 for b in ("add", "mob")} for x in ("asig", "shap")}
        fz["a_known_baseline_explains"] = dict(within_0p30=near, holm_p_asig=({k: v["p_holm"] for k, v in tests.items()}),
                                              triggered=bool(all(any(v.values()) for v in near.values())))
        fz["b_oracle_wrong"] = dict(asigj_minus_loo=S["asigj"]["rate"] - S["loo"]["rate"],
                                   banz_avoids_rot_share=1 - S["banz"]["rot_trader"],
                                   triggered=bool(abs(S["asigj"]["rate"] - S["loo"]["rate"]) >= 0.30 or 1 - S["banz"]["rot_trader"] >= 0.5))
        fz["c_pay_without_discovery"] = dict(aligned_runs_gap_gt_0p05={x: S[x]["aligned_but_gap_gt_0p05"] for x in ("asig", "shap")},
                                            gap_part=bool(S["asig"]["aligned_but_gap_gt_0p05"] + S["shap"]["aligned_but_gap_gt_0p05"] > 0),
                                            note="the over-confidence part is judged from ECE of those runs (see RESULTS.md)")
        lo = []
        if "shuffle" in ctl: lo.append(ctl["shuffle"]["total_pay"][1])
        if "decoy" in ctl: lo.append(ctl["decoy"]["decoy_pay"][1])
        fz["d_noise_earns"] = dict(ci_lower=lo, triggered=bool(any(x > 0 for x in lo)))
    out["falsifiers"] = fz

    # B variants
    Bv = {}
    for toy in ("N", "M"):
        for arm in ("loo", "add", "shap", "asig"):
            reps, runs = pooled([f"B_{toy}_{arm}_s0"])
            if reps:
                Bv[f"B{toy.lower()}_{arm}"] = arm_row(reps, runs)
    out["B"] = Bv
    # sanity
    san = {}
    for arm in ("mhard", "msoft", "vqsup"):
        d = load(f"san_{arm}_K4")
        if d:
            r = d["reps"][0]
            san[arm] = dict(suff_gap=r["suff_gap"], model_gap=r["model_gap"], H_bits=r["traders"][0]["H_bits"],
                            nmi_nat1=r["traders"][0]["nmi_nat1"], nmi_nat2=r["traders"][0]["nmi_nat2"])
    out["sanity_K4"] = san
    # diagnostic (not pre-registered): maker exploration 0.05
    dg = {}
    for arm in ("loo", "shap", "banz", "asig"):
        reps, runs = pooled([f"diag_x05_S_N_{arm}_s0", f"diag_x05_S_N_{arm}_s1"])
        if reps:
            dg[arm] = arm_row(reps, runs)
            runs_by["x05_" + arm] = runs
    out["diag_explore_0p05"] = dg
    dg20 = {}
    for arm in ("shap", "loo"):
        reps, runs = pooled([f"diag_x20_S_N_{arm}_s0", f"diag_x20_S_N_{arm}_s1"])
        if reps:
            dg20[arm] = arm_row(reps, runs)
    out["diag_explore_0p2"] = dg20
    # held-out replication (seeds 2-3), test fixed before results: shap x05 vs additive and vs loo x05, Holm over 2
    rp = {}
    for tag in ("x05_S_N_shap", "x05_S_N_loo", "x05_S_N_banz", "x05_S_N_asig", "S_N_add", "S_N_mob1e-3"):
        reps, runs = pooled([f"rep_{tag}_s2", f"rep_{tag}_s3"])
        if reps:
            rp[tag] = arm_row(reps, runs)
    rt = {}
    if "x05_S_N_shap" in rp:
        X = rp["x05_S_N_shap"]
        raw = []
        for b in ("S_N_add", "x05_S_N_loo"):
            if b in rp:
                Y = rp[b]
                raw.append((b, stats.fisher_exact([[X["aligned"], X["n"] - X["aligned"]], [Y["aligned"], Y["n"] - Y["aligned"]]],
                                                  alternative="greater").pvalue, X["rate"] - Y["rate"],
                            newcombe(X["aligned"], X["n"], Y["aligned"], Y["n"])))
        ps = sorted(raw, key=lambda t: t[1])
        for i, (b, p_, d_, ci) in enumerate(ps):
            rt[b] = dict(p_one_sided=p_, p_holm=min(1.0, max((len(ps) - j) * ps[j][1] for j in range(i + 1))), diff=d_, diff_ci95=list(ci))
        rt["replicated"] = bool(rt and all(v["p_holm"] < 0.05 for k, v in rt.items() if k != "replicated"))
        # pooled over all 40 seeds (0-3), reported only
        both = lambda a, b: (a["aligned"] + b["aligned"], a["n"] + b["n"])
        if "shap" in dg and "S_N_add" in rp:
            k1, n1 = both(dg["shap"], X); k2, n2 = both(S["add"], rp["S_N_add"])
            rt["pooled_40_shapx05_vs_add"] = dict(shap=[k1, n1], add=[k2, n2], ci_shap=list(wilson(k1, n1)), ci_add=list(wilson(k2, n2)),
                                                 p_one_sided=stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="greater").pvalue)
    out["replication_seeds23"] = rp
    out["replication_test"] = rt

    # M0
    m0p = os.path.join(RES, "m0.json")
    m0 = json.load(open(m0p)) if os.path.exists(m0p) else None
    if m0:
        rd = m0["reader"]
        out["M0"] = dict(p_fixed=m0["p_fixed"], def_failure_vs_p=m0["def_failure_vs_p"], gate_pass=m0["gate_pass"],
                         gate=m0["gate"], rounds=m0["rounds"], T=m0["T"], prereg=m0["prereg_reading"],
                         reader={a: {k: v for k, v in r.items() if k != "S_R_per_round" and not k.startswith("c")}
                                 | {c: {kk: vv for kk, vv in r[c].items() if kk not in ("mean_gain_per_chunk", "chunk_starts")}
                                    for c in r if c.startswith("c")} for a, r in rd.items()})
    # Stage 4 gate
    g1 = None
    if "add" in S:
        g1 = {}
        for x in ("asig", "shap"):
            if x in S:
                X, A = S[x], S["add"]
                p = stats.fisher_exact([[X["aligned"], X["n"] - X["aligned"]], [A["aligned"], A["n"] - A["aligned"]]],
                                       alternative="greater").pvalue
                g1[x] = dict(rate=X["rate"], add_rate=A["rate"], p_one_sided=p, beats=bool(X["rate"] > A["rate"] and p < 0.05))
    g2 = m0["prereg_reading"]["stage4_gate_ii_open"] if m0 else None
    out["stage4_gate"] = dict(i_arm6_or_arm4_beats_additive=g1, i_open=bool(g1 and any(v["beats"] for v in g1.values())),
                              ii_m0_leaves_b_open=g2,
                              open=bool(g1 and any(v["beats"] for v in g1.values()) and g2))
    ut = json.load(open(os.path.join(RES, "unit_tests.json")))
    out["unit_tests_all_pass"] = ut.get("all_pass")
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=float)

    # charts
    ch = []
    order = [t for t, _ in ARMS_S if t in S]
    ch.append(dict(id="p3_t1_aligned", type="bar", title="T1 variant S (toy N): share of seeds ALIGNED, 95% Wilson CI",
                   x=[dict(ARMS_S)[t] for t in order], y=[S[t]["rate"] for t in order],
                   err_lo=[S[t]["ci95"][0] for t in order], err_hi=[S[t]["ci95"][1] for t in order],
                   n=[S[t]["n"] for t in order], y_label="aligned share"))
    ch.append(dict(id="p3_t1_suffgap", type="bar", title="T1 variant S: Bayes gap of the code partition (nats/example, 100k fresh)",
                   x=[dict(ARMS_S)[t] for t in order], y=[S[t]["suff_gap"][0] for t in order],
                   err_lo=[S[t]["suff_gap"][1] for t in order], err_hi=[S[t]["suff_gap"][2] for t in order], y_label="nats"))
    ch.append(dict(id="p3_t1_nmi_traj", type="line", title="T1 variant S: best-trader NMI with b1 (mean over seeds) during training",
                   series={t: traj_mean(runs_by[t]) for t in runs_by}))
    if dg:
        ch.append(dict(id="p3_t1_diag_explore", type="bar", title="Diagnostic: maker exploration 0.05 vs pre-registered maker, aligned share",
                       x=[f"{a} {v}" for a in dg for v in ("pre-reg", "explore")],
                       y=[S[a]["rate"] if v == "pre-reg" else dg[a]["rate"] for a in dg for v in ("pre-reg", "explore")]))
    if m0:
        ch.append(dict(id="p3_m0_SR", type="line", title="M0: reader score S_R per CA round (9x9 percolated, mean over test mazes)",
                       series={a: m0["reader"][a]["S_R_per_round"] for a in m0["reader"]}, x_label="round", y_label="S_R (nats)"))
        ch.append(dict(id="p3_m0_c6", type="bar", title="M0: mean gain per 6-round chunk", series={
            a: dict(x=m0["reader"][a]["c6"]["chunk_starts"], y=m0["reader"][a]["c6"]["mean_gain_per_chunk"]) for a in m0["reader"]}))
    json.dump(ch, open(os.path.join(RES, "charts.json"), "w"), indent=1, default=float)

    # print tables
    print("T1 variant S (toy N), 20 seeds per arm")
    print("%-28s %8s %15s %6s %6s %6s %6s %18s %8s" % ("arm", "aligned", "95% CI", "nat1", "nat2", "rot", "const", "Bayes gap [CI]", "ECE"))
    for t, lab in ARMS_S:
        if t in S:
            r = S[t]
            print("%-28s %4d/%-3d [%.2f, %.2f] %6.2f %6.2f %6.2f %6.2f %6.3f [%.3f,%.3f] %8.4f" % (
                lab, r["aligned"], r["n"], *r["ci95"], r["has_nat1"], r["has_nat2"], r["rot_trader"], r["const_traders_mean"],
                *r["suff_gap"], r["ece"]))
    print("Moebius lambda pick:", mob_pick)
    print("tests AS-IG vs:", json.dumps(tests, indent=0, default=float))
    print("criterion 1:", crit)
    print("controls:", json.dumps(ctl, default=float))
    print("B:", {k: (v["aligned"], v["n"], round(v["suff_gap"][0], 3), v["profiles"][:3]) for k, v in Bv.items()})
    print("sanity:", san)
    print("diag x05:", {k: (v["aligned"], v["n"], round(v["suff_gap"][0], 3), round(v["const_traders_mean"], 2), v["profiles"][:3]) for k, v in dg.items()})
    print("diag x20:", {k: (v["aligned"], v["n"], round(v["suff_gap"][0], 3), v["profiles"][:3]) for k, v in dg20.items()})
    print("replication s2-3:", {k: (v["aligned"], v["n"], [round(c, 2) for c in v["ci95"]], round(v["suff_gap"][0], 3), round(v["both_nat"], 2), v["profiles"][:3]) for k, v in rp.items()})
    print("replication test:", json.dumps(rt, default=float))
    print("both natural factors present (S):", {t: S[t]["both_nat"] for t in S}, "rotated+sufficient:", {t: S[t]["rot_sufficient"] for t in S})
    print("oracle agreement (nat+dup, all3, rot, dup):", {t: (S[t]["oracle_nat_dup"], S[t]["oracle_all3"], S[t]["rot_trader"], S[t]["dup"]) for t in S})
    print("falsifiers:", json.dumps(fz, default=float))
    print("stage4 gate:", out["stage4_gate"])
    if m0:
        print("M0:", json.dumps(out["M0"]["prereg"]), "T", m0["T"], "p", m0["p_fixed"])


if __name__ == "__main__":
    main()
