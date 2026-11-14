"""P12 analysis: val selection, MDE, primary test (paired t over seeds + hierarchical bootstrap), readouts.
Reads results/vm/*/results/eval/*.{npz,json} (GPU arms) and results/eval/*.{npz,json} (solver arms, CPU).
Writes results/summary.json and charts/p12_charts.json. Prints the headline lines."""
import glob, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KS = (2, 5, 10, 20)
TESTS = [("test_W9", 9), ("test_W15", 15), ("test_W25", 25), ("test_W33", 33), ("test_W49", 49), ("test_W65", 65)]
PRIMARY = [("test_W33", 33), ("test_W49", 49), ("test_W65", 65)]
SHIFT = [("shift_p0.1_W33", 33), ("shift_p0.5_W33", 33)]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
LOOP_ARMS = ["DT", "DT+fb", "DT+pen", "DT-TD", "DTxARB", "DTxARB-noFB", "DTxARB-bp"]
BASELINES = ["DT+fb", "DT+dec", "DT+ARB(test)", "ARB-kappa"]


def ci(xs):
    xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], float)
    if len(xs) == 0:
        return {"mean": None, "n": 0}
    if len(xs) == 1:
        return {"mean": float(xs[0]), "n": 1}
    h = T975[len(xs) - 1] * xs.std(ddof=1) / math.sqrt(len(xs))
    return {"mean": float(xs.mean()), "lo": float(xs.mean() - h), "hi": float(xs.mean() + h), "sd": float(xs.std(ddof=1)), "n": len(xs)}


def load_evals():
    ev = {}
    for f in sorted(glob.glob(os.path.join(HERE, "results", "vm", "*", "results", "eval", "*.json")) +
                    glob.glob(os.path.join(HERE, "results", "eval", "*.json"))):
        name = os.path.basename(f)[:-5]
        z = np.load(f[:-5] + ".npz")
        ev[name] = {"json": json.load(open(f)), "d": {k: z[k] for k in z.files}}
    return ev


def data_L():
    out = {}
    for s, _ in TESTS + SHIFT + [("val_W25", 25)]:
        p = os.path.join(HERE, "data", s + ".npz")
        if os.path.exists(p):
            out[s] = np.load(p)["L"]
    return out


def main():
    ev = load_evals(); L = data_L()
    gen = json.load(open(os.path.join(HERE, "data", "genstats.json")))
    p99 = gen["pool_W9"]["L_p99"]
    cpf = sorted(glob.glob(os.path.join(HERE, "results", "cpu_phase.json")) +
                 glob.glob(os.path.join(HERE, "results", "vm", "*", "results", "cpu_phase.json")))
    cpu = json.load(open(cpf[0])) if cpf else {}
    runs = {}
    for name, e in ev.items():
        m = re.match(r"(.+)_s(\d+)_w(\d+)$", name)
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = e
    W = sorted({int(re.match(r".+_w(\d+)$", n).group(1)) for n in ev if re.match(r".+_s\d+_w\d+$", n) and not n.startswith("CostNet")} or {32})

    # ---------------------------------------------------------------- per-seed per-maze exact vectors at the val-selected k
    def loop_sel(e):
        d = e["d"]
        vals = {k: float(np.mean(d[f"val_W25/exact@{k * 25}"])) for k in KS if f"val_W25/exact@{k * 25}" in d}
        k = max(vals, key=vals.get)
        return k, vals

    per = {}   # per[arm][seed] = {"k":..., "val":..., set: {"exact": arr, "valid": arr, ...}}
    for arm in LOOP_ARMS:
        for seed, e in runs.get(arm, {}).items():
            k, vals = loop_sel(e); d = e["d"]
            r = {"k": k, "val": vals[k], "val_by_k": vals}
            for s, n in TESTS + SHIFT + [("val_W25", 25)]:
                t = k * n
                if f"{s}/exact@{t}" in d:
                    r[s] = {m: d[f"{s}/{m}@{t}"] for m in ("exact", "valid", "synd", "f1", "costratio", "sem_mean", "sem_sum", "sem_max")}
                    if f"{s}/u_exact@{t}" in d:
                        r[s]["u_exact"] = d[f"{s}/u_exact@{t}"]; r[s]["u_valid"] = d[f"{s}/u_valid@{t}"]
            per.setdefault(arm, {})[seed] = r
            if arm == "DT":
                sel = e["json"].get("selected_posthoc", {})
                for pa in ("DT+dec", "DT+ARB(test)"):
                    if pa not in sel:
                        continue
                    rr = {"k": sel[pa]["k"], "val": sel[pa]["val_exact"], "param": sel[pa].get("c", sel[pa].get("param"))}
                    for s, n in TESTS + SHIFT:
                        if f"{s}/{pa}/exact" in d:
                            rr[s] = {m: d[f"{s}/{pa}/{m}"] for m in ("exact", "valid", "f1", "costratio")}
                    per.setdefault(pa, {})[seed] = rr
    for seed, e in runs.get("CostNet+BB", {}).items():
        d = e["d"]; rr = {"k": None, "val": float(np.mean(d["val_W25/exact"])) if "val_W25/exact" in d else None}
        for s, n in TESTS + SHIFT:
            if f"{s}/exact" in d:
                rr[s] = {m: d[f"{s}/{m}"] for m in ("exact", "valid", "f1", "costratio")}
        per.setdefault("CostNet+BB", {})[seed] = rr
    solver = {}
    if "arbk" in ev:
        d = ev["arbk"]["d"]
        grid = sorted({k.split("/")[1] for k in d if k.startswith("val_W25/")})
        best = max(grid, key=lambda g: float(np.mean(d[f"val_W25/{g}/exact"])))
        rr = {"k": None, "val": float(np.mean(d[f"val_W25/{best}/exact"])), "param": best,
              "val_grid": {g: float(np.mean(d[f"val_W25/{g}/exact"])) for g in grid}}
        for s, n in TESTS + SHIFT:
            if f"{s}/{best}/exact" in d:
                rr[s] = {m: d[f"{s}/{best}/{m}"] for m in ("exact", "valid", "f1", "costratio")}
        seeds = sorted(per.get("DTxARB", {0: 0}).keys())
        per["ARB-kappa"] = {sd: rr for sd in seeds}   # deterministic: same vector for every seed
        solver["ARB-kappa"] = {k: v for k, v in rr.items() if not k.startswith(("test", "shift"))}
    for nm in ("a0flow", "a0deg"):
        if nm in ev:
            solver[nm] = {k: float(np.nanmean(v)) for k, v in ev[nm]["d"].items() if k.endswith(("/exact", "/valid"))}

    def primary_vec(r, metric="exact"):
        xs = []; ls = []
        for s, n in PRIMARY:
            if s not in r:
                return None
            msk = L[s] > p99
            xs.append(np.asarray(r[s][metric], float)[msk])
        return np.concatenate(xs)

    # ---------------------------------------------------------------- table: exact by n (mean over seeds, 95% CI)
    table = {}
    for arm, sd in per.items():
        row = {"seeds": sorted(sd.keys()), "k_selected": {s: sd[s]["k"] for s in sd}, "val": ci([sd[s]["val"] for s in sd])}
        for s, n in TESTS + SHIFT:
            row[s] = {m: ci([float(np.nanmean(sd[x][s][m])) for x in sd if s in sd[x]]) for m in ("exact", "valid", "f1", "costratio")}
        row["primary"] = ci([float(np.mean(primary_vec(sd[x]))) for x in sd if primary_vec(sd[x]) is not None])
        row["primary_valid"] = ci([float(np.mean(primary_vec(sd[x], "valid"))) for x in sd if primary_vec(sd[x]) is not None])
        table[arm] = row

    # ---------------------------------------------------------------- MDE (val only) and the primary paired test
    res = {"pool_L_p99": p99, "W": W}
    main_arm = "DTxARB"; seeds = sorted(per.get(main_arm, {}).keys())
    best_bl = {}; val_diffs = []; diffs = []; pairs = []
    for s in seeds:
        cands = {b: per[b][s]["val"] for b in BASELINES if b in per and s in per[b] and per[b][s]["val"] is not None}
        if not cands:
            continue
        b = max(cands, key=cands.get); best_bl[s] = b
        val_diffs.append(per[main_arm][s]["val"] - cands[b])
        pm = primary_vec(per[main_arm][s]); pb = primary_vec(per[b][s])
        if pm is not None and pb is not None:
            diffs.append(float(pm.mean() - pb.mean())); pairs.append((pm, pb))
    if len(val_diffs) >= 2:
        res["MDE_from_val"] = {"sd_val_paired_diff": float(np.std(val_diffs, ddof=1)),
                               "MDE": float(2.78 * np.std(val_diffs, ddof=1) / math.sqrt(len(val_diffs))), "n_seeds": len(val_diffs)}
    res["best_baseline_per_seed"] = best_bl
    res["primary_diff_per_seed"] = diffs
    res["primary_paired_t"] = ci(diffs)
    if pairs:
        rng = np.random.default_rng(0); B = 10000; boots = np.empty(B)
        for i in range(B):
            ss = rng.integers(len(pairs), size=len(pairs)); v = []
            for j in ss:
                pm, pb = pairs[j]; mi = rng.integers(len(pm), size=len(pm))
                v.append(pm[mi].mean() - pb[mi].mean())
            boots[i] = np.mean(v)
        res["primary_hier_bootstrap"] = {"lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                                         "mean": float(boots.mean()), "B": B}
    # best baseline by mean (for the falsification rule)
    bl_means = {b: table[b]["primary"]["mean"] for b in BASELINES if b in table and table[b]["primary"]["mean"] is not None}
    res["baseline_primary_means"] = bl_means
    main_primary = table.get(main_arm, {}).get("primary", {}).get("mean")
    v49 = table.get(main_arm, {}).get("test_W49", {}).get("valid", {}).get("mean")
    n9_main = table.get(main_arm, {}).get("test_W9", {}).get("exact", {}).get("mean")
    n9_dt = table.get("DT", {}).get("test_W9", {}).get("exact", {}).get("mean")
    pt = res["primary_paired_t"]; hb = res.get("primary_hier_bootstrap", {})
    succ = (pt.get("mean") is not None and pt["mean"] >= 0.05 and pt.get("lo", -1) > 0 and hb.get("lo", -1) > 0
            and v49 is not None and v49 >= 0.99 and n9_main is not None and n9_dt is not None and n9_main >= n9_dt - 0.02)
    # no DTxARB (or no baseline) result yet: the verdict is incomplete, neither success nor falsified
    fals = (main_primary is not None and bool(bl_means)
            and (main_primary <= max(bl_means.values()) or (v49 is not None and v49 < 0.95)))
    res["verdict"] = {"complete": main_primary is not None and bool(bl_means),
                      "success": bool(succ), "falsified": bool(fals), "main_primary": main_primary,
                      "best_baseline_primary_by_mean": max(bl_means.items(), key=lambda x: x[1]) if bl_means else None,
                      "DTxARB_valid_n49": v49, "DTxARB_exact_n9": n9_main, "DT_exact_n9": n9_dt}

    # ---------------------------------------------------------------- readouts
    ro = {}
    # attribution: proposer u vs prices b inside the arbitrageur arms; arms vs each other on the primary cell
    for arm in ("DTxARB", "DTxARB-noFB", "DTxARB-bp"):
        if arm in per:
            ro[f"{arm}_proposer_vs_prices_primary"] = {
                "prices_b": ci([float(np.mean(primary_vec(per[arm][s]))) for s in per[arm] if primary_vec(per[arm][s]) is not None]),
                "proposer_u": ci([float(np.mean(primary_vec(per[arm][s], "u_exact"))) for s in per[arm]
                                  if "test_W33" in per[arm][s] and "u_exact" in per[arm][s]["test_W33"] and primary_vec(per[arm][s], "u_exact") is not None])}
    for a, b in (("DTxARB", "DTxARB-noFB"), ("DTxARB", "DTxARB-bp"), ("DTxARB", "DT+fb"), ("DTxARB", "DT"), ("DT+fb", "DT"),
                 ("DT+pen", "DT"), ("DT+ARB(test)", "DT"), ("DT+dec", "DT"), ("DT-TD", "DT")):
        if a in per and b in per:
            common = sorted(set(per[a]) & set(per[b]))
            ds = [float(np.mean(primary_vec(per[a][s])) - np.mean(primary_vec(per[b][s]))) for s in common
                  if primary_vec(per[a][s]) is not None and primary_vec(per[b][s]) is not None]
            ro[f"primary_diff[{a} - {b}]"] = ci(ds)
    # MATS replication: post-hoc consistency projection on DT's logits vs DT, by n
    if "DT+ARB(test)" in per and "DT" in per:
        ro["MATS_replication_posthoc_projection_minus_DT_by_n"] = {
            s: ci([float(np.mean(per["DT+ARB(test)"][x][s]["exact"]) - np.mean(per["DT"][x][s]["exact"])) for x in per["DT"]
                   if x in per["DT+ARB(test)"] and s in per["DT+ARB(test)"][x] and s in per["DT"][x]]) for s, _ in TESTS}
    # coherence-pay blindness: of non-exact predictions, the fraction that are degree-valid (syndrome zero)
    cb = {}
    for arm in LOOP_ARMS:
        if arm not in per:
            continue
        for s, _ in PRIMARY:
            v = []
            for x in per[arm]:
                if s in per[arm][x]:
                    ex = per[arm][x][s]["exact"].astype(bool); sy = per[arm][x][s]["synd"].astype(bool)
                    if (~ex).sum():
                        v.append(float(sy[~ex].mean()))
            cb[f"{arm}/{s}"] = ci(v)
    ro["coherence_pay_blindness_frac_nonexact_degree_valid"] = cb
    # per-cell AUROC (semantic loss -> cell error), split by maze valid / invalid, at the selected k, n = 33
    au = {}
    for arm in LOOP_ARMS:
        for x, e in runs.get(arm, {}).items():
            k = per[arm][x]["k"]
            for tag in ("valid", "invalid"):
                v = e["json"]["cell_auroc"].get(f"test_W33/k{k}/{tag}")
                au.setdefault(f"{arm}/{tag}", []).append(v)
    ro["cell_auroc_semloss_n33"] = {k: ci(v) for k, v in au.items()}
    # maze-level error detection: AUROC of total semantic loss for "not exact", primary sets pooled
    def auroc(score, lab):
        score = np.asarray(score, float); lab = np.asarray(lab, bool)
        if lab.sum() == 0 or (~lab).sum() == 0:
            return None
        from scipy.stats import rankdata
        r = rankdata(score); return float((r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))
    ed = {}
    for arm in LOOP_ARMS:
        if arm in per:
            ed[arm] = ci([auroc(primary_vec(per[arm][x], "sem_sum"), ~primary_vec(per[arm][x]).astype(bool)) for x in per[arm]
                          if primary_vec(per[arm][x]) is not None])
    ro["maze_error_detection_auroc_semsum_primary"] = ed
    # semantic loss vs t, halting, first-exact vs L, overthinking (n = 33), FW arbitrage vs semantic loss
    curves = {}; halt = {}; solve = {}; over = {}; fw = {}
    for arm in LOOP_ARMS:
        for x, e in runs.get(arm, {}).items():
            d = e["d"]; s = "test_W33"
            ts = sorted(int(k.split("@")[1]) for k in d if k.startswith(f"{s}/sem_mean@"))
            curves.setdefault(arm, []).append({"t": ts, "sem_mean": [float(np.mean(d[f"{s}/sem_mean@{t}"])) for t in ts],
                                               "exact": [float(np.mean(d[f"{s}/exact@{t}"])) for t in ts]})
            for hk in ("stab", "synhalt", "softhalt"):
                tt = d[f"{s}/{hk}_t"]; exx = d[f"{s}/{hk}_exact"]
                halt.setdefault(f"{arm}/{hk}", []).append({"halted_frac": float((tt > 0).mean()), "exact_at_halt": float(exx.mean()),
                                                           "median_t": float(np.median(tt[tt > 0])) if (tt > 0).any() else None})
            fe = d[f"{s}/first_exact"]; bins = [(0, 40), (40, 60), (60, 1e9)]
            solve.setdefault(arm, []).append({f"L{a}-{b}": (float(np.median(fe[(L[s] >= a) & (L[s] < b) & (fe > 0)]))
                                                            if ((L[s] >= a) & (L[s] < b) & (fe > 0)).any() else None) for a, b in bins})
            peak = max(float(np.mean(d[f"{s}/exact@{t}"])) for t in ts)
            over.setdefault(arm, []).append({"peak_over_t": peak, "at_20n": float(np.mean(d[f"{s}/exact@{20 * 33}"])),
                                             "first_exact_then_lost": float(((fe > 0) & (d[f"{s}/exact@{20 * 33}"] == 0)).mean())})
            k = per[arm][x]["k"]
            if f"{s}/fw_arb_mean@{k * 33}" in d:
                a_ = d[f"{s}/fw_arb_mean@{k * 33}"]; b_ = d[f"{s}/fw_sem_mean@{k * 33}"]
                fw.setdefault(arm, []).append({"fw_arb_mean": float(np.nanmean(a_)), "sem_mean": float(np.nanmean(b_)),
                                               "corr": float(np.corrcoef(a_[np.isfinite(a_)], b_[np.isfinite(a_)])[0, 1]) if np.isfinite(a_).sum() > 2 else None})
    ro["sem_loss_vs_t_n33"] = curves; ro["halting_n33"] = halt; ro["first_exact_median_by_L_n33"] = solve
    ro["overthinking_n33"] = over; ro["fw_arbitrage_n33_first50"] = fw
    # exact by L bin on the pooled primary sets
    lb = {}
    for arm, sd in per.items():
        for a, b in ((0, 40), (40, 60), (60, 100), (100, 1e9)):
            v = []
            for x in sd:
                if all(s in sd[x] for s, _ in PRIMARY):
                    ex = np.concatenate([np.asarray(sd[x][s]["exact"], float) for s, _ in PRIMARY])
                    LL = np.concatenate([L[s] for s, _ in PRIMARY]); m = (LL >= a) & (LL < b)
                    if m.any():
                        v.append(float(ex[m].mean()))
            lb[f"{arm}/L{a}-{int(b) if b < 1e9 else 'inf'}"] = ci(v)
    ro["exact_by_L_primary"] = lb
    # compute
    comp = {}
    for arm in list(LOOP_ARMS) + ["CostNet+BB", "U1-DT"]:
        for x, e in runs.get(arm, {}).items():
            inf = e["json"]["info"]
            comp.setdefault(arm, []).append({"train_s": inf.get("train_seconds"), "train_peak_MB": inf.get("peak_mem_MB"),
                                             "eval_s": e["json"].get("total_s"), "eval_peak_MB": e["json"].get("peak_mem_MB"),
                                             "params": inf.get("params")})
    ro["compute"] = comp
    # U1 anchor
    if "U1-DT" in runs:
        d = runs["U1-DT"][0]["d"]
        ro["U1_anchor"] = {s: {f"k{k}": float(np.mean(d[f"{s}/exact@{k * n}"])) for k in KS if f"{s}/exact@{k * n}" in d}
                           for s, n in (("doc_U1_9", 9), ("doc_U1_33", 33), ("doc_U1_65", 65), ("doc_U2_33", 33), ("gate_W9", 9))}
    out = {"results": res, "table": table, "readouts": ro, "solver": solver, "cpu_phase": cpu, "genstats": gen}
    os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1, default=float)
    charts = {"exact_vs_n": {arm: {"n": [n for _, n in TESTS], "mean": [table[arm][s]["exact"]["mean"] for s, _ in TESTS],
                                   "lo": [table[arm][s]["exact"].get("lo") for s, _ in TESTS], "hi": [table[arm][s]["exact"].get("hi") for s, _ in TESTS]}
                             for arm in table},
              "primary_diff_per_seed": {"diffs": diffs, "best_baseline": best_bl},
              "sem_loss_vs_t_n33": {arm: v for arm, v in curves.items()}}
    json.dump(charts, open(os.path.join(HERE, "charts", "p12_charts.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    for arm in table:
        print(f"{arm:14s} primary {table[arm]['primary']}  n9 {table[arm]['test_W9']['exact'].get('mean')}")


if __name__ == "__main__":
    main()
