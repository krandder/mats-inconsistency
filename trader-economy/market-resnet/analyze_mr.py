"""Summarize market-resnet results into results/summary.json and charts/*.json.
Phase 1: results/vm/trader-mresnet-*/<arm>_s<seed>.eval.json
Phase 2: results/vm/trader-mresnet-*/p2/<name>_s<seed>.json
(VMs were recreated after spot preemptions; a file on a later VM overrides the same name on an earlier one.)
Numbers: mean over seeds with a 95% t-interval half-width."""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VMS = sorted(glob.glob(os.path.join(HERE, "results/vm/trader-mresnet-*")),
             key=lambda d: int(re.search(r"(\d+)$", d).group(1)))


def latest(sub, pat):
    """basename -> path, later VM wins"""
    got = {}
    for d in VMS:
        for f in glob.glob(os.path.join(d, sub, pat)):
            got[os.path.basename(f)] = f
    return [got[k] for k in sorted(got)]
CH = os.path.join(HERE, "charts")
os.makedirs(CH, exist_ok=True)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def ci(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    m = float(np.mean(v))
    h = float(T95.get(len(v) - 1, 2.0) * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None
    return {"mean": m, "ci95": h, "n": len(v)}


def fmt(c, d=3):
    if c is None:
        return "n/a"
    if c["ci95"] is None:
        return f"{c['mean']:.{d}f} (n={c['n']})"
    return f"{c['mean']:.{d}f} ± {c['ci95']:.{d}f}"


def interp_acc(curve, steps):
    """Accuracy of a halting curve at a given mean step count (linear in steps)."""
    pts = sorted((d["steps"], d["acc"]) for d in curve)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if steps <= xs[0]:
        return ys[0]
    if steps >= xs[-1]:
        return ys[-1]
    return float(np.interp(steps, xs, ys))


def steps_to(curve, target):
    """Fewest mean steps at which the curve reaches the target accuracy (None if never)."""
    ok = [d["steps"] for d in curve if d["acc"] >= target]
    return min(ok) if ok else None


def phase1():
    runs = {}
    for f in latest("", "*_s[0-9].eval.json"):
        m = re.match(r"(.+)_s(\d)\.eval\.json", os.path.basename(f))
        runs.setdefault(m.group(1), {})[int(m.group(2))] = json.load(open(f))
    out = {"arms": {}, "equal_compute": {}, "halting_bug": {}}
    for arm, seeds in runs.items():
        A = {}
        for sname in ["id_8_32", "L32", "L48", "mix_8_64", "L64", "L128"]:
            rs = [r["sets"][sname] for r in seeds.values() if sname in r["sets"]]
            if not rs:
                continue
            Tt = list(seeds.values())[0]["args"]["T"]
            A[sname] = {
                "acc_at_Ttrain": ci([r["fixed"][Tt]["acc"] for r in rs]),
                "acc_at_cap": ci([r["fixed"][-1]["acc"] for r in rs]),
                "best_acc": ci([max(d["acc"] for d in r["fixed"]) for r in rs]),
                "nll_at_cap": ci([r["fixed"][-1]["nll_bit"] for r in rs]),
                "monotone_cap": ci([r["monotone_frac_cap"] for r in rs]),
                "neg_pay_mass_cap": ci([r["neg_pay_mass_cap"] for r in rs]),
                "oracle_steps": ci([r["oracle_first_best"]["steps"] for r in rs]),
            }
            if "calib" in rs[0]:
                A[sname]["calib_corr"] = ci([r["calib"]["corr"] for r in rs])
                A[sname]["calib_corr_within_T"] = ci([r["calib"]["corr_within_train_T"] for r in rs])
        cr = [r["credit"] for r in seeds.values() if "credit" in r]
        if cr:
            A["credit"] = {"share_h": ci([c["share_h"] for c in cr]),
                           "own_b_write_share": ci([c["own_b_write_share"] for c in cr]),
                           "completeness_err": ci([c["completeness_err"] for c in cr]),
                           "h_credit_by_lag": np.mean([c["h_credit_by_lag"] for c in cr], 0).tolist()}
        out["arms"][arm] = A
    # equal compute: priced halting (C family) vs PonderNet (D) vs random halting (E) vs fixed steps (A)
    for sname in ["mix_8_64", "L64", "L128", "id_8_32"]:
        E = {}
        for arm, seeds in runs.items():
            if not arm.startswith("C"):
                continue
            rows = []
            for s, r in seeds.items():
                if sname not in r["sets"]:
                    continue
                cur = r["sets"][sname]["priced"]
                top = max(d["acc"] for d in cur)
                # operating point: cheapest cost whose accuracy is within 0.005 of the curve's best
                op = min((d for d in cur if d["acc"] >= top - 0.005), key=lambda d: d["steps"])
                st = op["steps"]
                row = {"seed": s, "steps": st, "acc": op["acc"], "cost": op["cost"],
                       "E_perm": op["E_perm"]["acc"], "E_geom": op["E_geom"]["acc"],
                       "post_halt_gain": op["post_halt_gain"], "tau_by_len": op["tau_by_len"]}
                for other in ["D", "D_lp0.2"]:
                    if other in runs and s in runs[other] and sname in runs[other][s]["sets"]:
                        row[other] = interp_acc(runs[other][s]["sets"][sname]["ponder"], st)
                        row[other + "_steps_to_same_acc"] = steps_to(runs[other][s]["sets"][sname]["ponder"], op["acc"])
                if "A" in runs and s in runs["A"] and sname in runs["A"][s]["sets"]:
                    fx = runs["A"][s]["sets"][sname]["fixed"]
                    row["A_fixed"] = float(np.interp(st, np.arange(len(fx)), [d["acc"] for d in fx]))
                fx = r["sets"][sname]["fixed"]
                row["self_fixed"] = float(np.interp(st, np.arange(len(fx)), [d["acc"] for d in fx]))
                rows.append(row)
            if rows:
                keys = [k for k in rows[0] if k not in ("seed", "tau_by_len") and all(k in r for r in rows)]
                E[arm] = {k: ci([r[k] for r in rows]) for k in keys if all(r[k] is not None for r in rows)}
                E[arm]["tau_by_len_seed0"] = rows[0]["tau_by_len"]
        out["equal_compute"][sname] = E
    # halting-settlement variants: post-halt gain across costs (larger = self-fulfilling stops)
    for arm in ["C", "Ctt", "Ctt_real", "Ctt_exec", "Ctt_eps"]:
        if arm not in runs:
            continue
        H = {}
        for sname in ["id_8_32", "mix_8_64", "L64"]:
            cs = [r["sets"][sname]["priced"] for r in runs[arm].values() if sname in r["sets"]]
            if cs:
                H[sname] = [{"cost": cs[0][i]["cost"], "steps": ci([c[i]["steps"] for c in cs]),
                             "acc": ci([c[i]["acc"] for c in cs]),
                             "post_halt_gain": ci([c[i]["post_halt_gain"] for c in cs])} for i in range(len(cs[0]))]
        out["halting_bug"][arm] = H
    # nonzero train-time cost sweep: operating point at halting cost c (0.01 / 0.05 / 0.2) for
    # the C family, vs PonderNet (D, D_lp0.2) and fixed steps (A) at the same mean steps,
    # and random halting with the same step histogram (E_perm)
    out["cost_sweep"] = {}
    for sname in ["mix_8_64", "L64", "L128", "id_8_32"]:
        S = {}
        for arm, seeds in runs.items():
            if not arm.startswith("C"):
                continue
            for c in ["0.01", "0.05", "0.2"]:
                rows = []
                for sd, r in seeds.items():
                    if sname not in r["sets"] or "priced_at" not in r["sets"][sname]:
                        continue
                    d = r["sets"][sname]["priced_at"][c]
                    row = {"steps": d["steps"], "acc": d["acc"], "nll_bit": d["nll_bit"],
                           "E_perm": d["E_perm"]["acc"], "post_halt_gain": d["post_halt_gain"]}
                    for other in ["D", "D_lp0.2"]:
                        if other in runs and sd in runs[other] and sname in runs[other][sd]["sets"]:
                            row[other] = interp_acc(runs[other][sd]["sets"][sname]["ponder"], d["steps"])
                    if "A" in runs and sd in runs["A"] and sname in runs["A"][sd]["sets"]:
                        fx = runs["A"][sd]["sets"][sname]["fixed"]
                        row["A_fixed"] = float(np.interp(d["steps"], np.arange(len(fx)), [e["acc"] for e in fx]))
                    rows.append(row)
                if rows:
                    keys = [k for k in rows[0] if all(k in r for r in rows)]
                    S[f"{arm}@{c}"] = {k: ci([r[k] for r in rows]) for k in keys}
        out["cost_sweep"][sname] = S
    return runs, out


def phase2():
    runs = {}
    for f in latest("p2", "*_s[0-9].json"):
        m = re.match(r"(.+)_s(\d)\.json", os.path.basename(f))
        r = json.load(open(f))
        runs.setdefault(m.group(1), {})[int(m.group(2))] = r
    out = {}
    for name, seeds in runs.items():
        o = {"seeds": sorted(seeds)}
        o["final_train_nll"] = ci([r["log"][-1]["final_nll_bit"] for r in seeds.values()])
        o["train_sec"] = ci([r.get("train_sec") for r in seeds.values()])
        o["train_peak_mem_gb"] = ci([r.get("train_peak_mem_gb") for r in seeds.values()])
        ev = {}
        for r in seeds.values():
            for e in r.get("eval", []):
                ev.setdefault((e["L"], e["pick"], e["D"]), []).append((e["acc"], e["nll_bit"]))
        o["depth"] = [{"L": k[0], "pick": k[1], "D": k[2], "acc": ci([a for a, _ in v]), "nll_bit": ci([b for _, b in v])}
                      for k, v in sorted(ev.items())]
        if any("halting" in r for r in seeds.values()):
            hs = [r["halting"] for r in seeds.values() if "halting" in r]
            o["halting"] = [{"kind": hs[0][i]["kind"], "cost": hs[0][i]["cost"],
                             **{k: ci([h[i][k] for h in hs]) for k in
                                ["steps", "acc", "nll_bit", "post_halt_gain", "E_perm_acc", "fixed_acc_same_mean"]}}
                            for i in range(len(hs[0]))]
            o["halting_calib"] = [{"kind": c[0]["kind"], "corr": ci([x["calib_corr"] for x in c]),
                                   "pred_mean": ci([x["pred_mean"] for x in c]), "real_mean": ci([x["real_mean"] for x in c])}
                                  for c in zip(*[r["halting_calib"] for r in seeds.values() if "halting_calib" in r])]
        lg = [r["log"] for r in seeds.values()]
        if "respawns" in lg[0][-1]:
            o["respawns"] = ci([l[-1]["respawns"] for l in lg])
            o["entries"] = ci([l[-1]["entries"] for l in lg])
            o["final_wealth"] = [l[-1]["wealth"] for l in lg]
            o["slot_transfers"] = ci([l[-1]["slot_transfers"] for l in lg])
            # bubble / drift diagnostics: total wealth and its spread over training
            o["wealth_traj"] = [{"it": e["it"], "total": float(np.sum(e["wealth"])), "max_share": float(np.max(e["wealth"]) / max(np.sum(e["wealth"]), 1e-9)),
                                 "nll": e["final_nll_bit"]} for e in lg[0] if "wealth" in e and "it" in e]
        if "bounty_paid" in lg[0][-1]:
            its = [l[-1]["it"] + 1 for l in lg]
            o["bounty_paid_per_ep"] = ci([l[-1]["bounty_paid"] / n for l, n in zip(lg, its)])
            o["funded_frac"] = ci([l[-1]["funded_eps"] / n for l, n in zip(lg, its)])
            o["funded_frac_last500"] = ci([np.mean([e["funded"] for e in l if e.get("it", 0) >= n - 500]) for l, n in zip(lg, its)])
            o["fund_p_final"] = [l[-1].get("fund_p") for l in lg]
        tc = [r["args"].get("train_halt", -1) for r in seeds.values()]
        if "halting" in o and tc[0] > 0:
            # priced halting at the TRAINING cost (voc rule, the one used in training)
            o["at_train_cost"] = [h for h in o["halting"] if h["kind"] == "voc" and abs(h["cost"] - tc[0]) < 1e-12]
        out[name] = o
    # the same halting points for the cost-0 run (post-hoc halting only), for comparison
    if "econ_cov_voc" in out and "halting" in out["econ_cov_voc"]:
        out["_cost0_voc_points"] = [h for h in out["econ_cov_voc"]["halting"] if h["kind"] == "voc" and h["cost"] in (1e-3, 3e-3, 1e-2)]
    md = latest("p2", "memdepth.json")
    if md:
        out["_memdepth"] = json.load(open(md[0]))
    return runs, out


def charts(p1, p2):
    C = {}
    # 1. accuracy vs fixed steps on L64 and L128 (extrapolation), per arm
    for sname in ["L64", "L128", "mix_8_64"]:
        ser = []
        for arm, seeds in sorted(p1[0].items()):
            fx = [r["sets"][sname]["fixed"] for r in seeds.values() if sname in r["sets"]]
            if fx:
                ser.append({"name": arm, "x": list(range(len(fx[0]))),
                            "y": np.mean([[d["acc"] for d in f] for f in fx], 0).round(4).tolist()})
        C[f"p1_fixed_steps_{sname}"] = {"title": f"Exact-sequence accuracy vs fixed steps, {sname} (train T=24, lengths 8-32)",
                                        "xlabel": "steps", "ylabel": "accuracy", "series": ser}
    # 2. halting: accuracy vs mean steps (priced, ponder, random) on mix_8_64, seed 0
    for sname in ["mix_8_64", "L64"]:
        ser = []
        for arm, seeds in sorted(p1[0].items()):
            if 0 not in seeds or sname not in seeds[0]["sets"]:
                continue
            r = seeds[0]["sets"][sname]
            if "priced" in r:
                ser.append({"name": f"{arm} priced", "x": [d["steps"] for d in r["priced"]], "y": [d["acc"] for d in r["priced"]]})
                if arm == "C":
                    ser.append({"name": "E random (C histogram)", "x": [d["steps"] for d in r["priced"]],
                                "y": [d["E_perm"]["acc"] for d in r["priced"]]})
                    ser.append({"name": "E random (geometric)", "x": [d["E_geom"]["steps"] for d in r["priced"]],
                                "y": [d["E_geom"]["acc"] for d in r["priced"]]})
            if "ponder" in r:
                ser.append({"name": f"{arm} PonderNet", "x": [d["steps"] for d in r["ponder"]], "y": [d["acc"] for d in r["ponder"]]})
        C[f"p1_halting_{sname}"] = {"title": f"Accuracy vs mean steps under halting, {sname} (seed 0)",
                                    "xlabel": "mean steps", "ylabel": "accuracy", "series": ser}
    # 2b. cost sweep: operating points at halting cost 0.01/0.05/0.2 (mean over seeds) vs the
    # PonderNet curves (mean over seeds, by threshold q) and random halting (same histogram)
    for sname in ["mix_8_64", "L64"]:
        ser = []
        cs = p1[1]["cost_sweep"].get(sname, {})
        for arm in sorted({k.split("@")[0] for k in cs}):
            pts = [cs[f"{arm}@{c}"] for c in ["0.01", "0.05", "0.2"] if f"{arm}@{c}" in cs]
            ser.append({"name": f"{arm} priced (cost 0.01, 0.05, 0.2)", "x": [q["steps"]["mean"] for q in pts],
                        "y": [q["acc"]["mean"] for q in pts]})
            ser.append({"name": f"{arm} random, same histogram", "x": [q["steps"]["mean"] for q in pts],
                        "y": [q["E_perm"]["mean"] for q in pts]})
        for arm in ["D", "D_lp0.2"]:
            cur = [r["sets"][sname]["ponder"] for r in p1[0].get(arm, {}).values() if sname in r["sets"]]
            if cur:
                ser.append({"name": f"{arm} PonderNet", "x": np.mean([[d["steps"] for d in c] for c in cur], 0).tolist(),
                            "y": np.mean([[d["acc"] for d in c] for c in cur], 0).tolist()})
        C[f"p1_cost_sweep_{sname}"] = {"title": f"Priced halting at three costs vs PonderNet and random halting, {sname} (mean over seeds)",
                                       "xlabel": "mean steps", "ylabel": "accuracy", "series": ser}
    # 3. post-halt gain vs cost (self-fulfilling stops)
    ser = []
    for arm, H in p1[1]["halting_bug"].items():
        if "mix_8_64" in H:
            ser.append({"name": arm, "x": [h["cost"] for h in H["mix_8_64"][1:]],
                        "y": [h["post_halt_gain"]["mean"] for h in H["mix_8_64"][1:]]})
    C["p1_post_halt_gain"] = {"title": "Log-score gain left on the table after the halt (nats/example), mix_8_64",
                              "xlabel": "halting cost", "ylabel": "post-halt gain", "xscale": "log", "series": ser}
    # 4. IG credit by lag
    ser = [{"name": arm, "x": list(range(len(A["credit"]["h_credit_by_lag"]))), "y": A["credit"]["h_credit_by_lag"]}
           for arm, A in p1[1]["arms"].items() if "credit" in A]
    C["p1_credit_by_lag"] = {"title": "Aumann-Shapley credit of h-writes to later payments, by lag",
                             "xlabel": "lag (steps)", "ylabel": "credit (nats)", "series": ser}
    # 5. phase 2 depth scaling
    for L in [32, 320, 3200, 32000]:
        ser = []
        for name, o in sorted(p2[1].items()):
            if name.startswith("_"):
                continue
            for pick in ["roundrobin", "auction", "-"]:
                pts = [d for d in o["depth"] if d["L"] == L and d["pick"] == pick]
                if pts:
                    ser.append({"name": f"{name} {pick}".replace(" -", ""), "x": [d["D"] for d in pts],
                                "y": [d["acc"]["mean"] for d in pts]})
        if ser:
            C[f"p2_depth_L{L}"] = {"title": f"Phase 2: accuracy vs depth at length {L} (trained on 8-32, depth 24)",
                                   "xlabel": "depth (steps)", "ylabel": "accuracy", "xscale": "log", "series": ser}
    ser = []
    for name, o in sorted(p2[1].items()):
        if not name.startswith("_") and o.get("wealth_traj"):
            ser.append({"name": name + " total wealth", "x": [w["it"] for w in o["wealth_traj"]], "y": [w["total"] for w in o["wealth_traj"]]})
    if ser:
        C["p2_wealth"] = {"title": "Total trader wealth over training (seed 0); a bubble would show as wealth growing without the loss falling",
                          "xlabel": "iteration", "ylabel": "wealth (nats)", "series": ser}
    # phase 2 priced halting: accuracy vs mean steps over the voc cost grid (mean over seeds),
    # cost-0 training vs training with a nonzero compute cost, and random halting (same histogram)
    ser = []
    for name, o in sorted(p2[1].items()):
        if name.startswith("_") or not (name == "econ_cov_voc" or "priced" in name or "halt" in name) or "halting" not in o:
            continue
        hv = [h for h in o["halting"] if h["kind"] == "voc" and h["cost"] > 0]
        ser.append({"name": f"{name} voc", "x": [h["steps"]["mean"] for h in hv], "y": [h["acc"]["mean"] for h in hv],
                    "cost": [h["cost"] for h in hv]})
        if name == "econ_cov_voc":
            ser.append({"name": "random halting, same histogram (cost-0 run)", "x": [h["steps"]["mean"] for h in hv],
                        "y": [h["E_perm_acc"]["mean"] for h in hv]})
            ser.append({"name": "fixed depth at the same mean (cost-0 run)", "x": [h["steps"]["mean"] for h in hv],
                        "y": [h["fixed_acc_same_mean"]["mean"] for h in hv]})
    if ser:
        C["p2_priced_halting"] = {"title": "Phase 2: accuracy vs mean steps under VOC halting, lengths 4-64 (mean over seeds)",
                                  "xlabel": "mean steps", "ylabel": "accuracy", "series": ser}
    if "_memdepth" in p2[1]:
        md = p2[1]["_memdepth"]
        C["p2_memdepth"] = {"title": "Peak GPU memory of one training step vs depth (L = depth, batch 16)",
                            "xlabel": "depth", "ylabel": "GB", "xscale": "log",
                            "series": [{"name": m, "x": [r["D"] for r in md if r["mode"] == m and r["ok"]],
                                        "y": [r["peak_gb"] for r in md if r["mode"] == m and r["ok"]]}
                                       for m in sorted({r["mode"] for r in md})]}
    for k, v in C.items():
        json.dump(v, open(os.path.join(CH, k + ".json"), "w"))
    return list(C)


if __name__ == "__main__":
    p1 = phase1()
    p2 = phase2()
    json.dump({"phase1": p1[1], "phase2": p2[1]}, open(os.path.join(HERE, "results/summary.json"), "w"), indent=1)
    print("charts:", charts(p1, p2))
    for arm, A in sorted(p1[1]["arms"].items()):
        print(arm, " ".join(f"{s}:{fmt(A[s]['acc_at_cap'])}" for s in ["id_8_32", "L64", "L128"] if s in A))
    if "-v" in sys.argv:
        for sname in ["mix_8_64", "L64"]:
            print(f"\n# cost sweep {sname}: steps acc | E_perm D D_lp0.2 A_fixed | post_halt_gain")
            for k, v in sorted(p1[1]["cost_sweep"][sname].items()):
                print(k, " ".join(f"{q}={fmt(v.get(q), 3 if q != 'steps' else 1)}" for q in
                                  ["steps", "acc", "E_perm", "D", "D_lp0.2", "A_fixed", "post_halt_gain"]))
        print("\n# phase 2 priced (voc halting at the training cost) vs the cost-0 run")
        for n, o in sorted(p2[1].items()):
            if not n.startswith("_") and o.get("at_train_cost"):
                for h in o["at_train_cost"]:
                    print(n, h["cost"], " ".join(f"{q}={fmt(h[q])}" for q in ["steps", "acc", "E_perm_acc", "fixed_acc_same_mean", "post_halt_gain"]))
        for h in p2[1].get("_cost0_voc_points", []):
            print("econ_cov_voc(cost0)", h["cost"], " ".join(f"{q}={fmt(h[q])}" for q in ["steps", "acc", "E_perm_acc", "fixed_acc_same_mean", "post_halt_gain"]))
