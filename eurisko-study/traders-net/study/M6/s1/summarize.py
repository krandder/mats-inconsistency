"""M6 s1: re-reads results/*.json -> results/SUMMARY-s1.csv, results/PAIRED-s1.csv, results/verdict-s1.json.
Stop rule per PREREG.md. Run: python3 summarize.py"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
ARMS = ["A", "B", "CG", "CQ", "FB01", "FB05", "WD", "BOOST", "ES"]
POOLS = ["B", "CG", "CQ", "FB01", "FB05"]
KILL = ["ES", "WD", "BOOST"]
SEEDS = [1, 2, 3]
runs = {}
for f in glob.glob(f"{R}/[MI][SE]-*-s[0-9].json"):
    r = json.load(open(f)); runs[(r["regime"], r["arm"], r["seed"])] = r


def get(reg, arm, s):
    if arm == "ES":
        r = runs.get((reg, "A", s)); return None if r is None else r["score_es"]
    r = runs.get((reg, arm, s)); return None if r is None else r["score"]


def val(sc, key):
    return None if sc is None else sc.get(key)


KEYS = ["learn_bce", "humility", "viol_all", "ie_regret", "lfsr_learnt"]
rows, paired, verdict = [], [], {"complete": True, "regimes": {}}
for reg in ["MS", "IE"]:
    V = {}
    for arm in ARMS:
        line = {"regime": reg, "arm": arm}
        for k in KEYS:
            xs = [val(get(reg, arm, s), k) for s in SEEDS]
            xs = [x for x in xs if x is not None]
            line[k + "_mean"] = float(np.mean(xs)) if xs else None
            line[k + "_sd"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
            line["n"] = len(xs)
        if arm != "ES":
            line["es_epoch"] = [runs[(reg, "A" if arm == "ES" else arm, s)]["es_epoch"] for s in SEEDS if (reg, arm, s) in runs]
            line["train_humility_final"] = float(np.mean([runs[(reg, arm, s)]["curve"][-1]["train_humility"] for s in SEEDS if (reg, arm, s) in runs] or [np.nan]))
        rows.append(line)
        if line["n"] < 3:
            verdict["complete"] = False
    for arm in POOLS:
        a_l = [val(get(reg, "A", s), "learn_bce") for s in SEEDS]
        x_l = [val(get(reg, arm, s), "learn_bce") for s in SEEDS]
        if None in a_l + x_l:
            continue
        rel = [x / a - 1 for x, a in zip(x_l, a_l)]
        bce_clause = sum(r_ >= 0.01 for r_ in rel)
        hum = [val(get(reg, arm, s), "humility") for s in SEEDS]
        kmin = [min(val(get(reg, k, s), "humility") for k in KILL) for s in SEEDS]
        prot_h = sum(h < m for h, m in zip(hum, kmin))
        prot_ie = None
        if reg == "IE":
            ie = [val(get(reg, arm, s), "ie_regret") for s in SEEDS]
            iemin = [min(val(get(reg, k, s), "ie_regret") for k in KILL) for s in SEEDS]
            prot_ie = sum(x <= m for x, m in zip(ie, iemin))
            paired.append({"regime": reg, "arm": arm, "vs": "min_kill", "metric": "ie_regret",
                           "diff": [x - m for x, m in zip(ie, iemin)]})
        lmin = [min(val(get(reg, k, s), "learn_bce") for k in KILL) for s in SEEDS]
        beats = sum(x < l and h < m for x, l, h, m in zip(x_l, lmin, hum, kmin))
        protected = prot_h == 3 or prot_ie == 3
        V[arm] = {"rel_learn_vs_A": rel, "bce_clause_seeds": bce_clause, "humility_below_min_kill_seeds": prot_h,
                  "ie_regret_le_min_kill_seeds": prot_ie, "beats_kill_controls_seeds": beats,
                  "dropped": bool(bce_clause == 3 and not protected)}
        paired.append({"regime": reg, "arm": arm, "vs": "A", "metric": "learn_bce_rel", "diff": rel})
        paired.append({"regime": reg, "arm": arm, "vs": "min_kill", "metric": "humility", "diff": [h - m for h, m in zip(hum, kmin)]})
    verdict["regimes"][reg] = V
surv = [a for a in POOLS if all(not verdict["regimes"][r].get(a, {}).get("dropped", False) for r in ("MS", "IE"))
        and all(a in verdict["regimes"][r] for r in ("MS", "IE"))]
verdict["stress_survivors"] = surv
verdict["any_beats_kill_controls_3of3"] = [f"{r}:{a}" for r in ("MS", "IE") for a, v in verdict["regimes"][r].items()
                                           if v["beats_kill_controls_seeds"] == 3]
import csv
with open(f"{R}/SUMMARY-s1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["es_epoch", "train_humility_final"], extrasaction="ignore")
    w.writeheader(); [w.writerow(r) for r in rows]
with open(f"{R}/PAIRED-s1.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["regime", "arm", "vs", "metric", "s1", "s2", "s3", "mean", "sd"])
    for p in paired:
        d = p["diff"]; w.writerow([p["regime"], p["arm"], p["vs"], p["metric"], *[round(x, 5) for x in d],
                                   round(float(np.mean(d)), 5), round(float(np.std(d, ddof=1)), 5)])
json.dump(verdict, open(f"{R}/verdict-s1.json", "w"), indent=1)
for r in rows:
    print(r["regime"], r["arm"], r["n"], *(f"{k}={r[k+'_mean']:.4f}±{r[k+'_sd']:.4f}" for k in ("learn_bce", "humility", "ie_regret") if r.get(k + "_sd") is not None))
print(json.dumps({k: v for k, v in verdict.items() if k != "regimes"}))

# ---- secondary (not gating): ex-LFSR16 learnable, each arm's own early-stopped score, L0, pool k_eff, train humility
EXL = ["periodic", "ca110", "prime", "and", "or"]
sec = {}
for reg in ["MS", "IE"]:
    for arm in ["A", "B", "CG", "CQ", "FB01", "FB05", "WD", "BOOST", "L0"]:
        rs = [runs.get((reg, "A" if arm == "L0" else arm, s)) for s in SEEDS]
        if None in rs:
            continue
        sc = [r["score_L0"] if arm == "L0" else r["score"] for r in rs]
        d = {"learn_exlfsr": [float(np.mean([x["fam"][f]["bce"] for f in EXL])) for x in sc],
             "learn": [x["learn_bce"] for x in sc], "humility": [x["humility"] for x in sc],
             "lfsr_learnt": [x["lfsr_learnt"] for x in sc], "ie_regret": [x.get("ie_regret") for x in sc]}
        if arm != "L0":
            d.update({"own_es_learn": [r["score_es"]["learn_bce"] for r in rs], "own_es_humility": [r["score_es"]["humility"] for r in rs],
                      "own_es_exlfsr": [float(np.mean([r["score_es"]["fam"][f]["bce"] for f in EXL])) for r in rs],
                      "own_es_epoch": [r["es_epoch"] for r in rs], "train_humility_final": [r["curve"][-1]["train_humility"] for r in rs],
                      "k_eff_final": [r["curve"][-1].get("k_eff") for r in rs]})
        for k in list(d):
            xs = [x for x in d[k] if x is not None]
            if xs and not isinstance(xs[0], int):
                d[k + "_ms"] = [round(float(np.mean(xs)), 5), round(float(np.std(xs, ddof=1)), 5)]
        sec[f"{reg}:{arm}"] = d
json.dump(sec, open(f"{R}/SECONDARY-s1.json", "w"), indent=1)
for k, d in sec.items():
    print(k, "exlfsr", d["learn_exlfsr_ms"], "ownES", d.get("own_es_learn_ms"), d.get("own_es_humility_ms"), "ie", d.get("ie_regret_ms"), "lfsr", d["lfsr_learnt"])
