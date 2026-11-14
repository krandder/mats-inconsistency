"""Summarise M6 w1 runs -> results/SUMMARY-w1.csv, results/verdict-w1.json. Re-runnable check."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or"]
runs = {}
for f in sorted(glob.glob(f"{R}/*-s[0-9].json")):
    d = json.load(open(f)); runs[(d["regime"], d["arm"], d["seed"])] = d


def get(reg, arm):
    """arm name -> {seed: score dict}; ES/B-ES derived from A/B ME runs; L0 from A's market."""
    out = {}
    for (rg, a, s), d in runs.items():
        if rg != reg:
            continue
        if a == arm:
            out[s] = d["score"]
        elif arm == "ES" and a == "A" and "score_es" in d:
            out[s] = d["score_es"]
        elif arm == "B-ES" and a == "B" and "score_es" in d:
            out[s] = d["score_es"]
        elif arm == "L0" and a == "A":
            out[s] = d["score_L0"]
    return out


def ms(v):
    v = np.array(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v))


rows = []
tab = {}
for reg in ("P", "ME"):
    for arm in ("A", "B", "ES", "B-ES", "WD", "BOOST", "L0"):
        g = get(reg, arm)
        if not g:
            continue
        tab[(reg, arm)] = g
        seeds = sorted(g)
        cols = {"learn_bce": [g[s]["learn_bce"] for s in seeds], "humility": [g[s]["humility"] for s in seeds],
                "viol_all": [g[s]["viol_all"] for s in seeds], "lfsr_learnt": [g[s]["lfsr_learnt"] for s in seeds]}
        for f in FAM:
            cols["bce_" + f] = [g[s]["fam"][f]["bce"] for s in seeds]
        cols["ece_all"] = [g[s]["fam"]["all"]["ece"] for s in seeds]
        for k, v in cols.items():
            m, sd, n = ms(v)
            rows.append(dict(regime=reg, arm=arm, metric=k, mean=m, sd=sd, n=n,
                             seeds="/".join(map(str, seeds)), values="/".join(f"{x:.4f}" for x in v)))
with open(f"{R}/SUMMARY-w1.csv", "w") as fh:
    fh.write("regime,arm,metric,mean,sd,n,seeds,values\n")
    for r in rows:
        fh.write(f"{r['regime']},{r['arm']},{r['metric']},{r['mean']:.4f},{r['sd']:.4f},{r['n']},{r['seeds']},{r['values']}\n")

# stop rule: B killed iff B >= 1.01*A learn_bce in 3/3 seeds in BOTH regimes
verdict = {"rule": "B killed iff learn_bce(B) >= 1.01*learn_bce(A) in 3/3 seeds in both P and ME"}
killed = []
for reg in ("P", "ME"):
    A, B = tab.get((reg, "A"), {}), tab.get((reg, "B"), {})
    ss = sorted(set(A) & set(B))
    worse = [B[s]["learn_bce"] >= 1.01 * A[s]["learn_bce"] for s in ss]
    verdict[f"{reg}_B_worse_by_1pct"] = dict(zip(map(str, ss), worse))
    killed.append(len(ss) == 3 and all(worse))
verdict["complete"] = all(len(tab.get((r, a), {})) == 3 for r in ("P", "ME") for a in ("A", "B", "WD", "BOOST"))
verdict["B_killed"] = all(killed) if verdict["complete"] else None
verdict["base_arms_survive"] = (not verdict["B_killed"]) if verdict["complete"] else None
# humility (ME): B vs best kill control per paired seed
kc = [tab.get(("ME", a), {}) for a in ("ES", "WD", "BOOST")]
Bm = tab.get(("ME", "B"), {})
ss = sorted(set(Bm).intersection(*[set(k) for k in kc])) if all(kc) else []
verdict["ME_humility_B_minus_bestkill"] = {str(s): Bm[s]["humility"] - min(k[s]["humility"] for k in kc) for s in ss}
verdict["ME_learn_B_minus_bestkill"] = {str(s): Bm[s]["learn_bce"] - min(k[s]["learn_bce"] for k in kc) for s in ss}
verdict["B_beats_kill_humility_3of3"] = bool(ss) and len(ss) == 3 and all(v < 0 for v in verdict["ME_humility_B_minus_bestkill"].values())
json.dump(verdict, open(f"{R}/verdict-w1.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
for r in rows:
    if r["metric"] in ("learn_bce", "humility", "viol_all", "lfsr_learnt"):
        print(f"{r['regime']:3} {r['arm']:6} {r['metric']:12} {r['mean']:.4f} ± {r['sd']:.4f}  [{r['values']}]")

# paired differences vs A (same seed = same stream and init)
with open(f"{R}/PAIRED-w1.csv", "w") as fh:
    fh.write("regime,arm,metric,diff_mean,diff_sd,values\n")
    for reg in ("P", "ME"):
        A = tab[(reg, "A")]
        for arm in ("B", "ES", "B-ES", "WD", "BOOST", "L0"):
            if (reg, arm) not in tab:
                continue
            for met in ("learn_bce", "humility"):
                v = [tab[(reg, arm)][s][met] - A[s][met] for s in sorted(A)]
                m, sd, _ = ms(v)
                fh.write(f"{reg},{arm},{met},{m:.4f},{sd:.4f},{'/'.join(f'{x:+.4f}' for x in v)}\n")
print(open(f"{R}/PAIRED-w1.csv").read())
