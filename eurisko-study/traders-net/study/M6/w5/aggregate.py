"""Aggregate results/*.json into results/summary.csv (mean, sd over seeds) and check the stop rule."""
import glob, json, os, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = {}
for f in sorted(glob.glob(os.path.join(R, "*-s[0-9].json"))):
    r = json.load(open(f)); rows.setdefault(r["arm"], []).append(r)


def get(r, path):
    for k in path.split("."):
        if r is None: return None
        r = r.get(k)
    return r


METRICS = ["learn_bce", "noise_excess", "learn_bce_l0",
           "measures_vs_oracle.G.spearman", "measures_vs_oracle.S.spearman", "measures_vs_oracle.K.spearman",
           "within_question_spearman.G.vs_loo.mean", "within_question_spearman.S.vs_loo.mean",
           "within_question_spearman.K.vs_loo.mean", "within_question_spearman.G.vs_alone.mean",
           "within_question_spearman.K.vs_alone.mean", "pooled_vs_alone.G", "pooled_vs_alone.K",
           "measures_vs_oracle.G.sign_agree_useful", "measures_vs_oracle.K.sign_agree_useful",
           "frac_useful", "head_corr_mean", "gini_K.gini", "gini_K.z", "final_max_stake_mean",
           "book_market_gain", "book_sumK"]
out = []
for arm in ["L0", "U", "Cs", "Cm", "Cw"]:
    for m in METRICS:
        v = [get(r, m) for r in sorted(rows.get(arm, []), key=lambda r: r["seed"])]
        v = [x for x in v if x is not None]
        if not v: continue
        out.append({"arm": arm, "metric": m, "n": len(v), "mean": float(np.mean(v)),
                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                    "per_seed": " ".join(f"{x:.5g}" for x in v)})
with open(os.path.join(R, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
# stop rule: committed arm BCE >= U*1.01 in 3/3 seeds -> dropped
U = {r["seed"]: r["learn_bce"] for r in rows["U"]}
Ug = {r["seed"]: r["within_question_spearman"]["G"]["vs_loo"]["mean"] for r in rows["U"]}
verdict = {}
for arm in ["Cs", "Cm", "Cw"]:
    rr = {r["seed"]: r for r in rows.get(arm, [])}
    worse = [rr[s]["learn_bce"] >= 1.01 * U[s] for s in sorted(rr)]
    h2 = [rr[s]["within_question_spearman"]["K"]["vs_loo"]["mean"] >= Ug[s] + 0.2 for s in sorted(rr)]
    verdict[arm] = {"bce_worse_1pct": worse, "dropped": len(worse) == 3 and all(worse),
                    "H2_K_beats_G_by_0.2": h2, "rel_bce_vs_U": [rr[s]["learn_bce"] / U[s] - 1 for s in sorted(rr)]}
json.dump(verdict, open(os.path.join(R, "stoprule.json"), "w"), indent=1)
for o in out: print(f'{o["arm"]:3s} {o["metric"]:45s} {o["mean"]:+.4f} ± {o["sd"]:.4f}  [{o["per_seed"]}]')
print(json.dumps(verdict, indent=1))
