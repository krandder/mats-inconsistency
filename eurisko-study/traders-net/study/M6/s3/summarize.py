"""Rebuild SUMMARY.csv and VERDICT.json from results/li-<arm>-s<seed>.json (PREREG stop rules)."""
import csv, glob, json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["a", "b", "cg", "cq", "fb01", "fb05", "wd", "boost"]
TEST = ["b", "cg", "cq", "fb01", "fb05"]
SEEDS = [1, 2, 3]
R = {}
for f in glob.glob(os.path.join(HERE, "results", "li-*.json")):
    r = json.load(open(f)); R[(r["arm"], r["seed"])] = r


def get(r, path):
    x = r
    for k in path:
        x = x[k]
    return float(x)


METRICS = {
    "P1_Dlearnable_bce": ("net", "P1"), "P2_pre_prove_viol": ("net", "P2"), "P3_twin_viol": ("net", "P3"),
    "twin_viol_D64": ("net", "twin_viol_D64_half"), "twin_viol_D512": ("net", "twin_viol_D512_half"),
    "prove_bce_pre": ("net", "prove_bce_pre"), "prove_bce_post": ("net", "prove_bce_post"),
    "and_viol": ("net", "viol", "and", "mean"), "or_viol": ("net", "viol", "or", "mean"),
    "ex3_viol_pre": ("net", "viol", "pre:ex3", "mean"), "ex3_viol_all": ("net", "viol", "ex3", "mean"),
    "impl_viol_pre": ("net", "viol", "pre:impl", "mean"),
}
for fam in ["delay0", "delay64", "delay512", "and", "or", "prove", "exp64", "exp256", "exp1024",
            "twin64", "solo64", "twin512", "solo512", "base0"]:
    METRICS[f"bce:{fam}"] = ("net", "bce_half", fam)
MAKER = {"P1_Dlearnable_bce": ("maker", "P1"), "P2_pre_prove_viol": ("maker", "P2"), "P3_twin_viol": ("maker", "P3"),
         "and_viol": ("maker", "viol", "and", "mean"), "or_viol": ("maker", "viol", "or", "mean"),
         "prove_bce_pre": ("maker", "prove_bce_pre"), "prove_bce_post": ("maker", "prove_bce_post"),
         "twin_viol_D64": ("maker", "twin_viol_D64_half"), "twin_viol_D512": ("maker", "twin_viol_D512_half")}
for fam in ["delay0", "delay64", "delay512", "and", "or", "exp64", "exp256", "exp1024", "twin64", "solo64", "twin512", "solo512"]:
    MAKER[f"bce:{fam}"] = ("maker", "bce_half", fam)

rows = []
for arm in ARMS:
    for name, path in METRICS.items():
        v = [get(R[(arm, s)], path) for s in SEEDS if (arm, s) in R]
        if len(v) == 3:
            rows.append([arm, name, np.mean(v), np.std(v, ddof=1)] + v)
for name, path in MAKER.items():
    v = [get(R[("a", s)], path) for s in SEEDS if ("a", s) in R]
    if len(v) == 3:
        rows.append(["maker", name, np.mean(v), np.std(v, ddof=1)] + v)
with open(os.path.join(HERE, "SUMMARY.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "metric", "mean", "sd", "s1", "s2", "s3"])
    for r in rows:
        w.writerow([r[0], r[1]] + [f"{x:.6g}" for x in r[2:]])

ok = all((a, s) in R for a in ARMS for s in SEEDS)
V = {"complete": ok, "n_runs": len(R), "arms": {}}
if ok:
    def m(arm, key):
        return np.array([get(R[(arm, s)], METRICS[key]) for s in SEEDS])
    for X in TEST:
        rel = m(X, "P1_Dlearnable_bce") / m("a", "P1_Dlearnable_bce") - 1
        worse3 = bool((rel >= 0.01).all())
        prot = {}
        for key in ("P2_pre_prove_viol", "P3_twin_viol"):
            kill = np.minimum(m("wd", key), m("boost", key))
            prot[key] = bool((m(X, key) < kill).all())
        fail = worse3 and not any(prot.values())
        V["arms"][X] = {"P1_rel_vs_a": rel.round(5).tolist(), "P1_worse_1pct_3of3": worse3,
                        "protection_beyond_kills": prot, "fails_D_stress": fail,
                        "coherent_before_resolution": bool((m(X, "P2_pre_prove_viol") <= 0.25).all())}
    for X in ["a", "wd", "boost"]:
        V["arms"].setdefault(X, {})["coherent_before_resolution"] = bool((m(X, "P2_pre_prove_viol") <= 0.25).all())
    V["any_coherent_before_resolution"] = any(v["coherent_before_resolution"] for v in V["arms"].values())
    V["survivors"] = [X for X in TEST if not V["arms"][X]["fails_D_stress"]]
    V["failed"] = [X for X in TEST if V["arms"][X]["fails_D_stress"]]
json.dump(V, open(os.path.join(HERE, "VERDICT.json"), "w"), indent=1)
print(json.dumps(V, indent=1))

# Secondary (not a stop rule): paired differences of each tested arm against the best kill control
# per metric (min over wd, boost per seed), and the exploratory pre-resolution prove BCE.
if ok:
    P = {}
    for X in TEST + ["a"]:
        P[X] = {}
        for key in ("P1_Dlearnable_bce", "twin_viol_D64", "prove_bce_pre"):
            kill = np.minimum(m("wd", key), m("boost", key))
            dlt = m(X, key) - kill
            P[X][key] = {"diff_per_seed": dlt.round(5).tolist(), "mean": round(float(dlt.mean()), 5),
                         "sd": round(float(dlt.std(ddof=1)), 5), "better_3of3": bool((dlt < 0).all())}
    json.dump(P, open(os.path.join(HERE, "PAIRED-vs-kill.json"), "w"), indent=1)
    print(json.dumps({X: {k: (v["mean"], v["better_3of3"]) for k, v in d.items()} for X, d in P.items()}))
