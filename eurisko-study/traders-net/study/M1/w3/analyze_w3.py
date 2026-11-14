"""Summarise run_w3 outputs -> results/summary-w3.csv, results/summary-w3.json (PREREG-w3 metrics)."""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
BOOKS = ["tangent", "lmsr", "meanbase", "meanbase_lmsr"]; GR = ["out", "head", "neuron"]
seeds = sorted(int(os.path.basename(f)[5:-5]) for f in glob.glob(os.path.join(R, "run-s*.json")))
ld = lambda tag, s: np.load(os.path.join(R, f"{tag}-s{s}.npz"))
rows = []  # metric, arm, book, gran, seed, value


def tot(d, b, g):  # family(group)-summed cumulative wealth, flattened
    x = d[f"{b}__{g}"]
    return x.sum(-1).ravel() if b != "absag" else x.ravel()


def rho(x, y):
    return float(spearmanr(x, y)[0])


for s in seeds:
    D = {t: ld(t, s) for t in ["P_pre", "P_post", "P_frozen", "P_random", "ME_in", "ME_ho"] if os.path.exists(os.path.join(R, f"{t}-s{s}.npz"))}
    for b in BOOKS:
        for g in GR:
            if "ME_in" in D:
                rows.append(("spearman_in_vs_ho", "ME", b, g, s, rho(tot(D["ME_in"], b, g), tot(D["ME_ho"], b, g))))
            rows.append(("spearman_pre_vs_post", "P", b, g, s, rho(tot(D["P_pre"], b, g), tot(D["P_post"], b, g))))
            rows.append(("spearman_pre_vs_frozen", "P", b, g, s, rho(tot(D["P_pre"], b, g), tot(D["P_frozen"], b, g))))
            rows.append(("spearman_pre_vs_random", "P", b, g, s, rho(tot(D["P_pre"], b, g), tot(D["P_random"], b, g))))
            rows.append(("spearman_vs_absag", "P_pre", b, g, s, rho(tot(D["P_pre"], b, g), tot(D["P_pre"], "absag", g))))
            for t in D:
                x = D[t][f"{b}__{g}"]; pr = x[..., 0].ravel(); steps = float(D[t]["steps"])
                rows.append(("prng_income_total_per_step", t, b, g, s, float(pr.sum()) / steps))
                rows.append(("rest_income_total_per_step", t, b, g, s, float(x[..., 1].sum()) / steps))
                if g != "out":
                    rows.append(("prng_frac_units_positive", t, b, g, s, float((pr > 0).mean())))
        for g in GR:
            for b2 in BOOKS:
                if b2 > b:
                    rows.append((f"spearman_{b}_vs_{b2}", "P_pre", "-", g, s, rho(tot(D["P_pre"], b, g), tot(D["P_pre"], b2, g))))
    for t in D:
        for k in ["gain", "R_tangent", "maker", "meanbase_const"]:
            v = D[t][f"acct__{k}"]
            rows.append((f"acct_{k}_prng_per_step", t, "-", "-", s, float(v[0]) / float(D[t]["steps"])))
            rows.append((f"acct_{k}_rest_per_step", t, "-", "-", s, float(v[1]) / float(D[t]["steps"])))
    run = json.load(open(os.path.join(R, f"run-s{s}.json")))
    for arm, dd in run["diag_absmean"].items():
        for k, v in dd.items():
            rows.append((f"diag_absmean_{k}", arm, "-", "-", s, v))
    if "ME_epochs" in run:
        last = run["ME_epochs"][-1]
        for k in ["in_prng", "ho_prng", "in_rest", "ho_rest"]:
            rows.append((f"ME_final_bce_{k}", "ME", "-", "-", s, last[k]))
        rows.append(("ME_epochs_done", "ME", "-", "-", s, last["epoch"]))
    for i, v in enumerate(run["P_loss_by_quarter"]):
        rows.append((f"P_trade_bce_q{i+1}", "P", "-", "-", s, v))

# aggregate
from collections import defaultdict
agg = defaultdict(list)
for m, a, b, g, s, v in rows:
    agg[(m, a, b, g)].append(v)
with open(os.path.join(R, "summary-w3.csv"), "w") as f:
    f.write("metric,arm,book,gran,mean,sd,n,values\n")
    for k, v in sorted(agg.items()):
        v = np.array(v)
        f.write(",".join(map(str, k)) + f",{v.mean():.6g},{(v.std(ddof=1) if len(v) > 1 else float('nan')):.4g},{len(v)},"
                + "/".join(f"{x:.4g}" for x in v) + "\n")
with open(os.path.join(R, "rows-w3.csv"), "w") as f:
    f.write("metric,arm,book,gran,seed,value\n")
    for r in rows:
        f.write(",".join(map(str, r)) + "\n")
print(open(os.path.join(R, "summary-w3.csv")).read())
