"""X1 w2 R2 re-derivation of M1(b) sign census and M3 part A on a given M2-style oracle dir.
Usage: python3 rederive.py ORACLE_DIR OUT_JSON   (ORACLE_DIR holds oracle_units_s{S}_step{K}.csv)
Uses neuron dloss_zero at the final checkpoint; currencies from wealth-net (read-only)."""
import csv, json, sys, numpy as np
from scipy.stats import spearmanr
WN = "/home/agent/workspaces/eurisko-study/wealth-net/results"
FINAL = {1: 31100, 2: 31200, 3: 31100}
od, outp = sys.argv[1], sys.argv[2]
per = {}
for s, st in FINAL.items():
    o = np.full(512, np.nan)
    for r in csv.DictReader(open(f"{od}/oracle_units_s{s}_step{st}.csv")):
        if r["gran"] == "neuron": o[int(r["flat"])] = float(r["dloss_zero"])
    assert not np.isnan(o).any()
    dec = np.load(f"{WN}/decay_oracle_s{s}.npy").astype(float).ravel()
    cur = np.load(f"{WN}/decay_currencies_s{s}.npz"); stl = np.load(f"{WN}/settle_currencies_s{s}_step{st}.npz")
    g = np.array(json.load(open(f"{WN}/train-s{s}/snapshots/step-{st:05d}.json"))["gradnorm_cum"]["neuron"], float).ravel()
    top = np.argsort(-o)[:26]
    d = {"rho_oracle_vs_wealthnet_decay_oracle": spearmanr(o, dec)[0]}
    for k in ("cum", "hl1000", "hl10000", "win2000"):
        d[f"census_top5_fracneg_books_{k}"] = float((cur[k][top] < 0).mean())
    for k in ("loo_heldout", "loo_train"):
        d[f"census_top5_fracneg_settle_{k}"] = float((stl[k][top] < 0).mean())
    d["A_rho_cum"] = spearmanr(cur["cum"], o)[0]; d["A_rho_gradnorm"] = spearmanr(g, o)[0]
    d["A_rho_settle_loo_train"] = spearmanr(stl["loo_train"], o)[0]; d["A_rho_settle_loo_heldout"] = spearmanr(stl["loo_heldout"], o)[0]
    d["A_settle_minus_gradnorm"] = d["A_rho_settle_loo_train"] - d["A_rho_gradnorm"]
    per[s] = {k: float(v) for k, v in d.items()}
summ = {k: [float(np.mean([per[s][k] for s in per])), float(np.std([per[s][k] for s in per], ddof=1))] for k in per[1]}
json.dump(dict(oracle_dir=od, per_seed=per, mean_sd=summ), open(outp, "w"), indent=1)
for k, v in summ.items(): print(f"{k:45s} {v[0]:.4f} +- {v[1]:.4f}  ", " ".join(f"{per[s][k]:.3f}" for s in per))
