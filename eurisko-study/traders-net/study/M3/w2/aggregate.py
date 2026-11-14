"""Aggregate results/seed{1,2,3}.json -> results/summary.csv, results/decision.json, results/postshift.png"""
import json, os, csv
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "results")
J = [json.load(open(os.path.join(R, f"seed{s}.json"))) for s in (1, 2, 3)]
arms = J[0]["meta"]["arms"]
keys = ["post2k_learnable", "post2k_all_switches", "lag_median", "lag_censored", "n_learnable", "stationary", "overall"]
rows = []
for a in arms:
    r = {"arm": a}
    for k in keys:
        v = np.array([j["arms"][a][k] for j in J], float)
        r[k + "_mean"] = round(float(v.mean()), 4); r[k + "_sd"] = round(float(v.std(ddof=1)), 4)
        r[k + "_seeds"] = "/".join(f"{x:.4g}" for x in v)
    rows.append(r)
with open(os.path.join(R, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# decision rule (PREREG): beat kelly on post2k_learnable by >= 0.05 in 3/3 seeds; stationary within 0.01 of kelly
k = np.array([j["arms"]["kelly"]["post2k_learnable"] for j in J]); ks = np.array([j["arms"]["kelly"]["stationary"] for j in J])
dec = {}
for a in arms:
    if a in ("kelly", "uniform", "shuffled"): continue
    p = np.array([j["arms"][a]["post2k_learnable"] for j in J]); st = np.array([j["arms"][a]["stationary"] for j in J])
    dec[a] = dict(gain_vs_kelly=[round(x, 4) for x in (k - p)], beats_3of3=bool(((k - p) >= 0.05).all()),
                  stationary_cost=[round(x, 4) for x in (st - ks)], stationary_ok=bool(((st - ks) <= 0.01).all()))
ok = [a for a, d in dec.items() if d["beats_3of3"] and d["stationary_ok"]]
best = min(ok, key=lambda a: np.mean([j["arms"][a]["post2k_learnable"] for j in J])) if ok else None
json.dump(dict(rule="PREREG.md stop/decision rule", per_arm=dec, qualifying=ok, best=best), open(os.path.join(R, "decision.json"), "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = np.mean([np.load(os.path.join(R, f"curve_s{s}.npy")) for s in (1, 2, 3)], 0)
    x = np.arange(-500, 4000)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for a in ["kelly", "ema1k", "fs1e-3", "win500", "grad1", "uniform"]:
        i = arms.index(a); c = np.convolve(C[i], np.ones(50) / 50, mode="same")
        ax.plot(x, c, label=a, lw=1.5)
    ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("steps after regime shift"); ax.set_ylabel("log loss (nats), 50-step mean")
    ax.set_title("Switched questions (new family learnable), mean of 3 seeds"); ax.legend(frameon=False); ax.set_ylim(0, 1.2)
    fig.tight_layout(); fig.savefig(os.path.join(R, "postshift.png"), dpi=110)
except Exception as e:
    print("plot skipped:", e)
print(json.dumps(dict(qualifying=ok, best=best)))
for r in rows: print(r["arm"], r["post2k_learnable_mean"], r["post2k_learnable_sd"], r["lag_median_mean"], r["lag_censored_seeds"], r["stationary_mean"], r["overall_mean"])
