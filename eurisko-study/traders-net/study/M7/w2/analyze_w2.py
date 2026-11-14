"""Aggregate M7 w2 runs: summary.csv (mean, sd over seeds), paired.csv (arm - best DeepSeek, per seed), decision.json."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results/gpu/runs")
OUT = os.path.join(HERE, "results")
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUNS, "*.json")))]
MET = ["bce_all", "bce_last10k", "bce_post_shift", "bce_pre_shift", "bce_heldout_1k", "bce_heldout_all",
       "max_load", "dead_frac", "mi_expert_family_bits", "forwards_per_step", "wall_s"]
arms = sorted({r["arm"] for r in rows}); Es = sorted({r["E"] for r in rows})
get = lambda a, E, m: {r["seed"]: r[m] for r in rows if r["arm"] == a and r["E"] == E}
lines = ["arm,E,n," + ",".join(f"{m}_mean,{m}_sd" for m in MET)]
for E in Es:
    for a in arms:
        vals = [np.array(list(get(a, E, m).values()), float) for m in MET]
        n = len(vals[0])
        if n == 0:
            continue
        lines.append(f"{a},{E},{n}," + ",".join(f"{v.mean():.5f},{(v.std(ddof=1) if n > 1 else float('nan')):.5f}" for v in vals))
open(os.path.join(OUT, "summary.csv"), "w").write("\n".join(lines) + "\n")
dec = {"runs": len(rows), "per_E": {}}
pl = ["arm,E,metric,best_ds,seed,diff"]
for E in Es:
    ds = {a: np.mean(list(get(a, E, "bce_all").values())) for a in ("ds3", "ds2") if get(a, E, "bce_all")}
    if not ds:
        continue
    best = min(ds, key=ds.get)
    dsv = get(best, E, "bce_all"); dsd = np.std(list(dsv.values()), ddof=1)
    dec["per_E"][E] = {"best_ds": best, "best_ds_bce": ds[best], "arms": {}}
    for a in arms:
        v = get(a, E, "bce_all")
        seeds = sorted(set(v) & set(dsv))
        if not seeds:
            continue
        d = np.array([v[s] - dsv[s] for s in seeds])
        for s, x in zip(seeds, d):
            pl.append(f"{a},{E},bce_all,{best},{s},{x:.6f}")
        for m in ("bce_post_shift", "bce_last10k"):
            vm, dm = get(a, E, m), get(best, E, m)
            for s in seeds:
                pl.append(f"{a},{E},{m},{best},{s},{vm[s] - dm[s]:.6f}")
        asd = np.std(list(v.values()), ddof=1)
        ml = np.array(list(get(a, E, "max_load").values()))
        dec["per_E"][E]["arms"][a] = {
            "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)), "wins": int((d < 0).sum()), "n": len(d),
            "beats": bool((d < 0).all() and len(d) == 3 and -d.mean() > max(asd, dsd) and (ml <= 0.5).all()),
            "rel_diff_pct": float(100 * d.mean() / ds[best]),
            "stop_maxload": bool((ml > 0.5).sum() >= 2), "stop_2pct_worse": bool(100 * d.mean() / ds[best] > 2)}
open(os.path.join(OUT, "paired.csv"), "w").write("\n".join(pl) + "\n")
W = ["dsfw", "dsfw_fast", "dsfw_mean"]
dec["wealth_beats_deepseek_bothE"] = any(all(dec["per_E"].get(E, {}).get("arms", {}).get(a, {}).get("beats", False) for E in Es) for a in W) and len(Es) == 2
dec["wealth_beats_deepseek_someE"] = {str(E): [a for a in W if dec["per_E"].get(E, {}).get("arms", {}).get(a, {}).get("beats")] for E in Es}
json.dump(dec, open(os.path.join(OUT, "decision.json"), "w"), indent=1)
print(open(os.path.join(OUT, "summary.csv")).read()); print(json.dumps(dec, indent=1))
