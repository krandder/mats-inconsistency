"""Gate B on the fresh oracle: same rule as M2/lead/gate_b.py (per-seed diff of book minus reference on random
coalitions at final; win = diff > max(seed sd of both) in 3/3). Reads results/R2_fresh/coalition_pred.csv."""
import csv, json, os, statistics as st
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
cp = list(csv.DictReader(open(f"{H}/results/R2_fresh/coalition_pred.csv")))
by = defaultdict(lambda: defaultdict(list))
for r in cp:
    if r["ckpt"] == "final" and r["kind"].startswith("random"): by[r["predictor"]][r["seed"]].append(float(r["rho"]))
sv = {p: {s: sum(v) / len(v) for s, v in d.items()} for p, d in by.items()}
out = {}
for b in ["v2_lmsr_int", "v2_lmsr_inst", "settle_loo_train"]:
    for rf in ["taylor_abs", "movement", "taylor"]:
        d = [round(sv[b][s] - sv[rf][s], 4) for s in sorted(sv[b])]
        sd = round(max(st.stdev(sv[b].values()), st.stdev(sv[rf].values())), 4)
        out[f"final {b} vs {rf}"] = {"diffs": d, "sd": sd, "win": all(x > sd for x in d)}
t, m = out["final v2_lmsr_int vs taylor_abs"], out["final v2_lmsr_int vs movement"]
out.update(diff_taylor=t["diffs"], sd_taylor=t["sd"], win_taylor=t["win"], diff_move=m["diffs"], sd_move=m["sd"], win_move=m["win"], win_best=t["win"] and m["win"])
json.dump(out, open(f"{H}/results/R2_gateb_fresh.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if not k.startswith("final")}))
