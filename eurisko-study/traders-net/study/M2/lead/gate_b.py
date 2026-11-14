"""Gate B per checkpoint: per-seed diffs of book minus reference on random coalitions (from w3/results/coalition_pred.csv),
same rule as w3/metrics.py (win = diff > max seed sd in 3/3 seeds). Also the stage-2 tracr books/movement rank identity.
Writes lead/gate_b.json."""
import csv, json, os, statistics as st, glob
from collections import defaultdict
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cp = list(csv.DictReader(open(f"{H}/w3/results/coalition_pred.csv")))
out = {}
for ck in ["20000", "final"]:
    by = defaultdict(lambda: defaultdict(list))
    for r in cp:
        if r["ckpt"] == ck and r["kind"].startswith("random"): by[r["predictor"]][r["seed"]].append(float(r["rho"]))
    sv = {p: {s: sum(v) / len(v) for s, v in d.items()} for p, d in by.items()}
    for b in ["v2_lmsr_int", "v2_lmsr_inst", "settle_loo_train"]:
        for rf in ["taylor_abs", "taylor_sq", "movement", "taylor"]:
            if b in sv and rf in sv:
                d = [round(sv[b][s] - sv[rf][s], 4) for s in sorted(sv[b])]
                sd = round(max(st.stdev(sv[b].values()), st.stdev(sv[rf].values())), 4)
                out[f"{ck} {b} vs {rf}"] = {"diffs": d, "sd": sd, "win": all(x > sd for x in d)}
try:
    import pandas as pd
    U = pd.concat([pd.read_csv(f) for f in glob.glob(f"{H}/w4/results/s2units_*.csv")])
    out["tracr_s2_spearman_books_signed_vs_movement"] = {g: round(float(d[["books_signed", "movement"]].corr(method="spearman").iloc[0, 1]), 4)
                                                         for g, d in U.groupby("gran")}
except ImportError:
    pass
json.dump(out, open(f"{H}/lead/gate_b.json", "w"), indent=1); print(json.dumps(out)[:600])
