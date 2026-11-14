"""Aggregate w3 results over seeds -> results/summary.csv, results/summary.json, results/checks.json"""
import json, os, glob, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = f"{HERE}/results"
ARMS = ["net-P4", "net-P16", "log-P4", "log-P16", "wn-8k", "wn-20k", "wn-final"]
TR = [f"{a}-{k}" for a in ("soup", "kt", "corr") for k in (2, 4, 8)]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
rows, S = [], {}
for arm in ARMS:
    rs = [json.load(open(f)) for f in sorted(glob.glob(f"{R}/{arm}-s[0-9].json"))]
    if not rs: continue
    F = rs[0]["families"]; S[arm] = {"n_seeds": len(rs), "mkt_bce_all": ms([r["mkt_bce_all"] for r in rs]),
                                     "mkt_bce_fam": {f: ms([r["mkt_bce"][i] for r in rs]) for i, f in enumerate(F)}}
    for cur in ("log", "share"):
        ec = [r[f"EC_{cur}"] for r in rs]
        S[arm][f"EC_{cur}"] = {m: ms([e[m] for e in ec]) for m in ("slope_h1", "slope_h2", "phi_last20", "E_end")}
        S[arm][f"EC_{cur}"]["argmax_end"] = [e["argmax_end"] for e in ec]
        S[arm][f"EC_{cur}"]["ratio_h2_h1"] = [e["slope_h2"] / e["slope_h1"] if e["slope_h1"] > 0 else None for e in ec]
        for t in TR:
            if cur not in rs[0]["traders"][t]: continue
            ph = [r["traders"][t][cur]["phi_last20"] for r in rs]
            pf = np.array([r["traders"][t][cur]["phi_fam"] for r in rs])
            row = {"arm": arm, "currency": cur, "trader": t, "phi_mean": ms(ph)[0], "phi_sd": ms(ph)[1],
                   "phi_pos": int((np.array(ph) > 0).sum()), "n": len(rs)}
            for i, f in enumerate(F):
                row[f"phi_{f}"] = float(pf[:, i].mean()); row[f"pos_{f}"] = int((pf[:, i] > 0).sum())
            rows.append(row)
with open(f"{R}/summary.csv", "w") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(S, open(f"{R}/summary.json", "w"), indent=1)

def get(arm, cur, t): return next(r for r in rows if r["arm"] == arm and r["currency"] == cur and r["trader"] == t)
C = {}
for arm in ("net-P4", "net-P16"):
    C[f"P1a {arm} kt-8 periodic phi>0 seeds (pred <=1)"] = get(arm, "log", "kt-8")["pos_periodic"]
    C[f"P1b {arm} kt-8 lfsr16 phi mean (pred <=0)"] = get(arm, "log", "kt-8")["phi_lfsr16"]
for arm in ("log-P4", "log-P16"):
    for f in ("and", "or", "ca110"):
        C[f"P2 {arm} best-of-kt {f} phi>0 seeds (pred 3)"] = max(get(arm, "log", f"kt-{k}")[f"pos_{f}"] for k in (2, 4, 8))
C["P3 corr>=kt phi (all arms,k) count"] = sum(get(a, "log", f"corr-{k}")["phi_mean"] >= get(a, "log", f"kt-{k}")["phi_mean"]
                                             for a in S for k in (2, 4, 8))
C["P3 total"] = len(S) * 3
for arm in ("net-P4", "log-P4"):
    C[f"P4 {arm} EC_share h2/h1 ratios (pred <0.2 for net, not for log)"] = S[arm]["EC_share"]["ratio_h2_h1"]
json.dump(C, open(f"{R}/checks.json", "w"), indent=1)
for k, v in C.items(): print(k, v)
for arm in S:
    for cur in ("log", "share"):
        e = S[arm][f"EC_{cur}"]
        print(arm, cur, "EC h1 %.4f±%.4f h2 %.4f±%.4f phi %.4f±%.4f" % (*e["slope_h1"], *e["slope_h2"], *e["phi_last20"]), e["argmax_end"])
