"""Per-trader sign agreement of G, S, K with the LOO oracle Delta (all live traders) -> results/signs.csv."""
import numpy as np, csv, os
H = os.path.dirname(os.path.abspath(__file__))
rows = []
for arm in ["U", "Cs", "Cm", "Cw"]:
    for s in [1, 2, 3]:
        d = np.load(os.path.join(H, "results", f"{arm}-s{s}.npz")); l = d["live"]
        D = d["Delta"][:, l]; nz = np.abs(D) > 1e-6
        r = dict(arm=arm, seed=s, n_nonzero=int(nz.sum()))
        for m in ["G", "S", "K"]:
            X = d[m][:, l]
            r[f"agree_{m}"] = float((np.sign(X[nz]) == np.sign(D[nz])).mean()) if nz.any() else float("nan")
        rows.append(r)
with open(os.path.join(H, "results", "signs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for arm in ["U", "Cs", "Cm", "Cw"]:
    v = [r for r in rows if r["arm"] == arm]
    print(arm, v[0]["n_nonzero"], {m: f'{np.mean([x["agree_"+m] for x in v]):.3f} ± {np.std([x["agree_"+m] for x in v], ddof=1):.3f}' for m in "GSK"})
