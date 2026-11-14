"""Post-hoc diagnostic (PREREG Addendum H): R7 label-head capacity on the T2m in-distribution pools, split by instance
type. Not a gate and not a criterion. For each pool (L64, L128), backbone seed and head seed, at every decision time
t < T_cap: predicted V_H = the stored R7 head output; realized G_H = S[min(t+H, T_cap)] - S[t] (cap: S[T_cap] - S[t]),
where S is the stored clamped log-score gain summed over positions. Reports Spearman and predicted/realized mean
ratio at H in {4, 16} for all instances, resolvable only and blocked only; mean and range over the 6 units.

    python capdiag.py --res results/vm/trader-cat-p1-7/t2m --out results/capdiag_t2m.json
"""
import argparse, json
import numpy as np

HORIZONS = [1, 2, 4, 8, 16, 32, 64, "cap"]


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for L in (64, 128):
        rows = []
        for b in (0, 1):
            O = np.load(f"{a.res}/obs_L{L}_b{b}.npz")
            E = np.load(f"{a.res}/eval_L{L}_b{b}.npz")
            S = E["S"].astype(np.float64)                     # (N, T+1)
            T = S.shape[1] - 1
            res = O["meta_resolvable"].astype(bool)
            for s in range(3):
                P = O[f"h{s}_R7"].astype(np.float64)          # (N, T, 8)
                row = {"bseed": b, "hseed": s}
                for j, H in enumerate(HORIZONS):
                    if H not in (4, 16):
                        continue
                    t = np.arange(T)
                    t2 = np.full(T, T) if H == "cap" else np.minimum(t + H, T)
                    G = S[:, t2] - S[:, t]                    # (N, T)
                    for name, m in (("all", np.ones_like(res)), ("resolvable", res), ("blocked", ~res)):
                        p, r = P[m, :, j].ravel(), G[m].ravel()
                        row[f"H{H}_{name}"] = {"spearman": spearman(p, r),
                                               "mean_ratio": float(p.mean() / (r.mean() + 1e-12)),
                                               "pred_mean": float(p.mean()), "real_mean": float(r.mean())}
                rows.append(row)
        summ = {}
        for k in [k for k in rows[0] if k.startswith("H")]:
            for stat in ("spearman", "mean_ratio"):
                v = np.array([r[k][stat] for r in rows])
                summ[f"{k}:{stat}"] = {"mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}
        rep[f"L{L}"] = {"units": rows, "summary": summ}
    json.dump(rep, open(a.out, "w"), indent=1)
    for L, d in rep.items():
        for k, v in d["summary"].items():
            print(f"{L} {k:32s} mean {v['mean']:7.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")


if __name__ == "__main__":
    main()
