"""Merge per-(seed, step) oracle files and write the numbers ORACLE.md quotes.

Writes (in results/):
  oracle_units.csv, oracle_coalitions.csv   merged wide tables
  oracle_long.csv                           seed,step,final,granularity,unit,flat,ablation,dloss (worker 3's format)
  summary_reliability.csv                   split-half Spearman and zero-vs-mean Spearman per seed/step/granularity
  summary_coalitions.csv                    additivity gap by coalition label group, per seed/step (zero and mean)
  summary_concentration.csv                 share of total single-neuron Δ in the top 1/5/10% neurons; n helpful/harmful
  summary_vs_wealthnet.csv                  Spearman of this oracle vs wealth-net decay_oracle (512-position, final step)
  summary.json                              mean ± sd over seeds of the headline numbers
"""
import glob, json, os, re, sys
WN = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../wealth-net"))
sys.path.insert(0, os.path.join(WN, ".pylib"))
sys.path.append("/tmp/traders-net/M2/w1/pylib")  # pandas (pip --target)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def group(label):
    if label.startswith("pair_top6"): return "pairs_top6"
    if label.startswith("top8_"): return "top8_family"
    if label.startswith("readouts_"): return "readouts_family"
    if re.match(r"top\d+_L\d", label): return "top_k_in_layer"
    return re.sub(r"\d+", "k", label) if label not in ("neurons", "heads", "mixed") and not label.startswith("neurons_L") else label.split("_L")[0] + ("_1layer" if "_L" in label else "")


def main():
    U = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "oracle_units_s*_step*.csv")))]
    C = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "oracle_coalitions_s*_step*.csv")))]
    C = [c for c in C if len(c)]  # 8k/30k are singles-only (header-only files)
    if not U:
        print("no files"); return
    u = pd.concat(U).sort_values(["seed", "step", "gran", "flat"]); u.to_csv(os.path.join(R, "oracle_units.csv"), index=False)
    c = pd.concat(C).sort_values(["seed", "step", "cid"]) if C else pd.DataFrame()
    if len(c): c.to_csv(os.path.join(R, "oracle_coalitions.csv"), index=False)
    lg = pd.concat([u[["seed", "step", "final", "gran", "unit", "flat"]].assign(ablation=mode, dloss=u[f"dloss_{mode}"])
                    for mode in ("zero", "mean")]).rename(columns={"gran": "granularity"})
    lg.to_csv(os.path.join(R, "oracle_long.csv"), index=False)

    rel = []
    for (s, st), g in u.groupby(["seed", "step"]):
        for gran in ("neuron", "head", "readout"):
            x = g[g.gran == gran]
            for sub, xx in ([("all", x)] + ([(f"L{l}", x[x.layer == l]) for l in (0, 1)] if gran == "neuron" else [])):
                rel.append({"seed": s, "step": st, "final": int(xx.final.iloc[0]), "gran": gran, "subset": sub, "n": len(xx),
                            "split_half_zero": spearmanr(xx.dloss_zero_A, xx.dloss_zero_B)[0],
                            "split_half_mean": spearmanr(xx.dloss_mean_A, xx.dloss_mean_B)[0],
                            "zero_vs_mean": spearmanr(xx.dloss_zero, xx.dloss_mean)[0],
                            "n_helpful_zero": int((xx.dloss_zero > 0).sum()), "n_harmful_zero": int((xx.dloss_zero < 0).sum()),
                            "max_dloss_zero": xx.dloss_zero.max(), "median_dloss_zero": xx.dloss_zero.median()})
    rel = pd.DataFrame(rel); rel.to_csv(os.path.join(R, "summary_reliability.csv"), index=False)

    con = []
    for (s, st), g in u[u.gran == "neuron"].groupby(["seed", "step"]):
        for mode in ("zero", "mean"):
            d = np.sort(g[f"dloss_{mode}"].values)[::-1]; pos = d[d > 0]
            # noise floor: sd of half difference /2 ~ se of the full-sample Δ
            se = np.std(g[f"dloss_{mode}_A"] - g[f"dloss_{mode}_B"]) / 2
            con.append({"seed": s, "step": st, "final": int(g.final.iloc[0]), "mode": mode, "sum_pos": pos.sum(),
                        "top1pct_share": d[:5].sum() / pos.sum(), "top5pct_share": d[:26].sum() / pos.sum(),
                        "top10pct_share": d[:51].sum() / pos.sum(), "se_single": se,
                        "n_above_2se": int((d > 2 * se).sum()), "n_below_-2se": int((d < -2 * se).sum())})
    con = pd.DataFrame(con); con.to_csv(os.path.join(R, "summary_concentration.csv"), index=False)

    if len(c):
        c["group"] = c.label.map(group)
        rows = []
        for (s, st, gp), g in c.groupby(["seed", "step", "group"]):
            for mode in ("zero", "mean"):
                d, ss = g[f"dloss_{mode}"].astype(float), g[f"sum_single_{mode}"].astype(float)
                rows.append({"seed": s, "step": st, "final": int(g.final.iloc[0]), "group": gp, "mode": mode, "n": len(g),
                             "mean_size": g["size"].mean(), "mean_dloss": d.mean(), "mean_sum_single": ss.mean(),
                             "mean_gap": (d - ss).mean(), "median_ratio": float(np.median(d / ss.where(ss.abs() > 1e-6))),
                             "frac_superadditive": float(((d - ss) > 0).mean()),
                             "spearman_dloss_vs_sum": spearmanr(d.values, ss.values)[0] if len(g) > 3 else np.nan,
                             "abs_gap_over_abs_dloss": float((d - ss).abs().sum() / max(d.abs().sum(), 1e-12))})
        pd.DataFrame(rows).to_csv(os.path.join(R, "summary_coalitions.csv"), index=False)

    vs = []
    for s in (1, 2, 3):
        f = os.path.join(WN, "results", f"decay_oracle_s{s}.npy")
        g = u[(u.seed == s) & (u.final == 1) & (u.gran == "neuron")].sort_values("flat")
        if os.path.exists(f) and len(g) == 512:
            o = np.load(f).ravel()
            vs.append({"seed": s, "n": 512, "spearman_vs_decay_oracle_512pos": spearmanr(o, g.dloss_zero.values)[0]})
    pd.DataFrame(vs).to_csv(os.path.join(R, "summary_vs_wealthnet.csv"), index=False)

    out = {}
    def ms(x):
        x = np.asarray(x, float); return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x, ddof=1)) if len(x) > 1 else None, "n": int(len(x))}
    for (st_final, gran, sub), g in rel.groupby(["final", "gran", "subset"]):
        key = f"{'final' if st_final else 'step' + str(sorted(g.step.unique()))}"
        for (step), gg in g.groupby("step" if not st_final else "final"):
            k2 = f"{'final' if st_final else 'step' + str(step)}/{gran}/{sub}"
            out[k2 + "/split_half_zero"] = ms(gg.split_half_zero); out[k2 + "/split_half_mean"] = ms(gg.split_half_mean)
            out[k2 + "/zero_vs_mean"] = ms(gg.zero_vs_mean)
    json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=1)
    print(rel.groupby(["final", "step", "gran", "subset"])[["split_half_zero", "split_half_mean", "zero_vs_mean"]].mean().round(3).to_string())
    print(con.groupby(["final", "mode"])[["top1pct_share", "top5pct_share", "n_above_2se", "n_below_-2se"]].agg(["mean", "std"]).round(3).to_string())
    if len(c):
        cs = pd.read_csv(os.path.join(R, "summary_coalitions.csv"))
        print(cs.groupby(["final", "group", "mode"])[["n", "mean_dloss", "mean_sum_single", "mean_gap", "frac_superadditive", "spearman_dloss_vs_sum"]].mean().round(3).to_string())
    if vs: print(pd.DataFrame(vs).to_string())


if __name__ == "__main__":
    main()
