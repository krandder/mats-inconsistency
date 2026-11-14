"""M1(b) analysis: per run and granularity at the final step: identity error, R^2/Spearman of
books wealth W on norm growth N (and N+rent-second), W on T, T on N. Writes per-run CSV and a
mean+-sd summary CSV. Usage: analyze_w2.py RUN_DIR OUT_DIR"""
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr


def r2(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def rho(x, y):
    return float(spearmanr(x, y).correlation)


def main(rd, od):
    os.makedirs(od, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(rd, "*.units.csv"))):
        tag = os.path.basename(f).replace(".units.csv", "")
        arm, seed = tag.rsplit("-s", 1)
        meta = json.load(open(os.path.join(rd, tag + ".json")))
        d = pd.read_csv(f)
        last = d.step.max()
        ev = meta["eval"][-1]
        for step in sorted(set([last] + [s for s in d.step.unique() if s in (1000, 5000)])):
            for gname, g in d[d.step == step].groupby("gran"):
                N, T, W, rent, sec = g.N.values, g["T"].values, g.W.values, g.rent.values, g.second.values
                pred = N + rent - sec                      # exact T under (W)SGD
                ident = float(np.max(np.abs(T - pred)) / max(1e-12, np.max(np.abs(T))))
                rows.append(dict(arm=arm, seed=int(seed), step=int(step), gran=gname, n_units=len(g),
                                 steps_done=int(last), eval_bce=ev["mean"],
                                 ident_err_rel=ident,
                                 r2_W_N=r2(N, W), rho_W_N=rho(N, W),
                                 r2_W_Nrent=r2(N + rent, W), rho_W_Nrent=rho(N + rent, W),
                                 r2_W_pred=r2(pred, W),
                                 r2_W_T=r2(T, W), rho_W_T=rho(T, W),
                                 r2_T_N=r2(N, T), rho_T_N=rho(N, T),
                                 leak_rel=float(np.abs(W - T).sum() / max(1e-12, np.abs(T).sum())),
                                 second_share=float(np.abs(sec).sum() / max(1e-12, np.abs(T).sum())),
                                 rent_share=float(np.abs(rent).sum() / max(1e-12, np.abs(T).sum())),
                                 frac_W_neg=float((W < 0).mean())))
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(od, "per_run.csv"), index=False)
    fin = per[per.step == per.steps_done]
    num = [c for c in fin.columns if c not in ("arm", "seed", "step", "gran", "n_units")]
    agg = fin.groupby(["arm", "gran"])[num].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_seeds"] = fin.groupby(["arm", "gran"]).size()
    agg.reset_index().to_csv(os.path.join(od, "summary.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(agg[[c for c in agg.columns if c.endswith("_mean")] + ["n_seeds"]].round(4).T)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
