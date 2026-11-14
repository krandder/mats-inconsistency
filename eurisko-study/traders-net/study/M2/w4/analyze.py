"""Metrics for M2/w4 (PREREG.md): reads results/units_*.csv, pairs_*.csv; writes results/summary_*.csv."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"
PRED = ["oracle_zero", "oracle_mean", "taylor", "abs_ag", "gradnorm", "magnitude", "conductance",
        "conductance256", "shapley", "loo_direct", "random"]
SIGNED = ["oracle_zero", "oracle_mean", "taylor", "conductance", "conductance256", "shapley", "loo_direct"]
EPS = 1e-7


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); ties within EPS."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    gt = (pos > neg + EPS).mean(); eq = (np.abs(pos - neg) <= EPS).mean()
    return gt + 0.5 * eq


def sp(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return spearmanr(a, b).correlation


def per_group(d):
    out = {}
    live = d[d.cls.isin(["C", "R", "N"])]
    used, unused = d[d.cls.isin(["C", "R"])], d[d.cls.isin(["Z", "D"])]
    Rr, N = d[d.cls == "R"], d[d.cls == "N"]
    for p in PRED:
        out[("sp_all", p)] = sp(d[p], d.oracle_zero)
        out[("sp_live", p)] = sp(live[p], live.oracle_zero)
        out[("sp_mean_live", p)] = sp(live[p], live.oracle_mean) if "oracle_mean" in live else np.nan  # stage 2 has no mean oracle (M2 lead fix)
        out[("auroc_used", p)] = auroc(used[p].abs(), unused[p].abs())
        thr = np.quantile(unused[p].abs(), 0.9) if len(unused) else np.nan
        out[("R_recall", p)] = float((Rr[p].abs() > thr + EPS).mean()) if len(Rr) else np.nan
        if p in SIGNED:
            out[("N_neg", p)] = float((N[p] < -EPS).mean()) if len(N) else np.nan
            Ns = N[N.oracle_zero.abs() > 1e-4]
            out[("N_signagree", p)] = float((np.sign(Ns[p]) == np.sign(Ns.oracle_zero)).mean()) if len(Ns) else np.nan
            lv = live[live.oracle_zero > 0]
            imp = lv[lv.oracle_zero >= lv.oracle_zero.quantile(0.75)] if len(lv) >= 4 else lv.iloc[:0]
            out[("neg_on_important", p)] = float((imp[p] < -EPS).mean()) if len(imp) else np.nan
    out[("n", "units")] = len(d); out[("n", "used")] = len(used); out[("n", "unused")] = len(unused)
    return out


def main():
    U = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/units_*_s*.csv"))])
    Pp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/pairs_*_s*.csv"))])
    rows = []
    for (prog, seed, beta, gran), d in U.groupby(["prog", "seed", "beta", "gran"]):
        for (m, p), v in per_group(d).items():
            rows.append(dict(prog=prog, seed=seed, beta=beta, gran=gran, metric=m, predictor=p, value=v))
    # pooled over programs (within seed): concatenate units of all programs
    for (seed, beta, gran), d in U.groupby(["seed", "beta", "gran"]):
        for (m, p), v in per_group(d).items():
            rows.append(dict(prog="POOLED", seed=seed, beta=beta, gran=gran, metric=m, predictor=p, value=v))
    for (seed, beta, gran), d in U[U.prog != "dyck"].groupby(["seed", "beta", "gran"]):
        for (m, p), v in per_group(d).items():
            rows.append(dict(prog="POOLED_nodyck", seed=seed, beta=beta, gran=gran, metric=m, predictor=p, value=v))
    long = pd.DataFrame(rows)
    long.to_csv(f"{R}/metrics_long.csv", index=False)
    agg = long.groupby(["prog", "beta", "gran", "metric", "predictor"]).value.agg(["mean", "std", "count"]).reset_index()
    agg.to_csv(f"{R}/summary_metrics.csv", index=False)
    # redundancy gap
    Pp["gap"] = (Pp.dpair - Pp.d1 - Pp.d2) / Pp.dpair.abs().clip(lower=1e-9)
    Pp["single_share"] = (Pp.d1 + Pp.d2) / Pp.dpair.abs().clip(lower=1e-9)
    Pp["hidden"] = ((Pp[["d1", "d2"]].max(axis=1) < 0.1 * Pp.dpair) & (Pp.dpair > 0.01)).astype(float)
    Pp["superadd"] = ((Pp[["d1", "d2"]].min(axis=1) > Pp.dpair) & (Pp.dpair > 0.01)).astype(float)
    h = Pp.groupby(["prog", "beta", "gran"]).agg(n_pairs=("hidden", "size"), hidden_frac=("hidden", "mean"),
                                                 single_exceeds_pair_frac=("superadd", "mean")).reset_index()
    h.to_csv(f"{R}/summary_pairs_hidden.csv", index=False)
    g = Pp[Pp.dpair.abs() > 1e-4].groupby(["prog", "beta", "gran", "seed"]).agg(gap=("gap", "median"), n=("gap", "size")).reset_index()
    g.to_csv(f"{R}/pairs_gap_by_seed.csv", index=False)
    g2 = g.groupby(["prog", "beta", "gran"]).gap.agg(["mean", "std", "count"]).reset_index()
    g2.to_csv(f"{R}/summary_pairs_gap.csv", index=False)
    # compact tables for the deliverable
    pd.set_option("display.width", 250)
    for beta in sorted(U.beta.unique()):
        for metric in ["auroc_used", "sp_live", "sp_all", "R_recall", "N_neg", "N_signagree", "neg_on_important", "sp_mean_live"]:
            a = agg[(agg.beta == beta) & (agg.metric == metric)]
            if a.empty:
                continue
            a = a.assign(cell=a["mean"].map("{:+.2f}".format) + "±" + a["std"].map("{:.2f}".format))
            t = a.pivot_table(index=["prog", "gran"], columns="predictor", values="cell", aggfunc="first")
            t = t[[p for p in PRED if p in t.columns]]
            print(f"\n### beta={beta:g} {metric} (mean ± sd over seeds)\n")
            print(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())
    print("\n### redundancy gap (median over pairs of (pair - d1 - d2)/|pair|), mean ± sd over seeds\n")
    print(g2.round(3).to_string(index=False))
    print("\n### R pairs: hidden from single ablation (max single < 10% of pair) and single > pair, pooled over seeds\n")
    print(h.round(3).to_string(index=False))


S2PRED = ["oracle_zero", "taylor", "loo_direct", "magnitude", "books_signed", "books_abs", "movement", "random"]
S2SIGNED = ["oracle_zero", "taylor", "loo_direct", "books_signed", "movement"]


def stage2():
    fs = sorted(glob.glob(f"{R}/s2units_*_s*.csv"))
    if not fs:
        return
    import json
    U = pd.concat([pd.read_csv(f) for f in fs])
    global PRED, SIGNED
    PRED, SIGNED = S2PRED, S2SIGNED
    rows = []
    for (prog, seed, gran), d in U.groupby(["prog", "seed", "gran"]):
        for (m, p), v in per_group(d).items():
            rows.append(dict(prog=prog, seed=seed, gran=gran, metric=m, predictor=p, value=v))
    for (seed, gran), d in U.groupby(["seed", "gran"]):
        for (m, p), v in per_group(d).items():
            rows.append(dict(prog="POOLED", seed=seed, gran=gran, metric=m, predictor=p, value=v))
    long = pd.DataFrame(rows); long.to_csv(f"{R}/s2_metrics_long.csv", index=False)
    agg = long.groupby(["prog", "gran", "metric", "predictor"]).value.agg(["mean", "std", "count"]).reset_index()
    agg.to_csv(f"{R}/s2_summary_metrics.csv", index=False)
    meta = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(f"{R}/s2meta_*.json"))])
    meta.to_csv(f"{R}/s2_meta.csv", index=False)
    print("\n## STAGE 2 (fine-tuned from perturbed net, beta=5)\n")
    print(meta[["prog", "seed", "noise_frac", "loss_start", "loss_end", "acc_end", "wall_s"]].to_string(index=False))
    for metric in ["auroc_used", "sp_live", "R_recall", "neg_on_important", "N_signagree"]:
        a = agg[agg.metric == metric]
        if a.empty:
            continue
        a = a.assign(cell=a["mean"].map("{:+.2f}".format) + "±" + a["std"].map("{:.2f}".format) + " n" + a["count"].astype(str))
        t = a.pivot_table(index=["prog", "gran"], columns="predictor", values="cell", aggfunc="first")
        t = t[[p for p in S2PRED if p in t.columns]]
        print(f"\n### s2 {metric}\n")
        print(t.to_markdown() if hasattr(t, "to_markdown") else t.to_string())


if __name__ == "__main__":
    main()
    stage2()
