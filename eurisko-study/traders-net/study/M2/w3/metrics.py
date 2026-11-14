"""M2 item 3: rank and additivity metrics (no model evaluation; reads worker 1 and worker 2 tables).

Inputs
  ../w1/results/oracle_units.csv       seed,step,gran,flat,dloss_zero,dloss_mean,...
  ../w1/results/oracle_coalitions.csv  seed,step,cid,kind,label,size,members,dloss_zero,dloss_mean,...
  predictor CSVs (long)                seed,step,gran,unit_flat,predictor,score (higher = more important)
Outputs (results/)
  spearman_long.csv      seed,step,gran,predictor,ablation,rho,rho_abs,n,top10_recall_top5
  spearman_summary.csv   step,gran,predictor,ablation,mean,sd,n_seeds (+ abs variant)
  additivity.csv         per seed,step,kind: median |gap|/|dloss(S)|, Spearman(sum singles, dloss(S))
  coalition_pred.csv     per seed,step,kind,predictor: Spearman(sum_{i in S} score_i, dloss(S))
  coalition_summary.csv  mean/sd over seeds
  stoprule.json          tie/win/loss verdict per granularity (final checkpoint, zero ablation)
Usage: python3 metrics.py PRED.csv [PRED2.csv ...]
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
W1 = os.environ.get("M2_W1", os.path.join(HERE, "../w1/results"))
OUT = os.environ.get("M2_OUT", os.path.join(HERE, "results"))
GRAN_N = {"neuron": 512, "head": 8, "readout": 64}
FINAL = {1: 31100, 2: 31200, 3: 31100}
BOOKS_PREFIX = ("wealth_v1", "books_v1", "v1", "books_v2", "v2", "tangent", "lmsr", "meanbase", "cp")
TIE_REFS = ("taylor", "movement")


LEAKY = ("_val", "_oracleseg")


def is_book(p):
    return p.lower().startswith(BOOKS_PREFIX) and not p.endswith(LEAKY)


def is_settle(p):
    return p.startswith("settle_") and not p.endswith(LEAKY)


def is_ref(p):
    return any(t in p.lower() for t in TIE_REFS) and not p.endswith(LEAKY)


def ck(seed, step):
    """checkpoint label shared across seeds (final differs by seed)."""
    return "final" if step == FINAL.get(seed) else str(step)


def unit_gran(u):
    return {"N": "neuron", "H": "head", "R": "readout"}[u[0]]


def unit_flat(u):
    if u[0] == "R":
        return int(u[1:])
    l, i = u[1:].split(".")
    return int(l) * (256 if u[0] == "N" else 4) + int(i)


def load_oracle():
    o = {}
    import glob
    fs = sorted(glob.glob(os.path.join(W1, "oracle_units_s*_step*.csv"))) or [os.path.join(W1, "oracle_units.csv")]
    for r in (x for f in fs for x in csv.DictReader(open(f))):
        s, st, g = int(r["seed"]), int(r["step"]), r["gran"]
        for ab in ("zero", "mean"):
            o.setdefault((s, st, g, ab), np.full(GRAN_N[g], np.nan))[int(r["flat"])] = float(r[f"dloss_{ab}"])
    return o


def load_preds(files):
    p = {}
    for f in files:
        for r in csv.DictReader(open(f)):
            s, st, g = int(r["seed"]), int(r["step"]), r["gran"]
            try:
                v = float(r.get("score", r.get("value")))
            except ValueError:
                continue
            p.setdefault((s, st, g, r["predictor"]), np.full(GRAN_N[g], np.nan))[int(r.get("unit_flat", r.get("flat")))] = v
    return p


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(spearmanr(a[ok], b[ok])[0]), int(ok.sum())


def msd(xs):
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return np.nan, np.nan, 0
    return float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else np.nan, len(xs)


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)


def main(files):
    os.makedirs(OUT, exist_ok=True)
    orc = load_oracle(); prd = load_preds(files)
    # built-in wealth-net predictors (v1 wealth and cumulative gradnorm) if not supplied
    long = []
    for (s, st, g, p), sc in prd.items():
        for ab in ("zero", "mean"):
            o = orc.get((s, st, g, ab))
            if o is None:
                continue
            r, n = rho(sc, o); ra, _ = rho(np.abs(sc), o)
            rec = np.nan
            if g == "neuron":
                ok = np.isfinite(sc) & np.isfinite(o)
                top5 = set(np.argsort(-np.where(ok, o, -np.inf))[:max(1, int(0.05 * ok.sum()))])
                top10 = set(np.argsort(-np.where(ok, sc, -np.inf))[:max(1, int(0.10 * ok.sum()))])
                rec = len(top5 & top10) / len(top5)
            long.append({"seed": s, "step": st, "ckpt": ck(s, st), "gran": g, "predictor": p, "ablation": ab,
                         "rho": round(r, 4), "rho_abs": round(ra, 4), "n": n, "top10_recall_top5": round(rec, 4)})
    long.sort(key=lambda r: (r["gran"], r["predictor"], r["ablation"], r["ckpt"], r["seed"]))
    write("spearman_long.csv", long)
    grp = defaultdict(list)
    for r in long:
        grp[(r["ckpt"], r["gran"], r["predictor"], r["ablation"])].append(r)
    summ = []
    for (c, g, p, ab), rs in sorted(grp.items()):
        m, sd, n = msd([r["rho"] for r in rs]); ma, sda, _ = msd([r["rho_abs"] for r in rs])
        mr, sdr, _ = msd([r["top10_recall_top5"] for r in rs])
        summ.append({"ckpt": c, "gran": g, "predictor": p, "ablation": ab, "rho_mean": round(m, 4), "rho_sd": round(sd, 4),
                     "rho_abs_mean": round(ma, 4), "rho_abs_sd": round(sda, 4), "recall_mean": round(mr, 4),
                     "recall_sd": round(sdr, 4), "n_seeds": n})
    write("spearman_summary.csv", summ)

    # stop rule: final checkpoint, zero ablation; "books" = gradient books (v1/v2), "settle" = exact
    # output settlement (wealth-net settle.py), each against the best of Taylor/movement variants
    verdict = {}
    S = {(r["gran"], r["predictor"]): r for r in summ if r["ckpt"] == "final" and r["ablation"] == "zero"}
    for cls, test in (("books", is_book), ("settle", is_settle)):
        for g in GRAN_N:
            books = {p: S[(gg, p)] for (gg, p) in S if gg == g and test(p)}
            refs = {p: S[(gg, p)] for (gg, p) in S if gg == g and is_ref(p)}
            key = g if cls == "books" else f"{g}_settle"
            if not books or not refs:
                verdict[key] = {"status": "missing", "books": sorted(books), "refs": sorted(refs)}; continue
            bb = max(books.values(), key=lambda r: r["rho_mean"]); br = max(refs.values(), key=lambda r: r["rho_mean"])
            margin = np.nanmax([bb["rho_sd"], br["rho_sd"]])
            d = bb["rho_mean"] - br["rho_mean"]
            st = "tie" if abs(d) <= margin else ("win" if d > 0 else "loss")
            verdict[key] = {"status": st, "best_book": bb["predictor"], "book_rho": bb["rho_mean"], "book_sd": bb["rho_sd"],
                            "best_ref": br["predictor"], "ref_rho": br["rho_mean"], "ref_sd": br["rho_sd"],
                            "diff": round(d, 4), "margin": round(float(margin), 4)}

    # additivity
    import glob
    cfs = sorted(glob.glob(os.path.join(W1, "oracle_coalitions_s*_step*.csv"))) or \
        [f for f in [os.path.join(W1, "oracle_coalitions.csv")] if os.path.exists(f)]
    add, cp = [], []
    if cfs:
        co = defaultdict(list)
        for r in (x for f in cfs for x in csv.DictReader(open(f))):
            co[(int(r["seed"]), int(r["step"]))].append(r)
        for (s, st), rs in co.items():
            for kind in sorted(set(r["kind"] for r in rs)) + ["all"]:
                sub = [r for r in rs if kind == "all" or r["kind"] == kind]
                for ab in ("zero", "mean"):
                    D = np.array([float(r[f"dloss_{ab}"]) for r in sub])
                    o = {g: orc.get((s, st, g, ab)) for g in GRAN_N}
                    sums = []
                    for r in sub:
                        tot = 0.0
                        for u in r["members"].split(";"):
                            g = unit_gran(u); tot += o[g][unit_flat(u)] if o[g] is not None else np.nan
                        sums.append(tot)
                    sums = np.array(sums); gap = D - sums
                    rr, n = rho(sums, D)
                    add.append({"seed": s, "step": st, "ckpt": ck(s, st), "kind": kind, "ablation": ab, "n": n,
                                "median_rel_gap": round(float(np.nanmedian(np.abs(gap) / np.maximum(np.abs(D), 1e-6))), 4),
                                "mean_gap": round(float(np.nanmean(gap)), 5), "frac_superadditive": round(float(np.nanmean(gap > 0)), 4),
                                "rho_sum_singles": round(rr, 4)})
                    if ab != "zero":
                        continue
                    preds = sorted(set(p for (ss, sst, g, p) in prd if ss == s and sst == st))
                    for p in preds:
                        vals = []
                        for r in sub:
                            tot = 0.0
                            for u in r["members"].split(";"):
                                sc = prd.get((s, st, unit_gran(u), p))
                                tot += sc[unit_flat(u)] if sc is not None else np.nan
                            vals.append(tot)
                        rr, n = rho(np.array(vals), D)
                        if n >= 4:
                            cp.append({"seed": s, "step": st, "ckpt": ck(s, st), "kind": kind, "predictor": p, "n": n, "rho": round(rr, 4)})
        write("additivity.csv", add); write("coalition_pred.csv", cp)
        g2 = defaultdict(list)
        for r in cp:
            g2[(r["ckpt"], r["kind"], r["predictor"])].append(r["rho"])
        cs = [{"ckpt": c, "kind": k, "predictor": p, "rho_mean": round(msd(v)[0], 4), "rho_sd": round(msd(v)[1], 4),
               "n_seeds": msd(v)[2], "seeds": ";".join(str(x) for x in v)} for (c, k, p), v in sorted(g2.items())]
        g3 = defaultdict(list)
        for r in add:
            g3[(r["ckpt"], r["kind"], r["ablation"])].append(r)
        asum = [{"ckpt": c, "kind": k, "ablation": ab,
                 "median_rel_gap_mean": round(msd([x["median_rel_gap"] for x in v])[0], 4),
                 "median_rel_gap_sd": round(msd([x["median_rel_gap"] for x in v])[1], 4),
                 "rho_sum_singles_mean": round(msd([x["rho_sum_singles"] for x in v])[0], 4),
                 "rho_sum_singles_sd": round(msd([x["rho_sum_singles"] for x in v])[1], 4),
                 "frac_superadditive_mean": round(msd([x["frac_superadditive"] for x in v])[0], 4)}
                for (c, k, ab), v in sorted(g3.items())]
        write("coalition_summary.csv", cs); write("additivity_summary.csv", asum)
        # Gate B coalition win: a book beats the best of taylor/movement on random coalitions by > sd in 3/3 seeds
        rnd = [r for r in cp if r["ckpt"] == "final" and r["kind"].startswith("random")]
        by = defaultdict(dict)
        for r in rnd:
            by[r["predictor"]].setdefault(r["seed"], []).append(r["rho"])
        seedv = {p: {s: float(np.mean(v)) for s, v in d.items()} for p, d in by.items()}
        books = [p for p in seedv if is_book(p)]
        refs = [p for p in seedv if is_ref(p)]
        win = {}
        for b in books:
            for rf in refs:
                ss = sorted(set(seedv[b]) & set(seedv[rf]))
                d = [seedv[b][s] - seedv[rf][s] for s in ss]
                sd = max(np.std(list(seedv[b].values()), ddof=1), np.std(list(seedv[rf].values()), ddof=1)) if len(ss) > 1 else np.nan
                win[f"{b} vs {rf}"] = {"diffs": [round(x, 4) for x in d], "sd": round(float(sd), 4),
                                       "win": bool(len(d) == 3 and all(x > sd for x in d))}
        verdict["coalition_win"] = {"any": any(v["win"] for v in win.values()), "pairs": win}
    json.dump(verdict, open(os.path.join(OUT, "stoprule.json"), "w"), indent=1)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) and "status" in v else v.get("any") if isinstance(v, dict) else v)
                      for k, v in verdict.items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
