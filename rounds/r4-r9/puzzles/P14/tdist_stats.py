"""P14 T-dist (secondary task, descriptive only): which code each arm finds, and joint-probe accuracy.
Reads results/runs/*_dist_eval_s*.json (untagged) and writes results/summary_tdist.json + charts/tdist_by_arm.json.

Code-type rule (written 2026-09-29 ~22:08Z, after the PREREG freeze but before any T-dist eval run finished; PREREG
only says "report thermometer vs dense", so this rule is a declared departure):
- a live bit "is" thermometer event gk (k in 0,1,2; complements count, NMI is complement-symmetric) when
  NMI(bit, gk) >= 0.7 and gk is its best match among g0, g1, g2;
- run type "thermometer" if the live bits match >= 2 distinct thermometer events;
  "dense" if they match <= 1 and the joint live partition has NMI(joint, dist class {0,1,2,>2}) >= 0.7;
  otherwise "other-sufficient" if the joint probe is sufficient (gap_all <= 0.02 nats/cell) and "insufficient" if not.
  (Refined at ~22:09Z after a dry run on the seed-100/101 TUNING runs showed 6-15-state joint codes that are sufficient
  but neither thermometer nor 4-state dense; still before any T-dist eval result.)
Thresholds 0.5 and 0.9 are reported as a sensitivity check.
Usage: tdist_stats.py [--runs DIR] [--mode eval|tune]  (tune: the tuned-value runs on seeds 100-101, a dry run)."""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["V", "BP-D", "BP-L1-cm", "BP-L0-cm"]
THERMO = ["g0", "g1", "g2"]


def mean_ci(x):
    x = [v for v in x if v is not None]
    if not x:
        return [None, None, None]
    from scipy import stats
    m = float(np.mean(x))
    if len(x) < 2:
        return [m, m, m]
    h = float(stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x)))
    return [m, m - h, m + h]


def classify(fin, thr):
    live = fin.get("live") or []
    ident = fin.get("identity") or {}
    matched = []
    for j in live:
        d = ident.get(str(j))
        if not d:
            continue
        best = max(THERMO, key=lambda c: d[c])
        if d[best] >= thr:
            matched.append(best)
    jn = (fin.get("joint_nmi") or {}).get("dist_class")
    if len(set(matched)) >= 2:
        t = "thermometer"
    elif jn is not None and jn >= thr:
        t = "dense"
    elif fin.get("gap_all") is not None and fin["gap_all"] <= 0.02:
        t = "other-sufficient"
    else:
        t = "insufficient"
    return t, sorted(set(matched)), matched


def load(rdir, mode, hp):
    runs = defaultdict(dict)
    if mode == "eval":
        for f in glob.glob(os.path.join(rdir, "*_dist_eval_s*.json")):
            if len(os.path.basename(f)[:-5].split("_")) != 4:
                continue
            d = json.load(open(f))
            runs[d["arm"]][d["seed"]] = d
    else:
        tag = {"fee": "c", "lam": "lam", "tau": "tau"}
        for arm, v in hp["tdist"].items():
            p, x = next(iter(v.items()))
            for f in glob.glob(os.path.join(rdir, f"{arm}_dist_tune_s*_{tag[p]}{x:g}.json")):
                d = json.load(open(f))
                runs[d["arm"]][d["seed"]] = d
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "results", "runs"))
    ap.add_argument("--mode", default="eval")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    hp = json.load(open(os.path.join(HERE, "results", "hp.json")))
    runs = load(a.runs, a.mode, hp)
    out = {"mode": a.mode, "hp": hp.get("tdist"), "arms": {}}
    for arm in ARMS:
        R = runs.get(arm, {})
        if not R:
            continue
        per = {}
        for s, d in sorted(R.items()):
            fin = d["final"]
            t, ms, mlist = classify(fin, 0.7)
            per[s] = {"type": t, "matched": ms, "matched_bits": mlist,
                      "type_0.5": classify(fin, 0.5)[0], "type_0.9": classify(fin, 0.9)[0],
                      "acc_all": fin.get("acc_all"), "Lstar_acc": fin.get("Lstar_acc"), "gap_all": fin.get("gap_all"),
                      "joint_nmi_dist": (fin.get("joint_nmi") or {}).get("dist_class"),
                      "n_states": (fin.get("joint_nmi") or {}).get("n_states"), "n_live": fin.get("n_live"),
                      "collapse": fin.get("collapse"),
                      "ll9": fin["output"]["9"]["greedy"]["logloss"] if fin.get("output") else None,
                      "acc9": fin["output"]["9"]["greedy"]["acc"] if fin.get("output") else None}
        n = len(per)
        out["arms"][arm] = {
            "n": n, "types": dict(Counter(v["type"] for v in per.values())),
            "types_0.5": dict(Counter(v["type_0.5"] for v in per.values())),
            "types_0.9": dict(Counter(v["type_0.9"] for v in per.values())),
            "matched_sets": dict(Counter(",".join(v["matched"]) or "-" for v in per.values())),
            "acc_all": mean_ci([v["acc_all"] for v in per.values()]),
            "Lstar_acc": mean_ci([v["Lstar_acc"] for v in per.values()]),
            "gap_all": mean_ci([v["gap_all"] for v in per.values()]),
            "joint_nmi_dist": mean_ci([v["joint_nmi_dist"] for v in per.values()]),
            "n_live": mean_ci([v["n_live"] for v in per.values()]),
            "ll9": mean_ci([v["ll9"] for v in per.values()]), "acc9": mean_ci([v["acc9"] for v in per.values()]),
            "collapse": sum(bool(v["collapse"]) for v in per.values()), "per_seed": per}
    fn = a.out or os.path.join(HERE, "results", "summary_tdist.json" if a.mode == "eval" else "summary_tdist_tune.json")
    json.dump(out, open(fn, "w"), indent=1)
    if a.mode == "eval":
        cdir = os.path.join(HERE, "charts")
        os.makedirs(cdir, exist_ok=True)
        pts = []
        for arm, r in out["arms"].items():
            m, lo, hi = r["acc_all"]
            pts.append({"x": arm, "y": m, "ci95": max(m - lo, hi - m), "lo": lo, "hi": hi})
        json.dump({"title": "P14 T-dist: joint-probe accuracy on Y = min(dist, 5) (mean over 10 seeds, 95% t CI)",
                   "x_label": "arm", "y_label": "joint-probe accuracy", "series": [{"name": "all live bits", "points": pts}]},
                  open(os.path.join(cdir, "tdist_by_arm.json"), "w"), indent=1)
    for arm, r in out["arms"].items():
        print(f"{arm:10s} n={r['n']:2d} types={r['types']} matched={r['matched_sets']} acc_all={r['acc_all'][0]:.4f}"
              f" [{r['acc_all'][1]:.4f},{r['acc_all'][2]:.4f}] oracle={r['Lstar_acc'][0]:.4f}"
              f" gap_all={r['gap_all'][0]:.4f} jNMI={r['joint_nmi_dist'][0]:.3f} live={r['n_live'][0]:.2f}"
              f" collapse={r['collapse']}")
    print("wrote", fn)


if __name__ == "__main__":
    main()
