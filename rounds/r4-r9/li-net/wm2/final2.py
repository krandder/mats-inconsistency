"""TEST analysis for the wm2 final phase (run once, after the champion is fixed on validation).

  python3 final2.py --task t1 --champ TAG [--others TAG,...] [--runs /tmp/wm2/runs] [--wave F]
  python3 final2.py --task t2 --champ TAG --blind TAG [--others ...]

Reads runs/<wave>/<task>/F-<tag>/*.json (split 'test'), compares the champion with F-B-rbtd8 and F-B-k0 (paired by seed =
same training boards): KL x1e3 per N (mean, 95% t-interval), paired differences with 95% t-intervals, geometric-mean
ratios, outcome log-loss differences, and the pre-registered decision (PREREG.md). T2 adds the selection-unaware exact
posterior's KL to the selection-aware one on the TEST boards (a Bayesian that ignores where probes land) and the
board-blind-location twin. Writes results/test-<task>-final.{json,md} and curves.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze2 as A  # noqa: E402

NS = [100, 300, 1000, 3000, 10000]


def kl_bern(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def n_to_reach(curve, target):
    pts = sorted(curve.items())
    if pts[0][1] <= target:
        return float(pts[0][0])
    for (n0, k0), (n1, k1) in zip(pts, pts[1:]):
        if k1 <= target:
            f = (math.log(k0) - math.log(target)) / (math.log(k0) - math.log(k1))
            return float(math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))))
    return float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--champ", required=True)
    ap.add_argument("--blind", default="")
    ap.add_argument("--others", default="")
    ap.add_argument("--pairs", default="", help="extra paired comparisons A:B,C:D (A minus B)")
    ap.add_argument("--runs", default="/tmp/wm2/runs")
    ap.add_argument("--wave", default="F")
    ap.add_argument("--data", default="/tmp/wm2")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    rows, _ = A.load(a.runs, a.task, "test", [a.wave], strip_prefix="F-")
    tags = [a.champ, "B-rbtd8", "B-k0"] + ([a.blind] if a.blind else []) + [t for t in a.others.split(",") if t]
    res_time = __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())
    rows = [r for r in rows if r["tag"] in tags]
    res = {"task": a.task, "tags": tags, "run_at": res_time, "n_runs": {t: sum(r["tag"] == t for r in rows) for t in tags}, "per_tag": {}, "paired": {}}
    for t in tags:
        res["per_tag"][t] = {}
        for n in NS:
            v = [r for r in rows if r["tag"] == t and r["n"] == n]
            if not v:
                continue
            m, h = A.tci([r["kl"] for r in v])
            res["per_tag"][t][n] = {"kl_mean": m, "kl_ci95": h, "seeds": sorted(r["seed"] for r in v),
                                    "logloss": float(np.mean([r["logloss"] for r in v])), "ece": float(np.mean([r["ece"] for r in v])),
                                    "kl_stage": [float(np.mean([r[f"kl_s{i}"] for r in v])) for i in range(4)],
                                    "flops_per_board": float(np.mean([r["flops_per_board"] or 0 for r in v])),
                                    "flops_per_board_total": float(np.mean([r["flops_per_board_total"] or 0 for r in v])),
                                    "best_step": float(np.mean([r["best_step"] for r in v])), "stops": sorted({r["stop"] for r in v})}
    for t in tags:
        for base in ("B-rbtd8", "B-k0") + ((a.blind,) if a.blind and t != a.blind else ()):
            if t == base:
                continue
            key = f"{t} vs {base}"
            res["paired"][key] = {}
            for n in NS:
                A_ = {r["seed"]: r for r in rows if r["tag"] == t and r["n"] == n}
                B_ = {r["seed"]: r for r in rows if r["tag"] == base and r["n"] == n}
                s = sorted(set(A_) & set(B_))
                if len(s) < 2:
                    continue
                d = [A_[k]["kl"] - B_[k]["kl"] for k in s]
                m, h = A.tci(d)
                dl = [A_[k]["logloss"] - B_[k]["logloss"] for k in s]
                ml, hl = A.tci(dl)
                lr = [math.log(A_[k]["kl"] / B_[k]["kl"]) for k in s]
                mr, hr = A.tci(lr)
                res["paired"][key][n] = {"dkl": m, "dkl_ci95": h, "below0": bool(m + h < 0), "ratio_geo": math.exp(mr),
                                         "ratio_ci95": [math.exp(mr - hr), math.exp(mr + hr)], "dlogloss": ml, "dlogloss_ci95": hl, "seeds": s}
    for pair in [x for x in a.pairs.split(",") if x]:
        t, base = pair.split(":")
        key = f"{t} vs {base}"
        res["paired"][key] = {}
        for n in NS:
            A_ = {r["seed"]: r for r in rows if r["tag"] == t and r["n"] == n}
            B_ = {r["seed"]: r for r in rows if r["tag"] == base and r["n"] == n}
            s_ = sorted(set(A_) & set(B_))
            if len(s_) < 2:
                continue
            m, h = A.tci([A_[k]["kl"] - B_[k]["kl"] for k in s_])
            mr, hr = A.tci([math.log(A_[k]["kl"] / B_[k]["kl"]) for k in s_])
            ml, hl = A.tci([A_[k]["logloss"] - B_[k]["logloss"] for k in s_])
            res["paired"][key][n] = {"dkl": m, "dkl_ci95": h, "below0": bool(m + h < 0), "ratio_geo": math.exp(mr),
                                     "ratio_ci95": [math.exp(mr - hr), math.exp(mr + hr)], "dlogloss": ml, "dlogloss_ci95": hl, "seeds": s_}
    ch = a.champ
    pr = res["paired"].get(f"{ch} vs B-rbtd8", {})
    pk = res["paired"].get(f"{ch} vs B-k0", {})
    res["decision"] = {
        "beats_rbtd8_every_n_test": bool(len(pr) == len(NS) and all(v["below0"] for v in pr.values())),
        "fails_vs_rbtd8_at": [n for n in NS if n in pr and not pr[n]["below0"]],
        "beats_k0_every_n_test": bool(len(pk) == len(NS) and all(v["below0"] for v in pk.values())),
    }
    # boards the champion needs to reach rbtd8's (and k0's) 10k-board KL (log-linear interpolation; seed bootstrap)
    rng = np.random.default_rng(0)
    for base in ("B-rbtd8", "B-k0"):
        sa = {n: [r["kl"] for r in rows if r["tag"] == ch and r["n"] == n] for n in NS}
        sb = [r["kl"] for r in rows if r["tag"] == base and r["n"] == 10000]
        if all(sa[n] for n in NS) and sb:
            frac = n_to_reach({n: float(np.mean(sa[n])) for n in NS}, float(np.mean(sb))) / 10000
            boots = []
            for _ in range(2000):
                c = {n: float(np.mean(rng.choice(sa[n], len(sa[n])))) for n in NS}
                boots.append(n_to_reach(c, float(np.mean(rng.choice(sb, len(sb))))) / 10000)
            boots = np.where(np.isfinite(boots), boots, 1e3)
            res["decision"][f"frac_boards_to_reach_{base}_10k"] = [frac, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    if a.task == "t2":
        d = np.load(os.path.join(a.data, "data-t2", "test-n5000.post.npz"))
        k = kl_bern(d["post"], d["post_unaware"])
        res["unaware_exact_kl_test"] = {"mean": float(k.mean()), "stage": k.mean((0, 2)).tolist()}
        ref = float(k.mean())
        res["decision_t2"] = {}
        for t in tags:
            if t in ("B-rbtd8", "B-k0"):
                continue
            per = {}
            for n in NS:
                v = [r["kl"] for r in rows if r["tag"] == t and r["n"] == n]
                if len(v) >= 2:
                    m, h = A.tci(np.array(v) - ref)
                    per[n] = {"kl_minus_unaware": m, "ci95": h, "below0": bool(m + h < 0)}
            res["decision_t2"][t] = per
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"test-{a.task}-final.json"), "w"), indent=1)
    L = [f"TEST ({a.task}), KL x 1e3, mean over seeds +- 95% t-interval half-width [seeds]:", "",
         "| arm | " + " | ".join(f"N={n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for t in tags:
        cells = []
        for n in NS:
            e = res["per_tag"][t].get(n)
            cells.append(f"{1e3 * e['kl_mean']:.3f} +- {1e3 * e['kl_ci95']:.3f} [{len(e['seeds'])}]" if e else "")
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "Paired differences (KL x 1e3, arm minus base, same seeds = same training boards; 95% t-interval) and geometric-mean ratio:", "",
          "| comparison | " + " | ".join(f"N={n}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
    for key, per in res["paired"].items():
        L.append(f"| {key} | " + " | ".join((f"{1e3 * per[n]['dkl']:+.3f} +- {1e3 * per[n]['dkl_ci95']:.3f} (x{per[n]['ratio_geo']:.2f})" if n in per else "") for n in NS) + " |")
    L += ["", f"Decision: {json.dumps(res['decision'])}"]
    if a.task == "t2":
        L += ["", f"Selection-unaware exact posterior (ignores where probes land): KL to the selection-aware posterior on TEST = {1e3 * res['unaware_exact_kl_test']['mean']:.2f}e-3 "
              f"(per stage {', '.join(f'{1e3 * x:.2f}' for x in res['unaware_exact_kl_test']['stage'])})",
              "T2 rule (b)(i), arm KL minus that reference (x1e3, 95% t-interval): " + json.dumps({t: {n: round(1e3 * v['kl_minus_unaware'], 3) for n, v in per.items()} for t, per in res['decision_t2'].items()})]
    open(os.path.join(a.out, f"test-{a.task}-final.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    for kind in ("boards", "compute"):
        A.plot(rows, tags, os.path.join(a.out, f"test-{a.task}-curve_kl_vs_{kind}.png"), "test", f"TEST {a.task} (5 seeds) - vs {kind}", kind,
               ref=("selection-unaware exact posterior", res["unaware_exact_kl_test"]["mean"]) if a.task == "t2" else None)


if __name__ == "__main__":
    main()
