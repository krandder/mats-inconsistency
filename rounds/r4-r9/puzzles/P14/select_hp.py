"""Label-only tuning selection (seeds 100-102, validation set 777).
Score = validation single-bit probe loss of the best live bit (no live bit: the walls-only probe loss).
Constraint: validation log loss (R's greedy-bit output on 777) within 0.02 of the no-penalty counterpart's.
Choose the feasible value with the lowest mean score; flag edge choices for bracketing (3x steps, <= 2 times).
Usage: select_hp.py [--task reach|dist] -> prints the table, writes results/hp.json (or merges into it)."""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs")
TOL = 0.02
GRIDS = {"V": "fee", "BP-D": "fee", "BP-L1-D-cm": "fee", "BP-L1-cm": "lam", "V-L1": "lam", "BP-L0-cm": "lam",
         "DisARM-L1-cm": "lam", "STG-L1-cm": "lam", "STG": "tau"}
COUNTER = {"V": "V-noD", "BP-D": "BP", "BP-L1-D-cm": "BP", "BP-L1-cm": "BP", "V-L1": "V-noD", "BP-L0-cm": "BP",
           "DisARM-L1-cm": "DisARM", "STG-L1-cm": "STG*", "STG": None}
sys.path.insert(0, HERE)
from make_queue import FEES, L1, L0, TAUS  # noqa: E402

BASE_GRID = {"V": FEES, "BP-D": FEES, "BP-L1-D-cm": FEES, "BP-L1-cm": L1, "V-L1": L1, "BP-L0-cm": L0,
             "DisARM-L1-cm": L1, "STG-L1-cm": L1, "STG": TAUS}
BPSTAR_POOL = ["BP-L1-cm", "BP-L0-cm", "BP-L1-D-cm", "DisARM-L1-cm", "STG-L1-cm"]


def load(task):
    runs = []
    for f in glob.glob(os.path.join(RUNS, f"*_{task}_tune_s*.json")):
        runs.append(json.load(open(f)))
    return runs


def score_of(r, none_loss, joint):
    fin = r["final"]
    if joint:
        if fin.get("gap_all") is None:
            return none_loss
        return fin["gap_all"] + fin["Lstar"]
    if fin.get("gap_1") is None:
        return none_loss
    return fin["gap_1"] + fin["Lstar"]


def ll_of(r):
    # the arm's own validation log loss: R's greedy-bit output log loss on 777 (greedy is the primary readout)
    return r["final"]["output"]["9"]["greedy"]["logloss"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reach")
    ap.add_argument("--none-loss", type=float, default=None, help="walls-only probe loss on 777")
    a = ap.parse_args()
    joint = a.task == "dist"
    runs = load(a.task)
    none_loss = a.none_loss
    if none_loss is None:  # no live bit: the score is the walls-only probe loss on 777 (results/none_loss.json)
        none_loss = json.load(open(os.path.join(HERE, "results", "none_loss.json")))["777"][a.task]["walls_only_loss"]
    by = defaultdict(lambda: defaultdict(list))
    for r in runs:
        arm = r["arm"]
        hp = r["hp"]
        if arm == "STG":
            key = ("tau", hp["tau"])
        elif arm in GRIDS:
            key = (GRIDS[arm], hp[GRIDS[arm]])
        else:
            key = ("base", 0)
        if arm == "STG-L1-cm":
            key = ("lam", hp["lam"])
        by[arm][key].append({"seed": r["seed"], "lam": hp.get("lam"), "tau": hp.get("tau"),
                             "score": score_of(r, none_loss, joint), "ll": ll_of(r),
                             "suff1": r["final"].get("suff1"), "gap_1": r["final"].get("gap_1"),
                             "n_live": r["final"].get("n_live"), "delistings": r.get("delistings")})
    out = {"task": a.task, "tol": TOL, "none_loss": none_loss, "table": {}, "choice": {}}
    base_ll = {}
    for arm in ("V-noD", "BP", "DisARM"):
        v = by[arm].get(("base", 0), [])
        if v:
            base_ll[arm] = float(np.mean([x["ll"] for x in v]))
    # STG: tau chosen at lambda = 0 by the same score (no constraint: it is its own counterpart)
    order = ["STG", "V", "BP-D", "BP-L1-cm", "V-L1", "BP-L0-cm", "DisARM-L1-cm", "BP-L1-D-cm", "STG-L1-cm"]
    for arm in order:
        if arm not in by:
            continue
        # stage 2 (run for every lambda / tau): keep only runs at the counterpart's chosen value
        pin = None
        if arm == "BP-L1-D-cm" and "BP-L1-cm" in out["choice"]:
            pin = ("lam", out["choice"]["BP-L1-cm"]["lam"])
        if arm == "STG-L1-cm" and "STG" in out["choice"]:
            pin = ("tau", out["choice"]["STG"]["tau"])
        if pin is not None:
            by[arm] = {k: [x for x in v if abs(x[pin[0]] - pin[1]) < 1e-12] for k, v in by[arm].items()}
            by[arm] = {k: v for k, v in by[arm].items() if v}
            if not by[arm]:
                print(f"{arm}: no runs at {pin[0]}={pin[1]:g}")
                continue
        rows = []
        for (kname, val), lst in sorted(by[arm].items(), key=lambda kv: kv[0][1]):
            rows.append({"param": kname, "value": val, "n": len(lst),
                         "score": float(np.mean([x["score"] for x in lst])),
                         "ll": float(np.mean([x["ll"] for x in lst])),
                         "suff1": int(sum(bool(x["suff1"]) for x in lst)),
                         "gap_1": [x["gap_1"] for x in lst], "n_live": [x["n_live"] for x in lst],
                         "delistings": [x["delistings"] for x in lst]})
        if pin is not None:
            out.setdefault("pinned", {})[arm] = {pin[0]: pin[1]}
        cp = COUNTER[arm]
        if cp == "STG*":
            tau = out["choice"].get("STG", {}).get("tau")
            v = by["STG"].get(("tau", tau), [])
            ref = float(np.mean([x["ll"] for x in v])) if v else None
        else:
            ref = base_ll.get(cp) if cp else None
        for r in rows:
            r["feasible"] = True if ref is None else bool(r["ll"] <= ref + TOL)
        # bracketing: extension values (3x steps beyond the base grid, run together) enter one at a time, only
        # while the choice sits at that edge, at most twice -- the same outcome as extending sequentially
        base = set(BASE_GRID[arm])
        near = lambda v: any(abs(v - b) <= 1e-9 * max(1.0, abs(b)) for b in base)
        vis = [r for r in rows if near(r["value"])]
        ext_lo = sorted([r for r in rows if not near(r["value"]) and r["value"] < min(base)], key=lambda r: -r["value"])
        ext_hi = sorted([r for r in rows if not near(r["value"]) and r["value"] > max(base)], key=lambda r: r["value"])
        pick = lambda rs: (min([r for r in rs if r["feasible"]], key=lambda r: r["score"])
                           if any(r["feasible"] for r in rs) else min(rs, key=lambda r: r["ll"]))
        best = pick(vis)
        n_ext = 0
        for _ in range(2):
            vals = [r["value"] for r in vis]
            if best["value"] == min(vals) and ext_lo:
                vis.append(ext_lo.pop(0))
            elif best["value"] == max(vals) and ext_hi:
                vis.append(ext_hi.pop(0))
            else:
                break
            n_ext += 1
            best = pick(vis)
        for r in rows:
            r["used"] = any(r is x for x in vis)
        feas = [r for r in vis if r["feasible"]]
        binds = any(not r["feasible"] for r in vis)
        vals = [r["value"] for r in vis]
        edge = None
        if len(vals) > 1 and best["value"] == min(vals):
            edge = "low"
        elif len(vals) > 1 and best["value"] == max(vals):
            edge = "high"
        out["table"][arm] = {"rows": rows, "counterpart": cp, "counterpart_ll": ref,
                             "tolerance_binds": binds, "no_feasible": not feas, "extensions": n_ext}
        out["choice"][arm] = {best["param"]: best["value"], "score": best["score"], "ll": best["ll"],
                              "edge": edge, "extensions": n_ext, "suff1_tune": best["suff1"], "gap_1_tune": best["gap_1"]}
    pool = [(out["choice"][a]["score"], BPSTAR_POOL.index(a), a) for a in BPSTAR_POOL if a in out["choice"]]
    if pool:
        s0 = min(pool)[0]
        tied = [p for p in pool if abs(p[0] - s0) < 1e-9]
        star = "BP-L1-cm" if any(p[2] == "BP-L1-cm" for p in tied) else min(pool)[2]
        out["BP*"] = star
        ch = out["choice"][star]
        gaps = [g for g in ch["gap_1_tune"] if g is not None]
        out["endpoint_switch"] = bool(ch["suff1_tune"] == 3 and len(gaps) == 3 and max(gaps) < 0.005)
    for arm, t in out["table"].items():
        print(f"\n{arm}  (counterpart {t['counterpart']} ll={t['counterpart_ll']})  tolerance binds: {t['tolerance_binds']}")
        for r in t["rows"]:
            print(f"   {r['param']}={r['value']:<8g} n={r['n']} score={r['score']:.4f} ll={r['ll']:.4f} feas={r['feasible']} used={r['used']} "
                  f"suff1={r['suff1']} live={r['n_live']} del={r['delistings']}")
        print("   ->", out["choice"][arm])
    print("\nBP* =", out.get("BP*"), " endpoint switch:", out.get("endpoint_switch"))
    fn = os.path.join(HERE, "results", f"hp_select_{a.task}.json")
    json.dump(out, open(fn, "w"), indent=1)
    print("wrote", fn)


if __name__ == "__main__":
    main()
