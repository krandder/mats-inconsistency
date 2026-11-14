"""P11 selection and gates. Validation mazes only; test mazes are never read here except in 'headroom' (the
headroom gate is defined on the test endpoints).

  python3 select_p11.py OUT             pilot -> OUT/params.json (hp per arm, training + floor gates, proceed flag)
  python3 select_p11.py OUT dtl         R-prog pilot -> OUT/dtl_grid.json: DT-L pilot a values = the R-prog winner
                                        and its better-scoring grid neighbour (the only one at a grid end)
  python3 select_p11.py OUT headroom    core R-prog -> OUT/headroom.json (skip_s3 when R-prog's best validated
                                        readout is >= 95% on E1, E2 and E3)

Selection score: mean exact_acc on the 128 validation size-17 mazes at 10x and 100x. A tie is a score within
1/128 of the best; ties go to the middle of the grid (for 4 values: the 2nd, then the 3rd). R-prog and DT-L are
scored with their best readout; across readout families a tie goes to the simpler family, in the order
inst, stop, ces, ema, frz, lmsr, tv.
"""
import glob
import json
import os
import sys

TIE = 1 / 128
GRIDS = {"R-prog": ("a", [0.01, 0.1, 0.5, 1.0]), "DT-L": ("a", [0.01, 0.1, 0.5, 1.0]),
         "M-budget": ("W", [3, 6, 12, 24]), "M-TV": ("V", [4, 8, 16, 32]), "M-cap": ("eps", [0.1, 0.3, 1.0]),
         "M-clip": ("c", [0.1, 0.3, 1.0, 3.0]), "M-B": None, "R-final": None}
FAMILY = ["inst", "stop", "ces", "ema", "frz", "lmsr", "tv"]


def hpkey(hp):
    return "_".join(f"{k}{v}" for k, v in sorted(hp.items())) or "none"


def table(setres):
    """-> {readout: {ratio: acc}} for online streams and derived readouts."""
    ratios = [c["ratio"] for c in setres["checkpoints"]]
    tab = {}
    for c in setres["checkpoints"]:
        for n, v in c["acc"].items():
            tab.setdefault(n, {})[c["ratio"]] = v
    for n, d in setres.get("derived", {}).items():
        tab[n] = dict(zip(ratios, d["acc"]))
    return tab


def family(name):
    for f in FAMILY:
        if name.startswith(f):
            return f
    return name


def grid_value(name):
    f = family(name)
    rest = name[len(f):]
    try:
        return float(rest.split("_")[0]) + (float(rest.split("_")[1]) * 1e-3 if "_" in rest else 0.0)
    except ValueError:
        return 0.0


def middle_rank(i, n):
    return (abs(i - (n - 1) / 2), i)


def score(tab, r):
    return 0.5 * (tab[r][10] + tab[r][100])


def best_readout(tab, allowed=None):
    """-> (readout, score). Ties: simpler family first, then the middle of that family's grid."""
    names = [r for r in tab if 10 in tab[r] and 100 in tab[r] and (allowed is None or family(r) in allowed)]
    sc = {r: score(tab, r) for r in names}
    top = max(sc.values())
    tied = [r for r in names if sc[r] >= top - TIE]
    fam = min(tied, key=lambda r: FAMILY.index(family(r)) if family(r) in FAMILY else 99)
    fam = family(fam)
    grid = sorted({grid_value(r) for r in names if family(r) == fam})
    cand = [r for r in tied if family(r) == fam]
    r = min(cand, key=lambda r: middle_rank(grid.index(grid_value(r)), len(grid)))
    return r, sc[r]


def pick_grid(scores, grid):
    """scores {grid value: score} -> chosen value (tie -> middle of the grid)."""
    top = max(scores.values())
    tied = [g for g in grid if g in scores and scores[g] >= top - TIE]
    return min(tied, key=lambda g: middle_rank(grid.index(g), len(grid)))


def load_eval(d, plan):
    p = os.path.join(d, f"eval_{plan}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def arm_scores(out, arm, vals):
    """R-prog-style pilot scores (best readout on val17 at 10x/100x) for the grid values that exist."""
    g = GRIDS[arm]
    sc = {}
    for v in vals:
        ev = load_eval(os.path.join(out, "pilot", f"{arm}_{hpkey({g[0]: v})}_s0"), "pilot")
        if ev is not None:
            sc[v] = best_readout(table(ev["sets"]["val17"]))[1]
    return sc


def dtl(out):
    vals = GRIDS["R-prog"][1]
    sc = arm_scores(out, "R-prog", vals)
    win = pick_grid(sc, vals)
    i = vals.index(win)
    nb = [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals) and vals[j] in sc]
    nb = max(nb, key=lambda v: (sc[v], -abs(vals.index(v) - (len(vals) - 1) / 2))) if nb else vals[i - 1 if i else 1]
    res = {"a": sorted([win, nb]), "r_prog_winner": win, "neighbour": nb, "r_prog_scores": {str(k): v for k, v in sc.items()}}
    json.dump(res, open(os.path.join(out, "dtl_grid.json"), "w"), indent=1)
    print(json.dumps(res))
    return 0


def grid_vals(out, arm):
    g = GRIDS[arm]
    if arm == "DT-L" and os.path.exists(os.path.join(out, "dtl_grid.json")):
        return json.load(open(os.path.join(out, "dtl_grid.json")))["a"]
    return [None] if g is None else g[1]


def select(out):
    res = {"rule": __doc__.strip().splitlines()[9:13], "arms": {}, "hp": {}, "flags": []}
    for arm, g in GRIDS.items():
        vals = grid_vals(out, arm)
        rows = {}
        for v in vals:
            hp = {} if g is None else {g[0]: v}
            d = os.path.join(out, "pilot", f"{arm}_{hpkey(hp)}_s0")
            ev = load_eval(d, "pilot")
            if ev is None:
                rows[v] = {"missing": True}
                continue
            tab17 = table(ev["sets"]["val17"])
            if arm in ("R-prog", "DT-L", "R-final"):
                ro, sc = best_readout(tab17)
            else:
                ro, sc = "inst", score(tab17, "inst")
            tab9, tab29 = table(ev["sets"]["val9"]), table(ev["sets"]["val29"])
            tr = json.load(open(os.path.join(d, "train.json")))
            sat = ev["sets"]["val17"].get("sat_frac")
            rows[v] = {"score": sc, "readout": ro, "inst_score": score(tab17, "inst"),
                       "val9_1x_inst": tab9["inst"][1], "val17_peak_inst": max(tab17["inst"].values()),
                       "val29_peak_inst": max(tab29["inst"].values()),
                       "val17_inst": tab17["inst"], "val9_inst": tab9["inst"], "sat_frac_val17": sat,
                       "train_final_loss": tr["log"][-1]["loss"] if tr["log"] else None,
                       "train_sec": tr["train_sec"], "final_loss_finite": tr["final_loss_finite"]}
            if sat is not None and sat > 0.2:
                res["flags"].append(f"{arm} {hp}: saturated pixel-steps {sat:.2f} > 0.2")
        ok = {v: r["score"] for v, r in rows.items() if "score" in r}
        if not ok:
            res["arms"][arm] = {"rows": rows, "chosen": None}
            continue
        chosen = None if g is None else pick_grid(ok, vals)
        res["arms"][arm] = {"rows": {str(k): v for k, v in rows.items()}, "chosen": chosen}
        res["hp"][arm] = {} if g is None else {g[0]: chosen}
    rp = res["arms"].get("R-prog", {})
    rrow = rp.get("rows", {}).get(str(rp.get("chosen")), {})
    gates = {}
    gates["training"] = {"value": rrow.get("val9_1x_inst"), "threshold": 0.9,
                         "pass": rrow.get("val9_1x_inst") is not None and rrow["val9_1x_inst"] >= 0.9}
    gates["floor"] = {"val17_peak": rrow.get("val17_peak_inst"), "val29_peak": rrow.get("val29_peak_inst"),
                      "thresholds": [0.5, 0.25],
                      "pass": rrow.get("val17_peak_inst") is not None and rrow["val17_peak_inst"] >= 0.5
                      and rrow["val29_peak_inst"] >= 0.25}
    core = ("R-prog", "DT-L", "M-B", "M-budget", "M-TV")
    missing = [a for a in core if a not in res["hp"]]
    res["gates"] = gates
    res["missing_core"] = missing
    res["proceed"] = bool(gates["training"]["pass"] and gates["floor"]["pass"] and not missing)
    res.update({"steps": 6000, "width": 32, "core_seeds": [1, 2, 3, 4, 5], "secondary_seeds": [1, 2, 3],
                "reuse_pilot_seed0": False})
    for a in ("R-final", "M-cap", "M-clip"):
        if a not in res["hp"]:
            res["hp"][a] = {} if GRIDS[a] is None else {GRIDS[a][0]: GRIDS[a][1][len(GRIDS[a][1]) // 2]}
            res["flags"].append(f"{a}: pilot missing, default hp used")
    json.dump(res, open(os.path.join(out, "params.json"), "w"), indent=1)
    print(json.dumps({"proceed": res["proceed"], "hp": res["hp"], "gates": gates, "flags": res["flags"]}))
    return 0 if res["proceed"] else 3


def headroom(out):
    p = json.load(open(os.path.join(out, "params.json")))
    hp = p["hp"]["R-prog"]
    runs = sorted(glob.glob(os.path.join(out, "core", f"R-prog_{hpkey(hp)}_s*")))
    evs = [load_eval(d, "core") for d in runs]
    evs = [e for e in evs if e is not None]
    tabs = [table(e["sets"]["val17"]) for e in evs]
    common = set.intersection(*[set(t) for t in tabs])
    mean_tab = {r: {k: sum(t[r][k] for t in tabs) / len(tabs) for k in (10, 100)} for r in common}
    ro, sc = best_readout(mean_tab)
    cells = {"E1": ("test17", 1000), "E2": ("test29", 100), "E3": ("test59", 100)}
    val = {}
    for k, (s, r) in cells.items():
        val[k] = sum(table(e["sets"][s])[ro][r] for e in evs) / len(evs)
    skip = all(v >= 0.95 for v in val.values())
    res = {"readout": ro, "val_score": sc, "n_seeds": len(evs), "endpoints": val, "skip_s3": skip}
    json.dump(res, open(os.path.join(out, "headroom.json"), "w"), indent=1)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    o = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "select"
    sys.exit({"headroom": headroom, "dtl": dtl, "select": select}[mode](o))
