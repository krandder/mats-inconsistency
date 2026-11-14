"""P15 analysis.
  python analyze.py pilot           pilot summary (collapse check, |lam| growth, timing) -> results/pilot_summary.json
  python analyze.py freeze          tuning -> frozen.json (config per arm x primary cell, selected on 2x validation
                                    exact at the best c, indist log loss as tiebreak; P-eq width likewise)
  python analyze.py report          full grid -> results/analysis.json + results/charts.json (+ printed summary)
Reads every results JSON under results/vm_out*/results/ (synced from the VMs)."""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CS = (0.1, 0.25, 0.5, 1, 2, 4)
PRIM = {"prefix": 16, "maze": 32}
BP = ("A", "A-prog", "A-ds", "A-1step")                                   # discrete backprop arms (S1)
S2_BASE = ("A", "A-prog", "A-ds", "A-1step", "A-soft-prog", "P-eq", "P-pub")
A_ALL = ("A", "A-prog", "A-ds", "A-1step", "A-soft-prog")
SOLVE = 0.99


# ------------------------------------------------------------------ loading
def load_all():
    R = {}
    for f in sorted(glob.glob(os.path.join(OUT, "vm_out*", "results", "*.json"))):
        jid = os.path.basename(f)[:-5]
        try:
            R[jid] = json.load(open(f))
        except Exception as ex:  # noqa: BLE001
            print("unreadable", f, ex)
    return R


def size_T(kind, size):
    return size if kind == "prefix" else size * size


def c_idx(item, kind, c):
    T = max(1, int(round(c * size_T(kind, item["size"]))))
    Ts = item["Ts"]
    if T in Ts:
        return Ts.index(T)
    le = [i for i, t in enumerate(Ts) if t <= T]      # capped grids (DT maze cap)
    return le[-1] if le else 0


def ex_at(item, kind, c):
    return np.array(item["exact"])[:, c_idx(item, kind, c)]


def tci(v):
    """mean and 95% t-interval over seeds."""
    v = np.asarray(v, float)
    n = len(v)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    m = float(v.mean())
    if n < 2 or v.std() == 0:
        return {"mean": m, "lo": m, "hi": m, "n": n}
    h = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


def cp(k, n, a=0.05):
    lo = float(stats.beta.ppf(a / 2, k, n - k + 1)) if k > 0 else 0.0
    hi = float(stats.beta.ppf(1 - a / 2, k + 1, n - k)) if k < n else 1.0
    return lo, hi


def rnd(x, d=4):
    if isinstance(x, dict):
        return {k: rnd(v, d) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, d) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), d)
    if isinstance(x, np.integer):
        return int(x)
    return x


# ------------------------------------------------------------------ pilot
def pilot():
    R = load_all()
    out = {}
    for jid, r in R.items():
        if not jid.startswith("pilot_"):
            continue
        j = r["job"]
        ind = r.get("indist", {})
        ex = np.array(ind.get("exact", [np.nan]))
        row = {"arm": j["arm"], "kind": j["kind"], "D": j["D"], "steps": r["steps"], "wall_train_s": r["wall_train"],
               "s_per_step": r["wall_train"] / max(1, r["steps"]), "E": len(r["elements"]),
               "indist_exact": ex.tolist(), "indist_ll": ind.get("logloss")}
        if "diag" in ind:
            nmi = np.array([[e["nmi_g"] for e in lv] for lv in ind["diag"]])       # (D, E)
            row["nmi_mean_over_levels"] = nmi.mean(0).tolist()
            row["collapse"] = [bool(a < 0.9 or b < 0.1) for a, b in zip(ex, nmi.mean(0))]
        lam = r.get("lam_abs_by_level")
        if lam:
            lam = np.array(lam, float)          # levels 1..D; level D is the head message
            tr = lam[:-1]                         # trader messages, levels 1..D-1
            down = tr[:-1] / np.maximum(tr[1:], 1e-12)    # |lam_k| / |lam_{k+1}|: growth with distance from top
            row["lam_abs_by_level"] = lam.tolist()
            row["lam_ratio_per_level_down_geomean"] = float(np.exp(np.log(np.maximum(down, 1e-12)).mean()))
            row["lam_ratio_per_level_down_max"] = float(down.max())
            row["bl_fan_trigger"] = bool(row["lam_ratio_per_level_down_geomean"] > 1.5)
        for which in ("val", "test"):
            if which in r:
                row[which] = [{"size": it["size"], "exact_by_c": {str(c): np.round(ex_at(it, j["kind"], c), 3).tolist()
                                                                   for c in CS}} for it in r[which]]
        out[jid] = row
        print(jid, {k: rnd(v, 3) for k, v in row.items() if k not in ("val", "test", "lam_abs_by_level")})
    json.dump(rnd(out), open(os.path.join(OUT, "pilot_summary.json"), "w"), indent=1)


# ------------------------------------------------------------------ freeze (tuning)
def best_c(item_list_per_elem, kind, items, idx):
    """items: val items (one size); idx: element indices -> (best mean exact, best c)."""
    best = (-1, None)
    for c in CS:
        m = float(np.mean([ex_at(items[0], kind, c)[i] for i in idx]))
        if m > best[0] + 1e-12:
            best = (m, c)
    return best


def freeze():
    R = load_all()
    frozen = {}
    for jid, r in sorted(R.items()):
        if not jid.startswith("tune_") or "val" not in r:
            continue
        j = r["job"]
        cell = f"{j['kind']}_D{j['D']}"
        kind = j["kind"]
        ll = np.array(r["indist"]["logloss"]) if "indist" in r else np.zeros(len(r["elements"]))
        if j["arm"] == "P-eq":
            idx = list(range(len(r["elements"])))
            sc, c = best_c(None, kind, r["val"], idx)
            cand = {"width": j["width"], "lr": j.get("lr", 1e-3), "val_exact": sc, "c": c, "ll": float(ll[idx].mean())}
            cur = frozen.setdefault(cell, {}).get("P-eq")
            if cur is None or (sc, -cand["ll"]) > (cur["val_exact"], -cur["ll"]):
                frozen[cell]["P-eq"] = cand
            continue
        cfgs = j["configs"]
        rows = []
        for ci, cfg in enumerate(cfgs):
            idx = [i for i, e in enumerate(r["elements"]) if e["config"] == cfg]
            sc, c = best_c(None, kind, r["val"], idx)
            rows.append({"config": cfg, "val_exact": sc, "c": c, "ll": float(ll[idx].mean()),
                         "val_exact_seeds": [float(ex_at(r["val"][0], kind, c)[i]) for i in idx]})
        rows.sort(key=lambda z: (-z["val_exact"], z["ll"]))
        frozen.setdefault(cell, {})[j["arm"]] = dict(rows[0], all=rows)
    json.dump(rnd(frozen), open(os.path.join(HERE, "frozen.json"), "w"), indent=1)
    for cell, arms in frozen.items():
        for a, v in arms.items():
            print(cell, a, {k: rnd(v[k], 3) for k in v if k != "all"})


# ------------------------------------------------------------------ report
def arm_cells(R, pref="full_"):
    """{(kind, D, mod): {arm: result}} from full-grid jobs (+ P-pub)."""
    cells = {}
    for jid, r in R.items():
        if jid.startswith(pref):
            j = r["job"]
            cells.setdefault((j["kind"], j["D"], j.get("mod", 2)), {})[j["arm"]] = r
        elif jid.startswith("P-pub_"):
            if jid.endswith("_pubonly") and jid[:-len("_pubonly")] in R:
                continue  # the published net alone; superseded by the file that also has our 3 retrains
            j = {"kind": r["kind"], "D": r["D"]}
            cells.setdefault((r["kind"], r["D"], 2), {})["P-pub"] = r
    return cells


def frozen_c(r, kind):
    """c chosen on the run's own validation set (mean over elements; ties -> smaller c)."""
    if "val" not in r or not r["val"]:
        return 1
    idx = list(range(np.array(r["val"][0]["exact"]).shape[0]))
    return best_c(None, kind, r["val"], idx)[1]


def curve_ll(r):
    """mean-over-seeds log loss at each budget mark: (fracs, ll_mean, steps)."""
    cv = [c for c in r.get("curve", []) if "logloss" in c]
    if not cv:
        return None
    B = r["B_D"] * r["job"].get("frac", 1.0)
    return (np.array([c["flops"] / r["B_D"] for c in cv]), np.array([np.mean(c["logloss"]) for c in cv]),
            np.array([c["step"] for c in cv]), np.array([np.mean(c["exact"]) for c in cv]), B)


def flops_to(r, thr=0.05):
    c = curve_ll(r)
    if c is None:
        return None
    fr, ll, _, _, _ = c
    hit = np.where(ll <= thr)[0]
    return float(fr[hit[0]] * r["B_D"]) if len(hit) else float("inf")


def steps_to(r, thr=0.05):
    c = curve_ll(r)
    if c is None:
        return None
    _, ll, st, _, _ = c
    hit = np.where(ll <= thr)[0]
    return float(st[hit[0]]) if len(hit) else float("inf")


def ll_at_step(r, step):
    c = curve_ll(r)
    _, ll, st, _, _ = c
    if step >= st[-1]:
        return float(ll[-1]), True
    if step <= st[0]:
        return float(ll[0]), False
    return float(np.interp(step, st, ll)), False


def ratio(a, b):
    if a is None or b is None:
        return None
    if math.isinf(a) and math.isinf(b):
        return None
    if math.isinf(a):
        return float("inf")
    if math.isinf(b):
        return 0.0
    return a / b


def report():
    R = load_all()
    cells = arm_cells(R)
    timing = {}
    for jid, r in R.items():
        if jid.startswith("timing_"):
            j = r["job"]
            timing[(j["arm"], j["kind"], j["D"])] = r["wall_train"] / max(1, r["steps"])
    gates = {}
    for f in sorted(glob.glob(os.path.join(OUT, "vm_out*", "gates", "*.json"))):
        g = json.load(open(f))
        gates[os.path.basename(f)[:-5]] = g.get("pass", g.get("pass_t1"))
    A = {"gates": gates, "cells": {}, "tasks": {}}
    # G5 composite.  The first "g5 pub" run fed the published prefix net raw 0/1 bits (Bansal's loader centres them,
    # x - 0.5) and scored chance; "g5 pubprefix" re-ran that half with centring.  Its maze half is unaffected.
    gd = {}
    for f in sorted(glob.glob(os.path.join(OUT, "vm_out*", "gates", "G5_*.json"))):
        gd[os.path.basename(f)[:-5]] = json.load(open(f))
    if gd:
        rt = [gd[k] for k in sorted(gd) if k.startswith("G5_retrain_s")]
        pub, pp = gd.get("G5_pub", {}), gd.get("G5_pub_prefix", {})
        g5 = {"ref_max_logit_diff": max(gd.get("G5_ref", {}).get(k, float("nan")) for k in
                                        ("prefix_max_abs_logit_diff", "maze_max_abs_logit_diff")),
              "ref_pass": gd.get("G5_ref", {}).get("pass"),
              "pub_prefix_512_raw_bits": pub.get("prefix_512", {}).get("peak"),
              "pub_prefix_512": pp.get("peak"), "pub_prefix_512_T": pp.get("peak_T"), "pub_prefix_cp95_lo": pp.get("cp95_lo"),
              "pub_maze_59": pub.get("maze_59", {}).get("peak"), "pub_maze_59_T": pub.get("maze_59", {}).get("peak_T"),
              "pub_maze_cp95_lo": pub.get("maze_cp95_lo"), "published": {"prefix_512": 0.9712, "maze_59": 0.9730},
              "retrain_peaks": [r.get("peak") for r in rt], "retrain_n": len(rt)}
        g5["pub_pass"] = (None if pp.get("peak") is None or g5["pub_maze_59"] is None
                          else bool(pp["peak"] >= 0.921 and g5["pub_maze_59"] >= 0.923))
        g5["retrain_pass"] = (None if len(rt) < 2 else bool(sum(r.get("peak", 0) >= 0.92 for r in rt) >= 2))
        g5["pass"] = (None if None in (g5["ref_pass"], g5["pub_pass"], g5["retrain_pass"])
                      else bool(g5["ref_pass"] and g5["pub_pass"] and g5["retrain_pass"]))
        A["G5"] = g5
    charts = {"curves": {}, "extrap": {}, "delta": {}}
    for (kind, D, mod), arms in sorted(cells.items()):
        key = f"{kind}_D{D}" + (f"_mod{mod}" if mod != 2 else "")
        C = {}
        for arm, r in sorted(arms.items()):
            row = {}
            if arm == "P-pub":
                c = frozen_c(r, kind)
                row["c"] = c
                row["test"] = {it["size"]: tci(ex_at(it, kind, c)) for it in r["test"]}
                row["test_seeds"] = {it["size"]: ex_at(it, kind, c).tolist() for it in r["test"]}
                row["val_exact"] = float(np.mean(ex_at(r["val"][0], kind, c))) if r.get("val") else None
                row["n"] = len(r["elements"])
                C[arm] = row
                continue
            ind = r.get("indist", {})
            ex = np.array(ind.get("exact", []))
            row["indist_exact"] = tci(ex)
            row["indist_exact_seeds"] = ex.tolist()
            row["indist_ll"] = tci(ind.get("logloss", []))
            row["n"] = len(ex)
            row["n_ge_099"] = int((ex >= SOLVE).sum())
            row["n_lt_09"] = int((ex < 0.9).sum())
            row["collapsed_lt_05"] = int((ex < 0.5).sum())
            row["flops_to_005"] = flops_to(r)
            row["steps_to_005"] = steps_to(r)
            row["steps"] = r["steps"]
            row["wall_train"] = r["wall_train"]
            if "val" in r:
                c = frozen_c(r, kind)
                row["c"] = c
                row["val_exact"] = float(np.mean(ex_at(r["val"][0], kind, c)))
                row["test"] = {it["size"]: tci(ex_at(it, kind, c)) for it in r.get("test", [])}
                row["test_seeds"] = {it["size"]: ex_at(it, kind, c).tolist() for it in r.get("test", [])}
                if r.get("test"):
                    last = r["test"][-1]
                    row["fp_exact_largest"] = tci(last.get("fp_exact", []))
                    row["c4_largest"] = tci(ex_at(last, kind, 4))
                    row["oracle_T_largest"] = tci(np.array(last["exact"]).max(1))
                    if "rule" in last:
                        rec = [bool(z.get("recovered")) for z in last["rule"]]
                        sol = (ex_at(last, kind, c) >= SOLVE).tolist()
                        row["rule_recovered"] = rec
                        row["rule_agree_with_solve"] = float(np.mean([a == b for a, b in zip(rec, sol)]))
                    if len(last.get("cs", CS)) > len(CS):  # P-eq (DT) results carry no cs list
                        row["c_extra_largest"] = {str(cc): tci(ex_at(last, kind, cc)) for cc in last["cs"][len(CS):]}
            if "diag" in ind:
                dg = ind["diag"]
                row["diag"] = {k: np.array([[e[k] for e in lv] for lv in dg]).mean(1).tolist()
                               for k in ("nmi_g", "states", "kl", "lam_abs", "Hq")}
            cv = curve_ll(r)
            if cv is not None:
                charts["curves"].setdefault(key, {})[arm] = {"frac": cv[0].tolist(), "ll": cv[1].tolist(),
                                                             "exact": cv[3].tolist(),
                                                             "wall": [c.get("wall") for c in r["curve"]]}
            C[arm] = row
        A["cells"][key] = C
        if D == PRIM.get(kind) and mod == 2:
            charts["extrap"][key] = {a: {str(s): v for s, v in row.get("test", {}).items()} for a, row in C.items()}

    # ------------- S1 / F1 per task
    for kind in ("prefix", "maze"):
        T = {}
        pk = f"{kind}_D{PRIM[kind]}"
        C = A["cells"].get(pk, {})
        bl = C.get("Bl")
        if not bl:
            continue
        others = {"prefix": [f"prefix_D32"], "maze": [f"maze_D16"]}[kind]
        s1_acc = {"primary": {"n_ge_099": bl["n_ge_099"], "n": bl["n"], "pass": bl["n_ge_099"] >= 8}}
        for o in others:
            b = A["cells"].get(o, {}).get("Bl")
            if b:
                s1_acc[o] = {"n_ge_099": b["n_ge_099"], "n": b["n"], "pass": b["n_ge_099"] >= 4}
        T["S1_accuracy"] = s1_acc
        deltas, ratios, wdeltas, wratios, collapse32 = {}, {}, {}, {}, None
        for Dd in (8, 16, 32):
            ck = f"{kind}_D{Dd}"
            cc = A["cells"].get(ck, {})
            if "Bl" not in cc:
                continue
            bps = [a for a in BP if a in cc]
            if not bps:
                continue
            best = max(bps, key=lambda a: (cc[a].get("val_exact") or 0, -(cc[a]["indist_ll"]["mean"] or 9)))
            d = cc["Bl"]["indist_ll"]["mean"] - cc[best]["indist_ll"]["mean"]
            deltas[Dd] = {"delta": d, "best": best, "best_indist_exact": cc[best]["indist_exact"]["mean"],
                          "bl_ll": cc["Bl"]["indist_ll"], "best_ll": cc[best]["indist_ll"]}
            ratios[Dd] = ratio(cc["Bl"]["flops_to_005"], cc[best]["flops_to_005"])
            if Dd == 32:
                collapse32 = all(cc[a]["collapsed_lt_05"] >= cc[a]["n"] / 2 for a in bps)
            # equal wall-clock (timing runs: s/step at E=1)
            tb, tbest = timing.get(("Bl", kind, Dd)), timing.get((best, kind, Dd))
            rb = cells[(kind, Dd, 2)]["Bl"]
            rbest = cells[(kind, Dd, 2)][best]
            if tb and tbest and curve_ll(rbest) is not None:
                wall_bl = rb["steps"] * tb
                llb, clamped = ll_at_step(rbest, wall_bl / tbest)
                wdeltas[Dd] = {"delta": cc["Bl"]["indist_ll"]["mean"] - llb, "best_clamped_at_end": clamped,
                               "s_per_step_bl": tb, "s_per_step_best": tbest}
                sb, sbest = cc["Bl"]["steps_to_005"], cc[best]["steps_to_005"]
                wratios[Dd] = ratio(None if sb is None else sb * tb, None if sbest is None else sbest * tbest)
        T["delta"] = deltas
        T["flops_ratio_to_005"] = ratios
        T["wall_delta"] = wdeltas
        T["wall_ratio_to_005"] = wratios
        charts["delta"][kind] = {str(k): v["delta"] for k, v in deltas.items()}

        def growth_ok(dl, rt):
            if 8 not in dl or 32 not in dl:
                return None
            c1 = dl[32]["delta"] <= dl[8]["delta"] + 0.02
            c2 = (rt.get(32) is not None and rt.get(8) not in (None, 0.0) and not math.isinf(rt[8])
                  and rt[32] <= 1.5 * rt[8])
            return bool(c1 or c2)
        g = growth_ok(deltas, ratios)
        gw = growth_ok(wdeltas, wratios)
        T["growth"] = "not evaluable" if collapse32 else g
        T["growth_wall"] = gw
        s1 = bool(all(v["pass"] for v in s1_acc.values()) and g is True and not collapse32)
        T["S1"] = s1
        T["S1_label"] = ("FLOP-only" if (s1 and gw is False) else None)
        # F1
        bac = C.get("Bl-ac")
        f1a = bl["n_lt_09"] >= 6 * bl["n"] / 10 and (bac is not None and bac["n_lt_09"] >= 6 * bac["n"] / 10)
        dl = [deltas[k]["delta"] for k in (8, 16, 32) if k in deltas]
        f1b = len(dl) == 3 and dl[0] < dl[1] < dl[2] and dl[2] > 0.05 and not collapse32
        T["F1_task"] = {"acc_arm": bool(f1a), "delta_monotone": bool(f1b), "fires": bool(f1a or f1b)}

        # ------------- S2 / F2 / F3 at the primary cell
        largest = max(bl["test"], key=lambda s: int(s)) if bl.get("test") else None
        if largest is not None:
            base = [a for a in S2_BASE if a in C and C[a].get("val_exact") is not None]
            bestb = max(base, key=lambda a: C[a]["val_exact"]) if base else None
            T["S2_best_baseline"] = bestb
            if bestb:
                blv = np.array(bl["test_seeds"][largest])
                bv = np.array(C[bestb]["test_seeds"][largest])
                ks, kb = int((blv >= SOLVE).sum()), int((bv >= SOLVE).sum())
                fisher = stats.fisher_exact([[ks, len(blv) - ks], [kb, len(bv) - kb]], alternative="greater")
                n = min(len(blv), len(bv))
                if bestb != "P-pub" and n >= 2:
                    dd = blv[:n] - bv[:n]
                    pair = tci(dd)
                else:
                    pair = {"welch_diff": float(blv.mean() - bv.mean())}
                cplo = cp(ks, len(blv))[0]
                diff = float(blv.mean() - bv.mean())
                use_cp = bestb == "P-pub" and len(bv) < 5
                s2_main = diff >= 0.10 and ((cplo > bv.mean() + 0.10) if use_cp else (ks > kb and fisher.pvalue < 0.05))
                T["S2_primary"] = {"size": largest, "bl_mean": float(blv.mean()), "base_mean": float(bv.mean()),
                                   "diff": diff, "bl_solves": ks, "bl_n": len(blv), "base_solves": kb,
                                   "base_n": len(bv), "fisher_p": float(fisher.pvalue), "paired": pair,
                                   "bl_cp95": cp(ks, len(blv)), "cp_rule": use_cp, "pass_primary": bool(s2_main)}
            # F2 on this task: every test size, best baseline >= Bl - 0.02
            f2 = True
            for s in bl["test"]:
                bm = bl["test"][s]["mean"]
                bb = max((C[a]["test"][s]["mean"] for a in base if s in C[a].get("test", {})), default=-1)
                f2 &= bb >= bm - 0.02
            T["F2_task"] = bool(f2)
            # F3 attribution at the primary cell (extrapolation metric; indist also reported)
            bm = bl["test"][largest]["mean"]
            near = {a: C[a]["test"][largest]["mean"] for a in A_ALL + ("Bl-local", "Bl-shuffle")
                    if a in C and largest in C[a].get("test", {})}
            T["F3_near"] = {a: bool(v >= bm - 0.05) for a, v in near.items()}
            T["F3"] = bool(any(T["F3_near"].values()))
            near_ind = {a: C[a]["indist_exact"]["mean"] for a in A_ALL + ("Bl-local", "Bl-shuffle") if a in C}
            T["F3_indist_near"] = {a: bool(v >= bl["indist_exact"]["mean"] - 0.05) for a, v in near_ind.items()}
            T["other_task_best_gap"] = None
        # G0 arm-C part
        cC = C.get("C")
        T["G0_C"] = None if cC is None else {"indist_exact": cC["indist_exact"], "pass": cC["indist_exact"]["mean"] < 0.5}
        A["tasks"][kind] = T
    # S2 cross-task clause: on the other task Bl is not worse than its best baseline by > 0.10
    for kind, other in (("prefix", "maze"), ("maze", "prefix")):
        if kind in A["tasks"] and other in A["tasks"] and "S2_primary" in A["tasks"][other]:
            o = A["tasks"][other]["S2_primary"]
            ok = o["bl_mean"] >= o["base_mean"] - 0.10
            A["tasks"][kind]["other_task_best_gap"] = o["bl_mean"] - o["base_mean"]
            if "S2_primary" in A["tasks"][kind]:
                A["tasks"][kind]["S2"] = bool(A["tasks"][kind]["S2_primary"]["pass_primary"] and ok)
    A["F1"] = all(A["tasks"].get(k, {}).get("F1_task", {}).get("fires", False) for k in ("prefix", "maze"))
    A["F2"] = all(A["tasks"].get(k, {}).get("F2_task", False) for k in ("prefix", "maze"))
    A["timing_s_per_step"] = {f"{a}|{k}|{d}": v for (a, k, d), v in sorted(timing.items())}
    mem = os.path.join(OUT, "vm_out", "results", "memory.json")
    if os.path.exists(mem):
        charts["memory"] = json.load(open(mem))["rows"]
    json.dump(rnd(A), open(os.path.join(OUT, "analysis.json"), "w"), indent=1, default=str)
    json.dump(rnd(charts), open(os.path.join(OUT, "charts.json"), "w"), default=str)
    for kind, T in A["tasks"].items():
        print("==", kind)
        for k, v in T.items():
            print(" ", k, json.dumps(rnd(v, 3), default=str)[:400])
    print("F1", A["F1"], "F2", A["F2"], "gates", gates)
    print("G5", json.dumps(rnd(A.get("G5"), 4), default=str))


if __name__ == "__main__":
    {"pilot": pilot, "freeze": freeze, "report": report}[sys.argv[1]]()
