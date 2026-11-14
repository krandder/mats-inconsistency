"""Aggregate P7 runs into results/summary.json and charts/*.json (means, 95% t-intervals over seeds)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ci, spearman, auroc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MAZE_ARMS = ("A", "B", "Bl", "C", "D", "Dl")
PREFIX_ARMS = ("A", "B", "Bl", "Bx", "C", "D", "Dl")
CH = os.path.join(HERE, "charts")


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        r = json.load(open(f))
        a = r["args"]
        key = (a["arm"] + a.get("tag", ""), a.get("D", a.get("d")))
        out.setdefault(key, []).append(r)
    return out


def agg(runs, get):
    vals = []
    for r in runs:
        try:
            v = get(r)
        except (KeyError, IndexError, TypeError):
            v = None
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    return ci(vals)


def fmt(c, nd=3):
    if c is None:
        return "n/a"
    if c["ci95"] is None:
        return f"{c['mean']:.{nd}f}"
    return f"{c['mean']:.{nd}f}±{c['ci95']:.{nd}f}"


def maze_summary(sub="maze"):
    runs = load(f"{sub}/maze_*_s*.json")
    S = {}
    for (arm, D), rs in sorted(runs.items()):
        e = {"n_seeds": len(rs)}
        for size in ("9", "15", "21"):
            e[f"logloss_{size}"] = agg(rs, lambda r: r["eval"][size]["logloss_open"])
            e[f"acc_{size}"] = agg(rs, lambda r: r["eval"][size]["acc_open"])
            e[f"greedy_acc_{size}"] = agg(rs, lambda r: r["eval"][size]["greedy_acc_open"])
            e[f"acc_boundary_{size}"] = agg(rs, lambda r: r["eval"][size]["acc_boundary"])
        e["kl_level"] = [agg(rs, lambda r, k=k: r["eval"]["9"]["kl_level"][k]) for k in range(D)]
        e["align"] = []
        for k in range(D):
            lv = {"level": k + 1, "g_radius": rs[0]["eval"]["9"]["align"][k]["g_radius"]}
            for m in ("nmi_g", "probe_g", "probe_g_base", "extra_g", "nmi_y", "nmi_dist", "probe_dist",
                      "probe_dist_base", "nmi_onpath", "kl_auroc_err", "probe_err_rate", "n_states"):
                lv[m] = agg(rs, lambda r, k=k, m=m: r["eval"]["9"]["align"][k][m])
            e["align"].append(lv)
        S[f"{arm}_D{D}"] = e
    return S


def maze_base_rates():
    """Majority accuracy and marginal log loss on open cells of each test set (the no-information floor)."""
    sys.path.insert(0, HERE)
    import maze  # noqa
    out = {}
    for size, num in ((9, 300), (15, 120), (21, 60)):
        ts = maze.make_set(num, size, seed=777 + size)
        om = ts["open"].reshape(-1)
        y = ts["y"].reshape(-1).numpy()[om]
        p = float(y.mean())
        out[str(size)] = {"frac_y1": p, "majority_acc": max(p, 1 - p),
                          "marginal_logloss": float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))}
    return out


def maze_voc():
    """Does the per-cell hindsight gap of a shallow chain predict where a deeper chain helps?"""
    sys.path.insert(0, HERE)
    import maze  # noqa
    ts = maze.make_set(300, 9, seed=777 + 9)
    openm = ts["open"].reshape(-1)
    dist = ts["dist"].reshape(-1)[openm]
    out = {}
    for arm in MAZE_ARMS:
        for lo, hi in ((1, 4), (1, 2), (2, 4)):
            rows = []
            for s in range(10):
                f1 = os.path.join(RES, "maze", f"maze_{arm}_D{lo}_s{s}_cells.npz")
                f2 = os.path.join(RES, "maze", f"maze_{arm}_D{hi}_s{s}_cells.npz")
                if not (os.path.exists(f1) and os.path.exists(f2)):
                    continue
                a, b = np.load(f1), np.load(f2)
                gain = (a["ll"] - b["ll"])[openm]
                big = gain > 0.1
                rows.append({
                    "spearman_kl": spearman(a["kl_top"][openm], gain),
                    "spearman_ll": spearman(a["ll"][openm], gain),
                    "spearman_ent": spearman(a["ent"][openm], gain),
                    "auroc_kl": auroc(a["kl_top"][openm], big),
                    "auroc_ll": auroc(a["ll"][openm], big),
                    "auroc_ent": auroc(a["ent"][openm], big),
                    "frac_big": float(big.mean()),
                    "mean_gain": float(gain.mean()),
                    "gain_by_dist": [float(gain[np.minimum(dist, 6) == d].mean()) if (np.minimum(dist, 6) == d).any() else None for d in range(7)],
                    "kl_by_dist": [float(a["kl_top"][openm][np.minimum(dist, 6) == d].mean()) if (np.minimum(dist, 6) == d).any() else None for d in range(7)],
                })
            if rows:
                e = {k: ci([r[k] for r in rows]) for k in rows[0] if not k.endswith("by_dist")}
                e["gain_by_dist"] = [ci([r["gain_by_dist"][d] for r in rows if r["gain_by_dist"][d] is not None]) for d in range(7)]
                e["kl_by_dist"] = [ci([r["kl_by_dist"][d] for r in rows if r["kl_by_dist"][d] is not None]) for d in range(7)]
                e["n_seeds"] = len(rows)
                out[f"{arm}_D{lo}->D{hi}"] = e
    return out


def prefix_summary(sub="prefix"):
    runs = load(f"{sub}/prefix_*_s*.json")
    S = {}
    for (arm, d), rs in sorted(runs.items()):
        e = {"n_seeds": len(rs)}
        for L in ("16", "64", "256"):
            for m in ("logloss_obs", "acc_obs", "greedy_acc_obs", "acc_obs_last_quarter", "logloss", "acc"):
                e[f"{m}_{L}"] = agg(rs, lambda r, L=L, m=m: r["eval"][L][m])
        for m in ("kl_mean", "kl_obs", "kl_unobs"):
            e[m] = agg(rs, lambda r, m=m: r["eval"]["16"][m])
        for m in ("nmi_g", "probe_g", "probe_g_base", "extra_g", "nmi_y", "n_states"):
            e[m] = agg(rs, lambda r, m=m: r["eval"]["16"]["align"][m])
            e[m + "_obs"] = agg(rs, lambda r, m=m: r["eval"]["16"]["align_obs"][m])
        e["solved_frac"] = ci([1.0 if r["eval"]["256"]["acc_obs"] > 0.99 else 0.0 for r in rs])
        e["per_seed_acc_obs256"] = [round(r["eval"]["256"]["acc_obs"], 4) for r in rs]
        S[f"{arm}_d{d}"] = e
    return S


def chart_json(name, title, xlabel, ylabel, series, note=""):
    os.makedirs(CH, exist_ok=True)
    with open(os.path.join(CH, name + ".json"), "w") as f:
        json.dump({"title": title, "x_label": xlabel, "y_label": ylabel, "series": series, "note": note}, f, indent=1)


def main():
    maze = maze_summary()
    pref = prefix_summary()
    voc = maze_voc()
    base = maze_base_rates()
    abl = {"maze": maze_summary("ablations"), "prefix": prefix_summary("ablations")}
    summ = {"maze": maze, "maze_base_rates": base, "prefix": pref, "maze_value_of_computation": voc, "ablations": abl}
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    # charts
    ser = []
    for arm in MAZE_ARMS:
        pts = [{"x": D, "y": maze[f"{arm}_D{D}"]["logloss_9"]["mean"], "ci95": maze[f"{arm}_D{D}"]["logloss_9"]["ci95"]}
               for D in (1, 2, 4) if f"{arm}_D{D}" in maze]
        ser.append({"name": arm, "points": pts})
    chart_json("maze_logloss", "Maze: test log loss on open cells (9x9) by chain depth", "chain depth D", "log loss (nats)", ser)
    ser = []
    for arm in MAZE_ARMS:
        pts = []
        for D in (2, 4):
            k = f"{arm}_D{D}"
            if k in maze:
                for lv in maze[k]["align"][:-1]:
                    pts.append({"x": f"D{D} L{lv['level']}", "y": lv["nmi_g"]["mean"] if lv["nmi_g"] else None,
                                "ci95": lv["nmi_g"]["ci95"] if lv["nmi_g"] else None})
        ser.append({"name": arm, "points": pts})
    chart_json("maze_alignment", "Maze: I(z_k; reach within k*rho)/H at intermediate levels", "depth, level", "normalised MI", ser)
    ser = []
    for arm in PREFIX_ARMS:
        pts = [{"x": d, "y": pref[f"{arm}_d{d}"]["acc_obs_256"]["mean"], "ci95": pref[f"{arm}_d{d}"]["acc_obs_256"]["ci95"]}
               for d in (1, 2, 4) if f"{arm}_d{d}" in pref]
        ser.append({"name": arm, "points": pts})
    chart_json("prefix_acc256", "Prefix sum mod 3: accuracy on revealed positions at length 256 (trained at 16)", "outputs revealed every d steps", "accuracy (majority 0.667)", ser)
    ser = []
    for arm in PREFIX_ARMS:
        pts = [{"x": d, "y": pref[f"{arm}_d{d}"]["nmi_g"]["mean"], "ci95": pref[f"{arm}_d{d}"]["nmi_g"]["ci95"]}
               for d in (1, 2, 4) if f"{arm}_d{d}" in pref]
        ser.append({"name": arm, "points": pts})
    chart_json("prefix_alignment", "Prefix sum: I(z; running sum mod 3)/H", "outputs revealed every d steps", "normalised MI", ser)
    if voc:
        ser = []
        for k, e in voc.items():
            if k.endswith("D1->D4"):
                ser.append({"name": k, "points": [{"x": m, "y": e[m]["mean"], "ci95": e[m]["ci95"]} for m in ("spearman_kl", "spearman_ll", "spearman_ent")]})
        chart_json("maze_voc", "Maze: does the D=1 hindsight gap predict where D=4 helps? (Spearman per cell)", "signal", "Spearman with log-loss gain", ser)
    ser = []
    for fam, keys in (("q-form", ("B_D2", "B_long_D2", "D_viewsonly_D2", "D_rateonly_D2", "D_D2")),
                      ("likelihood form", ("Bl_D2", "Bl_long_D2", "Dl_viewsonly_D2", "Dl_rateonly_D2", "Dl_lam001_D2", "Dl_D2"))):
        pts = []
        for k in keys:
            e = maze.get(k) or abl["maze"].get(k)
            if e:
                pts.append({"x": k, "y": e["logloss_9"]["mean"], "ci95": e["logloss_9"]["ci95"]})
        ser.append({"name": fam, "points": pts})
    chart_json("maze_ablation_D2", "Maze depth 2: which part of arm D causes collapse? (test log loss, 9x9)", "variant",
               "log loss (nats; marginal 0.594)", ser,
               note="viewsonly = lam 0; rateonly = views off; lam001 = rate 0.01; long = 6000 steps")
    # print compact tables
    print("MAZE base rates", base)
    print("MAZE  arm_D | ll9 | acc9 | acc21 | greedy9 | kl per level")
    for k, e in maze.items():
        print(k, fmt(e["logloss_9"]), fmt(e["acc_9"]), fmt(e["acc_21"]), fmt(e["greedy_acc_9"]), [fmt(c, 2) for c in e["kl_level"]])
        for lv in e["align"]:
            print("   L", lv["level"], "r", lv["g_radius"], "nmi_g", fmt(lv["nmi_g"], 2), "probe", fmt(lv["probe_g"], 2), "/", fmt(lv["probe_g_base"], 2),
                  "extra_g", fmt(lv["extra_g"], 2), "nmi_y", fmt(lv["nmi_y"], 2), "nmi_dist", fmt(lv["nmi_dist"], 2), "kl_auroc_err", fmt(lv["kl_auroc_err"], 2), "states", fmt(lv["n_states"], 1))
    print("PREFIX arm_d | ll_obs16 | acc_obs16 | acc_obs64 | acc_obs256 | acc_all16 | nmi_s | extra_s | nmi_y | states | kl")
    for k, e in pref.items():
        print(k, fmt(e["logloss_obs_16"]), fmt(e["acc_obs_16"]), fmt(e["acc_obs_64"]), fmt(e["acc_obs_256"]), fmt(e["acc_16"]), fmt(e["nmi_g"], 2), fmt(e["extra_g"], 2), fmt(e["nmi_y"], 2), fmt(e["n_states"], 1), fmt(e["kl_mean"], 2), e["per_seed_acc_obs256"])
    print("ABLATIONS (tagged runs, results/ablations)")
    for k, e in abl["maze"].items():
        print(k, "n", e["n_seeds"], "ll9", fmt(e["logloss_9"]), "acc9", fmt(e["acc_9"]), "acc21", fmt(e["acc_21"]), "states", [fmt(lv["n_states"], 1) for lv in e["align"]], "nmi_g", [fmt(lv["nmi_g"], 2) for lv in e["align"]])
    for k, e in abl["prefix"].items():
        print(k, "n", e["n_seeds"], "ll_obs16", fmt(e["logloss_obs_16"]), "acc_obs256", fmt(e["acc_obs_256"]), "nmi_s", fmt(e["nmi_g"], 2), "states", fmt(e["n_states"], 1), e["per_seed_acc_obs256"])
    print("VOC")
    for k, e in voc.items():
        print(k, {m: fmt(e[m], 2) for m in ("spearman_kl", "spearman_ll", "spearman_ent", "auroc_kl", "auroc_ll", "auroc_ent", "frac_big", "mean_gain")})


if __name__ == "__main__":
    main()
