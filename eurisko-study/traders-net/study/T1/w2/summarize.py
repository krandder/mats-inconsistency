"""Aggregate results/delay*.json and results/li.json into mean ± sd tables (3 seeds).
Writes results/summary_delay.csv, results/summary_delay_stable.csv, results/summary_li.csv,
results/li_curves.csv and results/summary.json."""
import csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
F8 = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or"]


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def delay(name):
    p = os.path.join(RES, name + ".json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p)); runs = r["runs"]
    rows, glob = [], {}
    glob["tel_tan_max"] = max(x["tel_tan_max"] for x in runs)
    glob["tel_ex_max"] = max(x["tel_ex_max"] for x in runs)
    glob["norm_rel_err_max"] = max(x["norm_rel_err"] for x in runs)
    glob["mrp"] = r.get("mrp")
    for sh in (False, True):
        for D in sorted({x["D"] for x in runs}):
            rs = [x for x in runs if x["D"] == D and x["shuffle"] == sh]
            m, s = ms([x["stale_ratio"] for x in rs])
            glob[f"stale_ratio_D{D}_shuf{int(sh)}"] = [m, s]
            glob[f"W_trade_D{D}_shuf{int(sh)}"] = ms([x["W_trade"] for x in rs])
            glob[f"S_stale_D{D}_shuf{int(sh)}"] = ms([x["S_stale"] for x in rs])
            for f in F8:
                row = {"shuffle": int(sh), "D": D, "family": f, "n_seeds": len(rs)}
                for k in ["ex_mean", "gap_mean", "tan_mean", "self_tan_mean", "self_tan_se", "self_ex_mean", "kl_mean",
                          "mtm_bias_ex", "corr_mid_final"]:
                    vals = [x["fam"][f].get(k, np.nan) for x in rs]
                    m, s = ms(vals)
                    row[k] = round(m, 5); row[k + "_sd"] = round(s, 5)
                rows.append(row)
    with open(os.path.join(RES, f"summary_{name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return glob


def li():
    p = os.path.join(RES, "li.json")
    r = json.load(open(p)); runs = r["runs"]
    arms = list(dict.fromkeys(x["arm"] for x in runs))
    rows, curves, glob = [], [], {}
    for arm in arms:
        rs = [x for x in runs if x["arm"] == arm]
        g = {"n_seeds": len(rs), "max_conservation_err": max(x["max_conservation_err"] for x in rs)}
        if "bound" in rs[0]:
            g["bound"] = rs[0]["bound"]
            g["max_member_traded_E"] = max(x["max_member_traded_E"] for x in rs)
            g["bound_holds_all"] = all(x["bound_holds"] for x in rs)
            if "max_reader_clearing_E" in rs[0]:
                g["max_reader_clearing_E_per_seed"] = [round(x["max_reader_clearing_E"], 3) for x in rs]
        glob[arm] = g
        for f in F8 + ["diag"]:
            row = {"arm": arm, "family": f, "n_seeds": len(rs)}
            for k in ["E_C_end", "slope1", "slope2", "E_pat4_end", "E_pat8_end", "E_calib_max_end", "pat4_slope1",
                      "pat4_slope2", "calib_slope2", "phi_best", "market_logloss"]:
                m, s = ms([x["fam"][f][k] for x in rs])
                row[k] = round(m, 5); row[k + "_sd"] = round(s, 5)
            row["slope2_over_slope1"] = round(row["slope2"] / row["slope1"], 4) if abs(row["slope1"]) > 1e-9 else None
            row["slope2_pos_seeds"] = sum(1 for x in rs if x["fam"][f]["slope2"] > 0)
            rows.append(row)
            for x in rs:
                for t, e in zip(x["fam"][f]["curve_t"], x["fam"][f]["curve_EC"]):
                    curves.append({"arm": arm, "family": f, "seed": x["seed"], "t": t, "E_C": e})
    with open(os.path.join(RES, "summary_li.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(os.path.join(RES, "li_curves.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    return glob


if __name__ == "__main__":
    out = {"delay": delay("delay"), "delay_stable": delay("delay_stable"), "li": li()}
    json.dump(out, open(os.path.join(RES, "summary.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])
