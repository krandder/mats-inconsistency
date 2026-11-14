"""Summarise three_rules.py outputs against w4's exact ablation. Writes
lead/results/three_summary_{tangent,lmsr}.csv (per seed x step rows) and three_agg.csv (mean +- sd
over seeds of the per-seed checkpoint means)."""
import csv, glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
W4 = os.path.join(HERE, "..", "w4", "results")
sp = lambda a, b: float(spearmanr(np.ravel(a), np.ravel(b)).correlation)


def one(tag):
    rows = []
    for fn in sorted(glob.glob(f"{RES}/{tag}-s*-step*.json")):
        d = json.load(open(fn)); s, st = d["seed"], d["step"]
        w4 = json.load(open(f"{W4}/s{s}-step{st}.json"))
        fx = f"{RES}/w4fix-s{s}-step{st}.json"   # w4 rules re-aggregated without the double 1/B (lead/w4_fix.py)
        if os.path.exists(fx):
            w4 = dict(w4, rules=json.load(open(fx))["rules"])
        ah = np.array(w4["ablation"]["head"]).ravel(); an = np.array(w4["ablation"]["neuron"]).ravel()
        top2 = np.argsort(-ah)[:2]
        r3, rc = d["rules"]["three16"], d["rules"]["cp"]
        row = dict(seed=s, step=st, R=r3["R"])
        if r3.get("logscore_gain") not in (None, 0.0):
            row.update(logscore_gain=r3["logscore_gain"], maker=r3["maker"])
        for rn in ("three16", "three64", "cp"):
            r = d["rules"][rn]
            row[f"resid_max_rel_{rn}"] = max(abs(x) for x in r["resid"]) / abs(r["R"])
        row["softmax_defect_rel_three16"] = max(abs(x) for x in r3["sm_defect"]) / abs(r3["R"])
        H = np.array(r3["head"]).ravel(); Hv = np.array(r3["head_value"]).ravel(); Hr = np.array(r3["head_route"]).ravel()
        row["head_split_err_rel"] = float(np.abs(H - Hv - Hr).max() / np.abs(H).max())
        row["route_share_of_abs"] = float(np.abs(Hr).sum() / (np.abs(Hr).sum() + np.abs(Hv).sum()))
        row["heads_route_neg_frac"] = float((Hr < 0).mean())
        row["heads_value_neg_frac"] = float((Hv < 0).mean())
        row["heads_total_neg_frac"] = float((H < 0).mean())
        row["top2_route_neg_frac"] = float((Hr[top2] < 0).mean())
        row["top2_total_neg_frac"] = float((H[top2] < 0).mean())
        row["route_neg_query_frac_mean"] = float(np.mean(r3["route_neg_query_frac"]))
        row["sp_head_three_total"] = sp(H, ah); row["sp_head_three_route"] = sp(Hr, ah); row["sp_head_three_value"] = sp(Hv, ah)
        row["sp_head_cp"] = sp(rc["head"], ah)
        row["sp_head_live_w4"] = sp(w4["rules"]["live"]["head"], ah)
        row["sp_neuron_three"] = sp(r3["neuron"], an); row["sp_neuron_cp"] = sp(rc["neuron"], an)
        row["sp_neuron_live_w4"] = sp(w4["rules"]["live"]["neuron"], an)
        row["sp_neuronL0_three"] = sp(np.array(r3["neuron"])[0], np.array(w4["ablation"]["neuron"])[0])
        row["sp_neuronL0_cp"] = sp(np.array(rc["neuron"])[0], np.array(w4["ablation"]["neuron"])[0])
        row["sp_three_vs_cp_neuron"] = sp(r3["neuron"], rc["neuron"]); row["sp_three_vs_cp_head"] = sp(H, rc["head"])
        row["neuron_neg_frac_three"] = float((np.ravel(r3["neuron"]) < 0).mean())
        row["neuron_neg_frac_cp"] = float((np.ravel(rc["neuron"]) < 0).mean())
        if tag == "three" and os.path.exists(fx):  # my cp recompute vs w4's cp re-aggregated (tangent, same eval rows)
            a, b = np.ravel(rc["head"]), np.ravel(w4["rules"]["cp"]["head"])
            row["cp_vs_w4fixcp_sp_head"] = sp(a, b)
            row["cp_vs_w4fixcp_sp_neuron"] = sp(rc["neuron"], w4["rules"]["cp"]["neuron"])
            row["cp_over_w4fixcp_head_scale"] = float(np.dot(a, b) / np.dot(b, b))
        rows.append(row)
    return rows


def lockin(tag):
    out = []
    for s in (1, 2, 3):
        a = f"{RES}/{tag}-s{s}-step8000.json"; b = f"{RES}/{tag}-s{s}-step30000.json"
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        A, B = json.load(open(a))["rules"], json.load(open(b))["rules"]
        wa = json.load(open(f"{W4}/s{s}-step8000.json"))["ablation"]; wb = json.load(open(f"{W4}/s{s}-step30000.json"))["ablation"]
        out.append(dict(seed=s, head_three=sp(A["three16"]["head"], B["three16"]["head"]),
                        head_route=sp(A["three16"]["head_route"], B["three16"]["head_route"]),
                        head_cp=sp(A["cp"]["head"], B["cp"]["head"]), head_oracle=sp(wa["head"], wb["head"]),
                        neuron_three=sp(A["three16"]["neuron"], B["three16"]["neuron"]),
                        neuron_cp=sp(A["cp"]["neuron"], B["cp"]["neuron"]), neuron_oracle=sp(wa["neuron"], wb["neuron"])))
    return out


def agg(rows, tag):
    keys = [k for k in rows[0] if k not in ("seed", "step")]
    res = []
    for k in keys:
        per = [np.mean([r[k] for r in rows if r["seed"] == s and k in r]) for s in sorted({r["seed"] for r in rows})]
        res.append(dict(settle=tag, metric=k, mean=float(np.mean(per)), sd=float(np.std(per, ddof=1)) if len(per) > 1 else float("nan"),
                        max_over_ckpts=float(max(r[k] for r in rows if k in r)), n_seeds=len(per)))
    return res


def main():
    allagg = []
    for tag, name in (("three", "tangent"), ("three-lmsr", "lmsr")):
        rows = one(tag)
        if not rows:
            continue
        with open(f"{RES}/three_summary_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99))
            w.writeheader(); w.writerows(rows)
        allagg += agg(rows, name)
        li = lockin(tag)
        if li:
            with open(f"{RES}/three_lockin_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(li[0])); w.writeheader(); w.writerows(li)
            allagg += agg([dict(r, step=0) for r in li], name + "_lockin_8k_vs_30k")
    with open(f"{RES}/three_agg.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allagg[0])); w.writeheader(); w.writerows(allagg)
    for r in allagg:
        print(f"{r['settle']:28s} {r['metric']:32s} {r['mean']:+.4g} ± {r['sd']:.3g}  (max {r['max_over_ckpts']:+.3g}, n={r['n_seeds']})")


if __name__ == "__main__":
    main()
