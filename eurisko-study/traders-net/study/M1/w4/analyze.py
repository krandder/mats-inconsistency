"""Aggregate results/s*-step*.json into CSVs, a summary JSON and one PNG (mean +- sd over seeds)."""
import glob, json, os
import numpy as np
from scipy.stats import spearmanr

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(D, "results")
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s*-step*.json")))]
RULES = ["live", "v1", "cp", "attnlrp"]
rows_res, rows_ops, rows_sp, rows_x = [], [], [], []
for r in runs:
    ab = r["ablation"]; dn = np.array(ab["neuron"]); dh = np.array(ab["head"])
    for rn in RULES:
        b = r["rules"][rn]; Rv = abs(b["R"])
        rows_res.append(dict(seed=r["seed"], step=r["step"], rule=rn, R=b["R"],
                             **{f"resid_cut{k}": v / Rv for k, v in enumerate(b["resid"])},
                             closure_max=max(abs(c) for c in b["closure"]) / Rv))
        for op, v in b["ops"].items():
            rows_ops.append(dict(seed=r["seed"], step=r["step"], rule=rn, op=op, defect_rel=v / Rv))
        nb = np.array(b["neuron"]); hb = np.array(b["head"])
        sp = dict(seed=r["seed"], step=r["step"], rule=rn,
                  neuron_all=spearmanr(nb.ravel(), dn.ravel())[0],
                  neuron_L0=spearmanr(nb[0], dn[0])[0], neuron_L1=spearmanr(nb[1], dn[1])[0],
                  head_all=spearmanr(hb.ravel(), dh.ravel())[0])
        rows_sp.append(sp)
    for i, a in enumerate(RULES):
        for c in RULES[i + 1:]:
            na = np.array(r["rules"][a]["neuron"]).ravel(); nc = np.array(r["rules"][c]["neuron"]).ravel()
            ha = np.array(r["rules"][a]["head"]).ravel(); hc = np.array(r["rules"][c]["head"]).ravel()
            rows_x.append(dict(seed=r["seed"], step=r["step"], a=a, b=c, neuron_spearman=spearmanr(na, nc)[0],
                               head_spearman=spearmanr(ha, hc)[0]))


def write(name, rows):
    keys = list(rows[0].keys())
    with open(os.path.join(D, name), "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in keys) + "\n")


write("resid.csv", rows_res); write("ops.csv", rows_ops); write("ablation_spearman.csv", rows_sp); write("rule_vs_rule.csv", rows_x)


def ms(vals):
    v = np.array(vals, dtype=float)
    return [float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v)]


steps = sorted({r["step"] for r in runs})
summ = {"n_runs": len(runs), "steps": steps, "seeds": sorted({r["seed"] for r in runs}),
        "note": "values are [mean, sd, n] over seeds; residual and defects are relative to |R| on held-out eval",
        "resid_cut0": {}, "closure_max": {}, "ops": {}, "ops_absshare": {}, "spearman_vs_ablation": {}, "rule_vs_rule": {}}
for rn in RULES:
    for st in steps + ["all"]:
        sel = [x for x in rows_res if x["rule"] == rn and (st == "all" or x["step"] == st)]
        summ["resid_cut0"].setdefault(rn, {})[str(st)] = ms([x["resid_cut0"] for x in sel])
        summ["closure_max"].setdefault(rn, {})[str(st)] = float(max(x["closure_max"] for x in sel))
        sp = [x for x in rows_sp if x["rule"] == rn and (st == "all" or x["step"] == st)]
        summ["spearman_vs_ablation"].setdefault(rn, {})[str(st)] = {k: ms([x[k] for x in sp]) for k in ["neuron_all", "neuron_L0", "neuron_L1", "head_all"]}
    ops = sorted({x["op"] for x in rows_ops})
    summ["ops"][rn] = {op: ms([x["defect_rel"] for x in rows_ops if x["rule"] == rn and x["op"] == op]) for op in ops}
    # share of total |defect| by source family, pooled over all runs (per run then averaged)
    fams = {"LN": ["ln1", "ln2", "lnf"], "QK": ["QK"], "softmax": ["softmax"], "AV": ["AV"]}
    shares = {f: [] for f in fams}
    for r in runs:
        o = r["rules"][rn]["ops"]; tot = sum(abs(v) for v in o.values())
        for f, keys in fams.items():
            shares[f].append(sum(abs(v) for k, v in o.items() if k.split(".")[-1] in keys) / tot if tot > 1e-12 else 0.0)
    summ["ops_absshare"][rn] = {f: ms(v) for f, v in shares.items()}
for x in rows_x:
    k = f"{x['a']}~{x['b']}"
    summ["rule_vs_rule"].setdefault(k, {"neuron": [], "head": []})
    summ["rule_vs_rule"][k]["neuron"].append(x["neuron_spearman"]); summ["rule_vs_rule"][k]["head"].append(x["head_spearman"])
summ["rule_vs_rule"] = {k: {kk: ms(vv) for kk, vv in v.items()} for k, v in summ["rule_vs_rule"].items()}
json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ops = ["L0.ln1", "L0.QK", "L0.softmax", "L0.AV", "L0.ln2", "L1.ln1", "L1.QK", "L1.softmax", "L1.AV", "L1.ln2", "lnf"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(len(ops)); w = 0.2
    for i, rn in enumerate(RULES):
        m = [summ["ops"][rn][o][0] for o in ops]; s = [summ["ops"][rn][o][1] for o in ops]
        ax[0].bar(xs + (i - 1.5) * w, m, w, yerr=s, label=rn)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(ops, rotation=45, ha="right"); ax[0].axhline(0, c="k", lw=.5)
    ax[0].set_ylabel("local defect / |R| (mean ± sd, 3 seeds × 4 ckpts)"); ax[0].legend(frameon=False)
    ax[0].set_title("Where the books leak, by operation")
    for rn in RULES:
        m = [summ["spearman_vs_ablation"][rn][str(st)]["neuron_all"][0] for st in steps]
        s = [summ["spearman_vs_ablation"][rn][str(st)]["neuron_all"][1] for st in steps]
        ax[1].errorbar(steps, m, yerr=s, label=rn, marker="o", capsize=3)
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("Spearman(neuron income, zero-ablation ΔBCE)")
    ax[1].set_title("Neuron books vs held-out ablation"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(D, "leak.png"), dpi=110)
except Exception as e:
    print("plot failed", e)
print(json.dumps({k: summ[k] for k in ["resid_cut0", "ops_absshare"]}, indent=0)[:3000])
