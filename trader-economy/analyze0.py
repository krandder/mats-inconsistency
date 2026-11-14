"""Summary and charts for Experiment 0 (results/exp0_all_rows.json)."""
import json
import os

import numpy as np

from core import mean_ci

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results/exp0_all_rows.json")))
C = 3
NOISE = {"weight": [j * C + c for j in (6, 7) for c in range(C)], "column": [6, 7], "row": []}
SRC = "trader-economy/results/exp0_summary.json"


def key(c):
    return (c["gran"], c["arm"], c["cost"], c["seed"])


bp = {key(r["cfg"])[:1] + key(r["cfg"])[2:]: np.array(r["W"]) for r in rows if r["cfg"]["arm"] == "backprop"}
out = {}
for gran in ("weight", "column", "row"):
    for arm in ("backprop", "loo", "msr1", "shap"):
        for cost in (0.0, 0.01, 0.05):
            rs = [r for r in rows if (r["cfg"]["gran"], r["cfg"]["arm"], r["cfg"]["cost"]) == (gran, arm, cost)]
            if not rs:
                continue
            s = {}
            for k in ("ll", "ll_bayes", "sp_pay_truth", "sp_zshap_truth", "sp_ig_truth", "sp_ig_zshap", "budget_ratio",
                      "alive_end", "active_end"):
                v = [r[k] for r in rs if r[k] == r[k]]
                s[k] = mean_ci(v) if v else None
            s["deaths"] = mean_ci([sum(r["deaths"]) for r in rs])
            s["noise_deaths"] = mean_ci([sum(r["deaths"][i] for i in NOISE[gran]) for r in rs]) if NOISE[gran] else None
            s["useful_deaths"] = mean_ci([sum(d for i, d in enumerate(r["deaths"]) if i not in NOISE[gran]) for r in rs])
            s["rel_w_diff_vs_backprop"] = mean_ci([float(np.linalg.norm(np.array(r["W"]) - bp[(gran, cost, r["cfg"]["seed"])])
                                                         / np.linalg.norm(bp[(gran, cost, r["cfg"]["seed"])])) for r in rs])
            F = np.array([r["feat_norm"] for r in rs])
            s["feat_norm"] = [mean_ci(F[:, j]) for j in range(F.shape[1])]
            s["copy_over_orig_weight"] = mean_ci(F[:, 2] / F[:, 0])
            s["noise_weight_abs"] = mean_ci(F[:, 6] + F[:, 7])
            if gran == "column":
                Pm = np.array([r["pay"] for r in rs])
                s["pay"] = [mean_ci(Pm[:, j]) for j in range(Pm.shape[1])]
                s["copy_over_orig_pay"] = mean_ci(Pm[:, 2] / Pm[:, 0])
                Z = np.array([r["zshap"] for r in rs])
                s["zshap"] = [mean_ci(Z[:, j]) for j in range(Z.shape[1])]
                I = np.array([r["ig"] for r in rs])
                s["ig"] = [mean_ci(I[:, j]) for j in range(I.shape[1])]
                s["truth"] = rs[0]["truth"]
            out[f"{gran}|{arm}|{cost}"] = s
json.dump(out, open(os.path.join(HERE, "results/exp0_summary.json"), "w"), indent=1, default=float)

os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
LBL = {"backprop": "Backprop (global loss)", "loo": "Own pay, leave-one-out", "msr1": "Own pay, MSR one order",
       "shap": "Own pay, Shapley (16 orders)"}


def bars(cid, title, unit, items, note=""):
    json.dump(dict(id=cid, kind="bar", title=title, unit=unit, source=SRC, note=note,
                   bars=[dict(label=l, value=round(v[0], 4), ci=round(v[1], 4), highlight=h) for l, v, h in items]),
              open(os.path.join(HERE, "charts", cid + ".json"), "w"), indent=1)


bars("exp0_logloss", "Exp 0: test log loss by training rule, column agents, cost 0.01 (Bayes = %.3f)"
     % out["column|backprop|0.01"]["ll_bayes"][0], "nats",
     [(LBL[a], out[f"column|{a}|0.01"]["ll"], a == "shap") for a in LBL])
json.dump(dict(id="exp0_weights", kind="grouped_bar", title="Exp 0: |weights| out of each input, column agents, cost 0.01",
               source=SRC, groups=["x0 strong", "x1 medium", "x2 copy of x0", "x3 comp", "x4 comp", "x5 weak", "x6 noise", "x7 noise"],
               series={LBL[a]: [round(v[0], 3) for v in out[f"column|{a}|0.01"]["feat_norm"]] for a in LBL}),
          open(os.path.join(HERE, "charts", "exp0_weights.json"), "w"), indent=1)
json.dump(dict(id="exp0_pay_vs_truth", kind="grouped_bar",
               title="Exp 0: column agents at the backprop solution, nats/example: true value vs Shapley vs IG pay",
               source=SRC, groups=["x0 strong", "x1 medium", "x2 copy of x0", "x3 comp", "x4 comp", "x5 weak", "x6 noise", "x7 noise"],
               series={"true value (refit Shapley)": [round(v, 4) for v in out["column|backprop|0.01"]["truth"]],
                       "Shapley, zero-out": [round(v[0], 4) for v in out["column|backprop|0.01"]["zshap"]],
                       "Aumann-Shapley (IG)": [round(v[0], 4) for v in out["column|backprop|0.01"]["ig"]]}),
          open(os.path.join(HERE, "charts", "exp0_pay_vs_truth.json"), "w"), indent=1)
for k, s in out.items():
    f = lambda v: "-" if v is None else f"{v[0]:.3f}±{v[1]:.3f}"
    print(k, "ll", f(s["ll"]), "sp_pay", f(s["sp_pay_truth"]), "sp_z", f(s["sp_zshap_truth"]), "sp_ig", f(s["sp_ig_truth"]),
          "bud", f(s["budget_ratio"]), "dW", f(s["rel_w_diff_vs_backprop"]), "deaths", f(s["deaths"]), "noiseD", f(s["noise_deaths"]),
          "usefulD", f(s["useful_deaths"]), "copy/orig", f(s["copy_over_orig_weight"]), "comp", f(s["feat_norm"][3]),
          "noiseW", f(s["noise_weight_abs"]), "active", f(s["active_end"]))
