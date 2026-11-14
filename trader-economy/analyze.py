"""Summaries and chart JSON from results/econ_all_rows.json."""
import json
import os

import numpy as np

from core import mean_ci, subset_mi_table
from econ import AUDIT_NAMES, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results/econ_all_rows.json")))
SRC = "trader-economy/results/summary.json"


def group(kind, pred):
    return [r for r in rows[kind] if pred(r["cfg"])]


def default1(r):
    return r.get("fee", 0.004) == 0.004 and r.get("compute", 0.004) == 0.004 and "ref_every" not in r


out = {}
# ---- toy 1 arms
K1 = ["ll_out", "ll_lr", "ll_mlp", "ll_bayes", "sp_ideal", "sp_mech", "sp_info", "budget_ratio",
      "evals_per_epoch", "drift_early", "drift_late", "alive_end", "conc_end", "paid_in_per_step"]
t1 = {}
for arm in ("kelly", "loo", "msr1", "msrK", "shapley"):
    t1[arm] = summarize(group("toy1", lambda c: c["arm"] == arm and default1(c) and c.get("fee_mode", "owner") == "owner"), K1)
t1["msrK_burn"] = summarize(group("toy1", lambda c: c["arm"] == "msrK" and c.get("fee_mode") == "burn"), K1)
out["toy1"] = t1

# ---- sweep
sw = {}
for arm in ("msrK", "kelly"):
    for fee in (0.0, 0.004, 0.012, 0.03):
        for comp in (0.0, 0.004, 0.012, 0.03):
            rs = group("toy1", lambda c: c["arm"] == arm and c.get("ref_every") and c["fee"] == fee and c["compute"] == comp)
            s = summarize(rs, ["ll_out", "alive_end", "sp_info", "paid_in_per_step"])
            sw[f"{arm}|fee={fee}|compute={comp}"] = s
out["sweep"] = sw

# ---- audit
au = rows["audit"]
A = {}
for key in ("loo", "msr1", "msr1_rev", "msrK", "shapley", "ideal"):
    M = np.array([r[key] for r in au])
    A[key] = [mean_ci(M[:, i]) for i in range(M.shape[1])]
A["names"] = AUDIT_NAMES
A["full"] = mean_ci([r["full"] for r in au])
A["loo_budget_ratio"] = mean_ci([r["loo_budget_ratio"] for r in au])
A["kerr"] = {K: mean_ci([r["kerr"][K] for r in au]) for K in au[0]["kerr"]}
A["sp_vs_ideal"] = {a: mean_ci([r["sp_vs_ideal"][a] for r in au]) for a in au[0]["sp_vs_ideal"]}
out["audit"] = A

# ---- toy 2
K2 = ["ll_out", "ll_bayes", "sp_ideal", "sp_sys", "earn_useless", "earn_useful", "budget_out", "budget_src",
      "drift_early", "drift_late", "l1_alive_end", "l2_alive_end", "paid_in_per_step"]
t2 = {}
for arm in ("kelly", "royalty", "loo", "msr1", "msrK", "shapley"):
    t2[arm] = summarize(group("toy2", lambda c: c["arm"] == arm), K2)
out["toy2"] = t2
_, info = subset_mi_table()
out["info"] = info
json.dump(out, open(os.path.join(HERE, "results/summary.json"), "w"), indent=1, default=float)

# ---- charts (small bar charts, one message each)
os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)


def bars(cid, title, unit, items, note=""):
    json.dump(dict(id=cid, kind="bar", title=title, unit=unit, source=SRC, note=note,
                   bars=[dict(label=l, value=round(v[0], 4), ci=round(v[1], 4), highlight=h) for l, v, h in items]),
              open(os.path.join(HERE, "charts", cid + ".json"), "w"), indent=1)


LBL = {"kelly": "Kelly market (unregulated)", "loo": "Leave-one-out", "msr1": "MSR, one fixed order",
       "msrK": "MSR, 16 random orders", "shapley": "Exact Shapley", "royalty": "Royalty, equal split"}
bars("toy1_logloss", "Toy 1: output log loss, second half (lower is better)", "nats",
     [("Kelly market", t1["kelly"]["ll_out"], False), ("Regulated, MSR-16", t1["msrK"]["ll_out"], True),
      ("Logistic regression, all features", t1["msrK"]["ll_lr"], False), ("True probability", t1["msrK"]["ll_bayes"], False)])
bars("toy1_pay_vs_value", "Toy 1: does pay track true value? Spearman(epoch pay, ideal Shapley)", "rho",
     [(LBL[a], t1[a]["sp_ideal"], a == "msrK") for a in ("kelly", "loo", "msr1", "msrK", "shapley")])
bars("toy2_pay_vs_value", "Toy 2: does money reach useful layer-1 agents? Spearman(pay, ideal Shapley)", "rho",
     [(LBL[a].replace("16", "8"), t2[a]["sp_ideal"], a == "msrK") for a in ("kelly", "royalty", "loo", "msr1", "msrK", "shapley")])
bars("toy2_useless_died", "Toy 2: share of noise-only layer-1 agents that went bankrupt", "fraction",
     [(LBL[a].replace("16", "8"), (t2[a]["useless_died_frac"] or 0.0, 0.0), a == "msrK") for a in ("kelly", "royalty", "loo", "msrK")])
kk = sorted(A["kerr"], key=int)
json.dump(dict(id="msrK_error", kind="line", title="Sampled Shapley: error vs number of orders K", unit="relative L1 error",
               x=[int(k) for k in kk], y=[round(A["kerr"][k][0], 4) for k in kk], ci=[round(A["kerr"][k][1], 4) for k in kk],
               note="cost = K x agents coalition evaluations per epoch; exact Shapley = 2^agents", source=SRC),
          open(os.path.join(HERE, "charts", "msrK_error.json"), "w"), indent=1)
cats = [("substitutes x0+x4", [0, 1]), ("complements x6+x7", [4, 5]), ("noise x8+x9", [8, 9])]
json.dump(dict(id="audit_categories", kind="grouped_bar", title="Designed agents: pay per pair, nats per epoch", source=SRC,
               groups=[c for c, _ in cats],
               series={a: [round(float(sum(A[a][i][0] for i in ix)), 3) for _, ix in cats]
                       for a in ("ideal", "loo", "msr1", "msr1_rev", "shapley")}),
          open(os.path.join(HERE, "charts", "audit_categories.json"), "w"), indent=1)
print(json.dumps(out, indent=0, default=float)[:6000])
