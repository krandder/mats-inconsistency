"""Aggregate M5-w2 results over seeds: results/summary.csv (market, kind, metric, n, mean, sd) and summary.json."""
import csv, glob, json, os, re
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
acc = defaultdict(list)
for f in sorted(glob.glob(os.path.join(R, "*-s[0-9].json"))):
    d = json.load(open(f)); m = re.match(r"(.*)-s(\d)$", d["tag"]).group(1)
    for key, v in d["curves_summary"].items():
        kind, form, agg = key.split(":")
        for mk in ("final", "loglog", "slope_first_half", "slope_second_half"):
            acc[(m, kind, f"E_{form}_{agg}:{mk}")].append(v[mk])
    for k, v in d["meta"].get("viol", {}).items():
        acc[(m, k.replace("pre:", ""), ("pre_viol" if k.startswith("pre:") else "viol"))].append(v)
    if "bce_all" in d["meta"]:
        acc[(m, "all", "bce_heldout")].append(d["meta"]["bce_all"])
    by = defaultdict(list)
    for tr in d["traders"]:
        by[tr["kind"]].append(tr); by["all"].append(tr)
    if m.startswith("wn"):   # wealth-net: traders with no pi-parity leg (pi questions 40-47 are memorised in-sample)
        for tr in d["traders"]:
            if not any(40 <= l < 48 for l in tr["legs"]):
                by["all_nopi"].append(tr)
    for kind, trs in by.items():
        acc[(m, kind, "E_A_trader_sum")].append(sum(t["E_A"] for t in trs))
        acc[(m, kind, "E_clip_trader_sum")].append(sum(t["E_clip"] for t in trs))
        acc[(m, kind, "E_B_trader_sum")].append(sum(t["E_B"] for t in trs))
        acc[(m, kind, "n_traders")].append(len(trs))
        acc[(m, kind, "n_traders_E_A_pos")].append(sum(t["E_A"] > 0 for t in trs))
        acc[(m, kind, "invasion_eps_h1:max")].append(max(t["invasion_eps_h1"] for t in trs))
        acc[(m, kind, "invasion_eps_h2:max")].append(max(t["invasion_eps_h2"] for t in trs))
        acc[(m, kind, "invasion_eps_h2:sum")].append(sum(t["invasion_eps_h2"] for t in trs))
        acc[(m, kind, "member_max_E")].append(max(t["max_member_E"] for t in trs))
        acc[(m, kind, "member_share_leg0:mean")].append(float(np.mean([t["share_leg0_mean"] for t in trs])))
        acc[(m, kind, "member_frac_traders_share_gt_0.5")].append(float(np.mean([t["share_leg0_mean"] > 0.5 for t in trs])))
        acc[(m, kind, "fp_residual:max")].append(max(t["fp_residual"] for t in trs))
        mins = [t["minpnl_B"] for t in trs if t["minpnl_B"] is not None]
        acc[(m, kind, "bundle_min_pnl")].append(min(mins) if mins else float("nan"))
        acc[(m, kind, "bundle_trades:sum")].append(sum(t["n_B"] for t in trs))
rows = []
for (m, kind, met), v in sorted(acc.items()):
    a = np.array(v, float)
    rows.append(dict(market=m, kind=kind, metric=met, n=len(a), mean=float(np.nanmean(a)),
                     sd=float(np.nanstd(a, ddof=1)) if len(a) > 1 else float("nan"),
                     values=";".join(f"{x:.6g}" for x in a)))
with open(os.path.join(R, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
json.dump(rows, open(os.path.join(R, "summary.json"), "w"), indent=0)
print(len(rows), "rows")
