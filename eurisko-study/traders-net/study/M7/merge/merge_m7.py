"""M7 lead merge: one per-seed table over all workers, and the stage-1 gate under three rules.

Reads only files the workers wrote; writes merge/per_seed.csv, merge/arms.csv, merge/gate.json.
Each worker is compared against the DeepSeek and no-balancing runs of its OWN harness (same code, paired by seed).
Rules (per E, for a wealth arm vs the better same-harness DeepSeek on that worker's primary metric):
  R1 (w1/PREREG.md, "applied by the M7 merger"): lower in 3/3 paired seeds and not collapsed.
  R2 (w1/M7-w1.md recommendation after the noise floor): R1 and paired mean gap > 1% relative, at BOTH E.
  R3 (the arm's own worker PREREG): taken from the worker's decision file / report.
Collapsed = worst-layer max load > 0.5 in >= 2/3 seeds (STUDY-PLAN M7 stop rule).
"""
import csv, glob, json, os, statistics as st
H = os.path.dirname(os.path.abspath(__file__)); M7 = os.path.dirname(H)
rows = []
def add(w, harness, metric, arm, role, E, seed, val, ml, src):
    rows.append(dict(worker=w, harness=harness, metric=metric, arm=arm, role=role, E=int(E), seed=int(seed),
                     bce=float(val), max_load_worst=float(ml), source=src))
# w1 controls
for r in csv.DictReader(open(f"{M7}/w1/results/controls_runs.csv")):
    role = "ds" if r["arm"].startswith("deepseek") else ("none" if r["arm"] == "none" else "control")
    add("w1", "w1/moe_harness.py", "bce_2nd_half", r["arm"], role, r["E"], r["seed"], r["bce_2nd_half"],
        r["max_load_worst_layer"] or "nan", "w1/results/controls_runs.csv")
# w3 exp3
for f in sorted(glob.glob(f"{M7}/w3/results/gpu/runs/*.json")):
    d = json.load(open(f)); a = d["arm"][2:]
    role = {"deepseek": "ds", "none": "none", "exp3": "wealth", "fullinfo": "diagnostic"}[a]
    add("w3", "w1/moe_harness.py + w3/exp3.py", "bce_2nd_half", a, role, d["E"], d["seed"], d["bce_2nd_half"],
        d["max_load_worst_layer"], "w3/results/gpu/runs/" + os.path.basename(f))
# w5 bankruptcy+birth
for r in csv.DictReader(open(f"{M7}/w5/results/per_seed_gpu.csv")):
    a = r["arm"]; role = "ds" if a.startswith("deepseek") else ("none" if a == "none" else "wealth")
    add("w5", "w1/moe_harness.py + w5/bb.py", "bce_2nd_half", a, role, r["E"], r["seed"], r["bce_2nd_half"],
        r["max_load_worst_layer"], "w5/results/per_seed_gpu.csv")
# w2 family-wealth prior (own harness)
for f in sorted(glob.glob(f"{M7}/w2/results/all/*.json")):
    d = json.load(open(f)); a = d["arm"]
    role = {"ds3": "ds", "ds2": "ds", "none": "none", "dsrw": "control"}.get(a, "wealth")
    add("w2", "w2/moe_w2.py", "bce_all", a, role, d["E"], d["seed"], d["bce_all"], d["max_load"],
        "w2/results/all/" + os.path.basename(f))
    add("w2", "w2/moe_w2.py", "bce_last10k", a, role, d["E"], d["seed"], d["bce_last10k"], d["max_load"],
        "w2/results/all/" + os.path.basename(f))
# w4 global/per-family wealth prior (own harness, one MoE layer)
for r in csv.DictReader(open(f"{M7}/w4/runs.csv")):
    a = r["arm"] if r["arm"] in ("deepseek", "none") else f"{r['arm']}-b{int(float(r['beta']))}"
    role = {"deepseek": "ds", "none": "none"}.get(a, "wealth")
    add("w4", "w4/job/moe_w4.py", "bce_last", a, role, r["E"], r["seed"], r["bce_last"], r["max_load"], "w4/runs.csv")

with open(f"{H}/per_seed.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def sel(**k): return [r for r in rows if all(r[a] == b for a, b in k.items())]
def by_seed(rs): return {r["seed"]: r for r in rs}
out = []
for (wk, metric) in sorted({(r["worker"], r["metric"]) for r in rows}):
    for E in (8, 64):
        base = [r for r in sel(worker=wk, metric=metric, E=E)]
        if not base: continue
        ds_arms = sorted({r["arm"] for r in base if r["role"] == "ds"})
        best = min(ds_arms, key=lambda a: st.mean(r["bce"] for r in base if r["arm"] == a))
        ds = by_seed(r for r in base if r["arm"] == best)
        nn = by_seed(r for r in base if r["arm"] == "none")
        for arm in sorted({r["arm"] for r in base}):
            s = by_seed(r for r in base if r["arm"] == arm)
            v = [s[k]["bce"] for k in sorted(s)]; ml = [s[k]["max_load_worst"] for k in sorted(s)]
            d = [s[k]["bce"] - ds[k]["bce"] for k in sorted(s) if k in ds]
            dn = [s[k]["bce"] - nn[k]["bce"] for k in sorted(s) if k in nn]
            ncol = sum(x > 0.5 for x in ml)
            out.append(dict(worker=wk, metric=metric, E=E, arm=arm, role=next(iter(s.values()))["role"], n=len(v),
                bce_mean=round(st.mean(v), 5), bce_sd=round(st.stdev(v), 5) if len(v) > 1 else "",
                ref_ds=best, d_vs_ds_mean=round(st.mean(d), 5), d_vs_ds_sd=round(st.stdev(d), 5),
                rel_vs_ds_pct=round(100 * st.mean(d) / st.mean(ds[k]["bce"] for k in ds), 2),
                wins_vs_ds=sum(x < 0 for x in d), d_vs_none_mean=round(st.mean(dn), 5) if dn else "",
                wins_vs_none=sum(x < 0 for x in dn) if dn else "",
                max_load_worst_mean=round(st.mean(ml), 3), collapsed_seeds=ncol, collapsed=ncol >= 2))
with open(f"{H}/arms.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

# gate
own = {  # R3: each worker's own pre-registered verdict (file path in the value)
  ("w2", "dsfw"): (False, "w2/results/decision.json: gap not > larger seed sd"),
  ("w2", "dsfw_fast"): (False, "w2/results/decision.json: beats at E=8 only (partial)"),
  ("w2", "dsfw_mean"): (False, "w2/results/decision.json"),
  ("w3", "exp3"): (False, "w3/results/decision.json"),
  ("w5", "bb"): (False, "w5/results/verdict_gpu.json"), ("w5", "bb-fresh"): (False, "w5/results/verdict_gpu.json"),
}
for a in ("global-b100", "global-b1000", "perfam-b100", "perfam-b1000"):
    own[("w4", a)] = (False, "w4/M7-w4.md 'Pre-registered rules applied'; w4/paired.csv")
primary = {"w2": "bce_all", "w3": "bce_2nd_half", "w4": "bce_last", "w5": "bce_2nd_half"}
gate = {"rules": __doc__, "arms": {}}
for (wk, arm), (r3, src) in sorted(own.items()):
    res = {}
    for metric in sorted({r["metric"] for r in out if r["worker"] == wk}):
        cells = {o["E"]: o for o in out if o["worker"] == wk and o["arm"] == arm and o["metric"] == metric}
        r1 = {E: (c["wins_vs_ds"] == c["n"] and not c["collapsed"]) for E, c in cells.items()}
        r2 = all(r1[E] and cells[E]["rel_vs_ds_pct"] < -1.0 for E in cells)
        res[metric] = dict(R1_by_E=r1, R2_bothE=r2, primary=(metric == primary[wk]))
    gate["arms"][f"{wk}:{arm}"] = dict(by_metric=res, R3_own_prereg=r3, R3_source=src)
p = lambda k: any(v["by_metric"][m][k] if k == "R2_bothE" else all(v["by_metric"][m]["R1_by_E"].values())
                  for v in gate["arms"].values() for m in v["by_metric"] if v["by_metric"][m]["primary"])
gate["any_R1_bothE_primary"] = p("R1"); gate["any_R2_bothE_primary"] = p("R2_bothE")
gate["any_R3"] = any(v["R3_own_prereg"] for v in gate["arms"].values())
gate["passing_R2_primary"] = [k for k, v in gate["arms"].items() for m, x in v["by_metric"].items() if x["primary"] and x["R2_bothE"]]
json.dump(gate, open(f"{H}/gate.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in gate.items() if k != "rules" and k != "arms"}, indent=1))
