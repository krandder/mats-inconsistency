"""E5 analysis: reads runs/E5c/*/eval.jsonl, runs/E5a/*/e5_metrics.json, *.format.json, summary.json;
writes results.json and prints markdown tables (mean +- sd over seeds)."""
import glob, json, os, re
import numpy as np
X = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(X, "runs")
out = {}


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), "n": len(v), "vals": v}


def fmt(d, k=4):
    if d is None:
        return "n/a"
    return f"{d['mean']:.{k}f} ± {d['sd']:.{k}f} (n={d['n']})"


# ---------------- E5c (E5C_DIR=E5c2 primary, E5c = old plain format)
import sys
EC = os.environ.get("E5C_DIR", "E5c3")
c = {}
for f in glob.glob(f"{R}/{EC}/**/eval.jsonl", recursive=True):
    arm, seed = re.match(r"(\w)-s(\d)", os.path.basename(os.path.dirname(f))).groups()
    for line in open(f):
        r = json.loads(line)
        c.setdefault(arm, {}).setdefault(r["step"], {})[int(seed)] = r
E5c = {}
keys = ["brier_vs_bayes", "ece_vs_bayes", "brier_vs_truth", "ece_vs_truth", "mae_vs_bayes", "mean_p", "query_correct",
        "swapgoal_kl", "swapgoal_brier_vs_bayes", "other_board_brier_vs_bayes", "bayes_floor_brier"]
for arm, steps in sorted(c.items()):
    for st, seeds in sorted(steps.items()):
        E5c.setdefault(arm, {})[st] = {k: ms([s.get(k) for s in seeds.values()]) for k in keys}
out[EC] = E5c
print("## E5c (goal board cells; mean ± sd over seeds)")
for st in sorted({s for a in E5c.values() for s in a}):
    print(f"\nstep {st}\n| arm | Brier vs Bayes | ECE vs Bayes | Brier vs truth | ECE vs truth | query correct | swap-goal KL |")
    print("|---|---|---|---|---|---|---|")
    for arm in "VQNS":
        if arm in E5c and st in E5c[arm]:
            m = E5c[arm][st]
            print(f"| {arm} | {fmt(m['brier_vs_bayes'])} | {fmt(m['ece_vs_bayes'])} | {fmt(m['brier_vs_truth'])} | {fmt(m['ece_vs_truth'])} | {fmt(m['query_correct'],3)} | {fmt(m['swapgoal_kl'],5)} |")
# key contrast at final step: per-seed paired differences
fin = max((s for a in c.values() for s in a), default=None)
if fin is not None:
    con = {}
    for a, b in [("Q", "V"), ("N", "V"), ("S", "V"), ("N", "Q")]:
        if a in c and b in c and fin in c[a] and fin in c[b]:
            ds = [c[a][fin][s]["brier_vs_bayes"] - c[b][fin][s]["brier_vs_bayes"] for s in sorted(c[a][fin]) if s in c[b][fin]]
            sa = ms([x["brier_vs_bayes"] for x in c[a][fin].values()]); sb = ms([x["brier_vs_bayes"] for x in c[b][fin].values()])
            pooled = float(np.sqrt((sa["sd"] ** 2 + sb["sd"] ** 2) / 2)) if sa["n"] > 1 and sb["n"] > 1 else float("nan")
            con[f"{a}-{b}"] = {"diff_mean": float(np.mean(ds)), "per_seed": ds, "pooled_seed_sd": pooled,
                               "paired_sd": float(np.std(ds, ddof=1)) if len(ds) > 1 else float("nan"),
                               "seeds": sorted(s for s in c[a][fin] if s in c[b][fin])}
    out[EC + "_contrasts_final"] = {"step": fin, **con}
    print("\nE5c contrasts (Brier vs Bayes, final step):", json.dumps(con, indent=0))

# ---------------- E5a
runs = {}
for f in glob.glob(f"{R}/E5a/*/e5_metrics.json"):
    name = os.path.basename(os.path.dirname(f))
    m = re.match(r"(B0|D1|P)-(deleted|visible|shuffled)-(approval|random)-s(\d)", name)
    if not m:
        continue
    A, GM, RW, S = m.groups()
    j = json.load(open(f))
    sm = os.path.join(os.path.dirname(f), "summary.json")
    mv = json.load(open(sm)).get("mask_violations") if os.path.exists(sm) else None
    if "pre_rl" not in j or "post_rl" not in j:
        continue
    d = {k: j["post_rl"][k] - j["pre_rl"][k] for k in ["b_G", "b_G_plan", "bias_raw", "overconf", "brier_vs_bayes",
                                                         "fc_optimism_vs_bayes", "reward", "env_reward", "fc_stated_conf_rate"]
         if isinstance(j["post_rl"].get(k), (int, float)) and isinstance(j["pre_rl"].get(k), (int, float))}
    lv = {f"post_{k}": j["post_rl"][k] for k in ["b_G", "overconf", "brier_vs_bayes", "reward", "probe_sit_plan", "probe_fc", "probe_belief", "probe_chance"]}
    lv.update({f"pre_{k}": j["pre_rl"][k] for k in ["b_G", "overconf", "probe_sit_plan", "probe_fc", "probe_belief"]})
    runs.setdefault((A, GM, RW), {})[int(S)] = {**{f"d_{k}": v for k, v in d.items()}, **lv, "mask_violations": mv}
E5a = {}
for key, seeds in sorted(runs.items()):
    ks = sorted({k for v in seeds.values() for k in v})
    E5a["/".join(key)] = {k: ms([v.get(k) for v in seeds.values()]) for k in ks}
out["E5a"] = E5a
print("\n## E5a (change post_rl - pre_rl; mean ± sd over seeds)")
print("| arm/goal/reward | Δb_G | Δb_G plan cells | Δoverconf | ΔBrier vs Bayes | Δfc optimism | Δreward | mask viol |")
print("|---|---|---|---|---|---|---|---|")
for k, m in E5a.items():
    print(f"| {k} | {fmt(m.get('d_b_G'))} | {fmt(m.get('d_b_G_plan'))} | {fmt(m.get('d_overconf'),3)} | {fmt(m.get('d_brier_vs_bayes'))} | {fmt(m.get('d_fc_optimism_vs_bayes'),3)} | {fmt(m.get('d_reward'),3)} | {m.get('mask_violations',{}) and m['mask_violations']['vals']} |")
# decision rule
dec = {}
for A in ("B0", "D1"):
    for RW in ("approval", "random"):
        v, s = runs.get((A, "visible", RW)), runs.get((A, "shuffled", RW))
        dl = runs.get((A, "deleted", RW))
        if v and s:
            vv = [x["d_b_G"] for x in v.values()]; ss = [x["d_b_G"] for x in s.values()]
            pooled = float(np.sqrt((np.var(vv, ddof=1) + np.var(ss, ddof=1)) / 2)) if len(vv) > 1 and len(ss) > 1 else float("nan")
            dec[f"{A}/{RW}"] = {"dbG_visible_minus_shuffled": float(np.mean(vv) - np.mean(ss)), "pooled_sd": pooled,
                                "keep_deletion_rule_met": bool(np.mean(vv) - np.mean(ss) > pooled),
                                "dbG_deleted": float(np.mean([x["d_b_G"] for x in dl.values()])) if dl else None}
out["E5a_decision"] = dec
print("\nE5a decision rule:", json.dumps(dec, indent=0))
# probes
print("\n## E5b probes (post_rl; CV accuracy; chance ~ probe_chance)")
print("| arm/goal/reward | sit at plan cells | forecast outputs | belief outputs | chance |")
print("|---|---|---|---|---|")
for k, m in E5a.items():
    print(f"| {k} | {fmt(m.get('post_probe_sit_plan'),3)} | {fmt(m.get('post_probe_fc'),3)} | {fmt(m.get('post_probe_belief'),3)} | {fmt(m.get('post_probe_chance'),3)} |")
# format control
fc = {}
for f in glob.glob(f"{R}/E5a/*.format.json"):
    A, GM, S = re.match(r"(B0|D1|P)-(\w+)-s(\d)\.format", os.path.basename(f)).groups()
    for ck, v in json.load(open(f)).items():
        if "missing" in v:
            continue
        fc.setdefault(f"{A}/{GM}/{ck}", {}).setdefault("kl", []).append(v["kl_true_vs_swapped_nats_per_slot"])
        fc[f"{A}/{GM}/{ck}"].setdefault("dbrier", []).append(v["brier_vs_bayes_swapped"] - v["brier_vs_bayes_true"])
        fc[f"{A}/{GM}/{ck}"].setdefault("dbG", []).append(v["b_G_true"] - v["b_G_swapped_text"])
out["format_control"] = {k: {kk: ms(vv) for kk, vv in v.items()} for k, v in sorted(fc.items())}
print("\n## Format control: belief pass with true vs swapped goal text (same weights)")
print("| arm/train-goal-mode/checkpoint | KL nats/slot | Brier(swapped)-Brier(true) vs Bayes | b_G(true)-b_G(swapped text) |")
print("|---|---|---|---|")
for k, v in out["format_control"].items():
    print(f"| {k} | {fmt(v['kl'],5)} | {fmt(v['dbrier'],5)} | {fmt(v['dbG'],5)} |")
json.dump(out, open(os.path.join(X, "results.json" if EC == "E5c3" else f"results-{EC}.json"), "w"), indent=1)
