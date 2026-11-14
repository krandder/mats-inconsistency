"""M7 w5: summarise bankruptcy+birth runs. Rerun: python3 analyze.py
Primary set  'gpu': results/gpu/runs/*.json, every arm incl. none/DeepSeek on the same L4 VM (job m7-w5-bb).
Secondary set 'cpu': results/cpu/runs/bb*.json (CPU reproduction) vs w1's controls (w1/results/controls_runs.csv, GPU).
Writes results/summary_<set>.csv, results/verdict_<set>.json, results/per_seed_<set>.csv."""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["bce_2nd_half", "bce_all", "bce_post_shift_2k", "bce2h.lfsr16", "bce2h.periodic_ho", "max_load",
        "max_load_worst_layer", "dead_frac", "cv_load", "mi_expert_family_bits", "n_births", "gini_income", "wall_s"]


def load_json(pattern, only_bb=False):
    rows = {}
    for f in glob.glob(pattern):
        r = json.load(open(f))
        arm = os.path.basename(f).rsplit("-E", 1)[0]
        if only_bb and not arm.startswith("bb"):
            continue
        rows[(arm, int(r["E"]), int(r["seed"]))] = r
    return rows


def load_w1():
    rows = {}
    for r in csv.DictReader(open(os.path.join(HERE, "../w1/results/controls_runs.csv"))):
        rr = {k: (float(v) if v not in ("", "nan") else np.nan) for k, v in r.items() if k != "arm"}
        rows[(r["arm"], int(rr["E"]), int(rr["seed"]))] = rr
    return rows


def g(r, k):
    v = r.get(k, np.nan)
    return float(v) if v is not None else np.nan


def summarise(rows, name):
    arms = sorted({a for a, _, _ in rows}); Es = sorted({e for _, e, _ in rows})
    out = []
    for a in arms:
        for E in Es:
            rs = [rows[(a, E, s)] for s in (1, 2, 3) if (a, E, s) in rows]
            if not rs:
                continue
            d = {"arm": a, "E": E, "n": len(rs), "collapsed_seeds": int(sum(g(r, "max_load_worst_layer") > 0.5 for r in rs)),
                 "lfsr_learnt_seeds": int(sum(g(r, "bce2h.lfsr16") < 0.3 for r in rs))}
            for k in KEYS:
                v = np.array([g(r, k) for r in rs])
                ok = np.isfinite(v)
                d[k + "_mean"] = float(v[ok].mean()) if ok.any() else np.nan
                d[k + "_sd"] = float(v[ok].std(ddof=1)) if ok.sum() > 1 else np.nan
            out.append(d)
    cols = ["arm", "E", "n", "collapsed_seeds", "lfsr_learnt_seeds"] + [k + s for k in KEYS for s in ("_mean", "_sd")]
    with open(os.path.join(HERE, f"results/summary_{name}.csv"), "w") as fh:
        fh.write(",".join(cols) + "\n")
        for d in out:
            fh.write(",".join(f"{d[c]:.6g}" if isinstance(d[c], float) else str(d[c]) for c in cols) + "\n")
    with open(os.path.join(HERE, f"results/per_seed_{name}.csv"), "w") as fh:
        fh.write("arm,E,seed," + ",".join(KEYS) + "\n")
        for (a, E, s) in sorted(rows):
            fh.write(f"{a},{E},{s}," + ",".join(f"{g(rows[(a, E, s)], k):.6g}" for k in KEYS) + "\n")
    return arms, Es


def verdict(bb_rows, ctrl_rows, name):
    rows = dict(ctrl_rows); rows.update(bb_rows)
    arms, Es = summarise(rows, name)
    ver = {}
    for a in [x for x in arms if x.startswith("bb")]:
        for E in Es:
            ds = [x for x in arms if x.startswith("deepseek") and all((x, E, s) in rows for s in (1, 2, 3))]
            if not ds or not all((a, E, s) in rows for s in (1, 2, 3)):
                continue
            best = min(ds, key=lambda x: np.mean([g(rows[(x, E, s)], "bce_2nd_half") for s in (1, 2, 3)]))
            for ref in [best] + [x for x in ("none",) if all((x, E, s) in rows for s in (1, 2, 3))]:
                diffs = [g(rows[(a, E, s)], "bce_2nd_half") - g(rows[(ref, E, s)], "bce_2nd_half") for s in (1, 2, 3)]
                rel = [g(rows[(a, E, s)], "bce_2nd_half") / g(rows[(ref, E, s)], "bce_2nd_half") - 1 for s in (1, 2, 3)]
                coll = sum(g(rows[(a, E, s)], "max_load_worst_layer") > 0.5 for s in (1, 2, 3))
                dm, dsd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
                ver[f"{a}@E{E}_vs_{ref}"] = dict(ref=ref, diffs=diffs, diff_mean=dm, diff_sd=dsd, rel_mean=float(np.mean(rel)),
                    wins=int(sum(x < 0 for x in diffs)), collapsed_seeds=int(coll),
                    beats_prereg=bool(dm < 0 and dm + dsd < 0 and coll < 2),
                    beats_w1_rule=bool(all(x < 0 for x in diffs) and coll < 2), worse_2pct=bool(np.mean(rel) > 0.02))
    json.dump(ver, open(os.path.join(HERE, f"results/verdict_{name}.json"), "w"), indent=1)
    return ver


os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
gpu = load_json(os.path.join(HERE, "results/gpu/runs/*.json"))
v1 = verdict({k: v for k, v in gpu.items() if k[0].startswith("bb")}, {k: v for k, v in gpu.items() if not k[0].startswith("bb")}, "gpu")
cpu = load_json(os.path.join(HERE, "results/cpu/runs/*.json"), only_bb=True)
v2 = verdict(cpu, load_w1(), "cpu")
for n, v in (("gpu", v1), ("cpu", v2)):
    print(n)
    for k, d in v.items():
        print(f"  {k}: diff {d['diff_mean']:+.4f} +- {d['diff_sd']:.4f} rel {d['rel_mean']:+.3%} wins {d['wins']}/3 collapsed {d['collapsed_seeds']}/3 beats {d['beats_prereg']}")
print(open(os.path.join(HERE, "results/summary_gpu.csv")).read())
