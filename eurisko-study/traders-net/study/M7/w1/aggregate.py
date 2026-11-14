"""Merge w1 Stage-1 control runs into results/controls_runs.csv (one row per run) and
results/controls.csv (mean, sd over seeds per arm x E), and apply the PREREG collapse flag."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "gpu", "runs")
PFX = sys.argv[2] if len(sys.argv) > 2 else ""
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(src, "*.json")))]
FAM8 = ["periodic", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
for r in rows:   # every family has 8 questions, all live in the 2nd half: equal weights
    r["bce2h_nolfsr"] = float(np.mean([r["bce2h." + f] for f in FAM8]))
    r["lfsr_learnt"] = int(r["bce2h.lfsr16"] < 0.4)
keys = ["bce_2nd_half", "bce2h_nolfsr", "lfsr_learnt", "bce_all", "bce_post_shift_2k", "max_load", "max_load_worst_layer", "dead_frac",
        "cv_load", "mi_expert_family_bits", "mi_norm", "active_flops_per_token", "n_params", "wall_s",
        "bce2h.periodic", "bce2h.lfsr16", "bce2h.ca30", "bce2h.ca110", "bce2h.prime", "bce2h.prng",
        "bce2h.and", "bce2h.or", "bce2h.periodic_ho"]
with open(os.path.join(HERE, "results", PFX + "controls_runs.csv"), "w") as fh:
    fh.write("arm,E,seed," + ",".join(keys) + "\n")
    for r in rows:
        fh.write(f"{r['arm']},{r['E']},{r['seed']}," + ",".join(str(r.get(k, "")) for k in keys) + "\n")
groups = {}
for r in rows:
    groups.setdefault((r["arm"], r["E"]), []).append(r)
out = []
with open(os.path.join(HERE, "results", PFX + "controls.csv"), "w") as fh:
    fh.write("arm,E,n_seeds," + ",".join(f"{k}_mean,{k}_sd" for k in keys) + ",collapsed_seeds\n")
    for (arm, E), rs in sorted(groups.items(), key=lambda t: (t[0][1], t[0][0])):
        vals = []
        for k in keys:
            v = np.array([float(r.get(k, np.nan)) for r in rs])
            vals += [np.nanmean(v), np.nanstd(v, ddof=1) if len(v) > 1 else np.nan]
        coll = sum(float(r.get("max_load_worst_layer", 0)) > 0.5 for r in rs)
        fh.write(f"{arm},{E},{len(rs)}," + ",".join(f"{x:.6g}" for x in vals) + f",{coll}\n")
        out.append((arm, E, len(rs), vals, coll))
for arm, E, n, vals, coll in out:
    d = dict(zip(keys, zip(vals[0::2], vals[1::2])))
    print(f"E={E:>2} {arm:15s} n={n} bce2h {d['bce_2nd_half'][0]:.4f}±{d['bce_2nd_half'][1]:.4f} "
          f"post-shift {d['bce_post_shift_2k'][0]:.4f}±{d['bce_post_shift_2k'][1]:.4f} "
          f"nolfsr {d['bce2h_nolfsr'][0]:.4f}±{d['bce2h_nolfsr'][1]:.4f} lfsrL {d['lfsr_learnt'][0]*n:.0f} maxload {d['max_load'][0]:.3f}±{d['max_load'][1]:.3f} worst {d['max_load_worst_layer'][0]:.3f} "
          f"dead {d['dead_frac'][0]:.3f} MI {d['mi_expert_family_bits'][0]:.3f}±{d['mi_expert_family_bits'][1]:.3f} collapsed {coll}/{n}")

# Paired differences vs the better DeepSeek u at each E (seed-paired: same stream and init seed).
by = {(r["arm"], r["E"], r["seed"]): r for r in rows}
with open(os.path.join(HERE, "results", PFX + "controls_paired.csv"), "w") as fh:
    fh.write("E,arm,ref,metric,n_pairs,diff_mean,diff_sd,n_arm_better,rel_diff_mean\n")
    for E in sorted({r["E"] for r in rows}):
        ds = [a for a in ("deepseek-u1e-3", "deepseek-u1e-2") if (a, E) in groups]
        if not ds:
            continue
        ref = min(ds, key=lambda a: np.mean([r["bce_2nd_half"] for r in groups[(a, E)]]))
        for arm in sorted({a for (a, e) in groups if e == E or a == "dense"}):
            if arm == ref:
                continue
            for met in ("bce_2nd_half", "bce2h_nolfsr", "bce_post_shift_2k"):
                d, rel = [], []
                for s in (1, 2, 3):
                    ra = by.get((arm, E if arm != "dense" else 8, s)); rr = by.get((ref, E, s))
                    if ra and rr:
                        d.append(ra[met] - rr[met]); rel.append((ra[met] - rr[met]) / rr[met])
                if d:
                    fh.write(f"{E},{arm},{ref},{met},{len(d)},{np.mean(d):.6g},{np.std(d, ddof=1) if len(d) > 1 else float('nan'):.3g},"
                             f"{sum(x < 0 for x in d)},{np.mean(rel):.4g}\n")
print(open(os.path.join(HERE, "results", PFX + "controls_paired.csv")).read())
