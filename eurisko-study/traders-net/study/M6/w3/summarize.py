"""Summarise M6/w3 results: mean +- sd per arm and the pre-registered stop rule."""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "results", "*-s[0-9].json"))):
    r = json.load(open(f)); R.setdefault(r["arm"], {})[r["seed"]] = r
ARMS = ["a", "b", "kwd", "kboost", "elog", "eexp", "fb01", "fb05"]
MINE = ["elog", "eexp", "fb01", "fb05"]
KEYS = ["learnable_bce", "learnable_bce_market", "noise_excess", "wealth_gini_abs", "wealth_frac_negative", "wealth_total_per_q"]
ms = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
rows, out = [], {"arms": {}, "stop_rule": {}}
fams = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
for a in ARMS:
    if a not in R: continue
    seeds = sorted(R[a]); d = {"seeds": seeds}
    for k in KEYS:
        d[k] = ms([R[a][s][k] for s in seeds])
    d["viol_and"] = ms([R[a][s]["viol"].get("and", np.nan) for s in seeds])
    d["viol_or"] = ms([R[a][s]["viol"].get("or", np.nan) for s in seeds])
    for fm in fams:
        d["bce_" + fm] = ms([R[a][s]["fam_bce"][fm] for s in seeds])
    d["lfsr16_learnt_seeds"] = int(sum(R[a][s]["fam_bce"]["lfsr16"] < 0.35 for s in seeds))
    out["arms"][a] = d
    rows.append([a, len(seeds)] + [f"{d[k][0]:.4f}" for k in KEYS] + [f"{d['viol_and'][0]:.4f}", f"{d['viol_or'][0]:.4f}"]
                + [f"{d['bce_'+fm][0]:.4f}" for fm in fams])
kills = [k for k in ("kwd", "kboost") if k in R]
for a in MINE:
    if a not in R or "a" not in R: continue
    seeds = sorted(set(R[a]) & set(R["a"]))
    worse = [R[a][s]["learnable_bce"] >= 1.01 * R["a"][s]["learnable_bce"] for s in seeds]
    best_kill = min(out["arms"][k]["noise_excess"][0] for k in kills) if kills else np.nan
    ne, ne_sd = out["arms"][a]["noise_excess"]
    protect = ne < best_kill - ne_sd
    beats = all(R[a][s]["learnable_bce"] < min(R[b][s]["learnable_bce"] for b in ("a", "b", "kwd", "kboost") if b in R) for s in seeds)
    out["stop_rule"][a] = {"worse_than_a_by_1pct_in": f"{sum(worse)}/{len(seeds)}",
                           "noise_protection_beyond_kills": bool(protect),
                           "dropped": bool(all(worse) and len(seeds) == 3 and not protect),
                           "beats_all_references_3of3": bool(beats and len(seeds) == 3)}
out["base_arms_survive"] = any(not v["dropped"] for v in out["stop_rule"].values()) if out["stop_rule"] else None
json.dump(out, open(os.path.join(HERE, "SUMMARY-w3.json"), "w"), indent=1)
hdr = ["arm", "n"] + KEYS + ["viol_and", "viol_or"] + ["bce_" + f for f in fams]
with open(os.path.join(HERE, "SUMMARY-w3.csv"), "w") as fh:
    fh.write(",".join(hdr) + "\n")
    for r in rows: fh.write(",".join(map(str, r)) + "\n")
for a, d in out["arms"].items():
    print(f"{a:7s} learn {d['learnable_bce'][0]:.4f}+-{d['learnable_bce'][1]:.4f}  mkt {d['learnable_bce_market'][0]:.4f}  noise {d['noise_excess'][0]:+.4f}+-{d['noise_excess'][1]:.4f}  lfsr {d['bce_lfsr16'][0]:.3f}({d['lfsr16_learnt_seeds']}/3)  ho {d['bce_periodic_ho'][0]:.3f}  and {d['viol_and'][0]:.4f} or {d['viol_or'][0]:.4f}  gini {d['wealth_gini_abs'][0]:.2f} neg {d['wealth_frac_negative'][0]:.2f}")
print(json.dumps(out["stop_rule"], indent=1)); print("base_arms_survive", out["base_arms_survive"])
