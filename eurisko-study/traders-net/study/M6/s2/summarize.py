"""M6 stress s2: aggregate results/*.json -> per_seed.csv, summary.csv, verdict.json (re-run to re-check)."""
import json, glob, os, math
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); R = f"{D}/results"
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
LEARN = [FAM.index(f) for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]]
NL = [i for i in LEARN if FAM[i] != "lfsr16"]
NOISEF = [FAM.index("ca30"), FAM.index("prng")]
ARMS = ["a", "b", "cg", "cq", "fb01", "fb05", "wd", "boost"]; KC = ["wd", "boost"]
CONDS = ["clean", "iid", "fixed"]; SEEDS = [1, 2, 3]
rec = {}
for fn in sorted(glob.glob(f"{R}/*-s*.json")):
    r = json.load(open(fn)); b = np.array(r["bce_all"]); b0 = np.array(r["bce_l0_all"])
    rec[(r["arm"], r["noise"], r["seed"])] = {
        "learn": float(np.nanmean(b[LEARN])), "learn_nolfsr": float(np.nanmean(b[NL])),
        "noisefam": float(np.max(b[NOISEF] - math.log(2))), "lfsr16": float(b[1]),
        "learn_l0": float(np.nanmean(b0[LEARN])), "learn_late": float(np.nanmean(np.array(r["bce_late"])[LEARN])),
        "k_noise": r.get("k_noise", float("nan")), "file": os.path.relpath(fn, D)}
keys = ["learn", "learn_nolfsr", "learn_late", "noisefam", "lfsr16", "learn_l0", "k_noise"]
with open(f"{D}/per_seed.csv", "w") as f:
    f.write("arm,noise,seed," + ",".join(keys) + ",file\n")
    for (a, c, s), v in sorted(rec.items()):
        f.write(f"{a},{c},{s}," + ",".join(f"{v[k]:.5f}" for k in keys) + f",{v['file']}\n")
def ms(x):
    x = [v for v in x if not (isinstance(v, float) and math.isnan(v))]
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"), len(x))
summ = {}
with open(f"{D}/summary.csv", "w") as f:
    f.write("arm,noise,n," + ",".join(f"{k}_mean,{k}_sd" for k in keys) + ",delta_learn_mean,delta_learn_sd,delta_nolfsr_mean,delta_nolfsr_sd\n")
    for a in ARMS:
        for c in CONDS:
            ss = [s for s in SEEDS if (a, c, s) in rec]
            if not ss: continue
            row = {k: ms([rec[(a, c, s)][k] for s in ss]) for k in keys}
            dl = [rec[(a, c, s)]["learn"] - rec[(a, "clean", s)]["learn"] for s in ss if (a, "clean", s) in rec]
            dn = [rec[(a, c, s)]["learn_nolfsr"] - rec[(a, "clean", s)]["learn_nolfsr"] for s in ss if (a, "clean", s) in rec]
            row["delta"] = ms(dl); row["delta_nolfsr"] = ms(dn); summ[f"{a}/{c}"] = row
            f.write(f"{a},{c},{len(ss)}," + ",".join(f"{row[k][0]:.5f},{row[k][1]:.5f}" for k in keys)
                    + f",{row['delta'][0]:.5f},{row['delta'][1]:.5f},{row['delta_nolfsr'][0]:.5f},{row['delta_nolfsr'][1]:.5f}\n")
ver = {"checked_files": len(rec), "expected_files": len(ARMS) * len(CONDS) * len(SEEDS)}
for c in ["iid", "fixed"]:
    vc = {}
    have = lambda a: all((a, c, s) in rec and (a, "clean", s) in rec for s in SEEDS)
    kc_ok = all(have(k) for k in KC)
    for a in ARMS:
        if not have(a): vc[a] = "incomplete"; continue
        g = lambda s, k="learn": rec[(a, c, s)][k]
        rel = [g(s) / rec[(a, "clean", s)]["learn"] - 1 for s in SEEDS]
        v = {"learn_noisy": [g(s) for s in SEEDS], "rel_delta_vs_clean": rel,
             "H3_harm_bounded_1pct_3of3": all(x <= 0.01 for x in rel),
             "rel_vs_l0": [g(s) / rec[(a, c, s)]["learn_l0"] - 1 for s in SEEDS]}
        v["beats_l0_3of3"] = all(x < 0 for x in v["rel_vs_l0"])
        if kc_ok and a not in KC:
            best = [min(rec[(k, c, s)]["learn"] for k in KC) for s in SEEDS]
            bestn = [min(rec[(k, c, s)]["learn_nolfsr"] for k in KC) for s in SEEDS]
            v["minKC_noisy"] = best; v["minKC_noisy_nolfsr"] = bestn
            v["H2_beats_kill_controls_3of3"] = all(g(s) < best[i] for i, s in enumerate(SEEDS))
            v["H2_beats_kill_controls_nolfsr_3of3"] = all(g(s, "learn_nolfsr") < bestn[i] for i, s in enumerate(SEEDS))
            v["H2_stress_win"] = v["H2_beats_kill_controls_3of3"] and v["H2_beats_kill_controls_nolfsr_3of3"]
        if a in ("cg", "cq"):
            kn = [rec[(a, c, s)]["k_noise"] for s in SEEDS]
            v["k_noise"] = kn; v["H1_defunds_noise_k_lt_0.1_3of3"] = all(x < 0.1 for x in kn)
        vc[a] = v
    vc["stress_winners"] = [a for a in ARMS if isinstance(vc.get(a), dict) and vc[a].get("H2_stress_win")]
    ver[c] = vc
json.dump(ver, open(f"{D}/verdict.json", "w"), indent=1)
print(open(f"{D}/summary.csv").read()); print(json.dumps({c: ver[c].get("stress_winners") for c in ["iid", "fixed"] if c in ver}))
