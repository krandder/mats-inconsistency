"""Summarise results/*.json into SUMMARY.csv (mean, sd, per-seed) and the stop-rule verdicts in
VERDICT.json. usage: python3 summarize.py"""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in glob.glob(os.path.join(HERE, "results", "*-s[123].json")):
    d = json.load(open(f)); R[(d["stream"], d["arm"], d["seed"])] = d
ARMS = ["plain", "plain_wd", "boost", "pass1", "pass2", "fp"]
SEEDS = [1, 2, 3]
rows = []

def metric(stream, name, get):
    for a in ARMS:
        v = [get(R[(stream, a, s)]) if (stream, a, s) in R else np.nan for s in SEEDS]
        rows.append(dict(stream=stream, arm=a, metric=name, mean=np.nanmean(v), sd=np.nanstd(v, ddof=1),
                         s1=v[0], s2=v[1], s3=v[2]))
    return {a: [get(R[(stream, a, s)]) if (stream, a, s) in R else np.nan for s in SEEDS] for a in ARMS}

m1 = metric("base", "M1_learnable_bce", lambda d: d["M1_learnable_bce"])
metric("base", "M2_noise_excess", lambda d: d["M2_noise_excess"])
for f in ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]:
    metric("base", f"bce:{f}", lambda d, f=f: d["net_bce"][f])
metric("base", "maker_M1_learnable_bce", lambda d: np.mean([d["maker_bce"][f] for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]]))
for k in ("and", "or"):
    metric("base", f"viol:{k}", lambda d, k=k: d["andor_viol"][k])
m3 = metric("li", "M3_D_coherence", lambda d: d["M3_D_coherence"])
metric("li", "maker_M3_D_coherence", lambda d: d["maker_M3_D_coherence"])
for k in ("taut_or", "contra", "impl", "ex3"):
    metric("li", f"pre:{k}", lambda d, k=k: d["viol"]["pre:" + k]["mean"])
for k in ("and", "or", "twin", "ex3"):
    metric("li", f"viol:{k}", lambda d, k=k: d["viol"][k]["mean"])
metric("li", "mean_bce_all_fams", lambda d: np.mean(list(d["net_bce"].values())))
for st in ("base", "li"):
    for k in ("iters_mean", "iters_max", "at_cap_frac", "resid_max", "nonconv_frac"):
        rows.append(dict(stream=st, arm="fp", metric="fp:" + k,
                         **{f"s{s}": R[(st, "fp", s)]["fp"][k] if (st, "fp", s) in R else np.nan for s in SEEDS}))
        r = rows[-1]; v = [r["s1"], r["s2"], r["s3"]]; r["mean"], r["sd"] = np.nanmean(v), np.nanstd(v, ddof=1)
    metric(st, "wall_s", lambda d: d["wall_s"])
with open(os.path.join(HERE, "SUMMARY.csv"), "w") as fh:
    w = csv.DictWriter(fh, ["stream", "arm", "metric", "mean", "sd", "s1", "s2", "s3"]); w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
V = {"source": "results/*.json via summarize.py", "rule_drop": "arm M1 >= plain M1 * 1.01 in 3/3 seeds",
     "rule_keep_fp": "fp M3 <= 0.5 * pass2 M3 (mean) and in >= 2/3 seeds"}
pl = np.array(m1["plain"])
for a in ("pass1", "pass2", "fp", "boost", "plain_wd"):
    x = np.array(m1[a]); worse = x >= pl * 1.01
    V[f"drop_{a}"] = bool(worse.all()); V[f"{a}_vs_plain_rel"] = (x / pl - 1).tolist()
fp3, p23 = np.array(m3["fp"]), np.array(m3["pass2"])
V["fp_vs_pass2_M3_rel"] = (fp3 / p23 - 1).tolist()
V["keep_fp"] = bool(np.nanmean(fp3) <= 0.5 * np.nanmean(p23) and (fp3 <= 0.5 * p23).sum() >= 2)
V["base_arms_survive_g"] = not V["drop_fp"]
V["n_results"] = len(R)
json.dump(V, open(os.path.join(HERE, "VERDICT.json"), "w"), indent=1)
print(json.dumps(V, indent=1))

# secondary: learnable BCE without lfsr16 (bimodal under prequential training, S0 w1), and lfsr16 learnt counts
L2 = ["periodic", "ca110", "prime", "and", "or", "periodic_ho"]
sec = {}
for a in ARMS:
    v = [float(np.mean([R[("base", a, s)]["net_bce"][f] for f in L2])) for s in SEEDS]
    sec[a] = {"M1_excl_lfsr16": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
              "lfsr16_learnt_seeds": int(sum(R[("base", a, s)]["net_bce"]["lfsr16"] < 0.5 for s in SEEDS))}
for a in ARMS:
    sec[a]["rel_vs_plain"] = (np.array(sec[a]["M1_excl_lfsr16"]) / np.array(sec["plain"]["M1_excl_lfsr16"]) - 1).tolist()
json.dump(sec, open(os.path.join(HERE, "SECONDARY.json"), "w"), indent=1)
print(json.dumps({a: (round(sec[a]["mean"], 5), round(sec[a]["sd"], 5), sec[a]["lfsr16_learnt_seeds"], [round(x, 4) for x in sec[a]["rel_vs_plain"]]) for a in ARMS}))
