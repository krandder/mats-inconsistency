"""M6 lead: merge worker verdicts into one stop-rule table (re-run to re-check).
Reads w1/results/verdict-w1.json, w2/results + lead/results (kill controls in w2's harness),
w3/SUMMARY-w3.json, w4/VERDICT.json, w5/results/stoprule.json. Writes lead/M6-verdict.json."""
import json, math, os, glob
import numpy as np
M6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(M6, p)))
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
LEARN = [FAM.index(f) for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]]
NOISE = [FAM.index("ca30"), FAM.index("prng")]


def w2rec(path):
    r = json.load(open(path)); b = np.array(r["bce_all"])
    return float(np.nanmean(b[LEARN])), float(np.max(b[NOISE] - math.log(2)))


w2 = {}
for p in glob.glob(f"{M6}/w2/results/*-s*.json") + glob.glob(f"{M6}/lead/results/*-s*.json"):
    r = json.load(open(p)); w2.setdefault(r["arm"], {})[r["seed"]] = w2rec(p) + (os.path.relpath(p, M6),)
out = {"arms": {}}
# (b) w1
v1 = J("w1/results/verdict-w1.json")
out["arms"]["b"] = {"owner": "w1", "dropped": v1["B_killed"], "evidence": "w1/results/verdict-w1.json",
                    "bce_worse_1pct": {"P": v1["P_B_worse_by_1pct"], "ME": v1["ME_B_worse_by_1pct"]},
                    "protection_beyond_kills": v1["B_beats_kill_humility_3of3"]}
# (c),(d) w2 with in-harness kill controls
kc_ok = all(k in w2 and len(w2[k]) == 3 for k in ("wd", "boost"))
kc = {}
if kc_ok:
    for k in ("wd", "boost"):
        kc[k] = {"learn": [w2[k][s][0] for s in (1, 2, 3)], "noise": [w2[k][s][1] for s in (1, 2, 3)]}
    out["w2_harness_kill_controls"] = {k: {"learn_mean_sd": [float(np.mean(v["learn"])), float(np.std(v["learn"], ddof=1))],
                                           "noise_mean_sd": [float(np.mean(v["noise"])), float(np.std(v["noise"], ddof=1))],
                                           "learn_rel_vs_a": [v["learn"][i] / w2["a"][s][0] - 1 for i, s in enumerate((1, 2, 3))],
                                           "per_seed_learn": v["learn"], "per_seed_noise": v["noise"]} for k, v in kc.items()}
for a in ("cg", "cq", "d0", "dfs"):
    rel = [w2[a][s][0] / w2["a"][s][0] - 1 for s in (1, 2, 3)]
    worse = [x >= 0.01 for x in rel]
    prot = None
    if kc_ok:
        prot = all(w2[a][s][1] < min(w2["wd"][s][1], w2["boost"][s][1]) for s in (1, 2, 3))
    out["arms"][a] = {"owner": "w2", "rel_vs_a": rel, "bce_worse_1pct_3of3": all(worse),
                      "noise_per_seed": [w2[a][s][1] for s in (1, 2, 3)],
                      "protection_beyond_kills": prot,
                      "dropped": bool(all(worse) and prot is False) if prot is not None else None,
                      "evidence": "w2/verdict.json, w2/results/, lead/results/"}
# (e),(f) w3
if os.path.exists(f"{M6}/w3/SUMMARY-w3.json"):
    s3 = J("w3/SUMMARY-w3.json")
    for a, v in s3.get("stop_rule", {}).items():
        n = len(s3["arms"][a]["seeds"])
        out["arms"][a] = {"owner": "w3", **v, "n_seeds": n, "dropped": v["dropped"] if n == 3 else None,
                          "evidence": "w3/SUMMARY-w3.json"}
# (g) + 1/2-pass w4
v4 = J("w4/VERDICT.json")
for a, k in (("g_fp", "fp"), ("g_1pass", "pass1"), ("g_2pass", "pass2")):
    out["arms"][a] = {"owner": "w4", "dropped": v4[f"drop_{k}"], "rel_vs_a": v4[f"{k}_vs_plain_rel"],
                      "evidence": "w4/VERDICT.json"}
out["arms"]["g_fp"]["fixed_point_kept"] = v4["keep_fp"]
out["arms"]["g_fp"]["fp_vs_2pass_violation_rel"] = v4["fp_vs_pass2_M3_rel"]
# RETARGET committed heads w5 (reference U = arm (b))
v5 = J("w5/results/stoprule.json")
for a, v in v5.items():
    out["arms"]["commit_" + a] = {"owner": "w5", "dropped": v["dropped"], "rel_vs_U": v["rel_bce_vs_U"],
                                  "evidence": "w5/results/stoprule.json"}
# sensitivity (not preregistered): w2 harness, lfsr16 BCE and learnable BCE without lfsr16, per seed
NL = [FAM.index(f) for f in ["periodic", "ca110", "prime", "and", "or", "periodic_ho"]]
sens = {}
for a in ("a", "wd", "boost", "cg", "cq", "d0", "dfs"):
    d = "lead" if a in ("wd", "boost") else "w2"
    rs = [json.load(open(f"{M6}/{d}/results/{a}-s{s}.json")) for s in (1, 2, 3)]
    sens[a] = {"lfsr16": [r["bce_all"][1] for r in rs],
               "learn_exlfsr": [float(np.mean([r["bce_all"][i] for i in NL])) for r in rs]}
out["w2_harness_exlfsr"] = sens
alive = [a for a, v in out["arms"].items() if v.get("dropped") is False]
undecided = [a for a, v in out["arms"].items() if v.get("dropped") is None]
out["surviving"] = alive; out["undecided"] = undecided
out["base_arms_survive"] = len(alive) > 0
json.dump(out, open(f"{M6}/lead/M6-verdict.json", "w"), indent=1)
print(json.dumps({"surviving": alive, "undecided": undecided, "dropped": [a for a, v in out["arms"].items() if v.get("dropped")],
                  "base_arms_survive": out["base_arms_survive"]}, indent=1))
