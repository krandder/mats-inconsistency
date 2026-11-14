"""M6 lead: per-seed paired view of w3's arms (rel vs a, ex-LFSR16 sensitivity, paired noise vs kill controls).
Reads w3/results/*-s[0-2].json; writes lead/w3_detail.json. Not preregistered (sensitivity only)."""
import json, glob, os
import numpy as np
M6 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = {}
for f in glob.glob(f"{M6}/w3/results/*-s[0-9].json"):
    r = json.load(open(f)); R.setdefault(r["arm"], {})[r["seed"]] = r
NL = ["periodic", "ca110", "prime", "and", "or", "periodic_ho"]
S = (0, 1, 2)
ex = lambda r: float(np.mean([r["fam_bce"][k] for k in NL]))
out = {}
for a in ["b", "kwd", "kboost", "elog", "eexp", "fb01", "fb05"]:
    out[a] = {"learn": [R[a][s]["learnable_bce"] for s in S],
              "rel_vs_a": [R[a][s]["learnable_bce"] / R["a"][s]["learnable_bce"] - 1 for s in S],
              "rel_vs_a_exlfsr": [ex(R[a][s]) / ex(R["a"][s]) - 1 for s in S],
              "rel_vs_b": [R[a][s]["learnable_bce"] / R["b"][s]["learnable_bce"] - 1 for s in S],
              "noise": [R[a][s]["noise_excess"] for s in S],
              "noise_below_both_kills_paired": [R[a][s]["noise_excess"] < min(R["kwd"][s]["noise_excess"], R["kboost"][s]["noise_excess"]) for s in S],
              "lfsr16": [R[a][s]["fam_bce"]["lfsr16"] for s in S],
              "viol_and": [R[a][s]["viol"].get("and") for s in S], "viol_or": [R[a][s]["viol"].get("or") for s in S]}
out["a"] = {"learn": [R["a"][s]["learnable_bce"] for s in S], "lfsr16": [R["a"][s]["fam_bce"]["lfsr16"] for s in S],
            "noise": [R["a"][s]["noise_excess"] for s in S], "market_noise": [R["a"][s]["noise_excess_market"] for s in S],
            "market_viol": [R["a"][s].get("viol_market") for s in S]}
json.dump(out, open(f"{M6}/lead/w3_detail.json", "w"), indent=1)
for a, v in out.items():
    print(a, {k: [round(x, 4) if isinstance(x, float) else x for x in vv] for k, vv in v.items()})
