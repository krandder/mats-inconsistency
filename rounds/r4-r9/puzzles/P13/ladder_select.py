"""Phase 2B selection (PREREG ECON-ACTIVE LADDER): a candidate qualifies with frac_capped_pix (s < 1) >= 0.10 over the last
25% of its iterations and 9x9 val exact >= 0.9 at 12n; both qualify -> higher v33 exact; neither -> ECON-active is not run
("economics could not be made to bind").  Writes results/ladder_selection.json.
03:20Z addendum: --tag 0p5 (default) reads the training log truncated at the 0.5 h step count (an extension resumes into
the same --out, so its log is cumulative); --tag 1p0 reads the 1.0 h extension (ev_<n>_x, log to 3300 steps) and writes
results/ladder_selection_1p0.json too. plan_main.py reads results/ladder_selection.json (the last applicable one).
Usage: ladder_select.py [--tag 0p5|1p0]"""
import json, os, sys
import numpy as np
from pilot_select import D, find

L = json.load(open(os.path.join(D, "results", "ladder_plan.json")))
tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "0p5"
suf = "_x" if tag == "1p0" else ""
out = {"cands": {}, "tag": tag}
for n, extra in L["cands"].items():
    ej, tj = find(f"ev_{n}{suf}.json"), find(f"{n}.json")
    r = {"extra": extra, "missing": not (ej and tj)}
    if ej and tj:
        S = json.load(open(ej))["sets"]
        T = json.load(open(tj))
        if tag == "0p5" and max(l["it"] for l in T["log"]) >= L["steps"]:
            T["log"] = [l for l in T["log"] if l["it"] < L["steps"]]
            r["log_truncated_at"] = L["steps"]
        it = max(l["it"] for l in T["log"])
        dg = [l["diag"] for l in T["log"] if "diag" in l and l["it"] >= 0.75 * it and l["diag"].get("iters")]
        r.update({s + "_exact": S[s]["exact@12n"] for s in ["v9", "m13", "v33", "m59"]})
        r["frac_capped_last25"] = float(np.mean([d["frac_capped_pix"] for d in dg])) if dg else 0.0
        r["respawns"] = dg[-1]["respawns"] if dg else None
        r["qualifies"] = r["frac_capped_last25"] >= 0.10 and r["v9_exact"] >= 0.9
    out["cands"][n] = r
q = [n for n, r in out["cands"].items() if r.get("qualifies")]
best = max(q, key=lambda n: out["cands"][n]["v33_exact"]) if q else None
out["selected_name"] = best
out["selected"] = out["cands"][best]["extra"] if best else None
out["S3_note"] = None if best else "economics could not be made to bind"
json.dump(out, open(os.path.join(D, "results", "ladder_selection.json"), "w"), indent=1)
json.dump(out, open(os.path.join(D, "results", f"ladder_selection_{tag}.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
