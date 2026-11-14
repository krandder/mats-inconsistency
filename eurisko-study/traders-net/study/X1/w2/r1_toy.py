"""X1 w2 R1: rerun M1/lead/toy_sign.py simulate() with fresh seeds 4-6 (clean /tmp copy), and the closed form."""
import json, sys, numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/traders-net/study/M1/lead")  # read-only import
import toy_sign as T
rows, checks = T.closed_form()
out = {"closed_form": checks, "sim": {}}
for opt in ("adam", "sgd"):
    F = [T.simulate(s, opt)[1] for s in (4, 5, 6)]
    # per-seed value = unit 0 (units 0 and 1 are identical by symmetry: same input, same zero init)
    per = lambda k: [f[k][0] for f in F]
    ident = [abs(f["credit"][0] - f[k][0]) if (k := "sgd_identity_rhs") and f[k] else None for f in F]
    d = {k: per(k) for k in ("credit", "settle", "ablation_dBCE")}
    d["unit0_equals_unit1"] = all(abs(f["credit"][0] - f["credit"][1]) < 1e-9 for f in F)
    d["noise_settle"] = [f["settle"][2] for f in F]; d["noise_credit"] = [f["credit"][2] for f in F]
    d["summary"] = {k: [float(np.mean(v)), float(np.std(v, ddof=1))] for k, v in d.items() if isinstance(v, list)}
    if opt == "sgd":
        d["sgd_identity_max_abs_err"] = max(abs(c - r) for f in F for c, r in zip(f["credit"], f["sgd_identity_rhs"]))
    out["sim"][opt] = d
json.dump(out, open("/home/agent/workspaces/eurisko-study/traders-net/study/X1/w2/results/R1_toy_fresh.json", "w"), indent=1)
print(json.dumps(out["closed_form"])); [print(o, out["sim"][o]["summary"], out["sim"][o]["unit0_equals_unit1"]) for o in out["sim"]]
print(out["sim"]["sgd"].get("sgd_identity_max_abs_err"))
