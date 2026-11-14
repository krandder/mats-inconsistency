"""Check: s2 clean runs (G1 VM) equal w2's / lead's runs of the same arm and seed on this box (max |diff| per-family BCE)."""
import json, os
D = os.path.dirname(os.path.abspath(__file__)); M6 = os.path.dirname(D)
out = {}
for arm, src in [("a", "w2/results/a"), ("cg", "w2/results/cg"), ("cq", "w2/results/cq"), ("wd", "lead/results/wd"), ("boost", "lead/results/boost")]:
    for s in (1, 2, 3):
        A = json.load(open(f"{M6}/{src}-s{s}.json"))["bce_all"]; B = json.load(open(f"{D}/results/{arm}-clean-s{s}.json"))["bce_all"]
        out[f"{arm}-s{s}"] = max(abs(x - y) for x, y in zip(A, B))
out["all_identical"] = all(v == 0 for v in out.values())
json.dump(out, open(f"{D}/check_clean.json", "w"), indent=1); print(out)
