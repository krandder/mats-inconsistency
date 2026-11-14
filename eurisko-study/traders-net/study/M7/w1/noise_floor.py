"""Cross-device replicate: the same 18 E=8 runs (same seed, same code) on the L4 (results/gpu/runs)
and on this box's CPU (results/cpu-e8-replicate). |GPU - CPU| per run is a floor for run-level
noise that seeds do not capture (nondeterministic kernels, float order). Writes results/noise_floor.json."""
import glob, json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
out = {"pairs": []}
for f in sorted(glob.glob(os.path.join(H, "results/cpu-e8-replicate/*.json"))):
    g = os.path.join(H, "results/gpu/runs", os.path.basename(f))
    if not os.path.exists(g):
        continue
    a, b = json.load(open(g)), json.load(open(f))
    FAM8 = ["periodic", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
    nl = lambda r: float(np.mean([r["bce2h." + x] for x in FAM8]))
    out["pairs"].append(dict(run=os.path.basename(f)[:-5], gpu=a["bce_2nd_half"], cpu=b["bce_2nd_half"],
                             absdiff=abs(a["bce_2nd_half"] - b["bce_2nd_half"]), absdiff_nolfsr=abs(nl(a) - nl(b)),
                             maxload_gpu=a.get("max_load"), maxload_cpu=b.get("max_load")))
d = np.array([p["absdiff"] for p in out["pairs"]]); dn = np.array([p["absdiff_nolfsr"] for p in out["pairs"]])
out["n"] = len(d)
out["absdiff_median"], out["absdiff_max"], out["absdiff_mean"] = float(np.median(d)), float(d.max()), float(d.mean())
out["absdiff_nolfsr_median"], out["absdiff_nolfsr_max"] = float(np.median(dn)), float(dn.max())
json.dump(out, open(os.path.join(H, "results/noise_floor.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "pairs"})
for p in out["pairs"]:
    print(p["run"], round(p["absdiff"], 4))
