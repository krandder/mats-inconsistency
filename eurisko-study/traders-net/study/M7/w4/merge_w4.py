"""Collect one directory per run (GPU-synced w4/gpu/results first, else local CPU /tmp/traders-net/M7w4/local),
run analyze_w4 on the union, then write paired differences vs deepseek (same E, same seed) and the
CPU-vs-VM cross-check for runs present in both places.

Outputs (w4/): runs.csv, summary.csv (from analyze_w4), paired.csv, crosscheck.csv, sources.csv.
"""
import csv, glob, os, shutil, subprocess, sys
import numpy as np

W4 = os.path.dirname(os.path.abspath(__file__))
GPU = os.path.join(W4, "gpu", "results")
LOC = "/tmp/traders-net/M7w4/local"
U = "/tmp/traders-net/M7w4/union/results"
os.makedirs(U, exist_ok=True)
for f in glob.glob(os.path.join(U, "*")):
    os.unlink(f)
src = {}
for base, tag in ((LOC, "cpu-local"), (GPU, "vm-cpu")):   # later wins -> VM preferred
    for d in glob.glob(os.path.join(base, "*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "info.json")) and os.path.exists(os.path.join(d, "raw.npz")):
            src[os.path.basename(d)] = (d, tag)
for n, (d, tag) in src.items():
    os.symlink(d, os.path.join(U, n))
with open(os.path.join(W4, "sources.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["run", "source"]); [w.writerow([n, t]) for n, (d, t) in sorted(src.items())]
subprocess.run([sys.executable, os.path.join(W4, "analyze_w4.py"), U], check=True)
for f in ("runs.csv", "summary.csv"):
    shutil.move(os.path.join(U, "..", f), os.path.join(W4, f))

rows = list(csv.DictReader(open(os.path.join(W4, "runs.csv"))))
key = lambda r: (r["E"], r["seed"])
ds = {key(r): r for r in rows if r["arm"] == "deepseek"}
out = []
groups = {}
for r in rows:
    if r["arm"] == "deepseek" or key(r) not in ds:
        continue
    groups.setdefault((r["E"], r["arm"], r["beta"]), []).append(r)
for (E, arm, beta), rs in sorted(groups.items()):
    d = np.array([float(r["bce_last"]) - float(ds[key(r)]["bce_last"]) for r in rs])
    rel = np.array([(float(r["bce_last"]) / float(ds[key(r)]["bce_last"]) - 1) for r in rs])
    out.append({"E": E, "arm": arm, "beta": beta, "n": len(rs),
                "d_bce_last_mean": d.mean(), "d_bce_last_sd": d.std(ddof=1) if len(d) > 1 else np.nan,
                "rel_mean": rel.mean(), "n_lower": int((d < 0).sum()),
                "n_collapsed(max_load>0.5)": int(sum(float(r["max_load"]) > 0.5 for r in rs))})
with open(os.path.join(W4, "paired.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0].keys())); w.writeheader(); w.writerows(out)
for o in out:
    print(o)

# CPU-vs-VM cross-check where both exist
cc = []
for d in glob.glob(os.path.join(LOC, "*", "raw.npz")):
    n = os.path.basename(os.path.dirname(d)); g = os.path.join(GPU, n, "raw.npz")
    if os.path.exists(g):
        a, b = np.load(d), np.load(g)
        za, zb = a["Z"].astype(float), b["Z"].astype(float)
        cc.append({"run": n, "max_abs_dz": float(np.abs(za - zb).max()),
                   "routing_equal_frac": float((a["cnt_fe"] == b["cnt_fe"]).mean())})
if cc:
    with open(os.path.join(W4, "crosscheck.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, list(cc[0].keys())); w.writeheader(); w.writerows(cc)
    print(cc)
