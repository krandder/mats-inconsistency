"""Per-family prequential BCE on the last 10k stream steps, from the /tmp npz of each run.
Usage: python3 family.py SEED [ARM ...] -> results/family_s{SEED}.csv"""
import csv, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "S0", "substrate"))
from streams_ext import make_streams, FAMILIES  # noqa
seed = int(sys.argv[1]); arms = sys.argv[2:] or ["absag", "taylor", "movement", "magnitude", "cbp", "random", "static128", "dense256"]
T, W = 40000, 32
s = make_streams(Q=64, T=T, seed=seed, heldout=8, regime_every=10000, regime_frac=0.25)
famt = s.fam_t[:, W:].T              # [n_pos, Q] family of each target
tpos = np.arange(T - W) + W
rows = []
for a in arms:
    p = f"/tmp/traders-net/M8w2/z_{a}_s{seed}.npz"
    if not os.path.exists(p):
        continue
    d = np.load(p); Z, Y, M = d["Z"].astype(np.float64), d["Y"], d["M"]
    bce = np.logaddexp(0, Z) - Y * Z
    live = M & np.isfinite(Z) & (tpos >= T - 10000)[:, None]
    row = {"arm": a, "seed": seed}
    for f, name in enumerate(FAMILIES):
        mm = live & (famt == f)
        row[name] = round(float(bce[mm].mean()), 4) if mm.any() else ""
    # lfsr16 learning is bimodal (S0 SUBSTRATE.md); report primaries with lfsr16 targets removed
    allv = M & np.isfinite(Z); nol = famt != 1
    row["lfsr16_learnt"] = int(row["lfsr16"] != "" and row["lfsr16"] < 0.35)
    row["bce_all_nolfsr"] = round(float(bce[allv & nol].mean()), 5)
    sw = [e for e in s.events if e["kind"] == "switch"]
    swm = np.zeros_like(allv)
    for b0 in sorted({e["t"] for e in sw}):
        qm = np.zeros(Z.shape[1], bool); qm[[e["q"] for e in sw if e["t"] == b0]] = True
        swm |= allv & ((tpos >= b0) & (tpos < b0 + 2000))[:, None] & qm[None, :]
    row["postshift_switched_nolfsr"] = round(float(bce[swm & nol].mean()), 5)
    rows.append(row)
with open(os.path.join(HERE, "results", f"family_s{seed}.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["arm", "seed"] + FAMILIES + ["lfsr16_learnt", "bce_all_nolfsr", "postshift_switched_nolfsr"]); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
