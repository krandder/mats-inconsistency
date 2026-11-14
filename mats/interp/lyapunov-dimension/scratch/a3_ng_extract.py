"""A3: read the nongauss lane's per-net results (read-only) and tabulate nMSE at the output layer for each family,
cut and net, with the EMP floor. Prints per-net values and the mean over nets.
python a3_ng_extract.py <glob of e1_*.json> ..."""
import glob, json, sys
import numpy as np

files = []
for pat in sys.argv[1:]:
    files += sorted(glob.glob(pat))
tab = {}
for fn in files:
    d = json.load(open(fn))
    n, L, seed = d['n'], d['L'], d['seed']
    for key, rec in d['runs'].items():
        rep, cut, fam = key.split('/')
        nm = rec['nmse']
        out = nm[-1]                       # last entry = output layer L
        tab.setdefault((n, L, int(cut)), {}).setdefault(fam, {})[seed] = (out, len(nm), rec.get('floor', [None])[-1])
for (n, L, cut), fams in sorted(tab.items()):
    print(f"\n== {n} x {L}, cut {cut} (nMSE of E[a_L] at L = {L}; per net; mean) ==")
    seeds = sorted({s for f in fams.values() for s in f})
    for fam in sorted(fams, key=lambda f: (f.rstrip('0123456789'), int(''.join(c for c in f if c.isdigit()) or 0))):
        vals = [fams[fam].get(s, (np.nan,))[0] for s in seeds]
        print(f"  {fam:7s} " + " ".join(f"{v:9.2e}" for v in vals) + f"   mean {np.nanmean(vals):9.2e}")
    print("  nets: " + " ".join(str(s) for s in seeds) + f"; series length {list(fams.values())[0][seeds[0]][1]}")
