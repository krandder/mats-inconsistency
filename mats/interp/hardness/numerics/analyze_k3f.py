"""Aggregate k3f_scaling results: factorized K = 3 and K = 2 against plain Monte Carlo at equal cost (ARC's count),
depth n/8 and n/4. Prints a markdown table.
usage: analyze_k3f.py"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for p in sorted(glob.glob(os.path.join(HERE, 'results', 'k3f_n*_seed*.json'))):
    d = json.load(open(p))
    rows.setdefault(d['n'], []).append(d)
rng = np.random.default_rng(1)


def geo_ci(x, B=4000):
    lx = np.log(np.asarray(x))
    bs = [np.mean(rng.choice(lx, len(lx))) for _ in range(B)]
    return math.exp(lx.mean()), math.exp(np.percentile(bs, 2.5)), math.exp(np.percentile(bs, 97.5))


print("| n | depth (alpha) | seeds | method | nMSE median | N_eq (MC samples at equal cost) | gain over MC: geo-mean [95% CI] |")
print("|---|---|---|---|---|---|---|")
for n in sorted(rows):
    for frac, lab in ((8, '1/8'), (4, '1/4')):
        depth = n // frac
        for meth in ('k2', 'k3f'):
            key = f'{meth}_d{depth}'
            ds = [d['runs'][key] for d in rows[n] if key in d['runs']]
            if not ds:
                continue
            nm = np.array([max(r['nmse'], 1e-12) for r in ds])
            neq = np.mean([r['N_eq'] for r in ds])
            gains = (1.0 / neq) / nm
            g, lo, hi = geo_ci(gains)
            print(f"| {n} | {depth} ({lab}) | {len(ds)} | {meth} | {np.median(nm):.2e} | {neq:.0f} | {g:.2f} [{lo:.2f}, {hi:.2f}] |")
