"""Aggregate k2_scaling results: K = 2 (and K = 1) nMSE and gain over plain Monte Carlo at equal cost (ARC's FLOP count),
along depth = alpha * width, alpha in {1/8, 1/4, 1/2}. Prints a markdown table and log-log slopes.
usage: analyze_k2s.py"""
import glob, json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
best = {}
for p in sorted(glob.glob(os.path.join(HERE, 'results', 'k2s_n*_L*_seed*.json'))):
    d = json.load(open(p))
    key = (d['n'], d['seed'])          # same seed = same net (truncation convention): keep the deepest run only
    if key not in best or d['Lmax'] > best[key]['Lmax']:
        best[key] = d
for (n, seed), d in sorted(best.items()):
    rows.setdefault(n, []).append(d)

rng = np.random.default_rng(0)


def boot_ci(x, f=np.mean, B=4000):
    x = np.asarray(x)
    bs = [f(rng.choice(x, len(x))) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


print("| alpha | n | depth | seeds | K=2 nMSE median | K=2 nMSE mean [95% CI] | MC 1/N_eq | gain = (1/N_eq)/nMSE: geo-mean [95% CI] | K=1 nMSE median |")
print("|---|---|---|---|---|---|---|---|---|")
fits = {}
for alpha in (1 / 8, 1 / 4, 1 / 2):
    xs, ys, ym = [], [], []
    for n in sorted(rows):
        l = int(round(alpha * n))
        ds = [d for d in rows[n] if d['Lmax'] >= l and l >= 1]
        if len(ds) < 2:
            continue
        k2 = np.array([d['k2']['nmse'][l - 1] for d in ds])
        k1 = np.array([d['k1']['nmse'][l - 1] for d in ds])
        neq = np.mean([d['N_equiv_k2'] for d in ds])
        gains = (1.0 / neq) / np.maximum(k2, 1e-12)
        lg = np.log(gains)
        ci_g = np.exp(boot_ci(lg))
        ci_m = boot_ci(k2)
        print(f"| 1/{int(1 / alpha)} | {n} | {l} | {len(ds)} | {np.median(k2):.2e} | {k2.mean():.2e} [{ci_m[0]:.1e}, {ci_m[1]:.1e}] | "
              f"{1 / neq:.2e} | {math.exp(lg.mean()):.2f} [{ci_g[0]:.2f}, {ci_g[1]:.2f}] | {np.median(k1):.2e} |")
        xs.append(math.log(n)); ys.append(math.log(np.median(k2))); ym.append(math.log(k2.mean()))
    if len(xs) >= 3:
        b_med = np.polyfit(xs, ys, 1)[0]
        b_mean = np.polyfit(xs, ym, 1)[0]
        fits[alpha] = (b_med, b_mean)
print()
for a, (bm, bmean) in fits.items():
    print(f"alpha = 1/{int(1 / a)}: slope of log nMSE_K2 vs log n: median {bm:.2f}, mean {bmean:.2f} "
          f"(plain MC at equal cost: -1; K = 2 keeps pace iff slope <= -1)")


def depth_table():
    depths = [4, 8, 16, 32, 64, 128, 256, 512]
    print()
    print("| width n (seeds) | " + " | ".join(f"depth {l}" for l in depths) + " |")
    print("|---|" + "---|" * len(depths))
    for n in sorted(rows):
        ds = rows[n]
        cells = []
        for l in depths:
            dd = [d for d in ds if d['Lmax'] >= l]
            if not dd or l > n // 2 and not any(d['Lmax'] >= l for d in ds):
                cells.append("-"); continue
            v = np.median([d['k2']['nmse'][l - 1] * d['N_equiv_k2'] for d in dd])
            cells.append(f"{v:.2f}")
        print(f"| {n} ({len(ds)}) | " + " | ".join(cells) + " |")


depth_table()
