"""wm2 audit, phase 2 extras (after TEST): (1) boards this round's rbtd8 needs to reach this round's k0 10k-board KL
(same log-linear interpolation and seed bootstrap as final2.py); (2) round-1 vs round-2 TEST KL of the same baseline
config on the same seeds (run-to-run variation); (3) per-stage TEST KL at N = 1000.
  PYTHONDONTWRITEBYTECODE=1 python3 audit/final_extra.py"""
import glob, json, math, os
import numpy as np
RES = '/home/agent/workspaces/eurisko-study/li-net/wm2/results'
assert os.path.exists(os.path.join(RES, 'test-t1-final.json')), 'phase 1: TEST closed'
NS = [100, 300, 1000, 3000, 10000]
def load(path):
    out = {}
    for p in glob.glob(path):
        r = json.load(open(p))
        if 'test' in r: out[(r['n'], r['seed'])] = r
    return out
def n_to_reach(curve, target):
    pts = sorted(curve.items())
    if pts[0][1] <= target: return float(pts[0][0])
    for (n0, k0), (n1, k1) in zip(pts, pts[1:]):
        if k1 <= target:
            f = (math.log(k0) - math.log(target)) / (math.log(k0) - math.log(k1))
            return float(math.exp(math.log(n0) + f * (math.log(n1) - math.log(n0))))
    return float('inf')
c = load('/tmp/wm2/runs/F/t1/F-WMr-D-rbtd8/*.json'); b = load('/tmp/wm2/runs/F/t1/F-B-rbtd8/*.json'); k = load('/tmp/wm2/runs/F/t1/F-B-k0/*.json')
rng = np.random.default_rng(0)
for name, arm in (('champion', c), ('rbtd8 (this round)', b)):
    sa = {n: [arm[(n, s)]['test']['kl'] for s in range(5)] for n in NS}
    sb = [k[(10000, s)]['test']['kl'] for s in range(5)]
    frac = n_to_reach({n: np.mean(sa[n]) for n in NS}, np.mean(sb)) / 1e4
    boots = [n_to_reach({n: np.mean(rng.choice(sa[n], 5)) for n in NS}, np.mean(rng.choice(sb, 5))) / 1e4 for _ in range(2000)]
    boots = np.where(np.isfinite(boots), boots, 1e3)
    print(f"(1) {name}: fraction of 10k boards to reach k0's 10k-board TEST KL = {frac:.3f} [{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]")
R1 = '/home/agent/workspaces/eurisko-study/li-net/smallnet/results/raw/F/'
for tag1, arm, nm in (('F-E17-rbtd8', b, 'rbtd8'), ('F-k0', k, 'k0')):
    r1 = load(R1 + tag1 + '/*.json')
    for n in NS:
        d = [1e3 * (arm[(n, s)]['test']['kl'] - r1[(n, s)]['test']['kl']) for s in range(5) if (n, s) in r1]
        m = np.mean(d); h = 2.776 * np.std(d, ddof=1) / math.sqrt(len(d))
        print(f"(2) {nm} N={n}: round-2 minus round-1 TEST KL x1e3, same seeds and config: {m:+.3f} +- {h:.3f} (per seed {[round(x, 3) for x in d]}); "
              f"means {1e3 * np.mean([r1[(n, s)]['test']['kl'] for s in range(5)]):.3f} -> {1e3 * np.mean([arm[(n, s)]['test']['kl'] for s in range(5)]):.3f}")
for name, arm in (('champion', c), ('rbtd8', b), ('k0', k)):
    st = np.mean([arm[(1000, s)]['test']['kl_stage'] for s in range(5)], 0)
    print(f"(3) {name} N=1000 TEST KL per stage x1e3: {[round(1e3 * x, 3) for x in st]}")
