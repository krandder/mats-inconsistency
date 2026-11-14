"""Amendment 2D (reconciliation with the stationary-numerics lane): mean interaction order of the SAME noise-regime window
integrand as Amendment 2B (partB.py rqmc setting: Model S, sigma = 1, burn-in B = 8n from h = 0, a = relu(W_1 h_0), window =
the d = min(B, 21201 // n) most recent layers), with two variable groupings:
  D_coord = (d n) E|a - a^(j)|^2 / 2 / V_win,  j a uniformly chosen scalar window coordinate, refreshed (all else shared);
  D_block = d E|a - a^(l)|^2 / 2 / V_win,      l a uniformly chosen window layer, its whole n-vector refreshed;
  V_win   = E|a - a^(W)|^2 / 2                 (the whole window refreshed, older noise shared).
Both are consistent ratio estimators of the variance-weighted mean dimension (Owen) over the respective variable groups;
D_block is the numerics lane's estimand (there: window 4n + 1 layers after a 12n deep past). Batches of 1024, SE over batches.
usage: amend2d_meandim.py n seed [--P 4096] [--out file]"""
import argparse, json, math, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--P', type=int, default=4096); ap.add_argument('--out')
a = ap.parse_args()
n, seed = a.n, a.seed
t0 = time.time()
B = 8 * n
d = min(B, 21201 // n)
Wc = st.WCache(seed, n, max_bytes=1.5e9)
CHK = 1024
nb = a.P // CHK
rows = []
for b in range(nb):
    rng = np.random.default_rng([7701, seed, n, b])
    jl = rng.integers(0, d, CHK)                                        # refreshed layer age (block refresh)
    jc_age = rng.integers(0, d, CHK); jc_unit = rng.integers(0, n, CHK)  # refreshed scalar coordinate
    H = np.zeros((4, CHK, n))                                           # base, coordinate, block, window refresh
    idx = np.arange(CHK)
    for l in range(-B + 1, 1):
        age = -l
        W = Wc(l)
        x = rng.standard_normal((CHK, n))
        xs = [x, x.copy(), x.copy(), x]
        if age < d:
            m = jc_age == age
            xs[1][idx[m], jc_unit[m]] = rng.standard_normal(int(m.sum()))
            mb = jl == age
            xs[2][mb] = rng.standard_normal((int(mb.sum()), n))
            xs[3] = rng.standard_normal((CHK, n))
        for i in range(4):
            H[i] = np.maximum(H[i] @ W.T, 0.0) + xs[i]
    W1 = Wc(1)
    A = [np.maximum(H[i] @ W1.T, 0.0) for i in range(4)]
    half = lambda X, Y: 0.5 * np.sum((X - Y) ** 2, 1)
    rows.append(dict(c=float(half(A[0], A[1]).mean()), blk=float(half(A[0], A[2]).mean()), win=float(half(A[0], A[3]).mean()),
                     V=float(np.sum(A[0].var(0, ddof=1)))))
    print(f'batch {b + 1}/{nb} {time.time() - t0:.0f}s', flush=True)
c = np.array([r['c'] for r in rows]); bl = np.array([r['blk'] for r in rows]); w = np.array([r['win'] for r in rows])
Dc = d * n * c.mean() / w.mean(); Db = d * bl.mean() / w.mean()
Dcb = d * n * c / w.mean(); Dbb = d * bl / w.mean()
res = dict(n=n, seed=seed, B=B, d_layers=d, P=nb * CHK, D_coord=float(Dc), D_coord_se=float(Dcb.std(ddof=1) / math.sqrt(nb)),
           D_block=float(Db), D_block_se=float(Dbb.std(ddof=1) / math.sqrt(nb)), V_win=float(w.mean()),
           V=float(np.mean([r['V'] for r in rows])), sec=time.time() - t0)
print(res, flush=True)
if a.out:
    json.dump(res, open(a.out, 'w'))
