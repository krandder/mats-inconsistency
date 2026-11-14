"""Amendment 2A (PREREG): does the noise-regime RQMC gain depend on WHICH Sobol' dimensions carry the integrand (projection
quality at fixed N, a dimension effect) with the integrand itself unchanged (same function, same order structure)?
Setting as work/stationary/scratch/partB.py rqmc mode (Model S, sigma = 1, burn-in B = 8n from h = 0, a = relu(W_1 h_0),
window = the d = min(B, 21201 // n) most recent layers, age 0 first, MC on older layers), except that the window's d n
coordinates are taken from Sobol' dimensions [off, off + d n) of a (off + d n)-dimensional scrambled sequence.
usage: amend2a_offset.py n seed [--offs 0,2048,8192] [--R 16] [--ms 10,12] [--out file]
G(N, off) = V / (N Var_rqmc), V = tr Var(a) from the stationary lane's truth file (seeds 1000/1001) or from part2_noise."""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.stats import qmc
from scipy.special import ndtri
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary/lib')
import st

ap = argparse.ArgumentParser(); ap.add_argument('n', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--offs', default='0,2048,8192'); ap.add_argument('--R', type=int, default=16)
ap.add_argument('--ms', default='10,12'); ap.add_argument('--out')
a = ap.parse_args()
n, seed = a.n, a.seed
t0 = time.time()
B = 8 * n
d = min(B, 21201 // n)
Wc = st.WCache(seed, n, max_bytes=1.5e9)
ST = '/home/agent/workspaces/mats/work/stationary/runs'
IB = '/home/agent/workspaces/mats/work/input-bits/runs'
if os.path.exists(f'{ST}/truth_n{n}_s{seed}.json') and seed < 4000:
    V = json.load(open(f'{ST}/truth_n{n}_s{seed}.json'))['trvar']
else:
    V = json.load(open(f'{IB}/p2noise_n{n}_s{seed}.json'))['V']
res = dict(n=n, seed=seed, B=B, d_layers=d, V=V, rows=[])
for off in [int(x) for x in a.offs.split(',')]:
    if off + d * n > 21201:
        continue
    for m in [int(x) for x in a.ms.split(',')]:
        N = 1 << m
        ests = []
        for r in range(a.R):
            eng = qmc.Sobol(d=off + d * n, scramble=True, seed=np.random.default_rng([8180, seed, n, m, r, off]))
            U = eng.random_base2(m)[:, off:]
            G = ndtri(np.clip(U, 1e-16, 1 - 1e-16))
            rng = np.random.default_rng([8181, seed, n, m, r, off])
            h = np.zeros((N, n))
            for l in range(-B + 1, 1):
                age = -l
                xi = G[:, age * n:(age + 1) * n] if age < d else rng.standard_normal((N, n))
                h = np.maximum(h @ Wc(l).T, 0.0) + xi
            ests.append(np.maximum(h @ Wc(1).T, 0.0).mean(0))
        vq = float(np.sum(np.array(ests).var(0, ddof=1)))
        res['rows'].append(dict(off=off, N=N, G=V / N / vq))
        print(f'off={off} N={N}: G {V / N / vq:.2f} | {time.time() - t0:.0f}s', flush=True)
res['sec'] = time.time() - t0
if a.out:
    json.dump(res, open(a.out, 'w'))
