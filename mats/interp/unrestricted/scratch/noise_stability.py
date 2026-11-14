"""Noise stability of the per-net mean mu(W) in the weights (low-degree class bound, FORMAL 4.1).
For K base nets W (seeds s) and independent W' (seeds s + 500000), W_rho = rho W + sqrt(1 - rho^2) W', rho = 1 - delta.
mu is estimated by radius-exact scrambled Sobol (2^m points, one scramble shared by the pair: common random numbers).
Reports, per delta: D(delta) = E|mu(W) - mu(W_rho)|^2 / (2 S1) = 1 - S(rho)/S(1), with S1 = sum_j Var_W mu_j;
the ratio S1 / E_W V (annealed-estimate worth); and the class bound N_D = E V / (S1 - min_rho S(rho) rho^-D).
Usage: noise_stability.py n L K [--m 14] [--seed0 700000]
"""
import argparse
import json
import math
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import make_weights, mean_radius, sobol_directions  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('K', type=int)
ap.add_argument('--m', type=int, default=14)
ap.add_argument('--seed0', type=int, default=700000)
ap.add_argument('--deltas', default='1e-5,1e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1')
a = ap.parse_args()
n, L = a.n, a.L
deltas = [float(x) for x in a.deltas.split(',')]
Er = mean_radius(n)
t0 = time.time()


def mu_and_v(W, U):
    h = U
    for Wl in W:
        h = np.maximum(h @ Wl.T, 0.0)
    g1 = h.mean(0)
    g2 = (h * h).mean(0)
    mu = Er * g1
    V = float(np.sum(n * g2 - Er ** 2 * g1 ** 2))
    return mu, V


base_mu, base_V = [], []
pair = {d: [] for d in deltas}
for i in range(a.K):
    s = a.seed0 + i
    W = make_weights(s, n, L)
    Wp = make_weights(s + 500000, n, L)
    U = sobol_directions(n, a.m, scramble=True, seed=s)
    mu0, V0 = mu_and_v(W, U)
    base_mu.append(mu0); base_V.append(V0)
    for d in deltas:
        rho = 1.0 - d
        c = math.sqrt(max(0.0, 1.0 - rho * rho))
        Wr = [rho * A + c * B for A, B in zip(W, Wp)]
        mur, _ = mu_and_v(Wr, U)
        pair[d].append(float(np.sum((mu0 - mur) ** 2)))
base_mu = np.array(base_mu)
S1 = float(np.sum(base_mu.var(0, ddof=1)))
EV = float(np.mean(base_V))
res = dict(n=n, L=L, K=a.K, m=a.m, S1=S1, EV=EV, annealed_worth=EV / S1, deltas=deltas)
rows = []
for d in deltas:
    x = np.array(pair[d])
    Dd = float(x.mean() / (2 * S1))
    se = float(x.std(ddof=1) / math.sqrt(len(x)) / (2 * S1))
    rows.append(dict(delta=d, rho=1 - d, D=Dd, D_se=se, S_rho=S1 * (1 - Dd), dmu2_over_EV=float(x.mean() / EV)))
res['rows'] = rows
# class bound: low-degree mass <= S(rho)/rho^D, minimised over measured rho in (0, 1)
bounds = {}
for D in (1, 2, 4, 8, 16, 32, 64, 128):
    best = min(r['S_rho'] / (r['rho'] ** D) for r in rows if 0 < r['rho'] < 1)
    tail = max(S1 - best, 0.0)
    bounds[D] = dict(low_mass_upper=best / S1, tail_lower=tail / S1, worth_samples_max=(EV / tail) if tail > 0 else None)
res['degree_bounds'] = bounds
res['seconds'] = time.time() - t0
out = os.path.join(os.path.dirname(__file__), '..', 'runs', f'nstab_n{n}_L{L}_K{a.K}.json')
with open(out, 'w') as f:
    json.dump(res, f)
print(f'n={n} L={L} K={a.K}: S1/EV={S1 / EV:.3f} ' + ' '.join(f'{r["delta"]:g}:{r["D"]:.3g}' for r in rows) +
      ' | worth<=' + ' '.join(f'D{D}:{(b["worth_samples_max"] or float("inf")):.3g}' for D, b in bounds.items()) +
      f' {time.time() - t0:.0f}s', flush=True)
