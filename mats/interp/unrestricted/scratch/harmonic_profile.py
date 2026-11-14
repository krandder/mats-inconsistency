"""Low-degree harmonic content of the per-net angular function g_W on S^{n-1} (design ceiling, FORMAL 2).
For each net: fractions f1, f2 of the angular variance sum_j Var_u g_j carried by spherical-harmonic degrees 1 and 2,
estimated without bias by the inner product of two independent half-sample means:
  degree 1: n |E[g_j(v) v]|^2;  degree 2: (n(n+2)/2) |E[g_j(v)(v v^T - I/n)]|_F^2.
A t-design that is MC-like above degree t gains at most 1/(1 - f1 - ... - ft) over radius-exact MC (heuristic ceiling).
Usage: harmonic_profile.py n L seeds(comma) [--m 18]
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net, sphere_from_gauss  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seeds')
ap.add_argument('--m', type=int, default=18)
ap.add_argument('--chunk', type=int, default=4096)
a = ap.parse_args()
n, L = a.n, a.L
res = dict(n=n, L=L, m=a.m, nets=[])
for seed in [int(s) for s in a.seeds.split(',')]:
    t0 = time.time()
    net = Net(seed, n, L)
    rng = np.random.default_rng([4242, seed, n, L])
    half = []
    for h in range(2):
        N = 1 << (a.m - 1)
        s_g = np.zeros(n); s_g2 = np.zeros(n)
        s1 = np.zeros((n, n))            # sum g_j v_i
        J = min(n, 16)                   # degree 2 on the first J outputs only (cost n^2 per output)
        s2 = np.zeros((J, n * n))        # sum g_j (v v^T - I/n)
        for i in range(0, N, a.chunk):
            U = sphere_from_gauss(rng.standard_normal((min(a.chunk, N - i), n)))
            g = net.forward(U, noise=False)
            s_g += g.sum(0); s_g2 += (g * g).sum(0)
            s1 += g.T @ U
            Q = (U[:, :, None] * U[:, None, :]).reshape(len(U), n * n)
            Q[:, ::n + 1] -= 1.0 / n
            s2 += g[:, :J].T @ Q
        half.append((s_g / N, s_g2 / N, s1 / N, s2 / N))
    (m0, q0, a0, b0), (m1, q1, a1, b1) = half
    J = b0.shape[0]
    varj = 0.5 * (q0 + q1) - m0 * m1                 # unbiased angular variance per output
    var = float(np.sum(varj)); varJ = float(np.sum(varj[:J]))
    e1 = float(n * np.sum(a0 * a1)); e1J = float(n * np.sum(a0[:J] * a1[:J]))
    e2J = float(n * (n + 2) / 2 * np.sum(b0 * b1))
    f1, f1J, f2 = e1 / var, e1J / varJ, e2J / varJ
    res['nets'].append(dict(seed=seed, ang_var=var, f1=f1, f1_J=f1J, f2_J=f2, J=J, seconds=time.time() - t0))
    print(f'n={n} L={L} seed={seed}: angular var {var:.3e}, f1={f1:.3f} (first {J}: {f1J:.3f}), f2={f2:.3f}, '
          f'ceiling(t=2)={1 / max(1e-9, 1 - f1J - f2):.2f} {time.time() - t0:.0f}s', flush=True)
out = os.path.join(os.path.dirname(__file__), '..', 'runs', f'harm_n{n}_L{L}_s{a.seeds.split(",")[0]}.json')
with open(out, 'w') as f:
    json.dump(res, f)
