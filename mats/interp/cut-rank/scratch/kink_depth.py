"""Conjecture C2 check: does the kink (1-t)^{3/2} coefficient of the finite-width back kernel grow with depth?

For a back of width n and depth d, f(u) = g(sqrt(n) u) averaged over output neurons has E|f(u) - f(u')|^2 / theta^2 ->
1 as theta -> 0 (averaged gradient identity). The kink makes it smaller: at infinite width
kappa_d(1 - s) = 1 - s + d c s^{3/2}, c = 2 sqrt 2 / (3 pi), so the relative deficit
def(theta) = 1 - E|Delta f|^2 / theta^2 ~ (2 d / (3 pi)) theta = 0.212 d theta for theta << 1/d.
We estimate def(theta)/theta at several theta and depths with B backs (one random base point and tangent per back).
usage: python3 scratch/kink_depth.py n B
"""
import sys, math
import numpy as np

n = int(sys.argv[1]); B = int(sys.argv[2])
rng = np.random.default_rng(20261031 + n)
DEPTHS = (1, 2, 4, 8, 16, 32)
THETAS = (0.01, 0.02, 0.05, 0.1)
s = math.sqrt(2.0 / n)
print(f'n={n} B={B}: relative deficit 1 - E|df|^2/theta^2 divided by theta (infinite-width prediction 0.212 d)')
for d in DEPTHS:
    acc = {t: [] for t in THETAS}
    for chunk in range(B // 1000):
        m = 1000
        u = rng.standard_normal((m, n)); u /= np.linalg.norm(u, axis=1, keepdims=True)
        v = rng.standard_normal((m, n)); v -= (v * u).sum(1, keepdims=True) * u; v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts = [u] + [math.cos(t) * u + math.sin(t) * v for t in THETAS]
        X = np.stack(pts, 1) * math.sqrt(n)               # (m, 1+T, n)
        for _ in range(d):
            W = rng.standard_normal((m, n, n)) * s
            X = np.maximum(np.einsum('mij,mpj->mpi', W, X), 0)
        f0 = X[:, 0]
        for i, t in enumerate(THETAS):
            df2 = ((X[:, i + 1] - f0) ** 2).mean(1)        # average over output neurons
            acc[t].append(df2 / (t * t))
    row = []
    for t in THETAS:
        a = np.concatenate(acc[t])
        mean, se = a.mean(), a.std() / math.sqrt(len(a))
        row.append(f'th={t}: {(1 - mean) / t:7.3f} +- {se / t:.3f}')
    print(f'd={d:2d} | ' + ' | '.join(row) + f' | pred {0.212 * d:.3f}', flush=True)
