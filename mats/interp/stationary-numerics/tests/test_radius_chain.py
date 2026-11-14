"""Convergence and correctness tests for lib/radius_chain.py.
1. Grid refinement: quantiles of Q at step h and h/2 agree to 1e-9 relative (spectral convergence).
2. Domain: widening [umin, umax] by 10 units each side changes quantiles by < 1e-9 relative.
3. Mass conservation of the Nystrom operator away from the right edge: |column mass - 1| < 1e-9.
4. Kesten constant: x P(Q > x) at x = 1e7 n equals C_n = 1/E[A log A] within 1e-5 relative.
5. Monte Carlo: direct simulation of Q' = AQ + B (n = 4, 8, 32) matches the CDF at the solver's 10/50/90% quantiles
   within 4 batch-means standard errors.
6. Reduction Q = (S/n) r: simulation of the r-chain with an independent chi2_n factor matches the same CDF.
Run: python tests/test_radius_chain.py  (exit 0 = all pass)."""
import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
from radius_chain import RadiusLaw, kesten_C, simulate_Q

QS = (0.1, 0.5, 0.9, 0.99)
fails = []


def check(name, ok, info):
    print(('PASS ' if ok else 'FAIL ') + name + ': ' + info, flush=True)
    if not ok:
        fails.append(name)


for n in (2, 3, 4, 8, 64, 256):
    t0 = time.time()
    R1 = RadiusLaw(n)
    R2 = RadiusLaw(n, h=R1.h / 2)
    R3 = RadiusLaw(n, h=R1.h, umin=R1.umin - 10, umax=R1.umax + 10)
    q1 = np.array([R1.Q_quantile(q) for q in QS])
    q2 = np.array([R2.Q_quantile(q) for q in QS])
    q3 = np.array([R3.Q_quantile(q) for q in QS])
    d12 = float(np.max(np.abs(q2 / q1 - 1)))
    d13 = float(np.max(np.abs(q3 / q1 - 1)))
    check(f'n={n} refine', d12 < 1e-9, f'max rel diff {d12:.2e} (h={R1.h:.4f}, N={R1.u.size})')
    check(f'n={n} domain', d13 < 1e-9, f'max rel diff {d13:.2e}')
    inner = R1.u < R1.umax - 6
    cm = float(np.max(np.abs(R1.colmass[inner] - 1)))
    check(f'n={n} mass', cm < 1e-9, f'max |colmass-1| {cm:.2e}')
    C = kesten_C(n)
    xt = 1e7 * n
    k = xt * R2.Q_sf(xt)
    check(f'n={n} Kesten', abs(k / C - 1) < 1e-5, f'x P(Q>x) = {k:.8f} vs C_n = {C:.8f} (rel {k / C - 1:.1e})')
    print(f'   n={n}: quantiles {np.round(q2, 6).tolist()}  ({time.time() - t0:.1f}s)', flush=True)

for n in (4, 8, 32):
    R = RadiusLaw(n)
    qv = [R.Q_quantile(q) for q in QS[:3]]
    X = simulate_Q(n, T=4000, P=2000, seed=1)
    nb = 20
    for qq, x in zip(QS[:3], qv):
        ind = (X <= x).astype(float)
        bm = ind.reshape(nb, -1).mean(1)
        est, se = bm.mean(), bm.std(ddof=1) / math.sqrt(nb)
        check(f'n={n} MC q={qq}', abs(est - qq) < 4 * se + 1e-4, f'MC P(Q<=x_q) = {est:.5f} +- {se:.5f}')
    # reduction: r-chain times independent chi2_n / n
    rng = np.random.default_rng([7, n])
    P, T = 2000, 4000
    r = np.full(P, 0.58 * n)
    acc = []
    for t in range(40 * n + T):
        M = rng.binomial(n, 0.5, P)
        A = (2.0 / n) * rng.chisquare(np.maximum(M, 1)) * (M > 0)
        r = 1.0 + A * r
        if t >= 40 * n:
            acc.append(r * rng.chisquare(n, P) / n)
    Y = np.array(acc)
    for qq, x in zip(QS[:3], qv):
        bm = (Y <= x).astype(float).reshape(nb, -1).mean(1)
        est, se = bm.mean(), bm.std(ddof=1) / math.sqrt(nb)
        check(f'n={n} reduction q={qq}', abs(est - qq) < 4 * se + 1e-4, f'r-chain P(Q<=x_q) = {est:.5f} +- {se:.5f}')

print('FAILS:', fails)
sys.exit(1 if fails else 0)
