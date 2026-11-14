"""Validation of lib/gram_chain.py against full-network replicas (weights drawn fresh per sample: the annealed law).
Both are exact samplers of the same law, so every comparison must agree within Monte Carlo error.
Scenarios (window of J layers after a shared state with stationary radius Q0):
  A: independent noise in every window layer (the cloud);  B: noise shared except one layer at age j (single refresh);
  C: R = 4 replicas, pairs (0,1) share nothing, (0,2) share everything except the last layer.
Checks: two-sample KS p-value > 1e-3 and |mean difference| < 4 SE for each functional.
Run: python tests/test_gram_chain.py (exit 0 = pass)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import numpy as np
from scipy.stats import ks_2samp
import gram_chain as gc

fails = []


def check(name, x, y):
    p = ks_2samp(x, y).pvalue
    se = math.sqrt(x.var() / x.size + y.var() / y.size)
    dm = abs(x.mean() - y.mean())
    ok = p > 1e-3 and dm < 4 * se
    print(('PASS ' if ok else 'FAIL ') + f'{name}: KS p = {p:.3f}, means {x.mean():.5g} vs {y.mean():.5g} (|d|/SE = {dm / se:.2f})',
          flush=True)
    if not ok:
        fails.append(name)


def net_run(n, Q0, J, shares, rng, R):
    """Full network: P samples, fresh W per sample and layer, R replicas starting at sqrt(n Q0) e_1.
    shares: list over layers of (P, R) label arrays."""
    P = Q0.size
    h = np.zeros((P, R, n))
    h[:, :, 0] = np.sqrt(n * Q0)[:, None]
    for t in range(J):
        W = rng.standard_normal((P, n, n)) * math.sqrt(2.0 / n)
        a = np.maximum(np.einsum('pij,prj->pri', W, h), 0.0)
        lab = shares[t]
        x = np.take_along_axis(rng.standard_normal((P, R, n)), np.broadcast_to(lab[:, :, None], (P, R, n)), axis=1)
        h = a + x
    return np.einsum('pai,pbi->pab', h, h) / n


def gram_run(n, Q0, J, shares, rng, R):
    G = np.repeat(Q0[:, None, None], R, 1).repeat(R, 2).astype(float)
    for t in range(J):
        G = gc.step(G, shares[t], n, rng)
    return G


for n in (8, 32):
    rng = np.random.default_rng([11, n])
    P = 3000
    Q0 = gc.sample_Q_stationary(n, P, rng)
    J = 2 * n
    # A: independent noise
    shA = [np.tile(np.arange(2), (P, 1)) for _ in range(J)]
    Gn = net_run(n, Q0, J, shA, np.random.default_rng([12, n]), 2)
    Gg = gram_run(n, Q0, J, shA, np.random.default_rng([13, n]), 2)
    dn, dg = gc.diffs(Gn, n), gc.diffs(Gg, n)
    for k in ('h', 'u', 'logR', 'Q'):
        check(f'n={n} A {k}', dn[k], dg[k])
    # B: single refresh at age j = n/2 (layer index J - 1 - j)
    jj = J - 1 - n // 2
    shB = [np.tile(np.array([0, 1]) if t == jj else np.array([0, 0]), (P, 1)) for t in range(J)]
    Gn = net_run(n, Q0, J, shB, np.random.default_rng([14, n]), 2)
    Gg = gram_run(n, Q0, J, shB, np.random.default_rng([15, n]), 2)
    dn, dg = gc.diffs(Gn, n), gc.diffs(Gg, n)
    for k in ('h', 'u', 'logR'):
        check(f'n={n} B {k}', dn[k], dg[k])
    # C: 4 replicas; 0,1 independent; 2 = copy of 0 except last layer; 3 independent
    shC = [np.tile(np.array([0, 1, 0, 3]) if t < J - 1 else np.array([0, 1, 2, 3]), (P, 1)) for t in range(J)]
    Gn = net_run(n, Q0, J // 2, shC[J // 2:], np.random.default_rng([16, n]), 4)
    Gg = gram_run(n, Q0, J // 2, shC[J // 2:], np.random.default_rng([17, n]), 4)
    for (i, j) in ((0, 1), (0, 2), (1, 3)):
        dn, dg = gc.diffs(Gn, n, i, j), gc.diffs(Gg, n, i, j)
        check(f'n={n} C pair {i}{j} h', dn['h'], dg['h'])
    # 4-replica cross term (c1-c2).(c3-c4)/n squared: tr C^2 ingredient
    xn = (Gn[:, 0, 2] - Gn[:, 0, 3] - Gn[:, 1, 2] + Gn[:, 1, 3])
    xg = (Gg[:, 0, 2] - Gg[:, 0, 3] - Gg[:, 1, 2] + Gg[:, 1, 3])
    check(f'n={n} C cross', xn, xg)

print('FAILS:', fails)
sys.exit(1 if fails else 0)
