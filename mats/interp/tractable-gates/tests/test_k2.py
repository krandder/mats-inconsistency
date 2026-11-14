"""K2 implementation test: after ONE layer from x ~ N(0, I) the Gaussian closure is exact, so K2's mean and covariance
must match Monte Carlo within sampling error (this tests the Gauss-Hermite moments and the Mehler off-diagonals, not the
closure). Also checks the residual closure after one layer, and that the closed-form and Mehler ReLU agree.
usage: python3 tests/test_k2.py   (exit 1 on failure)"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import gnet as G, k2gen as K2

fails = 0
herm = K2.Hermite(64, 32)
n = 8; N = 1 << 20
for name, p in [('relu', None), ('softplus', 1.0), ('softplus', 16.0), ('gelu', None), ('silu', None), ('tanh', None),
                ('gsrelu', 0.5), ('smoothstep', 1.0)]:
    spec = G.Spec(name, p)
    Ws = G.weights(spec, n, 2, 123, 0)
    hm = K2.Hermite(K2.q_for(name, p), 32)
    outs, fl = K2.k2_plain(name, p, Ws[:1], hm)
    X = np.random.default_rng(1).standard_normal((N, n))
    H = spec.phi(X @ Ws[0][0].T.astype(np.float64))
    mu = H.mean(0); C = np.cov(H.T)
    se_mu = H.std(0) / math.sqrt(N)
    zm = np.max(np.abs(outs[0][0] - mu) / se_mu)
    # covariance: compare entries against their MC standard error (approx sqrt(Var(h_i h_j)/N))
    Hc = H - mu
    se_c = np.sqrt(np.var(Hc[:, :, None] * Hc[:, None, :], axis=0) / N)
    zc = np.max(np.abs(outs[0][1] - C) / se_c)
    ok = zm < 5 and zc < 6
    fails += not ok
    print('PASS' if ok else 'FAIL', f'{spec.label}: max |z| mean {zm:.2f}, cov {zc:.2f}; flops {fl[0]}')
# biased tanh (tanhb, Amendment 1) after one layer
spec = G.Spec('tanh', None, bias_frac=0.13)
Ws = G.weights(spec, n, 2, 321, 0); bs = G.biases(spec, n, 2, 321, 0)
outs, fl = K2.k2_plain('tanh', None, Ws[:1], herm, bs=bs[:1])
X = np.random.default_rng(2).standard_normal((N, n))
H = np.tanh(X @ Ws[0][0].T.astype(np.float64) + bs[0].astype(np.float64)[None, :])
mu = H.mean(0); C = np.cov(H.T); Hc = H - mu
zm = np.max(np.abs(outs[0][0] - mu) / (H.std(0) / math.sqrt(N)))
zc = np.max(np.abs(outs[0][1] - C) / np.sqrt(np.var(Hc[:, :, None] * Hc[:, None, :], axis=0) / N))
ok = zm < 5 and zc < 6 and np.max(np.abs(mu)) > 1e-2
fails += not ok
print('PASS' if ok else 'FAIL', f'tanhb: max |z| mean {zm:.2f}, cov {zc:.2f}; max |mean| {np.max(np.abs(mu)):.3f} (nonzero: not degenerate)')
# closed-form vs Mehler ReLU at layer 2 (both closures; difference = Mehler truncation)
spec = G.Spec('relu'); Ws = G.weights(spec, n, 2, 5, 0)
a, _ = K2.k2_plain('relu', None, Ws, herm, closed_relu=True)
b, _ = K2.k2_plain('relu', None, Ws, herm, closed_relu=False)
d = np.max(np.abs(a[1][1] - b[1][1])) / np.max(np.abs(a[1][1]))
print('INFO relu closed vs Mehler(32) covariance rel diff at layer 2:', f'{d:.1e}')
# residual closure after one layer
rng = np.random.default_rng(3)
W = rng.standard_normal((n, n)) * math.sqrt(2 / n); V = rng.standard_normal((n, n)) * math.sqrt(1 / n)
for name, p in [('relu', None), ('gelu', None), ('softplus', 4.0)]:
    phi = G.Spec(name, p).phi
    mu, C, fl = K2.k2_residual(name, p, [W], [V], herm)
    X = np.random.default_rng(4).standard_normal((N, n))
    H = X + phi(X @ W.T) @ V.T
    zm = np.max(np.abs(mu - H.mean(0)) / (H.std(0) / math.sqrt(N)))
    Hc = H - H.mean(0)
    se_c = np.sqrt(np.var(Hc[:, :, None] * Hc[:, None, :], axis=0) / N)
    zc = np.max(np.abs(C - np.cov(H.T)) / se_c)
    ok = zm < 5 and zc < 6
    fails += not ok
    print('PASS' if ok else 'FAIL', f'residual {name}{"" if p is None else p}: max |z| mean {zm:.2f}, cov {zc:.2f}')
print('ALL PASS' if not fails else f'{fails} FAILURES')
sys.exit(1 if fails else 0)
