"""Tests for lib/acts.py: exact Hermite spectra against quadrature, the logistic scale mixture, Mehler sums against
Monte Carlo, gains. usage: python3 tests/test_acts.py   (prints PASS/FAIL lines; exit 1 on any failure)"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import acts as A

fails = 0


def check(name, ok, info=''):
    global fails
    print(('PASS' if ok else 'FAIL'), name, info)
    fails += (not ok)


# 1. logistic = normal scale mixture with sigma = 2K
sig, w = A.kol_quad()
x = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 7.0])
mix = np.array([np.sum(w * np.exp(-0.5 * (xi / sig) ** 2) / (A.SQ2PI * sig)) for xi in x])
logi = np.exp(-x) / (1 + np.exp(-x)) ** 2
check('logistic mixture density', np.max(np.abs(mix / logi - 1)) < 1e-6, f'max rel err {np.max(np.abs(mix / logi - 1)):.2e}')
check('mixing variance pi^2/3', abs(np.sum(w * sig ** 2) - math.pi ** 2 / 3) < 1e-6, f'{np.sum(w * sig ** 2):.8f}')

# 2. closed forms / mixtures against direct quadrature (K = 60; smooth integrands converge)
K = 60
for name, p in [('relu', None), ('gsrelu', 0.5), ('gelu', None), ('softplus', 1.0), ('softplus', 4.0),
                ('silu', None), ('tanh', None)]:
    phi, _ = A.act(name, p)
    for s in (0.7, 1.4142, 3.0):
        ce = A.coef(name, p, K, s)
        cq = A.quad_coef(phi, K, s, Q=2000 if name == 'relu' else 4000)
        err = np.max(np.abs(ce - cq)) / np.max(np.abs(cq))
        tol = 2e-3 if name == 'relu' else 1e-7    # relu: quadrature converges slowly on the kink, not the formula
        check(f'coef {A.label(name, p)} s={s}', err < tol, f'max rel err {err:.1e}')

# 3. Parseval: sum c_k^2 = E phi(sZ)^2 (to the truncation tail)
for name, p in [('gelu', None), ('softplus', 2.0), ('tanh', None), ('silu', None)]:
    phi, _ = A.act(name, p)
    xg, wg = A.gh_nodes(400)
    c = A.coef(name, p, 300, 1.4142)
    e2 = wg @ phi(1.4142 * xg) ** 2
    check(f'Parseval {A.label(name, p)}', abs(np.sum(c ** 2) / e2 - 1) < 1e-6, f'{np.sum(c ** 2) / e2 - 1:.1e}')

# 4. Mehler: E phi(sX) phi(sY) at correlation t against Monte Carlo
rng = np.random.default_rng(0)
N = 4_000_000
X = rng.standard_normal(N); Y0 = rng.standard_normal(N)
for name, p in [('softplus', 4.0), ('gelu', None), ('tanh', None)]:
    phi, _ = A.act(name, p)
    c = A.coef(name, p, 400, 1.4142)
    for t in (0.3, 0.9, 0.99):
        Y = t * X + math.sqrt(1 - t * t) * Y0
        mc = phi(1.4142 * X) * phi(1.4142 * Y)
        se = mc.std() / math.sqrt(N)
        me = np.sum(c ** 2 * t ** np.arange(401))
        check(f'Mehler {A.label(name, p)} t={t}', abs(me - mc.mean()) < 4 * se, f'diff {me - mc.mean():.2e} se {se:.1e}')

# 5. gains: relu He gain 2, chi 1
g2, chi = A.gain_for_q('relu', None, 2.0)
check('relu gain', abs(g2 - 2) < 1e-6 and abs(chi - 1) < 1e-6, f'g2 {g2:.6f} chi {chi:.6f}')
for name, p in [('softplus', 1.0), ('softplus', 16.0), ('gelu', None), ('silu', None), ('smoothstep', 1.0)]:
    g2, chi = A.gain_for_q(name, p, 2.0)
    print('INFO gain', A.label(name, p), f'g2 {g2:.4f} chi1 {chi:.4f}')

print('ALL PASS' if fails == 0 else f'{fails} FAILURES')
sys.exit(1 if fails else 0)
