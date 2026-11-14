"""Tests for lib/st.py and the exact statements used in FORMAL (run: python3 tests/test_st.py)."""
import os, sys, math
import numpy as np
from scipy.special import digamma
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import st


def test_scale_invariance():
    n, seed = 16, 7
    Wf = st.WCache(seed, n)
    h1 = np.random.default_rng(1).standard_normal((4, n))
    for sig in (1e-3, 1e-1):
        r1 = np.random.default_rng(5); r2 = np.random.default_rng(5)
        a = st.run_window(Wf, h1, -50, 0, 1.0, r1)
        b = st.run_window(Wf, sig * h1, -50, 0, sig, r2)
        assert np.max(np.abs(b / sig - a)) / np.max(np.abs(a)) < 1e-12


def test_lemma1_identity():
    rng = np.random.default_rng(3)
    n = 64
    h = rng.standard_normal(n); hp = rng.standard_normal(n) * 0.7 + 0.5 * h
    a, b = np.linalg.norm(h), np.linalg.norm(hp); th = math.acos(h @ hp / (a * b))
    pred = np.sum((h - hp) ** 2) - (2 * a * b / math.pi) * (math.sin(th) - th * math.cos(th))
    M = 4000
    acc = 0.0
    for _ in range(M // 100):
        W = rng.standard_normal((100, n, n)) * math.sqrt(2.0 / n)
        d = np.maximum(W @ h, 0) - np.maximum(W @ hp, 0)
        acc += np.sum(d * d)
    est = acc / (M // 100 * 100)
    assert abs(est / pred - 1) < 0.02, (est, pred)


def test_radius_chain_identity():
    """|h_l|^2 = (sqrt(g |h|^2) + sigma eta)^2 + sigma^2 C with g, eta, C computed from the realised W, xi."""
    rng = np.random.default_rng(11)
    n, sig = 32, 0.3
    h = rng.standard_normal(n)
    W = rng.standard_normal((n, n)) * math.sqrt(2.0 / n); xi = rng.standard_normal(n)
    a = np.maximum(W @ h, 0); hn = a + sig * xi
    g = np.sum(a * a) / np.sum(h * h); e = a / np.linalg.norm(a); eta = xi @ e; C = np.sum(xi * xi) - eta ** 2
    lhs = np.sum(hn * hn); rhs = (math.sqrt(g * np.sum(h * h)) + sig * eta) ** 2 + sig ** 2 * C
    assert abs(lhs - rhs) < 1e-10 * lhs


def test_gain_law():
    """g = |relu(W u)|^2 has mean 1 and variance 5/n for any unit u."""
    rng = np.random.default_rng(2)
    n = 32
    u = np.zeros(n); u[0] = 1.0
    W = rng.standard_normal((200000, n)) * math.sqrt(2.0 / n)
    g = np.sum(np.maximum(W * 1.0, 0) ** 2, 1)  # rows as independent W u draws (W u ~ N(0, 2/n I))
    assert abs(g.mean() - 1) < 0.01 and abs(g.var() * n / 5 - 1) < 0.03, (g.mean(), g.var() * n)


def test_qr_ginibre():
    """No gates: QR exponents of products of n x n N(0, 1/n) matrices = (1/2)(log(2/n) + psi((n-k+1)/2))."""
    n = 8
    rng = np.random.default_rng(4)
    Q = np.eye(n); acc = np.zeros(n); T = 20000
    for _ in range(T):
        Q, R = np.linalg.qr(rng.standard_normal((n, n)) / math.sqrt(n) @ Q)
        acc += np.log(np.abs(np.diag(R)))
    lam = np.sort(acc / T)[::-1]
    pred = 0.5 * (math.log(2.0 / n) + digamma((n - np.arange(1, n + 1) + 1) / 2.0))
    assert np.max(np.abs(lam - pred)) < 0.01, (lam, pred)


if __name__ == '__main__':
    for f in [test_scale_invariance, test_lemma1_identity, test_radius_chain_identity, test_gain_law, test_qr_ginibre]:
        f(); print('ok', f.__name__, flush=True)
