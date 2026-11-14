"""Unit tests: stats helpers, q families, subset sampling, description length."""
import os
import sys

import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmlib import stats, qfam, dl, estimators as E  # noqa: E402


def test_ess_auc():
    assert abs(stats.kish_ess(np.ones(100)) - 100) < 1e-9
    w = np.zeros(100); w[0] = 1
    assert abs(stats.kish_ess(w) - 1) < 1e-9
    rng = np.random.default_rng(0)
    assert stats.auc(np.ones(10) + 1, np.zeros(10)) == 1.0
    assert abs(stats.auc(rng.standard_normal(5000), rng.standard_normal(5000)) - 0.5) < 0.03


def test_tail_diagnostics():
    rng = np.random.default_rng(0)
    x = rng.pareto(2.0, 20000) + 1           # tail index 2 -> k = 0.5
    k = stats.psis_khat(np.log(x))
    assert 0.35 < k < 0.65, k
    assert stats.psis_khat(rng.standard_normal(20000) * 0.3) < 0.2
    h = stats.hill_tail_index(rng.pareto(3.0, 50000) + 1, frac=0.02)
    assert 2.4 < h < 3.6, h


def test_boot_mean_t_covers():
    rng = np.random.default_rng(1)
    cover = 0
    for r in range(100):
        m, se, lo, hi = stats.boot_mean_t(rng.exponential(1.0, 400) - 1.0, B=400, rng=rng)
        cover += lo <= 0 <= hi
    assert cover >= 88, cover


def test_condgaussian_fit_logpdf_sample():
    rng = np.random.default_rng(0)
    C = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, 0.3], [0.0, 0.3, 0.5]])
    z = rng.integers(0, 2, 200000)
    mus = np.array([[0, 1, 2], [3, -1, 0.0]])
    H = mus[z] + rng.standard_normal((len(z), 3)) @ np.linalg.cholesky(C).T
    q = qfam.CondGaussian('full').fit(H, z)
    assert np.abs(q.mu[1] - mus[1]).max() < 0.02 and np.abs(q.cov[0] - C).max() < 0.03
    lp = q.logpdf(H[:50], z[:50])
    ref = np.array([multivariate_normal(q.mu[k], q.cov[k]).logpdf(h) for h, k in zip(H[:50], z[:50])])
    assert np.abs(lp - ref).max() < 1e-9
    S = q.sample(z, rng=rng)
    assert np.abs(np.cov(S[z == 0], rowvar=False) - C).max() < 0.05
    for ct in ('diag', 'iso', 'lowrank:1'):
        qq = qfam.CondGaussian(ct).fit(H, z)
        assert np.all(np.linalg.eigvalsh(qq.cov[0]) > 0)


def test_subset_knockout_matches_gaussian_conditioning():
    rng = np.random.default_rng(0)
    C = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.0], [0.2, 0.0, 1.0]])
    q = qfam.CondGaussian('full').fit(rng.standard_normal((400000, 3)) @ np.linalg.cholesky(C).T)
    H = np.tile([[0.0, 1.5, -1.0]], (200000, 1))
    out = q.subset_sample(H, [0], rng=rng, mode='knockout')
    Ci = np.linalg.inv(C[1:, 1:])
    m_true = C[0, 1:] @ Ci @ np.array([1.5, -1.0]); v_true = C[0, 0] - C[0, 1:] @ Ci @ C[1:, 0]
    assert abs(out[:, 0].mean() - m_true) < 0.02 and abs(out[:, 0].var() - v_true) < 0.02
    assert np.all(out[:, 1:] == H[:, 1:])
    kin = q.subset_sample(H, [0], rng=rng, mode='knockin')
    assert np.all(kin[:, 0] == H[:, 0]) and abs(kin[:, 1].mean() - 0.0) < 0.02


def test_linear_gaussian_recovers_map():
    rng = np.random.default_rng(0)
    Phi = rng.standard_normal((50000, 4)); A = rng.standard_normal((4, 3))
    H = Phi @ A + 0.5 + 0.3 * rng.standard_normal((50000, 3))
    q = qfam.LinearGaussian('diag').fit(H, Phi)
    mu, covs, _ = q.mean_cov(Phi[:100])
    assert np.abs(mu - (Phi[:100] @ A + 0.5)).max() < 0.02
    assert np.abs(np.diag(covs[0]) - 0.09).max() < 0.005


def test_group_and_cell_rms_unbiased():
    rng = np.random.default_rng(0)
    n, M = 20000, 8
    dc = rng.normal(0, 0.1, n)                       # true per-cell gap, E dc^2 = 0.01
    Gp = rng.standard_normal((n, M)); Gq = rng.standard_normal((n, M)) + dc[:, None]
    r = E.cell_rms(Gq, Gp)
    assert abs(r['ms'] - np.mean(dc ** 2)) < 3 * r['ms_se']
    z = rng.integers(0, 3, 300000); shift = np.array([0.0, 0.1, -0.2])
    g = E.group_rms(rng.standard_normal(300000) + shift[z], z, rng.standard_normal(300000), z, B=50)
    assert abs(g['rms'] - np.sqrt(np.mean(shift ** 2))) < 0.01


def test_dl_quantisation_and_frontier():
    x = np.linspace(-1, 1, 101)
    assert np.abs(dl.quantize(x, 8) - x).max() < 1.0 / 127
    rng = np.random.default_rng(0)
    H = rng.standard_normal((20000, 3)) + np.array([0.3, -0.2, 0.1])
    q = qfam.CondGaussian('full').fit(H)
    probe = np.random.default_rng(5).standard_normal((2000, 3))
    ev = lambda m: float(m.logpdf(probe).mean())
    r = dl.param_bits(q, ev, tol=1e-3, N_fit=20000)
    assert r['b_star'] is not None and 4 <= r['b_star'] <= 16
    assert r['bits'] == r['n_eff'] * r['b_star'] + 16 * r['n_tensors']
    r2 = dl.param_bits(q, ev, tol=1e-6)
    assert r2['b_star'] > r['b_star']
    L = dl.Ledger()
    for name, s, rms in [('a', 10, 1.0), ('b', 20, 0.5), ('c', 30, 0.45), ('d', 40, 0.1), ('e', 35, 0.9)]:
        L.add(name, s, 0, rms=rms, bic=s)
    assert L.frontier() == ['a', 'b', 'd']
    assert abs(dl.break_even_N(8, 1e-3) - 8 * np.log(2) / 1e-3) < 1e-9
    assert dl.struct_bits(n_features=2, library_size=64) == 12 + np.log2(3) + np.log2(8)
