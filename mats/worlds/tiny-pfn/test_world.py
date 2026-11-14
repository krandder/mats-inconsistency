"""Unit tests for the tiny reliability world: exact Bayes vs an independent generic-joint enumeration, vs Monte
Carlo (prior sampling + generic joint-Gaussian conditioning), calibration on simulated episodes, and reductions."""
import math
import numpy as np
from scipy.stats import multivariate_normal, kstest, norm
import world as W


def _joint(v):
    """Full joint Gaussian over all S*Q readings (row-major s, q) and y*, given variances v; labelled y treated as
    known (mean shift), y* ~ N(0,1). No factorisation used."""
    S, Q, QL = W.S, W.Q, W.QL
    n = S * Q + 1
    C = np.zeros((n, n))
    for s in range(S):
        for t in range(S):
            for q in range(Q):
                C[s * Q + q, t * Q + q] = (v[s] if s == t else 0.0) + (1.0 if q == QL else 0.0)
        C[s * Q + QL, n - 1] = C[n - 1, s * Q + QL] = 1.0
    C[n - 1, n - 1] = 1.0
    return C


def _cond(o, y, v):
    """Returns (p(all readings | v), predictive mean and var of y* | readings, v) via generic conditioning."""
    S, Q, QL = W.S, W.Q, W.QL
    C = _joint(v)
    mean = np.array([y[q] if q < QL else 0.0 for s in range(S) for q in range(Q)])
    obs = np.arange(S * Q); x = o.reshape(-1)
    Coo = C[np.ix_(obs, obs)]; cy = C[-1, obs]
    p = multivariate_normal(mean, Coo).pdf(x)
    g = np.linalg.solve(Coo, cy)
    return p, g @ (x - mean), 1.0 - cy @ g


def test_combos_and_prior():
    assert W.COMB.shape == (27, 3) and len({tuple(c) for c in W.COMB}) == 27
    assert abs(np.exp(W.LOGPRIOR).sum() - 1) < 1e-12


def test_bayes_reduces_to_oracle_when_variances_known():
    d = W.sample_np(20, np.random.default_rng(0))
    old = W.LOGPRIOR.copy()
    try:
        for i in range(5):
            ci = int(np.flatnonzero((W.COMB == d['vidx'][i]).all(1))[0])
            W.LOGPRIOR = np.full(27, -1e6); W.LOGPRIOR[ci] = 0.0
            sub = {k: v[i:i + 1] for k, v in d.items()}
            out, nll, mom, w = W.references(sub)
            assert abs(mom['bayes'][0][0] - out['b'][0][0]) < 1e-9 and abs(mom['bayes'][1][0] - out['b'][1][0]) < 1e-9
            assert abs(nll['bayes'][0] - nll['b'][0]) < 1e-9
    finally:
        W.LOGPRIOR = old


def test_bayes_matches_generic_joint_enumeration():
    d = W.sample_np(4, np.random.default_rng(1))
    out, nll, mom, _ = W.references(d)
    for i in range(4):
        o, y, ys = d['o'][i], d['y'][i], d['y'][i, W.QL]
        num = den = mnum = 0.0
        for v in W.VC:
            p, m, vv = _cond(o, y, v)
            den += p / 27; num += p / 27 * norm.pdf(ys, m, math.sqrt(vv)); mnum += p / 27 * m
        assert abs(-math.log(num / den) - nll['bayes'][i]) < 1e-8
        assert abs(mnum / den - mom['bayes'][0][i]) < 1e-8


def test_bayes_matches_monte_carlo_importance():
    """Prior sampling of each source's variance (independent draws, not the COMB table) + generic conditioning."""
    rng = np.random.default_rng(2)
    d = W.sample_np(4, rng)
    out, nll, mom, _ = W.references(d)
    M = 200_000
    for i in range(4):
        o, y, ys = d['o'][i], d['y'][i], d['y'][i, W.QL]
        vs = rng.choice(W.VLEV, size=(M, W.S))
        cache = {}
        lik = np.empty(M); dens = np.empty(M); mean = np.empty(M)
        for j, v in enumerate(map(tuple, vs)):
            if v not in cache:
                p, m, vv = _cond(o, y, np.array(v)); cache[v] = (p, norm.pdf(ys, m, math.sqrt(vv)), m)
            lik[j], dens[j], mean[j] = cache[v]
        wts = lik / lik.sum()
        mc_nll = -math.log((wts * dens).sum()); mc_mean = (wts * mean).sum()
        # MC error: self-normalised IS over 27 atoms with 2e5 draws; tolerance well above its sd
        assert abs(mc_nll - nll["bayes"][i]) < 0.01, (mc_nll, nll['bayes'][i])
        assert abs(mc_mean - mom["bayes"][0][i]) < 0.01 * (1 + abs(mc_mean)), (mc_mean, mom['bayes'][0][i])


def _independent_sampler(n, rng):
    v = rng.choice(W.VLEV, size=(n, W.S))
    y = rng.normal(size=(n, W.Q))
    o = np.empty((n, W.S, W.Q))
    for s in range(W.S):
        o[:, s] = y + rng.normal(size=(n, W.Q)) * np.sqrt(v[:, s:s + 1])
    vidx = np.searchsorted(W.VLEV, v)
    return dict(vidx=vidx, v=v, y=y, o=o)


def test_calibration_on_simulated_episodes():
    rng = np.random.default_rng(3)
    n = 40_000
    d = _independent_sampler(n, rng)
    out, nll, mom, w = W.references(d)
    ys = d['y'][:, W.QL]
    means, vars_, ww = out['bayes']
    # PIT of the exact mixture predictive is uniform
    pit = (ww * norm.cdf((ys[:, None] - means) / np.sqrt(vars_)[None])).sum(1)
    assert kstest(pit, 'uniform').pvalue > 1e-3
    # moment calibration
    m, v = mom['bayes']
    err2 = (ys - m) ** 2
    assert abs(err2.mean() - v.mean()) < 3 * err2.std() / math.sqrt(n)
    slope = np.cov(ys, m)[0, 1] / m.var()
    assert abs(slope - 1) < 0.03
    # ordering of expected NLL: oracle < bayes < lab, a
    assert nll['b'].mean() < nll['bayes'].mean() < min(nll['lab'].mean(), nll['a'].mean(), nll['qonly'].mean())
    # posterior over source-0 variance level (marginal of the 27-way posterior) is calibrated
    for j in range(W.NV):
        p = ww[:, W.COMB[:, 0] == j].sum(1)
        hit = d['vidx'][:, 0] == j
        for lo in np.arange(0, 1, 0.1):
            sel = (p >= lo) & (p < lo + 0.1)
            if sel.sum() > 1000:
                assert abs(hit[sel].mean() - p[sel].mean()) < 4 * math.sqrt(0.25 / sel.sum()) + 0.005


def test_reductions():
    rng = np.random.default_rng(4)
    d = W.sample_np(100, rng)
    out, _, _, _ = W.references(d)
    ost = d['o'][:, :, W.QL]
    m, v = W.gls(ost, np.full(W.S, W.VBAR))
    assert np.allclose(m, out['a'][0]) and np.allclose(v, out['a'][1])
    assert np.allclose(out['a'][0], ost.sum(1) / W.VBAR / (1 + W.S / W.VBAR))
    eq = (d['vidx'] == d['vidx'][:, :1]).all(1)                 # all-equal variances: oracle weights are equal
    vv = d['v'][eq, 0]
    assert np.allclose(out['b'][0][eq], ost[eq].sum(1) / vv / (1 + W.S / vv))


def test_mixture_entropy_quadrature():
    means = np.zeros((2, 27)); vars_ = W.VC.mean(1) * 0 + 1.0 / W.PREC
    w = np.zeros((2, 27)); w[0, 0] = 1; w[1, 26] = 1
    H = W.mix_entropy(means, vars_, w)
    assert np.allclose(H, 0.5 * np.log(2 * np.pi * np.e * vars_[[0, 26]]), atol=1e-6)


import pytest


@pytest.mark.parametrize('DS', [16, 4])
def test_tokens_layout(DS):
    d = W.sample_np(7, np.random.default_rng(5))
    X, m = W.tokens_np(d, np.random.default_rng(6), ds=DS)
    assert X.shape == (7, W.S * W.Q + 1, 2 * DS + 4) and m.all()
    for s in range(W.S):
        for q in range(W.Q):
            t = s * W.Q + q
            assert np.allclose(X[:, t, :DS], X[:, s * W.Q, :DS])
            assert np.allclose(X[:, t, DS:2 * DS], X[:, q, DS:2 * DS])
            assert np.allclose(X[:, t, 2 * DS], d['o'][:, s, q])
            assert np.allclose(X[:, t, 2 * DS + 1], d['y'][:, q] if q < W.QL else 0)
            assert np.all(X[:, t, 2 * DS + 2] == (q == W.QL)) and np.all(X[:, t, 2 * DS + 3] == 0)
    assert np.allclose(X[:, -1, DS:2 * DS], X[:, W.QL, DS:2 * DS]) and np.all(X[:, -1, 2 * DS + 3] == 1)
    assert np.all(X[:, -1, :DS] == 0) and np.all(X[:, -1, 2 * DS:2 * DS + 3] == 0)
