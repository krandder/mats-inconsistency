"""Tests for junta.py: exact Bayes vs independent dense-Gaussian conditioning and vs ABC on the generative spec,
reductions, policy ignorability (replay against different hidden worlds), calibration under every policy."""
import itertools, math
import numpy as np
import pytest
from scipy.stats import multivariate_normal
import junta as J


def dense_posterior(w, items, y, root, q):
    """Independent implementation: joint Gaussian of (deduplicated reports, v_q) per (R, sigma2)."""
    idx = np.nonzero(root == np.arange(len(root)))[0]
    it, yy = items[idx], y[idx]
    LW, M, V = [], [], []
    for si, s2 in enumerate(w.sig2):
        for h, R in enumerate(w.hyps):
            cell = lambda x: tuple((int(x) >> r) & 1 for r in R)
            n = len(it)
            C = np.array([[float(cell(a) == cell(b)) + w.tau2 * (a == b) + s2 * (i == j)
                           for j, b in enumerate(it)] for i, a in enumerate(it)])
            k = np.array([float(cell(a) == cell(q)) + w.tau2 * (a == q) for a in it])
            lml = multivariate_normal(np.zeros(n), C).logpdf(yy)
            sol = np.linalg.solve(C, k)
            LW.append(lml); M.append(sol @ yy * 0 + k @ np.linalg.solve(C, yy)); V.append(1 + w.tau2 - k @ sol)
    lw = np.array(LW); lw -= lw.max(); lw -= math.log(np.exp(lw).sum())
    return lw, np.array(M), np.array(V)


@pytest.mark.parametrize('tau2', [0.0, 0.25])
@pytest.mark.parametrize('qmode', ['obs', 'unobs'])
def test_dense_gaussian(tau2, qmode):
    w = J.Junta(dict(B=4, sizes=[1, 2], tau2=tau2))
    rng = np.random.default_rng(1)
    for rep in range(4):
        items = rng.integers(0, 16, 7); items[3] = items[1]; items[5] = items[1]
        y = rng.standard_normal(7)
        root = np.arange(7); root[4] = 2; y[4] = y[2]; items[4] = items[2]     # an exact copy
        q = items[1] if qmode == 'obs' else 15 - items[0]
        lw, m, v = w.posterior(items, y, root, q)
        lw2, m2, v2 = dense_posterior(w, items, y, root, q)
        # component order: sigma2-major, hypothesis-minor in both
        assert np.allclose(np.exp(lw), np.exp(lw2), atol=1e-9)
        assert np.allclose(m, m2, atol=1e-9) and np.allclose(v, v2, atol=1e-9)


def test_item_ref_dense():
    """Item-only reference equals dense conditioning with every item its own cell."""
    w = J.Junta(dict(B=4, sizes=[1], tau2=0.25))
    rng = np.random.default_rng(2)
    items = np.array([1, 1, 2, 3, 3, 3]); y = rng.standard_normal(6); root = np.arange(6)
    for q in (3, 9):
        lw, m, v = w.posterior_item(items, y, root, q)
        for si, s2 in enumerate(w.sig2):
            C = np.array([[float(a == b) * (1 + w.tau2) + s2 * (i == j) for j, b in enumerate(items)]
                          for i, a in enumerate(items)])
            k = np.array([float(a == q) * (1 + w.tau2) for a in items])
            assert abs(m[si] - k @ np.linalg.solve(C, y)) < 1e-9
            assert abs(v[si] - (1 + w.tau2 - k @ np.linalg.solve(C, k))) < 1e-9


def test_abc_against_generative_spec():
    """ABC on the generative spec (prior sampling, accept when every report lands within delta)."""
    w = J.Junta(dict(B=3, sizes=[1], tau2=0.25, sig2=[0.1, 0.5]))
    items = np.array([0b011, 0b011, 0b100]); root = np.arange(3); q = 0b001
    yobs = np.array([0.8, 1.1, -0.4])
    rng = np.random.default_rng(3); M = 4_000_000; delta = 0.1
    acc_th = []
    for _ in range(6):
        h = rng.integers(0, w.H, M); s = w.sig2[rng.integers(0, 2, M)]
        g = rng.standard_normal((M, 2)); b = rng.standard_normal((M, 4)) * math.sqrt(w.tau2)   # items 3,4,1 + q slot
        pos = w.hpos[h, 0]
        c = lambda x: (x >> pos) & 1
        v3 = g[np.arange(M), c(3)] + b[:, 0]; v4 = g[np.arange(M), c(4)] + b[:, 1]; vq = g[np.arange(M), c(1)] + b[:, 2]
        ys = np.stack([v3, v3, v4], 1) + rng.standard_normal((M, 3)) * np.sqrt(s)[:, None]
        ok = (np.abs(ys - yobs) < delta).all(1)
        acc_th.append(vq[ok])
    th = np.concatenate(acc_th)
    lw, m, v = w.posterior(items, yobs, root, q)
    pw = np.exp(lw); mean = (pw * m).sum(); var = (pw * (v + m ** 2)).sum() - mean ** 2
    se = math.sqrt(var / len(th))
    assert len(th) > 2000
    assert abs(th.mean() - mean) < 4 * se + 0.01, (th.mean(), mean, se, len(th))
    assert abs(th.var() - var) < 0.08 * var, (th.var(), var)


def test_reductions():
    w = J.Junta(dict(B=6, sizes=[2], tau2=0.25, p_copy_zero=1.0))
    rng = np.random.default_rng(4)
    ep = w.sample(rng, 20, 40, 'uniform')
    for i in range(20):
        a = w.posterior(*w._episode(ep, i)); b = w.posterior(*w._episode(ep, i), dedup=False)
        assert all(np.allclose(x, y) for x, y in zip(a, b))          # no copies: copy-naive == Bayes
    wc = J.Junta(dict(B=6, sizes=[2], tau2=0.25, p_copy_zero=0.0, p_copy_max=0.6))
    ep = wc.sample(rng, 20, 40, 'uniform')
    for i in range(20):
        it, y, root, q = wc._episode(ep, i)
        keep = root == np.arange(len(root))
        a = wc.posterior(it, y, root, q); b = wc.posterior(it[keep], y[keep], np.arange(keep.sum()), q)
        assert all(np.allclose(x, z) for x, z in zip(a, b))          # copies carry no evidence
    # a single hypothesis: oracle == full Bayes
    w1 = J.Junta(dict(B=2, sizes=[2], tau2=0.25))
    ep = w1.sample(rng, 10, 30, 'uniform')
    for i in range(10):
        a = w1.posterior(*w1._episode(ep, i)); b = w1.posterior(*w1._episode(ep, i), hidx=0)
        assert all(np.allclose(x, z) for x, z in zip(a, b))


@pytest.mark.parametrize('policy', J.ALL_POLICIES)
def test_policy_ignorability_replay(policy):
    """Same presentation-policy randomness, different hidden world, observed values forced equal -> identical
    items, provenance and query. A policy that peeked at R, g or b would fail this."""
    w = J.Junta(dict(B=10, sizes=[2], tau2=0.25, p_copy_zero=0.0))
    a = w.sample(np.random.default_rng(5), 16, 60, policy, rng_hidden=np.random.default_rng(100))
    yo = np.zeros_like(a['y'])
    fresh = a['root'] == np.arange(60)[None, :]
    yo[fresh] = a['y'][fresh]
    b = w.sample(np.random.default_rng(5), 16, 60, policy, rng_hidden=np.random.default_rng(200), y_override=yo)
    assert (a['h'] != b['h']).any()
    assert np.array_equal(a['items'], b['items']) and np.array_equal(a['root'], b['root'])
    assert np.array_equal(a['q'], b['q']) and np.allclose(a['y'], b['y'])
    # oracle posterior depends only on the data: identical whatever policy label the episode carries
    for i in range(4):
        p1 = w.posterior(*w._episode(a, i)); p2 = w.posterior(*w._episode(b, i))
        assert all(np.allclose(x, z) for x, z in zip(p1, p2))


@pytest.mark.parametrize('policy', J.ALL_POLICIES)
@pytest.mark.parametrize('tau2', [0.0, 0.25])
def test_calibration(policy, tau2):
    """Sampler and oracle agree: Bayes z-scores have mean 0 / variance 1 and Bayes beats every reference."""
    w = J.Junta(dict(B=10, sizes=[2], tau2=tau2))
    ep = w.sample(np.random.default_rng(6), 1500, 32, policy)
    pb = w.exact_bayes(ep)
    import harness as Hn
    mu, var = Hn.moments(pb)
    z = (ep['target'] - mu) / np.sqrt(var)
    assert abs(z.mean()) < 0.1 and abs(z.var() - 1) < 0.15, (z.mean(), z.var())
    refs = w.references(ep)
    nb = Hn.nll(pb, ep['target']).mean()
    for k, p in refs.items():
        assert Hn.nll(p, ep['target']).mean() >= nb - 0.03 or k == 'oracle', k


def test_tokens_and_train_batch():
    w = J.Junta()
    ep = w.sample_train(np.random.default_rng(7), 64, 8, 64)
    assert ep['X'].shape == (64, 65, w.feat_dim) and ep['mask'][:, -1].all()
    assert (ep['mask'][:, :-1].sum(1) == ep['n']).all()
    assert (ep['X'][:, :-1][~ep['mask'][:, :-1]] == 0).all()
    assert (ep['X'][:, -1, -1] == 1).all() and (ep['X'][:, :-1, -1] == 0).all()
    # copies share the event-id vector and y; fresh tokens have distinct ids
    i = int(np.argmax((ep['root'] != np.arange(64)[None]).sum(1)))
    t = int(np.nonzero(ep['root'][i] != np.arange(64))[0][0])
    assert np.allclose(ep['ev'][i, t], ep['ev'][i, ep['root'][i, t]])
