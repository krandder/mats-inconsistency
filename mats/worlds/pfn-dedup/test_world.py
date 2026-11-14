import math
import numpy as np
from scipy.stats import multivariate_normal
import world as W


def test_partitions_and_prior():
    assert len(W.PARTS) == 203
    assert abs(W.PRIOR.sum() - 1) < 1e-12
    assert W.PRIOR[np.array([len(set(p)) == 6 for p in W.PARTS])].sum() > 0.3


def test_bayes_reduces_to_gls_when_partition_known():
    rng = np.random.default_rng(0)
    d = W.sample_np(50, rng)
    old = W.PRIOR.copy()
    try:
        for i in range(5):
            W.PRIOR = np.full(len(W.PARTS), 1e-300); W.PRIOR[d['pidx'][i]] = 1.0
            sub = {k: v[i:i + 1] for k, v in d.items()}
            out, nll, (md, vd), w = W.references(sub)
            assert abs(md[0] - out['c'][0][0]) < 1e-9 and abs(vd[0] - out['c'][1][0]) < 1e-9
            assert abs(nll['d'][0] - nll['c'][0]) < 1e-9
    finally:
        W.PRIOR = old


def _generic_nll(o, y, ystar, k):
    """Independent path: full joint Gaussian of all readings and y* per partition, no factorisation."""
    S, Q, QL = W.S, W.Q, W.QL
    dens = 0.0
    for p, pr in zip(W.PARTS, W.PRIOR):
        same = (p[:, None] == p[None, :]).astype(float)
        # variables: o[s,q] for all s,q (q<QL with known y), o[s,QL], and y*
        n = S * Q + 1
        mean = np.zeros(n); C = np.zeros((n, n))
        def idx(s, q): return s * Q + q
        for s in range(S):
            for q in range(Q):
                mean[idx(s, q)] = y[q] if q < QL else 0.0
                for t in range(S):
                    C[idx(s, q), idx(t, q)] = W.SG2 * same[s, t] + W.SE2 * (s == t) + (1.0 if q == QL else 0.0)
            C[idx(s, QL), n - 1] = C[n - 1, idx(s, QL)] = 1.0
        C[n - 1, n - 1] = 1.0
        obs = list(range(S * Q)); x = o.reshape(-1)  # o is (S, Q) row-major by s then q
        po = multivariate_normal(mean[obs], C[np.ix_(obs, obs)]).pdf(x)
        Coo = C[np.ix_(obs, obs)]; cy = C[n - 1, obs]
        g = np.linalg.solve(Coo, cy)
        m = g @ (x - mean[obs]); v = 1.0 - cy @ g
        dens += pr * po * math.exp(-0.5 * (ystar - m) ** 2 / v) / math.sqrt(2 * math.pi * v)
    # normalise by p(obs)
    pobs = 0.0
    for p, pr in zip(W.PARTS, W.PRIOR):
        same = (p[:, None] == p[None, :]).astype(float)
        n = S * Q
        mean = np.zeros(n); C = np.zeros((n, n))
        for s in range(S):
            for q in range(Q):
                mean[s * Q + q] = y[q] if q < QL else 0.0
                for t in range(S):
                    C[s * Q + q, t * Q + q] = W.SG2 * same[s, t] + W.SE2 * (s == t) + (1.0 if q == QL else 0.0)
        pobs += pr * multivariate_normal(mean, C).pdf(o.reshape(-1))
    return -math.log(dens / pobs)


def test_bayes_matches_generic_joint():
    rng = np.random.default_rng(1)
    d = W.sample_np(3, rng)
    out, nll, _, _ = W.references(d)
    for i in range(3):
        g = _generic_nll(d['o'][i], d['y'][i], d['y'][i, W.QL], d['k'][i])
        assert abs(g - nll['d'][i]) < 1e-6, (g, nll['d'][i])


def test_singletons_iid_equals_gls():
    rng = np.random.default_rng(2)
    d = W.sample_np(400, rng)
    out, nll, _, _ = W.references(d)
    sing = np.array([len(set(l)) == 6 for l in d['lab']])
    assert np.allclose(out['b'][0][sing], out['c'][0][sing])
