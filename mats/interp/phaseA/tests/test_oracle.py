"""Exactness tests of the round-2 oracle samplers (lib/oracle.py); world only, no network.

Self-consistency: if x' ~ p and x'' ~ p(x | psi(x')) then x'' ~ p and (x', x'') is exchangeable. We check moments
that a wrong posterior (over v_s, y*, or the SS-to-source assignment) would break, and that kept variables are kept."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import tp, oracle as O  # noqa: E402

N = 200000
D = tp.sample_episodes(N, 12345)
D2 = tp.sample_episodes(N, 54321)          # an independent reference sample from p
QQ0, LAB12 = tp.QQ[0], 1 * tp.Q + 2       # QQ token of source 0; LAB token (s=1, j=2)


def _c(d):
    Qx, Sx, R = O.cvars(d)
    return Qx, Sx, R


def _z(a, b):
    """z-score of the difference of two sample means (independent samples)."""
    return abs(a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))


def _zp(a, b):
    """paired: z-score of mean(a - b)."""
    dlt = a - b
    sd = dlt.std()
    return 0.0 if sd == 0 else abs(dlt.mean()) / (sd / np.sqrt(len(dlt)))


def test_beta_matches_cell_stats():
    d = tp.sample_episodes(3000, 7)
    cs = tp.cell_stats(d)
    Qx, Sx, _ = _c(d)
    assert np.abs(O.beta_of(Qx, Sx) - cs['beta']).max() < 1e-10


def test_keep_sets():
    rng = np.random.default_rng(0)
    d = tp.sample_episodes(5000, 3)
    Qx, Sx, Rx = _c(d)
    for var in ('lat', 'c', 'cID', 'cIDr'):
        for t in (QQ0, LAB12, tp.QI):
            x2, _ = O.draw(d, t, var, rng)
            Q2, S2, R2 = _c(x2)
            assert np.abs(Q2 - Qx).max() < 1e-12 and np.abs(S2 - Sx).max() < 1e-9, (var, t)
    # LAB token keeps its own residual coordinate and label (all variants except lat)
    for var in ('own', 'q0', 'c', 'U', 'cID'):
        x2, _ = O.draw(d, LAB12, var, rng)
        _, S2, R2 = _c(x2)
        assert np.abs(R2[:, 1, 2] - Rx[:, 1, 2]).max() < 1e-12 and np.abs(x2['y'][:, 2] - d['y'][:, 2]).max() < 1e-12, var
        assert np.abs(x2['semb'][:, 1] - d['semb'][:, 1]).max() == 0 and np.abs(x2['qemb'][:, 2] - d['qemb'][:, 2]).max() == 0
        if var in ('own', 'q0'):
            assert np.abs(S2[:, 1] - Sx[:, 1]).max() < 1e-9
    # cIDr keeps the own source's whole residual vector and all labels and IDs
    x2, _ = O.draw(d, QQ0, 'cIDr', rng)
    _, _, R2 = _c(x2)
    assert np.abs(R2[:, 0] - Rx[:, 0]).max() < 1e-12 and np.abs(x2['y'][:, :4] - d['y'][:, :4]).max() == 0
    assert np.abs(x2['semb'] - d['semb']).max() == 0
    # lat keeps no IDs
    x2, _ = O.draw(d, QQ0, 'lat', rng)
    assert np.abs(x2['semb'][:, 0] - d['semb'][:, 0]).max() > 0.1
    # U keeps Q and the multiset of S
    x2, info = O.draw(d, QQ0, 'U', rng)
    Q2, S2, _ = _c(x2)
    assert np.abs(Q2 - Qx).max() < 1e-12 and np.abs(np.sort(S2, 1) - np.sort(Sx, 1)).max() < 1e-9
    assert 0.2 < (np.abs(S2 - Sx).max(1) > 1e-9).mean() < 0.9      # the assignment really is resampled


def _marg_moments(d):
    Qx, Sx, R = _c(d)
    qbar = (Qx.sum(1, keepdims=True) - Qx) / 2
    return {'Q1Q2': Qx[:, 1] * Qx[:, 2], 'Q0Q1': Qx[:, 0] * Qx[:, 1], 'Q1sq': Qx[:, 1] ** 2,
            'S1': Sx[:, 1], 'S1Q1sq': Sx[:, 1] * Qx[:, 1] ** 2, 'pair0': Sx[:, 0] * (Qx[:, 0] - qbar[:, 0]) ** 2,
            'pair1': Sx[:, 1] * (Qx[:, 1] - qbar[:, 1]) ** 2, 'logS0logS1': np.log(Sx[:, 0]) * np.log(Sx[:, 1])}


def test_self_consistency_own_and_U():
    rng = np.random.default_rng(1)
    ref = _marg_moments(D2)
    for var, t in (('own', QQ0), ('U', QQ0), ('U', LAB12), ('own', LAB12)):
        x2, _ = O.draw(D, t, var, rng)
        m2 = _marg_moments(x2)
        for k in ref:
            assert _z(m2[k], ref[k]) < 4.5, (var, t, k, m2[k].mean(), ref[k].mean())
        # exchangeability of (x', x''): E[f(x') g(x'')] = E[g(x') f(x'')]
        Q1, S1, _ = _c(D); Q2, S2, _ = _c(x2)
        for a, b in ((S1[:, 1] * Q2[:, 1] ** 2, S2[:, 1] * Q1[:, 1] ** 2), (Q1[:, 0] * Q2[:, 2], Q2[:, 0] * Q1[:, 2]),
                     (np.log(S1[:, 2]) * Q2[:, 2] ** 2, np.log(S2[:, 2]) * Q1[:, 2] ** 2)):
            assert _zp(a, b) < 4.5, (var, t)


def test_q0_beta_match():
    rng = np.random.default_rng(2)
    d = tp.sample_episodes(20000, 11)
    x2, info = O.draw(d, QQ0, 'q0', rng, K=32, want_own=True)
    err = np.abs(info['beta_err'])
    assert np.median(err) < 0.01 and np.quantile(err, 0.95) < 0.05, (np.median(err), np.quantile(err, .95))
    Qx, Sx, _ = _c(d); Q2, S2, _ = _c(x2)
    assert np.abs(Q2[:, 0] - Qx[:, 0]).max() < 1e-12 and np.abs(S2[:, 0] - Sx[:, 0]).max() < 1e-9
    b2 = O.beta_of(Q2, S2)[:, 0]
    assert np.abs(b2 - O.beta_of(Qx, Sx)[:, 0] - info['beta_err']).max() < 1e-9
    o2 = info['own']; Qo, So, _ = _c(o2)
    assert np.abs(Qo[:, 0] - Qx[:, 0]).max() < 1e-12


def test_lab_sphere_slice():
    """Given (S_s, r_sj) the other three residual coordinates are uniform on the slice: zero conditional mean and
    E r_k^2 = (S - r_j^2)/3."""
    rng = np.random.default_rng(4)
    x2, _ = O.draw(D, LAB12, 'c', rng)
    _, Sx, Rx = _c(D); _, _, R2 = _c(x2)
    for k in (0, 1, 3):
        assert _zp(R2[:, 1, k] * Rx[:, 1, k], np.zeros(N)) < 4.5
        assert _zp(R2[:, 1, k] ** 2, (Sx[:, 1] - Rx[:, 1, 2] ** 2) / 3) < 4.5
