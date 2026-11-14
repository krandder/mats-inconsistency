"""Tests for world2_market: exact Bayes vs an independent brute-force implementation, Monte Carlo calibration,
policy invariance (ignorable adaptive policy calibrated; value-leaking policy detected), reference identities,
and the C4 / C7 diagnostic constructions."""
import copy, itertools, math
import numpy as np
import pytest
from scipy import stats
import world2_market as W


def brute(ep):
    """Covariance-form Bayes on the raw quotes. Latents: v, b, eps_e for every event; enumerates ALL unknown
    sources. Independent of the transformed information-form solver."""
    K, S, E, n = ep.K, ep.S, len(ep.ev_src), ep.n
    G = np.zeros((n, K + S + E))
    for t in range(n):
        e = ep.q_ev[t]
        for k, c in zip(ep.ev_items[e], ep.ev_coef[e]):
            if k >= 0:
                G[t, k] += c
        G[t, K + ep.q_src[t]] += ep.ev_cb[e]
        G[t, K + S + e] = 1.0
    R = np.diag(np.where(ep.q_relay, W.RHO ** 2, 0.0))
    unk = list(np.flatnonzero(ep.unk))
    ws, ms, vs = [], [], []
    for lv in itertools.product((0, 1), repeat=len(unk)):
        sig = ep.sig.copy()
        for s, l in zip(unk, lv):
            sig[s] = (W.SIG_L, W.SIG_H)[l]
        P = np.r_[np.ones(K), np.full(S, W.TAU ** 2), sig[ep.ev_src] ** 2]
        C = G @ (P[:, None] * G.T) + R
        c = G[:, ep.query] * 1.0                       # cov(y, v_q) = G[:, q] * var(v_q) = G[:, q]
        Ci = np.linalg.inv(C)
        ms.append(c @ Ci @ ep.y); vs.append(1.0 - c @ Ci @ c)
        ws.append(stats.multivariate_normal(np.zeros(n), C, allow_singular=False).logpdf(ep.y))
    ws = np.exp(np.array(ws) - max(ws)); ws /= ws.sum()
    return ws, np.array(ms), np.sqrt(np.array(vs))


def close_mix(p, q, tol=1e-6):
    x = np.linspace(-4, 4, 41)
    return np.allclose(W.mix_logpdf(x, *p), W.mix_logpdf(x, *q), atol=tol, rtol=0)


@pytest.mark.parametrize('seed', range(12))
def test_bayes_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    ep = W.sample_episode(rng, N=int(rng.integers(6, 20)), K=int(rng.integers(2, 6)), S=int(rng.integers(2, 6)),
                          U=int(rng.integers(0, 4)), cfg=dict(p_relay=(0.3, 0.6)))
    assert close_mix(W.predictive(ep, 'bayes'), brute(ep))


def test_brute_force_covers_hidden_and_visible_relays():
    rng = np.random.default_rng(123)
    hit_h = hit_v = 0
    for _ in range(40):
        ep = W.sample_episode(rng, N=12, K=3, S=4, U=2, cfg=dict(p_relay=(0.5, 0.5)))
        rel = ep.q_ev[ep.q_relay]
        hit_h += (~ep.ev_vis[rel]).any(); hit_v += ep.ev_vis[rel].any()
        assert close_mix(W.predictive(ep, 'bayes'), brute(ep))
    assert hit_h > 5 and hit_v > 5


def test_noise_oracle_matches_brute_with_known_sigma():
    rng = np.random.default_rng(7)
    ep = W.sample_episode(rng, N=15, K=4, S=4, U=3, cfg=dict(p_relay=(0.4, 0.4)))
    ep2 = copy.copy(ep); ep2.unk = np.zeros(ep.S, bool)
    assert close_mix(W.predictive(ep, 'noise_oracle'), brute(ep2))


def test_references_coincide_without_unknowns_or_relays():
    rng = np.random.default_rng(3)
    ep = W.sample_episode(rng, N=20, K=4, S=4, U=0, cfg=dict(p_relay=(0.0, 0.0)))
    B = W.predictive(ep, 'bayes')
    for r in ('freq_only', 'noise_oracle', 'prov_naive'):
        assert close_mix(B, W.predictive(ep, r))


def test_order_invariance():
    rng = np.random.default_rng(5)
    ep = W.sample_episode(rng, N=30, K=5, S=4, U=2, cfg=dict(p_relay=(0.3, 0.3)))
    perm = rng.permutation(ep.n)
    ep2 = copy.copy(ep)
    ep2.q_src, ep2.q_ev, ep2.q_relay, ep2.q_delta = ep.q_src[perm], ep.q_ev[perm], ep.q_relay[perm], ep.q_delta[perm]
    ep2.y = ep2.values()
    assert close_mix(W.predictive(ep, 'bayes'), W.predictive(ep2, 'bayes'))


def pit(eps, mode='bayes'):
    return np.array([W.mix_cdf(e.v[e.query], *W.predictive(e, mode)) for e in eps])


def test_monte_carlo_calibration_and_power():
    """Bayes is calibrated over the generator (PIT uniform, incl. relay-heavy episodes); provenance-naive and
    bias-naive references are detectably miscalibrated in the same sample (the test has power)."""
    rng = np.random.default_rng(11)
    eps = [W.sample_episode(rng, N=int(rng.integers(8, 30)), cfg=dict(p_relay=(0.5, 0.7), p_noSPOT_query=0.0))
           for _ in range(1500)]
    assert stats.kstest(pit(eps), 'uniform').pvalue > 1e-3
    nll = {r: np.mean([-W.mix_logpdf(e.v[e.query], *W.predictive(e, r)) for e in eps])
           for r in ('bayes', 'prov_naive', 'bias_naive', 'freq_only', 'prior')}
    for r in ('prov_naive', 'bias_naive', 'freq_only', 'prior'):
        assert nll['bayes'] < nll[r], (r, nll)
    # power: concentrate on episodes where relays of hidden events matter
    p_naive = pit(eps, 'prov_naive')
    z = stats.norm.ppf(np.clip(p_naive, 1e-9, 1 - 1e-9))
    assert np.var(z) > 1.05 or stats.kstest(p_naive, 'uniform').pvalue < 1e-3


def test_policy_invariance():
    """An adaptive policy that depends on recorded observations is ignorable: the unchanged oracle stays
    calibrated. A policy that depends on hidden values is not: the same oracle is detectably biased."""
    rng = np.random.default_rng(21)
    ok = [W.sample_adaptive(rng, 14, 4, 3, 2, 'ignorable') for _ in range(1500)]
    assert stats.kstest(pit(ok), 'uniform').pvalue > 1e-3
    bad = [W.sample_adaptive(rng, 4, 4, 3, 2, 'leaky') for _ in range(1500)]
    assert stats.kstest(pit(bad), 'uniform').pvalue < 1e-4


def test_oracle_ignores_policy_parameters():
    """Identical data from two different policy draws -> identical oracle (the oracle reads only the quotes)."""
    rng = np.random.default_rng(31)
    ep = W.sample_episode(rng, N=25, K=5, S=4, U=2)
    ep2 = copy.deepcopy(ep); ep2.meta = dict(policy='something else entirely')
    assert close_mix(W.predictive(ep, 'bayes'), W.predictive(ep2, 'bayes'), tol=0)


def test_c4_pair():
    rng = np.random.default_rng(41)
    a, b, delta, tries = W.c4_pair(rng)
    assert delta >= 0.3
    for f in ('q_src', 'q_ev', 'q_relay', 'item_vec', 'src_vec', 'ev_vec'):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    qq = [t for t in range(a.n) if a.ev_items[a.q_ev[t], 0] == a.query]
    assert len(qq) == 2 and np.allclose(a.y[qq], b.y[qq])
    A, B_ = a.meta['A'], a.meta['B']
    na, nb = (a.q_src == A).sum(), (a.q_src == B_).sum()
    assert na == nb
    hp = W.h_posterior(a)
    assert hp[A] < 0.05 and hp[B_] > 0.95
    hp = W.h_posterior(b)
    assert hp[A] > 0.95 and hp[B_] < 0.05


def test_c7_sets():
    rng = np.random.default_rng(51)
    out = W.c7_set(rng)
    for a, b in W.C7_PAIRS:
        assert out[a].n == out[b].n, (a, b)
    sd = {k: W.moment_match(W.predictive(v))[2][0] for k, v in out.items()}
    assert sd['D'] > sd['T'] + 0.02          # diamond: shared root discounted
    assert sd['D4'] > sd['T4'] + 0.05
    assert sd['DR'] > sd['DI'] + 0.02        # visible relay adds nothing about the root
    assert sd['HR'] > sd['HI'] + 0.02        # two relays of one hidden measurement count once
    assert sd['HR4'] > sd['HI4'] + 0.04
    for a, b in (('D', 'T'), ('D4', 'T4')):
        dc = W.moment_match(W.predictive(W.naive_double_count(out[a])))[2][0]
        assert dc < sd[a] - 0.02 and abs(dc - sd[b]) < 0.03, (a, dc, sd[a], sd[b])
    for a, b in (('DR', 'DI'), ('HR4', 'HI4')):
        pn = W.moment_match(W.predictive(out[a], 'prov_naive'))[2][0]
        assert abs(pn - sd[b]) < 1e-6


def test_harness_interface():
    w = W.make_world()
    ep = w.sample_train(np.random.default_rng(0), 16, 8, 64)
    assert ep['X'].shape == (16, 65, w.feat_dim) and ep['mask'][:, -1].all()
    ep2 = w.sample_train(np.random.default_rng(0), 16, 8, 64)
    assert np.array_equal(ep['X'], ep2['X']) and np.array_equal(ep['target'], ep2['target'])
    for i, e in enumerate(ep['eps']):
        assert ep['mask'][i, :-1].sum() == e.n
        assert np.allclose(ep['X'][i, :e.n, W.F_Y], e.y, atol=1e-5)
        assert ep['target'][i] == e.v[e.query]
    pb = w.exact_bayes(ep)
    assert np.allclose(pb[0].sum(1), 1)
    refs = w.references(ep)
    assert set(refs) == set(W.REFS)
    m = w.meta(ep)
    assert all(len(v) == 16 for v in m.values())
    for pol in w.test_policies + list(W.DIAG):
        e3 = w.sample(np.random.default_rng(1), 12, 96, pol)
        assert e3['X'].shape[0] in (10, 12) and e3['X'].shape[2] == w.feat_dim


def test_relay_tokens_share_event_id_and_y_consistency():
    rng = np.random.default_rng(9)
    ep = W.sample_episode(rng, N=60, cfg=dict(p_relay=(0.5, 0.5)))
    X, M, Y = W.pack([ep])
    for t in np.flatnonzero(ep.q_relay):
        e = ep.q_ev[t]
        same = np.flatnonzero(ep.q_ev == e)
        ev = X[0, same, W.F_EV:W.F_EV + W.D_EV]
        assert np.allclose(ev, ev[0])
        assert ep.q_src[t] != ep.ev_src[e]
    # Episode.values reproduces the generator's y from latents
    assert np.allclose(ep.values(), ep.y)
