"""Known-answer tests (fast versions of VALIDATION.md cases i-iv)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmlib import synth, estimators as E, ratio as R, qfam  # noqa: E402
from cmlib.consistency import consistency_check, identity_check  # noqa: E402

torch.set_num_threads(1)
oh3 = lambda z: np.eye(3)[z]
oh2 = lambda z: np.eye(2)[z]


# ---------------------------------------------------------------- closed forms of the synthetic worlds
def test_mixworld_closed_forms():
    W = synth.MixWorld(d=8, K=3, a=2.0)
    assert abs(W.delta(0.0) - 2 * 2.0 ** 4 / 10) < 1e-9          # quartic: Delta = 2 a^4 / 10
    assert abs(W.delta(1.0)) < 1e-12 and abs(W.kl_pq(1.0)) < 1e-10 and abs(W.chi2_pq(1.0)) < 1e-8
    La = W.L_alpha(0.0, [0.0, 1.0])
    assert abs(La[0] - W.EG(0.0)) < 1e-5 and abs(La[1] - W.EG(1.0)) < 1e-5
    rng = np.random.default_rng(0)
    z = W.sample_z(400000, rng); Hq = W.sample(z, 0.0, rng)
    w = np.exp(W.log_w(Hq, z, 0.0))
    assert abs(w.mean() - 1) < 0.005 and abs((w ** 2).mean() - 1 - W.chi2_pq(0.0)) < 0.02
    Wq = synth.MixWorld(d=8, K=3, a=2.0, g='quadratic')
    assert abs(Wq.delta(0.0)) < 1e-9                                 # moment-matched q: quadratic G has no gap


def test_scaleworld_closed_forms():
    S = synth.ScaleWorld(d=31, k=1.1)
    rng = np.random.default_rng(0)
    Hq = S.sample_q(200000, rng)
    w = np.exp(S.log_w(Hq))
    assert abs(w.mean() - 1) < 0.01
    assert abs(S.G(Hq).mean() - S.G(S.sample_p(200000, rng)).mean() - S.delta()) < 0.01


# ---------------------------------------------------------------- (i) p = q
def test_null_p_equals_q():
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=1)
    rng = np.random.default_rng(3); n = 20000
    z = W.sample_z(n, rng); Hp = W.sample(z, 1.0, rng); Hq = W.sample(z, 1.0, rng)
    rm, dg = R.fit_ratio(Hp, Hq, oh3(z), oh3(z), family='quad', seed=0, null_gate=True)
    assert abs(dg['auc'] - 0.5) < 0.02 and abs(dg['kl_plugin']) < 0.01 and abs(dg['norm_q'] - 1) < 0.02
    z2 = W.sample_z(n, rng); Hp2 = W.sample(z2, 1.0, rng); Hq2 = W.sample(z2, 1.0, rng)
    d = E.direct_gap(W.G(Hq2, z2), W.G(Hp2, z2), paired=False)
    assert d['lo'] <= 0 <= d['hi'] or abs(d['delta']) < 3 * d['se']
    lw = rm.log_w(Hq2, oh3(z2))
    c = E.cov_qw(W.G(Hq2, z2), lw, z2, B=100)
    assert abs(c['cov_plain']) < 0.1                                  # G scale ~ 3, Delta(t=0) = 3.2


# ---------------------------------------------------------------- (ii) analytic non-Gaussian ratio
def test_true_ratio_estimators_match_closed_form():
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=0)
    rng = np.random.default_rng(1); n = 60000
    z = W.sample_z(n, rng); Hp = W.sample(z, 1.0, rng); Hq = W.sample(z, 0.0, rng)
    Gp, Gq = W.G(Hp, z), W.G(Hq, z)
    lwp, lwq = W.log_w(Hp, z, 0.0), W.log_w(Hq, z, 0.0)
    truth = W.delta(0.0)
    c = E.cov_qw(Gq, lwq, z, B=200)
    assert abs(-c['cov_plain'] - truth) < 4 * c['plain_se']
    al = np.linspace(0, 1, 6)
    tq = E.temper_q(Gq, lwq, al, groups=z, B=100)
    tb = E.temper_bridge(Gp, lwp, Gq, lwq, al, groups_p=z, groups_q=z, B=100)
    exact = W.L_alpha(0.0, al)
    assert np.all(np.abs(tq['L'] - exact) < 4 * tq['se'] + 1e-3)
    assert np.all(np.abs(tb['L'] - exact) < 4 * tb['se'] + 1e-3)
    assert abs(tb['gap'] - truth) < 4 * tb['gap_se']
    assert abs(-tb['integral'] - truth) < 0.05 * truth + 4 * tb['integral_se']


def test_mlp_ratio_recovers_mixture():
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=0)
    rng = np.random.default_rng(2); n = 40000
    z = W.sample_z(n, rng); Hp = W.sample(z, 1.0, rng); Hq = W.sample(z, 0.0, rng)
    rm, dg = R.fit_ratio(Hp, Hq, oh3(z), oh3(z), family='mlp', seed=0)
    z2 = W.sample_z(n, rng); Hq2 = W.sample(z2, 0.0, rng)
    lw = rm.log_w(Hq2, oh3(z2))
    assert np.corrcoef(lw, W.log_w(Hq2, z2, 0.0))[0, 1] > 0.85
    assert dg['ece'] < 0.02 and abs(dg['norm_q'] - 1) < 0.05
    c = E.cov_qw(W.G(Hq2, z2), lw, z2, B=100)
    assert 0.75 * W.delta(0.0) < -c['cov_plain'] < 1.1 * W.delta(0.0)


def test_quad_classifier_blind_beyond_second_moments():
    """Documented limit: a quadratic classifier cannot see the residual of a moment-matched Gaussian q."""
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=0)
    rng = np.random.default_rng(2); n = 30000
    z = W.sample_z(n, rng)
    _, dg = R.fit_ratio(W.sample(z, 1.0, rng), W.sample(z, 0.0, rng), oh3(z), oh3(z), family='quad', seed=0)
    assert dg['auc'] < 0.52


# ---------------------------------------------------------------- (iii) downstream ignores structure
def test_ignored_structure_has_zero_cov():
    net = synth.IgnoreNet(c_bimodal=True, seed=0)
    rng = np.random.default_rng(0); n = 30000
    y = rng.integers(0, 2, n); q = qfam.CondGaussian('full').fit(net.sample(y, rng), y)
    y2 = rng.integers(0, 2, n); H2 = net.sample(y2, rng); Gp = net.G(H2, y2)
    HqB = q.subset_sample(H2, net.B, y2, rng=rng)
    dB = E.direct_gap(net.G(HqB, y2), Gp)
    assert abs(dB['delta']) < 3 * dB['se'] + 1e-3
    HqC = q.subset_sample(H2, net.C, y2, rng=rng)
    dC = E.direct_gap(net.G(HqC, y2), Gp)
    assert abs(dC['delta']) > 4 * dC['se']
    # h-space classifier sees B; output-space classifier does not, and its Cov is ~0
    y3 = rng.integers(0, 2, n); H3 = net.sample(y3, rng); Hq3 = q.subset_sample(H3, net.B, y3, rng=rng)
    y4 = rng.integers(0, 2, n); H4 = net.sample(y4, rng)
    _, dh = R.fit_ratio(H4, Hq3, oh2(y4), oh2(y3), family='mlp', seed=0)
    assert dh['auc'] > 0.7
    rmo, do = R.fit_ratio(net.logits(H4), net.logits(Hq3), oh2(y4), oh2(y3), family='mlp', seed=0, null_gate=True)
    assert do['auc'] < 0.52
    c = E.cov_qw(net.G(HqB, y2), rmo.log_w(net.logits(HqB), oh2(y2)), y2, B=100)
    assert abs(c['cov_plain']) < 0.01


def test_cross_layer_consistency_pass_and_fail():
    out = {}
    for cb in (False, True):
        net = synth.IgnoreNet(c_bimodal=cb, seed=0)
        rng = np.random.default_rng(0)
        yy = rng.integers(0, 2, 100000); ql = qfam.CondGaussian('full').fit(net.sample(yy, rng), yy)
        H = ql.sample(yy[:1000], rng=rng)
        assert identity_check(H, yy[:1000], net.F, net.G_next, net.G) == 0.0
        out[cb] = consistency_check(lambda z, rg: ql.sample(z, rng=rg), net.F, lambda: qfam.CondGaussian('full'),
                                    lambda z, rg: net.F(net.sample(z, rg)), net.G_next, np.array([0, 1]),
                                    n_fit=60000, n_eval=60000, n_auc=30000, B=100)
    assert out[False]['passed'] and out[False]['passed_two_sided'] and out[False]['auc'] < 0.53
    assert (not out[True]['passed']) and out[True]['auc'] > 0.6


# ---------------------------------------------------------------- (iv) q -> p
def test_bridge_bounded_where_is_explodes():
    """d = 31, q 20% too wide: plain IS from q has chi^2 ~ 1e2 while the bridge stays well-conditioned."""
    S = synth.ScaleWorld(d=31, k=1.2)
    rng = np.random.default_rng(0); n = 20000
    Hp, Hq = S.sample_p(n, rng), S.sample_q(n, rng)
    tb = E.temper_bridge(S.G(Hp), S.log_w(Hp), S.G(Hq), S.log_w(Hq), np.linspace(0, 1, 11), B=50)
    assert abs(tb['gap'] - S.delta()) < 4 * tb['gap_se'] + 0.01
    assert tb['ess'].min() > 1000
    c = E.cov_qw(S.G(Hq), S.log_w(Hq), B=50)
    assert c['ess'] < tb['ess'].min()


def test_refit_spread_exceeds_ratio_fixed_se_under_null():
    """Under p = q the ratio-fixed bootstrap SE of Cov_q(G, w-hat) is far too small; the half-sampling refit
    spread is the dominant variance term and must be added (VALIDATION.md (i))."""
    W = synth.MixWorld(d=8, K=3, a=2.0, seed=1)
    rng = np.random.default_rng(7); n = 12000
    z = W.sample_z(n, rng); Hp, Hq = W.sample(z, 1.0, rng), W.sample(z, 1.0, rng)
    z2 = W.sample_z(n, rng); Hq2 = W.sample(z2, 1.0, rng); Gq2 = W.G(Hq2, z2)
    rm, _ = R.fit_ratio(Hp, Hq, oh3(z), oh3(z), family='quad', seed=0)
    c = E.cov_qw(Gq2, rm.log_w(Hq2, oh3(z2)), z2, B=100)
    sp = R.refit_spread(Hp, Hq, oh3(z), oh3(z), lambda m: E.cov_qw(Gq2, m.log_w(Hq2, oh3(z2)), z2, B=2)['cov_plain'],
                        B=6, family='quad')
    assert sp.std(ddof=1) > 2 * c['plain_se']
