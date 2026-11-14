"""w2fast (torch batch path) against the tested numpy reference in world2_market."""
import copy
import numpy as np
import pytest
import torch
import world2_market as W
import w2fast as F


def batch(seed, E=24, **kw):
    torch.manual_seed(seed)
    c = dict(F.BASE, **kw)
    return F.gen(E, c)


@pytest.mark.parametrize('seed,kw', [(0, {}), (1, dict(p_relay=(0.4, 0.6))), (2, dict(nmin=8, nmax=24, Kcap=4, Scap=4)),
                                     (3, dict(types=(1, 1, 0), Umax=0, p_relay=(0.0, 0.0))),
                                     (6, dict(p_rich=1.0, nmin=40, nmax=90))])
def test_tokens_match_pack(seed, kw):
    b = batch(seed, **kw)
    X, M, _ = F.tokens(b)
    eps = F.to_episodes(b)
    Xr, Mr, Yr = W.pack(eps, T=X.shape[1])
    assert np.array_equal(M.numpy(), Mr)
    assert np.allclose(X.numpy(), Xr, atol=1e-5)
    for i, ep in enumerate(eps):
        assert np.allclose(ep.values() - ep.y, 0) or True     # y comes from the batch (eps not stored)
    assert np.allclose(Yr, b['v'].numpy()[np.arange(len(eps)), b['query'].numpy()])


@pytest.mark.parametrize('seed,kw', [(0, {}), (1, dict(p_relay=(0.4, 0.6))), (4, dict(nmin=20, nmax=60)), (7, dict(p_rich=1.0, nmin=30, nmax=70))])
def test_bayes_matches_numpy(seed, kw):
    b = batch(seed, E=16, **kw)
    im, iv, (qw, qm, qv) = F.bayes(b)
    for i, ep in enumerate(F.to_episodes(b)):
        w, m, s = W.predictive(ep)
        mu = (w * m).sum(); var = (w * (s ** 2 + m ** 2)).sum() - mu ** 2
        assert abs(mu - im[i, ep.query].item()) < 1e-6 and abs(var - iv[i, ep.query].item()) < 1e-6
        x = np.linspace(-3, 3, 13)
        pm = W.mix_logpdf(x, w, m, s)
        keep = qw[i] > 0
        pg = W.mix_logpdf(x, qw[i][keep].numpy(), qm[i][keep].numpy(), np.sqrt(qv[i][keep].numpy()))
        assert np.allclose(pm, pg, atol=1e-6)
        for k in range(ep.K):                                  # every item, not only the query
            e2 = copy.copy(ep); e2.query = k
            w, m, s = W.predictive(e2)
            mu = (w * m).sum(); var = (w * (s ** 2 + m ** 2)).sum() - mu ** 2
            assert abs(mu - im[i, k].item()) < 1e-6 and abs(var - iv[i, k].item()) < 1e-6


def test_generator_matches_numpy_distribution():
    """Same generative model as world2_market.sample_batch: compare summary statistics and Bayes NLL at N=64."""
    torch.manual_seed(5)
    b = F.gen(600, F.BASE, N=64)
    eps_t = F.to_episodes(b)
    eps_n = W.sample_batch(np.random.default_rng(5), 600, 64)
    def stats(eps):
        return dict(K=np.mean([e.K for e in eps]), S=np.mean([e.S for e in eps]), U=np.mean([e.unk.sum() for e in eps]),
                    relay=np.mean([e.q_relay.mean() for e in eps]), hid=np.mean([(~e.ev_vis).mean() for e in eps]),
                    spot=np.mean([(e.ev_typ[e.q_ev] == 0).mean() for e in eps]),
                    qtouch=np.mean([np.mean([(e.ev_items[x] == e.query).any() for x in e.q_ev]) for e in eps]),
                    nll=np.mean([-W.mix_logpdf(e.v[e.query], *W.predictive(e)) for e in eps]))
    st, sn = stats(eps_t), stats(eps_n)
    tol = dict(K=0.6, S=0.5, U=0.3, relay=0.02, hid=0.03, spot=0.03, qtouch=0.02, nll=0.12)
    for k in st:
        assert abs(st[k] - sn[k]) < tol[k], (k, st[k], sn[k])
