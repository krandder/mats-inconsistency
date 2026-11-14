"""Unit tests for World FA (fa_core.py, world_fa.py). Run: OMP_NUM_THREADS=1 python -m pytest -q test_fa.py

Bayes vs independent brute force / Monte Carlo; reductions; permutation invariance and binding; generator-Bayes
calibration; tokens; dedup and gate sets."""
import math
import os
import sys

import numpy as np
import pytest
import torch
from scipy import stats

import fa_brute as Bf
import fa_core as C

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'worlds'))

CFG_A = C.full_cfg(dict(prior='a'))                    # the chosen regime (signed loadings, PREREG-FA.md)
CFG_B = C.full_cfg(dict(prior='b'))                    # the chosen regime (common-source mixture)
BASE_A = dict(prior='a', a_m=(0.5, 1.5, 4.0), a_pf=None, a_alpha=(1.0, 1.0, 1.0, 1.0), a_psign=0.0)
CFG_B1 = C.full_cfg(dict(prior='b', b_hyper=[(2.0, 2.0, 0.0, 1.0)]))   # one hyperparameter (grid_b needs it)


def moments(p):
    mu, var = C.mixture_moments(*p)
    return mu.numpy(), var.numpy()


def logdens(p, i, xs):
    w, m, v = (x[i].numpy() for x in p)
    k = w > 0
    lp = np.log(w[k])[None] - 0.5 * np.log(2 * np.pi * v[k])[None] - 0.5 * (xs[:, None] - m[k][None]) ** 2 / v[k][None]
    mx = lp.max(1, keepdims=True)
    return (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))[:, 0]


# ------------------------------------------------------------------ (a) exact Bayes vs brute-force enumeration
@pytest.mark.parametrize('cfg_over', [{}, dict(a_psign=0.3), dict(a_alpha=(1.0, 3.0, 0.5)), 'chosen',
                                      dict(a_pm=(1, 2, 3), a_pf=(3, 1, 1, 1, 2)),
                                      dict(a_pm=(1, 1, 3), a_pm_block=(4, 1, 1))])
@pytest.mark.parametrize('N', [1, 2, 3, 4])
def test_a_matches_bruteforce(N, cfg_over):
    if cfg_over == 'chosen':
        cfg_over = {k: v for k, v in C.DEFAULTS.items() if k.startswith('a_')}
    if cfg_over.get('a_psign') and N > 3:
        pytest.skip('signed enumeration grows as 2^N per block; N <= 3 is enough')
    cfg = C.full_cfg(dict(BASE_A, **cfg_over))
    torch.manual_seed(10 * N + len(cfg_over))
    b = C.gen(4, N, torch.tensor([0, 1, 3, 7]), cfg)
    p, lev = C.bayes_a(b, cfg)
    mu, var = moments(p)
    eh, _ = C.hist_stats(b)
    for i in range(4):
        T = int(b['T'][i])
        w, m, v, blev = Bf.brute_a(eh[i, :T, :N].numpy(), b['r'][i, :N].numpy(), cfg)
        bmu = (w * m).sum(); bvar = (w * (v + m ** 2)).sum() - bmu ** 2
        assert abs(mu[i] - bmu) < 5e-5 * max(1, abs(bmu))
        assert abs(var[i] - bvar) < 1e-4 * bvar
        assert abs(lev[i].item() - blev) < 1e-6 * max(1, abs(blev))
        xs = bmu + math.sqrt(bvar) * np.linspace(-3, 3, 13)
        dens_b = np.log((w[None] * np.exp(-0.5 * (xs[:, None] - m[None]) ** 2 / v[None]) / np.sqrt(2 * np.pi * v[None])).sum(1))
        assert np.abs(logdens(p, i, xs) - dens_b).max() < 2e-3


# ------------------------------------------------------------------ (b) exact Bayes vs quadrature and importance sampling
@pytest.mark.parametrize('N,T', [(1, 0), (2, 0), (3, 2), (5, 10), (8, 3)])
def test_b_matches_v_quadrature(N, T):
    torch.manual_seed(N + 100 * T)
    b = C.gen(4, N, T, CFG_B1)
    p, _ = C.bayes_b(b, CFG_B1)
    mu, var = moments(p)
    eh, _ = C.hist_stats(b)
    for i in range(4):
        vs, lp = Bf.grid_b(eh[i].numpy(), b['r'][i].numpy(), CFG_B1)
        dx = vs[1] - vs[0]; pd = np.exp(lp)
        gm = (pd * vs).sum() * dx; gv = (pd * vs ** 2).sum() * dx - gm ** 2
        assert abs(mu[i] - gm) < 1e-6 and abs(var[i] - gv) < 1e-6
        xs = gm + math.sqrt(gv) * np.linspace(-4, 4, 17)
        assert np.abs(logdens(p, i, xs) - np.interp(xs, vs, lp)).max() < 1e-5


@pytest.mark.parametrize('N,T', [(2, 0), (2, 3), (3, 1)])
def test_b_matches_importance_sampling(N, T):
    cfg = C.full_cfg(dict(prior='b', b_hyper=[(1.0, 1.0, 0.0, 0.5), (3.0, 3.0, 0.4, 0.5)]))
    torch.manual_seed(7 + N + T)
    b = C.gen(3, N, T, cfg)
    p, lev = C.bayes_b(b, cfg)
    mu, var = moments(p)
    eh, _ = C.hist_stats(b)
    for i in range(3):
        m1, v1, ess, ilev = Bf.is_b(eh[i].numpy(), b['r'][i].numpy(), cfg, M=200000, seed=i)
        assert ess > 2000
        tol = 6 / math.sqrt(ess)
        assert abs(mu[i] - m1) < tol * math.sqrt(v1) and abs(var[i] / v1 - 1) < 3 * tol
        assert abs(lev[i].item() - ilev) < 0.03


# ------------------------------------------------------------------ generator <-> Bayes consistency (Monte Carlo)
@pytest.mark.parametrize('prior,E,nmax', [('a', 1500, 5), ('b', 20000, 12)])
def test_calibration_against_generator(prior, E, nmax):
    """For episodes drawn from the generator: E (v - m)^2 = E var (exact identity) and PIT ~ U(0, 1); Bayes has the
    lowest NLL among the non-oracle references."""
    cfg = C.full_cfg(dict(prior=prior))
    torch.manual_seed(42)
    b = C.gen(E, torch.randint(2, nmax + 1, (E,)), C.sample_T(E, cfg), cfg)
    refs = C.references(b, cfg)
    for name in ('bayes', 'bayes_T0'):
        w, m, v = refs[name]
        mu, var = C.mixture_moments(w, m, v)
        d = ((mu - b['v']) ** 2 - var).numpy()
        assert abs(d.mean()) < 4 * d.std() / math.sqrt(E), name
        pit = (w * 0.5 * (1 + torch.erf((b['v'][:, None] - m) / torch.sqrt(2 * v)))).sum(1).numpy()
        assert stats.kstest(pit, 'uniform').pvalue > 1e-3, name
    nl = {k: C.nll_t(p, b['v']).numpy() for k, p in refs.items()}
    for k in nl:
        if k in ('bayes', 'oracle'):
            continue
        d = nl[k] - nl['bayes']
        assert d.mean() > -2 * d.std() / math.sqrt(E), k


# ------------------------------------------------------------------ reductions
@pytest.mark.parametrize('cfg', [CFG_A, CFG_B], ids=['a', 'b'])
def test_T0_is_prior_level(cfg):
    """T = 0: full Bayes equals the history-free rule, and depends on the query reports only through their
    multiset (exchangeable prior)."""
    torch.manual_seed(3)
    b = C.gen(16, 5, 0, cfg)
    p1, p0 = C.bayes(b, cfg, True), C.bayes(b, cfg, False)
    assert np.allclose(moments(p1)[0], moments(p0)[0], atol=1e-10)
    b2 = dict(b); perm = torch.randperm(5)
    b2['r'] = b['r'][:, perm]
    m2, v2 = moments(C.bayes(b2, cfg))
    assert np.allclose(m2, moments(p1)[0], atol=1e-7) and np.allclose(v2, moments(p1)[1], atol=1e-7)


def test_b_known_sigma_is_gls():
    """IW concentrated at M (delta -> infinity): Bayes = GLS with Sigma = M."""
    sig2, rho = 1.7, 0.35
    cfg = C.full_cfg(dict(prior='b', b_hyper=[(1e7, sig2, rho, 1.0)]))
    torch.manual_seed(5)
    b = C.gen(8, 6, 4, cfg)
    mu, var = moments(C.bayes(b, cfg))
    M = sig2 * ((1 - rho) * torch.eye(6, dtype=torch.float64) + rho)
    gm, gv = C.gls(M.expand(8, 6, 6), b['r'], torch.ones(8, 6, dtype=torch.float64))
    assert np.allclose(mu, gm.numpy(), atol=2e-5) and np.allclose(var, gv.numpy(), rtol=2e-5)


def test_b_long_history_approaches_oracle():
    """T -> infinity: the IW posterior concentrates at the true Sigma: Bayes -> GLS with true Sigma."""
    torch.manual_seed(6)
    b = C.gen(8, 4, 20000, CFG_B)
    mu, var = moments(C.bayes(b, CFG_B))
    gm, gv = C.gls(b['Sigma'], b['r'], torch.ones(8, 4, dtype=torch.float64))
    assert np.abs(mu - gm.numpy()).max() < 0.03 * np.sqrt(gv.numpy()).max() + 1e-3
    assert np.abs(var / gv.numpy() - 1).max() < 0.03


def test_a_single_hypothesis_is_gls():
    """Family with one hypothesis (all singletons, one level): Bayes = GLS with Sigma = s2 I (equal weights)."""
    cfg = C.full_cfg(dict(prior='a', a_alpha=(1.0,), a_m=(1.5,), a_pm=None))
    torch.manual_seed(8)
    b = C.gen(6, 5, 3, cfg)
    mu, var = moments(C.bayes(b, cfg))
    gm, gv = C.gls(1.5 * torch.eye(5, dtype=torch.float64).expand(6, 5, 5), b['r'], torch.ones(6, 5, dtype=torch.float64))
    assert np.allclose(mu, gm.numpy(), atol=1e-6) and np.allclose(var, gv.numpy(), rtol=1e-5)


def test_a_diagonal_structure_is_precision_weighting():
    """No shared sources (block size <= 1), heterogeneous levels, long history: Bayes = precision weighting with
    the true per-trader variances."""
    cfg = C.full_cfg(dict(prior='a', a_alpha=(1.0,), a_pm=None))
    torch.manual_seed(9)
    b = C.gen(6, 6, 400, cfg)
    mu, var = moments(C.bayes(b, cfg))
    d = torch.diagonal(b['Sigma'], dim1=1, dim2=2)
    w = 1 / d
    pm = ((w * b['r']).sum(1) / (1 + w.sum(1))).numpy(); pv = (1 / (1 + w.sum(1))).numpy()
    assert np.allclose(mu, pm, atol=1e-6) and np.allclose(var, pv, rtol=1e-5)


# ------------------------------------------------------------------ permutation invariance and binding
def _permute(b, perm):
    out = dict(b)
    out['r'] = b['r'][:, perm]; out['r_hist'] = b['r_hist'][:, :, perm]
    out['Sigma'] = b['Sigma'][:, perm][:, :, perm]; out['uid'] = b['uid'][:, perm]
    return out


@pytest.mark.parametrize('cfg', [CFG_A, CFG_B], ids=['a', 'b'])
def test_consistent_permutation_invariance_and_binding(cfg):
    torch.manual_seed(11)
    b = C.gen(24, 6, 12, cfg)
    mu, var = moments(C.bayes(b, cfg))
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    bp = _permute(b, perm)
    mp, vp = moments(C.bayes(bp, cfg))
    assert np.allclose(mu, mp, atol=1e-8) and np.allclose(var, vp, rtol=1e-7)
    for k, p in C.references(b, cfg, with_bayes=False).items():      # references too
        q = C.references(bp, cfg, with_bayes=False)[k]
        assert np.allclose(moments(p)[0], moments(q)[0], atol=1e-8), k
    # tokens: the same multiset of tokens
    X, M = C.tokens(b, cfg); Xp, Mp = C.tokens(bp, cfg)
    for i in range(3):
        a = X[i][M[i]].numpy(); c = Xp[i][Mp[i]].numpy()
        assert np.allclose(a[np.lexsort(a.T[::-1])], c[np.lexsort(c.T[::-1])])
    # binding: permuting history columns RELATIVE to the query round changes Bayes
    bs = dict(b); bs['r_hist'] = b['r_hist'][:, :, perm]
    ms, vs = moments(C.bayes(bs, cfg))
    assert np.mean(np.abs(ms - mu) / np.sqrt(var)) > 0.05


# ------------------------------------------------------------------ tokens
def test_tokens_layout():
    cfg = CFG_B
    torch.manual_seed(12)
    b = C.gen(5, torch.tensor([2, 3, 4, 3, 2]), torch.tensor([0, 1, 5, 2, 3]), cfg)
    X, M = C.tokens(b, cfg)
    di, dr = cfg['d_id'], cfg['d_round']
    o = di + dr
    Nm, Tm = 4, 5
    assert X.shape == (5, Nm * (Tm + 1) + 1, o + 8)
    for i in range(5):
        n, t = int(b['N'][i]), int(b['T'][i])
        lN, lT = math.log(n), math.log1p(t)
        assert int(M[i].sum()) == n * (t + 1) + 1
        q = X[i, -1]
        assert q[o + C.F_QUERY] == 1 and q[:o + C.F_QUERY].abs().sum() == 0
        assert abs(q[o + C.F_LOGN] - lN) < 1e-6 and abs(q[o + C.F_LOGT] - lT) < 1e-6
        for tt in range(t):
            for j in range(n):
                x = X[i, tt * Nm + j]
                assert torch.allclose(x[:di], b['uid'][i, j].float()) and torch.allclose(x[di:o], b['rid'][i, tt].float())
                assert abs(x[o + C.F_R] - b['r_hist'][i, tt, j]) < 1e-5 and abs(x[o + C.F_V] - b['v_hist'][i, tt]) < 1e-5
                assert abs(x[o + C.F_E] - (b['r_hist'][i, tt, j] - b['v_hist'][i, tt])) < 1e-5
                assert x[o + C.F_HIST] == 1 and x[o + C.F_QREP] == 0
                assert abs(x[o + C.F_LOGN] - lN) < 1e-6 and abs(x[o + C.F_LOGT] - lT) < 1e-6
        for j in range(n):
            x = X[i, Tm * Nm + j]
            assert abs(x[o + C.F_R] - b['r'][i, j]) < 1e-5 and x[o + C.F_V] == 0 and x[o + C.F_E] == 0
            assert x[o + C.F_QREP] == 1 and x[o + C.F_HIST] == 0
            assert torch.allclose(x[di:o], b['rid'][i, Tm].float())


def test_tokens_column_mode():
    """Comparator tokens: 2N + 1; history token i carries h_i = T^-1/2 sum_t e_ti u_t (u_t = sqrt(d) x round id)."""
    cfg = C.full_cfg(dict(prior='b', tok_mode='column'))
    torch.manual_seed(16)
    b = C.gen(4, torch.tensor([3, 4, 2, 4]), torch.tensor([6, 0, 3, 9]), cfg)
    X, M = C.tokens(b, cfg)
    di, dr = cfg['d_id'], cfg['d_round']; o = di + dr
    assert X.shape == (4, 2 * 4 + 1, o + 8)
    eh, _ = C.hist_stats(b)
    for i in range(4):
        n, t = int(b['N'][i]), int(b['T'][i])
        assert int(M[i].sum()) == (n if t > 0 else 0) + n + 1
        for j in range(n):
            if t > 0:
                h = (eh[i, :t, j, None] * b['rid'][i, :t] * math.sqrt(dr)).sum(0) / math.sqrt(t)
                assert torch.allclose(X[i, j, di:o], h.float(), atol=1e-5) and X[i, j, o + C.F_HIST] == 1
            assert abs(X[i, 4 + j, o + C.F_R] - b['r'][i, j]) < 1e-5 and X[i, 4 + j, o + C.F_QREP] == 1


# ------------------------------------------------------------------ prior (f): enumerated family
def test_f_family_sizes():
    k = lambda tpl, N: len(C.family(N, C._tkey([dict(marg=1.0, **t) for t in tpl]))[1])
    assert k([{'blocks': [[2, 0.8, 0]]}], 4) == 6                     # GPT-6 isolating case
    assert k([{'blocks': [[2, 0.8, 1]]}], 4) == 6                     # anti-correlated pair (sign flip = same Sigma)
    assert k([{'blocks': [[5, 0.8, 0]]}], 8) == 56                    # Gemini: 5-clique + 3 singletons
    assert k([{'blocks': [[2, 0.8, 0], [2, 0.5, 0]]}], 4) == 6         # pair A fixes pair B
    assert k([{'blocks': [[2, 0.8, 0], [2, 0.8, 0]]}], 4) == 3        # identical blocks are unordered
    assert k([{'blocks': [[2, 0.8, 0]]}, {'blocks': [[3, 0.6, 1]], 'common': 0.3}], 5) == 10 + 10 * 3


F_TPL = [{'blocks': [[2, 0.8, 0]], 'marg': 1.0}, {'blocks': [[3, 0.7, 1]], 'marg': 1.5, 'common': 0.4, 'w': 2.0}]


@pytest.mark.parametrize('T', [0, 3, 12])
def test_f_matches_bruteforce_and_query_marginal_matters(T):
    """Weights must include the query reports' marginal N(r; 0, Sigma_h + 11^T) (critique item 1): our posterior
    equals an independent scipy enumeration with that factor, and differs from history-only weighting."""
    from scipy.special import logsumexp
    cfg = C.full_cfg(dict(prior='f', f_templates=F_TPL))
    S, lp, _, _ = C.family(5, C._tkey(cfg['f_templates']))
    torch.manual_seed(17 + T)
    b = C.gen(6, 5, T, cfg)
    (w, m, v), lev = C.bayes_f(b, cfg)
    (w0, m0, v0), _ = C.bayes_f(b, cfg, query_weights=False)
    eh, _ = C.hist_stats(b)
    for i in range(6):
        r = b['r'][i].numpy()
        lw = np.array([lp[h] + (stats.multivariate_normal(np.zeros(5), S[h]).logpdf(eh[i, :T].numpy()).sum() if T else 0)
                       + stats.multivariate_normal(np.zeros(5), S[h] + 1).logpdf(r) for h in range(len(lp))])
        ww = np.exp(lw - logsumexp(lw))
        assert np.abs(ww - w[i].numpy()).max() < 1e-9 and abs(logsumexp(lw) - lev[i].item()) < 1e-8
        mu = [(np.linalg.inv(S[h]).sum(0) @ r) / (1 + np.linalg.inv(S[h]).sum()) for h in range(len(lp))]
        assert np.allclose(m[i].numpy(), mu, atol=1e-9)
    assert (w - w0).abs().max() > 0.05


def test_f_calibration_against_generator():
    cfg = C.full_cfg(dict(prior='f', f_templates=F_TPL))
    torch.manual_seed(18)
    E = 20000
    b = C.gen(E, 5, C.sample_T(E, cfg), cfg)
    w, m, v = C.bayes(b, cfg)
    mu, var = C.mixture_moments(w, m, v)
    d = ((mu - b['v']) ** 2 - var).numpy()
    assert abs(d.mean()) < 4 * d.std() / math.sqrt(E)
    pit = (w * 0.5 * (1 + torch.erf((b['v'][:, None] - m) / torch.sqrt(2 * v)))).sum(1).numpy()
    assert stats.kstest(pit, 'uniform').pvalue > 1e-3


def test_f_heldout_structures_and_beta_star():
    import world_fa as WF
    tpl = [dict(F_TPL[0], train=True), dict(F_TPL[1], train=False)]
    W = WF.make_world(dict(prior='f', f_templates=tpl))
    ep = W.sample_train(np.random.default_rng(0), 400, 5, 5)
    assert (ep['b']['template'] == 0).all()                            # training never shows held-out structures
    ep = W.sample(np.random.default_rng(1), 200, 5, 'freshT16')
    assert (ep['b']['template'] == 1).all()
    refs = W.references(ep)
    assert 'bayes_train' in refs
    nb = refs['bayes']; nt = refs['bayes_train']
    import harness as H
    assert H.nll(nt, ep['target']).mean() > H.nll(nb, ep['target']).mean() + 0.05
    # beta* -> Sigma^-1 1 when the structure is identified
    cfg = C.full_cfg(dict(prior='f', f_templates=F_TPL))
    torch.manual_seed(19)
    b = C.gen(20, 5, 400, cfg)
    beta = C.beta_star(b, cfg)
    target = torch.linalg.solve(b['Sigma'], torch.ones(20, 5, 1, dtype=torch.float64))[..., 0]
    assert torch.allclose(beta, target, atol=1e-6)


def test_T_curriculum():
    import world_fa as WF
    W = WF.make_world(dict(prior='b', T_curr=[[0, 32, 64, 0.0], [3, 0, 8, 0.5]]))
    rng = np.random.default_rng(0)
    Ts = [W.sample_train(rng, 64, 3, 5)['b']['T'] for _ in range(6)]
    assert int(Ts[0].max()) <= 8                                       # first call = harness val set: final stage
    assert all(int(t.min()) >= 32 for t in Ts[1:4]) and all(int(t.max()) <= 8 for t in Ts[4:])


# ------------------------------------------------------------------ one-hop dedup and history gate sets
@pytest.mark.parametrize('cfg', [CFG_A, CFG_B], ids=['a', 'b'])
def test_dedup_and_gate_sets(cfg):
    import harness as H
    import world_fa as WF
    torch.manual_seed(13)
    G, N, T, k = 40, 6, 16, 3
    b = C.block_sets(G, N, T, k, WF.DEDUP_FS, cfg)
    A = len(WF.DEDUP_FS)
    v = b['v'].reshape(G, A)
    assert torch.allclose(v, v[:, :1].expand(G, A))                     # common values across arms
    Sig = b['Sigma'].reshape(G, A, N, N)
    for a, f in enumerate(WF.DEDUP_FS):                                  # marginal accuracies equal across arms
        assert torch.allclose(torch.diagonal(Sig[:, a], dim1=1, dim2=2), torch.diagonal(Sig[:, 0], dim1=1, dim2=2))
        cor = Sig[:, a, 0, 1] / Sig[:, a, 0, 0]
        assert torch.allclose(cor, torch.full_like(cor, f))
    _, var = moments(C.bayes(b, cfg))
    lv = np.log(var.reshape(G, A))
    assert lv[:, -1].mean() > lv[:, 0].mean()                            # near-copies: less information
    g = C.block_sets(G, N, T, k, WF.GATE_FS, cfg, gate=True)
    r = g['r'].reshape(G, A, N)
    assert torch.allclose(r, r[:, :1].expand(G, A, N))                  # identical query reports
    p = [x.numpy().reshape(G, A, -1) for x in C.bayes(g, cfg)]
    kl = H.kl_bayes_model(tuple(x[:, 0] for x in p), tuple(x[:, -1] for x in p))
    assert kl.mean() > 0.05                                              # history alone moves Bayes


def test_shuffled_history_control():
    torch.manual_seed(14)
    b = C.gen(40, 5, 16, CFG_B)
    s = C.shuffle_history(b)
    assert torch.equal(s['r'], b['r']) and torch.equal(s['v_hist'], b['v_hist'])
    eh, _ = C.hist_stats(b); es, _ = C.hist_stats(s)
    for i in range(3):                                                  # columns are a permutation
        assert torch.allclose(es[i], eh[i][:, s['perm'][i]])


@pytest.mark.parametrize('cfg', [CFG_A, CFG_B], ids=['a', 'b'])
def test_diag_bayes_equals_bayes_for_one_trader(cfg):
    """N = 1 has no correlations: the correlation-blind Bayes reference equals full Bayes."""
    torch.manual_seed(15)
    b = C.gen(12, 1, torch.tensor([0, 1, 4, 9] * 3), cfg)
    m1, v1 = moments(C.bayes(b, cfg)); m2, v2 = moments(C.diag_bayes(b, cfg))
    assert np.allclose(m1, m2, atol=1e-6) and np.allclose(v1, v2, rtol=1e-4)


# ------------------------------------------------------------------ binding arms (Amendment A3)
def test_ortho_ids():
    cfg = C.full_cfg(dict(prior='f', id_ortho=True))
    torch.manual_seed(20)
    b = C.gen(16, 4, 8, cfg)
    G = b['uid'] @ b['uid'].transpose(1, 2)
    assert torch.allclose(G, torch.eye(4).expand(16, 4, 4), atol=1e-5)


def test_attention_binding_fixtures():
    """Uniform attention gives ratio 1 exactly; an id-matching first layer gives a large own/other ratio."""
    import harness as H
    from diag_fa import attention_binding
    cfg = C.full_cfg(dict(prior='f', id_ortho=True))
    torch.manual_seed(21)
    N, T = 4, 6
    b = C.gen(64, N, T, cfg)
    X, M = C.tokens(b, cfg)
    F, di = C.feat_dim(cfg), cfg['d_id']
    model = H.SetPFN(F, 1, 64, 1, 'real', 2)
    with torch.no_grad():
        model.blocks[0].qkv.weight.zero_(); model.blocks[0].qkv.bias.zero_()
    r = attention_binding(model, X, M, N, T)
    assert abs(r['ratio_by_layer'][0] - 1) < 1e-9
    with torch.no_grad():
        model.inp.weight.zero_(); model.inp.bias.zero_()
        model.inp.weight[:di, :di] = torch.eye(di)                     # residual stream carries the trader id
        W = torch.zeros(3 * 64, 64); W[:di, :di] = 20 * torch.eye(di); W[64:64 + di, :di] = 20 * torch.eye(di)
        model.blocks[0].qkv.weight.copy_(W)
    r = attention_binding(model, X, M, N, T)
    assert r['ratio_by_layer'][0] > 20


# ------------------------------------------------------------------ A5 decomposition arms
ANTI = [{'blocks': [[2, 0.8, 1]], 'marg': 1.0}]


def test_qcol_qcov_tokens():
    torch.manual_seed(22)
    for mode in ('qcol', 'qcov'):
        cfg = C.full_cfg(dict(prior='f', f_templates=ANTI, tok_mode=mode, id_ortho=(mode == 'qcov')))
        b = C.gen(6, 4, 10, cfg)
        X, M = C.tokens(b, cfg)
        o = cfg['d_id'] + cfg['d_round']; base = o + 8; K = cfg['tok_col']
        assert X.shape == (6, 5, C.feat_dim(cfg)) and bool(M.all())
        eh, S = C.hist_stats(b)
        for i in range(4):
            col = X[:, i, base:base + K]
            assert torch.allclose(col[:, :10], (eh[:, :, i] / math.sqrt(10)).float(), atol=1e-6)
            assert col[:, 10:].abs().max() == 0
            assert torch.allclose(X[:, i, o + C.F_R], b['r'][:, i].float())
        # inner products of the columns are the sample covariances S / T
        cols = X[:, :4, base:base + K].double()
        assert torch.allclose(cols @ cols.transpose(1, 2), S / 10, atol=1e-5)
        if mode == 'qcov':
            c = X[:, :4, base + K + 1:base + K + 1 + cfg['d_id']].double()
            read = c @ b["uid"].double().transpose(1, 2)                # <c_i, uid_j> = S_ij / T off the diagonal
            off = ~torch.eye(4, dtype=torch.bool)
            assert torch.allclose(read[:, off], (S / 10)[:, off], atol=1e-5)
            assert torch.allclose(X[:, :4, base + K].double(), torch.diagonal(S / 10, dim1=1, dim2=2), atol=1e-5)


def test_templates_by_N_and_partner_ratio():
    import harness as H
    from diag_fa import attention_binding
    cfg = C.full_cfg(dict(prior='f', f_templates=ANTI,
                          f_templates_byN={'2': [ANTI[0], {'blocks': [], 'marg': 1.0}]}))
    assert len(C.family(2, C._tkey(C.templates_for(cfg, 2)))[1]) == 2
    assert len(C.family(4, C._tkey(C.templates_for(cfg, 4)))[1]) == 6
    torch.manual_seed(23)
    b2 = C.gen(2000, 2, 0, cfg)
    corr = b2['Sigma'][:, 0, 1]
    assert 0.4 < float((corr < 0).double().mean()) < 0.6                  # anti pair or independent, 50/50
    # partner ratio: uniform attention -> 1; a first layer matching ids of partners is not needed for the fixture
    cq = C.full_cfg(dict(prior='f', f_templates=ANTI, tok_mode='qcol'))
    b = C.gen(64, 4, 6, cq)
    X, M = C.tokens(b, cq)
    model = H.SetPFN(C.feat_dim(cq), 1, 64, 1, 'real', 2)
    with torch.no_grad():
        model.blocks[0].qkv.weight.zero_(); model.blocks[0].qkv.bias.zero_()
    r = attention_binding(model, X, M, 4, 6, cfg=cq, sigma=b['Sigma'])
    assert abs(r['partner_ratio_by_layer'][0] - 1) < 1e-9 and 'ratio_by_layer' not in r
    # a column-matching first layer (q = k = error column) attends to the anti-correlated partner LESS than to others
    o = cq['d_id'] + cq['d_round'] + 8
    with torch.no_grad():
        model.inp.weight.zero_(); model.inp.bias.zero_()
        model.inp.weight[:64, o:o + 64] = torch.eye(64)
        W = torch.zeros(192, 64); W[:64] = torch.eye(64); W[64:128] = torch.eye(64)
        model.blocks[0].qkv.weight.copy_(W)
    b = C.gen(64, 4, 60, cq)
    X, M = C.tokens(b, cq)
    r = attention_binding(model, X, M, 4, 60, cfg=cq, sigma=b['Sigma'])
    assert r['partner_ratio_by_layer'][0] < 0.5


# ------------------------------------------------------------------ A7
def test_slot_ids_and_random_column_slots():
    cfg = C.full_cfg(dict(prior='f', f_templates=ANTI, id_mode='slot'))
    torch.manual_seed(24)
    b = C.gen(5, 4, 3, cfg)
    assert torch.equal(b['uid'], torch.eye(cfg['d_id'])[:4].expand(5, 4, cfg['d_id']))
    cq = C.full_cfg(dict(prior='f', f_templates=ANTI, tok_mode='qcol', tok_col=128, col_random_slots=True))
    b = C.gen(6, 4, 10, cq)
    X, M = C.tokens(b, cq)
    base = cq['d_id'] + cq['d_round'] + 8
    cols = X[:, :4, base:base + 128].double()
    eh, S = C.hist_stats(b)
    assert torch.allclose(cols @ cols.transpose(1, 2), S / 10, atol=1e-5)        # inner products unchanged
    nz = (cols.abs() > 0).any(1)                                                  # occupied slots = the rounds' slots
    for e in range(6):
        assert set(torch.nonzero(nz[e])[:, 0].tolist()) <= set(b['colperm'][e, :10].tolist())


def test_pair_sets_f():
    cfg = C.full_cfg(dict(prior='f', f_templates=ANTI))
    torch.manual_seed(25)
    B1 = 0.2 * np.eye(3) + 0.8
    b = C.pair_sets_f(10, 6, 8, [np.eye(3), B1], cfg, gate=True)
    assert torch.equal(b['r'][0::2], b['r'][1::2]) and torch.equal(b['v'][0::2], b['v'][1::2])
    assert torch.allclose(b['Sigma'][1, :3, :3], torch.as_tensor(B1))
    assert not torch.equal(b['r_hist'][0::2], b['r_hist'][1::2])
