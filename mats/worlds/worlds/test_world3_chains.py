import numpy as np
import pytest
import world3_chains as W


def _one(ch, b, m=0):
    keys = ['U', 'ru', 'cu', 're', 'ce'] + (['Mj', 'rm', 'cm'] if 'Mj' in ch else [])
    return {k: ch[k][b, m] for k in keys}


@pytest.mark.parametrize('maj', [False, True])
def test_forward_backward_matches_enumeration(maj):
    rng = np.random.default_rng(0)
    cfg = W.cfg_of(maj=maj, n=9, M=8, V=128)
    for l in range(0, 9):
        ep = W.sample(6, rng, cfg, l)
        ch = ep['chains']
        fb = W.chain_marginals(**_one(ch, slice(None)))
        for b in range(6):
            bf = W.brute_marginals(**_one(ch, b))
            np.testing.assert_allclose(fb[b], bf, atol=1e-10)


@pytest.mark.parametrize('maj', [False, True])
def test_oracle_is_function_of_visible_messages(maj):
    """Policy invariance: the oracle recomputed from the token list alone (no l, no positions, no policy) agrees,
    including after relabelling ids, reordering tokens and flipping XOR orientation."""
    rng = np.random.default_rng(1)
    cfg = W.cfg_of(maj=maj)
    for l in (0, 2, 5, 8):
        ep = W.sample(5, rng, cfg, l)
        ref = W.references(ep)['bayes']
        for b in range(5):
            p = W.posterior_from_tokens(ep, b, cfg)
            assert abs(p - ref[b]) < 1e-5
            # relabel ids, reverse message order, swap XOR orientation: same posterior
            e2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ep.items()}
            perm = rng.permutation(cfg['V'])
            ids = e2['ids'][b]
            e2['ids'][b] = np.where(ids == cfg['V'], cfg['V'], perm[np.minimum(ids, cfg['V'] - 1)])
            T = len(e2['typ'][b]) - 1
            o = np.r_[np.arange(T)[::-1], T]
            for k in ('typ', 'ids', 'bit', 'rel', 'valid'):
                e2[k][b] = e2[k][b][o]
            x = e2['typ'][b] == W.XOR
            e2['ids'][b][x, :2] = e2['ids'][b][x][:, ::-1][:, 1:]
            assert abs(W.posterior_from_tokens(e2, b, cfg) - ref[b]) < 1e-5


def test_two_policies_same_data_same_posterior():
    """Same messages produced under different (visible) l / distractor policies give the same oracle."""
    rng = np.random.default_rng(2)
    cfg = W.cfg_of()
    ep = W.sample(64, rng, cfg, 3)
    ch = dict(ep['chains'])
    ch2 = dict(ch); ch2['L'] = np.zeros_like(ch['L'])          # pretend another policy chose these chains
    e2 = dict(ep); e2['chains'] = ch2
    np.testing.assert_array_equal(W.references(ep)['bayes'], W.references(e2)['bayes'])


def test_query_distance_and_structure():
    rng = np.random.default_rng(3)
    cfg = W.cfg_of()
    for l in range(0, 9):
        ep = W.sample(200, rng, cfg, l)
        ch = ep['chains']
        q = ch['qpos'][:, 0]
        d = np.abs(np.arange(cfg['n'])[None] - q[:, None])
        dmin = np.where(ch['U'][:, 0], d, 99).min(1)
        assert (dmin == l).all()
        # every chain (distractors too) has its nearest report at its own virtual distance
        assert ch['U'].any(-1).all()
        # one query token, last; ids distinct within an episode
        assert (ep['typ'][:, -1] == W.QUERY).all() and (ep['typ'][:, :-1] != W.QUERY).all()
        ni = ep['node_ids'].reshape(200, -1)
        assert all(len(set(r)) == len(r) for r in ni)
        # query id is the id of the queried node
        assert (ep['ids'][:, -1, 0] == ep['node_ids'][np.arange(200), 0, q]).all()


def test_bayes_is_calibrated_against_generator():
    rng = np.random.default_rng(4)
    cfg = W.cfg_of(maj=True)
    ep = W.sample(40000, rng, cfg, rng.integers(0, 9, 40000))
    p = W.references(ep)['bayes']
    y = ep['y']
    for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 500:
            assert abs(y[m].mean() - p[m].mean()) < 0.03
    # the oracle beats every reference in log loss
    refs = W.references(ep)
    ll = {k: -np.mean(np.log(np.where(y == 1, v, 1 - v))) for k, v in refs.items()}
    assert ll['bayes'] <= ll['nearest_only'] + 1e-3 <= ll['prior'] + 1e-3


def test_information_matched_across_l():
    rng = np.random.default_rng(5)
    cfg = W.cfg_of()
    info = []
    for l in range(1, 9):
        ep = W.sample(20000, rng, cfg, l)
        p = W.references(ep)['bayes']
        info.append(W.bern_kl(p, np.full_like(p, 0.5)).mean())
    info = np.array(info)
    assert info.min() > 0.6 * info.max(), info


def test_features():
    rng = np.random.default_rng(6)
    cfg = W.cfg_of(maj=True)
    ep = W.sample(50, rng, cfg, 4)
    X, valid = W.features(ep, cfg)
    V = cfg['V']
    assert X.shape[-1] == W.N_FEAT(cfg)
    nid = X[..., :V].sum(-1)
    want = np.select([ep['typ'] == W.UNARY, ep['typ'] == W.XOR, ep['typ'] == W.MAJ, ep['typ'] == W.QUERY],
                     [1, 2, 3, 1])
    np.testing.assert_array_equal(nid[valid], want[valid])
    assert (X[~valid] == 0).all()
    np.testing.assert_array_equal(X[..., V:V + 4].argmax(-1)[valid], ep['typ'][valid])
    np.testing.assert_array_equal(X[..., V + 4:V + 7].argmax(-1)[valid], ep['bit'][valid])
    np.testing.assert_allclose(X[..., -1], ep['rel'])


def test_harness_adapter():
    w = W.make_world({'maj': True})
    rng = np.random.default_rng(7)
    ep = w.sample_train(rng, 64, 0, 4)
    assert ep['X'].shape[-1] == w.feat_dim and ep['mask'][:, -1].all()
    assert set(np.unique(ep['l'])) <= set(range(5))
    for pol in w.test_policies:
        ep = w.sample(np.random.default_rng(8), 32, 7, pol)
        assert (ep['l'] == 7).all()
        pb = w.exact_bayes(ep)
        np.testing.assert_allclose(pb.sum(1), 1)
        m = w.meta(ep)
        assert all(len(v) == 32 for v in m.values())
    # deterministic given rng
    a = w.sample(np.random.default_rng(9), 16, 5, 'std'); b = w.sample(np.random.default_rng(9), 16, 5, 'std')
    np.testing.assert_array_equal(a['X'], b['X'])


def test_curriculum():
    w = W.make_world({'curriculum': 3})
    rng = np.random.default_rng(10)
    assert w.sample_train(rng, 400, 0, 4)['l'].max() == 4          # first call (validation): full range
    seen = [int(w.sample_train(rng, 200, 0, 4)['l'].max()) for _ in range(18)]
    assert seen[:3] == [0, 0, 0] and seen[3:6] == [1, 1, 1] and max(seen) == 4 and seen[-1] == 4


def test_anneal_curriculum():
    w = W.make_world({'anneal': 10, 'flip0': 0.05})
    rng = np.random.default_rng(11)
    v = w.sample_train(rng, 2000, 0, 4)                         # validation: target world (iid latents)
    x = v['chains']['x']
    assert abs((x[..., 1:] != x[..., :-1]).mean() - 0.5) < 0.02
    e = w.sample_train(rng, 2000, 0, 4)                         # step 0: strongly correlated latents
    x = e['chains']['x']
    assert abs((x[..., 1:] != x[..., :-1]).mean() - 0.05) < 0.01
    for _ in range(10):
        e = w.sample_train(rng, 2000, 0, 4)
    x = e['chains']['x']
    assert abs((x[..., 1:] != x[..., :-1]).mean() - 0.5) < 0.02
    # evaluation episodes are always the target world
    x = w.sample(np.random.default_rng(1), 2000, 3)['chains']['x']
    assert abs((x[..., 1:] != x[..., :-1]).mean() - 0.5) < 0.02


def test_dense_queries():
    w = W.make_world()
    rng = np.random.default_rng(12)
    ep = w.sample_train(rng, 300, 0, 4)
    Xq, y, d = W.dense_queries(ep, rng, w.wcfg, 16)
    assert Xq.shape == (300, 16, w.feat_dim) and (d <= 4).all() and set(np.unique(d)) == set(range(5))
    # each extra query token equals the real query token of an episode asking about that node
    V = w.wcfg['V']
    nid = Xq[..., :V].argmax(-1)
    flat_ids = ep['node_ids'].reshape(300, -1); flat_x = ep['chains']['x'].reshape(300, -1)
    for b in range(5):
        for k in range(16):
            j = int(np.where(flat_ids[b] == nid[b, k])[0][0])
            assert flat_x[b, j] == y[b, k]
    np.testing.assert_array_equal(Xq[0, 0, V:], ep['X'][0, -1, V:])


@pytest.mark.parametrize('flip', [0.0, 0.1, 0.3])
def test_markov_prior_world(flip):
    rng = np.random.default_rng(13)
    cfg = W.cfg_of(n=9, flip=flip, maj=True)
    ep = W.sample(8, rng, cfg, 3)
    ch = ep['chains']
    fb = W.chain_marginals(**_one(ch, slice(None)), flip=ch['flip'][:, 0])
    for b in range(8):
        np.testing.assert_allclose(fb[b], W.brute_marginals(**_one(ch, b), flip=flip), atol=1e-10)
    ep = W.sample(30000, rng, W.cfg_of(flip=flip), rng.integers(0, 9, 30000))
    p = W.references(ep)['bayes']; y = ep['y']
    for lo, hi in [(0, .3), (.3, .7), (.7, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum() > 500:
            assert abs(y[m].mean() - p[m].mean()) < 0.03


def test_signed_feature():
    w = W.make_world({'signed': True})
    ep = w.sample(np.random.default_rng(14), 20, 2)
    V = w.wcfg['V']
    assert ep['X'].shape[-1] == V + 9
    s = ep['X'][..., -1]
    v = ep['valid'] & (ep['typ'] != W.QUERY)
    np.testing.assert_allclose(s[v], (2 * ep['bit'][v] - 1) * ep['rel'][v], rtol=1e-6)
    assert (s[:, -1] == 0).all()
    Xq, _, _ = W.dense_queries(ep, np.random.default_rng(0), w.wcfg, 4)
    np.testing.assert_array_equal(Xq[0, 0, V:], ep['X'][0, -1, V:])
