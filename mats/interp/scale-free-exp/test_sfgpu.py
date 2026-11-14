"""The torch sampler draws the same worlds as the numpy sampler: compare exact-Bayes NLL means (and the coupling-blind
gap), pair counts, and check the token builder against the numpy one on identical episodes."""
import math
import numpy as np
import torch
import sfworld as W
import sfgpu as SG
import sftrain as T


def to_np(d):
    return {k: (v.cpu().double().numpy() if torch.is_tensor(v) else v) for k, v in d.items()}


def check(mode, N, **kw):
    g = torch.Generator().manual_seed(1)
    dt = to_np(SG.sample_t(6000, N, g, 'cpu', mode=mode, **kw))
    dt['blk'] = dt['blk'].astype(np.int64); dt['grp'] = dt['grp'].astype(np.int64)
    dt.update(a1=kw.get('a1', 1.0), a2=kw.get('a2', 1.0))
    dn = W.sample(6000, N, np.random.default_rng(2), mode=mode, **kw)
    bt, bn = W.bayes(dt)['nll'], W.bayes(dn)['nll']
    z = (bt.mean() - bn.mean()) / math.sqrt(bt.var() / len(bt) + bn.var() / len(bn))
    print(mode, N, kw, 'torch', round(bt.mean(), 4), 'numpy', round(bn.mean(), 4), 'z', round(z, 2))
    assert abs(z) < 3.5
    if mode != 'indep':
        it, inn = W.indep_bayes(dt)['nll'] - bt, W.indep_bayes(dn)['nll'] - bn
        z2 = (it.mean() - inn.mean()) / math.sqrt(it.var() / len(it) + inn.var() / len(inn))
        print('   coupling gap torch', round(it.mean(), 4), 'numpy', round(inn.mean(), 4), 'z', round(z2, 2))
        assert abs(z2) < 3.5


def test_worlds():
    check('indep', 6)
    check('groups', 8, rho=0.5, gsize=4)
    check('pairs', 6, rho=0.9, a1=1.0, a2=1.0)


def test_pair_counts():
    g = torch.Generator().manual_seed(3)
    lab = SG.pairs_partition_t(40000, 4, 1.0, 2.0, g, 'cpu').numpy()
    ln = W.sample_pairs_partition(40000, 4, np.random.default_rng(4), 1.0, 2.0)
    for k in range(3):
        pt = np.mean([4 - len(set(r)) == k for r in lab]); pn = np.mean([4 - len(set(r)) == k for r in ln])
        assert abs(pt - pn) < 0.012, (k, pt, pn)
    assert (lab >= 0).all()


def test_tokens_match():
    """Same episode and same id draws: torch and numpy token builders agree."""
    rng = np.random.default_rng(0)
    d = W.sample(7, 4, rng, mode='groups', rho=0.5, gsize=2)
    a = T.build_inputs(d, 'tok', np.random.default_rng(9), 'cpu', want_group=True)
    dt = {k: (torch.from_numpy(v).float() if isinstance(v, np.ndarray) and v.dtype.kind == 'f' else
              (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)) for k, v in d.items()}
    b = SG.build_inputs_t(dt, 'tok', torch.Generator().manual_seed(0), 'cpu', want_group=True)
    # ids differ (different generators): compare every non-id channel and the index tensors
    DS = SG.DS
    assert torch.allclose(a['tok'][..., 2 * DS:2 * DS + 4], b['tok'][..., 2 * DS:2 * DS + 4], atol=1e-6)
    for k in ('src', 'qq', 'grp'):
        assert torch.equal(a[k].long(), b[k].long()), k
    # id channels: same structure (equal within a source / question / group)
    for x in (a['tok'], b['tok']):
        s = x[0, :-1, :DS].reshape(4, 5, DS)
        assert torch.allclose(s[:, 0:1].expand_as(s), s)


if __name__ == '__main__':
    test_pair_counts(); print('ok pair counts')
    test_tokens_match(); print('ok tokens')
    test_worlds(); print('ok worlds')
