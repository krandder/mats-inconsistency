"""Tests: grid Bayes against tiny-pfn's exact enumeration (N = 3) and against brute-force Gaussian-mixture
enumeration over (partition, variance config) for coupled worlds."""
import itertools, math, sys
import numpy as np
import sfworld as W

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import world as TW  # noqa: E402  (read-only use)


def brute(d, partitions_with_logprior):
    """Exact predictive mixture by enumeration. partitions_with_logprior: list of (labels per episode-independent
    partition as a list of blocks, log prior). Returns nll (n,), mean (n,)."""
    n, N = d['o'].shape[:2]
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    ostar = d['o'][:, :, W.QL]; ystar = d['y'][:, W.QL]
    comps_lw, comps_m, comps_v = [], [], []
    for blocks, lpP in partitions_with_logprior:
        for cfg in itertools.product(range(W.NV), repeat=N):
            v = W.VLEV[list(cfg)]
            Sig = np.diag(v).astype(float)
            for b in blocks:
                if len(b) > 1:
                    for i in b:
                        for j in b:
                            if i != j:
                                Sig[i, j] = d['rho'] * math.sqrt(v[i] * v[j])
            Si = np.linalg.inv(Sig); ld = np.linalg.slogdet(Sig)[1]
            quad = np.einsum('nsq,st,ntq->n', R, Si, R)
            ll = -0.5 * quad - 0.5 * W.QL * (N * W.LOG2PI + ld)
            SQ = Sig + 1.0
            SQi = np.linalg.inv(SQ); ldq = np.linalg.slogdet(SQ)[1]
            llq = -0.5 * np.einsum('ns,st,nt->n', ostar, SQi, ostar) - 0.5 * (N * W.LOG2PI + ldq)
            lw = lpP - N * math.log(W.NV) + ll + llq
            one = np.ones(N)
            prec = 1 + one @ Si @ one
            m = (ostar @ Si @ one) / prec
            comps_lw.append(lw); comps_m.append(m); comps_v.append(1 / prec)
    lw = np.stack(comps_lw, 1); lw -= lw.max(1, keepdims=True); w = np.exp(lw); w /= w.sum(1, keepdims=True)
    M = np.stack(comps_m, 1); V = np.array(comps_v)[None, :]
    comp = -0.5 * np.log(2 * np.pi * V) - 0.5 * (ystar[:, None] - M) ** 2 / V
    mx = comp.max(1, keepdims=True)
    nll = -(mx[:, 0] + np.log((w * np.exp(comp - mx)).sum(1)))
    return nll, (w * M).sum(1)


def all_single_pair_partitions(N):
    out = []
    def rec(free, blocks):
        if not free:
            out.append(list(blocks)); return
        i = free[0]; rest = free[1:]
        rec(rest, blocks + [[i]])
        for t in rest:
            rec([x for x in rest if x != t], blocks + [[i, t]])
    rec(list(range(N)), [])
    return out


def test_indep_matches_tinypfn():
    rng = np.random.default_rng(0)
    d = W.sample(3000, 3, rng)
    B = W.bayes(d)
    td = dict(vidx=d['vidx'], v=d['v'], y=d['y'], o=d['o'])
    _, nll, mom, _ = TW.references(td)
    assert np.abs(B['nll'] - nll['bayes']).max() < 1e-6, np.abs(B['nll'] - nll['bayes']).max()
    assert np.abs(B['mean'] - mom['bayes'][0]).max() < 1e-6
    assert np.abs(B['var'] - mom['bayes'][1]).max() < 1e-6


def test_groups_gsize1_equals_indep():
    rng = np.random.default_rng(1)
    d = W.sample(500, 6, rng, mode='groups', rho=0.5, gsize=1)
    d2 = dict(d); d2['mode'] = 'indep'
    assert np.abs(W.bayes(d)['nll'] - W.bayes(d2)['nll']).max() < 1e-9


def test_groups_vs_brute():
    for N, gs in ((4, 2), (4, 4), (3, 3)):
        rng = np.random.default_rng(2 + N + gs)
        d = W.sample(400, N, rng, mode='groups', rho=0.6, gsize=gs)
        blocks = [list(range(g * gs, (g + 1) * gs)) for g in range(N // gs)]
        nll_b, mean_b = brute(d, [(blocks, 0.0)])
        B = W.bayes(d)
        assert np.abs(B['nll'] - nll_b).max() < 1e-6, (N, gs, np.abs(B['nll'] - nll_b).max())
        assert np.abs(B['mean'] - mean_b).max() < 1e-6


def test_pairs_vs_brute():
    for N in (3, 4, 5):
        a1, a2 = 1.0, 2.0
        rng = np.random.default_rng(10 + N)
        d = W.sample(200, N, rng, mode='pairs', rho=0.7, a1=a1, a2=a2)
        parts = all_single_pair_partitions(N)
        lp = np.array([sum(math.log(a1) if len(b) == 1 else math.log(a2) for b in P) for P in parts])
        lp -= np.logaddexp.reduce(lp)
        nll_b, mean_b = brute(d, list(zip(parts, lp)))
        B = W.bayes(d)
        assert np.abs(B['nll'] - nll_b).max() < 1e-6, (N, np.abs(B['nll'] - nll_b).max())
        assert np.abs(B['mean'] - mean_b).max() < 1e-6


def test_pairs_sampler_prior():
    """Empirical partition frequencies match the prior (N = 4)."""
    N, a1, a2 = 4, 1.0, 2.0
    rng = np.random.default_rng(5)
    lab = W.sample_pairs_partition(20000, N, rng, a1, a2)
    npairs = np.array([N - len(set(r)) for r in lab])
    parts = all_single_pair_partitions(N)
    w = np.array([a1 ** sum(len(b) == 1 for b in P) * a2 ** sum(len(b) == 2 for b in P) for P in parts]); w /= w.sum()
    for k in range(3):
        pk = sum(wi for wi, P in zip(w, parts) if sum(len(b) == 2 for b in P) == k)
        assert abs((npairs == k).mean() - pk) < 0.015, (k, (npairs == k).mean(), pk)


def test_calibration_large_N():
    """PIT of y* under grid Bayes is uniform at N = 24 (indep) and N = 12 groups."""
    for kw in (dict(N=24, mode='indep'), dict(N=12, mode='groups', rho=0.5, gsize=4)):
        rng = np.random.default_rng(7)
        d = W.sample(3000, rng=rng, **kw)
        B = W.bayes(d)
        ys = d['y'][:, W.QL]
        p = np.exp(B['logp'])
        cdf = np.array([np.trapezoid(p[i] * (B['grid'][i] <= ys[i]), B['grid'][i]) for i in range(len(ys))])
        hist = np.histogram(cdf, bins=10, range=(0, 1))[0] / len(cdf)
        assert np.abs(hist - 0.1).max() < 0.03, (kw, hist)
        # mean squared error of the posterior mean equals the mean posterior variance
        assert abs(((B['mean'] - ys) ** 2).mean() / B['var'].mean() - 1) < 0.1


if __name__ == '__main__':
    for name, f in list(globals().items()):
        if name.startswith('test_'):
            f(); print('ok', name, flush=True)
