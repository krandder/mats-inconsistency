"""World 1 with S sources (seventh agent; a tool for a later registered episode): tiny-pfn's world.py generalised from S = 3 to any S,
with the same draws, token layout and exact Bayes. With S = 3 every function reproduces world.py exactly (checked by check()).
    y_q ~ N(0, 1); o_{s,q} = y_q + e_{s,q}; e_{s,q} ~ N(0, v_s); v_s iid uniform on VLEV; Q = 5 questions, 4 labelled, 1 query.
Exact Bayes: a mixture over the NV^S variance combinations (729 for S = 6), weights from the labelled residuals and the query
readings' marginal, as world.references(). Tokens: one per (source, question), row-major by source, then the query token; features
[source id (DS), question id (DS), o, y (0 if missing), missing flag, query flag] (world.assemble).
"""
import itertools, math
import numpy as np

Q, QL = 5, 4
VLEV = np.array([0.1, 0.5, 2.0])
NV = len(VLEV)
DS = 16


class World:
    def __init__(self, S):
        self.S = S
        self.COMB = np.array(list(itertools.product(range(NV), repeat=S)), dtype=np.int64)
        self.VC = VLEV[self.COMB]
        self.LOGPRIOR = np.full(len(self.COMB), -math.log(len(self.COMB)))
        self.PREC = 1.0 + (1.0 / self.VC).sum(1)
        SIGQ = np.stack([np.diag(v) + 1.0 for v in self.VC])
        self.SIGQI = np.linalg.inv(SIGQ)
        self.LOGDETQ = np.linalg.slogdet(SIGQ)[1]

    def sample_np(self, n, rng):
        S = self.S
        vidx = rng.integers(0, NV, (n, S)); v = VLEV[vidx]
        y = rng.standard_normal((n, Q))
        e = rng.standard_normal((n, S, Q)) * np.sqrt(v)[:, :, None]
        return dict(vidx=vidx, v=v, y=y, o=y[:, None, :] + e)

    def bayes(self, d, chunk=2000):
        """Exact Bayes mixture (means (n, C), vars (C,), weights (n, C)) for y at the query question."""
        S, n = self.S, len(d['y'])
        ostar = d['o'][:, :, QL]
        R = d['o'][:, :, :QL] - d['y'][:, None, :QL]
        ss = (R ** 2).sum(2)
        ll = -0.5 * ss[:, :, None] / VLEV - 0.5 * QL * np.log(2 * np.pi * VLEV)            # (n, S, NV)
        W = np.empty((n, len(self.COMB))); M = np.empty((n, len(self.COMB)))
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            llc = sum(ll[sl, s, self.COMB[:, s]] for s in range(S))
            quadq = np.einsum('ns,pst,nt->np', ostar[sl], self.SIGQI, ostar[sl])
            logq = -0.5 * quadq - 0.5 * self.LOGDETQ[None] - 0.5 * S * math.log(2 * math.pi)
            lw = self.LOGPRIOR[None] + llc + logq; lw = lw - lw.max(1, keepdims=True)
            w = np.exp(lw); W[sl] = w / w.sum(1, keepdims=True)
            M[sl] = np.einsum('ps,ns->np', 1.0 / self.VC, ostar[sl]) / self.PREC[None]
        return M, 1.0 / self.PREC, W

    def tokens_np(self, d, rng, ds=DS):
        n = len(d['y'])
        semb = rng.standard_normal((n, self.S, ds)) / 4
        qemb = rng.standard_normal((n, Q, ds)) / 4
        return self.assemble(semb, qemb, d['o'], d['y'])

    def assemble(self, semb, qemb, o, y):
        n = len(o); S = self.S; ds = semb.shape[-1]
        labd = np.concatenate([y[:, :QL], np.zeros((n, Q - QL))], 1)
        miss = np.concatenate([np.zeros((n, QL)), np.ones((n, Q - QL))], 1)
        F = 2 * ds + 4; T = S * Q + 1
        X = np.zeros((n, T, F), dtype=np.float32)
        for r in range(S):
            for q in range(Q):
                t = r * Q + q
                X[:, t, :ds] = semb[:, r]; X[:, t, ds:2 * ds] = qemb[:, q]
                X[:, t, 2 * ds] = o[:, r, q]; X[:, t, 2 * ds + 1] = labd[:, q]; X[:, t, 2 * ds + 2] = miss[:, q]
        X[:, T - 1, ds:2 * ds] = qemb[:, QL]; X[:, T - 1, 2 * ds + 3] = 1.0
        return X, np.ones((n, T), dtype=bool)


def mix_nll(means, vars_, w, y):
    comp = -0.5 * np.log(2 * np.pi * vars_)[None] - 0.5 * (y[:, None] - means) ** 2 / vars_[None]
    mx = comp.max(1, keepdims=True)
    return -(mx[:, 0] + np.log((w * np.exp(comp - mx)).sum(1)))


def check():
    import sys
    sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
    import world as W0
    w3 = World(3)
    d0 = W0.sample_np(500, np.random.default_rng(5)); d1 = w3.sample_np(500, np.random.default_rng(5))
    assert all(np.array_equal(d0[k], d1[k]) for k in d0)
    out, nll, mom, wb = W0.references(d0); M, V, Wn = w3.bayes(d1)
    assert np.allclose(out['bayes'][0], M) and np.allclose(out['bayes'][1], V) and np.allclose(wb, Wn)
    assert np.allclose(nll['bayes'], mix_nll(M, V, Wn, d1['y'][:, QL]))
    X0, m0 = W0.tokens_np(d0, np.random.default_rng(7)); X1, m1 = w3.tokens_np(d1, np.random.default_rng(7))
    assert np.array_equal(X0, X1) and np.array_equal(m0, m1)
    w6 = World(6); d6 = w6.sample_np(100, np.random.default_rng(1)); M6, V6, W6 = w6.bayes(d6)
    assert W6.shape == (100, 729) and np.allclose(W6.sum(1), 1)
    print('world_n: S = 3 reproduces world.py exactly; S = 6 Bayes runs (729 components)')


if __name__ == '__main__':
    check()
