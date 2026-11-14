"""Harness invariances: padding does not change predictions; context order does not matter (sum head and sink on)."""
import numpy as np, torch
import harness as H
import junta as J


def test_padding_and_permutation():
    torch.manual_seed(0)
    w = J.Junta(dict(B=8))
    m = H.SetPFN(w.feat_dim, 2, 32, 4, 'real', 3, sink=True, sum_head=8).to(H.DEV).eval()
    ep = w.sample(np.random.default_rng(1), 6, 40, 'uniform', n=np.array([40, 33, 20, 7, 40, 12]))
    p = H.predict(m, ep['X'], ep['mask'])
    for i in range(6):
        n = ep['n'][i]
        X = np.concatenate([ep['X'][i:i + 1, :n], ep['X'][i:i + 1, -1:]], 1); M = np.ones((1, n + 1), bool)
        q = H.predict(m, X, M)
        perm = np.r_[np.random.default_rng(i).permutation(n), n]
        r = H.predict(m, X[:, perm], M)
        for a, b, c in zip(p, q, r):
            assert np.allclose(a[i], b[0], atol=1e-4) and np.allclose(b[0], c[0], atol=1e-4)
