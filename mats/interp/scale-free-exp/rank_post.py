"""Bayes with rank-r additive messages: KL(exact Bayes || posterior built from rank-r projections of each source's
log-likelihood), per N. This is the constructive message size a sum-pooling tree needs (indep world).
Basis: top-r right singular vectors of the prior-weighted, row-centred single-source log-likelihood family on
y in [-6, 6]. Output audit/rank_post.json."""
import json, math
import numpy as np
import sfworld as W

rng = np.random.default_rng(31)
YG = np.linspace(-6, 6, 1201)
wy = np.exp(-0.5 * YG ** 2); wy /= wy.sum()


def basis(n=20000):
    d = W.sample(n, 1, rng)
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    L = W._single_ll(R, d['o'][:, :, W.QL], np.tile(YG, (n, 1)))[:, 0]
    Lc = L - (L * wy).sum(1, keepdims=True)
    _, s, Vt = np.linalg.svd(Lc * np.sqrt(wy)[None] / math.sqrt(n), full_matrices=False)
    Bf = Vt / np.sqrt(wy)[None]                       # basis functions on YG (orthonormal under wy)
    return Bf, s


Bf, sv = basis()
out = dict(sv=sv[:30].tolist(), by_N={})
for N in (3, 6, 12, 24):
    d = W.sample(1500, N, rng)
    B = W.bayes(d)
    R = d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]
    n = len(d['y'])
    Lw = W._single_ll(R, d['o'][:, :, W.QL], np.tile(YG, (n, 1)))           # (n, N, 1201) on the window
    Lc = Lw - (Lw * wy).sum(-1, keepdims=True)
    res = {}
    for r in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20):
        Br = Bf[:r]                                                          # (r, G)
        coef = np.einsum('nsg,rg,g->nsr', Lc, Br, wy)                        # projections (orthonormal basis)
        tot = coef.sum(1)                                                    # (n, r): the summed message
        approx_w = -0.5 * YG ** 2 + tot @ Br                                 # log posterior (unnorm.) on YG
        # evaluate on each episode's exact grid by interpolation
        la = np.stack([np.interp(B['grid'][i], YG, approx_w[i]) for i in range(n)])
        # normalise the approximation on the window grid (where it may put mass the exact grid misses)
        mx = approx_w.max(1, keepdims=True)
        logZ = mx[:, 0] + np.log(np.trapezoid(np.exp(approx_w - mx), YG, axis=1))
        lq = la - logZ[:, None]
        p = np.exp(B['logp'])
        kl = np.trapezoid(p * (B['logp'] - lq), B['grid'], axis=1)
        res[r] = dict(kl=float(kl.mean()), kl_se=float(kl.std() / math.sqrt(n)), kl_q90=float(np.quantile(kl, 0.9)))
        print(N, r, res[r], flush=True)
    out['by_N'][N] = res
json.dump(out, open('audit/rank_post.json', 'w'), indent=1)
