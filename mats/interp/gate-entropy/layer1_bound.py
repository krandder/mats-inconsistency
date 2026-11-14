"""Rigorous lower bound on the layer-1 gate entropy of the random MLPs (gate-entropy, 2026-10-29).

Layer-1 pre-activations are exactly Gaussian: z = W_1 x + b, x ~ N(0, I). By the chain rule and because g_{<i} is a
function of z_{<i}:  H(G) = sum_i H(g_i | g_{<i}) >= sum_i H(g_i | z_{<i}),  and z_i | z_{<i} ~ N(m_i, L_ii^2) with L the
Cholesky factor of Cov(z) (m_i = b_i + sum_{j<i} L_ij u_j). So the bound is sum_i E_u[h(state_i | m_i, L_ii)], estimated by
Monte Carlo over u (standard error reported). ReLU: binary state 1[z > 0]. tanh: ternary state (z > 1, |z| <= 1, z < -1).
usage: python layer1_bound.py OUT.json
"""
import sys, os, json, math
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext'))
import anet_copy as A
import rnet_copy as R
import torch


def ent_bits(ps):
    ps = np.clip(ps, 1e-300, 1)
    return -(ps * np.log2(ps)).sum(0)


def bound(W, b, kind, n_mc=1 << 13, seed=0, chunk=512):
    C = W @ W.T
    L = np.linalg.cholesky(C + 1e-12 * np.eye(len(C)))
    d = np.diag(L).copy(); Ls = np.tril(L, -1)
    rng = np.random.default_rng(seed)
    tot = []
    for i in range(0, n_mc, chunk):
        u = rng.standard_normal((len(C), min(chunk, n_mc - i)))
        m = b[:, None] + Ls @ u                                   # conditional means (m, k)
        if kind == 'relu':
            p = norm.cdf(m / d[:, None]); h = ent_bits(np.stack([p, 1 - p]))
        else:
            pu = norm.sf((1 - m) / d[:, None]); pd = norm.cdf((-1 - m) / d[:, None])
            h = ent_bits(np.stack([pu, pd, np.clip(1 - pu - pd, 0, 1)]))
        tot.append(h.sum(0))                                       # per-sample sum over gates
    t = np.concatenate(tot)
    return float(t.mean()), float(t.std() / math.sqrt(len(t))), float(np.log2(np.e) * 0 + d.min()), float(d.max())


out = {}
for n in (32, 64, 128, 256):
    W = A.make_weights(0, n, 1)[0].numpy()
    hb, se, dmin, dmax = bound(W, np.zeros(n), 'relu')
    out[f'arc_n{n}'] = dict(m=n, lower=hb, se=se, frac_of_m=hb / n, chol_diag_min=dmin, chol_diag_max=dmax)
    print('arc', n, out[f'arc_n{n}'], flush=True)
for g in (3.0, 0.25):
    for n in (32, 64, 128, 256):
        net = R.RNet(n, 32, 'tanh', g, 0.01, 0, bias=0.5)          # L = 32: b_1 is drawn after all W_l
        W = net.W[0].numpy(); b = net.b[0].numpy()
        hb, se, dmin, dmax = bound(W, b, 'tanh')
        out[f'tanh_g{g:g}_n{n}'] = dict(m_neurons=n, lower=hb, se=se, frac_of_log2_3_m=hb / (n * math.log2(3)), chol_diag_min=dmin)
        print('tanh', g, n, out[f'tanh_g{g:g}_n{n}'], flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
