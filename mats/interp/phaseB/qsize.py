"""E10.5 (post hoc, PREREG-B amendment 17 addendum 5): does the size of the reading's encoding in the QQ tokens' layer-1
output carry the sharpening? Small-dose, accuracy-preserving patch, applied only where layer 2 computes the values:
  x1'_QQs = x1ref_QQs + lambda_s (x1_QQs - x1ref_QQs),  lambda_s = (gbar / g_s)^a,
x1ref = layer-1 output with the three query readings set to the Bayes mean (same noise draw), g_s = |d x1_QQs / d o*_s|
(E10.4, K = 8), gbar = the episode's geometric mean over the three sources. a = 0.1, 0.25 equalises a fraction a of the
log-gain differences between sources. Control: the same scalings with the sources' gains permuted at random within each
episode (seed 5). Effective weights (K = 8, E9 draws) and K = 64 deviations (E8 draws), first 4,000 test episodes.
usage: python qsize.py NET [NET ...] -> runs/NET/qsize.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import qw
import qweights as QWT
import qe9
from qpatch import T, QT, QQ

torch.set_num_threads(1)
N4 = 4000
VARS = {'base': None, 'a10': (0.1, False), 'a25': (0.25, False), 'p10': (0.1, True), 'p25': (0.25, True)}


def lambdas(g, a, perm, rng_seed=5):
    gg = g.copy()
    if perm:
        rng = np.random.default_rng(rng_seed); idx = np.argsort(rng.random(gg.shape), 1); gg = np.take_along_axis(gg, idx, 1)
    gbar = np.exp(np.log(gg).mean(1, keepdims=True)); return (gbar / gg) ** a


def fwd(model, X, Xref, al, sg, z, lam=None):
    n = len(X); b = model.blocks[1]; x1 = qe9.layer1(model, X, al, sg, z); xv = x1
    if lam is not None:
        x1r = qe9.layer1(model, Xref, al, sg, z)
        L = torch.ones(n, T, 1); L[:, QQ, 0] = lam
        xv = x1r + L * (x1 - x1r)
    sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
    q, k, _ = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1); v = b.qkv(b.ln1(xv)).split(b.h * b.dh, -1)[2]
    A = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
    att = b.o((A @ sp(v)).transpose(1, 2).reshape(n, T, b.h * b.dh))
    xa2 = N.sphere(x1 + att) + sg * z[4]; u = xa2 + b.mlp[2](model.f_m(2, xa2) + sg * z[5]); r2 = N.sphere(u) + sg * z[6]
    return model.head(r2)


def run(nm):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); X, valid = Tz['X'][:N4], Tz['valid'][:N4]
    _, test = QWT.worlds(); Xref = X.copy(); Xref[:, QQ, qe9.VAL] = test['mB'][:, None].astype(np.float32)
    g = np.load(os.path.join('runs', nm, 'qsens.npz'))['x1']
    LAM = {k: (None if v is None else torch.as_tensor(lambdas(g, *v), dtype=torch.float32)) for k, v in VARS.items()}
    out = {}
    for k, lam in LAM.items():
        Ws = []
        for bi, i in enumerate(range(0, N4, 500)):
            gen = torch.Generator().manual_seed(2 + 1000 * bi)
            Xi = torch.from_numpy(X[i:i + 500]).clone().requires_grad_(True); Xr = torch.from_numpy(Xref[i:i + 500]); al = model.allow_mask(torch.from_numpy(valid[i:i + 500])); m = 0
            for _ in range(8):
                mu, ls, lg = fwd(model, Xi, Xr, al, sg, qe9.noise_list(len(Xi), model.d, gen), None if lam is None else lam[i:i + 500])
                m = m + (torch.softmax(lg, -1) * mu).sum(-1) / 8
            gr, = torch.autograd.grad(m.sum(), Xi); Ws.append(gr[:, QQ, qe9.VAL].numpy().astype(np.float64))
        out[f'w_{k}'] = np.concatenate(Ws)
    Tt = {k: Tz[k][:N4] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt, Xrt = torch.from_numpy(X), torch.from_numpy(Xref); al = model.allow_mask(torch.from_numpy(valid))
    heads = {k: ([], [], []) for k in VARS}; gen = torch.Generator().manual_seed(1); K = 64
    with torch.no_grad():
        for _ in range(K):
            z = qe9.noise_list(N4, model.d, gen)
            for k, lam in LAM.items():
                mu, ls, lg = fwd(model, Xt, Xrt, al, sg, z, lam)
                heads[k][0].append(mu); heads[k][1].append(ls); heads[k][2].append(torch.log_softmax(lg, -1) - math.log(K))
    for k in VARS:
        dv = qw.deviations_t(tuple(torch.cat(t, 1).numpy() for t in heads[k]), Tt)
        for f_ in ('mean', 'logsd', 'kl'): out[f'{k}_{f_}'] = dv[f_]
    np.savez(os.path.join('runs', nm, 'qsize.npz'), **out)
    print(nm, 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        if not os.path.exists(os.path.join('runs', nm, 'qsize.npz')): run(nm)
