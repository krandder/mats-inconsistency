"""E13 per-net runs (PREREG-B amendment 20): encoding strength of the modified source in both versions of each E12 pair
(E13.1); patched big-miss weights with lambda = g(spread)/g(big) on the modified source or, control, on another source;
K = 64 accuracy check (E13.2). usage: python e13_run.py NET [NET ...] -> runs/NET/e13.npz
"""
import math, os, sys, time
import numpy as np
import torch
from torch.func import jvp
import nnoise as N
import qe9
import qsize
from qpatch import T, QQ

torch.set_num_threads(1)
VAL = qe9.VAL


def enc_strength(model, X, valid, src, sg, K=8, bs=500):
    out = []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); n = len(Xi); s = src[i:i + bs]
        tg = torch.zeros_like(Xi); tg[torch.arange(n), torch.as_tensor(QQ)[s], VAL] = 1.0
        J = torch.zeros(n, model.d)
        for _ in range(K):
            z = qe9.noise_list(n, model.d, g)
            _, t1 = jvp(lambda Xa: qe9.layer1(model, Xa, al, sg, z), (Xi,), (tg,))
            J += t1[torch.arange(n), torch.as_tensor(QQ)[s]].detach() / K
        out.append(J.norm(dim=-1).numpy())
    return np.concatenate(out)


def eff_w_patched(model, X, Xref, valid, sg, lam, K=8, bs=500):
    Ws = []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); Xr = torch.from_numpy(Xref[i:i + bs]); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); m = 0
        for _ in range(K):
            mu, ls, lg = qsize.fwd(model, Xi, Xr, al, sg, qe9.noise_list(len(Xi), model.d, g), lam[i:i + bs])
            m = m + (torch.softmax(lg, -1) * mu).sum(-1) / K
        gr, = torch.autograd.grad(m.sum(), Xi); Ws.append(gr[:, QQ, VAL].numpy().astype(np.float64))
    return np.concatenate(Ws)


def run(nm):
    t0 = time.time(); p = os.path.join('runs', nm, 'e13.npz')
    if os.path.exists(p): return
    model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    z = np.load('results/e12_pairs.npz'); src = z['src']; m = len(src); rows = np.arange(m)
    X0, X1, v0, v1 = z['X0'], z['X1'], z['valid0'], z['valid1']
    g0 = enc_strength(model, X0, v0, src, sg); g1 = enc_strength(model, X1, v1, src, sg)
    lam_val = g0 / np.maximum(g1, 1e-6)
    Xref = X1.copy(); Xref[:, QQ, VAL] = z['mB'][:, None].astype(np.float32)
    L = np.ones((m, 3), np.float32); L[rows, src] = lam_val
    other = (src + 1 + np.random.default_rng(13).integers(0, 2, m)) % 3
    Lc = np.ones((m, 3), np.float32); Lc[rows, other] = lam_val
    w_p = eff_w_patched(model, X1, Xref, v1, sg, torch.as_tensor(L))[rows, src]
    w_c = eff_w_patched(model, X1, Xref, v1, sg, torch.as_tensor(Lc))[rows, src]
    # accuracy: K = 64 mean deviations of big-miss episodes, unpatched vs patched, same draws
    import qw
    Tt = {k[len('T1_'):]: z[k] for k in z.files if k.startswith('T1_')}
    Xt, Xrt = torch.from_numpy(X1), torch.from_numpy(Xref); al = model.allow_mask(torch.from_numpy(v1)); gen = torch.Generator().manual_seed(1); K = 64
    heads = {k: ([], [], []) for k in ('base', 'patch')}
    with torch.no_grad():
        for _ in range(K):
            zz = qe9.noise_list(m, model.d, gen)
            for k, lam in (('base', None), ('patch', torch.as_tensor(L))):
                mu, ls, lg = qsize.fwd(model, Xt, Xrt, al, sg, zz, lam)
                heads[k][0].append(mu); heads[k][1].append(ls); heads[k][2].append(torch.log_softmax(lg, -1) - math.log(K))
    dev = {k: qw.deviations_t(tuple(torch.cat(t_, 1).numpy() for t_ in heads[k]), Tt)['mean'] for k in heads}
    np.savez(p, g0=g0, g1=g1, lam=lam_val, w_patch=w_p, w_ctrl=w_c, other=other, dev_base=dev['base'], dev_patch=dev['patch'])
    print(nm, 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]: run(nm)
