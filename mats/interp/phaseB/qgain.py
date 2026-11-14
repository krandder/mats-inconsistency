"""E9.3 (post hoc, PREREG-B amendment 16 addendum 1): reading gains of the layer-2 QQ values.
g[h, s] = d <v_{QQ_s, h} - a, b> / |b|^2 / d o*_s : how strongly source s's layer-2 value (head h), read along its fitted
reading direction b (qe9 fit, fit episodes), moves with source s's own reading; noise-marginal over K = 8 draws (same
generator seeds as E9.1), autograd through layer 1. Also the query row's layer-2 QQ masses A[h, s] in the same draws.
usage: python qgain.py NET [NET ...] -> runs/NET/qgain.npz (g (n, h, 3), A (n, h, 3))
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import qe9
from qpatch import T, QT, QQ

torch.set_num_threads(1)


def gains(model, X, valid, sg, P, K=8, bs=500):
    G, M = [], []; b = model.blocks[1]; h = b.h
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); n = len(Xi)
        gs = torch.zeros(n, h, 3); ms = torch.zeros(n, h, 3)
        for _ in range(K):
            z = qe9.noise_list(n, model.d, g)
            x1 = qe9.layer1(model, Xi, al, sg, z)
            q, k, v = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1)
            sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2); q, k, v = sp(q), sp(k), sp(v)
            coord = ((v[:, :, QQ, :] - P['a'][None]) * P['b'][None]).sum(-1) / (P['b'][None] ** 2).sum(-1)      # (n, h, 3)
            for hh in range(h):
                for s in range(3):
                    gr, = torch.autograd.grad(coord[:, hh, s].sum(), Xi, retain_graph=True)
                    gs[:, hh, s] += gr[:, QQ[s], qe9.VAL] / K
            with torch.no_grad():
                A = ((q @ k.transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
                ms += A[:, :, QT][:, :, QQ] / K
        G.append(gs.numpy()); M.append(ms.numpy())
    return np.concatenate(G), np.concatenate(M)


if __name__ == '__main__':
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = 4000
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'qgain.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        P, _ = qe9.fit_params(model, sg, nm)
        g, A = gains(model, Tz['X'][:n], Tz['valid'][:n], sg, P)
        np.savez(p, g=g, A=A)
        print(nm, 'done %.0fs' % (time.time() - t0), flush=True)
