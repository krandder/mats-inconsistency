"""E10.4 (post hoc, PREREG-B amendment 17 addendum 4): size of the reading's sensitivity in the QQ token's layer-1 output and
layer-2 value, against source reliability. usage: python qsens.py NET [NET ...] -> runs/NET/qsens.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import qe9
from qpatch import T, QQ

torch.set_num_threads(1)


def sens(model, X, valid, sg, K=8, bs=500):
    from torch.func import jvp
    b = model.blocks[1]; out1, outv = [], []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); n = len(Xi)
        J1 = torch.zeros(n, 3, model.d); JV = torch.zeros(n, b.h, 3, b.dh)
        for _ in range(K):
            z = qe9.noise_list(n, model.d, g)
            def fn(Xa):
                x1 = qe9.layer1(model, Xa, al, sg, z)
                v = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1)[2].view(n, T, b.h, b.dh).transpose(1, 2)
                return x1, v
            for s in range(3):
                tg = torch.zeros_like(Xi); tg[:, QQ[s], qe9.VAL] = 1.0
                _, (tx1, tv) = jvp(fn, (Xi,), (tg,))
                J1[:, s] += tx1[:, QQ[s]].detach() / K; JV[:, :, s] += tv[:, :, QQ[s]].detach() / K
        out1.append(J1.norm(dim=-1).numpy()); outv.append(JV.norm(dim=-1).mean(1).numpy())
    return np.concatenate(out1), np.concatenate(outv)


if __name__ == '__main__':
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = 4000
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'qsens.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False)
        a, v = sens(model, Tz['X'][:n], Tz['valid'][:n], meta['sigma'])
        np.savez(p, x1=a, v=v); print(nm, 'done %.0fs' % (time.time() - t0), flush=True)
