"""E8.4 (post hoc, PREREG-B amendment 15 addendum 4): layer-2 query-row attention masses on the QQ tokens and on each
source's five tokens, per head, averaged over K = 4 noise draws (layer-1 noise included), on the E8.1 fit episodes and the
first 8,000 test episodes. usage: python qattnw.py NET [NET ...] -> runs/NET/qattnw.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import qweights as QWT
import qmech as QM
from qpatch import T, QT, SRC, QQ

torch.set_num_threads(1)


def masses(model, X, valid, sg, K=4, bs=2000, seed=0):
    g = torch.Generator().manual_seed(seed); b = model.blocks[1]; d_ = model.d; QQm, SRm = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Xi = torch.from_numpy(X[i:i + bs]); al = model.allow_mask(torch.from_numpy(valid[i:i + bs])); n = len(Xi)
            qq = 0; sr = 0
            for _ in range(K):
                nz = lambda w: torch.randn((n, T, w), generator=g)
                x = model.f_emb(Xi) + sg * nz(d_)
                xa = model.f_a(1, x, al) + sg * nz(d_); m = model.f_m(1, xa) + sg * nz(2 * d_); x1 = model.f_r(1, xa, m) + sg * nz(d_)
                q, k, v = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1); sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
                A = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)[:, :, QT]
                qq = qq + A[:, :, QQ] / K; sr = sr + torch.stack([A[:, :, S].sum(-1) for S in SRC], -1) / K
            QQm.append(qq.numpy()); SRm.append(sr.numpy())
    return np.concatenate(QQm), np.concatenate(SRm)


if __name__ == '__main__':
    fit, _ = QWT.worlds(); t8 = QM.test8000()
    import devpred as DP
    valid8 = np.load(os.path.join(DP.TP, 'runs/testset.npz'))['valid'][:QM.N8]
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, 'qattnw.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        fq, fs = masses(model, fit['X'], fit['v'], sg, seed=21); tq, ts = masses(model, t8['X'], valid8, sg, seed=22)
        np.savez(p, fit_qq=fq, fit_src=fs, test_qq=tq, test_src=ts)
        print(nm, 'done %.0fs' % (time.time() - t0), 'mean QQ mass per head', fq.sum(2).mean(0).round(2), flush=True)
