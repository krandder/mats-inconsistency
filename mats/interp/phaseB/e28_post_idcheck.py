import os, sys, json
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
import e26_summ as SM, e28_data as E8
dat = np.load('results/e28_data.npz'); res0 = dat['res'][:, 0]; semb = dat['semb']; qemb = dat['qemb']; y = dat['y']
rbar = res0.mean(-1)                                     # (n, 3)
def fit(Y, X, ntr):
    b, *_ = np.linalg.lstsq(X[:ntr], Y[:ntr], rcond=None)
    pr = X[ntr:] @ b
    return float(1 - ((Y[ntr:] - pr) ** 2).sum() / (Y[ntr:] ** 2).sum())
for g, tags in (('A', SM.A_BASES), ('B', SM.B_BASES)):
    out = []
    for t in tags:
        z = np.load(os.path.join('results/e28/M', t + '.npz')); n = int(z['nb'])
        dm = np.stack([z['m_base'] - z['m_neg%d' % s] for s in range(3)], 1)       # (n, 3)
        w = z['w_base']
        Y = dm.ravel()
        # features per (base, source): 2 rbar * w * [1, semb_s (16)], and 2 * res_q * w * qemb_q summed (16), held-out R^2
        F1 = (2 * rbar[:n, :, None] * w[:, :, None] * np.concatenate([np.ones((n, 3, 1)), semb[:n]], 2)).reshape(-1, 17)
        F2 = (2 * w[:, :, None] * np.einsum('nsq,nqd->nsd', res0[:n], qemb[:n, :4])).reshape(-1, 16)
        F3 = (2 * rbar[:n, :, None] * w[:, :, None] * semb[:n]).reshape(-1, 16) * 0 + (2 * w[:, :, None] * np.einsum('nsq,nsd->nsd', res0[:n], semb[:n])).reshape(-1, 16)
        ntr = len(Y) // 2
        out.append((t, round(fit(Y, F1, ntr), 3), round(fit(Y, F2, ntr), 3), round(fit(Y, np.concatenate([F1, F2], 1), ntr), 3)))
    print(g, out)
