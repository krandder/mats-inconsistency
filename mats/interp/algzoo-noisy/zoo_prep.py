"""Amendment 9: ARC's noiseless zoo models (16x10 and 32x10, seeds 0-4) as controls for the width sweep.
Writes runs/sweep/Z_w{d}_s{seed}_sig0.0.npz (r = 1, sigma = 0) and a .json with the same criterion fields as
train_batched.py (paired, noiseless, 2^21 sequences, against ARC's example and zoo seed 2)."""
import json, math
import numpy as np
import torch
import az
torch.set_num_threads(1)
refs = {'16:10:ex': az.load_arc(16, 10, example=True), '16:10:2': az.load_arc(16, 10, seed=2)}
g = np.random.default_rng(5150); X = g.standard_normal((2 ** 21, 10)); Y = np.argsort(X, -1)[:, -2]


def ce_acc(W):
    lo = az.np_forward(W, X)
    lo = lo - lo.max(1, keepdims=True); lp = lo - np.log(np.exp(lo).sum(1, keepdims=True))
    return -lp[np.arange(len(Y)), Y], (lo.argmax(1) == Y).astype(float)


R = {k: ce_acc(W) for k, W in refs.items()}
for d in (16, 32):
    for s in range(5):
        W = az.load_arc(d, 10, seed=s); cv, av = ce_acc(W); ev = {}
        for k, (co, ao) in R.items():
            dl = cv - co; da = av - ao; n_ = len(Y)
            ev[k] = dict(n=n_, L_var=float(cv.mean()), L_orig=float(co.mean()), dL=float(dl.mean()), dL_se=float(dl.std() / math.sqrt(n_)),
                         rel_gap=float(cv.mean() / co.mean() - 1), A_var=float(av.mean()), A_orig=float(ao.mean()), dA=float(da.mean()),
                         dA_se=float(da.std() / math.sqrt(n_)), meets_loss=bool(cv.mean() <= 1.02 * co.mean()),
                         meets_acc=bool(av.mean() >= ao.mean() - 2 * da.std() / math.sqrt(n_)))
            ev[k]['meets'] = ev[k]['meets_loss'] and ev[k]['meets_acc']
        base = f'runs/sweep/Z_w{d}_s{s}_sig0.0'
        np.savez(base + '.npz', W_hi=W[0], W_hh=W[1], W_oh=W[2], r=np.ones((10, d)), sigma=0.0)
        json.dump(dict(mode='arc_zoo', d=d, n=10, seed=s, sigma=0.0, length=2 ** 30, n_steps=2 ** 20, eval_refs=ev, eval=ev['16:10:ex']), open(base + '.json', 'w'), indent=1)
        print(d, s, 'L %.4f A %.4f meets %s' % (ev['16:10:ex']['L_var'], ev['16:10:ex']['A_var'], ev['16:10:ex']['meets']), flush=True)
