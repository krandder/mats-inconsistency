"""E8.1 (PREREG-B amendment 15): implied source weights of each net vs Bayes source weights.
w_s(x) = d m(x) / d o*_s (noise-marginal predictive mean, K = 8 draws, autograd w.r.t. source s's query-reading feature);
W_s(x) = d mu_B / d o*_s by central differences of the exact Bayes mean. Fit set: 8,192 fresh episodes; test: first 4,000
test episodes. Saves per net: runs/NET/qweights.npz (fit_w, fit_m, test_w, test_m) and, once, results/qweights_world.npz.
usage: python qweights.py NET [NET ...]
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import devpred as DP

torch.set_num_threads(1)
QQ = [4, 9, 14]; VAL = 32


def bayes_mean(d):
    _, _, mom, _ = W.references(d); return mom['bayes'][0]


def bayes_weights(d, h=1e-3):
    out = []
    for s in range(3):
        dp = {k: v.copy() for k, v in d.items()}; dm = {k: v.copy() for k, v in d.items()}
        dp['o'][:, s, W.QL] += h; dm['o'][:, s, W.QL] -= h
        out.append((bayes_mean(dp) - bayes_mean(dm)) / (2 * h))
    return np.stack(out, 1)


def worlds():
    p = 'results/qweights_world.npz'
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z['fit'].item(), z['test'].item()
    rng = np.random.default_rng(99_700); d = W.sample_np(8192, rng); X, v = W.tokens_np(d, rng, 16)
    fit = dict(d=d, X=X, v=v, mB=bayes_mean(d), WB=bayes_weights(d))
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); n = 4000
    dT = dict(y=Tz['y'][:n].copy(), o=Tz['o'][:n].copy(), v=Tz['v'][:n].copy(), vidx=Tz['vidx'][:n].copy())
    test = dict(d=dT, X=Tz['X'][:n], v=Tz['valid'][:n], mB=bayes_mean(dT), WB=bayes_weights(dT))
    np.savez(p, fit=np.array(fit, dtype=object), test=np.array(test, dtype=object))
    return fit, test


def implied(model, X, valid, sg, K=8, bs=512, seed=0):
    g = torch.Generator().manual_seed(seed); Ws, Ms = [], []
    for i in range(0, len(X), bs):
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); vi = torch.from_numpy(valid[i:i + bs])
        m = 0
        for k in range(K):
            mu, ls, lg = model(Xi, vi, sigma=sg, gen=g); m = m + (torch.softmax(lg, -1) * mu).sum(-1) / K
        gr, = torch.autograd.grad(m.sum(), Xi)
        Ws.append(gr[:, QQ, VAL].detach().numpy().astype(np.float64)); Ms.append(m.detach().numpy().astype(np.float64))
    return np.concatenate(Ws), np.concatenate(Ms)


if __name__ == '__main__':
    fit, test = worlds()
    for nm in sys.argv[1:]:
        t0 = time.time(); p = os.path.join('runs', nm, 'qweights.npz')
        if os.path.exists(p): continue
        model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        fw, fm = implied(model, fit['X'], fit['v'], sg, seed=1); tw, tm = implied(model, test['X'], test['v'], sg, seed=2)
        np.savez(p, fit_w=fw, fit_m=fm, test_w=tw, test_m=tm)
        print(nm, 'done %.0fs' % (time.time() - t0), 'mean |w - W| fit', np.abs(fw - fit['WB']).mean().round(4), flush=True)
