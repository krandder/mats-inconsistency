"""E17 per-net measures (PREREG-B amendment 24): the departure against noise level, layer-2 attention capacity and depth.
Clean nets (tiny-pfn-interp/heads, pfn2.PFN) are measured exactly as in E15 (deterministic). Noisy nets (nnoise.NoisyPFN, any
depth) use noise-averaged effective weights: w_s = d m / d o*_s with m = the mean over K = 8 noise draws of the predictive mean
(generator seed 2 + 1000 x batch, batch 500; K = 1 when sigma = 0), and the test NLL gap of the K = 4 noise-marginalised mixture.
usage: python e17_run.py LIST_JSON I0 I1 -> results/e17/<tag>.npz
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import e15_run as R

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'
QQ = [4, 9, 14]; VI = 32


def eff_w_noisy(model, X, valid, sg, K=8, bs=500):
    K = 1 if sg == 0 else K
    W_, M_ = [], []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); vi = torch.from_numpy(valid[i:i + bs]); m = 0
        for _ in range(K):
            mu, ls, lg = model(Xi, vi, sigma=sg, gen=g); m = m + (torch.softmax(lg, -1) * mu).sum(-1) / K
        gr, = torch.autograd.grad(m.sum(), Xi)
        W_.append(gr[:, QQ, VI].numpy().astype(np.float64)); M_.append(m.detach().numpy().astype(np.float64))
    return np.concatenate(W_), np.concatenate(M_)


def gap_noisy(model, T, sg, K=4):
    K = 1 if sg == 0 else K
    mu, ls, lg = N.predict_mix(model, T['X'], T['valid'], K, 20_001)
    return float((N.mix_nll(mu, ls, lg, T['ystar']) - T['nll_bayes']).mean())


def main():
    rows = json.load(open(sys.argv[1]))[int(sys.argv[2]):int(sys.argv[3])]
    os.makedirs('results/e17', exist_ok=True)
    z = np.load('results/e12_pairs.npz'); tz = np.load(os.path.join(TP, 'runs/testset.npz'))
    T = {k: tz[k] for k in ('X', 'valid', 'ystar', 'nll_bayes')}; Xt, vt = tz['X'][:4000], tz['valid'][:4000]
    for r in rows:
        p = os.path.join('results/e17', f"{r['tag']}.npz")
        if os.path.exists(p):
            continue
        t0 = time.time(); meta = json.load(open(os.path.join(r['run'], 'meta.json'))); out = {}
        if r['kind'] == 'clean':
            model = P2.build(meta); model.load_state_dict(torch.load(os.path.join(r['run'], 'model.pt'), map_location='cpu'))
            model.eval(); model.requires_grad_(False)
            out['gap'] = R.gap(model, T)
            for k in (0, 1):
                out[f'ws{k}'], out[f'mean{k}'] = R.eff_w(model, z[f'X{k}'], z[f'valid{k}'], VI)
            out['test_w'], out['test_m'] = R.eff_w(model, Xt, vt, VI)
        else:
            model, meta = N.load_run(r['run']); model.requires_grad_(False); sg = meta['sigma']
            out['gap'] = gap_noisy(model, T, sg)
            for k in (0, 1):
                out[f'ws{k}'], out[f'mean{k}'] = eff_w_noisy(model, z[f'X{k}'], z[f'valid{k}'], sg)
            out['test_w'], out['test_m'] = eff_w_noisy(model, Xt, vt, sg)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(r['tag'], 'gap %.4f' % out['gap'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
