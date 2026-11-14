"""E15 per-checkpoint measures (PREREG-B amendment 22) for clean nets (tiny-pfn/pfn.py PFN; no noise, no sphere).
  gap  test NLL minus exact Bayes NLL on the 16,000 test episodes;
  M1   effective weights w_s = d m / d o*_s of all three sources (m = predictive mean; autograd; deterministic) in both
       versions of every E12 pair (ds-16: results/e12_pairs.npz; ds-4: results/e15_pairs_ds4.npz);
  M2   the same on the first 4,000 test episodes (for the slope on the Bayes weights);
  M3   layer-1 encoding strength g = |d h1[QQ_src] / d o*_src| (h1 = residual after block 1; forward mode), both versions.
usage: python e15_run.py WHICH I0 I1   (WHICH = best | phase1; rows I0..I1-1 of results/e15_nets.json)
       -> results/e15/<tag>__<WHICH>.npz
"""
import json, os, sys, time
import numpy as np
import torch
from torch.func import jvp
import nnoise as N
import pfn as P

torch.set_num_threads(1)
TP = '/home/agent/workspaces/mats/work/tiny-pfn'
QQ = [4, 9, 14]


def pred_mean(head):
    if len(head) == 2:
        return head[0]
    mu, ls, lg = head
    return (torch.softmax(lg, -1) * mu).sum(-1)


def eff_w(model, X, valid, vi, bs=1000):
    W_, M_ = [], []
    for i in range(0, len(X), bs):
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True)
        m = pred_mean(model(Xi, torch.from_numpy(valid[i:i + bs])))
        gr, = torch.autograd.grad(m.sum(), Xi)
        W_.append(gr[:, QQ, vi].numpy().astype(np.float64)); M_.append(m.detach().numpy().astype(np.float64))
    return np.concatenate(W_), np.concatenate(M_)


def enc_g(model, X, valid, src, vi, bs=1000):
    out = []; qq = torch.as_tensor(QQ)
    for i in range(0, len(X), bs):
        Xi = torch.from_numpy(X[i:i + bs]); n = len(Xi); s = torch.as_tensor(src[i:i + bs])
        al = N.NoisyPFN.allow_mask(torch.from_numpy(valid[i:i + bs]))
        tg = torch.zeros_like(Xi); tg[torch.arange(n), qq[s], vi] = 1.0
        _, t1 = jvp(lambda Xa: model.blocks[0](model.inp(Xa), al)[0], (Xi,), (tg,))
        out.append(t1[torch.arange(n), qq[s]].norm(dim=-1).detach().numpy().astype(np.float64))
    return np.concatenate(out)


def gap(model, T):
    head = P.predict(model, T['X'], T['valid'])
    return float((P.head_nll_np(head, T['ystar']) - T['nll_bayes']).mean())


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = json.load(open('results/e15_nets.json'))[i0:i1]
    os.makedirs('results/e15', exist_ok=True)
    data = {16: (np.load('results/e12_pairs.npz'), np.load(os.path.join(TP, 'runs/testset.npz'))),
            4: (np.load('results/e15_pairs_ds4.npz'), np.load(os.path.join(TP, 'runs/testset_ds4.npz')))}
    src = np.load('results/e12_pairs.npz')['src']
    for r in rows:
        p = os.path.join('results/e15', f"{r['tag']}__{which}.npz")
        if os.path.exists(p):
            continue
        if which == 'phase1':                       # phase-1 checkpoints of acquired nets only (depth 2, best gap < 0.10)
            pb = os.path.join('results/e15', f"{r['tag']}__best.npz")
            if r['depth'] != 2 or float(np.load(pb)['gap']) >= 0.10:
                continue
        ck =os.path.join(r['run'], 'model.pt' if which == 'best' else 'model_phase1.pt')
        t0 = time.time(); meta = json.load(open(os.path.join(r['run'], 'meta.json')))
        model = P.build(meta); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval(); model.requires_grad_(False)
        pz, tz = data[r['ds']]; vi = 2 * r['ds']
        T = {k: tz[k] for k in ('X', 'valid', 'ystar', 'nll_bayes')}
        out = dict(gap=gap(model, T))
        for k in (0, 1):
            out[f'ws{k}'], out[f'mean{k}'] = eff_w(model, pz[f'X{k}'], pz[f'valid{k}'], vi)
            out[f'g{k}'] = enc_g(model, pz[f'X{k}'], pz[f'valid{k}'], src, vi)
        out['test_w'], out['test_m'] = eff_w(model, tz['X'][:4000], tz['valid'][:4000], vi)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(r['tag'], which, 'gap %.4f' % out['gap'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
