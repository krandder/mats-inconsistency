"""Part 3: deep-part estimators applied to the exact (oracle) law at layer l0 (PREREG P6), and K2 from the input.
usage: python p3_att.py TRUTHPREFIX [TRUTHPREFIX ...]  -> prints and writes TRUTHPREFIX_att.json
ATT-lin:   push the mean state E[h_l0] through the noiseless deep layers (= attractor-linear map J* applied to the
           mean when every state shares the reference gates).
ATT-K2b:   16 log-radius bins (+ zero bin); per bin N(mean, cov) pushed by K2 (exact ReLU moments) through layers
           l0+1..L with the noise; weighted sum. ATT-K2b-st: same with the Stein-linearised K2.
ATT-K2-1:  one Gaussian with the mean and covariance of h_l0 (no binning).
K2-input:  K2 (exact ReLU moments) from x ~ N(0, I) through all layers (ARC-style Gaussian closure).
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, k2

torch.set_num_threads(1)
DT = nd.DT


def nmse(est, truth, trvar):
    return float(((est - truth) ** 2).sum() / trvar)


for pref in sys.argv[1:]:
    t0 = time.time()
    S = json.load(open(pref + '.json'))
    O = np.load(pref + '_oracle.npz')
    net = nd.Net(S['fam'], S['seed'], S['n'], S['L'], S['sigma'], r=S['r'] if S['sigma'] > 0 else None)
    truth = torch.tensor(S['mean'], dtype=DT); trv = S['trvar']; n = net.n; L = net.L
    res = dict(trvar=trv, mean_sq=float(truth.pow(2).sum()), flops_path=net.flops_path(), l0={})
    if trv <= 0 or not np.isfinite(trv):
        res['dead'] = True
    else:
        # K2 from the input (exact) and Stein
        for ex in (True, False):
            mu, C, fl = k2.prop(net, torch.zeros(n, dtype=DT), torch.eye(n, dtype=DT), 0, L, exact=ex)
            res['k2_input' + ('' if ex else '_st')] = dict(nmse=nmse(mu, truth, trv), flops=fl)
        for l0 in S['L0S']:
            w = torch.tensor(O[f'w_{l0}'], dtype=DT); s1 = torch.tensor(O[f's_{l0}'], dtype=DT); s2 = torch.tensor(O[f'ss_{l0}'], dtype=DT)
            m = torch.tensor(O[f'm_{l0}'], dtype=DT)
            row = {}
            if net.fam == 'plain':
                h = m.clone()
                for l in range(l0, L):
                    h = torch.relu(net.W[l] @ h)
                row['att_lin'] = dict(nmse=nmse(h, truth, trv), flops=(L - l0) * (k2.mmf(n, n, 1) + n))
            for ex, tag in ((True, 'att_k2b'), (False, 'att_k2b_st')):
                est = torch.zeros(n, dtype=DT); fl = 0
                for b in range(w.shape[0]):
                    if w[b] <= 0: continue
                    mu = s1[b] / w[b]; C = s2[b] / w[b] - mu[:, None] * mu[None, :]
                    C = 0.5 * (C + C.T)
                    mL, _, f = k2.prop(net, mu, C, l0, L, exact=ex)
                    est += w[b] * mL; fl += f
                row[tag] = dict(nmse=nmse(est, truth, trv), flops=fl)
            W = w.sum(); mu = s1.sum(0) / W; C = s2.sum(0) / W - mu[:, None] * mu[None, :]
            mL, _, f = k2.prop(net, mu, 0.5 * (C + C.T), l0, L, exact=True)
            row['att_k2_1'] = dict(nmse=nmse(mL, truth, trv), flops=f)
            res['l0'][l0] = row
    res['sec'] = time.time() - t0
    json.dump(res, open(pref + '_att.json', 'w'))
    nm = os.path.basename(pref)
    if res.get('dead'):
        print(nm, 'DEAD'); continue
    print(f"{nm}: K2-input nMSE={res['k2_input']['nmse']:.3g} (flops {res['k2_input']['flops']:.3g}; MC path {res['flops_path']}) "
          f"K2-input-stein={res['k2_input_st']['nmse']:.3g}", flush=True)
    for l0, row in res['l0'].items():
        print(f"   l0={l0:>3}: " + ' '.join(f"{k}={v['nmse']:.3g}" for k, v in row.items()), flush=True)
