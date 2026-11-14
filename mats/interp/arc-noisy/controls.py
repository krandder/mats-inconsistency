"""Noiseless code checks (no claims): ARC Fig. 2 trend at width 256, L = 2, 4, 8 (advantage over Monte Carlo at equal
FLOPs) and K=2 against Monte Carlo at 256 x 32, on nets 200-204 with sigma = 0.
Advantage = nMSE_MC(equal FLOPs) / nMSE_kprop, MC cost per path = 2 n^2 L (ARC's counting of the L matmuls; ARC's own
figure uses L + 1 linear layers). kprop FLOPs from ARC's counter (run_kprop.py at L = 8 and 32; L = 2, 4 from short runs
here). ARC's text: about 1,000x at 2 hidden layers, 10-100x at 4 (factorized K >= 3)."""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anet, truthlib, kprop_noisy as KN

n = 256
NETS = [200, 201, 202, 203, 204]


def load_truth(seed, Lnet):
    p = os.path.join(HERE, 'truth', f'truth_256_{Lnet}_s0_seed{seed}.npz')
    if not os.path.exists(p):
        return None
    d = np.load(p); N = float(d['N']); m = d['S1'] / N; v = d['S2'] / N - m * m
    return dict(N=N, mean=m, var=v)


flops_short = {}
for L in (2, 4):
    net = anet.Net(200, sigma=0.0)
    for v, (k, fac, kind) in dict(k1=(1, False, 'simple'), k2=(2, False, 'simple'), k2a=(2, False, 'augment'),
                                  k3f=(3, True, 'simple'), k3fa=(3, True, 'augment')).items():
        _, fl, _ = KN.kprop_noisy(net, L, k, factor=fac, kind=kind)
        flops_short[(L, v)] = fl
print('| L | variant | kprop FLOPs (ARC) | median nMSE (5 nets) | advantage over MC at equal FLOPs (median) |')
print('|---|---|---|---|---|')
for L in (2, 4, 8):
    for v in ('k1', 'k2', 'k2a', 'k3f', 'k3fa'):
        e = []; adv = []; fl = None
        for s in NETS:
            T = load_truth(s, 8)
            p = os.path.join(HERE, 'results', f'kprop_{s}_s0_L8_{v}.json')
            if T is None or not os.path.exists(p):
                continue
            d = json.load(open(p))
            fl = d['flops'] if L == 8 else flops_short[(L, v)]
            mu = np.array(d['means'][L - 1])
            nm = ((mu - T['mean'][L - 1]) ** 2).mean() / T['var'][L - 1].mean() - 1 / T['N']
            e.append(nm); adv.append((2 * n * n * L / fl) / max(nm, 1e-12))
        if e:
            print(f'| {L} | {v} | {fl:.2e} | {np.median(e):.2e} | {np.median(adv):.1f} |')
print()
e = []; adv = []
for s in NETS:
    T = load_truth(s, 32)
    p = os.path.join(HERE, 'results', f'kprop_{s}_s0_L32_k2.json')
    if T is None or not os.path.exists(p):
        continue
    d = json.load(open(p)); mu = np.array(d['means'][31])
    nm = ((mu - T['mean'][31]) ** 2).mean() / T['var'][31].mean() - 1 / T['N']
    e.append(nm); fl = d['flops']
    full = 32 * (2 * n * n * (2 * n - 1) + 220 * n + 3 * n * n)          # flopscope-style count of the same K=2
    adv.append(((2 * n * n * 32 / fl) / nm, (2 * n * n * 32 / full) / nm))
if e:
    adv = np.array(adv)
    print(f'K=2 at 256 x 32, sigma = 0: median nMSE {np.median(e):.2e} over {len(e)} nets; advantage over MC at equal FLOPs: '
          f'{np.median(adv[:, 0]):.2f} with ARC\'s counter ({fl:.2e} FLOPs), {np.median(adv[:, 1]):.2f} with full-cost matmuls')
