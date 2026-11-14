"""Evaluate trained narrow nets (final files or checkpoints): RNN accuracy / cross-entropy on 2nd argmax, MLP sorting
nMSE, noise on and off, on 2^18 fresh inputs. usage: python p4_eval.py OUTJSON NPZ [NPZ ...]"""
import sys, os, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets

torch.set_num_threads(1)
DT = torch.float64
out = {}
for p in sys.argv[2:]:
    z = np.load(p)
    kind = 'rnn' if 'W_hh' in z.files else 'mlp'
    net = tnets.RNNNet(p) if kind == 'rnn' else tnets.MLPNet(p)
    g = torch.Generator().manual_seed(4242)
    acc = {True: 0.0, False: 0.0}; ce = {True: 0.0, False: 0.0}; nt = 0
    for i in range(8):
        x = torch.randn(1 << 15, net.n, generator=g, dtype=DT)
        for noise in (True, False):
            o = net.forward(x, gen=g, noise=noise)
            if kind == 'rnn':
                y = x.argsort(-1)[:, -2]
                acc[noise] += float((o.argmax(-1) == y).double().sum())
                ce[noise] += float(torch.nn.functional.cross_entropy(o, y, reduction='sum'))
            else:
                y = x.sort(-1).values
                ce[noise] += float((o - y).pow(2).mean(1).sum())
                acc[noise] += float(((o - y).abs().amax(1) < 0.1).double().sum())
        nt += 1 << 15
    if kind == 'rnn':
        r = dict(kind=kind, T=net.T, d=net.d, sigma=net.sigma, acc_on=acc[True] / nt, acc_off=acc[False] / nt,
                 ce_on=ce[True] / nt, ce_off=ce[False] / nt, chance=1.0 / net.T)
    else:
        vy = float(np.mean(np.var(np.sort(np.random.RandomState(0).standard_normal((1 << 20, net.n)), 1), 0)))
        r = dict(kind=kind, arch=net.arch, L=net.L, sigma=net.sigma, nmse_on=ce[True] / nt / vy, nmse_off=ce[False] / nt / vy,
                 frac_all_within_0p1_on=acc[True] / nt)
    out[os.path.basename(p)] = r
    print(os.path.basename(p), json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
