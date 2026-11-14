"""Check the closure tails of p5_split by sampling the closure's own output law (not the net), and report how
well the closure gets the output variance. usage: python p5_tailcheck.py KIND NPZ LOCATEJSON SPLITJSON OUTJSON
For k = 0 (K2) and S1 with k = 8 (exact ReLU moments): draw 2^22 samples from the output Gaussian mixture, estimate
P(|output| > t) at the truth's thresholds (from SPLITJSON), and tr Cov(output) of the closure vs the truth's."""
import sys, os, json, math
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, k2mix

torch.set_num_threads(1)
DT = torch.float64
KIND, NPZ, LOC, SPL, OUT = sys.argv[1:6]
net = tnets.RNNNet(NPZ) if KIND == 'rnn' else tnets.MLPNet(NPZ)
loc = json.load(open(LOC)); spl = json.load(open(SPL))
g = torch.Generator().manual_seed(53_000_011)
res = {}
for tag, k in (('k0', 0), ('S1_k8', 8)):
    sp = {}
    for (l, i, _) in loc['gate_acc']['top'][:k]:
        sp.setdefault(l, []).append(i)
    comps, fl = (k2mix.rnn_mix if KIND == 'rnn' else k2mix.res_mix)(net, sp, True)
    out = k2mix.readout(net, comps)
    W = torch.tensor([c[0] for c in out], dtype=DT); W = W / W.sum()
    mu_all = sum(float(W[j]) * out[j][1] for j in range(len(out)))
    cov_all = sum(float(W[j]) * (out[j][2] + torch.outer(out[j][1] - mu_all, out[j][1] - mu_all)) for j in range(len(out)))
    NS = 1 << 22
    cnt = torch.multinomial(W, NS, replacement=True, generator=g).bincount(minlength=len(out))
    exceed = {q: 0 for q in spl['tails']}
    for j, (wj, mj, Cj) in enumerate(out):
        nj = int(cnt[j])
        if nj == 0: continue
        ev, Q = torch.linalg.eigh(0.5 * (Cj + Cj.T)); A = Q * ev.clamp_min(0).sqrt()[None, :]
        for s0 in range(0, nj, 1 << 18):
            m = min(1 << 18, nj - s0)
            y = mj + torch.randn(m, len(mj), generator=g, dtype=DT) @ A.T
            nrm = y.norm(dim=1)
            for q, (t, p) in spl['tails'].items():
                exceed[q] += int((nrm > t).sum())
    res[tag] = dict(ncomp=len(out), trcov_closure=float(torch.trace(cov_all)), trvar_truth=spl['trvar'],
                    tails={q: dict(p_closure=exceed[q] / NS, p_true=spl['tails'][q][1]) for q in spl['tails']})
    print(f"{os.path.basename(NPZ)} {tag}: tr Cov closure / truth = {res[tag]['trcov_closure']/spl['trvar']:.3g}; tails (closure vs truth): " +
          ' '.join(f"q={q}: {v['p_closure']:.3g} vs {v['p_true']:.3g}" for q, v in res[tag]['tails'].items()), flush=True)
json.dump(res, open(OUT, 'w'))
