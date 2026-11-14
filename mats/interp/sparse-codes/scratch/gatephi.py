"""Neuron gate phi (ReLU gates a_j > 0), for the claims review (finding 8): median |phi| among gates firing in 1-99% of
samples and among near-boundary gates (binary entropy > 0.1 bit, the boundary-family definition). usage: gatephi.py N"""
import json, os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nets
torch.set_num_threads(3)
n = int(sys.argv[1])
net = nets.SNet('arc', n, 65, 0.01)
X = net.sample(1 << 15, torch.Generator().manual_seed(4321), [8, 16, 24])   # same states as featcorr.py
out = {}
for l, A in X.items():
    g = (A > 0).double(); p = g.mean(0)
    h = -(p * torch.log2(p.clamp_min(1e-12)) + (1 - p) * torch.log2((1 - p).clamp_min(1e-12)))
    res = {}
    for tag, sel in (('mid', (p > 0.01) & (p < 0.99)), ('nearboundary', h > 0.1)):
        if int(sel.sum()) < 2:
            continue
        P = torch.corrcoef(g[:, sel].T); off = ~torch.eye(P.shape[0], dtype=torch.bool)
        res[tag] = dict(count=int(sel.sum()), phi_absmed=float(P[off].abs().median()), phi_absmax=float(P[off].abs().max()))
    out[l] = res
    print(l, res, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', f'gatephi_n{n}.json'), 'w'), indent=1)
