"""Signed bias decomposition of the deterministic arms (post hoc diagnostic, evaluation nets).
The output of these nets is synchronised (a_L ~ s u_L), so every estimator's error lies along one direction; errors
from different sources can cancel. For c in {2, 4, 8}: est_ds(c) = noiseless downstream (c -> 32 noiseless, J_ref)
applied to the exact law at c (2^17 sampled paths). Then
  noise+downstream bias b_ds(c) = est_ds(c) - truth,  forward bias of TS config F: b_fwd = est_TS(F, c) - est_ds(c),
and est_TS - truth = b_fwd + b_ds. Also the BO estimates against the sigma = 0 truth when it exists (projection +
quadrature bias alone). Signed scalars in units of SD(a_L) along u = truth/|truth|.
usage: python bias_decomp.py SEED"""
import sys, json, torch, math
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1])
R = '/home/agent/workspaces/mats/work/narrow-deep-m1/runs'
T = json.load(open(f'{R}/truth_s{SEED}_sig0.001.json'))
net = nd.Net('plain', SEED, 8, 256, 1e-3, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); sd = math.sqrt(T['trvar'])
u = truth / truth.norm()
g = torch.Generator().manual_seed(37)
P = 1 << 17
h = torch.randn(P, n, generator=g, dtype=DT)
res = dict(seed=SEED, ds={}, ts=[], bo=[])
TS = json.load(open(f'{R}/ts_s{SEED}_sig0.001.json'))
l = 0
for c in (2, 4, 8):
    while l < c:
        a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]; l += 1
    cand = bvf.noiseless_pass(net, h[:256], c, 32, bvf.Flops())
    deep = bvf.Deep(net, 32, cand, torch.ones(256, dtype=DT), bvf.Flops())
    V = deep.value(bvf.noiseless_pass(net, h, c, 32, bvf.Flops()), bvf.Flops())
    est_ds = V.mean(0)
    b_ds = float(((est_ds - truth) @ u) / sd)
    perp = float(((est_ds - truth) - ((est_ds - truth) @ u) * u).norm() / sd)
    res['ds'][c] = dict(b=b_ds, perp=perp, se=float(V.std(0).norm() / sd / math.sqrt(P)))
    for r in TS['rows']:
        if r['c'] != c:
            continue
        est = torch.tensor(r['est'], dtype=DT)
        b_ts = float(((est - truth) @ u) / sd)
        res['ts'].append(dict(F=r['F'], c=c, tau=r['tau'], b_ts=b_ts, b_fwd=b_ts - b_ds, b_ds=b_ds))
    print(f"s{SEED} c={c}: downstream+noise bias b_ds = {b_ds:+.4f} SD (perp {perp:.1e}, MC se {res['ds'][c]['se']:.1e}); TS: " +
          ' '.join(f"{x['F']}{'' if x['F']=='K2X' else x['tau']}: total {x['b_ts']:+.4f} = fwd {x['b_fwd']:+.4f} + ds {x['b_ds']:+.4f};"
                   for x in res['ts'] if x['c'] == c), flush=True)
try:
    T0 = json.load(open(f'{R}/truth_s{SEED}_sig0.json'))
    tr0 = torch.tensor(T0['mean'], dtype=DT)
    res['noise_total'] = float(((truth - tr0) @ u) / sd)
    print(f"s{SEED}: total noise effect (sigma=1e-3 truth - sigma=0 truth) = {res['noise_total']:+.4f} SD", flush=True)
    B = json.load(open(f'{R}/bo_s{SEED}_sig0.001.json'))
    for r in B['rows']:
        if r['mode'] != 'norm':
            continue
        e0 = torch.tensor(r['est_full'], dtype=DT)
        res['bo'].append(dict(k=r['k'], m=r['m'], b_vs_noiseless=float(((e0 - tr0) @ u) / sd), b_vs_truth=float(((e0 - truth) @ u) / sd)))
    print(f"s{SEED} BO (norm, full): bias vs sigma=0 truth / vs truth (SD): " +
          ' '.join(f"k{x['k']}m{x['m']}: {x['b_vs_noiseless']:+.4f}/{x['b_vs_truth']:+.4f}" for x in res['bo']), flush=True)
except FileNotFoundError:
    pass
json.dump(res, open(f'{R}/biasdecomp_s{SEED}.json', 'w'), indent=1)
