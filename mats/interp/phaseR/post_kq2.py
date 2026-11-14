"""Add log_KqLm1_k2 (K=2 chain for layers 1..L-1, exact last layer, mean-shift IS) to rare_*.json levels that lack it.
usage: python post_kq2.py results/rare_<net>.json"""
import json, math, sys, torch
import rnet as R, fam as F
from rare_run import kq_event_single

LOG2 = math.log(2)
p = sys.argv[1]
r = json.load(open(p))
net = R.RNet(r['d'], r['L'], r['act'], r['g'], r['sigma'], r['seed'], use_ln='_noLN' not in p)
k2c = R.k2_propagate(net, keep=True); q = F.Gauss(k2c[-2][0], k2c[-2][1])
gen = torch.Generator().manual_seed(4242)
r.setdefault('k2chain_flops', net.flops_k2() * (net.L - 1) // net.L)
for lev in r['levels']:
    ch = lev['chain']
    if 'log_KqLm1_k2' not in ch:
        v = kq_event_single(net, q, lev['t'], gen)
        lp = lev['truth']['logp']
        ch.update(log_KqLm1_k2=v, bits_err_KqLm1_k2=(v - lp) / LOG2, bits_err_KqLm1=(ch['log_KqLm1'] - lp) / LOG2,
                  bits_err_qL=(ch['log_qL'] - lp) / LOG2, post='post_kq2.py')
        print(lev['k'], round(ch['bits_err_KqLm1_k2'], 2), round(ch['bits_err_KqLm1'], 2), flush=True)
json.dump(r, open(p, 'w'), indent=1)
