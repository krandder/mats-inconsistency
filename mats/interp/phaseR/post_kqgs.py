"""Recompute the chain's last-layer-exact estimates (K q_{L-1})(A) with latent splitting for rare_*.json files whose
values came from mean-shift importance sampling (unreliable at d = 64 for p < 2^-30). Re-creates the identical fitted G chain
(same generator seed, first use) and the K=2 chain. Old values kept as *_is. usage: python post_kqgs.py results/rare_*.json"""
import json, math, sys
import torch
import rnet as R, fam as F
from rare_run import g_chain, kq_event_gs, kq_event_mix_gs

LOG2 = math.log(2)
for p in sys.argv[1:]:
    r = json.load(open(p))
    if all(lev['chain'].get('kq_method') == 'gs' for lev in r['levels']):
        print(p, 'already gs'); continue
    net = R.RNet(r['d'], r['L'], r['act'], r['g'], r['sigma'], r['seed'], use_ln='_noLN' not in p)
    gen = torch.Generator().manual_seed(100 + r['seed'])
    qs, _ = g_chain(net, gen)                       # identical to the run's chain (first use of the same generator)
    k2c = R.k2_propagate(net, keep=True); qk2 = F.Gauss(k2c[-2][0], k2c[-2][1])
    gen2 = torch.Generator().manual_seed(777 + r['seed'])
    for lev in r['levels']:
        ch = lev['chain']
        if ch.get('kq_method') == 'gs': continue
        lp = lev['truth']['logp']; t = lev['t']
        a, ra = kq_event_mix_gs(net, qs[-2], t, gen2)
        b, rb = kq_event_gs(net, qk2.mu, qk2.L, t, gen2)
        ch['log_KqLm1_is'] = ch.get('log_KqLm1'); ch['log_KqLm1_k2_is'] = ch.get('log_KqLm1_k2')
        ch.update(log_KqLm1=a, log_KqLm1_k2=b, rel_sd_KqLm1=ra, rel_sd_KqLm1_k2=rb, kq_method='gs', post='post_kqgs.py',
                  bits_err_KqLm1=(a - lp) / LOG2, bits_err_KqLm1_k2=(b - lp) / LOG2, bits_err_qL=(ch['log_qL'] - lp) / LOG2)
        old = ch['log_KqLm1_is']; oldk = ch['log_KqLm1_k2_is']
        print(p.split('/')[-1], lev['k'], f"fitted: gs {(a - lp) / LOG2:+.2f} (is {(old - lp) / LOG2 if old is not None else float('nan'):+.2f}) "
              f"K2: gs {(b - lp) / LOG2:+.2f} (is {(oldk - lp) / LOG2 if oldk is not None else float('nan'):+.2f}) rel sd {ra:.2f}/{rb:.2f}", flush=True)
        json.dump(r, open(p, 'w'), indent=1)
