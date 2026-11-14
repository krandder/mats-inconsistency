"""Review fix 2: recompute the chain's last-layer-exact estimates (K q_3)(A) for every threshold of rare_<net>.json with
frozen-tuning splitting in the Gaussian latent (fitted Gaussian chain with its defensive part, and the K = 2 chain).
usage: python chain_fz.py results/rare_<net>.json  -> results/chainfz_<net>.json
The fitted chain is re-created from its seed (first use of Generator(100 + seed)) and checked against the stored q_4(A).
"""
import json, math, os, sys, time
import torch
import rnet as R, fam as F
from rare_run import g_chain, kq_event_gs_frozen, kq_event_mix_gs_frozen

torch.set_num_threads(1)
LOG2 = math.log(2)


def main(p):
    r = json.load(open(p)); d = r['d']; seed = r['seed']; noln = '_noLN' in p
    net = R.RNet(d, r['L'], r['act'], r['g'], r['sigma'], seed, use_ln=not noln)
    qs, _ = g_chain(net, torch.Generator().manual_seed(100 + seed))
    chk = max(abs(qs[-1].prob_halfspace(net.w, lev['t']) - lev['chain']['log_qL']) for lev in r['levels'])
    assert chk < 1e-8, chk
    k2c = R.k2_propagate(net, keep=True); qk2 = F.Gauss(k2c[-2][0], k2c[-2][1])
    gen = torch.Generator().manual_seed(7700 + seed + 7 * d + (1 if noln else 0))
    ts = [lev['t'] for lev in r['levels']]
    t0 = time.time()
    fit = kq_event_mix_gs_frozen(net, qs[-2], ts, gen, R=16)
    k2 = kq_event_gs_frozen(net, qk2.mu, qk2.L, ts, gen, R=16)
    out = dict(src=os.path.basename(p), d=d, act=r['act'], g=r['g'], sigma=r['sigma'], seed=seed, chain_check=chk,
               secs=time.time() - t0, per_level=[])
    for lev in r['levels']:
        t = lev['t']; lf, sef, parts = fit[t]
        row = dict(k=lev['k'], t=t, log_KqLm1=lf, rel_se_KqLm1=sef, parts={k: {kk: v[kk] for kk in ('logp', 'rel_se', 'half_bits', 'max_share')}
                                                                         for k, v in parts.items()},
                   log_KqLm1_k2=k2[t]['logp'], rel_se_KqLm1_k2=k2[t]['rel_se'],
                   old_log_KqLm1=lev['chain']['log_KqLm1'], old_log_KqLm1_k2=lev['chain'].get('log_KqLm1_k2'))
        out['per_level'].append(row)
        print(f"k={lev['k']}: fitted {(lf - row['old_log_KqLm1']) / LOG2:+.2f} bits vs old (rel se {sef:.2f}); "
              f"K2 {(row['log_KqLm1_k2'] - row['old_log_KqLm1_k2']) / LOG2:+.2f} (rel se {row['rel_se_KqLm1_k2']:.2f})", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(p), os.path.basename(p).replace('rare_', 'chainfz_')), 'w'), indent=1)
    print('done', f"{out['secs']:.0f}s", flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
