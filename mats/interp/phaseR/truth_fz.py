"""Review fix 2: reference rare-event probabilities by fixed-level splitting with the pCN scale frozen from an independent
pilot (PREREG-R amendment 2). One run per repeat to the deepest threshold gives every level of rare_<net>.json.
usage: python truth_fz.py results/rare_<net>.json [N] [R] [moves]  -> results/truthfz_<net>.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import rnet as R, rare_gs as GS

torch.set_num_threads(1)
LOG2 = math.log(2)


def main(p, N=None, Rr=None, mv=10):
    r = json.load(open(p))
    d = r['d']
    N = N or (8192 if d >= 128 else 16384)
    Rr = Rr or (16 if d >= 128 else (24 if d >= 64 else 32))
    net = R.RNet(d, r['L'], r['act'], r['g'], r['sigma'], r['seed'], use_ln='_noLN' not in p)
    gen = torch.Generator().manual_seed(6100 + r['seed'] + 7 * d + (1 if '_noLN' in p else 0))
    score, draw = GS.net_score_draw(net, gen)
    ts = [lev['t'] for lev in r['levels']]
    t0 = time.time()
    fe = GS.frozen_estimates(score, draw, ts, gen, N, Rr, n_moves=mv, levels=r['levels_curve'], pilot_N=8192)
    per = net.L * (2 * d * d + 12 * d)
    out = dict(src=os.path.basename(p), d=d, act=r['act'], g=r['g'], sigma=r['sigma'], seed=r['seed'], N=N, R=Rr, moves=mv,
               levels=fe['levels'], bs=fe['bs'], flops_total=(fe['n_eval'] + fe['pilot_n_eval']) * per, secs=time.time() - t0,
               per_level=[])
    for lev in r['levels']:
        s = fe['per_target'][lev['t']]
        old = lev['truth']['logp']; hi = lev.get('truth_hi', {}).get('logp')
        row = dict(k=lev['k'], t=lev['t'], **s, old_logp=old, old_rel_se=lev['truth']['rel_se'], hi_logp=hi,
                   bulk_logp=lev['crosscheck']['bulk_logp'], bulk_rel_sd=lev['crosscheck']['bulk_rel_sd'])
        bits = lambda x: (x - s['logp']) / LOG2 if x is not None and np.isfinite(s['logp']) else float('nan')
        row.update(d_old_bits=bits(old), d_hi_bits=bits(hi), d_bulk_bits=bits(row['bulk_logp']))
        flags = []
        if not s['rel_se'] <= 0.25: flags.append('relSE')
        if not abs(s['half_bits']) <= 1: flags.append('halves')
        if not abs(row['d_bulk_bits']) <= 1: flags.append('bulk')
        if not abs(row['d_old_bits']) <= 1: flags.append('old')
        row['flags'] = flags; row['provisional'] = bool(flags)
        row.pop('logs_raw', None)
        out['per_level'].append(row)
        print(f"k={lev['k']} new 2^{s['logp'] / LOG2:.2f} (rel se {s['rel_se']:.3f}, halves {s['half_bits']:+.2f}, max share "
              f"{s['max_share']:.2f}) old {old / LOG2:.2f} bulk {row['bulk_logp'] / LOG2:.2f} flags {flags}", flush=True)
    name = os.path.basename(p).replace('rare_', 'truthfz_')
    json.dump(out, open(os.path.join(os.path.dirname(p), name), 'w'), indent=1)
    print('done', name, f"{out['secs']:.0f}s", flush=True)


if __name__ == '__main__':
    a = sys.argv
    main(a[1], int(a[2]) if len(a) > 2 and a[2] != '-' else None, int(a[3]) if len(a) > 3 and a[3] != '-' else None,
         int(a[4]) if len(a) > 4 else 10)
