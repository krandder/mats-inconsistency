"""cmech.py WORLD (POST HOC, evaluation side): how much each predictor still reads its sampled trajectories.
 to_one  mean |forecast - one pass of M| over t1 queries with 20-29 revealed checks, K = 4
 draw    mean |forecast from draw A - forecast from draw B| over the t2 queries, K = 4 (two disjoint trajectory sets)
 exc_large  excess on t1 queries with 20-29 revealed checks, K = 4."""
import json
import sys

import numpy as np

w = sys.argv[1]
E = np.load('out/w%s/exact.npz' % w)
sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))
mk = (~E['t1_mask']) & (E['t1_mask'].sum(1) >= 20)[:, None]
ps = E['t1_pstar']
kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
out = {}
for n in (64, 1024, 20480):
    for a in ('data', 'i@1', 'i@10', 'iio@1', 'iio@10', 'iis@0.1', 'iis@1', 'iis@10', 'iii@1', 'iiisym@10', 'synth@1'):
        try:
            A = np.load('out/w%s/%s_n%d_d0_r0.npz' % (w, a, n))
        except FileNotFoundError:
            continue
        p = np.clip(sig(A['t1_K4']), 1e-6, 1 - 1e-6)
        out['%s n=%d' % (a, n)] = dict(to_one=float(np.abs(p - E['t1_p_one'])[mk].mean()),
                                       draw=float(np.abs(sig(A['t2_K4_r0']) - sig(A['t2_K4_r1'])).mean()),
                                       exc_large=float(kl(ps, p)[mk].mean()))
json.dump(out, open('out/mechanism_w%s.json' % w, 'w'), indent=0)
for k, v in out.items():
    print('%-22s to_one %.3f  draw %.3f  exc_large %.3f' % (k, v['to_one'], v['draw'], v['exc_large']))
