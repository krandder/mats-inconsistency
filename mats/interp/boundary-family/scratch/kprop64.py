"""ARC's kprop (copied, noise added exactly as in work/arc-noisy) on width-64 nets: estimates of E[a_L] for every
L <= Lmax (keep=True), for K=1, K=2 simple/augment, K=3 factorized simple/augment.
python scratch/kprop64.py SEEDS(comma) LMAX OUTDIR"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf
from kprop_noisy import kprop_noisy

seeds = [int(v) for v in sys.argv[1].split(',')]
LMAX = int(sys.argv[2]); OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
VARS = [('K1', 1, False, 'simple'), ('K2', 2, False, 'simple'), ('K2a', 2, False, 'augment'),
        ('K3f', 3, True, 'simple'), ('K3fa', 3, True, 'augment')]
for seed in seeds:
    net = bf.get_net(seed, 64, 32, 0.001)
    res = dict(seed=seed, n=64, sigma=0.001)
    for name, K, fac, kind in VARS:
        t0 = time.time()
        means, fl, secs = kprop_noisy(net, LMAX, K, factor=fac, kind=kind, keep=True)
        # flops for each depth L: rerun cost scales ~linearly; store the LMAX cost and per-depth costs separately
        res[name] = dict(means=[m.tolist() for m in means], flops_Lmax=fl, secs=secs)
        print(seed, name, f'{time.time() - t0:.1f}s', fl, flush=True)
    # per-depth costs (ARC counter) at the depths used
    for name, K, fac, kind in VARS:
        res[name]['flops_by_L'] = {}
        for L in (8, 16, 32):
            if L <= LMAX:
                _, fl, _ = kprop_noisy(net.truncate(L) if hasattr(net, 'truncate') else net, L, K, factor=fac,
                                       kind=kind, keep=False)
                res[name]['flops_by_L'][L] = fl
    json.dump(res, open(os.path.join(OUT, f'kprop64_{seed}.json'), 'w'))
