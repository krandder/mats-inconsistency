"""Propagator validation on random-initialised models (before any training).

For one depth-8 network (and its first L blocks, as ARC does), compare K=2 and K=4 propagated estimates with
large-sample Monte Carlo: E[mu], E[logs], E[nll] (variance-normalised squared error and the equivalent number of
samples N_eff = Var/err^2) and the residual-stream statistics at the last point.

usage: python validate_init.py --out DIR [--n 524288] [--seeds 0 1] [--stress 1 2]
"""
import argparse, json, os, time
import torch
import world_small as W
from model import Net
from prop import Propagator
from mcstats import mc_truth, stat_errors

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=1 << 19)
ap.add_argument('--seeds', type=int, nargs='+', default=[0])
ap.add_argument('--stress', type=float, nargs='+', default=[1.0])
ap.add_argument('--depths', type=int, nargs='+', default=[1, 2, 4, 6, 8])
ap.add_argument('--dtype', default='float32')
ap.add_argument('--device', default='cpu')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dt = getattr(torch, a.dtype)
rows = []
for seed in a.seeds:
    for stress in a.stress:
        torch.manual_seed(seed)
        net = Net(8).double()
        with torch.no_grad():                       # stress: scale query/key weights (larger attention scores)
            for b in net.blocks:
                b.qkv.weight[:2 * net.d] *= stress
        for L in a.depths:
            t0 = time.time()
            truth, pts = mc_truth(net, a.n, seed=777 + seed, upto=L, device=a.device)
            tmc = time.time() - t0
            for K in (2, 4):
                pr = Propagator(K=K, dtype=dt, device=a.device)
                t0 = time.time()
                with torch.no_grad():
                    o = pr.run(net.to(a.device), upto=L)
                tp = time.time() - t0
                r = dict(seed=seed, stress=stress, depth=L, K=K, J=pr.J, sec_prop=tp, sec_mc=tmc, n_mc=a.n)
                for q in ('mu', 'logs', 'nll'):
                    est = o['out'][f'E_{q}'].item(); tru = truth[f'E_{q}']; var = truth[f'V_{q}']
                    r[f'est_{q}'] = est; r[f'true_{q}'] = tru; r[f'var_{q}'] = var
                    r[f'nse_{q}'] = (est - tru) ** 2 / var
                    r[f'neff_{q}'] = var / max((est - tru) ** 2, 1e-30)
                    r[f'mcnoise_nse_{q}'] = 1.0 / a.n
                e = stat_errors({k: v.double().cpu() for k, v in o['points'][-1].items()}, pts[-1])
                r.update({f'last_{k}': v for k, v in e.items()})
                rows.append(r)
                print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
                json.dump(rows, open(os.path.join(a.out, 'validate_init.json'), 'w'), indent=1)
