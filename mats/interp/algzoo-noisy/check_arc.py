"""Accuracy of ARC's noiseless models (all seeds) and of each with test-time-only noise at sigma in a grid."""
import sys, json, time
import torch
import az
torch.set_num_threads(1)
out = {}
cfgs = [(2, 2), (4, 3), (16, 10), (8, 5), (16, 5), (8, 10), (32, 10)]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2 ** 22
for (d, n) in cfgs:
    for s in list(range(5)) + (['ex'] if (d, n) == (16, 10) else []):
        t0 = time.time()
        W = az.load_arc(d, n, seed=None if s == 'ex' else s, example=(s == 'ex'))
        m = az.model_from(W, sigma=0.0)
        res = {'noiseless': az.accuracy(m, N, noise=False)}
        m.calibrate_r(2 ** 18)
        for sg in (0.01, 0.03, 0.1):
            m.sigma = sg
            m.calibrate_r(2 ** 18)
            res[f'test_noise_{sg}'] = az.accuracy(m, N, noise=True)
        out[f'{d}_{n}_{s}'] = res
        print(d, n, s, {k: round(v['acc'], 5) for k, v in res.items()}, round(time.time() - t0, 1), flush=True)
        json.dump(out, open('results/arc_baseline.json', 'w'), indent=1)
