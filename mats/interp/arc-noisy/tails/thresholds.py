"""Thresholds (PREREG-TAILS section 1): per (net, L, output i = 0..7), one adaptive splitting run at sigma = 0.001
(N = 2000, rho = 0.1, 5 pCN moves per level, scale adapted towards acceptance 0.3, 13 levels); t_(i,k) = its k-th level,
k = 4, 6, 8, 10, 12. python tails/thresholds.py SEEDS(comma) OUTDIR [NNEUR]"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seeds = [int(v) for v in sys.argv[1].split(',')]; OUT = sys.argv[2]
NN = int(sys.argv[3]) if len(sys.argv) > 3 else 8
DEPTHS = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [4, 8, 32]
if not torch.cuda.is_available():
    torch.set_num_threads(int(os.environ.get('TH_THREADS', '2')))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
os.makedirs(OUT, exist_ok=True)
for seed in seeds:
    res = dict(seed=seed, neurons=list(range(NN)), thr={}, levels={})
    t0 = time.time()
    for L in DEPTHS:
        net = TNet(seed, L, 0.001, device=dev)
        gen = torch.Generator(device=dev).manual_seed(50_000 + 97 * seed + L)
        idx = torch.arange(NN, device=dev)
        p = bgs.pilot(net, bgs.neuron_scores(net, idx), NN, 2000, gen, [[float('inf')]] * NN, max_levels=13)
        res['thr'][str(L)] = [[p['levels'][b][k - 1] for k in (4, 6, 8, 10, 12)] for b in range(NN)]
        res['levels'][str(L)] = p['levels']
    json.dump(res, open(os.path.join(OUT, f'thr_{seed}.json'), 'w'))
    print(seed, f'{time.time() - t0:.0f}s', flush=True)
