"""Exploration on a dev net (not a prereg test): adaptive GS to ~1e-12 for a few neurons at depths 4/8/32;
level curve vs the Gaussian tail; geometry of the final particles. python explore_gs.py SEED SIGMA NEURONS(comma) N"""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
import gs as GS
from scipy.special import log_ndtr
from scipy.stats import chi

torch.set_num_threads(4)
seed, sigma = int(sys.argv[1]), float(sys.argv[2])
neurons = [int(v) for v in sys.argv[3].split(',')]
N = int(sys.argv[4])
for L in (4, 8, 32):
    net = TNet(seed, L, sigma)
    ex = np.load(f'/tmp/tails_explore1_{seed}_{sigma:g}_L{L}.npz')
    for i in neurons:
        gen = torch.Generator().manual_seed(777 + i)
        t0 = time.time()
        o = GS.gs_run(lambda Z: net.z(Z, i), lambda m: net.draw(m, gen), N, gen, float('inf'), rho=0.1, n_moves=5,
                      max_levels=12, keep=True)
        lev = o['levels'][:-1]; lf = np.cumsum(o['logfrac'][:-1])
        mu, sd = float(ex['mu'][i]), float(ex['sd'][i])
        Zf = o['Z']; x = Zf[:, 0].double()
        r = x.norm(dim=1)
        u = x / r[:, None]
        C = torch.cov(x.T)
        ev = torch.linalg.eigvalsh(C).flip(0)
        mdir = u.mean(0)
        print(f'L={L} neuron {i}: mu {mu:.3f} sd {sd:.3f} skew {float(ex["sk"][i]):.2f} exkurt {float(ex["ku"][i]):.2f}  ({time.time()-t0:.0f}s, acc last {np.mean(o["acc"][-1]):.2f}, b last {o["bs"][-1][-1]:.2f})')
        for k in (3, 5, 7, 9, 11):
            if k < len(lev):
                g = lev[k]; lp = lf[k]
                print(f'   level {k+1}: t={g:.3f} ({(g-mu)/sd:.2f} sd)  GS log10 p {lp/math.log(10):.2f}  Gaussian log10 p {log_ndtr(-(g-mu)/sd)/math.log(10):.2f}')
        print(f'   final particles: |x| mean {float(r.mean()):.2f} (typ {math.sqrt(255.5):.2f}) sd {float(r.std()):.2f}; |mean u| {float(mdir.norm()):.3f};'
              f' cov eig top5 {ev[:5].numpy().round(2)} bottom3 {ev[-3:].numpy().round(3)} median {float(ev.median()):.2f}; unique {len(torch.unique(x[:,0]))}')
