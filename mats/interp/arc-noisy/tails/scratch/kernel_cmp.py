"""Exploration (dev net): pCN vs elliptical-slice moves inside fixed-level splitting. Work-normalised relative variance
at each target level. python kernel_cmp.py SEED SIGMA L NEURON N R"""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
import gs as GS

torch.set_num_threads(4)
seed, sigma, L, i, N, R = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
net = TNet(seed, L, sigma)
gen = torch.Generator().manual_seed(9100 + i + 31 * L)
score = lambda Z: net.z(Z, i)
draw = lambda m: net.draw(m, gen)
pil = GS.gs_run(score, draw, 2000, gen, float('inf'), rho=0.1, n_moves=5, max_levels=12)
lev = pil['levels'][:-1]                       # 12 levels ~ 1e-1 .. 1e-12
bs = GS.pad_bs(pil['bs'], len(lev), 5)
print('levels', np.round(lev, 3).tolist())
tgt = [3, 5, 7, 9, 11]


def report(name, runs, nev):
    out = []
    for j in tgt:
        s = GS.summarize(runs[:, j])
        rv1 = (s['rel_se'] * math.sqrt(R)) ** 2           # relative variance of one run
        out.append('%.1e: log10p %.2f rv1 %.2f wnrv %.1e' % (10.0 ** -(j + 1), s['logp'] / math.log(10), rv1, rv1 * nev / R))
    print(f'{name:>14} evals/run {nev / R:.0f}  ' + ' | '.join(out), flush=True)


t0 = time.time()
runs, nev = GS.frozen_runs(score, draw, lev, bs, gen, N, R, 5)
report('pCN m=5', runs, nev)
for mv in (1, 2, 4):
    rr = []; ne = 0; cap = 0
    for r in range(R):
        o = GS.gs_run_ess(score, draw, N, gen, lev[-1], levels=lev[:-1], n_moves=mv)
        c = np.cumsum(o['logfrac']); rr.append(np.concatenate([c, np.full(len(lev) - len(c), -np.inf)])); ne += o['n_eval']; cap += o['capped']
    report(f'ESS m={mv}', np.array(rr), ne)
    print('   capped', cap, f'{time.time() - t0:.0f}s', flush=True)
o = GS.gs_run_ess(score, draw, N, gen, lev[-1], levels=lev[:-1], n_moves=2, keep=True)
x = o['Z'][:, 0].double(); ev = torch.linalg.eigvalsh(torch.cov(x.T)).flip(0)
print('ESS m=2 final particles: |x| %.2f, cov eig top5 %s median %.2f' % (float(x.norm(dim=1).mean()), ev[:5].numpy().round(1), float(ev.median())))
