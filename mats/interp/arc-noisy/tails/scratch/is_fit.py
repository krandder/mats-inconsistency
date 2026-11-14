"""Exploration (dev net): importance sampling over x with a Gaussian fitted to splitting particles in the event.
Relative variance per IS sample, for mean-shift (C = I) and for C = I + top-k expansion. python is_fit.py SEED SIGMA L NEURON"""
import math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
import gs as GS

torch.set_num_threads(4)
seed, sigma, L, i = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
net = TNet(seed, L, sigma)
gen = torch.Generator().manual_seed(9300 + i + 31 * L)
score = lambda Z: net.z(Z, i)
draw = lambda m: net.draw(m, gen)
pil = GS.gs_run(score, draw, 2000, gen, float('inf'), rho=0.1, n_moves=5, max_levels=12)
lev = pil['levels'][:-1]; bs = GS.pad_bs(pil['bs'], len(lev), 5)
runs, _ = GS.frozen_runs(score, draw, lev, bs, gen, 1000, 8, 5)
ref = {j: GS.summarize(runs[:, j])['logp'] for j in range(len(lev))}
n = net.n
for j in (5, 7, 11):
    t = lev[j]
    o = GS.gs_run(score, draw, 4000, gen, t, levels=lev[:j], bs=[r + r for r in bs], n_moves=10, keep=True)
    X = o['Z'][:, 0].double()
    m = X.mean(0); C = torch.cov(X.T)
    evl, evec = torch.linalg.eigh(C); evl, evec = evl.flip(0), evec.flip(1)
    print(f'level {j + 1} t={t:.3f} ref log10p {ref[j] / math.log(10):.2f}; |m| {float(m.norm()):.2f}; top eig {evl[:6].numpy().round(1)}')
    for k in (0, 1, 3, 10, 30):
        lam = torch.clamp(evl[:k], min=1.0)
        V = evec[:, :k]
        M = 40000
        G = torch.randn(M, n, generator=gen, dtype=torch.float64)
        # x = m + (I + V (sqrt(lam) - 1) V^T) G  -> covariance I + V (lam - 1) V^T
        coef = (G @ V) * (lam.sqrt() - 1)
        x = m + G + coef @ V.T
        # log q(x) - log phi(x): q = N(m, C_k), C_k^{-1} = I + V (1/lam - 1) V^T, log det C_k = sum log lam
        d = x - m
        dv = d @ V
        quad_q = (d * d).sum(1) + ((dv ** 2) * (1 / lam - 1)).sum(1)
        logw = -0.5 * (x * x).sum(1) + 0.5 * quad_q + 0.5 * torch.log(lam).sum()
        Zs = draw(M); Zs[:, 0] = x.to(Zs.dtype)
        hit = (score(Zs) > t).double()
        w = torch.exp(logw - ref[j]) * hit        # ratio to the reference p
        rv = float(w.var() / max(float(w.mean()) ** 2, 1e-300))
        print(f'   k={k:2d}: est/ref {float(w.mean()):.3f}  hit rate {float(hit.mean()):.3f}  rel var per sample {rv:.1f}  ESS frac {float(w.sum() ** 2 / (w ** 2).sum() / M):.4f}', flush=True)
