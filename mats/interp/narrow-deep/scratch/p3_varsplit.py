"""Where does the output's variance come from? (coordinator's request, 2026-10-29 ~22:10Z)
usage: python p3_varsplit.py FAM N L SIGMA SEEDS OUTJSON
Shares of tr Var(out) explained by conditioning on: the input x (rest: noise), its direction u = x/|x|, its norm
|x|. One-way ANOVA with M groups x K draws, debiased: Var(E[out|G]) = Var_between - E Var_within / K.
For a noiseless positively homogeneous net out = |x| F(u): direction share = (E|x|)^2 tr Var F(u) / tr Var(out).
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]; OUT = sys.argv[6]
M, K = 4096, 8
DT = nd.DT


def anova(Y):
    """Y (M, K, n) -> share of total variance explained by the group mean."""
    gm = Y.mean(1)                                    # (M, n)
    within = Y.var(1, unbiased=True).sum(-1).mean()   # E tr Var within
    between = gm.var(0, unbiased=True).sum()
    tot = Y.reshape(-1, Y.shape[-1]).var(0, unbiased=True).sum()
    return float((between - within / Y.shape[1]) / tot), float(tot)


res = {}
for seed in SEEDS:
    t0 = time.time()
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(13_000_019 + seed)
    out = {}
    # input share: same x, K noise draws
    x = torch.randn(M, N, generator=g, dtype=DT).repeat_interleave(K, 0)
    Y = net.forward(x, gen=g).reshape(M, K, N)
    out['input'], tot = anova(Y)
    # direction share: same u, K draws of (radius, noise)
    u = torch.randn(M, N, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
    rad = torch.randn(M * K, N, generator=g, dtype=DT).norm(dim=1, keepdim=True)
    Y = net.forward(u.repeat_interleave(K, 0) * rad, gen=g).reshape(M, K, N)
    out['direction'], _ = anova(Y)
    # norm share: same radius, K draws of (direction, noise)
    rad = torch.randn(M, N, generator=g, dtype=DT).norm(dim=1, keepdim=True).repeat_interleave(K, 0)
    u = torch.randn(M * K, N, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
    Y = net.forward(u * rad, gen=g).reshape(M, K, N)
    out['norm'], _ = anova(Y)
    out['trvar'] = tot
    res[seed] = out
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: shares input={out['input']:.3f} direction={out['direction']:.3f} "
          f"norm={out['norm']:.3f} (trVar {tot:.3g}; {time.time()-t0:.0f}s)", flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, M=M, K=K, res=res), open(OUT, 'w'))
