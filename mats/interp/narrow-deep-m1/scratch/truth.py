"""Ground truth for E[a_L] and one tail probability (plain MC, float64).
usage: python truth.py SEED SIGMA OUTPREFIX [LOG2P=24] [N=8] [L=256]
- E[a_L], per-coordinate variance, tr Var; per-block means (64 blocks) for an empirical SE.
- Tail: t = (1 - 1e-4) quantile of |a_L| over the first 1/8 of the paths (threshold set; 2^21 of 2^24); p = P(|a_L| > t) over the remaining
  paths (disjoint, so p is unbiased for that t); also t and p for 1e-3.
- Layer-c oracle law summaries (used by the forward-part study only): E[h_c], Cov(h_c), E[a_c] and tr Var(a_c) for
  c = 1..16 (state after noise h_c; a_c post-ReLU, before noise).
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
import nd

torch.set_num_threads(1)
SEED, SIG, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
LOG2P = int(sys.argv[4]) if len(sys.argv) > 4 else 24
N = int(sys.argv[5]) if len(sys.argv) > 5 else 8
L = int(sys.argv[6]) if len(sys.argv) > 6 else 32 * N if len(sys.argv) > 5 else 256
DT = nd.DT
CH = 1 << 16
NB = 1 << (LOG2P - 16)
CUTS = list(range(1, 17))
t0 = time.time()
net = nd.Net('plain', SEED, N, L, SIG)
g = torch.Generator().manual_seed(77_000_001 + 13 * SEED + int(round(SIG * 1e6)))
so = torch.zeros(N, dtype=DT); sso = torch.zeros(N, dtype=DT)
blocks = []
acc = {c: dict(h=torch.zeros(N, dtype=DT), hh=torch.zeros(N, N, dtype=DT), a=torch.zeros(N, dtype=DT),
               aa=torch.zeros(N, dtype=DT)) for c in CUTS}
norms_thr = []; norms_rest = []
scale = None
for b in range(NB):
    x = torch.randn(CH, N, generator=g, dtype=DT)
    h = x
    for l in range(L):
        a = torch.relu(h @ net.W[l].T)
        if l < L - 1 and SIG > 0:
            h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
        else:
            h = a
        if (l + 1) in acc:
            A = acc[l + 1]
            A['h'] += h.sum(0); A['hh'] += h.T @ h; A['a'] += a.sum(0); A['aa'] += a.pow(2).sum(0)
    out = h
    so += out.sum(0); sso += out.pow(2).sum(0)
    blocks.append(out.mean(0))
    nr = out.norm(dim=1)
    if scale is None:
        scale = float(nr.pow(2).mean().sqrt()) or 1.0
    (norms_thr if b < max(1, NB // 8) else norms_rest).append((nr / scale).float())
P = NB * CH
mean = so / P; var = sso / P - mean ** 2; trv = float(var.sum())
B = torch.stack(blocks)                                       # (NB, N)
se2_nmse = float(((B - mean) ** 2).sum(1).sum() / (NB - 1) / NB / trv) if trv > 0 else None   # fixed 23:10Z (was 1/NB too small)
nt = torch.cat(norms_thr).double(); nr_ = torch.cat(norms_rest).double()
tails = {}
for q in (1e-3, 1e-4):
    t = float(torch.quantile(nt, 1 - q)) if trv > 0 else 0.0
    p = float((nr_ > t).double().mean())
    tails[str(q)] = dict(t=t * scale, p=p, se=math.sqrt(max(p * (1 - p), 1e-300) / len(nr_)), n_rest=len(nr_),
                         n_thr=len(nt))
lay = {}
for c in CUTS:
    A = acc[c]
    mh = A['h'] / P; Ch = A['hh'] / P - torch.outer(mh, mh)
    ma = A['a'] / P; va = A['aa'] / P - ma ** 2
    lay[c] = dict(mean_h=mh.tolist(), cov_h=Ch.tolist(), mean_a=ma.tolist(), trvar_a=float(va.sum()))
res = dict(seed=SEED, sigma=SIG, n=N, L=L, paths=P, mean=mean.tolist(), var=var.tolist(), trvar=trv,
           truth_nmse_expected=1.0 / P, truth_nmse_blocks=se2_nmse, block_se_fixed=True, out_scale=scale, tails=tails, r=net.r.tolist(),
           flops_path=net.flops_path(), layers=lay, sec=time.time() - t0)
json.dump(res, open(OUT + '.json', 'w'))
print(f"seed {SEED} sig {SIG:g} n={N} L={L}: P=2^{LOG2P} trVar={trv:.4g} |mean|^2/trVar={float(mean.pow(2).sum())/max(trv,1e-300):.4g} "
      f"block-SE nMSE={se2_nmse} tails={ {k: (v['p'], round(v['se']/max(v['p'],1e-300),3)) for k, v in tails.items()} } "
      f"({time.time()-t0:.0f}s)", flush=True)
