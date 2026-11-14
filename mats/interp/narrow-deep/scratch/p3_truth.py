"""Part 3 truth and oracle statistics (PREREG Part 3).
usage: python p3_truth.py FAM N L SIGMA SEED OUTPREFIX [LOG2_PATHS=22]
Truth: plain MC, 2^22 paths (float64): E[out], per-coordinate variance, |out| quantiles and exceedances.
Oracle at layers l0 (state after layer l0's noise; residual: the stream h_l0): E[h_l0], and per log-radius bin
(16 quantile bins from a 2^16 pilot, plus a "zero" bin for |h| = 0): weight, sum h, sum h h^T.
Writes OUTPREFIX.json (summary) and OUTPREFIX_oracle.npz.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA, SEED, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
LOG2 = int(sys.argv[7]) if len(sys.argv) > 7 else 22
NB = 16; CH = 1 << 16
L0S = [l for l in (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224) if l < L]
DT = nd.DT
t0 = time.time()
net = nd.Net(FAM, SEED, N, L, SIGMA)
# pilot: bin edges
g = torch.Generator().manual_seed(8_000_009 + SEED)
x = torch.randn(CH, N, generator=g, dtype=DT)
_, kept = net.forward(x, gen=g, keep=set(L0S))
edges = {}
for l0 in L0S:
    r = kept[l0][0].norm(dim=1)
    lr = torch.log(r[r > 0])
    edges[l0] = torch.quantile(lr, torch.linspace(0, 1, NB + 1, dtype=DT)[1:-1]) if len(lr) > NB else torch.zeros(NB - 1, dtype=DT)
acc = {l0: dict(w=torch.zeros(NB + 1, dtype=DT), s=torch.zeros(NB + 1, N, dtype=DT), ss=torch.zeros(NB + 1, N, N, dtype=DT),
                m=torch.zeros(N, dtype=DT)) for l0 in L0S}
so = torch.zeros(N, dtype=DT); sso = torch.zeros(N, dtype=DT); norms = []
nt = 0
g = torch.Generator().manual_seed(9_000_011 + SEED)
for it in range(1 << (LOG2 - 16)):
    x = torch.randn(CH, N, generator=g, dtype=DT)
    out, kept = net.forward(x, gen=g, keep=set(L0S))
    so += out.sum(0); sso += out.pow(2).sum(0); norms.append(out.norm(dim=1).float())
    for l0 in L0S:
        h = kept[l0][0]
        r = h.norm(dim=1)
        b = torch.where(r > 0, torch.bucketize(torch.log(r.clamp_min(1e-300)), edges[l0]) + 1, torch.zeros_like(r, dtype=torch.long))
        A = acc[l0]
        A['w'] += torch.bincount(b, minlength=NB + 1).to(DT)
        A['s'].index_add_(0, b, h)
        A['ss'].index_add_(0, b, h[:, :, None] * h[:, None, :])
        A['m'] += h.sum(0)
    nt += CH
mean = so / nt; var = sso / nt - mean ** 2
norms = torch.cat(norms).double()
qs = {}
for q in (1e-2, 1e-3, 1e-4):
    t = float(torch.quantile(norms[:1 << 20], 1 - q))
    p = float((norms > t).double().mean())
    qs[str(q)] = dict(t=t, p=p, se=math.sqrt(p * (1 - p) / nt))
summ = dict(fam=FAM, n=N, L=L, sigma=SIGMA, seed=SEED, paths=nt, mean=mean.tolist(), var=var.tolist(), trvar=float(var.sum()),
            truth_se2_nmse=float(1.0 / nt), out_sq=float((sso / nt).sum()), tails=qs, L0S=L0S, r=net.r.tolist(),
            flops_path=net.flops_path(), sec=time.time() - t0)
np.savez(OUT + '_oracle.npz', **{f'w_{l0}': acc[l0]['w'].numpy() / nt for l0 in L0S},
         **{f's_{l0}': acc[l0]['s'].numpy() / nt for l0 in L0S}, **{f'ss_{l0}': acc[l0]['ss'].numpy() / nt for l0 in L0S},
         **{f'm_{l0}': acc[l0]['m'].numpy() / nt for l0 in L0S}, **{f'edges_{l0}': edges[l0].numpy() for l0 in L0S})
json.dump(summ, open(OUT + '.json', 'w'))
print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={SEED}: trVar={summ['trvar']:.4g} |mean|^2={float(mean.pow(2).sum()):.4g} "
      f"tails={ {k: round(v['t'], 5) for k, v in qs.items()} } ({time.time()-t0:.0f}s)", flush=True)
