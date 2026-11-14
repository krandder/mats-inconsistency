"""Activation-region cells (Part 3, exact-representation question): how many joint gate patterns over layers 1..l do
the inputs occupy? usage: python p3_cells.py FAM N L SIGMA SEEDS OUTJSON
Per layer l: number of distinct joint patterns (G_1..G_l) among 2^20 inputs, plug-in entropy (bits), and the number
of patterns holding 50/90/99% of the inputs. A noiseless net is linear on each cell, so an exact cell representation
of the law at layer l needs (at least) the occupied cells and one polyhedral Gaussian integral per cell.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]; OUT = sys.argv[6]
P = 1 << 20
DT = nd.DT
LAY = [l for l in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if l <= L]
res = {}
for seed in SEEDS:
    t0 = time.time()
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(14_000_029 + seed)
    x = torch.randn(P, N, generator=g, dtype=DT)
    h = x
    key = torch.zeros(P, dtype=torch.int64)
    rng = np.random.RandomState(seed)
    rows = []
    pw = torch.tensor([1 << i for i in range(N)], dtype=torch.int64)
    for l in range(L):
        z = h @ net.W[l].T
        gate = (z > 0).to(torch.int64)
        code = (gate * pw).sum(1)
        # rolling 63-bit hash of the joint pattern
        key = (key * 1_000_003 + code * 7_919 + l) % (2 ** 61 - 1)
        a = torch.relu(z)
        e = torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIGMA > 0 else 0.0
        h = a + e if FAM == 'plain' else h + ((a + e) @ net.V[l].T) * net.c
        if (l + 1) in LAY:
            _, cnt = torch.unique(key, return_counts=True)
            c = cnt.double().sort(descending=True).values
            p = c / P
            H = float(-(p * torch.log2(p)).sum())
            cum = torch.cumsum(c, 0) / P
            cov = {str(q): int((cum < q).sum()) + 1 for q in (0.5, 0.9, 0.99)}
            rows.append(dict(l=l + 1, distinct=int(len(c)), H_bits=H, cover=cov, singletons=int((c == 1).sum())))
    res[seed] = rows
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: " + ' '.join(f"l{r['l']}:{r['distinct']}/{r['H_bits']:.1f}b/c99={r['cover']['0.99']}" for r in rows) +
          f" ({time.time()-t0:.0f}s)", flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, P=P, res=res), open(OUT, 'w'))
