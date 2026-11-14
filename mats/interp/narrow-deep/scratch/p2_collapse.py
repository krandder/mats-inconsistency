"""Part 2 (PREREG P2, P4): collapse at initialisation, per layer.
usage: python p2_collapse.py FAM N L SIGMA SEEDS(comma) OUTJSON
Per layer l: dead fraction (no active unit), mean active fraction, pairwise angle between two inputs' states
(pairs 2i, 2i+1, independent noise), gate disagreement between two inputs and between two noise draws of one
input, participation ratio of Cov(h_l), top-PC share of E[h h^T], mean/sd of log|h_l|. Per path: min_l m_l.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]
OUT = sys.argv[6]
P = 4096; PN = 1024


def pr(C):
    ev = torch.linalg.eigvalsh(C).clamp_min(0)
    return float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-300))


res = {}
for seed in SEEDS:
    t0 = time.time()
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(5_000_011 + seed)
    x = torch.randn(P, N, generator=g, dtype=nd.DT)
    xn = torch.randn(PN, N, generator=g, dtype=nd.DT).repeat_interleave(2, 0)   # noise-only pairs
    X = torch.cat([x, xn], 0)
    h = X.clone()
    ever_dead = torch.zeros(X.shape[0], dtype=torch.bool)
    minm = torch.full((X.shape[0],), N, dtype=torch.long)
    rows = []
    for l in range(L):
        z = h @ net.W[l].T
        gate = z > 0
        m = gate.sum(1)
        minm = torch.minimum(minm, m)
        dead = m == 0
        ever_dead |= dead
        a = torch.relu(z)
        e = torch.randn(a.shape, generator=g, dtype=nd.DT) * net.s[l] if SIGMA > 0 else 0.0
        if FAM == 'plain':
            h = a + e
        else:
            h = h + ((a + e) @ net.V[l].T) * net.c
        hp = h[:P]
        alive = ~ever_dead[:P] if FAM == 'plain' else torch.ones(P, dtype=torch.bool)
        A, B = hp[0::2], hp[1::2]
        ok = alive[0::2] & alive[1::2]
        cos = (A * B).sum(1) / (A.norm(dim=1) * B.norm(dim=1) + 1e-300)
        th = torch.arccos(cos.clamp(-1, 1))[ok]
        gd = (gate[:P][0::2] != gate[:P][1::2]).double().mean(1)[ok]
        gn = (gate[P:][0::2] != gate[P:][1::2]).double().mean(1)
        oka = alive
        hc = hp[oka]
        ln = torch.log(hc.norm(dim=1) + 1e-300)
        C = torch.cov(hc.T) if hc.shape[0] > 2 else torch.zeros(N, N, dtype=nd.DT)
        S2 = hc.T @ hc / max(hc.shape[0], 1)
        ev2 = torch.linalg.eigvalsh(S2)
        rows.append(dict(l=l + 1, dead_now=float(dead[:P].double().mean()), ever_dead=float(ever_dead[:P].double().mean()),
                         act=float(m[:P].double().mean() / N),
                         th_mean=float(th.mean()) if len(th) else None, th_med=float(th.median()) if len(th) else None,
                         th_q90=float(th.quantile(0.9)) if len(th) else None,
                         logth_mean=float(torch.log(th.clamp_min(1e-300)).mean()) if len(th) else None,
                         gd_pair=float(gd.mean()) if len(gd) else None, gd_noise=float(gn.mean()),
                         pr_cov=pr(C), top_share=float(ev2[-1] / (ev2.sum() + 1e-300)),
                         logn_mean=float(ln.mean()) if len(ln) else None, logn_sd=float(ln.std()) if len(ln) > 1 else None))
    md = minm[:P]
    res[seed] = dict(rows=rows, minm_le=[float((md <= k).double().mean()) for k in range(4)],
                     r=net.r.tolist(), sec=time.time() - t0)
    last = rows[-1]
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: ever_dead@64/128/{L}="
          f"{[round(rows[i-1]['ever_dead'],3) for i in (64,128,L) if i<=L]} th_med@16/64/128/{L}="
          f"{[None if rows[i-1]['th_med'] is None else round(rows[i-1]['th_med'],4) for i in (16,64,128,L) if i<=L]} "
          f"gd_pair@16/64/{L}={[None if rows[i-1]['gd_pair'] is None else round(rows[i-1]['gd_pair'],4) for i in (16,64,L) if i<=L]} "
          f"gd_noise@{L}={round(last['gd_noise'],4)} pr@{L}={round(last['pr_cov'],2)} top@{L}={round(last['top_share'],4)} "
          f"minm<=0..3={[round(v,3) for v in res[seed]['minm_le']]} ({time.time()-t0:.1f}s)", flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, seeds=SEEDS, P=P, res=res), open(OUT, 'w'))
