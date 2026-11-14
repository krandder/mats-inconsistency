"""U4 kill test (PREREG-TAILS section 5): importance sampling over x from a Gaussian fitted to splitting particles in
the event (L = 4, k = 8 by default). Mean + top-k covariance directions (k = 0, 3, 10; eigenvalues clamped at >= 1, unit
variance elsewhere); noise from its prior. Relative variance per importance sample, against the truth reference.
python tails/x4.py SEED THRDIR TRUTHDIR OUTDIR [L] [KIDX]"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seed, THR, TRU, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
L = int(sys.argv[5]) if len(sys.argv) > 5 else 4
KI = int(sys.argv[6]) if len(sys.argv) > 6 else 2
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
thr = json.load(open(os.path.join(THR, f'thr_{seed}.json')))['thr'][str(L)]
tru = json.load(open(os.path.join(TRU, f'truth_{seed}_s0.001.json')))['by_L'][str(L)]
NN = len(thr)
net = TNet(seed, L, 0.001, device=dev)
gen = torch.Generator(device=dev).manual_seed(90_000 + 97 * seed + L)
res = dict(seed=seed, L=L, k_index=KI, events=[])
for i in range(NN):
    t = thr[i][KI]
    v = np.array(tru['logp'][i][KI]); m = v.max(); lref = math.log(np.exp(v - m).mean()) + m
    idx = torch.tensor([i], device=dev)
    sf = bgs.neuron_scores(net, idx)
    p = bgs.pilot(net, sf, 1, 1000, gen, [[t]])
    Lv, Bs = bgs.pad(p['levels'], p['bs'], 5)
    # one frozen pass with N = 4000, keeping the particles in the event at the end
    lev = Lv.to(dev); N = 4000
    Z = bgs.draw(net, 1, N, gen); Y = sf(Z)
    for k in range(lev.shape[1]):
        g = lev[:, k]
        Z, Y = bgs._select(Z, Y, Y.double() > g[:, None], gen)
        Z, Y, _ = bgs._moves(net, sf, Z, Y, g, gen, 10, bs_k=torch.cat([Bs[:, min(k, Bs.shape[1] - 1)]] * 2, 1).to(dev))
    X = Z[0, :, 0].double()
    mu = X.mean(0); C = torch.cov(X.T)
    evl, evec = torch.linalg.eigh(C); evl, evec = evl.flip(0), evec.flip(1)
    row = dict(i=i, lref=lref, m_norm=float(mu.norm()), top_eig=evl[:5].tolist(), by_k={})
    for kk in (0, 3, 10):
        lam = torch.clamp(evl[:kk], min=1.0); Vk = evec[:, :kk]
        M = 40000
        G = torch.randn(M, net.n, generator=gen, device=dev, dtype=torch.float64)
        x = mu + G + ((G @ Vk) * (lam.sqrt() - 1)) @ Vk.T
        d = x - mu; dv = d @ Vk
        logw = -0.5 * (x * x).sum(1) + 0.5 * ((d * d).sum(1) + ((dv ** 2) * (1 / lam - 1)).sum(1)) + 0.5 * torch.log(lam).sum()
        Zs = bgs.draw(net, 1, M, gen); Zs[0, :, 0] = x.to(Zs.dtype)
        hit = (sf(Zs)[0] > t).double()
        w = torch.exp(logw - lref) * hit                           # ratio to the reference p
        row['by_k'][str(kk)] = dict(est_over_ref=float(w.mean()), relvar_vs_ref=float((w ** 2).mean() - 1),
                                    hit_rate=float(hit.mean()), ess_frac=float(w.sum() ** 2 / max(float((w ** 2).sum()), 1e-300) / M))
    res['events'].append(row)
    print(i, {k: round(v['relvar_vs_ref'], 1) for k, v in row['by_k'].items()}, flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f'x4_{seed}_L{L}.json'), 'w'))
print('done')
