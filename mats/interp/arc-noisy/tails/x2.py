"""Exploration X2 (development nets): how much the noise can do for layer-wise twisting. For an event A at p ~ 1e-8,
r = P(A and A') / P(A), where A' is the same event with the same input and independent noise. Any estimator that samples
x from its prior and only reweights or twists the noise has relative variance >= r / p - 1, so its gain over plain
Monte Carlo is at most ~1 / r. python tails/x2.py SEED L X1JSON OUTDIR [NNEUR] [LEVEL_INDEX]"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seed, L, x1p, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
NN = int(sys.argv[5]) if len(sys.argv) > 5 else 8
LI = int(sys.argv[6]) if len(sys.argv) > 6 else 2                 # 0..4 -> 1e-4 .. 1e-12
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
x1 = json.load(open(x1p))
T_ = x1['thr'][str(L)] if isinstance(x1['thr'], dict) else x1['thr']
thr = [T_[i][LI] for i in range(NN)]
res = dict(seed=seed, L=L, level_index=LI, thr=thr, by_sigma={})
t0 = time.time()
for sigma in (0.001, 0.003, 0.01):
    net = TNet(seed, L, sigma, device=dev)
    gen = torch.Generator(device=dev).manual_seed(30_000 + 97 * seed + L + int(1e4 * sigma))
    idx = torch.arange(NN, device=dev)

    W = net.W[-1][idx]

    def single(Z, W=W):
        B, N = Z.shape[:2]
        h = net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1)
        return (h * W[:, None, :]).sum(-1)

    def coupled(Z, W=W):
        B, N = Z.shape[:2]
        Za = Z[:, :, :L]; Zb = torch.cat([Z[:, :, :1], Z[:, :, L:]], 2)
        return torch.minimum(single(Za.contiguous()), single(Zb.contiguous()))

    out = {}
    for name, sf, LL in (('single', single, L), ('coupled', coupled, 2 * L - 1)):
        net.latent_L = LL                                          # latent length used by bgs.draw
        p = bgs.pilot(net, sf, NN, 1000, gen, [[t] for t in thr])
        Lv, Bs = bgs.pad(p['levels'], p['bs'], 5)
        R = 8
        logs = []
        for r in range(R):
            o = bgs.frozen(net, sf, 1000, gen, Lv, Bs, n_moves=5)
            logs.append(o['logp'][:, -1].numpy())
        net.latent_L = None
        out[name] = np.array(logs).T.tolist()                      # (NN, R)
    res['by_sigma'][str(sigma)] = out
    print(sigma, f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f'x2_{seed}_L{L}.json'), 'w'))
print('done')
