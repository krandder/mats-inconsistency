"""Diagnostic 4 (width 256, ARC dev nets, ARC 2^25-path truths): best case for a propagated cell family.
At layer m: oracle cells from true samples (quantile bins of the top-k PCA modes of z_m, true within-cell moments),
then every cell is pushed with its own exact Gaussian closure to layer L (no further splitting). Compared with the
single Gaussian with the true moments at m pushed the same way, and the Gaussian closure from layer 1.
python scratch/diag4.py SEED CUTS NLOG2 [K_LIST]"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf, gmprop

seed = int(sys.argv[1]); cuts = [int(v) for v in sys.argv[2].split(',')]; N = 1 << int(sys.argv[3])
KS = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [1, 2, 3]
L = 32; n = 256
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
net = bf.get_net(seed, n, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); NT = float(d['N'])
TM = torch.tensor(d['S1'] / NT); TV = torch.tensor(d['S2'] / NT) - TM ** 2
err = lambda e, l: float(((e - TM[l - 1]) ** 2).mean() / TV[l - 1].mean()) - 1 / NT
t0 = time.time()
smp = bf.mc_stats(net, N, seed=60_000 + seed, L=max(cuts), cuts=tuple(cuts), keep_n=N, out_moments=False)
print(f'samples {time.time() - t0:.0f}s', flush=True)
W = [w.double() for w in net.W]; s = [float(v) for v in net.s]


def propagate(cells, m):
    """cells at layer m (pre-activation) -> per-layer estimates of E[a_l], l = m..L."""
    out = {m: gmprop.readout_mean(cells)}
    for l in range(m, L):
        cells = [gmprop.push(c, W[l], s[l - 1] ** 2)[0] for c in cells]
        out[l + 1] = gmprop.readout_mean(cells)
    return out


res = {}
for m in cuts:
    Z = smp['cuts'][m].double()
    r = {}
    est = propagate([gmprop.Cell(1.0, Z.mean(0), torch.cov(Z.T))], m)
    r['G'] = {l: err(e, l) for l, e in est.items()}
    mz = Z.mean(0); _, U = torch.linalg.eigh(torch.cov(Z.T)); U = U.flip(1)
    for k in KS:
        Y = (Z - mz) @ U[:, :k]
        nb = 16 if k == 1 else 4
        lab = torch.zeros(Z.shape[0], dtype=torch.long)
        for j in range(k):
            qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
            lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
        cells = []
        for c in torch.unique(lab):
            S = lab == c
            cells.append(gmprop.Cell(float(S.double().mean()), Z[S].mean(0), torch.cov(Z[S].T)))
        est = propagate(cells, m)
        r[f'C{k}'] = {l: err(e, l) for l, e in est.items()}
        print(m, k, f'{time.time() - t0:.0f}s', flush=True)
    res[m] = r
    print(m, ' '.join(f'{name}: ' + ' '.join(f'{l}:{v:.2e}' for l, v in sorted(rr.items()) if l in (m, m + 1, 16, 24, 28, 32))
                      for name, rr in r.items()), flush=True)
os.makedirs('runs/diag4', exist_ok=True)
json.dump(res, open(f'runs/diag4/diag4_{seed}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
