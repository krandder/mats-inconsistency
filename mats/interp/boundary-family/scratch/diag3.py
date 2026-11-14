"""Diagnostic 3: does one deductive push keep the accuracy of latent cells?
At layer l: cells = quantile bins of the top-k PCA modes of the true z_l (4^k cells; 16 for k = 1), Gaussian per cell
with the true within-cell moments (oracle). Push every cell through layer l+1 with the per-cell exact Gaussian closure
(rectified moments, bivariate-normal covariances, noise), then read out E[a_{l+1}] from the pushed mixture.
Compared with: G (single Gaussian with the true moments of z_l, pushed), the oracle cells refitted at l+1 (C_k at l+1),
and the plain one-step readout at l+1 with the true moments (G at l+1).
python scratch/diag3.py SEED L CUTS NLOG2   (env WID = width)"""
import json, math, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf
from gauss_relu import relu_moments, ncdf, npdf

seed, L = int(sys.argv[1]), int(sys.argv[2])
cuts = [int(v) for v in sys.argv[3].split(',')]
N = 1 << int(sys.argv[4])
WID = int(os.environ.get('WID', '64'))
net = bf.get_net(seed, WID, 32, 0.001)
allc = sorted(set(cuts + [c + 1 for c in cuts]))
t0 = time.time()
smp = bf.mc_stats(net, N, seed=50_000 + seed, L=max(allc), cuts=tuple(allc), keep_n=N, out_moments=False)
I = torch.eye(WID, dtype=torch.float64)


def relu_mean(m, s):
    t = m / s
    return s * (t * ncdf(t) + npdf(t))


def push(m, C, l):
    """z_{l+1} | cell from z_l | cell ~ N(m, C); W index l (0-based layer l+1)."""
    mu, Ca, _ = relu_moments(m, C)
    W = net.W[l].double()
    s2 = float(net.s[l - 1]) ** 2
    return W @ mu, W @ (Ca + s2 * I) @ W.T


def cells(Z, k):
    mz = Z.mean(0); _, U = torch.linalg.eigh(torch.cov(Z.T)); U = U.flip(1)
    Y = (Z - mz) @ U[:, :k]
    nb = 16 if k == 1 else 4
    lab = torch.zeros(Z.shape[0], dtype=torch.long)
    for j in range(k):
        qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
        lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
    out = []
    for c in torch.unique(lab):
        S = lab == c
        out.append((float(S.double().mean()), Z[S].mean(0), torch.cov(Z[S].T)))
    return out


def readout(mix):
    m = 0
    for w, mm, C in mix:
        m = m + w * relu_mean(mm, C.diagonal().clamp_min(1e-300).sqrt())
    return m


res = {}
for l in cuts:
    Z = smp['cuts'][l].double(); Z1 = smp['cuts'][l + 1].double()
    ma1 = torch.relu(Z1).mean(0); va1 = float(torch.relu(Z1).var(0).mean())
    e = lambda est: float(((est - ma1) ** 2).mean()) / va1
    r = {}
    r['G_at_l1'] = e(readout([(1.0, Z1.mean(0), torch.cov(Z1.T))]))
    m0, C0 = push(Z.mean(0), torch.cov(Z.T), l)
    r['G_pushed'] = e(readout([(1.0, m0, C0)]))
    for k in (1, 2, 3):
        mix = cells(Z, k)
        pushed = [(w, *push(mm, C, l)) for w, mm, C in mix]
        r[f'C{k}_pushed'] = e(readout(pushed))
        r[f'C{k}_at_l1'] = e(readout(cells(Z1, k)))
    res[l] = r
    print(l, json.dumps({kk: float(f'{v:.4g}') for kk, v in r.items()}), flush=True)
os.makedirs('runs/diag3', exist_ok=True)
json.dump(res, open(f'runs/diag3/diag3_{seed}_w{WID}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
