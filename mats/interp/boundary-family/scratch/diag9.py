"""Diagnostic 9 (tails, representation ceiling): can a Gaussian-cell family at the last hidden layer represent the
output tails at all? Dev net with ARC part-2 splitting truths (net 102-105, depth 32, sigma = 0.001, outputs 0-7, five
thresholds per output at p ~ 1e-4 ... 1e-12).
Fit on 2^NLOG2 Monte Carlo paths of h_31 (post-activation + noise): G (one Gaussian), PCA1 (16 bins), PCA3 (64 cells),
Gaussian per cell. Then P(z_32,i > t) = sum_c pi_c Phi-bar((t - w_i . m_c) / sqrt(w_i S_c w_i)) exactly; error
|log(est / truth)| per event and the fraction within 50%.
python scratch/diag9.py SEED NLOG2"""
import json, math, os, sys, time
import numpy as np
import torch
from scipy.special import log_ndtr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf
from common import load_truth

seed = int(sys.argv[1]); N = 1 << int(sys.argv[2])
net = bf.get_net(seed, 256, 32, 0.001)
T = load_truth(f'/home/agent/workspaces/mats/work/arc-noisy/tails/runs/arcn-tdev1/truth/truth_{seed}_s0.001.json')[32]
thr = T['thr']; lp = T['lp']                       # (8, 5) thresholds, log truths
t0 = time.time()
smp = bf.mc_stats(net, N, seed=120_000 + seed, L=31, cuts=(31,), keep_n=N, out_moments=False)
Z = smp['cuts'][31].double()
g = torch.Generator().manual_seed(1)
H = torch.relu(Z) + float(net.s[30]) * torch.randn(Z.shape, generator=g, dtype=torch.float64)
Wo = net.W[31].double()[:8]                           # outputs 0-7
print(f'samples {time.time() - t0:.0f}s', flush=True)
mu = H.mean(0); C = torch.cov(H.T); ev, U = torch.linalg.eigh(C); U = U.flip(1)


def mix_tail(cells):
    out = np.zeros((8, 5))
    for b in range(8):
        w = Wo[b]
        terms = []
        for p, m, S in cells:
            mm = float(w @ m); sd = math.sqrt(float(w @ S @ w))
            terms.append(np.log(p) + log_ndtr(-(thr[b] - mm) / sd))
        out[b] = np.logaddexp.reduce(np.array(terms), axis=0)
    return out


def cells_pca(k):
    Y = (H - mu) @ U[:, :k]; nb = 16 if k == 1 else 4
    lab = torch.zeros(N, dtype=torch.long)
    for j in range(k):
        qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
        lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
    return [(float((lab == c).double().mean()), H[lab == c].mean(0), torch.cov(H[lab == c].T)) for c in torch.unique(lab)]


res = {}
# empirical check: the fit sample's own tail where it has hits (1e-4 level only)
zo = H @ Wo.T
emp = np.array([[float((zo[:, b] > thr[b, k]).double().mean()) for k in range(5)] for b in range(8)])
for name, cells in (('G', [(1.0, mu, C)]), ('PCA1', cells_pca(1)), ('PCA3', cells_pca(3))):
    est = mix_tail(cells)
    le = np.abs(est - lp)
    within = np.abs(np.exp(est - lp) - 1) <= 0.5
    res[name] = dict(median_abs_log_err=[float(np.median(le[:, k])) for k in range(5)],
                     frac_within50=[float(within[:, k].mean()) for k in range(5)],
                     median_log10_ratio=[float(np.median((est - lp)[:, k]) / math.log(10)) for k in range(5)])
    print(name, json.dumps(res[name]), flush=True)
res['emp_1e-4_within50'] = float((np.abs(emp[:, 0] / np.exp(lp[:, 0]) - 1) <= 0.5).mean())
os.makedirs('runs/diag9', exist_ok=True)
json.dump(res, open(f'runs/diag9/diag9_{seed}.json', 'w'), indent=1)
print('done', f'{time.time() - t0:.0f}s', flush=True)
