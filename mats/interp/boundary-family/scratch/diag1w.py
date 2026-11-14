"""Diagnostic 1 (dev nets, width 64): where is the non-Gaussianity that one ReLU step needs?
For cut layers l: true pre-activation samples z_l. One-step moment error of a = ReLU(z_l):
  G   : Gaussian with the true mean/cov of z_l (what a Gaussian closure computes when its input moments are exact)
  Ck  : cells = quantile bins of the top-k PCA latent of z_l (k = 1, 2, 3; 4^k cells), Gaussian per cell with the true
        within-cell mean/cov (an oracle Gaussian mixture of the latent-cell family)
  Sk  : same cells, but the true per-neuron marginals within each cell (empirical) for the mean only
Errors: mean of a (nMSE in units of mean Var(a)) and covariance of a (relative Frobenius).
Also per-neuron standardized skewness / excess kurtosis of z_l, overall and within cells.
python scratch/diag1.py SEED L CUTS NLOG2"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf
from gauss_relu import relu_moments

seed, L = int(sys.argv[1]), int(sys.argv[2])
cuts = [int(v) for v in sys.argv[3].split(',')]
N = 1 << int(sys.argv[4])
WID = int(os.environ.get("WID", "64"))
net = bf.get_net(seed, WID, 32, 0.001)
t0 = time.time()
smp = bf.mc_stats(net, N, seed=30_000 + seed, L=L, cuts=tuple(cuts), keep_n=N, out_moments=False)
print(f'samples {time.time() - t0:.0f}s', flush=True)
out = {}
for l in cuts:
    Z = smp['cuts'][l].double()
    A = torch.relu(Z)
    ma = A.mean(0); Ca = torch.cov(A.T); va = float(Ca.diagonal().mean())
    mz = Z.mean(0); Cz = torch.cov(Z.T)
    sd = Cz.diagonal().sqrt()
    Zs = (Z - mz) / sd
    sk = (Zs ** 3).mean(0); ku = (Zs ** 4).mean(0) - 3
    r = dict(t_abs_med=float((mz / sd).abs().median()), frac_t_lt1=float(((mz / sd).abs() < 1).float().mean()),
             skew_med_abs=float(sk.abs().median()), exkurt_med=float(ku.median()))
    ev = torch.linalg.eigvalsh(Cz).flip(0)
    r['pr_z'] = float(ev.sum() ** 2 / (ev ** 2).sum())
    r['top_ev_frac'] = [float(v) for v in (ev[:6] / ev.sum())]
    mg, Cg, _ = relu_moments(mz, Cz)
    r['G_mean'] = float(((mg - ma) ** 2).mean()) / va
    r['G_cov'] = float((Cg - Ca).norm() / Ca.norm())
    evals, U = torch.linalg.eigh(Cz); U = U.flip(1)
    for k in (1, 2, 3):
        Y = (Z - mz) @ U[:, :k]
        nb = 4 if k > 1 else 16
        lab = torch.zeros(Z.shape[0], dtype=torch.long)
        for j in range(k):
            qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
            lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
        mm = torch.zeros(Z.shape[1], dtype=torch.float64); E2 = torch.zeros_like(Cz)
        me = torch.zeros_like(mm)
        sk_in = []; ku_in = []
        for c in torch.unique(lab):
            S = lab == c; w = float(S.double().mean())
            Zc = Z[S]
            m_c = Zc.mean(0); C_c = torch.cov(Zc.T)
            mu_c, cov_c, _ = relu_moments(m_c, C_c)
            mm += w * mu_c; E2 += w * (cov_c + torch.outer(mu_c, mu_c))
            me += w * torch.relu(Zc).mean(0)
            s_c = C_c.diagonal().sqrt(); Zcs = (Zc - m_c) / s_c
            sk_in.append(w * (Zcs ** 3).mean(0).abs()); ku_in.append(w * ((Zcs ** 4).mean(0) - 3))
        Cm = E2 - torch.outer(mm, mm)
        r[f'C{k}_mean'] = float(((mm - ma) ** 2).mean()) / va
        r[f'C{k}_cov'] = float((Cm - Ca).norm() / Ca.norm())
        r[f'C{k}_skew_in_med'] = float(torch.stack(sk_in).sum(0).median())
        r[f'C{k}_exkurt_in_med'] = float(torch.stack(ku_in).sum(0).median())
        r[f'S{k}_mean_check'] = float(((me - ma) ** 2).mean()) / va
    out[l] = r
    print(l, json.dumps({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
os.makedirs('runs/diag1', exist_ok=True)
json.dump(out, open(f'runs/diag1/diag1_{seed}_w{WID}_L{L}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
