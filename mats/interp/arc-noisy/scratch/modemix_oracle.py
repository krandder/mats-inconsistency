"""Readout 31 -> 32 with TRUE mean/cov of h_31 (truth M2_31) and the empirical law of the top-k mode coefficients
(N fresh samples); residual Gaussian integrated exactly. Scored against the 2^26 truth of E[a_32]."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, gauss_relu as GR
torch.set_num_threads(2)
seed, sigma, lg = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); d = T['d']; Nt = T['N']
mu31 = torch.tensor(d['S1'][30] / Nt); C31 = torch.tensor(d['M2_31'] / Nt) - torch.outer(mu31, mu31)
C31 = C31 + float(net.s[30]) ** 2 * torch.eye(256, dtype=torch.float64)          # h_31 = a_31 + noise
W = net.W[31]; truth = T['mean'][31]; v32 = T['var'][31].mean()
ev, V = torch.linalg.eigh(C31); ev = ev.flip(0); V = V.flip(1)
Mz = W @ mu31
gauss = GR.relu_mean(Mz, torch.einsum('ij,jk,ik->i', W, C31, W).sqrt())
print(f"Gaussian readout (true moments): nMSE {((gauss.numpy() - truth) ** 2).mean() / v32:.3e}")
g = torch.Generator().manual_seed(5)
for rep in range(2):
    N = 1 << lg
    A = net.forward(torch.randn(N, 256, generator=g, dtype=torch.float32), g, L=31, dtype=torch.float32).double()
    h = A + float(net.s[30]) * torch.randn(A.shape, generator=g, dtype=torch.float64)
    c_all = (h - mu31) @ V
    for k in (1, 2, 3, 5, 8, 16):
        Vk = V[:, :k]; Cr = C31 - Vk @ torch.diag(ev[:k]) @ Vk.T
        sr = torch.einsum('ij,jk,ik->i', W, Cr, W).clamp_min(1e-30).sqrt()
        WV = W @ Vk
        est = torch.zeros(256, dtype=torch.float64)
        for b in range(0, N, 1 << 14):
            Z = Mz[None, :] + c_all[b:b + (1 << 14), :k] @ WV.T
            est += GR.relu_mean(Z, sr[None, :]).sum(0)
        est /= N
        # noise floor of this estimator: variance of the per-sample readout / N
        print(f"rep {rep} N=2^{lg} k={k:2d}: nMSE {((est.numpy() - truth) ** 2).mean() / v32:.3e}", flush=True)
    print(f"   plain MC on these samples: {((torch.relu(h @ W.T).mean(0).numpy() - truth) ** 2).mean() / v32:.3e}  (1/N = {1/N:.1e})")
