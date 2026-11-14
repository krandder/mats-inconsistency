"""Meet-in-the-middle oracle: draw h_m ~ N(mu_m, Sigma_m) with the TRUE moments (truth M2_m), push through the real
layers m+1..L, compare the mean with the truth. Bias of 'Gaussian description up to m, exact sampling after'."""
import sys, time, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib
torch.set_num_threads(2)
seed, sigma, m, lg = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); dd = T['d']; Nt = T['N']; L = 32
mu = torch.tensor(dd['S1'][m - 1] / Nt); C = torch.tensor(dd[f'M2_{m}'] / Nt) - torch.outer(mu, mu)
C = C + float(net.s[m - 1]) ** 2 * torch.eye(256, dtype=torch.float64)
ev, V = torch.linalg.eigh(C); R = (V * ev.clamp_min(0).sqrt()) @ V.T
g = torch.Generator().manual_seed(31); N = 1 << lg
S = torch.zeros(256, dtype=torch.float64); S2 = torch.zeros(256, dtype=torch.float64); done = 0; t0 = time.time()
Wf = [w.float() for w in net.W]
while done < N:
    b = min(1 << 14, N - done)
    h = (mu + torch.randn(b, 256, generator=g, dtype=torch.float64) @ R).float()
    for l in range(m, L):
        a = torch.relu(h @ Wf[l].T)
        if l < L - 1:
            h = a + float(net.s[l]) * torch.randn(a.shape, generator=g, dtype=torch.float32)
    S += a.double().sum(0); S2 += (a.double() ** 2).sum(0); done += b
est = S / N; v_seed = (S2 / N - est ** 2).mean()
e = truthlib.nmse(est, T, L)
print(f"seed {seed} m={m}: Gaussian-seeded (true moments) nMSE@32 {e:.2e}  (sampling floor ~{float(v_seed / T['var'][31].mean()) / N:.1e}, "
      f"seeded var / true var {float(v_seed / T['var'][31].mean()):.3f})  [{time.time() - t0:.0f}s]", flush=True)
