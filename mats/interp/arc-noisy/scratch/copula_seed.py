"""Seed oracle with per-neuron marginals: h_m,i = F_i^{-1}(Phi(g_i)), g ~ N(0, R), F_i the empirical marginal of h_m,i and
R the normal-score correlation, both from 2^lgA true paths; then the real layers m+1..32 on 2^lgB seeded paths.
Compare with the Gaussian seed (same true mean/cov) and the truth. Shows whether per-neuron marginals + Gaussian
dependence (the neuron/copula code) is a sufficient family at depth."""
import sys, time, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib
torch.set_num_threads(2)
seed, sigma, m, lgA, lgB = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); L = 32
g = torch.Generator().manual_seed(77)
NA = 1 << lgA
H = []
for b in range(0, NA, 1 << 14):
    a = net.forward(torch.randn(1 << 14, 256, generator=g, dtype=torch.float32), g, L=m, dtype=torch.float32)
    H.append(a + float(net.s[m - 1]) * torch.randn(a.shape, generator=g, dtype=torch.float32))
H = torch.cat(H)                                            # (NA, 256) true samples of h_m
Hs, _ = torch.sort(H, 0)                                    # empirical quantile tables per neuron
ranks = torch.argsort(torch.argsort(H, 0), 0).float()
Z = torch.special.ndtri((ranks + 0.5) / NA)                  # normal scores
R = torch.corrcoef(Z.T.double())
ev, V = torch.linalg.eigh(R); Rh = (V * ev.clamp_min(0).sqrt()) @ V.T
Wf = [w.float() for w in net.W]
NB = 1 << lgB; S = torch.zeros(256, dtype=torch.float64); done = 0; t0 = time.time()
while done < NB:
    bsz = min(1 << 14, NB - done)
    u = torch.special.ndtr(torch.randn(bsz, 256, generator=g, dtype=torch.float64) @ Rh)
    idx = (u * NA).long().clamp(0, NA - 1)
    h = torch.gather(Hs, 0, idx)                            # per-neuron empirical quantiles
    for l in range(m, L):
        a = torch.relu(h @ Wf[l].T)
        if l < L - 1:
            h = a + float(net.s[l]) * torch.randn(a.shape, generator=g, dtype=torch.float32)
    S += a.double().sum(0); done += bsz
est = S / NB
print(f"seed {seed} m={m}: copula seed (true marginals from 2^{lgA}, Gaussian dependence) nMSE@32 {truthlib.nmse(est, T, L):.2e} "
      f"(sampling floor ~{1 / NB:.1e}; Gaussian seed with exact moments gave 2.1e-4 at m=24, 5.0e-4 at 16, 2.0e-4 at 8) [{time.time() - t0:.0f}s]", flush=True)
