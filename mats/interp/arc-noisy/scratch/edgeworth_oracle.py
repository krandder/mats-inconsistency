"""Readout l -> l+1: true mean/cov of h_l (truth) + per-neuron gamma3, gamma4 of z_{l+1} from N samples (Edgeworth),
or the same cumulants reconstructed from the top-k mode coefficients' joint cumulants (low-rank model)."""
import sys, math, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, gauss_relu as GR
torch.set_num_threads(2)
seed, sigma, lg, ell = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); dd = T['d']; Nt = T['N']
I = torch.eye(256, dtype=torch.float64)
mu = torch.tensor(dd['S1'][ell - 1] / Nt)
C = torch.tensor(dd[f'M2_{ell}'] / Nt) - torch.outer(mu, mu) + float(net.s[ell - 1]) ** 2 * I
W = net.W[ell]; truth = T['mean'][ell]; v = T['var'][ell].mean()
Mz = W @ mu; s = torch.einsum('ij,jk,ik->i', W, C, W).sqrt(); t = Mz / s


def edge(g3, g4):
    ph = GR.npdf(t)
    return s * (t * GR.ncdf(t) + ph + ph * (-(g3 / 6) * t + (g4 / 24) * (t * t - 1) + (g3 * g3 / 72) * (t ** 4 - 6 * t * t + 3)))


def nm(e):
    return float(((e.numpy() - truth) ** 2).mean() / v)


print(f"layer {ell}->{ell + 1}  Gaussian readout {nm(GR.relu_mean(Mz, s)):.2e}", flush=True)
g = torch.Generator().manual_seed(9); N = 1 << lg
S2 = torch.zeros(256, dtype=torch.float64); S3 = torch.zeros_like(S2); S4 = torch.zeros_like(S2)
ev, V = torch.linalg.eigh(C); V = V.flip(1)
K = 16
c2 = torch.zeros(K, K, dtype=torch.float64); c3 = torch.zeros(K, K, K, dtype=torch.float64)
c4 = torch.zeros(K, K, K, K, dtype=torch.float64)
done = 0
while done < N:
    m = min(1 << 14, N - done)
    a = net.forward(torch.randn(m, 256, generator=g, dtype=torch.float32), g, L=ell, dtype=torch.float32).double()
    h = a + float(net.s[ell - 1]) * torch.randn(a.shape, generator=g, dtype=torch.float64)
    z = (h - mu) @ W.T
    S2 += (z ** 2).sum(0); S3 += (z ** 3).sum(0); S4 += (z ** 4).sum(0)
    cc = (h - mu) @ V[:, :K]
    c2 += cc.T @ cc
    c3 += torch.einsum('na,nb,nc->abc', cc, cc, cc)
    c4 += torch.einsum('na,nb,nc,nd->abcd', cc, cc, cc, cc)
    done += m
m2 = S2 / N; m3 = S3 / N; m4 = S4 / N
g3 = m3 / s ** 3; g4 = (m4 - 3 * m2 ** 2) / s ** 4
print(f"  Edgeworth, sampled per-neuron cumulants (N=2^{lg}): k3 only {nm(edge(g3, 0 * g4)):.2e}   k3+k4 {nm(edge(g3, g4)):.2e}", flush=True)
c2 /= N; c3 /= N; c4 /= N
k4m = c4 - (torch.einsum('ab,cd->abcd', c2, c2) + torch.einsum('ac,bd->abcd', c2, c2) + torch.einsum('ad,bc->abcd', c2, c2))
WV = W @ V[:, :K]
for k in (1, 2, 4, 8, 16):
    A = WV[:, :k]
    k3z = torch.einsum('ia,ib,ic,abc->i', A, A, A, c3[:k, :k, :k])
    k4z = torch.einsum('ia,ib,ic,id,abcd->i', A, A, A, A, k4m[:k, :k, :k, :k])
    print(f"  Edgeworth, cumulants from top-{k:2d} modes: k3 only {nm(edge(k3z / s ** 3, 0 * k3z)):.2e}   "
          f"k3+k4 {nm(edge(k3z / s ** 3, k4z / s ** 4)):.2e}", flush=True)
