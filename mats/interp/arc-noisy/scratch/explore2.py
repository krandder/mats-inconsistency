"""Oracle readout test: how much of the Gaussian closure's one-step error does a k-mode description remove?
Same samples for truth and approximations (coupled), so the comparison is free of most sampling noise."""
import sys, time, math, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet
torch.set_num_threads(2)
seed, sigma, m = int(sys.argv[1]), float(sys.argv[2]), 1 << int(sys.argv[3])
net = anet.Net(seed, sigma=sigma)
g = torch.Generator().manual_seed(11)
x = torch.randn(m, net.n, generator=g, dtype=torch.float64)
A = net.forward(x, g, keep=True)                       # a_l (pre-noise), l = 1..32
def stats(v):
    v = v - v.mean(0); s = v.std(0)
    return (v ** 3).mean(0) / s ** 3, (v ** 4).mean(0) / s ** 4 - 3
for l in (4, 8, 16, 24, 31):
    t0 = time.time()
    a = A[l - 1]                                      # a_l pre-noise; h_l = a_l + noise (noise tiny); next z = W h
    h = a + net.s[l - 1] * torch.randn(a.shape, generator=g, dtype=a.dtype) if sigma > 0 else a
    W = net.W[l]
    z = h @ W.T; y = torch.relu(z)                    # layer l+1
    truth = y.mean(0); var = y.var(0)
    mu = h.mean(0); C = torch.cov(h.T)
    ev, V = torch.linalg.eigh(C); ev = ev.flip(0); V = V.flip(1)
    Mz = W @ mu; sz = (torch.einsum('ij,jk,ik->i', W, C, W)).sqrt()
    gauss = anet.relu_gauss_mean(Mz, sz)
    def nm(e): return float(((e - truth) ** 2).mean() / var.mean())
    out = [f"layer {l}->{l+1}: var/q {var.mean() / y.pow(2).mean():.4f}  Gauss-readout nMSE {nm(gauss):.2e}"]
    c_all = (h - mu) @ V
    sk, ku = stats(c_all[:, :3])
    out.append(f"   top modes var share {(ev[:1].sum()/ev.sum()):.3f}/{(ev[:3].sum()/ev.sum()):.3f}/{(ev[:8].sum()/ev.sum()):.3f}; skew c1..3 {sk.numpy().round(2)} exkurt {ku.numpy().round(2)}")
    for k in (1, 2, 3, 5, 8, 16, 32):
        Vk = V[:, :k]; c = c_all[:, :k]
        Cr = C - Vk @ torch.diag(ev[:k]) @ Vk.T               # residual covariance (Gaussian part)
        sr = torch.einsum('ij,jk,ik->i', W, Cr, W).clamp_min(1e-30).sqrt()
        Zk = (mu + c @ Vk.T) @ W.T                          # (m, n): mean of z given the mode coefficients
        mix = anet.relu_gauss_mean(Zk, sr[None, :]).mean(0)
        out.append(f"   k={k:2d}: empirical modes + Gaussian residual nMSE {nm(mix):.2e}")
    print("\n".join(out), f"[{time.time()-t0:.0f}s]", flush=True)
print('MC floor of this coupled test ~', 1.0 / m)
