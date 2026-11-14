import sys, torch, time
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, kprop_noisy as KN
torch.set_num_threads(2)
for (n, L, sigma) in ((64, 3, 0.3), (128, 4, 0.1)):
    W = anet.make_weights(5, n, L)
    net = anet.Net(5, n=n, L=L, sigma=sigma, W=W, r=anet.pilot_rms(W, sigma, m=1<<14))
    g = torch.Generator().manual_seed(1)
    S = torch.zeros(n, dtype=torch.float64); S2 = torch.zeros(n, dtype=torch.float64); N = 0
    for b in range(16):
        a = net.forward(torch.randn(1 << 16, n, generator=g, dtype=torch.float64), g)
        S += a.sum(0); S2 += (a * a).sum(0); N += a.shape[0]
    mu = S / N; var = S2 / N - mu * mu
    print(f"n={n} L={L} sigma={sigma}: MC floor {1/N:.1e}")
    for k in (1, 2, 3):
        for noise in (True, False):
            est, fl, sec = KN.kprop_noisy(net, L, k, factor=(k == 3), noise=noise)
            print(f"  K={k} noise_handled={noise}: nMSE {((est - mu) ** 2).mean() / var.mean():.3e}  flops {fl:.3e}  {sec:.1f}s")
