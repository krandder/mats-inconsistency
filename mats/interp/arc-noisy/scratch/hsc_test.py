"""Homogeneous scale-mixture closure (HSC) vs exact Gaussian closure (GC), sample-free, one net."""
import sys, torch, numpy as np, json, os
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, gauss_relu as GR
torch.set_num_threads(2)
seed, sigma = int(sys.argv[1]), float(sys.argv[2])
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma)
I = torch.eye(256, dtype=torch.float64)
def gc(Lmax):
    mu = torch.zeros(256, dtype=torch.float64); Sig = I.clone(); out = []
    for l in range(Lmax):
        W = net.W[l]; mu, Ca, _ = GR.relu_moments(W @ mu, W @ Sig @ W.T); out.append(mu.clone())
        Sig = Ca + float(net.s[l]) ** 2 * I
    return out
def hsc(Lmax, keep_cross=False):
    mu = torch.zeros(256, dtype=torch.float64); Sig = I.clone(); El2 = 1.0; out = []; vg = []
    for l in range(Lmax):
        W = net.W[l]; mu2, C2, _ = GR.relu_moments(W @ mu, W @ Sig @ W.T); out.append(mu2.clone())
        u = mu2 / mu2.norm(); v = float(u @ C2 @ u) / float(mu2 @ mu2); vg.append(v)
        P = I - torch.outer(u, u)
        Sig = P @ C2 @ P + (float(net.s[l]) ** 2 / El2 * (1 + 0)) * I
        El2 *= (1 + v); mu = mu2
    return out, vg, El2
G = gc(32); H, vg, El2 = hsc(32)
for l in (2, 4, 8, 16, 24, 32):
    print(f"layer {l:2d}: GC nMSE {truthlib.nmse(G[l-1], T, l):.3e}   HSC nMSE {truthlib.nmse(H[l-1], T, l):.3e}")
print("HSC scale increments sd per layer:", ' '.join(f"{np.sqrt(v):.3f}" for v in vg[::4]), " final sd(lambda)", np.sqrt(El2 - 1))
for kf in ('k1', 'k2', 'k2a', 'k2m', 'k3f', 'k3fa'):
    for L in (8, 32):
        p = f'results/kprop_{seed}_s{sigma:g}_L{L}_{kf}.json'
        if os.path.exists(p):
            d = json.load(open(p)); m = d['means']
            print(f"kprop {kf} L={L}: nMSE@8 {truthlib.nmse(np.array(m[7]), T, 8):.3e}" + (f"  @32 {truthlib.nmse(np.array(m[31]), T, 32):.3e}" if L == 32 else '') + f"  flops {d['flops']:.2e}")
