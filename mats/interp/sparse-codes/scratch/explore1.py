"""Exploration on dev net seed 64 (declared in PREREG as seen): spectra at noise precision, neuron-basis sparsity,
noise-driven variance. usage: explore1.py FAMILY N SIGMA"""
import math, sys, os, time, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nets
torch.set_num_threads(int(os.environ.get('NT', '4')))
fam, n, sigma = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
LAY = [1, 2, 4, 8, 16, 24, 32]
net = nets.SNet(fam, n, 64, sigma, device='cpu', dtype=torch.float64)
g = torch.Generator().manual_seed(12345)
N = 1 << 16
t0 = time.time()
A = net.sample(N, g, LAY)
print(net.name(), f"sampled {N} in {time.time()-t0:.1f}s")
LN2 = math.log(2)
for l in LAY:
    a = A[l]; s = net.noise_sd(l); s2 = s * s
    mu = a.mean(0); c = a - mu
    C = c.T @ c / N
    lam = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
    r2 = float((a * a).mean())
    tv = float(lam.sum())
    pr = float(lam.sum() ** 2 / (lam ** 2).sum())
    kno = int((lam > s2).sum())
    # PCA residual (per-coordinate noise units) after k components
    tail = lam.flip(0).cumsum(0).flip(0)          # tail[k] = sum_{i>=k} lam_i
    def pca_err(k):
        return float(tail[k]) / s2 / n if k < n else 0.0
    ks = [1, 2, 4, 8, 16, 32, n // 2]
    pcs = {k: pca_err(k) for k in ks}
    # k needed for PCA residual below the noise (per-coordinate MSE < s^2) and below 1 bit total
    kbelow = next((k for k in range(n + 1) if (float(tail[k]) if k < n else 0.0) < n * s2), n)
    k1bit = next((k for k in range(n + 1) if (float(tail[k]) if k < n else 0.0) / (2 * s2) / LN2 < 1.0), n)
    line = (f"L{l:2d} s={s:.3g} r2={r2:.3g} var/coord={tv/n:.3g} ({tv/n/s2:.0f} s^2) PR={pr:.1f} "
            f"#eig>s2={kno} kPCA(belowNoise)={kbelow} kPCA(<1bit)={k1bit} PCAerr[noise units/coord] "
            + ' '.join(f"k{k}:{v:.3g}" for k, v in pcs.items()))
    if net.kind == 'relu':
        act = (a > 0).double()
        nact = act.sum(1)
        # neuron-basis k-sparse code around the mean: keep top-k |a_i - mu_i|
        dev = c.abs()
        srt = dev.sort(dim=1, descending=True).values ** 2
        tot = srt.sum(1)
        cum = srt.cumsum(1)
        nb = {k: float((tot - cum[:, k - 1]).mean()) / s2 / n for k in ks}
        # zero-based neuron code (no mean): keep the active set exactly -> L0 = #active
        line += (f" | active mean={float(nact.mean()):.1f} sd={float(nact.std()):.1f} max={int(nact.max())} "
                 f"| neuronTopK(centered) err: " + ' '.join(f"k{k}:{v:.3g}" for k, v in nb.items()))
    print(line, flush=True)
# noise-driven variance: E_x Var(a_l | x) per coordinate, in s^2 units (64 inputs x 128 noise draws)
g2 = torch.Generator().manual_seed(777)
X = torch.randn(64, n, generator=g2, dtype=torch.float64)
Xr = X.repeat_interleave(128, 0)
h = Xr; vals = {}
for l in range(1, 33):
    a = net.act(l, net.pre(l, h))
    if l in LAY:
        v = a.view(64, 128, n).var(1).mean()
        tot = a.var(0).mean()
        vals[l] = (float(v) / net.noise_sd(l) ** 2, float(v / tot))
    h = a + net.noise_sd(l) * torch.randn(a.shape, generator=g2, dtype=torch.float64)
print('noise-driven Var(a|x) per coord in s^2 units (fraction of total var): ' +
      ' '.join(f"L{l}:{v[0]:.2f}({v[1]:.3f})" for l, v in vals.items()), flush=True)
