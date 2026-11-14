"""How a mean error made at layer l survives to layer 32: shift h_l by a fixed vector delta (random direction, norm
eps * |E h_l| ), push coupled paths (same x, same noise) to layer 32, energy ratio |E a_32(shift) - E a_32|^2 / |delta|^2,
in the units of the predicted law prod_k (1 - arccos(c_k)/pi) with c_k the mean-energy fraction (replica cosine) of z_k."""
import sys, math, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet
torch.set_num_threads(2)
seed, lg = int(sys.argv[1]), int(sys.argv[2])
N = 1 << lg
for sigma in (0.001, 0.01, 0.03, 0.1):
    net = anet.Net(seed, sigma=sigma)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(N, 256, generator=g, dtype=torch.float32)
    noise = [torch.randn(N, 256, generator=g, dtype=torch.float32) for _ in range(31)]
    A = net.forward(x, g, keep=True, dtype=torch.float32, noise=noise)
    # replica cosines of pre-activations at each layer, from the plain run (mean energy fraction of a_k ~ of z_k)
    cos = []
    for k in range(32):
        a = A[k].double(); m = a.mean(0)
        h = a + float(net.s[k]) * noise[k].double() if k < 31 else a
        z = h @ net.W[k + 1].T if k < 31 else None
        if z is not None:
            mz = z.mean(0); cos.append(float((mz @ mz) / (z * z).sum(1).mean()))
    rows = []
    for l in (4, 8, 16, 24, 28):
        base = A[31].double().mean(0)
        hl = A[l - 1] + float(net.s[l - 1]) * noise[l - 1]
        d = torch.randn(256, generator=g, dtype=torch.float32); d = d / d.norm() * 0.01 * A[l - 1].mean(0).norm()
        h = hl + d
        for k in range(l, 32):
            a = torch.relu(h @ net.W[k].T.float())
            if k < 31:
                h = a + float(net.s[k]) * noise[k]
        shifted = a.double().mean(0)
        ratio = float(((shifted - base) ** 2).sum() / (d.double() ** 2).sum())
        pred = float(np.prod([1 - math.acos(min(1.0, c)) / math.pi for c in cos[l - 1:31]]))   # layers l+1..32
        rows.append(f"l={l:2d}: energy kept {ratio:.3e} (predicted {pred:.3e})")
    print(f"sigma={sigma}: " + ' | '.join(rows), flush=True)
