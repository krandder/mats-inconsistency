"""Unit checks for lib/bo.py, lib/bvf.py, lib/ghm.py (dev net 103; run: python test_lib.py)."""
import sys, math, itertools
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf, bo, ghm
torch.set_num_threads(1)
DT = torch.float64
ok = True


def check(name, cond, info=''):
    global ok
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + info if info else ''), flush=True)
    ok = ok and cond


# 1. sphere rule: weights sum to 1; integrates x_i^2 -> 1/k and x_i^4 -> 3/(k(k+2)) on S^(k-1), odd moments 0
for k in (2, 3, 4, 5):
    for m in (8, 16):
        X, w = bo.sphere_rule(k, m)
        s2 = (w[:, None] * X ** 2).sum(0)
        s4 = (w[:, None] * X ** 4).sum(0)
        s1 = (w[:, None] * X).sum(0)
        check(f'sphere_rule k={k} m={m}', abs(float(w.sum()) - 1) < 1e-12 and float((s2 - 1 / k).abs().max()) < 2e-2
              and float((s4 - 3 / (k * (k + 2))).abs().max()) < 3e-2 and float(s1.abs().max()) < 1e-12,
              f'max|E x^2 - 1/k| = {float((s2 - 1/k).abs().max()):.1e}, max|E x^4 - 3/(k(k+2))| = {float((s4 - 3/(k*(k+2))).abs().max()):.1e}')
# 2. E chi_k and the radial rule
for k in (2, 3, 5, 8):
    r, w = bo.radial_rule(k, 8)
    e1 = abs(float((w * r).sum()) - bo.chi_mean(k)) / bo.chi_mean(k)
    check(f'radial rule k={k} (exact for E rho^2; E rho approximate, used only by the unused noise table)',
          e1 < 1e-2 and abs(float((w * r * r).sum()) - k) < 1e-10, f'E rho rel err {e1:.1e}')
# 3. BO identity on a linear function: E[f(U t)] for f(x) = a.x+ (ReLU of a linear form) vs closed form
k = 3
U = torch.linalg.qr(torch.randn(8, k, dtype=DT, generator=torch.Generator().manual_seed(1)))[0]
a = torch.randn(8, dtype=DT, generator=torch.Generator().manual_seed(2))
X, w = bo.sphere_rule(k, 64)
est = bo.chi_mean(k) * float((w * torch.relu((X @ U.T) @ a)).sum())
exact = float((U.T @ a).norm()) / math.sqrt(2 * math.pi)        # E ReLU(b.t), t ~ N(0, I_k)
check('BO on ReLU(a.x) restricted to a 3-D slice', abs(est - exact) / exact < 1e-3, f'{est:.6g} vs {exact:.6g}')
# 4. deep part: for a state in the reference class, J_ref h equals the noiseless output; ghm one layer vs MC
net = nd.Net('plain', 103, 8, 256, 1e-3)
g = torch.Generator().manual_seed(3)
h = torch.randn(4096, 8, generator=g, dtype=DT)
for l in range(64):
    h = torch.relu(h @ net.W[l].T) + torch.randn(h.shape, generator=g, dtype=DT) * net.s[l]
deep = bvf.Deep(net, 64, h[:256], torch.ones(256, dtype=DT), bvf.Flops())
key, aL = bvf.future_classes(net, h, 64, bvf.Flops())
key_ref, _ = bvf.future_classes(net, deep.v[None] * 1.0, 64, bvf.Flops())
same = key == key_ref[0]
V = deep.value(h[same], bvf.Flops())
rel = float(((V - aL[same]).norm(dim=1) / aL[same].norm(dim=1).clamp_min(1e-300)).max())
check('J_ref exact on the reference class (noiseless)', rel < 1e-9, f'{int(same.sum())} of 4096 states in class, max rel err {rel:.1e}')
w0 = torch.ones(1, dtype=DT); m0 = torch.zeros(1, 8, dtype=DT); C0 = torch.eye(8, dtype=DT)[None]
fl = ghm.Flops()
w1, m1, C1 = ghm.relu_layer(w0, m0, C0, net.W[0], float(net.s[0]) ** 2, 1e-9, fl)
mm = ghm.mixture_mean(w1, m1)
xs = torch.randn(1 << 21, 8, generator=g, dtype=DT)
hs = torch.relu(xs @ net.W[0].T) + torch.randn(xs.shape, generator=g, dtype=DT) * net.s[0]
err = float((mm - hs.mean(0)).norm() / hs.std(0).norm())
check('GHM layer 1 mean vs MC (2^21)', err < 5e-3 and abs(float(w1.sum()) - 1) < 1e-9, f'rel err {err:.1e}, {len(w1)} pieces, weight {float(w1.sum()):.12f}')
print('ALL PASS' if ok else 'SOME FAIL')
