"""Active subspace of the depth-L output w.r.t. the input: C = E[J^T J] (J = d a_L / d x, summed over output neurons),
and of the top output mode's coefficient alone. If C's spectrum is flat, QMC in a few input directions has nothing to use."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib
torch.set_num_threads(2)
seed, sigma, L, M = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
net = anet.Net(seed, sigma=sigma)
T = truthlib.load(seed, sigma); m = torch.tensor(T['mean'][L - 1]); v1 = m / m.norm()
g = torch.Generator().manual_seed(8)
W = [w.double() for w in net.W]
Cfull = torch.zeros(256, 256, dtype=torch.float64); Ctop = torch.zeros(256, 256, dtype=torch.float64)
for b in range(M // 64):
    x = torch.randn(64, 256, generator=g, dtype=torch.float64, requires_grad=True)
    h = x
    for l in range(L):
        a = torch.relu(h @ W[l].T)
        h = a + float(net.s[l]) * torch.randn(a.shape, generator=g, dtype=torch.float64) if l < L - 1 else a
    # full Jacobian rows via 256 backward passes would be costly; use random projections: E[J^T u u^T J] with u ~ N(0, I)
    for _ in range(4):
        u = torch.randn(64, 256, generator=g, dtype=torch.float64)
        gx, = torch.autograd.grad((h * u).sum(), x, retain_graph=True)
        Cfull += gx.T @ gx
    gt, = torch.autograd.grad((h @ v1).sum(), x)
    Ctop += gt.T @ gt
for name, C in (('all outputs', Cfull), ('top mode (mean direction)', Ctop)):
    ev = torch.linalg.eigvalsh(C).flip(0); ev = ev / ev.sum()
    print(f"L={L} {name}: share of E[J^T J] in top 1/4/16/64 input directions: {ev[:1].sum():.3f} {ev[:4].sum():.3f} {ev[:16].sum():.3f} {ev[:64].sum():.3f}"
          f"  (isotropic would be {1/256:.3f} {4/256:.3f} {16/256:.3f} {64/256:.3f})", flush=True)
