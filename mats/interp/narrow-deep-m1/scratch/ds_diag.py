"""Diagnostic (dev net only): split the downstream error into deep part, middle noise, projection and interpolation."""
import sys, json, torch, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 103
c = int(sys.argv[2]) if len(sys.argv) > 2 else 8
c_d = int(sys.argv[3]) if len(sys.argv) > 3 else 32
T = json.load(open(f'runs/truth_s{SEED}_sig0.001.json'))
net = nd.Net('plain', SEED, 8, 256, 1e-3, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
g = torch.Generator().manual_seed(5)
P = 1 << 17
h = torch.randn(P, n, generator=g, dtype=DT)
for l in range(c):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
mh = h.mean(0); M2 = h.T @ h / P; Ch = M2 - torch.outer(mh, mh)
fl = bvf.Flops()
cand = bvf.noiseless_pass(net, h[:256], c, c_d, fl)
deep = bvf.Deep(net, c_d, cand, torch.ones(256, dtype=DT), fl)
print('deep class share', deep.class_share, deep.nclass, flush=True)
hh = h.clone()
for l in range(c, L):
    a = torch.relu(hh @ net.W[l].T); hh = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if l < L - 1 else a
print('MC downstream from true h_c (sampling err ~1/P=%.1g): %.3g' % (1 / P, float((hh.mean(0) - truth).pow(2).sum() / trv)), flush=True)
Vd = deep.value(bvf.noiseless_pass(net, h, c, c_d, bvf.Flops()), bvf.Flops())
print('noiseless middle + deep (no projection): %.3g' % float((Vd.mean(0) - truth).pow(2).sum() / trv), flush=True)
VL = bvf.noiseless_pass(net, h, c, L, bvf.Flops())
print('fully noiseless downstream c->L: %.3g' % float((VL.mean(0) - truth).pow(2).sum() / trv), flush=True)
hm = h.clone()
for l in range(c, c_d):
    a = torch.relu(hm @ net.W[l].T); hm = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
Vm = deep.value(hm, bvf.Flops())
print('noisy middle (sampled) + deep value: %.3g' % float((Vm.mean(0) - truth).pow(2).sum() / trv), flush=True)
hm2 = hm.clone()
for l in range(c_d, L):
    a = torch.relu(hm2 @ net.W[l].T); hm2 = a
print('noisy middle (sampled) + noiseless deep: %.3g' % float((hm2.mean(0) - truth).pow(2).sum() / trv), flush=True)
Ms = torch.zeros(n, n, dtype=DT)
for x in h[:2048]:
    _, Jm = bvf.jac_pass(net, x, c, c_d, bvf.Flops()); Jt = deep.J @ Jm; Ms += Jt.T @ Jt
Ms /= Ms.trace()
ev, Uc = torch.linalg.eigh(Ch); Chh = Uc @ torch.diag(ev.clamp_min(0).sqrt()) @ Uc.T
S = Chh @ Ms @ Chh; eS, VS = torch.linalg.eigh(S); eS, VS = eS.flip(0), VS.flip(1)
U8 = torch.linalg.qr(Chh @ VS)[0]
print('eS', [f'{e:.2e}' for e in eS.tolist()], flush=True)
for k in (2, 3, 4, 5, 6):
    U = U8[:, :k]; Pp = torch.eye(n, dtype=DT) - U @ U.T
    B = Pp @ M2 @ U @ torch.linalg.inv(U.T @ M2 @ U)
    t = h @ U
    hr = t @ (U + B).T
    Vp = deep.value(bvf.noiseless_pass(net, hr, c, c_d, bvf.Flops()), bvf.Flops())
    e_proj = float((Vp.mean(0) - truth).pow(2).sum() / trv)
    e0 = U.T @ mh
    row = [f'k={k} projection only: {e_proj:.3g}']
    for G in (8, 16, 32):
        for X in (0.5, 1.0, 2.0):
            if k >= 4 and G > 16: continue
            if k >= 5 and G > 8: continue
            grid = bvf.Grid(net, c, c_d, deep, U, B, e0, G, X, bvf.Flops())
            Vg, out = grid.value(t, bvf.Flops())
            row.append(f'G{G}X{X}:{float((Vg.mean(0) - truth).pow(2).sum() / trv):.2g}(out {out:.3f})')
    print(' | '.join(row), flush=True)
