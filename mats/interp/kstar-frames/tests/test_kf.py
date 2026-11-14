"""Tests for lib/kf.py. Run: nice -n 19 ionice -c3 python tests/test_kf.py  (exit 0 = all pass)."""
import math, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import kf, nd

torch.set_num_threads(2)
DT = torch.float64
fails = []


def check(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{info}]' if info else ''))
    if not cond:
        fails.append(name)


def a_from_zc(net, zc, eps, c):
    """a_L as a function of z_c with fixed downstream noise eps (list of L-1)."""
    h = torch.relu(zc) + net.s[c - 1] * eps[c - 1] if c < net.L else None
    if c == net.L:
        return torch.relu(zc)
    for l in range(c, net.L):
        z = h @ net.W[l].T
        a = torch.relu(z)
        if l < net.L - 1:
            h = a + net.s[l] * eps[l]
    return a


# 1. Jacobian against autograd and finite differences, with noise
net = nd.Net('plain', 7, 16, 8, 1e-3)
g = kf.gen(1)
x = kf.draw_x(4, 16, g)
zs, eps = kf.forward(net, x, g)
D = kf.masks(zs)
Js = kf.downstream_jacobians(net, D, range(1, 9))
for c in (1, 3, 5, 8):
    errs = []
    for p in range(4):
        zc = zs[c - 1][p].clone().requires_grad_(False)
        ep = [e[p] for e in eps]
        Jauto = torch.autograd.functional.jacobian(lambda z: a_from_zc(net, z, ep, c), zc)
        errs.append(float((Jauto - Js[c][p]).abs().max()))
    check(f'jacobian vs autograd c={c}', max(errs) < 1e-12, f'max abs err {max(errs):.2e}')
c = 3
u = torch.randn(16, generator=g, dtype=DT)
fd = []
for p in range(4):
    ep = [e[p] for e in eps]
    d = (a_from_zc(net, zs[c - 1][p] + 1e-7 * u, ep, c) - a_from_zc(net, zs[c - 1][p], ep, c)) / 1e-7
    fd.append(float((d - Js[c][p] @ u).norm() / (Js[c][p] @ u).norm()))
check('jacobian vs finite differences c=3', max(fd) < 1e-5, f'max rel err {max(fd):.2e}')

# 2. Euler identities in the noiseless net: J_c z_c = a_L, P_c x = z_c
net0 = nd.Net('plain', 7, 16, 8, 0.0)
zs0, _ = kf.forward(net0, x, g)
D0 = kf.masks(zs0)
J0 = kf.downstream_jacobians(net0, D0, range(1, 9))
P0 = kf.past_products(net0, D0, range(1, 9))
aL = torch.relu(zs0[-1])
e1 = max(float(((J0[c] @ zs0[c - 1][:, :, None]).squeeze(-1) - aL).abs().max()) for c in range(1, 9))
e2 = max(float(((P0[c] @ x[:, :, None]).squeeze(-1) - zs0[c - 1]).abs().max()) for c in range(1, 9))
check('Euler J_c z_c = a_L (sigma=0)', e1 < 1e-10, f'{e1:.2e}')
check('Euler P_c x = z_c (sigma=0)', e2 < 1e-10, f'{e2:.2e}')

# 3. past products against autograd with noise
def zc_from_x(net, xx, eps, c):
    h = xx
    for l in range(c):
        z = h @ net.W[l].T
        if l < c - 1:
            h = torch.relu(z) + net.s[l] * eps[l]
    return z
Ps = kf.past_products(net, D, (2, 5))
err = max(float((torch.autograd.functional.jacobian(lambda xx: zc_from_x(net, xx, [e[p] for e in eps], cc), x[p])
                 - Ps[cc][p]).abs().max()) for p in range(4) for cc in (2, 5))
check('past product vs autograd (noisy)', err < 1e-12, f'{err:.2e}')

# 4. scores: true frame scores 1, random frame ~ chance
net = nd.Net('plain', 8, 64, 16, 1e-3)
x = kf.draw_x(64, 64, g)
zs, eps = kf.forward(net, x, g)
D = kf.masks(zs)
J = kf.downstream_jacobians(net, D, (8,))[8]
s, V = kf.svd_right(J)
k = 10
O, CS, T = kf.scores(V[:, :, :k], V, s, k)
tail = 1 - (s[:, :k] ** 2).sum(-1) / (s ** 2).sum(-1)
check('scores of true frame', float((O - 1).abs().max()) < 1e-10 and float((CS - 1).abs().max()) < 1e-10
      and float((T - tail).abs().max()) < 1e-10)
Gr = kf.random_frame(D[7], k, g)
O, CS, T = kf.scores(Gr, V, s, k)
chance = float((k / D[7].sum(-1)).mean())
check('random frame overlap ~ chance', abs(float(O.mean()) - chance) < 0.05, f'O {float(O.mean()):.3f} chance {chance:.3f}')
check('true frame lies in range(D_c)', float(((1 - D[7])[:, :, None] * V[:, :, :k]).abs().max()) < 1e-10)

# 5. dual transport inverts the backward relation exactly when m_c <= m_{c-1}
c = 8
Fc = kf.project_active(torch.randn(64, 64, k, generator=g, dtype=DT), D[c - 1])
Fprev = kf.backward_one(net, Fc, D, c - 1)            # orth(D_{c-1} W_c^T F_c)
Gd = kf.dual_forward(net, Fprev, D, c - 1, c)
ok = (D[c - 1].sum(-1) <= D[c - 2].sum(-1))
O, _, _ = kf.scores(Gd, Fc, torch.ones(64, k, dtype=DT), k)
check('dual transport inverts backward step (m_c <= m_{c-1})', float((O[ok] - 1).abs().max()) < 1e-8,
      f'{int(ok.sum())} paths, min O {float(O[ok].min()):.10f}')

# 6. push-forward of zero steps is identity; one step stays in range(D_c)
G1 = kf.push_forward(net, Fprev, D, c - 1, c)
check('push-forward lands in range(D_c)', float(((1 - D[c - 1])[:, :, None] * G1).abs().max()) < 1e-10)
check('push-forward zero steps', float((kf.push_forward(net, Fprev, D, c - 1, c - 1) - Fprev).abs().max()) == 0.0)

# 7. batched dual transport = per-path loop (including paths whose active count drops below k)
net = nd.Net('plain', 11, 16, 12, 1e-3)
x = kf.draw_x(96, 16, g)
zs, eps = kf.forward(net, x, g)
D = kf.masks(zs)
J = kf.downstream_jacobians(net, D, (3, 7))
for k in (4, 8):
    F3 = kf.svd_right(J[3])[1][:, :, :k]
    Gb = kf.dual_forward(net, F3, D, 3, 7)
    Gl = kf.dual_forward_loop(net, F3, D, 3, 7)
    rk = torch.linalg.matrix_rank(Gl)
    ov = (Gb.transpose(-1, -2) @ Gl).pow(2).sum((-1, -2)) / rk
    check(f'batched dual = loop dual (k={k})', float((ov - 1).abs().max()) < 1e-6,
          f'min overlap {float(ov.min()):.8f}, paths with rank < k: {int((rk < k).sum())}')

# 8. eigh paths used on CUDA (forced here on CPU) agree with LAPACK SVD
net = nd.Net('plain', 12, 64, 24, 1e-3)
x = kf.draw_x(64, 64, g)
zs, eps = kf.forward(net, x, g)
D = kf.masks(zs)
J = kf.downstream_jacobians(net, D, (8, 12))
for c in (8, 12):
    s1, V1 = kf.svd_right(J[c]); s2, V2 = kf.svd_right(J[c], use_eigh=True)
    k = int(kf.n_surv(s1).double().mean().round())
    rel = float(((s2[:, :k] - s1[:, :k]).abs() / s1[:, :1]).max())
    ov = (V1[:, :, :k].transpose(-1, -2) @ V2[:, :, :k]).pow(2).sum((-1, -2)) / k
    cnt = float((kf.svdvals(J[c], use_eigh=True) >= 0.01 * s2[:, :1]).sum(-1).double().mean())
    check(f'eigh frames = SVD frames (c={c}, k={k})', rel < 1e-9 and float((ov - 1).abs().max()) < 1e-6
          and abs(cnt - float(kf.n_surv(s1).double().mean())) < 1e-9, f'max rel sv err {rel:.1e}, min overlap {float(ov.min()):.9f}')
P = kf.past_products(net, D, (8,))[8]
A8 = D[7][:, :, None] * P
k = 10
L1, L2 = kf.left_frame(A8, k), kf.left_frame(A8, k, use_eigh=True)
ov = (L1.transpose(-1, -2) @ L2).pow(2).sum((-1, -2)) / k
check('eigh left frame = SVD left frame', float((ov - 1).abs().max()) < 1e-6, f'min overlap {float(ov.min()):.9f}')
F8 = kf.svd_right(J[8])[1][:, :, :k]
Ge = kf.dual_forward(net, F8, D, 8, 12, use_eigh=True)
Gl = kf.dual_forward_loop(net, F8, D, 8, 12)
rk = torch.linalg.matrix_rank(Gl)
ov = (Ge.transpose(-1, -2) @ Gl).pow(2).sum((-1, -2)) / rk
check('eigh dual transport = loop dual transport', float((ov - 1).abs().max()) < 1e-5, f'min overlap {float(ov.min()):.8f}')

print('ALL PASS' if not fails else f'{len(fails)} FAIL: {fails}')
sys.exit(1 if fails else 0)
