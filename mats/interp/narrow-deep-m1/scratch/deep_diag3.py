"""Diagnostic (dev net only): deep-part variants applied to the exact law at c_d (sampled), nMSE vs truth:
 D1 J_ref + K2-Stein table from point masses, 10 radii over 9 decades (current);
 D2 the same with 40 radii over 8 decades;
 D3 J_ref + GHM(tau=1e-3) table, 16 radii;
 D0 noiseless deep (no noise correction).
Also prints, per radius, the table values against MC from the point mass (2^16 noise draws) for D2's grid subset."""
import sys, json, torch, math, time
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf, k2, ghm
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 103
c, c_d = 8, int(sys.argv[2]) if len(sys.argv) > 2 else 32
SIG = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
T = json.load(open(f'runs/truth_s{SEED}_sig{SIG:g}.json'))
net = nd.Net('plain', SEED, 8, 256, SIG, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
g = torch.Generator().manual_seed(11)
P = 1 << 17
h = torch.randn(P, n, generator=g, dtype=DT)
for l in range(c_d):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
deep = bvf.Deep(net, c_d, h[:256], torch.ones(256, dtype=DT), bvf.Flops())
print('class share', deep.class_share, 'nclass', deep.nclass, flush=True)
V0 = h @ deep.J.T
print('D0 noiseless deep (J_ref): %.3g' % float((V0.mean(0) - truth).pow(2).sum() / trv), flush=True)
VL = bvf.noiseless_pass(net, h, c_d, L, bvf.Flops())
print('D0b fully noiseless deep pass: %.3g' % float((VL.mean(0) - truth).pow(2).sum() / trv), flush=True)
V1 = deep.value(h, bvf.Flops())
print('D1 K2-Stein table 10 radii: %.3g' % float((V1.mean(0) - truth).pow(2).sum() / trv), flush=True)
rho = (h @ deep.r) / deep.rv
q = torch.quantile(rho, torch.tensor([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], dtype=DT))
print('rho quantiles (0.1%..99.9%):', [f'{float(v):.3g}' for v in q], 'rho_ref', float(deep.v.norm()), flush=True)


def table(rhos, how, tau=1e-3):
    out = []
    C0 = torch.diag(k2.noise_var(net, c_d - 1, n))
    for r_ in rhos:
        if how == 'k2':
            m, _, _ = k2.plain_prop(net, deep.v * r_, C0, c_d, L, exact=False)
        else:
            w = torch.ones(1, dtype=DT); mu = (deep.v * r_)[None].clone(); C = C0[None].clone()
            fl = ghm.Flops()
            for l in range(c_d, L):
                s2 = float(net.s[l]) ** 2 if l < L - 1 else None
                w, mu, C = ghm.relu_layer(w, mu, C, net.W[l], s2, tau, fl)
            m = ghm.mixture_mean(w, mu)
        out.append(m - r_ * deep.a0)
    return torch.stack(out)


def apply(rhos, tab):
    lr = torch.log(torch.tensor(rhos, dtype=DT))
    x = torch.log(rho.clamp_min(1e-300))
    idx = torch.searchsorted(lr, x).clamp(1, len(lr) - 1)
    wgt = ((x - lr[idx - 1]) / (lr[idx] - lr[idx - 1])).clamp(0, 1)
    cor = tab[idx - 1] + wgt[:, None] * (tab[idx] - tab[idx - 1])
    return V0 + cor


lo, hi = float(q[0]) / 3, float(q[-1]) * 3
rhos40 = torch.logspace(math.log10(lo), math.log10(hi), 40, dtype=DT).tolist()
t0 = time.time()
tab2 = table(rhos40, 'k2')
print('D2 K2-Stein table 40 radii over the support: %.3g (%.0fs)' % (float((apply(rhos40, tab2).mean(0) - truth).pow(2).sum() / trv), time.time() - t0), flush=True)
rhos16 = torch.logspace(math.log10(lo), math.log10(hi), 16, dtype=DT).tolist()
t0 = time.time()
tab3 = table(rhos16, 'ghm', 1e-3)
print('D3 GHM(1e-3) table 16 radii: %.3g (%.0fs)' % (float((apply(rhos16, tab3).mean(0) - truth).pow(2).sum() / trv), time.time() - t0), flush=True)
# MC check of the tables at a few radii
for i in (0, 5, 10, 15):
    r_ = rhos16[i]
    x0 = (deep.v * r_)[None].expand(1 << 16, n).clone()
    hh = x0 + torch.randn(x0.shape, generator=g, dtype=DT) * net.s[c_d - 1]
    for l in range(c_d, L):
        a = torch.relu(hh @ net.W[l].T); hh = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if l < L - 1 else a
    mc = hh.mean(0) - r_ * deep.a0
    k2v = tab2[min(range(40), key=lambda j: abs(math.log(rhos40[j] / r_)))]
    sc = float((r_ * deep.a0).norm())
    print(f'rho={r_:.3g}: |noise effect| MC {float(mc.norm())/sc:.3g} (MC se {float(hh.std(0).norm())/sc/256:.2g}) | K2 table {float(k2v.norm())/sc:.3g} | GHM {float(tab3[i].norm())/sc:.3g} (relative to |rho a0|)', flush=True)
