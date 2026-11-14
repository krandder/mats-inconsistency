"""Dev test (coordinator suggestion 2): deterministic Gaussian-sum filter for the deep part's noise effect, against MC
from point masses rho v_ref at c_d (dev 103 by default), and applied to the exact law at c_d (nMSE vs truth)."""
import sys, json, torch, math, time
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf, k2, gsf
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 103
c_d = 32
T = json.load(open(f'runs/truth_s{SEED}_sig0.001.json'))
net = nd.Net('plain', SEED, 8, 256, 1e-3, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']; fp = net.flops_path()
g = torch.Generator().manual_seed(11)
P = 1 << 17
h = torch.randn(P, n, generator=g, dtype=DT)
for l in range(c_d):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
deep = bvf.Deep(net, c_d, h[:256], torch.ones(256, dtype=DT), bvf.Flops())
v = deep.v; a0 = deep.a0
rho = (h @ deep.r) / deep.rv
q = torch.quantile(rho, torch.tensor([0.001, 0.999], dtype=DT))
lo, hi = float(q[0]) / 3, float(q[1]) * 3
C0 = torch.diag(k2.noise_var(net, c_d - 1, n))
cfgs = [dict(M=5, tau=0.5, depth=2, kmax=16), dict(M=5, tau=0.5, depth=3, kmax=48), dict(M=7, tau=0.4, depth=3, kmax=96)]
for r_ in [2.11e-05, 4.67e-4, 0.0103, 0.229]:
    x0 = (v * r_)[None].expand(1 << 16, n).clone()
    hh = x0 + torch.randn(x0.shape, generator=g, dtype=DT) * net.s[c_d - 1]
    for l in range(c_d, L):
        a = torch.relu(hh @ net.W[l].T); hh = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if l < L - 1 else a
    mc = hh.mean(0); sc = float((r_ * a0).norm())
    row = f"rho={r_:.3g}: MC effect {float((mc - r_ * a0).norm())/sc:.3g} (se {float(hh.std(0).norm())/sc/256:.2g})"
    for cf in cfgs:
        t0 = time.time(); fl = gsf.Flops()
        mg, K = gsf.gsf_run(net, v * r_, C0, c_d, L, fl, **cf)
        row += f" | GSF{cf['M']},{cf['tau']},{cf['depth']},{cf['kmax']}: effect {float((mg - r_ * a0).norm())/sc:.3g}, err vs MC {float((mg - mc).norm())/sc:.2g} ({fl.n/fp:.0f} paths, {time.time()-t0:.0f}s)"
    print(row, flush=True)
# table on the support + apply to the exact law at c_d
for cf in cfgs[:2]:
    rhos = torch.logspace(math.log10(lo), math.log10(hi), 16, dtype=DT).tolist()
    t0 = time.time(); fl = gsf.Flops(); tab = []
    for r_ in rhos:
        mg, K = gsf.gsf_run(net, v * r_, C0, c_d, L, fl, **cf)
        tab.append(mg - r_ * a0)
    tab = torch.stack(tab); lr = torch.log(torch.tensor(rhos, dtype=DT))
    x = torch.log(rho.clamp_min(1e-300)); idx = torch.searchsorted(lr, x).clamp(1, len(lr) - 1)
    wgt = ((x - lr[idx - 1]) / (lr[idx] - lr[idx - 1])).clamp(0, 1)
    cor = tab[idx - 1] + wgt[:, None] * (tab[idx] - tab[idx - 1])
    V = h @ deep.J.T + cor
    e = float((V.mean(0) - truth).pow(2).sum() / trv)
    e0 = float(((h @ deep.J.T).mean(0) - truth).pow(2).sum() / trv)
    print(f"GSF table {cf}: exact law at c_d through J_ref + table: nMSE {e:.3g} (noiseless deep {e0:.3g}); table cost {fl.n/fp:.0f} paths ({time.time()-t0:.0f}s)", flush=True)
