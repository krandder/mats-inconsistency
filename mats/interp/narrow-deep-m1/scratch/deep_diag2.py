"""Diagnostic (dev net only): the deep part's noise effect along the reference ray, phi(rho) = E[a_L | h_cd = rho v].
MC (2^18 noise draws) vs K2-Stein vs K2 exact vs GHM(tau) from the point mass (plus layer c_d's noise)."""
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
trv = T['trvar']
g = torch.Generator().manual_seed(7)
h = torch.randn(1 << 12, n, generator=g, dtype=DT)
for l in range(c):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
cand = bvf.noiseless_pass(net, h[:256], c, c_d, bvf.Flops())
deep = bvf.Deep(net, c_d, cand, torch.ones(256, dtype=DT), bvf.Flops())
v = deep.v
Hd = bvf.noiseless_pass(net, h, c, c_d, bvf.Flops())
rho_med = float(((Hd @ deep.r) / deep.rv).median())
print('rho_med', rho_med, 'class share', deep.class_share)
for f in (0.01, 0.1, 1.0, 10.0):
    rho = rho_med * f
    t0 = time.time()
    x0 = (v * rho)[None].expand(1 << 18, n).clone()
    hh = x0 + torch.randn(x0.shape, generator=g, dtype=DT) * net.s[c_d - 1]
    for l in range(c_d, L):
        a = torch.relu(hh @ net.W[l].T); hh = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if l < L - 1 else a
    mc = hh.mean(0); mse_mc = hh.var(0).sum() / (1 << 18)
    C0 = torch.diag(k2.noise_var(net, c_d - 1, n))
    m_st, _, _ = k2.plain_prop(net, v * rho, C0, c_d, L, exact=False)
    m_ex, _, _ = k2.plain_prop(net, v * rho, C0, c_d, L, exact=True)
    row = f'rho={f:g}x: |MC|^2/trv={float(mc.pow(2).sum()/trv):.3g} MC se2/trv={float(mse_mc/trv):.2g} noiseless-MC {float((rho*deep.a0-mc).pow(2).sum()/trv):.3g} K2st-MC {float((m_st-mc).pow(2).sum()/trv):.3g} K2ex-MC {float((m_ex-mc).pow(2).sum()/trv):.3g}'
    for tau in (1e-3, 1e-5):
        fl = ghm.Flops()
        w = torch.ones(1, dtype=DT); mu = (v * rho)[None].clone(); C = C0[None].clone()
        for l in range(c_d, L):
            s2 = float(net.s[l]) ** 2 if l < L - 1 else None
            w, mu, C = ghm.relu_layer(w, mu, C, net.W[l], s2, tau, fl)
        mg = ghm.mixture_mean(w, mu)
        row += f' GHM{tau:g}-MC {float((mg-mc).pow(2).sum()/trv):.3g} ({len(w)} comps, {fl.n/net.flops_path():.0f} paths)'
    print(row, f'({time.time()-t0:.0f}s)', flush=True)
