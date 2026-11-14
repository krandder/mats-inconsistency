"""Diagnostic (dev net only): noise treatment in the deep part.
Compares, for the true law at c (sampled), the noiseless middle c->c_d followed by:
 A: J_ref h + table from point masses at c_d (current);
 B: J_ref h + table from K2 started at c_d with the state's angular spread (rho^2 Sigma_ang + s^2 I; Sigma_ang from a
    K2 run c->c_d from a point mass at c along the reference, i.e. noise-built);
 C: J_ref h_noiseless-at-c_d + table Delta_c(rho_c): K2 (exact or Stein) from point masses rho v_c at c through c->L
    minus the noiseless output, rho_c = radial coordinate at c;
 D: per-bin K2-Stein from the sampled law at c_d (previous lane's deep part, 16 log-radius bins; oracle law).
"""
import sys, json, torch, math
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf, k2
torch.set_num_threads(1)
DT = torch.float64
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 103
c = int(sys.argv[2]) if len(sys.argv) > 2 else 8
c_d = int(sys.argv[3]) if len(sys.argv) > 3 else 32
SIG = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
T = json.load(open(f'runs/truth_s{SEED}_sig{SIG:g}.json'))
net = nd.Net('plain', SEED, 8, 256, SIG, r=T['r']); n, L = 8, 256
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']
g = torch.Generator().manual_seed(5)
P = 1 << 17
h = torch.randn(P, n, generator=g, dtype=DT)
for l in range(c):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
fl = bvf.Flops()
cand = bvf.noiseless_pass(net, h[:256], c, c_d, fl)
deep = bvf.Deep(net, c_d, cand, torch.ones(256, dtype=DT), fl)
Hd = bvf.noiseless_pass(net, h, c, c_d, bvf.Flops())
VA = deep.value(Hd, bvf.Flops())
print('A point-mass table at c_d: %.3g' % float((VA.mean(0) - truth).pow(2).sum() / trv), flush=True)
# reference at c: the majority-class probe at c (same index as deep's choice: recompute)
key, aL = bvf.future_classes(net, h[:256], c, bvf.Flops())
uk, inv = torch.unique(key, return_inverse=True)
cm = torch.zeros(len(uk), dtype=DT).index_add_(0, inv, aL.norm(dim=1))
sel = torch.nonzero(inv == int(cm.argmax()))[:, 0]
vc = h[sel[aL.norm(dim=1)[sel].argmax()]]; vc = vc / vc.norm()
# noiseless output map along v_c: a0c = F_{c->L}(v_c)
a0c = bvf.noiseless_pass(net, vc[None], c, L, bvf.Flops())[0]
Jc_end, Jc = bvf.jac_pass(net, vc, c, L, bvf.Flops())
rc = (Jc.T @ a0c) / a0c.pow(2).sum()
rho_typ = float((h @ rc).abs().median())
for exact in (False, True):
    rhos = [rho_typ * 10 ** e for e in torch.linspace(-4, 2, 13).tolist()]
    tab = []
    for rho in rhos:
        m, Cm, f = k2.plain_prop(net, vc * rho, torch.zeros(n, n, dtype=DT), c, L, exact=exact)
        tab.append(m - rho * a0c)
    tab = torch.stack(tab); lr = torch.log(torch.tensor(rhos, dtype=DT))
    rho_h = (h @ rc).clamp_min(1e-300)
    x = torch.log(rho_h)
    idx = torch.searchsorted(lr, x).clamp(1, len(lr) - 1)
    wgt = ((x - lr[idx - 1]) / (lr[idx] - lr[idx - 1])).clamp(0, 1)
    cor = tab[idx - 1] + wgt[:, None] * (tab[idx] - tab[idx - 1])
    VL0 = bvf.noiseless_pass(net, Hd, c_d, L, bvf.Flops())
    VC = VL0 + cor
    print(f'C table from c (K2 exact={exact}) + noiseless: %.3g' % float((VC.mean(0) - truth).pow(2).sum() / trv), flush=True)
# D: previous lane's binned K2-Stein from the sampled law at c_d (noisy middle)
hm = h.clone()
for l in range(c, c_d):
    a = torch.relu(hm @ net.W[l].T); hm = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
for NB in (8, 16):
    r = hm.norm(dim=1); lr2 = torch.log(r.clamp_min(1e-300))
    qs = torch.quantile(lr2, torch.linspace(0, 1, NB + 1, dtype=DT)[1:-1]); b = torch.bucketize(lr2, qs)
    est = torch.zeros(n, dtype=DT)
    for kk in range(NB):
        s_ = b == kk; hk = hm[s_]
        mL, _, _ = k2.plain_prop(net, hk.mean(0), torch.cov(hk.T), c_d, L, exact=False)
        est += hk.shape[0] / P * mL
    print(f'D binned K2-Stein from sampled law at c_d ({NB} bins): %.3g' % float((est - truth).pow(2).sum() / trv), flush=True)
# E: single K2-Stein / exact from the sampled moments at c_d
for exact in (False, True):
    mL, _, _ = k2.plain_prop(net, hm.mean(0), torch.cov(hm.T), c_d, L, exact=exact)
    print(f'E single Gaussian K2 (exact={exact}) from sampled moments at c_d: %.3g' % float((mL - truth).pow(2).sum() / trv), flush=True)
