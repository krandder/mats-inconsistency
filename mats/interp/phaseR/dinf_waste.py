"""Numerical illustration of T3 (D_inf waste of Gaussian-tailed families is infinite): along rays h = lam * v,
log q(h) - log K q_0(h) for the moment-matched Gaussian q_1 (layer 1, d = 8, tanh 0.75, sigma 0.3); K q_0 = p_1 exactly.
log K q_0 by tempered SMC (more particles far out). usage: python dinf_waste.py -> results/dinf_waste.json"""
import json, math, os
import torch
import rnet as R, fam as F, kq as KQ

torch.set_num_threads(1)
DT = torch.float64
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
d = 8
net = R.RNet(d, 4, 'tanh', 0.75, 0.3, 0)
gen = torch.Generator().manual_seed(1)
q0 = F.Gauss(torch.zeros(d, dtype=DT), torch.eye(d, dtype=DT))
X = q0.sample(1 << 16, gen); H = net.f(1, X) + 0.3 * torch.randn(1 << 16, d, generator=gen, dtype=DT)
q = F.Gauss.fit(H)
out = []
for k in range(3):
    v = H[k] / H[k].norm()
    lams = torch.tensor([0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], dtype=DT) * math.sqrt(d)
    P = lams[:, None] * v[None]
    lk, _ = KQ.log_kq_smc(net, 1, q0, P, gen, Np=4096, ess_frac=0.8, n_moves=8)
    r = (q.logpdf(P) - lk) / math.log(2)
    out.append(dict(lam_over_sqrtd=[0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], log2_q_over_Kq=r.tolist()))
    print([round(x, 1) for x in r.tolist()], flush=True)
json.dump(out, open(os.path.join(OUT, 'dinf_waste.json'), 'w'), indent=1)
