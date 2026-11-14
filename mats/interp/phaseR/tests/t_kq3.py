import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import rnet as R, fam as F, kq as KQ
gen = torch.Generator().manual_seed(3)
from t_kq2 import node_logkq
d, sig = 8, 0.3
net = R.RNet(d, 3, 'tanh', 0.75, sig, 0)
X = torch.randn(1 << 15, d, generator=gen, dtype=R.DT)
H1 = net.f(1, X) + sig * torch.randn(1 << 15, d, generator=gen, dtype=R.DT)
q = F.Gauss.fit(H1)
H = q.sample(32, gen)
ln1 = node_logkq(net, 2, q, H, 1 << 21); ln2 = node_logkq(net, 2, q, H, 1 << 23)
print('node 2^21 vs 2^23: mean diff', float((ln1 - ln2).mean()), 'maxabs', float((ln1 - ln2).abs().max()))
for Np, ef, nm in ((256, 0.5, 5), (1024, 0.7, 8), (4096, 0.8, 8)):
    t = time.time(); li, ns = KQ.log_kq_smc(net, 2, q, H, gen, Np=Np, ess_frac=ef, n_moves=nm); ti = time.time() - t
    e = li - ln2
    print(f'Np={Np} ess={ef} moves={nm}: SMC-node mean {float(e.mean()):+.3f} sd {float(e.std()):.3f} maxabs {float(e.abs().max()):.3f}; steps {float(ns.median()):.0f}; {ti:.1f}s', flush=True)
