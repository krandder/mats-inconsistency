"""Gate-cell design scan: KL(Bayes arm f_q || Bayes arm f=0) with identical query reports. usage: python gate_scan.py a|b"""
import sys, itertools, numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/worlds'); sys.path.insert(0, '/home/agent/workspaces/mats/work/world-fa')
import harness as H, fa_core as C
prior = sys.argv[1]
cfg = C.full_cfg(dict(prior=prior))
G = 120
Ns = (6,) if prior == 'a' else (8, 16)
for N, T, k, fq in itertools.product(Ns, (16, 32), (3, 4), (0.75, 0.9, 0.97)):
    torch.manual_seed(5)
    b = C.block_sets(G, N, T, k, (fq, 0.0), cfg, gate=True)
    p = [x.numpy().reshape(G, 2, -1) for x in C.bayes(b, cfg)]
    kl = H.kl_bayes_model(tuple(x[:, 0] for x in p), tuple(x[:, 1] for x in p))
    print(prior, N, T, k, fq, 'KL mean %.3f median %.3f' % (kl.mean(), np.median(kl)), flush=True)
