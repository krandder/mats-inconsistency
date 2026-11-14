"""Depth scaling of the Gaussian closure on residual narrow nets (Amendment 5).
usage: python p3_resdepth.py FAM N L SIGMA SEEDS LOG2_TRUTH OUTJSON
Per net: truth E[out] by MC (2^LOG2_TRUTH paths); K2 from the input (exact ReLU moments and Stein-linearised) with
its nMSE and cost in MC paths; the simple samplers at the K2 budgets (RQMC, RQMC-reg, MC-reg; R = 16 replicates;
RQMC sizes rounded down to a power of 2); plain MC's nMSE there is 1/N.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd, k2

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]; LOG2 = int(sys.argv[6]); OUT = sys.argv[7]
DT = nd.DT; R = 16
res = {}
for seed in SEEDS:
    t0 = time.time()
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(19_000_031 + seed)
    so = 0; sso = 0; nt = 0
    for i in range(1 << max(LOG2 - 16, 0)):
        o = net.forward(torch.randn(1 << 16, N, generator=g, dtype=DT), gen=g)
        so = so + o.sum(0); sso = sso + o.pow(2).sum(0); nt += 1 << 16
    truth = so / nt; trv = float((sso / nt - truth ** 2).sum())
    fp = net.flops_path()
    row = dict(trvar=trv, mean_sq=float(truth.pow(2).sum()), truth_paths=nt, flops_path=fp)
    for ex, tag in ((True, 'k2'), (False, 'k2_st')):
        mu, C, fl = k2.prop(net, torch.zeros(N, dtype=DT), torch.eye(N, dtype=DT), 0, L, exact=ex)
        row[tag] = dict(nmse=float(((mu - truth) ** 2).sum() / trv), flops=fl, paths=fl / fp)
    for tag in ('k2', 'k2_st'):
        Np = max(16, int(row[tag]['paths']))
        Nq = 1 << int(math.floor(math.log2(Np)))
        errs = {'RQMC': [], 'RQMC-reg': [], 'MC-reg': []}
        for rep in range(R):
            sob = torch.quasirandom.SobolEngine(N, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
            xq = torch.special.ndtri(sob.draw(Nq, dtype=DT).clamp(1e-12, 1 - 1e-12)); yq = net.forward(xq, gen=g)
            errs['RQMC'].append(float(((yq.mean(0) - truth) ** 2).sum() / trv))
            Bq = torch.linalg.lstsq(xq - xq.mean(0), yq - yq.mean(0)).solution.T
            errs['RQMC-reg'].append(float(((yq.mean(0) - Bq @ xq.mean(0) - truth) ** 2).sum() / trv))
            x = torch.randn(Np, N, generator=g, dtype=DT); y = net.forward(x, gen=g)
            Bc = torch.linalg.lstsq(x - x.mean(0), y - y.mean(0)).solution.T
            errs['MC-reg'].append(float(((y.mean(0) - Bc @ x.mean(0) - truth) ** 2).sum() / trv))
        row[tag]['at_budget'] = dict(N_mc=Np, mc=1.0 / Np, N_rqmc=Nq, **{k: float(np.mean(v)) - 1.0 / nt for k, v in errs.items()})
    res[seed] = row
    b = row['k2_st']['at_budget']; b2 = row['k2']['at_budget']
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: K2 nMSE={row['k2']['nmse']:.3g} ({row['k2']['paths']:.0f} paths; MC {b2['mc']:.2g}, "
          f"RQMC {b2['RQMC']:.2g}, RQMC-reg {b2['RQMC-reg']:.2g}, MC-reg {b2['MC-reg']:.2g}) | Stein K2 nMSE={row['k2_st']['nmse']:.3g} "
          f"({row['k2_st']['paths']:.0f} paths; MC {b['mc']:.2g}, RQMC {b['RQMC']:.2g}, RQMC-reg {b['RQMC-reg']:.2g}, MC-reg {b['MC-reg']:.2g}) "
          f"[truth {nt} paths; {time.time()-t0:.0f}s]", flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, res=res), open(OUT, 'w'))
