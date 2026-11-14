"""How linear is each net? Share of tr Var(output) explained by the best linear (affine) function of the input x,
and by the best quadratic one (R^2 of least squares on 2^16 fresh samples, scored on another 2^16).
usage: python p3_linshare.py FAM N L SIGMA SEEDS OUTJSON"""
import sys, os, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]; OUT = sys.argv[6]
DT = nd.DT
res = {}


def feats(x, quad):
    f = [torch.ones(x.shape[0], 1, dtype=DT), x]
    if quad:
        iu = torch.triu_indices(x.shape[1], x.shape[1])
        f.append((x[:, :, None] * x[:, None, :])[:, iu[0], iu[1]])
    return torch.cat(f, 1)


for seed in SEEDS:
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(17_000_023 + seed)
    xa = torch.randn(1 << 16, N, generator=g, dtype=DT); ya = net.forward(xa, gen=g)
    xb = torch.randn(1 << 16, N, generator=g, dtype=DT); yb = net.forward(xb, gen=g)
    tv = float(yb.var(0).sum())
    out = {}
    for quad in (False, True):
        Fa, Fb = feats(xa, quad), feats(xb, quad)
        coef = torch.linalg.lstsq(Fa, ya).solution
        r = yb - Fb @ coef
        out['quadratic' if quad else 'linear'] = 1 - float(r.var(0).sum()) / tv
    res[seed] = out
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: R2 linear={out['linear']:.3f} quadratic={out['quadratic']:.3f}", flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, res=res), open(OUT, 'w'))
