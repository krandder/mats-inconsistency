"""How Gaussian is the state? (explains where K2 can work). usage: python p3_gauss.py FAM N L SIGMA SEEDS OUTJSON
At layers l in a log grid: mean |skewness| and mean |excess kurtosis| of the pre-activations z_l = W_l h_{l-1}
(coordinates; these are what the ReLU closure sees), over 2^18 inputs; and the same for 64 random unit projections
of the state h_l. Gaussian: 0 and 0 (sampling error ~ sqrt(6/N) and sqrt(24/N))."""
import sys, os, json
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
FAM, N, L, SIGMA = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
SEEDS = [int(s) for s in sys.argv[5].split(',')]; OUT = sys.argv[6]
DT = nd.DT
LAY = [l for l in (1, 2, 4, 8, 16, 32, 64, 128, 256) if l <= L]


def sk(X):
    Xc = X - X.mean(0); s = Xc.pow(2).mean(0).sqrt().clamp_min(1e-300)
    return (Xc.pow(3).mean(0) / s ** 3), (Xc.pow(4).mean(0) / s ** 4 - 3)


res = {}
for seed in SEEDS:
    net = nd.Net(FAM, seed, N, L, SIGMA)
    g = torch.Generator().manual_seed(23_000_041 + seed)
    x = torch.randn(1 << 18, N, generator=g, dtype=DT)
    _, kept = net.forward(x, gen=g, keep=set(LAY))
    P = torch.randn(N, 64, generator=g, dtype=DT); P = P / P.norm(dim=0, keepdim=True)
    rows = []
    for l in LAY:
        h, z = kept[l]
        s1, k1 = sk(z); s2, k2_ = sk(h @ P)
        rows.append(dict(l=l, z_abs_skew=float(s1.abs().mean()), z_abs_exkurt=float(k1.abs().mean()),
                         proj_abs_skew=float(s2.abs().mean()), proj_abs_exkurt=float(k2_.abs().mean())))
    res[seed] = rows
    print(f"{FAM} n={N} L={L} sig={SIGMA:g} seed={seed}: " + ' '.join(
        f"l{r['l']}: z|sk|={r['z_abs_skew']:.2f},|ek|={r['z_abs_exkurt']:.2f} proj|sk|={r['proj_abs_skew']:.2f},|ek|={r['proj_abs_exkurt']:.2f};" for r in rows), flush=True)
json.dump(dict(fam=FAM, n=N, L=L, sigma=SIGMA, res=res), open(OUT, 'w'))
