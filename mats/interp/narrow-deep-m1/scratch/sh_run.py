"""Practical arm SH (NOT sample-free): RQMC on layers 1..c_h, then the deterministic two-sided downstream.
usage: python sh_run.py SEED SIGMA TRUTHJSON OUTJSON [C_H=8] [C_D=32] [K=4] [G=10] [X=1.5]
Variants per budget N_eq (MC paths' FLOPs):
- SH-direct: each sampled state at c_h pushed noiselessly to c_d, then the deep value function (J_ref + K2 noise table);
- SH-grid: each sampled state's relevant coordinates looked up on the V_{c_h} grid (k directions, radius exact).
The downstream (deep part, directions, grid) is built once per replicate from a pilot of 256 shallow states (its
cost and the pilot's are charged to every estimate). Shallow inputs are scrambled Sobol (largest power of 2 that fits),
noise iid. nMSE minus the truth's 1/P.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf

torch.set_num_threads(1)
SEED, SIG, TJ, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
CH_ = int(sys.argv[5]) if len(sys.argv) > 5 else 8
CD = int(sys.argv[6]) if len(sys.argv) > 6 else 32
K = int(sys.argv[7]) if len(sys.argv) > 7 else 4
G = int(sys.argv[8]) if len(sys.argv) > 8 else 10
X = float(sys.argv[9]) if len(sys.argv) > 9 else 1.5
DT = nd.DT
T = json.load(open(TJ))
net = nd.Net('plain', SEED, T['n'], T['L'], SIG, r=T['r'])
n, L = net.n, net.L
truth = torch.tensor(T['mean'], dtype=DT); trv = T['trvar']; tse = 1.0 / T['paths']
fp = net.flops_path()
LOGS = [9, 11, 13, 15, 17]
REPS = {9: 32, 11: 32, 13: 16, 15: 8, 17: 4}
g = torch.Generator().manual_seed(81_000_011 + 97 * SEED + int(round(SIG * 1e6)))
t0 = time.time()
f_shallow = nd.Net.flops_path(net, L=CH_)                       # input draw + c_h layers with noise


def shallow(x):
    h = x
    for l in range(CH_):
        a = torch.relu(h @ net.W[l].T)
        h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if SIG > 0 else a
    return h


def build(pilot):
    fl = bvf.Flops()
    m = pilot.mean(0); M2 = pilot.T @ pilot / len(pilot); Cov = M2 - torch.outer(m, m)
    fl.add(len(pilot) * 3 * n * n)
    cand = bvf.noiseless_pass(net, pilot[:64], CH_, CD, fl)
    deep = bvf.Deep(net, CD, cand, torch.ones(len(cand), dtype=DT), fl)
    probes = pilot[:64]; pw = torch.ones(len(probes), dtype=DT)
    U8, eS = bvf.relevant_directions(net, CH_, CD, deep, probes, pw, None, Cov, n, fl)
    U = U8[:, :K]
    Pp = torch.eye(n, dtype=DT) - U @ U.T
    B = Pp @ M2 @ U @ torch.linalg.inv(U.T @ M2 @ U)
    grid = bvf.Grid(net, CH_, CD, deep, U, B, U.T @ m, G, X, fl)
    return deep, grid, fl.n


res = dict(seed=SEED, sigma=SIG, c_h=CH_, c_d=CD, k=K, G=G, X=X, est={})
for lg in LOGS:
    Bud = (1 << lg) * fp; R = REPS[lg]
    errs = {'SH-direct': []}; info = {}
    for rep in range(R):
        pilot = shallow(torch.randn(256, n, generator=g, dtype=DT))
        deep, grid, f_build = build(pilot)
        f_setup = 256 * f_shallow + f_build
        per_direct = f_shallow + (CD - CH_) * (bvf.mmf(1, n, n) + n) + (bvf.mmf(1, n, n) + 2 * n + 40 + 3 * n)
        per_grid = f_shallow + bvf.mmf(1, n, K) + (bvf.mmf(1, K, K) + 3 * K + (2 ** (K - 1)) * (2 * n + 2 * K)) + \
            (bvf.mmf(1, n, n) + 2 * n + 40 + 3 * n)
        for var, per in (('SH-direct', per_direct),):
            Ns = (Bud - f_setup) / per
            if Ns < 64:
                continue
            Np = 1 << int(math.floor(math.log2(Ns)))
            sob = torch.quasirandom.SobolEngine(n, scramble=True, seed=int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
            xq = torch.special.ndtri(sob.draw(Np, dtype=DT).clamp(1e-15, 1 - 1e-15))
            tot = torch.zeros(n, dtype=DT)
            for b in range(0, Np, 1 << 16):
                h = shallow(xq[b:b + (1 << 16)])
                if var == 'SH-direct':
                    hd = bvf.noiseless_pass(net, h, CH_, CD, bvf.Flops())
                    V = deep.value(hd, bvf.Flops())
                else:
                    V, _ = grid.value(h @ grid.U, bvf.Flops())
                tot += V.sum(0)
            est = tot / Np
            errs[var].append(float((est - truth).pow(2).sum() / trv))
            info[var] = dict(N_samples=Np, setup_paths=f_setup / fp, per_sample_paths=per / fp)
    res['est'][lg] = {k: dict(nmse=float(np.mean(v)) - tse, se=float(np.std(v) / math.sqrt(len(v))), R=len(v), raw=v, **info.get(k, {}))
                      for k, v in errs.items() if v}
    print(f"s{SEED} sig{SIG:g} N=2^{lg}: " + ' '.join(f"{k}={v['nmse']:.3g} (n={v['N_samples']})" for k, v in res['est'][lg].items()) +
          f" ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(OUT, 'w'))
