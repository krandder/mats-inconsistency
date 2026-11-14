"""Grid K (PREREG): K2 Gaussian closure against MC and RQMC at equal FLOPs, random plain nets, one (arm, width).

usage: python3 scratch/k2_grid.py NAME P n [--seeds 0,1,2,3] [--ntruth 21]   (P = 'none' if no parameter)
Per net (lib/gnet.py, pilot-normalised, depth 16; depths 2, 4, 8, 16 are truncations):
- truth: plain MC with 2^ntruth Gaussian inputs (float64), mean and tr Var of the output at each depth;
- K2 (lib/k2gen.py): Gauss-Hermite Q from k2gen.q_for (64; softplus beta 4/8/16: 128/256/512; smoothstep 200), Mehler order 32; ReLU closed form, plus Mehler-ReLU as a
  secondary (post hoc label "relu_mehler");
- N_eq = FLOPs(K2) / FLOPs(one MC sample); MC's nMSE = 1/N_eq;
- RQMC: scrambled Sobol (scipy, Owen), inverse normal CDF, N in {2^floor(log2 N_eq), 2^10, 2^14}, 16 replicates.
Writes runs/k2_LABEL_nN.json.
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy.stats import qmc, norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import gnet as G
import k2gen as K2

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('p'); ap.add_argument('n', type=int)
ap.add_argument('--seeds', default='0,1,2,3'); ap.add_argument('--ntruth', type=int, default=21)
ap.add_argument('--bias_frac', type=float, default=0.0)
a = ap.parse_args()
p = None if a.p == 'none' else float(a.p)
spec = G.Spec(a.name, p, 'plain', bias_frac=a.bias_frac)
n = a.n; LMAX = 16; DEPTHS = [2, 4, 8, 16]
herm = K2.Hermite(K2.q_for(a.name, p), 32)


def forward_means(Ws, X, bs=None):
    """Sum and sum of squares of the output at each depth in DEPTHS, rows of X are inputs."""
    S1 = {}; S2 = {}
    H = X
    for l, (W,) in enumerate(Ws, 1):
        H = spec.phi(H @ W.T.astype(np.float64) + (0 if bs is None else bs[l - 1].astype(np.float64)[None, :]))
        if l in DEPTHS:
            S1[l] = H.sum(0); S2[l] = (H * H).sum(0)
    return S1, S2


OUTF = os.path.join(HERE, '..', 'runs', f'k2_{spec.label}_n{n}.json')
LOCK = OUTF[:-5] + '.lock'
try:                                      # two queue workers may share one job list: skip done or claimed jobs
    if os.path.exists(OUTF):
        raise FileExistsError
    os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL))
except FileExistsError:
    print('SKIP (done or claimed)', OUTF, flush=True); sys.exit(0)
res = dict(arm=spec.label, n=n, g2=spec.g2, chi=spec.chi, nets={})
t0 = time.time()
for seed in [int(s) for s in a.seeds.split(',')]:
    tag = G.tag_of('tgk2', spec.label, n)
    Ws = G.weights(spec, n, LMAX, tag, seed)
    bs = G.biases(spec, n, LMAX, tag, seed)
    # truth
    Nt = 1 << a.ntruth; bsz = 1 << 15
    g = G.rng(tag, seed, 5)
    T1 = {l: np.zeros(n) for l in DEPTHS}; T2 = {l: np.zeros(n) for l in DEPTHS}
    for i in range(Nt // bsz):
        S1, S2 = forward_means(Ws, g.standard_normal((bsz, n)), bs)
        for l in DEPTHS:
            T1[l] += S1[l]; T2[l] += S2[l]
    truth = {l: T1[l] / Nt for l in DEPTHS}
    trvar = {l: float(np.sum(T2[l] / Nt - truth[l] ** 2)) for l in DEPTHS}
    # K2
    outs, fls = K2.k2_plain(a.name, p, Ws, herm, closed_relu=True, bs=bs)
    extra = None
    if a.name == 'relu':
        extra = K2.k2_plain(a.name, p, Ws, herm, closed_relu=False)
    net = {}
    for l in DEPTHS:
        mcf = K2.mc_flops_plain(a.name, n, l, bias=bs is not None)
        Neq = fls[l - 1] / mcf
        e_k2 = float(np.sum((outs[l - 1][0] - truth[l]) ** 2) / trvar[l])
        row = dict(trvar=trvar[l], mean_sq=float(np.sum(truth[l] ** 2)), k2_flops=fls[l - 1], mc_flops_per_sample=mcf,
                   N_eq=Neq, nmse_mc=1.0 / Neq, nmse_k2=e_k2, truth_nmse_floor=1.0 / Nt)
        if extra is not None:
            row['nmse_k2_relu_mehler'] = float(np.sum((extra[0][l - 1][0] - truth[l]) ** 2) / trvar[l])
            row['N_eq_relu_mehler'] = extra[1][l - 1] / mcf
        net[str(l)] = row
    # RQMC at several N (shared across depths: one pass gives all depths)
    Neq_p2 = 1 << int(math.floor(math.log2(max(net['16']['N_eq'], 2))))
    for N in sorted({Neq_p2, 1 << 10, 1 << 14}):
        errs = {l: [] for l in DEPTHS}
        for r in range(16):
            sob = qmc.Sobol(d=n, scramble=True, seed=G.tag_of('sob', spec.label, n, seed, N, r))
            U = sob.random(N)
            X = norm.ppf(np.clip(U, 1e-12, 1 - 1e-12))
            S1, _ = forward_means(Ws, X, bs)
            for l in DEPTHS:
                errs[l].append(float(np.sum((S1[l] / N - truth[l]) ** 2) / trvar[l]))
        for l in DEPTHS:
            net[str(l)][f'nmse_rqmc_{N}'] = float(np.mean(errs[l]))
            net[str(l)][f'nmse_rqmc_{N}_se'] = float(np.std(errs[l]) / 4)
            if N == Neq_p2:
                net[str(l)]['N_rqmc_eq'] = N
                net[str(l)]['nmse_rqmc_eq'] = float(np.mean(errs[l]))
    net['preact_rms'] = G.preact_std(spec, Ws, n, N=8192, seed=seed, bs=bs)
    res['nets'][str(seed)] = net
    print(f'{spec.label} n={n} seed={seed} done ({time.time() - t0:.0f}s)', flush=True)
json.dump(res, open(os.path.join(HERE, '..', 'runs', f'k2_{spec.label}_n{n}.json'), 'w'), indent=1)
print('WROTE', f'k2_{spec.label}_n{n}.json', round(time.time() - t0, 1), flush=True)
