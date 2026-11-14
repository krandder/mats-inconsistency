"""POST HOC (final-review finding 5): ReLU's closure with the same Mehler machinery and cost convention as the smooth
gates, scored against MC and against RQMC at its own (smaller) budget.

The registered grid K used ReLU's exact bivariate-normal closed form (narrow-deep's gauss_relu, about 2,500 FLOPs per unit
pair with its 96/54 CDF/PDF weights), which made ReLU's N_eq about 1,300 against about 200 for the smooth gates. Here, on
the same nets and the same truth (identical seeds: lib/gnet.py tag 'tgk2', truth stream G.rng(tag, seed, 5)), ReLU uses
Gauss-Hermite (64 nodes) + Mehler order 32, costed like every other gate (a ReLU evaluation = 1 FLOP), and RQMC is rerun
at the largest power of 2 <= its N_eq (16 replicates, scipy's scrambled Sobol: LMS + digital shift).
usage: python3 scratch/k2_relu_mehler.py n [n ...] -> runs/k2_relu_mehler_nN.json
"""
import json, math, os, sys, time
import numpy as np
from scipy.stats import qmc, norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import gnet as G
import k2gen as K2

DEPTHS = [2, 4, 8, 16]
spec = G.Spec('relu', None, 'plain')
herm = K2.Hermite(64, 32)
for n in [int(x) for x in sys.argv[1:]]:
    t0 = time.time()
    out = dict(arm='relu_mehler', n=n, nets={})
    tag = G.tag_of('tgk2', spec.label, n)
    for seed in range(4):
        Ws = G.weights(spec, n, 16, tag, seed)
        g = G.rng(tag, seed, 5); Nt = 1 << 21; bsz = 1 << 15
        T1 = {l: np.zeros(n) for l in DEPTHS}; T2 = {l: np.zeros(n) for l in DEPTHS}

        def fwd(X):
            S1 = {}; S2 = {}; H = X
            for l, (W,) in enumerate(Ws, 1):
                H = np.maximum(H @ W.T.astype(np.float64), 0)
                if l in DEPTHS:
                    S1[l] = H.sum(0); S2[l] = (H * H).sum(0)
            return S1, S2
        for i in range(Nt // bsz):
            S1, S2 = fwd(g.standard_normal((bsz, n)))
            for l in DEPTHS:
                T1[l] += S1[l]; T2[l] += S2[l]
        truth = {l: T1[l] / Nt for l in DEPTHS}
        trvar = {l: float(np.sum(T2[l] / Nt - truth[l] ** 2)) for l in DEPTHS}
        outs, fls = K2.k2_plain('relu', None, Ws, herm, closed_relu=False)
        net = {}
        for l in DEPTHS:
            mcf = K2.mc_flops_plain('relu', n, l)
            Neq = fls[l - 1] / mcf
            net[str(l)] = dict(nmse_k2=float(np.sum((outs[l - 1][0] - truth[l]) ** 2) / trvar[l]), N_eq=Neq,
                               nmse_mc=1.0 / Neq)
        N = 1 << int(math.floor(math.log2(max(net['16']['N_eq'], 2))))
        errs = {l: [] for l in DEPTHS}
        for r in range(16):
            U = qmc.Sobol(d=n, scramble=True, seed=G.tag_of('sobM', n, seed, N, r)).random(N)
            S1, _ = fwd(norm.ppf(np.clip(U, 1e-12, 1 - 1e-12)))
            for l in DEPTHS:
                errs[l].append(float(np.sum((S1[l] / N - truth[l]) ** 2) / trvar[l]))
        for l in DEPTHS:
            net[str(l)].update(N_rqmc=N, nmse_rqmc=float(np.mean(errs[l])), nmse_rqmc_se=float(np.std(errs[l]) / 4))
            x = net[str(l)]
            x['win'] = bool(x['nmse_k2'] < x['nmse_mc'] and x['nmse_k2'] < x['nmse_rqmc'])
        out['nets'][str(seed)] = net
    json.dump(out, open(os.path.join(HERE, '..', 'runs', f'k2_relu_mehler_n{n}.json'), 'w'), indent=1)
    rows = []
    for l in DEPTHS:
        v = [out['nets'][s][str(l)] for s in out['nets']]
        rows.append(f"L{l}: K2/MC {np.median([x['nmse_k2'] * x['N_eq'] for x in v]):.3g} wins {sum(x['win'] for x in v)} "
                    f"N_eq {np.median([x['N_eq'] for x in v]):.0f}")
    print(f'relu_mehler n={n} ' + ' | '.join(rows) + f' ({time.time() - t0:.0f}s)', flush=True)
