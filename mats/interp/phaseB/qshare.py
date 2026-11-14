"""E6 (PREREG-B amendment 13): shared vs seed-specific deviation across independently trained 2L sigma = 0.05 nets.

usage: python qshare.py dev RUN_DIR [--fit]   -> RUN_DIR/share_test.npz (8,000 test episodes, K = 64, 2 replicates)
                                                 (--fit also: RUN_DIR/qsys_dev.npz-style fit deviations, 65,536 episodes)
       python qshare.py stats                 -> results/share.json
"""
import json, math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import devpred as DP
import qsys as QS

torch.set_num_threads(1)
FEATS = ('mean', 'logsd', 'kl')
MAIN = [('L2_s0.05', 4), ('L2seed1_s0.05', 1), ('L2seed2_s0.05', 2), ('L2ws0_s0.05', 0), ('L2ws3_s0.05', 3)]
CROSS = [('L2mx0_s0.05', 0), ('L2mx1_s0.05', 1), ('L2mx4_s0.05', 4)]
N_TEST = 8000


def test_set():
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:N_TEST] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    return Tz['X'][:N_TEST], Tz['valid'][:N_TEST], Tt


def dev(run_dir, fit=False):
    t0 = time.time(); model, meta = N.load_run(run_dir); model.requires_grad_(False); sg = meta['sigma']
    X, v, Tt = test_set()
    path = os.path.join(run_dir, 'share_test.npz')
    if not os.path.exists(path):
        d1 = QS.net_dev(model, X, v, Tt, sg, 64, 1); d2 = QS.net_dev(model, X, v, Tt, sg, 64, 7)
        np.savez(path, **{'r1_' + k: d1[k] for k in FEATS}, **{'r2_' + k: d2[k] for k in FEATS})
    print('test deviations %.0fs' % (time.time() - t0), flush=True)
    if fit and not os.path.exists(os.path.join(run_dir, 'qsys_dev.npz')):
        rng = np.random.default_rng(99_500); d = W.sample_np(65536, rng); Xf, vf = W.tokens_np(d, rng, meta['ds']); T = QS.tt(d)
        devF = QS.net_dev(model, Xf, vf, T, sg, 64, 31)
        np.savez(os.path.join(run_dir, 'qsys_dev.npz'), **{'F_' + k: devF[k] for k in FEATS})
        print('fit deviations %.0fs' % (time.time() - t0), flush=True)
    print('done', time.time() - t0, flush=True)


def stats(B=1000):
    R = {}
    for nm, _ in MAIN + CROSS:
        p = os.path.join('runs', nm, 'share_test.npz')
        if os.path.exists(p):
            z = np.load(p); R[nm] = {k: (z['r1_' + k], z['r2_' + k]) for k in FEATS}
    idx = {nm: i for nm, i in MAIN + CROSS}
    main = [nm for nm, _ in MAIN if nm in R]; cross = [nm for nm, _ in CROSS if nm in R]
    pairs_main = [(a, b) for i, a in enumerate(main) for b in main[i + 1:]]
    pairs_cross = [(a, b) for a in cross for b in main if idx[a] != idx[b]] + [(a, b) for i, a in enumerate(cross) for b in cross[i + 1:]]
    rng = np.random.default_rng(0); n = N_TEST; boots = [rng.integers(0, n, n) for _ in range(B)]
    def cov(a, b, ii=None):
        if ii is not None: a, b = a[ii], b[ii]
        return float(np.mean((a - a.mean()) * (b - b.mean())))
    def rho(pairs, k, ii=None):
        vals = []
        for a, b in pairs:
            Va = cov(R[a][k][0], R[a][k][1], ii); Vb = cov(R[b][k][0], R[b][k][1], ii)
            cab = 0.5 * (cov(R[a][k][0], R[b][k][1], ii) + cov(R[a][k][1], R[b][k][0], ii))   # independent noise draws across nets
            vals.append(cab / math.sqrt(max(Va, 1e-30) * max(Vb, 1e-30)))
        return vals
    out = dict(n_test=n, nets=list(R), pairs_main=pairs_main, pairs_cross=pairs_cross, features={})
    for k in FEATS:
        res = {}
        for tag, pairs in (('main', pairs_main), ('cross', pairs_cross)):
            if not pairs: continue
            vals = rho(pairs, k); bs = [float(np.mean(rho(pairs, k, ii))) for ii in boots]
            res[tag] = dict(mean=float(np.mean(vals)), ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                            pair_min=float(np.min(vals)), pair_max=float(np.max(vals)), per_pair={f'{a}~{b}': v for (a, b), v in zip(pairs, vals)})
        # per-net reliability (noise-free share of the measured variance) and floor
        res['reliability'] = {nm: cov(R[nm][k][0], R[nm][k][1]) / float(np.var(R[nm][k][0])) for nm in R}
        res['floor'] = {nm: DP.r2(R[nm][k][0], R[nm][k][1]) for nm in R}
        out['features'][k] = res
        print(k, {t: (round(res[t]['mean'], 3), [round(c, 3) for c in res[t]['ci']], round(res[t]['pair_min'], 3), round(res[t]['pair_max'], 3)) for t in ('main', 'cross') if t in res}, flush=True)
    # E5 named-feature fractions on the latent scale (E5 used the first 2,000 test episodes)
    out['e5_named'] = {}
    for nm, _ in MAIN[:3]:
        q = json.load(open(os.path.join('runs', nm, 'qsys.json')))
        z = np.load(os.path.join('runs', nm, 'share_test.npz'))
        out['e5_named'][nm] = {}
        for k in FEATS:
            a, b = z['r1_' + k][:2000], z['r2_' + k][:2000]
            rel = cov(a, b) / float(np.var(a))
            out['e5_named'][nm][k] = dict(R2=q['systematic'][k]['R2'], registered_fraction=q['systematic'][k]['fraction'], latent_fraction=q['systematic'][k]['R2'] / rel, reliability_2000=rel)
    json.dump(out, open('results/share.json', 'w'), indent=1)
    print(json.dumps(out['e5_named'], indent=0)[:1500])


if __name__ == '__main__':
    if sys.argv[1] == 'dev':
        dev(sys.argv[2], fit='--fit' in sys.argv)
    else:
        stats()
