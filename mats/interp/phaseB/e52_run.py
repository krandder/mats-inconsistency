"""E52 (PREREG-B amendment 60; CPU, existing nets): does size shrink the shared part of the departure or the part that varies
between trainings? World 1, fixed 90,000-step schedule, ctrl and mag formats, in four configurations:
  w32: width 32, depth 2 (E37's nets, acquired ones: ctrl 11, mag 12);  w64, w128: width 64 / 128, depth 2 (E38, 10 each);
  d3: width 32, depth 3 (E38, 10 each).
Data and estimator as E51's world 1: E27's bases [OFF, OFF + NB) (registered: OFF = 0, NB = 3,000; the base inputs only),
the +-6 grid around the Bayes mean and e26_run.set_stats. Per net: the base predictive (heads) and its mean. Seed ensembles: the
normalised geometric mean of 9 different nets of one configuration and format on the base input, over 5 random 9-subsets of the
acquired nets (numpy seed 52); their members' mean KL and the KL reduction of the ensemble.
usage: python e52_run.py CONFIG  (w32 | w64 | w128 | d3) -> results/e52/<config>.npz
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e30_eval as E30  # noqa: E402
import e30_fmt as F  # noqa: E402

torch.set_num_threads(1)
OFF = int(os.environ.get('E52_OFF', 0))                                  # override: plumbing checks only (other bases)
NB = int(os.environ.get('E52_NB', 3000))
OUT = os.environ.get('E52_OUT', 'results/e52')
NSUB, KSUB = 5, 9
PATS = dict(w32=('runs/e37/e37_%s_s*', 'results/e37/e37_%s_s%d.npz'), w64=('runs/e38/e38_w64%s_s*', 'results/e38/e38_w64%s_s%d.npz'),
            w128=('runs/e38/e38_w128%s_s*', 'results/e38/e38_w128%s_s%d.npz'), d3=('runs/e38/e38_d3%s_s*', 'results/e38/e38_d3%s_s%d.npz'))


def pmean(hd):
    mu, ls, lg = hd
    return (np.exp(lg) * mu).sum(1)


def main():
    cf = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz')
    sl = slice(OFF, OFF + NB)
    g = z['bmean'][sl, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][sl], z['bvars'], z['bwts'][sl], g)
    se = z['semb'][sl, 0] if z['semb'].ndim == 4 else z['semb'][sl]
    qe = z['qemb'][sl, 0] if z['qemb'].ndim == 4 else z['qemb'][sl]
    out = dict(bayes_mean=z['bmean'][sl])
    rpat, fpat = PATS[cf]
    for arm in ('ctrl', 'mag'):
        runs = sorted((d for d in glob.glob(rpat % arm) if os.path.isdir(d)), key=lambda p: int(p.rsplit('_s', 1)[1]))
        runs = [r for r in runs if float(np.load(fpat % (arm, int(r.rsplit('_s', 1)[1])))['gap']) < 0.10]
        hds, M, K = [], [], []
        for r in runs:
            model, a, _ = E30.load(r)
            X, v = F.make_X(z['o'][sl, 0], z['y'][sl], se, qe, a)
            h = RU.heads(model, 'clean', X, v)
            kl, _, _ = RU.set_stats([h], lpB, g)
            hds.append(h); M.append(pmean(h)); K.append(kl.mean())
        out['%s_seeds' % arm] = np.array([int(r.rsplit('_s', 1)[1]) for r in runs])
        out['%s_M' % arm] = np.stack(M); out['%s_kl' % arm] = np.array(K)
        rr = np.random.default_rng(52); sk, sp = [], []
        for _ in range(NSUB):
            sub = sorted(rr.choice(len(runs), KSUB, replace=False).tolist())
            kl, pc, _ = RU.set_stats([hds[j] for j in sub], lpB, g); sk.append(kl.mean()); sp.append(pc.mean())
        out['%s_seed_kl' % arm] = np.array(sk); out['%s_seed_pc' % arm] = np.array(sp)
    np.savez(os.path.join(OUT, '%s.npz' % cf), **out)
    print(cf, {k: v.shape for k, v in out.items()})


if __name__ == '__main__':
    main()
