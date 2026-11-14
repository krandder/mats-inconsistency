"""E64b code dependence and seed ensembles (PREREG-B amendment 72, addendum 1): e62_run.py's estimator (E59's, unchanged) for E64's retrained arms
(split, splitnc). Original E62 docstring follows.
E62 (PREREG-B amendment 70; CPU, existing nets): does given source membership also remove the code dependence and the
run-to-run variation? E59's measures (e59_run.py's estimator, unchanged) for E60's learned within-source pools (own, ownf) and,
for the run-to-run part only, E30's fixed-code nets (fix: the same three source codes in every episode, routing still by content).
Per net (E51/E52's estimator: E27's first NB = 3,000 bases, the +-6 grid around the Bayes mean, e26_run.set_stats):
  base KL (kl);
  own, ownf: the identity-code family (E29's SID family on its first 6,000 bases: the base and 8 redraws of the three source codes;
  e30_eval.family_cost): its mean KL and the KL reduction of geometric averaging over it (code_pc, code_kl). Not for fix, whose
  format replaces every code by the fixed ones (the family is constant there);
  the sign family is taken from the nets' registered evaluations (results/e60, results/e30).
Seed ensembles: normalised geometric means of K different nets of an arm on the base inputs, 5 random K-subsets (numpy seed 62),
K = 7 for every arm and K = 9 where an arm has >= 9 acquired nets; their members' mean KL and the ensemble's KL reduction.
usage: python e64b_codes.py ARM  (split | splitnc) -> results/e64b_codes/<arm>.npz
Overrides (plumbing only): E62_ROOT (prefix for runs/ and results/ of the input nets), E62_NB, E62_NC, E62_OUT.
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64b as P64  # noqa: E402

P2.build = lambda meta: P64.build_any(meta, P2, P2.fin_of(meta['ds']))
import e26_run as RU  # noqa: E402
import e29_data as E9  # noqa: E402
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402

E30.F = F
torch.set_num_threads(1)
NB = int(os.environ.get('E62_NB', 3000)); NC = int(os.environ.get('E62_NC', 6000))
OUT = os.environ.get('E62_OUT', 'results/e64b_codes')
ROOT = os.environ.get('E62_ROOT', '')
NSUB = 5
PATS = dict(split=('runs/e64b/e64_split_s*', 'results/e64b/e64_split_s%d.npz'), splitnc=('runs/e64b/e64_splitnc_s*', 'results/e64b/e64_splitnc_s%d.npz'))


def main():
    arm = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz'); z9 = np.load('results/e29_data.npz')
    sl = slice(0, NB)
    g = z['bmean'][sl, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][sl], z['bvars'], z['bwts'][sl], g)
    se = z['semb'][sl, 0] if z['semb'].ndim == 4 else z['semb'][sl]
    qe = z['qemb'][sl, 0] if z['qemb'].ndim == 4 else z['qemb'][sl]
    rpat, fpat = PATS[arm]
    rpat, fpat = os.path.join(ROOT, rpat), os.path.join(ROOT, fpat)
    runs = sorted((d for d in glob.glob(rpat) if os.path.isdir(d) and os.path.exists(os.path.join(d, 'meta.json'))),
                  key=lambda p: int(p.rsplit('_s', 1)[1]))
    runs = [r for r in runs if os.path.exists(fpat % int(r.rsplit('_s', 1)[1])) and float(np.load(fpat % int(r.rsplit('_s', 1)[1]))['gap']) < 0.10]
    hds, K, CP, CK = [], [], [], []
    for r in runs:
        model, a, _ = E30.load(r)
        X, v = F.make_X(z['o'][sl, 0], z['y'][sl], se, qe, a)
        h = RU.heads(model, 'clean', X, v)
        kl, _, _ = RU.set_stats([h], lpB, g)
        hds.append(h); K.append(kl.mean())
        if arm == 'fix':
            cp, ck = np.nan, np.nan
        else:
            cp, ck, _ = E30.family_cost(model, a, z9, E9.SID, NC)
        CP.append(cp); CK.append(ck)
        print(arm, os.path.basename(r), 'kl %.5f code %.5f / %.5f' % (K[-1], cp, ck), flush=True)
    out = dict(seeds=np.array([int(r.rsplit('_s', 1)[1]) for r in runs]), kl=np.array(K), code_pc=np.array(CP), code_kl=np.array(CK))
    rr = np.random.default_rng(64)
    for k in (7, 9):
        if len(runs) < k:
            continue
        sk, sp, sm = [], [], []
        for _ in range(NSUB):
            sub = sorted(rr.choice(len(runs), k, replace=False).tolist())
            kl, pc, _ = RU.set_stats([hds[j] for j in sub], lpB, g); sk.append(kl.mean()); sp.append(pc.mean()); sm.append(np.mean([K[j] for j in sub]))
        out['ens%d_kl' % k] = np.array(sk); out['ens%d_pc' % k] = np.array(sp); out['ens%d_member_kl' % k] = np.array(sm)
    np.savez(os.path.join(OUT, '%s.npz' % arm), **out)
    print(arm, {k: v.shape for k, v in out.items()})


if __name__ == '__main__':
    main()
