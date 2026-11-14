"""E59 (PREREG-B amendment 67, planned; CPU, existing nets): why are uniform-pool nets closer to Bayes? World 1, E53's arms (uni,
unif: ctrl format; unif_noln: mag and sq formats) against E37's ctrl nets (acquired nets only).
Per net (E51/E52's estimator: E27's first NB = 3,000 bases, the +-6 grid around the Bayes mean, e26_run.set_stats):
  base KL (kl);
  the identity-code family (E29's SID family on its first 6,000 bases: the base and 8 redraws of the three source codes; e30_eval.
  family_cost): its mean KL and the KL reduction of averaging over it (code_pc, code_kl);
  the sign family (E27's flip_ind) is taken from the nets' registered evaluations (e53_eval.py / e37_eval.py: sign_pc, sign_kl).
Seed ensembles: normalised geometric means of K different nets of an arm on the base inputs, 5 random K-subsets (numpy seed 59),
K = 7 for every arm and K = 9 where an arm has >= 9 acquired nets; their members' mean KL and the ensemble's KL reduction.
usage: python e59_run.py ARM  (ctrl | uni | unif | unifnoln_mag | unifnoln_sq) -> results/e59/<arm>.npz
"""
import glob, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_uni as PU  # noqa: E402

P2.build = lambda meta: PU.build_any(meta, P2, P2.fin_of(meta['ds']))
import e26_run as RU  # noqa: E402
import e29_data as E9  # noqa: E402
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402

E30.F = F
torch.set_num_threads(1)
NB = int(os.environ.get('E59_NB', 3000)); NC = int(os.environ.get('E59_NC', 6000))
OUT = os.environ.get('E59_OUT', 'results/e59')
NSUB = 5
PATS = dict(ctrl=('runs/e37/e37_ctrl_s*', 'results/e37/e37_ctrl_s%d.npz'), uni=('runs/e53/e53_uni_s*', 'results/e53/e53_uni_s%d.npz'),
            unif=('runs/e53/e53_unif_s*', 'results/e53/e53_unif_s%d.npz'),
            unifnoln_mag=('runs/e53/e53_unifnoln_mag_s*', 'results/e53/e53_unifnoln_mag_s%d.npz'),
            unifnoln_sq=('runs/e53/e53_unifnoln_sq_s*', 'results/e53/e53_unifnoln_sq_s%d.npz'))


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
        cp, ck, _ = E30.family_cost(model, a, z9, E9.SID, NC); CP.append(cp); CK.append(ck)
        print(arm, os.path.basename(r), 'kl %.5f code %.5f / %.5f' % (K[-1], cp, ck), flush=True)
    out = dict(seeds=np.array([int(r.rsplit('_s', 1)[1]) for r in runs]), kl=np.array(K), code_pc=np.array(CP), code_kl=np.array(CK))
    rr = np.random.default_rng(59)
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
