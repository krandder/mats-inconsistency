"""E51 (PREREG-B amendment 59; CPU, existing nets): is the nets' sign dependence shared across seeds or net-specific, and is the
gain from averaging a net over sign patterns an ensemble effect?
Settings and data (the sign families used before, identical for every net of a setting):
  w1: world 1, E27's first NB = 3,000 bases, flip_ind family (base + 8 independent sign flips of the labelled errors); signed nets
      E37's ctrl (seeds 0-11, acquired ones used); invariant nets E37's mag;
  d1, d2: world 2, E40's 6,000 cost episodes (seed 40,040) and its 8 per-example flip patterns (seed 40,041); signed nets E40's
      ctrl; invariant nets E40's canon and E48's prod.
Per signed net: the predictive means m[k, i] on every family member k = 0..8 and episode i; per-net family statistics (mean KL
over the members and the KL reduction of their normalised geometric mean, e26_run.set_stats). Seed ensembles: for each format,
the normalised geometric mean over the base-input predictives of 9 different nets (5 random 9-subsets of the acquired nets,
numpy seed 51), with the same estimator and grid as the sign family (world 2: the per-episode grid of e40_eval.family_grid built
over every density involved).
usage: python e51_run.py SETTING   (w1 | d1 | d2) -> results/e51/<setting>.npz
"""
import glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e27_data as E7  # noqa: E402
import e30_eval as E30  # noqa: E402
import e30_fmt as F  # noqa: E402
import e40_eval as EV  # noqa: E402
import e48_eval_t2 as ET  # noqa: E402
import t2_world as T  # noqa: E402
import t2lin as L  # noqa: E402

torch.set_num_threads(1)
NB = int(os.environ.get('E51_NB', 3000))                                  # override: plumbing checks only
OUT = os.environ.get('E51_OUT', 'results/e51')                            # override: plumbing checks only
NSUB, KSUB = 5, 9


def pmean(hd):
    mu, ls, lg = hd
    return (np.exp(lg) * mu).sum(1)


def subsets(n):
    r = np.random.default_rng(51)
    return [sorted(r.choice(n, KSUB, replace=False).tolist()) for _ in range(NSUB)]


def main():
    st = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    out = {}
    if st == 'w1':
        z = np.load('results/e27_data.npz'); idx = E7.fam_idx('flip_ind')
        g = z['bmean'][:NB, None] + RU.GRID[None]
        lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
        out['bayes_mean'] = z['bmean'][:NB]
        fam = {}
        for arm in ('ctrl', 'mag'):
            nets = [r for r in sorted((d for d in glob.glob('runs/e37/e37_%s_s*' % arm) if os.path.isdir(d)), key=lambda p: int(p.rsplit('_s', 1)[1]))
                    if float(np.load('results/e37/%s.npz' % os.path.basename(r))['gap']) < 0.10]
            out['%s_seeds' % arm] = np.array([int(r.rsplit('_s', 1)[1]) for r in nets])
            base_hds, M, fk, fp = [], [], [], []
            for r in nets:
                model, a, _ = E30.load(r)
                ks = idx if arm == 'ctrl' else [0]
                hds = []
                for k in ks:
                    se = z['semb'][:NB, k] if z['semb'].ndim == 4 else z['semb'][:NB]
                    qe = z['qemb'][:NB, k] if z['qemb'].ndim == 4 else z['qemb'][:NB]
                    X, v = F.make_X(z['o'][:NB, k], z['y'][:NB], se, qe, a)
                    hds.append(RU.heads(model, 'clean', X, v))
                base_hds.append(hds[0]); M.append(np.stack([pmean(h) for h in hds]))
                if arm == 'ctrl':
                    kl, pc, _ = RU.set_stats(hds, lpB, g); fk.append(kl.mean()); fp.append(pc.mean())
            out['%s_M' % arm] = np.stack(M)
            if arm == 'ctrl':
                out['ctrl_fam_kl'] = np.array(fk); out['ctrl_fam_pc'] = np.array(fp)
            sk, sp = [], []
            for sub in subsets(len(nets)):
                kl, pc, _ = RU.set_stats([base_hds[j] for j in sub], lpB, g); sk.append(kl.mean()); sp.append(pc.mean())
            out['%s_seed_kl' % arm] = np.array(sk); out['%s_seed_pc' % arm] = np.array(sp)
            fam[arm] = base_hds
    else:
        D = int(st[1])
        NF = int(os.environ.get('E51_NF', EV.NF))                          # override: plumbing checks only
        fd = T.sample(EV.NF, np.random.default_rng(40_040), D, 8)
        fd = {k: v[:NF] for k, v in fd.items()}; bay = T.bayes(fd)
        r = np.random.default_rng(40_041)
        pats = [np.ones((EV.NF, 8))] + [np.where(r.random((EV.NF, 8)) < 0.5, -1.0, 1.0) for _ in range(8)]
        pats = [p[:NF] for p in pats]
        out['bayes_mean'] = (bay[2] * bay[0]).sum(1)
        for arm, pat in (('ctrl', 'runs/e40/e40_d%dctrl_s%d'), ('canon', 'runs/e40/e40_d%dcanon_s%d'), ('prod', 'runs/e48/e48_d%dprod_s%d')):
            base_hds, M, fk, fp = [], [], [], []
            for s in range(12):
                model, meta = ET.load(pat % (D, s))
                ps = pats if arm == 'ctrl' else pats[:1]
                hds = [EV.heads(model, *L.tokens(EV.flipped(fd, p), meta['fmt'])) for p in ps]
                base_hds.append(hds[0]); M.append(np.stack([pmean(h) for h in hds]))
                if arm == 'ctrl':
                    gg, lp = EV.family_grid(bay, hds)
                    kl, pc, _ = RU.set_stats(hds, lp, gg); fk.append(kl.mean()); fp.append(pc.mean())
            out['%s_seeds' % arm] = np.arange(12); out['%s_M' % arm] = np.stack(M)
            if arm == 'ctrl':
                out['ctrl_fam_kl'] = np.array(fk); out['ctrl_fam_pc'] = np.array(fp)
            sk, sp = [], []
            for sub in subsets(12):
                hs = [base_hds[j] for j in sub]
                gg, lp = EV.family_grid(bay, hs)
                kl, pc, _ = RU.set_stats(hs, lp, gg); sk.append(kl.mean()); sp.append(pc.mean())
            out['%s_seed_kl' % arm] = np.array(sk); out['%s_seed_pc' % arm] = np.array(sp)
    np.savez(os.path.join(OUT, '%s.npz' % st), **out)
    print(st, {k: v.shape for k, v in out.items()})


if __name__ == '__main__':
    main()
