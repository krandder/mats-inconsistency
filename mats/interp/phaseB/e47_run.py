"""E47 per-checkpoint markers (PREREG-B amendment 55): when does a net's final A/B type become visible in training?
For each ctrl-format, width-32 depth-2 run under amendment 45's fixed schedule (E37 ctrl, E43 branches, E44 hybrids, E46 perturbed),
at steps 0, 2,000, 3,000, 5,000 and 10,000 (checkpoints common to all of them):
  kl        test KL (gap) on the first 4,000 of tiny-pfn's test episodes;
  lab, summ block 2's query-row attention mass on the 12 labelled and the 3 summary tokens, and xq (block 1's cross-source
            same-question attention), on the base episodes of the first 1,000 bases of E28's data;
  sign      the KL reduction of geometric averaging over E28's 9-sample independent sign-flip family on those 1,000 bases
            (uncorrected), and that family's mean KL.
usage: python e47_run.py RUNDIR [RUNDIR ...] -> results/e47/<run tag>.npz   (skips runs already done)
"""
import glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e28_data as E8  # noqa: E402
import e37_eval as EV  # noqa: E402
import world as W  # noqa: E402
import pfn as P  # noqa: E402
import pfn2 as P2  # noqa: E402

torch.set_num_threads(1)
STEPS = [0, 2000, 3000, 5000, 10000]
NT, NA = 4000, 1000


def main():
    OUT = os.environ.get('E47_OUT', 'results/e47')                       # override: plumbing checks only
    os.makedirs(OUT, exist_ok=True)
    tz = np.load(os.path.join(EV.TP, 'runs/testset.npz'))
    z = np.load('results/e28_data.npz')
    D = {k: z[k][:NA] for k in ('y', 'semb', 'qemb', 'o')}
    g = z['bmean'][:NA, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NA], z['bvars'], z['bwts'][:NA], g)
    fi = E8.fam_idx('flip_ind')
    Xf = [W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y']) for k in fi]
    for run in [r for d in sys.argv[1:] for r in sorted(glob.glob(os.path.join(d, '*/meta.json')))]:
        run = os.path.dirname(run); tag = os.path.basename(run); p = os.path.join(OUT, tag + '.npz')
        meta = json.load(open(os.path.join(run, 'meta.json')))
        if (os.path.exists(p) or meta.get('fmt', 'ctrl') != 'ctrl' or meta.get('fmt0', 'ctrl') != 'ctrl' or meta.get('width') != 32
                or meta.get('depth') != 2):
            continue
        t0 = time.time(); P2.fin_of = lambda ds: 2 * ds + 4
        model = P2.build(meta); model.eval(); model.requires_grad_(False)
        out = {k: [] for k in ('step', 'kl', 'lab', 'summ', 'xq', 'sign_pc', 'sign_kl')}
        for st in STEPS:
            ck = os.path.join(run, 'ckpt/step_%06d.pt' % st)
            if not os.path.exists(ck):
                continue
            model.load_state_dict(torch.load(ck, map_location='cpu'))
            kl = float((P.head_nll_np(P.predict(model, tz['X'][:NT], tz['valid'][:NT]), tz['ystar'][:NT]) - tz['nll_bayes'][:NT]).mean())
            lab, summ, xq = EV.att_masses(model, *Xf[0])
            fkl, pc, _ = RU.set_stats([RU.heads(model, 'clean', *x) for x in Xf], lpB, g)
            for k, v in (('step', st), ('kl', kl), ('lab', float(lab[-1])), ('summ', float(summ[-1])), ('xq', float(xq)),
                         ('sign_pc', float(pc.mean())), ('sign_kl', float(fkl.mean()))):
                out[k].append(v)
        np.savez(p + '.tmp.npz', **{k: np.array(v) for k, v in out.items()}); os.replace(p + '.tmp.npz', p)
        print(tag, len(out['step']), 'ckpts %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
