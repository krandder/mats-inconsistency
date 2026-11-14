"""E42 per-checkpoint measures (PREREG-B amendment 50): do E31's and E35's single-lineage dynamics generalise across lineages?
For E37's ctrl arm (runs/e37/e37_ctrl_s<seed>/ckpt), at steps 3,000, 10,000, 12,000, 15,000, 20,000, 25,000, 30,000, 40,000,
60,000 and 90,000, on the first 3,000 bases of E28's data, exactly as E31: the KL-exact (uncorrected) cost of independent sign
flips and of whole-vector negation (9 samples each) and their mean KL; on the base variant, E27's block-1 marker xq; and E35's
block-2 query-row attention mass on the 12 labelled tokens (lab) and on the 3 summary tokens (summ).
usage: python e42_run.py SEED [SEED ...] -> results/e42/s<SEED>_step<STEP>.npz
"""
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import e26_run as RU  # noqa: E402
import e27_run as R27  # noqa: E402
import e28_data as E8  # noqa: E402
import e37_eval as EV  # noqa: E402
import world as W  # noqa: E402
import pfn2 as P2  # noqa: E402

torch.set_num_threads(1)
NB = 3000
STEPS = [3000, 10000, 12000, 15000, 20000, 25000, 30000, 40000, 60000, 90000]


def main():
    z = np.load('results/e28_data.npz')
    D = {k: z[k][:NB] for k in ('y', 'semb', 'qemb', 'o')}
    g = z['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
    fi, ni = E8.fam_idx('flip_ind'), E8.fam_idx('negate_ind')
    os.makedirs(os.environ.get('E42_OUT', 'results/e42'), exist_ok=True)
    steps = [int(x) for x in os.environ['E42_STEPS'].split(',')] if os.environ.get('E42_STEPS') else STEPS   # override: checks only
    for seed in [int(s) for s in sys.argv[1:]]:
        run = os.environ.get('E42_RUN', 'runs/e37/e37_ctrl_s%d') % seed          # override: plumbing checks only
        meta = json.load(open(os.path.join(run, 'meta.json'))); assert meta.get('fmt', 'ctrl') == 'ctrl'
        P2.fin_of = lambda ds: 2 * ds + 4
        for step in steps:
            p = os.path.join(os.environ.get('E42_OUT', 'results/e42'), 's%d_step%06d.npz' % (seed, step))
            if os.path.exists(p):
                continue
            t0 = time.time()
            model = P2.build(meta); model.load_state_dict(torch.load(os.path.join(run, 'ckpt/step_%06d.pt' % step), map_location='cpu'))
            model.eval(); model.requires_grad_(False)
            hds = {}
            for k in sorted(set(fi + ni)):
                X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
                hds[k] = RU.heads(model, 'clean', X, valid)
            out = dict(step=step, seed=seed)
            for name, idx in (('flip', fi), ('neg', ni)):
                kl, pc, mass = RU.set_stats([hds[k] for k in idx], lpB, g)
                out[name + '_pc'] = float(pc.mean()); out[name + '_kl'] = float(kl.mean()); out[name + '_mass'] = float(mass.min())
            X0, v0 = W.assemble(D['semb'], D['qemb'], D['o'][:, 0], D['y'])
            out['xq'] = float(R27.cross_question_attention(model, X0, v0))
            lab, summ, _ = EV.att_masses(model, X0, v0)
            out['lab'], out['summ'] = float(lab[-1]), float(summ[-1])
            np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
            print(seed, step, {k: round(v, 5) for k, v in out.items() if isinstance(v, float)}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
