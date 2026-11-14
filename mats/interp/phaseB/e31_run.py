"""E31 per-checkpoint measures (PREREG-B amendment 39) on E16's replays (base_mix4_s5 = B lineage, base_mix4_s6 = A lineage).
On the first 3,000 bases of E28's data: the KL-exact (uncorrected) cost of independent sign flips and of whole-vector negation
(9 samples each, as E28), their mean KL; and on the base variant the block-1 cross-source same-question attention marker xq (E27).
usage: python e31_run.py SEED PART NPARTS -> results/e31/s<SEED>_step<STEP>.npz
"""
import glob, json, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e27_run as R27
import e28_data as E8
import world as W
import pfn as P

torch.set_num_threads(1)
NB = 3000


def main():
    seed, part, nparts = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    meta = json.load(open(f'runs/base_mix4_s{seed}/meta.json'))
    cks = sorted(glob.glob(f'runs/dyn_base_s{seed}/ckpt/step_*.pt'))[part::nparts]
    z = np.load('results/e28_data.npz')
    D = {k: z[k][:NB] for k in ('y', 'semb', 'qemb', 'o')}
    g = z['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
    os.makedirs('results/e31', exist_ok=True)
    for ck in cks:
        step = int(os.path.basename(ck)[5:11]); p = f'results/e31/s{seed}_step{step:06d}.npz'
        if os.path.exists(p):
            continue
        t0 = time.time()
        model = P.build(meta); model.load_state_dict(torch.load(ck, map_location='cpu')); model.eval(); model.requires_grad_(False)
        hds = {}
        idx = sorted(set(E8.fam_idx('flip_ind') + E8.fam_idx('negate_ind')))
        for k in idx:
            X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
            hds[k] = RU.heads(model, 'clean', X, valid)
        out = dict(step=step)
        for f in ('flip_ind', 'negate_ind'):
            kl, pc, m = RU.set_stats([hds[k] for k in E8.fam_idx(f)], lpB, g)
            out['kl_' + f], out['pc_' + f], out['mass_' + f] = float(kl.mean()), float(pc.mean()), float(m.min())
        X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, 0], D['y'])
        out['xq'] = float(R27.cross_question_attention(model, X, valid))
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(seed, step, 'KL %.4f flip %.5f neg %.5f xq %.4f %.0fs' % (out['kl_flip_ind'], out['pc_flip_ind'], out['pc_negate_ind'],
                                                                     out['xq'], time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
