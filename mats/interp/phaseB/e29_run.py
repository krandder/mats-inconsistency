"""E29 per-net measures (PREREG-B amendment 37) on results/e29_data.npz.
Per net: the predictive mean m for variants 0..9 (negation shifts under the original and a redrawn source identity vector);
for the source-ID family {0, 10..17} and the question-ID family {0, 18..25}, the KL-exact cost over the 9 samples (uncorrected),
the mean KL and the 9-sample log-odds variance.
usage: python e29_run.py SET I0 I1 -> results/e29/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import json, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e29_data as E9
import world as W

torch.set_num_threads(1)


def pred_mean(hd):
    mu, ls, lw = hd
    return (np.exp(lw) * mu).sum(1)


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load(os.environ.get('E29_DATA', 'results/e29_data.npz')); nb = min(6000, len(z['y']))
    D = {k: z[k][:nb] for k in ('y', 'semb', 'qemb', 'o')}
    ystar = D['y'][:, W.QL]
    pts = ystar[:, None] + np.concatenate([[0.0], RU.ED.DELTAS])[None]
    g = z['bmean'][:nb, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:nb], z['bvars'], z['bwts'][:nb], g)
    out_dir = os.path.join(os.environ.get('E29_OUT', 'results/e29'), which); os.makedirs(out_dir, exist_ok=True)
    for row in rows:
        p = os.path.join(out_dir, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row)
        L = np.zeros((nb, E9.NVAR, len(RU.ED.DELTAS) + 1)); hds = {}; m = np.zeros((nb, 10))
        for k in range(E9.NVAR):
            X, valid = W.assemble(D['semb'][:, k], D['qemb'][:, k], D['o'][:, k], D['y'])
            hd = RU.heads(model, kind, X, valid)
            L[:, k, :-1] = RU.ED.logodds(RU.logp(hd, pts)); L[:, k, -1] = RU.score(hd, ystar); hds[k] = hd
            if k < 10:
                m[:, k] = pred_mean(hd)
        out = dict(nb=nb, m=m)
        mass = np.ones(nb)
        for name, idx in (('sid', E9.SID), ('qid', E9.QID)):
            out['t_' + name] = L[:, idx].var(1, ddof=1)
            kl, pc, ms = RU.set_stats([hds[k] for k in idx], lpB, g)
            out['kl_' + name], out['pc_' + name] = kl, pc; mass = np.minimum(mass, ms)
        out['mass'] = mass
        dm = np.stack([m[:, 0] - m[:, 1 + s] for s in range(3)], 1); dm2 = np.stack([m[:, 4 + s] - m[:, 7 + s] for s in range(3)], 1)
        print(which, row['tag'], 'sid %.5f qid %.5f KL %.4f stable %.2f %.0fs' % (out['pc_sid'].mean(), out['pc_qid'].mean(),
              out['kl_sid'].mean(), (dm * dm2).sum() / (dm ** 2).sum(), time.time() - t0), flush=True)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)


if __name__ == '__main__':
    main()
