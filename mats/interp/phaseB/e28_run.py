"""E28 per-net measures (PREREG-B amendment 36) on results/e28_data.npz.
Per net: for each family F (flip_ind, negate_ind, negate_joint), the KL-exact cost over its 9 samples (pc_F, uncorrected), the
mean KL (kl_F) and the 9-sample log-odds variance (t_F); for the base and the three negation pairs, the predictive mean m and
the effective weights w_s = dm/do*_s (autograd, deterministic; clean nets only).
usage: python e28_run.py SET I0 I1 -> results/e28/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import json, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e28_data as E8
import e15_run as R15
import world as W

torch.set_num_threads(1)


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load(os.environ.get('E28_DATA', 'results/e28_data.npz')); nb = min(6000, len(z['y']))
    D = {k: z[k][:nb] for k in ('y', 'semb', 'qemb', 'o')}
    ystar = D['y'][:, W.QL]
    pts = ystar[:, None] + np.concatenate([[0.0], RU.ED.DELTAS])[None]
    g = z['bmean'][:nb, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:nb], z['bvars'], z['bwts'][:nb], g)
    out_dir = os.path.join(os.environ.get('E28_OUT', 'results/e28'), which); os.makedirs(out_dir, exist_ok=True)
    for row in rows:
        p = os.path.join(out_dir, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row)
        L = np.zeros((nb, E8.NVAR, len(RU.ED.DELTAS) + 1)); hds = {}
        out = dict(nb=nb)
        for k in range(E8.NVAR):
            X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
            hd = RU.heads(model, kind, X, valid)
            L[:, k, :-1] = RU.ED.logodds(RU.logp(hd, pts)); L[:, k, -1] = RU.score(hd, ystar); hds[k] = hd
            if k == 0 or k >= E8.NEG0:
                model.requires_grad_(False)
                w, m = R15.eff_w(model, X, valid, 32)
                tag = 'base' if k == 0 else 'neg%d' % (k - E8.NEG0)
                out['w_' + tag], out['m_' + tag] = w, m
        mass = np.ones(nb)
        for f in E8.FAMILIES:
            idx = E8.fam_idx(f)
            out['t_' + f] = L[:, idx].var(1, ddof=1)
            kl, pc, ms = RU.set_stats([hds[k] for k in idx], lpB, g)
            out['kl_' + f], out['pc_' + f] = kl, pc; mass = np.minimum(mass, ms)
        out['mass'] = mass
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], ' '.join('%s %.5f' % (f, out['pc_' + f].mean()) for f in E8.FAMILIES),
              'KL %.4f %.0fs' % (out['kl_flip_ind'].mean(), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
