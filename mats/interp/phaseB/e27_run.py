"""E27 per-net measures (PREREG-B amendment 35) on results/e27_data.npz.

For each net and each redraw family F (flip_ind, flip_joint, perm_ind, perm_joint; 9 samples each, the base included):
  t_F    (n, 7)  sample variance (ddof 1) over the family of the log-odds at the six deltas and of the score at y*;
  kl_F, pc_F (n,) mean KL(Bayes || net) over the family and -log Zbar of the geometric mean of the family's densities
                  (uncorrected; x 9/8 in the summary);
  mass   (n,)    smallest grid mass of any density (coverage check).
Layer-1 attention marker (clean nets): xq = mean over labelled tokens (s, q) and heads of the block-1 attention mass on the other
two sources' tokens of the same question q, on the base variant (uniform attention would give 2/15).
usage: python e27_run.py SET I0 I1 -> results/e27/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import json, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e27_data as E7
import world as W
import pfn as P

torch.set_num_threads(1)


def cross_question_attention(model, X, valid, bs=2000):
    tot, cnt = 0.0, 0
    lab = [(s, q) for s in range(3) for q in range(4)]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out = model(torch.from_numpy(X[i:i + bs]), torch.from_numpy(valid[i:i + bs]), want=True)
            a = out[-1][0].numpy()                                            # block-1 attention (B, h, T, T)
            for s, q in lab:
                others = [5 * t + q for t in range(3) if t != s]
                tot += a[:, :, 5 * s + q, others].sum(-1).sum(); cnt += a.shape[0] * a.shape[1]
    return tot / cnt


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load(os.environ.get('E27_DATA', 'results/e27_data.npz')); nb = min(6000, len(z['y']))
    D = {k: z[k][:nb] for k in ('y', 'semb', 'qemb', 'o')}
    ystar = D['y'][:, W.QL]
    pts = ystar[:, None] + np.concatenate([[0.0], RU.ED.DELTAS])[None]
    g = z['bmean'][:nb, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:nb], z['bvars'], z['bwts'][:nb], g)
    LB = z['LB'][:nb]
    out_dir = os.path.join(os.environ.get('E27_OUT', 'results/e27'), which); os.makedirs(out_dir, exist_ok=True)
    for row in rows:
        p = os.path.join(out_dir, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row)
        L = np.zeros((nb, E7.NVAR, len(RU.ED.DELTAS) + 1)); hds = {}
        for k in range(E7.NVAR):
            X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
            hd = RU.heads(model, kind, X, valid)
            L[:, k, :-1] = RU.ED.logodds(RU.logp(hd, pts)); L[:, k, -1] = RU.score(hd, ystar); hds[k] = hd
        out = dict(nb=nb, D0=(L[:, 0] - LB[:, 0]).astype(np.float32))
        mass = np.ones(nb)
        for f in E7.FAMILIES:
            idx = E7.fam_idx(f)
            out['t_' + f] = L[:, idx].var(1, ddof=1)
            kl, pc, m = RU.set_stats([hds[k] for k in idx], lpB, g)
            out['kl_' + f], out['pc_' + f] = kl, pc; mass = np.minimum(mass, m)
        out['mass'] = mass
        if kind == 'clean':
            X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, 0], D['y'])
            out['xq'] = cross_question_attention(model, X, valid)
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], ' '.join('%s %.5f' % (f, out['pc_' + f].mean() * 9 / 8) for f in E7.FAMILIES),
              'KL %.4f xq %.3f %.0fs' % (out['kl_flip_ind'].mean(), out.get('xq', np.nan), time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
