"""E68 (PREREG-B amendment 76; CPU, checkpoint weights): how the A allocation forms. For every lineage of E47 (E37's ctrl runs, E43's
branches and new initialisations, E44's hybrids, E46's perturbed runs) at its 10 checkpoints (0, 2k, 3k, 5k, 10k, 20k, 40k, 60k, 80k,
90k), E63's block-1 summary-head measures on E27's first NB = 1,000 bases: per head P_h (own labelled mass), C_h (mass on the three
summaries), and, for every head (pooling or not), the mean norm-scaled own logit L_own_h and the mean signed coefficient K_h over the
source's own labelled keys (e63_run.h3_logits).
usage: python e68_run.py I0 I1   (rows of the sorted lineage list) -> results/e68/<tag>.json
"""
import glob, json, os, sys
import numpy as np
import torch

import e63_run as E63

S, Q, QL = E63.S, E63.Q, E63.QL
STEPS = [0, 2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]
NB = int(os.environ.get('E68_NB', 1000)); OUT = os.environ.get('E68_OUT', 'results/e68')


def main():
    i0, i1 = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz'); sl = slice(0, NB)
    X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], 'ctrl')
    tags = sorted(os.path.basename(f)[:-4] for f in glob.glob('results/e47/*.npz'))[i0:i1]
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    for tag in tags:
        p = os.path.join(OUT, tag + '.json')
        plumb = os.environ.get('E68_PLUMB') == '1'                    # plumbing only: the explored nets, nothing else
        if os.path.exists(p) or ((tag in E63.EXPLORED) != plumb):
            continue
        run = os.path.join('runs', tag.split('_')[0], tag)
        if not os.path.isdir(run):
            continue
        out = {}
        for st in STEPS:
            ck = os.path.join(run, 'ckpt', 'step_%06d.pt' % st)
            if not os.path.exists(ck):
                continue
            model, arm, meta = E63.load(run, st)
            A, qh, blk = E63.block1_parts(model, X); H = A.shape[1]
            P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1)
            C = np.stack([A[:, :, E63.SUMM[s], E63.SUMM].sum(-1) for s in range(S)], 1)
            L, ratio, odd = E63.h3_logits(model, X, qh, blk)
            Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
            out[st] = dict(P_h=[float(P[:, :, h].mean()) for h in range(H)], C_h=[float(C[:, :, h].mean()) for h in range(H)],
                           L_own_h=[float(Lown[:, :, h].mean()) for h in range(H)], K_h=[float(K[:, :, h].mean()) for h in range(H)],
                           K_rms_h=[float(np.sqrt((K[:, :, h] ** 2).mean())) for h in range(H)])
        json.dump(out, open(p + '.tmp', 'w')); os.replace(p + '.tmp', p)
        print(tag, {st: [round(x, 2) for x in o['P_h']] for st, o in out.items()}, flush=True)


if __name__ == '__main__':
    main()
