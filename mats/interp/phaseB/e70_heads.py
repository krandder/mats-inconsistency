"""E70 head measures (PREREG-B amendment 78; a copy of e69_heads.py writing to results/e70_heads): E63's block-1 summary-head measures (e63_run.py's functions, head-count agnostic) for the
4-head nets of runs/e69 (best-validation model.pt), on E27's first 2,000 bases: per head P_h, C_h (self + other summaries), osum_h,
olab_h, the mean norm-scaled own logit L_own_h and signed coefficient K_h over the source's own labelled keys (for every head).
usage: python e69_heads.py RUN [RUN ...] -> results/e69_heads/<tag>.json
"""
import json, os, sys
import numpy as np

import e63_run as E63

S, Q, QL = E63.S, E63.Q, E63.QL
NB = 2000
OUT = 'results/e70_heads'


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz'); sl = slice(0, NB)
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    for run in sys.argv[1:]:
        model, arm, meta = E63.load(run, None)
        X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
        A, qh, blk = E63.block1_parts(model, X); H = A.shape[1]
        P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1)
        slf = np.stack([A[:, :, E63.SUMM[s], E63.SUMM[s]] for s in range(S)], 1)
        osum = np.stack([A[:, :, E63.SUMM[s], [E63.SUMM[t] for t in range(S) if t != s]].sum(-1) for s in range(S)], 1)
        olab = np.stack([A[:, :, E63.SUMM[s], [j for t in range(S) if t != s for j in E63.LABS[t]]].sum(-1) for s in range(S)], 1)
        L, ratio, odd = E63.h3_logits(model, X, qh, blk)
        Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
        rat = np.stack([ratio[:, own_i[s]] for s in range(S)], 1)
        o = dict(tag=os.path.basename(run), heads=H, P_h=[float(P[:, :, h].mean()) for h in range(H)], self_h=[float(slf[:, :, h].mean()) for h in range(H)],
                 osum_h=[float(osum[:, :, h].mean()) for h in range(H)], olab_h=[float(olab[:, :, h].mean()) for h in range(H)],
                 L_own_h=[float(Lown[:, :, h].mean()) for h in range(H)], K_h=[float(K[:, :, h].mean()) for h in range(H)],
                 dlogit_r3_h=[float((Lown[:, :, h] * (rat - 1.0)).mean()) for h in range(H)])
        o['C_h'] = [o['self_h'][h] + o['osum_h'][h] for h in range(H)]
        o['pool_heads'] = [h for h in range(H) if o['P_h'][h] >= 0.5]; o['n_pool'] = len(o['pool_heads'])
        o['n_consensus'] = int(sum(c >= 0.5 for c in o['C_h']))
        ks = [o['K_h'][h] for h in o['pool_heads']]
        o['opposite_pair'] = bool(any(a * b < 0 for i, a in enumerate(ks) for b in ks[i + 1:]))
        json.dump(o, open(os.path.join(OUT, o['tag'] + '.json'), 'w'), indent=1)
        print(json.dumps({k: ([round(x, 3) for x in v] if isinstance(v, list) else v) for k, v in o.items()}), flush=True)


if __name__ == '__main__':
    main()
