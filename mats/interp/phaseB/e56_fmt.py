"""E56 input formats for world 1 (PREREG-B amendment 64, planned): e48_fmt.py's arms (unchanged, imported) plus
  sep   the ctrl format with the summary (query-question) tokens' readings moved out of the o column into one extra feature
        column (index 2 ds + 4 = 36): on summary tokens the o column is 0 and the new column holds o*_s; labelled tokens and the
        query token are as in ctrl (new column 0). The o column's input weight then touches only the labelled tokens' values, so
        the LayerNorm cap scale of the labelled errors (lngeom.py) is decoupled from how strongly the summary readings enter.
        Same width as sgnmag, so train_sw.py --pair_init gives every shared parameter the ctrl initialisation of the seed.
The summary readings' feature index (for effective weights): READ_COL[arm] (32 for every other arm, 36 for sep).
"""
import numpy as np
import e48_fmt as F0
import world as W

S, Q, QL = W.S, W.Q, W.QL
ARMS = F0.ARMS + ('sep',)
ids_from_X = F0.ids_from_X
fixed_ids = F0.fixed_ids
READ_COL = {a: 32 for a in ARMS}
READ_COL['sep'] = 36


def fin(arm, ds):
    return 2 * ds + 4 + (1 if arm in ('sgnmag', 'sgnsq', 'sep') else 0)


def make_X(o, y, semb, qemb, arm, ds=16):
    if arm != 'sep':
        return F0.make_X(o, y, semb, qemb, arm, ds)
    X, m = W.assemble(semb, qemb, o, y)
    extra = np.zeros(X.shape[:2] + (1,), dtype=X.dtype)
    for s in range(S):
        t = s * Q + QL                                             # the source's summary token
        extra[:, t, 0] = X[:, t, 2 * ds]
        X[:, t, 2 * ds] = 0.0
    return np.concatenate([X, extra], 2), m
