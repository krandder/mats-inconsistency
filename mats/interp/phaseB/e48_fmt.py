"""E48 input formats for world 1 (PREREG-B amendment 56): e30_fmt.py's arms (unchanged, imported) plus two arms whose labelled
tokens carry the per-token sufficient statistic r^2 = (o - y)^2 LINEARLY (exact Bayes depends on a source's labelled errors only
through sum_q r_q^2):
  sq      labelled tokens carry y_q + r_q^2 instead of o_q (no signs; same width and layout as ctrl and mag, so a seed's
          initialisation and episodes are exactly those of its ctrl and mag nets; sq differs from mag only in |r| -> r^2);
  sgnsq   the original signed readings plus one extra feature r_q^2 on labelled tokens (0 elsewhere); the layout of sgnmag, so
          with train_sw.py --pair_init a seed's initialisation is exactly that of its E39 sgp (signs + |r|) net, and sgnsq
          differs from sgp only in |r| -> r^2 in the extra feature.
Query-question (summary) tokens and the query token are unchanged in every arm.
"""
import numpy as np
import e30_fmt as F0
import world as W

S, Q, QL = W.S, W.Q, W.QL
ARMS = F0.ARMS + ('sq', 'sgnsq')
ids_from_X = F0.ids_from_X
fixed_ids = F0.fixed_ids


def fin(arm, ds):
    return 2 * ds + 4 + (1 if arm in ('sgnmag', 'sgnsq') else 0)


def make_X(o, y, semb, qemb, arm, ds=16):
    if arm in F0.ARMS:
        return F0.make_X(o, y, semb, qemb, arm, ds)
    r2 = (o[:, :, :QL] - y[:, None, :QL]) ** 2                       # (n, S, QL): per-token sufficient statistic
    if arm == 'sq':
        o2 = o.copy(); o2[:, :, :QL] = y[:, None, :QL] + r2
        return W.assemble(semb, qemb, o2, y)
    X, m = W.assemble(semb, qemb, o, y)                              # sgnsq
    extra = np.zeros(X.shape[:2] + (1,), dtype=X.dtype)
    for s in range(S):
        extra[:, s * Q:s * Q + QL, 0] = r2[:, s]
    return np.concatenate([X, extra], 2), m
