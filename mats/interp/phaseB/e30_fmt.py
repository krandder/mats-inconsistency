"""E30 input-format arms (PREREG-B amendment 38), shared by train_fmt.py (training on the GPU VM) and e30_eval.py (CPU).
  ctrl    the original format;
  mag     labelled tokens carry y_q + |o_q - y_q| instead of o_q (signs of labelled errors removed: exact sign invariance);
  sgnmag  representation control for mag: the original signed readings plus one extra feature |o_q - y_q| on labelled tokens
          (0 elsewhere), so the magnitude is as directly available as in mag while the signs stay visible;
  fix     fixed identity vectors (seed 424242, same distribution as the per-episode draws) in every episode;
  both    mag and fix.
"""
import numpy as np
import world as W

S, Q, QL = W.S, W.Q, W.QL
ARMS = ('ctrl', 'mag', 'sgnmag', 'fix', 'both')


def fixed_ids(ds):
    r = np.random.default_rng(424242)
    return r.standard_normal((S, ds)) / 4, r.standard_normal((Q, ds)) / 4


def fin(arm, ds):
    return 2 * ds + 4 + (1 if arm == 'sgnmag' else 0)


def transform(o, y, semb, qemb, arm, ds):
    if arm in ('fix', 'both'):
        fs, fq = fixed_ids(ds)
        semb = np.broadcast_to(fs, semb.shape).copy(); qemb = np.broadcast_to(fq, qemb.shape).copy()
    if arm in ('mag', 'both'):
        o = o.copy(); o[:, :, :QL] = y[:, None, :QL] + np.abs(o[:, :, :QL] - y[:, None, :QL])
    return o, semb, qemb


def assemble_fmt(semb, qemb, o, y, arm):
    X, m = W.assemble(semb, qemb, o, y)
    if arm == 'sgnmag':
        extra = np.zeros(X.shape[:2] + (1,), dtype=X.dtype)
        for s in range(S):
            extra[:, s * Q:s * Q + QL, 0] = np.abs(o[:, s, :QL] - y[:, :QL])
        X = np.concatenate([X, extra], 2)
    return X, m


def make_X(o, y, semb, qemb, arm, ds=16):
    o2, s2, q2 = transform(o, y, semb, qemb, arm, ds)
    return assemble_fmt(s2, q2, o2, y, arm)


def ids_from_X(X, ds=16):
    """Recover per-source and per-question identity vectors from standard-format tokens."""
    return X[:, [s * Q for s in range(S)], :ds].astype(np.float64), X[:, :Q, ds:2 * ds].astype(np.float64)
