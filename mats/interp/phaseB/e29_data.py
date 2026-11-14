"""E29 data (PREREG-B amendment 37): is the nets' sign dependence tied to the random identity vectors?

Identity vectors (one per source, one per question; N(0, I/16) per coordinate, redrawn every episode) carry no information, so
Bayes is unchanged when they are redrawn. Variants (all with the base's errors, labels and answer):
  0          base;
  1 + s      base with source s's residual vector negated (s = 0, 1, 2);
  4 + s      base with source s's identity vector redrawn;
  7 + s      base with source s's identity vector redrawn (the same new vector as 4 + s) and its residuals negated;
  10..17     all three source identity vectors redrawn (8 draws)   [source-ID family = {0, 10..17}];
  18..25     all five question identity vectors redrawn (8 draws)  [question-ID family = {0, 18..25}].
Saved: per-variant readings and identity vectors, Bayes's log-odds/score (identical for all variants of a base), the base's Bayes
mixture, and each source's mean labelled residual.
usage: python e29_data.py -> results/e29_data.npz   (python e29_data.py PATH SEED N for checks)
"""
import os, sys
import numpy as np
import e26_data as ED

S, Q, QL = ED.S, ED.Q, ED.QL
R = 8
NVAR = 10 + 2 * R
SID = [0] + list(range(10, 10 + R))
QID = [0] + list(range(10 + R, 10 + 2 * R))
SEED, NBASE = 2026_1029_37, 6000


def build(seed, n, ds=16):
    base = ED.build(seed, n, ds)
    rng = np.random.default_rng([seed, 29])
    res0, err0, y = base['res'][:, 0], base['err'][:, 0], base['y']
    semb = np.repeat(base['semb'][:, None], NVAR, 1).copy()                 # (n, NVAR, S, ds)
    qemb = np.repeat(base['qemb'][:, None], NVAR, 1).copy()                 # (n, NVAR, Q, ds)
    res = np.repeat(res0[:, None], NVAR, 1).copy()
    for s in range(S):
        res[:, 1 + s, s] = -res0[:, s]
        new = rng.standard_normal((n, ds)) / 4
        semb[:, 4 + s, s] = new; semb[:, 7 + s, s] = new; res[:, 7 + s, s] = -res0[:, s]
    for j in range(R):
        semb[:, 10 + j] = rng.standard_normal((n, S, ds)) / 4
        qemb[:, 10 + R + j] = rng.standard_normal((n, Q, ds)) / 4
    o = np.empty((n, NVAR, S, Q))
    o[..., :QL] = y[:, None, None, :QL] + res
    o[..., QL] = y[:, None, None, QL] + err0[:, None]
    vidx = np.repeat(base['vidx'][:, :1], NVAR, 1)
    return dict(y=y, semb=semb, qemb=qemb, vidx=vidx, res=res, o=o)


def main():
    p, seed, n = 'results/e29_data.npz', SEED, NBASE
    if len(sys.argv) > 1:
        p, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    assert not os.path.exists(p), p
    D = build(seed, n)
    ystar = D['y'][:, QL]
    pts = ystar[:, None] + np.concatenate([[0.0], ED.DELTAS])[None]
    LB = np.zeros((n, NVAR, len(ED.DELTAS) + 1))
    for k in range(NVAR):
        means, vars_, wts, (mm, vv) = ED.bayes_parts(ED.episode(D, k))
        LB[:, k, :-1] = ED.logodds(ED.mix_logp_points(means, vars_, wts, pts)); LB[:, k, -1] = ED.mix_score(means, vars_, wts, ystar)
        if k == 0:
            base = dict(bmeans=means, bvars=vars_, bwts=wts, bmean=mm, bvar=vv)
    inv = float(np.abs(LB - LB[:, :1]).max())
    print('max |Bayes change| over all variants %.2e' % inv)
    assert inv < 1e-8
    np.savez(p, LB=LB, rbar=D['res'][:, 0].mean(-1), deltas=ED.DELTAS, seed=seed, **D, **base)
    print('saved', p)


if __name__ == '__main__':
    main()
