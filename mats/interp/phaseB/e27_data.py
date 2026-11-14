"""E27 data (PREREG-B amendment 35): which sign and assignment information the nets use.

Every variant keeps the context c (labels, y*, identity vectors), every source's variance level, query-reading error and the
magnitudes |r_{s,q}| of its four labelled residuals, so Bayes is unchanged (it depends only on SS_s and o*_s). Families, each with
R = 8 random draws (the base is the ninth sample of every family):
  flip_ind    r_{s,q} -> eps_{s,q} r_{s,q}, signs independent across sources and questions;
  flip_joint  r_{s,q} -> eps_q r_{s,q}, one sign per question shared by the three sources (keeps every pairwise difference of
              readings on a labelled question up to a common sign, i.e. keeps how far each reading sits from the others);
  perm_ind    each source's four residuals (with their signs) permuted across its labelled questions, independently;
  perm_joint  one permutation of the four labelled questions applied to all three sources' residuals (keeps which residuals sit
              together on a question; changes which label and question identity they sit with).
Variant index: 0 base; 1 + 8 f + j for family f (order above) and draw j.
usage: python e27_data.py -> results/e27_data.npz   (python e27_data.py PATH SEED N for checks)
"""
import os, sys
import numpy as np
import e26_data as ED

S, Q, QL = ED.S, ED.Q, ED.QL
FAMILIES = ['flip_ind', 'flip_joint', 'perm_ind', 'perm_joint']
R = 8
NVAR = 1 + R * len(FAMILIES)
SEED, NBASE = 2026_1029_35, 6000


def fam_idx(f):
    return [0] + [1 + R * FAMILIES.index(f) + j for j in range(R)]


def build(seed, n, ds=16):
    base = ED.build(seed, n, ds)                                             # reuse E26's factor draws; variant 0 = its base
    rng = np.random.default_rng([seed, 27])                                  # separate stream for the redraws
    res0 = base['res'][:, 0]; err0 = base['err'][:, 0]; vidx0 = base['vidx'][:, 0]
    res = np.zeros((n, NVAR, S, QL)); res[:, 0] = res0
    for f in FAMILIES:
        for j in range(R):
            k = 1 + R * FAMILIES.index(f) + j
            if f == 'flip_ind':
                res[:, k] = res0 * rng.choice([-1.0, 1.0], size=(n, S, QL))
            elif f == 'flip_joint':
                res[:, k] = res0 * rng.choice([-1.0, 1.0], size=(n, 1, QL))
            elif f == 'perm_ind':
                perm = np.argsort(rng.random((n, S, QL)), axis=2)
                res[:, k] = np.take_along_axis(res0, perm, axis=2)
            else:
                perm = np.argsort(rng.random((n, 1, QL)), axis=2)
                res[:, k] = np.take_along_axis(res0, np.repeat(perm, S, axis=1), axis=2)
    vidx = np.repeat(vidx0[:, None], NVAR, 1); err = np.repeat(err0[:, None], NVAR, 1)
    o = np.empty((n, NVAR, S, Q))
    o[..., :QL] = base['y'][:, None, None, :QL] + res
    o[..., QL] = base['y'][:, None, None, QL] + err
    return dict(y=base['y'], semb=base['semb'], qemb=base['qemb'], vidx=vidx, res=res, err=err, o=o)


def main():
    p, seed, n = 'results/e27_data.npz', SEED, NBASE
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
    ss = ((D['res'] ** 2).sum(3)); dss = float(np.abs(ss - ss[:, :1]).max())
    print('max |Bayes change| over all variants %.2e; max |SS change| %.2e' % (inv, dss))
    assert inv < 1e-8 and dss < 1e-9
    np.savez(p, LB=LB, deltas=ED.DELTAS, seed=seed, **D, **base)
    print('saved', p)


if __name__ == '__main__':
    main()
