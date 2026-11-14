"""E28 data (PREREG-B amendment 36): is the nets' sign dependence odd (a mean-error reading) or even (a sign-pattern reading)?

Every variant keeps the context, every source's level, query error and residual magnitudes, so Bayes is unchanged.
Families (8 draws each; the base is the ninth sample):
  flip_ind      independent sign flips of every labelled residual (as E27): odd and even sign effects;
  negate_ind    each source's whole residual vector multiplied by its own random sign: keeps every within-source sign pattern
                up to a global sign, so only effects that are odd under negating a source's errors (e.g. its mean error) change;
  negate_joint  all three sources' residual vectors multiplied by one common random sign.
Negation pairs: variants 25 + s = the base with source s's residual vector negated (s = 0, 1, 2), for the mean-shift regression.
Variant index: 0 base; 1 + 8 f + j for family f and draw j; 25 + s negation pairs.
Saved with Bayes's log-odds/score for every variant, Bayes's predictive mean and effective weights dm/do*_s of the base, and each
source's mean labelled residual.
usage: python e28_data.py -> results/e28_data.npz   (python e28_data.py PATH SEED N for checks)
"""
import os, sys
import numpy as np
import e26_data as ED
import qweights as QWT

S, Q, QL = ED.S, ED.Q, ED.QL
FAMILIES = ['flip_ind', 'negate_ind', 'negate_joint']
R = 8
NEG0 = 1 + R * len(FAMILIES)
NVAR = NEG0 + S
SEED, NBASE = 2026_1029_36, 6000


def fam_idx(f):
    return [0] + [1 + R * FAMILIES.index(f) + j for j in range(R)]


def build(seed, n, ds=16):
    base = ED.build(seed, n, ds)
    rng = np.random.default_rng([seed, 28])
    res0 = base['res'][:, 0]
    res = np.zeros((n, NVAR, S, QL)); res[:, 0] = res0
    for f in FAMILIES:
        for j in range(R):
            k = 1 + R * FAMILIES.index(f) + j
            if f == 'flip_ind':
                res[:, k] = res0 * rng.choice([-1.0, 1.0], size=(n, S, QL))
            elif f == 'negate_ind':
                res[:, k] = res0 * rng.choice([-1.0, 1.0], size=(n, S, 1))
            else:
                res[:, k] = res0 * rng.choice([-1.0, 1.0], size=(n, 1, 1))
    for s in range(S):
        res[:, NEG0 + s] = res0; res[:, NEG0 + s, s] = -res0[:, s]
    vidx = np.repeat(base['vidx'][:, :1], NVAR, 1); err = np.repeat(base['err'][:, :1], NVAR, 1)
    o = np.empty((n, NVAR, S, Q))
    o[..., :QL] = base['y'][:, None, None, :QL] + res
    o[..., QL] = base['y'][:, None, None, QL] + err
    return dict(y=base['y'], semb=base['semb'], qemb=base['qemb'], vidx=vidx, res=res, err=err, o=o)


def main():
    p, seed, n = 'results/e28_data.npz', SEED, NBASE
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
    d0 = ED.episode(D, 0)
    WB = QWT.bayes_weights({k: v.copy() for k, v in d0.items()})
    rbar = D['res'][:, 0].mean(2)                                               # (n, S) mean labelled residual per source
    inv = float(np.abs(LB - LB[:, :1]).max())
    ss = (D['res'] ** 2).sum(3); dss = float(np.abs(ss - ss[:, :1]).max())
    print('max |Bayes change| %.2e; max |SS change| %.2e' % (inv, dss))
    assert inv < 1e-8 and dss < 1e-9
    np.savez(p, LB=LB, WB=WB, rbar=rbar, deltas=ED.DELTAS, seed=seed, **D, **base)
    print('saved', p)


if __name__ == '__main__':
    main()
