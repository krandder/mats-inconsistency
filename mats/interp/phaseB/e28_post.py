"""E28 post hoc (written after E28's registered results; not registered): which odd statistic of a source's labelled errors moves
the prediction? On the negation pairs, dm_s = m(base) - m(base with source s negated) is odd in source s's residuals. Each
candidate odd feature f (computed on the base residuals; f(-r) = -f(r)) enters as 2 f; regressors are 2 f times the net's base
weight w_s, and 2 f times w_s times the reading's offset d_s = o*_s - (Bayes mean without... approximated by the net's base
mean m), i.e. an odd term that scales a source's influence by the direction of its reading. R^2 against sum dm^2, per group.
Features: mean error rbar; third moment m3 = sum r^3 / SS; signed largest error r_max; sign count sum sign(r) / 4.
usage: python e28_post.py -> results/e28_post.json
"""
import json, os
import numpy as np
import e26_summ as SM
import e28_data as E8


def feats(res):
    ss = (res ** 2).sum(-1)
    imax = np.abs(res).argmax(-1)
    rmax = np.take_along_axis(res, imax[..., None], -1)[..., 0]
    return dict(rbar=res.mean(-1), m3=(res ** 3).sum(-1) / np.maximum(ss, 1e-9), rmax=rmax, signs=np.sign(res).mean(-1))


def r2(Y, X):
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(1 - ((Y - X @ b) ** 2).sum() / (Y ** 2).sum()), b


def main():
    dat = np.load('results/e28_data.npz')
    res0 = dat['res'][:, 0]; o_star = dat['o'][:, 0, :, E8.QL]
    F = feats(res0)
    out = {}
    for g, tags in (('A', SM.A_BASES), ('B', SM.B_BASES)):
        rows = {}
        Ys, Xs = [], {k: [] for k in ('w', 'wd')}
        per = {}
        for t in tags:
            z = np.load(os.path.join('results/e28/M', t + '.npz')); n = int(z['nb'])
            w = z['w_base']; m = z['m_base'][:, None]
            dm = np.stack([z['m_base'] - z['m_neg%d' % s] for s in range(3)], 1)
            d = o_star[:n] - m
            Ys.append(dm.ravel())
            for k in ('w', 'wd'):
                cols = [(2 * F[f][:n] * w * (1 if k == 'w' else d)).ravel() for f in F]
                Xs[k].append(np.stack(cols, 1))
        Y = np.concatenate(Ys)
        res = {}
        names = list(F)
        for k in ('w', 'wd'):
            X = np.concatenate(Xs[k])
            for i, f in enumerate(names):
                res['%s x %s' % (f, 'w' if k == 'w' else 'w x reading offset')] = r2(Y, X[:, i:i + 1])[0]
            res['all four x %s' % ('w' if k == 'w' else 'w x reading offset')] = r2(Y, X)[0]
        Xall = np.concatenate([np.concatenate(Xs['w']), np.concatenate(Xs['wd'])], 1)
        res['all eight'] = r2(Y, Xall)[0]
        res['rms dm'] = float(np.sqrt((Y ** 2).mean()))
        out[g] = res
    json.dump(out, open('results/e28_post.json', 'w'), indent=1)
    for g, r in out.items():
        print(g, {k: round(v, 3) for k, v in r.items()})


if __name__ == '__main__':
    main()
