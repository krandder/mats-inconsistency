"""E28 post hoc, exploratory (rewritten after the Codex review of E28): how much of the negation-induced mean shift dm_s do
fitted predictors explain on held-out episodes? Gradient boosting, no synthetic augmentation, train/test split by base episode
(all three sources of a base on the same side), features from the base version only, no hidden query answer.
Feature sets: own = source s's labelled residuals ordered by magnitude (signed, 4), its reading offset o*_s - m(base), its weight
w_s; all = own + the other sources' weights, sums of squares and reading offsets + the four labels.
A learner's held-out R^2 is a lower bound on predictability from these features, not an upper bound.
usage: python e28_post_flex.py -> results/e28_post_fits.json
"""
import json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import e26_summ as SM
import e28_data as E8


def main():
    dat = np.load('results/e28_data.npz'); res0 = dat['res'][:, 0]; y = dat['y']; o = dat['o'][:, 0]
    out = {}
    for g, tags in (('A', SM.A_BASES), ('B', SM.B_BASES)):
        out[g] = {}
        for t in tags:
            z = np.load(os.path.join('results/e28/M', t + '.npz')); n = int(z['nb'])
            dm = np.stack([z['m_base'] - z['m_neg%d' % s] for s in range(3)], 1)
            w = z['w_base']; m = z['m_base']
            Xo, Xa, Y, B = [], [], [], []
            for s in range(3):
                ot = [u for u in range(3) if u != s]
                order = np.argsort(-np.abs(res0[:n, s]), axis=-1)
                srt = np.take_along_axis(res0[:n, s], order, -1)
                own = np.concatenate([srt, (o[:n, s, 4] - m)[:, None], w[:, s:s + 1]], 1)
                oth = np.concatenate([w[:, ot], (res0[:n, ot] ** 2).sum(-1), o[:n, ot, 4] - m[:, None], y[:n, :4]], 1)
                Xo.append(own); Xa.append(np.concatenate([own, oth], 1)); Y.append(dm[:, s]); B.append(np.arange(n))
            Xo, Xa, Y, B = map(np.concatenate, (Xo, Xa, Y, B))
            test_bases = np.random.default_rng(0).permutation(n)[: n // 3]
            te = np.isin(B, test_bases); tr = ~te
            r = {}
            for name, X in (('own', Xo), ('all', Xa)):
                mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X[tr], Y[tr])
                pr = mdl.predict(X[te])
                r[name] = float(1 - ((Y[te] - pr) ** 2).sum() / ((Y[te] - Y[tr].mean()) ** 2).sum())
            out[g][t] = r
        print(g, {t: {k: round(v, 3) for k, v in r.items()} for t, r in out[g].items()})
    json.dump(out, open('results/e28_post_fits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
