"""E29 post hoc (written after E29's registered results; not registered): which property of a source's random identity vector
moves the prediction? For each base and source s, dmID_s = m(base) - m(base with s's identity vector redrawn). Candidate
features (changes from base to redraw): squared norm |e_s|^2 (a binding-strength proxy), the summed dot products with the other
sources' vectors (cross-talk), each times the source's reading offset o*_s - m(base). OLS without intercept, R^2 against
sum dmID^2, per group; plus a held-out gradient-boosting fit on the old and new vectors' norms and dot products and the offset.
usage: python e29_post.py -> results/e29_post.json
"""
import json, os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import e26_summ as SM


def main():
    dat = np.load('results/e29_data.npz'); semb = dat['semb']; o = dat['o']
    out = {}
    for g, tags in (('A', SM.A_BASES), ('B', SM.B_BASES)):
        Y, X, H, B = [], [], [], []
        for t in tags:
            z = np.load(os.path.join('results/e29/M', t + '.npz')); n = int(z['nb']); m = z['m']
            for s in range(3):
                e0, e1 = semb[:n, 0, s], semb[:n, 4 + s, s]
                ot = [u for u in range(3) if u != s]
                n0, n1 = (e0 ** 2).sum(-1), (e1 ** 2).sum(-1)
                c0 = sum((e0 * semb[:n, 0, u]).sum(-1) for u in ot); c1 = sum((e1 * semb[:n, 0, u]).sum(-1) for u in ot)
                off = o[:n, 0, s, 4] - m[:, 0]
                Y.append(m[:, 0] - m[:, 4 + s])
                X.append(np.stack([(n0 - n1) * off, (c0 - c1) * off, n0 - n1, c0 - c1], 1))
                H.append(np.stack([n0, n1, c0, c1, off, o[:n, 0, s, 4] - dat['y'][:n, 4] * 0 - m[:, 0]], 1)); B.append(np.arange(n))
        Y, X, H, B = map(np.concatenate, (Y, X, H, B))
        res = {}
        for name, cols in (('norm x offset', [0]), ('cross-talk x offset', [1]), ('both x offset', [0, 1]), ('all four', [0, 1, 2, 3])):
            Xi = X[:, cols]; b, *_ = np.linalg.lstsq(Xi, Y, rcond=None)
            res[name] = float(1 - ((Y - Xi @ b) ** 2).sum() / (Y ** 2).sum())
        te = np.isin(B, np.random.default_rng(0).permutation(B.max() + 1)[: (B.max() + 1) // 3])
        mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(H[~te], Y[~te])
        pr = mdl.predict(H[te]); res['boosting held out'] = float(1 - ((Y[te] - pr) ** 2).sum() / ((Y[te] - Y[~te].mean()) ** 2).sum())
        res['rms dmID'] = float(np.sqrt((Y ** 2).mean()))
        out[g] = res
        print(g, {k: round(v, 3) for k, v in res.items()})
    json.dump(out, open('results/e29_post.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
