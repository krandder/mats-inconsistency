"""E54 data (PREREG-B amendment 62): world-2 matched pairs that keep every sufficient statistic of the context (X^T X, X^T y,
y.y) and change only how the residual sum of squares is spread over the examples. Exact Bayes is identical within a pair.
For each base episode (t2_world.sample, N = 8 context examples, dimension D):
  w_hat = (X^T X)^-1 X^T y (least squares), e = y - X w_hat (X^T e = 0), RSS = e.e, P = I - X (X^T X)^-1 X^T (rank N - D);
  big:    e_big = s sqrt(RSS) P d_j / |P d_j|  for a random example j and sign s: most of RSS on example j (share P_jj);
  spread: e_spr = sqrt(RSS) P sig / |P sig|   for a random sign vector sig in {-1, +1}^N;
  y_big = X w_hat + e_big, y_spr = X w_hat + e_spr: both keep X^T y = X^T X w_hat and y.y = |X w_hat|^2 + RSS.
Calibration (descriptive): the spread version with its residual scaled by f = 0.6 and 0.8 (RSS x 0.36 and 0.64).
Also saved: the leave-j-out least-squares prediction shift on the big version, d = x*.(w_hat_{-j} - w_hat) (the direction in
which discounting the outlier would move the mean), the largest residual share of each version, and Bayes's predictive (checked
identical within pairs).
usage: python e54_data.py D NBASE SEED OUT.npz
"""
import sys
import numpy as np

sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp-B')
import t2_world as T  # noqa: E402

N = 8
CAL = (0.6, 0.8)


def build(D, nb, seed):
    rng = np.random.default_rng(seed)
    d = T.sample(nb, rng, D, N)
    x, y, xq = d['x'].astype(np.float64), d['y'].astype(np.float64), d['xq'].astype(np.float64)
    G = np.einsum('nkd,nke->nde', x, x); Gi = np.linalg.inv(G)
    wh = np.einsum('nde,nke,nk->nd', Gi, x, y)
    fit = np.einsum('nkd,nd->nk', x, wh); e = y - fit; rss = (e * e).sum(1)
    H = np.einsum('nkd,nde,nle->nkl', x, Gi, x); P = np.eye(N)[None] - H
    j = rng.integers(0, N, nb); s = np.where(rng.random(nb) < 0.5, -1.0, 1.0)
    ub = P[np.arange(nb), :, j]; ub = ub / np.linalg.norm(ub, axis=1, keepdims=True)
    sig = np.where(rng.random((nb, N)) < 0.5, -1.0, 1.0)
    us = np.einsum('nkl,nl->nk', P, sig); us = us / np.linalg.norm(us, axis=1, keepdims=True)
    e_big = s[:, None] * np.sqrt(rss)[:, None] * ub; e_spr = np.sqrt(rss)[:, None] * us
    y_big = fit + e_big; y_spr = fit + e_spr
    keep = np.ones(N, bool)
    dshift = np.empty(nb)
    for i in range(nb):                                      # leave-j-out least squares on the big version
        k = keep.copy(); k[j[i]] = False
        wj = np.linalg.lstsq(x[i, k], y_big[i, k], rcond=None)[0]
        wf = np.linalg.lstsq(x[i], y_big[i], rcond=None)[0]
        dshift[i] = xq[i] @ (wj - wf)
    share = lambda ee: (ee ** 2).max(1) / (ee ** 2).sum(1)
    out = dict(x=x, xq=xq, y_big=y_big, y_spr=y_spr, y_orig=y, li=d['li'], w=d['w'], j=j, rss=rss, dshift=dshift,
               share_big=share(e_big), share_spr=share(e_spr))
    for f in CAL:                                            # calibration: the spread version with its residual scaled by f
        out['y_spr_f%02d' % round(100 * f)] = fit + f * e_spr
    bb = T.bayes(dict(x=x, y=y_big, xq=xq)); bs = T.bayes(dict(x=x, y=y_spr, xq=xq)); bo = T.bayes(dict(x=x, y=y, xq=xq))
    out['bayes_mu'], out['bayes_va'], out['bayes_wt'] = bb
    out['inv_max'] = float(max(np.abs(bb[k] - bs[k]).max() for k in range(3)))
    out['inv_max_orig'] = float(max(np.abs(bb[k] - bo[k]).max() for k in range(3)))
    out['cons_max'] = float(max(np.abs(np.einsum('nkd,nk->nd', x, y_big) - np.einsum('nkd,nk->nd', x, y)).max(),
                                np.abs((y_big ** 2).sum(1) - (y ** 2).sum(1)).max(), np.abs((y_spr ** 2).sum(1) - (y ** 2).sum(1)).max()))
    return out


if __name__ == '__main__':
    D, nb, seed, path = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    out = build(D, nb, seed)
    np.savez(path, **out)
    print('D %d bases %d: Bayes invariance max |diff| big vs spread %.2e, big vs original %.2e; statistics conserved to %.2e; '
          'largest residual share big %.3f (median), spread %.3f' % (D, nb, out['inv_max'], out['inv_max_orig'], out['cons_max'],
                                                                     np.median(out['share_big']), np.median(out['share_spr'])))
