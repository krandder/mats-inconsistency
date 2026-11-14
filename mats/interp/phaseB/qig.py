"""E8.1b (post hoc, PREREG-B amendment 15 addendum 2): path-averaged (integrated-gradient) source weight errors.
Path in query-reading space from the agreement point o_ref = (c, c, c), c = mu_B(o), to the observed readings o:
  dev(o) - dev(o_ref) = sum_s (o*_s - c) * dbar_s,  dbar_s = int_0^1 [w_hat_s - W_s](o_ref + t (o - o_ref)) dt,
Gauss-Legendre nodes; network side = noise-marginal predictive mean (K = 8, the same noise draws at every node).
Saves results/qig_world.npz (Bayes side) and runs/NET/qig.npz (fit_dbar, fit_mref, fit_m, test_dbar, test_mref, test_m).
usage: python qig.py NET [NET ...]
"""
import os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import qweights as QWT

torch.set_num_threads(1)
QQ, VAL = QWT.QQ, QWT.VAL
J = int(os.environ.get('QIG_J', 8))


def nodes(J):
    x, w = np.polynomial.legendre.leggauss(J); return 0.5 * (x + 1), 0.5 * w


def path_d(d, c, t):
    dd = {k: v.copy() for k, v in d.items()}; dd['o'][:, :, W.QL] = c[:, None] + t * (d['o'][:, :, W.QL] - c[:, None]); return dd


def path_X(X, d, c, t):
    Xt = X.copy(); Xt[:, QQ, VAL] = (c[:, None] + t * (d['o'][:, :, W.QL] - c[:, None])).astype(np.float32); return Xt


def bayes_side(w, J):
    ts, ws = nodes(J); c = w['mB']
    WB = sum(wj * QWT.bayes_weights(path_d(w['d'], c, tj)) for tj, wj in zip(ts, ws))
    mref = QWT.bayes_mean(path_d(w['d'], c, 0.0))
    return dict(dbarB=WB, mBref=mref)


def world_side():
    p = f'results/qig_world_J{J}.npz'
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z['fit'].item(), z['test'].item()
    fit, test = QWT.worlds(); bf, bt = bayes_side(fit, J), bayes_side(test, J)
    np.savez(p, fit=np.array(bf, dtype=object), test=np.array(bt, dtype=object)); return bf, bt


def net_side(model, w, sg, seed, J, n=None):
    ts, ws = nodes(J); c = w['mB'] if n is None else w['mB'][:n]; d = w['d'] if n is None else {k: v[:n] for k, v in w['d'].items()}
    X = w['X'] if n is None else w['X'][:n]; v = w['v'] if n is None else w['v'][:n]
    dbar = 0
    for tj, wj in zip(ts, ws):
        wt, _ = QWT.implied(model, path_X(X, d, c, tj), v, sg, seed=seed); dbar = dbar + wj * wt
    _, mref = QWT.implied(model, path_X(X, d, c, 0.0), v, sg, seed=seed)
    _, m1 = QWT.implied(model, X, v, sg, seed=seed)
    return dbar, mref, m1


if __name__ == '__main__':
    bf, bt = world_side(); fit, test = QWT.worlds()
    for nm in sys.argv[1:]:
        p = os.path.join('runs', nm, f'qig_J{J}.npz')
        if os.path.exists(p): continue
        t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
        fd, fr, fm = net_side(model, fit, sg, 1, J); td, tr, tm = net_side(model, test, sg, 2, J)
        np.savez(p, fit_dbar=fd - bf['dbarB'], fit_mref=fr, fit_m=fm, test_dbar=td - bt['dbarB'], test_mref=tr, test_m=tm)
        lhs = (tm - test['mB']) - (tr - bt['mBref']); rhs = ((td - bt['dbarB']) * (test['d']['o'][:, :, W.QL] - test['mB'][:, None])).sum(1)
        print(nm, 'done %.0fs' % (time.time() - t0), 'identity check (test) R2 %.4f' % (1 - ((lhs - rhs) ** 2).sum() / ((lhs - lhs.mean()) ** 2).sum()), flush=True)
