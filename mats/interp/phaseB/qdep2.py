"""E10.2b (post hoc, PREREG-B amendment 17 addendum 1): E10.2's departure model with 13 added input features
(labelled residual magnitudes, source-ID geometry, query-question ID geometry, raw readings).
usage: python qdep2.py -> results/qdep2.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import world as W
import qweights as QWT
import qmech as QM
import qdep as QD
from sklearn.ensemble import HistGradientBoostingRegressor

N8, N4, NB = 8000, 4000, 500


def extra(d, X, E):
    n = len(E['mB']); o = E['o']
    res = np.abs(d['o'][:, :, :W.QL] - d['y'][:, None, :W.QL]); res = np.sort(res, 2)                       # (n, 3, 4)
    semb = np.stack([X[:, 5 * s, :16] for s in range(3)], 1).astype(np.float64); nrm = np.linalg.norm(semb, axis=2)
    sn = semb / nrm[:, :, None]; C = np.einsum('nsd,ntd->nst', sn, sn)
    qq = X[:, 15, 16:32].astype(np.float64); ql = np.stack([X[:, q, 16:32] for q in range(W.QL)], 1).astype(np.float64)
    qc = np.sort(np.einsum('nd,nqd->nq', qq / np.linalg.norm(qq, axis=1, keepdims=True), ql / np.linalg.norm(ql, axis=2, keepdims=True)), 1)
    rows = []
    for s in range(3):
        oth = [t for t in range(3) if t != s]
        o1 = np.where(E['p'][:, oth[0]] >= E['p'][:, oth[1]], oth[0], oth[1]); o2 = np.where(o1 == oth[0], oth[1], oth[0])
        g = lambda a, idx: a[np.arange(n), idx]
        rows.append(np.c_[res[:, s], C[np.arange(n), s, o1], C[np.arange(n), s, o2], nrm[:, s], qc, o[:, s], g(o, o1)])
    return np.stack(rows, 1)                                                                                     # (n, 3, 13)


def main():
    fit, _ = QWT.worlds(); t8 = QM.test8000()
    EF = QD.ep_quant(fit['d'], fit['X'], fit['WB'], fit['mB']); E8 = QD.ep_quant(t8['d'], t8['X'], t8['WB'], t8['mB'])
    FF = np.concatenate([QD.features(EF), extra(fit['d'], fit['X'], EF)], 2); F8 = np.concatenate([QD.features(E8), extra(t8['d'], t8['X'], E8)], 2)
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in QM.ALL}
    dev = {nm: (S[nm]['r1_mean'][:N8], S[nm]['r2_mean'][:N8]) for nm in QM.ALL}
    G = {nm: np.load(os.path.join('runs', nm, 'qig_J8.npz')) for nm in QM.ALL}
    rng = np.random.default_rng(0); boots = [rng.integers(0, N8, N8) for _ in range(NB)]
    DIS8 = E8['sd'] >= np.quantile(E8['sd'], 2 / 3); idis = np.where(DIS8)[0]; off8 = E8['o'] - E8['mB'][:, None]
    out = {}
    for g, nets in (('all', QM.ALL), ('A', QM.A_NETS), ('B', QM.B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0); yd = np.mean([G[nm]['fit_dbar'] for nm in nets], 0)
        hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, max_iter=500, early_stopping=True,
                                            validation_fraction=0.1, random_state=0).fit(FF.reshape(-1, FF.shape[2]), yd.reshape(-1))
        dh = hgb.predict(F8.reshape(-1, F8.shape[2])).reshape(-1, 3); pred = (off8 * dh).sum(1)
        R, sc = QM.ens_score(avg8, pred, dev, nets); bs = [QM.ens_score(avg8, pred, dev, nets, ii)[1] for ii in boots]
        Rd, scd = QM.ens_score(avg8, pred, dev, nets, idis)
        out[g] = dict(R2=R, score=sc, ci=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], score_dis=scd, R2_dis=Rd,
                      dbar_R2_test=float(QM.r2(np.mean([G[nm]['test_dbar'] for nm in nets], 0).ravel(), dh[:N4].ravel())), n_iter=int(hgb.n_iter_))
        print('E10.2b', g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[g].items()}, flush=True)
    json.dump(out, open('results/qdep2.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
