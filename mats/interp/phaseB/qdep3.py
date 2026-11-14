"""E10.2b feature groups (post hoc, PREREG-B amendment 17 addendum 2): refit E10.2b without one feature group at a time.
usage: python qdep3.py -> results/qdep3.json
"""
import json, os
import numpy as np
import nnoise  # noqa: F401
import qweights as QWT
import qmech as QM
import qdep as QD
import qdep2 as QD2
from sklearn.ensemble import HistGradientBoostingRegressor

N8 = 8000
GROUPS = {'E10.2 base features': list(range(0, 16)), 'labelled residual magnitudes': list(range(16, 20)),
          'source-ID geometry': list(range(20, 23)), 'query-question ID cosines': list(range(23, 27)), 'raw readings': [27, 28]}


def main():
    fit, _ = QWT.worlds(); t8 = QM.test8000()
    EF = QD.ep_quant(fit['d'], fit['X'], fit['WB'], fit['mB']); E8 = QD.ep_quant(t8['d'], t8['X'], t8['WB'], t8['mB'])
    FF = np.concatenate([QD.features(EF), QD2.extra(fit['d'], fit['X'], EF)], 2); F8 = np.concatenate([QD.features(E8), QD2.extra(t8['d'], t8['X'], E8)], 2)
    S = {nm: np.load(os.path.join('runs', nm, 'share_test.npz')) for nm in QM.ALL}
    dev = {nm: (S[nm]['r1_mean'][:N8], S[nm]['r2_mean'][:N8]) for nm in QM.ALL}
    G = {nm: np.load(os.path.join('runs', nm, 'qig_J8.npz')) for nm in QM.ALL}
    off8 = E8['o'] - E8['mB'][:, None]; out = {}
    for g, nets in (('all', QM.ALL), ('A', QM.A_NETS), ('B', QM.B_NETS)):
        avg8 = np.mean([dev[nm][0] for nm in nets], 0); yd = np.mean([G[nm]['fit_dbar'] for nm in nets], 0); out[g] = {}
        for name, idx in [('none dropped', [])] + list(GROUPS.items()):
            keep = [i for i in range(FF.shape[2]) if i not in idx]
            hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, max_iter=500, early_stopping=True,
                                                validation_fraction=0.1, random_state=0).fit(FF[:, :, keep].reshape(-1, len(keep)), yd.reshape(-1))
            pred = (off8 * hgb.predict(F8[:, :, keep].reshape(-1, len(keep))).reshape(-1, 3)).sum(1)
            out[g][name] = QM.ens_score(avg8, pred, dev, nets)[1]
        print(g, {k: round(v, 3) for k, v in out[g].items()}, flush=True)
    json.dump(out, open('results/qdep3.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
