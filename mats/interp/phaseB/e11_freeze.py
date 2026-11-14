"""E11.1 step 1 (PREREG-B amendment 18): fit and pickle the frozen E10.2b (29 features) and E10.2 (16 features) departure models
on the 8,192 E8 fit episodes, before any fresh episode exists. usage: python e11_freeze.py -> results/e11_frozen.pkl
"""
import os, pickle
import numpy as np
import nnoise  # noqa: F401
import qweights as QWT
import qmech as QM
import qdep as QD
import qdep2 as QD2
from sklearn.ensemble import HistGradientBoostingRegressor


def main():
    fit, _ = QWT.worlds(); EF = QD.ep_quant(fit['d'], fit['X'], fit['WB'], fit['mB'])
    F16 = QD.features(EF); F29 = np.concatenate([F16, QD2.extra(fit['d'], fit['X'], EF)], 2)
    G = {nm: np.load(os.path.join('runs', nm, 'qig_J8.npz')) for nm in QM.ALL}
    models = {}
    for g, nets in (('all', QM.ALL), ('A', QM.A_NETS), ('B', QM.B_NETS)):
        yd = np.mean([G[nm]['fit_dbar'] for nm in nets], 0)
        for tag, FF in (('E10.2b', F29), ('E10.2', F16)):
            models[(tag, g)] = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, max_iter=500, early_stopping=True,
                                                             validation_fraction=0.1, random_state=0).fit(FF.reshape(-1, FF.shape[2]), yd.reshape(-1))
    assert not os.path.exists('results/e11_frozen.pkl')
    pickle.dump(models, open('results/e11_frozen.pkl', 'wb'))
    print({k: m.n_iter_ for k, m in models.items()})


if __name__ == '__main__':
    main()
