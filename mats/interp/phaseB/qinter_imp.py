"""E7.3 post hoc: which interactions carry the shared mean deviation? Permutation importance of the 36 interaction features
(model: gradient-boosted trees fitted on 90% of the 65,536 fit episodes, importance on the other 10%, grouped by kind).
usage: python qinter_imp.py -> results/qinter_imp.json
"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
import qshare2 as S2
import qinter as QI

NAMES = ([f'o*{s} x cos(s{s},s{t})' for s in range(3) for t in range(3)] + [f'o*{s} x labelled SS{s}' for s in range(3)]
         + [f'o*{s} x Bayes weight{s}' for s in range(3)] + [f'o*{s} - Bayes mean' for s in range(3)]
         + [f'o*{a} x o*{b}' for a in range(3) for b in range(a, 3)] + [f'o*{s} x residual{s},q{q}' for s in range(3) for q in range(4)])
dF, XF, vF, TF = S2.fit_world(); avgF, _ = S2.load_fit_avg()
I = QI.interactions(dF, XF, TF); assert I.shape[1] == len(NAMES), (I.shape, len(NAMES))
n = len(I); perm = np.random.default_rng(0).permutation(n); va, tr = perm[:n // 10], perm[n // 10:]
m = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=63, max_iter=600, early_stopping=True, random_state=0).fit(I[tr], avgF['mean'][tr])
pi = permutation_importance(m, I[va], avgF['mean'][va], n_repeats=5, random_state=0, scoring='r2')
order = np.argsort(-pi.importances_mean)
rows = [dict(feature=NAMES[i], drop_in_R2=float(pi.importances_mean[i]), sd=float(pi.importances_std[i])) for i in order]
kinds = {}
for r in rows:
    k = r['feature'].split(' x ')[1].split('(')[0].split(',')[0].strip() if ' x ' in r['feature'] else r['feature'].split(' - ')[1] if ' - ' in r['feature'] else r['feature']
    k = k.rstrip('0123456789')
    kinds[k] = kinds.get(k, 0.0) + r['drop_in_R2']
out = dict(val_R2=float(m.score(I[va], avgF['mean'][va])), top=rows[:12], by_kind=kinds)
json.dump(out, open('results/qinter_imp.json', 'w'), indent=1); print(json.dumps(out, indent=1)[:2500])
