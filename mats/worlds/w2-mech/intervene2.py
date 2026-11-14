"""Amendment 2: overlap-edge ablation for EV rows (blocks 1, 2, 1+2) and D4 equalisation in blocks 3(+4)."""
import json, sys, time
import numpy as np
import torch
import intervene_lib as L

NET = sys.argv[1]
t0 = time.time()
C = L.setup(NET)
E, T = C['E'], C['T']
eps = C['eps']
rows = C['AEV'] | C['BEV']
OV = np.zeros((E, T, T), bool); CT = np.zeros((E, T, T), bool)
for i, e in enumerate(eps):
    n = e.n
    it = e.ev_items[e.q_ev]
    inc = np.zeros((n, e.K)); rr, cc = np.nonzero(it >= 0); inc[rr, it[rr, cc]] = 1
    ov = (inc @ inc.T) > 0
    oth = e.q_src[:, None] != e.q_src[None, :]
    r = np.random.default_rng(5000 + int(C['pair'][i]))
    for t in np.flatnonzero(rows[i, :n]):
        keys = np.flatnonzero(ov[t] & oth[t])
        OV[i, t, keys] = True
        cand = np.flatnonzero(~ov[t] & oth[t])
        CT[i, t, r.choice(cand, min(len(keys), len(cand)), replace=False)] = True
print('mean overlap keys per EV row', OV.sum(-1)[rows].mean(), flush=True)


def ablate(blocks, Dm):
    def fac(sl):
        D = torch.as_tensor(Dm[sl])[:, None]

        def ed(li, a):
            if li not in blocks:
                return a
            a = a.masked_fill(D, 0.0)
            return a / a.sum(-1, keepdim=True)
        return ed
    return fac


res = dict(net=NET, base=C['score'](*C['run'](None)))
conds = {'ovl_b1': ablate({0}, OV), 'ctrl_b1': ablate({0}, CT), 'ovl_b2': ablate({1}, OV), 'ctrl_b2': ablate({1}, CT),
         'ovl_b1b2': ablate({0, 1}, OV), 'ctrl_b1b2': ablate({0, 1}, CT)}
if NET.startswith('D4'):
    conds['equalise_b3'] = L.equalise(C, [2]); conds['equalise_b3b4'] = L.equalise(C, [2, 3])
for k, fac in conds.items():
    res[k] = C['score'](*C['run'](fac))
    print(k, res[k], f'{time.time() - t0:.0f}s', flush=True)
json.dump(res, open(f'out/intervene2_{NET}.json', 'w'), indent=1)
