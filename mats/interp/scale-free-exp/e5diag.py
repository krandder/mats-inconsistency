"""E5 diagnostic: which variables drive the tree net's deep NLL tail events? Monte Carlo episodes with NLL > L:
distribution of y*, the shared labels, and the largest standardised source error (query and labelled)."""
import json, math, sys
import numpy as np
import torch
import sfworld as W
import est5 as E

net, _ = E.load_tree(sys.argv[1])
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
rng = np.random.default_rng([9, 9])
rows = []
n = 0
while n < M:
    d = W.sample(200_000, 12, rng)
    with torch.no_grad():
        mu, ls, lg = [t.numpy().astype(np.float64) for t in net(torch.from_numpy(W.src_features(d)))]
    ys = d['y'][:, W.QL]
    nl = E.nll_mix(mu, ls, lg, ys[:, None])
    idx = np.nonzero(nl > 10)[0]
    for i in idx:
        e = (d['o'][i] - d['y'][i][None, :]) / np.sqrt(d['v'][i])[:, None]
        rows.append(dict(nll=float(nl[i]), ystar=float(ys[i]), ymax_lab=float(np.abs(d['y'][i, :4]).max()),
                         max_q_err=float(np.abs(e[:, 4]).max()), max_lab_err=float(np.abs(e[:, :4]).max()),
                         v_of_maxq=float(d['v'][i][np.abs(e[:, 4]).argmax()]),
                         lab_ss_of_maxq=float((e[np.abs(e[:, 4]).argmax(), :4] ** 2).sum())))
    n += 200_000
print(len(rows), 'episodes with NLL > 10 out of', n)
for k in ('ystar', 'ymax_lab', 'max_q_err', 'max_lab_err', 'v_of_maxq', 'lab_ss_of_maxq', 'nll'):
    v = np.array([r[k] for r in rows])
    print(k, 'quantiles 10/50/90:', np.round(np.quantile(v, [0.1, 0.5, 0.9]), 2))
json.dump(rows, open('results/e5diag_tail.json', 'w'))
