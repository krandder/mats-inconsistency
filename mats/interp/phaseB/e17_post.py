"""E17 post hoc (written after the registered E17 results): gap-matched comparison of 3-layer and 2-layer clean nets.
usage: python e17_post.py -> results/e17_post.json
"""
import json
import numpy as np

o = json.load(open('results/e17.json')); P = o['per_net']
two = [(d['gap'], d['large']['rom']) for t, d in P.items() if d['kind'] == 'clean' and d['acquired'] and d['arm'] != 'L3']
three = [(d['gap'], d['large']['rom'], t) for t, d in P.items() if d['kind'] == 'clean' and d['acquired'] and d['arm'] == 'L3']
g2, r2 = np.array([x[0] for x in two]), np.array([x[1] for x in two])
b = np.polyfit(g2, r2, 1)
res = [float(x[1] - np.polyval(b, x[0])) for x in three]
gmax = max(x[0] for x in three); lo = [x for x in two if x[0] <= gmax]
out = dict(n_two=len(two), fit_intercept=float(b[1]), fit_slope=float(b[0]), three_residuals={x[2]: r for x, r in zip(three, res)},
           mean_residual=float(np.mean(res)), n_negative=int(sum(r < 0 for r in res)),
           two_in_range=dict(n=len(lo), mean_rom=float(np.mean([x[1] for x in lo])), mean_gap=float(np.mean([x[0] for x in lo]))),
           three=dict(n=len(three), mean_rom=float(np.mean([x[1] for x in three])), mean_gap=float(np.mean([x[0] for x in three]))))
json.dump(out, open('results/e17_post.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'three_residuals'}))
