"""Order-distribution reading of Amendment 2B's generating function C(t)/C(1) (post hoc, descriptive) and its sensitivity.
Non-negative least squares of C(t)/C(1) at t = 0.25, 0.5, 0.75, 1 on t, t^2, t^3 and t^12 (a stand-in for "high order");
the 2-vs-3 split is ill-conditioned and the tail's order is not identified. Sensitivity: moving 10% of the variance from
order k1 to order k2 changes C(0.75) by 0.1 (0.75^k1 - 0.75^k2) and the mean order by 0.1 (k2 - k1).
Output: results/order_fits.json."""
import json
import numpy as np
from scipy.optimize import nnls

D_ = '/home/agent/workspaces/mats/work/input-bits'
ts = np.array([0.25, 0.5, 0.75, 1.0])
out = {}
for n in (8, 16, 32, 64, 128):
    rs = [json.load(open(f'{D_}/runs/a2b_n{n}_s{s}.json')) for s in (1000, 1001, 4000, 4001)]
    Cn = np.median([np.array(r['C'][1:]) / r['C'][4] for r in rs], 0)
    fits = {}
    for kh in (6, 12, 30):
        A = np.stack([ts, ts ** 2, ts ** 3, ts ** kh], 1)
        x, res = nnls(A, Cn)
        fits[f'tail_order_{kh}'] = dict(s1=float(x[0]), s2=float(x[1]), s3=float(x[2]), s_tail=float(x[3]), resid=float(res))
    out[n] = dict(C_norm=Cn.tolist(), fits=fits)
    print(n, np.round(Cn, 3), {k: [round(v['s1'], 3), round(v['s2'] + v['s3'], 3), round(v['s_tail'], 3)] for k, v in fits.items()})
sens = {f'{k1}->{k2}': dict(dC075=0.1 * (0.75 ** k1 - 0.75 ** k2), dmean=0.1 * (k2 - k1)) for k1, k2 in ((4, 20), (10, 100), (5, 50))}
out['sensitivity'] = sens
print(sens)
json.dump(out, open(f'{D_}/results/order_fits.json', 'w'), indent=1)
