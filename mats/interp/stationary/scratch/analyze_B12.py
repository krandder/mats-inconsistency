"""Score Amendments B1 (antithetic at n = 512, 1024) and B2 (RQMC exponent over N = 2^10..2^16)."""
import json, os, glob
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary'
out = {}
anti = {}
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    vals = []
    for f in glob.glob(f'{D}/runs/anti_n{n}_s*.json'):
        vals.append(json.load(open(f))['pair_var_over_half_trvar'])
    if vals:
        anti[n] = vals
out['anti_by_n'] = anti
b1 = [v for n in (512, 1024) for v in anti.get(n, [])]
out['PB9'] = dict(values={n: anti.get(n) for n in (512, 1024)}, hit=bool(b1) and all(0.30 <= v <= 0.45 for v in b1),
                  refuted=any(v < 0.25 or v > 0.5 for v in b1))
rq = {}
for f in sorted(glob.glob(f'{D}/runs/rqmcL_n*_s*.json')):
    r = json.load(open(f)); n = r['n']; s = r['seed']
    T = json.load(open(f'{D}/runs/truth_n{n}_s{s}.json'))
    pts = [(q['N'], q['var_sum']) for q in r['rows']]
    Ns = np.array([p[0] for p in pts], float); vs = np.array([p[1] for p in pts])
    sel = Ns >= 1024
    e_hi = float(-np.polyfit(np.log(Ns[sel]), np.log(vs[sel]), 1)[0]) if sel.sum() >= 2 else None
    sel2 = Ns <= 4096
    e_lo = float(-np.polyfit(np.log(Ns[sel2]), np.log(vs[sel2]), 1)[0])
    rq[f'{n}_{s}'] = dict(exp_hi=e_hi, exp_lo=e_lo, gains=[(int(N), round(T['trvar'] / N / v, 2)) for N, v in pts])
out['PB10'] = dict(rows=rq, hit=bool(rq) and all(1.0 <= v['exp_hi'] <= 1.15 for v in rq.values() if v['exp_hi'] is not None),
                   refuted=any(v['exp_hi'] is not None and v['exp_hi'] >= 1.2 for k, v in rq.items() if k.startswith('8_')))
json.dump(out, open(f'{D}/results/partB12_scores.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
