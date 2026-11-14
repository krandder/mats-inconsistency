"""Score PREREG Part S (exact small-width densities) -> results/partS_scores.json and a table."""
import json, glob
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary-numerics'
R = {n: [json.load(open(f)) for f in sorted(glob.glob(f'{D}/runs/S_n{n}_s*.json'))] for n in (2, 3, 4)}
S = {}
ok1 = True
for n, rs in R.items():
    for r in rs:
        lim = 5e-3 if n < 4 else 5e-2
        ok1 &= r['TV_refine'] < lim
        if n == 2:
            ok1 &= r['TV_pullback_two_starts'] < 1e-6
        if n == 3:
            ok1 &= r['TV_pullback_two_starts'] < 1e-4
S['S1'] = dict(hit=bool(ok1), TV_refine={str(n): [r['TV_refine'] for r in rs] for n, rs in R.items()},
               TV_two_starts={str(n): [r['TV_pullback_two_starts'] for r in rs] for n, rs in R.items()})
ok2 = all(r['vs_gaussian']['TV'] >= 0.05 and r['vs_mixture']['TV'] <= 0.5 * r['vs_gaussian']['TV'] for n in (2, 3) for r in R[n])
S['S2'] = dict(hit=bool(ok2), rows={str(n): [(r['seed'], r['Q_mean'], r['vs_gaussian']['TV'], r['vs_mixture']['TV'],
                                              r['vs_gaussian']['KL'], r['vs_mixture']['KL']) for r in rs] for n, rs in R.items()})
med = {n: float(np.median([r['vs_gaussian']['TV'] for r in rs])) for n, rs in R.items() if rs}
S['S3'] = dict(hit=bool(len(med) == 3 and med[2] > med[3] > med[4]), median_TV_gauss=med)
u = [x for r in R[2] for x in r['next_layer_units']]
eg = [abs(x['boundary_density_gauss'] / x['boundary_density'] - 1) for x in u]
em = [abs(x['boundary_density_mix'] / x['boundary_density'] - 1) for x in u]
S['S4'] = dict(hit=bool(np.mean(np.array(eg) > 0.2) >= 0.5 and np.mean(np.array(em) <= 0.1) >= 0.5),
               gauss_rel_err=eg, mix_rel_err=em)
json.dump(S, open(f'{D}/results/partS_scores.json', 'w'), indent=1)
for k, v in S.items():
    print(k, v['hit'])
print('n seed | Q_mean | TV gauss | TV mix | KL gauss | KL mix | refine TV | gate entropy (bits) | largest |skew|, |exkurt|')
for n, rs in R.items():
    for r in rs:
        sk = max(abs(p['skew']) for p in r['projections']); ku = max(abs(p['exkurt']) for p in r['projections'])
        print('%d %d | %.2f | %.3f | %.3f | %.4f | %.4f | %.1e | %.2f | %.2f, %.2f' % (
            n, r['seed'], r['Q_mean'], r['vs_gaussian']['TV'], r['vs_mixture']['TV'], r['vs_gaussian']['KL'],
            r['vs_mixture']['KL'], r['TV_refine'], r['gate_entropy_layer1_bits'], sk, ku))
print('S4 gauss rel err', np.round(eg, 2), 'mix', np.round(em, 2))
