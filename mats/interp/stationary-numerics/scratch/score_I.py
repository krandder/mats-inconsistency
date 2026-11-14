"""Score PREREG Part I (interaction order across width; causal gate-freezing tests) -> results/partI_scores.json, figure."""
import json, glob, math, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '/home/agent/workspaces/mats/work/stationary-numerics/lib')

D = '/home/agent/workspaces/mats/work/stationary-numerics'
L0 = {int(k): v for k, v in json.load(open(f'{D}/results/l0_theory.json')).items()}
NS = (8, 16, 32, 64, 128, 256)
K = ('h', 'u', 'logR', 'a', 'g')
I1 = {}
for n in NS:
    rows = []
    for f in sorted(glob.glob(f'{D}/runs/I1_n{n}_s*.json')):
        r = json.load(open(f))['results']
        rows.append(dict(seed=int(f.split('_s')[-1][:-5]), **{k: r['true'][k]['D'] for k in K},
                         **{k + '_se': r['true'][k]['D_se'] for k in K}, frozen_h=r['frozen']['h']['D'],
                         frozen_h_se=r['frozen']['h']['D_se'], frozen_u=r['frozen']['u']['D'], Q0n=r['true']['Q0_median'] / n))
    I1[n] = rows
med = {n: {k: float(np.median([r[k] for r in I1[n]])) for k in K + ('frozen_h', 'frozen_u')} for n in NS}
# bootstrap over nets for the median
rng = np.random.default_rng(1)
med_se = {}
for n in NS:
    med_se[n] = {}
    for k in K:
        v = np.array([r[k] for r in I1[n]])
        med_se[n][k] = float(np.std([np.median(v[rng.integers(0, v.size, v.size)]) for _ in range(2000)]))
S = {}
rho_u = spearmanr(NS, [med[n]['u'] for n in NS]).correlation
rho_h = spearmanr(NS, [med[n]['h'] for n in NS]).correlation
I1a = (rho_u >= 0.8 and med[256]['u'] / med[16]['u'] >= 1.5 and rho_h >= 0.8 and med[256]['h'] / med[16]['h'] >= 1.3
       and med[256]['g'] / med[16]['g'] >= 3)
S['I1a'] = dict(hit=bool(I1a), spearman_u=rho_u, spearman_h=rho_h, ratio_u=med[256]['u'] / med[16]['u'],
                ratio_h=med[256]['h'] / med[16]['h'], ratio_g=med[256]['g'] / med[16]['g'],
                medians={str(n): med[n] for n in NS}, median_se={str(n): med_se[n] for n in NS})
fit = {}
for k in K:
    x = np.log([n for n in NS if n >= 16]); y = np.log([med[n][k] for n in NS if n >= 16])
    fit[k] = float(np.polyfit(x, y, 1)[0])
S['I1a']['loglog_slope_16_256'] = fit
S['I1b'] = dict(hit=bool(all(0.6 <= med[n]['u'] / L0[n]['D_ang_u'] <= 1.4 for n in NS if n >= 32)),
                ratio={str(n): med[n]['u'] / L0[n]['D_ang_u'] for n in NS})
zs = [abs(r['frozen_h'] - 1) / r['frozen_h_se'] for n in NS for r in I1[n]]
S['I2a'] = dict(hit=bool(sum(z < 3 for z in zs) >= 46), n_within_3SE=int(sum(z < 3 for z in zs)), max_z=float(max(zs)))
V = ['true', 'frozen', 'free_0.5', 'free_1', 'free_2', 'free_4', 'recent_q', 'recent_n', 'clamp']
I2 = {}
for n in (32, 64, 128, 256):
    rows = []
    for f in sorted(glob.glob(f'{D}/runs/I2_n{n}_s*.json')):
        r = json.load(open(f))['results']
        row = {}
        for obs in ('u', 'h'):
            Dt, Df = r['true'][obs]['D'], r['frozen'][obs]['D']
            for v in V:
                row[f'D_{obs}_{v}'] = r[v][obs]['D']
                row[f'E_{obs}_{v}'] = (r[v][obs]['D'] - Df) / (Dt - Df)
        for v in V:
            row[f'frac_free_{v}'] = r[v]['frac_free']
        rows.append(row)
    I2[n] = rows


def m(n, key):
    return float(np.median([r[key] for r in I2[n]])) if I2[n] else float('nan')


S['I2b'] = dict(hit=bool(m(64, 'E_u_free_1') < 0.5 and m(64, 'E_u_free_4') > 0.7 and m(256, 'E_u_free_4') < m(32, 'E_u_free_4')),
                E_u={str(n): {v: m(n, f'E_u_{v}') for v in V[2:]} for n in I2},
                E_h={str(n): {v: m(n, f'E_h_{v}') for v in V[2:]} for n in I2},
                frac_free={str(n): {v: m(n, f'frac_free_{v}') for v in V[2:6]} for n in I2})
S['I2c'] = dict(hit=bool(all(m(n, 'E_u_recent_n') >= 0.8 and 0.3 <= m(n, 'E_u_recent_q') <= 0.9 for n in I2)))
S['I2d'] = dict(hit=bool(all(m(n, 'D_u_clamp') >= 0.7 * m(n, 'D_u_true') for n in I2)),
                ratio={str(n): m(n, 'D_u_clamp') / m(n, 'D_u_true') for n in I2})
json.dump(dict(scores=S, I1=I1, I2=I2), open(f'{D}/results/partI_scores.json', 'w'), indent=1)
for k, v in S.items():
    print(k, {kk: vv for kk, vv in v.items() if kk.startswith('hit')})
print('median D over 8 nets (+- bootstrap SE):')
for n in NS:
    print(n, ' '.join('%s %.2f+-%.2f' % (k, med[n][k], med_se[n][k]) for k in K), ' L0 u %.2f' % L0[n]['D_ang_u'],
          ' frozen h %.3f' % med[n]['frozen_h'])
print('slopes 16-256:', {k: round(v, 3) for k, v in fit.items()})
for n in I2:
    print(n, 'E_u', {v: round(m(n, f'E_u_{v}'), 2) for v in V[2:]})
    print(n, 'E_h', {v: round(m(n, f'E_h_{v}'), 2) for v in V[2:]})
