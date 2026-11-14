"""Leaf 14: the tractability-learnability frontier (PREREG P12, P15) from the summaries of grids C, K, T, R and the PFN arm.
Learnability: sorting nMSE (mean over seeds) / ReLU's (pinned arms / pinned ReLU's); PFN test KL / GELU's.
Tractability, per arm: (a) trained sorting nets: K2-to-MC ratio nMSE_K2 * N_eq (median over seeds; < 1 = K2 beats MC at
equal FLOPs) and whether K2 also beats RQMC; (b) random nets: K2-to-MC ratio at (n = 16, L = 8) and (n = 8, L = 8);
(c) random nets: raw r_in(0.1) at (16, 8, 8) / ReLU's; (d) trained gates' tail share T_8/T_1.
usage: python3 scratch/pareto.py  -> logs/pareto.md, runs/pareto.json, logs/pareto.png
"""
import glob, json, math, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'runs')


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def sort_stats(prefix):
    v = [json.load(open(f))['nmse'] for f in sorted(glob.glob(os.path.join(R, 'sort', prefix + '_s*.json')))]
    return (float(np.mean(v)), v) if v else (None, [])


k2rows = load(os.path.join(R, 'k2_summary.json')) or []
k2 = {(r['arm'], r['n'], r['L']): r for r in k2rows}
an = {}
for f in glob.glob(os.path.join(R, 'an_cut_*_n16_c8_L8.json')):
    an[os.path.basename(f)[len('an_cut_'):-len('_n16_c8_L8.json')]] = json.load(open(f))
pfn = load(os.path.join(R, 'pfn_summary.json')) or {}
ARMS = ['relu', 'softplus16', 'softplus4', 'softplus1', 'gelu', 'silu', 'tanhb', 'smoothstep1', 'swiglu', 'bilinear']
PINS = ['relu', 'softplus16', 'softplus8', 'softplus4', 'softplus2', 'softplus1']
out = {'unpinned': {}, 'pinned': {}, 'prenorm': {}}
relu_sort = sort_stats('sort_relu_L16')[0]; relu_pin = sort_stats('pin_relu_L16')[0]; relu_pn = sort_stats('pn_relu_L16')[0]
for fam, arms, pre, base in (('unpinned', ARMS, 'sort', relu_sort), ('pinned', PINS, 'pin', relu_pin),
                             ('prenorm', ['relu', 'swiglu', 'bilinear'], 'pn', relu_pn)):
    for a in arms:
        m, v = sort_stats(f'{pre}_{a}_L16')
        Rj = load(os.path.join(R, f'R_{a}.json' if pre == 'sort' else f'R_{pre}_{a}.json'))
        row = dict(sort_nmse=m, sort_seeds=v, sort_rel=(m / base if (m and base) else None))
        if Rj and Rj['nets']:
            nets = list(Rj['nets'].values())
            if 'nmse_k2' in nets[0]:
                row['trained_k2_mc'] = float(np.median([x['nmse_k2'] * x['N_eq'] for x in nets]))
                row['trained_k2_wins'] = int(sum(x['k2_win'] for x in nets))
                row['trained_rqmc_gain'] = float(np.median([x['nmse_mc'] / x['nmse_rqmc_eq'] for x in nets]))
            row['trained_tail8'] = (None if a == 'bilinear' else      # identity gate: no nonlinear tail (undefined)
                                    float(np.median([x['tail_share']['8'] for x in nets if x['tail_share']['8'] is not None])))
            row['trained_rqmc1024_gain'] = float(np.median([(1 / 1024) / x['nmse_rqmc_1024'] for x in nets]))
            row['sharp_ratio'] = float(np.median([x['sharp_ratio_median'] for x in nets]))
        if fam == 'unpinned':
            for n in (8, 16):
                r_ = k2.get((a, n, 8))
                row[f'random_k2_mc_n{n}_L8'] = r_['ratio'] if r_ else None
            if a in an and 'relu' in an:
                row['random_rin01_rel'] = an[a]['raw']['r_in']['0.1'] / max(an['relu']['raw']['r_in']['0.1'], 1)
            if a in pfn:
                g = pfn.get('gelu', {}).get('kl_mean')
                row['pfn_kl_rel_gelu'] = pfn[a]['kl_mean'] / g if g else None
        out[fam][a] = row
json.dump(out, open(os.path.join(R, 'pareto.json'), 'w'), indent=1)
lines = ['| family | arm | sorting nMSE (rel. ReLU) | trained K2/MC | trained K2 wins | trained RQMC gain | trained tail T8/T1 | '
         'sharpening | random K2/MC n8,L8 / n16,L8 | random r_in(0.1) rel. ReLU (16,8,8) | PFN KL rel. GELU |', '|---|' * 1 + '---|' * 10]
f = lambda x, p='.3g': '-' if x is None else format(x, p)
for fam in ('unpinned', 'pinned', 'prenorm'):
    for a, r in out[fam].items():
        lines.append(f"| {fam} | {a} | {f(r.get('sort_nmse'), '.4f')} ({f(r.get('sort_rel'), '.2f')}) | {f(r.get('trained_k2_mc'))} | "
                     f"{r.get('trained_k2_wins', '-')} | {f(r.get('trained_rqmc_gain'))} | {f(r.get('trained_tail8'))} | "
                     f"{f(r.get('sharp_ratio'), '.2f')} | {f(r.get('random_k2_mc_n8_L8'))} / {f(r.get('random_k2_mc_n16_L8'))} | "
                     f"{f(r.get('random_rin01_rel'), '.2f')} | {f(r.get('pfn_kl_rel_gelu'), '.2f')} |")
open(os.path.join(HERE, '..', 'logs', 'pareto.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
