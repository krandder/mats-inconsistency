"""Grid C table and PREREG scoring (P1-P4, with Amendment 1's tanhb for P3).
usage: python3 scratch/score_C.py  (reads runs/an_cut_*.json, logs/runs.jsonl) -> logs/C_table.md, runs/C_scores.json"""
import glob, json, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..', 'runs')
meta = {}
for line in open(os.path.join(HERE, '..', 'logs', 'runs.jsonl')):
    d = json.loads(line); meta[d['run']] = d
CFG = {8: 'n8_c4_L4', 16: 'n16_c8_L8'}
ARMS = ['relu', 'softplus16', 'softplus4', 'softplus1', 'gelu', 'silu', 'gsrelu0.5', 'smoothstep1', 'tanhb', 'tanh']
SHARP = ['softplus16', 'softplus4', 'gelu', 'silu', 'gsrelu0.5', 'smoothstep1']      # P1's explicit list
ANALYTIC = ['softplus16', 'softplus4', 'softplus1', 'gelu', 'silu', 'gsrelu0.5', 'tanhb']
tab = {}
for n, cfg in CFG.items():
    for arm in ARMS:
        f = os.path.join(R, f'an_cut_{arm}_{cfg}.json')
        if os.path.exists(f):
            tab[(arm, n)] = json.load(open(f))
    for arm in ('relu', 'softplus4'):
        f = os.path.join(R, f'an_cut_{arm}_{cfg}_ctrl.json')
        if os.path.exists(f):
            tab[(arm + '_ctrl', n)] = json.load(open(f))


def r(arm, n, kind='raw', est='r_in', e='0.1'):
    t = tab.get((arm, n))
    return None if t is None else t[kind][est][e]


def secant(arm, n, kind='raw'):
    a, b = r(arm, n, kind, 'r_in', '0.1'), r(arm, n, kind, 'r_in', '0.03')
    if not a or not b:
        return None
    return math.log(b / a) / math.log(0.1 / 0.03)


lines = ['| config | arm | chi_1 | raw r_in 0.3/0.1/0.03/0.01 | raw r_ho 0.1 | scale-norm r_in 0.3/0.1/0.03/0.01 | secant 0.1->0.03 raw / norm | tail_1/Vbar | Vbar/E M^2 | energy share: fronts/backs carrying 80% |',
         '|---|---|---|---|---|---|---|---|---|---|']
for n, cfg in CFG.items():
    for arm in ARMS + ['relu_ctrl', 'softplus4_ctrl']:
        t = tab.get((arm, n))
        if t is None:
            continue
        m = meta.get(f'cut_{arm}_{cfg}' if not arm.endswith('_ctrl') else f'cut_{arm[:-5]}_{cfg}_ctrl', {})
        chi = m.get('chi')
        s1, s2 = secant(arm, n), secant(arm, n, 'scale_norm')
        lines.append(f"| ({n},{cfg.split('_')[1][1:]},{cfg.split('_')[2][1:]}) | {arm} | {'' if chi is None else f'{chi:.2f}'} | "
                     f"{'/'.join(str(t['raw']['r_in'][e]) for e in ('0.3', '0.1', '0.03', '0.01'))} | {t['raw']['r_ho']['0.1']} | "
                     f"{'/'.join(str(t['scale_norm']['r_in'][e]) for e in ('0.3', '0.1', '0.03', '0.01'))} | "
                     f"{'-' if s1 is None else f'{s1:.2f}'} / {'-' if s2 is None else f'{s2:.2f}'} | {t['raw']['tail1_rel']:.3f} | "
                     f"{(m['Vbar'] / m['EM2']) if m.get('EM2') else float('nan'):.3g} | {m.get('front_share80', float('nan')):.2f} / {m.get('back_share80', float('nan')):.2f} |")
scores = {}
# resolution of eps = 0.01: both controls' |tail_in(1)| < 1e-4 Vbar
for n in CFG:
    ok = all(abs(tab[(c, n)]['raw']['tail_in_rel'][1]) < 1e-4 for c in ('relu_ctrl', 'softplus4_ctrl') if (c, n) in tab)
    scores[f'eps0.01_resolved_n{n}'] = ok if all((c, n) in tab for c in ('relu_ctrl', 'softplus4_ctrl')) else None
# P1
p1 = []
for n in CFG:
    rr = r('relu', n)
    for arm in SHARP:
        v = r(arm, n)
        if v is not None and rr is not None:
            p1.append(dict(n=n, arm=arm, r=v, relu=rr, below=v < (2 / 3) * rr))
scores['P1'] = dict(rows=p1, refuted=any(x['below'] for x in p1))
# P2
if r('softplus1', 8) is not None:
    scores['P2'] = dict(softplus1=r('softplus1', 8), relu=r('relu', 8), refuted=r('softplus1', 8) < r('relu', 8))
# P3 (tanhb, Amendment 1)
p3 = [dict(n=n, tanhb=r('tanhb', n), relu=r('relu', n)) for n in CFG if r('tanhb', n) is not None]
scores['P3'] = dict(rows=p3, refuted=any(x['tanhb'] < x['relu'] for x in p3), complete=len(p3) == 2)
# P4
p4 = []
for n in CFG:
    sr = secant('relu', n)
    for arm in ARMS:
        s = secant(arm, n)
        if s is not None:
            p4.append(dict(n=n, arm=arm, secant=s, relu=sr))
ge2 = any(x['secant'] >= 2 for x in p4)
below = [a for a in ANALYTIC if all((x['relu'] is not None and x['secant'] <= x['relu'] - 0.4) for x in p4 if x['arm'] == a)
         and sum(1 for x in p4 if x['arm'] == a) == 2]
scores['P4'] = dict(rows=p4, any_ge2=ge2, analytic_below_relu_by_0p4_at_both=below, refuted=ge2 or bool(below))
open(os.path.join(HERE, '..', 'logs', 'C_table.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(scores, open(os.path.join(R, 'C_scores.json'), 'w'), indent=1)
print('\n'.join(lines)); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'rows'}) for k, v in scores.items()}, indent=0))
