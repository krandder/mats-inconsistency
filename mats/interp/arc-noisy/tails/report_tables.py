"""Markdown tables for the report from analyze_tails.py --json outputs (results/summary_s*.json).
python tails/report_tables.py results/summary_s0.001.json [more ...]"""
import json, math, sys

KS = ('4', '6', '8', '10', '12')
NAMES = [('C-MC', 'Monte Carlo (analytic)'), ('C-Kd', 'Kerdock design'), ('C-GS', 'splitting, no psi (per event)'),
         ('C-K2', 'kprop K = 2'), ('C-K3f', "ARC's tail method (factorized K = 3)"), ('C-K3fa', 'factorized K = 3, augmented'),
         ('C-MCx-gauss', 'MC moments + Gaussian tail'), ('C-MCx-edge', 'MC moments + Edgeworth tail'),
         ('U3', 'ours: MC moments + saddlepoint'), ('U3r', 'ours: MC moments + saddlepoint, chi mixture'),
         ('U1', 'ours: mode splitting (rule < 1)'), ('U1m', 'ours: mode splitting (rule <= 0.81)'),
         ('U2', 'post hoc: saddlepoint on the K3f description'), ('U2r', 'post hoc: same, chi mixture'),
         ('U2a', 'post hoc: saddlepoint on the K3f augmented description')]


def f(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '-'
    return 'never' if not math.isfinite(x) else ('%.1e' % x).replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')


for path in sys.argv[1:]:
    S = json.load(open(path))
    sig = path.split('_s')[-1].rstrip('.json')
    for L in ('4', '8', '32'):
        for valid in ('all', 'valid'):
            print(f'\n### sigma = {sig}, L = {L}, {"all events" if valid == "all" else "validated events only"}\n')
            print('| method | ' + ' | '.join(f'1e-{k}' for k in KS) + ' |')
            print('|---|' + '---|' * 5)
            for key, lab in NAMES:
                ev = S.get(f'{sig}|{L}|{key}|ev|{valid}'); am = S.get(f'{sig}|{L}|{key}|am|{valid}')
                if ev is None:
                    continue
                cells = []
                for k in KS:
                    e, a = ev.get(k), am.get(k)
                    if e is None:
                        cells.append('-'); continue
                    cells.append(f'{f(e["b50"])} / {f(a["b50"])} ({e["reached"]:.2f})')
                print(f'| {lab} | ' + ' | '.join(cells) + ' |')
            for ours in ('U3', 'U3r', 'U1', 'U1m'):
                for am in ('ev', 'am'):
                    w = S.get(f'{sig}|{L}|win|{ours}|{am}|{valid}')
                    if w is None:
                        continue
                    print(f'\n{ours} vs best sampling/hybrid competitor ({"amortised" if am == "am" else "per event"}): ' + '; '.join(
                        f'1e-{k}: {w[k]["best"]} {f(w[k]["ratio"])} [{f(w[k]["lo"])}, {f(w[k]["hi"])}]' + (' WIN' if w[k]['lo'] > 1 else '')
                        for k in KS if k in w))
