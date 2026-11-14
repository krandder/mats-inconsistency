"""Assemble the evaluation tables (markdown) from runs/*.json. usage: python analyze.py [OUT.md]"""
import sys, json, glob, math, os
import numpy as np

R = '/home/agent/workspaces/mats/work/narrow-deep-m1/runs'
HEALTHY = [203, 216, 217, 220, 221, 223]
SECOND = [200, 201, 202, 206]
out = []
P = out.append


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def rqmc_fn(S, small=None):
    xs, ys = [], []
    items = dict(S['est'])
    if small:
        items.update(small['est'])
    for lg, d in sorted(items.items(), key=lambda kv: int(kv[0])):
        if int(lg) >= 20:      # 2^20 is at the truth's resolution (review fix): never used
            continue
        if 'RQMC' in d and d['RQMC']['nmse'] > 0:
            xs.append(math.log(2 ** int(lg))); ys.append(math.log(d['RQMC']['nmse']))
    xs, ys = np.array(xs), np.array(ys)

    def f(N):
        x = math.log(max(N, 1e-9))
        if len(xs) < 2:
            return float('nan')
        if x <= xs[0]:
            s = (ys[1] - ys[0]) / (xs[1] - xs[0]); return math.exp(ys[0] + s * (x - xs[0]))
        if x >= xs[-1]:
            s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]); return math.exp(ys[-1] + s * (x - xs[-1]))
        return math.exp(float(np.interp(x, xs, ys)))
    return f


def g(x, f='.2g'):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return '-'
    return format(x, f)


for sig in ('0.001', '0.01'):
    P(f'\n## sigma = {sig}\n')
    # truths
    P('| net | class | tr Var(a_L) | E^2/Var | p(1e-4) | truth nMSE | noise effect vs sigma=0 truth (nMSE) |')
    P('|---|---|---|---|---|---|---|')
    for s in HEALTHY + SECOND:
        T = load(f'{R}/truth_s{s}_sig{sig}.json')
        if not T:
            continue
        T0 = load(f'{R}/truth_s{s}_sig0.json')
        ne = None
        if T0 and T['trvar'] > 0:
            ne = float(np.sum((np.array(T['mean']) - np.array(T0['mean'])) ** 2) / T['trvar'])
        cls = 'healthy' if s in HEALTHY else ('dead-noise' if s in (200, 201) else 'intermediate')
        P(f"| {s} | {cls} | {T['trvar']:.3g} | {np.sum(np.array(T['mean'])**2)/max(T['trvar'],1e-300):.3g} | {T['tails']['0.0001']['p']:.3g} | {1/T['paths']:.1g} | {g(ne)} |")
    # samplers
    P('\n| net | RQMC nMSE at 2^7 / 2^9 / 2^11 / 2^13 / 2^15 / 2^17 / 2^20 | RQMC x N at 2^13 | best HYB at 2^9 / 2^13 / 2^17 |')
    P('|---|---|---|---|')
    for s in HEALTHY + SECOND:
        S = load(f'{R}/samp_s{s}_sig{sig}.json')
        if not S:
            continue
        rq = [S['est'].get(str(l), {}).get('RQMC', {}).get('nmse') for l in (7, 9, 11, 13, 15, 17, 20)]
        hy = []
        for l in (9, 13, 17):
            d = S.get('hyb', {}).get(str(l), {})
            hy.append(min([v['nmse'] for v in d.values()]) if d else None)
        r13 = S['est'].get('13', {}).get('RQMC', {}).get('nmse')
        P(f"| {s} | {' / '.join(g(v) for v in rq)} | {g(r13 * 8192 if r13 else None)} | {' / '.join(g(v) for v in hy)} |")
    # TS
    P('\n| net | TS config | cost (paths) | nMSE | RQMC at that cost | RQMC / TS |')
    P('|---|---|---|---|---|---|')
    for s in HEALTHY + SECOND:
        S = load(f'{R}/samp_s{s}_sig{sig}.json'); E = load(f'{R}/ts_s{s}_sig{sig}.json')
        if not S or not E:
            continue
        f = rqmc_fn(S, load(f'{R}/rqmcsmall_s{s}_sig{sig}.json'))
        for r in E['rows']:
            rq = f(r['cost_paths'])
            T_ = load(f'{R}/truth_s{s}_sig{sig}.json'); res10 = 10.0 / T_['paths']
            ratio = f"{rq/r['nmse']:.2g}" if r['nmse'] >= res10 else 'unresolved (nMSE < 10/P)'
            P(f"| {s} | {r['F']} c={r['c']} tau={r['tau']:g} | {r['cost_paths']:.0f} | {r['nmse']:.2g} | {rq:.2g} | {ratio} |")
    # BO
    P('\n| net | BO (norm dirs) k / m | nodes | cost (paths) | nMSE | RQMC at cost | RQMC / BO | BO-full nMSE, cost | oracle-dir nMSE |')
    P('|---|---|---|---|---|---|---|---|---|')
    for s in HEALTHY + SECOND:
        S = load(f'{R}/samp_s{s}_sig{sig}.json'); E = load(f'{R}/bo_s{s}_sig{sig}.json')
        if not S or not E:
            continue
        f = rqmc_fn(S, load(f'{R}/rqmcsmall_s{s}_sig{sig}.json'))
        orc = {(r['k'], r['m']): r['nmse'] for r in E['rows'] if r['mode'] == 'oracle'}
        for r in E['rows']:
            if r['mode'] != 'norm':
                continue
            rq = f(r['cost_paths'])
            P(f"| {s} | {r['k']} / {r['m']} | {r['nodes']} | {r['cost_paths']:.0f} | {r['nmse']:.2g} | {rq:.2g} | {rq/r['nmse']:.2g} | {r['nmse_full']:.2g}, {r['cost_full']:.0f} | {g(orc.get((r['k'], r['m'])))} |")
    # SH
    P('\n| net | SH-direct nMSE at 2^9 / 2^11 / 2^13 / 2^15 / 2^17 | RQMC same budgets | ratio RQMC / SH |')
    P('|---|---|---|---|')
    for s in HEALTHY + SECOND:
        S = load(f'{R}/samp_s{s}_sig{sig}.json'); E = load(f'{R}/sh_s{s}_sig{sig}.json')
        if not S or not E:
            continue
        sh = [E['est'].get(str(l), {}).get('SH-direct', {}).get('nmse') for l in (9, 11, 13, 15, 17)]
        rq = [S['est'].get(str(l), {}).get('RQMC', {}).get('nmse') for l in (9, 11, 13, 15, 17)]
        ra = [(b / a if (a and b and a > 0) else None) for a, b in zip(sh, rq)]
        P(f"| {s} | {' / '.join(g(v) for v in sh)} | {' / '.join(g(v) for v in rq)} | {' / '.join(g(v) for v in ra)} |")
    # tails
    P('\n| net | tail q=1e-4: rel. MSE at 2^13 / 2^17: MC (exact) | RQMC | RB |')
    P('|---|---|---|---|')
    for s in HEALTHY + SECOND:
        S = load(f'{R}/samp_s{s}_sig{sig}.json'); T = load(f'{R}/truth_s{s}_sig{sig}.json')
        if not S or not T:
            continue
        p = T['tails']['0.0001']['p']
        mc = [(1 - p) / (2 ** l * p) for l in (13, 17)]
        rq = [S['tails'].get(str(l), {}).get('0.0001', {}).get('RQMC', {}).get('rel_mse') for l in (13, 17)]
        rb = [S['tails'].get(str(l), {}).get('0.0001', {}).get('RB', {}).get('rel_mse') for l in (13, 17)]
        P(f"| {s} | {' / '.join(g(v) for v in mc)} | {' / '.join(g(v) for v in rq)} | {' / '.join(g(v) for v in rb)} |")

# forward part
P('\n## Forward part (sigma = 1e-3)\n')
P('| net | method | nMSE of E[a_c], c = 1..8 | meet e_c at c = 2 / 4 / 8 | FLOPs (paths) to c = 8 |')
P('|---|---|---|---|---|')
for s in HEALTHY:
    F = load(f'{R}/fwd_s{s}_sig0.001.json')
    if not F:
        continue
    fp = F['flops_path']
    for name, row in F['methods'].items():
        if 'error' in row:
            P(f"| {s} | {name} | failed: {row['error'][:60]} | | |"); continue
        ea = ' '.join(g(row['err_a'][str(c)] if str(c) in row['err_a'] else row['err_a'].get(c)) for c in range(1, 9))
        mt = row.get('meet_gauss') or row.get('meet') or {}
        me = ' / '.join(g(mt.get(str(c), mt.get(c))) for c in (2, 4, 8))
        mp = row.get('meet_poly', {})
        if mp:
            best = min([v for v in mp.values() if v is not None] or [float('nan')])
            me += f" (poly best {g(best)})"
        fl8 = row['flops_at'].get('8', row['flops_at'].get(8))
        P(f"| {s} | {name} | {ea} | {me} | {g(fl8 / fp if fl8 else None, '.3g')} |")
# k by cut
P('\n## Directions needed per cut (Amendment 1): smallest k with projected-meet nMSE <= 1e-5 (<= 1e-4), S oracle / probe\n')
P('| net | ' + ' | '.join(f'c={c}' for c in range(9)) + ' |')
P('|---|' + '---|' * 9)
for s in HEALTHY:
    K = load(f'{R}/kbycut2_s{s}_sig0.001.json')      # corrected oracle (review): observation directions C^-1/2 V
    if not K:
        continue
    cells = []
    for c in range(9):
        row = K['cuts'].get(str(c))
        if not row:
            cells.append('-'); continue
        def kneed(tag, thr):
            # registered statistic: smallest k with projected-meet nMSE <= thr; post-hoc "stable" k in brackets
            es = row['projmeet'][tag]
            sm = next((e['k'] for e in es if e['nmse'] <= thr), 8)
            st = next((e['k'] for i, e in enumerate(es) if all(x['nmse'] <= thr for x in es[i:])), 8)
            return f"{sm}" + (f"[{st}]" if st != sm else '')
        cells.append(f"{kneed('S', 1e-5)} ({kneed('S', 1e-4)}) / {kneed('P', 1e-5)} ({kneed('P', 1e-4)})")
    P(f"| {s} | " + ' | '.join(cells) + ' |')
P('\n## Downstream-distinct classes (99% of |a_L| mass) vs upstream histories (2^16 inputs)\n')
P('| net | c=1 | c=2 | c=4 | c=8 | c=12 | c=16 | c=24 | c=32 |')
P('|---|---|---|---|---|---|---|---|---|')
for s in HEALTHY:
    K = load(f'{R}/kbycut_s{s}_sig0.001.json')
    if not K:
        continue
    cl = K['classes']
    P(f"| {s} | " + ' | '.join(f"{cl[str(c)]['down']['256']['w99']} vs {cl[str(c)]['up']['distinct']}" if str(c) in cl else '-' for c in (1, 2, 4, 8, 12, 16, 24, 32)) + ' |')
txt = '\n'.join(out)
if len(sys.argv) > 1:
    open(sys.argv[1], 'w').write(txt)
print(txt)
