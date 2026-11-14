"""Score PREREG Part 3 from runs/p3_n*_L*_s*.json -> results/part3_scores.json and a table.
Per (n, L, seed, method): exponent = -slope of log nMSE on log cost over the measured budgets (cost in mv; for ML2-RQMC the
envelope over its allocations at each N2); gain = nMSE_MC(C*) / nMSE_method(C*) at C* = 2^12 L mv (log-log interpolation;
pilot charged where the method lists cost_pilot / cost_charged, and also without)."""
import glob, json, math, sys
import numpy as np

D_ = '/home/agent/workspaces/mats/work/input-bits'
runs = [json.load(open(f)) for f in sorted(glob.glob(f'{D_}/runs/p3_n*_L*_s*.json'))]


def curve(rows, key='cost'):
    pts = sorted((r[key], r['nmse']) for r in rows if r['nmse'] > 0)
    return np.array(pts)


def envelope(rows):
    best = {}
    for r in rows:
        k = r['N2']
        if k not in best or r['nmse'] * r['cost'] < best[k]['nmse'] * best[k]['cost']:
            best[k] = r
    return list(best.values())


def interp_gain(pts, Cstar):
    """MC has nMSE = L/C. Gain at C* by log-log interpolation (or nearest-end extrapolation with the local slope)."""
    c, e = np.log(pts[:, 0]), np.log(pts[:, 1])
    x = math.log(Cstar)
    if x <= c[0]:
        i = 0
    elif x >= c[-1]:
        i = len(c) - 2
    else:
        i = int(np.searchsorted(c, x)) - 1
    i = max(0, min(i, len(c) - 2))
    y = e[i] + (e[i + 1] - e[i]) * (x - c[i]) / (c[i + 1] - c[i])
    return math.exp(y)


out = []
for r in runs:
    n, L, s = r['n'], r['L'], r['seed']
    Cstar = (1 << 12) * L
    for meth, rows in r['methods'].items():
        rows2 = envelope(rows) if meth == 'ML2-RQMC' else rows
        for key in ('cost', 'cost_pilot', 'cost_charged'):
            if key != 'cost' and key not in rows2[0]:
                continue
            pts = curve(rows2, key)
            if len(pts) < 2:
                continue
            sl = -float(np.polyfit(np.log(pts[:, 0]), np.log(pts[:, 1]), 1)[0])
            sl_last = -float((math.log(pts[-1, 1]) - math.log(pts[-2, 1])) / (math.log(pts[-1, 0]) - math.log(pts[-2, 0])))
            g = (L / Cstar) / interp_gain(pts, Cstar)
            out.append(dict(n=n, L=L, seed=s, method=meth, accounting=key, exponent=sl, exponent_last=sl_last, gain=g))
json.dump(out, open(f'{D_}/results/part3_scores.json', 'w'), indent=1)
# table: median over seeds
import collections
tab = collections.defaultdict(list)
for o in out:
    tab[(o['method'], o['accounting'], o['n'], o['L'])].append(o)
print('| method | accounting | n | L | gain at 2^12 L mv (median) | exponent | last-step exponent |')
print('|---|---|---|---|---|---|---|')
for k in sorted(tab):
    v = tab[k]
    print(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {np.median([x["gain"] for x in v]):.2f} | {np.median([x["exponent"] for x in v]):.2f} | '
          f'{np.median([x["exponent_last"] for x in v]):.2f} |')
