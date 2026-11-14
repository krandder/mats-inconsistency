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
            rec = dict(n=n, L=L, seed=s, method=meth, accounting=key, exponent=sl, exponent_last=sl_last, gain=g)
            if key != 'cost' and Cstar < pts[0, 0]:
                # review r2: below the method's own setup cost there is no estimator; keep the old number only as history
                rec.update(gain=None, unaffordable=True, gain_extrapolated_historical=g)
            out.append(rec)
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
    gv = [x['gain'] for x in v if x['gain'] is not None]
    gtxt = f'{np.median(gv):.2f}' if gv else 'UNAFFORDABLE'
    print(f'| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {gtxt} | {np.median([x["exponent"] for x in v]):.2f} | '
          f'{np.median([x["exponent_last"] for x in v]):.2f} |')

# ---------------- rules P3.1-P3.6 (implemented after the hash, as flagged by astra round 2; Amendment 2C)
# Charged accounting: a gain at C* is UNAFFORDABLE when the method's setup (pilot) alone exceeds C*. Post hoc accounting:
# the gain at the method's largest measured budget, charged, against MC at the same budget.
REQ = [(n, L, s) for n in (64, 128, 256) for L in sorted({32, n // 2, n, 2 * n}) for s in (4004, 4005)]
have = {(r['n'], r['L'], r['seed']) for r in runs}
missing = [k for k in REQ if k not in have]
RQ = ['RQMC', 'RQMC-RB', 'RQMC-ROT', 'RQMC-CVQ', 'ML2-RQMC']


def get(n, L, meth, acc='cost'):
    return [o for o in out if o['n'] == n and o['L'] == L and o['method'] == meth and o['accounting'] == acc]


def mgain(n, L, meth, acc='cost'):
    v = [o['gain'] for o in get(n, L, meth, acc) if o['gain'] is not None]
    return float(np.median(v)) if v else None


def setup_cost(r, meth):
    rows = r['methods'][meth]
    for key in ('cost_pilot', 'cost_charged'):
        if key in rows[0]:
            return min(x[key] - x['cost'] for x in rows)
    return 0.0


R3 = {'missing': missing}
# P3.1
ok, det = [], []
for n in (64, 128, 256):
    for L in sorted({32, n // 2, n, 2 * n}):
        for meth in RQ:
            v = [o['exponent'] for o in get(n, L, meth)]
            if not v:
                continue
            e = float(np.median(v))
            band = (0.9, 1.3) if (n == 256 and L >= 128) else (0.9, 1.6)
            ok.append(band[0] <= e <= band[1]); det.append(dict(n=n, L=L, method=meth, exponent=e, band=band))
R3['P3.1'] = dict(hit=all(ok) if ok else None, n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
g64, g256 = mgain(64, 32, 'RQMC-RB'), mgain(256, 32, 'RQMC-RB')
R3['P3.2'] = dict(hit=(g256 < g64) if None not in (g64, g256) else None, g64=g64, g256=g256)
ok, det = [], []
for r in runs:
    if r['L'] >= r['n'] // 2:
        ok.append(0.35 <= r['cvq']['ratio'] <= 0.75); det.append(dict(n=r['n'], L=r['L'], seed=r['seed'], ratio=r['cvq']['ratio']))
R3['P3.3'] = dict(hit=all(ok) if ok else None, n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
ok, det = [], []
for n in (64, 128, 256):
    for L in sorted({32, n // 2, n, 2 * n}):
        if L < n // 2:
            continue
        a_, b_ = mgain(n, L, 'RQMC-ROT'), mgain(n, L, 'RQMC-RB')
        if None not in (a_, b_):
            ok.append(a_ >= 1.2 * b_); det.append(dict(n=n, L=L, rot=a_, rb=b_))
R3['P3.4'] = dict(hit=(sum(ok) >= len(ok) / 2) if ok else None, n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
ml = [mgain(n, 2 * n, 'ML2') for n in (64, 128, 256)]
ok5a = None not in ml and ml[0] < ml[1] < ml[2] and ml[2] >= 1.5
ok, det = [], []
for n in (64, 128, 256):
    for L in sorted({32, n // 2, n, 2 * n}):
        if L < n:
            continue
        e_, a_, b_ = mgain(n, L, 'ML2-RQMC'), mgain(n, L, 'RQMC-RB'), mgain(n, L, 'ML2')
        if None not in (e_, a_, b_):
            ok.append(e_ >= max(a_, b_)); det.append(dict(n=n, L=L, ml2rqmc=e_, rqmcrb=a_, ml2=b_))
R3['P3.5'] = dict(hit=bool(ok5a and sum(ok) >= len(ok) / 2), ML2_L2n=ml, envelope_ok=f'{sum(ok)}/{len(ok)}', cells=det)
# P3.6 as registered: charged gain at C*; UNAFFORDABLE counts as no progress
prog, det = [], []
byrun = {(r['n'], r['L'], r['seed']): r for r in runs}
for meth in ['RBMC', 'RQMC', 'RQMC-RB', 'RQMC-ROT', 'CVQ', 'RQMC-CVQ', 'ML2', 'MLD', 'ML2-RQMC']:
    gs = {}
    for n in (64, 256):
        L = 2 * n
        vals = []
        for s in (4004, 4005):
            r = byrun.get((n, L, s))
            if not r:
                continue
            sc = setup_cost(r, meth)
            if sc >= (1 << 12) * L:
                vals.append(None)
            else:
                acc = 'cost' if sc == 0 else ('cost_pilot' if 'cost_pilot' in r['methods'][meth][0] else 'cost_charged')
                v = [o['gain'] for o in get(n, L, meth, acc) if o['seed'] == s and o['gain'] is not None]
                vals.append(v[0] if v else None)
        gs[n] = 'UNAFFORDABLE' if any(x is None for x in vals) else (float(np.median(vals)) if vals else None)
    grows = isinstance(gs.get(64), float) and isinstance(gs.get(256), float) and gs[256] >= 1.5 * gs[64]
    prog.append(grows); det.append(dict(method=meth, gain_64=gs.get(64), gain_256=gs.get(256), grows_1p5=grows))
rate = []
for L in (128, 256, 512):
    for meth in RQ:
        e = [o['exponent'] for o in get(256, L, meth)]; el = [o['exponent_last'] for o in get(256, L, meth)]
        if e and el and L >= 256 and float(np.median(e)) >= 1.2 and float(np.median(el)) >= 1.2:
            rate.append(dict(L=L, method=meth))
R3['P3.6'] = dict(verdict='PROGRESS' if any(prog) or rate else 'constants only', width_growth=det, rate_hits=rate)
# post hoc: charged gain at each method's largest budget (pilot amortised over its own run), and pilot-free gains
ph = []
for (n, L, s), r in byrun.items():
    for meth, rows in r['methods'].items():
        key = 'cost_pilot' if 'cost_pilot' in rows[0] else ('cost_charged' if 'cost_charged' in rows[0] else 'cost')
        big = max(rows, key=lambda x: x[key])
        ph.append(dict(n=n, L=L, seed=s, method=meth, budget_mv=big[key], gain_charged=(L / big[key]) / big['nmse'],
                       gain_free=(L / big['cost']) / big['nmse']))
R3['posthoc_largest_budget'] = ph
json.dump(dict(per_run=out, rules=R3), open(f'{D_}/results/part3_scores.json', 'w'), indent=1, default=float)
for k in ('missing', 'P3.1', 'P3.2', 'P3.3', 'P3.4', 'P3.5', 'P3.6'):
    v = R3[k]
    print(k, json.dumps(v if k == 'missing' else {kk: vv for kk, vv in v.items() if kk not in ('cells',)}, default=float)[:700])
