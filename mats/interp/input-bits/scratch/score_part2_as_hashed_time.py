"""Score PREREG Part 2 (H2.1-H2.6) -> results/part2_scores.json. Rules as registered (hash 841b6a0a)."""
import glob, json, math, os, sys
import numpy as np

D_ = '/home/agent/workspaces/mats/work/input-bits'
ST = '/home/agent/workspaces/mats/work/stationary/runs'
med = lambda v: float(np.median(v)) if len(v) else float('nan')
rho = {(v['d'], v['N']): v['rho2'] for v in json.load(open(f'{D_}/results/rho_calibration.json')).values()}
NN = [8, 16, 32, 64, 128]
NI = [32, 64, 128, 256]
S = {}
# ---------------- noise regime
noise = {}
for n in NN:
    for s in (1000, 1001, 4000, 4001):
        f = f'{D_}/runs/p2noise_n{n}_s{s}.json'
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        if s < 4000:
            q = json.load(open(f'{ST}/rqmc_n{n}_s{s}.json')); trv = json.load(open(f'{ST}/truth_n{n}_s{s}.json'))['trvar']
            anti = json.load(open(f'{ST}/anti_n{n}_s{s}.json'))['pair_var_over_half_trvar']
        else:
            q = json.load(open(f'{D_}/runs/p2rqmcnoise_n{n}_s{s}.json')); trv = r['V']; anti = None
        G = {x['N']: trv / x['N'] / x['var_sum'] for x in q['rows']}
        noise[(n, s)] = dict(S1=r['S1_win'], W1_all=r['W1_all'], s_even=r['s_even'], G=G, anti=anti, V=r['V'], trv=trv)
cellsN = {}
for n in NN:
    rs = [v for (nn, s), v in noise.items() if nn == n]
    if not rs:
        continue
    cellsN[n] = dict(S1=med([x['S1'] for x in rs]), s_even=med([x['s_even'] for x in rs]),
                     G1024=med([x['G'].get(1024, np.nan) for x in rs]),
                     E=med([x['G'].get(1024, np.nan) * (1 - x['S1']) for x in rs]),
                     odd3=med([1 - x['s_even'] - x['W1_all'] for x in rs]), nets=len(rs))
# ---------------- input regime
P1B = {}
for f in glob.glob(f'{D_}/runs/p1b_n*_s*.json'):
    r = json.load(open(f)); P1B[(r['n'], r['seed'])] = r
P2I = {}
for f in glob.glob(f'{D_}/runs/p2input_n*_s*.json'):
    r = json.load(open(f)); P2I[(r['n'], r['seed'])] = r
cellsI = {}
for n in NI:
    for L in sorted({max(1, n // 8), n // 4, n // 2, n, 32}):
        if L > 2 * n:
            continue
        S1 = [P1B[(n, s)]['by_depth'][str(L)]['S1'] for s in range(4000, 4004) if (n, s) in P1B]
        W2o = [P1B[(n, s)]['by_depth'][str(L)]['W2_off'] for s in range(4000, 4004) if (n, s) in P1B]
        for N in (256, 1024, 4096):
            G = [x['G'] for s in range(4000, 4004) if (n, s) in P2I for x in P2I[(n, s)]['by_depth'][str(L)]['rows'] if x['N'] == N]
            Gr = [x['G_res_true'] for s in range(4000, 4004) if (n, s) in P2I for x in P2I[(n, s)]['by_depth'][str(L)]['rows'] if x['N'] == N]
            if not G or not S1:
                continue
            r2 = rho[(n, N)]
            gp = 1 / (1 - med(S1) - (1 - r2) * med(W2o))
            gres_pred = 1 / (1 - (1 - r2) * med(W2o) / (1 - med(S1)))
            cellsI[(n, L, N)] = dict(n=n, L=L, N=N, G=med(G), S1=med(S1), W2_off=med(W2o), rho2=r2, G_pred=gp, ratio=med(G) / gp,
                                     floor=0.75 / (1 - med(S1)), G_res=med(Gr), G_res_pred=gres_pred)
# H2.1
okN = [c['G1024'] >= 0.75 / (1 - c['S1']) for c in cellsN.values()]
okI = [c['G'] >= c['floor'] for k, c in cellsI.items() if k[2] in (1024, 4096)]
S['H2.1'] = dict(hit=all(okN) and all(okI), noise=f'{sum(okN)}/{len(okN)}', input=f'{sum(okI)}/{len(okI)}',
                 input_misses=[(k, round(c['G'], 2), round(c['floor'], 2)) for k, c in cellsI.items() if k[2] in (1024, 4096) and c['G'] < c['floor']])
# H2.2
rat = [c['ratio'] for c in cellsI.values()]
inb = [1 / 1.5 <= x <= 1.5 for x in rat]
S['H2.2'] = dict(hit=(np.mean(inb) >= 0.8 if inb else False) and 0.8 <= med(rat) <= 1.25, frac_in_band=float(np.mean(inb)) if inb else None,
                 median_ratio=med(rat), cells=[dict(k=list(k), **{kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in c.items()}) for k, c in sorted(cellsI.items())])
# H2.3
a1 = all(abs(cellsN[m]['S1'] - cellsN[16]['S1']) <= 0.15 for m in (64, 128) if m in cellsN) and all(0.15 <= c['s_even'] <= 0.35 for c in cellsN.values())
b1 = cellsN.get(8, {}).get('E', 0) >= 1.5 and all(cellsN[m]['E'] <= 1.3 for m in (64, 128) if m in cellsN)
S['H2.3'] = dict(hit=a1 and b1, a=a1, b=b1, cells={n: c for n, c in cellsN.items()})
# H2.4
cons = [abs(2 * v['s_even'] - v['anti']) <= 0.08 for (n, s), v in noise.items() if v['anti'] is not None]
odd_ok = all(0.03 <= c['odd3'] <= 0.25 for c in cellsN.values()) and abs(cellsN[128]['odd3'] - cellsN[8]['odd3']) <= 0.10 if 128 in cellsN and 8 in cellsN else False
S['H2.4'] = dict(hit=all(cons) and odd_ok, consistency=f'{sum(cons)}/{len(cons)}', odd3={n: c['odd3'] for n, c in cellsN.items()},
                 pairs=[(n, s, round(2 * v['s_even'], 3), round(v['anti'], 3)) for (n, s), v in noise.items() if v['anti'] is not None])
# H2.5
M = {}
for f in glob.glob(f'{D_}/runs/p2mlmc_n*_s*.json'):
    r = json.load(open(f)); M[(r['n'], r['seed'])] = r
okA, detA = [], []
best = {}
for n in NI:
    Ls = sorted({L for L in {n // 4, n // 2, n, 2 * n, 32} if L >= n // 4})
    for L in Ls:
        rs = [M[(n, s)]['by_depth'][str(L)] for s in range(4000, 4004) if (n, s) in M]
        if not rs:
            continue
        best[(n, L)] = med([x['best_gain2'] for x in rs])
        if n < 64:
            continue
        for r in (L // 8, L // 4, L // 2):
            m = [x for y in rs for x in y['rows'] if x['r'] == r]
            if not m:
                continue
            q = med([x['V2_share'] for x in m]) / med([x['pred_share'] for x in m])
            okA.append(1 / 1.5 <= q <= 1.5); detA.append(dict(n=n, L=L, r=r, q=q))
bLn = [best.get((n, n)) for n in (64, 128, 256)]
bL32 = [best.get((n, 32)) for n in (64, 128, 256)]
b_ok = None not in bLn and bLn[0] < bLn[1] < bLn[2] and None not in bL32 and bL32[2] <= bL32[0]
S['H2.5'] = dict(hit=(np.mean(okA) >= 0.75 if okA else False) and b_ok, a_frac=float(np.mean(okA)) if okA else None, b=b_ok,
                 best_L_n=bLn, best_L32=bL32, best_all={f'{k[0]},{k[1]}': v for k, v in best.items()}, a_cells=detA)
# H2.6 verdicts
verd = {}
if 16 in cellsN and 128 in cellsN:
    dS = cellsN[16]['S1'] - cellsN[128]['S1']
    fallG = cellsN[16]['G1024'] / cellsN[128]['G1024']
    Es = [cellsN[m]['E'] for m in (16, 32, 64, 128)]
    mono = all(np.diff(Es) > 0) or all(np.diff(Es) < 0)
    floor = all(okN)
    verd['noise'] = ('SUPPORTED' if dS >= 0.10 and floor and not mono else 'REFUTED' if abs(dS) < 0.10 and fallG >= 1.5 else 'UNRESOLVED',
                     dict(dS1=dS, G_fall=fallG, E_by_n=Es))
for tag, cells in (('input L=32', [(n, 32) for n in (64, 128, 256)]), ('input L=n/2', [(n, n // 2) for n in (64, 128, 256)])):
    try:
        S1s = [cellsI[(n, L, 1024)]['S1'] for n, L in cells]
        Gs = [cellsI[(n, L, 1024)]['G'] for n, L in cells]
        rs = [cellsI[(n, L, 1024)]['ratio'] for n, L in cells]
    except KeyError:
        continue
    dS = S1s[0] - S1s[-1]; fallG = Gs[0] / Gs[-1]
    mono = all(np.diff(rs) > 0) or all(np.diff(rs) < 0)
    floor = all(cellsI[(n, L, N)]['G'] >= cellsI[(n, L, N)]['floor'] for n, L in cells for N in (1024, 4096))
    verd[tag] = ('SUPPORTED' if dS >= 0.10 and floor and not mono else 'REFUTED' if abs(dS) < 0.10 and fallG >= 1.5 else 'UNRESOLVED',
                 dict(S1=S1s, G1024=Gs, ratio=rs))
S['H2.6'] = dict(hit=None, verdicts=verd)
json.dump(S, open(f'{D_}/results/part2_scores.json', 'w'), indent=1, default=float)
for k, v in S.items():
    print(k, 'HIT' if v['hit'] else ('MISS' if v['hit'] is False else '-'),
          json.dumps({kk: vv for kk, vv in v.items() if kk not in ('cells', 'a_cells', 'hit')}, default=float)[:700])
