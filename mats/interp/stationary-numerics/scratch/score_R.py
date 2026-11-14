"""Score PREREG Part R (radius law) -> results/partR_scores.json, results/partR_tables.md."""
import json, math, os
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary-numerics'
SL = '/home/agent/workspaces/mats/work/stationary'
rc = json.load(open(f'{D}/results/radius_corrections.json'))
ns = sorted(int(k) for k in rc)
S = {}
tab = []

# R1
xs_reg = [0.1, 0.15, 0.2, 0.3, 0.4, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0]
ok1 = True
r1rows = []
for n in ns:
    if n < 64:
        continue
    rows = [r for r in rc[str(n)]['cdf_rows'] if any(abs(r['x'] - x) < 1e-9 for x in xs_reg) or abs(r['x'] - 0.4 / math.log(2)) < 1e-9]
    m1 = max(abs(r['R1']) for r in rows)
    mF2 = max(abs(r['F2']) for r in rows)
    r1rows.append((n, m1, 1.5 * mF2 / n))
    ok1 &= m1 <= 1.5 * mF2 / n
ratios = {}
for xv in (0.4 / math.log(2), 1.0, 2.0):
    a = [r for r in rc['512']['cdf_rows'] if abs(r['x'] - xv) < 1e-9][0]['R2']
    b = [r for r in rc['1024']['cdf_rows'] if abs(r['x'] - xv) < 1e-9][0]['R2']
    ratios['%.4f' % xv] = a / b
ok1b = all(1.6 <= v <= 2.6 for v in ratios.values())
S['R1'] = dict(hit=bool(ok1 and ok1b), max_R1_vs_bound=r1rows, R2_ratio_512_1024=ratios)

# R2 median
def med_formula(n):
    return 0.5770780164 - 0.0632661929 / n + 0.5192608 / n ** 2
med = {n: rc[str(n)]['median_over_n'] for n in ns}
dev = {n: med[n] - med_formula(n) for n in ns}
ok2 = all(abs(dev[n]) < 1e-7 for n in ns if n >= 256) and abs(dev[16]) < 5e-3
S['R2'] = dict(hit=bool(ok2), deviation={str(n): dev[n] for n in ns}, rel_dev_n2=dev[2] / med_formula(2))

# R3 Kesten tail
k8 = {n: rc[str(n)]['tail_xP']['100000000.0'] / rc[str(n)]['C_n'] - 1 for n in ns}
k4 = {n: (1 - rc[str(n)]['tail_xP']['10000.0'] / rc[str(n)]['C_n']) / (n / (1e4 * n)) for n in ns}
ok3 = all(abs(v) < 2e-6 for v in k8.values()) and all(0.1 <= k4[n] <= 0.5 for n in ns if n >= 32)
S['R3'] = dict(hit=bool(ok3), rel_dev_at_1e8n={str(n): k8[n] for n in ns}, deficit_coeff_at_1e4n={str(n): k4[n] for n in ns})

# R4 C_n series
c2 = {n: n ** 2 * (rc[str(n)]['C_n'] - rc[str(n)]['C_series']) for n in ns}
m3 = {n: rc[str(n)]['m_n_times_n3_resid'] for n in ns}
ok4 = abs(c2[512] / c2[1024] - 1) < 0.05 and abs(m3[1024] + 1) < 0.01
S['R4'] = dict(hit=bool(ok4), n2_resid={str(n): c2[n] for n in ns}, m_n3_resid={str(n): m3[n] for n in ns})

# R5 left tail
J2, K1 = 0.21424, 0.31071
lr = {n: -math.log(rc[str(n)]['left_tail']['P(r<=2)']) / n for n in ns}
lq = {n: -math.log(rc[str(n)]['left_tail']['P(Q<=1)']) / n for n in ns}
ok5 = abs(lr[1024] / J2 - 1) < 0.1 and abs(lq[1024] / K1 - 1) < 0.1 and lq[1024] < 0.36
S['R5'] = dict(hit=bool(ok5), rate_r2={str(n): lr[n] for n in ns}, rate_Q1={str(n): lq[n] for n in ns}, J2=J2, K1=K1)

# R6
S['R6'] = dict(hit=bool(abs(S['R2']['rel_dev_n2']) > 0.01), rel_dev_n2=S['R2']['rel_dev_n2'])

# R7 moments
e1 = {n: n * (rc[str(n)]['E_sqrt_Q_over_n'] - math.sqrt(2 * math.pi / 5)) for n in ns}
e2 = {n: n * (rc[str(n)]['E_log_Q_over_n'] - rc[str(n)]['E_log_Q_over_n_limit']) for n in ns}
ok7 = abs(e1[512] / e1[1024] - 1) < 0.1 and abs(e2[512] / e2[1024] - 1) < 0.1
S['R7'] = dict(hit=bool(ok7), n_err_sqrt={str(n): e1[n] for n in ns}, n_err_log={str(n): e2[n] for n in ns})

# R8a MC
r8a = {}
ok8a = True
for n in (8, 16, 32):
    mc = json.load(open(f'{D}/runs/radius_mc_n{n}.json'))
    rows = {}
    for q, (est, se) in mc['cdf_at_exact_quantiles'].items():
        z = (est - float(q)) / se
        rows['q' + q] = z
        ok8a &= abs(z) < 4
    lt = rc[str(n)]['left_tail']
    for key, sk in (('P(r<=2)', 'P(r<=2)'), ('P(Q<=1)', 'P(Q<=1)'), ('P(Q<=2)', 'P(Q<=2)')):
        est, se = mc[key]
        z = (est - lt[sk]) / se if se > 0 else float('nan')
        rows[key] = dict(mc=est, se=se, exact=lt[sk], z=z)
        ok8a &= abs(z) < 4
    r8a[str(n)] = rows
# R8b, R8c stationary lane
r8b, r8c = {}, {}
ok8b = ok8c = True
for n in (8, 16, 32, 64, 128, 256):
    sl = json.load(open(f'{SL}/runs/radius_n{n}.json'))
    ex = [rc[str(n)]['quantiles'][k] for k in ('0.1', '0.5', '0.9', '0.99')]
    cb = [c / e for c, e in zip(sl['chain_q'], ex)]
    nb = [c / e for c, e in zip(sl['net_q'], ex)]
    r8b[str(n)] = cb
    r8c[str(n)] = nb
    ok8b &= all(abs(v - 1) <= 0.01 for v in cb[:3]) and abs(cb[3] - 1) <= 0.05
    if n <= 64:
        ok8c &= abs(nb[0] - 1) <= 0.03 and abs(nb[1] - 1) <= 0.03 and abs(nb[2] - 1) <= 0.10
S['R8'] = dict(hit_a=bool(ok8a), hit_b=bool(ok8b), hit_c=bool(ok8c), mc_z=r8a, chain_over_exact=r8b, nets_over_exact=r8c)
json.dump(S, open(f'{D}/results/partR_scores.json', 'w'), indent=1)

lines = ['| n | median/n (exact) | 2-term formula error | C_n | n^2 x (C_n - 3-term series) | -ln P(r<=2)/n | -ln P(Q<=1)/n | q10, q50, q90, q99 of Q/n |',
         '|---|---|---|---|---|---|---|---|']
for n in ns:
    q = rc[str(n)]['quantiles']
    lines.append('| %d | %.8f | %+.2e | %.6f | %.4f | %.4f | %.4f | %.4f, %.4f, %.3f, %.2f |' % (
        n, med[n], dev[n], rc[str(n)]['C_n'], c2[n], lr[n], lq[n], q['0.1'] / n, q['0.5'] / n, q['0.9'] / n, q['0.99'] / n))
open(f'{D}/results/partR_tables.md', 'w').write('\n'.join(lines) + '\n')
for k, v in S.items():
    print(k, {kk: vv for kk, vv in v.items() if kk.startswith('hit')})
print('\n'.join(lines))
