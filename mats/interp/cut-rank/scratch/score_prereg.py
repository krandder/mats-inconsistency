"""Score the pre-registered predictions P1-P8 (PREREG.md section 4) from runs/analysis_*.json and the crux JSON.
Only the registered estimators are used (r_in, r_cf, fixed-back r_in, moments LS, crux r_loc, per-net fraction).
Noise-floor rule: eps = 0.01 is excluded when eps^2 Vbar is below 3x the control floor (controls give |tail| ~ 1-2e-4
Vbar), so tests use eps = 0.1 as registered.
usage: python3 scratch/score_prereg.py
"""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')


def A(name):
    p = os.path.join(R, f'analysis_{name}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def r_in(name, eps='0.1'):
    a = A(name)
    return None if a is None else a['ensemble']['r_in'][eps]


def r_cf(name, eps='0.1'):
    a = A(name)
    return None if a is None else a['ensemble']['r_cf'][eps]


out = []
# P1: n = 16, c = 8, ratio r(0.1) at L' = 4 over L' = 32
a, b = r_in('cut_n16_c8_L4'), r_in('cut_n16_c8_L32')
if a and b:
    ratio = a / b
    verdict = 'mine supported (<= 2)' if ratio <= 2 else ('premise supported (>= 4)' if ratio >= 4 else 'inconclusive')
    out.append(f'P1 k* dependence: r_in(0.1) L\'=4: {a}, L\'=32: {b}, ratio {ratio:.2f} -> {verdict}')
# P2: c/n = L'/n = 1/2, n = 32 vs n = 8 (compare at equal F' = 256 too)
a8, a32 = A('cut_n8_c4_L4'), A('cut_n32_c16_L16')
if a8 and a32:
    r8, r32 = a8['ensemble']['r_in']['0.1'], a32['ensemble']['r_in']['0.1']
    r8_256 = a8['saturation_in']['256']['0.1']
    sat32 = r32 >= a32['F'] / 2; sat8 = r8 >= a8['F'] / 2
    ratio = r32 / r8
    if ratio >= 4 or (sat32 and not sat8):
        v = 'mine supported'
    elif ratio <= 2:
        v = 'premise supported (flat in n)'
    else:
        v = 'inconclusive'
    out.append(f'P2 width at fixed k*: r_in(0.1) n=8: {r8} (F\'=256: {r8_256}), n=32: {r32}; ratio {ratio:.2f} (vs F\'=256: {r32 / r8_256:.2f}) -> {v}')
    a64 = A('cut_n64_c32_L32')
    if a64:
        out.append(f'    n=64 (32,32): r_in(0.1) = {a64["ensemble"]["r_in"]["0.1"]} (F=256)')
# P3: synchronisation
sync = ['cut_n8_c64_L8', 'cut_n8_c8_L64', 'cut_n8_c64_L64', 'cut_n16_c128_L16', 'cut_n16_c16_L128']
vals = [(s, r_in(s), r_cf(s)) for s in sync if A(s)]
if vals:
    n_coll = sum(1 for s, ri, rc in vals if rc <= 5)
    n_high = sum(1 for s, ri, rc in vals if ri >= 10)
    v = 'collapse supported' if n_coll >= 4 else ('mine supported (no collapse)' if n_high >= 3 else 'inconclusive')
    out.append(f'P3 synchronisation ({len(vals)}/5 configs): ' + ', '.join(f'{s}: r_in {ri}, r_cf {rc}' for s, ri, rc in vals)
               + f' -> collapse count {n_coll}, high count {n_high}: {v}')
# P4: front depth at n = 8, L' = 8
seq = [r_in(f'cut_n8_c{c}_L8') for c in (2, 8, 64)]
if all(x is not None for x in seq):
    out.append(f'P4 front depth n=8 L\'=8, c = 2/8/64: r_in(0.1) = {seq} -> '
               + ('supported (non-decreasing)' if seq[0] <= seq[1] <= seq[2] else 'refuted'))
# P5: fixed back, R3 readouts, n = 16, c = 8: median over backs at L' = 4 vs 32
fa, fb = A('cut_n16_c8_L4'), A('cut_n16_c8_L32')
if fa and fb:
    ma = np.median([x['R3']['r_in']['0.1'] for x in fa['fixed_back']])
    mb = np.median([x['R3']['r_in']['0.1'] for x in fb['fixed_back']])
    v = 'supported (>= 2x)' if ma >= 2 * mb else ('refuted (<= 1)' if ma / mb <= 1 else 'inconclusive')
    out.append(f'P5 fixed back R3: median r_b(0.1) L\'=4: {ma}, L\'=32: {mb}, ratio {ma / mb:.2f} -> {v}')
# P6: moment order D(0.1) >= 2 at n = 8 window configs (registered LS estimator)
for nm in ('cut_n8_c4_L4', 'cut_n8_c8_L8'):
    a = A(nm)
    if a:
        res = {int(k): v['resid_rel'] for k, v in a['moments'].items()}
        D = next((K for K in sorted(res) if res[K] <= 0.01), None)
        out.append(f'P6 {nm}: LS residual/Vbar by K {[round(res[k], 4) for k in sorted(res)]}; D(0.1) = {D if D is not None else ">" + str(max(res))}'
                   f' -> {"supported" if (D is None or D >= 2) else "refuted"}')
# P7: crux at n = 16, (8, 8), rho = 0.2
p = os.path.join(R, 'crux_n16_c8_L8.json')
if os.path.exists(p):
    cr = json.load(open(p))
    rr = {row['k']: row['r']['0.1'] for row in cr['rows'] if row['kind'] == 'relevant' and row['rho'] == 0.2}
    ks = sorted(rr)
    inc = all(rr[ks[i]] <= rr[ks[i + 1]] for i in range(len(ks) - 1))
    ratio = rr[8] / rr[4]
    v = 'supported' if (inc and ratio >= 2) else ('refuted' if ratio <= 9 / 5 else 'inconclusive')
    out.append(f'P7 crux n=16: r_loc(0.1) by k {rr}; r(8)/r(4) = {ratio:.2f}, increasing {inc} -> {v}')
# P8: per-net fraction at (16, 8, 8)
a = A('cut_n16_c8_L8')
if a:
    fr = a['per_net']['frac_err_gt_0p01V']
    out.append(f'P8 typical net (16,8,8): s = {a["per_net"]["s"]}, fraction of (front, back) with error > 0.01 V = {fr:.2f}'
               f' -> {"supported" if fr >= 0.5 else "refuted"}')
print('\n'.join(out))
