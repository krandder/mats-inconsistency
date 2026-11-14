"""E74 descriptive extras (post hoc, ninth agent; not registered): per-seed head measures at S = 6 and S = 3, median-based and per-seed
KL ratios, the registered P297 statistic without the two barely acquired S = 6 ctrl nets (s8, s9), and the type changes between S = 3 and
S = 6 for the same initial values. Reads results/e74/S6 and results/e74/S3 only.
usage: python e74_desc.py -> results/e74_desc.json (and prints a summary)
"""
import json, glob
import numpy as np


def load(pat):
    return {int(json.load(open(p))['seed']): json.load(open(p)) for p in glob.glob(pat)}


C6, O6 = load('results/e74/S6/ctrl_s*.json'), load('results/e74/S6/own_s*.json')
C3, O3 = load('results/e74/S3/e37_ctrl_s*.json'), load('results/e74/S3/e60_own_s*.json')
acq = lambda d: {s for s, o in d.items() if o['gap'] < 0.10}
s4 = sorted(acq(C6) & acq(O6) & acq(C3) & acq(O3))
out = dict(seeds_all4=s4, per_seed={})
for s in range(12):
    row = {}
    for nm, d in (('ctrl6', C6), ('own6', O6), ('ctrl3', C3), ('own3', O3)):
        o = d.get(s)
        if o is None:
            continue
        r = dict(gap=o['gap'], type=o['type'], rom_large=o['rom_large'], rom_middle=o['rom_middle'])
        if 'n_pool' in o and nm.startswith('ctrl'):
            ph = o['pool_heads']
            r.update(n_pool=o['n_pool'], n_cons=o['n_consensus'], opp=o['opposite_pair'], L_own_pool=[o['L_own_h'][h] for h in ph],
                     K_pool=[o['K_h'][h] for h in ph], dl3_pool=[o['dlogit_r3_h'][h] for h in ph], leak_pool=o['leak_pool'],
                     P_h=o['P_h'], C_h=o['C_h'], leak_h=o['leak_h'])
        row[nm] = r
    out['per_seed'][s] = row
g = lambda d, j: np.array([d[s]['gap'] for s in j])
lr = np.log(g(C6, s4) / g(O6, s4)) - np.log(g(C3, s4) / g(O3, s4))
out['paired_log_ratio_per_seed'] = dict(zip(map(str, s4), lr.tolist()))
out['paired_log_ratio_median'] = float(np.median(lr)); out['paired_log_ratio_positive'] = int((lr > 0).sum())
med = lambda d, j: float(np.median(g(d, j)))
out['median_ratio'] = dict(R6=med(C6, s4) / med(O6, s4), R3=med(C3, s4) / med(O3, s4))
out['median_ratio']['ratio'] = out['median_ratio']['R6'] / out['median_ratio']['R3']
rng = np.random.default_rng(741)
s2 = [s for s in s4 if s not in (8, 9)]
f = lambda j: (g(C6, j).mean() / g(O6, j).mean()) / (g(C3, j).mean() / g(O3, j).mean())
bs = [f(list(rng.choice(s2, len(s2)))) for _ in range(2000)]
out['P297_stat_without_s8_s9'] = [float(f(s2)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(s2)]
out['type_change_3_to_6'] = {str(s): '%s->%s' % (C3[s]['type'], C6[s]['type']) for s in s4}
ty = lambda d, s: d[s]['type'] if d[s]['gap'] < 0.10 else 'stall'
a6 = [s for s in acq(C6)]
out['ctrl6_rule'] = {str(s): dict(type=C6[s]['type'], n_pool=C6[s]['n_pool'], n_cons=C6[s]['n_consensus'], opp=C6[s]['opposite_pair']) for s in sorted(a6)}
# within-cell associations at S = 6 (B nets with one pooling head)
b6 = [s for s in sorted(a6) if C6[s]['type'] == 'B' and C6[s]['n_pool'] == 1]
b3 = [s for s in sorted(acq(C3)) if C3[s]['type'] == 'B' and C3[s]['n_pool'] == 1]
for nm, d, ss in (('B6', C6, b6), ('B3', C3, b3)):
    out[nm] = dict(seeds=ss, rom_large=[d[s]['rom_large'] for s in ss], L_own=[d[s]['L_own_h'][d[s]['pool_heads'][0]] for s in ss],
                   dl3=[d[s]['dlogit_r3_h'][d[s]['pool_heads'][0]] for s in ss], leak=[d[s]['leak_pool'] for s in ss],
                   K=[d[s]['K_h'][d[s]['pool_heads'][0]] for s in ss])
for nm, d in (('A6', C6), ('A3', C3)):
    ss = [s for s in sorted(acq(d)) if d[s]['type'] == 'A']
    out[nm] = dict(seeds=ss, rom_large=[d[s]['rom_large'] for s in ss], L_own=[[d[s]['L_own_h'][h] for h in d[s]['pool_heads']] for s in ss],
                   dl3=[[d[s]['dlogit_r3_h'][h] for h in d[s]['pool_heads']] for s in ss], K=[[d[s]['K_h'][h] for h in d[s]['pool_heads']] for s in ss],
                   leak=[d[s]['leak_pool'] for s in ss])
# first validation check (every 1,000 steps) with val NLL - val Bayes NLL < 0.10, from the runs' stored curves (meta.json 'curve'); added
# after the review (the draft's acquisition steps came from an inline read of the same curves)
def first_cross(run, thr=0.10):
    m = json.load(open(run + '/meta.json'))
    for row in m['curve']:
        if row[1] - m['val_bayes'] < thr:
            return int(row[0])
    return None


out['first_val_gap_below_0.10'] = {
    'ctrl6': {s: first_cross('runs/e74/ctrl_s%d' % s) for s in range(12)}, 'own6': {s: first_cross('runs/e74/own_s%d' % s) for s in range(12)},
    'ctrl3': {s: first_cross('runs/e37/e37_ctrl_s%d' % s) for s in range(12)}, 'own3': {s: first_cross('runs/e60/e60_own_s%d' % s) for s in range(12)}}
out['S3_s8_no_pooling_head'] = dict(P_h=C3[8]['P_h'], C_h=C3[8]['C_h'], leak_h=C3[8]['leak_h'])
json.dump(out, open('results/e74_desc.json', 'w'), indent=1, default=float)
print('first_val_gap_below_0.10', json.dumps(out['first_val_gap_below_0.10']))
print('S3_s8', json.dumps(out['S3_s8_no_pooling_head']))
for k in ('seeds_all4', 'paired_log_ratio_per_seed', 'paired_log_ratio_median', 'paired_log_ratio_positive', 'median_ratio',
          'P297_stat_without_s8_s9', 'type_change_3_to_6', 'ctrl6_rule', 'B6', 'B3', 'A6', 'A3'):
    print(k, json.dumps(out[k], default=lambda x: round(float(x), 4)))
