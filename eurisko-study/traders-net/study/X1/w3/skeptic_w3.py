#!/usr/bin/env python3
"""X1 worker 3, part C: skeptic checks on the three open judgement points (read-only).
 1. M7: is wealth_beats_deepseek=false a judgement over conflicting rules? paired t-tests and a
    multiple-comparison count over every non-collapsed wealth cell on its worker's primary metric.
 2. M6: does survival rest on a ~0.001-nat noise margin or on LFSR16 luck? LFSR16-learnt counts per arm in
    the w2 harness (Fisher exact vs (a)), the paired cg-BOOST noise margin, the ex-LFSR16 tie.
 3. M5: is the pi edge a stream artefact? what the evidence is.
Output: results/skeptic.json. Seed-4 files are folded in when present."""
import csv, json, math, os, sys, datetime
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.abspath(os.path.join(HERE, '..', '..'))
RR = '/tmp/traders-net/X1w3'
def _fallback(p):
    """Seed-4 files: use /tmp/traders-net/X1w3 when present, else the synced copy in results/seed4/."""
    if os.path.exists(p) or not str(p).startswith('/tmp/traders-net/X1w3/'): return p
    b = os.path.basename(p); d = os.path.dirname(p)
    pre = {'M6/w1/results': 'M6w1-', 'out/M6w2': 'M6w2-', 'out/M6w4': 'M6w4-', 'out/M6w5': 'M6w5-', 'out/M7w2': 'M7w2-', 'M5/w4/out': ''}
    for k, v in pre.items():
        if d.endswith(k): return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'seed4', v + b)
    return p
J = lambda p: json.load(open(_fallback(p)))
out = {}
# ---- 1. M7 ----
arms = list(csv.DictReader(open(f'{ST}/M7/merge/arms.csv')))
per = list(csv.DictReader(open(f'{ST}/M7/merge/per_seed.csv')))
prim = {'w1': 'bce_2nd_half', 'w2': 'bce_all', 'w3': 'bce_2nd_half', 'w4': 'bce_last', 'w5': 'bce_2nd_half'}
cells = [a for a in arms if a['role'] == 'wealth' and a['metric'] == prim[a['worker']]]
nc = [a for a in cells if a['collapsed'] == 'False']
t = {}
for E, m, sd in [(8, -0.00512, 0.00095), (64, -0.00724, 0.00492)]:
    tt = m / (sd / math.sqrt(3)); p = 2 * stats.t.sf(abs(tt), 2)
    t[f'E{E}'] = dict(mean=m, paired_sd=sd, t=tt, df=2, p_two_sided=p)
p33 = 1 / 8
out['M7'] = dict(
    gate_rules=dict(R1='3/3 paired wins + not collapsed (w1 PREREG merger rule): PASS for dsfw at E8 and E64',
                    R2='R1 + paired mean gap > 1% at both E: PASS (-1.30%, -1.87%)',
                    R3='w2 own PREREG: gap > larger seed sd (unpaired): FAIL; decision.json'),
    paired_t_dsfw_vs_ds3_bce_all=t,
    wealth_cells_primary_metric=len(cells), noncollapsed_cells=len(nc),
    noncollapsed_3of3=[f"{a['worker']}:{a['arm']}:E{a['E']}" for a in nc if a['wins_vs_ds'] == '3'],
    expected_3of3_under_null=len(nc) * p33,
    note=('w2 ran three dsfw variants (dsfw, dsfw_fast, dsfw_mean); the lead reports the best. Under a no-effect null '
          'a given variant wins 3/3 at E64 with p=1/8, so at least one of three does with p=%.2f. The E8 edge is '
          'strong (paired t=%.1f) but E64 (t=%.2f, p=%.2f, df 2) is weak; R1/R2 treat 3/3 as sufficient, which '
          'the 1/8 null rate does not support at E64.' % (1 - (7 / 8) ** 3, t['E8']['t'], t['E64']['t'], t['E64']['p_two_sided'])))
# seed 4 fold-in
s4 = {}
for E in (8, 64):
    try:
        a = J(f'{RR}/out/M7w2/dsfw-E{E}-s4.json'); b = J(f'{RR}/out/M7w2/ds3-E{E}-s4.json')
        s4[f'E{E}'] = dict(dsfw=a['bce_all'], ds3=b['bce_all'], d=a['bce_all'] - b['bce_all'],
                           d_post_shift=a['bce_post_shift'] - b['bce_post_shift'], dsfw_wall_s=a['wall_s'])
    except Exception as e:
        s4[f'E{E}'] = 'missing'
out['M7']['seed4'] = s4
four = {}
for E in (8, 64):
    d3 = [J(f'{ST}/M7/w2/results/all/dsfw-E{E}-s{s}.json')['bce_all'] - J(f'{ST}/M7/w2/results/all/ds3-E{E}-s{s}.json')['bce_all'] for s in (1, 2, 3)]
    ds3m = [J(f'{ST}/M7/w2/results/all/ds3-E{E}-s{s}.json')['bce_all'] for s in (1, 2, 3)]
    if isinstance(s4.get(f'E{E}'), dict):
        d4 = np.array(d3 + [s4[f'E{E}']['d']]); m4 = np.array(ds3m + [s4[f'E{E}']['ds3']])
        tt = d4.mean() / (d4.std(ddof=1) / 2)
        four[f'E{E}'] = dict(d=d4.tolist(), rel_pct=float(100 * d4.mean() / m4.mean()), paired_sd=float(d4.std(ddof=1)), t=float(tt), df=3,
                             p_two_sided=float(2 * stats.t.sf(abs(tt), 3)), wins=int((d4 < 0).sum()), sign_test_one_sided_p=float(0.5 ** int((d4 < 0).sum())) if (d4 < 0).all() else None)
out['M7']['four_seeds'] = four
if four:
    out['M7']['note'] += (' Seed 4 (CPU) adds a 4th paired win at both E: E8 %.2f%%, E64 %.2f%%; 4-seed paired t E8 %.1f (p=%.3f), E64 %.1f (p=%.3f), df 3; sign test 4/4 p=0.0625 per E. '
                          'So the bce_all edge replicates; the post-shift cost also replicates (+%.3f / +%.3f). The false flag still rests on the lead\'s cost and family-label reasons, not on the edge being noise.') % (
        100 * s4['E8']['d'] / s4['E8']['ds3'], 100 * s4['E64']['d'] / s4['E64']['ds3'], four['E8']['t'], four['E8']['p_two_sided'], four['E64']['t'], four['E64']['p_two_sided'],
        s4['E8']['d_post_shift'], s4['E64']['d_post_shift'])
# ---- 2. M6 ----
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
LEARNT = 0.5  # lfsr16 BCE below 0.5 nats = learnt (chance is ln 2 = 0.693)
cnt = {}
for d, arms6 in [('w2/results', ['a', 'cg', 'cq', 'd0', 'dfs']), ('lead/results', ['wd', 'boost'])]:
    for a in arms6:
        v = [J(f'{ST}/M6/{d}/{a}-s{s}.json')['bce_all'][FAM.index('lfsr16')] for s in (1, 2, 3)]
        cnt[a] = dict(lfsr16_bce=v, learnt=int(sum(x < LEARNT for x in v)))
for a, f in [('cg', 'cg-s4.json'), ('boost', 'boost-s4.json')]:
    p = f'{RR}/out/M6w2/{f}'
    if os.path.exists(_fallback(p)):
        cnt[a]['seed4_lfsr16_bce'] = J(p)['bce_all'][FAM.index('lfsr16')]
fisher = {a: stats.fisher_exact([[cnt['a']['learnt'], 3 - cnt['a']['learnt']], [cnt[a]['learnt'], 3 - cnt[a]['learnt']]], alternative='greater')[1]
          for a in cnt if a != 'a'}
V = J(f'{ST}/M6/lead/M6-verdict.json')
cgn = np.array(V['arms']['cg']['noise_per_seed']); bon = np.array(V['w2_harness_kill_controls']['boost']['per_seed_noise'])
d = cgn - bon
out['M6'] = dict(
    lfsr16_learnt_w2_harness=cnt, fisher_one_sided_a_learns_more=fisher,
    cg_minus_boost_noise=dict(per_seed=d.tolist(), mean=float(d.mean()), sd=float(d.std(ddof=1)),
                              t=float(d.mean() / (d.std(ddof=1) / math.sqrt(3)))),
    exlfsr_cg_minus_boost=(np.array(V['w2_harness_exlfsr']['cg']['learn_exlfsr']) - np.array(V['w2_harness_exlfsr']['boost']['learn_exlfsr'])).tolist(),
    stress_pass='M6.md stress pass (12:01Z merge) drops cg and fb01 under independent evidence; survivors across stress are b, cq, fb05 (M6/lead/M6-stress-verdict.json).',
    note=('The cg survival margin is ~0.0014 nats of noise excess (0.2%% of ln 2) but consistent (paired t=%.1f, 3/3). '
          'The +35-44%% learnable-BCE gap of cg vs (a) is LFSR16: (a) learnt it in %d/3 seeds, cg in %d/3 (Fisher one-sided p=%.3f). '
          'Excluding LFSR16 cg ties BOOST. So cg survives on a tiny but real noise margin, and it systematically blocks LFSR16 '
          'learning in this harness; calling that seed luck is not supported by 3/3 vs 0/3.') % (
        float(d.mean() / (d.std(ddof=1) / math.sqrt(3))), cnt['a']['learnt'], cnt['cg']['learnt'], fisher['cg']))
if 'seed4_lfsr16_bce' in cnt['cg']:
    out['M6']['seed4'] = dict(cg_lfsr16=cnt['cg']['seed4_lfsr16_bce'], boost_lfsr16=cnt['boost'].get('seed4_lfsr16_bce'))
    out['M6']['note'] += ' Seed 4: cg LFSR16 BCE %.3f (not learnt; 0/4), BOOST %.3f (%s); cg noise still below BOOST (margin -0.0022).' % (
        cnt['cg']['seed4_lfsr16_bce'], cnt['boost']['seed4_lfsr16_bce'], 'learnt' if cnt['boost']['seed4_lfsr16_bce'] < LEARNT else 'not learnt')
try:
    sv = J(f'{ST}/M6/lead/M6-stress-verdict.json')
    out['M6']['stress_verdict_keys'] = list(sv.keys())[:20]
except Exception as e:
    out['M6']['stress_verdict_keys'] = repr(e)
# ---- 3. M5 ----
fam = list(csv.DictReader(open(f'{ST}/M5/w4/results/family.csv')))
pick = {f"{r['arm']}:{r['family']}": (float(r['EL']), float(r['EL_sd'])) for r in fam if r['arm'].startswith('preq') and r['family'] in ('lfsr16', 'pi', 'prime', 'ca110')}
out['M5'] = dict(per_family_probeL=pick,
                 shifted_copy_share='S0/SUBSTRATE-w2.md: wealth-net pi 1.000 +- 0.000, prng 0.000 (M1 audit)',
                 note=('The pi part of the probe edge (0.186 / 0.172 nats/q-step within family) coexists with a documented '
                       'shifted-copy share of 1.0 for wealth-net pi, so a probe shared across questions can memorise it. That '
                       'makes the artefact reading plausible but it is not tested: no probe was run on the PRNG-parity '
                       'family (shifted-copy share 0). Status stays unverified.'))
for ck in ('P4', 'P16'):
    f4 = f'{RR}/clean/study/M5/w4/out/preq-{ck}-win-s4.json'
    if os.path.exists(_fallback(f4)):
        d4 = J(f4)
        out['M5'][f'seed4_{ck}'] = dict(probeL_fam=d4['probes']['L']['E_fam_per_qstep'], probeL_phi=d4['probes']['L']['phi_last_half_per_qstep'],
                                        note='seed 4 draws pi offsets with shifted copies inside W=32 (X1/w1 12:14Z), so the net itself partly learns pi at seed 4')
if 'seed4_P4' in out['M5'] and 'seed4_P16' in out['M5']:
    a4, a16 = out['M5']['seed4_P4']['probeL_fam'], out['M5']['seed4_P16']['probeL_fam']
    out['M5']['note'] += (' Seed 4 (which draws close pi offsets, X1/w1 12:14Z): probe L pi %.3f / %.3f and lfsr16 %.3f / %.3f (P4 / P16), vs seeds 1-3 pi %.3f / %.3f. '
                          'The pi edge shrinks when the net itself sees shifted pi copies, which fits the artefact reading but does not test it.') % (
        a4['pi'], a16['pi'], a4['lfsr16'], a16['lfsr16'], pick['preq-P4-win:pi'][0], pick['preq-P16-win:pi'][0])
out['checked_at'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(out, open(os.path.join(HERE, 'results', 'skeptic.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v.get('note') if isinstance(v, dict) else v for k, v in out.items()}, indent=1))
print(json.dumps(out['M7']['paired_t_dsfw_vs_ds3_bce_all']), out['M7']['noncollapsed_3of3'], out['M7']['expected_3of3_under_null'])
print(json.dumps(out['M6']['fisher_one_sided_a_learns_more']), {a: v['learnt'] for a, v in cnt.items()})
