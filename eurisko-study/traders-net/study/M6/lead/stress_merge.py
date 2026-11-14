#!/usr/bin/env python3
"""M6 stress merge: rebuild lead/M6-stress-verdict.json from the stress workers' verdict files
(s1 MS/IE, s2 noise traders, s3 D families) and check every stress number quoted in M6.md
against its source file. Run from study/M6/: python3 lead/stress_merge.py  (exit 1 on any mismatch)."""
import csv, json, sys, datetime
J = lambda p: json.load(open(p))
s1 = J('s1/results/verdict-s1.json'); s1sum = {(r['regime'], r['arm']): r for r in csv.DictReader(open('s1/results/SUMMARY-s1.csv'))}
s1sec = J('s1/results/SECONDARY-s1.json')
s2 = J('s2/verdict.json'); s2sum = {(r['arm'], r['noise']): r for r in csv.DictReader(open('s2/summary.csv'))}
s3 = J('s3/VERDICT.json'); s3p = J('s3/PAIRED-vs-kill.json')
s3sum = {(r['arm'], r['metric']): r for r in csv.DictReader(open('s3/SUMMARY.csv'))}
base = J('lead/M6-verdict.json')

ARMS = ['b', 'cg', 'cq', 'fb01', 'fb05']
table = {}
for a in ARMS:
    A = a.upper(); row = {}
    for reg in ('MS', 'IE'):
        d = s1['regimes'][reg][A]
        row[reg] = {'dropped': d['dropped'], 'beats_kill_3of3': d['beats_kill_controls_seeds'] == 3,
                    'kill_set': 'ES,WD,BOOST', 'src': 's1/results/verdict-s1.json'}
    d = s2['iid'][a]
    row['noise_iid'] = {'dropped': None, 'beats_kill_3of3': bool(d['H2_stress_win']), 'kill_set': 'WD,BOOST',
                        'src': 's2/verdict.json iid'}
    d = s2['fixed'][a]
    row['noise_fixed'] = {'dropped': None, 'beats_kill_3of3': bool(d['H2_stress_win']), 'kill_set': 'WD,BOOST',
                          'src': 's2/verdict.json fixed'}
    d = s3['arms'][a]
    bk = a in s3p and s3p[a]['P1_Dlearnable_bce']['better_3of3']
    row['D'] = {'dropped': d['fails_D_stress'], 'beats_kill_3of3': bool(bk), 'kill_set': 'WD,BOOST',
                'src': 's3/VERDICT.json, s3/PAIRED-vs-kill.json'}
    row['dropped_any_regime'] = any(v['dropped'] for k, v in row.items() if isinstance(v, dict) and v['dropped'])
    row['beats_kill_in'] = [k for k, v in row.items() if isinstance(v, dict) and v.get('beats_kill_3of3')]
    table[a] = row

# Numbers quoted in the stress sections of M6.md -> (quoted, computed from file, tolerance)
f = lambda x: float(x)
checks = [
 ('s3 cg P1 D-learnable mean', 0.3800, f(s3sum[('cg', 'P1_Dlearnable_bce')]['mean']), 5e-4),
 ('s3 boost P1 mean', 0.3902, f(s3sum[('boost', 'P1_Dlearnable_bce')]['mean']), 5e-4),
 ('s3 wd P1 mean', 0.4279, f(s3sum[('wd', 'P1_Dlearnable_bce')]['mean']), 5e-4),
 ('s3 a P1 mean', 0.4357, f(s3sum[('a', 'P1_Dlearnable_bce')]['mean']), 5e-4),
 ('s3 cg - best kill', -0.0102, s3p['cg']['P1_Dlearnable_bce']['mean'], 5e-5),
 ('s3 cg twin D64', 0.029, f(s3sum[('cg', 'twin_viol_D64')]['mean']), 5e-4),
 ('s3 a twin D64', 0.062, f(s3sum[('a', 'twin_viol_D64')]['mean']), 5e-4),
 ('s3 a prove_bce_pre', 1.66, f(s3sum[('a', 'prove_bce_pre')]['mean']), 5e-3),
 ('s3 cg prove_bce_pre', 0.715, f(s3sum[('cg', 'prove_bce_pre')]['mean']), 5e-4),
 ('s3 min P2 pre-prove viol', 0.506, min(f(s3sum[(a, 'P2_pre_prove_viol')]['mean']) for a in ['a','b','cg','cq','fb01','fb05','wd','boost']), 5e-4),
 ('s3 max P2 pre-prove viol', 0.559, max(f(s3sum[(a, 'P2_pre_prove_viol')]['mean']) for a in ['a','b','cg','cq','fb01','fb05','wd','boost']), 5e-4),
 ('s1 MS A humility', 2.61, f(s1sum[('MS', 'A')]['humility_mean']), 5e-3),
 ('s1 MS FB01 humility', 0.224, f(s1sum[('MS', 'FB01')]['humility_mean']), 5e-4),
 ('s1 MS ES humility', 0.011, f(s1sum[('MS', 'ES')]['humility_mean']), 5e-4),
 ('s1 MS ES learnable', 0.297, f(s1sum[('MS', 'ES')]['learn_bce_mean']), 5e-4),
 ('s1 MS own-ES learnable min', 0.296, min(s1sec[f'MS:{a}']['own_es_learn_ms'][0] for a in ['A','B','CG','CQ','FB01','FB05','WD','BOOST']), 5e-4),
 ('s1 MS own-ES learnable max', 0.302, max(s1sec[f'MS:{a}']['own_es_learn_ms'][0] for a in ['A','B','CG','CQ','FB01','FB05','WD','BOOST']), 5e-4),
 ('s1 IE CQ regret', 0.028, f(s1sum[('IE', 'CQ')]['ie_regret_mean']), 5e-4),
 ('s1 IE ES regret', 0.031, f(s1sum[('IE', 'ES')]['ie_regret_mean']), 5e-4),
 ('s1 IE A regret', 0.043, f(s1sum[('IE', 'A')]['ie_regret_mean']), 5e-4),
 ('s1 IE CG regret', 0.066, f(s1sum[('IE', 'CG')]['ie_regret_mean']), 5e-4),
 ('s1 IE market regret', 0.195, s1sec['IE:L0']['ie_regret_ms'][0], 5e-4),
 ('s2 cg iid delta', 0.0011, f(s2sum[('cg', 'iid')]['delta_learn_mean']), 5e-5),
 ('s2 cg iid k_noise', 5e-5, f(s2sum[('cg', 'iid')]['k_noise_mean']), 1e-5),
 ('s2 cq iid k_noise', 0.050, f(s2sum[('cq', 'iid')]['k_noise_mean']), 5e-4),
 ('s2 boost iid delta', 0.025, f(s2sum[('boost', 'iid')]['delta_learn_mean']), 1e-3),
 ('s2 fb01 iid delta', 0.019, f(s2sum[('fb01', 'iid')]['delta_learn_mean']), 5e-4),
 ('s2 fb05 iid delta', 0.060, f(s2sum[('fb05', 'iid')]['delta_learn_mean']), 5e-4),
 ('s2 b iid delta (low end 0.25)', 0.25, f(s2sum[('b', 'iid')]['delta_learn_mean']), 5e-3),
 ('s2 a iid delta (high end 0.27)', 0.27, f(s2sum[('a', 'iid')]['delta_learn_mean']), 5e-3),
]
bad = [c for c in checks if abs(c[1] - c[2]) > c[3]]
out = {
 'checked_at': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ'),
 'base_arms_survive': base.get('base_arms_survive'),
 'stress_complete': bool(s1['complete'] and s2['checked_files'] == s2['expected_files'] and s3['complete']),
 'table': table,
 'dropped_somewhere': [a for a in ARMS if table[a]['dropped_any_regime']],
 'survive_all_stress': [a for a in ARMS if not table[a]['dropped_any_regime']],
 'beats_kill_controls_somewhere': {a: table[a]['beats_kill_in'] for a in ARMS if table[a]['beats_kill_in']},
 'beats_kill_controls_including_ES': [a for a in ARMS if table[a]['MS']['beats_kill_3of3'] or table[a]['IE']['beats_kill_3of3']],
 'number_checks': [{'claim': c[0], 'quoted': c[1], 'file_value': round(c[2], 6), 'ok': abs(c[1]-c[2]) <= c[3]} for c in checks],
 'n_checks': len(checks), 'n_bad': len(bad),
}
json.dump(out, open('lead/M6-stress-verdict.json', 'w'), indent=1, sort_keys=True)
print(json.dumps({k: out[k] for k in ('stress_complete', 'dropped_somewhere', 'survive_all_stress',
      'beats_kill_controls_somewhere', 'beats_kill_controls_including_ES', 'n_checks', 'n_bad')}))
for c in bad: print('MISMATCH', c)
sys.exit(1 if bad else 0)
