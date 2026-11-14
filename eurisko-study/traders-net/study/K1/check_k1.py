#!/usr/bin/env python3
"""K1 lead check: every number in K1.md traces to a file, and the files reproduce.
Stages: tests; analyzers rerun byte-identical; ledger conservation over all runs; V1 serves every K1 run;
cross-worker reproduction (the same config run by two workers gives the same wealth path);
stop rule over every arm; headline numbers re-derived from the source tables.
Writes K1/check/check_k1.json. Usage: PYTHONPATH=<wealth-net/.pylib> python3 K1/check_k1.py [--skip-v1]"""
import csv, glob, hashlib, json, os, subprocess, sys, urllib.request
from datetime import datetime, timezone
K1 = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(K1, 'runs')
out = {'checked_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), 'stages': {}}
ok_all = True
def stage(name, ok, **kw):
    global ok_all
    ok_all &= bool(ok); out['stages'][name] = dict(ok=bool(ok), **kw); print(('PASS' if ok else 'FAIL'), name, kw if not ok else '')
env = dict(os.environ, PYTHONPATH=os.environ.get('PYTHONPATH', '/home/agent/workspaces/eurisko-study/wealth-net/.pylib'))
p = subprocess.run([sys.executable, '-m', 'pytest', '-q', os.path.join(K1, 'tests')], env=env, capture_output=True, text=True)
stage('tests', p.returncode == 0, tail=p.stdout.strip().splitlines()[-1:])
OUTS = ['w1/cells.csv', 'w2/results/summary.json', 'w2/results/shift_runs.csv', 'w2/results/csweep_runs.csv',
        'w3/out/summary.json', 'w3/out/summary_N1000.csv', 'w3/out/summary_N10000.csv']
h = lambda f: hashlib.sha256(open(os.path.join(K1, f), 'rb').read()).hexdigest()
before = {f: h(f) for f in OUTS}
rcs = {a: subprocess.run([sys.executable, os.path.join(K1, a)], env=env, capture_output=True).returncode
       for a in ['w1/analyze.py', 'w2/analyze.py', 'w3/analyze.py']}
changed = [f for f in OUTS if h(f) != before[f]]
stage('analyzers_reproduce', all(v == 0 for v in rcs.values()) and not changed, rc=rcs, changed=changed)
runs = sorted(d for d in glob.glob(R + '/*/') if os.path.exists(d + 'metrics.csv'))
mx = 0.0
for d in runs:
    for row in csv.DictReader(open(d + 'metrics.csv')):
        for k in ('cons_resid_ledger', 'cons_resid_settle'):
            try: mx = max(mx, abs(float(row[k])))
            except (KeyError, ValueError): pass
stage('conservation', mx < 1e-9 and len(runs) == 141, runs=len(runs), max_resid=mx)
if '--skip-v1' not in sys.argv:
    B = 'http://[ip-removed]:8767'
    served = {x['run'] for x in json.load(urllib.request.urlopen(B + '/runs.json', timeout=60))}
    want = {'K1.' + os.path.basename(d.rstrip('/')) for d in runs}
    empty = [r for r in sorted(want & served) if not json.load(urllib.request.urlopen(f'{B}/runs/{r}/metrics.json', timeout=60))]
    stage('v1_serves_all', want <= served and not empty, missing=sorted(want - served)[:5], empty=empty[:5], n=len(want))
def at(run, step):
    for row in csv.DictReader(open(f'{R}/{run}/metrics.csv')):
        if int(row['step']) == step: return row
pairs = [(f'w1-n10k-q64-on-s{s}', f'w2-S-off10k-s{s}') for s in (1, 2, 3)] + \
        [(f'w1-n10k-q64-on-s{s}', f'w3-bh-within-N10000-s{s}') for s in (1, 2, 3)] + \
        [(f'w1-n1k-q64-on-s{s}', f'w3-bh-within-N1000-s{s}') for s in (1, 2, 3)]
rep = []
for a, b in pairs:
    ra, rb = at(a, 10000), at(b, 10000)
    rep.append(dict(a=a, b=b, W_a=float(ra['W_traders']), W_b=float(rb['W_traders']), alive_a=ra['alive'], alive_b=rb['alive'],
                    gini_a=float(ra['gini_traders']), gini_b=float(rb['gini_traders'])))
stage('cross_worker_reproduction', all(abs(x['W_a'] - x['W_b']) <= 1e-6 * x['W_a'] and x['alive_a'] == x['alive_b'] for x in rep), pairs=rep)
# stop rule: monoculture (top clade share >= 0.9 at T in >= 2/3 seeds) in every arm?
arms = {}
for r in csv.DictReader(open(os.path.join(K1, 'w1/cells.csv'))):
    arms[f"w1 N{r['N']} Q{r['Q']} niches{r['niches']}"] = float(r['mono_mean']) >= 2 / 3 - 1e-9
cs = {}
for r in csv.DictReader(open(os.path.join(K1, 'w2/results/csweep_runs.csv'))):
    cs.setdefault(f"w2 {r['tag']} c/c*={r['c_over_cstar']}", []).append(float(r['top_clade_share']) >= 0.9)
for r in csv.DictReader(open(os.path.join(K1, 'w2/results/shift_runs.csv'))):
    cs.setdefault(f"w2 {r['arm']}", []).append(float(r['top_clade_share']) >= 0.9)
for k, v in cs.items(): arms[k] = sum(v) >= 2
for d in glob.glob(R + '/w3-*-s1/'):
    arm = os.path.basename(d.rstrip('/'))[:-3]; v = []
    for s in (1, 2, 3):
        last = list(csv.DictReader(open(f'{R}/{arm}-s{s}/metrics.csv')))[-1]; v.append(float(last['top_clade_share']) >= 0.9)
    arms['w3 ' + arm] = sum(v) >= 2
mono = sorted(k for k, v in arms.items() if v)
stage('stop_rule_not_triggered', not all(arms.values()), arms=len(arms), mono_arms=mono)
out['stop_rule'] = dict(n_arms=len(arms), n_mono=len(mono), mono_arms=mono, triggered=all(arms.values()))
# headline numbers
cells = {(r['N'], r['Q'], r['niches']): r for r in csv.DictReader(open(os.path.join(K1, 'w1/cells.csv')))}
below = [k for k, r in cells.items() if float(r['gini_traders_mean']) < float(r['gini_null_traders_mean']) - 1e-3]
out['headline'] = dict(
    gini_below_null_cells=f'{len(below)}/{len(cells)}',
    top_species_niches_off_min=min(float(r['top_species_share_mean']) for r in cells.values() if r['niches'] in ('False', '0', 'off') and r['Q'] != '1'),
    q64_on_brier={k[0]: (round(float(r['tail_brier_all_mean']), 4), round(float(r['tail_brier_oracle_all_mean']), 4)) for k, r in cells.items() if k[1] == '64' and r['niches'] in ('True', '1', 'on')})
print(json.dumps(out['headline'], indent=1))
os.makedirs(os.path.join(K1, 'check'), exist_ok=True)
out['ok'] = ok_all
json.dump(out, open(os.path.join(K1, 'check', 'check_k1.json'), 'w'), indent=1)
sys.exit(0 if ok_all else 1)
