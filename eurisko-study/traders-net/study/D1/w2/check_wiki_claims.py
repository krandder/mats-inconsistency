#!/usr/bin/env python3
"""Re-run the read-only checks behind every status claim in traders-net/wiki-traders-net.md.

Writes study/D1/w2/wiki-check.json (checked_at + per-claim evidence) and prints one line per claim.
Read-only on everything outside study/D1/w2/: reads files, lists GCE instances/disks, fetches two URLs,
and runs a copy of V1's Grafana check whose output goes to study/D1/w2/grafana-check/.
Usage: python3 study/D1/w2/check_wiki_claims.py [--skip-grafana]
Exit 0 when every claim holds; 1 otherwise.
"""
import json, os, re, subprocess, sys, urllib.request
from datetime import datetime, timezone

ROOT = '/home/agent/workspaces/eurisko-study/traders-net'
S = ROOT + '/study'
HERE = S + '/D1/w2'
res = []


def claim(cid, text, ok, evidence, detail):
    res.append(dict(id=cid, claim=text, holds=bool(ok), evidence=evidence, detail=detail))
    print('%-5s %-22s %s' % ('OK' if ok else 'FAIL', cid, detail))


def j(p):
    with open(os.path.join(S, p)) as f:
        return json.load(f)


# 1. claims register counts and its own check
cc = j('X1/lead/results/check_claims.json')
head = open(S + '/X1/CLAIMS.md').read()
m = re.search(r'Rows: (\d+); \*\*verified (\d+), refuted (\d+), unverified (\d+)\*\*', head)
claim('register', 'CLAIMS.md: 209 rows, 153 verified / 19 refuted / 37 unverified; check_claims ok',
      m and m.groups() == ('209', '153', '19', '37') and cc.get('ok') is True and not cc.get('missing_paths'),
      'study/X1/CLAIMS.md; study/X1/lead/results/check_claims.json',
      'rows=%s check_claims ok=%s at %s' % (m.groups() if m else None, cc.get('ok'), cc.get('checked_at')))

# 2. gate flags
lock = j('M1/CURRENCY.lock')
claim('gate-A', 'currency locked: committed-trader settlement only',
      lock.get('headline') == 'no bookkeeping currency for generic neurons; committed-trader settlement only',
      'study/M1/CURRENCY.lock', 'locked_at %s' % lock.get('locked_at'))
m6 = j('M6/lead/M6-verdict.json')
claim('gate-B4', 'M6 base arms survive', m6.get('base_arms_survive') is True, 'study/M6/lead/M6-verdict.json',
      'surviving=%s' % m6.get('surviving'))
m7 = j('M7/merge/gate.json')
claim('gate-B3', 'no wealth arm passes all three M7 rules (wealth_beats_deepseek=false)', m7.get('any_R3') is False,
      'study/M7/merge/gate.json', 'any_R3=%s passing_R2=%s' % (m7.get('any_R3'), m7.get('passing_R2_primary')))
gates = {g['gate'].split(':')[0]: g['flag'] for g in j('X1/lead/gates.json')}
want = {'A': 'currency_locked = true', 'B1': 'beats_baseline = false', 'B2': 'coalition_win = false',
        'B3': 'wealth_beats_deepseek = false', 'B4': 'base_arms_survive = true'}
claim('gates-X1', 'X1 gate flags as listed', gates == want, 'study/X1/lead/gates.json', json.dumps(gates))

# 3. GPU spend and zero VMs
cost = open(S + '/G1/COST.md').read()
m = re.search(r'Total \(ceiling\): \$([\d.]+) of \$45\.00 cap', cost)
claim('gpu-spend', 'GPU spend $6.07 of the $45 cap', m and m.group(1) == '6.07', 'study/G1/COST.md',
      'total=%s' % (m.group(1) if m else None))
gc = ['gcloud', '--project=GCP_PROJECT', '--quiet', 'compute']
try:
    vms = subprocess.run(gc + ['instances', 'list', '--filter=labels.traders-net:* OR name ~ ^tn-gpu-',
                               '--format=value(name)'], capture_output=True, text=True, timeout=120)
    dks = subprocess.run(gc + ['disks', 'list', '--filter=name ~ ^tn-gpu- OR labels.traders-net:*',
                               '--format=value(name)'], capture_output=True, text=True, timeout=120)
    running = os.listdir(S + '/G1/queue/running')
    okv = vms.returncode == 0 and dks.returncode == 0
    claim('zero-vms', 'no traders-net GPU VM, disk or running job left',
          okv and not vms.stdout.strip() and not dks.stdout.strip() and not running,
          'gcloud compute instances/disks list (project GCP_PROJECT); study/G1/queue/running/',
          'rc=%d/%d instances=%r disks=%r running=%r' % (vms.returncode, dks.returncode, vms.stdout.split(),
                                                        dks.stdout.split(), running))
except Exception as e:  # cannot check -> the claim shows as unverified, not as true
    claim('zero-vms', 'no traders-net GPU VM left', False, 'gcloud', 'UNVERIFIED: %r' % e)

# 4. Grafana dashboard loads (copy of V1/check.py; output under D1/w2/grafana-check/)
if '--skip-grafana' not in sys.argv:
    p = subprocess.run([sys.executable, HERE + '/check_v1_copy.py'], capture_output=True, text=True, timeout=900)
    g = json.load(open(HERE + '/grafana-check/latest.json'))
    gs = [s for s in g['stages'] if s['stage'] == 'grafana']
    panels = {str(x.get('id')): x.get('verdict') for x in (gs[0].get('panels', []) if gs else [])}
    claim('grafana', 'dashboard traders-net loads; no panel FAILs or PENDING', p.returncode == 0 and g.get('pass')
          and not [v for v in panels.values() if v in ('FAIL', 'PENDING')],
          'study/D1/w2/grafana-check/latest.json', 'checked_at %s pass=%s panels=%s' % (
              g.get('checked_at'), g.get('pass'), panels))

# 5. HTML page served
for cid, url, pat in [('page', 'http://[ip-removed]:8765/traders-net/index.html', r'<title>[^<]+'),
                      ('wealth-net-page', 'http://[ip-removed]:8765/wealth-net/', r'<title>[^<]+')]:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode('utf-8', 'replace')
            t = re.search(pat, body)
            claim(cid, '%s answers 200 with a titled page' % url, r.status == 200 and t, url,
                  'status %d %s bytes=%d' % (r.status, t.group(0) if t else 'no title', len(body)))
    except Exception as e:
        claim(cid, '%s answers 200' % url, False, url, 'not served: %r' % e)

out = dict(checked_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), claims=res,
           all_hold=all(r['holds'] for r in res))
json.dump(out, open(HERE + '/wiki-check.json', 'w'), indent=2)
print('all_hold=%s -> %s' % (out['all_hold'], HERE + '/wiki-check.json'))
sys.exit(0 if out['all_hold'] else 1)
