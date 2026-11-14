#!/usr/bin/env python3
"""D1 check: every number in IDEAS-NET.md has a source file, and every path it names exists.

Run from anywhere: python3 study/D1/check_ideas.py   (read-only except study/D1/check_ideas.json)
N1 each numeric token (URLs removed) appears verbatim in study/X1/CLAIMS.md, or else in one of the
   secondary sources below (each such hit is listed with its file, so it can be marked unverified);
   tokens that are dates, years or labels are listed in LABELS with the reason.
P1 each backtick token that looks like a path exists under study/, traders-net/ or eurisko-study/.
Exit 1 if a number has no source or a path is missing.
"""
import json, os, re, sys, datetime
ST = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TN = os.path.dirname(ST); ES = os.path.dirname(TN)
SRC = os.path.join(ST, 'D1', 'IDEAS-NET.md')
claims = open(os.path.join(ST, 'X1', 'CLAIMS.md')).read()
SECONDARY = ['X1/lead/gates.json', 'M1/lead/results/sign_census_summary.csv', 'M6/M6.md', 'L1/NOVELTY.md',
             'X1/STATUS.md', '../../wealth-net/RESULTS.md']
sec = {s: open(os.path.join(ST, s)).read() for s in SECONDARY}
LABELS = {'1989': 'year of OBD', '2019': 'year of Taylor pruning paper', '2026': 'date', '09': 'date',
          '28': 'date', '23': 'time 23:13Z', '13': 'time 13:51', '00': 'time 00:25Z', '25': 'time 00:25Z',
          '51': 'time', '07': 'time', '1': 'list/section index', '2': 'index', '3': 'index / seeds',
          '4': 'index', '5': 'index / seeds', '6': 'index', '7': 'index / figure', '8': 'Q8 / E8',
          '64': 'E64', '10': 'arms', '12': 'seconds', '209': 'CLAIMS header', '153': 'CLAIMS header',
          '19': 'CLAIMS header', '37': 'CLAIMS header', '16': 'LFSR16'}
t = open(SRC).read()
t2 = re.sub(r'https?://\S+', ' ', t)
t2 = re.sub(r'`[^`]+`', ' ', t2)           # paths are checked separately
toks = sorted(set(re.findall(r'(?<![\w.])[-−+]?\d+(?:\.\d+)?(?:e-?\d+)?', t2)))
res = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
           in_claims=[], secondary={}, labels={}, missing=[], paths_missing=[])
for n in toks:
    core = n.replace('−', '-').lstrip('+-')
    if re.search(r'(?<![\d.])' + re.escape(core) + r'(?![\d])', claims):
        res['in_claims'].append(n); continue
    hit = [s for s, x in sec.items() if re.search(r'(?<![\d.])' + re.escape(core) + r'(?![\d])', x)]
    if hit: res['secondary'][n] = hit; continue
    if core in LABELS: res['labels'][n] = LABELS[core]; continue
    res['missing'].append(n)
for p in sorted(set(re.findall(r'`([^`\s]+)`', t))):
    if not re.search(r'/|\.(csv|json|md|py|png|lock)$', p): continue
    p0 = p.split(' ')[0]
    cands = [os.path.join(r, p0) for r in (ST, TN, ES)]
    if not any(os.path.exists(c) for c in cands): res['paths_missing'].append(p)
res['ok'] = not res['missing'] and not res['paths_missing']
json.dump(res, open(os.path.join(ST, 'D1', 'check_ideas.json'), 'w'), indent=1)
print('numbers: %d in CLAIMS.md, %d secondary %s, %d labels, missing %s; paths missing %s; ok=%s' % (
    len(res['in_claims']), len(res['secondary']), res['secondary'], len(res['labels']), res['missing'],
    res['paths_missing'], res['ok']))
sys.exit(0 if res['ok'] else 1)
