#!/usr/bin/env python3
"""Rebuild the per-lane table in mats/ARTIFACTS.md from mats/ARTIFACTS.tsv (the rest of the page is kept as is).
usage: artifacts_summary.py REPO_MATS_DIR [--check]   (--check: exit 1 if the table in ARTIFACTS.md is out of date)"""
import os, sys
from collections import OrderedDict, defaultdict

M = sys.argv[1]
W = '/home/agent/workspaces/mats/work/'
MB = 1 << 20
WHAT = {'checkpoint': 'checkpoints', 'array': 'arrays', 'raw-output': 'raw run outputs', 'transcript': 'reviewer CLI transcripts'}
ORDER = ['mats/worlds/', 'mats/interp/', 'mats/reports/']

lines = open(os.path.join(M, 'ARTIFACTS.tsv')).read().splitlines()
hdr = lines[0].split('\t')
rows = [dict(zip(hdr, l.split('\t'))) for l in lines[1:]]
agg = OrderedDict()
for r in rows:
    key = r['lane'] if r['kind'] != 'transcript' else 'transcripts'
    a = agg.setdefault(key, dict(n=0, size=0, kinds=defaultdict(int), where=set(), rawext=set()))
    a['n'] += 1; a['size'] += int(r['size']); a['kinds'][r['kind']] += 1
    if r['kind'] == 'raw-output':
        a['rawext'].add(os.path.splitext(r['path'])[1].lstrip('.').lower())
    gp = r['general_path']
    a['where'].add('work/' + gp[len(W):].split('/')[0] + '/' if gp.startswith(W) else os.path.dirname(gp) + '/')


def label(k, n, a):
    """'12 arrays', '1 checkpoint', '6 raw run outputs (JSON)': the raw-output label names the file types present."""
    w = WHAT[k]
    if k == 'raw-output':
        names = {'json': 'JSON', 'log': 'logs', 'jsonl': 'JSONL', 'txt': 'text'}
        w = 'raw run outputs (' + ', '.join(sorted(names.get(e, e) for e in a['rawext'])) + ')'
    if n == 1:
        w = w.replace('checkpoints', 'checkpoint').replace('arrays', 'array').replace('outputs', 'output').replace('transcripts', 'transcript')
    return f'{n:,} {w}'


def fmt_size(b):
    return f'{b / MB:,.1f} MB' if b < 1024 * MB else f'{b / (1024 * MB):,.2f} GB'


# lane order: the order lanes first appear in the existing table, new lanes after, transcripts last
page = open(os.path.join(M, 'ARTIFACTS.md')).read().split('\n')
start = page.index('## Per lane')
t0 = next(i for i in range(start, len(page)) if page[i].startswith('| Lane'))
t1 = next(i for i in range(t0, len(page)) if page[i].startswith('| **Total**'))
seen = [l.split('|')[1].strip().strip('`').rstrip('/') for l in page[t0 + 2:t1]]
rank = lambda k: (k == 'transcripts', seen.index(k) if k in seen else len(seen), k)
table = page[t0:t0 + 2]
tot_n = tot_s = 0
for key, a in sorted(agg.items(), key=lambda kv: rank(kv[0])):
    what = ', '.join(label(k, n, a) for k, n in sorted(a['kinds'].items(), key=lambda t: -t[1]))
    where = ', '.join(f'`{w}`' for w in sorted(a['where']))
    lane = '`' + key + '/`' if key != 'transcripts' else 'review folders (see below)'
    table.append(f"| {lane} | {where} | {what} | {a['n']:,} | {fmt_size(a['size'])} |")
    tot_n += a['n']; tot_s += a['size']
table.append(f'| **Total** | | | **{tot_n:,}** | **{fmt_size(tot_s)}** |')
new = page[:t0] + table + page[t1 + 1:]
if '--check' in sys.argv:
    same = new == page
    print('table up to date' if same else 'table out of date')
    sys.exit(0 if same else 1)
open(os.path.join(M, 'ARTIFACTS.md'), 'w').write('\n'.join(new))
print(f'{len(rows)} rows, {len(agg)} lanes, {fmt_size(tot_s)}')
