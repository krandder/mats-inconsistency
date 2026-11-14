"""E18 data-quality check: share of answered calls whose text is a bare number (optionally in bold), per arm, and examples of
the rest. usage: python e18_format.py"""
import collections, json, re

PAT = re.compile(r'^\s*(\*\*)?\s*[-+]?\d+(\.\d+)?\s*(\*\*)?\s*$')
by = collections.defaultdict(lambda: [0, 0]); odd = []
for line in open('results/e18_calls.jsonl'):
    r = json.loads(line)
    if not r['ok']:
        continue
    by[r['arm']][0] += 1
    if PAT.match(r['text']):
        by[r['arm']][1] += 1
    else:
        odd.append((r['arm'], r['text'][:150]))
print({k: f'{v[1]}/{v[0]} bare' for k, v in by.items()})
for o in odd[:10]:
    print(o)
