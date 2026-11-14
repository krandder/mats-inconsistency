"""Merge analysis JSONs: every (depth, keys) arm present in NEW replaces that arm in OLD; result goes to OUT.

usage: python merge_json.py OLD NEW OUT
"""
import json, os, sys

old, new, out = sys.argv[1:4]
for fn in ('analysis2.json', 'mech2.json', 'mech3.json', 'mech4.json'):
    a = json.load(open(os.path.join(old, fn))); b = json.load(open(os.path.join(new, fn)))
    rows = lambda x: x['models'] if isinstance(x, dict) else x
    arms = {(r['depth'], r['keys']) for r in rows(b)}
    merged = [r for r in rows(a) if (r['depth'], r['keys']) not in arms] + rows(b)
    if isinstance(a, dict):   # mech2: {'reference', 'models'}
        assert a['reference'] == b['reference']
        merged = {'reference': a['reference'], 'models': merged}
    json.dump(merged, open(os.path.join(out, fn), 'w'))
    print(fn, len(rows(merged)))
