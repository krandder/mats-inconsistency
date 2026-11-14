"""Peek at R1 progress from the streamed GPU log (the JSON outputs only come back at the end).
python scratch/peek.py /tmp/l4runs/ngfam1/gpu-run.out [cut]"""
import re, sys
from collections import defaultdict
pat = re.compile(r'^net (\d+) rep (\d) cut (\d+) (\S+)\s+(.*?)\s+[\d.]+s$')
D = defaultdict(dict)
for line in open(sys.argv[1], errors='ignore'):
    m = pat.match(line.strip())
    if not m:
        continue
    net, rep, cut, fam, rest = m.groups()
    vals = dict((int(k), float(v)) for k, v in re.findall(r'L(\d+):(\S+)', rest))
    D[(int(cut), fam)][(int(net), int(rep))] = vals
cuts = sorted(set(c for c, _ in D))
for c in cuts:
    fams = [f for (cc, f) in D if cc == c]
    Ls = sorted(set(L for f in fams for v in D[(c, f)].values() for L in v))
    print(f'cut {c}: mean over (net, rep) [count]; L = ' + ' '.join(map(str, Ls)))
    for f in fams:
        v = D[(c, f)]
        row = []
        for L in Ls:
            xs = [x[L] for x in v.values() if L in x]
            row.append(f'{sum(xs) / len(xs):9.2e}' if xs else '        -')
        print(f'  {f:6s} [{len(v)}] ' + ' '.join(row))
