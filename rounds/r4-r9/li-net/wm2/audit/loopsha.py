import hashlib, re, glob
L = open('/home/agent/workspaces/eurisko-study/li-net/wm2/LOOP.md', 'rb').read()
shas = {}
for p in glob.glob('/tmp/wm2/logs/launch-wm2-*.log'):
    for l in open(p):
        m = re.search(r'(\S+Z) LAUNCH (\S+) .*loop_sha=([0-9a-f]+)', l)
        if m: shas.setdefault(m.group(3), []).append((m.group(1), m.group(2)))
lines = L.split(b'\n')
found = {}
for k in range(len(lines) + 1):
    for tail in (b'', b'\n'):
        pref = b'\n'.join(lines[:k]) + tail
        h = hashlib.sha256(pref).hexdigest()[:16]
        if h in shas and h not in found:
            found[h] = k
print('current LOOP.md sha', hashlib.sha256(L).hexdigest()[:16], 'lines', len(lines))
for h, launches in sorted(shas.items(), key=lambda x: min(x[1])):
    print(h, sorted(launches)[0][0], [x[1] for x in sorted(launches)][:3], '...' if len(launches) > 3 else '', '-> prefix of current file up to line', found.get(h))
