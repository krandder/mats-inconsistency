import glob, re, collections
J = '/home/agent/workspaces/eurisko-study/li-net/wm2/jobs/'
files = sorted(glob.glob(J + 'F1-*.txt') + glob.glob(J + 'F2-*.txt'))
lines = [l.strip() for f in files for l in open(f) if l.startswith('--')]
print('F1+F2 job lines:', len(lines), 'files', len(files))
cov = collections.defaultdict(set)
bad = []
for l in lines:
    task = l.split()[1]; mode = l.split()[2]
    n = int(re.search(r'--n (\d+)', l).group(1)); s = int(re.search(r'--seed (\d+)', l).group(1))
    tag = re.search(r'--(?:tag|wmtag) (\S+)', l).group(1)
    pat = 25 if n <= 1000 else 40
    need = ['es_boards=3000', 'max_steps=40000', f'patience={pat}']
    if any(x not in l for x in need) or (mode != 'wm' and '--test' not in l):
        bad.append(l[:120])
    cov[(task, tag, n)].add(s)
print('lines missing protocol keys or --test (forecasters):', len(bad), bad[:3])
for k in sorted(cov):
    if cov[k] != {0, 1, 2, 3, 4}: print('  incomplete seeds', k, sorted(cov[k]))
print('(task, tag) -> number of N with seeds 0-4:', dict(collections.Counter((k[0], k[1]) for k, v in cov.items() if v == {0,1,2,3,4})))
