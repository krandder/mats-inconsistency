#!/usr/bin/env python3
"""X1 lead check: re-run every worker check script, re-render the lead register, and verify the register.
Run: python3 study/X1/check_claims.py   (about 1-3 min; read-only outside study/X1/ and /tmp)
Checks:
 C1 worker checks: w1 make_claims.py + pi_offsets.py (JSON equal), w2 check_claims.py (all match),
    w3 check_w3.py (0 mismatch), w4 check_w4.py (number file checks ok), w5 recompute_summary.py (3 ways pass 1e-6).
 C2 lead: tiebreak.py and make_claims.py run; CLAIMS.md counts == claims.json counts; worker row counts ==
    the totals each worker states.
 C3 every row has status in {verified, refuted, unverified}, a checked_at, an evidence path and a rerun command.
 C4 every evidence path token in every row exists (study/, traders-net/, eurisko-study/ or X1/<worker>/ roots;
    /tmp scratch paths are listed separately, not required).
Output: X1/lead/results/check_claims.json; exit 1 if any C1-C3 check fails or any non-/tmp path is missing."""
import glob, json, os, re, subprocess, sys, datetime, tempfile, shutil
X1 = os.path.dirname(os.path.abspath(__file__)); ST = os.path.dirname(X1); TN = os.path.dirname(ST); ES = os.path.dirname(TN)
ENV = dict(os.environ, PYTHONPATH=os.path.join(ES, 'wealth-net', '.pylib'), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
res = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), checks=[])
def chk(name, ok, detail=''):
    res['checks'].append(dict(check=name, ok=bool(ok), detail=detail)); print(('PASS ' if ok else 'FAIL ') + name, detail if not ok else '')
def run(cmd, cwd=TN, timeout=900):
    p = subprocess.run(cmd, cwd=cwd, env=ENV, shell=True, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr
J = lambda p: json.load(open(p))

# C1 worker checks
rc, o, e = run('python3 study/X1/w1/make_claims.py')
t = re.search(r'Totals: (\d+) verified, (\d+) unverified, (\d+) refuted \(of (\d+)\)', open(f'{X1}/w1/CLAIMS.md').read())
chk('C1 w1 make_claims.py rc 0', rc == 0, e[-300:]); w1tot = int(t.group(4)) if t else None
tmp = tempfile.mkdtemp(prefix='x1check-', dir='/tmp')
old = J(f'{X1}/w1/results/pi_offsets.json')
rc, o, e = run(f'PYTHONPATH={ES}/wealth-net/.pylib:{ES}/wealth-net python3 study/X1/w1/pi_offsets.py 2000 {tmp}/pi.json')
new = J(f'{tmp}/pi.json') if os.path.exists(f'{tmp}/pi.json') else None
chk('C1 w1 pi_offsets.py reproduces frac_seeds_with_pair_in_window', new is not None and new.get('frac_seeds_with_pair_in_window') == old.get('frac_seeds_with_pair_in_window'),
    f"new {None if new is None else new.get('frac_seeds_with_pair_in_window')} vs {old.get('frac_seeds_with_pair_in_window')}; {o[-200:]}")
rc, o, e = run('python3 study/X1/w2/check_claims.py'); d = J(f'{X1}/w2/results/claims_check.json')
chk('C1 w2 check_claims.py: every quoted M1/M2/M3 number matches', rc == 0 and d['n'] == d['n_match'], f"{d['n_match']}/{d['n']} {e[-200:]}")
rc, o, e = run('python3 X1/w3/check_w3.py', cwd=ST); d = J(f'{X1}/w3/results/static_check.json')
bad = sum(v['MISMATCH'] + v['ERROR'] for v in d['summary'].values())
chk('C1 w3 check_w3.py: M5/M6/M7 quotes match their files', rc == 0 and bad == 0, f"{d['summary']} {e[-200:]}")
rc, o, e = run('python3 X1/w3/compare_w3.py', cwd=ST)
chk('C1 w3 compare_w3.py rc 0 (seed-4 files from X1/w3/results/seed4 when /tmp is gone)', rc == 0, e[-300:])
rc, o, e = run('python3 study/X1/w4/check_w4.py'); d = J(f'{X1}/w4/check_w4.json')
chk('C1 w4 check_w4.py: number file checks ok', rc == 0 and d['filechecks_ok'], e[-300:])
rc, o, e = run('python3 study/X1/w5/v1/recompute_summary.py')
try:
    d = json.loads(o); ok = all(v['pass_1e6'] for v in d['compare'].values())
except Exception as ex:
    d, ok = None, False
chk('C1 w5 recompute_summary.py: V1 summary equals CSV, exporter and raw snapshots (rel 1e-6)', rc == 0 and ok, (e or o)[-300:])
if d: json.dump(d, open(f'{tmp}/recompute_summary.json', 'w'))

# C2 lead render
rc, o, e = run('python3 X1/lead/tiebreak.py', cwd=ST); chk('C2 lead tiebreak.py rc 0', rc == 0, e[-300:])
rc, o, e = run('python3 X1/make_claims.py', cwd=ST); chk('C2 lead make_claims.py rc 0', rc == 0, e[-300:])
C = J(f'{X1}/lead/results/claims.json'); md = open(f'{X1}/CLAIMS.md').read()
m = re.search(r'verified (\d+), refuted (\d+), unverified (\d+)', md)
chk('C2 CLAIMS.md counts equal claims.json', m and [int(x) for x in m.groups()] == [C['counts'].get(k, 0) for k in ('verified', 'refuted', 'unverified')], m.group(0) if m else 'no count line')
per = {w: sum(r['worker'] == w for r in C['rows']) for w in ('w1', 'w2', 'w3', 'w4', 'w5', 'lead')}
want = dict(w1=w1tot, w2=int(re.search(r'Rows: (\d+)', open(f'{X1}/w2/CLAIMS.md').read()).group(1)),
            w3=sum(int(x) for x in re.findall(r'(\d+)', re.search(r'Claim statuses: ([^\n]+)', open(f'{X1}/w3/CLAIMS.md').read()).group(1))),
            w4=int(re.search(r'Rows: (\d+)', open(f'{X1}/w4/CLAIMS.md').read()).group(1)))
chk('C2 every worker row is in the register (counts equal the worker totals)', all(per[k] == v for k, v in want.items()), f'{per} vs {want}')
rows_md = len([l for l in md.splitlines() if re.match(r'\| \S+ \| .* \| \*\*(verified|refuted|unverified)', l)])
chk('C2 CLAIMS.md renders one line per row', rows_md == len(C['rows']), f'{rows_md} vs {len(C["rows"])}')
cur, badt = None, []
for l in open(f'{X1}/CLAIMS.md', encoding='utf-8').read().split('\n'):
    if l.startswith('|'):
        n = len(re.findall(r'(?<!\\)\|', l))
        if cur is None: cur = n
        elif n != cur: badt.append(l[:80])
    else: cur = None
chk('C2 every CLAIMS.md table row has its header column count (pipes escaped)', not badt, '; '.join(badt[:5]))

# C3 fields
badf = [r['id'] for r in C['rows'] if r['status'] not in ('verified', 'refuted', 'unverified') or not r['checked_at'] or not r['evidence'].strip() or not r['rerun'].strip()]
chk('C3 every row has status, checked_at, evidence and rerun', not badf, str(badf[:10]))

# C4 evidence paths
ROOTS = [ST, TN, ES, os.path.join(ST, 'G1')]
TOK = re.compile(r'(/?[A-Za-z0-9_.{}*,\-]+(?:/[A-Za-z0-9_.{}*,\-<>]+)+/?)')
def expand(p):
    m = re.search(r'\{([^{}]*)\}', p)
    if not m: return [p]
    return [x for alt in m.group(1).split(',') for x in expand(p[:m.start()] + alt + p[m.end():])]
missing, tmp_missing, n_tok = [], [], 0
for r in C['rows']:
    wroot = os.path.join(X1, r['worker']) if r['worker'].startswith('w') else os.path.join(X1, 'lead')
    ev = re.sub(r'https?://\S+', ' ', r['evidence'])
    ev = re.sub(r'\{[A-Z_]+\}', '*', ev)  # {ARM}-style placeholders are globs
    if '`' in ev: ev = ' '.join(re.findall(r'`([^`]*)`', ev))  # backticked cells: only the backticked spans are paths
    for tok in TOK.findall(ev):
        tok = tok.rstrip('.,;:)').split(':')[0]
        if tok.startswith(('http', '//')) or '/' not in tok or re.match(r'^[\d./]+$', tok): continue
        if re.search(r'<[^>]+>', tok): continue
        n_tok += 1
        cands = expand(tok)
        def exists(c):
            if c.startswith('/'): return bool(glob.glob(c))
            return any(glob.glob(os.path.join(root, c)) for root in ROOTS + [wroot, os.path.join(ST, 'X1')])
        if all(exists(c) for c in cands): continue
        (tmp_missing if tok.startswith('/tmp/') else missing).append(f"{r['id']}: {tok}")
chk('C4 every non-/tmp evidence path exists', not missing, '; '.join(missing[:20]))
res.update(n_rows=len(C['rows']), counts=C['counts'], n_path_tokens=n_tok, missing_paths=missing, tmp_paths_gone=tmp_missing)
shutil.rmtree(tmp, ignore_errors=True)
res['ok'] = all(c['ok'] for c in res['checks'])
json.dump(res, open(f'{X1}/lead/results/check_claims.json', 'w'), indent=1)
print(json.dumps(dict(ok=res['ok'], n_rows=res['n_rows'], counts=res['counts'], path_tokens=n_tok, missing=len(missing), tmp_gone=len(tmp_missing))))
sys.exit(0 if res['ok'] else 1)
