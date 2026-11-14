#!/usr/bin/env python3
"""X1 lead: merge the five worker claim registers (X1/w1..w5/CLAIMS.md) into X1/CLAIMS.md with one status
scheme (X1/PREREG.md section 1), apply the lead seed-5 tie-breaks (X1/lead/results/tiebreak.json), and write
X1/lead/results/claims.json. Run from anywhere: python3 study/X1/make_claims.py
Do not edit CLAIMS.md by hand; edit this script (headline, gate and finding text live here) and rerun."""
import json, os, re, datetime, collections
X1 = os.path.dirname(os.path.abspath(__file__)); STUDY = os.path.dirname(X1)
NOW = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
WS_ORDER = ['S0', 'L1', 'T1', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9', 'K1', 'V1', 'G1']

def tables(path):
    """Yield (header, rows) for each markdown table in the file."""
    hdr, rows, out = None, [], []
    for l in open(path, encoding='utf-8'):
        if l.startswith('|'):
            cells = [c.strip() for c in l.rstrip('\n').strip().strip('|').split(' | ')]
            if l.startswith('|---'): continue
            if hdr is None: hdr = [c.lower() for c in cells]
            else: rows.append(cells)
        elif hdr is not None:
            out.append((hdr, rows)); hdr, rows = None, []
    if hdr is not None: out.append((hdr, rows))
    return out

def norm_status(s):
    s0 = s.replace('*', '').strip().lower()
    if s0.startswith('verified'): return 'verified'
    if s0.startswith('partly verified') or s0.startswith('plausible') or s0.startswith('unverified'): return 'unverified'
    if s0.startswith('refuted'): return 'refuted'
    if re.match(r'\d+ verified, \d+ unverified', s0): return 'unverified'
    raise ValueError(s)

def wsof(i):
    m = re.match(r'(S0|L1|T1|M\d|K1|V1|G1)', i); return m.group(1) if m else '?'

rows = []
def add(worker, ws, rid, claim, wstat, basis, checked, ev, rerun, notes):
    rows.append(dict(worker=worker, ws=ws, id=rid, claim=claim, worker_status=wstat.replace('*', ''), status=norm_status(wstat),
                     basis=basis, checked_at=checked, evidence=ev, rerun=rerun, notes=notes))

# w1: ws,id,claim,status,why,committed,fresh,tie-break s5,evidence,rerun (paths relative to traders-net/)
w1p = os.path.join(X1, 'w1', 'CLAIMS.md')
w1time = re.search(r'Generated (\S+)', open(w1p).read()).group(1)
for hdr, rs in tables(w1p):
    if hdr[:2] != ['ws', 'id']: continue
    for c in rs:
        ws, rid, claim, st, why, com, fresh, tb, ev, rr = c
        basis = 'fresh-seed' if fresh not in ('-', '') else ('deterministic-rerun' if re.search(r'rerun|identical|pristine', why + ev) else 'static')
        ev2 = re.sub(r'(?<![\w/])study/', '', ev)
        note = why + (f' [committed {com}; fresh {fresh}; seed-5 tie-break {tb}]' if fresh not in ('-', '') else '')
        add('w1', ws, rid, claim, st, basis, w1time, ev2, f'(from traders-net/) {rr}', note)

# w2: id,claim,claimed,re-read,fresh rerun,within 2 sd,status,checked_at,evidence,rerun,notes (paths relative to study/)
for hdr, rs in tables(os.path.join(X1, 'w2', 'CLAIMS.md')):
    if hdr[0] != 'id': continue
    for c in rs:
        rid, claim, claimed, rr_, fresh, within, st, ck, ev, rerun, notes = c
        rid = rid.replace(' (X1 finding)', '')
        basis = 'static' if fresh in ('n/a', '-', '') else ('fresh-segment' if rid.startswith('M2') else ('fresh-seed' if rid.startswith(('M1-a', 'M1-b', 'M3-B')) else 'deterministic-rerun'))
        add('w2', wsof(rid), rid, claim, st, basis, ck, ev, f'(from traders-net/) {rerun}',
            f'claimed {claimed}; fresh {fresh}; within 2 sd {within}. {notes}')

# w3: id,claim,status,basis,checked_at,evidence,rerun,notes (paths relative to study/)
for hdr, rs in tables(os.path.join(X1, 'w3', 'CLAIMS.md')):
    if hdr[0] != 'id': continue
    for c in rs:
        rid, claim, st, basis, ck, ev, rerun, notes = c
        b = 'fresh-seed' if 'seed-4 rerun' in basis else ('analysis' if basis.startswith('analysis') else 'static')
        wst = st.replace('*', '')
        n = notes + (' [w3 "partly verified": one of (inside 2 sd, qualitative holds) failed]' if 'partly' in wst else '')
        add('w3', wsof(rid), rid, claim, st, b, ck, ev, f'(from study/) {rerun.strip("`")}', n)

# w4: id,claim,claimed,re-read,fresh seed 4,within 2 sd,status,checked_at,evidence,rerun,notes (paths relative to study/)
for hdr, rs in tables(os.path.join(X1, 'w4', 'CLAIMS.md')):
    if hdr[0] != 'id': continue
    for c in rs:
        rid, claim, claimed, rr_, fresh, within, st, ck, ev, rerun, notes = c
        basis = 'fresh-seed' if fresh != 'n/a' else ('deterministic-rerun' if 'byte-identical' in ev + notes else 'static')
        add('w4', wsof(rid), rid, claim, st, basis, ck, ev, f'(from traders-net/) {rerun}',
            f'claimed {claimed}; seed 4: {fresh}. {notes}')

# w5: #,claim,status,checked_at,evidence,rerun (paths relative to X1/w5/ unless G1/ or V1/)
for hdr, rs in tables(os.path.join(X1, 'w5', 'CLAIMS.md')):
    if hdr[0] != '#': continue
    evof = {}
    for c in rs:
        rid, claim, st, ck, ev, rerun = c
        ev2 = re.sub(r'`((?:v1|g1|gpu-job)/)', r'`X1/w5/\1', ev)
        notes = ''
        if '`' not in ev2 and '/' not in ev2.split(' ')[0]:
            # w5 put a finding in the evidence cell; move it to notes and inherit the file of the row it reruns as
            notes = ev2
            m = re.match(r'as (\S+)', rerun)
            if m and m.group(1) in evof: ev2 = evof[m.group(1)]
            elif rid == 'V1.7': ev2 = '`../wealth-net/results/train-s*/snapshots` (read-only, listed by the rerun)'
            elif rid.startswith('V1.6'): ev2 = '`X1/w5/v1/fresh_seed_z.json`'
            else: ev2 = '`X1/w5/CLAIMS.md`'
        if rid.startswith('V1.6') and 'fresh_seed_z' not in ev2: ev2 = '`X1/w5/v1/fresh_seed_z.json`; ' + ev2
        evof[rid] = ev2
        basis = 'fresh-seed' if rid.startswith('V1.6') or rid == 'G1.12' else ('live' if 'live' in st else 'static')
        add('w5', wsof(rid), rid, claim, st, basis, ck, ev2, f'(from /tmp, W5=study/X1/w5) {rerun}', notes)

# lead seed-5 tie-breaks
tbp = os.path.join(X1, 'lead', 'results', 'tiebreak.json')
TB = json.load(open(tbp))['rows'] if os.path.exists(tbp) else {}
TBRUN = '(clean /tmp copies as in X1/PREREG.md section 3) bash X1/lead/runq.sh (jobs X1/lead/jobs.txt); python3 X1/lead/compare_s5_w3.py; python3 X1/lead/check_s5_w4.py; python3 X1/lead/tiebreak.py'
def f(x): return 'NA' if x is None else f'{x:.4g}'
for r in rows:
    t = TB.get(r['id'])
    if not t or r['worker'] not in ('w3', 'w4'): continue
    q = '; '.join(f"{x['quantity']}: 1-3 {f(x['mean123'])}+-{f(x['sd123'])}, s4 {f(x['seed4'])} (z {f(x['z4'])}), s5 {f(x['seed5'])} (z {f(x['z5'])}), 5-seed {f(x['mean5'])}+-{f(x['sd5'])} -> {x['status']}" for x in t['quantities'])
    r['lead_prev_status'] = r['status']; r['status'] = t['status']; r['basis'] = 'fresh-seed (4 and 5)'
    r['checked_at'] = json.load(open(tbp))['checked_at']
    r['evidence'] += '; X1/lead/results/tiebreak.json; X1/lead/results/seed5/'
    r['rerun'] += f' || lead: {TBRUN}'
    r['notes'] = f"LEAD seed-5 tie-break ({t['kind']}): {t['why']}. {q}. Worker note: {r['notes']}"

# X1 lead findings as rows (each has its own evidence)
LEAD_ROWS = json.load(open(os.path.join(X1, 'lead', 'lead_rows.json')))
for r in LEAD_ROWS:
    add('lead', r['ws'], r['id'], r['claim'], r['status'], r['basis'], r['checked_at'], r['evidence'], r['rerun'], r['notes'])

# ---------------------------------------------------------------- render
cnt = collections.Counter(r['status'] for r in rows)
byws = collections.defaultdict(collections.Counter)
for r in rows: byws[r['ws']][r['status']] += 1
H = json.load(open(os.path.join(X1, 'lead', 'headlines.json')))
idx = {r['id']: r for r in rows}
def E(x): return str(x).replace('\n', ' ').replace('\\|', '|').replace('|', '\\|')
L = []
L.append('# X1 CLAIMS: skeptic and reproduction register for the traders-net study\n')
L.append(f'Rendered {NOW} by `python3 study/X1/make_claims.py` from `X1/w1..w5/CLAIMS.md`, `X1/lead/results/tiebreak.json`, '
         '`X1/lead/headlines.json` and `X1/lead/lead_rows.json`. Do not edit by hand. Check: `python3 study/X1/check_claims.py` '
         '(re-runs every worker check and verifies that every evidence path exists; output `X1/lead/results/check_claims.json`).\n')
L.append(f'Rows: {len(rows)}; **verified {cnt["verified"]}, refuted {cnt["refuted"]}, unverified {cnt["unverified"]}**. '
         'Rule: `X1/PREREG.md` (one scheme for all workers; w3 "partly verified" and w5 "plausible" map to unverified, '
         'w5 "verified from receipts/live" to verified). Fresh-seed rule: fresh value within the committed 3-seed mean +- 2 sd '
         '(sample sd) and the qualitative claim holds; w1 and the lead ran seed 5 as tie-break when seed 4 fell outside.\n')
L.append('Paths: evidence is relative to `study/` (`/tmp/...` = X1 rerun scratch, may be trashed). Rerun commands say their working directory.\n')
L.append('| ws | verified | refuted | unverified | worker |\n|---|---|---|---|---|')
wmap = {}
for r in rows: wmap.setdefault(r['ws'], set()).add(r['worker'])
for ws in WS_ORDER + sorted(set(byws) - set(WS_ORDER)):
    if ws in byws: L.append(f"| {ws} | {byws[ws]['verified']} | {byws[ws]['refuted']} | {byws[ws]['unverified']} | {', '.join(sorted(wmap[ws]))} |")
L.append('\n## Headline reproduction per workstream\n')
L.append('| ws | headline | reproduced? | rows (status) | evidence |\n|---|---|---|---|---|')
for ws in WS_ORDER:
    h = H[ws]
    st = ', '.join(f"{i} ({idx[i]['status']})" for i in h['rows'])
    L.append(f"| {ws} | {E(h['headline'])} | {E(h['verdict'])} | {st} | {E(h['evidence'])} |")
G = json.load(open(os.path.join(X1, 'lead', 'gates.json')))
L.append('\n## Plan gates (STUDY-PLAN.md section 4)\n')
L.append('| gate | flag | X1 status | evidence | note |\n|---|---|---|---|---|')
for g in G:
    L.append(f"| {E(g['gate'])} | {g['flag']} | {E(g['status'])} | {E(g['evidence'])} | {E(g['note'])} |")
F = json.load(open(os.path.join(X1, 'lead', 'findings.json')))
L.append('\n## Cross-cutting findings (lead)\n')
for i, x in enumerate(F, 1):
    L.append(f"{i}. **{x['title']}** {x['text']} Evidence: {x['evidence']}.")
U = [r for r in rows if r['status'] == 'unverified']
L.append('\n## Known unverified (' + str(len(U)) + ' rows)\n')
L.append(', '.join(f"{r['id']}" for r in sorted(U, key=lambda r: (WS_ORDER.index(r['ws']) if r['ws'] in WS_ORDER else 99, r['id']))) + '. Reasons are in each row.')
L.append('\n## Register\n')
for ws in WS_ORDER + sorted(set(byws) - set(WS_ORDER)):
    rs = [r for r in rows if r['ws'] == ws]
    if not rs: continue
    L.append(f'\n### {ws}\n')
    L.append('| id | claim | status | basis | checked_at | evidence | rerun | notes | src |\n|---|---|---|---|---|---|---|---|---|')
    for r in sorted(rs, key=lambda r: ({'refuted': 0, 'unverified': 1, 'verified': 2}[r['status']], r['id'])):
        st = r['status'] + (f" (was {r['lead_prev_status']})" if r.get('lead_prev_status') and r['lead_prev_status'] != r['status'] else '')
        cells = [r['id'], r['claim'], f"**{st}**", r['basis'], r['checked_at'], r['evidence'], r['rerun'], r['notes'],
                 f"{r['worker']} ({r['worker_status']})"]
        L.append('| ' + ' | '.join(E(c) for c in cells) + ' |')
open(os.path.join(X1, 'CLAIMS.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
json.dump(dict(rendered_at=NOW, counts=dict(cnt), by_ws={k: dict(v) for k, v in byws.items()}, rows=rows),
          open(os.path.join(X1, 'lead', 'results', 'claims.json'), 'w'), indent=1)
print(json.dumps(dict(rows=len(rows), **cnt)))
