#!/usr/bin/env python3
"""Rebuild site/wave1-results.html from synced run metrics (no network, no model)."""
import json, math, pathlib, sys, time
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/home/agent/data/vast-wave1')
DIS = pathlib.Path('/home/agent/data/jev-disjoint-v1-filtered-20260919')
SRC = pathlib.Path(__file__).resolve().parents[1] / 'site' / 'wave1-results.html'
out = {}
status = {l.split()[0]: l for l in (R / 'results' / 'status.txt').read_text().splitlines()} if (R / 'results' / 'status.txt').exists() else {}
for d in sorted((R / 'results').iterdir()):
    if not d.is_dir() or d.name == 'timing': continue
    m = d / 'metrics.json'
    if not m.exists(): out[d.name] = {'epochs': [], 'status': 'waiting'}; continue
    h = json.load(open(m))
    eps = [{'epoch': r['epoch'], 'logloss': r['observed']['log_loss'], 'brier': r['observed']['brier'], 'cons': r['consistency_mse'], 'min': r.get('seconds', 0) / 60} for r in h]
    st = 'running'
    if d.name in status: st = 'done' if 'EXIT=0' in status[d.name] else 'failed'
    test = None
    if (d / 'test-metrics.json').exists():
        t = json.load(open(d / 'test-metrics.json')); o = t['observed']
        test = {'logloss': o['log_loss'], 'brier': o['brier'], 'acc': o['accuracy'], 'ece': o['ece'], 'cons': t['consistency_mse'], 'forms': {k: v['log_loss'] for k, v in t['full_by_form'].items()}}
    out[d.name] = {'epochs': eps, 'status': st, 'test': test}
def base(split):
    q = {json.loads(l)['query_id']: json.loads(l)['form'] for l in open(DIS / split / 'queries.jsonl')}
    lab = [(q[json.loads(l)['query_id']], json.loads(l)['label']) for l in open(DIS / split / 'one-label-targets.jsonl')]
    ll = lambda p, y: -(y * math.log(p) + (1 - y) * math.log(1 - p))
    pf = lambda f: .25 if ' AND ' in f else .75 if ' OR ' in f else .5
    tr = {}
    for f, y in [(qq[json.loads(l)['query_id']], json.loads(l)['label']) for qq in [{json.loads(l)['query_id']: json.loads(l)['form'] for l in open(DIS / 'train' / 'queries.jsonl')}] for l in open(DIS / 'train' / 'one-label-targets.jsonl')]:
        tr.setdefault(f, []).append(y)
    rate = {f: sum(v) / len(v) for f, v in tr.items()}
    return {'constant': math.log(2), 'boolean': sum(ll(pf(f), y) for f, y in lab) / len(lab),
            'train_rate': sum(ll(rate[f], y) for f, y in lab) / len(lab), 'n': len(lab)}
out['_base_validation'] = base('validation'); out['_base_test'] = base('test')
import statistics as st
diag = {}
for d in sorted((R / 'results').iterdir()):
    if not d.is_dir() or d.name == 'timing': continue
    for f in sorted(d.glob('predictions-validation-epoch-*.jsonl')):
        rows = [json.loads(l) for l in open(f)]; ep = int(f.stem.split('-')[-1])
        AND = [r['probability'] for r in rows if ' AND ' in r['form']]
        p = {r['pair_id'] + '|' + r['form']: r['probability'] for r in rows}; pairs = sorted({r['pair_id'] for r in rows})
        xs = [p[q + '|P'] for q in pairs]; ys = [p[q + '|NOT P'] for q in pairs]
        mx, my = st.mean(xs), st.mean(ys)
        corr = sum((a - mx) * (c - my) for a, c in zip(xs, ys)) / math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((c - my) ** 2 for c in ys))
        diag.setdefault(d.name, {})[ep] = {'sd_and': st.pstdev(AND), 'mean_and': st.mean(AND), 'corr': corr}
out['_diag'] = diag
rescore = {}
for d in sorted((R / 'results').iterdir()):
    if not d.is_dir(): continue
    for f in sorted(d.glob('rescore-test-epoch-*.json')):
        m = json.load(open(f)); rescore.setdefault(d.name, {})[int(f.stem.split('-')[-1])] = {'logloss': m['observed']['log_loss'], 'brier': m['observed']['brier'], 'acc': m['observed']['accuracy']}
out['_rescore'] = rescore
x = json.load(open(R / 'instance.json'))
out['_meta'] = {'elapsed_h': (time.time() - x['created_at']) / 3600, 'dph': 8.90, 'synced': (R / 'last-sync.txt').read_text().strip()}
html = SRC.read_text()
a, b = html.index('<script id="data" type="application/json">') + len('<script id="data" type="application/json">'), html.index('</script>', html.index('<script id="data"'))
SRC.write_text(html[:a] + json.dumps(out, separators=(',', ':')) + html[b:])
print('built', SRC, {k: v['status'] for k, v in out.items() if not k.startswith('_')})
