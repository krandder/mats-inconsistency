#!/usr/bin/env python3
"""Prove the traders-net Grafana panels load. Writes check/check-<UTC>.json and check/latest.json.

Three stages (PREREG-w2.md):
  hermetic  serve.py on 127.0.0.1 over a temp fixture: endpoints answer, an appended row shows up on
            the next request (CSVs re-read per refresh), a half-written row is skipped, nan -> empty.
  exporter  every run in the live /runs.json answers 200 with >= 1 data row.
  grafana   through Grafana itself (kubectl exec into the grafana pod on deploy-host, anonymous viewer):
            dashboard uid traders-net exists; panel 1's Infinity query returns a frame for every run;
            every panel's column regex matches >= 1 numeric non-empty field in >= 1 run.
            Panel verdicts: PASS, PENDING (no run has the columns yet), FAIL.
Exit 0 when nothing FAILs. Usage: check.py [--skip-grafana] [--runs REGEX]
"""
import argparse, copy, csv, importlib.util, io, json, os, re, shutil, subprocess, sys, tempfile, threading, time
import urllib.request
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

HERE = '/home/agent/workspaces/eurisko-study/traders-net/study/V1'
OUT = '/home/agent/workspaces/eurisko-study/traders-net/study/D1/w2/grafana-check'
PENDING_SOURCE = {r'ec_': 'M5 exploit.py (E_C) via S0 runlog.py', r'price_': 'S0 runlog.py (price per family)',
                  r'births': 'S0 runlog.py / K1', r'bankrupt': 'S0 runlog.py / K1', r'gini_null': 'S0 panel.py'}


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def csv_rows(body):
    rows = list(csv.reader(io.StringIO(body.decode())))
    return rows[0] if rows else [], rows[1:]


# ---------------------------------------------------------------- hermetic
def hermetic():
    res = {'stage': 'hermetic', 'checks': []}

    def ok(name, cond, detail=''):
        res['checks'].append({'name': name, 'pass': bool(cond), 'detail': detail})

    spec = importlib.util.spec_from_file_location('v1serve', os.path.join(HERE, 'grafana', 'serve.py'))
    serve = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve)
    tmp = tempfile.mkdtemp(prefix='v1check-', dir='/tmp')
    try:
        serve.STUDY, serve.WEALTH = tmp, os.path.join(tmp, 'no-wealth-net')
        d = os.path.join(tmp, 'Z9', 'runs', 'fixture')
        os.makedirs(d)
        p = os.path.join(d, 'metrics.csv')
        head = ['step', 'price_ca30', 'cons_resid_total', 'ec_pat4', 'bce_ca30', 'brier_ca30']
        with open(p, 'w') as f:
            f.write(','.join(head) + '\n100,0.61,0.002,0.5,0.4,0.12\n200,0.64,nan,0.7,0.35,0.1\n')
        srv = ThreadingHTTPServer(('127.0.0.1', 0), serve.Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = 'http://127.0.0.1:%d' % srv.server_address[1]
        try:
            st, body = get(base + '/runs.json')
            runs = json.loads(body)
            ok('runs.json lists the fixture', st == 200 and [r['run'] for r in runs] == ['Z9.fixture'], body.decode()[:200])
            st, body = get(base + '/runs/Z9.fixture/metrics.csv')
            h, rows = csv_rows(body)
            ok('csv served with header and 2 rows', st == 200 and h == head and len(rows) == 2, '%s %d' % (h, len(rows)))
            ok('nan served as empty', rows[1][2] == '', repr(rows[1]))
            with open(p, 'a') as f:
                f.write('300,0.66,0.001,0.8,0.33,0.09\n400,0.7')        # a full row + a half-written one
            h, rows = csv_rows(get(base + '/runs/Z9.fixture/metrics.csv')[1])
            ok('appended row visible on next request (re-read per refresh)', len(rows) == 3 and rows[-1][0] == '300',
               'rows=%d last=%s' % (len(rows), rows[-1][0] if rows else None))
            ok('half-written last row skipped', all(len(r) == len(head) for r in rows))
            st, body = get(base + '/runs/Z9.fixture/metrics.json')
            js = json.loads(body)
            ok('json: same rows, numbers typed, nan -> null', st == 200 and len(js) == 3
               and isinstance(js[0]['price_ca30'], (int, float)) and js[1]['cons_resid_total'] is None,
               json.dumps(js[:2])[:300])
            try:
                get(base + '/runs/Z9.nope/metrics.csv')
                ok('unknown run is 404', False)
            except urllib.error.HTTPError as e:
                ok('unknown run is 404', e.code == 404, str(e.code))
            try:
                get(base + '/runs/..%2F..%2Fetc/metrics.csv')
                ok('path traversal refused', False)
            except urllib.error.HTTPError as e:
                ok('path traversal refused', e.code == 404, str(e.code))
        finally:
            srv.shutdown()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)     # own /tmp scratch
    res['pass'] = all(c['pass'] for c in res['checks'])
    return res


# ---------------------------------------------------------------- exporter
def exporter(base, run_rx):
    res = {'stage': 'exporter', 'base': base, 'runs': []}
    st, body = get(base + '/runs.json')
    runs = [r for r in json.loads(body) if re.search(run_rx, r['run'])]
    for r in runs:
        try:
            st, body = get(base + '/runs/%s/metrics.csv' % r['run'], timeout=120)
            h, rows = csv_rows(body)
            last = rows[-1][h.index('step')] if rows and 'step' in h else None
            res['runs'].append({'run': r['run'], 'status': st, 'rows': len(rows), 'cols': len(h), 'last_step': last,
                                'columns': h, 'pass': st == 200 and len(rows) >= 1})
        except Exception as e:  # noqa: BLE001 - report, do not crash
            res['runs'].append({'run': r['run'], 'error': repr(e), 'pass': False})
    res['pass'] = bool(res['runs']) and all(r['pass'] for r in res['runs'])
    return res


# ---------------------------------------------------------------- grafana
REMOTE = ("kubectl -n observability exec -i deploy/grafana -- sh -c '"
          "while IFS= read -r l; do case \"$l\" in GET\\ *) wget -qO- \"http://localhost:3000${l#GET }\";; "
          "*) wget -qO- --header=Content-Type:application/json --post-data=\"$l\" http://localhost:3000/api/ds/query;; "
          "esac; echo; done'")


def grafana_calls(lines, host):
    """Send GET paths / ds-query bodies to Grafana in one ssh session; one JSON reply per line."""
    for attempt in range(3):
        p = subprocess.run(['ssh', '-o', 'ConnectTimeout=20', host, REMOTE], input='\n'.join(lines) + '\n',
                           text=True, capture_output=True, timeout=600)
        out = [l for l in p.stdout.split('\n') if l.strip()]
        if len(out) == len(lines):
            return [json.loads(l) for l in out]
        time.sleep(5)
    raise RuntimeError('grafana: %d replies for %d requests; stderr: %s' % (len(out), len(lines), p.stderr[-400:]))


def walk(panels):
    for p in panels:
        yield p
        yield from walk(p.get('panels', []))


def frame_fields(frame):
    names = [f['name'] for f in frame['schema']['fields']]
    types = [f.get('type') for f in frame['schema']['fields']]
    vals = frame.get('data', {}).get('values', [[] for _ in names])
    return {n: (t, v) for n, t, v in zip(names, types, vals)}


def grafana(host, uid, run_rx, exp_runs):
    res = {'stage': 'grafana', 'uid': uid, 'runs': {}, 'panels': []}
    dash = grafana_calls(['GET /api/dashboards/uid/' + uid], host)[0]
    if 'dashboard' not in dash:
        res.update(pass_=False, error='dashboard not found: %s' % json.dumps(dash)[:300])
        res['pass'] = False
        return res
    d = dash['dashboard']
    res['version'], res['title'] = d.get('version'), d.get('title')
    panels = list(walk(d.get('panels', [])))
    src = next(p for p in panels if p.get('id') == 1)
    target = src['targets'][0]
    runs = [r['run'] for r in exp_runs if re.search(run_rx, r['run'])]
    bodies = []
    for r in runs:
        t = json.loads(json.dumps(target).replace('${run}', r).replace('$run', r))
        bodies.append(json.dumps({'queries': [t], 'from': 'now-6h', 'to': 'now'}, separators=(',', ':')))
    # the run variable query as Grafana runs it
    var = next((v for v in d.get('templating', {}).get('list', []) if v.get('name') == 'run'), None)
    if var and var.get('query', {}).get('infinityQuery'):
        q = dict(var['query']['infinityQuery'], datasource=target.get('datasource'))
        bodies.append(json.dumps({'queries': [q], 'from': 'now-6h', 'to': 'now'}, separators=(',', ':')))
    replies = grafana_calls(bodies, host)
    if var and var.get('query', {}).get('infinityQuery'):
        vr = replies.pop()
        try:
            fr = vr['results']['variable']['frames'][0]
            vals = frame_fields(fr).get('run', (None, []))[1]
        except (KeyError, IndexError):
            vals = []
        res['run_variable'] = {'values': len(vals), 'pass': set(runs) <= set(vals)}
    exp_last = {r['run']: r.get('last_step') for r in exp_runs}
    frames = {}
    for r, rep in zip(runs, replies):
        a = rep.get('results', {}).get('A', {})
        fr = (a.get('frames') or [None])[0]
        info = {'status': a.get('status'), 'error': a.get('error')}
        if fr:
            ff = frame_fields(fr)
            frames[r] = ff
            steps = [v for v in ff.get('step', (None, []))[1] if v is not None]
            info.update(fields=len(ff), rows=len(steps), last_step=steps[-1] if steps else None)
            try:
                info['fresh'] = exp_last.get(r) is not None and float(steps[-1]) == float(exp_last[r])
            except (TypeError, ValueError, IndexError):
                info['fresh'] = False
        info['pass'] = bool(fr) and not info['error'] and info.get('rows', 0) >= 1
        res['runs'][r] = info
    for p in panels:
        if p.get('type') in ('row',) or p.get('id') == 1 and p.get('type') == 'table':
            continue
        pat = None
        for t in p.get('transformations', []):
            if t.get('id') == 'filterFieldsByName':
                inc = t['options'].get('include', {})
                pat = inc.get('pattern') or ('^(%s)$' % '|'.join(map(re.escape, inc.get('names', []))) if inc.get('names') else None)
        uses_src = p.get('id') == 1 or any(t.get('panelId') == 1 for t in p.get('targets', []))
        if not uses_src or not pat:
            res['panels'].append({'id': p['id'], 'title': p.get('title'), 'verdict': 'SKIP',
                                  'why': 'not a panel-1 column panel'})
            continue
        rx = re.compile(pat)
        have, bad = {}, []
        for r, ff in frames.items():
            m = [n for n in ff if rx.search(n) and n != 'step']
            good = []
            for n in m:
                t, v = ff[n]
                nonnull = [x for x in v if x is not None]
                if t != 'number':
                    if nonnull:
                        bad.append('%s:%s is %s' % (r, n, t))
                elif nonnull:
                    good.append(n)
            if good:
                have[r] = len(good)
        if bad:
            verdict = 'FAIL'
        elif have:
            verdict = 'PASS'
        else:
            verdict = 'PENDING'
        why = ''
        if verdict == 'PENDING':
            why = '; '.join(sorted({s for k, s in PENDING_SOURCE.items() if k in pat})) or 'no run has these columns yet'
        res['panels'].append({'id': p['id'], 'title': p.get('title'), 'pattern': pat, 'verdict': verdict,
                              'runs_with_data': have, 'bad': bad[:10], 'why': why})
    res['pass'] = (all(i['pass'] for i in res['runs'].values()) and bool(res['runs'])
                   and all(p['verdict'] != 'FAIL' for p in res['panels'])
                   and res.get('run_variable', {}).get('pass', True))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', default='http://[ip-removed]:8767')
    ap.add_argument('--host', default='deploy-host', help='ssh host with kubectl for the fleet cluster')
    ap.add_argument('--uid', default='traders-net')
    ap.add_argument('--runs', default='.', help='regex on run ids to check')
    ap.add_argument('--skip-grafana', action='store_true')
    a = ap.parse_args()
    t0 = datetime.now(timezone.utc)
    report = {'checked_at': t0.isoformat(timespec='seconds'), 'stages': []}
    report['stages'].append(hermetic())
    try:
        ex = exporter(a.base, a.runs)
    except Exception as e:  # noqa: BLE001
        ex = {'stage': 'exporter', 'base': a.base, 'error': repr(e), 'pass': False, 'runs': []}
    report['stages'].append(ex)
    if not a.skip_grafana:
        try:
            report['stages'].append(grafana(a.host, a.uid, a.runs, ex['runs']))
        except Exception as e:  # noqa: BLE001
            report['stages'].append({'stage': 'grafana', 'error': repr(e), 'pass': False})
    for r in ex.get('runs', []):
        r.pop('columns', None)
    report['pass'] = all(s['pass'] for s in report['stages'])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'check-%s.json' % t0.strftime('%Y%m%dT%H%M%SZ'))
    for p in (path, os.path.join(OUT, 'latest.json')):
        with open(p, 'w') as f:
            json.dump(report, f, indent=1)
    for s in report['stages']:
        line = '%-9s %s' % (s['stage'], 'PASS' if s['pass'] else 'FAIL')
        if s['stage'] == 'grafana' and 'panels' in s:
            line += '  panels: ' + ', '.join('%s=%s' % (p['id'], p['verdict']) for p in s['panels'])
        if s.get('error'):
            line += '  ' + s['error'][:200]
        print(line)
    print(path)
    sys.exit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
