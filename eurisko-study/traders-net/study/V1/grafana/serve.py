#!/usr/bin/env python3
"""Read-only HTTP export of traders-net run metrics for Grafana (Infinity datasource).

GET /runs.json                -> [{"run", "rows", "updated", "source"}]
GET /runs/<id>/metrics.csv    -> the run's CSV (x column `step`; see ../SCHEMA.md)
GET /runs/<id>/metrics.json   -> the same rows as a JSON list of objects, numbers typed, empty -> null
                                 (Grafana Infinity infers number columns from JSON, not from CSV)
Everything else is 404. Sources are re-read on every request (wealth-net snapshots are cached by
directory mtime and file count), so live runs show up at once.

Run ids:
  <ws>.<run>        traders-net/study/<ws>/runs/<run>/metrics.csv, served as written (nan -> empty,
                    a half-written last row skipped)
  wealth-net.<tag>  derived from wealth-net/results/<tag>/snapshots/step-*.json (read-only)
"""
import argparse, csv, glob, io, json, os, re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.realpath(os.path.join(HERE, '../..'))
WEALTH = os.path.realpath(os.path.join(STUDY, '../../wealth-net/results'))
NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')
SCALES = ['out', 'head', 'neuron', 'comp']
FAMILIES = ['periodic', 'lfsr16', 'ca30', 'ca110', 'prime', 'pi', 'and', 'or']   # wealth-net streams.py order
CONS = ['cut0', 'cut1', 'cut2', 'cut3', 'cut4']   # cons_resid_rel_mean[k]: residual-stream cut k (embed, after L0 attn, L0 mlp, L1 attn, L1 mlp); wealth-net books.py


def csv_runs():
    for p in sorted(glob.glob(os.path.join(STUDY, '*', 'runs', '*', 'metrics.csv'))):
        ws, run = p.split(os.sep)[-4], p.split(os.sep)[-2]
        if NAME.match(ws) and NAME.match(run):
            yield f'{ws}.{run}', p


def wn_runs():
    for d in sorted(glob.glob(os.path.join(WEALTH, '*', 'snapshots'))):
        tag = d.split(os.sep)[-2]
        if NAME.match(tag):
            yield f'wealth-net.{tag}', d


def clean_csv(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    if not rows:
        return b'', 0
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    w.writerow(rows[0])
    n = 0
    for row in rows[1:]:
        if len(row) != len(rows[0]):
            continue
        w.writerow(['' if v.lower() in ('nan', 'inf', '-inf') else v for v in row])
        n += 1
    return out.getvalue().encode(), n


def fmt(x):
    return '' if x is None else format(float(x), '.7g')


_cache = {}


def wn_csv(d):
    files = sorted(glob.glob(os.path.join(d, 'step-*.json')))
    key = (len(files), os.path.getmtime(files[-1]) if files else 0)
    if _cache.get(d, (None,))[0] == key:
        return _cache[d][1]
    cols = (['step', 'loss', 'R_total'] + [f'wealth_{s}' for s in SCALES]
            + [f'wf_{s}_{f}' for s in SCALES for f in FAMILIES]
            + [f'gini_{s}' for s in SCALES] + [f'churn_{s}' for s in SCALES] + [f'turnover_{s}' for s in SCALES]
            + [f'neg_{s}' for s in SCALES] + [f'cons_resid_{c}' for c in CONS] + ['cons_sum_resid_max']
            + [f'bce_{f}' for f in FAMILIES] + [f'brier_{f}' for f in FAMILIES]
            + [f'bce_logistic_{f}' for f in FAMILIES])
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    w.writerow(cols)
    n = 0
    for p in files:
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, ValueError):      # being written
            continue
        row = [s['step'], fmt(s['loss']), fmt(s['R_total'])]
        row += [fmt(sum(s['wealth'][k])) for k in SCALES]
        row += [fmt(v) for k in SCALES for v in s['wealth_by_family'][k]]
        row += [fmt(s['gini_pos'][k]) for k in SCALES]
        row += [fmt(None if s['wealth_rank_corr_vs_1000_steps_ago'][k] is None else 1 - s['wealth_rank_corr_vs_1000_steps_ago'][k]) for k in SCALES]
        row += [fmt(s['turnover_income_rank_corr'][k]) for k in SCALES]
        row += [fmt(s['n_negative'][k] / max(1, len(s['wealth'][k]))) for k in SCALES]
        row += [fmt(v) for v in s['cons_resid_rel_mean']] + [fmt(s['sum_resid_max'])]
        row += [fmt(v) for v in s['ev_bce']] + [fmt(v) for v in s['ev_brier']] + [fmt(v) for v in s['ev_bce_log']]
        w.writerow(row)
        n += 1
    body = (out.getvalue().encode(), n)
    _cache[d] = (key, body)
    return body


def to_json(body):
    rows = list(csv.reader(io.StringIO(body.decode())))
    out = []
    for row in rows[1:]:
        o = {}
        for k, v in zip(rows[0], row):
            try:
                o[k] = float(v) if v != '' else None
            except ValueError:
                o[k] = v
        out.append(o)
    return json.dumps(out, allow_nan=False).encode()


def mtime(p):
    return datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(timespec='seconds')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        if self.command == 'GET':
            self.wfile.write(body)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parts = urlsplit(self.path).path.strip('/').split('/')
        try:
            if parts == ['runs.json']:
                runs = [{'run': r, 'rows': clean_csv(p)[1], 'updated': mtime(p), 'source': 'study'} for r, p in csv_runs()]
                runs += [{'run': r, 'rows': wn_csv(d)[1], 'updated': mtime(d), 'source': 'wealth-net'} for r, d in wn_runs()]
                return self.send(200, json.dumps(runs).encode(), 'application/json')
            if len(parts) == 3 and parts[0] == 'runs' and parts[2] in ('metrics.csv', 'metrics.json'):
                body = None
                for r, p in csv_runs():
                    if r == parts[1]:
                        body = clean_csv(p)[0]
                for r, d in wn_runs():
                    if r == parts[1]:
                        body = wn_csv(d)[0]
                if body is not None:
                    if parts[2] == 'metrics.json':
                        return self.send(200, to_json(body), 'application/json')
                    return self.send(200, body, 'text/csv; charset=utf-8')
        except (OSError, KeyError, ValueError, TypeError) as e:
            return self.send(500, f'error: {e!r}\n'.encode(), 'text/plain')
        self.send(404, b'not found\n', 'text/plain')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bind', default='[ip-removed]')
    ap.add_argument('--port', type=int, default=8767)
    a = ap.parse_args()
    ThreadingHTTPServer((a.bind, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
