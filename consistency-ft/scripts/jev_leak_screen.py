#!/usr/bin/env python3
"""Jev (TypeSafe Noul) leak screen for the Polymarket evidence pool.

Each evidence snippet gets one POST to https://api.typesafe.ai/v1/systemone carrying the
three leak questions (outcome_leak, post_date_event, date_inconsistent) as "noul"
questions over the same state; a noul answer is a yes-probability.

Subcommands
  calibrate  run three wordings (given / strict / simple) of every leak question on the
             Astra-labelled sample (scripts/astra_label_snippets.py), pick per type the
             wording and threshold maximising recall of Astra's positives subject to
             precision >= 0.5 (fallback: threshold 0.5 on the given wording when Astra
             finds < 15 positives or no threshold reaches the precision), then verify the
             deployed mixed-wording configuration on the same rows
  screen     run the chosen wordings over every row of leak-questions.jsonl
             (<= --rps requests/s, retry with backoff on 429/5xx, resumable)
  report     write leak-flags.jsonl (sorted by row) and leak-screen-report.md

The key is read from $TYPESAFE_API_KEY or [removed] and
never written anywhere. Jev is not a ground truth and Astra's labels are a proxy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

POOL = Path('/home/agent/data/polymarket-evidence-exa-20260922')
KEY_FILE = Path('[removed]')
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-latest'
TYPES = ('outcome_leak', 'post_date_event', 'date_inconsistent')
WORDING_IDS = ('given', 'strict', 'simple')
ASTRA_LABEL = {'outcome_leak': 'states_outcome', 'post_date_event': 'post_date_event',
               'date_inconsistent': 'date_inconsistent'}
PRICE_PER_INPUT_TOKEN = 0.042 / 1e6  # output tokens are free
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
MIN_PRECISION = 0.5
MIN_POSITIVES = 15
FALLBACK_THRESHOLD = 0.5


def day(ts):
    return (ts or '')[:10]


def given_text(record, kind):
    return next(q['text'] for q in record['questions'] if q['key'] == kind)


WORDINGS = {
    'outcome_leak': {
        'given': lambda r: given_text(r, 'outcome_leak'),
        'strict': lambda r: (f"Does this text report the actual, already-determined outcome of the question "
                             f"\"{r['question']}\" (how it resolved), rather than a forecast, poll, odds, "
                             f"or speculation about it?"),
        'simple': lambda r: f"Does this text say whether the answer to \"{r['question']}\" turned out to be yes or no?",
    },
    'post_date_event': {
        'given': lambda r: given_text(r, 'post_date_event'),
        'strict': lambda r: (f"Does this text report, as having already happened, an event that took place after "
                             f"{r['prediction_date']} (not a scheduled or expected future event)?"),
        'simple': lambda r: f"Does this text mention something that happened after {day(r['prediction_date'])}?",
    },
    'date_inconsistent': {
        'given': lambda r: given_text(r, 'date_inconsistent'),
        'strict': lambda r: (f"Was this text evidently written or updated after {r['publishedDate']}, its stated "
                             f"publication date, for example because it reports events or cites dates later than "
                             f"that, or because a date in its URL contradicts it?"),
        'simple': lambda r: f"Is the stated publication date {day(r['publishedDate'])} wrong for this text?",
    },
}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_key():
    if os.environ.get('TYPESAFE_API_KEY'):
        return os.environ['TYPESAFE_API_KEY'].strip()
    match = re.search(r'(?m)^(?:export )?TYPESAFE_API_KEY=(.+)$', KEY_FILE.read_text())
    if not match:
        raise RuntimeError('TYPESAFE_API_KEY unavailable')
    return match.group(1).strip().strip('"\'')


def state_for(record):
    return (f"Title: {record.get('title') or ''}\nURL: {record.get('url') or ''}\n"
            f"Stated publication date: {record.get('publishedDate') or 'unknown'}\n\n{record['snippet']}")


def payload_for(record, wordings):
    """wordings: {type: wording_id}; all three questions in one request over the same state."""
    return {'state': state_for(record), 'model': MODEL,
            'questions': {kind: {'type': 'noul', 'instructions': WORDINGS[kind][wordings[kind]](record)}
                          for kind in TYPES}}


class RateLimiter:
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def acquire(self):
        with self.lock:
            slot = max(self.next_time, time.monotonic())
            self.next_time = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class JevClient:
    def __init__(self, key, rps, max_attempts=8, timeout=60):
        self.key, self.limiter, self.max_attempts, self.timeout = key, RateLimiter(rps), max_attempts, timeout
        self.lock = threading.Lock()
        self.status_counts, self.attempts = {}, 0

    def _count(self, status):
        with self.lock:
            self.attempts += 1
            self.status_counts[str(status)] = self.status_counts.get(str(status), 0) + 1

    def call(self, payload):
        """Returns (data, attempts, elapsed). Raises RuntimeError after max_attempts or on a non-retryable error."""
        body = json.dumps(payload).encode()
        for attempt in range(1, self.max_attempts + 1):
            self.limiter.acquire()
            tick = time.perf_counter()  # latency of this attempt, excluding the rate-limit wait
            request = urllib.request.Request(ENDPOINT, data=body, headers={
                'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            status, raw, retryable = None, None, False
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status, raw = response.status, response.read().decode()
            except urllib.error.HTTPError as exc:
                status, raw = exc.code, exc.read().decode(errors='replace')[:500]
                retryable = status in RETRYABLE
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                status, raw, retryable = 'transport', str(exc)[:300], True
            self._count(status)
            if status == 200:
                try:
                    data = json.loads(raw)
                    parse_answers(data, payload)
                except (ValueError, KeyError, TypeError) as exc:
                    raise RuntimeError(f'invalid response: {exc}: {raw[:300]}')
                return data, attempt, time.perf_counter() - tick
            if not retryable:
                raise RuntimeError(f'HTTP {status}: {raw}')
            time.sleep(min(60.0, 2.0 ** attempt) * (0.5 + random.random()))
        raise RuntimeError(f'retries exhausted (last status {status}: {raw})')


def parse_answers(data, payload):
    answers = data['answers']
    out = {}
    for kind in payload['questions']:
        p = answers[kind]['noul']
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'invalid probability for {kind}')
        out[kind] = float(p)
    return out


def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def usage_of(data):
    usage = data.get('usage') or {}
    return int(usage.get('input_tokens') or 0), int(usage.get('output_tokens') or 0)


def run_requests(client, jobs, out_path, workers, label, progress_every=100):
    """jobs: list of (key, record, wordings); appends {key..., p, usage} lines to out_path; returns rows."""
    done_keys = {json.dumps(r['key']) for r in read_jsonl(out_path)}
    todo = [job for job in jobs if json.dumps(job[0]) not in done_keys]
    print(f'{label}: {len(jobs)} requests, {len(todo)} to do', flush=True)
    rows = []

    def work(job):
        key, record, wordings = job
        data, attempts, elapsed = client.call(payload_for(record, wordings))
        inp, outp = usage_of(data)
        return {'key': key, 'row': record['row'], 'p': parse_answers(data, payload_for(record, wordings)),
                'wordings': wordings, 'served_model': data.get('model'), 'input_tokens': inp,
                'output_tokens': outp, 'attempts': attempts, 'elapsed_seconds': round(elapsed, 3)}

    with open(out_path, 'a') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for count, result in enumerate(pool.map(work, todo), 1):
            handle.write(json.dumps(result) + '\n')
            handle.flush()
            rows.append(result)
            if count % progress_every == 0 or count == len(todo):
                print(f'{label}: {count}/{len(todo)} done', flush=True)
    return read_jsonl(out_path)


def confusion(probs, labels, threshold):
    tp = sum(1 for p, y in zip(probs, labels) if p >= threshold and y)
    fp = sum(1 for p, y in zip(probs, labels) if p >= threshold and not y)
    fn = sum(1 for p, y in zip(probs, labels) if p < threshold and y)
    tn = len(probs) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision,
            'recall': recall, 'flagged': tp + fp, 'flag_rate': (tp + fp) / len(probs) if probs else None}


def best_threshold(probs, labels, min_precision=MIN_PRECISION):
    """Highest-recall threshold with precision >= min_precision; ties -> higher precision, then higher threshold."""
    best, best_key = None, None
    for threshold in sorted(set(probs)):
        cm = confusion(probs, labels, threshold)
        if cm['precision'] is None or cm['precision'] < min_precision or not cm['tp']:
            continue
        key = (cm['recall'], cm['precision'], threshold)
        if best_key is None or key > best_key:
            best, best_key = cm, key
    return best


def select(kind, per_wording, labels, min_positives=MIN_POSITIVES):
    """per_wording: {wording_id: [probs aligned with labels]}. Returns the selection dict for one type."""
    positives = sum(labels)
    table = {}
    for wording, probs in per_wording.items():
        table[wording] = {'best': best_threshold(probs, labels),
                          'at_0.5': confusion(probs, labels, FALLBACK_THRESHOLD),
                          'mean_p_positive': (sum(p for p, y in zip(probs, labels) if y) / positives) if positives else None,
                          'mean_p_negative': (sum(p for p, y in zip(probs, labels) if not y) / (len(labels) - positives))
                          if len(labels) - positives else None}
    choice = {'type': kind, 'positives': positives, 'n': len(labels), 'table': table}
    candidates = [(w, t['best']) for w, t in table.items() if t['best']]
    if positives < min_positives:
        choice.update(wording='given', threshold=FALLBACK_THRESHOLD, fallback=True,
                      reason=f'Astra found only {positives} positives (< {min_positives}); threshold 0.5 on the given wording')
    elif not candidates:
        choice.update(wording='given', threshold=FALLBACK_THRESHOLD, fallback=True,
                      reason=f'no wording/threshold reached precision >= {MIN_PRECISION}; threshold 0.5 on the given wording')
    else:
        wording, best = max(candidates, key=lambda item: (item[1]['recall'], item[1]['precision'],
                                                          WORDING_IDS.index(item[0]) == 0))
        choice.update(wording=wording, threshold=best['threshold'], fallback=False,
                      reason=f"max recall with precision >= {MIN_PRECISION}")
    choice['confusion'] = confusion(per_wording[choice['wording']], labels, choice['threshold'])
    return choice


def split_half_estimate(kind, per_wording, labels, rows_order):
    """Select on one half (by sample order parity), evaluate on the other; both directions."""
    estimates = []
    for fold in (0, 1):
        train = [i for i, row in enumerate(rows_order) if i % 2 == fold]
        test = [i for i, row in enumerate(rows_order) if i % 2 != fold]
        sub = select(kind, {w: [ps[i] for i in train] for w, ps in per_wording.items()}, [labels[i] for i in train],
                     min_positives=math.ceil(MIN_POSITIVES / 2))  # the positive floor scales with the half
        cm = confusion([per_wording[sub['wording']][i] for i in test], [labels[i] for i in test], sub['threshold'])
        estimates.append({'fold': fold, 'wording': sub['wording'], 'threshold': sub['threshold'],
                          'test_positives': sum(labels[i] for i in test), **cm})
    return estimates


def calibrate(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sample = read_jsonl(args.astra_dir / 'sample.jsonl')
    labels_rows = {r['row']: r for r in read_jsonl(args.astra_dir / 'astra-labels.jsonl')}
    if not sample or not labels_rows:
        raise SystemExit('sample.jsonl / astra-labels.jsonl missing; run scripts/astra_label_snippets.py first')
    sample = [r for r in sample if r['row'] in labels_rows]
    client = JevClient(load_key(), args.rps)
    started = time.monotonic()
    jobs = [([r['row'], w], r, {kind: w for kind in TYPES}) for w in WORDING_IDS for r in sample]
    results = run_requests(client, jobs, out / 'calibration-jev.jsonl', args.workers, 'calibration')
    by_key = {json.dumps(r['key']): r for r in results}
    order = [r['row'] for r in sample]
    selection, held_out = {}, {}
    for kind in TYPES:
        labels = [bool(labels_rows[row][ASTRA_LABEL[kind]]) for row in order]
        per_wording = {w: [by_key[json.dumps([row, w])]['p'][kind] for row in order] for w in WORDING_IDS}
        selection[kind] = select(kind, per_wording, labels)
        held_out[kind] = split_half_estimate(kind, per_wording, labels, order)
    chosen = {kind: selection[kind]['wording'] for kind in TYPES}
    thresholds = {kind: selection[kind]['threshold'] for kind in TYPES}
    # Verification: the deployed mixed-wording request on the same rows (also a repeatability check).
    verify_jobs = [([r['row'], 'deployed'], r, chosen) for r in sample]
    verify = {r['row']: r for r in run_requests(client, verify_jobs, out / 'verification-jev.jsonl', args.workers,
                                                 'verification')}
    verification = {}
    for kind in TYPES:
        labels = [bool(labels_rows[row][ASTRA_LABEL[kind]]) for row in order]
        probs = [verify[row]['p'][kind] for row in order]
        cal_probs = [by_key[json.dumps([row, chosen[kind]])]['p'][kind] for row in order]
        verification[kind] = {**confusion(probs, labels, thresholds[kind]),
                              'mean_abs_diff_vs_calibration': sum(abs(a - b) for a, b in zip(probs, cal_probs)) / len(order),
                              'flag_agreement_vs_calibration': sum((a >= thresholds[kind]) == (b >= thresholds[kind])
                                                                   for a, b in zip(probs, cal_probs)) / len(order)}
    all_rows = results + list(verify.values())
    usage = {'requests': len(all_rows), 'input_tokens': sum(r['input_tokens'] for r in all_rows),
             'output_tokens': sum(r['output_tokens'] for r in all_rows),
             'http_attempts': sum(r['attempts'] for r in all_rows),  # persisted per row, so resumes stay right
             'http_status_counts_this_session': client.status_counts}
    usage['cost_dollars'] = round(usage['input_tokens'] * PRICE_PER_INPUT_TOKEN, 4)
    astra_summary = json.loads((args.astra_dir / 'summary.json').read_text()) if (args.astra_dir / 'summary.json').exists() else {}
    calibration = {'created_at': now(), 'model': MODEL, 'served_models': sorted({r['served_model'] for r in all_rows if r.get('served_model')}),
                   'sample_rows': len(order), 'astra': {'labelled': len(labels_rows), 'positives': astra_summary.get('positives'),
                                                        'positives_by_stratum': astra_summary.get('positives_by_stratum'),
                                                        'strata': astra_summary.get('strata'), 'usage': astra_summary.get('usage'),
                                                        'failed_batches': astra_summary.get('failed_batches')},
                   'wording_examples': {kind: {w: WORDINGS[kind][w](sample[0]) for w in WORDING_IDS} for kind in TYPES},
                   'min_precision': MIN_PRECISION, 'min_positives': MIN_POSITIVES,
                   'chosen_wordings': chosen, 'thresholds': thresholds, 'selection': selection,
                   'split_half': held_out, 'verification': verification, 'usage': usage,
                   'elapsed_seconds': round(time.monotonic() - started, 1)}
    (out / 'calibration.json').write_text(json.dumps(calibration, indent=1) + '\n')
    for kind in TYPES:
        s = selection[kind]
        print(f"{kind}: positives={s['positives']} -> wording={s['wording']} threshold={s['threshold']} "
              f"recall={s['confusion']['recall']} precision={s['confusion']['precision']} fallback={s['fallback']}", flush=True)
    print(f"usage: {usage}", flush=True)


def load_progress(path):
    if path.exists():
        return json.loads(path.read_text())
    return {'started_at': now(), 'elapsed_seconds': 0.0, 'done': 0, 'total': None, 'input_tokens': 0,
            'output_tokens': 0, 'http_attempts': 0, 'http_status_counts': {}, 'failures': 0, 'resumes': 0}


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=1) + '\n')
    temporary.replace(path)


def screen(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((out / 'calibration.json').read_text())
    chosen, thresholds = calibration['chosen_wordings'], calibration['thresholds']
    partial, progress_path, errors_path = out / 'leak-flags.partial.jsonl', out / 'screen-progress.json', out / 'screen-errors.jsonl'
    done = set()
    if partial.exists():
        with partial.open() as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line)['row'])
    progress = load_progress(progress_path)
    progress['resumes'] += 1 if progress['done'] else 0
    progress.update(chosen_wordings=chosen, thresholds=thresholds, rps=args.rps, workers=args.workers)
    client = JevClient(load_key(), args.rps)
    session_start = time.monotonic()
    base_elapsed = progress['elapsed_seconds']
    consecutive_failures, lock = 0, threading.Lock()

    def work(record):
        payload = payload_for(record, chosen)
        data, attempts, elapsed = client.call(payload)
        probs = parse_answers(data, payload)
        inp, outp = usage_of(data)
        return {'row': record['row'], 'id': record['id'], 'url': record['url'], 'query_idx': record.get('query_idx'),
                'snippet_index': record.get('snippet_index'), 'publishedDate': record.get('publishedDate'),
                'prediction_date': record['prediction_date'],
                'p': {kind: probs[kind] for kind in TYPES},
                'flags': {kind: probs[kind] >= thresholds[kind] for kind in TYPES},
                'wordings': dict(chosen), 'served_model': data.get('model'),
                'input_tokens': inp, 'output_tokens': outp, 'attempts': attempts, 'elapsed_seconds': round(elapsed, 3)}

    def flush_progress():
        progress.update(elapsed_seconds=round(base_elapsed + time.monotonic() - session_start, 1), last_update=now(),
                        http_attempts=progress_http_base + client.attempts,
                        http_status_counts={k: base_status.get(k, 0) + v for k, v in
                                            {**{k: 0 for k in base_status}, **client.status_counts}.items()})
        save_json(progress_path, progress)

    progress_http_base, base_status = progress['http_attempts'], dict(progress['http_status_counts'])
    with (args.pool / 'leak-questions.jsonl').open() as source:
        progress['total'] = sum(1 for line in source if line.strip())  # known up front for the progress lines
    if args.limit:
        progress['total'] = min(progress['total'], args.limit)
    total, submitted, completed_now = 0, 0, 0
    pending = {}
    stop_reason = None
    with partial.open('a') as handle, errors_path.open('a') as errors, ThreadPoolExecutor(max_workers=args.workers) as pool:
        def drain(block):
            nonlocal consecutive_failures, completed_now, stop_reason
            if not pending:
                return
            finished, _ = wait(list(pending), return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for future in finished:
                record = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - recorded, screen continues
                    consecutive_failures += 1
                    progress['failures'] += 1
                    errors.write(json.dumps({'row': record['row'], 'id': record['id'], 'at': now(), 'error': str(exc)[:500]}) + '\n')
                    errors.flush()
                    if consecutive_failures >= args.max_consecutive_failures:
                        stop_reason = f'{consecutive_failures} consecutive failures'
                    continue
                consecutive_failures = 0
                handle.write(json.dumps(result) + '\n')
                handle.flush()
                progress['done'] += 1
                progress['input_tokens'] += result['input_tokens']
                progress['output_tokens'] += result['output_tokens']
                completed_now += 1
                if completed_now % 500 == 0:
                    flush_progress()
                    rate = completed_now / max(1e-9, time.monotonic() - session_start)
                    print(f"screen: {progress['done']}/{progress['total'] or '?'} done, {rate:.2f} rows/s this session, "
                          f"failures {progress['failures']}, status {client.status_counts}", flush=True)

        with (args.pool / 'leak-questions.jsonl').open() as source:
            for row, line in enumerate(source):
                total += 1
                if args.limit and total > args.limit:
                    break
                if row in done or stop_reason:
                    continue
                record = json.loads(line)
                record['row'] = row
                while len(pending) >= args.workers * 4:
                    drain(block=True)
                pending[pool.submit(work, record)] = record
                submitted += 1
        while pending:
            drain(block=True)
    progress['stop_reason'] = stop_reason
    progress['finished_at'] = now() if progress['done'] >= (progress['total'] or 0) else None
    flush_progress()
    print(f"screen finished: done {progress['done']}/{progress['total']}, failures {progress['failures']}, "
          f"stop_reason {stop_reason}, elapsed {progress['elapsed_seconds']}s", flush=True)
    if stop_reason:
        sys.exit(2)


def pct(x, digits=2):
    return 'n/a' if x is None else f'{100 * x:.{digits}f}%'


def fmt(x, digits=3):
    return 'n/a' if x is None else f'{x:.{digits}f}'


def report(args):
    out = args.out
    calibration = json.loads((out / 'calibration.json').read_text())
    progress = json.loads((out / 'screen-progress.json').read_text())
    rows = {}
    with (out / 'leak-flags.partial.jsonl').open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                rows[record['row']] = record  # last write wins
    ordered = [rows[row] for row in sorted(rows)]
    total_rows = progress.get('total') or len(ordered)
    final_fields = ('row', 'id', 'url', 'query_idx', 'snippet_index', 'publishedDate', 'prediction_date', 'p', 'flags',
                    'wordings', 'served_model', 'input_tokens', 'output_tokens')
    with (args.pool / 'leak-flags.jsonl').open('w') as handle:
        for record in ordered:
            handle.write(json.dumps({k: record[k] for k in final_fields if k in record}) + '\n')
    n = len(ordered)
    flagged = {kind: sum(r['flags'][kind] for r in ordered) for kind in TYPES}
    any_flag = sum(any(r['flags'].values()) for r in ordered)
    n_types = [sum(r['flags'].values()) for r in ordered]
    # near/far strata as in the calibration sample
    def near(record):
        pub = record.get('publishedDate')
        if not pub:
            return False
        delta = datetime.fromisoformat(record['prediction_date'].replace('Z', '+00:00')) - datetime.fromisoformat(pub.replace('Z', '+00:00'))
        return delta.days < 7
    strata = {'near (published < 7 days before prediction date)': [r for r in ordered if near(r)],
              'far (>= 7 days)': [r for r in ordered if not near(r)]}
    by_question = {}
    for record in ordered:
        by_question.setdefault(record['id'], []).append(record)
    questions = len(by_question)
    lose_all = {kind: sum(all(r['flags'][kind] for r in rs) for rs in by_question.values()) for kind in TYPES}
    lose_all_any = sum(all(any(r['flags'].values()) for r in rs) for rs in by_question.values())
    keep_count = [sum(not any(r['flags'].values()) for r in rs) for rs in by_question.values()]
    input_tokens = sum(r['input_tokens'] for r in ordered)
    cost = input_tokens * PRICE_PER_INPUT_TOKEN
    cal_cost = calibration['usage']['cost_dollars']
    served = sorted({r.get('served_model') for r in ordered if r.get('served_model')})
    elapsed_h = progress['elapsed_seconds'] / 3600
    lines = [f"# Leak screen report — polymarket-evidence-exa-20260922", '',
             f"Generated {now()}. Jev model requested `{MODEL}`, served {served}. Calibration created {calibration['created_at']}.",
             '', '## Summary', '',
             f"- Rows screened: {n} / {total_rows} ({pct(n / total_rows if total_rows else None)}); questions covered: {questions}",
             f"- Any flag: {any_flag} rows ({pct(any_flag / n)}); rows with 2+ flags: {sum(k >= 2 for k in n_types)}; all 3: {sum(k == 3 for k in n_types)}",
             *[f"- {kind}: {flagged[kind]} rows ({pct(flagged[kind] / n)}), wording `{calibration['chosen_wordings'][kind]}`, threshold {calibration['thresholds'][kind]}" for kind in TYPES],
             f"- Questions losing all evidence (every row flagged by any type): {lose_all_any} / {questions} ({pct(lose_all_any / questions)})",
             *[f"- Questions losing all evidence to {kind} alone: {lose_all[kind]} ({pct(lose_all[kind] / questions)})" for kind in TYPES],
             f"- Median unflagged rows per question: {sorted(keep_count)[len(keep_count) // 2]}; questions with <= 2 unflagged rows: {sum(k <= 2 for k in keep_count)}",
             f"- Screen cost: {input_tokens:,} input tokens = ${cost:.4f} (output free); calibration+verification ${cal_cost:.4f}; total ${cost + cal_cost:.4f}",
             f"- Screen runtime: {elapsed_h:.2f} h wall ({progress['elapsed_seconds']:.0f} s) at <= {progress.get('rps')} req/s, {progress.get('workers')} workers; "
             f"HTTP attempts {progress['http_attempts']}, status counts {progress['http_status_counts']}; row failures {progress['failures']}; stop reason {progress.get('stop_reason')}",
             '', '## Flag rates by stratum', '', '| stratum | rows | outcome_leak | post_date_event | date_inconsistent | any |', '|---|---:|---:|---:|---:|---:|']
    for name, rs in strata.items():
        if rs:
            lines.append(f"| {name} | {len(rs)} | " + ' | '.join(pct(sum(r['flags'][k] for r in rs) / len(rs)) for k in TYPES)
                         + f" | {pct(sum(any(r['flags'].values()) for r in rs) / len(rs))} |")
    lines += ['', '## Probability distribution (deployed wordings)', '',
              '| type | mean p | p >= 0.25 | p >= 0.5 | p >= 0.75 | p >= 0.9 |', '|---|---:|---:|---:|---:|---:|']
    for kind in TYPES:
        ps = [r['p'][kind] for r in ordered]
        lines.append(f"| {kind} | {fmt(sum(ps) / n)} | " + ' | '.join(pct(sum(p >= t for p in ps) / n) for t in (0.25, 0.5, 0.75, 0.9)) + ' |')
    astra = calibration['astra']
    astra_rows = read_jsonl(args.astra_dir / 'astra-labels.jsonl')
    outcome_only = sum(r['states_outcome'] and not r['post_date_event'] and not r['date_inconsistent'] for r in astra_rows)
    combos = {}
    for r in astra_rows:
        combo = '+'.join(k for k in ('states_outcome', 'post_date_event', 'date_inconsistent') if r[k]) or 'none'
        combos[combo] = combos.get(combo, 0) + 1
    lines += ['', '## Calibration against Astra labels', '',
              f"Sample: {calibration['sample_rows']} rows labelled by GPT-6 Astra (medium reasoning, Codex proxy, text-and-dates only), "
              f"stratified {astra.get('strata', {}).get('near_sampled')} near (< 7 days) / {astra.get('strata', {}).get('far_sampled')} far, seed {astra.get('strata', {}).get('seed')}. "
              f"Astra positives: {astra.get('positives')} (by stratum: {astra.get('positives_by_stratum')}). Failed Astra batches: {astra.get('failed_batches') or 'none'}.",
              '', f"Astra label combinations: {dict(sorted(combos.items(), key=lambda kv: -kv[1]))}. "
              f"{outcome_only} of the {astra.get('positives', {}).get('states_outcome')} states_outcome positives carry neither a post-date event nor a date "
              "inconsistency: the result was reported before the sampled prediction date, i.e. the market was effectively already resolved when the "
              "prediction was dated. Those are outcome disclosures, not date leaks; whether to drop the snippet or the question is a downstream decision.",
              '', f"Rule: per type, the wording and threshold (flag when p >= threshold) maximising recall of Astra positives subject to precision >= {calibration['min_precision']}; "
              f"fallback to threshold {FALLBACK_THRESHOLD} on the given wording when positives < {calibration['min_positives']} or no threshold reaches the precision floor. "
              'Thresholds are selected and evaluated on the same 300 rows (in-sample); the split-half rows give a rough held-out estimate.', '',
              '| type | Astra + | wording | threshold | TP | FP | FN | TN | recall | precision | flag rate | fallback |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for kind in TYPES:
        s = calibration['selection'][kind]
        c = s['confusion']
        lines.append(f"| {kind} | {s['positives']} | {s['wording']} | {c['threshold']} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} | "
                     f"{fmt(c['recall'])} | {fmt(c['precision'])} | {pct(c['flag_rate'])} | {'yes: ' + s['reason'] if s['fallback'] else 'no'} |")
    lines += ['', '### All wordings (best threshold with precision >= 0.5, and the fixed 0.5 threshold)', '',
              '| type | wording | best thr | recall | precision | flagged | thr 0.5 recall | thr 0.5 precision | thr 0.5 flagged | mean p (Astra +) | mean p (Astra -) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in TYPES:
        for wording in WORDING_IDS:
            t = calibration['selection'][kind]['table'][wording]
            b, h = t['best'], t['at_0.5']
            lines.append(f"| {kind} | {wording} | {b['threshold'] if b else 'none'} | {fmt(b['recall']) if b else 'n/a'} | {fmt(b['precision']) if b else 'n/a'} | "
                         f"{b['flagged'] if b else 'n/a'} | {fmt(h['recall'])} | {fmt(h['precision'])} | {h['flagged']} | {fmt(t['mean_p_positive'])} | {fmt(t['mean_p_negative'])} |")
    lines += ['', '### Split-half held-out estimate (select on one half, test on the other)', '',
              '| type | fold | wording | threshold | test + | TP | FP | FN | recall | precision |', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind in TYPES:
        for e in calibration['split_half'][kind]:
            lines.append(f"| {kind} | {e['fold']} | {e['wording']} | {e['threshold']} | {e['test_positives']} | {e['tp']} | {e['fp']} | {e['fn']} | {fmt(e['recall'])} | {fmt(e['precision'])} |")
    lines += ['', '### Deployed configuration re-run on the sample (mixed wordings in one request; repeatability)', '',
              '| type | TP | FP | FN | TN | recall | precision | mean abs p diff vs calibration run | flag agreement |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in TYPES:
        v = calibration['verification'][kind]
        lines.append(f"| {kind} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} | {fmt(v['recall'])} | {fmt(v['precision'])} | "
                     f"{fmt(v['mean_abs_diff_vs_calibration'])} | {pct(v['flag_agreement_vs_calibration'])} |")
    lines += ['', '### Wordings', '']
    for kind in TYPES:
        for wording in WORDING_IDS:
            lines.append(f"- `{kind}` / `{wording}`: {calibration['wording_examples'][kind][wording]}")
    lines += ['', 'State sent with each request: `Title: ...`, `URL: ...`, `Stated publication date: ...`, blank line, the Exa highlights snippet. '
              'The three questions of one snippet share one request (one `state`, three `noul` questions).', '',
              '## Caveats', '',
              '- Jev is not a ground truth: the probabilities are a model readout, and a flag is a screening signal, not a verified leak.',
              '- Astra labels are a proxy for human judgement (a single medium-effort pass, batched 10 snippets per session, judging text and dates only); '
              'recall/precision above are relative to that proxy, not to true leakage.',
              '- Thresholds were chosen on the same 300 rows they are evaluated on; expect lower recall/precision on the full pool. Small positive counts make the estimates noisy.',
              '- Wordings were calibrated one wording set per request; the deployed request mixes the chosen wordings (see the re-run table for the effect).',
              '- Rows missing from leak-flags.jsonl (if any) failed all retries; rerun `screen` to fill them.',
              '- `outcome_leak` also fires on results legitimately published before the prediction date (markets already resolved at the sampled prediction date); '
              'it measures outcome disclosure, not only temporal leakage.',
              '- Exa `publishedDate` is itself unreliable (fabricated time-of-day, lax end-date filter), so `date_inconsistent` partly measures Exa metadata quality, not leakage.',
              '- The cost figure is token-based at the published rate ($0.042 per million input tokens, output free), not an invoice.',
              '', '## Files', '',
              f"- `{args.pool / 'leak-flags.jsonl'}`: one line per screened row: `row` (0-based line index in leak-questions.jsonl), `id`, `url`, `query_idx`, `snippet_index`, dates, `p` (three probabilities), `flags`, `wordings`, tokens",
              f"- `{out}`: `calibration.json`, `calibration-jev.jsonl`, `verification-jev.jsonl`, `leak-flags.partial.jsonl` (completion order), `screen-progress.json`, `screen-errors.jsonl`",
              f"- `{args.astra_dir}`: `sample.jsonl`, `astra-labels.jsonl`, `sessions/` (Codex receipts), `summary.json`",
              '- Scripts: `scripts/jev_leak_screen.py`, `scripts/astra_label_snippets.py` in consistency-ft-next-run-20260919 (branch plan/next-run-20260919)']
    (args.pool / 'leak-screen-report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:20]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['calibrate', 'screen', 'report'])
    parser.add_argument('--pool', type=Path, default=POOL)
    parser.add_argument('--out', type=Path, default=None, help='default <pool>/jev-screen')
    parser.add_argument('--astra-dir', type=Path, default=None, help='default <pool>/astra-labels')
    parser.add_argument('--rps', type=float, default=8.0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0, help='screen: only the first N rows (testing)')
    parser.add_argument('--max-consecutive-failures', type=int, default=20)
    args = parser.parse_args()
    args.out = args.out or (args.pool / 'jev-screen')
    args.astra_dir = args.astra_dir or (args.pool / 'astra-labels')
    {'calibrate': calibrate, 'screen': screen, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
