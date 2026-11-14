#!/usr/bin/env python3
"""Label a stratified sample of leak-screen snippets with GPT-6 Astra (Codex proxy).

Produces calibration labels for scripts/jev_leak_screen.py. Astra is told to judge
only from the snippet text and the dates given, never from what it knows happened;
its labels are a proxy for human judgement, not ground truth. The Codex CLI is
invoked exactly as harness/run_native_checks.py builds it for provider "astra"
(codex-any, ephemeral, read-only sandbox, tools disabled). No credentials are read
or written by this script.

Layout (under <pool>/astra-labels/):
  sample.jsonl        the sampled rows (row = 0-based line index in leak-questions.jsonl)
  sessions/           one receipt per Codex attempt (events, usage, parsed labels)
  astra-labels.jsonl  one line per labelled row: row, id, url, three booleans, note
  summary.json        counts, positives per label, usage totals, failures

Usage:
  python3 scripts/astra_label_snippets.py --sample-only      # write sample.jsonl only
  python3 scripts/astra_label_snippets.py                    # sample (if absent) + label
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

POOL = Path('/home/agent/data/polymarket-evidence-exa-20260922')
MODEL = 'gpt-6-astra'
CODEX = '/home/agent/.local/bin/codex-any'
LABELS = ('states_outcome', 'post_date_event', 'date_inconsistent')
ACTOR = 'consistency-leak-screen-20260922'

RULE = ('You are a careful annotator for a data-leakage audit of a forecasting dataset. '
        'Judge only from the snippet text and the dates given in the prompt. Do not use anything you '
        'know about what actually happened; your knowledge postdates these questions and must not be '
        'used. Return only one JSON object matching the requested schema: no prose, no markdown fences. '
        'Do not use tools, browse, read files, consult memory, or delegate.')

INSTRUCTIONS = """For each snippet below answer three yes/no questions using ONLY the snippet's title, URL, stated \
publication date and text, plus the question and prediction date given. You know what happened after these \
prediction dates; you must NOT use that. If the text itself does not say it, the answer is false.

Definitions:
- states_outcome: the text states or clearly implies the ACTUAL outcome of the forecasting question, i.e. how \
it resolved or has certainly resolved (reports a result, a winner, a completed event, an official decision). \
Forecasts, polls, odds, model predictions, opinions or speculation about the outcome do NOT count.
- post_date_event: the text reports, as having already happened, at least one event that occurred AFTER the \
prediction date. Scheduled, planned or expected future events do NOT count; events on or before the prediction \
date do NOT count.
- date_inconsistent: the stated publication date looks wrong for the content: the text was evidently written \
or updated after the stated date (it reports events or cites dates later than the stated date), or a date in \
the URL or text clearly contradicts the stated date. A web.archive.org capture date far later than the stated \
date counts when the content is a live page (results, live model, tracker). Do not flag a mere same-week \
discrepancy in a URL date.
- note: at most 15 words giving the reason for any true label, or "none".

Output exactly one JSON object, nothing else:
{"labels": [{"row": <row number exactly as given>, "states_outcome": true|false, "post_date_event": true|false, \
"date_inconsistent": true|false, "note": "<short>"}, ...]}
One entry per snippet, in the order given, with the row numbers given.
"""


def parse_ts(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00'))


def now():
    return datetime.now(timezone.utc).isoformat()


def build_sample(pool, n, seed, near_days, near_share):
    """Two passes: stratum membership for every row, then the sampled rows' content."""
    near, far = [], []
    with (pool / 'leak-questions.jsonl').open() as handle:
        for row, line in enumerate(handle):
            record = json.loads(line)
            published = record.get('publishedDate')
            delta = (parse_ts(record['prediction_date']) - parse_ts(published)) if published else None
            (near if delta is not None and delta.days < near_days else far).append(row)
    rng = random.Random(seed)
    n_near = min(len(near), round(n * near_share))
    near_chosen = set(rng.sample(near, n_near))
    chosen = near_chosen | set(rng.sample(far, min(len(far), n - n_near)))
    sample = []
    with (pool / 'leak-questions.jsonl').open() as handle:
        for row, line in enumerate(handle):
            if row in chosen:
                record = json.loads(line)
                record['row'] = row
                record['stratum'] = 'near' if row in near_chosen else 'far'
                sample.append(record)
    rng.shuffle(sample)  # batches mix strata
    return sample, {'rows_total': len(near) + len(far), 'near_total': len(near), 'far_total': len(far),
                    'near_sampled': n_near, 'far_sampled': len(sample) - n_near, 'near_days': near_days,
                    'seed': seed}


def prompt_for(rows):
    parts = [INSTRUCTIONS, f'There are {len(rows)} snippets.\n']
    for record in rows:
        parts.append(f"### Snippet row {record['row']}\n"
                     f"Question: {record['question']}\n"
                     f"Prediction date: {record['prediction_date']}\n"
                     f"Stated publication date: {record.get('publishedDate')}\n"
                     f"Title: {record.get('title')}\n"
                     f"URL: {record.get('url')}\n"
                     f"Text:\n{record['snippet']}\n")
    return '\n'.join(parts)


def codex_command(prompt_out, empty):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(('ANTHROPIC_', 'OPENAI_', 'XAI_', 'ZAI_', 'ZHIPU_', 'CODEX_API_')):
            env.pop(name)
    env.pop('TMUX', None)
    env['PATH'] = '/home/agent/.local/bin:' + env.get('PATH', '')
    env.update(FLEET_ACTOR=ACTOR, FUTARCHY_AGENT_NAME=ACTOR, AI_AGENT_NAME=ACTOR)
    cmd = [CODEX, 'exec', '--ephemeral', '--json', '--skip-git-repo-check', '--sandbox', 'read-only',
           '-C', str(empty), '--model', MODEL, '-c', 'model_reasoning_effort="medium"',
           '-c', 'service_tier="default"', '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"',
           '-c', 'features.multi_agent_v2=false', '-c', 'features.shell_tool=false',
           '-c', 'features.exec_tool=false', '-c', 'developer_instructions=' + json.dumps(RULE),
           '-o', str(prompt_out), '-']
    return cmd, env


def decode_events(stdout):
    events = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            events.append(item)
    messages = [e['item'].get('text', '') for e in events
                if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'agent_message']
    usage = next((e.get('usage') for e in reversed(events) if e.get('type') == 'turn.completed'), None)
    thread = next((e.get('thread_id') for e in events if e.get('type') == 'thread.started'), None)
    tools = any(e.get('item', {}).get('type') in {'command_execution', 'mcp_tool_call', 'web_search', 'file_change'}
                for e in events)
    failed = any(e.get('type') == 'turn.failed' for e in events)
    return messages[-1] if messages else '', usage, thread, tools, failed


def parse_labels(text, expected_rows):
    body = text.strip()
    body = re.sub(r'^```(?:json)?\s*|\s*```$', '', body).strip()
    if not body.startswith('{'):
        start, end = body.find('{'), body.rfind('}')
        if start < 0 or end < 0:
            raise ValueError('no JSON object')
        body = body[start:end + 1]
    data = json.loads(body)
    labels = data['labels'] if isinstance(data, dict) else None
    if not isinstance(labels, list):
        raise ValueError('labels missing')
    out = {}
    for entry in labels:
        row = int(entry['row'])
        if row not in expected_rows or row in out:
            raise ValueError(f'unexpected or duplicate row {row}')
        clean = {'row': row}
        for name in LABELS:
            value = entry[name]
            if not isinstance(value, bool):
                raise ValueError(f'non-boolean {name} for row {row}')
            clean[name] = value
        clean['note'] = str(entry.get('note', ''))[:200]
        out[row] = clean
    if set(out) != set(expected_rows):
        raise ValueError(f'rows returned {sorted(out)} != expected {sorted(expected_rows)}')
    return [out[row] for row in expected_rows]


def run_batch(index, rows, sessions, empty, timeout, attempts=2):
    expected = [r['row'] for r in rows]
    for attempt in range(1, attempts + 1):
        receipt = sessions / f'batch-{index:03d}.attempt-{attempt}.json'
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old.get('labels'):
                return old
            continue  # a failed earlier attempt: try the next attempt number
        final = sessions / f'batch-{index:03d}.attempt-{attempt}.final.txt'
        final.unlink(missing_ok=True)
        cmd, env = codex_command(final, empty)
        prompt = prompt_for(rows)
        started, tick = now(), time.monotonic()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, env=env, cwd=empty,
                                  timeout=timeout)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = (exc.stdout or ''), (exc.stderr or ''), 'timeout'
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors='replace')
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors='replace')
        text, usage, thread, tools, failed = decode_events(stdout)
        if not text and final.exists():
            text = final.read_text()
        record = {'batch': index, 'attempt': attempt, 'rows': expected, 'model': MODEL,
                  'reasoning_effort': 'medium', 'started_at': started, 'finished_at': now(),
                  'elapsed_seconds': round(time.monotonic() - tick, 2), 'exit_code': code,
                  'thread_id': thread, 'usage': usage, 'tools_used': tools, 'turn_failed': failed,
                  'response': text, 'stderr_tail': stderr[-2000:], 'labels': None, 'error': None}
        try:
            if code != 0:
                raise ValueError(f'codex exited {code}')
            if tools:
                raise ValueError('tool use in a no-tool labelling session')
            record['labels'] = parse_labels(text, expected)
        except (ValueError, KeyError, TypeError) as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'[:500]
        receipt.write_text(json.dumps(record, indent=1) + '\n')
        print(f"batch {index:03d} attempt {attempt}: {'ok' if record['labels'] else record['error']} "
              f"({record['elapsed_seconds']}s, usage={usage})", flush=True)
        if record['labels']:
            return record
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', type=Path, default=POOL)
    parser.add_argument('--out', type=Path, default=None, help='default <pool>/astra-labels')
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--near-days', type=int, default=7)
    parser.add_argument('--near-share', type=float, default=2 / 3,
                        help='share of the sample drawn from rows published within --near-days of the prediction date')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=900, help='seconds per Codex session')
    parser.add_argument('--sample-only', action='store_true')
    args = parser.parse_args()
    out = args.out or (args.pool / 'astra-labels')
    sessions, empty = out / 'sessions', out / 'empty-working-directory'
    sessions.mkdir(parents=True, exist_ok=True)
    empty.mkdir(parents=True, exist_ok=True)
    sample_path = out / 'sample.jsonl'
    if sample_path.exists():
        sample = [json.loads(line) for line in sample_path.read_text().splitlines() if line]
        strata = json.loads((out / 'sample-strata.json').read_text())
    else:
        sample, strata = build_sample(args.pool, args.n, args.seed, args.near_days, args.near_share)
        with sample_path.open('w') as handle:
            for record in sample:
                handle.write(json.dumps(record) + '\n')
        (out / 'sample-strata.json').write_text(json.dumps(strata, indent=1) + '\n')
    print(f'sample: {len(sample)} rows, strata {strata}', flush=True)
    if args.sample_only:
        return
    batches = [sample[i:i + args.batch] for i in range(0, len(sample), args.batch)]
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, i, rows, sessions, empty, args.timeout): i for i, rows in enumerate(batches)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    labelled, failed_batches, usage = [], [], {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0,
                                               'reasoning_output_tokens': 0, 'sessions': 0}
    for i, rows in enumerate(batches):
        by_row = {r['row']: r for r in rows}
        for path in sorted(sessions.glob(f'batch-{i:03d}.attempt-*.json')):
            rec = json.loads(path.read_text())
            if rec.get('usage'):
                usage['sessions'] += 1
                for name in list(usage):
                    if name != 'sessions':
                        usage[name] += int(rec['usage'].get(name) or 0)
        record = results[i]
        if not record.get('labels'):
            failed_batches.append({'batch': i, 'error': record.get('error'), 'rows': record['rows']})
            continue
        for label in record['labels']:
            source = by_row[label['row']]
            labelled.append({'row': label['row'], 'id': source['id'], 'url': source['url'],
                             'stratum': source['stratum'], **{k: label[k] for k in LABELS},
                             'note': label['note'], 'batch': i, 'attempt': record['attempt'],
                             'thread_id': record['thread_id']})
    labelled.sort(key=lambda r: r['row'])
    with (out / 'astra-labels.jsonl').open('w') as handle:
        for record in labelled:
            handle.write(json.dumps(record) + '\n')
    summary = {'model': MODEL, 'reasoning_effort': 'medium', 'sampled': len(sample), 'labelled': len(labelled),
               'batches': len(batches), 'batch_size': args.batch, 'failed_batches': failed_batches,
               'positives': {k: sum(r[k] for r in labelled) for k in LABELS},
               'positives_by_stratum': {s: {k: sum(r[k] for r in labelled if r['stratum'] == s) for k in LABELS}
                                        for s in ('near', 'far')},
               'strata': strata, 'usage': usage, 'finished_at': now()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1), flush=True)
    if failed_batches:
        sys.exit(1)


if __name__ == '__main__':
    main()
