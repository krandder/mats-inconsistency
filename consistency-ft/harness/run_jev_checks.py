#!/usr/bin/env python3
"""Collect independent TypeSafe Noul receipts; no credentials enter artifacts.

Run on the host holding TYPESAFE_API_KEY, then export with export_jev.py.
Transport failures alone may be retried. A served model change fails closed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os
import re
import time
import urllib.request
import urllib.error

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
PROTOCOL = 'typesafe-noul-independent-v1'
INSTRUCTIONS = ('Using your existing knowledge and the event definition and resolution criteria in the state, '
                'is the proposition true? Estimate its likelihood; do not use external tools or browsing.')

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(data):
    return hashlib.sha256(data).hexdigest()

def request_for(text, model):
    return {'model': model, 'state': text,
            'questions': {'proposition': {'type': 'noul', 'instructions': INSTRUCTIONS}}}

def parse_response(data, model):
    if data.get('model') != model:
        raise ValueError('served model changed or missing')
    answers = data.get('answers', {})
    if set(answers) != {'proposition'} or answers['proposition'].get('type') != 'noul':
        raise ValueError('unexpected response shape')
    p = answers['proposition'].get('noul')
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('invalid probability')
    return p

def get_key():
    if os.environ.get('TYPESAFE_API_KEY'):
        return os.environ['TYPESAFE_API_KEY']
    text = (Path.home() / '[removed]').read_text()
    match = re.search(r'(?m)^(?:export )?TYPESAFE_API_KEY=(.+)$', text)
    if not match:
        raise RuntimeError('TYPESAFE_API_KEY unavailable')
    return match.group(1).strip().strip('\"\x27')

def collect(prop, model, output, key):
    ident, text = prop['id'], prop['text']
    if sha(text.encode()) != ident:
        raise ValueError('proposition identity mismatch')
    payload = request_for(text, model)
    previous = sorted(output.glob(ident + '.attempt-*.json'))
    for path in previous:
        row = json.loads(path.read_text())
        if row['request'] != payload or row['protocol_id'] != PROTOCOL:
            raise ValueError('incompatible resume')
        if row['completed']:
            return row
        if row['error_kind'] != 'infrastructure' and row.get('http_status') != 529:
            raise RuntimeError('non-retryable prior receipt')
    for attempt in range(len(previous) + 1, 4):
        started, tick = now(), time.perf_counter()
        raw, status, error = None, None, None
        try:
            request = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                raw, status = response.read().decode(), response.status
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read().decode()
            error = 'infrastructure' if status in (408, 429, 500, 502, 503, 504, 529) else 'http_error'
        except (urllib.error.URLError, TimeoutError):
            error = 'infrastructure'
        data, p = None, None
        if not error:
            try:
                data = json.loads(raw)
                p = parse_response(data, model)
            except (ValueError, KeyError, TypeError):
                error = 'invalid_response'
        row = {'id': ident, 'attempt_id': f'{ident}.attempt-{attempt:03d}',
               'protocol_id': PROTOCOL, 'requested_model': model, 'served_model': data.get('model') if data else None,
               'provider': 'typesafe', 'reasoning': 'not-exposed', 'request': payload,
               'request_sha256': sha(json.dumps(payload, sort_keys=True).encode()),
               'http_status': status, 'response': raw, 'response_sha256': sha(raw.encode()) if raw is not None else None,
               'probability': p, 'usage': data.get('usage') if data else None,
               'started_at': started, 'finished_at': now(), 'elapsed_seconds': time.perf_counter() - tick,
               'completed': error is None, 'error_kind': error}
        path = output / (row['attempt_id'] + '.json')
        with path.open('x') as handle:
            json.dump(row, handle, indent=2)
            handle.write('\n')
        if error is None:
            return row
        if error != 'infrastructure':
            raise RuntimeError(f'{ident}: {error} HTTP {status}; receipt retained')
        time.sleep(attempt * 2)
    raise RuntimeError(f'{ident}: retries exhausted')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--propositions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='jev-1.13.0')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    props = [json.loads(line) for line in args.propositions.read_text().splitlines() if line]
    config = {'protocol_id': PROTOCOL, 'model': args.model, 'endpoint': ENDPOINT,
              'instructions': INSTRUCTIONS, 'propositions_sha256': sha(args.propositions.read_bytes()),
              'question_count': len(props), 'questions_per_request': 1}
    campaign = args.output / 'campaign.json'
    if campaign.exists() and json.loads(campaign.read_text()) != config:
        raise RuntimeError('incompatible campaign resume')
    if not campaign.exists():
        campaign.write_text(json.dumps(config, indent=2) + '\n')
    key = get_key()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(collect, prop, args.model, args.output, key) for prop in props]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 25 == 0 or count == len(props):
                print(f'{count}/{len(props)} completed', flush=True)

if __name__ == '__main__':
    main()
