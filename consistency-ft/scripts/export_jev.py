#!/usr/bin/env python3
"""Validate immutable Jev API receipts and add one distinct probability channel."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import violation
from harness.run_jev_checks import PROTOCOL, INSTRUCTIONS, ENDPOINT, request_for, parse_response
from scripts.export_benchmark import load_snapshot, encoded, digest


def export(source, destination):
    manifest, propositions, tuples, runs = load_snapshot(destination)
    config = json.loads((source / 'campaign.json').read_text())
    model = config['model']
    if config != {'protocol_id': PROTOCOL, 'model': model, 'endpoint': ENDPOINT,
                  'instructions': INSTRUCTIONS, 'propositions_sha256': manifest['propositions_sha256'],
                  'question_count': len(propositions), 'questions_per_request': 1}:
        raise ValueError('campaign configuration or source catalog changed')
    texts = {p['id']: p['text'] for p in propositions}
    rows, current = [], {}
    for path in sorted(source.glob('*.attempt-*.json')):
        row = json.loads(path.read_text())
        ident = row['id']
        if row['attempt_id'] != path.stem or ident not in texts:
            raise ValueError('receipt identity mismatch')
        request = request_for(texts[ident], model)
        if (row['request'] != request or row['protocol_id'] != PROTOCOL
                or row['requested_model'] != model
                or row['request_sha256'] != digest(json.dumps(request, sort_keys=True).encode())):
            raise ValueError('receipt request mismatch')
        raw = row['response']
        if row['response_sha256'] != (digest(raw.encode()) if raw is not None else None):
            raise ValueError('raw response mismatch')
        if row['completed']:
            if row['error_kind'] or row['http_status'] != 200:
                raise ValueError('invalid completion classification')
            parsed = json.loads(raw)
            if (parse_response(parsed, model) != row['probability']
                    or parsed.get('usage') != row['usage'] or row['served_model'] != model):
                raise ValueError('receipt response mismatch')
            if ident in current:
                raise ValueError('duplicate completed observation: no cherry-picking')
            current[ident] = row
        elif row['probability'] is not None:
            raise ValueError('failed attempt contains scored probability')
        if row.get('http_status') == 529 and row['error_kind'] == 'http_error':
            row = {**row, 'original_error_kind': 'http_error', 'error_kind': 'infrastructure'}
        rows.append(row)
    if set(current) != set(texts):
        raise ValueError(f'incomplete collection: {len(current)}/{len(texts)}')
    identity = f'typesafe/{model}@noul'
    campaign = 'typesafe-20260917-original'
    run_id = f'{campaign}/{model}-noul'
    receipts = [{**row, 'campaign': campaign, 'is_current': row is current.get(row['id']),
                 'raw_receipt_sha256': digest((source / (row['attempt_id'] + '.json')).read_bytes())} for row in rows]
    scores = []
    for group in tuples:
        probs = {role: current[ident]['probability'] for role, ident in group['propositions'].items()}
        scores.append({'id': group['id'], 'check_type': group['check_type'], 'probabilities': probs,
                       'violation': violation(group['check_type'], probs), 'model': identity,
                       'reasoning': 'not-exposed', 'run_id': run_id, 'transport': 'typesafe-systemone-api'})
    filename = campaign + '.jsonl'
    receipts_file = campaign + '.receipts.jsonl'
    score_bytes = ''.join(encoded(row) + '\n' for row in scores).encode()
    receipt_bytes = ''.join(encoded(row) + '\n' for row in receipts).encode()
    entry = {'run_id': run_id, 'file': filename, 'source_file': filename,
             'model': identity, 'requested_model': model, 'reasoning': 'not-exposed', 'provider': 'typesafe',
             'kind': 'checks', 'transport': 'typesafe-systemone-api', 'campaign': campaign,
             'protocol_id': PROTOCOL, 'scope_id': None, 'recorded_at': max(r['finished_at'] for r in rows),
             'timestamp_source': 'api_receipt_finished_at', 'status': 'completed',
             'sha256': digest(score_bytes),
             'receipts_file': receipts_file, 'receipts_sha256': digest(receipt_bytes),
             'target_tuples': len(tuples), 'target_propositions': len(propositions),
             'coverage': {'scored_tuples': len(scores), 'valid_propositions': len(current),
                          'observed_propositions': len(current), 'completed_propositions': len(current),
                          'missing_probabilities': 0, 'attempts': len(rows), 'receipt_records': len(rows),
                          'imported_observations': 0, 'errors': dict(Counter(r['error_kind'] for r in rows if r['error_kind']))},
             'protocol': {**config, 'latest_alias_resolved_from': 'jev-latest',
                          'resolution_evidence': '2026-09-17 exploratory API receipts: all returned jev-1.13.0',
                          'tuned': False, 'abstention_supported': False, 'reasoning_control': 'not exposed',
                          'readout': 'direct Noul yes probability; no verbal answer or token logprobs'}}
    existing = next((r for r in manifest['runs'] if r['run_id'] == run_id), None)
    if existing and existing != entry:
        raise ValueError('published run is immutable; use a distinct campaign')
    (destination / filename).write_bytes(score_bytes)
    (destination / receipts_file).write_bytes(receipt_bytes)
    manifest['runs'] = sorted([r for r in manifest['runs'] if r['run_id'] != run_id] + [entry], key=lambda r: r['run_id'])
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    totals = {key: sum((r.get('usage') or {}).get(key, 0) for r in rows) for key in ['input_tokens', 'output_tokens']}
    summary = {'run_id': run_id, 'model': identity, 'protocol_id': PROTOCOL, 'questions': len(current),
               'groups': len(scores), 'attempts': len(rows), 'failed_attempts': len(rows) - len(current),
               'mean_all_groups': statistics.mean(s['violation'] for s in scores),
               'median_elapsed_seconds': statistics.median(r['elapsed_seconds'] for r in current.values()),
               'usage': totals, 'estimated_usd': totals['input_tokens'] * 0.042 / 1_000_000,
               'cost_basis': 'Published $0.042 per million input tokens, output free; reported-token estimate, not invoice',
               'attempts_without_usage': sum(r.get('usage') is None for r in rows),
               'first_started_at': min(r['started_at'] for r in rows), 'last_finished_at': entry['recorded_at']}
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, default=ROOT / 'data/benchmark-runs')
    args = parser.parse_args()
    export(args.receipts, args.snapshots)
