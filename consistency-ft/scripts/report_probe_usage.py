#!/usr/bin/env python3
"""Summarize private native-probe receipts and measured subscription windows.

Default operation is offline. --capture performs the existing read-only quota
checks plus a session-filtered proxy usage lookup; it never invokes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.run_probes import build_report, read_jsonl
from harness.run_native_probes import COLLECTOR, load_campaign
from scripts.measure_subscription_usage import capture, remote_python
from scripts.report_subscription_usage import TOKEN_KEYS, iso, quota_limits


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def campaign_records(directory):
    result, sessions, campaigns = [], {}, set()
    for path in sorted(Path(directory).iterdir()):
        if not path.is_dir() or not (path / 'manifest.json').exists():
            continue
        manifest, native_protocol = load_campaign(path)
        if manifest['campaign_id'] in campaigns:
            raise ValueError('A frozen campaign occurs in more than one configuration directory')
        campaigns.add(manifest['campaign_id'])
        if manifest['provider'] not in {'astra', 'luna', 'grok', 'glm'}:
            raise ValueError('Expected a native subscription provider')
        report = build_report(path)
        if report['synthetic']:
            raise ValueError('Synthetic probes cannot enter subscription usage accounting')
        ledger = path / 'receipts.jsonl'
        saved = read_jsonl(ledger) if ledger.exists() else []
        receipts = [row['receipt'] for row in saved]
        totals, available = Counter(), 0
        for receipt in receipts:
            if (receipt.get('collector') != COLLECTOR
                    or receipt.get('native_protocol_fingerprint') != native_protocol['fingerprint']
                    or receipt.get('system_instruction_sha256') != native_protocol['system_instruction_sha256']
                    or receipt.get('requested_model') != manifest['model']
                    or receipt['provenance'].get('transport') != 'subscription-native-cli'
                    or receipt['provenance'].get('adapter') != COLLECTOR):
                raise ValueError('Receipt lacks the frozen native collector/protocol identity')
            if receipt['usage'] is not None:
                usage = receipt['usage']
                if (set(usage) != set(TOKEN_KEYS)
                        or usage['total_tokens'] != usage['input_tokens'] + usage['output_tokens']
                        or usage['cached_input_tokens'] > usage['input_tokens']
                        or usage['reasoning_output_tokens'] > usage['output_tokens']):
                    raise ValueError('Normalized usage totals or subset counters are inconsistent')
                totals.update(usage)
                available += 1
            sid = receipt.get('provenance', {}).get('session_id')
            if sid and manifest['provider'] in {'astra', 'luna'}:
                if sid in sessions:
                    raise ValueError('A native session occurs in more than one probe attempt')
                sessions[sid] = path.name
        elapsed = [r['elapsed_seconds'] for r in receipts if isinstance(r.get('elapsed_seconds'), (int, float))]
        result.append({'configuration': path.name, 'provider': manifest['provider'],
                       'model': manifest['model'], 'reasoning': manifest['reasoning'],
                       'campaign_id': manifest['campaign_id'],
                       'native_protocol_fingerprint': native_protocol['fingerprint'],
                       'target_probes': report['catalog_count'],
                       'observed_probes': report['observed_probes'],
                       'completed_observations': report['completed_observations'],
                       'attempt_receipts': len(receipts), 'attempts_with_usage': available,
                       'attempts_without_usage': len(receipts) - available,
                       'launch_reservations': report['launch_reservations'],
                       'launches_without_final_receipt': report['launches_without_final_receipt'],
                       'tokens': {k: totals[k] for k in TOKEN_KEYS},
                       'median_elapsed_seconds': statistics.median(elapsed) if elapsed else None,
                       'elapsed_observations': len(elapsed),
                       'ledger_sha256': hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None})
    if not result:
        raise ValueError('No prepared native probe treatments found')
    return result, sessions


def quota_changes(directory, accounts, label='after'):
    """Only compare matched windows; never equate shared deltas with marginal use."""
    directory = Path(directory)
    changes = []
    before, after = (read(directory / f'quota-astra-{s}.json') for s in ('before', label))
    if before and after:
        left = {p['profile']: p for p in before['profiles']}
        right = {p['profile']: p for p in after['profiles']}
        for index, account in enumerate(accounts, 1):
            name = account.removesuffix('-codex')
            a, b = left.get(name, {}), right.get(name, {})
            wa, wb = ({w['window']: w for w in p.get('windows', [])} for p in (a, b))
            for window in sorted(set(wa) | set(wb)):
                x, y = wa.get(window, {}), wb.get(window, {})
                av, bv = x.get('used_percent'), y.get('used_percent')
                known_reset = bool(x and y and x.get('reset_after_seconds') is not None and y.get('reset_after_seconds') is not None)
                same = known_reset and abs((a.get('ts', before['captured_at']) + x['reset_after_seconds']) -
                                           (b.get('ts', after['captured_at']) + y['reset_after_seconds'])) <= 120
                changes.append({'provider': 'OpenAI', 'subscription': index, 'window': window,
                                'before_percent': av, 'after_percent': bv,
                                'same_window_established': bool(same),
                                'delta_percentage_points': bv - av if av is not None and bv is not None and same else None})
    before, after = (read(directory / f'quota-glm-{s}.json') for s in ('before', label))
    if before and after:
        left, right = ({(w['type'], w['unit'], w['number']): w for w in quota_limits(p)} for p in (before, after))
        for window in sorted(set(left) | set(right)):
            a, b = left.get(window, {}), right.get(window, {})
            av, bv = a.get('percentage'), b.get('percentage')
            # No reset ID is returned for some short windows. Preserve the
            # readings but do not infer that the windows must be identical.
            same = a.get('nextResetTime') is not None and a.get('nextResetTime') == b.get('nextResetTime')
            changes.append({'provider': 'GLM coding plan', 'window': list(window),
                            'before_percent': av, 'after_percent': bv,
                            'same_window_established': same,
                            'delta_percentage_points': bv - av if av is not None and bv is not None and same else None})
    before, after = (read(directory / f'quota-grok-{s}.json') for s in ('before', label))
    if before and after:
        same = bool(before.get('reset_display')) and before.get('reset_display') == after.get('reset_display')
        changes.append({'provider': 'Grok SuperGrok Heavy', 'window': 'weekly',
                        'before_percent': before['used_percent'], 'after_percent': after['used_percent'],
                        'same_window_established': same,
                        'delta_percentage_points': after['used_percent'] - before['used_percent'] if same else None})
    return changes


def capture_probe_usage(directory, label, grok_percent=None, grok_reset_display=None):
    _, sessions = campaign_records(directory)
    capture(directory, label, grok_percent, grok_reset_display)
    # The forecast-specific capture helper cannot discover factual probe ledgers.
    # Replace its proxy lookup with the exact sessions from these validated ledgers.
    code = '''
import sys,json,glob
ids=set(json.load(sys.stdin))
for path in sorted(glob.glob('/var/lib/primeproxy/unified-claude/usage/usage*.jsonl')):
 for line in open(path):
  try: row=json.loads(line)
  except ValueError: continue
  if row.get('session_id') in ids: print(json.dumps(row))
'''
    raw = remote_python('prime-any-proxy-1', code, json.dumps(sorted(sessions)), sudo=True)
    path = Path(directory) / f'proxy-usage-{label}.jsonl'
    path.write_text(raw)
    path.chmod(0o600)


def generate(directory, label='after'):
    directory = Path(directory)
    configurations, sessions = campaign_records(directory)
    path = directory / f'proxy-usage-{label}.jsonl'
    rows = read_jsonl(path) if path.exists() else []
    matched = [r for r in rows if r.get('session_id') in sessions]
    accounts = sorted({r['account'] for r in matched})
    account_usage = []
    for index, account in enumerate(accounts, 1):
        records = [r for r in matched if r['account'] == account]
        totals = Counter()
        for row in records:
            totals.update(row.get('tokens') or {})
        account_usage.append({'subscription': index, 'proxy_requests': len(records),
                              'requests_by_configuration': dict(Counter(sessions[r['session_id']] for r in records)),
                              'input_tokens': totals['input'], 'output_tokens': totals['output']})
    totals = Counter()
    for row in configurations:
        totals.update(row['tokens'])
    snapshots = {}
    for provider in ('astra', 'glm', 'grok'):
        snapshots[provider] = {}
        for state in ('before', label):
            path = directory / f'quota-{provider}-{state}.json'
            value = read(path)
            snapshots[provider][state] = ({'captured_at': iso(value['captured_at']),
                                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} if value else None)
    return {'schema_version': 1, 'campaign': directory.name,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'billing': 'Existing native subscriptions; no separately billed inference API calls.',
            'configurations': configurations, 'total_tokens': {k: totals[k] for k in TOKEN_KEYS},
            'attempt_receipts': sum(c['attempt_receipts'] for c in configurations),
            'launches_without_final_receipt': sum(c['launches_without_final_receipt'] for c in configurations),
            'proxy': {'native_sessions': len(sessions),
                      'matched_sessions': len({r['session_id'] for r in matched}),
                      'matched_requests': len(matched), 'accounts': account_usage},
            'quota_snapshots': snapshots, 'quota_changes': quota_changes(directory, accounts, label),
            'limitations': [
                'Native receipt tokens and proxy tokens describe overlapping work and must not be added together.',
                'Failed attempts are included. Missing usage or unsaved launch outcomes are unknown consumption, not zero.',
                'The sum of observed tokens is a lower bound when any attempt usage is unavailable.',
                'Subscription readings are shared-account measurements, rounded by providers and affected by concurrent fleet work.',
                'Window resets or missing reset identities prevent a defensible percentage-point subtraction.',
                'A displayed zero change does not mean zero consumption. No exact token-to-subscription-fraction conversion is available.',
                'Agent development and source-review work is outside these probe-session token totals.'
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--label', default='after')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--grok-percent', type=float)
    parser.add_argument('--grok-reset-display')
    args = parser.parse_args()
    if args.capture:
        capture_probe_usage(args.run_dir, args.label, args.grok_percent, args.grok_reset_display)
    report = generate(args.run_dir, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'attempt_receipts': report['attempt_receipts'], 'tokens': report['total_tokens'],
                      'matched_proxy_requests': report['proxy']['matched_requests']}))


if __name__ == '__main__':
    main()
