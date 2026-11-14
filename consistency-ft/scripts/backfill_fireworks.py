#!/usr/bin/env python3
"""Plan or resume the legacy Fireworks backfill, with per-request evidence.

No network requests unless --execute is supplied. --mock exercises the same
checkpoint/output paths offline. This script never changes billing or deploys.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (BASE_URL, CHECK_FIELDS, FireworksClient, MockClient, _api_key,
                    load_tuples, proposition_text, violation)
from elicit.run_elicit import base_propositions

TARGETS = ['minimax-m2p7', 'inkling', 'muse-glimmer-30b']
PROBE_MODELS = ['gpt-oss-20b', 'gpt-oss-120b', 'qwen3p7-plus', 'qwen3p8-max',
                'deepseek-v4-flash', 'deepseek-v4-pro', 'glm-5p2', 'kimi-k3',
                'minimax-m3', 'nemotron-lightning-30b', 'muse-glimmer-30b',
                'qwen3p8-2p4t-a95b', 'kimi-k2p6', 'kimi-k2p7-code',
                'minimax-m2p7', 'nemotron-3-ultra', 'inkling']


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())


class StopRun(RuntimeError):
    pass


class Budget:
    """Conservative reservations persist across resumes, including failed calls."""
    def __init__(self, path, ceiling, max_calls):
        self.path, self.ceiling, self.max_calls = path, ceiling, max_calls
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        self.reserved = sum(row['reserved_usd'] for row in rows)
        self.calls = len(rows)

    def reserve(self, model, prompt, max_tokens, prices):
        # UTF-8 bytes plus ample chat-template overhead bounds ordinary tokenizers;
        # reserve completion cap even when generation is short or fails.
        input_bound = len(prompt.encode()) + 1024
        charge = (input_bound * prices['input'] + max_tokens * prices['output']) / 1e6
        if self.calls >= self.max_calls or self.reserved + charge > self.ceiling:
            raise StopRun('Local call/spend allowance exhausted; no further requests sent')
        append(self.path, {'at': stamp(), 'model': model, 'reserved_usd': charge,
                           'input_token_bound': input_bound, 'completion_token_cap': max_tokens})
        self.reserved += charge
        self.calls += 1


class RecordedClient(FireworksClient):
    def __init__(self, model, args, budget, prices, folder):
        self.model = model
        self.key = _api_key() if not args.mock else None
        self.args, self.budget, self.prices, self.folder = args, budget, prices, folder
        self.receipts = folder / 'receipts.jsonl'
        self.cache = {}
        self.rejected_efforts = set()
        if self.receipts.exists():
            for line in self.receipts.read_text().splitlines():
                row = json.loads(line)
                if row['http_status'] == 200:
                    self.cache[row['request_hash']] = row['response']
                elif row['http_status'] == 400 and row['model'] == model:
                    message = json.dumps(row['response']).lower()
                    if 'reasoning_effort' in message or 'reasoning effort' in message:
                        self.rejected_efforts.add(row['effective_reasoning'])
        if not args.mock and not self.key:
            raise StopRun('Missing existing FIREWORKS_API_KEY / ~/.config/fireworks/env')

    def _chat(self, prompt, **options):
        for effort in dict.fromkeys([self.args.reasoning, self.args.fallback]):
            if effort is None:
                continue
            if effort in self.rejected_efforts:
                continue
            payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0, **options, 'max_tokens': self.args.max_tokens,
                       'reasoning_effort': effort}
            key = digest(payload)
            if key in self.cache:
                return self.cache[key]
            self.budget.reserve(self.model, prompt, self.args.max_tokens, self.prices)
            if self.args.mock:
                lp = bool(options.get('logprobs'))
                content = 'True' if lp else str(MockClient._probability('verbal', prompt))
                data = {'model': self.model, 'choices': [{'finish_reason': 'stop',
                    'message': {'content': content}, 'logprobs': {'content': [
                        {'token': 'True', 'logprob': -0.5, 'top_logprobs': [
                            {'token': 'True', 'logprob': -0.5},
                            {'token': 'False', 'logprob': -1.0}]}]}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 5}}
                status = 200
            else:
                try:
                    response = requests.post(BASE_URL + '/chat/completions',
                        headers={'Authorization': 'Bearer ' + self.key}, json=payload, timeout=90)
                except requests.RequestException as exc:
                    append(self.receipts, {'at': stamp(), 'request_hash': key,
                        'model': self.model, 'effective_reasoning': effort, 'http_status': 0,
                        'error': type(exc).__name__, 'response': {}})
                    raise StopRun('Transport failure; reservation retained, inspect before resuming') from exc
                status = response.status_code
                try:
                    data = response.json()
                except ValueError:
                    data = {'error': 'Non-JSON HTTP response'}
                # Never persist a credential if a gateway echoes it in an error.
                data = json.loads(json.dumps(data).replace(self.key, '[REDACTED]'))
            append(self.receipts, {'at': stamp(), 'request_hash': key, 'model': self.model,
                'effective_reasoning': effort, 'requested_reasoning': self.args.reasoning,
                'http_status': status, 'request': payload, 'response': data,
                'pricing_per_million': self.prices, 'mock': self.args.mock})
            if status == 200:
                self.cache[key] = data
                return data
            message = json.dumps(data).lower()
            if (status == 400 and effort == self.args.reasoning and self.args.fallback
                    and ('reasoning_effort' in message or 'reasoning effort' in message)):
                self.rejected_efforts.add(effort)
                continue
            raise StopRun(f'HTTP {status}; inspect receipts before resuming (no automatic retry)')
        raise StopRun('Reasoning configuration rejected')


def score_checks(client, tuples, path):
    texts = sorted({proposition_text(q) for row in tuples for q in row['questions'].values()})
    probabilities, errors = {}, []
    for text in texts:
        try:
            probabilities[text] = client.verbal_probability(text)
        except ValueError as exc:
            errors.append({'proposition_sha256': digest(text), 'error': str(exc)[:300]})
    results = []
    for row in tuples:
        fields = CHECK_FIELDS[row['check_type']]
        if any(proposition_text(row['questions'][key]) not in probabilities for key in fields):
            continue
        probs = {key: probabilities[proposition_text(row['questions'][key])] for key in fields}
        results.append({'id': row['id'], 'check_type': row['check_type'], 'probabilities': probs,
                        'violation': violation(row['check_type'], probs), 'model': client.model})
    write_results(path, results)
    return {'stage': 'checks', 'model': client.model, 'scored': len(results),
            'expected': len(tuples), 'parse_errors': errors, 'output': str(path)}


def write_results(path, rows):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    tmp.replace(path)


def usage_summary(root):
    summary = {}
    receipts = root / 'receipts.jsonl'
    for line in receipts.read_text().splitlines() if receipts.exists() else []:
        row = json.loads(line)
        totals = summary.setdefault(row['model'], {'http_attempts': 0, 'successful_responses': 0,
            'prompt_tokens': 0, 'completion_tokens': 0, 'reasoning_tokens_reported': 0,
            'missing_usage_responses': 0, 'uncached_price_estimate_usd': 0})
        totals['http_attempts'] += 1
        if row['http_status'] != 200:
            continue
        totals['successful_responses'] += 1
        usage = row['response'].get('usage') or {}
        if 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            totals['missing_usage_responses'] += 1
        prompt = int(usage.get('prompt_tokens', 0))
        completion = int(usage.get('completion_tokens', 0))
        totals['prompt_tokens'] += prompt
        totals['completion_tokens'] += completion
        totals['reasoning_tokens_reported'] += int((usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0))
        prices = row['pricing_per_million']
        totals['uncached_price_estimate_usd'] += (prompt * prices['input'] + completion * prices['output']) / 1e6
    (root / 'usage-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def score_elicit(client, propositions, path):
    rows, errors = [], []
    for question in propositions:
        text = proposition_text(question)
        try:
            verbal = client.verbal_probability(text)
            logprob = client.true_false_probability(text)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append({'id': question.get('id'), 'error': str(exc)[:300]})
            continue
        rows.append({'id': question.get('id'), 'title': question['title'], 'model': client.model,
                     'verbalized_probability': verbal, 'logprob_probability': logprob,
                     'abs_diff': abs(verbal - logprob)})
    write_results(path, rows)
    return {'stage': 'elicit', 'model': client.model, 'scored': len(rows),
            'expected': len(propositions), 'parse_errors': errors, 'output': str(path)}


def probe(client, text):
    from common import VERBAL_PROMPT
    data = client._chat(VERBAL_PROMPT.format(proposition=text), max_tokens=client.args.max_tokens)
    choice = data.get('choices', [{}])[0]
    message = choice.get('message') or {}
    return {'stage': 'probe', 'model': client.model, 'returned_model': data.get('model'),
            'usage': data.get('usage'), 'content': message.get('content'),
            'visible_reasoning_chars': len(message.get('reasoning_content') or message.get('reasoning') or ''),
            'finish_reason': choice.get('finish_reason'),
            'interpretation': 'Accepted settings and visible reasoning only; no proof hidden reasoning is disabled.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['checks', 'elicit', 'probe', 'all'], default='all')
    p.add_argument('--models', nargs='+')
    p.add_argument('--tuples', default='data/tuples')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--elicit-limit', type=int, default=30)
    p.add_argument('--reasoning', default='none')
    p.add_argument('--fallback', help='Explicit fallback after HTTP 400 rejects requested reasoning, e.g. low')
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--output-dir', default='results/fireworks-backfill')
    p.add_argument('--run-id', default='backfill-20260912')
    p.add_argument('--prices', help='JSON mapping model slug to verified input/output USD per million')
    p.add_argument('--budget-usd', type=float)
    p.add_argument('--max-calls', type=int)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--mock', action='store_true')
    args = p.parse_args()
    if args.limit <= 0 or args.elicit_limit <= 0 or args.max_tokens <= 0:
        p.error('limits and max-tokens must be positive')
    models = args.models or (PROBE_MODELS if args.stage == 'probe' else TARGETS)
    tuples = load_tuples(args.tuples, args.limit)
    propositions = base_propositions(args.tuples, args.elicit_limit)
    texts = {proposition_text(q) for row in tuples for q in row['questions'].values()}
    elicit_texts = {proposition_text(q) for q in propositions}
    plan = {'models': models, 'tuples_per_model': len(tuples), 'check_prompts_per_model': len(texts),
            'elicit_prompts_per_model_upper_bound': 2 * len(propositions),
            'probe_prompts_per_model': 1, 'reasoning': args.reasoning, 'fallback': args.fallback,
            'all_stages_unique_prompts_per_model': len(texts | elicit_texts) + len(elicit_texts),
            'completion_token_cap': args.max_tokens, 'stage': args.stage,
            'network_enabled': args.execute and not args.mock,
            'muse_policy': 'Fresh complete cohort; never mix August partial scores with new settings.'}
    print(json.dumps(plan, indent=2))
    if not args.execute and not args.mock:
        return
    if not args.max_calls or args.max_calls <= 0 or not args.budget_usd or not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
        p.error('execution requires positive --max-calls and --budget-usd')
    prices = json.loads(Path(args.prices).read_text()) if args.prices else {}
    for model in models:
        slug = model.split('/')[-1]
        if args.mock:
            prices[slug] = {'input': 0, 'output': 0, 'source': 'offline mock'}
        price = prices.get(slug, {})
        if (not all(k in price for k in ('input', 'output', 'source'))
                or any(not isinstance(price[k], (float, int)) or not math.isfinite(price[k]) or price[k] < 0 for k in ('input', 'output'))):
            p.error('supply verified nonnegative input/output prices and source for ' + slug)
    root = Path(args.output_dir) / args.run_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = {**plan, 'mock': args.mock, 'tuples_sha256': digest(tuples),
                  'elicit_sha256': digest(propositions), 'prices': prices}
        manifest = root / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            p.error('run configuration changed; use a new --run-id')
        manifest.write_text(json.dumps(config, indent=2) + '\n')
        budget = Budget(root / 'reservations.jsonl', args.budget_usd, args.max_calls)
        summaries = []
        try:
            for model in models:
                slug = model.split('/')[-1]
                full_model = model if '/' in model else 'accounts/fireworks/models/' + slug
                if args.mock:
                    full_model = 'mock/' + full_model
                client = RecordedClient(full_model, args, budget, prices[slug], root)
                stages = ['probe', 'checks', 'elicit'] if args.stage == 'all' else [args.stage]
                for stage in stages:
                    if stage == 'probe':
                        result = probe(client, sorted(texts)[0])
                    elif stage == 'checks':
                        result = score_checks(client, tuples, root / f'checks_{slug}.jsonl')
                    else:
                        result = score_elicit(client, propositions, root / f'elicit_{slug}.jsonl')
                    summaries.append(result)
                    print(json.dumps(result))
                    (root / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        except StopRun as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2)
        finally:
            usage_summary(root)
        print(json.dumps({'calls_reserved': budget.calls, 'usd_reserved_upper_estimate': budget.reserved,
                          'complete_coverage': all(s.get('scored') == s.get('expected') for s in summaries)}))
        if any(s.get('scored') != s.get('expected') for s in summaries):
            raise SystemExit(3)


if __name__ == '__main__':
    main()
