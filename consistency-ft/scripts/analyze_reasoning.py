#!/usr/bin/env python3
"""Analyze existing Luna effort runs on equal cohorts; performs no inference."""
from __future__ import annotations
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_benchmark import load_snapshot
from scripts.report_subscription_usage import tokens

EFFORTS = ('none', 'low', 'medium', 'high', 'xhigh', 'max')
CAMPAIGNS = ('native-20260912', 'native-20260913-scoped-final')


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def pair_summary(left, right, ids, by_type):
    """Negative delta means the right setting has lower violation on identical IDs."""
    delta = {ident: right[ident]['violation'] - left[ident]['violation'] for ident in sorted(ids)}
    return {'n': len(delta), 'mean_delta': mean(delta.values()),
            'improved': sum(v < -1e-12 for v in delta.values()),
            'worsened': sum(v > 1e-12 for v in delta.values()),
            'unchanged': sum(abs(v) <= 1e-12 for v in delta.values()),
            'per_check': {kind: {'n': sum(by_type[i] == kind for i in delta),
                                 'mean_delta': mean(v for i, v in delta.items() if by_type[i] == kind)}
                          for kind in sorted(set(by_type[i] for i in delta))}}


def analyze_campaign(campaign, runs, tuples, manifest):
    selected = [r for r in runs if r['run_id'].split('/')[0] == campaign
                and r['requested_model'] == 'gpt-5.6-luna' and r['kind'] == 'checks']
    by_effort = {r['reasoning']: r for r in selected}
    if len(selected) != len(EFFORTS) or set(by_effort) != set(EFFORTS):
        raise ValueError('Exactly one run for each of the six Luna efforts is required')
    identities = {(r['requested_model'], r.get('protocol_id', 'numeric-v1'), r.get('scope_id'),
                   r['target_tuples'], r['target_propositions']) for r in selected}
    if len(identities) != 1:
        raise ValueError('Cannot compare different models, protocols, or target scopes as reasoning treatments')
    scores = {effort: {row['id']: row for row in r['rows']} for effort, r in by_effort.items()}
    if any(len(scores[e]) != len(by_effort[e]['rows']) for e in EFFORTS):
        raise ValueError('Duplicate score IDs')
    shared = set.intersection(*(set(scores[e]) for e in EFFORTS))
    if not shared:
        raise ValueError('No common scored tuples')
    by_id = {row['id']: row for row in tuples}
    if len(by_id) != len(tuples) or not shared <= set(by_id):
        raise ValueError('Missing or duplicated source tuples')
    for e in EFFORTS:
        if any(row['check_type'] != by_id[ident]['check_type'] for ident, row in scores[e].items()):
            raise ValueError('Score check type disagrees with source')
    shared_questions = {p for ident in shared for p in by_id[ident]['propositions'].values()}
    by_type = {ident: row['check_type'] for ident, row in by_id.items()}
    rows = []
    receipt_sets = []
    for effort in EFFORTS:
        r = by_effort[effort]
        current = [p for p in r['receipts'] if p['is_current']]
        by_question = {p['id']: p for p in current}
        if len(by_question) != len(current) or len(current) != r['target_propositions']:
            raise ValueError('Current receipt coverage is incomplete or duplicated')
        receipt_sets.append(set(by_question))
        if any(not p['completed'] or p.get('tools_used') for p in current):
            raise ValueError('Incomplete or tool-using observation')
        compared = [by_question[i] for i in sorted(shared_questions)]
        if any(p['probability'] is None for p in compared):
            raise ValueError('Shared scored group contains a missing probability')
        with_usage = [p for p in compared if p.get('usage')]
        observed_tokens = Counter()
        for p in with_usage:
            observed_tokens.update(tokens('luna', p['usage']))
        rows.append({'effort': effort, 'run_id': r['run_id'],
                     'available_scored_tuples': len(scores[effort]),
                     'completed_observations': len(current),
                     'numerical_observations': sum(p['probability'] is not None for p in current),
                     'missing_probabilities': sum(p['probability'] is None for p in current),
                     'shared_mean': mean(scores[effort][i]['violation'] for i in sorted(shared)),
                     'available_mean': mean(p['violation'] for p in r['rows']),
                     'shared_readout_usage': {'observations': len(compared),
                         'with_usage': len(with_usage), 'tokens': dict(observed_tokens),
                         'median_elapsed_seconds': statistics.median(p['elapsed_seconds'] for p in compared),
                         'mean_output_tokens': mean(tokens('luna', p['usage'])['output_tokens'] for p in with_usage),
                         'mean_reasoning_output_tokens': mean(tokens('luna', p['usage'])['reasoning_output_tokens'] for p in with_usage)},
                     'per_check': {kind: {'n': sum(by_type[i] == kind for i in shared),
                         'mean': mean(scores[effort][i]['violation'] for i in sorted(shared) if by_type[i] == kind)}
                         for kind in sorted(set(by_type[i] for i in shared))}})
    if any(s != receipt_sets[0] for s in receipt_sets[1:]):
        raise ValueError('Efforts used different source question sets')
    pairs = [{'left': left, 'right': right, **pair_summary(scores[left], scores[right], shared, by_type)}
             for left, right in combinations(EFFORTS, 2)]
    # Count exact-text links, to expose why treating tuples as independent is unsafe.
    parent = {i: i for i in shared}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    first = {}
    for ident in sorted(shared):
        for question in by_id[ident]['propositions'].values():
            if question in first:
                parent[find(ident)] = find(first[question])
            else:
                first[question] = ident
    components = Counter(find(i) for i in shared)
    return {'campaign': campaign, 'protocol_id': next(iter(identities))[1],
            'scope_id': next(iter(identities))[2], 'target_tuples': selected[0]['target_tuples'],
            'target_questions': selected[0]['target_propositions'], 'shared_tuples': len(shared),
            'shared_questions': len(shared_questions), 'shared_tuple_ids': sorted(shared),
            'shared_tuple_ids_sha256': hashlib.sha256(json.dumps(sorted(shared)).encode()).hexdigest(),
            'source_runs': {r['run_id']: {k: r[k] for k in ('sha256', 'receipts_sha256', 'recorded_at')} for r in selected},
            'dependence': {'connected_components_by_exact_shared_question': len(components),
                           'largest_component_tuples': max(components.values()),
                           'note': 'Different question texts can still describe the same event; these are not proven independent clusters.'},
            'settings': rows, 'paired_comparisons': pairs}


def generate(snapshots):
    manifest, _, tuples, runs = load_snapshot(snapshots)
    return {'schema_version': 1, 'analysis': 'existing-luna-reasoning-sensitivity',
            'source_manifest_sha256': hashlib.sha256((snapshots / 'manifest.json').read_bytes()).hexdigest(),
            'campaigns': [analyze_campaign(c, runs, tuples, manifest) for c in CAMPAIGNS],
            'fresh_requests': 0,
            'interpretation': [
                'All effort comparisons within a campaign use the identical six-effort intersection.',
                'Original and corrected protocols/cohorts are separate experiments; their raw means are not paired treatment effects.',
                'This is a single observed run per effort, with no randomized repeated trials; no causal or significance claim is made.',
                'Groups share questions and events. Counts of improved groups are not independent statistical trials.',
                'Coverage selection excludes refusals and abstentions; missing readouts are never zero.',
                'Latency and token counters describe original receipts on shared questions, including reused observations; they are not new consumption.',
                'Luna is one model. This does not complete the Fireworks model-wide medium/high sweep or choose an open training base.'
            ]}


def render(report):
    lines = ['# Reasoning-mode sensitivity: existing Luna runs', '',
             'Task: `hub-t8k9c.10`. This analysis makes no new model requests. It uses the',
             'six already published Luna efforts, with exact matching group IDs within each',
             'protocol. The broader historical Fireworks sweep remains unrun.', '']
    for c in report['campaigns']:
        lines += [f"## {c['campaign']} ({c['protocol_id']})", '',
                  f"{c['shared_tuples']} shared scored groups from {c['target_tuples']} target groups; "
                  f"{c['shared_questions']} unique questions contribute to those scores.", '',
                  '| Effort | Shared mean violation | Available groups | Missing probabilities | Output tokens / shared question | Median seconds / shared question |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for row in c['settings']:
            usage = row['shared_readout_usage']
            lines.append(f"| {row['effort']} | {row['shared_mean']:.6f} | {row['available_scored_tuples']}/{c['target_tuples']} | "
                         f"{row['missing_probabilities']}/{c['target_questions']} | {usage['mean_output_tokens']:.1f} | {usage['median_elapsed_seconds']:.2f} |")
        lines += ['', 'Paired differences below are **right minus left** on that same shared set.',
                  'Negative differences mean lower violation with the right-hand effort.', '',
                  '| Left → right | Mean difference | Groups improved / worsened / tied |', '| --- | ---: | ---: |']
        for pair in c['paired_comparisons']:
            if pair['left'] in ('none', 'low') and pair['right'] in ('medium', 'high', 'xhigh', 'max'):
                lines.append(f"| {pair['left']} → {pair['right']} | {pair['mean_delta']:+.6f} | {pair['improved']} / {pair['worsened']} / {pair['unchanged']} |")
        lines += ['', 'Per-check differences for **none → high**:', '', '| Check | Groups | Mean difference |', '| --- | ---: | ---: |']
        comparison = next(p for p in c['paired_comparisons'] if p['left'] == 'none' and p['right'] == 'high')
        for kind, stats in comparison['per_check'].items():
            lines.append(f"| {kind} | {stats['n']} | {stats['mean_delta']:+.6f} |")
        lines += ['', f"Exact shared question texts link these groups into {c['dependence']['connected_components_by_exact_shared_question']} "
                  f"components; the largest contains {c['dependence']['largest_component_tuples']} groups. "
                  'Additional dependence through differently worded versions of the same event is possible.', '']
    lines += ['## What this establishes and what remains', ''] + ['- ' + line for line in report['interpretation']]
    lines += ['', 'The machine-readable [report](../data/research/reasoning-sensitivity.json) contains all',
              '15 pairwise effort comparisons per campaign, every per-check delta, exact shared',
              'IDs, frozen source hashes/timestamps, usage coverage, and numerical missingness.', '',
              'Reproduce without inference:', '', '```sh', 'python3 scripts/analyze_reasoning.py', '```', '',
              'Remaining Fireworks work: verify supported/effective effort settings, complete the',
              'missing original models, and collect medium/high runs on the same original cohort',
              'with explicit coverage limits. A accepted request field alone does not prove that',
              'the provider applied the requested reasoning mode.', '']
    return '\n'.join(lines).replace('A accepted request', 'An accepted request')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, default=ROOT / 'data/benchmark-runs')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/research/reasoning-sensitivity.json')
    parser.add_argument('--markdown', type=Path, default=ROOT / 'docs/reasoning-sensitivity.md')
    args = parser.parse_args()
    result = generate(args.snapshots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    args.markdown.write_text(render(result))
    print(json.dumps({'fresh_requests': 0, 'shared_groups': {c['campaign']: c['shared_tuples'] for c in result['campaigns']}}))


if __name__ == '__main__':
    main()
