import json
from pathlib import Path
import shutil
import tempfile
import unittest

from harness.run_probes import import_receipts, read_jsonl
from harness.run_native_probes import COLLECTOR, prepare, load_campaign
from scripts.report_probe_usage import campaign_records, generate, quota_changes


class ProbeUsageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / 'source.jsonl'
        source.write_text(json.dumps({'question': 'Which studio made the film?', 'answer': 'North Studio',
                                     'accept': ['North Studio'], 'event_date': '2025-01-01',
                                     'quarter': '2025Q1', 'category': 'film', 'source': 'https://example.org/film'}) + '\n')
        cutoffs = self.root / 'cutoffs.json'
        cutoffs.write_text('{}')
        self.directory = self.root / 'astra'
        prepare(self.directory, probes_path=source, cutoffs_path=cutoffs, provider='astra', effort='low', as_of='2026-09-13')
        self.request = read_jsonl(self.directory / 'requests.jsonl')[0]

    def receipt(self, session='test-only-a', status='completed', usage=None, hour=1):
        row = {k: self.request[k] for k in ('probe_id', 'model', 'provider', 'reasoning', 'protocol_id', 'prompt_sha256')}
        _, protocol = load_campaign(self.directory)
        if usage is not None:
            usage = {'cached_input_tokens': 0, 'reasoning_output_tokens': 0, **usage}
        row.update(started_at=f'2026-09-13T{hour:02}:00:00Z', finished_at=f'2026-09-13T{hour:02}:00:01Z',
                   response='UNKNOWN' if status == 'completed' else '', status=status,
                   tools_used=False, usage=usage, elapsed_seconds=1,
                   collector=COLLECTOR, native_protocol_fingerprint=protocol['fingerprint'],
                   system_instruction_sha256=protocol['system_instruction_sha256'], requested_model='gpt-6-astra',
                   provenance={'transport': 'subscription-native-cli', 'adapter': COLLECTOR, 'session_id': session})
        if status == 'infrastructure_error':
            row['error'] = 'test transport failure'
        return row

    def save(self, rows):
        path = self.root / 'input.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        import_receipts(self.directory, path)

    def test_failed_usage_and_unknown_usage_are_not_erased_or_double_counted(self):
        self.save([self.receipt(status='infrastructure_error', usage={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}),
                   self.receipt(session='test-only-b', hour=2)])
        report = generate(self.root)
        self.assertEqual(report['attempt_receipts'], 2)
        self.assertEqual(report['total_tokens']['total_tokens'], 12)
        self.assertEqual(report['configurations'][0]['attempts_without_usage'], 1)
        self.assertEqual(report['configurations'][0]['completed_observations'], 1)
        self.assertEqual(report['proxy']['matched_requests'], 0)
        self.assertNotIn('test-only-b', json.dumps(report))

    def test_proxy_attribution_filters_other_fleet_sessions_and_does_not_add_tokens(self):
        self.save([self.receipt(usage={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})])
        proxy = [{'session_id': 'test-only-a', 'account': 'private-person-codex', 'tokens': {'input': 11, 'output': 2}},
                 {'session_id': 'unrelated', 'account': 'other-person-codex', 'tokens': {'input': 9000, 'output': 999}}]
        (self.root / 'proxy-usage-after.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in proxy))
        report = generate(self.root)
        self.assertEqual(report['proxy']['matched_requests'], 1)
        self.assertEqual(report['proxy']['accounts'][0]['input_tokens'], 11)
        self.assertEqual(report['total_tokens']['total_tokens'], 12)
        self.assertNotIn('private-person', json.dumps(report))
        self.assertNotIn('unrelated', json.dumps(report))

    def test_missing_reset_identity_and_changed_windows_prevent_subtraction(self):
        for state, percent, reset in [('before', 1, 100), ('after', 4, 200)]:
            (self.root / f'quota-glm-{state}.json').write_text(json.dumps({'response': {'data': {'limits': [
                {'type': 'TOKENS_LIMIT', 'unit': 3, 'number': 5, 'percentage': percent},
                {'type': 'TOKENS_LIMIT', 'unit': 6, 'number': 1, 'percentage': percent, 'nextResetTime': reset}]}}}))
            (self.root / f'quota-grok-{state}.json').write_text(json.dumps({'used_percent': percent, 'reset_display': str(reset)}))
        changes = quota_changes(self.root, [])
        self.assertEqual(len(changes), 3)
        self.assertTrue(all(c['delta_percentage_points'] is None for c in changes))

    def test_matching_openai_window_is_anonymous_shared_delta(self):
        for state, percent, timestamp, remaining in [('before', 10, 1000, 500), ('after', 13, 1100, 400)]:
            (self.root / f'quota-astra-{state}.json').write_text(json.dumps({'captured_at': timestamp, 'profiles': [
                {'profile': 'private-person', 'ts': timestamp, 'windows': [{'window': 'weekly', 'used_percent': percent,
                                                                            'reset_after_seconds': remaining}]}]}))
        changes = quota_changes(self.root, ['private-person-codex'])
        self.assertEqual(changes[0]['delta_percentage_points'], 3)
        self.assertEqual(changes[0]['subscription'], 1)
        self.assertNotIn('private-person', json.dumps(changes))

    def test_reused_session_cannot_double_attribute_a_retry(self):
        self.save([self.receipt(status='infrastructure_error'), self.receipt(hour=2)])
        with self.assertRaises(ValueError):
            campaign_records(self.root)

    def test_unbound_receipt_cannot_be_reported_as_native_subscription_usage(self):
        row = self.receipt()
        row.pop('native_protocol_fingerprint')
        row['provenance']['transport'] = 'arbitrary-import'
        self.save([row])
        with self.assertRaises(ValueError):
            generate(self.root)

    def test_inconsistent_normalized_total_is_rejected(self):
        self.save([self.receipt(usage={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 9999})])
        with self.assertRaises(ValueError):
            generate(self.root)

    def test_cloned_campaign_cannot_double_count_without_openai_sessions(self):
        self.save([self.receipt()])
        shutil.copytree(self.directory, self.root / 'cloned')
        with self.assertRaisesRegex(ValueError, 'campaign occurs'):
            campaign_records(self.root)


if __name__ == '__main__':
    unittest.main()
