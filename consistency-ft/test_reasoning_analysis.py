import unittest
from scripts.analyze_reasoning import analyze_campaign, pair_summary, EFFORTS


def fixture():
    tuples = [{'id': 'shared', 'check_type': 'Negation', 'propositions': {'P': 'p', 'not_P': 'q'}},
              {'id': 'extra', 'check_type': 'Negation', 'propositions': {'P': 'p', 'not_P': 'r'}}]
    runs = []
    for effort in EFFORTS:
        rows = [{'id': 'shared', 'check_type': 'Negation', 'violation': 0.8 if effort == 'none' else 0.3}]
        if effort == 'none':
            rows.append({'id': 'extra', 'check_type': 'Negation', 'violation': 0.0})
        runs.append({'run_id': 'test/luna-' + effort, 'requested_model': 'gpt-5.6-luna', 'kind': 'checks',
                     'reasoning': effort, 'protocol_id': 'numeric-v1', 'scope_id': None,
                     'target_tuples': 2, 'target_propositions': 3, 'rows': rows,
                     'sha256': 'score-sha', 'receipts_sha256': 'receipt-sha', 'recorded_at': '2026-01-01',
                     'receipts': [{'id': i, 'is_current': True, 'completed': True, 'probability': 0.5,
                                   'elapsed_seconds': 1, 'usage': {'input_tokens': 10, 'output_tokens': 2}}
                                  for i in ['p', 'q', 'r']]})
    return runs, tuples


class ReasoningAnalysisTests(unittest.TestCase):
    def test_common_cohort_prevents_available_mean_bias(self):
        runs, tuples = fixture()
        report = analyze_campaign('test', runs, tuples, {})
        self.assertEqual(report['shared_tuples'], 1)
        self.assertEqual(report['settings'][0]['available_mean'], 0.4)
        self.assertEqual(report['settings'][0]['shared_mean'], 0.8)
        delta = next(p for p in report['paired_comparisons'] if p['left'] == 'none' and p['right'] == 'high')
        self.assertAlmostEqual(delta['mean_delta'], -0.5)
        self.assertEqual(delta['improved'], 1)
        self.assertEqual(report['settings'][0]['shared_readout_usage']['observations'], 2)

    def test_paired_direction_and_ties_with_per_check(self):
        left = {i: {'violation': p} for i, p in [('a', .5), ('b', .1), ('c', .2)]}
        right = {i: {'violation': p} for i, p in [('a', .2), ('b', .4), ('c', .2)]}
        result = pair_summary(left, right, set(left), {'a': 'And', 'b': 'Or', 'c': 'Or'})
        self.assertAlmostEqual(result['mean_delta'], 0)
        self.assertEqual((result['improved'], result['worsened'], result['unchanged']), (1, 1, 1))
        self.assertAlmostEqual(result['per_check']['And']['mean_delta'], -.3)
        self.assertAlmostEqual(result['per_check']['Or']['mean_delta'], .15)

    def test_missing_or_duplicate_effort_and_changed_protocol_rejected(self):
        runs, tuples = fixture()
        with self.assertRaises(ValueError): analyze_campaign('test', runs[:-1], tuples, {})
        with self.assertRaises(ValueError): analyze_campaign('test', runs + [runs[0]], tuples, {})
        runs[0]['protocol_id'] = 'changed'
        with self.assertRaises(ValueError): analyze_campaign('test', runs, tuples, {})

    def test_missing_probability_cannot_enter_shared_scored_group(self):
        runs, tuples = fixture()
        runs[0]['receipts'][0]['probability'] = None
        with self.assertRaises(ValueError): analyze_campaign('test', runs, tuples, {})

    def test_different_questions_or_incomplete_execution_are_not_reasoning_comparison(self):
        runs, tuples = fixture()
        runs[0]['receipts'][-1]['id'] = 'new-question'
        with self.assertRaises(ValueError): analyze_campaign('test', runs, tuples, {})
        runs, tuples = fixture()
        runs[0]['receipts'][-1]['completed'] = False
        with self.assertRaises(ValueError): analyze_campaign('test', runs, tuples, {})


if __name__ == '__main__': unittest.main()
