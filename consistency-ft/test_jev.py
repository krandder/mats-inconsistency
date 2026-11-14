"""Direct-probability provenance, missingness and shared-cohort invariants."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness.run_jev_checks import parse_response, request_for, INSTRUCTIONS
from scripts.export_jev import export
from scripts.export_benchmark import load_snapshot, comparison_context, refresh
from scripts.build_dash import original_benchmark_runs, original_comparison_context, corrected_benchmark_runs, coverage_table

ROOT = Path(__file__).resolve().parent


class JevTests(unittest.TestCase):
    def test_probability_readout_rejects_model_drift_and_invalid_values(self):
        response = {'model': 'jev-test', 'answers': {'proposition': {'type': 'noul', 'noul': .4}}}
        self.assertEqual(parse_response(response, 'jev-test'), .4)
        with self.assertRaises(ValueError):
            parse_response(response, 'jev-other')
        for bad in [True, '0.4', float('nan'), -.1, 1.1, None]:
            response['answers']['proposition']['noul'] = bad
            with self.assertRaises(ValueError):
                parse_response(response, 'jev-test')

    def test_question_is_independent_and_preserves_complete_definition(self):
        text = 'Will A happen?\n\nResolution criteria: Exact dates and evidence.'
        request = request_for(text, 'jev-test')
        self.assertEqual(request['state'], text)
        self.assertEqual(request['questions'], {'proposition': {'type': 'noul', 'instructions': INSTRUCTIONS}})

    def test_published_receipts_and_original_corrected_comparisons(self):
        manifest, props, tuples, runs = load_snapshot()
        jev = [r for r in runs if r['provider'] == 'typesafe']
        self.assertEqual(len(jev), 1)
        run = jev[0]
        current = [r for r in run['receipts'] if r['is_current']]
        self.assertEqual({r['id'] for r in current}, {p['id'] for p in props})
        self.assertEqual(len(current), 591)
        self.assertEqual(len(run['rows']), 300)
        texts = {p['id']: p['text'] for p in props}
        for row in current:
            self.assertTrue(row['completed'])
            self.assertEqual(row['request'], request_for(texts[row['id']], run['requested_model']))
            self.assertEqual(parse_response(json.loads(row['response']), run['requested_model']), row['probability'])
        for corrected, expected in [(False, 264), (True, 148)]:
            def context(selected):
                if corrected:
                    return comparison_context(manifest, corrected_benchmark_runs(manifest, selected))
                chosen = original_benchmark_runs(selected)
                return original_comparison_context(props, tuples, chosen)
            before = context([r for r in runs if r['provider'] != 'typesafe'])
            after = context(runs)
            self.assertEqual(before['common_tuple_ids'], after['common_tuple_ids'])
            self.assertEqual(len(after['common_tuple_ids']), expected)
            for model, old in before['comparison_checks'].items():
                self.assertEqual(old['rows'], after['comparison_checks'][model]['rows'])
                self.assertEqual(old['stat'], after['comparison_checks'][model]['stat'])
            scoped = after['scoped_checks'][run['model']]
            self.assertEqual(scoped['coverage']['valid_propositions'], 274 if corrected else 591)
            self.assertIn('TypeSafe Noul API', coverage_table({run['model']: scoped}))

    def test_changed_published_export_fails_before_overwriting_files(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, source = root / 'snapshot', root / 'receipts'
            shutil.copytree(ROOT / 'data/benchmark-runs', directory)
            source.mkdir()
            manifest, _, _, runs = load_snapshot(directory)
            run = next(r for r in runs if r['provider'] == 'typesafe')
            config_keys = ['protocol_id', 'model', 'endpoint', 'instructions',
                           'propositions_sha256', 'question_count', 'questions_per_request']
            (source / 'campaign.json').write_text(json.dumps({k: run['protocol'][k] for k in config_keys}))
            for row in run['receipts']:
                # Different file provenance must not replace a published campaign.
                (source / (row['attempt_id'] + '.json')).write_text(json.dumps(row))
            before = {name: (directory / name).read_bytes() for name in
                      [run['file'], run['receipts_file'], 'manifest.json']}
            with self.assertRaisesRegex(ValueError, 'published run is immutable'):
                export(source, directory)
            self.assertEqual(before, {name: (directory / name).read_bytes() for name in before})

    def test_native_refresh_preserves_jev_receipts_identity(self):
        # Exercise real refresh in a copied snapshot; no private native runs required.
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(ROOT / 'data/benchmark-runs', root / 'data/benchmark-runs')
            shutil.copytree(ROOT / 'data/tuples', root / 'data/tuples')
            directory = root / 'data/benchmark-runs'
            before = next(r for r in load_snapshot(directory)[0]['runs'] if r['provider'] == 'typesafe')
            refresh(root, directory)
            after = next(r for r in load_snapshot(directory)[0]['runs'] if r['provider'] == 'typesafe')
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
