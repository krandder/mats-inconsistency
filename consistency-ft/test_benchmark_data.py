"""Regression tests for missingness, provenance, and reproducible public exports."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from common import VERBAL_PROMPT
from harness.run_native_checks import PROTOCOLS, key
from scripts.export_benchmark import load_snapshot, rebuild_database, refresh
from scripts.build_dash import headline_runs, latest_runs, ranking_chart


class BenchmarkDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = self.root / 'data/benchmark-runs'
        (self.root / 'data/tuples').mkdir(parents=True)
        (self.root / 'results').mkdir()
        self.receipts = self.root / 'runs/native-20260912/astra'
        self.receipts.mkdir(parents=True)
        tuples = [
            {'id': 'neg-1', 'check_type': 'Negation', 'questions': {'P': {'title': 'A'}, 'not_P': {'title': 'B'}}},
            {'id': 'neg-2', 'check_type': 'Negation', 'questions': {'P': {'title': 'A'}, 'not_P': {'title': 'C'}}},
        ]
        self.jsonl(self.root / 'data/tuples/Negation.jsonl', tuples)
        for proposition, probability, response in [('A', .8, '0.8'), ('B', None, 'I cannot answer this question.'), ('C', .25, '0.25')]:
            self.receipt(proposition, probability, response)
        legacy = {'id': 'neg-1', 'check_type': 'Negation', 'model': 'mock/sha256',
                  'probabilities': {'P': .8, 'not_P': .25}, 'violation': .05}
        self.jsonl(self.root / 'results/checks_smoke.jsonl', [legacy])
        elicit = {'id': 'legacy-question', 'title': 'Original title', 'model': 'legacy/model',
                  'verbalized_probability': .3, 'logprob_probability': .4, 'abs_diff': .1}
        self.jsonl(self.root / 'results/elicit30_model.jsonl', [elicit])

    @staticmethod
    def jsonl(path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def receipt(self, proposition, probability, response, suffix=''):
        ident = hashlib.sha256(proposition.encode()).hexdigest()
        raw = {'id': ident, 'provider': 'astra', 'requested_model': 'gpt-6-astra',
               'proposition': proposition, 'prompt': VERBAL_PROMPT.format(proposition=proposition),
               'response': response, 'probability': probability, 'exit_code': 0,
               'started_at': '2026-09-12T12:00:00+00:00', 'finished_at': '2026-09-12T12:00:01+00:00',
               'tools_used': False, 'usage': {'input_tokens': 42, 'output_tokens': 2, 'secret': 'PRIVATE'},
               'stdout': 'PRIVATE STDOUT', 'stderr': 'PRIVATE STDERR', 'session_id': 'PRIVATE SESSION'}
        if probability is None:
            raw['error'] = 'final response is not a single probability'
        (self.receipts / f'{ident}{suffix}.json').write_text(json.dumps(raw))
        return ident

    def export(self):
        refresh(self.root, self.snapshot)
        self.database = rebuild_database(self.snapshot, self.root / 'benchmark.sqlite')
        return load_snapshot(self.snapshot)

    def test_refusal_is_missing_and_invalidates_dependent_tuple(self):
        _, _, _, runs = self.export()
        native = next(r for r in runs if r['run_id'].endswith('/astra'))
        self.assertEqual([row['id'] for row in native['rows']], ['neg-2'])
        self.assertAlmostEqual(native['stat'], .05)
        self.assertEqual(native['coverage']['completed_propositions'], 3)
        self.assertEqual(native['coverage']['valid_propositions'], 2)
        self.assertEqual(native['coverage']['errors'], {'refusal': 1})
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM current_probabilities WHERE probability IS NULL AND completed=1').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM tuple_scores WHERE run_id LIKE "native-%"').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM ranking WHERE model="mock/sha256"').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT title FROM elicitation').fetchone()[0], 'Original title')
        public = '\n'.join(path.read_text() for path in self.snapshot.glob('*'))
        self.assertNotIn('PRIVATE', public)
        self.assertNotIn('session_id', public)

    def test_new_protocol_preserves_old_runs_and_explicit_abstentions(self):
        self.export()
        old_manifest = json.loads((self.snapshot / 'manifest.json').read_text())
        old_run = next(r for r in old_manifest['runs'] if r['run_id'].endswith('/astra'))
        old_scores = (self.snapshot / old_run['file']).read_bytes()
        campaign = 'native-next'
        directory = self.root / 'runs' / campaign
        (directory / 'astra').mkdir(parents=True)
        metadata = {'campaign': campaign, 'protocol_id': 'descriptive-v2', 'category': 'benchmark',
                    **PROTOCOLS['descriptive-v2'], 'target_tuples': 2, 'target_propositions': 3}
        (directory / 'campaign.json').write_text(json.dumps(metadata))
        for proposition, probability, response in [('A', .7, '0.7'), ('B', None, 'ABSTAIN'), ('C', .6, '0.6')]:
            ident = key(proposition)
            original = json.loads((self.receipts / (ident + '.json')).read_text())
            original.update(campaign=campaign, protocol_id='descriptive-v2', completed=True,
                system_instruction_sha256=key(PROTOCOLS['descriptive-v2']['system_instruction']),
                prompt=PROTOCOLS['descriptive-v2']['verbal_prompt'].format(proposition=proposition),
                response=response, probability=probability, finished_at='2026-09-13T12:00:01+00:00')
            (directory / 'astra' / (ident + '.json')).write_text(json.dumps(original))
        refresh(self.root, self.snapshot, campaigns=[campaign])
        self.assertEqual((self.snapshot / old_run['file']).read_bytes(), old_scores)
        database = rebuild_database(self.snapshot, self.root / 'benchmark.sqlite')
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM current_probabilities').fetchone()[0], 6)
            self.assertEqual(db.execute('SELECT protocol_id FROM ranking WHERE requested_model="gpt-6-astra"').fetchone()[0], 'descriptive-v2')
            self.assertEqual(db.execute('SELECT count(*) FROM current_probabilities WHERE error_kind="abstention" AND probability IS NULL AND completed=1').fetchone()[0], 1)
        metadata['category'] = 'diagnostic'
        (directory / 'campaign.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'diagnostic campaign cannot enter'):
            refresh(self.root, self.snapshot, campaigns=[campaign])

    def test_retries_remain_queryable_but_only_current_values_score(self):
        self.receipt('A', .1, '0.1', '.attempt-1234')
        _, _, _, runs = self.export()
        native = next(r for r in runs if r['run_id'].endswith('/astra'))
        self.assertEqual(native['coverage']['attempts'], 4)
        self.assertEqual(native['coverage']['observed_propositions'], 3)
        self.assertAlmostEqual(native['stat'], .05)
        with sqlite3.connect(self.database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM receipts').fetchone()[0], 4)
            self.assertEqual(db.execute('SELECT count(*) FROM current_probabilities').fetchone()[0], 3)

    def test_all_luna_reasoning_levels_exist_without_fabricated_scores(self):
        self.export()
        with sqlite3.connect(self.database) as db:
            rows = db.execute('SELECT reasoning, status, scored_count, mean_violation FROM ranking WHERE requested_model="gpt-5.6-luna"').fetchall()
        self.assertEqual({row[0] for row in rows}, {'none', 'low', 'medium', 'high', 'xhigh', 'max'})
        self.assertTrue(all(row[1:] == ('pending', 0, None) for row in rows))

    def test_frozen_timestamps_and_database_survive_checkout_mtimes(self):
        self.export()
        manifest_bytes = (self.snapshot / 'manifest.json').read_bytes()
        database_bytes = self.database.read_bytes()
        for path in (self.root / 'results').glob('*'):
            os.utime(path, (1, 1))
        for path in self.snapshot.glob('*'):
            os.utime(path, (2, 2))
        self.export()
        self.assertEqual((self.snapshot / 'manifest.json').read_bytes(), manifest_bytes)
        self.assertEqual(self.database.read_bytes(), database_bytes)
        # The rebuild reads only the published snapshot and still preserves history.
        for path in (self.root / 'results').glob('*'):
            path.unlink()
        rebuild_database(self.snapshot, self.database)
        self.assertEqual(self.database.read_bytes(), database_bytes)

    def test_tampered_probability_is_rejected_before_publication(self):
        _, _, _, runs = self.export()
        native = next(r for r in runs if r['run_id'].endswith('/astra'))
        path = self.snapshot / native['file']
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['probabilities']['P'] = .1
        self.jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, 'snapshot hash mismatch'):
            rebuild_database(self.snapshot, self.database)

    def test_raw_probability_must_agree_with_native_response(self):
        self.receipt('A', .9, '0.8')
        with self.assertRaisesRegex(ValueError, 'disagrees with native response'):
            self.export()

    def test_partial_runs_never_get_full_run_headline(self):
        ranking = [('partial', 0, 299), ('full', .2, 300)]
        self.assertEqual(headline_runs(ranking), [('full', .2, 300)])
        self.assertEqual(headline_runs(ranking[:1]), [])
        self.assertIn('299/300', ranking_chart(ranking))
        self.assertIn('PARTIAL RUN', ranking_chart(ranking))
        first = {'kind': 'checks', 'model': 'M', 'recorded_at': '2026-08-20', 'file': 'z'}
        last = {**first, 'recorded_at': '2026-09-12', 'file': 'a'}
        self.assertEqual(latest_runs([last, first])[('checks', 'M')], last)


if __name__ == '__main__':
    unittest.main()
