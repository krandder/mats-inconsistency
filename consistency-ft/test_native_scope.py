"""Reviewed scopes restrict scheduling, and imported readouts are never resampled."""
import hashlib
import json
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch

from harness.run_native_checks import PROTOCOLS, key, main
from harness.scope import validate_scope
from scripts.export_benchmark import comparison_context, refresh, rebuild_database, load_snapshot
from scripts.build_dash import latest_runs
from scripts.measure_subscription_usage import campaign_receipt_groups


class NativeScopeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tuples = [
            {"id": "keep", "check_type": "Negation", "questions": {"P": {"title": "A"}, "not_P": {"title": "not A"}}},
            {"id": "exclude", "check_type": "Negation", "questions": {"P": {"title": "B"}, "not_P": {"title": "not B"}}},
        ]
        directory = self.root / "data/tuples"
        directory.mkdir(parents=True)
        source = directory / "Negation.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in self.tuples))
        self.scope = {"scope_id": "reviewed-test", "included_tuple_ids": ["keep"],
            "excluded_tuple_ids": ["exclude"], "included_proposition_ids": [key("A"), key("not A")],
            "exclusions": [{"tuple_id": "exclude", "reason": "source semantic defect"}],
            "source_cohort": {"tuple_count": 2, "proposition_count": 4,
                "source_file_sha256": {"Negation.jsonl": hashlib.sha256(source.read_bytes()).hexdigest()}}}
        self.scope_path = self.root / "scope.json"
        self.scope_path.write_text(json.dumps(self.scope))

    def test_scope_requires_whole_tuple_union_and_original_source(self):
        result = validate_scope(self.scope_path, self.tuples, self.root / "data/tuples")
        self.assertEqual([r["id"] for r in result["tuples"]], ["keep"])
        self.scope["included_proposition_ids"].pop()
        self.scope_path.write_text(json.dumps(self.scope))
        with self.assertRaisesRegex(ValueError, "complete included-tuple union"):
            validate_scope(self.scope_path, self.tuples)
        self.scope["included_proposition_ids"].append(key("not A"))
        self.scope["source_cohort"]["source_file_sha256"]["Negation.jsonl"] = "changed"
        self.scope_path.write_text(json.dumps(self.scope))
        with self.assertRaisesRegex(ValueError, "source file fingerprints"):
            validate_scope(self.scope_path, self.tuples, self.root / "data/tuples")

    def test_import_preserves_values_and_abstentions_without_new_calls(self):
        original = self.root / "runs/interrupted"
        (original / "astra").mkdir(parents=True)
        (original / "campaign.json").write_text(json.dumps({"campaign": "interrupted", "category": "benchmark",
                                                           "protocol_id": "descriptive-v2", **PROTOCOLS["descriptive-v2"]}))
        source_bytes = {}
        for text, response, probability in [("A", "0.4", .4), ("not A", "ABSTAIN", None)]:
            record = {"id": key(text), "proposition": text, "campaign": "interrupted", "protocol_id": "descriptive-v2",
                "requested_model": "gpt-6-astra", "provider": "astra", "reasoning": "low", "completed": True,
                "response": response, "probability": probability, "exit_code": 0, "elapsed_seconds": 1,
                "started_at": "2026-09-13T01:00:00Z", "finished_at": "2026-09-13T01:00:01Z",
                "usage": {"input_tokens": 10, "output_tokens": 3},
                "prompt": PROTOCOLS["descriptive-v2"]["verbal_prompt"].format(proposition=text),
                "system_instruction_sha256": key(PROTOCOLS["descriptive-v2"]["system_instruction"])}
            path = original / "astra" / (key(text) + ".json")
            path.write_text(json.dumps(record))
            source_bytes[key(text)] = path.read_bytes()
        destination = self.root / "runs/scoped"
        argv = ["runner", "astra", "--campaign", "scoped", "--protocol", "descriptive-v2",
                "--run-dir", str(destination), "--scope-manifest", str(self.scope_path),
                "--import-run-dir", str(original)]
        with patch("harness.run_native_checks.ROOT", self.root), patch("sys.argv", argv), \
             patch("harness.run_native_checks.subprocess.Popen") as launch:
            self.assertEqual(main(), 0)
            launch.assert_not_called()
        imported = [json.loads(p.read_text()) for p in (destination / "astra").glob("*.json")]
        self.assertEqual(len(imported), 2)
        self.assertEqual({r["response"] for r in imported}, {"0.4", "ABSTAIN"})
        for record in imported:
            self.assertEqual(record["imported_from"]["receipt_sha256"], hashlib.sha256(source_bytes[record["id"]]).hexdigest())
            self.assertEqual(record["finished_at"], "2026-09-13T01:00:01Z")
            self.assertEqual((original / "astra" / (record["id"] + ".json")).read_bytes(), source_bytes[record["id"]])
        status = json.loads((destination / "status-astra.json").read_text())
        self.assertEqual(status["target"], 2)
        self.assertEqual(status["excluded_tuples"], 1)
        self.assertEqual(status["imported_observations"], 2)
        self.assertEqual(status["missing_probabilities"], 1)
        self.assertEqual(status["scored_tuples"], 0)
        snapshots = self.root / "data/benchmark-runs"
        refresh(self.root, snapshots, campaigns=["scoped"])
        database = rebuild_database(snapshots, self.root / "benchmark.sqlite")
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM scope_exclusions').fetchone()[0], 1)
            self.assertEqual(db.execute('SELECT count(*) FROM receipts WHERE is_imported=1').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT count(*) FROM receipts WHERE proposition_id IN (?,?)', (key("B"), key("not B"))).fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT target_tuples,target_propositions FROM runs WHERE run_id="scoped/astra"').fetchone(), (1, 2))
            self.assertEqual(db.execute('SELECT headline_eligible FROM ranking WHERE run_id="scoped/astra"').fetchone()[0], 0)
            self.assertEqual(db.execute('PRAGMA integrity_check').fetchone()[0], "ok")

    def test_every_setting_uses_the_same_scored_intersection(self):
        # Sparse archived coverage must shrink every setting's displayed sample.
        scope = {"included_tuple_ids": ["first", "second"], "included_proposition_ids": ["a", "b"]}
        (self.root / "comparison-scope.json").write_text(json.dumps(scope))
        manifest = {"comparison_scope_id": "reviewed", "scopes": {"reviewed": {"file": "comparison-scope.json"}}}
        def run(model, rows):
            return {"model": model, "provider": "native", "kind": "checks", "file": model,
                    "recorded_at": "2026-09-13", "rows": [{"id": ident, "violation": value} for ident, value in rows],
                    "n": len(rows), "stat": 99, "coverage": {}, "target_tuples": 300}
        runs = [run("complete", [("first", .1), ("second", .8), ("outside", .9)]),
                run("sparse", [("first", .3), ("outside", .4)])]
        comparison = comparison_context(manifest, runs, self.root)
        self.assertEqual(comparison["common_tuple_ids"], {"first"})
        self.assertEqual(comparison["scoped_checks"]["complete"]["n"], 2)
        self.assertEqual(comparison["scoped_checks"]["sparse"]["n"], 1)
        self.assertEqual(comparison["comparison_checks"]["complete"]["stat"], .1)
        self.assertEqual({r["n"] for r in comparison["comparison_checks"].values()}, {1})
        self.assertEqual({r["target_tuples"] for r in comparison["comparison_checks"].values()}, {1})

    def test_ordered_multi_source_imports_preserve_abstentions_hashes_and_resume(self):
        first, second = [self.root / "runs" / name for name in ("first", "second")]
        for source in (first, second):
            (source / "astra").mkdir(parents=True)
            (source / "campaign.json").write_text(json.dumps({"campaign": source.name, "category": "benchmark",
                "protocol_id": "descriptive-v2", **PROTOCOLS["descriptive-v2"]}))

        def save(source, text, response, probability, **extra):
            record = {"id": key(text), "proposition": text, "campaign": source.name, "protocol_id": "descriptive-v2",
                "requested_model": "gpt-6-astra", "provider": "astra", "reasoning": "low", "completed": True,
                "response": response, "probability": probability, "exit_code": 0, "elapsed_seconds": 1,
                "started_at": "2026-09-13T01:00:00Z", "finished_at": "2026-09-13T01:00:01Z",
                "usage": {"input_tokens": 10, "output_tokens": 3},
                "prompt": PROTOCOLS["descriptive-v2"]["verbal_prompt"].format(proposition=text),
                "system_instruction_sha256": key(PROTOCOLS["descriptive-v2"]["system_instruction"]), **extra}
            path = source / "astra" / (key(text) + ".json")
            path.write_text(json.dumps(record))
            return path

        original = save(first, "A", "0.4", .4)
        save(second, "A", "0.4", .4, imported_from={"campaign": "first", "run_key": "astra",
             "receipt_id": key("A"), "receipt_sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
        abstention = save(second, "not A", "ABSTAIN", None, readout_status="abstention", error_kind="abstention")
        source_bytes = {path: path.read_bytes() for source in (first, second) for path in source.rglob("*.json")}

        def argv(campaign, sources):
            command = ["runner", "astra", "--campaign", campaign, "--protocol", "descriptive-v2",
                       "--run-dir", str(self.root / "runs" / campaign), "--scope-manifest", str(self.scope_path)]
            for source in sources:
                command += ["--import-run-dir", str(source)]
            return command

        destination = self.root / "runs/combined"
        with patch("harness.run_native_checks.ROOT", self.root), \
             patch("harness.run_native_checks.subprocess.Popen") as launch:
            with patch("sys.argv", argv("combined", [first, second])):
                self.assertEqual(main(), 0)
                imported_bytes = {path: path.read_bytes() for path in (destination / "astra").glob("*.json")}
                self.assertEqual(main(), 0)
                self.assertEqual({path: path.read_bytes() for path in (destination / "astra").glob("*.json")}, imported_bytes)
            # Without its origin already imported, nested provenance stays forbidden.
            with patch("sys.argv", argv("wrong-order", [second, first])):
                with self.assertRaisesRegex(ValueError, "nested import provenance.*--import-run-dir for first before second"):
                    main()
            launch.assert_not_called()

        self.assertEqual(len(imported_bytes), 2)
        for source in (original, abstention):
            row = json.loads((destination / "astra" / source.name).read_text())
            old = json.loads(source_bytes[source])
            self.assertEqual(row["imported_from"], {"campaign": old["campaign"], "run_key": "astra",
                "receipt_id": old["id"], "receipt_sha256": hashlib.sha256(source_bytes[source]).hexdigest()})
            for field in ("response", "probability", "completed", "started_at", "finished_at", "usage"):
                self.assertEqual(row[field], old[field])
        self.assertEqual({path: path.read_bytes() for path in source_bytes}, source_bytes)
        status = json.loads((destination / "status-astra.json").read_text())
        self.assertEqual((status["imported_observations"], status["missing_probabilities"], status["scored_tuples"]), (2, 1, 0))
        self.assertEqual(list(campaign_receipt_groups(destination)), [])  # no new launch/token attribution

    def test_sql_ranking_uses_scoped_target_and_matches_python_latest_ties(self):
        scope_bytes = self.scope_path.read_bytes()
        for campaign in ("scoped-a", "scoped-z"):
            directory = self.root / "runs" / campaign
            (directory / "astra").mkdir(parents=True)
            (directory / "scope.json").write_bytes(scope_bytes)
            metadata = {"campaign": campaign, "category": "benchmark", "protocol_id": "descriptive-v2",
                        **PROTOCOLS["descriptive-v2"], "scope_id": self.scope["scope_id"],
                        "scope_manifest_file": "scope.json", "scope_sha256": hashlib.sha256(scope_bytes).hexdigest()}
            (directory / "campaign.json").write_text(json.dumps(metadata))
            for text in ("A", "not A"):
                record = {"id": key(text), "provider": "astra", "requested_model": "gpt-6-astra", "reasoning": "low",
                          "proposition": text, "campaign": campaign, "protocol_id": "descriptive-v2",
                          "prompt": PROTOCOLS["descriptive-v2"]["verbal_prompt"].format(proposition=text),
                          "system_instruction_sha256": key(PROTOCOLS["descriptive-v2"]["system_instruction"]),
                          "response": "0.5", "probability": .5, "completed": True, "exit_code": 0,
                          "finished_at": "2026-09-13T01:00:00Z"}
                (directory / "astra" / (key(text) + ".json")).write_text(json.dumps(record))
        snapshots = self.root / "data/benchmark-runs"
        manifest = refresh(self.root, snapshots, campaigns=["scoped-a", "scoped-z"])
        # Force source-file ordering to disagree with snapshot-file ordering.
        for run in manifest["runs"]:
            if run["run_id"].startswith("scoped-a/"):
                run["source_file"] = "zzz-source"
            elif run["run_id"].startswith("scoped-z/"):
                run["source_file"] = "aaa-source"
        (snapshots / "manifest.json").write_text(json.dumps(manifest))
        runs = load_snapshot(snapshots)[3]
        expected = latest_runs(runs)[("checks", "native/gpt-6-astra@low")]["run_id"]
        self.assertEqual(expected, "scoped-z/astra")
        comparison = comparison_context(manifest, runs, snapshots)
        self.assertEqual(comparison["latest_checks"]["native/gpt-6-astra@low"]["run_id"], expected)
        with sqlite3.connect(rebuild_database(snapshots, self.root / "benchmark.sqlite")) as db:
            self.assertEqual(db.execute('SELECT run_id, target_tuples, scored_count, headline_eligible FROM ranking WHERE requested_model="gpt-6-astra"').fetchone(),
                             (expected, 1, 1, 1))
            self.assertEqual(db.execute('SELECT count(*) FROM ranking WHERE scored_count=0 AND headline_eligible=1').fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
