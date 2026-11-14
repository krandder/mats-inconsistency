"""Guard against false recall matches, hidden missingness and invalid cutoff claims."""
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from harness import run_probes as probes


class ProbeAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "campaign"
        self.catalog = self.root / "probes.jsonl"
        self.cutoffs = self.root / "cutoffs.json"
        self.probe = {"question": "Which laboratory released Example Widget?", "answer": "North Laboratory",
                      "accept": ["North Lab"], "event_date": "2026-07-20", "quarter": "2026Q3",
                      "category": "science", "source": "https://example.test/widget"}
        self.write_catalog([self.probe])
        self.cutoffs.write_text(json.dumps({"example-model": {"advertised": "2026-06", "release": "2026-08",
                                                               "source": "https://example.test/model"}}))
        self.manifest = self.prepare()

    def write_catalog(self, rows):
        self.catalog.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prepare(self, directory=None):
        return probes.prepare(directory or self.directory, self.catalog, self.cutoffs, model="example-model",
                              provider="example-provider", reasoning="low", as_of="2026-09-13")

    def receipt(self, **changes):
        request = probes.read_jsonl(self.directory / "requests.jsonl")[0]
        receipt = {k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "protocol_id", "prompt_sha256")}
        receipt.update(started_at="2026-09-12T10:00:00Z", finished_at="2026-09-12T10:00:01Z",
                       status="completed", response="North Lab", tools_used=False,
                       usage={"input_tokens": 123, "output_tokens": 4},
                       provenance={"transport": "subscription-native-cli", "adapter": "fixture-v1", "session_id": "PRIVATE-SESSION"})
        receipt.update(changes)
        return receipt

    def import_rows(self, rows):
        path = self.root / "input.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return probes.import_receipts(self.directory, path)

    def test_only_explicit_whole_answer_aliases_match(self):
        for response in ("NORTH LAB", "North Laboratory.", '"North Lab"'):
            self.assertTrue(probes.known(response, self.probe), response)
        for response in ("North", "Laboratory", "South Laboratory", "not North Lab", "North Lab or South Lab",
                         "I think North Lab", "North Laboratory was not responsible", "Unknown, perhaps North Lab"):
            self.assertEqual(probes.grade_reply(response, self.probe)["status"], "manual_review", response)

    def test_unknown_and_empty_are_distinct(self):
        self.assertEqual(probes.grade_reply(" UNKNOWN ", self.probe)["status"], "explicit_unknown")
        self.assertEqual(probes.grade_reply("", self.probe)["status"], "malformed_response")
        self.assertEqual(probes.grade_reply("I do not know", self.probe)["status"], "manual_review")

    def test_date_precision_is_not_month_substring_ordering(self):
        relation = probes.temporal_relation
        self.assertEqual(relation("2026-01-05", "2026-01-04"), "after_claimed_cutoff")
        self.assertEqual(relation("2026-01-03", "2026-01-04"), "before_claimed_cutoff")
        self.assertEqual(relation("2026-01", "2026-01-04"), "overlapping_date_precision")
        self.assertEqual(relation("2026-01-04", "2026-01-04"), "overlapping_date_precision")
        self.assertEqual(relation("2026-12", "2026"), "overlapping_date_precision")
        self.assertEqual(relation("2026-02-30", "2026-01"), "invalid_date")
        self.assertEqual(relation("2026-08", None), "cutoff_not_stated")

    def test_future_and_same_day_events_are_not_treated_as_known_past_facts(self):
        day = date(2026, 9, 13)
        self.assertEqual(probes.event_availability("2026-09-14", day), "future_event")
        self.assertEqual(probes.event_availability("2026-09", day), "overlapping_observation_day")
        self.assertEqual(probes.event_availability("2026-09-13", day), "overlapping_observation_day")
        self.assertEqual(probes.event_availability("2026-08", day), "date_elapsed")

    def test_full_catalog_survives_invalid_and_future_rows(self):
        rows = [self.probe, {**self.probe, "question": "Future widget?", "event_date": "2027-01", "quarter": "2027Q1"},
                {**self.probe, "question": "Unknown date widget?", "event_date": "2026-09", "quarter": "2026Q3"},
                {**self.probe, "question": "Invalid widget?", "event_date": "2026-02-30", "quarter": "2026Q1"}]
        self.write_catalog(rows)
        directory = self.root / "full"
        self.prepare(directory)
        report = probes.build_report(directory)
        self.assertEqual(report["catalog_count"], 4)
        self.assertEqual(report["request_ready"], 1)
        self.assertEqual(report["statuses"], {"pending": 1, "probe_requires_review": 3})
        self.assertEqual(len(report["probes"]), 4)
        self.assertFalse(report["full_catalog_completed"])

    def test_prompt_export_does_not_leak_reference_answers(self):
        request = probes.read_jsonl(self.directory / "requests.jsonl")[0]
        self.assertNotIn("North", request["prompt"])
        self.assertNotIn("answer", request)
        self.assertNotIn("accept", request)

    def test_failures_remain_failures_and_usage_survives_retry(self):
        failed = self.receipt(status="infrastructure_error", response="", error="transport interrupted",
                              usage={"input_tokens": 10, "output_tokens": 0})
        self.import_rows([failed])
        report = probes.build_report(self.directory)
        self.assertEqual(report["statuses"], {"infrastructure_error": 1})
        self.assertEqual(report["completed_observations"], 0)
        success = self.receipt(started_at="2026-09-12T10:01:00Z", finished_at="2026-09-12T10:01:01Z")
        self.import_rows([success])
        report = probes.build_report(self.directory)
        self.assertEqual(report["attempt_receipts"], 2)
        self.assertEqual(report["usage_totals_including_failed_attempts"]["input_tokens"], 133)
        self.assertEqual(report["statuses"], {"exact_alias_match": 1})

    def test_resume_is_idempotent_and_does_not_replace_completed_answers(self):
        first = self.receipt(response="UNKNOWN")
        self.assertEqual(self.import_rows([first])["added"], 1)
        before = (self.directory / "receipts.jsonl").read_bytes()
        self.assertEqual(self.import_rows([first])["already_present"], 1)
        self.assertEqual((self.directory / "receipts.jsonl").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "completed observation"):
            self.import_rows([self.receipt(started_at="2026-09-12T11:00:00Z", finished_at="2026-09-12T11:00:01Z")])
        self.assertEqual((self.directory / "receipts.jsonl").read_bytes(), before)

    def test_bad_batch_is_atomic_and_model_effort_prompt_mixing_is_rejected(self):
        for changes in ({"model": "other"}, {"provider": "other"}, {"reasoning": "max"},
                        {"protocol_id": "old"}, {"prompt_sha256": "0" * 64}, {"probe_id": "0" * 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.import_rows([self.receipt(), self.receipt(**changes)])
            self.assertFalse((self.directory / "receipts.jsonl").exists())

    def test_usage_and_provider_provenance_must_be_explicit(self):
        for changes in ({"usage": {"input_tokens": -1}}, {"usage": {"input_tokens": True}},
                        {"provenance": {"transport": "subscription-native-cli", "adapter": "fixture"}},
                        {"tools_used": None}, {"started_at": "2026-09-12T10:00:00"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.import_rows([self.receipt(**changes)])
        self.import_rows([self.receipt(usage=None)])
        report = probes.build_report(self.directory)
        self.assertEqual(report["attempts_with_usage_unavailable"], 1)
        self.assertEqual(report["usage_totals_including_failed_attempts"], {})

    def test_future_at_actual_response_time_is_not_a_post_cutoff_match(self):
        self.import_rows([self.receipt(started_at="2026-07-01T10:00:00Z", finished_at="2026-07-01T10:00:01Z")])
        report = probes.build_report(self.directory)
        self.assertEqual(report["statuses"], {"exact_alias_match": 1})
        self.assertEqual(report["candidate_post_cutoff_matches"], 0)
        self.assertEqual(report["probes"][0]["event_availability_at_observation"], "future_event")

    def test_tool_use_refusal_and_malformed_response_are_not_unknown(self):
        for index, (changes, status) in enumerate([({"tools_used": True}, "tool_use"),
                                                   ({"status": "refusal", "response": "I cannot answer."}, "refusal"),
                                                   ({"response": ""}, "malformed_response")]):
            directory = self.root / str(index)
            self.prepare(directory)
            source = self.root / "input.jsonl"
            source.write_text(json.dumps(self.receipt(**changes)) + "\n")
            probes.import_receipts(directory, source)
            report = probes.build_report(directory)
            self.assertEqual(report["statuses"], {status: 1})
            self.assertEqual(report["candidate_post_cutoff_matches"], 0)

    def test_no_match_or_all_matches_never_certifies_a_cutoff(self):
        for index, response in enumerate(["UNKNOWN", "North Lab"]):
            directory = self.root / ("verdict" + str(index))
            self.prepare(directory)
            source = self.root / "input.jsonl"
            source.write_text(json.dumps(self.receipt(response=response)) + "\n")
            probes.import_receipts(directory, source)
            report = probes.write_report(directory)
            self.assertIsNone(report["verified_cutoff"])
            self.assertIn("not established", (directory / "report.md").read_text())
            self.assertNotIn("cutoff plausible", (directory / "report.md").read_text())

    def test_absent_cutoff_is_not_inferred_from_release_date(self):
        self.cutoffs.write_text(json.dumps({"example-model": {"advertised": None, "release": "2025-01"}}))
        directory = self.root / "no-cutoff"
        self.prepare(directory)
        report = probes.build_report(directory)
        self.assertEqual(report["probes"][0]["cutoff_relation"], "cutoff_not_stated")
        self.assertIsNone(report["verified_cutoff"])

    def test_sources_are_not_verified_by_url_presence_or_claimed_confidence(self):
        review = probes.source_assessment(self.probe)
        self.assertEqual(review["citation_status"], "url_present_unverified")
        self.assertIsNone(review["source_publication_date"])
        self.assertEqual(probes.source_assessment({"source": "authored-from-model-knowledge"})["citation_status"], "no_external_citation")
        report = probes.build_report(self.directory)
        self.assertEqual(report["cutoff_catalog_claims"][0]["verification_status"], "not_independently_verified_by_runner")

    def test_literal_answer_leakage_is_flagged_without_silently_dropping_probe(self):
        probe = {**self.probe, "question": "Did North Lab release the widget?"}
        review = probes.inspect_probe(probe, date(2026, 9, 13))
        self.assertTrue(review["literal_answer_in_question"])
        self.assertTrue(review["request_ready"])

    def test_shared_session_across_probes_is_rejected(self):
        self.write_catalog([self.probe, {**self.probe, "question": "Which laboratory released Other Widget?"}])
        directory = self.root / "two-probes"
        manifest = self.prepare(directory)
        first = self.receipt()
        second = {**first, "probe_id": manifest["probes"][1]["probe_id"],
                  "prompt_sha256": probes.sha(probes.PROBE_PROMPT.format(question="Which laboratory released Other Widget?"))}
        source = self.root / "two-input.jsonl"
        source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
        with self.assertRaisesRegex(ValueError, "reused across probes"):
            probes.import_receipts(directory, source)
        self.assertFalse((directory / "receipts.jsonl").exists())

    def test_raw_text_is_not_truncated_and_report_omits_private_session(self):
        long_reply = "Explanation " * 100 + "not North Lab"
        self.import_rows([self.receipt(response=long_reply)])
        saved = probes.read_jsonl(self.directory / "receipts.jsonl")[0]
        self.assertEqual(saved["receipt"]["response"], long_reply)
        report = probes.write_report(self.directory)
        self.assertEqual(report["statuses"], {"manual_review": 1})
        self.assertNotIn("PRIVATE-SESSION", (self.directory / "report.json").read_text())

    def test_source_or_saved_receipt_mutation_is_detected(self):
        self.write_catalog([{**self.probe, "answer": "Changed Laboratory"}])
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.prepare()
        self.import_rows([self.receipt()])
        path = self.directory / "receipts.jsonl"
        rows = probes.read_jsonl(path)
        rows[0]["receipt"]["response"] = "UNKNOWN"
        path.write_text(json.dumps(rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            probes.build_report(self.directory)

    def test_mock_is_reproducible_across_hash_seeds_and_never_real_evidence(self):
        paths = []
        for seed in ("1", "273"):
            directory = self.root / ("mock" + seed)
            self.prepare(directory)
            subprocess.run([sys.executable, str(Path(probes.__file__)), "mock", "--run-dir", str(directory)],
                           check=True, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed})
            paths.append(directory / "receipts.jsonl")
            report = probes.build_report(directory)
            self.assertTrue(report["synthetic"])
            self.assertFalse(report["full_catalog_completed"])
            self.assertEqual(report["candidate_post_cutoff_matches"], 0)
            self.assertEqual(report["statuses"], {"explicit_unknown": 1})
            self.assertIsNone(report["verified_cutoff"])
        self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_mock_cannot_be_mixed_into_real_campaign(self):
        self.import_rows([self.receipt()])
        with self.assertRaisesRegex(ValueError, "synthetic and imported"):
            probes.mock_receipts(self.directory)

    def fireworks_campaign(self, count=1):
        self.write_catalog([{**self.probe, "question": f"Which laboratory released widget {i}?"} for i in range(count)])
        directory = self.root / "fireworks"
        probes.prepare(directory, self.catalog, self.cutoffs, model="example-model", provider="fireworks",
                       reasoning="low", as_of="2026-09-13")
        return directory

    def frozen_clock(self):
        mock = patch.object(probes, "datetime", wraps=datetime)
        clock = mock.start()
        clock.now.return_value = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
        self.addCleanup(mock.stop)

    @staticmethod
    def fake_provider(payload, **changes):
        data = {"id": "FAKE-" + probes.sha(payload["messages"][0]["content"]), "model": payload["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": "North Lab"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 3}}
        data.update(changes)
        return {"http_status": 200, "data": data}

    def test_live_plan_and_missing_allowance_cannot_initialize_transport(self):
        directory = self.fireworks_campaign()
        with patch.object(probes, "fireworks_requester", side_effect=AssertionError("network initializer called")):
            plan = probes.collect_fireworks(directory)
            self.assertEqual(plan["network_requests"], 0)
            with self.assertRaisesRegex(ValueError, "max-calls"):
                probes.collect_fireworks(directory, execute=True)

    def test_fake_collection_preserves_full_catalog_and_cumulative_allowance(self):
        self.frozen_clock()
        directory = self.fireworks_campaign(2)
        calls = []
        def request(payload):
            calls.append(payload)
            return self.fake_provider(payload)
        first = probes.collect_fireworks(directory, execute=True, max_calls=1, requester=request)
        self.assertEqual(first["status"], "call_allowance_exhausted")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(probes.build_report(directory)["catalog_count"], 2)
        self.assertEqual(probes.build_report(directory)["statuses"], {"exact_alias_match": 1, "pending": 1})
        probes.collect_fireworks(directory, execute=True, max_calls=1, requester=request)
        self.assertEqual(len(calls), 1)
        second = probes.collect_fireworks(directory, execute=True, max_calls=2, requester=request)
        self.assertEqual(second["status"], "complete")
        self.assertEqual(len(calls), 2)
        report = probes.build_report(directory)
        self.assertTrue(report["full_catalog_completed"])
        self.assertEqual(report["inference_requests_performed_by_runner"], 2)
        self.assertIsNone(report["verified_cutoff"])

    def test_fake_provider_error_stops_without_fallback_or_implicit_retry(self):
        self.frozen_clock()
        directory = self.fireworks_campaign()
        calls = []
        def failure(payload):
            calls.append(payload)
            return {"http_status": 429, "data": {"error": "test rate limit"}}
        result = probes.collect_fireworks(directory, execute=True, max_calls=3, requester=failure)
        self.assertEqual(result["status"], "provider_failure")
        self.assertEqual(len(calls), 1)
        report = probes.build_report(directory)
        self.assertEqual(report["statuses"], {"infrastructure_error": 1})
        self.assertEqual(report["attempts_with_usage_unavailable"], 1)
        result = probes.collect_fireworks(directory, execute=True, max_calls=3, requester=failure)
        self.assertEqual(result["status"], "failed_attempt_requires_explicit_retry")
        self.assertEqual(len(calls), 1)
        result = probes.collect_fireworks(directory, execute=True, max_calls=3, retry_errors=True,
                                          requester=self.fake_provider)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(probes.build_report(directory)["attempt_receipts"], 2)

    def test_saved_transport_result_is_recovered_without_another_call(self):
        self.frozen_clock()
        directory = self.fireworks_campaign()
        probes.collect_fireworks(directory, execute=True, max_calls=1, requester=self.fake_provider)
        (directory / "receipts.jsonl").unlink()
        with patch.object(probes, "fireworks_requester", side_effect=AssertionError("unexpected new call")):
            probes.collect_fireworks(directory, execute=True, max_calls=1)
        self.assertEqual(probes.build_report(directory)["attempt_receipts"], 1)

    def test_interrupted_reserved_launch_blocks_blind_resume(self):
        self.frozen_clock()
        directory = self.fireworks_campaign()
        def crash(payload):
            raise RuntimeError("simulated interruption")
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            probes.collect_fireworks(directory, execute=True, max_calls=1, requester=crash)
        report = probes.build_report(directory)
        self.assertEqual(report["launch_reservations"], 1)
        self.assertEqual(report["launches_without_final_receipt"], 1)
        self.assertIsNone(report["inference_requests_performed_by_runner"])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            probes.collect_fireworks(directory, execute=True, max_calls=2, requester=self.fake_provider)

    def test_provider_truncation_and_model_mismatch_do_not_count_as_recall_evidence(self):
        self.frozen_clock()
        directory = self.fireworks_campaign()
        def truncated(payload):
            return self.fake_provider(payload, choices=[{"finish_reason": "length", "message": {"content": "North Lab"}}])
        probes.collect_fireworks(directory, execute=True, max_calls=1, requester=truncated)
        self.assertEqual(probes.build_report(directory)["statuses"], {"truncated_response": 1})
        self.assertEqual(probes.build_report(directory)["candidate_post_cutoff_matches"], 0)
        other = self.root / "model-mismatch"
        probes.prepare(other, self.catalog, self.cutoffs, model="example-model", provider="fireworks",
                       reasoning="low", as_of="2026-09-13")
        probes.collect_fireworks(other, execute=True, max_calls=1,
                                 requester=lambda payload: self.fake_provider(payload, model="unexpected-model"))
        self.assertEqual(probes.build_report(other)["statuses"], {"provenance_review": 1})
        self.assertEqual(probes.build_report(other)["candidate_post_cutoff_matches"], 0)

    def test_collection_can_resume_after_preparation_day_without_relabeling_scope(self):
        self.frozen_clock()
        directory = self.root / "previous-day"
        probes.prepare(directory, self.catalog, self.cutoffs, model="example-model", provider="fireworks",
                       reasoning="low", as_of="2026-09-12")
        probes.collect_fireworks(directory, execute=True, max_calls=1, requester=self.fake_provider)
        report = probes.build_report(directory)
        self.assertEqual(report["as_of"], "2026-09-12")
        self.assertTrue(report["observed_through"].startswith("2026-09-13"))
        self.assertEqual(report["completed_observations"], 1)


if __name__ == "__main__":
    unittest.main()
