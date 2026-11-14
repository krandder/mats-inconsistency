"""Offline public-export checks with synthetic native streams, never providers."""
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from harness import run_native_probes as native
from harness import run_probes as probes
from scripts import export_native_probes as exporter


class NativeProbeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_tmp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_tmp.name)
        cls.make_fixture(cls.fixture)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(self.fixture, self.root, dirs_exist_ok=True)
        self.run = self.root / "runs"
        self.catalog = self.root / "catalog.jsonl"
        self.audit = self.root / "audit.json"

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(probes.canonical(value) + "\n")

    @classmethod
    def make_fixture(cls, root):
        catalog = [{"question": f"Who released widget {i}?", "answer": "North Lab", "accept": ["North Laboratory"],
                    "event_date": "2025-01-01", "quarter": "2025Q1", "category": "science",
                    "source": "https://example.org/widget"} for i in range(129)]
        catalog_path = root / "catalog.jsonl"
        catalog_path.write_text("".join(probes.canonical(p) + "\n" for p in catalog))
        cls.write(root / "cutoffs.json", {})
        audit_rows = []
        for i, p in enumerate(catalog, 1):
            audit_rows.append({"catalog_index": i, "probe_id": probes.sha(probes.canonical(p)), "original_probe": p,
                "answer_evidence_level": "primary", "core_answer_status": "corroborated_in_inspected_sources",
                "evidence": [{"capture_id": "synthetic-capture", "url": "https://example.org/widget", "excerpt": "North Lab released widgets.",
                              "relationship": "corroborates_core_answer"}],
                "date_flags": [], "prompt_flags": [], "grading_flags": [], "proposed_corrections": []})
        cls.write(root / "audit.json", {"schema_version": 1, "catalog_sha256": probes.sha(catalog_path.read_bytes()),
                  "probes": audit_rows, "raw_capture_directory": "/PRIVATE-CAPTURE-PATH", "inventory": {"catalog_count": 129}})
        for provider, effort in exporter.SETTINGS:
            directory = root / "runs" / (provider + "-" + effort)
            manifest = native.prepare(directory, provider=provider, effort=effort, as_of="2026-09-13",
                                      probes_path=catalog_path, cutoffs_path=root / "cutoffs.json")
            metadata = native.protocol_metadata(manifest)
            ledger, launches = [], []
            for i, request in enumerate(probes.read_jsonl(directory / "requests.jsonl")):
                identity = f"PRIVATE-SESSION-{provider}-{effort}-{i}"
                attempt = probes.sha(identity)
                start = "2026-09-13T00:00:00Z"
                launch = {"attempt_id": attempt, **{k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "prompt_sha256")},
                          "started_at": start, "native_protocol_fingerprint": metadata["fingerprint"],
                          "receipt_file": f"collection/attempts/{attempt}.jsonl", "final_file": f"collection/attempts/{attempt}.final.txt"}
                final = directory / launch["final_file"]
                final.parent.mkdir(parents=True, exist_ok=True)
                response = ["North Lab", "UNKNOWN", "South Lab"][i % 3]
                model = "zai-coding-plan/glm-5.3" if provider == "glm" else manifest["model"]
                command = ["PRIVATE-NATIVE-PATH", "--model", model]
                if provider in {"astra", "luna"}:
                    final.write_text(response)
                    command += ["-c", "model_reasoning_effort=" + json.dumps(effort),
                                "-c", "developer_instructions=" + json.dumps(native.SYSTEM_INSTRUCTION)]
                    events = [{"type": "thread.started", "thread_id": identity}, {"type": "turn.started"},
                              {"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                              {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 6,
                                  "cached_input_tokens": 5, "reasoning_output_tokens": 4}}]
                    stdout = "\n".join(json.dumps(e) for e in events)
                elif provider == "grok":
                    command += ["--reasoning-effort", effort, "--system-prompt-override", native.SYSTEM_INSTRUCTION, "-p", request["prompt"]]
                    stdout = json.dumps({"text": response, "stopReason": "end_turn", "sessionId": identity,
                        "requestId": "PRIVATE-REQUEST-" + str(i), "model": "grok-4.6-build", "num_turns": 1,
                        "usage": {"input_tokens": 10, "output_tokens": 8, "reasoning_tokens": 6},
                        "modelUsage": {"grok-4.6-build": {"inputTokens": 10, "outputTokens": 8,
                            "costUSD": 0.1, "account_name": "PRIVATE-ACCOUNT-NAME"}}})
                else:
                    command += ["--agent", "factual-probe", request["prompt"]]
                    stdout = "\n".join(json.dumps(e) for e in [
                        {"type": "step_start", "sessionID": identity, "part": {}},
                        {"type": "text", "sessionID": identity, "part": {"text": response}},
                        {"type": "step_finish", "sessionID": identity, "part": {"reason": "stop", "tokens": {
                            "input": 10, "output": 2, "reasoning": 8, "cache": {"read": 5, "write": 3}, "total": 28}}}])
                decoded = native.validate_output(provider, stdout, final, 0)
                usage, issue = native.normalize_usage(provider, decoded["raw_usage"])
                receipt = {**{k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "protocol_id", "prompt_sha256")},
                    "requested_model": manifest["model"], "collector": native.COLLECTOR,
                    "system_instruction_sha256": metadata["system_instruction_sha256"], "native_protocol_fingerprint": metadata["fingerprint"],
                    "started_at": start, "finished_at": "2026-09-13T00:00:01Z", "elapsed_seconds": 1, "exit_code": 0,
                    "status": "completed", "completed": True, "usage": usage, "usage_validation_issue": issue,
                    "provider_model_mismatch": False, "runtime_profile": "PRIVATE-ACCOUNT-PROFILE",
                    "provenance": {"transport": "subscription-native-cli", "adapter": native.COLLECTOR,
                                   "session_id": identity, "local_attempt_id": attempt, "request_id": identity + "-request"},
                    "native_evidence": {"stdout": stdout, "stderr": "PRIVATE-STDERR", "command": command, "process_id": 12345}, **decoded}
                attempt_path = directory / launch["receipt_file"]
                cls.write(attempt_path, receipt)
                ledger.append({"receipt_sha256": probes.sha(probes.canonical(receipt)), "import_file_sha256": probes.sha(attempt_path.read_bytes()), "receipt": receipt})
                launches.append(launch)
            (directory / "receipts.jsonl").write_text("".join(probes.canonical(r) + "\n" for r in ledger))
            (directory / "collection/launches.jsonl").write_text("".join(probes.canonical(r) + "\n" for r in launches))

    def export(self, **kwargs):
        return exporter.build_export(self.run, self.audit, catalog_path=self.catalog, **kwargs)

    def change_receipt(self, mutation):
        directory = self.run / "astra-low"
        ledger = probes.read_jsonl(directory / "receipts.jsonl")
        mutation(ledger[0]["receipt"])
        receipt = ledger[0]["receipt"]
        launch = probes.read_jsonl(directory / "collection/launches.jsonl")[0]
        path = directory / launch["receipt_file"]
        self.write(path, receipt)
        ledger[0].update(receipt_sha256=probes.sha(probes.canonical(receipt)), import_file_sha256=probes.sha(path.read_bytes()))
        (directory / "receipts.jsonl").write_text("".join(probes.canonical(r) + "\n" for r in ledger))

    def test_full_coverage_and_grades_recomputed_offline(self):
        with patch.object(native, "collect", side_effect=AssertionError("provider launch forbidden")), \
             patch.object(native, "report", side_effect=AssertionError("private report mutation forbidden")):
            report = self.export()
        self.assertEqual(report["completed_observations"], 1161)
        self.assertEqual(report["publication_status"], "COMPLETE")
        self.assertIsNone(report["verified_cutoff"])
        self.assertTrue(all(c["statuses"] == {"exact_alias_match": 43, "explicit_unknown": 43, "manual_review": 43} for c in report["configurations"]))
        self.assertEqual(len(report["configurations"][0]["probes"]), 129)

    def test_missing_observation_blocks_publication_but_preview_retains_pending(self):
        path = self.run / "astra-low/receipts.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "129 terminal"):
            self.export()
        report = self.export(allow_incomplete=True)
        self.assertEqual(report["publication_status"], "INCOMPLETE_LOCAL_PREVIEW")
        self.assertEqual(report["completed_observations"], 1160)
        self.assertEqual(report["configurations"][0]["probes"][-1]["original_key_grade"]["status"], "pending")
        self.assertEqual(report["configurations"][0]["unreconciled_launches"], 1)

    def test_missing_or_duplicated_treatment_rejected_even_in_preview(self):
        shutil.copytree(self.run / "astra-low", self.run / "clone")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.export(allow_incomplete=True)
        shutil.rmtree(self.run / "clone")
        shutil.rmtree(self.run / "luna-max")
        with self.assertRaisesRegex(ValueError, "All nine"):
            self.export(allow_incomplete=True)

    def test_private_fields_removed_and_html_escaped(self):
        audit = json.loads(self.audit.read_text())
        audit["probes"][0]["evidence"][0]["session_id"] = "PRIVATE-SESSION-in-source-overlay"
        audit["inventory"]["account"] = "PRIVATE-ACCOUNT-IN-INVENTORY"
        self.write(self.audit, audit)
        report = self.export()
        report["configurations"][0]["probes"][0]["response"] = "</td><script>alert('x')</script>"
        encoded = json.dumps(report)
        html = exporter.html_report(report)
        markdown = exporter.markdown(report)
        for private in ("PRIVATE-SESSION", "PRIVATE-REQUEST", "PRIVATE-ACCOUNT", "PRIVATE-CAPTURE", "PRIVATE-NATIVE", "PRIVATE-STDERR", "provider_request_id", '"session_id"', '"stdout"'):
            self.assertNotIn(private, encoded + html + markdown)
        self.assertNotIn("<script>alert('x')</script>", html + markdown)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("private_receipt_sha256", encoded)
        self.assertEqual(html.count("class='full'"), 9)
        self.assertIn("factual-recall-usage.json", html)

    def test_unexpected_model_metadata_is_hashed_without_private_key_leak(self):
        directory = self.run / "grok-low"
        envelope = probes.read_jsonl(directory / "receipts.jsonl")[0]
        receipt = envelope["receipt"]
        secret = "PRIVATE-ACCOUNT-IN-MODEL-KEY"
        receipt["raw_model_usage"][secret] = {"inputTokens": 77}
        receipt["native_validation"]["reported_models"].append(secret)
        _, metadata = native.load_campaign(directory)
        public = exporter.sanitized_attempt(envelope, 1, metadata)
        self.assertNotIn(secret, json.dumps(public))
        self.assertIn(probes.sha(secret), public["native_validation"]["unrecognized_reported_model_sha256"])

    def test_saved_request_prompt_tampering_is_rejected(self):
        path = self.run / "astra-low/requests.jsonl"
        rows = probes.read_jsonl(path)
        rows[0]["prompt"] = "Substituted question"
        path.write_text("".join(probes.canonical(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "Saved native requests"):
            self.export()

    def test_source_correction_is_preserved_without_changing_original_key(self):
        audit = json.loads(self.audit.read_text())
        audit["probes"][0]["proposed_corrections"] = [{"field": "answer", "original_value": "North Lab", "proposed_value": "South Lab",
            "rationale": "Synthetic source correction", "evidence_capture_ids": ["synthetic-capture"], "status": "proposed_not_applied"}]
        self.write(self.audit, audit)
        report = self.export()
        row = report["configurations"][0]["probes"][0]
        self.assertEqual(row["original_key_grade"]["status"], "exact_alias_match")
        self.assertEqual(row["original_probe"]["answer"], "North Lab")
        self.assertEqual(row["source_review"]["proposed_corrections"][0]["proposed_value"], "South Lab")

    def test_rehashed_response_cannot_override_native_stream(self):
        self.change_receipt(lambda r: r.update(response="FAKE new answer"))
        with self.assertRaisesRegex(ValueError, "recomputation mismatch: response"):
            self.export()

    def test_rehashed_usage_cannot_override_native_usage(self):
        self.change_receipt(lambda r: r["usage"].update(total_tokens=99999))
        with self.assertRaisesRegex(ValueError, "usage recomputation"):
            self.export()

    def test_rehashed_wrong_reasoning_command_rejected(self):
        def mutate(r):
            c = r["native_evidence"]["command"]
            c[c.index('model_reasoning_effort="low"')] = 'model_reasoning_effort="max"'
        self.change_receipt(mutate)
        with self.assertRaisesRegex(ValueError, "reasoning selection"):
            self.export()

    def test_attempt_hash_mismatch_and_source_scope_mismatch_rejected(self):
        directory = self.run / "astra-low"
        launch = probes.read_jsonl(directory / "collection/launches.jsonl")[0]
        path = directory / launch["receipt_file"]
        old = path.read_text()
        path.write_text(old + "\n")
        with self.assertRaisesRegex(ValueError, "attempt file hash"):
            self.export()
        path.write_text(old)
        audit = json.loads(self.audit.read_text())
        audit["probes"].pop()
        self.write(self.audit, audit)
        with self.assertRaisesRegex(ValueError, "all 129"):
            self.export()

    def test_failed_attempt_and_usage_retained_before_terminal_retry(self):
        directory = self.run / "astra-low"
        manifest, metadata = native.load_campaign(directory)
        ledger = probes.read_jsonl(directory / "receipts.jsonl")
        launches = probes.read_jsonl(directory / "collection/launches.jsonl")
        receipt, launch = copy.deepcopy(ledger[0]["receipt"]), copy.deepcopy(launches[0])
        attempt = probes.sha("synthetic failed prior attempt")
        session = "PRIVATE-SESSION-failed-prior-attempt"
        launch.update(attempt_id=attempt, started_at="2026-09-12T23:59:58Z",
                      receipt_file=f"collection/attempts/{attempt}.jsonl", final_file=f"collection/attempts/{attempt}.final.txt")
        final = directory / launch["final_file"]
        final.write_text(receipt["response"])
        stdout = receipt["native_evidence"]["stdout"].replace(receipt["session_id"], session)
        decoded = native.validate_output("astra", stdout, final, 1)
        receipt.update(decoded)
        receipt.update(status="infrastructure_error", completed=False, error="native_exit:1", exit_code=1,
                       started_at=launch["started_at"], finished_at="2026-09-12T23:59:59Z")
        receipt["native_evidence"]["stdout"] = stdout
        receipt["provenance"].update(local_attempt_id=attempt, session_id=session, request_id=session + "-request")
        path = directory / launch["receipt_file"]
        self.write(path, receipt)
        envelope = {"receipt": receipt, "receipt_sha256": probes.sha(probes.canonical(receipt)),
                    "import_file_sha256": probes.sha(path.read_bytes())}
        (directory / "receipts.jsonl").write_text("".join(probes.canonical(r) + "\n" for r in [envelope, *ledger]))
        (directory / "collection/launches.jsonl").write_text("".join(probes.canonical(r) + "\n" for r in [launch, *launches]))
        report = self.export()
        astra = report["configurations"][0]
        self.assertEqual(astra["completed_observations"], 129)
        self.assertEqual(astra["attempt_receipts"], 130)
        self.assertEqual(astra["usage_totals_including_failed_attempts"]["total_tokens"], 26 * 130)
        self.assertEqual([a["status"] for a in astra["probes"][0]["attempts"]], ["infrastructure_error", "completed"])
        self.assertEqual(astra["probes"][0]["original_key_grade"]["status"], "exact_alias_match")

    def test_incomplete_cli_cannot_write_any_publication_path(self):
        with self.assertRaises(SystemExit), patch("sys.stderr", new_callable=io.StringIO), \
             patch.object(exporter, "build_export", side_effect=AssertionError("should reject before reading")):
            exporter.main(["--allow-incomplete", "--json-output", str(self.root / "preview.json")])


if __name__ == "__main__":
    unittest.main()
