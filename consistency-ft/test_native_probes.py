"""Native factual collection: fake subprocesses and temporary evidence only."""
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from harness import run_native_probes as collector
from harness import run_probes as probes


class NativeProbeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.catalog = self.root / "catalog.jsonl"
        self.cutoffs = self.root / "cutoffs.json"
        self.cutoffs.write_text("{}")
        self.counter = itertools.count()
        self.calls, self.commands = [], []
        self.clock = patch.object(collector, "datetime", wraps=datetime)
        clock = self.clock.start()
        clock.now.return_value = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)
        self.addCleanup(self.clock.stop)
        self.environment = patch.dict(collector.os.environ, {"TMPDIR": str(self.root), "CONSISTENCY_CODEX_RUNTIME": "fixture-runtime"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def prepare(self, provider="astra", effort=None, count=3, name=None):
        rows = [{"question": f"Which lab created widget {i}?", "answer": "North Lab", "accept": ["North Laboratory"],
                 "event_date": "2026-07-20", "quarter": "2026Q3", "category": "science", "source": "https://example.test/widget"}
                for i in range(count)]
        self.catalog.write_text("".join(json.dumps(r) + "\n" for r in rows))
        directory = self.root / (name or f"{provider}-{effort or 'default'}")
        collector.prepare(directory, provider=provider, effort=effort, as_of="2026-09-13",
                          probes_path=self.catalog, cutoffs_path=self.cutoffs)
        return directory

    def command(self, provider, prompt, empty, final, effort, *, rule, campaign):
        self.commands.append({"provider": provider, "prompt": prompt, "rule": rule, "effort": effort, "empty": empty})
        model = "zai-coding-plan/glm-5.3" if provider == "glm" else collector.native.MODELS[provider]
        command = ["FAKE-NATIVE", "--model", model]
        env = {"FAKE_PROVIDER": provider, "FAKE_FINAL": str(final), "CONSISTENCY_CODEX_RUNTIME": "fixture-runtime"}
        if provider in {"astra", "luna"}:
            command += ["-c", "model_reasoning_effort=" + json.dumps(effort),
                        "-c", "developer_instructions=" + json.dumps(rule)]
        elif provider == "grok":
            command += ["--reasoning-effort", effort, "--system-prompt-override", rule]
        else:
            command += ["--agent", "forecast"]
            env["OPENCODE_CONFIG_CONTENT"] = json.dumps({"agent": {"forecast": {"prompt": rule, "description": "old", "tools": {"*": False}}}})
        return command, env, prompt

    def payload(self, provider, ident, response, mode):
        session = None if mode == "no_session" else "PRIVATE-FAKE-SESSION-" + ident
        if provider in {"astra", "luna"}:
            events = [{"type": "thread.started", "thread_id": session}, {"type": "turn.started"}]
            if mode == "reconnect":
                events.append({"type": "error", "message": collector.RECONNECT})
            if mode == "tools":
                events.append({"type": "item.completed", "item": {"type": "command_execution", "command": "FAKE"}})
            events += [{"type": "item.completed", "item": {"type": "agent_message", "text": response}},
                       {"type": "turn.completed", "usage": {"input_tokens": 20, "cached_input_tokens": 5,
                                                             "output_tokens": 6, "reasoning_output_tokens": 4}}]
            if mode == "fatal_after_final":
                events.append({"type": "error", "message": "unrecognized failure"})
            return "\n".join(json.dumps(e) for e in events)
        if provider == "grok":
            return json.dumps({"text": response, "stopReason": "end_turn", "sessionId": session,
                               "requestId": "FAKE-REQUEST-" + ident, "num_turns": 1, "model": "grok-4.6",
                               "usage": {"input_tokens": 10, "cache_read_input_tokens": 5,
                                         "cache_creation_input_tokens": 2, "output_tokens": 8, "reasoning_tokens": 6},
                               "modelUsage": {"grok-4.6": {"opaque": "retained"}}})
        events = [{"type": "step_start", "sessionID": session, "part": {}},
                  {"type": "text", "sessionID": session, "part": {"text": response}},
                  {"type": "step_finish", "sessionID": session, "part": {"reason": "stop", "tokens": {
                      "input": 10, "output": 2, "reasoning": 8, "cache": {"read": 5, "write": 3}, "total": 28}}}]
        return "\n".join(json.dumps(e) for e in events)

    def popen(self, *, mode="normal", response="North Lab", delay=0):
        owner = self
        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = next(owner.counter) + 10000
                self.returncode = 1 if mode == "exit_failure" else 0
                self.kwargs = kwargs
                self.first = True
                owner.calls.append((command, kwargs))
                owner.assertEqual(list(Path(kwargs["cwd"]).iterdir()), [])
                owner.assertTrue(kwargs["start_new_session"])
            def communicate(self, *args, **kwargs):
                if delay:
                    time.sleep(delay)
                if mode == "timeout" and self.first:
                    self.first = False
                    raise subprocess.TimeoutExpired("FAKE", 1)
                env = self.kwargs["env"]
                final = Path(env["FAKE_FINAL"])
                if env["FAKE_PROVIDER"] in {"astra", "luna"}:
                    final.write_text("different text" if mode == "mismatch" else response)
                stdout = owner.payload(env["FAKE_PROVIDER"], str(self.pid), response, mode)
                return stdout, "FAKE STDERR retained"
        return FakeProcess

    def run_collection(self, directory, *, max_calls=1, mode="normal", response="North Lab", workers=1, retry_errors=False, delay=0):
        with patch.object(collector.native, "native_command", side_effect=self.command), \
             patch.object(collector.subprocess, "Popen", side_effect=self.popen(mode=mode, response=response, delay=delay)), \
             patch.object(collector.native, "call", side_effect=AssertionError("forecast collector used")), \
             patch.object(collector.native, "parse_readout", side_effect=AssertionError("probability parser used")):
            return collector.collect(directory, execute=True, max_calls=max_calls, workers=workers, retry_errors=retry_errors)

    def test_supported_nine_treatments_are_exact(self):
        settings = [(p, e) for p, efforts in collector.EFFORTS.items() for e in efforts]
        self.assertEqual(len(settings), 9)
        for provider, effort in settings:
            directory = self.prepare(provider, effort, count=1)
            manifest, metadata = collector.load_campaign(directory)
            self.assertEqual(manifest["model"], collector.native.MODELS[provider])
            self.assertEqual(manifest["reasoning"], effort)
            self.assertEqual(metadata["system_instruction"], collector.SYSTEM_INSTRUCTION)
        for provider, effort in [("astra", "max"), ("grok", "none"), ("glm", "low"), ("luna", "ultra")]:
            with self.assertRaises(ValueError):
                collector.treatment(provider, effort)

    def test_offline_full_catalog_plan_never_initializes_native_client(self):
        directory = self.root / "real-source-plan"
        with patch.object(collector.native, "native_command", side_effect=AssertionError("native command invoked")), \
             patch.object(collector.subprocess, "Popen", side_effect=AssertionError("native process launched")):
            collector.prepare(directory, provider="astra", as_of="2026-09-13")
            plan = collector.collect(directory)
        self.assertEqual(plan["catalog_count"], 129)
        self.assertEqual(plan["request_ready"], 129)
        self.assertEqual(plan["completed_observations"], 0)
        self.assertEqual(plan["new_launches"], 0)

    def test_one_call_pilot_resumes_without_shrinking_scope_or_reasking(self):
        directory = self.prepare(count=3)
        first = self.run_collection(directory)
        self.assertEqual(first["status"], "call_allowance_exhausted")
        self.assertEqual(first["catalog_count"], 3)
        self.assertEqual(first["completed_observations"], 1)
        second = self.run_collection(directory, max_calls=3, workers=2)
        self.assertEqual(second["completed_observations"], 3)
        self.assertEqual(second["new_launches"], 2)
        third = self.run_collection(directory, max_calls=3)
        self.assertEqual(third["new_launches"], 0)
        self.assertEqual(len(self.calls), 3)
        for command in self.commands:
            self.assertEqual(command["rule"], collector.SYSTEM_INSTRUCTION)
            self.assertNotIn("North Lab", command["prompt"])
            self.assertFalse(command["empty"].exists())

    def test_each_provider_preserves_raw_and_normalized_usage(self):
        expected_totals = {"astra": 26, "luna": 26, "grok": 25, "glm": 28}
        for provider, total in expected_totals.items():
            directory = self.prepare(provider, count=1)
            self.run_collection(directory)
            saved = probes.read_jsonl(directory / "receipts.jsonl")[0]["receipt"]
            self.assertEqual(saved["usage"]["total_tokens"], total)
            self.assertTrue(saved["raw_usage"])
            self.assertIn("PRIVATE-FAKE-SESSION", saved["session_id"])
            self.assertEqual(saved["provenance"]["session_id"], saved["session_id"])
            self.assertEqual(saved["native_evidence"]["stderr"], "FAKE STDERR retained")
            self.assertIn("North Lab", saved["native_evidence"]["stdout"])
            self.assertEqual(saved["exit_code"], 0)
            self.assertIsNotNone(saved["elapsed_seconds"])
            self.assertTrue(saved["native_validation"]["no_tools_reported"])
            if provider == "grok":
                self.assertEqual(saved["raw_model_usage"], {"grok-4.6": {"opaque": "retained"}})
            if provider == "glm":
                self.assertEqual(saved["usage"]["input_tokens"], 18)
                self.assertEqual(saved["usage"]["output_tokens"], 10)
                self.assertIn("factual-probe", saved["native_evidence"]["command"])

    def test_unknown_and_unmatched_are_completed_and_not_retried(self):
        for index, response in enumerate(["UNKNOWN", "not North Lab"]):
            directory = self.prepare(count=1, name=str(index))
            self.run_collection(directory, response=response)
            report = collector.report(directory)
            expected = "explicit_unknown" if index == 0 else "manual_review"
            self.assertEqual(report["statuses"], {expected: 1})
            self.assertIsNone(report["verified_cutoff"])
            again = self.run_collection(directory, max_calls=2, retry_errors=True)
            self.assertEqual(again["new_launches"], 0)

    def test_tools_are_flagged_without_retrying_a_completed_answer(self):
        directory = self.prepare(count=1)
        self.run_collection(directory, mode="tools")
        self.assertEqual(collector.report(directory)["statuses"], {"tool_use": 1})
        self.assertEqual(self.run_collection(directory, max_calls=2, retry_errors=True)["new_launches"], 0)

    def test_exit_failure_remains_infrastructure_until_explicit_retry(self):
        directory = self.prepare(count=2)
        result = self.run_collection(directory, max_calls=2, mode="exit_failure", response="UNKNOWN")
        self.assertEqual(result["status"], "provider_failure")
        self.assertEqual(result["new_launches"], 1)
        self.assertEqual(collector.report(directory)["statuses"], {"infrastructure_error": 1, "pending": 1})
        result = self.run_collection(directory, max_calls=3)
        self.assertEqual(result["status"], "failed_attempt_requires_explicit_retry")
        self.assertEqual(result["new_launches"], 0)
        result = self.run_collection(directory, max_calls=3, retry_errors=True)
        self.assertEqual(result["completed_observations"], 2)
        self.assertEqual(collector.report(directory)["attempt_receipts"], 3)

    def test_completion_session_and_final_file_mismatches_fail_closed(self):
        for mode in ("no_session", "mismatch", "fatal_after_final"):
            directory = self.prepare(count=1, name=mode)
            result = self.run_collection(directory, mode=mode)
            self.assertEqual(result["statuses"], {"infrastructure_error": 1})
            self.assertEqual(result["completed_observations"], 0)

    def test_known_reconnect_requires_later_matching_final_and_completion(self):
        directory = self.prepare(count=1)
        self.run_collection(directory, mode="reconnect")
        record = probes.read_jsonl(directory / "receipts.jsonl")[0]["receipt"]
        self.assertTrue(record["completed"])
        self.assertEqual(record["native_validation"]["recovered_reconnections"], 1)

    def test_timeout_kills_only_its_process_group_and_saves_failure(self):
        directory = self.prepare(count=1)
        with patch.object(collector.os, "killpg") as kill:
            result = self.run_collection(directory, mode="timeout")
        kill.assert_called_once_with(10000, collector.signal.SIGKILL)
        self.assertEqual(result["status"], "provider_failure")
        record = probes.read_jsonl(directory / "receipts.jsonl")[0]["receipt"]
        self.assertEqual(record["exit_code"], "timeout")
        self.assertTrue(record["native_evidence"]["stdout"])

    def test_saved_attempt_is_recovered_without_another_subprocess(self):
        directory = self.prepare(count=1)
        self.run_collection(directory)
        (directory / "receipts.jsonl").unlink()
        with patch.object(collector.subprocess, "Popen", side_effect=AssertionError("duplicate native call")):
            result = collector.collect(directory, execute=True, max_calls=1)
        self.assertEqual(result["completed_observations"], 1)
        self.assertEqual(result["new_launches"], 0)

    def test_interrupted_launch_is_visible_and_blocks_blind_resume(self):
        directory = self.prepare(count=1)
        with patch.object(collector, "run_attempt", side_effect=RuntimeError("simulated interruption")):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                collector.collect(directory, execute=True, max_calls=1)
        report = collector.report(directory)
        self.assertEqual(report["launches_without_final_receipt"], 1)
        self.assertEqual(report["native_usage_unavailable"], 1)
        self.assertIsNone(report["inference_requests_performed_by_runner"])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            collector.collect(directory, execute=True, max_calls=2)

    def test_invalid_or_missing_usage_is_unknown_not_zero(self):
        for provider, usage in [("astra", None), ("grok", {}), ("glm", []),
                                ("astra", {"input_tokens": -1, "output_tokens": 2}),
                                ("glm", [{"input": 4}])]:
            normalized, issue = collector.normalize_usage(provider, usage)
            self.assertIsNone(normalized)
            self.assertTrue(issue)

    def test_nested_optional_usage_metadata_is_retained_without_losing_valid_counts(self):
        raw = {"input_tokens": 20, "output_tokens": 6, "cached_input_tokens": 5,
               "input_tokens_details": {"cached_tokens": 5}, "provider_note": "metadata"}
        normalized, issue = collector.normalize_usage("astra", raw)
        self.assertIsNone(issue)
        self.assertEqual(normalized["total_tokens"], 26)
        self.assertEqual(raw["input_tokens_details"], {"cached_tokens": 5})

    def test_known_grok_backend_alias_is_frozen_before_collection(self):
        directory = self.prepare("grok", count=1)
        manifest, metadata = collector.load_campaign(directory)
        self.assertEqual(metadata["reported_model_aliases"], ["grok-4.6", "grok-4.6-build"])
        payload = self.payload("grok", "fake", "North Lab", "normal")
        body = json.loads(payload)
        body["model"] = "grok-4.6-build"
        body["modelUsage"] = {"grok-4.6-build": {"opaque": "unchanged"}}
        final = self.root / "unused.final.txt"
        observed = collector.validate_output("grok", json.dumps(body), final, 0)
        self.assertEqual(observed["native_validation"]["reported_models"], ["grok-4.6-build"])
        self.assertFalse(any(m not in metadata["reported_model_aliases"] for m in observed["native_validation"]["reported_models"]))
        body["model"] = "unrelated-model"
        observed = collector.validate_output("grok", json.dumps(body), final, 0)
        self.assertTrue(any(m not in metadata["reported_model_aliases"] for m in observed["native_validation"]["reported_models"]))

    def test_protocol_mutation_and_runtime_switch_block_resume(self):
        directory = self.prepare(count=1)
        path = directory / "native-protocol.json"
        before = path.read_text()
        meta = json.loads(before)
        meta["system_instruction"] = "changed"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            collector.collect(directory)
        path.write_text(before)
        self.run_collection(directory)
        with patch.dict(collector.os.environ, {"CONSISTENCY_CODEX_RUNTIME": "different-runtime"}):
            with self.assertRaisesRegex(ValueError, "runtime treatment"):
                collector.collect(directory, execute=True, max_calls=2)

    def test_allowances_are_required_and_workers_are_bounded(self):
        directory = self.prepare(count=1)
        for kwargs in ({}, {"max_calls": 0}, {"max_calls": 1, "workers": 17}, {"max_calls": 1, "timeout": float("nan")}):
            with self.assertRaises(ValueError):
                collector.collect(directory, execute=True, **kwargs)
        self.assertFalse((directory / "collection/launches.jsonl").exists())

    def test_workers_and_call_allowance_bound_parallel_launches(self):
        directory = self.prepare(count=8)
        result = self.run_collection(directory, max_calls=3, workers=2, delay=.02)
        self.assertEqual(result["new_launches"], 3)
        self.assertEqual(result["completed_observations"], 3)
        self.assertEqual(result["catalog_count"], 8)
        self.assertEqual(len(self.calls), 3)


if __name__ == "__main__":
    unittest.main()
