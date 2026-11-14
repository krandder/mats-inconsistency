"""Provider token schemas must never inflate totals by double-counting reasoning/cache."""
import json
import io
import hashlib
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.measure_subscription_usage import capture, campaign_receipt_groups, openai_receipt_sessions, interrupted_launches
from scripts.report_subscription_usage import generate, tokens, quota_limits


class UsageTest(unittest.TestCase):
    def test_codex_subsets_are_not_added_twice(self):
        result = tokens("astra", {"input_tokens": 9000, "cached_input_tokens": 7000,
                                  "output_tokens": 100, "reasoning_output_tokens": 90})
        self.assertEqual(result["total_tokens"], 9100)
        self.assertEqual(result["cached_input_tokens"], 7000)

    def test_grok_cache_is_added_but_reasoning_already_in_output(self):
        result = tokens("grok", {"input_tokens": 13912, "cache_read_input_tokens": 128,
                                 "output_tokens": 471, "reasoning_tokens": 464})
        self.assertEqual(result["total_tokens"], 14511)
        self.assertEqual(result["output_tokens"], 471)

    def test_glm_reasoning_is_distinct_from_final_output(self):
        result = tokens("glm", [{"input": 351, "output": 5, "reasoning": 805,
                                 "cache": {"read": 50, "write": 10}}])
        self.assertEqual(result["total_tokens"], 1221)
        self.assertEqual(result["output_tokens"], 810)
        self.assertEqual(result["input_tokens"], 411)

    def test_missing_glm_usage_keeps_all_recorded_total_fields(self):
        self.assertEqual(tokens("glm", None), {"input_tokens": 0, "cached_input_tokens": 0,
                         "output_tokens": 0, "reasoning_output_tokens": 0, "total_tokens": 0})

    def test_both_quota_snapshot_formats(self):
        limits = [{"type": "TOKENS_LIMIT", "percentage": 3}]
        for wrapper in ("data", "response"):
            self.assertEqual(quota_limits({wrapper: {"data": {"limits": limits}}}), limits)


class CampaignUsageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main = self.root / "native-new"
        self.setup = self.root / "diagnostic-a"
        self.other_setup = self.root / "diagnostic-b"
        for directory in (self.main, self.setup, self.other_setup):
            directory.mkdir()
        for label, used, elapsed in (("before", 10, 0), ("after", 12, 10)):
            self.write(self.main / f"quota-astra-{label}.json", {
                "captured_at": 1000000 + elapsed,
                "profiles": [{"profile": "test", "windows": [{"window": "weekly", "used_percent": used,
                              "reset_after_seconds": 5000 - elapsed}]}]})
            self.write(self.main / f"quota-glm-{label}.json", {
                "captured_at": 1000000 + elapsed,
                "response": {"data": {"limits": [{"type": "TOKENS_LIMIT", "unit": 3, "number": 5,
                            "percentage": used, "nextResetTime": 1005000}]}}})
            self.write(self.main / f"quota-grok-{label}.json", {
                "captured_at": 1000000 + elapsed, "used_percent": used, "reset_display": "September 20"})
        self.receipt(self.main, "astra", "primary.json", "main-session", 100, 20)
        self.receipt(self.setup, "astra", "one.json", "diagnostic-session", 30, 5)
        self.receipt(self.setup, "astra", "one.attempt-001.json", "retried-session", 40, 7)
        self.receipt(self.other_setup, "astra", "two.json", "other-session", 10, 2)
        timeout = self.receipt(self.setup, "luna-max", "three.attempt-002.json", "timeout-session", 0, 0)
        timeout.update(usage=None, error="timeout")
        self.write(self.setup / "luna-max/three.attempt-002.json", timeout)
        self.receipt(self.setup, "luna-max", "three.json", "luna-session", 200, 10)
        self.write(self.setup / "glm-probe/four.json", {
            "provider": "glm", "requested_model": "glm-5.3", "usage": [{"input": 5, "output": 2, "reasoning": 3}],
            "elapsed_seconds": 1, "started_at": "2026-09-13T00:00:00Z", "finished_at": "2026-09-13T00:00:01Z"})
        self.write(self.setup / "runtime/config.json", {"provider": {"name": "configuration"}})
        self.write_proxy(["main-session", "diagnostic-session", "retried-session", "other-session", "luna-session", "unrelated-session"])

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def receipt(self, root, run_key, filename, session, input_tokens, output_tokens):
        provider = "luna" if run_key.startswith("luna-") else run_key
        value = {"id": filename.split(".")[0], "provider": provider, "requested_model": "test-model", "session_id": session,
                 "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                 "elapsed_seconds": 1, "started_at": "2026-09-13T00:00:00Z", "finished_at": "2026-09-13T00:00:01Z"}
        self.write(root / run_key / filename, value)
        return value

    def write_proxy(self, sessions):
        (self.main / "proxy-usage-after.jsonl").write_text("".join(
            json.dumps({"session_id": sid, "account": "test-codex", "tokens": {"input": 10, "output": 2}}) + "\n"
            for sid in sessions))

    def previous_report(self, campaign="native-previous", filename="previous.json"):
        totals = {"input_tokens": 900, "cached_input_tokens": 800, "output_tokens": 100,
                  "reasoning_output_tokens": 90, "total_tokens": 1000}
        report = {"campaign": campaign, "as_of": "2026-09-12T00:00:00Z",
                  "total_recorded_native_tokens_including_setup": totals,
                  "configurations": [{"run_key": "astra", "request_receipts": 2, "receipts_with_usage": 2, **totals}],
                  "providers": [{"provider": "historical", "used_delta": 98}]}
        path = self.root / filename
        self.write(path, report)
        return path, report

    def test_previous_reports_add_primary_totals_once_and_leave_current_fields(self):
        previous, source = self.previous_report()
        # An archived report can itself contain cumulative accounting. Only its
        # own campaign contributes when it is supplied again alongside others.
        source["cumulative_recorded_native_tokens_including_setup"] = {"total_tokens": 999999}
        source["cumulative_request_receipts"] = 999
        source["previous_campaigns"] = [{"campaign": "some-earlier-campaign"}]
        self.write(previous, source)
        second, _ = self.previous_report("another-previous", "second.json")
        current = generate(self.main, include_setup_dirs=[self.setup, self.other_setup])
        report = generate(self.main, include_setup_dirs=[self.setup, self.other_setup], previous_reports=[previous, second])
        for key in ("campaign", "as_of", "providers", "configurations", "total_recorded_native_tokens_including_setup", "openai_proxy_attribution"):
            self.assertEqual(report[key], current[key])
        self.assertEqual(report["cumulative_recorded_native_tokens_including_setup"], {
            "input_tokens": 2185, "cached_input_tokens": 1600, "output_tokens": 249,
            "reasoning_output_tokens": 183, "total_tokens": 2434})
        self.assertEqual(report["cumulative_request_receipts"], 11)
        self.assertEqual(report["previous_campaigns"][0], {
            "campaign": "native-previous", "as_of": source["as_of"],
            "total_recorded_native_tokens_including_setup": source["total_recorded_native_tokens_including_setup"],
            "request_receipts": 2, "recorded_cli_launches": 2, "interrupted_launches_without_receipts": 0,
            "native_usage_unknown_count": 0, "source_sha256": hashlib.sha256(previous.read_bytes()).hexdigest()})
        self.assertNotIn("used_delta", json.dumps(report["previous_campaigns"]))
        self.assertEqual(current["previous_campaigns"], [])
        self.assertEqual(current["cumulative_request_receipts"], 7)

    def test_interrupted_launches_do_not_synthesize_receipts_tokens_or_latency(self):
        launch = {"invocation_id": "private-launch-1", "run_key": "astra", "proposition_id": "pending",
                  "session_id": "interrupted-session", "started_at": None, "finished_at": "2026-09-13T01:00:00Z"}
        unidentified = {"invocation_id": "private-launch-2", "run_key": "grok", "proposition_id": None}
        self.write(self.main / "interrupted-invocations.json", {"campaign": "native-new", "invocations": [launch, launch, unidentified]})
        self.write(self.setup / "interrupted-invocations.json", {"invocations": [
            {"invocation_id": "private-launch-3", "run_key": "luna-max", "session_id": "setup-interrupted-session"}]})
        with (self.main / "proxy-usage-after.jsonl").open("a") as handle:
            handle.write(json.dumps({"session_id": "interrupted-session", "account": "test-codex", "tokens": {"input": 40, "output": 5}}) + "\n")
        previous, _ = self.previous_report()
        report = generate(self.main, include_setup_dirs=[self.setup, self.other_setup], previous_reports=[previous])
        self.assertEqual(report["total_recorded_native_tokens_including_setup"]["total_tokens"], 434)
        self.assertEqual(sum(r["request_receipts"] for r in report["configurations"]), 7)
        self.assertEqual(report["recorded_cli_launches"], 10)
        self.assertEqual(report["cumulative_request_receipts"], 9)
        self.assertEqual(report["cumulative_recorded_cli_launches"], 12)
        self.assertEqual(report["cumulative_interrupted_launches_without_receipts"], 3)
        self.assertEqual(report["native_usage_unknown_count"], 4)
        self.assertEqual(report["cumulative_native_usage_unknown_count"], 4)
        interrupted = report["interrupted_invocations"]
        self.assertEqual(interrupted["missing_receipt_launches"], 3)
        self.assertEqual(interrupted["overlap_check_incomplete_count"], 1)
        self.assertEqual(interrupted["proxy_only_usage"]["proxy_total_tokens"], 45)
        self.assertEqual(interrupted["proxy_only_usage"]["known_sessions"], 2)
        self.assertEqual(interrupted["proxy_only_usage"]["launches_without_positive_recorded_proxy_usage"], 2)
        self.assertEqual(report["openai_proxy_attribution"]["matched_proxy_requests"], 5)
        self.assertEqual(report["openai_proxy_attribution"]["accounts"][0]["proxy_input_tokens"], 50)
        self.assertTrue(all("median_seconds" not in row and "total_tokens" not in row for row in interrupted["configurations"]))
        public_json = json.dumps(report)
        for private in ("private-launch", "interrupted-session", "test-codex", str(self.root)):
            self.assertNotIn(private, public_json)

    def test_interruption_receipt_race_is_reconciled_and_ambiguous_match_rejected(self):
        path = self.main / "interrupted-invocations.json"
        self.write(path, {"invocations": [{"run_key": "astra", "session_id": "main-session", "invocation_id": "race"}]})
        report = generate(self.main)
        self.assertEqual(report["recorded_cli_launches"], 1)
        self.assertEqual(report["interrupted_invocations"]["completed_receipt_overlaps_excluded"], 1)
        self.write(path, {"invocations": [{"run_key": "astra", "proposition_id": "primary", "invocation_id": "uncertain"}]})
        with self.assertRaisesRegex(ValueError, "ambiguous completed-receipt overlap"):
            generate(self.main)

    def test_previous_interrupted_only_scopes_and_launches_are_preserved(self):
        previous, source = self.previous_report()
        source["interrupted_invocations"] = {"missing_receipt_launches": 3, "configurations": [
            {"campaign": "shared-setup", "run_key": "setup/shared-setup/grok", "missing_receipt_launches": 3}]}
        source["recorded_cli_launches"] = 5
        self.write(previous, source)
        report = generate(self.main, previous_reports=[previous])
        self.assertEqual(report["cumulative_request_receipts"], 3)
        self.assertEqual(report["cumulative_recorded_cli_launches"], 6)
        self.assertEqual(report["cumulative_native_usage_unknown_count"], 3)
        second, other = self.previous_report("second", "second.json")
        other["configurations"][0].update(campaign="shared-setup", run_key="grok")
        self.write(second, other)
        with self.assertRaisesRegex(ValueError, "overlapping configuration scope"):
            generate(self.main, previous_reports=[previous, second])

    def test_imports_are_observations_without_new_usage_launches_or_proxy_sessions(self):
        imported = self.receipt(self.main, "astra", "imported.json", "imported-session", 900, 100)
        imported["imported_from"] = {"campaign": "native-previous", "run_key": "astra",
                                    "receipt_sha256": "a" * 64, "receipt_id": "source-proposition"}
        self.write(self.main / "astra/imported.json", imported)
        self.write_proxy(["main-session", "imported-session"])
        previous, _ = self.previous_report()
        report = generate(self.main, previous_reports=[previous])
        self.assertEqual(report["imported_observations"], 1)
        self.assertEqual(report["imported_configurations"], [{"run_key": "astra", "imported_observations": 1}])
        self.assertEqual(report["total_recorded_native_tokens_including_setup"]["total_tokens"], 120)
        self.assertEqual(report["configurations"][0]["request_receipts"], 1)
        self.assertEqual(report["recorded_cli_launches"], 1)
        self.assertEqual(report["cumulative_recorded_cli_launches"], 3)
        self.assertEqual(report["cumulative_recorded_native_tokens_including_setup"]["total_tokens"], 1120)
        self.assertEqual(openai_receipt_sessions(campaign_receipt_groups(self.main)), {"main-session": "astra"})
        self.assertEqual(report["openai_proxy_attribution"]["matched_proxy_requests"], 1)
        imported["imported_from"] = {}
        self.write(self.main / "astra/imported.json", imported)
        with self.assertRaisesRegex(ValueError, "imported receipts require"):
            generate(self.main)

    def test_duplicate_report_campaigns_are_rejected(self):
        previous, _ = self.previous_report()
        with self.assertRaisesRegex(ValueError, "duplicate campaign ID"):
            generate(self.main, previous_reports=[previous, previous])
        previous, _ = self.previous_report(campaign="native-new")
        with self.assertRaisesRegex(ValueError, "duplicate campaign ID"):
            generate(self.main, previous_reports=[previous])

    def test_overlapping_diagnostic_scope_uses_receipt_campaign_metadata(self):
        for path in (self.setup / "astra").glob("*.json"):
            receipt = json.loads(path.read_text())
            receipt["campaign"] = "actual-diagnostic-campaign"
            self.write(path, receipt)
        previous, source = self.previous_report()
        source["configurations"][0].update(run_key="setup/a-renamed-directory/astra", campaign="actual-diagnostic-campaign")
        self.write(previous, source)
        report = generate(self.main, include_setup_dirs=[self.setup])
        row = next(row for row in report["configurations"] if row["run_key"] == "setup/diagnostic-a/astra")
        self.assertEqual(row["campaign"], "actual-diagnostic-campaign")
        with self.assertRaisesRegex(ValueError, "overlapping configuration scope"):
            generate(self.main, include_setup_dirs=[self.setup], previous_reports=[previous])
        # Legacy reports without explicit campaign metadata still have a
        # resolvable setup directory prefix.
        source["configurations"][0].pop("campaign")
        source["configurations"][0]["run_key"] = "setup/diagnostic-a/luna-max"
        self.write(previous, source)
        with self.assertRaisesRegex(ValueError, "overlapping configuration scope"):
            generate(self.main, include_setup_dirs=[self.setup], previous_reports=[previous])

    def test_overlap_between_previous_reports_is_rejected(self):
        previous, source = self.previous_report()
        second, other = self.previous_report("another-previous", "second.json")
        for path, value in ((previous, source), (second, other)):
            value["configurations"][0].update(run_key="setup/shared-diagnostic/astra")
            self.write(path, value)
        with self.assertRaisesRegex(ValueError, "overlapping configuration scope"):
            generate(self.main, previous_reports=[previous, second])

    def test_previous_report_double_counted_subsets_or_mismatched_totals_are_rejected(self):
        previous, source = self.previous_report()
        source["total_recorded_native_tokens_including_setup"]["total_tokens"] += 800
        self.write(previous, source)
        with self.assertRaisesRegex(ValueError, "without adding cache or reasoning"):
            generate(self.main, previous_reports=[previous])
        previous, source = self.previous_report()
        source["configurations"][0]["input_tokens"] -= 1
        self.write(previous, source)
        with self.assertRaisesRegex(ValueError, "configuration input_tokens sum"):
            generate(self.main, previous_reports=[previous])

    def test_setup_attempts_and_tokens_are_counted_once_and_excluded(self):
        report = generate(self.main, include_setup_dirs=[self.setup, self.other_setup, self.setup])
        rows = {row["run_key"]: row for row in report["configurations"]}
        self.assertEqual(report["campaign"], "native-new")
        self.assertEqual(rows["astra"]["category"], "benchmark (including reusable pilot)")
        diagnostic = rows["setup/diagnostic-a/astra"]
        self.assertEqual(diagnostic["category"], "excluded diagnostic/setup")
        self.assertEqual(diagnostic["request_receipts"], 2)
        self.assertEqual(diagnostic["total_tokens"], 82)
        self.assertEqual(rows["setup/diagnostic-a/luna-max"]["request_receipts"], 2)
        self.assertEqual(rows["setup/diagnostic-a/luna-max"]["receipts_with_usage"], 1)
        self.assertEqual(rows["setup/diagnostic-b/astra"]["total_tokens"], 12)
        self.assertEqual(rows["setup/diagnostic-a/glm-probe"]["total_tokens"], 10)
        self.assertEqual(sum(row["request_receipts"] for row in rows.values()), 7)
        self.assertEqual(report["total_recorded_native_tokens_including_setup"]["total_tokens"], 434)
        self.assertEqual(len(rows), 5)
        # Extra directories have no quota snapshots: deltas only use the main baseline.
        self.assertEqual([row["used_delta"] for row in report["providers"]], [2, 2, 2])

    def test_session_attribution_uses_setup_prefix_and_includes_attempts(self):
        sessions = openai_receipt_sessions(campaign_receipt_groups(self.main, [self.setup, self.other_setup]))
        self.assertEqual(sessions, {
            "main-session": "astra", "diagnostic-session": "setup/diagnostic-a/astra",
            "retried-session": "setup/diagnostic-a/astra", "other-session": "setup/diagnostic-b/astra",
            "timeout-session": "setup/diagnostic-a/luna-max", "luna-session": "setup/diagnostic-a/luna-max"})
        report = generate(self.main, include_setup_dirs=[self.setup, self.other_setup])
        attribution = report["openai_proxy_attribution"]
        self.assertEqual(attribution["receipt_sessions"], 6)
        self.assertEqual(attribution["matched_sessions"], 5)
        self.assertEqual(attribution["unmatched_sessions"], 1)
        self.assertEqual(attribution["matched_proxy_requests"], 5)
        self.assertEqual(attribution["accounts"][0]["requests_by_configuration"], {
            "astra": 1, "setup/diagnostic-a/astra": 2, "setup/diagnostic-a/luna-max": 1,
            "setup/diagnostic-b/astra": 1})
        self.assertEqual(attribution["accounts"][0]["proxy_input_tokens"], 50)
        public_json = json.dumps(report)
        self.assertNotIn("test-codex", public_json)
        self.assertNotIn("diagnostic-session", public_json)
        self.assertNotIn(str(self.root), public_json)

    def test_original_default_campaign_and_preflight_are_preserved(self):
        original = self.root / "native-20260912"
        self.main.rename(original)
        self.main = original
        self.write(self.main / "glm-preflight-steps1/one.json", {
            "provider": "glm", "requested_model": "glm-5.3", "usage": [{"input": 1, "output": 1}],
            "elapsed_seconds": 1, "started_at": "2026-09-13T00:00:00Z", "finished_at": "2026-09-13T00:00:01Z"})
        report = generate(original)
        self.assertEqual(report["campaign"], "native-20260912")
        self.assertEqual([r["run_key"] for r in report["configurations"]], ["astra", "glm-preflight-steps1"])
        self.assertEqual(report["configurations"][1]["category"], "excluded setup")
        self.assertEqual(report["total_recorded_native_tokens_including_setup"]["total_tokens"], 122)
        self.assertEqual(generate(original, campaign="explicit-name")["campaign"], "explicit-name")

    def test_capture_passes_all_campaign_sessions_and_keeps_snapshots_in_main(self):
        original_read = Path.read_text
        baseline = (self.main / "quota-astra-before.json").read_text()
        grok_after = (self.main / "quota-grok-after.json").read_text()
        self.write(self.main / "interrupted-invocations.json", {"invocations": [
            {"invocation_id": "capture-interrupted", "run_key": "astra", "session_id": "capture-interrupted-session"}]})

        def read_text(path, *args, **kwargs):
            if path == Path("/home/agent/.local/share/opencode/auth.json"):
                return json.dumps({"zai-coding-plan": {"key": "test-only"}})
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", autospec=True, side_effect=read_text), \
                patch("scripts.measure_subscription_usage.urllib.request.urlopen", return_value=io.StringIO("{}")), \
                patch("scripts.measure_subscription_usage.remote_python", side_effect=['{"profiles": []}', "matched usage\n"]) as remote, \
                redirect_stdout(io.StringIO()):
            capture(self.main, "after", include_setup_dirs=[self.setup, self.other_setup])
        proxy_call = remote.call_args_list[1]
        self.assertEqual(proxy_call.args[0], "prime-any-proxy-1")
        self.assertTrue(proxy_call.kwargs["sudo"])
        self.assertEqual(set(json.loads(proxy_call.args[2])), {
            "main-session", "diagnostic-session", "retried-session", "timeout-session", "other-session", "luna-session", "capture-interrupted-session"})
        self.assertEqual((self.main / "proxy-usage-after.jsonl").read_text(), "matched usage\n")
        self.assertEqual((self.main / "quota-astra-before.json").read_text(), baseline)
        self.assertEqual((self.main / "quota-grok-after.json").read_text(), grok_after)
        self.assertFalse(list(self.setup.glob("quota-*.json")))

    def test_included_diagnostics_do_not_hide_quota_resets(self):
        for provider in ("astra", "glm", "grok"):
            path = self.main / f"quota-{provider}-after.json"
            snapshot = json.loads(path.read_text())
            if provider == "astra":
                snapshot["profiles"][0]["windows"][0]["reset_after_seconds"] += 1000
            elif provider == "glm":
                snapshot["response"]["data"]["limits"][0]["nextResetTime"] += 1000
            else:
                snapshot["reset_display"] = "September 27"
            self.write(path, snapshot)
        report = generate(self.main, include_setup_dirs=[self.setup])
        self.assertEqual([row["used_delta"] for row in report["providers"]], [None, None, None])

    def test_ambiguous_group_or_session_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "main campaign"):
            list(campaign_receipt_groups(self.main, [self.main]))
        with self.assertRaisesRegex(ValueError, "does not exist"):
            list(campaign_receipt_groups(self.main, [self.root / "missing"]))
        duplicate_name = self.root / "nested/diagnostic-a"
        duplicate_name.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            list(campaign_receipt_groups(self.main, [self.setup, duplicate_name]))
        self.receipt(self.setup, "astra", "collision.json", "main-session", 1, 1)
        with self.assertRaisesRegex(ValueError, "multiple configuration groups"):
            openai_receipt_sessions(campaign_receipt_groups(self.main, [self.setup]))


if __name__ == "__main__":
    unittest.main()
