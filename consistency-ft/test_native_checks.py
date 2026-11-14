"""Guard against outcome-dependent retries and mixing reasoning treatments."""
import json
import os
import importlib.machinery
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness.run_native_checks import (
    CLARIFIED_RULE, PROTOCOLS, VERBAL_PROMPT, call, classify_readout, decode, is_completed, key,
    GLM_RUNTIME_PROFILE, empty_working_directory, main, native_command, parse_probability, parse_readout, register_campaign, write_results,
)


class NativeChecksTest(unittest.TestCase):
    def test_glm_runtime_is_external_and_disables_background_indexing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").write_text("gitdir: elsewhere")
            wrapper = root / "fleet-provider-fixture"
            wrapper.write_text('def command(provider, args, env):\n'
                               '    env = {k: v for k, v in env.items() if not k.startswith("OPENCODE_")}\n'
                               '    return ["opencode", "--model", "zai-coding-plan/glm-5.3"] + args, env\n')
            source_loader = importlib.machinery.SourceFileLoader
            with patch.dict(os.environ, {"TMPDIR": str(root)}), \
                 patch("harness.run_native_checks.importlib.machinery.SourceFileLoader",
                       side_effect=lambda name, path: source_loader(name, str(wrapper))):
                empty = empty_working_directory("glm", repo / "runs/campaign")
                self.assertEqual(empty.parent, root.resolve())
                self.assertEqual(list(empty.iterdir()), [])
                cmd, env, _ = native_command("glm", "question", empty, root / "out", "native-default",
                                             rule=CLARIFIED_RULE, campaign="scoped")
                self.assertEqual(cmd[cmd.index("--dir") + 1], str(empty))
                self.assertIn("zai-coding-plan/glm-5.3", cmd)
                self.assertEqual(cmd[-1], "question")
                self.assertEqual(env["OPENCODE_DISABLE_FFF"], "true")
                self.assertEqual(env["OPENCODE_EXPERIMENTAL_FILEWATCHER"], "false")
                self.assertEqual(env["OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER"], "true")
                config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
                self.assertFalse(config["snapshot"])
                self.assertEqual(config["agent"]["forecast"]["prompt"], CLARIFIED_RULE)
                directory = repo / "receipts"
                directory.mkdir()
                result = SimpleNamespace(returncode=0, communicate=lambda *a, **kw: ('{"type":"text","part":{"text":"0.25"}}', ''))
                with patch("harness.run_native_checks.subprocess.Popen", return_value=result) as launch:
                    record = call("glm", "question", directory, empty, 1, "native-default", campaign="scoped", protocol="descriptive-v2")
                self.assertEqual(record["runtime_profile"], GLM_RUNTIME_PROFILE)
                from scripts.export_benchmark import sanitize_receipt
                self.assertEqual(sanitize_receipt(record, Path("receipt.json"), "glm-5.3", "native-default")["runtime_profile"], GLM_RUNTIME_PROFILE)
                self.assertEqual(launch.call_args.kwargs["cwd"], empty)
                self.assertEqual(Path(launch.call_args.kwargs["env"]["TMPDIR"]).parent.parent, root.resolve())
            with patch.dict(os.environ, {"TMPDIR": str(repo)}):
                with self.assertRaisesRegex(ValueError, "outside Git"):
                    empty_working_directory("glm", repo / "runs/campaign")
            (empty / "unexpected").touch()
            with patch.dict(os.environ, {"TMPDIR": str(root)}):
                with self.assertRaisesRegex(ValueError, "remain empty"):
                    empty_working_directory("glm", repo / "runs/campaign")

    def test_diagnostic_abstentions_do_not_write_benchmark_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "data/tuples").mkdir(parents=True)
            row = {"id": "neg-1", "check_type": "Negation", "questions": {"P": {"title": "A"}, "not_P": {"title": "not A"}}}
            (directory / "data/tuples/Negation.jsonl").write_text(json.dumps(row) + "\n")
            def observed(provider, text, *args, **kwargs):
                return {"id": key(text), "probability": None, "response": "ABSTAIN", "completed": True,
                        "readout_status": "abstention", "error_kind": "abstention", "elapsed_seconds": .1}
            argv = ["runner", "astra", "--campaign", "test-diagnostic", "--protocol", "descriptive-v2",
                    "--diagnostic", "--pilot", "1", "--run-dir", str(directory / "runs/test-diagnostic")]
            with patch("harness.run_native_checks.ROOT", directory), patch("sys.argv", argv), \
                 patch("harness.run_native_checks.call", side_effect=observed):
                self.assertEqual(main(), 0)
            campaign_dir = directory / "runs/test-diagnostic"
            meta = json.loads((campaign_dir / "campaign.json").read_text())
            self.assertEqual(meta["target_propositions"], 2)
            self.assertEqual(meta["category"], "diagnostic")
            self.assertFalse(list((directory / "results").glob("*.jsonl")))
            status = json.loads((campaign_dir / "status-astra.json").read_text())
            self.assertEqual(status["failed"], 0)
            self.assertEqual(status["missing_probabilities"], 1)
            self.assertEqual(status["scored_tuples"], 0)
            with self.assertRaisesRegex(ValueError, "campaign metadata differs"):
                register_campaign(campaign_dir, {**meta, "category": "benchmark"})

    def test_all_observed_missing_readouts_are_explicit_refusals(self):
        root = Path(__file__).resolve().parent / "data/benchmark-runs"
        missing = []
        for path in root.glob("native-20260912-*.receipts.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["is_current"] and row.get("probability") is None:
                    missing.append(row)
        self.assertEqual(len(missing), 157)
        for row in missing:
            self.assertEqual(classify_readout(row["response"]), "refusal", row["attempt_id"])
            with self.assertRaises(ValueError):
                parse_readout(row["response"], "descriptive-v2")

    def test_explicit_abstention_is_valid_but_never_a_probability(self):
        self.assertIsNone(parse_readout(" ABSTAIN\n", "descriptive-v2"))
        self.assertIsNone(parse_readout("abstain", "descriptive-v2"))
        with self.assertRaises(ValueError):
            parse_readout("ABSTAIN", "numeric-v1")
        for response in ("ABSTAIN (not 0.5)", "I cannot provide that. 0.5", "0.4 or ABSTAIN"):
            with self.assertRaises(ValueError):
                parse_readout(response, "descriptive-v2")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            result = SimpleNamespace(returncode=0, communicate=lambda *a, **kw: ('{"text":"ABSTAIN"}', ''))
            with patch("harness.run_native_checks.native_command", return_value=(["native"], {}, None)), \
                 patch("harness.run_native_checks.subprocess.Popen", return_value=result):
                record = call("grok", "question", directory, directory, 1,
                              campaign="diagnostic-v2", protocol="descriptive-v2")
            self.assertTrue(record["completed"])
            self.assertIsNone(record["probability"])
            self.assertEqual(record["readout_status"], "abstention")
            self.assertNotIn("error", record)
            with patch("harness.run_native_checks.subprocess.Popen") as launch:
                self.assertEqual(call("grok", "question", directory, directory, 1,
                                      campaign="diagnostic-v2", protocol="descriptive-v2"), record)
                launch.assert_not_called()

    def test_protocol_and_campaign_reach_client_and_reject_cache_mixing(self):
        command, env, _ = native_command("luna", "question", Path("/tmp"), Path("/tmp/out"), "max",
                                         rule=CLARIFIED_RULE, campaign="native-new")
        self.assertIn("developer_instructions=" + json.dumps(CLARIFIED_RULE), command)
        self.assertEqual(env["FLEET_ACTOR"], "consistency-native-new")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = {"id": key("question"), "requested_model": "gpt-5.6-luna", "reasoning": "max",
                      "campaign": "different-campaign", "protocol_id": "descriptive-v2",
                      "prompt": PROTOCOLS["descriptive-v2"]["verbal_prompt"].format(proposition="question"),
                      "system_instruction_sha256": key(CLARIFIED_RULE), "probability": .4, "completed": True}
            (directory / (record["id"] + ".json")).write_text(json.dumps(record))
            with patch("harness.run_native_checks.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "incompatible cached"):
                    call("luna", "question", directory, directory, 1, "max", campaign="native-new", protocol="descriptive-v2")
                launch.assert_not_called()

    def test_client_temporary_files_are_cleaned_but_receipt_is_kept(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            client_paths = []
            def launch(command, **options):
                client_tmp = Path(options["env"]["TMPDIR"])
                client_paths.append(client_tmp)
                (client_tmp / "native-library.so").write_bytes(b"temporary client file")
                return SimpleNamespace(returncode=0, communicate=lambda *a, **kw: ('{"text":"0.25"}', ''))
            with patch("harness.run_native_checks.native_command", return_value=(["native"], {}, None)), \
                 patch("harness.run_native_checks.subprocess.Popen", side_effect=launch):
                record = call("grok", "question", directory, directory, 1)
            self.assertEqual(record["probability"], .25)
            self.assertTrue((directory / (key("question") + ".json")).exists())
            self.assertFalse(client_paths[0].exists())

    def test_probability_requires_entire_response(self):
        for value in ("0", "0.32", "1.000", " 0.4\n"):
            self.assertGreaterEqual(parse_probability(value), 0)
        for value in ("I refuse, but 0.3", "50%", "1.2", "-0.1", "0.4 and 0.8", ""):
            with self.assertRaises(ValueError):
                parse_probability(value)

    def test_abstention_is_reused_without_calling_model(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = {"id": key("question"), "exit_code": 0,
                      "requested_model": "gpt-6-astra", "prompt": VERBAL_PROMPT.format(proposition="question"),
                      "response": "I cannot estimate this.", "error": "format"}
            (directory / (record["id"] + ".json")).write_text(json.dumps(record))
            with patch("harness.run_native_checks.subprocess.Popen") as launch:
                self.assertEqual(call("astra", "question", directory, directory, 1), record)
                launch.assert_not_called()
            self.assertFalse(is_completed({"exit_code": "timeout", "response": ""}))

    def test_resume_rejects_a_different_reasoning_treatment(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = {"id": key("question"), "exit_code": 0, "probability": 0.5,
                      "requested_model": "gpt-5.6-luna", "reasoning": "low",
                      "prompt": VERBAL_PROMPT.format(proposition="question")}
            (directory / (record["id"] + ".json")).write_text(json.dumps(record))
            with patch("harness.run_native_checks.subprocess.Popen") as launch:
                with self.assertRaisesRegex(ValueError, "incompatible cached"):
                    call("luna", "question", directory, directory, 1, "high")
                launch.assert_not_called()

    def test_luna_effort_reaches_native_client(self):
        for effort in ("none", "low", "medium", "high", "xhigh", "max"):
            cmd, env, _ = native_command("luna", "question", Path("/tmp"), Path("/tmp/out"), effort)
            self.assertIn('model_reasoning_effort="' + effort + '"', cmd)
            self.assertIn("gpt-5.6-luna", cmd)
            self.assertNotIn("TMUX", env)
            self.assertEqual(env["FLEET_ACTOR"], "consistency-native-20260912")

    def test_missing_probability_never_becomes_scored_tuple(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rows.jsonl"
            tuples = [{"id": "a", "check_type": "Negation", "questions": {
                "P": {"title": "P"}, "not_P": {"title": "not P"}}}]
            records = {key("P"): {"probability": 0.5}, key("not P"): {"error": "abstention"}}
            self.assertEqual(write_results("luna", tuples, records, path, "high", "luna-high"), 0)
            records[key("not P")] = {"probability": 0.3}
            self.assertEqual(write_results("luna", tuples, records, path, "high", "luna-high"), 1)
            row = json.loads(path.read_text())
            self.assertEqual(row["model"], "native/gpt-5.6-luna/high")
            self.assertEqual(row["reasoning"], "high")
            self.assertAlmostEqual(row["violation"], 0.2)


if __name__ == "__main__":
    unittest.main()
