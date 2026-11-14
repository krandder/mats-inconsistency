"""Durable collector fixtures; never load or sample a pretrained model."""
from collections import namedtuple
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from anchors import load_snapshot
from training import qwen_base_collection as c

TokenResult = namedtuple("TokenResult", "completion_token_ids token_logprobs finish_reason")


class FixtureFailure(RuntimeError):
    def __init__(self, partial):
        super().__init__("explicit fixture interruption")
        self.partial = partial


class FixtureTokenizer:
    def verify_runtime(self):
        return {"fixture": True}

    def render_and_encode(self, messages):
        return json.dumps(messages), (1, 2, 3)

    def decode_completion(self, tokens):
        return "0.0" if tokens[0] == 15 else "1.0"


class FixtureRuntime:
    def __init__(self, plan, *, fail_after=None, finish_reason="eos"):
        self.plan, self.fail_after, self.finish = plan, fail_after, finish_reason
        self.calls = 0

    def policy_sha256(self):
        return self.plan["initial_policy_sha256"]

    def generate_tokens(self, prompt, *, generator, eos_token_id, max_new_tokens, on_token):
        self.calls += 1
        tokens = [15 if generator["seed"] % 2 == 0 else 16, 13, 15, eos_token_id]
        sampled, logprobs = [], []
        try:
            for index, token in enumerate(tokens):
                generator["counter"] += 1
                sampled.append(token)
                logprobs.append(-.1)
                on_token({"token_index": index, "token_id": token, "sampled_logprob": -.1,
                          "is_eos": token == eos_token_id})
                if len(sampled) == self.fail_after:
                    raise RuntimeError("explicit fixture stop after retained token")
        except BaseException:
            raise FixtureFailure(TokenResult(tuple(sampled), tuple(logprobs), "error"))
        return TokenResult(tuple(sampled), tuple(logprobs), self.finish)


def fixture_spec():
    def pin(path):
        return {"path": str(path), "sha256": c.sha256(path.read_bytes())}
    reference = c.ROOT / "data/research/qwen-numerical-backward-20260913/identity-before.json"
    execution = json.loads(reference.read_text())["execution"]
    execution["runtime_source_sha256"] = c.source_identity()["training/qwen_lora_runtime.py"]
    execution["categorical_sampler"] = c.categorical_sampler_identity()
    return {"schema_version": 1, "collection_id": "explicit-software-fixture-not-candidate-observations",
            "master_seed": 12345, "max_new_tokens": 32, "protocol_id": "numeric-v1",
            "protocol": __import__("harness.run_native_checks", fromlist=["PROTOCOLS"]).PROTOCOLS["numeric-v1"],
            "sources": c.source_identity(), "execution": execution,
            "initial_runtime_identity": pin(reference),
            "checkpoint_manifest": pin(c.ROOT / "data/research/qwen-checkpoint-artifacts-20260913.json"),
            "tokenizer_manifest": pin(c.ROOT / "data/research/qwen-tokenizer-artifacts-20260913.json"),
            **{name: pin(c.ROOT / "data/episodes" / (name + ".jsonl")) for name in c.SPLIT_HASHES}}


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.spec = fixture_spec()
        self.spec_path = self.root / "spec.json"
        self.spec_path.write_bytes(c.canonical_bytes(self.spec))
        self.directory = self.root / "collection"
        self.plan_hash = c.create_plan(self.spec_path, c.sha256(self.spec_path.read_bytes()), self.directory)
        self.plan, self.spec, self.rows = c.load_plan(self.directory, self.plan_hash)

    def ledger(self, directory=None, plan_hash=None):
        return c.AttemptLedger(directory or self.directory, self.rows,
                               binding=(self.plan, self.spec, plan_hash or self.plan_hash))

    def run_batch(self, *, directory=None, plan_hash=None, max_attempts=2, token_budget=64, runtime=None):
        directory, plan_hash = directory or self.directory, plan_hash or self.plan_hash
        runtime = runtime or FixtureRuntime(self.plan)
        with c.owner_lock(directory):
            result = c.run_tranche(self.ledger(directory, plan_hash), self.plan, self.spec, plan_hash, "fixture-invocation",
                runtime=runtime, tokenizer=FixtureTokenizer(), generator_factory=lambda seed: {"seed": seed, "counter": 0},
                rng_digest=c.digest, max_attempts=max_attempts, token_budget=token_budget,
                deadline=time.monotonic() + 120, verify_inputs=lambda: None, phase=lambda *args, **kwargs: None)
        return runtime, result

    def test_full_catalog_and_seed_plan_are_pure_and_original(self):
        self.assertEqual(len(self.rows), 2926)
        self.assertEqual(self.plan["target_episodes"], 1619)
        self.assertEqual([r["id"] for r in self.rows], sorted(r["id"] for r in self.rows))
        self.assertEqual(len({r["seed"] for r in self.rows}), 2926)
        self.assertTrue(all(c.proposition_id(r["text"]) == r["id"] for r in self.rows))
        command = "from training.qwen_base_collection import load_plan; import sys; load_plan(sys.argv[1],sys.argv[2]); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", command, str(self.directory), self.plan_hash], cwd=c.ROOT, check=True)

    def test_live_integer_config_keys_match_the_pinned_json_identity(self):
        actual = json.loads(Path(self.spec["initial_runtime_identity"]["path"]).read_text())
        actual["execution"] = copy.deepcopy(self.spec["execution"])
        actual["execution"]["model_config"]["id2label"] = {
            int(key): value for key, value in actual["execution"]["model_config"]["id2label"].items()}
        actual["execution_sha256"] = c.digest(actual["execution"])
        self.assertNotEqual(actual["execution"], self.spec["execution"])
        self.assertEqual(actual["execution_sha256"], self.plan["execution_sha256"])
        kwargs = {"policy_sha256": self.plan["initial_policy_sha256"],
                  "sampling_distribution": c.SAMPLING, "zero_initial_b": True}
        c.validate_initial_runtime_identity(actual, self.plan, self.spec, **kwargs)
        changed = copy.deepcopy(actual)
        changed["execution"]["cpu_threads"] = 3
        with self.assertRaisesRegex(ValueError, "execution_canonical_content"):
            c.validate_initial_runtime_identity(changed, self.plan, self.spec, **kwargs)
        changed = copy.deepcopy(actual)
        changed["execution_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "execution_sha256"):
            c.validate_initial_runtime_identity(changed, self.plan, self.spec, **kwargs)
        with self.assertRaisesRegex(ValueError, "sampling_distribution"):
            c.validate_initial_runtime_identity(actual, self.plan, self.spec,
                **dict(kwargs, sampling_distribution=dict(c.SAMPLING, top_k=1)))
        with self.assertRaisesRegex(ValueError, "zero_initial_B"):
            c.validate_initial_runtime_identity(actual, self.plan, self.spec, **dict(kwargs, zero_initial_b=False))

    def test_pins_sources_split_and_scientific_defaults_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "external immutable plan"):
            c.load_plan(self.directory, "0" * 64)
        inputs = {key: c.pinned_input(self.spec, key) for key in
                  ("initial_runtime_identity", "checkpoint_manifest", "tokenizer_manifest", "train", "valid", "heldout")}
        for key, value, message in (("sources", {}, "source differs"), ("lambda_anchor", 2., "spec fields")):
            changed = copy.deepcopy(self.spec)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, message):
                c.validate_spec(changed, inputs)
        changed_inputs = dict(inputs, train=inputs["train"].splitlines(keepends=True)[0])
        with self.assertRaisesRegex(ValueError, "complete original split"):
            c.validate_spec(self.spec, changed_inputs)

    def test_categorical_source_and_execution_pins_are_required(self):
        inputs = {key: c.pinned_input(self.spec, key) for key in
                  ("initial_runtime_identity", "checkpoint_manifest", "tokenizer_manifest", "train", "valid", "heldout")}
        self.assertIn("training/categorical.py", self.spec["sources"])
        self.assertEqual(self.spec["execution"]["categorical_sampler"], c.categorical_sampler_identity())
        changed = copy.deepcopy(self.spec)
        del changed["sources"]["training/categorical.py"]
        with self.assertRaisesRegex(ValueError, "source differs"):
            c.validate_spec(changed, inputs)
        changed = copy.deepcopy(self.spec)
        del changed["execution"]["categorical_sampler"]
        with self.assertRaisesRegex(ValueError, "explicit execution"):
            c.validate_spec(changed, inputs)
        changed = copy.deepcopy(self.spec)
        changed["execution"]["categorical_sampler"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "explicit execution"):
            c.validate_spec(changed, inputs)

    def test_lock_excludes_second_owner_and_status_reader(self):
        with c.owner_lock(self.directory):
            for shared in (False, True):
                with self.assertRaises(BlockingIOError), c.owner_lock(self.directory, shared=shared):
                    pass

    def test_existing_empty_or_partial_attempt_never_retried(self):
        row = self.rows[0]
        path = self.directory / "attempts" / row["id"]
        c.make_directory(path)
        (path / "tokens.jsonl").write_bytes(b'{"token_id":15')
        before = (path / "tokens.jsonl").read_bytes()
        runtime, result = self.run_batch(max_attempts=1)
        self.assertEqual(runtime.calls, 1)
        self.assertTrue((self.directory / "attempts" / self.rows[1]["id"] / "receipt.json").exists())
        self.assertFalse((path / "receipt.json").exists())
        self.assertEqual((path / "tokens.jsonl").read_bytes(), before)
        summary = self.ledger().inspect()
        self.assertEqual(summary["attempted"], 2)
        self.assertEqual(summary["states"][row["id"]], "interrupted-or-incomplete")
        self.assertFalse(summary["complete_valid_coverage"])

    def test_same_seed_tokens_rng_across_one_or_separate_invocations(self):
        other = self.root / "second-collection"
        other_hash = c.create_plan(self.spec_path, c.sha256(self.spec_path.read_bytes()), other)
        self.run_batch(max_attempts=2)
        self.run_batch(directory=other, plan_hash=other_hash, max_attempts=1)
        self.run_batch(directory=other, plan_hash=other_hash, max_attempts=1)
        for row in self.rows[:2]:
            first = json.loads((self.directory / "attempts" / row["id"] / "receipt.json").read_text())
            second = json.loads((other / "attempts" / row["id"] / "receipt.json").read_text())
            for key in ("seed", "completion_token_ids", "token_logprobs", "rng_before_sha256", "rng_after_sha256"):
                self.assertEqual(first[key], second[key])

    def test_full_cap_admission_never_shrinks_treatment(self):
        runtime, result = self.run_batch(token_budget=31)
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(result["tokens_this_invocation"], 0)
        runtime, result = self.run_batch(token_budget=35)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(result["remaining_token_budget"], 31)
        self.assertEqual(result["fixed_per_proposition_cap"], 32)

    def test_no_eligible_attempt_does_not_initialize_runtime(self):
        with patch.object(c, "initialize_runtime", side_effect=AssertionError("must not load weights")):
            result = c.collect(self.directory, self.plan_hash, checkpoint=Path("absent"), tokenizer_artifacts=Path("absent"),
                               max_attempts=1, token_budget=31, wall_seconds=1, cpu_soft_seconds=2,
                               cpu_hard_seconds=3, address_space_bytes=36 * 2**30)
        self.assertEqual(result["status"], "no-eligible-new-attempt")
        self.assertFalse(result["weights_loaded"])

    def test_callback_failure_retains_token_and_resume_skips_failed_id(self):
        with self.assertRaisesRegex(RuntimeError, "all partial evidence retained"):
            self.run_batch(runtime=FixtureRuntime(self.plan, fail_after=1))
        failed = self.directory / "attempts" / self.rows[0]["id"]
        receipt = json.loads((failed / "receipt.json").read_text())
        self.assertEqual(len(receipt["completion_token_ids"]), 1)
        self.assertEqual(len((failed / "tokens.jsonl").read_bytes().splitlines()), 1)
        self.assertFalse(receipt["runtime_integrity_verified"])
        runtime, _ = self.run_batch(max_attempts=1)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(self.ledger().inspect()["valid_probabilities"], 1)

    def test_invalid_claimed_probability_never_inflates_coverage(self):
        self.run_batch(max_attempts=1)
        path = self.directory / "attempts" / self.rows[0]["id"] / "receipt.json"
        original = json.loads(path.read_text())
        for key, value in (("runtime_integrity_verified", False), ("finish_reason", "length"), ("probability", .7)):
            modified = dict(original, **{key: value})
            path.chmod(0o644)
            path.write_bytes(c.canonical_bytes(modified) + b"\n")
            summary = self.ledger().inspect()
            self.assertEqual(summary["valid_probabilities"], 0)
            self.assertEqual(summary["states"][self.rows[0]["id"]], "invalid-probability")

    def test_changed_seed_and_token_evidence_are_detected(self):
        self.run_batch(max_attempts=1)
        folder = self.directory / "attempts" / self.rows[0]["id"]
        path = folder / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt["seed"] += 1
        path.chmod(0o644)
        path.write_bytes(c.canonical_bytes(receipt) + b"\n")
        self.assertEqual(self.ledger().inspect()["states"][self.rows[0]["id"]], "corrupt-retained-attempt")

    def test_partial_export_and_implicit_treatments_rejected(self):
        self.run_batch(max_attempts=1)
        with self.assertRaisesRegex(ValueError, "1/2926"):
            c.export_anchor(self.directory, self.plan_hash, self.root / "export", mode="exact")
        self.assertFalse((self.root / "export").exists())
        with self.assertRaises(ValueError):
            c.export_anchor(self.directory, self.plan_hash, self.root / "export", mode="symmetric-mixture")
        with self.assertRaises(ValueError):
            c.export_anchor(self.directory, self.plan_hash, self.root / "export", mode="exact", smoothing_epsilon=.01)

    def test_full_explicit_fixture_export_is_compatible_with_existing_snapshot(self):
        # These are declared software fixtures, not candidate observations.
        # Other tests exercise fsync; disable only the expensive disk flush for
        # this 2926-row format bridge fixture, preserving actual file structure.
        with patch.object(c.os, "fsync"):
            self.run_batch(max_attempts=2926, token_budget=2926 * 32)
        self.assertTrue(self.ledger().inspect()["complete_valid_coverage"])
        for mode, epsilon in (("exact", None), ("symmetric-mixture", .01)):
            output = self.root / mode
            snapshot = c.export_anchor(self.directory, self.plan_hash, output, mode=mode, smoothing_epsilon=epsilon)
            frozen = load_snapshot(output / "anchor.json", expected_sha256=snapshot["snapshot_sha256"],
                expected_base=snapshot["payload"]["base"], episodes_path=c.ROOT / "data/episodes/train.jsonl", split="train")
            self.assertEqual(len(frozen.probabilities), 2926)
            self.assertEqual(frozen.smoothing_epsilon, epsilon)
            self.assertEqual(set(frozen.probabilities.values()), {0.0, 1.0})
            manifest = json.loads((output / "anchor-manifest.json").read_text())
            self.assertEqual(manifest["runs"][0]["training_identity"], self.plan["training_identity"])
            # Exercise the consumer's original-receipt provenance bridge itself,
            # in addition to its complete-snapshot loader (no Torch required).
            from training.episode_adapter import EpisodeRewardAdapter
            consumer = EpisodeRewardAdapter.__new__(EpisodeRewardAdapter)
            consumer._binding = {"training_identity": self.plan["training_identity"]}
            consumer._verify_anchor_checkpoint(output / "anchor.json", output / "anchor-manifest.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
