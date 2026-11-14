"""Analytical KL checks and adversarial offline snapshot/provenance fixtures."""

import copy
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from anchors import (a2_reward, bernoulli_kl, canonical_bytes, freeze_from_manifest,
                     load_snapshot, mean_bernoulli_kl, proposition_id, sha256, write_snapshot)


class BernoulliKLTests(unittest.TestCase):
    def test_known_values_and_direction(self):
        self.assertAlmostEqual(bernoulli_kl(.8, .5), .8 * math.log(1.6) + .2 * math.log(.4))
        self.assertAlmostEqual(bernoulli_kl(.5, .8), math.log(1.25))
        self.assertNotAlmostEqual(bernoulli_kl(.8, .5), bernoulli_kl(.5, .8))
        self.assertEqual(bernoulli_kl(.3, .3), 0)

    def test_endpoints(self):
        for p in (0, 1):
            self.assertEqual(bernoulli_kl(p, p), 0)
        for p, q in ((1, 0), (0, 1), (.1, 0), (.9, 1)):
            self.assertEqual(bernoulli_kl(p, q), math.inf)
        self.assertAlmostEqual(bernoulli_kl(0, .2), -math.log(.8))
        self.assertAlmostEqual(bernoulli_kl(1, .2), -math.log(.2))

    def test_near_equality_and_extreme_finite_values(self):
        # Decimal independently evaluates the defining expression at high
        # precision, including adjacent binary floats with KL around 1e-32.
        pairs = [(math.nextafter(.5, 1), .5), (.000100001, .0001),
                 (1 - 1e-12, 1 - 2e-12), (.2, .3), (2e-300, 1e-300),
                 (.5, math.nextafter(0, 1))]
        for p, q in pairs:
            with self.subTest(p=p, q=q), localcontext() as ctx:
                ctx.prec = 400
                dp, dq = Decimal.from_float(p), Decimal.from_float(q)
                expected = float(dp * (dp.ln() - dq.ln()) + (1 - dp) * ((1 - dp).ln() - (1 - dq).ln()))
                actual = bernoulli_kl(p, q)
                self.assertGreaterEqual(actual, 0)
                self.assertTrue(math.isclose(actual, expected, rel_tol=2e-12), (actual, expected))

    def test_mean_is_normalized_not_sum(self):
        expected = bernoulli_kl(.8, .5)
        self.assertAlmostEqual(mean_bernoulli_kl({"a": .8, "b": .8}, {"a": .5, "b": .5}), expected)
        self.assertAlmostEqual(mean_bernoulli_kl({"a": .8, "b": .5}, {"a": .5, "b": .5}), expected / 2)
        for candidate, base in (({}, {}), ({"a": .5}, {"b": .5}), ({"a": .5}, {"a": .5, "b": .5})):
            with self.assertRaises(ValueError):
                mean_bernoulli_kl(candidate, base)

    def test_explicit_smoothing_is_different_and_finite(self):
        epsilon = .01
        expected = .99 * math.log(99) + .01 * math.log(1 / 99)
        self.assertAlmostEqual(bernoulli_kl(1, 0, smoothing_epsilon=epsilon), expected)
        self.assertEqual(bernoulli_kl(0, 0, smoothing_epsilon=epsilon), 0)
        for bad in (-.1, 0, .5, 1, 1e-30, True, "0.01", math.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                bernoulli_kl(.5, .5, smoothing_epsilon=bad)

    def test_rejects_invalid_probabilities_without_clipping(self):
        for bad in (-1e-12, 1 + 1e-12, math.inf, -math.inf, math.nan, None, True, "0.5", 10 ** 1000):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    bernoulli_kl(bad, .5)
                with self.assertRaises(ValueError):
                    bernoulli_kl(.5, bad)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.episodes_path = self.root / "train.jsonl"
        self.episode = {"id": "fixture/negation", "check_type": "Negation", "constraint": "Negation",
                        "prompts": [{"role": "P", "text": "A fair die shows an even number."},
                                    {"role": "not_P", "text": "A fair die does not show an even number."}]}
        self.write_episodes([self.episode])
        self.protocol = {"verbal_prompt": "Estimate.\n{proposition}", "system_instruction": "Return one probability.", "parser": "numeric-v1"}
        self.run = {"run_id": "fixture/base", "requested_model": "fixture-model", "reasoning": "low",
                    "protocol_id": "fixture-v1", "campaign": "fixture", "receipts_file": "receipts.jsonl",
                    "recorded_at": "2026-09-01T00:00:01+00:00", "timestamp_source": "native_receipt"}
        self.receipts = []
        self.props = []
        for p in self.episode["prompts"]:
            ident = proposition_id(p["text"])
            self.props.append({"id": ident, "text": p["text"]})
            self.receipts.append({"id": ident, "is_current": True, "requested_model": "fixture-model", "reasoning": "low",
                                  "protocol_id": "fixture-v1", "campaign": "fixture", "completed": True, "exit_code": 0,
                                  "tools_used": False, "error_kind": None, "readout_status": "probability", "probability": .5,
                                  "response": "0.5", "finished_at": "2026-09-01T00:00:00+00:00",
                                  "prompt_sha256": proposition_id(self.protocol["verbal_prompt"].format(proposition=p["text"])),
                                  "system_instruction_sha256": proposition_id(self.protocol["system_instruction"])})
        self.write_sources()

    def write_episodes(self, rows):
        self.episodes_path.write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))

    def write_sources(self):
        for filename, rows in (("receipts.jsonl", self.receipts), ("propositions.jsonl", self.props)):
            (self.root / filename).write_bytes(b"".join(canonical_bytes(r) + b"\n" for r in rows))
        self.run["receipts_sha256"] = sha256((self.root / "receipts.jsonl").read_bytes())
        manifest = {"runs": [self.run], "native_protocols": {"fixture-v1": self.protocol},
                    "propositions_file": "propositions.jsonl", "propositions_sha256": sha256((self.root / "propositions.jsonl").read_bytes())}
        self.manifest_path.write_bytes(canonical_bytes(manifest))

    def freeze(self, **kwargs):
        return freeze_from_manifest(self.manifest_path, "fixture/base", self.episodes_path, "train",
                                    created_at="2026-09-13T00:00:00+00:00", **kwargs)

    def load(self, snapshot, **kwargs):
        path = self.root / "snapshot.json"
        path.write_bytes(canonical_bytes(snapshot))
        return load_snapshot(path, expected_sha256=snapshot["snapshot_sha256"], expected_base=snapshot["payload"]["base"],
                             episodes_path=self.episodes_path, split="train", **kwargs)

    def test_complete_snapshot_roundtrip_and_analytical_reward(self):
        snapshot = self.freeze()
        frozen = self.load(snapshot)
        self.assertEqual(frozen.status, "complete-for-bound-split")
        reward = a2_reward(self.episode, {"P": .8, "not_P": .8}, frozen, lambda_kl=2)
        self.assertAlmostEqual(reward["exploitability"], .6)
        self.assertAlmostEqual(reward["mean_kl"], .8 * math.log(1.6) + .2 * math.log(.4))
        self.assertAlmostEqual(reward["reward"], -.6 - 2 * reward["mean_kl"])
        with self.assertRaises(TypeError):
            frozen.probabilities[self.props[0]["id"]] = .1

    def test_endpoint_infinite_reward_and_zero_weight(self):
        self.receipts[0].update(probability=0, response="0")
        self.receipts[1].update(probability=1, response="1")
        self.write_sources()
        frozen = self.load(self.freeze())
        candidate = {"P": .5, "not_P": .5}
        result = a2_reward(self.episode, candidate, frozen, lambda_kl=1)
        self.assertEqual(result["reward"], -math.inf)
        self.assertEqual(result["mean_kl"], math.inf)
        result = a2_reward(self.episode, candidate, frozen, lambda_kl=0)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["mean_kl"], math.inf)
        smooth = self.load(self.freeze(smoothing_epsilon=.01))
        self.assertTrue(math.isfinite(a2_reward(self.episode, candidate, smooth, lambda_kl=1)["reward"]))

    def test_finite_penalty_overflow_is_not_reported_as_endpoint_infinity(self):
        for row in self.receipts:
            row.update(probability=.001, response="0.001")
        self.write_sources()
        frozen = self.load(self.freeze())
        with self.assertRaisesRegex(OverflowError, "finite KL penalty"):
            a2_reward(self.episode, {"P": 1, "not_P": 1}, frozen, lambda_kl=1e308)

    def test_partial_is_explicit_and_cannot_score_missing_episode(self):
        self.receipts.pop()
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "partial anchor"):
            self.freeze()
        snapshot = self.freeze(allow_partial=True)
        self.assertEqual(snapshot["payload"]["coverage"]["target_propositions"], 2)
        self.assertEqual(snapshot["payload"]["coverage"]["anchored_propositions"], 1)
        with self.assertRaisesRegex(ValueError, "partial snapshot"):
            self.load(snapshot)
        frozen = self.load(snapshot, allow_partial=True)
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            a2_reward(self.episode, {"P": .5, "not_P": .5}, frozen, lambda_kl=0)

    def test_refusals_and_nulls_never_become_anchors(self):
        self.receipts[0].update(probability=None, response="ABSTAIN", readout_status="abstention")
        self.write_sources()
        snapshot = self.freeze(allow_partial=True)
        self.assertEqual(snapshot["payload"]["coverage"]["missing_propositions"][self.props[0]["id"]], "abstention")
        self.receipts[0].update(readout_status="refusal", response="I cannot estimate this.")
        self.write_sources()
        self.assertNotIn(self.props[0]["id"], self.freeze(allow_partial=True)["payload"]["anchors"])
        self.receipts[0].update(readout_status="probability", response="0.5")
        self.write_sources()
        with self.assertRaises(ValueError):
            self.freeze(allow_partial=True)

    def test_bound_source_hash_catches_tampering(self):
        with (self.root / "receipts.jsonl").open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            self.freeze()

    def test_pinned_hash_catches_tampering_even_if_internal_hash_recomputed(self):
        snapshot = self.freeze()
        expected = snapshot["snapshot_sha256"]
        snapshot["payload"]["anchors"][self.props[0]["id"]]["probability"] = .6
        path = self.root / "tampered.json"
        for recompute in (False, True):
            if recompute:
                snapshot["snapshot_sha256"] = sha256(canonical_bytes(snapshot["payload"]))
            path.write_bytes(canonical_bytes(snapshot))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_snapshot(path, expected_sha256=expected, expected_base=snapshot["payload"]["base"],
                              episodes_path=self.episodes_path, split="train")

    def test_changed_dataset_split_or_expected_treatment_rejected(self):
        snapshot = self.freeze()
        path = self.root / "snapshot.json"
        write_snapshot(path, snapshot)
        for field, wrong in (("model", "other-model"), ("reasoning", "max"),
                             ("protocol_id", "other-v1"), ("protocol_sha256", "0" * 64)):
            base = dict(snapshot["payload"]["base"])
            base[field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "binding mismatch"):
                load_snapshot(path, expected_sha256=snapshot["snapshot_sha256"], expected_base=base,
                              episodes_path=self.episodes_path, split="train")
        with self.assertRaisesRegex(ValueError, "dataset split"):
            load_snapshot(path, expected_sha256=snapshot["snapshot_sha256"], expected_base=snapshot["payload"]["base"],
                          episodes_path=self.episodes_path, split="valid")
        self.episode["prompts"][0]["text"] += " Changed resolution."
        self.write_episodes([self.episode])
        with self.assertRaisesRegex(ValueError, "dataset split"):
            load_snapshot(path, expected_sha256=snapshot["snapshot_sha256"], expected_base=snapshot["payload"]["base"],
                          episodes_path=self.episodes_path, split="train")

    def test_duplicate_and_mismatched_ids_rejected(self):
        self.receipts.append(dict(self.receipts[0]))
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "duplicate or unknown"):
            self.freeze()
        self.receipts.pop()
        self.props[0]["text"] += " Changed resolution."
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "id/text"):
            self.freeze()

    def test_duplicate_episode_ids_or_roles_rejected(self):
        self.write_episodes([self.episode, self.episode])
        with self.assertRaisesRegex(ValueError, "duplicate episode"):
            self.freeze()
        self.episode["prompts"][1]["role"] = "P"
        self.write_episodes([self.episode])
        with self.assertRaisesRegex(ValueError, "roles"):
            self.freeze()

    def test_shared_proposition_reused_without_duplicate_anchor(self):
        second = copy.deepcopy(self.episode)
        second["id"] = "fixture/another-episode"
        self.write_episodes([self.episode, second])
        snapshot = self.freeze()
        self.assertEqual(len(snapshot["payload"]["anchors"]), 2)
        self.assertEqual(len(snapshot["payload"]["coverage"]["complete_episode_ids"]), 2)

    def test_receipt_mismatch_and_hidden_clipping_rejected(self):
        original = dict(self.receipts[0])
        for field, value in (("requested_model", "wrong"), ("reasoning", "high"),
                             ("protocol_id", "wrong"), ("prompt_sha256", "wrong"),
                             ("system_instruction_sha256", "wrong"), ("probability", 1.1),
                             ("response", "0.7"), ("finished_at", None)):
            self.receipts[0] = {**original, field: value}
            self.write_sources()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.freeze(allow_partial=True)

    def test_legacy_incomplete_provenance_rejected(self):
        self.run["reasoning"] = "none-or-low (legacy fallback)"
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "legacy fallback"):
            self.freeze(allow_partial=True)

    def test_allowed_ids_restrict_source_without_reducing_denominator(self):
        ident = self.props[0]["id"]
        snapshot = self.freeze(allow_partial=True, allowed_ids=[ident])
        self.assertEqual(snapshot["payload"]["coverage"]["target_propositions"], 2)
        self.assertEqual(list(snapshot["payload"]["anchors"]), [ident])
        for ids in ([ident, ident], ["unknown"]):
            with self.assertRaises(ValueError):
                self.freeze(allow_partial=True, allowed_ids=ids)

    def test_no_overwrite_or_symlink_replacement(self):
        path = self.root / "frozen.json"
        snapshot = self.freeze()
        write_snapshot(path, snapshot)
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            write_snapshot(path, snapshot)
        self.assertEqual(path.read_bytes(), before)
        link = self.root / "link.json"
        link.symlink_to("missing.json")
        with self.assertRaises(FileExistsError):
            write_snapshot(link, snapshot)
        self.assertTrue(link.is_symlink())

    def test_reward_rejects_changed_episode_bad_roles_or_bad_lambda(self):
        frozen = self.load(self.freeze())
        changed = copy.deepcopy(self.episode)
        changed["prompts"][0]["text"] += " Changed resolution."
        with self.assertRaises(ValueError):
            a2_reward(changed, {"P": .5, "not_P": .5}, frozen, lambda_kl=1)
        for candidate in ({"P": .5}, {"P": .5, "not_P": .5, "extra": .5}, {"P": -1, "not_P": .5}):
            with self.assertRaises(ValueError):
                a2_reward(self.episode, candidate, frozen, lambda_kl=1)
        for weight in (-1, math.inf, math.nan, True, "1", 10 ** 1000):
            with self.assertRaises(ValueError):
                a2_reward(self.episode, {"P": .5, "not_P": .5}, frozen, lambda_kl=weight)

    def test_cli_offline_and_duplicate_json_keys(self):
        output = self.root / "cli-snapshot.json"
        command = [sys.executable, "scripts/freeze_base_beliefs.py", "--manifest", str(self.manifest_path),
                   "--run-id", "fixture/base", "--episodes", str(self.episodes_path), "--split", "train", "--output", str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(json.loads(first.stdout)["status"], "complete-for-bound-split")
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(second.returncode, 2)
        self.manifest_path.write_text('{"runs": [], "runs": []}')
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.freeze()


if __name__ == "__main__":
    unittest.main()
