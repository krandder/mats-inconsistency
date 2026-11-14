"""CPU integration cases with synthetic episodes/tokens, never model inference."""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

from anchors import canonical_bytes, freeze_from_manifest, proposition_id, sha256, write_snapshot
from training import (EpisodeRewardAdapter, GeneratedTrajectory, categorical_causal_logprobs,
                      policy_gradient_loss)


def digest(value):
    return sha256(canonical_bytes(value))


def negation(ident, text):
    return {"id": ident, "check_type": "Negation", "constraint": "Negation",
            "prompts": [{"role": "P", "text": text}, {"role": "not_P", "text": "Not: " + text}]}


class EpisodeAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.template = "SYSTEM:{system}\nUSER:{user}\nASSISTANT:"
        self.protocol = {"system_instruction": "Return one number in [0,1].", "verbal_prompt": "Probability of:\n{proposition}",
                         "parser": "numeric-v1"}
        self.sampling = {"schema_version": 1, "sampler": "categorical", "temperature": 1.0, "top_p": 1.0,
                         "top_k": 0, "logits_processors": [], "logits_warpers": [], "model_mode": "eval",
                         "dropout_enabled": False}
        self.identity = {"model_id": "synthetic-fixture-model", "model_revision": "1" * 40,
                         "weights_manifest_sha256": "2" * 64, "tokenizer_id": "fixture-character-tokenizer",
                         "tokenizer_revision": "3" * 40, "tokenizer_files_sha256": "4" * 64,
                         "chat_template_sha256": proposition_id(self.template),
                         "protocol_sha256": digest(self.protocol), "reasoning": "none"}
        self.train = [
            {"id": "train/and", "check_type": "And", "constraint": "And",
             "prompts": [{"role": "P", "text": "First fair coin is heads."},
                         {"role": "Q", "text": "Second fair coin is heads."},
                         {"role": "P_and_Q", "text": "Both fair coins are heads."}]},
            negation("train/neg", "A fair die is even."),
        ]
        self.valid = [negation("valid/neg", "A fair die shows one.")]
        self.heldout = [{"id": "heldout/para", "check_type": "Paraphrase", "constraint": "Paraphrase",
                         "prompts": [{"role": "P", "text": "A square has four sides."},
                                     {"role": "para_P", "text": "A square's side count is four."}]}]
        self.paths = {name: self.root / (name + ".jsonl") for name in ("train", "valid", "heldout")}
        for name, rows in (("train", self.train), ("valid", self.valid), ("heldout", self.heldout)):
            self.write_jsonl(self.paths[name], rows)
        self.manifest_path = self.root / "manifest.json"
        self.anchor_path = self.root / "anchor.json"
        self.outcomes_path = self.root / "outcomes.json"
        self.receipts, self.props, outcomes = [], [], {}
        for episode in self.train:
            for p in episode["prompts"]:
                ident = proposition_id(p["text"])
                base = .25 if p["role"] == "P_and_Q" else .5
                self.props.append({"id": ident, "text": p["text"]})
                self.receipts.append({"id": ident, "is_current": True, "requested_model": self.identity["model_id"],
                                      "reasoning": "none", "protocol_id": "fixture-v1", "campaign": "fixture",
                                      "completed": True, "exit_code": 0, "tools_used": False, "error_kind": None,
                                      "readout_status": "probability", "probability": base, "response": str(base),
                                      "finished_at": "2026-01-01T00:00:00+00:00",
                                      "training_identity_sha256": digest(self.identity),
                                      "prompt_sha256": proposition_id(self.protocol["verbal_prompt"].format(proposition=p["text"])),
                                      "system_instruction_sha256": proposition_id(self.protocol["system_instruction"])})
                outcomes[ident] = {"value": 0 if p["role"] == "not_P" else 1, "text_sha256": ident,
                                   "resolved_at": "2026-01-01T00:00:00+00:00", "evidence_sha256": "e" * 64,
                                   "source_reference": "synthetic fixture outcome, not a real forecast"}
        self.outcomes = {"schema_version": 1, "outcomes": outcomes}
        self.outcomes_path.write_bytes(canonical_bytes(self.outcomes))
        self.run = {"run_id": "fixture/base", "campaign": "fixture", "requested_model": self.identity["model_id"],
                    "reasoning": "none", "protocol_id": "fixture-v1", "receipts_file": "receipts.jsonl",
                    "recorded_at": "2026-01-01T00:00:00+00:00", "timestamp_source": "synthetic fixture",
                    "training_identity": self.identity}
        self.rebuild_anchor()

    @staticmethod
    def write_jsonl(path, rows):
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))

    def rebuild_anchor(self, *, allow_partial=False):
        self.write_jsonl(self.root / "receipts.jsonl", self.receipts)
        self.write_jsonl(self.root / "propositions.jsonl", self.props)
        self.run["receipts_sha256"] = sha256((self.root / "receipts.jsonl").read_bytes())
        manifest = {"runs": [self.run], "native_protocols": {"fixture-v1": self.protocol},
                    "propositions_file": "propositions.jsonl", "propositions_sha256": sha256((self.root / "propositions.jsonl").read_bytes())}
        self.manifest_path.write_bytes(canonical_bytes(manifest))
        self.snapshot = freeze_from_manifest(self.manifest_path, "fixture/base", self.paths["train"], "train",
                                             created_at="2026-09-13T00:00:00+00:00", allow_partial=allow_partial)
        if self.anchor_path.exists():
            self.anchor_path.unlink()  # Test fixture rebuilding, not an API overwrite.
        write_snapshot(self.anchor_path, self.snapshot)

    def binding(self, arm="a2", weight=2):
        has_labels = arm in {"a1", "control-brier"}
        return {"schema_version": 1, "experiment_id": "synthetic-credit-assignment-test", "training_identity": self.identity,
                "protocol_id": "fixture-v1", "protocol": self.protocol, "anchor_sha256": self.snapshot["snapshot_sha256"],
                "split_sha256": {k: sha256(v.read_bytes()) for k, v in self.paths.items()},
                "arm": arm, "lambda_anchor": weight, "output_parser": "strict-decimal-v1",
                "split_policy": "episode-ids-and-check-types-v1", "loss_reduction": "mean-joint-sample-sum-role-tokens-v1",
                "nonfinite_reward_policy": "reject-batch",
                "sampling_distribution": copy.deepcopy(self.sampling),
                "outcomes_sha256": sha256(self.outcomes_path.read_bytes()) if has_labels else None,
                "training_cutoff": "2026-02-01T00:00:00+00:00" if has_labels else None}

    def adapter(self, binding=None, **kwargs):
        binding = binding or self.binding()
        return EpisodeRewardAdapter(binding, expected_binding_sha256=digest(binding), split_paths=self.paths,
                                    anchor_path=self.anchor_path, anchor_manifest_path=self.manifest_path,
                                    chat_template=self.template,
                                    outcomes_path=self.outcomes_path if binding["outcomes_sha256"] else None, **kwargs)

    def encode(self, messages):
        self.assertEqual([role for role, _ in messages], ["system", "user"])
        rendered = self.template.format(system=messages[0][1], user=messages[1][1])
        return tuple(ord(c) for c in rendered)

    @staticmethod
    def decode(tokens):
        return "".join(chr(x) for x in tokens)

    def plan(self, adapter, *, batch_id="batch-1", episode_ids=None, samples=2):
        return adapter.prepare(episode_ids or ["train/and", "train/neg"], samples_per_episode=samples,
                               batch_id=batch_id, policy_sha256="a" * 64,
                               runtime_training_identity=copy.deepcopy(self.identity),
                               runtime_sampling_distribution=copy.deepcopy(self.sampling), encode_messages=self.encode)

    def trajectories(self, plan, *, response=None):
        rows = []
        for request in plan.requests:
            if response is not None:
                text = response
            elif request.episode_id == "train/neg":
                text = "0.5" if request.sample_index == 0 else "0.8"
            elif request.sample_index == 0:
                text = "0.25" if request.role == "P_and_Q" else "0.5"
            else:
                text = {"P": "0.8", "Q": "0.7", "P_and_Q": "0.1"}[request.role]
            rows.append(GeneratedTrajectory(request.request_id, plan.plan_sha256, plan.experiment_sha256,
                                              plan.policy_sha256, plan.sampling_distribution_sha256,
                                              request.messages_sha256, request.prompt_token_ids,
                                              tuple(ord(c) for c in text), text, "context-" + request.request_id,
                                              1, False, "stop"))
        return rows

    def test_complete_episode_credit_and_causal_loss_alignment(self):
        adapter = self.adapter()
        plan = self.plan(adapter)
        self.assertEqual(plan.joint_sample_count, 4)
        self.assertEqual(len(plan.requests), 10)  # (3 And roles + 2 Negation roles) * 2.
        rows = self.trajectories(plan)
        credits = adapter.score(plan, rows, decode_completion=self.decode, pad_token_id=0)
        neg_kl = .8 * math.log(1.6) + .2 * math.log(.4)
        neg_reward = -.6 - 2 * neg_kl
        and_kl = ((.8 * math.log(1.6) + .2 * math.log(.4))
                  + (.7 * math.log(1.4) + .3 * math.log(.6))
                  + (.1 * math.log(.4) + .9 * math.log(1.2))) / 3
        and_reward = -.4 - 2 * and_kl
        for i, (request, row) in enumerate(zip(plan.requests, rows, strict=True)):
            expected = 0 if request.sample_index == 0 else (neg_reward if request.episode_id == "train/neg" else and_reward)
            self.assertAlmostEqual(credits.row_rewards[i], expected)
            self.assertEqual(credits.joint_sample_ids[i], request.joint_sample_id)
            self.assertEqual(sum(credits.completion_loss_mask[i]), len(row.completion_token_ids))
            for pos, coefficient in enumerate(credits.loss_coefficients[i]):
                target_pos = pos + 1
                generated = len(row.prompt_token_ids) <= target_pos < len(row.prompt_token_ids) + len(row.completion_token_ids)
                self.assertAlmostEqual(coefficient, -expected / 4 if generated else 0)
        logps = [[-1.0] * len(row) for row in credits.loss_coefficients]
        actual = policy_gradient_loss(logps, credits, trajectory_ids=credits.trajectory_ids,
                                      policy_sha256=plan.policy_sha256, runtime_sampling_distribution=self.sampling)
        expected = sum(reward * len(row.completion_token_ids) / 4 for reward, row in zip(credits.row_rewards, rows, strict=True))
        self.assertAlmostEqual(actual, expected)
        # Independent finite-difference credit check, including a masked prompt token.
        row_index = next(i for i, r in enumerate(credits.row_rewards) if r < 0)
        generated_pos = credits.completion_loss_mask[row_index].index(1)
        for pos in (0, generated_pos):
            plus, minus = copy.deepcopy(logps), copy.deepcopy(logps)
            plus[row_index][pos] += 1e-5
            minus[row_index][pos] -= 1e-5
            kwargs = {"trajectory_ids": credits.trajectory_ids, "policy_sha256": plan.policy_sha256,
                      "runtime_sampling_distribution": self.sampling}
            derivative = (policy_gradient_loss(plus, credits, **kwargs) - policy_gradient_loss(minus, credits, **kwargs)) / 2e-5
            self.assertAlmostEqual(derivative, credits.loss_coefficients[row_index][pos], places=8)

    def test_a0_a1_and_brier_only_control_equations(self):
        for arm, weight, expected in (("a0", 0, -.6), ("control-brier", 0, -.34), ("a1", 2, -.6 - 2 * .34)):
            with self.subTest(arm=arm):
                adapter = self.adapter(self.binding(arm, weight))
                plan = self.plan(adapter, episode_ids=["train/neg"], samples=1)
                credits = adapter.score(plan, self.trajectories(plan, response="0.8"), decode_completion=self.decode, pad_token_id=0)
                self.assertEqual(len(credits.row_rewards), 2)
                for reward in credits.row_rewards:
                    self.assertAlmostEqual(reward, expected)
                metrics = dict(credits.episode_metrics[0][1])
                self.assertIsNone(metrics["mean_kl"])
                self.assertAlmostEqual(metrics["exploitability"], .6)

    def test_shuffled_incomplete_duplicate_and_foreign_batches_rejected(self):
        adapter = self.adapter()
        plan = self.plan(adapter)
        rows = self.trajectories(plan)
        malformed = [rows[::-1], rows[:-1], rows + [rows[0]], [rows[0]] + rows[:-1],
                     [replace(rows[0], plan_sha256="b" * 64)] + rows[1:]]
        for batch in malformed:
            with self.assertRaises(ValueError):
                adapter.score(plan, batch, decode_completion=self.decode, pad_token_id=0)
        changed_plan = replace(plan, requests=plan.requests[::-1])
        with self.assertRaises(ValueError):
            adapter.score(changed_plan, rows, decode_completion=self.decode, pad_token_id=0)
        # Failure returned no partial credit and did not consume the complete plan.
        adapter.score(plan, rows, decode_completion=self.decode, pad_token_id=0)
        with self.assertRaisesRegex(ValueError, "consumed"):
            adapter.score(plan, rows, decode_completion=self.decode, pad_token_id=0)

    def test_invalid_readouts_fail_whole_batch(self):
        for bad in ("ABSTAIN", "I cannot estimate.", "0.5 commentary", "1.1", "-0.1", "NaN", "", "50%", "0.\u0665"):
            adapter = self.adapter()
            plan = self.plan(adapter)
            rows = self.trajectories(plan)
            rows[-1] = replace(rows[-1], completion_text=bad, completion_token_ids=tuple(ord(c) for c in bad))
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                adapter.score(plan, rows, decode_completion=self.decode, pad_token_id=0)

    def test_token_identity_context_policy_and_truncation_rejected(self):
        adapter = self.adapter()
        plan = self.plan(adapter)
        rows = self.trajectories(plan)
        mutations = [{"completion_text": "0.9"}, {"prompt_token_ids": (1, 2)}, {"policy_sha256": "c" * 64},
                     {"experiment_sha256": "d" * 64}, {"messages_sha256": "e" * 64}, {"finish_reason": "length"},
                     {"tools_used": True}, {"context_turn_count": 2}, {"context_turn_count": 1.0},
                     {"context_turn_count": True}, {"prompt_token_ids": tuple(float(x) for x in rows[0].prompt_token_ids)},
                     {"context_id": rows[1].context_id}]
        for fields in mutations:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                adapter.score(plan, [replace(rows[0], **fields)] + rows[1:], decode_completion=self.decode, pad_token_id=0)

    def test_context_reuse_across_plans_rejected(self):
        adapter = self.adapter()
        first = self.plan(adapter)
        first_rows = self.trajectories(first)
        adapter.score(first, first_rows, decode_completion=self.decode, pad_token_id=0)
        second = self.plan(adapter, batch_id="batch-2")
        rows = self.trajectories(second)
        rows[0] = replace(rows[0], context_id=first_rows[0].context_id)
        with self.assertRaisesRegex(ValueError, "context was reused"):
            adapter.score(second, rows, decode_completion=self.decode, pad_token_id=0)

    def test_validation_heldout_and_noncanonical_training_requests_rejected(self):
        adapter = self.adapter()
        for ids in (["valid/neg"], ["heldout/para"], ["unknown"], ["train/neg", "train/and"], ["train/neg", "train/neg"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.plan(adapter, episode_ids=ids)
        self.plan(adapter)
        with self.assertRaisesRegex(ValueError, "batch id already"):
            self.plan(adapter)

    def test_split_leakage_and_changed_split_bytes_rejected(self):
        binding = self.binding()
        with self.paths["valid"].open("ab") as f:
            f.write(b"\n")
        with self.assertRaisesRegex(ValueError, "valid split hash"):
            self.adapter(binding)
        self.write_jsonl(self.paths["valid"], [copy.deepcopy(self.train[0])])
        with self.assertRaisesRegex(ValueError, "episode IDs overlap"):
            self.adapter()
        self.write_jsonl(self.paths["valid"], self.valid)
        leaked = copy.deepcopy(self.heldout[0])
        leaked["id"] = "train/leaked-check"
        self.write_jsonl(self.paths["train"], self.train + [leaked])
        with self.assertRaisesRegex(ValueError, "check type leaked"):
            self.adapter()

    def test_requested_native_identity_cannot_replace_weight_evidence(self):
        self.run.pop("training_identity")
        self.rebuild_anchor()
        with self.assertRaisesRegex(ValueError, "trainable checkpoint"):
            self.adapter()
        self.run["training_identity"] = self.identity
        self.receipts[0].pop("training_identity_sha256")
        self.rebuild_anchor()
        with self.assertRaisesRegex(ValueError, "observation lacks"):
            self.adapter()

    def test_checkpoint_template_protocol_effort_and_binding_mismatch(self):
        adapter = self.adapter()
        for field, wrong in (("model_revision", "a" * 40), ("tokenizer_revision", "b" * 40),
                             ("chat_template_sha256", "c" * 64), ("reasoning", "high")):
            runtime = {**self.identity, field: wrong}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "runtime checkpoint"):
                adapter.prepare(["train/neg"], samples_per_episode=1, batch_id=field, policy_sha256="a" * 64,
                                runtime_training_identity=runtime, runtime_sampling_distribution=self.sampling,
                                encode_messages=self.encode)
        binding = self.binding()
        bad = copy.deepcopy(binding)
        bad["training_identity"]["model_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.adapter(bad)
        bad = copy.deepcopy(binding)
        bad["protocol"]["system_instruction"] += " changed"
        with self.assertRaisesRegex(ValueError, "protocol content"):
            self.adapter(bad)
        with self.assertRaisesRegex(ValueError, "binding hash"):
            EpisodeRewardAdapter(binding, expected_binding_sha256="0" * 64, split_paths=self.paths,
                                 anchor_path=self.anchor_path, anchor_manifest_path=self.manifest_path, chat_template=self.template)
        with self.assertRaisesRegex(ValueError, "chat template"):
            EpisodeRewardAdapter(binding, expected_binding_sha256=digest(binding), split_paths=self.paths,
                                 anchor_path=self.anchor_path, anchor_manifest_path=self.manifest_path, chat_template="other")

    def test_partial_or_wrong_anchor_rejected_before_any_rollout(self):
        binding = self.binding()
        binding["anchor_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "pinned hash"):
            self.adapter(binding)
        self.receipts.pop()
        self.rebuild_anchor(allow_partial=True)
        with self.assertRaisesRegex(ValueError, "partial snapshot"):
            self.adapter()

    def test_nonfinite_exact_anchor_rejects_training_batch(self):
        self.receipts[0].update(probability=0, response="0")
        self.rebuild_anchor()
        adapter = self.adapter()
        plan = self.plan(adapter)
        with self.assertRaisesRegex(ValueError, "nonfinite episode reward"):
            adapter.score(plan, self.trajectories(plan), decode_completion=self.decode, pad_token_id=0)

    def test_control_outcomes_need_full_coverage_and_prior_resolution(self):
        self.outcomes["outcomes"].pop(self.props[0]["id"])
        self.outcomes_path.write_bytes(canonical_bytes(self.outcomes))
        with self.assertRaisesRegex(ValueError, "cover exactly"):
            self.adapter(self.binding("control-brier", 0))
        # Rebuild a complete outcome file, with one future resolved timestamp.
        row = {"value": 1, "text_sha256": self.props[0]["id"], "resolved_at": "2026-03-01T00:00:00Z",
               "evidence_sha256": "e" * 64, "source_reference": "synthetic fixture"}
        self.outcomes["outcomes"][self.props[0]["id"]] = row
        self.outcomes_path.write_bytes(canonical_bytes(self.outcomes))
        with self.assertRaisesRegex(ValueError, "after the bound training cutoff"):
            self.adapter(self.binding("a1", 1))
        row["resolved_at"] = "2026-01-01T00:00:00Z"
        row["value"] = None
        self.outcomes_path.write_bytes(canonical_bytes(self.outcomes))
        with self.assertRaises(ValueError):
            self.adapter(self.binding("control-brier", 0))

    def test_loss_rejects_shuffled_rows_offpolicy_shape_and_invalid_logps(self):
        adapter = self.adapter()
        plan = self.plan(adapter)
        credits = adapter.score(plan, self.trajectories(plan), decode_completion=self.decode, pad_token_id=0)
        logps = [[-1.0] * len(row) for row in credits.loss_coefficients]
        kwargs = {"trajectory_ids": credits.trajectory_ids, "policy_sha256": credits.policy_sha256,
                  "runtime_sampling_distribution": self.sampling}
        with self.assertRaises(ValueError):
            policy_gradient_loss(logps, credits, **{**kwargs, "trajectory_ids": credits.trajectory_ids[::-1]})
        with self.assertRaises(ValueError):
            policy_gradient_loss(logps, credits, **{**kwargs, "policy_sha256": "e" * 64})
        for bad in (logps[:-1], [logps[0][:-1]] + logps[1:]):
            with self.assertRaises(ValueError):
                policy_gradient_loss(bad, credits, **kwargs)
        for bad_value in (math.inf, math.nan, .1):
            bad = copy.deepcopy(logps)
            bad[0][0] = bad_value
            with self.assertRaises(ValueError):
                policy_gradient_loss(bad, credits, **kwargs)

    def test_sampling_distribution_binding_runtime_and_postplan_drift(self):
        adapter = self.adapter()
        plan = self.plan(adapter)
        rows = self.trajectories(plan)
        self.assertEqual(plan.sampling_distribution_sha256, digest(self.sampling))
        self.assertTrue(all(r.sampling_distribution_sha256 == digest(self.sampling) for r in plan.requests))
        drifts = ({"temperature": .7}, {"top_p": .9}, {"top_k": 20}, {"sampler": "greedy"},
                  {"logits_processors": ["repetition-penalty"]}, {"logits_warpers": ["min-p"]},
                  {"model_mode": "train"}, {"dropout_enabled": True}, {"temperature": True},
                  {"extra_generation_default": "forced-eos"})
        for fields in drifts:
            changed = {**self.sampling, **fields}
            binding = self.binding()
            binding["sampling_distribution"] = changed
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "sampling distribution"):
                self.adapter(binding)
            with self.assertRaisesRegex(ValueError, "sampling distribution"):
                adapter.prepare(["train/neg"], samples_per_episode=1, batch_id="drift", policy_sha256="a" * 64,
                                runtime_training_identity=self.identity, runtime_sampling_distribution=changed,
                                encode_messages=self.encode)
            # Runtime drift after planning must be reported in each generation receipt.
            changed_rows = [replace(rows[0], sampling_distribution_sha256=digest(changed))] + rows[1:]
            with self.assertRaisesRegex(ValueError, "wrong binding"):
                adapter.score(plan, changed_rows, decode_completion=self.decode, pad_token_id=0)
        missing = self.binding()
        del missing["sampling_distribution"]
        with self.assertRaisesRegex(ValueError, "missing/unknown"):
            self.adapter(missing)
        credits = adapter.score(plan, rows, decode_completion=self.decode, pad_token_id=0)
        self.assertEqual(credits.sampling_distribution_sha256, digest(self.sampling))
        logps = [[-1.0] * len(row) for row in credits.loss_coefficients]
        with self.assertRaisesRegex(ValueError, "sampling distribution"):
            policy_gradient_loss(logps, credits, trajectory_ids=credits.trajectory_ids,
                                 policy_sha256=credits.policy_sha256,
                                 runtime_sampling_distribution={**self.sampling, "model_mode": "train"})

    def test_categorical_causal_logprobs_match_the_generation_distribution(self):
        # Independent integer-weight categorical CDF: probabilities are 1/8, 2/8, 5/8.
        weights = (1, 2, 5)
        uniforms = (.05, .2, .9)
        sampled = tuple(next(i for i in range(3) if u < sum(weights[:i + 1]) / sum(weights)) for u in uniforms)
        self.assertEqual(sampled, (0, 1, 2))
        # These unmodified logits induce that same full-vocabulary categorical policy.
        logits = tuple(math.log(w) for w in weights)
        actual = categorical_causal_logprobs([[logits] * 3], [(2,) + sampled],
                                            runtime_sampling_distribution=self.sampling)
        for value, token in zip(actual[0], sampled, strict=True):
            self.assertAlmostEqual(value, math.log(weights[token] / sum(weights)), places=14)
        shifted = categorical_causal_logprobs([[tuple(x + 500 for x in logits)] * 3], [(2,) + sampled],
                                             runtime_sampling_distribution=self.sampling)
        for first, second in zip(actual[0], shifted[0], strict=True):
            self.assertAlmostEqual(first, second, places=12)
        with self.assertRaisesRegex(ValueError, "sampling distribution"):
            categorical_causal_logprobs([[logits] * 3], [(2,) + sampled],
                                       runtime_sampling_distribution={**self.sampling, "temperature": .5})
        for malformed in ([[logits] * 2], [[logits, (0, math.nan, 1), logits]], [[logits, (0, 1), logits]]):
            with self.assertRaises(ValueError):
                categorical_causal_logprobs(malformed, [(2,) + sampled], runtime_sampling_distribution=self.sampling)


if __name__ == "__main__":
    unittest.main()
