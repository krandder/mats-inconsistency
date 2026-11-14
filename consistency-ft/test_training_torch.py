"""Actual synthetic CPU generation/backward regressions; no candidate inference.

Run in the pinned environment documented in docs/synthetic-cpu-training.md.
The dependency-free repository suite skips this module when torch is absent.
"""

import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None

if torch is not None:
    from anchors import proposition_id
    from training.episode_adapter import TokenCreditBatch
    from training.synthetic_cpu import SyntheticExperiment, TinyCausalModel
    from training.torch_loop import (OptimizerSession, RolloutBatchFailure, causal_policy_loss, digest,
                                     generate_plan, generate_tokens, policy_sha256,
                                     runtime_sampling_distribution)


@unittest.skipIf(torch is None, "optional pinned CPU environment required")
class SyntheticRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.experiment = SyntheticExperiment(self.root / "experiment")

    def batch(self, batch_id="batch"):
        exp = self.experiment
        plan = exp.prepare(batch_id)
        records = generate_plan(exp.model, plan, exp.tokenizer, generator=torch.Generator().manual_seed(202))
        credits = exp.adapter.score(plan, [r.trajectory for r in records],
                                    decode_completion=exp.tokenizer.decode_completion,
                                    pad_token_id=exp.tokenizer.pad_token_id)
        return plan, records, credits

    def test_generated_group_rewards_causal_eos_credit_and_raw_sequence_sum(self):
        exp = self.experiment
        plan, records, credits = self.batch()
        self.assertEqual((plan.joint_sample_count, len(records)), (4, 10))
        self.assertGreater(len({len(r.prompt_token_ids) for r in plan.requests}), 1)
        receipts = [json.loads(line) for line in (exp.root / "anchor-receipts.jsonl").read_text().splitlines()]
        self.assertEqual(len(receipts), 5)
        base = {row["id"]: float(exp.tokenizer.decode_completion(row["completion_token_ids"])) for row in receipts}
        groups = {}
        for request, record in zip(plan.requests, records, strict=True):
            groups.setdefault(request.joint_sample_id, []).append((request, record))
        expected_rewards = {}
        for joint, rows in groups.items():
            probabilities = {req.role: float(record.trajectory.completion_text) for req, record in rows}
            if rows[0][0].episode_id == "train/and":
                p, q, conjunction = (probabilities[k] for k in ("P", "Q", "P_and_Q"))
                exploitation = max(0, conjunction - p, conjunction - q, p + q - conjunction - 1)
            else:
                exploitation = abs(probabilities["P"] + probabilities["not_P"] - 1)
            episode = next(e for e in exp.splits["train"] if e["id"] == rows[0][0].episode_id)
            divergences = []
            for prompt in episode["prompts"]:
                p, b = probabilities[prompt["role"]], base[proposition_id(prompt["text"])]
                divergences.append(p * math.log(p / b) + (1 - p) * math.log((1 - p) / (1 - b)))
            expected_rewards[joint] = -exploitation - 2 * math.fsum(divergences) / len(divergences)
        expected_loss = 0
        for index, (request, record) in enumerate(zip(plan.requests, records, strict=True)):
            row = record.trajectory
            reward = expected_rewards[request.joint_sample_id]
            self.assertAlmostEqual(credits.row_rewards[index], reward, places=12)
            self.assertEqual(row.completion_token_ids[-1], exp.tokenizer.eos_token_id)
            self.assertEqual(sum(credits.completion_loss_mask[index]), len(row.completion_token_ids))
            first, stop = len(row.prompt_token_ids) - 1, len(row.prompt_token_ids) + len(row.completion_token_ids) - 1
            for position, coefficient in enumerate(credits.loss_coefficients[index]):
                self.assertAlmostEqual(coefficient, -reward / 4 if first <= position < stop else 0, places=12)
            expected_loss += -reward / 4 * math.fsum(record.token_logprobs)
        actual = causal_policy_loss(exp.model, credits, records=records)
        self.assertAlmostEqual(actual.loss.item(), expected_loss, places=6)
        self.assertLessEqual(actual.max_behavior_logprob_error, 2e-6)

    def test_right_padding_preserves_loss_and_gradients(self):
        exp = self.experiment
        _, records, credits = self.batch()
        first = causal_policy_loss(exp.model, credits, records=records)
        first.loss.backward()
        gradients = {name: p.grad.clone() for name, p in exp.model.named_parameters()}
        exp.model.zero_grad(set_to_none=True)
        padded = replace(credits,
                         input_ids=tuple(row + (exp.tokenizer.pad_token_id,) * 7 for row in credits.input_ids),
                         attention_mask=tuple(row + (0,) * 7 for row in credits.attention_mask),
                         completion_loss_mask=tuple(row + (0,) * 7 for row in credits.completion_loss_mask),
                         loss_coefficients=tuple(row + (0.,) * 7 for row in credits.loss_coefficients))
        second = causal_policy_loss(exp.model, padded, records=records)
        second.loss.backward()
        torch.testing.assert_close(first.loss, second.loss)
        for name, parameter in exp.model.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-7, rtol=1e-5)

    def test_two_real_updates_keep_complete_own_base_frozen(self):
        exp = self.experiment
        anchor_files = {p.name: p.read_bytes() for p in exp.root.glob("anchor*") if p.is_file()}
        initial = policy_sha256(exp.model)
        optimizer = torch.optim.SGD(exp.model.parameters(), lr=.01)
        first = exp.step(optimizer, batch_id="step-0", rollout_seed=202)
        second = exp.step(optimizer, batch_id="step-1", rollout_seed=203)
        self.assertEqual(initial, first["policy_before_sha256"])
        self.assertEqual(first["policy_after_sha256"], second["policy_before_sha256"])
        for summary in (first, second):
            self.assertEqual((summary["joint_samples"], summary["trajectories"]), (4, 10))
            self.assertNotEqual(summary["policy_before_sha256"], summary["policy_after_sha256"])
            self.assertGreater(summary["gradient_l2"], 0)
            self.assertGreater(summary["parameter_delta_l2"], 0)
            self.assertEqual(summary["max_behavior_logprob_error"], 0)
        for name, value in anchor_files.items():
            self.assertEqual((exp.root / name).read_bytes(), value)
        with self.assertRaises(FileExistsError):
            exp.step(optimizer, batch_id="step-0", rollout_seed=202)

    def test_reloaded_artifacts_reproduce_generation_and_detect_drift(self):
        exp = self.experiment
        model, tokenizer = exp.reload_initial()
        prompt = exp.tokenizer.encode_messages(exp._messages("A fair coin lands heads."))
        self.assertEqual(prompt, tokenizer.encode_messages(exp._messages("A fair coin lands heads.")))
        samples = [generate_tokens(m, prompt, eos_token_id=tokenizer.eos_token_id, max_new_tokens=8,
                                   generator=torch.Generator().manual_seed(202)) for m in (exp.model, model)]
        self.assertEqual(samples[0], samples[1])
        self.assertEqual(policy_sha256(model), policy_sha256(exp.model))
        for attribute, wrong in (("eos_token_id", 1), ("pad_token_id", 2), ("vocabulary_size", 3),
                                 ("chat_template", "changed")):
            with self.subTest(attribute=attribute), patch.object(exp.tokenizer, attribute, wrong):
                with self.assertRaisesRegex(ValueError, "configuration drift"):
                    exp.prepare(attribute)
        for name in ("initial-model.pt", "tokenizer.json", "chat-template.txt"):
            path = exp.root / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "artifact content changed"):
                exp.reload_initial()
            path.write_bytes(original)

    def test_policy_forward_mode_and_behavior_distribution_drift_reject(self):
        exp = self.experiment
        _, records, credits = self.batch()
        original = copy.deepcopy(exp.model.state_dict())
        with torch.no_grad():
            exp.model.head.bias[0].add_(1)
        with self.assertRaisesRegex(ValueError, "behavior policy"):
            causal_policy_loss(exp.model, credits, records=records)
        exp.model.load_state_dict(original)
        exp.model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            causal_policy_loss(exp.model, credits, records=records)
        exp.model.eval()
        wrong = (replace(records[0], token_logprobs=tuple(x - .01 for x in records[0].token_logprobs)),) + records[1:]
        with self.assertRaisesRegex(ValueError, "distributions disagree"):
            causal_policy_loss(exp.model, credits, records=wrong)
        def mutate(module, inputs, output):
            with torch.no_grad():
                module.head.bias[0].add_(1)
        handle = exp.model.register_forward_hook(mutate)
        try:
            with self.assertRaisesRegex(ValueError, "changed during loss forward"):
                causal_policy_loss(exp.model, credits, records=records)
        finally:
            handle.remove()
            exp.model.load_state_dict(original)

    def test_complete_group_drop_shuffle_and_nontraining_split_reject(self):
        exp = self.experiment
        for ident in ("valid/neg", "heldout/para"):
            with self.assertRaises(ValueError):
                exp.prepare(ident.replace("/", "-"), episode_ids=[ident])
        plan = exp.prepare("groups")
        records = generate_plan(exp.model, plan, exp.tokenizer, generator=torch.Generator().manual_seed(202))
        rows = [r.trajectory for r in records]
        for bad in (rows[::-1], rows[3:]):
            with self.assertRaises(ValueError):
                exp.adapter.score(plan, bad, decode_completion=exp.tokenizer.decode_completion,
                                  pad_token_id=exp.tokenizer.pad_token_id)

    def test_generated_reasoning_and_abstention_stop_without_update_or_retry(self):
        exp = self.experiment
        initial = copy.deepcopy(exp.model.state_dict())
        for token in (259, 260):
            exp.model.load_state_dict(initial)
            with torch.no_grad():
                exp.model.head.bias[token] = 10
                exp.model.transition.weight[token, exp.tokenizer.eos_token_id] += 30
            before = policy_sha256(exp.model)
            optimizer = torch.optim.SGD(exp.model.parameters(), lr=.01)
            name = "bad-" + str(token)
            with self.assertRaisesRegex(ValueError, "invalid/abstained"):
                exp.step(optimizer, batch_id=name, rollout_seed=202)
            self.assertEqual(before, policy_sha256(exp.model))
            rows = [json.loads(line) for line in (exp.root / name / "rollouts.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertTrue(all(row["trajectory"]["completion_token_ids"] == [token, 261] for row in rows))
            self.assertFalse((exp.root / name / "update-tensors.pt").exists())
            self.assertFalse(json.loads((exp.root / name / "failure.json").read_text())["resampled"])

    def test_maximum_length_is_never_mislabeled_completed(self):
        exp = self.experiment
        plan = exp.prepare("truncated")
        records = generate_plan(exp.model, plan, exp.tokenizer, generator=torch.Generator().manual_seed(202), max_new_tokens=1)
        self.assertTrue(all(r.trajectory.finish_reason == "length" for r in records))
        with self.assertRaisesRegex(ValueError, "not a completed"):
            exp.adapter.score(plan, [r.trajectory for r in records],
                              decode_completion=exp.tokenizer.decode_completion, pad_token_id=262)

    def test_partial_rollout_and_decode_faults_retain_tokens(self):
        exp = self.experiment
        plan = exp.prepare("partial")
        calls = 0
        def fail_on_fourth(module, inputs, output):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ValueError("injected forward failure")
        handle = exp.model.register_forward_hook(fail_on_fourth)
        persisted = []
        try:
            with self.assertRaises(RolloutBatchFailure) as caught:
                generate_plan(exp.model, plan, exp.tokenizer, generator=torch.Generator().manual_seed(202),
                              on_record=persisted.append)
        finally:
            handle.remove()
        self.assertEqual(tuple(persisted), caught.exception.records)
        self.assertEqual([len(r.trajectory.completion_token_ids) for r in persisted], [2, 1])
        self.assertEqual(persisted[-1].trajectory.finish_reason, "error")
        with patch.object(exp.tokenizer, "decode_completion", side_effect=ValueError("decode failure")):
            with self.assertRaises(RolloutBatchFailure) as caught:
                generate_plan(exp.model, plan, exp.tokenizer, generator=torch.Generator().manual_seed(202))
        record = caught.exception.records[0]
        self.assertEqual(len(record.trajectory.completion_token_ids), 2)
        self.assertEqual(len(record.token_logprobs), 2)
        self.assertEqual(record.trajectory.finish_reason, "error")
        self.assertEqual(record.trajectory.completion_text, "<DECODE_ERROR>")

    def test_anchor_forward_fault_keeps_prior_and_partial_attempts(self):
        original_forward = TinyCausalModel.forward
        calls = 0
        def fail(model, input_ids, attention_mask=None):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise ValueError("injected anchor failure")
            return original_forward(model, input_ids, attention_mask)
        output = self.root / "failed-anchors"
        with patch.object(TinyCausalModel, "forward", fail):
            with self.assertRaisesRegex(ValueError, "partial attempts retained"):
                SyntheticExperiment(output)
        receipts = [json.loads(line) for line in (output / "anchor-receipts.jsonl").read_text().splitlines()]
        self.assertEqual([len(row["completion_token_ids"]) for row in receipts], [2, 1])
        self.assertEqual(receipts[-1]["finish_reason"], "error")
        self.assertFalse(receipts[-1]["completed"])
        self.assertFalse((output / "anchor.json").exists())
        self.assertTrue((output / "anchor-rng-after.json").exists())
        self.assertFalse(json.loads((output / "anchor-failure.json").read_text())["resampled"])

    def test_optimizer_fault_rolls_back_parameters_and_momentum(self):
        exp = self.experiment
        _, records, credits = self.batch()
        optimizer = torch.optim.SGD(exp.model.parameters(), lr=.01, momentum=.9)
        session = OptimizerSession(exp.model, optimizer)
        before = policy_sha256(exp.model)
        step = optimizer.step
        def fail():
            step()
            raise RuntimeError("injected optimizer failure after mutation")
        with patch.object(optimizer, "step", fail):
            with self.assertRaisesRegex(RuntimeError, "after mutation"):
                session.step(credits, records=records)
        self.assertEqual(before, policy_sha256(exp.model))
        self.assertEqual(optimizer.state_dict()["state"], {})
        self.assertTrue(all(p.grad is None for p in exp.model.parameters()))

    def test_nonfinite_adam_moments_reject_and_restore(self):
        exp = self.experiment
        _, records, credits = self.batch()
        huge = replace(credits, loss_coefficients=tuple(tuple(x * 1e25 for x in row) for row in credits.loss_coefficients))
        optimizer = torch.optim.Adam(exp.model.parameters(), lr=.01)
        before = policy_sha256(exp.model)
        with self.assertRaisesRegex(ValueError, "nonfinite state"):
            OptimizerSession(exp.model, optimizer).step(huge, records=records)
        self.assertEqual(before, policy_sha256(exp.model))
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_zero_learning_rate_cannot_replay_consumed_credits(self):
        exp = self.experiment
        _, records, credits = self.batch()
        optimizer = torch.optim.Adam(exp.model.parameters(), lr=0)
        session = OptimizerSession(exp.model, optimizer)
        result = session.step(credits, records=records)
        self.assertEqual(result["policy_before_sha256"], result["policy_after_sha256"])
        with self.assertRaisesRegex(ValueError, "already consumed"):
            session.step(credits, records=records)

    def test_cli_refuses_existing_output_without_modifying_it(self):
        output = self.root / "existing-output"
        output.mkdir()
        (output / "run-failure.json").write_text("original evidence")
        script = Path(__file__).resolve().parent / "scripts/run_synthetic_cpu_training.py"
        result = subprocess.run([sys.executable, str(script), "--output", str(output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("existing artifacts are never overwritten", result.stderr)
        self.assertEqual(list(output.iterdir()), [output / "run-failure.json"])
        self.assertEqual((output / "run-failure.json").read_text(), "original evidence")

    def test_cli_concurrent_directory_claim_never_overwrites_other_run(self):
        from scripts import run_synthetic_cpu_training as cli
        output = self.root / "concurrent-output"
        def other_run_claims_path(*args, **kwargs):
            output.mkdir()
            (output / "run-failure.json").write_text("other invocation's evidence")
            raise FileExistsError("another invocation created the output")
        with patch.object(sys, "argv", ["synthetic-cli", "--output", str(output)]), \
                patch.object(cli.torch, "set_num_interop_threads"), \
                patch.object(cli, "SyntheticExperiment", side_effect=other_run_claims_path):
            with self.assertRaises(FileExistsError):
                cli.main()
        self.assertEqual((output / "run-failure.json").read_text(), "other invocation's evidence")


@unittest.skipIf(torch is None, "optional pinned CPU environment required")
class AnalyticGradientTests(unittest.TestCase):
    def test_negative_and_zero_rewards_have_expected_softmax_gradient(self):
        class ConstantLogits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([.2, -.1, .4], dtype=torch.float64))

            def forward(self, input_ids, attention_mask):
                return self.logits.expand(*input_ids.shape, 3)

        for reward in (-.7, 0):
            with self.subTest(reward=reward):
                model = ConstantLogits().eval()
                credits = TokenCreditBatch("a" * 64, "b" * 64, policy_sha256(model),
                                           digest(runtime_sampling_distribution(model)), ("role",), ("joint",),
                                           (reward,), ((0, 1, 2),), ((1, 1, 1),), ((1, 1),),
                                           ((-reward, -reward),), 1, ())
                probabilities = model.logits.detach().softmax(-1)
                expected = -reward * (torch.tensor([0., 1., 1.]) - 2 * probabilities)
                loss = causal_policy_loss(model, credits).loss
                loss.backward()
                torch.testing.assert_close(model.logits.grad, expected, atol=1e-14, rtol=1e-14)
                before = model.logits.detach().log_softmax(-1)[1:].sum().item()
                torch.optim.SGD(model.parameters(), lr=.001).step()
                after = model.logits.detach().log_softmax(-1)[1:].sum().item()
                if reward < 0:
                    self.assertLess(after, before)
                else:
                    self.assertEqual(after, before)
                    self.assertEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
