"""Reduced actual-Qwen mechanics only; no pretrained/candidate result claims.

Numeric training readouts below are explicit hand-built adapter fixtures. The
separate generation tests sample a reduced random Qwen model, not candidate weights.
"""

import copy
from dataclasses import replace
import hashlib
import importlib.util
import math
import unittest
from unittest.mock import patch

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
if AVAILABLE:
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    import test_training_integration as offline
    import training.qwen_lora_runtime as runtime_module
    from training.qwen_lora_runtime import QwenLoraRuntime, tensor_byte_sha256
    from training.torch_loop import GenerationFailure, RolloutBatchFailure, RolloutRecord


@unittest.skipUnless(AVAILABLE, "optional pinned CPU + Qwen runtime required")
class QwenLoraRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def model(dtype=None):
        config = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
            linear_value_head_dim=8, linear_conv_kernel_dim=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"], max_position_embeddings=1024,
            rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": 1.,
                             "mrope_section": [2, 2, 4]}, pad_token_id=0, eos_token_id=255)
        config._attn_implementation = "eager"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260913)
            model = Qwen3_5ForCausalLM(config).eval()
        if dtype is not None:
            model.to(dtype=dtype)
            # Preserve explicit FP32 recurrent scales, as in the real checkpoint.
            for name, parameter in model.named_parameters():
                if name.endswith(("A_log", "dt_bias")):
                    parameter.data = parameter.data.float()
        return model

    def runtime(self, checkpoint=True, dtype=None):
        return QwenLoraRuntime(self.model(dtype), lora_seed=13, expected_target_count=31,
                               checkpoint_decoder=checkpoint)

    def credits(self, runtime, *, batch_id="batch", include_eos=False):
        fixture = offline.EpisodeAdapterTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        adapter = fixture.adapter()
        plan = adapter.prepare(["train/and", "train/neg"], samples_per_episode=2, batch_id=batch_id,
            policy_sha256=runtime.policy_sha256(), runtime_training_identity=fixture.identity,
            runtime_sampling_distribution=runtime.sampling_distribution(), encode_messages=fixture.encode)
        trajectories = fixture.trajectories(plan)
        if include_eos:
            trajectories = [replace(row, completion_token_ids=row.completion_token_ids + (255,),
                                    finish_reason="eos") for row in trajectories]
        records = []
        with torch.no_grad():
            for row in trajectories:
                ids = torch.tensor([row.prompt_token_ids + row.completion_token_ids])
                logits = runtime.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
                positions = torch.arange(len(row.prompt_token_ids) - 1, ids.shape[1] - 1)
                targets = ids[0, len(row.prompt_token_ids):]
                logps = logits[0, positions].log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)
                records.append(RolloutRecord(row, tuple(logps.tolist())))
        decode = (lambda tokens: fixture.decode(tokens[:-1] if tokens and tokens[-1] == 255 else tokens)) if include_eos else fixture.decode
        credits = adapter.score(plan, trajectories, decode_completion=decode, pad_token_id=0)
        return plan, tuple(records), credits, fixture

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_bf16_raw_hash_is_chunked_and_dtype_safe(self):
        for tensor in (torch.tensor(1.25, dtype=torch.bfloat16), torch.arange(17, dtype=torch.bfloat16),
                       torch.tensor([], dtype=torch.bfloat16), torch.arange(8, dtype=torch.float32)):
            expected = hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            self.assertEqual(tensor_byte_sha256(tensor, chunk_bytes=3), expected)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            tensor_byte_sha256(torch.ones(3, 3).t())

    def test_zero_initialized_adapters_preserve_exact_logits_and_frozen_base(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = self.model(dtype)
            ids = torch.tensor([[2, 3, 4, 5, 6]])
            with torch.no_grad():
                before = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.clone()
            runtime = QwenLoraRuntime(model, lora_seed=13, expected_target_count=31)
            with torch.no_grad():
                after = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(len(runtime.target_shapes), 31)
            self.assertEqual(sum(p.numel() for p in runtime.adapter_parameters.values()), 34240)
            self.assertTrue(all(p.dtype == torch.float32 and p.requires_grad for p in runtime.adapter_parameters.values()))
            self.assertTrue(all(not p.requires_grad for p in runtime._base_tensors.values()))
            self.assertTrue(all(not module.training for module in model.modules()))
            runtime.verify_frozen_base()

    def test_last_position_generation_and_selected_causal_loss_logits(self):
        runtime = self.runtime()
        widths = []
        hook = runtime.model.lm_head.register_forward_pre_hook(lambda module, args: widths.append(args[0].shape[1]))
        generated = runtime.generate_tokens((2, 3, 4), eos_token_id=255, max_new_tokens=3,
                                             generator=torch.Generator().manual_seed(12))
        self.assertEqual(widths, [1, 1, 1])
        with torch.no_grad():
            logits = runtime._logits((2, 3, 4) + generated.completion_token_ids[:-1], torch.arange(2, 5))
            actual = logits.log_softmax(-1).gather(-1, torch.tensor(generated.completion_token_ids)[:, None]).squeeze(-1)
        torch.testing.assert_close(actual, torch.tensor(generated.token_logprobs), atol=2e-6, rtol=2e-6)
        plan, records, credits, _ = self.credits(runtime, include_eos=True)
        widths.clear()
        rows = runtime._validate_credits(credits, records, plan=plan)
        loss, error = runtime.role_loss(rows[-1], records[-1].token_logprobs)
        self.assertEqual(widths, [len(records[-1].trajectory.completion_token_ids)])
        self.assertEqual(records[-1].trajectory.completion_token_ids[-1], 255)
        self.assertAlmostEqual(loss.item(), -credits.row_rewards[-1] / 4 * sum(records[-1].token_logprobs), places=5)
        self.assertLessEqual(error, 2e-6)
        hook.remove()

    def test_token_callback_preserves_exact_sampling_rng_and_eos(self):
        runtime = self.runtime()
        prompt = (2, 3, 4)
        preliminary = runtime.generate_tokens(prompt, eos_token_id=255, max_new_tokens=3,
                                               generator=torch.Generator().manual_seed(12))
        self.assertEqual(preliminary.finish_reason, "length")
        self.assertNotEqual(preliminary.completion_token_ids[0], preliminary.completion_token_ids[1])
        original = runtime._logits
        for eos in (255, preliminary.completion_token_ids[1]):
            with self.subTest(eos=eos):
                plain_generator = torch.Generator().manual_seed(12)
                plain_forward_states = []
                def plain_forward(prefix, positions):
                    plain_forward_states.append(plain_generator.get_state().clone())
                    return original(prefix, positions)
                with patch.object(runtime, "_logits", plain_forward):
                    plain = runtime.generate_tokens(prompt, eos_token_id=eos, max_new_tokens=3,
                                                    generator=plain_generator)
                observed_generator = torch.Generator().manual_seed(12)
                events, callback_states, order = [], [], []
                def observed_forward(prefix, positions):
                    order.append("forward")
                    return original(prefix, positions)
                def observe(event):
                    order.append("callback")
                    events.append(dict(event))
                    callback_states.append(observed_generator.get_state().clone())
                    event.clear()  # Receipt mutation must not alter retained tokens.
                with patch.object(runtime, "_logits", observed_forward):
                    observed = runtime.generate_tokens(prompt, eos_token_id=eos, max_new_tokens=3,
                                                       generator=observed_generator, on_token=observe)
                self.assertEqual(observed, plain)
                self.assertTrue(torch.equal(observed_generator.get_state(), plain_generator.get_state()))
                self.assertEqual(order, [value for _ in events for value in ("forward", "callback")])
                self.assertEqual(events, [dict(token_index=index, token_id=token,
                                              sampled_logprob=plain.token_logprobs[index], is_eos=token == eos)
                                          for index, token in enumerate(plain.completion_token_ids)])
                expected_states = plain_forward_states[1:] + [plain_generator.get_state()]
                for observed_state, expected_state in zip(callback_states, expected_states, strict=True):
                    self.assertTrue(torch.equal(observed_state, expected_state))
                self.assertEqual(observed.finish_reason, "length" if eos == 255 else "eos")

    def test_actual_reduced_qwen_matches_independent_cdf_oracle(self):
        from training.categorical import categorical_sampler_identity

        runtime = self.runtime(dtype=torch.bfloat16)
        prompt = (2, 3, 4)
        with torch.no_grad():
            logits = runtime._logits(prompt, 1)[0]
        weights = logits.softmax(-1).tolist()
        reference = torch.Generator().manual_seed(12)
        uniform = torch.rand((), dtype=torch.float64, generator=reference).item()
        total = math.fsum(weights)
        expected = next(index for index in range(len(weights))
                        if uniform < math.fsum(weights[:index + 1]) / total)
        generator = torch.Generator().manual_seed(12)
        generated = runtime.generate_tokens(prompt, eos_token_id=255, max_new_tokens=1, generator=generator)
        self.assertEqual(generated.completion_token_ids, (expected,))
        self.assertEqual(generated.token_logprobs, (logits.log_softmax(-1)[expected].item(),))
        self.assertTrue(torch.equal(generator.get_state(), reference.get_state()))
        self.assertEqual(runtime._execution_identity()["categorical_sampler"], categorical_sampler_identity())

    def test_token_callback_failure_retains_sample_and_stops_without_retry(self):
        runtime = self.runtime()
        prompt = (2, 3, 4)
        reference_generator = torch.Generator().manual_seed(12)
        reference = runtime.generate_tokens(prompt, eos_token_id=255, max_new_tokens=2,
                                            generator=reference_generator)
        for eos in (255, reference.completion_token_ids[-1]):
            with self.subTest(eos=eos):
                generator = torch.Generator().manual_seed(12)
                events = []
                def fail(event):
                    events.append(event)
                    if len(events) == 2:
                        raise OSError("durable token write failed")
                with patch.object(runtime, "_logits", wraps=runtime._logits) as forwards:
                    with self.assertRaisesRegex(GenerationFailure, "durable token write failed") as caught:
                        runtime.generate_tokens(prompt, eos_token_id=eos, max_new_tokens=3,
                                                generator=generator, on_token=fail)
                self.assertEqual(forwards.call_count, 2)
                self.assertEqual(len(events), 2)
                self.assertEqual(caught.exception.partial.completion_token_ids, reference.completion_token_ids)
                self.assertEqual(caught.exception.partial.token_logprobs, reference.token_logprobs)
                self.assertEqual(caught.exception.partial.finish_reason, "error")
                self.assertTrue(torch.equal(generator.get_state(), reference_generator.get_state()))
                self.assertEqual(events[-1]["is_eos"], eos == reference.completion_token_ids[-1])
                runtime.verify_frozen_base()

    def test_microbatch_gradients_match_uncheckpointed_joint_objective(self):
        checkpointed, plain = self.runtime(), self.runtime(checkpoint=False)
        plan, records, credits, _ = self.credits(checkpointed, include_eos=True)
        plain_plan, plain_records, plain_credits, _ = self.credits(plain, include_eos=True)
        optimizer = torch.optim.SGD(checkpointed.adapter_parameters.values(), lr=.01)
        rows = plain._validate_credits(plain_credits, plain_records, plan=plain_plan)
        whole_loss = sum(plain.role_loss(row, record.token_logprobs)[0] for row, record in zip(rows, plain_records))
        whole_loss.backward()
        expected_gradients = {name: p.grad.clone() for name, p in plain.adapter_parameters.items()}
        with patch.object(runtime_module, "checkpoint", wraps=runtime_module.checkpoint) as checkpoints:
            result = checkpointed.step(credits, records=records, optimizer=optimizer, plan=plan)
        self.assertEqual(checkpoints.call_count, 40)  # Four blocks × ten role forwards under eval.
        self.assertTrue(all(call.kwargs["use_reentrant"] is False for call in checkpoints.call_args_list))
        self.assertAlmostEqual(result["loss"], whole_loss.item(), places=5)
        self.assertEqual((result["joint_samples"], result["trajectories"]), (4, 10))
        self.assertNotEqual(result["policy_before_sha256"], result["policy_after_sha256"])
        self.assertEqual(set(result["adapter_state"]), set(checkpointed.adapter_parameters))
        for name, value in expected_gradients.items():
            torch.testing.assert_close(result["gradients"][name], value, atol=2e-7, rtol=2e-5)
        checkpointed.verify_frozen_base()

    def test_wrong_global_denominator_and_incomplete_rows_reject_before_backward(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=.01)
        wrong = replace(credits, loss_coefficients=tuple(tuple(x / 10 for x in row) for row in credits.loss_coefficients))
        with self.assertRaisesRegex(ValueError, "global joint denominator"):
            runtime.step(wrong, records=records, optimizer=optimizer, plan=plan)
        with self.assertRaisesRegex(ValueError, "complete behavior"):
            runtime.step(credits, records=records[:-1], optimizer=optimizer, plan=plan)
        self.assertTrue(all(p.grad is None for p in runtime.adapter_parameters.values()))

    def test_sampled_evaluated_mismatch_stops_update_without_relaxing_tolerance(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        wrong = (replace(records[0], token_logprobs=tuple(x + .01 for x in records[0].token_logprobs)),) + records[1:]
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=.01)
        before = runtime.policy_sha256()
        with self.assertRaisesRegex(ValueError, "distributions disagree"):
            runtime.step(credits, records=wrong, optimizer=optimizer, plan=plan)
        self.assertEqual(runtime.policy_sha256(), before)
        self.assertTrue(all(p.grad is None for p in runtime.adapter_parameters.values()))

    def test_coherently_removed_role_cannot_bypass_original_plan(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        fields = ("trajectory_ids", "joint_sample_ids", "row_rewards", "input_ids", "attention_mask",
                  "completion_loss_mask", "loss_coefficients")
        incomplete = replace(credits, **{name: getattr(credits, name)[1:] for name in fields})
        self.assertEqual(len(set(incomplete.joint_sample_ids)), 4)
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=.01)
        with patch.object(optimizer, "step") as update:
            with self.assertRaisesRegex(ValueError, "every request.*original complete"):
                runtime.step(incomplete, records=records[1:], optimizer=optimizer, plan=plan)
            update.assert_not_called()
        self.assertTrue(all(parameter.grad is None for parameter in runtime.adapter_parameters.values()))
        changed_plan = replace(plan, requests=plan.requests[1:])
        with self.assertRaisesRegex(ValueError, "plan digest"):
            runtime.step(incomplete, records=records[1:], optimizer=optimizer, plan=changed_plan)

    def test_zero_lr_replay_is_rejected(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=0)
        result = runtime.step(credits, records=records, optimizer=optimizer, plan=plan)
        self.assertEqual(result["policy_before_sha256"], result["policy_after_sha256"])
        with self.assertRaisesRegex(ValueError, "replay"):
            runtime.step(credits, records=records, optimizer=optimizer, plan=plan)

    def test_optimizer_failure_restores_adapter_and_existing_momentum_only(self):
        runtime = self.runtime()
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=.01, momentum=.9)
        plan, records, credits, _ = self.credits(runtime, batch_id="first")
        runtime.step(credits, records=records, optimizer=optimizer, plan=plan)
        plan, records, credits, _ = self.credits(runtime, batch_id="second")
        before, state = runtime.policy_sha256(), copy.deepcopy(optimizer.state_dict())
        step = optimizer.step
        def fail():
            step()
            raise RuntimeError("after update failure")
        with patch.object(optimizer, "step", fail):
            with self.assertRaisesRegex(RuntimeError, "after update"):
                runtime.step(credits, records=records, optimizer=optimizer, plan=plan)
        self.assertEqual(before, runtime.policy_sha256())
        self.assert_nested_equal(state, optimizer.state_dict())
        self.assertTrue(all(p.grad is None for p in runtime.adapter_parameters.values()))

    def test_data_version_bypass_during_backward_poisoned_before_optimizer(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        optimizer = torch.optim.SGD(runtime.adapter_parameters.values(), lr=.01)
        base = next(iter(runtime._base_tensors.values()))
        version = base._version
        def corrupt(gradient):
            base.data.view(-1)[0].add_(1)
            return gradient
        hook = next(iter(runtime.adapter_parameters.values())).register_hook(corrupt)
        with patch.object(optimizer, "step") as update:
            with self.assertRaisesRegex(ValueError, "frozen-base content changed"):
                runtime.step(credits, records=records, optimizer=optimizer, plan=plan)
            update.assert_not_called()
        hook.remove()
        self.assertEqual(base._version, version)
        self.assertTrue(runtime.poisoned)
        with self.assertRaisesRegex(ValueError, "unusable"):
            runtime.policy_sha256()

    def test_nonfinite_adam_moments_restore_adapters_and_optimizer(self):
        runtime = self.runtime()
        plan, records, credits, _ = self.credits(runtime)
        huge = replace(credits, row_rewards=tuple(reward * 1e25 for reward in credits.row_rewards),
            loss_coefficients=tuple(tuple(-reward * 1e25 / credits.joint_sample_count if active else 0.
                                         for active in mask)
                                    for reward, mask in zip(credits.row_rewards, credits.completion_loss_mask)))
        optimizer = torch.optim.Adam(runtime.adapter_parameters.values(), lr=.01)
        before = runtime.policy_sha256()
        with self.assertRaisesRegex(ValueError, "nonfinite adapters or state"):
            runtime.step(huge, records=records, optimizer=optimizer, plan=plan)
        self.assertEqual(before, runtime.policy_sha256())
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_generation_end_full_hash_detects_data_bypass_and_retains_attempt(self):
        runtime = self.runtime()
        plan, _, _, fixture = self.credits(runtime)
        class FixtureTokenizer:
            eos_token_id = 255
            def verify_runtime(self):
                return {}
            encode_messages = staticmethod(fixture.encode)
            decode_completion = staticmethod(fixture.decode)
        base = next(iter(runtime._base_tensors.values()))
        original = runtime._logits
        def corrupt(prefix, positions):
            result = original(prefix, positions)
            base.data.view(-1)[0].add_(.1)
            return result
        saved = []
        with patch.object(runtime, "_logits", corrupt):
            with self.assertRaises(RolloutBatchFailure) as caught:
                runtime.generate_plan(plan, FixtureTokenizer(), generator=torch.Generator().manual_seed(12),
                                      max_new_tokens=1, on_record=saved.append)
        self.assertEqual(tuple(saved), caught.exception.records)
        self.assertEqual(len(saved), 10)
        self.assertTrue(all(len(record.trajectory.completion_token_ids) == 1 for record in saved))
        self.assertTrue(runtime.poisoned)

    def test_mode_backend_autocast_and_base_version_drift_reject(self):
        runtime = self.runtime()
        runtime.model.train()
        with self.assertRaisesRegex(ValueError, "mode"):
            runtime.policy_sha256()
        runtime.model.eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(ValueError, "autocast"):
                runtime.policy_sha256()
        original_backend = runtime.model.config._attn_implementation
        runtime.model.config._attn_implementation = "sdpa"
        try:
            with self.assertRaisesRegex(ValueError, "backend"):
                runtime.policy_sha256()
        finally:
            runtime.model.config._attn_implementation = original_backend
        with torch.no_grad():
            next(iter(runtime._base_tensors.values())).view(-1)[0].add_(1)
        with self.assertRaisesRegex(ValueError, "version"):
            runtime.policy_sha256()
        self.assertTrue(runtime.poisoned)

    def test_partial_generation_and_decode_failure_retain_all_tokens(self):
        runtime = self.runtime()
        plan, _, _, fixture = self.credits(runtime)
        class FixtureTokenizer:
            eos_token_id = 255
            def verify_runtime(self):
                return {"synthetic_fixture": True}
            encode_messages = staticmethod(fixture.encode)
            decode_completion = staticmethod(fixture.decode)
        tokenizer = FixtureTokenizer()
        calls, original = 0, runtime._logits
        def fail(prefix, positions):
            nonlocal calls
            calls += 1
            if calls == 5:
                raise ValueError("partial forward failure")
            return original(prefix, positions)
        saved = []
        with patch.object(runtime, "_logits", fail):
            with self.assertRaises(RolloutBatchFailure) as caught:
                runtime.generate_plan(plan, tokenizer, generator=torch.Generator().manual_seed(12),
                                       max_new_tokens=3, on_record=saved.append)
        self.assertEqual(tuple(saved), caught.exception.records)
        self.assertEqual([len(r.trajectory.completion_token_ids) for r in saved], [3, 1])
        with self.assertRaisesRegex(ValueError, "already attempted"):
            runtime.generate_plan(plan, tokenizer, generator=torch.Generator().manual_seed(12), max_new_tokens=3)
        plan, _, _, _ = self.credits(runtime, batch_id="decode")
        with patch.object(tokenizer, "decode_completion", side_effect=ValueError("decode fault")):
            with self.assertRaises(RolloutBatchFailure) as caught:
                runtime.generate_plan(plan, tokenizer, generator=torch.Generator().manual_seed(12), max_new_tokens=3)
        self.assertEqual(len(caught.exception.records[0].trajectory.completion_token_ids), 3)
        self.assertEqual(caught.exception.records[0].trajectory.completion_text, "<DECODE_ERROR>")


if __name__ == "__main__":
    unittest.main()
