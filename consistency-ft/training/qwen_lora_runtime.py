"""Qwen text-only, adapter-only on-policy runtime; never loads model weights.

Rank16/alpha32 is a provisional engineering treatment, not a scientific choice.
The caller supplies verified local weights, a bound tokenizer, complete own-base
anchors and EpisodeRewardAdapter-scored whole joint batches. No retry or output
filtering is implemented here. Durable attempt directories belong to the caller.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
import uuid

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from anchors import canonical_bytes
from .categorical import categorical_sampler_identity, sample_categorical
from .episode_adapter import GeneratedTrajectory, _sampling_hash
from .torch_loop import GenerationFailure, RolloutBatchFailure, RolloutRecord, TokenGeneration, _finite_state


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def tensor_byte_sha256(tensor, *, chunk_bytes=8 * 1024 * 1024, check_finite=False):
    """Hash contiguous raw bytes in bounded chunks, including BF16 and scalars.

    CPU memoryviews avoid a full `.tobytes()` allocation. Other devices copy only
    one bounded chunk to CPU at a time. No frozen-weight snapshot is retained.
    """
    if tensor.device.type == "meta" or not tensor.is_contiguous() or tensor.layout != torch.strided:
        raise ValueError("identity requires real contiguous strided tensors")
    flat = tensor.detach().reshape(-1)
    width = max(1, chunk_bytes // tensor.element_size())
    result = hashlib.sha256()
    for start in range(0, flat.numel(), width):
        part = flat[start:start + width]
        if check_finite and not bool(torch.isfinite(part).all()):
            raise ValueError("nonfinite tensor in initial frozen base")
        raw = part.view(torch.uint8).cpu().numpy()
        result.update(memoryview(raw))
    return result.hexdigest()


def tensor_manifest(tensors, *, check_finite=False):
    return {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                   "sha256": tensor_byte_sha256(tensor, check_finite=check_finite)}
            for name, tensor in sorted(tensors.items())}


def tensor_stamp(tensor, *, version=True):
    result = (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
              str(tensor.dtype), str(tensor.device), tensor.requires_grad)
    return result + (tensor._version,) if version else result


def kernel_identity(function):
    """Inspect the concrete implementation captured by this pinned dispatcher."""
    captured = inspect.getclosurevars(function).nonlocals
    implementation = captured.get("implementation")
    if implementation is None or "is_new_implementation" not in captured:
        raise ValueError("uninspected Qwen kernel dispatcher; require an explicit reviewed backend")
    return {"entrypoint": function.__module__ + "." + function.__qualname__,
            "selected_implementation": implementation.__module__ + "." + implementation.__qualname__,
            "torch_reference_selected": not captured["is_new_implementation"]}


class LoraLinear(nn.Module):
    def __init__(self, base, *, generator):
        super().__init__()
        self.base = base
        self.rank, self.alpha = 16, 32
        self.in_features, self.out_features = base.in_features, base.out_features
        initial = torch.empty((16, base.in_features), dtype=torch.float32, device="cpu")
        initial.uniform_(-1 / math.sqrt(base.in_features), 1 / math.sqrt(base.in_features), generator=generator)
        self.lora_A = nn.Parameter(initial.to(base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, 16), dtype=torch.float32, device=base.weight.device))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, inputs):
        base_output = self.base(inputs)
        update = F.linear(F.linear(inputs.float(), self.lora_A), self.lora_B) * (self.alpha / self.rank)
        return base_output + update.to(base_output.dtype)


class EvalCheckpointBlock(nn.Module):
    """Non-reentrant checkpointing independent of Transformers' training flag."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, **kwargs):
        if kwargs.get("past_key_values") is not None or kwargs.get("use_cache") is not False:
            raise ValueError("checkpointed decoder requires explicit cache-free execution")
        if torch.is_grad_enabled():
            return checkpoint(self.block, hidden_states, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return self.block(hidden_states, **kwargs)


class QwenLoraRuntime:
    def __init__(self, model, *, lora_seed, expected_target_count=248, checkpoint_decoder=True):
        from transformers import Qwen3_5ForCausalLM
        if type(model) is not Qwen3_5ForCausalLM:
            raise ValueError("supply an actual Qwen3_5ForCausalLM instance")
        if type(lora_seed) is not int or type(expected_target_count) is not int or expected_target_count < 1:
            raise ValueError("explicit seed and positive expected decoder target count required")
        if any(isinstance(module, LoraLinear) for module in model.modules()):
            raise ValueError("model already has runtime LoRA adapters")
        self.model, self.lora_seed = model, lora_seed
        self.poisoned = False
        self._generation_attempts, self._update_attempts = set(), set()
        self._base_tensors = dict(model.named_parameters()) | dict(model.named_buffers())
        devices = {str(tensor.device) for tensor in self._base_tensors.values()}
        if len(devices) != 1 or next(iter(devices)).split(":")[0] not in {"cpu", "cuda"}:
            raise ValueError("runtime requires one actual CPU or CUDA device")
        self.device = next(iter(self._base_tensors.values())).device
        if any(not tensor.is_contiguous() for tensor in self._base_tensors.values()):
            raise ValueError("frozen base tensors must be contiguous")
        targets = [(name, module) for name, module in model.named_modules()
                   if name.startswith("model.layers.") and type(module) is nn.Linear]
        if len(targets) != expected_target_count:
            raise ValueError("decoder Linear count differs from explicit architecture expectation")
        model.requires_grad_(False)
        model.config.use_cache = False
        model.eval()
        self.base_manifest = tensor_manifest(self._base_tensors, check_finite=True)
        self.base_sha256 = digest(self.base_manifest)
        self.target_shapes = {name: list(module.weight.shape) for name, module in targets}
        generator = torch.Generator(device="cpu").manual_seed(lora_seed)
        self._adapters = {}
        for name, module in targets:
            parent_name, attribute = name.rsplit(".", 1)
            adapter = LoraLinear(module, generator=generator)
            setattr(model.get_submodule(parent_name), attribute, adapter)
            self._adapters[name] = adapter
        self.checkpoint_decoder = bool(checkpoint_decoder)
        if checkpoint_decoder:
            for index, layer in enumerate(model.model.layers):
                layer.gradient_checkpointing = False
                model.model.layers[index] = EvalCheckpointBlock(layer)
        model.eval()
        self.adapter_parameters = {name + "." + suffix: parameter
                                   for name, module in self._adapters.items()
                                   for suffix, parameter in (("lora_A", module.lora_A), ("lora_B", module.lora_B))}
        self._base_stamps = {name: tensor_stamp(tensor) for name, tensor in self._base_tensors.items()}
        self._registered_stamps = {name: tensor_stamp(tensor, version=False) for name, tensor in self._live_tensors().items()}
        self._execution = self._execution_identity()
        self._execution_sha256 = digest(self._execution)
        self._quick_guard()

    def _live_tensors(self):
        return dict(self.model.named_parameters()) | dict(self.model.named_buffers())

    def _execution_identity(self):
        import transformers.models.qwen3_5.modeling_qwen3_5 as source
        return {"runtime_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "categorical_sampler": categorical_sampler_identity(),
                "qwen_source_sha256": hashlib.sha256(Path(inspect.getfile(source)).read_bytes()).hexdigest(),
                "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                "device": str(self.device), "model_class": type(self.model).__qualname__,
                "model_config": self.model.config.to_dict(),
                "attention_backend": self.model.config._attn_implementation,
                "base_dtypes": sorted({str(tensor.dtype) for tensor in self._base_tensors.values()}),
                "adapter_dtype": "torch.float32", "normalization_dtype": "torch.float32",
                "checkpoint_decoder": self.checkpoint_decoder, "checkpoint_use_reentrant": False,
                "effective_use_cache": False, "model_mode": "eval", "dropout": 0,
                "lora_rank": 16, "lora_alpha": 32, "lora_seed": self.lora_seed,
                "targets": self.target_shapes, "cpu_threads": torch.get_num_threads(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "kernel_entrypoints": {name: kernel_identity(getattr(source, name))
                                       for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")}}

    def _quick_guard(self):
        if self.poisoned:
            raise ValueError("runtime unusable after immutable-base corruption; reload verified checkpoint")
        if digest(self.base_manifest) != self.base_sha256:
            self.poisoned = True
            raise ValueError("frozen-base manifest changed; runtime unusable")
        if (any(module.training for module in self.model.modules())
                or torch.is_autocast_enabled(self.device.type)
                or digest(self._execution_identity()) != self._execution_sha256
                or any(module.rank != 16 or module.alpha != 32 for module in self._adapters.values())):
            raise ValueError("runtime mode, autocast, backend, configuration or LoRA treatment drift")
        if {name: tensor_stamp(tensor, version=False) for name, tensor in self._live_tensors().items()} != self._registered_stamps:
            self.poisoned = True
            raise ValueError("registered tensor identity/layout/trainability changed; runtime unusable")
        if {name: tensor_stamp(tensor) for name, tensor in self._base_tensors.items()} != self._base_stamps:
            self.poisoned = True
            raise ValueError("frozen-base version/content lifecycle changed; runtime unusable")

    def verify_frozen_base(self):
        """Full streaming content check catches `.data` writes bypassing versions."""
        self._quick_guard()
        if tensor_manifest(self._base_tensors) != self.base_manifest:
            self.poisoned = True
            raise ValueError("frozen-base content changed; runtime unusable; original weights must be reloaded")
        self._quick_guard()
        return self.base_sha256

    def checkpoint_identity(self):
        self.verify_frozen_base()
        return {"base_sha256": self.base_sha256, "base_manifest": copy.deepcopy(self.base_manifest),
                "execution_sha256": self._execution_sha256, "execution": copy.deepcopy(self._execution),
                "adapter_manifest": tensor_manifest(self.adapter_parameters),
                "adapter_parameters": sum(tensor.numel() for tensor in self.adapter_parameters.values())}

    def policy_sha256(self):
        self._quick_guard()
        return digest({"base_sha256": self.base_sha256, "execution_sha256": self._execution_sha256,
                       "adapters": tensor_manifest(self.adapter_parameters)})

    def sampling_distribution(self):
        self._quick_guard()
        return {"schema_version": 1, "sampler": "categorical", "temperature": 1.0,
                "top_p": 1.0, "top_k": 0, "logits_processors": [], "logits_warpers": [],
                "model_mode": "eval", "dropout_enabled": False}

    def _logits(self, prefix, positions):
        self._quick_guard()
        ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            past_key_values=None, logits_to_keep=positions, return_dict=True)
        logits = result.logits
        self._quick_guard()
        if (result.past_key_values is not None or logits.ndim != 3 or logits.shape[0] != 1
                or logits.shape[-1] != self.model.config.vocab_size or not torch.isfinite(logits).all()):
            raise ValueError("model returned cache, malformed logits or nonfinite logits")
        return logits[0].float()

    def _generate_tokens(self, prompt, *, eos_token_id, max_new_tokens, generator, on_token=None):
        if (not prompt or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in prompt)
                or type(eos_token_id) is not int or not 0 <= eos_token_id < self.model.config.vocab_size
                or type(max_new_tokens) is not int or max_new_tokens < 1 or generator.device != self.device):
            raise ValueError("invalid prompt, EOS, generation cap or generator device")
        if on_token is not None and not callable(on_token):
            raise ValueError("on_token must be callable or None")
        tokens, logprobs = [], []
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = self._logits(tuple(prompt) + tuple(tokens), 1)[0]
                    token, logprob = sample_categorical(logits, generator=generator)
                    tokens.append(token)
                    logprobs.append(logprob)
                    if on_token is not None:
                        on_token({"token_index": len(tokens) - 1, "token_id": token,
                                  "sampled_logprob": logprob, "is_eos": token == eos_token_id})
                    if token == eos_token_id:
                        return TokenGeneration(tuple(tokens), tuple(logprobs), "eos")
        except Exception as error:
            raise GenerationFailure(str(error), TokenGeneration(tuple(tokens), tuple(logprobs), "error")) from error
        return TokenGeneration(tuple(tokens), tuple(logprobs), "length")

    def generate_tokens(self, prompt, *, on_token=None, **kwargs):
        """Sample once; optionally persist each token before EOS or another forward.

        The callback receives a fresh event dict after sampling advances the
        supplied generator. It must not mutate the model or generator. Callback
        exceptions stop generation and retain that token in GenerationFailure.
        """
        before = self.policy_sha256()
        try:
            result = self._generate_tokens(prompt, on_token=on_token, **kwargs)
            self.verify_frozen_base()
            if self.policy_sha256() != before:
                raise ValueError("adapter policy changed during generation")
            return result
        except Exception as error:
            partial = error.partial if isinstance(error, GenerationFailure) else locals().get("result", TokenGeneration((), (), "error"))
            try:
                self.verify_frozen_base()
            except Exception as base_error:
                error = ValueError(str(error) + "; " + str(base_error))
            raise GenerationFailure(str(error), partial) from error

    def generate_plan(self, plan, tokenizer, *, generator, max_new_tokens, on_record=None):
        self._validate_plan(plan)
        if plan.plan_sha256 in self._generation_attempts:
            raise ValueError("rollout plan already attempted; do not regenerate")
        if (self.policy_sha256() != plan.policy_sha256
                or _sampling_hash(self.sampling_distribution()) != plan.sampling_distribution_sha256):
            raise ValueError("rollout policy or sampling differs from plan")
        tokenizer.verify_runtime()
        self._generation_attempts.add(plan.plan_sha256)
        records = []
        try:
            for request in plan.requests:
                generated, response, runtime_error = TokenGeneration((), (), "error"), "", None
                try:
                    if tuple(tokenizer.encode_messages(request.messages)) != request.prompt_token_ids:
                        raise ValueError("tokenizer/template prompt differs from plan")
                    generated = self._generate_tokens(request.prompt_token_ids, eos_token_id=tokenizer.eos_token_id,
                                                       max_new_tokens=max_new_tokens, generator=generator)
                    response = tokenizer.decode_completion(generated.completion_token_ids)
                except Exception as error:
                    runtime_error = str(error)
                    if isinstance(error, GenerationFailure):
                        generated = error.partial
                    generated = TokenGeneration(generated.completion_token_ids, generated.token_logprobs, "error")
                    try:
                        response = tokenizer.decode_completion(generated.completion_token_ids)
                    except Exception:
                        response = "<DECODE_ERROR>"
                trajectory = GeneratedTrajectory(request.request_id, plan.plan_sha256, plan.experiment_sha256,
                    plan.policy_sha256, plan.sampling_distribution_sha256, request.messages_sha256,
                    request.prompt_token_ids, generated.completion_token_ids, response, str(uuid.uuid4()),
                    1, False, generated.finish_reason)
                record = RolloutRecord(trajectory, generated.token_logprobs, runtime_error)
                records.append(record)
                if on_record is not None:
                    on_record(record)
                if runtime_error is not None:
                    raise ValueError(runtime_error)
            tokenizer.verify_runtime()
            self.verify_frozen_base()
            if self.policy_sha256() != plan.policy_sha256:
                raise ValueError("adapter weights changed during generation")
        except Exception as error:
            # Audit failed collections too; retain every completed/partial role.
            try:
                self.verify_frozen_base()
            except Exception as base_error:
                error = ValueError(str(error) + "; " + str(base_error))
            raise RolloutBatchFailure(str(error), records) from error
        return tuple(records)

    @staticmethod
    def _validate_plan(plan):
        fields = asdict(plan)
        claimed = fields.pop("plan_sha256")
        if (digest(fields) != claimed or not plan.requests
                or type(plan.joint_sample_count) is not int or plan.joint_sample_count < 1
                or len({request.request_id for request in plan.requests}) != len(plan.requests)
                or len({request.joint_sample_id for request in plan.requests}) != plan.joint_sample_count):
            raise ValueError("original complete rollout plan digest or request structure differs")
        for request in plan.requests:
            if (request.policy_sha256 != plan.policy_sha256 or request.experiment_sha256 != plan.experiment_sha256
                    or request.sampling_distribution_sha256 != plan.sampling_distribution_sha256
                    or request.messages_sha256 != digest(request.messages)
                    or request.joint_sample_id != digest([plan.experiment_sha256, plan.batch_id,
                                                          plan.policy_sha256, request.episode_id, request.sample_index])
                    or request.request_id != digest([request.joint_sample_id, request.role])):
                raise ValueError("rollout request differs from original plan binding")

    def _validate_credits(self, credits, records, *, plan):
        self._validate_plan(plan)
        if (credits.plan_sha256 != plan.plan_sha256 or credits.experiment_sha256 != plan.experiment_sha256
                or credits.policy_sha256 != plan.policy_sha256
                or credits.sampling_distribution_sha256 != plan.sampling_distribution_sha256
                or credits.trajectory_ids != tuple(request.request_id for request in plan.requests)
                or credits.joint_sample_ids != tuple(request.joint_sample_id for request in plan.requests)
                or credits.joint_sample_count != plan.joint_sample_count):
            raise ValueError("credit rows must match every request in the original complete rollout plan")
        if not records or tuple(record.trajectory.request_id for record in records) != credits.trajectory_ids:
            raise ValueError("complete behavior rows must match credit rows")
        if (self.policy_sha256() != credits.policy_sha256
                or _sampling_hash(self.sampling_distribution()) != credits.sampling_distribution_sha256):
            raise ValueError("credit behavior policy or sampling mismatch")
        count = len(records)
        if (type(credits.joint_sample_count) is not int or credits.joint_sample_count < 1
                or any(len(value) != count for value in (credits.input_ids, credits.attention_mask,
                    credits.completion_loss_mask, credits.loss_coefficients, credits.row_rewards, credits.joint_sample_ids))
                or len(set(credits.joint_sample_ids)) != credits.joint_sample_count):
            raise ValueError("whole joint credit batch dimensions differ")
        rows = []
        rewards = {}
        for index, record in enumerate(records):
            row = record.trajectory
            request = plan.requests[index]
            if (record.runtime_error is not None or row.finish_reason not in {"eos", "stop"}
                    or row.plan_sha256 != credits.plan_sha256 or row.policy_sha256 != credits.policy_sha256
                    or row.experiment_sha256 != credits.experiment_sha256
                    or row.sampling_distribution_sha256 != credits.sampling_distribution_sha256
                    or row.messages_sha256 != request.messages_sha256
                    or row.prompt_token_ids != request.prompt_token_ids):
                raise ValueError("errored, truncated or differently bound behavior row")
            tokens = row.prompt_token_ids + row.completion_token_ids
            prompt_length, real_length = len(row.prompt_token_ids), len(tokens)
            ids, attention = credits.input_ids[index], credits.attention_mask[index]
            mask, coefficients = credits.completion_loss_mask[index], credits.loss_coefficients[index]
            reward = credits.row_rewards[index]
            if (not prompt_length or not row.completion_token_ids or not math.isfinite(reward)
                    or any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in tokens)
                    or tuple(ids[:real_length]) != tokens or len(ids) < real_length or len(attention) != len(ids)
                    or tuple(attention) != (1,) * real_length + (0,) * (len(ids) - real_length)
                    or len(mask) != len(ids) - 1 or len(coefficients) != len(mask)
                    or len(record.token_logprobs) != len(row.completion_token_ids)
                    or any(not math.isfinite(value) for value in record.token_logprobs)):
                raise ValueError("malformed complete trajectory credit tensors")
            expected_mask = tuple(int(prompt_length <= position < real_length) for position in range(1, len(ids)))
            if tuple(mask) != expected_mask or any(not math.isfinite(coefficient) or coefficient != (
                    -reward / credits.joint_sample_count if active else 0.) for coefficient, active in zip(coefficients, mask)):
                raise ValueError("credit must preserve global joint denominator and generated-target mask")
            joint = credits.joint_sample_ids[index]
            if joint in rewards and rewards[joint] != reward:
                raise ValueError("joint roles have different rewards")
            rewards[joint] = reward
            rows.append((tokens[:-1], tuple(range(prompt_length - 1, real_length - 1)),
                         row.completion_token_ids, -reward / credits.joint_sample_count))
        return rows

    def role_loss(self, row, behavior_logprobs):
        prefix, positions, targets, coefficient = row
        indices = torch.tensor(positions, dtype=torch.long, device=self.device)
        logits = self._logits(prefix, indices)
        if logits.shape[0] != len(targets):
            raise ValueError("generated-position logits are misaligned")
        logps = logits.log_softmax(-1).gather(-1, torch.tensor(targets, device=self.device)[:, None]).squeeze(-1)
        observed = torch.tensor(behavior_logprobs, dtype=torch.float32, device=self.device)
        if not torch.allclose(logps.detach(), observed, atol=2e-6, rtol=2e-6):
            error = float((logps.detach() - observed).abs().max())
            raise ValueError(f"sampled and evaluated token distributions disagree; max_abs_error={error:.9g}")
        loss = logps.sum() * coefficient
        if not torch.isfinite(loss):
            raise ValueError("nonfinite role policy loss")
        return loss, float((logps.detach() - observed).abs().max())

    def step(self, credits, *, records, optimizer, plan):
        if credits.plan_sha256 in self._update_attempts:
            raise ValueError("credit update already attempted; replay rejected")
        rows = self._validate_credits(credits, records, plan=plan)
        self.verify_frozen_base()
        parameters = list(self.adapter_parameters.values())
        actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if {id(p) for p in parameters} != {id(p) for p in actual} or len(actual) != len(parameters):
            raise ValueError("optimizer must contain exactly the trainable adapters")
        if not _finite_state(optimizer.state_dict()):
            raise ValueError("optimizer starts with nonfinite state")
        before = {name: tensor.detach().cpu().clone() for name, tensor in self.adapter_parameters.items()}
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        policy_before = credits.policy_sha256
        adapter_stamps = {name: tensor_stamp(tensor) for name, tensor in self.adapter_parameters.items()}
        self._update_attempts.add(credits.plan_sha256)
        total_loss, max_error = 0., 0.
        try:
            optimizer.zero_grad(set_to_none=True)
            # Credits for all complete joints were validated before any backward.
            # Each role retains -R/N_joint; losses are summed, never averaged again.
            for row, record in zip(rows, records, strict=True):
                loss, error = self.role_loss(row, record.token_logprobs)
                loss.backward()
                total_loss += float(loss.detach())
                max_error = max(max_error, error)
                del loss
                self._quick_guard()
                if {name: tensor_stamp(tensor) for name, tensor in self.adapter_parameters.items()} != adapter_stamps:
                    raise ValueError("adapter values/version changed during backward")
            if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError("missing or nonfinite adapter gradients")
            if self.policy_sha256() != policy_before:
                raise ValueError("adapter content changed during loss/backward")
            self.verify_frozen_base()  # Full content verification immediately before update.
            gradients = {name: tensor.grad.detach().cpu().clone() for name, tensor in self.adapter_parameters.items()}
            optimizer.step()
            if (any(not torch.isfinite(parameter).all() for parameter in parameters)
                    or not _finite_state(optimizer.state_dict())):
                raise ValueError("optimizer produced nonfinite adapters or state")
            self.verify_frozen_base()
            adapter_state = {name: tensor.detach().cpu().clone() for name, tensor in self.adapter_parameters.items()}
            deltas = {name: tensor - before[name] for name, tensor in adapter_state.items()}
            return {"loss": total_loss, "max_behavior_logprob_error": max_error,
                    "joint_samples": credits.joint_sample_count, "trajectories": len(records),
                    "policy_before_sha256": policy_before, "policy_after_sha256": self.policy_sha256(),
                    "base_sha256": self.base_sha256, "adapter_state": adapter_state,
                    "gradients": gradients, "parameter_deltas": deltas}
        except Exception as error:
            with torch.no_grad():
                for name, parameter in self.adapter_parameters.items():
                    parameter.copy_(before[name].to(parameter.device))
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            try:
                self.verify_frozen_base()
            except Exception as base_error:
                raise ValueError(str(error) + "; " + str(base_error)) from error
            raise
