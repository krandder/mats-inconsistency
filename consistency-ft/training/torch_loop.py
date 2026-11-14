"""CPU custom-loop plumbing; no model download or provider integration.

This module adds real categorical generation and autograd to the existing
episode adapter. It accepts explicitly supplied local torch models/tokenizers.
It does not select a scientific model, alter rewards, or normalize advantages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import uuid

import torch

from anchors import canonical_bytes, sha256
from .categorical import categorical_sampler_identity, sample_categorical
from .episode_adapter import GeneratedTrajectory, _sampling_hash


def digest(value):
    return sha256(canonical_bytes(value))


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return sha256(value.numpy().tobytes())


def policy_manifest(model):
    """Hash actual parameter/buffer values and declared local model structure.

    This detects tensor drift. It is not attestation of arbitrary Python forward
    implementations; the synthetic runner separately records source-file hashes.
    """
    return {"model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "categorical_sampler": categorical_sampler_identity(),
            "configuration": getattr(model, "identity_config", {}),
            "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "sha256": tensor_sha256(value)}
                        for name, value in sorted(model.state_dict().items())}}


def policy_sha256(model):
    return digest(policy_manifest(model))


def runtime_sampling_distribution(model):
    """Report the implemented sampler plus inspected CPU/eval module state."""
    tensors = list(model.parameters()) + list(model.buffers())
    if not tensors or any(value.device.type != "cpu" for value in tensors):
        raise ValueError("this integration requires an explicitly supplied CPU model")
    if any(module.training for module in model.modules()):
        raise ValueError("generation and differentiable loss require eval mode")
    return {"schema_version": 1, "sampler": "categorical", "temperature": 1.0,
            "top_p": 1.0, "top_k": 0, "logits_processors": [], "logits_warpers": [],
            "model_mode": "eval", "dropout_enabled": False}


def _forward(model, input_ids, attention_mask):
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output if isinstance(output, torch.Tensor) else output.logits
    if (logits.device.type != "cpu" or logits.ndim != 3
            or tuple(logits.shape[:2]) != tuple(input_ids.shape)
            or not torch.is_floating_point(logits) or not torch.isfinite(logits).all()):
        raise ValueError("model must return finite CPU logits [rows, input_width, vocabulary]")
    return logits


@dataclass(frozen=True)
class TokenGeneration:
    completion_token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    finish_reason: str


@dataclass(frozen=True)
class RolloutRecord:
    trajectory: GeneratedTrajectory
    token_logprobs: tuple[float, ...]
    runtime_error: str | None = None


class GenerationFailure(ValueError):
    def __init__(self, message, partial):
        super().__init__(message)
        self.partial = partial


class RolloutBatchFailure(ValueError):
    def __init__(self, message, records):
        super().__init__(message)
        self.records = tuple(records)


@dataclass
class TensorLoss:
    loss: torch.Tensor
    causal_logprobs: torch.Tensor
    max_behavior_logprob_error: float | None


def generate_tokens(model, prompt_token_ids, *, eos_token_id, max_new_tokens, generator):
    """One fresh autoregressive rollout, full softmax, no retries or filtering."""
    runtime_sampling_distribution(model)
    if not prompt_token_ids or any(type(x) is not int or x < 0 for x in prompt_token_ids):
        raise ValueError("prompt requires nonnegative integer tokens")
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if type(eos_token_id) is not int or eos_token_id < 0 or generator.device.type != "cpu":
        raise ValueError("explicit EOS and CPU generator required")
    prefix, completion, behavior = list(prompt_token_ids), [], []
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ids = torch.tensor([prefix], dtype=torch.long, device="cpu")
                logits = _forward(model, ids, torch.ones_like(ids))[0, -1]
                if eos_token_id >= logits.numel():
                    raise ValueError("EOS outside model vocabulary")
                token, logp = sample_categorical(logits, generator=generator)
                if not math.isfinite(logp):
                    raise ValueError("nonfinite sampled log probability")
                prefix.append(token)
                completion.append(token)
                behavior.append(logp)
                if token == eos_token_id:
                    return TokenGeneration(tuple(completion), tuple(behavior), "eos")
    except Exception as error:
        raise GenerationFailure(str(error), TokenGeneration(tuple(completion), tuple(behavior), "error")) from error
    return TokenGeneration(tuple(completion), tuple(behavior), "length")


def generate_plan(model, plan, tokenizer, *, generator, max_new_tokens=8, on_record=None):
    """Generate every planned role independently; return attempts even if invalid."""
    sampling = runtime_sampling_distribution(model)
    if (policy_sha256(model) != plan.policy_sha256
            or _sampling_hash(sampling) != plan.sampling_distribution_sha256):
        raise ValueError("rollout weights or sampling distribution differ from plan")
    records = []
    for request in plan.requests:
        runtime_error = None
        generated = TokenGeneration((), (), "error")
        text = ""
        try:
            if tuple(tokenizer.encode_messages(request.messages)) != request.prompt_token_ids:
                raise ValueError("runtime tokenizer/template differs from planned tokens")
            generated = generate_tokens(model, request.prompt_token_ids, eos_token_id=tokenizer.eos_token_id,
                                        max_new_tokens=max_new_tokens, generator=generator)
            text = tokenizer.decode_completion(generated.completion_token_ids)
        except Exception as error:
            runtime_error = str(error)
            if isinstance(error, GenerationFailure):
                generated = error.partial
            else:
                generated = TokenGeneration(generated.completion_token_ids, generated.token_logprobs, "error")
            try:
                text = tokenizer.decode_completion(generated.completion_token_ids)
            except Exception:
                text = "<DECODE_ERROR>"
        trajectory = GeneratedTrajectory(
            request_id=request.request_id, plan_sha256=plan.plan_sha256,
            experiment_sha256=plan.experiment_sha256, policy_sha256=plan.policy_sha256,
            sampling_distribution_sha256=_sampling_hash(sampling), messages_sha256=request.messages_sha256,
            prompt_token_ids=request.prompt_token_ids, completion_token_ids=generated.completion_token_ids,
            completion_text=text,
            context_id=str(uuid.uuid4()), context_turn_count=1, tools_used=False,
            finish_reason=generated.finish_reason)
        record = RolloutRecord(trajectory, generated.token_logprobs, runtime_error)
        records.append(record)
        if on_record is not None:
            on_record(record)
        if runtime_error is not None:
            raise RolloutBatchFailure(runtime_error, records)
    if policy_sha256(model) != plan.policy_sha256:
        raise RolloutBatchFailure("weights changed during rollout collection", records)
    return tuple(records)


def causal_policy_loss(model, credits, *, records=None):
    """Differentiable on-policy loss; coefficient [i,j] targets token [i,j+1]."""
    sampling = runtime_sampling_distribution(model)
    if policy_sha256(model) != credits.policy_sha256:
        raise ValueError("loss weights differ from behavior policy")
    if _sampling_hash(sampling) != credits.sampling_distribution_sha256:
        raise ValueError("loss sampling distribution differs from rollout")
    ids = torch.tensor(credits.input_ids, dtype=torch.long, device="cpu")
    attention = torch.tensor(credits.attention_mask, dtype=torch.long, device="cpu")
    masks = torch.tensor(credits.completion_loss_mask, dtype=torch.bool, device="cpu")
    if (ids.ndim != 2 or ids.shape[1] < 2 or attention.shape != ids.shape
            or masks.shape != (ids.shape[0], ids.shape[1] - 1)
            or len(credits.trajectory_ids) != ids.shape[0]):
        raise ValueError("credit tensor shapes do not align")
    if not torch.all((attention == 0) | (attention == 1)) or torch.any(masks & ~attention[:, 1:].bool()):
        raise ValueError("generated targets must be attended and unpadded")
    logits = _forward(model, ids, attention)[:, :-1]
    # Match generation while preserving existing FP64 analytical use cases.
    if logits.dtype != torch.float64:
        logits = logits.float()
    if (policy_sha256(model) != credits.policy_sha256
            or _sampling_hash(runtime_sampling_distribution(model)) != credits.sampling_distribution_sha256):
        raise ValueError("model state or sampling mode changed during loss forward")
    if torch.any(ids < 0) or torch.any(ids >= logits.shape[-1]):
        raise ValueError("input token outside model vocabulary")
    logps = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    coefficients = torch.tensor(credits.loss_coefficients, dtype=logps.dtype, device="cpu").detach()
    if coefficients.shape != logps.shape or not torch.isfinite(coefficients).all():
        raise ValueError("invalid loss coefficients")
    if torch.any(coefficients[~masks] != 0) or not torch.isfinite(logps).all():
        raise ValueError("prompt/padding credit or nonfinite causal log probabilities")
    error = None
    if records is not None:
        if tuple(r.trajectory.request_id for r in records) != credits.trajectory_ids:
            raise ValueError("behavior log-probability rows differ from credit rows")
        errors = []
        for index, record in enumerate(records):
            observed = torch.tensor(record.token_logprobs, dtype=logps.dtype)
            evaluated = logps[index][masks[index]].detach()
            if (observed.shape != evaluated.shape or not torch.isfinite(observed).all()
                    or not torch.allclose(observed, evaluated, atol=2e-6, rtol=2e-6)):
                raise ValueError("sampled and evaluated token distributions disagree")
            errors.extend((observed - evaluated).abs().tolist())
        error = max(errors, default=0.0)
    loss = (logps * coefficients).sum()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite differentiable loss")
    return TensorLoss(loss, logps, error)


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    return True


class OptimizerSession:
    """Own in-process replay protection, including zero-reward/zero-LR steps.

    The synthetic runner also persists unique attempt directories. This class
    alone is not a crash-resume ledger; restoring arbitrary sessions is unsupported.
    """
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer
        self._consumed_plans = set()

    def step(self, credits, *, records):
        if credits.plan_sha256 in self._consumed_plans:
            raise ValueError("credit plan already consumed by this optimizer session")
        result = _optimizer_step(self.model, self.optimizer, credits, records=records)
        self._consumed_plans.add(credits.plan_sha256)
        return result


def _optimizer_step(model, optimizer, credits, *, records):
    """One validated backward/update; rollback parameters/optimizer on failure.

    The caller obtains credits from adapter.score only after storing all rollout
    attempts. No invalid-output resampling or partial group update occurs here.
    """
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("optimizer parameters do not exactly match this model")
    if not _finite_state(optimizer.state_dict()):
        raise ValueError("optimizer starts with nonfinite state")
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    try:
        result = causal_policy_loss(model, credits, records=records)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        if not gradients or any(not torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError("missing or nonfinite gradients")
        if policy_sha256(model) != credits.policy_sha256:
            raise ValueError("model parameters changed during backward")
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("optimizer produced nonfinite parameters")
        if not _finite_state(optimizer.state_dict()):
            raise ValueError("optimizer produced nonfinite state")
    except Exception:
        model.load_state_dict(before)
        optimizer.load_state_dict(optimizer_before)
        optimizer.zero_grad(set_to_none=True)
        raise
    deltas = {name: value.detach().clone() - before[name] for name, value in model.state_dict().items()}
    return {"loss": float(result.loss.detach()),
            "max_behavior_logprob_error": result.max_behavior_logprob_error,
            "policy_before_sha256": credits.policy_sha256, "policy_after_sha256": policy_sha256(model),
            "gradients": gradients, "parameter_deltas": deltas}
