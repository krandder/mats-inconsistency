"""Bind complete episode rollouts to raw rewards and causal-token loss weights.

This is a custom-loop adapter, not a trainer or a stock TRL reward callback.
It never loads a model or generates tokens. Callers supply verified tokenizer
functions and completed, separately generated trajectories in planned order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import re

from anchors import (a2_reward, canonical_bytes, episode_index, load_snapshot,
                     probability, proposition_id, sha256)
from common import CHECK_FIELDS
from metrics import exploit
from scripts.build_episodes import TRAIN_TYPES, HELDOUT_TYPES


_SAMPLING_DISTRIBUTION = {
    "schema_version": 1,
    "sampler": "categorical",
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "logits_processors": [],
    "logits_warpers": [],
    "model_mode": "eval",
    "dropout_enabled": False,
}


def _digest(value):
    return sha256(canonical_bytes(value))


def _sampling_hash(value):
    # Exact serialization also rejects booleans masquerading as numeric settings.
    if canonical_bytes(value) != canonical_bytes(_SAMPLING_DISTRIBUTION):
        raise ValueError("sampling distribution must be unfiltered categorical at temperature 1 in eval mode")
    return _digest(value)


def _hash(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires a SHA-256 digest")


def _revision(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value):
        raise ValueError(f"{name} requires an immutable commit/content revision, not a branch or alias")


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")


def _timestamp(value, name):
    _text(value, name)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError(f"{name} must include its timezone")
    return parsed


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=constant)


def _source_file(root, name, expected_hash):
    _text(name, "source filename")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("source filename escapes its archive")
    if sha256(path.read_bytes()) != expected_hash:
        raise ValueError("source file hash mismatch")
    return path


def _tokens(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} must contain nonempty token IDs")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in values):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return tuple(values)


def _validate_binding(binding):
    required = {"schema_version", "experiment_id", "training_identity", "protocol_id", "protocol",
                "anchor_sha256", "split_sha256", "arm", "lambda_anchor", "output_parser", "split_policy",
                "loss_reduction", "nonfinite_reward_policy", "outcomes_sha256", "training_cutoff",
                "sampling_distribution"}
    if (set(binding) != required or type(binding["schema_version"]) is not int
            or binding["schema_version"] != 1):
        raise ValueError("experiment binding has missing/unknown fields or schema")
    _text(binding["experiment_id"], "experiment_id")
    _sampling_hash(binding["sampling_distribution"])
    identity = binding["training_identity"]
    expected = {"model_id", "model_revision", "weights_manifest_sha256", "tokenizer_id", "tokenizer_revision",
                "tokenizer_files_sha256", "chat_template_sha256", "protocol_sha256", "reasoning"}
    if set(identity) != expected:
        raise ValueError("training identity fields are incomplete")
    for field in ("model_id", "tokenizer_id", "reasoning"):
        _text(identity[field], field)
    for field in ("model_revision", "tokenizer_revision"):
        _revision(identity[field], field)
    for field in ("weights_manifest_sha256", "tokenizer_files_sha256", "chat_template_sha256", "protocol_sha256"):
        _hash(identity[field], field)
    _text(binding["protocol_id"], "protocol_id")
    if identity["protocol_sha256"] != _digest(binding["protocol"]):
        raise ValueError("protocol content does not match its identity hash")
    for field in ("system_instruction", "verbal_prompt"):
        _text(binding["protocol"].get(field), field)
    if binding["protocol"]["verbal_prompt"].count("{proposition}") != 1:
        raise ValueError("user template must insert the full proposition exactly once")
    _hash(binding["anchor_sha256"], "anchor_sha256")
    if set(binding["split_sha256"]) != {"train", "valid", "heldout"}:
        raise ValueError("all three split hashes must be explicit")
    for split, value in binding["split_sha256"].items():
        _hash(value, split)
    fixed = {"output_parser": "strict-decimal-v1", "split_policy": "episode-ids-and-check-types-v1",
             "loss_reduction": "mean-joint-sample-sum-role-tokens-v1", "nonfinite_reward_policy": "reject-batch"}
    if any(binding[k] != v for k, v in fixed.items()):
        raise ValueError("unsupported parser, split, reduction, or nonfinite-reward policy")
    if binding["arm"] not in {"a0", "a1", "a2", "control-brier"}:
        raise ValueError("unknown experimental arm")
    weight = binding["lambda_anchor"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError("lambda_anchor must be a finite nonnegative number")
    try:
        valid_weight = math.isfinite(weight) and weight >= 0
    except OverflowError:
        valid_weight = False
    if not valid_weight:
        raise ValueError("lambda_anchor must be a finite nonnegative number")
    if binding["arm"] in {"a0", "control-brier"} and weight != 0:
        raise ValueError("A0 and Brier-only control require explicit lambda_anchor=0")
    if binding["arm"] in {"a1", "control-brier"}:
        _hash(binding["outcomes_sha256"], "outcomes_sha256")
        _timestamp(binding["training_cutoff"], "training_cutoff")
    elif binding["outcomes_sha256"] is not None or binding["training_cutoff"] is not None:
        raise ValueError("unresolved arms must not silently consume outcome labels")


@dataclass(frozen=True)
class RolloutRequest:
    request_id: str
    joint_sample_id: str
    episode_id: str
    sample_index: int
    role: str
    messages: tuple[tuple[str, str], ...]
    messages_sha256: str
    prompt_token_ids: tuple[int, ...]
    policy_sha256: str
    sampling_distribution_sha256: str
    experiment_sha256: str


@dataclass(frozen=True)
class RolloutPlan:
    batch_id: str
    experiment_sha256: str
    policy_sha256: str
    sampling_distribution_sha256: str
    requests: tuple[RolloutRequest, ...]
    joint_sample_count: int
    plan_sha256: str


@dataclass(frozen=True)
class GeneratedTrajectory:
    request_id: str
    plan_sha256: str
    experiment_sha256: str
    policy_sha256: str
    sampling_distribution_sha256: str
    messages_sha256: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    completion_text: str
    context_id: str
    context_turn_count: int
    tools_used: bool
    finish_reason: str


@dataclass(frozen=True)
class TokenCreditBatch:
    """Integer input/mask matrices and coefficients for causal log probabilities.

    Coefficient [i,j] multiplies log pi(input_ids[i,j+1] | input_ids[i,:j+1]).
    There is no coefficient for the first input token. Prompt and padding
    positions have zero coefficient. Summing the matrix product is the loss.
    """
    experiment_sha256: str
    plan_sha256: str
    policy_sha256: str
    sampling_distribution_sha256: str
    trajectory_ids: tuple[str, ...]
    joint_sample_ids: tuple[str, ...]
    row_rewards: tuple[float, ...]
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    completion_loss_mask: tuple[tuple[int, ...], ...]
    loss_coefficients: tuple[tuple[float, ...], ...]
    joint_sample_count: int
    episode_metrics: tuple[tuple[str, tuple[tuple[str, float | None], ...]], ...]


class EpisodeRewardAdapter:
    def __init__(self, binding, *, expected_binding_sha256, split_paths, anchor_path,
                 anchor_manifest_path, chat_template, outcomes_path=None):
        # Detach from mutable caller dictionaries before validating and pinning.
        self._binding = json.loads(canonical_bytes(binding))
        _validate_binding(self._binding)
        self.experiment_sha256 = _digest(self._binding)
        self.sampling_distribution_sha256 = _sampling_hash(self._binding["sampling_distribution"])
        if expected_binding_sha256 != self.experiment_sha256:
            raise ValueError("experiment binding hash mismatch")
        if proposition_id(chat_template) != self._binding["training_identity"]["chat_template_sha256"]:
            raise ValueError("chat template content hash mismatch")
        if set(split_paths) != {"train", "valid", "heldout"}:
            raise ValueError("train, valid, and heldout files are all required")
        indices = {name: episode_index(path, name) for name, path in split_paths.items()}
        for name, (meta, rows, _) in indices.items():
            if meta["sha256"] != self._binding["split_sha256"][name]:
                raise ValueError(f"{name} split hash mismatch")
            allowed_types = HELDOUT_TYPES if name == "heldout" else TRAIN_TYPES
            if any(row["check_type"] not in allowed_types for row in rows.values()):
                raise ValueError(f"check type leaked into {name}")
        ids = {name: set(value[1]) for name, value in indices.items()}
        if ids["train"] & (ids["valid"] | ids["heldout"]) or ids["valid"] & ids["heldout"]:
            raise ValueError("episode IDs overlap train/validation/heldout splits")
        self._episodes = indices["train"][1]
        self.split_proposition_overlap = {
            f"{a}/{b}": len(set(indices[a][2]) & set(indices[b][2]))
            for a, b in (("train", "valid"), ("train", "heldout"), ("valid", "heldout"))}
        identity = self._binding["training_identity"]
        base = {"model": identity["model_id"], "reasoning": identity["reasoning"],
                "protocol_id": self._binding["protocol_id"], "protocol_sha256": identity["protocol_sha256"]}
        self._frozen = load_snapshot(anchor_path, expected_sha256=self._binding["anchor_sha256"], expected_base=base,
                                     episodes_path=split_paths["train"], split="train", allow_partial=False)
        self._verify_anchor_checkpoint(anchor_path, anchor_manifest_path)
        self._outcomes = self._load_outcomes(outcomes_path, indices["train"][2])
        self._plans, self._consumed, self._contexts = {}, set(), set()

    def _verify_anchor_checkpoint(self, anchor_path, manifest_path):
        # A sidecar assertion cannot upgrade anonymous native-provider readouts.
        # The original hashed run AND every anchored receipt must carry identity.
        payload = _read_json(anchor_path)["payload"]
        source = payload["source"]
        manifest_path = Path(manifest_path)
        if sha256(manifest_path.read_bytes()) != source["manifest_sha256"]:
            raise ValueError("anchor source manifest hash mismatch")
        manifest = _read_json(manifest_path)
        runs = [r for r in manifest["runs"] if r["run_id"] == source["run_id"]]
        if len(runs) != 1 or _digest(runs[0]) != source["run_metadata_sha256"]:
            raise ValueError("anchor source run metadata mismatch")
        identity = self._binding["training_identity"]
        if runs[0].get("training_identity") != identity:
            raise ValueError("anchor source lacks matching trainable checkpoint/tokenizer/template identity")
        path = _source_file(manifest_path.parent, source["receipts_file"], source["receipts_sha256"])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        current = [r for r in rows if r.get("is_current") is True]
        receipts = {r["id"]: r for r in current}
        if len(receipts) != len(current):
            raise ValueError("duplicate current anchor receipt")
        identity_hash = _digest(identity)
        for ident, anchor in payload["anchors"].items():
            row = receipts.get(ident)
            if (row is None or row.get("training_identity_sha256") != identity_hash
                    or _digest(row) != anchor["published_receipt_sha256"]):
                raise ValueError("anchor observation lacks matching checkpoint identity evidence")

    def _load_outcomes(self, path, train_texts):
        if self._binding["arm"] not in {"a1", "control-brier"}:
            if path is not None:
                raise ValueError("outcome labels are not used by this arm")
            return None
        if path is None or sha256(Path(path).read_bytes()) != self._binding["outcomes_sha256"]:
            raise ValueError("missing or changed pinned outcomes")
        data = _read_json(path)
        if data.get("schema_version") != 1 or set(data.get("outcomes", {})) != set(train_texts):
            raise ValueError("outcomes must cover exactly the training proposition union")
        cutoff = _timestamp(self._binding["training_cutoff"], "training_cutoff")
        result = {}
        for ident, row in data["outcomes"].items():
            value = probability(row.get("value"), "resolved outcome")
            if value not in {0, 1} or row.get("text_sha256") != ident:
                raise ValueError("outcome is unresolved or has a text mismatch")
            if _timestamp(row.get("resolved_at"), "resolved_at") > cutoff:
                raise ValueError("outcome resolved after the bound training cutoff")
            _hash(row.get("evidence_sha256"), "outcome evidence_sha256")
            _text(row.get("source_reference"), "outcome source_reference")
            result[ident] = value
        return result

    def prepare(self, episode_ids, *, samples_per_episode, batch_id, policy_sha256,
                runtime_training_identity, runtime_sampling_distribution, encode_messages):
        """Plan fresh single-turn prompts; encoder applies the bound chat template."""
        if runtime_training_identity != self._binding["training_identity"]:
            raise ValueError("runtime checkpoint/tokenizer/template/protocol/effort identity mismatch")
        if _sampling_hash(runtime_sampling_distribution) != self.sampling_distribution_sha256:
            raise ValueError("runtime sampling distribution mismatch")
        _text(batch_id, "batch_id")
        _hash(policy_sha256, "current rollout policy_sha256")
        if batch_id in self._plans:
            raise ValueError("batch id already planned; use a fresh batch identity")
        if isinstance(samples_per_episode, bool) or not isinstance(samples_per_episode, int) or samples_per_episode < 1:
            raise ValueError("samples_per_episode must be a positive integer")
        if not episode_ids or list(episode_ids) != sorted(set(episode_ids)):
            raise ValueError("episode IDs must be unique and in canonical sorted order")
        if set(episode_ids) - set(self._episodes):
            raise ValueError("validation, heldout, or unknown episode requested for training")
        requests = []
        protocol = self._binding["protocol"]
        for episode_id in episode_ids:
            episode = self._episodes[episode_id]
            for sample in range(samples_per_episode):
                joint = _digest([self.experiment_sha256, batch_id, policy_sha256, episode_id, sample])
                for prompt in episode["prompts"]:
                    messages = (("system", protocol["system_instruction"]),
                                ("user", protocol["verbal_prompt"].format(proposition=prompt["text"])))
                    prompt_ids = _tokens(encode_messages(messages), "encoded prompt")
                    request = RolloutRequest(_digest([joint, prompt["role"]]), joint, episode_id, sample,
                                             prompt["role"], messages, _digest(messages), prompt_ids,
                                             policy_sha256, self.sampling_distribution_sha256, self.experiment_sha256)
                    requests.append(request)
        fields = {"batch_id": batch_id, "experiment_sha256": self.experiment_sha256,
                  "policy_sha256": policy_sha256, "sampling_distribution_sha256": self.sampling_distribution_sha256,
                  "requests": tuple(requests),
                  "joint_sample_count": len(episode_ids) * samples_per_episode}
        plan_hash = _digest({**fields, "requests": [asdict(r) for r in requests]})
        plan = RolloutPlan(**fields, plan_sha256=plan_hash)
        self._plans[batch_id] = plan
        return plan

    def score(self, plan, trajectories, *, decode_completion, pad_token_id):
        """Validate all trajectories before returning any reward/loss coefficients."""
        if self._plans.get(plan.batch_id) != plan or plan.plan_sha256 in self._consumed:
            raise ValueError("unknown, changed, or already consumed rollout plan")
        if len(trajectories) != len(plan.requests):
            raise ValueError("incomplete or oversized trajectory batch")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError("pad_token_id must be a nonnegative integer")
        groups, completions, contexts = {}, [], set()
        for request, trajectory in zip(plan.requests, trajectories, strict=True):
            if (trajectory.request_id != request.request_id or trajectory.plan_sha256 != plan.plan_sha256
                    or trajectory.experiment_sha256 != self.experiment_sha256
                    or trajectory.policy_sha256 != plan.policy_sha256
                    or trajectory.sampling_distribution_sha256 != plan.sampling_distribution_sha256
                    or trajectory.messages_sha256 != request.messages_sha256
                    or _tokens(trajectory.prompt_token_ids, "trajectory prompt") != request.prompt_token_ids):
                raise ValueError("trajectory is shuffled, mismatched, or generated under the wrong binding")
            _text(trajectory.context_id, "fresh context_id")
            if trajectory.context_id in self._contexts or trajectory.context_id in contexts:
                raise ValueError("context was reused across separate role prompts")
            contexts.add(trajectory.context_id)
            if (type(trajectory.context_turn_count) is not int or trajectory.context_turn_count != 1
                    or trajectory.tools_used is not False or trajectory.finish_reason not in {"stop", "eos"}):
                raise ValueError("trajectory is not a completed fresh single-turn, tool-free readout")
            token_ids = _tokens(trajectory.completion_token_ids, "completion")
            decoded = decode_completion(token_ids)
            if not isinstance(decoded, str) or decoded != trajectory.completion_text:
                raise ValueError("completion text does not match its generated token IDs")
            if not re.fullmatch(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", decoded.strip()):
                raise ValueError("invalid/abstained probability output; reject the entire batch")
            value = probability(float(decoded.strip()))
            group = groups.setdefault(request.joint_sample_id, {"episode_id": request.episode_id, "probabilities": {}})
            if request.role in group["probabilities"]:
                raise ValueError("duplicate role within a joint episode sample")
            group["probabilities"][request.role] = value
            completions.append(token_ids)
        metrics = {}
        for joint, group in groups.items():
            episode = self._episodes[group["episode_id"]]
            candidate = group["probabilities"]
            if set(candidate) != set(CHECK_FIELDS[episode["check_type"]]):
                raise ValueError("incomplete episode role group")
            arm, weight = self._binding["arm"], self._binding["lambda_anchor"]
            if arm == "a2":
                result = a2_reward(episode, candidate, self._frozen, lambda_kl=weight)
                reward, exploitation, kl = result["reward"], result["exploitability"], result["mean_kl"]
                brier = None
            else:
                exploitation, kl, brier = exploit(episode["check_type"], candidate), None, None
                if self._outcomes is not None:
                    brier = math.fsum((candidate[p["role"]] - self._outcomes[proposition_id(p["text"])]) ** 2
                                     for p in episode["prompts"]) / len(episode["prompts"])
                reward = -brier if arm == "control-brier" else -exploitation
                if arm == "a1":
                    reward -= weight * brier
            if not math.isfinite(reward):
                raise ValueError("nonfinite episode reward; reject batch without clipping or substitution")
            metrics[joint] = {"reward": reward, "exploitability": exploitation, "mean_kl": kl, "mean_brier": brier}
        if len(groups) != plan.joint_sample_count:
            raise ValueError("joint episode sample count mismatch")
        width = max(len(r.prompt_token_ids) + len(c) for r, c in zip(plan.requests, completions, strict=True))
        inputs, attention, masks, coefficients, row_rewards = [], [], [], [], []
        for request, completion in zip(plan.requests, completions, strict=True):
            prompt = request.prompt_token_ids
            tokens = prompt + completion
            reward = metrics[request.joint_sample_id]["reward"]
            mask = tuple(int(len(prompt) <= pos < len(tokens)) for pos in range(1, width))
            inputs.append(tokens + (pad_token_id,) * (width - len(tokens)))
            attention.append((1,) * len(tokens) + (0,) * (width - len(tokens)))
            masks.append(mask)
            coefficients.append(tuple(-reward / plan.joint_sample_count if active else 0.0 for active in mask))
            row_rewards.append(reward)
        batch = TokenCreditBatch(self.experiment_sha256, plan.plan_sha256, plan.policy_sha256,
                                  plan.sampling_distribution_sha256,
                                  tuple(r.request_id for r in plan.requests),
                                  tuple(r.joint_sample_id for r in plan.requests), tuple(row_rewards),
                                  tuple(inputs), tuple(attention), tuple(masks), tuple(coefficients),
                                  plan.joint_sample_count,
                                  tuple((key, tuple(value.items())) for key, value in metrics.items()))
        self._consumed.add(plan.plan_sha256)
        self._contexts.update(contexts)
        return batch


def categorical_causal_logprobs(causal_logits, input_ids, *, runtime_sampling_distribution):
    """CPU equivalent of unmodified logits.log_softmax(-1).gather(target_ids).

    Shapes are [trajectory, input_width-1, vocabulary] and [trajectory,
    input_width]. This numeric reference has no autograd or model inspection.
    """
    _sampling_hash(runtime_sampling_distribution)
    if not input_ids or len(causal_logits) != len(input_ids):
        raise ValueError("causal logits row count mismatch")
    result, width, vocabulary = [], None, None
    for row_logits, row_ids in zip(causal_logits, input_ids, strict=True):
        ids = _tokens(row_ids, "causal input")
        if len(ids) < 2 or len(row_logits) != len(ids) - 1 or (width is not None and len(ids) != width):
            raise ValueError("causal logits width mismatch")
        width = len(ids)
        logps = []
        for logits, target in zip(row_logits, ids[1:], strict=True):
            if not logits or (vocabulary is not None and len(logits) != vocabulary):
                raise ValueError("causal logits vocabulary mismatch")
            vocabulary = len(logits)
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logits):
                raise ValueError("causal logits must be finite unmodified values")
            if target >= vocabulary:
                raise ValueError("target token is outside the vocabulary")
            maximum = max(logits)
            logp = (logits[target] - maximum) - math.log(math.fsum(math.exp(x - maximum) for x in logits))
            if not math.isfinite(logp):
                raise ValueError("nonfinite categorical log probability")
            logps.append(logp)
        result.append(tuple(logps))
    return tuple(result)


def policy_gradient_loss(causal_logprobs, credits, *, trajectory_ids, policy_sha256,
                         runtime_sampling_distribution):
    """CPU reference loss for an on-policy custom loop, not a differentiable trainer.

    Input shape is [trajectory_count, padded_input_width-1], with entry [i,j]
    equal to log pi(input_ids[i,j+1] | input_ids[i,:j+1]). Rows must be in the
    exact returned trajectory order and evaluated under the rollout policy.
    Tensor loops should multiply these constant coefficients by the equivalent
    differentiable log probabilities and SUM, without another mean/reduction.
    """
    if tuple(trajectory_ids) != credits.trajectory_ids or policy_sha256 != credits.policy_sha256:
        raise ValueError("loss trajectory order or on-policy checkpoint mismatch")
    if _sampling_hash(runtime_sampling_distribution) != credits.sampling_distribution_sha256:
        raise ValueError("loss sampling distribution mismatch")
    if len(causal_logprobs) != len(credits.loss_coefficients):
        raise ValueError("loss row count mismatch")
    terms = []
    for logps, coeffs in zip(causal_logprobs, credits.loss_coefficients, strict=True):
        if len(logps) != len(coeffs):
            raise ValueError("causal log-probability width mismatch")
        for logp, coefficient in zip(logps, coeffs, strict=True):
            if isinstance(logp, bool) or not isinstance(logp, (int, float)) or not math.isfinite(logp) or logp > 0:
                raise ValueError("causal log probabilities must be finite and nonpositive")
            terms.append(logp * coefficient)
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError("nonfinite policy-gradient loss")
    return result
