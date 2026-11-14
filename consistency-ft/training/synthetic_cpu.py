"""Explicitly synthetic coin fixtures and tiny local model artifacts.

The finite initialization biases below make probability/EOS paths common for
software tests. They are model parameters, not generation-time logits masks.
No scientific checkpoint or natural-language capability is represented.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import re

import torch
from torch import nn

from anchors import canonical_bytes, freeze_from_manifest, proposition_id, sha256, write_snapshot
from .episode_adapter import EpisodeRewardAdapter
from .torch_loop import (GenerationFailure, OptimizerSession, TokenGeneration, digest, generate_plan, generate_tokens,
                         policy_manifest, policy_sha256, runtime_sampling_distribution,
                         tensor_sha256)


CHAT_TEMPLATE = "SYSTEM:{system}\nUSER:{user}\nASSISTANT:"
PROTOCOL = {"system_instruction": "Return one decimal probability in [0,1], with no commentary.",
            "verbal_prompt": "Probability of:\n{proposition}", "parser": "numeric-v1"}
TOKENIZER_CONFIG = {"schema_version": 1, "kind": "synthetic-utf8-pieces-v1",
                    "pieces": ["0.25", "0.50", "0.80", "<think>reasoning</think>0.50", "ABSTAIN"],
                    "eos_token_id": 261, "pad_token_id": 262, "vocabulary_size": 263}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_bytes(canonical_bytes(value) + b"\n")


def write_jsonl(path, values):
    Path(path).write_bytes(b"".join(canonical_bytes(value) + b"\n" for value in values))


class SyntheticTokenizer:
    """Reversible UTF-8 tokenizer with five explicit merged byte strings."""
    def __init__(self, config, chat_template):
        if config != TOKENIZER_CONFIG:
            raise ValueError("unsupported synthetic tokenizer configuration")
        self.config = json.loads(canonical_bytes(config))
        self.chat_template = chat_template
        self.eos_token_id = config["eos_token_id"]
        self.pad_token_id = config["pad_token_id"]
        self.vocabulary_size = config["vocabulary_size"]

    def encode(self, text):
        raw, index, tokens = text.encode(), 0, []
        pieces = sorted(enumerate(self.config["pieces"], 256), key=lambda item: len(item[1]), reverse=True)
        while index < len(raw):
            for token, piece in pieces:
                encoded = piece.encode()
                if raw.startswith(encoded, index):
                    tokens.append(token)
                    index += len(encoded)
                    break
            else:
                tokens.append(raw[index])
                index += 1
        return tuple(tokens)

    def decode_completion(self, tokens):
        tokens = tuple(tokens)
        if tokens and tokens[-1] == self.eos_token_id:
            tokens = tokens[:-1]
        parts = []
        for token in tokens:
            if type(token) is not int or not 0 <= token < self.vocabulary_size:
                raise ValueError("token outside synthetic vocabulary")
            if token < 256:
                parts.append(bytes([token]))
            elif token < 261:
                parts.append(self.config["pieces"][token - 256].encode())
            else:
                parts.append(b"<EOS>" if token == self.eos_token_id else b"<PAD>")
        # Invalid sampled UTF-8 remains visible and cannot pass the decimal parser.
        return b"".join(parts).decode(errors="replace")

    def encode_messages(self, messages):
        if tuple(role for role, _ in messages) != ("system", "user"):
            raise ValueError("expected one fresh system/user prompt")
        return self.encode(self.chat_template.format(system=messages[0][1], user=messages[1][1]))


class TinyCausalModel(nn.Module):
    """Seeded tiny GRU plus trainable bigram logits, constructed entirely locally."""
    def __init__(self, *, seed=20260913, vocabulary_size=263, embedding_size=8, hidden_size=12):
        super().__init__()
        self.identity_config = {"kind": "synthetic-gru-bigram-v1", "seed": seed,
                                "vocabulary_size": vocabulary_size, "embedding_size": embedding_size,
                                "hidden_size": hidden_size, "numeric_bias": "finite-v1"}
        # Local RNG scope does not reseed the independent rollout generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocabulary_size, embedding_size)
            self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, vocabulary_size)
            self.transition = nn.Embedding(vocabulary_size, vocabulary_size)
            with torch.no_grad():
                self.embedding.weight.normal_(0, .1)
                self.head.weight.normal_(0, .02)
                self.head.bias.normal_(-14, .01)
                self.head.bias[256:259].copy_(torch.tensor([.2, 0.0, -.2]))
                self.transition.weight.normal_(0, .01)
                # All logits stay finite. EOS is likely after any merged response.
                self.transition.weight[256:261, 261] += 30.0
        self.eval()

    def forward(self, input_ids, attention_mask=None):
        # GRU starts with a fresh zero hidden state on every call/row. Right
        # padding follows all real tokens and cannot affect preceding logits.
        hidden, _ = self.gru(self.embedding(input_ids))
        return self.head(hidden) + self.transition(input_ids)


def synthetic_splits():
    def negation(ident, text):
        return {"id": ident, "check_type": "Negation", "constraint": "Negation",
                "prompts": [{"role": "P", "text": text}, {"role": "not_P", "text": "Not: " + text}]}
    return {
        "train": [{"id": "train/and", "check_type": "And", "constraint": "And", "prompts": [
            {"role": "P", "text": "The first fair coin lands heads."},
            {"role": "Q", "text": "The second independent fair coin lands heads."},
            {"role": "P_and_Q", "text": "Both independent fair coins land heads."}]},
            negation("train/neg", "A third fair coin lands heads.")],
        "valid": [negation("valid/neg", "A fourth fair coin lands tails.")],
        "heldout": [{"id": "heldout/para", "check_type": "Paraphrase", "constraint": "Paraphrase",
                     "prompts": [{"role": "P", "text": "A fifth fair coin lands heads."},
                                 {"role": "para_P", "text": "The fifth fair coin's result is heads."}]}],
    }


class SyntheticExperiment:
    """Create/reload actual synthetic artifacts and elicit complete own-base anchors."""
    def __init__(self, root, *, model_seed=20260913, anchor_seed=101, lambda_anchor=2.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.model = TinyCausalModel(seed=model_seed)
        self.tokenizer = SyntheticTokenizer(TOKENIZER_CONFIG, CHAT_TEMPLATE)
        self.splits = synthetic_splits()
        self.split_paths = {name: self.root / (name + ".jsonl") for name in self.splits}
        for name, rows in self.splits.items():
            write_jsonl(self.split_paths[name], rows)
        write_json(self.root / "model-config.json", self.model.identity_config)
        write_json(self.root / "tokenizer.json", self.tokenizer.config)
        (self.root / "chat-template.txt").write_text(CHAT_TEMPLATE)
        write_json(self.root / "protocol.json", PROTOCOL)
        torch.save(self.model.state_dict(), self.root / "initial-model.pt")
        write_json(self.root / "weights-manifest.json", policy_manifest(self.model))
        self.artifact_sha256 = {name: sha256((self.root / name).read_bytes()) for name in (
            "model-config.json", "tokenizer.json", "chat-template.txt", "protocol.json",
            "initial-model.pt", "weights-manifest.json")}
        write_json(self.root / "artifact-manifest.json", self.artifact_sha256)
        self.model, self.tokenizer = self.reload_initial()
        self.initial_policy_sha256 = policy_sha256(self.model)
        tokenizer_manifest = {"tokenizer.json": self.artifact_sha256["tokenizer.json"],
                              "chat-template.txt": self.artifact_sha256["chat-template.txt"]}
        self.identity = {"model_id": "synthetic/tiny-gru-probability-v1",
                         "model_revision": self.artifact_sha256["initial-model.pt"],
                         "weights_manifest_sha256": digest(policy_manifest(self.model)),
                         "tokenizer_id": "synthetic/utf8-probability-pieces-v1",
                         "tokenizer_revision": self.artifact_sha256["tokenizer.json"],
                         "tokenizer_files_sha256": digest(tokenizer_manifest),
                         "chat_template_sha256": proposition_id(CHAT_TEMPLATE),
                         "protocol_sha256": digest(PROTOCOL), "reasoning": "none"}
        self.sampling = runtime_sampling_distribution(self.model)
        self.anchor_seed, self.model_seed = anchor_seed, model_seed
        self._elicit_anchors()
        self.binding = {"schema_version": 1, "experiment_id": "synthetic-cpu-integration-v1",
                        "training_identity": self.identity, "sampling_distribution": self.sampling,
                        "protocol_id": "synthetic-decimal-v1", "protocol": PROTOCOL,
                        "anchor_sha256": self.snapshot["snapshot_sha256"],
                        "split_sha256": {name: sha256(path.read_bytes()) for name, path in self.split_paths.items()},
                        "arm": "a2", "lambda_anchor": lambda_anchor, "output_parser": "strict-decimal-v1",
                        "split_policy": "episode-ids-and-check-types-v1",
                        "loss_reduction": "mean-joint-sample-sum-role-tokens-v1",
                        "nonfinite_reward_policy": "reject-batch", "outcomes_sha256": None, "training_cutoff": None}
        write_json(self.root / "experiment-binding.json", self.binding)
        self.adapter = self.make_adapter()
        self._optimizer_sessions = {}

    def reload_initial(self):
        """Reload known local state_dict with weights_only=True, never remote code."""
        for name, expected in self.artifact_sha256.items():
            if sha256((self.root / name).read_bytes()) != expected:
                raise ValueError("initial artifact content changed: " + name)
        config = json.loads((self.root / "model-config.json").read_text())
        model = TinyCausalModel(**{key: config[key] for key in ("seed", "vocabulary_size", "embedding_size", "hidden_size")})
        model.load_state_dict(torch.load(self.root / "initial-model.pt", map_location="cpu", weights_only=True))
        if policy_manifest(model) != json.loads((self.root / "weights-manifest.json").read_text()):
            raise ValueError("loaded tensor manifest does not match initial artifacts")
        tokenizer = SyntheticTokenizer(json.loads((self.root / "tokenizer.json").read_text()),
                                       (self.root / "chat-template.txt").read_text())
        return model.eval(), tokenizer

    def runtime_identity(self):
        for name, expected in self.artifact_sha256.items():
            if sha256((self.root / name).read_bytes()) != expected:
                raise ValueError("initial artifact content changed: " + name)
        if (self.tokenizer.config != json.loads((self.root / "tokenizer.json").read_text())
                or self.tokenizer.chat_template != (self.root / "chat-template.txt").read_text()
                or any(getattr(self.tokenizer, key) != self.tokenizer.config[key]
                       for key in ("eos_token_id", "pad_token_id", "vocabulary_size"))
                or self.model.identity_config != json.loads((self.root / "model-config.json").read_text())):
            raise ValueError("runtime model/tokenizer/template configuration drift")
        return json.loads(canonical_bytes(self.identity))

    def _messages(self, proposition):
        return (("system", PROTOCOL["system_instruction"]),
                ("user", PROTOCOL["verbal_prompt"].format(proposition=proposition)))

    def _elicit_anchors(self):
        generator = torch.Generator(device="cpu").manual_seed(self.anchor_seed)
        propositions = {proposition_id(p["text"]): p["text"] for e in self.splits["train"] for p in e["prompts"]}
        receipts = []
        write_json(self.root / "anchor-rng-before.json", {"seed": self.anchor_seed, "state_sha256": tensor_sha256(generator.get_state())})
        try:
            with (self.root / "anchor-receipts.jsonl").open("xb") as stream:
                for ident, text in sorted(propositions.items()):
                    tokens = self.tokenizer.encode_messages(self._messages(text))
                    sampled, response, runtime_error = TokenGeneration((), (), "error"), "", None
                    try:
                        sampled = generate_tokens(self.model, tokens, eos_token_id=self.tokenizer.eos_token_id,
                                                  max_new_tokens=8, generator=generator)
                        response = self.tokenizer.decode_completion(sampled.completion_token_ids)
                    except Exception as error:
                        runtime_error = str(error)
                        if isinstance(error, GenerationFailure):
                            sampled = error.partial
                        try:
                            response = self.tokenizer.decode_completion(sampled.completion_token_ids)
                        except Exception:
                            response = "<DECODE_ERROR>"
                    valid = (runtime_error is None and sampled.finish_reason == "eos"
                             and re.fullmatch(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", response))
                    receipt = {"id": ident, "is_current": True, "requested_model": self.identity["model_id"],
                               "reasoning": "none", "protocol_id": "synthetic-decimal-v1", "campaign": "synthetic-cpu",
                               "completed": sampled.finish_reason == "eos" and runtime_error is None,
                               "exit_code": 0, "tools_used": False,
                               "error_kind": None if valid else "invalid-synthetic-readout", "runtime_error": runtime_error,
                               "readout_status": "probability" if valid else "invalid",
                               "probability": float(response) if valid else None, "response": response, "finished_at": now(),
                               "training_identity_sha256": digest(self.identity),
                               "policy_sha256": policy_sha256(self.model), "sampling_distribution_sha256": digest(self.sampling),
                               "prompt_sha256": proposition_id(PROTOCOL["verbal_prompt"].format(proposition=text)),
                               "system_instruction_sha256": proposition_id(PROTOCOL["system_instruction"]),
                               "prompt_token_ids": tokens, "completion_token_ids": sampled.completion_token_ids,
                               "token_logprobs": sampled.token_logprobs, "finish_reason": sampled.finish_reason}
                    receipts.append(receipt)
                    stream.write(canonical_bytes(receipt) + b"\n")
                    stream.flush()
                    if runtime_error is not None:
                        raise ValueError("anchor generation failed; partial attempts retained")
        except Exception as error:
            write_json(self.root / "anchor-failure.json", {"error": str(error), "retained_attempts": len(receipts), "resampled": False})
            raise
        finally:
            write_json(self.root / "anchor-rng-after.json", {"state_sha256": tensor_sha256(generator.get_state())})
        if any(row["readout_status"] != "probability" for row in receipts):
            write_json(self.root / "anchor-failure.json", {"error": "invalid anchor readout",
                                                          "retained_attempts": len(receipts), "resampled": False})
            raise ValueError("invalid anchor readout retained; stop without retrying or filling")
        if policy_sha256(self.model) != self.initial_policy_sha256:
            raise ValueError("initial weights changed during anchor elicitation")
        write_jsonl(self.root / "anchor-propositions.jsonl", [{"id": key, "text": value} for key, value in sorted(propositions.items())])
        run = {"run_id": "synthetic/base", "campaign": "synthetic-cpu", "requested_model": self.identity["model_id"],
               "reasoning": "none", "protocol_id": "synthetic-decimal-v1", "receipts_file": "anchor-receipts.jsonl",
               "receipts_sha256": sha256((self.root / "anchor-receipts.jsonl").read_bytes()),
               "recorded_at": now(), "timestamp_source": "local-synthetic-generation", "training_identity": self.identity}
        manifest = {"runs": [run], "native_protocols": {"synthetic-decimal-v1": PROTOCOL},
                    "propositions_file": "anchor-propositions.jsonl",
                    "propositions_sha256": sha256((self.root / "anchor-propositions.jsonl").read_bytes())}
        write_json(self.root / "anchor-manifest.json", manifest)
        self.snapshot = freeze_from_manifest(self.root / "anchor-manifest.json", "synthetic/base",
                                             self.split_paths["train"], "train")
        write_snapshot(self.root / "anchor.json", self.snapshot)

    def make_adapter(self):
        return EpisodeRewardAdapter(self.binding, expected_binding_sha256=digest(self.binding),
                                    split_paths=self.split_paths, anchor_path=self.root / "anchor.json",
                                    anchor_manifest_path=self.root / "anchor-manifest.json", chat_template=CHAT_TEMPLATE)

    def prepare(self, batch_id, *, samples_per_episode=2, episode_ids=None):
        return self.adapter.prepare(episode_ids or sorted(e["id"] for e in self.splits["train"]),
                                    samples_per_episode=samples_per_episode, batch_id=batch_id,
                                    policy_sha256=policy_sha256(self.model), runtime_training_identity=self.runtime_identity(),
                                    runtime_sampling_distribution=runtime_sampling_distribution(self.model),
                                    encode_messages=self.tokenizer.encode_messages)

    def step(self, optimizer, *, batch_id, rollout_seed):
        # A fresh output directory prevents rerunning an attempted batch, even
        # after parsing failed. Receipt assembly can be inspected, not resampled.
        directory = self.root / batch_id
        if directory.parent != self.root or batch_id in {"", ".", ".."}:
            raise ValueError("batch id must name one local output directory")
        directory.mkdir(exist_ok=False)
        plan = self.prepare(batch_id)
        write_json(directory / "plan.json", asdict(plan))
        generator = torch.Generator(device="cpu").manual_seed(rollout_seed)
        write_json(directory / "rng-before.json", {"seed": rollout_seed, "state_sha256": tensor_sha256(generator.get_state())})
        before_hash = policy_sha256(self.model)
        before_state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        try:
            with (directory / "rollouts.jsonl").open("xb") as stream:
                def retain(record):
                    stream.write(canonical_bytes(asdict(record)) + b"\n")
                    stream.flush()
                records = generate_plan(self.model, plan, self.tokenizer, generator=generator, on_record=retain)
            credits = self.adapter.score(plan, [r.trajectory for r in records],
                                         decode_completion=self.tokenizer.decode_completion,
                                         pad_token_id=self.tokenizer.pad_token_id)
            write_json(directory / "credits.json", asdict(credits))
            session = self._optimizer_sessions.setdefault(id(optimizer), OptimizerSession(self.model, optimizer))
            if session.model is not self.model or session.optimizer is not optimizer:
                raise ValueError("optimizer session belongs to a different model instance")
            result = session.step(credits, records=records)
        except Exception as error:
            self.model.load_state_dict(before_state)
            optimizer.load_state_dict(optimizer_before)
            write_json(directory / "failure.json", {"error": str(error), "policy_before_sha256": before_hash,
                                                    "policy_after_sha256": policy_sha256(self.model), "resampled": False})
            raise
        finally:
            write_json(directory / "rng-after.json", {"state_sha256": tensor_sha256(generator.get_state())})
        torch.save({"gradients": result["gradients"], "parameter_deltas": result["parameter_deltas"],
                    "model_state": self.model.state_dict(), "optimizer_state": optimizer.state_dict()}, directory / "update-tensors.pt")
        summary = {key: value for key, value in result.items() if key not in {"gradients", "parameter_deltas"}}
        summary.update({"synthetic_only": True, "rollout_seed": rollout_seed,
                        "joint_samples": plan.joint_sample_count, "trajectories": len(records),
                        "anchor_sha256": self.snapshot["snapshot_sha256"],
                        "gradient_l2": float(sum(g.double().square().sum() for g in result["gradients"].values()).sqrt()),
                        "parameter_delta_l2": float(sum(d.double().square().sum() for d in result["parameter_deltas"].values()).sqrt()),
                        "per_parameter": {name: {"gradient_l2": float(result["gradients"][name].double().norm()),
                                                 "delta_l2": float(result["parameter_deltas"][name].double().norm())}
                                          for name in result["gradients"]}})
        write_json(directory / "summary.json", summary)
        return summary
