"""Pinned local Qwen text tokenizer preflight; no weight loading or inference.

The EOS/strict-decimal contract here is a proposed candidate treatment. It is not
proof that pretrained weights obey it. Call verify_runtime before/after each
planned batch; encoding checks inexpensive settings on each individual prompt.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path


MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data/research/qwen-tokenizer-artifacts-20260913.json"
NONTHINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


class CompletionDecodeError(ValueError):
    def __init__(self, message, completion_token_ids):
        super().__init__(message)
        # The caller must retain these exact IDs even when no faithful text
        # decoding exists. Do not repair, filter, or regenerate the sample.
        self.completion_token_ids = tuple(completion_token_ids)


class QwenTextTokenizer:
    def __init__(self, artifact_directory):
        self.artifact_directory = Path(artifact_directory)
        self.manifest = json.loads(MANIFEST_PATH.read_text())
        self._verify_artifacts()
        for name, expected in self.manifest["runtime_packages"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError("unverified tokenizer runtime version: " + name)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.artifact_directory,
                                                        local_files_only=True, trust_remote_code=False)
        self.chat_template = self._tokenizer.chat_template
        if sha256(self.chat_template.encode()) != self.manifest["chat_template_sha256"]:
            raise ValueError("loaded chat template differs from pinned text")
        self._backend_sha256 = sha256(self._tokenizer.backend_tokenizer.to_str().encode())
        self._vocabulary_ids = frozenset(self._tokenizer.get_vocab().values())
        self.eos_token_id = self._tokenizer.eos_token_id
        self.pad_token_id = self._tokenizer.pad_token_id
        self.vocabulary_size = len(self._tokenizer)
        self.model_config = json.loads((self.artifact_directory / "config.json").read_text())
        self._model_config_sha256 = sha256(canonical(self.model_config))
        if (self.eos_token_id, self.pad_token_id, self.vocabulary_size) != (248046, 248044, 248077):
            raise ValueError("unexpected special IDs or vocabulary for pinned Qwen tokenizer")
        self.verify_runtime()

    def _verify_artifacts(self):
        for name, expected in self.manifest["files"].items():
            raw = (self.artifact_directory / name).read_bytes()
            if len(raw) != expected["bytes"] or sha256(raw) != expected["sha256"]:
                raise ValueError("pinned tokenizer artifact differs: " + name)

    def _verify_settings(self):
        tokenizer = self._tokenizer
        if (tokenizer.chat_template != self.chat_template
                or sha256(self.chat_template.encode()) != self.manifest["chat_template_sha256"]
                or (tokenizer.eos_token_id, tokenizer.pad_token_id, len(tokenizer)) != (248046, 248044, 248077)
                or (self.eos_token_id, self.pad_token_id, self.vocabulary_size) != (248046, 248044, 248077)
                or sha256(canonical(self.model_config)) != self._model_config_sha256
                or tokenizer.backend_tokenizer.truncation is not None
                or tokenizer.backend_tokenizer.padding is not None):
            raise ValueError("runtime tokenizer/template/special-ID/truncation/padding drift")

    def verify_runtime(self):
        """Check real artifact bytes and complete loaded backend, outside hot path."""
        self._verify_artifacts()
        self._verify_settings()
        if sha256(self._tokenizer.backend_tokenizer.to_str().encode()) != self._backend_sha256:
            raise ValueError("loaded tokenizer backend changed")
        return {"model_id": self.manifest["model_id"], "inspected_revision": self.manifest["inspected_revision"],
                "tokenizer_files_sha256": sha256(canonical(self.manifest["files"])),
                "chat_template_sha256": self.manifest["chat_template_sha256"],
                "loaded_backend_sha256": self._backend_sha256,
                "runtime_packages": self.manifest["runtime_packages"],
                "enable_thinking": False, "add_generation_prompt": True,
                "proposed_terminal_token_id": self.eos_token_id, "pad_token_id": self.pad_token_id}

    def render_and_encode(self, messages, *, original_user_only=False):
        self._verify_settings()
        messages = tuple(messages)
        roles = ("user",) if original_user_only else ("system", "user")
        if (any(not isinstance(row, (tuple, list)) or len(row) != 2
                or not isinstance(row[1], str) or not row[1] for row in messages)
                or tuple(row[0] for row in messages) != roles):
            raise ValueError("expected fresh text-only " + "/".join(roles) + " messages")
        mapped = [{"role": role, "content": content} for role, content in messages]
        kwargs = {"add_generation_prompt": True, "enable_thinking": False,
                  "tools": None, "documents": None}
        rendered = self._tokenizer.apply_chat_template(mapped, tokenize=False, **kwargs)
        encoded = self._tokenizer.apply_chat_template(mapped, tokenize=True, return_dict=True,
                                                      padding=False, truncation=False, **kwargs)
        tokens = tuple(encoded["input_ids"])
        if (not rendered.endswith(NONTHINKING_SUFFIX)
                or any(value != 1 for value in encoded["attention_mask"])
                or self._tokenizer.decode(tokens, skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False) != rendered):
            raise ValueError("nonthinking prompt suffix, attention or exact tokenizer roundtrip failed")
        return rendered, tokens

    def encode_messages(self, messages):
        """EpisodeRewardAdapter callback for the proposed system/user treatment."""
        return self.render_and_encode(messages)[1]

    def encode_text(self, text):
        """Encode fixture/readout text without adding prompt or termination tokens."""
        self._verify_settings()
        return tuple(self._tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def decode_completion(self, completion_token_ids):
        """Retain all text/controls, omitting only one final proposed EOS token.

        Stock Qwen decoding silently discards IDs outside its tokenizer vocabulary,
        including 243 IDs inside the model's output vocabulary. Reject those IDs
        before decoding so a malformed continuation cannot become a valid number.
        """
        self._verify_settings()
        tokens = tuple(completion_token_ids)
        if any(type(token) is not int or token not in self._vocabulary_ids for token in tokens):
            raise CompletionDecodeError("unmapped or malformed token; retain raw continuation and reject", tokens)
        rendered_tokens = tokens[:-1] if tokens and tokens[-1] == self.eos_token_id else tokens
        return self._tokenizer.decode(rendered_tokens, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)
