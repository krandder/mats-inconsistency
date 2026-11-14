"""Durable, complete-original-split own-base collection; no scientific defaults.

Planning, inspecting and exporting never import Torch or load candidate weights.
The sole inference entry point is collect(), using the existing audited runtime.
An attempt directory permanently consumes its proposition even if the process
dies before its first token. Such a campaign cannot export a complete anchor.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import socket
import stat
import tempfile
import time
import traceback
import uuid

from anchors import canonical_bytes, episode_index, freeze_from_manifest, proposition_id, sha256, write_snapshot
from .categorical import categorical_sampler_identity

ROOT = Path(__file__).resolve().parents[1]
SPLIT_HASHES = {
    "train": "71ccc77578c201dc7f51bf06cb9f48f1fecfc1850e3f0472aa349a777924fd2e",
    "valid": "d45342c7d00ceb4fd9642e97647a695c3760d7a958ceb5461173008bd5bc4150",
    "heldout": "d5b9b660f9fb8c39f7313892897644350247b4c0173bb22871912cdc9fc0674e",
}
SOURCE_FILES = (
    "training/qwen_base_collection.py", "scripts/collect_qwen_base.py", "training/qwen_checkpoint.py",
    "training/categorical.py",
    "training/qwen_tokenizer.py", "training/qwen_lora_runtime.py", "training/torch_loop.py",
    "training/episode_adapter.py", "anchors.py", "common.py", "metrics.py", "scripts/build_episodes.py",
    "harness/run_native_checks.py", "harness/scope.py",
)
SAMPLING = {"schema_version": 1, "sampler": "categorical", "temperature": 1.0, "top_p": 1.0,
            "top_k": 0, "logits_processors": [], "logits_warpers": [], "model_mode": "eval",
            "dropout_enabled": False}
MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def digest(value):
    return sha256(canonical_bytes(value))


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def read_file(path, limit=64 * 1024 * 1024):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("expected bounded regular file: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if len(raw) > limit or any(getattr(before, key) != getattr(after, key) for key in
                               ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
        raise ValueError("file changed while reading: " + str(path))
    return raw


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create_file(path, raw):
    """Atomic no-replace publication, including fsync of parent directory."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(temporary)


def create_json(path, value):
    create_file(path, canonical_bytes(value) + b"\n")


def make_directory(path):
    path = Path(path)
    path.mkdir()
    fsync_directory(path.parent)


@contextmanager
def owner_lock(directory, *, shared=False):
    """One stable, never-replaced flock inode; owner death releases the lock."""
    path = Path(directory) / "owner.lock"
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def pinned_input(spec, key):
    row = spec[key]
    if set(row) != {"path", "sha256"}:
        raise ValueError("input requires explicit path/SHA256: " + key)
    raw = read_file(row["path"])
    if sha256(raw) != row["sha256"]:
        raise ValueError("external input pin differs: " + key)
    return raw


def source_identity():
    return {name: sha256(read_file(ROOT / name)) for name in SOURCE_FILES}


def validate_spec(spec, inputs):
    required = {"schema_version", "collection_id", "master_seed", "max_new_tokens", "protocol_id", "protocol",
                "sources", "execution", "initial_runtime_identity", "checkpoint_manifest", "tokenizer_manifest",
                "train", "valid", "heldout"}
    if set(spec) != required or type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("explicit collection spec fields differ; no lambda, smoothing or model defaults")
    if not isinstance(spec["collection_id"], str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}", spec["collection_id"]):
        raise ValueError("collection_id must be a stable simple identifier")
    if (type(spec["master_seed"]) is not int or not 0 <= spec["master_seed"] < 2**64
            or type(spec["max_new_tokens"]) is not int or not 1 <= spec["max_new_tokens"] <= 4096):
        raise ValueError("explicit uint64 master seed and 1..4096 token cap required")
    if spec["sources"] != source_identity():
        raise ValueError("externally pinned collector/runtime/dependency source differs")
    from harness.run_native_checks import PROTOCOLS
    from training.qwen_checkpoint import PINNED_FILES, UPSTREAM_METADATA_SHA256

    if spec["protocol_id"] != "numeric-v1" or spec["protocol"] != PROTOCOLS["numeric-v1"]:
        raise ValueError("this driver implements the reviewed original numeric-v1 protocol only")
    checkpoint = read_json(inputs["checkpoint_manifest"])
    if (checkpoint.get("schema_version") != 1 or checkpoint.get("model_id") != MODEL_ID
            or checkpoint.get("revision") != REVISION
            or checkpoint.get("upstream_metadata_sha256") != UPSTREAM_METADATA_SHA256
            or set(checkpoint.get("files", {})) != set(PINNED_FILES)
            or any((checkpoint["files"][name].get("bytes"), checkpoint["files"][name].get("sha256")) != pair
                   for name, pair in PINNED_FILES.items())):
        raise ValueError("checkpoint manifest differs from the published pinned candidate")
    tokenizer = read_json(inputs["tokenizer_manifest"])
    if (inputs["tokenizer_manifest"] != read_file(ROOT / "data/research/qwen-tokenizer-artifacts-20260913.json")
            or tokenizer["model_id"] != MODEL_ID or tokenizer["inspected_revision"] != REVISION):
        raise ValueError("tokenizer manifest differs from the published pinned candidate")
    for name in SPLIT_HASHES:
        if sha256(inputs[name]) != SPLIT_HASHES[name]:
            raise ValueError("requires the complete original split bytes: " + name)
    initial = read_json(inputs["initial_runtime_identity"])
    if (initial.get("base_sha256") != digest(initial["base_manifest"])
            or initial.get("execution_sha256") != digest(initial["execution"])
            or len(initial["base_manifest"]) != 429 or len(initial["adapter_manifest"]) != 496
            or initial["adapter_parameters"] != 43278336):
        raise ValueError("invalid initial base/adapter reference identity")
    execution = spec["execution"]
    reference = dict(initial["execution"])
    # This is an explicitly supplied new execution identity, never relabeling
    # previous engineering readouts as collection observations.
    reference["runtime_source_sha256"] = spec["sources"]["training/qwen_lora_runtime.py"]
    reference["categorical_sampler"] = categorical_sampler_identity()
    if (execution != reference or execution["device"] != "cpu" or execution["cpu_threads"] != 2
            or execution["lora_rank"] != 16 or execution["lora_alpha"] != 32
            or len(execution["targets"]) != 248 or execution["attention_backend"] != "eager"
            or execution["checkpoint_decoder"] is not True or execution["effective_use_cache"] is not False
            or execution["model_mode"] != "eval" or execution["deterministic_algorithms"] is not True):
        raise ValueError("explicit execution must match the tested CPU treatment with the pinned current runtime source")
    weights = {"base": initial["base_manifest"], "adapters": initial["adapter_manifest"]}
    identity = {"model_id": MODEL_ID, "model_revision": REVISION, "weights_manifest_sha256": digest(weights),
                "tokenizer_id": MODEL_ID, "tokenizer_revision": REVISION,
                "tokenizer_files_sha256": digest(tokenizer["files"]),
                "chat_template_sha256": tokenizer["chat_template_sha256"],
                "protocol_sha256": digest(spec["protocol"]), "reasoning": "none"}
    return identity, weights


def catalog_rows(episodes_path, spec, spec_sha256):
    binding, episodes, texts = episode_index(episodes_path, "train")
    if len(episodes) != 1619 or len(texts) != 2926 or binding["sha256"] != SPLIT_HASHES["train"]:
        raise ValueError("complete original training denominator must remain 1619 episodes / 2926 propositions")
    rows = []
    for index, (ident, text) in enumerate(sorted(texts.items())):
        seed_digest = digest({"master_seed": spec["master_seed"], "collection_id": spec["collection_id"],
                              "spec_sha256": spec_sha256, "proposition_id": ident})
        messages = [["system", spec["protocol"]["system_instruction"]],
                    ["user", spec["protocol"]["verbal_prompt"].format(proposition=text)]]
        rows.append({"index": index, "id": ident, "text": text, "messages": messages,
                     "messages_sha256": digest(messages), "seed": int(seed_digest[:16], 16),
                     "seed_derivation_sha256": seed_digest})
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("per-proposition uint64 seed collision; no inferred reseeding")
    return rows


def create_plan(spec_path, expected_spec_sha256, directory):
    """Copy and bind all inputs and deterministic catalog before any inference."""
    raw_spec = read_file(spec_path)
    if sha256(raw_spec) != expected_spec_sha256:
        raise ValueError("external spec SHA256 differs")
    spec = read_json(raw_spec)
    inputs = {key: pinned_input(spec, key) for key in
              ("initial_runtime_identity", "checkpoint_manifest", "tokenizer_manifest", "train", "valid", "heldout")}
    identity, weights = validate_spec(spec, inputs)
    rows = catalog_rows(spec["train"]["path"], spec, expected_spec_sha256)
    directory = Path(directory)
    make_directory(directory)
    make_directory(directory / "inputs")
    make_directory(directory / "attempts")
    make_directory(directory / "invocations")
    files = {"spec.json": raw_spec, "initial-weights-manifest.json": canonical_bytes(weights) + b"\n",
             "catalog.jsonl": b"".join(canonical_bytes(row) + b"\n" for row in rows)}
    for key, raw in inputs.items():
        files["inputs/" + key + (".jsonl" if key in SPLIT_HASHES else ".json")] = raw
    for name, raw in files.items():
        create_file(directory / name, raw)
    create_file(directory / "owner.lock", b"")
    initial = read_json(inputs["initial_runtime_identity"])
    plan = {"schema_version": 1, "collection_id": spec["collection_id"], "spec_sha256": expected_spec_sha256,
            "bound_files": {name: sha256(raw) for name, raw in files.items()},
            "training_identity": identity, "training_identity_sha256": digest(identity),
            "initial_base_sha256": initial["base_sha256"],
            "initial_adapter_manifest_sha256": digest(initial["adapter_manifest"]),
            "execution_sha256": digest(spec["execution"]),
            "initial_policy_sha256": digest({"base_sha256": initial["base_sha256"],
                "execution_sha256": digest(spec["execution"]), "adapters": initial["adapter_manifest"]}),
            "sampling_distribution": SAMPLING, "sampling_distribution_sha256": digest(SAMPLING),
            "target_episodes": 1619, "target_propositions": 2926, "eos_token_id": 248046,
            "created_at": now(), "candidate_weights_loaded": False, "inference_performed": False,
            "scientific_model_selected": False, "training_performed": False,
            "interrupted_attempt_policy": "permanently-consumed-never-retry",
            "ordering": "ascending-full-proposition-utf8-sha256",
            "seed_derivation": "uint64-big-endian-first-8-bytes-of-canonical-master-seed-collection-spec-proposition-sha256"}
    create_json(directory / "plan.json", plan)
    return sha256(read_file(directory / "plan.json"))


def load_plan(directory, expected_plan_sha256):
    directory = Path(directory)
    raw = read_file(directory / "plan.json")
    if sha256(raw) != expected_plan_sha256:
        raise ValueError("external immutable plan SHA256 differs")
    plan = read_json(raw)
    if plan.get("schema_version") != 1 or (plan.get("target_episodes"), plan.get("target_propositions")) != (1619, 2926):
        raise ValueError("invalid complete-original-split collection plan")
    for name, expected in plan["bound_files"].items():
        if name.startswith("/") or ".." in Path(name).parts or sha256(read_file(directory / name)) != expected:
            raise ValueError("bound plan input differs: " + name)
    spec = read_json(read_file(directory / "spec.json"))
    inputs = {key: read_file(directory / "inputs" / (key + (".jsonl" if key in SPLIT_HASHES else ".json")))
              for key in ("initial_runtime_identity", "checkpoint_manifest", "tokenizer_manifest", "train", "valid", "heldout")}
    identity, weights = validate_spec(spec, inputs)
    if identity != plan["training_identity"] or digest(identity) != plan["training_identity_sha256"]:
        raise ValueError("derived immutable training identity differs")
    rows = catalog_rows(directory / "inputs/train.jsonl", spec, plan["spec_sha256"])
    if read_file(directory / "catalog.jsonl") != b"".join(canonical_bytes(row) + b"\n" for row in rows):
        raise ValueError("complete deterministic catalog differs")
    return plan, spec, rows


class AttemptLedger:
    """Caller holds owner_lock. Directory existence, not a mutable status, consumes an ID."""
    def __init__(self, directory, rows, *, binding=None):
        self.directory = Path(directory)
        self.rows = rows
        self.by_id = {row["id"]: row for row in rows}
        self.binding = binding
        actual = {path.name for path in (self.directory / "attempts").iterdir()}
        if actual - set(self.by_id):
            raise ValueError("unknown attempt directories; refuse uncertain ledger state")
        if any(path.is_symlink() or not path.is_dir() for path in (self.directory / "attempts").iterdir()):
            raise ValueError("attempt marker is not a direct directory")

    def unattempted(self):
        return [row for row in self.rows if not (self.directory / "attempts" / row["id"]).exists()]

    def reserve(self, row, attempt):
        path = self.directory / "attempts" / row["id"]
        make_directory(path)  # Fsync consumption before any sampling or callback.
        create_json(path / "attempt.json", attempt)
        with (path / "tokens.jsonl").open("xb") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(path)
        return path

    def inspect(self):
        results = {}
        for row in self.rows:
            path = self.directory / "attempts" / row["id"]
            if not path.exists():
                results[row["id"]] = "unattempted"
                continue
            if not (path / "receipt.json").exists():
                results[row["id"]] = "interrupted-or-incomplete"
                continue
            try:
                receipt = read_json(read_file(path / "receipt.json"))
                attempt_raw = read_file(path / "attempt.json")
                tokens_raw = read_file(path / "tokens.jsonl")
                attempt = read_json(attempt_raw)
                if (receipt["id"] != row["id"] or attempt["id"] != row["id"]
                        or receipt["attempt_sha256"] != sha256(attempt_raw)
                        or receipt["token_events_sha256"] != sha256(tokens_raw)):
                    raise ValueError("attempt/receipt/token content pin differs")
                if tokens_raw and not tokens_raw.endswith(b"\n"):
                    raise ValueError("partial raw token-event line retained")
                events = [read_json(line) for line in tokens_raw.splitlines()]
                if receipt["runtime_integrity_verified"]:
                    if (len(events) != len(receipt["completion_token_ids"])
                            or any((event["token_index"], event["token_id"], event["sampled_logprob"])
                                != (i, receipt["completion_token_ids"][i], receipt["token_logprobs"][i])
                                for i, event in enumerate(events))):
                        raise ValueError("verified receipt differs from per-token evidence")
                if self.binding is not None:
                    validate_receipt_binding(row, attempt, receipt, *self.binding)
                results[row["id"]] = ("probability" if receipt_valid(receipt) else
                                       "invalid-probability" if receipt["readout_status"] == "probability" else
                                       receipt["readout_status"])
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                results[row["id"]] = "corrupt-retained-attempt"
        return {"target_propositions": len(self.rows), "attempted": sum(s != "unattempted" for s in results.values()),
                "valid_probabilities": sum(s == "probability" for s in results.values()),
                "unattempted": sum(s == "unattempted" for s in results.values()),
                "states": results, "complete_valid_coverage": all(s == "probability" for s in results.values())}


def receipt_valid(receipt):
    response = receipt.get("response")
    value = receipt.get("probability")
    tokens = receipt.get("completion_token_ids", [])
    logprobs = receipt.get("token_logprobs", [])
    return (receipt.get("readout_status") == "probability" and receipt.get("runtime_integrity_verified") is True
            and receipt.get("completed") is True and receipt.get("error_kind") is None
            and receipt.get("exit_code") == 0 and receipt.get("tools_used") is False
            and receipt.get("finish_reason") == "eos" and bool(tokens) and tokens[-1] == 248046
            and 248046 not in tokens[:-1] and len(tokens) == len(logprobs)
            and all(type(token) is int and 0 <= token < 248320 for token in tokens)
            and all(type(lp) in (float, int) and math.isfinite(lp) and lp <= 0 for lp in logprobs)
            and type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1
            and isinstance(response, str) and re.fullmatch(r"(?:0(?:\.[0-9]+)?|1(?:\.0+)?)", response.strip()) is not None
            and float(response.strip()) == value)


def validate_receipt_binding(row, attempt, receipt, plan, spec, plan_sha256):
    expected = {"id": row["id"], "collection_plan_sha256": plan_sha256,
                "training_identity_sha256": plan["training_identity_sha256"],
                "policy_sha256": plan["initial_policy_sha256"], "seed": row["seed"],
                "sampling_distribution_sha256": plan["sampling_distribution_sha256"]}
    for value in (attempt, receipt):
        if any(value.get(key) != required for key, required in expected.items()):
            raise ValueError("attempt/receipt immutable plan, seed, sampler or initial identity differs")
    if (attempt.get("messages") != row["messages"] or attempt.get("messages_sha256") != row["messages_sha256"]
            or receipt.get("prompt_token_ids") != attempt.get("prompt_token_ids")
            or receipt.get("invocation_id") != attempt.get("invocation_id")
            or receipt.get("requested_model") != MODEL_ID or receipt.get("reasoning") != "none"
            or receipt.get("campaign") != spec["collection_id"] or receipt.get("protocol_id") != spec["protocol_id"]
            or receipt.get("prompt_sha256") != proposition_id(row["messages"][1][1])
            or receipt.get("system_instruction_sha256") != proposition_id(row["messages"][0][1])
            or receipt.get("is_current") is not True
            or len(receipt.get("completion_token_ids", [])) > spec["max_new_tokens"]):
        raise ValueError("receipt prompt/treatment/invocation binding differs")


def usage(started):
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {"wall_seconds": time.monotonic() - started, "cpu_seconds": value.ru_utime + value.ru_stime,
            "peak_rss_bytes_linux": value.ru_maxrss * 1024}


def append_event(path, row):
    raw = canonical_bytes(row) + b"\n"
    with Path(path).open("ab", buffering=0) as stream:
        if stream.write(raw) != len(raw):
            raise OSError("short receipt write; attempt stays consumed")
        os.fsync(stream.fileno())


def run_tranche(ledger, plan, spec, plan_sha256, invocation_id, *, runtime, tokenizer,
                generator_factory, rng_digest, max_attempts, token_budget, deadline,
                verify_inputs, phase):
    """Durable orchestration shared with injected tiny fixtures; never implements sampling."""
    from harness.run_native_checks import classify_readout, parse_probability

    completed, used = 0, 0
    for row in ledger.unattempted():
        if completed >= max_attempts or token_budget - used < spec["max_new_tokens"] or time.monotonic() >= deadline:
            break
        verify_inputs()
        tokenizer.verify_runtime()
        if runtime.policy_sha256() != plan["initial_policy_sha256"]:
            raise ValueError("initial policy changed before attempt; stop without reserving another proposition")
        rendered, prompt_ids = tokenizer.render_and_encode(row["messages"])
        generator = generator_factory(row["seed"])
        attempt = {"id": row["id"], "catalog_index": row["index"], "invocation_id": invocation_id,
                   "collection_plan_sha256": plan_sha256, "training_identity_sha256": plan["training_identity_sha256"],
                   "policy_sha256": plan["initial_policy_sha256"], "sampling_distribution_sha256": plan["sampling_distribution_sha256"],
                   "seed": row["seed"], "rng_before_sha256": rng_digest(generator),
                   "messages": row["messages"], "messages_sha256": row["messages_sha256"],
                   "rendered_prompt_sha256": proposition_id(rendered), "prompt_token_ids": list(prompt_ids),
                   "max_new_tokens": spec["max_new_tokens"], "recorded_at": now(),
                   "eligibility": "no prior attempt directory; fixed catalog order; full cap fits token budget",
                   "remaining_token_budget_at_reservation": token_budget - used,
                   "remaining_wall_seconds_at_reservation": max(0.0, deadline - time.monotonic())}
        # Encoding/input checks may consume the remaining wall budget. Admission
        # is rechecked here so expired invocations reserve no further attempts.
        if time.monotonic() >= deadline:
            break
        path = ledger.reserve(row, attempt)
        phase("attempt_reserved", id=row["id"], index=row["index"], seed=row["seed"])
        role_started = time.monotonic()
        last_token = role_started
        events = []

        def on_token(event):
            nonlocal last_token
            stamp = time.monotonic()
            if (set(event) != {"token_index", "token_id", "sampled_logprob", "is_eos"}
                    or event["token_index"] != len(events)):
                raise ValueError("runtime token callback schema/order differs")
            retained = {**event, "id": row["id"], "invocation_id": invocation_id,
                        "rng_after_sha256": rng_digest(generator), "recorded_at": now(),
                        "token_wall_seconds": stamp - last_token, "attempt_wall_seconds": stamp - role_started}
            events.append(retained)
            append_event(path / "tokens.jsonl", retained)
            last_token = stamp
            if stamp >= deadline:
                raise TimeoutError("invocation deadline reached; sampled token retained; attempt is never retried")

        sampled = None
        runtime_error = decode_error = None
        integrity = False
        response = ""
        try:
            sampled = runtime.generate_tokens(prompt_ids, eos_token_id=248046,
                                               max_new_tokens=spec["max_new_tokens"], generator=generator,
                                               on_token=on_token)
            if runtime.policy_sha256() != plan["initial_policy_sha256"]:
                raise ValueError("adapter policy changed during generation")
            integrity = True  # Public generate_tokens already fully audited the base.
        except BaseException as error:
            sampled = getattr(error, "partial", None)
            runtime_error = {"type": type(error).__name__, "message": str(error)}
        tokens = list(sampled.completion_token_ids) if sampled is not None else [e["token_id"] for e in events]
        logprobs = list(sampled.token_logprobs) if sampled is not None else [e["sampled_logprob"] for e in events]
        finish = sampled.finish_reason if sampled is not None else "error"
        try:
            response = tokenizer.decode_completion(tokens)
        except Exception as error:
            decode_error = {"type": type(error).__name__, "message": str(error)}
        status = ("runtime-error" if runtime_error else "decode-error" if decode_error else
                  "truncated" if finish != "eos" else classify_readout(response))
        probability = parse_probability(response) if status == "probability" else None
        receipt = {key: attempt[key] for key in ("id", "invocation_id", "collection_plan_sha256", "training_identity_sha256",
                   "policy_sha256", "sampling_distribution_sha256", "seed", "prompt_token_ids")}
        receipt.update({"is_current": True, "requested_model": MODEL_ID, "reasoning": "none",
                        "protocol_id": spec["protocol_id"], "campaign": spec["collection_id"],
                        "completed": finish == "eos" and runtime_error is None and decode_error is None,
                        "exit_code": 1 if runtime_error or decode_error else 0, "tools_used": False,
                        "error_kind": None if status == "probability" else status,
                        "runtime_error": runtime_error, "decode_error": decode_error,
                        "runtime_integrity_verified": integrity, "readout_status": status,
                        "probability": probability, "response": response, "finish_reason": finish,
                        "completion_token_ids": tokens, "token_logprobs": logprobs,
                        "prompt_sha256": proposition_id(row["messages"][1][1]),
                        "system_instruction_sha256": proposition_id(row["messages"][0][1]),
                        "rng_before_sha256": attempt["rng_before_sha256"], "rng_after_sha256": rng_digest(generator),
                        "finished_at": now(), "wall_seconds": time.monotonic() - role_started,
                        "attempt_sha256": sha256(read_file(path / "attempt.json")),
                        "token_events_sha256": sha256(read_file(path / "tokens.jsonl")), "resampled": False})
        create_json(path / "receipt.json", receipt)
        completed += 1
        used += len(tokens)
        phase("attempt_retained", id=row["id"], status=status, tokens=len(tokens), attempt_wall_seconds=receipt["wall_seconds"])
        if runtime_error:
            raise RuntimeError("runtime attempt failed; all partial evidence retained; no automatic retry: " + str(runtime_error))
    return {"attempts_this_invocation": completed, "tokens_this_invocation": used,
            "remaining_token_budget": token_budget - used, "fixed_per_proposition_cap": spec["max_new_tokens"]}


def validate_initial_runtime_identity(actual, plan, spec, *, policy_sha256, sampling_distribution, zero_initial_b):
    """Compare the JSON identity contract, including live integer-key configs.

    Transformers uses integer id2label keys in memory; JSON object keys are
    strings. Canonical bytes are the same authoritative representation used by
    all pinned digests. This does not relax configuration or hash comparisons.
    """
    checks = {
        "base_sha256": actual["base_sha256"] == plan["initial_base_sha256"],
        "adapter_manifest_sha256": digest(actual["adapter_manifest"]) == plan["initial_adapter_manifest_sha256"],
        "execution_canonical_content": canonical_bytes(actual["execution"]) == canonical_bytes(spec["execution"]),
        "execution_sha256": actual["execution_sha256"] == plan["execution_sha256"],
        "initial_weights_manifest_sha256": digest({"base": actual["base_manifest"], "adapters": actual["adapter_manifest"]})
            == plan["training_identity"]["weights_manifest_sha256"],
        "initial_policy_sha256": policy_sha256 == plan["initial_policy_sha256"],
        "sampling_distribution": sampling_distribution == SAMPLING,
        "adapter_parameter_count": actual["adapter_parameters"] == 43278336,
        "zero_initial_B": zero_initial_b is True,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("loaded initial identity checks failed: " + ", ".join(failed))


def initialize_runtime(directory, checkpoint, tokenizer_artifacts, plan, spec, invocation, phase):
    """The only pretrained weight-loading path; called only by explicit collect."""
    import torch
    from training.qwen_checkpoint import load_qwen_text_checkpoint
    from training.qwen_lora_runtime import QwenLoraRuntime, tensor_byte_sha256
    from training.qwen_tokenizer import QwenTextTokenizer

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision(spec["execution"]["float32_matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = spec["execution"]["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = spec["execution"]["cudnn_allow_tf32"]
    tokenizer = QwenTextTokenizer(tokenizer_artifacts)
    token_identity = tokenizer.verify_runtime()
    if (token_identity["tokenizer_files_sha256"] != plan["training_identity"]["tokenizer_files_sha256"]
            or token_identity["chat_template_sha256"] != plan["training_identity"]["chat_template_sha256"]):
        raise ValueError("actual tokenizer differs from immutable training identity")
    create_json(invocation / "tokenizer-identity.json", token_identity)
    phase("weight_load_start")
    model, loaded = load_qwen_text_checkpoint(checkpoint, directory / "inputs/checkpoint_manifest.json", attn_implementation="eager")
    create_json(invocation / "loaded-checkpoint.json", loaded)
    phase("weight_load_complete")
    runtime = QwenLoraRuntime(model, lora_seed=spec["execution"]["lora_seed"], expected_target_count=248, checkpoint_decoder=True)
    actual = runtime.checkpoint_identity()
    create_json(invocation / "initial-runtime-identity.json", actual)
    validate_initial_runtime_identity(actual, plan, spec, policy_sha256=runtime.policy_sha256(),
        sampling_distribution=runtime.sampling_distribution(),
        zero_initial_b=not any(torch.count_nonzero(value).item() for name, value in runtime.adapter_parameters.items()
                              if name.endswith(".lora_B")))
    phase("initial_identity_verified", initial_policy_sha256=plan["initial_policy_sha256"])
    return runtime, tokenizer, lambda seed: torch.Generator(device="cpu").manual_seed(seed), \
        lambda generator: tensor_byte_sha256(generator.get_state())


def collect(directory, plan_sha256, *, checkpoint, tokenizer_artifacts, max_attempts, token_budget,
            wall_seconds, cpu_soft_seconds, cpu_hard_seconds, address_space_bytes):
    directory = Path(directory)
    limits = {"max_attempts": max_attempts, "token_budget": token_budget, "wall_seconds": wall_seconds,
              "cpu_soft_seconds": cpu_soft_seconds, "cpu_hard_seconds": cpu_hard_seconds,
              "address_space_bytes": address_space_bytes}
    if (any(type(value) is not int or value < 1 for value in limits.values())
            or cpu_hard_seconds <= cpu_soft_seconds or address_space_bytes > 36 * 2**30):
        raise ValueError("explicit positive bounded invocation limits required; AS cannot exceed 36GiB")
    with owner_lock(directory):
        plan, spec, rows = load_plan(directory, plan_sha256)
        ledger = AttemptLedger(directory, rows, binding=(plan, spec, plan_sha256))
        if token_budget < spec["max_new_tokens"] or not ledger.unattempted():
            return {"status": "no-eligible-new-attempt", "weights_loaded": False, **ledger.inspect()}
        invocation_id = str(uuid.uuid4())
        invocation = directory / "invocations" / invocation_id
        make_directory(invocation)
        started = time.monotonic()
        deadline = started + wall_seconds
        create_json(invocation / "plan.json", {"invocation_id": invocation_id, "collection_plan_sha256": plan_sha256,
                    "limits": limits, "started_at": now(), "host": socket.gethostname(), "pid": os.getpid(),
                    "eligible_ids_before_load": [row["id"] for row in ledger.unattempted()],
                    "requires_external_timeout_and_kill_supervisor": True})

        def phase(name, **details):
            event = {"phase": name, "recorded_at": now(), **usage(started), **details}
            append_event(invocation / "phases.jsonl", event)
            print(json.dumps(event, allow_nan=False), flush=True)

        def expired(signum, frame):
            raise TimeoutError("collection invocation received bounded-stop signal " + str(signum))

        runtime = tokenizer = None
        result = {"status": "failed", "invocation_id": invocation_id, "training_performed": False,
                  "provider_calls": 0, "resampled": False}
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGXCPU)}
        try:
            resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft_seconds, cpu_hard_seconds))
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ[name] = "1"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            for sig in old_handlers:
                signal.signal(sig, expired)
            signal.alarm(wall_seconds)
            runtime, tokenizer, generator_factory, rng_digest = initialize_runtime(
                directory, checkpoint, tokenizer_artifacts, plan, spec, invocation, phase)
            result.update(run_tranche(ledger, plan, spec, plan_sha256, invocation_id,
                          runtime=runtime, tokenizer=tokenizer, generator_factory=generator_factory, rng_digest=rng_digest,
                          max_attempts=max_attempts, token_budget=token_budget, deadline=deadline,
                          verify_inputs=lambda: load_plan(directory, plan_sha256), phase=phase))
            phase("invocation_integrity_audit_start")
            if (runtime.verify_frozen_base() != plan["initial_base_sha256"]
                    or runtime.policy_sha256() != plan["initial_policy_sha256"]):
                raise ValueError("initial base or adapter policy changed by end of invocation")
            tokenizer.verify_runtime()
            load_plan(directory, plan_sha256)
            result.update({"status": "bounded-invocation-complete", "base_and_adapters_unchanged": True})
        except BaseException as error:
            result.update({"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        finally:
            signal.alarm(0)
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
            result.update(usage(started))
            result["coverage"] = ledger.inspect()
            create_json(invocation / "report.json", result)
        return result


def export_anchor(directory, plan_sha256, output, *, mode, smoothing_epsilon=None):
    """Separate explicit treatment derivation; never exports a partial snapshot."""
    if mode not in {"exact", "symmetric-mixture"} or (mode == "exact" and smoothing_epsilon is not None):
        raise ValueError("choose explicit exact or symmetric-mixture treatment; exact takes no epsilon")
    if mode == "symmetric-mixture" and (type(smoothing_epsilon) not in (float, int)
            or not math.isfinite(smoothing_epsilon) or not 0 < smoothing_epsilon < .5 or 1 - smoothing_epsilon == 1):
        raise ValueError("explicit representable epsilon in (0,.5) required")
    directory, output = Path(directory), Path(output)
    with owner_lock(directory):
        plan, spec, rows = load_plan(directory, plan_sha256)
        ledger = AttemptLedger(directory, rows, binding=(plan, spec, plan_sha256))
        coverage = ledger.inspect()
        if not coverage["complete_valid_coverage"] or coverage["valid_probabilities"] != 2926:
            raise ValueError("complete anchor unavailable: " + str(coverage["valid_probabilities"]) + "/2926 valid; attempts never replaced")
        make_directory(output)
        receipts = b"".join(read_file(directory / "attempts" / row["id"] / "receipt.json") for row in rows)
        create_file(output / "anchor-receipts.jsonl", receipts)
        create_file(output / "anchor-propositions.jsonl", read_file(directory / "catalog.jsonl"))
        create_file(output / "initial-weights-manifest.json", read_file(directory / "initial-weights-manifest.json"))
        create_file(output / "collection-plan.json", read_file(directory / "plan.json"))
        create_file(output / "collection-spec.json", read_file(directory / "spec.json"))
        run = {"run_id": spec["collection_id"], "campaign": spec["collection_id"], "requested_model": MODEL_ID,
               "reasoning": "none", "protocol_id": spec["protocol_id"], "receipts_file": "anchor-receipts.jsonl",
               "receipts_sha256": sha256(receipts), "recorded_at": now(), "timestamp_source": "local-pinned-qwen-collection",
               "training_identity": plan["training_identity"], "collection_plan_sha256": plan_sha256,
               "initial_policy_sha256": plan["initial_policy_sha256"], "execution_sha256": plan["execution_sha256"],
               "sampling_distribution_sha256": plan["sampling_distribution_sha256"],
               "weights_manifest_file": "initial-weights-manifest.json",
               "weights_manifest_file_sha256": sha256(read_file(output / "initial-weights-manifest.json"))}
        manifest = {"runs": [run], "native_protocols": {spec["protocol_id"]: spec["protocol"]},
                    "propositions_file": "anchor-propositions.jsonl",
                    "propositions_sha256": sha256(read_file(output / "anchor-propositions.jsonl"))}
        create_json(output / "anchor-manifest.json", manifest)
        snapshot = freeze_from_manifest(output / "anchor-manifest.json", spec["collection_id"],
                                        directory / "inputs/train.jsonl", "train", smoothing_epsilon=smoothing_epsilon)
        write_snapshot(output / "anchor.json", snapshot)
        create_json(output / "training-identity.json", plan["training_identity"])
        create_json(output / "export-report.json", {"status": snapshot["payload"]["status"],
                    "snapshot_sha256": snapshot["snapshot_sha256"], "collection_plan_sha256": plan_sha256,
                    "treatment": snapshot["payload"]["treatment"], "coverage": snapshot["payload"]["coverage"],
                    "lambda_selected": False, "experiment_binding_created": False, "training_performed": False})
        return snapshot
