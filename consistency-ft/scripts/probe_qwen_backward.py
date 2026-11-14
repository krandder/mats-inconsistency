#!/usr/bin/env python3
"""One retained P-role numerical sum-logprob derivative; no optimizer or training.

Requires an externally hashed, successful complete candidate-generation receipt.
Run under ``timeout --kill-after=10s 900s`` with external stdout/stderr/time capture.
This script cannot enforce that external supervisor or survive SIGKILL/OOM.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EPISODE_ID = "2028/AndChecker:11"
LORA_SEED = 2026091304
LOADER_SHA256 = "974a203e90e189afcf1f3328c6f5128fe2c1ea68a3c518792a49cefd75aa84ba"
INPUT_FILES = ("plan.json", "loaded-checkpoint.json", "readouts.jsonl", "report.json", "token-events.jsonl")
ATOL = RTOL = 2e-6


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)


def file_bytes(path, limit=32 * 1024 * 1024):
    path = Path(path)
    before = path.stat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("input must be a bounded regular file: " + str(path))
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(raw) > limit or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError("input changed while reading: " + str(path))
    return raw


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_inputs(directory, digests_path, expected_digest):
    """Pure receipt validation: never imports Torch or reads candidate weights."""
    digest_raw = file_bytes(digests_path)
    if sha256(digest_raw) != expected_digest:
        raise ValueError("externally recorded input-digests SHA256 differs")
    pins = strict_json(digest_raw)
    if not isinstance(pins, dict) or set(pins) != set(INPUT_FILES):
        raise ValueError("input-digests must pin all five complete generation artifacts")
    raw_files = {name: file_bytes(Path(directory) / name) for name in INPUT_FILES}
    for name, raw in raw_files.items():
        if sha256(raw) != pins[name]:
            raise ValueError("generation input digest mismatch: " + name)
    plan, loaded, report = (strict_json(raw_files[name]) for name in
                            ("plan.json", "loaded-checkpoint.json", "report.json"))
    raw_lines = raw_files["readouts.jsonl"].splitlines(keepends=True)
    if len(raw_lines) != 3 or any(not line.endswith(b"\n") or not line.strip() for line in raw_lines):
        raise ValueError("require all three completed original readout rows")
    records = [strict_json(line) for line in raw_lines]
    if (report.get("status") != "candidate-generation-probe-complete"
            or any(report.get(key) is not True for key in
                   ("candidate_weights_loaded", "candidate_forward_performed", "base_weights_unchanged"))
            or any(report.get(key) is not False for key in
                   ("candidate_backward_performed", "training_performed", "complete_training_anchor"))
            or report.get("attempted_role_readouts") != 3
            or report.get("generated_tokens") != sum(len(row["completion_token_ids"]) for row in records)
            or report.get("valid_probability_readouts") != sum(row["readout_status"] == "probability" for row in records)
            or loaded.get("candidate_weights_loaded") is not True
            or loaded.get("loader_source_sha256") != LOADER_SHA256):
        raise ValueError("whole base probe did not finish with verified unchanged pretrained weights")
    if (plan.get("episode_id") != EPISODE_ID or plan.get("role_seeds") != [2026091301, 2026091302, 2026091303]
            or plan.get("selected_before_generation") is not True
            or plan.get("model_mode") != "eval" or plan.get("use_cache") is not False
            or plan.get("threads") != 2 or plan.get("softmax_dtype") != "float32"
            or (plan.get("temperature"), plan.get("top_p"), plan.get("top_k")) != (1.0, 1.0, 0)
            or plan.get("logits_processors") != [] or plan.get("eos_token_id") != 248046):
        raise ValueError("original generation plan differs from the predeclared treatment")
    train_raw = file_bytes(ROOT / "data/episodes/train.jsonl")
    episode, = [strict_json(line) for line in train_raw.splitlines()
                if line.strip() and strict_json(line)["id"] == EPISODE_ID]
    if (sha256(train_raw) != plan["train_file_sha256"] or sha256(canonical(episode)) != plan["episode_sha256"]
            or [row["role"] for row in records] != [row["role"] for row in episode["prompts"]]):
        raise ValueError("original corpus, episode or role order changed")
    row = records[0]  # Never search for a later valid answer or choose by content.
    prompt = episode["prompts"][0]
    protocol = plan["protocol"]
    messages = [["system", protocol["system_instruction"]],
                ["user", protocol["verbal_prompt"].format(proposition=prompt["text"])]]
    if (row["role"] != "P" or prompt["role"] != "P" or row.get("seed") != 2026091301
            or row.get("finish_reason") != "eos" or row.get("readout_status") != "probability"
            or row.get("error") is not None or row.get("messages") != messages
            or row.get("messages_sha256") != sha256(canonical(messages))
            or row.get("proposition_sha256") != sha256(prompt["text"].encode())
            or plan.get("protocol_sha256") != sha256(canonical(protocol))):
        raise ValueError("first predeclared P row is not a valid unchanged original EOS readout")
    p, c, b = row["prompt_token_ids"], row["completion_token_ids"], row["token_logprobs"]
    if (not p or not c or len(c) > plan["max_new_tokens_per_role"] or len(c) != len(b)
            or any(type(token) is not int or not 0 <= token < 248320 for token in p + c)
            or c[-1] != 248046 or 248046 in c[:-1]
            or any(type(value) not in (int, float) or not math.isfinite(value) or value > 0 for value in b)):
        raise ValueError("malformed original prompt/completion/behavior logprobs")
    events = [strict_json(line) for line in raw_files["token-events.jsonl"].splitlines() if line.strip()]
    selected_events = [event for event in events if event["role"] == "P"]
    if (len(selected_events) != len(c) or any((event["index"], event["token_id"], event["logprob"])
            != (index, c[index], b[index]) for index, event in enumerate(selected_events))):
        raise ValueError("P readout differs from retained original per-token events")
    return {"pins": pins, "digest_raw": digest_raw, "raw_files": raw_files,
            "first_line": raw_lines[0], "row": row, "plan": plan, "loaded": loaded,
            "report": report, "episode": episode}


def resources(started):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"wall_seconds": time.monotonic() - started,
            "cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_rss_bytes_linux": usage.ru_maxrss * 1024}


def captured_logprobs(output, targets, behavior):
    """Detach only selected values from the one role_loss model forward."""
    import torch

    with torch.no_grad():
        logits = output.logits.detach()
        if tuple(logits.shape) != (1, len(targets), 248320):
            raise ValueError("unexpected selected-position diagnostic logits shape")
        actual = logits[0].float().log_softmax(-1).gather(
            -1, torch.tensor(targets, device=logits.device)[:, None]).squeeze(-1)
        observed = torch.tensor(behavior, dtype=torch.float32, device=logits.device)
        finite = bool(torch.isfinite(actual).all())
        return {"token_ids": list(targets), "behavior_logprobs": list(behavior),
                "evaluated_logprobs": [float(value) if math.isfinite(float(value)) else str(float(value))
                                       for value in actual.cpu().tolist()],
                "finite": finite, "atol": ATOL, "rtol": RTOL,
                "within_original_tolerance": bool(torch.allclose(actual, observed, atol=ATOL, rtol=RTOL)),
                "max_absolute_error": float((actual - observed).abs().max()) if finite else None,
                "source": "detached selected logits from the sole role_loss model forward"}


def gradient_summary(parameters):
    """Bounded reductions only; no gradient clones, hashes or serialized values."""
    import torch

    rows = {}
    for name, parameter in parameters.items():
        grad = parameter.grad
        if grad is None:
            rows[name] = {"present": False, "finite": False}
            continue
        finite, nonzero, squared_l2, max_abs = True, 0, 0.0, 0.0
        for chunk in grad.detach().reshape(-1).split(1024 * 1024):
            if not bool(torch.isfinite(chunk).all()):
                finite = False
                continue
            nonzero += int(torch.count_nonzero(chunk))
            squared_l2 += float(chunk.double().square().sum())
            max_abs = max(max_abs, float(chunk.abs().max()))
        rows[name] = {"present": True, "finite": finite, "shape": list(grad.shape),
                      "dtype": str(grad.dtype), "nonzero_elements": nonzero,
                      "l2_norm": math.sqrt(squared_l2) if finite else None,
                      "max_abs": max_abs if finite else None}
    return {"parameters": rows, "all_present_and_finite": all(
                row["present"] and row["finite"] for row in rows.values()),
            "nonzero_A_tensors": sum(row.get("nonzero_elements", 0) > 0 for name, row in rows.items()
                                     if name.endswith(".lora_A")),
            "nonzero_B_tensors": sum(row.get("nonzero_elements", 0) > 0 for name, row in rows.items()
                                     if name.endswith(".lora_B")),
            "zero_A_gradients_allowed": True, "gradient_values_saved": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "manifest", "tokenizer-artifacts", "base-probe", "input-digests", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--input-digests-sha256", required=True, help="Captured before this process")
    parser.add_argument("--runtime-source-sha256", required=True, help="Reviewed settled runtime source")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    state = {"status": "started", "candidate_weights_loaded": False, "candidate_forward_performed": False,
             "candidate_backward_started": False, "candidate_backward_performed": False,
             "optimizer_created": False, "optimizer_updates": 0, "training_performed": False,
             "whole_joint_experiment": False, "anchors_or_credits_constructed": False,
             "provider_inference_requests": 0, "resampled": False,
             "scope": "One retained original P readout: coefficient=1 derivative of sum generated-token logprobs."}
    events_path = args.output / "phases.jsonl"

    def phase(name, **details):
        state["last_phase"] = name
        event = {"phase": name, "recorded_at": datetime.now(timezone.utc).isoformat(),
                 **resources(started), **details}
        with events_path.open("ab", buffering=0) as stream:
            stream.write(canonical(event) + b"\n")
            os.fsync(stream.fileno())
        print(json.dumps(event, allow_nan=False), flush=True)

    def expired(signum, frame):
        raise TimeoutError("bounded derivative probe received signal " + str(signum))

    runtime = before = tokenizer = tokenizer_identity = inputs = source_hashes = None
    failure = None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (36 * 2**30, 36 * 2**30))
        resource.setrlimit(resource.RLIMIT_CPU, (900, 910))
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for sig in (signal.SIGALRM, signal.SIGXCPU, signal.SIGTERM):
            signal.signal(sig, expired)
        signal.alarm(900)
        phase("validate_inputs_start")
        inputs = load_inputs(args.base_probe, args.input_digests, args.input_digests_sha256)
        source_names = ("scripts/probe_qwen_backward.py", "training/qwen_checkpoint.py",
                        "training/qwen_lora_runtime.py", "training/qwen_tokenizer.py", "training/torch_loop.py",
                        "training/episode_adapter.py", "anchors.py", "harness/run_native_checks.py")
        source_hashes = {name: sha256(file_bytes(ROOT / name)) for name in source_names}
        if (source_hashes["training/qwen_checkpoint.py"] != LOADER_SHA256
                or source_hashes["training/qwen_lora_runtime.py"] != args.runtime_source_sha256):
            raise ValueError("reviewed loader/runtime source pin differs")
        for name, raw in (("input-digests.json", inputs["digest_raw"]), ("input-readout.jsonl", inputs["first_line"])):
            with (args.output / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        plan = {"episode_id": EPISODE_ID, "role": "P", "selection": "first predeclared role, never content-selected",
                "input_digests_sha256": args.input_digests_sha256,
                "input_file_sha256": inputs["pins"], "input_readout_line_sha256": sha256(inputs["first_line"]),
                "source_sha256": source_hashes, "lora_seed": LORA_SEED, "lora_rank": 16, "lora_alpha": 32,
                "initialization": "seeded FP32 A; zero FP32 B; zero effective adapter update",
                "expected_targets": 248, "expected_adapter_parameters": 43278336,
                "coefficient": 1.0, "loss": "sum generated-token logprobs including EOS",
                "atol": ATOL, "rtol": RTOL, "checkpoint_decoder": True,
                "threads": 2, "interop_threads": 1, "attention_implementation": "eager",
                "address_space_limit_bytes": 36 * 2**30, "cpu_seconds_soft_hard": [900, 910],
                "python_alarm_seconds": 900, "required_external_timeout_seconds": 900,
                "required_external_kill_after_seconds": 10,
                "paths": {key: str(value.resolve()) for key, value in vars(args).items() if isinstance(value, Path)}}
        write_json(args.output / "plan.json", plan)
        phase("validate_inputs_complete")
        import torch
        from harness.run_native_checks import PROTOCOLS, classify_readout, parse_probability
        from training.qwen_checkpoint import load_qwen_text_checkpoint
        from training.qwen_lora_runtime import QwenLoraRuntime, tensor_manifest
        from training.qwen_tokenizer import QwenTextTokenizer

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        tokenizer = QwenTextTokenizer(args.tokenizer_artifacts)
        tokenizer_identity = tokenizer.verify_runtime()
        row = inputs["row"]
        rendered, prompt_ids = tokenizer.render_and_encode(row["messages"])
        response = tokenizer.decode_completion(row["completion_token_ids"])
        if (tokenizer_identity != inputs["plan"]["tokenizer_identity"]
                or inputs["plan"]["protocol"] != PROTOCOLS["numeric-v1"]
                or list(prompt_ids) != row["prompt_token_ids"]
                or sha256(rendered.encode()) != row["rendered_prompt_sha256"]
                or response != row["response"] or classify_readout(response) != "probability"
                or parse_probability(response) != row["probability"]):
            raise ValueError("original tokenizer/protocol/prompt/completion identity differs")
        phase("load_start")
        model, load_report = load_qwen_text_checkpoint(args.checkpoint, args.manifest, attn_implementation="eager")
        state["candidate_weights_loaded"] = True
        write_json(args.output / "loaded-checkpoint.json", load_report)
        for key in ("model_id", "revision", "manifest_sha256", "loader_source_sha256", "checkpoint_files",
                    "text_tensor_manifest_sha256", "nonpersistent_buffers", "runtime"):
            if load_report[key] != inputs["loaded"][key]:
                raise ValueError("loaded base differs from original behavior base: " + key)
        phase("load_complete", text_parameters=load_report["text_parameters"])
        phase("adapter_install_start")
        runtime = QwenLoraRuntime(model, lora_seed=LORA_SEED, expected_target_count=248, checkpoint_decoder=True)
        before = runtime.checkpoint_identity()
        write_json(args.output / "identity-before.json", before)
        state.update({"base_sha256_before": before["base_sha256"],
                      "adapter_sha256_before": sha256(canonical(before["adapter_manifest"]))})
        if (len(runtime.target_shapes) != 248 or before["adapter_parameters"] != 43278336
                or len(runtime.adapter_parameters) != 496
                or any(not p.requires_grad or p.dtype != torch.float32 or p.grad is not None
                       for p in runtime.adapter_parameters.values())
                or any(torch.count_nonzero(p).item() for name, p in runtime.adapter_parameters.items()
                       if name.endswith(".lora_B"))):
            raise ValueError("initial rank16 all-target zero-update adapter contract differs")
        phase("adapter_install_complete", targets=248, adapter_parameters=43278336)
        completion = tuple(row["completion_token_ids"])
        numerical_row = (tuple(prompt_ids) + completion[:-1],
                         tuple(range(len(prompt_ids) - 1, len(prompt_ids) + len(completion) - 1)),
                         completion, 1.0)
        write_json(args.output / "numerical-input.json", {"prefix": numerical_row[0], "positions": numerical_row[1],
                   "targets": completion, "coefficient": 1.0, "behavior_logprobs": row["token_logprobs"]})
        captured = []

        def capture(module, arguments, output):
            state["candidate_forward_performed"] = True
            if captured:
                raise ValueError("unexpected second top-level forward during one role_loss")
            detail = captured_logprobs(output, completion, row["token_logprobs"])
            captured.append(detail)
            # Persist even when role_loss will immediately reject this comparison.
            write_json(args.output / "logprob-comparison.json", detail)
            phase("forward_values_retained", **detail)

        phase("role_loss_start")
        handle = model.register_forward_hook(capture)
        try:
            loss, error = runtime.role_loss(numerical_row, row["token_logprobs"])
        finally:
            handle.remove()
        if len(captured) != 1 or not captured[0]["within_original_tolerance"]:
            raise ValueError("sole numerical forward was not retained and within unchanged tolerance")
        state.update({"sum_logprob": float(loss.detach()), "max_behavior_logprob_error": error})
        phase("role_loss_complete", sum_logprob=state["sum_logprob"], max_behavior_logprob_error=error)
        phase("backward_start")
        state["candidate_backward_started"] = True
        loss.backward()
        state["candidate_backward_performed"] = True
        del loss
        phase("backward_complete")
        gradients = gradient_summary(runtime.adapter_parameters)
        write_json(args.output / "gradient-summary.json", gradients)
        if not gradients["all_present_and_finite"]:
            raise ValueError("missing or nonfinite adapter gradients; retained without update")
        state["adapter_gradients_present_and_finite"] = True
    except BaseException as error:
        failure = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
        write_json(args.output / "failure-initial.json", {**state, **failure, **resources(started), "status": "failed"})
        phase("failure_retained", **failure)
    finally:
        audit_errors = []
        if runtime is not None and before is not None:
            if resources(started)["wall_seconds"] < 840 and resources(started)["cpu_seconds"] < 840:
                phase("integrity_audit_start")
                try:
                    state["base_sha256_after"] = runtime.verify_frozen_base()
                    state["base_weights_unchanged"] = state["base_sha256_after"] == before["base_sha256"]
                    if not state["base_weights_unchanged"]:
                        audit_errors.append("base content digest differs after numerical probe")
                except Exception as error:
                    audit_errors.append("base audit: " + repr(error))
                try:
                    after = tensor_manifest(runtime.adapter_parameters)
                    write_json(args.output / "adapter-manifest-after.json", after)
                    state["adapter_sha256_after"] = sha256(canonical(after))
                    state["adapter_weights_unchanged"] = after == before["adapter_manifest"]
                    if not state["adapter_weights_unchanged"]:
                        audit_errors.append("adapter bytes changed without an optimizer update")
                except Exception as error:
                    audit_errors.append("adapter audit: " + repr(error))
                phase("integrity_audit_complete")
            else:
                audit_errors.append("insufficient remaining declared budget for full integrity audit")
        try:
            if inputs is not None:
                load_inputs(args.base_probe, args.input_digests, args.input_digests_sha256)
                state["inputs_unchanged"] = True
            if source_hashes is not None:
                if any(sha256(file_bytes(ROOT / name)) != expected for name, expected in source_hashes.items()):
                    raise ValueError("source changed during diagnostic")
                state["sources_unchanged"] = True
            if tokenizer is not None and tokenizer.verify_runtime() != tokenizer_identity:
                raise ValueError("tokenizer changed during diagnostic")
        except Exception as error:
            audit_errors.append("input/source/tokenizer audit: " + repr(error))
        if audit_errors:
            failure = failure or {"error_type": "IntegrityAuditFailure", "error": "; ".join(audit_errors)}
            failure["audit_errors"] = audit_errors
        state.update(resources(started))
        state["status"] = "failed" if failure else "candidate-numerical-backward-probe-complete"
        if failure:
            state.update(failure)
        write_json(args.output / ("failure.json" if failure else "report.json"), state)
        signal.alarm(0)
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
