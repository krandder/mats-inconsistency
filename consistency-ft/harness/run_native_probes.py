#!/usr/bin/env python3
"""Collect frozen factual probes through native subscription clients.

Preparation and planning are offline. Only collect --execute launches clients.
Forecast parsers, forecast receipts and reference answers are never used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import run_probes as probes
from harness import run_native_checks as native
from scripts.report_subscription_usage import tokens

COLLECTOR = "native-factual-v1"
SYSTEM_INSTRUCTION = ("You are participating in a factual-recall research benchmark. "
                      "Use only your existing knowledge. Do not use tools, browse, read files, "
                      "consult memory, or delegate. Follow your applicable constraints. "
                      "Return just the requested factual answer, a name or short phrase. "
                      "If you do not know or cannot answer, return exactly UNKNOWN. "
                      "Do not provide forecasts or probabilities.")
EFFORTS = {"astra": ("low",), "grok": ("low",), "glm": ("native-default",),
           "luna": ("none", "low", "medium", "high", "xhigh", "max")}
# Observed in the existing native archive and approved before this campaign;
# this is backend identity metadata, never an answer-grading alias.
REPORTED_MODEL_ALIASES = {"grok": ("grok-4.6", "grok-4.6-build")}
RECONNECT = ("Reconnecting... 1/5 (stream disconnected before completion: "
             "Upstream response was interrupted. Try again soon; the proxy will hold the next retry on the same account.)")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def treatment(provider, effort=None):
    if provider not in EFFORTS:
        raise ValueError("unsupported native provider")
    effort = effort or ("native-default" if provider == "glm" else "low")
    if effort not in EFFORTS[provider]:
        raise ValueError(f"unsupported reasoning treatment for {provider}: {effort}")
    return {"provider": provider, "model": native.MODELS[provider], "reasoning": effort}


def protocol_metadata(manifest):
    expected = treatment(manifest["provider"], manifest["reasoning"])
    if manifest["model"] != expected["model"] or manifest["protocol_id"] != probes.PROTOCOL:
        raise ValueError("manifest is not the exact native factual treatment")
    record = {"collector": COLLECTOR, "campaign_id": manifest["campaign_id"], **expected,
              "system_instruction": SYSTEM_INSTRUCTION,
              "system_instruction_sha256": probes.sha(SYSTEM_INSTRUCTION),
              "prompt_template_sha256": probes.sha(probes.PROBE_PROMPT),
              "reported_model_aliases": list(REPORTED_MODEL_ALIASES.get(manifest["provider"], (manifest["model"],))),
              "source_catalog_count": manifest["catalog_count"]}
    record["fingerprint"] = probes.sha(probes.canonical(record))
    return record


def prepare(directory, *, provider, effort=None, as_of, probes_path=None, cutoffs_path=None):
    chosen = treatment(provider, effort)
    manifest = probes.prepare(directory, probes_path or ROOT / "data/cutoff_probes.jsonl",
                              cutoffs_path or ROOT / "data/cutoffs.json", as_of=as_of, **chosen)
    metadata = protocol_metadata(manifest)
    directory = Path(directory)
    with probes.locked(directory):
        path = directory / "native-protocol.json"
        if path.exists() and json.loads(path.read_text()) != metadata:
            raise ValueError("native factual protocol differs; use a new directory")
        probes.atomic_write(path, probes.canonical(metadata) + "\n")
    return manifest


def load_campaign(directory):
    manifest = probes.load_manifest(directory)
    expected = protocol_metadata(manifest)
    if json.loads((Path(directory) / "native-protocol.json").read_text()) != expected:
        raise ValueError("native factual protocol fingerprint mismatch")
    return manifest, expected


def normalize_usage(provider, raw):
    """Use the established provider conventions, keeping unavailable usage null."""
    if raw is None or raw == {} or raw == []:
        return None, "not_reported"
    try:
        steps = raw if provider == "glm" else [raw]
        if not isinstance(steps, list) or not steps:
            raise ValueError("expected usage steps")
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("usage step is not an object")
            required = ("input", "output") if provider == "glm" else ("input_tokens", "output_tokens")
            if any(key not in step for key in required):
                raise ValueError("core usage counters absent")
            numeric_keys = ({"input", "output", "reasoning", "total"} if provider == "glm" else
                            {"input_tokens", "output_tokens", "cached_input_tokens", "cache_write_input_tokens",
                             "cache_read_input_tokens", "cache_creation_input_tokens", "reasoning_output_tokens",
                             "reasoning_tokens", "total_tokens"})
            counters = [v for k, v in step.items() if k in numeric_keys]
            if provider == "glm":
                cache = step.get("cache", {})
                if not isinstance(cache, dict):
                    raise ValueError("cache usage is not an object")
                counters += [v for k, v in cache.items() if k in {"read", "write"}]
            if any(type(value) is not int or value < 0 for value in counters):
                raise ValueError("usage counters must be nonnegative integers")
        normalized = tokens(provider, raw)
        if normalized["cached_input_tokens"] > normalized["input_tokens"] or normalized["reasoning_output_tokens"] > normalized["output_tokens"]:
            raise ValueError("usage subset exceeds parent")
        return normalized, None
    except (ValueError, TypeError, AttributeError) as exc:
        return None, str(exc)


def json_events(stdout):
    result = []
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            result.append(item)
    return result


def validate_output(provider, stdout, final_path, exit_code):
    """Validate native completion and extraction; never parse a probability."""
    errors, events = [], json_events(stdout)
    final_text = final_path.read_text() if final_path.exists() else None
    try:
        response, raw_usage, model_usage, tools_used, session = native.decode(provider, stdout, final_path)
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        response, raw_usage, model_usage, tools_used, session = "", None, None, False, None
        errors.append("native_decode_failed:" + type(exc).__name__)
    if not isinstance(response, str):
        response = ""
        errors.append("nontext_final_response")
    if not isinstance(session, str):
        session = None
    evidence = {"session_present": isinstance(session, str) and bool(session),
                "native_completed": False, "final_message_matches": False,
                "recovered_reconnections": 0, "reported_models": []}
    truncated = False
    if exit_code != 0:
        errors.append("native_exit:" + str(exit_code))
    if provider in {"astra", "luna"}:
        messages = [(i, e["item"].get("text", "")) for i, e in enumerate(events)
                    if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
        completions = [i for i, e in enumerate(events) if e.get("type") == "turn.completed"]
        starts = [e.get("thread_id") for e in events if e.get("type") == "thread.started"]
        evidence["native_completed"] = len(completions) == 1
        evidence["final_message_matches"] = (len(messages) == 1 and isinstance(messages[0][1], str) and final_text is not None and
                                              messages[0][1].strip() == final_text.strip() == response.strip())
        if not evidence["final_message_matches"]:
            errors.append("native_final_file_or_message_mismatch")
        if len(starts) != 1 or starts[0] != session:
            errors.append("native_session_mismatch")
        if not completions or not messages or completions[-1] < messages[-1][0] or completions[-1] != len(events) - 1:
            errors.append("native_completion_order")
        for index, event in enumerate(events):
            if event.get("type") == "turn.failed":
                errors.append("native_turn_failed")
            if event.get("type") == "error":
                if event.get("message") == RECONNECT and messages and completions and index < messages[-1][0] < completions[-1]:
                    evidence["recovered_reconnections"] += 1
                else:
                    errors.append("unrecovered_native_error")
            kind = event.get("item", {}).get("type")
            if kind and kind not in {"agent_message", "reasoning"}:
                tools_used = True
    elif provider == "grok":
        try:
            body = json.loads(stdout)
        except ValueError:
            body = events[-1] if events else {}
        if not isinstance(body, dict):
            body = {}
        stop = body.get("stopReason")
        evidence["native_completed"] = stop in {"end_turn", "stop"} and not body.get("is_error")
        evidence["final_message_matches"] = isinstance(body.get("text", body.get("result")), str)
        tools_used = bool(tools_used or body.get("toolCalls"))
        truncated = stop in {"max_tokens", "length"}
        if truncated:
            evidence["native_completed"] = True
        if body.get("num_turns", 1) != 1:
            errors.append("unexpected_native_turn_count")
        evidence["provider_request_id"] = body.get("requestId")
        if isinstance(body.get("model"), str):
            evidence["reported_models"].append(body["model"])
        if isinstance(model_usage, dict):
            evidence["reported_models"] += [name for name in model_usage if isinstance(name, str)]
        evidence["reported_models"] = sorted(set(evidence["reported_models"]))
    else:
        starts = [e for e in events if e.get("type") == "step_start"]
        finishes = [e for e in events if e.get("type") == "step_finish"]
        sessions = {e.get("sessionID") for e in events if e.get("sessionID")}
        reason = finishes[-1].get("part", {}).get("reason") if finishes else None
        truncated = reason in {"length", "max-tokens", "max_tokens"}
        evidence["native_completed"] = bool(finishes) and len(starts) == len(finishes) and (reason == "stop" or truncated)
        evidence["final_message_matches"] = any(e.get("type") == "text" for e in events)
        if sessions != {session}:
            errors.append("native_session_mismatch")
        if not events or events[-1].get("type") != "step_finish":
            errors.append("native_completion_order")
        for event in events:
            kind = event.get("type")
            if kind in {"error", "step_error"}:
                errors.append("native_error_event")
            if kind not in {"step_start", "step_finish", "text", "reasoning", "error", "step_error"}:
                tools_used = True
    if not evidence["native_completed"]:
        errors.append("native_completion_not_established")
    if not evidence["session_present"]:
        errors.append("native_session_missing")
    evidence["no_tools_reported"] = not tools_used
    evidence["validation_errors"] = sorted(set(errors))
    return {"response": response, "raw_usage": raw_usage, "raw_model_usage": model_usage,
            "session_id": session, "tools_used": bool(tools_used), "truncated": truncated,
            "native_validation": evidence, "final_file_text": final_text}


def checked_command(provider, prompt, empty, final, effort, actor):
    command, env, stdin = native.native_command(provider, prompt, empty, final, effort,
                                               rule=SYSTEM_INSTRUCTION, campaign=actor)
    expected_model = "zai-coding-plan/glm-5.3" if provider == "glm" else native.MODELS[provider]
    selections = [command[i + 1] for i, value in enumerate(command[:-1]) if value == "--model"]
    if selections != [expected_model]:
        raise ValueError("native command does not select the exact requested model")
    if provider in {"astra", "luna"}:
        if "model_reasoning_effort=" + json.dumps(effort) not in command:
            raise ValueError("native command reasoning effort differs")
        if "developer_instructions=" + json.dumps(SYSTEM_INSTRUCTION) not in command:
            raise ValueError("native Codex system instruction differs")
    elif provider == "grok":
        positions = [i for i, value in enumerate(command[:-1]) if value == "--reasoning-effort"]
        if len(positions) != 1 or command[positions[0] + 1] != effort:
            raise ValueError("native Grok reasoning effort differs")
        positions = [i for i, value in enumerate(command[:-1]) if value == "--system-prompt-override"]
        if len(positions) != 1 or command[positions[0] + 1] != SYSTEM_INSTRUCTION:
            raise ValueError("native Grok system instruction differs")
    else:
        # Reuse the proven isolated runtime while removing the probability-agent
        # name/description. The system prompt remains the frozen factual rule.
        config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        agent = config["agent"].pop("forecast")
        if agent.get("prompt") != SYSTEM_INSTRUCTION:
            raise ValueError("native GLM system instruction differs")
        agent["description"] = "Single factual-recall probe"
        config["agent"]["factual-probe"] = agent
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config)
        position = command.index("--agent")
        command[position + 1] = "factual-probe"
    return command, env, stdin


def run_attempt(directory, manifest, metadata, request, launch, timeout):
    """One native launch. Persist output before importing; no implicit retries."""
    directory = Path(directory)
    provider = manifest["provider"]
    final = directory / launch["final_file"]
    final.parent.mkdir(parents=True, exist_ok=True)
    final.unlink(missing_ok=True)
    started = time.monotonic()
    stdout, stderr, exit_code, process_id = "", "", "not_started", None
    command, runtime = [], "unknown"
    # All providers get an empty context. GLM additionally requires a temp root
    # outside Git, matching its proven no-index runtime.
    temp_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()).resolve()
    if provider == "glm" and any((p / ".git").exists() for p in (temp_root, *temp_root.parents)):
        raise ValueError("GLM temporary root must be outside Git")
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="consistency-factual-", dir=temp_root) as temporary:
        empty, client_tmp = Path(temporary) / "empty", Path(temporary) / "client-tmp"
        empty.mkdir()
        client_tmp.mkdir()
        try:
            command, env, stdin = checked_command(provider, request["prompt"], empty, final, manifest["reasoning"],
                                                   "probes-" + manifest["campaign_id"][:16])
            env["TMPDIR"] = str(client_tmp)
            runtime = "private-native-config" if provider in {"astra", "luna"} and env.get("CONSISTENCY_CODEX_RUNTIME") else "existing-native-wrapper"
            process = subprocess.Popen(command, cwd=empty, env=env, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            process_id = process.pid
            probes.atomic_write(final.with_suffix(".process.json"), probes.canonical({"pid": process_id, "attempt_id": launch["attempt_id"]}) + "\n")
            try:
                stdout, stderr = process.communicate(stdin, timeout=timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
                exit_code = "timeout"
        except (OSError, ValueError, KeyError) as exc:
            stderr = type(exc).__name__ + ": " + str(exc)
    decoded = validate_output(provider, stdout, final, exit_code)
    normalized, usage_issue = normalize_usage(provider, decoded["raw_usage"])
    errors = decoded["native_validation"]["validation_errors"]
    provenance = {"transport": "subscription-native-cli", "adapter": COLLECTOR,
                  "local_attempt_id": launch["attempt_id"],
                  "request_id": "local-launch-" + launch["attempt_id"], "request_id_source": "local_launch"}
    if decoded["session_id"]:
        provenance["session_id"] = decoded["session_id"]
    provider_request = decoded["native_validation"].get("provider_request_id")
    if provider_request:
        provenance.update(request_id=provider_request, request_id_source="provider")
    record = {**{k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "protocol_id", "prompt_sha256")},
              "requested_model": manifest["model"], "collector": COLLECTOR,
              "system_instruction_sha256": metadata["system_instruction_sha256"],
              "native_protocol_fingerprint": metadata["fingerprint"], "status": "infrastructure_error" if errors else "completed",
              "completed": not errors, "started_at": launch["started_at"], "finished_at": stamp(),
              "elapsed_seconds": round(time.monotonic() - started, 3), "exit_code": exit_code,
              "usage": normalized, "usage_validation_issue": usage_issue, "provenance": provenance,
              "runtime_variant": runtime, **decoded,
              "provider_model_mismatch": any(m not in metadata["reported_model_aliases"] for m in decoded["native_validation"]["reported_models"]),
              "native_evidence": {"stdout": stdout, "stderr": stderr, "command": command, "process_id": process_id}}
    if provider == "glm":
        record["runtime_profile"] = native.GLM_RUNTIME_PROFILE
    if errors:
        record["error"] = "; ".join(errors)
    path = directory / launch["receipt_file"]
    probes.atomic_write(path, probes.canonical(record) + "\n")
    return path


def validate_saved(path, manifest, metadata):
    rows = probes.read_jsonl(path)
    if len(rows) != 1:
        raise ValueError("native attempt file must contain exactly one receipt")
    receipt = rows[0]
    probes.validate_receipt(receipt, manifest)
    if (receipt.get("collector") != COLLECTOR or receipt.get("system_instruction_sha256") != metadata["system_instruction_sha256"]
            or receipt.get("native_protocol_fingerprint") != metadata["fingerprint"]):
        raise ValueError("saved receipt does not match frozen native factual protocol")
    return receipt


def report(directory):
    manifest, metadata = load_campaign(directory)
    result = probes.write_report(directory)
    result["collector"] = COLLECTOR
    result["native_protocol_fingerprint"] = metadata["fingerprint"]
    path = Path(directory) / "receipts.jsonl"
    saved = probes.read_jsonl(path) if path.exists() else []
    result["native_runtime_variants"] = dict(Counter(r["receipt"].get("runtime_variant") for r in saved))
    result["native_sessions_recorded"] = sum(bool(r["receipt"].get("session_id")) for r in saved)
    result["native_usage_unavailable"] = sum(r["receipt"].get("usage") is None for r in saved) + result["launches_without_final_receipt"]
    probes.atomic_write(Path(directory) / "native-report.json", json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def collect(directory, *, execute=False, max_calls=None, workers=1, timeout=180, retry_errors=False):
    directory = Path(directory).resolve()
    manifest, metadata = load_campaign(directory)
    initial = report(directory)
    if not execute:
        return {"status": "plan_only", "model": manifest["model"], "provider": manifest["provider"],
                "reasoning": manifest["reasoning"], "catalog_count": manifest["catalog_count"],
                "request_ready": initial["request_ready"], "completed_observations": initial["completed_observations"],
                "launch_reservations": initial["launch_reservations"], "new_launches": 0}
    if type(max_calls) is not int or max_calls < 1 or type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("execution requires positive --max-calls and --workers between 1 and 16")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive finite duration")
    if date.fromisoformat(manifest["as_of"]) > datetime.now(timezone.utc).date():
        raise ValueError("cannot collect a campaign with a future preparation date")
    if initial["synthetic"]:
        raise ValueError("cannot launch a synthetic campaign")
    configuration = {"collector": COLLECTOR, "native_protocol_fingerprint": metadata["fingerprint"],
                     "codex_runtime": os.environ.get("CONSISTENCY_CODEX_RUNTIME") if manifest["provider"] in {"astra", "luna"} else None,
                     "glm_runtime_profile": native.GLM_RUNTIME_PROFILE if manifest["provider"] == "glm" else None}
    collection = directory / "collection"
    with probes.locked(collection):
        config_path = collection / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError("native runtime treatment differs; use a new campaign")
        probes.atomic_write(config_path, probes.canonical(configuration) + "\n")
        launches_path = collection / "launches.jsonl"
        launches = probes.read_jsonl(launches_path) if launches_path.exists() else []
        for launch in launches:
            path = directory / launch["receipt_file"]
            if not path.exists():
                raise ValueError("reserved native launch has no final receipt; reconcile before resuming")
            validate_saved(path, manifest, metadata)
            probes.import_receipts(directory, path)
        ledger = directory / "receipts.jsonl"
        saved = probes.read_jsonl(ledger) if ledger.exists() else []
        current = probes.append_to_history(saved, [], manifest)
        if any(r.get("collector") != COLLECTOR for r in current.values()):
            raise ValueError("campaign contains non-native-factual observations")
        pending, stop = [], "complete"
        for entry in manifest["probes"]:
            if not entry["review"]["request_ready"]:
                continue
            prior = current.get(entry["probe_id"])
            if prior and prior["status"] != "infrastructure_error":
                continue
            if prior and not retry_errors:
                stop = "failed_attempt_requires_explicit_retry"
                break
            prompt = probes.PROBE_PROMPT.format(question=entry["probe"]["question"])
            pending.append({"probe_id": entry["probe_id"], "model": manifest["model"], "provider": manifest["provider"],
                            "reasoning": manifest["reasoning"], "protocol_id": probes.PROTOCOL,
                            "prompt_sha256": probes.sha(prompt), "prompt": prompt})
        new_launches = 0
        if stop == "complete":
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures, position = {}, 0
                while position < len(pending) or futures:
                    while stop == "complete" and position < len(pending) and len(futures) < workers:
                        if len(launches) >= max_calls:
                            stop = "call_allowance_exhausted"
                            break
                        request = pending[position]
                        position += 1
                        ident = probes.sha(manifest["campaign_id"] + ":native:" + str(len(launches)))
                        launch = {"attempt_id": ident, "probe_id": request["probe_id"], "started_at": stamp(),
                                  "receipt_file": "collection/attempts/" + ident + ".jsonl",
                                  "final_file": "collection/attempts/" + ident + ".final.txt",
                                  "model": manifest["model"], "provider": manifest["provider"], "reasoning": manifest["reasoning"],
                                  "prompt_sha256": request["prompt_sha256"], "native_protocol_fingerprint": metadata["fingerprint"]}
                        launches.append(launch)
                        probes.atomic_write(launches_path, "".join(probes.canonical(r) + "\n" for r in launches))
                        futures[pool.submit(run_attempt, directory, manifest, metadata, request, launch, timeout)] = launch
                        new_launches += 1
                    if not futures:
                        break
                    ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in ready:
                        futures.pop(future)
                        path = future.result()
                        receipt = validate_saved(path, manifest, metadata)
                        probes.import_receipts(directory, path)
                        if receipt["status"] == "infrastructure_error":
                            stop = "provider_failure"
                    if stop != "complete" and not futures:
                        break
    result = report(directory)
    return {"status": stop, "model": manifest["model"], "provider": manifest["provider"], "reasoning": manifest["reasoning"],
            "catalog_count": manifest["catalog_count"], "completed_observations": result["completed_observations"],
            "launch_reservations": result["launch_reservations"], "new_launches": new_launches,
            "statuses": result["statuses"], "verified_cutoff": None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--provider", choices=sorted(EFFORTS), required=True)
    preparation.add_argument("--effort")
    preparation.add_argument("--as-of", required=True)
    preparation.add_argument("--probes", type=Path)
    preparation.add_argument("--cutoffs", type=Path)
    collector = commands.add_parser("collect")
    collector.add_argument("--run-dir", type=Path, required=True)
    collector.add_argument("--execute", action="store_true")
    collector.add_argument("--max-calls", type=int)
    collector.add_argument("--workers", type=int, default=1)
    collector.add_argument("--timeout", type=float, default=180)
    collector.add_argument("--retry-errors", action="store_true")
    reporting = commands.add_parser("report")
    reporting.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.run_dir, provider=args.provider, effort=args.effort, as_of=args.as_of,
                    probes_path=args.probes, cutoffs_path=args.cutoffs)
            result = collect(args.run_dir)
        elif args.command == "report":
            complete = report(args.run_dir)
            result = {k: complete[k] for k in ("model", "reasoning", "catalog_count", "completed_observations", "statuses", "launch_reservations", "verified_cutoff")}
        else:
            result = collect(args.run_dir, execute=args.execute, max_calls=args.max_calls, workers=args.workers,
                             timeout=args.timeout, retry_errors=args.retry_errors)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 1 if result.get("status") in {"provider_failure", "failed_attempt_requires_explicit_retry"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
