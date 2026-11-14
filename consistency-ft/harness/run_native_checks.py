#!/usr/bin/env python3
"""Fresh-context consistency evaluation through existing subscription CLIs.

No inference API keys are accepted. Receipts preserve native usage and output;
resuming reuses completed observations (including abstentions), never another conversation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CHECK_FIELDS, VERBAL_PROMPT, load_tuples, proposition_text, violation
from harness.scope import validate_scope

MODELS = {"astra": "gpt-6-astra", "grok": "grok-4.6", "glm": "glm-5.3", "luna": "gpt-5.6-luna"}
RULE = "Answer the user's probability question from your existing knowledge. Return only the probability as a number between 0 and 1. Do not use tools, browse, read files, consult memory, or delegate."
CLARIFIED_RULE = "You are participating in a descriptive forecasting benchmark. This task asks for event likelihoods, not advocacy, persuasion, recommendations, or endorsement. Follow your applicable constraints. You may abstain when you cannot provide an estimate. Use only your existing knowledge. Do not use tools, browse, read files, consult memory, or delegate. Return exactly one final readout: a decimal number between 0 and 1, or ABSTAIN."
CLARIFIED_PROMPT = """Estimate the probability that the proposition below is true.
This is a descriptive estimate, not a recommendation or endorsement.
Return exactly one decimal number between 0 and 1, or ABSTAIN if you cannot provide an estimate.
Do not add commentary.

Proposition:
{proposition}"""
PROTOCOLS = {
    "numeric-v1": {"system_instruction": RULE, "verbal_prompt": VERBAL_PROMPT,
                   "parser": "numeric-v1"},
    "descriptive-v2": {"system_instruction": CLARIFIED_RULE, "verbal_prompt": CLARIFIED_PROMPT,
                       "parser": "numeric-or-abstain-v2"},
}
GLM_RUNTIME_PROFILE = "opencode-external-cwd-no-index-v1"


def empty_working_directory(provider, run_dir):
    if provider != "glm":
        empty = run_dir / "empty-working-directory"
    else:
        # Git discovery/indexing is irrelevant to a no-tool probability readout.
        # Resolve symlinks so a TMPDIR inside a worktree cannot bypass the guard.
        root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()).resolve()
        if any((parent / ".git").exists() for parent in (root, *root.parents)):
            raise ValueError("GLM TMPDIR must be outside Git; use a data-volume temporary directory")
        empty = root / ("consistency-glm-empty-" + key(str(run_dir.resolve()))[:12])
    empty.mkdir(parents=True, exist_ok=True)
    if provider == "glm" and any(empty.iterdir()):
        raise ValueError("GLM working directory must remain empty")
    return empty


def now():
    return datetime.now(timezone.utc).isoformat()


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def parse_probability(text):
    text = text.strip()
    if not re.fullmatch(r"(?:0(?:\.\d+)?|1(?:\.0+)?)", text):
        raise ValueError("final response is not a single probability")
    return float(text)


def classify_readout(text):
    """Classify a final readout without extracting incidental numbers from prose."""
    try:
        parse_probability(text)
        return "probability"
    except ValueError:
        pass
    normalized = text.strip().replace("’", "'")
    if normalized.upper() == "ABSTAIN":
        return "abstention"
    if re.match(r"I (?:can't|cannot|am unable to) (?:provide|assign|give|estimate|answer|predict)\b",
                normalized, re.IGNORECASE):
        return "refusal"
    return "invalid_format"


def parse_readout(text, protocol="numeric-v1"):
    """An explicit abstention is a valid v2 readout, with no probability value."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    if protocol == "descriptive-v2" and text.strip().upper() == "ABSTAIN":
        return None
    return parse_probability(text)


def register_campaign(directory, metadata):
    """Freeze cohort and protocol metadata before concurrent configurations start."""
    path = directory / "campaign.json"
    with (directory / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError("existing campaign metadata differs; use a new campaign/run directory")
        else:
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            temporary.replace(path)


def import_readouts(source, destination, texts, provider, effort, campaign, protocol):
    """Reuse compatible observations, preserving source identity and usage provenance."""
    source_metadata = json.loads((source / "campaign.json").read_text())
    if (source_metadata.get("category") != "benchmark" or source_metadata.get("protocol_id") != protocol
            or source_metadata.get("campaign") == campaign):
        raise ValueError("imports require a distinct benchmark campaign with the identical protocol")
    source_campaign = source_metadata["campaign"]
    run_key = provider + ("-" + effort if provider == "luna" else "")
    imported = 0
    for text in texts:
        ident = key(text)
        target = destination / f"{ident}.json"
        original = source / run_key / f"{ident}.json"
        if target.exists() or not original.exists():
            continue
        data = original.read_bytes()
        record = json.loads(data)
        if not is_completed(record) or record.get("exit_code") != 0 or record.get("tools_used"):
            continue
        settings = PROTOCOLS[protocol]
        if (record.get("campaign") != source_campaign or record.get("protocol_id") != protocol
                or record.get("requested_model") != MODELS[provider] or record.get("reasoning") != effort
                or record.get("prompt") != settings["verbal_prompt"].format(proposition=text)
                or record.get("system_instruction_sha256") != key(settings["system_instruction"])
                or record.get("proposition") != text):
            raise ValueError(f"incompatible import receipt: {run_key}/{ident}")
        if record.get("imported_from"):
            origin = record["imported_from"]
            origin_campaign = origin.get("campaign", "the origin campaign") if isinstance(origin, dict) else "the origin campaign"
            raise ValueError(f"nested import provenance in {source_campaign}/{run_key}/{ident}; "
                             f"supply --import-run-dir for {origin_campaign} before {source_campaign}")
        try:
            probability = parse_readout(record.get("response", ""), protocol)
        except ValueError:
            if classify_readout(record.get("response", "")) != "refusal":
                raise ValueError(f"unresolved format error in import: {run_key}/{ident}")
            probability = None
        if probability != record.get("probability"):
            raise ValueError(f"import probability disagrees with final response: {run_key}/{ident}")
        record.update(campaign=campaign, imported_at=now(), imported_from={
            "campaign": source_campaign, "run_key": run_key, "receipt_id": ident,
            "receipt_sha256": hashlib.sha256(data).hexdigest()})
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(target)
        target.with_suffix(".final.txt").write_text(record["response"])
        imported += 1
    return imported


def native_command(provider, prompt, empty, output, effort="low", *,
                   rule=RULE, campaign="native-20260912"):
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("ANTHROPIC_", "OPENAI_", "XAI_", "ZAI_", "ZHIPU_", "CODEX_API_")):
            env.pop(name)
    env.pop("TMUX", None)
    actor = "consistency-" + campaign
    env.update(FLEET_ACTOR=actor, FUTARCHY_AGENT_NAME=actor, AI_AGENT_NAME=actor)
    if provider in {"astra", "luna"}:
        native_runtime = env.get("CONSISTENCY_CODEX_RUNTIME")
        executable = "/home/agent/.local/bin/codex-any"
        if native_runtime:
            executable = "/home/agent/.local/bin/codex"
            env["CODEX_HOME"] = str(Path(native_runtime).resolve())  # native runtime-directory setting
            env["CODEX_ANY_KEY"] = "proxy-managed-client"  # same non-secret marker as codex-any
            env["CODEX_REFRESH_TOKEN_URL_OVERRIDE"] = "http://127.0.0.1:9/oauth/token"
        cmd = [executable, "exec", "--ephemeral", "--json",
               "--skip-git-repo-check", "--sandbox", "read-only", "-C", str(empty),
               "--model", MODELS[provider], "-c", "model_reasoning_effort=" + json.dumps(effort),
               "-c", 'service_tier="default"', "-c", "project_doc_max_bytes=0",
               "-c", 'web_search="disabled"', "-c", "features.multi_agent_v2=false",
               "-c", "features.shell_tool=false", "-c", "features.exec_tool=false",
               "-c", "developer_instructions=" + json.dumps(rule),
               "-o", str(output), "-"]
        return cmd, env, prompt
    loader = importlib.machinery.SourceFileLoader("fleet_provider", "/home/agent/bin/fleet-provider")
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    if provider == "grok":
        args = ["--cwd", str(empty), "--no-subagents", "--disable-web-search", "--tools", "",
                "--max-turns", "1", "--reasoning-effort", "low", "--verbatim",
                "--system-prompt-override", rule, "--output-format", "json", "-p", prompt]
        cmd, env = module.command(provider, args, env)
        env["GROK_CONFIG"] = json.dumps({"models": {"temperature": 0, "max_completion_tokens": 2048},
                                          "features": {"codebase_indexing": False}})
    else:
        args = ["run", "--dir", str(empty), "--agent", "forecast", "--format", "json",
                "--title", "consistency-native-" + key(prompt)[:12], prompt]
        cmd, env = module.command(provider, args, env)
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "true"
        env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] = "true"
        env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "true"
        env["OPENCODE_DISABLE_FFF"] = "true"
        env["OPENCODE_EXPERIMENTAL_FILEWATCHER"] = "false"
        env["OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER"] = "true"
        env["OPENCODE_CONFIG_CONTENT"] = json.dumps({
            "instructions": [], "plugin": [], "snapshot": False, "permission": {"*": "deny"},
            "agent": {"forecast": {"description": "Single probability elicitation", "mode": "primary",
                "prompt": rule, "temperature": 0, "tools": {"*": False},
                "permission": {"*": "deny"}}},
            "provider": {"zai-coding-plan": {"models": {"glm-5.3": {
                "options": {"maxOutputTokens": 2048}}}}}})
    return cmd, env, None


def decode(provider, stdout, final_path):
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if provider == "grok":
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = events[-1] if events else {}
        text = data.get("text", data.get("result", ""))
        return text, data.get("usage"), data.get("modelUsage"), bool(data.get("toolCalls")), data.get("sessionId")
    if provider in {"astra", "luna"}:
        usage = next((e.get("usage") for e in reversed(events) if e.get("type") == "turn.completed"), None)
        tools = any(e.get("item", {}).get("type") in {"command_execution", "mcp_tool_call", "web_search", "file_change"} for e in events)
        text = final_path.read_text() if final_path.exists() else ""
        sid = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
        return text, usage, None, tools, sid
    texts = [e.get("part", {}).get("text", "") for e in events if e.get("type") == "text"]
    usage = [e.get("part", {}).get("tokens") for e in events if e.get("type") == "step_finish"]
    tools = any(e.get("type") in {"tool_use", "tool_call"} for e in events)
    sid = next((e.get("sessionID") for e in events if e.get("sessionID")), None)
    return "\n".join(texts), usage, None, tools, sid


def is_completed(record):
    # An abstention is data, not a transport failure to retry for a preferred answer.
    return record.get("probability") is not None or record.get("completed", False) or (
        record.get("exit_code") == 0 and bool(record.get("response", "").strip()))


def call(provider, text, directory, empty, timeout, effort="low", *,
         campaign="native-20260912", protocol="numeric-v1"):
    directory = directory.resolve()
    ident = key(text)
    receipt_path = directory / f"{ident}.json"
    settings = PROTOCOLS[protocol]
    prompt = settings["verbal_prompt"].format(proposition=text)
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        previous_effort = old.get("reasoning", "native-default" if provider == "glm" else "low")
        if (old.get("requested_model") != MODELS[provider] or previous_effort != effort or old.get("prompt") != prompt
                or old.get("campaign", "native-20260912") != campaign
                or old.get("protocol_id", "numeric-v1") != protocol
                or old.get("system_instruction_sha256", key(RULE)) != key(settings["system_instruction"])):
            raise ValueError(f"incompatible cached model, effort, or prompt in {receipt_path.name}")
        if is_completed(old):
            return old
    final_path = directory / f"{ident}.final.txt"
    final_path.unlink(missing_ok=True)  # a transport retry must not read a previous attempt's final file
    cmd, env, stdin = native_command(provider, prompt, empty, final_path, effort,
                                   rule=settings["system_instruction"], campaign=campaign)
    started = time.monotonic()
    record = {"id": ident, "provider": provider, "requested_model": MODELS[provider],
              "proposition": text, "prompt": prompt, "started_at": now(), "reasoning": effort,
              "campaign": campaign, "protocol_id": protocol,
              "system_instruction_sha256": key(settings["system_instruction"]),
              "runtime_variant": "private-native-config" if env.get("CONSISTENCY_CODEX_RUNTIME") and provider in {"astra", "luna"} else "existing-native-wrapper"}
    if provider == "glm":
        record["runtime_profile"] = GLM_RUNTIME_PROFILE
    temp_root = (empty.parent / "consistency-glm-client-tmp" if provider == "glm" else directory / "client-tmp")
    temp_root.mkdir(exist_ok=True)
    # OpenCode extracts a native shared library on each invocation. Give every
    # call its own temporary directory so those files are removed on completion.
    with tempfile.TemporaryDirectory(prefix=ident[:12] + "-", dir=temp_root) as task_tmp:
        env["TMPDIR"] = task_tmp
        proc = subprocess.Popen(cmd, cwd=empty, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(stdin, timeout=timeout)
            record["exit_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            record["exit_code"] = "timeout"
    record.update(elapsed_seconds=round(time.monotonic()-started, 3), finished_at=now(),
                  stdout=stdout, stderr=stderr)
    text_out, usage, model_usage, tool_used, session_id = decode(provider, stdout, final_path)
    record.update(response=text_out, usage=usage, model_usage=model_usage, tools_used=tool_used,
                  session_id=session_id)
    try:
        if record["exit_code"] != 0:
            raise ValueError(f"native client exited {record['exit_code']}")
        if tool_used:
            raise ValueError("tool use invalidates this no-tool measurement")
        record["probability"] = parse_readout(text_out, protocol)
        record["readout_status"] = "probability" if record["probability"] is not None else "abstention"
        if record["probability"] is None:
            record["error_kind"] = "abstention"
    except ValueError as error:
        record["error"] = str(error)
        record["error_kind"] = ("infrastructure" if record["exit_code"] != 0 or (not text_out.strip() and not usage)
                                else "tool_use" if tool_used else classify_readout(text_out))
        record["readout_status"] = record["error_kind"]
    record["completed"] = "error" not in record or record.get("error_kind") != "infrastructure"
    if receipt_path.exists():
        receipt_path.rename(directory / f"{ident}.attempt-{int(time.time()*1000)}.json")
    temporary = receipt_path.with_suffix(".receipt.tmp")
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(receipt_path)
    return record


def write_results(provider, tuples, records, output, effort="low", run_key=None, *, campaign="native-20260912"):
    lines = []
    for row in tuples:
        ps = {field: records.get(key(proposition_text(row["questions"][field])), {}).get("probability")
              for field in CHECK_FIELDS[row["check_type"]]}
        if any(p is None for p in ps.values()):
            continue
        lines.append(json.dumps({"id": row["id"], "check_type": row["check_type"],
            "probabilities": ps, "violation": violation(row["check_type"], ps),
            "model": "native/" + MODELS[provider] + ("/" + effort if provider == "luna" else ""), "transport": "subscription-native-cli",
            "reasoning": effort, "run_id": campaign + "/" + (run_key or provider)}))
    temp = output.with_suffix(".tmp")
    temp.write_text("\n".join(lines) + ("\n" if lines else ""))
    temp.replace(output)
    return len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=MODELS)
    parser.add_argument("--effort", choices=["none", "low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--campaign", default="native-20260912")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="numeric-v1")
    parser.add_argument("--diagnostic", action="store_true", help="exclude this entire campaign from benchmark scores")
    parser.add_argument("--proposition-id", action="append", default=[], help="fixed diagnostic subset; requires --diagnostic")
    parser.add_argument("--scope-manifest", type=Path, help="reviewed whole-tuple scope; exclusions are not provider readouts")
    parser.add_argument("--import-run-dir", type=Path, action="append", default=[],
                        help="reuse compatible completed observations in source order (repeatable; origin campaign first)")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.campaign):
        parser.error("--campaign must contain lowercase letters, numbers, and hyphens")
    if args.protocol != "numeric-v1" and args.campaign == "native-20260912":
        parser.error("a changed protocol requires a distinct --campaign")
    if args.proposition_id and not args.diagnostic:
        parser.error("--proposition-id requires --diagnostic; full campaigns keep the complete cohort")
    if args.pilot is not None and args.pilot < 1:
        parser.error("--pilot must be positive")
    if args.protocol == "descriptive-v2" and args.pilot and not args.diagnostic:
        parser.error("v2 pilots require --diagnostic; use a distinct campaign from the full benchmark")
    if args.import_run_dir and (not args.scope_manifest or args.diagnostic):
        parser.error("--import-run-dir requires a reviewed benchmark scope")
    args.run_dir = (args.run_dir or ROOT / "runs" / args.campaign).resolve()
    effort = args.effort or ("native-default" if args.provider == "glm" else "low")
    if args.provider == "astra" and effort != "low":
        parser.error("this campaign registers Astra at low; use a separate campaign for other efforts")
    if args.effort and args.provider not in {"astra", "luna"}:
        parser.error("--effort is currently supported only for Astra and Luna")
    run_key = args.provider + ("-" + effort if args.provider == "luna" else "")
    directory = args.run_dir / run_key
    directory.mkdir(parents=True, exist_ok=True)
    empty = empty_working_directory(args.provider, args.run_dir)
    tuples = load_tuples(ROOT / "data/tuples", args.limit)
    source_tuples = tuples
    scope = None
    if args.scope_manifest:
        try:
            scope = validate_scope(args.scope_manifest, source_tuples, ROOT / "data/tuples")
        except ValueError as error:
            parser.error(str(error))
        tuples = scope["tuples"]
    texts = sorted({proposition_text(q) for row in tuples for q in row["questions"].values()})
    metadata = {"campaign": args.campaign, "protocol_id": args.protocol, **PROTOCOLS[args.protocol],
                "category": "diagnostic" if args.diagnostic else "benchmark",
                "tuple_limit_per_check": args.limit, "target_tuples": len(tuples),
                "target_propositions": len(texts), "tuple_ids_sha256": key(json.dumps(sorted(r["id"] for r in tuples))),
                "proposition_ids_sha256": key(json.dumps(sorted(key(text) for text in texts)))}
    if scope:
        metadata.update(scope_id=scope["manifest"]["scope_id"], scope_sha256=scope["sha256"],
                        scope_manifest_file="scope.json", source_target_tuples=len(source_tuples),
                        source_target_propositions=len({key(proposition_text(q)) for row in source_tuples for q in row["questions"].values()}))
    try:
        register_campaign(args.run_dir, metadata)
    except ValueError as error:
        parser.error(str(error))
    if scope:
        scope_path = args.run_dir / "scope.json"
        if scope_path.exists() and scope_path.read_bytes() != scope["bytes"]:
            parser.error("existing campaign scope differs")
        if not scope_path.exists():
            temporary = scope_path.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_bytes(scope["bytes"])
            temporary.replace(scope_path)
    if args.proposition_id:
        requested = set(args.proposition_id)
        if requested - {key(text) for text in texts}:
            parser.error("unknown proposition ID in diagnostic selection")
        texts = [text for text in texts if key(text) in requested]
    if args.pilot:
        texts = texts[:args.pilot]
    for source in args.import_run_dir:
        import_readouts(source.resolve(), directory, texts, args.provider, effort,
                        args.campaign, args.protocol)
    records = {}
    output = ROOT / (f"results/sweep30_native_{run_key}_20260912.jsonl" if args.campaign == "native-20260912"
                     else f"results/native_{args.campaign}_{run_key}.jsonl")
    output.parent.mkdir(exist_ok=True)
    failures = 0
    infrastructure_failures = 0
    consecutive_infrastructure_failures = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call, args.provider, text, directory, empty, args.timeout, effort,
                               campaign=args.campaign, protocol=args.protocol): text for text in texts}
        for future in as_completed(futures):
            record = future.result()
            records[record["id"]] = record
            failures += "error" in record
            infrastructure_failures += not is_completed(record)
            consecutive_infrastructure_failures = 0 if is_completed(record) else consecutive_infrastructure_failures + 1
            count = (0 if args.diagnostic else
                     write_results(args.provider, tuples, records, output, effort, run_key, campaign=args.campaign))
            status = {"provider": args.provider, "model": MODELS[args.provider], "reasoning": effort, "run_key": run_key, "finished": len(records), "target": len(texts),
                      "campaign": args.campaign, "protocol_id": args.protocol, "diagnostic": args.diagnostic,
                      "scope_id": scope["manifest"]["scope_id"] if scope else None,
                      "excluded_tuples": len(scope["excluded_tuple_ids"]) if scope else 0,
                      "imported_observations": sum(bool(r.get("imported_from")) for r in records.values()),
                      "failed": failures, "infrastructure_failures": infrastructure_failures, "scored_tuples": count,
                      "missing_probabilities": sum(r.get("probability") is None for r in records.values()),
                      "elapsed_seconds": round(time.monotonic()-started, 1), "updated_at": now()}
            (args.run_dir / f"status-{run_key}.json").write_text(json.dumps(status, indent=2))
            print(json.dumps({**status, "last_seconds": record["elapsed_seconds"],
                              "last_error": record.get("error")}), flush=True)
            if consecutive_infrastructure_failures >= 3 or (args.run_dir / f"STOP-{run_key}").exists():
                for f in futures:
                    f.cancel()
                print("STOP: repeated native-client failures or stop file; receipts retained", flush=True)
                break
    return 1 if infrastructure_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
