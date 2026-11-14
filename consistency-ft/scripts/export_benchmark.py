#!/usr/bin/env python3
"""Snapshot private runtime evidence and rebuild the public, queryable benchmark.

Use --refresh to import local results/ and runs/. Without it, only tracked data
is read. Output never depends on a checkout's mtimes or the current time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import CHECK_FIELDS, VERBAL_PROMPT, load_tuples, proposition_text, violation
from harness.run_native_checks import PROTOCOLS, RULE, classify_readout, parse_probability
from harness.scope import validate_scope

SNAPSHOTS = ROOT / "data/benchmark-runs"
CAMPAIGN = "native-20260912"
COMPARISON_POLICIES = ("all-settings", "complete-historical-plus-native")
NATIVE_CONFIGS = [
    ("astra", "astra", "gpt-6-astra", "low"),
    ("grok", "grok", "grok-4.6", "low"),
    ("glm", "glm", "glm-5.3", "native-default"),
] + [(f"luna-{effort}", "luna", "gpt-5.6-luna", effort)
     for effort in ("none", "low", "medium", "high", "xhigh", "max")]
LEGACY_FIELDS = {
    "id", "model", "check_type", "probabilities", "violation", "title",
    "verbalized_probability", "logprob_probability", "abs_diff",
}
RECEIPT_FIELDS = {
    "id", "provider", "requested_model", "reasoning", "started_at", "finished_at",
    "elapsed_seconds", "exit_code", "response", "probability", "completed",
    "error_kind", "error", "tools_used", "usage", "model_usage",
    "runtime_variant", "runtime_profile", "campaign", "protocol_id", "system_instruction_sha256", "readout_status", "imported_from", "imported_at",
}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(encoded(row) + "\n" for row in rows))
    return digest(path.read_bytes())


def utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def finite_probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def sanitize_receipt(raw, path, model, reasoning):
    """Explicit allowlist: no CLI logs, local paths, auth data or session IDs."""
    row = {key: raw[key] for key in RECEIPT_FIELDS if key in raw}
    row["requested_model"] = raw.get("requested_model", model)
    row["reasoning"] = raw.get("reasoning", reasoning)
    row["runtime_variant"] = raw.get("runtime_variant", "existing-native-wrapper")
    row["attempt_id"] = path.name.removesuffix(".json")
    row["is_current"] = ".attempt-" not in path.name
    row["completed"] = raw.get("completed", raw.get("exit_code") == 0 and bool(raw.get("response")))
    if raw.get("tools_used") or raw.get("exit_code") != 0 or not finite_probability(raw.get("probability")):
        row["probability"] = None
        row["error_kind"] = ("infrastructure" if raw.get("exit_code") != 0 or not row["completed"] else
                             "tool_use" if raw.get("tools_used") else classify_readout(raw.get("response", "")))
        if row["error_kind"] == "probability":
            row["error_kind"] = "unparsed_probability"
        if row["error_kind"] != raw.get("error_kind"):
            row["original_error_kind"] = raw.get("error_kind")
            row["error_classification"] = "native-readout-v2"
    else:
        row["error_kind"] = None
    row["prompt_sha256"] = digest(raw.get("prompt", "").encode())
    # Provider usage structures contain token/cost counts and actual model names.
    # Keep numeric leaves only; no arbitrary strings from nested provider output.
    def numeric_tree(value):
        if isinstance(value, dict):
            return {str(k): numeric_tree(v) for k, v in sorted(value.items())
                    if isinstance(v, (dict, list, int, float)) and not isinstance(v, bool)}
        if isinstance(value, list):
            return [numeric_tree(v) for v in value if isinstance(v, (dict, list, int, float))]
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None
    for field in ("usage", "model_usage"):
        row[field] = numeric_tree(raw.get(field))
    return row


def validate_scores(rows, kind):
    seen = set()
    for row in rows:
        if row["id"] in seen:
            raise ValueError(f"duplicate observation {row['id']}")
        seen.add(row["id"])
        if kind == "checks":
            if set(row["probabilities"]) != set(CHECK_FIELDS[row["check_type"]]):
                raise ValueError(f"wrong probability roles: {row['id']}")
            if not all(finite_probability(p) for p in row["probabilities"].values()):
                raise ValueError(f"invalid probability: {row['id']}")
            expected = violation(row["check_type"], row["probabilities"])
            if not math.isclose(row["violation"], expected, abs_tol=1e-12):
                raise ValueError(f"violation does not match probabilities: {row['id']}")
        else:
            if not all(finite_probability(row[k]) for k in ("verbalized_probability", "logprob_probability")):
                raise ValueError(f"invalid elicitation probability: {row['id']}")
            expected = abs(row["verbalized_probability"] - row["logprob_probability"])
            if not math.isclose(row["abs_diff"], expected, abs_tol=1e-12):
                raise ValueError(f"elicitation disagreement mismatch: {row['id']}")


def refresh(root=ROOT, destination=SNAPSHOTS, campaigns=None, comparison_policy=None):
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"runs": []}
    selected_policy = comparison_policy if comparison_policy is not None else old.get("comparison_policy")
    if selected_policy is not None and selected_policy not in COMPARISON_POLICIES:
        raise ValueError(f"Unknown comparison policy: {selected_policy}")
    previous = {entry["run_id"]: entry for entry in old["runs"]}
    tuples = load_tuples(root / "data/tuples", 30)
    propositions = {digest(proposition_text(q).encode()): proposition_text(q)
                    for row in tuples for q in row["questions"].values()}
    target_tuples = [{"id": row["id"], "check_type": row["check_type"], "propositions": {
        role: digest(proposition_text(row["questions"][role]).encode())
        for role in CHECK_FIELDS[row["check_type"]]}} for row in tuples]
    proposition_sha = write_jsonl(destination / "propositions.jsonl",
        [{"id": ident, "text": text} for ident, text in sorted(propositions.items())])
    tuple_sha = write_jsonl(destination / "tuples.jsonl", target_tuples)
    entries = []
    paths = sorted(set((root / "results").glob("checks_*.jsonl")) |
                   set((root / "results").glob("sweep30_*.jsonl")) |
                   set((root / "results").glob("elicit*.jsonl")))
    for path in paths:
        if path.name.startswith("sweep30_native_"):
            continue  # native scores are derived from the exact receipt snapshot below
        by_model = defaultdict(list)
        for raw in read_jsonl(path):
            by_model[raw["model"]].append({key: value for key, value in raw.items() if key in LEGACY_FIELDS})
        for model, rows in sorted(by_model.items()):
            run_id = f"legacy/{path.stem}/{digest(model.encode())[:12]}"
            filename = f"{path.stem}-{digest(model.encode())[:12]}.jsonl"
            kind = "elicit" if path.name.startswith("elicit") else "checks"
            validate_scores(rows, kind)
            timestamp = previous.get(run_id, {}).get("recorded_at", utc(path.stat().st_mtime))
            entries.append({"run_id": run_id, "file": filename, "source_file": path.name,
                "model": model, "requested_model": model, "reasoning": "none-or-low (legacy fallback)",
                "provider": "mock" if model == "mock/sha256" else "fireworks", "kind": kind,
                "transport": "mock" if model == "mock/sha256" else "fireworks-api",
                "recorded_at": timestamp, "timestamp_source": "legacy_file_mtime_frozen_at_import",
                "sha256": write_jsonl(destination / filename, rows),
                "target_tuples": len(tuples) if kind == "checks" else None,
                "target_propositions": None, "status": "historical", "coverage": {"scored_tuples": len(rows)}})
    # Preserve imported history when results/ is no longer present on this machine.
    entries.extend(entry for ident, entry in previous.items()
                   if (ident.startswith("legacy/") or entry.get("transport") == "typesafe-systemone-api")
                   and ident not in {entry["run_id"] for entry in entries})
    campaign_ids = sorted(set(campaigns or []) | {CAMPAIGN} | {
        entry["run_id"].split("/")[0] for entry in previous.values()
        if entry.get("transport") == "subscription-native-cli"})
    campaign_metadata, campaign_scopes, scope_entries = {}, {}, {}
    for campaign in campaign_ids:
        metadata_path = root / "runs" / campaign / "campaign.json"
        metadata = (json.loads(metadata_path.read_text()) if metadata_path.exists()
                    else old.get("native_campaigns", {}).get(campaign))
        if metadata is None:
            if campaign != CAMPAIGN:
                raise ValueError(f"missing protocol metadata for campaign: {campaign}")
            metadata = {"campaign": campaign, "protocol_id": "numeric-v1", "category": "benchmark",
                        **PROTOCOLS["numeric-v1"]}
        if metadata.get("category") != "benchmark":
            raise ValueError(f"diagnostic campaign cannot enter benchmark scores: {campaign}")
        protocol = metadata["protocol_id"]
        if protocol not in PROTOCOLS or any(metadata.get(k) != v for k, v in PROTOCOLS[protocol].items()):
            raise ValueError(f"unrecognized or altered protocol: {campaign}")
        selected_tuples, selected_propositions = tuples, set(propositions)
        if metadata.get("scope_id"):
            scope_path = root / "runs" / campaign / metadata["scope_manifest_file"]
            if not scope_path.exists():
                scope_path = destination / old["scopes"][metadata["scope_id"]]["file"]
            scope = validate_scope(scope_path, tuples, root / "data/tuples")
            if scope["sha256"] != metadata["scope_sha256"] or scope["manifest"]["scope_id"] != metadata["scope_id"]:
                raise ValueError(f"campaign scope identity mismatch: {campaign}")
            selected_tuples, selected_propositions = scope["tuples"], scope["proposition_ids"]
            campaign_scopes[campaign] = scope
            filename = f"scope-{scope['sha256'][:12]}.json"
            (destination / filename).write_bytes(scope["bytes"])
            scope_entries[metadata["scope_id"]] = {"file": filename, "sha256": scope["sha256"],
                "target_tuples": len(selected_tuples), "target_propositions": len(selected_propositions)}
        for field, expected in {
            "target_tuples": len(selected_tuples), "target_propositions": len(selected_propositions),
            "tuple_ids_sha256": digest(json.dumps(sorted(row["id"] for row in selected_tuples)).encode()),
            "proposition_ids_sha256": digest(json.dumps(sorted(selected_propositions)).encode()),
        }.items():
            if field in metadata and metadata[field] != expected:
                raise ValueError(f"campaign cohort mismatch ({field}): {campaign}")
        campaign_metadata[campaign] = metadata
    configurations = [(campaign, config) for campaign in campaign_ids for config in NATIVE_CONFIGS]
    for campaign, (run_key, provider, model, reasoning) in configurations:
        protocol = campaign_metadata[campaign]["protocol_id"]
        protocol_settings = PROTOCOLS[protocol]
        scope = campaign_scopes.get(campaign)
        selected_propositions = scope["proposition_ids"] if scope else set(propositions)
        selected_tuple_ids = {row["id"] for row in scope["tuples"]} if scope else {row["id"] for row in tuples}
        selected_target_tuples = [row for row in target_tuples if row["id"] in selected_tuple_ids]
        run_id = f"{campaign}/{run_key}"
        directory = root / "runs" / campaign / run_key
        if not directory.exists() and run_id in previous:
            entries.append(previous[run_id])
            continue
        receipts = []
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                print(f"Warning: receipt is being written, omitted {path.name}", file=sys.stderr)
                continue
            if raw.get("id") not in selected_propositions:
                if scope:
                    raise ValueError(f"out-of-scope receipt: {campaign}/{path.name}")
                continue
            if digest(raw["proposition"].encode()) != raw["id"]:
                raise ValueError(f"receipt proposition hash mismatch: {path.name}")
            if raw.get("prompt") != protocol_settings["verbal_prompt"].format(proposition=raw["proposition"]):
                raise ValueError(f"receipt prompt mismatch: {path.name}")
            if (raw.get("campaign", CAMPAIGN) != campaign or raw.get("protocol_id", "numeric-v1") != protocol
                    or raw.get("system_instruction_sha256", digest(RULE.encode())) != digest(protocol_settings["system_instruction"].encode())):
                raise ValueError(f"receipt campaign/protocol mismatch: {campaign}/{path.name}")
            if finite_probability(raw.get("probability")):
                if parse_probability(raw.get("response", "")) != raw["probability"]:
                    raise ValueError(f"receipt probability disagrees with native response: {path.name}")
            if raw.get("requested_model", model) != model or raw.get("reasoning", reasoning) != reasoning:
                raise ValueError(f"receipt model/reasoning mismatch: {run_key}/{path.name}")
            if raw.get("imported_from"):
                origin = raw["imported_from"]
                original = root / "runs" / origin["campaign"] / origin["run_key"] / (origin["receipt_id"] + ".json")
                if original.exists() and digest(original.read_bytes()) != origin["receipt_sha256"]:
                    raise ValueError(f"import source receipt changed: {campaign}/{path.name}")
                if origin["receipt_id"] != raw["id"]:
                    raise ValueError(f"import proposition identity mismatch: {campaign}/{path.name}")
            receipts.append(sanitize_receipt(raw, path, model, reasoning))
        current = {row["id"]: row for row in receipts if row["is_current"]}
        identity = f"native/{model}@{reasoning}"
        rows = []
        for row in selected_target_tuples:
            ps = {role: current.get(ident, {}).get("probability") for role, ident in row["propositions"].items()}
            if all(value is not None for value in ps.values()):
                rows.append({"id": row["id"], "check_type": row["check_type"], "probabilities": ps,
                    "violation": violation(row["check_type"], ps), "model": identity,
                    "reasoning": reasoning, "run_id": run_id, "transport": "subscription-native-cli"})
        valid = sum(row.get("probability") is not None for row in current.values())
        completed = sum(bool(row["completed"]) for row in current.values())
        status = "complete" if valid == len(selected_propositions) else "completed-with-missing" if completed == len(selected_propositions) else "partial" if receipts else "pending"
        status_path = root / "runs" / campaign / f"status-{run_key}.json"
        if status_path.exists():
            try:
                status_record = json.loads(status_path.read_text())
                if status_record.get("status") == "unsupported":
                    status = "unsupported"
            except json.JSONDecodeError:
                pass
        timestamps = [r.get("imported_at", r["finished_at"]) for r in receipts if r.get("finished_at")]
        filename = f"{campaign}-{run_key}.jsonl"
        receipt_filename = f"{campaign}-{run_key}.receipts.jsonl"
        entries.append({"run_id": run_id, "file": filename,
            "source_file": (f"sweep30_native_{run_key}_20260912.jsonl" if campaign == CAMPAIGN
                            else f"native_{campaign}_{run_key}.jsonl"), "model": identity,
            "campaign": campaign, "protocol_id": protocol, "scope_id": scope["manifest"]["scope_id"] if scope else None,
            "requested_model": model, "reasoning": reasoning, "provider": provider,
            "kind": "checks", "transport": "subscription-native-cli",
            "recorded_at": max(timestamps) if timestamps else None,
            "timestamp_source": "native_receipt_finished_or_imported_at" if any(r.get("imported_from") for r in receipts) else "native_receipt_finished_at", "status": status,
            "sha256": write_jsonl(destination / filename, rows),
            "receipts_file": receipt_filename, "receipts_sha256": write_jsonl(destination / receipt_filename, receipts),
            "target_tuples": len(selected_target_tuples), "target_propositions": len(selected_propositions),
            "source_target_tuples": len(tuples), "source_target_propositions": len(propositions),
            "coverage": {"scored_tuples": len(rows), "valid_propositions": valid,
                "observed_propositions": len(current), "completed_propositions": completed,
                "missing_probabilities": len(selected_propositions) - valid,
                "attempts": sum(not r.get("imported_from") for r in receipts), "receipt_records": len(receipts),
                "imported_observations": sum(bool(r.get("imported_from")) for r in current.values()),
                "excluded_tuples": len(tuples) - len(selected_target_tuples),
                "outside_scope_propositions": len(propositions) - len(selected_propositions),
                "errors": dict(sorted(Counter(row["error_kind"] for row in current.values() if row.get("error_kind")).items()))}})
    manifest = {"schema_version": 2, "campaign": campaign_ids[-1],
        "native_campaigns": campaign_metadata, "native_protocols": PROTOCOLS, "scopes": scope_entries,
        "comparison_scope_id": campaign_metadata[campaign_ids[-1]].get("scope_id"),
        "target_tuples": len(tuples), "target_propositions": len(propositions),
        "propositions_file": "propositions.jsonl", "propositions_sha256": proposition_sha,
        "tuples_file": "tuples.jsonl", "tuples_sha256": tuple_sha,
        "verbal_prompt": PROTOCOLS[campaign_metadata[campaign_ids[-1]]["protocol_id"]]["verbal_prompt"],
        "native_protocol": {"protocol_id": campaign_metadata[campaign_ids[-1]]["protocol_id"],
            "system_instruction": PROTOCOLS[campaign_metadata[campaign_ids[-1]]["protocol_id"]]["system_instruction"],
            "fresh_context_per_proposition": True, "tools_allowed": False,
            "parser": PROTOCOLS[campaign_metadata[campaign_ids[-1]]["protocol_id"]]["parser"],
            "codex": {"client_version": "0.154.0", "requested_service_tier": "default",
                "reasoning": "per-run setting", "completion_cap": "not exposed by this native command"},
            "grok": {"client_version": "1.0.13", "requested_temperature": 0,
                "requested_max_completion_tokens": 2048, "requested_reasoning": "low"},
            "glm": {"client_version": "OpenCode 1.18.29", "requested_temperature": 0,
                "requested_max_output_tokens": 2048, "reasoning": "native-default",
                "cap_observation": "native reasoning can exceed the requested output limit"}},
        "runs": sorted(entries, key=lambda entry: entry["run_id"])}
    if selected_policy is not None:
        manifest["comparison_policy"] = selected_policy
    usage_path = destination / "subscription-usage.json"
    if usage_path.exists():
        manifest["subscription_usage_file"] = usage_path.name
        manifest["subscription_usage_sha256"] = digest(usage_path.read_bytes())
    usage_reports = {}
    for path in sorted(destination.glob("subscription-usage-*.json")) + ([usage_path] if usage_path.exists() else []):
        report = json.loads(path.read_text())
        usage_reports[report["campaign"]] = {"file": path.name, "sha256": digest(path.read_bytes())}
    manifest["subscription_usage_reports"] = usage_reports
    audit_path = destination / "native-readout-audit.json"
    if audit_path.exists():
        manifest["readout_audit_file"] = audit_path.name
        manifest["readout_audit_sha256"] = digest(audit_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return manifest


def load_snapshot(directory=SNAPSHOTS):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("subscription_usage_file"):
        if digest((directory / manifest["subscription_usage_file"]).read_bytes()) != manifest["subscription_usage_sha256"]:
            raise ValueError("snapshot hash mismatch: subscription usage report")
    if manifest.get("readout_audit_file"):
        if digest((directory / manifest["readout_audit_file"]).read_bytes()) != manifest["readout_audit_sha256"]:
            raise ValueError("snapshot hash mismatch: native readout audit")
    for scope_id, entry in manifest.get("scopes", {}).items():
        if digest((directory / entry["file"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"snapshot hash mismatch: reviewed scope {scope_id}")
    for campaign, entry in manifest.get("subscription_usage_reports", {}).items():
        if digest((directory / entry["file"]).read_bytes()) != entry["sha256"]:
            raise ValueError(f"snapshot hash mismatch: subscription usage report for {campaign}")
    def verified(filename, sha):
        path = directory / filename
        if digest(path.read_bytes()) != sha:
            raise ValueError(f"snapshot hash mismatch: {filename}")
        return read_jsonl(path)
    propositions = verified(manifest["propositions_file"], manifest["propositions_sha256"])
    tuples = verified(manifest["tuples_file"], manifest["tuples_sha256"])
    runs = []
    for entry in manifest["runs"]:
        rows = verified(entry["file"], entry["sha256"])
        validate_scores(rows, entry["kind"])
        receipts = verified(entry["receipts_file"], entry["receipts_sha256"]) if entry.get("receipts_file") else []
        stat_key = "violation" if entry["kind"] == "checks" else "abs_diff"
        runs.append({**entry, "rows": rows, "receipts": receipts, "n": len(rows),
            "stat": sum(row[stat_key] for row in rows) / len(rows) if rows else None})
    return manifest, propositions, tuples, runs


def comparison_context(manifest, runs, directory=SNAPSHOTS):
    """Apply the export manifest's participation policy before intersecting scores.

    ``all-settings`` preserves the original behavior when no policy is declared.
    ``complete-historical-plus-native`` admits every native setting and only
    historical runs that scored their original target, before scope filtering.
    Incomplete archives remain visible with no comparison score.
    """
    scope_id = manifest.get("comparison_scope_id")
    if not scope_id:
        return None
    scope_entry = manifest["scopes"][scope_id]
    scope = json.loads((directory / scope_entry["file"]).read_text())
    policy = manifest.get("comparison_policy", "all-settings")
    return build_comparison_context(scope_id, scope, runs, policy)


def build_comparison_context(scope_id, scope, runs, policy="all-settings"):
    """Compute a common cohort without changing or writing a scope manifest."""
    if policy not in COMPARISON_POLICIES:
        raise ValueError(f"Unknown comparison policy: {policy}")
    target_ids = set(scope["included_tuple_ids"])
    proposition_ids = set(scope["included_proposition_ids"])
    latest = {}
    for run in runs:
        if run["kind"] != "checks" or run.get("provider") == "mock" or run["model"] == "mock/sha256":
            continue
        model = run["model"]
        if model not in latest or (run.get("recorded_at") or "", run["file"]) > (latest[model].get("recorded_at") or "", latest[model]["file"]):
            latest[model] = run
    participating, unranked = {}, {}
    for model, run in latest.items():
        native = run.get("transport") == "subscription-native-cli"
        original_target = run.get("target_tuples", 0) or 0
        historical_complete = original_target > 0 and len({row["id"] for row in run["rows"]}) == original_target
        if policy == "all-settings" or native or historical_complete:
            participating[model] = run
        else:
            unranked[model] = "incomplete historical coverage"
    common_ids = set(target_ids)
    for run in participating.values():
        common_ids.intersection_update(row["id"] for row in run["rows"])
    if not participating:
        common_ids.clear()
    def filtered(run, ids):
        rows = [row for row in run["rows"] if row["id"] in ids]
        coverage = {**run.get("coverage", {}), "scored_tuples": len(rows)}
        if run.get("transport") == "typesafe-systemone-api":
            scoped_receipts = [r for r in run.get("receipts", []) if r["id"] in proposition_ids and r["is_current"]]
            coverage.update(valid_propositions=sum(finite_probability(r.get("probability")) for r in scoped_receipts),
                            completed_propositions=sum(r["completed"] for r in scoped_receipts))
        return {**run, "rows": rows, "n": len(rows),
                "stat": sum(row["violation"] for row in rows) / len(rows) if rows else None,
                "unscoped_n": run.get("n", len(run["rows"])), "unscoped_stat": run.get("stat"),
                "original_target_tuples": run.get("target_tuples"),
                "target_tuples": len(ids), "target_propositions": len(proposition_ids), "coverage": coverage}
    comparison_checks = {}
    for model, run in latest.items():
        eligible = model in participating
        comparison_checks[model] = {**filtered(run, common_ids if eligible else set()),
                                   "target_tuples": len(common_ids), "comparison_eligible": eligible,
                                   "unranked_reason": unranked.get(model)}
    return {"scope_id": scope_id, "scope": scope, "policy": policy, "target_tuple_ids": target_ids,
            "target_proposition_ids": proposition_ids, "common_tuple_ids": common_ids,
            "latest_checks": latest, "participating_checks": participating, "unranked_models": unranked,
            "scoped_checks": {model: filtered(run, target_ids) for model, run in latest.items()},
            "comparison_checks": comparison_checks}


def rebuild_database(directory=SNAPSHOTS, output=ROOT / "data/benchmark.sqlite"):
    manifest, propositions, tuples, runs = load_snapshot(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    conn = sqlite3.connect(temporary)
    conn.executescript('''
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE models (model TEXT PRIMARY KEY, requested_model TEXT NOT NULL,
            reasoning TEXT NOT NULL, provider TEXT NOT NULL, transport TEXT NOT NULL);
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, model TEXT NOT NULL REFERENCES models,
            kind TEXT NOT NULL, source_file TEXT NOT NULL, snapshot_file TEXT NOT NULL, protocol_id TEXT NOT NULL,
            scope_id TEXT REFERENCES scopes, recorded_at TEXT,
            timestamp_source TEXT NOT NULL, status TEXT NOT NULL, target_tuples INTEGER,
            target_propositions INTEGER, scored_count INTEGER NOT NULL, mean_violation REAL,
            mean_elicitation_gap REAL, coverage_json TEXT NOT NULL, snapshot_sha256 TEXT NOT NULL);
        CREATE TABLE propositions (id TEXT PRIMARY KEY, text TEXT NOT NULL);
        CREATE TABLE tuple_propositions (tuple_id TEXT NOT NULL, check_type TEXT NOT NULL,
            role TEXT NOT NULL, proposition_id TEXT NOT NULL REFERENCES propositions,
            PRIMARY KEY(tuple_id, role));
        CREATE TABLE receipts (run_id TEXT NOT NULL REFERENCES runs, attempt_id TEXT NOT NULL,
            proposition_id TEXT NOT NULL REFERENCES propositions, is_current INTEGER NOT NULL,
            completed INTEGER NOT NULL, is_imported INTEGER NOT NULL, probability REAL CHECK(probability BETWEEN 0 AND 1),
            error_kind TEXT, response TEXT, started_at TEXT, finished_at TEXT, elapsed_seconds REAL,
            usage_json TEXT, model_usage_json TEXT, receipt_json TEXT NOT NULL,
            PRIMARY KEY(run_id, attempt_id));
        CREATE TABLE tuple_scores (run_id TEXT NOT NULL REFERENCES runs, tuple_id TEXT NOT NULL,
            check_type TEXT NOT NULL, violation REAL NOT NULL, probabilities_json TEXT NOT NULL,
            PRIMARY KEY(run_id, tuple_id));
        CREATE TABLE elicitation (run_id TEXT NOT NULL REFERENCES runs, proposition_id TEXT NOT NULL,
            title TEXT NOT NULL, verbalized_probability REAL NOT NULL, logprob_probability REAL NOT NULL,
            abs_diff REAL NOT NULL, PRIMARY KEY(run_id, proposition_id));
        CREATE TABLE subscription_usage (campaign TEXT NOT NULL, provider TEXT NOT NULL, window TEXT, report_json TEXT NOT NULL);
        CREATE TABLE scopes (scope_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL, manifest_json TEXT NOT NULL);
        CREATE TABLE scope_exclusions (scope_id TEXT NOT NULL REFERENCES scopes, tuple_id TEXT NOT NULL,
            reason TEXT NOT NULL, PRIMARY KEY(scope_id, tuple_id));
        CREATE TABLE comparison_tuples (scope_id TEXT NOT NULL REFERENCES scopes, tuple_id TEXT NOT NULL,
            check_type TEXT NOT NULL, PRIMARY KEY(scope_id, tuple_id));
        CREATE TABLE comparison_settings (scope_id TEXT NOT NULL REFERENCES scopes, run_id TEXT NOT NULL REFERENCES runs,
            comparison_eligible INTEGER NOT NULL CHECK(comparison_eligible IN (0,1)),
            unranked_reason TEXT, reviewed_scored_count INTEGER NOT NULL, PRIMARY KEY(scope_id, run_id));
        CREATE VIEW current_probabilities AS SELECT r.run_id, m.model, m.requested_model, m.reasoning,
            s.protocol_id, s.scope_id, r.proposition_id, r.probability, r.completed, r.is_imported, r.error_kind FROM receipts r
            JOIN runs s USING(run_id) JOIN models m USING(model) WHERE r.is_current=1;
        CREATE VIEW ranking AS SELECT r.*, m.requested_model, m.reasoning, m.transport,
            CASE WHEN r.target_tuples > 0 AND r.scored_count = r.target_tuples THEN 1 ELSE 0 END AS headline_eligible
            FROM runs r JOIN models m USING(model) WHERE kind='checks' AND m.provider != 'mock'
            AND r.run_id=(SELECT candidate.run_id FROM runs candidate
                WHERE candidate.model=r.model AND candidate.kind=r.kind
                ORDER BY coalesce(candidate.recorded_at, '') DESC, candidate.snapshot_file DESC LIMIT 1);
        CREATE VIEW comparison_ranking AS SELECT c.scope_id, r.run_id, r.model, r.requested_model,
            r.reasoning, r.protocol_id, r.transport, c.comparison_eligible, c.unranked_reason,
            c.reviewed_scored_count, r.scored_count AS original_scored_count, r.target_tuples AS original_target_tuples,
            (SELECT count(*) FROM comparison_tuples WHERE scope_id=c.scope_id) AS comparison_target_tuples,
            CASE WHEN c.comparison_eligible=1 THEN count(t.tuple_id) ELSE NULL END AS scored_count,
            CASE WHEN c.comparison_eligible=1 THEN avg(t.violation) ELSE NULL END AS mean_violation
            FROM comparison_settings c JOIN ranking r USING(run_id)
            LEFT JOIN tuple_scores t ON t.run_id=r.run_id AND c.comparison_eligible=1
                AND t.tuple_id IN (SELECT tuple_id FROM comparison_tuples WHERE scope_id=c.scope_id)
            GROUP BY c.scope_id, r.run_id;
    ''')
    conn.execute("INSERT INTO metadata VALUES (?,?)", ("manifest_sha256", digest((directory / "manifest.json").read_bytes())))
    conn.execute("INSERT INTO metadata VALUES (?,?)", ("schema_version", str(manifest["schema_version"])))
    if manifest.get("native_protocol"):
        conn.execute("INSERT INTO metadata VALUES (?,?)", ("native_protocol", encoded(manifest["native_protocol"])))
    for key in ("native_protocols", "native_campaigns"):
        if manifest.get(key):
            conn.execute("INSERT INTO metadata VALUES (?,?)", (key, encoded(manifest[key])))
    for scope_id, entry in manifest.get("scopes", {}).items():
        scope = json.loads((directory / entry["file"]).read_text())
        conn.execute("INSERT INTO scopes VALUES (?,?,?)", (scope_id, entry["sha256"], encoded(scope)))
        conn.executemany("INSERT INTO scope_exclusions VALUES (?,?,?)",
            [(scope_id, row["tuple_id"], row["reason"]) for row in scope["exclusions"]])
    comparison = comparison_context(manifest, runs, directory)
    if comparison:
        types = {row["id"]: row["check_type"] for row in tuples}
        conn.executemany("INSERT INTO comparison_tuples VALUES (?,?,?)",
            [(comparison["scope_id"], ident, types[ident]) for ident in sorted(comparison["common_tuple_ids"])])
        summary = {"scope_id": comparison["scope_id"], "reviewed_target_tuples": len(comparison["target_tuple_ids"]),
                   "comparison_policy": comparison["policy"],
                   "common_scored_tuples": len(comparison["common_tuple_ids"]),
                   "model_settings": len(comparison["latest_checks"]),
                   "participating_model_settings": len(comparison["participating_checks"]),
                   "participating_models": sorted(comparison["participating_checks"]),
                   "unranked_models": comparison["unranked_models"],
                   "selection_effect": "Only tuples scored by every participating setting are compared; missing readouts can shrink this intersection. Participation follows the declared scope policy and original historical coverage, not response values.",
                   "per_setting_scoped_counts": {model: row["n"] for model, row in comparison["scoped_checks"].items()}}
        conn.execute("INSERT INTO metadata VALUES (?,?)", ("comparison_scope", encoded(summary)))
    if manifest.get("readout_audit_file"):
        audit = json.loads((directory / manifest["readout_audit_file"]).read_text())
        conn.execute("INSERT INTO metadata VALUES (?,?)", ("native_readout_audit", encoded(audit)))
    conn.executemany("INSERT INTO propositions VALUES (?,?)", [(r["id"], r["text"]) for r in propositions])
    for row in tuples:
        conn.executemany("INSERT INTO tuple_propositions VALUES (?,?,?,?)",
            [(row["id"], row["check_type"], role, ident) for role, ident in sorted(row["propositions"].items())])
    for run in runs:
        conn.execute("INSERT OR IGNORE INTO models VALUES (?,?,?,?,?)",
            tuple(run[key] for key in ("model", "requested_model", "reasoning", "provider", "transport")))
        conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            run["run_id"], run["model"], run["kind"], run["source_file"], run["file"], run.get("protocol_id", "legacy-unspecified"), run.get("scope_id"), run["recorded_at"],
            run["timestamp_source"], run["status"], run["target_tuples"], run["target_propositions"],
            run["n"], run["stat"] if run["kind"] == "checks" else None,
            run["stat"] if run["kind"] == "elicit" else None, encoded(run["coverage"]), run["sha256"]))
        for receipt in run["receipts"]:
            conn.execute("INSERT INTO receipts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                run["run_id"], receipt["attempt_id"], receipt["id"], receipt["is_current"], receipt["completed"], bool(receipt.get("imported_from")),
                receipt.get("probability"), receipt.get("error_kind"), receipt.get("response"),
                receipt.get("started_at"), receipt.get("finished_at"), receipt.get("elapsed_seconds"),
                encoded(receipt.get("usage")), encoded(receipt.get("model_usage")), encoded(receipt)))
        for row in run["rows"]:
            if run["kind"] == "checks":
                conn.execute("INSERT INTO tuple_scores VALUES (?,?,?,?,?)", (run["run_id"], row["id"],
                    row["check_type"], row["violation"], encoded(row["probabilities"])))
            else:
                conn.execute("INSERT INTO elicitation VALUES (?,?,?,?,?,?)", (run["run_id"], row["id"],
                    row["title"], row["verbalized_probability"], row["logprob_probability"], row["abs_diff"]))
    if comparison:
        conn.executemany("INSERT INTO comparison_settings VALUES (?,?,?,?,?)", [
            (comparison["scope_id"], run["run_id"], run["comparison_eligible"], run["unranked_reason"],
             comparison["scoped_checks"][model]["n"])
            for model, run in comparison["comparison_checks"].items()])
    usage_files = {entry["file"] for entry in manifest.get("subscription_usage_reports", {}).values()}
    usage_path = directory / "subscription-usage.json"
    if usage_path.exists():
        usage_files.add(usage_path.name)
    for filename in sorted(usage_files):
        report = json.loads((directory / filename).read_text())
        if filename == "subscription-usage.json":
            conn.execute("INSERT INTO metadata VALUES (?,?)", ("subscription_usage", encoded(report)))
        conn.execute("INSERT INTO metadata VALUES (?,?)", ("subscription_usage/" + report["campaign"], encoded(report)))
        for entry in report.get("providers", []):
            conn.execute("INSERT INTO subscription_usage VALUES (?,?,?,?)",
                (report["campaign"], entry["provider"], entry.get("window"), encoded(entry)))
    conn.commit()
    # This is a fresh database; VACUUM would duplicate it in the system temp
    # filesystem and is unnecessary for a deterministic rebuild.
    conn.close()
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="snapshot local runtime results and sanitized receipts")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--campaign", action="append", default=[], help="additional full campaign to import; diagnostics are rejected")
    parser.add_argument("--comparison-policy", choices=COMPARISON_POLICIES,
                        help="comparison participants for --refresh; preserves the prior choice when omitted")
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS)
    parser.add_argument("--database", type=Path, default=ROOT / "data/benchmark.sqlite")
    args = parser.parse_args()
    if args.comparison_policy is not None and not args.refresh:
        parser.error("--comparison-policy requires --refresh")
    if args.refresh:
        refresh(args.root, args.snapshots, args.campaign, comparison_policy=args.comparison_policy)
    database = rebuild_database(args.snapshots, args.database)
    _, _, _, runs = load_snapshot(args.snapshots)
    print(f"Wrote {database}: {len(runs)} runs, {sum(r['n'] for r in runs)} scored observations, "
          f"{sum(len(r['receipts']) for r in runs)} native receipts")


if __name__ == "__main__":
    main()
