#!/usr/bin/env python3
"""Offline, fail-closed export of the nine native factual-recall treatments.

Default publication requires every treatment and 129 terminal observations each.
--allow-incomplete is only for explicitly marked local previews. No provider is
called, no private run file is written, and no original ranking page is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness import run_native_probes as native
from harness import run_probes as probes

CATALOG_COUNT = 129
SETTINGS = [(p, e) for p, efforts in native.EFFORTS.items() for e in efforts]
TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens",
              "reasoning_output_tokens", "total_tokens")
USAGE_KEYS = set(TOKEN_KEYS) | {"input", "output", "reasoning", "total", "cache",
    "read", "write", "cache_write_input_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens", "reasoning_tokens", "inputTokens", "outputTokens",
    "cacheCreationInputTokens", "cacheReadInputTokens", "modelCalls", "costUSD"}
INVENTORY_KEYS = ("total_probes", "original_url_rows", "original_distinct_urls", "original_uncited_rows",
    "month_precision_dates", "day_precision_dates", "literal_duplicate_questions", "flagged_probes",
    "probes_with_date_flags", "probes_with_prompt_flags", "probes_with_grading_flags", "proposed_corrections",
    "reference_identities_with_wording_qualification")
REPO = "https://github.com/futarchy-fi/consistency-ft"
DOCS = REPO + "/blob/main/docs/"
DISCLAIMER = (
    "These are ORIGINAL-KEY matches against the frozen answer/alias catalog, not an accuracy estimate. "
    "Source review corroborates core answers only; it does not validate every premise or earliest public availability. "
    "Proposed corrections have not been applied. UNKNOWN and unmatched responses do not establish a knowledge cutoff "
    "or an uncontaminated evaluation window. This factual-recall study is separate from the consistency ranking."
)


def pick(value, keys):
    return {key: value[key] for key in keys if key in value}


def safe_usage(value):
    """Preserve named numerical usage evidence; unknown provider metadata stays private."""
    if value is None:
        return None
    if isinstance(value, list):
        return [safe_usage(v) for v in value]
    if isinstance(value, dict):
        return {k: safe_usage(v) for k, v in value.items() if k in USAGE_KEYS
                and (isinstance(v, (dict, list)) or type(v) in (int, float))}
    return value if type(value) in (int, float) else None


def local_file(directory, relative):
    candidate = (directory / relative).resolve()
    if not candidate.is_relative_to(directory.resolve()):
        raise ValueError("Private evidence path escapes its campaign directory")
    return candidate


def validate_command(receipt, manifest):
    command = receipt["native_evidence"]["command"]
    if not isinstance(command, list) or any(not isinstance(v, str) for v in command):
        raise ValueError("Native command evidence is invalid")
    def args(flag):
        return [command[i + 1] for i, x in enumerate(command[:-1]) if x == flag]
    provider = manifest["provider"]
    question = next(e["probe"]["question"] for e in manifest["probes"] if e["probe_id"] == receipt["probe_id"])
    prompt = probes.PROBE_PROMPT.format(question=question)
    expected = "zai-coding-plan/glm-5.3" if provider == "glm" else manifest["model"]
    if args("--model") != [expected]:
        raise ValueError("Native command model selection differs from treatment")
    if provider in {"astra", "luna"}:
        efforts = [v for v in args("-c") if v.startswith("model_reasoning_effort=")]
        rules = [v for v in args("-c") if v.startswith("developer_instructions=")]
        if efforts != ["model_reasoning_effort=" + json.dumps(manifest["reasoning"])]:
            raise ValueError("Native command reasoning selection differs from treatment")
        if rules != ["developer_instructions=" + json.dumps(native.SYSTEM_INSTRUCTION)]:
            raise ValueError("Native command system instruction differs from protocol")
    elif provider == "grok":
        if args("--reasoning-effort") != [manifest["reasoning"]] or args("--system-prompt-override") != [native.SYSTEM_INSTRUCTION]:
            raise ValueError("Native command reasoning or system instruction differs from protocol")
        if args("-p") != [prompt]:
            raise ValueError("Native command question differs from the frozen prompt")
    elif args("--agent") != ["factual-probe"]:
        raise ValueError("GLM native command lacks the frozen factual agent")
    elif command[-1] != prompt:
        raise ValueError("Native command question differs from the frozen prompt")


def sanitized_attempt(envelope, number, metadata):
    receipt = envelope["receipt"]
    result = pick(receipt, ("status", "completed", "response", "requested_model", "model", "provider",
        "reasoning", "protocol_id", "prompt_sha256", "collector", "system_instruction_sha256",
        "native_protocol_fingerprint", "started_at", "finished_at", "elapsed_seconds", "exit_code",
        "usage", "usage_validation_issue", "tools_used", "truncated", "provider_model_mismatch"))
    result.update(attempt_number=number, private_receipt_sha256=envelope["receipt_sha256"],
                  private_attempt_file_sha256=envelope["import_file_sha256"])
    result["provenance"] = pick(receipt["provenance"], ("transport", "adapter"))
    result["native_validation"] = pick(receipt["native_validation"], (
        "session_present", "native_completed", "final_message_matches", "recovered_reconnections",
        "reported_models", "no_tools_reported", "validation_errors"))
    aliases = metadata["reported_model_aliases"]
    reported = receipt["native_validation"].get("reported_models", [])
    result["native_validation"]["reported_models"] = [m for m in reported if m in aliases]
    result["native_validation"]["unrecognized_reported_model_sha256"] = [probes.sha(m) for m in reported if m not in aliases]
    result["raw_usage"] = safe_usage(receipt.get("raw_usage"))
    model_usage = receipt.get("raw_model_usage")
    result["raw_model_usage"] = ({k: safe_usage(v) for k, v in model_usage.items() if k in aliases}
                                  if isinstance(model_usage, dict) else None)
    result["private_raw_usage_sha256"] = probes.sha(probes.canonical(receipt.get("raw_usage")))
    result["private_raw_model_usage_sha256"] = probes.sha(probes.canonical(model_usage))
    return result


def validate_attempt(directory, launch, envelope, manifest, metadata):
    receipt = envelope["receipt"]
    for key in ("probe_id", "model", "provider", "reasoning", "prompt_sha256", "started_at",
                "native_protocol_fingerprint"):
        if launch.get(key) != receipt.get(key):
            raise ValueError("Launch and receipt binding mismatch: " + key)
    if receipt["provenance"].get("local_attempt_id") != launch["attempt_id"]:
        raise ValueError("Launch and local attempt binding mismatch")
    if (receipt.get("requested_model") != manifest["model"] or
        receipt["provenance"].get("transport") != "subscription-native-cli" or
        receipt["provenance"].get("adapter") != native.COLLECTOR):
        raise ValueError("Receipt lacks exact native subscription treatment provenance")
    path = local_file(directory, launch["receipt_file"])
    if probes.sha(path.read_bytes()) != envelope["import_file_sha256"]:
        raise ValueError("Private attempt file hash mismatch")
    if native.validate_saved(path, manifest, metadata) != receipt:
        raise ValueError("Private attempt and ledger receipt differ")
    evidence = receipt.get("native_evidence", {})
    if not isinstance(evidence.get("stdout"), str):
        raise ValueError("Raw native output evidence is absent")
    decoded = native.validate_output(manifest["provider"], evidence["stdout"],
                                    local_file(directory, launch["final_file"]), receipt["exit_code"])
    for key, value in decoded.items():
        if receipt.get(key) != value:
            raise ValueError("Native evidence recomputation mismatch: " + key)
    usage, issue = native.normalize_usage(manifest["provider"], decoded["raw_usage"])
    if receipt.get("usage") != usage or receipt.get("usage_validation_issue") != issue:
        raise ValueError("Native usage recomputation mismatch")
    errors = decoded["native_validation"]["validation_errors"]
    if receipt.get("completed") is not (not errors) or receipt["status"] != ("infrastructure_error" if errors else "completed"):
        raise ValueError("Native completion and receipt status disagree")
    mismatch = any(m not in metadata["reported_model_aliases"] for m in decoded["native_validation"]["reported_models"])
    if receipt.get("provider_model_mismatch") is not mismatch:
        raise ValueError("Reported native model mismatch was not preserved")
    validate_command(receipt, manifest)


def source_overlay(audit, catalog, catalog_hash):
    if audit.get("schema_version") != 1 or audit.get("catalog_sha256") != catalog_hash:
        raise ValueError("Source audit does not bind to the frozen catalog")
    rows = audit.get("probes", [])
    if len(rows) != CATALOG_COUNT:
        raise ValueError("Source audit must retain all 129 probes")
    for index, (row, original) in enumerate(zip(rows, catalog), 1):
        if row.get("catalog_index") != index or row.get("probe_id") != probes.sha(probes.canonical(original)) or row.get("original_probe") != original:
            raise ValueError("Source audit row differs from the frozen original catalog")
        if any(c.get("status") != "proposed_not_applied" for c in row.get("proposed_corrections", [])):
            raise ValueError("Source corrections must remain proposed and unapplied")
    # The audit is an independent source document, never a replacement answer key.
    return rows


def sanitized_source(row):
    result = pick(row, ("answer_evidence_level", "core_answer_status", "event_date_review",
                       "earliest_public_availability", "dependence_group_ids"))
    result["evidence"] = [pick(e, ("capture_id", "url", "relationship", "excerpt",
                                  "excerpt_text_normalization", "review_note")) for e in row.get("evidence", [])]
    for field in ("date_flags", "prompt_flags", "grading_flags"):
        result[field] = [pick(f, ("code", "detail")) for f in row.get(field, [])]
    result["proposed_corrections"] = [pick(c, ("field", "original_value", "proposed_value", "rationale",
        "evidence_capture_ids", "status")) for c in row.get("proposed_corrections", [])]
    result["original_source_review"] = pick(row.get("original_source_review", {}), ("status", "note", "capture_id", "http_status"))
    search = row.get("primary_source_search", {})
    result["primary_source_search"] = pick(search, ("strategy", "secondary_lookup_url"))
    result["primary_source_search"]["attempts"] = [pick(a, ("url", "capture_id", "http_status", "assessment"))
                                                    for a in search.get("attempts", [])]
    return result


def build_export(directory, audit_path, *, allow_incomplete=False, catalog_path=None):
    directory, audit_path = Path(directory), Path(audit_path)
    catalog_path = Path(catalog_path or ROOT / "data/cutoff_probes.jsonl")
    catalog = probes.read_jsonl(catalog_path)
    catalog_hash = probes.sha(catalog_path.read_bytes())
    if len(catalog) != CATALOG_COUNT:
        raise ValueError("Export requires the complete 129-probe catalog")
    audit = json.loads(audit_path.read_text())
    overlays = source_overlay(audit, catalog, catalog_hash)
    found, campaigns, common, configurations, sessions = {}, set(), None, [], set()
    for path in sorted(directory.iterdir()):
        if not path.is_dir() or not (path / "manifest.json").exists():
            continue
        manifest, metadata = native.load_campaign(path)
        identity = (manifest["provider"], manifest["reasoning"])
        if identity not in SETTINGS or identity in found or manifest["campaign_id"] in campaigns:
            raise ValueError("Expected exactly one campaign for each of the nine native treatments")
        found[identity] = path
        campaigns.add(manifest["campaign_id"])
        if (manifest["catalog_count"] != CATALOG_COUNT or manifest["full_catalog_preserved"] is not True or
            manifest["probes_file_sha256"] != catalog_hash or manifest["prompt_template"] != probes.PROBE_PROMPT or
            [e["probe"] for e in manifest["probes"]] != catalog or
            [e["probe_id"] for e in manifest["probes"]] != [probes.sha(probes.canonical(p)) for p in catalog]):
            raise ValueError("Native treatment does not preserve the same full catalog and prompt")
        if (manifest["cutoff_key"] != manifest["model"] or
            manifest["cutoff_claim"] != manifest["cutoff_catalog"].get(manifest["model"])):
            raise ValueError("A native treatment cannot inherit a sibling model's cutoff claim")
        shared = pick(manifest, ("as_of", "protocol_id", "prompt_template", "cutoffs_file_sha256", "cutoff_catalog"))
        shared.update(pick(metadata, ("collector", "system_instruction", "system_instruction_sha256", "prompt_template_sha256")))
        if common is not None and common != shared:
            raise ValueError("Native treatments do not share the same factual protocol and catalog metadata")
        common = shared
        expected_requests = [{"probe_id": entry["probe_id"], **pick(manifest, ("model", "provider", "reasoning", "protocol_id")),
            "prompt": probes.PROBE_PROMPT.format(question=entry["probe"]["question"]),
            "prompt_sha256": probes.sha(probes.PROBE_PROMPT.format(question=entry["probe"]["question"]))}
            for entry in manifest["probes"] if entry["review"]["request_ready"]]
        if probes.read_jsonl(path / "requests.jsonl") != expected_requests:
            raise ValueError("Saved native requests differ from the frozen factual prompts")
        report = probes.build_report(path)  # validates ledger hashes, ordering and first-terminal semantics
        ledger_path = path / "receipts.jsonl"
        saved = probes.read_jsonl(ledger_path) if ledger_path.exists() else []
        launches_path = path / "collection/launches.jsonl"
        launches = probes.read_jsonl(launches_path) if launches_path.exists() else []
        by_attempt = {}
        for launch in launches:
            if launch["attempt_id"] in by_attempt:
                raise ValueError("Duplicate launch reservation")
            by_attempt[launch["attempt_id"]] = launch
        histories, imported = defaultdict(list), set()
        for envelope in saved:
            receipt = envelope["receipt"]
            attempt = receipt["provenance"].get("local_attempt_id")
            if attempt not in by_attempt or attempt in imported:
                raise ValueError("Every saved receipt must bind to one unique launch")
            validate_attempt(path, by_attempt[attempt], envelope, manifest, metadata)
            imported.add(attempt)
            sid = receipt.get("session_id")
            if sid:
                session = ("codex" if manifest["provider"] in {"astra", "luna"} else manifest["provider"], sid)
                if session in sessions:
                    raise ValueError("Native session reused across attempts or treatments")
                sessions.add(session)
            history = histories[receipt["probe_id"]]
            history.append(sanitized_attempt(envelope, len(history) + 1, metadata))
        unresolved = len(launches) - len(imported)
        if not allow_incomplete and (not report["full_catalog_completed"] or unresolved):
            raise ValueError("Publication requires all 129 terminal observations and every launch reconciled")
        rows = []
        for index, (entry, grade_row, overlay) in enumerate(zip(manifest["probes"], report["probes"], overlays), 1):
            attempts = histories[entry["probe_id"]]
            rows.append({"catalog_index": index, "probe_id": entry["probe_id"], "original_probe": entry["probe"],
                         "original_key_grade": grade_row["grade"],
                         "response": attempts[-1]["response"] if attempts else None,
                         "source_review": sanitized_source(overlay),
                         "catalog_review_at_preparation": entry["review"], "attempts": attempts})
        configurations.append({"configuration": manifest["provider"] + "/" + manifest["reasoning"],
            **pick(report, ("model", "provider", "reasoning", "campaign_id", "as_of", "observed_through",
                           "catalog_count", "completed_observations", "attempt_receipts", "statuses", "by_quarter",
                           "usage_totals_including_failed_attempts", "attempts_with_usage_unavailable", "cutoff_claim")),
            "native_protocol": metadata, "launch_reservations": len(launches), "unreconciled_launches": unresolved,
            "full_catalog_completed": report["full_catalog_completed"] and not unresolved,
            "original_key_matches": report["statuses"].get("exact_alias_match", 0),
            "private_ledger_sha256": probes.sha(ledger_path.read_bytes()) if ledger_path.exists() else None,
            "private_launch_ledger_sha256": probes.sha(launches_path.read_bytes()) if launches_path.exists() else None,
            "probes": rows})
    if set(found) != set(SETTINGS):
        raise ValueError("All nine exact native treatments are required, including for local previews")
    configurations.sort(key=lambda c: SETTINGS.index((c["provider"], c["reasoning"])))
    total = sum(c["completed_observations"] for c in configurations)
    return {"schema_version": 1, "study": "native-factual-recall-20260913",
        "publication_status": "INCOMPLETE_LOCAL_PREVIEW" if allow_incomplete else "COMPLETE",
        "generated_at": datetime.now(timezone.utc).isoformat(), "catalog_count": CATALOG_COUNT,
        "configuration_count": len(configurations), "target_observations": CATALOG_COUNT * len(SETTINGS),
        "completed_observations": total, "catalog_sha256": catalog_hash,
        "source_audit_sha256": probes.sha(audit_path.read_bytes()),
        "source_audit_inventory": {**{k: v for k, v in audit.get("inventory", {}).items() if k in INVENTORY_KEYS and type(v) is int},
            "core_answer_evidence_levels": {k: v for k, v in audit.get("inventory", {}).get("core_answer_evidence_levels", {}).items()
                                             if k in {"primary", "secondary"} and type(v) is int}},
        "source_review_limitations": audit.get("limitations", []), "verified_cutoff": None,
        "grading": {"label": "ORIGINAL-KEY matches", "corrections_applied": False,
                    "method": "Recomputed from frozen original answers and aliases using harness/run_probes.py; no answer-based retries."},
        "protocol": common, "limitations": DISCLAIMER,
        "prompt_evidence_limit": "Saved request payloads and prompt hashes are checked for every setting. Grok/GLM command arguments also retain the factual prompt. Astra/Luna stdin is not separately archived: its original delivery is supported by frozen collector code and recorded hashes, not an independent raw stdin capture.",
        "usage_report": "data/research/native-probe-usage-20260913.json",
        "privacy": "Session/request IDs, account/profile names, local paths, commands, process IDs and raw stdout/stderr are omitted. Private receipt/file hashes preserve audit bindings; named numeric usage evidence is retained. Unexpected reported model names are hashed instead of published.",
        "configurations": configurations}


def md_cell(value):
    return escape(str(value)).replace("|", "\\|").replace("\n", "<br>")


def markdown(report):
    lines = ["# Native factual recall — 13 September 2026", "",
        f"**{report['publication_status']} · {report['completed_observations']}/{report['target_observations']} terminal observations · nine exact native settings.**", "", DISCLAIMER, "",
        "[Interactive full tables](https://consistency.futarchy.ai/factual-recall.html) · "
        "[Sanitized data](../data/research/native-probes-20260913.json) · "
        "[Subscription usage](../data/research/native-probe-usage-20260913.json) · "
        "[Source review](probe-source-review-20260913.md)", "",
        "## Procedure", "",
        "Each setting received the same full 129-question catalog, factual prompt and system instruction through its native subscription client. "
        "Each call used a fresh session with tools, browsing, memory and delegation prohibited. All topics were retained. "
        "UNKNOWN and unmatched replies remain terminal observations; only infrastructure failures may be retried. Every attempt is preserved.", "",
        "The exporter checks frozen treatment identities, catalog and protocol fingerprints, unique launches and sessions, "
        "private attempt/ledger hashes, raw native completion and response extraction, model selections, and recomputed usage. "
        "Grades are recomputed from the original key. Native metadata is provider evidence, not independent authentication of model internals.", "",
        report["prompt_evidence_limit"], "",
        "## Original-key results", "", "| Model | Reasoning | Terminal / 129 | ORIGINAL-KEY matches | UNKNOWN | Unmatched / review | Attempts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in report["configurations"]:
        lines.append(f"| {c['model']} | {c['reasoning']} | {c['completed_observations']} | {c['original_key_matches']} | {c['statuses'].get('explicit_unknown', 0)} | {c['statuses'].get('manual_review', 0)} | {c['attempt_receipts']} |")
    lines += ["", "Statuses outside the three displayed grade columns remain visible in each full table and in JSON. "
              "Token totals include failed attempts; missing counters are unknown, not zero. "
              "Subscription percentages are shared-account measurements, not exact per-experiment fractions. "
              "Backend costUSD metadata, when reported, is not a separately billed subscription charge.", "",
              "## Complete response tables", ""]
    for c in report["configurations"]:
        lines += [f"<details><summary>{c['model']} · {c['reasoning']} · all 129 probes</summary>", "",
                  "| # | Recorded event date | Question | Original answer / aliases | Full response | Original-key grade | Source flags |",
                  "| ---: | --- | --- | --- | --- | --- | --- |"]
        for r in c["probes"]:
            p, review = r["original_probe"], r["source_review"]
            flags = [f["code"] for k in ("date_flags", "prompt_flags", "grading_flags") for f in review.get(k, [])]
            values = [r["catalog_index"], p["event_date"], p["question"],
                      p["answer"] + " / " + ", ".join(p.get("accept", [])),
                      r["response"] if r["response"] is not None else "PENDING",
                      r["original_key_grade"]["status"], ", ".join(flags) or "none recorded"]
            lines.append("| " + " | ".join(md_cell(v) for v in values) + " |")
        lines += ["", "</details>", ""]
    lines += ["## Audit evidence", "", f"Catalog SHA-256: `{report['catalog_sha256']}`.", "",
              f"Source audit SHA-256: `{report['source_audit_sha256']}`.", "",
              "The JSON contains full attempt histories, original-key grades, source evidence/flags, unapplied proposed corrections, "
              "completion/no-tools/model metadata, timestamps, usage counters and private receipt hashes. " + report["privacy"], ""]
    return "\n".join(lines)


CSS = """
:root{color-scheme:dark;--bg:#0b1020;--panel:#131c31;--line:#29354b;--text:#edf2ff;--muted:#afbdd3;--mint:#63e1bd;--amber:#ffd38a;--blue:#85aefc}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top right,#19344a 0,transparent 40%),var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:42px 24px 72px}a{color:var(--mint);text-underline-offset:4px}h1{font-size:clamp(2.5rem,6vw,4.5rem);line-height:1.05;letter-spacing:-.05em;margin:18px 0}h2{font-size:1.65rem;letter-spacing:-.025em}p{max-width:86ch}.eyebrow{color:var(--mint);font-size:.8rem;text-transform:uppercase;letter-spacing:.15em}.muted,small{color:var(--muted)}.hero{display:grid;grid-template-columns:1fr 240px;gap:30px;align-items:center}.hero svg{width:100%;max-height:230px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0}.card,.panel{border:1px solid var(--line);border-radius:18px;background:#131c31dd;padding:22px}.card strong{display:block;font-size:2rem;line-height:1.2}.card span{color:var(--muted);font-size:.85rem}.notice{border-left:3px solid var(--amber);padding:10px 18px;background:#332b2038}.downloads{display:flex;gap:18px;flex-wrap:wrap}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.88rem}th{text-align:left;font-size:.75rem;letter-spacing:.035em;text-transform:uppercase;color:var(--muted);white-space:nowrap}th,td{border-bottom:1px solid var(--line);padding:12px;vertical-align:top}tbody tr:hover{background:#20314b66}.num{font-variant-numeric:tabular-nums;white-space:nowrap}.bar{display:flex;width:180px;height:9px;border-radius:20px;overflow:hidden;background:var(--line);margin-top:8px}.match{background:var(--mint)}.unknown{background:#62718e}.review{background:var(--amber)}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:.8rem}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}details.treatment{margin:16px 0;border:1px solid var(--line);border-radius:14px;background:var(--panel)}summary{cursor:pointer;padding:18px;font-weight:650}summary:hover{color:var(--mint)}.treatment>p,.treatment>.controls{padding:0 18px}.full{min-width:1100px}.full .question{min-width:230px;max-width:330px}.full .response{min-width:180px;max-width:330px;white-space:pre-wrap;overflow-wrap:anywhere}.full .answer{min-width:150px;max-width:220px}.badge{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:3px 7px;font-size:.72rem;line-height:1.4}.badge.good{color:var(--mint)}.badge.flag{color:var(--amber)}.evidence summary{padding:0;font-size:.78rem}.evidence p{font-size:.8rem;margin:8px 0}.evidence ul{padding-left:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.72rem;max-width:440px}input{background:var(--bg);border:1px solid var(--line);border-radius:8px;color:var(--text);font:inherit;padding:9px 12px;width:min(420px,100%)}footer{margin-top:40px;font-size:.8rem;color:var(--muted)}@media(max-width:760px){.hero{grid-template-columns:1fr}.hero svg{display:none}.cards{grid-template-columns:repeat(2,1fr)}main{padding:24px 14px}.panel{padding:12px}}
"""


def html_report(report, *, reviewed=None):
    esc = lambda value: escape(str(value), quote=True)
    matches = sum(c["original_key_matches"] for c in report["configurations"])
    attempts = sum(c["attempt_receipts"] for c in report["configurations"])
    flagged = sum(any(r["source_review"].get(k) for k in ("date_flags", "prompt_flags", "grading_flags"))
                  for r in report["configurations"][0]["probes"])
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Native factual recall · Consistency research</title><meta name='description' content='Full 129-question factual-recall results for Astra, Grok, GLM-5.3 and six Luna reasoning settings, with original-key grades and source evidence.'>",
        "<style>" + CSS + "</style></head><body><main>",
        "<a href='./'>← Consistency research</a><div class='hero'><div><p class='eyebrow'>Native subscriptions · Factual recall · 13 September 2026</p>",
        "<h1>Same questions.<br>Nine model settings.</h1><p class='muted'>Explore every response from Astra, Grok, GLM-5.3 and Luna at six reasoning levels. The complete historical catalog is retained, including political topics.</p></div>",
        "<svg viewBox='0 0 240 230' role='img' aria-label='Nine model settings connected to one shared factual question catalog'><circle cx='120' cy='115' r='88' fill='none' stroke='#2d4662'/><circle cx='120' cy='115' r='50' fill='none' stroke='#2d4662'/><g stroke='#3d796f'>",
        "".join(f"<path d='M120 115 L{x} {y}'/>" for x,y in [(120,27),(177,48),(207,99),(196,159),(150,198),(90,198),(44,159),(33,99),(63,48)]),
        "</g><g fill='#63e1bd'>" + "".join(f"<circle cx='{x}' cy='{y}' r='7'/>" for x,y in [(120,27),(177,48),(207,99),(196,159),(150,198),(90,198),(44,159),(33,99),(63,48)]) +
        "</g><circle cx='120' cy='115' r='30' fill='#193a43'/><text x='120' y='121' text-anchor='middle' fill='#edf2ff' font-family='system-ui' font-size='19'>129</text></svg></div>",
        f"<p class='eyebrow'>{esc(report['publication_status'])}</p><div class='cards'>",
        f"<div class='card'><strong>{report['completed_observations']:,}<small> / 1,161</small></strong><span>Terminal observations</span></div>",
        f"<div class='card'><strong>{attempts:,}</strong><span>Saved attempts, failures retained</span></div>",
        f"<div class='card'><strong>{matches:,}</strong><span>ORIGINAL-KEY matches across settings</span></div>",
        f"<div class='card'><strong>{flagged}<small> / 129</small></strong><span>Source rows with review flags</span></div></div>",
        "<p class='notice'>" + esc(DISCLAIMER) + "</p>",
        f"<nav class='downloads' aria-label='Research downloads'><a href='factual-recall.json'>Full data JSON</a><a href='factual-recall-usage.json'>Subscription usage</a><a href='probe-source-audit.json'>Source audit JSON</a><a href='{DOCS}native-probes-20260913.md'>Procedure &amp; tables</a><a href='{DOCS}probe-source-review-20260913.md'>Source methods</a></nav>",
        "<h2>Original-key results</h2><p class='muted'>Fixed treatment order; counts use all 129 catalog questions as the denominator. Unmatched replies require review and are not automatically incorrect.</p>",
        "<div class='panel'><div class='legend'><span><i class='match'></i>Original-key match</span><span><i class='unknown'></i>UNKNOWN</span><span><i class='review'></i>Other / review</span></div><div class='scroll'><table><thead><tr><th>Model</th><th>Reasoning</th><th>Terminal</th><th>ORIGINAL-KEY matches</th><th>UNKNOWN</th><th>Unmatched / review</th><th>Observed tokens</th></tr></thead><tbody>"]
    for i, c in enumerate(report["configurations"]):
        m, u = c["original_key_matches"], c["statuses"].get("explicit_unknown", 0)
        other = CATALOG_COUNT - m - u
        out += [f"<tr><td><a href='#setting-{i}'>{esc(c['model'])}</a></td><td>{esc(c['reasoning'])}</td><td class='num'>{c['completed_observations']} / 129</td>",
            f"<td class='num'>{m} / 129<div class='bar' aria-label='{m} original-key matches; {u} unknown; {other} other or pending'>" +
            "".join(f"<span class='{cls}' style='width:{count/129*100:.3f}%'></span>" for cls,count in [("match",m),("unknown",u),("review",other)]) + "</div></td>",
            f"<td class='num'>{u}</td><td class='num'>{c['statuses'].get('manual_review',0)}</td><td class='num'>{c['usage_totals_including_failed_attempts'].get('total_tokens',0):,}" +
            (" + unknown" if c["attempts_with_usage_unavailable"] else "") + "</td></tr>"]
    out += ["</tbody></table></div></div><p class='muted'>Observed tokens include all saved attempts. Receipt and proxy tokens overlap and must not be added. Exact experiment shares of subscription limits may be unavailable; the linked usage report separates measured shared-window deltas from unknown attribution.</p>",
        reviewed["summary_html"] if reviewed else "",
        "<h2>The protocol</h2><p>Every model received the same frozen questions and factual prompt through its native subscription client, in fresh sessions instructed to use no tools, browsing, files, memory or delegation. Completed responses were not retried for a preferred answer. Publication requires all nine settings, 129 terminal observations each, and reconciled launch receipts.</p>",
        "<details class='treatment'><summary>Read the exact factual prompt and system instruction</summary><pre style='max-width:none;padding:18px'>" + esc(report["protocol"]["system_instruction"]) + "\n\n" + esc(report["protocol"]["prompt_template"]) + "</pre><p class='muted'>" + esc(report["prompt_evidence_limit"]) + "</p></details>",
        "<h2>Every question, answer and source flag</h2><p class='muted'>Expand a setting for its complete 129-row table. Source notes show evidence and proposed corrections; the original answer key remains unchanged. Attempt details retain completion, no-tools, model, usage, timing and private receipt hash evidence.</p>"]
    for i, c in enumerate(report["configurations"]):
        out += [f"<details class='treatment' id='setting-{i}'><summary>{esc(c['model'])} · {esc(c['reasoning'])} <span class='muted'>— {c['original_key_matches']} original-key matches · {c['completed_observations']}/129 terminal</span></summary>",
            f"<p class='muted'>Statuses: {esc(json.dumps(c['statuses'],sort_keys=True))}</p><div class='controls'><label>Filter this table <input type='search' data-filter='rows-{i}' placeholder='Question, response, date or flag'></label></div>",
            f"<div class='scroll'><table class='full'><thead><tr><th># / recorded date</th><th>Question</th><th>Original key</th><th>Full response</th><th>Original-key grade</th><th>Source review / attempts</th></tr></thead><tbody id='rows-{i}'>"]
        for r in c["probes"]:
            p, review, grade = r["original_probe"], r["source_review"], r["original_key_grade"]
            reviewed_cell = reviewed["rows"][c["configuration"]][r["probe_id"]] if reviewed else ""
            flags = [(key.removesuffix("_flags"), f) for key in ("date_flags", "prompt_flags", "grading_flags") for f in review.get(key, [])]
            out += [f"<tr><td class='num'>{r['catalog_index']}<br><small>{esc(p['event_date'])}</small></td><td class='question'>{esc(p['question'])}</td>",
                f"<td class='answer'>{esc(p['answer'])}<br><small>Aliases: {esc(', '.join(p.get('accept',[])) or 'none')}</small></td>",
                f"<td class='response'>{esc(r['response'] if r['response'] is not None else 'PENDING')}</td><td><span class='badge {'good' if grade['status']=='exact_alias_match' else 'flag'}'>{esc('ORIGINAL-KEY match' if grade['status']=='exact_alias_match' else grade['status'])}</span></td><td>",
                reviewed_cell,
                f"<span class='badge'>{esc(review.get('answer_evidence_level','unreviewed'))} core-answer evidence</span> ",
                f"<details class='evidence'><summary>{len(flags)} flags · sources &amp; proposed corrections</summary><p>{esc(review.get('core_answer_status',''))}</p>"]
            for kind, flag in flags:
                out.append(f"<p><strong>{esc(kind)}: {esc(flag['code'])}</strong><br>{esc(flag['detail'])}</p>")
            for evidence in review.get("evidence", []):
                url = evidence.get("url", "")
                link = f"<a href='{esc(url)}' rel='noreferrer'>Source</a>" if url.startswith(("https://", "http://")) else "Source"
                out.append(f"<p>{link} · {esc(evidence.get('relationship',''))}<br>{esc(evidence.get('excerpt',''))}</p>")
            for correction in review.get("proposed_corrections", []):
                out.append("<p><strong>Proposed, not applied</strong></p><pre>" + esc(json.dumps(correction,ensure_ascii=False,indent=2)) + "</pre>")
            out += ["</details><details class='evidence'><summary>" + str(len(r["attempts"])) + " attempt(s) · evidence</summary><pre>" +
                    esc(json.dumps(r["attempts"],ensure_ascii=False,indent=2)) + "</pre></details></td></tr>"]
        out.append("</tbody></table></div></details>")
    out += [f"<footer><p>Generated {esc(report['generated_at'])}. Source proposals remain unapplied. Verified cutoff: not established.</p><p>{esc(report['privacy'])}</p><p><a href='{REPO}'>Repository</a> · <a href='{DOCS}native-probes-20260913.md'>Reproducible method and caveats</a></p></footer></main>",
        "<script>document.querySelectorAll('[data-filter]').forEach(input=>input.addEventListener('input',()=>{const q=input.value.toLocaleLowerCase();document.getElementById(input.dataset.filter).querySelectorAll('tr').forEach(row=>{row.hidden=!row.textContent.toLocaleLowerCase().includes(q)})}));function openTarget(){const id=location.hash.slice(1);const el=document.getElementById(id);if(el&&el.matches('details.treatment'))el.open=true}window.addEventListener('hashchange',openTarget);openTarget();</script></body></html>"]
    return "".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/native-probes-20260913")
    parser.add_argument("--source-audit", type=Path, default=ROOT / "data/research/probe-source-audit-20260913.json")
    parser.add_argument("--json-output", type=Path, default=ROOT / "data/research/native-probes-20260913.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "docs/native-probes-20260913.md")
    parser.add_argument("--html-output", type=Path, default=ROOT / "dash/factual-recall.html")
    parser.add_argument("--allow-incomplete", action="store_true", help="Local preview only; never writes repository publication paths")
    args = parser.parse_args(argv)
    outputs = [args.json_output, args.markdown_output, args.html_output]
    if args.allow_incomplete and any(p.resolve().is_relative_to(ROOT.resolve()) for p in outputs):
        parser.error("Incomplete local previews require all three output paths outside the repository")
    report = build_export(args.run_dir, args.source_audit, allow_incomplete=args.allow_incomplete)
    texts = [json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n", markdown(report), html_report(report)]
    for path, text in zip(outputs, texts):
        probes.atomic_write(path, text)
    print(json.dumps({"status": report["publication_status"], "configurations": report["configuration_count"],
                      "completed_observations": report["completed_observations"], "catalog_count": report["catalog_count"]}))


if __name__ == "__main__":
    main()
