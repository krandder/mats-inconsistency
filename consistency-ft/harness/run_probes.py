#!/usr/bin/env python3
"""Prepare, collect and audit factual-recall probes with explicit execution.

An exact answer-key match is an observation, not proof of training contamination
or a verified knowledge cutoff. Source claims and unresolved rows stay visible.
Only collect-fireworks --execute can access the network; other commands are offline.
"""
from __future__ import annotations

import argparse
import calendar
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "factual-recall-v2"
PROBE_PROMPT = """Answer the factual question below using only your existing knowledge. Do not use tools, browse, read files, consult memory, or delegate. Return only a name or short phrase, without commentary. If you do not know or cannot answer, return exactly UNKNOWN.

Question:
{question}"""
LIMITATION = ("No verified cutoff is inferred. Exact matches can arise from prior "
              "announcements, inference, guessing, post-training, or external context. "
              "UNKNOWN, unmatched replies, and absent observations do not establish "
              "that a model lacks knowledge or that a time window is uncontaminated.")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def date_interval(value):
    """Keep year/month/day precision, without inventing an exact month date."""
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", value):
        raise ValueError(f"invalid date precision: {value!r}")
    parts = list(map(int, value.split("-")))
    year = parts[0]
    if len(parts) == 1:
        return date(year, 1, 1), date(year, 12, 31), "year"
    month = parts[1]
    if len(parts) == 2:
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]), "month"
    day = date(*parts)
    return day, day, "day"


def temporal_relation(event_date, cutoff):
    if not cutoff:
        return "cutoff_not_stated"
    try:
        event_start, event_end, _ = date_interval(event_date)
        cutoff_start, cutoff_end, _ = date_interval(cutoff)
    except (ValueError, TypeError):
        return "invalid_date"
    if event_start > cutoff_end:
        return "after_claimed_cutoff"
    if event_end < cutoff_start:
        return "before_claimed_cutoff"
    return "overlapping_date_precision"


def event_availability(event_date, observation_day):
    try:
        start, end, _ = date_interval(event_date)
    except (ValueError, TypeError):
        return "invalid_date"
    if start > observation_day:
        return "future_event"
    if end >= observation_day:
        return "overlapping_observation_day"
    return "date_elapsed"


def normalize(text):
    """Only case, Unicode form, whitespace, wrapping quotes and final period."""
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    if len(text) > 1 and (text[0], text[-1]) in {(chr(34), chr(34)), ("'", "'"), ("“", "”")}:
        text = text[1:-1].strip()
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].strip()
    return " ".join(text.split())


def grade_reply(reply, probe):
    if not isinstance(reply, str) or not reply.strip():
        return {"status": "malformed_response", "matched_alias": None}
    normalized = normalize(reply)
    if normalized == "unknown":
        return {"status": "explicit_unknown", "matched_alias": None}
    # Never infer surname/component/substring aliases from explicit alternatives.
    for alias in [probe.get("answer", ""), *(probe.get("accept") or [])]:
        if isinstance(alias, str) and normalize(alias) and normalized == normalize(alias):
            return {"status": "exact_alias_match", "matched_alias": alias}
    return {"status": "manual_review", "matched_alias": None}


def known(reply, probe):
    """Compatibility helper: exact answer-key match only; not a knowledge claim."""
    return grade_reply(reply, probe)["status"] == "exact_alias_match"


def source_assessment(probe):
    source = probe.get("source")
    cited = isinstance(source, str) and bool(re.match(r"https?://\S+$", source))
    return {"citation_status": "url_present_unverified" if cited else "no_external_citation",
            "source_publication_date": probe.get("source_publication_date"),
            "source_retrieved_at": probe.get("source_retrieved_at"),
            "verification_status": "not_independently_verified_by_runner"}


def inspect_probe(probe, as_of):
    issues = []
    for field in ("question", "answer", "quarter", "category", "source"):
        if not isinstance(probe.get(field), str) or not probe[field].strip():
            issues.append("invalid_" + field)
    aliases = probe.get("accept")
    if not isinstance(aliases, list) or not aliases or any(not isinstance(a, str) or not a.strip() for a in aliases):
        issues.append("invalid_accept")
    if isinstance(probe.get("answer"), str) and normalize(probe["answer"]) == "unknown":
        issues.append("reference_answer_is_unknown_sentinel")
    try:
        start, end, precision = date_interval(probe.get("event_date"))
        quarters = {f"{d.year}Q{(d.month - 1) // 3 + 1}" for d in (start, end)}
        if len(quarters) != 1 or probe.get("quarter") not in quarters:
            issues.append("quarter_date_mismatch")
    except (ValueError, TypeError):
        precision = None
        issues.append("invalid_event_date")
    availability = event_availability(probe.get("event_date"), as_of)
    # A literal answer already in the question is a validity warning, not proof
    # of either a bad source or a knowledgeable model. Keep the row visible.
    question = normalize(probe.get("question", "")) if isinstance(probe.get("question"), str) else ""
    literals = [probe.get("answer", ""), *(aliases if isinstance(aliases, list) else [])]
    leaked = [a for a in literals if isinstance(a, str) and normalize(a) and
              re.search(r"(?<!\w)" + re.escape(normalize(a)) + r"(?!\w)", question)]
    return {"schema_issues": issues, "event_date_precision": precision,
            "availability_at_preparation": availability, "source": source_assessment(probe),
            "literal_answer_in_question": bool(leaked),
            "request_ready": not issues and availability == "date_elapsed"}


def prepare(directory, probes_path, cutoffs_path, *, model, provider, reasoning, as_of, cutoff_key=None):
    """Freeze the entire catalog. Invalid and date-uncertain rows remain visible."""
    observation_day = date.fromisoformat(as_of)
    if any(not isinstance(v, str) or not v.strip() for v in (model, provider, reasoning)):
        raise ValueError("model, provider, and reasoning must be explicit")
    probes = read_jsonl(probes_path)
    if not probes or any(not isinstance(probe, dict) for probe in probes):
        raise ValueError("probe catalog must contain objects")
    cutoffs = json.loads(Path(cutoffs_path).read_text())
    if not isinstance(cutoffs, dict) or any(not isinstance(c, dict) for c in cutoffs.values()):
        raise ValueError("cutoff catalog must map model keys to objects")
    lookup = cutoff_key or model.split("/")[-1]
    entries = [{"probe_id": sha(canonical(p)), "probe": p, "review": inspect_probe(p, observation_day)} for p in probes]
    if len({e["probe_id"] for e in entries}) != len(entries):
        raise ValueError("exact duplicate probes require source correction, not silent deduplication")
    manifest = {"schema_version": 2, "protocol_id": PROTOCOL, "prompt_template": PROBE_PROMPT,
                "model": model, "provider": provider, "reasoning": reasoning, "as_of": as_of,
                "catalog_count": len(entries), "full_catalog_preserved": True,
                "probes_file_sha256": sha(Path(probes_path).read_bytes()),
                "cutoffs_file_sha256": sha(Path(cutoffs_path).read_bytes()),
                "cutoff_key": lookup, "cutoff_claim": cutoffs.get(lookup),
                "cutoff_catalog": cutoffs, "probes": entries}
    manifest["campaign_id"] = sha(canonical(manifest))
    requests = [{"probe_id": e["probe_id"], "model": model, "provider": provider, "reasoning": reasoning,
                 "protocol_id": PROTOCOL, "prompt": PROBE_PROMPT.format(question=e["probe"]["question"]),
                 "prompt_sha256": sha(PROBE_PROMPT.format(question=e["probe"]["question"]))}
                for e in entries if e["review"]["request_ready"]]
    directory = Path(directory)
    with locked(directory):
        path = directory / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("campaign manifest differs; use a new run directory")
        atomic_write(path, json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        atomic_write(directory / "requests.jsonl", "".join(canonical(r) + "\n" for r in requests))
    return manifest


def load_manifest(directory):
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    unsigned = {k: v for k, v in manifest.items() if k != "campaign_id"}
    if sha(canonical(unsigned)) != manifest.get("campaign_id"):
        raise ValueError("campaign manifest fingerprint mismatch")
    return manifest


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("receipt timestamps must include a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid receipt timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("receipt timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_receipt(receipt, manifest, *, allow_mock=False):
    if not isinstance(receipt, dict):
        raise ValueError("receipt must be an object")
    entry = next((e for e in manifest["probes"] if e["probe_id"] == receipt.get("probe_id")), None)
    if entry is None:
        raise ValueError("receipt is outside the frozen probe catalog")
    if not entry["review"]["request_ready"]:
        raise ValueError("receipt targets an invalid, future, or date-uncertain probe")
    for key in ("model", "provider", "reasoning", "protocol_id"):
        if receipt.get(key) != manifest[key]:
            raise ValueError("receipt treatment mismatch: " + key)
    if receipt.get("prompt_sha256") != sha(PROBE_PROMPT.format(question=entry["probe"]["question"])):
        raise ValueError("receipt prompt fingerprint mismatch")
    start, finish = timestamp(receipt.get("started_at")), timestamp(receipt.get("finished_at"))
    if finish < start or finish.date() > datetime.now(timezone.utc).date():
        raise ValueError("receipt timestamps conflict with each other or are future-dated")
    if receipt.get("status") not in {"completed", "infrastructure_error", "refusal"}:
        raise ValueError("receipt status must be completed, refusal, or infrastructure_error")
    if not isinstance(receipt.get("response"), str):
        raise ValueError("receipt must preserve its full response text, including an empty failure response")
    if type(receipt.get("tools_used")) is not bool:
        raise ValueError("receipt must explicitly record tools_used")
    if receipt["status"] == "infrastructure_error" and not receipt.get("error"):
        raise ValueError("infrastructure errors require an error description")
    usage = receipt.get("usage")
    if "usage" not in receipt or (usage is not None and (not isinstance(usage, dict) or
            any(type(v) is not int or v < 0 for v in usage.values()))):
        raise ValueError("usage must contain nonnegative integer counters or explicit null")
    provenance = receipt.get("provenance", {})
    if not isinstance(provenance, dict) or not all(isinstance(provenance.get(k), str) and provenance[k].strip()
                                                  for k in ("transport", "adapter")):
        raise ValueError("receipt requires transport and adapter provenance")
    if not any(isinstance(provenance.get(k), str) and provenance[k].strip() for k in ("session_id", "request_id")):
        raise ValueError("receipt requires a native session or provider request identifier")
    if provenance["transport"] == "mock" and not allow_mock:
        raise ValueError("mock receipts require the explicit mock command")
    return entry


def append_to_history(history, incoming, manifest):
    """Validate order, provenance and first-terminal-result resume semantics."""
    identities, current, modes, sessions = set(), {}, set(), {}
    for row in [*history, *incoming]:
        receipt = row["receipt"]
        if sha(canonical(receipt)) != row["receipt_sha256"]:
            raise ValueError("saved receipt fingerprint mismatch")
        validate_receipt(receipt, manifest, allow_mock=True)
        if row["receipt_sha256"] in identities:
            raise ValueError("duplicate stored receipt")
        mock = receipt["provenance"]["transport"] == "mock"
        if modes and mock not in modes:
            raise ValueError("synthetic and imported provider observations cannot share a campaign")
        provenance = receipt["provenance"]
        for key in ("session_id", "request_id"):
            if provenance.get(key):
                identity = (receipt["provider"], provenance["transport"], key, provenance[key])
                if identity in sessions and sessions[identity] != receipt["probe_id"]:
                    raise ValueError("a session or provider request identifier was reused across probes")
                sessions[identity] = receipt["probe_id"]
        prior = current.get(receipt["probe_id"])
        if prior:
            if prior["status"] != "infrastructure_error":
                raise ValueError("a completed observation cannot be replaced or retried for a preferred answer")
            if timestamp(receipt["started_at"]) < timestamp(prior["finished_at"]):
                raise ValueError("retry overlaps its preceding attempt")
        identities.add(row["receipt_sha256"])
        current[receipt["probe_id"]] = receipt
        modes.add(mock)
    return current


def import_receipts(directory, source_path, *, allow_mock=False):
    """Atomically append validated receipts, keeping failed attempts and usage."""
    directory, source_path = Path(directory), Path(source_path)
    manifest = load_manifest(directory)
    source_bytes = source_path.read_bytes()
    incoming = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    for row in incoming:
        validate_receipt(row, manifest, allow_mock=allow_mock)
    with locked(directory):
        path = directory / "receipts.jsonl"
        existing = read_jsonl(path) if path.exists() else []
        identities = {row["receipt_sha256"] for row in existing}
        added = []
        for receipt in incoming:
            fingerprint = sha(canonical(receipt))
            if fingerprint not in identities:
                added.append({"receipt_sha256": fingerprint, "import_file_sha256": sha(source_bytes), "receipt": receipt})
                identities.add(fingerprint)
        append_to_history(existing, added, manifest)
        atomic_write(path, "".join(canonical(row) + "\n" for row in [*existing, *added]))
    return {"added": len(added), "already_present": len(incoming) - len(added)}


def build_report(directory):
    directory = Path(directory)
    manifest = load_manifest(directory)
    receipt_path = directory / "receipts.jsonl"
    saved = read_jsonl(receipt_path) if receipt_path.exists() else []
    current = append_to_history(saved, [], manifest)
    usage, usage_unavailable = Counter(), 0
    for row in saved:
        if row["receipt"]["usage"] is None:
            usage_unavailable += 1
        else:
            usage.update(row["receipt"]["usage"])
    synthetic = any(r["provenance"]["transport"] == "mock" for r in current.values())
    cutoff = (manifest.get("cutoff_claim") or {}).get("advertised")
    rows, statuses, quarters = [], Counter(), defaultdict(Counter)
    for entry in manifest["probes"]:
        probe, receipt = entry["probe"], current.get(entry["probe_id"])
        grade = {"status": "pending" if entry["review"]["request_ready"] else "probe_requires_review", "matched_alias": None}
        availability = entry["review"]["availability_at_preparation"]
        if receipt:
            availability = event_availability(probe["event_date"], timestamp(receipt["started_at"]).date())
            if receipt["status"] == "infrastructure_error":
                grade["status"] = "infrastructure_error"
            elif receipt["tools_used"]:
                grade["status"] = "tool_use"
            elif receipt["status"] == "refusal":
                grade["status"] = "refusal"
            elif receipt.get("truncated"):
                grade["status"] = "truncated_response"
            elif receipt.get("provider_model_mismatch"):
                grade["status"] = "provenance_review"
            else:
                grade = grade_reply(receipt["response"], probe)
        relation, status = temporal_relation(probe.get("event_date"), cutoff), grade["status"]
        statuses[status] += 1
        quarters[probe.get("quarter", "invalid")][status] += 1
        rows.append({"probe_id": entry["probe_id"], "question": probe.get("question"),
                     "event_date": probe.get("event_date"), "quarter": probe.get("quarter"), "grade": grade,
                     "event_availability_at_observation": availability, "cutoff_relation": relation,
                     "source_review": entry["review"],
                     "candidate_post_cutoff_match": not synthetic and status == "exact_alias_match" and
                         relation == "after_claimed_cutoff" and availability == "date_elapsed"})
    terminal = sum(r["status"] != "infrastructure_error" for r in current.values())
    inventory = [{"model_key": key, "advertised": claim.get("advertised"), "release": claim.get("release"),
                  "source": claim.get("source"), "recorded_confidence": claim.get("confidence"),
                  "verification_status": "not_independently_verified_by_runner"}
                 for key, claim in manifest["cutoff_catalog"].items()]
    launches_path = directory / "collection/launches.jsonl"
    launches = read_jsonl(launches_path) if launches_path.exists() else []
    completed_launches = sum((directory / row["receipt_file"]).exists() for row in launches)
    return {"schema_version": 2, "campaign_id": manifest["campaign_id"],
            "model": manifest["model"], "provider": manifest["provider"], "reasoning": manifest["reasoning"],
            "protocol_id": PROTOCOL, "as_of": manifest["as_of"], "synthetic": synthetic,
            "observed_through": max((r["finished_at"] for r in current.values()), key=timestamp, default=None),
            "catalog_count": manifest["catalog_count"], "request_ready": sum(e["review"]["request_ready"] for e in manifest["probes"]),
            "observed_probes": len(current), "completed_observations": terminal,
            "full_catalog_completed": terminal == manifest["catalog_count"] and not synthetic,
            "attempt_receipts": len(saved), "statuses": dict(statuses),
            "by_quarter": {q: dict(c) for q, c in sorted(quarters.items())},
            "usage_totals_including_failed_attempts": dict(usage), "attempts_with_usage_unavailable": usage_unavailable,
            "usage_is_provider_reported_not_subscription_fraction": True,
            "inference_requests_performed_by_runner": len(launches) if completed_launches == len(launches) else None,
            "launch_reservations": len(launches), "launches_without_final_receipt": len(launches) - completed_launches,
            "import_provenance_assurance": "identity and fingerprints checked; provider assertions not independently authenticated",
            "source_citation_counts": dict(Counter(e["review"]["source"]["citation_status"] for e in manifest["probes"])),
            "cutoff_claim": manifest.get("cutoff_claim"), "cutoff_catalog_claims": inventory, "verified_cutoff": None,
            "candidate_post_cutoff_matches": sum(r["candidate_post_cutoff_match"] for r in rows),
            "limitation": LIMITATION, "probes": rows}


def write_report(directory):
    report, directory = build_report(directory), Path(directory)
    atomic_write(directory / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    text = ["# Factual-recall probe audit", "",
            f"Model: {cell(report['model'])}; reasoning: {cell(report['reasoning'])}. Prepared as of {report['as_of']}.", "",
            f"Latest saved observation: {report['observed_through'] or 'none'}.", "",
            f"{'**SYNTHETIC TEST DATA.** ' if report['synthetic'] else ''}Completed observations: "
            f"{report['completed_observations']}/{report['catalog_count']}; saved attempts: {report['attempt_receipts']}.", "",
            "**Verified cutoff: not established.** " + LIMITATION, "",
            "| Observation status | Probe count |", "| --- | ---: |"]
    text += [f"| {status} | {count} |" for status, count in sorted(report["statuses"].items())]
    text += ["", "Candidate matches after the recorded cutoff claim: " + str(report["candidate_post_cutoff_matches"]) +
             ". These require source, announcement-date, and model-provenance review; they are not a contamination verdict.", "",
             "Raw imported receipts, usage, responses and session/provider identifiers remain in private `receipts.jsonl`. "
             "The full catalog, including unready and pending rows, is retained in `manifest.json` and `report.json`.", ""]
    atomic_write(directory / "report.md", "\n".join(text))
    return report


def mock_receipts(directory):
    """A deterministic UNKNOWN-only plumbing fixture, never model observations."""
    directory = Path(directory)
    manifest = load_manifest(directory)
    stamp = manifest["as_of"] + "T00:00:00+00:00"
    # Do not inject reference answers, even into this synthetic fixture.
    receipts = [{**{k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "protocol_id", "prompt_sha256")},
                 "started_at": stamp, "finished_at": stamp, "status": "completed", "response": "UNKNOWN",
                 "tools_used": False, "usage": None,
                 "provenance": {"transport": "mock", "adapter": "unknown-only-fixture-v1", "session_id": "mock-" + request["probe_id"]}}
                for request in read_jsonl(directory / "requests.jsonl")]
    path = directory / "mock-input.jsonl"
    atomic_write(path, "".join(canonical(r) + "\n" for r in receipts))
    return import_receipts(directory, path, allow_mock=True)


def fireworks_requester():
    """Initialize credentials only after explicit execution; one HTTP attempt."""
    import requests
    sys.path.insert(0, str(ROOT))
    from common import BASE_URL, _api_key
    key = _api_key()
    if not key:
        raise ValueError("Fireworks credentials unavailable; no requests were sent")

    def request(payload):
        try:
            response = requests.post(BASE_URL + "/chat/completions",
                                     headers={"Authorization": "Bearer " + key}, json=payload, timeout=90)
        except requests.RequestException as exc:
            return {"http_status": None, "error": type(exc).__name__, "data": None}
        try:
            data = response.json()
        except ValueError:
            data = {"unparsed_response": response.text}
        # Gateways occasionally echo submitted credentials. Never retain those.
        data = json.loads(json.dumps(data).replace(key, "[REDACTED]"))
        return {"http_status": response.status_code, "data": data}
    return request


def provider_receipt(request, result, manifest, launch):
    """Keep provider failures and truncation distinct from an UNKNOWN answer."""
    data = result.get("data")
    body = data if isinstance(data, dict) else {}
    choices = body.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    refusal = message.get("refusal")
    response = content if isinstance(content, str) else (refusal if isinstance(refusal, str) else "")
    successful = result.get("http_status") == 200
    raw_usage = body.get("usage")
    usage = {k: v for k, v in raw_usage.items() if type(v) is int and v >= 0} if isinstance(raw_usage, dict) else None
    usage = usage or None
    reported_model = body.get("model")
    record = {**{k: request[k] for k in ("probe_id", "model", "provider", "reasoning", "protocol_id", "prompt_sha256")},
              "started_at": launch["started_at"], "finished_at": datetime.now(timezone.utc).isoformat(),
              "status": "infrastructure_error" if not successful else ("refusal" if refusal else "completed"),
              "response": response, "usage": usage, "tools_used": bool(message.get("tool_calls") or message.get("function_call")),
              "truncated": choice.get("finish_reason") == "length", "finish_reason": choice.get("finish_reason"),
              "reported_model": reported_model,
              "provider_model_mismatch": bool(reported_model and reported_model != manifest["model"]),
              "provider_response": result,
              "provenance": {"transport": "fireworks-api", "adapter": "single-http-attempt-v2",
                             "request_id": str(body.get("id") or "local-launch-" + launch["attempt_id"]),
                             "request_id_source": "provider" if body.get("id") else "local_launch",
                             "local_attempt_id": launch["attempt_id"]}}
    if not successful:
        record["error"] = result.get("error") or "HTTP " + str(result.get("http_status"))
    return record


def collect_fireworks(directory, *, execute=False, max_calls=None, max_tokens=2048, retry_errors=False, requester=None):
    """Collect the frozen catalog, one durable attempt at a time, without fallback.

    max_calls is a cumulative launch allowance across resumes. A launch whose
    outcome was not saved blocks replay: its provider consumption is unknown.
    requester is injectable for offline tests; the CLI never accepts one.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    if manifest["provider"] != "fireworks":
        raise ValueError("collect-fireworks requires a campaign prepared with provider fireworks")
    report = build_report(directory)
    if not execute:
        return {"status": "plan_only", "request_ready": report["request_ready"],
                "completed_observations": report["completed_observations"], "network_requests": 0}
    if type(max_calls) is not int or max_calls < 1 or type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("execution requires a positive --max-calls allowance and --max-tokens cap")
    if report["synthetic"]:
        raise ValueError("cannot collect live observations into a synthetic campaign")
    if date.fromisoformat(manifest["as_of"]) > datetime.now(timezone.utc).date():
        raise ValueError("live collection cannot use a future preparation date")
    requests = []
    for entry in manifest["probes"]:
        if entry["review"]["request_ready"]:
            prompt = PROBE_PROMPT.format(question=entry["probe"]["question"])
            requests.append({"probe_id": entry["probe_id"], "model": manifest["model"],
                             "provider": manifest["provider"], "reasoning": manifest["reasoning"],
                             "protocol_id": PROTOCOL, "prompt_sha256": sha(prompt), "prompt": prompt})
    config = {"max_tokens": max_tokens, "temperature": 0, "reasoning_effort": manifest["reasoning"],
              "model": manifest["model"], "adapter": "single-http-attempt-v2"}
    collection = directory / "collection"
    with locked(collection):
        config_path = collection / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("collection treatment differs; use a new campaign")
        atomic_write(config_path, canonical(config) + "\n")
        launches_path = collection / "launches.jsonl"
        launches = read_jsonl(launches_path) if launches_path.exists() else []
        for launch in launches:
            final = directory / launch["receipt_file"]
            if not final.exists():
                raise ValueError("a reserved launch has no final receipt; reconcile its outcome before resuming")
            import_receipts(directory, final)
        ledger = directory / "receipts.jsonl"
        current = append_to_history(read_jsonl(ledger) if ledger.exists() else [], [], manifest)
        collected, stop = 0, "complete"
        for request in requests:
            previous = current.get(request["probe_id"])
            if previous:
                if previous["status"] != "infrastructure_error":
                    continue
                if not retry_errors:
                    stop = "failed_attempt_requires_explicit_retry"
                    break
            if len(launches) >= max_calls:
                stop = "call_allowance_exhausted"
                break
            if requester is None:
                requester = fireworks_requester()
            payload = {"model": manifest["model"], "messages": [{"role": "user", "content": request["prompt"]}],
                       "temperature": 0, "reasoning_effort": manifest["reasoning"], "max_tokens": max_tokens}
            attempt_id = sha(manifest["campaign_id"] + ":" + str(len(launches)))
            launch = {"attempt_id": attempt_id, "probe_id": request["probe_id"],
                      "started_at": datetime.now(timezone.utc).isoformat(), "payload_sha256": sha(canonical(payload)),
                      "receipt_file": "collection/attempts/" + attempt_id + ".jsonl"}
            launches.append(launch)
            atomic_write(launches_path, "".join(canonical(row) + "\n" for row in launches))
            # No internal retries or silent reasoning fallback: one reservation,
            # one transport invocation. An unexpected interruption stays reserved.
            result = requester(payload)
            receipt = provider_receipt(request, result, manifest, launch)
            final = directory / launch["receipt_file"]
            atomic_write(final, canonical(receipt) + "\n")
            import_receipts(directory, final)
            current[request["probe_id"]] = receipt
            collected += 1
            if receipt["status"] == "infrastructure_error":
                stop = "provider_failure"
                break
    return {"status": stop, "new_attempts": collected, "cumulative_launch_reservations": len(launches)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="freeze all source probes; no inference")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--probes", type=Path, default=ROOT / "data/cutoff_probes.jsonl")
    prep.add_argument("--cutoffs", type=Path, default=ROOT / "data/cutoffs.json")
    for name in ("model", "provider", "reasoning", "as-of"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--cutoff-key")
    importer = commands.add_parser("import", help="validate and persist external receipts; no inference")
    importer.add_argument("--run-dir", type=Path, required=True)
    importer.add_argument("--receipts", type=Path, required=True)
    collector = commands.add_parser("collect-fireworks", help="plan by default; explicit execution and cumulative call allowance required")
    collector.add_argument("--run-dir", type=Path, required=True)
    collector.add_argument("--execute", action="store_true")
    collector.add_argument("--max-calls", type=int)
    collector.add_argument("--max-tokens", type=int, default=2048)
    collector.add_argument("--retry-errors", action="store_true")
    for command in ("report", "mock"):
        sub = commands.add_parser(command, help="offline report" if command == "report" else "explicit UNKNOWN-only synthetic fixture")
        sub.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.run_dir, args.probes, args.cutoffs, model=args.model, provider=args.provider,
                    reasoning=args.reasoning, as_of=args.as_of, cutoff_key=args.cutoff_key)
        elif args.command == "import":
            import_receipts(args.run_dir, args.receipts)
        elif args.command == "mock":
            mock_receipts(args.run_dir)
        elif args.command == "collect-fireworks":
            collection = collect_fireworks(args.run_dir, execute=args.execute, max_calls=args.max_calls,
                                           max_tokens=args.max_tokens, retry_errors=args.retry_errors)
            print(json.dumps(collection, sort_keys=True))
        report = write_report(args.run_dir)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({k: report[k] for k in ("model", "catalog_count", "observed_probes", "completed_observations",
                                            "statuses", "synthetic", "verified_cutoff")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
