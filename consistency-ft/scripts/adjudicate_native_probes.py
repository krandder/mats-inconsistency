#!/usr/bin/env python3
"""Validate independent, source-grounded response reviews and join frozen results.

This is an offline evidence join. It does not grade prose automatically, change
the original answer key, invoke providers, or infer knowledge cutoffs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REVIEW_FIELDS = {"item_id", "verdict", "evidence_capture_ids", "rationale", "qualifications"}
VERDICTS = {"supported_core_answer", "contradicted_core_answer", "incomplete_core_answer", "abstained", "unresolved"}
SETTINGS = [("astra", "low"), ("grok", "low"), ("glm", "native-default")] + [
    ("luna", e) for e in ("none", "low", "medium", "high", "xhigh", "max")]
MODEL_IDS = {"astra": "gpt-6-astra", "grok": "grok-4.6", "glm": "glm-5.3", "luna": "gpt-5.6-luna"}
LIMITATION = (
    "Agent-reviewed alignment of requested historical factual answers with retained sources; "
    "not human adjudication, whole-response truth, a certified accuracy estimate, a verified "
    "knowledge cutoff or proof of training contamination. Original-key grades remain unchanged. "
    "Reviewers share a model family and source packet; agreement does not supply new independent factual evidence."
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON key: " + key)
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError("Nonfinite JSON value: " + value)

    raw = Path(path).read_bytes()
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant), sha(raw)


def require_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Review text must be nonempty")
    if (any(s in value for s in ("/home/", "/srv/", "/tmp/", "/private/", "/Users/", "C:\\Users\\", "raw_capture_directory"))
            or re.search(r"\b(?:session(?:_id)?|account(?:_name|_id)?|profile|api[_ -]?key|access[_ -]?token|authorization)\s*[:=]", value, re.I)
            or re.search(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", value, re.I)):
        raise ValueError("Private path, session or credential-like metadata in public review text")


def load_inputs(study_path, audit_path, rubric_path, packet_path):
    study, study_hash = read(study_path)
    audit, audit_hash = read(audit_path)
    rubric, rubric_hash = read(rubric_path)
    packet, packet_hash = read(packet_path)
    if any(type(d.get("schema_version")) is not int or d["schema_version"] != 1 for d in (study, audit, rubric, packet)):
        raise ValueError("Unsupported study/source/rubric/packet schema")
    if (rubric["study_sha256"] != study_hash or rubric["source_audit_sha256"] != audit_hash
            or study["source_audit_sha256"] != audit_hash
            or rubric["catalog_sha256"] != study["catalog_sha256"]
            or audit["catalog_sha256"] != study["catalog_sha256"]):
        raise ValueError("Frozen study/source/rubric binding mismatch")
    if set(rubric["allowed_verdicts"]) != VERDICTS:
        raise ValueError("Unsupported review vocabulary")
    if (study["publication_status"] != "COMPLETE" or study["catalog_count"] != 129
            or study["completed_observations"] != 1161 or study["configuration_count"] != 9):
        raise ValueError("Expected the complete nine-setting native study")
    identities = [(c["provider"], c["reasoning"]) for c in study["configurations"]]
    if identities != SETTINGS:
        raise ValueError("Native settings missing, duplicated or reordered")
    sources = {r["probe_id"]: r for r in audit["probes"]}
    if len(sources) != 129 or len(audit["probes"]) != 129:
        raise ValueError("Source catalog is not complete and unique")
    items, observations = {}, []
    for c in study["configurations"]:
        if c["configuration"] != c["provider"] + "/" + c["reasoning"] or c["model"] != MODEL_IDS[c["provider"]]:
            raise ValueError("Configuration label or exact model identity mismatch")
        if (len(c["probes"]) != 129 or c["completed_observations"] != 129
                or not c["full_catalog_completed"] or c["unreconciled_launches"]):
            raise ValueError("Treatment coverage is incomplete")
        seen = set()
        for row in c["probes"]:
            pid, response = row["probe_id"], row["response"]
            if pid in seen or pid not in sources or not isinstance(response, str):
                raise ValueError("Invalid or duplicate source join")
            seen.add(pid)
            source = sources[pid]
            if row["original_probe"] != source["original_probe"] or row["catalog_index"] != source["catalog_index"]:
                raise ValueError("Study question differs from frozen source audit")
            iid = sha(canonical([pid, response]))
            observations.append({"configuration": c["configuration"], "probe_id": pid,
                                 "item_id": iid, "original_key_status": row["original_key_grade"]["status"]})
            items[iid] = {"item_id": iid, "probe_id": pid, "catalog_index": row["catalog_index"],
                          "question": row["original_probe"]["question"], "response": response,
                          "reference_answer": row["original_probe"]["answer"],
                          "source_review": {k: source[k] for k in ("answer_evidence_level", "core_answer_status",
                              "evidence", "date_flags", "prompt_flags", "grading_flags", "proposed_corrections")}}
    manual = {k: v for k, v in sorted(items.items()) if v["response"].strip().upper() != "UNKNOWN"}
    expected = {"schema_version": 1, "rubric_sha256": rubric_hash, "study_sha256": study_hash,
                "source_audit_sha256": audit_hash, "items": list(manual.values())}
    if packet != expected:
        raise ValueError("Blinded packet does not exactly match frozen question/response/source pairs")
    return {"study": study, "audit": audit, "rubric": rubric, "items": items,
            "manual": manual, "observations": observations, "study_sha256": study_hash,
            "source_audit_sha256": audit_hash, "rubric_sha256": rubric_hash, "packet_sha256": packet_hash}


def validate_decision(row, item, supplemental_evidence=()):
    if set(row) != REVIEW_FIELDS or row["item_id"] != item["item_id"] or row["verdict"] not in VERDICTS:
        raise ValueError("Review decision fields or verdict are invalid")
    require_text(row["rationale"])
    qualifications = row["qualifications"]
    if (not isinstance(qualifications, list) or not all(isinstance(q, str) for q in qualifications)
            or len(set(qualifications)) != len(qualifications)):
        raise ValueError("Review qualifications must be a unique list")
    for q in qualifications:
        require_text(q)
    evidence = row["evidence_capture_ids"]
    allowed = {e["capture_id"] for e in item["source_review"]["evidence"]}
    allowed.update(e["capture_id"] for e in supplemental_evidence)
    if (not isinstance(evidence, list) or not all(isinstance(e, str) for e in evidence)
            or len(set(evidence)) != len(evidence) or not set(evidence) <= allowed):
        raise ValueError("Review cites missing, duplicate or unrelated evidence")
    if row["verdict"] != "abstained" and not evidence:
        raise ValueError("Every substantive verdict requires retained source evidence")


def load_review(path, reviewer, inputs):
    doc, file_hash = read(path)
    required = {"schema_version", "reviewer", "rubric_sha256", "packet_sha256", "reviews"}
    if (set(doc) != required or type(doc["schema_version"]) is not int or doc["schema_version"] != 1
            or doc["reviewer"] != reviewer or doc["rubric_sha256"] != inputs["rubric_sha256"]
            or doc["packet_sha256"] != inputs["packet_sha256"]):
        raise ValueError("Independent review identity or frozen input binding mismatch")
    result = {}
    for row in doc["reviews"]:
        iid = row["item_id"]
        if iid not in inputs["manual"] or iid in result:
            raise ValueError("Missing/unknown/duplicate manual review item")
        validate_decision(row, inputs["manual"][iid])
        result[iid] = row
    if set(result) != set(inputs["manual"]):
        raise ValueError("Independent review does not cover every non-UNKNOWN pair")
    return result, file_hash


def load_supplemental(path, inputs, capture_directory):
    if capture_directory is None:
        raise ValueError("Supplemental evidence requires retained raw captures")
    doc, file_hash = read(path)
    if set(doc) != {"schema_version", "evidence"} or type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        raise ValueError("Unsupported supplemental evidence schema")
    required = {"probe_id", "capture_id", "url", "captured_at", "body_sha256", "text_sha256", "excerpt",
                "excerpt_text_normalization", "claim_supported", "publication_date", "limitations"}
    seen = {e["capture_id"] for e in inputs["audit"]["source_captures"]}
    probe_ids = {p["probe_id"] for p in inputs["audit"]["probes"]}
    result = {}
    for entry in doc["evidence"]:
        if set(entry) != required or entry["probe_id"] not in probe_ids:
            raise ValueError("Supplemental evidence fields or probe binding mismatch")
        cid = entry["capture_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", cid) or cid in seen:
            raise ValueError("Supplemental capture ID is invalid or already used")
        seen.add(cid)
        parts = urlsplit(entry["url"])
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password:
            raise ValueError("Supplemental source requires a public HTTP(S) URL")
        for field in ("body_sha256", "text_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry[field]):
                raise ValueError("Supplemental capture requires SHA-256 hashes")
        captured = datetime.fromisoformat(entry["captured_at"].replace("Z", "+00:00"))
        if captured.utcoffset() is None:
            raise ValueError("Supplemental capture timestamp must include timezone")
        for field in ("excerpt", "claim_supported", "publication_date"):
            require_text(entry[field])
        if not isinstance(entry["limitations"], list):
            raise ValueError("Supplemental limitations must be a list")
        for value in entry["limitations"]:
            require_text(value)
        if entry["excerpt_text_normalization"] != "consecutive whitespace collapsed to one space":
            raise ValueError("Unsupported source excerpt normalization")
        raw = (Path(capture_directory) / (cid + ".body")).read_bytes()
        txt = (Path(capture_directory) / (cid + ".txt")).read_bytes()
        if sha(raw) != entry["body_sha256"] or sha(txt) != entry["text_sha256"]:
            raise ValueError("Supplemental raw capture hash mismatch")
        normalize = lambda s: re.sub(r"\s+", " ", s).strip()
        if normalize(entry["excerpt"]) not in normalize(txt.decode()):
            raise ValueError("Supplemental excerpt absent from retained source text")
        result.setdefault(entry["probe_id"], []).append(entry)
    return result, file_hash


def build(inputs, review_a_path, review_b_path, resolutions_path=None, *, supplemental_path=None, supplemental_capture_dir=None):
    left, left_hash = load_review(review_a_path, "review-a", inputs)
    right, right_hash = load_review(review_b_path, "review-b", inputs)
    supplemental, supplemental_hash = ({}, None) if supplemental_path is None else load_supplemental(
        supplemental_path, inputs, supplemental_capture_dir)
    disagreements = {iid for iid in left if left[iid]["verdict"] != right[iid]["verdict"]}
    resolved, resolution_hash = {}, None
    if resolutions_path is not None:
        doc, resolution_hash = read(resolutions_path)
        if (set(doc) != {"schema_version", "rubric_sha256", "review_a_sha256", "review_b_sha256", "resolutions", "supplemental_evidence_sha256"}
                or type(doc["schema_version"]) is not int or doc["schema_version"] != 1 or doc["rubric_sha256"] != inputs["rubric_sha256"]
                or doc["review_a_sha256"] != left_hash or doc["review_b_sha256"] != right_hash
                or doc["supplemental_evidence_sha256"] != supplemental_hash):
            raise ValueError("Resolution does not bind to the two frozen independent reviews")
        for row in doc["resolutions"]:
            iid = row["item_id"]
            if iid not in left or iid in resolved:
                raise ValueError("Unknown or duplicate resolution item")
            item = inputs["manual"][iid]
            validate_decision(row, item, supplemental.get(item["probe_id"], []))
            resolved[iid] = row
    if disagreements - set(resolved):
        raise ValueError(f"{len(disagreements - set(resolved))} independent-review disagreements lack documented resolution")
    units = []
    for iid, item in sorted(inputs["items"].items()):
        if iid not in left:
            decision = {"item_id": iid, "verdict": "abstained", "evidence_capture_ids": [],
                        "rationale": "Exact UNKNOWN under the frozen mechanical rule.", "qualifications": []}
            reviewers, origin = [], "mechanical_exact_unknown"
        else:
            reviewers = [{"reviewer": name, **row} for name, row in (("review-a", left[iid]), ("review-b", right[iid]))]
            if iid in resolved:
                decision = resolved[iid]
                origin = "resolved_disagreement" if iid in disagreements else "documented_review_override"
            else:
                decision = {"item_id": iid, "verdict": left[iid]["verdict"],
                            "evidence_capture_ids": sorted(set(left[iid]["evidence_capture_ids"] + right[iid]["evidence_capture_ids"])),
                            "rationale": "Both independent reviews support this label; their separate rationales are retained below.",
                            "qualifications": list(dict.fromkeys(left[iid]["qualifications"] + right[iid]["qualifications"]))}
                origin = "independent_review_agreement"
        units.append({**item, "decision": decision, "origin": origin, "independent_reviews": reviewers,
                      "supplemental_evidence": supplemental.get(item["probe_id"], [])})
    indexed = {u["item_id"]: u for u in units}
    configurations = []
    for c in inputs["study"]["configurations"]:
        rows = []
        for observation in inputs["observations"]:
            if observation["configuration"] != c["configuration"]:
                continue
            unit = indexed[observation["item_id"]]
            rows.append({**observation, "reviewed_verdict": unit["decision"]["verdict"], "review_origin": unit["origin"]})
        if len(rows) != 129 or len({r["probe_id"] for r in rows}) != 129:
            raise ValueError("Final setting join must contain exactly 129 unique probes")
        configurations.append({"configuration": c["configuration"], "model": c["model"], "reasoning": c["reasoning"],
            "observations": len(rows), "verdict_counts": dict(Counter(r["reviewed_verdict"] for r in rows)),
            "original_matches_not_supported_as_complete": sum(r["original_key_status"] == "exact_alias_match" and r["reviewed_verdict"] != "supported_core_answer" for r in rows),
            "original_unmatched_now_supported": sum(r["original_key_status"] == "manual_review" and r["reviewed_verdict"] == "supported_core_answer" for r in rows),
            "probes": rows})
    if sum(c["observations"] for c in configurations) != 1161:
        raise ValueError("Final study join must contain exactly 1,161 observations")
    return {"schema_version": 1, "study": "native-factual-recall-adjudication-20260913",
        "generated_at": datetime.now(timezone.utc).isoformat(), "status": "COMPLETE_AGENT_REVIEW",
        **{k: inputs[k] for k in ("study_sha256", "source_audit_sha256", "rubric_sha256", "packet_sha256")},
        "review_a_sha256": left_hash, "review_b_sha256": right_hash, "resolutions_sha256": resolution_hash,
        "supplemental_evidence_sha256": supplemental_hash,
        "observations": len(inputs["observations"]), "unique_pairs": len(units), "manual_pairs": len(left),
        "mechanical_unknown_pairs": len(units) - len(left), "independent_verdict_disagreements": len(disagreements),
        "documented_overrides_of_agreements": len(set(resolved) - disagreements),
        "new_native_provider_requests": 0, "original_key_changed": False, "verified_cutoff": None,
        "limitations": LIMITATION, "configurations": configurations, "units": units}


def markdown(report):
    totals = Counter()
    for configuration in report["configurations"]:
        totals.update(configuration["verdict_counts"])
    promoted = sum(c["original_unmatched_now_supported"] for c in report["configurations"])
    lines = ["# Source-grounded native response review — 13 September 2026", "", LIMITATION, "",
        f"All **{report['observations']:,} observations** map to **{report['unique_pairs']} distinct question/response pairs**. "
        f"The frozen rule labels {report['mechanical_unknown_pairs']} exact UNKNOWN pairs as abstentions. Two agents independently "
        f"reviewed all {report['manual_pairs']} other pairs, with model labels, original grades and frequencies omitted from their packets.", "",
        f"Independent label disagreements: **{report['independent_verdict_disagreements']}**. "
        f"Documented overrides of initial agreements: **{report['documented_overrides_of_agreements']}**. Both initial reviews remain available. "
        "These reviews made no new native benchmark requests. "
        "Agent review work consumes subscriptions separately from the previously measured native campaign.", "",
        f"Final observation labels: **{totals['supported_core_answer']} supported core answers, "
        f"{totals['contradicted_core_answer']} contradicted, {totals['incomplete_core_answer']} incomplete, "
        f"{totals['abstained']} abstained, {totals['unresolved']} unresolved**. "
        f"The review supports **{promoted} observations missed by the original key**; original grades remain unchanged.", "",
        "[Interactive full tables](https://consistency.futarchy.ai/factual-recall.html) · "
        "[Review data](../data/research/native-probe-adjudication-20260913.json) · "
        "[Frozen rubric](../data/research/probe-adjudication-rubric-20260913.json) · "
        "[Independent source audit](probe-source-review-20260913.md)", "",
        "## Observed labels by setting", "",
        "Supported means the requested core identity/result is supported; source flags and response qualifications still apply. "
        "Counts use the complete common catalog and are shown in fixed treatment order. They are not a certified accuracy ranking.", "",
        "| Model / reasoning | Supported core | Contradicted core | Incomplete core | Abstained | Unresolved | Original matches not fully supported | Original unmatched now supported |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in report["configurations"]:
        counts = c["verdict_counts"]
        values = [counts.get(k, 0) for k in ("supported_core_answer", "contradicted_core_answer", "incomplete_core_answer", "abstained", "unresolved")]
        values += [c["original_matches_not_supported_as_complete"], c["original_unmatched_now_supported"]]
        lines.append("| " + c["model"] + " / " + c["reasoning"] + " | " + " | ".join(str(x) for x in values) + " |")
    lines += ["", "## Every distinct response and its source alignment", "",
              "Identical question/response pairs share one decision. Full independent rationales, qualifications and resolutions are retained in JSON.", "",
              "| Catalog row | Response | Decision | Review path | Evidence / rationale |", "| ---: | --- | --- | --- | --- |"]
    for unit in sorted(report["units"], key=lambda u: (u["catalog_index"], u["item_id"])):
        decision = unit["decision"]
        urls = {e["capture_id"]: e["url"] for e in unit["source_review"]["evidence"] + unit["supplemental_evidence"]}
        refs = " ".join("[source](<" + urls[cid].replace(">", "%3E") + ">)" for cid in decision["evidence_capture_ids"])
        cell = lambda s: escape(str(s)).replace("|", "\\|").replace("\n", "<br>")
        lines.append("| " + " | ".join(cell(x) for x in (unit["catalog_index"], unit["response"], decision["verdict"], unit["origin"]))
                     + " | " + refs + " " + cell(decision["rationale"]) + " |")
    lines += ["", "## Integrity and limits", "",
        "The join verifies the original study, source audit and rubric hashes; regenerates the exact blinded packet; "
        "requires complete unique reviews; rejects unrelated source references; and binds every resolution to both independent review hashes. "
        "UNKNOWNs and adverse outcomes are retained. Original study JSON and original-key grades are not overwritten.", "",
        "Source support inherits the audit's primary/secondary evidence levels and question/premise/date flags. "
        "Factual recall cannot establish when the model learned an answer, and failed recall cannot certify a safe upper cutoff. "
        "Fresh historical-provider measurements, independent source quality review where needed and actual training remain separate unfinished work.", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=ROOT / "data/research/native-probes-20260913.json")
    p.add_argument("--source-audit", type=Path, default=ROOT / "data/research/probe-source-audit-20260913.json")
    p.add_argument("--rubric", type=Path, default=ROOT / "data/research/probe-adjudication-rubric-20260913.json")
    p.add_argument("--review-dir", type=Path, default=ROOT / "runs/probe-adjudication-20260913")
    p.add_argument("--resolutions", type=Path)
    p.add_argument("--supplemental-evidence", type=Path)
    p.add_argument("--supplemental-captures", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "data/research/native-probe-adjudication-20260913.json")
    p.add_argument("--markdown", type=Path, default=ROOT / "docs/native-probe-adjudication-20260913.md")
    args = p.parse_args()
    inputs = load_inputs(args.study, args.source_audit, args.rubric, args.review_dir / "review-packet.json")
    report = build(inputs, args.review_dir / "review-a.json", args.review_dir / "review-b.json", args.resolutions,
                   supplemental_path=args.supplemental_evidence, supplemental_capture_dir=args.supplemental_captures)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    args.markdown.write_text(markdown(report))
    print(json.dumps({k: report[k] for k in ("status", "observations", "unique_pairs", "manual_pairs", "independent_verdict_disagreements")}))


if __name__ == "__main__":
    main()
