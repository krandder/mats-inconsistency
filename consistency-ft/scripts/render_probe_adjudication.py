#!/usr/bin/env python3
"""Render a validated agent-review overlay without rewriting the original study.

Inputs are public frozen JSON files only. This command performs no inference,
source searches, answer grading, or edits to the original-key measurements.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import adjudicate_native_probes as adjudication
from scripts import export_native_probes as original

MODELS = {"astra": "gpt-6-astra", "grok": "grok-4.6", "glm": "glm-5.3", "luna": "gpt-5.6-luna"}
VERDICT_LABELS = {
    "supported_core_answer": "Supported core answer",
    "contradicted_core_answer": "Contradicted core answer",
    "incomplete_core_answer": "Incomplete core answer",
    "abstained": "Abstained",
    "unresolved": "Unresolved",
}
SOURCE_FIELDS = ("answer_evidence_level", "core_answer_status", "evidence", "date_flags",
                 "prompt_flags", "grading_flags", "proposed_corrections")
ORIGINS = {"mechanical_exact_unknown", "independent_review_agreement", "resolved_disagreement", "documented_review_override"}
DEFAULT_REVIEW_A = ROOT / "data/research/probe-adjudication-review-a-20260913.json"
DEFAULT_REVIEW_B = ROOT / "data/research/probe-adjudication-review-b-20260913.json"
DEFAULT_RESOLUTIONS = ROOT / "data/research/probe-adjudication-resolutions-20260913.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def verify_supplement(path, expected_hash, sources):
    if expected_hash is None:
        require(path is None, "Unreferenced supplemental evidence was supplied")
        return {}
    require(path is not None and digest(expected_hash), "Referenced supplemental evidence file is required")
    document, file_hash = adjudication.read(path)
    require(file_hash == expected_hash, "Supplemental evidence byte hash mismatch")
    require(set(document) == {"schema_version", "evidence"} and type(document["schema_version"]) is int
            and document["schema_version"] == 1 and isinstance(document["evidence"], list), "Unsupported supplemental evidence schema")
    required = {"probe_id", "capture_id", "url", "captured_at", "body_sha256", "text_sha256", "excerpt",
                "excerpt_text_normalization", "claim_supported", "publication_date", "limitations"}
    seen = {e["capture_id"] for s in sources.values() for e in s["evidence"]}
    result = {}
    for entry in document["evidence"]:
        require(set(entry) == required and entry["probe_id"] in sources, "Supplemental source fields or probe identity differ")
        cid = entry["capture_id"]
        require(isinstance(cid, str) and bool(re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", cid)) and cid not in seen,
                "Supplemental capture identity is invalid or duplicated")
        seen.add(cid)
        parsed = urlsplit(entry["url"])
        require(parsed.scheme in {"https", "http"} and parsed.netloc and not parsed.username and not parsed.password,
                "Supplemental source must have a public HTTP(S) URL")
        require(digest(entry["body_sha256"]) and digest(entry["text_sha256"]), "Supplemental private capture hashes are invalid")
        require(datetime.fromisoformat(entry["captured_at"].replace("Z", "+00:00")).utcoffset() is not None,
                "Supplemental source timestamp requires a timezone")
        for field in ("excerpt", "claim_supported", "publication_date"):
            adjudication.require_text(entry[field])
        require(isinstance(entry["limitations"], list), "Supplemental limitations must be a list")
        for value in entry["limitations"]:
            adjudication.require_text(value)
        require(entry["excerpt_text_normalization"] == "consecutive whitespace collapsed to one space",
                "Unsupported supplemental excerpt normalization")
        result.setdefault(entry["probe_id"], []).append(entry)
    return result


def verify_review_records(review, expected_units, supplemental, *, review_a_path=None,
                          review_b_path=None, resolutions_path=None):
    """Bind every displayed review and override to its frozen public file."""
    manual = {iid: item for iid, item in expected_units.items()
              if item["response"].strip().upper() != "UNKNOWN"}
    inputs = {"manual": manual, "rubric_sha256": review["rubric_sha256"],
              "packet_sha256": review["packet_sha256"]}
    independent = []
    for name, path, field in (("review-a", review_a_path or DEFAULT_REVIEW_A, "review_a_sha256"),
                              ("review-b", review_b_path or DEFAULT_REVIEW_B, "review_b_sha256")):
        require(Path(path).is_file(), "Referenced independent review file is required")
        rows, file_hash = adjudication.load_review(path, name, inputs)
        require(file_hash == review[field], "Independent review byte hash mismatch: " + name)
        independent.append(rows)
    expected_hash = review.get("resolutions_sha256")
    if expected_hash is None:
        require(resolutions_path is None, "Unreferenced resolution file was supplied")
        return independent, {}
    path = resolutions_path or DEFAULT_RESOLUTIONS
    require(Path(path).is_file(), "Referenced resolution file is required")
    document, file_hash = adjudication.read(path)
    require(file_hash == expected_hash, "Resolution evidence byte hash mismatch")
    bindings = ("rubric_sha256", "review_a_sha256", "review_b_sha256", "supplemental_evidence_sha256")
    require(set(document) == {"schema_version", "resolutions", *bindings}
            and type(document["schema_version"]) is int and document["schema_version"] == 1
            and isinstance(document["resolutions"], list), "Unsupported resolution evidence schema")
    require(all(document[field] == review.get(field) for field in bindings),
            "Resolution does not bind to the frozen rubric, independent reviews and supplement")
    resolutions = {}
    for row in document["resolutions"]:
        iid = row.get("item_id")
        require(iid in manual and iid not in resolutions, "Unknown or duplicate frozen resolution item")
        item = manual[iid]
        adjudication.validate_decision(row, item, supplemental.get(item["probe_id"], []))
        resolutions[iid] = row
    return independent, resolutions


def verify_overlay(study_path, review_path, audit_path, rubric_path, *, supplemental_path=None,
                   review_a_path=None, review_b_path=None, resolutions_path=None):
    """Check file bindings, unit identities, review provenance and all joins."""
    study, study_hash = adjudication.read(study_path)
    review, review_hash = adjudication.read(review_path)
    audit, audit_hash = adjudication.read(audit_path)
    rubric, rubric_hash = adjudication.read(rubric_path)
    require(all(type(d.get("schema_version")) is int and d["schema_version"] == 1 for d in (study, review, audit, rubric)),
            "Unsupported study/review/source/rubric schema")
    require(review.get("status") == "COMPLETE_AGENT_REVIEW" and review.get("schema_version") == 1,
            "Only a complete agent-review overlay can be displayed")
    require(study_hash == review.get("study_sha256") == rubric.get("study_sha256"),
            "Original study byte hash differs from the reviewed study")
    require(audit_hash == review.get("source_audit_sha256") == rubric.get("source_audit_sha256") == study.get("source_audit_sha256"),
            "Frozen source audit byte hash mismatch")
    require(rubric_hash == review.get("rubric_sha256"), "Frozen review rubric byte hash mismatch")
    require(study.get("catalog_sha256") == audit.get("catalog_sha256") == rubric.get("catalog_sha256"),
            "Frozen catalog binding mismatch")
    require(set(rubric.get("allowed_verdicts", [])) == set(VERDICT_LABELS), "Unsupported review rubric vocabulary")
    require(review.get("original_key_changed") is False and review.get("verified_cutoff") is None
            and review.get("new_native_provider_requests") == 0, "Review metadata misstates its scope")
    require(all(digest(review.get(k)) for k in ("packet_sha256", "review_a_sha256", "review_b_sha256")),
            "Review evidence hashes are missing or malformed")
    require(review.get("resolutions_sha256") is None or digest(review["resolutions_sha256"]),
            "Resolution evidence hash is malformed")
    require(study.get("publication_status") == "COMPLETE" and study.get("catalog_count") == 129
            and study.get("configuration_count") == 9 and study.get("completed_observations") == 1161,
            "Original study must preserve the complete nine-setting catalog")
    configurations = study.get("configurations", [])
    require([(c.get("provider"), c.get("reasoning")) for c in configurations] == adjudication.SETTINGS,
            "The exact nine settings in fixed treatment order are required")
    source_rows = audit.get("probes", [])
    require(len(source_rows) == 129 and [s.get("catalog_index") for s in source_rows] == list(range(1, 130)),
            "Source audit catalog coverage or order is invalid")
    sources = {s["probe_id"]: s for s in source_rows}
    require(len(sources) == 129, "Source audit probe IDs are not unique")
    supplemental = verify_supplement(supplemental_path, review.get("supplemental_evidence_sha256"), sources)
    for source in source_rows:
        require(source["probe_id"] == adjudication.sha(adjudication.canonical(source["original_probe"])),
                "Source probe ID is not the canonical original-question identity")
    expected_units, expected_observations = {}, {}
    for c in configurations:
        name = c["provider"] + "/" + c["reasoning"]
        require(c.get("configuration") == name and c.get("model") == MODELS[c["provider"]],
                "Native model or configuration identity differs from the frozen treatment")
        rows = c.get("probes", [])
        require(len(rows) == 129 and c.get("completed_observations") == 129 and c.get("full_catalog_completed") is True
                and c.get("unreconciled_launches") == 0, "Native treatment lacks 129 reconciled terminal observations")
        require([r.get("probe_id") for r in rows] == [s["probe_id"] for s in source_rows],
                "Native treatment does not retain the complete common source order")
        observations = []
        for row in rows:
            pid, response = row["probe_id"], row["response"]
            source = sources[pid]
            require(isinstance(response, str) and row["original_probe"] == source["original_probe"]
                    and row["catalog_index"] == source["catalog_index"], "Original response/question source join differs")
            iid = adjudication.sha(adjudication.canonical([pid, response]))
            expected_units[iid] = {"item_id": iid, "probe_id": pid, "catalog_index": row["catalog_index"],
                "question": row["original_probe"]["question"], "response": response,
                "reference_answer": row["original_probe"]["answer"],
                "source_review": {k: source[k] for k in SOURCE_FIELDS}}
            observations.append({"configuration": name, "probe_id": pid, "item_id": iid,
                                 "original_key_status": row["original_key_grade"]["status"]})
        expected_observations[name] = observations
    independent, resolutions = verify_review_records(review, expected_units, supplemental,
        review_a_path=review_a_path, review_b_path=review_b_path, resolutions_path=resolutions_path)
    units, disagreements, overrides, manual, mechanical = {}, 0, 0, 0, 0
    for unit in review.get("units", []):
        iid = unit.get("item_id")
        require(iid in expected_units and iid not in units, "Review contains an unknown or duplicate question/response unit")
        expected = expected_units[iid]
        require({k: unit.get(k) for k in expected} == expected, "Reviewed unit response/question/source identity mismatch")
        extra = supplemental.get(expected["probe_id"], [])
        require(unit.get("supplemental_evidence", []) == extra, "Reviewed supplemental evidence differs from its frozen source document")
        decision, reviewers, origin = unit["decision"], unit["independent_reviews"], unit["origin"]
        require(origin in ORIGINS, "Unknown review origin")
        adjudication.validate_decision(decision, expected, extra)
        if expected["response"].strip().upper() == "UNKNOWN":
            mechanical += 1
            require(origin == "mechanical_exact_unknown" and reviewers == [] and decision == {
                "item_id": iid, "verdict": "abstained", "evidence_capture_ids": [],
                "rationale": "Exact UNKNOWN under the frozen mechanical rule.", "qualifications": []},
                "Exact UNKNOWN was relabeled or falsely described as independent review")
        else:
            manual += 1
            require(isinstance(reviewers, list) and [r.get("reviewer") for r in reviewers] == ["review-a", "review-b"],
                    "Every non-UNKNOWN unit requires both independent reviewer decisions")
            for row in reviewers:
                adjudication.validate_decision({k: v for k, v in row.items() if k != "reviewer"}, expected)
            require(reviewers == [{"reviewer": name, **rows[iid]}
                                  for name, rows in zip(("review-a", "review-b"), independent)],
                    "Embedded decisions differ from the frozen independent reviews")
            left, right = reviewers
            different = left["verdict"] != right["verdict"]
            disagreements += different
            if iid in resolutions:
                require(origin == ("resolved_disagreement" if different else "documented_review_override")
                        and decision == resolutions[iid], "Displayed decision differs from the frozen resolution")
                overrides += not different
            else:
                require(origin == "independent_review_agreement" and not different,
                        "Independent disagreement or override lacks a frozen resolution")
                require(decision == {"item_id": iid, "verdict": left["verdict"],
                    "evidence_capture_ids": sorted(set(left["evidence_capture_ids"] + right["evidence_capture_ids"])),
                    "rationale": "Both independent reviews support this label; their separate rationales are retained below.",
                    "qualifications": list(dict.fromkeys(left["qualifications"] + right["qualifications"]))},
                    "Agreement lost or invented independent evidence, rationale or qualifications")
        units[iid] = unit
    require(set(units) == set(expected_units), "Reviewed units do not cover every original question/response pair")
    require(review.get("unique_pairs") == len(units) and review.get("manual_pairs") == manual
            and review.get("mechanical_unknown_pairs") == mechanical and review.get("observations") == 1161
            and review.get("independent_verdict_disagreements") == disagreements
            and review.get("documented_overrides_of_agreements") == overrides,
            "Review aggregate counts differ from the individual decisions")
    summaries = review.get("configurations", [])
    require([c.get("configuration") for c in summaries] == list(expected_observations),
            "Reviewed setting coverage/order differs from the original study")
    for c, base in zip(summaries, configurations):
        expected = expected_observations[c["configuration"]]
        rows = [{**r, "reviewed_verdict": units[r["item_id"]]["decision"]["verdict"],
                 "review_origin": units[r["item_id"]]["origin"]} for r in expected]
        require(c.get("model") == base["model"] and c.get("reasoning") == base["reasoning"]
                and c.get("observations") == 129 and c.get("probes") == rows,
                "Reviewed 129-row join or original-key grade differs from the original study")
        counts = dict(Counter(r["reviewed_verdict"] for r in rows))
        require(c.get("verdict_counts") == counts, "Displayed verdict counts do not recompute from the reviewed rows")
        downgraded = sum(r["original_key_status"] == "exact_alias_match" and r["reviewed_verdict"] != "supported_core_answer" for r in rows)
        promoted = sum(r["original_key_status"] == "manual_review" and r["reviewed_verdict"] == "supported_core_answer" for r in rows)
        require(c.get("original_matches_not_supported_as_complete") == downgraded
                and c.get("original_unmatched_now_supported") == promoted, "Original-key comparison counts differ")
    return study, review, review_hash


def esc(value):
    return escape(str(value), quote=True)


def source_links(unit, ids):
    evidence = {e["capture_id"]: e for e in unit["source_review"]["evidence"] + unit.get("supplemental_evidence", [])}
    parts = []
    for cid in ids:
        source = evidence[cid]
        url = source.get("url", "")
        link = f"<a href='{esc(url)}' rel='noreferrer'>Source {esc(cid[:8])}</a>" if url.startswith(("https://", "http://")) else f"Source {esc(cid[:8])}"
        parts.append(link)
    return " · ".join(parts)


def qualifications_html(values):
    if not values:
        return ""
    return "<p><strong>Qualifications</strong></p><ul>" + "".join("<li>" + esc(v) + "</li>" for v in values) + "</ul>"


def row_html(unit):
    decision = unit["decision"]
    supported = decision["verdict"] == "supported_core_answer"
    parts = [f"<div class='agent-review' data-reviewed-verdict='{esc(decision['verdict'])}'><span class='badge {'good' if supported else 'flag'}'>Agent review: {esc(VERDICT_LABELS[decision['verdict']])}</span>",
        "<details class='evidence'><summary>Reviewed label, qualifications &amp; independent rationales</summary>",
        "<p class='muted'>" + esc(unit["origin"].replace("_", " ")) + "</p>",
        "<p>" + esc(decision["rationale"]) + "</p>", qualifications_html(decision["qualifications"]),
        "<p>" + source_links(unit, decision["evidence_capture_ids"]) + "</p>"]
    for independent in unit["independent_reviews"]:
        parts.extend(["<p><strong>" + esc(independent["reviewer"]) + " · " + esc(VERDICT_LABELS[independent["verdict"]]) + "</strong></p>",
            "<p>" + esc(independent["rationale"]) + "</p>", qualifications_html(independent["qualifications"]),
            "<p>" + source_links(unit, independent["evidence_capture_ids"]) + "</p>"])
    for extra in unit.get("supplemental_evidence", []):
        parts.extend(["<p><strong>Supplementary evidence added after independent review</strong></p>",
            "<p>" + source_links(unit, [extra["capture_id"]]) + " · published " + esc(extra["publication_date"]) + "</p>",
            "<p>" + esc(extra["claim_supported"]) + "</p><blockquote>" + esc(extra["excerpt"]) + "</blockquote>",
            qualifications_html(extra["limitations"]),
            "<p class='muted'>Captured " + esc(extra["captured_at"]) + ". Retained text SHA-256: " + esc(extra["text_sha256"]) + "</p>"])
    if not unit["independent_reviews"]:
        parts.append("<p>Exact UNKNOWN under the frozen rule; no manual factual verdict was needed.</p>")
    parts.append("</details></div>")
    return "".join(parts)


def overlay_html(review):
    summary = ["<section id='agent-reviewed-results'><h2>Agent-reviewed core answers</h2>",
        "<p>These labels assess the requested historical identity or result against retained sources. Original-key results above remain a separate measurement. All 129 questions remain in each setting.</p>",
        "<p class='notice'>" + esc(adjudication.LIMITATION) + "</p>",
        "<nav class='downloads' aria-label='Agent review downloads'><a href='native-probe-adjudication.json'>Reviewed labels &amp; rationales</a><a href='probe-adjudication-rubric.json'>Frozen review rubric</a>" +
        f"<a href='{original.DOCS}native-probe-adjudication-20260913.md'>Review method &amp; resolutions</a>" +
        ("<a href='probe-adjudication-supplement.json'>Supplementary source evidence</a>" if review.get("supplemental_evidence_sha256") else "") + "</nav>",
        "<p class='muted'>Fixed treatment order; counts use all 129 observations. Exact UNKNOWN responses are mechanical abstentions. " +
        f"Two agents reviewed {review['manual_pairs']} distinct other question/response pairs; {review['independent_verdict_disagreements']} label disagreements received documented resolutions. " +
        f"Documented overrides of initial agreements: {review['documented_overrides_of_agreements']}; the original independent labels remain visible.</p>",
        "<div class='panel scroll'><table id='agent-reviewed-summary'><thead><tr><th>Model</th><th>Reasoning</th>" +
        "".join("<th>" + esc(label) + "</th>" for label in VERDICT_LABELS.values()) + "</tr></thead><tbody>"]
    rows, units = {}, {u["item_id"]: u for u in review["units"]}
    rendered = {iid: row_html(unit) for iid, unit in units.items()}
    for i, c in enumerate(review["configurations"]):
        summary.append(f"<tr><td><a href='#setting-{i}'>{esc(c['model'])}</a></td><td>{esc(c['reasoning'])}</td>" +
            "".join(f"<td class='num'>{c['verdict_counts'].get(key, 0)} / 129</td>" for key in VERDICT_LABELS) + "</tr>")
        rows[c["configuration"]] = {r["probe_id"]: rendered[r["item_id"]] for r in c["probes"]}
    summary.append("</tbody></table></div><p class='muted'>A supported core answer can still carry question defects or unverified extra claims. Expand a response below for both independent rationales and the final decision’s qualifications.</p></section>")
    return {"summary_html": "".join(summary), "rows": rows}


def render(study_path, review_path, audit_path, rubric_path, *, supplemental_path=None,
           review_a_path=None, review_b_path=None, resolutions_path=None):
    study, review, _ = verify_overlay(study_path, review_path, audit_path, rubric_path,
        supplemental_path=supplemental_path, review_a_path=review_a_path,
        review_b_path=review_b_path, resolutions_path=resolutions_path)
    return original.html_report(study, reviewed=overlay_html(review))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "data/research/native-probes-20260913.json")
    parser.add_argument("--review", type=Path, default=ROOT / "data/research/native-probe-adjudication-20260913.json")
    parser.add_argument("--source-audit", type=Path, default=ROOT / "data/research/probe-source-audit-20260913.json")
    parser.add_argument("--rubric", type=Path, default=ROOT / "data/research/probe-adjudication-rubric-20260913.json")
    parser.add_argument("--review-a", type=Path, default=DEFAULT_REVIEW_A)
    parser.add_argument("--review-b", type=Path, default=DEFAULT_REVIEW_B)
    parser.add_argument("--resolutions", type=Path, help="Defaults to the public resolution file when the review references one")
    parser.add_argument("--supplemental-evidence", type=Path, help="Required when the final review binds supplementary evidence")
    parser.add_argument("--output", type=Path, default=ROOT / "dash/factual-recall.html")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    inputs = [p.resolve() for p in (args.study, args.review, args.source_audit, args.rubric,
                                   args.review_a, args.review_b, args.resolutions or DEFAULT_RESOLUTIONS)]
    if args.supplemental_evidence is not None:
        inputs.append(args.supplemental_evidence.resolve())
    if output in inputs or (output.is_relative_to(ROOT.resolve()) and output != (ROOT / "dash/factual-recall.html").resolve()):
        parser.error("Only the separate factual-recall page or an external local preview may be written")
    html = render(args.study, args.review, args.source_audit, args.rubric,
                  supplemental_path=args.supplemental_evidence, review_a_path=args.review_a,
                  review_b_path=args.review_b, resolutions_path=args.resolutions)
    original.probes.atomic_write(output, html)
    print("Rendered validated agent-review labels; original input files were not rewritten.")


if __name__ == "__main__":
    main()
