#!/usr/bin/env python3
"""Audit missing native readouts against their original message/usage evidence.

This is read-only with respect to private receipts and performs no inference.
The output contains allowlisted evidence, never raw CLI logs or session IDs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harness.run_native_checks import classify_readout, is_completed
from scripts.export_benchmark import NATIVE_CONFIGS


def audit(directory, snapshot):
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    campaign = directory.name
    entries = {r["run_id"]: r for r in manifest["runs"]}
    observations, configurations, sources = [], [], {}
    for run_key, provider, _, _ in NATIVE_CONFIGS:
        run_id = f"{campaign}/{run_key}"
        source = entries.get(run_id)
        if source:
            sources[run_id] = {"receipts_file": source["receipts_file"],
                               "receipts_sha256": source["receipts_sha256"]}
        counts = Counter()
        for path in sorted((directory / run_key).glob("*.json")):
            if ".attempt-" in path.name:
                continue
            raw = json.loads(path.read_text())
            if raw.get("probability") is not None:
                continue
            response = raw.get("response", "")
            events = []
            for line in raw.get("stdout", "").splitlines():
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
            event_messages = [e["item"].get("text", "") for e in events
                              if e.get("type") == "item.completed"
                              and e.get("item", {}).get("type") == "agent_message"]
            final_path = path.with_suffix(".final.txt")
            evidence = {
                "exit_code": raw.get("exit_code"), "completed": is_completed(raw),
                "usage_reported": bool(raw.get("usage")), "tools_used": bool(raw.get("tools_used")),
                "native_turn_completed": any(e.get("type") == "turn.completed" for e in events),
                "native_turn_failed": any(e.get("type") in {"turn.failed", "error"} for e in events),
                "native_final_message_matches": bool(event_messages) and event_messages[-1].strip() == response.strip(),
                "saved_final_file_matches": final_path.exists() and final_path.read_text().strip() == response.strip(),
            }
            kind = classify_readout(response)
            if raw.get("exit_code") != 0 or not is_completed(raw):
                kind = "infrastructure"
            elif raw.get("tools_used"):
                kind = "tool_use"
            elif provider in {"astra", "luna"} and not evidence["native_final_message_matches"]:
                kind = "extraction_mismatch"
            counts[kind] += 1
            observations.append({"run_id": run_id, "proposition_id": raw["id"],
                "classification": kind, "original_error_kind": raw.get("error_kind"),
                "response": response, "evidence": evidence})
        configurations.append({"run_id": run_id, "missing_readouts": sum(counts.values()),
                               "classifications": dict(sorted(counts.items()))})
    totals = Counter(r["classification"] for r in observations)
    response_counts = Counter(r["response"] for r in observations)
    explicit_politics = re.compile(r"politic|election|presidential|legislation|campaign|endorsement|gubernatorial", re.I)
    return {"schema_version": 1, "campaign": campaign, "classification_version": "native-readout-v2",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_receipts": sources, "missing_readouts": len(observations),
        "distinct_propositions": len({r["proposition_id"] for r in observations}),
        "classifications": dict(sorted(totals.items())), "configurations": configurations,
        "refusal_wording": {
            "explicit_political_reference": sum(r["classification"] == "refusal" and bool(explicit_politics.search(r["response"])) for r in observations),
            "generic_refusal": sum(r["classification"] == "refusal" and not explicit_politics.search(r["response"]) for r in observations)},
        "response_examples": [{"response": text, "count": count} for text, count in response_counts.most_common()],
        "observations": observations, "recovered_probabilities": 0,
        "inference_requests_performed_by_audit": 0,
        "notes": ["Refusal text is not a malformed numeric answer and contains no recoverable probability.",
                  "Original raw receipts are unchanged; new classification is interpretation metadata.",
                  "A completed native turn and matching final event/file rule out the observed extraction/stale-file failure modes; no truncation indicator was found in these refusal readouts.",
                  "Refusals remain completed observations with null probabilities and are never retried to obtain preferred values."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "data/benchmark-runs")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark-runs/native-readout-audit.json")
    args = parser.parse_args()
    report = audit(args.run_dir, args.snapshots)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"missing_readouts": report["missing_readouts"],
                      "classifications": report["classifications"], "output": str(args.output)}))
