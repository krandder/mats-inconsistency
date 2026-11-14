"""Validate a reviewed whole-tuple scope against the original source catalog."""
import hashlib
import json
from pathlib import Path

from common import proposition_text


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def validate_scope(path, source_tuples, tuple_directory=None):
    raw_bytes = Path(path).read_bytes()
    manifest = json.loads(raw_bytes)
    if not isinstance(manifest.get("scope_id"), str) or not manifest["scope_id"]:
        raise ValueError("scope_id is required")
    by_id = {row["id"]: row for row in source_tuples}
    included = manifest.get("included_tuple_ids", [])
    if not included or len(included) != len(set(included)) or set(included) - set(by_id):
        raise ValueError("scope must identify unique, known, nonempty whole tuples")
    excluded = manifest.get("excluded_tuple_ids", [])
    if set(excluded) != set(by_id) - set(included) or len(excluded) != len(set(excluded)):
        raise ValueError("scope exclusions must exactly cover the remaining source tuples")
    selected = [row for row in source_tuples if row["id"] in set(included)]
    props = {sha256(proposition_text(q).encode()) for row in selected for q in row["questions"].values()}
    if set(manifest.get("included_proposition_ids", [])) != props:
        raise ValueError("scope proposition IDs must equal the complete included-tuple union")
    source = manifest.get("source_cohort", {})
    source_props = {sha256(proposition_text(q).encode()) for row in source_tuples for q in row["questions"].values()}
    if source.get("tuple_count") != len(source_tuples) or source.get("proposition_count") != len(source_props):
        raise ValueError("scope source cohort counts do not match")
    if tuple_directory is not None:
        files = {p.name: sha256(p.read_bytes()) for p in Path(tuple_directory).glob("*.jsonl")}
        fingerprints = {Path(name).name: value for name, value in source.get("source_file_sha256", {}).items()}
        if fingerprints != files:
            raise ValueError("scope source file fingerprints do not match")
    reasons = {row["tuple_id"]: row["reason"] for row in manifest.get("exclusions", [])}
    if set(reasons) != set(excluded) or any(not reason for reason in reasons.values()):
        raise ValueError("every scope exclusion requires a source-based reason")
    return {"manifest": manifest, "sha256": sha256(raw_bytes), "bytes": raw_bytes,
            "tuples": selected, "proposition_ids": props, "excluded_tuple_ids": set(excluded)}
