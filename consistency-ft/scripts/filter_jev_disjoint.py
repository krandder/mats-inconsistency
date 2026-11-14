#!/usr/bin/env python3
"""Filter a frozen JEV pair export by excluded parent IDs.

No new pair edges are introduced: a pair is dropped when either proposition's
parent is excluded; propositions, queries, targets and the hidden-label audit are
restricted to the surviving pairs.  The output directory must not exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SPLITS = ("train", "validation", "test")
PAIR_FILES = ("queries.jsonl", "one-label-targets.jsonl", "full-label-targets.jsonl", "hidden-label-audit.jsonl")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def filter_split(source: Path, target: Path, excluded: set[str]) -> dict:
    target.mkdir(parents=True)
    pairs = _jsonl(source / "pairs.jsonl")
    kept = [p for p in pairs if p["p_parent"] not in excluded and p["q_parent"] not in excluded]
    kept_ids = {p["pair_id"] for p in kept}
    kept_props = {p["p_id"] for p in kept} | {p["q_id"] for p in kept}
    propositions = [r for r in _jsonl(source / "propositions.jsonl") if r["id"] in kept_props]
    if {r["id"] for r in propositions} != kept_props:
        raise ValueError(f"{source}: surviving pairs reference missing propositions")
    _write(target / "pairs.jsonl", kept)
    _write(target / "propositions.jsonl", propositions)
    counts = {"pairs": len(kept), "removed_pairs": len(pairs) - len(kept), "propositions": len(propositions)}
    query_pair: dict[str, str] = {}
    for name in PAIR_FILES:
        rows = _jsonl(source / name)
        if name == "queries.jsonl":
            rows = [r for r in rows if r["pair_id"] in kept_ids]
            query_pair = {r["query_id"]: r["pair_id"] for r in rows}
            counts["expressions"] = len(rows)
        elif name == "hidden-label-audit.jsonl":
            rows = [r for r in rows if r.get("pair_id", query_pair.get(r.get("query_id"))) in kept_ids]
        else:
            rows = [r for r in rows if query_pair.get(r["query_id"]) is not None]
        _write(target / name, rows)
        counts[name] = len(rows)
    singles = [r for r in _jsonl(source / "single-targets.jsonl") if r["id"] in kept_props]
    _write(target / "single-targets.jsonl", singles)
    counts["single-targets.jsonl"] = len(singles)
    audit = [r for r in _jsonl(target / "hidden-label-audit.jsonl") if "entailed_by_observed_label" in r]
    if audit:  # the disjoint export carries per-query entailment; the pairings export uses another schema
        counts["entailed_hidden"] = sum(r["entailed_by_observed_label"] for r in audit)
        counts["underdetermined_hidden"] = sum(not r["entailed_by_observed_label"] for r in audit)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--excluded", required=True, help="JSON list of parent IDs to exclude")
    parser.add_argument("--also-exclude", default=None, help="JSON file whose 'uncertain_parent_ids' are excluded too")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise SystemExit(f"{output} already exists; outputs are never overwritten")
    excluded = set(json.loads(Path(args.excluded).read_text()))
    policy = "exclude pairs whose P or Q parent is in the explicit/suspicious leakage list"
    if args.also_exclude:
        uncertain = set(json.load(open(args.also_exclude))["uncertain_parent_ids"])
        excluded |= uncertain
        policy += f"; also exclude {len(uncertain)} uncertain parents (Kelvin, 2026-09-19)"
    output.mkdir(parents=True)
    counts = {split: filter_split(source / split, output / split, excluded) for split in SPLITS}
    files = {str(p.relative_to(output)): _sha(p) for p in sorted(output.rglob("*.jsonl"))}
    manifest = {"version": args.version, "source_dataset": str(source),
                "source_manifest_sha256": _sha(source / "manifest.json"),
                "exclusion_policy": policy, "excluded_parent_count": len(excluded),
                "excluded_parent_ids_sha256": hashlib.sha256(json.dumps(sorted(excluded)).encode()).hexdigest(),
                "selection_method": "filter existing frozen export; no new pair edges introduced",
                "training_ready": False,
                "caveat": "historical text edits are only partially screened; family correlations remain",
                "counts": counts, "files_sha256": files}
    (output / "excluded-parent-ids.json").write_text(json.dumps(sorted(excluded), indent=1) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
