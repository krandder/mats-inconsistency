#!/usr/bin/env python3
"""Build deterministic train/validation/held-out episode files."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHECK_FIELDS, load_tuples, proposition_text  # noqa: E402


TRAIN_TYPES = {"Negation", "And", "Or", "AndOr", "But", "Cond"}
HELDOUT_TYPES = {"Paraphrase", "Consequence", "CondCond", "ExpEvidence"}


def split(row):
    if row["check_type"] in HELDOUT_TYPES:
        return "heldout"
    assert row["check_type"] in TRAIN_TYPES
    bucket = int.from_bytes(hashlib.sha256(row["id"].encode()).digest()[:8], "big") % 10
    return "valid" if bucket == 0 else "train"


def episode(row):
    return {
        "id": row["id"],
        "check_type": row["check_type"],
        "prompts": [
            {"role": role, "text": proposition_text(row["questions"][role])}
            for role in CHECK_FIELDS[row["check_type"]]
        ],
        "constraint": row["check_type"],
    }


def main():
    output = Path("data/episodes")
    output.mkdir(parents=True, exist_ok=True)
    records = {name: [] for name in ("train", "valid", "heldout")}
    for row in load_tuples("data/tuples"):
        records[split(row)].append(episode(row))

    for name, rows in records.items():
        with (output / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        counts = Counter(row["check_type"] for row in rows)
        detail = ", ".join(f"{check_type}={counts[check_type]}" for check_type in CHECK_FIELDS if counts[check_type])
        print(f"{name}.jsonl: {len(rows)} ({detail})")

    ids = [{row["id"] for row in rows} for rows in records.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert {row["check_type"] for row in records["heldout"]} <= HELDOUT_TYPES
    print("PASS: episode ids do not overlap; heldout check types are frozen")


if __name__ == "__main__":
    main()
