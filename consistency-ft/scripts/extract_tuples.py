#!/usr/bin/env python3
"""Extract Paleka et al.'s published 2028 instantiated check tuples."""

import argparse
import json
from pathlib import Path


SOURCES = {
    "Negation": "NegChecker.jsonl",
    "Paraphrase": "ParaphraseChecker.jsonl",
    "Consequence": "ConsequenceChecker.jsonl",
    "And": "AndChecker.jsonl",
    "Or": "OrChecker.jsonl",
    "AndOr": "AndOrChecker.jsonl",
    "But": "ButChecker.jsonl",
    "Cond": "CondChecker.jsonl",
    "CondCond": "CondCondChecker.jsonl",
    "ExpEvidence": "ExpectedEvidenceChecker.jsonl",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("vendor/consistency-forecasting/src/data/tuples/2028"))
    parser.add_argument("--output", type=Path, default=Path("data/tuples"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for check_type, filename in SOURCES.items():
        source = args.source / filename
        if not source.exists():
            raise SystemExit(f"Missing {source}; clone dpaleka/consistency-forecasting into vendor/ first")
        count = 0
        with source.open() as src, (args.output / f"{check_type}.jsonl").open("w") as dst:
            for count, line in enumerate(src, 1):
                original = json.loads(line)
                questions = {
                    key: {field: value.get(field) for field in ("id", "title", "body")}
                    for key, value in original.items()
                    if key != "metadata" and isinstance(value, dict) and value.get("title")
                }
                record = {
                    "id": f"2028/{filename.removesuffix('.jsonl')}:{count}",
                    "check_type": check_type,
                    "questions": questions,
                }
                dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"{check_type:12} {count:5d}")


if __name__ == "__main__":
    main()
