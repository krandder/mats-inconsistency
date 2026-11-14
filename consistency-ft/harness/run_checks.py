#!/usr/bin/env python3
"""Run basic probabilistic-consistency checks over extracted tuples."""

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHECK_FIELDS, MODEL, load_tuples, make_client, proposition_text, violation  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuples", default="data/tuples")
    parser.add_argument("--output")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, help="maximum tuples per check type")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--exploit", action="store_true", help="include arbitrage exploitability")
    args = parser.parse_args()
    if args.exploit:
        from metrics import exploit

    tuples = load_tuples(args.tuples, args.limit)
    client = make_client(args.mock, args.model)
    texts = sorted({proposition_text(q) for row in tuples for q in row["questions"].values()})

    def probability_or_none(text):
        try:
            return client.verbal_probability(text)
        except Exception as exc:  # ponytail: one bad reply must not kill a 3k-tuple sweep
            print(f"WARN: skipping proposition ({exc})", file=sys.stderr)
            return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        probabilities = dict(zip(texts, pool.map(probability_or_none, texts)))

    output = Path(args.output or f"results/checks_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = defaultdict(list)
    exploit_summaries = defaultdict(list)
    with output.open("w") as f:
        skipped = 0
        for row in tuples:
            fields = CHECK_FIELDS[row["check_type"]]
            probs = {key: probabilities[proposition_text(row["questions"][key])] for key in fields}
            if any(value is None for value in probs.values()):
                skipped += 1
                continue
            magnitude = violation(row["check_type"], probs)
            result = {"id": row["id"], "check_type": row["check_type"], "probabilities": probs,
                      "violation": magnitude, "model": client.model}
            if args.exploit:
                result["exploit"] = exploit(row["check_type"], probs)
                exploit_summaries[row["check_type"]].append(result["exploit"])
            f.write(json.dumps(result, separators=(",", ":")) + "\n")
            summaries[row["check_type"]].append(magnitude)

    if skipped:
        print(f"Skipped {skipped} tuples with unparseable member probabilities", file=sys.stderr)
    print(f"Wrote {len(tuples) - skipped} results to {output}")
    exploit_header = f" {'mean_exploit':>16}" if args.exploit else ""
    print(f"{'check_type':12} {'n':>5} {'mean_violation':>16}{exploit_header} {'frac_violated_>0.05':>20}")
    for name in CHECK_FIELDS:
        values = summaries.get(name)
        if values:
            exploit_column = (f" {sum(exploit_summaries[name])/len(values):16.6f}"
                              if args.exploit else "")
            print(f"{name:12} {len(values):5d} {sum(values)/len(values):16.6f}"
                  f"{exploit_column} {sum(v > 0.05 for v in values)/len(values):20.3f}")


if __name__ == "__main__":
    main()
