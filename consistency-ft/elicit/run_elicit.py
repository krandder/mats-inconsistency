#!/usr/bin/env python3
"""Compare verbalized and True/False-logprob probability readouts."""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL, load_tuples, make_client, proposition_text  # noqa: E402


def pearson(xs, ys):
    mx, my = fmean(xs), fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else float("nan")


def ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in ordered[start:end]:
            result[index] = rank
        start = end
    return result


def base_propositions(tuple_dir, limit):
    seen, propositions = set(), []
    for row in load_tuples(tuple_dir):
        for key in ("P", "Q"):
            question = row["questions"].get(key)
            if question:
                identity = question.get("id") or proposition_text(question)
                if identity not in seen:
                    seen.add(identity)
                    propositions.append(question)
                    if limit is not None and len(propositions) >= limit:
                        return propositions
    return propositions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuples", default="data/tuples")
    parser.add_argument("--output")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    propositions = base_propositions(args.tuples, args.limit)
    client = make_client(args.mock, args.model)
    texts = [proposition_text(question) for question in propositions]
    def safe(fn):
        def call(text):
            try:
                return fn(text)
            except Exception as exc:  # ponytail: skip bad replies, don't kill the run
                print(f"WARN: {exc}", file=sys.stderr)
                return None
        return call

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        verbal = list(pool.map(safe(client.verbal_probability), texts))
        logprob = list(pool.map(safe(client.true_false_probability), texts))
    keep = [i for i, (v, lp) in enumerate(zip(verbal, logprob)) if v is not None and lp is not None]
    if len(keep) < len(texts):
        print(f"Skipped {len(texts) - len(keep)} propositions", file=sys.stderr)
    propositions = [propositions[i] for i in keep]
    verbal = [verbal[i] for i in keep]
    logprob = [logprob[i] for i in keep]

    output = Path(args.output or f"results/elicit_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    differences = []
    with output.open("w") as f:
        for question, v, lp in zip(propositions, verbal, logprob):
            difference = abs(v - lp)
            differences.append(difference)
            f.write(json.dumps({"id": question.get("id"), "title": question["title"],
                                "verbalized_probability": v, "logprob_probability": lp,
                                "abs_diff": difference, "model": client.model},
                               separators=(",", ":")) + "\n")

    print(f"Wrote {len(propositions)} results to {output}")
    print(f"Pearson correlation:  {pearson(verbal, logprob):.6f}")
    print(f"Spearman correlation: {pearson(ranks(verbal), ranks(logprob)):.6f}")
    print(f"Mean |difference|:    {fmean(differences):.6f}")
    print(f"Disagreement > 0.2:   {sum(d > 0.2 for d in differences)/len(differences):.3f}")


if __name__ == "__main__":
    main()
