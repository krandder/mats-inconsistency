#!/usr/bin/env python3
"""Derive explicit CPU collection scenarios from retained corpus and timing data.

No model import, generation, network call or training. The transferred timing
coefficients are illustrative scenarios, not confidence bounds or a fitted
throughput model. Raw tokenizer records must match the published preflight pin.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {name: (ROOT / name).read_bytes() for name in (
        "data/episodes/train.jsonl",
        "data/research/qwen-tokenizer-preflight-20260913.json",
        "data/research/qwen-candidate-execution-20260913/readouts.jsonl",
        "data/research/qwen-candidate-execution-20260913/token-events.jsonl",
        "data/research/qwen-numerical-backward-20260913/phases.jsonl")}
    train_raw = inputs["data/episodes/train.jsonl"]
    preflight = json.loads(inputs["data/research/qwen-tokenizer-preflight-20260913.json"])
    records_raw = args.prompt_records.read_bytes()
    pin = preflight["evidence_artifacts"]["prompts.jsonl"]
    if (len(records_raw) != pin["bytes"] or sha256(records_raw) != pin["sha256"]
            or sha256(train_raw) != preflight["split_sha256"]["train"]):
        raise ValueError("tokenizer evidence or original split differs from published pins")
    episodes = [json.loads(line) for line in train_raw.splitlines()]
    propositions = {sha256(row["text"].encode()): row["text"]
                    for episode in episodes for row in episode["prompts"]}
    records = [row for row in map(json.loads, records_raw.splitlines())
               if row["treatment"] == "proposed-native-numeric-v1"
               and row["proposition_sha256"] in propositions]
    if (len(episodes) != 1619 or len(propositions) != 2926 or len(records) != 2926
            or {row["proposition_sha256"] for row in records} != propositions.keys()
            or any(row["prompt_tokens"] != len(row["prompt_token_ids"])
                   or row["messages"][1][1] != preflight["treatments"]["proposed-native-numeric-v1"]
                   ["protocol"]["verbal_prompt"].format(proposition=propositions[row["proposition_sha256"]])
                   for row in records)):
        raise ValueError("complete unique training-proposition join differs")
    lengths = sorted(row["prompt_tokens"] for row in records)
    readouts = [json.loads(line) for line in inputs[
        "data/research/qwen-candidate-execution-20260913/readouts.jsonl"].splitlines()]
    events = [json.loads(line) for line in inputs[
        "data/research/qwen-candidate-execution-20260913/token-events.jsonl"].splitlines()]
    by_role = {row["role"]: row for row in readouts}
    coefficients = []
    for event in events:
        row = by_role[event["role"]]
        index = event["index"]
        if (event["token_id"] != row["completion_token_ids"][index]
                or event["logprob"] != row["token_logprobs"][index]):
            raise ValueError("retained timing event differs from generated readout")
        coefficients.append(event["wall_seconds"] / (len(row["prompt_token_ids"]) + index))
    if len(events) != 13 or len(readouts) != 3:
        raise ValueError("unexpected calibration scope")
    phases = {row["phase"]: row for row in map(json.loads, inputs[
        "data/research/qwen-numerical-backward-20260913/phases.jsonl"].splitlines())}
    audit_seconds = (phases["integrity_audit_complete"]["wall_seconds"]
                     - phases["integrity_audit_start"]["wall_seconds"])
    scenarios = []
    for completion_tokens in (4, 8, 16, 32):
        prefix_tokens = sum(n * completion_tokens + completion_tokens * (completion_tokens - 1) // 2
                            for n in lengths)
        scenarios.append({"assumed_completion_tokens_per_proposition_including_eos": completion_tokens,
                          "total_generated_tokens": len(lengths) * completion_tokens,
                          "total_causal_prefix_tokens_processed_without_cache": prefix_tokens,
                          "generation_hours_at_min_observed_coefficient": prefix_tokens * min(coefficients) / 3600,
                          "generation_hours_at_max_observed_coefficient": prefix_tokens * max(coefficients) / 3600})
    result = {
        "schema_version": 1,
        "scope": "Conditional resource arithmetic for complete original own-base collection; no execution or statistical forecast.",
        "original_training_episodes": len(episodes), "distinct_training_propositions": len(lengths),
        "prompt_tokens": {"sum": sum(lengths), "min": min(lengths), "max": max(lengths),
                          "mean": sum(lengths) / len(lengths),
                          "nearest_rank_percentiles": {str(p): lengths[math.ceil(len(lengths) * p / 100) - 1]
                                                       for p in (50, 90, 95, 99)}},
        "prompt_records_sha256": sha256(records_raw),
        "source_sha256": {**{name: sha256(raw) for name, raw in inputs.items()},
                          "scripts/estimate_qwen_anchor_resources.py": sha256(Path(__file__).read_bytes())},
        "observations": {"selected_prompts": len(readouts), "token_timings": len(events),
                         "cpu_threads": 2, "cache": False,
                         "seconds_per_prefix_token_min": min(coefficients),
                         "seconds_per_prefix_token_max": max(coefficients),
                         "separate_combined_integrity_audit_seconds": audit_seconds},
        "scenarios": scenarios,
        "additional_integrity_proxy": {"assumption": "One audit equal to the separate observed combined audit per proposition.",
                                       "hours": len(lengths) * audit_seconds / 3600,
                                       "measured_collection_overhead": False},
        "limitations": [
            "Three selected prompts and 13 tokens do not establish population throughput, error rate or confidence bounds.",
            "Constant seconds per causal-prefix token is an explicit extrapolation; hardware load and kernels may differ.",
            "Generation calibration used the frozen base without installed LoRA; collection overhead must be measured.",
            "Scenarios assume identical completion length for every proposition; 32 is a cap scenario, not valid coverage.",
            "Process reloads, token/receipt fsync, tokenizer verification, adapter work and retries are excluded.",
            "No replacement samples are authorized by this arithmetic; invalid or interrupted attempts remain missing anchors.",
            "The integrity figure is a transferred proxy, not a measured per-readout collector cost.",
            "No current dollar price, paid resource allowance, scientific treatment selection or training time is inferred."],
        "provider_inference_requests": 0, "new_paid_resources_allocated": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"propositions": len(lengths), "prompt_tokens": sum(lengths),
                      "four_token_scenario_hours": [scenarios[0]["generation_hours_at_min_observed_coefficient"],
                                                     scenarios[0]["generation_hours_at_max_observed_coefficient"]]}))


if __name__ == "__main__":
    main()
