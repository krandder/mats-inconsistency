#!/usr/bin/env python3
"""Measure the complete original corpus with pinned Qwen tokenization, no model."""

import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import resource
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import VERBAL_PROMPT
from harness.run_native_checks import PROTOCOLS
from training.qwen_tokenizer import (CompletionDecodeError, MANIFEST_PATH, NONTHINKING_SUFFIX,
                                      QwenTextTokenizer, canonical, sha256)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def download_artifacts(directory):
    """Fetch three hash-pinned public data files only, without authentication."""
    directory.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST_PATH.read_text())
    for name, expected in manifest["files"].items():
        with urllib.request.urlopen(expected["url"], timeout=120) as response:
            raw = response.read(expected["bytes"] + 1)
        if len(raw) != expected["bytes"] or sha256(raw) != expected["sha256"]:
            raise ValueError("public artifact content differs: " + name)
        (directory / name).write_bytes(raw)


def distribution(values):
    ordered = sorted(values)
    return {"count": len(ordered), "sum": sum(ordered), "min": ordered[0], "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
            "nearest_rank_percentiles": {str(p): ordered[max(0, math.ceil(p * len(ordered) / 100) - 1)]
                                          for p in (50, 90, 95, 99)}}


def collect(output, artifacts):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    tokenizer = QwenTextTokenizer(artifacts)
    before = tokenizer.verify_runtime()
    protocol = PROTOCOLS["numeric-v1"]
    if protocol["verbal_prompt"] != VERBAL_PROMPT:
        raise ValueError("named native numeric-v1 no longer shares the original verbal prompt")
    treatments = {"proposed-native-numeric-v1": {"protocol_id": "numeric-v1", "protocol": protocol,
                     "roles": ["system", "user"], "scientific_selection": False},
                  "original-api-user-only": {"protocol_id": "original-api-verbal-user-only",
                     "protocol": {"verbal_prompt": VERBAL_PROMPT}, "roles": ["user"],
                     "comparison_only": True,
                     "note": "Same original user-message bytes; local Qwen template, not proof of hosted Fireworks rendering."}}
    for treatment in treatments.values():
        treatment["protocol_sha256"] = sha256(canonical(treatment["protocol"]))
    write_json(output / "treatments.json", treatments)
    (output / "chat-template.jinja").write_text(tokenizer.chat_template)
    cache, stats, split_hashes, propositions = {}, {}, {}, {}
    with (output / "prompts.jsonl").open("xb") as prompts, (output / "episode-lengths.jsonl").open("xb") as ledger:
        for split in ("train", "valid", "heldout"):
            path = ROOT / "data" / "episodes" / (split + ".jsonl")
            raw = path.read_bytes()
            split_hashes[split] = sha256(raw)
            episodes = [json.loads(line) for line in raw.splitlines() if line.strip()]
            propositions[split] = {sha256(prompt["text"].encode()) for episode in episodes for prompt in episode["prompts"]}
            sizes = Counter(len(episode["prompts"]) for episode in episodes)
            split_stats = {"episodes": len(episodes), "roles": sum(size * count for size, count in sizes.items()),
                           "unique_propositions": len(propositions[split]), "group_size_histogram": dict(sorted(sizes.items())),
                           "check_type_counts": dict(sorted(Counter(e["check_type"] for e in episodes).items())),
                           "treatments": {}}
            for treatment, definition in treatments.items():
                lengths, sums, maxima, by_check = [], [], [], {}
                for episode in episodes:
                    rows = []
                    for prompt in episode["prompts"]:
                        text = prompt["text"]
                        ident = sha256(text.encode())
                        key = (treatment, ident)
                        if key not in cache:
                            user = VERBAL_PROMPT.format(proposition=text)
                            messages = (("user", user),) if definition["roles"] == ["user"] else (
                                ("system", protocol["system_instruction"]), ("user", user))
                            rendered, tokens = tokenizer.render_and_encode(messages,
                                                               original_user_only=definition["roles"] == ["user"])
                            record = {"treatment": treatment, "proposition_sha256": ident,
                                      "messages": messages, "messages_sha256": sha256(canonical(messages)),
                                      "rendered_prompt": rendered, "rendered_prompt_sha256": sha256(rendered.encode()),
                                      "prompt_token_ids": tokens, "prompt_token_ids_sha256": sha256(canonical(tokens)),
                                      "prompt_tokens": len(tokens), "exact_roundtrip": True}
                            prompts.write(canonical(record) + b"\n")
                            cache[key] = {k: record[k] for k in ("proposition_sha256", "prompt_tokens", "rendered_prompt_sha256", "prompt_token_ids_sha256")}
                        row = {"role": prompt["role"], **cache[key]}
                        rows.append(row)
                    counts = [row["prompt_tokens"] for row in rows]
                    lengths.extend(counts)
                    sums.append(sum(counts))
                    maxima.append(max(counts))
                    by_check.setdefault(episode["check_type"], []).extend(counts)
                    ledger.write(canonical({"split": split, "treatment": treatment, "episode_id": episode["id"],
                                             "check_type": episode["check_type"], "role_count": len(rows),
                                             "sum_prompt_tokens_one_joint_sample": sum(counts), "roles": rows}) + b"\n")
                split_stats["treatments"][treatment] = {"role_prompt_tokens": distribution(lengths),
                    "joint_prompt_token_sum": distribution(sums), "joint_max_role_prompt_tokens": distribution(maxima),
                    "role_prompt_tokens_by_check_type": {check: distribution(values) for check, values in sorted(by_check.items())}}
            stats[split] = split_stats
    controls = {}
    prefix = tokenizer.encode_text("0.5")
    for text in ("0.5", "0.5 commentary", "<think>reasoning</think>0.5", "ABSTAIN", "0.5<|endoftext|>"):
        ids = tokenizer.encode_text(text) + (tokenizer.eos_token_id,)
        controls[text] = {"fixture_only": True, "token_ids": ids, "decoded_completion": tokenizer.decode_completion(ids)}
    backend = tokenizer._tokenizer
    vocabulary = set(backend.get_vocab().values())
    model_vocab = tokenizer.model_config["text_config"]["vocab_size"]
    unmapped = [token for token in range(model_vocab) if token not in vocabulary]
    tail_rows = []
    for token in unmapped:
        raw = backend.decode(prefix + (token,), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        try:
            tokenizer.decode_completion(prefix + (token,))
        except CompletionDecodeError as error:
            rejected = error.completion_token_ids == prefix + (token,)
        else:
            rejected = False
        tail_rows.append({"token_id": token, "raw_backend_decode_of_numeric_fixture_plus_id": raw,
                          "adapter_rejected_and_retained_exact_ids": rejected})
    write_json(output / "unmapped-decoder-audit.json", {"fixture_only": True, "all_unmapped_model_output_ids": tail_rows})
    if not all(row["adapter_rejected_and_retained_exact_ids"] for row in tail_rows):
        raise ValueError("unmapped ID silently survived proposed completion contract")
    after = tokenizer.verify_runtime()
    if before != after:
        raise ValueError("tokenizer runtime changed during corpus measurement")
    for split, expected in split_hashes.items():
        if sha256((ROOT / "data/episodes" / (split + ".jsonl")).read_bytes()) != expected:
            raise ValueError("dataset bytes changed during measurement")
    import transformers.models.qwen2.tokenization_qwen2 as qwen_source
    import transformers.tokenization_utils_base as base_source
    import transformers.tokenization_utils_tokenizers as backend_source
    import transformers.utils.chat_template_utils as template_source
    sources = {name: sha256((ROOT / name).read_bytes()) for name in (
        "training/qwen_tokenizer.py", "scripts/preflight_qwen_tokenizer.py", "harness/run_native_checks.py",
        "common.py", "data/research/qwen-tokenizer-artifacts-20260913.json",
        "training/requirements-qwen-tokenizer-py312-linux-x86_64.lock")}
    runtime_sources = {module.__name__: sha256(Path(inspect.getfile(module)).read_bytes())
                       for module in (qwen_source, base_source, backend_source, template_source)}
    report = {"schema_version": 1, "status": "tokenizer-preflight-complete", "model_weights_loaded": False,
              "generated_answers": 0, "anchors_generated": 0, "scientific_checkpoint_selected": False,
              "artifact_manifest": json.loads(MANIFEST_PATH.read_text()), "runtime_identity": before,
              "source_sha256": sources, "runtime_source_sha256": runtime_sources,
              "python": sys.version.split()[0], "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
              "treatments": treatments, "split_sha256": split_hashes, "splits": stats,
              "unique_propositions_all_splits": len(set.union(*propositions.values())),
              "unique_train_anchor_propositions_required": len(propositions["train"]),
              "prompt_records": len(cache), "all_prompt_roundtrips_exact": True,
              "prompt_suffix": NONTHINKING_SUFFIX,
              "tokenizer_base_vocab_size": backend.vocab_size, "tokenizer_total_vocab_size": len(backend),
              "model_output_vocab_size": model_vocab, "tokenizer_eos_token_id": tokenizer.eos_token_id,
              "tokenizer_pad_token_id": tokenizer.pad_token_id,
              "model_config_eos_token_id": tokenizer.model_config["text_config"]["eos_token_id"],
              "unmapped_output_id_count": len(unmapped), "all_unmapped_ids_rejected_with_raw_ids_retained": True,
              "stock_decoder_silently_dropped_all_unmapped_ids_after_numeric_fixture": all(
                  row["raw_backend_decode_of_numeric_fixture_plus_id"] == "0.5" for row in tail_rows),
              "decoder_fixtures": controls,
              "wall_seconds_after_imports": time.monotonic() - started,
              "peak_rss_bytes_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    report["evidence_artifacts"] = {p.name: {"sha256": sha256(p.read_bytes()), "bytes": p.stat().st_size}
                                    for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--artifacts", required=True, type=Path, help="Directory containing three pinned public files")
    parser.add_argument("--download", action="store_true", help="Fetch only allowlisted public files into a NEW artifacts directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.download:
        download_artifacts(args.artifacts)
    report = collect(args.output, args.artifacts)
    print(json.dumps({"status": report["status"], "prompt_records": report["prompt_records"],
                      "split_role_counts": {k: v["roles"] for k, v in report["splits"].items()},
                      "unmapped_ids_rejected": report["unmapped_output_id_count"],
                      "wall_seconds_after_imports": report["wall_seconds_after_imports"]}, indent=2))


if __name__ == "__main__":
    main()
