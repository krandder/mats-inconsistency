#!/usr/bin/env python3
"""Inspect pinned Qwen checkpoint metadata and an optional full-size meta model.

Reads config/index/safetensors JSON headers only. It never loads tensor payloads,
generates answers, or chooses scientific hyperparameters. Printed memory figures
are arithmetic lower bounds, not measured training-memory or throughput results.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request


MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
CONFIG_SHA256 = "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"
INDEX_SHA256 = "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique)


def fetch_metadata(directory):
    """Fetch pinned config/index and exact header ranges, never tensor payloads."""
    directory.mkdir(parents=True, exist_ok=False)
    root_url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/"
    for name, expected in (("config.json", CONFIG_SHA256), ("model.safetensors.index.json", INDEX_SHA256)):
        with urllib.request.urlopen(root_url + name, timeout=60) as response:
            content = response.read(1_000_001)
        if len(content) > 1_000_000 or hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("downloaded config/index differs from pinned bytes")
        (directory / name).write_bytes(content)
    index = read_json(directory / "model.safetensors.index.json")
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("invalid pinned shard path")
        url = root_url + shard

        def read_range(start, end, suffix):
            request = urllib.request.Request(url + suffix, headers={"Range": f"bytes={start}-{end}",
                                             "User-Agent": "consistency-ft-checkpoint-header-inspection"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ValueError("server did not honor exact metadata-only range; refusing body")
                content = response.read(end - start + 2)
                if len(content) != end - start + 1:
                    raise ValueError("unexpected range body length")
                return content, int(content_range.split("/")[1])

        prefix, total_bytes = read_range(0, 7, "")
        size = struct.unpack("<Q", prefix)[0]
        if not 0 < size < 2_000_000:
            raise ValueError("invalid or unbounded safetensors header length")
        # Distinct query avoids a range-insensitive intermediary replaying the prefix.
        header, header_total_bytes = read_range(8, 7 + size, "?header=1")
        if total_bytes != header_total_bytes:
            raise ValueError("shard size changed between header ranges")
        path = directory / (shard + ".header.json")
        path.write_bytes(header)
        content = read_json(path)
        receipt = {"source_url": url, "ranges": [[0, 7], [8, 7 + size]],
                   "header_bytes": size, "sha256": file_hash(path), "tensor_data_bytes_downloaded": 0,
                   "tensor_count": len(content) - int("__metadata__" in content),
                   "source_total_file_bytes": total_bytes}
        (directory / (shard + ".receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    return directory / "config.json", directory / "model.safetensors.index.json", directory


def inspect(config_path, index_path, headers_dir, *, inspect_runtime=False):
    if file_hash(config_path) != CONFIG_SHA256 or file_hash(index_path) != INDEX_SHA256:
        raise ValueError("config/index bytes differ from the inspected immutable revision")
    config, index = read_json(config_path), read_json(index_path)
    tensors, sources = {}, []
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard:
            raise ValueError("invalid shard path")
        path = Path(headers_dir) / (shard + ".header.json")
        receipt = read_json(Path(headers_dir) / (shard + ".receipt.json"))
        expected_url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{shard}"
        size = path.stat().st_size
        if (receipt["source_url"] != expected_url or receipt["sha256"] != file_hash(path)
                or receipt["header_bytes"] != size
                or receipt["ranges"] != [[0, 7], [8, 7 + size]]
                or receipt["tensor_data_bytes_downloaded"] != 0):
            raise ValueError("header/receipt identity mismatch")
        header = read_json(path)
        header.pop("__metadata__", None)
        if len(header) != receipt["tensor_count"]:
            raise ValueError("header tensor count differs from receipt")
        expected_names = {name for name, file in index["weight_map"].items() if file == shard}
        if set(header) != expected_names or tensors.keys() & header.keys():
            raise ValueError("header names disagree with pinned shard index")
        previous_end = 0
        for name, item in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"]):
            shape, offsets, dtype = item["shape"], item["data_offsets"], item["dtype"]
            if (dtype not in {"BF16", "F32"} or not isinstance(shape, list)
                    or any(type(n) is not int or n < 1 for n in shape)
                    or len(offsets) != 2 or any(type(n) is not int for n in offsets)
                    or offsets[0] != previous_end
                    or offsets[1] - offsets[0] != math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]):
                raise ValueError("invalid tensor shape/dtype/contiguous payload extent: " + name)
            previous_end = offsets[1]
        if previous_end + 8 + size != receipt["source_total_file_bytes"]:
            raise ValueError("header extents do not cover the recorded complete shard payload")
        tensors.update(header)
        sources.append(receipt)
    groups = defaultdict(lambda: {"parameters": 0, "payload_bytes": 0, "tensors": 0})
    families = defaultdict(lambda: {"module_count": 0, "shapes": set(), "rank16_parameters": 0})
    text_tensors = {}
    for name, item in tensors.items():
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            group = "text"
            runtime_name = name.replace("model.language_model.", "model.", 1)
            text_tensors[runtime_name] = item
        elif name.startswith("model.visual."):
            group = "vision"
        elif name.startswith("mtp."):
            group = "mtp"
        else:
            raise ValueError("unclassified checkpoint tensor: " + name)
        groups[group]["parameters"] += math.prod(item["shape"])
        groups[group]["payload_bytes"] += item["data_offsets"][1] - item["data_offsets"][0]
        groups[group]["tensors"] += 1
        if name.startswith("model.language_model.layers.") and len(item["shape"]) == 2:
            family = name.split(".", 4)[-1]
            families[family]["module_count"] += 1
            families[family]["shapes"].add(tuple(item["shape"]))
            families[family]["rank16_parameters"] += 16 * sum(item["shape"])
    if sum(g["payload_bytes"] for g in groups.values()) != index["metadata"]["total_size"]:
        raise ValueError("header bytes disagree with pinned index total_size")
    for row in families.values():
        row["shapes"] = sorted(row["shapes"])
    adapter_parameters = sum(v["rank16_parameters"] for v in families.values())
    text_bytes = groups["text"]["payload_bytes"]
    report = {
        "model_id": MODEL_ID, "inspected_revision": REVISION,
        "inspection_source_sha256": file_hash(__file__),
        "checkpoint_config_sha256": CONFIG_SHA256, "checkpoint_index_sha256": INDEX_SHA256,
        "candidate_weights_loaded": False, "generated_forecasts": 0,
        "scientific_base_or_lora_treatment_selected": False,
        "header_sources": sources, "checkpoint_groups": dict(groups),
        "checkpoint_tensor_dtypes": dict(Counter(t["dtype"] for t in tensors.values())),
        "proposed_rank16_decoder_linear_families": dict(sorted(families.items())),
        "arithmetic_memory_bounds": {
            "scope": "Header-derived arithmetic; not measured memory, hardware fit, or throughput.",
            "proposed_lora_rank": 16, "trainable_adapter_parameters": adapter_parameters,
            "frozen_text_base_payload_bytes": text_bytes,
            "fp32_adapter_parameters_gradients_two_adam_moments_bytes": adapter_parameters * 16,
            "text_base_plus_fp32_adapter_grad_adam_bytes": text_bytes + adapter_parameters * 16,
            "existing_helper_base_plus_full_snapshot_plus_full_deltas_bytes": text_bytes * 3,
            "excludes": ["activations", "logits", "attention/recurrent/KV state", "allocator/runtime",
                         "temporary clones", "LoRA initialization/loading overhead", "rollback/recovery overhead"],
            "assumptions": ["Preserve stored BF16/F32 base dtypes; do not cast the whole base to FP32.",
                            "Freeze all text base weights; omit vision and MTP from the text-only class.",
                            "Rank16 adapters on every decoder Linear; no embedding/lm_head/norm/conv adapter.",
                            "FP32 adapters, gradients and two FP32 Adam moments; no separate master copy."],
        },
    }
    if inspect_runtime:
        import importlib.metadata
        import torch
        import transformers
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        if transformers.__version__ != "5.17.0" or torch.__version__ != "2.6.0+cpu":
            raise ValueError("runtime inspection requires the recorded Transformers5.17.0 / Torch2.6.0+cpu")
        text_config = Qwen3_5TextConfig.from_dict(dict(config["text_config"],
                                                    tie_word_embeddings=config["tie_word_embeddings"]))
        text_config._attn_implementation = "eager"
        with torch.device("meta"):
            model = Qwen3_5ForCausalLM(text_config).eval()
        state = model.state_dict()
        runtime_tensors = list(model.parameters()) + list(model.buffers())
        if set(state) != set(text_tensors) or any(not t.is_meta for t in runtime_tensors):
            raise ValueError("runtime meta state and checkpoint text keys do not match")
        for name, tensor in state.items():
            if list(tensor.shape) != text_tensors[name]["shape"]:
                raise ValueError("runtime/checkpoint shape mismatch: " + name)
        linear = {name: [module.out_features, module.in_features]
                  for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Linear) and name.startswith("model.layers.")}
        checkpoint_linear = {name.removesuffix(".weight"): item["shape"]
                             for name, item in text_tensors.items()
                             if name.startswith("model.layers.") and len(item["shape"]) == 2}
        if linear != checkpoint_linear:
            raise ValueError("proposed decoder adapter targets differ from actual Linear modules")
        report["runtime_meta_inspection"] = {
            "torch": torch.__version__, "transformers": transformers.__version__,
            "meta_only": True, "all_text_keys_and_shapes_match": True,
            "text_tensors": len(state), "parameters": sum(t.numel() for t in model.parameters()),
            "decoder_linear_modules": len(linear), "weight_data_allocated": False,
            "resolved_dependency_versions": {name: importlib.metadata.version(name) for name in
                                             ("numpy", "tokenizers", "huggingface-hub", "safetensors")},
            "meta_parameter_dtype": str(next(model.parameters()).dtype),
            "scope": "Actual full-size architecture, meta device only; no forward, backward or loaded weights.",
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--headers-dir", type=Path)
    parser.add_argument("--fetch-pinned-metadata", type=Path,
                        help="New directory for pinned config/index and exact safetensors header ranges")
    parser.add_argument("--inspect-runtime", action="store_true")
    args = parser.parse_args()
    local = (args.config, args.index, args.headers_dir)
    if args.fetch_pinned_metadata:
        if any(local):
            parser.error("choose either fetching pinned metadata or the three local input paths")
        local = fetch_metadata(args.fetch_pinned_metadata)
    elif not all(local):
        parser.error("provide all three local input paths or --fetch-pinned-metadata")
    print(json.dumps(inspect(*local,
                             inspect_runtime=args.inspect_runtime), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
