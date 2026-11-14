"""Load one pinned local Qwen text checkpoint, without networking or inference.

The caller supplies the already downloaded archive and its pinned manifest.
Weights are assigned into a meta model without a second initialized parameter
copy. Keep the checkpoint files immutable while the returned model is alive:
safetensors may retain memory-mapped storage from these files.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import stat
import struct
import sys


MODEL_ID = "Qwen/Qwen3.5-9B"
REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
UPSTREAM_METADATA_SHA256 = "7a2e1ecd8e3349269e5ee6dd74bdbdc93c8d0b38e7f506c84b62df26b88d39a5"
PINNED_FILES = {
    "config.json": (3126, "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05"),
    "model.safetensors.index.json": (79657, "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5"),
    "model.safetensors-00001-of-00004.safetensors": (5276436216, "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e"),
    "model.safetensors-00002-of-00004.safetensors": (5335161512, "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330"),
    "model.safetensors-00003-of-00004.safetensors": (5368717440, "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec"),
    "model.safetensors-00004-of-00004.safetensors": (3325995712, "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6"),
}
CHUNK_BYTES = 8 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2, "F32": 4}
TORCH_DTYPES = {"BF16": "torch.bfloat16", "F32": "torch.float32"}


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_identity(path):
    info = Path(path).stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("checkpoint artifact is not a regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _local_file(directory, name):
    path = directory / name
    if Path(name).name != name or path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError("checkpoint artifact must be a direct local regular file")
    _file_identity(path)
    return path


def _manifest(manifest_path):
    with Path(manifest_path).open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise ValueError("checkpoint manifest is too large")
    value = _json(raw)
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("model_id") != MODEL_ID or value.get("revision") != REVISION
            or value.get("upstream_metadata_sha256") != UPSTREAM_METADATA_SHA256
            or set(value.get("files", {})) != set(PINNED_FILES)):
        raise ValueError("checkpoint manifest is not the pinned Qwen archive")
    for name, (size, digest) in PINNED_FILES.items():
        row = value["files"][name]
        if type(row.get("bytes")) is not int or (row["bytes"], row.get("sha256")) != (size, digest):
            raise ValueError("checkpoint file pin differs from upstream: " + name)
    return value, hashlib.sha256(raw).hexdigest()


def _small_pinned_json(path, size, expected_hash):
    before = _file_identity(path)
    if before[2] != size:
        raise ValueError("checkpoint metadata size mismatch: " + path.name)
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if (len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_hash
            or _file_identity(path) != before):
        raise ValueError("checkpoint metadata content changed: " + path.name)
    return _json(raw), before


def _scan_safetensors(path, *, expected_size, expected_hash, expected_names, chunk_bytes=CHUNK_BYTES):
    """Verify a whole shard and each tensor in one bounded-memory file pass."""
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    before = _file_identity(path)
    if before[2] != expected_size:
        raise ValueError("checkpoint shard size mismatch: " + path.name)
    whole = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("missing safetensors header length")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size < 2_000_000:
            raise ValueError("unbounded safetensors header")
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError("truncated safetensors header")
        whole.update(prefix)
        whole.update(raw_header)
        header = _json(raw_header)
        header.pop("__metadata__", None)
        if set(header) != set(expected_names):
            raise ValueError("shard tensor names differ from pinned index")
        for name, row in header.items():
            shape, offsets, dtype = row.get("shape"), row.get("data_offsets"), row.get("dtype")
            if (dtype not in DTYPE_BYTES or not isinstance(shape, list)
                    or any(type(n) is not int or n < 1 for n in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(n) is not int or n < 0 for n in offsets)
                    or offsets[1] - offsets[0] != math.prod(shape) * DTYPE_BYTES[dtype]):
                raise ValueError("invalid safetensors shape/dtype/extent: " + name)
        end, inventory = 0, {}
        for name, row in sorted(header.items(), key=lambda pair: pair[1]["data_offsets"]):
            start, stop = row["data_offsets"]
            if start != end:
                raise ValueError("noncontiguous safetensors payload")
            remaining, tensor_digest = stop - start, hashlib.sha256()
            while remaining:
                block = stream.read(min(chunk_bytes, remaining))
                if not block:
                    raise ValueError("truncated safetensors payload")
                whole.update(block)
                tensor_digest.update(block)
                remaining -= len(block)
            inventory[name] = {"shape": row["shape"], "dtype": row["dtype"],
                               "bytes": stop - start, "sha256": tensor_digest.hexdigest(),
                               "data_offsets": [start, stop], "shard": path.name}
            end = stop
        if end + 8 + header_size != expected_size or stream.read(1):
            raise ValueError("safetensors payload extents differ from file size")
    if whole.hexdigest() != expected_hash or _file_identity(path) != before:
        raise ValueError("checkpoint shard hash mismatch or concurrent change: " + path.name)
    return inventory, {"bytes": expected_size, "sha256": whole.hexdigest(),
                       "header_bytes": header_size, "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                       "tensors": len(inventory)}, before


def tensor_sha256(tensor, *, chunk_bytes=CHUNK_BYTES):
    """Hash CPU storage bytes, including BF16, without whole-tensor byte copies."""
    import torch

    if (type(chunk_bytes) is not int or chunk_bytes < 1 or tensor.device.type != "cpu"
            or tensor.layout != torch.strided or not tensor.is_contiguous()):
        raise ValueError("tensor hashing requires contiguous CPU storage and positive chunk size")
    # Reinterpret after flattening, so scalar BF16 tensors also work. NumPy sees
    # uint8, never an unsupported BF16 dtype. Both conversions are views.
    view = memoryview(tensor.detach().reshape(-1).view(torch.uint8).numpy())
    result = hashlib.sha256()
    for offset in range(0, len(view), chunk_bytes):
        result.update(view[offset:offset + chunk_bytes])
    return result.hexdigest()


def _assign_verified_tensors(model, directory, text_inventory):
    """Strict assignment mechanic shared with small serialized-fixture tests."""
    from safetensors import safe_open

    expected = model.state_dict()
    if set(expected) != set(text_inventory) or any(not value.is_meta for value in expected.values()):
        raise ValueError("meta-model state keys do not exactly match checkpoint text tensors")
    for name, tensor in expected.items():
        if list(tensor.shape) != text_inventory[name]["shape"]:
            raise ValueError("meta-model/checkpoint shape mismatch: " + name)
    state = {}
    for shard in sorted({row["shard"] for row in text_inventory.values()}):
        with safe_open(str(_local_file(directory, shard)), framework="pt", device="cpu") as archive:
            for name, row in text_inventory.items():
                if row["shard"] != shard:
                    continue
                tensor = archive.get_tensor(row["checkpoint_name"])
                if list(tensor.shape) != row["shape"] or str(tensor.dtype) != TORCH_DTYPES[row["dtype"]]:
                    raise ValueError("loaded tensor differs from verified checkpoint bytes: " + name)
                state[name] = tensor
    # assign=True swaps in verified storages. A conventional copy_ into an
    # allocated model would create a second full parameter allocation.
    model.load_state_dict(state, strict=True, assign=True)
    for name, tensor in model.state_dict().items():
        if (tensor.is_meta or list(tensor.shape) != text_inventory[name]["shape"]
                or str(tensor.dtype) != TORCH_DTYPES[text_inventory[name]["dtype"]]
                or tensor.data_ptr() != state[name].data_ptr()):
            raise ValueError("strict assignment did not preserve verified tensor storage: " + name)
        if tensor_sha256(tensor) != text_inventory[name]["sha256"]:
            raise ValueError("loaded tensor differs from verified checkpoint bytes: " + name)
    model.requires_grad_(False)
    model.eval()


def _realize_rotary_buffers(model):
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    expected = {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}
    state_names = set(model.state_dict())
    actual = {name for name, _ in model.named_buffers() if name not in state_names}
    if actual != expected:
        raise ValueError("unexpected nonpersistent Qwen buffers; refuse partial realization")
    original = {name: value for name, value in model.named_buffers() if name in expected}
    if any(not value.is_meta for value in original.values()):
        raise ValueError("expected uninitialized meta rotary buffers")
    # Reconstruct only these two config-derived buffers, never model.to_empty()
    # or model.to(), which would allocate/cast the full checkpoint.
    with torch.device("cpu"):
        replacement = Qwen3_5TextRotaryEmbedding(model.config)
    replacement.eval()
    model.model.rotary_emb = replacement
    result = {}
    for name, tensor in model.named_buffers():
        if name in expected:
            if (tensor.device.type != "cpu" or tensor.shape != original[name].shape
                    or tensor.dtype != original[name].dtype or not torch.isfinite(tensor).all()):
                raise ValueError("config-derived rotary buffer realization failed")
            result[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256": tensor_sha256(tensor), "source": "pinned-config-and-runtime-formula",
                            "checkpoint_tensor": False}
    return result


def load_qwen_text_checkpoint(checkpoint_directory, manifest_path, *, attn_implementation="eager"):
    """Return ``(frozen_cpu_eval_model, evidence)`` from the pinned local archive.

    This performs actual pretrained weight loading, but no forward, generation,
    backward, adapter creation, training, network request or checkpoint write.
    """
    if sys.byteorder != "little":
        raise ValueError("this exact raw-byte loader requires a little-endian host")
    if attn_implementation not in {"eager", "sdpa"}:
        raise ValueError("unverified attention implementation")
    directory = Path(checkpoint_directory)
    _, manifest_digest = _manifest(manifest_path)
    import torch
    import transformers
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    if (torch.__version__ != "2.6.0+cpu" or transformers.__version__ != "5.17.0"
            or importlib.metadata.version("safetensors") != "0.8.0"):
        raise ValueError("loader requires recorded Torch2.6.0+cpu/Transformers5.17.0/safetensors0.8.0")
    identities = {}
    metadata, file_evidence = {}, {}
    for name in ("config.json", "model.safetensors.index.json"):
        path = _local_file(directory, name)
        size, expected_hash = PINNED_FILES[name]
        metadata[name], identities[name] = _small_pinned_json(path, size, expected_hash)
        file_evidence[name] = {"bytes": size, "sha256": expected_hash}
    config, index = metadata["config.json"], metadata["model.safetensors.index.json"]
    tensors = {}
    shards = sorted(set(index["weight_map"].values()))
    if set(shards) != set(PINNED_FILES) - set(metadata):
        raise ValueError("pinned index shard set mismatch")
    for shard in shards:
        size, expected_hash = PINNED_FILES[shard]
        inventory, receipt, identity = _scan_safetensors(
            _local_file(directory, shard), expected_size=size, expected_hash=expected_hash,
            expected_names={name for name, file in index["weight_map"].items() if file == shard})
        if tensors.keys() & inventory.keys():
            raise ValueError("duplicate tensor across checkpoint shards")
        tensors.update(inventory)
        file_evidence[shard], identities[shard] = receipt, identity
    if (len(tensors) != 775 or sum(row["bytes"] for row in tensors.values()) != index["metadata"]["total_size"]):
        raise ValueError("checkpoint inventory differs from pinned complete archive")
    text_inventory, excluded = {}, Counter()
    for name, row in tensors.items():
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            runtime_name = name.replace("model.language_model.", "model.", 1)
            text_inventory[runtime_name] = {**row, "checkpoint_name": name}
        elif name.startswith("model.visual."):
            excluded["vision"] += 1
        elif name.startswith("mtp."):
            excluded["mtp"] += 1
        else:
            raise ValueError("unexpected checkpoint component: " + name)
    if len(text_inventory) != 427 or dict(excluded) != {"vision": 333, "mtp": 15}:
        raise ValueError("checkpoint text/vision/MTP coverage differs from pinned archive")
    if config["tie_word_embeddings"] is not False:
        raise ValueError("pinned output head and embedding must remain untied")
    text_config = Qwen3_5TextConfig.from_dict(dict(config["text_config"], tie_word_embeddings=False))
    text_config._attn_implementation = attn_implementation
    text_config.use_cache = False
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(text_config).eval()
    _assign_verified_tensors(model, directory, text_inventory)
    buffers = _realize_rotary_buffers(model)
    if (any(t.device.type != "cpu" or t.is_meta for t in list(model.parameters()) + list(model.buffers()))
            or any(p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules())
            or model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()):
        raise ValueError("loaded model violates CPU/frozen/eval/untied-buffer contract")
    for name, before in identities.items():
        if _file_identity(_local_file(directory, name)) != before:
            raise ValueError("checkpoint artifact changed during verified loading: " + name)
    if _manifest(manifest_path)[1] != manifest_digest:
        raise ValueError("checkpoint manifest changed during loading")
    evidence = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
                "manifest_sha256": manifest_digest,
                "loader_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "candidate_weights_loaded": True, "candidate_forward_performed": False,
                "generated_forecasts": 0, "training_performed": False,
                "scientific_checkpoint_selected": False, "network_requests": 0,
                "checkpoint_files": file_evidence, "text_tensors": text_inventory,
                "text_tensor_manifest_sha256": _digest(text_inventory),
                "text_parameters": sum(math.prod(row["shape"]) for row in text_inventory.values()),
                "stored_text_bytes": sum(row["bytes"] for row in text_inventory.values()),
                "stored_text_tensor_dtypes": dict(Counter(row["dtype"] for row in text_inventory.values())),
                "excluded_checkpoint_tensors": dict(excluded), "nonpersistent_buffers": buffers,
                "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "safetensors": importlib.metadata.version("safetensors"),
                            "attention_implementation": attn_implementation, "use_cache": False,
                            "model_config_sha256": _digest(model.config.to_dict())},
                "scope": "Exact local pretrained text-weight assignment and config-derived buffers; no forward, backward, generation or training."}
    return model, evidence
