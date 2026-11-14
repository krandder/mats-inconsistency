# Pinned local Qwen text checkpoint loader

`training/qwen_checkpoint.py` implements actual local weight loading for
`Qwen/Qwen3.5-9B` revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
It performs no download, network request, forward pass, generation, backward,
adapter creation, or training. Implementing this loader does not establish that
the candidate has been loaded or that its runtime works; a separate run receipt
must record any actual candidate execution.

```python
from training.qwen_checkpoint import load_qwen_text_checkpoint

model, evidence = load_qwen_text_checkpoint(
    checkpoint_directory,
    pinned_manifest_path,
    attn_implementation="eager",
)
```

The explicit local directory must contain the pinned `config.json`, shard index,
and all four complete safetensors files. The manifest uses schema version 1,
`model_id`, `revision`, `upstream_metadata_sha256`, and a `files` mapping from
these six filenames to `bytes` and `sha256`. The loader hashes the exact manifest
bytes and checks its file pins against the captured upstream revision metadata.
Other provenance fields may be included. The caller owns downloading and keeping
the files immutable; the loader does not create or alter them.

The loader verifies each full shard against its upstream SHA-256 while computing
per-tensor payload hashes in the same bounded-memory pass. It checks the actual
header names against the pinned index, BF16/F32 dtype sizes, shapes, contiguous
payload extents, complete shard size, and complete archive inventory. It does
not treat a self-reported tensor digest as evidence of checkpoint content.

The text-only `Qwen3_5ForCausalLM` is constructed on the meta device. The loader
requires exactly 427 text state keys and the exact checkpoint shapes, using the
documented `model.language_model.` → `model.` mapping. It deliberately excludes
333 vision tensors and 15 multi-token prediction tensors. `safe_open/get_tensor`
and `load_state_dict(strict=True, assign=True)` install verified storages while
preserving their actual BF16/F32 dtypes; no full initialized model or rollback
copy is allocated. Loaded tensor bytes are hashed through chunked uint8 views,
including BF16 and scalar tensors, and checked against the verified file payloads.
The embedding and output head remain separate, untied parameters.

Two nonpersistent rotary buffers do not appear in `state_dict`. The loader
reconstructs only these small buffers on CPU from the pinned configuration and
recorded Transformers formula. Unknown nonpersistent buffers, remaining meta
tensors, missing or extra text weights, dtype changes, and changed checkpoint
files are rejected. It never uses a whole-model `to_empty()` or dtype conversion.

The returned model is CPU-only, frozen, in eval mode, and has `use_cache=False`.
The recorded loader environment is PyTorch `2.6.0+cpu`, Transformers `5.17.0`, and
safetensors `0.8.0`; `eager` or `sdpa` is an explicit backend argument, not evidence
that a forward/backward path has passed. Returned evidence includes full-file,
header, loaded-tensor and realized-buffer manifests, loader source identity, and
an explicit no-forward/no-training scope. The files may back memory-mapped tensor
storage and must remain immutable for the model's lifetime.

`test_qwen_checkpoint.py` tests authoritative manifest rejection, bounded header
validation, BF16/scalar hashing, storage-preserving strict assignment, payload
tampering, exact key/shape/dtype coverage, and rotary buffer realization. Its only
serialized weights are a tiny, explicitly non-scientific fixture. It neither
reads the candidate checkpoint nor generates candidate answers. Tensor tests
skip when the optional CPU/tokenizer libraries are absent.
