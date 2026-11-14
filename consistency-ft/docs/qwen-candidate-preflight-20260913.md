# Qwen3.5-9B candidate preflight — 13 September 2026

**The actual tokenizer and full-size model structure are now checked; loaded
weights, complete own-base anchors and candidate training remain unfinished.**
This narrows the next implementation to a text-only Qwen LoRA runtime. It does
not choose the scientific base, an adapter treatment, or a paid platform.

The inspected candidate is `Qwen/Qwen3.5-9B` at immutable revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The
[architecture report](../data/research/qwen-candidate-architecture-20260913.json)
records checkpoint-header evidence, actual meta-device structure inspection,
source hashes and memory arithmetic. The separate
[tokenizer report](qwen-tokenizer-preflight-20260913.md) covers real tokenization
and the complete original episode corpus; none of these steps generates answers.

## What is verified

Only the safetensors headers were fetched, using exact HTTP byte ranges:
**94,432 bytes of header JSON plus four eight-byte length prefixes**. The collector
rejects a server that ignores the requested range before reading its body. Every
tensor name matches its pinned shard index; dtype, shape, contiguous payload
extents, shard lengths and the index's total byte count agree. Tensor payloads
were not downloaded or loaded.

Transformers **5.17.0** and PyTorch **2.6.0+cpu** successfully instantiated the
actual full-size `Qwen3_5ForCausalLM` architecture on the **meta device**. All
**427 text tensor names and shapes** match the checkpoint after its documented
`model.language_model` → `model` prefix conversion. Its **248 decoder Linear
modules** also match the proposed adapter target inventory. Meta tensors have no
weight data: this verifies structure, not pretrained loading, inference, gradients,
or training performance. The parent configuration's `tie_word_embeddings=false`
is preserved; both large embedding and output-head matrices are required.

| Checkpoint component | Parameters | Stored tensor bytes | GiB |
| --- | ---: | ---: | ---: |
| Text model and output head | 8,953,803,264 | 17,907,614,208 | 16.678 |
| Vision | 456,010,480 | 912,020,960 | 0.849 |
| Multi-token prediction | 243,290,624 | 486,581,248 | 0.453 |
| Complete archive | 9,653,104,368 | 19,306,216,416 | 17.980 |

These are mixed BF16/F32 stored tensor sizes. The text-only class provides an
explicit path to omit vision and multi-token prediction. Actual loading must
still establish complete expected text-key coverage and reject unexpected
missing, reshaped or silently initialized text weights.

The tokenizer inspection measures both the proposed native `numeric-v1`
system/user framing and the original API's user-only prompt. They differ by
**46 tokens per prompt** and remain separate treatments. Under `numeric-v1`,
training prompts (4,855 role occurrences) have median **211**, 99th percentile **327**, and maximum
**406** tokens; the maximum across the three splits is **494**. These counts
include the full real proposition text and explicit empty-thinking prompt suffix,
with no truncation or filtering. The training anchor still requires actual
readouts for **all 2,926 distinct training propositions**.

## Concrete output-handling problems

The pinned tokenizer defines `<|im_end|>` (`248046`) as EOS and
`<|endoftext|>` (`248044`) as padding, whereas the model's text config sets EOS
to `248044`. There is no pinned `generation_config.json`. A candidate run must
declare its termination treatment explicitly and verify it with real outputs;
inheriting these inconsistent defaults is insufficient.

The model produces **248,320 logits**, but the tokenizer defines **248,077 IDs**.
The stock decoder silently drops the remaining **243 IDs**, even with
`skip_special_tokens=false`: a numeric fixture followed by an unmapped token can
decode as a valid number. The new tokenizer adapter rejects unmapped IDs before
decoding. It retains control and reasoning content, removes only its explicitly
declared terminal token, and leaves the full sampled IDs available for failure
receipts and token credit. This changes validation, not the sampling distribution.
The proposed termination contract is not evidence of observed model behavior.

## Memory and runtime implications

An illustrative rank-16 adapter over all 248 decoder Linear modules has
**43,278,336 trainable parameters**. This includes the MLP, full-attention
projections and hybrid linear-attention projections; merely targeting `q_proj`
and `v_proj` misses the latter. Embeddings, output head, norms, convolution,
vision and multi-token prediction are excluded from this adapter proposal.

The frozen text payload plus FP32 adapter weights, gradients and two Adam moments
requires at least **17.323 GiB**. This excludes activations, logits, recurrent/KV
state, loading overhead, temporary storage and recovery. It establishes neither
a fit on any GPU nor measured CPU feasibility. No resource has been reserved.

The existing [synthetic CPU helper](synthetic-cpu-training.md) needs these changes
before it can serve this candidate:

| Observed issue | Required candidate integration |
| --- | --- |
| `tensor_sha256` calls NumPy conversion, which raises `TypeError: Got unsupported ScalarType BFloat16` for BF16. | Hash raw tensor bytes with dtype/shape identity, supporting BF16 and retaining immutable-base checks. |
| Rollback and parameter deltas clone the entire state, including frozen weights. Base + full snapshot + full deltas alone total **50.033 GiB** for Qwen's text weights. | Preserve a verified frozen base; snapshot, roll back and save only trainable adapter state and explicitly mutable buffers. Maintain optimizer rollback and attempt retention. |
| The helper hard-codes CPU devices and materializes logits for all prompt positions. | Make device/backend/precision identity explicit; use last-position logits for rollout and only generated causal positions for loss, keeping the full vocabulary at every such position. |
| Transformers' native gradient-checkpointing branch requires `self.training`; the current sampler/loss contract requires eval mode. | Implement and verify explicit non-reentrant block checkpointing under eval, or separately prove and version a changed mode contract. Enabling the native flag alone is insufficient. |
| Qwen defaults to `use_cache=true`; the helper does not override it. | Disable cache for the differentiable forward and bind the actual attention/DeltaNet implementation. The reference DeltaNet path uses FP32 intermediates. |

For scale, a single `[1,512,248320]` BF16 logits tensor occupies **242.5 MiB**;
ten role trajectories together occupy **2.368 GiB per tensor**, before log-softmax
and backward buffers. FP32 doubles those figures. A completion-length limit is
still a separate treatment choice; the observed prompt lengths do not choose it.

Accumulate role microbatches only after all related rollouts have passed joint
scoring, retain the original global joint-sample denominator, and perform one
optimizer step. Keep EOS credit, complete attempt receipts, exact A2 rewards and
explicit endpoint handling. Validate BF16 sampled/evaluated log probabilities on
the actual runtime rather than quietly loosening the existing numerical check.

## Reproduce the structural inspection

The standard-library command fetches only pinned config/index data and exact
checkpoint-header ranges into a **new** directory, then prints the inventory:

```sh
python3 scripts/inspect_qwen_candidate.py \
  --fetch-pinned-metadata runs/qwen-checkpoint-metadata
```

After that capture, it can be rechecked entirely offline:

```sh
python3 scripts/inspect_qwen_candidate.py \
  --config runs/qwen-checkpoint-metadata/config.json \
  --index runs/qwen-checkpoint-metadata/model.safetensors.index.json \
  --headers-dir runs/qwen-checkpoint-metadata
```

Add `--inspect-runtime` in an isolated environment exposing the recorded
Transformers 5.17.0 and PyTorch 2.6.0+cpu versions for the actual meta-model check.
The retained run used the existing isolated CPU interpreter with the separate
tokenizer environment's site-packages on `PYTHONPATH`, offline; neither environment
was modified. Resolved dependency versions and exact library-source hashes are in
the architecture report. A fresh independent metadata fetch reproduced all
header-derived counts and memory arithmetic, and independent review checked the
inventory, source findings and collector.

The combined repository suite passed **227 tests with no skips**, including all
16 CPU tensor-runtime tests and six actual-tokenizer tests. The architecture
report retains the complete suite log hash; this validation does not include
loaded candidate weights or a candidate forward/backward pass.

This work makes no benchmark changes and supplies no new anchor probabilities.
Candidate-specific runtime implementation, actual checkpoint loading, complete
own-base elicitation, measured memory/throughput, matched candidate baselines and
the platform decision remain open. Fireworks account access remains a separate
unchanged blocker; this preflight makes no provider-inference or billing requests.
