# Qwen text LoRA runtime — 13 September 2026

`training/qwen_lora_runtime.py` connects an explicitly supplied
`Qwen3_5ForCausalLM` to the existing episode reward adapter with actual LoRA,
categorical generation, causal loss, backward, and one optimizer update per
complete joint batch. It does not load weights or select a scientific treatment.
The validation here uses a **reduced randomly initialized Qwen architecture** and
explicit synthetic numerical readouts; it establishes no pretrained-model
training result. The [recorded evidence](../data/research/qwen-lora-runtime-20260913.json)
pins the tested source and scope.

The default provisional treatment freezes the complete original base and adds
**rank16, alpha32, dropout0, FP32 LoRA** to all **248 decoder Linear targets**.
The module records each target's original name and shape and requires the caller's
expected target count (248 by default). It excludes embeddings, LM head, norms,
convolutions, vision and MTP. The [candidate structure preflight](qwen-candidate-preflight-20260913.md)
derives **43,278,336 trainable adapter parameters** for the full inspected text
architecture. This arithmetic is separate from the reduced numerical tests.

## Caller interface

The caller supplies a previously verified local text model. The runtime preserves
its existing device and all original BF16/F32 tensor dtypes; it does not cast the
base to FP32, merge adapters, or invoke a model download. It accepts one actual CPU
or CUDA device; the validation reported here exercises CPU only. Set execution
options such as CPU thread count before constructing the runtime.

```python
from training.qwen_lora_runtime import QwenLoraRuntime

# model comes from the separately verified checkpoint loader.
runtime = QwenLoraRuntime(model, lora_seed=20260913)
checkpoint_identity = runtime.checkpoint_identity()
optimizer = torch.optim.Adam(runtime.adapter_parameters.values(), lr=explicit_lr)

# adapter is an existing EpisodeRewardAdapter with the complete matching anchor,
# immutable initial training identity, exact protocol and all split hashes.
plan = adapter.prepare(
    episode_ids, samples_per_episode=explicit_samples, batch_id=fresh_batch_id,
    policy_sha256=runtime.policy_sha256(),
    runtime_training_identity=bound_initial_training_identity,
    runtime_sampling_distribution=runtime.sampling_distribution(),
    encode_messages=tokenizer.encode_messages,
)
records = runtime.generate_plan(
    plan, tokenizer, generator=declared_generator, max_new_tokens=explicit_cap,
    on_record=persist_each_record_immediately,
)
credits = adapter.score(
    plan, [record.trajectory for record in records],
    decode_completion=tokenizer.decode_completion, pad_token_id=tokenizer.pad_token_id,
)
update = runtime.step(credits, records=records, optimizer=optimizer, plan=plan)
```

The existing adapter must score **all complete joint samples before any
backward**. Its full-split anchor requirement remains unchanged: the original
training split needs **2,926 matching own-base proposition readouts**. A tiny
candidate probe cannot replace this anchor. The runtime does not manufacture the
initial binding, outcomes, anchors, probabilities or rewards. It accepts the
existing `TokenCreditBatch` together with the original full `RolloutPlan`,
recomputes the plan digest, and checks every original request's identity, order,
prompt and joint mapping. Jointly removing a role from receipts and credit rows
cannot bypass this check. It also rechecks complete row bindings, target masks,
finite rewards and the exact global joint denominator before backward.

The caller owns durable unique attempt directories, raw receipt/RNG persistence,
checkpoint/optimizer serialization and externally pinned experiment identities.
`checkpoint_identity()` performs a complete base audit and returns base,
adapter, execution and target manifests. `policy_sha256()` combines the last
verified immutable base with current adapter contents and execution identity;
it performs lightweight base guards rather than rereading the full base on
every token. The original checkpoint identity remains fixed while adapter policy
hashes change. `generate_tokens()` is also available for an individually audited
explicit prompt, returning tokens, sampled log probabilities and finish reason.

`generate_tokens(..., on_token=callback)` optionally calls the collector after
each sampled token and log probability have been retained and the supplied RNG
has advanced, before EOS returns or the next forward starts. Each fresh event
dict contains zero-based `token_index`, integer `token_id`, FP32-derived
`sampled_logprob` and boolean `is_eos`. The collector can capture the supplied
generator's state from its closure and synchronously persist the event. The
callback must not change the model or generator. Its return value is ignored;
an exception stops generation without retry and `GenerationFailure.partial`
includes the sampled token whose callback failed, including EOS.

Durability belongs to the collector: it must flush/fsync before returning if it
requires persisted token receipts. A forced termination before that write
finishes can still leave the current token unpersisted. This callback adds no
resume, resampling, token filtering, or sampling-distribution change. Historical
evidence source hashes remain pinned to the code that produced that evidence;
the callback revision has its own current runtime/execution source identity.

## Categorical sampler correction

Both this runtime and the generic Torch loop now use `training/categorical.py`.
A software-only replay of a retained Torch 2.6 CPU/MKL RNG state reproduced an
exponential-noise floor in the former `torch.multinomial` path: synthetic weights
with less than 2e-12 mass at that token selected it over near-unit mass elsewhere.
This is a sampler defect, not evidence that the model assigned that token high
probability. The regression uses the retained seed and full 248,320-entry
vocabulary with explicitly synthetic logits; it loads no pretrained weights.

The replacement computes the full FP32 softmax, accumulates and normalizes its
CDF in FP64, draws one FP64 uniform in [0, 1), and finds the first CDF entry
strictly greater than the draw. Right-sided search skips zero-mass intervals,
including at exactly zero or an internal boundary. No clipping, filtering,
retries or output repair is applied. The selected log probability remains the
unmodified FP32 log-softmax value used by causal loss. Finite FP64 CDF and uniform
resolution still limit representable probabilities; this is not exact
real-number distribution sampling.

The generic Torch loop promotes BF16/FP16 logits to FP32 in both generation and
loss; existing FP64 logits retain FP64 normalization and selected log probabilities.
Qwen continues to supply FP32 logits. Both paths accumulate the CDF in FP64.

Runtime execution and generic policy identities bind the helper's algorithm and
source SHA-256. Collector and synthetic-runner source maps include the helper.
The change advances RNG differently and requires a new execution/policy identity;
old evidence and campaign source pins are preserved. Reduced random-Qwen and
synthetic tests establish implementation mechanics, not corrected full-candidate
results or authorization to restart collection.

## Exact update and memory behavior

LoRA A is initialized from a caller-seeded local CPU generator and B is zero.
The extra branch evaluates in FP32, applies alpha/rank, then casts back to the
original Linear output dtype. Zero initialization preserves initial logits
exactly in the reduced FP32 and mixed BF16/F32 tests. Actual selected checkpoint
verification remains a separate caller responsibility.

Every module stays in eval mode and every forward explicitly passes
`use_cache=False` and `past_key_values=None`. Decoder blocks use
`torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` whenever gradients
are enabled, independently of Transformers' native training-mode checkpoint flag.
This works even though frozen embeddings do not require gradients. The test
observed 40 explicit checkpoint calls for four reduced blocks × ten role forwards
and matched the gradients against an uncheckpointed whole-joint loss.

Generation computes only the last position's vocabulary logits. Each training
role uses the unpadded causal prefix and asks Qwen for only the generated-target
positions' logits. FP32 softmax/log-softmax spans the **entire model vocabulary**;
there is no temperature scaling, mask, forced token, top-k/top-p filtering or
answer extraction. Every role is generated from a fresh complete prefix without
reused conversation or model cache state.

Each role's loss retains the original coefficient `-R_joint/N_joint_samples`.
Role backward calls accumulate gradients and one optimizer step follows:

```text
L = -(1/N_joint_samples) * sum_joint R_joint
                  * sum_roles sum_generated_tokens log pi(token | prefix)
```

There is no extra average over roles, tokens or microbatches; no reward or
advantage normalization, clipping, baseline or token-policy KL. EOS remains a
generated target even when rendering omits it. Prompt/padding targets receive no
credit. The runtime's selected-position loss is checked against each recorded
behavior log probability at the existing **atol=2e-6, rtol=2e-6**. The reduced BF16
diagnostic matched exactly; this does not establish tolerance sufficiency for
the full pretrained candidate. A mismatch fails the step and reports its maximum
absolute error instead of loosening the comparison.

Only adapter values and optimizer state are snapshotted for rollback. Returned
gradients, deltas and updated state also contain adapters only. The frozen base
is neither cloned for rollback nor included in zero-valued delta tensors.
Optimizer snapshots, adapter/gradient/result copies and activations still consume
memory; this implementation makes no unmeasured full-candidate fit or throughput
claim. Results include policy/base hashes, aggregate loss, log-probability error,
joint/trajectory counts, adapter state, gradients and parameter deltas.

## Base integrity and failure handling

Raw tensor hashing supports BF16 by viewing contiguous values as bytes. Hashes
stream through bounded 8MiB CPU memoryviews; non-CPU tensors copy only one bounded
chunk at a time. Dtype and shape are bound separately. Initialization checks that
base tensors are finite. The runtime records original base names/references
before wrapping modules and verifies:

1. Tensor object identity, pointer, shape, stride, dtype, device and trainability
   against the active module registry; every frozen tensor's version counter
   before/after each forward and role backward.
2. Complete frozen-base content at initialization, after a full generation
   collection (including failed collections), at update entry, after backward
   immediately before the optimizer, and after the optimizer.
3. Complete adapter content against the behavior policy before update, plus
   adapter version guards during backward; finite adapter values, gradients and
   recursively checked optimizer state.

Full streaming checks detect `.data` writes that bypass version counters. The
tests inject that corruption during generation and backward: all attempted
tokens remain available, the optimizer cannot accept the corrupted batch, and
the runtime becomes **unusable until a verified base is reloaded**. Adapter-only
rollback cannot reconstruct corrupted base weights and never pretends to do so.
Whole-base reads still have real bandwidth/time costs despite avoiding copies.

The execution identity records actual model configuration, attention backend,
dispatcher-selected recurrent/convolution implementation, device, dtypes,
checkpointing, package/source hashes, threads, matmul precision, TF32 and
determinism settings. Mode, autocast, configuration, backend and treatment drift
fail. These checks are not attestation of arbitrary Python monkeypatches, native
kernel correctness, external concurrent writers or mutation that is maliciously
made and restored between checks. This is a single-owner runtime; arbitrary
concurrent use, distributed execution and external mutation are unsupported.

Invalid, truncated or failed readouts must be rejected by the existing adapter
before `step`. The existing Qwen tokenizer retains reasoning/control text and
rejects all 243 unmapped model-output IDs with their original tokens. The runtime
streams each rollout record through `on_record`, and `RolloutBatchFailure.records`
retains prior and partial role attempts even when generation, decoding or base
verification fails. No failed answer is repaired or regenerated. A failed
`generate_plan` cannot be attempted again in that runtime.

An attempted update is consumed even if a later ordinary Python exception causes
rollback; zero-learning-rate updates also cannot replay their credit. A normal
optimizer failure restores adapter values and existing optimizer moments, clears
gradients and leaves base checks active. Process crashes, forced termination,
cross-process replay and durable resume are outside this in-process mechanism.
The caller must preserve failed attempts and enforce its durable run ownership.

## Verification scope

The sampler correction passes all 56 focused tests in 27.805 seconds:
`python -m unittest test_categorical test_qwen_lora_runtime test_training_torch test_qwen_base_collection -v`.
These include exact CDF boundaries/zero-mass gaps, full-vocabulary point masses,
retained-seed exponential-floor reproduction, independent Python CDF oracles for
both consumers, exact behavior log probabilities and RNG advancement, and rejection
of old collector source/execution pins. The existing FP64 analytic gradient check
retains its 1e-14 tolerance. An initial unconditional FP32 loss conversion failed
that check; preserving generic FP64 precision corrected the implementation without
changing its assertion. Without optional numerical dependencies, the same command
passes 14 collector tests and skips 42 numerical tests (16.465 seconds), including
a subprocess assertion that collector planning never imports Torch.

The original 14 focused tests passed, including an independent review run. With
the per-token callback, all 16 focused tests pass (6.111 seconds); the added tests
compare exact tokens, log probabilities and RNG states with/without the callback,
including EOS and callback failure after a sampled token. They use the actual
Transformers 5.17.0 Qwen implementation with
four reduced decoder blocks, 31 Linear targets and 34,240 adapter parameters.
They verify BF16 hashing, exact zero-update logits, final/generated-position
projection, checkpointed microbatch gradients, complete joint normalization,
rejection of coherently removed roles against the original plan,
policy mismatch/replay, parameter and backend drift, rollback of warmed momentum
and nonfinite Adam state, and partial generation/decode/base-corruption retention.
The numerical training readouts are explicitly hand-built coin/die/geometry
fixtures from the existing adapter tests, not answers from the reduced model.
Separate generation tests use random reduced weights and make no forecast claim.

Run `python -m unittest test_qwen_lora_runtime -v` in an environment with the
already pinned CPU torch and Qwen tokenizer dependencies. On General, the tests
used the existing CPU venv with the tokenizer venv's site-packages on `PYTHONPATH`;
both environments were reused read-only. The optional module skips cleanly when
those dependencies are absent. No pretrained weights, credentials, provider
requests, global installs or paid resources were used by this implementation
task. Full candidate loading, generated output validity, complete anchors,
memory/throughput and actual training remain separate measurements.
