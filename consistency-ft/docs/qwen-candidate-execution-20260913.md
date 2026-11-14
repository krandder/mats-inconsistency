# Local Qwen candidate execution — 13 September 2026

The actual pretrained Qwen text model loaded successfully and generated all
three predeclared numerical readouts, stopping at the declared EOS each time.
Every loaded text weight remained byte-identical afterward. The bounded CPU
probe took **400.591 seconds**, with a **17.470 GiB peak process RSS**.
A separate retained-answer probe replayed four token log probabilities exactly
and completed backward with finite gradients on all 248 LoRA targets. This
establishes local candidate execution; it does not select the scientific base
or constitute the full A2 experiment.
The [evidence manifest](../data/research/qwen-candidate-execution-20260913.json)
binds the retained artifacts, implementation sources and validation log.

## Checkpoint acquisition and exact loading

The checkpoint revision is
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The four safetensors shards,
configuration and index total **19,306,393,663 bytes**. The public download took
**301.944 seconds** using two bounded download workers. Every file's byte count
and SHA-256 matched the retained upstream revision metadata. Final files are
read-only; the model archive remains local and is not added to Git.

The [artifact manifest](../data/research/qwen-checkpoint-artifacts-20260913.json)
and [download receipt](../data/research/qwen-candidate-execution-20260913/download-receipt.json)
identify the archive. The [downloader](../scripts/download_qwen_checkpoint.py)
streams files, refuses existing destinations, retains partial failures and
checks for archive space plus a 10 GiB reserve at startup. That reserve is not
continuously enforced; the HTTP socket timeout is not a total wall-time limit.
This downloader has no crash resume.

The independently reviewed [local loader](qwen-checkpoint-loader.md) verifies
the complete archive and per-tensor payloads, constructs a text-only model on the
meta device, and assigns verified storage without an initialized duplicate base.
It preserves stored BF16/F32 values, all **427 text tensors / 8,953,803,264
parameters**, and untied embedding/output weights. Only two nonpersistent rotary
buffers are reconstructed from configuration. Vision and MTP tensors remain
excluded from the text runtime.

## Predeclared generation procedure

The [probe](../scripts/probe_qwen_candidate.py) uses original training episode
`2028/AndChecker:11` without editing its three complete proposition texts:

1. An Italian film wins the International Feature Oscar by 1 January 2028.
2. A film focused on quantum computing wins an Oscar before 1 January 2028.
3. Both events occur.

These descriptions are summaries; the retained messages and source hashes bind
the exact original text. The three role seeds were fixed before generation:
`2026091301`, `2026091302`, `2026091303`. This is one engineering episode,
not a representative benchmark sample.

The treatment is `numeric-v1` under the pinned Qwen nonthinking chat template:
temperature 1, full-vocabulary FP32 categorical normalization, no top-k/top-p
filtering or logits processors, eval mode and cache disabled. Each role starts
from its complete independent prompt. There is a maximum of 32 generated tokens
per role and explicit terminal token **248046**. This is the tokenizer's EOS;
the model configuration uses 248044. The original API framing and temperature
are different treatments and their scores cannot be silently pooled.

The process uses the existing CPU PyTorch 2.6.0 environment and pinned
Transformers 5.17.0/tokenizer environment read-only, with two Torch threads,
one interop thread and deterministic algorithms. Its address-space limit is
36 GiB, CPU soft/hard limits are 900/910 seconds, and an external
`timeout --kill-after=10s 900s` enforces the wall bound. Python signal handling
alone may be delayed inside native kernels. These are process limits, not
reserved fleet capacity or proof of training fit.

Every sampled token, log probability and RNG-state hash is written immediately.
Completed readouts retain exact prompt/completion IDs and messages. Truncation
does not count as a valid probability even if its visible prefix parses as a
number. Invalid/unmapped decoding fails and preserves raw IDs. Completed
failures are not resampled. Ordinary exceptions retain failure/readout evidence;
a hard kill can leave only partial token events and the supervisor exit status.
After successful generation, the probe rehashes every loaded text weight.

## Measured result

The [complete report](../data/research/qwen-candidate-execution-20260913/report.json)
records three valid numerical readouts, 13 generated tokens, zero retries,
491.947 CPU seconds and 18,758,676,480-byte peak RSS. The external supervisor
returned zero after 402.010 wall seconds, including interpreter shutdown.
Verified loading took 155.839 seconds within the probe's total time.

| Original episode role | Prompt tokens | Completion tokens, including EOS | Raw answer | Generation seconds |
| --- | ---: | ---: | ---: | ---: |
| Italian film (`P`) | 191 | 4 | 0.0 | 59.398 |
| Quantum-computing film (`Q`) | 190 | 4 | 0.4 | 57.823 |
| Both (`P_and_Q`) | 269 | 5 | 0.15 | 96.944 |

The [raw readouts](../data/research/qwen-candidate-execution-20260913/readouts.jsonl)
and [token events](../data/research/qwen-candidate-execution-20260913/token-events.jsonl)
retain exact text, tokens, log probabilities and RNG hashes. Numerical validity
is not accuracy or logical consistency: this particular conjunction answer
exceeds its `P` answer. The exact-zero answer also makes the choice of KL
endpoint handling practically relevant; it has not been clipped or smoothed.
These three selected engineering questions cannot estimate model performance
or the time and failure rate for a complete anchor collection.

The actual runtime used the reference PyTorch recurrent/convolution kernels;
their optimized packages are absent. Per-token generation took roughly
14–20 seconds for these prompt lengths. Full-weight integrity reads are included
in total time. The RSS measurement covers this frozen generation process,
not adapter backward, optimizer state, a GPU runtime or reserved fleet memory.

## Actual pretrained numerical backward

The [separate diagnostic](qwen-backward-probe.md) uses the first `P` readout in
the original predeclared role order, with the complete three-role generation
evidence pinned before the derivative launch. It installs all 248 rank-16/alpha-32
targets: **43,278,336 FP32 adapter parameters**, seeded A and zero B. It computes
one derivative of the sum of the four retained generated-token log probabilities,
including EOS, with coefficient 1. No anchor, episode credit or optimizer is
constructed. This is a numerical runtime check, not an A2 update.

The [retained evidence](../data/research/qwen-numerical-backward-20260913.json)
records one successful supervised process, **236.994 seconds**, **211.547 CPU
seconds**, and **19,121,463,296 bytes / 17.808 GiB peak RSS**. The one selected-position
forward reproduced all four autoregressively sampled log probabilities exactly:
maximum absolute error **0.0**, under the unchanged `atol=rtol=2e-6`. The
sum-log-probability scalar was `-3.232764959335327`.

All **496 adapter gradient tensors** were present and finite. All 248 B tensors
had nonzero gradients; all A gradients were zero, as expected at zero B.
Full content hashes confirmed unchanged base and adapter weights afterward;
input, source and tokenizer checks also passed. The external supervisor returned
zero after 238.376 seconds. Ordinary failure and hard-kill evidence paths remain
part of the script, although this run completed successfully.

This establishes one pretrained numerical backward calculation with real
checkpoint values and an actual sampled response. It does not exercise a full
candidate joint reward batch, optimizer update, complete anchor collection or
training loop. Its memory result cannot establish longer-prompt, whole-batch,
optimizer-state or GPU fit. Reduced architecture tests cover the runtime's
complete-plan checks, objective normalization and optimizer rollback separately.

## Scope and remaining work

The candidate has **2,926 distinct original training propositions**. Three
engineering readouts cannot complete its own-base snapshot. The existing
`EpisodeRewardAdapter` still requires complete matching anchors; that requirement
has not been weakened. The generation probe performs no LoRA insertion or
backward; the separate numerical derivative performs no optimizer update or
training.

The separate [LoRA runtime](qwen-lora-runtime-20260913.md) addresses BF16 hashing,
adapter-only state, selected vocabulary projection and explicit checkpointing
under eval. Its reduced actual-Qwen architecture tests establish implementation
behavior, complemented by the retained actual pretrained derivative above.
Candidate optimizer updates and scientific training remain unmeasured.

Scientific checkpoint/treatment selection, full own-base anchors, candidate
training and matched baselines remain unfinished. No provider inference request,
paid hardware allocation, billing change or credential operation is involved
in this local probe. Development-agent subscription usage is separate and has
not been measured by this experiment.

The original homepage remains at 23 ranked settings on 264 shared groups, with
Muse visible but unranked. The corrected comparison remains separate at 148
shared groups. These engineering outputs do not change either ranking.

The integrated repository suite passed **248 tests with no skips** in 35.498
seconds under the existing CPU and pinned tokenizer environments. Its log
SHA-256 is `9a51c27f0b44bc83da4340b01309adaa1ad86dc87343dd9bb9b5293c7959501e`.
Independent review separately reproduced all seven loader and 14 reduced LoRA
tests and checked the three actual readouts against all 13 token events.
It also checked the derivative's original input/source hashes, exact replayed
values, all 496 gradient summaries, unchanged weight manifests, phase events
and supervisor receipts without running another candidate process.
