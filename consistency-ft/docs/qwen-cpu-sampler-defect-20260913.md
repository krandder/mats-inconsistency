# Qwen CPU sampler defect — 13 September 2026

**The local Qwen collection stopped after a reproduced numerical defect in the
installed CPU sampler produced `0 Regiment`.** The retained response is invalid;
it has no assigned probability. The campaign contains **22 attempted
propositions, 21 parse-valid responses, one invalid response and 2,904 untouched
propositions**, out of the original 2,926. Parsing alone does not establish that
these observations follow the intended categorical sampling distribution.

Tranche 04 completed its eight already declared requests and exited normally.
The sequence stopped before tranches 05–07, as its inter-invocation validity
gate required. Every original receipt is preserved. No response was repaired,
replaced, clipped or retried; no complete anchor or training result exists.
See the [collection snapshot](qwen-own-base-collection-progress-20260913.md).

## What failed

At catalog index 20, the response token IDs were `[15, 70215, 248046]`.
The tokenizer faithfully decodes these as `0 Regiment` followed by EOS. The
recorded log probability of token 70215 was `-27.26654815673828`, corresponding
to nominal model mass of approximately `1.439755e-12`.

The frozen runtime passes FP32 softmax weights to `torch.multinomial` and
records the selected token's log-softmax from the same logits. In the installed
PyTorch `2.6.0+cpu` build, the one-sample CPU path generates exponential noise
`q`, computes `p / q`, and selects the largest score. Its MKL exponential
implementation shifts possible zero values by the smallest positive normal
float32 number, `2^-126`, rather than preventing this numerical endpoint.

A software-only reconstruction used the retained seed, vocabulary size 248,320,
and two-thread/one-interop configuration. It loaded no model and produced no
new benchmark answers. Its initial and subsequent generator-state hashes match
the retained hashes for **all 98 sampled token transitions across all 22
attempts**. Exactly one reconstructed transition exposed that float32 floor:
the invalid response's second token.

| Quantity at the defective draw | Reconstructed or retained value |
| --- | ---: |
| Seed | 16432092232865781795 |
| Selected token | 70215 (` Regiment`) |
| Exponential noise at selected token | 1.1754943508222875e-38 |
| Smallest noise at any other token | 4.1723342292243615e-6 |
| Selected token's nominal race score | approximately 1.2248e26 |
| Upper bound on every competing score, since probability ≤ 1 | approximately 239,673.992 |

The resulting margin is approximately **5.11e20**. This explains why an
otherwise extremely unlikely word wins this draw; it is stronger evidence
than merely noticing an unusual answer. A second agent reran the retained
diagnostic and reproduced all deterministic report fields and input hashes.
The two checks concern the same observations and do not add model samples.

## Evidence and limits

The [diagnostic manifest](../data/research/qwen-sampler-diagnostic-20260913.json)
links exact source and evidence hashes. It includes the
[98-transition reconstruction](../data/research/qwen-sampler-diagnostic-20260913/noise-replay.json)
and PyTorch sources pinned to the installed build's commit
`2236df1770800ffea5697b11b0bb0d910b2e59e1`:

- [Single-sample multinomial implementation](../data/research/qwen-sampler-diagnostic-20260913/Distributions.cpp), lines 584–614.
- [MKL exponential implementation](../data/research/qwen-sampler-diagnostic-20260913/DistributionKernels.cpp), lines 107–184.

The original full logits, probability vectors and exponential noise were not
retained. The noise values above are deterministic reconstructions tied to the
retained RNG trajectory. Nominal mass is computed from the retained
log-softmax; ordinary softmax/log-softmax rounding does not explain the
20-order dominance margin. This investigation identifies the observed failure
mechanism, not its population frequency or effect on aggregate model scores.
The 97 transitions without this floor do not certify the sampler generally.

The observation records still retain their original parser classifications.
The incomplete campaign must not be promoted to a verified categorical anchor,
merged with a future corrected campaign, or used for partial-anchor training.
This defect concerns the local CPU sampler; the site's native-provider ranking
data and HTML were not changed by this investigation.

## Reproduction

Use the retained PyTorch `2.6.0+cpu` environment and the immutable collection:

```bash
python -B scripts/diagnose_qwen_cpu_sampler.py \
  --collection /path/to/qwen-own-base-20260913-v2/collection \
  --vocab-size 248320 \
  --output /path/to/new-noise-replay.json
```

The output path must be new. To reconstruct the input from Git, copy the first
archive's `v2/collection/attempts` and the progress archive's `collection/attempts`
into one new collection directory, preserving their bytes. Input path fields
will differ from the original report; transition values and receipt hashes
should agree. This command executes only random-vector software diagnostics,
with no checkpoint, tokenizer forward pass, inference quota or provider calls.

## Correction status

The corrected [shared sampler](../training/categorical.py) uses normalized
FP64 cumulative probabilities and one FP64 uniform variate, with right-sided
search to skip zero-mass bins. Qwen retains its unfiltered FP32 softmax, full
vocabulary, temperature 1 and selected-token log probability. The generic
training path preserves existing FP64 inputs and promotes lower-precision
inputs to at least FP32. Finite-precision CDF and uniform resolution still
apply; the implementation does not claim exact real-number arithmetic.

[Independent validation](../data/research/qwen-cdf-validation-20260913.json)
executed **all 274 unique repository tests successfully**. The full discovery
run passed 268 and initially skipped six tokenizer tests because the isolated
checkout lacked their default artifact path; those six then passed using the
existing pinned artifacts. There are no unresolved skips or failures. An
initial generic FP64 precision regression was caught and corrected while
preserving the existing `1e-14` test, with no relaxed assertion or tolerance.

The helper's algorithm and source hash are bound to execution and policy
identities. The new code rejects the actual stopped v2 plan before execution;
every original collection file remained byte-identical during that check.
A corrected collection must use a separate, explicitly versioned full-catalog
plan, retain v2 unchanged and carry over none of its observations. Software
verification does not establish corrected candidate sampling or a complete
anchor. At the validation cutoff, no corrected candidate collection or
scientific training had run. The subsequent, separately versioned
[v3 collection](qwen-cdf-collection-20260913.md) now records its first two valid
responses against the full 2,926-proposition catalog; no earlier observations
were imported. Later requests appear in the [v3 continuation report](qwen-cdf-collection-progress-20260913.md).
Complete matching anchors and scientific training remain unfinished.
