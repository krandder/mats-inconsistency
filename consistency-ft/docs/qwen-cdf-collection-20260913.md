# Qwen corrected-CDF collection — first v3 tranche, 13 September 2026

The first v3 candidate invocation completed **two parse-valid EOS readouts** using
the corrected CDF sampler, with unchanged base and adapter weights. It generated
**nine tokens** and exited **0** after **333.585 external wall seconds**.
At **2026-09-13T13:31:06.009048+00:00**, coverage was **2/2,926 distinct full
propositions** from the original **1,619 training episodes**; **2,924 remained
untouched**, with **zero invalid or incomplete attempts** in this snapshot.

This is actual corrected candidate execution under a new pinned plan. No earlier
v1/v2 observation is counted or reused. The [evidence manifest](../data/research/qwen-cdf-collection-20260913.json)
contains only the first v3 tranche and its preparation evidence. No complete
own-base anchor, scientific model selection, optimizer update, backward pass,
training result or ranking is established.

## First invocation and readouts

Invocation `d4e87097-92d5-4663-b1c0-22401ff214c3` is recorded in the
[internal report](../data/research/qwen-cdf-collection-20260913/collection/invocations/d4e87097-92d5-4663-b1c0-22401ff214c3/report.json) and
[supervisor receipt](../data/research/qwen-cdf-collection-20260913/tranche-01/result.json).
The separate [terminal status](../data/research/qwen-cdf-collection-20260913/status-after-tranche-01.json) confirms
the same full-denominator counts. All v3 observations at this cutoff appear below;
full proposition IDs and original responses remain in the linked receipts and JSON.

| Catalog index / receipt | Seed | Status | Raw response | Prompt tokens | Completion tokens | Receipt seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| [0](../data/research/qwen-cdf-collection-20260913/collection/attempts/00378ee8bd196abc8a83ad6fd27d65a27f76ae40af5b10b1b957128f5e51c292/receipt.json) | 6678723356247500948 | probability | `0.35` | 277 | 5 | 117.442 |
| [1](../data/research/qwen-cdf-collection-20260913/collection/attempts/0041c4392b30880f8e9b9804eab765b363d2d856d760a5b1f05e91c0ea805828/receipt.json) | 7057277058622107250 | probability | `0.0` | 261 | 4 | 95.977 |

The `probability` status is the collector's retained parser/receipt category. Both
readouts reached EOS **248046** and passed the recorded runtime-integrity checks.
Each attempt retains its exact prompt, response, token IDs, sampled logprobs,
RNG-state hashes and timing. No attempt was retried, replaced, repaired or filtered.
These two observations alone do not validate population sampling behavior or
provide complete episode coverage.

| Measurement | First tranche / cumulative v3 |
| --- | ---: |
| Reserved attempts / parse-valid readouts | 2 / 2 |
| Generated tokens, including EOS | 9 |
| Internal wall seconds | 331.936 |
| External wall seconds | 333.585 |
| Internal CPU seconds | 510.483 |
| External user / system CPU seconds | 476.22 / 35.48 |
| Internal peak RSS, bytes | 20,777,639,936 |
| External peak RSS, bytes | 20,777,639,936 |
| External swaps | 0 |

The [GNU time report](../data/research/qwen-cdf-collection-20260913/tranche-01/time.txt) includes process shutdown
overhead. With only one v3 invocation, cumulative and per-invocation totals are
identical; no earlier campaign or software-validation resources enter these totals.
The [configuration](../data/research/qwen-cdf-collection-20260913/tranche-01-config.json) bound **two attempts / 64
tokens**, a **32-token fixed cap per proposition**, internal/external wall limits
**1,500/1,800 seconds**, a **10-second kill grace**, CPU soft/hard limits
**3,400/3,410 seconds**, and a **36 GiB** address-space limit. A new attempt required
room for the entire fixed cap. The nine generated tokens left **55 tokens** unused.

## New sampler and execution identity

The sealed source commit is `0af9ec3b6fe41b9b76996d318087a52e172f8035`. The
[corrected shared sampler](../training/categorical.py), algorithm
`softmax-at-least-fp32-fp64-cdf-uniform-right-v1`, has source SHA-256
`24d0e0132a5aa219f74c3146ad5516bbad6d81e0c6ecfe7bf37d7424c68475ba`.
Qwen retains its unfiltered FP32 softmax, full vocabulary, temperature 1 and
selected-token log probability. The helper normalizes cumulative probabilities in
FP64, draws one FP64 uniform variate per token and uses right-sided CDF search to
skip zero-mass bins. It avoids the installed exponential-noise endpoint path that
stopped v2. Finite-precision CDF and uniform resolution still apply; this is not a
claim of exact real-number arithmetic.

The new [plan](../data/research/qwen-cdf-collection-20260913/collection/plan.json) SHA-256 is
`a0d07bda3b81d880199e62b310caa43e8051f50b5e5da30f912d457f3439a5ce`; its
[spec](../data/research/qwen-cdf-collection-20260913/collection/spec.json) SHA-256 is
`5afa578df2f430d5add27f7e00c823b1bab722cdb1ef12086b7cabc5bf16491c`. The helper, runtime, collector and generic training-loop
source pins differ from v2. The corrected sampler identity and new runtime-source
binding are explicit execution fields, producing these distinct identities:

| Identity | Stopped v2 | Corrected v3 |
| --- | --- | --- |
| Execution SHA-256 | `587e09573b1ab7fcd568131ae860a141392137596eec5e90910fe81a3b73e476` | `0d2a9d55452046a8d0f71ae3495cc4ac2845c353f200781ed36b6bad71244d71` |
| Initial policy SHA-256 | `3a3d21b39d0576306d3f8c8c80de5f05f49bff6665e071880f3915b3c222e681` | `ffd73988534e4871d94a92ae6f170d0f80d50cb12d5a4d6dadfecc8051e0fe10` |

The model remains `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The training identity, initial
base tensors and seeded-A/zero-B adapter tensor manifests remain identical:
**248 rank-16/alpha-32 targets / 43,278,336 FP32 adapter parameters**, LoRA seed
**2026091304**. Execution retains the nonthinking `numeric-v1` protocol,
eval mode, eager attention, no cache, two CPU threads and one interop thread.
The historical initial-runtime input is reused only for its pinned tensor/reference
identity; the new spec explicitly replaces its runtime-source binding and adds the
CDF sampler. This invocation independently records and verifies the resulting new
execution identity before generation and unchanged base/adapters at completion.

The master seed remains **2026091310**, but collection ID and spec hash enter
per-proposition seed derivation. All **2,926 v3 seeds** were verified and differ from
v2. Proposition IDs, texts, prompts, order and the full original split remain the
same. V3 begins a separate ledger; it imports no old responses and does not replace
or reinterpret the stopped campaign's observations.

## Evidence, validation and subscription usage

The raw archive contains **30 files / 6,864,837 bytes**:
the new plan/spec/catalog, first-tranche attempts and invocation identities,
preparation receipts, supervisor configuration/source/logs/resource files, and
terminal status. **Seven byte-identical bound files** are reused by hash-checked
reference: the initial weight manifest, checkpoint/tokenizer manifests, historical
initial-runtime identity, and all three original splits. No old response file is
referenced as a v3 observation. Future v3 configs, envelopes and results are excluded.

The manifest maps every referenced input to its expected path inside `collection/`;
copying those exact bytes beside the retained new files reconstructs this snapshot.
The checkpoint tensor archive remains separate. Every raw or referenced file has a
byte count and SHA-256. Publication checks verified all **14 source pins**, all
**2,926 catalog rows and seed derivations**, the two attempt/receipt/token bindings,
new runtime identity, terminal state map and supervisor hashes. Earlier archives
were not changed.

The published [CDF software validation](../data/research/qwen-cdf-validation-20260913.json) passed **274 unique
repository tests**, with no unresolved failures or skips. Its initial run passed
268 and skipped six tokenizer tests due to an absent default artifact path; those
six then passed using the existing pinned artifacts. This publication ran no new
tests or model invocations. That prior software validation and these actual
candidate receipts provide different evidence; neither produces a complete anchor.

Preparation flags record no inference at creation time. The terminal status
command's `preflight-verified-no-inference` label describes that read-only command,
which generated no additional observations; the invocation receipts and terminal
counts record the actual two outputs.

This local CPU candidate invocation made **zero provider inference requests** and
used **zero provider inference subscription quota**, with no new paid resources
allocated. **Agent development/review subscription usage remains unmeasured** and
is separate from those zero inference-quota figures.
