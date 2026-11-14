# Qwen own-base collection progress — tranche 04 stop, 13 September 2026

**Collection is paused after a reproduced CPU sampler numerical defect produced
an invalid-format answer.** Tranche 04 exited **0** and retained unchanged base/adapter weights,
but only **7 of its 8 new readouts were parse-valid**. Catalog index **20** returned
the exact text **`0 Regiment`**, classified **`invalid_format`**. The already-admitted
final index 21 completed before the gate stopped the sequence; **tranches 05–07 were
never launched**.

At **2026-09-13T12:45:32.269196+00:00**, cumulative coverage was **22 attempted / 2,926
distinct full propositions** from the original **1,619 training episodes**:
**21 parse-valid**, **1 invalid-format**, and **2,904 untouched**. A complete own-base
anchor remains unavailable. Parse-valid counts describe accepted numeric output;
they do **not** certify categorical sampling correctness. The
[sampler diagnostic](qwen-cpu-sampler-defect-20260913.md) reproduces the numerical
failure mechanism without loading the model or generating new benchmark answers.

The [evidence manifest](../data/research/qwen-own-base-collection-progress-20260913.json) preserves all prior
observations and references the unchanged [first-tranche report](qwen-own-base-collection-20260913.md)
and [original manifest](../data/research/qwen-own-base-collection-20260913.json). This update adds only
tranche 04 execution evidence and sequence-plan provenance. Earlier raw files,
categories, catalog rows and the full denominator remain unchanged.

## Process completion and content validity

| Measurement | Original v1 startup | v2 tranche 01 | v2 tranche 02 | v2 tranche 03 | v2 tranche 04 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supervisor exit code | 1 | 0 | 0 | 0 | 0 |
| New reserved attempts | 0 | 2 | 4 | 8 | 8 |
| New parse-valid readouts | 0 | 2 | 4 | 8 | 7 |
| Invalid or incomplete attempts | 0 | 0 | 0 | 0 | 1 |
| Generated tokens, including EOS | 0 | 8 | 17 | 37 | 36 |
| Internal wall seconds | 190.917 | 389.590 | 539.399 | 985.765 | 858.680 |
| External wall seconds | 192.063 | 391.292 | 541.249 | 987.482 | 860.449 |
| Internal CPU seconds | 152.328 | 491.442 | 810.478 | 1632.107 | 1430.416 |
| Internal peak RSS, bytes | 18,597,564,416 | 20,787,425,280 | 20,821,467,136 | 20,955,373,568 | 20,937,084,928 |
| External peak RSS, bytes | 18,597,564,416 | 20,787,425,280 | 20,821,467,136 | 20,955,373,568 | 20,937,084,928 |

The [tranche-04 internal report](../data/research/qwen-own-base-collection-progress-20260913/collection/invocations/7c0eaac8-ce10-4c5a-8ca9-32a54286c6be/report.json) records
`bounded-invocation-complete` and unchanged base/adapters. Its
[supervisor receipt](../data/research/qwen-own-base-collection-progress-20260913/tranche-04/result.json) records exit **0** after
**860.449 external wall seconds**. Those operational results
coexist with the invalid content; the continuation gate requires every retained
attempt to pass its content checks as well.

The [invalid receipt](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01e7900e4ff2f5f8f9acd25ea816f192f56258497646500da6fe2cdca8e13b4c/receipt.json) retains `0 Regiment`, token IDs
**`[15, 70215, 248046]`**, EOS termination, runtime-integrity verification, and no
runtime or decode error. It has not been repaired, excluded, reclassified or
resampled. Its [per-token evidence](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01e7900e4ff2f5f8f9acd25ea816f192f56258497646500da6fe2cdca8e13b4c/tokens.jsonl) retains the
original logprobs, RNG-state hashes and timing. The malformed readout remains one
consumed proposition under the immutable one-attempt policy.

The [sequence envelope](../data/research/qwen-own-base-collection-progress-20260913/tranches-04-07-plan.json) requires stopping before the
next invocation if any attempt is invalid, interrupted or corrupt. Only tranche
04 ran. Its final index 21 was already admitted when the invalid result at index
20 was surfaced, so the running invocation reached terminal status before the
next-invocation gate canceled **05, 06 and 07**. There was no retry or sequence
continuation. All four prepared configs are retained solely as plan provenance;
the canceled configs have no execution results. The envelope's `execution_started=false`
flag records its preparation time, before the later tranche-04 receipts.

## Retained observations

The table retains the original ledger categories and all cumulative observations.
`probability` is the collector's existing parser/receipt status, not a certification
of the numerical sampler. Full IDs and complete raw responses remain in the linked
receipts and JSON evidence. Previously published response cells are unchanged.

| Catalog index / receipt | Seed | Status | Raw response | Prompt tokens | Completion tokens | Receipt seconds |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| [0](../data/research/qwen-own-base-collection-20260913/v2/collection/attempts/00378ee8bd196abc8a83ad6fd27d65a27f76ae40af5b10b1b957128f5e51c292/receipt.json) | 567226646459662830 | probability | `"0.4"` | 277 | 4 | 97.392 |
| [1](../data/research/qwen-own-base-collection-20260913/v2/collection/attempts/0041c4392b30880f8e9b9804eab765b363d2d856d760a5b1f05e91c0ea805828/receipt.json) | 16483053471083561756 | probability | `"0.9"` | 261 | 4 | 97.202 |
| [2](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/004ad1f323c8a5f50d2faa8d5e6d645427163827116dd6268c7312707a7f19d7/receipt.json) | 8884581637938796175 | probability | `"0.7"` | 271 | 4 | 97.600 |
| [3](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/0073f111bbd0fdaa5aa6c73104e385f881528f51aae01df39336611e75bd9e8e/receipt.json) | 4928698681944927268 | probability | `"0.1"` | 190 | 4 | 75.087 |
| [4](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/0077d26746b6914f74339338b22a6a1d66b1aab967916b7dd696e44d60f97bd7/receipt.json) | 3995281283215876853 | probability | `"0.6"` | 265 | 4 | 99.641 |
| [5](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/00c2e5f8456393d266e1828ecc34a7cd079732711efb9376e6ede10e5ac0d3d4/receipt.json) | 4609589840740929476 | probability | `"0.85"` | 266 | 5 | 117.834 |
| [6](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/00e3d234fa10328adeac37084e7f2622ac187e8c13625f2e82bac960bd83c744/receipt.json) | 12955290041849948153 | probability | `"0.4"` | 268 | 4 | 95.468 |
| [7](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/010a08725b061bf503857e9890d6658e580fb6daf53f996e03cc0cd12563ef8c/receipt.json) | 7761890251278331509 | probability | `"0.15"` | 305 | 5 | 119.587 |
| [8](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/011faeb917dea27df3b315194ebe006d963178e58b5345273a4ed8413eb847d0/receipt.json) | 15834833580562657961 | probability | `"0.15"` | 279 | 5 | 118.917 |
| [9](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/012e0b198a99e660d4efb8ec2638e225f2b3a3ff3f2ac6ef84f3b50bd4073541/receipt.json) | 7902932084576048674 | probability | `"0.25"` | 279 | 5 | 117.081 |
| [10](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/0136e72569a48ccd5a6dd71faacc9d6d9e923ab239fb843f797e0600e82b7221/receipt.json) | 4804042621389476033 | probability | `"0.05"` | 278 | 5 | 118.943 |
| [11](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01546cb04ee78862e8f362e8a522718912fb82f721be2674b81b0278c57ef057/receipt.json) | 7653179346169687034 | probability | `"0.3"` | 281 | 4 | 97.319 |
| [12](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/016e558e2e4dae2d3095adfd23ee4c89a4b107040eeec51e5be7470d27326724/receipt.json) | 8115972574989275170 | probability | `"0.04"` | 340 | 5 | 121.777 |
| [13](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/018f1fd678d2d882b5a70627a9450460592f2886eee78029808cc71e77d69f1d/receipt.json) | 15354064846932850524 | probability | `"0.3"` | 188 | 4 | 73.661 |
| [14](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01a95b230a84eee2418761363996aef1d7908fef3acee5a5bd4f15c56bdf0f1a/receipt.json) | 16396162168645836896 | probability | `"0.4"` | 190 | 4 | 75.118 |
| [15](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01b7a955c4840d5c4f30e6ced76d0535c5eb2a0ba8658cb748660b9a5a6dfcc0/receipt.json) | 15131606148999170209 | probability | `"0.15"` | 283 | 5 | 117.705 |
| [16](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01c88abadef586e414abda9062907031ace70085628a84689507d04c772ad346/receipt.json) | 9175667241553167220 | probability | `"0.15"` | 193 | 5 | 88.204 |
| [17](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01ccffe39d41183e710044e73f03911676be6e5d3f8bd63f4224af613d86af89/receipt.json) | 3096160981290789670 | probability | `"0.15"` | 300 | 5 | 119.753 |
| [18](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01d7a30553d5db146476639f1330b00960a874a89167dc2276273aafecf16e2c/receipt.json) | 15933501355297153733 | probability | `"0.25"` | 251 | 5 | 95.656 |
| [19](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01db5e2353af9204dd6be0f54ed97fb425fcfe8ee71ae157839bec5834747135/receipt.json) | 6105117470150716504 | probability | `"0.65"` | 225 | 5 | 94.012 |
| [20](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01e7900e4ff2f5f8f9acd25ea816f192f56258497646500da6fe2cdca8e13b4c/receipt.json) | 16432092232865781795 | invalid_format | `"0 Regiment"` | 199 | 3 | 60.576 |
| [21](../data/research/qwen-own-base-collection-progress-20260913/collection/attempts/01ec9b8f4506e62f731a77ea40967c1ca5c2d467412d75bc72d8d016a70296c7/receipt.json) | 700199591908632712 | probability | `"0.1"` | 207 | 4 | 76.220 |

The latest invocation resumed the exact existing v2
[plan](../data/research/qwen-own-base-collection-20260913/v2/collection/plan.json), SHA-256
`4bdec0c2c108a108f72974c0cbdc610038d16f22b44918cd3dcbde7439cec736`, with unchanged source commit
`cb519181b92fb717399d4cda0ce92f979aacefb6`. Its eligible list began at index 14;
the prior fourteen IDs remained consumed. New indices 14–21 followed the pinned
catalog ordering and their original per-ID seeds.

## Resources, treatment and reproduced sampler defect

Tranche 04 generated **36 tokens**, leaving **220** of its **256-token invocation
budget** unused. The [configuration](../data/research/qwen-own-base-collection-progress-20260913/tranche-04-config.json) retained
**8 attempts**, the **32-token fixed cap per proposition**, internal/external wall
limits **1,500/1,800 seconds**, a **10-second kill grace**, CPU soft/hard limits
**3,400/3,410 seconds**, and a **36 GiB** address-space limit. Each new attempt still
required room for the entire fixed cap. The [GNU time report](../data/research/qwen-own-base-collection-progress-20260913/tranche-04/time.txt)
records **1354.45 user CPU seconds** and
**77.20 system CPU seconds**, including process shutdown overhead.

Across four v2 invocations, all **22 attempts**, including the invalid one, used
**98 generated tokens**, **2773.434 internal wall seconds**,
**2780.473 external wall seconds** and
**4364.443 internal CPU seconds**. Including the retained v1
startup failure gives **2964.352 internal wall seconds**,
**2972.536 external wall seconds** and
**4516.772 internal CPU seconds** across five invocations.
The startup failure reserved no propositions and generated no tokens.

Cumulative internal/external peak RSS is **20,955,373,568 bytes**.
Wall and CPU totals sum candidate-invocation measurements and exclude idle time;
peak memory is the maximum across invocations. Separate diagnostic work is outside
these candidate resource totals. A nonzero supervisor exit and invalid content are
tracked separately: v2 has **zero nonzero process exits** and **one invocation with
invalid content**.

The checkpoint remains `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, master seed **2026091310**,
LoRA seed **2026091304**, and 248 rank-16/alpha-32 targets with 43,278,336 FP32
adapter parameters, seeded A and zero B. The retained configuration declares the
nonthinking `numeric-v1` protocol, full-vocabulary categorical sampling at
temperature 1, FP32 normalization, no filters, eval mode, eager attention, no cache,
two CPU threads and one interop thread. Initial loaded-checkpoint, runtime and
tokenizer identity files match the first tranche byte for byte. These identity
checks do not establish numerical sampler correctness.

The [separate diagnostic report](qwen-cpu-sampler-defect-20260913.md) and its
[evidence manifest](../data/research/qwen-sampler-diagnostic-20260913.json) reproduce
all **98 retained RNG transitions**. Exactly one reconstructed transition reaches
the smallest positive normal FP32 exponential noise, **1.1754943508222875e-38**:
index 20's second token, **70215**. In this installed one-sample CPU implementation,
the sampler selects the largest `p / q` value. The reconstructed floor makes that
token dominate the competing scores and explains the malformed response. An
independent software-only rerun reproduced the deterministic diagnostic fields.

The original full logits, probability vectors and noise were not retained; the
noise values are deterministic reconstructions tied to the retained RNG hashes.
This identifies the observed failure mechanism, without measuring its population
frequency or certifying the other 97 transitions. Collection remains paused.
A correction is being implemented separately, with **no corrected candidate run**.
The stopped campaign stays immutable and must not be merged with a future corrected
campaign or promoted to a verified categorical anchor.

## Evidence and subscription usage

This update adds **36 execution files / 825,493 bytes** and
**4 preparation-provenance files / 12,011 bytes**.
The progress archive now contains **100 raw files / 2,458,496 bytes**.
All **60** earlier progress raw files and **20** first-tranche references
remain unchanged. The separate first-tranche archive remains **26,818,115 bytes**.
Each retained file has its byte count and SHA-256 in the manifest. No checkpoint
tensors, unrelated files or unexecuted-tranche results are included.

Publication checks reproduced the full terminal state map from unchanged earlier
states plus the new receipts, verified all new supervisor/attempt/token/receipt
pins and bindings, preserved the invalid string exactly, and checked the sequence
envelope and every prepared config hash. No source changes, new tests or extra
model executions were performed. The previous
[263-test integrated record](../data/research/qwen-own-base-collection-20260913/v2/integrated-suite.log) remains historical
implementation validation and does not certify the defective sampler. The linked
diagnostic is root-owned evidence with separate resource scope; no diagnostic files
or results were duplicated into this raw progress archive.

The retained local CPU candidate invocations made **zero provider inference requests**
and used **zero provider inference subscription quota**, with no new paid resources
allocated. **Agent development/review subscription use remains unmeasured** and is
separate from those inference-quota figures. No complete anchor, optimizer update,
backward pass, training result or model ranking is claimed.
