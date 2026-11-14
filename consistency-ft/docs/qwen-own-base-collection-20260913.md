# Qwen own-base collection — first tranche, 13 September 2026

The corrected collector completed **two valid EOS probability readouts from the
full 2,926-proposition training catalog**. Its first bounded invocation took
**391.292 external wall seconds** and retained unchanged
base and adapter weights. Coverage is **2/2,926 propositions**, with **2,924 still
unattempted** at this snapshot. No complete own-base anchor or training result has
been produced. The [evidence manifest](../data/research/qwen-own-base-collection-20260913.json)
binds all retained files and measurements.

This report also preserves the earlier startup failure. The two invocations use
separate immutable plans. The snapshot ends at **2026-09-13T11:51:34.246250+00:00**; later
collection work is outside these counts.

## Measured invocations

| Measurement | Original startup (v1) | Corrected first tranche (v2) |
| --- | ---: | ---: |
| Supervisor exit code | 1 | 0 |
| Reserved / valid propositions | 0 / 0 | 2 / 2 |
| Generated tokens, including EOS | 0 | 8 |
| Internal wall seconds | 190.917 | 389.590 |
| External wall seconds | 192.063 | 391.292 |
| Internal CPU seconds | 152.328 | 491.442 |
| Internal peak RSS, GiB | 17.320 | 19.360 |

The original [internal failure report](../data/research/qwen-own-base-collection-20260913/collection/invocations/c4dcb4be-f356-4370-872f-25c58d65a1b4/report.json) and
[supervisor receipt](../data/research/qwen-own-base-collection-20260913/tranche-01/result.json) remain unchanged. The
corrected [internal report](../data/research/qwen-own-base-collection-20260913/v2/collection/invocations/71e30d44-5da6-4e2f-b470-ef2dc554fc43/report.json) and
[supervisor receipt](../data/research/qwen-own-base-collection-20260913/v2/tranche-01/result.json) record successful completion
and byte-verified unchanged base/adapters. The external `time.txt` files also retain
process resource totals, shutdown overhead and their separate peak-RSS readings.

## Initialization failure and correction

Source commit `6fad54bb9a780f5d0726c0b2d95d545b23639802` stopped before reserving any proposition.
The live Transformers configuration stores `id2label` keys as integers; JSON
stores object keys as strings. Direct Python dictionary equality rejected the
live execution object even though its canonical JSON and SHA-256 matched the
pinned plan. The loaded base, initial adapters, sampler and canonical execution
identity matched; the failure was in the collector's representation comparison.

A config-only reproduction confirmed the type conversion without constructing a
model or loading weights. Source commit `cb519181b92fb717399d4cda0ce92f979aacefb6` compares canonical
execution JSON bytes, preserves the existing exact execution hash and all
base/adapter/policy/sampler guards, and identifies any failing guard by name.
The added regression exercises live integer keys and rejects actual configuration,
hash, sampler and nonzero-initial-B drift. The integrated suite increased from
**262 tests / 52.765 seconds** to **263 tests / 56.231 seconds**, with no skips.
Both exact suite logs are retained in the archive.

## Immutable full-plan treatment

Both plans bind the original **1,619 training episodes / 2,926 distinct full
propositions**, plus the exact validation and held-out split bytes. The catalog
was ordered by full-proposition UTF-8 SHA-256 before inference. Only its first two
untouched IDs were attempted in v2; invalid answers would remain retained without
replacement or a smaller denominator.

The pretrained checkpoint is `Qwen/Qwen3.5-9B`, revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. The initial training identity includes
both frozen pretrained text tensors and all **248 rank-16/alpha-32 LoRA targets /
43,278,336 FP32 adapter parameters**: seeded A, zero B, and zero effective initial
adapter update. The LoRA seed is **2026091304**. The initial training identity,
base/adapter manifests, execution hash and policy hash are identical across v1
and v2; their exact values are in the evidence manifest.

The treatment uses the pinned nonthinking chat template and original `numeric-v1`
protocol, full-vocabulary categorical sampling at temperature 1, FP32 normalization,
no top-k/top-p filtering, eval mode, eager attention, and no cache. Generated EOS
is **248046**. The collection master seed is **2026091310**. The explicit new v2
spec necessarily derives different per-proposition seeds from its new spec hash;
v1 had sampled no tokens.

The first tranche was limited to **two attempts**, a **32-token fixed cap per
proposition**, and a **64-token invocation budget**. Each new attempt required room
for the entire fixed cap. The internal wall limit was **1,500 seconds**, external
wall limit **1,800 seconds** with a **10-second kill grace**, CPU soft/hard limits
**3,400/3,410 seconds**, and address-space limit **36 GiB**. The actual runtime used
two CPU threads and one interop thread. The eight sampled tokens left **56 tokens**
of invocation budget unused.

## Retained readouts

The labels below summarize the propositions. Their linked receipts and the full
catalog preserve the exact original messages and resolution criteria.

| Proposition | Per-proposition seed | Prompt tokens | Raw answer | Completion tokens | Receipt seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Italy referendum AND no NATO nuclear-policy adoption](../data/research/qwen-own-base-collection-20260913/v2/collection/attempts/00378ee8bd196abc8a83ad6fd27d65a27f76ae40af5b10b1b957128f5e51c292/receipt.json) | 567226646459662830 | 277 | 0.4 | 4 | 97.392 |
| [UK election OR Amazon US federal-law lobbying success](../data/research/qwen-own-base-collection-20260913/v2/collection/attempts/0041c4392b30880f8e9b9804eab765b363d2d856d760a5b1f05e91c0ea805828/receipt.json) | 16483053471083561756 | 261 | 0.9 | 4 | 97.202 |

Receipt time includes generation and the runtime's integrity checks. Every token
ID, sampled logprob, EOS flag, RNG-state hash and token interval is retained in
`tokens.jsonl`; prompt/completion IDs and raw responses are retained in the final
receipts. These two observations alone do not provide complete episode coverage,
a ranking, or evidence of training quality. No optimizer or backward pass ran in
these collection invocations, and no anchor snapshot or lambda/smoothing treatment
was selected.

## Evidence and subscription usage

The immutable raw archive contains **59 files /
26,818,115 bytes**, including both full plan/spec/catalog/input
sets, loaded checkpoint and initial-runtime identities, the original failure,
all new attempt/token/response receipts, supervisor configuration/source/logs,
resource reports and integrated test logs. Original v1 evidence stays at the
archive root; corrected evidence is under `v2/`. All raw files have byte counts
and SHA-256 values in the evidence manifest. The 19 GB tensor archive remains
local; only its authoritative artifact manifest is included. The runtime lock
file and unrelated Beads snapshots are omitted.

Plan flags such as `candidate_weights_loaded=false` describe the moment the
immutable plan was created before inference. Later invocation receipts record
actual checkpoint loading and, for v2, generation.

Both invocations made **zero provider inference requests** and used **zero provider
inference subscription quota**. They used existing local CPU resources and allocated
no new paid resources. **Agent development/review subscription consumption was not
measured** and is not represented by those zero inference-quota figures.
