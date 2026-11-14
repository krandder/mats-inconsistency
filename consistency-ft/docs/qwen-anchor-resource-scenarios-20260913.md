# Resources for complete Qwen own-base collection

The original training split requires **2,926 distinct full propositions** from
**1,619 episodes**. Under the existing proposed `numeric-v1` Qwen framing,
those distinct prompts contain **735,396 tokens**. This is the full anchor
catalog, without topic exclusions or a smaller replacement cohort.

The [retained calculation](../data/research/qwen-anchor-resource-scenarios-20260913.json)
joins every proposition to the previously verified real-tokenizer records and
pins its inputs. Unique anchor prompts have median **259**, 95th percentile
**313**, and maximum **406** tokens. The earlier median 211 / 95th percentile
305 describe all **4,855 role occurrences**, where repeated propositions count
again. Collection deduplicates them, changing the length distribution.

## Conditional CPU scenarios

The actual frozen-base probe measured 13 generated-token timings on three
selected prompts, using two CPU threads and no cache. Dividing each timing by
the corresponding full causal-prefix length gives observed coefficients of
0.06978–0.07952 seconds per prefix token. Applying those two coefficients to the
complete catalog produces the following **illustrative scenarios**:

| Assumed completion tokens per proposition, including EOS | Generated tokens | Generation hours |
| ---: | ---: | ---: |
| 4 | 11,704 | 57.4–65.4 |
| 8 | 23,408 | 115.6–131.8 |
| 16 | 46,816 | 234.9–267.7 |
| 32 | 93,632 | 484.3–551.9 |

For each prompt length `n` and assumed completion length `k`, the uncached model
processes `k*n + k*(k-1)/2` input-prefix tokens. The script sums that quantity
over all 2,926 distinct prompts. The ranges transfer the smallest and largest
observed coefficients; they are **not confidence intervals, throughput bounds,
or a prediction of completion time**. The three prompts were selected for an
engineering check, and actual answer lengths, CPU load and runtime overhead
remain unmeasured for a complete collection. The 32-token case includes possible
truncation and cannot imply valid anchor coverage.

The current public runtime verifies all frozen base bytes after each readout.
A separate combined base/adapter integrity audit took 15.714 seconds. If one
equivalent audit occurred for every proposition, it would add **12.8 hours**;
this is another transferred scenario, not measured collector overhead. Model
reloads, adapter installation, token-event fsync and other bookkeeping are also
excluded from the generation figures. The calibration used bare frozen weights;
the collection runtime with zero-initialized adapters needs its own measurement.

These results justify a durable full-catalog plan with bounded invocations and
explicit consumed-attempt accounting. Each invocation must report how much of
the original denominator is valid, invalid, interrupted and unattempted. Missing
answers cannot become a complete snapshot through retries, replacement values
or a reduced denominator. No inference, hardware allocation or training was
performed for this calculation.

## Reproduction

`scripts/estimate_qwen_anchor_resources.py` reads the original training split,
the retained tokenizer preflight, three generation readouts and their token
timings, and the numerical-backward phase log. It imports no model and performs
no inference or networking. Pass the full `prompts.jsonl` retained by the Qwen
tokenizer preflight; its byte size and SHA-256 must match the published pin.

```sh
python3 scripts/estimate_qwen_anchor_resources.py \
  --prompt-records /path/to/qwen-tokenizer-preflight/prompts.jsonl \
  --output /new/path/resource-scenarios.json
```

Actual measured loading, generation, derivative and memory results remain in
the [candidate execution report](qwen-candidate-execution-20260913.md). These
scenarios add full-catalog planning arithmetic; they do not replace those
measurements or establish full candidate training fit.
