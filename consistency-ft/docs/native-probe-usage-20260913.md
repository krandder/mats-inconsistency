# Native factual-recall subscription usage — 13 September 2026

All **1,161 native launches** have saved terminal observations and token counters. The nine one-call pilots were reused in the completed campaigns, not added again. No infrastructure retry or unreconciled launch remains. All 903 Astra/Luna native sessions match exactly 903 proxy requests across seven anonymized subscriptions.

No separately billed inference API calls were made. These measurements cover this factual-recall campaign; archived consistency runs and agent development/source-review work are separate.

## Native receipt counters

Input includes cached input; output includes reasoning output. Subset counters must not be added again. Native clients supply different context overheads and provider-specific accounting, so total input is not a direct measure of reasoning compute.

| Setting | Calls | Input | Cached input subset | Output | Reasoning output subset | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-6-astra / low | 129 | 1,114,091 | 697,856 | 2,763 | 1,793 | 1,116,854 |
| glm-5.3 / native-default | 129 | 35,349 | 2,560 | 18,092 | 17,343 | 53,441 |
| grok-4.6 / low | 129 | 1,802,846 | 248,064 | 28,531 | 27,684 | 1,831,377 |
| gpt-5.6-luna / high | 129 | 757,275 | 403,712 | 7,125 | 6,047 | 764,400 |
| gpt-5.6-luna / low | 129 | 757,287 | 384,256 | 3,140 | 2,187 | 760,427 |
| gpt-5.6-luna / max | 129 | 757,297 | 398,848 | 14,145 | 13,009 | 771,442 |
| gpt-5.6-luna / medium | 129 | 757,315 | 389,120 | 4,692 | 3,673 | 762,007 |
| gpt-5.6-luna / none | 129 | 757,293 | 384,256 | 836 | 0 | 758,129 |
| gpt-5.6-luna / xhigh | 129 | 757,239 | 393,984 | 8,673 | 7,574 | 765,912 |

**Total: 7,583,989 observed tokens**, including 3,302,656 cached input tokens. All 1,161 attempts supply normalized usage. Proxy token totals describe overlapping work and are not added to this total.

## Shared subscription meters

These are before/after readings of shared accounts, rounded by the provider. Concurrent fleet work can contribute. They establish neither exact marginal consumption by this experiment nor a token-to-quota conversion. A displayed zero change does not mean zero consumption.

| Provider / subscription | Window | Before | After | Comparable change |
| --- | --- | ---: | ---: | ---: |
| OpenAI / 1 | weekly | 12% | 12% | 0 percentage points |
| OpenAI / 2 | weekly | 32% | 32% | 0 percentage points |
| OpenAI / 3 | weekly | 61% | 62% | 1 percentage point |
| OpenAI / 4 | weekly | 1% | 1% | 0 percentage points |
| OpenAI / 5 | weekly | 5% | 5% | 0 percentage points |
| OpenAI / 6 | weekly | 57% | 67% | 10 percentage points |
| OpenAI / 7 | weekly | 30% | 31% | 1 percentage point |
| GLM coding plan | Auxiliary time allowance | 0% | 0% | 0 percentage points |
| GLM coding plan | 5-hour token allowance | 0% | 1% | Unknown: reset identity unavailable |
| GLM coding plan | Weekly token allowance | 1% | 1% | 0 percentage points |
| Grok SuperGrok Heavy | weekly | 2% | 3% | 1 percentage point |

Grok showed the same reset display, **September 19, 14:25**, before and after. GLM did not expose a reset identity for its five-hour window, so its 0% → 1% readings are retained without a percentage-point subtraction. OpenAI account numbers are anonymous labels; account identities are omitted.

The baseline Grok reading was captured at 07:51:51 UTC, before the pilots. The final quota capture started at 08:20:19 UTC, after all nine treatments completed. Exact capture timestamps and snapshot hashes are in the [machine-readable report](../data/research/native-probe-usage-20260913.json).

## Reproduction

```bash
python3 scripts/report_probe_usage.py \
  --run-dir runs/native-probes-20260913 \
  --output data/research/native-probe-usage-20260913.json
```

The default command is offline. `--capture` performs read-only quota and exact-session proxy checks; Grok percentages/reset text must be transcribed from the actual native `/usage` display. Private raw snapshots and session identities stay in the ignored campaign directory.

[Full factual-recall results](native-probes-20260913.md) · [Research progress and missing work](research-followups-20260913.md)
