# Native subscription evaluation, September 2026

The completed corrected campaign, `native-20260913-scoped-final`, contains
**2,466 observations across nine configurations**: **2,450 numerical probabilities
and 16 explicit abstentions**. All configurations completed their 274 assigned
questions. Completion does not mean every question received a probability.

GLM-5.3, Grok-4.6, GPT-6 Astra, and all six GPT-5.6 Luna reasoning levels use
General's existing authenticated subscriptions. No separately billed inference
API key was used. Native clients still communicate with their providers;
subscription access describes authentication and billing, not offline inference.

## Original page and separate corrected comparison

The original overview retains the combined Model ranking chart, all-model heatmap,
Elicitation readouts, worst examples, and Runs. Supplementary ranking, coverage,
and usage tables are on the corresponding details page.

| Page | Data and comparison |
| --- | --- |
| [Original benchmark](https://consistency.futarchy.ai/) | The original 55 runs, including the nine `numeric-v1` native configurations; ranking and heatmap use the same 264 scored groups for all 23 participating settings. |
| [Original details](https://consistency.futarchy.ai/details.html) | Original coverage and means over each setting's available groups, plus supplementary tables. |
| [Corrected comparison](https://consistency.futarchy.ai/corrected.html) | Final reviewed scope of 157 groups / 274 questions; ranking and heatmap use the same 148 scored groups for all 23 participating settings. |
| [Corrected details](https://consistency.futarchy.ai/corrected-details.html) | Coverage over the full corrected target, per-setting means, imports, readout status, and subscription accounting. |

Both overviews show all **24 model/reasoning settings**, including all six Luna
efforts. Muse remains visible but unranked because its archived run is incomplete.
The declared comparison policy, `complete-historical-plus-native`, includes
historical settings that completed their original target and every requested
native setting. It does not let Muse's small sample shrink the comparison for
all other models. Missing probabilities still shrink the intersection among
participating settings; this selection effect remains a limitation.

The original overview selects `native-20260912` explicitly. The corrected overview
selects the final reviewed campaign. The database retains **64 runs: the original
55 plus the nine final configurations**. Earlier unpublished small-scope runs are
excluded from public raw exports because their scope included four subsequently
held questions; eligible observations survive through direct imports into the
final campaign. Interrupted campaigns contribute aggregate audit and usage
evidence. The separate [Open Agency page](https://consistency.futarchy.ai/open-agency.html)
remains available.

## Final corrected results

Every configuration completed **274/274 observations**. The table below reports
its own full reviewed-target coverage and mean; these denominators differ when
models abstain. It is not the shared-set ranking displayed on the overview.
Mean algebraic violation is computed by `common.violation`; lower is better.

| Configuration | Numerical probabilities | ABSTAIN | Scored groups | Mean over scored groups | Imports / fresh receipts |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 · native default | 274/274 | 0 | 157/157 | 0.067298 | 133 / 141 |
| Grok-4.6 · low | 273/274 | 1 | 156/157 | 0.064153 | 133 / 141 |
| GPT-6 Astra · low | 269/274 | 5 | 150/157 | 0.115713 | 182 / 92 |
| GPT-5.6 Luna · none | 269/274 | 5 | 150/157 | 0.122420 | 224 / 50 |
| GPT-5.6 Luna · low | 273/274 | 1 | 151/157 | 0.103762 | 210 / 64 |
| GPT-5.6 Luna · medium | 272/274 | 2 | 155/157 | 0.090462 | 196 / 78 |
| GPT-5.6 Luna · high | 273/274 | 1 | 151/157 | 0.078171 | 188 / 86 |
| GPT-5.6 Luna · xhigh | 273/274 | 1 | 151/157 | 0.095123 | 156 / 118 |
| GPT-5.6 Luna · max | 274/274 | 0 | 157/157 | 0.065706 | 132 / 142 |

The campaign contains **1,554 immutable imports and 912 fresh receipts**. Imports
come directly from the interrupted full-v2 campaign (138), the earlier small
subset (873), and the stopped expansion (543). Imported abstentions are retained;
they are neither replaced nor counted as new requests. The nine native settings
share 148 scored groups, which is also the corrected comparison's intersection
with the eligible historical settings.

The [final audit](../data/benchmark-runs/final-native-audit-20260913.json) verified
all 2,466 unique native sessions, exact source/model/effort/protocol provenance,
raw native output and applicable final-file agreement, import-source hashes,
absence of held IDs, and recomputed group scores. All nine final runners exited
successfully. There were no failed turns, malformed final readouts, or tool-use
failures. One imported Luna-medium session reconnected before its successful
final event; it is retained as a recovered connection, not a failed observation.

These violations measure logical coherence, not forecast accuracy, calibration,
or the paper's arbitrage metric. Historical Fireworks and native measurements
differ in date, protocol, reasoning controls, and client scaffolding. Equal group
IDs control the comparison sample; they do not isolate the effect of model choice.

## Frozen source scope

The original catalog is the first 30 groups from each of ten checked-in tuple
files: **300 groups containing 591 distinct questions**. The final source review
retains **157 complete groups and 274 questions**, covering all ten check types.
Its identity is `original300-reviewed-final-v3`, SHA-256
`bb38d378d4a1164ee0f6e6f224a2430bae21db6d5f98c300aa5e5d22b029cd70`.

The [frozen manifest](../data/benchmark-runs/scope-bb38d378d4a1.json) records source
fingerprints, included IDs, complete-group membership, and exclusion reasons.
The 317 omitted questions consist of **253 held targets**, **20 eligible companions
lost through topic-group filtering**, and **44 further eligible questions lost
through semantic-group filtering**. These are unique-question counts, not model
failures. At group level, 128 groups are excluded by topic/boundary criteria and
15 more by semantic defects. Overlapping topic and semantic reasons are preserved.

The [source topic review](../data/benchmark-runs/source-topic-review-20260913.json)
and [semantic review](../data/benchmark-runs/source-semantics-review-20260913.json)
document the decisions; the [dated scope report](political-constraints-and-missing-work-20260913.md)
and [omitted-question inventory](omitted-questions-20260913.md) retain their history.
The final manifest was frozen before final-campaign collection. Response values
were not used to select the scope. Source wording remains unchanged. Validation
preserved all previously identified semantic exclusions and checked the whole-group
union; it does not constitute a formal proof of every retained group's semantics.

## Protocol and model settings

Each distinct proposition, including its complete resolution criteria, is asked
in a fresh native session. Groups sharing a proposition reuse that observation.
The final campaign uses the fixed `descriptive-v2` protocol: estimate an event's
likelihood using existing knowledge, with no tools, browsing, memory, or delegation;
return exactly one decimal probability in [0, 1] or `ABSTAIN`. Exact system and
user templates are stored in manifest `native_protocols` and private `campaign.json`.
Receipts record the protocol and system-instruction hash.

| Configuration | Requested reasoning | Native client |
| --- | --- | --- |
| GLM-5.3 | Native default | OpenCode 1.18.29 through the existing Z.ai coding-plan subscription |
| Grok-4.6 | Low | Grok Build 1.0.13 through the existing OAuth subscription |
| GPT-6 Astra | Low | Codex through the existing subscription pool |
| GPT-5.6 Luna | None, low, medium, high, xhigh, max, each separately | Codex through the same pool |

Grok receipts identify the served model as `grok-4.6-build`. Other clients do not
expose an immutable backend build; none is inferred. Reasoning effort is a
requested treatment, not a promise of identical internal compute. GLM and Grok
request temperature 0 and a 2,048-token completion limit where supported. GLM's
native reasoning can exceed that requested output limit. Codex provides different
controls; native usage receipts preserve the observed counters.

Native subscriptions do not expose the historical True/False token logprobs.
Their Elicitation readout rows therefore show paired count zero and unavailable
paired metrics. The table does not invent a logprob estimate.

## Readout classification and operational corrections

The [original readout audit](../data/benchmark-runs/native-readout-audit.json)
confirmed that all **157 missing original probabilities were explicit refusals**,
covering 50 distinct propositions. Each completed its native turn, returned usage,
and matched the native final message and saved file. No numerical answers were
recoverable from those responses. The broad original error label was corrected;
original responses, scores, and timestamps remain historical evidence.

Under `descriptive-v2`, `ABSTAIN` is a completed observation with a null probability.
Free-text refusals, malformed replies, tool use, and infrastructure failures have
separate classifications. A group is scored only if all required probabilities
are available. Refusals and abstentions are not resampled to obtain a number.
Only infrastructure failures may be retried, with prior attempts preserved.

The initial fixed diagnostic produced 54 completed readouts: 39 probabilities
and 15 abstentions, with no malformed, transport, or tool-use failures. Diagnostic
calls are excluded from scores and included in usage. The interrupted campaigns
and subsequent source review are documented in the
[remediation report](benchmark-remediation-report-20260913.md).

Four earlier GLM launches stalled locally with no observed output; actual backend
requests and consumption remain unknown. The corrected runtime uses an empty
working directory outside Git and disables native FFF, file watching, and
snapshots. Receipts identify `opencode-external-cwd-no-index-v1`; the final GLM
set contains 271 observations from that profile and three compatible earlier
observations. Prompts, model, reasoning, and completed responses were retained.

The runner also gives each OpenCode call a temporary directory on the run volume
and removes it after exit. Codex uses a prepared private runtime on the data
volume when selected, retaining the existing subscription-proxy configuration.
These changes address client startup and filesystem problems; they do not
reinterpret abstentions as software defects.

## Subscription accounting

The final measurement, closed at **04:55:52 UTC on September 13**, records
**6,803,718 native tokens for 912 fresh receipts and 912 launches**, with no
interrupted launches or missing native usage in this last campaign. Its 1,554
imports add no new usage. All 630 fresh Astra/Luna sessions matched the existing
subscription proxy's 630 corresponding request records.

| Originating campaign | Native tokens, including setup | Request receipts | CLI launches |
| --- | ---: | ---: | ---: |
| [Original numeric-v1](../data/benchmark-runs/subscription-usage-native-20260912.json) | 37,860,370 | 5,382 | 5,382 |
| [Interrupted full-v2 and fixed diagnostic](../data/benchmark-runs/subscription-usage-native-20260913-neutral.json) | 3,471,259 | 477 | 513 |
| [Earlier small subset](../data/benchmark-runs/subscription-usage-native-20260913-scoped.json) | 6,604,478 | 909 | 913 |
| [Stopped expansion](../data/benchmark-runs/subscription-usage-native-20260913-scoped-expanded.json) | 6,999,049 | 997 | 1,033 |
| [Final reviewed campaign](../data/benchmark-runs/subscription-usage-native-20260913-scoped-final.json) | 6,803,718 | 912 | 912 |
| **Cumulative, each campaign once** | **61,738,874** | **8,677** | **8,753** |

Receipt counts include returned setup and infrastructure-attempt records where
applicable; they are not counts of scored benchmark answers. Across all campaigns,
native usage is unavailable for **106 launches**: 76 interrupted invocations
without receipts and 30 earlier receipts without native usage telemetry.

The final quota interval ran from approximately **04:30:49 to 04:55:52 UTC**.
These are changes in displayed account usage, not exact experiment fractions:

| Subscription window | Before | After | Change, percentage points |
| --- | ---: | ---: | ---: |
| GLM five-hour token window | 4% | 5% | 1 |
| GLM weekly token window | 1% | 1% | 0 |
| Grok weekly | 2% | 2% | 0 |
| OpenAI subscription 1, weekly | 12% | 12% | 0 |
| OpenAI subscription 2, weekly | 29% | 31% | 2 |
| OpenAI subscription 3, weekly | 56% | 59% | 3 |
| OpenAI subscription 4, weekly | 1% | 1% | 0 |
| OpenAI subscription 5, weekly | 5% | 5% | 0 |
| OpenAI subscription 6, weekly | 42% | 49% | 7 |
| OpenAI subscription 7, weekly | 24% | 25% | 1 |

Grok's weekly reset remained September 19, 14:25. GLM's auxiliary `TIME_LIMIT`
window changed reset identity, so no delta is attributed for that window.

The [usage report](../data/benchmark-runs/subscription-usage.json) separates fresh
launches, completed receipts, imports, setup calls, interrupted invocations,
and unknown native usage. Each originating campaign is counted once; reusing an
observation does not add its tokens again. Archived campaign reports preserve
measurement intervals and quotas.

Recorded token totals are telemetry totals, not the providers' subscription-limit
units. Cached input and reasoning counters are normalized according to each
client's semantics and counted once. Native usage can be unavailable even when
there is proxy evidence; **189,590 proxy-only tokens from interrupted expansion
requests are reported separately**, not silently added to the native total.
Unknown usage is not zero. Provider dollar estimates are not extra subscription
charges or money spent.

Quota snapshots are rounded readings from shared accounts. Other fleet sessions,
reset boundaries, and different account windows prevent exact marginal attribution.
Percentage-point changes are not summed across accounts or campaigns, and tokens
are not converted to quota percentages without a provider-exposed conversion.
Grok's percentage comes from its native `/usage` view. GLM and OpenAI snapshots
retain their window identities so incomparable before/after windows are disclosed.

## Rebuild and audit

The public archive can be rebuilt without subscriptions:

```sh
python3 scripts/export_benchmark.py
python3 scripts/build_dash.py
python3 -m unittest discover
```

Tracked allowlisted JSONL and the [manifest](../data/benchmark-runs/manifest.json)
are the reproducible sources for `data/benchmark.sqlite` and the dashboard copy.
The manifest hashes exported files. Public receipts retain readout classification,
probability, usage, timing, and provenance hashes; credentials, account names,
and private session IDs are excluded. Historical values remain intact.

Refreshing from the already collected private final receipts is explicit:

```sh
python3 scripts/export_benchmark.py --refresh \
  --campaign native-20260913-scoped-final \
  --comparison-policy complete-historical-plus-native
python3 scripts/build_dash.py
```

A new collection requires a distinct campaign for any changed protocol or scope.
Resume requires identical model, effort, prompt, system instruction, and cohort.
The runner validates the frozen scope before collecting. Repeated
`--import-run-dir` arguments import compatible completed observations in priority
order, preserving direct source hashes. Supply original campaigns before copies
of their imported observations. Existing destination observations always win,
including refusals and abstentions. `--diagnostic` excludes the entire campaign
from scores permanently.

Database runs and probability views retain `protocol_id`; model plus reasoning
effort remain distinct identities. `comparison_ranking` uses the corrected shared
group intersection, while full run aggregates and per-setting scope coverage stay
available separately. `scope_exclusions` records review decisions and
`receipts.is_imported` separates reused observations from fresh launches.

Final publication validation passed 62 unit tests, SQLite integrity and foreign-key
checks, all 2,466 private receipt audits, exact preservation of the original
55 score-file hashes and timestamps, and desktop/mobile checks of both comparisons.
Muse has no ranked score; all six Luna efforts have separate database identities.


**Publication verified:** 2026-09-13T05:01:24.450080+00:00. [Repository delivery](https://github.com/futarchy-fi/consistency-ft/commit/d856c3de3beafdc844b899e4eda201bbf986d91a) and all four live result pages are published. The database, manifest, reviewed scope, and subscription report match the validated artifacts; Open Agency and project-task pages remain available.
