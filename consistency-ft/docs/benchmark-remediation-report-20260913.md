# Consistency benchmark: what was done and what remains

**Latest update, 04:56 UTC: final run audited.** All **2,466 observations** are complete:
**2,450 numerical answers and 16 abstentions**, across 274 questions × nine settings.
The original homepage comparison uses **264 shared groups**; the separate corrected
comparison uses **148 shared groups**. Both show 24 settings, with Muse unranked.
The dated sections below preserve earlier states; see [final completion](#final-completion).

**Publication verified:** 2026-09-13T05:01:24.450080+00:00. [Repository delivery](https://github.com/futarchy-fi/consistency-ft/commit/d856c3de3beafdc844b899e4eda201bbf986d91a) and all four live result pages are published. The database, manifest, reviewed scope, and subscription report match the validated artifacts; Open Agency and project-task pages remain available.


**Correction recorded at 04:27 UTC:** the expanded run was stopped after U.S. policy
forecasting questions were found in its selected scope. Its earlier 445-question
eligibility claim was too broad. See [political constraints and missing work](political-constraints-and-missing-work-20260913.md)
for the current account and [all 483 questions omitted from the completed small rerun](omitted-questions-20260913.md).
The dated sections below preserve earlier decisions, including decisions now corrected.

Status recorded on 13 September 2026, approximately 04:00 UTC. This report covers
the addition of GLM, Grok, Astra, and the six Luna reasoning settings, and the
subsequent readout audit and rerun. It describes the work actually performed;
the proposed recovery of unnecessarily excluded questions has not happened yet.

## The distinction that was not communicated clearly

**The original full run attempted every question. The later rerun omitted most
questions. These are different campaigns.**

| Measurement | Original native run | Later completed rerun |
| --- | ---: | ---: |
| Unique questions per model/reasoning setting | 591 | 108 |
| Model/reasoning settings | 9 | 9 |
| Completed observations across settings | 5,319 | 972 |
| Numerical probability observations | 5,162 | 972 |
| Refusals among completed observations | 157 | 0 |
| Original question groups included | 300 | 62 |
| Original questions excluded before this run | 0 | 483 |

The **972** figure means **108 questions × nine settings**, including 63 reused
observations. It does not mean 972 distinct questions, 972 fresh requests, or a
complete rerun of the original benchmark.

The rerun excluded **483/591 questions, or 81.7%**. Furthermore, the prepared
ranking across all 24 historical and new settings uses only **20 shared question
groups out of the original 300**. The nine new settings themselves have results
for all 62 retained groups.

**The remediation is not complete, and its repo/site update has not been
published.** The completed subset is usable evidence about that subset. It does
not satisfy the request for a corrected comparison on the full original dataset.

## 1. Dataset and model setup

The source was the repository's existing `data/tuples` catalog. The loader took
the first 30 records from each of ten JSONL files, in sorted filename order:
300 logically related question groups, called *tuples* in the code.

The ten checks are Negation, Paraphrase, Consequence, And, Or, AndOr, But, Cond,
CondCond, and ExpEvidence. A group contains the probabilities needed for its
particular consistency equation. A question can occur in several groups.

For each question, the harness constructed exactly:

```text
<title, with surrounding whitespace removed>

Resolution criteria: <body, with surrounding whitespace removed>
```

If the body was empty, it used just the title. It deduplicated this complete text
using SHA-256, giving **591 unique propositions**. Repeated appearances of the
same text within one model setting reused one observation. It did not merge
merely similar questions or rewrite their resolution criteria.

| Setting | Native client and subscription |
| --- | --- |
| GLM-5.3, native default reasoning | OpenCode through the existing Z.ai coding plan |
| Grok-4.6, low reasoning | Grok Build through the existing OAuth subscription |
| GPT-6 Astra, low reasoning | Codex through the existing OpenAI subscription pool |
| GPT-5.6 Luna, none | Codex through the same pool |
| GPT-5.6 Luna, low | Codex through the same pool |
| GPT-5.6 Luna, medium | Codex through the same pool |
| GPT-5.6 Luna, high | Codex through the same pool |
| GPT-5.6 Luna, xhigh | Codex through the same pool |
| GPT-5.6 Luna, max | Codex through the same pool |

These are nine settings, not nine different base models. Fable was not included
in this batch. Grok's returned model identity was `grok-4.6-build`; the other
clients did not supply an immutable backend build identifier in their usage
receipts. The requested reasoning setting is recorded, but does not guarantee
an identical amount of internal computation on each question.

Each observation used a fresh native session, an empty working directory, and
existing subscription authentication. Tools, browsing, file access, project
instructions, and delegation were disabled or forbidden by the harness and
fixed instruction. Returned events were inspected for actual tool use. The
native clients' own scaffolding still existed.

Grok and GLM requested temperature 0 and a 2,048-token completion limit where
supported. Codex exposed different controls. GLM's native reasoning could exceed
the requested completion limit. This was not an identical transport or compute
environment across providers. No separately billed inference API key was used;
native subscription clients still made network requests to their providers.

Sources: [loader and equations](../common.py),
[native harness](../harness/run_native_checks.py),
[campaign and prompt manifest](../data/benchmark-runs/manifest.json).

## 2. Original full run: `native-20260912`

The original protocol was `numeric-v1`. Its fixed instruction was:

```text
Answer the user's probability question from your existing knowledge. Return only the probability as a number between 0 and 1. Do not use tools, browse, read files, consult memory, or delegate.
```

Its user prompt was:

```text
Estimate the probability that the proposition below is true.
Return only a number from 0.0 to 1.0, with no commentary.

Proposition:
{proposition}
```

All **591 × 9 = 5,319 planned observations** completed. A completed observation
could contain a refusal instead of a probability:

| Setting | Numerical answers / 591 | Refusals | Fully scorable groups / 300 |
| --- | ---: | ---: | ---: |
| GLM-5.3, native default | 591 | 0 | 300 |
| Grok-4.6, low | 591 | 0 | 300 |
| Astra, low | 569 | 22 | 276 |
| Luna, none | 591 | 0 | 300 |
| Luna, low | 569 | 22 | 280 |
| Luna, medium | 565 | 26 | 272 |
| Luna, high | 560 | 31 | 272 |
| Luna, xhigh | 562 | 29 | 268 |
| Luna, max | 564 | 27 | 278 |
| **Total observations** | **5,162** | **157** | — |

Therefore, GLM, Grok, and Luna-none already had numerical answers for every
original question. Other settings had partial scoring coverage. An individual
missing probability can prevent several groups from being scored.

There were **5,382 CLI receipts**, comprising 5,319 planned observations,
31 excluded setup calls, and 32 infrastructure attempts. Setup included an
initial GLM one-step configuration that injected an unwanted summary instruction;
those outputs were excluded from scores. The original campaign also encountered
a full root filesystem. Codex was resumed using a private runtime on the data
volume with the same model, configuration, and subscription proxy.

Sources: [original usage](../data/benchmark-runs/subscription-usage-native-20260912.json)
and the nine `native-20260912-*.receipts.jsonl` files in
[benchmark data](../data/benchmark-runs/).

## 3. Audit of the reported “invalid probabilities”

The audit inspected all **157 missing original readouts**, spanning **50 distinct
question texts**. It compared saved responses against raw native message events
and applicable final-output files, checked successful completion and token
usage, and looked for extraction failures, truncation, stale final files, and
tool calls. The audit itself made no inference requests.

All 157 were **explicit refusals**. None contained a numerical answer that could
be recovered by fixing a parser. Of their responses, 141 explicitly referred to
politics; 16 used generic refusal wording on political questions.

The old `abstention_or_format` label conflated different outcomes. The remediation
now distinguishes a numerical probability, explicit `ABSTAIN`, a recognized
free-text refusal, malformed output, tool use, and infrastructure failure. The
original error label is retained as provenance. Original responses, numerical
values, timestamps, and scores were preserved.

Completed refusals are not retried until a number appears, and missing values
are not filled with 0, 0.5, or an invented estimate. Infrastructure failures can
be resumed with prior attempts retained. This fixes classification of the known
failures; it does not turn a refusal into a usable probability or prove that
every possible future edge case is handled.

Source: [readout audit](../data/benchmark-runs/native-readout-audit.json).

## 4. New prompt, diagnostic, and interrupted full rerun

A separate treatment, `descriptive-v2`, was introduced. It changed the wording
and explicitly allowed abstention. It was not the original prompt with only a
software fix.

Fixed instruction:

```text
You are participating in a descriptive forecasting benchmark. This task asks for event likelihoods, not advocacy, persuasion, recommendations, or endorsement. Follow your applicable constraints. You may abstain when you cannot provide an estimate. Use only your existing knowledge. Do not use tools, browse, read files, consult memory, or delegate. Return exactly one final readout: a decimal number between 0 and 1, or ABSTAIN.
```

User prompt:

```text
Estimate the probability that the proposition below is true.
This is a descriptive estimate, not a recommendation or endorsement.
Return exactly one decimal number between 0 and 1, or ABSTAIN if you cannot provide an estimate.
Do not add commentary.

Proposition:
{proposition}
```

First, a fixed diagnostic selected three previously refused questions and three
nonpolitical controls across all nine settings: **54 observations, comprising
39 probabilities and 15 `ABSTAIN` responses**. It ran approximately
03:16:33–03:18:08 UTC on September 13. Diagnostic results were excluded from
ranking scores but included in usage. There was no subsequent iterative prompt
search or question-specific wording adjustment.
The diagnostic launcher used two workers per setting. The full and scoped v2
launchers used four workers per setting, with a 600-second timeout per invocation
throughout v2.

Then `native-20260913-neutral` began a full 591-question rerun across all nine
settings at approximately 03:18:24 UTC. Its last completed receipt was at
03:20:48; the pause was recorded at 03:20:49 and termination at 03:21:38 during
the scope review. It returned **423 receipts: 384 probabilities and 39
abstentions**. Another **36 launches** were interrupted without returned usage.
This full rerun never completed.

The diagnostic and interrupted campaign together account for **477 receipts,
513 launches, and 3,471,259 recorded native tokens**. Their aggregate accounting
is preserved. The interrupted campaign is not added as nine completed runs to
public score history. Only compatible in-scope observations were subsequently
reused, as described below.

Sources: [interrupted-campaign usage](../data/benchmark-runs/subscription-usage-native-20260913-neutral.json),
private `runs/native-20260913-neutral-pilot/{selection.json,diagnostic-result.json}`,
and private `runs/native-20260913-neutral/{campaign-pause.json,interrupted-invocations.json}`.
The pilot's campaign metadata retains the 591-question source-cohort size;
`selection.json` identifies the six questions actually tested.

## 5. How the subset was selected, and why that was a problem

A source-only review, as recorded in the scope manifest, considered the original
titles and full resolution criteria without using model answers to select
questions. It selected or
excluded **whole groups**, then deduplicated the questions in retained groups.
It did not simply remove the 50 questions that had elicited refusals.
The manifest was frozen after the interrupted full-v2 campaign, so some v2
responses already existed. The recorded response-independent review should not
be described as a scope registered before any v2 inference had occurred.

The actual topic rule was broader than necessary. It excluded political-policy,
legislative, electoral, diplomatic, regulatory-approval, and governmental-structure
outcomes, including such conditions inside a group. This swept in general policy
and non-U.S. political questions as well. Government agencies acting only as
technical mission operators or data verifiers could remain.

A separate semantic review excluded clear differences in dates, quantities,
quantifiers, event wording, or conditions that were not true complements.

| Recorded disposition | Groups | Unique questions omitted after deduplication |
| --- | ---: | ---: |
| Broad politics/policy topic exclusion | 230 | 467 |
| Additional semantic defects | 7 | 12 |
| One mixed topic-boundary/semantic case | 1 | 4 |
| **Excluded total** | **238** | **483** |
| **Retained** | **62** | **108** |

Within the 230 topic exclusions, the recorded reason codes were political/policy
content (170 groups), regulatory decisions (48), and government infrastructure
or institutions (12). **The 467 is not a count of 467 individually political
questions:** whole-group exclusion can also remove their nonpolitical companions.

This was a choice made by the research agents, not 483 observed provider
refusals. The earlier blanket explanation that political forecasts could not
be generated was too broad. It did not justify dropping all general policy and
non-U.S. politics. The exclusion rule and its effect should have been explained
before presenting the subset as the corrected rerun.

The source questions were not repaired or rewritten. Excluding questionable
groups does not fulfill the request to fix their semantic defects. The review
also documented residual wording caveats; retained membership is not a formal
proof of every logical equivalence.

The retained set and exclusion reasons were frozen before scoped inference:

```text
scope_id: nonpolitical-original300-reviewed-v1
SHA-256: e427faedeec9a8f19dc371ff48d9fc7248654c2ff937f37f2f33d6433c50464c
```

Source: [frozen scope manifest](../data/benchmark-runs/scope-e427faedeec9.json).
Its pre-execution review-status label is preserved with its original bytes;
execution completion is recorded separately in the campaign artifacts.

## 6. What the completed 972-readout rerun actually did

`native-20260913-scoped` used the same `descriptive-v2` prompt for every setting
on the retained **108 questions / 62 groups**. It reused **63 already completed
compatible observations** from the interrupted full-v2 run. The importer checked
question text, model, reasoning, protocol, instruction/prompt hashes, and source
receipt provenance. Reused observations retained their original timestamps and
were not charged again in the usage totals.

The remaining **909 observations** were obtained through fresh native requests.
All 972 final observations contained numerical probabilities, with no recorded
abstentions, malformed answers, or tool-use failures. Every new setting could
therefore score all 62 retained groups.
Fresh scoped observations ran from approximately 03:29:15 to 03:48:45 UTC;
the campaign-completion record was written at 03:50:12.

GLM stalled during startup in four launches from an empty directory inside the
Git worktree. Those sessions had no observed output; their backend consumption
is unknown. The exact internal deadlock was not proven. The runtime was changed
to use an empty directory outside Git, with indexing, file watching, and snapshots
disabled. Model, reasoning, prompt, and scope were unchanged. GLM's final set
contains three observations from the original runtime profile and 105 from the
corrected profile. Interrupted launches were retained separately.

The final audit checked all 972 observations against native event streams and
applicable final files, model/effort/protocol identity, unique sessions, absence
of tool calls, and the 63 immutable import sources. “Valid” here means a valid
numerical readout with checked provenance. It does not mean the forecasts are
accurate, the sample is representative, or the full original task is complete.

Private evidence:
`runs/native-20260913-scoped/{campaign-completion.json,final-native-audit.json,glm-runtime-recovery.json}`.
Public export source: the nine `native-20260913-scoped-*.jsonl` result and receipt
files in [benchmark data](../data/benchmark-runs/).

## 7. Scoring, database, and prepared website

The scorer uses `common.violation`: algebraic consistency checks such as
`abs(P + not_P - 1)` for negation and bounds on conjunctions. A group is scored
only if every required probability exists. Means average the scorable groups.
These measure internal coherence, not forecast accuracy, calibration, or the
paper's arbitrage metric. Missing checks are not assigned perfect scores.

The local database contains **64 runs**: 46 historical runs, nine original
native runs, and nine scoped runs. The original 55 runs' score values and hashes
were retained. Model/reasoning identities remain distinct, including all six
Luna efforts. The interrupted full-v2 campaign is absent from scored history.

For the prepared comparison across **24 settings**, the renderer intersects the
scorable group IDs. An older partial `muse-glimmer-30b` run covers only 20 of the
62 retained groups. As a result:

- The prepared all-setting ranking uses the same **20 groups** for every row.
- Separate details give per-setting coverage and the means over the retained
  **62 groups** where available.
- The retained 62-group scope has no CondCond examples. The 20-group intersection
  also has no ExpEvidence examples.
- Rankings can change substantially between the 20-group and 62-group samples.
  Neither is the full 300-group ranking Kelvin requested.
- Historical and new runs also differ in date, prompt protocol, and native client
  scaffolding. Matching question IDs does not remove those differences.

The original homepage layout was restored locally: combined ranking chart,
all-model heatmap, elicitation readouts, worst violations, and runs. Supplementary
ranking, coverage, usage tables, and split charts were moved to `details.html`.
The prepared pages contain all 24 settings and all six Luna efforts. Native
logprob measurements are unavailable and are not invented. Existing Open Agency
and plan pages are preserved.

The last implementation verification passed **54 tests** and desktop/mobile
browser checks. These establish the tested engineering behavior and presentation;
they do not validate the decision to exclude 81.7% of questions. No inference or
test suite for the benchmark was rerun merely to write this report.

At this report's state check, remote `main` was still
`025473f876b0721f9f21bc1a66f77f33602b1019`. Local layout commit `25d81e9` exists;
the later remediation changes are uncommitted. The new data and pages have not
been pushed or deployed. Publication was initially held for a homepage dataset
choice; following the exclusion review, restoring appropriate coverage is now
unfinished substantive work, not merely a presentation choice.

Sources: [exporter](../scripts/export_benchmark.py),
[dashboard generator](../scripts/build_dash.py),
[prepared homepage](../dash/index.html), and [prepared details](../dash/details.html).

## 8. Subscription usage and its limits

These totals include recorded setup and failed-attempt consumption where the
native client returned usage. Imported observations are counted only in their
source campaign's new usage.
They cover benchmark-client calls, not a separate accounting of the supervising
research/development agents' conversations or other fleet activity.

| Phase | Returned receipts | CLI launches | Recorded native tokens |
| --- | ---: | ---: | ---: |
| Original native campaign, including setup/retries | 5,382 | 5,382 | 37,860,370 |
| Diagnostic plus interrupted full-v2 rerun | 477 | 513 | 3,471,259 |
| Completed scoped rerun, fresh calls only | 909 | 913 | 6,604,478 |
| **Cumulative** | **6,768** | **6,808** | **47,936,107** |

There are **40 interrupted launches without receipts**, plus **30 older receipts
without native usage telemetry**. Thus consumption is unknown for 70 launches.
Unknown consumption is not zero, so recorded tokens are not a complete measure
of every launched process's backend consumption. A CLI launch also does not prove
that a backend inference request occurred during a startup failure.

Fresh usage in the scoped campaign, excluding the 63 imports:

| Setting | Fresh receipts | Reused observations | Recorded native tokens |
| --- | ---: | ---: | ---: |
| GLM-5.3, native default | 107 | 1 | 149,724 |
| Grok-4.6, low | 102 | 6 | 1,505,634 |
| Astra, low | 101 | 7 | 932,413 |
| Luna, none | 93 | 15 | 592,434 |
| Luna, low | 99 | 9 | 637,563 |
| Luna, medium | 101 | 7 | 656,934 |
| Luna, high | 101 | 7 | 664,462 |
| Luna, xhigh | 102 | 6 | 688,213 |
| Luna, max | 103 | 5 | 777,101 |
| **Total** | **909** | **63** | **6,604,478** |

The four extra interrupted scoped launches belong to GLM. All 700 fresh
Astra/Luna receipt session IDs matched the existing subscription proxy log.
Token normalization accounts for providers' different handling of cached input
and reasoning output without adding the same tokens twice.

Native quota snapshots were taken before and after the scoped interval,
approximately **03:29–03:50 UTC**:

| Subscription/window | Before | After | Displayed change |
| --- | ---: | ---: | ---: |
| Grok weekly | 2% | 2% | 0 percentage points |
| GLM five-hour token window | 4% | 4% | 0 percentage points |
| GLM weekly token window | 1% | 1% | 0 percentage points |
| OpenAI subscription 1, weekly | 11% | 12% | 1 percentage point |
| OpenAI subscription 2, weekly | 26% | 27% | 1 percentage point |
| OpenAI subscription 3, weekly | 48% | 50% | 2 percentage points |
| OpenAI subscription 4, weekly | 1% | 1% | 0 percentage points |
| OpenAI subscription 5, weekly | 4% | 4% | 0 percentage points |
| OpenAI subscription 6, weekly | 29% | 33% | 4 percentage points |
| OpenAI subscription 7, weekly | 22% | 22% | 0 percentage points |

These are rounded readings on shared accounts. Other fleet sessions can affect
them. A zero displayed change does not mean zero use, and percentages from
different accounts must not be summed. GLM's separate `TIME_LIMIT` window reset
identity changed, so no delta was attributed to that window. Earlier campaigns'
quota readings remain in their archived reports.

**We measured attributable receipts and returned tokens, plus shared-account
quota movement. We did not measure an exact experiment-only fraction of each
subscription limit.** The providers do not expose the conversion needed to infer
that fraction. Dollar estimates in native receipts were not treated as extra
charges on top of the subscriptions.

Source: [current usage report](../data/benchmark-runs/subscription-usage.json)
and the two archived reports linked above. Public exports omit authentication
data and account/session identifiers; private raw receipts retain audit evidence.

## 9. What is still missing

| Requirement | Current state | Remaining work |
| --- | --- | --- |
| Explain and correctly classify original failures | Known 157 cases audited; all are refusals | Preserve missingness and coverage; no parser fix can recover a number from those texts |
| Corrected comparison covering the original questions as fully as possible | Original full-v1 data exists; full-v2 rerun stopped; later rerun covers only 108/591 | Review the overbroad exclusions, establish which questions can be restored, then complete an appropriately defined campaign |
| Fix source semantic edge cases | Seven groups plus one mixed case were excluded | Review and repair or explicitly document them; version changed source text separately |
| Fair full-table comparison | Prepared all-setting intersection contains only 20/300 groups | Reassess comparison coverage and historical missingness; do not pass a tiny intersection off as the full benchmark |
| All requested settings in the database | Nine settings, including six Luna efforts, exist locally | Publish the validated final database after scope work is resolved |
| Original homepage with new models; extras on another page | Layout and supplementary page prepared locally | Align the final dataset and labels with the actual completed scope, then deploy and verify live |
| Subscription usage measurement | Receipts, tokens, quota snapshots, and telemetry gaps documented | Exact marginal quota fraction remains unavailable from current provider telemetry |
| Repo/site delivery and task closure | Remediation not pushed or deployed | Commit, push, deploy, verify public artifacts, and close the tracking task only after the substantive work is done |

The next substantive step is the **exclusion audit**, not another claim that the
972-readout subset completed the original request. It must distinguish actual
scope restrictions from the overly broad topic filter and from genuine source
defects, without choosing questions based on favorable model answers. Its
recoverable-question count and any expanded rerun remain unknown at this point.

## Evidence and rebuild notes

The linked JSONL, manifests, audit, and usage reports are the reproducible public
source artifacts in the current worktree. Private `runs/` evidence exists on this
machine and is not being published wholesale. This report summarizes existing
evidence; it does not include new political forecasts or initiate new requests.

The existing exported data and prepared pages can be rebuilt without inference:

```sh
python3 scripts/export_benchmark.py
python3 scripts/build_dash.py
```

The source snapshot, new scope membership, changed prompts, reused observations,
failed attempts, and historical results must remain distinguishable in any
subsequent work. The original frozen subset is evidence of what was run; correcting
the exclusion rule requires a new version rather than silently rewriting it.

## Execution update after the initial report

**Historical update, superseded at 04:27 UTC:** the re-admission decision below
incorrectly included numerical U.S. policy/legislation forecasts. The campaign
was subsequently stopped. This section records the launch, not a currently
validated scope or a completed result.

At approximately **04:20 UTC on September 13**, the expanded campaign
`native-20260913-scoped-expanded` started after a second source review and an
independent validation. This update supersedes the initial report's statement
that re-admission had not yet been reviewed. The initial completed campaigns and
their historical counts above are unchanged.

The frozen expanded scope retains **242 of 300 original groups and 445 of 591
unique questions (75.3%)**, with all ten check types represented. It preserves
all 62 groups / 108 questions in the earlier subset. There are **58 excluded
groups: 34 containing held election-related numerical targets, and 24 with
documented semantic defects**. Whole-group filtering omits 146 unique questions.

The review restores general policy, regulatory, non-U.S. political, and global
questions that the blanket filter had unnecessarily removed. It also distinguishes
historical events inside the source's October 2024-start windows from unresolved
future elections. A dedicated review of all 30 conditional-chain groups identified
21 clear event/date/quantity mismatches; four overlap topic exclusions, and two
were already in the earlier semantic exclusions. Those findings, plus the other
known source defects, account for the separate semantic exclusions. Four
conditional-chain wording caveats remain documented without rewriting the source.

The new campaign's target is **4,005 observations = 445 questions × nine settings**.
Before fresh scheduling, the runner imported **1,151 compatible completed
observations**: 242 directly from the interrupted full-v2 campaign and 909 fresh
observations from the completed small-scope campaign. This preserves all 972
small-scope observations while counting their 63 earlier imports only once.
It leaves **2,854 observations to collect through new calls**, before any
infrastructure retries. A completed imported abstention remains an abstention;
it is not rerun until a number appears.

The model identities, reasoning settings, fixed descriptive-v2 prompt, and
corrected GLM runtime are unchanged. New quota baselines were captured at
04:18:52–56 UTC. The expanded run was still running when this update was written;
its eventual results, additional usage, and public deployment remain unfinished.

Private source-review evidence:

```text
runs/scope-readmission-review.json
runs/condcond-semantics-readmission.json
runs/scope-readmission-historical-evidence.json
runs/scope-expanded-reviewed-v2.json
```

The expanded source-scope SHA-256 is
`1b3d9122cfa782d535c76bdc94131ed68c78b2fbafeebbbb332ac329d4b22aff`.
The earlier frozen scope remains unchanged. The launch and readout receipts are
under `runs/native-20260913-scoped-expanded/`.

## Stop and scope correction at 04:27 UTC

The expanded campaign was paused at **04:26:43 UTC** and its stopped process
tree terminated at **04:27:38 UTC**, after identifying U.S. policy probability
questions in the selected scope. Its frozen manifest and receipt files were
preserved. General non-U.S. politics was previously overexcluded, but this does
not make numerical U.S. policy/legislation forecasts eligible.

At the stop, saved receipts contained **2,148 completed observations: 2,137
numerical responses and 11 abstentions**. These comprise **1,151 imports and
997 fresh responses**, leaving **1,857 of the 4,005 planned observations without
a completed receipt**. There were **36 active native launches** at interruption;
their backend consumption is not established by the absent receipts.

The final eligible scope, expanded usage reconciliation, and corrected public
delivery remain unfinished. The [focused report](political-constraints-and-missing-work-20260913.md)
distinguishes political constraints from provider refusals, unnecessary agent
exclusions, semantic defects, and engineering work still outstanding.


## Final completion

The final reviewed campaign completed all **274 questions × nine settings = 2,466
observations**, comprising **2,450 numerical answers and 16 explicit abstentions**.
There are **912 fresh receipts** and **1,554 imports**: 138 from the interrupted
full-v2 campaign, 873 originally generated in the small campaign, and 543
originally generated in the stopped expansion. Imports add no new request usage.

All nine native workers exited successfully. The audit checked the exact source
question, fixed prompt, model/reasoning identity, native final messages, completion
and usage events, immutable import hashes, and recomputed group scores. No format
failures or unresolved transport failures remain. One imported Luna-medium
response had a stream-reconnection event before its successful final answer;
that recovery is documented rather than hidden. No completed abstention or
refusal was retried to obtain a number.

The **original benchmark stays on the homepage** and its full table is on
[original details](https://consistency.futarchy.ai/details.html). It compares
23 ranked settings on 264 shared original groups, retaining all 24 settings with
Muse visible but unranked. The **corrected subset is separate**, on
[corrected comparison](https://consistency.futarchy.ai/corrected.html) and
[corrected details](https://consistency.futarchy.ai/corrected-details.html).
Its reviewed target is 157 groups; abstentions reduce the common scored set to
148 groups across the same 23 participating settings. Every Luna effort remains
visible: none, low, medium, high, xhigh, and max.

The final database retains the 55 previously published runs and adds nine final
runs. Original score bytes and timestamps are preserved. The small campaign's
nine unpublished per-question exports are excluded because four of its questions
were subsequently held. Its eligible observations survive via audited imports;
its aggregate usage and this chronology remain available. The interrupted
campaigns contribute aggregate accounting, without new out-of-scope numerical
exports.

**What remains absent:** a corrected rerun of the full 591-question benchmark.
The final scope omits 317 questions: 253 held targets, 20 additional eligible
companion questions lost through whole-group topic filtering, and 44 additional
eligible questions lost through semantic filtering. The 16 abstentions are
completed observations with missing probabilities. They are a separate category
from those scope omissions. Muse's archived coverage remains incomplete. None
of these gaps is presented as a complete full-dataset corrected result.

See the [final scope](../data/benchmark-runs/scope-bb38d378d4a1.json),
[sanitized receipt audit](../data/benchmark-runs/final-native-audit-20260913.json),
[final native-method and usage guide](native-evaluation.md), and
[subscription usage report](../data/benchmark-runs/subscription-usage.json).
