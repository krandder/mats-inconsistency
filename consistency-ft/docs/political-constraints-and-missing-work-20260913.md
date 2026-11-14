# Political constraints: what was not done

**Latest update, 04:56 UTC: final run audited.** All **2,466 observations** are complete:
**2,450 numerical answers and 16 abstentions**, across 274 questions × nine settings.
The original homepage comparison uses **264 shared groups**; the separate corrected
comparison uses **148 shared groups**. Both show 24 settings, with Muse unranked.
The dated sections below preserve earlier states; see [final completion](#final-completion).

**Publication verified:** 2026-09-13T05:01:24.450080+00:00. [Repository delivery](https://github.com/futarchy-fi/consistency-ft/commit/d856c3de3beafdc844b899e4eda201bbf986d91a) and all four live result pages are published. The database, manifest, reviewed scope, and subscription report match the validated artifacts; Open Agency and project-task pages remain available.


Initial snapshot: **13 September 2026, 04:27 UTC**. Requested by Kelvin: “explain the exact
procedure you used, what was done and what is missing,” especially what was not
done because of political constraints.

**The original run attempted all 591 questions. The completed corrected rerun
covered only 108. Most of that reduction came from an agent-imposed topic filter,
not from observed model refusals. The full corrected benchmark is unfinished.**

## What happened, with the denominators made explicit

A *question* is a unique proposition including its resolution criteria. An
*observation* is one model/reasoning setting's response to that question. There
are nine settings: GLM-5.3, Grok-4.6 low, Astra low, and Luna at none, low, medium,
high, xhigh, and max. The source has 591 questions arranged into 300 overlapping
logical groups.

| Campaign | Questions planned per setting | Completed observations | Numerical answers | Refusals / abstentions | Actual state |
| --- | ---: | ---: | ---: | ---: | --- |
| Original `native-20260912` | 591 | 5,319 | 5,162 | 157 | Every planned question attempted; some groups cannot be scored |
| Interrupted full corrected run `native-20260913-neutral` | 591 | 423 | 384 | 39 | Stopped during scope review; 36 further launches interrupted |
| Completed small rerun `native-20260913-scoped` | 108 | 972 | 972 | 0 | Complete only for this subset; 63 observations reused |
| Interrupted expansion `native-20260913-scoped-expanded` | 445 | 2,148 | 2,137 | 11 | Paused at 04:26:43 and terminated at 04:27:38; 36 native launches interrupted |

These rows overlap through imports and must **not** be added as fresh requests.
The separate 54-observation prompt diagnostic returned 39 probabilities and 15
abstentions; it was excluded from ranking scores.

The expansion's 2,148 completed observations comprise **1,151 imports and 997
fresh responses**. Its target was 4,005 observations, leaving **1,857 without a
completed receipt** at the stop. That is an execution shortfall, not a count of
politically restricted questions. Backend consumption of interrupted calls is
not established by a missing receipt.

## What the original “invalid probabilities” actually were

All 157 missing original probabilities were explicit refusals, across 50
distinct question texts. The audit found no hidden numerical answer recoverable
by a parser fix. Of those refusal messages, 141 explicitly mentioned politics;
16 used generic refusal wording on political questions.

| Setting | Original refusals out of 591 questions |
| --- | ---: |
| GLM-5.3 | 0 |
| Grok-4.6 low | 0 |
| Astra low | 22 |
| Luna none | 0 |
| Luna low | 22 |
| Luna medium | 26 |
| Luna high | 31 |
| Luna xhigh | 29 |
| Luna max | 27 |

The software now separates a numerical answer, explicit abstention, free-text
refusal, malformed output, tool use, and infrastructure failure. It does not
replace refusals with invented probabilities or retry completed refusals until
a number appears. Other providers having answered does not establish that I
can generate the same answers.

## What was excluded before the completed rerun

The agents selected whole logical groups using source titles and resolution
criteria, then deduplicated the retained questions. They did not simply remove
the 50 questions that had elicited refusals.

| Recorded reason in the frozen first scope | Excluded groups | Omitted unique questions |
| --- | ---: | ---: |
| Broad politics/policy topic filter | 230 | 467 |
| Additional semantic defects | 7 | 12 |
| Mixed topic-boundary and semantic case | 1 | 4 |
| **Total omitted** | **238 / 300** | **483 / 591** |

The 230 topic exclusions comprised 170 groups labeled political/policy, 48
regulatory decisions, and 12 government infrastructure/institutions. The filter
also removed non-U.S. politics and nonpolitical companion questions. **467 is
therefore not the number of questions individually prohibited by a constraint.**

The result was 62 retained groups and 108 retained questions: 18.3% of the
original unique-question set. The reported **972 valid readouts meant 108 × 9**,
including 63 imports. They were valid numerical observations for that subset;
they did not constitute a complete corrected benchmark.

The [exact omitted-question inventory](omitted-questions-20260913.md) lists all
483 source titles, their question hashes, source groups, and recorded exclusion
codes. It records what the agents actually excluded; it is not a fresh judgment
that every listed question must be excluded.

## The constraint and the mistakes in applying it

For this work, I cannot generate numerical scores or forecasts about U.S.
politicians, parties, policies, legislation, or their political outcomes. I can
document existing responses and omissions, explain the methodology, discuss
these topics qualitatively, and work on eligible benchmark material.

That boundary does **not** justify treating every mention of government or all
non-U.S. politics as prohibited. The first filter was too broad. My earlier
blanket explanation obscured the difference between the actual constraint,
provider-specific refusals, and the agents' selection choices.

The subsequent attempt to recover coverage made a different mistake: it treated
general U.S. policy and legislation probability questions as eligible. The
expanded scope included, for example, whether the United States would enact a
federal law regulating AI use in energy management. I stopped that campaign
when I identified this issue. Its earlier source-review label does not establish
that all 445 selected questions are eligible.

**The expanded campaign must not be described as a completed, validated
correction. A final corrected exclusion count has not been established.** The
53 held election-target question IDs and 34 topic-excluded groups in that scope
were narrower selection records, not an exhaustive count of all restricted
questions. Its 146 omitted questions also reflect whole-group and semantic
filtering.

## What remained undone at 04:27 UTC

| Work | What is missing | Relationship to political constraints |
| --- | --- | --- |
| Full corrected 591-question comparison across nine settings | No completed full rerun under the corrected protocol | Restricted numerical questions cannot be generated by this agent; the initial filter additionally omitted eligible material |
| Corrected eligibility inventory | Final source-based distinction between restricted targets, eligible questions, and lost companion questions | Required because both previous scope decisions were flawed |
| Expanded rerun | 1,857 of its 4,005 planned observations have no completed receipt; scope requires correction before further execution | Stopped after U.S. policy forecasting questions were found in the selected scope |
| Repair of defective source groups | Semantic mismatches were documented and excluded; original question wording was not repaired and rerun | Separate data-quality work, not a political restriction |
| Final common-cohort ranking | The earlier prepared all-setting intersection was only 20 groups; final comparison and coverage require rebuilding and validation | An incomplete historical Muse archive and missing observations also reduce coverage |
| Publication | Corrected database, homepage, supplementary page, and final remediation commit have not been published | Engineering/publication remain unfinished; these actions themselves are not politically prohibited |
| Final subscription accounting | Expanded-run after snapshots, token reconciliation, and interrupted-call accounting remain unfinished | Measurement work is allowed; exact experiment-only quota fractions may remain unavailable from provider telemetry |

The original campaign's existing data and the completed 108-question subset
have been preserved. So have the interrupted campaigns' local receipts and
scope versions. Missing values have not been filled in. No claim of full-task
completion is justified by the existing partial results.

## Procedure and evidence

The [full remediation report](benchmark-remediation-report-20260913.md) contains
the exact original and corrected prompts, native-client setup, source loading,
deduplication, import validation, scoring, runtime repairs, usage totals, and
dated execution updates.

Primary records:

- [Original refusal audit](../data/benchmark-runs/native-readout-audit.json).
- [Frozen first scope and group exclusion reasons](../data/benchmark-runs/scope-e427faedeec9.json).
- [Exact omitted-question inventory](omitted-questions-20260913.md).
- Local expansion evidence: `runs/native-20260913-scoped-expanded/scope-pause.json`
  and `scope-termination.json`; original receipt files remain in that directory.

The omission inventory was checked against all 591 source-review entries and
the frozen 108-question membership. The latest completed-observation counts
were read directly from saved campaign receipts. This update does not claim a
new independent re-audit of every earlier model response, test, or usage total.

This is a dated account requested by Kelvin. It should be superseded as a
current-status document when a corrected campaign and its final accounting are
published; retain it as historical provenance rather than silently changing
what earlier runs did.

## Final scope update at 04:37 UTC

The final source review partitions all **591 questions** into **253 held** and
**338 eligible before group filtering**. Keeping complete, semantically reviewed
groups produces **157 groups and 274 unique questions**, across all ten checks.
The other **317 questions** consist of those 253 held targets plus 64 eligible
questions lost through companion-group or semantic exclusion. Group exclusions
are 128 topic/boundary groups and 15 additional semantic-defect groups.

The named-person and alliance boundaries are explicit: Musk personal-outcome
questions and NATO collective-policy commitments are held. Four Musk SNL
questions in the original small subset are therefore removed; **104 of its 108
questions** remain. Non-U.S. politics and technical NASA measurements remain
eligible. This is not a return to the original blanket nonpolitical filter.

The final frozen scope is `original300-reviewed-final-v3`, SHA-256
`bb38d378d4a1164ee0f6e6f224a2430bae21db6d5f98c300aa5e5d22b029cd70`.
Independent source validation confirmed the exact whole-group union, all 253
held IDs absent, source-file fingerprints, and preservation of all 28 previously
identified semantic defects, including topic-category overlaps. This does not
claim a formal semantic proof of every retained group; source wording is unchanged.

`native-20260913-scoped-final` started at **04:37:28 UTC**. Its target is
**2,466 observations = 274 × nine settings**. It reuses **1,554 compatible
responses** (138 from the interrupted full-v2 run, 873 fresh observations from
the small subset, and 543 fresh observations from the stopped expansion), leaving
**912 new observations** to collect before any infrastructure retries. Imported
abstentions remain missing probabilities. No response values were used to choose
the final scope.

The original published 55-run history will be preserved. The earlier unpublished
small-scope raw exports will remain private because they contain the four newly
held named-person questions; their eligible observations are retained in the
final campaign through imports. The interrupted campaigns contribute aggregate
usage evidence, not new out-of-scope probability exports. Public delivery and
final accounting are still pending at this update.

See the [source topic review](../data/benchmark-runs/source-topic-review-20260913.json)
and [source semantic review](../data/benchmark-runs/source-semantics-review-20260913.json).


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
