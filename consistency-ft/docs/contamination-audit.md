# Factual-recall contamination audit

The native campaign and response review are complete for **nine settings × 129
questions = 1,161 observations**. See the [full campaign](native-probes-20260913.md),
[source audit](probe-source-review-20260913.md) and
[agent response adjudication](native-probe-adjudication-20260913.md).
Historical-provider measurements remain unfinished and **no model cutoff has
been verified**. Bead `hub-t8k9c.9` remains open. The old runner's
`CONTAMINATED` / `cutoff plausible` verdicts are no longer produced.

An answer-key match is evidence about a response. It does not by itself locate
the model's training cutoff: guesses, earlier announcements, inference,
post-training data, retrieval, and answer-key errors can produce matches.
An `UNKNOWN` response, mismatch, refusal, or failed request cannot establish
that the model lacks knowledge. A release date is not a verified knowledge bound.

## What is actually in the repository

The source inspected on 13 September 2026 contains **129 probes**, not the older
90-probe inventory described in historical notes. The extra rows are preserved.
There are no literal duplicate question strings; related questions about the
same event can still be correlated and have not been deduplicated statistically.

| Source property | Observed count |
| --- | ---: |
| Full catalog | 129 |
| Month-precision event dates | 90 |
| Day-precision event dates | 39 |
| Source recorded as `authored-from-model-knowledge` | 60 |
| Source URL present | 69 |
| Recorded source publication date | 0 |
| Recorded source retrieval timestamp | 0 |
| Schema/date checks passed as of 2026-09-13 | 129 |
| Cutoff-catalog model entries | 19 |

The event-date range recorded by the catalog is July 2024 through August 2026.
Passing a schema/date check does not validate the question, reference answer,
URL, event date, or earliest public availability of the answer. The table above
describes the original catalog before independent source review. The later
[source audit](probe-source-review-20260913.md) retained evidence for all 129
core answers while preserving the original catalog and its unresolved flags.
Some cutoff entries describe primary sources, others secondary or conflicting
reports; those recorded descriptions are preserved as claims, not upgraded to
verified facts. The native Astra, Grok, GLM-5.3 and Luna identities have no matching
entries in the current cutoff catalog and must not inherit sibling-model dates.

Source fingerprints:

- `data/cutoff_probes.jsonl`: `926436aee906283794c0f97bfd8e8432f97e9fdcd9bb6f7d4ae46fdb3ef6c40b`
- `data/cutoffs.json`: `07b5fa19ae3a0651b4ba6c695821ea6391acda27dc35f23d20bdad91eb8262f4`

## Preparation and offline reporting

The interface uses explicit commands. Preparation freezes the complete catalog,
source fingerprints, model/provider/reasoning identity, cutoff-claim snapshot,
as-of date and prompt. It writes every source row to `manifest.json`, including
rows needing review. `requests.jsonl` contains only eligible dated questions and
the prompt metadata; reference answers and aliases are not supplied as context.
There is no implicit `--limit` or smaller-cohort completion claim.

```bash
python3 harness/run_probes.py prepare \
  --run-dir runs/cutoff-gpt-oss-20b \
  --model accounts/fireworks/models/gpt-oss-20b \
  --provider fireworks --reasoning low --as-of 2026-09-13

python3 harness/run_probes.py report --run-dir runs/cutoff-gpt-oss-20b
```

Use one directory for each model/reasoning treatment. `--cutoff-key` can specify
an exact catalog key where necessary; no sibling-model lookup is performed.
Changing the source, aliases, date, model, reasoning or protocol requires a new
campaign directory. Existing manifests are fingerprint checked on resume.

`report.json` includes every probe, unresolved status, date relationship, source
review fields, quarter counts, coverage, imported usage and the cutoff claim
inventory. Its `verified_cutoff` is always `null`. `report.md` is a short summary.
The source catalog and cutoff inventory remain unchanged by this runner.

## Live Fireworks collection: retained, not invoked here

Collection without `--execute` is a plan and does not initialize credentials or
make a request:

```bash
python3 harness/run_probes.py collect-fireworks --run-dir runs/cutoff-gpt-oss-20b
```

After provider access and execution authorization are established, collection
can be started explicitly. Prepare the campaign using the preparation day's UTC
date. Collection can resume on later days without relabeling that frozen scope;
the report records the latest actual observation separately. This example is a
command description, not an execution record:

```bash
python3 harness/run_probes.py collect-fireworks \
  --run-dir runs/cutoff-gpt-oss-20b \
  --execute --max-calls 129 --max-tokens 2048
```

`--max-calls` is a **cumulative launch allowance** across resumes, including
failures and unresolved launches. It does not shrink the target catalog.
`--max-tokens` limits each completion and is frozen with the collection's
reasoning treatment. These are call/token limits, not a guaranteed dollar cap;
pricing and provider-side accounting remain separate. The existing Fireworks
credential lookup is used only after explicit execution. No billing or account
settings are changed.

Each launch is reserved durably before one HTTP attempt. There is no hidden
retry loop and no automatic switch from `none` to `low` reasoning. A provider
failure is saved and stops collection; `--retry-errors` is necessary to retry
a known infrastructure failure. Completed answers, `UNKNOWN`, refusals,
unmatched responses and truncated answers are not retried for a different value.

The complete provider response and receipt are saved atomically per attempt,
then imported into the ledger. If interruption occurs between those writes,
resume imports the saved receipt without a new call. A launch with no saved
outcome blocks blind replay: its backend consumption is unknown and must be
reconciled. Ordinary failed attempts retain all available usage; absent usage
is `null`, not zero. Returned model names, finish reasons, and tool calls are
retained. Unexpected model identity, truncation, and tool use are reported
separately and do not become recall evidence.

## Import contract and provenance

Existing provider observations can be imported without making a request:

```bash
python3 harness/run_probes.py import \
  --run-dir runs/cutoff-gpt-oss-20b --receipts /path/to/provider-receipts.jsonl
```

Each input JSONL object requires:

- `probe_id`, `model`, `provider`, `reasoning`, `protocol_id` and `prompt_sha256`
  matching the prepared request.
- Timezone-aware `started_at` and `finished_at` timestamps.
- `status`: `completed`, `refusal`, or `infrastructure_error`; failures also
  require `error`. `response` retains the full text, including an empty failure.
- Explicit boolean `tools_used` and `usage`: nonnegative integer counters, or
  `null` when unavailable. Additional provider metadata may be retained.
- `provenance` with `transport`, `adapter`, and a native `session_id` or provider
  `request_id`. Reusing a session/request identifier across different questions
  is rejected. A failed request without a provider ID can use a clearly labeled
  local launch ID, as the Fireworks adapter does.

The ledger preserves the original record, its SHA-256 and the import-file
SHA-256. Imports are atomic, repeatable and do not overwrite completed results.
These checks establish identity and local integrity; they do not authenticate
untrusted provider assertions. Raw receipts and session/request identifiers
belong in private `runs/` storage. Summary reports omit those identifiers.

The previous v1 output lacks enough model, prompt, attempt and provider
provenance to import directly. It must not be silently relabeled as a v2 run.

## Grading and temporal interpretation

Only the whole canonical answer or an explicitly listed alias can match.
Normalization handles case, Unicode form, whitespace, enclosing quotes and a
single final period. It does not guess surnames, match four-letter components,
or extract an answer from longer prose. For example, a reference `North Lab`
does not match `not North Lab`, `North`, or `North Lab or South Lab`.

`UNKNOWN` is a distinct completed response. Empty responses, infrastructure
failures, provider refusals, tool use, truncation, unexpected provider model
identity and unmatched text have separate statuses. Unmatched text is
`manual_review`, not automatically an incorrect answer or proof of ignorance.
Literal reference-answer leakage into a question is flagged for source review.

Dates are intervals: a month spans the whole month and a year spans the whole
year. A day-precision event after a day-precision cutoff in the same month is
recognized. Overlapping dates remain uncertain. Future events and event-date
ranges overlapping the preparation day remain in coverage but are not queued.
The event must also predate the actual observation day to count as a candidate
post-cutoff match. An event date does not establish when its answer first became
publicly available, so these candidates require manual source review.

## Validation and work still required

Run the offline regression suite with:

```bash
python3 -m unittest -v test_probes
```

Tests use synthetic questions and injected fake transports in temporary private
directories that are deleted afterward. Their adapter/provider labels are
fixtures, not evidence of live requests. They cover negative
alias matches, day/month boundaries, future questions, full-catalog missingness,
provenance mismatch, atomic imports, failure/retry usage, completed-answer
preservation, allowance exhaustion, interruption recovery, and deterministic
mock output. The `mock` command emits only explicitly synthetic `UNKNOWN`
fixtures, never reference answers or empirical knowledge evidence.

Remaining live work: verify source answers and their information-availability
dates; establish model-specific cutoff claims from primary sources; run the
full probe catalog for each authorized model/reasoning setting; review unmatched
responses and potential source leakage; reconcile provider consumption; and
publish the evidence table with uncertainty intact. A temporally clean Brier
window cannot be certified merely because this test returns few matches.

## Native subscription collection

`harness/run_native_probes.py` implements the factual native collector for these
exact settings: Astra low; Grok low; GLM-5.3 native-default; and Luna none, low,
medium, high, xhigh and max. It reuses only native command construction and text
decoding from the existing runtime. It does not use the probability parser,
forecast prompts, or forecast receipts. Its implementation was validated with
fake subprocesses and temporary evidence; no provider was invoked during that
implementation and validation. Actual campaign observations must be reported
separately by the orchestrating agent.

Prepare one directory per treatment. These commands prepare and display a plan
without starting a native client:

```bash
python3 harness/run_native_probes.py prepare \
  --run-dir runs/native-probes-20260913/astra-low \
  --provider astra --effort low --as-of 2026-09-13

python3 harness/run_native_probes.py collect \
  --run-dir runs/native-probes-20260913/astra-low
```

The common manifest freezes all 129 probes and the factual user prompt. A
separate fingerprinted `native-protocol.json` freezes the factual system
instruction, exact model/effort, collector version, and allowed reported model
identities. The previously observed Grok backend name `grok-4.6-build` is an
explicit pre-campaign alias for requested `grok-4.6`; unrelated model names are
not accepted. This backend identity allowance is independent of answer aliases.
All returned `modelUsage` keys are retained for auditing.

The root orchestrator supplies the existing `CONSISTENCY_CODEX_RUNTIME` path
for Astra/Luna and an external `TMPDIR` for GLM. The collector does not copy
credentials. Each invocation gets a fresh empty working directory and temporary
client directory. GLM uses its established no-index runtime with the agent name
and description changed to factual recall. The frozen factual instruction is
passed as the native system/developer instruction for every provider.

After quota baselines and review, the orchestrator can execute one observation
per treatment, inspect it, and then resume the same full-catalog campaign:

```bash
python3 harness/run_native_probes.py collect \
  --run-dir runs/native-probes-20260913/astra-low \
  --execute --max-calls 1 --workers 1 --timeout 180

python3 harness/run_native_probes.py collect \
  --run-dir runs/native-probes-20260913/astra-low \
  --execute --max-calls 129 --workers 4 --timeout 180
```

These examples describe the CLI; they do not record executions. `--max-calls`
is a cumulative native-launch allowance, including failed or interrupted
attempts. A one-call pilot retains a 129-question target and leaves 128 pending.
`--workers` permits 1–16 simultaneous launches. The collector stops submitting
new work after a provider failure and saves already-running attempts. Known
infrastructure failures require `--retry-errors`; no completed answer is
replaced. Native clients may reconnect internally, so launch counts are not
claims about the provider's internal request or subscription accounting.

The first available terminal response is retained whether it is `UNKNOWN`, an
exact match, unmatched prose, or a response that used a tool. Tool use is
flagged and excluded from recall evidence. Codex completion requires one
matching final message/file, a native session, and a successful terminal event;
only the previously observed recoverable reconnect notice is allowlisted before
that matching completion. Failed turns, unknown error events, extraction
mismatches and missing sessions remain infrastructure failures. Grok's terminal
reason and GLM's step/session completion are checked separately. Timeout kills
only the attempt's own process group and saves the resulting evidence.

The quota reporter can read these durable paths and fields:

| Path / field | Meaning |
| --- | --- |
| `collection/launches.jsonl` | Reservation written before each attempted launch; full target is unchanged |
| `collection/attempts/<id>.jsonl` | One complete native receipt, saved before common-ledger import |
| `collection/attempts/<id>.final.txt` | Codex final-file evidence when produced |
| `receipts.jsonl` | Common receipt/import-hash envelopes, retaining failed attempts |
| Receipt `session_id` and `provenance.session_id` | Native session for attribution; absent sessions are not invented |
| Receipt `raw_usage` and `raw_model_usage` | Exact decoded provider usage, including nested details |
| Receipt `usage` | Normalized counters using the established provider conventions; unavailable/invalid counters are `null` |
| Receipt `elapsed_seconds`, `exit_code`, `requested_model` | Native execution timing and requested treatment |
| Receipt `native_evidence` / `native_validation` | Full stdout/stderr, command, process ID, and completion/extraction checks |
| `native-report.json` | Common coverage report plus runtime/session and unknown-usage counts |

Cache and reasoning counters remain subsets of normalized input/output totals;
they are not added twice. The full raw usage remains available even when
normalization is unavailable. A launch with no saved receipt remains explicit
and blocks blind replay. A saved receipt missing only its ledger import is
recovered without launching another client.

Run the native collector's offline checks with
`python3 -m unittest -v test_native_probes`. These checks include all nine
treatment declarations, all four native transports, prompt/protocol freezing,
alias handling, usage normalization, a one-call pilot and full-target resume,
timeouts, reconnects, tool use and interruption recovery. Fixture session and
provider labels remain temporary test data, not live measurement evidence.


## September 13 completed native campaign and isolated publication

The [completed campaign](native-probes-20260913.md) retains 1,161 observations
across all nine settings. See its [independent source review](probe-source-review-20260913.md)
and [subscription measurements](native-probe-usage-20260913.md).
[Response adjudication](native-probe-adjudication-20260913.md) is also complete:
836 supported core answers, 5 contradicted, 1 incomplete and 319 abstentions.
Two agents reviewed all 157 distinct non-UNKNOWN pairs, with zero label
disagreements; one initial agreement was subsequently overridden using a retained
Google announcement. Both original reviews and their qualifications remain.
Historical-provider measurements are still unfinished. No verified cutoff is
inferred from original-key matches or reviewed source alignment.

The original study and source audit are frozen. **Do not rerun
`scripts/export_native_probes.py` for this reviewed campaign:** its generated
timestamp would change the original study hash and invalidate review bindings.
Render from the frozen study, audit, rubric and response-review files, then
synchronize only the reviewed page and its new downloads:

```bash
python3 scripts/render_probe_adjudication.py \
  --supplemental-evidence data/research/probe-adjudication-supplement-20260913.json
cp data/research/native-probe-adjudication-20260913.json dash/native-probe-adjudication.json
cp data/research/probe-adjudication-rubric-20260913.json dash/probe-adjudication-rubric.json
cp data/research/probe-adjudication-supplement-20260913.json dash/probe-adjudication-supplement.json
rsync -a dash/factual-recall.html dash/native-probe-adjudication.json \
  dash/probe-adjudication-rubric.json dash/probe-adjudication-supplement.json \
  deploy-host:fleet/apps/consistency-dash/
```

Only reviewed public documents are published; raw capture directories remain
private. Existing homepage, corrected-comparison and Open Agency
files are preserved and checked against recorded deployment hashes. The broader
`scripts/deploy_dash.sh` rebuilds all dashboard pages and is unnecessary for this
isolated publication.
