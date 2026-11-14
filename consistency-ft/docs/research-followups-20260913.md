# Research follow-up: September 13, 2026

The original-benchmark homepage remains at **23 ranked settings on 264 shared
groups**, with Muse visible but unranked. The separate corrected comparison
remains at **148 shared groups**, from its reviewed 157-group / 274-question
scope. The Open Agency page and all existing benchmark observations are preserved.

This follow-up contributes analysis, tested training integration components,
independent source review, and a live native-subscription factual-recall campaign.
**No paid inference, GPU rental, candidate-model training, or billing change occurred.**
Factual-recall requests and quota snapshots are accounted separately from the
archived consistency benchmark and from agent development work.

## Progress and remaining work

| Bead | Delivered in this update | Still required |
| --- | --- | --- |
| `hub-t8k9c.1` — Fireworks unblock/backfill | [removed] | [removed] |
| `hub-t8k9c.10` — Reasoning sweep | Reproducible [six-effort Luna comparison](reasoning-sensitivity.md), paired by identical groups within each campaign, with per-check scores, coverage, tokens, latency, and source hashes. | Fresh matched medium/high observations for the historical Fireworks models; small-model comparisons needed for base selection. One Luna run per effort cannot establish a causal or monotonic compute effect. |
| `hub-t8k9c.11` — Frozen base beliefs / A2 | [Exact Bernoulli-KL reward and immutable snapshots](kl-anchor.md), the [episode-to-token adapter](training-integration.md), and a [real CPU generation/autograd/optimizer path](synthetic-cpu-training.md) verified with synthetic weights and complete five-proposition anchors. The [actual Qwen tokenizer](qwen-tokenizer-preflight-20260913.md) covers the full corpus. [Local pretrained execution](qwen-candidate-execution-20260913.md) now verifies three numerical EOS-terminated readouts, one numerical backward with finite adapter gradients, and unchanged weights; the [LoRA runtime](qwen-lora-runtime-20260913.md) passes reduced architecture checks. | Select and pin the scientific base, prompt/template, reasoning mode, lambda and endpoint convention. Collect complete candidate-base training anchors, verify full candidate joint-batch/optimizer execution, then run and measure the candidate experiment. |
| `hub-t8k9c.9` — Contamination runner | [Native collectors](contamination-audit.md) completed all **1,161 observations** across nine settings. The [129-row source review](probe-source-review-20260913.md) retains 202 captures, 35 flagged probes and 12 unapplied corrections. [Response adjudication](native-probe-adjudication-20260913.md) now covers every observation, preserving the original key and both agent reviews. | Historical provider observations remain. Source/premise/date qualifications are retained; no verified cutoff or uncontaminated evaluation window is established. |
| `hub-t8k9c.13` — Platform/model choice | [Capability assessment](training-platform-decision.md), [cost guidance](../COSTS.md), [hashed sources](../data/research/platform-evidence-20260913.json), and a [runtime audit](training-runtime-readiness-20260913.md). The [Qwen preflight](qwen-candidate-preflight-20260913.md) verifies tokenizer lengths, 427 text tensors and 248 decoder Linear targets. [Actual frozen CPU generation](qwen-candidate-execution-20260913.md) took 400.591 seconds for three readouts with 17.470 GiB peak RSS; one retained-answer derivative took 236.994 seconds / 17.808 GiB. | Matched candidate baselines, scientific checkpoint/treatment selection, full candidate optimizer/training memory and throughput, and a current resource quote/allowance. The measured generation and single-derivative probes do not establish training or GPU fit. The platform and scientific base are not committed. |

All five beads remain unfinished. The new CPU experiment trains a locally
initialized 75,484-parameter synthetic fixture. It establishes the software path;
three pretrained candidate readouts now establish bounded local generation,
and one retained response passed actual numerical backward with finite gradients.
Full candidate anchors, scientific training and GPU feasibility remain unverified.

## Completed native factual measurement

All **1,161 observations** are complete: 129 questions each for Astra low, Grok
low, GLM-5.3 native-default, and Luna none/low/medium/high/xhigh/max. Each setting's
one-call pilot was retained within its full campaign. There were **zero
infrastructure failures, answer-based retries, tool-use observations, model
mismatches or unreconciled launches**. The full catalog was retained.

The separate [factual-recall page](https://consistency.futarchy.ai/factual-recall.html)
and [procedure/full response tables](native-probes-20260913.md) preserve original-key
grades, independently reviewed source evidence, and proposed corrections without
silently applying them. These are not certified accuracy estimates or verified
cutoffs. Grok/GLM saved command arguments bind to the question bytes. Astra/Luna
stdin was not archived: their prompt assurance relies on the frozen collector
code and recorded hashes rather than an independent raw-stdin replay.

The [usage report](native-probe-usage-20260913.md) accounts for **7,583,989 native
tokens**, including 3,302,656 cached input tokens as a subset. All 1,161 receipts
have usage counters; all 903 Astra/Luna sessions match exact proxy records across
seven anonymous subscriptions. Grok's shared weekly meter rose **2% → 3%**;
GLM's weekly meter stayed **1% → 1%**. OpenAI shared weekly changes ranged from
0 to 10 percentage points. These rounded, shared readings include possible
concurrent fleet work and do not measure the experiment's exact marginal quota
fraction. GLM's short-window readings changed 0% → 1%, but missing reset identity
prevents a defensible subtraction.

## Completed native response review

The [frozen review](native-probe-adjudication-20260913.md) reduces 1,161 observations
to **208 distinct question/response pairs**. A mechanical rule labels 51 exact
UNKNOWN pairs as abstentions. Two agents independently reviewed the remaining
157 pairs with model, effort, original grade and frequency omitted. Their labels
agreed on every pair; they share a model family and source packet, so agreement
does not provide independent factual evidence or human adjudication.

One initial unresolved answer, “Gemini 3 Pro,” now has supporting evidence in
a dated Google announcement captured after the reviews. Its documented override
retains both original unresolved reviews, the new source and its limitations.
Final observation labels are **836 supported core answers, 5 contradicted,
1 incomplete, 319 abstained, 0 unresolved**. This supports 144 observations missed
by the original key; no original exact match was downgraded. Original question,
source and date qualifications remain. No new native benchmark requests were
made for adjudication; agent review/development use is separate from campaign
subscription measurements. These labels assess the requested core answer and
do not certify every claim in the response, accuracy, cutoffs or contamination.

## Denominators and interpretation

The Luna-only original intersection is 264 groups / 516 contributing questions.
The Luna-only corrected intersection is **150 groups / 260 contributing
questions**. The corrected website compares a larger set of models, including
Astra and Grok, and therefore retains **148** shared groups. These are different
participant intersections, not conflicting counts.

The KL integration example reuses three reviewed film propositions from final
GLM receipts. It anchors **3 of 2,926 training propositions** and **1 of 1,619
training episodes**. A requested native model name does not establish a pinned
trainable weight revision. The example is not the selected base's full snapshot.

The factual-recall source contains **129 probes**, despite the old task's
90-probe description. The runner preserves the entire catalog, including
invalid, future, date-uncertain, and pending rows. Matching a factual answer
does not itself prove training contamination. UNKNOWN and failed recall do not
certify a safe upper knowledge cutoff. Synthetic UNKNOWN-only fixtures are
identified as synthetic and cannot complete a real campaign.

## Synthetic CPU training integration

The [CPU integration report](synthetic-cpu-training.md) covers actual categorical
generation, frozen own-base elicitation, episode rewards, differentiable loss,
backward and optimizer updates. Its model, tokenizer and fair-coin fixtures are
constructed locally; the finite initial probability/EOS biases are explicit.
They are not downloaded candidate weights or a scientific base selection.

An independent run used model seed `20260913`, anchor seed `101`, and predeclared
rollout seeds `202` and `203`, with SGD at `0.01` and one CPU thread. It collected
all **five synthetic anchor propositions** and **20 role trajectories / eight
joint samples**, completing two nonzero parameter updates without replacement
sampling. The initial anchors remained byte-identical. Reloading the retained
checkpoints reproduced all **50 generated-token log probabilities** exactly and
verified both checkpoint transitions and every saved parameter delta.

The independent [finite-difference verifier](../scripts/verify_cpu_loss.py)
checks uneven completion lengths, full joint reward credit, EOS, prompt/padding
zeros and optimizer direction. All **210 derivatives** matched within
`4.2e-11`; [its recorded result](../data/research/synthetic-cpu-loss-oracle-20260913.json)
also confirms rejection of stale rollout weights. Two fresh stochastic batch
losses are not a measure of optimization quality. None of these fixtures enters
the consistency ranking or substitutes for the candidate's 2,926-proposition
training anchor.

## What unblocks the next measurement

The [Qwen candidate preflight](qwen-candidate-preflight-20260913.md) supplies
actual corpus token lengths and checkpoint structure without loading weights or
generating answers. All 11,098 distinct prompt records across two separate
framings roundtrip with the real tokenizer. The proposed training framing has
maximum prompt length 406 tokens and still requires 2,926 own-base readouts.
The stock decoder silently drops 243 model-output IDs; the candidate adapter
rejects them and retains the original tokens. Its termination treatment is
explicit because the pinned tokenizer and model config disagree about EOS.

The full-size text architecture matches all 427 checkpoint tensor names/shapes
on the meta device. An illustrative rank-16 adapter over 248 decoder Linear
modules has a 17.323 GiB base/adapter/gradient/Adam-state lower bound, before
activations and recovery. The original synthetic helper's full-state copies
alone would require 50.033 GiB. The new [candidate loader](qwen-checkpoint-loader.md)
and [LoRA runtime](qwen-lora-runtime-20260913.md) address duplicate base storage,
BF16 hashing, adapter-only rollback, selected vocabulary projections and
explicit checkpointing under eval. Their reduced tests also reject a
coherently omitted role by checking the complete original rollout plan.

The [actual pretrained generation probe](qwen-candidate-execution-20260913.md)
loaded all 427 text tensors from the verified public archive and generated all
three predeclared original film roles with numeric answers and explicit EOS.
It retained every sampled token and confirmed that every loaded text weight
remained unchanged. Total time was 400.591 seconds, including 155.839 seconds
for verified loading; peak process RSS was 17.470 GiB. These are frozen CPU
generation measurements, not measured training or GPU fit. No own-base snapshot
covering the complete 2,926-proposition training split has been collected.

The separate [pretrained numerical backward probe](qwen-backward-probe.md)
reproduced all four sampled log probabilities of the predeclared first response
exactly, then differentiated their sum through the candidate LoRA runtime.
All 496 adapter gradient tensors were finite, with nonzero gradients in all
248 B tensors; base and adapter values stayed unchanged. It took 236.994 seconds
and peaked at 17.808 GiB RSS. This uses no fabricated anchor or episode credit,
performs no optimizer update, and does not establish candidate training.

Four fresh September 13 authenticated requests returned HTTP 412 /
PRECONDITION_FAILED for **[removed]**. The recovered Mac browser exposes only
**Futarchy / [removed]**, a different account. Its visible credits and cap cannot
diagnose the project account's suspension. The project account's own browser
login is still needed; no key was created, copied or substituted.

Once the account mismatch is resolved, inspect the actual suspension cause
read-only. A concrete inference run then needs reviewed models, scope,
reasoning settings, prices, and call/cost allowances. M2.7 is not currently
listed for serverless use, so it cannot be assumed to recover with the account.

The original project preference was **Qwen 3.5 9B on Fireworks**. A bounded
self-hosted gpt-oss-20b LoRA smoke test is an alternative engineering control
after runtime preparation and an approved allowance. Its official one-H100
example is SFT; it does not prove that this RL workload fits. Compare Qwen 3.5
9B before a scientific base commitment. The
published Fireworks Qwen RL shape is 3 × B200, nominally $39/hour at the current
rate, and the current RFT guide explicitly says not to assume a free tier.

## Validation

The complete repository suite passes **248 tests with the isolated CPU and pinned tokenizer runtimes**, including the native
collector, export integrity/privacy, quota accounting, training integration,
adjudication and its renderer. Supplemental source hashes, byte-verified frozen
reviews and resolutions, original-grade joins and complete coverage are validated
before rendering. Altered rationales and recounted forged overrides are rejected.
The suite includes actual local tensor/model execution, seven checkpoint-loader
tests, 14 candidate LoRA tests and six tests with the real pinned Qwen tokenizer.
Checks requiring optional runtimes/artifacts skip when those are absent;
the recorded integrated run had no skips. Runtime checks
cover replay with unchanged weights, optimizer-state overflow and rollback,
partial failed-generation retention, and output-directory collision protection.
Tests use synthetic inputs and fake provider transports, with no additional
provider requests. Independent export review caught and resolved privacy,
prompt-argument and cross-model session checks before publication.

Independent review reproduced the Luna report from its archived scores and
receipts, including all 30 paired comparisons and reported usage/latency.
The pinned partial GLM snapshot reloads successfully. All **18** referenced
Fireworks/platform source hashes match the retained public-source captures.
All 129 independent source-audit joins, 202 capture records and 400 retained
body/text file hashes passed a separate integrity check. Proposed corrections
remain unapplied. Existing homepage, corrected comparison and Open Agency
files match a retained deployment baseline; factual recall is published on its
own page. All 1,161 exported rows were checked against native evidence and the
usage report, with private session identities omitted. The original factual page
was also verified in a genuine Mac browser, including all nine full tables,
filtering, JSON downloads and mobile layout without overflow. The reviewed
overlay was subsequently published in commit `5475b2039e45900b8cea917ea976c5de352d7312`
and checked in the same genuine Mac browser: all 1,161 responses, original grades
and reviewed labels; all nine setting summaries and filters; all six JSON
downloads; supplemental citations and the retained initial unresolved reviews.
Desktop/mobile captures show no page overflow or JavaScript errors; wide tables
remain horizontally scrollable. Live homepage and corrected-page checks confirm
23 ranked settings, Muse visible but unranked, and 264 / 148 shared groups.
All 18 other existing static files, including Open Agency, remain byte-identical
to the pre-publication baseline. The existing status publisher separately
refreshes `plan-status.json`.
