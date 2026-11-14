# consistency-ft

Finetuning LLMs for internally consistent (coherent) probabilistic beliefs —
not accuracy, consistency. Private brainstorm → v0 experiments.

## The bet

An inconsistent model is one whose knowledge is fragmented: it prices the same
proposition differently depending on the route the question takes into its
weights. Training for coherence forces reconciliation, and reconciliation may
be where extra capability leaks out (see HYPOTHESES.md).

## Core design (as of 2026-08-19 brainstorm)

- **Loss**: consistency violation over logically related proposition tuples
  (negation, conjunction bounds, implication, conditional chains). Candidate
  scalarizations: semantic loss (LoCo-LM style), arbitrage/Dutch-book
  exploitability (Paleka metric / RLE's LP), or distillation from
  arbitrage-repaired targets.
- **Anchor**: pure consistency has degenerate optima (all-0.5, constant-yes,
  arbitrary self-consistent belief sets). Every published success anchors with
  ground truth or a confidence term. **Our open lane: anchor with KL to the
  previous iteration's beliefs** — coherentize what the model already believes,
  works on unresolved propositions, iterate.
- **Data**: compose proposition tuples *by construction* from knowledge-graph
  atoms (exact logic, no LLM-verified constraint labels). Keep natural-language
  forecasting tuples (Paleka) as held-out generalization test.
- **Headline metric**: held-out-check transfer (+ ground-truth Brier where
  available). In-training consistency is not evidence — Paleka's inference-time
  arbitrage improved only the optimized checks.
- **Elicitation**: measure verbalized vs logit vs probe readouts separately;
  cross-elicitation agreement is itself a consistency check. If we add a
  scalar proposition head, tie it to verbalized outputs with a loss
  (decoupling is a documented failure mode).

## Key prior art (details in LITERATURE.md)

- Paleka et al. 2412.18544 — 10 checks, arbitrage violation metric, question
  pipeline. Our v0 evaluation harness. Training explicitly left as future work.
- RLE (Matta/Naphade/Zou, ICML'26 workshop; CMU thesis has the formulation) —
  RL reward `R = −Exploit(c,S) − λ·Brier(c,e)`; Exploit = LP value of optimal
  adversary stakes. Adapter-scale only; anchored on ground truth. We swap the
  Brier anchor for KL-to-previous-beliefs.
- LoCo-LMs 2409.13724 — semantic-loss finetuning works and transfers; the
  proven supervised recipe.
- No public code for RLE (krishmatta/consistency is an empty placeholder);
  lijuliana/coherencebench-v0 has a harness + projection-repair layer worth
  mining.

Research connection: [consistency-ft and the Open Agency Architecture
roadmap](docs/open-agency-connection.md) — probability elicitation, shared
world models, and a proposed experiment on downstream decision quality.
[Illustrated page with interactive example](https://consistency.futarchy.ai/open-agency.html)
([HTML source](site/open-agency.html)).

## Status

Brainstorm stage. Collaborators: Kelvin + two frontier-lab researchers.

## Usage

Clone the published tuples and extract them (the sparse vendor checkout is
ignored):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/dpaleka/consistency-forecasting.git vendor/consistency-forecasting
git -C vendor/consistency-forecasting sparse-checkout set src/data/tuples/2028
python3 scripts/extract_tuples.py
```

Run the offline smoke checks, or use a Fireworks key from
`FIREWORKS_API_KEY` / `~/.config/fireworks/env`:

```bash
python3 test_checks.py
python3 harness/run_checks.py --mock --limit 5
python3 elicit/run_elicit.py --mock --limit 5
python3 harness/run_checks.py --limit 5
python3 elicit/run_elicit.py --limit 5
```

Both runners write timestamped JSONL files under `results/`. The consistency
runner reports basic violation magnitudes only; arbitrage/LP is left as TODO.


## Reproducible ranking and model database

The published dashboard builds from versioned, sanitized evidence in
[`data/benchmark-runs/`](data/benchmark-runs/), including all recovered August
check and elicitation runs. [`manifest.json`](data/benchmark-runs/manifest.json)
freezes historical timestamps and records content hashes. Fresh checkouts do
not need the ignored `results/` or private native CLI logs.

The [homepage](https://consistency.futarchy.ai/) shows the **original benchmark**:
the original 300-group catalog, historical runs, and `native-20260912` results for
GLM-5.3, Grok-4.6, Astra, and all six Luna reasoning levels. Its 23 participating
settings share 264 scorable original groups. The incomplete Muse archive remains
visible but unranked. [Original details](https://consistency.futarchy.ai/details.html)
show each setting's original coverage alongside the shared comparison.

The [corrected comparison](https://consistency.futarchy.ai/corrected.html) uses
the separate reviewed `descriptive-v2` campaign: 157 groups and 274 questions.
[Corrected details](https://consistency.futarchy.ai/corrected-details.html) provide
its full table, exclusions, coverage, and subscription accounting. The original
homepage retains the combined ranking chart, heatmap, paired elicitation,
examples, and run history. All pages build from the same versioned evidence;
original and corrected cohorts are selected explicitly, never mixed as one run.

```bash
python3 scripts/export_benchmark.py   # rebuild data/benchmark.sqlite from tracked evidence
python3 scripts/build_dash.py        # dashboard + downloadable database in dash/
python3 -m unittest test_benchmark_data -v
```

After collecting more native receipts, explicitly refresh the tracked snapshot
before rebuilding. Review and commit the resulting data changes:

```bash
python3 scripts/export_benchmark.py --refresh --campaign native-20260913-scoped-final \
  --comparison-policy complete-historical-plus-native
python3 scripts/build_dash.py
```

Each native provider/model/reasoning setting has its own database identity,
including all six GPT-5.6-Luna levels: `none`, `low`, `medium`, `high`, `xhigh`,
`max`. Pending configurations have no fabricated score. Refusals and invalid
outputs retain a null probability; dependent tuples are omitted. Reviewed
scope exclusions are recorded separately from provider readouts. Both comparisons use one common scored tuple set across their participating
settings. Incomplete historical settings remain visible but unranked. Native
subscription and historical Fireworks API runs differ in agent scaffolds, dates,
and reasoning settings. These scores measure algebraic
coherence; they do not establish forecasting accuracy or calibration.

The SQLite tables preserve model settings, runs, native proposition receipts
and retries, tuple scores, old verbal/logprob readouts, and subscription usage.
`comparison_ranking` refers to the **corrected** common scored tuple intersection.
`ranking` retains latest raw run aggregates; `runs` preserves the published
history and the final campaign. The original homepage explicitly selects the
original native campaign alongside historical models. `current_probabilities` selects each native proposition's current receipt
with protocol, scope, and import provenance.
For example:

```sql
SELECT requested_model, reasoning, protocol_id, scope_id,
       scored_count, mean_violation
FROM comparison_ranking ORDER BY mean_violation;

SELECT reasoning, COUNT(*) AS completed,
       COUNT(probability) AS valid_probabilities
FROM current_probabilities
WHERE requested_model = 'gpt-5.6-luna' AND completed = 1
  AND scope_id = 'original300-reviewed-final-v3'
GROUP BY reasoning;
```

Usage snapshots live in
[`data/benchmark-runs/subscription-usage.json`](data/benchmark-runs/subscription-usage.json)
when measured. Token counts and subscription quota deltas are separate evidence;
account-wide quota changes can include other fleet work. Private CLI logs,
session IDs, and authentication data are excluded from public exports.

See [native evaluation methodology](docs/native-evaluation.md),
[the remediation account](docs/benchmark-remediation-report-20260913.md),
[political constraints and missing work](docs/political-constraints-and-missing-work-20260913.md), and
[research tasks and dependencies](docs/BEADS.md).

The [September 13 research follow-up](docs/research-followups-20260913.md)
records progress and missing work on Fireworks recovery, reasoning sensitivity,
frozen base beliefs, factual-recall probes, and the training-platform decision.
These follow-ups do not add new provider observations to the ranking.

The separate [native factual-recall study](https://consistency.futarchy.ai/factual-recall.html)
contains all 129 questions for Astra, Grok, GLM-5.3 and six Luna reasoning levels.
See its [procedure and full tables](docs/native-probes-20260913.md),
[source review](docs/probe-source-review-20260913.md), and
[measured subscription usage](docs/native-probe-usage-20260913.md).
The [response review](docs/native-probe-adjudication-20260913.md) adds source-alignment
labels to all 1,161 observations: 836 supported core answers, 5 contradicted,
1 incomplete and 319 abstentions. Original-key grades, source flags and both
independent agent reviews remain visible. These labels do not establish certified
accuracy or a verified knowledge cutoff.

The [training-runtime audit](docs/training-runtime-readiness-20260913.md) records
General's available CPU environment and candidate-specific gaps.
The [synthetic CPU integration](docs/synthetic-cpu-training.md) now runs actual
generation, frozen own-base anchors, differentiable A2 loss and optimizer updates
with a tiny local model. No candidate-model training experiment has run.

The [Qwen candidate preflight](docs/qwen-candidate-preflight-20260913.md) checks
the actual tokenizer against the full episode corpus and the full-size model
structure against checkpoint headers. It identifies a decoder failure for
unmapped output tokens and the required memory/runtime changes.
The [actual candidate execution](docs/qwen-candidate-execution-20260913.md)
now verifies local pretrained loading and three complete numerical readouts,
with unchanged base weights, 400.591 seconds total and 17.470 GiB peak CPU
process RSS. A separate retained-answer backward probe reproduced four sampled
log probabilities exactly and produced finite gradients on all 248 LoRA targets,
with 17.808 GiB peak RSS and unchanged weights. The
[Qwen LoRA runtime](docs/qwen-lora-runtime-20260913.md) also passes reduced
complete-batch/optimizer checks. Complete candidate own-base elicitation,
candidate optimizer updates and the scientific training experiment remain unfinished.

The [durable Qwen collector](docs/qwen-base-collection.md) binds all 2,926 original
training propositions to the actual initial base and seeded adapters, retains
each sampled token, and resumes only untouched questions. Failed or invalid
attempts remain missing; export requires complete valid coverage. The
[collection execution record](docs/qwen-own-base-collection-20260913.md) retains
startup failures alongside measured requests. The
[collection progress report](docs/qwen-own-base-collection-progress-20260913.md)
records subsequent invocations and cumulative coverage against the same full catalog.
That v2 campaign stopped at 22 attempted propositions: 21 parse-valid responses
and one invalid response. A reproduced [CPU sampler defect](docs/qwen-cpu-sampler-defect-20260913.md)
prevents treating that incomplete campaign as a verified categorical anchor;
all original receipts remain preserved.
The separately versioned [corrected v3 collection](docs/qwen-cdf-collection-20260913.md)
began with two completed requests: 2 valid responses / 2,926 original
propositions, 9 generated tokens, unchanged base and adapters, and zero provider
inference calls. It uses the corrected full-vocabulary CDF sampler and imports
none of the earlier campaigns' observations. This is partial collection;
complete matching anchors and candidate training remain unfinished.
The [v3 continuation report](docs/qwen-cdf-collection-progress-20260913.md)
records **34 valid responses / 2,926 propositions**, 156 generated tokens and
zero invalid responses through tranche 05. Complete anchors remain unfinished.
The [full-catalog resource scenarios](docs/qwen-anchor-resource-scenarios-20260913.md)
explain the CPU work still needed without treating a small probe as throughput
evidence for the whole population.
