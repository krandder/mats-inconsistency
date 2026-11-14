# Supervised semantic replication

> **Results of runs built on this plan are invalid (found 2026-09-23).** `proposition_text` dropped `answer_text`, which for multiple-choice markets (about 79% of propositions) is the option name that defines the proposition, and it left creator edits ("Update 2025-...") in the criteria for about 23% of propositions. Both are fixed in `training/supervised_pairs.py` (option shown, edits stripped, as-of date, plain-language statements). The 20-Sep Qwen3-0.6B numbers are a record, not a baseline.

This is the replacement for the September 18 identifier-only pilot. It remains
supervised LoRA training: each pair has one directly observed binary label, and
the consistency arm adds a differentiable Boolean penalty. There is no RL,
REINFORCE, GRPO, reward model, or sampled action.

## Prompt and data contract

Each prompt contains the complete `parent_question` and `criteria_text` for P
and Q. The current `answer_text` field is excluded because it can contain the
resolved outcome and would leak the label. Every pair must contain exactly the
12 named forms in `training/supervised_pairs.py`.

The loader builds `form -> query_id` explicitly and checks the stored signs and
P/Q IDs. Training and evaluation never recover relations by sorting query IDs.
Missing, duplicate, unknown, or malformed forms stop the run before model
inference. Parent IDs are already disjoint across the archived train,
validation, and test splits; the run manifest hashes every source file.

## Losses and metrics

The accuracy arm uses BCE on the observed query in each pair. The consistency
arm adds the named relation penalty: literal complements, De Morgan complements,
and Frechet lower/upper bounds for all four conjunctions and four disjunctions.
The penalty is computed from the semantic names, so a row reorder cannot change
which propositions are constrained.

Each epoch reports observed-label log loss, Brier score, accuracy, ECE, full-label
metrics by form where available, and every named relation violation. Epoch zero
is the frozen-base baseline; constant 0.5 and empirical-prior baselines are
reported by the analysis layer. Checkpoints are selected by the predeclared
tuple `(observed log loss, consistency MSE)` and the complete epoch curve is
retained. No test metric is used for selection.

## Reproducibility and archive

`scripts/train_supervised_pairs.py` records model/tokenizer settings, package
versions, seed, LoRA configuration, YES/NO token IDs, sequence truncation
counts, source file hashes, metrics, optimizer state, RNG state, and each
adapter checkpoint. `training-live.json` and `events.jsonl` are written after
each training and validation milestone. The script is parameterized for the
archived data at `jev-pairings-v1-filtered-20260918` and does not assume the
old `/workspace/consistency` absolute paths.

The code and unit tests are ready. A candidate-model run is still pending GPU
provisioning; no new scientific result is claimed by this preparation commit.
