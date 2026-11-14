# Next GPU run: plan after the supervised semantic fix (2026-09-19)

> **Results of runs built on this plan are invalid (found 2026-09-23).** `proposition_text` dropped `answer_text`, which for multiple-choice markets (about 79% of propositions) is the option name that defines the proposition, and it left creator edits ("Update 2025-...") in the criteria for about 23% of propositions. Both are fixed in `training/supervised_pairs.py` (option shown, edits stripped, as-of date, plain-language statements). The 20-Sep Qwen3-0.6B numbers are a record, not a baseline.

Status: decisions taken 2026-09-19 (Kelvin): direct `[YES/NO]` readout, no
reasoning traces; the 41 uncertain-leakage parents are excluded; spare GPU time
goes to a λ sweep and a second seed. Budget is **$100** freshly loaded on Vast
(the earlier $45 account is gone). Nothing rented, nothing launched.

## What changed and why the last run does not count

The 18 September filtered run (`filtered-20260918`, 4×H100, both one-label arms,
early-stopped at epochs 4/5, best validation log loss ≈0.603/0.605) was an
**identifier-only technical run**: the prompt carried proposition IDs, not text,
the twelve forms were recovered by sorting query IDs, and relation penalties were
applied to row positions 0–3. Its numbers say nothing about forecasting.

PR #3 (merged today, `training/supervised_pairs.py`, `scripts/train_supervised_pairs.py`)
fixes this: full `parent_question` + `criteria_text` in every prompt, `answer_text`
excluded (outcome leak), forms mapped by name with hard validation, Fréchet +
De Morgan + complement penalties on named forms, checkpoint selection by the
predeclared `(observed log loss, consistency MSE)` tuple, test scored once at the
end. Unit tests pass locally.

## Fixes on branch `plan/next-run-20260919`

- `--train-labels {one,full,single}`; singles files are keyed by proposition ID and label only P and Q.
- `--patience N` early stopping on the selection tuple; every epoch checkpoint kept.
- Per-pair predictions with form, label, observed flag and family IDs, for validation each epoch and for test.
- `--pair-weights` applies the graph's `pair_weight`.
- `--preflight-only` tokenizes every prompt and reports length quantiles and over-length counts.
- Test is scored once, on the **best validation checkpoint**, not the last epoch.
- Criteria text capped at 1,500 characters per proposition and truncation from the left, so the `Answer [YES/NO]:` readout can never be cut off. Reason: validation-week criteria are about five times longer than train (median ≈2,750 vs ≈520 characters), which at a 1,024-token limit truncated most validation prompts.
- `scripts/filter_jev_disjoint.py` filters a frozen export by parent ID without adding edges.

Datasets on General (hash manifests inside each):

| Dataset | Train pairs | Val pairs | Test pairs | Excludes |
|---|---:|---:|---:|---|
| `jev-disjoint-v1-filtered-20260919` | 2,164 | 422 | 1,066 | 137 leakage + 41 uncertain parents |
| `jev-pairings-v1-filtered-u41c-20260919` | 8,192 | 1,530 | 3,935 | 41 uncertain, on top of the 18 Sept 137 |

Stale partial directories to delete by hand (dcg blocks agent deletes):
`jev-pairings-v1-filtered-20260919`, `-u41-`, `-u41b-`.

## Gaps identified earlier (all addressed above except where noted)

| # | Gap | Fix | Effort |
|---|---|---|---|
| 1 | Script trains only from `one-label-targets.jsonl`; arms 1–3 (singles baseline, full-label accuracy, full-label +consistency) cannot be run | add `--train-labels {one,full,single}` and a `--forms` restriction for the singles arm | small |
| 2 | Only `jev-pairings-v1` has a leakage-filtered export. Its one-label mask was **rejected** (88.5 % of hidden labels propagate). No filtered `jev-disjoint-v1` exists | build `jev-disjoint-v1-filtered-20260919`: drop the same 137 excluded parents (and decide on the 41 uncertain ones) from the disjoint matching, re-run the hidden-label entailment audit, hash and manifest it | small–medium |
| 3 | No early stopping; every arm runs `--epochs` regardless | add `--patience 2` on the selection tuple; keep every epoch checkpoint | small |
| 4 | Metrics are aggregates only; no per-pair predictions | dump `predictions-epoch-XX.jsonl` (pair_id, form, p, label, family_id) so family-clustered CIs and baselines can be computed offline | small |
| 5 | Agreed plan called for fixed reasoning traces from the base model; the fixed script is a direct `Answer [YES/NO]:` readout with no thinking | **decision needed** (below). Direct readout is cheaper and matches the corrected pilot's readout; traces add a generation stage and a second code path that has not been re-validated since the identifier bug | — |
| 6 | Full-label graph pair weights (`.5·(1/deg P + 1/deg Q)`) from the agreed plan are not applied | apply as per-pair loss weight in the full-label arms only | small |
| 7 | `--max-length 4096` default; pair prompts carry two criteria blocks | preflight: tokenize all splits, report length distribution and truncation count; set max length to cover ≥99.5 % (expect ≤1024) | trivial |

## Arms (matched within experiment, one seed first)

| Arm | Data | Train labels | Loss | GPU |
|---|---|---|---|---|
| A1 singles baseline | pairings-filtered | `single-targets` (P, Q only) | BCE | 2 (after A4/A5) |
| A2 full-label accuracy | pairings-filtered | all 12 | BCE, pair weights | 0 |
| A3 full-label +consistency | pairings-filtered | all 12 | BCE + λ·penalty | 1 |
| A4 one-label accuracy | disjoint-filtered | 1 per pair (fixed mask) | BCE | 2 |
| A5 one-label +consistency | disjoint-filtered | same mask | BCE + λ·penalty | 3 |

Shared: Qwen3-0.6B `6130ef31…`, LoRA r=16 α=32 dropout .05 on attention+MLP
projections, AdamW lr 1e-5, batch 8 pairs (96 sequences), bf16, seed 20260919,
max 6 epochs with patience 2, λ=1.0. Identical prompts, tokenizer, order and
evaluation across arms. Second wave on GPUs 2–3 after A4/A5 finish: A5 with λ = 0.3 and λ = 3, then a
second seed of A4 and A5. A2/A3 stay single-seed at pilot scale.

The main scientific comparison is A4 vs A5 (does a Boolean penalty help when
labels are scarce and cannot be propagated) with A2 vs A3 as the full-label
control. A1 tells whether pair prompts hurt singles at all.

## Budget and time

Estimate from the 18 September throughput scaled by prompt length (ID prompts →
≈300–600 tokens): ≈25–30 min per epoch per arm for the 8,280-pair full-label
data, ≈10 min for the 2,567-pair disjoint data, validation included. Wall clock
≈3.5 h for the first wave on 4×H100 at the quoted ≈$6.94/h ≈ $25; the λ sweep
and second seed add ≈1.5 h ≈ $10. Total ≈ $35–40 of the $100. Hard stop remains 5 h or a
30-minute absent lease, whichever first; export before teardown, verify hashes
on General, then destroy and confirm via API.

## Launch checklist

1. Land gaps 1, 3, 4, 6, 7 in one PR with tests; build the filtered disjoint export (gap 2) and commit its manifest.
2. Resolve the 41 uncertain parents (recommend exclude; they are 41 of ≈5,000 train parents).
3. Local CPU smoke: 2 pairs × 1 epoch per arm on the tiny fixture, then 20 real pairs; confirm exact one-token ` YES`/` NO`, finite grads, adapter save/reload, base weights unchanged.
4. Rent; run `bootstrap-one-label.sh`; stage code+data tarballs; GPU smoke on 20 pairs; measure real epoch time; recompute the projection before launching all arms.
5. Launch arms per the GPU map; sync `events.jsonl`, `metrics.json`, predictions and adapters to General after every epoch.
6. On completion: score best checkpoints on test once; compute baselines (constant .5, empirical prior, independent marginals .5/.25/.75) on the same rows; family-clustered bootstrap CIs; report hidden-label strata (entailed vs underdetermined) separately for A4/A5; per-form Brier/log loss and every named violation.
7. Publish a results notebook page only after hash verification; raw paths and keys stay private.

## Remaining go/no-go

Kelvin's explicit go is needed to rent (checklist step 4). Steps 1–3 are on this
branch; preflight and CPU smoke results are appended below when complete.

## Not in this run

RL / exploitability reward, KL-to-previous-beliefs anchor, cross-session
overlap coherence, effort ladder and predicted-movement targets (see the
explicit-beliefs design page). Those need a working supervised baseline first.

## Preflight and CPU smoke results (2026-09-20)

Preflight, 1,500-character criteria cap, Qwen3-0.6B tokenizer:

| Dataset / split | Median tokens | Max tokens | Prompts over 1,024 |
|---|---:|---:|---:|
| disjoint train | 314 | 1,052 | 8 of 25,968 |
| disjoint validation | 773 | 942 | 0 of 5,064 |
| disjoint test | 313 | 896 | 0 of 12,792 |

Use `--max-length 1280`: zero truncation on every split. Validation prompts are
still about 2.5 times longer than train or test, so validation loss is not
directly comparable to train loss. Compare arms only within the same split.

CPU smoke on a 3-pair train / 2-pair validation / 2-pair test fixture, all
exit 0, best-checkpoint test scoring and prediction dumps verified:

| Mode | Result |
|---|---|
| consistency, full labels | 3 epochs, consistency MSE 0.121 → 0.107 → 0.091, best epoch 3 |
| accuracy, single labels | 2 epochs, best epoch 2 |
| accuracy, one label | early stop after epoch 2, best epoch 1, best-adapter reload path exercised |

The smoke found and fixed three defects in the merged PR code: full-sequence
logits (about 20 GB per batch), a whole batch in one backward pass (over 100 GB
of activations at real prompt lengths), and a duplicated `epoch` argument that
crashed every run after its first validation. Training now does one pair per
forward and backward with gradient accumulation. Two smokes must not share a
62 GB CPU host; one was killed for memory when they did.

Not measured yet: GPU throughput. Measure it on the rented box with a 20-pair
run before launching all arms, and recompute the cost projection then.
