# GRPO pilot: training a forecaster for accuracy, consistency, or both (2026-10-08)

Status: **pre-registered** 2026-10-08, before any training step (bead pool-xo98yf; Kelvin: "Let us do that",
"you can setup this run correctly and put it on the queue"). The Results section is filled in after the runs;
everything above it stays as written, and any change is listed under "Deviations" with its reason.

## Question

Every earlier consistency result applied consistency after the fact (coherence projection of a model's 12
answers) or distilled consistent targets into a student. This is the first direct test of training a model
**for** consistency with reinforcement learning: does a reward for coherence across a pair's 12 Boolean forms,
added to an accuracy reward, make the model's **single** forecasts better on held-out questions, asked one at a
time, with no projection at test time?

## Arms (one seed each in the pilot)

| arm | reward per rollout |
|---|---|
| A (accuracy) | log score of the answered form against its realized truth |
| C (consistency) | negative incoherence of the model's own 12 answers for the pair |
| A+C | A + λ·C, λ = 1.0 |

All arms use the same prompts in the same order, the same number of steps, rollouts, completion tokens, LoRA,
optimizer and seed, and they compute both reward terms on every step (the unused one is only logged), so
compute per arm is the same.

**Rewards, exactly** (`training/grpo_consistency.py`, unit tests `test_grpo_consistency.py`). For rollout i of
form f in pair k, p_i is the parsed "Probability:" (the v7 parser, `verbalized_baseline.parse_probability`)
clipped to [0.01, 0.99].
- a_i = y_f·log p_i + (1 − y_f)·log(1 − p_i).
- c_i = −min(D(v_i), 6), where v_i puts p_i in slot f and, in each of the other 11 slots, the mean of that
  form's parsed rollouts in the same step (same pair), and D(v) = Σ_g KL_Bernoulli(v_g ‖ π_g(v)) with π the kl
  coherence projection of `scripts/coherence_projection.py` (one distribution over the four atoms of P, Q).
  Within a group (G rollouts of the same prompt) only p_i varies, so the group-mean baseline isolates the extra
  incoherence that this answer causes. The cap at 6 nats bounds the reward.
- **Why λ = 1:** both terms are in nats attributable to the one answer (its log loss, and the KL it adds to the
  set's distance from coherence), so λ = 1 prices a nat of incoherence like a nat of log loss, with no tuning on
  outcomes.
- Unparseable reply: the arm's floor minus 1 (A −5.61, C −7, A+C −11.61), below every parsed reward.
- Advantage = reward − group mean, no division by the group SD (Turtel et al. 2025, as in
  `training/turtel_grpo.py`).

## Model and compute

**Qwen3-4B** (`Qwen/Qwen3-4B`, revision 1cfa9a7208912126459214e8b04321603b3df60c), thinking off (chat template
`enable_thinking=False` and the prompt's `/no_think`), LoRA r 32 / α 64 on all linear projections, bf16.

Why 4B and not 8B: the cluster allows 6 GPUs per user, so the three arms can run **at the same time with equal
compute only at 2 L40s each** (one serves vLLM rollouts, one trains), not 4. On one 46 GB L40, 8B training with
~1.1k-token sequences, gradient checkpointing and a reference pass is about twice as slow per step as 4B
(estimated 4–5 min vs 2–2.5 min per 384-completion step), which would leave about one pass over the training
pairs inside the 24 h job limit. Earlier results also say the smaller the model, the more incoherent it is
(projection gain 0.074 nats for Qwen3-32B vs 0.036 for 235B, `docs/strength-scaling-20260930.md`), so a 4B
model has the most room for a consistency reward to act. 8B is the planned follow-up if the pilot shows signal.
The model and revision are fixed: if they cannot be loaded, the run stops (no other model is substituted).

Cluster: MATS Slurm, partition `compute`, 1 node with 8× L40 46 GB, ≤ 4 GPUs per job, ≤ 6 per user, 24 h per
job. Each arm is a chain of three 23 h 50 min jobs (`--dependency=afterany`); each link resumes from the latest
checkpoint. Software: torch 2.13.0+cu129, vllm 0.28.0+cu129, trl 1.13.0, transformers 5.16.1, peft 0.20.0
(the node's driver is 535, CUDA 12.2; the cu129 builds run under CUDA minor-version compatibility, checked in
the smoke run).

**GRPO settings** (TRL GRPOTrainer, vLLM server mode): G = 8 rollouts per prompt at temperature 1.0, top-p
1.0, max 384 completion tokens; 4 pairs × 12 forms × 8 = 384 completions per optimizer step (micro-batch 8,
gradient accumulation 48, one generation per step, on-policy); LR 1e-5 constant, AdamW (0.9, 0.999, 1e-8,
no weight decay), grad clip 1.0, KL β 0.005 against the base (adapter off), PPO clip 0.2, loss "grpo".
Training data are consumed in time order (not shuffled).

**Number of steps** (fixed before any arm starts, from the smoke run only): one pass over the training pairs is
182 steps. The arms run **two passes (364 steps)** if the smoke run's mean time per step (steps 5–20, including
the validation share) × 364 ≤ 44 h, so that two of the three chain links suffice; otherwise **one pass (182
steps)**. The number is written in the "Run record" section below before the arm jobs are submitted.

**Checkpoints**: every 25 min of wall time and at the end (`--save-minutes 25`), keeping the last two.

## Data (`scripts/grpo_pilot_data.py`, outputs in `/home/agent/data/grpo-pilot-20261008/data`, `manifest.json`)

**Pairs.** All 2,264 pairs of `jev-market-context-open-20260923`: the September pairing graph (BGE nearest +
one random nomination, max-weight matching, one pair per proposition) with the open rule: **both questions open
at t = 00:00 UTC of (earlier closing week − 7 days)**, market and option created before t, neither resolved at
or before t. The 178 leak-flagged parents are already excluded. Pooled over the dataset's family splits and
re-split **by time**:

1. Order pairs by t (ties by sha256 of the pair id); cut at 60% and 75%: train (earliest), validation, test
   (latest).
2. **Embargo**: a training pair is kept only if both questions resolved (asof.sqlite `resolution_time`) before
   the first validation t (2025-10-20). So no training label postdates any validation or test forecast time.
   Validation labels are used only for the gates below, never as training targets, and are not embargoed.
3. **Questions held out, not just pairs**: a validation or test pair is dropped if its P or Q parent market or
   family appears in an earlier split; a test pair also if it shares one with validation.

| split | t range | pairs (raw → after embargo → after hold-out) | rows (× 12 forms) |
|---|---|---|---|
| train | 2025-05-05 … 2025-10-13 (raw cut to 2025-10-20) | 1,358 → 729 → 729 | 8,748 |
| validation | 2025-10-20 … 2025-11-17 | 340 → 340 → 127 | 1,524 |
| test | 2025-11-17 … 2025-12-22 | 566 → 566 → 200 | 2,400 |

**No leakage in the text.** Question, option name and criteria are the text **as of t** (the market-context
`question_at_t`, `answer_text_at_t`, `criteria.text` from the edit history), and paragraphs marked as creator
edits ("Update 2025-…", "Edit:") are still stripped (unmarked edits made before t stay: they were public at t). The option name of every multiple-choice proposition is shown
(wave one dropped it). No prices or comments are shown.

**Prompt.** The v7b_explicit wording (`training/verbalized_baseline.py`; the literal forms show only their own
question, compounds both; negations written out with the TRUE/FALSE note), one form per call, system line "You
are a careful forecaster. You estimate probabilities of future events." One change for RL: the answer line asks
for brief reasoning ("Think briefly (at most five sentences), then end your reply with a final line … Probability:
0.xx"), so the policy has something to shape besides the number. The same prompt is used in training,
validation and test.

**External single-question test set (never trained on, never used for gates).** 447 binary questions of 2026
with known truth from `minibench/autoresearch` (registry sha256 in the manifest): 207 Metaculus MiniBench (bt-scale
truth; mb-45000 excluded as a confirmed leak), 92 Metaculus Summer FutureEval, 148 Manifold 2026 (title and
description as of the as-of time from asof.sqlite `market_versions`; mf-66016 and mf-98908 dropped because they
were already decided at their as-of date). Metaculus texts are the bt-scale/autoresearch snapshots. Each is rendered
as the literal "P" prompt of a one-question pair with today = the question's as-of time. All 2025 and 2026
questions postdate Qwen3's training data (released 2025-04-29), so outcome recall is not possible.

## Metrics and analysis (fixed now; `scripts/grpo_pilot_analyze.py`)

Evaluation (`training/grpo_pilot_eval.py`): vLLM, greedy (temperature 0), one form per call, max 384 tokens,
the base model and each arm's final adapter (the adapter at the stop step for a gated arm). Probabilities
clipped to [0.01, 0.99]; an unparseable reply counts as 0.5.

**Primary: held-out single-forecast log loss.** The rows are P and Q of the 200 test pairs (400 Manifold
questions, each answered alone) plus the 447 external questions: 847 single forecasts. Contrasts, each paired on
the same rows: **A+C − A** (the main question), A+C − base, A − base, C − base, C − A. 95% intervals from a
**paired bootstrap clustered by question** (cluster = parent market for Manifold options, the question
otherwise), 10,000 resamples, seed 20261008, percentile intervals; the same resample indices for every contrast.
Also reported per source (Manifold 2025 test, MiniBench, FutureEval, Manifold 2026), with Brier.

**Reading (A+C − A on the pooled singles):** "consistency helps" if the 95% CI upper bound < 0; "hurts" if the
lower bound > 0; otherwise "no detectable difference", with the CI stated. Each arm vs base is read the same way.

**Secondary (on the 200 test pairs, all 12 forms answered one per call):**
- coherence: negation correlation corr(p(A), p(not A)) over A ∈ {P, Q} (ideal −1); share of AND forms more than
  0.1 above min of their two literals (bound violation); share of OR forms more than 0.1 below the max of their
  literals; mean incoherence D per pair;
- joint log loss: mean log loss of the four AND forms (the joint outcomes), and all-12 log loss;
- projection gain after training: all-12 log loss after the kl projection minus raw, per model, and the
  difference in gains between each arm and base (a consistency reward should shrink the gain).
Same clustered bootstrap (cluster = pair).

## Gates (automatic stops, `training/grpo_consistency.py`)

Validation runs at step 0 (= base) and every 25 steps on all 1,524 validation rows, greedy, through the rollout
server after a weight sync.
- **G1 format break**: validation parse rate < 0.90.
- **G2 collapse**: share of literal answers in [0.45, 0.55] exceeds the base share by more than 0.40, or the SD
  of the P and Q answers < 0.05, or the AUC of the P and Q answers against their labels < 0.55 while the base AUC
  is ≥ 0.60 (answers collapsing to 0.5, to one value, or losing all discrimination).
- **G3 kill**: validation single log loss (P and Q forms) worse than base by more than **X = 0.05 nats** at two
  consecutive validations.
- **G4 training format**: training parse rate < 0.50 for 5 consecutive steps.
A gate writes `STOP` with the reason, saves a checkpoint, ends training and the chain; the arm is then evaluated
as it stood and reported as stopped (with the reason). A stopped C arm (e.g. G2) is itself a result: the
consistency reward alone was hackable.

## Pilot → confirmation plan (pre-registered now)

If the pilot's A+C − A primary contrast has a CI that excludes 0, or a point estimate of at least 0.01 nats in
either direction, run **seeds 2 and 3** (seed 20261009, 20261010) of all three arms with this exact config
(steps fixed by this pilot's rule). Confirmatory test: A+C − A on the same 847 singles, using each arm's
seed-averaged log loss per question (mean over the 3 seeds), same clustered bootstrap; plus the seed-level
signs (3 of 3 same sign required to call it robust). No change of rewards, λ, data or steps between pilot and
confirmation.

## Problems (any of these is reported in place of a reading)

P1 the model or software stack cannot run as specified (no substitution); P2 a base or arm eval has more than 5%
unparseable rows on the primary set; P3 an arm did not reach its step count and was not gated (cluster
failure): it is reported with its step count and rerun if possible; P4 the independent verifier finds a
leakage or split defect.

## Smoke run (before the arms)

~20 steps of the A+C arm on 2 GPUs (`--eval-every 10 --eval-limit 20`), then the base eval on all held-out rows
and the LoRA eval path on 50 rows. Checks: server/driver compatibility, whole-pair batches (asserted in the
reward), finite loss, KL > 0, LoRA weights moving, a checkpoint saved and resumed, step time. Its adapter is
discarded.

## Verification before submission

An independent verifier agent reviews this pre-registration, `scripts/grpo_pilot_data.py` and its outputs for
leakage and split defects (time order, embargo, question hold-out, as-of text, option names, label alignment)
before the arm jobs are submitted. Its findings and any fix are recorded below.

**Verifier verdict (2026-10-08, before any training step): PASS WITH NOTES.** An independent agent rebuilt the
cuts, checked the embargo on all 729 training pairs (latest training resolution 2025-10-19 21:05Z; 176
independent-option answers resolved before their parent market, all final), found no shared parent, family or
question id across splits (near-duplicate wordings: train–validation 2, train–test 1, validation–test 1), matched
as-of text to `market_versions`/`answer_versions` on 45 questions, found 0 of 7,984 multiple-choice rows without
an option and no outcome-revealing text, and checked every label (12 forms, all pairs) against the Boolean
function of the asof.sqlite outcomes, and the reward code against this document (to 1e-14). Its notes and what
was done before training:
- G2 could miss a collapse in which every question gets the same answer (negations keep the spread): G2 now uses
  the spread and AUC of the P and Q answers (above).
- Manifold 2026 snapshot text was current text, and two questions were decided at their as-of date: fixed above.
- Base rates differ: the Manifold test questions resolve YES 18% of the time, against 38.5% in train, 41% in
  validation and 37% in the external set. The pooled primary result partly measures calibration level; the
  per-source results (pre-registered) carry the reading for each source, and the report says so.
- 31 validation pairs share t = 2025-11-17 with 4 test pairs (the order cut, ties by hash); they share no
  question, parent or family.
- 8,748 training rows are 182.25 steps of 48 prompts; TRL's sampler drops the incomplete last chunk (1 pair)
  each pass.

## Run record

- **Stack check (smoke 18051, 03:34Z):** torch 2.13.0+cu129, trl 1.13.0, transformers 5.16.1, peft 0.20.0 and the
  vLLM 0.28.0+cu129 server run on the node's 535 driver (torchcodec, built for CUDA 13, was removed: vLLM
  treats it as optional). Base validation on 20 pairs: parse rate 1.000, single log loss 0.816 (worse than 0.5
  everywhere, 0.693), negation corr −0.38.
- **Smoke steps 1–5:** whole-pair batches accepted by the reward (384 completions, 4 pairs × 12 × 8 per step),
  parse rate 1.0, mean completion 102–107 tokens, none clipped, finite loss, grad norm 0.07–0.11, reward
  computation 2–3 s, **289 s per step** (tqdm mean over 5 steps), time-based checkpoint written at step 5.
- **Step count:** 289 s × 364 ≈ 29.2 h (+ ~15 validations of ~1–2 min) ≤ 44 h, so the arms run **364 steps
  (two passes)**. Deviation from the rule's wording: the time per step is the mean of steps 1–5, not 5–20, so
  the arms could be queued while the smoke continued (the cluster queue is busy); a mean above 435 s per step
  over steps 5–20 would have changed the decision, and is checked below.
- **Arm jobs (submitted 2026-10-08 ~04:05Z, chains of 3 × 23 h 50 min, `afterany`):** A = 18052 → 18053 →
  18054; A+C = 18055 → 18056 → 18057; C = 18058 → 18059 → 18060. Run dirs
  `/cluster/home/user/grpo-pilot/runs/arm-{acc,acc_cons,cons}`, eval outputs `/cluster/home/user/grpo-pilot/eval/`.
- **Fixes after submission (code only; rewards, data, gates and settings unchanged):** (1) the in-process vLLM
  eval failed in the smoke because FlashInfer's sampler JIT-compiles with ninja, which the node lacks; the eval
  now sets `VLLM_USE_FLASHINFER_SAMPLER=0` (greedy decoding), commit e14850d; the base eval and a LoRA-path check
  on the smoke adapter run as job 18063. (2) The first A link (18052) stopped 7 s after start: the full base
  validation sent 1,524 prompts in one request and the rollout server allows 1,024; validation now goes in
  chunks of 512 (commit 45eea1c). A restarts from step 0 in its next link (18053). A+C (18055, started 05:30Z)
  ran the fixed code: base validation on all 127 pairs: single log loss 0.882, parse rate 1.000, negation corr
  −0.10, P/Q AUC 0.57 (below 0.60, so the AUC part of G2 is inactive for this run).
- Smoke steps 6–20 (A+C, 20 validation pairs): 263–318 s per step; validation single log loss 0.816 (step 0) →
  0.731 (step 10) → 0.767 (step 20); resume from checkpoint-12 continued with the next pairs in order (no
  repeats). The 364-step decision stands.
- Spend: $0 (the free MATS cluster; no paid API).

## Results

(after the runs)

## Deviations

(none yet)
