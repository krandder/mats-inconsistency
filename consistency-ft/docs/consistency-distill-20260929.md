# Consistency distillation, 2026-09-29

**Question.** Post-hoc coherence projection (kl) of Qwen3-32B's 12 answers per pair moves test log loss from 0.594 to 0.526. Can a small student trained on the projected answers give that accuracy in one call per question, without projecting at inference time?

**Answer (Qwen3-0.6B only; the 1.7B arms were lost to spot preemption).** Partly. The student trained on projected targets (C) beats the student trained on raw teacher targets (R) by 0.013 nats: 0.555 vs 0.568, 95% CI of the difference [-0.017, -0.009]. It also beats the raw teacher (0.594) and the base-rate table (0.570). It covers only about a third of the distance from base rate to the projected teacher (0.526). Once each student's own answers are projected, C and R are the same (difference -0.003, CI [-0.007, +0.001]). So the C student's gain is mostly coherence that projection could add at inference anyway.

Numbers: `docs/consistency-distill-20260929.json` (full report), `docs/consistency-distill-20260929-chart.json` (chart data). Code: `scripts/distill_data.py`, `scripts/distill_student.py`, `scripts/distill_analyze.py`, `scripts/gpu/`.

## Design

- **Pairs.** From jev-disjoint-v1-filtered-20260919: train 2,164 pairs (25,968 rows), validation 422 (5,064 rows), test 1,066 (12,792 rows). Each pair has 12 forms: 4 singles (P, not P, Q, not Q) and 8 joints.
- **Teacher.** Qwen3-32B with the v7b_explicit prompt, thinking off, on DashScope, one form per call, temperature 0. No teacher answers are missing in any split, and no model was substituted.
  - Train answers cost $2.247 (new calls in this run).
  - The validation and test answers are reused from prompt-negation-fix-20260929, where they cost $0.08 and $1.12.
  - Every teacher prompt was re-rendered and checked against its logged sha256.
- **Student.** Prompt: the teacher's prompt with the answer line swapped for "Reply with exactly one word: YES ... NO ...", in the Qwen3 chat template with thinking off. P(YES) = sigmoid(logit YES − logit NO) at the first assistant token.
  - Adapter: LoRA r16, α32, dropout 0.05, on all linear layers.
  - Loss: BCE against a soft target.
  - Optimizer: lr 2e-4, effective batch 32. Max length 1,152; 8 train prompts were truncated.
  - Validation: log loss every 200 steps (and at step 0), patience 3, best checkpoint kept.
  - Test: scored once per arm.
- **Arms.** All arms use the same hyperparameters.

  | Arm | Target |
  |---|---|
  | Z | Zero-shot (no training) |
  | R | Teacher raw |
  | C | Teacher kl-projected |
  | O | Outcome labels |
  | CO | 0.5 · projected + 0.5 · label |

- **Scoring.** Log loss is clipped at [0.01, 0.99]. Metrics are reported over all 12 forms, the 4 singles and the 8 joints, plus Brier, ECE (10 bins), accuracy and coherence. Coherence is measured two ways:
  - the correlation of p(A) with p(not A), where −1 is perfectly coherent;
  - the mean `relation_penalties` total.

  Every metric is repeated after kl-projecting each student's own 12 answers. Confidence intervals are paired bootstraps over pairs, 2,000 resamples.
- **Excluded.** The one-pass-per-pair mode from commit 26b63f6 was not used, because it is unverified.

## Results (test, 1,066 pairs)

| | LL all 12 | singles | joints | LL after projection | Brier | ECE | corr | penalty |
|---|---|---|---|---|---|---|---|---|
| teacher raw | 0.5943 | 0.6696 | 0.5566 | 0.5256 | 0.1979 | 0.062 | −0.611 | 0.0177 |
| teacher projected | 0.5256 | 0.5965 | 0.4902 | 0.5256 | 0.1754 | 0.008 | −1.000 | 0 |
| base-rate table | 0.5700 | 0.6504 | 0.5298 | 0.5700 | 0.1927 | 0.067 | −1.000 | 0 |
| 0.6B Z | 1.2599 | 1.4213 | 1.1792 | 0.6711 | 0.3958 | 0.342 | +0.326 | 0.0974 |
| 0.6B R | 0.5680 | 0.6478 | 0.5281 | 0.5562 | 0.1919 | 0.041 | −0.721 | 0.0019 |
| **0.6B C** | **0.5552** | 0.6338 | 0.5159 | 0.5533 | 0.1870 | 0.025 | −0.911 | 0.0004 |
| 0.6B O | 0.5973 | 0.6881 | 0.5518 | 0.5819 | 0.2054 | 0.038 | −0.807 | 0.0038 |
| 0.6B CO | 0.6096 | 0.7078 | 0.5605 | 0.6015 | 0.2106 | 0.043 | −0.923 | 0.0018 |
| 1.7B Z | 1.7931 | 1.8027 | 1.7883 | 0.6145 | 0.4368 | 0.414 | +0.423 | 0.1923 |

Paired bootstrap, Qwen3-0.6B, difference in log loss over all 12 forms (negative means the first arm is better):

| Contrast | Raw LL | LL after projection | corr |
|---|---|---|---|
| **C − R (key)** | **−0.0128 [−0.0166, −0.0089]** | −0.0029 [−0.0067, +0.0010] | −0.189 [−0.211, −0.167] |
| C − O | −0.0420 [−0.0504, −0.0332] | −0.0286 [−0.0369, −0.0199] | −0.104 |
| CO − O | +0.0124 [+0.0040, +0.0212] | +0.0195 [+0.0113, +0.0282] | −0.115 |
| C − Z | −0.7047 [−0.7300, −0.6813] | −0.1178 [−0.1318, −0.1040] | −1.236 |
| C − teacher raw | −0.0390 [−0.0526, −0.0260] | +0.0277 [+0.0163, +0.0386] | −0.299 |
| C − base rate | −0.0148 [−0.0205, −0.0089] | −0.0167 [−0.0225, −0.0109] | +0.089 |

The key test holds for singles as well: C − R = −0.0139 [−0.0212, −0.0066].

## Reading

- **Projected targets beat raw targets.** At 0.6B, projected targets are the better training signal: better log loss, Brier and ECE, and a clearly more coherent student (corr −0.91 vs −0.72).
- **The gain is mostly coherence.** Most of the improvement is coherence that projecting the student's own answers would also give. After projection, C − R is not significant.
- **The student is close to base rate.** It learns little about the questions beyond the per-form base rate: C beats the base-rate table by only 0.015 nats. Training on outcomes alone (O) is worse than the base-rate table. The teacher's projected answers carry information that a 0.6B student does not absorb from 2,164 training pairs.
- **CO is worse than O.** This is unexpected, and the CI excludes zero. One possible cause is that mixing the two targets makes them noisier. This was not investigated.

## What went wrong and what is missing

- **Qwen3-1.7B is incomplete.** Only Z is scored.
  - The C and R runs finished on spot VM cdistill-b, but they were not synced before GCE preempted it at 09:31:27Z (termination action DELETE). The runs in progress (O and CO) were lost with it.
  - The earlier O and CO runs were lost to the preemption of cdistill-c at 08:31:03Z.
  - The remote logs, read before the preemption, showed C test all-12 LL 0.5479 and O 0.5945. Their predictions are gone, so these numbers are not in the tables and have no CIs.
  - Lesson: sync every arm the moment it finishes, not per VM.
- **1.7B ran 1 epoch, 0.6B ran 2.** The cut was needed to fit the $35 cap (ledger NOTE at 06:59Z). 1.7B also needed gradient checkpointing, because it ran out of memory on the 40 GB A100 without it.
- **L4 test.** A g2-standard-8 (L4) was tried for 1.7B. It was deleted after 22 minutes: its step-0 evaluation took more than 12 min vs about 1.8 min on an A100.
- **Qwen3-4B was not run** (budget).

## Cost and cleanup

- **DashScope:** $2.247 of the $6 cap (train teacher answers).
- **GCP:** $30.54 of the $35 cap. This is the ceiling estimate from `gpu/LEDGER.md` (list rates of A100 $3.69/h and L4 $0.86/h, times each VM's lifetime). Four spot VMs were used: cdistill-a, cdistill-b, cdistill-c and cdistill-l1.
- **Cleanup:**
  - All four VMs are deleted: two by the launcher after syncing, two by spot preemption.
  - Checked at 09:43Z with `gcloud --project=GCP_PROJECT compute instances list --filter="labels.cdistill:*"`, which listed 0 instances. The `disks list --filter="name~cdistill"` check also listed 0 items.
  - Synced adapters and predictions (0.6B all arms, 1.7B Z) passed sha256 verification (46 files). They are in `/home/agent/data/consistency-distill-20260929/gpu/cdistill-a/`.
