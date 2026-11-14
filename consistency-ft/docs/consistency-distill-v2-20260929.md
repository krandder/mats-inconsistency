# Consistency distillation v2: bigger student and teacher, 2026-09-29

**Question.** In [v1](consistency-distill-20260929.md), a Qwen3-0.6B student trained on kl-projected (consistent) Qwen3-32B answers (C) beat one trained on the raw answers (R) by 0.013 nats. Does C still beat R with a Qwen3-1.7B student? Does the edge grow with student size? Does a stronger teacher, Qwen3-235B-A22B, help?

**Answer. Status: Informative.**

- **C beats R at 1.7B with both teachers**, by −0.0245 [−0.0283, −0.0204] on 32B targets and −0.0058 [−0.0091, −0.0025] on 235B targets. The 32B figure is inflated by a weak R run (see What went wrong).
- **The edge does not reliably grow with student size.** On 32B targets it grows by −0.0117 [−0.0173, −0.0064], but that rests on the weak 1.7B R run. On 235B targets it shrinks by +0.0060 [+0.0016, +0.0104].
- **The 235B teacher helps.**
  - Raw teacher log loss falls from 0.5943 to 0.5474 (−0.0469 [−0.0572, −0.0361]); projected, from 0.5256 to 0.5129 (−0.0128 [−0.0211, −0.0038]).
  - The C students gain −0.0120 [−0.0155, −0.0085] at 0.6B and −0.0111 [−0.0150, −0.0073] at 1.7B.
- **The gain is still mostly coherence.** Once each student's own answers are projected, C and R on 235B targets are the same at both sizes: 0.6B −0.0017 [−0.0044, +0.0010], 1.7B +0.0019 [−0.0013, +0.0050].
- **The best single-call student is 1.7B C on 235B targets, at 0.5360.**
  - It beats the raw 235B teacher by −0.0114 [−0.0210, −0.0019] and the base-rate table (0.5700).
  - It stays 0.023 behind the projected 235B teacher (0.5129).

Numbers: `docs/consistency-distill-v2-20260929.json` (full report), `docs/consistency-distill-v2-20260929-chart.json` (chart data). Code: `scripts/distill_data.py --teacher 235b`, `scripts/distill_student.py`, `scripts/distill2_analyze.py`, `scripts/gpu/distill2_gpu.sh` (with `distill_gpu.sh`, `distill2_remote.sh`, `distill_startup.sh`). Ledger: `/home/agent/data/consistency-distill-20260929/gpu/LEDGER.md` (cdistill-v2 lines). Synced adapters, predictions and events: `/home/agent/data/consistency-distill-v2-20260929/gpu/{a,b,c,d,e}/current/`.

## Design

Same pairs, prompts, student recipe and scoring as [v1](consistency-distill-20260929.md); only the teacher and the student size change.

- **Pairs.** jev-disjoint-v1-filtered-20260919: train 2,164 pairs, validation 422, test 1,066; 12 forms per pair (4 singles, 8 joints).
- **Teachers.**
  - Qwen3-32B: the v1 answers, reused.
  - Qwen3-235B-A22B (new): the same v7b_explicit prompt, thinking off, temperature 0, one form per call on DashScope, for all three splits.
    - 43,824 calls, 0 missing, 0 retried; the served model was qwen3-235b-a22b on every call (no substitution).
    - Cost $17.746 of the $18 cap: validation $3.023, train $9.833, test $4.891.
  - Projected (C) targets: kl projection of each pair's 12 answers, as in v1.
- **Students.** Same as v1: LoRA r16, α32, dropout 0.05 on all linear layers; BCE on the soft target; lr 2e-4, effective batch 32; 2 epochs (1,622 steps); seed 20260929; validation log loss every 200 steps, patience 3, best adapter kept; test scored once per arm.
  - Changes from v1: Qwen3-1.7B now trains 2 epochs (v1: 1); all v2 arms use gradient checkpointing (same math, less memory).
- **Arms.** New in v2: 1.7B C and R on 32B targets; 1.7B C and R on 235B targets; 0.6B C and R on 235B targets. Reused from v1: 0.6B Z, C and R on 32B targets; 1.7B Z.
- **Hardware.** Spot VMs on Google Cloud. 1.7B on 32B targets: one A100 40GB (a2-highgpu-1g), both arms side by side. The four 235B-target arms: one L4 (g2-standard-8) each. v1's 0.6B arms ran on an A100. GPU kernels are not bit-deterministic, so repeated runs differ slightly (about 0.0015 in validation log loss in the resume test).
- **Checkpointing and resume.** Each arm saves LoRA weights, optimizer, scheduler, step and RNG state every 5 min (10 min on the A100) and after every evaluation. A supervisor per VM group syncs results to General every 4 min, checks sha256, re-creates a preempted VM in the next zone and resumes from the last verified sync. VMs self-delete after 6 h and are deleted after the final verified sync.
- **Scoring.** As v1: log loss clipped to [0.01, 0.99] over all 12 forms, the 4 singles and the 8 joints; Brier, ECE (10 bins), coherence (correlation of p(A) with p(not A), −1 is coherent); every metric again after kl-projecting each student's own 12 answers. Paired bootstrap over pairs, 2,000 resamples.

## Results (test, 1,066 pairs)

Log loss (lower is better); "after projection" kl-projects each model's own 12 answers per pair. corr is the correlation of p(A) with p(not A) (−1 is coherent). 0.6B Z, 0.6B C and R on 32B targets, and 1.7B Z are from v1.

| | LL all 12 | singles | joints | LL after projection | proj. singles | proj. joints | Brier | ECE | corr | best step (val LL) |
|---|---|---|---|---|---|---|---|---|---|---|
| 32B teacher raw | 0.5943 | 0.6696 | 0.5566 | 0.5256 | 0.5965 | 0.4902 | 0.1979 | 0.062 | −0.611 | – |
| 32B teacher projected | 0.5256 | 0.5965 | 0.4902 | 0.5256 | 0.5965 | 0.4902 | 0.1754 | 0.008 | −1.000 | – |
| 235B teacher raw | 0.5474 | 0.5949 | 0.5237 | 0.5129 | 0.5775 | 0.4806 | 0.1823 | 0.049 | −0.781 | – |
| 235B teacher projected | 0.5129 | 0.5775 | 0.4806 | 0.5129 | 0.5775 | 0.4806 | 0.1702 | 0.014 | −1.000 | – |
| base-rate table | 0.5700 | 0.6504 | 0.5298 | 0.5700 | 0.6504 | 0.5298 | 0.1927 | 0.067 | −1.000 | – |
| 0.6B Z | 1.2599 | 1.4213 | 1.1792 | 0.6711 | 0.7775 | 0.6180 | 0.3958 | 0.342 | +0.325 | 0 (1.1269) |
| 0.6B C @235B | 0.5432 | 0.6164 | 0.5066 | 0.5419 | 0.6160 | 0.5049 | 0.1820 | 0.020 | −0.926 | 1200 (0.5858) |
| 0.6B R @235B | 0.5550 | 0.6215 | 0.5217 | 0.5436 | 0.6184 | 0.5062 | 0.1865 | 0.034 | −0.737 | 1000 (0.5970), stopped at 1600/1622 |
| 0.6B C @32B | 0.5552 | 0.6338 | 0.5159 | 0.5533 | 0.6294 | 0.5152 | 0.1870 | 0.025 | −0.910 | 1200 (0.5830) |
| 0.6B R @32B | 0.5680 | 0.6478 | 0.5281 | 0.5562 | 0.6338 | 0.5173 | 0.1919 | 0.041 | −0.721 | 800 (0.5936), stopped at 1400/1622 |
| 1.7B Z | 1.7931 | 1.8027 | 1.7883 | 0.6145 | 0.7032 | 0.5701 | 0.4368 | 0.414 | +0.422 | 0 (1.6248) |
| 1.7B C @235B | 0.5360 | 0.6070 | 0.5005 | 0.5347 | 0.6071 | 0.4985 | 0.1791 | 0.019 | −0.946 | 1200 (0.5864) |
| 1.7B R @235B | 0.5418 | 0.6125 | 0.5065 | 0.5328 | 0.6048 | 0.4968 | 0.1815 | 0.021 | −0.885 | 800 (0.5972), stopped at 1400/1622 |
| 1.7B C @32B | 0.5471 | 0.6215 | 0.5099 | 0.5456 | 0.6198 | 0.5086 | 0.1836 | 0.021 | −0.923 | 1200 (0.5871) |
| 1.7B R @32B | 0.5716 | 0.6433 | 0.5358 | 0.5592 | 0.6374 | 0.5201 | 0.1934 | 0.045 | −0.646 | 200 (0.5914), stopped at 800/1622 |

Key paired bootstrap contrasts (difference in log loss over all 12 forms; negative means the first arm is better; 2,000 resamples):

| Contrast | Raw LL | LL after projection |
|---|---|---|
| **1.7B C − R, 32B targets** | **−0.0245 [−0.0283, −0.0204]** | −0.0136 [−0.0174, −0.0094] |
| **1.7B C − R, 235B targets** | **−0.0058 [−0.0091, −0.0025]** | +0.0019 [−0.0013, +0.0050] |
| 0.6B C − R, 32B targets (v1) | −0.0128 [−0.0166, −0.0089] | −0.0029 [−0.0067, +0.0010] |
| 0.6B C − R, 235B targets | −0.0118 [−0.0144, −0.0091] | −0.0017 [−0.0044, +0.0010] |
| **(C − R) at 1.7B minus at 0.6B, 32B targets** | **−0.0117 [−0.0173, −0.0064]** | −0.0107 [−0.0160, −0.0055] |
| **(C − R) at 1.7B minus at 0.6B, 235B targets** | **+0.0060 [+0.0016, +0.0104]** | +0.0035 [−0.0007, +0.0079] |
| **235B − 32B teacher, raw** | **−0.0469 [−0.0572, −0.0361]** | −0.0128 [−0.0211, −0.0038] |
| 1.7B C: 235B − 32B targets | −0.0111 [−0.0150, −0.0073] | −0.0110 [−0.0149, −0.0072] |
| 0.6B C: 235B − 32B targets | −0.0120 [−0.0155, −0.0085] | −0.0114 [−0.0149, −0.0079] |
| 1.7B C on 235B − 235B teacher raw | −0.0114 [−0.0210, −0.0019] | +0.0218 [+0.0130, +0.0303] |

All contrasts, with singles and joints:

| Contrast | Raw LL all 12 | singles | joints | LL after projection |
|---|---|---|---|---|
| Qwen3-0.6B C-R @32b | −0.0128 [−0.0166, −0.0089] | −0.0139 [−0.0212, −0.0066] | −0.0122 [−0.0155, −0.0090] | −0.0029 [−0.0067, +0.0010] |
| Qwen3-0.6B C-Z @32b | −0.7047 [−0.7300, −0.6813] | −0.7875 [−0.8215, −0.7552] | −0.6633 [−0.6870, −0.6401] | −0.1178 [−0.1318, −0.1040] |
| Qwen3-0.6B R-Z @32b | −0.6919 [−0.7159, −0.6691] | −0.7736 [−0.8056, −0.7426] | −0.6511 [−0.6740, −0.6287] | −0.1150 [−0.1275, −0.1024] |
| Qwen3-0.6B C@32b - teacher 32b raw | −0.0390 [−0.0526, −0.0260] | −0.0358 [−0.0603, −0.0130] | −0.0407 [−0.0525, −0.0294] | +0.0277 [+0.0163, +0.0386] |
| Qwen3-0.6B R@32b - teacher 32b raw | −0.0263 [−0.0409, −0.0131] | −0.0218 [−0.0483, +0.0017] | −0.0285 [−0.0402, −0.0172] | +0.0305 [+0.0188, +0.0417] |
| Qwen3-0.6B C-R @235b | −0.0118 [−0.0144, −0.0091] | −0.0051 [−0.0096, −0.0006] | −0.0151 [−0.0181, −0.0123] | −0.0017 [−0.0044, +0.0010] |
| Qwen3-0.6B C-Z @235b | −0.7167 [−0.7423, −0.6925] | −0.8049 [−0.8407, −0.7713] | −0.6726 [−0.6957, −0.6499] | −0.1292 [−0.1438, −0.1148] |
| Qwen3-0.6B R-Z @235b | −0.7049 [−0.7305, −0.6805] | −0.7998 [−0.8340, −0.7676] | −0.6575 [−0.6816, −0.6346] | −0.1276 [−0.1414, −0.1136] |
| Qwen3-0.6B C@235b - teacher 235b raw | −0.0042 [−0.0142, +0.0058] | +0.0215 [+0.0063, +0.0373] | −0.0171 [−0.0268, −0.0078] | +0.0290 [+0.0196, +0.0382] |
| Qwen3-0.6B R@235b - teacher 235b raw | +0.0076 [−0.0023, +0.0172] | +0.0266 [+0.0119, +0.0417] | −0.0019 [−0.0109, +0.0067] | +0.0307 [+0.0215, +0.0395] |
| Qwen3-0.6B C: 235b-32b | −0.0120 [−0.0155, −0.0085] | −0.0174 [−0.0222, −0.0126] | −0.0093 [−0.0126, −0.0061] | −0.0114 [−0.0149, −0.0079] |
| Qwen3-0.6B R: 235b-32b | −0.0130 [−0.0171, −0.0089] | −0.0263 [−0.0328, −0.0194] | −0.0064 [−0.0098, −0.0029] | −0.0126 [−0.0165, −0.0087] |
| Qwen3-1.7B C-R @32b | −0.0245 [−0.0283, −0.0204] | −0.0217 [−0.0270, −0.0163] | −0.0258 [−0.0300, −0.0216] | −0.0136 [−0.0174, −0.0094] |
| Qwen3-1.7B C-Z @32b | −1.2460 [−1.2726, −1.2196] | −1.1811 [−1.2228, −1.1380] | −1.2784 [−1.3072, −1.2490] | −0.0688 [−0.0783, −0.0594] |
| Qwen3-1.7B R-Z @32b | −1.2215 [−1.2469, −1.1958] | −1.1594 [−1.1989, −1.1173] | −1.2525 [−1.2815, −1.2238] | −0.0553 [−0.0631, −0.0479] |
| Qwen3-1.7B C@32b - teacher 32b raw | −0.0471 [−0.0607, −0.0346] | −0.0480 [−0.0719, −0.0269] | −0.0467 [−0.0582, −0.0354] | +0.0200 [+0.0089, +0.0303] |
| Qwen3-1.7B R@32b - teacher 32b raw | −0.0227 [−0.0364, −0.0099] | −0.0263 [−0.0509, −0.0046] | −0.0209 [−0.0321, −0.0091] | +0.0336 [+0.0223, +0.0441] |
| Qwen3-1.7B C-R @235b | −0.0058 [−0.0091, −0.0025] | −0.0055 [−0.0116, +0.0005] | −0.0059 [−0.0100, −0.0019] | +0.0019 [−0.0013, +0.0050] |
| Qwen3-1.7B C-Z @235b | −1.2571 [−1.2843, −1.2305] | −1.1957 [−1.2372, −1.1514] | −1.2878 [−1.3165, −1.2586] | −0.0798 [−0.0902, −0.0696] |
| Qwen3-1.7B R-Z @235b | −1.2513 [−1.2789, −1.2241] | −1.1901 [−1.2311, −1.1453] | −1.2818 [−1.3111, −1.2523] | −0.0817 [−0.0937, −0.0697] |
| Qwen3-1.7B C@235b - teacher 235b raw | −0.0114 [−0.0210, −0.0019] | +0.0121 [−0.0024, +0.0268] | −0.0232 [−0.0320, −0.0141] | +0.0218 [+0.0130, +0.0303] |
| Qwen3-1.7B R@235b - teacher 235b raw | −0.0056 [−0.0148, +0.0038] | +0.0176 [+0.0030, +0.0326] | −0.0172 [−0.0253, −0.0088] | +0.0199 [+0.0114, +0.0284] |
| Qwen3-1.7B C: 235b-32b | −0.0111 [−0.0150, −0.0073] | −0.0145 [−0.0197, −0.0095] | −0.0094 [−0.0130, −0.0059] | −0.0110 [−0.0149, −0.0072] |
| Qwen3-1.7B R: 235b-32b | −0.0298 [−0.0362, −0.0232] | −0.0307 [−0.0394, −0.0222] | −0.0293 [−0.0353, −0.0230] | −0.0264 [−0.0326, −0.0199] |
| (C-R) 1.7B minus (C-R) 0.6B @32b | −0.0117 [−0.0173, −0.0064] | −0.0078 [−0.0156, +0.0001] | −0.0137 [−0.0187, −0.0089] | −0.0107 [−0.0160, −0.0055] |
| C@32b: 1.7B-0.6B | −0.0081 [−0.0130, −0.0034] | −0.0123 [−0.0186, −0.0060] | −0.0060 [−0.0106, −0.0016] | −0.0077 [−0.0125, −0.0029] |
| R@32b: 1.7B-0.6B | +0.0036 [−0.0008, +0.0080] | −0.0045 [−0.0121, +0.0031] | +0.0077 [+0.0044, +0.0110] | +0.0031 [−0.0012, +0.0072] |
| (C-R) 1.7B minus (C-R) 0.6B @235b | +0.0060 [+0.0016, +0.0104] | −0.0005 [−0.0068, +0.0065] | +0.0092 [+0.0047, +0.0136] | +0.0035 [−0.0007, +0.0079] |
| C@235b: 1.7B-0.6B | −0.0072 [−0.0116, −0.0028] | −0.0094 [−0.0152, −0.0036] | −0.0061 [−0.0100, −0.0021] | −0.0072 [−0.0116, −0.0028] |
| R@235b: 1.7B-0.6B | −0.0132 [−0.0183, −0.0081] | −0.0090 [−0.0176, −0.0009] | −0.0153 [−0.0197, −0.0110] | −0.0107 [−0.0159, −0.0057] |
| teacher 235b-32b raw | −0.0469 [−0.0572, −0.0361] | −0.0747 [−0.0932, −0.0556] | −0.0330 [−0.0429, −0.0229] | −0.0128 [−0.0211, −0.0038] |
| teacher 235b-32b projected | −0.0128 [−0.0211, −0.0038] | −0.0190 [−0.0292, −0.0089] | −0.0096 [−0.0175, −0.0012] | −0.0128 [−0.0211, −0.0038] |

## Reading

- **C beats R at 1.7B, but by how much depends on the R run.**
  - On 235B targets, where both 1.7B arms trained normally, the edge is −0.006: half the 0.6B edge.
  - After projecting each student's own answers, it is +0.002 (not significant).
- **No evidence that the edge grows with student size.** The one contrast that says it grows (32B targets, −0.0117) compares against a 1.7B R arm that stopped at step 800 with its best adapter at step 200. That arm is no better than the 0.6B R arm (+0.0036 [−0.0008, +0.0080]). Every other arm improves with size: C by −0.0081 (32B) and −0.0072 (235B), R by −0.0132 (235B).
- **The stronger teacher helps as much as consistent targets do, or more.** Switching teachers moves each student about as much as C vs R does at 0.6B, and twice as much at 1.7B (−0.011 to −0.013 for C at both sizes and for R at 0.6B). It also makes every student better calibrated and more coherent.
- **Students now match the raw teacher, not beat it by much.**
  - On 32B targets, every trained student beat its raw teacher by 0.023 to 0.047.
  - On 235B targets, 0.6B C and 0.6B R and 1.7B R are within the CI of the raw teacher; only 1.7B C beats it (−0.011).
  - On singles the 235B students are worse than the raw 235B teacher (0.6B C +0.0215 [+0.0063, +0.0373]).
  - The projected 235B teacher (0.5129) stays well ahead of every student.
- **Projection at inference still beats distillation.** The best student after projecting its own answers (1.7B R on 235B targets, 0.5328) is 0.020 behind the projected teacher.

## Checkpointing, preemptions and the resume test

- **Preemptions survived: 12, on 21 spot VMs, with every arm finished.**
  - 6 hit running VMs: a1, c1, t6, b1, c2, c4.
  - 5 hit VMs during startup: t3, t8, t9, b2, c3.
  - 1 was deliberate: t7, the resume test.
  - Resumed runs restarted from their last sha256-verified sync: b3 from step 374, c4 from step 463, c5 from step 800.
  - a1 was preempted at 17:39Z, 25 min after creation, before its supervisor had adopted it. a4 therefore started group a from step 0.
- **Resume test: passed** (ledger NOTE 19:40Z).
  - The arm: a 0.6B C arm on 32B targets, 150 pairs, run twice ("kill" and an uninterrupted "ref").
  - "kill" was preempted on purpose at step 40 (simulate-maintenance-event on t7, 19:00:16Z), after its step-40 checkpoint had synced.
  - After two more startup preemptions (t8, t9), t10 restored step-000040 (sha256 ok) and resumed at step 40. The run finished at step 112.
  - Its evaluations after the resume are within 0.0015 of ref's. The gap between kill and ref was the same size before any kill, so it is GPU nondeterminism.
  - ref itself resumed at step 40 after the real t6 preemption.

## What went wrong and what is missing

- **1.7B R on 32B targets stopped early.**
  - It stopped at step 800, with its best at step 200. Its validation log loss was noisy: 0.5914, 0.6069, 0.6153, 0.5964 at steps 200 to 800.
  - The same patience-3 rule applied to every arm; it also stopped 1.7B R on 235B targets at step 1400 and the 0.6B R arms at 1400 and 1600.
  - This run probably inflates the 1.7B C − R contrast on 32B targets and the "edge grows with size" contrast on 32B targets.
  - It was not rerun: one test evaluation per arm.
- **Mixed hardware.**
  - 1.7B on 32B targets ran two arms per A100; the four 235B-target arms ran on one L4 each.
  - v1's 0.6B arms ran on an A100 without gradient checkpointing.
  - GPU nondeterminism alone moves validation log loss by about 0.0015.
- **One seed per arm.** The CIs cover test-pair sampling, not training noise.
- **The budget changed mid-run.**
  - At 19:24Z (Kelvin's 24 h experiment budget, relayed 19:17Z), spend began to be counted at the Spot price per region (Cloud Billing Catalog API), with a cap of $25.
  - Before that it was counted at the on-demand list ceiling, with a cap of $30.
  - Ledger `spend_now` values before 19:24Z (and on the group-t supervisor's lines) are at the ceiling.
- **The resume-test group hit its limit.** Group t reached its 8-VM limit after startup preemptions; the limit was raised to 12.
- **Qwen3-4B was not run.**

## Cost and cleanup

- **DashScope:** $17.746 of the $18 cap (235B answers, all splits).
- **GCP:** $19.21 at Spot prices of the $25 cap, or $30.93 at the on-demand list ceiling.
  - Spot prices come from the Cloud Billing Catalog, effective 2026-09-29 07:00Z, plus $0.03/h for disk and IP. Examples: L4 g2-standard-8 $0.51–0.58/h; A100 a2-highgpu-1g $2.20–2.36/h.
  - Price evidence: `/home/agent/data/consistency-distill-v2-20260929/gpu/spot-prices-20260929.json`.
- **Cleanup: all 21 VMs are deleted.**
  - The VMs were a1, a4, b1–b3, c1–c5, d1, e1, t1 and t3–t10. Each was deleted by its supervisor after the final sha256-verified sync, or by spot preemption.
  - Checked at 2026-09-30 00:43:12Z: `gcloud --project=GCP_PROJECT compute instances list --filter='name~^cdistill'` listed 0 instances, and `compute disks list --filter='name~^cdistill'` listed 0 disks.
  - The watchdog is stopped. The ledger CLOSED line is at 00:43:55Z.
