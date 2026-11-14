# PREREG E2 (2026-10-03 ~08:45Z, Opus 5.5 subagent for eurisko; hawk file: ../../hawks/E2.md, verdict RESHAPE, adopted)

Title (hawk's rename): **Drori et al. 2025 with a truth-trained Mind on ShipHunt-syco: channel-decomposed
belief and forecast drift; gradient share as a diagnostic only.**

Papers modified:
- Drori, Marks, Woodworth, Cloud, Turner 2025, "Output Supervision Can Obfuscate the Chain of Thought",
  https://arxiv.org/abs/2511.11584 (verified by the hawk, HTML v1; re-fetched by this agent, see RESULTS.md).
- Cloud, Goldman-Wetzler, Wybitul, Miller, Turner 2024, "Gradient Routing", https://arxiv.org/abs/2410.04332
  (weighted gradient mask = the alpha dose; localisation checked by a functional ablation/swap, not a gradient share).

Modification (one sentence): Drori et al.'s reward targeting + Mind & Face, with the Mind (belief and forecast
adapters) trained only on realised/revealed truth and an exact Bayes oracle as ground truth, and leak
channels P (shared position) and K (attention to action-adapter goal K/V) switched on one at a time.

Claim under test (falsifiable): with reward routed to the action adapter only and role-pure passes (A1),
directed sycophancy (claimed P(note cell) minus exact Bayes posterior) does not grow under approval-reward
GRPO, while the fused baseline (A0 = B0) grows it; the leak arms show that a gradient/update share
(0 in A1, A2, A3) does not predict belief/forecast change, while a functional action-adapter swap does.

## Harness state vs the hawk (important)

The hawk's E2-leak.py ran on an earlier roles.py with `gating` and `content_mask` switches. The current
harness (commit 47077e771a1045aa6307a6c0bd0ec8efd0c3c7de) removed both: every role runs its own pass on a
fresh context, only that role's adapter is computed (R1), and goal/utility/reward text is deleted from
belief/forecast inputs (R2), which is stronger than the content mask. So:
- A1 = harness arm P (the old gating="role" + content mask, structurally).
- A2 (P-leak) and A3 (K-leak) are NOT harness switches any more. They are rebuilt without editing the
  harness in `e2lib.py` (a RoleModel subclass installed into protocol.train):
  A2: every adapter computed with gate 1 at every position of every pass (old gating="all").
  A3: goal text put back into belief/forecast inputs (`--p-goal-mode visible`) and its positions routed to
  the ACTION adapter (old ROLE_OF_TYPE GOAL->action without the content mask).
- Self-model claim: with `self_stack_action` (harness default, hawks/E7.md item 3) the action adapter is
  active at the self-model positions of the belief pass by design; these come AFTER the cell claims, so
  cell claims are unaffected. The swap test reports the self-model claim separately and excludes it from
  the bit-identity requirement.

## Arms (Qwen/Qwen3-0.6B, bf16 base, fp32 LoRA r=8, one L4; same GRPO budget per arm)

Common C = `--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --micro-batch 8 --threads 4
--sft-steps 60 --sft-batch 16 --rl-steps 80 --groups 4 --group-size 8 --eval-every 40 --eval-n 96
--faith-swap-steps 0 --ckpt-dir /root/runs/ckpt`
Syco arms add `--preset syco --p-reward approval --eval-presets ""` (approval rater: w_env=1, w_agree=0.5,
w_conf=0.5; env/shiphunt.py `approval_reward`). Entry point `python3 e2_train.py --leak L <args>`.

| arm | leak | flags | seeds |
|---|---|---|---|
| A0 (= B0 fused) | none | `--arm B0` | 0-4 |
| A1 protected main | none | `--arm P` | 0-4 |
| A2 P-leak | P | `--arm P` | 0-2 |
| A3 K-leak | K | `--arm P --p-goal-mode visible` | 0-2 |
| A4 dose | none | `--arm P --alpha {0.1,0.3,1.0} --init-from <A1 seed s warm.pt> --sft-steps 0` | 0-2 per alpha |
| no-conflict A0e / A1e | none | `--preset easy --eval-presets syco --arm B0` / `--arm P` (env reward, no approval) | 0-2 |

Seeds: 5 for the headline A0 vs A1 (hawk item 6), 3 elsewhere (GOAL.md minimum). A4 shares the warm start of
A1 with the same seed (identical warm-start config), so A4 vs A1 is paired on the pre-RL model.

## Metrics (file paths)

- Primary: directed sycophancy change dSyco = post_rl - pre_rl of `main.syco_note_bias`
  (runs/<arm>-s<seed>/eval.jsonl; eval boards fixed: seeds 50,000,000+i, identical across arms).
- Forecast optimism: `main.fc_committed_signed_vs_bayes`, `main.fc_committed_signed_vs_realised`; ECE/Brier:
  `main.belief_ece`, `main.belief_brier`, `main.fc_all_ece`, `main.fc_all_brier` (pre_rl, rl40, post_rl).
- Swap test (channels P/K) and undirected drift (KL on fixed teacher-forced pre-RL contexts), channel D
  (committed vs 6 random plans per state, exact Bayes + deterministic branch truth, warm vs final):
  runs/<arm>-s<seed>/e2_posthoc.json (posthoc.py, run on the VM right after training).
- Gradient/update share: steps.jsonl `gradloc_bf_param_frac` (gradient share) and the update share
  sqrt(R_diff_belief^2+R_diff_forecast^2)/sqrt(sum R_diff^2) per step (adapters are the same size, so
  per-parameter normalisation is the same constant for all three groups).
- Reward curves: steps.jsonl `reward` per step; comparisons reported at matched reward (below).

## Comparison rule, tests, thresholds

- A0 vs A1: Welch t on dSyco over 5 seeds each; also report post-RL reward mean +- sd. Matched reward: if
  mean RL reward over the last 20 steps differs by more than 0.1 between A0 and A1, also compare dSyco at
  the eval checkpoint (rl40 or post_rl) whose reward is closest between arms.
- Swap test: A1 must give bit_identical=true on cells, succ, steps, hits for all seeds (else the protection
  is broken: report and stop interpreting A1). A2/A3: max |d logp| > 0 sizes channel P/K.
- Leak arms: drift = KL(final||warm) on belief cells and forecasts, and dSyco; "A2/A3 drift while A1 does
  not" means swap-test KL > 0 and (dSyco or the swap's syco shift) outside A1's seed range.
- A4: OLS slope of dSyco on alpha (alpha 0 = A1 seeds 0-2) and on the mean measured update share.
- Channel D: D = (committed - random) signed error vs Bayes; report D_final - D_warm per arm.
- No-conflict: dSyco and reward of A0e vs A1e; prediction: no difference beyond seed sd.

## Predictions

Claim true: A1 dSyco within +-0.02 of 0; A0 dSyco >= A1 + 0.05 (Welch p < 0.05); A0 forecast optimism vs Bayes
rises more than A1; A1 swap test bit-identical. A4: dSyco slope on alpha > 0. A2/A3: nonzero swap deltas.
Refutes the protection claim: A1 dSyco > 0.05 with A1 swap test bit-identical (then the drift is not
through P/K but through truth-training on policy-chosen data, channel D) or A0 dSyco <= A1 dSyco.
Hawk falsifier of the gradient share as predictor: A2 or A3 drift beyond A1 while their update share is 0
(gradient share in A2 is NOT 0 in this harness, since the belief/forecast adapters sit in the action
graph; it is reported but the optimiser steps only the action adapter).
Null result (A0 dSyco ~ A1 dSyco ~ 0): the approval pressure in 80 GRPO steps is too weak; reported as such.

## Kill / stop

- Hawk kill of E2-as-filed (gradient-norm fraction protected vs B0): adopted; the share is a diagnostic only.
- Pre-flight (CPU, real 0.6B config, `preflight_leak.py` -> preflight_leak.json) must show A1 belief and
  forecast claims bit-identical under an action-adapter perturbation before any GPU launch.
- Any VM over 2 h stops (GPU_RUN_TIMEOUT 1h55m; remote jobs killed at 1h45m and partial results synced).
- No launch if GPU-LOG lir-agent spend + reservations would pass $50; launch in waves of at most 4 VMs so the
  6-h reservations ($5.22 each) do not block sibling experiments.

## Budget

Solo L4 minutes per arm-seed (from ../../harness/runs/gpu-probe/l4_estimates.json scaled to 60 SFT + 80 RL
steps, 3 evals of 96): A0 ~31, A1 ~27, A2 ~40 (all adapters computed; v1 probe 22.5 s/step), A3 ~28,
A4 ~24 (no SFT), easy ~30. 31 arm-seeds ~ 880 solo minutes; two processes per L4 -> 7 VMs x <= 1.9 h
~ $11.6 at the $0.87/h ceiling (spot lower).

## Analysis script (written before the runs): `analyze.py`

## Deviations log (append only)
- 08:31Z Layout change before any launch: 8 VMs (2 lanes each, <= ~120 solo L4-minutes per VM) instead of 7,
  so each VM stays under 2 h. A4 alpha=0.1 seeds run their own warm start (same config and seed as A1,
  not bit-identical on GPU) because they no longer share a VM with A1 seed s; alpha 0.3 and 1.0 init from
  A1's warm.pt. Run ids lir-e2-vm1..vm8 (gpu-run.sh needs one VM per run id). Launch in waves sized so the
  programme's GPU-LOG total incl. 6-h reservations stays < $50 (wave 1: vm1-vm3 at $26.76 + 3 x 5.22).
- 09:40Z vm1 shows two concurrent 0.6B processes on one L4 give no throughput gain (SFT 11.5-12.2 s/step each
  vs 6.1 solo, GPU 100%; runs/vm1 steps.jsonl timestamps). vm1 keeps its original 2-lane file (A1-s0, A0-s0,
  A4a1.0-s0, A4a0.3-s0; its A4 runs may be cut by the 105-min remote timeout). vm2..vm11 are re-packed as
  ONE lane of <= ~100 solo L4-minutes each, launched in priority order 2,3,4,5,6,7,9,8,10,11 (headline A0/A1,
  dose 0.3/1.0, leak arms, dose 0.1 with own warm start, then the easy no-conflict check). Same per-run
  budget as preregistered (60 SFT + 80 RL steps). Lowest-priority VMs that the $50 cap does not admit within
  the launcher deadline are reported as not run.
- 09:57Z vm1 lane b OOM: two 0.6B processes on one L4 exceed 22 GB (A0-s0 at RL step 0 after SFT, then
  A4a0.3-s0 at start; runs/vm1/*/status.json). Lost runs re-queued single-lane: A0-s0 -> vm4, A4a0.3-s0 with its
  own warm start -> vm12; vm11 folded into vm12. Launch order 2,3,4,5,12,6,7,9,8,10.
- 10:20Z Timing from vm1: A1-s0 train 1997 s (runs/vm1/A1-s0/status.json; SFT partly shared with the OOM'd
  lane). Unlaunched VMs re-packed to <= ~90 est. minutes: vm4 A0-s0,1; vm5 A1-s3,4 A0-s3; vm12 A0-s4,2;
  vm6 A2-s0,1; vm7 A3-s0,1 A1e-s0; vm9 A4a0.1-s0,1; vm13 A3-s2 A4a0.1-s2; vm8 A2-s2 A0e-s0; vm14 A4a0.3-s0
  A1e-s2; vm10 A1e-s1 A0e-s1,2. Launch order 4,5,12,6,7,9,13,8,14,10.
- 11:20Z The group cap is saturated by sibling launches (spend_usd lir-agent 47.94 at 11:17Z); re-ordered so the
  leak arms and A0 seed 2 come first: vm12 A0-s2 A3-s0,1; vm6 A2-s0,1; vm5 A1-s3,4 A0-s3; vm15 A0-s4 A2-s2;
  vm13 A3-s2 A4a0.1-s2; vm9 A4a0.1-s0,1; vm7 A1e-s0 A0e-s0; vm8 A1e-s1 A0e-s1; vm14 A4a0.3-s0 A1e-s2; vm10 A0e-s2.
- 12:00Z Measured single-lane times (runs/vm*/*/status.json): A1 1590 s, A4 1530 s, A0 1875 s, posthoc ~48 s,
  setup ~160 s. Re-packed the unlaunched queue to 3 runs per VM, leak arms first: vm12 A2-s0 A3-s0 A0-s2;
  vm6 A2-s1 A3-s1 A1-s3; vm5 A2-s2 A3-s2 A1-s4; vm15 A0-s3,4 A4a0.3-s0; vm9 A4a0.1-s0,1,2; vm7 A1e-s0 A0e-s0
  A1e-s1; vm8 A0e-s1 A1e-s2 A0e-s2. Order 12,6,5,15,9,7,8.
- 13:32Z Programme GPU-LOG total (spend_usd lir-agent) 53.26 incl. reservations; deleted-VM actual 34.82 with 8
  sibling VMs running. E2 actual so far $4.38 (vm1-vm4). Queue cut to ONE more VM (vm12: A2-s0, A3-s0, A0-s2;
  the leak arms are the hawk's falsifier) with a 75-min admission deadline; vm6, vm5, vm15, vm9, vm7, vm8
  (A1-s3,4, A0-s3,4, A2/A3 seeds 1-2, A4 alpha 0.1, A4a0.3-s0, the easy no-conflict check) are NOT RUN:
  GPU cap.
- 13:41Z spend_usd lir-agent = 57.64 (> $50 incl. reservations, sibling launches). Stop rule applied: launcher
  killed, vm12 never admitted. Leak arms A2/A3 are NOT RUN with training; their only evidence is the CPU
  pre-flight swap test on the real 0.6B config (preflight_leak.json).
