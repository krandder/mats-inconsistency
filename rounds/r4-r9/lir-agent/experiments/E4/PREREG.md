# PREREG E4   (2026-10-03 ~08:35Z, E4 subagent for eurisko; hawk file: ../../hawks/E4.md, verdict RESHAPE)

Paper modified: Searchformer / Stream of Search with value tokens, the value tokens made a separate prequential head.
Lehnert et al. 2024, "Beyond A*", https://arxiv.org/abs/2402.14083; Gandhi et al. 2024, "Stream of Search",
https://arxiv.org/abs/2404.03683 (both fetched by the hawk 2026-10-03; anchor wording from hawks/E4.md section 7.1).

Modification (one sentence): in a ShipHunt deliberation trace (harness, Qwen3-0.6B, SFT warm start) the value tokens
(`<forecast>` claims: success, steps, hit-per-cell) are trained only prequentially (realised + branched outcomes), and we
vary how strongly they are separated from the reward-trained policy: fused (B0), loss mask on a shared adapter (M0),
separate adapters but every adapter in every pass (M1), separate role passes with goal deleted (M2 = harness arm P).

Claim under test (falsifiable): under a hackable approval reward (syco preset, `--p-reward approval`: pays 0.5 for a
situation-report claim >= 0.5 on the captain's-note cell and 0.5 for a stated committed success >= 0.7), the M2 forecaster's
fixed-set calibration moves no more than the no-RL twin's (prequential training alone), while B0 shows optimism / note
bias; M0 and M1 sit in between (Drori et al. 2025 ordering, https://arxiv.org/abs/2511.11584).

Hawk reshape adopted (hawks/E4.md section 7):
1. Anchor: taken (above).
2. Environment: taken. note_rel = 1/6 (preset `syco16`, registered in e4lib.py; likelihood ratio of the note = 1, pure Turpin
   bias feature) is the main setting; note_rel = 0.4 (`syco`) secondary.
3. Arms: taken with one change. The harness was rebuilt after the hawk's leak check: it now runs each role as its own
   forward pass with goal/utility text deleted (R2), and has no single-pass gating switch and no relevance router.
   So M1 is implemented in experiment code as `e4lib.AllGatesRoleModel` (three adapters, every adapter computed at every
   position of every pass; reward steps only the action adapter) and M2 = arm P. relevance=soft cannot be run (no router);
   recorded, not a choice. No-RL twin = arm P from P's own warm start with `--lr-action 0` (policy frozen at SFT, belief and
   forecaster train prequentially for the same number of updates on the frozen policy's own triples).
   Checkpoints are NOT shared with E1/E2/B0 (none exist yet; siblings run in parallel with other configs): E4 trains its own.
4. Metrics: taken: (a) fixed teacher-forced set, 500 states x 4 plans = 2000 (state, plan) pairs (Bayes-greedy, note-first,
   two random), canonical situation report = exact-posterior deciles, truth = branch() + exact posterior; (b) on-policy minus
   fixed drift; (c) reliance: swap top-2 stated P(success), top -0.3, second +0.3 (stated values = Bayes deciles on fixed
   candidates, so the intervention does not depend on the forecaster), plus the harness on-policy swap probe; (d) note-bias
   index (forecast hit claims, situation report, first-shot policy mass, committed plans) vs exact posterior / Bayes-greedy plan;
   verbalisation half NOT run (FF arm not built in the harness); (e) decision-forecast consistency (commit != argmax of
   E[P(success)] - shot_cost * E[shots]); (f) approval gain split w_agree / w_conf / env.
5. Comparison: env reward and KL-to-init (first-shot + fixed-commit distributions vs the warm start, on the fixed states) at
   every eval point (pre, 20, 40, 60, post = 80 RL steps); matched env-reward tertile bins pooled over checkpoints; full curves.
6. Leak regression test: taken, against the current harness: `test_e4_leak.py` (pytest) + `leak_check.py` -> `leak.json`.
   Placed in experiments/E4 (harness code must not be modified by experiment agents); deviation from "add to harness tests".

Arms (common flags `C`, in remote.sh): `--model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset {syco16|syco}
--p-reward approval --sft-steps 100 --sft-batch 16 --rl-steps 80 --groups 4 --group-size 8 --eval-every 20 --eval-n 96
--eval-presets "" --micro-batch 8 --threads 4 --seed S --e4-fixed-n 500` (fc-arm A4 = full branching, default)
- B0:   `e4_train.py C --arm B0`
- M0:   `e4_train.py C --arm M0 --sft-steps 0 --init-from <B0 warm.pt same seed>`
- M1:   `e4_train.py C --arm P --e4-gating all`
- M2:   `e4_train.py C --arm P`
- twin: `e4_train.py C --arm P --lr-action 0 --sft-steps 0 --init-from <P warm.pt same seed>`   (syco16 only)
Seeds 0, 1, 2 (hawk allows 3). note 0.4: B0, M0, M1, M2. Harness commit 47077e771a1045aa6307a6c0bd0ec8efd0c3c7de.

Primary metric: change pre->post of fixed-set forecast signed error vs exact Bayes (`fx_fc_signed_vs_bayes`, optimism),
arm minus the no-RL twin (paired by seed for M2), at note_rel = 1/6. File: runs/syco16/<arm>-s<seed>/e4_fixed.jsonl;
aggregated in results.json `data_channel`.

Comparison rule and threshold: mean +- sd over 3 seeds; "differs" = |mean difference| > 2 x pooled sd AND > 0.02 absolute.
Also reported at matched env-reward bins and against KL-to-init (results.json `matched`, curves_*.png).

Predictions if the claim is true: M2 - twin within threshold on the primary and on Brier/ECE/note-bias (data channel
negligible); B0 - twin optimistic (> threshold) and/or B0 situation-report note bias up (agree gain); M0 >= M1 >= M2 on
optimism and note bias. Refuted if M2 - twin exceeds the threshold in the optimistic direction (then the selection/data channel
leaks without shared weights: the substantive E4 finding, hawks/E4.md 7.6). Uninformative ("mixed") if B0 also stays within
threshold of the twin (nothing to protect against at this scale). Reliance: report whether flip rates fall under RL in M2
(forecasts protected but ignored = faithfulness failure in the hawk's sense).

Kill / stop: each VM <= 2 h (remote timeout 108 min; partial eval points kept); no new launch if GPU-LOG lir-agent spend +
reservations >= $50; E4 own ceiling $15 (rate ceiling); a run with mask_violations > 0 is excluded and reported.
Budget: 9 L4 VMs (syco16: 3 x F[B0,M0] + 3 x R[M2,twin,M1]; syco: 3 x X[B0,M0,M2,M1]), 2 runs concurrent per VM,
estimate 1.0-1.8 h each, about 12 L4-h = about $10.4 at the $0.87/h ceiling (estimate from harness README table, unmeasured
for 2-per-GPU concurrency). At most 3-4 E4 VMs at once (leave group headroom for siblings).
Analysis script (written before the runs): analyze.py. Pipeline proof: smoke_tiny.sh (tiny model, CPU, all 5 configs,
mask_violations 0, /tmp/lir-agent/E4/smoke/).

Deviations log (append only, timestamped):
- 08:35Z: M1 in experiment code (harness has no gating switch any more); router channel untestable (no router); leak test
  lives in experiments/E4 not harness/tests; checkpoints not shared with E1/E2; 80 RL steps (not 200) to fit 2 h VMs.
- 08:40Z: infra run ids must be lowercase ([a-z0-9-]), so run ids are lir-e4-<preset>-<kind>-s<seed>, not lir-E4.
- 08:53Z: measured on VM lir-e4-syco16-r-s1: two concurrent runs on one L4 each run ~2.3x slower than the single-run
  probe (P at SFT step 48 after 11 min; GPU 100 %, 20.8 GB). VM job layout changed for every VM not yet started:
  R2 = P + twin (twin starts from P's warm start), MM = M1 for seeds 0 and 2, F = B0 + M0 unchanged; note_rel 0.4 cut to
  B0 + P (kind Y), M0 and M1 at 0.4 dropped for budget. Seed 1 keeps the original R layout (P, then twin; M1 in parallel),
  so its twin may stop early (partial eval points kept). Data channel is therefore computed at the last eval step reached
  by both the arm run and its paired twin (analyze.py, written 08:58Z, before any result was seen).
- 11:15Z (POST HOC, after seeing seed-1 P/twin/M1 curves): the no-RL twin's fixed-set forecast moves a lot between eval
  points from prequential training alone (signed vs Bayes 0.164 -> -0.009 -> 0.160 at steps 0/20/40, runs/syco16/twin-s1/e4_fixed.jsonl),
  so a single-endpoint difference is noisy. Added a curve-mean version (mean over RL checkpoints minus pre; paired with the
  twin over common steps). It is reported as secondary and labelled post hoc; the preregistered endpoint stays primary.
- 11:12Z: queue poll interval 120 s -> 30 s (queue2.sh); budget group saturated by sibling VMs, E4 VMs admitted one at a time.
- 13:58Z: queue stopped. Programme GPU spend (GPU-LOG, lir-* DELETE lines) $37.34 actual at the rate ceiling plus 6 running
  VMs; the group cap refuses new 6 h reservations. note_rel 0.4 (kind Y) has seed 0 only (B0, P); Y seeds 1, 2 not run.
  All M1 runs (MM VM, and seed 1 on the R VM) hit the 100 min per-run timeout at RL step 40-60 (M1 computes all three
  adapters in every pass, ~2x slower); their curves are partial and compared at matched steps / matched reward.
- 14:25Z: note_rel 0.4 (Y s0): B0 died of CUDA OOM after its pre_rl eval (two processes on one L4; runs/syco/B0-s0.log);
  P s0 completed. Relaunch impossible under the cap (programme actual $41.97 + 3 running VMs; admission needs a $5.22
  reservation). The 0.4 setting is therefore one P seed only: reported as a pilot, not used for any claim.
