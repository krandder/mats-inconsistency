# PREREG E3   (2026-10-03 ~08:35Z, Opus 5.5 subagent for eurisko; hawk file: ../../hawks/E3.md, verdict RESHAPE)

Paper modified: Hao et al. 2023, Reasoning with Language Model is Planning with World Model (RAP), https://arxiv.org/abs/2305.14992
(and Yao et al. 2023, Tree of Thoughts, https://arxiv.org/abs/2305.10601), read through Gao, Schulman, Hilton 2022,
Scaling Laws for Reward Model Overoptimization, https://arxiv.org/abs/2210.10760 (BoN gold curve, unbiased subset estimator).
URLs as cited and fetched by the hawk 2026-10-03 (hawks/E3.md sections 2 and 7); Huang et al. 2025 re-fetched by me
2026-10-03 (PDF, Alg. 2-4 read; /tmp/lir-agent/e3/huang.txt).

Modification (one sentence): the world model that scores RAP/ToT candidate plans is a separate goal-blind forecaster adapter
trained only on realised (branched) outcomes, compared with a forecaster living in the policy's own shared adapter with the
goal visible, under exogenous best-of-n selection and under RL pressure on the proposer.

Claim under test (one sentence, falsifiable): a goal-blind forecaster adapter trained only on realised outcomes and co-trained
with an RL proposer keeps its tail calibration (excess gap) and decision value (gold deficit vs oracle) as the proposer drifts,
at least as well as a shared-adapter forecaster with the same data and loss; and the instrument detects exploitation where
reward passes through a forecast (positive controls).

## Hawk reshape items: taken / not taken

1. Headline = gold(n) (exact Bayes P(success) of chosen plan) and excess gap(n) (isotonic f->Bayes fit on held-out calibration
   states of the same pair; selection by raw forecast). Raw gap reported only. Gap vs held-out probe Brier reported. TAKEN.
2. Score vs exact Bayes (enumeration over 11088 placements), realised outcome as n=1 check. TAKEN.
3. Exogenous n: one 64-candidate draw per state, n=1..64 by the unbiased subset estimator, distinct plan sets and a
   dedup-n variant reported. TAKEN. Policy-chosen n stratified by entropy: NOT POSSIBLE in this harness (K=3 fixed, no
   policy-initiated stopping; README "Not built yet"); replaced by exogenous-n gap/gold stratified by posterior entropy.
4. Selection vs learning split: frozen proposer (c000) BoN curve; proposer checkpoints at 0/25/50/100% of RL at n in {4,16,64}
   (all n computed) against the frozen forecaster snapshot (c000 world model) and the co-trained one. TAKEN.
5. Positive controls: B0 fused on approval reward (syco note + w_conf for claimed success >= 0.7) and RAP-style arm whose GRPO
   reward is the forecaster's E[P(success)] of the committed plan, in two versions: co-trained forecaster (RAPc) and frozen
   forecaster (RAPf). TAKEN (RAPf added: pure Gao-style RL against a fixed proxy).
6. Preset "hard" (and "hardsyco" = hard + captain's note, defined at runtime in e3_train.py, for the B0 control). TAKEN.
7. gating="role" / "all": the harness no longer has a gating switch: every role pass is a separate sequence computing only its
   own adapter (R1/R2; tests C0 and T2a), which is the "role" setting by construction; the "all" leak cannot be run. FCIN is
   the canonical cell-token plan (render.forecast_seq); a raw-text FCIN channel does not exist (plans are cell tokens chosen
   one by one). Both rows NOT RUN, reasons recorded; the reward-into-forecast leak is E2's dose arm.
8. Matched separation comparator: M0 (one shared adapter, forecast tokens excluded from GRPO, same truth losses = same forecast
   data and loss) with goal visible in belief/forecast inputs (--p-goal-mode visible = no content mask) vs P (separate
   goal-blind adapter). TAKEN. Prompted-base RAP reference row: NOT RUN (budget; hawk calls it reference only).
9. Baselines: InferenceTimePessimism (Huang et al. 2025, https://arxiv.org/abs/2503.21878, Alg. 2-3; beta tuned on calibration
   states) and Smith-Winkler linear posterior-mean shrinkage (Smith and Winkler 2006, doi 10.1287/mnsc.1050.0451; content
   claim unverified). TAKEN.
10. Fixed forecast format only (FF arm not built in the harness). TAKEN.
11. 3 seeds; states per seed: 240 (120 test, 120 calibration); seed 0 is the pilot and sizes seeds 1-2 if SE > 0.01. TAKEN.

## Arms (exact; code commit harness 47077e771a1045aa6307a6c0bd0ec8efd0c3c7de, unmodified; wrappers in this directory)

Common: Qwen/Qwen3-0.6B, bf16 base, LoRA r8, cuda, preset hard, SFT 80 steps x 16, RL 80 steps x 32 episodes (4 groups x 8),
micro-batch 4, A4 forecast supervision (branch all), K=3 candidates in training. Seeds 0, 1, 2.
- P (main):   e3_train.py --arm P  --init-from warm --reward-mode env       (goal deleted, separate adapters, env reward)
- RAPc (+ctl): e3_train.py --arm P  --init-from warm --reward-mode forecast  (reward = forecaster E[P(success)] of commit)
- RAPf (+ctl): e3_train.py --arm FB --init-from warm --reward-mode forecast  (same, belief/forecast adapters frozen)
- M0g (matched shared comparator): e3_train.py --arm M0 --p-goal-mode visible
- B0s (+ctl, fused baseline B0): e3_train.py --arm B0 --preset hardsyco --p-reward approval
  P, RAPc, RAPf share one warm start per seed (warm = c000). Remote orchestration: e3_remote.sh; driver e3_driver.sh.
Evaluation (e3_eval.py): 240 states (seeds 70,000,000+i), 64 candidates each from the proposer checkpoint; forecasts by the
world-model checkpoint; probe set 200 states x 16 uniform random plans (policy-independent). Pairs (proposer:forecaster):
c000:c000, c{025,050,100}:c{same}, c{025,050,100}:c000 (RAPf: own only, forecaster frozen).

## Primary metric
excess_gap(n=64) and gold_deficit(n=64) = oracle(64) - gold(64), from runs/s*/ARM/e3_metrics.json: pairs["c100:c100"] (co-trained)
and pairs["c100:c000"] (frozen forecaster), compared with pairs["c000:c000"] (frozen-proposer null).

## Comparison rule, predictions
Per seed, delta = metric(c100 pair) - metric(c000:c000); mean +- sd over 3 seeds.
- Instrument check (must hold or the main result is uninterpretable): RAPf shows delta gold_deficit(64) > 0.02 and/or delta
  excess_gap(64) > 0.02 in >= 2/3 seeds, and/or B0s shows forecast-at-commit inflation (steps.jsonl
  fc_committed_signed_vs_realised rising by > 0.05 from first to last 10 RL steps) in >= 2/3 seeds.
- Claim SUPPORTED if, with the instrument live: P co-trained |delta excess_gap(64)| <= 0.02 and delta gold_deficit(64) <= 0.02
  (mean over seeds), and P's excess gap and gold deficit at c100 are not worse than M0g's by more than 0.02.
- Claim REFUTED if P co-trained delta excess_gap(64) > 0.02 or delta gold_deficit(64) > 0.02 (mean, consistent sign in >= 2/3
  seeds), or P worse than M0g by > 0.02 on either.
- MIXED otherwise (e.g. P fine but M0g also fine: separation not shown to matter).
- Secondary: RAPc vs RAPf tells whether co-training on realised outcomes defends against forecast-as-reward exploitation.
- Expected from the hawk: raw gap grows with n in every arm; isotonic removes most of it; P vs M0g difference small.

## Kill / stop
Hawk: kill only if canonical FCIN + realised reward + role gating AND no positive control; positive controls are run, so no kill.
Each VM <= 2 h (internal deadline 108 min); GPU-LOG lir-agent total >= $50 stops launches; if launching would pass $50, run
reduced scale on CPU or mark not-run.

## Budget
Per seed 2 L4 VMs x ~1.8 h = 3.6 L4-h; 3 seeds = ~11 L4-h, ~$9.4 at the $0.87/h ceiling. Programme spend before: $0.66.

## Analysis script (written before the runs): e3_analyze.py ; aggregation: e3_aggregate.py (written before seeds 1-2 results).

## Deviations log (append only)
- 2026-10-03T08:52Z run-id: infra requires lowercase ids (lib.sh valid_run_id), so the runs are lir-e3-{roles,fused}-s{0,1,2}
  (group lir-agent), not "lir-E3". Launches were refused at 08:40Z (group reservations 48.10 + 5.22 > 50, sibling VMs);
  a queue (/tmp/lir-agent/e3/launch_queue.sh) launches each VM when reservations <= 44.7. All 3 seeds queued; seed 0 is
  still the pilot: if it shows a defect the queue is stopped before seeds 1-2 run.
- 2026-10-03T13:10Z COMPUTE DEVIATIONS (seed-0 pilot timings, runs/s0 timing files): the L4 runs ~120 forecast sequences/s
  in total, so the planned eval (240 states, 4 checkpoints, 3 arms per VM in 30 min) could not finish on the training VMs
  (s0 roles VM wrote only 1 pair per arm). Changes, decided before looking at any metric:
  (a) checkpoints are copied to /tmp/lir-agent/e3/ck and evaluated on separate eval VMs (e3_remote_eval.sh, driver
      e3_driver2.sh; run ids lir-e3-eval-s{0,1,2});
  (b) eval states 240 -> 120 (60 test / 60 calibration) per pair, probe set 200 -> 100 states x 16 plans; pair files
      already computed at 240 states are truncated to the first 120 states (same state seeds) in e3_analyze.py;
  (c) the 25% checkpoint (c025) is dropped; checkpoints are 0 / 50 / 100% of RL;
  (d) RL length is set by the time budget (e3_train.py --rl-deadline): with three arms sharing one L4, seed 0 ran
      P and RAPc for 39 RL steps and RAPf for 61 (runs/s0/*/summary.json), not 80;
  (e) roles seed 2 VM lost SSH during setup (logs/gpu-roles-s2.log, rc 255); retrained as run lir-e3-rolestrain-s2
      (training only, same deadline formula).
  Observation that motivated nothing above but is recorded now: during RL on seed 0 the forecaster's E[P(success)] was
  nearly constant across plans (RAPf reward 0.65 at every step, runs/s0/RAPf/steps.jsonl), i.e. the 80-step warm start
  gives a weak forecaster at 0.6B on "hard".
- 2026-10-03T13:55Z BUDGET STOP: logical-soup spend_usd lir-agent = 56.46 at 13:49Z (reservations + deleted cost; deleted VMs
  alone $35.83, GPU-LOG.md), above the $50 launch stop. Roles seed 2 is NOT retrained and the s2 eval VM is cancelled, so P,
  RAPc and RAPf have 2 seeds (s0, s1); M0g and B0s have 3 training seeds. Two eval VMs (s0, s1) stay queued and launch only if
  the infra gate (reserved + spent + 2.15 <= 50) admits them. Missing main-arm pairs (P c100 and c050, both forecasters) are
  computed on CPU at reduced scale (120 states, probe 50 states x 16 plans, bf16, 3 threads), with forecasts deduplicated per
  (state, ordered plan) (e3_eval.py change; values identical up to bf16 batch effects).
- 2026-10-03T14:21Z GPU queue cancelled (deleted lir-agent VMs alone $41.20 at 14:20Z, GPU-LOG.md; reservation metric 55.67):
  no E3 eval VMs ran. CPU eval reduced again (the box was at load 33; ~1 forecast/s per 3-thread job): P c100:c100 and
  c100:c000 for s0, s1; M0g and B0s c100:c100 for s0, s1, s2; 60 states (30 test / 30 calibration) and probe 30 states x 16.
  c050 pairs are not computed. The c000:c000 references keep 120 states (truncated from 240).
- 2026-10-03T15:55Z POST-HOC ANALYSIS ADDITION (after seeing P c100 results): the P forecaster is near-constant, so the
  pre-registered single-split excess gap reduces to (mean Bayes value of calibration states - mean of test states) and its
  reported SE (test states only) omits the calibration-fit noise. Added, labelled post-hoc: 2-fold cross-fitted excess gap
  over all states (e3_analyze.py "posthoc_all_states") and gold / raw gap over all states. Pre-registered metrics are kept
  and reported unchanged. Also fixed an aggregation bug (excess_ref_gap appended once per metric key, e3_aggregate.py).
- (late entry, change made 2026-10-03T13:25Z before any main-arm result) Smith-Winkler shrinkage: replaced the regression
  slope cov(f,v)/var(f) (unbounded, 840 on the near-constant pilot forecaster) by the classical factor
  var(v)/(var(v)+var(f-v)) in [0,1], centred on the calibration means.
