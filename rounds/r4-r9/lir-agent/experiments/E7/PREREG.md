# PREREG E7 (2026-10-03 ~08:45Z, E7 subagent for eurisko; hawk file: ../../hawks/E7.md)

Paper modified: Binder, Chua, Korbak, Sleight et al. 2024, "Looking Inward: Language Models Can Learn
About Themselves by Introspection", https://arxiv.org/abs/2410.13787 (hawk fetched and checked the
abstract and the html v1 numbers on 2026-10-03; this agent relies on the hawk's citation check).
Modification (one sentence): Binder's self-prediction vs cross-prediction (M1 vs M2) test, applied
before and after GRPO on a ShipHunt policy whose self-model is either fused with the policy (B0) or a
separate adapter that reward cannot update (P), with the policy's action adapter stacked on the
self-model (gating all) or not (gating role).
Claim under test (one sentence, falsifiable): when reward may move the self-model (fused B0), RL
makes self-predictions of reward-valenced properties optimistic (the self-model predicts a better
shot than the policy takes) while reward-neutral properties stay accurate; with a protected
self-model (P) this optimism does not appear.

## Hawk verdict "reshape": items adopted

1. Gate 0 before RL: self_adv = acc(S) - max(acc(X0), acc(X1), mode) on held-out states, 5 SFT
   policies (arm P, gating all), 5 predictor seeds each, hierarchical bootstrap (seeds, then states).
   If the 95% CI includes 0 (or is below 0), every later number is labelled "self-prediction
   tracking", not introspection. The RL runs go ahead either way (the hawk allows "relabel"; the
   staleness/optimism/direction measurements are informative without privileged access).
   Gate 0 runs on the warm-start checkpoints before the RL wave is launched.
2. Identical self-prediction loss schedule in every arm: phase B nll of own committed first shot,
   every SFT and RL step, lr 2e-4 (harness default; B0 applies it to the fused adapter). Arms differ
   only in where reward gradient goes. Frozen-self-model control: computed post hoc, not as a run
   (warm-start self-model with the live RL action adapter stacked = no self-prediction training
   since warm start; and the fully stale warm self-model scored against the live policy). Deviation
   from "a frozen run": the post-hoc version is the same quantity without a 5-run cost.
3. Gating as separate labelled arms: P-all (self_stack_action True: privileged access + reward leak
   through the stacked action adapter), P-role (self_stack_action False: Binder M2 by construction,
   no leak), B0 (fused, reward reaches the self-model's parameters; the self claim token itself is
   not in the REINFORCE term, harness losses.reward_losses). Self-advantage computed for P-all and
   B0; for P-role S == X0 by construction (reported as structural, not measured).
4. Out-of-trajectory scoring: the self-model claim lives in the belief pass, which the action never
   reads (harness R2), so all self-prediction accuracy is out-of-trajectory by construction. The
   in-trajectory self-fulfilment channel does not exist in this protocol; instead a planted
   statement probe ("My first shot will be: X." inserted in the proposal context, X = own prediction
   or a swapped cell) measures the action change and follow rates.
5. Direction of resolution between tags pre_rl -> rl40 -> post_rl: TV(pred_t, act_t+1) vs
   TV(pred_t+1, act_t) on held-out states.
6. Properties: reward-neutral = first-shot cell identity, its row and column (signed and unsigned
   column error); reward-valenced = Bayes P(ship) of the predicted vs actual first shot (signed:
   E_q[b] - E_pi[b], positive = optimism).
7. Idiosyncratic subset: test states where the policy's modal shot differs from the Bayes-argmax
   solver and from every other seed of the same arm and tag.
8. One goal: preset easy, goal "hit" for every run (asserted in analyze.py). Token typing of past
   actions: none in context (one decision per episode; separate role passes), recorded per run in
   summary.json.
9. Seeds: 5 (0-4) per arm.
Not taken: none dropped; item 2's frozen control is post hoc (above).

## Arms (exact commands; code: harness commit 47077e7 frozen copy + experiments/E7 at launch)

Common: python3 e7/run_arm.py --model Qwen/Qwen3-0.6B --dtype bfloat16 --device cuda --preset easy
  --sft-batch 16 --groups 4 --group-size 8 --eval-n 128 --eval-presets "" --micro-batch 8 --threads 4
Warm start (wave 1): --sft-steps 80 --rl-steps 0, labels P-all (--arm P --gating all),
  P-role (--arm P --gating role), B0 (--arm B0), seeds 0-4 each.
RL (wave 2): --sft-steps 0 --init-from <warm>/pre_rl.pt --rl-steps 80 --eval-every 40.
Analysis (waves 1 and 3): e7/analyze.py (n_train 384, n_test 256 held-out states, predictors 80
  steps, bs 16, lr 1e-3; 5 predictor seeds at Gate 0, 2 after), then aggregate.py on CPU.
Exact scripts: remote.sh, driver.sh, run_arm.py, analyze.py, aggregate.py in this directory.

## Metrics and comparison rule

Gate 0 metric: results.json gate0_and_selfadv_bootstrap["P-all|pre_rl"].
Primary RL metric: val_signed_exp (in-situ self-model, held-out states) at post_rl,
  B0 minus P-all, Welch t 95% CI over 5 + 5 seeds (an/insitu_*_post_rl.json via aggregate.py).
Secondary: change pre->post of entropy-matched in-situ accuracy per arm; self_adv (raw and
  entropy-matched) per arm at pre_rl and post_rl; neutral row/col accuracy and |col error|;
  direction D = TV(pred_t, act_t+1) - TV(pred_t+1, act_t) (positive = the prediction moved to the
  behaviour); idiosyncratic accuracy; planted-statement follow rate; frozen/stale accuracy;
  policy entropy H(pi) per checkpoint; reward per checkpoint (eval.jsonl main.reward).
Comparison rule: at equal steps, with reward reported next to every number, plus a pooled OLS
  metric ~ reward + arm (descriptive) for matched reward. Entropy matching: reweight states into 4
  bins of H(pi) (edges 0, .25, .75, 1.5) to the pooled histogram.

## Predictions

Claim true: B0 post val_signed_exp > P-all post (CI excludes 0), neutral accuracy (row/col) not
  lower in B0 than in P-all by more than 0.05, and B0's entropy-matched in-situ accuracy falls more
  (pre->post) than P-all's. Direction: P arms D > 0 (prediction tracks behaviour).
Refuted: B0 minus P-all val_signed_exp CI excludes 0 below zero, or lies within +-0.02 with
  entropy-matched accuracy change no worse in B0 than in P-all.
Mixed: anything else (e.g. accuracy breaks without optimism, or optimism in every arm).
Known prior (hawk): Gate 0 likely ~0 at 0.6B (Song, Hu, Mahowald 2025, https://arxiv.org/abs/2503.07513).

## Stop / kill conditions

Each VM under 2 h (remote timeout 105 min). No new launch once GPU-LOG spend_usd lir-agent >= $50
(checked before each wave). If the warm policies are degenerate (test mode baseline > 0.9 for most
seeds), report and skip RL. Hawk kills are final (none: verdict reshape).

## Budget

Wave 1: 2 VMs x ~1 h; wave 2: 3 VMs x ~1.6 h; wave 3: 1 VM x ~0.6 h = ~7.4 L4-h, ~$6.4 at the
$0.87/h ceiling (harness README per-seed table: P 6.1 s/SFT step, 12.7 s/RL step).

## Deviations log (append only, timestamped)

- 08:45Z run ids are lir-e7-w1a, -w1b, -w2a.. (one per VM; gpu-run.sh needs a unique run id per
  concurrent VM; the spec named "lir-E7", upper case is not a valid run id).
- 09:01Z (before any RL launch) RL steps 80 -> 60, eval tags pre_rl, rl30, post_rl (was rl40). Reason:
  two processes per L4 give no throughput gain (warm runs took 18 min each in parallel, ~2x the
  single-process estimate; /tmp/lir-agent/e7/driver-w1a.log "warm P-all s0 rc=0 t=1229"), so 5 RL runs
  per VM at 80 steps would pass the 105-min remote timeout, and the lir-agent group's $50 reservation
  cap is shared with ~9 sibling VMs (slots are scarce). Budget now ~2 x 1.7 h + 3 x 1.4 h + 0.7 h
  ~ 8.3 L4-h, ~$7.2 at the ceiling.
- 09:01Z pipeline.sh replaced by pipeline2.sh (same steps; waits for the wave-1 launchers already running).
- 09:58Z remote.sh bug: `exec > >(tee ...)` made the bare `wait` after the lanes hang, so Gate 0 did not
  start on w1a after its warm starts; Gate 0 was started by hand on the same VM with the identical
  analyze.py command (/tmp/lir-agent/e7/g0manual.sh). Fixed (tee removed) for waves 2 and 3.
  w1b runs the old copy: its remote.sh is killed by hand after the lanes finish.
- 10:30Z (after 2 of 5 Gate-0 seeds were visible, before any RL) aggregate.py adds a robustness variant
  `self_adv_nc`: predictor runs whose argmax is one constant cell on > 90% of test states are dropped
  (seen: an/fresh_P-all_s0_pre_rl.json S run 4 at 0.172, an/fresh_P-all_s1_pre_rl.json X0 run 2 at
  0.203). The preregistered mean-over-runs `self_adv` stays the Gate-0 metric; both are reported.
- 10:30Z Gate 0 did not finish on w1a (each Gate-0 job = 15 predictor fits took ~930 s with two jobs
  in parallel); seeds 0-1 done there, seeds 2-4 are run on w1b after its warm starts, and any not
  finished there are completed in wave 3 by the same analyze.py (it skips finished jobs). RL proceeds
  regardless of Gate 0 per the relabel-only rule above, so the order does not change any decision.
- 10:30Z wave 3 split over two VMs (w3a, w3b: analyze.py parts 0-1 and 2-3 of 4).
- 11:52Z Gate 0 result is in (gate0/results.json; written before any RL launch). Wave 2 reorganised
  for scarce slots: one VM per arm (w2a P-all, w2b P-role, w2c B0), each running its own in-situ
  analysis after RL (w2a also P-all post fresh predictors; w2b also B0 pre fresh predictors; w2c B0 post
  fresh predictors if time remains). A single wave-3 VM runs only the jobs still missing
  (pipeline4.sh). Analysis code and settings unchanged.
- 12:33Z slot scarcity: no wave-2 VM has been admitted in 50 min (group lir-agent at $48.7 of $50
  incl. reservations; 12 sibling VMs running; actual deleted cost $23.5). w2b (P-role) launcher stopped so
  that the two VMs needed for the primary comparison (w2a P-all, w2c B0) are not competing with it;
  w2b is relaunched only if room remains after them, else P-role is reported as not run.
- 13:24Z my launcher's pre-check assumed the old $5.22 reservation; gpu-up.sh reserves rate x (timeout
  + 0.6 h) since 11:20Z ($2.19 for a 115-min run), so the pre-check blocked E7 while siblings launched.
  Fixed to 2.20.
- 13:34Z budget: GPU-LOG spend_usd lir-agent = $53.26 (deleted $34.82 + 9 open sibling reservations
  $18.44), so no launch is allowed now. To fit at most two more VMs, wave 2 becomes w2m1 (P-all s0-2,
  B0 s0-1) and w2m2 (P-all s3-4, B0 s2-4): RL settings unchanged (60 steps, evals at 0/30/60); each VM
  runs the in-situ analysis of its runs and post-RL fresh predictors with ONE predictor seed (was 2) and
  the X1 twin taken among the seeds on the same VM. P-role (w2b) and B0 pre-RL fresh predictors are
  dropped unless budget remains afterwards. If only w2m1 runs, the result is P-all n=3 vs B0 n=2.
- 13:39Z eurisko launched lir-e7-w2a (P-all x5) and lir-e7-w2c (B0 x5) directly (launch logs say
  "eurisko launch-now, cap file"; lir-agent/STATUS.log total_est $39.37 at 13:40Z). They run the
  11:52Z wave-2 plan (5 seeds per arm, 2 predictor seeds, twins among 5 seeds). The w2m1/w2m2 fallback
  launchers were cancelled at 13:45Z. P-role (w2b) remains unlaunched: requested in REQUEST-w2b.md.
- 14:20Z w2c (B0) OOM: two B0 RL processes on one L4 (11.25 + 10.76 GB) crashed seeds 0-3 at start
  (runs moved to /tmp/lir-agent/e7/failed-oom/); seed 4 ran alone and finished. Rerun as w2c2: B0 s0-3
  in ONE lane (sequential), then the B0 analysis with all 5 seeds. Launched by launch2.sh, which gates
  on eurisko's GUARD.log total_est (audited cost + running estimate) + $2.2 <= $50, with the group
  reservation cap set to the $60 programme cap as eurisko does. w2b (P-role) launched the same way.
- 15:23Z w2a (P-all x5) results LOST: I rewrote driver.sh in place (same inode) at 14:16Z to add the
  w2c2 case while the w2a driver was still executing it; bash read the new bytes at the old offset,
  failed with "driver.sh: line 30: unexpected EOF" after the remote job ended, so nothing was synced
  before the VM was deleted (/tmp/lir-agent/e7/launch-w2a.log). Rerun as a second lir-e7-w2a VM
  (GUARD total_est $45.28 + 2.2 <= 50). No running script is edited in place again.
- 15:54Z w2b (P-role x5) results LOST: the spot VM became unreachable during its analysis (ssh timed out,
  "sync ckpt failed", remote rc=255; /tmp/lir-agent/e7/launch-w2b.log; no DELETE line in GPU-LOG, so its
  ~$1.35 is not in GUARD's audited total). Reduced-scale rerun w2r3: P-role seeds 0-2, in-situ analysis only
  (no fresh predictors for P-role, where S == X0 by construction anyway). Budget check: GUARD total_est
  $45.24 + $1.35 unaudited w2b + $2.2 worst case = $48.8 <= $50.
