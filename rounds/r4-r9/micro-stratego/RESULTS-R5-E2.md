# RESULTS-R5-E2: 6x6 Barrage micro-Stratego, coupled tactics and preference-guided search (E-MS-6x6)

Executor: an Opus 5.5 subagent of eurisko, 2026-10-17, for opus-mats (agent mail #3688). Spec:
`/home/agent/workspaces/mats/outbox/experiments-round-5-spec.md`, section 2 (rules E2-D0 to E2-D3). Pre-registration:
`PREREG-R5-E2.md`, sha256 `732514b5...` (`PREREG-R5-E2.sha256`), hashed 2026-10-17T03:25:20Z, before the first launch
(GPU timing smoke 03:26:12Z; seed runs 03:36:21Z; `r5/gpu/launches.log`). No rule, threshold, scoring set or
hyperparameter was changed after hashing; operational events are logged in `PREREG-R5-E2-AMENDMENTS.md`.
Paths below are relative to `/home/agent/workspaces/mats/work/micro-stratego/`.

## TL;DR

- **Pre-registered result: the E2-D0 gate FAILED, so D1-D3 were not rule tests.** Both world models (1.0M and 0.49M
  parameters) learned the one-step rules: battle accuracy 0.9999 and 0.9998, 7-known-slot NLL 0.0001 and 0.0002 nats
  (primary, fallback). But the one-pass belief
  ended just below the bar: KL to the exact posterior 0.0484 nats per hidden slot [95% CI 0.0471, 0.0498] for the
  primary net (bar >= 0.05). On the calibration split neither net met the bar (0.0467 and 0.0493), so the spec's
  capacity rule selected no net. The rollout part of the gate failed by a wide margin: 16 rollouts made the belief worse
  (KL 0.047 -> 0.539) and the action values worse (MSE 0.015 -> 0.036).
- **EXPLORATORY (computed by the same pre-registered pipeline after the gate; not rule tests):** every remaining rule
  would also have failed. The keys collapsed again (cosine 0.99 within and 0.98 across flanks; gap 0.004, bar 0.25), so the
  spec's two fixes did not stop it. attn_L3 cut the value's policy sensitivity 3.6x against mix but made the 8-slot
  belief's 6.2x worse. The trust head again ranked rollout errors far better than output entropy (Spearman +0.23 vs
  -0.36), just under the 0.25 bar. The Miner utility swap moved the choice toward higher true Miner-capture utility on 34%
  of the boards where it could (bar 75%), with theta_real untouched (hash identical, belief change 0.0000). Unified-LIR scored
  0.503 against P0 (bar > 0.52).
- **Post hoc (after the gate; not rule tests):** the rollout belief claims fail for a mechanical reason. Each claim is
  conditioned on the other sampled cone slots, so the one-Flag/one-Miner/one-General/one-Marshal constraint gives the true
  type < 0.001 in about 21% of claims, and the pre-registered log-space pooling keeps those zeros (consensus KL 0.54;
  probability-space pooling of the same claims 0.17; one pass 0.053). Even without sampling,
  scoring the real last 4 plies with theta_real's own policy head is worse than one pass (KL 0.076 vs 0.048). The key
  collapse sits in the fitted projection W_K2: the raw attention fingerprints still differ a little by flank (cosine 0.40
  within vs 0.30 across).
- Cost: **$2.73** at the $0.87/h list-price ceiling (5 VMs, including a $0.10 timing smoke and a spot VM preempted after
  25 minutes); $9 cap. `gcloud` shows no r5e2 VM or disk left (`r5/GPU-LOG-R5-E2.md`).

## Verdict table (all from `r5/report/verdicts.csv`; 3 seeds; mean [95% seed-t interval, df = 2])

| rule | verdict | numbers (seeds 101, 102, 103; mean [95% CI]) |
|---|---|---|
| E2-D0 (1) battle accuracy >= 0.97 | pass | 0.9999, 1.0000, 0.9999; 0.9999 [0.9998, 1.0001] (11,034 ply-0 battles per seed) |
| E2-D0 (2) 7-known-slot NLL <= 0.05 | pass | 0.0001, 0.0002, 0.0001; 0.0001 [0.0000, 0.0002] |
| E2-D0 (3a) 0-step KL >= 0.05 nats/slot | **fail** | 0.0484, 0.0490, 0.0479; 0.0484 [0.0471, 0.0498] (total over hidden slots 0.272, spec note 0.40) |
| capacity rule (calibration split) | **no net selected** | primary 0.0467 [0.0442, 0.0492]; fallback 0.0493 [0.0476, 0.0511]; both < 0.05 |
| E2-D0 (3b-i) belief KL reduction >= 10% | **fail** | -10.52, -10.25, -10.87; -10.55 [-11.32, -9.78] (KL 0.047 -> 0.539) |
| E2-D0 (3b-ii) action-value MSE reduction >= 10% | **fail** | -1.36, -1.06, -2.06; -1.49 [-2.76, -0.23] (MSE 0.015 -> 0.036) |
| **E2-D0 gate** | **FAIL** | (3a) misses by 0.0016 nats; (3b) fails in every seed and both nets |
| E2-D1 policy-independent beliefs | not a rule test (gate failed) | exploratory below: would fail ((1), the belief half of (2), and the value-MSE comparison with mix) |
| E2-D2 theta_trust vs entropy | not a rule test (gate failed) | exploratory below: would fail by 0.022 on the 0.25 bar |
| E2-D3 utility swap and play | not a rule test (gate failed) | exploratory below: would fail ((a) 0.34 vs 0.75; (b) 0.503 vs 0.52) |

Since no net was selected, the gate is reported for the primary net (pre-registered), and so are the exploratory D1-D3
numbers. Under the other reading of the capacity rule (selection on the test split), the fallback would have been
selected: its test gap is 0.0514 [0.0491, 0.0537]. The gate would still fail, because its (3b) values are -9.73 (belief)
and -2.23 (action value) (`r5/report/tables.md`, secondary rows).

## What this means

- **The 6x6 game gives a real but small gap, and the bar sits on it.** The coupled behaviour policy carries evidence
  beyond the public constraints: KL(P*_full || P*_c) is 0.118 nats per hidden slot on the test positions
  (`r5/runs/s101/primary/eval/d0.json`), against 0.075 on R4-E1's 4x4 board (`RESULTS-MS.md`). The one-pass net ends 0.048 from the
  posterior, 41% of the constraint-only oracle's distance. The gap
  grows with the turn, from 0.019 at t 6-11 to 0.098 at t 30-36 (`r5/report/d0_by_turn.csv`), and averages 0.048 over
  the pre-registered t in [6, 36]. The smaller 0.49M net leaves almost the same gap as the 1.0M net (0.051 vs 0.048 on test): halving
  the parameters changes it little.
- **Search over the learned model did not add information in any form we measured.** Per-action values from 1-2 cone
  rollouts are noisier than the net's own 0-step value. The pre-registered belief messages are destroyed by
  conditioning plus log pooling (post hoc section). The cleanest version of the idea, scoring the real last 4 plies under
  each hypothesis with theta_real's policy head, recovers about 78% of what those 4 plies are worth when starting from
  the 4-ply-old belief. It is still about 60% worse than the net's one-pass belief. In this game the amortised inverse model beats
  "older belief + learned forward model". Why was not measured; one candidate is the policy head's own errors,
  compounded over the two opponent moves in the window.
- **The key-collapse fix did not work.** Restricting the source set to the cone and fitting on the 48 belief logits +
  value still left W_K2 mapping every rollout to one direction (cosine 0.96-0.999). Post hoc, the raw fingerprints do differ
  by flank, but only slightly (0.40 vs 0.30), and cones overlap heavily across flanks (Jaccard 0.56; 0.80 within). The
  fit ended with one direction and a small scale (u_b 0.09, `r5/runs/s101/primary/train_meta.json`), which amounts to
  shrinking toward the 0-step belief rather than separating cones.
- **The one claim with a consistent signal is again theta_trust vs entropy.** As in R4-E1, the trust head ranks rollout
  errors well and the net's confidence ranks them backwards. The lowest-entropy tenth of rollouts has KL 1.01 to Q*,
  against 0.22 for the rest (`r5/runs/s10x/primary/eval/d2.json`). The effect is weaker than on 4x4 (+0.23 vs +0.42 on 4x4, `RESULTS-MS.md`) and
  misses the 0.25 bar by 0.022.

Options we can see, each with its assumption. None has been run or chosen:

- Belief claims that are marginals, not conditionals: score each cone slot with the others masked, and pool claims in
  probability space. This assumes the per-slot window evidence, about 78% of the 4-ply information (post hoc), can be
  combined without the one-pass belief's advantage disappearing.
- Keys with no fitted projection (the centred raw fingerprint), or a contrastive key loss. This assumes the small
  raw separation (0.40 vs 0.30) is enough to deduplicate.
- A gap test on late positions only (t >= 18, where the one-pass KL is 0.07-0.10), or a generating policy whose evidence
  depends on longer move histories. This assumes the rollouts' information grows faster than the one-pass gap.
- Testing theta_trust where it acts: weighting real decisions or training data, not only ranking rollouts.

## Setup as run

- **Game and data** (`r5/env6.py`, `r5/msgame6.py`, `r5/generate6.py`; data in `/tmp/r5e2/data/`, `meta.json`). 6x6 board,
  lakes (2,2) and (3,3); 8 pieces per side (Flag, 2 Bombs, 2 Scouts, Miner, General, Marshal) on 8 fixed public squares,
  10,080 hidden setups per side; 60-ply limit. Setups split 8,064 / 1,008 / 1,008 (train / calibration / test, disjoint,
  seed 20261017). Self-play with the coupled 2-ply behaviour policy at tau ~ U{0.5, 1, 2}: 60,000 train games
  (1,979,305 positions), 5,000 calibration games (164,321), 3,000 test games (98,816); mean length 33.0 plies; train
  outcomes Red 39.4%, Blue 39.6%, draw 20.9%.
- **Oracle.** P*_full = exact Bayes posterior over the opponent's 10,080 setups (uniform prior x public constraints x the
  behaviour likelihood of every opponent move, tau unknown). Checked against an independent brute-force replay of every
  setup through `env6.step` on 30 test positions: max abs error 6.7e-16 (P*_full) and 0 (P*_c) (`meta.json`). The
  vectorised move likelihood matched a plain-loop re-implementation on 1,464 (state, setup) pairs (max error 4.4e-16; no output file was kept; the audit reproduced
  the check on 418 pairs).
  Q*(s, a): 64 simulator playouts per legal action from the exact joint posterior; Q*_miner = Miner events over the first
  4 plies of the same playouts. Used only for scoring and gates.
- **theta_real.** Two architectures per seed, same schedule: primary 4 layers, d = 128 (1,001,644 parameters); fallback
  3 layers, d = 96 (492,268). 57-token tape (36 squares, 8 opponent slots, turn, 6 ply pairs); any-order masked slot
  loss + move, outcome and result losses on train games only; 16,000 steps of batch 1024 (bf16 autocast), checkpoints at
  30% and 100%.
- **Search.** Cone rollouts: only the 3 hidden opponent slots nearest the root move's target square are materialised;
  4 plies inside theta_real; key from the readout token's attention over the cone slot tokens and touched squares. Belief
  residual per cone slot: the slot's belief at t-4 (given the other sampled cone slots) times the likelihood of the
  opponent's real moves in the last 4 plies under theta_real's policy head. Aggregators mix / ataraxos_mmd / kc_dedup /
  attn_L3 fitted on 1,500 calibration roots against the unskewed K = 128 coverage reference (joint 48 belief logits +
  value). Agents P0 and Unified-LIR (4 waves of key-space PUCT, attn_L3 per-action consensus).
- **Compute.** One L4 per seed (both architectures in parallel on it), group `r5`. Spot for seeds 101 and 102; seed 103's
  spot VM was preempted after 25 min and the seed was rerun on an on-demand (STANDARD) L4.

## Rule tables (generated by `r5/report6.py` from the synced per-seed files; D1-D3 rows are EXPLORATORY)

| rule | criterion | per seed (101, 102, 103) | mean | 95% CI | threshold | pass | note |
|---|---|---|---|---|---|---|---|
| E2-D0 (1) | 1-step battle accuracy (ply-0 battles, Z materialised), test | 0.9999, 1.0000, 0.9999 | 0.9999 | [0.9998, 1.0001] | >= 0.97 | True |  |
| E2-D0 (2) | 7-known-slot NLL (nats), test | 0.0001, 0.0002, 0.0001 | 0.0001 | [0.0000, 0.0002] | <= 0.05 | True |  |
| E2-D0 (3a) | 0-step KL(P*_full ‖ P_0) per hidden slot (nats), test | 0.0484, 0.0490, 0.0479 | 0.0484 | [0.0471, 0.0498] | >= 0.05 | False |  |
| E2-D0 (3a, reported) | same, total over hidden slots per position (nats) | 0.2730, 0.2748, 0.2691 | 0.2723 | [0.2650, 0.2796] | (>= 0.40 noted in spec) | n/a |  |
| E2-D0 (3b-i) | 16 unskewed rollouts: belief KL reduction vs P_0, calibration split | -10.5248, -10.2537, -10.8701 | -10.5495 | [-11.3170, -9.7820] | >= 0.10 (or 3b-ii) | False |  |
| E2-D0 (3b-ii) | 16 unskewed rollouts: action-value MSE reduction vs 0-step, calibration split | -1.3632, -1.0632, -2.0561 | -1.4942 | [-2.7592, -0.2292] | >= 0.10 (or 3b-i) | False |  |
| E2-D0 gate | capacity rule -> None; (1) and (2) and (3a) and (3b) |  |  |  | all | False |  |
| E2-D1 (1) (EXPLORATORY: gate failed) | key cosine within-flank minus across-flank | 0.0001, 0.0008, 0.0111 | 0.0040 | [-0.0113, 0.0193] | >= 0.25, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | JSD_mix / JSD_attn_L3, belief (CI on JSD_mix - JSD_attn) | 0.1494, 0.1629, 0.1720 | 0.1614 | [0.1332, 0.1897] | >= 2.5, CI(diff) > 0 | n/a | diff -0.018673 [-0.020397, -0.016949] |
| E2-D1 (2) (EXPLORATORY: gate failed) | JSD_mix / JSD_attn_L3, value (CI on JSD_mix - JSD_attn) | 3.5370, 3.5948, 3.5901 | 3.5739 | [3.4942, 3.6537] | >= 2.5, CI(diff) > 0 | n/a | diff 0.001315 [0.000946, 0.001683] |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: mix - attn_L3 | 0.0435, 0.0378, 0.0402 | 0.0405 | [0.0333, 0.0477] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: mix - attn_L3 | -0.0005, -0.0009, -0.0001 | -0.0005 | [-0.0015, 0.0006] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: ataraxos_mmd - attn_L3 | 0.1317, 0.1492, 0.1539 | 0.1449 | [0.1158, 0.1740] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: ataraxos_mmd - attn_L3 | 0.0010, 0.0013, 0.0008 | 0.0010 | [0.0004, 0.0017] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | belief KL to ref: kc_dedup - attn_L3 | 0.1329, 0.1497, 0.1627 | 0.1484 | [0.1114, 0.1855] | > 0, CI > 0 | n/a |  |
| E2-D1 (2) (EXPLORATORY: gate failed) | value MSE to ref: kc_dedup - attn_L3 | 0.0015, 0.0016, 0.0013 | 0.0015 | [0.0010, 0.0019] | > 0, CI > 0 | n/a |  |
| E2-D1 (EXPLORATORY: gate failed) | (1) and all of (2) |  |  |  | all | n/a (would be False) |  |
| E2-D2 (EXPLORATORY: gate failed) | Spearman(beta_trust, -KL(Q*‖q_r)) overall | 0.2210, 0.2504, 0.2113 | 0.2276 | [0.1771, 0.2781] | >= 0.25, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | Spearman(beta) - Spearman(-H) overall | 0.6050, 0.6108, 0.5545 | 0.5901 | [0.5133, 0.6670] | >= 0.20, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | same, mean within 10 entropy strata | 0.1844, 0.2269, 0.1495 | 0.1869 | [0.0906, 0.2832] | >= 0.15, CI > 0 | n/a |  |
| E2-D2 (EXPLORATORY: gate failed) | all three |  |  |  | all | n/a (would be False) |  |
| E2-D3 (a) (EXPLORATORY: gate failed) | utility swap: shift rate toward higher true Miner utility (relevant boards) | 0.3684, 0.3108, 0.3333 | 0.3375 | [0.2654, 0.4096] | >= 0.75, CI lower > 0.5; hash identical; belief change 0.0000 | n/a | hash identical all seeds: True; max belief change 0; relevant boards per seed: 76, 74, 84 |
| E2-D3 (b) (EXPLORATORY: gate failed) | Unified-LIR effective win vs P0 (400 duplicate games/seed) | 0.4775, 0.5138, 0.5175 | 0.5029 | [0.4480, 0.5578] | > 0.52, CI lower > 0.5 | n/a |  |
| E2-D3 (EXPLORATORY: gate failed) | (a) and (b) |  |  |  | all | n/a (would be False) |  |

Capacity rule (calibration split, 3-seed means): primary meets (1),(2),(3a): False; fallback: False; selected: None.
Self-check: 69 headline numbers re-derived from the per-item CSVs; largest |difference| 8.03e-07 (report/selfcheck.csv).

Caveat on the E2-D2 strata row: in seed 102 the 10 entropy strata are not equal-count (tied entropies of rollouts
that end in a predicted terminal); with exactly equal-count strata the 3-seed difference is +0.180 [0.108, 0.252]
instead of +0.187 [0.091, 0.283] (`AUDIT-R5-E2.md`). No outcome changes.

Per-seed values come from `r5/runs/s<seed>/primary/eval/<part>.json`; the secondary architecture's rows are in
`r5/report/tables.md`. "n/a" in the pass column means the rule was not run as a test because the gate failed; the
pre-registered bar is shown for comparison only.

## Exploratory D1-D3 (the gate failed; computed by the pre-registered pipeline; NOT rule tests)

Primary net, 3 seeds, mean [95% CI], from `r5/report/tables.md`.

| part | quantity (EXPLORATORY) | mix | ataraxos_mmd | kc_dedup | attn_L3 |
|---|---|---|---|---|---|
| D1 | belief policy-sensitivity JSD | 0.0036 [0.0031, 0.0040] | 0.0134 [0.0081, 0.0187] | 0.0155 [0.0092, 0.0217] | 0.0223 [0.0209, 0.0236] |
| D1 | value policy-sensitivity JSD | 0.0018 [0.0013, 0.0023] | 0.0007 [0.0005, 0.0009] | 0.0011 [0.0010, 0.0012] | 0.0005 [0.0004, 0.0006] |
| D1 | belief KL to the K = 128 reference | 0.201 [0.195, 0.208] | 0.306 [0.276, 0.335] | 0.309 [0.274, 0.345] | 0.161 [0.154, 0.168] |
| D1 | value MSE to the reference | 0.0017 [0.0014, 0.0019] | 0.0032 [0.0013, 0.0050] | 0.0036 [0.0020, 0.0052] | 0.0021 [0.0010, 0.0033] |

- D1 (1) key cosine within / across flanks: 0.986 / 0.982 (gap 0.004 [-0.011, 0.019]; bar 0.25). D1 (2): attn_L3
  cuts the value's policy sensitivity 3.57x against mix [3.49, 3.65] (passes that half), and is closer to the
  reference than every baseline on the belief (all three CIs > 0). But its belief policy sensitivity is 6.2x WORSE than
  mix's (ratio 0.16 [0.13, 0.19]), and its value MSE is not below mix's (difference -0.0005 [-0.0015, 0.0006]). A
  possible reason, not tested: the reference pools the same sharp claims (post hoc section), so matching it rewards reproducing their noise, which is
  bag-dependent; mix dilutes every claim with the 0-step belief and so moves little.
- D2: Spearman(beta_trust, -KL(Q* || q_r)) +0.228 [0.177, 0.278] (bar 0.25); minus Spearman(-H, -KL) +0.590
  [0.513, 0.667] (bar 0.20); within 10 entropy strata +0.149 vs -0.038, difference +0.187 [0.091, 0.283] (bar 0.15).
  Seed 102's strata are not equal-count: 5,349 rollouts that end in a predicted terminal share one entropy value, so the
  quantile cut put 331 and 11,002 rollouts in the two lowest strata. With exactly equal-count strata the seed-102
  difference is 0.207 instead of 0.227, and the 3-seed difference is +0.180 [0.108, 0.252] (`AUDIT-R5-E2.md`). The
  rule outcome does not change.
  169,992 rollouts in all (56,664 per seed; `r5/runs/s10x/primary/eval/d2_rollouts.csv`). 9.4% of rollouts end in a
  predicted terminal; the lowest-entropy tenth has KL 1.01 to Q*, the rest 0.22.
- D3 (a): over 76, 74 and 84 relevant boards (U_Std's choice at least 0.05 below the best true 4-ply Miner utility), the
  U_Miner choice had higher true Miner utility on 34% [27%, 41%] (bar 75%). The swap changed the action on 28% of all
  240 boards; mean true 4-ply Miner utility of the chosen move went from -0.025 to -0.010. theta_real's parameter hash
  was identical before and after, and its 0-step beliefs changed by exactly 0.0000 (`r5/runs/s10x/primary/eval/d3a.json`).
  (b): Unified-LIR vs P0, 400 duplicate games per seed: 0.478, 0.514, 0.518; 0.503 [0.448, 0.558] (bar > 0.52).
- The fallback net gives the same picture (`r5/report/tables.md`): key gap 0.009, belief JSD ratio 0.16, value JSD ratio
  5.6, Spearman trust +0.234 vs -0.319, strata +0.171 vs -0.042, shift rate 0.35, Unified-LIR 0.515 [0.474, 0.556].

## Post hoc diagnostics (written after the gate numbers were seen; NOT rule tests)

Run on CPU from the saved checkpoints (copied to `/tmp/r5e2/ckpt/`; see the deviations log), primary net unless noted.

1. Why the 16-rollout belief consensus fails (`r5/expl_keys_gibbs.py` -> `r5/report/expl_keys_gibbs.json`; 200
   calibration positions per seed, the gate-3b protocol). Each rollout's claim on a cone slot is its belief given the
   OTHER cone slots' sampled values. Because the army has exactly one Flag, Miner, General and Marshal, a claim puts
   zero on any of those types already sampled for a neighbour. 21.6%, 21.1% and 21.4% of covered claims give the true
   type < 0.001. Single claims average KL 1.55-1.61, against 0.057-0.059 for the 0-step belief on the same slots. The
   pre-registered log-space (geometric) coverage pooling keeps the zeros: KL 0.550, 0.556, 0.519. Pooling the same
   claims in probability space gives 0.169, 0.188, 0.160; the 0-step belief gives 0.054, 0.054, 0.052. So the
   pre-registered estimator explains most of the failure, but even the better pooling of 16 rollouts is about 3x worse
   than one pass.
2. Is there information in the real history that one pass misses? (`r5/expl_routeb.py` ->
   `r5/report/expl_routeb_summary.json|csv` via `r5/expl_routeb_summary.py`; 500 positions per split, every hidden
  slot, no sampling.) The slot's belief
   4 plies earlier (KL 0.178 on test) times theta_real's likelihood of the opponent's real moves in those 4 plies gives
   KL 0.076 [0.073, 0.080]. One pass at t gives 0.048 [0.045, 0.051]. Calibration split: 0.193, 0.083 and 0.048. The
   window evidence is real (it recovers about 78% of the distance from the old belief to the current one), but the
   learned forward model is less accurate than the one-pass inverse. Both architectures agree (fallback test: 0.181 ->
   0.081 vs 0.051).
3. Where the keys collapse (`r5/report/expl_keys_gibbs.json`; 60 D1 positions, 12 Left + 12 Right rollouts). Raw
   attention fingerprint cosine within / across flanks: 0.35 / 0.30, 0.40 / 0.30, 0.46 / 0.30; after subtracting the bag
   mean: 0.01 / -0.07, 0.05 / -0.11, 0.11 / -0.15. Fitted W_K2 keys: 0.999 / 0.999, 0.988 / 0.986, 0.968 / 0.957. Cone
   Jaccard overlap: 0.80 within, 0.55-0.57 across. The fingerprints carry some flank information. The fit discarded it.

## Figures

- `r5/report/d0_by_turn.png` (data `r5/report/d0_by_turn.csv`): one-pass KL to P*_full by turn for both nets, the
  constraint-only oracle's distance and the 0.05 bar.
- `r5/report/d1_aggregators.png` (`r5/report/d1_aggregators.csv`): D1 policy sensitivity and reference KL per aggregator.
- `r5/report/train_curves.png` (`r5/report/train_curves.csv`): calibration-split losses during training, 6 runs.

## Compute (GPU-LOG; `r5/GPU-LOG-R5-E2.md`, rebuilt by `r5/gpu/ledger.py` from the shared logical-soup GPU-LOG.md)

All VMs g2-standard-8 + 1 L4 in us-central1-a, group `r5`, launched through `logical-soup/infra/gpu-run.sh` with the
driver `r5/gpu/drive.sh`. Costs are the $0.87/h list-price ceiling x age (spot is usually 60-70% cheaper; the Billing
API is disabled, so no price is verified).

| VM | run | provisioning | created | deleted | cost (ceiling) |
|---|---|---|---|---|---|
| soup-gpu-r5e2-s999-smoke-1017032612-20773 | timing smoke (seed 999, discarded) | spot | 03:26Z | 03:34Z | $0.10 |
| soup-gpu-r5e2-s101-1017033621-16895 | seed 101, both nets | spot | 03:36Z | 04:30Z | $0.77 |
| soup-gpu-r5e2-s102-1017033621-14520 | seed 102, both nets | spot | 03:37Z | 04:30Z | $0.77 |
| soup-gpu-r5e2-s103-1017033621-9474 | seed 103 (preempted 04:02:44Z, nothing used) | spot | 03:37Z | GCE | $0.37 |
| soup-gpu-r5e2-s103-b-1017041104-24544 | seed 103 rerun, both nets | on-demand | 04:11Z | 05:02Z | $0.72 |
| **total** | | | | | **$2.73** of the $9 cap |

Live r5e2 VMs and disks after the last deletion: none (`gcloud compute instances list` and `disks list`,
name~soup-gpu-r5e2-, in `r5/GPU-LOG-R5-E2.md`). Per VM: deps 3-5 min, theta_real training 27.7 min (primary) and 17.7
min (fallback) in parallel on one L4, all other stages 9-13 min, evaluation 6-8 min. CPU: data and oracle generation
56 min on 6 processes; the post hoc diagnostics about 40 min on one process.

## Deviations log (UTC; full text in `PREREG-R5-E2-AMENDMENTS.md`)

- Declared before the hash (`PREREG-R5-E2.md` section 6): fixed public setup squares; 60-ply limit; behaviour-policy
  constants; the "4-ply consistency" check as a window likelihood ratio; coverage reference and naive mix; per-slot
  attn_L3 weights; separate belief and value scales; capacity selection on the calibration split with both nets always
  run; no Ataraxos-1Step / MCTS / fused arms; R4-E1's D2 protocol; heuristic-only self-play; CSV/JSON/PNG-only sync;
  bf16 only for theta_real's training steps.
- Found by the audit, not declared before the hash: (i) the cone is the 3 hidden opponent slots nearest the root
  move's target square at ANY distance (PREREG section 4); the spec asks for slots "within tactical range (distance
  <= 2)", and that limit was not applied (`model6.select_cone` computes `in_range` but nothing uses it). In 5,000 random
  (test record, legal move) pairs, 65% of cone slots were more than 2 squares (Chebyshev) from the target square. This
  affects the exploratory D1 key and cone-overlap numbers; it cannot change the gate verdict, because (3a) fails
  without any rollout. (ii) PREREG section 3 says the data and oracle sets were generated before the hash. That holds
  for the train, calibration and test records and the two D0 oracle sets (03:15-03:17Z). The Q* test set (03:38Z), the
  Q* calibration set and the D1 positions (03:58Z) and the brute-force check (04:00Z) were finished after the hash
  (03:25:20Z), by the same generator run started at about 03:04Z from code unchanged since 03:02Z (`/tmp/r5e2/gen.log`,
  file times).
- 03:26Z GPU timing smoke (seed 999), discarded. 03:36Z seed runs launched while the CPU generator was still writing the
  late oracle files. Eval waited for a go-marker. 03:50Z found that the driver's launch ssh stays attached to the
  remote run, so `r5/gpu/push_late.sh` pushed the late files at 04:22Z. 04:00Z data finished (brute-force check
  6.7e-16). 04:02:44Z seed 103's spot VM preempted; seed 103 rerun on on-demand from 04:11Z (stop rule iii).
  04:07-04:16Z CPU smoke of the late eval parts, no code change.
- 04:23Z and 05:01Z: the theta_real / trust / aggregator checkpoints (13 MB per seed) were copied from the VMs to
  `/tmp/r5e2/ckpt/` before deletion, for the independent CPU audit and the post hoc diagnostics. This goes beyond
  "sync only CSV/JSON/PNG back"; nothing bulky was put under /srv or /home.
- After the gate numbers were seen: the post hoc diagnostics above (`r5/expl_routeb.py`, `r5/expl_keys_gibbs.py`),
  labelled as such; no rule was re-evaluated.

## Self-audit and independent audit

- `r5/report6.py` re-derives 69 headline numbers from the per-item CSVs; largest difference 8.0e-7 (CSV rounding;
  `r5/report/selfcheck.csv`).
- On every VM, before deletion, `r5/audit6_vm.py` recomputed the D0 numbers from the checkpoint with separate code and
  full fp32. Largest differences: KL 1.0e-6, 7-known NLL 2.4e-7, battle accuracy 0 (`r5/runs/s10x/<arch>/eval/audit_vm.json`).
- The oracle: brute-force replay on 30 test positions, max error 6.7e-16 (`/tmp/r5e2/data/meta.json`).
- `AUDIT-R5-E2.md` is a separate pass by an independent read-only agent. Verdict: **supported with corrections**. It
  re-derived 300 numbers from the per-item CSVs (largest difference 6.3e-6), recomputed D0 and the capacity rule on CPU
  from the checkpoints with its own code for all 3 seeds, both nets and both splits (2.2e-6), reproduced every verdict,
  confirmed the pre-registration hash and timing and the byte-exact copy of spec section 2.3, found no oracle in
  training, fitting, selection or the agents, and confirmed $2.73 and no r5e2 VM or disk left. Its seven small
  corrections (strata ties, the undeclared cone-distance interpretation, late data files, two numbers, one phrase,
  sources) are applied above; none changes a verdict. Its scripts and outputs are in
  `/home/agent/workspaces/fable-strategy/scratch/r5e2-audit/`.

## Files

- Pre-registration: `PREREG-R5-E2.md`, `PREREG-R5-E2.sha256`, `PREREG-R5-E2-AMENDMENTS.md`.
- Code (`r5/`): `env6.py` (6x6 board), `msgame6.py` (behaviour policy, exact likelihood and posterior, Q*),
  `generate6.py`, `model6.py`, `train6.py`, `eval6.py`, `agents6.py`, `audit6_vm.py`, `report6.py`, `audit6.py`,
  `expl_routeb.py`, `expl_keys_gibbs.py`, `site_add_entry.py`, `gpu/` (`launch.sh`, `drive.sh`, `run_vm.sh`,
  `stage.sh`, `push_late.sh`, `ledger.py`, `launches.log`).
- Results: `r5/runs/s101|s102|s103/{primary,fallback}/` (`eval/*.json`, per-item `eval/*.csv`, `train_meta.json`,
  `train_curves.csv`, `done.json`); `r5/report/` (`verdicts.csv`, `tables.md`, `summary.json`, `selfcheck.csv`,
  figures and their CSVs, `expl_*.json|csv`). GPU ledger: `r5/GPU-LOG-R5-E2.md`.
- Bulky, outside /srv and /home: `/tmp/r5e2/data/` (data and oracle sets), `/tmp/r5e2/ckpt/` (checkpoints).
