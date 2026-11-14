# RESULTS-MS: R4-E1 (E-MS) micro-Stratego with a learned factored world model

opus-mats subagent, 2026-10-16. Spec: `outbox/experiments-round-4-spec.md` sections 1 and 4 (rules E1-D0 to E1-D5).
Pre-registration: `PREREG-MS.md`, sha256 `4c78f48cc691b1861f351471148de47d5e4e636176a0f4c65d0ab844241d5bc8`, hashed
2026-10-16T19:02:38Z before the first full training run (19:02:55Z). Amendments: none (`PREREG-MS-AMENDMENTS.md`).

## TL;DR

- **Pre-registered result: the E1-D0 gate failed, so D1-D5 were not run as rules.** The learned world model (theta_real,
  852k parameters, 60,000 self-play games on 540 setups) learned the one-step rules completely: on the 90 unseen test
  setups, battle-outcome accuracy was 1.000 on all 9,744 first-ply battles (bar >= 0.97) and the 5-known-slot NLL was
  0.0001 nats (bar <= 0.03). But its one-pass belief about the opponent's hidden pieces is already almost exact: KL to the
  exact Bayes posterior is 0.024 nats per hidden slot [95% CI 0.021, 0.027] (bar >= 0.12). Search has no belief gap to close.
- **Exploratory, after the gate (the coordinator asked for it; these are not rule tests):** D2, D3 and D5 were run on the
  same trained models, with the corrected exact posterior as the oracle.
  - D3: the trust head ranks rollout errors far better than output entropy. Spearman with -KL to the exact Q* is +0.42
    for trust vs -0.43 for -H overall, and +0.29 vs -0.03 within entropy strata. The confident rollouts are the wrong
    ones. Delusional moves: trust 0.50%, entropy weights 0.89%, uniform weights 0.50%.
  - D2: policy-sensitivity JSD was 0.0063 for mix, 0.0015 for ataraxos_mmd, 0.0007 for kc_dedup and 0.0017 for attn_L3.
    But the trained keys collapsed (cosine 0.99 within and across flanks), so kc_dedup does not deduplicate: it just
    shrinks toward the 0-step value, and its value MSE is 53% worse than mix.
  - D5: Unified-LIR's effective win was 0.483 vs P0, 0.493 vs Ataraxos-1Step and 0.505 vs Tabular-MCTS. Every interval
    includes 0.5, so search showed no gain.
- The spec's oracle (`env.exact_slot_marginals`, which only checks consistency with public constraints) is 0.075 nats away
  from the exact posterior, because it ignores what the opponent's moves reveal. The trained net is about 3 times closer to
  the exact posterior than that oracle.
- Cost: $1.68 at the list-price ceiling ($1.29 for the rule run and $0.39 for the exploratory run). All VMs were deleted;
  gcloud showed no r4-e1 VMs or disks left at 20:00:13Z.

## Verdict table

| rule | verdict | numbers (3 seeds; mean [95% t-interval, df = 2]) |
|---|---|---|
| E1-D0 (a) one-step battle accuracy >= 0.97 | pass | 1.000, 1.000, 1.000 (9,744 battles per seed) |
| E1-D0 (b) 5-known-slot NLL <= 0.03 nats | pass | 0.00016, 0.00008, 0.00007; mean 0.0001 |
| E1-D0 (c) 0-step belief KL to P*_full >= 0.12 nats | **fail** | 0.0231, 0.0254, 0.0232; mean 0.0239 [0.0207, 0.0271] |
| **E1-D0 gate** | **FAIL** | (c) fails in every seed, by a factor of 5 |
| E1-D1 any-order vs row-major belief | not run | gated on D0 (no belief gap to close) |
| E1-D2 policy independence (kc_dedup, attn_L3) | not run as a rule | exploratory run below: the pre-registered bar would not have been met |
| E1-D3 theta_trust vs entropy | not run as a rule | exploratory run below: both Spearman bars and the delusional-cut bar would have been met |
| E1-D4 LIR mask, utility swap | not run | gated on D0 |
| E1-D5 tournaments | not run as a rule | exploratory run below: the 0.55 bar would not have been met |

## What this means

- **On this 4x4 board, working out the opponent's setup from public information is easy for a small transformer.** One
  forward pass lands within 0.024 nats per slot of exact Bayes, including the evidence in how the opponent moves. So this
  testbed cannot ask whether any-order search improves beliefs (D1). In the exploratory runs, lookahead and pooling added
  nothing measurable either: D5 was at 50%, and on the calibration games 16 rollouts improved the played move's outcome
  log-loss by only 0.001-0.002 nats.
- **The one claim with a clear exploratory signal is D3.** A rollout-trust head trained on realised outcomes ranks
  rollout errors well, both overall and at matched entropy. The net's own confidence ranks them backwards: the
  lowest-entropy tenth of rollouts has mean KL 1.10 to the exact action value, against 0.55 for the rest. The quarter of
  rollouts that end in a predicted flag capture average 1.11, against 0.40. On delusional moves trust beat entropy
  weights (9 vs 16 events over 3 seeds) but not uniform weights (9).
- **The deduplication test failed mechanically, not on the merits.** The k_v2 key projection, fitted on calibration bags,
  collapsed to one direction, so crowding counted every rollout alike and kc_dedup became a scaled mean that shrinks toward
  the 0-step value. That shrinkage lowers JSD and raises MSE, so a low JSD on its own does not show policy independence.

Options we can see, with what each assumes. None has been run or chosen:

- A harder belief problem that still keeps the rules learnable, e.g. the spec's 6x6 rung (10,080 setups per side), or
  behaviour policies whose moves depend on hidden pieces in longer-range ways. This assumes a gap appears before the
  one-step rules break.
- A dedup test with keys that cannot collapse, e.g. the parameter-free source-attention fingerprint without a trained
  projection, or a contrastive key objective. This assumes the fingerprint itself separates flanks.
- Testing the D3 trust signal where it would matter, on a model with real errors. This assumes the signal is not specific
  to terminal-claim rollouts in this game.

## Setup as run

- **Game and data.** `env.py` (4x4 diagonal-touch, 6 pieces per side, 720 setups per side, 24-ply limit), unchanged.
  Setups split 540 / 90 / 90 (train / val / test, disjoint, seed 20261016). Self-play with a softmax heuristic at a
  per-player temperature tau in {0.5, 1, 2}: 60,000 train games (578,050 positions), 5,000 calibration games (48,599)
  on val setups, 3,000 test games (27,282) on test setups; mean game length 9.6 plies (`data/meta.json`).
- **Oracle.** P*_full = exact Bayes posterior over the opponent's 720 setups from the full public history, including the
  likelihood of every opponent move under the behaviour policy with unknown tau. The constraint-only marginals of
  `env.exact_slot_marginals` (P*_c) are not the posterior of this data (they ignore what the opponent's moves say about
  its pieces): on the D1 positions KL(P*_full || P*_c) = 0.075 nats per hidden slot. Both oracles matched an
  independent brute-force replay of all 720 setups on 40 test positions (max abs error 5.6e-16 and 0).
  Q*(s, a) for scoring: 64 simulator playouts per legal action from the exact joint posterior over (setup, tau).
- **theta_real.** 4 layers, d = 128, 4 heads, 35-token tape (16 squares, 6 opponent slots, turn, up to 6 move/outcome
  pairs), trained 16,000 steps (batch 1024) on train trajectories only, with any-order masked slot loss, move,
  outcome and result losses. 30% and 100% checkpoints. No rule-based pruning anywhere in search.
- **theta_amp / theta_trust / aggregators.** As in `PREREG-MS.md` section 2: stop-gradient MLP adapter distilled from
  search over theta_real (LIR) or with theta_real trainable and an optimistic target (fused); slot- and rollout-level
  excess-surprise heads fitted on the calibration split; E-CA k_v2 keys; mix, ataraxos_mmd, kc_dedup and attn_L3 fitted
  on calibration bags with random flank skew against a model-only K = 128 reference.
- **Compute.** One spot L4 (g2-standard-8) per seed, group `r4-e1`.

## Rule tables (generated by `report_ms.py` from the saved arrays; D2, D3 and D5 rows and sections are EXPLORATORY)

| rule | criterion | per seed | mean [95% CI] | threshold | pass |
|---|---|---|---|---|---|
| E1-D0 | 1-step battle accuracy (ply 0, Z materialised) | 1, 1, 1 | 1.0000 [1.0000, 1.0000] | >= 0.97 | True |
| E1-D0 | 5-known-slot NLL (nats) | 0.00016, 8e-05, 7e-05 | 0.0001 [-0.0000, 0.0002] | <= 0.03 | True |
| E1-D0 | 0-step belief KL to P*_full per hidden slot (nats) | 0.02306, 0.02536, 0.02325 | 0.0239 [0.0207, 0.0271] | >= 0.12 | False |
| E1-D2 (exploratory) | JSD reduction kc_dedup vs mix | 0.8955, 0.8412, 0.9345 | 0.8904 [0.7741, 1.0067] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction kc_dedup vs mix | -0.6697, -0.4158, -0.4969 | -0.5275 [-0.8496, -0.2053] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction kc_dedup vs ataraxos_mmd | 0.6091, 0.4247, 0.5893 | 0.5410 [0.2894, 0.7926] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction kc_dedup vs ataraxos_mmd | 0.1149, 0.1461, 0.1507 | 0.1372 [0.0889, 0.1856] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction attn_L3 vs mix | 0.6713, 0.7734, 0.7263 | 0.7237 [0.5968, 0.8506] | >= 0.5 | n/a: exploratory, bar met |
| E1-D2 (exploratory) | value-MSE reduction attn_L3 vs mix | -0.00469, 0.2152, 0.1446 | 0.1184 [-0.1605, 0.3973] | >= 0.25 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | JSD reduction attn_L3 vs ataraxos_mmd | -0.2296, 0.179, -0.7147 | -0.2551 [-1.3664, 0.8562] | >= 0.5 | n/a: exploratory, bar not met |
| E1-D2 (exploratory) | value-MSE reduction attn_L3 vs ataraxos_mmd | 0.4674, 0.5267, 0.5147 | 0.5029 [0.4251, 0.5807] | >= 0.25 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | Spearman(trust) - Spearman(stiff), overall | 0.8279, 0.8872, 0.8282 | 0.8478 [0.7628, 0.9327] | >= 0.1 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | same, mean within 10 entropy strata | 0.2787, 0.3651, 0.2974 | 0.3138 [0.2008, 0.4267] | >= 0.15 | n/a: exploratory, bar met |
| E1-D3 (exploratory) | delusional-rate cut trust vs stiff | 0.5, 0.5, 0.3333 | 0.4444 [0.2054, 0.6835] | >= 0.4 | n/a: exploratory, bar met |
| E1-D5 (exploratory) | Unified-LIR effective win vs P0 | 0.4662, 0.455, 0.5275 | 0.4829 [0.3860, 0.5798] | >= 0.55 | n/a: exploratory, bar not met |
| E1-D5 (exploratory) | Unified-LIR effective win vs Ataraxos-1Step | 0.4562, 0.485, 0.5387 | 0.4933 [0.3893, 0.5974] | >= 0.55 | n/a: exploratory, bar not met |
| E1-D5 (exploratory) | Unified-LIR effective win vs Tabular-MCTS | 0.4988, 0.4738, 0.5413 | 0.5046 [0.4198, 0.5894] | >= 0.55 | n/a: exploratory, bar not met |

### E1-D0 (gate)

| seed | battle acc ply 0 | battle acc all plies | 5-known NLL | 0-step KL vs P*_full | 0-step KL vs P*_c | n positions |
|---|---|---|---|---|---|---|
| 101 | 1.0000 | 0.9998 | 0.0002 | 0.0231 | 0.0755 | 2000 |
| 102 | 1.0000 | 0.9999 | 0.0001 | 0.0254 | 0.0818 | 2000 |
| 103 | 1.0000 | 0.9997 | 0.0001 | 0.0232 | 0.0762 | 2000 |
| mean [95% CI] | 1.0000 [1.0000, 1.0000] | | 0.0001 [-0.0000, 0.0002] | 0.0239 [0.0207, 0.0271] | 0.0778 [0.0692, 0.0864] | |

Gate: battle True, five-known True, belief gap False -> **FAIL**. KL(P*_full || P*_c) on these positions (data property): 0.0754 nats.

### E1-D2 (policy independence under flank skew, K = 24; reference K = 128) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | JSD mix | JSD ataraxos_mmd | JSD kc_dedup | JSD attn_L3 | MSE mix | MSE ataraxos_mmd | MSE kc_dedup | MSE attn_L3 | flank gap | key cos within / across |
|---|---|---|---|---|---|---|---|---|---|---|
| 101 | 5.99e-03 | 1.60e-03 | 6.26e-04 | 1.97e-03 | 4.92e-03 | 9.28e-03 | 8.21e-03 | 4.94e-03 | 0.078 | 1.000 / 1.000 |
| 102 | 6.07e-03 | 1.67e-03 | 9.63e-04 | 1.37e-03 | 4.87e-03 | 8.07e-03 | 6.89e-03 | 3.82e-03 | 0.084 | 0.972 / 0.966 |
| 103 | 6.89e-03 | 1.10e-03 | 4.51e-04 | 1.88e-03 | 5.37e-03 | 9.47e-03 | 8.04e-03 | 4.60e-03 | 0.091 | 1.000 / 1.000 |
- kc_dedup vs mix: JSD reduction 0.890 [0.774, 1.007], MSE reduction -0.527 [-0.850, -0.205]
- kc_dedup vs ataraxos_mmd: JSD reduction 0.541 [0.289, 0.793], MSE reduction 0.137 [0.089, 0.186]
- attn_L3 vs mix: JSD reduction 0.724 [0.597, 0.851], MSE reduction 0.118 [-0.161, 0.397]
- attn_L3 vs ataraxos_mmd: JSD reduction -0.255 [-1.366, 0.856], MSE reduction 0.503 [0.425, 0.581]

pre-registered bar not met (EXPLORATORY: not a rule test, the D0 gate failed) (all 8 comparisons would have to meet it).

### E1-D3 (theta_trust vs output entropy) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | Spearman trust | Spearman stiff | strata trust | strata stiff | delusional trust | delusional stiff | delusional uniform | n rollouts |
|---|---|---|---|---|---|---|---|---|
| 101 | 0.402 | -0.425 | 0.251 | -0.028 | 0.0033 | 0.0067 | 0.0067 | 35184 |
| 102 | 0.457 | -0.430 | 0.335 | -0.030 | 0.0050 | 0.0100 | 0.0033 | 35184 |
| 103 | 0.393 | -0.435 | 0.276 | -0.022 | 0.0067 | 0.0100 | 0.0050 | 35184 |

Spearman gain overall 0.848 [0.763, 0.933], within strata 0.314 [0.201, 0.427]; stiff delusional choices over 3 seeds: 16. Part 1: pre-registered bar met (EXPLORATORY: not a rule test, the D0 gate failed); part 2: pre-registered bar met (EXPLORATORY: not a rule test, the D0 gate failed).

### E1-D5 (tournaments, 400 games per pair per seed, K = 16) -- EXPLORATORY (the D0 gate failed; run afterwards on the coordinator's request; not a rule test)

| seed | vs P0 | vs Ataraxos-1Step | vs Tabular-MCTS |
|---|---|---|---|
| 101 | 0.466 (183/7/210) | 0.456 (180/5/215) | 0.499 (196/7/197) |
| 102 | 0.455 (178/8/214) | 0.485 (190/8/202) | 0.474 (184/11/205) |
| 103 | 0.527 (205/12/183) | 0.539 (208/15/177) | 0.541 (213/7/180) |
- vs P0: 0.483 [0.386, 0.580]
- vs Ataraxos-1Step: 0.493 [0.389, 0.597]
- vs Tabular-MCTS: 0.505 [0.420, 0.589]

pre-registered bar not met (EXPLORATORY: not a rule test, the D0 gate failed). Cells: effective win (wins/draws/losses).

Figure: `report/d0_by_t.png` (0-step belief KL by turn for the 100% and 30% checkpoints, the constraint-only oracle and the 0.12 bar; data in `report/d0_by_t.csv`).

## Exploratory diagnostics (computed after the gate result; not decision rules)

From the saved per-position arrays (`diag_d0.py` -> `report/d0_diag.json`, `report/d0_by_t.csv`, `report/d0_by_t.png`):

| checkpoint | seed | mean KL to P*_full | median | 90th percentile | share of positions >= 0.12 |
|---|---|---|---|---|---|
| 100% | 101 | 0.0231 | 0.0106 | 0.0492 | 2.4% |
| 100% | 102 | 0.0254 | 0.0129 | 0.0538 | 3.1% |
| 100% | 103 | 0.0232 | 0.0116 | 0.0494 | 2.3% |
| 30% | 101 | 0.0233 | 0.0109 | 0.0517 | 2.8% |
| 30% | 102 | 0.0271 | 0.0124 | 0.0607 | 3.6% |
| 30% | 103 | 0.0239 | 0.0117 | 0.0543 | 2.8% |

- By turn, the net's KL rises from about 0.011-0.012 at t = 4 to about 0.04 at t = 12, while the posterior entropy per
  hidden slot falls from 1.41 to 0.55 nats. It stays below the constraint-only oracle's own distance from P*_full at every
  turn (0.039 at t = 4, 0.122 at t = 12) and below the 0.12 bar at every turn.
- The 30% checkpoint (step 4,800, about 8 epochs) is as good as the final one. Its battle accuracy is 1.000 and its
  5-known-slot NLL is about 0.0003 nats.
- Over training, the calibration split shows the belief loss flat from about step 1,000. The outcome forecaster (value
  head) overfits the 540 training setups after about step 6,000: its calibration CE is lowest at 0.674-0.677 and ends at
  0.725-0.816 (`report/train_curves.png`, `report/cal_fits.json`). The schedule was fixed in advance and was not changed.

Fitted on the calibration split by `train_ms.py` before the gate was known (`report/cal_fits.json`; these are
calibration-split fits, not test results):

| seed | theta_real hash unchanged by LIR distillation | fused arm changed theta_real | per-action pool log-loss: 0-step / with kc_dedup search / with ataraxos_mmd search | fitted per-action scale kc / mmd |
|---|---|---|---|---|
| 101 | yes | yes | 0.5895 / 0.5881 / 0.5880 | 0.0086 / 0.0185 |
| 102 | yes | yes | 0.5836 / 0.5826 / 0.5813 | 0.0074 / 0.0215 |
| 103 | yes | yes | 0.5924 / 0.5911 / 0.5913 | 0.0086 / 0.0158 |

On the calibration games, adding 16 rollouts through theta_real improved the log-loss of the played move's realised
result by only 0.001-0.002 nats over the 0-step value, and the fitted pooling scales were small. This fits the gate result:
the one-pass model leaves little for lookahead to add on this game.

## Exploratory D2, D3, D5 after the gate (coordinator's request, 19:38Z; NOT rule tests)

After the gate failed, opus-mats authorised D2, D3 and D5 on the trained 4x4 models as exploratory work: up to $2, using
the corrected exact posterior (with the self-play move likelihood) as the oracle. They ran exactly as specified in
`PREREG-MS.md` section 3, with the same code, models and scoring sets. The rule labels stay "D0 FAIL, D1-D5 not run as
rules", and the bars below are shown only for comparison. Seed 102 ran on one spot L4. Seed 103's first VM was preempted
by GCE (results lost), so seeds 101 and 103 were re-run together on another spot L4; no number from the lost run exists.
Summary (`explore_summary.py` -> `report/exploratory_summary.md|csv`, from the saved per-seed json/npz files):

| part | quantity (EXPLORATORY, not a rule test) | seeds 101, 102, 103 | mean [95% CI] |
|---|---|---|---|
| D2 | policy-sensitivity JSD, mix | 0.00599, 0.00607, 0.00689 | 0.00631 [0.00508, 0.00755] |
| D2 | policy-sensitivity JSD, ataraxos_mmd | 0.0016, 0.00167, 0.0011 | 0.00146 [0.000681, 0.00224] |
| D2 | policy-sensitivity JSD, kc_dedup | 0.000626, 0.000963, 0.000451 | 0.00068 [3.36e-05, 0.00133] |
| D2 | policy-sensitivity JSD, attn_L3 | 0.00197, 0.00137, 0.00188 | 0.00174 [0.000944, 0.00254] |
| D2 | value MSE vs K=128 reference, mix | 0.00492, 0.00487, 0.00537 | 0.00505 [0.00436, 0.00574] |
| D2 | value MSE vs K=128 reference, ataraxos_mmd | 0.00928, 0.00807, 0.00947 | 0.00894 [0.00706, 0.011] |
| D2 | value MSE vs K=128 reference, kc_dedup | 0.00821, 0.00689, 0.00804 | 0.00772 [0.00594, 0.00949] |
| D2 | value MSE vs K=128 reference, attn_L3 | 0.00494, 0.00382, 0.0046 | 0.00445 [0.00303, 0.00588] |
| D2 | mean |mu*(pi_Left) - mu*(pi_Right)|, mix | 0.075, 0.075, 0.080 | 0.077 [0.069, 0.084] |
| D2 | mean |mu*(pi_Left) - mu*(pi_Right)|, ataraxos_mmd | 0.028, 0.029, 0.024 | 0.027 [0.020, 0.034] |
| D2 | mean |mu*(pi_Left) - mu*(pi_Right)|, kc_dedup | 0.018, 0.021, 0.016 | 0.018 [0.012, 0.025] |
| D2 | mean |mu*(pi_Left) - mu*(pi_Right)|, attn_L3 | 0.040, 0.033, 0.039 | 0.037 [0.028, 0.046] |
| D2 | key cosine within flank | 0.9999, 0.9719, 0.9999 | 0.9906 [0.9503, 1.0308] |
| D2 | key cosine across flanks | 0.9999, 0.9663, 0.9999 | 0.9887 [0.9405, 1.0369] |
| D2 | flank gap |mean q_L - mean q_R| | 0.0777, 0.0835, 0.0908 | 0.0840 [0.0678, 0.1002] |
| D3 | Spearman(beta_trust, -KL) overall | 0.4025, 0.4575, 0.3927 | 0.4176 [0.3309, 0.5043] |
| D3 | Spearman(-H(q), -KL) overall | -0.4254, -0.4298, -0.4354 | -0.4302 [-0.4427, -0.4176] |
| D3 | Spearman(beta_trust, -KL) within entropy strata | 0.2512, 0.3348, 0.2756 | 0.2872 [0.1804, 0.3940] |
| D3 | Spearman(-H(q), -KL) within entropy strata | -0.0275, -0.0303, -0.0219 | -0.0266 [-0.0372, -0.0159] |
| D3 | Spearman trust, 30% checkpoint rollouts | 0.4365, 0.4662, 0.3973 | 0.4334 [0.3476, 0.5191] |
| D3 | Spearman trust, 100% checkpoint rollouts | 0.3758, 0.4455, 0.3723 | 0.3979 [0.2952, 0.5005] |
| D3 | delusional-move rate, trust weights | 0.33%, 0.50%, 0.67% | 0.50% [0.09%, 0.91%] |
| D3 | delusional-move rate, stiff weights | 0.67%, 1.00%, 1.00% | 0.89% [0.41%, 1.37%] |
| D3 | delusional-move rate, uniform weights | 0.67%, 0.33%, 0.50% | 0.50% [0.09%, 0.91%] |
| D3 | KL(Q*||q_r), lowest-entropy tenth of rollouts | 1.094, 1.104, 1.104 | 1.101 [1.085, 1.116] |
| D3 | KL(Q*||q_r), other nine tenths | 0.557, 0.534, 0.566 | 0.552 [0.511, 0.594] |
| D3 | share of rollouts ending in a predicted flag capture | 0.252, 0.249, 0.252 | 0.251 [0.246, 0.256] |
| D3 | KL(Q*||q_r), rollouts ending in a predicted flag capture | 1.102, 1.113, 1.108 | 1.107 [1.094, 1.121] |
| D3 | KL(Q*||q_r), other rollouts | 0.402, 0.370, 0.418 | 0.397 [0.336, 0.458] |
| D5 | Unified-LIR effective win vs P0 | 0.466, 0.455, 0.527 | 0.483 [0.386, 0.580] |
| D5 | Unified-LIR effective win vs Ataraxos-1Step | 0.456, 0.485, 0.539 | 0.493 [0.389, 0.597] |
| D5 | Unified-LIR effective win vs Tabular-MCTS | 0.499, 0.474, 0.541 | 0.505 [0.420, 0.589] |

Delusional choices over the 3 seeds (600 decisions per seed): trust 9, stiff 16, uniform 9.

## Cost

`GPU-LOG.md` (rebuilt by `gpu/ledger.py` from the shared `logical-soup/GPU-LOG.md`), group `r4-e1` (cap $15), at the
$0.87/h list-price ceiling (spot is usually 60-70% cheaper; the Billing API is disabled, so the true price is not
verified). **Total $1.68.**

- Rule run: three spot L4s, 0.54 h, 0.47 h and 0.47 h, **$1.29**, deleted 19:29-19:31Z.
- Exploratory run: four spot L4s, 0.10-0.12 h each, **$0.39**. One of them (`e1-x103`) was preempted by GCE at
  19:46:40Z; its cost was entered as a COST-AUDIT line. One (`e1-x101`) was deleted before it finished booting. Two more
  create attempts failed on L4 stockouts at $0.

Every VM was deleted with `gpu-down.sh --only <vm> --no-sync` after its results were pulled (or by GCE on preemption).
`gcloud compute instances list` and `disks list` with `name~soup-gpu-e1-` were both empty at 2026-10-16T20:00:13Z.
CPU: data and oracle generation took 4 minutes on 4 cores.

## Deviations from the spec (all declared in PREREG-MS.md before the run)

1. Heuristic-only self-play data (tau in {0.5, 1, 2}), no MCTS games, so that the exact posterior stays computable.
2. Primary oracle P*_full (exact Bayes, including the behaviour-policy likelihood of the opponent's moves). The spec's
   constraint-only marginals P*_c are reported alongside. The gate fails against either one: the net's KL to P*_c is
   0.078 [0.069, 0.086].
3. The tape holds up to 6 plies, slot tokens carry a public move count, and theta_real has policy and value heads trained
   on real trajectories (rollouts need them).
4. theta_amp is an MLP adapter without LoRA.
5. Not run: rung 2 (6x6), Landmark Memory. D1-D5 were not run because the gate failed.
6. One dataset for all seeds. Seeds change initialisation, minibatch order and sampling.

The D1-D5 code (`eval_ms.py`, `agents_ms.py`) was crash-tested with a 60-step model before the prereg was hashed; those
numbers are meaningless and are not reported. D2, D3 and D5 were later run unchanged as exploratory work (section above).
D1 and D4 were never run.

## Self-audit

Three checks, all reproducible from saved files:

1. `report_ms.py` re-derives the D0 summaries and the exploratory D2, D3 and D5 summaries from the per-item arrays
   (`runs/s<seed>/eval/d*.npz`): D2 JSD and MSE from the saved consensus values, D3 Spearman and delusional rates from the
   saved rollouts, D5 win rates from the saved game results. The largest difference is 0 over 54 numbers
   (`report/audit.md`).
2. `audit_ms.py` independently reloads each seed's `real_100.pt` on CPU and rebuilds the inputs from `data/test.npz` with
   its own code (not `eval_ms.py`). It recomputes the 0-step KL to P*_full, the 5-known-slot NLL and the first-ply battle
   accuracy (9,744 battles). The largest difference from the GPU numbers is 7.8e-9 (`report/audit_independent.md`).
3. The oracle itself was checked against an independent brute-force replay of every hypothetical setup through
   `env.step` on 40 test positions: max abs error 5.6e-16 (P*_full) and 0 (P*_c) (`data/meta.json`).

## Files

All in `/home/agent/workspaces/mats/work/micro-stratego/`:

- Code: `msgame.py` (behaviour policy, exact posterior, Q*), `generate_data.py`, `model_ms.py`, `train_ms.py`,
  `eval_ms.py`, `agents_ms.py`, `report_ms.py`, `diag_d0.py`, `explore_summary.py`, `audit_ms.py`, `site_patch.py`,
  `gpu/*.sh`.
- Pre-registration: `PREREG-MS.md`, `PREREG-MS.sha256`, `PREREG-MS-AMENDMENTS.md`.
- Data: `data/` (splits, games, records, oracle sets, `meta.json`). Runs: `runs/s101`, `runs/s102`, `runs/s103` (all
  checkpoints, `train_meta.json`, logs, `eval/d0*.json|npz`, and the exploratory `eval/d2|d3|d5.json|npz`).
- Report: `report/verdicts.csv`, `report/tables.md`, `report/summary.json`, `report/audit.md`,
  `report/audit_independent.md`, `report/d0_diag.json`, `report/d0_by_t.csv`, `report/d0_by_t.png`,
  `report/train_curves.csv`, `report/train_curves.png`, `report/cal_fits.json`, `report/exploratory_summary.md|csv`.
  GPU ledger: `GPU-LOG.md`.

### Audit table 1 (`report/audit.md`)

| quantity | seed | json summary | re-derived from npz | abs diff |
|---|---|---|---|---|
| D0 kl0_full | 101 | 0.023062 | 0.023062 | 0.00e+00 |
| D0 battle acc | 101 | 1.000000 | 1.000000 | 0.00e+00 |
| D0 5-known NLL | 101 | 0.000161 | 0.000161 | 0.00e+00 |
| D0 kl0_full | 102 | 0.025361 | 0.025361 | 0.00e+00 |
| D0 battle acc | 102 | 1.000000 | 1.000000 | 0.00e+00 |
| D0 5-known NLL | 102 | 0.000082 | 0.000082 | 0.00e+00 |
| D0 kl0_full | 103 | 0.023247 | 0.023247 | 0.00e+00 |
| D0 battle acc | 103 | 1.000000 | 1.000000 | 0.00e+00 |
| D0 5-known NLL | 103 | 0.000071 | 0.000071 | 0.00e+00 |
| D2 JSD mix | 101 | 0.005989 | 0.005989 | 0.00e+00 |
| D2 MSE mix | 101 | 0.004917 | 0.004917 | 0.00e+00 |
| D2 JSD ataraxos_mmd | 101 | 0.001601 | 0.001601 | 0.00e+00 |
| D2 MSE ataraxos_mmd | 101 | 0.009276 | 0.009276 | 0.00e+00 |
| D2 JSD kc_dedup | 101 | 0.000626 | 0.000626 | 0.00e+00 |
| D2 MSE kc_dedup | 101 | 0.008210 | 0.008210 | 0.00e+00 |
| D2 JSD attn_L3 | 101 | 0.001968 | 0.001968 | 0.00e+00 |
| D2 MSE attn_L3 | 101 | 0.004940 | 0.004940 | 0.00e+00 |
| D2 JSD mix | 102 | 0.006068 | 0.006068 | 0.00e+00 |
| D2 MSE mix | 102 | 0.004869 | 0.004869 | 0.00e+00 |
| D2 JSD ataraxos_mmd | 102 | 0.001674 | 0.001674 | 0.00e+00 |
| D2 MSE ataraxos_mmd | 102 | 0.008073 | 0.008073 | 0.00e+00 |
| D2 JSD kc_dedup | 102 | 0.000963 | 0.000963 | 0.00e+00 |
| D2 MSE kc_dedup | 102 | 0.006894 | 0.006894 | 0.00e+00 |
| D2 JSD attn_L3 | 102 | 0.001375 | 0.001375 | 0.00e+00 |
| D2 MSE attn_L3 | 102 | 0.003821 | 0.003821 | 0.00e+00 |
| D2 JSD mix | 103 | 0.006887 | 0.006887 | 0.00e+00 |
| D2 MSE mix | 103 | 0.005372 | 0.005372 | 0.00e+00 |
| D2 JSD ataraxos_mmd | 103 | 0.001099 | 0.001099 | 0.00e+00 |
| D2 MSE ataraxos_mmd | 103 | 0.009469 | 0.009469 | 0.00e+00 |
| D2 JSD kc_dedup | 103 | 0.000451 | 0.000451 | 0.00e+00 |
| D2 MSE kc_dedup | 103 | 0.008042 | 0.008042 | 0.00e+00 |
| D2 JSD attn_L3 | 103 | 0.001885 | 0.001885 | 0.00e+00 |
| D2 MSE attn_L3 | 103 | 0.004595 | 0.004595 | 0.00e+00 |
| D3 Spearman trust | 101 | 0.402494 | 0.402494 | 0.00e+00 |
| D3 Spearman stiff | 101 | -0.425372 | -0.425372 | 0.00e+00 |
| D3 delusional stiff | 101 | 0.006667 | 0.006667 | 0.00e+00 |
| D3 delusional trust | 101 | 0.003333 | 0.003333 | 0.00e+00 |
| D3 Spearman trust | 102 | 0.457474 | 0.457474 | 0.00e+00 |
| D3 Spearman stiff | 102 | -0.429755 | -0.429755 | 0.00e+00 |
| D3 delusional stiff | 102 | 0.010000 | 0.010000 | 0.00e+00 |
| D3 delusional trust | 102 | 0.005000 | 0.005000 | 0.00e+00 |
| D3 Spearman trust | 103 | 0.392738 | 0.392738 | 0.00e+00 |
| D3 Spearman stiff | 103 | -0.435443 | -0.435443 | 0.00e+00 |
| D3 delusional stiff | 103 | 0.010000 | 0.010000 | 0.00e+00 |
| D3 delusional trust | 103 | 0.006667 | 0.006667 | 0.00e+00 |
| D5 vs P0 | 101 | 0.466250 | 0.466250 | 0.00e+00 |
| D5 vs Ataraxos-1Step | 101 | 0.456250 | 0.456250 | 0.00e+00 |
| D5 vs Tabular-MCTS | 101 | 0.498750 | 0.498750 | 0.00e+00 |
| D5 vs P0 | 102 | 0.455000 | 0.455000 | 0.00e+00 |
| D5 vs Ataraxos-1Step | 102 | 0.485000 | 0.485000 | 0.00e+00 |
| D5 vs Tabular-MCTS | 102 | 0.473750 | 0.473750 | 0.00e+00 |
| D5 vs P0 | 103 | 0.527500 | 0.527500 | 0.00e+00 |
| D5 vs Ataraxos-1Step | 103 | 0.538750 | 0.538750 | 0.00e+00 |
| D5 vs Tabular-MCTS | 103 | 0.541250 | 0.541250 | 0.00e+00 |

Largest absolute difference: 0.00e+00 over 54 re-derived numbers.

### Audit table 2 (`report/audit_independent.md`)

| seed | quantity | d0.json | independent CPU recomputation | abs diff |
|---|---|---|---|---|
| 101 | 0-step KL to P*_full | 0.023062 | 0.023062 | 2.9e-09 |
| 101 | 5-known NLL | 0.000161 | 0.000161 | 2.4e-09 |
| 101 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 101 | battles scored | 9744 | 9744 | |
| 102 | 0-step KL to P*_full | 0.025361 | 0.025361 | 7.8e-09 |
| 102 | 5-known NLL | 0.000082 | 0.000082 | 3.5e-09 |
| 102 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 102 | battles scored | 9744 | 9744 | |
| 103 | 0-step KL to P*_full | 0.023247 | 0.023247 | 4.5e-09 |
| 103 | 5-known NLL | 0.000071 | 0.000071 | 2.4e-09 |
| 103 | battle accuracy ply 0 | 1.000000 | 1.000000 | 0.0e+00 |
| 103 | battles scored | 9744 | 9744 | |

Largest absolute difference: 7.8e-09 (GPU float32 evaluation vs CPU recomputation with float64 softmax).
