# PREREG-R5-E2: R5-E2 (E-MS-6x6) 6x6 8-piece Barrage micro-Stratego with coupled tactics and preference-guided search

Pre-registration for experiment R5-E2 of `/home/agent/workspaces/mats/outbox/experiments-round-5-spec.md` (section 2;
spec file sha256 7958f340afa8ed473bc8d6e20609be75db06eedda5cccb94e5939a1a36340b8f). Executor: an Opus 5.5 subagent of
eurisko (scientific lead), assigned by opus-mats (agent mail #3688). Written 2026-10-17T03:24:55Z, BEFORE any GPU launch for this
experiment. Work done before this file was hashed, all on CPU: the new 6x6 code under `r5/`, the data and oracle sets
(`r5/generate6.py`; bulky files in `/tmp/r5e2/data/`), the data-only pre-check in section 3, and code review. No model
of this experiment had been trained and no rule quantity had been computed. The SHA-256 of this file is recorded in
`PREREG-R5-E2.sha256` (hash, file, UTC time) before the first launch; any later change goes to
`PREREG-R5-E2-AMENDMENTS.md` with its UTC time and reason, and no amendment may change a rule, threshold, scoring set or
hyperparameter after a test number has been seen.

Budget: GPU group `r5` (cap $22 shared with R5-E1; the r5 guard deletes VMs older than 4.2 h and kills the youngest at $21
estimated), own ceiling $9 at the $0.87/h list-price ceiling, own ledger `r5/GPU-LOG-R5-E2.md`. Plan: one spot L4 per seed
(3 VMs), launched only through `logical-soup/infra/gpu-run.sh` with GPU_BUDGET_GROUP=r5, GPU_GROUP_BUDGET_USD=22, run ids
`r5e2-*`, GPU_RUN_TIMEOUT <= 150 min. Seeds 101, 102, 103. Code: `/home/agent/workspaces/mats/work/micro-stratego/r5/`
(the R4-E1 code one level up is not modified).

## 1. Pass/fail rules, copied verbatim from section 2.3 of the spec

### 2.3 Pre-Registered Pass/Fail Rules (`R5-E2`)
- **`E2-D0` (Gate — Calibrated on a 5-min pre-check before locking):**
  On 3 seeds (`101, 102, 103`), evaluated on held-out setups:
  1. 1-step battle outcome accuracy $\ge 0.97$;
  2. 7-known-slot permutation NLL $\le 0.05$ nats;
  3. **Genuine 0-Step Belief Gap + Rollout Signal:** 0-step belief $\mathrm{KL}(P^*_{\text{full}} \Vert P_0) \ge \mathbf{0.05\text{ nats/slot}}$ (total $\ge 0.40$ nats across the 8 slots; note: if the 4-layer `d=128` net gets $<0.05$ at step 16k, use the pre-registered `d=96, 3-layer` capacity checkpoint that passes (1) and (2) while leaving $\ge 0.05$ nats/slot for search) **AND** 16 unskewed rollouts reduce belief KL or action-value MSE by $\ge 10\%$ on the calibration split.
- **`E2-D1` (Preference-Edge Search & Policy-Independent 8-Slot Beliefs):**
  When search is driven by Left-Flank Preference $U_{\text{Left}}$ (80% of rollouts on cols `0..2`) vs. Right-Flank Preference $U_{\text{Right}}$ (80% of rollouts on cols `3..5`):
  1. **Non-Collapsed Keys:** Mean key cosine within-flank minus across-flank is **$\ge 0.25$** (95% CI $>0$; fixing `R4-E1`'s `0.99 vs 0.99`);
  2. **Policy-Independent Merged 8-Slot Belief & Value:** `attn_L3` cuts policy-sensitivity $\mathrm{JSD}(\mu^*(U_{\text{Left}}), \mu^*(U_{\text{Right}}))$ by **$\ge 2.5\times$** compared to `mix`, while achieving lower MSE/KL to the unskewed $K=128$ reference than `mix`, `ataraxos_mmd`, and `kc_dedup`.
- **`E2-D2` (Self-Trust $\theta_{\text{trust}}$ vs. Entropy on $6\times 6$):**
  Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) $\ge +0.25$ overall and exceeds Spearman($-H(q_r), -\mathrm{KL}$) by $\ge +0.20$ overall and $\ge +0.15$ within entropy strata.
- **`E2-D3` (Zero-Retraining Preference Swap $U_{\text{Std}} \leftrightarrow U_{\text{Miner}}$ & Head-to-Head Play):**
  Swapping the utility edge $U_{\text{Std}} \leftrightarrow U_{\text{Miner}}$ at test time shifts action selection toward higher Miner-capture utility on $\ge 75\%$ of boards with `0.0000` change to $\theta_{\text{real}}$'s beliefs; and `Unified-LIR` (`stiff_voi` + $\theta_{\text{trust}}$ + `attn_L3`) achieves $> 52\%$ effective win rate vs. `P0` (400 duplicate-paired games/seed).

## 2. Operational definitions (how each rule above is measured and decided)

Conventions (as R4-E1's PREREG-MS.md). Each statistic is computed per seed (101, 102, 103) on the test split (the 1,008
held-out test setups) unless the rule names the calibration split. The decision uses the 3-seed mean and the seed-t 95%
interval (df = 2, t = 4.303) of the per-seed values. A rule with an interval requirement passes only if the mean meets
the threshold AND the interval excludes the null (0 for differences and reductions; 0.5 for win rates and for the D3 shift
rate). The D0 gate is judged on 3-seed means (R4-E1's gate convention); its intervals are reported. Ratios are tested
through the paired difference of their two terms (the ratio's mean must meet the bar, the difference's interval must
exclude 0). Relative reduction = 1 - ours / theirs.

### E2-D0 (gate)
- (1) Battle accuracy: for every test position whose first real ply is a battle, all opponent slots materialised with
  their true types, the argmax of theta_real's outcome head at A_0 equals the real outcome. Pass: mean >= 0.97.
- (2) 7-known-slot NLL: on the 2,000 D0 test positions (t in [6, 36], one per game), for each hidden opponent slot, the
  other seven slots set to their true types, -log q(true type). Mean over (position, hidden slot). Pass: <= 0.05 nats.
- (3a) 0-step gap: P_0 = theta_real's belief with nothing materialised. Per position, the mean over hidden opponent
  slots of KL(P*_full,s || P_0,s); mean over the 2,000 D0 test positions. Pass: >= 0.05 nats per hidden slot. The spec's
  "total >= 0.40 nats across the 8 slots" equals 8 x 0.05 only when all 8 slots are hidden; the decision uses the
  per-hidden-slot mean (the R4-E1 convention for its 0.12 bar), and the total over hidden slots is reported.
- Capacity rule (the spec's note): BOTH architectures are trained for every seed with the same 16k-step schedule and the
  same downstream pipeline: primary = 4 layers, d = 128 (1,001,644 parameters); fallback = 3 layers, d = 96 (492,268).
  Selection uses the CALIBRATION split only (1,000 calibration positions with P*_full, t in [6, 36]; battle accuracy and
  7-known NLL on calibration records), 3-seed means: if the primary meets (1), (2) and (3a) there it is THE model for
  every rule; else if the fallback meets all three it is; else the gate fails (reported for the primary). The selected
  architecture's TEST-split values then decide (1), (2), (3a). The other architecture's numbers are reported only as
  secondary, never as rule outcomes. (Selecting on the calibration split keeps the test split out of the selection.)
- (3b) Rollout signal, calibration split, selected architecture. (i) Belief: on the 1,000 calibration D0 positions, 16
  cone rollouts (definition in section 4) whose root moves are drawn uniformly from the legal moves (unskewed); the
  16-rollout consensus = per hidden slot softmax(0-step logits + mean residual over the rollouts whose cone contains the
  slot) (0-step belief if none) - the same coverage estimator as the K = 128 reference of D1; reduction = 1 -
  KL(P*_full || consensus) / KL(P*_full || P_0) (means over hidden slots, then positions). (ii) Action value: on 200
  calibration positions with Q* (64 simulator playouts per legal action), 16 rollouts round-robin over the legal moves
  (depth 4); per action the mean leaf effective win (win + 0.5 draw) vs theta_real's own 0-step action value; reduction
  = 1 - MSE_16 / MSE_0 against Q*_eff over the actions with >= 1 rollout. (3b) passes if the 3-seed mean reduction is
  >= 0.10 for (i) or for (ii).
- Gate = (1) and (2) and (3a) and (3b), with a selected architecture. If the gate fails, D1-D3 are NOT reported as rule
  tests. The pipeline still computes them; they are reported only as EXPLORATORY, labelled as such (as R4-E1 did).

### E2-D1 (preference-edge search, policy-independent 8-slot beliefs)
- Positions: 500 two-flank test positions (t in [4, 36], the player to move has legal moves on both flanks; flank of a
  move = canonical from-column 0..2 Left, 3..5 Right).
- Rollouts: cone rollouts, depth 4, stiff_voi cone order, root move uniform within its flank. Per position two
  independent pools of 24 Left + 24 Right rollouts. U_Left takes n_L ~ Binomial(24, 0.8) Left rollouts and 24 - n_L
  Right rollouts from pool 1 (80% of rollouts on cols 0..2); U_Right takes n_L ~ Binomial(24, 0.2) from pool 2. The
  unskewed K = 128 reference uses a third pool of 64 Left + 64 Right rollouts: belief = per hidden slot
  softmax(0-step logits + mean residual over the reference rollouts whose cone contains the slot) (0-step belief if
  none); value = mean leaf effective win of the 128 rollouts.
- Aggregators (formulas in section 4) are fitted on the calibration split only, against the same kind of reference.
- (1) Key gap: in each policy's bag of 24, mean cosine <k_r, k_b> over pairs on the same flank (r != b) minus the mean
  over pairs on different flanks; mean over positions whose bag holds both flanks, then over the two policies.
  Pass: 3-seed mean >= 0.25 and interval > 0.
- (2) Policy sensitivity: belief JSD = mean over positions of the mean over hidden slots of the 6-way
  JSD(mu*_s(U_Left), mu*_s(U_Right)); value JSD = Bernoulli JSD of the two merged effective-win values. "Cuts ... by
  >= 2.5x compared to mix": per seed JSD_mix / JSD_attn_L3; the 3-seed mean ratio must be >= 2.5 and the interval of
  JSD_mix - JSD_attn_L3 must exclude 0, for BOTH the belief JSD and the value JSD (the rule names the "merged 8-slot
  belief & value"). Accuracy ("lower MSE/KL to the unskewed K = 128 reference"): belief KL(ref || mu*) (mean over hidden
  slots, mean over the two policies) and value MSE (mean over the two policies of (mu*_v - ref_v)^2); attn_L3 must be
  lower than EACH of mix, ataraxos_mmd and kc_dedup on BOTH (6 paired comparisons, each interval of the difference > 0).
  D1 (2) passes only if all of these hold. D1 passes if (1) and (2) pass.

### E2-D2 (theta_trust vs entropy)
- Positions: 240 Q* test positions, one per game: 120 general (t in [4, 36]) and 120 Miner-contact (the mover's Miner is
  alive within Chebyshev distance 2 of an opponent piece; a public-information filter).
- Rollouts: every legal action x depth d in 1..6 x checkpoint {30%, 100%} x 2 repeats, random cone order (R4-E1's D3
  protocol); q_r = the rollout's leaf win/draw/loss claim; y = -KL(Q*(s,a) || q_r), Q* from 64 simulator playouts per
  action from the exact joint posterior over (opponent setup, opponent tau). Predictors: beta_r (theta_trust) and
  -H(q_r). Within strata = mean of the within-stratum Spearman over 10 equal-count strata of H(q_r).
- Pass: Spearman(beta, y) >= 0.25 (interval > 0) AND Spearman(beta, y) - Spearman(-H, y) >= 0.20 overall (interval > 0)
  AND the within-strata difference >= 0.15 (interval > 0).

### E2-D3 (zero-retraining utility swap, head-to-head play)
- (a) On the same 240 Q* positions the Unified-LIR agent (K = 16) chooses an action under U_Std (effective win) and,
  with no retraining, under U_Miner (mean predicted Miner events of the action's rollouts over their 4 plies: +1 enemy
  Miner captured, -1 own Miner lost; ties broken by l0). True Miner utility Q*_miner(s, a) = mean Miner events over the
  first 4 plies of the same 64 simulator playouts. A board is relevant if max_a Q*_miner(s, a) - Q*_miner(s, a_Std) >=
  0.05 (U_Std's choice leaves room to shift); a relevant board shifts if Q*_miner(s, a_Miner) > Q*_miner(s, a_Std).
  Pass: shift rate over relevant boards, 3-seed mean >= 0.75 with interval lower bound > 0.5, AND theta_real's
  parameter hash identical before and after both runs, AND the max |change| of theta_real's 0-step beliefs on these
  positions is exactly 0.0000.
- (b) 400 games per seed: 200 setup pairs drawn from the test setups, each played twice with colours swapped
  (duplicate-paired); Unified-LIR (U_Std, K = 16) vs P0 (argmax of theta_real's 0-step action value + theta_amp
  correction). Effective win = (wins + 0.5 draws) / 400. Pass: 3-seed mean > 0.52 with interval lower bound > 0.5.
- D3 passes if (a) and (b) pass.

## 3. Environment and data (generated before this file was hashed; `r5/generate6.py`, `/tmp/r5e2/data/meta.json`)

- Board (`r5/env6.py`, new file; the 4x4 code is untouched): 6x6, lakes (2,2) and (3,3); 8 pieces per side (Flag,
  2 Bombs, 2 Scouts, Miner, General, Marshal) on 8 FIXED public squares per side - Red front (1,0) (1,1) (1,4) (1,5),
  back (0,1) (0,2) (0,3) (0,4); Blue = the 180-degree rotation - so the hidden setup is the assignment of the multiset to
  the 8 slots: 8!/(2!2!) = 10,080 setups per side. Rules as `env.py` (Scouts move any distance, a move longer than 1
  reveals a Scout; battles reveal both pieces). Limit 60 plies (draw).
- Behaviour policy (`r5/msgame6.py`), the coupled 2-ply score of the spec: battle value against the public marginals;
  progress 0.3 per row; Scout +0.6 advancing onto an open lane; Miner +0.8 closing in on an unmoved opponent back-row
  piece when a friendly General or Marshal is within Chebyshev distance 2 of the target square; -0.8 for leaving the
  orthogonal neighbourhood of the own Flag unless another friendly Bomb or Marshal stays next to it; reply term -0.7 x
  threat (max over adjacent opponent pieces of P_public(mobile and beats k) x value of k), halved if a friendly General
  or Marshal defends the square. Softmax at tau ~ U{0.5, 1, 2} per player per game. Its vectorised likelihood matched an
  independent plain-loop implementation on 1,464 (state, setup) pairs (max |score difference| 4.4e-16).
- Oracle: P*_full(Z | h_t) = exact Bayes posterior over the opponent's 10,080 setups: uniform prior x public-constraint
  consistency x the behaviour likelihood of every opponent move, tau unknown (uniform mixture). P*_c = the
  constraint-only marginals. Verified against an independent brute-force replay (every hypothetical setup replayed
  through env6.step, public observations compared ply by ply, opponent move probabilities from the plain-loop policy) on
  30 test positions (max abs error recorded in meta.json). Q*(s, a): 64 simulator playouts per legal action from the
  exact joint posterior, own tau ~ U{0.5, 1, 2}, behaviour policy to the end; Q*_miner over the first 4 plies of the same
  playouts. Oracle and simulator are used ONLY for scoring and gates (D0, D0-3b, D2, D3a); no training loss, fitted
  scale, aggregator or agent reads them.
- Setup split (seed 20261017): 8,064 train / 1,008 calibration (val) / 1,008 test setups, disjoint, same split for both
  colours. Self-play games: 60,000 train / 5,000 calibration / 3,000 test (1,979,305 / 164,321 / 98,816 positions; mean length 33.0 / 32.9 / 32.9 plies; train outcomes Red 39.4%, Blue 39.6%, draw 20.9%).
- Scoring sets: D0 test 2,000 positions and D0 calibration 1,000 positions (t in [6, 36], one per game); Q* test 240
  positions (section 2), Q* calibration 200 positions (t in [4, 36]); D1 500 two-flank test positions.
- Pre-check before locking (the spec's "calibrated on a 5-min pre-check before locking", interpreted as data-only, no
  model): on 460 mid-game positions of 40 behaviour-policy games KL(P*_full || P*_c) per hidden slot was 0.037 (t 4-9),
  0.085 (t 10-15), 0.122 (t 16-23), 0.147 (t 24-39), mean 0.102 over t in [6, 30] (R4-E1's 4x4 data: 0.075): the
  moves carry evidence beyond the public constraints, so a 0.05 gap is possible in principle. No threshold was changed.

## 4. Model, training, rollouts, aggregators, agents (`r5/model6.py`, `r5/train6.py`, `r5/agents6.py`)

- theta_real: the R4-E1 any-order tape model with 6x6 sizes (57 tokens: 36 squares, 8 opponent slots, turn, 6 ply pairs;
  policy head 36 x 36 = 1,296 from-to logits). Trained on TRAIN trajectories only: CE on masked hidden slots (each hidden
  slot revealed with probability u ~ U(0, 1)), CE on each real ply's move, outcome and revealed types, CE on the game
  result. AdamW lr 1e-3, wd 0.01, batch 1024, 16,000 steps, warmup 300, cosine, clip 1, bf16 autocast for these steps
  (all other computation fp32/TF32). Checkpoints at 30% (step 4,800) and 100%. The root encoding is R4-E1's (current
  board, per-slot public moved / moved-long flags, move count, revealed bit); the root does not contain the move history.
- Cone rollout (the spec's fix 1): the cone = the 3 hidden opponent slots nearest (Chebyshev) to the root move's target
  square (ties: distance to the from-square, then slot index); only these are materialised, by any-order unmasking from
  theta_real's belief head (stiff_voi = lowest H(q_s) + e_hat_s first, or random); all other hidden slots stay MASK.
  Value: 4 plies (or d) inside theta_real from the root move (later moves from the policy head, outcomes from the
  transition head), leaf win/draw/loss q_r. Key: k_r = normalise(W_K2 LayerNorm(sum_j a_j pos_j)) with a = the readout
  token's last-layer head-averaged attention renormalised over the source set = the cone's slot tokens + the squares the
  rollout's moves touch (R4-E1 used every hidden slot).
- Belief residual (the spec's fix 2, "4-ply consistency"): for each cone slot s and type k, the log-likelihood of the
  opponent's REAL moves in the last 4 real plies (a tape rooted at t0 = max(t - 4, t mod 2), real plies teacher-forced,
  theta_real's policy head) with Z_s = k and the other cone slots at their sampled values, minus the same with Z_s
  masked; log mu_r[s, :] = log softmax(log q_real(Z_s | root at t0, other cone slots) + that log-likelihood ratio);
  dl_r[s] = log mu_r[s] - log P_0,s (clamped to +-8), zero outside the cone. Root-value residual dl^v_r = logit(e_r) -
  theta_real's 0-step root value logit. Everything comes from theta_real's own heads and the public history; no oracle.
- theta_trust: slot-level head (stiff_voi) and rollout-level head as R4-E1 (stop-grad features incl. depth, transition
  entropies, 2-path disagreement over the cone, |l_r - l0(a)|, H(q_r), q_r, terminal flag), fitted on CAL against the
  excess surprise of the claim vs the realised slot / result; beta_r = 1 / (e_hat + 0.05).
- theta_amp (for P0 and the agents' anchors): stop-grad 2-layer MLP correction to theta_real's 0-step action value,
  distilled (LIR, chi_real = 0, theta_real frozen and hash-checked) on 6,000 TRAIN roots from the mean of 8 cone rollouts
  per legal action. No fused arm (R5-E2 has no parameter-mask rule).
- Aggregators over a bag (l0_b = 0-step log-beliefs, l0_v = 0-step root value logit): mix = mean over rollouts of
  softmax(l0_b + dl_r) and of e_r (naive average; a rollout's claim outside its cone is the 0-step belief);
  ataraxos_mmd: l* = l0 + eta sum_r beta_r dl_r (eta_b for beliefs, eta_v for the value); kc_dedup: l* = l0 + u sum_r
  (beta_r / c_r) dl_r with c_r = sum_b max(0, <k_r, k_b>)^2 (u_b, u_v); attn_L3: 3 pre-LN residual self-attention layers
  (d 64, 4 heads) over z_r = [k_r, W_v dl_r (48), dl^v_r, log beta_r, log c_r], outputting non-negative per-rollout
  weights for each of the 8 slots and for the value: l*_b[s] = l0_b[s] + sum_r w_r[s] dl_r[s], l*_v = l0_v + sum_r w_r[8]
  dl^v_r. Fitting (CAL, 1,500 two-flank roots, pools 32 + 32, reference 64 + 64, bags of 24 with p_L ~ Beta(0.5, 0.5),
  1,500 Adam steps each): loss = KL(ref_b || mu*_b) (mean over hidden slots) + BCE(l*_v, ref_v), joint over the 48
  belief logits and the value. W_K2 is fitted through kc_dedup's fit and then frozen (attn_L3 reads detached keys).
- Agents: P0 = argmax_a l0_amp(a). Unified-LIR = R4-E1's 4 waves of K/4 rollouts by key-space PUCT (c = 1, theta_real's
  policy prior, visit counts replaced by key-deduplicated counts), stiff_voi cones, theta_trust beta_r, per-action
  attn_L3 value consensus l*_a = l0_amp(a) + sum_r w_r[value] (logit(e_r) - l0_amp(a)) over the action's rollouts,
  argmax under U. In live games the 4-ply window comes from the agent's own game history.

## 5. Arms, seeds, metrics and stop rules

- Arms: aggregators mix / ataraxos_mmd / kc_dedup / attn_L3 (D1); predictors beta_trust / -H (D2); utility edges
  U_Std / U_Miner and agents Unified-LIR / P0 (D3); architectures primary / fallback (capacity rule only).
- Seeds 101, 102, 103: model initialisation, minibatch order and every search / evaluation sample. One dataset.
- Per-item outputs (synced back as CSV/JSON/PNG only): d0_<split>_positions.csv, d0_<split>_nll7.csv,
  d0_<split>_battles.csv, d0b_*.csv, d1_positions.csv, d2_rollouts.csv, d3a_positions.csv, d3b_games.csv, plus a JSON
  summary per part, train_meta.json, train_curves.csv and audit_vm.json (an independent recomputation of the D0 headline
  numbers from the checkpoint with separate code, run on each VM before it is deleted).
- Stop rules. (i) Gate fail -> D1-D3 only exploratory (labelled). (ii) Budget: before every launch my own ledger
  (r5/GPU-LOG-R5-E2.md, $0.87/h ceiling) plus the next VM's reservation must stay <= $9; the r5 group guard
  (deletes r5 VMs older than 4.2 h, kills the youngest at $21 group-estimated) and GPU_RUN_TIMEOUT <= 150 min apply.
  (iii) Preemption or a dead VM: the seed is relaunched (at most 2 relaunches per seed, STANDARD provisioning after the
  first preemption); no partial result of a lost run is used. (iv) A crash is fixed (bug fix only) and the seed rerun,
  logged in PREREG-R5-E2-AMENDMENTS.md. (v) No change to any rule, threshold, scoring set or hyperparameter after any
  test number is seen; amendments carry a UTC time and a reason. (vi) A short GPU timing smoke (seed 999, ~300 training
  steps, smoke-sized stages, outputs discarded, never reported as a result) may run after this file is hashed and
  before the three seed runs, to set the run timeout and catch GPU-only bugs.

## 6. Deviations from the spec and interpretations (declared before running)

1. Fixed public initial squares (the spec's "choose 8 occupied squares" read as a fixed choice; otherwise there would be
   495 x 10,080 setups, not 10,080).
2. Limit 60 plies (40 gave 38% draws, 60 gives 21%), chosen on behaviour-policy games before any model existed.
3. Behaviour-policy constants not given by the spec (Flag-neighbour penalty 0.8, reply weight 0.7, progress 0.3) are
   mine; Miner +0.8 and Scout +0.6 are the spec's; "2-ply tactical reply term" = the threat term above (no search).
4. The "4-ply consistency" check is the window likelihood ratio of section 4 (real opponent moves in the last 4 plies,
   scored by theta_real's policy head under the cone hypothesis).
5. K = 128 reference and mix as defined in sections 2 and 4 (the spec does not define how the reference pools an 8-slot
   belief; a plain average would only measure how often a slot was in some cone).
6. attn_L3 outputs per-slot weights (8 + 1) instead of one scalar per rollout; d 64, 4 heads (R5 section 1 sizes).
7. ataraxos_mmd and kc_dedup have separate belief and value scales.
8. The capacity rule's selection is made on the calibration split, both architectures always run.
9. No Ataraxos-1Step / Tabular-MCTS opponents and no fused arm: R5-E2's rules do not use them.
10. D2 uses R4-E1's D3 protocol (mixed 30% / 100% checkpoints, depths 1..6, random cone order, 2 repeats).
11. Heuristic-only self-play (the exact posterior needs a closed-form move likelihood); split 8,064 / 1,008 / 1,008.
12. Only CSV/JSON/PNG files are synced back; checkpoints stay on the VMs and are deleted with them. The CPU-side audit
    re-derives the headline numbers from the per-item CSVs; audit_vm.json is the on-VM independent recomputation.
13. bf16 autocast only for theta_real's training steps; fp32 / TF32 elsewhere.

## 7. Reporting

Every number in RESULTS-R5-E2.md comes from the synced per-seed files through `r5/report6.py`, which also re-derives the
headline numbers from the per-item CSVs (report/selfcheck.csv). AUDIT-R5-E2.md is a separate pass. Negative and null
results are reported as such; failures stay failures; no tuning after test numbers are seen.
