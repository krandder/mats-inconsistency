# PREREG-R7-E3: R7-E3 (E-MS-TRUST-SEARCH) 6x6 Stratego: pure salience keys K3/K3c, bottom-tail trust gating, active low-trust rollout replacement, mixed-reliability bags

Pre-registration for experiment R7-E3 of `/home/agent/workspaces/mats/outbox/experiments-round-7-spec.md` (section 2;
spec sha256 cbece88d738601336e5a1fe70a71de7805cfd7087455f39dc83249a35aa24e88). Executor: an Opus 5.5 subagent of
eurisko, assigned by opus-mats (agent mail #3717). Written 2026-10-17, BEFORE any launch for this experiment (the hash
time is in `PREREG-R7-E3.sha256`). `outbox/experiments-round-7-amendments.md` (09:24Z) exists: it amends R7-E2 (readings 4-7) and R7-E1 (reading
9) only and contains nothing for R7-E3. Work
done before hashing: the code in `r7/` (reusing `r5/` and `r6/` unchanged), a check that the R5-E2 checkpoints load and
match their recorded parameter hashes, two small CPU diagnostics and a CPU crash smoke (all disclosed in section 6,
item 11). The SHA-256 of this file goes into `PREREG-R7-E3.sha256` (hash, file, UTC time) before the first launch;
later changes go to `PREREG-R7-E3-AMENDMENTS.md` with time and reason.

Budget: group `r7` (GPU_BUDGET_GROUP=r7, GPU_GROUP_BUDGET_USD=6.5, shared with R7-E1; the guard deletes r7 VMs older
than 4.2 h and stops launches at $6 estimated); own ceiling $1.50 (the spec's ranked-table cap for R7-E3; estimate
~$0.35), run ids `r7e3-*`, launches only through `logical-soup/infra/gpu-run.sh`. Seeds 101, 102, 103. Code:
`/home/agent/workspaces/mats/work/micro-stratego/r7/`.

## 1. Decision rules, copied verbatim from section 2.2 of the spec

### 2.2 Pre-Registered Decision Rules for `R7-E3`

- **`E3-D0` (Gate):** Inherited from `R6-E3` (same checkpoints and marginal window rollouts: battle accuracy $\ge 0.99$, 7-known NLL $\le 0.01$, 16 marginal rollouts cut KL vs. $t-4$ anchor by $\ge 40\%$; current-turn one-pass KL $\sim 0.048$ reported alongside).
- **`E3-D1` (Pure Salience Keys `K3` / `K3c` & Policy-Independent Multi-Slot Aggregation):** Across the 3 seeds on the 500 two-flank test positions:
  1. Either `K3` or `K3c` achieves within-flank minus across-flank key cosine gap **$\ge 0.28$** (with 95% seed-t CI lower bound $> 0.25$).
  2. `attn_L3` (with that key head) achieves Left/Right preference belief $\mathrm{JSD}$ at least **$5.0\times$ lower** than both `scaled_mix` and `ataraxos_mmd`, and achieves strictly lower 8-slot $\mathrm{KL}(P^*_{\text{full}} \Vert \mu)$ than `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all paired 95% seed-t CIs $> 0$).
- **`E3-D2` (Active Trust Gating & Pruning vs. Stiffness and Uniform Pooling):** Across the 3 seeds on the 240 $Q^*$ test positions $\times$ legal actions (mixed-reliability bags):
  1. **Within-Pair & Overall Ranking:** Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) is **$\ge +0.30$** overall and **$\ge +0.25$** within $(s, a)$ pairs, and exceeds Spearman($-H(q_r), -\mathrm{KL}(Q^* \Vert q_r)$) by **$\ge +0.55$** overall and **$\ge +0.20$** within 10 equal-count entropy strata (all paired 95% CIs $> 0$).
  2. **Action-Value MSE Reduction:** Both **`trust_gate`** (or `trust_pool`) and **`trust_resample`** reduce action-value MSE against $Q^*(s, a)$ by **$\ge 20\%$ relative to `stiff_pool` ($\gamma=2$)** and achieve strictly lower MSE than **`unif_pool`** (paired 95% seed-t CIs of $\mathrm{MSE}(\text{unif\_pool}) - \mathrm{MSE}(\text{trust\_gate}) > 0$ and $\mathrm{MSE}(\text{unif\_pool}) - \mathrm{MSE}(\text{trust\_resample}) > 0$).

## 2. Operational definitions (how each rule above is measured and decided)

Conventions as in PREREG-R6-E3.md. Each statistic is computed per seed (101, 102, 103) on held-out data; the decision
uses the 3-seed mean and the seed-t 95% interval (df = 2, t = 4.303) of the per-seed values. A rule with an interval
requirement passes only if the mean meets the bar AND the interval excludes the null (0 for Spearman correlations,
differences and reductions; 0.25 where the rule names a lower bound of 0.25). Every "exceeds by" difference needs its
interval > 0. Ratios are tested through the paired difference of their two terms (the 3-seed mean of the per-seed ratio
must meet the bar, the interval of the difference must exclude 0). The E3-D0 gate is judged on 3-seed means (the R4-R6
gate convention); intervals are reported. Relative reduction = 1 - ours / theirs, 3-seed mean of the per-seed values.
Only the R5-E2 PRIMARY checkpoints are used (4 layers, d = 128); theta_real is never retrained.

### E3-D0 (gate), inherited from R6-E3
- Exactly R6-E3's gate (PREREG-R6-E3.md section 2), recomputed in this run by the same function (`r6/run_e3.py`
  `part_d0`, unchanged) on the same positions: (0) anchor KL(P*_full || P_anchor) >= 0.12 nats per hidden slot on the
  1,000 calibration D0 positions (P_anchor = theta_real's 0-step belief at t0 = max(t - 4, t mod 2)); (1) battle accuracy
  >= 0.99 and 7-known-slot NLL <= 0.01 on the test D0 positions; (2) 16 marginal window rollouts (root moves uniform over
  the legal moves, union call) cut KL(P*_full || .) against the anchor by >= 40% on the calibration positions.
  theta_real's current one-pass belief KL at t is reported next to it. The R7 text lists (1) and (2); (0) is part of the
  inherited gate and is kept. Gate = (0) and (1) and (2). If the gate fails, E3-D1 and E3-D2 are computed and reported as
  EXPLORATORY, not as rule tests.

### E3-D1 (pure salience keys and policy-independent multi-slot aggregation)
- Protocol: exactly R6-E3's D1 (PREREG-R6-E3.md): the 500 two-flank test positions (`d1_pos.npz`; `oracle_d1_test.npz`
  for scoring only); per position the 4-rollout greedy base cover shared by both bags plus 20 flank-skewed rollouts
  (U_Left: n_L ~ Binomial(20, 0.85) from a pool of 24 Left and 24 Right rollouts; U_Right: Binomial(20, 0.15) from an
  independent pool); r6 marginal rollouts (dl_r = the marginal window message on the rollout's cone slots, 0 elsewhere);
  the R5-E2 rollout-trust beta inside ataraxos_mmd, kc_dedup and attn_L3, as in R6-E3.
- Salience (spec 2.1.1): s_r,i = ||dl_r,i||_2 / (sqrt(sum_j ||dl_r,j||_2^2) + 1e-6) over the 8 slots; bag-centred
  salience st_r = (s_r - mean_bag s) / (||s_r - mean_bag s||_2 + 1e-6), the mean over the rollouts of the same bag.
- Keys: K3: k_r = normalize(W_K s_r), W_K 16 x 8 without bias; K3c: k_r = normalize(W_K [s_r || st_r]), W_K 16 x 16
  without bias. Each head is fitted on the calibration split only, jointly with its own kc_dedup (crowding
  c_r = sum_b max(0, <k_r, k_b>)^2), by kc_dedup's calibration loss against the model-only reference (KL(ref || mu)
  over hidden slots + binary cross-entropy of the value logit against the reference value: the spec's "calibration BCE
  loss to the model-only reference") plus L_key = mean_{r != b} (<k_r, k_b> - <t_r/|t_r|, t_b/|t_b|>)^2 with target
  t = s for K3 and t = st for K3c (key_align_loss); then the head is frozen and its own attn_L3 is fitted (r5 AttnAgg
  over [k_r, W_v dl_r, dl^v_r, log beta_r, log c_r]). scaled_mix (alpha on unskewed bags) and ataraxos_mmd are fitted as in
  R6-E3. Same 1,500 two-flank calibration roots, pools (base cover, 32 Left + 32 Right) and references (64 + 64 rollouts,
  union call) as R6-E3; training bags with p_L ~ Beta(0.5, 0.5); Adam, 1,500 steps per fit (lr 1e-2 scales, 3e-3
  kc_dedup + key, 1e-3 attn_L3).
- (1) Key gap of a head: in each policy's bag, the mean cosine <k_r, k_b> over same-flank pairs (r != b) minus the mean
  over cross-flank pairs, per position; mean of the two policies' bags; mean over positions (R6-E3's statistic). Pass if
  K3 or K3c has a 3-seed mean >= 0.28 AND a 95% interval lower bound > 0.25.
- "That key head": the head that passes (1); if both pass or neither passes, the head with the larger 3-seed mean
  CALIBRATION key gap (the same statistic on 85/15 bags of the 1,500 calibration roots, model-only, computed at fit
  time). kc_dedup in (2) uses the same head.
- (2) JSD = mean over positions of the mean over hidden slots of the 6-way JSD(mu_s(U_Left), mu_s(U_Right)). For X in
  {scaled_mix, ataraxos_mmd}: 3-seed mean of the per-seed ratio JSD_X / JSD_attn_L3 >= 5.0 and the interval of
  JSD_X - JSD_attn_L3 > 0. KL = mean over positions of the mean over hidden slots of KL(P*_full,s || mu_s), averaged over
  the two policies; for X in {scaled_mix, ataraxos_mmd, kc_dedup}: the interval of KL_X - KL_attn_L3 > 0. (2) needs all
  five comparisons.
- D1 = (1) and (2). Reported descriptively: both heads' gaps and cosines, both heads' kc_dedup and attn_L3, union call,
  mix, and R6-E3's K4 gap (0.228) for comparison.

### E3-D2 (active trust gating and pruning on mixed-reliability bags)
- Positions: the 240 R5-E2 Q* test positions (`oracle_q_test.npz`; Q* from 64 simulator playouts per legal move), every
  legal move: the (s, a) pairs.
- Clean rollout: R6-E3's D2 rollout unchanged (r5 stiff_voi cone sample from theta_real's belief on the 3 hidden slots
  nearest the root move's target square, depth 4, policy temperature 1, 100% checkpoint). It reproduces r5's rollout
  draw for draw (checked before hashing).
- Exploratory rollout: the same 3 cone slots; each cone type is drawn (same stiff_voi order) from softmax(belief logits
  / 2.5); then the drawn types of 2 cone slots chosen uniformly are exchanged ("1 random piece-type swap"; no swap if the
  cone has fewer than 2 slots); plies 1-3 are drawn from softmax(policy logits / 2.0) over the whole policy head (r5's
  rollout temperature; no legality mask). theta_real is not told the tier.
- Per (s, a): 16 clean rollouts c1..c16 and 8 exploratory rollouts x1..x8, all independent. Mixed-reliability bag
  (K = 16) = c1..c8 + x1..x8. q_r = the rollout's leaf win/draw/loss claim, e_r = q_r[win] + 0.5 q_r[draw], H(q_r) its
  entropy in nats (maximum log 3).
- New theta_trust (spec: "trained on the calibration split's mixed-tier rollouts"): r5's rollout-trust regression
  (TrustNet: 2 hidden layers of 64, softplus output e_hat; beta = 1 / (e_hat + 0.05)), target = excess surprise of q_r
  against the REALISED game result (r5 `excess_surprise`), standardised inputs = r5's stop-grad rollout trust features
  (theta_real's mean root hidden state; n_ply; mean and max transition entropy; 2-path cone disagreement;
  |leaf value logit - 0-step action logit|; H; e; terminal) plus two consistency features: cone_lp = the log-probability
  of the final cone types under theta_real's UNTEMPERED belief, teacher-forced in the sampling order, per cone slot; and
  ply_lp = the mean untempered policy log-probability of the moves drawn at plies 1-3 (0 if there were none). The tier
  label is NOT an input. Training data: 4,000 calibration roots (n_ply >= 1, t in [4, 36]), the move really played,
  8 clean + 8 exploratory rollouts each (64,000 rollouts); Adam lr 1e-3, weight decay 1e-4, 3,000 steps of 512.
- Hyperparameters, on 1,000 further calibration roots (disjoint from the 4,000), 12 clean + 8 exploratory rollouts of the
  move really played (mixed bag = clean 1-8 + exploratory 1-8), each chosen by the log loss of the pooled effective win
  against the realised result (win 1, draw 0.5, loss 0), ties to the first grid value: alpha in {0.5, 1, 2} (trust_pool);
  (q, tau_g) in {0.25, 0.35, 0.50} x {0.1, 0.25, 0.5} (trust_gate); gamma in {0, 0.5, 1, 2, 4, 8, 16} for the stiffness
  weights (descriptive only; the rule uses gamma = 2). Q* is never read for fitting or selection.
- (1) On the mixed bags of all (s, a), beta = the new theta_trust: (a) Spearman(beta_r, -KL(Q*(s, a) || q_r)) over all
  mixed-bag rollouts of a seed >= 0.30 (interval > 0); (b) within pairs: the mean over (s, a) of the Spearman over its 16
  rollouts (pairs where beta or KL is constant are skipped) >= 0.25 (interval > 0); (c) (a) minus
  Spearman(-H(q_r), -KL) over the same rollouts >= 0.55 (interval of the difference > 0); (d) within 10 exactly
  equal-count strata of H(q_r) (by rank, ties by index order), the mean within-stratum Spearman(beta) minus the mean
  within-stratum Spearman(-H) >= 0.20 (interval > 0). (1) = (a) and (b) and (c) and (d).
- (2) Pooled estimate per (s, a) from its mixed bag: unif_pool = mean of e_r; stiff_pool (gamma = 2):
  w_r = (1 - H(q_r) / log 3)^2; trust_pool: w_r = beta_r^alpha; trust_gate: w_r = sigmoid((beta_r - Q_q(beta_sa)) /
  (tau_g std(beta_sa) + 1e-6)), Q_q = the linear-interpolation q-quantile of the bag's 16 betas, std with ddof 1;
  trust_resample: initial batch c1..c6 + x1..x6 (the 50/50 mix), drop the 4 lowest beta (ties: the earlier index is
  dropped), add 4 fresh clean rollouts c9..c12, uniform mean of the 12. MSE = mean over all (s, a) of
  (Q_hat - Q*_eff)^2, Q*_eff = Q*[win] + 0.5 Q*[draw]. An arm X meets the criterion if the relative reduction
  1 - MSE_X / MSE_stiff >= 0.20 (3-seed mean) with the interval of MSE_stiff - MSE_X > 0, AND the interval of
  MSE_unif - MSE_X > 0. (2) passes if (trust_gate meets it OR trust_pool meets it) AND trust_resample meets it.
- D2 = (1) and (2).
- Reported descriptively (not rules): the fitted gamma and its pool; trust_pool with the R5-E2 head (R6-E3's arm); clean8
  (uses the tier label: a reference, not an agent rule), explore8, clean16; trust_trim12 on the homogeneous clean bag
  c1..c16 with the new head and with the R5-E2 head (the spec's descriptive "trust_trim12"); trust_pool on clean16;
  effective sample sizes; delusional-move rates (R6-E3 definition); per-tier diagnostics (KL, |e - Q*_eff|, share with
  H < 0.15, flag-capture endings, piece-count violations, moves with untempered probability < 1e-3, cone_lp, ply_lp,
  beta); the share of exploratory rollouts among trust_resample's pruned ones; Spearmans of the R5-E2 head.

## 3. Data and checkpoints (all on disk before this file was hashed)

- R5-E2 data in `/tmp/r5e2/data/`, unchanged, including `oracle_d1_test.npz` from R6-E3 (sha256 8dccda9f...).
- R5-E2 checkpoints in `/tmp/r5e2/ckpt/s{101,102,103}/primary/`: theta_real (real_100.pt; sha256 641c9f17..., b3af3190...,
  797e048c...), the slot-trust heads, the R5-E2 rollout-trust head (rollout_trust.pt; f72a2ade..., 6a486ce2...,
  51c4137d...). Each theta_real's parameter hash equals `param_hash_final` in its train_meta.json; the run checks it at
  the start and the end. No retraining of theta_real. Every pooling rule, key head and the new theta_trust are fitted in
  this run (R6-E3's fitted heads were not saved).

## 4. Code

`r7/r7lib.py` (K3/K3c, tiered rollouts, trust regression, pooling rules), `r7/run_r7e3.py` (one seed: calibration fits,
theta_trust, selection, D0, D1, D2), `r7/report_r7e3.py` (applies section 2 to the synced files; written before the
results exist), `r7/gpu/` (staging, VM runner, driver, launcher, ledger). `r5/` and `r6/` are imported unchanged.

## 5. Arms, seeds, metrics, stop rules, compute

- Arms: union / mix / scaled_mix / ataraxos_mmd / kc_dedup (K3, K3c) / attn_L3 (K3, K3c) (D1); beta vs -H (D2-1);
  unif_pool / stiff_pool / trust_pool / trust_gate / trust_resample (D2-2), plus the descriptive arms above. Seeds 101,
  102, 103 select the checkpoint and every sample.
- Per-item outputs (synced back as CSV/JSON only): eval/fit.json, trust.json, d0.json, d0_gate_{cal,test}.csv,
  d0_battles.csv, d0_nll7.csv, d1.json, d1_positions.csv, d2.json, d2_actions.csv, d2_rollouts.csv, done.json.
- Stop rules: (i) gate fail -> D1, D2 exploratory only (labelled). (ii) Own ceiling $1.50: one on-demand (STANDARD) L4
  runs all three seeds in parallel; GPU_RUN_TIMEOUT 60 min, so the worst-case reservation (1.6 h x $0.87 = $1.39) stays
  under the ceiling; group r7 cap $6.50 and its guard apply. (iii) A crash is fixed (bug fix only) and rerun, logged in
  PREREG-R7-E3-AMENDMENTS.md. (iv) No change to a rule, threshold, scoring set, grid or design after any test number is
  seen. (v) `outbox/experiments-round-7-amendments*.md` is checked before launch and before scoring; an amendment is
  applied only if it arrives before scoring, and logged. (vi) The driver uses the R6-E3 launch fix (the remote redirect
  runs inside sudo) and checks within 90 s that the three seeds are running.

## 6. Interpretations of the spec and deviations (declared before running)

1. Exploratory tier: the spec offers "3-slot cone sampled uniformly at random over hidden slots (or at high temperature
   tau_cone = 2.5 with 1 random piece-type swap)". We use the second, on the standard cone: a cone drawn uniformly over
   hidden slots usually leaves out the attacked piece, while the spec's mechanism (sampled flag captures and uniqueness
   traps at the attacked piece) needs it in the cone. "Swap" = exchanging two cone slots' drawn types.
2. tau_ply = 2.0 divides the whole policy head's logits (r5's rollout temperature), with no legality mask. A pre-hash
   diagnostic (item 11a) found that the policy head puts 98.5% of its mass on legal moves at temperature 1 and 85.2% at
   temperature 2, so the exploratory tier plays some moves the rules forbid; this is part of its unreliability and is
   reported (moves with untempered probability < 1e-3). This choice was made after that diagnostic and before any test
   number.
3. Clean tier = R6-E3 setting (policy temperature 1, untempered cone): the spec's "tau_tier = 0.3 / 2.0" are tier labels;
   it says "(R6-E3 setting)" for the clean tier.
4. stiff_pool uses (1 - H / log 3)^2: H is the entropy of a 3-outcome claim (maximum log 3); with log 2 the base would be
   negative for H > log 2.
5. theta_trust's "tier/cone-consistency features" = cone_lp and ply_lp (how consistent the drawn cone and plies are with
   theta_real's own untempered distributions); the tier label itself is not an input, so the head has to find unreliable
   rollouts from what it can observe. The R5-E2 head stays inside the D1 aggregators (exact R6-E3 protocol) and is
   reported in D2 descriptively.
6. trust_resample's initial batch of 12 = exactly 6 clean + 6 exploratory (the 50/50 mix, stratified), the first six of
   each tier of the mixed bag; the 4 replacements are fresh clean rollouts outside the mixed bag.
7. "trust_trim12 on the homogeneous R6-E3 clean rollout bag" = 16 fresh clean rollouts per (s, a) in this run (R6-E3's
   bag protocol), so that the new head can score them.
8. "That key head" (D1 (2)) and kc_dedup's head: as defined in section 2.
9. "(or trust_pool)": either trust_gate or trust_pool can satisfy the first half of D2 (2), as written.
10. Compute on one on-demand L4 for all seeds; CSV/JSON-only sync back.
11. Pre-hash diagnostics and smoke (disclosed; every number here was seen before hashing):
    (a) Policy legal mass on 256 calibration roots (seed-101 checkpoint): 0.985 of the policy head's probability is on
    legal moves at temperature 1, 0.946 at 1.5 and 0.852 at 2.0 (item 2).
    (b) Tier check on 48 calibration roots (first legal move, 16 rollouts per tier, seed-101 checkpoint): the clean tier
    reproduces r5's rollouts draw for draw (identical e, q, trust features and R5-E2 beta). Clean vs exploratory: share
    with H < 0.15 0.124 vs 0.103; flag-capture endings 0.100 vs 0.096; piece-count violations 0 vs 0.030; moves with
    untempered probability < 1e-3 0.004 vs 0.124 per move; cone_lp -0.99 vs -3.65; ply_lp -2.25 vs -3.56.
    (c) `r7/run_r7e3.py --smoke` on CPU (seed 101): 32 calibration roots and 20 steps per aggregator fit; 48 + 32
    calibration roots and 20 steps for theta_trust and the selection; 48 calibration and 48 test D0 positions; the first
    48 D1 and the first 12 D2 positions (132 pairs) of the TEST sets. One crash was fixed (a torch argsort keyword). Its
    numbers were seen, for example: calibration key gap K3 0.19, K3c 0.51; test key gap K3 0.23, K3c 0.56; mean
    per-rollout KL to Q* 0.363 clean vs 0.361 exploratory; Spearman(beta, -KL) 0.16 overall and 0.19 within pairs vs
    -0.44 and -0.64 for -H; MSE unif 0.0160, stiff (gamma 2) 0.0432, trust_pool 0.0172, trust_gate 0.0229,
    trust_resample 0.0221. Nothing in the design, rules or grids was changed after the smoke; every definition above was
    fixed before it ran, apart from the argsort bug fix.

## 7. Reporting

RESULTS-R7-E3.md, AUDIT-R7-E3.md (a separate pass by an independent agent) and a GPU-LOG section, under
`work/micro-stratego/r7/`; every number from the synced per-seed files via `r7/report_r7e3.py`; failures stay failures;
exploratory and post hoc items are labelled. Page `/home/agent/memo-serve/lir-agent/r7-e3.html`; /experiments
entry staged without running the sync script.
