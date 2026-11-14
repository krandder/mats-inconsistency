# PREREG-R6-E3: R6-E3 (E-MS-MARGINAL) 6x6 Stratego marginal window rollouts, centered+salience keys, trust-weighted pooling

Pre-registration for experiment R6-E3 of `/home/agent/workspaces/mats/outbox/experiments-round-6-spec.md` (section 3;
spec sha256 8be631f1cccafb6ab8e6faeb14faa6ffe4cd3f174b372ec07cd9cace02bf170c). Executor: an Opus 5.5 subagent of eurisko,
assigned by opus-mats (agent mail #3704). Written 2026-10-17T06:34:20Z, BEFORE any launch for this experiment. No
`outbox/experiments-round-6-amendments*.md` existed when this was written. Work done before hashing: the code in `r6/`
(reusing `r5/` unchanged), the oracle file `oracle_d1_test.npz` for scoring, a check that the R5-E2 checkpoints load and
match their recorded parameter hashes, and a CPU crash smoke (disclosed in section 6, item 9). The SHA-256 of this file
goes into `PREREG-R6-E3.sha256` (hash, file, UTC time) before the first launch; later changes go to
`PREREG-R6-E3-AMENDMENTS.md` with time and reason.

Budget: group `r6` (GPU_BUDGET_GROUP=r6, GPU_GROUP_BUDGET_USD=12, shared with R6-E1; the guard deletes r6 VMs older than
4.2 h and stops launches at $11 estimated); own ceiling $1.80 (the spec's ranked-table estimate for R6-E3), run ids
`r6e3-*`, launches only through `logical-soup/infra/gpu-run.sh`. Seeds 101, 102, 103. Code:
`/home/agent/workspaces/mats/work/micro-stratego/r6/`.

## 1. Pass/fail rules, copied verbatim from section 3.3 of the spec

### 3.3 Pre-Registered Pass/Fail Rules (`R6-E3`)
- **`E3-D0` (Gate — Marginal 4-Ply Window Rollouts Beat the $t-4$ Anchor):**
  On the 3 seeds (`101, 102, 103`), starting from the $t-4$ anchor belief (`KL_anchor >= 0.12` nats/slot):
  1. 1-step battle accuracy $\ge 0.99$ and 7-known-slot NLL $\le 0.01$;
  2. Pooling $K=16$ unskewed marginal cone rollouts cuts per-hidden-slot $\mathrm{KL}(P^*_{\text{full}} \Vert \mu)$ over the anchor by **$\ge 40\%$** on the calibration split (confirming marginal window rollouts strongly improve beliefs, fixing `R5-E2`'s `-1055%` neighbor-veto bug!).
- **`E3-D1` (Non-Collapsed Keys & Policy-Independent 8-Slot Beliefs under Flank Preference Skew):**
  Under Left-flank ($U_{\text{Left}}$) vs. Right-flank ($U_{\text{Right}}$) preference skew (4 base cover rollouts + 20 85/15 flank-skewed rollouts):
  1. **Non-Collapsed Keys:** Within-flank minus across-flank key cosine is **$\ge 0.25$** (95% CI $>0$, fixing `R5-E2`'s `0.004`);
  2. **Policy-Independence & Accuracy:** `attn_L3` cuts 8-slot belief $\mathrm{JSD}\bigl(\mu^*(U_{\text{Left}}), \mu^*(U_{\text{Right}})\bigr)$ by **$\ge 2.5\times$** compared to `ataraxos_mmd` and `scaled_mix` (paired 95% CIs $>0$), and achieves lower 8-slot belief KL to $P^*_{\text{full}}$ than `scaled_mix`, `ataraxos_mmd`, and `kc_dedup` (all paired 95% CIs $>0$).
- **`E3-D2` (Self-Trust $\theta_{\text{trust}}$ Beats Entropy Stiffness in Both Ranking and Action-Value Pooling):**
  1. Spearman($\beta_{\text{trust}}, -\mathrm{KL}(Q^* \Vert q_r)$) $\ge \mathbf{0.20}$ overall (95% CI $>0$), exceeds Spearman($-H(q_r), -\mathrm{KL}$) by $\ge \mathbf{0.45}$ overall (95% CI $>0$) and by $\ge \mathbf{0.15}$ within 10 equal-count entropy strata;
  2. **Decision Impact:** Weighting per-action rollouts by $\beta_{\text{trust}}$ (`trust_pool`) achieves strictly lower action-value $\mathrm{MSE}(Q^*, \hat{Q})$ and strictly lower delusional-move rate on the 240 $Q^*$ test positions than entropy-stiffness weighting (`stiff_pool`) (paired 95% CI of MSE reduction $>0$, relative MSE reduction $\ge \mathbf{20\%}$).

## 2. Operational definitions (how each rule above is measured and decided)

Conventions as in PREREG-R5-E2.md. Each statistic is computed per seed (101, 102, 103) on held-out data; the decision uses
the 3-seed mean and the seed-t 95% interval (df = 2, t = 4.303) of the per-seed values. A rule with an interval
requirement passes only if the mean meets the bar AND the interval excludes the null (0 for differences and reductions).
Every "exceeds by" difference also needs its interval > 0 (the R5 convention, applied to the strata criterion too). Ratios
are tested through the paired difference of their two terms (the 3-seed mean ratio must meet the bar, the interval of
the difference must exclude 0). The E3-D0 gate is judged on 3-seed means (the R4/R5 gate convention); intervals are
reported. Relative reduction = 1 - ours / theirs. Only the R5-E2 PRIMARY checkpoints are used (4 layers, d = 128).

### E3-D0 (gate)
- (0) Anchor condition "(KL_anchor >= 0.12 nats/slot)": P_anchor = theta_real's 0-step belief at t0 = max(t - 4, t mod 2)
  (nothing materialised, the root tape of the same game 4 plies earlier, same observer). On the 1,000 calibration D0
  positions (`oracle_d0_cal.npz`, t in [6, 36]), the mean over positions of the mean over hidden opponent slots of
  KL(P*_full,s || P_anchor,s) must be >= 0.12.
- (1) Battle accuracy >= 0.99: argmax of theta_real's outcome head at A_0 for every test position whose first real ply is
  a battle, all opponent slots materialised with their true types. 7-known-slot NLL <= 0.01: on the 2,000 test D0
  positions, for each hidden slot with the other seven set to their true types, -log q(true type), averaged.
- (2) Marginal window rollouts, calibration split: on the same 1,000 calibration positions, 16 rollouts whose root moves
  are drawn uniformly (with replacement) from the legal moves. Each rollout's cone is the 3 hidden opponent slots nearest
  the root move's target square (r5 `select_cone`). The pooled belief mu is the union call: each hidden slot covered by
  at least one cone gets log mu_s = log softmax(log P_anchor,s + LL_s), where LL_s(k) is the log-likelihood of the
  opponent's real moves in the window t0..t-1 under theta_real's policy head with Z_s = k and every other hidden slot
  MASK; uncovered hidden slots keep P_anchor. Reduction = 1 - KL(P*_full || mu) / KL(P*_full || P_anchor) (means over
  hidden slots, then positions). Pass: >= 0.40. (All rollouts that cover a slot carry the same message, so pooling them
  by plain averaging over the covering rollouts and by counting the message once are the same thing.)
- Gate = (0) and (1) and (2). If the gate fails, E3-D1 and E3-D2 are NOT rule tests; they are still computed and
  reported as EXPLORATORY.

### E3-D1 (non-collapsed keys, policy-independent 8-slot beliefs)
- Positions: the 500 R5-E2 two-flank test positions (`d1_pos.npz`); P*_full for them in `oracle_d1_test.npz` (computed
  before this file was hashed, for scoring only). Flank of a move = canonical from-column 0..2 Left, 3..5 Right.
- Bags (P-FullSupport analogue): 4 base-cover rollouts + 20 flank-skewed rollouts = 24. Base cover: per position, 4
  legal moves picked greedily to maximise the number of hidden slots covered by their cones (ties by the lowest
  legal-list index; when nothing new can be covered, the lowest unpicked index); the same 4 rollouts (same samples)
  are in both policies' bags. Skewed part: U_Left takes n_L ~ Binomial(20, 0.85) rollouts from a pool of 24 Left
  rollouts and 20 - n_L from a pool of 24 Right rollouts (pool 1; root moves uniform within the flank); U_Right takes
  n_L ~ Binomial(20, 0.15) from an independent pool 2.
- Rollout: the r5 value rollout (stiff_voi cone sample, depth 4, theta_real only) gives the value claim, theta_trust's
  beta_r and the attention fingerprint e_r (readout attention over the cone slot tokens and touched squares, r5);
  the belief residual dl_r[s] = log mu_s - log P_anchor,s on the rollout's cone slots, 0 elsewhere (marginal: no
  neighbour sampling).
- Keys K4: k_r = normalise(W [LayerNorm(e_r - mean over the bag of e) || s_r]), s_r[s] = ||dl_r[s]||_2 (8-slot salience).
- (1) Key gap: in each policy's bag, the mean cosine <k_r, k_b> over pairs on the same flank (r != b) minus the mean over
  pairs on different flanks, per position; mean over positions; mean of the two policies. Pass: >= 0.25, interval > 0.
- (2) Policy sensitivity: JSD = mean over positions of the mean over hidden slots of the 6-way
  JSD(mu*_s(U_Left), mu*_s(U_Right)). For X in {ataraxos_mmd, scaled_mix}: 3-seed mean of the per-seed ratio
  JSD_X / JSD_attn_L3 >= 2.5 and the interval of JSD_X - JSD_attn_L3 > 0. Accuracy: KL = mean over positions of the mean
  over hidden slots of KL(P*_full,s || mu*_s), averaged over the two policies; for X in {scaled_mix, ataraxos_mmd,
  kc_dedup}: interval of KL_X - KL_attn_L3 > 0. D1 (2) needs all five comparisons.
- D1 = (1) and (2).

### E3-D2 (theta_trust vs entropy stiffness: ranking and pooling)
- Positions: the 240 R5-E2 Q* test positions (`oracle_q_test.npz`; Q* from 64 simulator playouts per legal action).
- Rollouts: for every legal action, K = 16 rollouts with that root move, the agents' setting: 100% checkpoint, stiff_voi
  cone sample, depth 4. q_r = leaf win/draw/loss claim; e_r = q_r[win] + 0.5 q_r[draw]; H(q_r) its entropy; beta_r =
  theta_trust's precision 1 / (e_hat + 0.05) (R5-E2's rollout-trust head, unchanged).
- (1) Over all rollouts of a seed: Spearman(beta_r, -KL(Q*(s,a) || q_r)) >= 0.20 (interval > 0); minus Spearman(-H(q_r),
  -KL) >= 0.45 (interval > 0); within 10 exactly equal-count strata of H(q_r) (strata by rank, ties broken by index
  order), the mean within-stratum Spearman difference >= 0.15 (interval > 0).
- (2) Pooled estimate per (position, legal action): unif_pool = mean of e_r; stiff_pool = sum exp(-gamma H(q_r)) e_r /
  sum exp(-gamma H(q_r)); trust_pool = sum beta_r e_r / sum beta_r. gamma is fitted on the CALIBRATION split without the
  oracle: on 1,000 calibration positions (t in [4, 36]), 16 rollouts of the action really played, the gamma in
  {0, 0.25, 0.5, 1, 2, 4, 8, 16} that minimises the log loss of the stiff-pooled value against the realised result
  (win 1, draw 0.5, loss 0). MSE = mean over all (position, legal action) pairs of (Q_hat - Q*_eff)^2, Q*_eff =
  Q*[win] + 0.5 Q*[draw]. Delusional move: per position the chosen move is the argmax of Q_hat; delusional if
  Q_hat(chosen) > 0.80 and Q*_eff(chosen) < 0.30; rate over the 240 positions. Pass: relative MSE reduction 1 -
  MSE_trust / MSE_stiff >= 0.20 (3-seed mean) with the interval of MSE_stiff - MSE_trust > 0, AND the 3-seed mean
  delusional rate of trust_pool strictly below stiff_pool's (if both are 0 this fails as written; reported as at floor).
- D2 = (1) and (2). unif_pool is reported as a secondary arm.

## 3. Data and checkpoints (all on disk before this file was hashed)

- R5-E2 data in `/tmp/r5e2/data/` (`meta.json`; brute-force oracle check 6.7e-16), unchanged; the R5-E2 setup split
  (8,064 / 1,008 / 1,008 setups), games and scoring sets. New: `oracle_d1_test.npz` (P*_full for the 500 D1 positions,
  same code path as r5 `generate6.post_job`; sha256 8dccda9f...).
- R5-E2 checkpoints in `/tmp/r5e2/ckpt/s{101,102,103}/primary/`: theta_real (real_100.pt; sha256 641c9f17..., b3af3190...,
  797e048c...), the slot-trust heads, the rollout-trust head theta_trust (rollout_trust.pt; f72a2ade..., 6a486ce2...,
  51c4137d...) and theta_amp. Each theta_real's parameter hash equals `param_hash_final` in its train_meta.json (checked
  2026-10-17 06:17Z, and again by the run, which also checks that the hash is unchanged at the end). No retraining. R5-E2's
  fitted aggregators (attn_l3.pt) are not used: every pooling rule is refitted here.

## 4. Pooling rules and fitting (`r6/e3lib.py`, `r6/run_e3.py`; reuses `r5/` unchanged)

- Aggregators over a bag (l0_b = log P_anchor, l0_v = theta_real's 0-step root value logit at t): union call (each
  covered slot's message once; reported, not a rule arm); mix (naive mean of claims, reported); scaled_mix: l*_b = l0_b +
  alpha_b sum_r dl_r, l*_v = l0_v + alpha_v sum_r dl^v_r; ataraxos_mmd: l* = l0 + eta sum_r beta_r dl_r; kc_dedup: l* = l0 +
  u sum_r (beta_r / c_r) dl_r, c_r = sum_b max(0, <k_r, k_b>)^2 with K4 keys; attn_L3: r5 AttnAgg (3 pre-LN residual
  self-attention layers, d 64, 4 heads, per-slot weights) over [k_r, W_v dl_r, dl^v_r, log beta_r, log c_r] with
  detached K4 keys.
- Fitting on the CALIBRATION split only, against model-only references (no oracle): 1,500 two-flank calibration roots;
  per root the base cover (4), pools of 32 Left + 32 Right rollouts, and a reference of 64 Left + 64 Right rollouts.
  Reference belief = union call over base + reference rollouts; reference value = mean leaf effective win of the 128
  reference rollouts. Training bags = base + 20 skewed with p_L ~ Beta(0.5, 0.5) per root (P-FullSupport skew);
  scaled_mix's alpha on unskewed bags (p_L = 0.5). Loss = KL(ref_b || mu*_b) (mean over hidden slots) + BCE(l*_v, ref_v);
  kc_dedup's loss adds L_key = mean_{r != b} (<k_r, k_b> - <s_r/|s_r|, s_b/|s_b|>)^2 with weight 1 and fits W and u;
  then W is frozen and attn_L3 is fitted. Adam, 1,500 steps each (lr 1e-2 scales, 3e-3 kc, 1e-3 attn_L3).

## 5. Arms, seeds, metrics, stop rules, compute

- Arms: union call / mix / scaled_mix / ataraxos_mmd / kc_dedup / attn_L3 (D1); beta_trust vs -H (D2-1); unif_pool /
  stiff_pool / trust_pool (D2-2). Seeds 101, 102, 103 select the checkpoint and every rollout sample.
- Per-item outputs (synced back as CSV/JSON only): eval/fit.json, d0.json, d0_gate_{cal,test}.csv, d0_battles.csv,
  d0_nll7.csv, d1.json, d1_positions.csv, d2.json, d2_actions.csv, d2_rollouts.csv, done.json.
- Stop rules: (i) gate fail -> D1, D2 exploratory only (labelled). (ii) Own ceiling $1.80 (ranked-table estimate): one
  on-demand (STANDARD) L4 runs all three seeds in parallel; GPU_RUN_TIMEOUT 75 min, so the worst-case reservation
  (1.85 h x $0.87 = $1.61) stays under the ceiling; group r6 cap $12 and its guard apply. (iii) A crash is fixed (bug fix
  only) and rerun, logged in PREREG-R6-E3-AMENDMENTS.md. (iv) No change to a rule, threshold, scoring set or
  hyperparameter after any test number is seen. (v) `outbox/experiments-round-6-amendments*.md` is checked before launch
  and before scoring; an amendment that arrives is applied only if it arrives before scoring, and logged.

## 6. Interpretations of the spec and deviations (declared before running)

1. Gate (2)'s pooling is the union call (parameter-free), because every rollout covering a slot carries the same marginal
   message; scaled_mix's 16-rollout reduction is reported too.
2. "(KL_anchor >= 0.12 nats/slot)" is a gate condition, measured on the calibration split.
3. "4 base cover rollouts" = greedy maximum hidden-slot coverage, shared by both bags; "85/15" = Binomial(20, 0.85 / 0.15)
   from independent pools.
4. scaled_mix's alpha "chosen on the calibration split to minimise unskewed KL" (spec section 1) is fitted against the
   model-only reference, not the oracle: oracles are for evaluation only.
5. K4 is fitted through kc_dedup's calibration loss plus L_key; attn_L3 reuses the frozen K4 keys.
6. D2 uses one rollout set (16 per legal action, depth 4, 100% checkpoint, stiff_voi), not R5-E2's mixed-depth,
   mixed-checkpoint D2 protocol; D2 (1) and (2) are both computed on it.
7. gamma is fitted on CAL against realised results; trust_pool uses beta_r without a fitted exponent.
8. The anchor is a deliberately stale belief (4 plies old). theta_real's current one-pass belief at t (KL about 0.05 in
   R5-E2) is reported next to every gate and D1 number, so that a gate pass is not read as "search beats one pass".
9. Pre-hash CPU smoke (disclosed): before this file was hashed, `r6/run_e3.py --smoke` ran on CPU with the final seed-101
   checkpoint on small subsets (48 calibration and 48 test D0 positions, 48 D1 positions, 12 D2 positions, 32
   calibration roots, 20 fitting steps) to find crashes. Its numbers were seen (for example the gate (2) reduction was
   0.39 on the 48 calibration positions). Nothing in the design was changed after the smoke; every definition above was
   fixed before it ran.
10. Compute on one on-demand L4 for all seeds (not three VMs), CSV/JSON-only sync back.

## 7. Reporting

RESULTS-R6-E3.md, AUDIT-R6-E3.md (a separate pass by an independent agent) and a GPU-LOG section, under
`work/micro-stratego/r6/`; every number from the synced per-seed files via `r6/report_e3.py`; failures stay failures;
exploratory and post hoc items are labelled.
