# PREREG-R8-E3: R8-E3 (E-MS-SAFE-ANCHOR) 6x6 Stratego: current-turn anchor safety, conditional corner worlds vs epistemic rollout corruption

Pre-registration for experiment R8-E3 of `/home/agent/workspaces/mats/outbox/experiments-round-8-spec.md` (section 3;
spec sha256 b19a33e098a3309e115c4cc696841c4377fa9ce4a72d1278a8e5bdae95f5bae5). Executor: an Opus 5.5 subagent of
eurisko, assigned by opus-mats (agent mail #3734). Written 2026-10-17, BEFORE any launch for this experiment (the hash
time is in `PREREG-R8-E3.sha256`). `outbox/experiments-round-8-amendments.md` (15:57Z) exists; it amends R8-E1 readings 3, 6 and 7 only and contains
nothing for R8-E3 (its reading 3 asks R8-E1 to fit on a smooth model-only reference; section 6, item 1, says why
R8-E3 fits its belief aggregators on true piece types instead). Work done before hashing: the code in `r8/` (reusing `r5/`, `r6/` and
`r7/` unchanged), a check that the R5-E2 checkpoints load and match their recorded parameter hashes, and a CPU crash
smoke (disclosed in section 6). The SHA-256 of this file goes into `PREREG-R8-E3.sha256` (hash, file, UTC time) before
the first launch; later changes go to `PREREG-R8-E3-AMENDMENTS.md` with time and reason.

Budget: group `r8` (GPU_BUDGET_GROUP=r8, GPU_GROUP_BUDGET_USD=2.2, shared with R8-E1; the guard deletes r8 VMs older
than 90 min and stops launches at $2.0 estimated); own ceiling $0.50 (the spec's hard cap for R8-E3; estimate ~$0.35);
run ids `r8e3-*`; launches only through `logical-soup/infra/gpu-run.sh`, whose EXIT trap deletes the VM as soon as the
driver returns (spec protocol item 3). Seeds 101, 102, 103. Code: `/home/agent/workspaces/mats/work/micro-stratego/r8/`.

## 1. Decision rules, copied verbatim from section 3.2 of the spec

### 3.2 Pre-Registered Decision Rules (`R8-E3`)

- **`E3-D1` (Current-Turn Anchor Safety & Anytime Monotonicity — Kelvin's Question 1):**
  Anchored at the **current-turn one-pass belief $\ell^{(0)}_t$** ($\mathrm{KL}(0) \approx 0.0472$ nats) across rollout counts $K \in \{0, 1, 2, 4, 8, 16\}$ under Left/Right preference skew:
  1. **Naive baselines degrade the strong anchor:** Both `raw_count` replacement and fixed-step `ataraxos_mmd` ($\gamma = 0.25$) degrade KL relative to $\ell^{(0)}_t$ ($\mathrm{KL}(16) > \mathrm{KL}(0) + 0.010$ nats, 95% CI $> 0$).
  2. **`attn_L3-K3c` (and `looped_attn_T3-K3c`) never degrade $\ell^{(0)}_t$ and strictly improve it:** Across $K \in \{1, 2, 4, 8, 16\}$, $\max_{K \ge 1}[\mathrm{KL}(K) - \mathrm{KL}(0)] \le +0.0005$ nats in every seed, $\mathrm{KL}(16) < \mathrm{KL}(0)$ with paired 3-seed 95% CI strictly $< 0$, AND at $K=16$ `attn_L3-K3c` achieves Left/Right belief $\mathrm{JSD}$ **$\ge 8\times$ lower** than `scaled_mix` and `ataraxos_mmd` while beating `scaled_mix`, `ataraxos_mmd`, and `kc_dedup(K3c)` in exact-posterior KL (all paired 95% CIs $> 0$).
- **`E3-D2` (Conditional Corner Worlds vs. Epistemic Rollout Corruption — Kelvin's Question 6):**
  1. **Clean Determinization Bags (Window Consistency vs. Stiffness):** On homogeneous 16-rollout bags $Z^{(r)} \sim q_{\text{real}}(\cdot \mid h_{t-4})$, Bayes window-consistency pooling (`cons_pool`: $w_r \propto \exp(\tau_{\text{cons}}^{-1} \mathrm{LL}_{\text{window}}(Z^{(r)}))$, $\tau_{\text{cons}}$ fit on `cal`) achieves strictly lower root action-value MSE and lower $\mathrm{Bias}^2$ than `unif_pool`, `stiff_pool`, and `trust_pool` (paired 95% CI on $\mathrm{MSE}_{\text{unif}} - \mathrm{MSE}_{\text{cons\_pool}} > 0$ and $\mathrm{MSE}_{\text{stiff}} - \mathrm{MSE}_{\text{cons\_pool}} > 0$, cutting `stiff_pool` MSE by $\ge 60\%$).
  2. **Mixed Epistemic-Noise Bags (`10` clean depth-4 rollouts + `6` epistemically noisy 1-ply/high-temperature $\tau_{\text{ply}}=3.5$ rollouts with random piece swaps, tier label hidden from all aggregators):**
     - $\theta_{\text{trust}}$ (trained on `cal` excess surprise using stop-grad trust + `cone_lp` + `ply_lp` features, no tier label) achieves Spearman $\ge +0.40$ with $-\mathrm{KL}(Q^* \Vert q_r)$ overall and within $(s,a)$ pairs, beating stiffness $-H(q)$ by $\ge +0.60$ overall and $\ge +0.20$ within entropy strata.
     - Both `trust_gate` (dropping bottom-$M$ by $\beta_r$) and combined `cons_trust_pool` ($w_r \propto w_r^{\text{cons}} \cdot \beta_r^\alpha$, tuned on `cal`) beat **`unif_pool` by $\ge 20\%$ MSE** and **`stiff_pool` by $\ge 60\%$ MSE** on test (all paired 3-seed 95% CIs $> 0$), while on clean-only bags `cons_trust_pool` never underperforms `unif_pool` ($\mathrm{MSE}_{\text{cons\_trust}} \le \mathrm{MSE}_{\text{unif}}$).

## 2. Operational definitions (how each rule above is measured and decided)

Conventions as in PREREG-R7-E3.md. Each statistic is computed per seed (101, 102, 103) on held-out test data; the
decision uses the 3-seed mean and the seed-t 95% interval (df = 2, t = 4.303) of the per-seed values. A rule with an
interval requirement passes only if the mean meets the bar AND the interval excludes the null. Ratios are tested through
the paired difference of their two terms (the 3-seed mean of the per-seed ratio meets the bar; the interval of the
difference excludes 0). Relative reduction = 1 - ours / theirs (3-seed mean of the per-seed values). Only the R5-E2
PRIMARY checkpoints are used (4 layers, d = 128); theta_real is never retrained. The spec defines no gate for R8-E3.

### Shared objects
- Anchor ell_t: theta_real's one-pass 0-step belief at the current turn t (the belief R6/R7 reported as "one pass at
  t", KL about 0.047). Every belief aggregator is anchored at it: mu = log_softmax(ell_t + sum_r w_r[s] dl_r[s]), so
  K = 0 rollouts returns ell_t exactly.
- Window: t0 = max(t - 4, t mod 2) (t0 = t - 4 for every scored position, t in [4, 36]); the opponent's REAL moves in
  [t0, t).
- Consistency rollout (beliefs): the r5 cone C_r (the 3 hidden opponent slots nearest the target square of the rollout's
  root move, uniform within a flank); a determinization Z_r of C_r drawn from theta_real's belief at t0 (stiff_voi
  any-order unmasking on the t0 tape); its window log-likelihood LL_r = sum over the opponent's window moves of
  log pi(move) under theta_real's policy head with C_r := Z_r and the other hidden slots MASK; the residual
  dl_r[s] = lmu_s - ell_t,s on its cone slots (0 elsewhere), lmu_s = log softmax(log P_t0,s + LL_s(.)) the R6 marginal
  window posterior of slot s (the rollout's claim, the same for every rollout covering s); beta_r = the R5-E2
  rollout-trust precision of its depth-4 value rollout (as R6/R7).
- Determinized value rollout (action values): the cone of the root move at t; cone types Z_r drawn from theta_real's
  belief at t0 as above (Z ~ q_real(. | h_t-4)), written into the root tape at t; r5's depth-4 value rollout at policy
  temperature 1 ("clean"). q_r = leaf win/draw/loss claim, e_r = q_r[win] + 0.5 q_r[draw], H(q_r) its entropy (max log 3),
  LL_window(Z_r) as above. NOISY (epistemically corrupted) rollout: the same draw, then every cone slot's type is replaced
  by a uniformly drawn DIFFERENT type ("random piece swaps"); 3 of the 6 noisy rollouts are myopic (depth 1: the value
  head right after the root move, "1-ply"), 3 are random playouts (depth 4, plies 1-3 drawn from softmax(policy logits /
  3.5) over the whole policy head, "high-temperature tau_ply = 3.5"). theta_real and every aggregator never see the tier.

### E3-D1 (current-turn anchor safety and anytime monotonicity)
- Positions: the 500 R5-E2 two-flank test positions (`d1_pos.npz`; `oracle_d1_test.npz`, P*_full, for scoring only).
- Preference skew: U_Left draws each rollout from the Left flank with probability 0.85, U_Right with 0.15. Per position
  and policy, an independent 16-rollout sequence (position j is Left with probability p_L; Left and Right rollouts are
  taken in order from that policy's own pools of 16 Left + 16 Right consistency rollouts). The bag at K is the sequence's
  first K rollouts, K in {1, 2, 4, 8, 16} (nested bags, so KL(K) is an anytime curve); K = 0 is ell_t. No base cover
  (a bag of 1 or 2 rollouts cannot hold one).
- Aggregators, all anchored at ell_t:
  - raw_count (replacement): on each covered slot, mu = 0.99 x the consistency-weighted frequency of the drawn types
    (weights exp(LL_r - max LL)) + 0.01 / 6; uncovered slots keep ell_t.
  - ataraxos_mmd, fixed step gamma = 0.25: ell_t + 0.25 sum_r beta_r dl_r.
  - scaled_mix: ell_t + alpha sum_r dl_r (alpha fitted).
  - kc_dedup(K3c): ell_t + u sum_r (beta_r / c_r) dl_r, c_r = sum_b max(0, <k_r, k_b>)^2; R7-E3's K3c key head (bag-centred
    salience of dl) fitted jointly with u, plus L_key, then frozen.
  - attn_L3-K3c: 3 untied pre-LN self-attention blocks (d 64, 4 heads) over an ANCHOR token (W_a ell_t + a learned
    vector) and rollout tokens [k_r (K3c), W_v dl_r, log beta_r, log c_r, LL_r / 5]; output per-slot non-negative
    weights w_r[s] (softplus, initialised at 0.02); mu = log_softmax(ell_t + sum_r w_r[s] dl_r[s]).
  - looped_attn_T3-K3c: the same with ONE block applied 3 times with a step embedding (weight-tied, as R7-E1).
  - Descriptive only: union_t (each covered slot gets lmu), ataraxos_mmd with a fitted step.
- Fitting (calibration split only): 2,000 two-flank calibration roots, per root pools of 16 Left + 16 Right consistency
  rollouts. Target: the TRUE hidden piece types of the calibration positions (cross-entropy averaged over hidden slots).
  These are realised data, the labels theta_real was trained on, not the exact Bayes posterior; P*_full is never read
  for fitting. Training bags: prefixes K ~ U{1..16} of 16-rollout sequences with p_L ~ Beta(0.5, 0.5) per root
  (scaled_mix: p_L = 0.5). Adam, 1,200 steps per fit (lr 1e-2 for the scalars, 3e-3 kc_dedup + key, 1e-3 attention).
- KL(K) = mean over positions of the mean over hidden slots of KL(P*_full,s || mu_s), averaged over the two policies.
  JSD(K) = mean over positions of the mean over hidden slots of the 6-way JSD(mu_s(U_Left), mu_s(U_Right)).
- (1) For raw_count AND for ataraxos_mmd (gamma 0.25): 3-seed mean of KL(16) - KL(0) > 0.010 and its interval > 0.
- (2) attn_L3-K3c: (a) in every seed, max over K in {1, 2, 4, 8, 16} of KL(K) - KL(0) <= 0.0005; (b) KL(16) - KL(0)
  has interval < 0; (c) at K = 16, for X in {scaled_mix, ataraxos_mmd (gamma 0.25)}: 3-seed mean of JSD_X / JSD_attn
  >= 8 and interval of JSD_X - JSD_attn > 0; (d) at K = 16, for X in {scaled_mix, ataraxos_mmd (gamma 0.25),
  kc_dedup(K3c)}: interval of KL_X - KL_attn > 0. looped_attn_T3-K3c (named in the rule): (a) and (b). (2) passes only if
  attn_L3-K3c meets (a)-(d) AND looped_attn_T3-K3c meets (a) and (b).
- D1 = (1) and (2).

### E3-D2 (conditional corner worlds vs epistemic rollout corruption)
- Positions: the 240 R5-E2 Q* test positions (`oracle_q_test.npz`; Q* from 64 simulator playouts per legal move), every
  legal move: 2,361 (s, a) pairs per seed.
- Per (s, a): 16 clean determinized rollouts c1..c16 and 6 noisy ones (3 myopic, 3 tau-3.5). Clean bag = c1..c16
  (E3-D2 (1)); mixed bag = c1..c10 + the 6 noisy (E3-D2 (2)).
- theta_trust: R7-E3's regression (TrustNet; target = excess surprise of q_r against the REALISED game result; inputs =
  r5's stop-grad trust features + cone_lp + ply_lp; no tier label), trained on 3,000 calibration roots (n_ply >= 1,
  t in [4, 36]), the move really played, 10 clean + 3 + 3 noisy rollouts each (48,000 rollouts), 3,000 Adam steps.
  cone_lp = log-probability of the final cone types under theta_real's UNTEMPERED belief at t (teacher-forced, per cone
  slot); ply_lp = mean untempered policy log-probability of the moves drawn at plies 1-3 (0 if none).
- Hyperparameters on 1,000 further calibration roots (disjoint), 16 clean + 6 noisy rollouts of the move really played;
  each chosen by the log loss of the pooled effective win against the realised result (win 1, draw 0.5, loss 0); ties to
  the first grid value: tau_cons in {0.25, 0.5, 1, 2, 4, 8, 16, inf} for cons_pool on clean bags (inf = uniform);
  alpha in {0.5, 1, 2} for trust_pool on clean bags; M in {2, 4, 6, 8} for trust_gate on mixed bags; (tau_cons, alpha)
  in the tau grid x {0, 0.5, 1, 2} for cons_trust_pool on mixed bags; gamma for stiffness (descriptive). Q* is never
  read for fitting or selection.
- Pools (weights within a bag): unif_pool 1; stiff_pool (1 - H(q_r)/log 3)^2 (R7-E3's gamma = 2 rule); trust_pool
  beta_r^alpha; cons_pool exp(LL_window(Z_r) / tau_cons); trust_gate 1 for all but the M lowest-beta rollouts, 0 for
  those (ties: the earlier index is dropped); cons_trust_pool exp(LL_window / tau_cons) x beta_r^alpha. Q_hat = the
  weighted mean of e_r. MSE = mean over (s, a) of (Q_hat - Q*_eff)^2, Q*_eff = Q*[win] + 0.5 Q*[draw].
  Bias^2 = MSE - mean over (s, a) of [within-bag sample variance of e_r x sum_r w~_r^2] (normalised weights treated as
  fixed); it includes Q*'s own playout noise equally for every rule.
- (1) Clean bags: interval of MSE_unif - MSE_cons > 0; interval of MSE_stiff - MSE_cons > 0 with 1 - MSE_cons /
  MSE_stiff >= 0.60; 3-seed mean MSE_cons < MSE_trust; 3-seed mean Bias^2_cons below Bias^2 of unif_pool, stiff_pool and
  trust_pool. All required.
- (2a) Mixed bags, beta = the new theta_trust: Spearman(beta_r, -KL(Q*(s,a) || q_r)) >= 0.40 over all mixed-bag
  rollouts of a seed (interval > 0) and >= 0.40 within pairs (mean over (s, a) of the per-bag Spearman; constant bags
  skipped; interval > 0); Spearman(beta) - Spearman(-H) >= 0.60 overall (interval > 0) and, within 10 exactly
  equal-count strata of H (by rank, ties by index), mean within-stratum Spearman(beta) - mean within-stratum
  Spearman(-H) >= 0.20 (interval > 0).
- (2b) Mixed bags, for X in {trust_gate, cons_trust_pool}: 1 - MSE_X / MSE_unif >= 0.20 with interval of
  MSE_unif - MSE_X > 0, AND 1 - MSE_X / MSE_stiff >= 0.60 with interval of MSE_stiff - MSE_X > 0; AND on the clean bags
  cons_trust_pool (same tau_cons and alpha as tuned on mixed bags) has MSE <= MSE_unif in every seed.
- D2 = (1) and (2a) and (2b). Reported descriptively: per-tier diagnostics (KL, squared error, share with H < 0.15,
  beta, LL, cone_lp, ply_lp, flag-capture endings), the clean10 reference (uses the tier: not an agent rule), cons_pool
  and trust_pool on mixed bags, cons_pool at tau = 1 on clean bags, effective sample sizes, the share of noisy rollouts
  among trust_gate's dropped ones, the R5-E2 head's Spearman, LL's own Spearman, delusional-move rates.

## 3. Data and checkpoints (all on disk before this file was hashed)

- R5-E2 data in `/tmp/r5e2/data/`, unchanged (including `oracle_d1_test.npz` from R6-E3, sha256 8dccda9f...).
- R5-E2 checkpoints `/tmp/r5e2/ckpt/s{101,102,103}/primary/` (real_100.pt sha256 641c9f17..., b3af3190..., 797e048c...;
  rollout_trust.pt f72a2ade..., 6a486ce2..., 51c4137d...); each theta_real's parameter hash equals `param_hash_final` in
  its train_meta.json, checked by the run at the start and the end. Every aggregator, key head, the new theta_trust and
  every pooling hyperparameter is fitted in this run.

## 4. Code

`r8/r8lib.py` (window log-likelihood, determinized and consistency rollouts, anchored aggregators, pooling),
`r8/run_r8e3.py` (one seed: D1 fits, theta_trust and selection, D1, D2), `r8/report_r8e3.py` (applies section 2 to the
synced files; written before the results exist), `r8/gpu/` (staging, VM runner, driver, launcher, ledger). `r5/`, `r6/`
and `r7/` are imported unchanged.

## 5. Arms, seeds, metrics, stop rules, compute

- Seeds 101, 102, 103 select the checkpoint and every sample. Outputs (synced back as CSV/JSON only): eval/fit.json,
  trust.json, d1.json, d1_positions.csv, d2.json, d2_actions.csv, d2_rollouts.csv, done.json.
- Stop rules: (i) own ceiling $0.50 of actual (ledger) cost: one on-demand (STANDARD) L4 runs all three seeds in parallel
  with GPU_RUN_TIMEOUT 30 min, so the VM lives at most about 0.55 h (about $0.48 at $0.87/h); gpu-run.sh's EXIT trap
  deletes it as soon as the driver returns, and the driver returns right after the final sync. (The budget guard's
  reservation for this VM, 1.1 h x $0.87 = $0.96, is an admission number, not spend.) (ii) A crash is fixed (bug fix
  only) and rerun only if the ledger stays under $0.50, logged in PREREG-R8-E3-AMENDMENTS.md. (iii) No change to a rule,
  threshold, scoring set, grid or design after any test number is seen. (iv) `outbox/experiments-round-8-amendments*.md`
  is checked before launch and before scoring; an amendment is applied only if it arrives before scoring, and logged.

## 6. Interpretations of the spec and deviations (declared before running)

1. Training target of the D1 aggregators: the true hidden types of calibration positions (cross-entropy). A model-only
   reference built from rollouts is worse than ell_t (R6/R7: union call 0.096, every slot scored 0.078, vs 0.047), so
   fitting toward it would teach the aggregators to degrade the anchor; the exact posterior P*_full is oracle and never
   used for fitting.
   (The round-8 amendments ask R8-E1 to fit on a smooth model-only union-call reference; in ShipHunt that
   reference, built from 128 reads, is stronger than the root. Here every rollout-built reference is weaker than
   the anchor, so the only target that can teach an improvement over ell_t is the realised hidden state.)
2. "4-ply consistency rollouts weighted by LL_window(Z^(r))": each rollout carries a determinization of its cone drawn at
   t0 and its window log-likelihood. raw_count weights its draws by exp(LL); attn_L3 and looped_attn_T3 receive LL as a
   token feature; the residual of every rollout is its claim minus the anchor (lmu - ell_t).
3. "raw_count replacement" = consistency-weighted determinization frequencies replacing ell_t on covered slots, with a 1%
   uniform floor (an unfloored count gives infinite KL).
4. Skewed bags are prefixes of a 16-rollout sequence (each draw Left with probability 0.85 / 0.15), no base cover.
5. "(and looped_attn_T3-K3c)": the never-degrade and strict-improvement parts of D1 (2) are required of both aggregators;
   the JSD and KL comparisons at K = 16 name attn_L3-K3c only.
6. Noisy tier: "1-ply/high-temperature tau_ply = 3.5 rollouts with random piece swaps" = 3 myopic depth-1 rollouts and
   3 depth-4 rollouts at policy temperature 3.5, all with every cone slot's type swapped for a different random type.
7. Clean D2 rollouts draw the cone from the t0 belief ("Z^(r) ~ q_real(. | h_t-4)"), so that window-consistency weights
   exp(LL / tau) are importance weights toward the belief at t; the same clean rollouts c1..c10 are in the mixed bag.
8. stiff_pool keeps R7-E3's (1 - H / log 3)^2 (gamma = 2); "trust_gate (dropping bottom-M by beta_r)" uses a hard drop
   with M tuned on calibration.
9. Bias^2 is the plug-in split MSE - rollout noise (R7-E3's post hoc decomposition), noise from the within-bag variance
   with the weights treated as fixed.
10. Criteria stated without an interval in the rule ("strictly lower ... than trust_pool", "lower Bias^2", "never
    underperforms") are judged on 3-seed means, except "never underperforms", which must hold in every seed.
11. Pre-hash smoke (disclosed; its numbers were seen before hashing): `r8/run_r8e3.py --smoke` on CPU (seed 101; 32
    calibration roots and 15 steps per D1 fit; 48 + 32 calibration roots and 20 steps for theta_trust and the selection;
    the first 32 D1 and the first 8 D2 positions (100 pairs) of the TEST sets). For example: KL(0) 0.042 on the 32 D1
    positions; at K = 16 raw_count 0.675, ataraxos_mmd (gamma 0.25) 3.88, union_t 0.058, scaled_mix 0.138, attn_L3-K3c
    0.063, looped_attn_T3-K3c 0.045 (all with 15-step fits); on the 100 D2 pairs, clean bags MSE unif 0.0185,
    cons_pool (tau 1) 0.0186, stiff_pool 0.0448; per-rollout KL to Q* 0.374 clean, 0.236 myopic noisy, 0.440 tau-3.5
    noisy; mixed bags MSE unif 0.0184, trust_gate 0.0217, cons_trust_pool 0.0194. Nothing in the design, rules or grids
    was changed after the smoke; every definition above was fixed before it ran.

## 7. Reporting

RESULTS-R8-E3.md, AUDIT-R8-E3.md (a separate pass by an independent agent) and a GPU-LOG section, under
`work/micro-stratego/r8/`; every number from the synced per-seed files via `r8/report_r8e3.py`; failures stay failures;
exploratory and post hoc items are labelled. Page `/home/agent/memo-serve/lir-agent/r8-e3.html`; /experiments
entry staged without running the sync script.
