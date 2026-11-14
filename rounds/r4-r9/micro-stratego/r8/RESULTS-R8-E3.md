# RESULTS-R8-E3: 6x6 Stratego, current-turn anchor safety and conditional corner worlds vs epistemic rollout corruption (E-MS-SAFE-ANCHOR)

Executor: an Opus 5.5 subagent of eurisko, 2026-10-17, for opus-mats (agent mail #3734). Spec:
`/home/agent/workspaces/mats/outbox/experiments-round-8-spec.md`, section 3 (rules E3-D1, E3-D2). Pre-registration:
`r8/PREREG-R8-E3.md`, sha256 `dc5e5681...` (`r8/PREREG-R8-E3.sha256`), hashed 2026-10-17T15:58:30Z, before the
registered launch (15:58:44Z, `r8/gpu/launches.log`). The round-8 amendments file has nothing for R8-E3. It was read before
hashing (R8-E1 only) and after its rewrite at 15:58:46Z (still R8-E1 only). It was rewritten again at 16:01:43Z, before
scoring, to add Part B (R8-E2 readings); my pre-scoring check logged the wrong last-rewrite time and missed Part B.
Neither part concerns R8-E3, so nothing changes (`r8/PREREG-R8-E3-AMENDMENTS.md`). Paths below are
relative to `/home/agent/workspaces/mats/work/micro-stratego/`. The R5-E2 primary checkpoints were reused unchanged (no
retraining; parameter hashes checked at the start and end of every run).

## TL;DR

- **Pre-registered result: E3-D1 FAILED and E3-D2 FAILED.**
- **E3-D1 (1) passes: naive pooling wrecks a strong anchor.** Starting from theta_real's current-turn belief (KL 0.049 to
  the exact posterior on the 500 test positions), raw determinization counts reach KL 0.640 at K = 16 and the fixed-step
  ataraxos_mmd (gamma 0.25) reaches 4.78.
- **E3-D1 (2) fails, and the failure is mine: the attention aggregators memorised the calibration positions.** As fitted
  (against the true hidden piece types of 2,000 calibration positions, 1,200 steps, no held-out stopping), attn_L3-K3c
  raised KL from 0.049 to 0.635 at K = 16 and looped_attn_T3-K3c to 0.585. attn_L3-K3c's calibration cross-entropy fell
  from 1.12 to 0.98 (looped_attn_T3-K3c: 0.99) while its test KL rose thirteenfold (looped_attn_T3-K3c: twelvefold). The round-8 amendments warned R8-E1 about exactly this risk of fitting on hard
  labels. The same fit did no harm to the one-parameter scaled_mix: it never degraded the anchor and improved it at
  every K (0.0488 -> 0.0463).
- **Post hoc rerun with held-out early stopping (exploratory; no verdict changes): the anchor is safe, but the attention
  adds nothing.** Early stopping on 500 held-out calibration positions kept the attention aggregators at their
  initialisation in every seed: no checkpoint, checked every 50 steps, improved held-out cross-entropy; it was already
  worse at step 50 in every seed. That initial state is a fixed
  0.02 step per rollout. It never degrades the anchor (in every seed even its worst K is 0.0002 below K = 0) and lowers KL by
  0.0025 [0.0023, 0.0028] at K = 16 (5%), the same as scaled_mix. It would still miss the JSD clause (5.1x below
  scaled_mix, bar 8x) and does not beat scaled_mix on KL (both against the scaled_mix refitted with early stopping in
  the same rerun; against the registered scaled_mix: ratio 4.3 [2.8, 5.9], KL difference +0.00004 [-0.0003, +0.0004]).
- **E3-D2 (1) fails: window-consistency weighting did not lower MSE.** Its 3-seed mean Bias^2 is slightly lower than the
  plain mean's, but the difference, 0.0004 [-0.0003, 0.0011], includes 0, and its weights add noise. On clean determinization
  bags, cons_pool's Bias^2 is 0.01146 against 0.01186 for the plain mean, but its MSE is 0.01412 against 0.01394
  (difference -0.0002 [-0.0016, 0.0012]). The window log-likelihood hardly tells good worlds from bad ones (Spearman with
  -KL -0.04).
- **E3-D2 (2) fails: the "epistemically noisy" rollouts were not that noisy.** The myopic depth-1 rollouts with corrupted
  cones were more accurate than the clean depth-4 ones (squared error 0.040 vs 0.045). The temperature-3.5 rollouts were
  worse (0.059), but far from the spec's 0.12-0.18. The trust head ranks rollout errors at 0.330 overall and 0.354
  within an action (bar 0.40), beating stiffness by the required margins (0.71 overall, 0.23 within strata). But
  trust_gate (MSE 0.0153) and cons_trust_pool (0.0149) were both worse than the plain mean (0.0143).
- Cost: **$0.46** at the $0.87/h list-price ceiling by the ledger (VM creation to delete call), $0.49 counting each VM
  until its deletion completed (gcloud operations log: 800, 592 and 650 s); both under the $0.50 ceiling (three on-demand L4 VMs: the registered run $0.18, an exploratory
  rerun aborted by a driver bug $0.13, its relaunch $0.15; ceiling $0.50). No r8e3 VM or disk left.

## Verdict table (from `r8/report/verdicts.csv`; 3 seeds; mean [95% seed-t interval, df = 2])

| rule | verdict | numbers (seeds 101, 102, 103; mean [95% CI]) |
|---|---|---|
| E3-D1 (1) raw_count: KL(16) - KL(0) > 0.010 | pass | 0.5890, 0.5976, 0.5861; 0.5909 [0.5759, 0.6058] |
| E3-D1 (1) ataraxos_mmd (gamma 0.25): KL(16) - KL(0) > 0.010 | pass | 4.6622, 4.3518, 5.1703; 4.7281 [3.7016, 5.7545] |
| E3-D1 (2) attn_L3-K3c: max over K of KL(K) - KL(0) <= 0.0005 every seed | **fail** | 0.5461, 0.6317, 0.5802 |
| E3-D1 (2) attn_L3-K3c: KL(16) - KL(0) < 0 | **fail** | 0.5860 [0.4790, 0.6930] |
| E3-D1 (2) K = 16 JSD ratio scaled_mix / attn >= 8 | **fail** | 0.0141, 0.0176, 0.0123; 0.0147; difference -0.0184 [-0.0198, -0.0171] (attn's JSD 0.0187 is 68x higher) |
| E3-D1 (2) K = 16 JSD ratio ataraxos_mmd / attn >= 8 | **fail** | 3.17, 3.27, 3.31; 3.25; difference 0.0421 [0.0368, 0.0473] (the interval passes; the ratio misses the 8x bar) |
| E3-D1 (2) K = 16 KL: scaled_mix - attn > 0 | **fail** | -0.5885 [-0.6957, -0.4814] |
| E3-D1 (2) K = 16 KL: ataraxos_mmd - attn > 0 | pass | 4.1421 [3.0601, 5.2241] |
| E3-D1 (2) K = 16 KL: kc_dedup(K3c) - attn > 0 | **fail** | -0.5860 [-0.6921, -0.4800] |
| E3-D1 (2) looped_attn_T3-K3c: max over K <= 0.0005 every seed | **fail** | 0.5070, 0.5556, 0.5456 |
| E3-D1 (2) looped_attn_T3-K3c: KL(16) - KL(0) < 0 | **fail** | 0.5361 [0.4723, 0.5998] |
| **E3-D1** | **FAIL** | (1) passes, (2) fails |
| E3-D2 (1) clean bags: MSE(unif) - MSE(cons_pool) > 0 | **fail** | 0.0002, -0.0008, 0.0001; -0.0002 [-0.0016, 0.0012] |
| E3-D2 (1) cons_pool cuts stiff_pool MSE >= 60% | pass | 0.6313, 0.5855, 0.7156; 0.6441; difference 0.0257 [0.0171, 0.0343] |
| E3-D2 (1) MSE(trust_pool) > MSE(cons_pool) (mean) | pass | 0.0003 |
| E3-D2 (1) Bias^2 of unif / stiff / trust above cons_pool's (means) | pass | 0.01186 / 0.03432 / 0.01191 vs 0.01146 |
| **E3-D2 (1)** | **FAIL** | not below unif_pool |
| E3-D2 (2a) Spearman(beta, -KL) overall >= 0.40 | **fail** | 0.2913, 0.3915, 0.3086; 0.3305 [0.1974, 0.4635] |
| E3-D2 (2a) same within (s, a) pairs >= 0.40 | **fail** | 0.3346, 0.3688, 0.3594; 0.3543 [0.3104, 0.3982] |
| E3-D2 (2a) minus Spearman(-H) overall >= 0.60 | pass | 0.6879, 0.7795, 0.6710; 0.7128 [0.5678, 0.8578] |
| E3-D2 (2a) same within 10 entropy strata >= 0.20 | pass | 0.1777, 0.2921, 0.2118; 0.2272 [0.0814, 0.3730] |
| E3-D2 (2b) trust_gate cuts unif MSE >= 20% | **fail** | -0.0596, -0.0285, -0.1235; -0.0705 (worse; difference -0.0010 [-0.0023, 0.0004]) |
| E3-D2 (2b) trust_gate cuts stiff MSE >= 60% | pass | 0.6171; difference 0.0247 [0.0191, 0.0303] |
| E3-D2 (2b) cons_trust_pool cuts unif MSE >= 20% | **fail** | -0.0974, 0.0068, -0.0293; -0.0400 (difference -0.0006 [-0.0026, 0.0014]) |
| E3-D2 (2b) cons_trust_pool cuts stiff MSE >= 60% | pass | 0.6270; difference 0.0251 [0.0190, 0.0312] |
| E3-D2 (2b) clean bags: MSE(cons_trust) <= MSE(unif) every seed | **fail** | +0.0013, -0.0001, +0.0006 |
| **E3-D2** | **FAIL** | (1), (2a) and (2b) each fail |

## What this means

1. **A strong current-turn anchor is easy to break and easy to protect.** Pooling that replaces the anchor (raw_count)
   or adds every rollout's evidence at a fixed step (ataraxos_mmd, gamma 0.25) destroys it; a likely reason, not tested
   here, is that the window evidence is already in the one-pass belief, so adding it again double counts
   (`r8/report/tables.md`). The rules with a small fitted step, scaled_mix and ataraxos_mmd with a fitted step (and, post
   hoc, the early-stopped attention), stayed below the anchor at every K in every seed; kc_dedup(K3c) rose +0.0009 above it
   at its worst K in seed 103, over the 0.0005 bar. The fitted scaled_mix (alpha 0.039) never moves KL above the anchor and lowers it from
   0.0488 to 0.0463 by K = 8. The anchor is already close to the exact posterior (KL 0.049); the best rule here
   closed 5% of that gap.
2. **The registered attention fits are a fitting failure, not evidence about attention.** I fitted them against the true
   piece types of 2,000 calibration positions (prereg section 6, item 1) without held-out stopping. A 100k-parameter
   network fitted on the same 2,000 positions for 1,200 full-batch steps learned those positions' answers (held-out loss
   rose from the first check in the post hoc rerun; the route, for example the anchor token acting as a position
   identifier, was not tested): calibration
   cross-entropy fell from 1.117 to 0.976 (`r8/runs/s*/eval/fit.json`) while test KL rose from 0.049 to 0.635. Even one
   rollout moved test KL to 0.247. The R8-E1 amendment had flagged this risk for hard-label targets; I judged a
   model-only target worse here (every rollout-built reference is weaker than the anchor) and did not guard against
   memorisation.
3. **With held-out early stopping (post hoc), the attention aggregators learn nothing and stay safe.** In every seed the
   best held-out step was step 0, the initialisation (recorded as -1 in fit_es.json), for both attn_L3-K3c and looped_attn_T3-K3c
   (`r8/report/expl_es.json`). Their outputs then equal the anchor plus a fixed 0.02 step per rollout, which is why the
   two networks give identical numbers. That rule is safe (KL(16) - KL(0) = -0.0025 [-0.0028, -0.0023]; every K below
   K = 0 in every seed) and as good as scaled_mix. It would still fail the rule's JSD clause against scaled_mix (5.1x,
   bar 8x) and its KL clause (difference +0.0004 [-0.0004, +0.0011]) (against the scaled_mix refitted with early
   stopping in the same rerun; against the registered scaled_mix: ratio 4.3 [2.8, 5.9], KL difference +0.00004
   [-0.0003, +0.0004]).
4. **Window consistency gave no measurable gain here.** cons_pool's 3-seed mean Bias^2 is lower (0.01146 vs 0.01186),
   as the spec predicted, but the difference, 0.0004 [-0.0003, 0.0011], includes 0, and LL does not rank rollouts by
   accuracy (Spearman with -KL -0.042 overall, -0.006 within pairs). Two opponent moves carry little evidence about
   three cone pieces, so the weights mostly add noise (effective sample 14 of 16; noise 0.0027 vs 0.0021). The fitted
   temperatures were 8, 1 and 8, so the weights were nearly flat in two seeds.
5. **The noisy tier did not create the separation the spec needs.** Corrupting every cone type barely hurt a myopic
   depth-1 rollout: the value head right after the root move is a good estimator here; why corrupted cones barely
   hurt it was not tested (the value head reads the corrupted slot types too, so an attack is not the only route) (squared error 0.040 vs 0.045 for clean; KL 0.27 vs 0.36). The temperature-3.5
   playouts were worse (0.059; 56% of their moves were ones the policy head rates below 1e-3) but still nowhere near
   0.12-0.18. With so little difference to find, the trust head's drops were only slightly richer in noisy rollouts than
   chance (40% vs 37.5% of the bag), and every trust rule lost to the plain mean. As in R7-E3, trust ranks errors far
   better than stiffness does (-H: -0.38 overall), and stiffness weighting is the worst pool (MSE 0.040).

## Exploratory: early-stopped D1 rerun (post hoc; `r8/expl_d1_es.py`, `r8/runs_es/`, `r8/report/expl_es.json`)

Written after the registered verdicts were seen. It changes only the fitting: every D1 aggregator is fitted on 1,500
calibration positions with early stopping on 500 further held-out calibration positions (validation cross-entropy
against the true types every 50 steps; the best step is kept). Rollouts, bags, test positions and scoring are the
registered code, so the 500 test bags are the registered ones (anchor KL identical).

| aggregator | KL at K = 0, 1, 2, 4, 8, 16 (3-seed means) | worst K >= 1 minus K = 0, per seed | KL(16) - KL(0) [95% CI] |
|---|---|---|---|
| attn_L3-K3c (registered) | 0.0488, 0.2473, 0.3701, 0.4968, 0.5893, 0.6348 | +0.546, +0.632, +0.580 | +0.5860 [+0.4790, +0.6930] |
| attn_L3-K3c (early-stopped) | 0.0488, 0.0486, 0.0484, 0.0480, 0.0473, 0.0463 | -0.0002 in every seed | -0.0025 [-0.0028, -0.0023] |
| looped_attn_T3-K3c (early-stopped) | identical to attn_L3-K3c (both kept their initialisation) | -0.0002 in every seed | -0.0025 [-0.0028, -0.0023] |
| scaled_mix (registered) | 0.0488, 0.0484, 0.0480, 0.0473, 0.0463, 0.0463 | -0.0004 in every seed | -0.0025 [-0.0029, -0.0021] |
| kc_dedup(K3c) (registered) | 0.0488, 0.0476, 0.0480, 0.0480, 0.0485, 0.0488 | -0.0010, +0.0000, +0.0009 | -0.0000 [-0.0024, +0.0024] |

Best held-out step: 0 (the initialisation) for both attention aggregators in all three seeds; 1,200 / 600 / 500 for
scaled_mix; 1,200 for kc_dedup. Held-out cross-entropy of the anchor 1.0864 / 1.1116 / 1.1380; of attn at K = 16
1.0844 / 1.1089 / 1.1348.

## Setup as run

- Checkpoints: the R5-E2 primary models (4 layers, d = 128) in `/tmp/r5e2/ckpt/s{101,102,103}/primary/`, unchanged.
- Anchor: theta_real's one-pass belief at t (KL 0.0488 on the 500 D1 test positions; the t-4 anchor of R6/R7 is 0.179,
  every slot scored from it 0.080; `r8/runs/s*/eval/d1.json`).
- E3-D1: 500 two-flank test positions; per position and policy, a 16-rollout sequence skewed 85/15 from its own pools;
  bags are prefixes K = 1, 2, 4, 8, 16. Aggregators fitted on 2,000 calibration positions against the true hidden types
  (attn_L3-K3c 108,616 parameters, looped_attn_T3-K3c 41,800).
- E3-D2: 240 Q* test positions x every legal move = 2,361 (s, a) pairs per seed; 16 clean determinized rollouts (cone
  types drawn from the belief at t-4) and 6 noisy ones (every cone type changed; 3 depth-1, 3 depth-4 at policy
  temperature 3.5) per pair. theta_trust trained on 3,000 calibration positions x 16 rollouts (48,000); pooling
  hyperparameters chosen on 1,000 further calibration positions by log loss against realised results. Selected:
  tau_cons 8, 1, 8; alpha (trust_pool) 0.5 in every seed; M (trust_gate) 2, 2, 4; (tau, alpha) for cons_trust_pool
  (2, 0.5), (8, 0.5), (2, 0.5); stiffness gamma 0 in every seed (`r8/runs/s*/eval/trust.json`).
- Compute: one on-demand L4 ran the three seeds in parallel (about 473 s per seed); CSV/JSON-only sync back.

## Descriptive numbers (`r8/report/tables.md`; 3-seed means)

| quantity | value |
|---|---|
| D1 KL at K = 16: union_t (each covered slot gets the window claim) | 0.0736 |
| D1 KL at K = 16: ataraxos_mmd with a fitted step (0.0035) | 0.0481 |
| D1 JSD at K = 16: scaled_mix / kc_dedup / attn (registered) / ataraxos_mmd (gamma 0.25) | 0.00027 / 0.00056 / 0.0187 / 0.0608 |
| D2 clean bags MSE: unif / stiff / trust / cons / cons_trust / cons at tau 1 | 0.01394 / 0.03982 / 0.01440 / 0.01412 / 0.01453 / 0.01501 |
| D2 clean bags Bias^2: unif / stiff / trust / cons / cons_trust / cons at tau 1 | 0.01186 / 0.03432 / 0.01191 / 0.01146 / 0.01146 / 0.01143 |
| D2 mixed bags MSE: unif / stiff / trust_gate / cons_trust / cons / trust / clean10 (tier label, reference) | 0.01432 / 0.03996 / 0.01527 / 0.01489 / 0.01434 / 0.01482 / 0.01529 |
| per-rollout squared error: clean / myopic noisy / tau-3.5 noisy | 0.0451 / 0.0397 / 0.0592 |
| per-rollout KL to Q*: clean / myopic / tau-3.5 | 0.359 / 0.272 / 0.434 |
| share with H < 0.15: clean / myopic / tau-3.5 | 0.135 / 0.024 / 0.100 |
| mean LL_window: clean / myopic / tau-3.5 | -6.18 / -6.30 / -6.31 |
| cone_lp: clean / noisy; ply_lp: clean / tau-3.5 | -2.07 / -6.14 to -6.17; -2.22 / -9.52 |
| mean beta (new head): clean / myopic / tau-3.5 | 4.02 / 3.96 / 3.37 |
| Spearman with -KL on mixed bags: beta / -H / LL / R5-E2 head | 0.330 / -0.382 / -0.042 / 0.193 |
| share of noisy rollouts among trust_gate's dropped ones (bag share 0.375) | 0.399 |
| delusional moves (R6-E3 definition) | 0 for every rule except stiff_pool (about 0.005) |

## Figures

- `r8/report/e8_d1.png`: KL against K for every aggregator, anchored at the current-turn belief (registered fits).
- `r8/report/e8_d2.png`: D2 action-value MSE per pooling rule on clean and mixed bags.

## Compute (GPU-LOG; `r8/GPU-LOG-R8-E3.md`, rebuilt by `r8/gpu/ledger.py` from the shared logical-soup GPU-LOG.md)

| VM | purpose | created | deleted | cost (ceiling) |
|---|---|---|---|---|
| soup-gpu-r8e3-main-1017155844-4062 | registered run (STANDARD L4) | 15:59Z | 16:12Z | $0.18 (0.21 h) |
| soup-gpu-r8e3-es-1017161527-19250 | exploratory rerun, aborted by a driver bug | 16:16Z | 16:25Z | $0.13 (0.15 h) |
| soup-gpu-r8e3-es2-1017162557-241 | exploratory rerun, relaunched | 16:26Z | 16:36Z | $0.15 |

Total **$0.46** of the $0.50 R8-E3 ceiling by the ledger (VM creation to delete call); $0.49 counting each VM until its
deletion completed (gcloud operations log: 800, 592 and 650 s, from the audit). Group r8 cap $2.20, shared with R8-E1. Each VM was deleted by gpu-run.sh's
EXIT trap within about a minute of its driver returning. The $0.87/h rate is the unverified on-demand list-price
ceiling. gcloud instances and disks list (project GCP_PROJECT, name~soup-gpu-r8e3-) showed nothing left.

## Deviations log (UTC; full text in `r8/PREREG-R8-E3-AMENDMENTS.md`)

- 15:58:30Z prereg hashed; 15:58:44Z registered launch; the amendments file was rewritten at 15:58:46Z (still R8-E1 only)
  and at 16:01:43Z (adding Part B, R8-E2 only); the pre-scoring check gave the wrong time and missed Part B (corrected
  after the audit). Nothing in it is for R8-E3.
- 16:04:01Z the driver verified the three seeds were running; pipeline finished 16:10:27Z; VM deleted about 16:12Z.
- After scoring (post hoc, exploratory): a D1-only rerun with held-out early stopping, announced to opus-mats first
  (thread 3734). Its first VM was wasted: a sed slip made the driver's process check look for "[r]xpl_d1_es.py", so it
  never saw the run, started it a second time and gave up after 2.5 minutes ($0.13). Fixed and relaunched once with an
  11-minute timeout and a smaller split (1,500 + 500 calibration positions) so the total stayed under $0.50.
- The pre-hash CPU smoke used 32 D1 and 8 D2 test positions (disclosed in the prereg, section 6, item 11).

## Self-audit and independent audit

`r8/report_r8e3.py` re-derived 318 numbers from the per-item CSVs (every D1 KL and JSD at every K, the D2 trust
Spearman and every pool's MSE); the largest difference from the JSON summaries is 4.1e-7 (`r8/report/selfcheck.csv`).
Independent audit: `r8/AUDIT-R8-E3.md` (copied from
/home/agent/workspaces/fable-strategy/scratch/audit-r8e3/AUDIT-R8-E3.md, a separate read-only pass): supported
with corrections; every verdict reproduced, the anchor KL, LL_window and both rollout tiers recomputed independently from
the seed-101 checkpoint on CPU. Its ten corrections (provenance, wording, cost, completeness) are applied above; none
changes a verdict.

## Files

- Pre-registration: `r8/PREREG-R8-E3.md`, `r8/PREREG-R8-E3.sha256`, `r8/PREREG-R8-E3-AMENDMENTS.md`.
- Code: `r8/r8lib.py`, `r8/run_r8e3.py`, `r8/report_r8e3.py`, `r8/expl_d1_es.py`, `r8/expl_es_summary.py`, `r8/gpu/`;
  imports `r7/r7lib.py`, `r6/e3lib.py`, `r6/run_e3.py` and `r5/` unchanged.
- Results: `r8/runs/s{101,102,103}/eval/`, `r8/runs_es/s{101,102,103}/eval/` (exploratory), `r8/report/`,
  `r8/GPU-LOG-R8-E3.md`.
- Page: `/home/agent/memo-serve/lir-agent/r8-e3.html`.
