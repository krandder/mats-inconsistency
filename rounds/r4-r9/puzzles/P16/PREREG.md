# P16 pre-registration: amortized in-context valuation of a newly listed bit

Frozen 2026-09-29 ~21:05Z, before the Stage 0.5 pilot. Nothing below may change after the pilot except by a
dated addendum at the end. Start of the P16 clock: 2026-09-29T19:53:41Z (`START_TIME`).

## Scope (stated in PREREG, RESULTS and the catalogue)

- P16 tests a valuation tool. It tests **no economic mechanism**.
- A PASS supports a **valuation tool only**: a frozen reader whose prequential gain can price a newly listed bit.
- P16 **does not bear on the Bansal falsifier** (THEORY.md's programme falsifier, beat Bansal at large extrapolation).
- It is a cheap side experiment off P14's critical path. Cap $15, watchdog kill at $13.

## Question

A new bit b is listed after the bits B (|B| = m in {2, 6, 12}); b is a predicate of a hidden x (10 bits) that the
reader never sees. Can one frozen reader FR, meta-trained in context, score b from k = 16 rows since listing, by its
prequential log-score gain V^preq_k, and separate useful bits (exact I(b; Y | B) >= 0.05 bits) from useless ones
(CMI < 1e-12, not T3b) on bit and null types never seen in meta-training, beyond the baselines below and beyond a
matcher over the training type library?

## What is run (and what the spec asked for that is not)

Spec arms kept:

| arm | what | seeds |
|---|---|---|
| FR | axial 2D-attention reader, d=128, 6 layers, 4 heads, two-stream causal, joint AR (XLNet) decoder over w outputs | 3 (s0, s1, s2) |
| FR-direct | same backbone, candidate flag, attention-pooled head regressing exact CMI (MSE) + useful/useless BCE, GPU-oracle labels | 2 (drop order: seed 2 goes first) |
| FR-small | FR at F_meta/10 (compute slope, undertraining flag) | 1 |
| C1 | per-output KT running counts (sanity) | - |
| C4 | sparse parent-set BMA (sets <= 3 from B, b, y_<j), forward + lam * reverse CI direction, tuned | - |
| C5 | CI test by betting: C4 predictor with observed b vs 8 CRT twins from C4's reverse p-hat(b|B), ONS sizing | 1 |
| S1 | SGD reader from scratch, online full-batch Adam on the replay buffer, with and without b | 1 |
| S2 | warm-start SGD reader: pretrained on rows < H, zero-init weight for b, prequential fine-tuning | 1 |
| E1 | TabPFN v2 off the shelf (`tabpfn` 9.0.0, `ModelVersion.V2`, default n_estimators), chain of w classifiers, 4-row blocked prequential | on a paired subsample |
| L-all / L-train | Neyman-Pearson LR over the full type library / the 8 in-distribution types (library-matcher control) | - |
| CONS | S2 consumer gain: S2 trained on rows 1..H+128 with and without b, scored on a fresh 256-row block | 1 |

Simplifications, decided now and binding (budget, $15 cap; and one agent session of build time):

1. **Dropped arms:** C3 (CTW), S3 (column-equivariant Reptile reader) and FR-fact.
   - F5 is therefore **untestable** and is reported as such, never as passed.
   - The joint-readout metric (FR-fact recovery) is not reported.
   - C3 was a "keep only if C4 does not dominate" arm. Its absence weakens F1 slightly: F1 uses max(C4, C5, S1, S2).
2. **Meta-training compute:** F_meta = **1e16 counted FLOPs** for every meta-trained arm (FR-small: 1e15).
   - This follows the spec's own rule. The MFU smoke measured 2.29 TFLOP/s effective (150 steps).
   - After a one-decoder-call-per-step fix it measured 4.23 TFLOP/s (200 steps, GPU 97-100% busy). Both are below 5, so F_meta drops to 1e16.
   - FR-direct measured about 2.0 TFLOP/s: the oracle labels cost time but are not counted.
3. **Test set:** 100 episodes per type × stratum (20 types × 3 strata = 6000), not 300 (18k).
   - Built from the test namespace with the retry rules (POS_RETRY redraw to CMI >= 0.05; HN2 redraw to CMI < 1e-12; at most 30 tries).
   - The primary slice then has 1200 held-out positives against 1200 held-out negatives.
4. **Not run:**
   - the generator-shift slice;
   - the menu set (decision regret);
   - the 320-row length check;
   - V^KL;
   - the E1 blocking-penalty measurement.
5. **Adversarial Goodhart slice:** reduced to 60 episodes (|B| = 6), each with up to 96 CMI-0 candidates.
   - The candidate pool is the zero-CMI part of the type library: T0 noise, T3a, HN1-HN4 instances, each oracle-verified < 1e-12.
   - Arms FR (seed 0), C4 and C5.
   - If this slice is not finished within budget, F4 is judged on its FPR clause only, and RESULTS.md says so.
6. **CRT debiasing:** run for FR (seed 0) and C4 only, on a negative-type subsample.
   - The subsample is T3a, T3b, T4, T6, HN1 and HN4, 20 per stratum, with 16 twins, if the budget allows. It is secondary; F4 is judged on the raw score.
7. **Bootstrap:** every episode has its own y-rule seed, so the "clustered by y rule" bootstrap is the episode bootstrap.
   - 2000 paired resamples.
   - The max over baselines is recomputed in each resample.
8. **Stochastic baselines** (C5, S1, S2, CONS) run one seed each (seeded per batch), not 3.
9. **FR evaluation:** one random output order per evaluation batch, seeded by the batch's first episode index, not averaged over orders.
10. **Calibration:** slope and R² of V_16/16 against the per-row settlement gain on the next 16 rows (not 128).
11. **E1 subsample (never n_estimators):**
    - Pilot: 8 episodes per held-out type × stratum (192), k = 16 only.
    - Test: 12 per held-out type × stratum (288), plus 5 per in-distribution type × stratum (120), k = 16 only.
    - The measured cost is 0.38 s per TabPFN call, 48 calls per episode at k = 16, about 18 s per episode. The spec's 2.2e5 calls would cost about 23 VM-hours.
12. **Tuning:** 504 tune-namespace episodes (8 in-distribution types × 3 strata × 21), objective AUROC at k = 16 (ID positives vs ID negatives).
    - C4: alpha in {1, 2, 4, 8} × rho in {.25, .5}, then lam in {0, .5, 1} post hoc.
    - C5: bet scale in {.5, 1, 2} with C4's tuned predictor.
    - S1, S2: lr in {1e-3, 3e-3, 1e-2} × E_steps in {10, 20}, width 64.
    - C4 and C5 are tuned **before** the pilot, so the gate uses tuned baselines. That is conservative against FR.

## Generator, labels, oracle (as built; Stage-0 tests passed on the L4 VM, `results/tests.json`)

- H = 64 pre-listing rows; rows H..H+127 carry b; x is 10 bits; w = 6 output bits at test (4-6 in training).
- Types:
  - in-distribution: T0-T6 (with T3a, T3b);
  - held-out positives: HP1, HP2, HP3a, HP3b;
  - held-out negatives: HN1-HN4;
  - leak: Ly, Lpar, LinS, Lw4.
- Namespaces are disjoint: train 1, pilot 2, tune 3, test 4.
- Meta-training uses in-distribution types (90%) plus the leak slice (10%), never HP or HN types.
- Useful: CMI >= 0.05 bits. Useless: CMI < 1e-12 and not T3b. Everything else is left out of AUROC but kept for Spearman and consumer metrics.
- Exact CMI comes from the matmul oracle (fp64 on GPU).
- Stage-0 results:
  - T2 matches 1 − H(eps) to 6e-16; T4 = 1 bit exactly; all negative types <= 3e-16.
  - The Monte Carlo check (38 episodes) has max |z| 2.34.
  - FR causal masks are bitwise exact (joint and fact).
  - Pre-birth with/without identity is exact.
  - Settlement rows are independent.
  - C1 on parity(y) is at chance (AUROC .615 vs T0 on 40 + 40 episodes, inside the ±.2 tautology band).

## Estimators

- Primary: raw row-prequential gain V^preq_k = Σ over rows H..H+k−1 of [log R(y_t | ·, with b) − log R(y_t | ·, without b)], each term one step ahead.
- Settlement: the gain on rows H+k..H+k+15 given rows < H+k, at k in {4, 8, 16, 32, 64}. Reported for FR and C4; used in PASS (a)'s sign check.
- Arm scores:
  - L arms use the log LR after k rows.
  - C5 uses log wealth.
  - FR-direct uses its useful-logit or its predicted CMI, whichever has the higher AUROC on the tune set. Choosing the better one favours FR-direct, which is conservative for F6.
  - E1 uses the 4-row blocked prequential gain.

## Training

- AdamW (betas .9/.98, wd .01), bf16 autocast, clip 1.0.
- lr 1e-3 with a 300-step warmup, cosine in the fraction of F_meta used (floor 0.1).
- Batch 32·14/(m+1+w), at least 8, with (m, w) shared per batch.
- 30% column drop and a 35% random birth row per column.
- Batches are deterministic in (seed, batch index).
- Checkpoint every 10 min (model, optimizer, step, RNG, data position; keep 2), sha256-synced.
- Validation NLL on 16 fixed batches every 250 steps.

## Pre-registered decision rules

**Stage 0.5 gate** (pilot namespace; 8 held-out types × 3 strata × 42 = 1008 episodes; k = 16).
FR training proceeds only if both hold:
- (i) AUROC(L-all) − max(AUROC(E1), AUROC(C4), AUROC(C5)) >= 0.05 on the primary contrast (HP positives vs HN negatives).
  - This must hold both on the full pilot (without E1) and on E1's paired 192-episode subsample (with E1).
  - Otherwise stop and report that TabPFN or C4 suffices.
- (ii) The mean novelty index over the 8 held-out types is >= 0.05.
  - Novelty of type t = AUROC_L-all(t) − AUROC_L-train(t).
  - For a positive type the contrast is that type vs all HN negatives; for a negative type it is all HP positives vs that type.
  - If (ii) fails: universality is reported as untestable. The held-out types are not redesigned (no budget for a re-pilot), and training proceeds only if (i) holds.

**Trained-enough precondition** (before reading held-out results):
- Validation NLL improved by < 1% over the last 20% of steps.
- AUROC(FR, k=16) >= 0.80 in distribution (x-computable ID types, T3b excluded).
- Failing either makes held-out results "inconclusive (undertrained)", not F2.
- FR-small → FR in-distribution gain > 0.05 is flagged "compute-limited".

**PASS** requires the gate, the precondition, all of (a)-(c), and none of F1-F4, F6 firing. F5 is untestable.
- (a) On the held-out primary slice at k = 16, AUROC(FR) − max over {C1, C4, C5, S1, S2, E1} >= 0.05.
  - AUROC(FR) is the mean of the 3 seeds' AUROCs.
  - E1 enters on its paired subsample: the difference is also computed there, and both must hold.
  - The bootstrap 95% CI lower bound must be > 0, with the max recomputed per resample.
  - The difference must be positive for each FR seed separately.
  - It must also be positive under the settlement estimator, FR vs C4, the arm with settlement.
- (b) AUROC(FR, 16) >= 0.85 in distribution and >= 0.80 on the held-out primary slice.
  - Exception: where L-all < 0.90 and L-all >= every arm in the bootstrap, FR needs >= 80% of L-all's above-chance AUROC.
- (c) k90(FR) <= ½ of the smallest baseline k90.
  - k90 is the smallest k at which a threshold fixed on tune episodes at 5% FPR (in distribution) reaches >= 90% TPR with <= 10% FPR on the primary slice.
  - It is censored above 128, and a censored baseline counts as 256. (c) fails if FR is censored.

**Falsifiers:**
- F1, prior-free Bayes suffices: at every k in {8, 16, 32} on the primary slice, max(C4, C5, S1, S2) is within 0.02 AUROC of FR, **and** within 0.05 in Spearman(V_k, exact CMI) over the primary slice.
- F2, library matcher: the precondition is met, the mean novelty is >= 0.05, and AUROC(FR) <= AUROC(L-train) + 0.02 on the primary slice.
- F3, an external tool suffices: AUROC(E1) >= AUROC(FR) − 0.02 on the paired subsample.
- F4, hallucinated structure: either clause below fires.
  - At the 80%-TPR threshold (from the primary slice's positives), FR's raw FPR is > 15% on the pooled held-out negatives or > 25% on any single negative type (T0, T3a, HN1-HN4).
  - The adversarially selected CMI-0 bit's settlement gain has a 95% CI above 0.
- F5: untestable (S3 dropped).
- F6, the pay is not the ingredient: AUROC(FR-direct) >= AUROC(FR) − 0.02 on the primary slice. This uses the mean over FR-direct seeds.

**Status rule for the final report:**
- "Gain" only for a PASS with a large margin: (a)'s CI lower bound >= 0.05.
- "Informative" for a clean falsifier or a clean failure of (a)-(c).
- "Problem" for an undertrained or broken run.

## Reported, not part of PASS

- AUROC vs k (4, 8, 16, 32, 64, 128) per slice (ID, held-out, both cross slices, leak), per stratum, per estimator.
- Per-null FPR at 80% TPR.
- Spearman(V_64/64, CMI) on informative bits.
- Consumer Spearman (V_16 vs CONS), all bits and T3b alone.
- Fraction of oracle.
- FR − L-train on held-out types.
- FR-direct − FR.
- Calibration.
- Compute: FLOPs per valuation, wall-clock, amortized F_meta, TabPFN call time.

## Budget and drop order

- Watchdog at $13 (all trader-b3-p16-* VM-hours at $0.71/h).
- Drop order as the estimate nears $13:
  1. the adversarial slice and CRT;
  2. E1 test episodes (halve);
  3. FR-direct seed 2;
  4. FR seed 3. A PASS on 2 seeds is marked provisional.

## Addendum A (2026-09-29 23:20Z): Stage 0.5 gate decision, written before any FR training

The text above this addendum is the frozen pre-registration: its first 12541 bytes hash to sha256 668f1d4e82e72ee95ab19cfc6cb1fd9015a860c1ab6a24418a74a98dee775ab2 (`head -c 12541 PREREG.md | sha256sum`; the same hash is in PREREG.sha256).

**Tuned baselines** (tune split, 504 ID episodes, ID AUROC at k = 16; `results/vm_out/tuned.json`):
- C4: alpha 8, rho 0.25, lambda 1 (AUROC 0.774).
- C5: the C4 settings, scale 2 (0.574).
- S1: lr 3e-3, 10 steps (0.581).
- S2: lr 1e-3, 10 steps (0.557).

**Gate result** (pilot namespace, 504 HP positives vs 504 HN negatives, k = 16; `results/vm_out/gate.json`; the 95% intervals are a class-stratified episode bootstrap, 2000 resamples, reported in addition to the point rule; `gate_ci.json`):
- (i), full pilot: AUROC L-all 0.966, C4 0.733, C5 0.637. The difference is +0.233 [0.204, 0.263], so it passes (>= 0.05).
- (i), on E1's paired subsample (96 + 96 episodes): L-all 0.962, E1 0.587, C4 0.762, C5 0.677. The difference is +0.200 [0.139, 0.263], so it passes.
- (ii): mean novelty is 0.194, which passes (>= 0.05). By type: HP1 0.307, HP2 0.123, HP3a 0.183, HP3b 0.162, HN1 0.213, HN2 0.158, HN3 0.151, HN4 0.253.
- **Decision: both gates pass. FR training proceeds** as pre-registered: FR seeds 0-2, FR-small seed 0, FR-direct seeds 0-1, F_meta 1e16.

**Operational notes** (none changes a decision rule):
- **Preemption.** Spot VM trader-b3-p16-1 was preempted mid-pilot at about 21:30Z. The pilot L and E1 arms were re-run per type on trader-b3-p16-4, with the same code, seeds and episodes. Every tune and pilot C4/C5 result predates the preemption and was sha256-synced.
- **Undefined L-all scores.** In 3 of the 1008 pilot episodes, the L-all library has no instance on one side of the useful/useless split, so the likelihood ratio is undefined. Those episodes are left out of L-all's AUROC only.
- **E1 cost.** Measured at 0.63 s per TabPFN call (9216 calls) with 4 processes sharing the VM, against 0.38 s alone. The E1 test subsample stays as planned: 12 per type-stratum on the held-out types and 5 on the ID types.
- **Run layout.**
  - Plan trainA: FR seeds 0-2 and FR-small, their test/tune evals, the adversarial slice (FR seed 0), and E1 on test.
  - Plan trainB: FR-direct seeds 0-1 and their evals, L and all baselines on test and tune, and CONS.
  - The two plans run on two L4 spot VMs, resuming from markers and checkpoints.

## Addendum B (2026-09-30 00:23Z): FR seed 0 misses the precondition; one exploratory run added

Written while FR seeds 1-2, FR-small, FR-direct and the test baselines are still running. No decision rule above changes.
- **What was seen.** FR seed 0 finished its 1e16-FLOP budget (4915 steps). Validation NLL went from 0.575 to 0.465 and improved by 0.29% over the last 20% of steps, so the first precondition clause holds. Its test ID AUROC at k = 16 is 0.719, below the 0.80 the precondition needs. The precondition uses the mean over the 3 FR seeds, so the pre-registered verdict is very likely "inconclusive (undertrained)" (status "Problem").
- **Protocol note.** The same preview also showed FR seed 0's held-out primary AUROC by k (0.568, 0.617, 0.701, 0.807, 0.883, 0.927 at k = 4 ... 128). The precondition is meant to be read before held-out results; since every rule was already frozen, this changes nothing, but it is stated here.
- **Exploratory run (not pre-registered, enters no decision rule).** Plan trainX: FR seed 0 retrained from scratch at F_meta = 4e16 (4x), with the same code, data stream and schedule. It is evaluated on the same test and tune episodes, and written to a separate output tree (`results/vm_out/x/`) that the final analysis never reads. With FR-small (1e15) and FR (1e16), it gives a three-point compute curve for ID AUROC at k = 16. It starts only when one of the two VM slots frees, and it stays inside the $13 watchdog.
- **CRT debiasing** (simplification 6) was not built, and is not run (drop-order item 1).
