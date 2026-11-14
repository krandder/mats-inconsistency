# Fully non-Gaussian families on deep critical ReLU nets: do they scale on depth? (29 October 2026)

Lane: nongauss-family. Work directory: `work/nongauss-family/`. 16:55Z to about 20:30Z.

Brief (Kelvin, via main): "we should just stop assuming any gaussian part whatsoever then. Let us seek a family that is
fully non-gaussian, even if its bad at the input, and hope that it at least scales on depth." The target is M1, a
sample-free estimator that matches sampling at depths that grow with width. This lane asks the representation question
first. Each family is fitted to the true state at a cut. The true network runs from there, and we check whether the
output error stays flat as depth grows.

Later questions from Kelvin, relayed by main at about 18:00Z and 18:10Z, are answered in section 4:
- "How can you tell it's not just that we haven't found the right structure that has low-error per layer, or that can
  even self-correct?"
- a "bidirectional measure of importance of directions" (Lyapunov vectors);
- a learned compressor.

Every test was pre-registered and hashed before it ran: `PREREG.md` with four amendments and a deviations list, hashes
in `prereg_hashes.txt`. One GPT-6 design critique and one GPT-6 claims review were taken; section 7 lists what changed.
All numbers are oracle numbers: families are fitted to 2^20 true paths. Nothing here is a sample-free estimator, and no
costs are claimed. The scope is one noise level (sigma = 0.001) and one depth-to-width ratio (L/n = 1/4). Single cuts
used four nets per size; composition and Lyapunov used two, and the compressor one.

## Summary

**None of the fully non-Gaussian families we built was shown to scale on depth.** By "scale" we mean passing both the
single-cut test and the layer-by-layer composition test at a useful level. What helped was carrying more of the state's
*joint* law. Making the pieces of a family non-Gaussian helped little.

1. **The scale is free, up to the noise.** Without noise, positive homogeneity means a family needs only E|z| and the
   |z|-weighted law of the direction z/|z|. The remembered scale (sd of log|z| about 0.19) then drops out exactly.
   With sigma = 0.001 the identity is not exact, because the noise does not scale with |z|. The exact direction law
   with an independently drawn radius (PEMP) stays within the pre-registered tolerance of the resampled fit paths
   (test T0: pass; differences -4e-7 and -2e-7, intervals within +-2e-6).
2. **The original families' errors grow after deep cuts** (256 x 64, four nets, cuts at layers 32 and 48, output at 64).
   - From one layer after the cut to the output, every family grows about 5-10x on average, Gaussian or not. Mean nMSE
     at L = 64 from cut 32:
     - one Gaussian (G): 2.8e-4;
     - all principal components independent with exact non-Gaussian marginals (PI): 1.8e-4;
     - two-point components: 3.4e-4;
     - 64 cells with exact-marginal residuals (PC3I): 1.9e-5;
     - the top-32 components drawn jointly (PB32): 1.6e-5;
     - fit-set floor: 7e-7.
   - No fully non-Gaussian family from the original pre-registration passes test T1. Three fail, and ten are
     inconclusive because the four-net interval is wide (net 114's errors are 2-4x the others'). On every net each of
     them is above 1e-5 at cut 32, except PB32 on one net.
   - Exceptions and limits:
     - Two later families grow less: PBS16 about 1.5x, and PBA16 passes the cut-32 growth test.
     - From the shallower cut 16, errors peak 12-18 layers after the cut and then fall by 15-40% (e.g. PI 1.75e-4 at
       layer 28, 1.0e-4 at 64).
   - Non-Gaussian marginals buy 1.2-1.7x. Inside cells, Gaussian residuals are within 1.2-1.6x of exact marginals.
     Transferring the shape from 8 or 16 layers earlier, or from another net (the "attractor"), is at G's level.
3. **The error falls as the jointly carried block grows.** The block is the top-k weighted principal components of the
   direction law, drawn jointly from one fit path; the other components are drawn independently with exact marginals.
   - Cut 32 to L = 64: 1.8e-4 (k = 0), 1.1e-4 (8), 4.4e-5 (16), 1.6e-5 (32), 1.4e-6 (64), 7.7e-7 (128).
   - PB64 and PB128 pass T1's rule (Amendment 1). Their blocks are 64- and 128-dimensional empirical joint laws, i.e.
     structured sampling, not mechanistic families.
4. **How many directions are needed as width grows** (L/n = 1/4: 128 x 32, 256 x 64, 512 x 128; Kelvin's tests). Three
   measures say the count stays roughly constant. The PB ladder's pre-registered verdict is "about linear in n", and
   that verdict is not resolved.
   - **Lyapunov, per path.** The downstream Jacobian keeps 13 / 8 / 4 directions within e^-1 of its top singular value
     at L'/n = 1/32, 1/16, 1/8, the same at all three widths (pre-registered rule: O(1)). That is 4-5x fewer than the
     Gaussian-product value 2n/L'. These counts are per path and relative to each path's top direction, so they do not
     identify one common subspace.
   - **Bidirectional.** State spread x downstream sensitivity puts 90% of its weight in 11-13 directions and 99% in
     33-46 (rule: "few", 1.4-1.5x from 128 to 512).
   - **Learned compressor** (net 113, cut at 3L/4). A 16-dimensional nonlinear code (MLP) reproduces each path's output
     to 0.84%, 0.40% and 0.48% of its variance at the three widths (rule: compact).
     - Through its deterministic decoder, the bias of the mean is 6e-6, 1.6e-6 and 3e-6.
     - At the earlier cut L/2 (256) it needs 64 dimensions, and still misses 1% (1.05%).
     - The shuffled-residual bias reaches 1e-5 only for a PCA code with k = 64; no MLP code reaches it.
   - **PB ladder.**
     - At fixed k the error does not grow with width (PB32 at the L/2 cut: 2.1e-5 / 1.6e-5 / 1.4e-5).
     - The smallest block that reaches the floor is 64 at 128 and 256, and 256 at 512 (pre-registered rule: "about
       linear in n").
     - The 512 values decide this, and they sit near that size's noise. PB64 and PB128 are at 3-4e-6, against a
       2.5e-6 threshold at L/2. PB128 is at 5e-7, against a 4e-7 threshold set by a negative mean floor, at 3L/4.
       Those runs also had half-size fit sets.
5. **Re-projecting at every layer accumulates error, even for good families** (from layer 32 to 64, 2^21 particles;
   plain Monte Carlo and the bootstrap chain are at or below 7e-6 at 256 x 64):
   - G, PG, PI: 4e-4 (net 113) and 2.3-2.8e-3 (net 114);
   - PB16: 2.2e-4 and 7.7e-4; PC3I: 2.8e-4 and 3.1e-4;
   - PB64: 5.7e-5 and 3.2e-5, higher at the output than at layer 48 on both nets;
   - PB128: 1.4e-6 and 1.2e-5;
   - at 128 x 32, PB64 (half of all directions): 3.3e-5 and 8e-6.

   The pre-registered rule calls G, PG, PI, PB16 and PC3I unbounded. The large-block chains are inconclusive: the rule's
   bias extrapolation is dominated by particle noise, and it even labels the 128 x 32 bootstrap control "unbounded".
   For the good families, single cuts understate the composed error by 5-40x.
6. **Over the observed depths the direction law has a stable scale but a changing shape.** These are statistics in the
   gnomonic chart, centred on the mean direction.
   - The angular spread falls like 5/l, and the log-size spread saturates.
   - From layer 16 to 63 the participation ratio falls from 18 to 6, and the top component becomes more platykurtic
     (excess kurtosis -0.25 to -0.65).
   - This does not exclude an attractor at larger depth. The PCA-based shape transfer we tried does not find one.

**For M1.** Under these tests, the representations that did well were large joint blocks (64-128 dimensions at
256 x 64) or a learned nonlinear 16-64-dimensional code. The structured families all stayed well above the floor in
single cuts and accumulated error under composition: moments, cells with Gaussian or independent residuals, attractor
shapes, and subspaces of at most 32 directions chosen by variance, sensitivity or both. Three of four width measures
suggest the needed number of directions does not grow with width at fixed L/n. That leaves open a low-dimensional
representation whose joint law is pushed layer by layer with very small error. None of what we built is that, and these
results do not show that one must exist, or that it must take this form.

## Handoff

**Code** (`work/nongauss-family/`; copied from `work/boundary-family` and `work/arc-noisy`, which were not edited):
- `lib/ng.py`: the pre-registered families, the sphere chart, propagation, composition and floor-corrected scores.
- `lib/ng2.py`: the families of Amendments 1-2 (PB64/128, PBA, PBS, PBW).
- Runners: `scratch/run.py` and `scratch/run2.py` (single cuts, `lin`, composition `e2`).
- Tests: `scratch/amp.py` (sensitivity spectra), `scratch/lyap.py` (Lyapunov), `scratch/comp.py` (compressor),
  `scratch/e0.py` (descriptive).
- Analysis: `scratch/analyze.py` and `scratch/analyze2.py` produce `results/tables.md` and `results/tables2.md`;
  `scratch/e0_table.py` produces `results/e0_table.md`.
- GPU: `scratch/launch_gpu.sh`, `scratch/gpu_job*.sh`; `scratch/retry_launch.sh` retries through L4 stockouts.

**Records:**
- `DESIGN.md`: v1, plus section 6 for the changes after the critique.
- `PREREG.md`: amendments 1-4 and deviations; hashes in `prereg_hashes.txt`.
- `review/`: the design critique and the claims review.
- `GPU-LOG.md`: 4.9 L4-hours, $4.22 ceiling of the $5 cap; all VMs deleted.

**Raw outputs** (`runs/gpu/`):
- `ngfam1/`: r1 single cuts, r2 composition, r3 128 x 32 and 512 x 128, r4 linearised sum (net 113), and the truths;
- `ngfam2c/a1`, `ngfam3/a3`, `ngfam4/a2` (T5a, T5b, T6), `ngfam5t2/a4`;
- `runs/cpu_a2` (T5b at 512 on the CPU) and `runs/e0`.

The NPZ files keep the mean vectors, not the second moments. The spectrum and compressor JSONs keep summaries, not full
spectra or trained models.

**Nets:**
- 113-116 at 128 x 32, 256 x 64 and 512 x 128;
- 112 at 256 x 64 (exploration and the ATTX reference);
- 64 x 16 versions of 112-113 for code checks.

Truths: 2^24 Monte Carlo paths (2^23 at 512).

## 1. Setup

**Nets and estimand.** ARC's family as in `work/arc-noisy`: x ~ N(0, I_n), W iid N(0, 2/n), no bias. The noise is
sigma r_l after each ReLU, with sigma = 0.001 and r_l fixed by a seeded pilot. The estimand is E[a_L] at every output
layer L. The metric is ARC's nMSE, minus the truth's floor 1/N and the estimate's own sampling floor
tr Cov_F(a_L) / (N tr Var a_L).

**The measurement.** R_F(l; L): fit family F to 2^20 true paths of z_l, draw 2^21 samples from it, and run the exact
noisy layers l+1..L. The main families have two independent replicates. Composition refits F at every layer and draws
fresh particles.

**The chart.** For degree-1 homogeneous f and no noise, E f(z) = E|z| E_nu f(w), with w = z/|z| and nu the
|z|-weighted law of w.
- Polar families fit nu with weights |z|, draw a radius independently, and output (radius) x (sample).
- Their samples are ambient approximations fitted to sphere coordinates. A sampled direction need not have unit norm,
  so the effective radius is (radius) x |sample|, where |sample| = 1 + O(spread^2).
- The sphere chart replaced v1's gnomonic chart after the critique.

**Families** (all oracle fits; the definitions are the docstrings of `lib/ng.py` and `lib/ng2.py`):

| family | what it carries | Gaussian part |
|---|---|---|
| EMP | resample of the fit paths (the floor) | none; weighted points |
| PEMP | exact weighted direction law, independent radius | none; homogeneity control |
| G, PG | one Gaussian on z / on the direction | yes (controls) |
| PCA3, PC3G | 64 quantile cells on the top-3 PCs, Gaussian within (z / sphere) | yes (controls) |
| I, PI | every PC score independent with its exact marginal (z / sphere) | none |
| PR | PC scores two-point (+-sd) | none |
| PB_k | top-k PC scores drawn jointly (one fit path), the rest independent with exact marginals | none |
| PC3I, PC3R | the cells, with exact-marginal / two-point independent components inside | none |
| ATT8, ATT16, ATTX | joint whitened PC scores of layer l-8 / l-16 / net 112, recoloured with layer l's PCs | none |
| PBA_k, PBS_k, PBW_k | joint block on: top-k PCs plus top-k downstream-sensitivity directions / sensitivity only / Kelvin's bidirectional directions (state spread x sensitivity); the rest regressed on the block, then independent with exact marginals | none |

Every family except EMP and PEMP reproduces the covariance of its chart, up to sampling.

## 2. Single cuts at 256 x 64 (R1, Amendment 1)

Mean nMSE over nets 113-116 (main families: replicate mean of two). The full tables are `results/tables.md` and
`results/tables2.md`.

| family | R(32;33) | R(32;40) | R(32;64) | R(48;49) | R(48;56) | R(48;64) |
|---|---|---|---|---|---|---|
| EMP (floor) | 1.1e-6 | 9.7e-7 | 7.5e-7 | 1.0e-6 | 8.3e-7 | 1.0e-6 |
| PEMP | 4.1e-7 | 4.2e-7 | 3.7e-7 | 7.9e-7 | 7.0e-7 | 8.5e-7 |
| G | 3.8e-5 | 1.4e-4 | 2.8e-4 | 2.6e-5 | 8.4e-5 | 2.1e-4 |
| PG | 3.2e-5 | 9.8e-5 | 2.5e-4 | 1.9e-5 | 6.7e-5 | 2.4e-4 |
| I | 3.4e-5 | 1.3e-4 | 2.6e-4 | 2.2e-5 | 6.1e-5 | 1.3e-4 |
| PI | 2.4e-5 | 7.6e-5 | 1.8e-4 | 1.4e-5 | 4.5e-5 | 1.4e-4 |
| PR | 4.2e-5 | 1.5e-4 | 3.4e-4 | 4.1e-5 | 1.5e-4 | 4.8e-4 |
| PB2 / PB4 / PB8 | 2.2 / 1.9 / 1.3e-5 | 7.6 / 6.3 / 5.2e-5 | 1.9 / 1.5 / 1.1e-4 | 1.4 / 1.1 / 0.7e-5 | 4.1 / 2.8 / 1.8e-5 | 1.2 / 0.8 / 0.6e-4 |
| PB16 | 6.3e-6 | 2.2e-5 | 4.4e-5 | 3.6e-6 | 8.3e-6 | 2.3e-5 |
| PB32 | 2.6e-6 | 7.2e-6 | 1.6e-5 | 9.9e-7 | 2.0e-6 | 6.0e-6 |
| PB64 | 7.8e-7 | 1.5e-6 | 1.4e-6 | 1.0e-7 | 1.8e-7 | 7.5e-7 |
| PB128 | 5.1e-7 | 4.9e-7 | 7.7e-7 | 5.9e-7 | 5.6e-7 | 7.1e-7 |
| PCA3 | 7.6e-6 | 2.8e-5 | 4.2e-5 | 2.8e-6 | 7.8e-6 | 1.7e-5 |
| PC3G | 4.1e-6 | 1.5e-5 | 3.0e-5 | 1.6e-6 | 4.5e-6 | 1.0e-5 |
| PC3I | 3.7e-6 | 1.2e-5 | 1.9e-5 | 2.2e-6 | 4.2e-6 | 8.5e-6 |
| PC3R | 6.8e-6 | 2.3e-5 | 6.7e-5 | 2.3e-6 | 6.8e-6 | 1.5e-5 |
| ATT8 / ATT16 / ATTX | 3.7 / 3.0 / 4.3e-5 | 1.4 / 1.0 / 2.0e-4 | 2.3 / 2.1 / 4.2e-4 | 2.8 / 2.0 / 2.7e-5 | 7.4 / 6.2 / 7.8e-5 | 1.7 / 2.2 / 1.9e-4 |
| PBA8 / PBA16 | 9.7 / 3.0e-6 | 2.5e-5 / 5.9e-6 | 2.1e-5 / 6.0e-6 | 4.2 / 1.3e-6 | 9.3 / 2.1e-6 | 2.2e-5 / 3.2e-6 |
| PBS16 | 1.5e-5 | 3.0e-5 | 2.3e-5 | 6.3e-6 | 1.1e-5 | 2.7e-5 |
| PBW8 / PBW16 / PBW32 | 13 / 6.7 / 1.9e-6 | 43 / 19 / 5.6e-6 | 7.0e-5 / 3.9e-5 / 9.0e-6 | 7.5 / 3.3 / 0.5e-6 | 19 / 7.2 / 1.1e-6 | 5.1e-5 / 2.0e-5 / 4.2e-6 |

**Pre-registered verdicts.**
- **T0** (PEMP - EMP <= 2e-6): pass at both cuts (-3.8e-7 [-1.9e-6, 1.1e-6]; -2.0e-7 [-1.2e-6, 8.1e-7]). T0 compares the
  two squared errors against the truth, not the distance between the two estimates.
- **T1.** The rule needs, with the 4-net t-interval:
  - level R(32;64) <= 1e-5 and R(48;64) <= 1e-5;
  - growth R(32;64) - R(32;40) <= 3e-6 and R(48;64) - R(48;56) <= 3e-6.

  Results:
  - Fail: G, I, PCA3, PC3G, PC3I and ATTX.
  - Inconclusive: PG, PI, PR, PB2-PB32, PC3R, ATT8 and ATT16.
  - Pass: PB64 and PB128 (Amendment 1 families, scored with T1's rule).
  - Partial: PBA16 passes the cut-48 level and the cut-32 growth test; PBW32 passes the cut-48 level; their other tests
    are inconclusive.
- **H1** (does adding sensitivity directions help at equal block size?): not supported. PBA8 is within 1.5x of PB16 on
  all four nets at cut 48; at cut 32 it is 1.4-3.7x better.
- **H2** (block size needed for the floor at 256 x 64): k = 64 at both cuts.

**Readings.**
- **Growth.** Growth after a deep cut is common to every original family. Its size does not depend on whether the
  family has a Gaussian part.
- **Coherence.** The output error vectors of cuts 32 and 48 are positively aligned for G: cosine 0.47, 0.46, 0.41 and
  0.09 on the four nets.
- **Direction of the error.** Most of the error is orthogonal to the true mean vector: 77% for G, 83% for PC3I and 91%
  for PB16. So it is not mainly a scale error.
- **Chart.** The polar chart alone gains at most 1.7x: G 2.8e-4 against PG 2.5e-4 at cut 32, and PG is worse at cut 48.
- **Other lane.** The boundary-family lane's oracle study (nets 101-110, 256 x 32, cut 24, Gaussian cells) shows the
  same pattern at layer 32: G 5.6e-5 to 3.5e-4, PCA3 1.0e-5 to 5.5e-5, growing 2.4-9.2x (G) and 2.8-11.8x (PCA3)
  over the 8 layers after the cut. Their exact near-boundary marginals (PCA3NB) lower PCA3's error at layer 32 by about
  11%.

## 3. Composition: re-projecting at every layer (R2, R4, Amendment 4)

The chain refits F at every layer from l0 and draws fresh particles. Values are nMSE at the output with 2^21 particles,
for nets 113 / 114.

| chain | 256 x 64, L = 48 | 256 x 64, L = 64 | 128 x 32, L = 24 | 128 x 32, L = 32 |
|---|---|---|---|---|
| plain Monte Carlo | 1.9e-6 / -4e-7 | 2.7e-6 / -5e-7 | 0 / -3e-7 | 0 / -4e-7 |
| bootstrap (EMP) | 4e-7 / 9.7e-6 | 6.6e-6 / 2.0e-6 | 2.4e-6 / 2.5e-6 | 3.7e-6 / 6.1e-6 |
| G | 2.4e-4 / 4.0e-4 | 4.5e-4 / 2.3e-3 | | |
| PG | 2.4e-4 / 3.6e-4 | 3.9e-4 / 2.8e-3 | | |
| PI | 2.1e-4 / 3.4e-4 | 3.9e-4 / 2.4e-3 | | |
| PB16 | 1.3e-4 / 1.7e-4 | 2.2e-4 / 7.7e-4 | | |
| PC3I | 5.8e-5 / 7.1e-5 | 2.8e-4 / 3.1e-4 | | |
| PB32 | | | 8.0e-5 / 4.7e-5 | 2.5e-4 / 6.9e-5 |
| PB64 | 2.1e-5 / 1.2e-5 | 5.7e-5 / 3.2e-5 | 1.7e-5 / 2.1e-6 | 3.3e-5 / 8.2e-6 |
| PB128 | 1.3e-6 / 1.0e-5 | 1.4e-6 / 1.2e-5 | | |

**Verdicts.**
- **T3** (pre-registered): G, PG, PI, PB16 and PC3I are "unbounded"; MC and EMP at 256 x 64 are "bounded and useful".
- **Amendment 4:** PB64, PB128 and PB32 at 128 are inconclusive. The bias extrapolation B = 2C(2^21) - C(2^20) is
  dominated by particle noise: the bootstrap chain alone gives B = -7e-5 on net 113. The literal rule even labels the
  128 x 32 bootstrap control "unbounded". So the large-block chains' asymptotic bias near 1e-5 is unresolved.

**Readings.**
- **The best families lose most of their single-cut advantage.** PC3I goes from 1.9e-5 to 3e-4 and PB16 from 4.4e-5 to
  2-8e-4, while G goes from 2.8e-4 to 4e-4 and 2e-3. PB64 ends 20-40x above its single-cut error. Its 32-layer curves
  are not monotone, but are higher at the output than at layer 48 on both nets.
- **Re-projection errors are not additive.** The linearised sum of the 32 single-cut error vectors (R4, net 113)
  predicts nMSE 7e-2 for G and 6e-3 for PB16, against measured chains of 4.5e-4 and 2.2e-4 (155x and 27x). A chain
  projects distributions that are already approximate, not the true ones, so a sum of single-cut errors is the wrong
  model for it.
- **Two cuts compose nearly additively, with a positive cross term.** G on net 114: 1.3e-3 against 4.8e-4 + 5.1e-4;
  the interaction residual |d|^2 is 4e-5.

## 4. Kelvin's questions: is there structure we have not found? (Amendments 2-3)

**Per-path Lyapunov spectra (T5b).** The finite-time singular values of d a_{l+L'} / d z_l were computed on 128 paths
of nets 113-114. The 512 runs were done on the CPU; see section 8.

| L'/n | N_surv(e^-1) at 128 / 256 / 512 | N_surv(e^-2) | Gaussian-product 2n/L' |
|---|---|---|---|
| 1/32 | 13.1 / 13.7 / 13.3 | 27 / 31 / 32 | 64 |
| 1/16 | 7.2 / 8.0 / 8.0 | 16 / 18 / 18 | 32 |
| 1/8 | 4.0 / 4.4 / 4.3 | 9 / 10 / 10 | 16 |

- The per-layer gap between the first two exponents scales as 1/n: 0.030, 0.016 and 0.008 at L'/n = 1/32. That is
  about 8x the Gaussian-product value 1/(2n).
- Rule: O(1) at every L'/n (ratio 1.01-1.12). Kelvin's prediction (about 2n/L' directions survive, O(1) at fixed
  L'/n) holds in form, with 4-5x fewer directions.
- These counts are relative to each path's top direction. They neither identify a common subspace nor measure
  absolute contraction.

**Spread-weighted downstream sensitivity (T5a, the bidirectional measure).** S = R M R, with M = E[J^T J] and R the
square root of the state covariance; the radial direction is removed. Nets 113-116; numbers are the directions holding
90% / 99% of S's trace.

| cut | 128 x 32 | 256 x 64 | 512 x 128 |
|---|---|---|---|
| L/2 | 11 / 33 | 12 / 39 | 13 / 46 |
| 3L/4 | 8 / 25 | 9 / 26 | 11 / 37 |

- Rule: "few" (k99 grows 1.4-1.5x for 4x width).
- The raw sensitivity P M P is spread wider and grows faster: k99 = 83 / 117 / 179 at L/2.
- As the output horizon grows, S concentrates slightly, while its trace grows (at 256, cut 32: k99 44 -> 39, trace
  4.4 -> 6.7). Concentration of this kind is not self-correction.

**A family exact in those directions does not stop the growth (T5c).**
- PBW_k (the joint block on the top-k bidirectional directions) is 1.1-1.7x better than PB_k at equal k.
- Its mean error still grows after the cut: PBW32 goes from 1.9e-6 to 9.0e-6 from cut 32, and from 5e-7 to 4.2e-6
  from cut 48. By the rule's intervals these growth tests are inconclusive.
- This is consistent with Kelvin's caveat that gate flips are nonlinear and the linear Lyapunov picture misses them.
  It does not isolate that cause.

**Block size against width (Amendment 3, H3).**

| cut | 128 x 32: PB32 / PB64 | 256 x 64: PB32 / PB64 / PB128 | 512 x 128: PB32 / PB64 / PB128 / PB256 |
|---|---|---|---|
| L/2 | 2.3e-5 / 2.7e-6 | 1.6e-5 / 1.4e-6 / 7.7e-7 | 1.4e-5 / 3.2e-6 / 3.6e-6 / 1e-7 |
| 3L/4 | 9.8e-6 / 3.0e-6 | 6.0e-6 / 7.5e-7 / 7.1e-7 | 8.5e-6 / 3.7e-6 / 5.2e-7 / -5e-7 |

- Under the pre-registered rule (threshold 3 x mean floor + 2e-6), k* = 64 / 64 / 256 at both cuts, ratio 4: "about
  linear in n".
- My first draft clipped a negative floor to zero, which is not what was pre-registered. That clipping gave 128 at
  3L/4 and the label "sublinear"; section 7 has the correction.
- The verdict rests on 512 values near that size's noise:
  - fit sets half the size (2^19), one replicate;
  - per-net PB128 values of 0.4-6e-6 at L/2;
  - a 4e-7 threshold at 3L/4.

  So k* at 512 is not resolved between 64 and 256.
- At fixed k the error does not grow with width (R3; ratios 512 / 128 of 0.18-0.84 for every approximate family).

**Learned compressor (T6, net 113).** A direction encoder maps to a k-dimensional code and a decoder maps back; the
radius is kept. It is trained end to end through the true layers on the pathwise output error (3000 Adam steps).

| size, cut | channel floor | MLP pathwise nMSE, k = 1 / 4 / 16 / 64 | MLP mean bias, k = 16 / 64 | shuffled-residual bias, PCA k = 64 |
|---|---|---|---|---|
| 128 x 32, 24 | 9.1e-5 | 0.24 / 0.089 / 0.0084 / 0.0049 | 5.7e-6 / 2.7e-6 | 4.4e-7 |
| 256 x 64, 48 | 5.2e-4 | 0.11 / 0.046 / 0.0040 / 0.0015 | 1.6e-6 / 4.9e-7 | 1.3e-6 |
| 512 x 128, 96 | 1.4e-3 | 0.082 / 0.033 / 0.0048 / 0.0012 | 3.1e-6 / 2.7e-7 | 1.2e-6 |
| 256 x 64, 32 | 1.0e-3 | 0.16 / 0.070 / 0.021 / 0.011 | 2.2e-5 / 3.8e-6 | 5.1e-6 |

- The channel floor is the output variance that the activation noise after the cut creates by itself, a comparison
  scale. The evaluation shares the downstream noise, so the floor is not a lower bound on this loss.
- T6 rule: "compact structure". k_path (pathwise <= 1%) is 16 at all three widths at 3L/4, from the MLP. k_bias
  (shuffled residual <= 1e-5) is 64 at all three, from the PCA code; no MLP code meets it.
- At the L/2 cut (256), the MLP misses 1% even at k = 64 (1.05%). The trained linear bottleneck reaches it (0.95%).
- At k = 16, a trained linear bottleneck is about 3x worse than the MLP, and a PCA projection 7-9x worse.

**What this answers, and what it does not.**
- Three measures say the number of directions the downstream net cares about is tens, and grows little or not at all
  from width 128 to 512 at fixed L/n: the per-path Lyapunov count, the bidirectional sensitivity count, and the
  learned code size. The PB ladder's literal verdict says "about linear", but at 512 it is within noise.
- Our structured families did not fail for want of a low-dimensional subspace. Families exact in the top 32 directions
  chosen by variance, sensitivity or both still grow after the cut. The large blocks (64-128) that reach the floor in
  single cuts accumulate under composition.
- **Limits.** This is not a proof that no compact deductive family exists:
  - one noise level, one L/n, four or fewer nets per measure;
  - the compressor was trained on samples, on one net;
  - a learned code whose latent law is pushed exactly was not tried.

## 5. The direction law across depth (E0, descriptive; nets 113-116, 256 x 64; gnomonic chart)

| layer | spread | spread x l | PR | sd log rho | PC1 skew / excess kurtosis | gate-relevant neurons | R^2 of their pre-activations on top-4 / top-16 PCs |
|---|---|---|---|---|---|---|---|
| 16 | 0.29 | 4.7 | 18 | 0.180 | +0.13 / -0.25 | 52 | 0.31 / 0.71 |
| 32 | 0.15 | 4.8 | 11 | 0.190 | +0.24 / -0.37 | 33 | 0.49 / 0.83 |
| 48 | 0.10 | 4.8 | 7.7 | 0.191 | +0.21 / -0.48 | 22 | 0.59 / 0.90 |
| 63 | 0.09 | 5.6 | 5.9 | 0.194 | +0.20 / -0.65 | 17 | 0.69 / 0.93 |

E0 uses the gnomonic chart: rho = u . z with u the mean direction, and delta = z / rho - u. The families use the sphere
chart; the two agree to first order in the spread.
- The spread falls like 5/l, and the size spread saturates by about layer 30.
- Over these depths the shape keeps changing: the dimension keeps collapsing and the top component becomes more
  platykurtic. That fits repeated folding at the gates.
- The scale mixture accounts for most of z's per-neuron non-Gaussianity: median |skew| is 0.44-0.49 for z and
  0.16-0.25 in the chart.

## 6. Predictions against results (PREREG.md)

| prediction | result |
|---|---|
| P0 T0 passes | held |
| P1 G fails T1; R(32;64) 1-4e-4 and growing; R(48;64) > 3e-5 | held (2.8e-4; 2.1e-4) |
| P2 PG within 2x of G | held |
| P3 PI 2-5x better than G, fails the level at cut 32; PR and I worse than PI | partly: PI is only 1.5x better; its T1 verdict is inconclusive (wide interval), though it is above 1e-5 on every net; PR and I are worse (held) |
| P4 PB ladder monotone; PB32 may pass the level at cut 48 | held (PB2 about equal to PI; PB32 6e-6 at cut 48, interval inconclusive) |
| P5 PC3I the best fully non-Gaussian family; no family passes T1 | partly: PB32 is slightly better than PC3I; no original family passes |
| P6 ATT no better than PG | held |
| P7 G and PI width-growing or flat; PB16 width-growing | missed: every approximate family's error falls with width at fixed L/n |
| P8 G, PG, PI compositions unbounded; PB16 and PC3I above 1e-5 | held (all five unbounded) |
| P9 G's cut-32 and cut-48 error cosine > 0.3 | held on 3 of 4 nets |
| A1: H1 not supported; PB64 3-5e-6 and PB128 at the floor at cut 48; PB128 above 5e-6 at cut 32 | H1 held; PB64 was better (7.5e-7); PB128 at the floor at both cuts (missed) |
| A2: T5a sublinear | missed: "few" (1.4-1.5x) |
| A2: T5b N_surv 3-8 and O(1) | O(1) held; 13 at L'/n = 1/32 (outside 3-8) |
| A2: PBW_k within 2x of PB_k and failing growth at cut 32 | within 2x held; growth inconclusive by the rule (mean growth positive for every k) |
| A2: T6 no k <= 64 reaches 1% pathwise; channel floor 1e-5 to 1e-4; k_bias grows | missed on all three |
| A3: k* about linear in n | held by the literal rule, but the 512 values are within noise |
| A4: PB128 bounded and useful; PB64 inconclusive; at 128 PB64 bounded and PB32 unbounded | all inconclusive (noisy B); PB64 composes to 3-6e-5 |

## 7. Review

**Design critique** (GPT-6, `review/design_critique.md`; 15 findings, 4 critical). All were taken; DESIGN.md section 6
lists the changes:
- the missing ReLU in the exploration gate census;
- the sphere chart;
- PEMP as the homogeneity control;
- weighted fits and family-specific floors;
- more particles for composition, with MC and bootstrap controls;
- the fixed-L/n series;
- the PB ladder as the direct independence test;
- no claim of a deductive push.

**Claims review** (GPT-6, `review/claims_review.md`; 14 findings: 1 critical, 10 major, 3 minor). It recomputed 31,612
score entries from the raw outputs (largest discrepancy 4.4e-9). All findings were applied:
1. **(critical) H3 used a clipped floor.** `analyze2.py` used max(mean floor, 0) instead of the pre-registered
   3 x mean floor + 2e-6. Corrected: the literal verdict is "about linear in n" (k* = 64 / 64 / 256), not "sublinear",
   and the uncertainty at 512 is stated. H2 at 256 x 64 is unaffected (floors positive).
2. **The headline and "For M1" overstated failure and necessity.** Reworded to "none was shown to pass both
   representation and composition", with the scope stated.
3. **"The scale drops out exactly" is false with noise.** Restricted to the noiseless identity, plus T0's tolerance.
4. **The polar families are ambient approximations, not laws on the sphere.** Stated in section 1.
5. **The compressor's "under 0.5%" was wrong at 128 (0.84%).** Fixed. The model identities (MLP for k_path, PCA for
   k_bias) and the 1.05% miss at L/2 are now given.
6. **Concentration of the sensitivity spectrum is not self-correction**, since its trace grows. The PBW result does not
   isolate gates. Both reworded.
7. **Composition B is noise-dominated**, and the rule labels a control "unbounded". Both stated; the analysis now
   prints control verdicts.
8. **The growth claim was too universal.** Restricted to the original families at deep cuts, with the exceptions
   (PBS16, PBA16) and the late decline after cut 16.
9. **PB128 carries 128 joint coordinates.** Fixed.
10. **T2's geometric ratio dropped nets with nonpositive corrected errors.** This affects only the EMP and PEMP control
    ratios, which are unresolved; it is now listed in section 8.
11. **E0 is a gnomonic-chart statistic.** Labelled. The ranges are corrected (PR 18 to 6 from layer 16), and "attractor
    in scale only" is softened.
12. **The linear-sum explanation and "rising layer by layer" were wrong.** Corrected.
13. **The parallel-lane numbers were imprecise.** Made exact.
14. **Provenance.** Hashes match. The NPZ files lack second moments, and the spectrum and compressor files keep
    summaries; both are stated in the handoff.

## 8. Deviations and compute

- **Second VM.** T5a, T5b and T6 failed there without running, because amp.py, lyap.py and comp.py were not staged.
  They were rerun unchanged on a fourth VM.
- **T5b at 512.** Stopped on the GPU after 24 minutes: cusolver SVDs failed to converge and fell back to a slow
  method. It was rerun unchanged on the CPU.
- **R4.** Stopped after net 113 to stay within the $5 cap.
- **First-draft analysis.** H3 clipped a negative floor (corrected after the claims review). T2's geometric ratios
  excluded nonpositive values, which affects only the control ratios.
- **Stockouts.** L4 stockouts caused four failed launches; a retry loop over 12 zones found capacity.
- **GPU total.** 4.9 L4-hours, $4.22 at ceiling rates; all VMs deleted (19:51Z).
- **CPU.** E0, T5b at 512 and all code checks ran at nice 19 / ionice idle, with at most 3 processes.
