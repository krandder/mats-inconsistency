# Boundary-aware families on deep random ReLU nets (29 October 2026)

Lane: boundary-family. Work directory: `work/boundary-family/`. Started 14:06Z; this report 17:00Z.

Target: M1, a sample-free estimator that matches sampling on ARC's He-ReLU MLPs at 256 x 32, sigma = 0.001.
Development nets only. CPU only, no GPU.

The two kill tests and the gate measurement were pre-registered (PREREG.md and amendment 1, hashed). Everything
else is labelled as exploration. All claims were checked in a GPT-6 claims review.

## Summary

**The prototypes we built failed both pre-registered development gates, so no evaluation run was made. Three
development nets do not rule out M1 or broader cell families. They do show what blocks these families:**
- in the deep layers about 40-140 gates per layer sit near their boundary, and they are nearly uncorrelated pairwise;
- cells fitted to the true state represent it well on one net out of three;
- we found no way to build such cells from the weights.

- **Gates: no compression by our criteria.** Measured at 64 x 8, 128 x 16 and 256 x 32, i.e. L/n = 1/8.
  - 43-136 gates per layer have entropy above 0.1 bit (layers 6-31). Their marginal entropies sum to 29-83 bits.
  - Pairwise |phi| has median 0.05-0.08 and maximum 0.67. Pairwise statistics do not establish joint independence;
    the gate-entropy census gives the joint picture.
  - Single-linkage grouping by co-switching merges nothing at |phi| 0.8. Grouping by downstream effect merges a
    little at 0.8: at most 136 gates down to 111.
  - The top 3 or 8 PCA directions predict the sign of 26-68% of these gates in at least 90% of paths. We subtracted
    no majority-sign baseline, so this overstates what the latent itself explains.
- **The factored models we built fail (pre-registered test K-F, nets 101 and 102).**
  - The models: an exact PCA latent of 3 or 8 dimensions, with Gaussian residuals that are diagonal given it.
  - They get the next layer's per-neuron variance wrong by 1.6-10%. One Gaussian gets it wrong by 1.5-2.1%.
  - Their one-step mean error is 0.84-8.2 times the Gaussian's; the test required at most 1/3.
  - Unfactored cells (64 Gaussians with full covariances) reached 2.2-4.3 times better than the Gaussian.
  - The test rejects these models only, not every conditionally independent model.
- **Fitted cells represent the state well on one net in three (pre-registered test K1', an oracle diagnostic).
  Section 7 extends this to ten nets: the gain over one Gaussian has median 4.8x (2.4-7.2x) and tracks how
  collapsed the net is.**
  - The setup: fit 64 cells on the top 3 PCA modes to 2^18 true paths at layer 24, then propagate exactly to
    layer 32.
  - Net 101: nMSE 4.8e-6, against 6.8e-5 for one Gaussian. This passes, but only within fit-set variation: with an
    independent fit set (2^19 paths, another seed; section 7) net 101 gives 1.0e-5 against 7.2e-5, which would not
    pass.
  - Net 102: 8.1e-5, against 2.0e-4. This fails the bar, which needed at most 2e-5 and at most 1/10 of the
    Gaussian's.
  - Net 100 (exploration): 4.0e-5, against 2.6e-4.
  - Pushing with a per-cell Gaussian closure instead of exactly gives 7.7e-6, 1.1e-4 and 7.1e-5.
  - For scale, the medians over ARC's evaluation nets are 2.2e-4 for ARC's K3f and 6.7e-6 for the Kerdock sampler at
    2.72e11 FLOPs. Those are different nets and budgets, so this is no matched comparison.
- **Splitting a Gaussian does not create the information; cells built from the weights stay near K=2.**
  - On net 101, splitting the fitted Gaussian at layer 24 into 27 cells leaves the representation error at 7.0e-5
    (one Gaussian: 6.8e-5).
  - The split helps the push: total nMSE 9.7e-5 against 1.5e-4.
  - Re-binned latent cells built from the weights reach 1.65e-3 at 64 x 16 with 64 cells. The Gaussian closure gives
    3.5e-3. Plain Monte Carlo at about the same cost would sit near 8e-5. That cost, about 2e9 FLOPs, is our estimate;
    the code does not count it.
  - With 16 cells at 256 x 32 they reach 6.7e-4 (net 100) and 1.0e-3 (net 101). That is K=2's level (6.2e-4 and
    1.15e-3) and 2-7x worse than K3f (3.2e-4 and 1.5e-4).
- **The question's relevance axes did not beat the collapse modes for the mean.**
  - E[g g^T] for the all-outputs mean is less concentrated than the state covariance. Its top eigenvalue holds 9-12%
    of the trace, against 36-59% for the state and 0.4% for an isotropic matrix.
  - Its top axes are nearly orthogonal to the PCA modes: principal cosines at most 0.21.
  - 64 cells on its top 3 axes did worse than 64 PCA cells on all three nets: 1.3e-4 against 4.0e-5 on net 100,
    1.2e-5 against 4.8e-6 on 101, 8.2e-5 against 8.1e-5 on 102.
- **Tails: these fitted mixtures underestimate the depth-32 tails.**
  - The test: 64 cells, 16 PCA1 bins, or one Gaussian, each fitted to 2^18 true paths at layer 31 of net 102, scored
    on 8 outputs per level.
  - At 1e-6 the median underestimates are 7.5x, 3.7x and 240x.
  - This is one net and one fit, not a proof that every Gaussian-cell family must fail.
- **Scale.**
  - At 64 x 16, 81% of the Gaussian closure's error is a +2.2% overestimate of the mean's norm. A log-radial chart
    cuts the total error 2.5x.
  - At 256 x 32 the chart does nothing: 9.0e-4 against 7.6e-4.
  - The mechanism we proposed (a scale mixture misread at near-boundary neurons) is not isolated.
- **Not built.**
  - Candidate B (backward functions): the design critique classed it as an objective until a forward family passes,
    and none did.
  - Output moments and tails were not run deductively.

**What this means for M1** (development evidence only):
- A description of 256 x 32 has to carry deep-layer gate structure that our pairwise criteria could not compress.
  The census puts the joint entropy at layer 32 at 28-40 bits.
- It also has to carry within-cell structure, which fitted cells have and Gaussian splits do not.
- Even cells fitted to 2^18 true paths at layer 24 come near the sampler's median error on only one of three nets.
  Their per-cell push over the last 8 layers costs an estimated 4e10 FLOPs.
- We see no route to M1 through these prototypes. The one untested idea is in section 4.

## Handoff

**Code** (`work/boundary-family/`; copied from `work/arc-noisy`, which was not edited; ARC's kprop is copied in
`ext/`):
- `lib/bf.py`: nets, Monte Carlo, exact propagation from a cut.
- `lib/gmprop.py`: Gaussian-mixture propagation: latent, grid and boundary splits (forced gates), and Ward merging.
  The critique's two forced-gate bugs are fixed.
- `lib/lcg.py`: latent-conditional Gaussian, re-binned every layer.
- `lib/lrg.py`: log-radial Gaussian. The non-PSD chart is fixed, with an assert; the reruns give identical numbers.
- `lib/gauss_relu.py`: exact Gaussian ReLU moments; `relu_moments2` gives the same results faster.
- `scratch/`: oracles and tests (`diag1`-`diag9`, `lcg_test`, `lcg256`, `gm_test`, `grid_test`, `lrg_test`,
  `lrg256`).

**Records:**
- `DESIGN.md`: v1, then v2 after the critique, then sections 7-9 (factored gates, latent-conditional Gaussian,
  status).
- `PREREG.md` with amendment 1, hashed in `prereg_hashes*.txt`.
- `review/design_critique.md` and `review/claims_review.md`.

**Raw outputs:**
- K1' evidence: `logs/diag5_100_m24.log`, `logs/k1_101_m24.log` and `logs/k1_102_m24.log`. The SPLIT3 run
  overwrote `runs/diag5/diag5_101_m24.json`, so the K1' log is the record for net 101.
- K-F: `logs/kf_10[12].log`.
- Gate statistics: `runs/diag7/`.
- Tails: `runs/diag9/`.
- Width 64: `logs/w64_*.log`, rerun for provenance.

**Truths:**
- Width 256: the ARC lane's Monte Carlo means for dev nets 100-102 (2^25-2^26 paths) and its splitting tail truths
  for net 102.
- Width 64: `runs/explore1/` (2^22 paths, net 500).

**Nets:** width 256 dev nets 100 (exploration), 101 and 102 (pre-registered). Width 64: net 500. Width 128: net 600.
No evaluation net was touched, and no fresh seed of 64 or above.

## 1. Setup and the decisive measurement

**Nets and metric.**
- ARC's family as in the ARC lane: x ~ N(0, I), z_l = W_l h_{l-1}, a_l = ReLU(z_l), h_l = a_l + s_l eps, W_l iid
  N(0, 2/n).
- The noise scales s_l are part of the net definition.
- Metric: ARC's nMSE of the per-neuron means, minus the truth's 1/N.

**Representation versus push.** This is the critique's decisive split. Fit a family to Monte Carlo paths of the
pre-activation z_m at a cut. Then measure, at layer 32, against the independent truth:
- the *representation* error: sample the fitted family and run the real remaining layers;
- the *push* error: propagate every cell by its exact Gaussian closure (rectified means, bivariate-normal covariances,
  noise).

Errors are not additive, and a push can in principle cancel some representation error. So representation is a
reference, not a strict bound. Here every push error came out above its representation error.

**Families:**
- **G**: one Gaussian.
- **PCAk**: quantile cells of the top-k principal modes of z_m, each with its own fitted full covariance. k = 3 gives
  4 x 4 x 4 = 64 cells.
- **ASk**: the same cells on the top-k eigenvectors of E[g g^T], with g = d(r . a_32)/d z_m and r ~ N(0, I). These are
  Kelvin's relevance axes.
- **GATE6**: sign patterns of the six most relevant near-boundary gates (relevance = entropy x sqrt(E g_i^2)).
  Patterns with fewer than 600 paths are pooled.
- **SPLIT3**: the fitted single Gaussian, split 27 ways along its top 3 modes (a deductive control).
- **F3, F8, F3C**: factored models. An exact top-k latent, with Gaussian residual coordinates that are independent
  given it. F3C uses per-cell residual moments.
- **IND**: a product of the exact marginals.

## 2. Results

### 2.1 Representation and push at layer 24, width 256 (nMSE of E[a_32])

Each cell gives representation / push.

| family (cells) | net 100, exploration | net 101, K1' | net 102, K1' |
|---|---|---|---|
| G (1) | 2.6e-4 / 3.7e-4 | 6.8e-5 / 1.5e-4 | 2.0e-4 / 3.1e-4 |
| PCA1 (16) | 2.1e-4 / 3.3e-4 | - | - |
| PCA3 (64) | 4.0e-5 / 7.1e-5 | **4.8e-6 / 7.7e-6** | **8.1e-5 / 1.1e-4** |
| AS1 (16) | 1.9e-4 / 3.5e-4 | - | - |
| AS3 (64) | 1.3e-4 / 2.5e-4 | 1.2e-5 / 1.8e-5 | 8.2e-5 / 1.1e-4 |
| GATE6 (52 / 43 / 56) | 1.0e-4 / 1.4e-4 | 3.7e-5 / 5.9e-5 | 1.5e-4 / 1.9e-4 |
| SPLIT3 (27), deductive control | - | 7.0e-5 / 9.7e-5 | - |

- **Sizes:** fits on 2^18 paths, representation runs on 2^20. Both errors include fitting and sampling uncertainty.
- **ARC's methods from layer 1, for reference:** K3f 3.2e-4 (net 100) and 1.5e-4 (net 101); K=2 6.2e-4 and 1.15e-3.
- **K1' as registered:** on both nets, the best representation error had to be at most 1/10 of G's and at most 2e-5.
  Net 101 passed (PCA3). Net 102 failed (best 8.1e-5, 2.5x below G). The candidate fails by the written rule.
- **Against the prediction** (G 2-3e-4; best family 1.2-2e-4; AS3 no better than G; GATE6 within 30% of G): the
  families did better than predicted, and the kill came from net 102.
- **AS versus PCA:** the AS-versus-PCA comparison holds for k = 3 only. At k = 1 on net 100, AS1 (1.85e-4) beat PCA1
  (2.09e-4).

### 2.2 Gate structure (measurement D: pre-registered, descriptive only)

"Near-boundary" means entropy above 0.1 bit. "Latent3/8" is the fraction of those gates whose sign the top-k PCA
latent predicts in at least 90% of paths. No majority-sign baseline was subtracted.

| net (width x depth) | layers | near-boundary gates | summed marginal entropy (bits) | median / max abs phi | latent3 / latent8 |
|---|---|---|---|---|---|
| 500 (64 x 8) | 6 / 7 | 48 / 43 | 31 / 29 | 0.07 / 0.49; 0.08 / 0.45 | 33% / 44%; 26% / 37% |
| 600 (128 x 16) | 8 / 12 / 15 | 92 / 79 / 79 | 55 / 44 / 42 | 0.05-0.06 / at most 0.47 | 41-53% / 42-61% |
| 101 (256 x 32) | 16 / 24 / 31 | 101 / 57 / 65 | 57 / 33 / 37 | 0.05-0.08 / at most 0.67 | 42-49% / 49-68% |
| 102 (256 x 32) | 16 / 24 / 31 | 136 / 87 / 84 | 83 / 52 / 42 | 0.05-0.06 / at most 0.48 | 34-55% / 39-60% |
| 100 (256 x 32) | 24 | 120 | 68 | no pair above 0.8 | - |

- **Co-switching groups.** Single linkage on |phi| leaves every gate separate at every threshold from 0.5 to 0.95.
  The one exception is net 101 at layer 31, with 52 groups at 0.5.
- **Downstream-effect groups.** These use finite-difference effect vectors on a_32. At 0.8 the gate counts go
  136 to 111, 101 to 95, 92 to 89 and 59 to 57; elsewhere they are unchanged. At 0.5, single linkage chains the gates
  through a shared component (median |cos| 0.06-0.26).

### 2.3 Factored models (test K-F, pre-registered amendment 1)

One exact push from layer m. Each cell gives three numbers:
- the median over neurons of |Var ratio - 1| of z_{m+1};
- the median |skew difference| of z_{m+1};
- the nMSE of E[a_{m+1}], with no 1/N subtracted.

The last column compares two independent 2^17-path sets: one noisy reference, not a universal floor.

| net, m | G | F3 | F3C | F8 | C3 (unfactored) | set A vs set B |
|---|---|---|---|---|---|---|
| 101, 16 | 1.9% / 0.30 / 6.1e-5 | 9.2% / 0.17 / 3.3e-4 | 9.1% / 0.057 / 3.2e-4 | 4.9% / 0.10 / 9.2e-5 | 0.7% / 0.050 / 2.8e-5 | 0.35% / 0.007 / 8.5e-6 |
| 101, 24 | 1.5% / 0.40 / 3.8e-5 | 5.8% / 0.14 / 1.4e-4 | 5.8% / 0.034 / 1.2e-4 | 2.4% / 0.076 / 5.6e-5 | 0.5% / 0.049 / 8.8e-6 | 0.35% / 0.008 / 4.4e-6 |
| 101, 31 | 1.9% / 0.42 / 3.9e-5 | 3.8% / 0.092 / 6.1e-5 | 3.5% / 0.031 / 4.3e-5 | 1.6% / 0.054 / 3.3e-5 | 0.5% / 0.048 / 1.1e-5 | 0.32% / 0.009 / 3.0e-6 |
| 102, 16 | 2.1% / 0.27 / 6.8e-5 | 10% / 0.19 / 5.3e-4 | 10% / 0.069 / 5.1e-4 | 4.7% / 0.12 / 1.7e-4 | 0.8% / 0.060 / 2.7e-5 | 0.38% / 0.008 / 1.3e-5 |
| 102, 24 | 2.1% / 0.37 / 5.0e-5 | 10% / 0.18 / 4.1e-4 | 9.9% / 0.068 / 4.0e-4 | 3.8% / 0.093 / 1.0e-4 | 0.7% / 0.073 / 1.8e-5 | 0.33% / 0.008 / 1.1e-5 |
| 102, 31 | 1.9% / 0.41 / 7.5e-5 | 7.9% / 0.18 / 2.0e-4 | 7.1% / 0.070 / 1.8e-4 | 3.9% / 0.10 / 7.7e-5 | 0.8% / 0.068 / 2.1e-5 | 0.36% / 0.009 / 1.1e-5 |

- **IND** (a product of the exact marginals): 29-65% variance error, mean nMSE 2.5e-3 to 7.3e-3.
- **The rule:** at m = 24 and 31 on both nets, some factored model must reach at most 1/3 of G's mean error with at
  most 2% variance error. None does, as predicted.
- **What fails is the second-order structure along W.** A diagonal residual misses W (off-diagonal) W^T. The factored
  models capture the skewness better than G does.

### 2.4 Inside the cells, and the deductive latent-conditional Gaussian

**Within-bin distributions are non-Gaussian, and more so with depth** (main's kill test 2 for the latent-conditional
Gaussian). These numbers are for z within the 64 PCA3 bins, which is not the residual given continuous latent
coordinates: bin truncation adds some non-Gaussianity. Paired fit, width 256, net 100. Median per-neuron |skew| /
excess kurtosis:

| layer | median abs skew | median excess kurtosis |
|---|---|---|
| 8 | 0.07 | 0.045 |
| 16 | 0.13 | 0.11 |
| 24 | 0.17 | 0.22 |
| 31 | 0.22 | 0.27 |

At width 64 these reach 0.40 and 0.56.

**The deductive latent-conditional Gaussian** (`lib/lcg.py`): cells re-binned on the top-k axes every layer, each
pushed by its exact closure. The code counts no FLOPs, so costs below are estimates.
- **64 x 16:** 2.8e-3 / 2.0e-3 / 1.65e-3 at 4 / 16 / 64 cells. For comparison: Gaussian closure 3.5e-3, K=2 3.5e-3,
  K3f 7.5e-3, log-radial chart 1.4e-3. Every kprop variant at width 64 is worse than K=2. Log: `logs/w64_lcg.log`.
- **256 x 32, 16 cells** (an estimated 4e10 FLOPs): 6.7e-4 on net 100 (K=2 6.2e-4, K3f 3.2e-4, Gaussian closure
  7.6e-4); 1.0e-3 on net 101 (K=2 1.15e-3, K3f 1.5e-4).
- **By layer on net 101** (layers 8 / 16 / 24): this method gives 1.9e-4 / 4.5e-4 / 7.5e-4, against K3f 3.9e-6 /
  2.4e-5 / 3.6e-5. K3f is far better through layer 24.

### 2.5 Other deductive attempts (width 64, depth 16, net 500; logs in `logs/w64_gm.log` and `logs/w64_grid.log`)

| method | nMSE |
|---|---|
| latent splits, moment-preserving, 9 / 27 / 81 cells | 3.0e-3 / 2.9e-3 / 2.5e-3 |
| exact boundary splits with forced gates, after the bug fix, 4 / 16 / 64 cells | 3.9e-3 / 3.1e-3 / 3.0e-3 |
| one grid split at layer 4, no re-binning, 3 / 9 / 27 cells | 3.4e-3 / 2.8e-3 / 2.5e-3 |

One early latent-split run reached 9e-4. Its split rule shrank the variance by 4.5% per split, and the gain came from
cancelling the closure's scale overestimate. That run is withdrawn, and its output was not archived.

### 2.6 Tails of fitted Gaussian mixtures (net 102, depth 32)

Mixtures were fitted to 2^18 true paths of h_31. P(z_32,i > t) was read exactly from the mixture, for 8 outputs x 5
thresholds, against the ARC lane's splitting truths. Those truths have relative SEs up to about 42% at the deepest
levels.

Median log10(estimate / truth):

| mixture | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| one Gaussian | -0.78 | -2.4 | -4.6 | -7.2 | -10.0 |
| 16 PCA1 bins | -0.26 | -0.57 | -0.99 | -1.8 | -2.8 |
| 64 PCA3 cells | -0.38 | -0.88 | -1.6 | -2.6 | -4.0 |

For comparison, from ARC part 2 (evaluation nets 0-19, all events), Monte Carlo moments put events within 50% as
follows:

| readout of the Monte Carlo moments | at 1e-4 | at 1e-6 |
|---|---|---|
| plain order-4 saddlepoint (U3) | 56% | 18% |
| chi-mixture saddlepoint (U3r) | 55% | 29% |

### 2.7 Scale (log-radial chart)

| setting | Gaussian closure | log-radial chart |
|---|---|---|
| 64 x 16, net 500 | 3.5e-3; 81% along the mean, scale +2.2% | 1.4e-3; scale +0.26% |
| 256 x 32, net 100 | 7.6e-4; 50% along the mean, +0.47% | 9.0e-4; -0.36% |

- The chart's PSD bug was fixed after the claims review, and the numbers did not change.
- The proposed mechanism is not isolated. Width, depth and seed all differ between the two settings.

## 3. Review

**Design critique** (GPT-6, `review/design_critique.md`): 16 findings, 3 critical. All were taken:
- the two forced-gate bugs were fixed;
- the log-radial chart was fixed after the claims review flagged it as still open;
- the paired oracles were relabelled;
- the split between representation and push was added.

**Claims review** (GPT-6, `review/claims_review.md`): 16 findings (13 major, 3 minor, no critical). It confirmed both
kill-test verdicts. All findings were applied:
- **Representation as a reference:** representation is a reference, not a bound; split-versus-push effects are stated
  as total-nMSE ratios.
- **Grouping counts:** downstream-effect grouping counts were corrected.
- **Pairwise versus joint:** pairwise statistics are no longer read as joint independence. The majority-sign caveat is
  added.
- **Relevance axes:** "nearly isotropic" is withdrawn, and the AS-versus-PCA claim is restricted to k = 3.
- **Log-radial chart:** fixed and rerun.
- **Verdict scope:** narrowed to the prototypes tested.
- **Fitted cells:** called fitted, not perfect.
- **Costs:** FLOPs are labelled as estimates.
- **Tails:** the tail conclusion is narrowed, and the ARC lane's U3 and U3r numbers are attributed correctly.
- **Within-bin:** "conditionally Gaussian residual" is renamed "within-bin".
- **Width-64 provenance:** results were rerun into logs.
- **Cumulant closure:** it is described as a proposal.
- **Numbers and labels:** minor numbers fixed (0.84-8.2x; 3.7 / 7.5 / 240x; kurtosis 0.045; truth 2^25-2^26), and
  diag8 labels corrected.

## 4. What could still work (untested)

**A polynomial closure from a cumulant description at a switch layer.** This is a proposal, and it needs its own test.
- Approximate the conditional means of z given the PCA latent as quadratic, and the conditional covariances as linear
  in it, with coefficients from the third and fourth joint cumulants.
- Build the 64 cells from that, then propagate the cells.
- Cheapest check first: build the cells at layer 24 from the true cumulants. Compare their representation error with
  the fitted cells' (4.8e-6 on net 101, 8.1e-5 on net 102).
- Note that finite cumulants do not determine the cells, and that on net 102 even fitted cells fall short.

**For tails:** heavier-tailed components per cell, or a backward refinement for one event at a time.

## 5. Compute

- CPU only. The host ran at load 45-65 on 16 cores; at most 3 processes, at nice 19 / ionice idle from 15:40Z.
- The ARC lane's truths were reused. No GPU.

## 6. Handoff for a fresh agent: fully non-Gaussian families (Kelvin's item 2, 29 Oct ~16:55Z)

Kelvin: "we should just stop assuming any gaussian part whatsoever then. Let us seek a family that is fully
non-gaussian, even if its bad at the input, and hope that it at least scales on depth." What this lane can hand over:

**The criterion and a correction to it (pre-register before any run).**
- Kelvin's version: with survival S(l, L) = (l/L)^3, the total error is about sum_l e_l (l/L)^3, so a family scales on
  depth only if its per-layer re-projection error e_l decays faster than 1/l in the deep regime (e_l ~ l^-a gives a
  total ~ L^(1-a)).
- Two caveats from our data:
  1. If the errors add coherently (same sign every layer, like the Gaussian closure's scale bias at width 64), then
     amplitudes add, the total goes like L^(2-a), and a must exceed 2. Measure coherence: project at two cuts and
     compare the total with the sum of the single-cut errors.
  2. The (l/L)^3 law was measured for mean shifts (ARC lane: energy kept at 32 from l = 4 / 8 / 16 / 24 / 28 was
     0.009 / 0.10 / 0.11 / 0.25 / 0.74; (l/L)^3 gives 0.002 / 0.016 / 0.125 / 0.42 / 0.67). Distributional errors do not
     follow it. Gaussianizing the true state at layer 24 has a one-step mean error of 5.8e-5 but 2.6e-4 at layer 32
     (net 100): it grows 4.5x instead of shrinking to 0.42. Gaussianizing at layer 8 leaves 2.0e-4 at 32 (ARC lane),
     about the same as at 24, not 27x less.
- So measure the composite directly. R_F(l; L) is the output error at L when the true state is projected onto family
  F at a single cut l and propagated exactly afterwards. `scratch/diag5.py` (rep) and `scratch/diag10.py` (all layers
  24..32) already do this for cell families. The test is whether sum_l R_F(l; L), with L growing with n (64 x 8,
  128 x 16, 256 x 32), falls below the sampler at equal cost and does not grow with L.
- For a deductive family, add the family's own push error at each layer, measured separately (the push in
  `diag5.py`).

**Data the next agent starts from** (width 256, dev nets; all one-step numbers are paired, so they are closure
residuals):
- One Gaussian with true moments: one-step error 2.6e-5 / 2.9e-5 / 5.8e-5 / 2.3e-5 at layers 8 / 16 / 24 / 31 (net 100).
  It does not decay with depth.
- 64 PCA cells: 6.7e-6 / 4.8e-6 / 6.4e-6 / 6.6e-6, near a paired floor. Single-cut output error R(24; 32) is 4.0e-5,
  4.8e-6 and 8.1e-5 on nets 100, 101 and 102. Item 1 (section 7) extends this to nets 103-110.
- Within-bin non-Gaussianity grows with depth: median |skew| / excess kurtosis 0.07 / 0.045 at layer 8 and
  0.22 / 0.27 at layer 31. Nothing we measured suggests a shrinking, self-similar shape yet. At width 64, though, the
  direction part (scale divided out) had roughly stable cumulants from layer 4 to 15 (|skew| 0.22-0.34, excess
  kurtosis 0.18-0.35): worth checking at 256 for candidate (a).
- Gate structure: 40-140 near-boundary gates per deep layer, pairwise nearly uncorrelated. The census puts the joint
  entropy at 28-40 bits at layer 32.

**On the candidates:**
- **(a) Attractor family** (common direction, log-size law, a shrinking angular shape). Cheapest first test: do the
  rescaled angular deviations' per-neuron cumulants (or the law of d_l x l) converge with depth at fixed width? Do it
  at 256 x 32 and 128 x 64. Our log-radial chart (`lib/lrg.py`) is the size half of it; at 256 x 32 the size is not
  where the error is (section 2.7).
- **(b) Exact particles in k = 2-4 remembered directions, with the remainder's second moments carried without a
  shape.** This is consistent if every near-boundary gate's normal lies in the remembered space, because then the
  remainder only meets decided, linear neurons, and second order is enough (K-F shows it must be kept in full, not
  diagonal). The risk: 40-140 near-boundary gates per layer do not fit in k = 2-4. The top-8 latent predicts only
  37-68% of them at 90%, before a majority baseline. The gates left outside need a shape.
- **(c) Weighted points pushed exactly** have zero re-projection error. That is structured sampling, not mechanistic.
- **A candidate we add (d): per-cell shape corrections.** Cells for the strong low-dimensional part, with the
  measured within-bin skewness and kurtosis of the near-boundary coordinates carried per cell (cumulants or quantile
  maps). A first oracle exists: PCA3NB in `diag10.py` (exact within-cell marginals of the near-boundary coordinates,
  Gaussian copula). If it does not beat PCA3, the shape at near-boundary neurons is not what is missing.

**Tools:**
- `lib/bf.py`: Monte Carlo, cuts, exact propagation.
- `scratch/diag5.py`, `diag8.py`, `diag10.py`: representation, push and factored tests.
- The ARC lane's truths: dev nets 100-111 at width 256.
- Width-64 and width-128 nets are cheap (seeds 500+, 600+).
- The host runs everything at nice 19 / ionice idle, at most 3 processes.

## 7. Kelvin's item 1: why did the fitted cells work on net 101 and not on net 102? (17:35Z)

Kelvin: "Cells fitted to the truth at layer 24: didn't it work? Even if non-deductive? I wonder why it worked in one
case and not the other."

**Method.** The same oracle was rerun on ten development nets, 101-110 (`scratch/diag10.py`, summary
`scratch/d10_analyze.py`, raw `runs/diag10/`).
- A fresh fit set of 2^19 paths per net, with another seed than K1'.
- Errors of E[a_l] at every layer l = 24..32, against the ARC truths.
- Families: G; PCA3 (64 cells); PCA4 (256 cells); PCA3NB (PCA3 with the exact within-cell marginals of the
  near-boundary coordinates, Gaussian copula inside each cell); EMP (the resampled fit set, i.e. the fit-sampling floor).
- This is exploration, not pre-registered.

**Answer.**
- **Nets 101 and 102 are the two ends, not the typical case.** The gain of 64 cells over one Gaussian at layer 32 has
  median 4.8x across the ten nets (range 2.4-7.2x). By net: 101: 7.2; 103: 7.0; 109: 6.2; 104: 5.6; 105: 4.9; 107: 4.8;
  106: 3.9; 102: 2.6; 108: 2.6; 110: 2.4.
- **K1' on net 101 was partly luck of the fit set.** Net 101 gives 1.0e-5 against 7.2e-5 here, not 4.8e-6 against
  6.8e-5 as in K1' (2^18 paths, another seed). With this fit set it would not pass.
- **The gain tracks how collapsed the state is at layer 24.** Spearman correlations over the ten nets (13 properties
  tried, p-values not corrected):

  | property | rho | p |
  |---|---|---|
  | share of the variance in the three cell axes | +0.75 | 0.013 |
  | participation ratio | -0.70 | 0.025 |
  | size-mode share (the top mode is the mean direction on every net, cosine 0.99) | +0.70 | 0.025 |
  | gate entropy (near-boundary and all) | -0.60 | 0.07 |
  | near-boundary gate count | -0.50 | 0.14 |
  | within-bin kurtosis (all or near-boundary) | about 0 | |

  Net 101 is among the most collapsed: participation ratio 2.7, 76% of the variance in the top 3 modes, 57
  near-boundary gates (33 bits). Net 102 is among the least collapsed: 4.3, 60%, 87 gates (52 bits). Net 110 is the
  least: 5.4, 97 gates (57 bits), gain 2.4x.
- **The gain is set at the cut, not lost on the way.**
  - The ratio of the Gaussian's error to the cells' error is roughly constant from layer 25 to layer 32 on every net:
    101 6.4 to 7.2; 102 3.0 to 2.6; 108 2.9 to 2.6.
  - Both errors grow about the same from 25 to 32 (median 2.3x for the Gaussian, 2.5x for the cells). The network
    amplifies whatever distributional error is present at the cut.
- **The loss is not in the near-boundary neurons' marginal shapes.**
  - Giving the cells the exact within-cell marginals at the near-boundary coordinates cuts the layer-24 readout error
    by a median 3.5x.
  - At layer 32 it changes the error by a median of only 1.12x (0.97-1.33).
  - At layer 25, neurons near their boundary (|t| < 1.5, 14-26% of neurons) carry 30-60% of the squared error. By
    layer 32 the error is spread about in proportion.
- **The loss is partly in the cells' dimension.** A fourth cell axis (256 cells) improves layer 32 by a median 1.41x
  (1.05-2.54x).

**Reading.** What the cells miss is joint structure outside their axes. The less collapsed the net, the more of the
state lies outside the top 3 modes, and the smaller the gain. It is not a per-neuron shape at the near-boundary
neurons.

This fits the earlier findings:
- factored (diagonal) residuals fail;
- the gates are nearly uncorrelated pairwise.

It suggests that a family which works on the typical net needs more axes, or an explicit model of the dependence
beyond them. Even the fitted 256-cell family stays at 7e-6 to 4e-5 at layer 32, 1.1-28x above its fit-sampling floor
(median 6.8x).
