## 3. Deductive estimation where the whole state is small

**Setup.**
- Estimand: E[output] (a vector), scored as nMSE = |estimate - truth|^2 / tr Var(output), as ARC does. Truth: plain
  MC with 2^21-2^22 paths, so its own nMSE is 2-5e-7.
- Costs are FLOPs under the flopscope conventions of `work/arc-noisy` (matmul M N (2K-1), normal draw 16, cdf 96,
  ...), expressed in units of one full MC path, so "35 paths" means the FLOPs of 35 sampled paths. With N paths,
  plain MC's nMSE is exactly 1/N in expectation.
- Simple samplers: MC; antithetic pairs (ANT); scrambled-Sobol RQMC on the input; and the regression control variate
  on the input x (E x = 0 is known), MC-reg and RQMC-reg. The best simple sampler is the lowest of these at each
  budget. Replicates: R = 16.
- Deductive (sample-free):
  - K2: ARC's Gaussian closure, mean and covariance pushed through each ReLU in closed form (exact bivariate ReLU
    moments, copied from `work/arc-noisy`);
  - K2-Stein: the same with the covariance linearised (Stein's lemma; exact means and variances), which costs n^3
    instead of n^2 x 2,500 per layer.
- Hybrid HYB(l0): MC for the first l0 layers, then a deductive deep part: 8 log-radius bins of the sampled states,
  each pushed by K2-Stein ("st") or by the noiseless map of the bin mean ("lin").

### 3.1 Residual nets (all n directions alive): a sample-free closure beats the best sampler at small budgets

| residual, sigma = 1e-3 | nets | K2 nMSE (range) | K2-Stein nMSE | K2-Stein cost (MC paths) | at that budget: MC / best simple sampler | K2 (exact) cost |
|---|---|---|---|---|---|---|
| n = 16, L = 16 | 4 | [RD] | [RD] | 35 | 0.029 / [RD] | 512 |
| n = 16, L = 64 | 4 | 2.5e-5 - 8.3e-5 | 2.6e-5 - 7.7e-5 | 35 | 0.029 / [RD] | 517 |
| n = 16, L = 256 | 4 | 5.3e-5 - 1.4e-4 | 5.4e-5 - 1.3e-4 | 35 | 0.029 / [RD] | 518 |
| n = 16, L = 1024 | 2 | [RD] | [RD] | 35 | 0.029 / [RD] | [RD] |
| n = 8, L = 64 | 4 | 1.1e-4 - 3.0e-4 | 1.1e-4 - 3.2e-4 | 18 | 0.056 / [RD] | 411 |
| n = 8, L = 256 / 1024 | 2 / 2 | [RD] | [RD] | 18 | [RD] | [RD] |

[RD] = from runs/resdepth_*.json (to fill).

- **At its own budget (about 35 MC paths at n = 16), K2-Stein beats every simple sampler by about 30x [check] in MSE**
  (MEASURED). RQMC is the strongest sampler here: its error falls like N^-1.5 to N^-2. It catches up with K2's
  accuracy only at about 300-1,000 paths (n = 16) and 100-300 paths (n = 8). Stated as cost, K2-Stein reaches in 35
  paths' FLOPs what RQMC needs 10-30x more for.
- **It has a floor.** K2's error is a bias, nMSE 2.5e-5 to 3e-4. Above about 1,000 paths RQMC wins: at 16,384 paths
  RQMC reaches 1e-7 to 1e-6. The exact-moment K2 costs 15x more than K2-Stein for the same accuracy, so it never
  beats RQMC.
- **Depth does not hurt it (MEASURED, P10).** The ratio of K2's nMSE at L = 1024 to L = 64 is [RD], and every depth
  from 16 to 1024 stays within [RD].
  - Its cost grows linearly in L like a sample's, so the advantage at equal FLOPs does not shrink with depth.
  - Why (HEURISTIC): each layer adds V_l b_l / sqrt(L), with V_l independent of everything before it. A closure error
    made in layer l therefore enters along a fresh random direction. The output error is a random walk with L steps
    of size 1/sqrt(L), i.e. of the size of a typical one-layer error, whatever L is.
- **What makes this regime easy.** It is not that the input is forgotten:
  - the input direction explains 90-97% of the output variance, its norm 0.4-4%;
  - the cell count is maximal: every one of 2^20 inputs has its own activation pattern by layer 8 (n = 16);
  - but the map is close to quadratic: a linear function of x explains 82-87% of the output variance and a quadratic
    one 96-97%, at every depth (the plain nets: 8-60% and 10-90%);
  - the mean, which the linear part cannot produce, comes from the ReLU means, which a Gaussian closure gets right
    when the projections of the state are close to Gaussian.
- **Hybrids do not help here.** With no synchronisation, the radius-binned deep part has a bias of 4e-5 to 1e-4
  (HYB-st) or 0.05-0.4 (HYB-lin), so HYB loses to RQMC by 10-100x at every budget.

### 3.2 Plain nets: the deep part is free, the shallow part is the blocker

**K2 from the input fails.** nMSE is 0.016-1.0 on the four n = 16, L = 256 nets and 2-27 on the healthy n = 8 nets.
On a net whose output is mostly noise it reaches 1.2e7. The collapsed state is a ray whose radius has a log-normal
spread, and a Gaussian with the same mean and covariance puts mass at negative radius.

**The deep part from the exact law at layer l0 (an oracle for the shallow part).** Radius-binned K2 from layer l0 to
L = 256, nMSE (truth resolution about 5e-7). Healthy nets are those where the input explains >= 99% of the output
variance:

| net (input share of output variance) | l0 = 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| n = 16 seed 0 (0.99) | 0.18 | 0.054 | 6.7e-7 | 3.3e-5 | 1.8e-7 | 2.0e-7 |
| n = 16 seed 1 (1.00) | 5e-4 | 4e-4 | 3.2e-4 | 3.6e-4 | 3.7e-7 | 3.7e-7 |
| n = 16 seed 3 (0.99) | 1e-3 | 1.7e-3 | 3.1e-3 | 2.8e-3 | 1.9e-5 | 2.0e-5 |
| n = 8 seed 7 (0.99) | 0.053 | 1.7e-3 | 1.4e-5 | 1.7e-5 | 1.7e-5 | 2.1e-5 |
| n = 16 seed 2 (0.77, near-dead) | 0.24 | 0.024 | 1.8e-3 | 7.6e-3 | 1.9e-3 | 1.6e-3 |

- On healthy nets the layers beyond l0 = 16-64 cost one weight-derived trajectory per radius bin, and the error
  stays at 1e-7 to 2e-5 (MEASURED).
- The l0 needed for <= 1e-4 is 16 (n = 8) and 32-64 (n = 16), consistent with the synchronisation length ~n.
- On near-dead nets, where noise carries 23-89% of the output variance, neither the binned K2 nor the
  noiseless mean push works.

**Hybrid at equal FLOPs (column (b) of ARC's bar).** n = 16, L = 256, seed 0, nMSE:

| budget (MC paths) | MC | best simple sampler | best hybrid |
|---|---|---|---|
| 256 | 3.6e-3 | 1.3e-3 (RQMC-reg) | 3.1e-4 (HYB16-lin) |
| 1,024 | 6.2e-4 | 2.2e-4 (MC-reg) | 1.1e-4 (HYB32-st) |
| 4,096 | 3.5e-4 | 9.3e-5 (RQMC) | 5.8e-5 (HYB32-st) |
| 16,384 | 3.3e-5 | 1.6e-5 (RQMC-reg) | 2.1e-5 (HYB32-st) |

The hybrid gains 2-4x over the best sampler at small budgets and nothing at 16k paths, where its deep-part bias
(about 1e-5) shows (MEASURED, one net; the planned second seed was dropped for time, Amendment 6).

**Why no sample-free method reaches the shallow part (MEASURED).**
- The shallow part is a narrow net of depth l0 ~ 16-64 on Gaussian input.
- An exact cell representation needs the joint activation patterns of layers 1..l. Among 2^20 inputs:
  - n = 8 (sigma = 0): 3,600-5,300 distinct patterns at layer 2, 76,000-82,000 at layer 8, and about 100,000 from
    layer 16 on. Synchronisation stops further splitting; without noise, 90% of the patterns hold a single input.
  - n = 16: about 1,000,000 by layer 8, i.e. nearly every input in its own cell.
  - Each cell would also need a Gaussian integral over a polyhedral cone in 8-16 dimensions.
- A grid at noise resolution needs (1/sigma)^(n-1) cells.
- So in the plain family the M1 difficulty does not disappear in narrow-deep nets. It moves into the first
  ~n ln(1/tolerance) layers, a net whose depth is proportional to its width: the regime M1 is stated for.

### 3.3 Tails: P(|output| > t) at the 1e-3 and 1e-4 quantiles

- **The Gaussian closure is wrong in the tails (MEASURED, residual n = 16).** K2's Gaussian, with its tail computed
  exactly by Imhof's formula (checked against sampling that Gaussian), gives 5.8e-5 and 8.2e-5 for true 9.4e-4 and
  9.8e-4, and 1.3e-6 and 2.3e-6 for true 9.9e-5. That is 12-76x too small. The quadratic part of the map makes the
  tails heavier than the closure's Gaussian.
- **Positive homogeneity gives a better sampler (MEASURED).** A bias-free ReLU net satisfies out(r u) = r out(u)
  without noise. So the radius can be integrated exactly per sampled direction ("RB"): P = E_u Q_chi_n(t / |out(u)|).
  It lowers the relative MSE 3-40x at equal FLOPs:
  - residual n = 16 at q = 1e-3, 4,096 paths: 0.011 and 0.005 vs MC 0.18 and 0.29;
  - plain n = 16 at q = 1e-4, 16,384 paths: 0.13 vs MC 0.39.
  - With sigma = 1e-3 the identity is approximate; we scale the noise to a typical radius.
- The plain hybrid's attractor-linear deep map leaves a tail bias of ~35% relative (0.14 relative MSE at 16k paths).
- Nothing here is a certified bound (M3).
